"""
U-Net 裂缝语义分割训练脚本。
用法: python train.py
"""
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

from model import CrackUNet

# ======================== 配置 ========================
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/youluoyuan-crack/youluoyuan")
# /home/fs-ai/YOLO-crack-detection/dataset/ultralytics  # YOLO官方裂缝数据集，只在box检测上性能良好
SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"

# 动态加载均值和标准差
STATS_FILE = Path(__file__).resolve().parent / "dataset_stats.json"
if STATS_FILE.exists():
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        IMAGENET_MEAN = stats["mean"]
        IMAGENET_STD = stats["std"]
        dataset_name = stats.get("dataset_name", "Unknown")
    print(f"✅ 加载自定义数据集 ({dataset_name}) 统计信息: Mean={IMAGENET_MEAN}, Std={IMAGENET_STD}")
else:
    print("⚠️ 未找到 dataset_stats.json，回退到 ImageNet 默认值")
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "encoder_lr": 1e-4,      # encoder 微调，0.1x
    "decoder_lr": 1e-3,      # decoder 正常学习率
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 50,           # early stopping
    "num_workers": 4,
    "seed": 42,
}


# ======================== Dataset ========================
class CrackDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, augment: bool = False):
        self.images = sorted(images_dir.glob("*.jpg"))
        self.masks_dir = masks_dir
        self.augment = augment
        # 验证 image-mask 配对
        for img_path in self.images[:5]:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            assert mask_path.exists(), f"找不到 mask: {mask_path}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        # 转 tensor 并归一化
        image = TF.to_tensor(image)  # (3, H, W), float32 [0,1]
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0  # (1, H, W) [0,1]

        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image):
        # 水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # 垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # 随机旋转 (-30, 30)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        # 颜色增强（只作用于 image）
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.3, 1.7))
        if random.random() > 0.5:
            image = TF.adjust_hue(image, random.uniform(-0.015, 0.015))

        return image, mask


# ======================== Loss ========================
def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice 联合损失，处理裂缝细线的类不平衡。"""
    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)  # smooth=1

    return bce + dice.mean()


# ======================== Metrics ========================
@torch.no_grad()
def compute_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算 batch 平均 IoU。"""
    pred = (torch.sigmoid(logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ======================== Training ========================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)

        total_loss += loss.item() * images.size(0)
        total_iou += compute_iou(logits, masks)
        n_batches += 1

    avg_loss = total_loss / len(loader.dataset)
    avg_iou = total_iou / n_batches
    return avg_loss, avg_iou


def main():
    cfg = CONFIG
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 数据集
    train_ds = CrackDataset(
        DATASET_ROOT / "train" / "images",
        DATASET_ROOT / "train" / "masks",
        augment=True,
    )
    val_ds = CrackDataset(
        DATASET_ROOT / "val" / "images",
        DATASET_ROOT / "val" / "masks",
        augment=False,
    )
    train_loader = DataLoader[Any](
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # 模型
    model = CrackUNet(pretrained=True).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 差分学习率: encoder 慢 10x
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": cfg["encoder_lr"]},
        {"params": model.decoder_params(), "lr": cfg["decoder_lr"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"], eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")

    # 输出目录
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SAVE_DIR.parent / "training_log.csv"

    best_iou = 0.0
    patience_counter = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "lr"])

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val IoU':>8} | {'LR':>10}")
    print("-" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup: 前几个 epoch 线性增长学习率
        if epoch <= cfg["warmup_epochs"]:
            warmup_factor = epoch / cfg["warmup_epochs"]
            for pg in optimizer.param_groups:
                if pg["lr"] == cfg["encoder_lr"]:
                    pg["lr"] = cfg["encoder_lr"] * warmup_factor
                else:
                    pg["lr"] = cfg["decoder_lr"] * warmup_factor

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_iou = validate(model, val_loader, device)

        if epoch > cfg["warmup_epochs"]:
            scheduler.step()

        current_lr = optimizer.param_groups[1]["lr"]  # decoder lr
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_iou:>8.4f} | {current_lr:>10.6f}  ({elapsed:.0f}s)")

        # 日志
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_iou:.6f}", f"{current_lr:.8f}"])

        # Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_iou,
        }, SAVE_DIR / "last.pt")

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
            }, SAVE_DIR / "best.pt")
            print(f"        -> best model saved (IoU={best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Best model: {SAVE_DIR / 'best.pt'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
