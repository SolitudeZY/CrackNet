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

# ======================== 评估指标计算 ========================
def calculate_metrics(preds, targets, threshold=0.5):
    """
    计算语义分割的标准指标: Precision, Recall, IoU (Jaccard), 和 Dice (F1-Score)
    preds: (N, 1, H, W) sigmoid 概率值 [0, 1]
    targets: (N, 1, H, W) 真实标签 [0, 1]
    """
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "iou": 0.0,
        "dice": 0.0
    }
    
    # 二值化
    pred_mask = (preds > threshold).float()
    targets_flat = targets.view(-1)
    preds_flat = pred_mask.view(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    
    # 防止除零错误
    metrics["precision"] = tp / (tp + fp + 1e-6)
    metrics["recall"] = tp / (tp + fn + 1e-6)
    metrics["iou"] = tp / (tp + fp + fn + 1e-6)
    metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + 1e-6) # 等价于 F1-Score
    
    return metrics

from U_Net_MobileNetV2_model import U_Net_MobileNetV2

# ======================== 自动检测图像尺寸 ========================
def detect_dataset_img_size(images_dir: Path) -> int:
    """随机读取几张图片，自动推断应该使用的 IMG_SIZE"""
    print(f"正在自动检测数据集图片尺寸...")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        print("未找到图片，默认使用 640")
        return 640
    
    # 抽查前 5 张图片
    sizes = []
    for img_path in image_files[:5]:
        with Image.open(img_path) as img:
            sizes.append(max(img.size)) # 取宽高的最大值
            
    # 取最常见的尺寸
    detected_size = max(set(sizes), key=sizes.count)
    
    # 必须是 32 的整数倍（U-Net 有 5 次下采样：2^5 = 32）
    aligned_size = ((detected_size + 31) // 32) * 32
    print(f"检测到最大边长约 {detected_size}，对齐到32的倍数后，IMG_SIZE 自动设置为: {aligned_size}")
    return aligned_size

# ======================== 配置 ========================
DATASET_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"

# 自动检测图片尺寸
IMG_SIZE = detect_dataset_img_size(DATASET_ROOT / "traincrop" / "traincrop")

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
    "epochs": 300,
    "batch_size": 8,         # 减小 batch_size 避免 Attention 机制导致的 OOM (原为16)
    "encoder_lr": 1e-4,      # encoder 微调，0.1x
    "decoder_lr": 1e-3,      # decoder 正常学习率
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 50,           # early stopping
    "num_workers": 4,
    "seed": 42,
    "grad_accumulation": 2,  # 梯度累加步数，相当于把等效 batch_size 恢复到 16
}


# ======================== Dataset ========================
class CrackDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path = None, augment: bool = False):
        self.images = sorted(list(images_dir.glob("*.jpg")))
        self.labels_dir = labels_dir
        self.augment = augment
        
        # 验证 image-label 配对 (CRACK500 格式: 同名 .png)
        for img_path in self.images[:5]:
            mask_path = img_path.with_suffix(".png")
            if not mask_path.exists() and self.labels_dir:
                label_path = self.labels_dir / (img_path.stem + ".txt")
                assert label_path.exists(), f"找不到 label: {label_path} 或 {mask_path}"
            else:
                assert mask_path.exists(), f"找不到 mask: {mask_path}"

    def __len__(self):
        return len(self.images)

    def _parse_yolo_label(self, line: str, img_w: int, img_h: int):
        """解析一行 YOLO 标注，支持多边形 (Segmentation) 和 边界框 (Detection)"""
        parts = line.strip().split()
        if len(parts) == 5:
            # Bounding Box 格式: class x_center y_center width height
            _, x_c, y_c, w, h = map(float, parts)
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h
            return "bbox", [x1, y1, x2, y2]
        else:
            # Polygon 格式: class x1 y1 x2 y2 ...
            coords = [float(v) for v in parts[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                points.append((x, y))
            return "polygon", points

    def _generate_mask_from_label(self, label_path: Path, img_w: int, img_h: int) -> Image.Image:
        """从 txt 标签实时生成二值掩码 (PIL Image)"""
        from PIL import ImageDraw
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    label_type, coords = self._parse_yolo_label(line, img_w, img_h)
                    if label_type == "polygon" and len(coords) >= 3:
                        draw.polygon(coords, fill=255)
                    elif label_type == "bbox":
                        draw.rectangle(coords, fill=255)
        return mask

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        
        # 尝试读取 CRACK500 的 png 掩码，如果没有则从 YOLO txt 生成
        mask_path = img_path.with_suffix(".png")
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            mask = self._generate_mask_from_label(label_path, img_w, img_h)

        if self.augment:
            image, mask = self._augment(image, mask)

        # 统一缩放至模型输入尺寸
        image = TF.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.NEAREST)

        # 转 tensor 并归一化
        image = TF.to_tensor(image)  # (3, H, W), float32 [0,1]
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        
        # 将 PIL Mask (L模式，0或255) 转换为 tensor
        # 确保二值化并归一化到 [0, 1]
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)
        # 如果像素值是 255，除以 255 变成 1.0；如果已经是 0/1 就不变
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        # 确保严格的 0/1 标签，避免插值带来的灰度值
        mask_tensor = (mask_tensor > 0.5).float()

        return image, mask_tensor

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
def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal + Dice Loss，用于解决类别不平衡同时保持边缘精细。
    避免使用过大的 pos_weight 导致裂缝被预测得过粗。
    """
    # 1. Focal Loss
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = alpha_t * (1 - p_t) ** gamma * bce
    focal_loss = focal_loss.mean()

    # 2. Dice Loss
    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)

    return focal_loss + dice.mean()


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
def train_one_epoch(model, loader, optimizer, scaler, device, accumulation_steps=2):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = focal_dice_loss(logits, masks)
            loss = loss / accumulation_steps # 将 loss 除以累加步数

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = focal_dice_loss(logits, masks)

        total_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(logits)
        batch_metrics = calculate_metrics(probs, masks)
        for k in all_metrics:
            all_metrics[k].append(batch_metrics[k])

    avg_loss = total_loss / len(loader.dataset)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


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
        DATASET_ROOT / "traincrop" / "traincrop",
        augment=True,
    )
    val_ds = CrackDataset(
        DATASET_ROOT / "valcrop" / "valcrop",
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # 模型
    model = U_Net_MobileNetV2(pretrained=True).to(device)
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
    best_dice = 0.0
    patience_counter = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "iou", "dice", "lr"])

    print(f"\n{'Epoch':>5} | {'TrainLoss':>10} | {'ValLoss':>8} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>8}")
    print("-" * 75)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup: 前几个 epoch 线性增长学习率
        if epoch <= cfg["warmup_epochs"]:
            warmup_factor = epoch / cfg["warmup_epochs"]
            optimizer.param_groups[0]["lr"] = cfg["encoder_lr"] * warmup_factor
            optimizer.param_groups[1]["lr"] = cfg["decoder_lr"] * warmup_factor

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, accumulation_steps=cfg.get("grad_accumulation", 1))
        val_loss, val_metrics = validate(model, val_loader, device)

        if epoch > cfg["warmup_epochs"]:
            scheduler.step()

        current_lr = optimizer.param_groups[1]["lr"]  # decoder lr
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_metrics['precision']:>6.4f} | {val_metrics['recall']:>6.4f} | {val_metrics['iou']:>6.4f} | {val_metrics['dice']:>6.4f} | {current_lr:>8.6f} ({elapsed:.0f}s)")

        # 日志
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", 
                f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}", 
                f"{val_metrics['iou']:.6f}", f"{val_metrics['dice']:.6f}", 
                f"{current_lr:.8f}"
            ])

        # Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, SAVE_DIR / "last.pt")

        is_best_iou = val_metrics['iou'] > best_iou
        is_best_dice = val_metrics['dice'] > best_dice
        # 保存逻辑：如果是best_iou或best_dice，就进行保存，否则触发早停机制
        if is_best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_iou.pt")
            print(f"        -> best IoU saved: {best_iou:.4f}")
            
        if is_best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, SAVE_DIR / "best_dice.pt")
            print(f"        -> best Dice saved: {best_dice:.4f}")
        
        # 只要有一个指标创新高，就重置 patience_counter
        if is_best_iou or is_best_dice:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}, Best val Dice: {best_dice:.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
