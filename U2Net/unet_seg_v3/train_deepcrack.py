"""
DeepCrack 策略训练脚本: 多尺度侧输出监督 + 多数据集联合训练与测试。

核心策略 (来自 DeepCrack, Zou et al. TIP 2019):
  1. 多尺度侧输出监督: 对 6 个输出 (5 侧输出 + 1 融合) 均施加 BCE Loss，等权求和
  2. 多数据集联合训练: CRACK500 + CrackTree260
  3. 多数据集联合测试: 训练结束后在 CRKWH100、CrackLS315、Stone331 上分别评估
  4. albumentations 数据增强 (弹性变形、噪声、模糊、CLAHE 等)

用法: python train_deepcrack.py
"""
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from DeepCrack_MobileNetV2 import DeepCrack_MobileNetV2


# ======================== 配置 ========================
CRACK500_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")
DEEPCRACK_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).resolve().parent / "runs" / "train_deepcrack" / f"exp_{current_time}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = RUN_DIR / "weights"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 归一化: 跨域训练使用 ImageNet 统一值 (而非 CRACK500 专属统计值)
# 原因: CRACK500 的 mean/std 对 CrackLS315(激光成像)、Stone331(石材表面) 等域外数据不适用，
#       会导致输入分布偏移，模型在这些数据集上几乎无法预测。
#       ImageNet 统计值是视觉模型的通用标准，MobileNetV2 也是用它预训练的。
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]
print(f"归一化: ImageNet 标准值 (跨域通用) Mean={DATASET_MEAN}, Std={DATASET_STD}")

IMG_SIZE = 512  # DeepCrack 原版使用 512x512

CONFIG = {
    # 训练
    "epochs": 300,
    "batch_size": 8,
    "encoder_lr": 1e-4,
    "decoder_lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "patience": 80,
    "num_workers": 12,
    "seed": 42,
    "grad_accumulation": 2,
    "attention_type": "CBAM",
    # DeepCrack 损失
    "pos_weight": 2.0,       # 正样本权重 (裂缝稀疏，给予更高权重)
    # 断点续训
    "resume": False,
    "resume_ckpt": "",
}


# ======================== 数据增强 ========================
def build_train_augmentation(img_size: int) -> A.Compose:
    """
    训练增强管线 (配合离线增强后的 35100 张数据)。

    离线增强已覆盖: 9 旋转 × 3 翻转 × 5 裁剪 → 几何多样性充足。
    在线只做轻量像素级增强 (不再做 ElasticTransform/GridDistortion 等重 CPU 操作):
      - 亮度/对比度/色相抖动 (模拟不同光照)
      - 轻度噪声和模糊 (模拟传感器)
      - Normalize

    注: 离线数据已经是 512×512，无需 resize/crop。
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.15),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def build_val_augmentation(img_size: int) -> A.Compose:
    """验证/测试: 仅 resize + normalize。"""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


# ======================== Dataset ========================
class CrackDataset(Dataset):
    """通用裂缝分割数据集，支持 CRACK500 / DeepCrack 多种格式。"""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        augmentation: Optional[A.Compose] = None,
        img_size: int = 512,
        img_suffixes: Optional[List[str]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.augmentation = augmentation
        self.img_size = img_size

        if img_suffixes is not None:
            suffixes = img_suffixes
        elif self.masks_dir:
            suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]
        else:
            suffixes = [".jpg", ".JPG", ".jpeg"]

        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix in suffixes])
        if not self.images:
            raise FileNotFoundError(f"在 {self.images_dir} 中未找到任何图片")

        self._verify_pairs()
        print(f"  {self.images_dir.name}: {len(self.images)} 张图片")

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        stem = img_path.stem
        if self.masks_dir:
            for ext in (".bmp", ".png", ".jpg", ".tif"):
                p = self.masks_dir / (stem + ext)
                if p.exists():
                    return p
        else:
            p = img_path.with_suffix(".png")
            if p.exists() and p != img_path:
                return p
        return None

    def _verify_pairs(self):
        missing = [p.name for p in self.images[:5] if self._find_mask(p) is None]
        if missing:
            raise FileNotFoundError(f"以下图片找不到对应掩码: {missing}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self._find_mask(img_path)
        if mask_path is None:
            raise RuntimeError(f"找不到掩码: {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"无法读取掩码: {mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = ((mask > 127).astype(np.uint8)) * 255

        if self.augmentation:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] - DATASET_MEAN[c]) / DATASET_STD[c]

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # ======================== DeepCrack 损失函数 ========================
        # 修正 target mask，因为离线生成的掩码读进来变成了 0 和 255，并且除以 255 后可能不是严格的 0/1
        # 需要确保 mask_t 在 0 和 1 之间
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        mask_t = torch.from_numpy((mask_f > 0.5).astype(np.float32)).unsqueeze(0)

        return image_t, mask_t


# ======================== 指标计算 ========================
def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred_mask = (preds > threshold).float()
    t_flat = targets.view(-1)
    p_flat = pred_mask.view(-1)
    tp = (p_flat * t_flat).sum().item()
    fp = (p_flat * (1 - t_flat)).sum().item()
    fn = ((1 - p_flat) * t_flat).sum().item()
    return {
        "precision": tp / (tp + fp + 1e-6),
        "recall": tp / (tp + fn + 1e-6),
        "iou": tp / (tp + fp + fn + 1e-6),
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-6),
    }


# ======================== DeepCrack 损失函数 ========================
class DeepCrackLoss(nn.Module):
    """
    DeepCrack 多尺度 BCE 损失:
      L_total = BCE(fused, y) + sum_{k=1}^{5} BCE(side_k, y)
    每个输出都被独立监督，等权求和。
    使用 pos_weight 处理正负样本不平衡。
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: (fused, side1, side2, side3, side4, side5) — 6 个 logits
            target: (B, 1, H, W) 二值标签
        Returns:
            total_loss, loss_dict
        """
        pw = self.pos_weight.to(target.device)
        names = ["fused", "side1", "side2", "side3", "side4", "side5"]
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device)

        for name, logit in zip(names, outputs):
            loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw)
            loss_dict[name] = loss.item()
            total = total + loss

        loss_dict["total"] = total.item()
        return total, loss_dict


# ======================== 训练/验证 ========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: DeepCrackLoss,
    device: torch.device,
    accumulation_steps: int = 2,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)  # (fused, side1, ..., side5)
            loss, _ = criterion(outputs, masks)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    criterion: DeepCrackLoss,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    多数据集联合验证。
    Returns:
        avg_loss: 所有数据集的加权平均 loss
        avg_metrics: 所有数据集的加权平均指标 (按样本数加权)
        per_dataset: {数据集名: {precision, recall, iou, dice}} 逐数据集指标
    """
    model.eval()
    pw = criterion.pos_weight.to(device)

    per_dataset: Dict[str, Dict[str, float]] = {}
    total_loss = 0.0
    total_samples = 0
    # 加权求和用
    weighted_metrics = {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}

    for ds_name, loader in val_loaders.items():
        ds_loss = 0.0
        ds_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}
        n_samples = len(loader.dataset)

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast("cuda"):
                fused = model(images)
                if isinstance(fused, tuple):
                    fused = fused[0] # Take only the final fused output for validation
                loss = F.binary_cross_entropy_with_logits(fused, masks, pos_weight=pw)
            ds_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(fused)
            m = calculate_metrics(probs, masks)
            for k in ds_metrics:
                ds_metrics[k].append(m[k])

        ds_avg = {k: np.mean(v) for k, v in ds_metrics.items()}
        per_dataset[ds_name] = ds_avg
        total_loss += ds_loss
        total_samples += n_samples

        for k in weighted_metrics:
            weighted_metrics[k] += ds_avg[k] * n_samples

        if verbose:
            print(f"    {ds_name:>12s}: P={ds_avg['precision']:.4f} R={ds_avg['recall']:.4f} "
                  f"IoU={ds_avg['iou']:.4f} Dice={ds_avg['dice']:.4f}")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_metrics = {k: v / total_samples for k, v in weighted_metrics.items()} if total_samples > 0 else weighted_metrics

    return avg_loss, avg_metrics, per_dataset


# ======================== 多数据集联合测试 ========================
@torch.no_grad()
def evaluate_on_dataset(
    model: nn.Module,
    dataset_name: str,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """在单个测试数据集上评估模型性能。"""
    model.eval()
    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            fused = model(images)
            if isinstance(fused, tuple):
                fused = fused[0] # Take only the final fused output for validation
        probs = torch.sigmoid(fused)
        m = calculate_metrics(probs, masks)
        for k in all_metrics:
            all_metrics[k].append(m[k])

    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    print(f"  [{dataset_name:>12s}] Precision={avg['precision']:.4f}  Recall={avg['recall']:.4f}  "
          f"IoU={avg['iou']:.4f}  Dice={avg['dice']:.4f}")
    return avg


def multi_dataset_test(model: nn.Module, device: torch.device) -> Dict[str, Dict[str, float]]:
    """
    在 DeepCrack 的三个测试集上分别评估 (遵循论文设定)。
    """
    val_aug = build_val_augmentation(IMG_SIZE)
    results = {}

    test_datasets = [
        ("CRKWH100", DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_img",
         DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_gt"),
        ("CrackLS315", DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_img",
         DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_gt"),
        ("Stone331", DEEPCRACK_ROOT / "Stone331" / "Stone331_img",
         DEEPCRACK_ROOT / "Stone331" / "Stone331_gt"),
    ]

    print("\n" + "=" * 70)
    print("多数据集联合测试结果")
    print("=" * 70)

    for name, img_dir, mask_dir in test_datasets:
        if not img_dir.exists():
            print(f"  [{name:>12s}] 跳过 (目录不存在)")
            continue
        try:
            ds = CrackDataset(
                images_dir=img_dir,
                masks_dir=mask_dir,
                augmentation=val_aug,
                img_size=IMG_SIZE,
            )
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
            results[name] = evaluate_on_dataset(model, name, loader, device)
        except Exception as e:
            print(f"  [{name:>12s}] 跳过 (错误: {e})")

    # 计算所有测试集的平均指标
    if results:
        avg_all = {}
        for k in ("precision", "recall", "iou", "dice"):
            avg_all[k] = np.mean([r[k] for r in results.values()])
        print("-" * 70)
        print(f"  [{'AVERAGE':>12s}] Precision={avg_all['precision']:.4f}  Recall={avg_all['recall']:.4f}  "
              f"IoU={avg_all['iou']:.4f}  Dice={avg_all['dice']:.4f}")
        results["AVERAGE"] = avg_all

    print("=" * 70)
    return results


# ======================== 数据集构建 ========================
def build_datasets() -> Tuple[Dataset, Dict[str, DataLoader]]:
    """
    遵循 DeepCrack 原论文设定:
      训练集: CrackTree260_augmented (35100 张离线增强 + 在线增强)
      验证集: CRKWH100 + CrackLS315 + Stone331 (三数据集联合验证)
    """
    train_aug = build_train_augmentation(IMG_SIZE)
    val_aug = build_val_augmentation(IMG_SIZE)

    # ---- 训练集: 离线增强后的 CrackTree260 (35100 张) ----
    aug_img_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "images"
    aug_mask_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "masks"
    if not (aug_img_dir.exists() and aug_mask_dir.exists()):
        raise FileNotFoundError(
            f"离线增强数据未找到: {aug_img_dir}\n"
            f"请先运行: python augment_cracktree260.py"
        )

    print("\n[训练集] CrackTree260_augmented (DeepCrack 论文离线增强):")
    train_ds = CrackDataset(
        images_dir=aug_img_dir,
        masks_dir=aug_mask_dir,
        augmentation=train_aug,
        img_size=IMG_SIZE,
    )

    # ---- 验证集: DeepCrack 三测试集 ----
    val_datasets_cfg = [
        ("CRKWH100", DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_img",
         DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_gt"),
        ("CrackLS315", DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_img",
         DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_gt"),
        ("Stone331", DEEPCRACK_ROOT / "Stone331" / "Stone331_img",
         DEEPCRACK_ROOT / "Stone331" / "Stone331_gt"),
    ]

    val_loaders: Dict[str, DataLoader] = {}
    print("\n[验证集] DeepCrack 三数据集联合验证:")
    for name, img_dir, mask_dir in val_datasets_cfg:
        if not img_dir.exists():
            print(f"  {name}: 跳过 (目录不存在)")
            continue
        try:
            ds = CrackDataset(
                images_dir=img_dir,
                masks_dir=mask_dir,
                augmentation=val_aug,
                img_size=IMG_SIZE,
            )
            val_loaders[name] = DataLoader(
                ds, batch_size=CONFIG["batch_size"], shuffle=False,
                num_workers=CONFIG["num_workers"], pin_memory=True,
            )
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")

    total_val = sum(len(loader.dataset) for loader in val_loaders.values())
    print(f"[验证集] 总计: {total_val} 张 ({len(val_loaders)} 个数据集)")

    return train_ds, val_loaders


# ======================== 主训练循环 ========================
def main():
    cfg = CONFIG

    # 随机种子
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据集 (v3: 多数据集联合训练 + 联合验证)
    train_ds, val_loaders = build_datasets()
    nw = cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,  # 避免每个 epoch 重新 fork worker
        prefetch_factor=3 if nw > 0 else None,  # 每个 worker 预取 3 批
    )

    # 模型
    model = DeepCrack_MobileNetV2(
        pretrained=True,
        attention_type=cfg["attention_type"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: DeepCrack_MobileNetV2 (Attention={cfg['attention_type']})")
    print(f"Parameters: {n_params:,}")

    # 优化器: 差分学习率
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": cfg["encoder_lr"]},
        {"params": model.decoder_params(), "lr": cfg["decoder_lr"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"], eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")

    # DeepCrack 损失
    criterion = DeepCrackLoss(pos_weight=cfg["pos_weight"])

    # 状态
    start_epoch = 1
    best_iou = 0.0
    best_dice = 0.0
    best_recall = 0.0
    best_precision = 0.0
    patience_counter = 0

    # 断点续训
    if cfg.get("resume") and cfg.get("resume_ckpt"):
        ckpt_path = Path(cfg["resume_ckpt"])
        if ckpt_path.exists():
            print(f"Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            if "val_metrics" in ckpt:
                best_iou = ckpt["val_metrics"].get("iou", 0.0)
                best_dice = ckpt["val_metrics"].get("dice", 0.0)
                best_recall = ckpt["val_metrics"].get("recall", 0.0)
                best_precision = ckpt["val_metrics"].get("precision", 0.0)
            print(f"Resumed from epoch {start_epoch - 1}, best IoU={best_iou:.4f}")

    # 日志
    log_path = RUN_DIR / "training_log.csv"
    val_ds_names = list(val_loaders.keys())
    if start_epoch == 1:
        header = ["epoch", "train_loss", "val_loss", "precision", "recall", "iou", "dice", "lr"]
        # 追加每个验证数据集的 IoU 列
        for ds_name in val_ds_names:
            header.append(f"iou_{ds_name}")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # 保存配置
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump({**cfg, "img_size": IMG_SIZE, "model": "DeepCrack_MobileNetV2", "params": n_params}, f, indent=2)

    print(f"\n{'Epoch':>5} | {'TrLoss':>8} | {'VLoss':>7} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>9} |", end="")
    for ds_name in val_ds_names:
        # 截取短名方便显示
        short = ds_name.replace("CRACK500_", "C5_").replace("CrackLS", "LS")
        print(f" {short:>7}", end="")
    print()
    print("-" * (85 + 8 * len(val_ds_names)))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup
        if epoch <= cfg["warmup_epochs"]:
            factor = epoch / cfg["warmup_epochs"]
            optimizer.param_groups[0]["lr"] = cfg["encoder_lr"] * factor
            optimizer.param_groups[1]["lr"] = cfg["decoder_lr"] * factor

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            accumulation_steps=cfg["grad_accumulation"],
        )

        # 多数据集联合验证 (每 10 个 epoch 打印详细逐数据集信息)
        verbose = (epoch % 10 == 0) or (epoch <= 3)
        val_loss, val_metrics, per_dataset = validate(model, val_loaders, criterion, device, verbose=verbose)

        if epoch > cfg["warmup_epochs"]:
            scheduler.step()

        lr = optimizer.param_groups[1]["lr"]
        elapsed = time.time() - t0

        # 主行输出: 加权平均指标 + 各数据集 IoU
        line = (
            f"{epoch:>5} | {train_loss:>8.4f} | {val_loss:>7.4f} | "
            f"{val_metrics['precision']:>6.4f} | {val_metrics['recall']:>6.4f} | "
            f"{val_metrics['iou']:>6.4f} | {val_metrics['dice']:>6.4f} | "
            f"{lr:>9.2e} |"
        )
        for ds_name in val_ds_names:
            ds_iou = per_dataset.get(ds_name, {}).get("iou", 0.0)
            line += f" {ds_iou:>7.4f}"
        line += f"  ({elapsed:.0f}s)"
        print(line)

        # 日志 (含各数据集 IoU)
        row = [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}",
            f"{val_metrics['iou']:.6f}", f"{val_metrics['dice']:.6f}",
            f"{lr:.8f}",
        ]
        for ds_name in val_ds_names:
            row.append(f"{per_dataset.get(ds_name, {}).get('iou', 0.0):.6f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Checkpoint: last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "per_dataset": per_dataset,
        }, SAVE_DIR / "last.pt")

        # Best checkpoints
        improved = False

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       SAVE_DIR / "best_iou.pt")
            print(f"        -> best IoU: {best_iou:.4f}")
            improved = True

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       SAVE_DIR / "best_dice.pt")
            print(f"        -> best Dice: {best_dice:.4f}")
            improved = True

        if val_metrics["precision"] > best_precision:
            best_precision = val_metrics["precision"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       SAVE_DIR / "best_precision.pt")
            print(f"        -> best Precision: {best_precision:.4f}")
            improved = True

        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       SAVE_DIR / "best_recall.pt")
            print(f"        -> best Recall: {best_recall:.4f}")
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    # ======================== 训练完成，多数据集联合测试 ========================
    print(f"\nTraining complete.")
    print(f"Best: IoU={best_iou:.4f}, Dice={best_dice:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f}")

    # 加载 best_iou 权重进行最终测试
    best_ckpt = SAVE_DIR / "best_iou.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n加载 best_iou.pt (epoch {ckpt['epoch']}) 进行多数据集联合测试...")
    else:
        print("\n使用最后一个 epoch 的权重进行多数据集联合测试...")

    test_results = multi_dataset_test(model, device)

    # 保存测试结果
    test_results_path = RUN_DIR / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n测试结果已保存到: {test_results_path}")
    print(f"训练日志: {log_path}")
    print(f"权重目录: {SAVE_DIR}")


if __name__ == "__main__":
    main()
