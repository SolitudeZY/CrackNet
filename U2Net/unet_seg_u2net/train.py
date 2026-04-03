"""
U2NETP 裂缝语义分割训练脚本。

核心策略:
  1. 多尺度监督: 对 7 个输出 (d0-d6) 均施加 Lovász-Hinge + Dice Loss
  2. d0 (fused output) 权重加倍
  3. 按原始图像 ID 划分 train/val (避免数据泄露)
  4. 外部测试集联合验证 (CRKWH100, CrackLS315, Stone331)
  5. AdamW + CosineAnnealingLR + Warmup
  6. Mixed Precision Training (AMP)

数据集选择:
  - crack500 (默认): CRACK500 数据集 (1896 训练), 快速迭代
  - deepcrack:        CrackTree260_augmented (31590 训练), 大规模训练

损失函数选择:
  - lovasz (默认): Lovász-Hinge + Dice — 直接优化 IoU, 适合极度不均衡
  - bce:           BCE + Dice — 传统方案, 用于对比

用法:
  python train.py                                      # CRACK500 + Lovász (快速迭代)
  python train.py --dataset deepcrack                  # DeepCrack 大数据集
  python train.py --loss bce                           # 切换 BCE 损失
  python train.py --resume runs/.../weights/last.pt    # 断点续训
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import U2NETP
from dataset import (
    CrackDataset,
    build_train_augmentation,
    build_val_augmentation,
    split_by_original_image,
)
from losses import U2NetLovaszLoss, U2NetBCEDiceLoss

# ======================== 路径 ========================
DEEPCRACK_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/deepcrack")
CRACK500_ROOT = Path("/home/fs-ai/YOLO-crack-detection/dataset/crack500/CRACK500")

IMG_SIZE = 320  # U2Net 默认输入尺寸

CONFIG = {
    "epochs": 100,
    "batch_size": 32,       # 16GB GPU 安全值; 32GB 可用 --batch-size 48
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "patience": 20,
    "num_workers": 12,
    "seed": 42,
    "grad_accumulation": 1,  # 大 batch 不再需要梯度累积
    "val_ratio": 0.1,
    "loss": "lovasz",
    "dataset": "deepcrack",
}


# ======================== 指标计算 ========================
def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred_mask = (preds > threshold).float()
    t_flat = targets.view(-1)
    p_flat = pred_mask.view(-1)

    tp = (p_flat * t_flat).sum().item()
    fp = (p_flat * (1 - t_flat)).sum().item()
    fn = ((1 - p_flat) * t_flat).sum().item()
    tn = ((1 - p_flat) * (1 - t_flat)).sum().item()

    # 避免分母为0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # 裂缝分割是严重类别不平衡问题，通常背景(0)占绝大多数
    # 当 target 全为背景(0) 且 prediction 也全为背景(0) 时,
    # tp=0, fp=0, fn=0, tn>0, 此时 IoU 和 Dice 应该为 1 (完全预测正确)
    if tp == 0 and fp == 0 and fn == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = tp / (tp + fp + fn + 1e-6)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
    }


# ======================== 训练/验证 ========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            loss, _ = criterion(outputs, masks)

            # 防止损失中的 Nan 破坏所有权重
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Warning] Batch {i} returned NaN loss, skipping...")
                optimizer.zero_grad()
                continue

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if accumulation_steps <= 1 or (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # 防止梯度爆炸
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            # 只有当梯度没有无限大时才更新权重
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * (accumulation_steps if accumulation_steps > 1 else 1) * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """多数据集联合验证。"""
    model.eval()
    per_dataset: Dict[str, Dict[str, float]] = {}
    total_loss = 0.0
    total_samples = 0
    weighted_metrics = {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}

    for ds_name, loader in val_loaders.items():
        ds_loss = 0.0
        ds_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}
        n_samples = len(loader.dataset)

        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    d0 = outputs[0]
                else:
                    d0 = outputs
                loss = F.binary_cross_entropy_with_logits(d0.float(), masks.float())
            ds_loss += loss.item() * images.size(0)

            # The network outputs logits. For inference metrics we need probabilities.
            probs = torch.sigmoid(d0.float())
            m = calculate_metrics(probs, masks)

            # 只在计算有效的图片(非全背景)或者包含有效预测的情况下累加，
            # 否则有些只包含极少裂缝的数据集会被全背景图像拉高/拉低均值。
            # 这里统一所有图片的计算（包含背景图片）
            for k in ds_metrics:
                ds_metrics[k].append(m[k])

        ds_avg = {k: (np.mean(v) if len(v) > 0 else 0.0) for k, v in ds_metrics.items()}
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


# ======================== 数据集构建: CRACK500 ========================
def build_crack500_datasets(cfg: dict) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    CRACK500 数据集 (快速迭代)。
    训练: 1896 张, 验证: 348 张, 测试: 1124 张。
    """
    train_aug = build_train_augmentation(IMG_SIZE)
    val_aug = build_val_augmentation(IMG_SIZE)
    nw = cfg["num_workers"]
    bs = cfg["batch_size"]

    train_dir = CRACK500_ROOT / "traincrop" / "traincrop"
    val_dir = CRACK500_ROOT / "valcrop" / "valcrop"
    test_dir = CRACK500_ROOT / "testcrop" / "testcrop"

    print(f"\n[CRACK500] 训练集:")
    train_ds = CrackDataset(images_dir=train_dir, augmentation=train_aug, img_size=IMG_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    val_loaders: Dict[str, DataLoader] = {}

    if val_dir.exists():
        print(f"\n[CRACK500] 验证集:")
        val_ds = CrackDataset(images_dir=val_dir, augmentation=val_aug, img_size=IMG_SIZE)
        val_loaders["C500_val"] = DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        )

    if test_dir.exists():
        print(f"\n[CRACK500] 测试集:")
        test_ds = CrackDataset(images_dir=test_dir, augmentation=val_aug, img_size=IMG_SIZE)
        val_loaders["C500_test"] = DataLoader(
            test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        )

    # DeepCrack 外部测试集
    deepcrack_tests = [
        ("CRKWH100", DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_img",
         DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_gt"),
        ("CrackLS315", DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_img",
         DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_gt"),
        ("Stone331", DEEPCRACK_ROOT / "Stone331" / "Stone331_img",
         DEEPCRACK_ROOT / "Stone331" / "Stone331_gt"),
    ]
    print(f"\n[DeepCrack] 外部测试集:")
    for name, img_dir, mask_dir in deepcrack_tests:
        if not img_dir.exists():
            print(f"  {name}: 跳过 (目录不存在)")
            continue
        try:
            ds = CrackDataset(images_dir=img_dir, masks_dir=mask_dir, augmentation=val_aug, img_size=IMG_SIZE)
            val_loaders[name] = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")

    total_val = sum(len(ld.dataset) for ld in val_loaders.values())
    print(f"\n[验证集] 总计: {total_val} 张 ({len(val_loaders)} 个数据集)")
    return train_loader, val_loaders


# ======================== 数据集构建: DeepCrack ========================
def build_deepcrack_datasets(cfg: dict) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    DeepCrack CrackTree260_augmented 大数据集 (31590 训练)。
    """
    train_aug = build_train_augmentation(IMG_SIZE)
    val_aug = build_val_augmentation(IMG_SIZE)
    nw = cfg["num_workers"]
    bs = cfg["batch_size"]

    aug_img_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "images"
    aug_mask_dir = DEEPCRACK_ROOT / "CrackTree260_augmented" / "masks"
    if not (aug_img_dir.exists() and aug_mask_dir.exists()):
        raise FileNotFoundError(f"数据未找到: {aug_img_dir}")

    train_files, val_files = split_by_original_image(aug_img_dir, val_ratio=cfg["val_ratio"], seed=cfg["seed"])

    print(f"\n[DeepCrack] 训练集:")
    train_ds = CrackDataset(
        images_dir=aug_img_dir, masks_dir=aug_mask_dir,
        augmentation=train_aug, img_size=IMG_SIZE, file_list=train_files,
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    val_loaders: Dict[str, DataLoader] = {}

    # 内部验证集
    if val_files:
        print(f"\n[DeepCrack] 内部验证集:")
        val_ds = CrackDataset(
            images_dir=aug_img_dir, masks_dir=aug_mask_dir,
            augmentation=val_aug, img_size=IMG_SIZE, file_list=val_files,
        )
        val_loaders["CT260_val"] = DataLoader(
            val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        )

    # 外部测试集
    test_datasets = [
        ("CRKWH100", DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_img",
         DEEPCRACK_ROOT / "CRKWH100" / "CRKWH100_gt"),
        ("CrackLS315", DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_img",
         DEEPCRACK_ROOT / "CrackLS315" / "CrackLS315_gt"),
        ("Stone331", DEEPCRACK_ROOT / "Stone331" / "Stone331_img",
         DEEPCRACK_ROOT / "Stone331" / "Stone331_gt"),
    ]
    print(f"\n[DeepCrack] 外部测试集:")
    for name, img_dir, mask_dir in test_datasets:
        if not img_dir.exists():
            print(f"  {name}: 跳过 (目录不存在)")
            continue
        try:
            ds = CrackDataset(images_dir=img_dir, masks_dir=mask_dir, augmentation=val_aug, img_size=IMG_SIZE)
            val_loaders[name] = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")

    total_val = sum(len(ld.dataset) for ld in val_loaders.values())
    print(f"\n[验证集] 总计: {total_val} 张 ({len(val_loaders)} 个数据集)")
    return train_loader, val_loaders


# ======================== 主训练循环 ========================
def main():
    parser = argparse.ArgumentParser(description="U2NETP 裂缝分割训练")
    parser.add_argument("--dataset", type=str, default="crack500", choices=["crack500", "deepcrack"],
                        help="数据集: crack500 (1896张, 快速) 或 deepcrack (31590张, 大规模)")
    parser.add_argument("--loss", type=str, default="lovasz", choices=["lovasz", "bce"],
                        help="损失函数: lovasz 或 bce")
    parser.add_argument("--resume", type=str, default="", help="断点续训 checkpoint 路径")
    parser.add_argument("--batch-size", type=int, default=0, help="覆盖默认 batch size")
    parser.add_argument("--epochs", type=int, default=0, help="覆盖默认 epoch 数")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["dataset"] = args.dataset
    cfg["loss"] = args.loss
    if args.resume:
        cfg["resume"] = args.resume
    if args.batch_size > 0:
        cfg["batch_size"] = args.batch_size
    if args.epochs > 0:
        cfg["epochs"] = args.epochs

    # 输出目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / "runs" / "train" / f"exp_{current_time}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir = run_dir / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 随机种子
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 固定输入尺寸, 让 cuDNN 自动选最快算法
    print(f"Device: {device}")
    print(f"Output: {run_dir}")

    # 数据集
    if cfg["dataset"] == "crack500":
        train_loader, val_loaders = build_crack500_datasets(cfg)
    else:
        train_loader, val_loaders = build_deepcrack_datasets(cfg)

    # 模型
    model = U2NETP(3, 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: U2NETP ({n_params:,} params)")
    print(f"Input: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Dataset: {cfg['dataset']} | Batch: {cfg['batch_size']} | Epochs: {cfg['epochs']}")

    # 损失函数
    if cfg["loss"] == "lovasz":
        criterion = U2NetLovaszLoss(per_image=True)
        print(f"Loss: Lovász-Hinge + Dice")
    else:
        criterion = U2NetBCEDiceLoss()
        print(f"Loss: BCE + Dice")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # CosineAnnealingLR 从 warmup 结束后开始，T_max 为实际有效 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epochs"]), eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")

    # 状态
    start_epoch = 1
    best_iou = 0.0
    best_dice = 0.0
    patience_counter = 0

    # 断点续训
    if cfg.get("resume"):
        ckpt_path = Path(cfg["resume"])
        if ckpt_path.exists():
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            if "val_metrics" in ckpt:
                best_iou = ckpt["val_metrics"].get("iou", 0.0)
                best_dice = ckpt["val_metrics"].get("dice", 0.0)
            print(f"Resumed from epoch {start_epoch - 1}, best IoU={best_iou:.4f}")

    # 日志
    log_path = run_dir / "training_log.csv"
    val_ds_names = list(val_loaders.keys())
    if start_epoch == 1:
        header = ["epoch", "train_loss", "val_loss", "precision", "recall", "iou", "dice", "lr"]
        for ds_name in val_ds_names:
            header.append(f"iou_{ds_name}")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # 保存配置
    with open(run_dir / "config.json", "w") as f:
        json.dump({**cfg, "img_size": IMG_SIZE, "model": "U2NETP", "params": n_params}, f, indent=2)

    # 表头
    print(f"\n{'Epoch':>5} | {'TrLoss':>8} | {'VLoss':>7} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>9} |", end="")
    for ds_name in val_ds_names:
        short = ds_name[:7]
        print(f" {short:>7}", end="")
    print()
    print("-" * (85 + 8 * len(val_ds_names)))

    # DeepCrack 大数据集: 验证集也大 (3510+746), 每 N 个 epoch 验证一次节省时间
    val_interval = 5 if cfg["dataset"] == "deepcrack" else 1
    last_val_loss = float("nan")
    last_val_metrics = {"precision": float("nan"), "recall": float("nan"), "iou": float("nan"), "dice": float("nan")}
    last_per_dataset: Dict[str, Dict[str, float]] = {}

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        # Warmup (手工设定)
        if epoch <= cfg["warmup_epochs"]:
            factor = epoch / cfg["warmup_epochs"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg["lr"] * factor
        else:
            scheduler.step()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            accumulation_steps=cfg["grad_accumulation"],
        )

        # 验证 (大数据集间隔验证)
        did_validate = (epoch % val_interval == 0 or epoch == start_epoch or epoch == cfg["epochs"])
        if did_validate:
            verbose = (epoch % 10 == 0) or (epoch <= 3)
            val_loss, val_metrics, per_dataset = validate(model, val_loaders, device, verbose=verbose)
            last_val_loss = val_loss
            last_val_metrics = val_metrics
            last_per_dataset = per_dataset
        else:
            val_loss = last_val_loss
            val_metrics = last_val_metrics
            per_dataset = last_per_dataset

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # 输出
        line = (
            f"{epoch:>5} | {train_loss:>8.4f} | {val_loss:>7.4f} | "
            f"{val_metrics['precision']:>6.4f} | {val_metrics['recall']:>6.4f} | "
            f"{val_metrics['iou']:>6.4f} | {val_metrics['dice']:>6.4f} | "
            f"{lr:>9.2e} |"
        )
        for ds_name in val_ds_names:
            ds_iou = per_dataset.get(ds_name, {}).get("iou", 0.0)
            line += f" {ds_iou:>7.4f}"
        if not did_validate:
            line += "  [skip val]"
        line += f"  ({elapsed:.0f}s)"
        print(line)

        # 日志
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
        }, save_dir / "last.pt")

        # Best checkpoints (仅在执行验证的 epoch 更新)
        improved = False

        if did_validate and val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       save_dir / "best_iou.pt")
            print(f"        -> best IoU: {best_iou:.4f}")
            improved = True

        if did_validate and val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
                       save_dir / "best_dice.pt")
            print(f"        -> best Dice: {best_dice:.4f}")
            improved = True

        if did_validate:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
                    break

    print(f"\nTraining complete.")
    print(f"Best: IoU={best_iou:.4f}, Dice={best_dice:.4f}")
    print(f"Weights: {save_dir}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
