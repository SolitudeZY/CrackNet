"""
U2Net 裂缝语义分割训练脚本，默认训练全量 U2NET 于 CRACK500。

示例:
  python train.py
  python train.py --model u2net --img-size 320 --epochs 200
  python train.py --model u2netp --loss bce
  python train.py --resume runs/train/exp_xxx/weights/last.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CRACK500Dataset, build_train_augmentation, build_val_augmentation
from losses import U2NetBCEDiceLoss, U2NetBoundaryLoss, U2NetLovaszLoss
from model import build_model

CRACK500_ROOT = Path("/home/fs-ai/unet-crack/dataset/CRACK500")

DEFAULT_CONFIG = {
    "model": "u2net",
    "epochs": 180,
    "batch_size": 4,
    "img_size": 512,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "patience": 60,
    "num_workers": 8,
    "seed": 42,
    "grad_accumulation": 2,
    "loss": "boundary",
    "pretrained": False,
    "amp_mode": "auto",
    "label_smoothing": 0.0,
}

COMMON_PRETRAINED_PATHS = {
    "u2net": [
        Path("/home/fs-ai/unet-crack/U-2-Net/saved_models/u2net/u2net.pth"),
        Path("/home/fs-ai/unet-crack/U-2-Net/saved_models/u2net/u2net_portrait.pth"),
    ],
    "u2netp": [
        Path("/home/fs-ai/unet-crack/U-2-Net/saved_models/u2netp/u2netp.pth"),
    ],
}


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


def build_loaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    if not CRACK500_ROOT.exists():
        raise FileNotFoundError(f"CRACK500 数据集未找到: {CRACK500_ROOT}")

    train_ds = CRACK500Dataset(
        dataset_dir=CRACK500_ROOT,
        split="train",
        augmentation=build_train_augmentation(cfg["img_size"]),
        img_size=cfg["img_size"],
    )
    val_ds = CRACK500Dataset(
        dataset_dir=CRACK500_ROOT,
        split="val",
        augmentation=build_val_augmentation(cfg["img_size"]),
        img_size=cfg["img_size"],
    )

    num_workers = cfg["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    valid_samples = 0
    skipped_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
        outputs_fp32 = tuple(output.float() for output in outputs)
        masks_fp32 = masks.float()
        loss, _ = criterion(outputs_fp32, masks_fp32)

        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps * images.size(0)
        valid_samples += images.size(0)

    if skipped_batches > 0:
        print(f"        [train] skipped non-finite batches: {skipped_batches}")

    if valid_samples == 0:
        return float("nan")
    return total_loss / valid_samples


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    valid_samples = 0
    skipped_batches = 0
    all_metrics = {"precision": [], "recall": [], "iou": [], "dice": []}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
        d0 = outputs[0].float()
        masks_fp32 = masks.float()
        loss = F.binary_cross_entropy_with_logits(d0, masks_fp32)

        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        total_loss += loss.item() * images.size(0)
        valid_samples += images.size(0)
        probs = torch.sigmoid(d0)
        metrics = calculate_metrics(probs, masks_fp32)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    if skipped_batches > 0:
        print(f"        [val] skipped non-finite batches: {skipped_batches}")

    avg_loss = total_loss / max(valid_samples, 1)
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


def resolve_pretrained_path(model_name: str, user_path: str) -> Path | None:
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"预训练权重不存在: {path}")
        return path

    for path in COMMON_PRETRAINED_PATHS.get(model_name, []):
        if path.exists():
            return path
    return None


def load_pretrained_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_inference_checkpoint(
    epoch: int,
    model: nn.Module,
    cfg: dict,
    val_metrics: dict[str, float],
) -> dict:
    return {
        "epoch": epoch,
        "model_name": cfg["model"],
        "img_size": cfg["img_size"],
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "config": cfg,
    }


def resolve_amp_enabled(
    device: torch.device,
    model_name: str,
    amp_mode: str,
    pretrained_active: bool,
) -> bool:
    if device.type != "cuda":
        return False
    if amp_mode == "on":
        return True
    if amp_mode == "off":
        return False
    if model_name == "u2net" and pretrained_active:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full U2NET on CRACK500")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"], choices=["u2net", "u2netp"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--img-size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_CONFIG["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--grad-accumulation", type=int, default=DEFAULT_CONFIG["grad_accumulation"])
    parser.add_argument("--loss", type=str, default=DEFAULT_CONFIG["loss"], choices=["lovasz", "bce", "boundary"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--dataset-root", type=str, default=str(CRACK500_ROOT))
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["pretrained"],
        help="是否加载预训练权重后再进行裂缝微调",
    )
    parser.add_argument("--pretrained-path", type=str, default="", help="预训练全量 U2Net 权重路径")
    parser.add_argument(
        "--amp-mode",
        type=str,
        default=DEFAULT_CONFIG["amp_mode"],
        choices=["auto", "on", "off"],
        help="AMP 模式: auto=随机初始化自动开、预训练全量U2Net自动关",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_CONFIG["label_smoothing"],
        help="二值标签平滑系数，默认0关闭；建议裂缝任务保持很小，如0.01-0.03",
    )
    return parser.parse_args()


def main() -> None:
    global CRACK500_ROOT
    args = parse_args()
    cfg = vars(args).copy()

    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    if dataset_root != CRACK500_ROOT:
        CRACK500_ROOT = dataset_root

    run_dir = Path(__file__).resolve().parent / "runs" / "train" / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = run_dir / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg["model"], in_ch=3, out_ch=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if cfg["loss"] == "lovasz":
        criterion = U2NetLovaszLoss(per_image=True, label_smoothing=cfg["label_smoothing"])
    elif cfg["loss"] == "bce":
        criterion = U2NetBCEDiceLoss(label_smoothing=cfg["label_smoothing"])
    else:
        criterion = U2NetBoundaryLoss(label_smoothing=cfg["label_smoothing"])

    start_epoch = 1
    best_iou = 0.0
    best_dice = 0.0
    patience_counter = 0
    pretrained_path = None
    pretrained_active = False

    if cfg["pretrained"] and not cfg["resume"]:
        pretrained_path = resolve_pretrained_path(cfg["model"], cfg["pretrained_path"])
        if pretrained_path is None:
            print("\n[Warning] 未找到本地预训练权重，将从随机初始化开始训练。")
            print("          可通过 --pretrained-path 指定全量 U2Net 权重。")
        else:
            missing_keys, unexpected_keys = load_pretrained_weights(model, pretrained_path, device)
            pretrained_active = True
            print(f"\nLoaded pretrained weights: {pretrained_path}")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")

    amp_enabled = resolve_amp_enabled(device, cfg["model"], cfg["amp_mode"], pretrained_active)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg["epochs"] - cfg["warmup_epochs"]),
        eta_min=1e-6,
    )

    if cfg["resume"]:
        ckpt_path = Path(cfg["resume"]).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        val_metrics = ckpt.get("val_metrics", {})
        best_iou = float(val_metrics.get("iou", 0.0))
        best_dice = float(val_metrics.get("dice", 0.0))

    log_path = run_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "precision", "recall", "iou", "dice", "lr"])

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **cfg,
                "model_params": n_params,
                "dataset": "CRACK500",
                "dataset_root": str(CRACK500_ROOT),
                "device": str(device),
                "resolved_pretrained_path": str(pretrained_path) if pretrained_path else "",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Device: {device}")
    print(f"Model: {cfg['model']} ({n_params:,} params)")
    print(f"AMP enabled: {amp_enabled}")
    print(f"AMP mode: {cfg['amp_mode']}")
    print(f"Pretrained active: {pretrained_active}")
    print(f"Label smoothing: {cfg['label_smoothing']}")
    print(f"Dataset: {CRACK500_ROOT}")
    print(f"Output: {run_dir}")
    print(f"\n{'Epoch':>5} | {'TrLoss':>8} | {'VLoss':>7} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>9} |")
    print("-" * 85)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        if epoch <= cfg["warmup_epochs"]:
            warmup_factor = epoch / max(1, cfg["warmup_epochs"])
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * warmup_factor

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            accumulation_steps=cfg["grad_accumulation"],
            amp_enabled=amp_enabled,
        )
        val_loss, val_metrics = validate(model, val_loader, device, amp_enabled)

        if epoch > cfg["warmup_epochs"] and np.isfinite(train_loss):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"{epoch:>5} | {train_loss:>8.4f} | {val_loss:>7.4f} | "
            f"{val_metrics['precision']:>6.4f} | {val_metrics['recall']:>6.4f} | "
            f"{val_metrics['iou']:>6.4f} | {val_metrics['dice']:>6.4f} | {lr:>9.2e} |  ({elapsed:.0f}s)"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}",
                f"{val_metrics['iou']:.6f}",
                f"{val_metrics['dice']:.6f}",
                f"{lr:.8f}",
            ])

        last_state = {
            "epoch": epoch,
            "model_name": cfg["model"],
            "img_size": cfg["img_size"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": cfg,
        }
        torch.save(last_state, save_dir / "last.pt")

        improved = False
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(build_inference_checkpoint(epoch, model, cfg, val_metrics), save_dir / "best_iou.pt")
            print(f"        -> best IoU: {best_iou:.4f}")
            improved = True
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(build_inference_checkpoint(epoch, model, cfg, val_metrics), save_dir / "best_dice.pt")
            print(f"        -> best Dice: {best_dice:.4f}")
            improved = True

        patience_counter = 0 if improved else patience_counter + 1
        if patience_counter >= cfg["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={cfg['patience']})")
            break

    print(f"\nTraining complete.")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Weights: {save_dir}")


if __name__ == "__main__":
    main()
