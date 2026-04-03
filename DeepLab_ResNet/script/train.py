"""
Training script for DeepLabV3+ ResNet101 crack segmentation (v2).

Key improvements over v1:
- OneCycleLR scheduler (better exploration, prevents early convergence)
- EMA (Exponential Moving Average) for stable validation
- Stronger augmentation pipeline
- Better warmup strategy
- Separate best_dice checkpoint

  python train.py --crack500-only
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dataset import build_dataloaders, SampledDataset, CRACK500_ROOT, DEEPCRACK_ROOT
from losses import build_loss_with_aux
from metrics import average_metric_list, compute_metrics
from model import build_model
from sliding_window import sliding_window_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ ResNet101 crack segmentation")
    parser.add_argument("--crack500-root", type=str, default=str(CRACK500_ROOT))
    parser.add_argument("--deepcrack-root", type=str, default=str(DEEPCRACK_ROOT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay factor (0 to disable)")
    parser.add_argument("--selection-metric", type=str, default="iou", choices=["precision", "iou", "dice"])
    parser.add_argument("--selection-threshold", type=float, default=0.5)
    parser.add_argument("--auto-selection-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selection-threshold-min", type=float, default=0.4)
    parser.add_argument("--selection-threshold-max", type=float, default=0.8)
    parser.add_argument("--selection-threshold-step", type=float, default=0.02)
    parser.add_argument("--selection-min-recall", type=float, default=0.18)
    parser.add_argument("--selection-min-iou", type=float, default=0.18)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--tversky-weight", type=float, default=0.45)
    parser.add_argument("--boundary-weight", type=float, default=0.4)
    parser.add_argument("--detail-weight", type=float, default=0.2)
    parser.add_argument("--tversky-alpha", type=float, default=0.55)
    parser.add_argument("--tversky-beta", type=float, default=0.45)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument("--crack500-only", action="store_true", default=True,
                        help="Only use CRACK500 dataset (default: True)")
    parser.add_argument("--use-deepcrack", action="store_true",
                        help="Also include DeepCrack datasets")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model parameters for more stable evaluation.

    Maintains a shadow copy of model weights updated as:
        shadow_weight = decay * shadow_weight + (1 - decay) * model_weight

    Uses decay warmup: starts at 0.9 (fast tracking) and linearly increases
    to the target decay over the first warmup_steps steps. This ensures
    the EMA model is usable for evaluation from the very first epoch.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup_steps: int = 500):
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @property
    def decay(self) -> float:
        if self.step_count >= self.warmup_steps:
            return self.target_decay
        # Linear warmup from 0.9 to target_decay
        t = self.step_count / max(self.warmup_steps, 1)
        return 0.9 + (self.target_decay - 0.9) * t

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        self.step_count += 1
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        # Also update buffers (e.g. BatchNorm running stats)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: torch.nn.Module, lr: float, backbone_lr_scale: float, weight_decay: float) -> torch.optim.Optimizer:
    """Differential learning rate: backbone gets lr * backbone_lr_scale."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
    print(f"  Head/Decoder params: {sum(p.numel() for p in head_params):,}")

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_scale},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """OneCycleLR for aggressive exploration then gentle annealing."""
    total_steps = epochs * steps_per_epoch
    pct_start = min(warmup_epochs / epochs, 0.3)  # cap at 30% warmup
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr * 0.1, lr],  # [backbone_max_lr, head_max_lr]
        total_steps=total_steps,
        pct_start=pct_start,  # warmup fraction
        anneal_strategy="cos",
        div_factor=10.0,      # start_lr = max_lr / 10
        final_div_factor=100.0,  # end_lr = start_lr / 100
    )


# ---------------------------------------------------------------------------
# Training & Validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accumulation: int,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)  # (main, aux) in train mode
            loss = criterion(outputs, masks) / grad_accumulation

        scaler.scale(loss).backward()

        if (step + 1) % grad_accumulation == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Step the OneCycleLR scheduler per optimizer step
            scheduler.step()

            # Update EMA after each optimizer step
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * grad_accumulation * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def search_best_threshold(
    predictions: list[tuple[torch.Tensor, torch.Tensor]],
    metric_name: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    min_recall: float,
    min_iou: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-8, threshold_step)
    threshold_metrics: list[tuple[float, dict[str, float]]] = []

    for threshold in thresholds:
        metric_list = [compute_metrics(prob_map, mask, float(threshold)) for prob_map, mask in predictions]
        avg_metrics = average_metric_list(metric_list)
        threshold_metrics.append((float(threshold), avg_metrics))

    def rank_key(metrics: dict[str, float]) -> tuple[float, float]:
        if metric_name == "precision":
            return (metrics["precision"], metrics["iou"])
        if metric_name == "dice":
            return (metrics["dice"], metrics["precision"])
        return (metrics["iou"], metrics["precision"])

    constrained = [
        (threshold, metrics)
        for threshold, metrics in threshold_metrics
        if metrics["recall"] >= min_recall and metrics["iou"] >= min_iou
    ]
    candidates = constrained if constrained else threshold_metrics

    best_threshold, best_metrics = max(candidates, key=lambda item: rank_key(item[1]))
    return best_threshold, best_metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    stride: int,
    use_amp: bool,
    selection_metric: str,
    selection_threshold: float,
    auto_selection_threshold: bool,
    selection_threshold_min: float,
    selection_threshold_max: float,
    selection_threshold_step: float,
    selection_min_recall: float,
    selection_min_iou: float,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    predictions = []
    # Use base CrackLoss for validation (model returns single tensor in eval)
    val_criterion = criterion.main_loss if hasattr(criterion, "main_loss") else criterion

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        prob_map = sliding_window_predict(
            model=model,
            image=images,
            patch_size=patch_size,
            stride=stride,
            device=device,
            use_amp=use_amp,
            tta=False,
        )
        logits = torch.logit(prob_map.clamp(min=1e-4, max=1 - 1e-4))
        loss = val_criterion(logits, masks)
        total_loss += loss.item()
        predictions.append((prob_map.detach().cpu(), masks.detach().cpu()))

    if auto_selection_threshold:
        best_threshold, metrics = search_best_threshold(
            predictions,
            metric_name=selection_metric,
            threshold_min=selection_threshold_min,
            threshold_max=selection_threshold_max,
            threshold_step=selection_threshold_step,
            min_recall=selection_min_recall,
            min_iou=selection_min_iou,
        )
    else:
        best_threshold = float(selection_threshold)
        metric_list = [compute_metrics(prob_map, mask, best_threshold) for prob_map, mask in predictions]
        metrics = average_metric_list(metric_list)

    metrics = {**metrics, "threshold": best_threshold, "selection_score": metrics[selection_metric]}
    return total_loss / len(loader), best_threshold, metrics


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
) -> None:
    data = {
        "epoch": epoch,
        "model_name": "deeplabv3plus_resnet101",
        "patch_size": args.patch_size,
        "stride": args.stride,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "config": vars(args),
    }
    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        data["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        data["ema_state_dict"] = ema.state_dict()
    torch.save(data, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    crack500_only = args.crack500_only and not args.use_deepcrack

    train_loader, val_loader, _ = build_dataloaders(
        crack500_root=args.crack500_root,
        deepcrack_root=args.deepcrack_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crack500_only=crack500_only,
    )

    model = build_model(pretrained=True).to(device)
    criterion = build_loss_with_aux(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
        boundary_weight=args.boundary_weight,
        detail_weight=args.detail_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay)

    # steps_per_epoch = number of optimizer steps (accounting for grad accumulation)
    steps_per_epoch = len(train_loader) // args.grad_accumulation + 1
    scheduler = build_scheduler(
        optimizer, args.epochs, steps_per_epoch,
        args.warmup_epochs, args.lr,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # EMA for more stable evaluation
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 1
    best_iou = 0.0
    best_precision = 0.0
    best_dice = 0.0
    best_selection_score = 0.0
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "ema_state_dict" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        prev_metrics = ckpt.get("val_metrics", {})
        best_iou = float(prev_metrics.get("iou", 0.0))
        best_precision = float(prev_metrics.get("precision", 0.0))
        best_dice = float(prev_metrics.get("dice", 0.0))
        best_selection_score = float(prev_metrics.get("selection_score", prev_metrics.get(args.selection_metric, 0.0)))

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "device": str(device),
             "params": sum(p.numel() for p in model.parameters()),
             "model_name": "deeplabv3plus_resnet101_cbam"},
            f, indent=2, ensure_ascii=False,
        )

    log_path = run_dir / "history.jsonl"
    print(f"Device: {device}")
    print(f"CRACK500: {args.crack500_root}")
    print(f"DeepCrack: {args.deepcrack_root}")
    print(f"Run dir: {run_dir}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Backbone LR: {args.lr * args.backbone_lr_scale:.1e}, Head LR: {args.lr:.1e}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Scheduler: OneCycleLR (warmup={args.warmup_epochs} epochs)")
    print(
        f"Selection: metric={args.selection_metric}, "
        f"threshold={'auto' if args.auto_selection_threshold else f'{args.selection_threshold:.2f}'}"
    )
    if args.auto_selection_threshold:
        print(
            f"Threshold constraints: recall>={args.selection_min_recall:.2f}, "
            f"iou>={args.selection_min_iou:.2f}, range=[{args.selection_threshold_min:.2f}, {args.selection_threshold_max:.2f}]"
        )
    print(f"Loss bias: tversky(alpha={args.tversky_alpha}, beta={args.tversky_beta}, gamma={args.tversky_gamma})")
    print(f"\n{'Epoch':>5} | {'TrainLoss':>10} | {'ValLoss':>8} | {'Thr':>5} | {'Prec':>6} | {'Recall':>6} | {'IoU':>6} | {'Dice':>6} | {'LR':>9}")
    print("-" * 94)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Resample augmented dataset each epoch
        for ds in _iter_sampled_datasets(train_loader.dataset):
            ds.resample()

        train_loss = train_one_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            device=device, use_amp=use_amp,
            grad_accumulation=args.grad_accumulation, ema=ema,
        )

        # Validate with EMA model if available, otherwise use regular model
        eval_model = ema.shadow if ema is not None else model
        val_loss, best_threshold, val_metrics = validate(
            model=eval_model, loader=val_loader, criterion=criterion,
            device=device, patch_size=args.patch_size,
            stride=args.stride, use_amp=use_amp,
            selection_metric=args.selection_metric,
            selection_threshold=args.selection_threshold,
            auto_selection_threshold=args.auto_selection_threshold,
            selection_threshold_min=args.selection_threshold_min,
            selection_threshold_max=args.selection_threshold_max,
            selection_threshold_step=args.selection_threshold_step,
            selection_min_recall=args.selection_min_recall,
            selection_min_iou=args.selection_min_iou,
        )

        lr = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0
        ema_tag = " (EMA)" if ema is not None else ""
        print(
            f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {best_threshold:>5.2f} | "
            f"{val_metrics['precision']:>6.4f} | {val_metrics['recall']:>6.4f} | "
            f"{val_metrics['iou']:>6.4f} | {val_metrics['dice']:>6.4f} | {lr:>9.2e} ({elapsed:.0f}s){ema_tag}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "lr": lr, **val_metrics,
            }, ensure_ascii=False) + "\n")

        # Save last checkpoint (use the training model, not EMA, for resume)
        save_checkpoint(weights_dir / "last.pt", epoch, model, args, val_metrics,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler, ema=ema)

        # Save best checkpoints (use EMA model weights for best checkpoints)
        save_model = ema.shadow if ema is not None else model
        selection_score = float(val_metrics[args.selection_metric])
        improved = False
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoint(weights_dir / "best.pt", epoch, save_model, args, val_metrics)
            print(f"        -> best {args.selection_metric}: {best_selection_score:.4f} @ thr={best_threshold:.2f}")
            improved = True
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(weights_dir / "best_iou.pt", epoch, save_model, args, val_metrics)
            print(f"        -> best IoU: {best_iou:.4f}")
        if val_metrics["precision"] > best_precision:
            best_precision = val_metrics["precision"]
            save_checkpoint(weights_dir / "best_precision.pt", epoch, save_model, args, val_metrics)
            print(f"        -> best Precision: {best_precision:.4f}")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(weights_dir / "best_dice.pt", epoch, save_model, args, val_metrics)
            print(f"        -> best Dice: {best_dice:.4f}")

        patience_counter = 0 if improved else patience_counter + 1
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\nTraining complete.")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Weights: {weights_dir}")


def _iter_sampled_datasets(dataset):
    """Recursively find SampledDataset instances in ConcatDataset."""
    from torch.utils.data import ConcatDataset
    if isinstance(dataset, SampledDataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            yield from _iter_sampled_datasets(ds)


if __name__ == "__main__":
    main()
