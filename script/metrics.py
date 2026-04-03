"""
Binary segmentation metrics for CrackNet.
"""
from __future__ import annotations

import numpy as np
import torch


def _ensure_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x)).float()


def _to_prob(pred: torch.Tensor) -> torch.Tensor:
    if pred.min() < 0.0 or pred.max() > 1.0:
        return torch.sigmoid(pred)
    return pred


def compute_metrics(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_t = _to_prob(_ensure_tensor(pred))
    target_t = _ensure_tensor(target)
    pred_bin = (pred_t > threshold).float()
    tp = (pred_bin * target_t).sum()
    fp = (pred_bin * (1.0 - target_t)).sum()
    fn = ((1.0 - pred_bin) * target_t).sum()
    return {
        "precision": ((tp + 1e-6) / (tp + fp + 1e-6)).item(),
        "recall": ((tp + 1e-6) / (tp + fn + 1e-6)).item(),
        "iou": ((tp + 1e-6) / (tp + fp + fn + 1e-6)).item(),
        "dice": ((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)).item(),
    }


def average_metric_list(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0, "dice": 0.0}
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}
