"""
Sliding-window inference helpers for DeepLab.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def make_gaussian_weight(patch_size: int, device: torch.device, sigma: float | None = None) -> torch.Tensor:
    if sigma is None:
        sigma = patch_size / 6.0
    ax = torch.arange(patch_size, dtype=torch.float32, device=device) - (patch_size - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    return (g1.unsqueeze(0) * g1.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _pad_image(image: torch.Tensor, patch_size: int, stride: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = image.shape
    if h > patch_size:
        pad_h = (math.ceil((h - patch_size) / stride) * stride + patch_size) - h
    else:
        pad_h = patch_size - h
    if w > patch_size:
        pad_w = (math.ceil((w - patch_size) / stride) * stride + patch_size) - w
    else:
        pad_w = patch_size - w

    if pad_h > 0 or pad_w > 0:
        mode = "reflect" if pad_h < h and pad_w < w else "constant"
        image = F.pad(image, [0, pad_w, 0, pad_h], mode=mode, value=0)
    return image, h, w


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tta: bool = False,
) -> torch.Tensor:
    model.eval()
    image, h, w = _pad_image(image, patch_size, stride)
    _, _, hp, wp = image.shape

    weight = make_gaussian_weight(patch_size, device=device)
    pred_sum = torch.zeros(1, 1, hp, wp, device=device)
    weight_sum = torch.zeros(1, 1, hp, wp, device=device)

    for y in range(0, hp - patch_size + 1, stride):
        for x in range(0, wp - patch_size + 1, stride):
            patch = image[:, :, y : y + patch_size, x : x + patch_size].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(patch)
                if tta:
                    patch_h = torch.flip(patch, dims=[3])
                    patch_v = torch.flip(patch, dims=[2])
                    logits_h = torch.flip(model(patch_h), dims=[3])
                    logits_v = torch.flip(model(patch_v), dims=[2])
                    logits = (logits + logits_h + logits_v) / 3.0
            pred_sum[:, :, y : y + patch_size, x : x + patch_size] += logits.float() * weight
            weight_sum[:, :, y : y + patch_size, x : x + patch_size] += weight

    logits = pred_sum / weight_sum.clamp(min=1e-8)
    return torch.sigmoid(logits[:, :, :h, :w])
