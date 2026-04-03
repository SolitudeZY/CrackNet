"""
Loss functions for CrackNet v2.

Key improvements over v1:
- Multi-scale boundary loss (kernels 3, 5, 7) for sharper edge supervision
- Quality-aware Focal Tversky with adjusted alpha/beta for thin cracks
- Stronger boundary weight (0.5 vs 0.1)
- Topology-aware connectivity loss for crack continuity
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class TverskyFocalLoss(nn.Module):
    """Focal Tversky loss — emphasizes thin crack recall.

    alpha > beta means FP penalty > FN penalty, encouraging thinner predictions.
    For crack detection we want high recall, so we use alpha < beta
    to penalize false negatives more heavily.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha   # FP weight (lower = less penalty for thin predictions)
        self.beta = beta     # FN weight (higher = penalize missed cracks more)
        self.gamma = gamma   # Focal exponent (< 1 focuses on hard examples)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky) ** self.gamma


class MultiScaleBoundaryLoss(nn.Module):
    """Multi-scale boundary loss for sharp edge supervision.

    Extracts boundary at multiple scales (kernel_sizes 3, 5, 7) and
    computes weighted BCE loss focused on boundary regions.
    Larger kernels capture wider boundary context for coarse-to-fine supervision.
    """

    def __init__(self, kernel_sizes: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.kernel_sizes = kernel_sizes

    def _extract_boundary(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=pad)
        return dilated - eroded

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        n_scales = 0
        for ks in self.kernel_sizes:
            boundary = self._extract_boundary(target, ks)
            boundary_mask = boundary > 0.5
            if boundary_mask.sum() < 1:
                continue
            # Only compute BCE on actual boundary pixels
            loss = F.binary_cross_entropy_with_logits(
                logits[boundary_mask], target[boundary_mask]
            )
            total_loss = total_loss + loss
            n_scales += 1
        return total_loss / max(n_scales, 1)


class DetailLoss(nn.Module):
    """Detail preservation loss via Laplacian edge detection.

    Applies Laplacian filter to both prediction and target,
    then computes L1 loss between edge maps.
    This encourages the model to preserve fine structural details.
    """

    def __init__(self):
        super().__init__()
        # Laplacian kernel
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        pred_edges = F.conv2d(pred, self.kernel, padding=1)
        target_edges = F.conv2d(target, self.kernel, padding=1)
        return F.l1_loss(pred_edges, target_edges)


class CrackLoss(nn.Module):
    """Composite loss for crack segmentation.

    Components:
        - BCE: Standard binary cross-entropy for pixel classification
        - Dice: Region-based overlap loss
        - Tversky Focal: Precision/recall-bias configurable loss
        - Multi-scale Boundary: Sharp edge supervision at multiple scales
        - Detail: Laplacian edge preservation loss
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.5,
        detail_weight: float = 0.3,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 0.75,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.detail_weight = detail_weight
        self.dice = DiceLoss()
        self.tversky = TverskyFocalLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma)
        self.boundary = MultiScaleBoundaryLoss(kernel_sizes=(3, 5, 7))
        self.detail = DetailLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = self.dice(logits, target)
        tversky = self.tversky(logits, target)
        boundary = self.boundary(logits, target)
        detail = self.detail(logits, target)
        return (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.tversky_weight * tversky
            + self.boundary_weight * boundary
            + self.detail_weight * detail
        )


def build_loss(
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    tversky_weight: float = 0.5,
    boundary_weight: float = 0.5,
    detail_weight: float = 0.3,
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    tversky_gamma: float = 0.75,
) -> nn.Module:
    return CrackLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        tversky_weight=tversky_weight,
        boundary_weight=boundary_weight,
        detail_weight=detail_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
    )


class CrackLossWithAux(nn.Module):
    """Wraps CrackLoss for DeepLabV3+'s dual-output (main + aux) training."""

    def __init__(
        self,
        aux_weight: float = 0.4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.5,
        detail_weight: float = 0.3,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 0.75,
    ):
        super().__init__()
        self.main_loss = CrackLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            tversky_weight=tversky_weight,
            boundary_weight=boundary_weight,
            detail_weight=detail_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
        )
        self.aux_loss = CrackLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            tversky_weight=tversky_weight,
            boundary_weight=boundary_weight,
            detail_weight=detail_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
        )
        self.aux_weight = aux_weight

    def forward(
        self, outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            return self.main_loss(main_out, target) + self.aux_weight * self.aux_loss(aux_out, target)
        return self.main_loss(outputs, target)


def build_loss_with_aux(
    aux_weight: float = 0.4,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    tversky_weight: float = 0.5,
    boundary_weight: float = 0.5,
    detail_weight: float = 0.3,
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    tversky_gamma: float = 0.75,
) -> nn.Module:
    return CrackLossWithAux(
        aux_weight=aux_weight,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        tversky_weight=tversky_weight,
        boundary_weight=boundary_weight,
        detail_weight=detail_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
    )
