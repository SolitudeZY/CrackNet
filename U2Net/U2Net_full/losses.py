"""
损失函数集合: Lovász/BCE/Boundary 多尺度损失。

Lovász-Hinge (Berman et al., CVPR 2018):
  "The Lovász-Softmax loss: A tractable surrogate for the optimization
   of the intersection-over-union measure in neural networks"

  核心思想: IoU 是离散集合函数, 不可微。Lovász extension 将其松弛为
  连续凸函数, 使得可以直接优化 IoU 而非像素级 BCE。
  对于裂缝这种极度不均衡场景 (正像素 <5%), BCE 容易退化为预测全背景,
  而 Lovász-Hinge 直接优化 IoU 可以有效缓解。

参考实现: https://github.com/bermanmaxim/LovaszSoftmax
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_binary_target(target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Binary label smoothing: 0 -> eps/2, 1 -> 1-eps/2."""
    if smoothing <= 0.0:
        return target
    smoothing = min(max(float(smoothing), 0.0), 0.5)
    return target * (1.0 - smoothing) + 0.5 * smoothing


# ==================== Lovász-Hinge Loss ====================

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    计算 Lovász extension 的梯度 (即 Jaccard loss 的次梯度)。

    Args:
        gt_sorted: 按预测误差降序排列的 ground truth, shape [P]
    Returns:
        Lovász 梯度, shape [P]
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor):
    """展平 logits 和 labels 为 1D。"""
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovász-Hinge loss (展平版本)。

    Args:
        logits: [P] 预测 logits (未经 sigmoid)
        labels: [P] 二值标签 {0, 1}
    Returns:
        Lovász-Hinge loss 标量
    """
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0  # {0,1} → {-1,+1}
    errors = 1.0 - logits * signs  # hinge: 1 - margin
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(
    logits: torch.Tensor,
    labels: torch.Tensor,
    per_image: bool = False,
) -> torch.Tensor:
    """
    Binary Lovász-Hinge loss。

    Args:
        logits: [B, 1, H, W] 预测 logits
        labels: [B, 1, H, W] 二值标签 {0, 1}
        per_image: True=逐图像计算再取平均, False=全 batch 展平计算
    Returns:
        loss 标量
    """
    if per_image:
        losses = []
        for logit, label in zip(logits, labels):
            l_flat, lb_flat = _flatten_binary_scores(logit, label)
            losses.append(lovasz_hinge_flat(l_flat, lb_flat))
        return torch.stack(losses).mean()
    else:
        l_flat, lb_flat = _flatten_binary_scores(logits, labels)
        return lovasz_hinge_flat(l_flat, lb_flat)


# ==================== Dice Loss ====================

def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """标准 Dice Loss (从 logits 计算)。"""
    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = (probs * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """边界聚焦 BCE loss。"""
    pad = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    if boundary.sum() < 1:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits * boundary, target * boundary, reduction="sum")
    return loss / boundary.sum().clamp(min=1.0)


# ==================== 组合损失 ====================

class U2NetLovaszLoss(nn.Module):
    """
    U2Net 多尺度 Lovász-Hinge + Dice 损失。

    对 7 个输出 (d0-d6) 分别计算:
      per_output = lovasz_hinge + dice_loss
      total = 2 * per_output_d0 + Σ per_output_di (i=1..6)

    相比 BCE + Dice:
      - Lovász-Hinge 直接优化 IoU → 对极度不均衡 (裂缝 <5% 像素) 效果更好
      - Dice 提供平滑的全局重叠度量，与 Lovász 互补
      - 移除了 BCE → 避免被大量背景像素主导梯度
    """

    def __init__(self, per_image: bool = True, label_smoothing: float = 0.0):
        super().__init__()
        self.per_image = per_image
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device)
        smoothed_target = smooth_binary_target(target, self.label_smoothing)

        for i, logit in enumerate(outputs):
            lh = lovasz_hinge(logit, target, per_image=self.per_image)
            dl = dice_loss(logit, smoothed_target)
            combined = lh + dl
            weight = 2.0 if i == 0 else 1.0
            total = total + weight * combined
            loss_dict[f"d{i}_lovasz"] = lh.item()
            loss_dict[f"d{i}_dice"] = dl.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


class U2NetBCEDiceLoss(nn.Module):
    """
    U2Net 多尺度 BCE + Dice 损失 (原版, 用于对比)。

    total = 2*(BCE_d0 + Dice_d0) + Σ(BCE_di + Dice_di) for i=1..6
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device)
        smoothed_target = smooth_binary_target(target, self.label_smoothing)

        for i, logit in enumerate(outputs):
            bce = F.binary_cross_entropy_with_logits(logit, smoothed_target)
            dl = dice_loss(logit, smoothed_target)
            combined = bce + dl
            weight = 2.0 if i == 0 else 1.0
            total = total + weight * combined
            loss_dict[f"d{i}"] = combined.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


class U2NetBoundaryLoss(nn.Module):
    """
    U2Net 多尺度 Boundary-Enhanced Loss。

    纯边界损失容易退化成“大面积前景”预测，因此这里采用:
      per_output = BCE + Dice + boundary_weight * Boundary
      total = 2*per_output(d0) + Σper_output(di), i=1..6

    这样既保留区域监督，又用边界项强调细裂缝轮廓。
    """

    def __init__(
        self,
        kernel_size: int = 3,
        label_smoothing: float = 0.0,
        boundary_weight: float = 0.25,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.label_smoothing = label_smoothing
        self.boundary_weight = boundary_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device)
        smoothed_target = smooth_binary_target(target, self.label_smoothing)

        for i, logit in enumerate(outputs):
            bce = F.binary_cross_entropy_with_logits(logit, smoothed_target)
            dl = dice_loss(logit, smoothed_target)
            bl = boundary_loss(logit, smoothed_target, kernel_size=self.kernel_size)
            combined = self.bce_weight * bce + self.dice_weight * dl + self.boundary_weight * bl
            weight = 2.0 if i == 0 else 1.0
            total = total + weight * combined
            loss_dict[f"d{i}_bce"] = bce.item()
            loss_dict[f"d{i}_dice"] = dl.item()
            loss_dict[f"d{i}_boundary"] = bl.item()
            loss_dict[f"d{i}_combined"] = combined.item()

        loss_dict["total"] = total.item()
        return total, loss_dict


if __name__ == "__main__":
    # 模拟极度不均衡场景: 裂缝像素仅 2%
    B, H, W = 4, 320, 320
    logits_tuple = tuple(torch.randn(B, 1, H, W) for _ in range(7))

    # 稀疏标签: ~2% 正像素
    target = (torch.rand(B, 1, H, W) > 0.98).float()
    pos_ratio = target.mean().item() * 100
    print(f"正像素比例: {pos_ratio:.1f}%")

    # BCE + Dice
    bce_dice = U2NetBCEDiceLoss()
    loss_bd, dict_bd = bce_dice(logits_tuple, target)
    print(f"\nBCE + Dice:    total={loss_bd.item():.4f}")

    # Lovász + Dice
    lovasz = U2NetLovaszLoss(per_image=True)
    loss_lv, dict_lv = lovasz(logits_tuple, target)
    print(f"Lovász + Dice:  total={loss_lv.item():.4f}")
