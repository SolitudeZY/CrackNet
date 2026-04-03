"""
损失函数集合: Lovász-Hinge + Dice 多尺度损失。

Lovász-Hinge (Berman et al., CVPR 2018):
  "The Lovász-Softmax loss: A tractable surrogate for the optimization
   of the intersection-over-union measure in neural networks"

  核心思想: IoU 是离散集合函数, 不可微。Lovász extension 将其松弛为
  连续凸函数, 使得可以直接优化 IoU 而非像素级 BCE。
  对于裂缝这种极度不均衡场景 (正像素 <5%), BCE 容易退化为预测全背景,
  而 Lovász-Hinge 直接优化 IoU 可以有效缓解。

注意: Lovász 内部涉及排序+累积求和+除法, 在 AMP fp16 下数值不稳定,
      必须强制在 fp32 下计算。本实现已处理此问题。

参考实现: https://github.com/bermanmaxim/LovaszSoftmax
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    if gts == 0.0:
        return gt_sorted * 0.0

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)

    # 避免除0
    jaccard = 1.0 - intersection / (union + 1e-6)

    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1].clone()
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovász-Hinge loss (展平版本)。

    Args:
        logits: [P] 预测 logits (未经 sigmoid), fp32
        labels: [P] 二值标签 {0, 1}, fp32
    Returns:
        Lovász-Hinge loss 标量
    """
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels - 1.0  # {0,1} → {-1,+1}
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
    Binary Lovász-Hinge loss (AMP 安全: 强制 fp32)。

    Args:
        logits: [B, 1, H, W] 预测 logits
        labels: [B, 1, H, W] 二值标签 {0, 1}
        per_image: True=逐图像计算再取平均, False=全 batch 展平计算
    """
    # 强制 fp32 — Lovász 的排序/累积求和在 fp16 下会溢出
    logits = logits.float()
    labels = labels.float()

    if per_image:
        losses = []
        for logit, label in zip(logits, labels):
            l_flat = logit.view(-1)
            lb_flat = label.view(-1)
            losses.append(lovasz_hinge_flat(l_flat, lb_flat))
        return torch.stack(losses).mean()
    else:
        return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


# ==================== Dice Loss ====================

def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """标准 Dice Loss (AMP 安全: 强制 fp32)。"""
    logits = logits.float()
    target = target.float()
    probs = torch.sigmoid(logits)
    smooth = 1.0 # 使用较大的平滑因子，类似 1.0 而不是 1e-5，可以提供更平滑的梯度
    intersection = (probs * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)


# ==================== 组合损失 ====================

class U2NetLovaszLoss(nn.Module):
    """
    U2Net 多尺度 Lovász-Hinge + Dice 损失。

    对 7 个输出 (d0-d6) 分别计算:
      per_output = lovasz_hinge + dice_loss
      total = 2 * per_output_d0 + Σ per_output_di (i=1..6)

    内部强制 fp32 计算, 与 AMP 兼容。
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device, dtype=torch.float32)

        for i, logit in enumerate(outputs):
            # 强制fp32
            logit_f32 = logit.float()
            target_f32 = target.float()

            # 由于部分批次可能出现不稳定的梯度，增加 nan 检查
            lh = lovasz_hinge(logit_f32, target_f32, per_image=self.per_image)
            dl = dice_loss(logit_f32, target_f32)

            # 使用加权和组合损失
            combined = lh + dl
            if torch.isnan(combined) or torch.isinf(combined):
                combined = torch.tensor(0.0, device=target.device, dtype=torch.float32, requires_grad=True)

            weight = 2.0 if i == 0 else 1.0

            total = total + weight * combined
            loss_dict[f"d{i}_lovasz"] = lh.item() if not torch.isnan(lh) and not torch.isinf(lh) else 0.0
            loss_dict[f"d{i}_dice"] = dl.item() if not torch.isnan(dl) and not torch.isinf(dl) else 0.0

        loss_dict["total"] = total.item()
        return total, loss_dict


class U2NetBCEDiceLoss(nn.Module):
    """
    U2Net 多尺度 BCE + Dice 损失 (用于对比)。

    total = 2*(BCE_d0 + Dice_d0) + Σ(BCE_di + Dice_di) for i=1..6
    """

    def __init__(self):
        super().__init__()

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        outputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total = torch.tensor(0.0, device=target.device, dtype=torch.float32)

        for i, logit in enumerate(outputs):
            logit_f32 = logit.float()
            target_f32 = target.float()

            bce = F.binary_cross_entropy_with_logits(logit_f32, target_f32)
            dl = dice_loss(logit_f32, target_f32)
            combined = bce + dl

            if torch.isnan(combined) or torch.isinf(combined):
                combined = torch.tensor(0.0, device=target.device, dtype=torch.float32, requires_grad=True)

            weight = 2.0 if i == 0 else 1.0
            total = total + weight * combined
            loss_dict[f"d{i}"] = combined.item() if not torch.isnan(combined) and not torch.isinf(combined) else 0.0

        loss_dict["total"] = total.item()
        return total, loss_dict


if __name__ == "__main__":
    # 模拟 AMP + 极度不均衡
    B, H, W = 4, 320, 320
    target = (torch.rand(B, 1, H, W) > 0.98).float().cuda()
    print(f"正像素比例: {target.mean().item() * 100:.1f}%")

    logits_tuple = tuple(torch.randn(B, 1, H, W).cuda() for _ in range(7))

    # 在 AMP 下测试 Lovász
    lovasz = U2NetLovaszLoss(per_image=True).cuda()
    with torch.amp.autocast("cuda"):
        loss, d = lovasz(logits_tuple, target)
    print(f"Lovász+Dice (AMP): {loss.item():.4f}, dtype={loss.dtype}")
    assert not torch.isnan(loss), "Lovász loss is NaN under AMP!"
    assert loss.dtype == torch.float32, "Lovász should output fp32"
    print("AMP 兼容性测试通过")
