"""
U2Net 裂缝分割模型定义。

提供:
  - U2NET:   全量版 U²-Net
  - U2NETP:  轻量版 U²-Net
  - build_model: 按名称构建模型

说明:
  - forward 返回 7 个 logits 输出: (d0, d1, d2, d3, d4, d5, d6)
  - 训练时请配合 BCEWithLogits/Lovasz 等基于 logits 的损失
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    """ReLU-BN-Conv 基础块。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return self.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class RSU7(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class U2NET(nn.Module):
    """全量版 U²-Net。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.cbam1 = CBAM(64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.cbam2 = CBAM(128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.cbam3 = CBAM(256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.cbam4 = CBAM(512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.cbam5 = CBAM(512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.cbam6 = CBAM(512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.cbam5d = CBAM(512)
        self.stage4d = RSU4(1024, 128, 256)
        self.cbam4d = CBAM(256)
        self.stage3d = RSU5(512, 64, 128)
        self.cbam3d = CBAM(128)
        self.stage2d = RSU6(256, 32, 64)
        self.cbam2d = CBAM(64)
        self.stage1d = RSU7(128, 16, 64)
        self.cbam1d = CBAM(64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx1 = self.cbam1(self.stage1(x))
        hx = self.pool12(hx1)
        hx2 = self.cbam2(self.stage2(hx))
        hx = self.pool23(hx2)
        hx3 = self.cbam3(self.stage3(hx))
        hx = self.pool34(hx3)
        hx4 = self.cbam4(self.stage4(hx))
        hx = self.pool45(hx4)
        hx5 = self.cbam5(self.stage5(hx))
        hx = self.pool56(hx5)
        hx6 = self.cbam6(self.stage6(hx))
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.cbam5d(self.stage5d(torch.cat((hx6up, hx5), 1)))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.cbam4d(self.stage4d(torch.cat((hx5dup, hx4), 1)))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.cbam3d(self.stage3d(torch.cat((hx4dup, hx3), 1)))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.cbam2d(self.stage2d(torch.cat((hx3dup, hx2), 1)))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.cbam1d(self.stage1d(torch.cat((hx2dup, hx1), 1)))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NETP(nn.Module):
    """轻量版 U²-Net。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.cbam1 = CBAM(64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 16, 64)
        self.cbam2 = CBAM(64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(64, 16, 64)
        self.cbam3 = CBAM(64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(64, 16, 64)
        self.cbam4 = CBAM(64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(64, 16, 64)
        self.cbam5 = CBAM(64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(64, 16, 64)
        self.cbam6 = CBAM(64)

        self.stage5d = RSU4F(128, 16, 64)
        self.cbam5d = CBAM(64)
        self.stage4d = RSU4(128, 16, 64)
        self.cbam4d = CBAM(64)
        self.stage3d = RSU5(128, 16, 64)
        self.cbam3d = CBAM(64)
        self.stage2d = RSU6(128, 16, 64)
        self.cbam2d = CBAM(64)
        self.stage1d = RSU7(128, 16, 64)
        self.cbam1d = CBAM(64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx1 = self.cbam1(self.stage1(x))
        hx = self.pool12(hx1)
        hx2 = self.cbam2(self.stage2(hx))
        hx = self.pool23(hx2)
        hx3 = self.cbam3(self.stage3(hx))
        hx = self.pool34(hx3)
        hx4 = self.cbam4(self.stage4(hx))
        hx = self.pool45(hx4)
        hx5 = self.cbam5(self.stage5(hx))
        hx = self.pool56(hx5)
        hx6 = self.cbam6(self.stage6(hx))
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.cbam5d(self.stage5d(torch.cat((hx6up, hx5), 1)))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.cbam4d(self.stage4d(torch.cat((hx5dup, hx4), 1)))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.cbam3d(self.stage3d(torch.cat((hx4dup, hx3), 1)))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.cbam2d(self.stage2d(torch.cat((hx3dup, hx2), 1)))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.cbam1d(self.stage1d(torch.cat((hx2dup, hx1), 1)))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NetForExport(nn.Module):
    """导出包装，只输出 d0 的 sigmoid 概率图。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x)[0])


def build_model(name: str = "u2net", in_ch: int = 3, out_ch: int = 1) -> nn.Module:
    model_name = name.lower()
    if model_name == "u2net":
        return U2NET(in_ch, out_ch)
    if model_name in {"u2netp", "u2net_p"}:
        return U2NETP(in_ch, out_ch)
    raise ValueError(f"Unsupported model name: {name}")


if __name__ == "__main__":
    for model_name in ("u2net", "u2netp"):
        model = build_model(model_name)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name}: {n_params:,} params, {n_params * 4 / 1024 / 1024:.1f} MB")
