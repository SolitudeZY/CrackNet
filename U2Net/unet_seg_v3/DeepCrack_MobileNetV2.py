"""
DeepCrack-style Multi-Scale Fusion U-Net with MobileNetV2 Encoder.

核心思想 (来自 DeepCrack, Zou et al. TIP 2019):
  编码器和解码器在每个尺度上进行特征拼接 (Pairwise Fusion)，
  并产生独立的侧输出 (Side Output)。所有侧输出上采样到全分辨率后融合为最终预测。
  训练时对所有 6 个输出 (5 侧 + 1 融合) 施加 BCE 监督。

相比原版 DeepCrack 的改进:
  - 用 MobileNetV2 替代 VGG-16/SegNet 编码器，适配边缘部署
  - 保留 CBAM/AG/ECA 注意力机制
  - 解码器使用 ConvTranspose2d 上采样 (而非 MaxUnpool)

输出: (B, 1, H, W) logits (无 sigmoid)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ================= Attention Modules (复用 v2) =================

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_planes // ratio, 1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


# ================= Decoder Block =================

class DecoderBlock(nn.Module):
    """ConvTranspose2d 上采样 + Attention + Concat Skip + 双卷积"""
    def __init__(self, in_ch, skip_ch, out_ch, attention_type='CBAM'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attention_type = attention_type

        if attention_type == 'AG':
            self.attention = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 16))
        elif attention_type == 'CBAM':
            self.attention = CBAM(skip_ch)
        elif attention_type == 'ECA':
            self.attention = ECA(skip_ch)
        else:
            self.attention = None

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.attention_type == 'AG':
            skip = self.attention(g=x, x=skip)
        elif self.attention_type in ('CBAM', 'ECA'):
            skip = self.attention(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ================= Side Output Fusion Module (DeepCrack 核心) =================

class SideOutputFusion(nn.Module):
    """
    将编码器特征和解码器特征在同一尺度拼接，通过卷积压缩为单通道侧输出。
    DeepCrack 论文 Eq.(1): S_k = σ(W_k * [E_k; D_k])
    """
    def __init__(self, enc_ch: int, dec_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(enc_ch + dec_ch, (enc_ch + dec_ch) // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d((enc_ch + dec_ch) // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d((enc_ch + dec_ch) // 4, 1, 1),
        )

    def forward(self, enc_feat, dec_feat):
        if enc_feat.shape[2:] != dec_feat.shape[2:]:
            dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([enc_feat, dec_feat], dim=1)
        return self.conv(fused)


class BottleneckSideOutput(nn.Module):
    """Bottleneck (最深层) 的侧输出，只有编码器特征。"""
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 8, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


# ================= DeepCrack MobileNetV2 =================

class DeepCrack_MobileNetV2(nn.Module):
    """
    MobileNetV2 Encoder + U-Net Decoder + DeepCrack Multi-Scale Side Output Fusion.

    架构:
      Encoder (MobileNetV2 预训练):
        s1 (16ch, /2) → s2 (24ch, /4) → s3 (32ch, /8) → s4 (96ch, /16) → bottleneck (1280ch, /32)

      Decoder (带注意力机制):
        d4 (256ch, /16) → d3 (128ch, /8) → d2 (64ch, /4) → d1 (32ch, /2)

      DeepCrack 侧输出融合 (5 个尺度):
        side5: bottleneck → 1ch → upsample to full
        side4: concat(s4, d4) → 1ch → upsample to full
        side3: concat(s3, d3) → 1ch → upsample to full
        side2: concat(s2, d2) → 1ch → upsample to full
        side1: concat(s1, d1) → 1ch → upsample to full

      最终融合:
        fused = conv(cat(side1, side2, side3, side4, side5)) → 1ch

    训练输出: (fused, side1, side2, side3, side4, side5) — 6 个 logits
    推理输出: fused — 1 个 logits
    """

    def __init__(self, pretrained=True, attention_type='CBAM'):
        super().__init__()

        # ---- Encoder: MobileNetV2 ----
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features

        self.enc1 = backbone[0:2]    # → 16ch, /2
        self.enc2 = backbone[2:4]    # → 24ch, /4
        self.enc3 = backbone[4:7]    # → 32ch, /8
        self.enc4 = backbone[7:14]   # → 96ch, /16
        self.enc5 = backbone[14:19]  # → 1280ch, /32

        # ---- Decoder ----
        self.d4 = DecoderBlock(1280, 96, 256, attention_type)
        self.d3 = DecoderBlock(256, 32, 128, attention_type)
        self.d2 = DecoderBlock(128, 24, 64, attention_type)
        self.d1 = DecoderBlock(64, 16, 32, attention_type)

        # ---- DeepCrack 侧输出融合模块 ----
        self.side5 = BottleneckSideOutput(1280)            # bottleneck only
        self.side4 = SideOutputFusion(96, 256)             # s4 + d4
        self.side3 = SideOutputFusion(32, 128)             # s3 + d3
        self.side2 = SideOutputFusion(24, 64)              # s2 + d2
        self.side1 = SideOutputFusion(16, 32)              # s1 + d1

        # ---- 最终融合: 5 个侧输出 → 1 通道 ----
        self.final_fuse = nn.Conv2d(5, 1, kernel_size=3, padding=1)

    def encoder_params(self):
        from itertools import chain
        return chain(
            self.enc1.parameters(), self.enc2.parameters(),
            self.enc3.parameters(), self.enc4.parameters(),
            self.enc5.parameters(),
        )

    def decoder_params(self):
        from itertools import chain
        return chain(
            self.d4.parameters(), self.d3.parameters(),
            self.d2.parameters(), self.d1.parameters(),
            self.side5.parameters(), self.side4.parameters(),
            self.side3.parameters(), self.side2.parameters(),
            self.side1.parameters(),
            self.final_fuse.parameters(),
        )

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)

        # ---- Encoder ----
        s1 = self.enc1(x)    # (B, 16,  H/2,  W/2)
        s2 = self.enc2(s1)   # (B, 24,  H/4,  W/4)
        s3 = self.enc3(s2)   # (B, 32,  H/8,  W/8)
        s4 = self.enc4(s3)   # (B, 96,  H/16, W/16)
        b  = self.enc5(s4)   # (B, 1280, H/32, W/32)

        # ---- Decoder ----
        d4 = self.d4(b, s4)   # (B, 256, H/16, W/16)
        d3 = self.d3(d4, s3)  # (B, 128, H/8,  W/8)
        d2 = self.d2(d3, s2)  # (B, 64,  H/4,  W/4)
        d1 = self.d1(d2, s1)  # (B, 32,  H/2,  W/2)

        # ---- 侧输出 (每个上采样到原始分辨率) ----
        so5 = F.interpolate(self.side5(b),       size=input_size, mode='bilinear', align_corners=False)
        so4 = F.interpolate(self.side4(s4, d4),  size=input_size, mode='bilinear', align_corners=False)
        so3 = F.interpolate(self.side3(s3, d3),  size=input_size, mode='bilinear', align_corners=False)
        so2 = F.interpolate(self.side2(s2, d2),  size=input_size, mode='bilinear', align_corners=False)
        so1 = F.interpolate(self.side1(s1, d1),  size=input_size, mode='bilinear', align_corners=False)

        # ---- 最终融合 ----
        # 使用 sigmoid 前的特征进行卷积，否则融合层学不到东西
        fused = self.final_fuse(torch.cat([so1, so2, so3, so4, so5], dim=1))

        if self.training:
            return fused, so1, so2, so3, so4, so5
        return fused


if __name__ == "__main__":
    model = DeepCrack_MobileNetV2(pretrained=False, attention_type='CBAM')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
    print(f"Encoder params: {sum(p.numel() for p in model.encoder_params()):,}")
    print(f"Decoder params: {sum(p.numel() for p in model.decoder_params()):,}")

    dummy = torch.randn(2, 3, 640, 640)

    model.train()
    outs = model(dummy)
    print(f"\nTraining outputs ({len(outs)}):")
    for i, o in enumerate(outs):
        print(f"  [{i}] shape={o.shape}")

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"\nInference output: shape={out.shape}")
