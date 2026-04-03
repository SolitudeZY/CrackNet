"""
U-Net with EfficientNet-B0 encoder for binary crack segmentation.
Output: (B, 1, 416, 416) logits (no sigmoid — applied in loss).
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DecoderBlock(nn.Module):
    """上采样 + concat skip + 2x(Conv3x3-BN-ReLU)"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CrackUNet(nn.Module):
    """
    EfficientNet-B0 encoder + 4-stage U-Net decoder.

    输入 416x416 时各层特征图:
        features[1]: 16ch,  208x208  (skip s1)
        features[2]: 24ch,  104x104  (skip s2)
        features[3]: 40ch,   52x52   (skip s3)
        features[5]: 112ch,  26x26   (skip s4)
        features[7]: 320ch,  13x13   (bottleneck)
    """

    # 提取 skip connection 的 encoder 层索引
    SKIP_INDICES = [1, 2, 3, 5]
    BOTTLENECK_IDX = 7

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        # 只保留 features[0..7]，不需要 features[8] (1280ch expand)
        self.encoder = nn.ModuleList(list(backbone.features[:self.BOTTLENECK_IDX + 1]))

        # Decoder: bottleneck(320) -> 26 -> 52 -> 104 -> 208
        #            skip channels:    112   40    24    16
        self.d4 = DecoderBlock(in_ch=320, skip_ch=112, out_ch=128)  # 13 -> 26
        self.d3 = DecoderBlock(in_ch=128, skip_ch=40,  out_ch=64)   # 26 -> 52
        self.d2 = DecoderBlock(in_ch=64,  skip_ch=24,  out_ch=32)   # 52 -> 104
        self.d1 = DecoderBlock(in_ch=32,  skip_ch=16,  out_ch=16)   # 104 -> 208

        # Head: 208 -> 416, 输出单通道 logits
        self.head = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def encoder_params(self):
        """返回 encoder 参数（用于差分学习率）。"""
        return self.encoder.parameters()

    def decoder_params(self):
        """返回 decoder + head 参数。"""
        from itertools import chain
        return chain(
            self.d4.parameters(), self.d3.parameters(),
            self.d2.parameters(), self.d1.parameters(),
            self.head.parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward，收集 skip connections
        skips = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.SKIP_INDICES:
                skips[i] = x

        # x 现在是 bottleneck: (B, 320, 13, 13)
        # Decoder
        x = self.d4(x, skips[5])   # -> (B, 128, 26, 26)
        x = self.d3(x, skips[3])   # -> (B, 64, 52, 52)
        x = self.d2(x, skips[2])   # -> (B, 32, 104, 104)
        x = self.d1(x, skips[1])   # -> (B, 16, 208, 208)

        return self.head(x)         # -> (B, 1, 416, 416)


if __name__ == "__main__":
    model = CrackUNet(pretrained=True)
    x = torch.randn(2, 3, 416, 416)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    enc = sum(p.numel() for p in model.encoder_params())
    dec = sum(p.numel() for p in model.decoder_params())
    print(f"Total params:   {total:,}")
    print(f"Encoder params: {enc:,}")
    print(f"Decoder params: {dec:,}")
