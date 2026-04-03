"""
U-Net with MobileNetV2 encoder for binary crack segmentation.
Designed for Edge Devices (MobileNetV2 is highly optimized for mobile/edge).
Supports arbitrary input sizes (must be multiples of 32, e.g., 416x416, 640x640).
Output: (B, 1, H, W) logits (no sigmoid — applied in loss).
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) 机制：
    使用高层特征(g)指导底层特征(x)的过滤，抑制背景噪声，使模型聚焦于裂缝区域，显著提升掩码边缘精细度。
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 尺寸对齐
        if g1.shape[2:] != x1.shape[2:]:
            import torch.nn.functional as F
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class DecoderBlock(nn.Module):
    """Upsample + Attention Gate + concat skip + 2x(Conv3x3-BN-ReLU)"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # 引入 Attention Gate，F_int 为中间通道数，取输出通道的一半
        self.ag = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 16))
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle cases where input size is not perfectly divisible by 32
        if x.shape[2:] != skip.shape[2:]:
            import torch.nn.functional as F
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # 注意力机制过滤 skip connection
        skip = self.ag(g=x, x=skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class U_Net_MobileNetV2(nn.Module):
    """
    MobileNetV2 encoder + 4-stage U-Net decoder.
    Very fast, lightweight, and suitable for edge deployment.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features
        
        # MobileNetV2 feature extraction indices and channels
        # Layer 1: output is 1/2 resolution, 16 channels
        # Layer 3: output is 1/4 resolution, 24 channels
        # Layer 6: output is 1/8 resolution, 32 channels
        # Layer 13: output is 1/16 resolution, 96 channels
        # Layer 18: output is 1/32 resolution, 320 channels (bottleneck)
        
        self.enc1 = backbone[0:2]   # skip1: 16 ch
        self.enc2 = backbone[2:4]   # skip2: 24 ch
        self.enc3 = backbone[4:7]   # skip3: 32 ch
        self.enc4 = backbone[7:14]  # skip4: 96 ch
        self.enc5 = backbone[14:19] # bottleneck: 1280 ch (Layer 18 expands to 1280)

        # Decoder
        # bottleneck (1280) -> upsample + skip4 (96)
        self.d4 = DecoderBlock(in_ch=1280, skip_ch=96, out_ch=256)
        self.d3 = DecoderBlock(in_ch=256, skip_ch=32, out_ch=128)
        self.d2 = DecoderBlock(in_ch=128, skip_ch=24, out_ch=64)
        self.d1 = DecoderBlock(in_ch=64, skip_ch=16, out_ch=32)

        # Head (Upsample back to original resolution)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def encoder_params(self):
        """Returns encoder parameters for differential learning rates."""
        from itertools import chain
        return chain(
            self.enc1.parameters(), self.enc2.parameters(),
            self.enc3.parameters(), self.enc4.parameters(),
            self.enc5.parameters()
        )

    def decoder_params(self):
        """Returns decoder + head parameters."""
        from itertools import chain
        return chain(
            self.d4.parameters(), self.d3.parameters(),
            self.d2.parameters(), self.d1.parameters(),
            self.head.parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.enc5(s4)

        # Decoder
        d4 = self.d4(b, s4)
        d3 = self.d3(d4, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)

        return self.head(d1)

if __name__ == "__main__":
    # Test with 640x640
    model = U_Net_MobileNetV2(pretrained=False)
    dummy_input = torch.randn(2, 3, 640, 640)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
