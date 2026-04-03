"""
U-Net with MobileNetV2 encoder for binary crack segmentation.
Designed for Edge Devices (MobileNetV2 is highly optimized for mobile/edge).
Supports arbitrary input sizes (must be multiples of 32, e.g., 416x416, 640x640).
Output: (B, 1, H, W) logits (no sigmoid — applied in loss).
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import math

# ================= Attention Modules =================

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) 机制：
    使用高层特征(g)指导底层特征(x)的过滤，抑制背景噪声。
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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 为了兼容极小的通道数，确保隐层至少有1个通道
        hidden_planes = max(in_planes // ratio, 1)
        self.fc1   = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module):
    结合了通道注意力（关注"是什么"）和空间注意力（关注"在哪里"）。
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class ECA(nn.Module):
    """
    ECA (Efficient Channel Attention):
    一种极轻量级的通道注意力机制，通过一维卷积实现局部跨通道交互。
    """
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ================= Decoder & Model =================

class DecoderBlock(nn.Module):
    """Upsample + Attention + concat skip + 2x(Conv3x3-BN-ReLU)"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention_type: str = 'AG'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attention_type = attention_type
        
        # 初始化 Attention 模块
        if attention_type == 'AG':
            self.attention = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 16))
        elif attention_type == 'CBAM':
            self.attention = CBAM(skip_ch)
        elif attention_type == 'ECA':
            self.attention = ECA(skip_ch)
        else:
            self.attention = None # 无注意力
            
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
        if self.attention_type == 'AG':
            skip = self.attention(g=x, x=skip)
        elif self.attention_type in ['CBAM', 'ECA']:
            skip = self.attention(skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class U_Net_MobileNetV2(nn.Module):
    """
    MobileNetV2 encoder + 4-stage U-Net decoder.
    支持动态切换 Attention 类型 ('AG', 'CBAM', 'ECA', 'None')
    支持深度监督 (Deep Supervision)
    """
    def __init__(self, pretrained: bool = True, attention_type: str = 'AG', deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights).features
        
        self.enc1 = backbone[0:2]   # skip1: 16 ch
        self.enc2 = backbone[2:4]   # skip2: 24 ch
        self.enc3 = backbone[4:7]   # skip3: 32 ch
        self.enc4 = backbone[7:14]  # skip4: 96 ch
        self.enc5 = backbone[14:19] # bottleneck: 1280 ch 

        # Decoder
        self.d4 = DecoderBlock(in_ch=1280, skip_ch=96, out_ch=256, attention_type=attention_type)
        self.d3 = DecoderBlock(in_ch=256, skip_ch=32, out_ch=128, attention_type=attention_type)
        self.d2 = DecoderBlock(in_ch=128, skip_ch=24, out_ch=64, attention_type=attention_type)
        self.d1 = DecoderBlock(in_ch=64, skip_ch=16, out_ch=32, attention_type=attention_type)

        # Head (主输出)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        # 深度监督分支 (Deep Supervision Heads)
        if self.deep_supervision:
            # 辅助输出头，将其直接上采样回原图分辨率
            self.ds_head3 = nn.Conv2d(128, 1, kernel_size=1)
            self.ds_head2 = nn.Conv2d(64, 1, kernel_size=1)

    def encoder_params(self):
        from itertools import chain
        return chain(
            self.enc1.parameters(), self.enc2.parameters(),
            self.enc3.parameters(), self.enc4.parameters(),
            self.enc5.parameters()
        )

    def decoder_params(self):
        from itertools import chain
        params = chain(
            self.d4.parameters(), self.d3.parameters(),
            self.d2.parameters(), self.d1.parameters(),
            self.head.parameters()
        )
        if self.deep_supervision:
            params = chain(params, self.ds_head3.parameters(), self.ds_head2.parameters())
        return params

    def forward(self, x: torch.Tensor):
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

        out = self.head(d1)
        
        if self.deep_supervision and self.training:
            import torch.nn.functional as F
            # 辅助输出，上采样到与 out 相同的尺寸
            out_d3 = self.ds_head3(d3)
            out_d3 = F.interpolate(out_d3, size=out.shape[2:], mode='bilinear', align_corners=False)
            
            out_d2 = self.ds_head2(d2)
            out_d2 = F.interpolate(out_d2, size=out.shape[2:], mode='bilinear', align_corners=False)
            
            # 训练时返回元组
            return out, out_d2, out_d3
            
        # 推理时只返回主输出
        return out

if __name__ == "__main__":
    # Test with 640x640
    model = U_Net_MobileNetV2(pretrained=False)
    dummy_input = torch.randn(2, 3, 640, 640)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
