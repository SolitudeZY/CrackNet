"""
DeepLabV3+ with ResNet101 backbone, skip-connection decoder, and CBAM attention
for high-precision crack segmentation.

Architecture overview:
  Encoder: ResNet101 backbone (ImageNet pretrained) + ASPP head from torchvision
  Decoder: DeepLabV3+ style with:
    - Low-level feature skip connection from ResNet layer1 (256ch)
    - CBAM (Channel + Spatial Attention) at decoder merge points
    - Gradual upsampling (4x -> 4x) instead of direct 8x bilinear

Training mode:  returns (main_logits, aux_logits) for dual-loss
Eval mode:      returns main_logits only

Output shape: (N, 1, H, W) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.resnet import ResNet101_Weights


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention: squeeze spatial dims -> MLP -> channel weights."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))  # (B, C)
        max_pool = x.amax(dim=(2, 3))  # (B, C)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise pooling -> conv -> spatial weights."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.bn(self.conv(torch.cat([avg_pool, max_pool], dim=1))))
        return x * attn


class CBAM(nn.Module):
    """CBAM: Channel Attention -> Spatial Attention (sequential)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# DeepLabV3+ Decoder with skip connections
# ---------------------------------------------------------------------------

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+ style decoder with low-level feature fusion and CBAM.

    Fuses high-level ASPP features (stride=8) with low-level backbone features
    (stride=4) for sharper boundary prediction.

    Architecture:
        1. Project low-level features (256ch) to 48ch via 1x1 conv
        2. Upsample ASPP output 2x to match low-level spatial size
        3. Concatenate [low-level 48ch, high-level 256ch] = 304ch
        4. Apply CBAM attention on fused features
        5. Refine with 3x3 convs -> 256ch
        6. Final 1x1 conv -> 1ch logits
        7. Upsample 4x to input resolution
    """

    def __init__(self, low_level_channels: int = 256, aspp_channels: int = 256):
        super().__init__()
        # Project low-level features to reduce channel count
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Fuse concatenated features: 48 (low) + 256 (high) = 304
        fused_channels = 48 + aspp_channels
        self.cbam = CBAM(fused_channels, reduction=16)

        self.refine = nn.Sequential(
            nn.Conv2d(fused_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Edge-aware refinement head
        self.edge_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(
        self, aspp_features: torch.Tensor, low_level_features: torch.Tensor, input_size: tuple[int, int]
    ) -> torch.Tensor:
        # Project low-level features
        low = self.low_level_proj(low_level_features)  # (B, 48, H/4, W/4)

        # Upsample ASPP features to match low-level spatial size
        high = F.interpolate(aspp_features, size=low.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate and fuse
        fused = torch.cat([low, high], dim=1)  # (B, 304, H/4, W/4)
        fused = self.cbam(fused)
        fused = self.refine(fused)  # (B, 256, H/4, W/4)

        # Predict logits
        logits = self.edge_head(fused)  # (B, 1, H/4, W/4)

        # Upsample to input resolution (4x)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class CrackDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with ResNet101, CBAM attention, and skip-connection decoder.

    Uses torchvision's DeepLabV3 as the encoder (backbone + ASPP).
    Adds a custom decoder for high-resolution boundary prediction.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Build the base DeepLabV3 model
        backbone_weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        base = deeplabv3_resnet101(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=1,
            aux_loss=True,
        )

        # Extract backbone and ASPP classifier
        self.backbone = base.backbone
        self.aux_classifier = base.aux_classifier  # FCNHead for auxiliary loss

        # torchvision DeepLabHead structure:
        #   [0] ASPP -> 256ch
        #   [1] Conv2d(256, 256, 3)
        #   [2] BatchNorm2d(256)
        #   [3] ReLU
        #   [4] Conv2d(256, num_classes, 1)  <- skip this, decoder replaces it
        classifier = base.classifier
        self.aspp_encoder = nn.Sequential(
            classifier[0],  # ASPP
            classifier[1],  # Conv2d(256, 256)
            classifier[2],  # BatchNorm2d
            classifier[3],  # ReLU
        )

        # Custom DeepLabV3+ decoder
        # ResNet layer1 outputs 256 channels at stride 4
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=256,
            aspp_channels=256,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[2:]

        # --- Encoder: backbone feature extraction ---
        # Run backbone stages manually to capture intermediate features
        features = {}
        x_enc = x
        for name, module in self.backbone.named_children():
            x_enc = module(x_enc)
            features[name] = x_enc

        # Low-level features from layer1 (stride=4, 256ch)
        low_level = features["layer1"]

        # High-level features from layer4 (stride=8 with dilated convs)
        high_level = features["layer4"]

        # --- ASPP encoding ---
        aspp_out = self.aspp_encoder(high_level)  # (B, 256, H/8, W/8)

        # --- Decoder with skip connection ---
        main_logits = self.decoder(aspp_out, low_level, input_size)  # (B, 1, H, W)

        if self.training:
            # Auxiliary output from layer3 features
            aux_logits = self.aux_classifier(features["layer3"])
            aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
            return main_logits, aux_logits

        return main_logits


def build_model(pretrained: bool = True) -> CrackDeepLabV3Plus:
    return CrackDeepLabV3Plus(pretrained=pretrained)


if __name__ == "__main__":
    model = build_model(pretrained=False)

    # Test eval mode (batch=1 ok)
    model.eval()
    y = model(torch.randn(1, 3, 512, 512))
    print(f"Eval output: {tuple(y.shape)}")

    # Test train mode (batch>=2 required due to ASPP BatchNorm)
    model.train()
    main, aux = model(torch.randn(2, 3, 512, 512))
    print(f"Train main: {tuple(main.shape)}, aux: {tuple(aux.shape)}")

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {params:,}")
    print(f"Trainable params: {trainable:,}")

    # Check decoder params specifically
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Decoder params: {decoder_params:,}")
