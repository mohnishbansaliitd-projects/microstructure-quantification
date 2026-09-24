"""U-Net (Ronneberger et al., 2015) encoder-decoder with skip connections, sized for small-sample micrographs."""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class MicrostructureUNet(nn.Module):
    """Lightweight U-Net for secondary-phase segmentation in steel micrographs."""

    def __init__(self, in_channels: int = 1, out_classes: int = 1, base_features: int = 32):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))

        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 4, base_features * 2)
        
        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 2, base_features)
        
        # Output Head
        self.outc = nn.Conv2d(base_features, out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits


def build_pretrained_unet(
    num_classes: int = 1,
    in_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
    pretrained_encoder_path: Optional[str] = None
) -> nn.Module:
    """segmentation-models-pytorch U-Net for fine-tuning on the UHCS micrographs. in_channels=1
    goes straight to smp, which averages the pretrained RGB kernels down instead of us tiling
    the grayscale channel to 3.

    pretrained_encoder_path optionally overwrites the encoder weights after the imagenet init,
    with a state dict from ssl_pretrain.py's self-supervised stage -- see that module for why.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )

    if pretrained_encoder_path is not None and os.path.exists(pretrained_encoder_path):
        checkpoint = torch.load(pretrained_encoder_path, map_location="cpu")
        encoder_state = checkpoint["encoder_state_dict"] if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint else checkpoint
        model.encoder.load_state_dict(encoder_state, strict=True)

    return model
