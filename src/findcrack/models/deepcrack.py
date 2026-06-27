try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)


    class DeepCrack(nn.Module):
        """
        DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation.
        Fuses hierarchical convolutional features from both the encoder and decoder stages 
        at the same scale.
        """
        def __init__(self, n_channels: int = 3, n_classes: int = 1):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes

            # Encoder (downsampling blocks)
            self.enc1 = DoubleConv(n_channels, 64)
            self.enc2 = DoubleConv(64, 128)
            self.enc3 = DoubleConv(128, 256)
            self.enc4 = DoubleConv(256, 512)
            self.enc5 = DoubleConv(512, 512)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Decoder (upsampling & concatenation blocks)
            self.dec5 = DoubleConv(512, 512)
            self.dec4 = DoubleConv(512 + 512, 256)
            self.dec3 = DoubleConv(256 + 256, 128)
            self.dec2 = DoubleConv(128 + 128, 64)
            self.dec1 = DoubleConv(64 + 64, 64)

            # Side prediction layers (maps feature channels to class channels at each scale)
            self.side1 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.side2 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.side3 = nn.Conv2d(128, n_classes, kernel_size=1)
            self.side4 = nn.Conv2d(256, n_classes, kernel_size=1)
            self.side5 = nn.Conv2d(512, n_classes, kernel_size=1)

            # Fusion layer that combines all 5 side predictions into the final output
            self.fuse = nn.Conv2d(n_classes * 5, n_classes, kernel_size=1)

        def forward(self, x):
            # 1. Encoder path
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            e5 = self.enc5(self.pool(e4))

            # 2. Decoder path (with bilinear interpolation upsampling)
            d5 = self.dec5(e5)
            
            d4_up = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=True)
            d4 = self.dec4(torch.cat([d4_up, e4], dim=1))

            d3_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
            d3 = self.dec3(torch.cat([d3_up, e3], dim=1))

            d2_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
            d2 = self.dec2(torch.cat([d2_up, e2], dim=1))

            d1_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
            d1 = self.dec1(torch.cat([d1_up, e1], dim=1))

            # 3. Extract side predictions and upsample to input dimensions
            h, w = x.shape[2:]
            s1 = F.interpolate(self.side1(d1), size=(h, w), mode='bilinear', align_corners=True)
            s2 = F.interpolate(self.side2(d2), size=(h, w), mode='bilinear', align_corners=True)
            s3 = F.interpolate(self.side3(d3), size=(h, w), mode='bilinear', align_corners=True)
            s4 = F.interpolate(self.side4(d4), size=(h, w), mode='bilinear', align_corners=True)
            s5 = F.interpolate(self.side5(d5), size=(h, w), mode='bilinear', align_corners=True)

            # 4. Fuse side predictions
            fused = self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1))

            return fused
else:
    class DeepCrack:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepCrack requires PyTorch. Please install PyTorch or "
                "install findcrack with standard extras: pip install findcrack[standard]"
            )

