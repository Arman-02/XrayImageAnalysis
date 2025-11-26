import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class BoneFractureSegmentationNet(nn.Module):
    """
    Simplified U-Net for bone fracture segmentation
    No skip connections to avoid dimension mismatches
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50 as encoder
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        else:
            resnet = models.resnet50(pretrained=pretrained)
        
        # Encoder blocks
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1  # 256 channels, 1/4 resolution
        self.encoder2 = resnet.layer2  # 512 channels, 1/8 resolution
        self.encoder3 = resnet.layer3  # 1024 channels, 1/16 resolution
        self.encoder4 = resnet.layer4  # 2048 channels, 1/32 resolution
        
        # Bottleneck
        self.bottleneck = DoubleConv(2048, 1024)
        
        # Decoder - simple upsampling without skip connections
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(1024, 512)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(512, 256)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(256, 128)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(128, 64)
        )
        
        self.decoder0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(64, 32)
        )
        
        # Final output head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)      # 1/4 resolution, 64 channels
        enc1 = self.encoder1(enc0)   # 1/4 resolution, 256 channels
        enc2 = self.encoder2(enc1)   # 1/8 resolution, 512 channels
        enc3 = self.encoder3(enc2)   # 1/16 resolution, 1024 channels
        enc4 = self.encoder4(enc3)   # 1/32 resolution, 2048 channels
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder
        dec4 = self.decoder4(bottleneck)   # 1/16 resolution
        dec3 = self.decoder3(dec4)         # 1/8 resolution
        dec2 = self.decoder2(dec3)         # 1/4 resolution
        dec1 = self.decoder1(dec2)         # 1/2 resolution
        dec0 = self.decoder0(dec1)         # 1/1 resolution (full)
        
        # Final output
        out = self.final_conv(dec0)
        return out

class DiceLoss(nn.Module):
    """Dice coefficient loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class BoundingBoxLoss(nn.Module):
    """Custom loss for bounding box prediction"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_bbox, target_bbox):
        coords_loss = self.mse_loss(pred_bbox[:, :4], target_bbox[:, :4])
        conf_loss = self.bce_loss(pred_bbox[:, 4:5], target_bbox[:, 4:5])
        return coords_loss + conf_loss

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model instantiation
    model = BoneFractureSegmentationNet(num_classes=1, backbone='resnet50', pretrained=True)
    model = model.to(device)
    
    # Dummy input
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("✓ Model test successful!")