"""StoneUNet - pixel-level kidney stone localisation.

Stones in the uploaded dataset are tiny (median ~29 px on a 512x512 slice), so this model
works at full resolution with a U-Net decoder over an ImageNet ResNet34 encoder and full
skip connections down to 1/2 resolution plus a full-resolution refinement stage.
Trained with Focal-BCE + soft Dice (the paper's segmentation-head ablation uses Focal BCE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import ImageNetNorm


class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_ch + skip_ch, out_ch), ConvBNReLU(out_ch, out_ch))

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StoneUNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True):
        super().__init__()
        self.norm = ImageNetNorm()
        if encoder == "resnet34":
            w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet34(weights=w)
        elif encoder == "resnet18":
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=w)
        else:
            raise ValueError(encoder)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # 1/2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.full_res = nn.Sequential(ConvBNReLU(3, 16), ConvBNReLU(16, 16))
        self.d4 = DecoderBlock(512, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d2 = DecoderBlock(128, 64, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.d0 = DecoderBlock(32, 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)   # prior ~1.8% foreground -> stable start for rare stones

    def forward(self, x):
        h = self.norm(x)
        s0 = self.full_res(h)                 # 1/1
        s1 = self.stem(h)                     # 1/2
        s2 = self.l1(self.pool(s1))           # 1/4
        s3 = self.l2(s2)                      # 1/8
        s4 = self.l3(s3)                      # 1/16
        b = self.l4(s4)                       # 1/32
        y = self.d4(b, s4)
        y = self.d3(y, s3)
        y = self.d2(y, s2)
        y = self.d1(y, s1)
        y = self.d0(y, s0)
        return self.head(y)                   # logits (B,1,H,W)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.dw, self.fw = dice_weight, focal_weight

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = (a_t * (1 - p_t) ** self.gamma * bce).mean()
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1.0) / (denom + 1.0)
        return self.fw * focal * 10.0 + self.dw * dice.mean()


if __name__ == "__main__":
    m = StoneUNet(pretrained=False)
    print(m(torch.rand(1, 3, 512, 512)).shape)
