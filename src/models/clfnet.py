"""ClfNet - dual-encoder classification subnetwork (Section 2.1.2 / Fig. 2.3).

 * Shallow encoder : N core blocks [Conv3x3-BN-ReLU-MaxPool-Conv3x3-BN-ReLU], trained from
                     scratch. Kernels found by the paper's Bayesian search: 128,32,96,64,256,224.
                     Blocks producing maps smaller than 7x7 are pruned (so 5 blocks at 224 input).
 * Deep encoder    : frozen ImageNet backbone truncated at 7x7 (SqueezeNet = paper's best
                     choice "ShN + SqN + CBAM"; ResNet18 and EfficientNetV2-S also supported).
 * Fusion head     : concat -> BN -> 1x1 conv (round((Cs+Cd)/2)) -> CBAM -> ReLU -> Dropout
                     -> AdaptiveAvgPool -> FC(2)   (softmax applied in the loss / at inference)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import ImageNetNorm, CBAM

PAPER_SHALLOW_KERNELS = (128, 32, 96, 64, 256, 224)


class CoreBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class ShallowEncoder(nn.Module):
    def __init__(self, kernels=PAPER_SHALLOW_KERNELS, input_size=224, target_size=7):
        super().__init__()
        blocks, in_ch, size = [], 3, input_size
        for k in kernels:
            if size // 2 < target_size:      # prune blocks that go below 7x7
                break
            blocks.append(CoreBlock(in_ch, k))
            in_ch, size = k, size // 2
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x):
        return self.blocks(x)


class DeepEncoder(nn.Module):
    """Frozen ImageNet backbone giving a 7x7 feature map."""

    def __init__(self, name="squeezenet", pretrained=True, freeze=True):
        super().__init__()
        name = name.lower()
        if name == "squeezenet":
            w = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
            self.features = torchvision.models.squeezenet1_1(weights=w).features
            self.out_channels = 512
        elif name == "resnet18":
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=w)
            self.features = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            self.out_channels = 512
        elif name in ("efficientnetv2s", "efficientnet_v2_s"):
            w = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.features = torchvision.models.efficientnet_v2_s(weights=w).features
            self.out_channels = 1280
        else:
            raise ValueError(f"Unknown deep encoder '{name}'")
        self.frozen = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:                    # keep BN statistics of the frozen backbone fixed
            self.features.eval()
        return self

    def forward(self, x):
        f = self.features(x)
        if f.shape[-1] != 7 or f.shape[-2] != 7:
            f = F.adaptive_avg_pool2d(f, 7)  # SqueezeNet gives 13x13 at 224 -> resample to 7x7
        return f


class ClfNet(nn.Module):
    def __init__(self, deep_encoder="squeezenet", use_shallow=True, use_deep=True, use_cbam=True,
                 shallow_kernels=PAPER_SHALLOW_KERNELS, input_size=224, dropout=0.5,
                 num_classes=2, pretrained=True, freeze_deep=True):
        super().__init__()
        assert use_shallow or use_deep
        self.norm = ImageNetNorm()
        self.shallow = ShallowEncoder(shallow_kernels, input_size) if use_shallow else None
        self.deep = DeepEncoder(deep_encoder, pretrained, freeze_deep) if use_deep else None
        c = (self.shallow.out_channels if use_shallow else 0) + (self.deep.out_channels if use_deep else 0)
        c_red = int(round(c / 2))
        self.fuse_bn = nn.BatchNorm2d(c)
        self.reduce = nn.Conv2d(c, c_red, 1)
        self.cbam = CBAM(c_red) if use_cbam else nn.Identity()
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_red, num_classes)
        self.feature_maps = None   # stored for Grad-CAM

    def forward(self, x):
        """x: [0,1] image or ROI, shape (B,3,224,224). Returns logits (B,num_classes)."""
        h = self.norm(x)
        feats = []
        if self.shallow is not None:
            feats.append(self.shallow(h))
        if self.deep is not None:
            feats.append(self.deep(h))
        if len(feats) == 2 and feats[0].shape[-2:] != feats[1].shape[-2:]:
            feats[0] = F.adaptive_avg_pool2d(feats[0], feats[1].shape[-2:])
        f = torch.cat(feats, dim=1)
        f = self.cbam(self.reduce(self.fuse_bn(f)))
        self.feature_maps = f            # Grad-CAM target (7x7 attention-refined features)
        f = self.drop(self.act(f))
        return self.fc(torch.flatten(self.pool(f), 1))


if __name__ == "__main__":
    m = ClfNet(pretrained=False)
    print(m(torch.rand(2, 3, 224, 224)).shape)
    print("trainable params (M):", sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6)
