"""Shared building blocks: ImageNet normalisation, CBAM attention, grouped/pointwise conv."""
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNorm(nn.Module):
    """Maps a [0,1] 3-channel tensor to ImageNet statistics (done inside the model so the
    raw [0,1] image is still available for the ROI multiplication)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class GroupedPointwise(nn.Module):
    """Grouped (depthwise) 3x3 conv -> pointwise 1x1 conv -> BN -> ReLU (paper Fig. 2.2)."""

    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.grouped = nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation,
                                 groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.grouped(x))))


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False))

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018): channel then spatial."""

    def __init__(self, ch, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))
