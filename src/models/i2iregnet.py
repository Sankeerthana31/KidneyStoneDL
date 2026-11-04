"""I2IRegNet - image-to-image regression subnetwork for kidney ROI localisation.

DeepLabV3+-style encoder/decoder with a ResNet18 feature-extraction backbone (FEB),
output stride 16, a grouped-convolution ASPP and two up-sampling units, following
Section 2.1.1 / Fig. 2.2 of Öksüz et al., "I2IReg-ClfNet", BSPC 119 (2026) 109857.

Input  : x in [0,1], shape (B,3,H,W), H and W divisible by 16 (224 in the paper)
Output : dict(map=M, roi=ROI_hat) where ROI_hat = M * x  (the "reference line for
         weighting": the predicted map is multiplied point-wise with the input so ROI
         pixels keep their original intensities).
"""
import torch
import torch.nn as nn
import torchvision

from .common import ImageNetNorm, GroupedPointwise


def _resnet18(pretrained: bool):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=weights)


class ASPP(nn.Module):
    """Four parallel grouped-conv branches (rates 1, 6, 12, 18), each followed by a
    pointwise conv with 256 kernels + BN + ReLU, concatenated -> 1024 channels."""

    def __init__(self, in_ch, out_ch=256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([GroupedPointwise(in_ch, out_ch, dilation=r) for r in rates])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class I2IRegNet(nn.Module):
    def __init__(self, pretrained: bool = True, out_channels: int = 3, task: str = "regression"):
        """task='regression' -> paper's I2IRegNet (output multiplied with the input).
        task='segmentation' -> same body, 1-channel logits (paper's ablation, Table 3.8)."""
        super().__init__()
        self.task = task
        self.norm = ImageNetNorm()
        r = _resnet18(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # 1/2   64 ch
        self.pool = r.maxpool                                      # 1/4
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3  # 1/4, 1/8, 1/16
        c_low, c_high = 64, 256

        self.aspp = ASPP(c_high, 256)                              # 14x14x1024

        # Up-sampling unit #1
        self.dec_conv1 = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=4, padding=2)   # x4 exactly
        self.low_conv = nn.Sequential(nn.Conv2d(c_low, 48, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        # Up-sampling unit #2
        self.unit2 = nn.Sequential(GroupedPointwise(256 + 48, 256), GroupedPointwise(256, 256))
        out_ch = out_channels if task == "regression" else 1
        self.mapping = nn.Conv2d(256, out_ch, 1)
        self.up2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=8, stride=4, padding=2)

    @staticmethod
    def _crop_to(t, ref):
        """'2D cropping' layers of the paper - keeps sizes aligned."""
        return t[..., : ref.shape[-2], : ref.shape[-1]]

    def forward(self, x):
        h = self.norm(x)
        h = self.stem(h)
        low = self.layer1(self.pool(h))            # 56x56x64
        high = self.layer3(self.layer2(low))       # 14x14x256

        y = self.aspp(high)
        y = self.up1(self.dec_conv1(y))
        y = self._crop_to(y, low)
        y = torch.cat([y, self.low_conv(low)], dim=1)   # 56x56x304
        y = self.unit2(y)
        y = self.up2(self.mapping(y))
        y = self._crop_to(y, x)

        if self.task == "segmentation":
            return {"logits": y}
        return {"map": y, "roi": y * x}


if __name__ == "__main__":
    m = I2IRegNet(pretrained=False)
    out = m(torch.rand(2, 3, 224, 224))
    print({k: v.shape for k, v in out.items()})
    print("params (M):", sum(p.numel() for p in m.parameters()) / 1e6)
