"""Quick self-test - run this first after installing requirements:

    python tests/test_setup.py

Checks: model shapes, parameter counts, loss backward passes, Grad-CAM, metrics, and (if present)
loading of the uploaded stone dataset. No internet / pretrained weights needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.models import I2IRegNet, ClfNet, I2IRegClfNet, CascadeLoss, StoneUNet, FocalDiceLoss
from src.gradcam import gradcam
from src.metrics import classification_metrics, StoneLocalisationMeter, RegressionMeter


def count(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable)) / 1e6


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 224, 224)

    reg = I2IRegNet(pretrained=False)
    o = reg(x)
    assert o["map"].shape == (2, 3, 224, 224) and o["roi"].shape == (2, 3, 224, 224)
    print(f"[ok] I2IRegNet  out {tuple(o['roi'].shape)}  params {count(reg):.2f} M (paper ~11.3 M)")

    for enc in ("squeezenet", "resnet18"):
        clf = ClfNet(deep_encoder=enc, pretrained=False)
        y = clf(x)
        assert y.shape == (2, 2)
        print(f"[ok] ClfNet ShN+{enc}+CBAM logits {tuple(y.shape)} trainable {count(clf, True):.2f} M")

    model = I2IRegClfNet(pretrained=False)
    o = model(x)
    loss, parts = CascadeLoss()(o, torch.rand(2, 3, 224, 224), torch.tensor([0, 1]))
    loss.backward()
    assert model.reg.mapping.weight.grad is not None, "no gradient reached I2IRegNet"
    print(f"[ok] I2IReg-ClfNet total params {count(model):.2f} M | loss {loss.item():.4f} {parts}")

    cam, prob = gradcam(model.clf, o["roi_clamped"].detach())
    assert cam.shape == (2, 224, 224)
    print(f"[ok] Grad-CAM {cam.shape}, probs {prob.round(3).tolist()}")

    su = StoneUNet(pretrained=False)
    xs = torch.rand(1, 3, 256, 256)
    ls = su(xs)
    assert ls.shape == (1, 1, 256, 256)
    tgt = torch.zeros_like(ls)
    tgt[..., 100:104, 120:124] = 1
    FocalDiceLoss()(ls, tgt).backward()
    with torch.no_grad():
        assert su.eval()(torch.rand(1, 3, 512, 512)).shape == (1, 1, 512, 512)
    print(f"[ok] StoneUNet logits {tuple(ls.shape)} params {count(su):.2f} M")

    m = classification_metrics([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.1])
    assert abs(m["accuracy"] - 0.75) < 1e-9
    sm = StoneLocalisationMeter()
    g = np.zeros((64, 64)); g[10:13, 10:13] = 1; g[40:42, 40:42] = 1
    p = np.zeros((64, 64)); p[11:13, 11:13] = 1; p[50:53, 5:8] = 1
    sm.update(p, g)
    r = sm.compute()
    assert r["stone_tp"] == 1 and r["stone_fn"] == 1 and r["stone_fp"] == 1
    rm = RegressionMeter()
    rm.update(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8), torch.ones(1, 1, 8, 8))
    print("[ok] metrics")

    root = Path(__file__).resolve().parents[1] / "data" / "stone_seg"
    if (root / "image").exists():
        from src.datasets import StoneSegDataset
        ds = StoneSegDataset(root, train=True)
        b = ds[0]
        print(f"[ok] stone dataset: {len(ds)} slices, train crop {tuple(b['image'].shape)}, "
              f"stone pixels in crop {int(b['mask'].sum())}")
        b = StoneSegDataset(root, train=False)[0]
        print(f"[ok] full slice {tuple(b['image'].shape)}, stone pixels {int(b['mask'].sum())}")
    else:
        print("[--] data/stone_seg not found (copy the uploaded dataset there)")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
