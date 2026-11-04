"""I2IReg-ClfNet: cascaded multi-task network (Fig. 2.1).

  x --> I2IRegNet --> M ; ROI_hat = M * x --> ClfNet --> stone logits
  L_total = w_reg * L_reg + w_clf * L_clf ,  L_reg = 1/(2N) * sum (ROI - ROI_hat)^2 ,
  L_clf = cross-entropy ;  (w_reg, w_clf) = (0.8, 0.2) in the paper.
"""
import torch.nn as nn
import torch.nn.functional as F

from .i2iregnet import I2IRegNet
from .clfnet import ClfNet


def half_mse(pred, target):
    return 0.5 * F.mse_loss(pred, target)


class I2IRegClfNet(nn.Module):
    def __init__(self, deep_encoder="squeezenet", use_shallow=True, use_cbam=True, pretrained=True, dropout=0.5):
        super().__init__()
        self.reg = I2IRegNet(pretrained=pretrained)
        self.clf = ClfNet(deep_encoder=deep_encoder, use_shallow=use_shallow, use_cbam=use_cbam,
                          pretrained=pretrained, dropout=dropout)

    def forward(self, x):
        r = self.reg(x)
        roi = r["roi"].clamp(0.0, 1.0)
        logits = self.clf(roi)
        return {"roi": r["roi"], "map": r["map"], "roi_clamped": roi, "logits": logits}


class CascadeLoss(nn.Module):
    def __init__(self, w_reg=0.8, w_clf=0.2, class_weights=None):
        super().__init__()
        self.w_reg, self.w_clf = w_reg, w_clf
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, out, roi_gt, label):
        l_reg = half_mse(out["roi"], roi_gt)
        l_clf = self.ce(out["logits"], label)
        return self.w_reg * l_reg + self.w_clf * l_clf, {"l_reg": l_reg.item(), "l_clf": l_clf.item()}
