"""Grad-CAM (Selvaraju et al., 2017) on ClfNet's attention-refined 7x7 feature maps."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(clf, x, target_class=1):
    """clf: ClfNet; x: (B,3,H,W) in [0,1]. Returns (cams (B,H,W) in [0,1], probs (B,C))."""
    was_training = clf.training
    clf.eval()
    x = x.detach().clone().requires_grad_(True)   # makes sure a graph exists even with frozen encoders
    with torch.enable_grad():
        logits = clf(x)
        fmap = clf.feature_maps
        fmap.retain_grad()
        score = logits[:, target_class].sum()
        clf.zero_grad(set_to_none=True)
        score.backward()
        w = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam.detach()
    mn = cam.flatten(1).min(1)[0][:, None, None]
    mx = cam.flatten(1).max(1)[0][:, None, None]
    cam = (cam - mn) / (mx - mn + 1e-8)
    clf.train(was_training)
    return cam.cpu().numpy(), torch.softmax(logits.detach(), 1).cpu().numpy()


def overlay_heatmap(gray01, cam01, alpha=0.45):
    """gray01 HxW [0,1], cam01 HxW [0,1] -> RGB uint8."""
    base = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((cam01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
