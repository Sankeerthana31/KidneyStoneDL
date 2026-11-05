"""End-to-end inference:

  CT slice --(I2IRegNet)--> kidney ROI  --(ClfNet)--> P(stone) + Grad-CAM
           --(StoneUNet)--> stone probability map, restricted to the (dilated) kidney ROI
           --> stone boxes / centroids / areas / side.

Any stage can be omitted: with only the StoneUNet checkpoint you get stone localisation on the
full slice; with only the cascade checkpoint you get kidney localisation + stone detection.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from .gradcam import gradcam, overlay_heatmap
from .io_utils import read_gray, resize, to_3ch_tensor
from .metrics import components
from .models import I2IRegNet, ClfNet, I2IRegClfNet, StoneUNet
from .utils import get_device


def _peek_cfg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("extra", {}).get("cfg", {}) if isinstance(ck, dict) else {}, ck


def _saved_threshold(ck):
    return ck.get("extra", {}).get("threshold") if isinstance(ck, dict) else None


def _load(model, ck):
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd)
    return model


def clean_kidney_mask(mask, keep=2, open_px=3):
    m = mask.astype(np.uint8)
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m
    order = np.argsort(-st[1:, 4])[:keep] + 1
    return np.isin(lab, order).astype(np.uint8)


def kidney_region(mask, dilate_px=15, hull=True):
    """Search region for stones: each kidney's convex hull (fills the C-shaped mask's hollow
    centre - the renal sinus/pelvis, where many stones sit), then dilated by `dilate_px`."""
    m = (mask > 0).astype(np.uint8)
    if hull and m.any():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        for c in cnts:
            cv2.fillPoly(filled, [cv2.convexHull(c)], 1)
        m = filled
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
    return m


class KidneyStonePipeline:
    def __init__(self, cascade_ckpt=None, reg_ckpt=None, clf_ckpt=None, stone_ckpt=None,
                 device="auto", cfg=None):
        self.device = get_device(device)
        self.cfg = {"stone_prob_threshold": 0.5, "seg_threshold": 0.5, "restrict_to_kidney": True,
                    "kidney_dilate_px": 30, "roi_threshold": 0.05, "min_area": 2,
                    "dicom_window": [400, 1800], "img_size": 224, "full_size": 512}
        self.cfg.update(cfg or {})
        self.reg = self.clf = self.stone = None

        if cascade_ckpt:
            c, ck = _peek_cfg(cascade_ckpt)
            m = I2IRegClfNet(deep_encoder=c.get("deep_encoder", "squeezenet"), use_shallow=c.get("use_shallow", True),
                             use_cbam=c.get("use_cbam", True), pretrained=False)
            _load(m, ck)
            self.reg, self.clf = m.reg, m.clf
            self.cfg["roi_threshold"] = c.get("roi_threshold", self.cfg["roi_threshold"])
        else:
            if reg_ckpt:
                _, ck = _peek_cfg(reg_ckpt)
                self.reg = _load(I2IRegNet(pretrained=False), ck)
                t = ck.get("extra", {}).get("roi_threshold") if isinstance(ck, dict) else None
                if t is not None:
                    self.cfg["roi_threshold"] = float(t)
            if clf_ckpt:
                c, ck = _peek_cfg(clf_ckpt)
                self.clf = _load(ClfNet(deep_encoder=c.get("deep_encoder", "squeezenet"),
                                        use_shallow=c.get("use_shallow", True), use_cbam=c.get("use_cbam", True),
                                        pretrained=False), ck)
                self.clf_input = c.get("input", "roi")
                t = _saved_threshold(ck)
                if t is not None and not (cfg or {}).get("override_threshold"):
                    self.cfg["stone_prob_threshold"] = float(t)
        if stone_ckpt:
            c, ck = _peek_cfg(stone_ckpt)
            self.stone = _load(StoneUNet(c.get("encoder", "resnet34"), pretrained=False), ck)
            self.cfg["full_size"] = c.get("full_size", 512)
        for m in (self.reg, self.clf, self.stone):
            if m is not None:
                m.to(self.device).eval()
        if not hasattr(self, "clf_input"):
            self.clf_input = "roi" if self.reg is not None else "image"

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _kidney(self, img224):
        x = to_3ch_tensor(img224)[None].to(self.device)
        out = self.reg(x)
        roi = out["roi"].clamp(0, 1)
        mask = (roi[0].mean(0).cpu().numpy() > self.cfg["roi_threshold"])
        return roi, clean_kidney_mask(mask)

    def __call__(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            img, spacing = read_gray(image_or_path, tuple(self.cfg["dicom_window"]))
        else:
            img, spacing = np.asarray(image_or_path, dtype=np.float32), None
            if img.max() > 1.0:
                img = img / 255.0
        H, W = img.shape
        S, F = self.cfg["img_size"], self.cfg["full_size"]
        img224 = resize(img, S)
        res = {"image": img, "size": (H, W), "pixel_spacing": spacing}

        # 1) kidney localisation
        roi = None
        if self.reg is not None:
            roi, kmask224 = self._kidney(img224)
            res["roi"] = roi[0].mean(0).cpu().numpy()
            res["kidney_mask"] = cv2.resize(kmask224, (W, H), interpolation=cv2.INTER_NEAREST)
            res["kidney_area_px"] = int(res["kidney_mask"].sum())

        # 2) stone detection (classification) + Grad-CAM
        if self.clf is not None:
            x = roi if (roi is not None and self.clf_input == "roi") else to_3ch_tensor(img224)[None].to(self.device)
            cam, probs = gradcam(self.clf, x, target_class=1)
            res["stone_probability"] = float(probs[0, 1])
            res["stone_present"] = bool(probs[0, 1] >= self.cfg["stone_prob_threshold"])
            res["gradcam"] = cv2.resize(cam[0], (W, H))

        # 3) stone localisation
        if self.stone is not None:
            with torch.no_grad():
                xs = to_3ch_tensor(resize(img, F))[None].to(self.device)
                prob = torch.sigmoid(self.stone(xs))[0, 0].cpu().numpy()
            prob = cv2.resize(prob, (W, H))
            # kidney found -> keep stones in its (hull + dilated) region; no kidney (e.g. ureter level) -> whole slice
            if self.cfg["restrict_to_kidney"] and "kidney_mask" in res and res["kidney_mask"].any():
                k = max(1, int(self.cfg["kidney_dilate_px"] * W / 512))
                prob = prob * kidney_region(res["kidney_mask"], k, hull=True)
            res["stone_prob_map"] = prob
            seg = prob > self.cfg["seg_threshold"]
            stones, _ = components(seg, self.cfg["min_area"])
            for s in stones:
                cx = s["centroid"][0]
                # radiological convention: patient's right kidney is on the image left
                s["side"] = "right kidney (image left)" if cx < W / 2 else "left kidney (image right)"
                s["max_prob"] = float(prob[s["bbox"][1]:s["bbox"][1] + s["bbox"][3],
                                           s["bbox"][0]:s["bbox"][0] + s["bbox"][2]].max())
                if spacing:
                    s["area_mm2"] = s["area"] * spacing[0] * spacing[1]
                    s["max_diameter_mm"] = max(s["bbox"][2] * spacing[1], s["bbox"][3] * spacing[0])
                s.pop("id", None)
            res["stones"] = stones
            res["stone_mask"] = seg.astype(np.uint8)
            if "stone_present" not in res:
                res["stone_present"] = len(stones) > 0
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def render(res):
        """Returns an RGB uint8 panel: CT | kidney ROI | Grad-CAM | stones."""
        img = res["image"]
        H, W = img.shape
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        panels = [("CT slice", gray.copy())]
        if "kidney_mask" in res:
            k = gray.copy()
            cnts, _ = cv2.findContours(res["kidney_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(k, cnts, -1, (0, 255, 0), max(1, W // 256))
            panels.append(("Kidney ROI (I2IRegNet)", k))
        if "gradcam" in res:
            panels.append((f"Grad-CAM  P(stone)={res['stone_probability']:.2f}", overlay_heatmap(img, res["gradcam"])))
        if "stones" in res:
            s_img = gray.copy()
            for i, s in enumerate(res["stones"], 1):
                x, y, w, h = s["bbox"]
                pad = max(3, W // 128)
                cv2.rectangle(s_img, (x - pad, y - pad), (x + w + pad, y + h + pad), (255, 0, 0), max(1, W // 256))
                cv2.putText(s_img, str(i), (x + w + pad + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * W / 512,
                            (255, 255, 0), 1, cv2.LINE_AA)
            s_img[res["stone_mask"] > 0] = (255, 40, 40)
            panels.append((f"Stones found: {len(res['stones'])}", s_img))
        tiles = []
        for title, p in panels:
            p = cv2.resize(p, (512, int(512 * H / W)))
            bar = np.full((28, p.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, p]))
        return np.hstack(tiles)

    @staticmethod
    def summary(res, name=""):
        out = {"file": name}
        for k in ("stone_present", "stone_probability", "kidney_area_px"):
            if k in res:
                out[k] = res[k]
        if "stones" in res:
            out["n_stones"] = len(res["stones"])
            out["stones"] = res["stones"]
        return out
