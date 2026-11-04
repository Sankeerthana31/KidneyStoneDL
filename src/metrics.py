"""Metrics used in the paper (Eq. 3.1-3.4) plus object-level stone localisation metrics."""
import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix


# ------------------------------------------------------------------ classification (Eq. 3.1, 3.2)
def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 or len(set(y_pred)) > 1 else 1.0
    return {"accuracy": acc, "recall": rec, "specificity": spec, "precision": prec, "f1": f1,
            "roc_auc": auc, "kappa": kappa, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


# ------------------------------------------------------------------ regression (Eq. 3.3, 3.4)
class RegressionMeter:
    """Accumulates MSE/RMSE/MAE over ROI images and IoU/Dice of the thresholded ROI vs the mask."""

    def __init__(self, roi_threshold=0.05):
        self.t = roi_threshold
        self.se = self.ae = self.n = 0.0
        self.tp = self.fp = self.fn = 0.0
        self.iou_list, self.dice_list = [], []

    def update(self, roi_pred, roi_gt, mask_gt):
        roi_pred = roi_pred.detach().float().clamp(0, 1).cpu().numpy()
        roi_gt = roi_gt.detach().float().cpu().numpy()
        mask_gt = mask_gt.detach().float().cpu().numpy() > 0.5
        self.se += ((roi_pred - roi_gt) ** 2).sum()
        self.ae += np.abs(roi_pred - roi_gt).sum()
        self.n += roi_pred.size
        pred_bin = roi_pred.mean(axis=1, keepdims=True) > self.t     # (B,1,H,W)
        for p, g in zip(pred_bin, mask_gt):
            tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
            self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
            if g.sum() > 0 or p.sum() > 0:
                self.iou_list.append(tp / (tp + fp + fn + 1e-12))
                self.dice_list.append(2 * tp / (2 * tp + fp + fn + 1e-12))

    def compute(self):
        mse = self.se / max(self.n, 1)
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": self.ae / max(self.n, 1),
                "iou": float(np.mean(self.iou_list)) if self.iou_list else float("nan"),
                "dice": float(np.mean(self.dice_list)) if self.dice_list else float("nan"),
                "iou_global": self.tp / (self.tp + self.fp + self.fn + 1e-12)}


# ------------------------------------------------------------------ stone localisation
def components(mask, min_area=1):
    """Connected components -> list of dict(bbox=(x,y,w,h), centroid=(cx,cy), area, label_id)."""
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append({"bbox": (int(x), int(y), int(w), int(h)), "centroid": (float(cents[i][0]), float(cents[i][1])),
                        "area": int(a), "id": i})
    return out, lab


class StoneLocalisationMeter:
    """Pixel Dice/IoU + object-level detection (a GT stone is found if any predicted pixel
    overlaps it or a predicted centroid is within `dist_tol` px)."""

    def __init__(self, dist_tol=5.0, min_area=2):
        self.dist_tol, self.min_area = dist_tol, min_area
        self.dice, self.iou = [], []
        self.tp = self.fp = self.fn = 0
        self.ptp = self.pfp = self.pfn = 0

    def update(self, pred_mask, gt_mask):
        p, g = pred_mask > 0, gt_mask > 0
        tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
        self.ptp, self.pfp, self.pfn = self.ptp + tp, self.pfp + fp, self.pfn + fn
        if g.any() or p.any():
            self.dice.append(2 * tp / (2 * tp + fp + fn + 1e-12))
            self.iou.append(tp / (tp + fp + fn + 1e-12))
        pc, plab = components(p, self.min_area)
        gc, glab = components(g, 1)
        matched_pred = set()
        for s in gc:
            gpix = glab == s["id"]
            hit = None
            for j, q in enumerate(pc):
                if j in matched_pred:
                    continue
                overlap = (plab[gpix] == q["id"]).any()
                d = np.hypot(q["centroid"][0] - s["centroid"][0], q["centroid"][1] - s["centroid"][1])
                if overlap or d <= self.dist_tol:
                    hit = j
                    break
            if hit is None:
                self.fn += 1
            else:
                self.tp += 1
                matched_pred.add(hit)
        self.fp += len(pc) - len(matched_pred)

    def compute(self):
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        return {"dice": float(np.mean(self.dice)) if self.dice else float("nan"),
                "iou": float(np.mean(self.iou)) if self.iou else float("nan"),
                "dice_global": 2 * self.ptp / max(2 * self.ptp + self.pfp + self.pfn, 1),
                "stone_precision": prec, "stone_recall": rec,
                "stone_f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "stone_tp": self.tp, "stone_fp": self.fp, "stone_fn": self.fn}
