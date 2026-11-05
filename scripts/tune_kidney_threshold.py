"""Find the best ROI threshold (and clean-up) for a trained I2IRegNet - no retraining.

The regression output is an image (kidney pixels keep their intensity); IoU/Dice need a
binary mask, so the output is thresholded. This script tries several thresholds on the
validation fold, with and without the pipeline's clean-up (keep the 2 largest regions),
picks the best, reports the hold-out with it, and saves it into the checkpoint so
predict_kidney_masks.py, the pipeline and the web app use it automatically.

    python scripts/tune_kidney_threshold.py --ckpt outputs/i2iregnet/fold0/best.pt --data data/kidney_masks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.datasets import KidneyROIDataset
from src.engine import loader
from src.models import I2IRegNet
from src.pipeline import clean_kidney_mask
from src.splits import load_splits, select
from src.utils import load_config, resolve, get_device, default_workers, save_json

THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


@torch.no_grad()
def collect(model, dl, device):
    maps, gts = [], []
    for b in tqdm(dl, leave=False):
        roi = model(b["image"].to(device))["roi"].clamp(0, 1).mean(1).cpu().numpy()   # (B,H,W)
        maps += list(roi)
        gts += list(b["mask"][:, 0].numpy() > 0.5)
    return maps, gts


def score(maps, gts, thr, clean):
    ious, dices, ious_k, empty_ok = [], [], [], []
    for m, g in zip(maps, gts):
        p = m > thr
        if clean:
            p = clean_kidney_mask(p).astype(bool)
        tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
        if g.sum() == 0:
            empty_ok.append(p.sum() == 0)
            if p.sum() == 0:
                continue
        iou = tp / (tp + fp + fn + 1e-9)
        ious.append(iou)
        dices.append(2 * tp / (2 * tp + fp + fn + 1e-9))
        if g.sum() > 0:
            ious_k.append(iou)
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)),
            "iou_kidney_slices": float(np.mean(ious_k)) if ious_k else float("nan"),
            "empty_slices_correct": float(np.mean(empty_ok)) if empty_ok else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/i2iregnet/fold0/best.pt")
    ap.add_argument("--data", default="data/kidney_masks")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    cfg = load_config()
    device = get_device(cfg["common"]["device"])
    ckpt_path = resolve(a.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = I2IRegNet(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    root = resolve(a.data)
    folds = load_splits(root / "splits.json")
    dl_va = loader(KidneyROIDataset(root, select(folds, a.fold, "val"), 224), 16, False, a.workers)
    dl_ho = loader(KidneyROIDataset(root, select(folds, a.fold, "holdout"), 224), 16, False, a.workers)
    print("running the model on validation and hold-out ...")
    va, ho = collect(model, dl_va, device), collect(model, dl_ho, device)

    print(f"\n{'threshold':>9} {'clean-up':>9} | {'val IoU':>8} {'val Dice':>8} {'IoU kidney slices':>18} {'empty correct':>14}")
    best = None
    for clean in (False, True):
        for t in THRESHOLDS:
            s = score(*va, t, clean)
            print(f"{t:9.2f} {('yes' if clean else 'no'):>9} | {s['iou']:8.4f} {s['dice']:8.4f} "
                  f"{s['iou_kidney_slices']:18.4f} {s['empty_slices_correct']:14.3f}")
            if best is None or s["iou"] > best[2]["iou"]:
                best = (t, clean, s)
    t, clean, s_va = best
    s_ho = score(*ho, t, clean)
    res = {"roi_threshold": t, "clean_up": clean, "val": s_va, "holdout": s_ho}
    save_json(res, ckpt_path.parent / "metrics_tuned.json")
    ck.setdefault("extra", {})["roi_threshold"] = t
    torch.save(ck, ckpt_path)
    print(f"\nbest: threshold {t:.2f}, clean-up {'yes' if clean else 'no'}  (saved into {ckpt_path.name})")
    print(f"hold-out: IoU {s_ho['iou']:.4f}  Dice {s_ho['dice']:.4f}  "
          f"IoU on kidney slices {s_ho['iou_kidney_slices']:.4f}  empty slices correct {s_ho['empty_slices_correct']:.3f}")


if __name__ == "__main__":
    main()
