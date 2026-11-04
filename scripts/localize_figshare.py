"""Run the trained StoneUNet (trained on the uploaded dataset) on the Figshare dataset.

Figshare has no stone masks, so pixel accuracy cannot be measured. Instead we check:
  * slice-level detection: a slice counts as Stone+ if at least one stone region is found
    -> accuracy / recall / specificity / precision / F1 / ROC-AUC / kappa against the labels
  * example images with the predicted stones boxed, for visual inspection

    python scripts/localize_figshare.py --ckpt outputs/stone_seg/fold0/best.pt
    python scripts/localize_figshare.py --ckpt outputs/stone_seg/fold0/best.pt --split all --examples 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.io_utils import read_gray, resize, to_3ch_tensor
from src.metrics import classification_metrics, components
from src.models import StoneUNet
from src.utils import load_config, resolve, get_device, save_json

KEYS = ["accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "kappa"]


def draw(img, prob, stones, title):
    rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgb[prob > 0.5] = (40, 40, 255)
    for s in stones:
        x, y, w, h = s["bbox"]
        cv2.rectangle(rgb, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 255), 1)
    bar = np.full((26, rgb.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, rgb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/stone_seg/fold0/best.pt")
    ap.add_argument("--split", default="holdout", choices=["holdout", "all"])
    ap.add_argument("--threshold", type=float, default=None, help="pixel threshold (default from config)")
    ap.add_argument("--min_area", type=int, default=None)
    ap.add_argument("--examples", type=int, default=20, help="number of example images to save per class")
    ap.add_argument("--out", default="outputs/stone_seg_on_figshare")
    ap.add_argument("--exclude_overlap", action="store_true",
                    help="drop hold-out slices that match a StoneUNet training slice (from check_overlap.py)")
    ap.add_argument("--near", type=float, default=0.95)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device(cfg["common"]["device"])
    thr = a.threshold if a.threshold is not None else cfg["stone_seg"]["threshold"]
    min_area = a.min_area if a.min_area is not None else cfg["stone_seg"]["min_area"]

    root = resolve(cfg["paths"]["kidney_data"])
    df = pd.read_csv(root / "labels.csv")
    if a.split == "holdout":
        folds = json.load(open(root / "splits.json", encoding="utf-8"))["folds"]
        df = df[df["filename"].map(folds) == -1]
    if a.exclude_overlap:
        ov = pd.read_csv(resolve("outputs") / "overlap_check.csv")
        bad = set(ov[(ov.best_corr >= a.near) & ov.closest_in_stone_train].filename)
        df = df[~df["filename"].isin(bad)]
        print(f"excluded {len(bad)} slices that overlap with StoneUNet training data")
        a.out = a.out + "_clean"
    print(f"{len(df)} Figshare slices ({a.split}): Stone+ {int(df.label.sum())}, Stone- {int((df.label == 0).sum())}")

    ck = torch.load(resolve(a.ckpt), map_location="cpu", weights_only=False)
    c = ck.get("extra", {}).get("cfg", {})
    model = StoneUNet(c.get("encoder", "resnet34"), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    size = c.get("full_size", 512)

    out = resolve(a.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    rows, saved = [], {0: 0, 1: 0}
    for r in tqdm(df.itertuples(), total=len(df)):
        img, _ = read_gray(root / "images" / r.filename)
        img = resize(img, size)
        with torch.no_grad():
            prob = torch.sigmoid(model(to_3ch_tensor(img)[None].to(device)))[0, 0].cpu().numpy()
        stones, _ = components(prob > thr, min_area)
        score = float(prob.max())
        rows.append({"filename": r.filename, "label": int(r.label), "n_stones": len(stones),
                     "max_prob": score, "stone_area_px": int(sum(s["area"] for s in stones))})
        if saved[r.label] < a.examples:
            title = f"{r.filename}  true={'Stone+' if r.label else 'Stone-'}  found={len(stones)}"
            cv2.imwrite(str(out / "examples" / f"{'pos' if r.label else 'neg'}_{Path(r.filename).stem}.png"),
                        draw(img, prob, stones, title))
            saved[r.label] += 1

    res = pd.DataFrame(rows)
    res.to_csv(out / "predictions.csv", index=False)
    y = res["label"].tolist()
    # slice is Stone+ if any stone region found; ROC-AUC uses the max stone probability
    m = classification_metrics(y, (res["n_stones"] > 0).astype(float).tolist(), 0.5)
    m["roc_auc"] = classification_metrics(y, res["max_prob"].tolist(), 0.5)["roc_auc"]
    save_json(m, out / "metrics.json")
    print("\nStoneUNet used as a detector on Figshare (stone found = Stone+):")
    print("  " + "  ".join(f"{k}={m[k]:.4f}" for k in KEYS))
    print(f"  TP={m['tp']}  FN={m['fn']}  TN={m['tn']}  FP={m['fp']}")
    print(f"  avg stones found: Stone+ slices {res[res.label == 1].n_stones.mean():.2f}, "
          f"Stone- slices {res[res.label == 0].n_stones.mean():.2f}")
    print(f"\nexample images (red = stone pixels, yellow box = stone) saved to {out / 'examples'}")


if __name__ == "__main__":
    main()
