"""Use a trained I2IRegNet to predict kidney masks for a dataset that has none
(e.g. the Figshare dataset in data/kidney). Masks are written to <data>/masks/ so that
  * train_clfnet.py --input roi   trains the classifier on the kidney ROI (paper's idea)
  * the pipeline / web app can keep stones only inside the kidneys

    python scripts/predict_kidney_masks.py --reg outputs/i2iregnet/fold0/best.pt --data data/kidney
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.io_utils import read_gray, resize, to_3ch_tensor
from src.models import I2IRegNet
from src.pipeline import clean_kidney_mask
from src.utils import load_config, resolve, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True)
    ap.add_argument("--data", default="data/kidney")
    ap.add_argument("--roi_threshold", type=float, default=None)
    ap.add_argument("--previews", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true", help="replace masks that already exist")
    a = ap.parse_args()
    cfg = load_config()
    device = get_device(cfg["common"]["device"])
    ck = torch.load(resolve(a.reg), map_location="cpu", weights_only=False)
    thr = (a.roi_threshold if a.roi_threshold is not None
           else ck.get("extra", {}).get("roi_threshold", cfg["i2iregnet"]["roi_threshold"]))
    print(f"ROI threshold: {thr}")
    model = I2IRegNet(pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    S = ck.get("extra", {}).get("cfg", {}).get("img_size", 224)

    root = resolve(a.data)
    (root / "masks").mkdir(exist_ok=True)
    (root / "mask_preview").mkdir(exist_ok=True)
    df = pd.read_csv(root / "labels.csv")
    n_empty, done = 0, 0
    for i, f in enumerate(tqdm(df["filename"])):
        out_p = root / "masks" / f
        if out_p.exists() and not a.overwrite:
            continue
        img, _ = read_gray(root / "images" / f)
        H, W = img.shape
        with torch.no_grad():
            roi = model(to_3ch_tensor(resize(img, S))[None].to(device))["roi"].clamp(0, 1)[0].mean(0).cpu().numpy()
        m = clean_kidney_mask(roi > thr)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_p), m * 255)
        n_empty += int(m.sum() == 0)
        done += 1
        if i % max(1, len(df) // a.previews) == 0:
            rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(str(root / "mask_preview" / f), rgb)
    print(f"\npredicted {done} kidney masks ({n_empty} empty = no kidney found) -> {root / 'masks'}")
    print(f"check a few overlays in {root / 'mask_preview'} (green = predicted kidney outline)")


if __name__ == "__main__":
    main()
