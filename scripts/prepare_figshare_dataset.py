"""Prepare the paper's Figshare dataset (doi:10.6084/m9.figshare.31156783) for this project.

Downloaded layout:
    Dataset/dataset_integrated_task/
        cross_validation/Test/Fold_1..Fold_5/{Positive,Negative}/*.jpg   <- each slice once
        cross_validation/Training/Fold_k/...                             <- copies of the other folds (ignored)
        holdout/{Positive,Negative}/*.jpg

Output (the paper's own patient-level folds are kept):
    data/kidney/images/<name>.png
    data/kidney/labels.csv   filename,label,patient_id   (1 = Positive/Stone+, 0 = Negative/Stone-)
    data/kidney/splits.json  Test Fold_k -> fold k-1 ; holdout -> -1

    python scripts/prepare_figshare_dataset.py --raw_dir data/raw_kidney
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io_utils import read_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw_kidney")
    ap.add_argument("--out_dir", default="data/kidney")
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    hits = list(raw.rglob("cross_validation"))
    if not hits:
        sys.exit(f"Could not find a 'cross_validation' folder inside {raw.resolve()}")
    base = hits[0].parent                      # .../dataset_integrated_task
    out = Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    sources = [(f"F{k}", k - 1, base / "cross_validation" / "Test" / f"Fold_{k}") for k in range(1, 6)]
    sources.append(("HO", -1, base / "holdout"))

    rows, folds = [], {}
    for tag, fold, d in sources:
        for cls, label in (("Positive", 1), ("Negative", 0)):
            files = sorted((d / cls).glob("*.jpg")) + sorted((d / cls).glob("*.png"))
            for p in tqdm(files, desc=f"{tag} {cls}", leave=False):
                name = f"{tag}_{'pos' if label else 'neg'}_{p.stem}.png"
                img, _ = read_gray(p)
                cv2.imwrite(str(out / "images" / name), (img * 255).round().astype(np.uint8))
                # patient IDs are not released; the authors' folds are already patient-level,
                # so the fold tag is used as the group id.
                rows.append({"filename": name, "label": label, "patient_id": tag})
                folds[name] = fold

    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    with open(out / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"folds": folds}, f, indent=1)

    print("\nfold        Stone+  Stone-   total")
    for tag, fold, _ in sources:
        s = df[df["patient_id"] == tag]
        print(f"{'hold-out' if fold == -1 else f'fold {fold}':10s} {int(s.label.sum()):7d} {int((s.label == 0).sum()):7d} {len(s):7d}")
    print(f"{'TOTAL':10s} {int(df.label.sum()):7d} {int((df.label == 0).sum()):7d} {len(df):7d}")
    print(f"\nsaved images, labels.csv and splits.json to {out.resolve()}")
    print("note: no kidney masks in this release -> train the classifier with --input image")


if __name__ == "__main__":
    main()
