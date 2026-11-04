"""Check whether images of the uploaded stone dataset also appear in the Figshare dataset
(same slice, or a near-identical neighbouring slice). Important before reporting
cross-dataset results, because overlap would leak training images into the test set.

    python scripts/check_overlap.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io_utils import read_gray
from src.splits import load_splits, select
from src.utils import load_config, resolve


def feats(paths, size=64):
    out = []
    for p in tqdm(paths, leave=False):
        img, _ = read_gray(p)
        v = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
        v -= v.mean()
        out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup", type=float, default=0.99, help="correlation counted as the same image")
    ap.add_argument("--near", type=float, default=0.95, help="correlation counted as near-duplicate")
    ap.add_argument("--fold", type=int, default=0, help="stone_seg fold whose TRAINING images are compared")
    a = ap.parse_args()
    cfg = load_config()
    sroot, kroot = resolve(cfg["paths"]["stone_data"]), resolve(cfg["paths"]["kidney_data"])

    s_train = set(select(load_splits(sroot / "splits.json"), a.fold, "train"))
    s_paths = sorted((sroot / "image").iterdir())
    df = pd.read_csv(kroot / "labels.csv")
    folds = json.load(open(kroot / "splits.json", encoding="utf-8"))["folds"]
    df["fold"] = df["filename"].map(folds)

    print(f"reading {len(s_paths)} uploaded slices and {len(df)} Figshare slices ...")
    S = feats(s_paths)
    K = feats([kroot / "images" / f for f in df["filename"]])
    sim = K @ S.T                                  # correlation, Figshare x uploaded
    best = sim.max(1)
    best_id = [s_paths[i].stem for i in sim.argmax(1)]
    df["best_corr"] = best
    df["closest_uploaded"] = best_id
    df["closest_in_stone_train"] = [i in s_train for i in best_id]
    df.to_csv(resolve("outputs") / "overlap_check.csv", index=False)

    def report(sub, name):
        d = (sub.best_corr >= a.dup).sum()
        n = ((sub.best_corr >= a.near) & (sub.best_corr < a.dup)).sum()
        print(f"{name:28s} n={len(sub):5d} | identical (>={a.dup}): {d:5d} | near (>={a.near}): {n:5d} | "
              f"median corr {sub.best_corr.median():.3f}")

    print()
    report(df, "Figshare - all")
    report(df[df.fold == -1], "Figshare - hold-out")
    ho = df[df.fold == -1]
    report(ho[ho.label == 1], "  hold-out Stone+")
    report(ho[ho.label == 0], "  hold-out Stone-")
    leak = ho[(ho.best_corr >= a.near) & ho.closest_in_stone_train]
    print(f"\nhold-out slices matching a StoneUNet TRAINING slice (corr >= {a.near}): {len(leak)}")
    print("details saved to outputs/overlap_check.csv")
    if len(leak) == 0:
        print("=> no overlap: the cross-dataset results are a fair external test.")
    else:
        print("=> overlap found: re-run localize_figshare.py with --exclude_overlap to report clean numbers.")


if __name__ == "__main__":
    main()
