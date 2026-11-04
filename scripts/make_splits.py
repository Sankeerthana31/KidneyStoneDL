"""Create patient-level folds (5 CV folds + 1 hold-out) for a dataset.

    python scripts/make_splits.py --dataset kidney   # uses patient_id in labels.csv
    python scripts/make_splits.py --dataset stone    # uploaded dataset: contiguous-ID groups
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.splits import make_folds, contiguous_groups, save_splits
from src.utils import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kidney", "stone"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else cfg["common"]["seed"]
    n_folds = cfg["common"]["n_folds"]

    if a.dataset == "kidney":
        root = resolve(cfg["paths"]["kidney_data"])
        df = pd.read_csv(root / "labels.csv")
        # stratify on patient-level label (does the patient have any Stone+ slice?)
        pat_lab = df.groupby("patient_id")["label"].max()
        labels = df["patient_id"].map(pat_lab).tolist()
        folds = make_folds(df["filename"].tolist(), df["patient_id"].astype(str).tolist(), labels, n_folds, seed)
        lab = dict(zip(df["filename"], df["label"]))
        for f in sorted(set(folds.values())):
            names = [n for n, k in folds.items() if k == f]
            c = Counter(lab[n] for n in names)
            pats = df[df["filename"].isin(names)]["patient_id"].nunique()
            print(f"fold {f:>2} ({'hold-out' if f == -1 else 'cv'}): {len(names)} slices, "
                  f"Stone+ {c[1]}, Stone- {c[0]}, patients {pats}")
    else:
        root = resolve(cfg["paths"]["stone_data"])
        ids = [p.stem for p in (root / "image").iterdir() if p.is_file()]
        groups = contiguous_groups(ids, cfg["stone_seg"]["group_max"])
        folds = make_folds(ids, [groups[i] for i in ids], None, n_folds, seed)
        for f in sorted(set(folds.values())):
            n = sum(1 for k in folds.values() if k == f)
            g = len({groups[i] for i, k in folds.items() if k == f})
            print(f"fold {f:>2} ({'hold-out' if f == -1 else 'cv'}): {n} slices from {g} slice-groups")

    save_splits(folds, root / "splits.json")
    print("saved", root / "splits.json")


if __name__ == "__main__":
    main()
