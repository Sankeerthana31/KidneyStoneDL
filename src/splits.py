"""Patient-level splits: 6 non-overlapping folds -> 1 hold-out + 5 cross-validation folds
(as in the paper, Table 3.1). Fold value -1 means hold-out."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def make_folds(names, groups, labels=None, n_folds=5, seed=42):
    """Returns {name: fold} with folds 0..n_folds-1 for CV and -1 for hold-out."""
    n_split = n_folds + 1
    names, groups = np.asarray(names), np.asarray(groups)
    if labels is not None and len(set(labels)) > 1:
        kf = StratifiedGroupKFold(n_splits=n_split, shuffle=True, random_state=seed)
        it = kf.split(names, np.asarray(labels), groups)
    else:
        # GroupKFold has no shuffle in older sklearn -> shuffle group ids ourselves
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
        it = GroupKFold(n_splits=n_split).split(names, groups=[perm[g] for g in groups])
    out = {}
    for k, (_, test_idx) in enumerate(it):
        fold = -1 if k == n_split - 1 else k
        for i in test_idx:
            out[str(names[i])] = fold
    return out


def contiguous_groups(ids, max_group=40):
    """For datasets without patient IDs (uploaded stone dataset): consecutive slice numbers
    almost always come from the same scan, so we group runs of consecutive IDs (chunked to
    `max_group`) to keep neighbouring slices out of different splits (prevents leakage)."""
    nums = sorted(int(i) for i in ids)
    groups, gid, prev, run_len = {}, 0, None, 0
    for n in nums:
        if prev is None or n != prev + 1 or run_len >= max_group:
            gid += 1
            run_len = 0
        groups[str(n)] = f"g{gid}"
        prev, run_len = n, run_len + 1
    return groups


def save_splits(folds, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"folds": folds}, f, indent=1)


def load_splits(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["folds"]


def select(folds, fold, part):
    """part: 'train' (CV folds != fold), 'val' (== fold), 'holdout' (-1), 'trainval' (all CV)."""
    if part == "train":
        return [n for n, f in folds.items() if f != fold and f != -1]
    if part == "val":
        return [n for n, f in folds.items() if f == fold]
    if part == "holdout":
        return [n for n, f in folds.items() if f == -1]
    if part == "trainval":
        return [n for n, f in folds.items() if f != -1]
    raise ValueError(part)
