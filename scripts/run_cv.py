"""Run 5-fold cross-validation for one training script and report mean ± std (CV5 and hold-out).

    python scripts/run_cv.py --script train_cascade.py
    python scripts/run_cv.py --script train_stone_seg.py --extra "--epochs 40"
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, choices=["train_i2iregnet.py", "train_clfnet.py",
                                                        "train_cascade.py", "train_stone_seg.py"])
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--extra", default="", help="extra args passed to the training script")
    ap.add_argument("--skip_training", action="store_true", help="only aggregate existing metrics.json")
    a = ap.parse_args()
    folds = [int(f) for f in a.folds.split(",")]
    results = []
    for k in folds:
        out = ROOT / "outputs" / "cv" / a.script.replace(".py", "") / f"fold{k}"
        if not a.skip_training:
            cmd = [sys.executable, str(ROOT / "scripts" / a.script), "--fold", str(k), "--out", str(out)] + shlex.split(a.extra)
            print(">>", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=ROOT)
        with open(out / "metrics.json", encoding="utf-8") as f:
            results.append(json.load(f))

    print("\n================ summary ================")
    for part in ("val", "holdout"):
        rows = [r[part] for r in results if part in r and r[part]]
        if not rows:
            continue
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not k.startswith(("tp", "tn", "fp", "fn", "stone_tp", "stone_fp", "stone_fn"))]
        print(f"\n{('CV5 (validation folds)' if part == 'val' else 'Hold-out')}:")
        summary = {}
        for k in keys:
            v = np.array([r[k] for r in rows], dtype=float)
            summary[k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
            print(f"  {k:18s} {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
        with open(ROOT / "outputs" / "cv" / a.script.replace(".py", "") / f"summary_{part}.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
