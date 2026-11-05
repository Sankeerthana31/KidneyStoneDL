"""Build a 2D kidney-localisation dataset from the TotalSegmentator dataset (CC BY 4.0).

Download (open, no request needed):
    Small subset (102 CT scans, 3.2 GB): https://zenodo.org/records/10047263
    Full dataset (1228 CT scans, 23.6 GB): https://zenodo.org/records/10047292
Extract it e.g. to data/raw_totalseg/ . Each scan folder looks like
    s0011/ct.nii.gz
    s0011/segmentations/kidney_left.nii.gz , kidney_right.nii.gz , ...

This script:
  * reorients every scan to the usual radiological axial view (anterior up,
    patient's right on the image left) - same view as the Figshare / uploaded slices
  * applies an abdominal window (default L=40, W=400) -> 8-bit PNG, resized to 512x512
  * keeps axial slices that contain kidney (every --step-th slice) plus some slices
    just above/below the kidneys with an empty mask (teaches "no kidney here")
  * writes  data/kidney_masks/images/*.png, masks/*.png, labels.csv (patient_id = scan id)
            and a few preview images in data/kidney_masks/preview/ to check orientation

    pip install nibabel
    python scripts/prepare_totalseg_kidney.py --raw_dir data/raw_totalseg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def to_view(slice_xy):
    """Canonical (RAS) array slice [x, y] -> radiological display [row, col]:
    rows: anterior at the top ; cols: patient's right on the image left."""
    return np.flipud(slice_xy.T)[:, ::-1]


def window(hu, level, width):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw_totalseg")
    ap.add_argument("--out_dir", default="data/kidney_masks")
    ap.add_argument("--level", type=float, default=40)
    ap.add_argument("--width", type=float, default=400)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--step", type=int, default=2, help="take every n-th kidney slice")
    ap.add_argument("--neg_margin", type=int, default=12, help="slices above/below kidneys used as empty-mask slices")
    ap.add_argument("--neg_per_scan", type=int, default=6)
    ap.add_argument("--min_kidney_px", type=int, default=150, help="ignore slices with tiny kidney tips")
    ap.add_argument("--max_scans", type=int, default=None)
    a = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        sys.exit("nibabel is missing - run:  pip install nibabel")

    raw, out = Path(a.raw_dir), Path(a.out_dir)
    for sub in ("images", "masks", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cts = sorted(raw.rglob("ct.nii.gz"))
    if a.max_scans:
        cts = cts[: a.max_scans]
    print(f"found {len(cts)} CT scans")
    rng = np.random.default_rng(0)
    rows, used, previews = [], 0, 0

    for ct_path in tqdm(cts):
        sid = ct_path.parent.name
        seg = ct_path.parent / "segmentations"
        kl, kr = seg / "kidney_left.nii.gz", seg / "kidney_right.nii.gz"
        if not (kl.exists() and kr.exists()):
            continue
        mk = nib.as_closest_canonical(nib.load(str(kl))).get_fdata(dtype=np.float32) > 0.5
        mk |= nib.as_closest_canonical(nib.load(str(kr))).get_fdata(dtype=np.float32) > 0.5
        per_slice = mk.sum(axis=(0, 1))
        zs = np.nonzero(per_slice >= a.min_kidney_px)[0]
        if len(zs) == 0:
            continue                                   # scan does not cover the kidneys
        ct = nib.as_closest_canonical(nib.load(str(ct_path))).get_fdata(dtype=np.float32)
        sx, sy = nib.as_closest_canonical(nib.load(str(ct_path))).header.get_zooms()[:2]
        used += 1

        pos = list(zs[:: a.step])
        lo, hi = int(zs.min()), int(zs.max())
        cand = [z for z in range(max(0, lo - a.neg_margin), lo) if per_slice[z] == 0] + \
               [z for z in range(hi + 1, min(ct.shape[2], hi + 1 + a.neg_margin)) if per_slice[z] == 0]
        neg = list(rng.choice(cand, size=min(a.neg_per_scan, len(cand)), replace=False)) if cand else []

        for z in pos + neg:
            img = to_view(window(ct[:, :, z], a.level, a.width))
            m = to_view(mk[:, :, z].astype(np.uint8))
            # make pixels square if spacing differs, then fit into a square canvas
            h, w = img.shape
            if abs(sx - sy) > 1e-3:
                w = int(round(w * sx / sy))
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            side = max(h, w)
            canvas_i = np.zeros((side, side), np.float32)
            canvas_m = np.zeros((side, side), np.uint8)
            y0, x0 = (side - h) // 2, (side - w) // 2
            canvas_i[y0:y0 + h, x0:x0 + w] = img
            canvas_m[y0:y0 + h, x0:x0 + w] = m
            img8 = cv2.resize((canvas_i * 255).astype(np.uint8), (a.size, a.size), interpolation=cv2.INTER_AREA)
            m8 = cv2.resize(canvas_m * 255, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
            name = f"{sid}_z{int(z):04d}.png"
            cv2.imwrite(str(out / "images" / name), img8)
            cv2.imwrite(str(out / "masks" / name), m8)
            rows.append({"filename": name, "label": int(m8.max() > 0), "patient_id": sid})
            if previews < 12 and z == pos[len(pos) // 2]:
                rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                cnts, _ = cv2.findContours((m8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 2)
                cv2.imwrite(str(out / "preview" / f"preview_{name}"), rgb)
                previews += 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    print(f"\nscans with kidneys: {used} | slices: {len(df)} "
          f"(with kidney {int(df.label.sum())}, without {int((df.label == 0).sum())})")
    print(f"saved to {out.resolve()}")
    print("CHECK the images in preview/: anterior (belly) at the top, liver on the image LEFT,")
    print("kidneys (green) on both sides of the spine - the same view as the Figshare slices.")


if __name__ == "__main__":
    main()
