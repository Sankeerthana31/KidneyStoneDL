"""Convert the paper's dataset (or any CT + kidney-mask dataset) into the project layout:

    data/kidney/images/<name>.png   8-bit CT slice
    data/kidney/masks/<name>.png    binary kidney mask (0/255)
    data/kidney/labels.csv          filename,label,patient_id   (label 1 = Stone+, 0 = Stone-)

The paper's dataset: Öksüz et al., I2IReg-ClfNet, Figshare doi:10.6084/m9.figshare.31156783.
Its exact folder layout may differ from release to release, so this script is tolerant:

  * class   : decided from the path - any folder containing 'stone+', 'stone_pos', 'positive',
              'stone' (without 'non'/'no')  -> 1 ;  'stone-', 'normal', 'healthy', 'negative',
              'non' -> 0.  Or give --labels_csv with columns filename,label[,patient_id].
  * masks   : any file whose path contains 'mask', 'label', 'gt', 'roi' or 'seg'.
              Masks may be binary, ROI images (mask x image) or filled green contours - all
              are binarised (>0 after a small threshold).
  * pairing : by file stem after removing suffixes such as _mask, -mask, _gt, _roi, _label.
  * patient : regex --patient_regex on the relative path; falls back to the parent folder.

Example:
    python scripts/prepare_kidney_dataset.py --raw_dir D:/downloads/I2IReg_dataset
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io_utils import IMG_EXTS, read_gray

SUFFIX_RE = re.compile(r"[_\-\s]?(mask|label|gt|roi|seg|annotation|annot)$", re.I)


def classify(rel: str):
    s = rel.lower().replace("\\", "/")
    if any(t in s for t in ("stone-", "stone_neg", "stone −", "normal", "healthy", "negative", "nonstone", "non_stone", "no_stone", "/neg")):
        return 0
    if any(t in s for t in ("stone+", "stone_pos", "positive", "stone", "/pos")):
        return 1
    return None


TOKEN_RE = re.compile(r"(?:^|[^a-z])(mask|masks|label|labels|gt|roi|rois|seg|annot\w*)(?:$|[^a-z])")
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs", "ct", "slices", "scans", "png", "dicom", "dcm", "original"}


def is_mask(rel: str):
    parts = rel.lower().replace("\\", "/").split("/")
    return any(TOKEN_RE.search(p) for p in parts)


def norm_dir(rel_dir: str):
    """Folder path with image/mask folder names removed -> used to pair images with masks."""
    parts = rel_dir.lower().replace("\\", "/").split("/")
    return "/".join(p for p in parts if p not in IMAGE_DIR_NAMES and not TOKEN_RE.search(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/kidney")
    ap.add_argument("--labels_csv", default=None, help="optional csv: filename,label[,patient_id]")
    ap.add_argument("--patient_regex", default=r"(?:patient|hasta|case|pat|p)[_\- ]?(\d+)")
    ap.add_argument("--mask_threshold", type=float, default=0.04)
    ap.add_argument("--dicom_level", type=float, default=40)
    ap.add_argument("--dicom_width", type=float, default=400)
    a = ap.parse_args()

    raw = Path(a.raw_dir)
    out = Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    files = [p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs, masks, masks_by_stem = {}, {}, {}
    for p in files:
        rel = str(p.relative_to(raw))
        stem = SUFFIX_RE.sub("", p.stem)
        key_dir = str(p.parent.relative_to(raw))
        if is_mask(rel):
            masks[(norm_dir(key_dir), stem)] = p
            masks_by_stem.setdefault(stem, []).append(p)
        else:
            imgs[(classify(rel), stem, key_dir)] = p
    print(f"found {len(imgs)} images and {len(masks)} masks")

    ext_labels = None
    if a.labels_csv:
        ext = pd.read_csv(a.labels_csv)
        ext_labels = {Path(str(r.filename)).stem: r for r in ext.itertuples()}

    pre = re.compile(a.patient_regex, re.I)
    rows, missing_mask, missing_label = [], 0, 0
    for (cls, stem, key_dir), p in tqdm(sorted(imgs.items(), key=lambda kv: str(kv[1]))):
        rel = str(p.relative_to(raw))
        label, pid = cls, None
        if ext_labels is not None and stem in ext_labels:
            r = ext_labels[stem]
            label = int(r.label)
            pid = getattr(r, "patient_id", None)
        if label is None:
            missing_label += 1
            continue
        if pid is None:
            m = pre.search(rel)
            pid = m.group(0) if m else Path(key_dir).name or "unknown"
        mp = masks.get((norm_dir(key_dir), stem))
        if mp is None and len(masks_by_stem.get(stem, [])) == 1:   # unique file name anywhere
            mp = masks_by_stem[stem][0]
        name = re.sub(r"[^\w\-]+", "_", f"{'S' if label else 'N'}_{pid}_{stem}") + ".png"
        img, _ = read_gray(p, (a.dicom_level, a.dicom_width))
        cv2.imwrite(str(out / "images" / name), (img * 255).round().astype(np.uint8))
        if mp is not None:
            m, _ = read_gray(mp)
            if m.ndim == 3:
                m = m.max(axis=2)
            binm = (m > a.mask_threshold).astype(np.uint8)
            # filled contours: if mask is only an outline, fill it
            cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(binm)
            cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
            if filled.sum() > 2 * binm.sum():
                binm = filled
            if binm.shape != img.shape:
                binm = cv2.resize(binm, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / "masks" / name), binm * 255)
        else:
            missing_mask += 1
        rows.append({"filename": name, "label": int(label), "patient_id": str(pid)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    print(df["label"].value_counts().rename({1: "Stone+", 0: "Stone-"}))
    print(f"patients: {df['patient_id'].nunique()} | images without mask: {missing_mask} | "
          f"skipped (class unknown): {missing_label}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
