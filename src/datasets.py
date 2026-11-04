"""PyTorch datasets.

1. KidneyROIDataset  - paper dataset (I2IReg-ClfNet, Figshare 10.6084/m9.figshare.31156783):
                       CT slice + kidney mask + Stone+/Stone- label  -> kidney localisation
                       (image-to-image regression) and stone detection (classification).
     data/kidney/images/<name>.png
     data/kidney/masks/<name>.png          (binary kidney mask, same file name)
     data/kidney/labels.csv                (filename,label,patient_id)   label: 1 = Stone+, 0 = Stone-
2. StoneSegDataset   - uploaded dataset: CT slice + kidney-stone mask -> stone localisation.
     data/stone_seg/image/<id>.tif
     data/stone_seg/label/<id>.tif
3. CTKidneyExternal  - CT KIDNEY DATASET (Islam et al. 2022) used by the paper for external tests.
     data/ct_kidney_external/Normal/*.jpg , data/ct_kidney_external/Stone/*.jpg
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import read_gray, read_mask, resize, to_3ch_tensor, list_images


# --------------------------------------------------------------------------- augmentation
def _rng():
    """Fresh generator seeded from torch, so each DataLoader worker gets different augmentations."""
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))


def random_affine(img, masks, rng, max_rot=10, scale=(0.9, 1.1), shift=0.05, hflip=True):
    h, w = img.shape
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    tx, ty = rng.uniform(-shift, shift) * w, rng.uniform(-shift, shift) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    masks = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for m in masks]
    if hflip and rng.random() < 0.5:
        img = img[:, ::-1]
        masks = [m[:, ::-1] for m in masks]
    return np.ascontiguousarray(img), [np.ascontiguousarray(m) for m in masks]


def intensity_jitter(img, rng, b=0.05, c=0.1, gamma=0.15):
    img = img * rng.uniform(1 - c, 1 + c) + rng.uniform(-b, b)
    img = np.clip(img, 0, 1) ** rng.uniform(1 - gamma, 1 + gamma)
    return img.astype(np.float32)


# --------------------------------------------------------------------------- 1. paper dataset
class KidneyROIDataset(Dataset):
    """Returns dict(image (3,S,S) [0,1], mask (1,S,S), roi (3,S,S) = mask*image, label, name)."""

    def __init__(self, root, names=None, size=224, augment=False, seed=0):
        self.root = Path(root)
        df = pd.read_csv(self.root / "labels.csv")
        if names is not None:
            df = df[df["filename"].isin(set(names))]
        self.df = df.reset_index(drop=True)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.df)

    def labels(self):
        return self.df["label"].astype(int).tolist()

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img, _ = read_gray(self.root / "images" / row["filename"])
        mpath = self.root / "masks" / row["filename"]
        mask = read_mask(mpath) if mpath.exists() else np.zeros_like(img)
        img = resize(img, self.size)
        mask = resize(mask, self.size, cv2.INTER_NEAREST)
        if self.augment:
            rng = _rng()
            img, (mask,) = random_affine(img, [mask], rng)
            img = intensity_jitter(img, rng)
        x = to_3ch_tensor(img)
        m = torch.from_numpy(mask).float().unsqueeze(0)
        return {"image": x, "mask": m, "roi": x * m, "label": int(row["label"]), "name": row["filename"]}


# --------------------------------------------------------------------------- 2. uploaded dataset
class StoneSegDataset(Dataset):
    """Kidney-stone segmentation (localisation). Training uses stone-centred random crops
    because stones occupy ~0.01-0.3% of the slice; validation/test use the full slice."""

    def __init__(self, root, ids=None, train=False, crop=256, pos_crop_prob=0.7, full_size=512, seed=0):
        self.root = Path(root)
        all_ids = sorted((p.stem for p in (self.root / "image").iterdir() if p.is_file()),
                         key=lambda s: (len(s), s))
        keep = None if ids is None else set(map(str, ids))
        self.ids = [i for i in all_ids if keep is None or i in keep]
        self.ext = {p.stem: p.name for p in (self.root / "image").iterdir()}
        self.lext = {p.stem: p.name for p in (self.root / "label").iterdir()}
        self.train, self.crop, self.pos_p, self.full = train, crop, pos_crop_prob, full_size

    def __len__(self):
        return len(self.ids)

    def _load(self, sid):
        img, _ = read_gray(self.root / "image" / self.ext[sid])
        mask = read_mask(self.root / "label" / self.lext[sid])
        img = resize(img, self.full)
        mask = resize(mask, self.full, cv2.INTER_NEAREST)
        return img, mask

    def _random_crop(self, img, mask, rng):
        c, H, W = self.crop, *img.shape
        ys, xs = np.nonzero(mask)
        if len(ys) and rng.random() < self.pos_p:
            k = rng.integers(len(ys))
            cy = ys[k] + rng.integers(-c // 3, c // 3 + 1)
            cx = xs[k] + rng.integers(-c // 3, c // 3 + 1)
        else:
            cy, cx = rng.integers(c // 2, H - c // 2 + 1), rng.integers(c // 2, W - c // 2 + 1)
        y0 = int(np.clip(cy - c // 2, 0, H - c))
        x0 = int(np.clip(cx - c // 2, 0, W - c))
        return img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]

    def __getitem__(self, i):
        sid = self.ids[i]
        img, mask = self._load(sid)
        if self.train:
            rng = _rng()
            img, (mask,) = random_affine(img, [mask], rng, max_rot=15, scale=(0.85, 1.15))
            img = intensity_jitter(img, rng)
            img, mask = self._random_crop(img, mask, rng)
        return {"image": to_3ch_tensor(img), "mask": torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
                "name": sid}


# --------------------------------------------------------------------------- 3. external dataset
class CTKidneyExternal(Dataset):
    """CT KIDNEY DATASET (Normal / Stone folders). label 1 = Stone."""

    def __init__(self, root, size=224, files=None):
        root = Path(root)
        if files is None:
            files = []
            for cls, lab in (("Normal", 0), ("Stone", 1)):
                files += [(p, lab) for p in list_images(root / cls)]
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def labels(self):
        return [l for _, l in self.files]

    def __getitem__(self, i):
        p, lab = self.files[i]
        img, _ = read_gray(p)
        img = resize(img, self.size)
        return {"image": to_3ch_tensor(img), "label": lab, "name": str(p)}
