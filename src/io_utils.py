"""Reading CT slices from PNG/JPG/TIF/BMP or DICOM into a float32 [0,1] grayscale array."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".dicom"}


def apply_window(hu, level=40.0, width=400.0):
    lo, hi = level - width / 2.0, level + width / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def read_dicom(path, level=40.0, width=400.0):
    import pydicom  # optional dependency
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    spacing = None
    if hasattr(ds, "PixelSpacing"):
        spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
    return apply_window(arr, level, width), spacing


def read_gray(path, window=(40.0, 400.0)):
    """Returns (img float32 HxW in [0,1], pixel_spacing_mm or None)."""
    path = Path(path)
    if path.suffix.lower() in (".dcm", ".dicom"):
        return read_dicom(path, *window)
    im = Image.open(path)
    if im.mode == "P":
        im = im.convert("L")
    arr = np.array(im)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
    arr = arr.astype(np.float32)
    if arr.max() > 255:               # 16-bit export
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1.0)
    else:
        arr = arr / 255.0
    return arr, None


def read_mask(path, threshold=0.5):
    """Binary mask from any image (palette / RGB green contours filled / 0-255 / 0-1)."""
    arr, _ = read_gray(path)          # already scaled to [0,1]
    return (arr >= threshold).astype(np.float32)


def resize(img, size, interp=cv2.INTER_LINEAR):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=interp)


def to_3ch_tensor(img):
    """HxW [0,1] numpy -> (3,H,W) float tensor."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).repeat(3, 1, 1)


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
