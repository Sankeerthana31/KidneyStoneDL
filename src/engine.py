"""Training / evaluation helpers shared by the scripts."""
import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .metrics import RegressionMeter, classification_metrics, StoneLocalisationMeter


def loader(ds, batch_size, shuffle, workers, limit=None, drop_last=False):
    if limit:
        idx = np.random.default_rng(0).permutation(len(ds))[:limit]
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last and len(ds) > batch_size,
                      persistent_workers=workers > 0)


def amp_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def to_dev(batch, device, keys):
    return [batch[k].to(device, non_blocking=True) if torch.is_tensor(batch[k]) else torch.as_tensor(batch[k]).to(device)
            for k in keys]


# ------------------------------------------------------------------ evaluation loops
@torch.no_grad()
def eval_regression(model, dl, device, roi_threshold=0.05):
    model.eval()
    meter = RegressionMeter(roi_threshold)
    for b in tqdm(dl, desc="eval-reg", leave=False):
        x, roi, mask = to_dev(b, device, ["image", "roi", "mask"])
        with amp_ctx(device):
            out = model(x)
        meter.update(out["roi"].float(), roi, mask)
    return meter.compute()


@torch.no_grad()
def eval_classifier(model, dl, device, input_key="roi", threshold=0.5):
    model.eval()
    ys, ps, names = [], [], []
    for b in tqdm(dl, desc="eval-clf", leave=False):
        (x,) = to_dev(b, device, [input_key])
        with amp_ctx(device):
            logits = model(x)
        ps += torch.softmax(logits.float(), 1)[:, 1].cpu().tolist()
        ys += [int(v) for v in b["label"]]
        names += list(b["name"])
    m = classification_metrics(ys, ps, threshold)
    return m, {"y": ys, "p": ps, "names": names}


@torch.no_grad()
def eval_cascade(model, dl, device, roi_threshold=0.05, threshold=0.5, with_reg=True):
    model.eval()
    meter = RegressionMeter(roi_threshold)
    ys, ps, names = [], [], []
    for b in tqdm(dl, desc="eval-cascade", leave=False):
        (x,) = to_dev(b, device, ["image"])
        with amp_ctx(device):
            out = model(x)
        if with_reg and "roi" in b:
            roi, mask = to_dev(b, device, ["roi", "mask"])
            meter.update(out["roi"].float(), roi, mask)
        ps += torch.softmax(out["logits"].float(), 1)[:, 1].cpu().tolist()
        ys += [int(v) for v in b["label"]]
        names += list(b["name"])
    m = classification_metrics(ys, ps, threshold)
    if with_reg and meter.n > 0:
        m.update({f"reg_{k}": v for k, v in meter.compute().items()})
    return m, {"y": ys, "p": ps, "names": names}


@torch.no_grad()
def eval_stone_seg(model, dl, device, threshold=0.5, min_area=2):
    model.eval()
    meter = StoneLocalisationMeter(min_area=min_area)
    for b in tqdm(dl, desc="eval-stone", leave=False):
        x, y = to_dev(b, device, ["image", "mask"])
        with amp_ctx(device):
            logits = model(x)
        pred = (torch.sigmoid(logits.float()) > threshold).cpu().numpy()[:, 0]
        gt = y.cpu().numpy()[:, 0]
        for p, g in zip(pred, gt):
            meter.update(p, g)
    return meter.compute()


def fmt(m, keys=None):
    keys = keys or [k for k, v in m.items() if isinstance(v, float)]
    return " ".join(f"{k}={m[k]:.4g}" for k in keys if k in m)
