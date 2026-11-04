"""Stage 2 - train ClfNet (dual-encoder stone classifier) on kidney ROIs.

    python scripts/train_clfnet.py --fold 0                       # ShN + SqN + CBAM (paper)
    python scripts/train_clfnet.py --fold 0 --deep_encoder resnet18
    python scripts/train_clfnet.py --fold 0 --input image         # full-slice baseline (no ROI)
    python scripts/train_clfnet.py --fold 0 --no_shallow          # single (deep) encoder ablation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from tqdm import tqdm

from src.datasets import KidneyROIDataset
from src.engine import loader, amp_ctx, make_scaler, eval_classifier, fmt
from src.metrics import classification_metrics
from src.models import ClfNet
from src.splits import load_splits, select
from src.utils import (load_config, merge_args, resolve, seed_everything, get_device, default_workers,
                       save_checkpoint, load_weights, save_json, CSVLogger)

KEYS = ["accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "kappa"]


def youden_threshold(y, p):
    """Decision threshold maximising sensitivity + specificity - 1 on the validation fold."""
    if len(set(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return float(min(max(t, 1e-4), 1 - 1e-4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--deep_encoder", choices=["squeezenet", "resnet18", "efficientnetv2s"])
    ap.add_argument("--input", choices=["roi", "image"])
    ap.add_argument("--no_shallow", action="store_true")
    ap.add_argument("--no_cbam", action="store_true")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg_all = load_config(a.config)
    cfg = merge_args(cfg_all["clfnet"], a)
    if a.no_shallow:
        cfg["use_shallow"] = False
    if a.no_cbam:
        cfg["use_cbam"] = False
    seed_everything(cfg_all["common"]["seed"])
    device = get_device(cfg_all["common"]["device"])
    root = resolve(cfg_all["paths"]["kidney_data"])
    tag = f"{'ShN+' if cfg['use_shallow'] else ''}{cfg['deep_encoder']}{'+CBAM' if cfg['use_cbam'] else ''}_{cfg['input']}"
    out = Path(a.out) if a.out else resolve(cfg_all["paths"]["outputs"]) / "clfnet" / tag / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)

    folds = load_splits(root / "splits.json")
    S = cfg["img_size"]
    tr = KidneyROIDataset(root, select(folds, a.fold, "train"), S, augment=True)
    va = KidneyROIDataset(root, select(folds, a.fold, "val"), S)
    ho = KidneyROIDataset(root, select(folds, a.fold, "holdout"), S)
    print(f"[{tag}] device={device} train={len(tr)} val={len(va)} holdout={len(ho)}")
    dl_tr = loader(tr, cfg["batch_size"], True, a.workers, a.limit, drop_last=True)
    dl_va = loader(va, cfg["batch_size"], False, a.workers, a.limit)
    dl_ho = loader(ho, cfg["batch_size"], False, a.workers, a.limit)

    model = ClfNet(deep_encoder=cfg["deep_encoder"], use_shallow=cfg["use_shallow"],
                   use_cbam=cfg["use_cbam"], dropout=cfg["dropout"], input_size=S).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = make_scaler(device)
    log = CSVLogger(out / "log.csv")
    best, key, best_thr = -1.0, cfg["input"], 0.5

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for b in tqdm(dl_tr, desc=f"epoch {ep}/{cfg['epochs']}", leave=False):
            x, y = b[key].to(device), b["label"].to(device)
            with amp_ctx(device):
                logits = model(x)
            loss = ce(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        m, raw = eval_classifier(model, dl_va, device, key)
        thr = youden_threshold(raw["y"], raw["p"])
        mt = classification_metrics(raw["y"], raw["p"], thr)
        log.log({"epoch": ep, "train_loss": run / max(n, 1), "threshold": thr,
                 **{f"val_{k}": mt[k] for k in KEYS}, "lr": opt.param_groups[0]["lr"]})
        print(f"[{ep:03d}] loss={run / max(n, 1):.4f} | val (thr {thr:.2f}) {fmt(mt, KEYS)}")
        score = m["roc_auc"] if m["roc_auc"] == m["roc_auc"] else mt["f1"]
        if score > best:
            best, best_thr = score, thr
            save_checkpoint(model, out / "best.pt", {"epoch": ep, "val": mt, "cfg": cfg, "threshold": thr})
        save_checkpoint(model, out / "last.pt", {"epoch": ep, "cfg": cfg, "threshold": thr})

    extra = load_weights(model, out / "best.pt", device)
    best_thr = extra.get("threshold", best_thr)
    _, raw_va = eval_classifier(model, dl_va, device, key)
    res = {"threshold": best_thr, "best_epoch": extra.get("epoch"),
           "val": classification_metrics(raw_va["y"], raw_va["p"], best_thr)}
    if len(ho):
        _, raw_ho = eval_classifier(model, dl_ho, device, key)
        res["holdout"] = classification_metrics(raw_ho["y"], raw_ho["p"], best_thr)
        res["holdout_thr0.5"] = classification_metrics(raw_ho["y"], raw_ho["p"], 0.5)
        save_json(raw_ho, out / "holdout_predictions.json")
    save_json(res, out / "metrics.json")
    print(f"best epoch {res['best_epoch']} | decision threshold {best_thr:.3f} (chosen on validation)")
    print("best val :", fmt(res["val"], KEYS))
    if "holdout" in res:
        print("hold-out :", fmt(res["holdout"], KEYS))


if __name__ == "__main__":
    main()
