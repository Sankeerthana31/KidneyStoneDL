"""Stage 1 - train I2IRegNet (kidney ROI localisation by image-to-image regression).

    python scripts/train_i2iregnet.py --fold 0
Loss: half-MSE between ROI = mask*image and ROI_hat (Eq. 2.1). Adam, lr 1e-4, 50 epochs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from src.datasets import KidneyROIDataset
from src.engine import loader, amp_ctx, make_scaler, eval_regression, fmt
from src.models import I2IRegNet, half_mse
from src.splits import load_splits, select
from src.utils import (load_config, merge_args, resolve, seed_everything, get_device, default_workers,
                       save_checkpoint, load_weights, save_json, CSVLogger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N samples per split")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg_all = load_config(a.config)
    cfg = merge_args(cfg_all["i2iregnet"], a)
    seed_everything(cfg_all["common"]["seed"])
    device = get_device(cfg_all["common"]["device"])
    root = resolve(cfg_all["paths"]["kidney_data"])
    out = Path(a.out) if a.out else resolve(cfg_all["paths"]["outputs"]) / "i2iregnet" / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)

    folds = load_splits(root / "splits.json")
    S = cfg["img_size"]
    tr = KidneyROIDataset(root, select(folds, a.fold, "train"), S, augment=True)
    va = KidneyROIDataset(root, select(folds, a.fold, "val"), S)
    ho = KidneyROIDataset(root, select(folds, a.fold, "holdout"), S)
    print(f"device={device} train={len(tr)} val={len(va)} holdout={len(ho)}")
    dl_tr = loader(tr, cfg["batch_size"], True, a.workers, a.limit, drop_last=True)
    dl_va = loader(va, cfg["batch_size"], False, a.workers, a.limit)
    dl_ho = loader(ho, cfg["batch_size"], False, a.workers, a.limit)

    model = I2IRegNet(pretrained=True).to(device)
    print(f"I2IRegNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    scaler = make_scaler(device)
    log = CSVLogger(out / "log.csv")
    best = -1.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for b in tqdm(dl_tr, desc=f"epoch {ep}/{cfg['epochs']}", leave=False):
            x, roi = b["image"].to(device), b["roi"].to(device)
            with amp_ctx(device):
                pred = model(x)["roi"]
            loss = half_mse(pred.float(), roi)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            n += x.size(0)
        m = eval_regression(model, dl_va, device, cfg["roi_threshold"])
        score = m["iou"] if m["iou"] == m["iou"] else -m["mse"]   # IoU, or -MSE if IoU undefined
        sched.step(score)
        log.log({"epoch": ep, "train_loss": run / max(n, 1), **{f"val_{k}": v for k, v in m.items()},
                 "lr": opt.param_groups[0]["lr"]})
        print(f"[{ep:03d}] loss={run / max(n, 1):.3e} | val {fmt(m, ['mse', 'rmse', 'mae', 'iou', 'dice'])}")
        if score > best or best == -1.0:
            best = score
            save_checkpoint(model, out / "best.pt", {"epoch": ep, "val": m, "cfg": cfg})
        save_checkpoint(model, out / "last.pt", {"epoch": ep})

    load_weights(model, out / "best.pt", device)
    res = {"val": eval_regression(model, dl_va, device, cfg["roi_threshold"]),
           "holdout": eval_regression(model, dl_ho, device, cfg["roi_threshold"]) if len(ho) else {}}
    save_json(res, out / "metrics.json")
    print("best val :", fmt(res["val"], ["mse", "rmse", "mae", "iou", "dice"]))
    if res["holdout"]:
        print("hold-out :", fmt(res["holdout"], ["mse", "rmse", "mae", "iou", "dice"]))


if __name__ == "__main__":
    main()
