"""Train StoneUNet for kidney stone localisation on the uploaded dataset (image/ + label/).

    python scripts/make_splits.py --dataset stone      # once
    python scripts/train_stone_seg.py --fold 0
Training uses 256x256 stone-centred crops (stones are tiny); validation runs on full 512x512 slices.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from src.datasets import StoneSegDataset
from src.engine import loader, amp_ctx, make_scaler, eval_stone_seg, fmt
from src.models import StoneUNet, FocalDiceLoss
from src.splits import load_splits, select
from src.utils import (load_config, merge_args, resolve, seed_everything, get_device, default_workers,
                       save_checkpoint, load_weights, save_json, CSVLogger)

KEYS = ["dice", "iou", "dice_global", "stone_precision", "stone_recall", "stone_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--encoder", choices=["resnet34", "resnet18"])
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg_all = load_config(a.config)
    cfg = merge_args(cfg_all["stone_seg"], a)
    seed_everything(cfg_all["common"]["seed"])
    device = get_device(cfg_all["common"]["device"])
    root = resolve(cfg_all["paths"]["stone_data"])
    out = Path(a.out) if a.out else resolve(cfg_all["paths"]["outputs"]) / "stone_seg" / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)

    folds = load_splits(root / "splits.json")
    kw = dict(crop=cfg["crop"], pos_crop_prob=cfg["pos_crop_prob"], full_size=cfg["full_size"])
    tr = StoneSegDataset(root, select(folds, a.fold, "train"), train=True, **kw)
    va = StoneSegDataset(root, select(folds, a.fold, "val"), train=False, **kw)
    ho = StoneSegDataset(root, select(folds, a.fold, "holdout"), train=False, **kw)
    print(f"device={device} train={len(tr)} val={len(va)} holdout={len(ho)}")
    dl_tr = loader(tr, cfg["batch_size"], True, a.workers, a.limit, drop_last=True)
    dl_va = loader(va, max(1, cfg["batch_size"] // 2), False, a.workers, a.limit)
    dl_ho = loader(ho, max(1, cfg["batch_size"] // 2), False, a.workers, a.limit)

    model = StoneUNet(cfg["encoder"], pretrained=True).to(device)
    print(f"StoneUNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * max(len(dl_tr), 1),
                                                pct_start=0.1)
    crit = FocalDiceLoss()
    scaler = make_scaler(device)
    log = CSVLogger(out / "log.csv")
    best = -1.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        run, n = 0.0, 0
        for b in tqdm(dl_tr, desc=f"epoch {ep}/{cfg['epochs']}", leave=False):
            x, y = b["image"].to(device), b["mask"].to(device)
            with amp_ctx(device):
                logits = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        if ep % a.val_every and ep != cfg["epochs"]:
            print(f"[{ep:03d}] loss={run / max(n, 1):.4f}")
            continue
        m = eval_stone_seg(model, dl_va, device, cfg["threshold"], cfg["min_area"])
        log.log({"epoch": ep, "train_loss": run / max(n, 1), **{f"val_{k}": m[k] for k in KEYS}})
        print(f"[{ep:03d}] loss={run / max(n, 1):.4f} | val {fmt(m, KEYS)}")
        score = m["dice"] + m["stone_f1"]
        if score > best:
            best = score
            save_checkpoint(model, out / "best.pt", {"epoch": ep, "val": m, "cfg": cfg})
        save_checkpoint(model, out / "last.pt", {"epoch": ep, "cfg": cfg})

    load_weights(model, out / "best.pt", device)
    res = {"val": eval_stone_seg(model, dl_va, device, cfg["threshold"], cfg["min_area"])}
    if len(ho):
        res["holdout"] = eval_stone_seg(model, dl_ho, device, cfg["threshold"], cfg["min_area"])
    save_json(res, out / "metrics.json")
    print("best val :", fmt(res["val"], KEYS))
    if "holdout" in res:
        print("hold-out :", fmt(res["holdout"], KEYS))


if __name__ == "__main__":
    main()
