"""Joint training of the cascaded I2IReg-ClfNet (kidney localisation + stone detection).

    python scripts/train_cascade.py --fold 0
    # optional warm start from the stage-wise models:
    python scripts/train_cascade.py --fold 0 --init_reg outputs/i2iregnet/fold0/best.pt \
                                    --init_clf outputs/clfnet/ShN+squeezenet+CBAM_roi/fold0/best.pt
L_total = 0.8 * L_reg + 0.2 * L_clf (Eq. 2.2), Adam lr 1e-4 with dynamic (plateau) LR, 50 epochs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from src.datasets import KidneyROIDataset
from src.engine import loader, amp_ctx, make_scaler, eval_cascade, fmt
from src.models import I2IRegClfNet, CascadeLoss
from src.splits import load_splits, select
from src.utils import (load_config, merge_args, resolve, seed_everything, get_device, default_workers,
                       save_checkpoint, load_weights, save_json, CSVLogger)

KEYS = ["accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "kappa",
        "reg_mse", "reg_rmse", "reg_mae", "reg_iou", "reg_dice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--w_reg", type=float)
    ap.add_argument("--w_clf", type=float)
    ap.add_argument("--deep_encoder", choices=["squeezenet", "resnet18", "efficientnetv2s"])
    ap.add_argument("--init_reg", default=None)
    ap.add_argument("--init_clf", default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg_all = load_config(a.config)
    cfg = merge_args(cfg_all["cascade"], a)
    seed_everything(cfg_all["common"]["seed"])
    device = get_device(cfg_all["common"]["device"])
    root = resolve(cfg_all["paths"]["kidney_data"])
    out = Path(a.out) if a.out else resolve(cfg_all["paths"]["outputs"]) / "cascade" / f"fold{a.fold}"
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

    model = I2IRegClfNet(deep_encoder=cfg["deep_encoder"], use_shallow=cfg["use_shallow"],
                         use_cbam=cfg["use_cbam"], dropout=cfg["dropout"]).to(device)
    if a.init_reg:
        load_weights(model.reg, a.init_reg, device)
        print("loaded I2IRegNet weights from", a.init_reg)
    if a.init_clf:
        load_weights(model.clf, a.init_clf, device)
        print("loaded ClfNet weights from", a.init_clf)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"I2IReg-ClfNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M "
          f"(trainable {sum(p.numel() for p in params) / 1e6:.2f} M)")
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = CascadeLoss(cfg["w_reg"], cfg["w_clf"])
    scaler = make_scaler(device)
    log = CSVLogger(out / "log.csv")
    best = -1.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = {"loss": 0.0, "l_reg": 0.0, "l_clf": 0.0}
        n = 0
        for b in tqdm(dl_tr, desc=f"epoch {ep}/{cfg['epochs']}", leave=False):
            x, roi, y = b["image"].to(device), b["roi"].to(device), b["label"].to(device)
            with amp_ctx(device):
                o = model(x)
            o = {k: v.float() for k, v in o.items()}
            loss, parts = crit(o, roi, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x.size(0)
            tot["loss"] += loss.item() * bs
            tot["l_reg"] += parts["l_reg"] * bs
            tot["l_clf"] += parts["l_clf"] * bs
            n += bs
        tot = {k: v / max(n, 1) for k, v in tot.items()}
        m, _ = eval_cascade(model, dl_va, device, cfg["roi_threshold"])
        sched.step(m["f1"])
        log.log({"epoch": ep, **tot, **{f"val_{k}": m.get(k, float("nan")) for k in KEYS},
                 "lr": opt.param_groups[0]["lr"]})
        print(f"[{ep:03d}] loss={tot['loss']:.4f} (reg {tot['l_reg']:.2e}, clf {tot['l_clf']:.3f}) | val {fmt(m, KEYS)}")
        score = m["f1"] + 0.1 * (m.get("reg_iou", 0.0) if m.get("reg_iou") == m.get("reg_iou") else 0.0)
        if score > best:
            best = score
            save_checkpoint(model, out / "best.pt", {"epoch": ep, "val": m, "cfg": cfg})
        save_checkpoint(model, out / "last.pt", {"epoch": ep, "cfg": cfg})

    load_weights(model, out / "best.pt", device)
    res = {"val": eval_cascade(model, dl_va, device, cfg["roi_threshold"])[0]}
    if len(ho):
        res["holdout"], raw = eval_cascade(model, dl_ho, device, cfg["roi_threshold"])
        save_json(raw, out / "holdout_predictions.json")
    save_json(res, out / "metrics.json")
    print("best val :", fmt(res["val"], KEYS))
    if "holdout" in res:
        print("hold-out :", fmt(res["holdout"], KEYS))


if __name__ == "__main__":
    main()
