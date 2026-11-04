"""External validation on the CT KIDNEY DATASET (Islam et al., 2022) - paper Section 3.4.4.2.

The paper keeps only the 848 axial Stone images and samples 848 Normal images, then
 (a) evaluates the cascade without fine-tuning, and
 (b) fine-tunes on 50% and tests on the other 50%.
That dataset has no kidney masks, so fine-tuning uses only the classification loss.

    python scripts/evaluate_external.py --ckpt outputs/cascade/fold0/best.pt
    python scripts/evaluate_external.py --ckpt outputs/cascade/fold0/best.pt --finetune --epochs 10

Expected folder: data/ct_kidney_external/Normal/*.jpg and data/ct_kidney_external/Stone/*.jpg
(keep only axial slices in the Stone folder if you want to reproduce the paper exactly).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datasets import CTKidneyExternal
from src.engine import loader, amp_ctx, make_scaler, eval_cascade, fmt
from src.io_utils import list_images
from src.models import I2IRegClfNet
from src.utils import load_config, resolve, get_device, default_workers, save_json, save_checkpoint, seed_everything

KEYS = ["accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "kappa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--balance", action="store_true", default=True, help="sample as many Normal as Stone images")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--finetune_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["common"]["seed"])
    device = get_device(cfg["common"]["device"])
    root = Path(a.data) if a.data else resolve(cfg["paths"]["external_data"])

    stone = [(p, 1) for p in list_images(root / "Stone")]
    normal = [(p, 0) for p in list_images(root / "Normal")]
    rng = np.random.default_rng(cfg["common"]["seed"])
    if a.balance and len(normal) > len(stone):
        normal = [normal[i] for i in rng.choice(len(normal), len(stone), replace=False)]
    files = stone + normal
    print(f"external set: {len(stone)} Stone + {len(normal)} Normal")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck.get("extra", {}).get("cfg", {})
    model = I2IRegClfNet(deep_encoder=c.get("deep_encoder", "squeezenet"), use_shallow=c.get("use_shallow", True),
                         use_cbam=c.get("use_cbam", True), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    out = Path(a.ckpt).parent / "external"
    out.mkdir(parents=True, exist_ok=True)

    if not a.finetune:
        m, raw = eval_cascade(model, loader(CTKidneyExternal(root, files=files), 32, False, a.workers), device, with_reg=False)
        print("without fine-tuning:", fmt(m, KEYS))
        save_json(m, out / "metrics_no_finetune.json")
        return

    tr_files, te_files = train_test_split(files, train_size=a.finetune_frac, stratify=[l for _, l in files],
                                          random_state=cfg["common"]["seed"])
    dl_tr = loader(CTKidneyExternal(root, files=tr_files), a.batch_size, True, a.workers, drop_last=True)
    dl_te = loader(CTKidneyExternal(root, files=te_files), 32, False, a.workers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    ce = nn.CrossEntropyLoss()
    scaler = make_scaler(device)
    for ep in range(1, a.epochs + 1):
        model.train()
        for b in tqdm(dl_tr, desc=f"finetune {ep}/{a.epochs}", leave=False):
            x, y = b["image"].to(device), b["label"].to(device)
            with amp_ctx(device):
                o = model(x)
            loss = ce(o["logits"].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        m, _ = eval_cascade(model, dl_te, device, with_reg=False)
        print(f"[{ep:02d}] test {fmt(m, KEYS)}")
    save_json(m, out / "metrics_finetuned.json")
    save_checkpoint(model, out / "finetuned.pt", {"cfg": c})
    print("saved to", out)


if __name__ == "__main__":
    main()
