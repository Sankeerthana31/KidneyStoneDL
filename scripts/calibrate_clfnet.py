"""Choose the stone/no-stone decision threshold of an already-trained ClfNet on the
validation fold (Youden's J), then report hold-out results with that threshold.
No retraining needed (~5 min on CPU). The threshold is saved into the checkpoint so
predict.py and the web app use it automatically.

    python scripts/calibrate_clfnet.py --ckpt outputs/clfnet/ShN+squeezenet+CBAM_image/fold0/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.datasets import KidneyROIDataset
from src.engine import loader, eval_classifier, fmt
from src.metrics import classification_metrics
from src.models import ClfNet
from src.splits import load_splits, select
from src.utils import load_config, resolve, get_device, default_workers, save_json

KEYS = ["accuracy", "recall", "specificity", "precision", "f1", "roc_auc", "kappa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device(cfg["common"]["device"])
    ckpt_path = resolve(a.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck.get("extra", {}).get("cfg", {})
    model = ClfNet(deep_encoder=c.get("deep_encoder", "squeezenet"), use_shallow=c.get("use_shallow", True),
                   use_cbam=c.get("use_cbam", True), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    key = c.get("input", "image")

    root = resolve(cfg["paths"]["kidney_data"])
    folds = load_splits(root / "splits.json")
    S = c.get("img_size", 224)
    dl_va = loader(KidneyROIDataset(root, select(folds, a.fold, "val"), S), 32, False, a.workers)
    dl_ho = loader(KidneyROIDataset(root, select(folds, a.fold, "holdout"), S), 32, False, a.workers)

    print("evaluating validation fold ...")
    _, va = eval_classifier(model, dl_va, device, key)
    fpr, tpr, thr = roc_curve(va["y"], va["p"])
    t = float(np.clip(thr[int(np.argmax(tpr - fpr))], 1e-4, 1 - 1e-4))
    print("evaluating hold-out ...")
    _, ho = eval_classifier(model, dl_ho, device, key)

    res = {"threshold": t,
           "val_thr0.5": classification_metrics(va["y"], va["p"], 0.5),
           "val": classification_metrics(va["y"], va["p"], t),
           "holdout_thr0.5": classification_metrics(ho["y"], ho["p"], 0.5),
           "holdout": classification_metrics(ho["y"], ho["p"], t)}
    save_json(res, ckpt_path.parent / "metrics_calibrated.json")
    ck.setdefault("extra", {})["threshold"] = t
    torch.save(ck, ckpt_path)

    print(f"\ndecision threshold chosen on validation: {t:.3f}  (saved into {ckpt_path.name})")
    print("hold-out, threshold 0.5   :", fmt(res["holdout_thr0.5"], KEYS))
    print(f"hold-out, threshold {t:.3f} :", fmt(res["holdout"], KEYS))


if __name__ == "__main__":
    main()