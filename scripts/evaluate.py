"""Evaluate a trained checkpoint on the hold-out (or a CV fold) and save plots.

    python scripts/evaluate.py --task cascade --ckpt outputs/cascade/fold0/best.pt
    python scripts/evaluate.py --task reg     --ckpt outputs/i2iregnet/fold0/best.pt
    python scripts/evaluate.py --task clf     --ckpt outputs/clfnet/ShN+squeezenet+CBAM_roi/fold0/best.pt
    python scripts/evaluate.py --task stone   --ckpt outputs/stone_seg/fold0/best.pt --save_examples 12
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from src.datasets import KidneyROIDataset, StoneSegDataset
from src.engine import loader, eval_regression, eval_classifier, eval_cascade, eval_stone_seg, fmt
from src.models import I2IRegNet, ClfNet, I2IRegClfNet, StoneUNet
from src.splits import load_splits, select
from src.utils import load_config, resolve, get_device, default_workers, save_json


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(raw, path):
    fpr, tpr, _ = roc_curve(raw["y"], raw["p"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reg", "clf", "cascade", "stone"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val", "trainval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--save_examples", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device(cfg["common"]["device"])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ccfg = ck.get("extra", {}).get("cfg", {})
    out = Path(a.ckpt).parent / f"eval_{a.split}"
    out.mkdir(parents=True, exist_ok=True)

    if a.task == "stone":
        root = resolve(cfg["paths"]["stone_data"])
        names = select(load_splits(root / "splits.json"), a.fold, a.split)
        ds = StoneSegDataset(root, names, train=False, full_size=ccfg.get("full_size", 512))
        model = StoneUNet(ccfg.get("encoder", "resnet34"), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        thr, ma = cfg["stone_seg"]["threshold"], cfg["stone_seg"]["min_area"]
        m = eval_stone_seg(model, loader(ds, 4, False, a.workers), device, thr, ma)
        print(fmt(m))
        save_json(m, out / "metrics.json")
        model.eval()
        for i in range(min(a.save_examples, len(ds))):
            b = ds[i]
            with torch.no_grad():
                p = torch.sigmoid(model(b["image"][None].to(device)))[0, 0].cpu().numpy()
            g = (b["image"][0].numpy() * 255).astype(np.uint8)
            rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            gt_c, _ = cv2.findContours(b["mask"][0].numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pr_c, _ = cv2.findContours((p > thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, gt_c, -1, (0, 255, 0), 1)     # green = ground truth
            cv2.drawContours(rgb, pr_c, -1, (0, 0, 255), 1)     # red = prediction
            cv2.imwrite(str(out / f"example_{b['name']}.png"), rgb)
        print("saved to", out)
        return

    root = resolve(cfg["paths"]["kidney_data"])
    names = select(load_splits(root / "splits.json"), a.fold, a.split)
    ds = KidneyROIDataset(root, names, ccfg.get("img_size", 224))
    dl = loader(ds, 16, False, a.workers)
    if a.task == "reg":
        model = I2IRegNet(pretrained=False)
        model.load_state_dict(ck["state_dict"])
        m = eval_regression(model.to(device), dl, device, cfg["i2iregnet"]["roi_threshold"])
        print(fmt(m))
        save_json(m, out / "metrics.json")
        return
    if a.task == "clf":
        model = ClfNet(deep_encoder=ccfg.get("deep_encoder", "squeezenet"), use_shallow=ccfg.get("use_shallow", True),
                       use_cbam=ccfg.get("use_cbam", True), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        thr = ck.get("extra", {}).get("threshold", 0.5)
        m, raw = eval_classifier(model.to(device), dl, device, ccfg.get("input", "roi"), threshold=thr)
        print(f"decision threshold: {thr:.3f}")
    else:
        model = I2IRegClfNet(deep_encoder=ccfg.get("deep_encoder", "squeezenet"), use_shallow=ccfg.get("use_shallow", True),
                             use_cbam=ccfg.get("use_cbam", True), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        m, raw = eval_cascade(model.to(device), dl, device, cfg["cascade"]["roi_threshold"])
    print(fmt(m))
    save_json(m, out / "metrics.json")
    save_json(raw, out / "predictions.json")
    plot_confusion(m, out / "confusion_matrix.png", f"{a.task} - {a.split}")
    if len(set(raw["y"])) > 1:
        plot_roc(raw, out / "roc.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
