"""Config, seeding, device, checkpoint and logging helpers."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_args(cfg_section: dict, args) -> dict:
    """CLI arguments that are not None override the YAML section."""
    out = dict(cfg_section)
    for k, v in vars(args).items():
        if v is not None:
            out[k] = v
    return out


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_workers():
    # Windows + many workers is slow to spawn; keep it modest.
    return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)


def save_checkpoint(model, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_weights(model, path, device="cpu", strict=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return ck.get("extra", {}) if isinstance(ck, dict) else {}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            if not self.header_written:
                f.write(",".join(row.keys()) + "\n")
                self.header_written = True
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        now = time.time()
        dt, self.t = now - self.t, now
        return dt
