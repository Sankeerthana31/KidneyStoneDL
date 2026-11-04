"""Run the full pipeline on one CT slice or a folder of slices (PNG/JPG/TIF/DICOM).

    # stone localisation only (works right after training on the uploaded dataset)
    python scripts/predict.py --input data/stone_seg/image/1000.tif --stone outputs/stone_seg/fold0/best.pt

    # kidney localisation + stone detection + stone localisation
    python scripts/predict.py --input my_ct_folder --cascade outputs/cascade/fold0/best.pt \
                              --stone outputs/stone_seg/fold0/best.pt
Outputs: <out>/<name>_result.png (CT | kidney ROI | Grad-CAM | stones) and <out>/results.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.io_utils import IMG_EXTS
from src.pipeline import KidneyStonePipeline
from src.utils import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cascade", default=None, help="I2IReg-ClfNet checkpoint")
    ap.add_argument("--reg", default=None, help="I2IRegNet checkpoint (if no cascade)")
    ap.add_argument("--clf", default=None, help="ClfNet checkpoint (if no cascade)")
    ap.add_argument("--stone", default=None, help="StoneUNet checkpoint")
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--no_restrict", action="store_true", help="do not mask stones to the kidney ROI")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if not any([a.cascade, a.reg, a.clf, a.stone]):
        ap.error("give at least one checkpoint (--cascade / --reg / --clf / --stone)")

    cfg = load_config(a.config)
    icfg = dict(cfg["inference"])
    icfg["min_area"] = cfg["stone_seg"]["min_area"]
    if a.no_restrict:
        icfg["restrict_to_kidney"] = False
    pipe = KidneyStonePipeline(a.cascade, a.reg, a.clf, a.stone, cfg["common"]["device"], icfg)

    inp = Path(a.input)
    files = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXTS)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for p in files:
        res = pipe(p)
        panel = pipe.render(res)
        cv2.imwrite(str(out / f"{p.stem}_result.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        s = pipe.summary(res, str(p))
        results.append(s)
        msg = f"{p.name}: "
        if "stone_probability" in s:
            msg += f"P(stone)={s['stone_probability']:.3f} -> {'STONE' if s['stone_present'] else 'no stone'}  "
        if "n_stones" in s:
            msg += f"| {s['n_stones']} stone region(s) " + ", ".join(
                f"[{st['side'].split(' (')[0]} @ ({st['centroid'][0]:.0f},{st['centroid'][1]:.0f}) {st['area']}px]"
                for st in s["stones"])
        print(msg)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved {len(files)} result image(s) + results.json to {out}")


if __name__ == "__main__":
    main()
