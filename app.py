"""Streamlit demo (the paper deployed its model in a Streamlit clinical interface).

    streamlit run app.py
Pick the checkpoints in the sidebar, then either click "Test" under a sample image or upload
your own CT slice (PNG/JPG/TIF/DICOM). Research prototype - not for clinical use.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.io_utils import read_gray  # noqa: E402
from src.pipeline import KidneyStonePipeline  # noqa: E402
from src.utils import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
# Set to True when the kidney-localisation model (I2IReg-ClfNet) has been trained.
SHOW_KIDNEY_MODEL = False
st.set_page_config(page_title="Kidney Stone CT", layout="wide")
st.title("Kidney stone detection & localisation in abdominal CT")
st.caption("Stone detection (ClfNet) + stone localisation (StoneUNet). Research use only - not a medical device.")


def find(keyword):
    """All best.pt checkpoints under outputs/ whose path contains `keyword`."""
    out = ROOT / "outputs"
    if not out.exists():
        return [""]
    return [""] + sorted(str(p.relative_to(ROOT)) for p in out.rglob("best.pt") if keyword in str(p))


def _num(p):
    m = re.findall(r"\d+", p.stem)
    return int(m[-1]) if m else 0


# Sample images left out of the demo gallery (they are still part of the reported hold-out metrics).
EXCLUDE = {"HO_pos_1.png", "HO_pos_221.png"}


@st.cache_data
def sample_images(n_per_class=4):
    """Test images neither model was trained on:
    Figshare hold-out (HO_pos / HO_neg) and the uploaded dataset's hold-out fold."""
    samples = []
    kimg = ROOT / "data" / "kidney" / "images"
    if kimg.exists():
        pos = [p for p in sorted(kimg.glob("HO_pos_*.png"), key=_num) if p.name not in EXCLUDE]
        neg = [p for p in sorted(kimg.glob("HO_neg_*.png"), key=_num) if p.name not in EXCLUDE]
        step_p = max(1, len(pos) // n_per_class)
        step_n = max(1, len(neg) // n_per_class)
        samples += [(str(p), "Stone+", "Figshare hold-out") for p in pos[::step_p][:n_per_class]]
        samples += [(str(p), "Stone-", "Figshare hold-out") for p in neg[::step_n][:n_per_class]]
    sroot = ROOT / "data" / "stone_seg"
    if (sroot / "splits.json").exists():
        folds = json.load(open(sroot / "splits.json", encoding="utf-8"))["folds"]
        ho = sorted((i for i, f in folds.items() if f == -1), key=int)
        step = max(1, len(ho) // n_per_class)
        for i in ho[::step][:n_per_class]:
            p = sroot / "image" / f"{i}.tif"
            if p.exists() and p.name not in EXCLUDE:
                samples.append((str(p), "Stone+", "Uploaded dataset hold-out"))
    return samples


@st.cache_data
def thumbnail(path):
    img, _ = read_gray(path)
    return img


cfg = load_config()
with st.sidebar:
    st.header("Models")
    cascade = st.selectbox("I2IReg-ClfNet (kidney ROI + detection)", find("cascade")) if SHOW_KIDNEY_MODEL else ""
    clf = st.selectbox("ClfNet - stone detection", find("clfnet"))
    stone = st.selectbox("StoneUNet - stone localisation", find("stone_seg"))
    st.header("Settings")
    restrict = (st.checkbox("Keep stones only inside kidney ROI (needs cascade)", value=True)
                if SHOW_KIDNEY_MODEL else False)
    seg_thr = st.slider("Stone mask threshold", 0.05, 0.95, float(cfg["inference"]["seg_threshold"]), 0.05)
    decision = st.radio("Final decision based on",
                        ["StoneUNet (stone found = Stone+)", "Classifier P(stone)", "Both must agree"], index=0)
    learned = st.checkbox("Use P(stone) threshold learned in training", value=True)
    clf_thr = st.slider("P(stone) threshold (if not learned)", 0.05, 0.95,
                        float(cfg["inference"]["stone_prob_threshold"]), 0.05, disabled=learned)


@st.cache_resource
def get_pipe(cascade, clf, stone, restrict, seg_thr, clf_thr, learned):
    icfg = dict(cfg["inference"], restrict_to_kidney=restrict, seg_threshold=seg_thr,
                stone_prob_threshold=clf_thr, override_threshold=not learned,
                min_area=cfg["stone_seg"]["min_area"])
    return KidneyStonePipeline(cascade_ckpt=str(ROOT / cascade) if cascade else None,
                               clf_ckpt=str(ROOT / clf) if (clf and not cascade) else None,
                               stone_ckpt=str(ROOT / stone) if stone else None,
                               device=cfg["common"]["device"], cfg=icfg)


def show_result(path, true_label=None, source=None):
    pipe = get_pipe(cascade, clf, stone, restrict, seg_thr, clf_thr, learned)
    with st.spinner("Running models ..."):
        res = pipe(path)
    st.subheader(f"Result - {Path(path).name}")
    by_seg = len(res["stones"]) > 0 if "stones" in res else None
    by_clf = (res["stone_probability"] >= pipe.cfg["stone_prob_threshold"]) if "stone_probability" in res else None
    if decision.startswith("StoneUNet") and by_seg is not None:
        detected, how = by_seg, "StoneUNet"
    elif decision.startswith("Classifier") and by_clf is not None:
        detected, how = by_clf, "classifier"
    elif decision.startswith("Both") and by_seg is not None and by_clf is not None:
        detected, how = by_seg and by_clf, "StoneUNet + classifier"
    else:
        detected, how = bool(by_seg if by_seg is not None else by_clf), "available model"
    verdict = f"STONE DETECTED ({how})" if detected else f"NO STONE ({how})"
    if true_label is not None:
        correct = (true_label == "Stone+") == detected
        msg = f"**{verdict}**  |  true label: **{true_label}** ({source})  ->  " + ("correct" if correct else "wrong")
        (st.success if correct else st.error)(msg)
    else:
        (st.warning if detected else st.info)(f"**{verdict}**")
    st.image(pipe.render(res), width="stretch")
    c1, c2, c3 = st.columns(3)
    if "stone_probability" in res:
        c1.metric("P(stone) - classifier", f"{res['stone_probability']:.3f}",
                  "says stone" if by_clf else "says no stone", delta_color="off")
    if "stones" in res:
        c2.metric("Stone regions found", len(res["stones"]))
    if "kidney_area_px" in res:
        c3.metric("Kidney ROI area (px)", res["kidney_area_px"])
    if res.get("stones"):
        st.dataframe([{"#": i, "side": s["side"], "centroid (x,y)": f"({s['centroid'][0]:.0f}, {s['centroid'][1]:.0f})",
                       "bbox (x,y,w,h)": str(s["bbox"]), "area px": s["area"],
                       "area mm²": round(s.get("area_mm2", float("nan")), 2), "max prob": round(s["max_prob"], 3)}
                      for i, s in enumerate(res["stones"], 1)])


if not (cascade or clf or stone):
    st.info("Select at least one trained model in the sidebar (train first - see README).")
    st.stop()

tab_samples, tab_upload = st.tabs(["Sample test images", "Upload your own"])

with tab_samples:
    samples = sample_images()
    if not samples:
        st.info("No sample images found. Prepare the datasets first (see README).")
    else:
        st.write("These images were **not used for training**. Click **Test** under an image.")
        cols_per_row = 6
        for r0 in range(0, len(samples), cols_per_row):
            cols = st.columns(cols_per_row)
            for c, (path, label, source) in zip(cols, samples[r0:r0 + cols_per_row]):
                with c:
                    st.image(thumbnail(path), caption=f"{Path(path).name}\n{label}", width="stretch", clamp=True)
                    if st.button("Test", key=f"btn_{path}"):
                        st.session_state["selected"] = (path, label, source)
        if "selected" in st.session_state:
            st.divider()
            show_result(*st.session_state["selected"])

with tab_upload:
    up = st.file_uploader("CT slice", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "dcm"])
    if up is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
            f.write(up.getbuffer())
            tmp = f.name
        show_result(tmp)
