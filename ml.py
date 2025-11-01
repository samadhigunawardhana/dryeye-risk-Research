# ml.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import joblib
import numpy as np
import streamlit as st

FEATURE_5 = ["blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index","osdi_score"]
VIDEO_FEATURES = FEATURE_5[:-1]
FEATURE_10 = [
    "blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index",
    "ibr_x_red","blink_per_sec","ibi_inv","ibi_lt6","red_gt0_3","ibr_gt0_2"
]

@st.cache_resource(show_spinner=False)
def load_artifacts():
    mp, lp = Path("best_model.joblib"), Path("label_encoder.joblib")
    if not (mp.exists() and lp.exists()):
        return None, None, None, "Place best_model.joblib & label_encoder.joblib beside this script."
    try:
        model = joblib.load(mp)
        le    = joblib.load(lp)
        names = getattr(model, "feature_names_in_", None)
        n_in  = getattr(model, "n_features_in_", None)
        return model, le, (names, n_in), None
    except Exception as e:
        return None, None, None, f"Failed loading artifacts: {e}"

def _expected_feature_names(model_meta) -> Optional[List[str]]:
    p = Path("feature_cols.json")
    if p.exists():
        try:
            cols = json.loads(p.read_text())
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
        except Exception:
            pass
    names, _ = model_meta if model_meta else (None, None)
    return list(names) if names is not None else None

def _build_10_from_video(f: Dict[str, float]) -> Tuple[List[float], List[str]]:
    br  = float(f["blink_rate_bpm"])
    ibr = float(f["incomplete_blink_ratio"])
    ibi = float(f["avg_ibi_sec"])
    red = float(f["redness_index"])
    vals = [
        br, ibr, ibi, red,
        ibr * red,
        br / 60.0,
        (0.0 if ibi == 0 else 1.0 / ibi),
        1.0 if ibi < 6.0 else 0.0,
        1.0 if red > 0.30 else 0.0,
        1.0 if ibr > 0.20 else 0.0,
    ]
    return vals, FEATURE_10

def build_input_vector(feats: Dict[str, float], osdi: float):
    names, n_in = load_artifacts()[2] if load_artifacts()[2] else (None, None)  # MODEL_META
    expected = _expected_feature_names(load_artifacts()[2])
    count = int(n_in) if n_in is not None else (len(expected) if expected else None)

    def is10(): return (expected and len(expected)==10 and set(expected)==set(FEATURE_10)) or count==10
    def is5():  return (expected and len(expected)==5  and set(expected)==set(FEATURE_5))  or count==5

    if is10():
        v10, order = _build_10_from_video(feats)
        if expected and expected != order:
            m = dict(zip(order, v10))
            v10 = [float(m[n]) for n in expected]
        return np.array([v10], dtype=float), "10"

    if is5():
        row = {
            "blink_rate_bpm": float(feats["blink_rate_bpm"]),
            "incomplete_blink_ratio": float(feats["incomplete_blink_ratio"]),
            "avg_ibi_sec": float(feats["avg_ibi_sec"]),
            "redness_index": float(feats["redness_index"]),
            "osdi_score": float(osdi),
        }
        order = expected if expected else FEATURE_5
        return np.array([[row[c] for c in order]], dtype=float), "5"

    st.error(
        "Loaded model expects a 5- or 10-feature set this app doesn’t support.\n"
        f"10-feature expected: {FEATURE_10}\n"
        f"5-feature expected: {FEATURE_5}\n"
        "Add a matching feature_cols.json or retrain."
    )
    st.stop()
