# app_dashboard_best.py
"""
Dashboard without webcam (stable everywhere).
Now uses a 5-feature model: 4 video features + OSDI score.
OSDI Score = (sum of responses × 25) / number answered
"""

from typing import Optional, Dict
import json, tempfile, importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===== IMPORTANT: model expects 5 features (video + OSDI) =====
FEATURE_COLS = [
    "blink_rate_bpm",
    "incomplete_blink_ratio",
    "avg_ibi_sec",
    "redness_index",
    "osdi_score",
]
VIDEO_FEATURES = FEATURE_COLS[:-1]  # first 4

st.set_page_config(page_title="Dry Eye Risk – Dashboard (Best)", layout="centered")
st.title("Dry Eye Risk (Blink + Redness + OSDI) – Dashboard (Best)")

# ---------------- OSDI helpers ----------------
QUESTIONS = [
    "Do your eyes feel dry or gritty after screen use?",
    "Do your eyes become red while using your laptop?",
    "Do you experience blurred vision during or after laptop work?",
    "Do you get headaches after extended laptop use?",
    "Do your eyes feel tired or heavy after screen time?",
    "Do you feel the need to rub your eyes while using your laptop?",
    "Do you feel burning or stinging sensations in your eyes during laptop use?",
]
CHOICES = [0, 1, 2, 3, 4]  # 0=Never ... 4=Always

def compute_osdi(responses: list[int]) -> float:
    answered = [v for v in responses if v is not None]
    if not answered:
        return 0.0
    return round((sum(answered) * 25.0) / len(answered), 1)

def osdi_form() -> Optional[float]:
    st.subheader("OSDI – Quick Symptoms Questionnaire")
    st.caption("Scale: 0 = Never … 4 = Always")
    with st.form("osdi_form"):
        values = []
        for i, q in enumerate(QUESTIONS):
            values.append(st.radio(
                f"{i+1}. {q}",
                options=CHOICES,
                index=0,
                horizontal=True,
                key=f"osdi_q{i}"
            ))
        submitted = st.form_submit_button("Compute OSDI")
        score = compute_osdi(values) if submitted else None
    return score

# Persist last OSDI in session
if "osdi_score" not in st.session_state:
    st.session_state["osdi_score"] = None

# ---------------- Load model artifacts ----------------
model_path = Path("best_model.joblib")
label_path = Path("label_encoder.joblib")
if not (model_path.exists() and label_path.exists()):
    st.error("Model artifacts not found. Run `python train_model_best.py --dataset dataset.csv`.")
    st.stop()

model = joblib.load(model_path)
label_encoder = joblib.load(label_path)
st.success("Model loaded.")

def predict_and_show(video_feats: Dict[str, float], osdi: Optional[float]):
    # Validate we have OSDI
    if osdi is None:
        st.warning("Please compute OSDI first (top of the page). Using 0.0 as a temporary fallback.")
        osdi = 0.0

    # Compose feature vector in required order
    x = np.array([[
        float(video_feats["blink_rate_bpm"]),
        float(video_feats["incomplete_blink_ratio"]),
        float(video_feats["avg_ibi_sec"]),
        float(video_feats["redness_index"]),
        float(osdi),
    ]], dtype=float)

    y_hat = model.predict(x)[0]
    try:
        label = label_encoder.inverse_transform([y_hat])[0]
    except Exception:
        label = str(y_hat)

    badge_map = {"Low": "🟢 Low", "Medium": "🟠 Medium", "High": "🔴 High"}
    badge = badge_map.get(label, label)

    st.subheader("Result")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**Blink rate:** {video_feats['blink_rate_bpm']} blinks/min")
        st.write(f"**Incomplete blink ratio:** {video_feats['incomplete_blink_ratio']}")
        st.write(f"**Avg inter-blink interval:** {video_feats['avg_ibi_sec']} sec")
        st.write(f"**Redness index:** {video_feats['redness_index']}")
        st.write(f"**OSDI score:** {osdi}")
    with c2:
        st.metric("Predicted Risk", badge)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🎥 From Video", "📦 From CSV", "🛠 Manual"])

with tab1:
    # OSDI first
    score = osdi_form()
    if score is not None:
        st.session_state["osdi_score"] = score
        st.success(f"OSDI computed: **{score}**")

    st.subheader("Upload a short eye video")
    st.caption("Use 5–20 s video focused on one eye in good light (mp4/mov/avi).")
    up = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="vid")

    if up:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmpdir.name) / up.name
        with open(tmp_path, "wb") as f:
            f.write(up.read())

        try:
            extractor = importlib.import_module("extract_features")
        except Exception as e:
            st.error(
                "Couldn't import extract_features.py. Place it next to this app and define:\n"
                "  def extract_from_video(video_path) -> dict\n"
                "that returns the 4 video features: "
                + ", ".join(VIDEO_FEATURES)
                + f"\n\nError: {e}"
            )
            st.stop()

        if not hasattr(extractor, "extract_from_video"):
            st.error("Your extract_features.py must define `extract_from_video(video_path) -> dict`")
            st.stop()

        with st.spinner("Extracting features…"):
            feats = extractor.extract_from_video(str(tmp_path))

        missing = [c for c in VIDEO_FEATURES if c not in feats]
        if missing:
            st.error(f"Extractor did not return required keys: {missing}\nReturned: {feats}")
        else:
            st.success("Features extracted.")
            with st.expander("See extracted features"):
                st.json(feats)
            predict_and_show(feats, st.session_state.get("osdi_score"))

with tab2:
    st.subheader("Upload CSV with features")
    st.caption("CSV must contain the 4 video columns; `osdi_score` is optional (falls back to current OSDI).")
    file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if file:
        df = pd.read_csv(file)
        miss_video = [c for c in VIDEO_FEATURES if c not in df.columns]
        if miss_video:
            st.error(f"Missing required video columns: {miss_video}")
        else:
            # If osdi_score not supplied per-row, use session OSDI or 0.0
            if "osdi_score" not in df.columns:
                default_osdi = st.session_state.get("osdi_score", 0.0) or 0.0
                df["osdi_score"] = default_osdi
                st.info(f"`osdi_score` missing in CSV; using OSDI={default_osdi} for all rows.")

            X = df[FEATURE_COLS].astype(float)
            y_hat = model.predict(X)
            try:
                labels = label_encoder.inverse_transform(y_hat)
            except Exception:
                labels = y_hat
            out = df.copy()
            out["predicted_label"] = labels
            st.dataframe(out.head(25))
            st.download_button(
                "Download predictions.csv",
                out.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

with tab3:
    st.subheader("Manual entry (for developers)")
    c1, c2 = st.columns(2)
    with c1:
        br = st.number_input("Blink rate (blinks/min)", 0.0, 60.0, 15.0, 0.1)
        ibi = st.number_input("Avg inter-blink interval (sec)", 0.0, 30.0, 4.5, 0.1)
    with c2:
        inc = st.number_input("Incomplete blink ratio", 0.0, 1.0, 0.12, 0.01, format="%.2f")
        red = st.number_input("Redness index", 0.0, 1.0, 0.22, 0.01, format="%.2f")
    osdi_manual = st.number_input("OSDI score", 0.0, 100.0, float(st.session_state.get("osdi_score") or 0.0), 0.1)

    if st.button("Predict"):
        predict_and_show(
            {
                "blink_rate_bpm": br,
                "incomplete_blink_ratio": inc,
                "avg_ibi_sec": ibi,
                "redness_index": red,
            },
            osdi_manual,
        )
