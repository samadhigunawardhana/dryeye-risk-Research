# app_dashboard_best.py
"""
Dashboard without webcam (stable everywhere).
Now includes a 7-item OSDI questionnaire:
OSDI Score = (sum of responses × 25) / number answered
"""

from typing import Optional
import json, tempfile, importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_COLS = ["blink_rate_bpm","incomplete_blink_ratio","avg_ibi_sec","redness_index"]

st.set_page_config(page_title="Dry Eye Risk – Dashboard (Best)", layout="centered")
st.title("Dry Eye Risk (Blink + Redness) – Dashboard (Best)")

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
    # official scoring: (sum of answered * 25) / number answered
    answered = [v for v in responses if v is not None]
    if not answered:
        return 0.0
    return round((sum(answered) * 25.0) / len(answered), 1)

def osdi_form():
    st.subheader("OSDI – Quick Symptoms Questionnaire")
    st.caption("Scale: 0 = Never … 4 = Always")
    with st.form("osdi_form"):
        cols = st.columns(1)
        values = []
        for i, q in enumerate(QUESTIONS):
            values.append(st.radio(
                f"{i+1}. {q}",
                options=CHOICES,
                index=0,  # default 0 = Never
                horizontal=True,
                key=f"osdi_q{i}"
            ))
        submitted = st.form_submit_button("Compute OSDI")
        score = compute_osdi(values) if submitted else None
    return score

# Persist the last computed OSDI in session
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

def predict_and_show(feats: dict, osdi: Optional[float] = None):
    x = np.array([[feats["blink_rate_bpm"],
                   feats["incomplete_blink_ratio"],
                   feats["avg_ibi_sec"],
                   feats["redness_index"]]], dtype=float)
    y_hat = model.predict(x)[0]
    try:
        label = label_encoder.inverse_transform([y_hat])[0]
    except Exception:
        label = str(y_hat)
    badge = {"Low":"🟢 Low","Medium":"🟠 Medium","High":"🔴 High"}.get(label, label)

    st.subheader("Result")
    c1,c2 = st.columns([2,1])
    with c1:
        st.write(f"**Blink rate:** {feats['blink_rate_bpm']} blinks/min")
        st.write(f"**Incomplete blink ratio:** {feats['incomplete_blink_ratio']}")
        st.write(f"**Avg inter-blink interval:** {feats['avg_ibi_sec']} sec")
        st.write(f"**Redness index:** {feats['redness_index']}")
        if osdi is not None:
            st.write(f"**OSDI score:** {osdi}")
    with c2:
        st.metric("Predicted Risk", badge)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🎥 From Video", "📦 From CSV", "🛠 Manual"])

with tab1:
    # 1) OSDI form at top (recommended flow)
    score = osdi_form()
    if score is not None:
        st.session_state["osdi_score"] = score
        st.success(f"OSDI computed: **{score}**")

    st.subheader("Upload a short eye video")
    st.caption("Use 5–20 s video focused on one eye in good light (mp4/mov/avi).")

    up = st.file_uploader("Upload video", type=["mp4","mov","avi"], key="vid")
    if up:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmpdir.name) / up.name
        with open(tmp_path, "wb") as f:
            f.write(up.read())

        try:
            extractor = importlib.import_module("extract_features")
        except Exception as e:
            st.error("Couldn't import extract_features.py. Place it next to this app and define:\n"
                     "def extract_from_video(video_path)->dict returning the 4 features.\n\n"
                     f"Error: {e}")
            st.stop()

        if not hasattr(extractor, "extract_from_video"):
            st.error("Your extract_features.py must define `extract_from_video(video_path)->dict`")
            st.stop()

        with st.spinner("Extracting features…"):
            feats = extractor.extract_from_video(str(tmp_path))

        missing = [c for c in FEATURE_COLS if c not in feats]
        if missing:
            st.error(f"Extractor did not return required keys: {missing}\nReturned: {feats}")
        else:
            st.success("Features extracted.")
            with st.expander("See extracted features"):
                st.json(feats)
            predict_and_show(feats, st.session_state.get("osdi_score"))

with tab2:
    st.subheader("Upload CSV with features")
    st.caption("CSV must contain: " + ", ".join(FEATURE_COLS) + ". Optional column: osdi_score")
    file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df[FEATURE_COLS].astype(float)
            y_hat = model.predict(X)
            try:
                labels = label_encoder.inverse_transform(y_hat)
            except Exception:
                labels = y_hat
            out = df.copy()
            out["predicted_label"] = labels
            st.dataframe(out.head(25))
            st.download_button("Download predictions.csv",
                               out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab3:
    st.subheader("Manual entry (for developers)")
    c1,c2 = st.columns(2)
    with c1:
        br = st.number_input("Blink rate (blinks/min)", 0.0, 60.0, 15.0, 0.1)
        ibi = st.number_input("Avg inter-blink interval (sec)", 0.0, 30.0, 4.5, 0.1)
    with c2:
        inc = st.number_input("Incomplete blink ratio", 0.0, 1.0, 0.12, 0.01, format="%.2f")
        red = st.number_input("Redness index", 0.0, 1.0, 0.22, 0.01, format="%.2f")
    # Optional: compute OSDI inline here too
    with st.expander("Optional: OSDI quick compute here"):
        values = [st.slider(f"{i+1}. {q}", 0, 4, 0) for i, q in enumerate(QUESTIONS)]
        st.caption("OSDI = (sum × 25) / N")
        st.write("OSDI score:", compute_osdi(values))
    if st.button("Predict"):
        predict_and_show({"blink_rate_bpm": br, "incomplete_blink_ratio": inc,
                          "avg_ibi_sec": ibi, "redness_index": red},
                         st.session_state.get("osdi_score"))
