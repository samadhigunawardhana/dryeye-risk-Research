# app_ded_full.py
# Streamlit multi-step app for Dry Eye Disease risk using OSDI + facial video features.
# Requires: best_model.joblib, label_encoder.joblib, extract_features.py (with extract_from_video(video_path)->dict)
# Optional (recommended): feature_cols.json  -> exact training order; if present, the app obeys it.

from __future__ import annotations
from pathlib import Path
import importlib
import json
import tempfile
from typing import Dict, Optional, List, Tuple

import joblib
import numpy as np
import streamlit as st

# --------------------------- App config ---------------------------
st.set_page_config(page_title="Dry Eye Risk – Full Flow", layout="centered")
PRIMARY_BTN = "primary"
NEUTRAL_BTN = "secondary"

# Some Streamlit versions expose st.rerun(), older ones only st.experimental_rerun().
def force_rerun():
    try:
        # Newer Streamlit
        st.rerun()
    except Exception:
        # Older Streamlit
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass  # last resort: do nothing (user may need to interact)

# App-provided 5-feature order (video + OSDI)
FEATURE_5 = [
    "blink_rate_bpm",
    "incomplete_blink_ratio",
    "avg_ibi_sec",
    "redness_index",
    "osdi_score",
]
VIDEO_FEATURES = FEATURE_5[:-1]  # first 4

# Standard 10-feature order (your current trained model)
FEATURE_10 = [
    "blink_rate_bpm",
    "incomplete_blink_ratio",
    "avg_ibi_sec",
    "redness_index",
    "ibr_x_red",
    "blink_per_sec",
    "ibi_inv",
    "ibi_lt6",
    "red_gt0_3",
    "ibr_gt0_2",
]

# --------------------------- Utilities ---------------------------
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    st.session_state.setdefault("osdi_score", None)
    st.session_state.setdefault("story_key", None)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("pred_label", None)
    st.session_state.setdefault("video_uploaded_name", None)
    st.session_state.setdefault("last_processed_token", None)  # avoid double-processing same file

init_state()

def goto(page: str):
    st.session_state.page = page

# ---- OSDI helpers ----
QUESTIONS = [
    "Do your eyes feel dry or gritty after screen use?",
    "Do your eyes become red while using your laptop?",
    "Do you experience blurred vision during or after laptop work?",
    "Do you get headaches after extended laptop use?",
    "Do your eyes feel tired or heavy after screen time?",
    "Do you feel the need to rub your eyes while using your laptop?",
    "Do you feel burning or stinging sensations in your eyes during laptop use?",
]
CHOICES = [0, 1, 2, 3, 4]  # 0=Never … 4=Always

def compute_osdi(responses: list[int]) -> float:
    answered = [v for v in responses if v is not None]
    if not answered:
        return 0.0
    return round((sum(answered) * 25.0) / len(answered), 1)

# ---- Content for stories ----
STORIES = {
    "funny": {
        "title": "The Confused Robot",
        "blurb": "A lighthearted, humorous tale",
        "text": (
            "Once upon a time in a sleek, high-tech city, a robot named Bolt decided he was tired "
            "of fixing circuits and programming satellites. He wanted something new. “I shall "
            "become… a chef!” he declared with confidence. His creator, Dr. Lemons, choked on his "
            "coffee but gave a supportive thumbs-up. Bolt dove headfirst into his new passion, "
            "downloading 1,200 cooking tutorials in 3 seconds. He chose his first recipe: vegetable "
            "soup. The instructions said, “Add water and let it simmer.” Bolt interpreted this quite "
            "literally... (keep reading for 5 minutes)"
        ),
    },
    "kids": {
        "title": "Stella the Smallest Star",
        "blurb": "A delightful story for children",
        "text": (
            "Far beyond the clouds lived Stella, the smallest star in her constellation. She wasn’t "
            "the brightest, but she was the bravest... (continue for 5 minutes)"
        ),
    },
    "ai": {
        "title": "Digital Eyes Open",
        "blurb": "A story about artificial intelligence",
        "text": (
            "In a quiet lab nestled within the hills of Silicon Valley, an AI named ARIA opened her "
            "digital eyes for the first time... (continue for 5 minutes)"
        ),
    },
    "classic": {
        "title": "The Ancient Oak",
        "blurb": "A timeless classic tale",
        "text": (
            "In the heart of a small village surrounded by meadows and hills stood an ancient oak... "
            "(continue for 5 minutes)"
        ),
    },
}

# ---- Load model + encoder (once) ----
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = Path("best_model.joblib")
    label_path = Path("label_encoder.joblib")
    if not (model_path.exists() and label_path.exists()):
        return None, None, None, "Model artifacts not found. Place best_model.joblib & label_encoder.joblib beside this script."
    try:
        model = joblib.load(model_path)
        le = joblib.load(label_path)
        # Try to read feature names or at least count
        names = getattr(model, "feature_names_in_", None)
        n_in = getattr(model, "n_features_in_", None)
        return model, le, (names, n_in), None
    except Exception as e:
        return None, None, None, f"Failed loading artifacts: {e}"

MODEL, LABELER, MODEL_META, ARTIFACT_ERR = load_artifacts()

def get_expected_feature_names() -> Optional[List[str]]:
    """
    Priority:
      1) feature_cols.json in the working folder
      2) model.feature_names_in_
      3) None (only count known)
    """
    json_path = Path("feature_cols.json")
    if json_path.exists():
        try:
            cols = json.loads(json_path.read_text())
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
        except Exception:
            pass
    names, _ = MODEL_META if MODEL_META else (None, None)
    if names is not None:
        return list(names)
    return None

# ---- Derived features for 10-col model ----
def build_10_from_video(feats: Dict[str, float]) -> Tuple[List[float], List[str]]:
    br = float(feats["blink_rate_bpm"])
    ibr = float(feats["incomplete_blink_ratio"])
    ibi = float(feats["avg_ibi_sec"])
    red = float(feats["redness_index"])

    ibr_x_red = ibr * red
    blink_per_sec = br / 60.0
    ibi_inv = 0.0 if ibi == 0 else 1.0 / ibi
    ibi_lt6 = 1.0 if ibi < 6.0 else 0.0
    red_gt0_3 = 1.0 if red > 0.30 else 0.0
    ibr_gt0_2 = 1.0 if ibr > 0.20 else 0.0

    vals = [br, ibr, ibi, red, ibr_x_red, blink_per_sec, ibi_inv, ibi_lt6, red_gt0_3, ibr_gt0_2]
    return vals, FEATURE_10

# --------------------------- Pages ---------------------------
def page_dashboard():
    st.title("Dry Eye Assessment – Dashboard")
    st.caption("This tool estimates DED risk using your OSDI score combined with video features.")

    st.subheader("OSDI – Quick Symptoms Questionnaire")
    st.caption("Scale: 0 = Never … 4 = Always")

    just_computed = False
    with st.form("osdi_form"):
        vals = []
        for i, q in enumerate(QUESTIONS):
            vals.append(st.radio(f"{i+1}. {q}", CHOICES, index=0, horizontal=True, key=f"osdi_q{i}"))
        if st.form_submit_button("Compute OSDI", use_container_width=True):
            st.session_state.osdi_score = compute_osdi(vals)
            just_computed = True

    if just_computed:
        st.success(f"Your OSDI Score: **{st.session_state.osdi_score}**")
        st.info("Next: pick a story to read while we record your face (5 minutes).")

    if st.session_state.osdi_score is not None:
        if st.button("Continue → Story Selection", type=PRIMARY_BTN, use_container_width=True):
            goto("stories")

def page_stories():
    st.title("Choose Your Story")
    st.caption("Select a category and then click **Select Story** to proceed to recording.")
    cols = st.columns(2)
    for i, key in enumerate(STORIES.keys()):
        with cols[i % 2]:
            with st.container(border=True):
                st.subheader(STORIES[key]["title"])
                st.caption(STORIES[key]["blurb"])
                st.write(STORIES[key]["text"][:220] + "…")
                st.button(
                    "Select Story",
                    key=f"sel_{key}",
                    on_click=lambda k=key: (setattr(st.session_state, "story_key", k), goto("record")),
                    type=PRIMARY_BTN,
                )

    st.button("← Back to Dashboard", on_click=lambda: goto("dashboard"))

def recorder_html() -> str:
    return """
    <style>
    .rec-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    video { width: 100%; max-height: 360px; background: #000; border-radius: 8px; }
    button { padding: 10px 16px; border-radius: 8px; border: 0; font-weight: 600; }
    .start { background: #0e78f9; color: #fff; }
    .stop  { background: #e11d48; color: #fff; }
    .disabled { opacity: .6; pointer-events: none; }
    .row { display: flex; gap: 12px; margin-top: 12px; }
    .timer { font-weight: 700; }
    </style>
    <div class="rec-wrap">
      <video id="preview" autoplay playsinline muted></video>
      <div class="row">
        <button class="start" id="startBtn">🎥 Start Recording</button>
        <button class="stop disabled" id="stopBtn">⏹ Stop</button>
        <span class="timer" id="timer">00:00 / 05:00</span>
      </div>
      <div id="after" style="margin-top:12px;"></div>
    </div>
    <script>
    const preview = document.getElementById('preview');
    const startBtn = document.getElementById('startBtn');
    const stopBtn  = document.getElementById('stopBtn');
    const timerEl  = document.getElementById('timer');
    const after    = document.getElementById('after');
    let mediaStream, recorder, chunks = [], ticker;

    function fmt(n){return String(n).padStart(2,'0');}
    function updateTimer(s){ const m=Math.floor(s/60), r=s%60; timerEl.textContent = `${fmt(m)}:${fmt(r)} / 05:00`; }

    async function start(){
      try{
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      }catch(err){ alert('Camera permission denied or unavailable.'); return; }
      preview.srcObject = mediaStream;
      chunks = [];
      recorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm;codecs=vp9' });
      recorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = onStop;
      recorder.start();
      startBtn.classList.add('disabled'); stopBtn.classList.remove('disabled');

      let s = 0; updateTimer(0);
      ticker = setInterval(()=>{ s+=1; updateTimer(s); if(s>=300){ stop(); } }, 1000);
    }

    function stop(){
      try{ recorder && recorder.state !== 'inactive' && recorder.stop(); }catch(_){}
      try{ mediaStream && mediaStream.getTracks().forEach(t => t.stop()); }catch(_){}
      clearInterval(ticker);
      startBtn.classList.remove('disabled'); stopBtn.classList.add('disabled');
    }

    function onStop(){
      const blob = new Blob(chunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      preview.srcObject = null; preview.src = url; preview.controls = true; preview.muted = false; preview.play();

      const a = document.createElement('a');
      a.href = url; a.download = `recording_${Date.now()}.webm`;
      a.textContent = '⬇️ Download Video';
      a.style = 'display:inline-block;margin-top:8px;padding:10px 16px;background:#0e78f9;color:#fff;border-radius:8px;text-decoration:none;font-weight:600';
      after.innerHTML = '<p>Recording complete! Download the video, then go to the next step to upload it for analysis.</p>';
      after.appendChild(a);
    }

    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    </script>
    """

def page_record():
    if st.session_state.story_key is None:
        st.warning("Please select a story first.")
        if st.button("Go to Story Selection"): goto("stories")
        return

    story = STORIES[st.session_state.story_key]
    st.title(story["title"])
    st.caption("Read the text while the camera records your face for exactly 5 minutes.")
    with st.container(border=True):
        st.write(story["text"])

    st.markdown("### Video Recording")
    st.components.v1.html(recorder_html(), height=520)

    c1, c2 = st.columns(2)
    with c1:
        st.button("← Back to Stories", on_click=lambda: goto("stories"), use_container_width=True)
    with c2:
        st.button("Continue → Dry Eye Risk Test", on_click=lambda: goto("predict"), type=PRIMARY_BTN, use_container_width=True)

# --------------------------- Prediction helpers ---------------------------
def build_10_from_video(feats: Dict[str, float]) -> Tuple[List[float], List[str]]:
    br = float(feats["blink_rate_bpm"])
    ibr = float(feats["incomplete_blink_ratio"])
    ibi = float(feats["avg_ibi_sec"])
    red = float(feats["redness_index"])

    ibr_x_red = ibr * red
    blink_per_sec = br / 60.0
    ibi_inv = 0.0 if ibi == 0 else 1.0 / ibi
    ibi_lt6 = 1.0 if ibi < 6.0 else 0.0
    red_gt0_3 = 1.0 if red > 0.30 else 0.0
    ibr_gt0_2 = 1.0 if ibr > 0.20 else 0.0

    vals = [br, ibr, ibi, red, ibr_x_red, blink_per_sec, ibi_inv, ibi_lt6, red_gt0_3, ibr_gt0_2]
    return vals, FEATURE_10

def build_input_vector(feats: Dict[str, float], osdi: float) -> Tuple[np.ndarray, str]:
    names, n_in = MODEL_META if MODEL_META else (None, None)
    expected_names = None

    # If a sidecar JSON exists, it overrides
    json_path = Path("feature_cols.json")
    if json_path.exists():
        try:
            expected_names = json.loads(json_path.read_text())
        except Exception:
            expected_names = None

    # Fall back to model.feature_names_in_
    if expected_names is None and names is not None:
        expected_names = list(names)

    # Fall back to count-only
    expected_count = int(n_in) if n_in is not None else (len(expected_names) if expected_names else None)

    # --- Path A: 10-feature video-only model ---
    def is_10_match():
        if expected_names is not None:
            return len(expected_names) == 10 and set(expected_names) == set(FEATURE_10)
        return expected_count == 10

    if is_10_match():
        vals10, order = build_10_from_video(feats)
        if expected_names is not None and expected_names != order:
            mapping = dict(zip(order, vals10))
            vals10 = [float(mapping[n]) for n in expected_names]
        X = np.array([vals10], dtype=float)
        return X, "10"

    # --- Path B: 5-feature video + OSDI model ---
    def is_5_match():
        if expected_names is not None:
            return len(expected_names) == 5 and set(expected_names) == set(FEATURE_5)
        return expected_count == 5

    if is_5_match():
        row_map = {
            "blink_rate_bpm": float(feats["blink_rate_bpm"]),
            "incomplete_blink_ratio": float(feats["incomplete_blink_ratio"]),
            "avg_ibi_sec": float(feats["avg_ibi_sec"]),
            "redness_index": float(feats["redness_index"]),
            "osdi_score": float(osdi),
        }
        order = expected_names if expected_names is not None else FEATURE_5
        X = np.array([[row_map[col] for col in order]], dtype=float)
        return X, "5"

    # Unsupported model layout
    st.error(
        "The loaded model expects a feature set that this app does not support.\n\n"
        "Fix one of the following:\n"
        f"1) Train a **10-feature video-only** model with columns:\n   {FEATURE_10}\n"
        f"2) Train a **5-feature** model (4 video + osdi_score) with columns:\n   {FEATURE_5}\n"
        "3) Provide an exact **feature_cols.json** (list of column names in order) matching either option above."
    )
    st.stop()

# --------------------------- Pages ---------------------------
def page_predict():
    st.title("Dry Eye Risk – Upload & Test")
    if ARTIFACT_ERR:
        st.error(ARTIFACT_ERR); st.stop()

    if st.session_state.osdi_score is None:
        st.warning("You haven't computed an OSDI score yet. You can proceed, but the app will use 0.0 as a fallback.")
    st.caption("Upload the 5-minute video you just downloaded. Accepted: .webm, .mp4, .mov, .avi")

    up = st.file_uploader("Upload your recorded video", type=["webm", "mp4", "mov", "avi"], key="video_uploader")
    if up is not None:
        # Create a light token from name+size to avoid reprocessing the same upload on reruns
        token = f"{up.name}:{up.size}"
        if st.session_state.last_processed_token == token and st.session_state.pred_label is not None:
            goto("result")
            force_rerun()

        st.session_state.video_uploaded_name = up.name
        tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmpdir.name) / up.name
        with open(tmp_path, "wb") as f:
            f.write(up.read())
        st.success(f"Video uploaded: **{up.name}**")

        # Feature extraction
        try:
            extractor = importlib.import_module("extract_features")
            if not hasattr(extractor, "extract_from_video"):
                st.error("`extract_features.py` must define `extract_from_video(video_path) -> dict` "
                         f"returning: {', '.join(VIDEO_FEATURES)}"); st.stop()
        except Exception as e:
            st.error(f"Couldn't import `extract_features.py`: {e}"); st.stop()

        with st.spinner("Extracting video features…"):
            feats: Dict[str, float] = extractor.extract_from_video(str(tmp_path))

        missing = [c for c in VIDEO_FEATURES if c not in feats]
        if missing:
            st.error(f"Extractor missing keys: {missing}\nReturned: {feats}"); st.stop()

        with st.expander("See extracted features"):
            st.json(feats)

        # Build X strictly (no padding)
        osdi = float(st.session_state.osdi_score or 0.0)
        X, mode = build_input_vector(feats, osdi)

        with st.spinner("Predicting risk…"):
            y_hat = MODEL.predict(X)[0]
            try:
                label = LABELER.inverse_transform([y_hat])[0]
            except Exception:
                label = str(y_hat)

        st.session_state.prediction = {"features": feats, "osdi": osdi, "raw": int(y_hat), "mode": mode}
        st.session_state.pred_label = label
        st.session_state.last_processed_token = token  # mark as processed

        goto("result")
        force_rerun()

    st.button("← Back to Recording", on_click=lambda: goto("record"))

def page_result():
    st.title("Your Result")
    if st.session_state.pred_label is None:
        st.warning("No prediction found yet.")
        st.button("Go to Prediction", on_click=lambda: goto("predict"))
        return

    label = st.session_state.pred_label
    if label == "High":
        header = "🔴 High Risk"
    elif label == "Medium":
        header = "🟠 Moderate Risk"
    else:
        header = "🟢 Low Risk"
    st.header(header)

    mode = st.session_state.prediction.get("mode")
    if mode == "10":
        st.caption("This prediction used **video-only engineered features (10 columns)**.")
    elif mode == "5":
        st.caption("This prediction used **4 video features + your OSDI score (5 columns)**.")

    with st.container(border=True):
        feats = st.session_state.prediction["features"]
        osdi = st.session_state.prediction["osdi"]
        st.subheader("Inputs used")
        if mode == "5":
            st.write(f"**OSDI Score:** {osdi}")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"- Blink rate: **{feats['blink_rate_bpm']}** blinks/min")
            st.write(f"- Avg inter-blink interval: **{feats['avg_ibi_sec']}** sec")
        with c2:
            st.write(f"- Incomplete blink ratio: **{feats['incomplete_blink_ratio']}**")
            st.write(f"- Redness index: **{feats['redness_index']}**")

    st.subheader("Recommendations & Tips")
    if label == "Low":
        st.success("Your Dry Eye Risk is Low. Great! Keep it up!")
        tips = [
            "Blink regularly, especially during screen time.",
            "Stay hydrated throughout the day.",
            "Take 20-20-20 breaks (every 20 minutes, look 20 feet away for 20 seconds).",
            "Use a humidifier in dry environments.",
            "Maintain a balanced diet rich in omega-3 fatty acids.",
        ]
    elif label == "Medium":
        st.warning("Your Dry Eye Risk is Moderate.")
        tips = [
            "All Low-risk tips, plus:",
            "Reduce continuous screen sessions; increase frequency of breaks.",
            "Consider lubricating eye drops if appropriate.",
            "Check screen height and lighting to reduce strain.",
        ]
    else:
        st.error("Your Dry Eye Risk is High. Please consider consulting an eye-care professional.")
        tips = [
            "Use preservative-free artificial tears as advised by a clinician.",
            "Reduce prolonged screen exposure; adopt fixed rest intervals.",
            "Avoid direct airflow to the eyes (fans/AC).",
            "Seek a professional evaluation for dry eye and meibomian gland function.",
        ]
    for i, t in enumerate(tips, 1):
        st.write(f"{i}. {t}")

    c1, c2 = st.columns(2)
    with c1:
        st.button("Test Another Video", on_click=lambda: goto("predict"), use_container_width=True)
    with c2:
        st.button("Back to Dashboard", on_click=lambda: goto("dashboard"), type=NEUTRAL_BTN, use_container_width=True)

# --------------------------- Router ---------------------------
PAGE_MAP = {
    "dashboard": page_dashboard,
    "stories": page_stories,
    "record": page_record,
    "predict": page_predict,
    "result": page_result,
}

PAGE_MAP[st.session_state.page]()
