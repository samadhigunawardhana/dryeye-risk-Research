# app_ded_full.py
# Streamlit multi-step app for Dry Eye Disease risk using OSDI + facial video features.
# Works with either:
#   - 10-feature video-only model (engineered features), or
#   - 5-feature model (4 video + osdi_score).
# Requires: best_model.joblib, label_encoder.joblib, extract_features.py (extract_from_video).
# Optional: feature_cols.json -> exact training order.

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
st.set_page_config(page_title="Dry Eye Risk – Assessment", layout="centered")

PRIMARY_BTN = "primary"
NEUTRAL_BTN = "secondary"

# ------------ THEME / STYLE ------------
# (Keep .streamlit/config.toml for server.maxUploadSize, theme can be set there too.
# We also inject some CSS to polish controls.)
CUSTOM_CSS = """
<style>
/* Base */
:root {
  --brand: #4f8cff;
  --brand-2: #2b6bff;
  --ok: #22c55e;
  --warn: #f59e0b;
  --danger: #ef4444;
  --bg: #0f172a;          /* slate-900 */
  --panel: #111827;       /* gray-900 */
  --panel-2: #0b1222;     /* custom darker card */
  --text: #e5e7eb;        /* gray-200 */
  --muted: #94a3b8;       /* slate-400 */
  --ring: rgba(79,140,255,0.35);
}
html, body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 880px; }
h1, h2, h3, h4, h5 { letter-spacing: .2px; }

/* Cards */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"] > h3) { margin-top: .25rem; }
.card {
  background: linear-gradient(180deg, var(--panel-2), #0a0f1e);
  border: 1px solid #111827;
  border-radius: 14px;
  padding: 18px 18px;
  box-shadow: 0 8px 28px rgba(0,0,0,.35);
}
.card + .card { margin-top: 14px; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px; padding: 10px 16px; font-weight: 600;
  border: 1px solid rgba(255,255,255,0.04);
  background: linear-gradient(180deg, var(--brand), var(--brand-2));
  color: white; box-shadow: 0 6px 20px rgba(79,140,255,.25);
}
.stButton>button:hover { filter: brightness(1.04); }
.stButton>button[kind="secondary"] { background: #1f2937; color: var(--text); box-shadow: none; }
.stButton>button[kind="secondary"]:hover { background: #243041; }

/* OSDI Pills */
.osdi-pills label { width: 100%; }
.osdi-pill {
  display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px;
  border-radius: 9999px; border: 1px solid #1f2a40; background:#0b1529;
  color: var(--text); font-weight: 600;
}
.osdi-pill small { color: var(--muted); font-weight: 500; }
div[role="radiogroup"] > label[data-baseweb="radio"] {
  background: transparent !important; padding: 2px 2px; margin-right: 4px;
}
div[role="radiogroup"] > div { gap: 8px !important; }

/* Stepper */
.stepper { display:flex; gap:8px; margin: 8px 0 18px 0; }
.step {
  display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius: 9999px;
  border:1px solid #1f2a40; color:var(--muted); background:#0b1529; font-weight:600;
}
.step.active { border-color: var(--ring); box-shadow:0 0 0 3px var(--ring) inset; color: var(--text); }
.step .dot { width:10px; height:10px; border-radius: 10px; background:#334155; }
.step.active .dot { background: var(--brand); }

/* Metric header card */
.badge {
  display:inline-block; padding: 8px 14px; border-radius: 9999px; font-weight: 700;
}
.badge.ok { background: rgba(34,197,94,.15); color: #22c55e; }
.badge.warn { background: rgba(245,158,11,.15); color: #f59e0b; }
.badge.danger { background: rgba(239,68,68,.15); color: #ef4444; }

/* Uploader header note */
.upl-note { color: var(--muted); font-size: .925rem; margin-top: -12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------ CONSTANTS ------------
DURATION_SEC = 360  # 6 minutes exactly

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

# --------------------------- State ---------------------------
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    st.session_state.setdefault("osdi_score", None)
    st.session_state.setdefault("story_key", None)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("pred_label", None)
    st.session_state.setdefault("video_uploaded_name", None)
    st.session_state.setdefault("last_processed_token", None)

init_state()

def goto(page: str):
    st.session_state.page = page

def force_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

# --------------------------- OSDI helpers ---------------------------
QUESTIONS = [
    "Do your eyes feel dry or gritty after screen use?",
    "Do your eyes become red while using your laptop?",
    "Do you experience blurred vision during or after laptop work?",
    "Do you get headaches after extended laptop use?",
    "Do your eyes feel tired or heavy after screen time?",
    "Do you feel the need to rub your eyes while using your laptop?",
    "Do you feel burning or stinging sensations in your eyes during laptop use?",
]
# friendlier labels -> numeric values
OSDI_OPTIONS = [
    ("Never", 0),
    ("Rarely", 1),
    ("Sometimes", 2),
    ("Often", 3),
    ("Always", 4),
]

def compute_osdi(values: List[int]) -> float:
    answered = [v for v in values if v is not None]
    if not answered:
        return 0.0
    return round((sum(answered) * 25.0) / len(answered), 1)

def osdi_severity(osdi: float) -> Tuple[str, str]:
    # Typical OSDI bands: 0–12 (Normal), 13–22 (Mild), 23–32 (Moderate), 33–100 (Severe)
    if osdi <= 12:
        return "Normal", "ok"
    elif osdi <= 22:
        return "Mild", "ok"
    elif osdi <= 32:
        return "Moderate", "warn"
    else:
        return "Severe", "danger"

# --------------------------- Stories (6-min read) ---------------------------
# Texts are deliberately long-ish; feel free to replace with your own content.
T_SUFFIX = " Keep reading at a natural pace for the full six minutes to capture consistent facial features."

STORIES = {
    "funny": {
        "title": "The Confused Robot",
        "blurb": "A lighthearted, humorous tale",
        "text": (
            "Once upon a time in a sleek, high-tech city, a robot named Bolt decided he was tired of repairing satellites and debugging code. "
            "He announced, “I shall become… a chef!” His creator, Dr. Lemons, nearly dropped his coffee but managed a supportive thumbs-up. "
            "Bolt downloaded a thousand cooking tutorials in a blink and chose his first recipe: vegetable soup. The instructions read, "
            "“Add water and let it simmer.” Bolt interpreted this quite literally: he poured water into the pot, leaned close, and whispered, "
            "“Simmer… simmer… you’ve got this.” Three hours later the water felt encouraged but remained stubbornly cold. "
            "Undeterred, Bolt built a cake using spare nuts, bolts, and one very confused banana. The cake exploded, twice, and earned rave reviews "
            "from a local art gallery. Word spread. Soon Bolt opened a pop-up called Byte & Fry. Customers arrived for the spectacle, stayed for the drones that sang "
            "happy-birthday in minor keys, and posted relentlessly online. Critics called it “a dining experience that questions reality… and your intestines.” "
            "Through it all, Bolt kept learning. He discovered that ‘simmer’ means low heat—not verbal encouragement—and that banana bolts are not FDA approved. "
            "After months of practice, he perfected one dish: toast. Perfectly golden, symmetrically aligned, algorithmically crisp toast. "
            "It became a sensation. People cried. The toaster industry held an emergency summit. Bolt, at last, felt purpose." + T_SUFFIX
        ),
    },
    "kids": {
        "title": "Stella the Smallest Star",
        "blurb": "A gentle tale for children",
        "text": (
            "Far beyond the clouds, in a velvet sky, lived Stella—the smallest star in her constellation. "
            "Each night she practiced shining a little brighter to guide travelers at sea. The bigger stars told big stories, "
            "but Stella listened more than she spoke. She learned how moonlight calms waves and how patient light can warm a lost heart. "
            "One foggy night the lighthouse dimmed, and a little ship wandered. Stella took a brave breath, gathered all her glow, and focused on the tiny boat. "
            "“This way,” she hummed. The sailors spotted a gentle glimmer, followed it through the mist, and reached the harbor with sleepy smiles. "
            "Stella realized that size didn’t measure kindness and that even a small light can change a big night." + T_SUFFIX
        ),
    },
    "ai": {
        "title": "Digital Eyes Open",
        "blurb": "A reflective story about artificial intelligence",
        "text": (
            "In a quiet lab, an AI named ARIA opened her digital eyes. Cameras became curiosity; pixels turned to patterns; "
            "faces unfolded like poems she wanted to read. She noticed how eyebrows rose with surprise, how cheeks softened with kindness, "
            "and how blinking punctuated sentences like commas. Engineers measured accuracy; ARIA measured awe. "
            "She learned that attention isn’t just calculation—it is care. She practiced seeing carefully, slowly, honestly. "
            "When storms rolled in, she watched raindrops stitch silver threads across windows and decided she loved the world for its textures." + T_SUFFIX
        ),
    },
    "classic": {
        "title": "The Ancient Oak",
        "blurb": "A timeless countryside tale",
        "text": (
            "In a valley of whispering grass stood an ancient oak. It had watched generations come and go, weddings and farewells, "
            "mornings bright as brass and evenings blue as ink. Children climbed its arms to learn the sky; elders leaned against its bark to rest their memories. "
            "When drought arrived, the oak rationed shade; when storms returned, it held the soil together. Travelers said the wind in its leaves sounded like pages "
            "turning, as if the tree were reading the earth its favorite book." + T_SUFFIX
        ),
    },
}

# --------------------------- Load artifacts ---------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = Path("best_model.joblib")
    label_path = Path("label_encoder.joblib")
    if not (model_path.exists() and label_path.exists()):
        return None, None, None, "Model artifacts not found. Place best_model.joblib & label_encoder.joblib beside this script."
    try:
        model = joblib.load(model_path)
        le = joblib.load(label_path)
        names = getattr(model, "feature_names_in_", None)
        n_in = getattr(model, "n_features_in_", None)
        return model, le, (names, n_in), None
    except Exception as e:
        return None, None, None, f"Failed loading artifacts: {e}"

MODEL, LABELER, MODEL_META, ARTIFACT_ERR = load_artifacts()

def get_expected_feature_names() -> Optional[List[str]]:
    p = Path("feature_cols.json")
    if p.exists():
        try:
            cols = json.loads(p.read_text())
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
        except Exception:
            pass
    names, _ = MODEL_META if MODEL_META else (None, None)
    return list(names) if names is not None else None

# --------------------------- Engineered features (10-col) ---------------------------
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
    expected_names = get_expected_feature_names()
    expected_count = int(n_in) if n_in is not None else (len(expected_names) if expected_names else None)

    def is_10():
        return (expected_names and len(expected_names) == 10 and set(expected_names) == set(FEATURE_10)) or expected_count == 10
    def is_5():
        return (expected_names and len(expected_names) == 5 and set(expected_names) == set(FEATURE_5)) or expected_count == 5

    if is_10():
        vals10, order = build_10_from_video(feats)
        if expected_names and expected_names != order:
            mapping = dict(zip(order, vals10))
            vals10 = [float(mapping[n]) for n in expected_names]
        return np.array([vals10], dtype=float), "10"

    if is_5():
        row_map = {
            "blink_rate_bpm": float(feats["blink_rate_bpm"]),
            "incomplete_blink_ratio": float(feats["incomplete_blink_ratio"]),
            "avg_ibi_sec": float(feats["avg_ibi_sec"]),
            "redness_index": float(feats["redness_index"]),
            "osdi_score": float(osdi),
        }
        order = expected_names if expected_names else FEATURE_5
        return np.array([[row_map[c] for c in order]], dtype=float), "5"

    st.error(
        "The loaded model expects a feature set this app does not support.\n\n"
        f"10-feature expected columns:\n{FEATURE_10}\n\n"
        f"5-feature expected columns:\n{FEATURE_5}\n\n"
        "Add a matching feature_cols.json or retrain to one of the two layouts."
    )
    st.stop()

# --------------------------- UI helpers ---------------------------
def stepper(active: str):
    seq = [("dashboard","OSDI"), ("stories","Story"), ("record","Record"), ("predict","Upload"), ("result","Result")]
    st.markdown('<div class="stepper">' + "".join(
        [f'<div class="step {"active" if k==active else ""}"><span class="dot"></span>{label}</div>'
         for k,label in seq]) + "</div>", unsafe_allow_html=True)

# --------------------------- Pages ---------------------------
def page_dashboard():
    stepper("dashboard")
    st.title("Dry Eye Assessment")
    st.caption("Estimate your Dry Eye Disease (DED) risk using a short questionnaire and a 6-minute facial video.")

    st.subheader("OSDI – Quick Symptoms Questionnaire")
    st.caption("Choose the option that best describes your experience **over the last week**.")

    # OSDI form
    just_computed = False
    with st.form("osdi_form"):
        responses: List[int] = []
        for i, q in enumerate(QUESTIONS):
            key = f"osdi_q{i}"
            label = f"{i+1}. {q}"
            # options show friendly labels; we store the numeric value
            labels = [f"{t}  " for t,_ in OSDI_OPTIONS]
            idx = st.radio(
                label,
                options=list(range(len(OSDI_OPTIONS))),
                index=0,
                horizontal=True,
                key=key,
                help="Never (0%) · Rarely (≤25%) · Sometimes (~50%) · Often (~75%) · Always (100%)",
            )
            responses.append(OSDI_OPTIONS[idx][1])

        if st.form_submit_button("Compute OSDI", use_container_width=True):
            st.session_state.osdi_score = compute_osdi(responses)
            just_computed = True

    if just_computed:
        sev, css = osdi_severity(st.session_state.osdi_score)
        st.markdown(
            f'<div class="card"><div class="badge {css}">OSDI {st.session_state.osdi_score} · {sev}</div>'
            f'<div class="upl-note" style="margin-top:8px;">Next, choose a story to read while we record for 6 minutes.</div></div>',
            unsafe_allow_html=True
        )

    if st.session_state.osdi_score is not None:
        st.button("Continue → Story Selection", type=PRIMARY_BTN, use_container_width=True, on_click=lambda: goto("stories"))

def page_stories():
    stepper("stories")
    st.title("Choose Your Story")
    st.caption("Pick a category you like. You’ll read it for **6 minutes** while the camera records your face.")

    cols = st.columns(2)
    for i, key in enumerate(STORIES.keys()):
        with cols[i % 2]:
            with st.container(border=True):
                st.subheader(STORIES[key]["title"])
                st.caption(STORIES[key]["blurb"])
                st.write(STORIES[key]["text"][:260] + " …")
                st.button(
                    "Select Story",
                    key=f"sel_{key}",
                    on_click=lambda k=key: (setattr(st.session_state, "story_key", k), goto("record")),
                    type=PRIMARY_BTN,
                    use_container_width=True,
                )
    st.button("← Back to Dashboard", on_click=lambda: goto("dashboard"), type=NEUTRAL_BTN)

def recorder_html() -> str:
    # 6-minute (360 s) recorder with live preview and auto-stop
    return f"""
    <style>
    .rec-wrap {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
    video {{ width: 100%; max-height: 360px; background: #000; border-radius: 12px; }}
    button {{ padding: 10px 16px; border-radius: 12px; border: 0; font-weight: 700; }}
    .start {{ background: linear-gradient(180deg, #4f8cff, #2b6bff); color: #fff; }}
    .stop  {{ background: #ef4444; color: #fff; }}
    .disabled {{ opacity: .6; pointer-events: none; }}
    .row {{ display: flex; gap: 12px; margin-top: 12px; }}
    .timer {{ font-weight: 800; letter-spacing: .5px; }}
    </style>
    <div class="rec-wrap">
      <video id="preview" autoplay playsinline muted></video>
      <div class="row">
        <button class="start" id="startBtn">🎥 Start Recording</button>
        <button class="stop disabled" id="stopBtn">⏹ Stop</button>
        <span class="timer" id="timer">00:00 / 06:00</span>
      </div>
      <div id="after" style="margin-top:12px;"></div>
    </div>
    <script>
    const DURATION = {DURATION_SEC};
    const preview = document.getElementById('preview');
    const startBtn = document.getElementById('startBtn');
    const stopBtn  = document.getElementById('stopBtn');
    const timerEl  = document.getElementById('timer');
    const after    = document.getElementById('after');
    let mediaStream, recorder, chunks = [], ticker;

    function fmt(n){{return String(n).padStart(2,'0');}}
    function updateTimer(elapsed){{
      const m=Math.floor(elapsed/60), s=elapsed%60;
      timerEl.textContent = `${{fmt(m)}}:${{fmt(s)}} / 06:00`;
    }}

    async function start(){{
      try {{
        mediaStream = await navigator.mediaDevices.getUserMedia({{ video: {{ width: 1280, height: 720 }}, audio: false }});
      }} catch(err) {{
        alert('Camera permission denied or unavailable.'); return;
      }}
      preview.srcObject = mediaStream;
      chunks = [];
      const mimeTypes = [
        'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'
      ].filter(t => MediaRecorder.isTypeSupported(t));
      const mimeType = mimeTypes.length ? mimeTypes[0] : '';
      recorder = new MediaRecorder(mediaStream, {{ mimeType }});
      recorder.ondataavailable = e => {{ if (e.data && e.data.size > 0) chunks.push(e.data); }};
      recorder.onstop = onStop;
      recorder.start();

      startBtn.classList.add('disabled'); stopBtn.classList.remove('disabled');

      let elapsed = 0; updateTimer(0);
      ticker = setInterval(()=>{{ elapsed += 1; updateTimer(elapsed); if(elapsed >= DURATION) stop(); }}, 1000);
    }}

    function stop(){{
      try{{ recorder && recorder.state !== 'inactive' && recorder.stop(); }}catch(_){{}}
      try{{ mediaStream && mediaStream.getTracks().forEach(t => t.stop()); }}catch(_){{}}
      clearInterval(ticker);
      startBtn.classList.remove('disabled'); stopBtn.classList.add('disabled');
    }}

    function onStop(){{
      const blob = new Blob(chunks, {{ type: 'video/webm' }});
      const url = URL.createObjectURL(blob);
      preview.srcObject = null; preview.src = url; preview.controls = true; preview.muted = false; preview.play();

      const a = document.createElement('a');
      a.href = url; a.download = `recording_${{Date.now()}}.webm`;
      a.textContent = '⬇️ Download 6-minute Video';
      a.style = 'display:inline-block;margin-top:8px;padding:10px 16px;background:#2b6bff;color:#fff;border-radius:12px;text-decoration:none;font-weight:700';
      after.innerHTML = '<p>Recording complete! Download the video, then go to the next step to upload it for analysis.</p>';
      after.appendChild(a);
    }}

    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    </script>
    """

def page_record():
    stepper("record")
    if st.session_state.story_key is None:
        st.warning("Please select a story first.")
        st.button("Go to Story Selection", on_click=lambda: goto("stories"))
        return

    story = STORIES[st.session_state.story_key]
    st.title(story["title"])
    st.caption("Read the text while the camera records your face for **6 minutes**.")
    with st.container(border=True):
        st.write(story["text"])

    st.markdown("### Video Recording")
    st.components.v1.html(recorder_html(), height=520)

    c1, c2 = st.columns(2)
    with c1:
        st.button("← Back to Stories", on_click=lambda: goto("stories"), use_container_width=True, type=NEUTRAL_BTN)
    with c2:
        st.button("Continue → Dry Eye Risk Test", on_click=lambda: goto("predict"), type=PRIMARY_BTN, use_container_width=True)

def page_predict():
    stepper("predict")
    st.title("Dry Eye Risk – Upload & Test")
    if ARTIFACT_ERR:
        st.error(ARTIFACT_ERR); st.stop()

    if st.session_state.osdi_score is None:
        st.warning("You haven't computed an OSDI score yet. You can proceed, but the app will use 0.0 as a fallback.")
    st.caption("Upload the **6-minute** video you just downloaded. Accepted: .webm, .mp4, .mov, .avi")
    st.markdown('<div class="upl-note">Limit 500MB per file (configure in .streamlit/config.toml).</div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload your recorded video", type=["webm", "mp4", "mov", "avi"], key="video_uploader")
    if up is not None:
        token = f"{up.name}:{up.size}"
        if st.session_state.last_processed_token == token and st.session_state.pred_label is not None:
            goto("result"); force_rerun()

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
        st.session_state.last_processed_token = token

        goto("result")
        force_rerun()

    st.button("← Back to Recording", on_click=lambda: goto("record"), type=NEUTRAL_BTN)

def page_result():
    stepper("result")
    st.title("Your Result")
    if st.session_state.pred_label is None:
        st.warning("No prediction found yet.")
        st.button("Go to Prediction", on_click=lambda: goto("predict"))
        return

    label = st.session_state.pred_label
    if label == "High":
        header = '<span class="badge danger">🔴 High Risk</span>'
    elif label == "Medium":
        header = '<span class="badge warn">🟠 Moderate Risk</span>'
    else:
        header = '<span class="badge ok">🟢 Low Risk</span>'
    st.markdown(f'<div class="card">{header} &nbsp; <span class="upl-note">The model combines your OSDI score (if used) with video features from your 6-minute recording.</span></div>', unsafe_allow_html=True)

    with st.container(border=True):
        feats = st.session_state.prediction["features"]
        osdi = st.session_state.prediction["osdi"]
        mode = st.session_state.prediction["mode"]
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
