# app.py
from __future__ import annotations

from pathlib import Path
import importlib
import tempfile
from typing import Dict, List, Tuple, Optional

import streamlit as st

from ui import inject_css, stepper, recorder_html
from osdi import QUESTIONS, OSDI_OPTIONS, compute_osdi, osdi_severity
from ml import VIDEO_FEATURES, FEATURE_5, load_artifacts, build_input_vector

# ---------------- App config ----------------
st.set_page_config(page_title="Dry Eye Risk – Assessment", layout="centered")
inject_css()
PRIMARY_BTN, NEUTRAL_BTN = "primary", "secondary"
DURATION_SEC = 360  # 6 minutes

# ---------------- Session state ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("page", "dashboard")
    ss.setdefault("osdi_score", None)
    ss.setdefault("story_key", None)
    ss.setdefault("prediction", None)
    ss.setdefault("pred_label", None)
    ss.setdefault("video_uploaded_name", None)
    ss.setdefault("last_processed_token", None)
init_state()

def goto(page: str): st.session_state.page = page

def force_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------------- Stories (6-min reads) ----------------
T_SUFFIX = " Keep reading at a natural pace for the full six minutes to capture consistent facial features."
STORIES: Dict[str, Dict[str, str]] = {
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
            "It became a sensation. People cried. The toaster industry held an emergency summit. Bolt, at last, felt purpose."
            + T_SUFFIX
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
            "Stella realized that size didn’t measure kindness and that even a small light can change a big night."
            + T_SUFFIX
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
            "When storms rolled in, she watched raindrops stitch silver threads across windows and decided she loved the world for its textures."
            + T_SUFFIX
        ),
    },
    "classic": {
        "title": "The Ancient Oak",
        "blurb": "A timeless countryside tale",
        "text": (
            "In a valley of whispering grass stood an ancient oak. It had watched generations come and go, weddings and farewells, "
            "mornings bright as brass and evenings blue as ink. Children climbed its arms to learn the sky; elders leaned against its bark to rest their memories. "
            "When drought arrived, the oak rationed shade; when storms returned, it held the soil together. Travelers said the wind in its leaves sounded like pages "
            "turning, as if the tree were reading the earth its favorite book."
            + T_SUFFIX
        ),
    },
}

# ---------------- Model artifacts ----------------
MODEL, LABELER, MODEL_META, ARTIFACT_ERR = load_artifacts()

# ---------------- Pages ----------------
def page_dashboard():
    stepper("dashboard")
    st.title("Dry Eye Assessment")
    st.caption("Estimate your DED risk using a short questionnaire and a 6-minute facial video.")

    st.subheader("OSDI – Quick Symptoms Questionnaire")
    st.caption("Choose the option that best describes your experience **over the last week**.")

    just = False
    with st.form("osdi_form"):
        numeric_values: List[int] = []
        label_to_val = dict(OSDI_OPTIONS)
        labels_only = [t for t, _ in OSDI_OPTIONS]
        for i, q in enumerate(QUESTIONS):
            label_chosen = st.radio(
                f"{i+1}. {q}",
                options=labels_only,
                index=0,
                horizontal=True,
                key=f"osdi_q{i}",
                help="Never (0%) · Rarely (≤25%) · Sometimes (~50%) · Often (~75%) · Always (100%)",
            )
            numeric_values.append(label_to_val[label_chosen])
        if st.form_submit_button("Compute OSDI", use_container_width=True):
            st.session_state.osdi_score = compute_osdi(numeric_values)
            just = True

    if just:
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
    st.components.v1.html(recorder_html(DURATION_SEC), height=520)

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
    st.markdown('<div class="upl-note">Limit 600MB per file (configure in .streamlit/config.toml).</div>', unsafe_allow_html=True)

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

        # Build model input
        osdi = float(st.session_state.osdi_score or 0.0)
        X, mode = build_input_vector(feats, osdi)

        # Predict
        with st.spinner("Predicting risk…"):
            y_hat = load_artifacts()[0].predict(X)[0]  # MODEL
            try:
                label = load_artifacts()[1].inverse_transform([y_hat])[0]  # LABELER
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
    st.markdown(
        f'<div class="card">{header} &nbsp; '
        f'<span class="upl-note">The model combines your OSDI score (if used) with video features from your 6-minute recording.</span>'
        f'</div>',
        unsafe_allow_html=True
    )

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
        with c2:
            st.write(f"- Avg inter-blink interval: **{feats['avg_ibi_sec']}** sec")
        c3, c4 = st.columns(2)
        with c3:
            st.write(f"- Incomplete blink ratio: **{feats['incomplete_blink_ratio']}**")
        with c4:
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

# ---------------- Router ----------------
PAGE_MAP = {
    "dashboard": page_dashboard,
    "stories": page_stories,
    "record": page_record,
    "predict": page_predict,
    "result": page_result,
}
PAGE_MAP[st.session_state.page]()
