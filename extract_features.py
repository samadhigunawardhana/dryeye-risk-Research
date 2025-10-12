# extract_features.py
# Usage from terminal (optional):
#   python extract_features.py                     # process all *.mp4 in folder
#   python extract_features.py P01_2025....mp4     # process one file
#
# What the dashboard needs:
#   A function named extract_from_video(video_path) -> dict with keys:
#     blink_rate_bpm, incomplete_blink_ratio, avg_ibi_sec, redness_index

import sys, cv2, math, csv, re, numpy as np
import mediapipe as mp
from pathlib import Path

mp_face = mp.solutions.face_mesh

# Eye rings (outer contour) and iris landmarks (MediaPipe indices)
L_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
R_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
L_IRIS = [469, 470, 471, 472]
R_IRIS = [474, 475, 476, 477]

def dist(a, b): 
    return math.dist(a, b)

def ear(landmarks, w, h, left=True):
    """6-point Eye Aspect Ratio (EAR)."""
    idx = [33,160,158,133,153,144] if left else [263,387,385,362,380,373]
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
    p1,p2,p3,p4,p5,p6 = pts
    return (dist(p2,p6) + dist(p3,p5)) / (2.0 * dist(p1,p4) + 1e-6)

def poly_mask(h, w, pts):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask

def circle_from_pts(pts):
    """Crude circle from 4 iris points: center=mean, radius=mean distance."""
    cx = np.mean([p[0] for p in pts]); cy = np.mean([p[1] for p in pts])
    r  = np.mean([math.hypot(p[0]-cx, p[1]-cy) for p in pts])
    return (int(cx), int(cy), int(max(r, 1)))

def redness_index(frame, landmarks):
    """Approx sclera redness: mean R/(R+G+B) over (eye ring - iris) on bright pixels."""
    h, w = frame.shape[:2]
    l_poly = [(landmarks[i].x*w, landmarks[i].y*h) for i in L_EYE_RING]
    r_poly = [(landmarks[i].x*w, landmarks[i].y*h) for i in R_EYE_RING]
    l_iris_pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in L_IRIS]
    r_iris_pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in R_IRIS]
    lcx,lcy,lr = circle_from_pts(l_iris_pts)
    rcx,rcy,rr = circle_from_pts(r_iris_pts)

    mask_l = poly_mask(h,w,l_poly); mask_r = poly_mask(h,w,r_poly)
    iris_l = np.zeros((h,w), np.uint8); cv2.circle(iris_l,(lcx,lcy), int(lr*1.2), 255,-1)
    iris_r = np.zeros((h,w), np.uint8); cv2.circle(iris_r,(rcx,rcy), int(rr*1.2), 255,-1)

    sclera_mask = cv2.bitwise_or(mask_l, mask_r)
    sclera_mask = cv2.bitwise_and(sclera_mask, cv2.bitwise_not(cv2.bitwise_or(iris_l, iris_r)))

    b,g,r = cv2.split(frame)
    sum_bgr = b.astype(np.float32)+g.astype(np.float32)+r.astype(np.float32)+1e-6
    red_ratio = r.astype(np.float32)/sum_bgr

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    valid = cv2.bitwise_and(sclera_mask, bright)
    vals = red_ratio[valid>0]
    if len(vals)==0: 
        return 0.0
    return float(np.mean(vals))

def analyze_video(path, EAR_THRESH=0.22, MIN_FRAMES=2, INCOMPLETE_CUTOFF=0.18,
                  FRAME_STEP=1, MAX_SECONDS=None, PROGRESS_EVERY=300):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = 0
    blinks = 0
    incomplete = 0
    low = 0
    min_ear = 1.0
    last_blink_frame = None
    ibis = []
    reds = []

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            h, w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                e = (ear(lm,w,h,True) + ear(lm,w,h,False))/2.0
                if e < EAR_THRESH:
                    low += 1
                    min_ear = min(min_ear, e)
                else:
                    if low >= MIN_FRAMES:
                        blinks += 1
                        if min_ear > INCOMPLETE_CUTOFF:
                            incomplete += 1
                        if last_blink_frame is not None:
                            ibis.append((frames - last_blink_frame)/fps)
                        last_blink_frame = frames
                    low = 0
                    min_ear = 1.0

                try:
                    reds.append(redness_index(frame, lm))
                except Exception:
                    pass

            if MAX_SECONDS is not None and frames / fps >= MAX_SECONDS:
                break

            for _ in range(max(1, FRAME_STEP)-1):
                cap.grab()
                frames += 1

    cap.release()

    dur = frames / fps if fps else 0.0
    bpm = (blinks/dur)*60.0 if dur>0 else 0.0
    inc_ratio = (incomplete/blinks) if blinks>0 else 0.0
    avg_ibi = float(np.mean(ibis)) if ibis else 0.0
    avg_red = float(np.mean(reds)) if reds else 0.0

    return dict(
        duration_sec=round(dur,2),
        blinks=int(blinks),
        blink_rate_bpm=round(bpm,2),
        incomplete_blink_ratio=round(inc_ratio,3),
        avg_ibi_sec=round(avg_ibi,2),
        redness_index=round(avg_red,3)
    )

# ---- the function the dashboard calls ----
def extract_from_video(video_path: str) -> dict:
    """Return the 4 features needed by the model from a video file path."""
    result = analyze_video(video_path)
    # keep only the 4 model features (duration/blinks are extra)
    return {
        "blink_rate_bpm": result["blink_rate_bpm"],
        "incomplete_blink_ratio": result["incomplete_blink_ratio"],
        "avg_ibi_sec": result["avg_ibi_sec"],
        "redness_index": result["redness_index"],
    }

# Optional CLI batch mode (not used by the dashboard)
def pid_from(name):
    m = re.match(r"([^_]+)_", name)
    return m.group(1) if m else "UNKNOWN"

def main():
    EAR_THRESH        = 0.22
    MIN_FRAMES        = 2
    INCOMPLETE_CUTOFF = 0.18
    FRAME_STEP        = 1
    MAX_SECONDS       = None

    args = sys.argv[1:]
    videos = [Path(a) for a in args] if args else sorted(Path(".").glob("*.mp4"))
    if not videos:
        print("No .mp4 files found. Put your video in this folder or pass a filename.")
        sys.exit(1)

    out = Path("features.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id","filename","duration_sec","blinks","blink_rate_bpm",
                    "incomplete_blink_ratio","avg_ibi_sec","redness_index"])
        for v in videos:
            try:
                m = analyze_video(
                    v,
                    EAR_THRESH=EAR_THRESH,
                    MIN_FRAMES=MIN_FRAMES,
                    INCOMPLETE_CUTOFF=INCOMPLETE_CUTOFF,
                    FRAME_STEP=FRAME_STEP,
                    MAX_SECONDS=MAX_SECONDS,
                    PROGRESS_EVERY=300
                )
                w.writerow([pid_from(v.name), v.name, m["duration_sec"], m["blinks"],
                            m["blink_rate_bpm"], m["incomplete_blink_ratio"],
                            m["avg_ibi_sec"], m["redness_index"]])
            except Exception as e:
                print(f"[ERROR] {v.name}: {e}")
    print(f"[OK] Saved {out.resolve()}")

if __name__ == "__main__":
    main()
