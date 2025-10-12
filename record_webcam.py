# record_webcam.py
import cv2, time
from pathlib import Path
from datetime import datetime

CAM_INDEX = 0
FPS = 20.0
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # macOS-friendly
DURATION_SEC = 15 * 60  # 15 minutes

def record_session(participant_id: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"{participant_id}_{ts}.mp4")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check Camera permissions or CAM_INDEX.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer = cv2.VideoWriter(str(out_path), FOURCC, FPS, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter failed. Try XVID + .avi if needed.")

    print(f"[INFO] Recording → {out_path.resolve()} (auto-stops at 15 min)")
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        writer.write(frame)
        cv2.imshow("Recording (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if time.time() - start >= DURATION_SEC: break

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    pid = input("Enter Participant ID (e.g., P01): ").strip() or "PXX"
    record_session(pid)
