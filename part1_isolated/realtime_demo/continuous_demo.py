"""
continuous_demo.py — Continuous ASL Sign Translation Demo
----------------------------------------------------------
Signs are detected and recognized automatically as you sign continuously.
The Phase TCN watches the live landmark stream and identifies sign boundaries;
the Recognition TCN then classifies each completed sign.

Keys:
  C = clear the accumulated sentence
  Q = quit

How it works:
  1. Phase TCN runs on a rolling window every few frames (IDLE mode).
  2. When a non-background phase is detected, the system enters SIGNING mode
     and accumulates landmark frames in a dedicated sign buffer.
  3. After BG_COOLDOWN_FRAMES consecutive background frames (sign finished),
     the buffered frames are fed to the full run_pipeline() for recognition.
  4. The predicted gloss is appended to the on-screen sentence.
"""

import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import mediapipe as mp          # noqa: F401  (needed by pipeline internals)
except ImportError:
    sys.exit("mediapipe not installed.  Run:  pip install mediapipe")

# Allow running directly: python realtime_demo/continuous_demo.py
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    N_HAND, N_POSE, TARGET_FPS,
    PhaseTCN, RecognitionTCNAttentionSafe,
    compute_motion_features, compute_phase_speed,
    majority_filter_1d,
    robust_normalize_keypoints,
    run_pipeline, make_landmarkers, extract_frame_landmarks,
    load_checkpoint, to_numpy,
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PHASE_CKPT_PATH = BASE_DIR / "weights" / "phase_tcn_best_safe_state.pt"
REC_CKPT_PATH   = BASE_DIR / "weights" / "recognition_tcn_attention_best.pt"
LABEL_MAP_PATH  = BASE_DIR / "data"    / "recognition_label_map_with_split_counts.csv"
HAND_MODEL_PATH = BASE_DIR / "assets"  / "hand_landmarker.task"
POSE_MODEL_PATH = BASE_DIR / "assets"  / "pose_landmarker_full.task"

# ─────────────────────────────────────────────────────────────
# TUNABLE CONSTANTS
# ─────────────────────────────────────────────────────────────
RING_SIZE           = int(5 * TARGET_FPS)   # 150 frames — rolling ring buffer
PHASE_INFER_EVERY   = 5                     # run Phase TCN every N frames
MIN_SIGN_FRAMES     = 20                    # reject signs shorter than this
BG_COOLDOWN_FRAMES  = 18                    # background frames that close a sign
CONF_THRESHOLD      = 0.20                  # min top-1 confidence to accept
MAX_SIGN_FRAMES     = RING_SIZE             # force recognition after this many frames
LOOKBACK_FRAMES     = 10                    # ring frames to seed sign_buf with at start
MAX_SENTENCE_WORDS  = 12                    # number of words kept visible on screen
PHASE_WINDOW        = 90                    # max frames used for continuous phase inference

# ─────────────────────────────────────────────────────────────
# UI COLOURS (BGR)
# ─────────────────────────────────────────────────────────────
C_GREEN  = (0, 220, 0)
C_RED    = (0, 0, 220)
C_ORANGE = (0, 165, 255)
C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)
C_DARK   = (30, 30, 30)
C_YELLOW = (0, 220, 220)

PHASE_NAMES  = {0: "Background", 1: "Preparation", 2: "Stroke", 3: "Retraction"}
PHASE_COLORS = {
    0: (100, 100, 100),
    1: (0, 200, 220),
    2: (0, 220, 0),
    3: (220, 80, 0),
}

# ─────────────────────────────────────────────────────────────
# RING BUFFER  (always-on rolling window for IDLE phase watch)
# ─────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-size circular buffer for landmark frames."""

    def __init__(self, size: int):
        self.size     = size
        self.pose_buf = np.zeros((size, N_POSE, 3), np.float32)
        self.lh_buf   = np.zeros((size, N_HAND, 3), np.float32)
        self.rh_buf   = np.zeros((size, N_HAND, 3), np.float32)
        self.pose_msk = np.zeros(size, dtype=bool)
        self.lh_msk   = np.zeros(size, dtype=bool)
        self.rh_msk   = np.zeros(size, dtype=bool)
        self.ptr      = 0
        self.count    = 0

    def push(self, pose, lh, rh, p_ok, l_ok, r_ok):
        i = self.ptr % self.size
        self.pose_buf[i] = pose
        self.lh_buf  [i] = lh
        self.rh_buf  [i] = rh
        self.pose_msk[i] = p_ok
        self.lh_msk  [i] = l_ok
        self.rh_msk  [i] = r_ok
        self.ptr   += 1
        self.count  = min(self.count + 1, self.size)

    def get_last_n(self, n: int):
        """Return last n frames in chronological order (oldest → newest)."""
        n = min(n, self.count)
        if n == 0:
            return (np.zeros((0, N_POSE, 3), np.float32),
                    np.zeros((0, N_HAND, 3), np.float32),
                    np.zeros((0, N_HAND, 3), np.float32),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=bool))
        end     = self.ptr % self.size
        indices = [(end - n + i) % self.size for i in range(n)]
        return (self.pose_buf[indices], self.lh_buf[indices], self.rh_buf[indices],
                self.pose_msk[indices], self.lh_msk[indices], self.rh_msk[indices])


# ─────────────────────────────────────────────────────────────
# SIGN BUFFER  (growable; accumulates one sign while SIGNING)
# ─────────────────────────────────────────────────────────────

class SignBuffer:
    """Growable buffer that accumulates frames of a single sign."""

    def __init__(self):
        self.pose_list: list = []
        self.lh_list:   list = []
        self.rh_list:   list = []
        self.pm_list:   list = []
        self.lm_list:   list = []
        self.rm_list:   list = []

    def __len__(self) -> int:
        return len(self.pose_list)

    def push(self, pose, lh, rh, p_ok, l_ok, r_ok):
        self.pose_list.append(pose.copy())
        self.lh_list  .append(lh.copy())
        self.rh_list  .append(rh.copy())
        self.pm_list  .append(bool(p_ok))
        self.lm_list  .append(bool(l_ok))
        self.rm_list  .append(bool(r_ok))

    def get_arrays(self, start: int = 0, end: int = None):
        """Return numpy arrays for frames [start:end]."""
        if end is None or end > len(self):
            end = len(self)
        if start >= end or end == 0:
            return (np.zeros((0, N_POSE, 3), np.float32),
                    np.zeros((0, N_HAND, 3), np.float32),
                    np.zeros((0, N_HAND, 3), np.float32),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=bool))
        return (np.stack(self.pose_list[start:end]),
                np.stack(self.lh_list  [start:end]),
                np.stack(self.rh_list  [start:end]),
                np.array(self.pm_list  [start:end], dtype=bool),
                np.array(self.lm_list  [start:end], dtype=bool),
                np.array(self.rm_list  [start:end], dtype=bool))

    def init_from_ring(self, ring: RingBuffer, n: int):
        """Seed the buffer from the last n frames of the ring (no current frame)."""
        pose, lh, rh, pm, lm, rm = ring.get_last_n(n)
        for i in range(len(pose)):
            self.push(pose[i], lh[i], rh[i], pm[i], lm[i], rm[i])

    def clear(self):
        self.pose_list.clear(); self.lh_list.clear(); self.rh_list.clear()
        self.pm_list.clear();   self.lm_list.clear(); self.rm_list.clear()


# ─────────────────────────────────────────────────────────────
# PHASE INFERENCE  (on arbitrary numpy arrays)
# ─────────────────────────────────────────────────────────────

def infer_phase_on_arrays(pose, lh, rh, pm, lm, rm,
                           phase_model, phase_mean, phase_std, device):
    """
    Run Phase TCN on the given landmark arrays.
    Returns (last_frame_phase int, per_frame_labels ndarray).
    """
    T = pose.shape[0]
    if T < 4:
        return 0, np.zeros(T, dtype=np.int64)

    pts, pos = robust_normalize_keypoints(pose, lh, rh, pm, lm, rm)
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([pos, vel, acc,
                        gs.reshape(-1, 1), hs.reshape(-1, 1), psp.reshape(-1, 1)],
                       axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0

    with torch.no_grad():
        xt     = torch.tensor(Xn).unsqueeze(0).to(device)
        probs4 = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()

    labels = majority_filter_1d(np.argmax(probs4, axis=-1).astype(np.int64))
    return int(labels[-1]), labels


# ─────────────────────────────────────────────────────────────
# LANDMARK DRAWING  (from numpy arrays — avoids a second detect call)
# ─────────────────────────────────────────────────────────────

def draw_landmarks_numpy(frame, pose_arr, lh_arr, rh_arr, p_ok, l_ok, r_ok):
    h, w = frame.shape[:2]
    if p_ok:
        for pt in pose_arr:
            x, y = int(pt[0] * w), int(pt[1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)
    if l_ok:
        for pt in lh_arr:
            x, y = int(pt[0] * w), int(pt[1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 3, (255, 120, 0), -1)
    if r_ok:
        for pt in rh_arr:
            x, y = int(pt[0] * w), int(pt[1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 3, (0, 120, 255), -1)


# ─────────────────────────────────────────────────────────────
# UI DRAWING
# ─────────────────────────────────────────────────────────────

def draw_sentence_panel(frame, sentence: list):
    """Semi-transparent top panel showing the accumulated gloss sentence."""
    h, w = frame.shape[:2]
    panel_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if not sentence:
        cv2.putText(frame, "Start signing ...",
                    (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)
    else:
        words = sentence[-MAX_SENTENCE_WORDS:]
        x     = 10
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.85
        thick = 2
        for i, word in enumerate(words):
            color = C_GREEN if i == len(words) - 1 else C_WHITE
            cv2.putText(frame, word, (x, 47), font, scale, color, thick)
            (tw, _), _ = cv2.getTextSize(word + " ", font, scale, thick)
            x += tw
            if x > w - 80:          # stop before overflow
                break

    cv2.line(frame, (0, panel_h), (w, panel_h), C_GRAY, 1)


def draw_phase_strip(frame, phase_history: list, current_phase: int):
    """Colour-coded phase history strip above the status bar."""
    h, w = frame.shape[:2]

    strip_y0, strip_y1 = h - 72, h - 52
    sx0, sx1 = 10, w - 10
    strip_w  = sx1 - sx0
    cv2.rectangle(frame, (sx0, strip_y0), (sx1, strip_y1), (40, 40, 40), -1)

    n = len(phase_history)
    if n > 0:
        for i, ph in enumerate(phase_history):
            x0 = sx0 + int(i       * strip_w / n)
            x1 = sx0 + int((i + 1) * strip_w / n)
            cv2.rectangle(frame, (x0, strip_y0), (x1, strip_y1),
                          PHASE_COLORS.get(int(ph), (100, 100, 100)), -1)

    ph_color = PHASE_COLORS.get(current_phase, C_GRAY)
    ph_name  = PHASE_NAMES.get(current_phase, "?")
    cv2.putText(frame, ph_name,
                (sx0, strip_y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_color, 2)


def draw_sign_indicator(frame, state: str, sign_len: int, max_len: int):
    """Pulsing dot + mini fill-bar shown while the system is collecting a sign."""
    if state != "sign":
        return
    h, w = frame.shape[:2]
    pulse = int(127 + 127 * np.sin(time.time() * 6))
    cv2.circle(frame, (w - 30, 95), 10, (0, pulse, pulse), -1)
    cv2.putText(frame, "SIGNING", (w - 120, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_RED, 2)
    bar_w = 100
    x0    = w - 120
    cv2.rectangle(frame, (x0, 108), (x0 + bar_w, 118), (60, 60, 60), -1)
    fill = int(min(sign_len / max(max_len, 1), 1.0) * bar_w)
    if fill > 0:
        cv2.rectangle(frame, (x0, 108), (x0 + fill, 118), C_RED, -1)


def draw_status_bar(frame, state: str, last_gloss: str,
                    last_conf: float, last_pred_t: float):
    """Bottom bar: current state message + fading last-prediction overlay."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), C_DARK, -1)

    if state == "sign":
        msg, color = "  SIGNING ...", C_RED
    elif state == "infer":
        msg, color = "  Recognizing ...", C_ORANGE
    else:
        msg, color = "  Idle -- sign freely   C = clear   Q = quit", C_GREEN

    cv2.putText(frame, msg, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    # Fade-out last prediction (shown for 3 seconds)
    if last_gloss and (time.time() - last_pred_t) < 3.0:
        alpha = max(0.0, 1.0 - (time.time() - last_pred_t) / 3.0)
        fade  = tuple(int(c * alpha) for c in C_YELLOW)
        txt   = f">> {last_gloss}  ({last_conf * 100:.0f}%)"
        cv2.putText(frame, txt, (w // 2 - 90, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, fade, 2)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── Sanity checks
    for p in [PHASE_CKPT_PATH, REC_CKPT_PATH, LABEL_MAP_PATH]:
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Label map
    id_to_gloss: dict = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_gloss[int(row["rec_label_id"])] = row["gloss"]
    print(f"Labels: {len(id_to_gloss)} classes")

    # ── Phase TCN
    print("Loading Phase TCN ...")
    pc    = load_checkpoint(PHASE_CKPT_PATH, device)
    pmean = to_numpy(pc["feature_mean"])
    pstd  = to_numpy(pc["feature_std"])
    phase_model = PhaseTCN(**{k: pc["config"][k] for k in
                              ["input_dim", "num_classes", "hidden_dim",
                               "num_blocks", "kernel_size", "dropout"]})
    phase_model.load_state_dict(pc["model_state_dict"])
    phase_model.to(device).eval()
    print(f"  val_accuracy = {pc.get('val_accuracy', 0):.4f}")

    # ── Recognition TCN
    print("Loading Recognition TCN ...")
    rc    = load_checkpoint(REC_CKPT_PATH, device)
    rmean = to_numpy(rc["feature_mean"])
    rstd  = to_numpy(rc["feature_std"])
    rec_model = RecognitionTCNAttentionSafe(
        input_dim        =rc["config"]["input_dim"],
        num_classes      =rc["config"]["num_classes"],
        hidden_dim       =rc["config"]["hidden_dim"],
        num_blocks       =rc["config"]["num_blocks"],
        kernel_size      =rc["config"]["kernel_size"],
        dropout          =rc["config"]["dropout"],
        attention_dropout=rc["config"]["attention_dropout"],
    )
    rec_model.load_state_dict(rc["model_state_dict"])
    rec_model.to(device).eval()
    print(f"  val_top1 = {rc.get('val_top1', 0):.4f}  "
          f"val_top5 = {rc.get('val_top5', 0):.4f}")

    # ── MediaPipe
    print("Initialising MediaPipe landmarkers ...")
    hand_lmk, pose_lmk = make_landmarkers(HAND_MODEL_PATH, POSE_MODEL_PATH)
    print("  Landmarkers ready.")

    # ── Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hand_lmk.close(); pose_lmk.close()
        sys.exit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 52)
    print("  Continuous ASL Translation ready.")
    print("  Sign freely.   C = clear sentence   Q = quit")
    print("=" * 52 + "\n")

    # ── State constants
    IDLE  = "idle"
    SIGN  = "sign"
    INFER = "infer"

    # ── Runtime state
    state          = IDLE
    frame_idx      = 0
    current_phase  = 0
    bg_count       = 0
    last_active_fi = 0          # index in sign_buf of last active (non-BG) frame

    ring     = RingBuffer(RING_SIZE)
    sign_buf = SignBuffer()

    sentence:    list  = []
    last_gloss:  str   = ""
    last_conf:   float = 0.0
    last_pred_t: float = 0.0

    phase_strip: deque = deque(maxlen=PHASE_WINDOW)   # for the phase-colour strip

    # ──────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose, lh, rh, p_ok, l_ok, r_ok = extract_frame_landmarks(
            rgb, hand_lmk, pose_lmk)
        draw_landmarks_numpy(frame, pose, lh, rh, p_ok, l_ok, r_ok)
        frame_idx += 1

        # ──────── INFER STATE ────────────────────────────────
        # Show a "Recognizing" frame, run the full pipeline,
        # then reset and return to IDLE.
        if state == INFER:
            h_f, w_f = frame.shape[:2]
            cv2.putText(frame, "Recognizing ...",
                        (w_f // 2 - 140, h_f // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ORANGE, 3)
            draw_sentence_panel(frame, sentence)
            draw_phase_strip(frame, list(phase_strip), current_phase)
            draw_status_bar(frame, INFER, last_gloss, last_conf, last_pred_t)
            cv2.imshow("ASL Continuous", frame)
            cv2.waitKey(1)

            # Trim to last active frame (drop trailing BG cooldown frames)
            end_fi = last_active_fi + 1
            if end_fi >= MIN_SIGN_FRAMES:
                ps_, lh_, rh_, pm_, lm_, rm_ = sign_buf.get_arrays(end=end_fi)
                try:
                    probs100, seg_status, _, _ = run_pipeline(
                        ps_, lh_, rh_, pm_, lm_, rm_,
                        phase_model, pmean, pstd,
                        rec_model,   rmean, rstd,
                        device,
                    )
                    top1_idx  = int(np.argmax(probs100))
                    top1_conf = float(probs100[top1_idx])
                    gloss     = id_to_gloss.get(top1_idx, f"ID_{top1_idx}")
                    print(f"  [{seg_status}]  {gloss:<25}  {top1_conf * 100:.1f}%")

                    if top1_conf >= CONF_THRESHOLD:
                        # Deduplicate: skip if same word as the last one
                        if not sentence or sentence[-1] != gloss:
                            sentence.append(gloss)
                        last_gloss  = gloss
                        last_conf   = top1_conf
                        last_pred_t = time.time()
                    else:
                        print(f"  --> Rejected  "
                              f"(conf {top1_conf:.2f} < threshold {CONF_THRESHOLD})")
                except Exception as exc:
                    print(f"  [PIPELINE ERROR] {exc}")
            else:
                print(f"  --> Skipped  "
                      f"(sign_len={end_fi} < MIN_SIGN_FRAMES={MIN_SIGN_FRAMES})")

            # Reset for the next sign
            sign_buf.clear()
            bg_count = last_active_fi = 0
            state    = IDLE
            ring.push(pose, lh, rh, p_ok, l_ok, r_ok)   # don't lose this frame
            continue

        # ──────── PHASE INFERENCE (every PHASE_INFER_EVERY frames) ───
        if frame_idx % PHASE_INFER_EVERY == 0:
            if state == SIGN and len(sign_buf) >= 4:
                # Run on the most recent PHASE_WINDOW frames of the sign buffer
                n_use  = min(PHASE_WINDOW, len(sign_buf))
                start  = len(sign_buf) - n_use
                ps_, lh_, rh_, pm_, lm_, rm_ = sign_buf.get_arrays(start=start)
                current_phase, ph_labels = infer_phase_on_arrays(
                    ps_, lh_, rh_, pm_, lm_, rm_,
                    phase_model, pmean, pstd, device,
                )
                phase_strip.extend(ph_labels)

            elif state == IDLE and ring.count >= 4:
                # Run on the most recent 30 frames of the ring buffer
                n_use = min(30, ring.count)
                ps_, lh_, rh_, pm_, lm_, rm_ = ring.get_last_n(n_use)
                current_phase, ph_labels = infer_phase_on_arrays(
                    ps_, lh_, rh_, pm_, lm_, rm_,
                    phase_model, pmean, pstd, device,
                )
                phase_strip.extend(ph_labels)

        # ──────── IDLE STATE ─────────────────────────────────
        if state == IDLE:
            if current_phase != 0:
                # Sign is starting — seed sign_buf with prep frames from ring
                # (ring does NOT yet contain the current frame → no duplicate)
                sign_buf.init_from_ring(ring, LOOKBACK_FRAMES)
                sign_buf.push(pose, lh, rh, p_ok, l_ok, r_ok)
                last_active_fi = len(sign_buf) - 1
                bg_count       = 0
                state          = SIGN
            ring.push(pose, lh, rh, p_ok, l_ok, r_ok)   # always keep ring fresh

        # ──────── SIGN STATE ─────────────────────────────────
        elif state == SIGN:
            sign_buf.push(pose, lh, rh, p_ok, l_ok, r_ok)
            ring.push(pose, lh, rh, p_ok, l_ok, r_ok)

            if current_phase != 0:
                last_active_fi = len(sign_buf) - 1
                bg_count       = 0
            else:
                bg_count += 1

            # Trigger recognition when sign ends or buffer is full
            if bg_count >= BG_COOLDOWN_FRAMES or len(sign_buf) >= MAX_SIGN_FRAMES:
                state = INFER

        # ──────── DRAW UI ─────────────────────────────────────
        draw_sentence_panel(frame, sentence)
        draw_phase_strip(frame, list(phase_strip), current_phase)
        draw_sign_indicator(frame, state, len(sign_buf), MAX_SIGN_FRAMES)
        draw_status_bar(frame, state, last_gloss, last_conf, last_pred_t)

        cv2.imshow("ASL Continuous", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c"):
            sentence.clear()
            print("  Sentence cleared.")

    # ── Cleanup
    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    cv2.destroyAllWindows()
    print("Continuous demo closed.")


if __name__ == "__main__":
    main()
