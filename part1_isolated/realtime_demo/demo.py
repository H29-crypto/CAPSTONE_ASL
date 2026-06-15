"""
demo.py — Phase-Aware ASL Sign Recognition Webcam Demo
-------------------------------------------------------
Usage:  python realtime_demo/demo.py
Keys:   R = record 2 seconds and predict
        Q = quit

Requires: torch, mediapipe, opencv-python, numpy
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe not installed.  Run:  pip install mediapipe")

from pipeline import (
    N_FRAMES, TARGET_FPS, N_POSE, N_HAND, POSE_IDXS,
    PhaseTCN, RecognitionTCNAttentionSafe,
    robust_normalize_keypoints, compute_motion_features, compute_phase_speed,
    majority_filter_1d,
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

# UI colours (BGR)
C_GREEN  = (0, 220, 0)
C_RED    = (0, 0, 220)
C_ORANGE = (0, 165, 255)
C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)
C_DARK   = (30, 30, 30)

PHASE_NAMES  = {0: "Background", 1: "Preparation", 2: "Stroke", 3: "Retraction"}
PHASE_COLORS = {
    0: (100, 100, 100),
    1: (0, 200, 220),
    2: (0, 220, 0),
    3: (220, 80, 0),
}


# ─────────────────────────────────────────────────────────────
# LIVE PHASE PREDICTION  (runs during RECORD state)
# ─────────────────────────────────────────────────────────────

def predict_current_phase(fi, pose_buf, lh_buf, rh_buf,
                           pose_msk, lh_msk, rh_msk,
                           phase_model, phase_mean, phase_std, device):
    """Run PhaseTCN on pose_buf[0:fi]. Returns (last_frame_phase, per_frame_labels)."""
    if fi < 4:
        return 0, np.zeros(fi, dtype=np.int64)

    pts, pos = robust_normalize_keypoints(
        pose_buf[:fi], lh_buf[:fi], rh_buf[:fi],
        pose_msk[:fi], lh_msk[:fi], rh_msk[:fi],
    )
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([
        pos, vel, acc,
        gs.reshape(-1, 1), hs.reshape(-1, 1), psp.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0

    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0

    with torch.no_grad():
        xt     = torch.tensor(Xn).unsqueeze(0).to(device)
        probs4 = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()

    labels = majority_filter_1d(np.argmax(probs4, axis=-1).astype(np.int64))
    return int(labels[-1]), labels


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def draw_landmarks_on_frame(frame, pose_res, hand_res):
    h, w = frame.shape[:2]
    if pose_res.pose_landmarks:
        for idx in POSE_IDXS:
            lm = pose_res.pose_landmarks[0][idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 200, 0), -1)
    if hand_res.hand_landmarks:
        for hand_lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            color = (255, 120, 0) if handed[0].category_name == "Left" else (0, 120, 255)
            for lm in hand_lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)


def draw_top5(frame, top5_idx, probs, id_to_gloss):
    h, w    = frame.shape[:2]
    panel_w = 380
    panel_h = 225
    x0, y0  = w - panel_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), C_GREEN, 2)
    cv2.putText(frame, "TOP-5 PREDICTIONS",
                (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

    for rank, idx in enumerate(top5_idx):
        gloss   = id_to_gloss.get(int(idx), f"ID_{idx}")
        conf    = float(probs[idx])
        yy      = y0 + 50 + rank * 34
        color   = C_WHITE if rank == 0 else C_GRAY
        bar_max = panel_w - 140
        bar_len = int(conf * bar_max)

        cv2.rectangle(frame, (x0 + 130, yy - 14), (x0 + 130 + bar_max, yy + 4), (60, 60, 60), -1)
        if bar_len > 0:
            cv2.rectangle(frame, (x0 + 130, yy - 14), (x0 + 130 + bar_len, yy + 4),
                          C_GREEN if rank == 0 else (60, 160, 60), -1)
        cv2.putText(frame, f"#{rank + 1} {gloss[:16]}",
                    (x0 + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"{conf * 100:.1f}%",
                    (x0 + panel_w - 65, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)


def draw_bottom_bar(frame, msg, color=C_WHITE):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), C_DARK, -1)
    cv2.putText(frame, msg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_record_bar(frame, fi, n_frames, phase_history=None, current_phase=0):
    h, w = frame.shape[:2]

    strip_y0, strip_y1 = h - 72, h - 56
    sx0, sx1 = 10, w - 10
    strip_w  = sx1 - sx0
    cv2.rectangle(frame, (sx0, strip_y0), (sx1, strip_y1), (40, 40, 40), -1)
    if phase_history:
        for i, ph in enumerate(phase_history):
            x0 = sx0 + int(i       * strip_w / n_frames)
            x1 = sx0 + int((i + 1) * strip_w / n_frames)
            cv2.rectangle(frame, (x0, strip_y0), (x1, strip_y1),
                          PHASE_COLORS.get(int(ph), (100, 100, 100)), -1)

    bar_w = int((fi / n_frames) * (w - 20))
    cv2.rectangle(frame, (10, h - 52), (w - 10, h - 42), (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, h - 52), (10 + bar_w, h - 42), C_RED, -1)

    ph_color = PHASE_COLORS.get(current_phase, C_GRAY)
    ph_name  = PHASE_NAMES.get(current_phase, "?")
    cv2.putText(frame, f"REC  {fi}/{n_frames}",
                (10, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2)
    cv2.putText(frame, ph_name,
                (w - 180, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_color, 2)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    for p in [PHASE_CKPT_PATH, REC_CKPT_PATH, LABEL_MAP_PATH]:
        if not p.exists():
            sys.exit(f"Missing: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    id_to_gloss = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_gloss[int(row["rec_label_id"])] = row["gloss"]
    print(f"Labels: {len(id_to_gloss)} classes")

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

    print("Loading Recognition TCN ...")
    rc    = load_checkpoint(REC_CKPT_PATH, device)
    rmean = to_numpy(rc["feature_mean"])
    rstd  = to_numpy(rc["feature_std"])
    rec_model = RecognitionTCNAttentionSafe(
        input_dim=rc["config"]["input_dim"], num_classes=rc["config"]["num_classes"],
        hidden_dim=rc["config"]["hidden_dim"], num_blocks=rc["config"]["num_blocks"],
        kernel_size=rc["config"]["kernel_size"], dropout=rc["config"]["dropout"],
        attention_dropout=rc["config"]["attention_dropout"])
    rec_model.load_state_dict(rc["model_state_dict"])
    rec_model.to(device).eval()
    print(f"  val_top1 = {rc.get('val_top1', 0):.4f}  val_top5 = {rc.get('val_top5', 0):.4f}")

    print("Initialising MediaPipe landmarkers ...")
    hand_lmk, pose_lmk = make_landmarkers(HAND_MODEL_PATH, POSE_MODEL_PATH)
    print("  Landmarkers ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hand_lmk.close(); pose_lmk.close()
        sys.exit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 50)
    print("  ASL Demo ready.  R = record  |  Q = quit")
    print("=" * 50)

    IDLE = 0; COUNTDOWN = 1; RECORD = 2; INFER = 3

    state         = IDLE
    cd_t          = 0.0
    fi            = 0
    top5_idx      = None
    probs100      = None
    seg_info      = ""
    phase_history = []
    current_phase = 0

    pose_buf = np.zeros((N_FRAMES, N_POSE, 3), np.float32)
    lh_buf   = np.zeros((N_FRAMES, N_HAND, 3), np.float32)
    rh_buf   = np.zeros((N_FRAMES, N_HAND, 3), np.float32)
    pose_msk = np.zeros(N_FRAMES, dtype=bool)
    lh_msk   = np.zeros(N_FRAMES, dtype=bool)
    rh_msk   = np.zeros(N_FRAMES, dtype=bool)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_res = hand_lmk.detect(mp_img)
        pose_res = pose_lmk.detect(mp_img)
        draw_landmarks_on_frame(frame, pose_res, hand_res)

        if state == IDLE:
            if top5_idx is not None:
                draw_top5(frame, top5_idx, probs100, id_to_gloss)
                cv2.putText(frame, seg_info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)
            draw_bottom_bar(frame, "R = record sign  |  Q = quit", C_GREEN)

        elif state == COUNTDOWN:
            remaining = 3 - int(time.time() - cd_t)
            if remaining <= 0:
                state = RECORD
                fi    = 0
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, str(remaining),
                            (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, C_ORANGE, 8)
                draw_bottom_bar(frame, "Get ready ...", C_ORANGE)

        elif state == RECORD:
            pa, la, ra, po, lo, ro = extract_frame_landmarks(rgb, hand_lmk, pose_lmk)
            if fi < N_FRAMES:
                pose_buf[fi] = pa; lh_buf[fi] = la; rh_buf[fi] = ra
                pose_msk[fi] = po; lh_msk[fi] = lo; rh_msk[fi] = ro
                fi += 1
                if fi >= 8 and fi % 5 == 0:
                    current_phase, frame_phases = predict_current_phase(
                        fi, pose_buf, lh_buf, rh_buf,
                        pose_msk, lh_msk, rh_msk,
                        phase_model, pmean, pstd, device,
                    )
                    phase_history = list(frame_phases)
            draw_record_bar(frame, fi, N_FRAMES, phase_history, current_phase)
            draw_bottom_bar(frame, "Recording ... hold your sign!", C_RED)
            if fi >= N_FRAMES:
                state = INFER

        elif state == INFER:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Processing ...",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ORANGE, 3)
            draw_bottom_bar(frame, "Running pipeline ...", C_ORANGE)
            cv2.imshow("ASL Demo", frame)
            cv2.waitKey(1)

            try:
                probs100, seg_status, as_, ae = run_pipeline(
                    pose_buf, lh_buf, rh_buf,
                    pose_msk, lh_msk, rh_msk,
                    phase_model, pmean, pstd,
                    rec_model,   rmean, rstd,
                    device,
                )
                top5_idx = np.argsort(probs100)[::-1][:5]
                seg_info = f"phase_seg={seg_status}  active=[{as_},{ae}]"
                print(f"\n  TOP-5:")
                for rk, ix in enumerate(top5_idx, 1):
                    g = id_to_gloss.get(int(ix), f"ID_{ix}")
                    print(f"  #{rk}  {g:<25}  {probs100[ix] * 100:.1f}%")
            except Exception as e:
                print(f"  [ERROR] {e}")
                top5_idx = None
                seg_info = f"error: {e}"

            state = IDLE
            continue

        cv2.imshow("ASL Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r") and state == IDLE:
            pose_buf[:] = 0; lh_buf[:] = 0; rh_buf[:] = 0
            pose_msk[:] = False; lh_msk[:] = False; rh_msk[:] = False
            phase_history = []
            current_phase = 0
            state = COUNTDOWN
            cd_t  = time.time()

    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
