"""
verify_pipeline.py — Offline checkpoint verification for Phase-Aware ASL Recognition.
--------------------------------------------------------------------------------------
Usage:  python realtime_demo/verify_pipeline.py

Verifies the full pipeline end-to-end without the interactive UI:
  1. Load both checkpoints and confirm weight shapes.
  2. Open webcam, countdown, record 2 seconds.
  3. Run MediaPipe → normalization → motion → Phase TCN → Recognition TCN.
  4. Print top-5 predictions.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline import (
    N_FRAMES, TARGET_FPS, N_POSE, N_HAND,
    PhaseTCN, RecognitionTCNAttentionSafe,
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
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    for p in [PHASE_CKPT_PATH, REC_CKPT_PATH, LABEL_MAP_PATH]:
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Label map
    id_to_gloss = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_gloss[int(row["rec_label_id"])] = row["gloss"]
    print(f"Label map: {len(id_to_gloss)} classes")

    # Phase TCN
    print("\nLoading Phase TCN checkpoint...")
    phase_ckpt  = load_checkpoint(PHASE_CKPT_PATH, device)
    pcfg        = phase_ckpt["config"]
    phase_mean  = to_numpy(phase_ckpt["feature_mean"])
    phase_std   = to_numpy(phase_ckpt["feature_std"])
    phase_model = PhaseTCN(**{k: pcfg[k] for k in
                              ["input_dim", "num_classes", "hidden_dim",
                               "num_blocks", "kernel_size", "dropout"]})
    phase_model.load_state_dict(phase_ckpt["model_state_dict"])
    phase_model.to(device).eval()

    w = phase_ckpt["model_state_dict"]["tcn.0.conv1.weight"]
    assert w.shape == (pcfg["hidden_dim"], pcfg["input_dim"], pcfg["kernel_size"]), \
        f"Phase TCN weight shape mismatch: {w.shape}"
    print(f"  input={pcfg['input_dim']}  hidden={pcfg['hidden_dim']}  "
          f"blocks={pcfg['num_blocks']} — OK")

    # Recognition TCN
    print("\nLoading Recognition TCN checkpoint...")
    rec_ckpt  = load_checkpoint(REC_CKPT_PATH, device)
    rcfg      = rec_ckpt["config"]
    rec_mean  = to_numpy(rec_ckpt["feature_mean"])
    rec_std   = to_numpy(rec_ckpt["feature_std"])
    rec_model = RecognitionTCNAttentionSafe(
        input_dim=rcfg["input_dim"],   num_classes=rcfg["num_classes"],
        hidden_dim=rcfg["hidden_dim"], num_blocks=rcfg["num_blocks"],
        kernel_size=rcfg["kernel_size"], dropout=rcfg["dropout"],
        attention_dropout=rcfg["attention_dropout"])
    rec_model.load_state_dict(rec_ckpt["model_state_dict"])
    rec_model.to(device).eval()

    w2 = rec_ckpt["model_state_dict"]["tcn.0.conv1.weight"]
    assert w2.shape == (rcfg["hidden_dim"], rcfg["input_dim"], rcfg["kernel_size"]), \
        f"Rec TCN weight shape mismatch: {w2.shape}"
    print(f"  input={rcfg['input_dim']}  hidden={rcfg['hidden_dim']}  "
          f"blocks={rcfg['num_blocks']}  classes={rcfg['num_classes']} — OK")

    # MediaPipe
    print("\nInitialising MediaPipe landmarkers...")
    hand_lmk, pose_lmk = make_landmarkers(HAND_MODEL_PATH, POSE_MODEL_PATH)
    print("  Landmarkers ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 60)
    print("VERIFICATION: press R to record 2 seconds, Q to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame.copy(), "Press R to record  |  Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ASL Verification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("r"):
            # Countdown
            for cnt in [3, 2, 1]:
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    ret2, frm = cap.read()
                    if not ret2:
                        break
                    cv2.putText(frm, f"Recording in {cnt}...",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("ASL Verification", frm)
                    cv2.waitKey(1)

            # Record
            pose_buf = np.zeros((N_FRAMES, N_POSE, 3), dtype=np.float32)
            lh_buf   = np.zeros((N_FRAMES, N_HAND, 3), dtype=np.float32)
            rh_buf   = np.zeros((N_FRAMES, N_HAND, 3), dtype=np.float32)
            pose_msk = np.zeros(N_FRAMES, dtype=bool)
            lh_msk   = np.zeros(N_FRAMES, dtype=bool)
            rh_msk   = np.zeros(N_FRAMES, dtype=bool)

            print(f"\nRecording {N_FRAMES} frames...")
            t_start = time.time()
            for fi in range(N_FRAMES):
                ret3, frm = cap.read()
                if not ret3:
                    break
                rgb  = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                pa, la, ra, po, lo, ro = extract_frame_landmarks(rgb, hand_lmk, pose_lmk)
                pose_buf[fi] = pa; lh_buf[fi] = la; rh_buf[fi] = ra
                pose_msk[fi] = po; lh_msk[fi] = lo; rh_msk[fi] = ro

                bar = int(20 * fi / N_FRAMES)
                cv2.putText(frm,
                            f"REC [{time.time()-t_start:.1f}s] " + "#" * bar + " " * (20 - bar),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("ASL Verification", frm)
                cv2.waitKey(1)

            print(f"  Recorded in {time.time()-t_start:.2f}s")
            print(f"  Pose: {pose_msk.sum()}/{N_FRAMES}  "
                  f"Left hand: {lh_msk.sum()}/{N_FRAMES}  "
                  f"Right hand: {rh_msk.sum()}/{N_FRAMES}")

            # Run pipeline
            print("\nRunning pipeline...")
            probs, seg_status, active_start, active_end = run_pipeline(
                pose_buf, lh_buf, rh_buf, pose_msk, lh_msk, rh_msk,
                phase_model, phase_mean, phase_std,
                rec_model,   rec_mean,   rec_std,
                device,
            )
            print(f"  Phase seg: {seg_status}  active=[{active_start},{active_end}]  "
                  f"T_active={active_end - active_start + 1}")

            # Top-5
            top5_idx = np.argsort(probs)[::-1][:5]
            print("\n  TOP-5 PREDICTIONS:")
            print("  " + "-" * 35)
            for rank, idx in enumerate(top5_idx, 1):
                gloss = id_to_gloss.get(idx, f"ID_{idx}")
                print(f"  #{rank}  {gloss:<25}  {probs[idx]*100:.1f}%")
            print("  " + "-" * 35)

            # Overlay on frame
            ret4, frm = cap.read()
            if ret4:
                overlay = frm.copy()
                cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frm, 0.5, 0, frm)
                for rank, idx in enumerate(top5_idx, 1):
                    gloss = id_to_gloss.get(idx, f"ID_{idx}")
                    cv2.putText(frm, f"#{rank} {gloss}  {probs[idx]*100:.1f}%",
                                (10, 30 + (rank - 1) * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frm, "Press R again or Q to quit",
                            (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("ASL Verification", frm)
                cv2.waitKey(3000)

    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    cv2.destroyAllWindows()
    print("\nVerification session ended.")


if __name__ == "__main__":
    main()
