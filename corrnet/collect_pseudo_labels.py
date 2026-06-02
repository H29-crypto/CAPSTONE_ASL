"""
collect_pseudo_labels.py — Guided webcam recording for MSLR training data
--------------------------------------------------------------------------
Records you signing Phoenix weather glosses, extracts MediaPipe landmarks,
and uses CorrNet as teacher to generate soft KD targets.

No Phoenix video download needed — this creates your own personalised
training set adapted to YOUR signing style.

Usage:
    python corrnet/collect_pseudo_labels.py

Controls (while recording prompt is visible):
    SPACE — start recording clip
    S     — skip this gloss/clip
    Q     — quit and save progress

Output:
    corrnet/data/pseudo_labels/{GLOSS}_{n:03d}.npz
    Each file contains:
        features        (T, 444) float32  — MediaPipe motion features
        teacher_logits  (T', C)  float32  — CorrNet soft labels
        hard_label      int64             — gloss class ID
        gloss           str               — gloss name
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    import ctcdecode
except Exception:
    import unittest.mock
    sys.modules["ctcdecode"] = unittest.mock.MagicMock()

import cv2
import numpy as np
import torch

BASE_DIR    = Path(__file__).resolve().parent.parent
CORRNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRNET_DIR))
sys.path.insert(0, str(BASE_DIR))

from corrnet_webcam import load_model, pad_and_make_batch
from realtime_demo.pipeline import (
    make_landmarkers, extract_frame_landmarks,
    robust_normalize_keypoints, compute_motion_features, compute_phase_speed,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_PATH = CORRNET_DIR / "weights" / "corrnet_phoenix2014T.pt"
DICT_PATH = CORRNET_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
HAND_PATH = BASE_DIR / "assets" / "hand_landmarker.task"
POSE_PATH = BASE_DIR / "assets" / "pose_landmarker_full.task"
OUT_DIR   = CORRNET_DIR / "data" / "pseudo_labels"

# ── Recording params ──────────────────────────────────────────────────────────
TARGET_FPS      = 25
RECORD_SECS     = 4.0
N_FRAMES        = int(RECORD_SECS * TARGET_FPS)   # 100 frames
CLIPS_PER_GLOSS = 5

# ── Gloss lists ───────────────────────────────────────────────────────────────
GLOSSES_WEATHER = [
    "REGEN", "WIND", "SONNE", "WOLKE", "SCHNEE",
    "HEUTE", "MORGEN", "WARM", "KALT", "STARK",
    "SCHWACH", "NORD", "SUED", "OST", "WEST",
    "KOMMEN", "BLEIBEN", "TEMPERATUR", "TROCKEN", "NEBEL",
]

# Numbers present in the Phoenix vocabulary (exclude SECHZEHN — not in vocab)
GLOSSES_NUMBERS = [
    "NULL", "EIN", "ZWEI", "DREI", "VIER", "FUENF",
    "SECHS", "SIEBEN", "ACHT", "NEUN", "ZEHN",
    "ELF", "ZWOELF", "DREIZEHN", "VIERZEHN", "FUENFZEHN",
    "SIEBZEHN", "ACHTZEHN", "NEUNZEHN", "ZWANZIG",
    "DREISSIG", "MINUS", "PLUS", "GRAD",
]

# ── Mode (set by --mode argument) ─────────────────────────────────────────────
# Resolved in main() after argparse
GLOSSES = GLOSSES_WEATHER   # default; overridden by --mode

# ── UI ─────────────────────────────────────────────────────────────────────────
C_GREEN = (0,210,0); C_RED = (30,30,220); C_ORG = (0,140,255)
C_WHITE = (255,255,255); C_GRAY = (140,140,140)
C_DARK  = (25,25,25);    C_TEAL = (180,210,0)


def extract_features(frames_rgb: list[np.ndarray],
                     hand_lmk, pose_lmk) -> np.ndarray:
    """Run MediaPipe on frames → compute 444-dim motion features."""
    pose_l, lh_l, rh_l = [], [], []
    pm_l,   lm_l, rm_l = [], [], []
    for f in frames_rgb:
        pose, lh, rh, pm, lm, rm = extract_frame_landmarks(f, hand_lmk, pose_lmk)
        pose_l.append(pose); lh_l.append(lh); rh_l.append(rh)
        pm_l.append(pm);     lm_l.append(lm); rm_l.append(rm)

    pose_a = np.stack(pose_l); lh_a = np.stack(lh_l); rh_a = np.stack(rh_l)
    pm_a   = np.array(pm_l, bool); lm_a = np.array(lm_l, bool); rm_a = np.array(rm_l, bool)

    pts, pos = robust_normalize_keypoints(pose_a, lh_a, rh_a, pm_a, lm_a, rm_a)
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([
        pos, vel, acc,
        gs.reshape(-1, 1), hs.reshape(-1, 1), psp.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return X  # (T, 444)


@torch.no_grad()
def teacher_logits(corrnet, frames_rgb: list, device) -> tuple[np.ndarray, int]:
    """Run CorrNet on frames → soft label logits (T', C)."""
    vid, lgt = pad_and_make_batch(frames_rgb)
    vid = vid.to(device); lgt = lgt.to(device)
    out = corrnet(vid, lgt)
    logits   = out["sequence_logits"]          # (T', B=1, C)
    feat_len = int(out["feat_len"][0].item())
    return logits[:feat_len, 0, :].cpu().numpy(), feat_len  # (T', C)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weather", "numbers", "all"],
                    default="weather",
                    help="Which gloss set to record: weather (default), numbers, or all")
    ap.add_argument("--clips", type=int, default=CLIPS_PER_GLOSS,
                    help="Clips to record per gloss (default 5)")
    args = ap.parse_args()

    global GLOSSES, CLIPS_PER_GLOSS
    CLIPS_PER_GLOSS = args.clips
    if args.mode == "numbers":
        GLOSSES = GLOSSES_NUMBERS
    elif args.mode == "all":
        GLOSSES = GLOSSES_WEATHER + GLOSSES_NUMBERS
    else:
        GLOSSES = GLOSSES_WEATHER

    print(f"\n  Mode: {args.mode}  ({len(GLOSSES)} glosses x {CLIPS_PER_GLOSS} clips)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    device = torch.device("cpu")

    print("\n  Loading CorrNet teacher ...")
    corrnet = load_model(CKPT_PATH, gloss_dict)
    corrnet.to(device).eval()

    print("  Starting MediaPipe ...")
    hand_lmk, pose_lmk = make_landmarkers(str(HAND_PATH), str(POSE_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    total = sum(1 for g in GLOSSES for n in range(CLIPS_PER_GLOSS)
                if (OUT_DIR / f"{g}_{n:03d}.npz").exists())
    print(f"\n  Data dir: {OUT_DIR}")
    print(f"  Already collected: {total} / {len(GLOSSES)*CLIPS_PER_GLOSS} clips")
    print("  SPACE=record  S=skip  Q=quit\n")

    saved = 0

    for gloss in GLOSSES:
        hard_label = gloss_dict.get(gloss, [0])[0]

        for clip_n in range(CLIPS_PER_GLOSS):
            out_path = OUT_DIR / f"{gloss}_{clip_n:03d}.npz"
            if out_path.exists():
                continue

            # ── Prompt screen ──────────────────────────────────────────────
            recording = False
            skip      = False
            while not recording:
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]

                ov = frame.copy()
                cv2.rectangle(ov, (0,0), (w,100), C_DARK, -1)
                cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

                cv2.putText(frame, f"SIGN:  {gloss}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, C_TEAL, 3)
                cv2.putText(frame,
                            f"Clip {clip_n+1}/{CLIPS_PER_GLOSS}   "
                            f"Saved today: {saved}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_GRAY, 1)
                cv2.putText(frame, "SPACE=record   S=skip gloss   Q=quit",
                            (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_GREEN, 2)

                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = True
                elif key in (ord('s'), ord('S')):
                    skip = True; break
                elif key in (ord('q'), ord('Q')):
                    cap.release(); hand_lmk.close()
                    pose_lmk.close(); cv2.destroyAllWindows()
                    print(f"\n  Quit. Clips saved this session: {saved}")
                    return

            if skip:
                print(f"  [{gloss}] skipped")
                break  # skip remaining clips for this gloss

            # ── Countdown ─────────────────────────────────────────────────
            for cnt in (3, 2, 1):
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    ret, frame = cap.read()
                    if not ret: break
                    h, w = frame.shape[:2]
                    cv2.putText(frame, str(cnt),
                                (w//2-40, h//2+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0, C_ORG, 10)
                    cv2.putText(frame, f"Sign: {gloss}",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TEAL, 2)
                    cv2.imshow("Data Collection", frame)
                    cv2.waitKey(1)

            # ── Record ─────────────────────────────────────────────────────
            print(f"  [{gloss} {clip_n+1}/{CLIPS_PER_GLOSS}] Recording ...")
            rgb_frames = []
            t0 = time.time()

            while len(rgb_frames) < N_FRAMES:
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb.copy())

                h, w   = frame.shape[:2]
                pct    = len(rgb_frames) / N_FRAMES
                bar_w  = int(pct * (w - 20))
                cv2.rectangle(frame, (10,h-8), (10+(w-20),h-3), (50,50,50), -1)
                if bar_w > 0:
                    cv2.rectangle(frame, (10,h-8), (10+bar_w,h-3), C_RED, -1)
                cv2.putText(frame,
                            f"REC  {time.time()-t0:.1f}s / {RECORD_SECS:.0f}s",
                            (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2)
                cv2.putText(frame, f"Sign: {gloss}",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_TEAL, 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            # ── Process & save ─────────────────────────────────────────────
            print(f"    Extracting landmarks ...")
            feats = extract_features(rgb_frames, hand_lmk, pose_lmk)

            print(f"    Running CorrNet teacher ...")
            soft_logits, feat_len = teacher_logits(corrnet, rgb_frames, device)

            np.savez(str(out_path),
                     features       = feats,
                     teacher_logits = soft_logits,
                     hard_label     = np.array(hard_label, dtype=np.int64),
                     gloss          = gloss,
                     feat_len       = np.array(feat_len, dtype=np.int64))
            saved += 1
            print(f"    Saved {out_path.name}  "
                  f"features={feats.shape}  logits={soft_logits.shape}")

    cap.release(); hand_lmk.close(); pose_lmk.close()
    cv2.destroyAllWindows()
    print(f"\n  Done! Clips saved this session: {saved}")
    print(f"  Total in {OUT_DIR}: "
          f"{len(list(OUT_DIR.glob('*.npz')))} clips")
    print(f"\n  Next: python corrnet/train_mslr.py")


if __name__ == "__main__":
    main()
