"""
demo_webcam_continuous.py — Real-time Continuous Sign Language Recognition
---------------------------------------------------------------------------
Uses the BiLSTM + CTC model on live webcam input to produce German Sign
Language (DGS) gloss predictions from the PHOENIX-2014-T vocabulary.

How it works:
    1. Press R → 3-second countdown → 4-second recording window
    2. Captured frames are passed through ResNet-18 (stride 2) → 512-dim features
    3. BiLSTM CTC decodes features → gloss sequence
    4. Predicted glosses are appended to the on-screen sentence

Keys:
    R = start a new recording
    C = clear the sentence
    Q = quit

Note:
    The model was trained on professional studio footage (PHOENIX-2014-T).
    Predictions on webcam input may differ from the benchmark WER numbers,
    but the pipeline is identical to the offline demo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Allow running directly: python demos/demo_webcam_continuous.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import SignLanguageRecognizer

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH    = PROJECT_ROOT / "weights" / "ctc_best_v2.pt"

# ── Recording constants ───────────────────────────────────────────────────────
RECORD_SECONDS  = 4.0
TARGET_FPS      = 25
N_FRAMES        = int(RECORD_SECONDS * TARGET_FPS)   # 100 frames → 50 after stride 2

# ── UI colours (BGR) ─────────────────────────────────────────────────────────
C_GREEN  = (0, 220, 0)
C_RED    = (0, 0, 220)
C_ORANGE = (0, 165, 255)
C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)
C_DARK   = (30, 30, 30)
C_YELLOW = (0, 220, 220)

MAX_SENTENCE_WORDS = 12


# ── UI helpers ────────────────────────────────────────────────────────────────

def draw_sentence_panel(frame: np.ndarray, sentence: list[str]) -> None:
    """Semi-transparent top panel showing the accumulated gloss sentence."""
    h, w    = frame.shape[:2]
    panel_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    if not sentence:
        cv2.putText(frame, "Press R to start signing ...",
                    (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)
    else:
        words = sentence[-MAX_SENTENCE_WORDS:]
        x     = 10
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        for i, word in enumerate(words):
            color = C_GREEN if i == len(words) - 1 else C_WHITE
            cv2.putText(frame, word, (x, 47), font, scale, color, thick)
            (tw, _), _ = cv2.getTextSize(word + " ", font, scale, thick)
            x += tw
            if x > w - 80:
                break

    cv2.line(frame, (0, panel_h), (w, panel_h), C_GRAY, 1)


def draw_bottom_bar(frame: np.ndarray, message: str, color: tuple = C_WHITE) -> None:
    """Status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), C_DARK, -1)
    cv2.putText(frame, message, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)


def draw_record_bar(frame: np.ndarray, frame_index: int, total_frames: int) -> None:
    """Red progress bar shown while recording."""
    h, w = frame.shape[:2]
    bar_w = int((frame_index / total_frames) * (w - 20))
    cv2.rectangle(frame, (10, h - 52), (w - 10, h - 42), (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, h - 52), (10 + bar_w, h - 42), C_RED, -1)

    elapsed = frame_index / TARGET_FPS
    cv2.putText(frame, f"REC  {elapsed:.1f}s / {RECORD_SECONDS:.0f}s",
                (10, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2)


def draw_last_prediction(frame: np.ndarray, gloss_text: str,
                          pred_time: float) -> None:
    """Fade-out overlay showing the most recent prediction."""
    if not gloss_text or (time.time() - pred_time) >= 3.0:
        return
    h, w  = frame.shape[:2]
    alpha = max(0.0, 1.0 - (time.time() - pred_time) / 3.0)
    fade  = tuple(int(c * alpha) for c in C_YELLOW)
    cv2.putText(frame, f">> {gloss_text}",
                (w // 2 - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, fade, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CKPT_PATH.exists():
        sys.exit(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Make sure ctc_best_v2.pt is in sign_language_demo/weights/"
        )

    # ── Load model ────────────────────────────────────────────────────────────
    recognizer = SignLanguageRecognizer(str(CKPT_PATH), device="auto")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 54)
    print("  German SL Continuous Demo  (live webcam)")
    print("  R = record   C = clear sentence   Q = quit")
    print("=" * 54 + "\n")

    # ── State machine ─────────────────────────────────────────────────────────
    IDLE      = "idle"
    COUNTDOWN = "countdown"
    RECORD    = "record"
    INFER     = "infer"

    state         = IDLE
    countdown_t   = 0.0
    frame_buf:  list[np.ndarray] = []     # RGB frames captured during RECORD
    sentence:   list[str]        = []     # accumulated gloss words
    last_pred:  str              = ""
    last_pred_t: float           = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── INFER — show processing frame, run model, then go back to IDLE ───
        if state == INFER:
            h_f, w_f = frame.shape[:2]
            cv2.putText(frame, "Processing ...",
                        (w_f // 2 - 120, h_f // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ORANGE, 3)
            draw_sentence_panel(frame, sentence)
            draw_bottom_bar(frame, "  Running BiLSTM CTC ...", C_ORANGE)
            cv2.imshow("German SL Continuous", frame)
            cv2.waitKey(1)

            try:
                predicted = recognizer.predict_from_frames(
                    frame_buf, frame_stride=2
                )
                gloss_text = " ".join(predicted) if predicted else "<no prediction>"
                print(f"  Predicted: {gloss_text}")

                if predicted:
                    sentence.extend(predicted)
                    last_pred   = gloss_text
                    last_pred_t = time.time()
                else:
                    last_pred   = "<no prediction>"
                    last_pred_t = time.time()

            except Exception as exc:
                print(f"  [ERROR] {exc}")

            frame_buf.clear()
            state = IDLE
            continue

        # ── IDLE ──────────────────────────────────────────────────────────────
        if state == IDLE:
            draw_sentence_panel(frame, sentence)
            draw_bottom_bar(
                frame,
                "  R = record sign   C = clear   Q = quit",
                C_GREEN,
            )
            draw_last_prediction(frame, last_pred, last_pred_t)

        # ── COUNTDOWN ─────────────────────────────────────────────────────────
        elif state == COUNTDOWN:
            remaining = 3 - int(time.time() - countdown_t)
            if remaining <= 0:
                state     = RECORD
                frame_buf = []
            else:
                h_f, w_f = frame.shape[:2]
                cv2.putText(frame, str(remaining),
                            (w_f // 2 - 30, h_f // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, C_ORANGE, 8)
                draw_sentence_panel(frame, sentence)
                draw_bottom_bar(frame, "  Get ready ...", C_ORANGE)

        # ── RECORD ────────────────────────────────────────────────────────────
        elif state == RECORD:
            if len(frame_buf) < N_FRAMES:
                frame_buf.append(rgb.copy())
            draw_record_bar(frame, len(frame_buf), N_FRAMES)
            draw_sentence_panel(frame, sentence)
            draw_bottom_bar(frame, "  Recording ... sign freely!", C_RED)
            if len(frame_buf) >= N_FRAMES:
                state = INFER

        cv2.imshow("German SL Continuous", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r") and state == IDLE:
            state       = COUNTDOWN
            countdown_t = time.time()
            print("  Starting countdown ...")
        if key == ord("c"):
            sentence.clear()
            last_pred = ""
            print("  Sentence cleared.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
