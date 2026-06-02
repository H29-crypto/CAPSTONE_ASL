"""
colab_demo.py — CorrNet German Sign Language Demo for Google Colab (T4 GPU)
============================================================================
Run this in a Colab notebook cell:

    !python corrnet/colab_demo.py

Or paste each SECTION into separate cells.

Setup (run once):
    !pip install -q gradio opencv-python-headless torch torchvision

Checkpoint (upload corrnet_phoenix2014T.pt to Google Drive, then):
    from google.colab import drive
    drive.mount('/content/drive')
    # Set CKPT_PATH below to your Drive path, e.g.
    # /content/drive/MyDrive/corrnet_phoenix2014T.pt
"""
from __future__ import annotations

# ── SECTION 1: Setup ──────────────────────────────────────────────────────────
import sys
import os
import time
import tempfile
from pathlib import Path

# Mock ctcdecode (Linux wheel is not available on Colab's default env)
try:
    import ctcdecode
except Exception:
    import unittest.mock
    sys.modules["ctcdecode"] = unittest.mock.MagicMock()

import cv2
import numpy as np
import torch

# ── SECTION 2: Paths ──────────────────────────────────────────────────────────
# When running inside the cloned repo from Colab:
#   /content/CAPSTONE_ASL_LOCAL/corrnet/colab_demo.py
# Adjust CKPT_PATH if your checkpoint is on Google Drive.

_HERE        = Path(__file__).resolve().parent          # corrnet/
_ROOT        = _HERE.parent                             # project root
CKPT_PATH    = _HERE / "weights" / "corrnet_phoenix2014T.pt"
DICT_PATH    = _HERE / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
GLOSS2TEXT   = _HERE / "gloss_to_text.py"

# Override checkpoint path here if on Drive:
# CKPT_PATH = Path("/content/drive/MyDrive/corrnet_phoenix2014T.pt")

sys.path.insert(0, str(_HERE))

# Load gloss_to_text without polluting sys.path
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gloss_to_text", GLOSS2TEXT)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
glosses_to_german = _mod.glosses_to_german
del _ilu, _spec, _mod

# ── SECTION 3: Model ──────────────────────────────────────────────────────────
SUBSAMPLE_STEP = 3   # keep every 3rd frame → ~33 frames from a 4s clip
TARGET_FPS     = 25

def _device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  GPU: {name}")
        return torch.device("cuda")
    print("  No GPU found — running on CPU (slower)")
    return torch.device("cpu")


def load_corrnet(ckpt_path: Path, gloss_dict: dict, device: torch.device):
    from collections import OrderedDict
    from corrnet_webcam import load_model
    model = load_model(ckpt_path, gloss_dict)
    model = model.to(device)
    try:
        model = torch.compile(model, backend="eager")
        print("  torch.compile enabled")
    except Exception:
        pass
    model.eval()
    return model


def extract_frames(video_path: str, max_frames: int = 100) -> list[np.ndarray]:
    """Read an MP4/WebM from disk → list of RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@torch.no_grad()
def run_inference(model, frames: list, device: torch.device):
    from corrnet_webcam import pad_and_make_batch, run_inference as _infer
    if not frames:
        return [], []
    sampled = frames[::SUBSAMPLE_STEP]
    return _infer(model, sampled, device)


# ── SECTION 4: Gradio interface ───────────────────────────────────────────────

def build_interface(model, gloss_dict: dict, device: torch.device):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run:  pip install gradio")
        return

    def predict(video_path):
        if video_path is None:
            return "No video received.", "", "—"

        t0 = time.time()
        frames = extract_frames(video_path)
        if len(frames) < 5:
            return "Clip too short — sign for at least 1 second.", "", "—"

        glosses, confs = run_inference(model, frames, device)
        elapsed = time.time() - t0

        if not glosses:
            return "No sign detected — try again.", "", f"{elapsed:.1f}s"

        german  = glosses_to_german(glosses, add_period=True) or " ".join(glosses)
        raw     = "  ".join(
            f"{g} ({c*100:.0f}%)" for g, c in zip(glosses, confs)
        )
        timing  = (f"{elapsed:.1f}s  |  {len(frames)} frames → "
                   f"{len(frames[::SUBSAMPLE_STEP])} sampled  |  "
                   f"device: {device.type.upper()}")
        return german, raw, timing

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Video(
            sources=["webcam", "upload"],
            label="Record or upload a signing clip (2–5 s)",
        ),
        outputs=[
            gr.Textbox(label="German translation", lines=2),
            gr.Textbox(label="Raw glosses + confidence", lines=2),
            gr.Textbox(label="Timing", lines=1),
        ],
        title="German Sign Language — CorrNet (Phoenix-2014-T)",
        description=(
            "**How to use:**\n"
            "1. Click *Record from webcam* and sign a short DGS weather phrase (2–5 s)\n"
            "2. Click *Submit* — CorrNet runs on the GPU and translates to German\n\n"
            "**Good phrases to try:** `HEUTE REGEN KOMMEN`  "
            "`MORGEN NORD WIND STARK`  `MINUS DREI GRAD`\n\n"
            "**Tip:** Stand in the portrait crop zone, upper body visible, plain background."
        ),
        examples=None,
        allow_flagging="never",
    )

    # share=True gives a public URL valid for 72 h (needed in Colab)
    demo.launch(share=True, server_port=7860)


# ── SECTION 5: Main ───────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  CorrNet German SL Demo — Colab Edition")
    print("="*60)

    for label, path in [("Checkpoint", CKPT_PATH), ("Gloss dict", DICT_PATH)]:
        if not path.exists():
            print(f"\n[ERROR] {label} not found: {path}")
            print("  Upload corrnet_phoenix2014T.pt to Google Drive and set CKPT_PATH.")
            return

    device     = _device()
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"  Vocabulary : {len(gloss_dict)} Phoenix glosses")

    print("  Loading CorrNet (ResNet18) …")
    model = load_corrnet(CKPT_PATH, gloss_dict, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model ready : {params:,} params")

    print("\n  Launching Gradio interface …")
    build_interface(model, gloss_dict, device)


if __name__ == "__main__":
    main()
