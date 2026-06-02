"""
eval_samples.py — Evaluate AdaptSign on the PHOENIX-2014-T sample videos
-------------------------------------------------------------------------
Runs AdaptSign (ViT-B/16 + BiLSTM CTC) on every video in
  sign_language_demo/sample_videos/
compares predictions against the ground-truth labels in test_labels.csv,
and reports per-video WER alongside an aggregate WER.

Usage:
    python adaptsign/eval_samples.py
    python adaptsign/eval_samples.py --max 50        # quick test on 50 videos
    python adaptsign/eval_samples.py --verbose        # show every prediction
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import OrderedDict
from pathlib import Path

# ── ctcdecode is Linux-only — mock before any adaptsign import ─────────────────
try:
    import ctcdecode  # noqa: F401
except Exception:
    import types, unittest.mock
    sys.modules["ctcdecode"] = unittest.mock.MagicMock()

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
ADAPTSIGN_DIR  = Path(__file__).resolve().parent
CKPT_PATH      = ADAPTSIGN_DIR / "weights" / "phoenix2014-T_best.pt"
DICT_PATH      = ADAPTSIGN_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
SAMPLES_DIR    = PROJECT_ROOT / "sign_language_demo" / "sample_videos"
LABELS_CSV     = SAMPLES_DIR / "test_labels.csv"

sys.path.insert(0, str(ADAPTSIGN_DIR))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gloss_to_text",
    PROJECT_ROOT / "corrnet" / "gloss_to_text.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
glosses_to_german = _mod.glosses_to_german
del _ilu, _spec, _mod


# ══════════════════════════════════════════════════════════════════════════════
# Greedy CTC decoder
# ══════════════════════════════════════════════════════════════════════════════

class GreedyDecoder:
    def __init__(self, gloss_dict: dict):
        self.i2g     = {v[0]: k for k, v in gloss_dict.items()}
        self.blank_id = 0

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        results = []
        for b in range(nn_output.shape[0]):
            length = int(vid_lgt[b].item())
            logits = nn_output[b, :length]
            indices = logits.argmax(dim=-1).cpu().tolist()
            seq, prev = [], None
            for idx in indices:
                if idx != prev:
                    prev = idx
                    if idx != self.blank_id:
                        seq.append((self.i2g.get(idx, f"<{idx}>"), len(seq)))
            results.append(seq)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(gloss_dict: dict, device: torch.device):
    from slr_network import SLRModel
    num_classes = len(gloss_dict) + 1
    model = SLRModel(
        num_classes      = num_classes,
        c2d_type         = "ViT-B/16",
        conv_type        = 2,
        use_bn           = True,
        hidden_size      = 1024,
        gloss_dict       = gloss_dict,
        loss_weights     = {"SeqCTC": 1.0},
        weight_norm      = True,
        share_classifier = True,
    )
    model.decoder = GreedyDecoder(gloss_dict)

    print(f"  Loading checkpoint …")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd   = OrderedDict([(k.replace(".module", ""), v) for k, v in sd.items()])
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    print("  Model ready.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Per-video inference
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Match AdaptSign training preprocessing exactly:
      exact 256×256 resize (non-aspect-ratio), center-crop 224×224, then /127.5 - 1 → [-1,1].
    Phoenix frames are 210×260; AdaptSign's offline preprocessing squashes them to 256×256
    before training, so we must replicate that non-aspect-ratio resize here.
    """
    img  = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    top  = (256 - 224) // 2   # = 16
    left = (256 - 224) // 2   # = 16
    img  = img[top:top+224, left:left+224]                  # (224,224,3) BGR
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # → RGB
    t    = torch.from_numpy(img).float().permute(2, 0, 1)   # (3,224,224), [0,255]
    return t / 127.5 - 1.0                                  # → [-1,1]


def load_frames(folder: Path) -> list[np.ndarray]:
    """Load all PNG frames from a Phoenix video folder (BGR numpy arrays), sorted."""
    frames = []
    for p in sorted(folder.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


@torch.no_grad()
def run_inference(model, frames: list[np.ndarray], device: torch.device) -> list[str]:
    if not frames:
        return []
    tensors = [_preprocess_frame(f) for f in frames]
    imgs    = torch.stack(tensors).unsqueeze(0).to(device)    # (1, T, 3, 224, 224)
    lengths = torch.tensor([len(frames)], dtype=torch.long).to(device)
    out     = model(imgs, lengths)
    sents   = out.get("recognized_sents", [[]])
    return [g for g, _ in sents[0]] if sents and sents[0] else []


# ══════════════════════════════════════════════════════════════════════════════
# WER
# ══════════════════════════════════════════════════════════════════════════════

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


def compute_wer(hyp: list[str], ref: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else float("inf")
    return _edit_distance(hyp, ref) / len(ref)


# ══════════════════════════════════════════════════════════════════════════════
# Load ground-truth labels
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(csv_path: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name  = row.get("name", "").strip()
            orth  = row.get("orth", "").strip()
            if name:
                labels[name] = orth.split() if orth else []
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max",     type=int, default=None,
                        help="Limit evaluation to first N videos")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-video predictions and references")
    args = parser.parse_args()

    # ── pre-flight ────────────────────────────────────────────────────────────
    for p, label in [(CKPT_PATH, "Checkpoint"), (DICT_PATH, "Gloss dict"), (LABELS_CSV, "Labels CSV")]:
        if not p.exists():
            sys.exit(f"[ERROR] {label} not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device      : {device}")
    print(f"  Checkpoint  : {CKPT_PATH.name}")

    print("  Loading gloss dictionary …")
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"  Vocabulary  : {len(gloss_dict)} glosses")

    print("  Loading AdaptSign model (CLIP ViT-B/16 auto-downloads ~340 MB on first run) …")
    model = load_model(gloss_dict, device)

    labels = load_labels(LABELS_CSV)

    # Collect video folders that exist on disk
    video_folders = sorted(
        [d for d in SAMPLES_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )
    if args.max:
        video_folders = video_folders[: args.max]

    DIVIDER = "=" * 65
    print(f"\n{DIVIDER}")
    print(f"  AdaptSign Evaluation on PHOENIX-2014-T sample videos")
    print(f"  Videos to evaluate : {len(video_folders)}")
    print(DIVIDER)

    total_errors = 0
    total_ref    = 0
    skipped      = 0
    t0           = time.time()

    for i, folder in enumerate(video_folders):
        name = folder.name
        ref  = labels.get(name)
        if ref is None:
            skipped += 1
            continue

        frames = load_frames(folder)
        if not frames:
            skipped += 1
            continue

        hyp  = run_inference(model, frames, device)
        errs = _edit_distance(hyp, ref)
        wer  = errs / len(ref) if ref else 0.0

        total_errors += errs
        total_ref    += len(ref)

        if args.verbose or (i % 50 == 0):
            elapsed = time.time() - t0
            german  = glosses_to_german(hyp) if hyp else ""
            print(f"  [{i+1:4d}/{len(video_folders)}]  {name[:45]:<45}")
            print(f"          REF : {' '.join(ref) if ref else '<empty>'}")
            print(f"          HYP : {' '.join(hyp) if hyp else '<no prediction>'}")
            if german:
                print(f"          DE  : {german}")
            print(f"          WER : {wer:.2%}  ({errs}/{len(ref)} errors)  "
                  f"[elapsed {elapsed:.0f}s]")
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    evaluated     = len(video_folders) - skipped
    agg_wer       = total_errors / total_ref if total_ref else float("inf")

    print(DIVIDER)
    print(f"  RESULTS (AdaptSign  |  ViT-B/16  |  PHOENIX-2014-T)")
    print(DIVIDER)
    print(f"  Videos evaluated   : {evaluated}")
    print(f"  Skipped            : {skipped}")
    print(f"  Total word errors  : {total_errors}")
    print(f"  Total ref words    : {total_ref}")
    print(f"  Aggregate WER      : {agg_wer:.2%}")
    print()
    print(f"  Paper benchmark    : Dev 18.6%  |  Test 18.9%  (full set, beam search)")
    print(f"  Note: greedy decode + sample subset; beam search would reduce WER slightly")
    print(f"  Elapsed            : {elapsed_total:.0f}s  ({elapsed_total/max(evaluated,1):.1f}s/video)")
    print(DIVIDER)


if __name__ == "__main__":
    main()
