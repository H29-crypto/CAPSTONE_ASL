"""
demo_phoenix_video.py — Run recognition on a single PHOENIX video folder.

Usage:
    python demos/demo_phoenix_video.py --video_folder sample_videos/27November_2009_Friday_tagesschau-7342
    python demos/demo_phoenix_video.py --video_folder sample_videos/27November_2009_Friday_tagesschau-7342 --no_reference
    python demos/demo_phoenix_video.py --video_folder sample_videos/27November_2009_Friday_tagesschau-7342 --checkpoint weights/ctc_best_v2.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Pure-Python edit distance — no C compiler required
def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]

# Make src/ importable when running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import SignLanguageRecognizer

# ── Default paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_CKPT    = PROJECT_ROOT / "weights" / "ctc_best_v2.pt"
TEST_LABELS_CSV = PROJECT_ROOT / "sample_videos" / "test_labels.csv"

DIVIDER = "=" * 53


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_reference_glosses(csv_path: Path, video_name: str) -> list[str] | None:
    """
    Look up ground-truth glosses for a video name from test_labels.csv.

    Args:
        csv_path:   Path to the CSV file.
        video_name: Folder basename to match against the 'name' column.

    Returns:
        List of gloss strings, or None if not found / file missing.
    """
    if not csv_path.exists():
        return None

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        sample_text = csv_file.read(512)
        csv_file.seek(0)
        delimiter = "|" if "|" in sample_text else ","
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        for row in reader:
            if row.get("name", "").strip() == video_name:
                raw_orth = row.get("orth", "").strip()
                return raw_orth.split() if raw_orth else []

    return None


def compute_wer(hypothesis: list[str], reference: list[str]) -> float:
    """
    Word Error Rate = editdistance(hyp, ref) / len(ref).

    Returns 0.0 when both are empty; inf when reference is empty but hyp is not.
    """
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return _edit_distance(hypothesis, reference) / len(reference)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Sign Language Recognition — single video demo"
    )
    parser.add_argument(
        "--video_folder",
        required=True,
        help="Path to a folder containing imagesXXXX.png frames",
    )
    parser.add_argument(
        "--no_reference",
        action="store_true",
        help="Skip ground-truth lookup (no reference available)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help=f"Path to model checkpoint (default: {DEFAULT_CKPT.name})",
    )
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    video_name   = video_folder.name

    # ── Load model ────────────────────────────────────────────────────────────
    recognizer = SignLanguageRecognizer(args.checkpoint, device="auto")

    # ── Count frames ─────────────────────────────────────────────────────────
    png_files          = sorted(video_folder.glob("*.png"))
    num_frames         = len(png_files)
    num_features       = len(range(0, num_frames, 2))   # stride 2

    # ── Run prediction ────────────────────────────────────────────────────────
    predicted_glosses  = recognizer.predict_from_phoenix_folder(str(video_folder))
    prediction_display = " ".join(predicted_glosses) if predicted_glosses else "<no prediction>"

    # ── Look up reference ─────────────────────────────────────────────────────
    reference_glosses: list[str] | None = None
    if not args.no_reference:
        reference_glosses = load_reference_glosses(TEST_LABELS_CSV, video_name)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  Sign Language Recognition Demo")
    print(DIVIDER)
    print(f"  Model:      {Path(args.checkpoint).name}")
    print(f"  Device:     {recognizer.device_str}")
    print(f"  Vocab size: {recognizer.vocab_size}")
    print(f"  Dev WER:    {recognizer.dev_wer:.2%}")
    print()
    print(f"  Video:      {video_name}")
    print(f"  Frames:     {num_frames}")
    print(f"  Features extracted: {num_features} (after stride 2)")
    print()
    print(f"  PREDICTION:  {prediction_display}")

    if reference_glosses is not None:
        reference_display = " ".join(reference_glosses) if reference_glosses else "<empty>"
        wer        = compute_wer(predicted_glosses, reference_glosses)
        num_errors = _edit_distance(predicted_glosses, reference_glosses)
        print(f"  REFERENCE:   {reference_display}")
        print()
        print(
            f"  Word Error Rate: {wer:.2f} "
            f"({num_errors} error{'s' if num_errors != 1 else ''} "
            f"out of {len(reference_glosses)} word{'s' if len(reference_glosses) != 1 else ''})"
        )
    elif not args.no_reference:
        print(f"  (Reference not found in {TEST_LABELS_CSV.name} for '{video_name}')")

    print(DIVIDER)


if __name__ == "__main__":
    main()
