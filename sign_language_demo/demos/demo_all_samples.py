"""
demo_all_samples.py — Run recognition on all sample videos and print a summary table.

Usage:
    python demos/demo_all_samples.py
    python demos/demo_all_samples.py --checkpoint weights/ctc_best_v2.pt
"""

from __future__ import annotations

import argparse
import csv
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import SignLanguageRecognizer

# ── Default paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_CKPT    = PROJECT_ROOT / "weights" / "ctc_best_v2.pt"
SAMPLES_DIR     = PROJECT_ROOT / "sample_videos"
TEST_LABELS_CSV = SAMPLES_DIR  / "test_labels.csv"

DIVIDER_LONG  = "=" * 68
DIVIDER_SHORT = "-" * 68
NAME_COL_W    = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_references(csv_path: Path) -> dict[str, list[str]]:
    """
    Load every name → gloss-list mapping from test_labels.csv.

    Returns an empty dict if the file does not exist.
    """
    references: dict[str, list[str]] = {}
    if not csv_path.exists():
        print(f"Warning: reference file not found at {csv_path}")
        return references

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        sample_text = csv_file.read(512)
        csv_file.seek(0)
        delimiter = "|" if "|" in sample_text else ","
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        for row in reader:
            name     = row.get("name", "").strip()
            raw_orth = row.get("orth", "").strip()
            if name:
                references[name] = raw_orth.split() if raw_orth else []

    return references


def compute_wer(hypothesis: list[str], reference: list[str]) -> float:
    """Word Error Rate = editdistance(hyp, ref) / len(ref)."""
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return _edit_distance(hypothesis, reference) / len(reference)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Sign Language Recognition — all-samples demo"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help=f"Path to model checkpoint (default: {DEFAULT_CKPT.name})",
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    recognizer = SignLanguageRecognizer(args.checkpoint, device="auto")
    references = load_all_references(TEST_LABELS_CSV)

    # ── Collect video folders ─────────────────────────────────────────────────
    video_folders = sorted([
        folder for folder in SAMPLES_DIR.iterdir()
        if folder.is_dir() and any(folder.glob("*.png"))
    ])

    if not video_folders:
        print(f"No video folders with .png frames found in {SAMPLES_DIR}")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────────────
    results: list[dict] = []

    for folder in video_folders:
        video_name = folder.name
        print(f"\nProcessing: {video_name}")

        predicted_glosses = recognizer.predict_from_phoenix_folder(str(folder))
        reference_glosses = references.get(video_name)

        if reference_glosses is not None:
            wer = compute_wer(predicted_glosses, reference_glosses)
        else:
            wer = float("nan")

        results.append({
            "name":      video_name,
            "predicted": predicted_glosses,
            "reference": reference_glosses,
            "wer":       wer,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{DIVIDER_LONG}")
    print("  Demo on all sample videos")
    print(DIVIDER_LONG)

    header = (
        f"{'Video':<{NAME_COL_W}}  "
        f"{'Ref':>5}  {'Pred':>5}  {'WER':>6}"
    )
    print(header)
    print(DIVIDER_SHORT)

    total_edit_errors = 0
    total_ref_words   = 0

    for result in results:
        ref_count  = len(result["reference"]) if result["reference"] is not None else "-"
        pred_count = len(result["predicted"])
        wer_value  = result["wer"]
        wer_str    = f"{wer_value:.2f}" if not math.isnan(wer_value) else "N/A"

        truncated_name = result["name"][:NAME_COL_W]
        print(
            f"{truncated_name:<{NAME_COL_W}}  "
            f"{str(ref_count):>5}  {pred_count:>5}  {wer_str:>6}"
        )

        if result["reference"] is not None and not math.isnan(wer_value):
            total_edit_errors += _edit_distance(
                result["predicted"], result["reference"]
            )
            total_ref_words += len(result["reference"])

    print(DIVIDER_SHORT)
    if total_ref_words > 0:
        aggregate_wer = total_edit_errors / total_ref_words
        print(f"Aggregate WER: {aggregate_wer:.2f}")
    else:
        print("Aggregate WER: N/A (no references loaded)")
    print(DIVIDER_LONG)

    # ── Detailed predictions ──────────────────────────────────────────────────
    print("\nDetailed predictions:\n")
    for result in results:
        print(f"[{result['name']}]")
        if result["reference"] is not None:
            ref_display = (
                " ".join(result["reference"]) if result["reference"] else "<empty>"
            )
            print(f"  REF:  {ref_display}")
        pred_display = (
            " ".join(result["predicted"]) if result["predicted"] else "<no prediction>"
        )
        print(f"  PRED: {pred_display}")
        print()


if __name__ == "__main__":
    main()
