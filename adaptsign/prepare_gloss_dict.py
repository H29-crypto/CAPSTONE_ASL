"""
prepare_gloss_dict.py — Generate gloss_dict.npy for AdaptSign (PHOENIX-2014-T)

Two modes:

  1. From annotation CSVs (recommended — gives real gloss names):
       python adaptsign/prepare_gloss_dict.py --csv-dir PATH_TO/PHOENIX-2014-T/annotations/manual

     PATH_TO should contain:
       PHOENIX-2014-T.train.corpus.csv
       PHOENIX-2014-T.dev.corpus.csv
       PHOENIX-2014-T.test.corpus.csv

  2. From checkpoint (fallback — shows "GLOSS_N" labels but lets the model run):
       python adaptsign/prepare_gloss_dict.py --from-checkpoint adaptsign/weights/phoenix2014-T_best.pt

Output:  adaptsign/preprocess/phoenix2014-T/gloss_dict.npy
"""

import argparse
import sys
from pathlib import Path
import numpy as np

ADAPTSIGN_DIR  = Path(__file__).parent
OUTPUT_PATH    = ADAPTSIGN_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
CSV_NAMES      = [
    "PHOENIX-2014-T.train.corpus.csv",
    "PHOENIX-2014-T.dev.corpus.csv",
    "PHOENIX-2014-T.test.corpus.csv",
]


def from_csvs(csv_dir: Path) -> dict:
    """Parse Phoenix-2014-T annotation CSVs → sorted gloss dict."""
    import pandas as pd

    sign_dict: dict[str, int] = {}
    for name in CSV_NAMES:
        path = csv_dir / name
        if not path.exists():
            sys.exit(f"[ERROR] CSV not found: {path}")
        df = pd.read_csv(path, sep="|")
        # column is 'orth' — space-separated gloss sequence per clip
        for row in df["orth"].dropna():
            for gloss in str(row).split():
                sign_dict[gloss] = sign_dict.get(gloss, 0) + 1

    sorted_items = sorted(sign_dict.items(), key=lambda d: d[0])
    gloss_dict = {k: [idx + 1, cnt] for idx, (k, cnt) in enumerate(sorted_items)}
    print(f"  Vocabulary size: {len(gloss_dict)} glosses")
    return gloss_dict


def from_checkpoint(ckpt_path: Path) -> dict:
    """Inspect checkpoint classifier shape → placeholder gloss dict."""
    import torch
    print("  Reading classifier shape from checkpoint ...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    # NormLinear weight shape: (in_dim, num_classes)
    num_classes = None
    for key in ("classifier.weight", "conv1d.fc.weight"):
        if key in sd:
            num_classes = sd[key].shape[1]
            break
    if num_classes is None:
        sys.exit("[ERROR] Cannot determine num_classes from checkpoint keys: "
                 + str(list(sd.keys())[:10]))
    vocab_size = num_classes - 1  # exclude CTC blank (index 0)
    print(f"  Detected num_classes = {num_classes}  →  vocab size = {vocab_size}")
    gloss_dict = {f"GLOSS_{i}": [i, 1] for i in range(1, num_classes)}
    return gloss_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv-dir", type=Path,
                       help="Directory containing the three Phoenix-2014-T corpus CSVs")
    group.add_argument("--from-checkpoint", type=Path, metavar="CKPT",
                       help="Path to AdaptSign checkpoint (.pt) for placeholder dict")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.csv_dir:
        print(f"Building gloss dict from CSVs in: {args.csv_dir}")
        gloss_dict = from_csvs(args.csv_dir)
    else:
        print(f"Building placeholder gloss dict from checkpoint: {args.from_checkpoint}")
        gloss_dict = from_checkpoint(args.from_checkpoint)

    np.save(str(OUTPUT_PATH), gloss_dict)
    print(f"  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
