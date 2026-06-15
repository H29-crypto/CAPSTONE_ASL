"""PHOENIX-2014 corpus parsing + train-only gloss vocabulary.

corpus.csv format (pipe-separated), multisigner release:
    id|folder|signer|annotation
where `folder` is typically like  <id>/1/*.png  (a glob relative to the
fullFrame directory of that split) and `annotation` is the space-separated
gloss sequence.
"""
import csv
from pathlib import Path

from .utils import corpus_path, save_json, load_json


def parse_corpus(cfg, split):
    """Return list of dicts: {id, folder, signer, glosses}."""
    items = []
    with open(corpus_path(cfg, split), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            glosses = row["annotation"].strip().split()
            items.append({
                "id": row["id"],
                "folder": row["folder"],
                "signer": row.get("signer", ""),
                "glosses": glosses,
            })
    return items


def build_vocab(train_items, blank_index=0):
    """Gloss -> id, with id 0 reserved for the CTC blank. Train split ONLY."""
    glosses = sorted({g for it in train_items for g in it["glosses"]})
    assert blank_index == 0, "code assumes blank=0 throughout"
    gloss2id = {g: i + 1 for i, g in enumerate(glosses)}   # 1..V
    return gloss2id


def save_vocab(gloss2id, path):
    save_json(gloss2id, path)


def load_vocab(path):
    g2i = load_json(path)
    i2g = {int(v): k for k, v in g2i.items()}
    return g2i, i2g


def encode(glosses, gloss2id):
    """Train-split glosses are guaranteed in-vocab; raises otherwise on purpose."""
    return [gloss2id[g] for g in glosses]


def frame_glob(cfg, split, folder):
    """Resolve the frame file glob for one sample."""
    from .utils import frames_dir
    base = frames_dir(cfg, split)
    folder = folder.strip()
    if folder.endswith("*.png"):
        return str(base / folder)
    return str(Path(base) / folder / "*.png")
