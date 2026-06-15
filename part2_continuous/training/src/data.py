"""Feature-space dataset for CSLR.

Loads pre-dumped RGB features and/or processed keypoint features per sample,
enforces the stream-alignment rules from the frozen plan, applies temporal
augmentation with THE SAME index map to both streams, and pads batches.
"""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import encode

log = logging.getLogger("data")

MAX_LEN_MISMATCH = 2   # frames; trim to min if within, else exclude + log


def _aug_indices(T, rescale, drop_p, rng):
    """Temporal augmentation as an index map so both streams transform identically."""
    idx = np.arange(T)
    if rescale and rescale > 0:
        factor = rng.uniform(1.0 - rescale, 1.0 + rescale)
        new_T = max(4, int(round(T * factor)))
        idx = np.clip(np.round(np.linspace(0, T - 1, new_T)).astype(int), 0, T - 1)
    if drop_p and drop_p > 0 and len(idx) > 8:
        keep = rng.random(len(idx)) >= drop_p
        if keep.sum() < 4:
            keep[:4] = True
        idx = idx[keep]
    return idx


class FeatureDataset(Dataset):
    """streams: subset of {'rgb','kp'}. Train mode adds augmentation + CTC targets."""

    def __init__(self, cfg, items, split, streams=("rgb", "kp"),
                 gloss2id=None, train_mode=False, ids_subset=None):
        self.cfg, self.split, self.streams = cfg, split, tuple(streams)
        self.train_mode = train_mode
        self.gloss2id = gloss2id
        self.rgb_dir = Path(cfg["paths"]["features_rgb"]) / split
        self.kp_dir = Path(cfg["paths"]["features_kp"]) / split
        aug = cfg["train"]["augmentation"]
        self.rescale, self.drop_p = aug["temporal_rescale"], aug["frame_drop_p"]

        if ids_subset is not None:
            ids_subset = set(ids_subset)
            items = [it for it in items if it["id"] in ids_subset]

        self.items, self.excluded = [], []
        for it in items:
            ok = all((self._path(s, it["id"])).exists() for s in self.streams)
            if ok:
                self.items.append(it)
            else:
                self.excluded.append(it["id"])
        if self.excluded:
            log.warning("%s/%s: %d samples missing feature files (excluded). First: %s",
                        split, "+".join(self.streams), len(self.excluded), self.excluded[:5])

    def _path(self, stream, sid):
        return (self.rgb_dir if stream == "rgb" else self.kp_dir) / f"{sid}.npy"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        feats = {s: np.load(self._path(s, it["id"])).astype(np.float32)
                 for s in self.streams}

        # ---- strict alignment (frozen-plan rule) -------------------------
        if len(self.streams) == 2:
            t_rgb, t_kp = len(feats["rgb"]), len(feats["kp"])
            d = abs(t_rgb - t_kp)
            if d > MAX_LEN_MISMATCH:
                raise RuntimeError(
                    f"{it['id']}: rgb T={t_rgb} vs kp T={t_kp} (diff {d} > "
                    f"{MAX_LEN_MISMATCH}). Inspect this sample; do not silently pass.")
            T = min(t_rgb, t_kp)
            feats = {s: f[:T] for s, f in feats.items()}

        # ---- shared-index augmentation (train only) ----------------------
        if self.train_mode:
            T = len(next(iter(feats.values())))
            idx = _aug_indices(T, self.rescale, self.drop_p, np.random)
            feats = {s: f[idx] for s, f in feats.items()}

        out = {"id": it["id"], "glosses": it["glosses"],
               "T": len(next(iter(feats.values())))}
        for s, f in feats.items():
            out[s] = torch.from_numpy(np.ascontiguousarray(f))
        if self.train_mode:
            out["target"] = torch.tensor(encode(it["glosses"], self.gloss2id),
                                         dtype=torch.long)
        return out


def collate(batch):
    """Pad to batch max T; return lengths + (train) flat CTC targets."""
    batch = sorted(batch, key=lambda b: -b["T"])
    out = {"ids": [b["id"] for b in batch],
           "glosses": [b["glosses"] for b in batch],
           "lengths": torch.tensor([b["T"] for b in batch], dtype=torch.long)}
    Tmax = int(out["lengths"][0])
    for s in ("rgb", "kp"):
        if s in batch[0]:
            D = batch[0][s].shape[1]
            x = torch.zeros(len(batch), Tmax, D)
            for i, b in enumerate(batch):
                x[i, :b["T"]] = b[s]
            out[s] = x
    if "target" in batch[0]:
        out["targets"] = torch.cat([b["target"] for b in batch])
        out["target_lengths"] = torch.tensor([len(b["target"]) for b in batch],
                                             dtype=torch.long)
    return out
