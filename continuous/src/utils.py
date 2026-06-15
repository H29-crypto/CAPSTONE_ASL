"""Shared utilities. Every notebook starts with: from src.utils import load_config."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # derived values
    kp = cfg["keypoints"]
    cfg["model"]["kp_in_dim"] = kp["num_points"] * len(kp["channels"])
    return cfg


def p(cfg, *keys):
    """Path helper: p(cfg, 'features_rgb') -> Path from cfg['paths']."""
    out = Path(cfg["paths"][keys[0]])
    for k in keys[1:]:
        out = out / k
    return out


def corpus_path(cfg, split):
    return (Path(cfg["paths"]["dataset_root"]) / cfg["paths"]["corpus_dir"]
            / cfg["paths"]["annotations_template"].format(split=split))


def frames_dir(cfg, split):
    return (Path(cfg["paths"]["dataset_root"]) / cfg["paths"]["corpus_dir"]
            / cfg["paths"]["frames_template"].format(split=split))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def update_json(path, new_items: dict):
    """Read-modify-write for incremental length/failure logs (resumable dumps)."""
    data = load_json(path) if os.path.exists(path) else {}
    data.update(new_items)
    save_json(data, path)
    return data
