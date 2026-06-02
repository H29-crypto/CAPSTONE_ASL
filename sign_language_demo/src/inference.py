"""
inference.py — High-level SignLanguageRecognizer API.

Loads a BiLSTMCTC_V2 checkpoint and exposes two prediction methods:
    predict_from_phoenix_folder()  — reads imagesXXXX.png from a directory
    predict_from_frames()          — accepts pre-loaded numpy RGB arrays
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .decoder import greedy_ctc_decode
from .feature_extractor import FeatureExtractor
from .model import BiLSTMCTC_V2


def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to the best available device string."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SignLanguageRecognizer:
    """
    High-level inference API for continuous sign language recognition.

    Loads a BiLSTMCTC_V2 checkpoint that must contain:
        model      — model state dict
        gloss2idx  — dict mapping gloss string → integer index
        idx2gloss  — dict mapping integer index → gloss string
        config     — dict with model hyper-parameters
        dev_wer    — validation WER at checkpoint time (float, 0–1 range)
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Load model and vocabulary from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint.
            device:          'auto' picks cuda > mps > cpu automatically.
                             Pass 'cpu', 'cuda', or 'mps' to force a device.

        Raises:
            FileNotFoundError: if the checkpoint file does not exist.
            KeyError:          if required keys are missing from the checkpoint.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Make sure ctc_best_v2.pt is in the weights/ directory."
            )

        self.device_str: str   = _resolve_device(device)
        self.device:     torch.device = torch.device(self.device_str)

        print(f"Loading checkpoint from {path.name} ...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        for required_key in ("model", "gloss2idx", "idx2gloss", "config"):
            if required_key not in checkpoint:
                raise KeyError(
                    f"Checkpoint is missing required key '{required_key}'. "
                    f"Found keys: {list(checkpoint.keys())}"
                )

        config = checkpoint["config"]

        self.gloss_to_index: dict[str, int] = checkpoint["gloss2idx"]
        self.index_to_gloss: dict[int, str] = checkpoint["idx2gloss"]
        self.vocab_size:     int            = config["vocab_size"]
        self.dev_wer:        float          = float(checkpoint.get("dev_wer", float("nan")))

        # Build model and load weights
        self.model = BiLSTMCTC_V2(
            input_dim    = config["input_dim"],
            hidden_dim   = config["hidden_dim"],
            num_layers   = config["num_layers"],
            vocab_size   = config["vocab_size"],
            dropout      = config["dropout"],
            lstm_dropout = config["lstm_dropout"],
            input_dropout= config["input_dropout"],
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()

        self.feature_extractor = FeatureExtractor(device=self.device_str)
        print(f"  Ready — device: {self.device_str}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def predict_from_phoenix_folder(
        self,
        folder_path: str,
        frame_stride: int = 2,
    ) -> list[str]:
        """
        Run recognition on a PHOENIX-style folder of PNG frames.

        Expects files named imagesXXXX.png (zero-padded); plain sorted()
        works correctly because filenames are zero-padded.

        Args:
            folder_path:  Directory containing imagesXXXX.png files.
            frame_stride: Keep every nth frame to match training (default 2).

        Returns:
            List of predicted gloss strings, e.g. ['WETTER', 'MORGEN'].
            Returns an empty list if no glosses are predicted.

        Raises:
            FileNotFoundError: if the folder or PNG files are not found.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Video folder not found: {folder}")

        png_files = sorted(folder.glob("*.png"))
        if not png_files:
            raise FileNotFoundError(f"No .png files found in: {folder}")

        frames: list[np.ndarray] = [
            np.array(Image.open(png).convert("RGB")) for png in png_files
        ]
        return self.predict_from_frames(frames, frame_stride=frame_stride)

    def predict_from_frames(
        self,
        frames: list,
        frame_stride: int = 2,
    ) -> list[str]:
        """
        Run recognition on a list of pre-loaded RGB frames.

        Args:
            frames:       List of H × W × 3 numpy arrays in RGB order.
            frame_stride: Keep every nth frame to match training (default 2).

        Returns:
            List of predicted gloss strings.
            Returns an empty list if no glosses are predicted.
        """
        strided_frames = frames[::frame_stride]
        if not strided_frames:
            return []

        # (T, 512) — extracted on the configured device, returned on CPU
        frame_features = self.feature_extractor.extract(strided_frames)

        # Add batch dimension → (1, T, 512)
        feature_tensor = frame_features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(feature_tensor)   # (1, T, vocab_size)

        # Remove batch dim → (T, vocab_size) then decode
        token_indices = greedy_ctc_decode(log_probs.squeeze(0), blank_idx=1)

        glosses: list[str] = [
            self.index_to_gloss.get(idx, "<unk>") for idx in token_indices
        ]
        return glosses
