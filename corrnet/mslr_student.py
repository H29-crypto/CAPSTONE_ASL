"""
mslr_student.py — MSLR Student Model for Continuous German SL Recognition
--------------------------------------------------------------------------
Lightweight MediaPipe → TemporalConv → BiLSTM → CTC pipeline.
Architecture follows KD-MSLRT (AAAI 2025) + reuses CorrNet modules.

Input:  (B, T, 444) MediaPipe motion features
Output: CTC gloss sequence + confidence scores

Inference: <10 ms on CPU  (vs 3-5 s for CorrNet)
Model size: ~8 MB          (vs 398 MB for CorrNet)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

CORRNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRNET_DIR))

from modules.tconv import TemporalConv
from modules.BiLSTM import BiLSTMLayer


# ══════════════════════════════════════════════════════════════════════════════
# Greedy CTC decoder  (same as corrnet_webcam.py — copied to avoid circular import)
# ══════════════════════════════════════════════════════════════════════════════

class GreedyDecoder:
    """CTC greedy decoder — returns (gloss, pos, confidence) triples."""

    def __init__(self, gloss_dict: dict):
        self.i2g      = {v[0]: k for k, v in gloss_dict.items()}
        self.blank_id = 0

    def decode(self, nn_output: torch.Tensor, vid_lgt: torch.Tensor,
               batch_first: bool = True, probs: bool = False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        results = []
        for b in range(nn_output.shape[0]):
            length  = int(vid_lgt[b].item())
            logits  = nn_output[b, :length]
            sm      = torch.softmax(logits, dim=-1)
            indices = logits.argmax(dim=-1).cpu().tolist()
            seq, prev, rs, rn = [], None, 0.0, 0
            for pos, idx in enumerate(indices):
                if idx != prev:
                    if prev is not None and prev != self.blank_id and seq:
                        g, p, _ = seq[-1]; seq[-1] = (g, p, rs / max(rn, 1))
                    prev, rs, rn = idx, 0.0, 0
                    if idx != self.blank_id:
                        seq.append((self.i2g.get(idx, f"<{idx}>"), len(seq), 0.0))
                if idx != self.blank_id:
                    rs += float(sm[pos, idx].item()); rn += 1
            if prev is not None and prev != self.blank_id and seq:
                g, p, _ = seq[-1]; seq[-1] = (g, p, rs / max(rn, 1))
            results.append(seq)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# MSLR Student Network
# ══════════════════════════════════════════════════════════════════════════════

class MSLRStudent(nn.Module):
    """
    MediaPipe Sign Language Recognition student model.

    Architecture (KD-MSLRT, AAAI 2025):
        (B, T, 444) landmarks
          → TemporalConv  [K5-P2-K5-P2, 444→512]  feat_len ≈ T/4
          → BiLSTM        [2 layers, bidirectional]
          → Linear        [512→num_classes]
          → CTC decoder
    """

    def __init__(
        self,
        gloss_dict: dict,
        input_size: int  = 444,
        hidden_size: int = 512,
        conv_type: int   = 2,
        num_layers: int  = 2,
        dropout: float   = 0.3,
    ):
        super().__init__()
        num_classes = len(gloss_dict) + 1   # +1 for CTC blank

        self.temporal_conv = TemporalConv(
            input_size  = input_size,
            hidden_size = hidden_size,
            conv_type   = conv_type,
            use_bn      = True,
            num_classes = num_classes,
        )
        self.bilstm = BiLSTMLayer(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            bidirectional = True,
        )
        # Shared classifier — same weights used by temporal_conv's conv_logits path
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.temporal_conv.fc = self.classifier

        self.decoder     = GreedyDecoder(gloss_dict)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict:
        """
        Args:
            x:       (B, T, input_size) landmark feature sequences
            lengths: (B,) actual sequence lengths

        Returns dict:
            sequence_logits  (T', B, C)
            conv_logits      (T', B, C)
            feat_len         (B,)
            recognized_sents list[list[(gloss, pos, conf)]]  — None during training
        """
        # (B, T, D) → (B, D, T)
        x_t = x.transpose(1, 2)

        conv_out    = self.temporal_conv(x_t, lengths)
        visual_feat = conv_out["visual_feat"]    # (T', B, hidden)
        feat_len    = conv_out["feat_len"]       # (B,)
        conv_logits = conv_out["conv_logits"]    # (T', B, C)

        bilstm_out  = self.bilstm(visual_feat, feat_len.cpu())
        predictions = bilstm_out["predictions"]  # (T', B, hidden)

        seq_logits = self.classifier(predictions)  # (T', B, C)

        recognized = None if self.training else \
            self.decoder.decode(seq_logits, feat_len, batch_first=False)

        return {
            "sequence_logits":  seq_logits,
            "conv_logits":      conv_logits,
            "feat_len":         feat_len,
            "recognized_sents": recognized,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[list, list]:
        """Convenience inference — returns (glosses, confidences)."""
        out  = self.forward(x, lengths)
        pred = out["recognized_sents"]
        if pred and pred[0]:
            glosses = [item[0] for item in pred[0]]
            confs   = [item[2] if len(item) > 2 else 1.0 for item in pred[0]]
            return glosses, confs
        return [], []

    def save(self, path: str | Path):
        torch.save({"model_state_dict": self.state_dict()}, str(path))

    @classmethod
    def load(cls, path: str | Path, gloss_dict: dict, **kwargs) -> "MSLRStudent":
        model = cls(gloss_dict, **kwargs)
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        sd    = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=True)
        return model
