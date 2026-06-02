"""
model.py — BiLSTMCTC_V2 architecture.

Must match the training checkpoint exactly — do not change layer ordering.
"""

import torch
import torch.nn as nn


class BiLSTMCTC_V2(nn.Module):
    """
    Bidirectional LSTM encoder with a CTC classification head for
    continuous sign language recognition.

    Input:  (batch, time, input_dim)  — ResNet-18 frame features
    Output: (batch, time, vocab_size) — log-probabilities (log_softmax applied)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        vocab_size: int = 1088,
        dropout: float = 0.5,
        lstm_dropout: float = 0.4,
        input_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_proj    = nn.Linear(input_dim, hidden_dim)
        self.input_act     = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = lstm_dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, time, input_dim) feature tensor.
            lengths: (batch,) actual sequence lengths — enables packed-sequence
                     optimisation; pass None to skip packing.

        Returns:
            (batch, time, vocab_size) log-probability tensor.
        """
        x = self.input_dropout(x)
        x = self.input_act(self.input_proj(x))

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        out, _ = self.lstm(x)

        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        logits = self.classifier(out)
        return logits.log_softmax(dim=-1)
