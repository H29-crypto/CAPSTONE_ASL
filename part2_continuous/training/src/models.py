"""Models for the frozen plan.

CSLRModel(streams=...) covers the whole experiment table with config flags:
  E1: streams=('rgb',)             aux_ctc=False
  E2: streams=('kp',)              aux_ctc=False
  E3: streams=('rgb','kp')         fusion='concat', aux_ctc=False
  E4: streams=('rgb','kp')         fusion='concat', aux_ctc=True
  ladder A: head 'bilstm' vs 'transformer'   |   ladder B: fusion 'concat' vs 'gated'

Matched-conditions rule: encoder_dim / head / schedule identical across rows.
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StreamEncoder(nn.Module):
    """Linear projection -> stacked Conv1d blocks -> optional temporal pooling."""

    def __init__(self, in_dim, dim, layers, kernel, dropout, pool=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Conv1d(dim, dim, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else None
        self.pool_factor = pool

    def forward(self, x, lengths):                    # x: [B, T, in_dim]
        x = self.proj(x).transpose(1, 2)              # [B, dim, T]
        x = self.tcn(x)
        if self.pool is not None:
            x = self.pool(x)
            lengths = torch.div(lengths, self.pool_factor, rounding_mode="floor").clamp(min=1)
        return x.transpose(1, 2), lengths             # [B, T', dim]


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.wa, self.wb = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, a, b):
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * self.wa(a) + (1 - g) * self.wb(b)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                             # [B, T, dim]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SequenceHead(nn.Module):
    """BiLSTM (packed) or TransformerEncoder (masked) over fused features."""

    def __init__(self, m):
        super().__init__()
        self.kind, dim = m["head"], m["encoder_dim"]
        if self.kind == "bilstm":
            self.rnn = nn.LSTM(dim, m["bilstm_hidden"], m["bilstm_layers"],
                               batch_first=True, bidirectional=True,
                               dropout=m["dropout"] if m["bilstm_layers"] > 1 else 0.0)
            self.out_dim = 2 * m["bilstm_hidden"]
        elif self.kind == "transformer":
            self.posenc = PositionalEncoding(dim)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=m["transformer_heads"],
                dim_feedforward=m["transformer_ff"], dropout=m["dropout"],
                batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, m["transformer_layers"])
            self.out_dim = dim
        else:
            raise ValueError(self.kind)

    def forward(self, x, lengths):                    # x: [B, T, dim]
        if self.kind == "bilstm":
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            y, _ = self.rnn(packed)
            y, _ = pad_packed_sequence(y, batch_first=True, total_length=x.size(1))
            return y
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        return self.tr(self.posenc(x), src_key_padding_mask=mask)


class CSLRModel(nn.Module):
    def __init__(self, cfg, num_classes, streams=("rgb", "kp")):
        super().__init__()
        m = cfg["model"]
        self.streams = tuple(streams)
        in_dims = {"rgb": m["rgb_in_dim"], "kp": m["kp_in_dim"]}
        self.enc = nn.ModuleDict({
            s: StreamEncoder(in_dims[s], m["encoder_dim"], m["tcn_layers"],
                             m["tcn_kernel"], m["dropout"], m["temporal_pool"])
            for s in self.streams})
        self.aux_ctc = bool(m["aux_ctc"]) and len(self.streams) > 1
        if self.aux_ctc:
            self.aux_head = nn.ModuleDict(
                {s: nn.Linear(m["encoder_dim"], num_classes) for s in self.streams})
        self.fusion_kind = m["fusion"]
        if len(self.streams) > 1:
            if self.fusion_kind == "concat":
                self.fuse = nn.Sequential(
                    nn.Linear(2 * m["encoder_dim"], m["encoder_dim"]),
                    nn.ReLU(inplace=True), nn.Dropout(m["dropout"]))
            elif self.fusion_kind == "gated":
                self.fuse = GatedFusion(m["encoder_dim"])
            else:
                raise ValueError(self.fusion_kind)
        self.head = SequenceHead(m)
        self.classifier = nn.Linear(self.head.out_dim, num_classes)

    def forward(self, batch, device):
        enc_out, lengths = {}, None
        for s in self.streams:
            enc_out[s], lengths = self.enc[s](batch[s].to(device),
                                              batch["lengths"].to(device))
        if len(self.streams) == 1:
            fused = enc_out[self.streams[0]]
        elif self.fusion_kind == "concat":
            fused = self.fuse(torch.cat([enc_out[s] for s in self.streams], dim=-1))
        else:
            fused = self.fuse(enc_out[self.streams[0]], enc_out[self.streams[1]])

        y = self.head(fused, lengths)
        out = {"main": self.classifier(y).log_softmax(-1).transpose(0, 1),  # [T,B,V]
               "out_lengths": lengths}
        if self.aux_ctc:
            for s in self.streams:
                out[f"aux_{s}"] = (self.aux_head[s](enc_out[s])
                                   .log_softmax(-1).transpose(0, 1))
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
