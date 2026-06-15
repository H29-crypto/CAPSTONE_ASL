"""Greedy CTC decoding + development WER.

This WER is for model selection and tracking ONLY. Every number that goes
in the report must come from the official PHOENIX evaluation — see
src/official_eval_adapter.py.
"""
import numpy as np
import torch


def greedy_decode(log_probs, lengths, blank=0):
    """log_probs [T,B,V] -> list of id sequences (collapse repeats, drop blank)."""
    ids = log_probs.argmax(-1).transpose(0, 1).cpu().numpy()   # [B, T]
    out = []
    for row, L in zip(ids, lengths.cpu().numpy()):
        seq, prev = [], blank
        for t in range(int(L)):
            c = int(row[t])
            if c != blank and c != prev:
                seq.append(c)
            prev = c
        out.append(seq)
    return out


def ids_to_glosses(seqs, id2gloss):
    return [[id2gloss[i] for i in s] for s in seqs]


def align_counts(ref, hyp):
    """Levenshtein DP on gloss lists -> (n_ref, sub, ins, dele) + alignment ops.

    ops: list of (op, ref_token_or_None, hyp_token_or_None) with op in
    {'C','S','I','D'} — used by notebook 08 to colour errors.
    """
    R, H = len(ref), len(hyp)
    D = np.zeros((R + 1, H + 1), dtype=np.int32)
    D[:, 0] = np.arange(R + 1)
    D[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + 1, D[i, j - 1] + 1)
    ops, i, j = [], R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None)); i -= 1
        else:
            ops.append(("I", None, hyp[j - 1])); j -= 1
    ops.reverse()
    sub = sum(1 for o in ops if o[0] == "S")
    ins = sum(1 for o in ops if o[0] == "I")
    dele = sum(1 for o in ops if o[0] == "D")
    return R, sub, ins, dele, ops


def wer(refs, hyps):
    """Corpus-level WER over lists of gloss lists. Returns dict with breakdown."""
    n = s = i = d = 0
    for r, h in zip(refs, hyps):
        R, sub, ins, dele, _ = align_counts(r, h)
        n += R; s += sub; i += ins; d += dele
    werr = 100.0 * (s + i + d) / max(n, 1)
    return {"wer": werr, "sub": s, "ins": i, "del": d, "n_ref": n}


@torch.no_grad()
def evaluate_greedy(model, loader, id2gloss, device):
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        out = model(batch, device)
        seqs = greedy_decode(out["main"], out["out_lengths"])
        hyps += ids_to_glosses(seqs, id2gloss)
        refs += batch["glosses"]
    return wer(refs, hyps), refs, hyps
