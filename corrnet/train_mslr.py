"""
train_mslr.py — Train MSLR student with knowledge distillation from CorrNet
---------------------------------------------------------------------------
L_total = L_CTC + lambda_kd * L_KD
L_CTC   = CTC(student_logits, hard_gloss_labels)
L_KD    = KLDiv(student/T, teacher/T)   temperature T=4

Data augmentation (from KD-MSLRT paper):
    - Spatial rotation ±15°
    - Temporal speed jitter ×[0.8, 1.2]
    - Random translation ±5%

Usage:
    python corrnet/train_mslr.py
    python corrnet/train_mslr.py --epochs 80 --data corrnet/data/pseudo_labels
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DIR    = Path(__file__).resolve().parent.parent
CORRNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRNET_DIR))
sys.path.insert(0, str(BASE_DIR))

from mslr_student import MSLRStudent

# ── Hyper-parameters ──────────────────────────────────────────────────────────
KD_TEMP   = 4.0
KD_WEIGHT = 0.5
LR        = 1e-3
WD        = 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# Landmark data augmentation  (KD-MSLRT paper)
# ══════════════════════════════════════════════════════════════════════════════

def aug_rotate(X: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    """Rotate (x,y) landmark coordinates by a random angle."""
    rad   = math.radians(random.uniform(-max_deg, max_deg))
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    X = X.copy()
    for i in range(0, 147, 3):       # pos features: indices 0..146, stride 3
        x, y = X[:, i].copy(), X[:, i+1].copy()
        X[:, i]   = cos_a * x - sin_a * y
        X[:, i+1] = sin_a * x + cos_a * y
    return X


def aug_speed(X: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Resample sequence at a random speed factor."""
    T     = len(X)
    T_new = max(4, int(T * random.uniform(lo, hi)))
    idx   = np.linspace(0, T - 1, T_new).astype(np.int64)
    return X[idx]


def aug_translate(X: np.ndarray, shift: float = 0.05) -> np.ndarray:
    """Random spatial translation of landmark coordinates."""
    X  = X.copy()
    dx = random.uniform(-shift, shift)
    dy = random.uniform(-shift, shift)
    for i in range(0, 147, 3):
        X[:, i]   += dx
        X[:, i+1] += dy
    return X


def augment(X: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p: X = aug_rotate(X)
    if random.random() < p: X = aug_speed(X)
    if random.random() < p: X = aug_translate(X)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PseudoDataset(Dataset):
    def __init__(self, paths: list[Path], do_augment: bool = True):
        self.paths      = paths
        self.do_augment = do_augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        d = np.load(str(self.paths[idx]), allow_pickle=True)
        X = d["features"].astype(np.float32)        # (T, 444)
        T = d["teacher_logits"].astype(np.float32)  # (T', C)
        y = int(d["hard_label"])

        if self.do_augment:
            X = augment(X)

        return torch.from_numpy(X), torch.from_numpy(T), y, len(X)


def collate(batch):
    """Pad to same length, sort descending (needed for BiLSTM packing)."""
    batch.sort(key=lambda b: b[3], reverse=True)
    max_T = batch[0][3]
    B, D  = len(batch), batch[0][0].shape[-1]

    feats  = torch.zeros(B, max_T, D)
    lens   = torch.zeros(B, dtype=torch.long)
    t_logs = [b[1] for b in batch]          # list of (T'_i, C) — variable len
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    l_lens = torch.ones(B, dtype=torch.long)

    for i, (X, _, _, T) in enumerate(batch):
        feats[i, :T] = X
        lens[i]      = T

    return feats, lens, t_logs, labels, l_lens


# ══════════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════════

def kd_loss(student_logits: torch.Tensor,
            teacher_logits: list[torch.Tensor],
            feat_len: torch.Tensor,
            tau: float = 4.0) -> torch.Tensor:
    """KL-divergence knowledge distillation loss."""
    loss = torch.tensor(0.0, device=student_logits.device)
    cnt  = 0
    for b in range(student_logits.shape[1]):
        T_s = int(feat_len[b].item())
        s   = student_logits[:T_s, b]       # (T_s, C)
        t   = teacher_logits[b].to(s.device)  # (T_t, C)
        Tm  = min(T_s, t.shape[0])
        if Tm == 0: continue
        s_log = F.log_softmax(s[:Tm] / tau, dim=-1)
        t_sm  = F.softmax(t[:Tm] / tau, dim=-1)
        loss  = loss + F.kl_div(s_log, t_sm, reduction="batchmean")
        cnt  += 1
    return loss / max(cnt, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Training / eval
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, ctc_fn, device, train: bool):
    model.train(train)
    tot_loss = tot_ctc = tot_kd = 0.0

    with torch.set_grad_enabled(train):
        for feats, lens, t_logs, labels, l_lens in loader:
            feats  = feats.to(device)
            lens   = lens.to(device)
            labels = labels.to(device)

            out       = model(feats, lens)
            seq_logits = out["sequence_logits"]   # (T', B, C)
            feat_len   = out["feat_len"]

            log_p = F.log_softmax(seq_logits, dim=-1)
            ctc   = ctc_fn(log_p, labels, feat_len.int(), l_lens.to(device).int()).mean()
            kd    = kd_loss(seq_logits, t_logs, feat_len, KD_TEMP)

            loss = ctc + KD_WEIGHT * kd

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            tot_loss += loss.item()
            tot_ctc  += ctc.item()
            tot_kd   += kd.item()

    n = max(len(loader), 1)
    return tot_loss / n, tot_ctc / n, tot_kd / n


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       default="corrnet/data/pseudo_labels")
    ap.add_argument("--out",        default="corrnet/weights/mslr_student.pt")
    ap.add_argument("--epochs",     type=int,   default=80)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--batch",      type=int,   default=4)
    ap.add_argument("--save_every", type=int,   default=10,
                    help="Save a periodic checkpoint every N epochs (default 10)")
    ap.add_argument("--resume",     default=None,
                    help="Path to a checkpoint to resume training from")
    args = ap.parse_args()

    data_dir = Path(args.data) if Path(args.data).is_absolute() \
               else BASE_DIR / args.data
    out_path = Path(args.out) if Path(args.out).is_absolute() \
               else BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        sys.exit(f"\n[ERROR] No training data at {data_dir}\n"
                 "Run first: python corrnet/collect_pseudo_labels.py\n")

    gloss_dict  = np.load(
        str(CORRNET_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"),
        allow_pickle=True).item()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Training data: {len(paths)} clips")

    # Split 90/10
    random.shuffle(paths)
    n_val   = max(1, len(paths) // 10)
    val_p   = paths[:n_val]
    train_p = paths[n_val:]

    train_dl = DataLoader(PseudoDataset(train_p, True),  batch_size=args.batch,
                          shuffle=True,  collate_fn=collate, drop_last=False)
    val_dl   = DataLoader(PseudoDataset(val_p, False), batch_size=1,
                          shuffle=False, collate_fn=collate)

    model = MSLRStudent(gloss_dict).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  MSLR student: {n_p:,} params  ({n_p*4/1024/1024:.1f} MB)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    ctc_fn    = nn.CTCLoss(reduction="none", zero_infinity=True)

    best        = float("inf")
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume) if Path(args.resume).is_absolute() \
                      else BASE_DIR / args.resume
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best        = ckpt.get("best_val_ctc", float("inf"))
            print(f"  Resumed from epoch {start_epoch-1}  "
                  f"val_ctc={ckpt.get('val_ctc', float('nan')):.4f}")
        else:
            print(f"  [WARN] Resume checkpoint not found: {resume_path}")

    print(f"\n  Training {len(train_p)} / val {len(val_p)}  "
          f"— epochs {start_epoch}..{args.epochs}")
    print("="*60)

    def _save(path, extra=None):
        payload = {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch":                epoch,
            "val_ctc":              vl_ctc,
            "best_val_ctc":         best,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    for epoch in range(start_epoch, args.epochs + 1):
        tr_l, tr_ctc, tr_kd = run_epoch(model, train_dl, optimizer, ctc_fn, device, True)
        vl_l, vl_ctc, _     = run_epoch(model, val_dl,   None,      ctc_fn, device, False)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Ep {epoch:3d}/{args.epochs}  "
              f"train={tr_l:.4f}(ctc={tr_ctc:.3f} kd={tr_kd:.3f})  "
              f"val_ctc={vl_ctc:.4f}  lr={lr_now:.1e}")

        if vl_ctc < best:
            best = vl_ctc
            _save(out_path)
            print(f"    ✓ Best saved → {out_path.name}  (val_ctc={best:.4f})")

        if epoch % args.save_every == 0:
            periodic = ckpt_dir / f"mslr_epoch_{epoch:03d}.pt"
            _save(periodic)
            print(f"    · Checkpoint → {periodic.name}")

    print(f"\n  Training complete. Best val_ctc={best:.4f}")
    print(f"  Best weights : {out_path}")
    print(f"  Checkpoints  : {ckpt_dir}")
    print(f"\n  Next: python corrnet/mslr_webcam.py")


if __name__ == "__main__":
    main()
