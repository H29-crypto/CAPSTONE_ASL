"""
demo.py — Phase-Aware ASL Sign Recognition Webcam Demo
------------------------------------------------------
Usage:  python demo.py
Keys:   R = record 2 seconds and predict
        Q = quit

Requires: torch, mediapipe, opencv-python, numpy
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import urllib.request

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
except ImportError:
    sys.exit("mediapipe not installed.  Run:  pip install mediapipe")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PHASE_CKPT_PATH = BASE_DIR / "models" / "phase_tcn_best_safe_state.pt"
REC_CKPT_PATH   = BASE_DIR / "models" / "recognition_tcn_attention_best.pt"
LABEL_MAP_PATH  = BASE_DIR / "manifests" / "recognition_label_map_with_split_counts.csv"

RECORD_SECONDS = 2.0
TARGET_FPS     = 30
N_FRAMES       = int(RECORD_SECONDS * TARGET_FPS)   # 60

POSE_IDXS = [0, 11, 12, 13, 14, 15, 16]
N_POSE    = 7
N_HAND    = 21

# MediaPipe Tasks API model files (downloaded once to models/)
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"
HAND_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
POSE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")

# UI colours (BGR)
C_GREEN  = (0, 220, 0)
C_RED    = (0, 0, 220)
C_ORANGE = (0, 165, 255)
C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)
C_DARK   = (30, 30, 30)

# Phase labels and per-phase colours (BGR)
PHASE_NAMES  = {0: "Background", 1: "Preparation", 2: "Stroke", 3: "Retraction"}
PHASE_COLORS = {
    0: (100, 100, 100),   # gray
    1: (0, 200, 220),     # yellow
    2: (0, 220, 0),       # green
    3: (220, 80, 0),      # blue
}


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES  (verbatim from training notebook cells 52/59)
# ─────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

    @staticmethod
    def _fix(out, ref):
        if out.shape[-1] > ref.shape[-1]:
            return out[:, :, :ref.shape[-1]]
        if out.shape[-1] < ref.shape[-1]:
            return F.pad(out, (0, ref.shape[-1] - out.shape[-1]))
        return out

    def forward(self, x):
        out = self.drop1(F.relu(self.bn1(self._fix(self.conv1(x), x))))
        out = self.drop2(F.relu(self.bn2(self._fix(self.conv2(out), x))))
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + self._fix(res, out))


class PhaseTCN(nn.Module):
    def __init__(self, input_dim, num_classes=4, hidden_dim=192,
                 num_blocks=5, kernel_size=5, dropout=0.20):
        super().__init__()
        blocks, in_ch = [], input_dim
        for i in range(num_blocks):
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size, 2 ** i, dropout))
            in_ch = hidden_dim
        self.tcn        = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):          # x: (B, T, 444)
        return self.classifier(self.tcn(x.transpose(1, 2))).transpose(1, 2)


class SafeMaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, dropout=0.10):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, mask):    # h: (B, T, H), mask: (B, T)
        scores = self.attn(h.float()).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        w      = torch.softmax(scores, dim=1).masked_fill(~mask.bool(), 0.0)
        return torch.sum(h.float() * w.unsqueeze(-1), dim=1), w


class RecognitionTCNAttentionSafe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256,
                 num_blocks=4, kernel_size=5, dropout=0.30, attention_dropout=0.10):
        super().__init__()
        blocks, in_ch = [], input_dim
        for i in range(num_blocks):
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size, 2 ** i, dropout))
            in_ch = hidden_dim
        self.tcn            = nn.Sequential(*blocks)
        self.attention_pool = SafeMaskedAttentionPooling(hidden_dim, attention_dropout)
        self.classifier     = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask):    # x: (B, T, 448)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pooled, attn = self.attention_pool(h, mask)
        return self.classifier(pooled.float()), attn


# ─────────────────────────────────────────────────────────────
# PREPROCESSING  (verbatim from training notebook cells 42/43/45)
# ─────────────────────────────────────────────────────────────

def _sf(x):
    a = np.asarray(x, dtype=np.float32)
    a[~np.isfinite(a)] = 0.0
    return a


def _mk(m, T):
    if m is None:
        return np.ones(T, dtype=bool)
    m = np.asarray(m)
    if m.ndim == 0:
        return np.ones(T, dtype=bool) * bool(m)
    if m.shape[0] != T:
        return np.ones(T, dtype=bool)
    if m.ndim == 1:
        return m > 0
    return np.any(m > 0, axis=tuple(range(1, m.ndim)))


def _interp(arr, vmask):
    arr = _sf(arr)
    T   = arr.shape[0]
    vmask = _mk(vmask, T)
    if T == 0:
        return arr
    if vmask.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    out  = arr.copy()
    out[~vmask] = np.nan
    flat = out.reshape(T, -1)
    x    = np.arange(T)
    for j in range(flat.shape[1]):
        col  = flat[:, j]
        good = np.isfinite(col)
        if   good.sum() == 0: flat[:, j] = 0.0
        elif good.sum() == 1: flat[:, j] = col[good][0]
        else:                 flat[:, j] = np.interp(x, x[good], col[good])
    out = flat.reshape(arr.shape).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def robust_normalize_keypoints(pose, lh, rh,
                                pose_mask=None, lh_mask=None, rh_mask=None):
    """Training normalization: center on mean of 7 pose points, scale by pose spread."""
    pose, lh, rh = _sf(pose), _sf(lh), _sf(rh)
    T = pose.shape[0]

    pose_i = _interp(pose, _mk(pose_mask, T))
    lh_i   = _interp(lh,   _mk(lh_mask,  T))
    rh_i   = _interp(rh,   _mk(rh_mask,  T))

    pose_xy   = pose_i[:, :, :2]                          # (T, 7, 2)
    center_xy = np.mean(pose_xy, axis=1, keepdims=True)   # (T, 1, 2)

    spread    = np.max(pose_xy, axis=1) - np.min(pose_xy, axis=1)  # (T, 2)
    fs        = np.maximum(spread[:, 0], spread[:, 1])
    fs        = fs[np.isfinite(fs) & (fs > 1e-6)]
    scale     = float(np.median(fs)) if len(fs) > 0 else 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    def _n(p):
        o = p.copy().astype(np.float32)
        o[:, :, :2] = (o[:, :, :2] - center_xy) / scale
        o[:, :,  2] =  o[:, :,  2]               / scale
        o[~np.isfinite(o)] = 0.0
        return o

    pts = np.concatenate([_n(pose_i), _n(lh_i), _n(rh_i)], axis=1).astype(np.float32)
    pos = pts.reshape(T, -1).astype(np.float32)
    pos[~np.isfinite(pos)] = 0.0
    return pts, pos


def compute_motion_features(pts, pos):
    T   = pos.shape[0]
    vel = np.zeros_like(pos,  dtype=np.float32)
    acc = np.zeros_like(pos,  dtype=np.float32)
    if T >= 2: vel[1:] = pos[1:] - pos[:-1]
    if T >= 3: acc[1:] = vel[1:] - vel[:-1]

    gs = np.sqrt(np.mean(vel ** 2, axis=1)).astype(np.float32)

    hpts = pts[:, 7:, :]                                  # (T, 42, 3) lh+rh
    hv   = np.zeros_like(hpts, dtype=np.float32)
    if T >= 2: hv[1:] = hpts[1:] - hpts[:-1]
    hs   = np.sqrt(np.mean(hv ** 2, axis=(1, 2))).astype(np.float32)

    for a in [vel, acc, gs, hs]:
        a[~np.isfinite(a)] = 0.0
    return vel, acc, gs, hs


def _r01(x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    if len(x) == 0:
        return x
    p5, p95 = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    d = p95 - p5
    if not np.isfinite(d) or d < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p5) / d, 0.0, 1.0).astype(np.float32)


def _smooth(x, w=7):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or w <= 1 or len(x) < 3:
        return x.copy()
    w = int(w)
    if w % 2 == 0: w += 1
    w = min(w, len(x))
    if w % 2 == 0: w -= 1
    if w < 3: return x.copy()
    pad = w // 2
    y   = np.convolve(np.pad(x, (pad, pad), "edge"),
                      np.ones(w, np.float32) / w, "valid").astype(np.float32)
    y[~np.isfinite(y)] = 0.0
    return y


def _aw(T):
    if T < 40:  return 5
    if T < 80:  return 7
    if T < 160: return 9
    if T < 300: return 13
    return 17


def compute_phase_speed(pts, pos, gs, hs):
    """Blended phase-speed signal (notebook cell 45)."""
    T   = pos.shape[0]
    p   = pts.reshape(T, 49, 3)
    lhc = np.mean(p[:, 7:28,  :], axis=1)   # left-hand centroid
    rhc = np.mean(p[:, 28:49, :], axis=1)   # right-hand centroid

    lhv = np.zeros_like(lhc, dtype=np.float32)
    rhv = np.zeros_like(rhc, dtype=np.float32)
    if T >= 2:
        lhv[1:] = lhc[1:] - lhc[:-1]
        rhv[1:] = rhc[1:] - rhc[:-1]

    cs = np.maximum(np.linalg.norm(lhv, axis=1),
                    np.linalg.norm(rhv, axis=1)).astype(np.float32)
    cs[~np.isfinite(cs)] = 0.0

    raw = (0.65 * _r01(cs) + 0.25 * _r01(hs) + 0.10 * _r01(gs)).astype(np.float32)
    sp  = _smooth(raw, _aw(T))
    if T >= 160:
        sp = _smooth(sp, 5)
    return np.clip(sp, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PHASE POST-PROCESSING  (verbatim from notebook cell 53)
# ─────────────────────────────────────────────────────────────

def _maj(labels, w=5):
    labels = np.asarray(labels, dtype=np.int64)
    T      = len(labels)
    if T == 0 or w <= 1:
        return labels.copy()
    if w % 2 == 0: w += 1
    pad    = w // 2
    padded = np.pad(labels, (pad, pad), "edge")
    out    = labels.copy()
    for i in range(T):
        out[i] = int(np.argmax(np.bincount(padded[i: i + w], minlength=4)))
    return out.astype(np.int64)


def _segs(mask):
    mask = np.asarray(mask, dtype=bool)
    segs, s = [], None
    for i, v in enumerate(mask):
        if v     and s is None: s = i
        elif not v and s is not None: segs.append((s, i - 1)); s = None
    if s is not None: segs.append((s, len(mask) - 1))
    return segs


def _fgap(mask, mg=2):
    mask = np.asarray(mask, dtype=bool).copy()
    ss   = _segs(mask)
    for i in range(len(ss) - 1):
        e, s2 = ss[i][1], ss[i + 1][0]
        if 0 < s2 - e - 1 <= mg:
            mask[e + 1: s2] = True
    return mask


def _rdrop(mask, ml=2):
    mask = np.asarray(mask, dtype=bool).copy()
    for s, e in _segs(mask):
        if (e - s + 1) < ml:
            mask[s: e + 1] = False
    return mask


def extract_active_region(pred, probs, pspeed):
    pred = np.asarray(pred, dtype=np.int64)
    T    = len(pred)

    if T < 8:
        return 0, T - 1, np.full(T, 2, np.int64), "too_short"

    am = _fgap(_rdrop(pred != 0, max(2, int(0.02 * T))), max(2, int(0.03 * T)))
    ss = _segs(am)

    if not ss:
        thr = 0.30 * float(np.max(pspeed)) if np.max(pspeed) > 1e-8 else 0.0
        if thr < 1e-8:
            return 0, T - 1, np.full(T, 2, np.int64), "fallback_flat"
        am = _fgap(_rdrop(pspeed >= thr, max(2, int(0.02 * T))), max(2, int(0.03 * T)))
        ss = _segs(am)
        if not ss:
            return 0, T - 1, np.full(T, 2, np.int64), "fallback_active"

    as_ = int(max(0, ss[0][0]))
    ae  = int(min(T - 1, ss[-1][1]))
    al  = ae - as_ + 1

    if al < 8:
        return 0, T - 1, np.full(T, 2, np.int64), "active_too_short"

    sm = np.zeros(T, dtype=bool)
    sm[as_: ae + 1] = pred[as_: ae + 1] == 2
    sm = _fgap(_rdrop(sm, max(2, int(0.03 * al))), max(1, int(0.02 * al)))
    ss2 = _segs(sm)

    if not ss2:
        pk  = as_ + int(np.argmax(probs[as_: ae + 1, 2]))
        sl  = min(max(4, int(round(0.20 * al))), al)
        sst = int(max(as_, min(pk - sl // 2, ae - sl + 1)))
        se_ = sst + sl - 1
    else:
        sst, se_ = max(ss2, key=lambda x: x[1] - x[0])

    sst, se_ = int(max(as_, sst)), int(min(ae, se_))
    if sst <= as_ and al >= 10: sst = as_ + 1
    if se_ >= ae  and al >= 10: se_ = ae  - 1

    if sst > se_:
        return 0, T - 1, np.full(T, 2, np.int64), "bad_stroke"

    labels = np.zeros(T, dtype=np.int64)
    labels[as_: sst]    = 1
    labels[sst: se_ + 1] = 2
    labels[se_ + 1: ae + 1] = 3
    return as_, ae, labels, "ok"


# ─────────────────────────────────────────────────────────────
# LIVE PHASE PREDICTION  (runs during RECORD state)
# ─────────────────────────────────────────────────────────────

def predict_current_phase(fi, pose_buf, lh_buf, rh_buf,
                           pose_msk, lh_msk, rh_msk,
                           phase_model, phase_mean, phase_std, device):
    """Run PhaseTCN on pose_buf[0:fi]. Returns (last_frame_phase, per_frame_labels)."""
    if fi < 4:
        return 0, np.zeros(fi, dtype=np.int64)

    pts, pos = robust_normalize_keypoints(
        pose_buf[:fi], lh_buf[:fi], rh_buf[:fi],
        pose_msk[:fi], lh_msk[:fi], rh_msk[:fi],
    )
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([
        pos, vel, acc,
        gs.reshape(-1, 1), hs.reshape(-1, 1), psp.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0

    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0

    with torch.no_grad():
        xt     = torch.tensor(Xn).unsqueeze(0).to(device)
        probs4 = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()

    labels = _maj(np.argmax(probs4, axis=-1).astype(np.int64))
    return int(labels[-1]), labels


# ─────────────────────────────────────────────────────────────
# FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(pose_seq, lh_seq, rh_seq, pm, lm, rm,
                 phase_model, phase_mean, phase_std,
                 rec_model,   rec_mean,   rec_std, device):
    """
    pose_seq / lh_seq / rh_seq : (T, 7/21/21, 3) raw MediaPipe coords
    *m  : (T,) bool detection masks
    Returns probs (100,), seg_status str, active_start int, active_end int
    """
    pts, pos = robust_normalize_keypoints(pose_seq, lh_seq, rh_seq, pm, lm, rm)
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([
        pos,                         # (T, 147)
        vel,                         # (T, 147)
        acc,                         # (T, 147)
        gs.reshape(-1, 1),           # (T,   1)
        hs.reshape(-1, 1),           # (T,   1)
        psp.reshape(-1, 1),          # (T,   1)
    ], axis=1).astype(np.float32)    # (T, 444)
    X[~np.isfinite(X)] = 0.0

    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0

    with torch.no_grad():
        xt     = torch.tensor(Xn).unsqueeze(0).to(device)
        probs4 = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()
    raw_pred = np.argmax(probs4, axis=-1).astype(np.int64)
    smooth   = _maj(raw_pred)

    as_, ae, ordered, status = extract_active_region(smooth, probs4, psp)

    X_crop  = X[as_: ae + 1].copy()
    ph_crop = np.clip(ordered[as_: ae + 1], 0, 3).astype(np.int64)
    X_crop  = ((X_crop - rec_mean) / rec_std).astype(np.float32)
    X_crop[~np.isfinite(X_crop)] = 0.0
    X_final = np.concatenate([X_crop, np.eye(4, dtype=np.float32)[ph_crop]], axis=1)

    T_act = X_final.shape[0]
    xt2   = torch.tensor(X_final).unsqueeze(0).to(device)
    msk   = torch.ones(1, T_act, dtype=torch.bool).to(device)
    with torch.no_grad():
        logits, _ = rec_model(xt2, msk)
        probs100  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return probs100, status, as_, ae


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE TASKS API — LANDMARK EXTRACTION
# ─────────────────────────────────────────────────────────────

def _download_if_needed(url, path):
    if not path.exists():
        print(f"  Downloading {path.name} …")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved {path.name} ({path.stat().st_size // 1024} KB)")


def make_landmarkers():
    """Downloads models if needed and returns (hand_lmk, pose_lmk) in IMAGE mode."""
    _download_if_needed(HAND_MODEL_URL, HAND_MODEL_PATH)
    _download_if_needed(POSE_MODEL_URL, POSE_MODEL_PATH)

    hand_opts = _mp_vision.HandLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    pose_opts = _mp_vision.PoseLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (_mp_vision.HandLandmarker.create_from_options(hand_opts),
            _mp_vision.PoseLandmarker.create_from_options(pose_opts))


def extract_frame_landmarks(rgb_frame, hand_lmk, pose_lmk):
    """
    Runs Tasks API detectors on one RGB frame.
    Identical landmark format to the training extract_keypoints function.
    Returns pose (7,3), lh (21,3), rh (21,3) and detection bools.
    """
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_res = hand_lmk.detect(mp_img)
    pose_res = pose_lmk.detect(mp_img)

    pose = np.zeros((N_POSE, 3), dtype=np.float32)
    pose_ok = False
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for o, i in enumerate(POSE_IDXS):
            pose[o] = (lms[i].x, lms[i].y, lms[i].z)
        pose_ok = True

    lh = np.zeros((N_HAND, 3), dtype=np.float32)
    rh = np.zeros((N_HAND, 3), dtype=np.float32)
    lh_ok = rh_ok = False
    if hand_res.hand_landmarks:
        for hand_lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            label  = handed[0].category_name   # "Left" or "Right"
            target = lh if label == "Left" else rh
            for i, lm in enumerate(hand_lms):
                target[i] = (lm.x, lm.y, lm.z)
            if label == "Left": lh_ok = True
            else:               rh_ok = True

    return pose, lh, rh, pose_ok, lh_ok, rh_ok


def draw_landmarks_on_frame(frame, pose_res, hand_res):
    """Draws detected landmarks as coloured dots (no drawing_utils dependency)."""
    h, w = frame.shape[:2]
    if pose_res.pose_landmarks:
        for idx in POSE_IDXS:
            lm = pose_res.pose_landmarks[0][idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 200, 0), -1)
    if hand_res.hand_landmarks:
        for hand_lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            color = (255, 120, 0) if handed[0].category_name == "Left" else (0, 120, 255)
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, color, -1)


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def draw_top5(frame, top5_idx, probs, id_to_gloss):
    h, w       = frame.shape[:2]
    panel_w    = 380
    panel_h    = 225
    x0         = w - panel_w - 10
    y0         = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), C_GREEN, 2)
    cv2.putText(frame, "TOP-5 PREDICTIONS",
                (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

    for rank, idx in enumerate(top5_idx):
        gloss = id_to_gloss.get(int(idx), f"ID_{idx}")
        conf  = float(probs[idx])
        yy    = y0 + 50 + rank * 34
        color = C_WHITE if rank == 0 else C_GRAY

        bar_max = panel_w - 140
        bar_len = int(conf * bar_max)
        cv2.rectangle(frame, (x0 + 130, yy - 14), (x0 + 130 + bar_max, yy + 4),
                      (60, 60, 60), -1)
        if bar_len > 0:
            bar_col = C_GREEN if rank == 0 else (60, 160, 60)
            cv2.rectangle(frame, (x0 + 130, yy - 14), (x0 + 130 + bar_len, yy + 4),
                          bar_col, -1)

        cv2.putText(frame, f"#{rank + 1} {gloss[:16]}",
                    (x0 + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"{conf * 100:.1f}%",
                    (x0 + panel_w - 65, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)


def draw_bottom_bar(frame, msg, color=C_WHITE):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), C_DARK, -1)
    cv2.putText(frame, msg, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_record_bar(frame, fi, n_frames, phase_history=None, current_phase=0):
    h, w = frame.shape[:2]

    # Phase timeline strip (above the progress bar)
    strip_y0, strip_y1 = h - 72, h - 56
    sx0, sx1 = 10, w - 10
    strip_w  = sx1 - sx0
    cv2.rectangle(frame, (sx0, strip_y0), (sx1, strip_y1), (40, 40, 40), -1)
    if phase_history:
        for i, ph in enumerate(phase_history):
            x0 = sx0 + int(i       * strip_w / n_frames)
            x1 = sx0 + int((i + 1) * strip_w / n_frames)
            cv2.rectangle(frame, (x0, strip_y0), (x1, strip_y1),
                          PHASE_COLORS.get(int(ph), (100, 100, 100)), -1)

    # Progress bar
    bar_w = int((fi / n_frames) * (w - 20))
    cv2.rectangle(frame, (10, h - 52), (w - 10, h - 42), (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (10, h - 52), (10 + bar_w, h - 42), C_RED, -1)

    # REC counter + current phase badge
    ph_color = PHASE_COLORS.get(current_phase, C_GRAY)
    ph_name  = PHASE_NAMES.get(current_phase, "?")
    cv2.putText(frame, f"REC  {fi}/{n_frames}",
                (10, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2)
    cv2.putText(frame, ph_name,
                (w - 180, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_color, 2)


# ─────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────

def _load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _np(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    for p in [PHASE_CKPT_PATH, REC_CKPT_PATH, LABEL_MAP_PATH]:
        if not p.exists():
            sys.exit(f"Missing: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Label map
    id_to_gloss = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_gloss[int(row["rec_label_id"])] = row["gloss"]
    print(f"Labels: {len(id_to_gloss)} classes")

    # Phase TCN
    print("Loading Phase TCN ...")
    pc    = _load_ckpt(PHASE_CKPT_PATH, device)
    pcfg  = pc["config"]
    pmean = _np(pc["feature_mean"])
    pstd  = _np(pc["feature_std"])
    phase_model = PhaseTCN(**{k: pcfg[k] for k in
                              ["input_dim", "num_classes", "hidden_dim",
                               "num_blocks", "kernel_size", "dropout"]})
    phase_model.load_state_dict(pc["model_state_dict"])
    phase_model.to(device).eval()
    print(f"  val_accuracy = {pc.get('val_accuracy', 0):.4f}")

    # Recognition TCN
    print("Loading Recognition TCN ...")
    rc    = _load_ckpt(REC_CKPT_PATH, device)
    rcfg  = rc["config"]
    rmean = _np(rc["feature_mean"])
    rstd  = _np(rc["feature_std"])
    rec_model = RecognitionTCNAttentionSafe(
        input_dim=rcfg["input_dim"], num_classes=rcfg["num_classes"],
        hidden_dim=rcfg["hidden_dim"], num_blocks=rcfg["num_blocks"],
        kernel_size=rcfg["kernel_size"], dropout=rcfg["dropout"],
        attention_dropout=rcfg["attention_dropout"])
    rec_model.load_state_dict(rc["model_state_dict"])
    rec_model.to(device).eval()
    print(f"  val_top1 = {rc.get('val_top1', 0):.4f}  "
          f"val_top5 = {rc.get('val_top5', 0):.4f}")

    # MediaPipe Tasks API landmarkers
    print("Initialising MediaPipe landmarkers ...")
    hand_lmk, pose_lmk = make_landmarkers()
    print("  Landmarkers ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hand_lmk.close(); pose_lmk.close()
        sys.exit("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 50)
    print("  ASL Demo ready.  R = record  |  Q = quit")
    print("=" * 50)

    # State machine
    IDLE = 0; COUNTDOWN = 1; RECORD = 2; INFER = 3

    state         = IDLE
    cd_t          = 0.0
    fi            = 0
    top5_idx      = None
    probs100      = None
    seg_info      = ""
    phase_history = []
    current_phase = 0

    pose_buf = np.zeros((N_FRAMES, N_POSE, 3), np.float32)
    lh_buf   = np.zeros((N_FRAMES, N_HAND, 3), np.float32)
    rh_buf   = np.zeros((N_FRAMES, N_HAND, 3), np.float32)
    pose_msk = np.zeros(N_FRAMES, dtype=bool)
    lh_msk   = np.zeros(N_FRAMES, dtype=bool)
    rh_msk   = np.zeros(N_FRAMES, dtype=bool)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_res = hand_lmk.detect(mp_img)
        pose_res = pose_lmk.detect(mp_img)

        # Live skeleton overlay (Tasks API — draw dots manually)
        draw_landmarks_on_frame(frame, pose_res, hand_res)

        # ── State logic ────────────────────────────────────────
        if state == IDLE:
            if top5_idx is not None:
                draw_top5(frame, top5_idx, probs100, id_to_gloss)
                cv2.putText(frame, seg_info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1)
            draw_bottom_bar(frame, "R = record sign  |  Q = quit", C_GREEN)

        elif state == COUNTDOWN:
            remaining = 3 - int(time.time() - cd_t)
            if remaining <= 0:
                state = RECORD
                fi    = 0
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, str(remaining),
                            (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, C_ORANGE, 8)
                draw_bottom_bar(frame, "Get ready ...", C_ORANGE)

        elif state == RECORD:
            pa, la, ra, po, lo, ro = extract_frame_landmarks(rgb, hand_lmk, pose_lmk)
            if fi < N_FRAMES:
                pose_buf[fi] = pa; lh_buf[fi] = la; rh_buf[fi] = ra
                pose_msk[fi] = po; lh_msk[fi] = lo; rh_msk[fi] = ro
                fi += 1
                if fi >= 8 and fi % 5 == 0:
                    current_phase, frame_phases = predict_current_phase(
                        fi, pose_buf, lh_buf, rh_buf,
                        pose_msk, lh_msk, rh_msk,
                        phase_model, pmean, pstd, device,
                    )
                    phase_history = list(frame_phases)
            draw_record_bar(frame, fi, N_FRAMES, phase_history, current_phase)
            draw_bottom_bar(frame, "Recording ... hold your sign!", C_RED)
            if fi >= N_FRAMES:
                state = INFER

        elif state == INFER:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Processing ...",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ORANGE, 3)
            draw_bottom_bar(frame, "Running pipeline ...", C_ORANGE)
            cv2.imshow("ASL Demo", frame)
            cv2.waitKey(1)

            try:
                probs100, seg_status, as_, ae = run_pipeline(
                    pose_buf, lh_buf, rh_buf,
                    pose_msk, lh_msk, rh_msk,
                    phase_model, pmean, pstd,
                    rec_model,   rmean, rstd,
                    device,
                )
                top5_idx = np.argsort(probs100)[::-1][:5]
                seg_info = f"phase_seg={seg_status}  active=[{as_},{ae}]"
                print(f"\n  TOP-5:")
                for rk, ix in enumerate(top5_idx, 1):
                    g = id_to_gloss.get(int(ix), f"ID_{ix}")
                    print(f"  #{rk}  {g:<25}  {probs100[ix] * 100:.1f}%")
            except Exception as e:
                print(f"  [ERROR] {e}")
                top5_idx = None
                seg_info = f"error: {e}"

            state = IDLE
            continue   # skip waitKey below so IDLE draws immediately

        cv2.imshow("ASL Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r") and state == IDLE:
            pose_buf[:] = 0; lh_buf[:] = 0; rh_buf[:] = 0
            pose_msk[:] = False; lh_msk[:] = False; rh_msk[:] = False
            phase_history = []
            current_phase = 0
            state = COUNTDOWN
            cd_t  = time.time()

    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
