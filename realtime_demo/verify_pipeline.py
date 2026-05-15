"""
verify_pipeline.py
------------------
Verifies the ASL recognition pipeline end-to-end before the webcam demo.
Uses the EXACT model architectures and preprocessing from the training notebooks.

Steps:
  1. Load both checkpoints and verify architecture shapes match.
  2. Open webcam, display live feed with countdown.
  3. Record 2 seconds at 30 fps while showing the sign.
  4. Run the full pipeline:
       raw MediaPipe landmarks
       -> robust_normalize_keypoints (training preprocessing)
       -> compute_motion_features
       -> phase_speed (blended, smoothed)
       -> PhaseTCN -> segment extraction
       -> RecognitionTCNAttentionSafe -> top-5
  5. Print top-5 predictions and confirm pipeline runs without error.
"""

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
    sys.exit("mediapipe not installed. Run: pip install mediapipe")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PHASE_CKPT_PATH = BASE_DIR / "models" / "phase_tcn_best_safe_state.pt"
REC_CKPT_PATH   = BASE_DIR / "models" / "recognition_tcn_attention_best.pt"
LABEL_MAP_PATH  = BASE_DIR / "manifests" / "recognition_label_map_with_split_counts.csv"

RECORD_SECONDS = 2.0
TARGET_FPS     = 30
N_FRAMES       = int(RECORD_SECONDS * TARGET_FPS)   # 60 frames

POSE_IDXS = [0, 11, 12, 13, 14, 15, 16]   # nose + upper-body landmarks
N_POSE    = 7
N_HAND    = 21

# MediaPipe Tasks API model files (downloaded once to models/)
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"
HAND_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
POSE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES (verbatim from training notebook cells 52/59)
# ─────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        def _fix(out, ref):
            if out.shape[-1] > ref.shape[-1]:
                return out[:, :, :ref.shape[-1]]
            if out.shape[-1] < ref.shape[-1]:
                return F.pad(out, (0, ref.shape[-1] - out.shape[-1]))
            return out

        out = _fix(self.conv1(x), x)
        out = F.relu(self.bn1(out))
        out = self.dropout1(out)
        out = _fix(self.conv2(out), x)
        out = F.relu(self.bn2(out))
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        residual  = _fix(residual, out)
        return F.relu(out + residual)


class PhaseTCN(nn.Module):
    def __init__(self, input_dim, num_classes=4, hidden_dim=192,
                 num_blocks=5, kernel_size=5, dropout=0.20):
        super().__init__()
        blocks, in_ch = [], input_dim
        for i in range(num_blocks):
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size,
                                        dilation=2**i, dropout=dropout))
            in_ch = hidden_dim
        self.tcn        = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):          # x: (B, T, 444)
        x = x.transpose(1, 2)     # (B, 444, T)
        h = self.tcn(x)            # (B, 192, T)
        return self.classifier(h).transpose(1, 2)  # (B, T, 4)


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
        w      = torch.softmax(scores, dim=1)
        w      = w.masked_fill(~mask.bool(), 0.0)
        pooled = torch.sum(h.float() * w.unsqueeze(-1), dim=1)
        return pooled, w


class RecognitionTCNAttentionSafe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256,
                 num_blocks=4, kernel_size=5, dropout=0.30, attention_dropout=0.10):
        super().__init__()
        blocks, in_ch = [], input_dim
        for i in range(num_blocks):
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size,
                                        dilation=2**i, dropout=dropout))
            in_ch = hidden_dim
        self.tcn            = nn.Sequential(*blocks)
        self.attention_pool = SafeMaskedAttentionPooling(hidden_dim, attention_dropout)
        self.classifier     = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask):    # x: (B, T, 448), mask: (B, T)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, 256)
        pooled, attn = self.attention_pool(h, mask)
        return self.classifier(pooled.float()), attn        # (B, 100), (B, T)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING (verbatim from training notebook cells 42/43/45)
# ─────────────────────────────────────────────────────────────

def _safe_float(x):
    a = np.asarray(x, dtype=np.float32)
    a[~np.isfinite(a)] = 0.0
    return a

def _ensure_mask(mask, T):
    if mask is None:
        return np.ones(T, dtype=bool)
    m = np.asarray(mask)
    if m.ndim == 0:
        return np.ones(T, dtype=bool) * bool(m)
    if m.shape[0] != T:
        return np.ones(T, dtype=bool)
    if m.ndim == 1:
        return m > 0
    return np.any(m > 0, axis=tuple(range(1, m.ndim)))

def _interpolate_missing(arr, valid_mask):
    """arr: (T, N, C), valid_mask: (T,) → fills missing frames by linear interp."""
    arr = _safe_float(arr)
    T   = arr.shape[0]
    valid_mask = _ensure_mask(valid_mask, T)
    if T == 0:
        return arr
    if valid_mask.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    out  = arr.copy()
    out[~valid_mask] = np.nan
    flat = out.reshape(T, -1)
    x    = np.arange(T)
    for j in range(flat.shape[1]):
        col  = flat[:, j]
        good = np.isfinite(col)
        if good.sum() == 0:
            flat[:, j] = 0.0
        elif good.sum() == 1:
            flat[:, j] = col[good][0]
        else:
            flat[:, j] = np.interp(x, x[good], col[good])
    out = flat.reshape(arr.shape).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def robust_normalize_keypoints(pose, lh, rh,
                                pose_mask=None, lh_mask=None, rh_mask=None):
    """
    Exactly as in training (notebook cell 42/43).
    pose: (T, 7, 3), lh: (T, 21, 3), rh: (T, 21, 3)
    Returns: normalized_points (T, 49, 3), positions (T, 147)
    """
    pose = _safe_float(pose)
    lh   = _safe_float(lh)
    rh   = _safe_float(rh)
    T    = pose.shape[0]

    pose_mask = _ensure_mask(pose_mask, T)
    lh_mask   = _ensure_mask(lh_mask,   T)
    rh_mask   = _ensure_mask(rh_mask,   T)

    pose_i = _interpolate_missing(pose, pose_mask)
    lh_i   = _interpolate_missing(lh,   lh_mask)
    rh_i   = _interpolate_missing(rh,   rh_mask)

    pose_xy    = pose_i[:, :, :2]                          # (T, 7, 2)
    center_xy  = np.mean(pose_xy, axis=1, keepdims=True)  # (T, 1, 2)  — mean of all 7

    spread_xy  = np.max(pose_xy, axis=1) - np.min(pose_xy, axis=1)   # (T, 2)
    frame_scales = np.maximum(spread_xy[:, 0], spread_xy[:, 1])
    frame_scales = frame_scales[np.isfinite(frame_scales) & (frame_scales > 1e-6)]
    scale = float(np.median(frame_scales)) if len(frame_scales) > 0 else 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    def _norm_part(part):
        out = part.copy().astype(np.float32)
        out[:, :, :2] = (out[:, :, :2] - center_xy) / scale
        out[:, :,  2] =  out[:, :,  2]               / scale
        out[~np.isfinite(out)] = 0.0
        return out

    pose_n = _norm_part(pose_i)
    lh_n   = _norm_part(lh_i)
    rh_n   = _norm_part(rh_i)

    normalized_points = np.concatenate([pose_n, lh_n, rh_n], axis=1).astype(np.float32)
    positions         = normalized_points.reshape(T, -1).astype(np.float32)
    positions[~np.isfinite(positions)] = 0.0
    return normalized_points, positions


def compute_motion_features(normalized_points, positions):
    """
    Exactly as in training (notebook cell 43).
    Returns: velocity (T,147), acceleration (T,147), global_speed (T,), hand_speed (T,)
    """
    T    = positions.shape[0]
    vel  = np.zeros_like(positions, dtype=np.float32)
    acc  = np.zeros_like(positions, dtype=np.float32)
    if T >= 2:
        vel[1:] = positions[1:] - positions[:-1]
    if T >= 3:
        acc[1:] = vel[1:] - vel[:-1]

    global_speed = np.sqrt(np.mean(vel ** 2, axis=1)).astype(np.float32)

    hand_points = normalized_points[:, 7:, :]   # (T, 42, 3)  lh + rh
    hand_vel    = np.zeros_like(hand_points, dtype=np.float32)
    if T >= 2:
        hand_vel[1:] = hand_points[1:] - hand_points[:-1]
    hand_speed = np.sqrt(np.mean(hand_vel ** 2, axis=(1, 2))).astype(np.float32)

    for arr in [vel, acc]:
        arr[~np.isfinite(arr)] = 0.0
    global_speed[~np.isfinite(global_speed)] = 0.0
    hand_speed[~np.isfinite(hand_speed)]     = 0.0
    return vel, acc, global_speed, hand_speed


def _robust_01(x):
    """Robust per-clip [0,1] normalization using 5th/95th percentile (notebook cell 45)."""
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    if len(x) == 0:
        return x
    p5, p95 = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    denom   = p95 - p5
    if not np.isfinite(denom) or denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - p5) / denom, 0.0, 1.0).astype(np.float32)
    y[~np.isfinite(y)] = 0.0
    return y


def _smooth_1d(x, window=7):
    """Moving-average smoothing (notebook cell 43/45)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or window <= 1 or len(x) < 3:
        return x.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(x))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return x.copy()
    pad    = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    y = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    y[~np.isfinite(y)] = 0.0
    return y


def _adaptive_window(T):
    """Smoothing window size from notebook cell 45."""
    if T < 40:  return 5
    if T < 80:  return 7
    if T < 160: return 9
    if T < 300: return 13
    return 17


def compute_phase_speed(normalized_points, positions, global_speed, hand_speed):
    """
    Blended phase-speed signal (notebook cell 45).
    centroid_speed = max(lh_centroid_speed, rh_centroid_speed)
    phase_speed = 0.65*robust_01(centroid) + 0.25*robust_01(hand) + 0.10*robust_01(global)
    Then smooth and clip to [0,1].
    """
    T      = positions.shape[0]
    points = normalized_points.reshape(T, 49, 3)
    lh_c   = np.mean(points[:, 7:28, :], axis=1)   # (T, 3) left hand centroid
    rh_c   = np.mean(points[:, 28:49, :], axis=1)  # (T, 3) right hand centroid

    lh_vel = np.zeros_like(lh_c, dtype=np.float32)
    rh_vel = np.zeros_like(rh_c, dtype=np.float32)
    if T >= 2:
        lh_vel[1:] = lh_c[1:] - lh_c[:-1]
        rh_vel[1:] = rh_c[1:] - rh_c[:-1]

    lh_sp  = np.linalg.norm(lh_vel, axis=1).astype(np.float32)
    rh_sp  = np.linalg.norm(rh_vel, axis=1).astype(np.float32)
    centroid_speed = np.maximum(lh_sp, rh_sp)
    centroid_speed[~np.isfinite(centroid_speed)] = 0.0

    centroid_norm = _robust_01(centroid_speed)
    hand_norm     = _robust_01(hand_speed)
    global_norm   = _robust_01(global_speed)

    phase_raw = (0.65 * centroid_norm
               + 0.25 * hand_norm
               + 0.10 * global_norm).astype(np.float32)

    w          = _adaptive_window(T)
    phase_sp   = _smooth_1d(phase_raw, window=w)
    if T >= 160:
        phase_sp = _smooth_1d(phase_sp, window=5)

    phase_sp = np.clip(phase_sp, 0.0, 1.0).astype(np.float32)
    phase_sp[~np.isfinite(phase_sp)] = 0.0
    return phase_sp


# ─────────────────────────────────────────────────────────────
# PHASE PREDICTION POST-PROCESSING (verbatim from notebook cell 53)
# ─────────────────────────────────────────────────────────────

def majority_filter_1d(labels, window=5):
    labels = np.asarray(labels, dtype=np.int64)
    T      = len(labels)
    if T == 0 or window <= 1:
        return labels.copy()
    if window % 2 == 0:
        window += 1
    pad    = window // 2
    padded = np.pad(labels, (pad, pad), mode="edge")
    out    = labels.copy()
    for i in range(T):
        vals        = padded[i: i + window]
        counts      = np.bincount(vals, minlength=4)
        out[i]      = int(np.argmax(counts))
    return out.astype(np.int64)


def _get_segments(mask):
    mask, segs, in_seg, start = np.asarray(mask, dtype=bool), [], False, 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            segs.append((start, i - 1)); in_seg = False
    if in_seg:
        segs.append((start, len(mask) - 1))
    return segs


def _fill_small_gaps(mask, max_gap=2):
    mask = np.asarray(mask, dtype=bool).copy()
    segs = _get_segments(mask)
    for i in range(len(segs) - 1):
        end1, start2 = segs[i][1], segs[i + 1][0]
        gap = start2 - end1 - 1
        if 0 < gap <= max_gap:
            mask[end1 + 1: start2] = True
    return mask


def _remove_small_islands(mask, min_len=2):
    mask = np.asarray(mask, dtype=bool).copy()
    for s, e in _get_segments(mask):
        if (e - s + 1) < min_len:
            mask[s: e + 1] = False
    return mask


def extract_phase_segments(pred_labels, probs, phase_speed):
    """
    Verbatim port of extract_phase_segments from notebook cell 53.
    Returns a dict with active_start, active_end, and per-phase boundaries.
    """
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    probs       = np.asarray(probs,       dtype=np.float32)
    phase_speed = np.asarray(phase_speed, dtype=np.float32)
    T           = len(pred_labels)

    seg = dict(segment_status="unknown", segment_quality_flag="",
               active_start=-1, active_end=-1,
               prep_start=-1,   prep_end=-1,
               stroke_start=-1, stroke_end=-1,
               retract_start=-1, retract_end=-1,
               used_fallback=False)

    if T < 8:
        seg["segment_status"] = "reject"; seg["segment_quality_flag"] = "too_short"
        return seg

    active_mask = pred_labels != 0
    active_mask = _fill_small_gaps(active_mask,    max_gap=max(2, int(0.03 * T)))
    active_mask = _remove_small_islands(active_mask, min_len=max(2, int(0.02 * T)))
    active_segs = _get_segments(active_mask)

    if len(active_segs) == 0:
        seg["used_fallback"] = True
        if np.max(phase_speed) <= 1e-8:
            seg["segment_status"] = "reject"
            seg["segment_quality_flag"] = "no_active_region"
            return seg
        threshold   = 0.30 * float(np.max(phase_speed))
        active_mask = phase_speed >= threshold
        active_mask = _fill_small_gaps(active_mask,    max_gap=max(2, int(0.03 * T)))
        active_mask = _remove_small_islands(active_mask, min_len=max(2, int(0.02 * T)))
        active_segs = _get_segments(active_mask)
        if len(active_segs) == 0:
            seg["segment_status"] = "reject"
            seg["segment_quality_flag"] = "no_active_region"
            return seg

    active_start = int(max(0, active_segs[0][0]))
    active_end   = int(min(T - 1, active_segs[-1][1]))
    active_len   = active_end - active_start + 1

    if active_len < 8:
        seg["segment_status"] = "reject"; seg["segment_quality_flag"] = "active_too_short"
        return seg

    stroke_mask = np.zeros(T, dtype=bool)
    stroke_mask[active_start: active_end + 1] = (pred_labels[active_start: active_end + 1] == 2)
    stroke_mask = _fill_small_gaps(stroke_mask,    max_gap=max(1, int(0.02 * active_len)))
    stroke_mask = _remove_small_islands(stroke_mask, min_len=max(2, int(0.03 * active_len)))
    stroke_segs = _get_segments(stroke_mask)

    if len(stroke_segs) == 0:
        seg["used_fallback"] = True
        stroke_probs = probs[active_start: active_end + 1, 2]
        peak_local   = int(np.argmax(stroke_probs))
        peak         = active_start + peak_local
        stroke_len   = max(4, int(round(0.20 * active_len)))
        stroke_len   = min(stroke_len, active_len)
        stroke_start = int(max(active_start, min(peak - stroke_len // 2,
                                                  active_end - stroke_len + 1)))
        stroke_end   = stroke_start + stroke_len - 1
    else:
        stroke_start, stroke_end = max(stroke_segs, key=lambda se: se[1] - se[0] + 1)

    stroke_start = int(max(active_start, stroke_start))
    stroke_end   = int(min(active_end,   stroke_end))

    if stroke_start <= active_start and active_len >= 10:
        stroke_start = active_start + 1
    if stroke_end >= active_end and active_len >= 10:
        stroke_end = active_end - 1

    if stroke_start > stroke_end:
        seg["segment_status"] = "reject"; seg["segment_quality_flag"] = "invalid_stroke_bounds"
        return seg

    prep_start    = active_start
    prep_end      = stroke_start - 1
    retract_start = stroke_end + 1
    retract_end   = active_end

    seg.update(active_start=active_start,  active_end=active_end,
               prep_start=prep_start,      prep_end=prep_end,
               stroke_start=stroke_start,  stroke_end=stroke_end,
               retract_start=retract_start, retract_end=retract_end,
               segment_status="ok")
    return seg


def make_ordered_labels(T, seg):
    """Assigns 0=bg, 1=prep, 2=stroke, 3=retract per frame (notebook cell 53)."""
    labels = np.zeros(T, dtype=np.int64)
    if seg["segment_status"] == "reject":
        return labels
    ps, pe = seg["prep_start"],    seg["prep_end"]
    ss, se = seg["stroke_start"],  seg["stroke_end"]
    rs, re = seg["retract_start"], seg["retract_end"]
    if ps >= 0 and pe >= ps:   labels[ps: pe + 1] = 1
    if ss >= 0 and se >= ss:   labels[ss: se + 1] = 2
    if rs >= 0 and re >= rs:   labels[rs: re + 1] = 3
    return labels


# ─────────────────────────────────────────────────────────────
# FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(pose_seq, lh_seq, rh_seq, pose_mask, lh_mask, rh_mask,
                 phase_model, phase_mean, phase_std,
                 rec_model,   rec_mean,   rec_std,
                 device):
    """
    pose_seq:  (T, 7, 3) raw MediaPipe coordinates
    lh_seq:    (T, 21, 3)
    rh_seq:    (T, 21, 3)
    *_mask:    (T,) bool, True = landmark detected that frame

    Returns: probs (100,) float32
    """
    T = pose_seq.shape[0]

    # 1. Robust normalization (training preprocessing)
    norm_pts, positions = robust_normalize_keypoints(
        pose_seq, lh_seq, rh_seq, pose_mask, lh_mask, rh_mask)

    # 2. Motion features
    velocity, acceleration, global_speed, hand_speed = compute_motion_features(
        norm_pts, positions)

    # 3. Phase-speed signal
    phase_sp = compute_phase_speed(norm_pts, positions, global_speed, hand_speed)

    # 4. Build 444-dim feature matrix
    X = np.concatenate([
        positions,                             # (T, 147)
        velocity,                              # (T, 147)
        acceleration,                          # (T, 147)
        global_speed.reshape(-1, 1),           # (T, 1)
        hand_speed.reshape(-1, 1),             # (T, 1)
        phase_sp.reshape(-1, 1),               # (T, 1)
    ], axis=1).astype(np.float32)              # (T, 444)
    X[~np.isfinite(X)] = 0.0

    # 5. PhaseTCN prediction
    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0
    with torch.no_grad():
        xt     = torch.tensor(Xn, dtype=torch.float32).unsqueeze(0).to(device)
        logits = phase_model(xt)                           # (1, T, 4)
        probs4 = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        raw_pred = np.argmax(probs4, axis=-1).astype(np.int64)

    # 6. Smooth + extract active segment
    smooth_pred = majority_filter_1d(raw_pred, window=5)
    seg         = extract_phase_segments(smooth_pred, probs4, phase_sp)

    if seg["segment_status"] == "reject":
        print(f"  [WARN] Phase TCN: segment rejected ({seg['segment_quality_flag']}). "
              "Using full clip as active region.")
        active_start, active_end = 0, T - 1
        # Fall back: label everything as stroke
        ordered = np.full(T, 2, dtype=np.int64)
    else:
        active_start = seg["active_start"]
        active_end   = seg["active_end"]
        ordered      = make_ordered_labels(T, seg)

    # 7. Build recognition features (448-dim)
    X_cont    = X[active_start: active_end + 1].copy()
    phase_crop = ordered[active_start: active_end + 1]
    phase_crop = np.clip(phase_crop, 0, 3).astype(np.int64)

    X_cont = ((X_cont - rec_mean) / rec_std).astype(np.float32)
    X_cont[~np.isfinite(X_cont)] = 0.0

    phase_onehot = np.eye(4, dtype=np.float32)[phase_crop]
    X_final      = np.concatenate([X_cont, phase_onehot], axis=1).astype(np.float32)

    # 8. Recognition TCN
    T_active = X_final.shape[0]
    xt  = torch.tensor(X_final, dtype=torch.float32).unsqueeze(0).to(device)
    msk = torch.ones(1, T_active, dtype=torch.bool).to(device)
    with torch.no_grad():
        logits_rec, _ = rec_model(xt, msk)
        probs_rec      = torch.softmax(logits_rec, dim=-1).squeeze(0).cpu().numpy()

    print(f"  Phase seg: {seg['segment_status']} ({seg['segment_quality_flag']}) "
          f"active=[{active_start},{active_end}] T_active={T_active}")
    return probs_rec


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
    Returns pose (7,3), lh (21,3), rh (21,3) and detection bools —
    identical format to the training extract_keypoints function.
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


# ─────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────

def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    import csv

    for p in [PHASE_CKPT_PATH, REC_CKPT_PATH, LABEL_MAP_PATH]:
        if not p.exists():
            sys.exit(f"Missing file: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load label map ───────────────────────────────────────
    id_to_gloss = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_gloss[int(row["rec_label_id"])] = row["gloss"]
    print(f"Label map: {len(id_to_gloss)} classes")

    # ── Load Phase TCN checkpoint ────────────────────────────
    print("\nLoading Phase TCN checkpoint...")
    phase_ckpt = load_checkpoint(PHASE_CKPT_PATH, device)
    pcfg       = phase_ckpt["config"]
    phase_mean = phase_ckpt["feature_mean"].detach().cpu().numpy().astype(np.float32) \
                 if isinstance(phase_ckpt["feature_mean"], torch.Tensor) \
                 else np.asarray(phase_ckpt["feature_mean"], dtype=np.float32)
    phase_std  = phase_ckpt["feature_std"].detach().cpu().numpy().astype(np.float32) \
                 if isinstance(phase_ckpt["feature_std"], torch.Tensor) \
                 else np.asarray(phase_ckpt["feature_std"], dtype=np.float32)

    phase_model = PhaseTCN(**{k: pcfg[k] for k in
                              ["input_dim","num_classes","hidden_dim","num_blocks","kernel_size","dropout"]})
    phase_model.load_state_dict(phase_ckpt["model_state_dict"])
    phase_model.to(device).eval()

    # Verify weight shape
    w = phase_ckpt["model_state_dict"]["tcn.0.conv1.weight"]
    assert w.shape == (pcfg["hidden_dim"], pcfg["input_dim"], pcfg["kernel_size"]), \
        f"Phase TCN weight shape mismatch: {w.shape}"
    print(f"  Phase TCN: input_dim={pcfg['input_dim']}, hidden={pcfg['hidden_dim']}, "
          f"blocks={pcfg['num_blocks']} — OK")

    # ── Load Recognition TCN checkpoint ─────────────────────
    print("\nLoading Recognition TCN checkpoint...")
    rec_ckpt  = load_checkpoint(REC_CKPT_PATH, device)
    rcfg      = rec_ckpt["config"]
    rec_mean  = rec_ckpt["feature_mean"].detach().cpu().numpy().astype(np.float32) \
                if isinstance(rec_ckpt["feature_mean"], torch.Tensor) \
                else np.asarray(rec_ckpt["feature_mean"], dtype=np.float32)
    rec_std   = rec_ckpt["feature_std"].detach().cpu().numpy().astype(np.float32) \
                if isinstance(rec_ckpt["feature_std"], torch.Tensor) \
                else np.asarray(rec_ckpt["feature_std"], dtype=np.float32)

    rec_model = RecognitionTCNAttentionSafe(
        input_dim=rcfg["input_dim"], num_classes=rcfg["num_classes"],
        hidden_dim=rcfg["hidden_dim"], num_blocks=rcfg["num_blocks"],
        kernel_size=rcfg["kernel_size"], dropout=rcfg["dropout"],
        attention_dropout=rcfg["attention_dropout"])
    rec_model.load_state_dict(rec_ckpt["model_state_dict"])
    rec_model.to(device).eval()

    w2 = rec_ckpt["model_state_dict"]["tcn.0.conv1.weight"]
    assert w2.shape == (rcfg["hidden_dim"], rcfg["input_dim"], rcfg["kernel_size"]), \
        f"Rec TCN weight shape mismatch: {w2.shape}"
    print(f"  Rec  TCN: input_dim={rcfg['input_dim']}, hidden={rcfg['hidden_dim']}, "
          f"blocks={rcfg['num_blocks']}, classes={rcfg['num_classes']} — OK")

    # ── Open webcam + MediaPipe (Tasks API) ─────────────────
    print("\nInitialising MediaPipe landmarkers...")
    hand_lmk, pose_lmk = make_landmarkers()
    print("  Landmarkers ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "=" * 60)
    print("VERIFICATION: press R to record 2 seconds, Q to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "Press R to record  |  Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ASL Verification", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("r"):
            # ── Countdown ────────────────────────────────────
            for cnt in [3, 2, 1]:
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    ret2, frm = cap.read()
                    if not ret2:
                        break
                    cv2.putText(frm, f"Recording in {cnt}...",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("ASL Verification", frm)
                    cv2.waitKey(1)

            # ── Record N_FRAMES ───────────────────────────────
            pose_buf = np.zeros((N_FRAMES, N_POSE, 3), dtype=np.float32)
            lh_buf   = np.zeros((N_FRAMES, N_HAND, 3), dtype=np.float32)
            rh_buf   = np.zeros((N_FRAMES, N_HAND, 3), dtype=np.float32)
            pose_msk = np.zeros(N_FRAMES, dtype=bool)
            lh_msk   = np.zeros(N_FRAMES, dtype=bool)
            rh_msk   = np.zeros(N_FRAMES, dtype=bool)

            print(f"\nRecording {N_FRAMES} frames...")
            t_start = time.time()
            for fi in range(N_FRAMES):
                ret3, frm = cap.read()
                if not ret3:
                    break
                rgb  = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                pa, la, ra, po, lo, ro = extract_frame_landmarks(rgb, hand_lmk, pose_lmk)
                pose_buf[fi] = pa
                lh_buf[fi]   = la
                rh_buf[fi]   = ra
                pose_msk[fi] = po
                lh_msk[fi]   = lo
                rh_msk[fi]   = ro

                elapsed = time.time() - t_start
                bar = int(20 * fi / N_FRAMES)
                cv2.putText(frm,
                            f"REC [{elapsed:.1f}s] " + "#" * bar + " " * (20 - bar),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("ASL Verification", frm)
                cv2.waitKey(1)

            print(f"  Recorded in {time.time()-t_start:.2f}s")
            print(f"  Pose detected: {pose_msk.sum()}/{N_FRAMES} frames")
            print(f"  Left hand:     {lh_msk.sum()}/{N_FRAMES} frames")
            print(f"  Right hand:    {rh_msk.sum()}/{N_FRAMES} frames")

            # ── Run pipeline ─────────────────────────────────
            print("\nRunning pipeline...")
            probs = run_pipeline(
                pose_buf, lh_buf, rh_buf, pose_msk, lh_msk, rh_msk,
                phase_model, phase_mean, phase_std,
                rec_model,   rec_mean,   rec_std,
                device,
            )

            # ── Show top-5 ───────────────────────────────────
            top5_idx = np.argsort(probs)[::-1][:5]
            print("\n  TOP-5 PREDICTIONS:")
            print("  " + "-" * 35)
            for rank, idx in enumerate(top5_idx, 1):
                gloss = id_to_gloss.get(idx, f"ID_{idx}")
                print(f"  #{rank}  {gloss:<25}  {probs[idx]*100:.1f}%")
            print("  " + "-" * 35)

            # ── Overlay on frame ─────────────────────────────
            ret4, frm = cap.read()
            if ret4:
                overlay = frm.copy()
                cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frm, 0.5, 0, frm)
                for rank, idx in enumerate(top5_idx, 1):
                    gloss = id_to_gloss.get(idx, f"ID_{idx}")
                    txt   = f"#{rank} {gloss}  {probs[idx]*100:.1f}%"
                    cv2.putText(frm, txt, (10, 30 + (rank - 1) * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frm, "Press R again or Q to quit",
                            (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow("ASL Verification", frm)
                cv2.waitKey(3000)

    cap.release()
    hand_lmk.close()
    pose_lmk.close()
    cv2.destroyAllWindows()
    print("\nVerification session ended.")


if __name__ == "__main__":
    main()
