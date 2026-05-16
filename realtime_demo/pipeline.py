"""
pipeline.py — Shared inference pipeline for Phase-Aware ASL Recognition.

Imported by both demo.py and verify_pipeline.py.
"""

import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
except ImportError:
    sys.exit("mediapipe not installed.  Run:  pip install mediapipe")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

POSE_IDXS      = [0, 11, 12, 13, 14, 15, 16]
N_POSE         = 7
N_HAND         = 21
RECORD_SECONDS = 2.0
TARGET_FPS     = 30
N_FRAMES       = int(RECORD_SECONDS * TARGET_FPS)   # 60

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
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
# PREPROCESSING
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
        if   good.sum() == 0: flat[:, j] = 0.0
        elif good.sum() == 1: flat[:, j] = col[good][0]
        else:                 flat[:, j] = np.interp(x, x[good], col[good])
    out = flat.reshape(arr.shape).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def robust_normalize_keypoints(pose, lh, rh,
                                pose_mask=None, lh_mask=None, rh_mask=None):
    """Center on mean of 7 pose points and scale by pose spread (matches training)."""
    pose, lh, rh = _safe_float(pose), _safe_float(lh), _safe_float(rh)
    T = pose.shape[0]

    pose_i = _interpolate_missing(pose, _ensure_mask(pose_mask, T))
    lh_i   = _interpolate_missing(lh,   _ensure_mask(lh_mask,   T))
    rh_i   = _interpolate_missing(rh,   _ensure_mask(rh_mask,   T))

    pose_xy   = pose_i[:, :, :2]
    center_xy = np.mean(pose_xy, axis=1, keepdims=True)

    spread = np.max(pose_xy, axis=1) - np.min(pose_xy, axis=1)
    fs     = np.maximum(spread[:, 0], spread[:, 1])
    fs     = fs[np.isfinite(fs) & (fs > 1e-6)]
    scale  = float(np.median(fs)) if len(fs) > 0 else 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    def _norm(p):
        o = p.copy().astype(np.float32)
        o[:, :, :2] = (o[:, :, :2] - center_xy) / scale
        o[:, :,  2] =  o[:, :,  2]               / scale
        o[~np.isfinite(o)] = 0.0
        return o

    pts = np.concatenate([_norm(pose_i), _norm(lh_i), _norm(rh_i)], axis=1).astype(np.float32)
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

    hpts = pts[:, 7:, :]
    hv   = np.zeros_like(hpts, dtype=np.float32)
    if T >= 2: hv[1:] = hpts[1:] - hpts[:-1]
    hs   = np.sqrt(np.mean(hv ** 2, axis=(1, 2))).astype(np.float32)

    for a in [vel, acc, gs, hs]:
        a[~np.isfinite(a)] = 0.0
    return vel, acc, gs, hs


def _robust_01(x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    if len(x) == 0:
        return x
    p5, p95 = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    d = p95 - p5
    if not np.isfinite(d) or d < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p5) / d, 0.0, 1.0).astype(np.float32)


def _smooth_1d(x, window=7):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or window <= 1 or len(x) < 3:
        return x.copy()
    window = int(window)
    if window % 2 == 0: window += 1
    window = min(window, len(x))
    if window % 2 == 0: window -= 1
    if window < 3: return x.copy()
    pad = window // 2
    y   = np.convolve(np.pad(x, (pad, pad), "edge"),
                      np.ones(window, np.float32) / window, "valid").astype(np.float32)
    y[~np.isfinite(y)] = 0.0
    return y


def _adaptive_window(T):
    if T < 40:  return 5
    if T < 80:  return 7
    if T < 160: return 9
    if T < 300: return 13
    return 17


def compute_phase_speed(pts, pos, gs, hs):
    """Blended phase-speed signal: 0.65*centroid + 0.25*hand + 0.10*global."""
    T   = pos.shape[0]
    p   = pts.reshape(T, 49, 3)
    lhc = np.mean(p[:, 7:28,  :], axis=1)
    rhc = np.mean(p[:, 28:49, :], axis=1)

    lhv = np.zeros_like(lhc, dtype=np.float32)
    rhv = np.zeros_like(rhc, dtype=np.float32)
    if T >= 2:
        lhv[1:] = lhc[1:] - lhc[:-1]
        rhv[1:] = rhc[1:] - rhc[:-1]

    cs = np.maximum(np.linalg.norm(lhv, axis=1),
                    np.linalg.norm(rhv, axis=1)).astype(np.float32)
    cs[~np.isfinite(cs)] = 0.0

    raw = (0.65 * _robust_01(cs) + 0.25 * _robust_01(hs) + 0.10 * _robust_01(gs)).astype(np.float32)
    sp  = _smooth_1d(raw, _adaptive_window(T))
    if T >= 160:
        sp = _smooth_1d(sp, 5)
    return np.clip(sp, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PHASE POST-PROCESSING
# ─────────────────────────────────────────────────────────────

def majority_filter_1d(labels, window=5):
    labels = np.asarray(labels, dtype=np.int64)
    T      = len(labels)
    if T == 0 or window <= 1:
        return labels.copy()
    if window % 2 == 0: window += 1
    pad    = window // 2
    padded = np.pad(labels, (pad, pad), "edge")
    out    = labels.copy()
    for i in range(T):
        out[i] = int(np.argmax(np.bincount(padded[i: i + window], minlength=4)))
    return out.astype(np.int64)


def _get_segments(mask):
    mask, segs, s = np.asarray(mask, dtype=bool), [], None
    for i, v in enumerate(mask):
        if v     and s is None: s = i
        elif not v and s is not None: segs.append((s, i - 1)); s = None
    if s is not None: segs.append((s, len(mask) - 1))
    return segs


def _fill_small_gaps(mask, max_gap=2):
    mask = np.asarray(mask, dtype=bool).copy()
    ss   = _get_segments(mask)
    for i in range(len(ss) - 1):
        e, s2 = ss[i][1], ss[i + 1][0]
        if 0 < s2 - e - 1 <= max_gap:
            mask[e + 1: s2] = True
    return mask


def _remove_small_islands(mask, min_len=2):
    mask = np.asarray(mask, dtype=bool).copy()
    for s, e in _get_segments(mask):
        if (e - s + 1) < min_len:
            mask[s: e + 1] = False
    return mask


def extract_phase_segments(pred_labels, probs, phase_speed):
    """
    Extracts active region and per-phase boundaries from phase TCN output.
    Returns a dict with segment_status, active_start/end, and phase boundaries.
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
        seg.update(segment_status="reject", segment_quality_flag="too_short")
        return seg

    active_mask = _fill_small_gaps(
        _remove_small_islands(pred_labels != 0, max(2, int(0.02 * T))),
        max(2, int(0.03 * T)))
    active_segs = _get_segments(active_mask)

    if not active_segs:
        seg["used_fallback"] = True
        if np.max(phase_speed) <= 1e-8:
            seg.update(segment_status="reject", segment_quality_flag="no_active_region")
            return seg
        thr         = 0.30 * float(np.max(phase_speed))
        active_mask = _fill_small_gaps(
            _remove_small_islands(phase_speed >= thr, max(2, int(0.02 * T))),
            max(2, int(0.03 * T)))
        active_segs = _get_segments(active_mask)
        if not active_segs:
            seg.update(segment_status="reject", segment_quality_flag="no_active_region")
            return seg

    active_start = int(max(0, active_segs[0][0]))
    active_end   = int(min(T - 1, active_segs[-1][1]))
    active_len   = active_end - active_start + 1

    if active_len < 8:
        seg.update(segment_status="reject", segment_quality_flag="active_too_short")
        return seg

    stroke_mask = np.zeros(T, dtype=bool)
    stroke_mask[active_start: active_end + 1] = (pred_labels[active_start: active_end + 1] == 2)
    stroke_mask = _fill_small_gaps(
        _remove_small_islands(stroke_mask, max(2, int(0.03 * active_len))),
        max(1, int(0.02 * active_len)))
    stroke_segs = _get_segments(stroke_mask)

    if not stroke_segs:
        seg["used_fallback"] = True
        peak         = active_start + int(np.argmax(probs[active_start: active_end + 1, 2]))
        stroke_len   = min(max(4, int(round(0.20 * active_len))), active_len)
        stroke_start = int(max(active_start,
                               min(peak - stroke_len // 2, active_end - stroke_len + 1)))
        stroke_end   = stroke_start + stroke_len - 1
    else:
        stroke_start, stroke_end = max(stroke_segs, key=lambda se: se[1] - se[0] + 1)

    stroke_start = int(max(active_start, stroke_start))
    stroke_end   = int(min(active_end,   stroke_end))

    if stroke_start <= active_start and active_len >= 10: stroke_start = active_start + 1
    if stroke_end   >= active_end   and active_len >= 10: stroke_end   = active_end   - 1

    if stroke_start > stroke_end:
        seg.update(segment_status="reject", segment_quality_flag="invalid_stroke_bounds")
        return seg

    seg.update(active_start=active_start,    active_end=active_end,
               prep_start=active_start,      prep_end=stroke_start - 1,
               stroke_start=stroke_start,    stroke_end=stroke_end,
               retract_start=stroke_end + 1, retract_end=active_end,
               segment_status="ok")
    return seg


def make_ordered_labels(T, seg):
    """Assigns 0=bg, 1=prep, 2=stroke, 3=retract per frame."""
    labels = np.zeros(T, dtype=np.int64)
    if seg["segment_status"] == "reject":
        return labels
    ps, pe = seg["prep_start"],    seg["prep_end"]
    ss, se = seg["stroke_start"],  seg["stroke_end"]
    rs, re = seg["retract_start"], seg["retract_end"]
    if ps >= 0 and pe >= ps: labels[ps: pe + 1] = 1
    if ss >= 0 and se >= ss: labels[ss: se + 1] = 2
    if rs >= 0 and re >= rs: labels[rs: re + 1] = 3
    return labels


# ─────────────────────────────────────────────────────────────
# FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(pose_seq, lh_seq, rh_seq, pose_mask, lh_mask, rh_mask,
                 phase_model, phase_mean, phase_std,
                 rec_model,   rec_mean,   rec_std,
                 device):
    """
    Returns: probs (100,) float32, seg_status str, active_start int, active_end int
    """
    T = pose_seq.shape[0]

    pts, pos = robust_normalize_keypoints(pose_seq, lh_seq, rh_seq, pose_mask, lh_mask, rh_mask)
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)

    X = np.concatenate([pos, vel, acc,
                        gs.reshape(-1, 1), hs.reshape(-1, 1), psp.reshape(-1, 1)],
                       axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0

    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0

    with torch.no_grad():
        xt     = torch.tensor(Xn, dtype=torch.float32).unsqueeze(0).to(device)
        probs4 = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()

    smooth_pred = majority_filter_1d(np.argmax(probs4, axis=-1).astype(np.int64))
    seg         = extract_phase_segments(smooth_pred, probs4, psp)

    if seg["segment_status"] == "reject":
        active_start, active_end = 0, T - 1
        ordered = np.full(T, 2, dtype=np.int64)
    else:
        active_start = seg["active_start"]
        active_end   = seg["active_end"]
        ordered      = make_ordered_labels(T, seg)

    flag       = seg["segment_quality_flag"]
    seg_status = f"{seg['segment_status']}:{flag}" if flag else seg["segment_status"]

    X_crop  = ((X[active_start: active_end + 1] - rec_mean) / rec_std).astype(np.float32)
    X_crop[~np.isfinite(X_crop)] = 0.0
    ph_crop = np.clip(ordered[active_start: active_end + 1], 0, 3).astype(np.int64)
    X_final = np.concatenate([X_crop, np.eye(4, dtype=np.float32)[ph_crop]], axis=1)

    T_act = X_final.shape[0]
    xt2   = torch.tensor(X_final, dtype=torch.float32).unsqueeze(0).to(device)
    msk   = torch.ones(1, T_act, dtype=torch.bool).to(device)
    with torch.no_grad():
        logits, _ = rec_model(xt2, msk)
        probs100  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return probs100, seg_status, active_start, active_end


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK EXTRACTION
# ─────────────────────────────────────────────────────────────

def _download_if_needed(url, path):
    if not path.exists():
        print(f"  Downloading {path.name} …")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved {path.name} ({path.stat().st_size // 1024} KB)")


def make_landmarkers(hand_model_path, pose_model_path):
    """Downloads assets if needed and returns (hand_lmk, pose_lmk) in IMAGE mode."""
    _download_if_needed(HAND_MODEL_URL, hand_model_path)
    _download_if_needed(POSE_MODEL_URL, pose_model_path)

    hand_opts = _mp_vision.HandLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    pose_opts = _mp_vision.PoseLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=str(pose_model_path)),
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
            label  = handed[0].category_name
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


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)
