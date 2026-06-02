"""
mslr_webcam.py — MSLR Real-Time Continuous German SL Demo
----------------------------------------------------------
Full pipeline:
  MediaPipe (real-time) → 444-dim features
  Phase TCN (ASL pipeline, language-independent) → sign boundaries
  MSLR student (<10 ms) → German gloss sequence
  gloss_to_text → readable German sentence

Keys:
  C — clear sentence
  Q — quit
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_DIR    = Path(__file__).resolve().parent.parent
CORRNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRNET_DIR))
sys.path.insert(0, str(BASE_DIR))
# Make realtime_demo importable
sys.path.insert(0, str(BASE_DIR / "realtime_demo"))

from mslr_student import MSLRStudent
from gloss_to_text import glosses_to_german

from pipeline import (
    make_landmarkers, extract_frame_landmarks,
    robust_normalize_keypoints, compute_motion_features, compute_phase_speed,
    PhaseTCN, load_checkpoint, to_numpy, majority_filter_1d,
    N_HAND, N_POSE, TARGET_FPS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
STUDENT_CKPT = CORRNET_DIR / "weights" / "mslr_student.pt"
PHASE_CKPT   = BASE_DIR    / "weights" / "phase_tcn_best_safe_state.pt"
DICT_PATH    = CORRNET_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
HAND_PATH    = BASE_DIR    / "assets" / "hand_landmarker.task"
POSE_PATH    = BASE_DIR    / "assets" / "pose_landmarker_full.task"

# ── State machine constants (same as ASL continuous demo) ─────────────────────
RING_SIZE          = int(5 * TARGET_FPS)
PHASE_INFER_EVERY  = 5
MIN_SIGN_FRAMES    = 20
BG_COOLDOWN        = 18
MAX_SIGN_FRAMES    = RING_SIZE
LOOKBACK_FRAMES    = 10
MAX_WORDS          = 10
PHASE_WINDOW       = 90
CONF_THRESHOLD     = 0.15

# ── Colours ────────────────────────────────────────────────────────────────────
C_GREEN = (0,210,0);  C_RED  = (30,30,220); C_ORG  = (0,140,255)
C_WHITE = (255,255,255); C_GRAY = (140,140,140); C_YELL = (0,220,220)
C_TEAL  = (180,210,0);   C_DARK = (25,25,25);    C_DIM  = (50,50,50)
PHASE_COLORS = {0:(100,100,100), 1:(0,200,220), 2:(0,220,0), 3:(220,80,0)}
PHASE_NAMES  = {0:"Background",  1:"Preparation", 2:"Stroke", 3:"Retraction"}


# ══════════════════════════════════════════════════════════════════════════════
# Ring / Sign buffers  (reproduced from continuous_demo.py)
# ══════════════════════════════════════════════════════════════════════════════

class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.pose_buf = np.zeros((size, N_POSE, 3), np.float32)
        self.lh_buf   = np.zeros((size, N_HAND, 3), np.float32)
        self.rh_buf   = np.zeros((size, N_HAND, 3), np.float32)
        self.pose_msk = np.zeros(size, bool)
        self.lh_msk   = np.zeros(size, bool)
        self.rh_msk   = np.zeros(size, bool)
        self.ptr = 0; self.count = 0

    def push(self, pose, lh, rh, pm, lm, rm):
        i = self.ptr % self.size
        self.pose_buf[i]=pose; self.lh_buf[i]=lh; self.rh_buf[i]=rh
        self.pose_msk[i]=pm;   self.lh_msk[i]=lm;  self.rh_msk[i]=rm
        self.ptr += 1; self.count = min(self.count+1, self.size)

    def get_last_n(self, n):
        n = min(n, self.count)
        if n == 0:
            return (np.zeros((0,N_POSE,3),np.float32),
                    np.zeros((0,N_HAND,3),np.float32),
                    np.zeros((0,N_HAND,3),np.float32),
                    np.zeros(0,bool), np.zeros(0,bool), np.zeros(0,bool))
        end     = self.ptr % self.size
        indices = [(end-n+i) % self.size for i in range(n)]
        return (self.pose_buf[indices], self.lh_buf[indices], self.rh_buf[indices],
                self.pose_msk[indices], self.lh_msk[indices], self.rh_msk[indices])


class SignBuffer:
    def __init__(self):
        self.pose_list=[]; self.lh_list=[]; self.rh_list=[]
        self.pm_list=[];   self.lm_list=[]; self.rm_list=[]

    def __len__(self): return len(self.pose_list)

    def push(self, pose, lh, rh, pm, lm, rm):
        self.pose_list.append(pose.copy()); self.lh_list.append(lh.copy())
        self.rh_list.append(rh.copy());    self.pm_list.append(bool(pm))
        self.lm_list.append(bool(lm));     self.rm_list.append(bool(rm))

    def get_arrays(self, start=0, end=None):
        if end is None or end > len(self): end = len(self)
        if start >= end or end == 0:
            return (np.zeros((0,N_POSE,3),np.float32),
                    np.zeros((0,N_HAND,3),np.float32),
                    np.zeros((0,N_HAND,3),np.float32),
                    np.zeros(0,bool), np.zeros(0,bool), np.zeros(0,bool))
        return (np.stack(self.pose_list[start:end]),
                np.stack(self.lh_list[start:end]),
                np.stack(self.rh_list[start:end]),
                np.array(self.pm_list[start:end], bool),
                np.array(self.lm_list[start:end], bool),
                np.array(self.rm_list[start:end], bool))

    def init_from_ring(self, ring, n):
        pose,lh,rh,pm,lm,rm = ring.get_last_n(n)
        for i in range(len(pose)):
            self.push(pose[i],lh[i],rh[i],pm[i],lm[i],rm[i])

    def clear(self):
        self.pose_list.clear(); self.lh_list.clear(); self.rh_list.clear()
        self.pm_list.clear();   self.lm_list.clear(); self.rm_list.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction & Phase TCN inference
# ══════════════════════════════════════════════════════════════════════════════

def to_features(pose, lh, rh, pm, lm, rm) -> np.ndarray:
    """(T,*,3) arrays → (T, 444) feature matrix."""
    pts, pos = robust_normalize_keypoints(pose, lh, rh, pm, lm, rm)
    vel, acc, gs, hs = compute_motion_features(pts, pos)
    psp = compute_phase_speed(pts, pos, gs, hs)
    X = np.concatenate([
        pos, vel, acc,
        gs.reshape(-1,1), hs.reshape(-1,1), psp.reshape(-1,1),
    ], axis=1).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def infer_phase(pose, lh, rh, pm, lm, rm,
                phase_model, phase_mean, phase_std, device):
    T = pose.shape[0]
    if T < 4: return 0, np.zeros(T, np.int64)
    X = to_features(pose, lh, rh, pm, lm, rm)
    Xn = ((X - phase_mean) / phase_std).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0.0
    with torch.no_grad():
        xt    = torch.tensor(Xn).unsqueeze(0).to(device)
        probs = torch.softmax(phase_model(xt), dim=-1).squeeze(0).cpu().numpy()
    labels = majority_filter_1d(np.argmax(probs, axis=-1).astype(np.int64))
    return int(labels[-1]), labels


# ══════════════════════════════════════════════════════════════════════════════
# MSLR inference
# ══════════════════════════════════════════════════════════════════════════════

def run_mslr(student, sign_buf, last_active_fi, device) -> tuple:
    end = last_active_fi + 1
    if end < MIN_SIGN_FRAMES:
        return [], [], 0.0

    pose, lh, rh, pm, lm, rm = sign_buf.get_arrays(end=end)
    if pose.shape[0] == 0:
        return [], [], 0.0

    X   = to_features(pose, lh, rh, pm, lm, rm)
    feat = torch.from_numpy(X).unsqueeze(0)          # (1, T, 444)
    lgt  = torch.tensor([X.shape[0]], dtype=torch.long)

    t0 = time.time()
    glosses, confs = student.predict(feat.to(device), lgt.to(device))
    ms = (time.time() - t0) * 1000
    return glosses, confs, ms


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

def draw_sentence(frame, sentence, german):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,100),C_DARK,-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.line(frame,(0,100),(w,100),(55,55,55),1)
    cv2.putText(frame,"German SL  |  MSLR  (<10ms CPU)",(10,18),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,C_TEAL,1)
    if not sentence:
        cv2.putText(frame,"Sign freely — auto-detecting ...",(10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,C_GRAY,2)
    else:
        x = 10
        for i,w_ in enumerate(sentence[-MAX_WORDS:]):
            col = C_GREEN if i==len(sentence[-MAX_WORDS:])-1 else C_WHITE
            cv2.putText(frame,w_,(x,55),cv2.FONT_HERSHEY_SIMPLEX,0.80,col,2)
            (tw,_),_ = cv2.getTextSize(w_+"  ",cv2.FONT_HERSHEY_SIMPLEX,0.80,2)
            x += tw
            if x > w-80: break
    if german:
        cv2.putText(frame,f'"{german}"',(10,85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,C_YELL,2)

def draw_phase_strip(frame, phase_history, current_phase):
    h, w = frame.shape[:2]
    y0,y1 = h-72,h-52; sx0,sx1 = 10,w-10; sw = sx1-sx0
    cv2.rectangle(frame,(sx0,y0),(sx1,y1),(40,40,40),-1)
    n = len(phase_history)
    if n:
        for i,ph in enumerate(phase_history):
            x0=sx0+int(i*sw/n); x1=sx0+int((i+1)*sw/n)
            cv2.rectangle(frame,(x0,y0),(x1,y1),
                          PHASE_COLORS.get(int(ph),(100,100,100)),-1)
    cv2.putText(frame,PHASE_NAMES.get(current_phase,"?"),(sx0,y0-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,PHASE_COLORS.get(current_phase,C_GRAY),2)

def draw_landmarks(frame, pose, lh, rh, pm, lm, rm):
    h, w = frame.shape[:2]
    if pm:
        for pt in pose:
            x,y=int(pt[0]*w),int(pt[1]*h)
            if 0<=x<w and 0<=y<h: cv2.circle(frame,(x,y),4,(0,200,0),-1)
    if lm:
        for pt in lh:
            x,y=int(pt[0]*w),int(pt[1]*h)
            if 0<=x<w and 0<=y<h: cv2.circle(frame,(x,y),3,(255,120,0),-1)
    if rm:
        for pt in rh:
            x,y=int(pt[0]*w),int(pt[1]*h)
            if 0<=x<w and 0<=y<h: cv2.circle(frame,(x,y),3,(0,120,255),-1)

def draw_status(frame, state, last_gloss, last_conf, last_t, infer_ms):
    h, w = frame.shape[:2]
    cv2.rectangle(frame,(0,h-40),(w,h),C_DARK,-1)
    now = time.time()
    if state=="sign":
        pulse=int(155+100*np.sin(now*6))
        cv2.circle(frame,(16,h-20),7,(30,30,pulse),-1)
        cv2.putText(frame,"  SIGNING ...",(30,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,C_RED,2)
    elif state=="infer":
        cv2.putText(frame,"  Recognizing ...",(10,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.62,C_ORG,2)
    else:
        cv2.putText(frame,"  Ready — sign freely    C=clear    Q=quit",
                    (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.58,C_GREEN,2)
    if last_gloss and (now-last_t)<4.0:
        a = max(0.0,1.0-(now-last_t)/4.0)
        col = tuple(int(c*a) for c in C_YELL)
        cv2.putText(frame,
                    f">> {last_gloss}  ({last_conf*100:.0f}%)  [{infer_ms:.0f}ms]",
                    (w//2-130,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.56,col,2)

def draw_pred_panel(frame, glosses, confs, ts):
    if not glosses or (time.time()-ts)>=6.0: return
    h, w = frame.shape[:2]
    n,pw = len(glosses),220; ph=22+n*28
    px0,py0 = w-pw-8,105
    ov=frame.copy()
    cv2.rectangle(ov,(px0,py0),(px0+pw,py0+ph),(18,18,18),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(px0,py0),(px0+pw,py0+ph),(70,70,70),1)
    cv2.putText(frame,"MSLR",(px0+6,py0+15),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GRAY,1)
    for i,(g,c) in enumerate(zip(glosses,confs)):
        y=py0+22+i*28; bw=int(c*(pw-14))
        bc=C_GREEN if c>=.55 else C_ORG if c>=.30 else C_GRAY
        cv2.rectangle(frame,(px0+6,y+10),(px0+6+max(bw,2),y+18),bc,-1)
        cv2.putText(frame,f"{g}  {c*100:.0f}%",(px0+6,y+8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,C_WHITE,1)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not STUDENT_CKPT.exists():
        sys.exit(
            f"\n[ERROR] Student weights not found: {STUDENT_CKPT}\n\n"
            "Step 1 — collect training data:\n"
            "  python corrnet/collect_pseudo_labels.py\n\n"
            "Step 2 — train student:\n"
            "  python corrnet/train_mslr.py\n"
        )

    device = torch.device("cpu")
    print(f"\n  Device: {device}")

    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"  Vocabulary: {len(gloss_dict)} German glosses")

    print("  Loading MSLR student ...")
    student = MSLRStudent.load(str(STUDENT_CKPT), gloss_dict)
    student.to(device).eval()
    n_p = sum(p.numel() for p in student.parameters())
    print(f"  Student: {n_p:,} params  ({n_p*4/1024/1024:.1f} MB)")

    print("  Loading Phase TCN (sign boundary detector) ...")
    pc          = load_checkpoint(str(PHASE_CKPT), device)
    phase_mean  = to_numpy(pc["feature_mean"])
    phase_std   = to_numpy(pc["feature_std"])
    phase_model = PhaseTCN(**{k: pc["config"][k] for k in
                               ["input_dim","num_classes","hidden_dim",
                                "num_blocks","kernel_size","dropout"]})
    phase_model.load_state_dict(pc["model_state_dict"])
    phase_model.to(device).eval()
    print(f"  Phase TCN accuracy={pc.get('val_accuracy',0):.4f}")

    print("  Starting MediaPipe ...")
    hand_lmk, pose_lmk = make_landmarkers(str(HAND_PATH), str(POSE_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        hand_lmk.close(); pose_lmk.close()
        sys.exit("[ERROR] Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("\n" + "="*58)
    print("  MSLR Student  •  German SL  •  Real-time")
    print("  Sign any German weather sign — auto-detected")
    print("  C = clear   Q = quit")
    print("="*58+"\n")

    IDLE="idle"; SIGN="sign"; INFER="infer"
    state         = IDLE
    frame_idx     = 0
    current_phase = 0
    bg_count      = 0
    last_active_fi = 0

    ring     = RingBuffer(RING_SIZE)
    sign_buf = SignBuffer()
    phase_strip = deque(maxlen=PHASE_WINDOW)

    sentence    = []
    german_text = ""
    last_gloss  = ""; last_conf = 0.0; last_pred_t = 0.0
    last_glosses = []; last_confs = []
    infer_ms    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose, lh, rh, pm, lm, rm = extract_frame_landmarks(rgb, hand_lmk, pose_lmk)
        draw_landmarks(frame, pose, lh, rh, pm, lm, rm)
        frame_idx += 1

        # ── INFER ──────────────────────────────────────────────────────────
        if state == INFER:
            draw_sentence(frame, sentence, german_text)
            draw_phase_strip(frame, list(phase_strip), current_phase)
            draw_status(frame, INFER, last_gloss, last_conf, last_pred_t, infer_ms)
            cv2.imshow("MSLR — German SL", frame)
            cv2.waitKey(1)

            glosses, confs, infer_ms = run_mslr(student, sign_buf, last_active_fi, device)

            if glosses:
                top_conf = max(confs) if confs else 0.0
                if top_conf >= CONF_THRESHOLD:
                    top_g = glosses[0]
                    if not sentence or sentence[-1] != top_g:
                        sentence.extend(glosses)
                        last_glosses = glosses
                        last_confs   = confs
                        last_gloss   = top_g
                        last_conf    = top_conf
                        last_pred_t  = time.time()
                        german_text  = glosses_to_german(glosses)
                        print(f"  [{infer_ms:.0f}ms] "
                              f"{' '.join(f'{g}({c*100:.0f}%)' for g,c in zip(glosses,confs))}"
                              f"  →  \"{german_text}\"")
                else:
                    print(f"  [{infer_ms:.0f}ms] conf={max(confs):.2f} < threshold")
            else:
                print(f"  [{infer_ms:.0f}ms] <no prediction>")

            sign_buf.clear(); bg_count = 0; last_active_fi = 0
            state = IDLE
            ring.push(pose, lh, rh, pm, lm, rm)
            continue

        # ── Phase inference ────────────────────────────────────────────────
        if frame_idx % PHASE_INFER_EVERY == 0:
            if state == SIGN and len(sign_buf) >= 4:
                n_use  = min(PHASE_WINDOW, len(sign_buf))
                start  = len(sign_buf) - n_use
                ps_,lh_,rh_,pm_,lm_,rm_ = sign_buf.get_arrays(start=start)
                current_phase, ph_l = infer_phase(
                    ps_,lh_,rh_,pm_,lm_,rm_,
                    phase_model, phase_mean, phase_std, device)
                phase_strip.extend(ph_l)
            elif state == IDLE and ring.count >= 4:
                n_use = min(30, ring.count)
                ps_,lh_,rh_,pm_,lm_,rm_ = ring.get_last_n(n_use)
                current_phase, ph_l = infer_phase(
                    ps_,lh_,rh_,pm_,lm_,rm_,
                    phase_model, phase_mean, phase_std, device)
                phase_strip.extend(ph_l)

        # ── IDLE ───────────────────────────────────────────────────────────
        if state == IDLE:
            if current_phase != 0:
                sign_buf.init_from_ring(ring, LOOKBACK_FRAMES)
                sign_buf.push(pose, lh, rh, pm, lm, rm)
                last_active_fi = len(sign_buf) - 1
                bg_count = 0; state = SIGN
            ring.push(pose, lh, rh, pm, lm, rm)

        # ── SIGN ───────────────────────────────────────────────────────────
        elif state == SIGN:
            sign_buf.push(pose, lh, rh, pm, lm, rm)
            ring.push(pose, lh, rh, pm, lm, rm)
            if current_phase != 0:
                last_active_fi = len(sign_buf) - 1; bg_count = 0
            else:
                bg_count += 1
            if bg_count >= BG_COOLDOWN or len(sign_buf) >= MAX_SIGN_FRAMES:
                state = INFER

        # ── Draw ───────────────────────────────────────────────────────────
        draw_sentence(frame, sentence, german_text)
        draw_phase_strip(frame, list(phase_strip), current_phase)
        draw_status(frame, state, last_gloss, last_conf, last_pred_t, infer_ms)
        if last_glosses and (time.time()-last_pred_t) < 6.0:
            draw_pred_panel(frame, last_glosses, last_confs, last_pred_t)

        cv2.imshow("MSLR — German SL", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("c"):
            sentence.clear(); german_text=""
            last_gloss=""; last_pred_t=0.0
            last_glosses=[]; last_confs=[]
            print("  Cleared.")

    cap.release(); hand_lmk.close(); pose_lmk.close()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
