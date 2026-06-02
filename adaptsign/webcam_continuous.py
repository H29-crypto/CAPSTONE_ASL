"""
webcam_continuous.py — AdaptSign Auto-Trigger German Sign Language Demo
------------------------------------------------------------------------
Motion detection triggers recording automatically. When the sign ends the
full clip is sent to AdaptSign (ViT-B/16 + BiLSTM + CTC) in a background
thread so the webcam window stays live. The prediction appears as soon as
inference finishes (~5-10 s on CPU with SUBSAMPLE_STEP=2).

Keys:
  C — clear accumulated sentence
  Q — quit
"""
from __future__ import annotations

import sys
import time
import threading
from collections import OrderedDict, deque
from pathlib import Path

try:
    import ctcdecode
except Exception:
    import types, unittest.mock
    sys.modules["ctcdecode"] = unittest.mock.MagicMock()

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
ADAPTSIGN_DIR = Path(__file__).resolve().parent
CKPT_PATH     = ADAPTSIGN_DIR / "weights" / "phoenix2014-T_best.pt"
DICT_PATH     = ADAPTSIGN_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"
sys.path.insert(0, str(ADAPTSIGN_DIR))

# Load gloss_to_text by absolute path to avoid polluting sys.path with the
# corrnet directory (which has its own modules/ package that would shadow
# adaptsign's modules/ and break the ViT-B/16 import in slr_network.py).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gloss_to_text",
    ADAPTSIGN_DIR.parent / "corrnet" / "gloss_to_text.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
glosses_to_german = _mod.glosses_to_german
del _ilu, _spec, _mod


def _cv2_safe(text: str) -> str:
    """Transliterate German umlauts to ASCII for OpenCV's bitmap font."""
    return (text.replace("ä","ae").replace("Ä","Ae")
                .replace("ö","oe").replace("Ö","Oe")
                .replace("ü","ue").replace("Ü","Ue")
                .replace("ß","ss"))

# ── Inference constants ────────────────────────────────────────────────────────
TARGET_FPS      = 25
SUBSAMPLE_STEP  = 2       # 100 frames → 50 → ~7 s inference, 9 time-steps
MAX_SIGN_FRAMES = 100     # hard cap (4 s at 25 fps)
MIN_SIGN_FRAMES = 20

# ── Motion detection ──────────────────────────────────────────────────────────
MOTION_THRESH  = 6.0
STILL_THRESH   = 3.0
STILL_FRAMES   = 20
MOTION_WARMUP  = 4
PRE_BUF_SIZE   = 12
EMA_ALPHA      = 0.5

# ── UI ─────────────────────────────────────────────────────────────────────────
MAX_WORDS   = 10
FADE_SECS   = 6.0

C_GREEN = (0, 210, 0);   C_RED  = (30,  30, 220); C_ORG  = (0,  140, 255)
C_WHITE = (255,255,255);  C_GRAY = (140,140,140);  C_YELL = (0,  220, 220)
C_TEAL  = (180,210,  0);  C_DARK = (25,  25,  25); C_DIM  = (50,  50,  50)


# ══════════════════════════════════════════════════════════════════════════════
# Greedy CTC decoder  — (gloss, pos, confidence) triples
# ══════════════════════════════════════════════════════════════════════════════

class GreedyDecoder:
    def __init__(self, gloss_dict):
        self.i2g      = {v[0]: k for k, v in gloss_dict.items()}
        self.blank_id = 0

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        results = []
        for b in range(nn_output.shape[0]):
            length  = int(vid_lgt[b].item())
            logits  = nn_output[b, :length]
            sm      = torch.softmax(logits, dim=-1)
            idx_seq = logits.argmax(dim=-1).cpu().tolist()
            seq, prev, rs, rn = [], None, 0.0, 0
            for pos, idx in enumerate(idx_seq):
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
# Model
# ══════════════════════════════════════════════════════════════════════════════

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path, gloss_dict, device):
    from slr_network import SLRModel
    model = SLRModel(
        num_classes=len(gloss_dict)+1, c2d_type="ViT-B/16", conv_type=2,
        use_bn=True, hidden_size=1024, gloss_dict=gloss_dict,
        loss_weights={"SeqCTC": 1.0}, weight_norm=True, share_classifier=True,
    )
    model.decoder = GreedyDecoder(gloss_dict)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd   = OrderedDict([(k.replace(".module",""), v) for k, v in sd.items()])
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print("  Model ready.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing & inference
# ══════════════════════════════════════════════════════════════════════════════

def _prep_frame(rgb):
    h, w = rgb.shape[:2]
    # Phoenix training frames are 210×260 (portrait, W/H≈0.808).
    # Webcam frames are landscape (640×480 or wider). Squashing directly
    # to 256×256 compresses the signer horizontally far outside the training
    # distribution. Crop a portrait strip first to match the training AR.
    target_w = int(h * 210 / 260)
    if target_w < w:
        x0 = (w - target_w) // 2
        rgb = rgb[:, x0:x0 + target_w]
    img = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    s   = (256 - 224) // 2
    img = img[s:s+224, s:s+224]
    t   = torch.from_numpy(img).float().permute(2, 0, 1)
    return t / 127.5 - 1.0

def preprocess_clip(frames, device):
    tensors = [_prep_frame(f) for f in frames]
    imgs    = torch.stack(tensors).unsqueeze(0)     # (1,T,3,224,224)
    lengths = torch.tensor([len(frames)], dtype=torch.long)
    return imgs.to(device), lengths.to(device)

@torch.no_grad()
def run_inference(model, frames, device):
    """Full ViT + temporal model inference on a list of RGB frames."""
    if not frames:
        return [], []
    sampled = frames[::SUBSAMPLE_STEP] if len(frames) > SUBSAMPLE_STEP else frames
    imgs, lengths = preprocess_clip(sampled, device)
    out  = model(imgs, lengths)
    pred = out.get("recognized_sents", [[]])
    if pred and pred[0]:
        glosses = [x[0] for x in pred[0]]
        confs   = [x[2] if len(x) > 2 else 1.0 for x in pred[0]]
        return glosses, confs
    return [], []


# ══════════════════════════════════════════════════════════════════════════════
# Motion detection
# ══════════════════════════════════════════════════════════════════════════════

def motion_score(gray, prev):
    if prev is None: return 0.0
    h, w = gray.shape
    roi  = (slice(int(h*.05),int(h*.92)), slice(int(w*.12),int(w*.88)))
    return float(np.mean(cv2.absdiff(gray[roi], prev[roi])))


# ══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ══════════════════════════════════════════════════════════════════════════════

def ui_crop_guide(frame):
    """Draw the portrait crop zone that matches Phoenix training aspect ratio (210:260)."""
    h, w = frame.shape[:2]
    target_w = int(h * 210 / 260)
    if target_w >= w:
        return  # frame is already portrait — no crop needed
    x0 = (w - target_w) // 2
    x1 = x0 + target_w
    # Dim everything outside the crop zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),  (x0, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, 0), (w,  h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    # Draw border lines
    cv2.line(frame, (x0, 0), (x0, h), C_TEAL, 2)
    cv2.line(frame, (x1, 0), (x1, h), C_TEAL, 2)
    # Label
    cv2.putText(frame, "stand here", (x0 + 6, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_TEAL, 1)


def ui_sentence(frame, sentence):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,78),C_DARK,-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.line(frame,(0,78),(w,78),(55,55,55),1)
    cv2.putText(frame,"German SL  |  AdaptSign  |  Translation",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEAL, 1)
    if not sentence:
        cv2.putText(frame,"Sign freely — auto-detecting ...",
                    (10,56), cv2.FONT_HERSHEY_SIMPLEX, 0.78, C_GRAY, 2)
    else:
        # sentence holds German strings; split into words for word-by-word display
        words = " ".join(sentence).split()[-MAX_WORDS:]
        x = 10
        for i, w_ in enumerate(words):
            col = C_GREEN if i == len(words)-1 else C_WHITE
            cv2.putText(frame, w_, (x, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.88, col, 2)
            (tw,_),_ = cv2.getTextSize(w_+"  ", cv2.FONT_HERSHEY_SIMPLEX, 0.88, 2)
            x += tw
            if x > frame.shape[1]-90: break

def ui_motion(frame, m, state):
    h, w = frame.shape[:2]
    y0,y1 = h-74, h-58
    bw    = w-20
    f     = int(min(m/25.0,1.0)*bw)
    cv2.rectangle(frame,(10,y0),(10+bw,y1),C_DIM,-1)
    if f>0:
        col = C_RED if state=="sign" else C_GREEN if m<MOTION_THRESH else C_ORG
        cv2.rectangle(frame,(10,y0),(10+f,y1),col,-1)
    tx = 10+int((MOTION_THRESH/25.0)*bw)
    cv2.line(frame,(tx,y0-3),(tx,y1+3),C_YELL,2)
    cv2.putText(frame,f"motion  {m:.1f}",(10,y0-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GRAY,1)

def ui_status(frame, state, n_frames, last_pred, last_t, infer_t):
    h, w = frame.shape[:2]
    cv2.rectangle(frame,(0,h-56),(w,h),C_DARK,-1)
    now = time.time()

    if state == "sign":
        pct   = min(n_frames/MAX_SIGN_FRAMES, 1.0)
        bar_w = int(pct*(w-20))
        cv2.rectangle(frame,(10,h-8),(10+(w-20),h-3),(50,50,50),-1)
        if bar_w>0: cv2.rectangle(frame,(10,h-8),(10+bar_w,h-3),C_RED,-1)
        pulse = int(155+100*np.sin(now*6))
        cv2.circle(frame,(18,h-34),8,(30,30,pulse),-1)
        cv2.putText(frame,f"  SIGNING  ({n_frames}/{MAX_SIGN_FRAMES} frames)",
                    (32,h-28),cv2.FONT_HERSHEY_SIMPLEX,0.60,C_RED,2)

    elif state == "infer":
        elapsed = now - infer_t
        dots    = "."*(int(elapsed*2.5)%4)
        cv2.putText(frame,f"  Analyzing{dots}  ({elapsed:.0f}s)",
                    (10,h-28),cv2.FONT_HERSHEY_SIMPLEX,0.62,C_ORG,2)

    else:
        cv2.putText(frame,"  Ready — sign freely    C=clear    Q=quit",
                    (10,h-28),cv2.FONT_HERSHEY_SIMPLEX,0.58,C_GREEN,2)

    if last_pred and (now-last_t)<FADE_SECS:
        a   = max(0.0, 1.0-(now-last_t)/FADE_SECS)
        col = tuple(int(c*a) for c in C_GREEN)
        cv2.putText(frame,f">> {last_pred}",(10,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,col,2)

def ui_pred_panel(frame, glosses, confs, ts):
    if not glosses or (time.time()-ts)>=FADE_SECS*1.5: return
    h, w  = frame.shape[:2]
    n,pw  = len(glosses), 220
    ph    = 22+n*28
    px0,py0 = w-pw-8, 85
    ov = frame.copy()
    cv2.rectangle(ov,(px0,py0),(px0+pw,py0+ph),(18,18,18),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(px0,py0),(px0+pw,py0+ph),(70,70,70),1)
    cv2.putText(frame,"Prediction",(px0+6,py0+15),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GRAY,1)
    for i,(g,c) in enumerate(zip(glosses,confs)):
        y  = py0+22+i*28
        bw = int(c*(pw-14))
        bc = C_GREEN if c>=.55 else C_ORG if c>=.30 else C_GRAY
        cv2.rectangle(frame,(px0+6,y+10),(px0+6+max(bw,2),y+18),bc,-1)
        cv2.putText(frame,f"{g}  {c*100:.0f}%",(px0+6,y+8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,C_WHITE,1)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not CKPT_PATH.exists(): sys.exit(f"\n[ERROR] Missing: {CKPT_PATH}\n")
    if not DICT_PATH.exists(): sys.exit(f"\n[ERROR] Missing: {DICT_PATH}\n")

    device = _device()
    print(f"\n  Device: {device}")
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"  Vocabulary: {len(gloss_dict)} glosses")
    model = load_model(CKPT_PATH, gloss_dict, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("[ERROR] Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print(f"\n  SUBSAMPLE_STEP={SUBSAMPLE_STEP} → ~{MAX_SIGN_FRAMES//SUBSAMPLE_STEP} frames per inference")
    print("="*58)
    print("  AdaptSign  •  German SL  •  Auto-trigger")
    print("  Sign freely — result appears automatically")
    print("  C = clear   Q = quit")
    print("="*58+"\n")

    IDLE = "idle"; SIGN = "sign"; INFER = "infer"

    state    = IDLE
    sentence = []
    last_pred = ""; last_t = 0.0; last_g = []; last_c = []

    prev_gray = None
    ema: float = 0.0; m_disp = 0.0
    warmup_cnt = 0; still_cnt = 0
    infer_t = 0.0

    sign_buf: list = []
    pre_buf: deque = deque(maxlen=PRE_BUF_SIZE)

    infer_result = None; infer_error = None
    _result_lock = threading.Lock()

    def _infer_thread(frames):
        nonlocal infer_result, infer_error
        try:
            result = run_inference(model, frames, device)
            with _result_lock:
                infer_result = result
        except Exception as e:
            with _result_lock:
                infer_error = str(e)

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(15,15),0)

        raw   = motion_score(gray, prev_gray)
        ema   = EMA_ALPHA*ema + (1-EMA_ALPHA)*raw
        prev_gray = gray
        m_disp = 0.7*m_disp + 0.3*ema

        pre_buf.append(rgb.copy())

        # ── Collect inference result ───────────────────────────
        with _result_lock:
            r, e = infer_result, infer_error
            infer_result = infer_error = None
        if r is not None:
            g, c = r
            if g:
                last_g, last_c = g, c
                german         = glosses_to_german(g, add_period=True)
                display        = _cv2_safe(german) if german else " ".join(g)
                last_pred      = display
                last_t         = time.time()
                sentence.append(display)
                print(f"  [RESULT] {' '.join(f'{x}({y*100:.0f}%)' for x,y in zip(g,c))}")
                if german:
                    print(f"  [DE]     {german}")
            else:
                print("  [RESULT] <no prediction>")
            state = IDLE
        if e:
            print(f"  [ERROR] {e}"); state = IDLE

        # ── IDLE ──────────────────────────────────────────────
        if state == IDLE:
            if ema > MOTION_THRESH:
                warmup_cnt += 1
                if warmup_cnt >= MOTION_WARMUP:
                    sign_buf   = list(pre_buf)
                    still_cnt  = 0; warmup_cnt = 0
                    state      = SIGN
                    print(f"  Sign detected (motion={ema:.1f}) ...")
            else:
                warmup_cnt = 0

        # ── SIGN ──────────────────────────────────────────────
        elif state == SIGN:
            sign_buf.append(rgb.copy())
            still_cnt = still_cnt+1 if ema < STILL_THRESH else 0
            force     = len(sign_buf) >= MAX_SIGN_FRAMES

            if still_cnt >= STILL_FRAMES or force:
                trim   = max(0, len(sign_buf)-still_cnt)
                frames = sign_buf[:trim] if trim >= MIN_SIGN_FRAMES else sign_buf
                sign_buf = []

                if len(frames) >= MIN_SIGN_FRAMES:
                    sampled_n = len(frames[::SUBSAMPLE_STEP])
                    print(f"  Sign ended ({len(frames)} frames → {sampled_n} after subsampling) → inferring ...")
                    infer_result = None; infer_error = None
                    infer_t = time.time()
                    threading.Thread(target=_infer_thread,
                                     args=(list(frames),), daemon=True).start()
                    state = INFER
                else:
                    print(f"  Too short ({len(frames)} frames) — skipped.")
                    state = IDLE

        # ── Draw ──────────────────────────────────────────────
        ui_crop_guide(frame)
        ui_sentence(frame, sentence)
        ui_motion(frame, m_disp, state)
        ui_status(frame, state, len(sign_buf), last_pred, last_t, infer_t)
        ui_pred_panel(frame, last_g, last_c, last_t)

        cv2.imshow("AdaptSign — German SL", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("c"):
            sentence.clear(); last_pred=""; last_t=0.0
            last_g=[]; last_c=[]; print("  Sentence cleared.")

    cap.release(); cv2.destroyAllWindows(); print("Demo closed.")


if __name__ == "__main__":
    main()
