"""
colab_demo.py — Self-contained German SL demo for Google Colab T4
==================================================================
Paste these two cells in Colab:

CELL 1 (once per session):
    !git clone https://github.com/H29-crypto/CAPSTONE_ASL.git
    %cd CAPSTONE_ASL
    !pip install -q gradio opencv-python-headless

CELL 2 (run the demo):
    from google.colab import drive
    drive.mount('/content/drive')
    exec(open('/content/CAPSTONE_ASL/corrnet/colab_demo.py').read())
    run_demo(
        ckpt='/content/drive/MyDrive/Capstonnnn/corrnet_phoenix2014T.pt',
    )
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from collections import OrderedDict

# ── Fix paths so corrnet modules are importable ───────────────────────────────
_CORRNET = Path(__file__).resolve().parent if '__file__' in dir() else Path('/content/CAPSTONE_ASL/corrnet')
_ROOT    = _CORRNET.parent

if str(_CORRNET) not in sys.path:
    sys.path.insert(0, str(_CORRNET))

# ── Mock ctcdecode (beam search not required) ─────────────────────────────────
try:
    import ctcdecode as _cd
except Exception:
    import unittest.mock as _um
    sys.modules['ctcdecode'] = _um.MagicMock()

import numpy as np
import torch
import cv2

# ── Preprocessing constants (match Phoenix-2014-T training) ──────────────────
_KERNEL_SEQ = ['K5','P2','K5','P2']
LEFT_PAD, TOTAL_STRIDE, _stride = 0, 1, 1
for _ks in _KERNEL_SEQ:
    if _ks[0]=='K': LEFT_PAD = LEFT_PAD*_stride + int((int(_ks[1])-1)/2)
    else:            _stride=int(_ks[1]); TOTAL_STRIDE*=_stride
SUBSAMPLE = 3   # keep every 3rd frame for speed

# ── Gloss-to-German translation ───────────────────────────────────────────────
import importlib.util as _ilu
_s = _ilu.spec_from_file_location('g2t', _CORRNET / 'gloss_to_text.py')
_m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
_g2g = _m.glosses_to_german

def _cv2safe(t):
    return t.replace('ä','ae').replace('ö','oe').replace('ü','ue')\
             .replace('Ä','Ae').replace('Ö','Oe').replace('Ü','Ue').replace('ß','ss')

# ── CTC greedy decoder ────────────────────────────────────────────────────────
def _greedy_decode(logits, length, i2g):
    sm   = torch.softmax(logits[:length], -1)
    idxs = logits[:length].argmax(-1).tolist()
    seq, prev, rs, rn = [], None, 0.0, 0
    for pos, idx in enumerate(idxs):
        if idx != prev:
            if prev and seq:
                g,p,_ = seq[-1]; seq[-1]=(g,p,rs/max(rn,1))
            prev,rs,rn = idx,0.0,0
            if idx: seq.append((i2g.get(idx,f'<{idx}>'), len(seq), 0.0))
        if idx: rs+=float(sm[pos,idx]); rn+=1
    if prev and seq: g,p,_=seq[-1]; seq[-1]=(g,p,rs/max(rn,1))
    return [x[0] for x in seq], [x[2] for x in seq]

# ── Frame preprocessing ───────────────────────────────────────────────────────
def _prep(bgr):
    """BGR frame → normalised (3,224,224) tensor matching Phoenix training."""
    h, w = bgr.shape[:2]
    tw = int(h * 210/260)                      # portrait crop (match Phoenix AR)
    if tw < w:
        x0 = (w-tw)//2; bgr = bgr[:, x0:x0+tw]
    img = cv2.resize(bgr, (256,256), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    s   = (256-224)//2; img = img[s:s+224, s:s+224]
    return torch.from_numpy(img).float().permute(2,0,1)/127.5 - 1.0

def _pad_batch(frames):
    """frames: list of BGR ndarray → (1, T_padded, 3, 224, 224), lengths."""
    T = len(frames)
    T_pad = (((T + TOTAL_STRIDE-1)//TOTAL_STRIDE)*TOTAL_STRIDE
             + 2*LEFT_PAD)
    tensors = [_prep(f) for f in frames]
    vid = torch.zeros(1, T_pad, 3, 224, 224)
    for i,t in enumerate(tensors):
        vid[0, LEFT_PAD+i] = t
    return vid, torch.tensor([T_pad], dtype=torch.long)

# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model(ckpt_path: str, gloss_dict: dict, device):
    from slr_network import SLRModel
    model = SLRModel(
        num_classes  = len(gloss_dict)+1,
        c2d_type     = 'resnet18',
        conv_type    = 2,
        use_bn       = True,
        hidden_size  = 1024,
        gloss_dict   = gloss_dict,
        loss_weights = {'SeqCTC':1.0},
        weight_norm  = True,
        share_classifier = True,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd   = ckpt.get('model_state_dict', ckpt)
    sd   = OrderedDict([(k.replace('.module',''),v) for k,v in sd.items()])
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _infer(model, frames_bgr: list, i2g: dict, device):
    if len(frames_bgr) < 5:
        return [], []
    sampled = frames_bgr[::SUBSAMPLE]
    vid, lgt = _pad_batch(sampled)
    out  = model(vid.to(device), lgt.to(device))
    pred = out.get('recognized_sents')
    if pred and pred[0]:
        g = [x[0] for x in pred[0]]
        c = [x[2] if len(x)>2 else 1.0 for x in pred[0]]
        return g,c
    seq_logits = out.get('sequence_logits')
    feat_len   = out.get('feat_len')
    if seq_logits is not None and feat_len is not None:
        logits = seq_logits[:,0,:]
        return _greedy_decode(logits, int(feat_len[0]), i2g)
    return [], []

# ── Main entry point ──────────────────────────────────────────────────────────
def run_demo(ckpt: str,
             dict_path: str = str(_CORRNET/'preprocess'/'phoenix2014-T'/'gloss_dict.npy')):
    """
    Call this from a Colab cell:
        run_demo(ckpt='/content/drive/MyDrive/.../corrnet_phoenix2014T.pt')
    """
    try:
        import gradio as gr
    except ImportError:
        print("Run:  !pip install gradio"); return

    ckpt = Path(ckpt); dict_path = Path(dict_path)
    assert ckpt.exists(),      f"Checkpoint not found: {ckpt}"
    assert dict_path.exists(), f"Gloss dict not found: {dict_path}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ''))

    gd   = np.load(str(dict_path), allow_pickle=True).item()
    i2g  = {v[0]:k for k,v in gd.items()}
    print(f"Vocab  : {len(gd)} glosses")

    print(f"Loading CorrNet from {ckpt.name} ...")
    model = _load_model(str(ckpt), gd, device)
    print(f"Ready  : {sum(p.numel() for p in model.parameters()):,} params\n")

    def predict(video_path):
        if not video_path:
            return "No video — record a clip first.", "", "—"
        t0  = time.time()
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 100:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()

        if len(frames) < 5:
            return "Too short — sign for at least 1 second.", "", "—"

        glosses, confs = _infer(model, frames, i2g, device)
        elapsed = time.time()-t0

        if not glosses:
            return "No sign detected — try again.", "", f"{elapsed:.2f}s"

        german  = _g2g(glosses, add_period=True) or ' '.join(glosses)
        display = _cv2safe(german)
        raw     = '  |  '.join(f"{g} {c*100:.0f}%" for g,c in zip(glosses,confs))
        timing  = f"{elapsed:.2f}s | {len(frames)} frames → {len(frames[::SUBSAMPLE])} sampled | {device.type.upper()}"
        return display, raw, timing

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Video(sources=['webcam','upload'],
                        label='Record or upload a signing clip (2-5 s)'),
        outputs=[
            gr.Textbox(label='German translation', lines=2),
            gr.Textbox(label='Glosses + confidence', lines=2),
            gr.Textbox(label='Timing', lines=1),
        ],
        title='German Sign Language — CorrNet (Phoenix-2014-T)',
        description=(
            "**Try these DGS weather phrases:**  \n"
            "`HEUTE REGEN KOMMEN` | `MORGEN NORD WIND STARK` | "
            "`MINUS DREI GRAD` | `WOCHENENDE SONNE FREUNDLICH`\n\n"
            "Stand upper-body frontal, plain background, within the portrait zone."
        ),
        flagging_mode='never',
    )
    demo.launch(share=True)
