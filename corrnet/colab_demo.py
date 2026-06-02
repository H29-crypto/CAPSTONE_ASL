"""
colab_demo.py — Sign Language Demo for Google Colab (T4 GPU)
=============================================================
Supports two models selectable from a Gradio dropdown:
  • AdaptSign  — ViT-B/16 + BiLSTM + CTC  (17.6% WER, best accuracy)
  • CorrNet    — ResNet18 + BiLSTM + CTC   (21%  WER, lighter model)

Both run at ~0.2-0.3 s/sign on T4 GPU.

── Colab setup (paste each block into a separate cell) ──────────────────────

CELL 1 — clone & install:
    !git clone https://github.com/H29-crypto/CAPSTONE_ASL.git
    %cd CAPSTONE_ASL
    !pip install -q gradio opencv-python-headless
    !pip install -q ctcdecode   # beam search — compiles on Linux (~3 min)

CELL 2 — mount Drive & run:
    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.insert(0, '/content/CAPSTONE_ASL/corrnet')

    import corrnet.colab_demo as d
    from pathlib import Path

    # Paths to your checkpoints on Google Drive
    d.ADAPTSIGN_CKPT = Path('/content/drive/MyDrive/phoenix2014-T_best.pt')
    d.CORRNET_CKPT   = Path('/content/drive/MyDrive/corrnet_phoenix2014T.pt')

    d.main()
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import OrderedDict

# ── ctcdecode — real on Linux Colab, mocked elsewhere ─────────────────────────
try:
    import ctcdecode
    BEAM_SEARCH_AVAILABLE = True
except Exception:
    import unittest.mock
    sys.modules["ctcdecode"] = unittest.mock.MagicMock()
    BEAM_SEARCH_AVAILABLE = False

import cv2
import numpy as np
import torch

# ── Paths (overridden from notebook cells) ────────────────────────────────────
_HERE          = Path(__file__).resolve().parent       # corrnet/
_ROOT          = _HERE.parent
ADAPTSIGN_CKPT = _ROOT / "adaptsign" / "weights" / "phoenix2014-T_best.pt"
CORRNET_CKPT   = _HERE / "weights"   / "corrnet_phoenix2014T.pt"
DICT_PATH      = _HERE / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "adaptsign"))

# Load gloss_to_text without polluting sys.path
import importlib.util as _ilu
_s = _ilu.spec_from_file_location("gloss_to_text", _HERE / "gloss_to_text.py")
_m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
glosses_to_german = _m.glosses_to_german
del _ilu, _s, _m

SUBSAMPLE_STEP = 3   # 100 raw frames → 33 fed to model

# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  CPU mode (slower — connect a T4 runtime for best speed)")
    return torch.device("cpu")


# ── Beam search decoder ───────────────────────────────────────────────────────
class BeamDecoder:
    """CTC beam search using ctcdecode when available, greedy fallback."""

    def __init__(self, gloss_dict: dict, beam_width: int = 5):
        self.i2g       = {v[0]: k for k, v in gloss_dict.items()}
        self.blank_id  = 0
        self.beam_width = beam_width
        labels = [""] + [self.i2g.get(i, "") for i in range(1, len(gloss_dict) + 1)]

        if BEAM_SEARCH_AVAILABLE:
            self._decoder = ctcdecode.CTCBeamDecoder(
                labels,
                beam_width=beam_width,
                blank_id=0,
                log_probs_input=False,
            )
            self._mode = "beam"
        else:
            self._mode = "greedy"

    def decode(self, logits: torch.Tensor, length: int):
        """
        logits : (T, C) — raw (not log-softmax) frame-level scores
        Returns : (glosses, confidences)
        """
        if self._mode == "beam":
            probs = torch.softmax(logits[:length].unsqueeze(0), dim=-1)  # (1,T,C)
            beam_results, _, _, out_len = self._decoder.decode(probs)
            ids  = beam_results[0][0][:out_len[0][0]].tolist()
            confs = [1.0] * len(ids)
        else:
            ids, confs = self._greedy(logits, length)

        glosses = [self.i2g.get(i, f"<{i}>") for i in ids]
        return glosses, confs

    def _greedy(self, logits, length):
        sm   = torch.softmax(logits[:length], dim=-1)
        idxs = logits[:length].argmax(dim=-1).tolist()
        seq, prev, rs, rn = [], None, 0.0, 0
        for pos, idx in enumerate(idxs):
            if idx != prev:
                if prev is not None and prev != self.blank_id and seq:
                    g, p, _ = seq[-1]; seq[-1] = (g, p, rs / max(rn, 1))
                prev, rs, rn = idx, 0.0, 0
                if idx != self.blank_id:
                    seq.append((idx, len(seq), 0.0))
            if idx != self.blank_id:
                rs += float(sm[pos, idx]); rn += 1
        if prev and prev != self.blank_id and seq:
            g, p, _ = seq[-1]; seq[-1] = (g, p, rs / max(rn, 1))
        ids   = [x[0] for x in seq]
        confs = [x[2] for x in seq]
        return ids, confs


# ── Model loading ─────────────────────────────────────────────────────────────
def load_adaptsign(ckpt_path: Path, gloss_dict: dict, device: torch.device):
    from slr_network import SLRModel
    model = SLRModel(
        num_classes=len(gloss_dict) + 1, c2d_type="ViT-B/16", conv_type=2,
        use_bn=True, hidden_size=1024, gloss_dict=gloss_dict,
        loss_weights={"SeqCTC": 1.0}, weight_norm=True, share_classifier=True,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd   = OrderedDict([(k.replace(".module", ""), v) for k, v in sd.items()])
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def load_corrnet(ckpt_path: Path, gloss_dict: dict, device: torch.device):
    from corrnet_webcam import load_model
    model = load_model(ckpt_path, gloss_dict)
    return model.to(device).eval()


# ── Frame extraction ──────────────────────────────────────────────────────────
def _prep_frame_adaptsign(rgb: np.ndarray) -> torch.Tensor:
    """Portrait-crop then resize to 256×256, centre-crop 224×224, /127.5-1."""
    h, w = rgb.shape[:2]
    tw = int(h * 210 / 260)
    if tw < w:
        x0 = (w - tw) // 2
        rgb = rgb[:, x0:x0 + tw]
    img = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    s   = (256 - 224) // 2
    img = img[s:s+224, s:s+224]
    return torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0


def extract_frames(video_path: str, max_frames: int = 100) -> list:
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_adaptsign(model, decoder, frames: list, device: torch.device):
    if not frames:
        return [], []
    sampled  = frames[::SUBSAMPLE_STEP]
    tensors  = [_prep_frame_adaptsign(f) for f in sampled]
    imgs     = torch.stack(tensors).unsqueeze(0).to(device)
    lengths  = torch.tensor([len(sampled)], dtype=torch.long).to(device)
    out      = model(imgs, lengths)
    sents    = out.get("recognized_sents", [[]])
    if sents and sents[0]:
        glosses = [x[0] for x in sents[0]]
        confs   = [x[2] if len(x) > 2 else 1.0 for x in sents[0]]
        return glosses, confs
    return [], []


@torch.no_grad()
def infer_corrnet(model, frames: list, device: torch.device):
    from corrnet_webcam import run_inference
    return run_inference(model, frames, device)


# ── Gradio ────────────────────────────────────────────────────────────────────
def build_interface(models: dict, gloss_dict: dict, device: torch.device):
    try:
        import gradio as gr
    except ImportError:
        print("Run:  pip install gradio"); return

    model_names = list(models.keys())

    def predict(video_path, model_name):
        if video_path is None:
            return "No video received.", "", "", "—"

        t0     = time.time()
        frames = extract_frames(video_path)
        if len(frames) < 5:
            return "Clip too short — sign for at least 1 second.", "", "", "—"

        m = models[model_name]
        if model_name == "AdaptSign (ViT-B/16) — best WER":
            glosses, confs = infer_adaptsign(m["model"], m["decoder"], frames, device)
        else:
            glosses, confs = infer_corrnet(m["model"], frames, device)

        elapsed = time.time() - t0

        if not glosses:
            return "No sign detected — try again.", "", "", f"{elapsed:.2f}s"

        german  = glosses_to_german(glosses, add_period=True) or " ".join(glosses)
        raw     = "  |  ".join(f"{g} {c*100:.0f}%" for g, c in zip(glosses, confs))
        conf_avg = sum(confs) / len(confs) * 100 if confs else 0
        timing  = (f"{elapsed:.2f}s  |  {len(frames)} frames → "
                   f"{len(frames[::SUBSAMPLE_STEP])} sampled  |  "
                   f"avg conf {conf_avg:.0f}%  |  {device.type.upper()}")

        return german, raw, timing

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Video(sources=["webcam", "upload"],
                     label="Record or upload a signing clip (2–5 s)"),
            gr.Dropdown(choices=model_names, value=model_names[0],
                        label="Model"),
        ],
        outputs=[
            gr.Textbox(label="German translation", lines=2),
            gr.Textbox(label="Glosses + confidence", lines=2),
            gr.Textbox(label="Timing", lines=1),
        ],
        title="German Sign Language Recognition — Phoenix-2014-T",
        description=(
            "**Good phrases to try (DGS weather glosses):**  \n"
            "`HEUTE REGEN KOMMEN`  |  `MORGEN NORD WIND STARK`  |  "
            "`MINUS DREI GRAD`  |  `WOCHENENDE SONNE FREUNDLICH`\n\n"
            "Stand in the portrait zone (upper body, frontal), plain background."
        ),
        allow_flagging="never",
    )
    demo.launch(share=True, server_port=7860)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*62)
    print("  Sign Language Demo — Colab Edition")
    print(f"  Beam search : {'ON (ctcdecode)' if BEAM_SEARCH_AVAILABLE else 'OFF (greedy — pip install ctcdecode)'}")
    print("="*62)

    device     = get_device()
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"  Vocabulary : {len(gloss_dict)} Phoenix glosses\n")

    models = {}

    if ADAPTSIGN_CKPT.exists():
        print("  Loading AdaptSign (ViT-B/16) ...")
        m = load_adaptsign(ADAPTSIGN_CKPT, gloss_dict, device)
        try:
            m = torch.compile(m, backend="eager")
        except Exception:
            pass
        models["AdaptSign (ViT-B/16) — best WER"] = {
            "model":   m,
            "decoder": BeamDecoder(gloss_dict, beam_width=5),
        }
        print(f"  AdaptSign ready  ({sum(p.numel() for p in m.parameters()):,} params)\n")
    else:
        print(f"  [SKIP] AdaptSign checkpoint not found: {ADAPTSIGN_CKPT}\n"
              "         Set d.ADAPTSIGN_CKPT = Path('/content/drive/...')\n")

    if CORRNET_CKPT.exists():
        print("  Loading CorrNet (ResNet18) ...")
        from corrnet_webcam import load_model
        m = load_model(CORRNET_CKPT, gloss_dict)
        m = m.to(device)
        try:
            m = torch.compile(m, backend="eager")
        except Exception:
            pass
        models["CorrNet (ResNet18) — faster load"] = {"model": m}
        print(f"  CorrNet ready  ({sum(p.numel() for p in m.parameters()):,} params)\n")
    else:
        print(f"  [SKIP] CorrNet checkpoint not found: {CORRNET_CKPT}\n"
              "         Set d.CORRNET_CKPT = Path('/content/drive/...')\n")

    if not models:
        print("[ERROR] No checkpoints found. Set d.ADAPTSIGN_CKPT / d.CORRNET_CKPT.")
        return

    print("  Launching Gradio ...")
    build_interface(models, gloss_dict, device)


if __name__ == "__main__":
    main()
