# Continuous Sign Language Recognition — PHOENIX-2014-T Demo

A local inference and demo system for continuous sign language recognition (CSLR).
The model is a BiLSTM encoder trained with CTC loss on the PHOENIX-2014-T dataset
(German Sign Language weather forecasts). Given a folder of PNG video frames, it
predicts a sequence of sign glosses (e.g. `MORGEN REGEN NORD`).

> **Domain note:** The model is trained exclusively on professional studio footage
> of German Sign Language (DGS) weather reports. It will not generalise well outside
> this domain.

---

## Performance

| Split | WER   | Notes                          |
|-------|-------|--------------------------------|
| Dev   | 72.12% | used for early stopping       |
| Test  | 70.12% | held-out evaluation           |

WER = Word Error Rate (lower is better). A WER of 70% means the model correctly
predicts roughly 30% of glosses on average. This is a competitive baseline result
for the PHOENIX-2014-T benchmark without beam search or a language model.

---

## Project Structure

```
sign_language_demo/
├── weights/
│   └── ctc_best_v2.pt          # Trained checkpoint (~46 MB)
├── sample_videos/
│   ├── 27November_2009_.../    # 58 frames
│   ├── 01September_2010_.../   # 112 frames
│   ├── 10March_2011_.../       # 130 frames
│   └── test_labels.csv         # Ground-truth glosses
├── src/
│   ├── model.py                # BiLSTMCTC_V2 architecture
│   ├── feature_extractor.py    # ResNet-18 → 512-dim features
│   ├── decoder.py              # Greedy CTC decode
│   └── inference.py            # SignLanguageRecognizer API
├── demos/
│   ├── demo_phoenix_video.py   # Single-video CLI demo
│   └── demo_all_samples.py     # All-samples summary table
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

### Verify the checkpoint loads correctly

```bash
python -c "
import torch
ckpt = torch.load('weights/ctc_best_v2.pt', map_location='cpu', weights_only=False)
print('Vocab size:', len(ckpt['gloss2idx']))
print('Dev WER:   ', round(ckpt['dev_wer'] * 100, 2), '%')
"
```

Expected output:
```
Vocab size: 1088
Dev WER:    72.12 %
```

---

## Running the Demos

All commands should be run from the `sign_language_demo/` directory.

### Demo 1 — Single video with reference

```bash
python demos/demo_phoenix_video.py \
    --video_folder sample_videos/27November_2009_Friday_tagesschau-7342
```

### Demo 1 — Single video without reference

```bash
python demos/demo_phoenix_video.py \
    --video_folder sample_videos/27November_2009_Friday_tagesschau-7342 \
    --no_reference
```

### Demo 2 — All sample videos (summary table)

```bash
python demos/demo_all_samples.py
```

---

## How It Works

1. **Frame loading** — PNG files are loaded with PIL and sorted lexicographically
   (zero-padded filenames ensure correct order).
2. **Frame stride** — Every other frame is kept (`stride=2`), matching training.
3. **Feature extraction** — Each kept frame passes through a pretrained ResNet-18
   (fc → Identity) after 224×224 resize and ImageNet normalisation → 512-dim vector.
4. **Recognition** — The BiLSTM encoder processes the feature sequence and outputs
   per-frame log-probabilities over 1088 tokens.
5. **CTC decoding** — Greedy argmax → collapse repeats → remove blanks → gloss list.
