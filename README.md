# Keypoint-Centric Sign Language Recognition

A capstone project on **sign language recognition (SLR)** spanning both of its
canonical settings, unified by a keypoint-centric methodology and a strict
controlled-comparison discipline:

- **Part I — Isolated ASL Recognition (phase-aware):** classify a single sign
  from a short clip, using MediaPipe keypoints and an explicit model of sign
  *phonological phase* (preparation → stroke → retraction).
- **Part II — Continuous Sign Language Recognition (multimodal):** transcribe an
  unsegmented signing stream into a gloss sequence on the PHOENIX-2014 benchmark,
  fusing frozen I3D video features with HRNet keypoints under CTC.

> **Central finding.** Across both regimes, **skeleton keypoints carry the
> dominant signal**, and structural augmentations (phase features in isolated; a
> fused video stream with auxiliary supervision in continuous) yield consistent
> but modest, honestly-characterized gains.

**Faculty of Engineering and Natural Sciences — Department of Artificial
Intelligence Engineering**
Authors: Hamdi Alakkad, Metin Yağız Bakış, Yaser Hadri, Aria Sokhangoo
Advisors: Prof. Fatih Kahraman, Asst. Prof. Arezoo Sadeghzadeh

---

## Table of Contents
- [Results at a glance](#results-at-a-glance)
- [Part I — Isolated ASL Recognition](#part-i--isolated-asl-recognition)
- [Part II — Continuous Sign Language Recognition](#part-ii--continuous-sign-language-recognition)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Reproducing the results](#reproducing-the-results)
- [Real-time demo](#real-time-demo)
- [Key design decisions and honest limitations](#key-design-decisions-and-honest-limitations)
- [Citation and acknowledgements](#citation-and-acknowledgements)
- [License](#license)

---

## Results at a glance

### Part I — Isolated ASL (ASL Citizen, top-100 classes, 777 test samples)

| Model | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|
| Baseline (no phase) | 67.95% | 85.97% | 0.6574 |
| **Phase-aware** | **68.85%** | **87.00%** | **0.6695** |

Phase-detection TCN: **91.1%** frame accuracy / **89.7%** macro-F1.
Paired McNemar test: **p ≈ 0.50** — the +0.90-point gain is consistent in
direction but **not statistically significant** at this sample size, and is
reported as a promising signal rather than a breakthrough.

### Part II — Continuous CSLR (PHOENIX-2014, full official splits)

WER (%), greedy decoding, internal metric (lower is better):

| Exp. | Streams | Dev WER | Test WER |
|---|---|---|---|
| E1 | Video only (frozen I3D) | 53.86 | 53.37 |
| **E2** | **Keypoints only (HRNet)** | **44.06** | **44.71** |
| E3 | Fusion (concat) | 49.19 | 47.86 |
| E4 | Fusion + auxiliary CTC | 46.03 | 45.10 |

**Keypoints beat generic video features by 8.66 test-WER points.** Naive fusion
underperforms the best single stream because resolution alignment discards
keypoint temporal detail; auxiliary CTC recovers 2.76 points but does not close
the gap. See [limitations](#key-design-decisions-and-honest-limitations).

> WER is computed with an internal, consistent implementation applied
> identically to all experiments, so **relative** comparisons are valid. It is
> **not** the official PHOENIX sclite metric, so absolute numbers are not
> directly comparable to published leaderboards.

---

## Part I — Isolated ASL Recognition

A four-stage, keypoint-based pipeline that exploits the three-phase phonological
structure of signs.

```
ASL Citizen clip
   │
   ▼
MediaPipe pose + both hands  →  49 landmarks (147-d positions)
   │
   ▼
normalization + derivatives  →  444-d motion features / frame
   │  (position, velocity, acceleration, global/hand/phase speed)
   ▼
Phase Detection TCN          →  per-frame phase
   │  (weak velocity-derived pseudo-labels:
   │   background / preparation / stroke / retraction)
   │  91.1% acc · 89.7% macro-F1
   ▼
active-region extraction  +  4 ordered phase one-hots (→ 448-d)
   │
   ▼
Recognition TCN + attention pooling
   │
   ▼
softmax over 100 ASL classes
```

**Design notes.**
- Phase labels are **weak pseudo-labels** from motion-peak heuristics on velocity
  curves — not human linguistic annotations. The 91.1% phase accuracy measures
  agreement with the heuristic, not with ground-truth phonology.
- The baseline shares the *same* architecture and active-region crop, differing
  only by the 4 phase one-hot features — a clean single-variable comparison.
- The phase-quality filter excluded signs with multi-peak/oscillatory motion
  (e.g., TWINS, COMB, PIPE), a documented bias toward simpler motion profiles.

---

## Part II — Continuous Sign Language Recognition

A feature-space, two-stream multimodal CSLR system. Every visual/keypoint
backbone is **frozen**, so the whole experiment ladder trains in minutes on a
single GPU.

```
                 PHOENIX-2014 video (5,672 / 540 / 629 sentences)
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   VIDEO STREAM                       KEYPOINT STREAM
 I3D-R50 (frozen, Kinetics)        HRNet COCO-WholeBody (133 pts)
 window 8, stride 4                 select 53 pts (hands + upper body)
 → [T/4, 2048]                      → [T, 53×5 = 265], shoulder-normalized
        │                                   │
        │                        ┌──────────┴──────────┐
        │                        │ resample to video   │  ← fusion-only step
        │                        │ length (~215 → ~54)  │
        │                        └──────────┬──────────┘
   stream encoder                      stream encoder
 (proj 512 + 2× Conv1d k5)         (proj 512 + 2× Conv1d k5)
        │                                   │
   (aux CTC head, E4 only)           (aux CTC head, E4 only)
        └─────────────────┬─────────────────┘
                          │
                 FUSION (concatenation)
                          │
            SEQUENCE HEAD (BiLSTM, 2 × 512)
                          │
                    MAIN CTC HEAD
                          │
                greedy CTC decoding
                          │
        gloss sequence → WER (1,231-gloss vocabulary)
```

**Experiment ladder (matched conditions — one variable per row):**
- **E1** video only · **E2** keypoints only (full frame rate, no resampling) ·
  **E3** both streams, concat fusion · **E4** E3 + per-stream auxiliary CTC heads.
- Identical encoder dims, sequence head, optimizer, schedule, augmentation, and
  decoding across rows, so differences are attributable to the modality / fusion
  / auxiliary-supervision factor alone.

---

## Repository structure

```
.
├── README.md
├── isolated/                        # Part I — phase-aware ASL (ASL Citizen)
│   ├── notebooks/
│   │   ├── Feature.ipynb            # MediaPipe keypoint extraction (Tasks API)
│   │   └── Un.ipynb                 # full training pipeline
│   ├── models/
│   │   ├── phase_tcn_best_safe_state.pt
│   │   └── recognition_tcn_attention_best.pt
│   ├── manifests/
│   │   └── recognition_label_map_with_split_counts.csv
│   ├── reports/final_results/       # result tables + figures
│   └── realtime_demo/
│       ├── demo.py                  # webcam record-one-sign demo
│       └── verify_pipeline.py       # offline checkpoint verification
│
├── continuous/                      # Part II — multimodal CSLR (PHOENIX-2014)
│   ├── config.yaml                  # single source of truth (paths + hyperparams)
│   ├── src/
│   │   ├── vocab.py                 # corpus parsing + train-only vocabulary
│   │   ├── data.py                  # feature dataset, alignment, augmentation
│   │   ├── models.py                # stream encoders, fusion, BiLSTM, CTC heads
│   │   ├── decode.py                # greedy CTC decode + WER (S/I/D)
│   │   ├── train.py                 # training engine (resumable, dev-WER select)
│   │   ├── utils.py                 # config/seed/path helpers
│   │   └── official_eval_adapter.py # PHOENIX sclite binding (stub + instructions)
│   └── notebooks/
│       ├── 01_dataset_eval_check.ipynb
│       ├── 02_dump_rgb_features_resumable.ipynb
│       ├── 03_prepare_hrnet_keypoints.ipynb
│       └── 04_ctc_sanity_check.ipynb
│
└── docs/
    └── final_report.pdf
```

---

## Installation

Both parts use Python 3.10+ and PyTorch ≥ 2.0. They are developed primarily in
Google Colab (single GPU) with Google Drive for persistent storage.

```bash
# clone
git clone https://github.com/H29-crypto/CAPSTONE_ASL.git
cd CAPSTONE_ASL

# (recommended) virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# core dependencies
pip install torch numpy pandas opencv-python matplotlib scipy pyyaml

# Part I (isolated): MediaPipe for landmark extraction + the webcam demo
pip install mediapipe

# Part II (continuous): PyTorchVideo for the frozen I3D feature extractor
pip install pytorchvideo gdown
```

> **Note on environments.** The continuous pipeline runs entirely in Colab.
> The Part I **webcam demo** is best run **locally** (VS Code, Python 3.10): Colab
> has unreliable real-time camera access and dependency conflicts among
> MediaPipe / TensorFlow / protobuf / NumPy.

---

## Datasets

Neither dataset is redistributed here. Download from the original sources.

### Part I — ASL Citizen
Community-sourced, signer-independent isolated-ASL dataset (Desai et al.,
NeurIPS 2023). Full set: 83,401 videos, 2,731 classes, 52 signers. This project
uses a **top-100-class subset** (by training frequency); reported results are on
the **post-filter** split (1,215 train / 242 val / 777 test) after
phase-segmentation quality filtering.

- Project page / download: see the ASL Citizen release.

### Part II — RWTH-PHOENIX-Weather 2014 (continuous, DGS)
The standard gloss-annotated CSLR benchmark (Koller et al., 2015). **Full
official multisigner splits** (5,672 / 540 / 629). The archive ships frames,
annotations, and the official evaluation scripts — **no keypoints and no
pretrained models** (keypoints are obtained separately, below).

- `phoenix-2014.v3.tar.gz` from the RWTH PHOENIX page.

### Part II — Keypoints (separate third-party artifact)
PHOENIX does not include keypoints. We use **pre-extracted HRNet whole-body
keypoints** released by the **MSKA** project (Guan et al., 2025): one file per
split, 133-point COCO-WholeBody, per-frame, with confidences. We select 53
points (upper body + both hands) and normalize them ourselves.

```python
import gdown
gdown.download_folder(
    "https://drive.google.com/drive/folders/1D_iVtqeARBLO7WcZCTGCAdHXkKqHfF9X",
    output="data/keypoints_raw", quiet=False, use_cookies=False)
```

---

## Reproducing the results

### Part I (isolated)
1. Extract MediaPipe keypoints for the top-100 subset (`isolated/notebooks/Feature.ipynb`).
2. Run the training pipeline (`isolated/notebooks/Un.ipynb`): motion-feature
   construction and weak phase pseudo-labels → Phase Detection TCN →
   active-region extraction → Recognition TCN (phase-aware **and** baseline).
3. Result tables/figures are written under `isolated/reports/final_results/`.

### Part II (continuous)
All paths live in `continuous/config.yaml`; set them once.

1. **Verify the dataset** (`01_dataset_eval_check.ipynb`) — confirm official
   counts (5,672 / 540 / 629) and build the train-only vocabulary.
2. **Extract video features** (`02_dump_rgb_features_resumable.ipynb`) — frozen
   I3D, 8-frame windows, stride 4 → one `.npy` per sentence. Resumable.
3. **Prepare keypoints** (`03_prepare_hrnet_keypoints.ipynb`) — select 53 points,
   shoulder-normalize, add velocities, standardize with **train-only** stats.
4. **Sanity check** (`04_ctc_sanity_check.ipynb`) — stream-alignment asserts,
   `T ≥ 2L` CTC-feasibility check, one forward / CTC / greedy-decode pass.
5. **Train any experiment** with a one-line call (each ≈ 30 min on a T4):

```python
from src.train import run_training

run_training(cfg, ("rgb",),        "E1_rgb")                                    # video only
run_training(cfg, ("kp",),         "E2_kp")                                     # keypoints only
run_training(cfg, ("rgb","kp"),    "E3_fusion")                                 # concat fusion
run_training(cfg, ("rgb","kp"),    "E4_aux", overrides={"model.aux_ctc": True}) # + aux CTC
```

6. **Evaluate on test once per model** — loads each checkpoint's own saved config
   so it is scored with exactly the feature directories and dimensions it was
   trained on.

Training is fully **resumable**: every job checkpoints `best.pt` (by dev WER) and
`last.pt` (latest epoch); re-running continues from the latest state.

---

## Real-time demo (Part I)

```bash
cd isolated/realtime_demo
python demo.py     # downloads MediaPipe model files on first launch
```

Press **R** to record a ~2-second clip; the pipeline extracts MediaPipe
landmarks, runs the Phase TCN to find the active region, and returns the top
prediction among 100 classes.

> The demo is **record-one-sign**, not continuous rolling-buffer: the model is
> trained on isolated clips, so a clean recorded segment matches the training
> distribution and gives stable predictions. There is no live demo for the
> continuous (PHOENIX) model — it requires HRNet whole-body keypoints and outputs
> DGS weather-domain glosses, both out of scope for a webcam.

---

## Key design decisions and honest limitations

This project deliberately reports negative and null results.

- **Frozen features by design.** Both backbones (I3D, HRNet) are frozen; this is
  a compute-budget choice (single free GPU) and sets a deliberate performance
  ceiling. End-to-end PHOENIX systems (e.g., CorrNet, ≈19% WER) are not a
  like-for-like comparison.
- **Generic I3D, not sign-pretrained.** The planned sign-pretrained frontend was
  unavailable, so Kinetics-I3D is used — which doubles as the "generic visual
  features" control. This is part of *why* keypoints win so decisively.
- **Fusion underperforms the best single stream (E3 < E2).** Aligning per-frame
  keypoints to the coarser video stride (~215 → ~54 steps) discards the temporal
  detail driving E2. Auxiliary CTC (E4) recovers part of it but not all. The
  indicated fix — re-extracting video features at stride 2 — is future work.
- **149 training sentences (2.6%) are CTC-infeasible at stride 4** (`T < 2L`) and
  are neutralized via `zero_infinity`. Dev/test are unaffected.
- **Internal WER, not official sclite.** Valid for the relative comparisons made
  here; binding the official PHOENIX scripts is listed as future work.
- **Raw 1,231-gloss vocabulary** (vs the ≈1,081 normalized vocabulary common in
  the literature) is kept for internal consistency.
- **Phase labels are weak pseudo-labels**, not linguistic ground truth; the
  isolated +0.9% gain is not statistically significant (McNemar p ≈ 0.50).
- **Signer protocols differ between parts:** ASL Citizen is signer-independent;
  PHOENIX-2014 multisigner shares signers across splits (the standard, easier
  protocol). Results are labeled accordingly and not over-claimed.

**Future work.** Stride-2 video re-extraction + matched-resolution re-runs; bind
the official PHOENIX evaluator and certify against a published checkpoint; beam
search + a gloss language model; gated-fusion and Transformer-head ablations;
cross-stream self-distillation; injecting phase posteriors into the continuous
stream (bridging the two parts); human-annotated phases and a larger vocabulary
for an adequately-powered isolated test.

---

## Citation and acknowledgements

If you reference this work:

```bibtex
@misc{capstone_slr_2026,
  title  = {Keypoint-Centric Sign Language Recognition: Phase-Aware Isolated
            ASL Recognition and Multimodal Continuous Sign Language Recognition},
  author = {Alakkad, Hamdi and Bak{\i}\c{s}, Metin Ya\u{g}{\i}z and Hadri, Yaser and
            Sokhangoo, Aria},
  year   = {2026},
  note   = {Capstone Project, Department of Artificial Intelligence Engineering}
}
```

**Datasets, models, and methods used:**
- **ASL Citizen** — Desai et al., *ASL Citizen: A Community-Sourced Dataset for
  Advancing Isolated Sign Language Recognition*, NeurIPS 2023.
- **RWTH-PHOENIX-Weather 2014** — Koller et al., *Continuous sign language
  recognition: Towards large vocabulary statistical recognition systems handling
  multiple signers*, CVIU 2015.
- **MSKA** (HRNet keypoints for PHOENIX) — Guan et al., *MSKA: Multi-stream
  keypoint attention network for sign language recognition and translation*,
  Pattern Recognition 2025.
- **I3D** — Carreira & Zisserman, CVPR 2017; checkpoint via **PyTorchVideo**
  (Fan et al., 2021), Kinetics-pretrained `i3d_r50`.
- **HRNet** — Sun et al., CVPR 2019; **COCO-WholeBody** — Jin et al., ECCV 2020.
- **MediaPipe** — Lugaresi et al., 2019.
- **CTC** — Graves et al., ICML 2006.
- **TCN** — Bai et al., 2018; **ED-TCN** — Lea et al., CVPR 2017.
- **SAM-SLR** (keypoint-subset inspiration, Part I) — Jiang et al., CVPRW 2021.
- **Sign phonology** — Liddell & Johnson (1989); Brentari (1998).

This work uses only open-source software and publicly released, research-licensed
datasets and checkpoints. No new human-subject data were collected. Gloss
recognition is **not** translation; outputs are gloss sequences, and the system
is intended to support — not replace — human interpreters.

---

## License

> Add a license of your choice (e.g., MIT for code). Note that the **datasets and
> third-party checkpoints retain their own licenses** and are not redistributed
> here — users must obtain them from the original sources under their respective
> terms.
