# Keypoint-Centric Sign Language Recognition

A capstone project on **sign language recognition (SLR)** across both of its
canonical settings, unified by a keypoint-centric methodology and a strict
controlled-comparison discipline:

- **Part I — Isolated ASL Recognition (phase-aware):** classify a single sign
  from a short clip, using MediaPipe keypoints and an explicit model of sign
  *phonological phase* (preparation → stroke → retraction).
- **Part II — Continuous Sign Language Recognition (multimodal):** transcribe an
  unsegmented signing stream into a gloss sequence on the PHOENIX-2014 benchmark,
  fusing frozen I3D video features with HRNet keypoints under CTC.

> **Central findings.**
> (1) Across both regimes, **skeleton keypoints carry the dominant signal**.
> (2) In continuous recognition, **multimodal fusion beats the best single
> stream — but only once the two streams share a compatible temporal
> resolution.** Diagnosing and fixing that resolution mismatch is the core
> contribution of Part II.

**Faculty of Engineering and Natural Sciences — Department of Artificial
Intelligence Engineering, Bahçeşehir University**
Authors: Hamdi Alakkad, Metin Yağız Bakış, Yaser Hadri, Aria Sokhangoo
Advisors: Prof. Fatih Kahraman, Asst. Prof. Arezoo Sadeghzadeh

---

## Table of Contents
- [Results at a glance](#results-at-a-glance)
- [Part I — Isolated ASL Recognition](#part-i--isolated-asl-recognition)
- [Part II — Continuous Sign Language Recognition](#part-ii--continuous-sign-language-recognition)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets and external resources](#datasets-and-external-resources)
- [Reproducing the results](#reproducing-the-results)
- [Demos and figures](#demos-and-figures)
- [Key design decisions and honest limitations](#key-design-decisions-and-honest-limitations)
- [Citations](#citations)
- [Acknowledgements](#acknowledgements)
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

Test WER (%), greedy decoding, internal metric (lower is better):

| Exp. | Streams | Stride-4 | Stride-2 |
|---|---|---|---|
| E1 | Video only (frozen I3D) | 53.37 | 50.55 |
| E2 | Keypoints only (HRNet) | 44.71 | — *(resolution-independent)* |
| E3 | Fusion (concat) | 47.86 | 44.37 |
| **E4** | **Fusion + auxiliary CTC** | 45.10 | **40.61** |

**The headline result.** At the original stride-4 resolution, naive fusion
*underperformed* the keypoint-only model (E4 45.10 vs E2 44.71). We diagnosed
the cause as a **temporal-resolution mismatch** — aligning per-frame keypoints
down to the coarse video stride discarded the detail driving the keypoint
stream. Re-extracting the video stream at **stride 2** halved the mismatch, and
**fusion then beat the best single stream by 4.10 points (E4 40.61 vs E2
44.71)**, confirming the diagnosis. Every configuration improved at finer
resolution, and the fusion configurations improved most (−4.49 for E4) —
exactly as predicted, since fusion was the configuration most harmed by the
mismatch.

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
- A real-time **record-one-sign webcam demo** runs the full pipeline live.

---

## Part II — Continuous Sign Language Recognition

A feature-space, two-stream multimodal CSLR system. Both backbones are
**frozen**, so the whole experiment ladder trains in minutes on a single GPU.

```
                 PHOENIX-2014 video (5,672 / 540 / 629 sentences)
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   VIDEO STREAM                       KEYPOINT STREAM
 I3D-R50 (frozen, Kinetics)        HRNet COCO-WholeBody (133 pts)
 window 8, stride 4 or 2            select 53 pts (hands + upper body)
 → [T/s, 2048]                     → [T, 53x5 = 265], shoulder-normalized
        │                                   │
        │                        ┌──────────┴──────────┐
        │                        │ resample to video   │  <- fusion-only step;
        │                        │ length (stride 4->2 │     the resolution
        │                        │ halves the loss)    │     bottleneck
        │                        └──────────┬──────────┘
   stream encoder                      stream encoder
 (proj 512 + 2x Conv1d k5)         (proj 512 + 2x Conv1d k5)
        │                                   │
   (aux CTC head, E4 only)           (aux CTC head, E4 only)
        └─────────────────┬─────────────────┘
                          │
                 FUSION (concatenation)
                          │
            SEQUENCE HEAD (BiLSTM, 2 x 512)
                          │
                    MAIN CTC HEAD
                          │
                greedy CTC decoding
                          │
        gloss sequence -> WER (1,231-gloss vocabulary)
```

**Experiment ladder (matched conditions — one variable per row):**
- **E1** video only · **E2** keypoints only (full frame rate, no resampling) ·
  **E3** both streams, concat fusion · **E4** E3 + per-stream auxiliary CTC heads.
- Each row was run at **two video resolutions** (stride 4 and stride 2) to study
  the effect of stream-resolution matching on fusion.
- Identical encoder dims, sequence head, optimizer, schedule, augmentation, and
  decoding across rows, so differences are attributable to the studied factor
  alone.

**The resolution study (Part II's core finding).**
1. At stride 4, fusion (E4 45.10) *underperformed* keypoints-only (E2 44.71).
2. Diagnosis: aligning per-frame keypoints (~215 steps) down to stride-4 video
   (~54 steps) discarded ~75% of keypoint temporal detail.
3. Prediction: stride-2 video (~108 steps) halves the mismatch and should let
   fusion recover.
4. Confirmation: at stride 2, **E4 reached 40.61 test WER, beating E2 by 4.10
   points** — fusion now wins, on held-out test data.

---

## Repository structure

> ⚠️ **Confirm against your actual commit.** Adjust paths/filenames to match what
> is really in the repo before publishing. Below is the intended layout.

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
│   ├── notebooks/
│   │   ├── 01_dataset_eval_check.ipynb
│   │   ├── 02_dump_rgb_features_resumable.ipynb   # stride 4 and stride 2
│   │   ├── 03_prepare_hrnet_keypoints.ipynb
│   │   └── 04_ctc_sanity_check.ipynb
│   └── results/
│       ├── final_results_stride2.json
│       ├── transcription_demo.html  # qualitative ref-vs-prediction view
│       └── resolution_study.png     # the headline figure
│
└── docs/
    └── final_report.pdf
```

---

## Installation

Both parts use Python 3.10+ and PyTorch >= 2.0, developed primarily in Google
Colab (single GPU) with Google Drive for persistent storage.

```bash
git clone https://github.com/H29-crypto/CAPSTONE_ASL.git
cd CAPSTONE_ASL

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# core
pip install torch numpy pandas opencv-python matplotlib scipy pyyaml

# Part I (isolated): MediaPipe for landmarks + the webcam demo
pip install mediapipe

# Part II (continuous): PyTorchVideo for the frozen I3D extractor, gdown for keypoints
pip install pytorchvideo gdown
```

> **Environments.** The continuous pipeline runs in Colab. The Part I **webcam
> demo** is best run **locally** (VS Code, Python 3.10): Colab has unreliable
> real-time camera access and dependency conflicts among MediaPipe / TensorFlow
> / protobuf / NumPy.

---

## Datasets and external resources

Nothing is redistributed here — obtain each from its original source under its
own license.

### Part I — ASL Citizen
Community-sourced, signer-independent isolated-ASL dataset (Desai et al.,
NeurIPS 2023). Full set: 83,401 videos, 2,731 classes, 52 signers. This project
uses a **top-100-class subset** (by training frequency); reported results are on
the **post-filter** split (1,215 train / 242 val / 777 test) after
phase-segmentation quality filtering.
Source: `https://www.microsoft.com/en-us/research/project/asl-citizen/`

### Part II — RWTH-PHOENIX-Weather 2014 (continuous, DGS)
The standard gloss-annotated CSLR benchmark (Koller et al., 2015). **Full
official multisigner splits** (5,672 / 540 / 629). The archive ships frames,
annotations, and the official evaluation scripts — **no keypoints, no
pretrained models** (keypoints are obtained separately, below).
Source: `https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/`

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
MSKA repository: `https://github.com/sutwangyan/MSKA`

### Pretrained feature extractor (Part II video stream)
**Kinetics-pretrained I3D-R50** via **PyTorchVideo** (auto-downloaded,
`i3d_r50`, ~214 MB). Used frozen, head removed, mean-pooled to 2,048-d per
8-frame window. It is a deliberately *generic* visual baseline (Kinetics
contains everyday actions, not signing).

### Architectural references
- **Part I** keypoint subset informed by **SAM-SLR** (Jiang et al., CVPRW 2021):
  `https://github.com/jackyjsy/CVPR21Chal-SLR`
- TCN block design follows **ED-TCN** (Lea et al., CVPR 2017).
- **Part II** keypoints from **MSKA** (above), whose reported keypoint-only WER
  (21.2%) indicates substantial headroom above this project's frozen-feature
  numbers, attributable to encoder capacity and training technique rather than
  to the modality.

---

## Reproducing the results

### Part I (isolated)
1. Extract MediaPipe keypoints for the top-100 subset (`isolated/notebooks/Feature.ipynb`).
2. Run the training pipeline (`isolated/notebooks/Un.ipynb`): motion-feature
   construction and weak phase pseudo-labels -> Phase Detection TCN ->
   active-region extraction -> Recognition TCN (phase-aware **and** baseline).
3. Result tables/figures are written under `isolated/reports/final_results/`.

### Part II (continuous)
All paths live in `continuous/config.yaml`; set them once.

1. **Verify the dataset** (`01_*`) — confirm official counts (5,672/540/629) and
   build the train-only vocabulary.
2. **Extract video features** (`02_*`) — frozen I3D, 8-frame windows. Run once at
   **stride 4** and once at **stride 2** (separate output directories). Resumable.
3. **Prepare keypoints** (`03_*`) — select 53 points, shoulder-normalize, add
   velocities, standardize with **train-only** stats; align to each video stride.
4. **Sanity check** (`04_*`) — stream-alignment asserts, `T >= 2L`
   CTC-feasibility check, one forward / CTC / greedy-decode pass.
5. **Train any experiment** with a one-line call (~30-40 min each on a T4):

```python
from src.train import run_training
run_training(cfg, ("rgb",),     "E1_rgb")                                    # video only
run_training(cfg, ("kp",),      "E2_kp")                                     # keypoints only
run_training(cfg, ("rgb","kp"), "E3_fusion")                                 # concat fusion
run_training(cfg, ("rgb","kp"), "E4_aux", overrides={"model.aux_ctc": True}) # + aux CTC
```
For the stride-2 study, repoint `config.yaml` to the stride-2 feature
directories and re-run E1/E3/E4 under `*_s2` run names (E2 is
resolution-independent and is not re-run).

6. **Evaluate on test once per model** — each checkpoint is loaded with its own
   saved config, guaranteeing it is scored with exactly the feature directories
   and dimensions it was trained on.

Training is fully **resumable**: every job checkpoints `best.pt` (by dev WER) and
`last.pt` (latest epoch); re-running continues from the latest state.

---

## Demos and figures

- **`continuous/results/transcription_demo.html`** — a qualitative, color-coded
  view of the best model (E4 stride-2) transcribing real PHOENIX-2014 test
  sentences: reference vs. predicted glosses, with correct / substitution /
  insertion / deletion highlighted. Makes the WER concrete and exposes
  interpretable failure modes (similar-sign substitutions and CTC
  repeat-insertions).
- **`continuous/results/resolution_study.png`** — the headline figure: test WER
  for E1/E3/E4 at stride 4 vs stride 2, with the E2 keypoint-only reference line,
  showing fusion crossing below it at stride 2.
- **Part I real-time demo** — `isolated/realtime_demo/demo.py`: press **R** to
  record a ~2-second clip; the pipeline extracts MediaPipe landmarks, runs the
  Phase TCN to find the active region, and returns the top prediction among 100
  classes. (There is no live demo for the continuous model: it needs HRNet
  whole-body keypoints and outputs DGS weather-domain glosses, both out of scope
  for a webcam.)

---

## Key design decisions and honest limitations

This project deliberately reports negative and null results, then resolves them.

- **Frozen features by design.** Both backbones (I3D, HRNet) are frozen — a
  compute-budget choice (single free GPU) that sets a deliberate performance
  ceiling. End-to-end PHOENIX systems (e.g., CorrNet ~19% WER) are not a
  like-for-like comparison.
- **Generic I3D, not sign-pretrained.** The planned sign-pretrained frontend was
  unavailable, so Kinetics-I3D is used — which doubles as the "generic visual
  features" control, and is part of *why* keypoints win.
- **Fusion required resolution matching.** Naive stride-4 fusion underperformed
  the best single stream; stride-2 re-extraction resolved it (E4 40.61 < E2
  44.71). Going finer still (stride 1) is untested future work.
- **Internal WER, not official sclite.** Valid for the relative comparisons made
  here; binding the official PHOENIX scripts is listed as future work.
- **Raw 1,231-gloss vocabulary** (vs the ~1,081 normalized vocabulary common in
  the literature) is kept for internal consistency.
- **149 stride-4 training sentences (2.6%)** were CTC-infeasible (`T < 2L`) and
  neutralized via `zero_infinity`; at stride 2 nearly all become feasible.
- **Phase labels are weak pseudo-labels**, not linguistic ground truth; the
  isolated +0.9% gain is not statistically significant (McNemar p ~ 0.50).
- **Signer protocols differ between parts:** ASL Citizen is signer-independent;
  PHOENIX-2014 multisigner shares signers across splits (the standard, easier
  protocol). Results are labeled accordingly and not over-claimed.

**Future work.** Stride-1 video features; bind the official PHOENIX evaluator and
certify against a published checkpoint; beam search + a gloss language model
(the transcription demo shows CTC repeat-insertions a LM would suppress);
gated-fusion and Transformer-head ablations; cross-stream self-distillation;
injecting phase posteriors into the continuous stream (bridging the two parts);
human-annotated phases and a larger vocabulary for an adequately-powered isolated
test.

---

## Citations

If you reference this work:

```bibtex
@misc{capstone_slr_2026,
  title  = {Keypoint-Centric Sign Language Recognition: Phase-Aware Isolated
            ASL Recognition and Multimodal Continuous Sign Language Recognition},
  author = {Alakkad, Hamdi and Bakis, Metin Yagiz and Hadri, Yaser and
            Sokhangoo, Aria},
  year   = {2026},
  note   = {Capstone Project, Department of Artificial Intelligence Engineering,
            Bahcesehir University}
}
```

**Datasets**
- A. Desai et al., "ASL Citizen: A Community-Sourced Dataset for Advancing
  Isolated Sign Language Recognition," *NeurIPS* (Datasets & Benchmarks), 2023.
- O. Koller, J. Forster, H. Ney, "Continuous sign language recognition: Towards
  large vocabulary statistical recognition systems handling multiple signers,"
  *CVIU*, 141:108-125, 2015.

**Models, methods, and tooling**
- M. Guan et al., "MSKA: Multi-stream keypoint attention network for sign
  language recognition and translation," *Pattern Recognition*, 165:111602, 2025.
- J. Carreira, A. Zisserman, "Quo Vadis, Action Recognition? A New Model and the
  Kinetics Dataset," *CVPR*, 2017.
- W. Kay et al., "The Kinetics Human Action Video Dataset," arXiv:1705.06950, 2017.
- K. Sun, B. Xiao, D. Liu, J. Wang, "Deep High-Resolution Representation Learning
  for Human Pose Estimation" (HRNet), *CVPR*, 2019.
- S. Jin et al., "Whole-Body Human Pose Estimation in the Wild" (COCO-WholeBody),
  *ECCV*, 2020.
- A. Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented
  Sequence Data with Recurrent Neural Networks," *ICML*, 2006.
- S. Bai, J. Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic
  Convolutional and Recurrent Networks for Sequence Modeling" (TCN),
  arXiv:1803.01271, 2018.
- C. Lea et al., "Temporal Convolutional Networks for Action Segmentation and
  Detection" (ED-TCN), *CVPR*, 2017.
- S. Jiang et al., "Skeleton Aware Multi-Modal Sign Language Recognition"
  (SAM-SLR), *CVPRW*, 2021.
- C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines,"
  arXiv:1906.08172, 2019.
- H. Fan et al., "PyTorchVideo: A Deep Learning Library for Video Understanding,"
  *ACM MM*, 2021.
- L. Hu, L. Gao, Z. Liu, W. Feng, "Continuous Sign Language Recognition with
  Correlation Network" (CorrNet), *CVPR*, 2023.
- Y. Min et al., "Visual Alignment Constraint for Continuous Sign Language
  Recognition" (VAC, auxiliary CTC), *ICCV*, 2021.
- Q. McNemar, "Note on the sampling error of the difference between correlated
  proportions or percentages," *Psychometrika*, 12(2):153-157, 1947.

**Sign phonology (Part I linguistic basis)**
- S. K. Liddell, R. E. Johnson, "American Sign Language: The Phonological Base,"
  *Sign Language Studies*, 64:195-277, 1989.
- D. Brentari, *A Prosodic Model of Sign Language Phonology*, MIT Press, 1998.

---

## Acknowledgements

This work uses only open-source software and publicly released,
research-licensed datasets and checkpoints. No new human-subject data were
collected. Gloss recognition is **not** translation: outputs are gloss
sequences, and the system is intended to support — not replace — human
interpreters. We thank our advisors, Prof. Fatih Kahraman and Asst. Prof. Arezoo
Sadeghzadeh, for their guidance.

---

## License

> Add a license of your choice (e.g., MIT for the code). Note that the
> **datasets and third-party checkpoints retain their own licenses** and are not
> redistributed here — obtain them from the original sources under their
> respective terms.
