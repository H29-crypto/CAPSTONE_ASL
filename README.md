# Phase-Aware ASL Sign Recognition

A keypoint-based isolated American Sign Language recognition pipeline that
uses phase-aware temporal segmentation (preparation → stroke → retraction)
to improve sign classification.

This is a capstone project (Bahçeşehir University, Faculty of Engineering
and Natural Sciences, Department of AI Engineering, 2026).

## Overview

The system recognizes individual ASL signs from video clips using:

1. **MediaPipe** keypoint extraction (pose + both hands, 49 landmarks/frame)
2. A **Phase Detection TCN** that classifies each frame into one of four
   phonological phases (background, preparation, stroke, retraction)
3. A **Phase-Aware Recognition TCN with attention** that consumes the
   extracted active segment and predicts one of 100 sign classes

The phase-aware design is grounded in sign language phonology
(Liddell & Johnson 1989; Brentari 1998).

## Results (signer-independent test set, 777 clips)

| Model                        | Top-1      | Top-5      | Macro F1  |
|------------------------------|------------|------------|-----------|
| Baseline (no phase one-hot)  | 67.95%     | 85.97%     | 0.6574    |
| Phase-aware                  | 68.85%     | 87.00%     | 0.6695    |
| **Δ phase-aware − baseline** | **+0.90%** | **+1.03%** | **+0.0120** |

Paired McNemar's test on top-1 predictions: p ≈ 0.50 — the improvement is
consistent in direction but not statistically significant at this sample
size. See the project report for the full methodology audit and limitations.

## Dataset

This project uses a top-100-class subset of **ASL Citizen** (Desai et al.,
NeurIPS 2023), selected by training-set frequency.

| Split | Original | After phase-segmentation filter |
|-------|---------:|--------------------------------:|
| Train |    1,800 |                           1,215 |
| Val   |      367 |                             242 |
| Test  |    1,286 |                             777 |

The dataset itself is not redistributed with this repo. Download it from:
<https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip>

## Repository structure

```
CAPSTONE_ASL/
├── notebooks/
│   ├── Feature.ipynb                                  # Keypoint extraction (MediaPipe Tasks API)
│   └── Un.ipynb                                       # Full training pipeline (90 cells)
├── models/
│   ├── phase_tcn_best_safe_state.pt                   # Trained Phase Detection TCN
│   ├── recognition_tcn_attention_best.pt              # Trained Recognition TCN + Attention
│   ├── hand_landmarker.task                           # MediaPipe hand landmark model
│   └── pose_landmarker_full.task                      # MediaPipe pose landmark model
├── manifests/
│   └── recognition_label_map_with_split_counts.csv    # 100-class label map
├── reports/final_results/
│   ├── final_results_summary.json
│   ├── final_results_tables.md
│   ├── figures/                                       # fig_01–fig_06 (PNG)
│   └── report_text/                                   # 01_abstract.md – 07_conclusion.md
└── realtime_demo/
    ├── demo.py                                        # Webcam record-one-sign demo
    └── verify_pipeline.py                             # Offline checkpoint verification
```

## Installation

Tested on Windows 10/11 with Python 3.10. Other platforms should work but
have not been tested.

```bash
# 1. Clone the repo
git clone https://github.com/H29-crypto/CAPSTONE_ASL.git
cd CAPSTONE_ASL

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

On Windows, if PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Running the demo

A working webcam is required.

```bash
python realtime_demo/demo.py
```

Controls:
- **R** — record a 2-second clip and predict the sign
- **Q** — quit

On first run, the script downloads the two MediaPipe model files
(`pose_landmarker_full.task` and `hand_landmarker.task`, ~17 MB total)
into `models/`.

To verify the inference pipeline without a webcam — useful for confirming
the checkpoints load correctly — run:

```bash
python realtime_demo/verify_pipeline.py
```

## Reproducing the training results

Training is done in Google Colab using the two notebooks in `notebooks/`:

1. `Feature.ipynb` — extracts MediaPipe keypoints for the ASL Citizen
   top-100 subset. Produces one `.npz` file per clip.
2. `Un.ipynb` — the full training pipeline:
   - **Cells 41–45**: motion feature construction and weak phase
     pseudo-label generation
   - **Cells 48–51**: Phase Detection TCN training and evaluation
   - **Cells 52–53**: active-region segment extraction
   - **Cells 56–62**: Recognition TCN training (phase-aware and baseline)

A T4 GPU is sufficient. The full pipeline takes approximately 4–6 hours
of total training compute.

## Known limitations

- The phase pseudo-labels are derived from a velocity-threshold heuristic,
  not from human-annotated phase boundaries. The reported Phase TCN
  accuracy of 91% measures agreement with these pseudo-labels, not with
  linguistic ground truth.
- The phase-segmentation filter is biased toward signs with single-motion
  structure. Signs with multi-peak or oscillatory motion (e.g., TWINS1,
  COMB2, PIPE2) had drop rates above 50%.
- The phase-aware improvement over the baseline (+0.9% top-1) is not
  statistically significant at the current sample size (McNemar p ≈ 0.50).
- The webcam demo is designed for one isolated sign at a time. Continuous
  signing is out of scope.

## References

Key works the design draws on (see the project report bibliography for
the complete list):

- A. Desai et al., "ASL Citizen: A Community-Sourced Dataset for
  Advancing Isolated Sign Language Recognition," NeurIPS 2023.
- S. Jiang et al., "Skeleton Aware Multi-Modal Sign Language
  Recognition," CVPRW 2021.
- C. Lea et al., "Temporal Convolutional Networks for Action Segmentation
  and Detection," CVPR 2017.
- C. Lugaresi et al., "MediaPipe: A Framework for Building Perception
  Pipelines," arXiv:1906.08172, 2019.
- S. K. Liddell and R. E. Johnson, "American Sign Language: The
  Phonological Base," *Sign Language Studies* 64, 1989.

## License

[Add a license here. For an academic capstone, a permissive license like
MIT or Apache 2.0 is common. If unsure, ask your advisor.]
