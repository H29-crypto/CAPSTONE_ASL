# Methodology

## Dataset Preparation

The recognition dataset contains 2234 verified samples across 100 sign classes. The final recognition split contains:

- Training samples: 1215
- Validation samples: 242
- Test samples: 777

Only clean samples were used for recognition training. Warning segments, fallback-generated segments, overly short phase regions, overly broad active regions, and low-agreement phase predictions were excluded from the strict recognition dataset.

## Landmark Extraction

The system uses MediaPipe pose and hand landmarks as the base representation. For each frame, the extracted representation consists of selected pose landmarks and left/right hand landmarks. These are combined into a 147-dimensional position vector.

The position representation is then normalized to reduce the effect of subject position and scale. This makes the model focus more on relative movement patterns instead of absolute image location.

## Motion Feature Generation

From the normalized landmark sequence, the system computes temporal motion features:

- Position features
- Velocity features
- Acceleration features
- Global speed
- Hand speed
- Phase speed

The continuous feature vector used for temporal modeling has 444 dimensions:

- 147 position features
- 147 velocity features
- 147 acceleration features
- 1 global speed feature
- 1 hand speed feature
- 1 phase speed feature

## Weak Phase Pseudo-Label Generation

The project does not use manually annotated linguistic phase labels. Instead, weak pseudo-labels are generated from motion curves. The phase pseudo-labels divide each sequence into four frame-level classes:

- 0: background
- 1: preparation
- 2: stroke
- 3: retraction

The pseudo-labels are derived from the active motion region and the dominant motion peak. The goal is not to create perfect linguistic annotations, but to create useful weak supervision for a phase detection model.

## Phase Detection TCN

A Phase Detection TCN is trained to predict frame-level phase labels from the 444-dimensional motion feature sequence. The model predicts one of four phase classes for every frame.

The Phase TCN achieved:

- Test accuracy: 0.9109
- Test Macro F1: 0.8974

This result shows that the model can learn the motion-derived phase structure with high agreement.

## Phase-Aware Segment Extraction

The trained Phase TCN is used to generate phase predictions for each sequence. These predictions are smoothed and converted into ordered segment boundaries:

- active signing region
- preparation region
- stroke region
- retraction region

The final recognition model uses only the active signing region. This reduces background-heavy frames and focuses the recognition model on the meaningful signing motion.

The phase segment generation produced:

- OK segments: 2891
- Warning segments: 5

Warning segments were excluded from strict recognition training.

## Recognition TCN with Attention

The final recognition model uses a TCN encoder followed by attention pooling. The input is the active-region crop of the sequence.

The baseline model uses only the 444 continuous motion features.

The phase-aware model uses:

- 444 continuous motion features
- 4 ordered phase one-hot features

This gives a final phase-aware input size of 448 dimensions per frame.

The attention layer learns to assign importance weights to frames inside the active signing region. The pooled sequence representation is passed to a classifier to predict one of 100 sign classes.
