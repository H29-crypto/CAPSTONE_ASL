# Abstract

This project presents a phase-aware isolated sign recognition pipeline using pose and hand landmarks extracted from video data. The system first converts sign videos into normalized landmark sequences, computes temporal motion features, generates weak phase pseudo-labels corresponding to preparation, stroke, and retraction phases, and trains a Phase Detection Temporal Convolutional Network (TCN) to estimate frame-level signing phases. These predicted phases are then used to extract active signing segments and to construct phase-aware features for a final sign recognition model.

The final recognition model uses a TCN encoder with attention pooling over active signing frames. The phase-aware version includes ordered phase one-hot features, while the baseline uses the same active-region crop and continuous motion features but removes the phase one-hot representation. On the 100-class recognition task, the phase-aware model achieved a test Top-1 accuracy of 0.6885, Top-5 accuracy of 0.8700, and Macro F1 score of 0.6695. The no-phase baseline achieved a test Top-1 accuracy of 0.6795, Top-5 accuracy of 0.8597, and Macro F1 score of 0.6574. The phase-aware representation therefore produced a small but consistent improvement across all three metrics. However, paired analysis showed that the Top-1 improvement was not statistically strong, so the contribution should be interpreted as a promising phase-aware design rather than a conclusive performance breakthrough.


# Introduction

Sign language recognition is a challenging computer vision and sequence modeling problem because signs are not defined by single static hand poses alone. A sign usually contains temporal structure, including movement onset, the main expressive motion, and movement offset. In sign linguistics, these temporal regions are commonly understood as preparation, stroke, and retraction phases. Many recognition systems treat a sign video as one continuous sequence and attempt to classify the entire motion directly. This project investigates whether explicitly modeling internal signing phases can improve isolated sign recognition.

The goal of this project is to build a complete phase-aware isolated sign recognition pipeline. Instead of relying only on raw video frames, the system uses MediaPipe-based pose and hand landmarks as a compact skeleton representation. Motion features such as velocity, acceleration, and speed curves are computed from the landmark sequences. These motion cues are used to generate weak phase pseudo-labels, which are then used to train a Phase Detection TCN. The predicted phase structure is finally incorporated into a TCN-attention sign recognition model.

The main research question is:

**Does adding phase-aware temporal information improve isolated sign recognition compared with a comparable model that does not use explicit phase features?**

To answer this, the final phase-aware model is compared against a no-phase baseline using the same active-region crops, same continuous motion features, same train/validation/test split, and same TCN-attention architecture. The only major difference is the inclusion or removal of ordered phase one-hot features.


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


# Experiments and Results

## Phase Detection Experiment

The Phase Detection TCN was evaluated on the clean test set. It achieved:

| Metric | Value |
|---|---:|
| Test accuracy | 0.9109 |
| Test Macro F1 | 0.8974 |
| Test sequences | 1020 |
| Test frames | 81953 |

This confirms that the motion-derived weak phase labels can be learned reliably by a temporal convolutional model.

## Recognition Experiment

Two recognition models were compared:

1. **Baseline TCN-Attention model:** active-region crop + 444 continuous features.
2. **Phase-aware TCN-Attention model:** active-region crop + 444 continuous features + 4 ordered phase one-hot features.

| Model | Test Top-1 | Test Top-5 | Test Macro F1 |
|---|---:|---:|---:|
| Baseline without phase one-hot | 0.6795 | 0.8597 | 0.6574 |
| Phase-aware with phase one-hot | 0.6885 | 0.8700 | 0.6695 |
| Delta | 0.0090 | 0.0103 | 0.0120 |

The phase-aware model improved all three test metrics:

- Top-1 improvement: 0.0090
- Top-5 improvement: 0.0103
- Macro F1 improvement: 0.0120

## Paired Error Analysis

A paired comparison was performed to check whether the same test samples were corrected or worsened by the phase-aware model.

| Category | Count |
|---|---:|
| Both models correct | 491 |
| Phase-aware only correct | 44 |
| Baseline only correct | 37 |
| Both models wrong | 205 |
| Net phase-aware gain | 7 |

The phase-aware model fixed 44 samples that the baseline missed, but it also hurt 37 samples that the baseline predicted correctly. The net gain was therefore 7 test samples.

The approximate McNemar p-value was 0.5050. This means the paired Top-1 improvement was not statistically strong.

## Attention Analysis

The recognition attention mechanism showed that the model does not only focus on the stroke phase. The average attention mass was distributed across preparation, stroke, and retraction. This suggests that the temporal context around the main sign motion contributes to recognition.


# Discussion

The final results show that the phase-aware representation provides a small but consistent improvement over the no-phase baseline. The phase-aware model improved Top-1 accuracy, Top-5 accuracy, and Macro F1. This supports the idea that explicit temporal phase information can help isolated sign recognition.

However, the improvement is modest. The paired analysis showed that the phase-aware model fixed 44 samples that the baseline missed, but also caused 37 samples to become incorrect. The net Top-1 gain was only 7 test samples out of 777. The approximate McNemar p-value was 0.5050, so the improvement should not be described as statistically significant.

The results suggest that phase-aware features are useful for some signs but harmful for others. Some classes improved substantially when phase information was added, while other classes lost performance. This may happen because weak phase pseudo-labels do not always correspond perfectly to the linguistically meaningful parts of a sign. In some cases, the model may over-rely on phase boundaries that are noisy or not discriminative for that class.

The strongest part of the project is the complete end-to-end pipeline. The system successfully moves from landmark extraction to motion features, phase pseudo-labeling, phase detection, segment extraction, recognition, baseline comparison, and paired error analysis. This gives the project a clear experimental structure and makes the final conclusion defensible.


# Limitations and Future Work

## Limitations

1. **Weak phase labels:** The preparation, stroke, and retraction labels were generated from motion curves rather than manually annotated by sign language experts. Therefore, they should be described as motion-derived weak pseudo-labels, not ground-truth linguistic phase labels.

2. **Small number of samples per class:** The final recognition dataset contains 100 classes but only 1215 training samples. This creates a difficult learning problem and increases the risk of overfitting.

3. **Generalization gap:** The validation performance was higher than the test performance. This suggests that the model learned the training/validation distribution well but had more difficulty generalizing to the test set.

4. **Phase features are not always helpful:** Some classes improved with phase-aware features, but others got worse. This means phase-aware modeling is promising but not universally beneficial in the current form.

5. **No real-time deployment yet:** The current project is evaluated offline using processed sequences. Real-time webcam inference remains a future deployment step.

## Future Work

1. **Increase dataset size:** More samples per class would likely improve recognition stability and reduce overfitting.

2. **Manual or semi-automatic phase correction:** Human-corrected phase labels could improve the Phase TCN and make phase boundaries more meaningful.

3. **Signer-independent evaluation:** If signer metadata is available, future experiments should use signer-independent train/test splits to better measure generalization.

4. **Stronger sequence models:** The TCN-attention model can be compared with BiLSTM, Transformer encoder, ST-GCN, or hybrid graph-temporal models.

5. **Real-time webcam demo:** The model can be adapted to real-time inference using a rolling landmark buffer.

6. **YOLO-assisted signer localization:** YOLO can be added before MediaPipe in the live pipeline to localize the signer and crop the person region, especially in cluttered scenes.


# Conclusion

This project developed a complete phase-aware isolated sign recognition pipeline using MediaPipe landmarks, motion-derived weak phase pseudo-labels, a Phase Detection TCN, phase-aware segment extraction, and a TCN-attention recognition model. The Phase TCN successfully learned the weak phase labels, achieving a test accuracy of 0.9109 and a test Macro F1 of 0.8974.

For the final 100-class recognition task, the phase-aware model achieved a test Top-1 accuracy of 0.6885, Top-5 accuracy of 0.8700, and Macro F1 of 0.6695. The no-phase baseline achieved a test Top-1 accuracy of 0.6795, Top-5 accuracy of 0.8597, and Macro F1 of 0.6574. The phase-aware model therefore produced a small improvement across all metrics.

The paired comparison showed that this improvement was modest and not statistically strong. Therefore, the final contribution should be framed carefully: the project demonstrates that phase-aware modeling is feasible and can slightly improve recognition performance, but stronger data, better phase annotations, and more robust architectures are needed to prove a larger and more conclusive benefit.




# Result Tables



# Final Results Tables

## Project Phase Status

| phase    | name                                    | status   | main_output                                    |
|:---------|:----------------------------------------|:---------|:-----------------------------------------------|
| Phase 0  | Reference repo / project path planning  | Done     | Project direction finalized                    |
| Phase 1  | MediaPipe keypoint extraction           | Done     | Pose/hand keypoints extracted                  |
| Phase 2  | Keypoint manifest + normalization       | Done     | Normalized 147D landmark representation        |
| Phase 3  | Motion features                         | Done     | Velocity, acceleration, speed curves           |
| Phase 4  | Motion curve analysis                   | Done     | Improved phase signal                          |
| Phase 5  | Weak phase pseudo-label generation      | Done     | Preparation/stroke/retraction pseudo-labels    |
| Phase 6  | Phase label visualization               | Done     | Visual sanity checks                           |
| Phase 7  | Clean phase-label manifest              | Done     | Strict validated phase-label set               |
| Phase 8  | Phase Detection TCN                     | Done     | Frame-level phase classifier                   |
| Phase 9  | Phase-aware segment extraction          | Done     | Ordered active/prep/stroke/retraction segments |
| Phase 10 | Phase-aware Recognition TCN + Attention | Done     | 100-class sign recognizer                      |
| Phase 11 | Baseline + paired error analysis        | Done     | Ablation comparison against no-phase baseline  |
| Phase 12 | Final report packaging                  | Current  | Report-ready results and summary tables        |
| Phase 13 | Offline / real-time demo                | Next     | Saved-video inference, then webcam demo        |

## Recognition Dataset Summary

| Item | Value |
|---|---:|
| Recognition samples | 2234 |
| Recognition classes | 100 |
| Train samples | 1215 |
| Validation samples | 242 |
| Test samples | 777 |

## Phase Detection TCN Test Summary

| Metric | Value |
|---|---:|
| Test accuracy | 0.9109 |
| Test macro F1 | 0.8974 |
| Test sequences | 1020 |
| Test frames | 81953 |

## Recognition Model Comparison

| model                            |   val_top1 |   val_top5 |   val_macro_f1 |   test_top1 |   test_top5 |   test_macro_f1 |   test_sequences |
|:---------------------------------|-----------:|-----------:|---------------:|------------:|------------:|----------------:|-----------------:|
| baseline_no_phase_onehot         |  0.809917  |   0.929752 |     0.80765    |  0.679537   |    0.859717 |        0.657445 |              777 |
| phase_aware_plus_phase_onehot    |  0.826446  |   0.929752 |     0.814734   |  0.688546   |    0.870013 |        0.669478 |              777 |
| delta_phase_aware_minus_baseline |  0.0165289 |   0        |     0.00708369 |  0.00900901 |    0.010296 |        0.012033 |              777 |

## Paired Top-1 Comparison

| Item | Value |
|---|---:|
| Both correct | 491 |
| Phase-aware only correct | 44 |
| Baseline only correct | 37 |
| Both wrong | 205 |
| Net phase-aware gain | 7 |
| McNemar chi2 approx | 0.4444 |
| McNemar p approx | 0.5050 |

## Paired Top-5 Comparison

| Item | Value |
|---|---:|
| Both Top-5 correct | 647 |
| Phase-aware only Top-5 correct | 29 |
| Baseline only Top-5 correct | 21 |
| Both Top-5 wrong | 80 |
| Net phase-aware Top-5 gain | 8 |

## Classes Most Improved by Phase-Aware Features

|   rec_label_id | gloss        |   support_phase |   f1_base |   f1_phase |   delta_f1 |   precision_base |   precision_phase |   recall_base |   recall_phase |
|---------------:|:-------------|----------------:|----------:|-----------:|-----------:|-----------------:|------------------:|--------------:|---------------:|
|             19 | CHEESEGRATER |               4 |  0        |   0.4      |   0.4      |         0        |          0.333333 |      0        |       0.5      |
|             58 | MAPLE        |               3 |  0.5      |   0.8      |   0.3      |         1        |          1        |      0.333333 |       0.666667 |
|             71 | PATIENT2     |               8 |  0.533333 |   0.8      |   0.266667 |         0.571429 |          0.857143 |      0.5      |       0.75     |
|             82 | SHOCKED      |              10 |  0.625    |   0.823529 |   0.198529 |         0.833333 |          1        |      0.5      |       0.7      |
|             94 | TRACTOR      |               3 |  0.666667 |   0.857143 |   0.190476 |         0.5      |          0.75     |      1        |       1        |
|             27 | DEMAND1      |               7 |  0.75     |   0.923077 |   0.173077 |         0.666667 |          1        |      0.857143 |       0.857143 |
|             21 | CLOSE        |               7 |  0.666667 |   0.833333 |   0.166667 |         0.545455 |          1        |      0.857143 |       0.714286 |
|             76 | RAZOR2       |               3 |  0.5      |   0.666667 |   0.166667 |         1        |          0.666667 |      0.333333 |       0.666667 |
|              2 | APPLE        |               2 |  0.5      |   0.666667 |   0.166667 |         0.5      |          1        |      0.5      |       0.5      |
|             46 | HOPE         |               7 |  0.615385 |   0.769231 |   0.153846 |         0.666667 |          0.833333 |      0.571429 |       0.714286 |

## Classes Most Hurt by Phase-Aware Features

|   rec_label_id | gloss          |   support_phase |   f1_base |   f1_phase |   delta_f1 |   precision_base |   precision_phase |   recall_base |   recall_phase |
|---------------:|:---------------|----------------:|----------:|-----------:|-----------:|-----------------:|------------------:|--------------:|---------------:|
|              1 | ANYONE         |               5 |  0.285714 |   0        |  -0.285714 |         0.5      |          0        |      0.2      |       0        |
|              3 | ARTICULATESIGN |               7 |  0.714286 |   0.461538 |  -0.252747 |         0.714286 |          0.5      |      0.714286 |       0.428571 |
|             84 | SINK           |               9 |  0.625    |   0.4      |  -0.225    |         0.714286 |          0.5      |      0.555556 |       0.333333 |
|             85 | SLIDE2         |               5 |  0.5      |   0.285714 |  -0.214286 |         0.666667 |          0.5      |      0.4      |       0.2      |
|             48 | HOW1           |               6 |  0.857143 |   0.666667 |  -0.190476 |         0.75     |          0.555556 |      1        |       0.833333 |
|             95 | TWINS1         |               5 |  0.545455 |   0.375    |  -0.170455 |         0.5      |          0.272727 |      0.6      |       0.6      |
|             75 | PRESS          |               7 |  0.833333 |   0.666667 |  -0.166667 |         1        |          0.625    |      0.714286 |       0.714286 |
|             92 | THIRD1         |               6 |  0.666667 |   0.5      |  -0.166667 |         1        |          1        |      0.5      |       0.333333 |
|             93 | TOMATO         |               8 |  0.666667 |   0.5      |  -0.166667 |         1        |          0.75     |      0.5      |       0.375    |
|             68 | OFFEND         |               5 |  0.615385 |   0.461538 |  -0.153846 |         0.5      |          0.375    |      0.8      |       0.6      |




# Figure Captions



# Final Figure Captions

## Figure 1. Recognition Model Comparison

**File:** `fig_01_recognition_model_comparison.png`

Comparison of baseline TCN-attention recognition model and phase-aware TCN-attention recognition model on the test set.

## Figure 2. Phase-Aware Improvement

**File:** `fig_02_phase_aware_delta.png`

Absolute test-set improvement of the phase-aware model over the no-phase baseline.

## Figure 3. Paired Prediction Comparison

**File:** `fig_03_paired_prediction_comparison.png`

Paired test-set comparison showing which samples were correct under both models, only the phase-aware model, only the baseline, or neither.

## Figure 4. Per-Class F1 Delta

**File:** `fig_04_per_class_f1_delta.png`

Classes most improved and most hurt by adding ordered phase one-hot features.

## Figure 5. Attention Mass by Phase

**File:** `fig_05_attention_mass_by_phase.png`

Average attention mass assigned by the recognition model to background, preparation, stroke, and retraction regions.

## Figure 6. Final Pipeline Summary

**File:** `fig_06_final_pipeline_summary.png`

Combined summary of the Phase Detection TCN and final recognition models.
