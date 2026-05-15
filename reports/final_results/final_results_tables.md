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
