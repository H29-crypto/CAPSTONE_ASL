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
