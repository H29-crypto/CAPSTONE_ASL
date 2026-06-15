# Discussion

The final results show that the phase-aware representation provides a small but consistent improvement over the no-phase baseline. The phase-aware model improved Top-1 accuracy, Top-5 accuracy, and Macro F1. This supports the idea that explicit temporal phase information can help isolated sign recognition.

However, the improvement is modest. The paired analysis showed that the phase-aware model fixed 44 samples that the baseline missed, but also caused 37 samples to become incorrect. The net Top-1 gain was only 7 test samples out of 777. The approximate McNemar p-value was 0.5050, so the improvement should not be described as statistically significant.

The results suggest that phase-aware features are useful for some signs but harmful for others. Some classes improved substantially when phase information was added, while other classes lost performance. This may happen because weak phase pseudo-labels do not always correspond perfectly to the linguistically meaningful parts of a sign. In some cases, the model may over-rely on phase boundaries that are noisy or not discriminative for that class.

The strongest part of the project is the complete end-to-end pipeline. The system successfully moves from landmark extraction to motion features, phase pseudo-labeling, phase detection, segment extraction, recognition, baseline comparison, and paired error analysis. This gives the project a clear experimental structure and makes the final conclusion defensible.
