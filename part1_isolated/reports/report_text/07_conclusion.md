# Conclusion

This project developed a complete phase-aware isolated sign recognition pipeline using MediaPipe landmarks, motion-derived weak phase pseudo-labels, a Phase Detection TCN, phase-aware segment extraction, and a TCN-attention recognition model. The Phase TCN successfully learned the weak phase labels, achieving a test accuracy of 0.9109 and a test Macro F1 of 0.8974.

For the final 100-class recognition task, the phase-aware model achieved a test Top-1 accuracy of 0.6885, Top-5 accuracy of 0.8700, and Macro F1 of 0.6695. The no-phase baseline achieved a test Top-1 accuracy of 0.6795, Top-5 accuracy of 0.8597, and Macro F1 of 0.6574. The phase-aware model therefore produced a small improvement across all metrics.

The paired comparison showed that this improvement was modest and not statistically strong. Therefore, the final contribution should be framed carefully: the project demonstrates that phase-aware modeling is feasible and can slightly improve recognition performance, but stronger data, better phase annotations, and more robust architectures are needed to prove a larger and more conclusive benefit.
