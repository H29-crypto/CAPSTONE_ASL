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
