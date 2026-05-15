# Introduction

Sign language recognition is a challenging computer vision and sequence modeling problem because signs are not defined by single static hand poses alone. A sign usually contains temporal structure, including movement onset, the main expressive motion, and movement offset. In sign linguistics, these temporal regions are commonly understood as preparation, stroke, and retraction phases. Many recognition systems treat a sign video as one continuous sequence and attempt to classify the entire motion directly. This project investigates whether explicitly modeling internal signing phases can improve isolated sign recognition.

The goal of this project is to build a complete phase-aware isolated sign recognition pipeline. Instead of relying only on raw video frames, the system uses MediaPipe-based pose and hand landmarks as a compact skeleton representation. Motion features such as velocity, acceleration, and speed curves are computed from the landmark sequences. These motion cues are used to generate weak phase pseudo-labels, which are then used to train a Phase Detection TCN. The predicted phase structure is finally incorporated into a TCN-attention sign recognition model.

The main research question is:

**Does adding phase-aware temporal information improve isolated sign recognition compared with a comparable model that does not use explicit phase features?**

To answer this, the final phase-aware model is compared against a no-phase baseline using the same active-region crops, same continuous motion features, same train/validation/test split, and same TCN-attention architecture. The only major difference is the inclusion or removal of ordered phase one-hot features.
