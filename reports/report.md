# IAM training report

The supplied checkpoint and its metadata were inspected. The checkpoint's model-state keys load strictly into the project's `IAMCRNN` architecture, and its saved character mapping and image height match the project.

- Completed epochs: 30
- Training GPU recorded: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Best validation CER: 6.74% (epoch 25)
- Held-out test CER: 7.79%
- Test loss: 0.4188

CER is character error rate, not character accuracy. These are recorded results from the submitted run.

The trained checkpoint is distributed separately as `iam_crnn_ctc.pth` because model binaries are not stored in this Git repository. Place it in the project root to use the IAM option in Streamlit.

This recognizer predicts one cropped handwritten text line at a time. Full-page segmentation and paragraph reconstruction are not implemented yet.
