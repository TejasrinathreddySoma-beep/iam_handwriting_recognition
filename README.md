# IAM Handwriting Recognition

A standalone IAM handwritten text-line recognition project using a convolutional neural network (CNN), bidirectional LSTM (BiLSTM), and Connectionist Temporal Classification (CTC).

## Project files

- `iam_model.py` — CRNN architecture, character vocabulary, and greedy CTC decoder.
- `train_iam.py` — IAM line dataset loader, preprocessing, training, validation, early stopping, checkpointing, and test evaluation.
- `requirements.txt` — Python dependencies for this project only.

## Dataset setup

The IAM Lines dataset and its ASCII annotations/splits are not included in this repository. Obtain them from the official [IAM Handwriting Database](https://www.iam.unibe.ch/). Arrange the extracted files under `data/iam/` in one of these supported layouts:

```text
data/iam/
├── lines.txt
├── trainset.txt
├── validationset1.txt
├── testset.txt
└── lines/
    └── a01/a01-000u/a01-000u-00.png
```

or use `ascii/lines.txt` and `ascii/sets/{trainset,validationset1,testset}.txt` where supplied by the dataset package. The loader locates annotation and split files in these locations. Keep the dataset local; it is excluded by `.gitignore`.

## Install and train

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your hardware from the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then run:

```bash
python train_iam.py
```

The script automatically selects Intel XPU, CUDA, or CPU when available. Training uses the IAM train split, selects the best checkpoint by validation character error rate (CER), applies early stopping, and evaluates the selected checkpoint on the test split. The checkpoint is written to `iam_crnn_ctc.pth` (ignored by Git).

## Evaluation and results

Training prints per-epoch training/validation loss and CER, followed by test loss and test CER. The original project version did not save a separate metrics CSV/JSON, plot, or report file, and no trained checkpoint or IAM dataset was present in the source repository for transfer. Consequently, this repository does not claim a measured accuracy/CER result. Run training locally to generate fresh evaluation results.

## Model overview

Grayscale line images are resized to a height of 64 pixels while preserving aspect ratio. A CNN extracts visual features, a two-layer bidirectional LSTM models the horizontal sequence, and a linear layer produces character logits for CTC training. Decoding uses greedy CTC collapse (removing blanks and repeated adjacent labels).

## Scope

This repository contains only the IAM line-recognition code. The separate MNIST/EMNIST digit and character-recognition project remains in `num_pred_cnn`.
