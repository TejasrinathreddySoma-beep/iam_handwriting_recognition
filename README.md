# IAM Handwriting Recognition

Standalone handwritten text-line recognition using a CNN + bidirectional LSTM + Connectionist Temporal Classification (CTC) model trained on the IAM dataset.

## Recorded training results

The committed training run completed **30 epochs** on an NVIDIA GeForce RTX 3050 4GB Laptop GPU.

| Metric | Result |
|---|---:|
| Best validation CER | **6.74%** (epoch 25) |
| Held-out test CER | **7.79%** |
| Test loss | **0.4188** |

CER means character error rate (lower is better); it is not character accuracy. These figures are copied from the training branch's committed metrics.

### Training curves

- [IAM character error rate curve](reports/iam_cer_curve.png)
- [IAM loss curve](reports/iam_loss_curve.png)

Detailed logs and report:
- [Epoch history](reports/iam_history.json)
- [IAM metrics](reports/iam_metrics.json)
- [Training summary](TRAINING_RESULTS.json)
- [Training report](reports/report.md)

## Project files

- `iam_model.py` — CNN feature extractor, BiLSTM sequence model, CTC output, character mapping, and greedy decoder.
- `train_iam.py` — IAM dataset loading, preprocessing, training, validation, and test evaluation.
- `predict_iam.py` — load a trained checkpoint and recognize a cropped text-line image.
- `reports/` — training curves, metrics, and epoch history.

## Setup

Install PyTorch separately using the official PyTorch installer for your OS and accelerator, then install the remaining dependencies:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

Download the IAM Lines images and the matching ASCII annotation and official split files from the IAM Handwriting Database. Extract them under:

```
data/iam/
  lines.txt
  lines/
  trainset.txt
  validationset1.txt
  testset.txt
```

The loader also supports annotation/split files under `ascii/` and `ascii/sets/`. Dataset files are not included in this repository.

## Train

```bash
python train_iam.py --device auto --epochs 30 --batch-size 4
```

Use `--device cuda` to require an NVIDIA GPU or `--device cpu` to train on CPU. The best checkpoint is saved as `iam_crnn_ctc.pth` in the project root. Resume with `--resume`.

## Web application

The project includes a standalone browser interface with page/line upload. The FastAPI backend can split a page into text-line regions and run the IAM recognizer on each crop.

Start it from the repository root:

```bash
uvicorn web_app:app --reload
```

Open http://127.0.0.1:8000 in your browser. The trained checkpoint must be in the project root for recognition to run.

## Predict

Put the trained `iam_crnn_ctc.pth` file in the project root and run:

```bash
python predict_iam.py path/to/cropped_line.png
```

**Checkpoint note:** the source branch excludes `iam_crnn_ctc.pth` via `.gitignore`; it was not present in the tracked tree, so it has not been transferred. Use the locally saved checkpoint from your completed training run for inference.

This recognizer processes one cropped handwritten text line at a time. Full-page line segmentation and paragraph reconstruction are not included.
