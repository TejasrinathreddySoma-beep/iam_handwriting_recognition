# IAM Handwritten Text-Line Recognition

A standalone handwritten text recognition system trained on the **IAM Handwriting Database (Lines)**. It reads a cropped handwritten line and predicts its transcription using a Convolutional Recurrent Neural Network (CRNN): a CNN extracts visual features, a bidirectional LSTM models the character sequence, and Connectionist Temporal Classification (CTC) learns alignment without requiring character-by-character segmentation.

The project includes training, single-line prediction, and a small web interface for uploading a line or a page image.

## Results from the completed training run

| Metric | Recorded result |
|---|---:|
| Epochs completed | 30 |
| Best validation CER | **6.74%** (epoch 25) |
| Held-out test CER | **7.79%** |
| Test loss | **0.4188** |
| Training device | NVIDIA GeForce RTX 3050 4GB Laptop GPU |

**CER (character error rate)** is the edit distance between predicted and reference text divided by the number of reference characters. Lower is better; CER is not the same as character accuracy. These figures are taken from the committed training-run metadata, not freshly rerun as part of this repository cleanup.

Training artifacts:
- [CER curve](reports/iam_cer_curve.png)
- [Loss curve](reports/iam_loss_curve.png)
- [Epoch history](reports/iam_history.json)
- [Detailed metrics](reports/iam_metrics.json)
- [Training summary](TRAINING_RESULTS.json)
- [Run notes](reports/report.md)

## How the system works

1. **Dataset parsing:** reads IAM line annotations and the official train, validation, and test split files.
2. **Preprocessing:** converts each scanned line to grayscale, improves contrast, resizes it to a fixed height, and preserves its aspect ratio.
3. **Batching:** pads variable-width lines to a common batch width and retains their original widths for calculating CTC input lengths.
4. **Visual encoder:** convolution and pooling layers turn each image into a left-to-right sequence of visual feature vectors.
5. **Sequence model:** a two-layer bidirectional LSTM learns context across the line.
6. **CTC objective:** CTC loss trains the sequence predictor without character-level bounding boxes.
7. **Decoding and evaluation:** greedy CTC decoding collapses repeated symbols and blanks; CER is computed using edit distance.
8. **Inference:** the predictor applies the same preprocessing and character mapping used during training.

## Challenges addressed

- **No character-level segmentation:** IAM provides line images and transcriptions, not a crop for every character. CTC allows the model to learn alignment between image features and text directly.
- **Variable line widths:** handwriting lines have different lengths. The collate function pads each batch while keeping the original widths so the CTC loss only evaluates valid time steps.
- **CTC length constraints:** a line must have enough output time steps for its target. The training loop checks input lengths against target lengths and reports a clear error when a line cannot fit.
- **Inconsistent scans:** grayscale conversion and contrast normalization provide a consistent input pipeline.
- **Unstable sequence training:** gradient clipping, a learning-rate scheduler, finite-loss checks, validation CER, best-checkpoint saving, and early stopping help stabilize training and select the best epoch.
- **Different compute hardware:** device selection supports CUDA, Intel XPU, Apple MPS, and CPU, with explicit device options.
- **Page layout versus line recognition:** the recognizer is line-based. The web app includes a simple horizontal projection-based line splitter for page uploads, but complex layouts, skew, tables, and overlapping lines can still require manual cropping or better document-layout analysis.

## Repository structure

```text
iam_handwriting_recognition/
├── iam_model.py              # CNN + BiLSTM + CTC architecture
├── iam_image.py               # Deskewing and page-to-line segmentation
├── iam_decoder.py             # Greedy and prefix-beam CTC decoding
├── train_iam.py               # Dataset, training, validation and test evaluation
├── predict_iam.py             # Single-line inference and confidence
├── evaluate_decoders.py       # Greedy/beam CER comparison on validation data
├── web_app.py                 # FastAPI backend
├── static/
│   └── index.html             # Browser upload, decoder controls and confidence
├── scripts/
│   └── package_checkpoint.py  # Versioned model-release packaging
├── tests/
│   └── test_iam.py            # Automated unit and inference tests
├── .github/workflows/
│   └── tests.yml              # CI test workflow
├── reports/
│   ├── iam_cer_curve.png
│   ├── iam_loss_curve.png
│   ├── iam_history.json
│   ├── iam_metrics.json
│   └── report.md
├── TRAINING_RESULTS.json
├── requirements.txt
└── .gitignore
```

The IAM dataset and trained checkpoint are not committed. They are large or access-controlled assets and should be kept locally.

## Implementation guide

### 1. Clone the repository

```bash
git clone https://github.com/TejasrinathreddySoma-beep/iam_handwriting_recognition.git
cd iam_handwriting_recognition
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch and project dependencies

Install a PyTorch build that matches your operating system and hardware using the [official PyTorch selector](https://pytorch.org/get-started/locally/). For NVIDIA GPU training, choose a CUDA-enabled build compatible with your driver. CPU-only PyTorch is also supported.

Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

### 4. Download and arrange the IAM Lines dataset

Obtain the IAM Handwriting Database and its line images, annotation file, and official split files from the [official IAM database website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). Follow the dataset's access and usage terms.

The training loader expects this layout (it also checks the common `ascii/` and `ascii/sets/` alternatives for annotation and split files):

```text
data/iam/
├── lines.txt
├── trainset.txt
├── validationset1.txt
├── testset.txt
├── a01/
│   └── a01-000u/
│       └── a01-000u-00.png
└── ...
```

If your archive stores the line images inside a `lines/` directory, that layout is also supported. The dataset is excluded by `.gitignore`; do not commit the IAM images or private credentials.

### 5. Train or reproduce the model

To train from scratch:

```bash
python train_iam.py --device auto --epochs 30 --batch-size 4
```

Choose a device explicitly when needed:

```bash
python train_iam.py --device cuda
python train_iam.py --device cpu
```

The script selects the best checkpoint by validation CER and writes `iam_crnn_ctc.pth` in the repository root. To continue from an existing checkpoint, use:

```bash
python train_iam.py --resume --device auto --epochs 30
```

The default batch size is 4 and the default data-loader worker count is 0 for a safe Windows setup. Training time depends on the dataset, GPU, and batch size. The trainer now checks CTC's repeated-label alignment requirement as well as the target length.

### 6. Predict one cropped line

After training, run:

```bash
python predict_iam.py path/to/handwritten_line.png
```

Or specify a checkpoint path:

```bash
python predict_iam.py path/to/handwritten_line.png --model iam_crnn_ctc.pth
```

The input should contain one text line. The model returns a plain transcription; inspect the result because handwriting recognition is not error-free.

### 7. Launch the web application

Start the FastAPI server from the repository root:

```bash
uvicorn web_app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload either a single line or a page image. In page mode, the backend estimates horizontal text bands using row-wise ink projection and recognizes each detected line.

The checkpoint `iam_crnn_ctc.pth` must be present in the repository root. The `/api/status` endpoint reports whether the model is ready.

## Important limitations

- The checked-in repository does **not** include `iam_crnn_ctc.pth`. The source training branch excluded it from Git. Train the model or copy your own compatible checkpoint into the project root before running inference.
- Page segmentation is a lightweight heuristic, not a full document-layout engine. Skewed pages, columns, tables, touching lines, or noisy backgrounds may need preprocessing or manual line crops.
- The model recognizes a single line at a time; it does not reconstruct page structure, paragraphs, or reading order across multiple columns.
- Greedy decoding remains the fast default. Prefix beam search and optional character bigram rescoring are available; compare their CER on the validation split before selecting a decoder.
- Results can vary with handwriting style, scan quality, and dataset split. The published metrics above describe the supplied run, not a guarantee for arbitrary documents.

## Implemented improvements

### Deskewing and full-page line segmentation

`iam_image.py` estimates small page rotation angles using horizontal ink projections, corrects skew, and extracts text lines using adaptive local-background thresholding and row projections. A projection-valley fallback handles some merged bands. This is designed for ordinary single-column pages; complex layouts still need a dedicated document-layout model.

### Greedy and beam-search decoding

`iam_decoder.py` includes greedy CTC decoding and prefix beam search. Beam search can optionally rescore character transitions with an add-one-smoothed character bigram model. Greedy remains the default for speed.

Compare decoder quality on the official validation split:

```bash
python evaluate_decoders.py --beam-width 10
```

To test optional language-model rescoring, prepare a UTF-8 text corpus from permitted training text (do not use validation/test transcriptions to build the LM), then run:

```bash
python evaluate_decoders.py --beam-width 10 --language-model path/to/training_corpus.txt --lm-weight 0.25
```

The script writes `decoder_comparison.json` with greedy and beam CER. Tune beam width and LM weight on validation data only, then report final test results once.

### Confidence and uncertainty indicators

Prediction returns a confidence-like score with its interpretation. Greedy reports mean maximum per-frame probability; beam search reports a normalized sequence ranking score. Both are explicitly marked **uncalibrated** and must not be interpreted as a statistically calibrated probability that the transcription is correct. The web UI displays these values for each line.

### Versioned checkpoint packaging

The checkpoint packager validates the model's character mapping and image height, then creates a versioned ZIP with a model manifest:

```bash
python scripts/package_checkpoint.py --checkpoint iam_crnn_ctc.pth --version v1.0.0
```

After creating the archive, install and authenticate the GitHub CLI, then publish it as a release asset:

```bash
gh release create v1.0.0 dist/iam-crnn-ctc-v1.0.0.zip --title "IAM CRNN-CTC v1.0.0" --notes "Versioned IAM handwriting model and manifest."
```

The trained checkpoint is not stored in this source repository. Run training or supply your compatible local checkpoint before packaging. Do not commit the raw checkpoint into Git history.

### Automated tests and continuous integration

The `tests/` suite covers IAM annotation and split parsing, image preprocessing, deskew/page segmentation, CTC output and repeated-label lengths, decoder behavior, and inference confidence. GitHub Actions runs the test suite on pushes and pull requests to `main`.

Install test dependencies and run locally:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

These additions provide implementation and evaluation tooling; beam/LM performance improvements must be measured on your validation data rather than assumed.

## References

- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [CTC paper: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
