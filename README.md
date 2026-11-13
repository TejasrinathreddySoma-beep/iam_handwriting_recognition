# IAM Handwriting + Printed Document OCR

One local web app with two selectable OCR engines:

- **Handwritten English:** your existing IAM CRNN (CNN + BiLSTM + CTC) checkpoint.
- **Printed English:** Tesseract OCR, which runs on CPU and does not require the IAM checkpoint.

The engines are selected explicitly. This prevents printed serif-font pages from being sent to a model trained on handwriting. This version does not yet automatically separate handwriting and print regions when both appear on the same page; process such content in separate crops with the matching engine.

## Existing IAM model results

| Metric | Recorded result |
|---|---:|
| Epochs completed | 30 |
| Best validation CER | **6.74%** (epoch 25) |
| Held-out test CER | **7.79%** |
| Test loss | **0.4188** |

These are historical metrics from the supplied training run, not a fresh evaluation.

## Windows installation

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/TejasrinathreddySoma-beep/iam_handwriting_recognition.git
cd iam_handwriting_recognition
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the [official PyTorch selector](https://pytorch.org/get-started/locally/). CPU-only PyTorch is sufficient for inference. Then install the project packages:

```powershell
python -m pip install -r requirements.txt
```

### Install Tesseract for printed text

Install the Windows Tesseract application from the [UB Mannheim Windows builds](https://github.com/UB-Mannheim/tesseract/wiki). The app checks this default path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

For a custom install location, set this environment variable before starting the server:

```powershell
$env:TESSERACT_CMD = "D:\path\to\tesseract.exe"
```

Note: `pytesseract` is only the Python wrapper. You must install the Tesseract application separately.

### Add your existing IAM checkpoint

Copy your compatible trained checkpoint into the repository root as:

```text
iam_crnn_ctc.pth
```

Do not commit the checkpoint to Git. It is a local model asset and is not included in this repository.

## Run

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

1. Choose **Whole page** or **Single text line**.
2. Choose **Handwritten (IAM)** for handwriting. The checkpoint is required.
3. Choose **Printed (Tesseract)** for machine-printed documents. The IAM checkpoint is not required.
4. Upload an image, recognize, then edit/copy/download the output.

The `/api/status` endpoint reports IAM checkpoint readiness and Tesseract availability independently.

## API

`POST /api/recognize` accepts multipart fields: `file` (required image), `text_type` (`handwriting` or `printed`), `mode` (`page` or `line`), `decoder` (`greedy` or `beam`), `beam_width` (1–100), and `use_language_model` (boolean). Defaults are handwriting, page, greedy, 10, and false. Decoder/LM settings apply to the IAM path only.

## Important limitations

- The repository does not include `iam_crnn_ctc.pth`; supply your compatible local checkpoint for handwriting inference.
- Printed OCR uses Tesseract English language data and is affected by scan resolution, contrast, font, and layout.
- The IAM model recognizes handwritten lines; page mode uses heuristic line segmentation and can struggle with columns, tables, skew, or touching lines.
- Mixed handwritten/printed pages are not automatically routed by region in this version.
- OCR can make mistakes. Proofread important output.
- Adding printed OCR does not alter the IAM architecture or require retraining.

## Main files

- `iam_model.py`: IAM CRNN architecture.
- `iam_image.py`: preprocessing, deskewing, and line segmentation.
- `iam_decoder.py`: greedy and prefix-beam CTC decoding.
- `predict_iam.py`: IAM inference.
- `train_iam.py`: IAM training.
- `web_app.py`: FastAPI backend for both OCR engines.
- `static/index.html`: browser interface.
- `tests/`: IAM automated tests.
