# Automatic Handwriting + Printed Document OCR

Upload an image and the application automatically attempts printed OCR first, then falls back to the existing IAM handwriting recognizer when printed OCR confidence is low. No engine, page/line, or decoder selection is required in the UI.

- **Printed English:** Tesseract OCR (CPU; no IAM checkpoint needed when its result is confident).
- **Handwritten English:** existing IAM CRNN (CNN + BiLSTM + CTC) checkpoint (CPU inference supported; no retraining).

## Windows setup

Install Python dependencies (install PyTorch separately from the [official selector](https://pytorch.org/get-started/locally/); CPU is sufficient):

```powershell
python -m pip install -r requirements.txt
```

Install the Tesseract Windows application and English language data from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki). The default path is `C:\Program Files\Tesseract-OCR\tesseract.exe`. If installed elsewhere, set:

```powershell
$env:TESSERACT_CMD = "D:\path\to\tesseract.exe"
```

For handwriting, copy your already-trained compatible checkpoint into the repository root as `iam_crnn_ctc.pth`. Do not commit the checkpoint to Git.

## Run

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload an image, and recognition starts automatically. The result can be edited, copied, or downloaded. `/api/status` reports whether Tesseract and the local IAM checkpoint are available.

## Automatic routing behavior

The backend first runs Tesseract page OCR and checks word-level confidence. When it finds at least two words and average confidence is at least 55/100, it returns the printed OCR result. Otherwise it uses IAM page-line segmentation and greedy CTC decoding. If the IAM checkpoint is absent and Tesseract has a low-confidence partial result, that partial result is returned with a warning; if neither path can produce output, the API explains what is missing.

This is a practical page-level automatic choice, not a perfect mixed-page segment classifier. A page containing both handwriting and print may need a future region-level hybrid pipeline to route each separate text region to its best recognizer. OCR results should be proofread.

## API

`POST /api/recognize`: multipart image field `file`. Engine selection and page mode are automatic. `GET /api/status`: engine/checkpoint availability.

## Main files

- `iam_model.py`: IAM CRNN architecture.
- `iam_image.py`: preprocessing, deskewing, and line segmentation.
- `iam_decoder.py`: CTC decoding utilities.
- `predict_iam.py`: IAM inference.
- `train_iam.py`: IAM training (not needed for inference).
- `web_app.py`: automatic OCR backend.
- `static/index.html`: upload-first interface.
- `tests/`: IAM tests.
