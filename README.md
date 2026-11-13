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

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and use the single upload area for images, PDFs, DOCX, TXT, or Markdown. The app extracts or recognizes text, saves the original and extracted text to your local library, and displays the result. `/api/status` reports whether Tesseract and the local IAM checkpoint are available.

## Automatic routing behavior

The backend first runs Tesseract page OCR and checks word-level confidence. When it finds at least two words and average confidence is at least 55/100, it returns the printed OCR result. Otherwise it uses IAM page-line segmentation and greedy CTC decoding. If the IAM checkpoint is absent and Tesseract has a low-confidence partial result, that partial result is returned with a warning; if neither path can produce output, the API explains what is missing.

This is a practical page-level automatic choice, not a perfect mixed-page segment classifier. A page containing both handwriting and print may need a future region-level hybrid pipeline to route each separate text region to its best recognizer. OCR results should be proofread.

## API

`POST /api/upload`: multipart file field `file`; processes supported images and documents and saves them to the library. `POST /api/recognize` remains available for image-only OCR clients. `GET /api/status`: engine/checkpoint availability.

## Main files

- `iam_model.py`: IAM CRNN architecture.
- `iam_image.py`: preprocessing, deskewing, and line segmentation.
- `iam_decoder.py`: CTC decoding utilities.
- `predict_iam.py`: IAM inference.
- `train_iam.py`: IAM training (not needed for inference).
- `web_app.py`: automatic OCR backend.
- `static/index.html`: upload-first interface.
- `tests/`: IAM tests.


## Personal document library

The web app also includes a local document library. Add PDF, DOCX, TXT, or Markdown files; search filenames and extracted contents; open extracted text; and download the original file later.

- Maximum upload size: 25 MB per file.
- Original files and the SQLite catalogue are stored under `data/documents/` and `data/documents.db`.
- The `data/` directory is ignored by Git, so private documents stay on your machine and are not committed to the repository.
- Text-based PDFs and DOCX files are supported. Scanned PDFs with no selectable text are rendered page by page and sent through the OCR pipeline (requires PyMuPDF).
- Legacy binary `.doc` files are not supported directly; open and save them as `.docx` in Word or LibreOffice.
- This is a local, single-user library without login or cloud sync. Keep backups of your `data/` folder.

Document endpoints: `POST /api/documents`, `GET /api/documents?q=...`, `GET /api/documents/{id}`, and `GET /api/documents/{id}/download`.


## Download and export formats

Each saved upload keeps its original file available through the **Original** download link. The extracted or recognized text can also be exported as:

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain text (`.txt`)
- Markdown (`.md`)
- HTML (`.html`)
- Rich Text Format (`.rtf`)

Export links are shown beside each item in the document library and in the recognition result. PDF generation uses ReportLab; Word generation uses python-docx. Install all dependencies with `python -m pip install -r requirements.txt`.


### Camera-photo OCR preprocessing

Image uploads now attempt automatic page orientation correction before OCR. The pipeline uses Tesseract orientation detection when available and compares 0°/180° printed-text evidence as a fallback. This helps with upside-down phone photos, but it does not guarantee accurate handwriting recognition. Perspective correction, page-boundary detection, and handwriting model accuracy remain dependent on image quality and the IAM model's training domain.
