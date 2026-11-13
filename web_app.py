"""Automatic OCR API: Tesseract for print, IAM CRNN fallback for handwriting."""
from functools import lru_cache
import io
import os
from pathlib import Path

import pytesseract
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pytesseract import Output, TesseractNotFoundError

from iam_image import split_page_into_lines, deskew_image
from predict_iam import DEFAULT_MODEL_PATH, get_inference_device, load_model, predict_line_details

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
_tesseract_override = os.environ.get("TESSERACT_CMD")
if _tesseract_override:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_override
elif os.name == "nt":
    default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_tesseract.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(default_tesseract)

app = FastAPI(title="Automatic Handwriting and Printed OCR")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_model_and_device():
    device = get_inference_device()
    model_path = ROOT / DEFAULT_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing IAM checkpoint: {model_path}")
    return load_model(model_path, device), device


def try_printed_ocr(image: Image.Image):
    """Return Tesseract text and a simple confidence estimate, or None if unavailable."""
    try:
        prepared = ImageOps.autocontrast(image.convert("L"))
        data = pytesseract.image_to_data(
            prepared, lang="eng", config="--oem 3 --psm 3",
            output_type=Output.DICT,
        )
    except (TesseractNotFoundError, OSError):
        return None

    words, confidences = [], []
    for word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        word = word.strip()
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if word and confidence >= 0:
            words.append(word)
            confidences.append(confidence)

    if not words:
        return {"lines": [], "confidence": 0.0, "word_count": 0}

    # Average confidence plus a minimum word count helps avoid treating random
    # character-like detections on handwriting as reliable printed OCR.
    score = sum(confidences) / len(confidences)
    raw_text = pytesseract.image_to_string(
        prepared, lang="eng", config="--oem 3 --psm 3"
    )
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return {"lines": lines, "confidence": score, "word_count": len(words)}


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    try:
        pytesseract.get_tesseract_version()
        printed_ready, printed_error = True, None
    except (TesseractNotFoundError, OSError) as exc:
        printed_ready, printed_error = False, str(exc)

    model_path = ROOT / DEFAULT_MODEL_PATH
    return {
        "printed_ready": printed_ready,
        "printed_error": printed_error,
        "iam_checkpoint_ready": model_path.is_file(),
        "device": "CPU" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
    }


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    # First attempt machine-print OCR. Only accept it when the word-level
    # confidence is reasonably strong; otherwise fall back to the IAM model.
    printed = try_printed_ocr(image)
    if printed and printed["word_count"] >= 2 and printed["confidence"] >= 55:
        return {
            "lines": [{"number": i + 1, "text": line, "confidence": None}
                      for i, line in enumerate(printed["lines"])],
            "count": len(printed["lines"]),
            "engine": "Tesseract OCR",
            "detected_type": "printed",
            "detection_confidence": round(printed["confidence"], 1),
            "device": "CPU",
        }

    try:
        model, device = get_model_and_device()
    except FileNotFoundError as exc:
        if printed and printed["lines"]:
            # Return low-confidence Tesseract output rather than hiding a
            # possible partial result when the IAM checkpoint is unavailable.
            return {
                "lines": [{"number": i + 1, "text": line, "confidence": None}
                          for i, line in enumerate(printed["lines"])],
                "count": len(printed["lines"]),
                "engine": "Tesseract OCR (low-confidence fallback)",
                "detected_type": "uncertain",
                "detection_confidence": round(printed["confidence"], 1),
                "warning": "IAM checkpoint is missing, so handwriting fallback was unavailable.",
                "device": "CPU",
            }
        raise HTTPException(
            status_code=503,
            detail=f"{exc}. Printed OCR was not confident enough to read this image as print.",
        )

    try:
        # A wide, shallow image is most likely already a cropped text line.
        # Do not run page segmentation on it: projection-based segmentation can
        # split cursive strokes into multiple fragments and destroy word shapes.
        width, height = image.size
        is_single_line = width >= 3 * height and height <= 350
        if is_single_line:
            crops = [image]
        else:
            crops = split_page_into_lines(image)
            if not crops:
                crops = [deskew_image(image)]
        results = [
            predict_line_details(model, crop, device=device, decoder="greedy")
            for crop in crops
        ]
        return {
            "lines": [{"number": i + 1, **result} for i, result in enumerate(results)],
            "count": len(results),
            "engine": "IAM CRNN-CTC",
            "detected_type": "handwriting",
            "detection_confidence": round(printed["confidence"], 1) if printed else None,
            "device": str(device),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}")


# Local document library: uploaded originals and extracted text are stored on this computer.
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from fastapi.responses import FileResponse
from docx import Document as WordDocument
from pypdf import PdfReader

DATA_DIR = ROOT / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "documents.db"
MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
ALLOWED_DOCUMENTS = {".pdf", ".docx", ".txt", ".md"}


def init_document_store():
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                extracted_text TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents(uploaded_at)")


init_document_store()


def extract_document_text(path: Path, extension: str) -> str:
    if extension == ".pdf":
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ValueError("Password-protected PDFs are not supported.")
            except Exception as exc:
                raise ValueError("Password-protected PDFs are not supported.") from exc
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()
    if extension == ".docx":
        doc = WordDocument(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts).strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()


def document_row(row):
    return {
        "id": row[0], "filename": row[1], "extension": row[2],
        "size_bytes": row[3], "uploaded_at": row[4],
        "preview": (row[5][:280].replace("\n", " ") + ("…" if len(row[5]) > 280 else "")),
    }


@app.post("/api/documents")
async def add_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "document").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_DOCUMENTS:
        raise HTTPException(status_code=400, detail="Supported files: PDF, DOCX, TXT, and MD. Legacy .doc files must be saved as .docx first.")
    contents = await file.read(MAX_DOCUMENT_BYTES + 1)
    if len(contents) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 25 MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    doc_id = uuid.uuid4().hex
    stored_name = doc_id + extension
    stored_path = DOCUMENT_DIR / stored_name
    try:
        stored_path.write_bytes(contents)
        extracted = extract_document_text(stored_path, extension)
        if not extracted:
            raise ValueError("No selectable text was found. This may be a scanned PDF; export it with OCR first.")
        uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO documents (id, filename, stored_name, extension, size_bytes, extracted_text, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, filename, stored_name, extension, len(contents), extracted, uploaded_at),
            )
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Could not read this document: {exc}")
    return {"id": doc_id, "filename": filename, "characters": len(extracted), "message": "Document added to your local library."}


@app.get("/api/documents")
def list_documents(q: str = ""):
    with sqlite3.connect(DB_PATH) as db:
        if q.strip():
            term = "%" + q.strip().replace("%", "\\%").replace("_", "\\_") + "%"
            rows = db.execute(
                "SELECT id, filename, extension, size_bytes, uploaded_at, extracted_text FROM documents WHERE lower(filename) LIKE lower(?) OR lower(extracted_text) LIKE lower(?) ORDER BY uploaded_at DESC LIMIT 100",
                (term, term),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, filename, extension, size_bytes, uploaded_at, extracted_text FROM documents ORDER BY uploaded_at DESC LIMIT 100"
            ).fetchall()
    return {"documents": [document_row(row) for row in rows], "count": len(rows)}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            "SELECT id, filename, extension, size_bytes, uploaded_at, extracted_text FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"id": row[0], "filename": row[1], "extension": row[2], "uploaded_at": row[4], "text": row[5]}


@app.get("/api/documents/{document_id}/download")
def download_document(document_id: str):
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT filename, stored_name FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = DOCUMENT_DIR / row[1]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Stored file is missing.")
    return FileResponse(path, filename=row[0], media_type="application/octet-stream")


