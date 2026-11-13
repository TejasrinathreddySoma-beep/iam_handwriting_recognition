"""Automatic OCR API: Tesseract for print, IAM CRNN fallback for handwriting."""
from functools import lru_cache
import io
import os
from pathlib import Path

import pytesseract
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pytesseract import Output, TesseractNotFoundError

from iam_image import split_page_into_lines, deskew_image, rectify_page, remove_notebook_lines
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


def auto_orient_image(image: Image.Image) -> Image.Image:
    """Correct common 90/180/270 degree camera rotations before OCR."""
    image = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
        rotation = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0) or 0)
        if rotation in (90, 180, 270) and confidence >= 1.0:
            # Tesseract's OSD reports the clockwise correction angle.
            return image.rotate(-rotation, expand=True, fillcolor=255)
        if rotation == 0 and confidence >= 1.0:
            return image
    except Exception:
        pass

    # OSD often struggles with handwriting. Compare both half-turns using
    # word-level OCR evidence; never rotate solely on a weak character guess.
    try:
        upside_down = image.rotate(180, expand=True, fillcolor=255)
        normal_score = _printed_orientation_score(image)
        flipped_score = _printed_orientation_score(upside_down)
        if flipped_score > normal_score * 1.15 and flipped_score >= 8.0:
            return upside_down
    except Exception:
        pass
    return image


def _printed_orientation_score(image: Image.Image) -> float:
    data = pytesseract.image_to_data(
        image, lang="eng", config="--oem 3 --psm 11", output_type=Output.DICT
    )
    score = 0.0
    for word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        word = word.strip()
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if word and confidence >= 0:
            score += confidence * min(len(word), 8) / 8
    return score


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
        image = auto_orient_image(image)
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
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape as xml_escape
from urllib.parse import quote

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



def save_to_library(filename: str, extension: str, contents: bytes, extracted: str):
    doc_id = uuid.uuid4().hex
    stored_name = doc_id + extension
    stored_path = DOCUMENT_DIR / stored_name
    stored_path.write_bytes(contents)
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO documents (id, filename, stored_name, extension, size_bytes, extracted_text, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, filename, stored_name, extension, len(contents), extracted, uploaded_at),
            )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    return doc_id


def recognize_handwriting_image(image: Image.Image):
    model, device = get_model_and_device()
    image = rectify_page(image)
    width, height = image.size
    if width >= 3 * height and height <= 350:
        crops = [image]
    else:
        crops = split_page_into_lines(image)
        if not crops:
            crops = [deskew_image(image)]
    results = [predict_line_details(model, crop, device=device, decoder="beam", beam_width=15) for crop in crops]
    return "\\n".join(item.get("text", "") for item in results), "IAM CRNN-CTC"


def recognize_uploaded_image(image: Image.Image):
    image = auto_orient_image(image)
    printed = try_printed_ocr(image)
    if printed and printed["word_count"] >= 2 and printed["confidence"] >= 55:
        return "\\n".join(printed["lines"]), "Tesseract OCR"
    try:
        text, engine = recognize_handwriting_image(image)
        if text.strip():
            return text, engine
    except FileNotFoundError:
        if printed and printed["lines"]:
            return "\\n".join(printed["lines"]), "Tesseract OCR (low-confidence fallback)"
        raise
    if printed and printed["lines"]:
        return "\\n".join(printed["lines"]), "Tesseract OCR (low-confidence fallback)"
    return "", "No text detected"


@app.post("/api/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """One upload endpoint: process file, save original and extracted text."""
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    supported = image_extensions | {".pdf", ".docx", ".txt", ".md"}
    if extension not in supported:
        raise HTTPException(status_code=400, detail="Supported formats: images, PDF, DOCX, TXT, and MD.")
    contents = await file.read(MAX_DOCUMENT_BYTES + 1)
    if len(contents) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="Maximum upload size is 25 MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    temp_path = DOCUMENT_DIR / (uuid.uuid4().hex + extension)
    extracted = ""
    engine = "Text extraction"
    try:
        temp_path.write_bytes(contents)
        if extension in image_extensions:
            image = Image.open(io.BytesIO(contents)).convert("L")
            extracted, engine = recognize_uploaded_image(image)
        elif extension == ".pdf":
            reader = PdfReader(str(temp_path))
            pages_text = [(page.extract_text() or "").strip() for page in reader.pages]
            extracted = "\\n\\n".join(text for text in pages_text if text).strip()
            if extracted:
                engine = "PDF text extraction"
            else:
                # Image-only/scanned PDF: render pages and pass each through OCR.
                import fitz
                pdf = fitz.open(stream=contents, filetype="pdf")
                page_results = []
                for page in pdf:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    page_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                    page_text, page_engine = recognize_uploaded_image(page_image)
                    if page_text.strip():
                        page_results.append(page_text)
                    engine = page_engine if len(pdf) == 1 else "OCR on scanned PDF"
                extracted = "\\n\\n".join(page_results).strip()
                if not extracted:
                    raise ValueError("No text was recognized in this scanned PDF.")
        elif extension == ".docx":
            extracted = extract_document_text(temp_path, extension)
            engine = "Word text extraction"
        else:
            extracted = contents.decode("utf-8", errors="replace").strip()
            engine = "Text file"
        if not extracted:
            raise ValueError("No text was found in this file.")
        doc_id = save_to_library(filename, extension, contents, extracted)
        return {
            "id": doc_id, "filename": filename, "characters": len(extracted),
            "text": extracted, "engine": engine,
            "message": "Processed and saved to your document library."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process {filename}: {exc}")
    finally:
        temp_path.unlink(missing_ok=True)


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



EXPORT_FORMATS = {
    "txt": ("text/plain; charset=utf-8", "txt"),
    "md": ("text/markdown; charset=utf-8", "md"),
    "html": ("text/html; charset=utf-8", "html"),
    "rtf": ("application/rtf", "rtf"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    "pdf": ("application/pdf", "pdf"),
}


def make_export_file(text: str, filename: str, export_format: str) -> bytes:
    """Render recognized/extracted text to a portable download format."""
    if export_format in {"txt", "md"}:
        return text.encode("utf-8")
    if export_format == "html":
        safe_title = xml_escape(filename)
        body = "<br>\n".join(xml_escape(line) for line in text.splitlines())
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{safe_title}</title></head><body>"
            f"<h1>{safe_title}</h1><main>{body}</main></body></html>"
        ).encode("utf-8")
    if export_format == "rtf":
        def rtf_escape(value):
            out = []
            for char in value:
                code = ord(char)
                if char in "\\\\{}":
                    out.append("\\" + char)
                elif char == "\n":
                    out.append("\\par\n")
                elif code < 128:
                    out.append(char)
                else:
                    signed = code if code < 32768 else code - 65536
                    out.append(f"\\u{signed}?")
            return "".join(out)
        return ("{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Arial;}}\\f0\\fs22 "
                + rtf_escape(text) + "}").encode("ascii", errors="replace")
    if export_format == "docx":
        doc = WordDocument()
        for line in text.splitlines():
            doc.add_paragraph(line)
        if not text.splitlines():
            doc.add_paragraph("")
        output = io.BytesIO()
        doc.save(output)
        return output.getvalue()
    if export_format == "pdf":
        output = io.BytesIO()
        doc = SimpleDocTemplate(
            output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
            title=filename, author="Document OCR",
        )
        styles = getSampleStyleSheet()
        style = styles["BodyText"]
        style.fontName = "Helvetica"
        style.fontSize = 10
        style.leading = 14
        # Use a Unicode font when available so common accented characters render.
        font_candidates = [
            Path(r"C:\\Windows\\Fonts\\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        for font_path in font_candidates:
            if font_path.is_file():
                try:
                    pdfmetrics.registerFont(TTFont("OCRUnicode", str(font_path)))
                    style.fontName = "OCRUnicode"
                    break
                except Exception:
                    pass
        story = [Paragraph(xml_escape(filename), getSampleStyleSheet()["Title"]), Spacer(1, 8)]
        for line in text.splitlines():
            story.append(Paragraph(xml_escape(line) if line else "&#160;", style))
        doc.build(story)
        return output.getvalue()
    raise ValueError("Unsupported export format.")


@app.get("/api/documents/{document_id}/export/{export_format}")
def export_document(document_id: str, export_format: str):
    export_format = export_format.lower().lstrip(".")
    if export_format not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail="Choose TXT, DOCX, PDF, Markdown, HTML, or RTF.")
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            "SELECT filename, extracted_text FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    original_name, extracted_text = row
    base_name = Path(original_name).stem or "recognized_text"
    media_type, extension = EXPORT_FORMATS[export_format]
    content = make_export_file(extracted_text, base_name, export_format)
    download_name = f"{base_name}_text.{extension}"
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(download_name)}"}
    return Response(content=content, media_type=media_type, headers=headers)


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


