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
