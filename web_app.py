"""FastAPI backend for handwritten and printed document OCR."""
from functools import lru_cache
import io
import os
from pathlib import Path

import pytesseract
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pytesseract import TesseractNotFoundError

from iam_decoder import CharacterNGramLM
from iam_image import split_page_into_lines, deskew_image
from predict_iam import DEFAULT_MODEL_PATH, get_inference_device, load_model, predict_line_details

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
LANGUAGE_MODEL_PATH = ROOT / "language_model.txt"

# Set TESSERACT_CMD if Tesseract is installed outside the default Windows path.
_tesseract_override = os.environ.get("TESSERACT_CMD")
if _tesseract_override:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_override
elif os.name == "nt":
    default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_tesseract.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(default_tesseract)

app = FastAPI(title="Handwriting and Printed Text OCR")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_model_and_device():
    device = get_inference_device()
    model_path = ROOT / DEFAULT_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing IAM checkpoint: {model_path}")
    return load_model(model_path, device), device


@lru_cache(maxsize=1)
def get_language_model():
    if LANGUAGE_MODEL_PATH.is_file():
        return CharacterNGramLM.from_file(LANGUAGE_MODEL_PATH)
    return None


def recognize_printed(image: Image.Image, mode: str) -> list[str]:
    """Recognize printed English text with Tesseract; IAM weights are not needed."""
    prepared = ImageOps.autocontrast(image.convert("L"))
    page_segmentation = "3" if mode == "page" else "7"
    text = pytesseract.image_to_string(
        prepared, lang="eng", config=f"--oem 3 --psm {page_segmentation}"
    )
    return [line.strip() for line in text.splitlines() if line.strip()]


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    try:
        _, device = get_model_and_device()
        iam_ready = True
        device_name = (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else str(device).upper()
        )
        iam_error = None
    except Exception as exc:
        iam_ready, device_name, iam_error = False, "Unavailable", str(exc)

    try:
        pytesseract.get_tesseract_version()
        printed_ready, printed_error = True, None
    except (TesseractNotFoundError, OSError) as exc:
        printed_ready, printed_error = False, str(exc)

    return {
        "ready": iam_ready,
        "iam_ready": iam_ready,
        "iam_error": iam_error,
        "model": "iam_crnn_ctc.pth",
        "device": device_name,
        "language_model": LANGUAGE_MODEL_PATH.is_file(),
        "printed_ready": printed_ready,
        "printed_error": printed_error,
        "printed_engine": "Tesseract OCR",
    }


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    text_type: str = Form("handwriting"),
    mode: str = Form("page"),
    decoder: str = Form("greedy"),
    beam_width: int = Form(10),
    use_language_model: bool = Form(False),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    if text_type not in {"handwriting", "printed"}:
        raise HTTPException(status_code=400, detail="text_type must be 'handwriting' or 'printed'.")
    if mode not in {"line", "page"}:
        raise HTTPException(status_code=400, detail="mode must be 'line' or 'page'.")
    if decoder not in {"greedy", "beam"}:
        raise HTTPException(status_code=400, detail="decoder must be 'greedy' or 'beam'.")
    if beam_width < 1 or beam_width > 100:
        raise HTTPException(status_code=400, detail="beam_width must be between 1 and 100.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    # Printed mode does not load the IAM model and works without its checkpoint.
    if text_type == "printed":
        try:
            lines = recognize_printed(image, mode)
        except TesseractNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Tesseract OCR was not found. Install the Tesseract Windows "
                    "application, or set TESSERACT_CMD to the full path of tesseract.exe."
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Printed OCR failed: {exc}")
        return {
            "lines": [
                {"number": i + 1, "text": line, "confidence": None, "decoder": "tesseract"}
                for i, line in enumerate(lines)
            ],
            "count": len(lines),
            "device": "CPU",
            "model": "Tesseract OCR",
            "text_type": "printed",
            "message": None if lines else "No printed text detected. Try a clearer image.",
        }

    try:
        model, device = get_model_and_device()
        language_model = get_language_model() if use_language_model else None
        if use_language_model and language_model is None:
            raise HTTPException(
                status_code=400,
                detail="Optional language model requested, but language_model.txt is not installed.",
            )
        crops = [deskew_image(image)] if mode == "line" else split_page_into_lines(image)
        if not crops:
            return {
                "lines": [], "count": 0, "device": str(device),
                "message": "No text lines detected. Try single-line mode or a clearer image.",
            }
        results = [
            predict_line_details(
                model, crop, device=device, decoder=decoder, beam_width=beam_width,
                language_model=language_model,
                lm_weight=0.25 if language_model else 0.0,
            )
            for crop in crops
        ]
        return {
            "lines": [{"number": i + 1, **result} for i, result in enumerate(results)],
            "count": len(results),
            "device": str(device),
            "model": "iam_crnn_ctc.pth",
            "decoder": decoder,
            "language_model": language_model is not None,
            "text_type": "handwriting",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}")
