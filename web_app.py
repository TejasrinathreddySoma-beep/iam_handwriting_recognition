"""FastAPI backend for the standalone HTML IAM handwriting reader."""
from pathlib import Path
from functools import lru_cache
import io

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d, binary_closing

from predict_iam import DEFAULT_MODEL_PATH, get_inference_device, load_model, predict_line

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
app = FastAPI(title="IAM Handwriting Reader")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def get_model_and_device():
    device = get_inference_device()
    model_path = ROOT / DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Missing IAM checkpoint: {model_path}")
    return load_model(model_path, device), device


def split_page(image: Image.Image):
    """Split a page into horizontal text bands using smoothed row ink projection."""
    gray = ImageOps.autocontrast(image.convert("L"))
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Expected a grayscale image.")
    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    if float(np.median(border)) < 127:
        arr = 255 - arr

    ink = arr < 205
    row_ink = ink.sum(axis=1).astype(float)
    smooth = gaussian_filter1d(row_ink, sigma=1.8)
    threshold = max(2.0, arr.shape[1] * 0.0035)
    active = binary_closing(smooth >= threshold, structure=np.ones(3, dtype=bool))
    indexes = np.flatnonzero(active)
    groups = []
    if len(indexes):
        first = previous = int(indexes[0])
        gap = max(5, int(arr.shape[0] * 0.008))
        for y in indexes[1:]:
            y = int(y)
            if y - previous > gap:
                if previous - first >= 5:
                    groups.append((first, previous + 1))
                first = y
            previous = y
        if previous - first >= 5:
            groups.append((first, previous + 1))

    # If the active bands merge, split at low-ink valleys between row-projection peaks.
    if len(groups) <= 2:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            smooth,
            distance=max(20, int(arr.shape[0] * 0.065)),
            prominence=max(2.0, float(smooth.max()) * 0.035),
        )
        if len(peaks) >= 3:
            cuts = [
                int(left + np.argmin(smooth[left:right + 1]))
                for left, right in zip(peaks[:-1], peaks[1:])
            ]
            bounds = [0] + cuts + [arr.shape[0]]
            candidate_groups = [
                (top, bottom) for top, bottom in zip(bounds[:-1], bounds[1:])
                if bottom - top >= 12 and smooth[top:bottom].max(initial=0) >= threshold
            ]
            if len(candidate_groups) > len(groups):
                groups = candidate_groups

    crops = []
    for y0, y1 in groups:
        region = ink[y0:y1]
        cols = np.flatnonzero(region.any(axis=0))
        if not len(cols):
            continue
        px = max(5, int(arr.shape[1] * 0.006))
        py = max(4, int(arr.shape[0] * 0.006))
        top, bottom = max(0, y0 - py), min(arr.shape[0], y1 + py)
        left, right = max(0, int(cols[0]) - px), min(arr.shape[1], int(cols[-1]) + 1 + px)
        if bottom - top >= 10 and right - left >= 20:
            crops.append(Image.fromarray(arr[top:bottom, left:right]))
    return crops


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    try:
        _, device = get_model_and_device()
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device).upper()
        return {"ready": True, "model": "iam_crnn_ctc.pth", "device": gpu_name}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...), mode: str = Form("page")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")
    try:
        model, device = get_model_and_device()
        crops = [image] if mode == "line" else split_page(image)
        if not crops:
            return {"lines": [], "count": 0, "device": str(device), "message": "No text lines detected. Try single-line mode or a clearer image."}
        texts = [predict_line(model, crop, device).strip() for crop in crops]
        return {
            "lines": [{"number": i + 1, "text": text} for i, text in enumerate(texts)],
            "count": len(texts),
            "device": str(device),
            "model": "iam_crnn_ctc.pth",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}")
