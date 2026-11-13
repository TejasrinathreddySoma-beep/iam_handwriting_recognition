"""Load the trained IAM CNN-BiLSTM-CTC model and recognize a cropped text line."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from iam_model import CHAR_TO_ID, IAMCRNN, greedy_decode

IMAGE_HEIGHT = 64
MAX_WIDTH = 1600
DEFAULT_MODEL_PATH = Path("iam_crnn_ctc.pth")


def get_inference_device():
    """Prefer CUDA for the trained RTX model, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    return device


def preprocess_line(image):
    """Match the grayscale, contrast, resize and normalization used in training."""
    image = ImageOps.autocontrast(image.convert("L"))
    width, height = image.size
    resized_width = min(MAX_WIDTH, max(4, round(width * IMAGE_HEIGHT / height)))
    image = image.resize((resized_width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def load_model(path=DEFAULT_MODEL_PATH, device=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"IAM model checkpoint not found: {path}. "
            "Place iam_crnn_ctc.pth in the project root."
        )
    device = device or get_inference_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("characters") != CHAR_TO_ID:
        raise ValueError("The checkpoint character mapping does not match iam_model.py.")
    if checkpoint.get("image_height") != IMAGE_HEIGHT:
        raise ValueError("The checkpoint image height does not match this predictor.")
    model = IAMCRNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def predict_line(model, image, device=None):
    device = device or next(model.parameters()).device
    pixels = preprocess_line(image)
    original_width = torch.tensor([pixels.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        log_probs = model(pixels.unsqueeze(0).to(device))
    input_lengths = IAMCRNN.output_lengths(original_width)
    return greedy_decode(log_probs, input_lengths)[0]


def predict_file(image_path, model_path=DEFAULT_MODEL_PATH):
    device = get_inference_device()
    model = load_model(model_path, device)
    with Image.open(image_path) as image:
        return predict_line(model, image, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one cropped handwritten text-line image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(predict_file(args.image, args.model))
