"""IAM line prediction with deskew preprocessing, beam search and confidence."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from iam_decoder import CharacterNGramLM, greedy_decode_with_confidence, prefix_beam_search
from iam_image import deskew_image
from iam_model import CHAR_TO_ID, IAMCRNN

IMAGE_HEIGHT = 64
MAX_WIDTH = 1600
DEFAULT_MODEL_PATH = Path("iam_crnn_ctc.pth")


def get_inference_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_line(image, deskew=False):
    """Apply grayscale/contrast normalization and training-compatible resizing."""
    image = deskew_image(image) if deskew else ImageOps.autocontrast(image.convert("L"))
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


def predict_line_details(
    model,
    image,
    device=None,
    decoder="greedy",
    beam_width=10,
    language_model=None,
    lm_weight=0.0,
    deskew=False,
):
    device = device or next(model.parameters()).device
    pixels = preprocess_line(image, deskew=deskew)
    original_width = torch.tensor([pixels.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        log_probs = model(pixels.unsqueeze(0).to(device))[0].detach().cpu()
    input_length = int(IAMCRNN.output_lengths(original_width)[0])
    if decoder == "beam":
        text, confidence = prefix_beam_search(
            log_probs, input_length, beam_width=beam_width,
            language_model=language_model, lm_weight=lm_weight,
        )
        confidence_kind = "beam sequence score (uncalibrated)"
    elif decoder == "greedy":
        text, confidence = greedy_decode_with_confidence(log_probs, input_length)
        confidence_kind = "mean frame confidence (uncalibrated)"
    else:
        raise ValueError("decoder must be 'greedy' or 'beam'")
    return {
        "text": text,
        "confidence": round(float(confidence), 4),
        "confidence_kind": confidence_kind,
        "decoder": decoder,
    }


def predict_line(model, image, device=None):
    """Backward-compatible simple text-only greedy prediction."""
    return predict_line_details(model, image, device=device)["text"]


def predict_file(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    decoder="greedy",
    language_model_path=None,
    lm_weight=0.0,
    deskew=False,
):
    device = get_inference_device()
    model = load_model(model_path, device)
    language_model = CharacterNGramLM.from_file(language_model_path) if language_model_path else None
    with Image.open(image_path) as image:
        return predict_line_details(
            model, image, device=device, decoder=decoder,
            language_model=language_model, lm_weight=lm_weight, deskew=deskew,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one cropped handwritten text-line image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--language-model", type=Path, help="Optional UTF-8 plain-text corpus for character bigram rescoring.")
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--deskew", action="store_true")
    args = parser.parse_args()
    if args.beam_width < 1:
        parser.error("--beam-width must be positive")
    result = predict_file(
        args.image, args.model, args.decoder, args.language_model,
        args.lm_weight, args.deskew,
    )
    print(result["text"])
    print(f"Confidence: {result['confidence']:.1%} ({result['confidence_kind']})")
