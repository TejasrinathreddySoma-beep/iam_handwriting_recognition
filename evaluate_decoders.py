"""Compare greedy and beam decoding CER on the IAM validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iam_decoder import CharacterNGramLM, prefix_beam_search
from iam_model import IAMCRNN, greedy_decode
from predict_iam import DEFAULT_MODEL_PATH, get_inference_device, load_model
from train_iam import DATA_ROOT, IAMLineDataset, collate_lines, edit_distance, read_annotations, read_split


def evaluate(model, loader, device, beam_width=10, language_model=None, lm_weight=0.0):
    totals = {"greedy_errors": 0, "beam_errors": 0, "characters": 0}
    model.eval()
    with torch.inference_mode():
        for images, _targets, _target_lengths, widths, texts in loader:
            images = images.to(device)
            input_lengths = IAMCRNN.output_lengths(widths).to(device)
            log_probs = model(images).cpu()
            greedy = greedy_decode(log_probs, input_lengths.cpu())
            for index, (truth, greedy_text) in enumerate(zip(texts, greedy)):
                beam_text, _score = prefix_beam_search(
                    log_probs[index], int(input_lengths[index].item()),
                    beam_width=beam_width, language_model=language_model,
                    lm_weight=lm_weight,
                )
                totals["greedy_errors"] += edit_distance(truth, greedy_text)
                totals["beam_errors"] += edit_distance(truth, beam_text)
                totals["characters"] += len(truth)
    denominator = max(1, totals["characters"])
    return {
        "samples_characters": totals["characters"],
        "greedy_cer": totals["greedy_errors"] / denominator,
        "beam_cer": totals["beam_errors"] / denominator,
        "beam_width": beam_width,
        "language_model": language_model is not None,
        "lm_weight": lm_weight if language_model is not None else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--language-model", type=Path)
    parser.add_argument("--lm-weight", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("decoder_comparison.json"))
    args = parser.parse_args()
    device = get_inference_device()
    model = load_model(args.checkpoint, device)
    annotations = read_annotations(args.data_root)
    ids = read_split(args.data_root, "validationset1")
    dataset = IAMLineDataset(args.data_root, ids, annotations)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_lines,
    )
    language_model = CharacterNGramLM.from_file(args.language_model) if args.language_model else None
    comparisons = [
        evaluate(model, loader, device, args.beam_width),
    ]
    if language_model:
        comparisons.append(evaluate(
            model, loader, device, args.beam_width,
            language_model=language_model, lm_weight=args.lm_weight,
        ))
    result = {"device": str(device), "checkpoint": str(args.checkpoint), "comparisons": comparisons}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
