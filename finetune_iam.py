"""Fine-tune an existing IAM CRNN-CTC checkpoint on labelled line images.

CSV format (UTF-8, header required):
    image,text,group
    images/page01_line01.png,"The first line",page01
    images/page01_line02.png,"The next line",page01

Use a writer ID or source page/document ID as group so related samples never
leak between train and validation. The source checkpoint is read-only; output
defaults to iam_crnn_ctc_finetuned.pth.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from iam_model import CHAR_TO_ID, IAMCRNN, ctc_minimum_lengths, greedy_decode

IMAGE_HEIGHT = 64
MAX_WIDTH = 1600


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "text", "group"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must have columns: image,text,group")
        for line_no, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            text = (row.get("text") or "").strip()
            group = (row.get("group") or "").strip()
            if not image_value or not text or not group:
                raise ValueError(f"CSV row {line_no}: image, text and group are all required.")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"CSV row {line_no}: image not found: {image_path}")
            unsupported = sorted(set(text) - set(CHAR_TO_ID))
            if unsupported:
                raise ValueError(
                    f"CSV row {line_no}: characters not supported by this checkpoint: "
                    + ", ".join(repr(c) for c in unsupported)
                )
            rows.append({"image": image_path, "text": text, "group": group})
    if len(rows) < 2:
        raise ValueError("At least two labelled line images are required.")
    return rows


def split_by_group(rows, validation_fraction, seed):
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 2:
        raise ValueError(
            "At least two distinct group values are required. Use writer IDs or "
            "page/document IDs so validation measures unseen handwriting/pages."
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    validation_count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:validation_count])
    train_rows = [row for row in rows if row["group"] not in validation_groups]
    validation_rows = [row for row in rows if row["group"] in validation_groups]
    return train_rows, validation_rows


class LabelledLineDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = ImageOps.autocontrast(source.convert("L"))
        width, height = image.size
        resized_width = min(MAX_WIDTH, max(4, round(width * IMAGE_HEIGHT / height)))
        image = image.resize((resized_width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        pixels = pixels.div_(255.0).unsqueeze(0)
        target = torch.tensor([CHAR_TO_ID[c] for c in row["text"]], dtype=torch.long)
        return pixels, target, row["text"], resized_width


def collate_lines(batch):
    images, targets, texts, widths = zip(*batch)
    padded_width = ((max(widths) + 3) // 4) * 4
    padded = torch.ones((len(images), 1, IMAGE_HEIGHT, padded_width), dtype=torch.float32)
    for i, image in enumerate(images):
        padded[i, :, :, :image.shape[-1]] = image
    return (
        padded, torch.cat(targets),
        torch.tensor([len(t) for t in targets], dtype=torch.long),
        torch.tensor(widths, dtype=torch.long), list(texts),
    )


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = errors = characters = 0.0
    for images, targets, target_lengths, widths, truths in loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = IAMCRNN.output_lengths(widths).to(device)
        minimums = ctc_minimum_lengths(targets, target_lengths)
        if torch.any(input_lengths < minimums):
            raise ValueError(
                "A transcription is too long for its image width under CTC. "
                "Use a wider/less tightly cropped line image or shorter text."
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            log_probs = model(images)
            loss = criterion(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CTC loss; check line crops and transcriptions.")
        total_loss += loss.item()
        predictions = greedy_decode(log_probs, input_lengths)
        for truth, prediction in zip(truths, predictions):
            errors += edit_distance(truth, prediction)
            characters += len(truth)
    return total_loss / max(1, len(loader)), errors / max(1, characters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image,text,group columns.")
    parser.add_argument("--checkpoint", type=Path, default=Path("iam_crnn_ctc.pth"),
                        help="Existing checkpoint to fine-tune (never modified).")
    parser.add_argument("--output", type=Path, default=Path("iam_crnn_ctc_finetuned.pth"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--freeze-cnn-epochs", type=int, default=1,
                        help="Freeze CNN for initial epochs; set 0 to fine-tune all layers immediately.")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive; workers cannot be negative.")
    if args.learning_rate <= 0 or not 0 < args.validation_fraction < 1:
        parser.error("learning-rate must be positive and validation-fraction between 0 and 1.")
    if args.freeze_cnn_epochs < 0:
        parser.error("freeze-cnn-epochs cannot be negative.")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("--output must differ from --checkpoint to preserve the original model.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but unavailable.")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (CPU is supported; training may be slow)")

    rows = read_manifest(args.manifest)
    train_rows, val_rows = split_by_group(rows, args.validation_fraction, args.seed)
    print(f"Labelled lines: {len(rows)} | train: {len(train_rows)} | validation: {len(val_rows)}")
    train_loader = DataLoader(LabelledLineDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, collate_fn=collate_lines)
    val_loader = DataLoader(LabelledLineDataset(val_rows), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, collate_fn=collate_lines)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Original checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("characters") != CHAR_TO_ID:
        raise ValueError("Checkpoint character mapping does not match the current model alphabet.")
    if checkpoint.get("image_height") != IMAGE_HEIGHT:
        raise ValueError("Checkpoint image height does not match the fine-tuning pipeline.")
    model = IAMCRNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_cer = float("inf")
    stale_epochs = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        freeze_cnn = epoch <= args.freeze_cnn_epochs
        for parameter in model.cnn.parameters():
            parameter.requires_grad_(not freeze_cnn)
        train_loss, train_cer = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_cer = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_cer)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, CER {train_cer:.2%} "
              f"| val loss {val_loss:.4f}, CER {val_cer:.2%} | CNN frozen: {freeze_cnn}")
        if val_cer < best_cer:
            best_cer = val_cer
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "characters": CHAR_TO_ID,
                "image_height": IMAGE_HEIGHT,
                "best_cer": best_cer,
                "epoch": epoch,
                "fine_tuned_from": str(args.checkpoint),
                "fine_tuning_groups": sorted({row["group"] for row in rows}),
            }, args.output)
            print(f"Saved best fine-tuned checkpoint: {args.output}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation CER did not improve.")
                break
    print(f"Fine-tuning complete. Best validation CER: {best_cer:.2%}")
    print(f"Original checkpoint preserved: {args.checkpoint}")
    print(f"Use the new checkpoint with the app's model path setting/configuration: {args.output}")


if __name__ == "__main__":
    main()
