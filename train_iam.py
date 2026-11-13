"""Train a grayscale IAM line recognizer with CNN + BiLSTM + CTC.

Expected IAM layout:
data/iam/
  lines.txt
  lines/a01/a01-000u/a01-000u-00.png
  trainset.txt, validationset1.txt, testset.txt
  (or ascii/lines.txt and ascii/sets/{trainset,validationset1,testset}.txt)
Download IAM Lines and its ASCII annotation/split files from the official IAM site.
"""
from pathlib import Path
import random

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader

from iam_model import CHAR_TO_ID, IAMCRNN, greedy_decode

DATA_ROOT = Path("data/iam")
IMAGE_HEIGHT = 64
MAX_WIDTH = 1600
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
WORKERS = 0
MODEL_PATH = Path("iam_crnn_ctc.pth")


def locate_file(root, candidates):
    root = Path(root)
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    for candidate in candidates:
        matches = list(root.rglob(Path(candidate).name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {candidates} under {root}")


def read_annotations(root):
    annotation_path = locate_file(root, ["lines.txt", "ascii/lines.txt"])
    annotations = {}
    with annotation_path.open("r", encoding="utf-8") as file:
        for raw in file:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.strip().split(maxsplit=8)
            if len(fields) < 9:
                continue
            line_id, status, *_rest, transcription = fields
            if status == "ok" and transcription.strip():
                annotations[line_id] = transcription.strip()
    return annotations


def read_split(root, name):
    path = locate_file(root, [f"{name}.txt", f"ascii/sets/{name}.txt", f"sets/{name}.txt"])
    ids = []
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            item = raw.strip().split()
            if item and not item[0].startswith("#"):
                ids.append(item[0])
    return ids


def line_image_path(root, line_id):
    parts = line_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected IAM line ID: {line_id}")
    return root / parts[0] / "-".join(parts[:2]) / f"{line_id}.png"


class IAMLineDataset(Dataset):
    def __init__(self, root, ids, annotations):
        self.root = root
        self.samples = [
            (line_id, annotations[line_id].replace("|", " "))
            for line_id in ids
            if line_id in annotations and line_image_path(root, line_id).is_file()
        ]
        if not self.samples:
            raise RuntimeError("No IAM line images matched annotations and split. Check the extracted dataset layout.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        line_id, text = self.samples[index]
        with Image.open(line_image_path(self.root, line_id)) as source:
            image = ImageOps.autocontrast(source.convert("L"))
        width, height = image.size
        new_width = max(4, int(round(width * IMAGE_HEIGHT / height)))
        new_width = min(new_width, MAX_WIDTH)
        image = image.resize((new_width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        import numpy as np
        pixels = torch.from_numpy(np.array(image, dtype="float32").copy()).div_(255.0).unsqueeze(0)
        target = []
        for character in text:
            if character not in CHAR_TO_ID:
                raise ValueError(f"Character {character!r} is not in the model alphabet; extend CHARACTERS in iam_model.py.")
            target.append(CHAR_TO_ID[character])
        return pixels, torch.tensor(target, dtype=torch.long), text, pixels.shape[-1]


def collate_lines(batch):
    images, targets, texts, widths = zip(*batch)
    max_width = ((max(widths) + 3) // 4) * 4
    padded = torch.ones((len(images), 1, IMAGE_HEIGHT, max_width), dtype=torch.float32)
    for index, image in enumerate(images):
        padded[index, :, :, :image.shape[-1]] = image
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    return padded, torch.cat(targets), target_lengths, torch.tensor(widths, dtype=torch.long), list(texts)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for col, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_chars = total_errors = 0
    for images, targets, target_lengths, widths, texts in loader:
        images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
        input_lengths = IAMCRNN.output_lengths(widths).to(device)
        if torch.any(input_lengths < target_lengths):
            raise RuntimeError("A line is too wide/long for CTC after resizing. Lower IMAGE_HEIGHT or increase MAX_WIDTH.")
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
            raise RuntimeError("CTC loss became non-finite. Check image widths and label lengths.")
        total_loss += float(loss.item())
        for truth, prediction in zip(texts, greedy_decode(log_probs, input_lengths)):
            total_errors += edit_distance(truth, prediction)
            total_chars += len(truth)
    return total_loss / max(1, len(loader)), total_errors / max(1, total_chars)


def main():
    random.seed(42)
    torch.manual_seed(42)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    annotations = read_annotations(DATA_ROOT)
    train_data = IAMLineDataset(DATA_ROOT, read_split(DATA_ROOT, "trainset"), annotations)
    valid_data = IAMLineDataset(DATA_ROOT, read_split(DATA_ROOT, "validationset1"), annotations)
    test_data = IAMLineDataset(DATA_ROOT, read_split(DATA_ROOT, "testset"), annotations)
    print(f"IAM lines — train: {len(train_data)}, validation: {len(valid_data)}, test: {len(test_data)}")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, collate_fn=collate_lines)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, collate_fn=collate_lines)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, collate_fn=collate_lines)

    model = IAMCRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_cer, patience, no_improvement = float("inf"), 6, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_cer = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss, valid_cer = run_epoch(model, valid_loader, criterion, device)
        scheduler.step(valid_cer)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f}, CER {train_cer:.2%} | valid loss {valid_loss:.4f}, CER {valid_cer:.2%}")
        if valid_cer < best_cer:
            best_cer, no_improvement = valid_cer, 0
            torch.save({"model_state_dict": model.state_dict(), "characters": CHAR_TO_ID, "image_height": IMAGE_HEIGHT}, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_cer = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test character error rate: {test_cer:.2%}")
    print("Training complete.")


if __name__ == "__main__":
    main()
