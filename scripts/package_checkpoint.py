"""Validate and package a trained IAM checkpoint for a versioned GitHub release."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from iam_model import CHAR_TO_ID
from predict_iam import IMAGE_HEIGHT


def package_checkpoint(checkpoint_path: Path, version: str, output_dir: Path) -> Path:
    if not version or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for ch in version):
        raise ValueError("Version must contain only letters, digits, dot, dash, and underscore.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("characters") != CHAR_TO_ID:
        raise ValueError("Checkpoint character mapping does not match the current IAM model.")
    if checkpoint.get("image_height") != IMAGE_HEIGHT:
        raise ValueError("Checkpoint image height does not match the inference pipeline.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict.")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"iam-crnn-ctc-{version}.zip"
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "version": version,
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": "CNN-BiLSTM-CTC",
        "image_height": IMAGE_HEIGHT,
        "num_characters": len(CHAR_TO_ID),
        "checkpoint": "iam_crnn_ctc.pth",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_cer": checkpoint.get("best_cer"),
    }
    manifest_path = output_dir / "MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(checkpoint_path, arcname="iam_crnn_ctc.pth")
        bundle.write(manifest_path, arcname="MODEL_MANIFEST.json")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "iam_crnn_ctc.pth")
    parser.add_argument("--version", required=True, help="For example: v1.0.0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    archive = package_checkpoint(args.checkpoint, args.version, args.output_dir)
    print(f"Release asset created: {archive}")
    print("Upload it to a GitHub Release, for example:")
    print(f"  gh release create {args.version} {archive} --title 'IAM CRNN-CTC {args.version}'")


if __name__ == "__main__":
    main()
