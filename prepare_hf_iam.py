"""Export the public Hugging Face IAM line dataset to this project's fine-tune CSV format.

Install optional dependencies first:
    python -m pip install -r requirements-data.txt

Example:
    python prepare_hf_iam.py --output data/iam_hf --max-samples 0

The exporter never downloads data into Git-tracked paths; data/ is gitignored.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "line"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Teklia/IAM-line",
                        help="Hugging Face dataset ID (default: Teklia/IAM-line).")
    parser.add_argument("--output", type=Path, default=Path("data/iam_hf"))
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 exports all rows; positive value limits the total for a quick test.")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be zero (all) or a positive integer.")

    output = args.output
    image_dir = output / "lines"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    manifest_path = output / "manifest.csv"
    count = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "text", "group"])
        writer.writeheader()
        for split_name, split_data in dataset.items():
            for row_index, row in enumerate(split_data):
                if args.max_samples and count >= args.max_samples:
                    break
                text = str(row.get("text") or "").strip()
                image = row.get("image")
                if not text or image is None:
                    skipped += 1
                    continue
                unsupported = sorted(set(text) - set(
                    " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#&'()*+,-./:;?@[]_\\""
                ))
                if unsupported:
                    skipped += 1
                    continue
                writer_id = next((row.get(key) for key in
                                  ("writer_id", "writer", "writerid", "writerId")
                                  if row.get(key) not in (None, "")), None)
                # The published IAM-line export may not expose writer IDs.
                # Unique row groups allow the trainer to make a reproducible
                # sample-level split, but cannot guarantee writer-independent
                # validation when writer metadata is absent.
                group = str(writer_id) if writer_id is not None else f"{split_name}-{row_index}"
                filename = f"{safe_name(split_name)}_{row_index:06d}.png"
                destination = image_dir / filename
                if hasattr(image, "convert"):
                    image.convert("L").save(destination, format="PNG")
                else:
                    Image.fromarray(image).convert("L").save(destination, format="PNG")
                writer.writerow({
                    "image": f"lines/{filename}",
                    "text": text,
                    "group": group,
                })
                count += 1
            if args.max_samples and count >= args.max_samples:
                break

    if count < 2:
        raise RuntimeError(
            f"Only {count} usable lines exported ({skipped} skipped). "
            "Check the dataset schema and model character alphabet."
        )
    print(f"Exported {count} labelled lines; skipped {skipped}.")
    print(f"Manifest: {manifest_path}")
    print("Next: python finetune_iam.py --manifest "
          f"{manifest_path} --checkpoint iam_crnn_ctc.pth "
          "--output iam_crnn_ctc_finetuned.pth --device cpu")
    print("Validation is writer-independent only if the dataset exposes writer IDs.")


if __name__ == "__main__":
    main()
