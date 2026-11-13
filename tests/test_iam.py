from pathlib import math
import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from iam_decoder import CharacterNGramLM, greedy_decode_with_confidence, prefix_beam_search
from iam_image import deskew_image, estimate_skew_angle, split_page_into_lines
from iam_model import BLANK_ID, CHAR_TO_ID, IAMCRNN, ctc_minimum_lengths
from predict_iam import preprocess_line, predict_line_details
from train_iam import read_annotations, read_split


def test_read_annotations_and_split(tmp_path):
    (tmp_path / "lines.txt").write_text(
        "# comment\n"
        "a01-000u-00-00 ok 154 19 100 20 0 abc Hello world\n"
        "a01-000u-00-01 err 150 18 90 18 0 def ignored\n",
        encoding="utf-8",
    )
    (tmp_path / "trainset.txt").write_text("# split\na01-000u-00-00\n", encoding="utf-8")
    assert read_annotations(tmp_path) == {"a01-000u-00-00": "Hello world"}
    assert read_split(tmp_path, "trainset") == ["a01-000u-00-00"]


def test_preprocess_line_shape_and_range():
    image = Image.new("RGB", (120, 30), "white")
    ImageDraw.Draw(image).text((8, 5), "Test", fill="black")
    tensor = preprocess_line(image)
    assert tensor.shape[0] == 1
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_ctc_output_lengths():
    widths = torch.tensor([4, 5, 63, 64, 65])
    assert IAMCRNN.output_lengths(widths).tolist() == [1, 1, 15, 16, 16]


def test_deskew_and_line_segmentation():
    page = Image.new("L", (420, 180), "white")
    draw = ImageDraw.Draw(page)
    draw.text((20, 25), "First handwritten line", fill="black")
    draw.text((20, 85), "Second handwritten line", fill="black")
    angle = estimate_skew_angle(page)
    corrected = deskew_image(page)
    assert -7.0 <= angle <= 7.0
    assert corrected.width >= page.width
    lines = split_page_into_lines(page)
    assert len(lines) >= 1


def test_greedy_and_beam_decoder_return_text_and_score():
    probs = torch.full((5, len(CHAR_TO_ID) + 1), -8.0)
    probs[:, BLANK_ID] = -3.0
    probs[0, CHAR_TO_ID["H"]] = -0.01
    probs[1, CHAR_TO_ID["H"]] = -0.01
    probs[2, BLANK_ID] = -0.01
    probs[3, CHAR_TO_ID["i"]] = -0.01
    probs[4, BLANK_ID] = -0.01
    probs = probs.log_softmax(dim=-1)
    greedy, confidence = greedy_decode_with_confidence(probs, 5)
    beam, score = prefix_beam_search(probs, 5, beam_width=5)
    assert greedy == "Hi"
    assert beam == "Hi"
    assert 0.0 <= confidence <= 1.0
    assert score > 0.0


def test_bigram_language_model():
    lm = CharacterNGramLM("hello handwriting")
    assert np.isfinite(lm.log_probability("h", "e"))



def test_inference_returns_transcription_and_confidence():
    class DummyRecognizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            steps = batch.shape[-1] // 4
            logits = torch.full((1, steps, len(CHAR_TO_ID) + 1), -8.0, device=batch.device)
            logits[:, :, BLANK_ID] = 0.0
            logits[0, 0, CHAR_TO_ID["A"]] = 8.0
            logits[0, 2, CHAR_TO_ID["B"]] = 8.0
            return logits.log_softmax(dim=-1)

    image = Image.new("L", (80, 32), "white")
    result = predict_line_details(DummyRecognizer(), image, decoder="greedy")
    assert result["text"] == "AB"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "uncalibrated" in result["confidence_kind"]


def test_beam_search_rejects_invalid_width():
    logits = torch.zeros((3, len(CHAR_TO_ID) + 1)).log_softmax(dim=-1)
    with pytest.raises(ValueError):
        prefix_beam_search(logits, 3, beam_width=0)



def test_ctc_minimum_lengths_include_repeated_adjacent_labels():
    targets = torch.tensor([CHAR_TO_ID["l"], CHAR_TO_ID["l"], CHAR_TO_ID["a"], CHAR_TO_ID["b"], CHAR_TO_ID["b"]])
    lengths = torch.tensor([2, 3])
    assert ctc_minimum_lengths(targets, lengths).tolist() == [3, 4]


def test_notebook_line_removal_keeps_page_readable():
    from iam_image import remove_notebook_lines
    page = Image.new("L", (500, 220), "white")
    draw = ImageDraw.Draw(page)
    for y in range(30, 210, 30):
        draw.line((0, y, 499, y), fill=190, width=1)
    draw.text((30, 55), "handwritten words", fill=20)
    cleaned = remove_notebook_lines(page)
    assert cleaned.size == page.size
    assert np.asarray(cleaned).mean() > 150


def test_page_rectification_returns_image():
    from iam_image import rectify_page
    page = Image.new("L", (320, 240), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((25, 20, 295, 220), outline=30, width=3)
    result = rectify_page(page)
    assert isinstance(result, Image.Image)
    assert result.width > 0 and result.height > 0


def test_beam_merge_accumulates_distinct_paths():
    from collections import defaultdict
    from iam_decoder import _merge, _logadd
    mapping = defaultdict(lambda: (-math.inf, -math.inf))
    _merge(mapping, (1,), blank=-2.0)
    _merge(mapping, (1,), blank=-3.0)
    _merge(mapping, (1,), nonblank=-1.5)
    blank, nonblank = mapping[(1,)]
    assert abs(blank - _logadd(-2.0, -3.0)) < 1e-6
    assert abs(nonblank - (-1.5)) < 1e-6
