"""Greedy and CTC prefix-beam decoders with optional character n-gram LM."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from iam_model import BLANK_ID, ID_TO_CHAR


class CharacterNGramLM:
    """Small add-one-smoothed character bigram language model from plain text."""

    def __init__(self, text: str):
        self.unigrams = Counter(text)
        self.bigrams = Counter(zip(text, text[1:]))
        self.vocabulary = max(1, len(self.unigrams))
        self.contexts = Counter(text[:-1])

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(Path(path).read_text(encoding="utf-8"))

    def log_probability(self, previous: str, character: str) -> float:
        if not previous:
            return -math.log(self.vocabulary + 1)
        numerator = self.bigrams[(previous, character)] + 1
        denominator = self.contexts[previous] + self.vocabulary + 1
        return math.log(numerator / denominator)


def greedy_decode_with_confidence(log_probs: torch.Tensor, input_length: int) -> tuple[str, float]:
    """Greedy CTC decoding and mean per-frame max probability (not calibrated)."""
    frame_probs = log_probs[:input_length].float().exp()
    if not len(frame_probs):
        return "", 0.0
    confidence = float(frame_probs.max(dim=-1).values.mean().item())
    path = frame_probs.argmax(dim=-1).tolist()
    output, previous = [], BLANK_ID
    for token in path:
        if token != BLANK_ID and token != previous:
            output.append(ID_TO_CHAR.get(token, ""))
        previous = token
    return "".join(output), confidence


def prefix_beam_search(
    log_probs: torch.Tensor,
    input_length: int,
    beam_width: int = 10,
    language_model: CharacterNGramLM | None = None,
    lm_weight: float = 0.0,
) -> tuple[str, float]:
    """Decode one [time, classes] log-probability matrix using CTC prefix beams.

    The optional bigram LM is a lightweight rescoring prior. Tune lm_weight on
    validation data; it is not applied unless a language-model file is supplied.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    probs = log_probs[:input_length].detach().float().cpu()
    if not len(probs):
        return "", 0.0
    # prefix -> (log probability ending blank, ending nonblank)
    beams = {(): (0.0, -math.inf)}
    for frame in probs:
        next_beams = defaultdict(lambda: (-math.inf, -math.inf))
        for prefix, (p_blank, p_nonblank) in beams.items():
            total = _logadd(p_blank, p_nonblank)
            _merge(next_beams, prefix, blank=_logadd(*next_beams[prefix][:1], total))
            top_values, top_indices = torch.topk(frame, k=min(frame.numel(), max(beam_width * 2, 20)))
            for logp, token_tensor in zip(top_values.tolist(), top_indices.tolist()):
                token = int(token_tensor)
                if token == BLANK_ID:
                    continue
                char = ID_TO_CHAR.get(token)
                if char is None:
                    continue
                if prefix and prefix[-1] == token:
                    # Repeating the last label without an intervening blank keeps the prefix.
                    _merge(next_beams, prefix, nonblank=_logadd(next_beams[prefix][1], p_nonblank + logp))
                    # A repeated label after blank extends the output.
                    new_prefix = prefix + (token,)
                    lm = _lm_increment(language_model, prefix, char) * lm_weight
                    _merge(next_beams, new_prefix, nonblank=_logadd(next_beams[new_prefix][1], p_blank + logp + lm))
                else:
                    new_prefix = prefix + (token,)
                    lm = _lm_increment(language_model, prefix, char) * lm_weight
                    _merge(next_beams, new_prefix, nonblank=_logadd(next_beams[new_prefix][1], total + logp + lm))
        beams = dict(sorted(
            next_beams.items(),
            key=lambda item: _logadd(*item[1]),
            reverse=True,
        )[:beam_width])
    best_prefix, best_scores = max(beams.items(), key=lambda item: _logadd(*item[1]))
    text = "".join(ID_TO_CHAR.get(token, "") for token in best_prefix)
    # A decoder score is a ranking score, not a calibrated probability.
    score = _logadd(*best_scores)
    return text, float(math.exp(min(0.0, score / max(1, input_length))))


def _lm_increment(lm, prefix, character):
    if lm is None:
        return 0.0
    previous = ID_TO_CHAR.get(prefix[-1], "") if prefix else ""
    return lm.log_probability(previous, character)


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = (a, b) if a >= b else (b, a)
    return high + math.log1p(math.exp(low - high))


def _merge(mapping, prefix, blank=None, nonblank=None):
    old_blank, old_nonblank = mapping[prefix]
    mapping[prefix] = (
        old_blank if blank is None else blank,
        old_nonblank if nonblank is None else nonblank,
    )
