"""CNN + BiLSTM model for IAM handwritten text-line recognition."""
import torch
from torch import nn

# CTC blank is index 0. Remaining indices map to these characters.
CHARACTERS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#&'()*+,-./:;?@[]_" + chr(34) + chr(92)
BLANK_ID = 0
CHAR_TO_ID = {character: index + 1 for index, character in enumerate(CHARACTERS)}
ID_TO_CHAR = {index + 1: character for index, character in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS) + 1


class IAMCRNN(nn.Module):
    """CNN feature extractor followed by a bidirectional LSTM and CTC logits."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, images):
        features = self.cnn(images)                 # [B, 512, 1, T]
        sequence = features.squeeze(2).transpose(1, 2)  # [B, T, 512]
        sequence, _ = self.sequence(sequence)
        return self.classifier(sequence).log_softmax(dim=2)

    @staticmethod
    def output_lengths(widths):
        # Two 2x2 max-pooling layers use floor rounding for the width.
        return widths // 4


def greedy_decode(log_probs, input_lengths):
    """Simple CTC greedy decoder; collapses repeats and removes blank."""
    paths = log_probs.argmax(dim=2).detach().cpu()
    decoded = []
    for row, length in zip(paths, input_lengths.detach().cpu().tolist()):
        chars, previous = [], BLANK_ID
        for token in row[:length].tolist():
            if token != BLANK_ID and token != previous:
                chars.append(ID_TO_CHAR.get(token, ""))
            previous = token
        decoded.append("".join(chars))
    return decoded



def ctc_minimum_lengths(flat_targets, target_lengths):
    """Minimum CTC time steps per sample, accounting for adjacent repeats."""
    minimums = []
    offset = 0
    for length in target_lengths.detach().cpu().tolist():
        target = flat_targets[offset:offset + length]
        repeats = int((target[1:] == target[:-1]).sum().item()) if length > 1 else 0
        minimums.append(int(length) + repeats)
        offset += int(length)
    return torch.tensor(minimums, dtype=torch.long, device=target_lengths.device)
