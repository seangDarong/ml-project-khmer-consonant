"""The CNN used for Khmer consonant classification.

Input : (batch, 1, 64, 64) float tensor, ink = 1, paper = 0
Output: (batch, 33) raw scores (logits). Apply softmax to get probabilities.
"""
import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool. Halves the image size, learns c_out patterns."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """3 conv blocks (32 -> 64 -> 128 filters) followed by a small classifier.

    64x64 -> 32x32 -> 16x16 -> 8x8, so the flattened size is 128 * 8 * 8 = 8192.
    """

    def __init__(self, num_classes: int = 33, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64), conv_block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),   # randomly switches off half the units while training, reduces overfitting
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
