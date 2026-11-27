"""Training helpers for the CNN. Used by notebooks/02_train.ipynb.

The main function is `train_model(...)`: give it training images and labels, get back a trained model
and a table with the loss/accuracy of every epoch.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import augment
from .model import SimpleCNN


def get_device() -> torch.device:
    """GPU if there is one (Colab), otherwise CPU. CPU is plenty for 64x64 images and keeps results repeatable."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Fix all random number generators so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ArrayDataset(Dataset):
    """Wraps numpy arrays (N, 64, 64) and labels (N,). Applies random augmentation if `aug=True`."""

    def __init__(self, X: np.ndarray, y: np.ndarray, aug: bool = False, seed: int = 0):
        self.X, self.y, self.aug = X, y, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = augment(self.X[i], self.rng) if self.aug else self.X[i]
        return torch.from_numpy(x[None].astype(np.float32)), int(self.y[i])


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Class probabilities for images X (N, 64, 64). Returns (N, 33)."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), 256):
        batch = torch.from_numpy(X[i:i + 256, None].astype(np.float32)).to(device)
        out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 33))


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                epochs: int = 40, lr: float = 3e-4, batch_size: int = 32,
                aug: bool = True, seed: int = 42, verbose: bool = True):
    """Train a SimpleCNN from scratch.

    lr = 3e-4 on purpose: with 1e-3 training was unstable on this data (the loss jumped above its starting
    value and some runs ended at ~0.47 val accuracy instead of ~0.8).

    Returns (model, history). `history` has one row per epoch: train_loss, train_acc and,
    if validation data is given, val_loss and val_acc. The model returned is the one from the LAST epoch.
    """
    set_seed(seed)
    device = get_device()
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(ArrayDataset(X_train, y_train, aug=aug, seed=seed),
                        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        row = {"epoch": epoch, "train_loss": total_loss / len(loader.dataset), "train_acc": correct / len(loader.dataset)}

        if X_val is not None:
            p = predict_probs(model, X_val, device)
            row["val_loss"] = float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-9).mean())
            row["val_acc"] = float((p.argmax(1) == y_val).mean())
        rows.append(row)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print("  ".join(f"{k} {v:.3f}" if k != "epoch" else f"epoch {v:2d}" for k, v in row.items()))

    return model, pd.DataFrame(rows)
