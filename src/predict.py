"""Predict a Khmer consonant from one image. Used by the demo app (app/app.py) and the evaluation.

    from src.predict import predict
    predict(drawing)          # [{"id": 0, "letter": "ក", "romanization": "kor", "confidence": 0.94}, ...]

`drawing` can be anything a canvas or photo gives you (gray / RGB / RGBA uint8 array, light-on-dark or
dark-on-light). It is converted to the training format by `preprocess_drawing`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .data import LETTERS, ROMAN, ROOT, preprocess_drawing
from .model import SimpleCNN
from .train import predict_probs

MODEL_PATH = ROOT / "models" / "cnn.pt"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> SimpleCNN:
    """Load the trained CNN once and keep it in memory."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/02_train.ipynb first.")
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def predict(image: np.ndarray, top_k: int = 3, already_preprocessed: bool = False) -> list[dict]:
    """Return the `top_k` most likely letters, best first, each with its probability."""
    x = np.asarray(image, np.float32) if already_preprocessed else preprocess_drawing(image)
    probs = predict_probs(load_model(), x[None])[0]
    best = np.argsort(probs)[::-1][:top_k]
    return [{"id": int(k), "letter": LETTERS[k], "romanization": ROMAN[k], "confidence": float(probs[k])} for k in best]
