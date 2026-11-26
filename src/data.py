"""Data loading, preprocessing and augmentation for the Khmer consonant dataset.

Owner: Person A. Used by training (B), evaluation (C) and the demo app (C).

Typical use:
    from src.data import load_split, preprocess_drawing, augment

    X_train, y_train = load_split("train")      # X: (N, 64, 64) float32, ink = 1, paper = 0
    X_test,  y_test  = load_split("test")

The dataset is built by notebooks/01_prepare_dataset.ipynb into outputs/dataset/.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "outputs" / "dataset"
EXCLUDE_CSV = ROOT / "data" / "exclude.csv"
CLASSES_JSON = ROOT / "data" / "classes.json"

IMG_SIZE = 64
NUM_CLASSES = 33

ROMAN = "ka kha ko kho ngo cha chha cho chho nho da ttha do ttho na ta tha to tho no ba pha po pho mo yo ro lo vo sa ha la qa".split()
LETTERS = list("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ")
assert len(ROMAN) == len(LETTERS) == NUM_CLASSES


# --------------------------------------------------------------------------- labels
def write_classes_json(path: Path = CLASSES_JSON) -> None:
    """Write the fixed class list (index -> letter, romanization) shared by the whole team."""
    classes = [{"id": i, "letter": LETTERS[i], "romanization": ROMAN[i]} for i in range(NUM_CLASSES)]
    path.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(dataset_dir: Path = DATASET_DIR, apply_exclusions: bool = True) -> pd.DataFrame:
    """Read labels.csv. Rows listed in data/exclude.csv (bad crops) are dropped."""
    df = pd.read_csv(dataset_dir / "labels.csv", encoding="utf-8-sig", dtype={"writer_id": str})
    if apply_exclusions and EXCLUDE_CSV.exists():
        bad = pd.read_csv(EXCLUDE_CSV, comment="#")["filename"].astype(str)
        df = df[~df.filepath.map(lambda p: Path(p).name).isin(set(bad))]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- images
def to_model_input(gray_dark_on_white: np.ndarray) -> np.ndarray:
    """Dataset image (dark ink on white, uint8) -> float32 in [0, 1] with ink = 1, paper = 0."""
    return (255 - gray_dark_on_white).astype(np.float32) / 255.0


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return to_model_input(img)


def load_split(split: str, dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for 'train', 'val' or 'test'. X: (N, 64, 64) float32, y: (N,) int64."""
    assert split in {"train", "val", "test"}, split
    df = load_labels(dataset_dir)
    df = df[df.split == split]
    X = np.stack([load_image(dataset_dir / f) for f in df.filepath]) if len(df) else np.empty((0, IMG_SIZE, IMG_SIZE), np.float32)
    return X, df.label_id.to_numpy(np.int64)


def compute_mean_std(X: np.ndarray) -> tuple[float, float]:
    """Pixel mean and std of the TRAIN set only (never use val/test statistics)."""
    return float(X.mean()), float(X.std())


# --------------------------------------------------------------------------- canvas / photo input (demo app)
def preprocess_drawing(img: np.ndarray, invert: bool | None = None) -> np.ndarray:
    """Turn any user drawing into the same format as the training data.

    img: 2-D (gray) or 3-/4-D (RGB/RGBA) uint8 array, e.g. from an HTML canvas or Gradio sketchpad.
    invert: True if the drawing is light ink on a dark background. None = guess from the corners.
    Returns a (64, 64) float32 array, ink = 1, paper = 0, centred with the same margin as training.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA canvas: transparent pixels count as white paper
            alpha = img[:, :, 3:4] / 255.0
            img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        img = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    if invert is None:
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        invert = np.median(border) < 128  # dark background -> light ink
    ink = img if invert else 255 - img
    _, ink = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(ys) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    s = max(h, w)
    pad = int(0.15 * s)  # same 15% margin as notebook 01
    sq = np.zeros((s + 2 * pad, s + 2 * pad), np.uint8)
    oy, ox = pad + (s - h) // 2, pad + (s - w) // 2
    sq[oy:oy + h, ox:ox + w] = ink
    out = cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- augmentation
def augment(x: np.ndarray, rng: np.random.Generator | None = None,
            max_rot: float = 10, max_shift: float = 0.1, scale: tuple[float, float] = (0.9, 1.1),
            thickness: bool = True) -> np.ndarray:
    """Random small rotation / shift / zoom (and optional stroke thickening or thinning).

    x: (64, 64) float32, ink = 1. Returns the same shape. Use on TRAIN images only.
    """
    rng = rng or np.random.default_rng()
    c = IMG_SIZE / 2
    M = cv2.getRotationMatrix2D((c, c), rng.uniform(-max_rot, max_rot), rng.uniform(*scale))
    M[:, 2] += rng.uniform(-max_shift, max_shift, 2) * IMG_SIZE
    out = cv2.warpAffine(x, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    if thickness and rng.random() < 0.5:  # mimics thick canvas strokes vs thin pen strokes
        k = np.ones((2, 2), np.uint8)
        out = cv2.dilate(out, k) if rng.random() < 0.5 else cv2.erode(out, k)
    return out


class KhmerDataset:
    """PyTorch-style Dataset (works with torch.utils.data.DataLoader, torch is not required to import this).

    Returns (tensor-like array of shape (1, 64, 64), label). Augments only when train=True.
    """

    def __init__(self, split: str, train: bool | None = None, seed: int = 0):
        self.X, self.y = load_split(split)
        self.train = (split == "train") if train is None else train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = augment(self.X[i], self.rng) if self.train else self.X[i]
        return x[None].astype(np.float32), int(self.y[i])
