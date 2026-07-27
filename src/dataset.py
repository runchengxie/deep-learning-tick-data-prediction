"""Data handling for DeepLOB reproduction.

Two paths are provided:

1. ``RandomLOBDataset`` -- generates synthetic (B, 1, T, 40) windows and
   random 3-class labels. Used for local smoke tests when the real FI-2010
   data is not available yet (per the agreed scope: no real data wiring yet).

2. ``FI2010WindowDataset`` -- the *interface* for the real FI-2010 benchmark.
   It is intentionally left un-wired to disk: we only define the contract
   (window size, 40 features, label column) so the training code can be
   written against it now and connected to the Hugging Face mirror later on
   Colab.

CRITICAL labelling note (easy to get wrong, see project README discussion):
  FI-2010 ships several normalised versions and pre-computed label columns.
  The paper's prediction horizons k = 10, 20, 50, 100 are INDICES into the
  label columns of the normalised file, NOT the number of raw events. The
  last 4 columns of the FI-2010 normalised data are the 3-class labels for
  k = 10, 20, 50, 100 (in that order) plus a stationary/raw column. Pick the
  correct column for the experiment you want to reproduce, otherwise you may
  happily reproduce a *different* task.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# FI-2010 normalised data layout (per the official dataset paper [1]):
#   first 40 columns : normalised LOB features  [pa,va,pb,vb] x 10 levels
#   next  4 columns  : 3-class labels for k = 10, 20, 50, 100  (and a raw col)
# We expose the mapping so dataset code and configs stay in sync.
K_TO_LABEL_COLUMN = {
    10: 40,
    20: 41,
    50: 42,
    100: 43,
}
WINDOW_SIZE = 100
NUM_FEATURES = 40
NUM_CLASSES = 3


class RandomLOBDataset(Dataset):
    """Synthetic dataset for smoke tests (no real data dependency)."""

    def __init__(
        self,
        num_samples: int = 2000,
        window_size: int = WINDOW_SIZE,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        # Shape (N, 1, T, D) to match the Conv2d (N,C,H,W) convention we use.
        self.x = rng.standard_normal((num_samples, 1, window_size, num_features)).astype(
            np.float32
        )
        # Roughly balanced labels.
        self.y = rng.integers(0, num_classes, size=(num_samples,)).astype(np.int64)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class FI2010WindowDataset(Dataset):
    """Interface for the real FI-2010 benchmark (NOT wired to disk yet).

    To activate on Colab later:
      - load the normalised .npy/.npz mirror (shanehans/FI2010)
      - build sliding windows of length ``window_size`` over the 40-feature
        columns WITHOUT copying all windows at once (use np.lib.stride_tricks
        or a tf.data/Dataset generator to avoid the 6.5GB float64 trap)
      - select the label column via ``K_TO_LABEL_COLUMN[k]``
      - cast features to float32 immediately
    """

    def __init__(
        self,
        data_path: str,
        k: int = 10,
        window_size: int = WINDOW_SIZE,
        split: str = "train",
    ):
        if k not in K_TO_LABEL_COLUMN:
            raise ValueError(f"k must be one of {list(K_TO_LABEL_COLUMN)}, got {k}")
        self.data_path = data_path
        self.k = k
        self.window_size = window_size
        self.split = split
        self.label_col = K_TO_LABEL_COLUMN[k]
        # Real loading happens here on Colab; left as a clear TODO so the
        # contract is explicit rather than silently broken.
        raise NotImplementedError(
            "FI2010WindowDataset is not wired to disk in this skeleton. "
            "Implement windowing + label selection on Colab using the "
            "K_TO_LABEL_COLUMN mapping above."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def get_dummy_batch(
    batch_size: int = 8, window_size: int = WINDOW_SIZE, num_features: int = NUM_FEATURES
) -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny helper that returns a random batch for ad-hoc checks."""
    x = torch.randn(batch_size, 1, window_size, num_features)
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x, y


if __name__ == "__main__":
    d = RandomLOBDataset(num_samples=16)
    x, y = d[0]
    print("sample x shape:", x.shape, "dtype:", x.dtype)
    print("sample y:", y.item())
    print("dataset length:", len(d))
