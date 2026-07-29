"""Data handling for DeepLOB reproduction.

Two paths are provided:

1. ``RandomLOBDataset`` -- generates synthetic (B, 1, T, 40) windows and
   random 3-class labels. Used for local smoke tests when the real FI-2010
   data is not available yet (per the agreed scope: no real data wiring yet).

2. ``FI2010WindowDataset`` -- the real FI-2010 benchmark loader. Reads the
   normalised mirror (.npy/.csv), casts to float32, builds sliding windows via
   stride tricks (memory-safe), and selects the label column via
   ``K_TO_LABEL_COLUMN[k]``. Designed to run on Colab with the HF mirror; a
   small synthetic .npy can also exercise it locally (see smoke_test.py).

标签列的关键说明（容易出错，详见项目 README）：
  FI-2010 提供多种归一化版本和预先算好的标签列。论文的预测时间跨度
  k = 10、20、50、100 是归一化文件里标签列的索引，不是原始事件个数。
  官方 FI-2010 .txt 文件带有 5 个标签列（0-indexed 144-148），存放
  k = 10、20、50、100 的三分类标签以及一额外时间跨度。用 K_TO_LABEL_COLUMN
  把 k 映射到 0-indexed 列。复现哪个实验就取对应列，取错列会静默地训练出
  一个不同的任务。
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


# FI-2010 OFFICIAL data layout (Ntakaris et al. 2017, arXiv:1705.03233):
#   Official .txt files: each row has 144 features + 5 label columns (0-indexed 144-148).
#   Label encoding: 1=up, 2=stationary, 3=down. The 5 columns are 5 classification
#   horizons; the FIRST label column (0-indexed 144) is k=10 used in the
#   paper's main Table II. We map k -> 0-indexed column below. If your file's label
#   order differs, adjust these indices.
#   NOTE: third-party mirrors (e.g. shanehans/FI2010 CSV) have DIFFERENT layouts
#   (we saw 130 features + 15 junk columns + dirty labels = 150). Always verify with
#   np.unique on the real file before trusting these constants.
K_TO_LABEL_COLUMN = {
    10: 144,
    20: 145,
    50: 146,
    100: 147,
}
WINDOW_SIZE = 100
NUM_FEATURES = 144
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
    """Real FI-2010 benchmark loader (Colab / GPU).

    Design (memory-safe, per the project plan):
      - reads the normalised .npy (official FI-2010 layout) produced by
        convert_fi2010.py
      - keeps features float32; builds sliding windows via stride tricks
        (NO full copy of all windows -> avoids the ~77 GiB OOM trap)
      - selects the label column via K_TO_LABEL_COLUMN[k]  <-- the k-pitfall

    Two split modes:
      * Proportional (default): splits the concatenated .npy into
        train/val/test by row fraction. Good for a quick pipeline smoke test,
        but NOT the paper's protocol.
      * Fold-based (pass `folds_path` + `test_fold`): uses the fold-id array
        saved by convert_fi2010.py. `test_fold` rows become the test set;
        all OTHER folds become train (and a `val_frac` slice of train is held
        out for early stopping). This matches the paper's Setup 2 (9-fold
        anchored cross-validation): run once per fold i, average the 9 tests.

    Expected column layout of the loaded array (N rows):
        cols [0:144]  -> 144 LOB features  (FI-2010 official .txt format)
        cols [144:149] -> 5 label columns; k=10/20/50/100 -> cols 144/145/146/147
    """

    def __init__(
        self,
        data_path: str,
        k: int = 10,
        window_size: int = WINDOW_SIZE,
        split: str = "train",
        split_fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 0,
        folds_path: str | None = None,
        test_fold: int | None = None,
        val_frac: float = 0.1,
    ):
        if k not in K_TO_LABEL_COLUMN:
            raise ValueError(f"k must be one of {list(K_TO_LABEL_COLUMN)}, got {k}")
        self.k = k
        self.window_size = window_size
        self.split = split
        self.label_col = K_TO_LABEL_COLUMN[k]

        # 1) Load raw data (float32 right away).
        data = self._load(data_path)  # (N, 149) float32
        features = data[:, :NUM_FEATURES].astype(np.float32)
        raw_labels = data[:, self.label_col].astype(np.int64)

        # 2) Normalize labels to {0,1,2}. FI-2010 labels are sometimes 1/2/3
        #    (down/stationary/up) or 0/1/2; map by sorting unique values so we
        #    never assume the exact encoding.
        self.label_map = {v: i for i, v in enumerate(sorted(np.unique(raw_labels)))}
        labels = np.array([self.label_map[v] for v in raw_labels], dtype=np.int64)

        # 3) Pick row indices for this split.
        if folds_path is not None:
            if test_fold is None:
                raise ValueError("folds_path given but test_fold is None")
            fold_ids = np.load(os.path.expanduser(folds_path))
            if fold_ids.shape[0] != features.shape[0]:
                raise ValueError(
                    f"folds array length {fold_ids.shape[0]} != data rows "
                    f"{features.shape[0]}"
                )
            test_idx = np.where(fold_ids == test_fold)[0]
            train_idx = np.where(fold_ids != test_fold)[0]
            # hold out val_frac of train for early stopping
            rng = np.random.default_rng(seed)
            train_idx = rng.permutation(train_idx)
            n_val = int(len(train_idx) * val_frac)
            if split == "train":
                sel = train_idx[n_val:]
            elif split == "val":
                sel = train_idx[:n_val]
            elif split == "test":
                sel = test_idx
            else:
                raise ValueError(f"split must be train/val/test, got {split}")
        else:
            # Proportional split (backward compatible smoke-test mode).
            n = features.shape[0]
            tr = int(n * split_fracs[0])
            va = int(n * (split_fracs[0] + split_fracs[1]))
            if split == "train":
                s, e = 0, tr
            elif split == "val":
                s, e = tr, va
            elif split == "test":
                s, e = va, n
            else:
                raise ValueError(f"split must be train/val/test, got {split}")
            sel = np.arange(s, e)

        feats_split = features[sel]
        labels_split = labels[sel]

        # 4) Sliding windows WITHOUT materialising all of them.
        #    Slide a length-`window_size` window over the TIME axis (rows).
        #    feats_split is (N, D). sliding_window_view on axis=0 with
        #    window_shape=w yields (num_win, D, w); we transpose to
        #    (num_win, w, D) which is what the model/DataLoader expect.
        #    CRITICAL: we must NOT call .reshape().astype() on it here -- that
        #    would force a full copy (204w * 100 * 144 float32 = ~77 GiB) and
        #    OOM. We keep the view and extract one window per __getitem__.
        if feats_split.shape[0] < window_size:
            raise ValueError(
                f"not enough rows ({feats_split.shape[0]}) for window {window_size}"
            )
        win = np.lib.stride_tricks.sliding_window_view(
            feats_split, window_shape=window_size, axis=0
        )  # (num_win, D, w)
        self.windows = np.transpose(win, (0, 2, 1))  # (num_win, w, D)
        # Label for a window ending at t is the label at t (standard convention).
        self.window_labels = labels_split[window_size - 1 :]

    @staticmethod
    def _load(path: str) -> np.ndarray:
        path = os.path.expanduser(path)
        if path.endswith(".npy") or path.endswith(".npz"):
            arr = np.load(path)
            if arr.ndim == 3:  # some mirrors store per-day arrays
                arr = arr.reshape(-1, arr.shape[-1])
            return arr.astype(np.float32)
        if path.endswith(".csv"):
            # HF mirror CSV: headerless, 44 columns. Use float32 directly.
            return np.loadtxt(path, delimiter=",", dtype=np.float32)
        raise ValueError(f"unsupported data file: {path}")

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        # Extract ONE window on demand (no full copy of all windows).
        # self.windows is a float32 view of shape (num_win, w, D); indexing
        # gives (w, D). Add channel dim -> (1, w, D), matching Conv2d (N,C,H,W)
        # after the DataLoader stacks a batch into (B, 1, w, D). The data is
        # already float32 (from _load), so no cast is needed here.
        x = np.expand_dims(self.windows[idx].copy(), axis=0)  # (1, w, D), copy->writable
        y = int(self.window_labels[idx])
        return x, y


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
