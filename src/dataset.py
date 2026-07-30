"""Data handling for DeepLOB reproduction.

Two paths are provided:

1. ``RandomLOBDataset`` -- generates synthetic (B, 1, T, 144) windows and
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

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class FI2010WindowDataset(Dataset):
    """Real FI-2010 benchmark loader (Colab / GPU).

    Design (memory-safe, per the project plan):
      - reads the normalised .npy (official FI-2010 layout) produced by
        convert_fi2010.py
      - keeps features float32; builds sliding windows via stride tricks
        (NO full copy of all windows -> avoids the ~77 GiB OOM trap)
      - selects the label column via K_TO_LABEL_COLUMN[k]  <-- the k-pitfall

    Two protocol modes (chosen by `protocol`, requires `meta_path`):

      * "standard9" (paper Table II, 9-fold anchored CV):
          for a given test fold i, train on the Training segments of all
          OTHER 8 folds, test on the Testing segment of fold i. The validation
          set is the time-last `val_frac` of each training segment (NO random
          permutation -> preserves time order, no leakage).

      * "light_setup2" (cheaper, ~20w train / 14w test):
          train on the Training segment of CF_7 only; test on the Testing
          segments of CF_7/8/9. Good for a quick credible run before the full
          9-fold sweep.

    Anti-leakage guarantee: windows are built PER SEGMENT. A 100-row window is
    never allowed to straddle two segments (i.e. two different stocks/days),
    so no future information leaks across boundaries.

    Backward-compatible mode: if `meta_path` is None but `folds_path` is given,
    the OLD fold-id logic runs (random-permuted train/val). Deprecated but kept
    so existing Drive checkpoints keep loading.

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
        protocol: str = "standard9",
        meta_path: str | None = None,
        test_cf: int | None = None,
        light_test_cf: list[int] | None = None,
    ):
        if k not in K_TO_LABEL_COLUMN:
            raise ValueError(f"k must be one of {list(K_TO_LABEL_COLUMN)}, got {k}")
        self.k = k
        self.window_size = window_size
        self.split = split
        self.label_col = K_TO_LABEL_COLUMN[k]

        # 1) Load raw data. mmap_mode="r" -> not loaded into RAM as a whole
        #    (see _load); builders slice only the rows they actually need.
        data = self._load(data_path)  # mmap'd (N, 149)

        # 3) Select which segments feed this split, then build windows.
        if meta_path is not None:
            # meta path: builder slices only the needed segments from the mmap
            # (light_setup2 ~1/9 of rows, standard9 ~8/9 train + 1/9 test), so
            # peak RAM stays tiny. Label normalization happens inside too.
            windows, win_labels = self._build_from_meta(
                data, meta_path, protocol, split, val_frac,
                test_fold, test_cf, light_test_cf,
            )
        else:
            # Legacy paths still need the full feature/label arrays.
            features = data[:, :NUM_FEATURES].astype(np.float32)
            raw_labels = data[:, self.label_col].astype(np.int64)
            self.label_map = {v: i for i, v in enumerate(sorted(np.unique(raw_labels)))}
            labels = np.array([self.label_map[v] for v in raw_labels], dtype=np.int64)
            if folds_path is not None:
                # Deprecated backward-compatible path.
                windows, win_labels = self._build_from_folds(
                    features, labels, folds_path, split, val_frac, seed, test_fold,
                )
            else:
                # Proportional split (backward compatible smoke-test mode).
                windows, win_labels = self._build_proportional(
                    features, labels, split, split_fracs
                )
            # Legacy builders return a single concatenated (num_win, w, D)
            # array; wrap it into the list-of-segments shape the rest of the
            # class expects.
            windows = [windows]
            win_labels = [win_labels]

        if len(windows) == 0:
            raise ValueError(
                f"no windows for split={split!r} (protocol={protocol}, "
                f"test_fold={test_fold}, test_cf={test_cf}). Check meta/segment selection."
            )
        # windows / win_labels are LISTS (one entry per segment), NOT
        # concatenated. Each entry is a (num_seg_win, w, D) float32 view. We
        # index across them lazily in __getitem__, so peak RAM stays at ONE
        # segment's windows instead of every window at once (which for
        # light_setup2 test = ~229k x 100 x 144 x 4B ≈ 12.3 GiB and OOMs Colab).
        self._windows = windows          # list[np.ndarray], each (n_i, w, D)
        self._window_labels = win_labels  # list[np.ndarray], each (n_i,)
        self._seg_lens = [w.shape[0] for w in windows]
        self._seg_offsets = np.concatenate(([0], np.cumsum(self._seg_lens)))

    # ------------------------------------------------------------------
    # Window builder helpers
    # ------------------------------------------------------------------
    def _windows_in_block(
        self, block_feat: np.ndarray, block_lab: np.ndarray, val_frac: float, hold_val: bool
    ):
        """Build non-leaking windows inside ONE contiguous segment.

        Returns (train_windows, train_labels, val_windows, val_labels). When
        `hold_val` is False, all windows go to train and val is empty. The
        validation windows are the TIME-LAST `val_frac` of the segment (never
        randomly shuffled). Because each call handles exactly one segment, a
        window can never cross a segment boundary.
        """
        w = self.window_size
        if block_feat.shape[0] < w:
            empty = np.empty((0, w, block_feat.shape[1]), dtype=np.float32)
            return empty, np.empty((0,), dtype=np.int64), empty, np.empty((0,), dtype=np.int64)
        win = np.lib.stride_tricks.sliding_window_view(
            block_feat, window_shape=w, axis=0
        )  # (num_win, D, w)
        win = np.transpose(win, (0, 2, 1))  # (num_win, w, D)
        win_lab = block_lab[w - 1 :]  # (num_win,)
        n_val = int(len(win) * val_frac) if hold_val else 0
        # Train = earlier windows, Val = time-last n_val windows. To keep the
        # two sets ROW-DISJOINT (no leakage), trim the trailing (w-1) train
        # windows: a train window ending at row r must end before the first val
        # window starts. On real FI-2010 segments (tens of thousands of rows)
        # this drops a negligible 99 windows; if a segment is too small to spare
        # them, we fall back to a plain window split (no trim).
        cut = len(win) - n_val
        if hold_val:
            train_end = cut - (w - 1)
            if train_end < 1:
                train_end = cut  # segment too small; accept boundary overlap
            tr_w, tr_l = win[:train_end], win_lab[:train_end]
            va_w, va_l = win[cut:], win_lab[cut:]
        else:
            tr_w, tr_l = win, win_lab
            va_w, va_l = win[:0], win_lab[:0]
        return tr_w, tr_l, va_w, va_l

    def _assemble(self, seg_list, features, labels, split, val_frac):
        """Build per-segment windows for the requested `split`.

        Returns (windows_list, labels_list) where each entry is the window
        array / label array of ONE segment. We deliberately DO NOT concatenate
        across segments here: concatenation of all windows for the test split of
        light_setup2 would materialise ~12.3 GiB at once and OOM Colab. Instead
        the lists are indexed lazily in __getitem__ (one window copied on demand).
        """
        tr_w = []
        tr_l = []
        va_w = []
        va_l = []
        for seg in seg_list:
            bf = features[seg["start"] : seg["end"]]
            bl = labels[seg["start"] : seg["end"]]
            hold_val = split in ("train", "val")
            tw, tl, vw, vl = self._windows_in_block(bf, bl, val_frac, hold_val)
            tr_w.append(tw)
            tr_l.append(tl)
            va_w.append(vw)
            va_l.append(vl)
        if split == "train":
            return tr_w, tr_l
        if split == "val":
            return va_w, va_l
        # test: take the train-side windows (hold_val=False -> all windows)
        return tr_w, tr_l

    def _build_from_meta(
        self, data, meta_path, protocol, split, val_frac,
        test_fold, test_cf, light_test_cf=None,
    ):
        import json

        with open(os.path.expanduser(meta_path), encoding="utf-8") as fh:
            meta = json.load(fh)
        segments = meta["segments"]

        if protocol == "standard9":
            if test_fold is None:
                raise ValueError("protocol=standard9 requires test_fold")
            train_segs = [
                s for s in segments if s["role"] == "train" and s["cf"] != test_fold + 1
            ]
            # test segment: the Testing segment of the test fold (cf is 1-based)
            test_segs = [
                s for s in segments if s["role"] == "test" and s["cf"] == test_fold + 1
            ]
        elif protocol == "light_setup2":
            if test_cf is None:
                raise ValueError("protocol=light_setup2 requires test_cf (the CF to train on)")
            test_cf_list = light_test_cf or [7, 8, 9]
            # Train on the Training segment of `test_cf`; test on the Testing
            # segments of every CF in the test list (default 7/8/9).
            train_segs = [s for s in segments if s["role"] == "train" and s["cf"] == test_cf]
            test_segs = [
                s for s in segments if s["role"] == "test" and s["cf"] in test_cf_list
            ]
        else:
            raise ValueError(f"unknown protocol {protocol!r}")

        # Only the segments this split needs are materialised from the mmap.
        needed = test_segs if split == "test" else train_segs
        blocks = [np.ascontiguousarray(data[s["start"] : s["end"]]) for s in needed]
        if not blocks:
            local = np.empty((0, data.shape[1]), dtype=np.float32)
        else:
            local = np.concatenate(blocks, axis=0)  # (M, 149), M = needed rows only
        # Local segment coordinates (contiguous 0..M) for _assemble.
        local_segs = []
        off = 0
        for s in needed:
            n = s["end"] - s["start"]
            local_segs.append({"start": off, "end": off + n})
            off += n

        features = local[:, :NUM_FEATURES].astype(np.float32)
        raw_labels = local[:, self.label_col].astype(np.int64)
        # Normalize labels to {0,1,2} (FI-2010 may use 1/2/3 or 0/1/2).
        self.label_map = {v: i for i, v in enumerate(sorted(np.unique(raw_labels)))}
        labels = np.array([self.label_map[v] for v in raw_labels], dtype=np.int64)

        if split == "test":
            return self._assemble(local_segs, features, labels, "test", val_frac)
        return self._assemble(local_segs, features, labels, split, val_frac)

    def _build_from_folds(
        self, features, labels, folds_path, split, val_frac, seed, test_fold
    ):
        if test_fold is None:
            raise ValueError("folds_path given but test_fold is None")
        fold_ids = np.load(os.path.expanduser(folds_path))
        if fold_ids.shape[0] != features.shape[0]:
            raise ValueError(
                f"folds array length {fold_ids.shape[0]} != data rows {features.shape[0]}"
            )
        test_idx = np.where(fold_ids == test_fold)[0]
        train_idx = np.where(fold_ids != test_fold)[0]
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
        feats_split = features[sel]
        labels_split = labels[sel]
        if feats_split.shape[0] < self.window_size:
            raise ValueError(
                f"not enough rows ({feats_split.shape[0]}) for window {self.window_size}"
            )
        win = np.lib.stride_tricks.sliding_window_view(
            feats_split, window_shape=self.window_size, axis=0
        )
        windows = np.transpose(win, (0, 2, 1))
        win_labels = labels_split[self.window_size - 1 :]
        return windows, win_labels

    def _build_proportional(self, features, labels, split, split_fracs):
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
        feats_split = features[s:e]
        labels_split = labels[s:e]
        if feats_split.shape[0] < self.window_size:
            raise ValueError(
                f"not enough rows ({feats_split.shape[0]}) for window {self.window_size}"
            )
        win = np.lib.stride_tricks.sliding_window_view(
            feats_split, window_shape=self.window_size, axis=0
        )
        windows = np.transpose(win, (0, 2, 1))
        win_labels = labels_split[self.window_size - 1 :]
        return windows, win_labels

    @staticmethod
    def _load(path: str) -> np.ndarray:
        # 只接受官方 FI-2010 的 .npy/.npz（由 convert_fi2010.py 产出）。
        # 第三方 CSV 镜像布局不同且标签被污染，本项目不使用。
        # 用 mmap_mode="r"：文件不整块读进 RAM，按需从磁盘映射读取，避免
        # 204 万行（~1.2GB）一次性加载把 Colab 默认内存顶爆（SIGKILL/OOM）。
        path = os.path.expanduser(path)
        if not (path.endswith(".npy") or path.endswith(".npz")):
            raise ValueError(f"只支持 .npy/.npz 文件，不支持：{path}")
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 3:  # 个别存档按天存成 3 维，展平成 (N, 149)
            arr = arr.reshape(-1, arr.shape[-1])
        return arr

    def __len__(self) -> int:
        return int(self._seg_offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global window index to (segment_idx, local_index)."""
        # np.searchsorted finds the first offset > index; subtract 1 -> owning seg.
        seg_idx = int(np.searchsorted(self._seg_offsets, index, side="right") - 1)
        local = int(index - self._seg_offsets[seg_idx])
        return seg_idx, local

    def __getitem__(self, index: int):
        # Extract ONE window on demand (no full copy of all windows). Only the
        # owning segment's window array is touched; the global index is mapped
        # to (segment, local) via the prefix-sum offsets. self._windows[i] is a
        # float32 view of shape (n_i, w, D); indexing gives (w, D). Add channel
        # dim -> (1, w, D) to match Conv2d (N,C,H,W) after the DataLoader stacks
        # a batch. .copy() makes it writable; data is already float32.
        seg_idx, local = self._locate(index)
        x = np.expand_dims(self._windows[seg_idx][local].copy(), axis=0)  # (1, w, D)
        y = int(self._window_labels[seg_idx][local])
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
