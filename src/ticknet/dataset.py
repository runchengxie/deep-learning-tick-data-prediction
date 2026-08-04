"""DeepLOB 使用的数据集。

FI-2010 原始矩阵共有 149 行。前 40 行是十档买卖盘的价格和数量，
第 40 至 143 行是手工特征，最后 5 行是预测标签。DeepLOB 只读取
前 40 个原始订单簿特征。

官方发布的 ``CF_1`` 至 ``CF_9`` 已经包含论文使用的训练段和测试段。
本模块按源文件片段构造窗口，避免在多个 ``CF`` 文件拼接处生成跨文件窗口。
公开矩阵没有提供文件内部的股票和日期边界，因此无法进一步识别这些边界。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

K_TO_LABEL_COLUMN = {
    10: 144,
    20: 145,
    30: 146,
    50: 147,
    100: 148,
}
WINDOW_SIZE = 100
NUM_FEATURES = 40
NUM_CLASSES = 3
RAW_FEATURE_COLUMNS = 144
TOTAL_COLUMNS = 149
SETUP2_TRAIN_CF = 7
SETUP2_TEST_CFS = (7, 8, 9)


class RandomLOBDataset(Dataset):
    """供本地冒烟训练使用的合成数据集。"""

    def __init__(
        self,
        num_samples: int = 2_000,
        window_size: int = WINDOW_SIZE,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal(
            (num_samples, 1, window_size, num_features),
            dtype=np.float32,
        )
        self.y = rng.integers(0, num_classes, size=num_samples, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        return self.x[index], self.y[index]


class FI2010WindowDataset(Dataset):
    """从转换后的 FI-2010 数据按需读取滑动窗口。

    ``setup1`` 对应论文 Table I。每次使用一个 ``CF`` 的 Training 文件
    训练，并在同一 ``CF`` 的 Testing 文件上测试。

    ``setup2`` 对应论文 Table II。它使用 ``CF_7`` 的 Training 文件训练，
    并在 ``CF_7``、``CF_8`` 和 ``CF_9`` 的 Testing 文件上测试。
    """

    def __init__(
        self,
        data_path: str,
        meta_path: str,
        *,
        k: int = 10,
        window_size: int = WINDOW_SIZE,
        split: str = "train",
        protocol: str = "setup2",
        test_cf: int | None = None,
        val_frac: float = 0.2,
    ):
        if k not in K_TO_LABEL_COLUMN:
            raise ValueError(f"k 应为 {list(K_TO_LABEL_COLUMN)} 中的一个，收到 {k}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split 应为 train、val 或 test，收到 {split}")
        if protocol not in {"setup1", "setup2"}:
            raise ValueError(f"protocol 应为 setup1 或 setup2，收到 {protocol}")
        if not 0 < val_frac < 1:
            raise ValueError(f"val_frac 应在 0 和 1 之间，收到 {val_frac}")
        if protocol == "setup1" and (test_cf is None or not 1 <= test_cf <= 9):
            raise ValueError("setup1 需要 1 至 9 的 test_cf")

        self.k = k
        self.window_size = window_size
        self.split = split
        self.label_col = K_TO_LABEL_COLUMN[k]

        data = self._load(data_path)
        segments = self._load_segments(meta_path, rows=int(data.shape[0]))
        selected = self._select_segments(
            segments,
            protocol=protocol,
            split=split,
            test_cf=test_cf,
        )
        self._windows, self._window_labels = self._build_windows(
            data,
            selected,
            split=split,
            val_frac=val_frac,
        )

        if not self._windows:
            raise ValueError(f"{protocol} 的 {split} 切分没有可用窗口，请检查数据和元数据")
        self._segment_lengths = [int(windows.shape[0]) for windows in self._windows]
        self._segment_offsets = np.concatenate(
            ([0], np.cumsum(self._segment_lengths, dtype=np.int64))
        )

    @staticmethod
    def _load(path: str) -> np.ndarray:
        expanded = os.path.expanduser(path)
        if not expanded.endswith(".npy"):
            raise ValueError(f"FI-2010 转换文件应为 .npy，收到 {expanded}")
        data = np.load(expanded, mmap_mode="r")
        if data.ndim != 2 or data.shape[1] != TOTAL_COLUMNS:
            raise ValueError(f"FI-2010 数组应为二维且有 {TOTAL_COLUMNS} 列，实际为 {data.shape}")
        if data.dtype != np.float32:
            raise ValueError(f"FI-2010 数组应为 float32，实际为 {data.dtype}")
        return data

    @staticmethod
    def _load_segments(meta_path: str, *, rows: int) -> list[dict[str, int | str]]:
        with open(os.path.expanduser(meta_path), encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("rows") != rows:
            raise ValueError(f"元数据记录 {metadata.get('rows')} 行，数据文件实际有 {rows} 行")
        segments = metadata.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("元数据缺少非空的 segments 列表")

        validated: list[dict[str, int | str]] = []
        for segment in segments:
            try:
                cf = int(segment["cf"])
                role = str(segment["role"])
                start = int(segment["start"])
                end = int(segment["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"无效的 segment：{segment}") from error
            if not 1 <= cf <= 9 or role not in {"train", "test"}:
                raise ValueError(f"无效的 segment：{segment}")
            if not 0 <= start < end <= rows:
                raise ValueError(f"segment 行范围越界：{segment}")
            validated.append({"cf": cf, "role": role, "start": start, "end": end})
        return validated

    @staticmethod
    def _select_segments(
        segments: Sequence[dict[str, int | str]],
        *,
        protocol: str,
        split: str,
        test_cf: int | None,
    ) -> list[dict[str, int | str]]:
        role = "test" if split == "test" else "train"
        if protocol == "setup1":
            wanted_cfs = {test_cf}
        elif role == "train":
            wanted_cfs = {SETUP2_TRAIN_CF}
        else:
            wanted_cfs = set(SETUP2_TEST_CFS)

        return [
            segment
            for segment in segments
            if segment["role"] == role and segment["cf"] in wanted_cfs
        ]

    def _build_windows(
        self,
        data: np.ndarray,
        segments: Sequence[dict[str, int | str]],
        *,
        split: str,
        val_frac: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        windows: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            feature_block = data[start:end, :NUM_FEATURES]
            label_block = self._normalise_labels(data[start:end, self.label_col])
            train_windows, train_labels, val_windows, val_labels = self._windows_in_block(
                feature_block,
                label_block,
                hold_validation=split in {"train", "val"},
                val_frac=val_frac,
            )
            selected_windows = val_windows if split == "val" else train_windows
            selected_labels = val_labels if split == "val" else train_labels
            if selected_windows.shape[0]:
                windows.append(selected_windows)
                labels.append(selected_labels)
        return windows, labels

    @staticmethod
    def _normalise_labels(raw_labels: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(raw_labels)):
            raise ValueError("标签列包含 NaN 或无穷值")
        rounded = np.rint(raw_labels)
        if not np.array_equal(raw_labels, rounded):
            raise ValueError("标签列包含非整数值")
        labels = rounded.astype(np.int64)
        unexpected = set(np.unique(labels).tolist()) - {1, 2, 3}
        if unexpected:
            raise ValueError(f"FI-2010 标签应为 1、2、3，发现 {sorted(unexpected)}")
        return labels - 1

    def _windows_in_block(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        hold_validation: bool,
        val_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        window_size = self.window_size
        if features.shape[0] < window_size:
            empty_windows = np.empty(
                (0, window_size, NUM_FEATURES),
                dtype=np.float32,
            )
            empty_labels = np.empty(0, dtype=np.int64)
            return empty_windows, empty_labels, empty_windows, empty_labels

        view = np.lib.stride_tricks.sliding_window_view(
            features,
            window_shape=window_size,
            axis=0,
        )
        all_windows = np.transpose(view, (0, 2, 1))
        all_labels = labels[window_size - 1 :]
        if not hold_validation:
            return all_windows, all_labels, all_windows[:0], all_labels[:0]

        validation_count = max(1, int(len(all_windows) * val_frac))
        validation_start = len(all_windows) - validation_count
        train_end = validation_start - (window_size - 1)
        if train_end < 1:
            raise ValueError("训练段太短，无法构造互不共享原始行的训练窗口和验证窗口")
        return (
            all_windows[:train_end],
            all_labels[:train_end],
            all_windows[validation_start:],
            all_labels[validation_start:],
        )

    def __len__(self) -> int:
        return int(self._segment_offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment_index = int(np.searchsorted(self._segment_offsets, index, side="right") - 1)
        local_index = int(index - self._segment_offsets[segment_index])
        return segment_index, local_index

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        segment_index, local_index = self._locate(index)
        window = self._windows[segment_index][local_index]
        x = np.expand_dims(window.copy(), axis=0)
        y = int(self._window_labels[segment_index][local_index])
        return x, y

    def close(self) -> None:
        """释放窗口视图持有的内存映射文件。"""
        self._windows.clear()
        self._window_labels.clear()

    def __enter__(self) -> FI2010WindowDataset:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def get_dummy_batch(
    batch_size: int = 8,
    window_size: int = WINDOW_SIZE,
    num_features: int = NUM_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一批可直接交给模型的合成数据。"""
    x = torch.randn(batch_size, 1, window_size, num_features)
    y = torch.randint(0, NUM_CLASSES, (batch_size,))
    return x, y
