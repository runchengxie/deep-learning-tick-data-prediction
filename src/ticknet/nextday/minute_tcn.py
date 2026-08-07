"""分钟级序列的时序模型（TCN）与分片数据集。

与 ``minute_baseline.py`` 消费同一套特征源与标签，但输入是未聚合的
``T x features`` 分钟序列，用于和聚合特征（HGB）做受控对比。
分片由 ``scripts/prepare_minute_shards.py`` 生成，布局为
``samples x time x features``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn as nn

from ticknet.dataset import NUM_CLASSES
from ticknet.nextday.dataset import file_sha256, manifest_fingerprint
from ticknet.nextday.splits import parse_date


class MinuteOutput(NamedTuple):
    """TCN 同时输出方向分类和连续横截面分数。"""

    logits: torch.Tensor
    score: torch.Tensor


def _causal_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """为因果卷积在序列左侧补零。"""
    pad = (kernel_size - 1) * dilation
    return nn.functional.pad(x, (pad, 0))


class CausalConv1d(nn.Module):
    """带膨胀的因果一维卷积，保证输出第 t 位只看输入 0..t。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_causal_pad(x, self.kernel_size, self.dilation))


class TCNBlock(nn.Module):
    """两个因果卷积 + 残差连接，与 Bai et al. (2018) 结构一致。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = torch.relu(self.conv1(x))
        h = self.dropout(h)
        h = torch.relu(self.conv2(h))
        h = self.dropout(h)
        return residual + h


class MinuteTCN(nn.Module):
    """分钟序列 TCN：输入 ``B x time x features``，输出分类和分数。"""

    def __init__(
        self,
        *,
        num_features: int,
        num_classes: int = NUM_CLASSES,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features 应为正整数")
        if num_layers < 1:
            raise ValueError("num_layers 应为正整数")
        if kernel_size < 1:
            raise ValueError("kernel_size 应为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 应在 [0, 1) 内")
        self.num_features = num_features
        self.input_projection = nn.Conv1d(num_features, hidden_channels, 1)
        blocks = [
            TCNBlock(
                hidden_channels,
                kernel_size,
                dilation=2**layer_index,
                dropout=dropout,
            )
            for layer_index in range(num_layers)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(hidden_channels, num_classes)
        self.score_head = nn.Linear(hidden_channels, 1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``B x hidden_channels`` 的序列级表示（取最后时间步）。"""
        if x.ndim != 3 or x.shape[2] != self.num_features:
            raise ValueError(
                f"MinuteTCN 输入应为 (B, T, {self.num_features})，实际为 {tuple(x.shape)}"
            )
        h = self.input_projection(x.transpose(1, 2))
        h = self.blocks(h)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> MinuteOutput:
        representation = self.dropout(self.encode_sequence(x))
        return MinuteOutput(
            logits=self.classification_head(representation),
            score=self.score_head(representation).squeeze(-1),
        )


def build_minute_tcn(
    *,
    num_features: int,
    num_classes: int = NUM_CLASSES,
    hidden_channels: int = 64,
    num_layers: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
) -> MinuteTCN:
    return MinuteTCN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    )


@dataclass(frozen=True)
class MinuteRecord:
    """一个股票日分钟序列样本在分片中的位置和监督信息。"""

    symbol: str
    trading_date: date
    label_date: date
    shard: int
    row: int
    label: int
    target_return: float
    minutes: int


def _validate_minute_record(raw: object, *, sample_index: int) -> MinuteRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"samples[{sample_index}] 应为对象")
    values = cast(dict[str, Any], raw)
    try:
        symbol = str(values["symbol"])
        trading_date = parse_date(str(values["trading_date"]))
        label_date = parse_date(str(values["label_date"]))
        shard = int(values["shard"])
        row = int(values["row"])
        label = int(values["label"])
        target_return = float(values["target_return"])
        minutes = int(values["minutes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"samples[{sample_index}] 缺少字段或字段类型无效") from error

    if not symbol:
        raise ValueError(f"samples[{sample_index}] 的 symbol 不能为空")
    if label_date <= trading_date:
        raise ValueError(f"samples[{sample_index}] 的标签日必须晚于输入日")
    if shard < 0 or row < 0:
        raise ValueError(f"samples[{sample_index}] 的 shard 和 row 不能为负数")
    if label not in range(NUM_CLASSES):
        raise ValueError(f"samples[{sample_index}] 的 label 应为 0、1 或 2")
    if not np.isfinite(target_return):
        raise ValueError(f"samples[{sample_index}] 的 target_return 不是有限值")
    if minutes < 1:
        raise ValueError(f"samples[{sample_index}] 的 minutes 应为正整数")
    return MinuteRecord(
        symbol=symbol,
        trading_date=trading_date,
        label_date=label_date,
        shard=shard,
        row=row,
        label=label,
        target_return=target_return,
        minutes=minutes,
    )


class MinuteShardDataset(torch.utils.data.Dataset):
    """读取 ``samples x time x features`` 的 float32 NPY 分片。

    每个索引严格对应一只股票和一个输入交易日。数据集按输入日和标签日共同落入的
    切分区段过滤样本，保证训练、验证、测试区间不重叠。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        date_split: Any,
        split: str,
        verify_checksums: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError("数据清单根节点应为对象")
        if manifest.get("format_version") != 1:
            raise ValueError("分钟分片 format_version 应为 1")
        computed_fingerprint = manifest_fingerprint(manifest)
        stored_fingerprint = manifest.get("dataset_fingerprint")
        if stored_fingerprint is not None:
            if not isinstance(stored_fingerprint, str) or len(stored_fingerprint) != 64:
                raise ValueError("数据清单 dataset_fingerprint 应为 SHA-256")
            if stored_fingerprint != computed_fingerprint:
                raise ValueError("数据清单 dataset_fingerprint 与内容不一致")
        self.dataset_fingerprint = stored_fingerprint or computed_fingerprint

        dtype_name = manifest.get("dtype")
        if dtype_name not in {"float16", "float32"}:
            raise ValueError("数据清单 dtype 应为 float16 或 float32")
        self.storage_dtype = np.dtype(dtype_name)
        self.window_minutes = self._positive_int(manifest, "window_minutes")
        self.num_features = self._positive_int(manifest, "feature_count")

        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("数据清单缺少非空的 shards 列表")
        self._shard_paths: list[Path] = []
        shard_rows: list[int] = []
        shard_bytes: list[int | None] = []
        shard_checksums: list[str | None] = []
        for index, raw_shard in enumerate(raw_shards):
            if not isinstance(raw_shard, dict):
                raise ValueError(f"shards[{index}] 应为对象")
            shard_values = cast(dict[str, Any], raw_shard)
            try:
                path = Path(str(shard_values["path"]))
                rows = int(shard_values["samples"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"shards[{index}] 无效") from error
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            if rows < 1:
                raise ValueError(f"shards[{index}] 的 samples 应为正整数")
            raw_bytes = shard_values.get("bytes")
            expected_bytes = None if raw_bytes is None else int(raw_bytes)
            raw_checksum = shard_values.get("sha256")
            if raw_checksum is not None and (
                not isinstance(raw_checksum, str) or len(raw_checksum) != 64
            ):
                raise ValueError(f"shards[{index}] 的 sha256 格式无效")
            self._shard_paths.append(path.resolve())
            shard_rows.append(rows)
            shard_bytes.append(expected_bytes)
            shard_checksums.append(raw_checksum)

        raw_samples = manifest.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("数据清单缺少非空的 samples 列表")
        all_records = [
            _validate_minute_record(raw, sample_index=index)
            for index, raw in enumerate(raw_samples)
        ]
        self._validate_positions(all_records, shard_rows)
        self._validate_shards(
            shard_rows,
            shard_bytes,
            shard_checksums,
            verify_checksums=verify_checksums,
        )

        self.records = [
            record
            for record in all_records
            if date_split.assign(record.trading_date, record.label_date) == split
        ]
        if not self.records:
            raise ValueError(f"{split} 日期区间没有可用样本")
        self._arrays: dict[int, np.ndarray] = {}

    @staticmethod
    def _positive_int(manifest: dict[str, Any], field: str) -> int:
        try:
            value = int(manifest[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"数据清单缺少有效的 {field}") from error
        if value < 1:
            raise ValueError(f"数据清单 {field} 应为正整数")
        return value

    @staticmethod
    def _validate_positions(records: list[MinuteRecord], shard_rows: list[int]) -> None:
        positions: set[tuple[int, int]] = set()
        keys: set[tuple[str, date]] = set()
        for record in records:
            if record.shard >= len(shard_rows) or record.row >= shard_rows[record.shard]:
                raise ValueError(f"样本位置越界：shard={record.shard}, row={record.row}")
            position = (record.shard, record.row)
            if position in positions:
                raise ValueError(f"样本位置重复：shard={record.shard}, row={record.row}")
            positions.add(position)
            key = (record.symbol, record.trading_date)
            if key in keys:
                raise ValueError(f"股票交易日重复：{record.symbol} {record.trading_date}")
            keys.add(key)

    def _validate_shards(
        self,
        expected_rows: list[int],
        expected_bytes: list[int | None],
        expected_checksums: list[str | None],
        *,
        verify_checksums: bool,
    ) -> None:
        expected_tail = (self.window_minutes, self.num_features)
        for index, (path, rows, byte_count, checksum) in enumerate(
            zip(
                self._shard_paths,
                expected_rows,
                expected_bytes,
                expected_checksums,
                strict=True,
            )
        ):
            if path.suffix != ".npy":
                raise ValueError(f"分片应为 .npy 文件，收到 {path}")
            if not path.is_file():
                raise ValueError(f"分片不存在：{path}")
            if byte_count is not None and path.stat().st_size != byte_count:
                raise ValueError(f"分片 {path} 的文件大小与数据清单不一致")
            if verify_checksums:
                if checksum is None:
                    raise ValueError(f"分片 {path} 缺少 sha256，不能执行完整校验")
                if file_sha256(path) != checksum:
                    raise ValueError(f"分片 {path} 的 sha256 与数据清单不一致")
            array = np.load(path, mmap_mode="r")
            if array.dtype != self.storage_dtype:
                raise ValueError(f"分片 {path} 应为 {self.storage_dtype}，实际为 {array.dtype}")
            if array.shape != (rows, *expected_tail):
                raise ValueError(
                    f"分片 {index} 形状应为 {(rows, *expected_tail)}，实际为 {array.shape}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, np.float32]:
        record = self.records[index]
        if record.shard not in self._arrays:
            self._arrays[record.shard] = np.load(
                self._shard_paths[record.shard],
                mmap_mode="r",
            )
        features = self._arrays[record.shard][record.row].astype(np.float32, copy=True)
        return features, record.label, np.float32(record.target_return)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    @property
    def target_returns(self) -> np.ndarray:
        return np.asarray([record.target_return for record in self.records], dtype=np.float64)

    @property
    def label_dates(self) -> list[date]:
        return [record.label_date for record in self.records]

    @property
    def symbols(self) -> list[str]:
        return [record.symbol for record in self.records]
