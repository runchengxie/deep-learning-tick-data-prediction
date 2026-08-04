"""从内存映射分片读取股票日级订单簿样本。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from torch.utils.data import Dataset

from deeplob.dataset import NUM_CLASSES, NUM_FEATURES
from deeplob.nextday.splits import WalkForwardSplit, parse_date

FORMAT_VERSION = 1
FEATURE_NAMES = tuple(
    name
    for level in range(1, 11)
    for name in (
        f"ask_price_{level}",
        f"ask_size_{level}",
        f"bid_price_{level}",
        f"bid_size_{level}",
    )
)


def file_sha256(path: str | Path) -> str:
    """顺序计算文件 SHA-256，不把大分片读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """计算不包含自身 fingerprint 字段的稳定数据清单指纹。"""
    content = {key: value for key, value in manifest.items() if key != "dataset_fingerprint"}
    payload = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SampleRecord:
    """一个股票交易日样本在分片中的位置和监督信息。"""

    symbol: str
    trading_date: date
    label_date: date
    shard: int
    row: int
    label: int
    target_return: float
    last_event_timestamp: datetime
    signal_timestamp: datetime
    valid_events: int


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是 ISO 时间字符串")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} 不是有效的 ISO 时间：{value!r}") from error


def _validate_record(raw: object, *, sample_index: int, total_events: int) -> SampleRecord:
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
        last_event = _parse_timestamp(
            values["last_event_timestamp"],
            field="last_event_timestamp",
        )
        signal = _parse_timestamp(values["signal_timestamp"], field="signal_timestamp")
        valid_events = int(values["valid_events"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error):
            raise ValueError(f"samples[{sample_index}] 无效：{error}") from error
        raise ValueError(f"samples[{sample_index}] 缺少字段或字段类型无效") from error

    if not symbol:
        raise ValueError(f"samples[{sample_index}] 的 symbol 不能为空")
    if label_date <= trading_date:
        raise ValueError(f"samples[{sample_index}] 的标签日必须晚于输入日")
    if shard < 0 or row < 0:
        raise ValueError(f"samples[{sample_index}] 的 shard 和 row 不能为负数")
    if label not in range(NUM_CLASSES):
        raise ValueError(f"samples[{sample_index}] 的 label 应为 0、1 或 2")
    if not math.isfinite(target_return):
        raise ValueError(f"samples[{sample_index}] 的 target_return 不是有限值")
    if last_event.date() != trading_date or signal.date() != trading_date:
        raise ValueError(f"samples[{sample_index}] 的信号时间必须属于输入交易日")
    try:
        after_signal = last_event > signal
    except TypeError as error:
        raise ValueError(f"samples[{sample_index}] 的两个时间戳时区格式不一致") from error
    if after_signal:
        raise ValueError(f"samples[{sample_index}] 包含信号时点之后的事件")
    if not 1 <= valid_events <= total_events:
        raise ValueError(f"samples[{sample_index}] 的 valid_events 超出窗口范围")
    return SampleRecord(
        symbol=symbol,
        trading_date=trading_date,
        label_date=label_date,
        shard=shard,
        row=row,
        label=label,
        target_return=target_return,
        last_event_timestamp=last_event,
        signal_timestamp=signal,
        valid_events=valid_events,
    )


class NextDayShardDataset(Dataset):
    """读取 ``N × chunks × time × 40`` 的 float32 NPY 分片。

    每个索引严格对应一只股票和一个输入交易日。数据集只选择输入日与标签日同时落入
    指定区间的记录，切分边界处的跨期标签会自动清除。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        date_split: WalkForwardSplit,
        split: str,
        verify_checksums: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError("数据清单根节点应为对象")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"数据清单 format_version 应为 {FORMAT_VERSION}")
        computed_fingerprint = manifest_fingerprint(manifest)
        stored_fingerprint = manifest.get("dataset_fingerprint")
        if stored_fingerprint is not None:
            if not isinstance(stored_fingerprint, str) or len(stored_fingerprint) != 64:
                raise ValueError("数据清单 dataset_fingerprint 应为 SHA-256")
            if stored_fingerprint != computed_fingerprint:
                raise ValueError("数据清单 dataset_fingerprint 与内容不一致")
        self.dataset_fingerprint = stored_fingerprint or computed_fingerprint

        self.chunks_per_sample = self._positive_int(manifest, "chunks_per_sample")
        self.chunk_size = self._positive_int(manifest, "chunk_size")
        self.num_features = self._positive_int(manifest, "num_features")
        if self.num_features != NUM_FEATURES:
            raise ValueError(f"当前 DeepLOB 编码器要求 {NUM_FEATURES} 个特征")
        dtype_name = manifest.get("dtype")
        if dtype_name not in {"float16", "float32"}:
            raise ValueError("数据清单 dtype 应为 float16 或 float32")
        self.storage_dtype = np.dtype(dtype_name)
        if tuple(manifest.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("数据清单 feature_names 与 DeepLOB 十档盘口排列不一致")

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
            if expected_bytes is not None and expected_bytes < 1:
                raise ValueError(f"shards[{index}] 的 bytes 应为正整数")
            raw_checksum = shard_values.get("sha256")
            if raw_checksum is not None and (
                not isinstance(raw_checksum, str) or len(raw_checksum) != 64
            ):
                raise ValueError(f"shards[{index}] 的 sha256 格式无效")
            self._shard_paths.append(path.resolve())
            shard_rows.append(rows)
            shard_bytes.append(expected_bytes)
            shard_checksums.append(raw_checksum)

        total_events = self.chunks_per_sample * self.chunk_size
        raw_samples = manifest.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("数据清单缺少非空的 samples 列表")
        all_records = [
            _validate_record(raw, sample_index=index, total_events=total_events)
            for index, raw in enumerate(raw_samples)
        ]
        self._validate_positions(all_records, shard_rows)
        self._validate_shards(
            shard_rows,
            shard_bytes,
            shard_checksums,
            verify_checksums=verify_checksums,
        )

        wanted = date_split.range_for(split)
        self.records = [
            record
            for record in all_records
            if date_split.assign(record.trading_date, record.label_date) == split
        ]
        self.purged_samples = sum(
            1
            for record in all_records
            if wanted.contains(record.trading_date) and not wanted.contains(record.label_date)
        )
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
    def _validate_positions(records: list[SampleRecord], shard_rows: list[int]) -> None:
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
        expected_tail = (self.chunks_per_sample, self.chunk_size, self.num_features)
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

    def __getitem__(
        self,
        index: int,
    ) -> tuple[np.ndarray, int, np.float32]:
        record = self.records[index]
        if record.shard not in self._arrays:
            self._arrays[record.shard] = np.load(
                self._shard_paths[record.shard],
                mmap_mode="r",
            )
        features = self._arrays[record.shard][record.row]
        model_features = np.expand_dims(features.astype(np.float32, copy=True), axis=1)
        return model_features, record.label, np.float32(record.target_return)

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
