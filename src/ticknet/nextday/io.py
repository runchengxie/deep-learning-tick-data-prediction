"""把日内事件窗口写成可在 Colab 中顺序搬运的 NPY 分片。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ticknet.dataset import NUM_FEATURES
from ticknet.nextday.dataset import (
    FEATURE_NAMES,
    FORMAT_VERSION,
    file_sha256,
    manifest_fingerprint,
)
from ticknet.nextday.labels import NextDayTarget


@dataclass(frozen=True)
class PreparedSample:
    """已按信号时点过滤、尚未定长分块的股票日级事件。"""

    target: NextDayTarget
    events: np.ndarray
    last_event_timestamp: datetime
    signal_timestamp: datetime


def pack_events(
    events: np.ndarray,
    *,
    chunks_per_sample: int,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    """取最后若干事件并整理为 ``chunks × time × 40``。

    事件不足时在左侧重复当天第一条盘口状态。这样不会引入信号时点之后的数据，
    ``valid_events`` 会保留真实事件数量，便于后续质量筛选。
    """
    if chunks_per_sample < 1 or chunk_size < 1:
        raise ValueError("chunks_per_sample 和 chunk_size 应为正整数")
    array = np.asarray(events)
    if array.ndim != 2 or array.shape[1] != NUM_FEATURES:
        raise ValueError(f"events 应为 N × {NUM_FEATURES}，实际为 {array.shape}")
    if array.dtype != np.float32:
        raise ValueError(f"events 应为 float32，实际为 {array.dtype}")
    if array.shape[0] < 1:
        raise ValueError("events 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError("events 包含 NaN 或无穷值")

    total_events = chunks_per_sample * chunk_size
    valid_events = min(array.shape[0], total_events)
    selected = array[-total_events:]
    if selected.shape[0] < total_events:
        padding = np.repeat(selected[:1], total_events - selected.shape[0], axis=0)
        selected = np.concatenate((padding, selected), axis=0)
    return selected.reshape(chunks_per_sample, chunk_size, NUM_FEATURES), valid_events


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_sharded_dataset(
    samples: Iterable[PreparedSample],
    output_dir: str | Path,
    *,
    chunks_per_sample: int = 10,
    chunk_size: int = 100,
    samples_per_shard: int = 512,
    storage_dtype: str = "float32",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """写入 NPY 分片和 ``manifest.json``。

    ``float16`` 只影响磁盘表示；数据集在送入模型前会转换回 ``float32``。它适合
    Colab 工作集，但上游必须先把绝对价格和大额数量变换到可安全表示的尺度。
    """
    if samples_per_shard < 1:
        raise ValueError("samples_per_shard 应为正整数")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype 应为 float16 或 float32")
    target_dtype = np.dtype(storage_dtype)
    root = Path(output_dir).expanduser().resolve()
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dtype": storage_dtype,
        "layout": "samples_chunks_time_features",
        "chunks_per_sample": chunks_per_sample,
        "chunk_size": chunk_size,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "shards": [],
        "samples": [],
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    buffer: list[np.ndarray] = []
    pending_records: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        shard_index = len(manifest["shards"])
        relative_path = Path("shards") / f"part-{shard_index:05d}.npy"
        array = np.stack(buffer).astype(target_dtype, copy=False)
        shard_path = root / relative_path
        _atomic_save(shard_path, array)
        manifest["shards"].append(
            {
                "path": relative_path.as_posix(),
                "samples": int(array.shape[0]),
                "bytes": shard_path.stat().st_size,
                "sha256": file_sha256(shard_path),
            }
        )
        for row, record in enumerate(pending_records):
            record["shard"] = shard_index
            record["row"] = row
            manifest["samples"].append(record)
        buffer.clear()
        pending_records.clear()

    seen: set[tuple[str, object]] = set()
    for sample in samples:
        target = sample.target
        key = (target.symbol, target.trading_date)
        if key in seen:
            raise ValueError(f"股票交易日重复：{target.symbol} {target.trading_date}")
        seen.add(key)
        if sample.last_event_timestamp.date() != target.trading_date:
            raise ValueError(f"{key} 的最后事件时间不属于输入交易日")
        if sample.signal_timestamp.date() != target.trading_date:
            raise ValueError(f"{key} 的信号时间不属于输入交易日")
        try:
            after_signal = sample.last_event_timestamp > sample.signal_timestamp
        except TypeError as error:
            raise ValueError(f"{key} 的两个时间戳时区格式不一致") from error
        if after_signal:
            raise ValueError(f"{key} 包含信号时点之后的数据")
        packed, valid_events = pack_events(
            sample.events,
            chunks_per_sample=chunks_per_sample,
            chunk_size=chunk_size,
        )
        buffer.append(packed)
        pending_records.append(
            {
                "symbol": target.symbol,
                "trading_date": target.trading_date.isoformat(),
                "label_date": target.label_date.isoformat(),
                "label": target.label,
                "raw_return": target.raw_return,
                "target_return": target.target_return,
                "last_event_timestamp": sample.last_event_timestamp.isoformat(),
                "signal_timestamp": sample.signal_timestamp.isoformat(),
                "valid_events": valid_events,
            }
        )
        if len(buffer) >= samples_per_shard:
            flush()
    flush()

    if not manifest["samples"]:
        raise ValueError("没有可写入的样本")
    manifest["dataset_fingerprint"] = manifest_fingerprint(manifest)
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path
