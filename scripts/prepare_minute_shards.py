"""把分钟级序列切分成可在 Colab 中顺序搬运的 NPY 分片。

与 ``run_minute_baseline.py`` 使用同一套股票池、标签、日期切分和分钟读取逻辑，
但**不聚合**：每个样本保留尾部长为 ``window_minutes`` 的分钟特征矩阵（默认
``T x 33``，tushare 为 ``T x 6``），供时序模型（TCN/GRU）直接消费。

产物是 ``samples x time x features`` 布局的 float16 分片 + ``manifest.json``，
含 ``dataset_fingerprint``，与 notebook 端 Colab 校验约定保持一致。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ticknet.nextday.dataset import file_sha256, manifest_fingerprint
from ticknet.nextday.minute_baseline import (
    MINUTE_FEATURE_SOURCES,
    MinuteBaselineConfig,
    MinuteExtractionReport,
    build_targets,
    load_minute_baseline_config,
    read_l2_minute_rows,
    read_tushare_minute_rows,
)
from ticknet.nextday.splits import parse_date

_KNOWN_SOURCES = MINUTE_FEATURE_SOURCES


def load_config(path: str | Path) -> MinuteBaselineConfig:
    try:
        return load_minute_baseline_config(path)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error


def _read_rows(
    config: MinuteBaselineConfig,
    targets: list[Any],
    report: MinuteExtractionReport,
) -> dict[tuple[int, str], list[tuple[int, np.ndarray]]]:
    if config.feature_source == "l2_cache":
        if not config.l2_root:
            raise SystemExit("feature_source 为 l2_cache 时必须提供 l2_root")
        return read_l2_minute_rows(
            config.l2_root,
            targets,
            keep_minutes=config.window_minutes,
            report=report,
        )
    if not config.tushare_root:
        raise SystemExit("feature_source 为 tushare 时必须提供 tushare_root")
    return read_tushare_minute_rows(
        config.tushare_root,
        targets,
        keep_minutes=config.window_minutes,
        report=report,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="把分钟级序列切分为 NPY 分片")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=sorted(_KNOWN_SOURCES), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument("--samples-per-shard", type=int, default=2048)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.source is not None:
        config = replace(config, feature_source=args.source)
    if config.formal:
        raise SystemExit("分钟序列分片尚未支持正式 return_end_date 清洗，请使用诊断标签配置")
    if config.window_minutes < 1 or config.min_window_minutes < 1:
        raise SystemExit("window_minutes 与 min_window_minutes 应为正整数")
    if config.min_window_minutes > config.window_minutes:
        raise SystemExit("min_window_minutes 不能大于 window_minutes")
    if args.samples_per_shard < 1:
        raise SystemExit("samples_per_shard 应为正整数")
    target_dtype = np.dtype(args.storage_dtype)

    report = MinuteExtractionReport()
    targets = build_targets(config)
    rows = _read_rows(config, targets, report)

    root = args.output.expanduser().resolve()
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "dtype": args.storage_dtype,
        "layout": "samples_time_features",
        "window_minutes": config.window_minutes,
        "shards": [],
        "samples": [],
    }

    train_start = parse_date(config.train_start)
    train_end = parse_date(config.train_end)
    collected: list[tuple[np.ndarray, dict[str, Any]]] = []
    feature_count = 0

    for target in targets:
        key = (_date_int(target.trading_date), target.symbol)
        day_rows = rows.get(key)
        if not day_rows:
            report.missing_rows += 1
            continue
        window_rows = day_rows[-config.window_minutes :]
        if len(window_rows) < config.min_window_minutes:
            report.insufficient_window += 1
            continue
        matrix = np.stack([row for _minute, row in window_rows])
        if matrix.ndim != 2 or matrix.shape[0] != len(window_rows):
            raise RuntimeError(f"{key} 窗口矩阵形状异常")
        matrix, missing_minutes = _pad_to_window(
            matrix,
            window_minutes=config.window_minutes,
        )
        feature_count = int(matrix.shape[1])
        record = {
            "symbol": target.symbol,
            "trading_date": target.trading_date.isoformat(),
            "label_date": target.label_date.isoformat(),
            "label": target.label,
            "raw_return": target.raw_return,
            "target_return": target.target_return,
            "minutes": len(window_rows),
        }
        if missing_minutes:
            record["padded_minutes"] = missing_minutes
        collected.append((matrix, record))
        report.written_samples += 1

    if not collected:
        raise ValueError("没有可写入的样本")

    median_fill = _compute_median_fill(
        collected,
        feature_count=feature_count,
        train_start=train_start,
        train_end=train_end,
    )
    nan_filled = sum(1 for matrix, _record in collected if not np.all(np.isfinite(matrix)))

    buffer: list[np.ndarray] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        shard_index = len(manifest["shards"])
        relative = Path("shards") / f"part-{shard_index:05d}.npy"
        array = np.stack(buffer)
        if not np.all(np.isfinite(array)):
            raise ValueError("填充后仍存在 NaN 或无穷值")
        if target_dtype == np.float16:
            _assert_finite_in_float16(array)
        array = array.astype(target_dtype, copy=False)
        shard_path = root / relative
        _atomic_save(shard_path, array)
        manifest["shards"].append(
            {
                "path": relative.as_posix(),
                "samples": int(array.shape[0]),
                "bytes": shard_path.stat().st_size,
                "sha256": file_sha256(shard_path),
            }
        )
        for row, record in enumerate(pending):
            record["shard"] = shard_index
            record["row"] = row
            manifest["samples"].append(record)
        buffer.clear()
        pending.clear()

    for matrix, record in collected:
        filled = _fill_nan_with_medians(matrix, median_fill)
        buffer.append(filled)
        pending.append(record)
        if len(buffer) >= args.samples_per_shard:
            flush()
    flush()

    manifest["feature_count"] = feature_count
    manifest["nan_fill"] = {
        "method": "per_feature_median",
        "scope": "train",
        "features_with_nan": int(np.count_nonzero(median_fill)),
        "samples_with_nan": nan_filled,
    }
    manifest["dataset_fingerprint"] = manifest_fingerprint(manifest)
    _atomic_json(root / "manifest.json", manifest)

    audit: dict[str, Any] = {
        "config": {
            "feature_source": config.feature_source,
            "window_minutes": config.window_minutes,
            "min_window_minutes": config.min_window_minutes,
            "top_n": config.top_n,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "nan_fill": "per_feature_median_on_train",
        },
        "extraction": report.__dict__,
        "manifest": str(root / "manifest.json"),
    }
    _atomic_json(root / "data-audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    print(f"已写入 {len(manifest['samples']):,} 个样本，清单：{root / 'manifest.json'}")


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary, path)


def _assert_finite_in_float16(array: np.ndarray) -> None:
    """float16 会溢出到 inf；若出现则明确报错而不是静默写坏数据。"""
    finfo = np.finfo(np.float16)
    if array.size and not np.all(np.isfinite(array.astype(np.float16, copy=False))):
        bad = array[~np.isfinite(array.astype(np.float16, copy=False))]
        raise ValueError(
            f"有 {bad.size} 个值超出 float16 范围 "
            f"({finfo.min}~{finfo.max})，请改用 float32 或先归一化"
        )


def _atomic_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def _date_int(trading_date: Any) -> int:
    return int(trading_date.strftime("%Y%m%d"))


def _pad_to_window(
    matrix: np.ndarray,
    *,
    window_minutes: int,
) -> tuple[np.ndarray, int]:
    """把分钟窗口统一到固定 ``window_minutes`` 高度。

    L2 分钟序列存在缺失，不足固定时间步的样本在尾部用 NaN 补零，
    交由逐列中位数填充统一处理，保证分片恒为 ``T x features``。
    """
    rows, columns = matrix.shape
    if rows == window_minutes:
        return matrix, 0
    if rows > window_minutes:
        raise RuntimeError(f"窗口超出 {window_minutes} 分钟：{matrix.shape}")
    padded = np.full((window_minutes, columns), np.nan, dtype=np.float64)
    padded[:rows, :] = matrix
    return padded, window_minutes - rows


def _compute_median_fill(
    collected: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    feature_count: int,
    train_start: Any,
    train_end: Any,
) -> np.ndarray:
    """在训练区间样本上统计每列非 NaN 值的中位数，返回逐列填充值。

    只使用训练区间统计填充值，避免验证与测试区间信息泄漏进填充。
    """
    train_arrays: list[np.ndarray] = []
    for matrix, record in collected:
        trading_date = parse_date(str(record["trading_date"]))
        if train_start <= trading_date <= train_end:
            train_arrays.append(matrix)
    if not train_arrays:
        raise ValueError("训练区间没有样本，无法统计填充值")
    stacked = np.concatenate(train_arrays, axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != feature_count:
        raise ValueError("训练矩阵形状异常")
    fill = np.zeros(feature_count, dtype=np.float64)
    for column in range(feature_count):
        values = stacked[:, column]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"特征列 {column} 在训练区间全部为 NaN，无法填充")
        fill[column] = float(np.median(finite))
    return fill


def _fill_nan_with_medians(matrix: np.ndarray, median_fill: np.ndarray) -> np.ndarray:
    """用逐列中位数填充 NaN，返回 float64 副本。"""
    filled = matrix.astype(np.float64, copy=True)
    for column, median in enumerate(median_fill):
        mask = np.isnan(filled[:, column])
        if np.any(mask):
            filled[mask, column] = median
    if not np.all(np.isfinite(filled)):
        raise ValueError("填充后仍存在 NaN 或无穷值")
    return filled


if __name__ == "__main__":
    main()
