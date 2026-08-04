"""把按股票日组织的事件数组转换为次日预测 NPY 分片。"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from ticknet.nextday.io import PreparedSample, write_sharded_dataset
from ticknet.nextday.labels import DailyBar, NextDayTarget, build_next_day_targets


def _read_calendar(path: Path) -> list[date]:
    dates = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                dates.append(date.fromisoformat(value))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number} 不是 YYYY-MM-DD 日期") from error
    return dates


def _read_daily_bars(path: Path) -> list[DailyBar]:
    bars = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"symbol", "trading_date", "open", "close"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} 必须包含 {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                bars.append(
                    DailyBar(
                        symbol=row["symbol"],
                        trading_date=date.fromisoformat(row["trading_date"]),
                        open=float(row["open"]),
                        close=float(row["close"]),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} 日线字段无效") from error
    return bars


def _read_benchmark(path: Path | None) -> dict[date, float] | None:
    if path is None:
        return None
    returns: dict[date, float] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"trading_date", "return"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} 必须包含 {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                trading_date = date.fromisoformat(row["trading_date"])
                value = float(row["return"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} 基准收益字段无效") from error
            if trading_date in returns:
                raise ValueError(f"{path}:{line_number} 基准日期重复")
            returns[trading_date] = value
    return returns


def _parse_event_record(raw: object, *, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_number} 应为 JSON 对象")
    values = cast(dict[str, Any], raw)
    required = {
        "symbol",
        "trading_date",
        "features_path",
        "last_event_timestamp",
        "signal_timestamp",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"{path}:{line_number} 缺少字段 {sorted(missing)}")
    return values


def _prepared_samples(
    event_manifest: Path,
    targets: dict[tuple[str, date], NextDayTarget],
) -> Iterator[PreparedSample]:
    seen: set[tuple[str, date]] = set()
    with event_manifest.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{event_manifest}:{line_number} 不是有效 JSON") from error
            values = _parse_event_record(raw, path=event_manifest, line_number=line_number)
            try:
                symbol = str(values["symbol"])
                trading_date = date.fromisoformat(str(values["trading_date"]))
                last_event = datetime.fromisoformat(str(values["last_event_timestamp"]))
                signal = datetime.fromisoformat(str(values["signal_timestamp"]))
            except ValueError as error:
                raise ValueError(f"{event_manifest}:{line_number} 日期时间字段无效") from error
            key = (symbol, trading_date)
            if key in seen:
                raise ValueError(f"{event_manifest}:{line_number} 股票交易日重复：{key}")
            seen.add(key)
            target = targets.get(key)
            if target is None:
                continue
            features_path = Path(str(values["features_path"]))
            if not features_path.is_absolute():
                features_path = event_manifest.parent / features_path
            if features_path.suffix != ".npy" or not features_path.is_file():
                raise ValueError(f"{event_manifest}:{line_number} 特征文件不存在：{features_path}")
            events = np.load(features_path, mmap_mode="r")
            yield PreparedSample(
                target=target,
                events=events,
                last_event_timestamp=last_event,
                signal_timestamp=signal,
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成股票日级 tick/LOB 到次日方向的训练分片")
    parser.add_argument("--daily-bars", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--events-manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-method",
        choices=["cross_sectional", "fixed"],
        default="cross_sectional",
    )
    parser.add_argument("--neutral-threshold", type=float, default=0.002)
    parser.add_argument("--lower-quantile", type=float, default=0.2)
    parser.add_argument("--upper-quantile", type=float, default=0.8)
    parser.add_argument("--min-cross-section", type=int, default=20)
    parser.add_argument("--chunks-per-sample", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--samples-per-shard", type=int, default=512)
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    targets = build_next_day_targets(
        _read_daily_bars(args.daily_bars),
        _read_calendar(args.calendar),
        label_method=args.label_method,
        neutral_threshold=args.neutral_threshold,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        benchmark_returns=_read_benchmark(args.benchmark),
        min_cross_section=args.min_cross_section,
    )
    target_index = {(target.symbol, target.trading_date): target for target in targets}
    manifest = write_sharded_dataset(
        _prepared_samples(args.events_manifest, target_index),
        args.output_dir,
        chunks_per_sample=args.chunks_per_sample,
        chunk_size=args.chunk_size,
        samples_per_shard=args.samples_per_shard,
        storage_dtype=args.storage_dtype,
    )
    print(f"已生成 {len(targets):,} 个标签，数据清单：{manifest}")


if __name__ == "__main__":
    main()
