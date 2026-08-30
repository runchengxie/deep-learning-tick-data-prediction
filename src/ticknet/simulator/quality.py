"""流式检查单个 raw L2 Parquet 文件的数据质量。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

_DAY_RE = re.compile(r"(?:order|trades)_(\d{4})-(\d{2})-(\d{2})\.parquet$")
_ID_COLUMNS = ("OrderID", "DealID")
_NUMERIC_COLUMNS = ("time_ms", "Price", "Volume", "LastPrice")


def _scalar(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def _expected_day(path: Path) -> int | None:
    match = _DAY_RE.search(path.name)
    if match is None:
        return None
    return int("".join(match.groups()))


def _update_range(
    array: Any,
    name: str,
    minimum: dict[str, Any],
    maximum: dict[str, Any],
) -> None:
    if array.null_count == len(array):
        return
    low = _scalar(pc.min(array))
    high = _scalar(pc.max(array))
    if low is not None:
        minimum[name] = low if name not in minimum else min(minimum[name], low)
    if high is not None:
        maximum[name] = high if name not in maximum else max(maximum[name], high)


def _bounded_values(array: Any, target: set[Any], limit: int) -> None:
    if len(target) >= limit or array.null_count == len(array):
        return
    for value in pc.unique(array).drop_null().to_pylist():
        if len(target) >= limit:
            break
        target.add(value)


def _count_nonpositive(array: Any) -> int:
    if array.null_count == len(array):
        return 0
    return int(_scalar(pc.sum(pc.cast(pc.less_equal(array, 0), "int64"))) or 0)


def _timestamp_order(
    batch: Any,
    last_timestamp: dict[Any, Any],
) -> int:
    if "ticker" not in batch.schema.names or "time_ms" not in batch.schema.names:
        return 0
    tickers = batch.column("ticker").to_numpy(zero_copy_only=False)
    timestamps = batch.column("time_ms").to_numpy(zero_copy_only=False)
    valid = np.array(
        [
            ticker is not None and timestamp is not None
            for ticker, timestamp in zip(tickers, timestamps, strict=True)
        ],
        dtype=bool,
    )
    if not valid.any():
        return 0
    backwards = 0
    for ticker in np.unique(tickers[valid]):
        positions = np.flatnonzero(valid & (tickers == ticker))
        values = timestamps[positions]
        backwards += int(np.count_nonzero(values[1:] < values[:-1]))
        previous = last_timestamp.get(ticker)
        if previous is not None and values[0] < previous:
            backwards += 1
        last_timestamp[ticker] = values[-1]
    return backwards


def _track_ids(
    array: Any,
    tracked_ids: set[Any],
    max_tracked_ids: int,
) -> tuple[int, bool]:
    if array.null_count == len(array):
        return 0, False
    values = array.drop_null().to_numpy(zero_copy_only=False)
    unique, counts = np.unique(values, return_counts=True)
    unique_values = set(unique.tolist())
    duplicate_rows = int(np.sum(counts - 1))
    duplicate_rows += sum(int(value in tracked_ids) for value in unique)
    if len(tracked_ids) >= max_tracked_ids:
        return duplicate_rows, True
    available = max_tracked_ids - len(tracked_ids)
    if len(unique_values) <= available:
        tracked_ids.update(unique_values)
        return duplicate_rows, False
    tracked_ids.update(list(unique_values)[:available])
    return duplicate_rows, True


def profile_parquet(
    path: str | Path,
    *,
    batch_size: int = 262_144,
    max_tracked_ids: int = 1_000_000,
) -> dict[str, Any]:
    """按批读取一个 Parquet 文件，并返回可 JSON 序列化的质量报告。"""
    parquet_path = Path(path).expanduser().resolve()
    parquet = pq.ParquetFile(parquet_path)
    columns = parquet.schema.names
    id_column = next((name for name in _ID_COLUMNS if name in columns), None)
    tracked_ids: set[Any] = set()
    tracked_tickers: set[Any] = set()
    tracked_days: set[Any] = set()
    observed_types: set[Any] = set()
    last_timestamp: dict[Any, Any] = {}
    nulls = defaultdict(int)
    nonpositive = defaultdict(int)
    minimum: dict[str, Any] = {}
    maximum: dict[str, Any] = {}
    rows = 0
    duplicate_id_rows = 0
    timestamp_backwards = 0
    id_tracking_truncated = False
    expected_day = _expected_day(parquet_path)

    for batch in parquet.iter_batches(batch_size=batch_size):
        rows += batch.num_rows
        for name in columns:
            array = batch.column(name)
            nulls[name] += array.null_count
            if name in _NUMERIC_COLUMNS:
                _update_range(array, name, minimum, maximum)
        if "ticker" in columns:
            _bounded_values(batch.column("ticker"), tracked_tickers, 100_000)
        if "TradingDay" in columns:
            _bounded_values(batch.column("TradingDay"), tracked_days, 100_000)
        if "OrderType" in columns:
            _bounded_values(batch.column("OrderType"), observed_types, 100)
        for name in ("Price", "Volume"):
            if name in columns:
                nonpositive[name] += _count_nonpositive(batch.column(name))
        if expected_day is not None and "TradingDay" in columns:
            mismatch = pc.not_equal(batch.column("TradingDay"), expected_day)
            nulls["TradingDay_mismatch"] += int(
                _scalar(pc.sum(pc.cast(pc.fill_null(mismatch, False), "int64"))) or 0
            )
        timestamp_backwards += _timestamp_order(batch, last_timestamp)
        if id_column is not None:
            duplicate, truncated = _track_ids(
                batch.column(id_column), tracked_ids, max_tracked_ids
            )
            duplicate_id_rows += duplicate
            id_tracking_truncated = id_tracking_truncated or truncated

    return {
        "path": str(parquet_path),
        "rows": rows,
        "columns": columns,
        "id_column": id_column,
        "distinct_ids_observed": len(tracked_ids),
        "duplicate_id_rows": duplicate_id_rows,
        "id_tracking_truncated": id_tracking_truncated,
        "distinct_tickers_observed": len(tracked_tickers),
        "trading_days": sorted(tracked_days),
        "expected_trading_day": expected_day,
        "trading_day_mismatch_rows": nulls.pop("TradingDay_mismatch", 0),
        "order_type_values_observed": sorted(observed_types),
        "nulls": dict(nulls),
        "nonpositive": dict(nonpositive),
        "minimum": minimum,
        "maximum": maximum,
        "timestamp_backwards": timestamp_backwards,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--max-tracked-ids", type=int, default=1_000_000)
    arguments = parser.parse_args(argv)
    payload = {
        "status": "complete",
        "reports": [
            profile_parquet(
                path,
                batch_size=arguments.batch_size,
                max_tracked_ids=arguments.max_tracked_ids,
            )
            for path in arguments.file
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
