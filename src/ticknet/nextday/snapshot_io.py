"""snapshot 原始盘口 IO：日线宽表读取、逐月尾部提取与样本迭代。

从 ``raw_snapshot.py`` 拆出。负责从月度/逐日 Parquet 读取十档盘口、按目标股票日
裁剪并保留最后 N 个有效 tick。依赖 ``snapshot_config`` 与 ``snapshot_features``，
不直接构造标签（标签在 ``snapshot_targets``）。
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ticknet.nextday.io import PreparedSample
from ticknet.nextday.snapshot_config import (
    RAW_FEATURE_COLUMNS,
    DailyPanel,
    ExtractionReport,
    SnapshotPreparationConfig,
    _valid_stock_symbols,
    _yyyymmdd,
)
from ticknet.nextday.snapshot_features import normalize_lob_events, valid_lob_event_rows
from ticknet.nextday.splits import parse_date


def read_wide_daily_panel(
    path: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> DailyPanel:
    """读取 ``value`` 日期列加股票宽列的日线 Parquet。"""
    source = Path(path).expanduser().resolve()
    parquet = pq.ParquetFile(source)
    available = set(parquet.schema_arrow.names)
    if "value" not in available:
        raise ValueError(f"{source} 缺少 value 日期列")
    selected = _valid_stock_symbols(available - {"value"}) if symbols is None else tuple(symbols)
    missing = set(selected) - available
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"{source} 缺少股票列：{preview}")
    selected = tuple(sorted(set(selected)))
    filters: list[tuple[str, str, int]] = []
    if start_date is not None:
        filters.append(("value", ">=", int(start_date.strftime("%Y%m%d"))))
    if end_date is not None:
        filters.append(("value", "<=", int(end_date.strftime("%Y%m%d"))))
    table = pq.read_table(
        source,
        columns=["value", *selected],
        filters=filters or None,
    )
    dates = tuple(_yyyymmdd(value) for value in table["value"].to_pylist())
    if selected:
        values = np.column_stack(
            [
                table[symbol].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
                for symbol in selected
            ]
        )
    else:
        values = np.empty((len(dates), 0), dtype=np.float64)
    return DailyPanel(dates=dates, symbols=selected, values=values)


def load_market_panels(basic_root: str | Path) -> tuple[DailyPanel, DailyPanel, DailyPanel]:
    """读取并严格对齐 open、close、volume 三个宽表。"""
    root = Path(basic_root).expanduser().resolve()
    paths = {
        "open": root / "open_data.parquet",
        "close": root / "close_data.parquet",
        "volume": root / "volume_data.parquet",
    }
    schemas = {name: set(pq.ParquetFile(path).schema_arrow.names) for name, path in paths.items()}
    common = set.intersection(*(schema - {"value"} for schema in schemas.values()))
    common_symbols = _valid_stock_symbols(common)
    open_panel = read_wide_daily_panel(paths["open"], symbols=common_symbols)
    close_panel = read_wide_daily_panel(paths["close"], symbols=common_symbols)
    volume_panel = read_wide_daily_panel(paths["volume"], symbols=common_symbols)
    for name, panel in (("close", close_panel), ("volume", volume_panel)):
        if panel.dates != open_panel.dates or panel.symbols != open_panel.symbols:
            raise ValueError(f"{name} 日线轴与 open 不一致")
    return open_panel, close_panel, volume_panel


def _month_range(start: date, end: date) -> Iterator[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def _row_group_symbol_range(parquet: pq.ParquetFile, row_group: int) -> tuple[str, str] | None:
    column_index = parquet.schema_arrow.get_field_index("ticker")
    statistics = parquet.metadata.row_group(row_group).column(column_index).statistics
    if statistics is None or not statistics.has_min_max:
        return None
    return str(statistics.min).zfill(6), str(statistics.max).zfill(6)


def _range_intersects_symbols(symbols: Sequence[str], lower: str, upper: str) -> bool:
    index = bisect.bisect_left(symbols, lower)
    return index < len(symbols) and symbols[index] <= upper


def _update_tail(
    buffers: dict[tuple[date, str], tuple[np.ndarray, np.ndarray]],
    key: tuple[date, str],
    times: np.ndarray,
    events: np.ndarray,
    *,
    total_events: int,
) -> None:
    if key in buffers:
        old_times, old_events = buffers[key]
        times = np.concatenate((old_times, times))
        events = np.concatenate((old_events, events), axis=0)
    order = np.argsort(times, kind="stable")
    if order.size > total_events:
        order = order[-total_events:]
    buffers[key] = times[order], events[order]


def _read_month_tail(
    path: Path,
    targets: Mapping[tuple[date, str], Any],
    config: SnapshotPreparationConfig,
    report: ExtractionReport,
) -> dict[tuple[date, str], tuple[np.ndarray, np.ndarray]]:
    parquet = pq.ParquetFile(path)
    required_columns = {"ticker", "TradingDay", "time_ms", *RAW_FEATURE_COLUMNS}
    missing = required_columns - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} 缺少 snapshot 字段：{sorted(missing)}")
    wanted_by_date: dict[int, set[str]] = {}
    for trading_date, symbol in targets:
        wanted_by_date.setdefault(int(trading_date.strftime("%Y%m%d")), set()).add(symbol)
    union_symbols = sorted(set().union(*wanted_by_date.values())) if wanted_by_date else []
    if not union_symbols:
        return {}

    buffers: dict[tuple[date, str], tuple[np.ndarray, np.ndarray]] = {}
    columns = ["ticker", "TradingDay", "time_ms", *RAW_FEATURE_COLUMNS]
    total_events = config.chunks_per_sample * config.chunk_size
    for row_group in range(parquet.metadata.num_row_groups):
        symbol_range = _row_group_symbol_range(parquet, row_group)
        if symbol_range is not None and not _range_intersects_symbols(union_symbols, *symbol_range):
            report.skipped_row_groups += 1
            continue
        report.scanned_row_groups += 1
        table = parquet.read_row_group(row_group, columns=columns)
        times = table["time_ms"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dates = table["TradingDay"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        tickers = np.asarray([str(value).zfill(6) for value in table["ticker"].to_pylist()])
        mask = (times >= config.scan_start_time_ms) & (times < config.signal_time_ms)
        if np.any(mask):
            mask &= np.isin(tickers, union_symbols)
        selected_rows = np.flatnonzero(mask)
        if selected_rows.size == 0:
            continue

        selected_dates = dates[selected_rows]
        selected_tickers = tickers[selected_rows]
        exact = np.fromiter(
            (
                ticker in wanted_by_date.get(int(trading_date), set())
                for trading_date, ticker in zip(selected_dates, selected_tickers, strict=True)
            ),
            dtype=np.bool_,
            count=selected_rows.size,
        )
        selected_rows = selected_rows[exact]
        if selected_rows.size == 0:
            continue
        selected_dates = dates[selected_rows]
        selected_tickers = tickers[selected_rows]
        selected_times = times[selected_rows]
        selected_features = np.column_stack(
            [
                table[column].to_numpy(zero_copy_only=False)[selected_rows]
                for column in RAW_FEATURE_COLUMNS
            ]
        ).astype(np.float64, copy=False)
        valid_rows = valid_lob_event_rows(selected_features)
        report.invalid_lob_rows += int((~valid_rows).sum())
        selected_dates = selected_dates[valid_rows]
        selected_tickers = selected_tickers[valid_rows]
        selected_times = selected_times[valid_rows]
        selected_features = selected_features[valid_rows]
        if selected_features.shape[0] == 0:
            continue

        boundaries = (
            np.flatnonzero(
                (selected_dates[1:] != selected_dates[:-1])
                | (selected_tickers[1:] != selected_tickers[:-1])
            )
            + 1
        )
        for indices in np.split(np.arange(selected_features.shape[0]), boundaries):
            key = (_yyyymmdd(selected_dates[indices[0]]), str(selected_tickers[indices[0]]))
            _update_tail(
                buffers,
                key,
                selected_times[indices],
                selected_features[indices],
                total_events=total_events,
            )
    return buffers


def _read_daily_month_tail(
    root: Path,
    year: int,
    month: int,
    targets: Mapping[tuple[date, str], Any],
    config: SnapshotPreparationConfig,
    report: ExtractionReport,
) -> dict[tuple[date, str], tuple[np.ndarray, np.ndarray]]:
    """月文件损坏时，从覆盖完整的逐日备份读取同月目标。"""
    target_dates = sorted({trading_date for trading_date, _symbol in targets})
    paths = {
        trading_date: root / f"snapshot_{trading_date:%Y%m%d}.parquet"
        for trading_date in target_dates
    }
    missing = [
        trading_date.isoformat() for trading_date, path in paths.items() if not path.is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"逐日 snapshot 备份不完整，缺少交易日：{preview}")

    buffers: dict[tuple[date, str], tuple[np.ndarray, np.ndarray]] = {}
    for trading_date, path in paths.items():
        daily_targets = {key: target for key, target in targets.items() if key[0] == trading_date}
        daily_buffers = _read_month_tail(path, daily_targets, config, report)
        overlap = set(buffers).intersection(daily_buffers)
        if overlap:
            raise ValueError(f"逐日 snapshot 产生重复股票日：{sorted(overlap)[:5]}")
        buffers.update(daily_buffers)

    report.daily_fallback_files += len(paths)
    report.daily_fallback_months.append(f"{year:04d}-{month:02d}")
    return buffers


def _timestamp(trading_date: date, relative_ms: int) -> datetime:
    return datetime.combine(trading_date, time(9, 30)) + timedelta(milliseconds=int(relative_ms))


def iter_snapshot_samples(
    config: SnapshotPreparationConfig,
    targets: Sequence[Any],
    report: ExtractionReport,
) -> Iterator[PreparedSample]:
    """逐月扫描目标股票的 row group，并只保留每个股票日最后 N 个有效 tick。"""
    target_index = {(target.trading_date, target.symbol): target for target in targets}
    report.requested_targets = len(target_index)
    root = Path(config.snapshot_root).expanduser().resolve()
    start = parse_date(config.start_date)
    end = parse_date(config.end_date)

    for year, month in _month_range(start, end):
        month_targets = {
            key: target
            for key, target in target_index.items()
            if key[0].year == year and key[0].month == month
        }
        if not month_targets:
            continue
        path = root / f"snapshot_{year:04d}{month:02d}.parquet"
        if not path.is_file():
            report.missing_snapshot += len(month_targets)
            continue
        counters = (
            report.invalid_lob_rows,
            report.scanned_row_groups,
            report.skipped_row_groups,
        )
        try:
            buffers = _read_month_tail(path, month_targets, config, report)
        except OSError as monthly_error:
            (
                report.invalid_lob_rows,
                report.scanned_row_groups,
                report.skipped_row_groups,
            ) = counters
            try:
                buffers = _read_daily_month_tail(
                    root,
                    year,
                    month,
                    month_targets,
                    config,
                    report,
                )
            except (OSError, ValueError) as fallback_error:
                raise OSError(
                    f"月度 snapshot {path} 无法读取，逐日备份回退也失败：{fallback_error}"
                ) from monthly_error
            report.monthly_file_errors += 1
        for key, target in sorted(month_targets.items()):
            buffered = buffers.get(key)
            if buffered is None:
                report.missing_snapshot += 1
                continue
            times, raw_events = buffered
            if not np.all(valid_lob_event_rows(raw_events)):
                raise RuntimeError("内部错误：盘口尾部仍包含无效事件")
            if raw_events.shape[0] < config.min_valid_events:
                report.insufficient_events += 1
                continue
            normalized = normalize_lob_events(
                raw_events,
                price_scale_bps=config.price_scale_bps,
                volume_log_scale=config.volume_log_scale,
                clip=config.normalized_clip,
            )
            report.written_samples += 1
            yield PreparedSample(
                target=target,
                events=normalized,
                last_event_timestamp=_timestamp(target.trading_date, int(times[-1])),
                signal_timestamp=_timestamp(target.trading_date, config.signal_time_ms),
            )


def _write_report(path: Path, content: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
