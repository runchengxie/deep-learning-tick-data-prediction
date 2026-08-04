"""把沪深十档 snapshot 月度 Parquet 转成端到端 DeepLOB 分片。"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from deeplob.nextday.io import PreparedSample, write_sharded_dataset
from deeplob.nextday.labels import DailyBar, NextDayTarget, build_next_day_targets
from deeplob.nextday.splits import parse_date

RAW_FEATURE_COLUMNS = tuple(
    name
    for level in range(1, 11)
    for name in (
        f"AskPrice{level}",
        f"AskVolume{level}",
        f"BidPrice{level}",
        f"BidVolume{level}",
    )
)
PRICE_INDICES = np.asarray(
    [index for index in range(len(RAW_FEATURE_COLUMNS)) if index % 2 == 0],
    dtype=np.int64,
)
VOLUME_INDICES = np.asarray(
    [index for index in range(len(RAW_FEATURE_COLUMNS)) if index % 2 == 1],
    dtype=np.int64,
)
SHANGHAI_SHENZHEN_STOCK = re.compile(r"^(?:000|001|002|003|300|301|600|601|603|605|688|689)\d{3}$")


@dataclass(frozen=True)
class DailyPanel:
    """日期 × 股票的宽表日线矩阵。"""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.dates), len(self.symbols)):
            raise ValueError("日线矩阵形状与日期、股票轴不一致")
        if tuple(sorted(set(self.dates))) != self.dates:
            raise ValueError("日线日期必须严格递增")
        if tuple(sorted(set(self.symbols))) != self.symbols:
            raise ValueError("日线股票代码必须严格递增")


@dataclass(frozen=True)
class SnapshotPreparationConfig:
    """真实 snapshot 端到端数据准备配置。"""

    snapshot_root: str
    basic_root: str
    benchmark_path: str
    output_dir: str
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    signal_time_ms: int = 19_500_000
    scan_start_time_ms: int = 18_000_000
    chunks_per_sample: int = 2
    chunk_size: int = 100
    min_valid_events: int = 150
    top_n: int = 400
    min_history_days: int = 120
    liquidity_lookback_days: int = 20
    min_liquidity_observations: int = 15
    lower_quantile: float = 0.2
    upper_quantile: float = 0.8
    min_cross_section: int = 20
    samples_per_shard: int = 2048
    storage_dtype: str = "float16"
    price_scale_bps: float = 100.0
    volume_log_scale: float = 16.0
    normalized_clip: float = 32.0

    def validate(self) -> None:
        start = parse_date(self.start_date)
        end = parse_date(self.end_date)
        if end < start:
            raise ValueError("end_date 不能早于 start_date")
        for field_name in ("snapshot_root", "basic_root", "benchmark_path", "output_dir"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} 不能为空")
        if not 0 <= self.scan_start_time_ms < self.signal_time_ms:
            raise ValueError("扫描起点必须早于信号时点且不能为负数")
        if self.chunks_per_sample < 1 or self.chunk_size < 1:
            raise ValueError("chunks_per_sample 和 chunk_size 应为正整数")
        total_events = self.chunks_per_sample * self.chunk_size
        if not 1 <= self.min_valid_events <= total_events:
            raise ValueError("min_valid_events 必须位于 1 和总事件数之间")
        if self.top_n < 1 or self.min_history_days < 1:
            raise ValueError("top_n 和 min_history_days 应为正整数")
        if self.liquidity_lookback_days < 1:
            raise ValueError("liquidity_lookback_days 应为正整数")
        if not 1 <= self.min_liquidity_observations <= self.liquidity_lookback_days:
            raise ValueError("min_liquidity_observations 超出流动性窗口")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype 应为 float16 或 float32")
        if self.samples_per_shard < 1:
            raise ValueError("samples_per_shard 应为正整数")
        if self.price_scale_bps <= 0 or self.volume_log_scale <= 0:
            raise ValueError("固定归一化尺度必须为正数")
        if self.normalized_clip <= 0:
            raise ValueError("normalized_clip 必须为正数")


@dataclass
class ExtractionReport:
    """原始盘口提取期间累积的数据质量计数。"""

    requested_targets: int = 0
    written_samples: int = 0
    missing_snapshot: int = 0
    insufficient_events: int = 0
    invalid_lob_rows: int = 0
    scanned_row_groups: int = 0
    skipped_row_groups: int = 0
    monthly_file_errors: int = 0
    daily_fallback_files: int = 0
    daily_fallback_months: list[str] = field(default_factory=list)


def _yyyymmdd(value: object) -> date:
    raw = str(value).replace("-", "")
    if len(raw) != 8:
        raise ValueError(f"无效交易日：{value!r}")
    return datetime.strptime(raw, "%Y%m%d").date()


def _valid_stock_symbols(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(name for name in names if SHANGHAI_SHENZHEN_STOCK.fullmatch(name)))


def read_wide_daily_panel(
    path: str | Path,
    *,
    symbols: Sequence[str] | None = None,
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
    table = parquet.read(columns=["value", *selected])
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


def build_dynamic_universe(
    open_panel: DailyPanel,
    close_panel: DailyPanel,
    volume_panel: DailyPanel,
    *,
    start_date: date,
    end_date: date,
    top_n: int,
    min_history_days: int,
    liquidity_lookback_days: int,
    min_liquidity_observations: int,
) -> dict[date, tuple[str, ...]]:
    """只用输入日前数据生成历史动态流动性股票池。"""
    if (open_panel.dates, open_panel.symbols) != (close_panel.dates, close_panel.symbols):
        raise ValueError("open 和 close 日线轴不一致")
    if (open_panel.dates, open_panel.symbols) != (volume_panel.dates, volume_panel.symbols):
        raise ValueError("open 和 volume 日线轴不一致")
    if top_n < 1 or min_history_days < 1 or liquidity_lookback_days < 1:
        raise ValueError("股票池参数必须为正整数")
    if not 1 <= min_liquidity_observations <= liquidity_lookback_days:
        raise ValueError("min_liquidity_observations 超出流动性窗口")

    historical_valid = (
        np.isfinite(open_panel.values)
        & (open_panel.values > 0)
        & np.isfinite(close_panel.values)
        & (close_panel.values > 0)
        & np.isfinite(volume_panel.values)
        & (volume_panel.values > 0)
    )
    has_history = historical_valid.any(axis=0)
    first_seen = np.where(has_history, historical_valid.argmax(axis=0), len(open_panel.dates))
    universe: dict[date, tuple[str, ...]] = {}

    for day_index, trading_date in enumerate(open_panel.dates):
        if trading_date < start_date or trading_date > end_date:
            continue
        window_start = max(0, day_index - liquidity_lookback_days)
        if day_index - window_start < liquidity_lookback_days:
            continue
        prior_close = close_panel.values[window_start:day_index]
        prior_volume = volume_panel.values[window_start:day_index]
        valid = historical_valid[window_start:day_index]
        observations = valid.sum(axis=0)
        turnover = np.where(valid, prior_close * prior_volume, 0.0)
        liquidity = np.divide(
            turnover.sum(axis=0),
            observations,
            out=np.full(observations.shape, np.nan, dtype=np.float64),
            where=observations > 0,
        )
        eligible = (
            (day_index - first_seen >= min_history_days)
            & (observations >= min_liquidity_observations)
            & np.isfinite(liquidity)
            & (liquidity > 0)
        )
        candidates = np.flatnonzero(eligible)
        if candidates.size == 0:
            universe[trading_date] = ()
            continue
        order = np.argsort(liquidity[candidates], kind="mergesort")[::-1]
        selected = candidates[order[:top_n]]
        universe[trading_date] = tuple(sorted(open_panel.symbols[index] for index in selected))
    return universe


def read_benchmark_open_close_returns(path: str | Path) -> dict[date, float]:
    source = Path(path).expanduser().resolve()
    table = pq.read_table(source, columns=["trade_date", "open", "close"])
    returns: dict[date, float] = {}
    for raw_date, raw_open, raw_close in zip(
        table["trade_date"].to_pylist(),
        table["open"].to_pylist(),
        table["close"].to_pylist(),
        strict=True,
    ):
        trading_date = _yyyymmdd(raw_date)
        open_price = float(raw_open)
        close_price = float(raw_close)
        if open_price > 0 and close_price > 0 and math.isfinite(open_price + close_price):
            returns[trading_date] = close_price / open_price - 1.0
    if not returns:
        raise ValueError(f"{source} 没有有效基准开收盘收益")
    return returns


def _bars_for_universe(
    open_panel: DailyPanel,
    close_panel: DailyPanel,
    universe: Mapping[date, Sequence[str]],
) -> list[DailyBar]:
    date_index = {trading_date: index for index, trading_date in enumerate(open_panel.dates)}
    symbol_index = {symbol: index for index, symbol in enumerate(open_panel.symbols)}
    required: set[tuple[str, date]] = set()
    calendar = open_panel.dates
    next_date = dict(pairwise(calendar))
    for trading_date, symbols in universe.items():
        label_date = next_date.get(trading_date)
        if label_date is None:
            continue
        required.update((symbol, label_date) for symbol in symbols)

    bars: list[DailyBar] = []
    for symbol, trading_date in sorted(required, key=lambda item: (item[1], item[0])):
        row = date_index.get(trading_date)
        column = symbol_index.get(symbol)
        if row is None or column is None:
            continue
        open_price = float(open_panel.values[row, column])
        close_price = float(close_panel.values[row, column])
        if (
            open_price > 0
            and close_price > 0
            and math.isfinite(open_price)
            and math.isfinite(close_price)
        ):
            bars.append(DailyBar(symbol, trading_date, open_price, close_price))
    return bars


def build_snapshot_targets(
    config: SnapshotPreparationConfig,
    open_panel: DailyPanel,
    close_panel: DailyPanel,
    volume_panel: DailyPanel,
) -> tuple[list[NextDayTarget], dict[date, tuple[str, ...]]]:
    start = parse_date(config.start_date)
    end = parse_date(config.end_date)
    universe = build_dynamic_universe(
        open_panel,
        close_panel,
        volume_panel,
        start_date=start,
        end_date=end,
        top_n=config.top_n,
        min_history_days=config.min_history_days,
        liquidity_lookback_days=config.liquidity_lookback_days,
        min_liquidity_observations=config.min_liquidity_observations,
    )
    benchmark_returns = read_benchmark_open_close_returns(config.benchmark_path)
    targets = build_next_day_targets(
        _bars_for_universe(open_panel, close_panel, universe),
        open_panel.dates,
        label_method="cross_sectional",
        lower_quantile=config.lower_quantile,
        upper_quantile=config.upper_quantile,
        benchmark_returns=benchmark_returns,
        min_cross_section=config.min_cross_section,
        universe=universe,
    )
    return targets, universe


def normalize_lob_events(
    raw_events: np.ndarray,
    *,
    price_scale_bps: float = 100.0,
    volume_log_scale: float = 16.0,
    clip: float = 32.0,
) -> np.ndarray:
    """使用固定、无拟合参数的尺度变换保留逐 tick 盘口状态。"""
    events = np.asarray(raw_events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != len(RAW_FEATURE_COLUMNS):
        raise ValueError(f"raw_events 应为 N × {len(RAW_FEATURE_COLUMNS)}")
    if events.shape[0] < 1:
        raise ValueError("raw_events 不能为空")
    prices = events[:, PRICE_INDICES]
    volumes = events[:, VOLUME_INDICES]
    if not np.all(np.isfinite(events)) or np.any(prices <= 0) or np.any(volumes < 0):
        raise ValueError("raw_events 包含无效价格或数量")
    reference_mid = (events[0, 0] + events[0, 2]) / 2.0
    if not math.isfinite(reference_mid) or reference_mid <= 0:
        raise ValueError("首个盘口缺少有效中间价")

    normalized = np.empty_like(events, dtype=np.float64)
    normalized[:, PRICE_INDICES] = ((prices / reference_mid - 1.0) * 10_000.0) / (price_scale_bps)
    normalized[:, VOLUME_INDICES] = np.log1p(volumes) / volume_log_scale
    np.clip(normalized, -clip, clip, out=normalized)
    return normalized.astype(np.float32)


def valid_lob_event_rows(raw_events: np.ndarray) -> np.ndarray:
    """返回价格、数量和有限值均有效的逐事件布尔掩码。"""
    events = np.asarray(raw_events)
    if events.ndim != 2 or events.shape[1] != len(RAW_FEATURE_COLUMNS):
        raise ValueError(f"raw_events 应为 N × {len(RAW_FEATURE_COLUMNS)}")
    prices = events[:, PRICE_INDICES]
    volumes = events[:, VOLUME_INDICES]
    return (
        np.all(np.isfinite(events), axis=1)
        & np.all(prices > 0, axis=1)
        & np.all(volumes >= 0, axis=1)
    )


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
    targets: Mapping[tuple[date, str], NextDayTarget],
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
                for trading_date, ticker in zip(
                    selected_dates,
                    selected_tickers,
                    strict=True,
                )
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
    targets: Mapping[tuple[date, str], NextDayTarget],
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
    targets: Sequence[NextDayTarget],
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


def prepare_snapshot_dataset(config: SnapshotPreparationConfig) -> tuple[Path, dict[str, Any]]:
    """执行股票池、标签、原始盘口提取并写出 Colab 可搬运分片。"""
    config.validate()
    open_panel, close_panel, volume_panel = load_market_panels(config.basic_root)
    targets, universe = build_snapshot_targets(config, open_panel, close_panel, volume_panel)
    if not targets:
        raise ValueError("指定日期和股票池没有生成任何次日标签")
    report = ExtractionReport()
    manifest = write_sharded_dataset(
        iter_snapshot_samples(config, targets, report),
        config.output_dir,
        chunks_per_sample=config.chunks_per_sample,
        chunk_size=config.chunk_size,
        samples_per_shard=config.samples_per_shard,
        storage_dtype=config.storage_dtype,
        metadata={
            "source": "cn_a_share_level2_snapshot",
            "signal_time_ms": config.signal_time_ms,
            "scan_start_time_ms": config.scan_start_time_ms,
            "min_valid_events": config.min_valid_events,
            "normalization": {
                "price": "first_selected_mid_relative_bps",
                "price_scale_bps": config.price_scale_bps,
                "volume": "log1p",
                "volume_log_scale": config.volume_log_scale,
                "clip": config.normalized_clip,
            },
        },
    )
    selected_counts = [len(symbols) for symbols in universe.values()]
    audit: dict[str, Any] = {
        "config": asdict(config),
        "extraction": asdict(report),
        "universe": {
            "dates": len(selected_counts),
            "minimum": min(selected_counts, default=0),
            "median": float(np.median(selected_counts)) if selected_counts else 0.0,
            "maximum": max(selected_counts, default=0),
        },
        "manifest": str(manifest),
    }
    _write_report(Path(config.output_dir) / "data-audit.json", audit)
    return manifest, audit


def _build_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从沪深十档 snapshot 月度 Parquet 生成端到端 DeepLOB 分片"
    )
    parser.add_argument("--config")
    for name, value in defaults.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, int):
            parser.add_argument(option, type=int)
        elif isinstance(value, float):
            parser.add_argument(option, type=float)
        else:
            parser.add_argument(option)
    parser.set_defaults(**defaults)
    return parser


def load_snapshot_config(argv: list[str] | None = None) -> SnapshotPreparationConfig:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)
    values = asdict(
        SnapshotPreparationConfig(
            snapshot_root="",
            basic_root="",
            benchmark_path="",
            output_dir="",
        )
    )
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        if not isinstance(file_values, dict):
            raise SystemExit("snapshot YAML 根节点应为对象")
        valid_names = {item.name for item in fields(SnapshotPreparationConfig)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"snapshot YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)
    parser = _build_parser(values)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = SnapshotPreparationConfig(**arguments)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    manifest, audit = prepare_snapshot_dataset(load_snapshot_config(argv))
    print(f"已写入 {audit['extraction']['written_samples']:,} 个端到端股票日样本")
    print(f"数据清单：{manifest}")


if __name__ == "__main__":
    main()
