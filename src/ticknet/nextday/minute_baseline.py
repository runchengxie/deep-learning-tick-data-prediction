"""分钟级特征基线的共享数据管线。

内部对照 A 和 B 的特征源有两个：

- L2 缓存（``level2_minute_cache/v1``）：snapshot、order、trade 三个模态合并，
  每分钟 30 个微观结构数值特征
- tushare 分钟（``minute_1m_v3``）：每分钟 OHLCV，共 6 个原始量价特征

两个读取器对外返回统一的按股票日组织的分钟行集合，再交给共享的窗口采样和
聚合逻辑。为了避免整年数据占满内存，读取时每个股票日只保留尾部
``keep_minutes`` 个分钟，采样时再从尾部取 ``window_minutes`` 个。
标签、日期切分和评估复用 ``labels`` / ``splits`` / ``metrics`` 模块。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ticknet.nextday.snapshot_config import SnapshotPreparationConfig
from ticknet.nextday.snapshot_io import load_market_panels
from ticknet.nextday.snapshot_targets import build_snapshot_targets
from ticknet.nextday.splits import WalkForwardSplit

L2_MODALITIES = ("snapshot", "order", "trade")
TUSHARE_FEATURE_COLUMNS = ("open", "close", "high", "low", "vol", "amount")
AGGREGATE_METHODS = ("mean", "std", "last", "delta")

DayRows = dict[tuple[int, str], list[tuple[int, np.ndarray]]]


@dataclass(frozen=True)
class MinuteSample:
    """一个股票日的分钟特征矩阵与对齐的监督信息。"""

    trading_date: date
    symbol: str
    label_date: date
    label: int
    target_return: float
    features: np.ndarray


@dataclass(frozen=True)
class MinuteBaselineConfig:
    """分钟基线配置：切分、股票池、特征源与窗口。"""

    basic_root: str
    benchmark_path: str
    start_date: str
    end_date: str
    top_n: int
    min_history_days: int
    liquidity_lookback_days: int
    min_liquidity_observations: int
    lower_quantile: float
    upper_quantile: float
    min_cross_section: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    feature_source: str = "l2_cache"
    l2_root: str = ""
    tushare_root: str = ""
    window_minutes: int = 60
    min_window_minutes: int = 30
    min_symbols_per_day: int = 20
    portfolio_quantile: float = 0.1
    seed: int = 0

    def date_split(self) -> WalkForwardSplit:
        return WalkForwardSplit.from_strings(
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=self.test_start,
            test_end=self.test_end,
        )

    def snapshot_config(self) -> SnapshotPreparationConfig:
        return SnapshotPreparationConfig(
            snapshot_root=self.l2_root,
            basic_root=self.basic_root,
            benchmark_path=self.benchmark_path,
            output_dir="",
            start_date=self.start_date,
            end_date=self.end_date,
            top_n=self.top_n,
            min_history_days=self.min_history_days,
            liquidity_lookback_days=self.liquidity_lookback_days,
            min_liquidity_observations=self.min_liquidity_observations,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
            min_cross_section=self.min_cross_section,
        )


@dataclass
class MinuteExtractionReport:
    """分钟行提取期间累积的质量计数。"""

    requested_targets: int = 0
    written_samples: int = 0
    missing_rows: int = 0
    insufficient_window: int = 0
    scanned_row_groups: int = 0


def build_targets(config: MinuteBaselineConfig) -> list[Any]:
    """复用 snapshot 链路的动态股票池与次日标签构建。"""
    snapshot_config = config.snapshot_config()
    open_panel, close_panel, volume_panel = load_market_panels(config.basic_root)
    targets, _universe = build_snapshot_targets(
        snapshot_config,
        open_panel,
        close_panel,
        volume_panel,
    )
    if not targets:
        raise ValueError("指定日期和股票池没有生成任何次日标签")
    return targets


def _date_int(trading_date: date) -> int:
    return int(trading_date.strftime("%Y%m%d"))


def _feature_columns(names: Sequence[str], modality: str) -> tuple[str, ...]:
    prefix = f"{modality}__"
    return tuple(name for name in names if name.startswith(prefix) and not name.endswith("__valid"))


def _trim_rows(
    rows: OrderedDict[int, np.ndarray],
    keep_minutes: int,
) -> None:
    while len(rows) > keep_minutes:
        rows.popitem(last=False)


def _stream_l2_modality(
    path: Path,
    wanted_dates: set[int],
    wanted_symbols: set[str],
    modality: str,
    keep_minutes: int,
    report: MinuteExtractionReport,
) -> DayRows:
    """流式读取单个 L2 模态，每股票日只保留尾部分钟。"""
    parquet = pq.ParquetFile(path)
    features = _feature_columns(parquet.schema_arrow.names, modality)
    if not features:
        raise ValueError(f"{path} 缺少 {modality} 特征列")
    columns = ["date", "ticker", "minute", *features]
    wanted_dates_array = np.fromiter(wanted_dates, dtype=np.int64)
    wanted_symbols_array = np.fromiter(wanted_symbols, dtype="U8")
    result: dict[tuple[int, str], OrderedDict[int, np.ndarray]] = {}
    for row_group in range(parquet.metadata.num_row_groups):
        report.scanned_row_groups += 1
        table = parquet.read_row_group(row_group, columns=columns)
        dates = table["date"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        tickers = np.asarray(table["ticker"].to_pylist(), dtype="U8")
        minutes = table["minute"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        mask = np.isin(dates, wanted_dates_array) & np.isin(tickers, wanted_symbols_array)
        if not np.any(mask):
            continue
        dates = dates[mask]
        tickers = tickers[mask]
        minutes = minutes[mask]
        feature_matrix = np.column_stack(
            [
                table[column].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)[mask]
                for column in features
            ]
        )
        for index in range(len(dates)):
            key = (int(dates[index]), str(tickers[index]))
            minute_rows = result.get(key)
            if minute_rows is None:
                minute_rows = OrderedDict()
                result[key] = minute_rows
            minute_rows[int(minutes[index])] = feature_matrix[index]
            _trim_rows(minute_rows, keep_minutes)
    return {key: list(rows.items()) for key, rows in result.items()}


def _merge_modalities(
    modal_rows: dict[tuple[str, int], DayRows],
    keep_minutes: int,
) -> DayRows:
    """按分钟对多个 L2 模态做严格内连接，保证每行特征维度一致。"""
    by_key: dict[tuple[int, str], list[dict[int, np.ndarray]]] = {}
    for (_modality, _year), rows in modal_rows.items():
        for key, day_rows in rows.items():
            by_key.setdefault(key, []).append(dict(day_rows))
    merged: dict[tuple[int, str], OrderedDict[int, np.ndarray]] = {}
    for key, per_modality in by_key.items():
        if len(per_modality) < len(L2_MODALITIES):
            continue
        common_minutes = set.intersection(*(set(rows) for rows in per_modality))
        if not common_minutes:
            continue
        combined: OrderedDict[int, np.ndarray] = OrderedDict()
        for minute_value in sorted(common_minutes):
            combined[minute_value] = np.concatenate([rows[minute_value] for rows in per_modality])
        _trim_rows(combined, keep_minutes)
        merged[key] = combined
    return {key: list(rows.items()) for key, rows in merged.items()}


def read_l2_minute_rows(
    l2_root: str | Path,
    targets: Sequence[Any],
    keep_minutes: int,
    report: MinuteExtractionReport,
) -> DayRows:
    """读取 L2 分钟缓存，按分钟对齐三个模态后返回每股票日的有序分钟行。"""
    root = Path(l2_root).expanduser().resolve()
    wanted = {(_date_int(target.trading_date), target.symbol) for target in targets}
    wanted_dates = {item[0] for item in wanted}
    wanted_symbols = {item[1] for item in wanted}
    years = sorted({target.trading_date.year for target in targets})
    modal_rows: dict[tuple[str, int], DayRows] = {}
    for year in years:
        for modality in L2_MODALITIES:
            monthly = root / "yearly" / modality / f"{year}.parquet"
            if not monthly.is_file():
                raise FileNotFoundError(f"缺少 {modality} 分钟缓存文件：{monthly}")
            modal_rows[(modality, year)] = _stream_l2_modality(
                monthly,
                wanted_dates,
                wanted_symbols,
                modality,
                keep_minutes,
                report,
            )
    return _merge_modalities(modal_rows, keep_minutes)


def _tushare_symbol(ts_code: str) -> str:
    return ts_code.split(".")[0]


def read_tushare_minute_rows(
    tushare_root: str | Path,
    targets: Sequence[Any],
    keep_minutes: int,
    report: MinuteExtractionReport,
) -> DayRows:
    """读取 tushare 分钟 OHLCV，返回每股票日的有序分钟行。"""
    root = Path(tushare_root).expanduser().resolve()
    wanted = {(_date_int(target.trading_date), target.symbol) for target in targets}
    wanted_dates = {item[0] for item in wanted}
    wanted_symbols = {item[1] for item in wanted}
    result: dict[tuple[int, str], OrderedDict[int, np.ndarray]] = {}
    for date_int in sorted(wanted_dates):
        partition = root / f"trade_date={date_int}"
        if not partition.is_dir():
            continue
        for file in sorted(partition.glob("*.parquet")):
            columns = ["ts_code", "trade_time", *TUSHARE_FEATURE_COLUMNS]
            table = pq.read_table(file, columns=columns)
            codes = np.asarray([_tushare_symbol(code) for code in table["ts_code"].to_pylist()])
            mask = np.isin(codes, np.fromiter(wanted_symbols, dtype="U8"))
            if not np.any(mask):
                continue
            epoch_minutes = (
                table["trade_time"]
                .to_numpy(zero_copy_only=False)
                .astype("datetime64[m]")
                .astype(np.int64)
            )
            minute_of_day = np.mod(epoch_minutes, 24 * 60)
            feature_matrix = np.column_stack(
                [
                    table[column].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
                    for column in TUSHARE_FEATURE_COLUMNS
                ]
            )
            for index in np.flatnonzero(mask):
                key = (date_int, str(codes[index]))
                minute_rows = result.get(key)
                if minute_rows is None:
                    minute_rows = OrderedDict()
                    result[key] = minute_rows
                minute_rows[int(minute_of_day[index])] = feature_matrix[index]
                _trim_rows(minute_rows, keep_minutes)
    return {key: list(rows.items()) for key, rows in result.items()}


def _aggregate_trailing(matrix: np.ndarray) -> np.ndarray:
    """对窗口内每个特征取均值、标准差、末值和末值减首值。"""
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("特征矩阵应为至少一行的二维数组")
    last = matrix[-1]
    first = matrix[0]
    return np.concatenate(
        (
            matrix.mean(axis=0),
            matrix.std(axis=0),
            last,
            last - first,
        )
    )


def build_samples(
    rows: Mapping[tuple[int, str], Sequence[tuple[int, np.ndarray]]],
    targets: Sequence[Any],
    *,
    window_minutes: int,
    min_window_minutes: int,
    report: MinuteExtractionReport,
) -> list[MinuteSample]:
    """把每股票日分钟行切成尾部窗口并聚合为样本。"""
    if window_minutes < 1 or min_window_minutes < 1:
        raise ValueError("窗口分钟数必须为正整数")
    if min_window_minutes > window_minutes:
        raise ValueError("min_window_minutes 不能大于 window_minutes")
    report.requested_targets = len(targets)
    samples: list[MinuteSample] = []
    for target in targets:
        key = (_date_int(target.trading_date), target.symbol)
        day_rows = rows.get(key)
        if not day_rows:
            report.missing_rows += 1
            continue
        window_rows = day_rows[-window_minutes:]
        if len(window_rows) < min_window_minutes:
            report.insufficient_window += 1
            continue
        matrix = np.stack([row for _minute, row in window_rows])
        samples.append(
            MinuteSample(
                trading_date=target.trading_date,
                symbol=target.symbol,
                label_date=target.label_date,
                label=target.label,
                target_return=target.target_return,
                features=_aggregate_trailing(matrix),
            )
        )
        report.written_samples += 1
    return samples
