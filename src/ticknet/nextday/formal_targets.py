"""正式 Top-K 的 open-to-following-open 标签与开盘交易状态。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ticknet.nextday.labels import NEUTRAL, NextDayTarget, _rank_labels
from ticknet.nextday.snapshot_config import DailyPanel, SnapshotPreparationConfig, _yyyymmdd
from ticknet.nextday.snapshot_features import build_dynamic_universe
from ticknet.nextday.snapshot_io import read_wide_daily_panel
from ticknet.nextday.splits import parse_date

FORMAL_TARGET_RETURN_CONTRACT = "next_open_to_following_open"
FORMAL_SUSPENDED_MARK_POLICY = "previous_close"


@dataclass(frozen=True)
class FormalMarketPanels:
    """生成正式收益与交易状态所需的对齐日线面板。"""

    open: DailyPanel
    high: DailyPanel
    low: DailyPanel
    close: DailyPanel
    volume: DailyPanel
    st: DailyPanel

    def __post_init__(self) -> None:
        axes = (self.open.dates, self.open.symbols)
        for name in ("high", "low", "close", "volume", "st"):
            panel = getattr(self, name)
            if (panel.dates, panel.symbols) != axes:
                raise ValueError(f"{name} 日线轴与 open 不一致")


@dataclass(frozen=True)
class OpenExecutionState:
    """一个股票在某交易日开盘的估值与可成交方向。"""

    mark_price: float
    can_buy: bool
    can_sell: bool
    status: str


@dataclass(frozen=True)
class FormalNextOpenTarget(NextDayTarget):
    """T 日信号对应 T+1 开盘到 T+2 开盘的正式监督目标。"""

    return_end_date: date = date.min
    portfolio_return: float = 0.0
    benchmark_return: float = 0.0
    can_buy: bool = True
    can_sell: bool = True
    in_universe: bool = True
    execution_status: str = "normal"


@dataclass
class FormalTargetBuildReport:
    """正式目标构建的覆盖和交易状态计数。"""

    requested_signal_dates: int = 0
    complete_signal_dates: int = 0
    incomplete_universe_dates: int = 0
    missing_market_state_dates: int = 0
    candidate_targets: int = 0
    status_only_targets: int = 0
    suspended_rows: int = 0
    one_price_limit_up_rows: int = 0
    one_price_limit_down_rows: int = 0


def _aligned_optional_panel(
    path: Path,
    *,
    dates: tuple[date, ...],
    symbols: tuple[str, ...],
) -> DailyPanel:
    available = set(pq.ParquetFile(path).schema_arrow.names) - {"value"}
    selected = tuple(symbol for symbol in symbols if symbol in available)
    source = read_wide_daily_panel(
        path,
        symbols=selected,
        start_date=dates[0],
        end_date=dates[-1],
    )
    if source.dates != dates:
        raise ValueError(f"{path.name} 日线日期与 open 不一致")
    values = np.zeros((len(dates), len(symbols)), dtype=np.float64)
    target_index = {symbol: index for index, symbol in enumerate(symbols)}
    for source_column, symbol in enumerate(source.symbols):
        values[:, target_index[symbol]] = source.values[:, source_column]
    return DailyPanel(dates=dates, symbols=symbols, values=values)


def load_formal_market_panels(
    basic_root: str | Path,
    *,
    end_date: date | None = None,
) -> FormalMarketPanels:
    """读取并对齐 open/high/low/close/volume/ST 宽表。"""
    root = Path(basic_root).expanduser().resolve()
    required_paths = {
        name: root / f"{name}_data.parquet" for name in ("open", "high", "low", "close", "volume")
    }
    schemas = {
        name: set(pq.ParquetFile(path).schema_arrow.names) - {"value"}
        for name, path in required_paths.items()
    }
    symbols = tuple(sorted(set.intersection(*schemas.values())))
    open_panel = read_wide_daily_panel(required_paths["open"], symbols=symbols, end_date=end_date)
    panels = {
        name: read_wide_daily_panel(path, symbols=symbols, end_date=end_date)
        for name, path in required_paths.items()
        if name != "open"
    }
    st = _aligned_optional_panel(
        root / "st_data.parquet",
        dates=open_panel.dates,
        symbols=open_panel.symbols,
    )
    return FormalMarketPanels(open=open_panel, st=st, **panels)


def read_benchmark_open_to_open_returns(
    path: str | Path,
    calendar: Sequence[date],
) -> dict[date, float]:
    """按股票交易日历计算 T+1 open 到 T+2 open 的基准收益。"""
    source = Path(path).expanduser().resolve()
    table = pq.read_table(source, columns=["trade_date", "open"])
    opens: dict[date, float] = {}
    for raw_date, raw_open in zip(
        table["trade_date"].to_pylist(), table["open"].to_pylist(), strict=True
    ):
        trading_date = _yyyymmdd(raw_date)
        open_price = float(raw_open)
        if open_price > 0 and math.isfinite(open_price):
            opens[trading_date] = open_price
    returns: dict[date, float] = {}
    for start, end in pairwise(calendar):
        if start not in opens or end not in opens:
            continue
        returns[start] = opens[end] / opens[start] - 1.0
    if not returns:
        raise ValueError(f"{source} 没有与股票日历对齐的 open-to-open 收益")
    return returns


def _previous_valid_close(close: np.ndarray) -> np.ndarray:
    previous = np.full(close.shape, np.nan, dtype=np.float64)
    last = np.full(close.shape[1], np.nan, dtype=np.float64)
    for row in range(close.shape[0]):
        previous[row] = last
        current = close[row]
        valid = np.isfinite(current) & (current > 0)
        last[valid] = current[valid]
    return previous


def _price_limit_ratio(symbol: str, trading_date: date, *, is_st: bool) -> Decimal:
    if is_st:
        return Decimal("0.05")
    if symbol.startswith(("4", "8")):
        return Decimal("0.30")
    if symbol.startswith(("688", "689")):
        return Decimal("0.20")
    if symbol.startswith(("300", "301")) and trading_date >= date(2020, 8, 24):
        return Decimal("0.20")
    return Decimal("0.10")


def _limit_price(previous_close: float, ratio: Decimal, *, upper: bool) -> float:
    multiplier = Decimal("1") + ratio if upper else Decimal("1") - ratio
    return float(
        (Decimal(str(previous_close)) * multiplier).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def _same_tick(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.0051)


def open_execution_state(
    *,
    symbol: str,
    trading_date: date,
    open_price: float,
    high_price: float,
    low_price: float,
    volume: float,
    previous_close: float,
    is_st: bool,
) -> OpenExecutionState | None:
    """按停牌与一字涨跌停规则生成开盘交易状态。"""
    if not math.isfinite(previous_close) or previous_close <= 0:
        return None
    valid_bar = (
        math.isfinite(open_price)
        and open_price > 0
        and math.isfinite(high_price)
        and high_price > 0
        and math.isfinite(low_price)
        and low_price > 0
        and math.isfinite(volume)
        and volume > 0
    )
    if not valid_bar:
        return OpenExecutionState(previous_close, False, False, "suspended")
    one_price = _same_tick(open_price, high_price) and _same_tick(open_price, low_price)
    if one_price:
        ratio = _price_limit_ratio(symbol, trading_date, is_st=is_st)
        if _same_tick(open_price, _limit_price(previous_close, ratio, upper=True)):
            return OpenExecutionState(open_price, False, True, "one_price_limit_up")
        if _same_tick(open_price, _limit_price(previous_close, ratio, upper=False)):
            return OpenExecutionState(open_price, True, False, "one_price_limit_down")
    return OpenExecutionState(open_price, True, True, "normal")


def _record_state(report: FormalTargetBuildReport, status: str) -> None:
    if status == "suspended":
        report.suspended_rows += 1
    elif status == "one_price_limit_up":
        report.one_price_limit_up_rows += 1
    elif status == "one_price_limit_down":
        report.one_price_limit_down_rows += 1


def _target_for_symbol(
    *,
    symbol: str,
    trading_date: date,
    label_date: date,
    return_end_date: date,
    column: int,
    label_row: int,
    end_row: int,
    panels: FormalMarketPanels,
    previous_close: np.ndarray,
    benchmark_return: float,
    in_universe: bool,
) -> FormalNextOpenTarget | None:
    start = open_execution_state(
        symbol=symbol,
        trading_date=label_date,
        open_price=float(panels.open.values[label_row, column]),
        high_price=float(panels.high.values[label_row, column]),
        low_price=float(panels.low.values[label_row, column]),
        volume=float(panels.volume.values[label_row, column]),
        previous_close=float(previous_close[label_row, column]),
        is_st=bool(panels.st.values[label_row, column] > 0),
    )
    end = open_execution_state(
        symbol=symbol,
        trading_date=return_end_date,
        open_price=float(panels.open.values[end_row, column]),
        high_price=float(panels.high.values[end_row, column]),
        low_price=float(panels.low.values[end_row, column]),
        volume=float(panels.volume.values[end_row, column]),
        previous_close=float(previous_close[end_row, column]),
        is_st=bool(panels.st.values[end_row, column] > 0),
    )
    if start is None or end is None:
        return None
    raw_return = end.mark_price / start.mark_price - 1.0
    if not math.isfinite(raw_return):
        return None
    return FormalNextOpenTarget(
        symbol=symbol,
        trading_date=trading_date,
        label_date=label_date,
        return_end_date=return_end_date,
        raw_return=raw_return,
        target_return=raw_return - benchmark_return,
        portfolio_return=raw_return,
        benchmark_return=benchmark_return,
        can_buy=start.can_buy,
        can_sell=start.can_sell,
        in_universe=in_universe,
        execution_status=start.status,
    )


def build_formal_next_open_targets(
    config: SnapshotPreparationConfig,
    panels: FormalMarketPanels,
) -> tuple[
    list[FormalNextOpenTarget],
    dict[date, tuple[str, ...]],
    FormalTargetBuildReport,
]:
    """构造完整 Top-N 候选、状态行和 open-to-following-open 标签。"""
    start_date = parse_date(config.start_date)
    end_date = parse_date(config.end_date)
    universe = build_dynamic_universe(
        panels.open,
        panels.close,
        panels.volume,
        start_date=start_date,
        end_date=end_date,
        top_n=config.top_n,
        min_history_days=config.min_history_days,
        liquidity_lookback_days=config.liquidity_lookback_days,
        min_liquidity_observations=config.min_liquidity_observations,
    )
    benchmark_returns = read_benchmark_open_to_open_returns(
        config.benchmark_path, panels.open.dates
    )
    previous_close = _previous_valid_close(panels.close.values)
    symbol_index = {symbol: index for index, symbol in enumerate(panels.open.symbols)}
    report = FormalTargetBuildReport()
    targets: list[FormalNextOpenTarget] = []
    previous_universe: set[str] = set()
    carried_status: set[str] = set()

    calendar = panels.open.dates
    for signal_row, (trading_date, label_date, return_end_date) in enumerate(
        zip(calendar, calendar[1:], calendar[2:], strict=False)
    ):
        if trading_date < start_date or trading_date > end_date or return_end_date > end_date:
            continue
        report.requested_signal_dates += 1
        selected = set(universe.get(trading_date, ()))
        if len(selected) != config.top_n:
            report.incomplete_universe_dates += 1
            previous_universe = selected
            continue
        benchmark_return = benchmark_returns.get(label_date)
        if benchmark_return is None or not math.isfinite(benchmark_return):
            report.missing_market_state_dates += 1
            previous_universe = selected
            continue
        label_row = signal_row + 1
        end_row = signal_row + 2
        candidate_rows: list[FormalNextOpenTarget] = []
        for symbol in sorted(selected):
            target = _target_for_symbol(
                symbol=symbol,
                trading_date=trading_date,
                label_date=label_date,
                return_end_date=return_end_date,
                column=symbol_index[symbol],
                label_row=label_row,
                end_row=end_row,
                panels=panels,
                previous_close=previous_close,
                benchmark_return=benchmark_return,
                in_universe=True,
            )
            if target is None:
                break
            candidate_rows.append(target)
        if len(candidate_rows) != config.top_n:
            report.missing_market_state_dates += 1
            previous_universe = selected
            continue

        carried_status.update(previous_universe - selected)
        carried_status.difference_update(selected)
        status_rows: list[FormalNextOpenTarget] = []
        resolved_status: set[str] = set()
        for symbol in sorted(carried_status):
            target = _target_for_symbol(
                symbol=symbol,
                trading_date=trading_date,
                label_date=label_date,
                return_end_date=return_end_date,
                column=symbol_index[symbol],
                label_row=label_row,
                end_row=end_row,
                panels=panels,
                previous_close=previous_close,
                benchmark_return=benchmark_return,
                in_universe=False,
            )
            if target is None:
                report.missing_market_state_dates += 1
                continue
            status_rows.append(replace(target, label=NEUTRAL))
            if target.can_sell:
                resolved_status.add(symbol)
        carried_status.difference_update(resolved_status)

        labelled_candidates = _rank_labels(
            candidate_rows,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        for target in (*labelled_candidates, *status_rows):
            _record_state(report, target.execution_status)
        targets.extend(labelled_candidates)
        targets.extend(status_rows)
        report.complete_signal_dates += 1
        report.candidate_targets += len(labelled_candidates)
        report.status_only_targets += len(status_rows)
        previous_universe = selected

    if not targets:
        raise ValueError("指定日期没有生成任何正式 open-to-following-open 标签")
    return (
        sorted(targets, key=lambda item: (item.trading_date, not item.in_universe, item.symbol)),
        universe,
        report,
    )
