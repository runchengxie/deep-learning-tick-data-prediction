"""snapshot 标签与目标构建：基准收益、次日标签与动态股票池编排。

从 ``raw_snapshot.py`` 拆出。依赖 ``snapshot_config``（配置/面板）与
``snapshot_features``（股票池），并对接 ``labels`` 模块生成次日方向标签。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import pairwise
from pathlib import Path

import pyarrow.parquet as pq

from ticknet.nextday.labels import DailyBar, NextDayTarget, build_next_day_targets
from ticknet.nextday.snapshot_config import (
    DailyPanel,
    SnapshotPreparationConfig,
    _yyyymmdd,
)
from ticknet.nextday.snapshot_features import build_dynamic_universe
from ticknet.nextday.splits import parse_date


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
