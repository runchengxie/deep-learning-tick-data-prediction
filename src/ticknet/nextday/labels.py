"""次日收益目标和横截面三分类标签。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from itertools import pairwise

import numpy as np

DOWN = 0
NEUTRAL = 1
UP = 2
LABEL_METHODS = {"fixed", "cross_sectional"}


@dataclass(frozen=True)
class DailyBar:
    """生成次日开盘到收盘收益所需的日线。"""

    symbol: str
    trading_date: date
    open: float
    close: float


@dataclass(frozen=True)
class NextDayTarget:
    """输入交易日对应的下一交易日监督目标。"""

    symbol: str
    trading_date: date
    label_date: date
    raw_return: float
    target_return: float
    label: int = NEUTRAL


def _validate_bars(bars: Iterable[DailyBar]) -> dict[tuple[str, date], DailyBar]:
    indexed: dict[tuple[str, date], DailyBar] = {}
    for bar in bars:
        if not bar.symbol:
            raise ValueError("symbol 不能为空")
        if not math.isfinite(bar.open) or not math.isfinite(bar.close):
            raise ValueError(f"{bar.symbol} {bar.trading_date} 的开收盘价不是有限值")
        if bar.open <= 0 or bar.close <= 0:
            raise ValueError(f"{bar.symbol} {bar.trading_date} 的开收盘价必须为正数")
        key = (bar.symbol, bar.trading_date)
        if key in indexed:
            raise ValueError(f"日线存在重复键：{bar.symbol} {bar.trading_date}")
        indexed[key] = bar
    if not indexed:
        raise ValueError("日线数据不能为空")
    return indexed


def _validate_calendar(calendar: Sequence[date]) -> list[date]:
    dates = list(calendar)
    if not dates:
        raise ValueError("交易日历不能为空")
    if dates != sorted(set(dates)):
        raise ValueError("交易日历必须严格递增且不能重复")
    return dates


def _fixed_label(target_return: float, neutral_threshold: float) -> int:
    if target_return > neutral_threshold:
        return UP
    if target_return < -neutral_threshold:
        return DOWN
    return NEUTRAL


def _rank_labels(
    targets: list[NextDayTarget],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> list[NextDayTarget]:
    """按收益分位点赋方向标签，切点处的并列收益留在中性组。"""
    returns = np.asarray([target.target_return for target in targets], dtype=np.float64)
    lower_cut = float(np.quantile(returns, lower_quantile))
    upper_cut = float(np.quantile(returns, upper_quantile))
    labelled: list[NextDayTarget] = []
    for target in targets:
        if target.target_return < lower_cut:
            label = DOWN
        elif target.target_return > upper_cut:
            label = UP
        else:
            label = NEUTRAL
        labelled.append(replace(target, label=label))
    return labelled


def build_next_day_targets(
    bars: Iterable[DailyBar],
    calendar: Sequence[date],
    *,
    label_method: str = "cross_sectional",
    neutral_threshold: float = 0.002,
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    benchmark_returns: Mapping[date, float] | None = None,
    min_cross_section: int = 20,
    universe: Mapping[date, Iterable[str]] | None = None,
) -> list[NextDayTarget]:
    """把当日样本对齐到下一交易日的开盘到收盘收益。

    ``calendar`` 决定真正的下一交易日。某只股票在当前日或下一交易日缺少日线时，
    对应样本不会生成，避免把停牌后的首个交易日误当成紧邻标签。传入
    ``benchmark_returns`` 时，分类和评估使用扣除基准后的超额收益。``universe``
    可为每个输入日指定历史时点可得的股票池；未入选股票不会生成该日标签。
    """
    if label_method not in LABEL_METHODS:
        raise ValueError(f"label_method 应为 {sorted(LABEL_METHODS)} 中的一个")
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold 不能为负数")
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("横截面分位点必须满足 0 < lower < upper < 1")
    if min_cross_section < 2:
        raise ValueError("min_cross_section 至少为 2")

    indexed = _validate_bars(bars)
    dates = _validate_calendar(calendar)
    symbols = sorted({symbol for symbol, _ in indexed})
    dated_universe = (
        {
            trading_date: {str(symbol) for symbol in selected if str(symbol)}
            for trading_date, selected in universe.items()
        }
        if universe is not None
        else None
    )
    by_label_date: dict[date, list[NextDayTarget]] = defaultdict(list)

    for trading_date, label_date in pairwise(dates):
        selected_symbols = (
            sorted(dated_universe.get(trading_date, set()))
            if dated_universe is not None
            else symbols
        )
        if not selected_symbols:
            continue
        if benchmark_returns is not None:
            if label_date not in benchmark_returns:
                raise ValueError(f"基准收益缺少交易日 {label_date}")
            benchmark_return = float(benchmark_returns[label_date])
            if not math.isfinite(benchmark_return):
                raise ValueError(f"{label_date} 的基准收益不是有限值")
        else:
            benchmark_return = 0.0

        for symbol in selected_symbols:
            current = indexed.get((symbol, trading_date))
            following = indexed.get((symbol, label_date))
            if following is None or (dated_universe is None and current is None):
                continue
            raw_return = following.close / following.open - 1.0
            target_return = raw_return - benchmark_return
            by_label_date[label_date].append(
                NextDayTarget(
                    symbol=symbol,
                    trading_date=trading_date,
                    label_date=label_date,
                    raw_return=raw_return,
                    target_return=target_return,
                )
            )

    labelled: list[NextDayTarget] = []
    for label_date in sorted(by_label_date):
        targets = by_label_date[label_date]
        if label_method == "cross_sectional":
            if len(targets) < min_cross_section:
                continue
            labelled.extend(
                _rank_labels(
                    targets,
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                )
            )
        else:
            labelled.extend(
                replace(
                    target,
                    label=_fixed_label(target.target_return, neutral_threshold),
                )
                for target in targets
            )

    return sorted(labelled, key=lambda item: (item.trading_date, item.symbol))
