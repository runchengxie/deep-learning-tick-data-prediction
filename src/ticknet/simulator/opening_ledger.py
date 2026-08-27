"""用盘前逐笔数据重建并审计开盘前的订单级盘口账本。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ticknet.eventstream.config import RAW_L2_ROOT, day_input_files, day_preopen_file

from .diagnostics import _read_trade_events
from .realdata import (
    _read_order_events,
    _read_snapshot_events,
    default_snapshot_event_lag_ms,
)

OpeningStatus = Literal["matched", "mismatched", "not_comparable"]
CoverageStatus = Literal[
    "covered", "preopen_file_missing", "preopen_ticker_missing", "opening_trade_gap"
]
Level = tuple[int, int]


@dataclass(frozen=True)
class OpeningOrder:
    """一笔盘前有效委托，价格单位为分，数量单位为股。"""

    order_id: str
    side: int
    price: int
    volume: int
    time_ms: int = 0


@dataclass(frozen=True)
class OpeningTrade:
    """一笔盘前成交及其买卖双方订单身份。"""

    buy_id: str
    sell_id: str
    volume: int
    time_ms: int = 0


@dataclass(frozen=True)
class OpeningCancel:
    """一笔盘前撤单及其被撤订单身份。"""

    order_id: str
    volume: int
    time_ms: int = 0


@dataclass(frozen=True)
class OpeningLedgerAudit:
    """盘前账本与首张完整快照的比较结果。"""

    status: OpeningStatus
    bid_levels: tuple[Level, ...]
    ask_levels: tuple[Level, ...]
    expected_bid_levels: tuple[Level, ...] | None
    expected_ask_levels: tuple[Level, ...] | None
    unknown_trade_count: int
    unknown_trade_volume: int
    unknown_cancel_count: int
    unknown_cancel_volume: int
    overdrawn_count: int
    overdrawn_volume: int


@dataclass(frozen=True)
class OpeningLagAudit:
    """一个候选事件时差对应的账本结果。"""

    lag_ms: int
    audit: OpeningLedgerAudit


@dataclass(frozen=True)
class OpeningLevelTrace:
    """指定价格档的订单级数量守恒明细。"""

    order_id: str
    side: int
    price: int
    original_volume: int
    traded_volume: int
    cancelled_volume: int
    remaining_volume: int


@dataclass(frozen=True)
class OpeningLevelDifference:
    """实际账本和快照在某一档的数量与价格差异。"""

    side: int
    rank: int
    actual: Level | None
    expected: Level | None
    volume_delta: int | None


def level_differences(
    actual: Sequence[Level],
    expected: Sequence[Level] | None,
    *,
    side: int,
) -> tuple[OpeningLevelDifference, ...]:
    """返回前十档逐档差异，``volume_delta`` 为实际减期望。"""
    if expected is None:
        return ()
    differences: list[OpeningLevelDifference] = []
    for rank in range(max(len(actual), len(expected))):
        actual_level = actual[rank] if rank < len(actual) else None
        expected_level = expected[rank] if rank < len(expected) else None
        if actual_level != expected_level:
            actual_volume = actual_level[1] if actual_level else 0
            expected_volume = expected_level[1] if expected_level else 0
            differences.append(
                OpeningLevelDifference(
                    side=side,
                    rank=rank + 1,
                    actual=actual_level,
                    expected=expected_level,
                    volume_delta=actual_volume - expected_volume,
                )
            )
    return tuple(differences)


def choose_best_lag(candidates: Sequence[tuple[int, OpeningLedgerAudit]]) -> OpeningLagAudit:
    """按匹配状态、身份缺口、盘口差异和绝对时差选择最佳候选。"""
    if not candidates:
        raise ValueError("至少需要一个候选 lag")

    def score(candidate: tuple[int, OpeningLedgerAudit]) -> tuple[int, int, int, int, int]:
        lag, audit = candidate
        status_rank = {"matched": 0, "mismatched": 1, "not_comparable": 2}[audit.status]
        identity_gap = (
            audit.unknown_trade_volume + audit.unknown_cancel_volume + audit.overdrawn_volume
        )
        level_gap = _level_gap(audit.bid_levels, audit.expected_bid_levels) + _level_gap(
            audit.ask_levels, audit.expected_ask_levels
        )
        return status_rank, identity_gap, level_gap, abs(lag), lag

    lag, audit = min(candidates, key=score)
    return OpeningLagAudit(lag_ms=lag, audit=audit)


def trace_opening_level(
    orders: Sequence[OpeningOrder],
    trades: Sequence[OpeningTrade],
    cancels: Sequence[OpeningCancel],
    *,
    side: int,
    price: int,
) -> tuple[OpeningLevelTrace, ...]:
    """返回指定价位的订单级原量、成交量、撤单量和剩余量。"""
    original = {
        order.order_id: [order.side, order.price, max(order.volume, 0), 0, 0] for order in orders
    }
    for trade in trades:
        for order_id in (trade.buy_id, trade.sell_id):
            if order_id in original:
                original[order_id][3] += max(trade.volume, 0)
    for cancel in cancels:
        if cancel.order_id in original:
            original[cancel.order_id][4] += max(cancel.volume, 0)
    return tuple(
        OpeningLevelTrace(
            order_id=order_id,
            side=data[0],
            price=data[1],
            original_volume=data[2],
            traded_volume=data[3],
            cancelled_volume=data[4],
            remaining_volume=max(data[2] - data[3] - data[4], 0),
        )
        for order_id, data in original.items()
        if data[0] == side and data[1] == price
    )


@dataclass(frozen=True)
class OpeningDayAudit:
    """单只股票交易日的盘前账本审计及其快照定位。"""

    day: int
    ticker: str
    snapshot_time_ms: int | None
    event_cutoff_time_ms: int | None
    preopen_file_present: bool
    preopen_ticker_present: bool
    audit: OpeningLedgerAudit
    coverage_status: CoverageStatus = "covered"


@dataclass(frozen=True)
class OpeningAuditSummary:
    """多个股票日审计结果的汇总。"""

    total_samples: int
    matched: int
    mismatched: int
    not_comparable: int
    identity_gap_samples: int
    comparable_match_rate: float | None


@dataclass(frozen=True)
class OpeningDayInputs:
    """单个股票日的已读取事件和首张完整快照。"""

    day: int
    ticker: str
    orders: tuple[OpeningOrder, ...]
    trades: tuple[OpeningTrade, ...]
    cancels: tuple[OpeningCancel, ...]
    snapshot_time_ms: int | None
    expected_bid_levels: tuple[Level, ...] | None
    expected_ask_levels: tuple[Level, ...] | None
    preopen_file_present: bool
    preopen_ticker_present: bool


@dataclass(frozen=True)
class OpeningDayLagScan:
    """一个股票日的候选 lag 结果和最佳选择。"""

    day: int
    ticker: str
    candidates: tuple[OpeningLagAudit, ...]
    best: OpeningLagAudit
    snapshot_time_ms: int | None
    coverage_status: CoverageStatus
    preopen_file_present: bool
    preopen_ticker_present: bool


def summarize_opening_audits(results: Sequence[OpeningDayAudit]) -> OpeningAuditSummary:
    """汇总样本，并只用可比较样本计算精确匹配率。"""
    matched = sum(result.audit.status == "matched" for result in results)
    mismatched = sum(result.audit.status == "mismatched" for result in results)
    not_comparable = sum(result.audit.status == "not_comparable" for result in results)
    identity_gap_samples = sum(
        1
        for result in results
        if result.audit.unknown_trade_count
        or result.audit.unknown_cancel_count
        or result.audit.overdrawn_count
    )
    comparable = matched + mismatched
    return OpeningAuditSummary(
        total_samples=len(results),
        matched=matched,
        mismatched=mismatched,
        not_comparable=not_comparable,
        identity_gap_samples=identity_gap_samples,
        comparable_match_rate=matched / comparable if comparable else None,
    )


def audit_opening_day(
    day: int,
    ticker: str,
    raw_root: Path = RAW_L2_ROOT,
    event_lag_ms: int | None = None,
) -> OpeningDayAudit:
    """读取一个股票日，并审计盘前账本对首张完整快照的重建。"""
    inputs = load_opening_day_inputs(day, ticker, raw_root)
    lag = default_snapshot_event_lag_ms(ticker) if event_lag_ms is None else event_lag_ms
    return _audit_loaded_opening_day(inputs, lag)


def scan_opening_day_lags(
    day: int,
    ticker: str,
    raw_root: Path = RAW_L2_ROOT,
    lags: Sequence[int] = tuple(range(-200, 201, 10)),
) -> OpeningDayLagScan:
    """一次读取股票日数据，并在内存中扫描多个事件时差候选。"""
    if not lags:
        raise ValueError("至少需要一个候选 lag")
    inputs = load_opening_day_inputs(day, ticker, raw_root)
    candidates = tuple(
        OpeningLagAudit(lag_ms=lag, audit=_audit_loaded_opening_day(inputs, lag).audit)
        for lag in lags
    )
    best = choose_best_lag([(candidate.lag_ms, candidate.audit) for candidate in candidates])
    cutoff = inputs.snapshot_time_ms + best.lag_ms if inputs.snapshot_time_ms is not None else None
    return OpeningDayLagScan(
        day=int(day),
        ticker=ticker,
        candidates=candidates,
        best=best,
        snapshot_time_ms=inputs.snapshot_time_ms,
        coverage_status=_coverage_status(inputs, cutoff, best.audit),
        preopen_file_present=inputs.preopen_file_present,
        preopen_ticker_present=inputs.preopen_ticker_present,
    )


def load_opening_day_inputs(
    day: int,
    ticker: str,
    raw_root: Path = RAW_L2_ROOT,
) -> OpeningDayInputs:
    """读取一个股票日的订单、成交、撤单和首张完整快照。"""
    paths = day_input_files(day, Path(raw_root))
    preopen_path = day_preopen_file(day, Path(raw_root))
    preopen_events = _read_order_events(preopen_path, ticker) if preopen_path.exists() else []
    regular_events = _read_order_events(paths["order"], ticker)
    all_orders = [*preopen_events, *regular_events]
    snapshots = [
        event
        for event in _read_snapshot_events(paths["snap"], day, ticker)
        if event.time_ms >= 0 and event.bid_levels is not None and event.ask_levels is not None
    ]
    snapshot = snapshots[0] if snapshots else None
    return OpeningDayInputs(
        day=int(day),
        ticker=ticker,
        orders=tuple(
            OpeningOrder(event.order_id, event.side, event.price, event.volume, event.time_ms)
            for event in all_orders
            if event.kind == "order"
        ),
        cancels=tuple(
            OpeningCancel(event.order_id, event.volume, event.time_ms)
            for event in all_orders
            if event.kind == "cancel"
        ),
        trades=tuple(
            OpeningTrade(event.buy_id, event.sell_id, event.volume, event.time_ms)
            for event in _read_trade_events(paths["trades"], ticker)
        ),
        snapshot_time_ms=snapshot.time_ms if snapshot else None,
        expected_bid_levels=snapshot.bid_levels if snapshot else None,
        expected_ask_levels=snapshot.ask_levels if snapshot else None,
        preopen_file_present=preopen_path.exists(),
        preopen_ticker_present=bool(preopen_events),
    )


def trace_opening_day_level(
    day: int,
    ticker: str,
    *,
    side: int,
    price: int,
    raw_root: Path = RAW_L2_ROOT,
    event_lag_ms: int | None = None,
) -> tuple[OpeningLevelTrace, ...]:
    """追踪一个股票日某价位在最佳或指定事件截止点的订单明细。"""
    inputs = load_opening_day_inputs(day, ticker, raw_root)
    lag = default_snapshot_event_lag_ms(ticker) if event_lag_ms is None else event_lag_ms
    cutoff = inputs.snapshot_time_ms + lag if inputs.snapshot_time_ms is not None else None
    if cutoff is None:
        return ()
    orders = tuple(order for order in inputs.orders if order.time_ms <= cutoff)
    trades = tuple(trade for trade in inputs.trades if trade.time_ms <= cutoff)
    cancels = tuple(cancel for cancel in inputs.cancels if cancel.time_ms <= cutoff)
    return trace_opening_level(orders, trades, cancels, side=side, price=price)


def _audit_loaded_opening_day(inputs: OpeningDayInputs, event_lag_ms: int) -> OpeningDayAudit:
    event_cutoff = (
        inputs.snapshot_time_ms + event_lag_ms if inputs.snapshot_time_ms is not None else None
    )
    orders = (
        tuple(
            order
            for order in inputs.orders
            if event_cutoff is not None and order.time_ms <= event_cutoff
        )
        if event_cutoff is not None
        else ()
    )
    cancels = (
        tuple(
            cancel
            for cancel in inputs.cancels
            if event_cutoff is not None and cancel.time_ms <= event_cutoff
        )
        if event_cutoff is not None
        else ()
    )
    trades = (
        tuple(
            trade
            for trade in inputs.trades
            if event_cutoff is not None and trade.time_ms <= event_cutoff
        )
        if event_cutoff is not None
        else ()
    )
    audit = audit_opening_ledger(
        orders,
        trades,
        cancels,
        expected_bid_levels=inputs.expected_bid_levels,
        expected_ask_levels=inputs.expected_ask_levels,
    )
    coverage_status = _coverage_status(inputs, event_cutoff, audit)
    if coverage_status != "covered":
        audit = replace(audit, status="not_comparable")
    return OpeningDayAudit(
        day=inputs.day,
        ticker=inputs.ticker,
        snapshot_time_ms=inputs.snapshot_time_ms,
        event_cutoff_time_ms=event_cutoff,
        preopen_file_present=inputs.preopen_file_present,
        preopen_ticker_present=inputs.preopen_ticker_present,
        audit=audit,
        coverage_status=coverage_status,
    )


def _coverage_status(
    inputs: OpeningDayInputs,
    event_cutoff: int | None,
    audit: OpeningLedgerAudit,
) -> CoverageStatus:
    if not inputs.preopen_file_present:
        return "preopen_file_missing"
    if not inputs.preopen_ticker_present:
        has_opening_trade = any(
            event_cutoff is not None and trade.time_ms <= event_cutoff for trade in inputs.trades
        )
        return "opening_trade_gap" if has_opening_trade else "preopen_ticker_missing"
    if audit.unknown_trade_count:
        return "opening_trade_gap"
    return "covered"


def audit_opening_ledger(
    orders: Sequence[OpeningOrder],
    trades: Sequence[OpeningTrade],
    cancels: Sequence[OpeningCancel],
    *,
    expected_bid_levels: Sequence[Level] | None,
    expected_ask_levels: Sequence[Level] | None,
    depth: int = 10,
) -> OpeningLedgerAudit:
    """按订单身份扣减盘前成交和撤单，并比较重建的前 ``depth`` 档。

    订单身份或数量出现问题时会保留账本的可计算部分，同时将结果标记为
    ``mismatched``，避免把不完整的身份链误报成精确重建。
    """
    remaining = {
        order.order_id: [order.side, order.price, max(order.volume, 0)] for order in orders
    }
    unknown_trade_count = 0
    unknown_trade_volume = 0
    unknown_cancel_count = 0
    unknown_cancel_volume = 0
    overdrawn_count = 0
    overdrawn_volume = 0

    def consume(order_id: str, volume: int) -> tuple[bool, int]:
        nonlocal overdrawn_count, overdrawn_volume
        order = remaining.get(order_id)
        if order is None:
            return False, 0
        requested = max(volume, 0)
        consumed = min(order[2], requested)
        excess = requested - consumed
        order[2] -= consumed
        if excess:
            overdrawn_count += 1
            overdrawn_volume += excess
        return True, consumed

    for trade in trades:
        buy_known, _ = consume(trade.buy_id, trade.volume)
        sell_known, _ = consume(trade.sell_id, trade.volume)
        if not buy_known or not sell_known:
            unknown_trade_count += 1
            unknown_trade_volume += max(trade.volume, 0)

    for cancel in cancels:
        known, _ = consume(cancel.order_id, cancel.volume)
        if not known:
            unknown_cancel_count += 1
            unknown_cancel_volume += max(cancel.volume, 0)

    bids = _top_levels(remaining, side=1, depth=depth)
    asks = _top_levels(remaining, side=-1, depth=depth)
    expected_bids = _normalise_levels(expected_bid_levels, depth)
    expected_asks = _normalise_levels(expected_ask_levels, depth)
    comparable = expected_bids is not None and expected_asks is not None
    has_identity_gap = unknown_trade_count or unknown_cancel_count or overdrawn_count
    status: OpeningStatus
    if not comparable:
        status = "not_comparable"
    elif has_identity_gap or bids != expected_bids or asks != expected_asks:
        status = "mismatched"
    else:
        status = "matched"

    return OpeningLedgerAudit(
        status=status,
        bid_levels=bids,
        ask_levels=asks,
        expected_bid_levels=expected_bids,
        expected_ask_levels=expected_asks,
        unknown_trade_count=unknown_trade_count,
        unknown_trade_volume=unknown_trade_volume,
        unknown_cancel_count=unknown_cancel_count,
        unknown_cancel_volume=unknown_cancel_volume,
        overdrawn_count=overdrawn_count,
        overdrawn_volume=overdrawn_volume,
    )


def _top_levels(remaining: dict[str, list[int]], *, side: int, depth: int) -> tuple[Level, ...]:
    levels: dict[int, int] = {}
    for order_side, price, volume in remaining.values():
        if order_side == side and volume > 0:
            levels[price] = levels.get(price, 0) + volume
    prices = sorted(levels, reverse=side == 1)
    return tuple((price, levels[price]) for price in prices[:depth])


def _normalise_levels(levels: Sequence[Level] | None, depth: int) -> tuple[Level, ...] | None:
    if levels is None:
        return None
    return tuple((int(price), int(volume)) for price, volume in levels[:depth] if volume > 0)


def _level_gap(actual: Sequence[Level], expected: Sequence[Level] | None) -> int:
    if expected is None:
        return 10**12
    actual_map = dict(actual)
    expected_map = dict(expected)
    prices = actual_map.keys() | expected_map.keys()
    return sum(abs(actual_map.get(price, 0) - expected_map.get(price, 0)) for price in prices)
