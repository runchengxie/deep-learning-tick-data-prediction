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
Level = tuple[int, int]


@dataclass(frozen=True)
class OpeningOrder:
    """一笔盘前有效委托，价格单位为分，数量单位为股。"""

    order_id: str
    side: int
    price: int
    volume: int


@dataclass(frozen=True)
class OpeningTrade:
    """一笔盘前成交及其买卖双方订单身份。"""

    buy_id: str
    sell_id: str
    volume: int


@dataclass(frozen=True)
class OpeningCancel:
    """一笔盘前撤单及其被撤订单身份。"""

    order_id: str
    volume: int


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
class OpeningDayAudit:
    """单只股票交易日的盘前账本审计及其快照定位。"""

    day: int
    ticker: str
    snapshot_time_ms: int | None
    event_cutoff_time_ms: int | None
    preopen_file_present: bool
    preopen_ticker_present: bool
    audit: OpeningLedgerAudit


@dataclass(frozen=True)
class OpeningAuditSummary:
    """多个股票日审计结果的汇总。"""

    total_samples: int
    matched: int
    mismatched: int
    not_comparable: int
    identity_gap_samples: int
    comparable_match_rate: float | None


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
    if event_lag_ms is None:
        event_lag_ms = default_snapshot_event_lag_ms(ticker)
    event_cutoff = snapshot.time_ms + event_lag_ms if snapshot else None
    orders = [
        OpeningOrder(event.order_id, event.side, event.price, event.volume)
        for event in all_orders
        if event.kind == "order" and event_cutoff is not None and event.time_ms <= event_cutoff
    ]
    cancels = [
        OpeningCancel(event.order_id, event.volume)
        for event in all_orders
        if event.kind == "cancel" and event_cutoff is not None and event.time_ms <= event_cutoff
    ]
    trades = [
        OpeningTrade(event.buy_id, event.sell_id, event.volume)
        for event in _read_trade_events(paths["trades"], ticker)
        if event_cutoff is not None and event.time_ms <= event_cutoff
    ]
    audit = audit_opening_ledger(
        orders,
        trades,
        cancels,
        expected_bid_levels=snapshot.bid_levels if snapshot else None,
        expected_ask_levels=snapshot.ask_levels if snapshot else None,
    )
    preopen_ticker_present = bool(preopen_events)
    if not preopen_ticker_present:
        audit = replace(audit, status="not_comparable")
    return OpeningDayAudit(
        day=int(day),
        ticker=ticker,
        snapshot_time_ms=snapshot.time_ms if snapshot else None,
        event_cutoff_time_ms=event_cutoff,
        preopen_file_present=preopen_path.exists(),
        preopen_ticker_present=preopen_ticker_present,
        audit=audit,
    )


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
