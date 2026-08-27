"""真实 L2 撮合重建的逐快照区间诊断。

该模块不改变撮合结果，只把 interval correctness 的证据结构化：
订单、撤单、无法按 ID 解析的撤单、模拟成交、真实成交和 L1 误差。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from ticknet.eventstream.config import MARKET_END_MS, RAW_L2_ROOT, day_input_files

from .matching import MatchingEngine
from .pack import SimulatorEvent
from .realdata import (
    _apply_realdata_order,
    default_snapshot_event_lag_ms,
    is_shenzhen_ticker,
    load_day_pack,
)

DiagnosticStatus = Literal["matched", "mismatched", "not_comparable"]


@dataclass(frozen=True)
class DiagnosticSummary:
    total_intervals: int
    matched: int
    mismatched: int
    not_comparable: int
    aggressor_mapping_known: bool
    missing_aggressor_mismatches: int | None
    engine_candidate_mismatches: int | None
    bid_price_mismatches: int
    bid_volume_mismatches: int
    ask_price_mismatches: int
    ask_volume_mismatches: int


@dataclass(frozen=True)
class IntervalDiagnostic:
    start_ms: int
    end_ms: int
    status: DiagnosticStatus
    simulated_bid: tuple[int, int] | None
    expected_bid: tuple[int, int] | None
    simulated_ask: tuple[int, int] | None
    expected_ask: tuple[int, int] | None
    bid_price_match: bool | None
    ask_price_match: bool | None
    bid_volume_error: int | None
    ask_volume_error: int | None
    order_count: int
    order_volume: int
    cancel_count: int
    cancel_volume: int
    unresolved_cancel_count: int
    unresolved_cancel_volume: int
    order_type_counts: dict[int, int]
    simulated_trade_count: int
    simulated_trade_volume: int
    real_trade_count: int
    real_trade_volume: int
    real_trade_unknown_buy_count: int
    real_trade_unknown_sell_count: int
    aggressor_mapping_known: bool
    missing_aggressor_trade_count: int | None
    missing_aggressor_trade_volume: int | None
    missing_passive_trade_count: int | None
    missing_passive_trade_volume: int | None
    snapshot_trade_volume: int
    snapshot_trade_count: int
    trade_volume_vs_snapshot_error: int
    trade_count_vs_snapshot_error: int
    simulated_crossed: bool


def summarize_diagnostics(rows: list[IntervalDiagnostic]) -> DiagnosticSummary:
    """汇总逐区间诊断，并把输入缺口与引擎候选错误分开。"""
    mismatches = [row for row in rows if row.status == "mismatched"]
    aggressor_mapping_known = bool(rows) and all(row.aggressor_mapping_known for row in rows)
    if aggressor_mapping_known:
        missing_aggressor_mismatches = sum(
            (row.missing_aggressor_trade_volume or 0) > 0 for row in mismatches
        )
        engine_candidate_mismatches = len(mismatches) - missing_aggressor_mismatches
    else:
        missing_aggressor_mismatches = None
        engine_candidate_mismatches = None

    return DiagnosticSummary(
        total_intervals=len(rows),
        matched=sum(row.status == "matched" for row in rows),
        mismatched=len(mismatches),
        not_comparable=sum(row.status == "not_comparable" for row in rows),
        aggressor_mapping_known=aggressor_mapping_known,
        missing_aggressor_mismatches=missing_aggressor_mismatches,
        engine_candidate_mismatches=engine_candidate_mismatches,
        bid_price_mismatches=sum(row.bid_price_match is False for row in mismatches),
        bid_volume_mismatches=sum(
            row.bid_price_match is True and row.bid_volume_error not in (None, 0)
            for row in mismatches
        ),
        ask_price_mismatches=sum(row.ask_price_match is False for row in mismatches),
        ask_volume_mismatches=sum(
            row.ask_price_match is True and row.ask_volume_error not in (None, 0)
            for row in mismatches
        ),
    )


def diagnose_day_intervals(
    day: int,
    raw_root: Path = RAW_L2_ROOT,
    ticker: str = "",
    event_lag_ms: int | None = None,
) -> list[IntervalDiagnostic]:
    """逐相邻真实快照重置账本，返回每个区间的结构化诊断。"""
    if event_lag_ms is None:
        event_lag_ms = default_snapshot_event_lag_ms(ticker)
    pack = load_day_pack(day, raw_root, ticker)
    snapshots = [snapshot for snapshot in pack.snapshots if 0 <= snapshot.time_ms < MARKET_END_MS]
    if len(snapshots) < 2:
        return []

    paths = day_input_files(day, Path(raw_root))
    real_trades = _read_trade_events(paths["trades"], ticker)
    snapshot_activity = _read_snapshot_activity(paths["snap"], day, ticker)
    stream_events = [event for event in pack.events if event.kind in ("order", "cancel")]
    known_order_ids = {event.order_id for event in stream_events if event.kind == "order"}

    first_event_time = snapshots[0].time_ms + event_lag_ms
    event_index = _first_after(stream_events, first_event_time)
    trade_index = _first_after(real_trades, first_event_time)
    diagnostics: list[IntervalDiagnostic] = []

    for start, target in pairwise(snapshots):
        target_event_time = target.time_ms + event_lag_ms
        interval_events: list[SimulatorEvent] = []
        while (
            event_index < len(stream_events)
            and stream_events[event_index].time_ms <= target_event_time
        ):
            interval_events.append(stream_events[event_index])
            event_index += 1

        interval_trades: list[SimulatorEvent] = []
        while (
            trade_index < len(real_trades) and real_trades[trade_index].time_ms <= target_event_time
        ):
            interval_trades.append(real_trades[trade_index])
            trade_index += 1

        snapshot_trade_volume, snapshot_trade_count = snapshot_activity.get(target.time_ms, (0, 0))
        diagnostics.append(
            _diagnose_interval(
                start,
                target,
                interval_events,
                interval_trades,
                known_order_ids,
                aggressor_mapping_known=is_shenzhen_ticker(ticker),
                snapshot_trade_volume=snapshot_trade_volume,
                snapshot_trade_count=snapshot_trade_count,
            )
        )

    return diagnostics


def _first_after(events: list[SimulatorEvent], time_ms: int) -> int:
    index = 0
    while index < len(events) and events[index].time_ms <= time_ms:
        index += 1
    return index


def _read_trade_events(path: Path, ticker: str) -> list[SimulatorEvent]:
    if not Path(path).exists():
        raise FileNotFoundError(f"trades parquet 不存在: {path}")
    table = pq.read_table(
        path,
        columns=["ticker", "time_ms", "DealID", "Price", "Volume", "BuyID", "SellID", "Side"],
        filters=[("ticker", "=", ticker)],
    )
    data = table.to_pydict()
    trades = [
        SimulatorEvent(
            time_ms=int(time_ms),
            kind="trade",
            deal_id=str(deal_id),
            buy_id=str(buy_id),
            sell_id=str(sell_id),
            price=round(float(price) * 100),
            volume=int(volume),
            side=int(side),
        )
        for time_ms, deal_id, price, volume, buy_id, sell_id, side in zip(
            data["time_ms"],
            data["DealID"],
            data["Price"],
            data["Volume"],
            data["BuyID"],
            data["SellID"],
            data["Side"],
            strict=True,
        )
    ]
    trades.sort(key=lambda event: (event.time_ms, event.deal_id))
    return trades


def _read_snapshot_activity(path: Path, day: int, ticker: str) -> dict[int, tuple[int, int]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"snapshot parquet 不存在: {path}")
    table = pq.read_table(
        path,
        columns=["ticker", "TradingDay", "time_ms", "Volume", "DealNum"],
        filters=[("ticker", "=", ticker), ("TradingDay", "=", int(day))],
    )
    data = table.to_pydict()
    return {
        int(time_ms): (int(float(volume or 0)), int(float(deal_num or 0)))
        for time_ms, volume, deal_num in zip(
            data["time_ms"], data["Volume"], data["DealNum"], strict=True
        )
    }


def _seed_engine(snapshot: SimulatorEvent) -> MatchingEngine:
    engine = MatchingEngine()
    for index, (price, volume) in enumerate(snapshot.bid_levels or ()):
        engine.lob.seed_level(1, price, volume, f"INIT-B{index}")
    for index, (price, volume) in enumerate(snapshot.ask_levels or ()):
        engine.lob.seed_level(-1, price, volume, f"INIT-A{index}")
    return engine


def _diagnose_interval(
    start: SimulatorEvent,
    target: SimulatorEvent,
    events: list[SimulatorEvent],
    real_trades: list[SimulatorEvent],
    known_order_ids: set[str],
    *,
    aggressor_mapping_known: bool,
    snapshot_trade_volume: int,
    snapshot_trade_count: int,
) -> IntervalDiagnostic:
    order_events = [event for event in events if event.kind == "order"]
    cancel_events = [event for event in events if event.kind == "cancel"]
    order_type_counts = dict(sorted(Counter(event.order_type for event in order_events).items()))

    unresolved_cancel_count = 0
    unresolved_cancel_volume = 0
    simulated_trade_count = 0
    simulated_trade_volume = 0
    simulated_bid: tuple[int, int] | None = None
    simulated_ask: tuple[int, int] | None = None

    baseline_comparable = start.expected_bid is not None and start.expected_ask is not None
    if baseline_comparable:
        engine = _seed_engine(start)
        for event in events:
            if event.kind == "cancel":
                if engine.has_order(event.order_id):
                    engine.cancel_order(event.order_id, event.volume)
                else:
                    unresolved_cancel_count += 1
                    unresolved_cancel_volume += event.volume
                    side = 1 if event.order_type == -1 else -1
                    engine.lob.reduce_level(side, event.price, event.volume)
            elif event.kind == "order":
                trade = _apply_realdata_order(engine, event)
                if trade is not None:
                    simulated_trade_count += 1
                    simulated_trade_volume += trade.volume
        simulated_bid = engine.lob.best_bid()
        simulated_ask = engine.lob.best_ask()

    expected_bid = target.expected_bid
    expected_ask = target.expected_ask
    target_comparable = expected_bid is not None and expected_ask is not None
    if not baseline_comparable or not target_comparable:
        status: DiagnosticStatus = "not_comparable"
    elif simulated_bid == expected_bid and simulated_ask == expected_ask:
        status = "matched"
    else:
        status = "mismatched"

    bid_price_match = _price_match(simulated_bid, expected_bid)
    ask_price_match = _price_match(simulated_ask, expected_ask)
    bid_volume_error = _volume_error(simulated_bid, expected_bid, bid_price_match)
    ask_volume_error = _volume_error(simulated_ask, expected_ask, ask_price_match)

    # 深市 raw Side=0 表示买方主动，Side=1 表示卖方主动；该映射已由双方
    # OrderID 的到达时间反推验证。沪市 ID 关联不足，主动/被动缺失量保持未知。
    missing_aggressor_trades = None
    missing_passive_trades = None
    if aggressor_mapping_known:
        missing_aggressor_trades = [
            event
            for event in real_trades
            if (event.buy_id if event.side == 0 else event.sell_id) not in known_order_ids
        ]
        missing_passive_trades = [
            event
            for event in real_trades
            if (event.sell_id if event.side == 0 else event.buy_id) not in known_order_ids
        ]

    return IntervalDiagnostic(
        start_ms=start.time_ms,
        end_ms=target.time_ms,
        status=status,
        simulated_bid=simulated_bid,
        expected_bid=expected_bid,
        simulated_ask=simulated_ask,
        expected_ask=expected_ask,
        bid_price_match=bid_price_match,
        ask_price_match=ask_price_match,
        bid_volume_error=bid_volume_error,
        ask_volume_error=ask_volume_error,
        order_count=len(order_events),
        order_volume=sum(event.volume for event in order_events),
        cancel_count=len(cancel_events),
        cancel_volume=sum(event.volume for event in cancel_events),
        unresolved_cancel_count=unresolved_cancel_count,
        unresolved_cancel_volume=unresolved_cancel_volume,
        order_type_counts=order_type_counts,
        simulated_trade_count=simulated_trade_count,
        simulated_trade_volume=simulated_trade_volume,
        real_trade_count=len(real_trades),
        real_trade_volume=sum(event.volume for event in real_trades),
        real_trade_unknown_buy_count=sum(
            event.buy_id not in known_order_ids for event in real_trades
        ),
        real_trade_unknown_sell_count=sum(
            event.sell_id not in known_order_ids for event in real_trades
        ),
        aggressor_mapping_known=aggressor_mapping_known,
        missing_aggressor_trade_count=(
            len(missing_aggressor_trades) if missing_aggressor_trades is not None else None
        ),
        missing_aggressor_trade_volume=(
            sum(event.volume for event in missing_aggressor_trades)
            if missing_aggressor_trades is not None
            else None
        ),
        missing_passive_trade_count=(
            len(missing_passive_trades) if missing_passive_trades is not None else None
        ),
        missing_passive_trade_volume=(
            sum(event.volume for event in missing_passive_trades)
            if missing_passive_trades is not None
            else None
        ),
        snapshot_trade_volume=snapshot_trade_volume,
        snapshot_trade_count=snapshot_trade_count,
        trade_volume_vs_snapshot_error=(
            sum(event.volume for event in real_trades) - snapshot_trade_volume
        ),
        trade_count_vs_snapshot_error=len(real_trades) - snapshot_trade_count,
        simulated_crossed=(
            simulated_bid is not None
            and simulated_ask is not None
            and simulated_bid[0] >= simulated_ask[0]
        ),
    )


def _price_match(
    simulated: tuple[int, int] | None,
    expected: tuple[int, int] | None,
) -> bool | None:
    if expected is None:
        return None
    if simulated is None:
        return False
    return simulated[0] == expected[0]


def _volume_error(
    simulated: tuple[int, int] | None,
    expected: tuple[int, int] | None,
    price_match: bool | None,
) -> int | None:
    if simulated is None or expected is None or price_match is not True:
        return None
    return simulated[1] - expected[1]
