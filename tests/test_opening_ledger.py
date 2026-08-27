from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.simulator.opening_ledger import (
    OpeningCancel,
    OpeningDayAudit,
    OpeningLedgerAudit,
    OpeningOrder,
    OpeningTrade,
    audit_opening_ledger,
    choose_best_lag,
    summarize_opening_audits,
    trace_opening_level,
)


def test_audit_opening_ledger_reconstructs_top_levels_after_auction_activity():
    orders = [
        OpeningOrder("B1", 1, 1000, 500),
        OpeningOrder("B2", 1, 995, 700),
        OpeningOrder("A1", -1, 1010, 600),
    ]
    trades = [OpeningTrade("B1", "A1", 200)]
    cancels = [OpeningCancel("B2", 100)]

    result = audit_opening_ledger(
        orders,
        trades,
        cancels,
        expected_bid_levels=((1000, 300), (995, 600)),
        expected_ask_levels=((1010, 400),),
    )

    assert result.status == "matched"
    assert result.bid_levels == ((1000, 300), (995, 600))
    assert result.ask_levels == ((1010, 400),)
    assert result.unknown_trade_count == 0
    assert result.unknown_cancel_count == 0
    assert result.overdrawn_count == 0


def test_audit_opening_ledger_reports_identity_gaps_and_overdrawn_activity():
    orders = [OpeningOrder("B1", 1, 1000, 100)]

    result = audit_opening_ledger(
        orders,
        trades=[OpeningTrade("UNKNOWN-B", "UNKNOWN-A", 30)],
        cancels=[OpeningCancel("B1", 120), OpeningCancel("UNKNOWN-C", 10)],
        expected_bid_levels=((1000, 0),),
        expected_ask_levels=(),
    )

    assert result.status == "mismatched"
    assert result.unknown_trade_count == 1
    assert result.unknown_trade_volume == 30
    assert result.unknown_cancel_count == 1
    assert result.unknown_cancel_volume == 10
    assert result.overdrawn_count == 1
    assert result.overdrawn_volume == 20


def test_audit_opening_ledger_marks_missing_snapshot_not_comparable():
    result = audit_opening_ledger(
        [OpeningOrder("B1", 1, 1000, 100)],
        [],
        [],
        expected_bid_levels=None,
        expected_ask_levels=((1010, 100),),
    )

    assert result.status == "not_comparable"


def test_audit_opening_day_reads_preopen_orders_trades_and_first_snapshot(tmp_path: Path):
    day = 20210104
    ticker = "000001"
    order_columns = [
        "ticker",
        "TradingDay",
        "time_ms",
        "OrderID",
        "Price",
        "Volume",
        "OrderType",
    ]
    preopen_rows = [
        {
            "ticker": ticker,
            "TradingDay": day,
            "time_ms": -100,
            "OrderID": "B1",
            "Price": 1000,
            "Volume": 100,
            "OrderType": 2,
        },
        {
            "ticker": ticker,
            "TradingDay": day,
            "time_ms": -90,
            "OrderID": "A1",
            "Price": 1010,
            "Volume": 80,
            "OrderType": 12,
        },
    ]
    trade_columns = [
        "ticker",
        "TradingDay",
        "time_ms",
        "DealID",
        "Price",
        "Volume",
        "Side",
        "bsflag",
        "BuyID",
        "SellID",
    ]
    trade_rows = [
        {
            "ticker": ticker,
            "TradingDay": day,
            "time_ms": -50,
            "DealID": "T1",
            "Price": 1000,
            "Volume": 40,
            "Side": 1,
            "bsflag": 1,
            "BuyID": "B1",
            "SellID": "A1",
        }
    ]
    snapshot_columns = [
        "ticker",
        "TradingDay",
        "time_ms",
        *[f"BidPrice{i}" for i in range(1, 11)],
        *[f"BidVolume{i}" for i in range(1, 11)],
        *[f"AskPrice{i}" for i in range(1, 11)],
        *[f"AskVolume{i}" for i in range(1, 11)],
    ]
    snapshot = {
        "ticker": ticker,
        "TradingDay": day,
        "time_ms": 0,
        **{f"BidPrice{i}": (1000 if i == 1 else None) for i in range(1, 11)},
        **{f"BidVolume{i}": (60 if i == 1 else 0) for i in range(1, 11)},
        **{f"AskPrice{i}": (1010 if i == 1 else None) for i in range(1, 11)},
        **{f"AskVolume{i}": (40 if i == 1 else 0) for i in range(1, 11)},
    }
    order_path = tmp_path / "order_preopen" / "202101" / "order_2021-01-04.parquet"
    regular_order_path = tmp_path / "order" / "202101" / "order_2021-01-04.parquet"
    trades_path = tmp_path / "trades" / "202101" / "trades_2021-01-04.parquet"
    snapshot_path = tmp_path / "snapshot" / "snapshot_202101.parquet"
    order_path.parent.mkdir(parents=True)
    regular_order_path.parent.mkdir(parents=True)
    trades_path.parent.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({c: [row[c] for row in preopen_rows] for c in order_columns}), order_path
    )
    regular_row = {**preopen_rows[0], "time_ms": 200, "OrderID": "LATE"}
    pq.write_table(pa.table({c: [regular_row[c]] for c in order_columns}), regular_order_path)
    pq.write_table(
        pa.table({c: [row[c] for row in trade_rows] for c in trade_columns}), trades_path
    )
    pq.write_table(pa.table({c: [snapshot[c]] for c in snapshot_columns}), snapshot_path)

    from ticknet.simulator.opening_ledger import audit_opening_day

    result = audit_opening_day(day, ticker, tmp_path)

    assert result.snapshot_time_ms == 0
    assert result.event_cutoff_time_ms == 140
    assert result.preopen_file_present is True
    assert result.audit.status == "matched"
    assert result.audit.bid_levels == ((1000, 60),)
    assert result.audit.ask_levels == ((1010, 40),)


def test_summarize_opening_audits_excludes_unavailable_samples_from_rate():
    matched = audit_opening_ledger([], [], [], expected_bid_levels=(), expected_ask_levels=())
    mismatched = audit_opening_ledger(
        [OpeningOrder("B", 1, 1000, 10)],
        [],
        [],
        expected_bid_levels=((999, 10),),
        expected_ask_levels=(),
    )
    unavailable = audit_opening_ledger(
        [], [], [], expected_bid_levels=None, expected_ask_levels=None
    )

    summary = summarize_opening_audits(
        [
            OpeningDayAudit(20210104, "000001", 0, 140, True, True, matched),
            OpeningDayAudit(20210105, "000001", 0, 140, True, True, mismatched),
            OpeningDayAudit(20210106, "000001", None, None, False, False, unavailable),
        ]
    )

    assert summary.total_samples == 3
    assert summary.matched == 1
    assert summary.mismatched == 1
    assert summary.not_comparable == 1
    assert summary.comparable_match_rate == 0.5


def test_summarize_opening_audits_counts_identity_gap_samples_once():
    audit = audit_opening_ledger(
        [],
        [OpeningTrade("missing-buy", "missing-sell", 100)],
        [],
        expected_bid_levels=(),
        expected_ask_levels=(),
    )

    summary = summarize_opening_audits(
        [
            OpeningDayAudit(20210104, "000001", 0, 140, True, True, audit),
        ]
    )

    assert summary.identity_gap_samples == 1


def test_choose_best_lag_prefers_exact_match_then_smallest_absolute_lag():
    empty = OpeningLedgerAudit(
        status="mismatched",
        bid_levels=((1000, 90),),
        ask_levels=((1010, 100),),
        expected_bid_levels=((1000, 100),),
        expected_ask_levels=((1010, 100),),
        unknown_trade_count=0,
        unknown_trade_volume=0,
        unknown_cancel_count=0,
        unknown_cancel_volume=0,
        overdrawn_count=0,
        overdrawn_volume=0,
    )
    exact = audit_opening_ledger([], [], [], expected_bid_levels=(), expected_ask_levels=())

    selected = choose_best_lag([(140, empty), (-10, exact), (10, exact)])

    assert selected.lag_ms == -10
    assert selected.audit.status == "matched"


def test_trace_opening_level_explains_order_residuals():
    rows = trace_opening_level(
        [
            OpeningOrder("B1", 1, 777, 10000),
            OpeningOrder("B2", 1, 777, 5000),
        ],
        [OpeningTrade("B1", "A1", 4000)],
        [OpeningCancel("B2", 2100)],
        side=1,
        price=777,
    )

    assert [
        (row.order_id, row.traded_volume, row.cancelled_volume, row.remaining_volume)
        for row in rows
    ] == [("B1", 4000, 0, 6000), ("B2", 0, 2100, 2900)]
