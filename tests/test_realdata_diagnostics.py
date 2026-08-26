from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.eventstream.config import MARKET_END_MS
from ticknet.simulator.diagnostics import diagnose_day_intervals, summarize_diagnostics

DAY = 20210104
TICKER = "000001"

ORDER_COLS = ["ticker", "TradingDay", "time_ms", "OrderID", "Price", "Volume", "OrderType"]
TRADE_COLS = [
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
SNAP_COLS = (
    ["ticker", "TradingDay", "time_ms", "Volume", "DealNum"]
    + [f"BidPrice{i}" for i in range(1, 11)]
    + [f"BidVolume{i}" for i in range(1, 11)]
    + [f"AskPrice{i}" for i in range(1, 11)]
    + [f"AskVolume{i}" for i in range(1, 11)]
)


def _table(rows: list[dict], cols: list[str]) -> pa.Table:
    return pa.table({col: [row[col] for row in rows] for col in cols})


def _snap(
    t: int,
    bid_volume: int,
    ask_volume: int,
    *,
    trade_volume: int = 0,
    trade_count: int = 0,
    ticker: str = TICKER,
) -> dict:
    row: dict = {
        "ticker": ticker,
        "TradingDay": DAY,
        "time_ms": t,
        "Volume": trade_volume,
        "DealNum": trade_count,
    }
    for level in range(1, 11):
        row[f"BidPrice{level}"] = 10.00 if level == 1 else None
        row[f"BidVolume{level}"] = bid_volume if level == 1 else 0
        row[f"AskPrice{level}"] = 10.10 if level == 1 else None
        row[f"AskVolume{level}"] = ask_volume if level == 1 else 0
    return row


def _write_fixture(
    root: Path,
    snapshots: list[dict] | None = None,
    *,
    ticker: str = TICKER,
) -> None:
    order_dir = root / "order" / "202101"
    trade_dir = root / "trades" / "202101"
    snap_dir = root / "snapshot"
    order_dir.mkdir(parents=True)
    trade_dir.mkdir(parents=True)
    snap_dir.mkdir(parents=True)

    orders = [
        {
            "ticker": ticker,
            "TradingDay": DAY,
            "time_ms": 130,
            "OrderID": "B",
            "Price": 10.00,
            "Volume": 50,
            "OrderType": 2,
        },
        {
            "ticker": ticker,
            "TradingDay": DAY,
            "time_ms": 135,
            "OrderID": "MISSING",
            "Price": 10.00,
            "Volume": 20,
            "OrderType": -1,
        },
    ]
    trades = [
        {
            "ticker": ticker,
            "TradingDay": DAY,
            "time_ms": 150,
            "DealID": "T1",
            "Price": 10.10,
            "Volume": 10,
            "Side": 1,
            "bsflag": 1,
            "BuyID": "UNKNOWN-B",
            "SellID": "UNKNOWN-S",
        }
    ]
    if snapshots is None:
        snapshots = [
            _snap(125, bid_volume=100, ask_volume=100, ticker=ticker),
            _snap(140, bid_volume=130, ask_volume=100, ticker=ticker),
            _snap(
                160,
                bid_volume=130,
                ask_volume=90,
                trade_volume=10,
                trade_count=1,
                ticker=ticker,
            ),
        ]

    pq.write_table(_table(orders, ORDER_COLS), order_dir / "order_2021-01-04.parquet")
    pq.write_table(_table(trades, TRADE_COLS), trade_dir / "trades_2021-01-04.parquet")
    pq.write_table(_table(snapshots, SNAP_COLS), snap_dir / "snapshot_202101.parquet")


def test_diagnose_day_intervals_attributes_matching_and_mismatching_windows(tmp_path: Path):
    _write_fixture(tmp_path)

    diagnostics = diagnose_day_intervals(DAY, tmp_path, TICKER, event_lag_ms=0)

    assert len(diagnostics) == 2

    first = diagnostics[0]
    assert first.start_ms == 125
    assert first.end_ms == 140
    assert first.status == "matched"
    assert first.order_count == 1
    assert first.order_volume == 50
    assert first.cancel_count == 1
    assert first.unresolved_cancel_count == 1
    assert first.unresolved_cancel_volume == 20
    assert first.order_type_counts == {2: 1}
    assert first.simulated_trade_volume == 0
    assert first.real_trade_volume == 0

    second = diagnostics[1]
    assert second.status == "mismatched"
    assert second.ask_price_match is True
    assert second.ask_volume_error == 10
    assert second.real_trade_count == 1
    assert second.real_trade_volume == 10
    assert second.real_trade_unknown_buy_count == 1
    assert second.real_trade_unknown_sell_count == 1
    assert second.missing_aggressor_trade_count == 1
    assert second.missing_aggressor_trade_volume == 10
    assert second.missing_passive_trade_count == 1
    assert second.missing_passive_trade_volume == 10
    assert second.simulated_trade_volume == 0
    assert second.snapshot_trade_volume == 10
    assert second.snapshot_trade_count == 1
    assert second.trade_volume_vs_snapshot_error == 0
    assert second.trade_count_vs_snapshot_error == 0


def test_summarize_diagnostics_separates_input_gaps_from_engine_candidates(tmp_path: Path):
    _write_fixture(tmp_path)
    diagnostics = diagnose_day_intervals(DAY, tmp_path, TICKER, event_lag_ms=0)

    summary = summarize_diagnostics(diagnostics)

    assert summary.total_intervals == 2
    assert summary.matched == 1
    assert summary.mismatched == 1
    assert summary.not_comparable == 0
    assert summary.aggressor_mapping_known is True
    assert summary.missing_aggressor_mismatches == 1
    assert summary.engine_candidate_mismatches == 0
    assert summary.ask_volume_mismatches == 1


def test_diagnose_day_intervals_shifts_event_window_with_lag(tmp_path: Path):
    _write_fixture(tmp_path)

    diagnostics = diagnose_day_intervals(DAY, tmp_path, TICKER, event_lag_ms=10)

    assert diagnostics[0].real_trade_count == 1
    assert diagnostics[0].real_trade_volume == 10
    assert diagnostics[0].trade_volume_vs_snapshot_error == 10
    assert diagnostics[1].real_trade_count == 0


def test_diagnose_day_intervals_marks_aggressor_mapping_unknown_for_shanghai(tmp_path: Path):
    ticker = "600000"
    _write_fixture(tmp_path, ticker=ticker)

    diagnostics = diagnose_day_intervals(DAY, tmp_path, ticker, event_lag_ms=0)

    second = diagnostics[1]
    assert second.aggressor_mapping_known is False
    assert second.missing_aggressor_trade_count is None
    assert second.missing_aggressor_trade_volume is None
    assert second.missing_passive_trade_count is None
    assert second.missing_passive_trade_volume is None


def test_diagnose_day_intervals_excludes_closing_auction(tmp_path: Path):
    snapshots = [
        _snap(MARKET_END_MS - 20, 100, 100),
        _snap(MARKET_END_MS - 10, 100, 100),
        _snap(MARKET_END_MS + 10, 500, 500),
    ]
    _write_fixture(tmp_path, snapshots=snapshots)

    diagnostics = diagnose_day_intervals(DAY, tmp_path, TICKER, event_lag_ms=0)

    assert len(diagnostics) == 1
    assert diagnostics[0].status == "matched"
