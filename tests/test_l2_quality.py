from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.simulator.quality import profile_parquet


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_profile_reports_nulls_ranges_dates_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "order_preopen" / "202401" / "order_2024-01-02.parquet"
    _write(
        path,
        [
            {
                "ticker": "000001",
                "TradingDay": 20240102,
                "time_ms": -100,
                "OrderID": 7,
                "Price": 1000,
                "Volume": 200,
                "OrderType": 2,
            },
            {
                "ticker": "000001",
                "TradingDay": 20240102,
                "time_ms": -120,
                "OrderID": 7,
                "Price": 0,
                "Volume": -1,
                "OrderType": -1,
            },
        ],
    )

    report = profile_parquet(path, batch_size=1)

    assert report["rows"] == 2
    assert report["nulls"]["ticker"] == 0
    assert report["trading_days"] == [20240102]
    assert report["duplicate_id_rows"] == 1
    assert report["nonpositive"]["Price"] == 1
    assert report["nonpositive"]["Volume"] == 1
    assert report["timestamp_backwards"] == 1


def test_profile_marks_id_tracking_truncation(tmp_path: Path) -> None:
    path = tmp_path / "trades.parquet"
    _write(
        path,
        [
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 1, "DealID": 1},
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 2, "DealID": 2},
        ],
    )

    report = profile_parquet(path, max_tracked_ids=1)

    assert report["id_tracking_truncated"] is True
    json.dumps(report, ensure_ascii=False)


def test_profile_counts_cross_batch_repeats_once(tmp_path: Path) -> None:
    path = tmp_path / "order.parquet"
    _write(
        path,
        [
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 1, "OrderID": 7},
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 2, "OrderID": 8},
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 3, "OrderID": 8},
            {"ticker": "000001", "TradingDay": 20240102, "time_ms": 4, "OrderID": 8},
        ],
    )

    report = profile_parquet(path, batch_size=2)

    assert report["duplicate_id_rows"] == 2
