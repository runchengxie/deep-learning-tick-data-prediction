from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.simulator.coverage import CoverageRow, scan_preopen_coverage


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _day_path(root: Path, kind: str, day: int, name: str) -> Path:
    text = str(day)
    iso = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return root / kind / text[:6] / f"{name}_{iso}.parquet"


def _write_preopen(root: Path, day: int, ticker: str, volume: int) -> None:
    _write(
        _day_path(root, "order_preopen", day, "order"),
        [
            {
                "ticker": ticker,
                "TradingDay": day,
                "time_ms": -100,
                "OrderID": "B1",
                "Price": 1000,
                "Volume": volume,
                "OrderType": 2,
            },
            {
                "ticker": ticker,
                "TradingDay": day,
                "time_ms": -80,
                "OrderID": "B2",
                "Price": 999,
                "Volume": 200,
                "OrderType": 2,
            },
        ],
    )


def test_scan_preopen_coverage_separates_file_and_ticker_presence(tmp_path: Path):
    day = 20240102
    _write_preopen(tmp_path, day, "600000", 300)
    _write(
        _day_path(tmp_path, "order", day, "order"),
        [{"ticker": "600000", "time_ms": 100, "Volume": 100}],
    )
    _write(
        _day_path(tmp_path, "trades", day, "trades"),
        [
            {"ticker": "600000", "time_ms": -10, "Volume": 250},
            {"ticker": "600000", "time_ms": 20, "Volume": 90},
        ],
    )
    _write(
        tmp_path / "snapshot" / "snapshot_202401.parquet",
        [{"ticker": "600000", "TradingDay": day, "time_ms": 0}],
    )

    assert scan_preopen_coverage(tmp_path) == (
        CoverageRow(
            day=day,
            ticker="600000",
            year=2024,
            month="202401",
            market="shanghai",
            batch="202401",
            preopen_file_present=True,
            preopen_ticker_present=True,
            order_file_present=True,
            order_ticker_present=True,
            trades_file_present=True,
            trades_ticker_present=True,
            snapshot_file_present=True,
            snapshot_ticker_present=True,
            preopen_order_count=2,
            preopen_order_volume=500,
            opening_trade_count=1,
            opening_trade_volume=250,
        ),
    )


def test_scan_preopen_coverage_reports_missing_related_files_and_ticker(tmp_path: Path):
    day = 20240103
    _write_preopen(tmp_path, day, "000001", 100)
    _write(
        _day_path(tmp_path, "order", day, "order"),
        [{"ticker": "600000", "time_ms": 100, "Volume": 100}],
    )

    row = scan_preopen_coverage(tmp_path)[0]

    assert row.ticker == "000001"
    assert row.market == "shenzhen"
    assert row.order_file_present is True
    assert row.order_ticker_present is False
    assert row.trades_file_present is False
    assert row.snapshot_file_present is False
    assert row.opening_trade_count == 0


def test_scan_preopen_coverage_scans_related_files_once_per_day(tmp_path: Path, monkeypatch):
    day = 20240104
    _write_preopen(tmp_path, day, "600000", 100)
    _write(
        _day_path(tmp_path, "order", day, "order"),
        [
            {"ticker": "600000", "time_ms": 100, "Volume": 100},
            {"ticker": "000001", "time_ms": 100, "Volume": 200},
        ],
    )
    _write(
        _day_path(tmp_path, "trades", day, "trades"),
        [
            {"ticker": "600000", "time_ms": -10, "Volume": 50},
            {"ticker": "000001", "time_ms": -10, "Volume": 70},
        ],
    )
    _write(
        tmp_path / "snapshot" / "snapshot_202401.parquet",
        [
            {"ticker": "600000", "TradingDay": day, "time_ms": 0},
            {"ticker": "000001", "TradingDay": day, "time_ms": 0},
        ],
    )

    calls = {"order": 0, "trades": 0, "snapshot": 0}
    from ticknet.simulator import coverage

    for name in calls:
        original = getattr(coverage, f"_read_{name}_coverage")

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(coverage, f"_read_{name}_coverage", counted)

    rows = scan_preopen_coverage(tmp_path)

    assert len(rows) == 1
    assert calls == {"order": 1, "trades": 1, "snapshot": 1}


def test_coverage_cli_writes_rows_and_grouped_summary(tmp_path: Path):
    day = 20240102
    _write_preopen(tmp_path, day, "600000", 300)
    json_path = tmp_path / "coverage.json"
    csv_path = tmp_path / "coverage.csv"
    script = Path(__file__).parents[1] / "scripts" / "audit_opening_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_opening_coverage", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.main(
            [
                "--raw-root",
                str(tmp_path),
                "--json-output",
                str(json_path),
                "--csv-output",
                str(csv_path),
            ]
        )
        == 0
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["market"]["shanghai"]["samples"] == 1
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("day,ticker,")
