from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ticknet.simulator.coverage import CoverageRow, load_or_build_coverage_index
from ticknet.simulator.eligibility import classify_coverage, summarize_eligibility


def _row(**changes) -> CoverageRow:
    values = {
        "day": 20240102,
        "ticker": "000001",
        "year": 2024,
        "month": "202401",
        "market": "shenzhen",
        "batch": "202401",
        "preopen_file_present": True,
        "preopen_ticker_present": True,
        "order_file_present": True,
        "order_ticker_present": True,
        "trades_file_present": True,
        "trades_ticker_present": True,
        "snapshot_file_present": True,
        "snapshot_ticker_present": True,
        "preopen_order_count": 10,
        "preopen_order_volume": 1000,
        "opening_trade_count": 2,
        "opening_trade_volume": 200,
    }
    values.update(changes)
    return CoverageRow(**values)


def test_classify_coverage_separates_shenzhen_and_shanghai_eligibility():
    shenzhen = classify_coverage(_row())
    shanghai = classify_coverage(_row(ticker="600000", market="shanghai"))

    assert shenzhen.primary_eligible is True
    assert shenzhen.shanghai_research_eligible is False
    assert shanghai.primary_eligible is False
    assert shanghai.shanghai_research_eligible is True
    assert "lag_not_calibrated" in shanghai.exclusion_reasons


def test_classify_coverage_excludes_incomplete_rows_and_2026():
    incomplete = classify_coverage(_row(order_ticker_present=False))
    future = classify_coverage(_row(day=20260105, year=2026))

    assert incomplete.primary_eligible is False
    assert incomplete.shanghai_research_eligible is False
    assert "related_ticker_missing" in incomplete.exclusion_reasons
    assert future.primary_eligible is False
    assert "outside_historical_window" in future.exclusion_reasons


def test_summarize_eligibility_counts_primary_and_research_rows():
    summary = summarize_eligibility(
        [
            classify_coverage(_row()),
            classify_coverage(_row(ticker="600000", market="shanghai")),
        ]
    )

    assert summary["total_rows"] == 2
    assert summary["primary_eligible"] == 1
    assert summary["shanghai_research_eligible"] == 1
    assert summary["by_market"]["shenzhen"]["primary_eligible"] == 1
    assert summary["by_market"]["shanghai"]["shanghai_research_eligible"] == 1


def test_manifest_cli_writes_json_and_csv(tmp_path: Path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "build_historical_data_manifest.py"
    spec = importlib.util.spec_from_file_location("historical_manifest", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "scan_preopen_coverage", lambda *args, **kwargs: (_row(),))

    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "manifest.csv"
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
    assert payload["summary"]["primary_eligible"] == 1
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("day,ticker,")


def test_coverage_index_reuses_unchanged_day_without_scanning(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "raw"
    index_path = tmp_path / "coverage-index.json"
    preopen_path = raw_root / "order_preopen" / "202401" / "order_2024-01-02.parquet"
    preopen_path.parent.mkdir(parents=True)
    preopen_path.write_bytes(b"preopen-v1")
    row = _row()
    calls = []

    def fake_scan(root, *, preopen_paths=None, limit_days=None):
        calls.append(tuple(preopen_paths or ()))
        return (row,)

    monkeypatch.setattr("ticknet.simulator.coverage.scan_preopen_coverage", fake_scan)

    first = load_or_build_coverage_index(raw_root, index_path)
    second = load_or_build_coverage_index(raw_root, index_path)

    assert first == (row,)
    assert second == (row,)
    assert len(calls) == 1
    assert calls[0] == (preopen_path,)


def test_coverage_index_rescans_day_when_source_changes(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "raw"
    index_path = tmp_path / "coverage-index.json"
    preopen_path = raw_root / "order_preopen" / "202401" / "order_2024-01-02.parquet"
    preopen_path.parent.mkdir(parents=True)
    preopen_path.write_bytes(b"preopen-v1")
    row = _row()
    calls = []

    def fake_scan(root, *, preopen_paths=None, limit_days=None):
        calls.append(tuple(preopen_paths or ()))
        return (row,)

    monkeypatch.setattr("ticknet.simulator.coverage.scan_preopen_coverage", fake_scan)

    load_or_build_coverage_index(raw_root, index_path)
    preopen_path.write_bytes(b"preopen-v2")
    load_or_build_coverage_index(raw_root, index_path)

    assert len(calls) == 2


def test_coverage_audit_cli_can_use_index(tmp_path: Path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "audit_opening_coverage.py"
    spec = importlib.util.spec_from_file_location("opening_coverage", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_or_build_coverage_index", lambda *args, **kwargs: (_row(),))

    json_path = tmp_path / "coverage.json"
    csv_path = tmp_path / "coverage.csv"
    assert (
        module.main(
            [
                "--raw-root",
                str(tmp_path),
                "--index-path",
                str(tmp_path / "index.json"),
                "--json-output",
                str(json_path),
                "--csv-output",
                str(csv_path),
            ]
        )
        == 0
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["year"]["2024"]["samples"] == 1
