"""全天 eventstream 月度 pilot 容量审计测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ticknet.eventstream.pilot import (
    build_manifest_universe,
    inventory_raw_month,
    normalize_month,
    run_pilot_audit,
)


def _feature_manifest(path: Path, day: int) -> Path:
    content = {
        "dataset_fingerprint": "a" * 64,
        "samples": [
            {"symbol": "600000", "trading_date": "2021-01-04"},
            {"symbol": "000001", "trading_date": "2021-01-04"},
            {"symbol": "600000", "trading_date": "2021-02-01"},
        ],
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_pilot_audit_reports_complete_synthetic_month(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    manifest = _feature_manifest(tmp_path / "manifest.json", packed_day["day"])
    universe_path = tmp_path / "universe.json"
    report_path = tmp_path / "pilot.json"

    report = run_pilot_audit(
        month="2021-01",
        raw_root=packed_day["raw_root"],
        pack_root=packed_day["pack_root"],
        feature_manifest=manifest,
        universe_output=universe_path,
        output=report_path,
        require_complete_pack=True,
    )

    assert report["raw"]["status"] == "complete"
    assert report["pack"]["status"] == "complete"
    assert report["pack"]["expected_days"] == 1
    assert report["pack"]["total_bytes"] > 0
    assert report["projections"]["five_month_packed_bytes"] > 0
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    assert universe["universes"]["20210104"] == ["000001", "600000"]
    assert report_path.is_file()


def test_raw_inventory_counts_snapshot_once(packed_day: dict) -> None:
    inventory = inventory_raw_month("202101", packed_day["raw_root"])

    assert inventory["streams"]["order"]["files"] == 1
    assert inventory["streams"]["trades"]["files"] == 1
    assert inventory["streams"]["snapshot"]["files"] == 1
    assert inventory["total_unique_input_bytes"] > 0


def test_manifest_universe_requires_matching_month(tmp_path: Path) -> None:
    manifest = _feature_manifest(tmp_path / "manifest.json", 20210104)
    with pytest.raises(ValueError, match="没有股票池"):
        build_manifest_universe(manifest, "2020-12")


@pytest.mark.parametrize("value", ["2021", "2021-1", "202113", "bad"])
def test_normalize_month_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match=r"月份|无效"):
        normalize_month(value)
