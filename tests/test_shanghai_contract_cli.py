from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ticknet.simulator.opening_ledger import (
    OpeningDayLagScan,
    OpeningLagAudit,
    audit_opening_ledger,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_shanghai_contract.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_shanghai_contract_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_json_and_csv_for_sample_and_lag_scan(tmp_path: Path, monkeypatch):
    module = _load_script()
    audit = audit_opening_ledger([], [], [], expected_bid_levels=(), expected_ask_levels=())
    candidate = OpeningLagAudit(10, audit)
    scan = OpeningDayLagScan(
        day=20220615,
        ticker="600000",
        candidates=(candidate,),
        best=candidate,
        snapshot_time_ms=0,
        coverage_status="covered",
        preopen_file_present=True,
        preopen_ticker_present=True,
    )
    monkeypatch.setattr(module, "scan_opening_day_lags", lambda *args: scan)

    json_path = tmp_path / "audit.json"
    csv_path = tmp_path / "audit.csv"
    assert (
        module.main(
            [
                "--sample",
                "20220615:600000",
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
    assert payload["summary"]["best_lag_counts"] == {"10": 1}
    assert "coverage_status" in csv_path.read_text(encoding="utf-8").splitlines()[0]
