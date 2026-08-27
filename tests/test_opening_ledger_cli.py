from __future__ import annotations

import importlib.util
from pathlib import Path

from ticknet.simulator.opening_ledger import OpeningDayAudit, audit_opening_ledger

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_opening_ledger.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_opening_ledger_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_repeated_samples_and_emits_json(monkeypatch, capsys):
    module = _load_script()
    audit = audit_opening_ledger([], [], [], expected_bid_levels=(), expected_ask_levels=())
    result = OpeningDayAudit(20210104, "000001", 0, 140, True, True, audit)
    monkeypatch.setattr(module, "audit_opening_day", lambda *args, **kwargs: result)

    exit_code = module.main(["--sample", "20210104:000001", "--sample", "20210105:000002"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"total_samples": 2' in output
    assert '"comparable_match_rate": 1.0' in output
