from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ticknet.simulator.correctness import CorrectnessResult

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_realdata.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_realdata_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_passes_mode_and_reports_comparable_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    called: dict[str, object] = {}

    def fake_verify(day: int, root: Path, ticker: str, mode: str, event_lag_ms: int):
        called.update(day=day, root=root, ticker=ticker, mode=mode, event_lag_ms=event_lag_ms)
        return [
            CorrectnessResult(True, 0, 0, "ok"),
            CorrectnessResult(False, 1, 0, "bad"),
            CorrectnessResult(False, 0, 0, "skip", comparable=False),
        ]

    monkeypatch.setattr(module, "verify_day_correctness", fake_verify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_realdata.py",
            "--day",
            "20210104",
            "--ticker",
            "000001",
            "--root",
            str(tmp_path),
            "--mode",
            "interval",
            "--event-lag-ms",
            "25",
        ],
    )

    assert module.main() == 0
    assert called == {
        "day": 20210104,
        "root": tmp_path,
        "ticker": "000001",
        "mode": "interval",
        "event_lag_ms": 25,
    }
    output = capsys.readouterr().out
    assert "mode=interval" in output
    assert "event_lag_ms=25" in output
    assert "可比较 2" in output
    assert "跳过 1" in output
    assert "完全一致: 1/2" in output
    assert "不一致: 1/2" in output
    assert "买一一致: 1/2" in output
    assert "卖一一致: 2/2" in output
