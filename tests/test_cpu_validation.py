from __future__ import annotations

from datetime import date

import pyarrow as pa
import pytest

from ticknet.research.cpu_validation import (
    compare_portfolio_digests,
    compare_portfolio_evaluations,
    digest_portfolio_backtester_result,
    run_cpu_smoke_pipeline,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, trading_date in enumerate((date(2024, 1, 2), date(2024, 1, 3))):
        for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
            feature = float(symbol_index + 1 + offset)
            rows.append(
                {
                    "symbol": symbol,
                    "trading_date": trading_date,
                    "label_date": date(2024, 1, 4 + offset),
                    "return_end_date": date(2024, 1, 5 + offset),
                    "feature": feature,
                    "target_return": feature / 100.0,
                    "is_train": offset == 0,
                }
            )
    return rows


def test_cpu_smoke_pipeline_produces_alpha_and_portfolio_artifacts() -> None:
    result = run_cpu_smoke_pipeline(pa.Table.from_pylist(_rows()), top_k=2)

    assert result.predictions.num_rows == 4
    assert result.alpha_signals.column_names[0] == "signal_date"
    assert result.evaluation.summary["mode"] == "topk_long_only"
    assert len(result.evaluation.daily) == 1
    assert result.metadata["device"] == "cpu"
    assert result.metadata["train_rows"] == 4
    assert result.metadata["test_rows"] == 4


def test_cpu_smoke_pipeline_rejects_training_rows_in_test_dates() -> None:
    rows = _rows()
    rows[-1]["is_train"] = True

    with pytest.raises(ValueError, match="test rows"):
        run_cpu_smoke_pipeline(pa.Table.from_pylist(rows), top_k=2)


def test_differential_comparison_reports_exact_and_tolerant_differences() -> None:
    result = run_cpu_smoke_pipeline(pa.Table.from_pylist(_rows()), top_k=2)
    same = compare_portfolio_evaluations(result.evaluation, result.evaluation)
    assert same["status"] == "match"
    assert same["mismatches"] == []

    changed = {
        "gross": result.evaluation.summary["gross"]["cumulative"],
        "net": float(result.evaluation.summary["net"]["cumulative"]) + 0.01,
        "net_max_drawdown": result.evaluation.summary["net"]["max_drawdown"],
    }
    mismatch = compare_portfolio_evaluations(
        result.evaluation,
        result.evaluation,
        right_summary_override=changed,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["mismatches"][0]["field"] == "summary.net"


def test_portfolio_backtester_result_is_converted_to_the_same_digest() -> None:
    class Series:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    result = (
        {"total_return": 0.02},
        Series([0.01, 0.01]),
        Series([0.02, 0.01]),
        Series([0.5, 0.25]),
        [
            {"period_end": "2024-01-03", "net_return": 0.01, "gross_return": 0.02},
            {"period_end": "2024-01-04", "net_return": 0.01, "gross_return": 0.01},
        ],
    )

    digest = digest_portfolio_backtester_result(result)

    assert digest["summary"]["net"] == pytest.approx(0.0201)
    assert digest["summary"]["gross"] == pytest.approx(0.0302)
    assert digest["daily"][0]["label_date"] == "2024-01-03"
    assert digest["daily"][0]["transaction_cost"] == pytest.approx(0.01)


def test_digests_can_be_compared_without_importing_either_backtest_engine() -> None:
    left = {
        "summary": {"gross": 0.03, "net": 0.02, "net_max_drawdown": -0.01},
        "daily": [],
    }
    right = {
        "summary": {"gross": 0.03, "net": 0.02, "net_max_drawdown": -0.01},
        "daily": [],
    }

    assert compare_portfolio_digests(left, right)["status"] == "match"
