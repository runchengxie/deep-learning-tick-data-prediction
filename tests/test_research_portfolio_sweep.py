"""M3 Top-K 网格诊断的确定性单元测试。"""

from copy import deepcopy

import pytest

from ticknet.research.portfolio_sweep import SweepDiagnosticError, summarize_topk_sweep


def _summary(*, buffer: int, cost: float) -> dict:
    mean_buy = mean_sell = 0.5 if buffer == 0 else 0.3
    gross_mean = 0.004 if buffer == 0 else 0.0039
    transaction_cost = (cost * (mean_buy + mean_sell) + 5.0 * mean_sell) / 10_000.0
    return {
        "policy": {"top_k": 25, "exit_buffer": buffer},
        "cost_model": {"per_side_bps": cost, "sell_stamp_tax_bps": 5.0},
        "evaluated_dates": 100,
        "date_range": ["2025-01-02", "2025-05-30"],
        "gross": {"mean_daily": gross_mean},
        "net": {"mean_daily": gross_mean - transaction_cost, "sharpe": 1.2},
        "turnover": {
            "mean_buy": mean_buy,
            "mean_sell": mean_sell,
            "mean_one_way": (mean_buy + mean_sell) / 2,
            "mean_transaction_cost": transaction_cost,
        },
        "ranking": {"mean_active_return": 0.003},
        "monthly_stability": {
            "2025-01": {"net_active_mean": 0.003},
            "2025-02": {"net_active_mean": 0.002},
            "2025-03": {"net_active_mean": -0.001},
        },
        "extreme_days": {"top_5_absolute_active_contribution": 0.2},
    }


def _grid() -> dict[str, dict]:
    return {
        f"k25.buffer{buffer}.cost{cost}": _summary(buffer=buffer, cost=cost)
        for buffer in (0, 10)
        for cost in (5.0, 10.0)
    }


def test_sweep_finds_cost_adjusted_buffer_sweet_spot_and_breakeven() -> None:
    diagnostic = summarize_topk_sweep(
        _grid(),
        evaluation_mode="formal",
        decision_cost_bps=10,
    )
    decision = diagnostic["decision"]
    assert decision["status"] == "TRADEABLE_REGION_FOUND"
    assert decision["sweet_spot_count"] == 2
    assert decision["best_candidate"]["exit_buffer"] == 10
    assert decision["best_candidate"]["net_active_mean_daily"] == pytest.approx(0.00225)
    assert decision["best_candidate"]["turnover_reduction_vs_buffer0"] == pytest.approx(0.2)
    breakeven = diagnostic["breakeven_by_policy"]
    assert breakeven[0]["absolute_return_breakeven_per_side_bps"] == pytest.approx(37.5)
    assert breakeven[0]["active_return_breakeven_per_side_bps"] == pytest.approx(27.5)
    assert diagnostic["grid"]["validated_combinations"] == 4
    assert diagnostic["comparability"]["same_date_sample"] is True


def test_sweep_returns_explicit_no_region_instead_of_best_effort_claim() -> None:
    grid = _grid()
    for summary in grid.values():
        summary["ranking"]["mean_active_return"] = 0.0001
    diagnostic = summarize_topk_sweep(grid, evaluation_mode="smoke")
    assert diagnostic["decision"] == {
        "status": "NO_TRADEABLE_REGION",
        "tradeable_region_found": 0,
        "sweet_spot_count": 0,
        "best_candidate": None,
    }


def test_sweep_rejects_incomplete_or_noncomparable_grid() -> None:
    grid = _grid()
    grid.pop("k25.buffer10.cost5.0")
    with pytest.raises(SweepDiagnosticError, match="网格不完整"):
        summarize_topk_sweep(grid, evaluation_mode="smoke")

    grid = _grid()
    changed = deepcopy(grid["k25.buffer10.cost10.0"])
    changed["date_range"] = ["2025-01-03", "2025-05-30"]
    grid["k25.buffer10.cost10.0"] = changed
    with pytest.raises(SweepDiagnosticError, match="不同的评估日期"):
        summarize_topk_sweep(grid, evaluation_mode="smoke")
