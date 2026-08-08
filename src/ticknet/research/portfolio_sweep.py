"""把 Top-K 成本网格收敛为可审计的 M3 诊断结论。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class SweepDiagnosticError(ValueError):
    """Top-K 网格不完整、不可比或缺少诊断字段。"""


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepDiagnosticError(f"{field} 必须为有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise SweepDiagnosticError(f"{field} 必须为有限数值")
    return number


def _nested_number(summary: Mapping[str, Any], *path: str) -> float:
    value: object = summary
    for field in path:
        if not isinstance(value, Mapping) or field not in value:
            raise SweepDiagnosticError(f"summary 缺少字段: {'.'.join(path)}")
        value = value[field]
    return _finite_number(value, field=".".join(path))


def _policy_identity(summary: Mapping[str, Any]) -> tuple[int, int, float]:
    policy = summary.get("policy")
    costs = summary.get("cost_model")
    if not isinstance(policy, Mapping) or not isinstance(costs, Mapping):
        raise SweepDiagnosticError("summary 缺少 policy/cost_model")
    top_k_value = policy.get("top_k")
    buffer_value = policy.get("exit_buffer")
    if isinstance(top_k_value, bool) or not isinstance(top_k_value, int) or top_k_value <= 0:
        raise SweepDiagnosticError("policy.top_k 必须为正整数")
    if isinstance(buffer_value, bool) or not isinstance(buffer_value, int) or buffer_value < 0:
        raise SweepDiagnosticError("policy.exit_buffer 必须为非负整数")
    cost = _finite_number(costs.get("per_side_bps"), field="cost_model.per_side_bps")
    if cost < 0:
        raise SweepDiagnosticError("cost_model.per_side_bps 不能为负数")
    return top_k_value, buffer_value, cost


def _positive_active_month_ratio(summary: Mapping[str, Any]) -> tuple[int, float]:
    monthly = summary.get("monthly_stability")
    if not isinstance(monthly, Mapping) or not monthly:
        raise SweepDiagnosticError("summary.monthly_stability 必须为非空对象")
    positive = 0
    for month, values in monthly.items():
        if not isinstance(values, Mapping):
            raise SweepDiagnosticError(f"monthly_stability.{month} 必须为对象")
        net_active_mean = _finite_number(
            values.get("net_active_mean"),
            field=f"monthly_stability.{month}.net_active_mean",
        )
        positive += net_active_mean > 0
    return len(monthly), positive / len(monthly)


def _breakeven_per_side_bps(
    summary: Mapping[str, Any],
    *,
    active: bool,
) -> float | None:
    """由收益、换手和印花税解出绝对收益或相对等权基准的成本门槛。"""
    gross_mean = (
        _nested_number(summary, "ranking", "mean_active_return")
        if active
        else _nested_number(summary, "gross", "mean_daily")
    )
    mean_buy = _nested_number(summary, "turnover", "mean_buy")
    mean_sell = _nested_number(summary, "turnover", "mean_sell")
    stamp_tax = _nested_number(summary, "cost_model", "sell_stamp_tax_bps")
    charged_notional = mean_buy + mean_sell
    if charged_notional <= 0:
        return None
    return (gross_mean * 10_000.0 - stamp_tax * mean_sell) / charged_notional


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _validate_thresholds(
    *,
    evaluation_mode: str,
    decision_cost_bps: float,
    minimum_evaluated_dates: int,
    minimum_positive_month_ratio: float,
    maximum_top5_absolute_contribution: float,
) -> float:
    if evaluation_mode not in {"smoke", "formal"}:
        raise SweepDiagnosticError("evaluation_mode 应为 smoke 或 formal")
    decision_cost = _finite_number(decision_cost_bps, field="decision_cost_bps")
    if decision_cost < 0:
        raise SweepDiagnosticError("decision_cost_bps 不能为负数")
    if minimum_evaluated_dates <= 0:
        raise SweepDiagnosticError("minimum_evaluated_dates 必须为正整数")
    for name, value in {
        "minimum_positive_month_ratio": minimum_positive_month_ratio,
        "maximum_top5_absolute_contribution": maximum_top5_absolute_contribution,
    }.items():
        number = _finite_number(value, field=name)
        if not 0 <= number <= 1:
            raise SweepDiagnosticError(f"{name} 必须在 [0, 1] 范围内")
    return decision_cost


def _index_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[int, int, float], Mapping[str, Any]], int, tuple[str, str]]:
    indexed: dict[tuple[int, int, float], Mapping[str, Any]] = {}
    expected_dates: int | None = None
    expected_range: tuple[str, str] | None = None
    for key, summary in summaries.items():
        identity = _policy_identity(summary)
        if identity in indexed:
            raise SweepDiagnosticError(f"Top-K 网格组合重复: {identity}")
        indexed[identity] = summary
        dates_value = summary.get("evaluated_dates")
        if isinstance(dates_value, bool) or not isinstance(dates_value, int):
            raise SweepDiagnosticError(f"{key}.evaluated_dates 必须为整数")
        date_range = summary.get("date_range")
        if not isinstance(date_range, list) or len(date_range) != 2:
            raise SweepDiagnosticError(f"{key}.date_range 必须包含起止日期")
        normalized_range = (str(date_range[0]), str(date_range[1]))
        if expected_dates is None:
            expected_dates = dates_value
            expected_range = normalized_range
        elif dates_value != expected_dates or normalized_range != expected_range:
            raise SweepDiagnosticError("Top-K 网格使用了不同的评估日期，不能横向比较")
    if expected_dates is None or expected_range is None:
        raise SweepDiagnosticError("Top-K 网格不能为空")
    return indexed, expected_dates, expected_range


def _build_candidate(
    summary: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
    top_k: int,
    buffer: int,
    decision_cost: float,
    minimum_evaluated_dates: int,
    minimum_positive_month_ratio: float,
    maximum_top5_absolute_contribution: float,
) -> dict[str, Any]:
    month_count, positive_month_ratio = _positive_active_month_ratio(summary)
    gross_mean = _nested_number(summary, "gross", "mean_daily")
    net_mean = _nested_number(summary, "net", "mean_daily")
    net_sharpe = _nested_number(summary, "net", "sharpe")
    turnover = _nested_number(summary, "turnover", "mean_one_way")
    transaction_cost = _nested_number(summary, "turnover", "mean_transaction_cost")
    gross_active = _nested_number(summary, "ranking", "mean_active_return")
    net_active = gross_active - transaction_cost
    top5 = _nested_number(
        summary,
        "extreme_days",
        "top_5_absolute_active_contribution",
    )

    turnover_reduction = gross_delta = net_delta = cost_saving = 0.0
    buffer_efficient = buffer == 0
    if baseline is not None and buffer != 0:
        turnover_reduction = _nested_number(baseline, "turnover", "mean_one_way") - turnover
        gross_delta = gross_mean - _nested_number(baseline, "gross", "mean_daily")
        net_delta = net_mean - _nested_number(baseline, "net", "mean_daily")
        cost_saving = (
            _nested_number(baseline, "turnover", "mean_transaction_cost") - transaction_cost
        )
        buffer_efficient = turnover_reduction > 1e-12 and net_delta >= -1e-12

    criteria = {
        "enough_dates": int(summary["evaluated_dates"]) >= minimum_evaluated_dates,
        "positive_net_sharpe": net_sharpe > 0,
        "positive_net_active_return": net_active > 0,
        "stable_months": positive_month_ratio >= minimum_positive_month_ratio,
        "limited_extreme_day_concentration": top5 <= maximum_top5_absolute_contribution,
        "buffer_efficient": buffer_efficient,
    }
    return {
        "top_k": top_k,
        "exit_buffer": buffer,
        "cost_bps": decision_cost,
        "evaluated_dates": int(summary["evaluated_dates"]),
        "month_count": month_count,
        "net_mean_daily": net_mean,
        "net_sharpe": net_sharpe,
        "gross_active_mean_daily": gross_active,
        "net_active_mean_daily": net_active,
        "positive_net_active_month_ratio": positive_month_ratio,
        "top_5_absolute_active_contribution": top5,
        "mean_one_way_turnover": turnover,
        "turnover_reduction_vs_buffer0": turnover_reduction,
        "gross_mean_delta_vs_buffer0": gross_delta,
        "cost_saving_vs_buffer0": cost_saving,
        "net_mean_delta_vs_buffer0": net_delta,
        "criteria": criteria,
        "passes_sweet_spot": all(criteria.values()),
    }


def summarize_topk_sweep(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_mode: str,
    decision_cost_bps: float = 10.0,
    minimum_evaluated_dates: int = 60,
    minimum_positive_month_ratio: float = 0.5,
    maximum_top5_absolute_contribution: float = 0.5,
) -> dict[str, Any]:
    """验证完整网格，并在指定成本下识别 Top-K 可交易甜点区。"""
    if not summaries:
        raise SweepDiagnosticError("Top-K 网格不能为空")
    decision_cost = _validate_thresholds(
        evaluation_mode=evaluation_mode,
        decision_cost_bps=decision_cost_bps,
        minimum_evaluated_dates=minimum_evaluated_dates,
        minimum_positive_month_ratio=minimum_positive_month_ratio,
        maximum_top5_absolute_contribution=maximum_top5_absolute_contribution,
    )
    indexed, expected_dates, expected_range = _index_summaries(summaries)

    top_ks = sorted({identity[0] for identity in indexed})
    buffers = sorted({identity[1] for identity in indexed})
    costs = sorted({identity[2] for identity in indexed})
    missing = [
        (top_k, buffer, cost)
        for top_k in top_ks
        for buffer in buffers
        for cost in costs
        if (top_k, buffer, cost) not in indexed
    ]
    if missing:
        raise SweepDiagnosticError(f"Top-K 网格不完整，缺少组合: {missing}")
    decision_matches = [cost for cost in costs if _same_number(cost, decision_cost)]
    if len(decision_matches) != 1:
        raise SweepDiagnosticError("decision_cost_bps 必须包含在 cost_bps 网格中")
    decision_cost = decision_matches[0]

    candidates: list[dict[str, Any]] = []
    breakeven: list[dict[str, Any]] = []
    for top_k in top_ks:
        baseline = indexed[(top_k, 0, decision_cost)] if 0 in buffers else None
        for buffer in buffers:
            summary = indexed[(top_k, buffer, decision_cost)]
            candidate = _build_candidate(
                summary,
                baseline=baseline,
                top_k=top_k,
                buffer=buffer,
                decision_cost=decision_cost,
                minimum_evaluated_dates=minimum_evaluated_dates,
                minimum_positive_month_ratio=minimum_positive_month_ratio,
                maximum_top5_absolute_contribution=maximum_top5_absolute_contribution,
            )
            candidates.append(candidate)
            breakeven.append(
                {
                    "top_k": top_k,
                    "exit_buffer": buffer,
                    "absolute_return_breakeven_per_side_bps": _breakeven_per_side_bps(
                        summary,
                        active=False,
                    ),
                    "active_return_breakeven_per_side_bps": _breakeven_per_side_bps(
                        summary,
                        active=True,
                    ),
                }
            )

    ranked = sorted(
        candidates,
        key=lambda row: (
            not bool(row["passes_sweet_spot"]),
            -float(row["net_active_mean_daily"]),
            -float(row["net_sharpe"]),
            float(row["mean_one_way_turnover"]),
            int(row["top_k"]),
            int(row["exit_buffer"]),
        ),
    )
    sweet_spots = [row for row in ranked if row["passes_sweet_spot"]]
    status = "TRADEABLE_REGION_FOUND" if sweet_spots else "NO_TRADEABLE_REGION"
    return {
        "schema_version": "m3-topk-diagnostic-v1",
        "evaluation_mode": evaluation_mode,
        "decision": {
            "status": status,
            "tradeable_region_found": int(bool(sweet_spots)),
            "sweet_spot_count": len(sweet_spots),
            "best_candidate": sweet_spots[0] if sweet_spots else None,
        },
        "grid": {
            "top_k": top_ks,
            "exit_buffer": buffers,
            "cost_bps": costs,
            "validated_combinations": len(indexed),
        },
        "comparability": {
            "evaluated_dates": expected_dates,
            "date_range": list(expected_range),
            "same_date_sample": True,
        },
        "thresholds": {
            "decision_cost_bps": decision_cost,
            "minimum_evaluated_dates": minimum_evaluated_dates,
            "minimum_positive_month_ratio": minimum_positive_month_ratio,
            "maximum_top5_absolute_contribution": maximum_top5_absolute_contribution,
        },
        "candidates_at_decision_cost": ranked,
        "breakeven_by_policy": sorted(
            breakeven,
            key=lambda row: (int(row["top_k"]), int(row["exit_buffer"])),
        ),
        "interpretation_limit": (
            "工程 smoke 只验证诊断链路，不能作为正式交易结论"
            if evaluation_mode == "smoke"
            else "正式诊断仍需结合基线模型和滚动窗口证据"
        ),
    }
