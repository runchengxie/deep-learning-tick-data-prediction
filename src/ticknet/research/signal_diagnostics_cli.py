"""运行两个连续滚动窗口的事件流信号与交易转化诊断。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.research.portfolio import PortfolioEvaluation, write_portfolio_artifacts
from ticknet.research.signal_diagnostics import (
    PolicyCandidate,
    SignalRow,
    calibrated_scores,
    compact_portfolio_summary,
    evaluate_policy,
    half_life_diagnostics,
    load_horizon_maps,
    load_joint_signal_rows,
    load_materialized_signal_rows,
    split_signal_rows,
    staggered_h5_evaluation,
)
from ticknet.research.signal_diagnostics_market import (
    build_market_attributes,
    reprice_dynamic_cost,
    reprice_staggered_dynamic_cost,
    risk_attribution,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _candidate_grid() -> list[PolicyCandidate]:
    candidates: list[PolicyCandidate] = []
    for alpha in (1.0, 0.5, 0.2):
        for threshold in (None, 0.0, 0.0005):
            for gap in (0.0, 0.00025, 0.0005):
                candidates.append(PolicyCandidate(alpha, threshold, gap))
    return candidates


def _calibrations(
    validation: list[SignalRow],
    oos: list[SignalRow],
    h5_targets,
) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    for alpha in (1.0, 0.5, 0.2):
        validation_scores, oos_scores, report = calibrated_scores(
            validation,
            oos,
            h5_targets,
            ema_alpha=alpha,
        )
        result[alpha] = {
            "validation": validation_scores,
            "oos": oos_scores,
            "report": report,
        }
    return result


def _run_matrix(
    candidates: list[PolicyCandidate],
    *,
    adjacent_validation: list[SignalRow],
    adjacent_oos: list[SignalRow],
    adjacent_calibrations: dict[float, dict[str, Any]],
    recent_validation: list[SignalRow],
    recent_oos: list[SignalRow],
    recent_calibrations: dict[float, dict[str, Any]],
    h1_targets,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, PortfolioEvaluation]]]:
    matrix: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, PortfolioEvaluation]] = {}
    for candidate in candidates:
        adjacent_scores = adjacent_calibrations[candidate.ema_alpha]
        recent_scores = recent_calibrations[candidate.ema_alpha]
        parts = {
            "adjacent_validation": evaluate_policy(
                adjacent_validation,
                adjacent_scores["validation"],
                h1_targets,
                candidate,
            ),
            "adjacent_oos": evaluate_policy(
                adjacent_oos,
                adjacent_scores["oos"],
                h1_targets,
                candidate,
            ),
            "recent_validation": evaluate_policy(
                recent_validation,
                recent_scores["validation"],
                h1_targets,
                candidate,
            ),
            "recent_oos": evaluate_policy(
                recent_oos,
                recent_scores["oos"],
                h1_targets,
                candidate,
            ),
        }
        evaluations[candidate.name] = parts
        matrix.append(
            {
                "name": candidate.name,
                "policy": {
                    "ema_alpha": candidate.ema_alpha,
                    "min_expected_return": candidate.min_expected_return,
                    "min_replacement_gap": candidate.min_replacement_gap,
                },
                "selection_scope": "adjacent_validation_only",
                **{name: compact_portfolio_summary(value) for name, value in parts.items()},
            }
        )
    return matrix, evaluations


def _select_candidates(matrix: list[dict[str, Any]], count: int = 3) -> list[str]:
    """只按较早折 validation 的成本后主动收益和换手选择少量规则。"""
    ordered = sorted(
        matrix,
        key=lambda row: (
            -float(row["adjacent_validation"]["mean_net_active_return"]),
            float(row["adjacent_validation"]["mean_one_way_turnover"]),
            row["policy"]["min_expected_return"] is not None,
            str(row["name"]),
        ),
    )
    return [str(row["name"]) for row in ordered[:count]]


def _candidate_by_name(candidates: list[PolicyCandidate], name: str) -> PolicyCandidate:
    return next(candidate for candidate in candidates if candidate.name == name)


def _plot_half_life(results: dict[str, dict[str, Any]], output: Path) -> None:
    plt.rcParams["svg.hashsalt"] = "ticknet-eventstream-half-life-v1"
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name, report in results.items():
        horizons = sorted(int(value) for value in report)
        axes[0].plot(
            horizons,
            [report[str(value)]["mean_rank_ic"] for value in horizons],
            marker="o",
            label=name,
        )
        axes[1].plot(
            horizons,
            [report[str(value)]["mean_extreme_spread"] * 10_000 for value in horizons],
            marker="o",
            label=name,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean Rank IC")
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Extreme spread (bp)")
    axes[1].set_xlabel("Holding horizon (trading days)")
    axes[1].set_xticks(range(1, 11))
    fig.tight_layout()
    temporary = output.with_suffix(".tmp.svg")
    fig.savefig(temporary, format="svg", metadata={"Date": None})
    plt.close(fig)
    os.replace(temporary, output)


def _write_evaluation_details(
    output: Path,
    name: str,
    evaluations: dict[str, PortfolioEvaluation],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition, evaluation in evaluations.items():
        paths = write_portfolio_artifacts(
            evaluation,
            output / "portfolio-details" / name / partition,
        )
        result[partition] = {
            key: {
                "path": str(Path(path).relative_to(output)),
                "sha256": file_sha256(Path(path)),
            }
            for key, path in paths.items()
        }
    return result


def _staggered_for_candidate(
    candidate: PolicyCandidate,
    *,
    adjacent_oos: list[SignalRow],
    recent_oos: list[SignalRow],
    adjacent_calibrations: dict[float, dict[str, Any]],
    recent_calibrations: dict[float, dict[str, Any]],
    h5_targets,
) -> dict[str, Any]:
    return {
        "adjacent_oos": staggered_h5_evaluation(
            adjacent_oos,
            adjacent_calibrations[candidate.ema_alpha]["oos"],
            h5_targets,
            min_expected_return=candidate.min_expected_return,
        ),
        "recent_oos": staggered_h5_evaluation(
            recent_oos,
            recent_calibrations[candidate.ema_alpha]["oos"],
            h5_targets,
            min_expected_return=candidate.min_expected_return,
        ),
        "replacement_gap_note": (
            "H5 cohort 在第五日收盘结束，下一 cohort 从次日开盘开始。"
            "由于标签不含隔夜收益，min_replacement_gap 只在 H1 日度组合中评估。"
        ),
    }


def _gate_report(
    *,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    dynamic: dict[str, dict[str, Any]],
    risks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    windows = ("adjacent_oos", "recent_oos")
    improvement = {
        name: float(candidate[name]["mean_net_active_return"])
        - float(baseline[name]["mean_net_active_return"])
        for name in windows
    }
    turnover_ratio = {
        name: float(candidate[name]["mean_one_way_turnover"])
        / max(1e-12, float(baseline[name]["mean_one_way_turnover"]))
        for name in windows
    }
    checks = {
        "two_oos_static_net_active_same_positive_direction": all(
            float(candidate[name]["mean_net_active_return"]) > 0 for name in windows
        ),
        "two_oos_dynamic_net_active_same_positive_direction": all(
            float(dynamic[name]["mean_net_active_return"]) > 0 for name in windows
        ),
        "improves_baseline_by_at_least_1bp_daily_in_both_oos": all(
            value >= 0.0001 for value in improvement.values()
        ),
        "turnover_at_most_80_percent_of_baseline_in_both_oos": all(
            value <= 0.8 for value in turnover_ratio.values()
        ),
        "top_5_days_below_half_of_absolute_active_move": all(
            float(candidate[name]["top_5_absolute_active_contribution"]) < 0.5 for name in windows
        ),
        "size_liquidity_volatility_within_one_z": all(
            all(abs(float(value)) <= 1.0 for value in risks[name]["mean_exposure_z"].values())
            for name in windows
            if risks[name]["status"] == "available"
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "daily_net_active_improvement": improvement,
        "turnover_ratio": turnover_ratio,
        "industry_gate": "not_evaluated_missing_local_source",
    }


def run_diagnostics(arguments: argparse.Namespace) -> dict[str, Any]:
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    adjacent_rows, adjacent_identity = load_materialized_signal_rows(
        arguments.adjacent_keys,
        arguments.adjacent_scores,
    )
    recent_rows = load_joint_signal_rows(
        {"validation": arguments.recent_validation, "oos": arguments.recent_oos}
    )
    adjacent_validation, adjacent_oos = split_signal_rows(adjacent_rows)
    recent_validation, recent_oos = split_signal_rows(recent_rows)
    if max(row.trading_date for row in adjacent_oos) >= min(row.trading_date for row in recent_oos):
        raise ValueError("两个 OOS 窗口不是按时间先后排列")

    horizons, sidecar_fingerprint = load_horizon_maps(
        arguments.horizon_sidecar,
        source_dataset_fingerprint=adjacent_identity["source_dataset_fingerprint"],
        required_keys={(row.symbol, row.trading_date) for row in [*adjacent_rows, *recent_rows]},
    )
    half_life = {
        "adjacent_oos": half_life_diagnostics(adjacent_oos, horizons),
        "recent_oos": half_life_diagnostics(recent_oos, horizons),
    }
    _atomic_json(output / "half-life.json", half_life)
    _plot_half_life(half_life, output / "half-life.svg")

    adjacent_calibrations = _calibrations(adjacent_validation, adjacent_oos, horizons[5])
    recent_calibrations = _calibrations(recent_validation, recent_oos, horizons[5])
    candidates = _candidate_grid()
    matrix, evaluations = _run_matrix(
        candidates,
        adjacent_validation=adjacent_validation,
        adjacent_oos=adjacent_oos,
        adjacent_calibrations=adjacent_calibrations,
        recent_validation=recent_validation,
        recent_oos=recent_oos,
        recent_calibrations=recent_calibrations,
        h1_targets=horizons[1],
    )
    selected_names = _select_candidates(matrix)
    baseline_name = PolicyCandidate(1.0, None, 0.0).name
    details = {
        baseline_name: _write_evaluation_details(
            output,
            baseline_name,
            evaluations[baseline_name],
        )
    }
    for name in selected_names:
        if name not in details:
            details[name] = _write_evaluation_details(
                output,
                name,
                evaluations[name],
            )

    staggered = {
        baseline_name: _staggered_for_candidate(
            _candidate_by_name(candidates, baseline_name),
            adjacent_oos=adjacent_oos,
            recent_oos=recent_oos,
            adjacent_calibrations=adjacent_calibrations,
            recent_calibrations=recent_calibrations,
            h5_targets=horizons[5],
        )
    }
    for name in selected_names:
        staggered[name] = _staggered_for_candidate(
            _candidate_by_name(candidates, name),
            adjacent_oos=adjacent_oos,
            recent_oos=recent_oos,
            adjacent_calibrations=adjacent_calibrations,
            recent_calibrations=recent_calibrations,
            h5_targets=horizons[5],
        )

    market_attributes = {
        "adjacent_oos": build_market_attributes(arguments.basic_root, adjacent_oos),
        "recent_oos": build_market_attributes(arguments.basic_root, recent_oos),
    }
    dynamic: dict[str, Any] = {}
    risks: dict[str, Any] = {}
    for name in dict.fromkeys([baseline_name, *selected_names]):
        dynamic[name] = {}
        risks[name] = {}
        for window, signals in (("adjacent_oos", adjacent_oos), ("recent_oos", recent_oos)):
            evaluation = evaluations[name][window]
            dynamic[name][window] = {
                "daily_h1": reprice_dynamic_cost(
                    evaluation,
                    market_attributes[window],
                ),
                "staggered_h5": reprice_staggered_dynamic_cost(
                    staggered[name][window],
                    market_attributes[window],
                ),
            }
            risks[name][window] = risk_attribution(
                evaluation,
                signals,
                market_attributes[window],
            )

    best_name = selected_names[0]
    baseline_summaries = {
        window: compact_portfolio_summary(evaluations[baseline_name][window])
        for window in ("adjacent_oos", "recent_oos")
    }
    best_summaries = {
        window: compact_portfolio_summary(evaluations[best_name][window])
        for window in ("adjacent_oos", "recent_oos")
    }
    gate = _gate_report(
        baseline=baseline_summaries,
        candidate=best_summaries,
        dynamic={window: dynamic[best_name][window]["daily_h1"] for window in best_summaries},
        risks=risks[best_name],
    )
    report = {
        "status": "complete",
        "experiments": [
            "EVT-HALFLIFE-001",
            "TRD-STAGGERED-H5-001",
            "TRD-RANK-EMA-001",
            "RISK-ATTR-001",
        ],
        "selection_contract": (
            "交易规则只按较早折 validation 的 H1 成本后主动收益选择，"
            "11 月和 12 月 OOS 只用于通过或否定门槛"
        ),
        "return_contracts": {
            "half_life_and_staggered": "next_open_to_horizon_close_excess_benchmark",
            "policy_matrix": "H5 validation calibration plus H1 next_open_to_same_close evaluation",
        },
        "identity": {
            "adjacent": adjacent_identity,
            "horizon_sidecar_fingerprint": sidecar_fingerprint,
            "recent_prediction_sha256": {
                "validation": file_sha256(arguments.recent_validation.expanduser().resolve()),
                "oos": file_sha256(arguments.recent_oos.expanduser().resolve()),
            },
        },
        "rows": {
            "adjacent_validation": len(adjacent_validation),
            "adjacent_oos": len(adjacent_oos),
            "recent_validation": len(recent_validation),
            "recent_oos": len(recent_oos),
        },
        "calibration": {
            "adjacent": {
                str(alpha): values["report"] for alpha, values in adjacent_calibrations.items()
            },
            "recent": {
                str(alpha): values["report"] for alpha, values in recent_calibrations.items()
            },
        },
        "baseline": baseline_name,
        "selected_by_adjacent_validation": selected_names,
        "best_candidate": best_name,
        "best_candidate_oos": best_summaries,
        "baseline_oos": baseline_summaries,
        "staggered_h5": {
            name: {
                window: {
                    key: value for key, value in report.items() if key not in {"rows", "sleeves"}
                }
                for window, report in values.items()
                if window.endswith("oos")
            }
            for name, values in staggered.items()
        },
        "dynamic_cost": dynamic,
        "risk_attribution": risks,
        "gate": gate,
        "artifacts": details,
    }
    _atomic_json(output / "policy-matrix.json", matrix)
    _atomic_json(output / "staggered-h5.json", staggered)
    _atomic_json(output / "summary.json", report)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "name": row["name"],
                    "ema_alpha": row["policy"]["ema_alpha"],
                    "min_expected_return": row["policy"]["min_expected_return"],
                    "min_replacement_gap": row["policy"]["min_replacement_gap"],
                    "adjacent_validation_net_active": row["adjacent_validation"][
                        "mean_net_active_return"
                    ],
                    "adjacent_oos_net_active": row["adjacent_oos"]["mean_net_active_return"],
                    "recent_oos_net_active": row["recent_oos"]["mean_net_active_return"],
                    "adjacent_oos_turnover": row["adjacent_oos"]["mean_one_way_turnover"],
                    "recent_oos_turnover": row["recent_oos"]["mean_one_way_turnover"],
                }
                for row in matrix
            ]
        ),
        output / "policy-matrix.parquet",
        compression="zstd",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行事件流信号半衰期和交易转化诊断")
    parser.add_argument("--adjacent-keys", type=Path, required=True)
    parser.add_argument("--adjacent-scores", type=Path, required=True)
    parser.add_argument("--recent-validation", type=Path, required=True)
    parser.add_argument("--recent-oos", type=Path, required=True)
    parser.add_argument("--horizon-sidecar", type=Path, required=True)
    parser.add_argument("--basic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    report = run_diagnostics(_parser().parse_args(argv))
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
