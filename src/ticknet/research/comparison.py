"""从 Registry 构建多 seed 实验对比和 walk-forward 稳健性摘要。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypedDict

import numpy as np

from ticknet.research.registry import ExperimentRegistry

COMPARABLE_STATUSES = frozenset({"completed", "frozen", "locked_tested"})


class ComparisonError(ValueError):
    """待比较实验或指标不满足确定性对比契约。"""


class NumericSummary(TypedDict):
    count: int
    mean: float
    std: float | None
    min: float
    max: float


def _unique_names(values: list[str], *, label: str) -> list[str]:
    names = [value.strip() for value in values]
    if not names or any(not name for name in names):
        raise ComparisonError(f"{label} 必须非空")
    if len(set(names)) != len(names):
        raise ComparisonError(f"{label} 不能重复")
    return names


def _source_experiments(
    registry: ExperimentRegistry,
    experiment_ids: list[str],
) -> dict[str, dict[str, Any]]:
    experiments: dict[str, dict[str, Any]] = {}
    for experiment_id in _unique_names(experiment_ids, label="experiment_ids"):
        experiment = registry.get_experiment(experiment_id)
        if experiment is None:
            raise ComparisonError(f"待比较实验不存在: {experiment_id}")
        if experiment["status"] not in COMPARABLE_STATUSES:
            raise ComparisonError(
                f"待比较实验尚未完成: {experiment_id} status={experiment['status']}"
            )
        experiments[experiment_id] = experiment
    return experiments


def _metric_seed_values(
    registry: ExperimentRegistry,
    experiment_id: str,
    metric: str,
) -> dict[int, float]:
    rows = [row for row in registry.get_metrics(experiment_id) if row["metric"] == metric]
    if not rows:
        raise ComparisonError(f"实验缺少指标: {experiment_id} {metric}")
    return {int(row["seed"]): float(row["value"]) for row in rows}


def _summary(values: list[float]) -> NumericSummary:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(values) > 1 else None,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _metric_directions(
    metrics: list[str],
    values: dict[str, str] | None,
) -> dict[str, str]:
    directions = dict.fromkeys(metrics, "higher")
    for metric, direction in (values or {}).items():
        if metric not in directions:
            raise ComparisonError(f"metric_directions 包含未比较指标: {metric}")
        if direction not in {"higher", "lower"}:
            raise ComparisonError(f"指标方向必须为 higher 或 lower: {metric}")
        directions[metric] = direction
    return directions


def aggregate_dataset_fingerprint(
    experiments: dict[str, dict[str, Any]],
) -> str:
    """按实验 ID 和数据指纹生成稳定的聚合 fingerprint。"""
    values = [
        {
            "experiment_id": experiment_id,
            "dataset_fingerprint": experiment["dataset_fingerprint"],
        }
        for experiment_id, experiment in sorted(experiments.items())
    ]
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_registered_experiments(
    registry: ExperimentRegistry,
    experiment_ids: list[str],
    metrics: list[str],
    *,
    baseline_id: str,
    metric_directions: dict[str, str] | None = None,
    require_same_fingerprint: bool = True,
) -> tuple[dict[str, Any], str]:
    """对每个指标报告 seed 分布、相对基线差值和同 seed 配对差值。"""
    experiments = _source_experiments(registry, experiment_ids)
    metric_names = _unique_names(metrics, label="metrics")
    directions = _metric_directions(metric_names, metric_directions)
    if baseline_id not in experiments:
        raise ComparisonError("baseline_id 必须包含在 experiment_ids 中")
    fingerprints = [str(row["dataset_fingerprint"] or "") for row in experiments.values()]
    if require_same_fingerprint:
        if any(not value for value in fingerprints):
            raise ComparisonError("待比较实验必须登记 dataset_fingerprint")
        if len(set(fingerprints)) != 1:
            raise ComparisonError("待比较实验必须使用相同 dataset_fingerprint")

    values_by_metric: dict[str, dict[str, dict[int, float]]] = {}
    for metric in metric_names:
        values_by_metric[metric] = {
            experiment_id: _metric_seed_values(registry, experiment_id, metric)
            for experiment_id in experiments
        }

    output: dict[str, Any] = {
        "baseline_id": baseline_id,
        "experiment_ids": list(experiments),
        "metrics": metric_names,
        "experiments": {},
    }
    for experiment_id in experiments:
        experiment_metrics: dict[str, Any] = {}
        for metric in metric_names:
            seed_values = values_by_metric[metric][experiment_id]
            baseline_values = values_by_metric[metric][baseline_id]
            common_seeds = sorted(set(seed_values) & set(baseline_values))
            paired_deltas = [seed_values[seed] - baseline_values[seed] for seed in common_seeds]
            direction_sign = 1.0 if directions[metric] == "higher" else -1.0
            paired_improvements = [direction_sign * value for value in paired_deltas]
            summary = _summary(list(seed_values.values()))
            baseline_mean = _summary(list(baseline_values.values()))["mean"]
            raw_delta = summary["mean"] - baseline_mean
            improvement = direction_sign * raw_delta
            experiment_metrics[metric] = {
                **summary,
                "direction": directions[metric],
                "seed_values": [
                    {"seed": seed, "value": value} for seed, value in sorted(seed_values.items())
                ],
                "delta_vs_baseline_mean": raw_delta,
                "improvement_vs_baseline_mean": improvement,
                "paired_seed_count": len(common_seeds),
                "paired_delta_mean": (float(np.mean(paired_deltas)) if paired_deltas else None),
                "paired_delta_std": (
                    float(np.std(paired_deltas, ddof=1)) if len(paired_deltas) > 1 else None
                ),
                "paired_improvement_mean": (
                    float(np.mean(paired_improvements)) if paired_improvements else None
                ),
                "paired_improvement_std": (
                    float(np.std(paired_improvements, ddof=1))
                    if len(paired_improvements) > 1
                    else None
                ),
            }
        output["experiments"][experiment_id] = {
            "status": experiments[experiment_id]["status"],
            "dataset_fingerprint": experiments[experiment_id]["dataset_fingerprint"],
            "evaluation_decision": experiments[experiment_id]["evaluation_decision"],
            "metrics": experiment_metrics,
        }
    return output, aggregate_dataset_fingerprint(experiments)


def summarize_walk_forward(
    registry: ExperimentRegistry,
    experiment_ids: list[str],
    metrics: list[str],
    *,
    minimum_windows: int = 3,
    require_distinct_fingerprints: bool = True,
    metric_directions: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    """把每个已完成实验视为一个窗口，并汇总跨窗口最差值与波动。"""
    if minimum_windows < 2:
        raise ComparisonError("minimum_windows 至少为 2")
    experiments = _source_experiments(registry, experiment_ids)
    if len(experiments) < minimum_windows:
        raise ComparisonError(f"walk-forward 窗口不足: {len(experiments)} < {minimum_windows}")
    fingerprints = [str(row["dataset_fingerprint"] or "") for row in experiments.values()]
    if any(not value for value in fingerprints):
        raise ComparisonError("walk-forward 实验必须登记 dataset_fingerprint")
    if require_distinct_fingerprints and len(set(fingerprints)) != len(fingerprints):
        raise ComparisonError("walk-forward 窗口必须使用不同 dataset_fingerprint")

    metric_names = _unique_names(metrics, label="metrics")
    directions = _metric_directions(metric_names, metric_directions)
    output: dict[str, Any] = {
        "experiment_ids": list(experiments),
        "window_count": len(experiments),
        "metrics": {},
    }
    for metric in metric_names:
        windows: dict[str, Any] = {}
        window_means: list[float] = []
        for experiment_id in experiments:
            seed_values = _metric_seed_values(registry, experiment_id, metric)
            summary = _summary(list(seed_values.values()))
            windows[experiment_id] = {
                **summary,
                "dataset_fingerprint": experiments[experiment_id]["dataset_fingerprint"],
                "seed_values": [
                    {"seed": seed, "value": value} for seed, value in sorted(seed_values.items())
                ],
            }
            window_means.append(summary["mean"])
        aggregate = _summary(window_means)
        if directions[metric] == "higher":
            worst_index = int(np.argmin(np.asarray(window_means)))
        else:
            worst_index = int(np.argmax(np.asarray(window_means)))
        output["metrics"][metric] = {
            "direction": directions[metric],
            "windows": windows,
            "window_count": len(window_means),
            "window_mean": aggregate["mean"],
            "window_std": aggregate["std"],
            "window_min": aggregate["min"],
            "window_max": aggregate["max"],
            "above_zero_window_ratio": sum(value > 0 for value in window_means) / len(window_means),
            "worst_window_value": window_means[worst_index],
            "worst_window_experiment_id": list(experiments)[worst_index],
        }
    return output, aggregate_dataset_fingerprint(experiments)
