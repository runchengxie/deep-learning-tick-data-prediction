"""把结构化 metric gates 转换为确定性的研究决策。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ticknet.research.spec import MetricGate


def flatten_numeric_metrics(
    values: object,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """递归提取嵌套 dict/list 中的有限数值，路径使用点号连接。"""
    flattened: dict[str, float] = {}
    if isinstance(values, dict):
        for key, value in values.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric_metrics(value, prefix=name))
    elif isinstance(values, (list, tuple)):
        for index, value in enumerate(values):
            name = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_numeric_metrics(value, prefix=name))
    elif isinstance(values, (int, float, np.integer, np.floating)) and not isinstance(values, bool):
        number = float(values)
        if prefix and math.isfinite(number):
            flattened[prefix] = number
    return flattened


def _passes(value: float, gate: MetricGate) -> bool:
    if gate.operator == "gt":
        return value > gate.threshold
    if gate.operator == "gte":
        return value >= gate.threshold
    if gate.operator == "lt":
        return value < gate.threshold
    return value <= gate.threshold


@dataclass(frozen=True)
class EvaluationResult:
    """KEEP、EXTEND、DISCARD 及每个 gate 的证据。"""

    decision: str
    gates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "gates": list(self.gates)}


def evaluate_metric_gates(
    per_seed_metrics: list[dict[str, Any]],
    gates: tuple[MetricGate, ...],
    *,
    stage: str,
) -> EvaluationResult:
    """对每个 gate 使用多 seed 均值；缺失指标确定性判为 DISCARD。"""
    flattened = [flatten_numeric_metrics(metrics) for metrics in per_seed_metrics]
    evidence: list[dict[str, Any]] = []
    all_passed = True
    for gate in gates:
        values = [metrics[gate.metric] for metrics in flattened if gate.metric in metrics]
        mean_value = float(np.mean(values)) if values else None
        passed = mean_value is not None and _passes(mean_value, gate)
        all_passed = all_passed and passed
        evidence.append(
            {
                "metric": gate.metric,
                "operator": gate.operator,
                "threshold": gate.threshold,
                "seed_values": values,
                "mean_value": mean_value,
                "passed": passed,
                "missing_seeds": len(per_seed_metrics) - len(values),
            }
        )
    if not all_passed:
        decision = "DISCARD"
    elif stage == "screening":
        decision = "EXTEND"
    else:
        decision = "KEEP"
    return EvaluationResult(decision=decision, gates=tuple(evidence))
