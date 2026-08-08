"""从 ExperimentRegistry 构造可重放的 ResearchContext。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ticknet.research.agents.context import ResearchContext
from ticknet.research.registry import ExperimentRegistry

BASELINE_STATUSES = frozenset({"completed", "frozen", "locked_tested"})
FAILURE_STATUSES = frozenset({"failed", "rejected"})


class ContextBuildError(ValueError):
    """Registry 证据不完整或不满足上下文契约。"""


def _json_object(value: object, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ContextBuildError(f"{label} 不是有效 JSON") from error
    if not isinstance(parsed, dict):
        raise ContextBuildError(f"{label} 必须为对象")
    return parsed


def _spec_values(experiment: dict[str, Any]) -> dict[str, Any]:
    return _json_object(
        experiment["spec_json"],
        label=f"实验 {experiment['experiment_id']} spec_json",
    )


def _novelty_signature(experiment: dict[str, Any]) -> str:
    return str(_spec_values(experiment).get("novelty_signature", "")).strip()


def _primary_metric_names(experiment: dict[str, Any]) -> list[str]:
    values = _spec_values(experiment).get("primary_metrics", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


@dataclass(frozen=True)
class ResearchContextBuilder:
    """只读取 Registry，不扫描仓库、行情数据或 locked predictions。"""

    registry: ExperimentRegistry
    recent_limit: int = 10
    anomaly_limit: int = 10
    failure_limit: int = 10

    def __post_init__(self) -> None:
        for name, value in {
            "recent_limit": self.recent_limit,
            "anomaly_limit": self.anomaly_limit,
            "failure_limit": self.failure_limit,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContextBuildError(f"{name} 必须为正整数")

    def build(
        self,
        research_question: str,
        *,
        baseline_experiment_id: str | None = None,
        compute_budget_hours: float = 4.0,
    ) -> ResearchContext:
        if not isinstance(research_question, str) or not research_question.strip():
            raise ContextBuildError("research_question 不能为空")
        if (
            isinstance(compute_budget_hours, bool)
            or not isinstance(compute_budget_hours, (int, float))
            or not math.isfinite(compute_budget_hours)
            or compute_budget_hours <= 0
        ):
            raise ContextBuildError("compute_budget_hours 必须为正数")

        experiments = self.registry.list_experiments(limit=None)
        baseline = self._select_baseline(experiments, baseline_experiment_id)
        recent = [self._experiment_summary(row) for row in experiments[: self.recent_limit]]
        failures = [
            self._failure_summary(row)
            for row in experiments
            if row["status"] in FAILURE_STATUSES or row["evaluation_decision"] == "DISCARD"
        ][: self.failure_limit]
        signatures = list(
            dict.fromkeys(
                signature for row in experiments if (signature := _novelty_signature(row))
            )
        )
        fingerprints = sorted(
            {str(row["dataset_fingerprint"]) for row in experiments if row["dataset_fingerprint"]}
        )
        context = ResearchContext(
            research_question=research_question,
            baseline_summary=self._baseline_summary(baseline) if baseline else {},
            recent_experiments=recent,
            open_anomalies=self._audit_anomalies(),
            historical_failures=failures,
            seen_novelty_signatures=signatures,
            data_access={
                "locked_test_access": False,
                "locked_test_requires_one_time_approval": True,
                "known_dataset_fingerprints": fingerprints,
            },
            compute_budget_hours=float(compute_budget_hours),
        )
        context.validate()
        return context

    def _select_baseline(
        self,
        experiments: list[dict[str, Any]],
        baseline_experiment_id: str | None,
    ) -> dict[str, Any] | None:
        if baseline_experiment_id is not None:
            baseline = self.registry.get_experiment(baseline_experiment_id)
            if baseline is None:
                raise ContextBuildError(f"基线实验不存在: {baseline_experiment_id}")
            if baseline["status"] not in BASELINE_STATUSES:
                raise ContextBuildError(
                    f"基线实验尚未完成: {baseline_experiment_id} status={baseline['status']}"
                )
            if baseline["evaluation_decision"] not in {"KEEP", "EXTEND"}:
                raise ContextBuildError(
                    f"基线实验没有可继承决策: {baseline_experiment_id} "
                    f"decision={baseline['evaluation_decision']}"
                )
            return baseline
        for decision in ("KEEP", "EXTEND"):
            candidate = next(
                (
                    row
                    for row in experiments
                    if row["status"] in BASELINE_STATUSES and row["evaluation_decision"] == decision
                ),
                None,
            )
            if candidate is not None:
                return candidate
        return None

    def _metric_means(self, experiment_id: str) -> dict[str, float]:
        return {
            str(row["metric"]): float(row["mean_value"])
            for row in self.registry.average_metrics([experiment_id])
        }

    def _prediction_artifact(self, experiment_id: str) -> dict[str, Any] | None:
        artifacts = self.registry.get_artifacts(experiment_id, name="predictions")
        seed_zero = [row for row in artifacts if int(row["seed"]) == 0]
        return seed_zero[0] if len(seed_zero) == 1 else None

    def _prediction_path(self, experiment_id: str) -> str | None:
        artifact = self._prediction_artifact(experiment_id)
        return str(artifact["path"]) if artifact is not None else None

    def _experiment_summary(self, experiment: dict[str, Any]) -> dict[str, Any]:
        metric_means = self._metric_means(str(experiment["experiment_id"]))
        primary_names = _primary_metric_names(experiment)
        return {
            "experiment_id": experiment["experiment_id"],
            "parent_id": experiment["parent_id"],
            "hypothesis": experiment["hypothesis"],
            "experiment_type": experiment["experiment_type"],
            "executor": experiment["executor"],
            "status": experiment["status"],
            "evaluation_decision": experiment["evaluation_decision"],
            "error": experiment["error"],
            "dataset_fingerprint": experiment["dataset_fingerprint"],
            "novelty_signature": _novelty_signature(experiment),
            "primary_metrics": {
                name: metric_means[name] for name in primary_names if name in metric_means
            },
        }

    def _baseline_summary(self, experiment: dict[str, Any]) -> dict[str, Any]:
        summary = self._experiment_summary(experiment)
        summary["metrics"] = self._metric_means(str(experiment["experiment_id"]))
        predictions_path = self._prediction_path(str(experiment["experiment_id"]))
        if predictions_path is not None:
            summary["predictions_path"] = predictions_path
        return summary

    def _failure_summary(self, experiment: dict[str, Any]) -> dict[str, Any]:
        failed_gates: list[dict[str, Any]] = []
        if experiment["evaluation_decision"] == "DISCARD":
            evaluation = next(
                (
                    row
                    for row in self.registry.get_reviews(str(experiment["experiment_id"]))
                    if row["review_type"] == "evaluation"
                ),
                None,
            )
            if evaluation is not None:
                payload = _json_object(
                    evaluation["payload_json"],
                    label=f"实验 {experiment['experiment_id']} evaluation review",
                )
                gates = payload.get("gates", [])
                if isinstance(gates, list):
                    failed_gates = [
                        gate for gate in gates if isinstance(gate, dict) and not gate.get("passed")
                    ]
        reason = str(experiment["error"] or "").strip()
        if not reason and experiment["evaluation_decision"] == "DISCARD":
            reason = "Evaluation=DISCARD"
        return {
            "experiment_id": experiment["experiment_id"],
            "parent_id": experiment["parent_id"],
            "status": experiment["status"],
            "evaluation_decision": experiment["evaluation_decision"],
            "hypothesis": experiment["hypothesis"],
            "novelty_signature": _novelty_signature(experiment),
            "reason": reason or "未记录失败原因",
            "failed_gates": failed_gates,
        }

    def _audit_anomalies(self) -> list[dict[str, Any]]:
        anomalies: list[dict[str, Any]] = []
        reviews = self.registry.list_reviews(review_type="audit_anomalies", limit=None)
        for review in reviews:
            payload = _json_object(
                review["payload_json"],
                label=f"实验 {review['experiment_id']} audit_anomalies review",
            )
            values = payload.get("anomalies", [])
            if not isinstance(values, list):
                raise ContextBuildError(
                    f"实验 {review['experiment_id']} audit_anomalies 必须为列表"
                )
            prediction_artifact = self._prediction_artifact(str(review["experiment_id"]))
            for value in values:
                if not isinstance(value, dict):
                    raise ContextBuildError(f"实验 {review['experiment_id']} anomaly 必须为对象")
                anomaly = {
                    "type": str(value.get("type", "unknown")),
                    "severity": str(value.get("severity", "unknown")),
                    "detail": str(value.get("detail", "")),
                    "source_experiment_id": review["experiment_id"],
                    "source_status": review["experiment_status"],
                    "source_evaluation_decision": review["experiment_evaluation_decision"],
                }
                if prediction_artifact is not None:
                    anomaly.update(
                        {
                            "predictions_path": str(prediction_artifact["path"]),
                            "prediction_source_seed": int(prediction_artifact["seed"]),
                            "prediction_artifact_name": str(prediction_artifact["name"]),
                        }
                    )
                anomalies.append(anomaly)
                if len(anomalies) >= self.anomaly_limit:
                    return anomalies
        return anomalies
