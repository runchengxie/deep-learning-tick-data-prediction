"""AgentX 确定性研究闭环使用的版本化实验契约。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

EXPERIMENT_TYPES = frozenset(
    {
        "data_audit",
        "robustness",
        "ablation",
        "baseline",
        "feature_addition",
        "architecture",
        "cost_analysis",
        "prediction_export",
        "comparison",
    }
)

EXECUTORS = frozenset(
    {
        "train_nextday",
        "train_minute_tcn",
        "train_ranker",
        "export_predictions",
        "audit_predictions",
        "topk_cost_sweep",
        "walk_forward_robustness",
        "compare_experiments",
    }
)

DETERMINISTIC_EXECUTORS = frozenset(
    {
        "export_predictions",
        "audit_predictions",
        "topk_cost_sweep",
        "walk_forward_robustness",
        "compare_experiments",
    }
)

IMPLEMENTED_EXECUTORS = frozenset(
    {
        "train_nextday",
        "train_minute_tcn",
        *DETERMINISTIC_EXECUTORS,
    }
)

EXECUTOR_EXPERIMENT_TYPES = {
    "export_predictions": "prediction_export",
    "audit_predictions": "data_audit",
    "topk_cost_sweep": "cost_analysis",
    "walk_forward_robustness": "robustness",
    "compare_experiments": "comparison",
}

GateOperator = Literal["gt", "gte", "lt", "lte"]


@dataclass(frozen=True)
class MetricGate:
    """可计算的成功或证伪条件。"""

    metric: str
    operator: GateOperator
    threshold: float

    def validate(self) -> None:
        if not self.metric.strip():
            raise ValueError("gate metric 不能为空")
        if self.operator not in {"gt", "gte", "lt", "lte"}:
            raise ValueError(f"不支持的 gate operator: {self.operator}")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MetricGate:
        allowed = {"metric", "operator", "threshold"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"success_gates 包含未知字段: {sorted(unknown)}")
        try:
            gate = cls(
                metric=str(values["metric"]),
                operator=cast(GateOperator, str(values["operator"])),
                threshold=float(values["threshold"]),
            )
        except KeyError as error:
            raise ValueError(f"success_gates 缺少字段: {error.args[0]}") from error
        gate.validate()
        return gate


@dataclass(frozen=True)
class ExecutionBudget:
    """单实验的硬预算，由 Runner 强制执行。"""

    timeout_seconds: int = 3600
    max_seeds: int = 3

    def validate(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("budget.timeout_seconds 必须为正整数")
        if self.max_seeds <= 0:
            raise ValueError("budget.max_seeds 必须为正整数")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExecutionBudget:
        allowed = {"timeout_seconds", "max_seeds"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"budget 包含未知字段: {sorted(unknown)}")
        budget = cls(
            timeout_seconds=int(values.get("timeout_seconds", 3600)),
            max_seeds=int(values.get("max_seeds", 3)),
        )
        budget.validate()
        return budget


@dataclass(frozen=True)
class ExperimentSpec:
    """一次实验的白名单执行器、输入、门禁、产物和预算契约。"""

    hypothesis: str
    experiment_type: str
    executor: str
    base_config: str = ""
    objective: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = (0,)
    primary_metrics: tuple[str, ...] = ("validation.daily_rank_ic_mean",)
    success_gates: tuple[MetricGate, ...] = ()
    artifact_contract: tuple[str, ...] = (
        "resolved_spec",
        "resolved_config",
        "stdout",
        "stderr",
        "result",
        "run_manifest",
    )
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    rationale: str = ""
    falsification_condition: str = ""
    parent_id: str | None = None
    novelty_signature: str = ""
    stage: str = "screening"

    def validate(self) -> None:
        self._validate_identity()
        self._validate_execution()
        self._validate_metrics()
        self._validate_artifacts()

    def _validate_identity(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("hypothesis 不能为空")
        if not (self.objective or self.hypothesis).strip():
            raise ValueError("objective 不能为空")
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"experiment_type 应为 {sorted(EXPERIMENT_TYPES)} 之一")
        if self.executor not in EXECUTORS:
            raise ValueError(f"executor 应为 {sorted(EXECUTORS)} 之一")
        expected_type = EXECUTOR_EXPERIMENT_TYPES.get(self.executor)
        if expected_type is not None and self.experiment_type != expected_type:
            raise ValueError(f"{self.executor} 必须使用 experiment_type={expected_type}")
        if not self.novelty_signature.strip():
            raise ValueError("novelty_signature 不能为空")
        if self.stage not in {"screening", "robustness", "release"}:
            raise ValueError("stage 应为 screening、robustness 或 release")

    def _validate_execution(self) -> None:
        if self.executor.startswith("train_") and not self.base_config.strip():
            raise ValueError(f"{self.executor} 需要 base_config")
        self._validate_seeds_and_budget()
        self._validate_executor_inputs()

    def _validate_seeds_and_budget(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds 必须非空且不能重复")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds 不能为负数")
        self.budget.validate()
        if len(self.seeds) > self.budget.max_seeds:
            raise ValueError("seeds 超过 budget.max_seeds")
        if self.executor in DETERMINISTIC_EXECUTORS and self.seeds != (0,):
            raise ValueError(f"{self.executor} 是确定性执行器，只允许 seed 0")

    def _validate_executor_inputs(self) -> None:
        if (
            self.executor == "audit_predictions"
            and not str(self.inputs.get("predictions_path", "")).strip()
        ):
            raise ValueError("audit_predictions 需要 inputs.predictions_path")
        if self.executor == "export_predictions":
            self._validate_export_inputs()
        if self.executor == "topk_cost_sweep":
            self._validate_topk_inputs()
        if self.executor in {"walk_forward_robustness", "compare_experiments"}:
            self._validate_comparison_inputs()

    def _validate_export_inputs(self) -> None:
        source_experiment_id = self.inputs.get("source_experiment_id")
        if not isinstance(source_experiment_id, str) or not source_experiment_id.strip():
            raise ValueError("export_predictions 需要 inputs.source_experiment_id")
        artifact_name = self.inputs.get("artifact_name", "predictions")
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError("inputs.artifact_name 不能为空")
        source_seed = self.inputs.get("source_seed", 0)
        if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
            raise ValueError("inputs.source_seed 必须为非负整数")

    def _validate_topk_inputs(self) -> None:
        has_experiment = self._validate_topk_source()
        top_ks, buffers, costs = self._validate_topk_grids()
        self._validate_topk_diagnostic_options(top_ks=top_ks, costs=costs)
        require_tradability, missing_policy, mode, target_contract = (
            self._validate_topk_contract_options()
        )
        if mode == "formal":
            self._validate_formal_topk_inputs(
                has_experiment=has_experiment,
                require_tradability=require_tradability,
                missing_policy=missing_policy,
                target_contract=target_contract,
                buffers=buffers,
            )

    def _validate_topk_source(self) -> bool:
        predictions_path = self.inputs.get("predictions_path")
        source_experiment_id = self.inputs.get("source_experiment_id")
        has_path = isinstance(predictions_path, str) and bool(predictions_path.strip())
        has_experiment = isinstance(source_experiment_id, str) and bool(
            source_experiment_id.strip()
        )
        if has_path == has_experiment:
            raise ValueError(
                "topk_cost_sweep 必须且只能提供 inputs.predictions_path 或 "
                "inputs.source_experiment_id"
            )
        if predictions_path is not None and not has_path:
            raise ValueError("inputs.predictions_path 必须为非空字符串")
        if source_experiment_id is not None and not has_experiment:
            raise ValueError("inputs.source_experiment_id 必须为非空字符串")
        if has_experiment:
            self._validate_registered_source_options()
        return has_experiment

    def _validate_topk_grids(
        self,
    ) -> tuple[list[int | float], list[int | float], list[int | float]]:
        top_ks = self._validate_numeric_grid("top_k", [25, 50, 75, 100], integer=True)
        buffers = self._validate_numeric_grid("exit_buffer", [0, 10, 25, 50], integer=True)
        costs = self._validate_numeric_grid("cost_bps", [5, 10, 15, 20], integer=False)
        if any(value <= 0 for value in top_ks):
            raise ValueError("inputs.top_k 只能包含正整数")
        if any(value < 0 for value in buffers):
            raise ValueError("inputs.exit_buffer 不能包含负数")
        if any(value < 0 for value in costs):
            raise ValueError("inputs.cost_bps 不能包含负数")
        return top_ks, buffers, costs

    def _validate_topk_diagnostic_options(
        self,
        *,
        top_ks: list[int | float],
        costs: list[int | float],
    ) -> None:
        default_cost = 10.0 if 10.0 in costs else float(costs[0])
        decision_cost = self._validate_nonnegative_number("decision_cost_bps", default_cost)
        if float(decision_cost) not in {float(value) for value in costs}:
            raise ValueError("inputs.decision_cost_bps 必须包含在 cost_bps 网格中")
        minimum = self.inputs.get("min_symbols_per_day", max(int(value) for value in top_ks))
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < max(top_ks):
            raise ValueError("inputs.min_symbols_per_day 必须为不小于最大 top_k 的整数")
        minimum_dates = self.inputs.get("minimum_evaluated_dates", 60)
        if (
            isinstance(minimum_dates, bool)
            or not isinstance(minimum_dates, int)
            or minimum_dates <= 0
        ):
            raise ValueError("inputs.minimum_evaluated_dates 必须为正整数")
        for name, default in {
            "minimum_positive_month_ratio": 0.5,
            "maximum_top5_absolute_contribution": 0.5,
        }.items():
            value = self._validate_nonnegative_number(name, default)
            if value > 1:
                raise ValueError(f"inputs.{name} 必须在 [0, 1] 范围内")
        self._validate_nonnegative_number("min_score_gap", 0.0)
        self._validate_nonnegative_number("sell_stamp_tax_bps", 5.0)

    def _validate_topk_contract_options(self) -> tuple[bool, str, str, str]:
        require_tradability = self.inputs.get("require_tradability", False)
        if not isinstance(require_tradability, bool):
            raise ValueError("inputs.require_tradability 必须为布尔值")
        missing_policy = self.inputs.get("missing_holding_policy", "liquidate")
        if missing_policy not in {"liquidate", "error"}:
            raise ValueError("inputs.missing_holding_policy 应为 liquidate 或 error")
        mode = self.inputs.get("evaluation_mode", "smoke")
        if mode not in {"smoke", "formal"}:
            raise ValueError("inputs.evaluation_mode 应为 smoke 或 formal")
        target_contract = self.inputs.get("target_return_contract", "unspecified")
        if not isinstance(target_contract, str) or not target_contract.strip():
            raise ValueError("inputs.target_return_contract 必须为非空字符串")
        return require_tradability, str(missing_policy), str(mode), target_contract

    def _validate_registered_source_options(self) -> None:
        artifact_name = self.inputs.get("artifact_name", "predictions")
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError("inputs.artifact_name 不能为空")
        source_seed = self.inputs.get("source_seed", 0)
        if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
            raise ValueError("inputs.source_seed 必须为非负整数")

    def _validate_numeric_grid(
        self,
        name: str,
        default: list[int],
        *,
        integer: bool,
    ) -> list[int | float]:
        values = self.inputs.get(name, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"inputs.{name} 必须为非空列表")
        if integer:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"inputs.{name} 只能包含整数")
            normalized: list[int | float] = [cast(int, value) for value in values]
        else:
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"inputs.{name} 只能包含有限数值")
            normalized = [float(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"inputs.{name} 不能重复")
        return normalized

    def _validate_nonnegative_number(self, name: str, default: float) -> float:
        value = self.inputs.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"inputs.{name} 必须为非负数")
        return float(value)

    def _validate_formal_topk_inputs(
        self,
        *,
        has_experiment: bool,
        require_tradability: bool,
        missing_policy: str,
        target_contract: str,
        buffers: list[int | float],
    ) -> None:
        if not has_experiment:
            raise ValueError("formal topk_cost_sweep 必须使用 Registry source_experiment_id")
        if not require_tradability:
            raise ValueError("formal topk_cost_sweep 必须设置 require_tradability=true")
        if missing_policy != "error":
            raise ValueError("formal topk_cost_sweep 必须设置 missing_holding_policy=error")
        if target_contract != "next_open_to_following_open":
            raise ValueError(
                "formal topk_cost_sweep 的 target_return_contract 必须为 "
                "next_open_to_following_open"
            )
        if 0 not in buffers:
            raise ValueError("formal topk_cost_sweep 的 exit_buffer 必须包含 0 作为对照")

    def _validate_comparison_inputs(self) -> None:
        experiment_ids = self.inputs.get("experiment_ids")
        if not isinstance(experiment_ids, list) or len(experiment_ids) < 2:
            raise ValueError(f"{self.executor} 需要至少两个 inputs.experiment_ids")
        if any(not isinstance(value, str) or not value.strip() for value in experiment_ids):
            raise ValueError("inputs.experiment_ids 只能包含非空字符串")
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("inputs.experiment_ids 不能重复")
        typed_ids = cast(list[str], experiment_ids)
        self._validate_metric_selection()
        if self.executor == "compare_experiments":
            self._validate_compare_options(typed_ids)
        if self.executor == "walk_forward_robustness":
            self._validate_walk_forward_options(typed_ids)

    def _validate_metric_selection(self) -> None:
        metrics = self.inputs.get("metrics")
        if metrics is not None and (
            not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(value, str) or not value.strip() for value in metrics)
        ):
            raise ValueError("inputs.metrics 必须为非空字符串列表")
        if isinstance(metrics, list) and len(set(metrics)) != len(metrics):
            raise ValueError("inputs.metrics 不能重复")
        selected_metrics = (
            set(cast(list[str], metrics))
            if isinstance(metrics, list)
            else set(self.primary_metrics)
        )
        metric_directions = self.inputs.get("metric_directions", {})
        if not isinstance(metric_directions, dict):
            raise ValueError("inputs.metric_directions 必须为对象")
        if any(not isinstance(key, str) for key in metric_directions):
            raise ValueError("inputs.metric_directions 的键必须为指标字符串")
        typed_directions = cast(dict[str, Any], metric_directions)
        unknown_directions = set(typed_directions) - selected_metrics
        if unknown_directions:
            raise ValueError(
                f"inputs.metric_directions 包含未比较指标: {sorted(unknown_directions)}"
            )
        invalid_directions = {
            str(key): value
            for key, value in typed_directions.items()
            if value not in {"higher", "lower"}
        }
        if invalid_directions:
            raise ValueError("inputs.metric_directions 只能使用 higher 或 lower")

    def _validate_compare_options(self, experiment_ids: list[str]) -> None:
        baseline_id = self.inputs.get("baseline_id", experiment_ids[0])
        if not isinstance(baseline_id, str) or baseline_id not in set(experiment_ids):
            raise ValueError("inputs.baseline_id 必须包含在 experiment_ids 中")
        require_same = self.inputs.get("require_same_fingerprint", True)
        if not isinstance(require_same, bool):
            raise ValueError("inputs.require_same_fingerprint 必须为布尔值")

    def _validate_walk_forward_options(self, experiment_ids: list[str]) -> None:
        minimum_windows = self.inputs.get("minimum_windows", 3)
        if (
            isinstance(minimum_windows, bool)
            or not isinstance(minimum_windows, int)
            or minimum_windows < 2
        ):
            raise ValueError("inputs.minimum_windows 必须为至少 2 的整数")
        if minimum_windows > len(experiment_ids):
            raise ValueError("inputs.minimum_windows 不能超过 experiment_ids 数量")
        require_distinct = self.inputs.get("require_distinct_fingerprints", True)
        if not isinstance(require_distinct, bool):
            raise ValueError("inputs.require_distinct_fingerprints 必须为布尔值")

    def _validate_metrics(self) -> None:
        if not self.primary_metrics or any(not metric.strip() for metric in self.primary_metrics):
            raise ValueError("primary_metrics 必须非空")
        if not self.success_gates:
            raise ValueError("success_gates 必须包含至少一个可计算 gate")
        for gate in self.success_gates:
            gate.validate()
        if not self.falsification_condition.strip():
            raise ValueError("falsification_condition 不能为空")

    def _validate_artifacts(self) -> None:
        if not self.artifact_contract or len(set(self.artifact_contract)) != len(
            self.artifact_contract
        ):
            raise ValueError("artifact_contract 必须非空且不能重复")
        if any(not name.strip() for name in self.artifact_contract):
            raise ValueError("artifact_contract 名称不能为空")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentSpec:
        """严格解析 YAML/LLM JSON，未知字段和旧 entry_point 都会被拒绝。"""
        allowed = {
            "hypothesis",
            "experiment_type",
            "executor",
            "base_config",
            "objective",
            "inputs",
            "config_overrides",
            "seeds",
            "primary_metrics",
            "success_gates",
            "artifact_contract",
            "budget",
            "rationale",
            "falsification_condition",
            "parent_id",
            "novelty_signature",
            "stage",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"ExperimentSpec 包含未知字段: {sorted(unknown)}")
        try:
            gates_raw = raw["success_gates"]
            if not isinstance(gates_raw, list):
                raise ValueError("success_gates 应为列表")
            budget_raw = raw.get("budget", {})
            if not isinstance(budget_raw, dict):
                raise ValueError("budget 应为对象")
            inputs = raw.get("inputs", {})
            overrides = raw.get("config_overrides", {})
            if not isinstance(inputs, dict) or not isinstance(overrides, dict):
                raise ValueError("inputs 和 config_overrides 应为对象")
            spec = cls(
                hypothesis=str(raw["hypothesis"]),
                experiment_type=str(raw["experiment_type"]),
                executor=str(raw["executor"]),
                base_config=str(raw.get("base_config", "")),
                objective=str(raw.get("objective", "")),
                inputs=dict(inputs),
                config_overrides=dict(overrides),
                seeds=tuple(int(seed) for seed in raw.get("seeds", [0])),
                primary_metrics=tuple(
                    str(metric)
                    for metric in raw.get("primary_metrics", ["validation.daily_rank_ic_mean"])
                ),
                success_gates=tuple(MetricGate.from_dict(dict(gate)) for gate in gates_raw),
                artifact_contract=tuple(
                    str(name)
                    for name in raw.get(
                        "artifact_contract",
                        [
                            "resolved_spec",
                            "resolved_config",
                            "stdout",
                            "stderr",
                            "result",
                            "run_manifest",
                        ],
                    )
                ),
                budget=ExecutionBudget.from_dict(budget_raw),
                rationale=str(raw.get("rationale", "")),
                falsification_condition=str(raw.get("falsification_condition", "")),
                parent_id=(str(raw["parent_id"]) if raw.get("parent_id") is not None else None),
                novelty_signature=str(raw.get("novelty_signature", "")),
                stage=str(raw.get("stage", "screening")),
            )
        except KeyError as error:
            raise ValueError(f"ExperimentSpec 缺少字段: {error.args[0]}") from error
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """一次实验的运行结果、评估决策与登记信息。"""

    experiment_id: str
    spec: ExperimentSpec
    status: str
    git_sha: str
    dataset_fingerprint: str | None
    per_seed_metrics: list[dict[str, Any]]
    artifact_dir: str
    evaluation_decision: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spec"] = self.spec.to_dict()
        return data
