"""实验定义：实验输入（ExperimentSpec）与实验输出（ExperimentResult）。

``ExperimentSpec`` 是研究闭环中所有下游（Runner、Registry、未来的 Brainstorm Agent）
共同使用的契约。Agent 或人工都不直接改训练配置，而是提交结构化提案，
由 ``policy.py`` 校验后由 ``runner.py`` 执行。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

EXPERIMENT_TYPES = frozenset(
    {
        "data_audit",
        "robustness",
        "ablation",
        "baseline",
        "feature_addition",
        "architecture",
        "cost_analysis",
    }
)


@dataclass(frozen=True)
class ExperimentSpec:
    """一次受控实验的完整定义。"""

    hypothesis: str
    experiment_type: str
    base_config: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = (0,)
    primary_metric: str = "daily_rank_ic_mean"
    expected_direction: str = "increase"
    rationale: str = ""
    falsification_condition: str = ""
    parent_id: str | None = None
    stage: str = "screening"
    entry_point: str = ""

    def validate(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("hypothesis 不能为空")
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"experiment_type 应为 {sorted(EXPERIMENT_TYPES)} 之一")
        if not self.base_config.strip():
            raise ValueError("base_config 不能为空")
        if not self.seeds:
            raise ValueError("seeds 不能为空")
        if self.expected_direction not in {"increase", "decrease"}:
            raise ValueError("expected_direction 应为 increase 或 decrease")
        if self.primary_metric not in {
            "daily_rank_ic_mean",
            "macro_f1",
            "mcc",
            "balanced_accuracy",
            "brier_score",
        }:
            raise ValueError(f"不支持的 primary_metric: {self.primary_metric}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    """一次实验的运行结果与登记信息。"""

    experiment_id: str
    spec: ExperimentSpec
    status: str
    git_sha: str
    dataset_fingerprint: str | None
    per_seed_metrics: list[dict[str, Any]]
    artifact_dir: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spec"] = self.spec.to_dict()
        return data
