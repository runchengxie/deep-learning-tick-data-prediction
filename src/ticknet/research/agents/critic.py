"""Critic Agent：审查 Brainstorm 提案的逻辑、可证伪性、重复与泄漏风险。

对应 AgentX 论文"Critic 是聪明但不可靠的"——这里 Critic 负责质量审查，
但最终否决权仍在 Policy（笨但绝对可靠）。Critic 返回结构化评审，不直接
拦截执行；Policy 才会。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ticknet.research.agents.client import LLMClient
from ticknet.research.agents.context import ResearchContext
from ticknet.research.spec import ExperimentSpec

REQUIRED_FIELDS = (
    "hypothesis",
    "objective",
    "falsification_condition",
    "experiment_type",
    "executor",
    "novelty_signature",
)


@dataclass
class Critique:
    """一次提案评审结果。"""

    approved: bool
    score: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "score": self.score,
            "issues": self.issues,
        }


class CriticAgent:
    """提案审查 Agent。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def review(
        self,
        spec: ExperimentSpec,
        context: ResearchContext | None = None,
    ) -> Critique:
        issues: list[str] = []
        for field_name in REQUIRED_FIELDS:
            if not getattr(spec, field_name).strip():
                issues.append(f"缺少必填字段: {field_name}")
        if len(spec.falsification_condition.strip()) < 8:
            issues.append("falsification_condition 应包含具体的可证伪条件")
        if not spec.rationale.strip():
            issues.append("缺少 rationale，无法判断机制")
        semantic_executors = {
            "data_audit": {"audit_predictions"},
            "cost_analysis": {"topk_cost_sweep"},
            "prediction_export": {"export_predictions"},
            "comparison": {"compare_experiments"},
            "robustness": {"walk_forward_robustness"},
        }
        allowed = semantic_executors.get(spec.experiment_type)
        if allowed is not None and spec.executor not in allowed:
            issues.append(
                f"{spec.experiment_type} 必须使用 {sorted(allowed)}，收到 {spec.executor}"
            )
        if context is not None:
            try:
                context.validate()
            except ValueError as error:
                issues.append(f"ResearchContext 无效: {error}")
            if spec.experiment_type not in set(context.allowed_actions):
                issues.append(f"上下文不允许 experiment_type: {spec.experiment_type}")
            if spec.executor not in set(context.available_executors):
                issues.append(f"上下文中 executor 不可用: {spec.executor}")
            if spec.novelty_signature in set(context.seen_novelty_signatures):
                issues.append(f"novelty_signature 已存在: {spec.novelty_signature}")
            if spec.budget.timeout_seconds > context.compute_budget_hours * 3600:
                issues.append("实验 timeout 超过 ResearchContext 算力预算")
        try:
            spec.validate()
        except ValueError as error:
            issues.append(f"ExperimentSpec 无效: {error}")
        score = max(0.0, 1.0 - 0.2 * len(issues))
        return Critique(approved=not issues, score=score, issues=issues)
