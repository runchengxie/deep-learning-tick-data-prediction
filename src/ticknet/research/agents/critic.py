"""Critic Agent：审查 Brainstorm 提案的逻辑、可证伪性、重复与泄漏风险。

对应 AgentX 论文"Critic 是聪明但不可靠的"——这里 Critic 负责质量审查，
但最终否决权仍在 Policy（笨但绝对可靠）。Critic 返回结构化评审，不直接
拦截执行；Policy 才会。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ticknet.research.agents.client import LLMClient
from ticknet.research.spec import ExperimentSpec

REQUIRED_FIELDS = ("hypothesis", "falsification_condition", "experiment_type")


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

    def review(self, spec: ExperimentSpec) -> Critique:
        issues: list[str] = []
        for field_name in REQUIRED_FIELDS:
            if not getattr(spec, field_name).strip():
                issues.append(f"缺少必填字段: {field_name}")
        if "falsification" not in spec.falsification_condition and not spec.falsification_condition:
            issues.append("falsification_condition 应包含明确的可证伪条件")
        if spec.expected_direction not in {"increase", "decrease"}:
            issues.append(f"expected_direction 无效: {spec.expected_direction}")
        if not spec.rationale.strip():
            issues.append("缺少 rationale，无法判断机制")
        score = max(0.0, 1.0 - 0.2 * len(issues))
        return Critique(approved=not issues, score=score, issues=issues)

    def review_with_llm(self, spec: ExperimentSpec, context_text: str) -> Critique:
        """LLM 增强评审（可选）：让模型补充逻辑/机制审查。"""
        if self.llm is None:
            return self.review(spec)
        prompt = (
            "审查以下量化实验提案，指出逻辑漏洞、混淆变量或更便宜的替代实验。"
            f"\n提案：{spec.to_dict()}\n上下文：{context_text}"
        )
        output = self.llm.generate("你是严格的量化研究评审。", prompt)
        issues = [line for line in output.splitlines() if line.strip()]
        score = max(0.0, 1.0 - 0.2 * len(issues))
        return Critique(approved=len(issues) <= 2, score=score, issues=issues)
