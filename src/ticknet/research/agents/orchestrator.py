"""研究编排器（Research Orchestrator）：一轮完整的研究步骤。

对应 AgentX 论文闭环：Brainstorm → Critic → Policy → Runner → Audit → Registry。
第一版为确定性的单步执行；循环调用由外部 CLI 或脚本控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ticknet.research.agents.brainstorm import BrainstormAgent
from ticknet.research.agents.context import ResearchContext
from ticknet.research.agents.critic import CriticAgent
from ticknet.research.policy import PolicyViolation
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentSpec


@dataclass
class ResearchStepResult:
    """一轮研究步骤的结果。"""

    status: str
    spec: ExperimentSpec | None = None
    critique: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "spec": self.spec.to_dict() if self.spec else None,
            "critique": self.critique,
            "result": self.result,
            "error": self.error,
        }


class ResearchOrchestrator:
    """执行一轮研究步骤。"""

    def __init__(
        self,
        registry: ExperimentRegistry,
        *,
        brainstorm: BrainstormAgent,
        critic: CriticAgent,
        runner: ExperimentRunner,
    ) -> None:
        self.registry = registry
        self.brainstorm = brainstorm
        self.critic = critic
        self.runner = runner
        self._experiment_counter = 0

    def research_step(
        self,
        context: ResearchContext,
        *,
        experiment_id: str | None = None,
    ) -> ResearchStepResult:
        """执行一轮 Brainstorm → Critic → Policy → Runner。"""
        if experiment_id is None:
            self._experiment_counter += 1
            experiment_id = f"EXP-AUTO-{self._experiment_counter:04d}"

        try:
            spec = self.brainstorm.propose(context)
        except ValueError as error:
            return ResearchStepResult(status="brainstorm_failed", error=str(error))

        critique = self.critic.review(spec)
        self.registry.record_review(
            experiment_id,
            review_type="critic",
            decision="APPROVED" if critique.approved else "REJECTED",
            payload=critique.to_dict(),
        )
        if not critique.approved:
            return ResearchStepResult(
                status="critic_rejected",
                spec=spec,
                critique=critique.to_dict(),
            )

        try:
            result = self.runner.run(spec, experiment_id=experiment_id)
        except PolicyViolation as error:
            self.registry.record_review(
                experiment_id,
                review_type="policy",
                decision="rejected",
                payload={"reason": str(error)},
            )
            return ResearchStepResult(
                status="policy_rejected",
                spec=spec,
                critique=critique.to_dict(),
                error=str(error),
            )
        except RunnerError as error:
            self.registry.record_review(
                experiment_id,
                review_type="runner",
                decision="failed",
                payload={"reason": str(error)},
            )
            return ResearchStepResult(
                status="runner_failed",
                spec=spec,
                critique=critique.to_dict(),
                error=str(error),
            )
        return ResearchStepResult(
            status="completed",
            spec=spec,
            critique=critique.to_dict(),
            result={"experiment_id": experiment_id, "status": result.status},
        )
