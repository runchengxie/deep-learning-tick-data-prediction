"""自动量化研究闭环（AgentX 式实验基础设施）。

确定性部分：ExperimentSpec / Policy / Runner / Registry。
第一版不含 LLM Agent；未来的 Brainstorm Agent 通过生成 ExperimentSpec 接入。
"""

from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentResult, ExperimentSpec

__all__ = [
    "ExperimentRegistry",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "PolicyViolation",
    "ResearchPolicy",
    "RunnerError",
]
