"""研究 Agent（第一版：确定性模板 + 可插拔 LLM）。"""

from ticknet.research.agents.brainstorm import BrainstormAgent
from ticknet.research.agents.client import LLMClient, TemplateClient, make_client
from ticknet.research.agents.context import ResearchContext
from ticknet.research.agents.critic import CriticAgent
from ticknet.research.agents.orchestrator import (
    ResearchOrchestrator,
    ResearchStepResult,
)

__all__ = [
    "BrainstormAgent",
    "CriticAgent",
    "LLMClient",
    "ResearchContext",
    "ResearchOrchestrator",
    "ResearchStepResult",
    "TemplateClient",
    "make_client",
]
