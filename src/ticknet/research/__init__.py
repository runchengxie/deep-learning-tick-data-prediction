"""自动量化研究闭环（AgentX 式实验基础设施）。

确定性部分：ExperimentSpec / Policy / Protocol / Runner / Registry / Audit /
Locked Test。第一版不含 LLM Agent；未来的 Brainstorm Agent 通过生成
ExperimentSpec 接入。
"""

from ticknet.research.audit import AuditReport, PredictionTable, audit_predictions
from ticknet.research.locked import (
    LockedTestApproval,
    LockedTestNotApproved,
    run_locked_test,
)
from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.portfolio import (
    CostModel,
    PortfolioEvaluation,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
    load_portfolio_predictions,
    write_portfolio_artifacts,
)
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentResult, ExperimentSpec

__all__ = [
    "AuditReport",
    "CostModel",
    "ExperimentRegistry",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "LockedTestApproval",
    "LockedTestNotApproved",
    "PolicyViolation",
    "PortfolioEvaluation",
    "PortfolioPolicy",
    "PortfolioPrediction",
    "PredictionTable",
    "ResearchPolicy",
    "ResearchProtocol",
    "RunnerError",
    "audit_predictions",
    "evaluate_topk_portfolio",
    "load_portfolio_predictions",
    "run_locked_test",
    "write_portfolio_artifacts",
]
