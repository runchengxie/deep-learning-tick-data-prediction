"""自动量化研究闭环（AgentX 式实验基础设施）。

确定性部分：ExperimentSpec / Policy / Protocol / Runner / Registry / Audit /
Locked Test。第一版不含 LLM Agent；未来的 Brainstorm Agent 通过生成
ExperimentSpec 接入。
"""

from ticknet.research.audit import AuditReport, PredictionTable, audit_predictions
from ticknet.research.comparison import (
    ComparisonError,
    compare_registered_experiments,
    summarize_walk_forward,
)
from ticknet.research.evaluation import (
    EvaluationResult,
    evaluate_metric_gates,
    flatten_numeric_metrics,
)
from ticknet.research.locked import (
    LockedTestApproval,
    LockedTestFailed,
    LockedTestNotApproved,
    issue_locked_test_approval,
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
from ticknet.research.registry import ExperimentRegistry, RegistryConflict
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import (
    DETERMINISTIC_EXECUTORS,
    EXECUTOR_EXPERIMENT_TYPES,
    EXECUTORS,
    ExecutionBudget,
    ExperimentResult,
    ExperimentSpec,
    MetricGate,
)

__all__ = [
    "DETERMINISTIC_EXECUTORS",
    "EXECUTORS",
    "EXECUTOR_EXPERIMENT_TYPES",
    "AuditReport",
    "ComparisonError",
    "CostModel",
    "EvaluationResult",
    "ExecutionBudget",
    "ExperimentRegistry",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "LockedTestApproval",
    "LockedTestFailed",
    "LockedTestNotApproved",
    "MetricGate",
    "PolicyViolation",
    "PortfolioEvaluation",
    "PortfolioPolicy",
    "PortfolioPrediction",
    "PredictionTable",
    "RegistryConflict",
    "ResearchPolicy",
    "ResearchProtocol",
    "RunnerError",
    "audit_predictions",
    "compare_registered_experiments",
    "evaluate_metric_gates",
    "evaluate_topk_portfolio",
    "flatten_numeric_metrics",
    "issue_locked_test_approval",
    "load_portfolio_predictions",
    "run_locked_test",
    "summarize_walk_forward",
    "write_portfolio_artifacts",
]
