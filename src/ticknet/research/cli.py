"""ticknet-research：实验执行、审计、一次性 locked approval 与 Agent 编排。

确定性部分包括 run/show/compare/audit/approve-locked-test/locked-test；agent-step
用模板 Brainstorm（不接 LLM）推进一轮研究。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ticknet.research.agents.brainstorm import BrainstormAgent
from ticknet.research.agents.client import make_client
from ticknet.research.agents.context import ResearchContext
from ticknet.research.agents.critic import CriticAgent
from ticknet.research.agents.orchestrator import ResearchOrchestrator
from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.comparison import ComparisonError, compare_registered_experiments
from ticknet.research.locked import (
    LockedTestApproval,
    LockedTestFailed,
    LockedTestNotApproved,
    issue_locked_test_approval,
    run_locked_test,
)
from ticknet.research.policy import PolicyViolation
from ticknet.research.registry import ExperimentRegistry, RegistryConflict
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentSpec

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = DEFAULT_REPOSITORY_ROOT / "results" / "registry.sqlite"
DEFAULT_ARTIFACT_ROOT = DEFAULT_REPOSITORY_ROOT / "research" / "experiments"


def _load_spec(path: str | Path) -> ExperimentSpec:
    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise SystemExit("实验 YAML 根节点应为对象")
    try:
        return ExperimentSpec.from_dict(values)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"ExperimentSpec 无效: {error}") from error


def _run_command(args: argparse.Namespace) -> None:
    spec = _load_spec(args.spec)
    registry = ExperimentRegistry(args.registry)
    runner = ExperimentRunner(
        registry,
        repository_root=args.root,
        artifact_root=args.artifacts,
    )
    experiment_id = args.id or f"EXP-{Path(args.spec).stem}"
    try:
        try:
            result = runner.run(spec, experiment_id=experiment_id)
        except (PolicyViolation, RunnerError, ValueError) as error:
            raise SystemExit(f"EXPERIMENT_REJECTED: {error}") from error
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        registry.close()


def _show_command(args: argparse.Namespace) -> None:
    registry = ExperimentRegistry(args.registry)
    experiment = registry.get_experiment(args.id)
    if experiment is None:
        registry.close()
        raise SystemExit(f"找不到实验: {args.id}")
    print(json.dumps(experiment, ensure_ascii=False, indent=2, default=str))
    registry.close()


def _compare_command(args: argparse.Namespace) -> None:
    registry = ExperimentRegistry(args.registry)
    try:
        metrics = args.metrics
        if metrics is None:
            metric_sets = [
                {str(row["metric"]) for row in registry.get_metrics(experiment_id)}
                for experiment_id in args.ids
            ]
            metrics = sorted(set.intersection(*metric_sets)) if metric_sets else []
        try:
            comparison, fingerprint = compare_registered_experiments(
                registry,
                args.ids,
                metrics,
                baseline_id=args.baseline or args.ids[0],
                metric_directions=dict.fromkeys(args.lower_is_better, "lower"),
            )
        except ComparisonError as error:
            raise SystemExit(f"COMPARE_REJECTED: {error}") from error
        print(
            json.dumps(
                {
                    "comparison": comparison,
                    "dataset_fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        registry.close()


def _agent_step_command(args: argparse.Namespace) -> None:
    registry = ExperimentRegistry(args.registry)
    llm = make_client(args.provider)
    brainstorm = BrainstormAgent(
        llm,
        default_base_config=args.base_config,
    )
    critic = CriticAgent(llm)
    runner = ExperimentRunner(
        registry,
        repository_root=args.root,
        artifact_root=args.artifacts,
    )
    orchestrator = ResearchOrchestrator(
        registry,
        brainstorm=brainstorm,
        critic=critic,
        runner=runner,
    )
    context = ResearchContext(
        research_question=args.question,
        baseline_summary=({"predictions_path": args.predictions} if args.predictions else {}),
        open_anomalies=(
            [
                {
                    "type": anomaly_type,
                    "severity": "high",
                    "detail": f"{anomaly_type} 异常",
                }
                for anomaly_type in args.anomaly.split(";")
                if anomaly_type
            ]
            if args.anomaly
            else []
        ),
    )
    step = orchestrator.research_step(
        context,
        experiment_id=args.id or None,
    )
    print(json.dumps(step.to_dict(), ensure_ascii=False, indent=2))
    registry.close()


def _audit_command(args: argparse.Namespace) -> None:
    table = PredictionTable.from_parquet(args.predictions)
    report = audit_predictions(
        table,
        min_symbols_per_day=args.min_symbols_per_day,
        portfolio_quantile=args.quantile,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def _approve_locked_test_command(args: argparse.Namespace) -> None:
    registry = ExperimentRegistry(args.registry)
    try:
        try:
            issued = issue_locked_test_approval(
                args.predictions,
                registry=registry,
                experiment_id=args.id,
                reason=args.reason,
                approved_by=args.approved_by,
                checkpoint_artifact_name=args.checkpoint_artifact_name,
            )
        except (LockedTestNotApproved, RegistryConflict) as error:
            raise SystemExit(f"LOCKED_APPROVAL_REJECTED: {error}") from error
        print(json.dumps(issued, ensure_ascii=False, indent=2))
    finally:
        registry.close()


def _locked_test_command(args: argparse.Namespace) -> None:
    registry = ExperimentRegistry(args.registry)
    try:
        try:
            result = run_locked_test(
                args.predictions,
                approval=LockedTestApproval(token=args.token),
                registry=registry,
                experiment_id=args.id,
                min_symbols_per_day=args.min_symbols_per_day,
                portfolio_quantile=args.quantile,
            )
        except (LockedTestFailed, LockedTestNotApproved, RegistryConflict, ValueError) as error:
            raise SystemExit(f"LOCKED_TEST_REJECTED: {error}") from error
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        registry.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动量化研究实验入口")
    parser.add_argument("--root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行一个实验")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--id", default=None)
    run_parser.set_defaults(func=_run_command)

    show_parser = subparsers.add_parser("show", help="查看一个实验")
    show_parser.add_argument("--id", required=True)
    show_parser.set_defaults(func=_show_command)

    compare_parser = subparsers.add_parser("compare", help="对比多个实验")
    compare_parser.add_argument("--ids", nargs="+", required=True)
    compare_parser.add_argument("--baseline", default=None)
    compare_parser.add_argument("--metrics", nargs="+", default=None)
    compare_parser.add_argument(
        "--lower-is-better",
        nargs="+",
        default=[],
        metavar="METRIC",
        help="显式标记越低越好的比较指标；其余指标默认越高越好",
    )
    compare_parser.set_defaults(func=_compare_command)

    audit_parser = subparsers.add_parser("audit", help="审计一组预测明细")
    audit_parser.add_argument("--predictions", required=True)
    audit_parser.add_argument("--min-symbols-per-day", type=int, default=50)
    audit_parser.add_argument("--quantile", type=float, default=0.1)
    audit_parser.set_defaults(func=_audit_command)

    approval_parser = subparsers.add_parser(
        "approve-locked-test",
        help="为已冻结内容签发一次性 locked-test token",
    )
    approval_parser.add_argument("--predictions", required=True)
    approval_parser.add_argument("--id", required=True)
    approval_parser.add_argument("--reason", required=True)
    approval_parser.add_argument("--approved-by", required=True)
    approval_parser.add_argument("--checkpoint-artifact-name", default="best_checkpoint")
    approval_parser.set_defaults(func=_approve_locked_test_command)

    locked_parser = subparsers.add_parser("locked-test", help="消费一次性批准并评估锁定测试集")
    locked_parser.add_argument("--predictions", required=True)
    locked_parser.add_argument("--id", required=True)
    locked_parser.add_argument("--token", required=True)
    locked_parser.add_argument("--min-symbols-per-day", type=int, default=50)
    locked_parser.add_argument("--quantile", type=float, default=0.1)
    locked_parser.set_defaults(func=_locked_test_command)

    agent_parser = subparsers.add_parser(
        "agent-step", help="推进一轮研究（Brainstorm→Critic→Runner）"
    )
    agent_parser.add_argument("--question", default="分钟聚合特征是否包含稳定的次日横截面信息")
    agent_parser.add_argument("--anomaly", default="")
    agent_parser.add_argument("--predictions", default="")
    agent_parser.add_argument("--base-config", default="configs/nextday-pilot.yaml")
    agent_parser.add_argument(
        "--provider",
        choices=["template", "openai", "deepseek"],
        default="template",
    )
    agent_parser.add_argument("--id", default=None)
    agent_parser.set_defaults(func=_agent_step_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
