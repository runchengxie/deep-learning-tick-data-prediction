"""ticknet-research 命令行：run / show / compare / audit。

对应 AgentX 落地路线建议的四条命令。第一版只实现确定性的 run/show/compare，
audit 与 agent-step 留接口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ticknet.research.policy import PolicyViolation
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentSpec

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = DEFAULT_REPOSITORY_ROOT / "results" / "registry.sqlite"
DEFAULT_ARTIFACT_ROOT = DEFAULT_REPOSITORY_ROOT / "research" / "experiments"


def _load_spec(path: str | Path) -> ExperimentSpec:
    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise SystemExit("实验 YAML 根节点应为对象")
    seeds = tuple(int(seed) for seed in values.get("seeds", [0]))
    spec = ExperimentSpec(
        hypothesis=str(values["hypothesis"]),
        experiment_type=str(values["experiment_type"]),
        base_config=str(values["base_config"]),
        config_overrides=dict(values.get("config_overrides", {})),
        seeds=seeds,
        primary_metric=str(values.get("primary_metric", "daily_rank_ic_mean")),
        expected_direction=str(values.get("expected_direction", "increase")),
        rationale=str(values.get("rationale", "")),
        falsification_condition=str(values.get("falsification_condition", "")),
        parent_id=values.get("parent_id"),
        stage=str(values.get("stage", "screening")),
        entry_point=str(values.get("entry_point", "")),
    )
    return spec


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
        result = runner.run(spec, experiment_id=experiment_id)
    except PolicyViolation as error:
        registry.record_review(experiment_id, "policy", "rejected", {"reason": str(error)})
        raise SystemExit(f"POLICY_REJECTED: {error}") from error
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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
    rows = registry.list_experiments()
    wanted = set(args.ids)
    selected = [row for row in rows if row["experiment_id"] in wanted]
    if len(selected) != len(wanted):
        missing = sorted(wanted - {row["experiment_id"] for row in selected})
        registry.close()
        raise SystemExit(f"找不到全部实验: {missing}")
    metrics_rows = registry.average_metrics(list(wanted))
    print(f"{'experiment':<12} {'metric':<24} {'mean':>12}")
    for row in metrics_rows:
        print(f"{row['experiment_id']:<12} {row['metric']:<24} {row['mean_value']:>12.5f}")
    registry.close()


def _agent_step_command(args: argparse.Namespace) -> None:
    raise SystemExit("agent-step 尚未实现：先完成 Experiment Harness 闭环。")


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
    compare_parser.set_defaults(func=_compare_command)

    agent_parser = subparsers.add_parser("agent-step", help="推进一轮研究（未实现）")
    agent_parser.set_defaults(func=_agent_step_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
