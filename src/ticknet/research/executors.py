"""ExperimentSpec.executor 对应的白名单确定性执行器。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.portfolio import (
    CostModel,
    MissingHoldingPolicy,
    PortfolioPolicy,
    evaluate_topk_portfolio,
    load_portfolio_predictions,
    write_portfolio_artifacts,
)
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.spec import ExperimentSpec


class ExecutorFailure(RuntimeError):
    """执行器失败，同时保留 stdout、stderr 和退出码供 Runner 归档。"""

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


@dataclass(frozen=True)
class ExecutorContext:
    """执行器只能访问 Runner 明确提供的受控上下文。"""

    spec: ExperimentSpec
    seed: int
    repository_root: Path
    seed_dir: Path
    config_path: Path
    registry: ExperimentRegistry


@dataclass(frozen=True)
class ExecutorOutput:
    """所有 executor 统一返回的结果契约。"""

    metrics: dict[str, Any]
    artifacts: dict[str, Path] = field(default_factory=dict)
    dataset_fingerprint: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    anomalies: tuple[dict[str, Any], ...] = ()


class ResearchExecutor(Protocol):
    def execute(self, context: ExecutorContext) -> ExecutorOutput: ...


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _resolve_input_path(value: object, repository_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _extract_result(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"(?m)^\{", stdout):
        try:
            value, _end = decoder.raw_decode(stdout[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ExecutorFailure("stdout 中没有可解析的 JSON 结果", stdout=stdout)
    return candidates[-1]


def _find_artifacts(result: dict[str, Any], repository_root: Path) -> dict[str, Path]:
    field_names = {
        "result_file": "training_result",
        "best_checkpoint": "best_checkpoint",
        "last_checkpoint": "last_checkpoint",
        "history": "history",
        "predictions": "predictions",
        "predictions_path": "predictions",
    }
    artifacts: dict[str, Path] = {}
    for field_name, artifact_name in field_names.items():
        value = result.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        path = _resolve_input_path(value, repository_root)
        if path.is_file():
            artifacts[artifact_name] = path
    return artifacts


@dataclass(frozen=True)
class CommandExecutor:
    """固定命令的训练执行器；命令不来自 ExperimentSpec。"""

    command: tuple[str, ...]

    def _resolved_command(self, repository_root: Path) -> list[str]:
        if not self.command:
            raise ExecutorFailure("命令执行器缺少命令")
        entry = self.command[0]
        candidate = Path(entry)
        if candidate.is_absolute() and candidate.is_file():
            resolved = str(candidate)
        else:
            local_candidates = (
                repository_root / ".venv" / "bin" / entry,
                repository_root / ".venv" / "Scripts" / entry,
            )
            local = next((path for path in local_candidates if path.is_file()), None)
            resolved = str(local) if local is not None else (shutil.which(entry) or "")
        if not resolved:
            raise ExecutorFailure(f"找不到固定命令入口: {entry}")
        return [resolved, *self.command[1:]]

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        command = [
            *self._resolved_command(context.repository_root),
            "--config",
            str(context.config_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=context.repository_root,
                capture_output=True,
                text=True,
                timeout=context.spec.budget.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ExecutorFailure(
                f"executor 超过 {context.spec.budget.timeout_seconds} 秒预算",
                stdout=_timeout_text(error.stdout),
                stderr=_timeout_text(error.stderr),
                exit_code=124,
            ) from error
        if completed.returncode != 0:
            raise ExecutorFailure(
                f"固定命令执行失败，退出码 {completed.returncode}",
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
            )
        result = _extract_result(completed.stdout)
        return ExecutorOutput(
            metrics=result,
            artifacts=_find_artifacts(result, context.repository_root),
            dataset_fingerprint=(
                str(result["dataset_fingerprint"])
                if result.get("dataset_fingerprint") is not None
                else None
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )


class AuditPredictionsExecutor:
    """读取预测明细并生成 audit.json，不启动训练。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        predictions = _resolve_input_path(
            context.spec.inputs["predictions_path"], context.repository_root
        )
        report = audit_predictions(
            PredictionTable.from_parquet(predictions),
            min_symbols_per_day=int(context.spec.inputs.get("min_symbols_per_day", 50)),
            portfolio_quantile=float(context.spec.inputs.get("portfolio_quantile", 0.1)),
        )
        output_path = context.seed_dir / "audit.json"
        output_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ExecutorOutput(
            metrics={"audit": report.to_dict()},
            artifacts={"audit": output_path},
            anomalies=tuple(report.anomalies),
        )


class TopKCostSweepExecutor:
    """对 K、buffer 和成本网格运行 M1 组合内核。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        inputs = context.spec.inputs
        predictions_path = _resolve_input_path(inputs["predictions_path"], context.repository_root)
        predictions = load_portfolio_predictions(predictions_path)
        top_ks = [int(value) for value in inputs.get("top_k", [25, 50, 75, 100])]
        buffers = [int(value) for value in inputs.get("exit_buffer", [0, 10, 25, 50])]
        costs = [float(value) for value in inputs.get("cost_bps", [5, 10, 15, 20])]
        stamp_tax = float(inputs.get("sell_stamp_tax_bps", 5.0))
        minimum = int(inputs.get("min_symbols_per_day", max(top_ks)))
        metrics: dict[str, Any] = {"topk": {}}
        artifacts: dict[str, Path] = {}
        for top_k in top_ks:
            for buffer in buffers:
                for cost in costs:
                    key = f"k{top_k}.buffer{buffer}.cost{cost:g}"
                    evaluation = evaluate_topk_portfolio(
                        predictions,
                        policy=PortfolioPolicy(
                            top_k=top_k,
                            exit_buffer=buffer,
                            min_score_gap=float(inputs.get("min_score_gap", 0.0)),
                            min_symbols_per_day=minimum,
                            missing_holding_policy=cast(
                                MissingHoldingPolicy,
                                str(inputs.get("missing_holding_policy", "liquidate")),
                            ),
                            require_tradability=bool(inputs.get("require_tradability", False)),
                        ),
                        cost_model=CostModel(
                            per_side_bps=cost,
                            sell_stamp_tax_bps=stamp_tax,
                        ),
                    )
                    target = context.seed_dir / "topk" / key
                    paths = write_portfolio_artifacts(evaluation, target)
                    metrics["topk"][key] = evaluation.summary
                    for artifact_type, path in paths.items():
                        artifacts[f"{key}.{artifact_type}"] = Path(path)
        sweep_path = context.seed_dir / "topk-sweep.json"
        sweep_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["topk_sweep"] = sweep_path
        return ExecutorOutput(metrics=metrics, artifacts=artifacts)


class CompareExperimentsExecutor:
    """从 Registry 生成指定实验的嵌套指标对比。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        experiment_ids = [str(value) for value in context.spec.inputs.get("experiment_ids", [])]
        if not experiment_ids:
            raise ExecutorFailure("compare_experiments 需要 inputs.experiment_ids")
        rows = context.registry.average_metrics(experiment_ids)
        found = {str(row["experiment_id"]) for row in rows}
        missing = set(experiment_ids) - found
        if missing:
            raise ExecutorFailure(f"待比较实验缺少指标: {sorted(missing)}")
        result = {"comparison": rows}
        path = context.seed_dir / "comparison.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExecutorOutput(metrics=result, artifacts={"comparison": path})


@dataclass(frozen=True)
class UnsupportedExecutor:
    """尚无确定性实现的白名单入口会明确失败，不回退到训练。"""

    name: str

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        raise ExecutorFailure(f"executor 尚未实现: {self.name}")


def default_executors(
    command_overrides: dict[str, tuple[str, ...] | list[str]] | None = None,
) -> dict[str, ResearchExecutor]:
    commands: dict[str, tuple[str, ...]] = {
        "train_nextday": ("ticknet-nextday-train",),
        "train_minute_tcn": ("ticknet-minute-tcn-train",),
    }
    for name, command in (command_overrides or {}).items():
        commands[name] = tuple(command)
    return {
        "train_nextday": CommandExecutor(commands["train_nextday"]),
        "train_minute_tcn": CommandExecutor(commands["train_minute_tcn"]),
        "train_ranker": UnsupportedExecutor("train_ranker"),
        "export_predictions": UnsupportedExecutor("export_predictions"),
        "audit_predictions": AuditPredictionsExecutor(),
        "topk_cost_sweep": TopKCostSweepExecutor(),
        "walk_forward_robustness": UnsupportedExecutor("walk_forward_robustness"),
        "compare_experiments": CompareExperimentsExecutor(),
    }
