"""ExperimentSpec.executor 对应的白名单确定性执行器。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow.parquet as pq

from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.comparison import (
    ComparisonError,
    compare_registered_experiments,
    summarize_walk_forward,
)
from ticknet.research.portfolio import (
    CostModel,
    MissingHoldingPolicy,
    PortfolioPolicy,
    evaluate_topk_portfolio,
    load_portfolio_predictions,
    write_portfolio_artifacts,
)
from ticknet.research.portfolio_sweep import summarize_topk_sweep
from ticknet.research.prediction_contract import (
    PredictionContractError,
    validate_formal_prediction_artifact,
)
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import ExperimentRegistry, file_sha256
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
    protocol: ResearchProtocol


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


@dataclass(frozen=True)
class PredictionSource:
    """已解析并校验过内容身份的 prediction 输入。"""

    path: Path
    sha256: str
    size_bytes: int
    mode: str
    dataset_fingerprint: str | None = None
    source_experiment_id: str | None = None
    source_seed: int | None = None
    artifact_name: str | None = None


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


def _registered_prediction_source(
    context: ExecutorContext,
    inputs: dict[str, Any],
) -> PredictionSource:
    source_experiment_id = str(inputs["source_experiment_id"])
    source_seed = int(inputs.get("source_seed", 0))
    artifact_name = str(inputs.get("artifact_name", "predictions"))
    experiment = context.registry.get_experiment(source_experiment_id)
    if experiment is None:
        raise ExecutorFailure(f"源实验不存在: {source_experiment_id}")
    if experiment["status"] not in {"completed", "frozen", "locked_tested"}:
        raise ExecutorFailure(
            f"源实验尚未完成: {source_experiment_id} status={experiment['status']}"
        )
    candidates = [
        row
        for row in context.registry.get_artifacts(
            source_experiment_id,
            name=artifact_name,
        )
        if int(row["seed"]) == source_seed
    ]
    if len(candidates) != 1:
        raise ExecutorFailure(
            f"源 prediction artifact 必须唯一: {source_experiment_id} "
            f"seed={source_seed} name={artifact_name}"
        )
    artifact = candidates[0]
    source = Path(str(artifact["path"])).expanduser().resolve()
    if not source.is_file():
        raise ExecutorFailure(f"源 prediction artifact 不存在: {source}")
    digest = file_sha256(source)
    if digest != artifact["sha256"]:
        raise ExecutorFailure(f"源 prediction artifact SHA-256 不一致: {source}")
    context.protocol.assert_predictions_safe(source)
    return PredictionSource(
        path=source,
        sha256=digest,
        size_bytes=source.stat().st_size,
        mode="registry_artifact",
        dataset_fingerprint=(
            str(experiment["dataset_fingerprint"])
            if experiment["dataset_fingerprint"] is not None
            else None
        ),
        source_experiment_id=source_experiment_id,
        source_seed=source_seed,
        artifact_name=artifact_name,
    )


def _prediction_source(context: ExecutorContext) -> PredictionSource:
    inputs = context.spec.inputs
    if str(inputs.get("source_experiment_id", "")).strip():
        source = _registered_prediction_source(context, inputs)
    else:
        path = _resolve_input_path(inputs["predictions_path"], context.repository_root)
        if not path.is_file():
            raise ExecutorFailure(f"预测明细不存在: {path}")
        context.protocol.assert_predictions_safe(path)
        source = PredictionSource(
            path=path,
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
            mode="direct_path",
        )
    if inputs.get("evaluation_mode", "smoke") == "formal" and not source.dataset_fingerprint:
        raise ExecutorFailure("formal topk_cost_sweep 的源实验必须包含 dataset_fingerprint")
    return source


def _materialize_prediction_source(source: PredictionSource, destination: Path) -> None:
    shutil.copyfile(source.path, destination)
    if file_sha256(destination) != source.sha256:
        raise ExecutorFailure("物化 prediction artifact 后 SHA-256 不一致")


def _prediction_source_metrics(
    source: PredictionSource,
    *,
    materialized_path: Path,
    evaluation_mode: str,
    target_return_contract: str,
) -> dict[str, Any]:
    schema = pq.read_schema(materialized_path)
    return {
        "mode": source.mode,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "source_experiment_id": source.source_experiment_id,
        "source_seed": source.source_seed,
        "artifact_name": source.artifact_name,
        "dataset_fingerprint": source.dataset_fingerprint,
        "evaluation_mode": evaluation_mode,
        "target_return_contract": target_return_contract,
        "tradability_columns_present": {"can_buy", "can_sell"} <= set(schema.names),
        "universe_membership_column_present": "in_universe" in schema.names,
    }


def _metric_directions(inputs: dict[str, Any]) -> dict[str, str]:
    raw = inputs.get("metric_directions", {})
    if not isinstance(raw, dict):
        raise ExecutorFailure("inputs.metric_directions 必须为对象")
    return {str(key): str(value) for key, value in raw.items()}


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


class ExportPredictionsExecutor:
    """校验并物化 Registry 中已有的 prediction artifact。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        source = _registered_prediction_source(context, context.spec.inputs)
        required = {"symbol", "trading_date", "label_date", "target_return", "score"}
        schema = pq.read_schema(source.path)
        missing = required - set(schema.names)
        if missing:
            raise ExecutorFailure(f"源 prediction artifact 缺少字段: {sorted(missing)}")

        destination = context.seed_dir / "predictions.parquet"
        _materialize_prediction_source(source, destination)
        identity = pq.read_table(
            destination,
            columns=["symbol", "trading_date", "label_date"],
        )
        metrics = {
            "export": {
                "row_count": identity.num_rows,
                "symbol_count": len(set(identity["symbol"].to_pylist())),
                "trading_date_count": len(set(identity["trading_date"].to_pylist())),
                "label_date_count": len(set(identity["label_date"].to_pylist())),
                "source_seed": source.source_seed,
            }
        }
        return ExecutorOutput(
            metrics=metrics,
            artifacts={"predictions": destination},
            dataset_fingerprint=source.dataset_fingerprint,
        )


class ImportPredictionsExecutor:
    """校验外部正式 prediction artifact，并把内容身份登记进 Registry。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        inputs = context.spec.inputs
        source = _resolve_input_path(inputs["predictions_path"], context.repository_root)
        context.protocol.assert_predictions_safe(source)
        try:
            report = validate_formal_prediction_artifact(
                source,
                expected_universe_size=int(inputs.get("expected_universe_size", 400)),
                expected_target_return_contract=str(inputs["target_return_contract"]),
            )
        except PredictionContractError as error:
            raise ExecutorFailure(str(error)) from error
        destination = context.seed_dir / "predictions.parquet"
        materialized = PredictionSource(
            path=source,
            sha256=report.sha256,
            size_bytes=source.stat().st_size,
            mode="formal_import",
            dataset_fingerprint=report.dataset_fingerprint,
        )
        _materialize_prediction_source(materialized, destination)
        return ExecutorOutput(
            metrics={"formal_prediction_import": report.to_dict()},
            artifacts={"predictions": destination},
            dataset_fingerprint=report.dataset_fingerprint,
        )


class TopKCostSweepExecutor:
    """对完整 K、buffer 和成本网格运行 M1 内核并生成 M3 诊断。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        inputs = context.spec.inputs
        source = _prediction_source(context)
        predictions_path = context.seed_dir / "source-predictions.parquet"
        _materialize_prediction_source(source, predictions_path)
        formal_report = None
        if inputs.get("evaluation_mode", "smoke") == "formal":
            try:
                formal_report = validate_formal_prediction_artifact(
                    predictions_path,
                    expected_universe_size=int(inputs.get("expected_universe_size", 400)),
                    expected_dataset_fingerprint=source.dataset_fingerprint,
                    expected_target_return_contract=str(inputs["target_return_contract"]),
                )
            except PredictionContractError as error:
                raise ExecutorFailure(str(error)) from error
        predictions = load_portfolio_predictions(predictions_path)
        top_ks = [int(value) for value in inputs.get("top_k", [25, 50, 75, 100])]
        buffers = [int(value) for value in inputs.get("exit_buffer", [0, 10, 25, 50])]
        costs = [float(value) for value in inputs.get("cost_bps", [5, 10, 15, 20])]
        stamp_tax = float(inputs.get("sell_stamp_tax_bps", 5.0))
        minimum = int(inputs.get("min_symbols_per_day", max(top_ks)))
        metrics: dict[str, Any] = {"topk": {}}
        artifacts: dict[str, Path] = {"source_predictions": predictions_path}
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
                            require_universe_membership=bool(
                                inputs.get("require_universe_membership", False)
                            ),
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
        evaluation_mode = str(inputs.get("evaluation_mode", "smoke"))
        default_decision_cost = 10.0 if 10.0 in costs else costs[0]
        diagnostic = summarize_topk_sweep(
            metrics["topk"],
            evaluation_mode=evaluation_mode,
            decision_cost_bps=float(inputs.get("decision_cost_bps", default_decision_cost)),
            minimum_evaluated_dates=int(inputs.get("minimum_evaluated_dates", 60)),
            minimum_positive_month_ratio=float(inputs.get("minimum_positive_month_ratio", 0.5)),
            maximum_top5_absolute_contribution=float(
                inputs.get("maximum_top5_absolute_contribution", 0.5)
            ),
        )
        diagnostic["source"] = _prediction_source_metrics(
            source,
            materialized_path=predictions_path,
            evaluation_mode=evaluation_mode,
            target_return_contract=str(inputs.get("target_return_contract", "unspecified")),
        )
        if formal_report is not None:
            diagnostic["source"]["formal_prediction_contract"] = formal_report.to_dict()
        metrics["diagnostic"] = diagnostic
        sweep_path = context.seed_dir / "topk-sweep.json"
        sweep_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        diagnostic_path = context.seed_dir / "m3-diagnostic.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["topk_sweep"] = sweep_path
        artifacts["m3_diagnostic"] = diagnostic_path
        return ExecutorOutput(
            metrics=metrics,
            artifacts=artifacts,
            dataset_fingerprint=source.dataset_fingerprint,
        )


class CompareExperimentsExecutor:
    """从 Registry 生成基线差值、seed 波动和配对差值。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        experiment_ids = [str(value) for value in context.spec.inputs.get("experiment_ids", [])]
        metrics = [
            str(value) for value in context.spec.inputs.get("metrics", context.spec.primary_metrics)
        ]
        baseline_id = str(
            context.spec.inputs.get(
                "baseline_id",
                experiment_ids[0] if experiment_ids else "",
            )
        )
        try:
            comparison, fingerprint = compare_registered_experiments(
                context.registry,
                experiment_ids,
                metrics,
                baseline_id=baseline_id,
                metric_directions=_metric_directions(context.spec.inputs),
                require_same_fingerprint=bool(
                    context.spec.inputs.get("require_same_fingerprint", True)
                ),
            )
        except ComparisonError as error:
            raise ExecutorFailure(str(error)) from error
        result = {"comparison": comparison}
        path = context.seed_dir / "comparison.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExecutorOutput(
            metrics=result,
            artifacts={"comparison": path},
            dataset_fingerprint=fingerprint,
        )


class WalkForwardRobustnessExecutor:
    """把多个 Registry 实验视为滚动窗口并汇总最差窗口和跨窗口波动。"""

    def execute(self, context: ExecutorContext) -> ExecutorOutput:
        inputs = context.spec.inputs
        experiment_ids = [str(value) for value in inputs.get("experiment_ids", [])]
        metrics = [str(value) for value in inputs.get("metrics", context.spec.primary_metrics)]
        try:
            robustness, fingerprint = summarize_walk_forward(
                context.registry,
                experiment_ids,
                metrics,
                minimum_windows=int(inputs.get("minimum_windows", 3)),
                require_distinct_fingerprints=bool(
                    inputs.get("require_distinct_fingerprints", True)
                ),
                metric_directions=_metric_directions(inputs),
            )
        except ComparisonError as error:
            raise ExecutorFailure(str(error)) from error
        result = {"robustness": robustness}
        path = context.seed_dir / "walk-forward.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExecutorOutput(
            metrics=result,
            artifacts={"walk_forward": path},
            dataset_fingerprint=fingerprint,
        )


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
        "import_predictions": ImportPredictionsExecutor(),
        "export_predictions": ExportPredictionsExecutor(),
        "audit_predictions": AuditPredictionsExecutor(),
        "topk_cost_sweep": TopKCostSweepExecutor(),
        "walk_forward_robustness": WalkForwardRobustnessExecutor(),
        "compare_experiments": CompareExperimentsExecutor(),
    }
