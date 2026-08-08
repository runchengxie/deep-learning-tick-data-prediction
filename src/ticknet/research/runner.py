"""白名单 executor、独立 artifact 和强制 Evaluation 的唯一运行入口。"""

from __future__ import annotations

import json
import re
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.evaluation import evaluate_metric_gates
from ticknet.research.executors import (
    ExecutorContext,
    ExecutorFailure,
    ResearchExecutor,
    default_executors,
)
from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import ExperimentRegistry, RegistryConflict
from ticknet.research.spec import ExperimentResult, ExperimentSpec


class RunnerError(RuntimeError):
    """实验执行或 artifact 契约失败。"""


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


class ExperimentRunner:
    """执行、审计、评估并登记一个受控实验。"""

    def __init__(
        self,
        registry: ExperimentRegistry,
        *,
        policy: ResearchPolicy | None = None,
        protocol: ResearchProtocol | None = None,
        repository_root: str | Path,
        artifact_root: str | Path = "research/experiments",
        executors: dict[str, ResearchExecutor] | None = None,
        command_overrides: dict[str, tuple[str, ...] | list[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or ResearchPolicy()
        self.protocol = protocol or ResearchProtocol()
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.executors = executors or default_executors(command_overrides)

    def reserve(self, spec: ExperimentSpec, *, experiment_id: str) -> None:
        """在 Critic/Policy 前登记提案，使任何终态都可审计。"""
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id) is None:
            raise RunnerError("experiment_id 只能包含字母、数字、点、下划线和连字符")
        artifact_dir = self.artifact_root / experiment_id
        if artifact_dir.exists():
            raise RunnerError(f"artifact 目录已存在，拒绝覆盖: {artifact_dir}")
        try:
            self.registry.create_experiment(
                experiment_id,
                spec,
                status="proposed",
                git_sha=self._git_sha(),
                artifact_dir=str(artifact_dir),
            )
        except RegistryConflict as error:
            raise RunnerError(str(error)) from error

    def run(self, spec: ExperimentSpec, *, experiment_id: str) -> ExperimentResult:
        if not self.registry.has_experiment(experiment_id):
            self.reserve(spec, experiment_id=experiment_id)
        else:
            existing = self.registry.get_experiment(experiment_id)
            if existing is None or existing["status"] != "proposed":
                raise RunnerError(f"实验 ID 已使用: {experiment_id}")
            stored = json.loads(str(existing["spec_json"]))
            normalized_spec = json.loads(json.dumps(spec.to_dict(), ensure_ascii=False))
            if stored != normalized_spec:
                raise RunnerError(f"已登记实验的 spec 不一致: {experiment_id}")

        try:
            self.policy.validate(spec)
            spec.validate()
        except (PolicyViolation, ValueError) as error:
            self.registry.update_experiment(experiment_id, status="rejected", error=str(error))
            self._record_review_once(
                experiment_id,
                "policy",
                "REJECTED",
                {"reason": str(error)},
            )
            raise

        executor = self.executors.get(spec.executor)
        if executor is None:
            error = RunnerError(f"未注册 executor: {spec.executor}")
            self.registry.update_experiment(experiment_id, status="failed", error=str(error))
            raise error

        artifact_dir = self.artifact_root / experiment_id
        if artifact_dir.exists():
            raise RunnerError(f"artifact 目录已存在，拒绝覆盖: {artifact_dir}")
        artifact_dir.mkdir(parents=True)
        git_sha = self._git_sha()
        resolved_spec_path = artifact_dir / "resolved-spec.json"
        environment_path = artifact_dir / "environment.json"
        _write_json(resolved_spec_path, spec.to_dict())
        _write_json(
            environment_path,
            {
                "git_sha": git_sha,
                "git_status": self._git_status(),
                "executor": spec.executor,
            },
        )
        self.registry.record_artifact(experiment_id, -1, "resolved_spec", resolved_spec_path)
        self.registry.record_artifact(experiment_id, -1, "environment", environment_path)

        try:
            resolved_config = self._resolve_config(spec, artifact_dir)
        except (RunnerError, ValueError) as error:
            self.registry.update_experiment(experiment_id, status="failed", error=str(error))
            self._record_review_once(
                experiment_id,
                "runner_failure",
                "FAILED",
                {"reason": str(error)},
            )
            raise
        try:
            self._validate_inputs(spec, resolved_config)
        except PolicyViolation as error:
            self.registry.update_experiment(experiment_id, status="rejected", error=str(error))
            self._record_review_once(
                experiment_id,
                "policy",
                "REJECTED",
                {"reason": str(error)},
            )
            raise

        self.registry.update_experiment(experiment_id, status="running")
        per_seed_metrics: list[dict[str, Any]] = []
        fingerprints: list[str] = []
        all_anomalies: list[dict[str, Any]] = []
        for seed in spec.seeds:
            metrics, fingerprint, anomalies = self._run_seed(
                executor,
                resolved_config,
                spec,
                experiment_id,
                seed,
            )
            per_seed_metrics.append(metrics)
            if fingerprint:
                fingerprints.append(fingerprint)
            all_anomalies.extend(anomalies)

        evaluation = evaluate_metric_gates(
            per_seed_metrics,
            spec.success_gates,
            stage=spec.stage,
        )
        self.registry.record_review(
            experiment_id,
            "evaluation",
            evaluation.decision,
            evaluation.to_dict(),
        )
        if all_anomalies:
            self.registry.record_review(
                experiment_id,
                "audit_anomalies",
                "RECORDED",
                {"anomalies": all_anomalies},
            )
        dataset_fingerprint = fingerprints[0] if fingerprints else None
        if len(set(fingerprints)) > 1:
            error = RunnerError("不同 seed 返回了不一致的数据指纹")
            self.registry.update_experiment(experiment_id, status="failed", error=str(error))
            raise error
        self.registry.update_experiment(
            experiment_id,
            status="completed",
            dataset_fingerprint=dataset_fingerprint,
            evaluation_decision=evaluation.decision,
        )
        result = ExperimentResult(
            experiment_id=experiment_id,
            spec=spec,
            status="completed",
            git_sha=git_sha,
            dataset_fingerprint=dataset_fingerprint,
            per_seed_metrics=per_seed_metrics,
            artifact_dir=str(artifact_dir),
            evaluation_decision=evaluation.decision,
            notes={"audit_anomalies": all_anomalies},
        )
        result_path = artifact_dir / "experiment-result.json"
        _write_json(result_path, result.to_dict())
        self.registry.record_artifact(experiment_id, -1, "experiment_result", result_path)
        return result

    def _resolve_config(self, spec: ExperimentSpec, artifact_dir: Path) -> dict[str, Any]:
        if not spec.base_config:
            config = {"inputs": spec.inputs}
        else:
            base = (self.repository_root / spec.base_config).resolve()
            if not base.is_relative_to(self.repository_root) or not base.is_file():
                raise RunnerError(f"base_config 不存在或越出仓库: {base}")
            with base.open(encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            if not isinstance(config, dict):
                raise RunnerError(f"base_config 根节点应为对象: {base}")
            config.update(spec.config_overrides)
            config["resume"] = False
            config["evaluate_test"] = False
        path = artifact_dir / "resolved-config.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        self.registry.record_artifact(artifact_dir.name, -1, "resolved_config", path)
        return config

    def _validate_inputs(self, spec: ExperimentSpec, config: dict[str, Any]) -> None:
        manifest_value = config.get("manifest_path")
        if manifest_value is not None:
            manifest = self._resolve_path(manifest_value)
            self.policy.validate_manifest(manifest, self.protocol)
        predictions_value = spec.inputs.get("predictions_path")
        if predictions_value is not None:
            predictions = self._resolve_path(predictions_value)
            self.protocol.assert_predictions_safe(predictions)

    def _run_seed(
        self,
        executor: ResearchExecutor,
        resolved_config: dict[str, Any],
        spec: ExperimentSpec,
        experiment_id: str,
        seed: int,
    ) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
        seed_dir = self.artifact_root / experiment_id / f"seed{seed}"
        seed_dir.mkdir()
        config = {**resolved_config, "seed": seed}
        if spec.executor.startswith("train_"):
            config["checkpoint_dir"] = str(seed_dir)
            config["checkpoint_name"] = f"{experiment_id}.{spec.executor}"
        config_path = seed_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        self.registry.start_run(experiment_id, seed)
        started_at = time.perf_counter()
        try:
            output = executor.execute(
                ExecutorContext(
                    spec=spec,
                    seed=seed,
                    repository_root=self.repository_root,
                    seed_dir=seed_dir,
                    config_path=config_path,
                    registry=self.registry,
                    protocol=self.protocol,
                )
            )
            metrics = dict(output.metrics)
            artifacts = dict(output.artifacts)
            anomalies = list(output.anomalies)
            predictions = artifacts.get("predictions")
            if predictions is not None:
                self.protocol.assert_predictions_safe(predictions)
                audit = audit_predictions(PredictionTable.from_parquet(predictions))
                audit_path = seed_dir / "audit.json"
                _write_json(audit_path, audit.to_dict())
                metrics["audit"] = audit.to_dict()
                artifacts["audit"] = audit_path
                anomalies.extend(audit.anomalies)
            result_path = seed_dir / "result.json"
            stdout_path = seed_dir / "stdout.log"
            stderr_path = seed_dir / "stderr.log"
            _write_json(result_path, metrics)
            stdout_path.write_text(output.stdout, encoding="utf-8")
            stderr_path.write_text(output.stderr, encoding="utf-8")
            artifacts.update(
                {
                    "resolved_config": config_path,
                    "stdout": stdout_path,
                    "stderr": stderr_path,
                    "result": result_path,
                }
            )
            self._assert_artifacts_local(seed_dir, artifacts)
            missing = set(spec.artifact_contract) - (
                set(artifacts) | {"resolved_spec", "run_manifest"}
            )
            if missing:
                raise ExecutorFailure(f"artifact contract 缺少: {sorted(missing)}")
            artifact_rows = self._record_artifacts(experiment_id, seed, artifacts)
            manifest_path = seed_dir / "run-manifest.json"
            _write_json(
                manifest_path,
                {
                    "experiment_id": experiment_id,
                    "executor": spec.executor,
                    "seed": seed,
                    "duration_seconds": time.perf_counter() - started_at,
                    "exit_code": output.exit_code,
                    "dataset_fingerprint": output.dataset_fingerprint,
                    "artifacts": artifact_rows,
                },
            )
            self.registry.record_artifact(experiment_id, seed, "run_manifest", manifest_path)
            self.registry.record_metrics(experiment_id, seed, metrics)
            self.registry.finish_run(
                experiment_id,
                seed,
                status="completed",
                duration_seconds=time.perf_counter() - started_at,
                result_path=str(result_path),
                exit_code=output.exit_code,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
            return metrics, output.dataset_fingerprint, anomalies
        except PolicyViolation as error:
            self._record_failed_run(
                experiment_id,
                seed,
                seed_dir,
                ExecutorFailure(str(error)),
                started_at,
            )
            self.registry.update_experiment(experiment_id, status="rejected", error=str(error))
            self._record_review_once(
                experiment_id,
                "policy",
                "REJECTED",
                {"seed": seed, "reason": str(error)},
            )
            raise
        except (ExecutorFailure, ValueError, RegistryConflict) as error:
            failure = error if isinstance(error, ExecutorFailure) else ExecutorFailure(str(error))
            self._record_failed_run(
                experiment_id,
                seed,
                seed_dir,
                failure,
                started_at,
            )
            self.registry.update_experiment(experiment_id, status="failed", error=str(error))
            self._record_review_once(
                experiment_id,
                "runner_failure",
                "FAILED",
                {"seed": seed, "reason": str(error)},
            )
            raise RunnerError(f"seed {seed} 执行失败: {error}") from error

    def _record_failed_run(
        self,
        experiment_id: str,
        seed: int,
        seed_dir: Path,
        failure: ExecutorFailure,
        started_at: float,
    ) -> None:
        stdout_path = seed_dir / "stdout.log"
        stderr_path = seed_dir / "stderr.log"
        failure_path = seed_dir / "failure.json"
        stdout_path.write_text(failure.stdout, encoding="utf-8")
        stderr_path.write_text(failure.stderr, encoding="utf-8")
        _write_json(
            failure_path,
            {
                "error": str(failure),
                "exit_code": failure.exit_code,
                "duration_seconds": time.perf_counter() - started_at,
            },
        )
        for name, path in {
            "stdout": stdout_path,
            "stderr": stderr_path,
            "failure": failure_path,
        }.items():
            with suppress(RegistryConflict):
                self.registry.record_artifact(experiment_id, seed, name, path)
        self.registry.finish_run(
            experiment_id,
            seed,
            status="failed",
            duration_seconds=time.perf_counter() - started_at,
            result_path=str(failure_path),
            exit_code=failure.exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    def _record_artifacts(
        self,
        experiment_id: str,
        seed: int,
        artifacts: dict[str, Path],
    ) -> list[dict[str, Any]]:
        return [
            self.registry.record_artifact(experiment_id, seed, name, path)
            for name, path in sorted(artifacts.items())
        ]

    def _assert_artifacts_local(self, seed_dir: Path, artifacts: dict[str, Path]) -> None:
        for name, path in artifacts.items():
            resolved = path.expanduser().resolve()
            if not resolved.is_file() or not resolved.is_relative_to(seed_dir):
                raise ExecutorFailure(f"artifact {name} 不在独立 seed 目录: {resolved}")

    def _record_review_once(
        self,
        experiment_id: str,
        review_type: str,
        decision: str,
        payload: dict[str, Any],
    ) -> None:
        if any(
            row["review_type"] == review_type for row in self.registry.get_reviews(experiment_id)
        ):
            return
        self.registry.record_review(experiment_id, review_type, decision, payload)

    def _resolve_path(self, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        return path.resolve()

    def _git_sha(self) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    def _git_status(self) -> str:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
        )
        return completed.stdout if completed.returncode == 0 else "unknown"
