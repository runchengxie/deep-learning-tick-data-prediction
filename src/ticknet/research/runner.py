"""实验执行器（ExperimentRunner）：研究闭环中唯一能跑实验的入口。

Runner 把 ExperimentSpec 解析为完整训练配置，调用底层训练入口，收集每个 seed
的结果并登记到 Registry。Agent 或人工只能提交 ExperimentSpec，不能直接执行
任意命令；任何对配置的修改都先经过 policy 校验。

第一版不接 LLM：ExperimentSpec 由人工或后续的 Brainstorm Agent 生成。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from ticknet.research.policy import ResearchPolicy
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.spec import ExperimentResult, ExperimentSpec


class RunnerError(RuntimeError):
    """实验执行失败。"""


class ExperimentRunner:
    """执行并登记一个受控实验。"""

    def __init__(
        self,
        registry: ExperimentRegistry,
        *,
        policy: ResearchPolicy | None = None,
        repository_root: str | Path,
        artifact_root: str | Path = "research/experiments",
        entry_points: dict[str, str | list[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or ResearchPolicy()
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.entry_points = entry_points or {
            "nextday": "ticknet-nextday-train",
            "minute_tcn": "ticknet-minute-tcn-train",
        }

    def run(
        self,
        spec: ExperimentSpec,
        *,
        experiment_id: str,
    ) -> ExperimentResult:
        self.policy.validate(spec)
        spec.validate()
        artifact_dir = self._prepare_artifact_dir(experiment_id)
        resolved_config = self._resolve_config(spec, artifact_dir)

        git_sha = self._git_sha()
        per_seed_metrics: list[dict[str, Any]] = []
        seed_results: list[tuple[int, str, float]] = []
        for seed in spec.seeds:
            started_at = time.perf_counter()
            seed_config = self._apply_seed(resolved_config, seed)
            try:
                metrics = self._run_seed(seed_config, spec, experiment_id, seed)
                status = "completed"
            except RunnerError:
                status = "failed"
                raise
            duration = time.perf_counter() - started_at
            per_seed_metrics.append(metrics)
            seed_results.append((seed, status, duration))

        for seed, status, duration in seed_results:
            self.registry.record_run(experiment_id, seed, status, duration, None)
        for seed, metrics in zip(spec.seeds, per_seed_metrics, strict=True):
            self.registry.record_metrics(experiment_id, seed, metrics)

        result = ExperimentResult(
            experiment_id=experiment_id,
            spec=spec,
            status="completed",
            git_sha=git_sha,
            dataset_fingerprint=metrics.get("dataset_fingerprint"),
            per_seed_metrics=per_seed_metrics,
            artifact_dir=str(artifact_dir),
        )
        self.registry.record_experiment(experiment_id, result, spec)
        return result

    def _prepare_artifact_dir(self, experiment_id: str) -> Path:
        path = self.artifact_root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_config(self, spec: ExperimentSpec, artifact_dir: Path) -> dict[str, Any]:
        base = self.repository_root / spec.base_config
        if not base.is_file():
            raise RunnerError(f"base_config 不存在: {base}")
        with base.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise RunnerError(f"base_config 根节点应为对象: {base}")
        for key, value in spec.config_overrides.items():
            config[key] = value
        config.setdefault("resume", False)
        config["evaluate_test"] = False
        resolved_path = artifact_dir / "resolved-config.yaml"
        with resolved_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        return config

    def _apply_seed(self, config: dict[str, Any], seed: int) -> dict[str, Any]:
        return {**config, "seed": seed}

    def _run_seed(
        self,
        config: dict[str, Any],
        spec: ExperimentSpec,
        experiment_id: str,
        seed: int,
    ) -> dict[str, Any]:
        seed_dir = self.artifact_root / experiment_id / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        config_path = seed_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

        entry = spec.entry_point or self.entry_points.get(spec.experiment_type)
        if entry is None:
            entry = self.entry_points.get("nextday")
        if entry is None:
            raise RunnerError("未配置可用的实验入口")

        if isinstance(entry, list):
            command = [*entry, "--config", str(config_path)]
        else:
            resolved_entry = self._resolve_entry(entry)
            command = [resolved_entry, "--config", str(config_path)]
        completed = subprocess.run(
            command,
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode != 0:
            raise RunnerError(f"seed {seed} 执行失败:\n{completed.stdout}\n{completed.stderr}")

        if not completed.stdout.strip():
            error = f"rc={completed.returncode}\nstderr={completed.stderr!r}"
            raise RunnerError(f"seed {seed} 无 stdout 输出:\n{error}")
        match = re.search(r"\{.*\}", completed.stdout, re.DOTALL)
        if match is None:
            raise RunnerError(f"seed {seed} stdout 中没有 JSON:\n{completed.stdout[:500]}")
        result = json.loads(match.group(0))
        return result

    def _resolve_entry(self, entry: str) -> str:
        """从仓库 .venv/bin 解析命令入口，兼容非交互 shell 缺少 PATH 的情况。"""
        candidates = [
            self.repository_root / ".venv" / "bin" / entry,
            self.repository_root / ".venv" / "Scripts" / entry,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        resolved = shutil.which(entry)
        if resolved is not None:
            return resolved
        raise RunnerError(f"找不到命令入口: {entry}")

    def _git_sha(self) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"
