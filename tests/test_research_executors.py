"""typed cost/audit executor 与训练后强制 Audit 的端到端测试。"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.research.comparison import ComparisonError, compare_registered_experiments
from ticknet.research.policy import PolicyViolation
from ticknet.research.registry import ExperimentRegistry, file_sha256
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def _write_predictions(
    path: Path,
    *,
    days: int = 3,
    symbols: int = 100,
    start: date = date(2025, 1, 2),
) -> None:
    rng = np.random.RandomState(7)
    rows = []
    for offset in range(days):
        trading_date = start + timedelta(days=offset)
        label_date = trading_date + timedelta(days=1)
        for index in range(symbols):
            score = float(rng.randn())
            rows.append(
                {
                    "symbol": f"{600000 + index:06d}",
                    "trading_date": trading_date.isoformat(),
                    "label_date": label_date.isoformat(),
                    "score": score,
                    "target_return": 0.02 * score + float(0.001 * rng.randn()),
                    "can_buy": True,
                    "can_sell": True,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_cost_analysis_uses_topk_executor_without_training(tmp_path) -> None:
    predictions = tmp_path / "predictions.parquet"
    _write_predictions(predictions, symbols=10)
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    spec = ExperimentSpec(
        hypothesis="Top-K buffer 能降低成本",
        objective="运行一个最小 Top-K 成本组合",
        experiment_type="cost_analysis",
        executor="topk_cost_sweep",
        inputs={
            "predictions_path": str(predictions),
            "top_k": [2],
            "exit_buffer": [1],
            "cost_bps": [0],
            "min_symbols_per_day": 10,
            "require_tradability": True,
        },
        seeds=(0,),
        primary_metrics=("topk.k2.buffer1.cost0.net.mean_daily",),
        success_gates=(MetricGate("topk.k2.buffer1.cost0.net.mean_daily", "gt", -1.0),),
        artifact_contract=(
            "resolved_spec",
            "resolved_config",
            "stdout",
            "stderr",
            "result",
            "run_manifest",
            "topk_sweep",
        ),
        rationale="验证成本 executor 的语义路由",
        falsification_condition="净日收益小于等于 -100% 则否定",
        novelty_signature="test-topk-cost-executor",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["command-that-must-not-run"]},
    )
    result = runner.run(spec, experiment_id="EXP-COST")
    assert result.status == "completed"
    assert "topk" in result.per_seed_metrics[0]
    assert (tmp_path / "artifacts/EXP-COST/seed0/topk-sweep.json").is_file()
    assert registry.get_runs("EXP-COST")[0]["exit_code"] == 0
    registry.close()


def test_training_prediction_artifact_is_automatically_audited(tmp_path) -> None:
    predictions = tmp_path / "source-predictions.parquet"
    _write_predictions(predictions)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "samples": [{"trading_date": "2025-01-02", "symbol": "600000"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text(
        f"manifest_path: {manifest}\n",
        encoding="utf-8",
    )
    mock = tmp_path / "mock_train.py"
    mock.write_text(
        "import json, pathlib, shutil, sys, yaml\n"
        "config_path = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "config = yaml.safe_load(config_path.read_text())\n"
        f"source = pathlib.Path({str(predictions)!r})\n"
        "target = pathlib.Path(config['checkpoint_dir']) / 'predictions.parquet'\n"
        "shutil.copyfile(source, target)\n"
        "print(json.dumps({'validation': {'"
        "daily_rank_ic_mean': 0.03}, 'dataset_fingerprint': 'fp', "
        "'predictions_path': str(target)}))\n",
        encoding="utf-8",
    )
    spec = ExperimentSpec(
        hypothesis="训练结果应自动进入预测审计",
        objective="验证训练、预测 artifact、Audit、Registry 和 Evaluation 串联",
        experiment_type="ablation",
        executor="train_nextday",
        base_config="base.yaml",
        seeds=(0,),
        primary_metrics=("validation.daily_rank_ic_mean", "audit.daily_ic_mean"),
        success_gates=(MetricGate("audit.daily_ic_mean", "gt", 0.0),),
        artifact_contract=(
            "resolved_spec",
            "resolved_config",
            "stdout",
            "stderr",
            "result",
            "run_manifest",
            "predictions",
            "audit",
        ),
        rationale="训练产物不能绕过 Evaluation Agent",
        falsification_condition="预测审计 IC 不大于 0 则否定",
        novelty_signature="test-training-auto-audit",
    )
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["python", str(mock)]},
    )
    result = runner.run(spec, experiment_id="EXP-AUTO-AUDIT")
    assert result.evaluation_decision == "EXTEND"
    assert result.per_seed_metrics[0]["audit"]["daily_ic_mean"] > 0
    metric_names = {row["metric"] for row in registry.get_metrics("EXP-AUTO-AUDIT")}
    assert "audit.daily_ic_mean" in metric_names
    assert (tmp_path / "artifacts/EXP-AUTO-AUDIT/seed0/audit.json").is_file()
    registry.close()


def test_training_cannot_emit_locked_predictions(tmp_path) -> None:
    predictions = tmp_path / "locked-predictions.parquet"
    _write_predictions(predictions, start=date(2026, 1, 2))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "samples": [{"trading_date": "2025-01-02", "symbol": "600000"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text(f"manifest_path: {manifest}\n", encoding="utf-8")
    mock = tmp_path / "mock_locked.py"
    mock.write_text(
        "import json, pathlib, shutil, sys, yaml\n"
        "config_path = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])\n"
        "config = yaml.safe_load(config_path.read_text())\n"
        f"source = pathlib.Path({str(predictions)!r})\n"
        "target = pathlib.Path(config['checkpoint_dir']) / 'predictions.parquet'\n"
        "shutil.copyfile(source, target)\n"
        "print(json.dumps({'validation': {'"
        "daily_rank_ic_mean': 0.03}, 'predictions_path': str(target)}))\n",
        encoding="utf-8",
    )
    spec = ExperimentSpec(
        hypothesis="训练输出不得绕过锁定期",
        objective="验证训练后 predictions 仍受协议控制",
        experiment_type="ablation",
        executor="train_nextday",
        base_config="base.yaml",
        success_gates=(MetricGate("validation.daily_rank_ic_mean", "gt", 0.0),),
        rationale="防止通过训练 artifact 读取 locked 数据",
        falsification_condition="任何 2026 prediction 都应拒绝",
        novelty_signature="test-locked-training-output",
    )
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["python", str(mock)]},
    )
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        runner.run(spec, experiment_id="EXP-LOCKED-OUTPUT")
    experiment = registry.get_experiment("EXP-LOCKED-OUTPUT")
    assert experiment is not None
    assert experiment["status"] == "rejected"
    assert registry.get_runs("EXP-LOCKED-OUTPUT")[0]["status"] == "failed"
    registry.close()


def _register_metric_experiment(
    registry: ExperimentRegistry,
    experiment_id: str,
    values: tuple[float, ...],
    *,
    fingerprint: str,
) -> None:
    metric = "validation.daily_rank_ic_mean"
    spec = ExperimentSpec(
        hypothesis=f"{experiment_id} 指标实验",
        objective="为确定性对比提供多 seed 指标",
        experiment_type="ablation",
        executor="train_nextday",
        base_config="base.yaml",
        seeds=tuple(range(len(values))),
        primary_metrics=(metric,),
        success_gates=(MetricGate(metric, "gt", -1.0),),
        rationale="合成 Registry fixture",
        falsification_condition="指标不大于 -1 则否定",
        novelty_signature=f"fixture-{experiment_id}",
    )
    registry.record_experiment(
        experiment_id,
        ExperimentResult(
            experiment_id=experiment_id,
            spec=spec,
            status="completed",
            git_sha="abc",
            dataset_fingerprint=fingerprint,
            per_seed_metrics=[],
            artifact_dir=f"/tmp/{experiment_id}",
            evaluation_decision="EXTEND",
        ),
        spec,
    )
    for seed, value in enumerate(values):
        registry.record_metrics(
            experiment_id,
            seed,
            {"validation": {"daily_rank_ic_mean": value}},
        )


def test_export_predictions_materializes_registered_artifact_and_audits(tmp_path) -> None:
    source_predictions = tmp_path / "source.parquet"
    _write_predictions(source_predictions)
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(
        registry,
        "EXP-SOURCE",
        (0.02,),
        fingerprint="source-fingerprint",
    )
    registry.record_artifact("EXP-SOURCE", 0, "predictions", source_predictions)
    spec = ExperimentSpec(
        hypothesis="已登记预测应可被确定性物化并重新审计",
        objective="验证 prediction artifact 的 checksum、复制和 Audit 链路",
        experiment_type="prediction_export",
        executor="export_predictions",
        inputs={"source_experiment_id": "EXP-SOURCE", "source_seed": 0},
        seeds=(0,),
        primary_metrics=("audit.daily_ic_mean",),
        success_gates=(MetricGate("audit.daily_ic_mean", "gt", 0.0),),
        artifact_contract=(
            "resolved_spec",
            "resolved_config",
            "stdout",
            "stderr",
            "result",
            "run_manifest",
            "predictions",
            "audit",
        ),
        rationale="预测导出不能绕过 Registry checksum 和协议",
        falsification_condition="物化后 Audit IC 不大于 0 则否定",
        novelty_signature="test-export-registered-predictions",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    result = runner.run(spec, experiment_id="EXP-EXPORT")
    exported = tmp_path / "artifacts/EXP-EXPORT/seed0/predictions.parquet"
    assert result.evaluation_decision == "EXTEND"
    assert result.dataset_fingerprint == "source-fingerprint"
    assert result.per_seed_metrics[0]["export"]["row_count"] == 300
    assert result.per_seed_metrics[0]["audit"]["daily_ic_mean"] > 0
    assert file_sha256(exported) == file_sha256(source_predictions)
    registry.close()


def test_export_predictions_rejects_mutated_source_artifact(tmp_path) -> None:
    source_predictions = tmp_path / "source.parquet"
    _write_predictions(source_predictions)
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(registry, "EXP-SOURCE", (0.02,), fingerprint="fp")
    registry.record_artifact("EXP-SOURCE", 0, "predictions", source_predictions)
    source_predictions.write_bytes(b"mutated")
    spec = ExperimentSpec(
        hypothesis="被修改的 Registry artifact 必须拒绝",
        objective="验证 export checksum 边界",
        experiment_type="prediction_export",
        executor="export_predictions",
        inputs={"source_experiment_id": "EXP-SOURCE"},
        primary_metrics=("export.row_count",),
        success_gates=(MetricGate("export.row_count", "gt", 0.0),),
        rationale="防止 artifact 登记后被替换",
        falsification_condition="checksum 不一致时必须失败",
        novelty_signature="test-export-checksum-rejection",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    with pytest.raises(RunnerError, match="SHA-256"):
        runner.run(spec, experiment_id="EXP-EXPORT-BAD")
    registry.close()


def test_export_predictions_cannot_bypass_locked_protocol(tmp_path) -> None:
    source_predictions = tmp_path / "locked-source.parquet"
    _write_predictions(source_predictions, start=date(2026, 1, 2))
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(registry, "EXP-LOCKED-SOURCE", (0.02,), fingerprint="fp")
    registry.record_artifact("EXP-LOCKED-SOURCE", 0, "predictions", source_predictions)
    spec = ExperimentSpec(
        hypothesis="Registry 中的预测仍不能绕过 locked 协议",
        objective="验证 prediction export 后仍由 Runner 检查日期边界",
        experiment_type="prediction_export",
        executor="export_predictions",
        inputs={"source_experiment_id": "EXP-LOCKED-SOURCE"},
        primary_metrics=("export.row_count",),
        success_gates=(MetricGate("export.row_count", "gt", 0.0),),
        rationale="物化已有 artifact 不会扩大数据权限",
        falsification_condition="任何 2026 prediction 都应拒绝",
        novelty_signature="test-export-locked-rejection",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        runner.run(spec, experiment_id="EXP-EXPORT-LOCKED")
    experiment = registry.get_experiment("EXP-EXPORT-LOCKED")
    assert experiment is not None
    assert experiment["status"] == "rejected"
    registry.close()


def test_compare_executor_reports_seed_volatility_and_paired_delta(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(
        registry,
        "EXP-BASE",
        (0.01, 0.02, 0.03),
        fingerprint="comparison-fp",
    )
    _register_metric_experiment(
        registry,
        "EXP-CANDIDATE",
        (0.03, 0.04, 0.05),
        fingerprint="comparison-fp",
    )
    brier = "validation.brier_score"
    for experiment_id, values in {
        "EXP-BASE": (0.20, 0.22, 0.24),
        "EXP-CANDIDATE": (0.18, 0.20, 0.22),
    }.items():
        for seed, value in enumerate(values):
            registry.record_metrics(
                experiment_id,
                seed,
                {"validation": {"brier_score": value}},
            )
    metric = "validation.daily_rank_ic_mean"
    gate = f"comparison.experiments.EXP-CANDIDATE.metrics.{metric}.delta_vs_baseline_mean"
    spec = ExperimentSpec(
        hypothesis="候选实验的多 seed Rank IC 高于基线",
        objective="比较 seed 波动、均值差和同 seed 配对差",
        experiment_type="comparison",
        executor="compare_experiments",
        inputs={
            "experiment_ids": ["EXP-BASE", "EXP-CANDIDATE"],
            "baseline_id": "EXP-BASE",
            "metrics": [metric, brier],
            "metric_directions": {brier: "lower"},
        },
        primary_metrics=(metric, brier),
        success_gates=(MetricGate(gate, "gt", 0.0),),
        artifact_contract=(
            "resolved_spec",
            "resolved_config",
            "stdout",
            "stderr",
            "result",
            "run_manifest",
            "comparison",
        ),
        rationale="均值差需要和 seed 波动一起解释",
        falsification_condition="候选均值不高于基线则否定",
        novelty_signature="test-multiseed-comparison",
    )
    result = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    ).run(spec, experiment_id="EXP-COMPARE")
    candidate = result.per_seed_metrics[0]["comparison"]["experiments"]["EXP-CANDIDATE"]
    summary = candidate["metrics"][metric]
    brier_summary = candidate["metrics"][brier]
    assert result.evaluation_decision == "EXTEND"
    assert summary["mean"] == pytest.approx(0.04)
    assert summary["std"] == pytest.approx(0.01)
    assert summary["direction"] == "higher"
    assert summary["delta_vs_baseline_mean"] == pytest.approx(0.02)
    assert summary["improvement_vs_baseline_mean"] == pytest.approx(0.02)
    assert summary["paired_delta_mean"] == pytest.approx(0.02)
    assert summary["paired_improvement_mean"] == pytest.approx(0.02)
    assert summary["paired_seed_count"] == 3
    assert brier_summary["direction"] == "lower"
    assert brier_summary["delta_vs_baseline_mean"] == pytest.approx(-0.02)
    assert brier_summary["improvement_vs_baseline_mean"] == pytest.approx(0.02)
    assert brier_summary["paired_improvement_mean"] == pytest.approx(0.02)
    registry.close()


def test_compare_rejects_different_dataset_fingerprints_by_default(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(registry, "EXP-A", (0.01,), fingerprint="fp-a")
    _register_metric_experiment(registry, "EXP-B", (0.02,), fingerprint="fp-b")
    with pytest.raises(ComparisonError, match="相同 dataset_fingerprint"):
        compare_registered_experiments(
            registry,
            ["EXP-A", "EXP-B"],
            ["validation.daily_rank_ic_mean"],
            baseline_id="EXP-A",
        )
    registry.close()


def test_walk_forward_executor_reports_worst_window_and_keep_decision(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _register_metric_experiment(registry, "EXP-W22", (0.01, 0.02), fingerprint="fp-22")
    _register_metric_experiment(registry, "EXP-W23", (0.02, 0.03), fingerprint="fp-23")
    _register_metric_experiment(registry, "EXP-W24", (0.03, 0.04), fingerprint="fp-24")
    brier = "validation.brier_score"
    for experiment_id, values in {
        "EXP-W22": (0.10, 0.12),
        "EXP-W23": (0.15, 0.17),
        "EXP-W24": (0.20, 0.22),
    }.items():
        for seed, value in enumerate(values):
            registry.record_metrics(
                experiment_id,
                seed,
                {"validation": {"brier_score": value}},
            )
    metric = "validation.daily_rank_ic_mean"
    rank_gate = f"robustness.metrics.{metric}.window_min"
    brier_gate = f"robustness.metrics.{brier}.window_max"
    spec = ExperimentSpec(
        hypothesis="滚动窗口 Rank IC 均保持为正",
        objective="汇总多窗口 seed 均值、窗口波动和最差窗口",
        experiment_type="robustness",
        executor="walk_forward_robustness",
        inputs={
            "experiment_ids": ["EXP-W22", "EXP-W23", "EXP-W24"],
            "metrics": [metric, brier],
            "metric_directions": {brier: "lower"},
            "minimum_windows": 3,
        },
        primary_metrics=(metric, brier),
        success_gates=(
            MetricGate(rank_gate, "gt", 0.0),
            MetricGate(brier_gate, "lt", 0.3),
        ),
        artifact_contract=(
            "resolved_spec",
            "resolved_config",
            "stdout",
            "stderr",
            "result",
            "run_manifest",
            "walk_forward",
        ),
        rationale="不能只看平均窗口而忽略最差年份",
        falsification_condition="任一窗口均值不为正则否定稳健性",
        novelty_signature="test-walk-forward-summary",
        stage="robustness",
    )
    result = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    ).run(spec, experiment_id="EXP-WALK")
    summary = result.per_seed_metrics[0]["robustness"]["metrics"][metric]
    brier_summary = result.per_seed_metrics[0]["robustness"]["metrics"][brier]
    assert result.evaluation_decision == "KEEP"
    assert summary["window_mean"] == pytest.approx(0.025)
    assert summary["window_min"] == pytest.approx(0.015)
    assert summary["worst_window_experiment_id"] == "EXP-W22"
    assert summary["worst_window_value"] == pytest.approx(0.015)
    assert summary["above_zero_window_ratio"] == 1.0
    assert brier_summary["direction"] == "lower"
    assert brier_summary["worst_window_experiment_id"] == "EXP-W24"
    assert brier_summary["worst_window_value"] == pytest.approx(0.21)
    registry.close()
