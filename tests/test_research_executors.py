"""typed cost/audit executor 与训练后强制 Audit 的端到端测试。"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.research.policy import PolicyViolation
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentSpec, MetricGate


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
