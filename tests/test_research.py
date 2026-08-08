"""ExperimentSpec v2、Registry v2 与 typed Runner 测试。"""

import json
from pathlib import Path

import pytest

from ticknet.research.evaluation import evaluate_metric_gates
from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.registry import ExperimentRegistry, RegistryConflict
from ticknet.research.runner import ExperimentRunner, RunnerError
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def _spec(
    *,
    hypothesis: str = "降低回归损失权重应改善横截面排序",
    experiment_type: str = "ablation",
    executor: str = "train_nextday",
    base_config: str = "base.yaml",
    inputs: dict | None = None,
    config_overrides: dict | None = None,
    seeds: tuple[int, ...] = (0,),
    success_gates: tuple[MetricGate, ...] | None = None,
    parent_id: str | None = None,
    stage: str = "screening",
) -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis=hypothesis,
        objective="验证受控实验是否改善主要指标",
        experiment_type=experiment_type,
        executor=executor,
        base_config=base_config,
        inputs=inputs or {},
        config_overrides=(
            {"regression_loss_weight": 0.2} if config_overrides is None else config_overrides
        ),
        seeds=seeds,
        primary_metrics=("validation.daily_rank_ic_mean",),
        success_gates=(
            (MetricGate("validation.daily_rank_ic_mean", "gt", 0.0),)
            if success_gates is None
            else success_gates
        ),
        rationale="受控测试",
        falsification_condition="验证 Rank IC 均值不大于 0 则否定",
        parent_id=parent_id,
        novelty_signature="test-regression-weight",
        stage=stage,
    )


def _manifest_and_config(tmp_path: Path, *, max_date: str = "2025-12-31") -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "samples": [
                    {"trading_date": "2024-01-02", "symbol": "600000"},
                    {"trading_date": max_date, "symbol": "600001"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text(
        f"manifest_path: {manifest}\nseed: 0\n",
        encoding="utf-8",
    )


def _result(experiment_id: str, spec: ExperimentSpec) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        spec=spec,
        status="completed",
        git_sha="abc123",
        dataset_fingerprint="fingerprint",
        per_seed_metrics=[{"validation": {"daily_rank_ic_mean": 0.02}}],
        artifact_dir=f"/tmp/{experiment_id}",
        evaluation_decision="EXTEND",
    )


def test_spec_v2_validates_and_rejects_arbitrary_entry_point() -> None:
    _spec().validate()
    values = _spec().to_dict()
    values["entry_point"] = "python /tmp/arbitrary.py"
    with pytest.raises(ValueError, match="未知字段"):
        ExperimentSpec.from_dict(values)


def test_spec_rejects_unknown_executor_and_missing_gate() -> None:
    with pytest.raises(ValueError, match="executor"):
        _spec(executor="shell").validate()
    with pytest.raises(ValueError, match="success_gates"):
        _spec(success_gates=()).validate()


def test_spec_rejects_deterministic_executor_with_multiple_seeds() -> None:
    spec = _spec(
        experiment_type="data_audit",
        executor="audit_predictions",
        base_config="",
        inputs={"predictions_path": "predictions.parquet"},
        config_overrides={},
        seeds=(0, 1),
    )
    with pytest.raises(ValueError, match="只允许 seed 0"):
        spec.validate()


def test_evaluation_decisions_are_deterministic() -> None:
    gate = (MetricGate("validation.daily_rank_ic_mean", "gt", 0.01),)
    passing = [
        {"validation": {"daily_rank_ic_mean": 0.02}},
        {"validation": {"daily_rank_ic_mean": 0.04}},
    ]
    assert evaluate_metric_gates(passing, gate, stage="screening").decision == "EXTEND"
    assert evaluate_metric_gates(passing, gate, stage="robustness").decision == "KEEP"
    assert evaluate_metric_gates([{}], gate, stage="release").decision == "DISCARD"


def test_policy_rejects_forbidden_unknown_and_excess_seed_fields() -> None:
    policy = ResearchPolicy()
    with pytest.raises(PolicyViolation, match="禁止修改"):
        policy.validate(_spec(config_overrides={"test_end": "2025-12-31"}))
    with pytest.raises(PolicyViolation, match="不允许修改"):
        policy.validate(_spec(config_overrides={"totally_new_param": 1}))
    with pytest.raises(PolicyViolation, match="预算超限"):
        policy.validate(_spec(seeds=(0, 1, 2, 3)))
    policy.validate(_spec(config_overrides={"lr": 0.0005, "dropout": 0.2}))


def test_registry_recursively_records_metrics_and_prevents_duplicates(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    spec = _spec()
    registry.record_experiment("EXP-001", _result("EXP-001", spec), spec)
    registry.record_run("EXP-001", 0, "completed", 12.5, "/tmp/result.json")
    registry.record_metrics(
        "EXP-001",
        0,
        {"validation": {"daily_rank_ic_mean": 0.02}, "test": {"mcc": 0.1}},
    )
    registry.record_review("EXP-001", "evaluation", "EXTEND", {"note": "ok"})
    metrics = {row["metric"]: row["value"] for row in registry.get_metrics("EXP-001")}
    assert metrics == {"test.mcc": 0.1, "validation.daily_rank_ic_mean": 0.02}
    with pytest.raises(RegistryConflict, match="run 已存在"):
        registry.record_run("EXP-001", 0, "completed", 1.0, None)
    with pytest.raises(RegistryConflict, match="metric 重复"):
        registry.record_metrics("EXP-001", 0, {"validation": {"daily_rank_ic_mean": 0.03}})
    with pytest.raises(RegistryConflict, match="review 重复"):
        registry.record_review("EXP-001", "evaluation", "KEEP")
    registry.close()


def test_registry_enforces_parent_and_experiment_uniqueness(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    parent = _spec()
    registry.record_experiment("EXP-000", _result("EXP-000", parent), parent)
    child = _spec(parent_id="EXP-000")
    registry.record_experiment("EXP-001", _result("EXP-001", child), child)
    fetched = registry.get_experiment("EXP-001")
    assert fetched is not None
    assert fetched["parent_id"] == "EXP-000"
    with pytest.raises(RegistryConflict, match="冲突"):
        registry.create_experiment(
            "EXP-001",
            child,
            status="proposed",
            git_sha="x",
            artifact_dir="/tmp/x",
        )
    conflicting = _spec(hypothesis="不同假设")
    with pytest.raises(RegistryConflict, match="不同 spec"):
        registry.record_experiment("EXP-001", _result("EXP-001", conflicting), conflicting)
    orphan = _spec(parent_id="EXP-MISSING")
    with pytest.raises(RegistryConflict, match="parent_id"):
        registry.create_experiment(
            "EXP-ORPHAN",
            orphan,
            status="proposed",
            git_sha="x",
            artifact_dir="/tmp/orphan",
        )
    registry.close()


def test_runner_policy_rejection_is_registered_before_execution(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    with pytest.raises(PolicyViolation, match="禁止修改"):
        runner.run(
            _spec(config_overrides={"evaluate_test": True}),
            experiment_id="EXP-BLOCKED",
        )
    fetched = registry.get_experiment("EXP-BLOCKED")
    assert fetched is not None
    assert fetched["status"] == "rejected"
    assert registry.get_reviews("EXP-BLOCKED")[0]["review_type"] == "policy"
    registry.close()


def test_runner_rejects_preexisting_artifact_directory(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    artifact_dir = tmp_path / "artifacts" / "EXP-CONFLICT"
    artifact_dir.mkdir(parents=True)
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    with pytest.raises(RunnerError, match="artifact 目录已存在"):
        runner.reserve(_spec(), experiment_id="EXP-CONFLICT")
    assert registry.get_experiment("EXP-CONFLICT") is None
    registry.close()


def test_runner_executes_typed_command_and_records_artifacts(tmp_path) -> None:
    _manifest_and_config(tmp_path)
    mock_script = tmp_path / "mock_train.py"
    mock_script.write_text(
        "import json\n"
        "print(json.dumps({'validation': {'"
        "daily_rank_ic_mean': 0.03}, 'dataset_fingerprint': 'fp'}))\n",
        encoding="utf-8",
    )
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["python", str(mock_script)]},
    )
    result = runner.run(_spec(config_overrides={}), experiment_id="EXP-MOCK")
    assert result.status == "completed"
    assert result.evaluation_decision == "EXTEND"
    assert result.per_seed_metrics[0]["validation"]["daily_rank_ic_mean"] == 0.03
    fetched = registry.get_experiment("EXP-MOCK")
    assert fetched is not None
    assert fetched["evaluation_decision"] == "EXTEND"
    assert {row["metric"] for row in registry.get_metrics("EXP-MOCK")} >= {
        "validation.daily_rank_ic_mean"
    }
    artifact_names = {
        row["name"]
        for row in registry._connection.execute(
            "SELECT name FROM artifacts WHERE experiment_id = 'EXP-MOCK'"
        )
    }
    assert {"resolved_spec", "run_manifest", "stdout", "stderr", "result"} <= artifact_names
    with pytest.raises(RunnerError, match="已使用"):
        runner.run(_spec(config_overrides={}), experiment_id="EXP-MOCK")
    registry.close()


def test_unimplemented_executor_fails_without_training_fallback(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    spec = _spec(executor="train_ranker", base_config="base.yaml")
    _manifest_and_config(tmp_path)
    with pytest.raises(RunnerError, match="尚未实现"):
        runner.run(spec, experiment_id="EXP-RANKER")
    fetched = registry.get_experiment("EXP-RANKER")
    assert fetched is not None
    assert fetched["status"] == "failed"
    assert registry.get_runs("EXP-RANKER")[0]["status"] == "failed"
    registry.close()
