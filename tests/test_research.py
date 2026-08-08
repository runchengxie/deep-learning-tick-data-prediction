"""AgentX 式实验基础设施（ticknet.research）测试。"""

import pytest

from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentResult, ExperimentSpec


def _spec(**overrides) -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis=overrides.get("hypothesis", "降低回归损失权重应改善横截面排序"),
        experiment_type=overrides.get("experiment_type", "ablation"),
        base_config=overrides.get("base_config", "configs/nextday.yaml"),
        config_overrides=overrides.get("config_overrides", {"regression_loss_weight": 0.2}),
        seeds=overrides.get("seeds", (0, 1)),
        primary_metric=overrides.get("primary_metric", "daily_rank_ic_mean"),
        expected_direction=overrides.get("expected_direction", "increase"),
        rationale=overrides.get("rationale", ""),
        falsification_condition=overrides.get("falsification_condition", ""),
        parent_id=overrides.get("parent_id"),
        stage=overrides.get("stage", "screening"),
    )


def test_spec_validate_accepts_valid():
    _spec().validate()


def test_spec_validate_rejects_bad_type():
    with pytest.raises(ValueError, match="experiment_type"):
        _spec(experiment_type="not_a_type").validate()


def test_spec_validate_rejects_empty_hypothesis():
    with pytest.raises(ValueError, match="hypothesis"):
        _spec(hypothesis="").validate()


def test_spec_validate_rejects_bad_metric():
    with pytest.raises(ValueError, match="primary_metric"):
        _spec(primary_metric="not_a_metric").validate()


def test_spec_validate_rejects_empty_seeds():
    with pytest.raises(ValueError, match="seeds"):
        _spec(seeds=()).validate()


def test_policy_rejects_forbidden_field():
    policy = ResearchPolicy()
    spec = _spec(config_overrides={"test_end": "2025-12-31"})
    with pytest.raises(PolicyViolation, match="禁止修改"):
        policy.validate(spec)


def test_policy_rejects_unknown_field():
    policy = ResearchPolicy()
    spec = _spec(config_overrides={"totally_new_param": 1})
    with pytest.raises(PolicyViolation, match="不允许修改"):
        policy.validate(spec)


def test_policy_rejects_too_many_seeds():
    policy = ResearchPolicy()
    spec = _spec(seeds=(0, 1, 2, 3, 4))
    with pytest.raises(PolicyViolation, match="预算超限"):
        policy.validate(spec)


def test_policy_rejects_bad_stage():
    policy = ResearchPolicy()
    spec = _spec(stage="production")
    with pytest.raises(PolicyViolation, match="stage"):
        policy.validate(spec)


def test_policy_accepts_allowed_overrides():
    policy = ResearchPolicy()
    spec = _spec(config_overrides={"lr": 0.0005, "dropout": 0.2})
    policy.validate(spec)


def test_registry_roundtrip(tmp_path):
    database = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(database)
    spec = _spec()
    result = ExperimentResult(
        experiment_id="EXP-001",
        spec=spec,
        status="completed",
        git_sha="abc123",
        dataset_fingerprint="fingerprint",
        per_seed_metrics=[{"daily_rank_ic_mean": 0.02}],
        artifact_dir="/tmp/EXP-001",
    )
    registry.record_experiment("EXP-001", result, spec)
    registry.record_run("EXP-001", 0, "completed", 12.5, None)
    registry.record_metrics("EXP-001", 0, {"daily_rank_ic_mean": 0.02, "mcc": 0.1})
    registry.record_review("EXP-001", "policy", "approved", {"note": "ok"})
    fetched = registry.get_experiment("EXP-001")
    assert fetched is not None
    assert fetched["experiment_id"] == "EXP-001"
    assert fetched["hypothesis"] == spec.hypothesis
    listed = registry.list_experiments()
    assert len(listed) == 1
    registry.close()


def test_registry_parent_link(tmp_path):
    database = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(database)
    child = _spec(parent_id="EXP-000")
    result = ExperimentResult(
        experiment_id="EXP-001",
        spec=child,
        status="completed",
        git_sha="x",
        dataset_fingerprint=None,
        per_seed_metrics=[],
        artifact_dir="/tmp",
    )
    registry.record_experiment("EXP-001", result, child)
    fetched = registry.get_experiment("EXP-001")
    assert fetched is not None
    assert fetched["parent_id"] == "EXP-000"
    registry.close()


def test_runner_policy_violation_blocks_execution(tmp_path):
    database = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(database)
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        entry_points={"nextday": "echo", "minute_tcn": "echo"},
    )
    spec = _spec(config_overrides={"evaluate_test": True})
    with pytest.raises(PolicyViolation, match="禁止修改"):
        runner.run(spec, experiment_id="EXP-BLOCKED")


def test_runner_executes_mock_entry(tmp_path):
    database = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(database)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "manifest_path: ./data/x/manifest.json\nseed: 0\n",
        encoding="utf-8",
    )
    spec = _spec(
        base_config="base.yaml",
        config_overrides={},
        seeds=(0,),
    )

    mock_script = tmp_path / "mock_train.py"
    mock_script.write_text(
        "import json, sys\n"
        "with open(sys.argv[sys.argv.index('--config') + 1]) as f:\n"
        "    pass\n"
        "print(json.dumps({'daily_rank_ic_mean': 0.03, 'dataset_fingerprint': 'fp'}))\n",
        encoding="utf-8",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        entry_points={"ablation": ["python", str(mock_script)]},
    )
    result = runner.run(spec, experiment_id="EXP-MOCK")
    assert result.status == "completed"
    assert result.git_sha == "unknown"
    assert result.per_seed_metrics[0]["daily_rank_ic_mean"] == 0.03
    assert registry.get_experiment("EXP-MOCK") is not None
    registry.close()
