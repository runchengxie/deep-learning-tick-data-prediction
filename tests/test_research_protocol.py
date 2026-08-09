"""Research Protocol 与 Locked Test 隔离测试。"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ticknet.research.locked import (
    LockedTestApproval,
    LockedTestFailed,
    LockedTestNotApproved,
    issue_locked_test_approval,
    run_locked_test,
)
from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def _research_spec(*, base_config: str = "base.yaml", stage: str = "screening") -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="test",
        objective="test protocol",
        experiment_type="ablation",
        executor="train_nextday",
        base_config=base_config,
        seeds=(0,),
        primary_metrics=("validation.daily_rank_ic_mean",),
        success_gates=(MetricGate("validation.daily_rank_ic_mean", "gt", 0.0),),
        rationale="protocol test",
        falsification_condition="IC 不改善则否定",
        novelty_signature="protocol-test",
        stage=stage,
    )


def _write_manifest(tmp_path, max_trading_date: str) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "samples": [
                    {"trading_date": "2024-01-02", "symbol": "600000"},
                    {"trading_date": max_trading_date, "symbol": "600001"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_protocol_accepts_research_only_manifest(tmp_path):
    manifest = _write_manifest(tmp_path, "2025-12-31")
    protocol = ResearchProtocol()
    protocol.assert_research_safe(manifest)


def test_protocol_rejects_locked_manifest(tmp_path):
    manifest = _write_manifest(tmp_path, "2026-01-05")
    protocol = ResearchProtocol()
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        protocol.assert_research_safe(manifest)


def test_protocol_rejects_missing_manifest(tmp_path):
    protocol = ResearchProtocol()
    with pytest.raises(PolicyViolation, match="manifest 不存在"):
        protocol.assert_research_safe(tmp_path / "nope.json")


def test_protocol_rejects_locked_prediction_table(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    predictions = tmp_path / "locked.parquet"
    pq.write_table(
        pa.table(
            {
                "trading_date": ["2025-12-31"],
                "label_date": ["2026-01-02"],
            }
        ),
        predictions,
    )
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        ResearchProtocol().assert_predictions_safe(predictions)


def test_protocol_rejects_hidden_locked_return_end_date(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    predictions = tmp_path / "locked-return-end.parquet"
    pq.write_table(
        pa.table(
            {
                "trading_date": ["2025-12-29"],
                "label_date": ["2025-12-31"],
                "return_end_date": ["2026-01-02"],
            }
        ),
        predictions,
    )
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        ResearchProtocol().assert_predictions_safe(predictions)


def test_policy_validate_manifest_delegates_to_protocol(tmp_path):
    policy = ResearchPolicy()
    protocol = ResearchProtocol()
    locked_manifest = _write_manifest(tmp_path, "2026-01-01")
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        policy.validate_manifest(locked_manifest, protocol)


def test_runner_blocks_locked_manifest(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["python", "-c", "print('{}')"]},
    )
    manifest = _write_manifest(tmp_path, "2026-03-01")
    base_config = tmp_path / "base.yaml"
    base_config.write_text(f"manifest_path: {manifest}\n", encoding="utf-8")
    spec = _research_spec()
    with pytest.raises(PolicyViolation, match="锁定测试期"):
        runner.run(spec, experiment_id="EXP-LOCKED-BLOCKED")
    registry.close()


def test_protocol_loads_versioned_yaml(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(
        "protocol_version: test-v2\n"
        "research_end: '2024-12-31'\n"
        "validation_end: '2025-12-31'\n"
        "locked_start: '2026-01-01'\n",
        encoding="utf-8",
    )
    protocol = ResearchProtocol.from_yaml(path)
    assert protocol.protocol_version == "test-v2"
    assert protocol.research_end == "2024-12-31"
    assert protocol.validation_end == "2025-12-31"
    assert protocol.locked_start == "2026-01-01"


def test_protocol_yaml_rejects_unknown_field(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(
        "protocol_version: test-v2\n"
        "research_end: '2024-12-31'\n"
        "validation_end: '2025-12-31'\n"
        "locked_start: '2026-01-01'\n"
        "allow_locked: true\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyViolation, match="未知字段"):
        ResearchProtocol.from_yaml(path)


def test_protocol_rejects_overlapping_boundaries():
    with pytest.raises(ValueError, match="日期边界"):
        ResearchProtocol(
            research_end="2025-12-31",
            validation_end="2026-01-01",
            locked_start="2026-01-01",
        )


def _write_predictions(path, *, seed: int = 1) -> None:
    rng = np.random.RandomState(seed)
    n = 100
    scores = rng.randn(n).astype(np.float64)
    returns = (0.3 * scores + 0.01 * rng.randn(n)).astype(np.float64)
    pq.write_table(
        pa.table(
            {
                "symbol": [f"{600000 + i:06d}" for i in range(n)],
                "trading_date": ["2026-01-02"] * n,
                "label_date": ["2026-01-03"] * n,
                "target_return": returns,
                "score": scores,
            }
        ),
        path,
    )


def _prepare_locked_candidate(tmp_path, registry, experiment_id="EXP-LOCKED-OK"):
    predictions = tmp_path / "predictions.parquet"
    _write_predictions(predictions)
    spec = _research_spec(stage="release")
    registry.record_experiment(
        experiment_id,
        ExperimentResult(
            experiment_id=experiment_id,
            spec=spec,
            status="completed",
            git_sha="abc",
            dataset_fingerprint="fp",
            per_seed_metrics=[],
            artifact_dir=str(tmp_path),
            evaluation_decision="KEEP",
        ),
        spec,
    )
    checkpoint = tmp_path / "model.best.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    registry.record_artifact(experiment_id, 0, "best_checkpoint", checkpoint)
    return predictions, checkpoint


def test_locked_test_requires_issued_approval(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions, _checkpoint = _prepare_locked_candidate(tmp_path, registry)
    with pytest.raises(LockedTestNotApproved, match="无效"):
        run_locked_test(
            predictions,
            approval=LockedTestApproval(token="APPROVED"),
            registry=registry,
            experiment_id="EXP-LOCKED-OK",
        )
    registry.close()


def test_locked_test_approval_is_bound_consumed_and_not_stored_raw(tmp_path):
    registry_path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(registry_path)
    predictions, _checkpoint = _prepare_locked_candidate(tmp_path, registry)
    issued = issue_locked_test_approval(
        predictions,
        registry=registry,
        experiment_id="EXP-LOCKED-OK",
        reason="正式发布前复核",
        approved_by="risk-reviewer",
    )
    assert issued["token"] != "APPROVED"
    assert issued["token"].encode() not in registry_path.read_bytes()
    experiment = registry.get_experiment("EXP-LOCKED-OK")
    assert experiment is not None
    assert experiment["status"] == "frozen"
    with pytest.raises(LockedTestNotApproved, match="completed"):
        issue_locked_test_approval(
            predictions,
            registry=registry,
            experiment_id="EXP-LOCKED-OK",
            reason="不得重复签发",
            approved_by="risk-reviewer",
        )

    result = run_locked_test(
        predictions,
        approval=LockedTestApproval(token=issued["token"]),
        registry=registry,
        experiment_id="EXP-LOCKED-OK",
        min_symbols_per_day=50,
    )
    assert result["mode"] == "locked_test"
    assert result["audit"]["daily_ic_mean"] > 0
    reviews = registry._connection.execute("SELECT review_type, decision FROM reviews").fetchall()
    types = {row["review_type"] for row in reviews}
    assert "locked_test_approval" in types
    assert "locked_test_result" in types
    approval_row = registry.get_locked_approvals("EXP-LOCKED-OK")[0]
    assert approval_row["status"] == "consumed"
    assert approval_row["consumed_at"]
    experiment = registry.get_experiment("EXP-LOCKED-OK")
    assert experiment is not None
    assert experiment["status"] == "locked_tested"
    with pytest.raises(LockedTestNotApproved, match="已消费"):
        run_locked_test(
            predictions,
            approval=LockedTestApproval(token=issued["token"]),
            registry=registry,
            experiment_id="EXP-LOCKED-OK",
        )
    registry.close()


@pytest.mark.parametrize("mutated", ["predictions", "checkpoint"])
def test_locked_approval_rejects_bound_content_changes(tmp_path, mutated):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions, checkpoint = _prepare_locked_candidate(tmp_path, registry)
    issued = issue_locked_test_approval(
        predictions,
        registry=registry,
        experiment_id="EXP-LOCKED-OK",
        reason="正式发布前复核",
        approved_by="risk-reviewer",
    )
    if mutated == "predictions":
        _write_predictions(predictions, seed=99)
    else:
        checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(LockedTestNotApproved, match="SHA-256"):
        run_locked_test(
            predictions,
            approval=LockedTestApproval(token=issued["token"]),
            registry=registry,
            experiment_id="EXP-LOCKED-OK",
        )
    assert registry.get_locked_approvals("EXP-LOCKED-OK")[0]["status"] == "issued"
    registry.close()


def test_locked_test_failure_still_consumes_and_records_approval(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions, _checkpoint = _prepare_locked_candidate(tmp_path, registry)
    issued = issue_locked_test_approval(
        predictions,
        registry=registry,
        experiment_id="EXP-LOCKED-OK",
        reason="正式发布前复核",
        approved_by="risk-reviewer",
    )
    with pytest.raises(LockedTestFailed, match="没有可审计"):
        run_locked_test(
            predictions,
            approval=LockedTestApproval(token=issued["token"]),
            registry=registry,
            experiment_id="EXP-LOCKED-OK",
            min_symbols_per_day=101,
        )
    assert registry.get_locked_approvals("EXP-LOCKED-OK")[0]["status"] == "consumed"
    experiment = registry.get_experiment("EXP-LOCKED-OK")
    assert experiment is not None
    assert experiment["status"] == "locked_test_failed"
    result_reviews = [
        row
        for row in registry.get_reviews("EXP-LOCKED-OK")
        if row["review_type"] == "locked_test_result"
    ]
    assert result_reviews[0]["decision"] == "FAILED"
    registry.close()


@pytest.mark.parametrize(
    ("stage", "decision", "fingerprint", "has_checkpoint", "message"),
    [
        ("screening", "KEEP", "fp", True, "stage=release"),
        ("release", "EXTEND", "fp", True, "Evaluation=KEEP"),
        ("release", "KEEP", None, True, "dataset_fingerprint"),
        ("release", "KEEP", "fp", False, "checkpoint artifact"),
    ],
)
def test_locked_approval_requires_frozen_release_evidence(
    tmp_path,
    stage,
    decision,
    fingerprint,
    has_checkpoint,
    message,
):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions = tmp_path / "predictions.parquet"
    _write_predictions(predictions)
    spec = _research_spec(stage=stage)
    registry.record_experiment(
        "EXP-NOT-READY",
        ExperimentResult(
            experiment_id="EXP-NOT-READY",
            spec=spec,
            status="completed",
            git_sha="abc",
            dataset_fingerprint=fingerprint,
            per_seed_metrics=[],
            artifact_dir=str(tmp_path),
            evaluation_decision=decision,
        ),
        spec,
    )
    if has_checkpoint:
        checkpoint = tmp_path / "candidate.pt"
        checkpoint.write_bytes(b"checkpoint")
        registry.record_artifact("EXP-NOT-READY", 0, "best_checkpoint", checkpoint)
    with pytest.raises(LockedTestNotApproved, match=message):
        issue_locked_test_approval(
            predictions,
            registry=registry,
            experiment_id="EXP-NOT-READY",
            reason="不应批准",
            approved_by="risk-reviewer",
        )
    registry.close()
