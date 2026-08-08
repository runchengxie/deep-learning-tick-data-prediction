"""Research Protocol 与 Locked Test 隔离测试。"""

import json

import pytest

from ticknet.research.locked import LockedTestApproval, LockedTestNotApproved, run_locked_test
from ticknet.research.policy import PolicyViolation, ResearchPolicy
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentSpec


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
        entry_points={"ablation": ["python", "-c", "print('{}')"]},
    )
    manifest = _write_manifest(tmp_path, "2026-03-01")
    base_config = tmp_path / "base.yaml"
    base_config.write_text(f"manifest_path: {manifest}\n", encoding="utf-8")
    spec = ExperimentSpec(
        hypothesis="test",
        experiment_type="ablation",
        base_config="base.yaml",
        seeds=(0,),
    )
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


def test_locked_test_requires_approval(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions = tmp_path / "predictions.parquet"
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(0)
    pq.write_table(
        pa.table(
            {
                "symbol": [f"{600000 + i:06d}" for i in range(100)],
                "trading_date": ["2025-01-02"] * 100,
                "label_date": ["2025-01-03"] * 100,
                "target_return": rng.randn(100).astype(np.float64),
                "score": rng.randn(100).astype(np.float64),
            }
        ),
        predictions,
    )
    with pytest.raises(LockedTestNotApproved, match="批准"):
        run_locked_test(
            predictions,
            approval=LockedTestApproval(reason="", token="NOT_APPROVED"),
            registry=registry,
            experiment_id="EXP-LOCKED",
        )
    registry.close()


def test_locked_test_runs_with_approval_and_records(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    predictions = tmp_path / "predictions.parquet"
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.RandomState(1)
    n = 100
    scores = rng.randn(n).astype(np.float64)
    returns = (0.3 * scores + 0.01 * rng.randn(n)).astype(np.float64)
    pq.write_table(
        pa.table(
            {
                "symbol": [f"{600000 + i:06d}" for i in range(n)],
                "trading_date": ["2025-01-02"] * n,
                "label_date": ["2025-01-03"] * n,
                "target_return": returns,
                "score": scores,
            }
        ),
        predictions,
    )
    result = run_locked_test(
        predictions,
        approval=LockedTestApproval(reason="正式发布前复核", token="APPROVED"),
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
    registry.close()
