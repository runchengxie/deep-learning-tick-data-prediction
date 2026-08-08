"""ticknet-research v2 CLI 的严格 spec、show、compare 与 token 测试。"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ticknet.research.cli import build_parser, main
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def _spec(*, stage: str = "screening") -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="test",
        objective="test objective",
        experiment_type="ablation",
        executor="train_nextday",
        base_config="base.yaml",
        seeds=(0,),
        primary_metrics=("validation.daily_rank_ic_mean",),
        success_gates=(MetricGate("validation.daily_rank_ic_mean", "gt", 0.0),),
        rationale="test rationale",
        falsification_condition="IC 不改善则否定",
        novelty_signature="cli-test",
        stage=stage,
    )


def _write_spec(tmp_path: Path, *, config_overrides: dict | None = None) -> Path:
    values = _spec().to_dict()
    values["config_overrides"] = config_overrides or {"regression_loss_weight": 0.2}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    return path


def _seed_registry(tmp_path: Path, experiment_id: str) -> Path:
    registry_path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(registry_path)
    spec = _spec()
    result = ExperimentResult(
        experiment_id=experiment_id,
        spec=spec,
        status="completed",
        git_sha="abc",
        dataset_fingerprint="fp",
        per_seed_metrics=[{"validation": {"daily_rank_ic_mean": 0.02}}],
        artifact_dir="/tmp",
        evaluation_decision="EXTEND",
    )
    registry.record_experiment(experiment_id, result, spec)
    registry.record_metrics(
        experiment_id,
        0,
        {"validation": {"daily_rank_ic_mean": 0.02}},
    )
    registry.close()
    return registry_path


def test_cli_parser_separates_locked_approval_and_consumption(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    captured = capsys.readouterr()
    for command in (
        "run",
        "show",
        "compare",
        "context",
        "agent-step",
        "approve-locked-test",
        "locked-test",
    ):
        assert command in captured.out
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "approve-locked-test",
                "--predictions",
                "x.parquet",
                "--id",
                "EXP-X",
                "--reason",
                "test",
            ]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "locked-test",
                "--predictions",
                "x.parquet",
                "--id",
                "EXP-X",
            ]
        )


def test_cli_agent_step_returns_structured_status(tmp_path, capsys) -> None:
    main(
        [
            "--root",
            str(tmp_path),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "agent-step",
            "--question",
            "测试研究问题",
        ]
    )
    captured = capsys.readouterr()
    assert "status" in captured.out
    assert "spec" in captured.out
    assert "context_fingerprint" in captured.out


def test_cli_issues_and_consumes_locked_approval(tmp_path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    predictions = tmp_path / "locked.parquet"
    scores = np.linspace(-1.0, 1.0, 60)
    pq.write_table(
        pa.table(
            {
                "symbol": [f"{600000 + index:06d}" for index in range(60)],
                "trading_date": ["2026-01-02"] * 60,
                "label_date": ["2026-01-03"] * 60,
                "target_return": scores * 0.02,
                "score": scores,
            }
        ),
        predictions,
    )
    registry = ExperimentRegistry(registry_path)
    spec = _spec(stage="release")
    registry.record_experiment(
        "EXP-RELEASE",
        ExperimentResult(
            experiment_id="EXP-RELEASE",
            spec=spec,
            status="completed",
            git_sha="abc",
            dataset_fingerprint="dataset-fp",
            per_seed_metrics=[],
            artifact_dir=str(tmp_path),
            evaluation_decision="KEEP",
        ),
        spec,
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry.record_artifact("EXP-RELEASE", 0, "best_checkpoint", checkpoint)
    registry.close()

    common = ["--registry", str(registry_path)]
    main(
        [
            *common,
            "approve-locked-test",
            "--predictions",
            str(predictions),
            "--id",
            "EXP-RELEASE",
            "--reason",
            "最终确认",
            "--approved-by",
            "risk-reviewer",
        ]
    )
    issued = json.loads(capsys.readouterr().out)
    main(
        [
            *common,
            "locked-test",
            "--predictions",
            str(predictions),
            "--id",
            "EXP-RELEASE",
            "--token",
            issued["token"],
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "locked_test"
    assert result["binding"]["checkpoint_count"] == 1


def test_cli_rejects_old_entry_point_and_policy_violation(tmp_path) -> None:
    old = _spec().to_dict()
    old["entry_point"] = "python arbitrary.py"
    old_path = tmp_path / "old.yaml"
    old_path.write_text(yaml.safe_dump(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="未知字段"):
        main(["run", "--spec", str(old_path), "--id", "EXP-OLD"])

    spec_path = _write_spec(tmp_path, config_overrides={"test_end": "2025-12-31"})
    with pytest.raises(SystemExit, match="EXPERIMENT_REJECTED"):
        main(
            [
                "--root",
                str(tmp_path),
                "--registry",
                str(tmp_path / "registry.sqlite"),
                "--artifacts",
                str(tmp_path / "artifacts"),
                "run",
                "--spec",
                str(spec_path),
                "--id",
                "EXP-BLOCKED",
            ]
        )


def test_cli_show_and_compare_registered_experiment(tmp_path, capsys) -> None:
    registry_path = _seed_registry(tmp_path, "EXP-SEED")
    common = [
        "--root",
        str(tmp_path),
        "--registry",
        str(registry_path),
        "--artifacts",
        str(tmp_path / "artifacts"),
    ]
    main([*common, "show", "--id", "EXP-SEED"])
    assert "EXP-SEED" in capsys.readouterr().out
    main([*common, "compare", "--ids", "EXP-SEED"])
    captured = capsys.readouterr()
    assert "EXP-SEED" in captured.out
    assert "validation.daily_rank_ic_mean" in captured.out

    main(
        [
            *common,
            "context",
            "--question",
            "验证 Registry 上下文",
            "--baseline-id",
            "EXP-SEED",
        ]
    )
    context = json.loads(capsys.readouterr().out)
    assert context["baseline_summary"]["experiment_id"] == "EXP-SEED"
    assert context["recent_experiments"][0]["evaluation_decision"] == "EXTEND"
    assert len(context["context_fingerprint"]) == 64
    assert context["data_access"]["locked_test_access"] is False


def test_cli_show_reports_missing_experiment(tmp_path) -> None:
    with pytest.raises(SystemExit, match="找不到实验"):
        main(
            [
                "--root",
                str(tmp_path),
                "--registry",
                str(tmp_path / "registry.sqlite"),
                "show",
                "--id",
                "EXP-MISSING",
            ]
        )
