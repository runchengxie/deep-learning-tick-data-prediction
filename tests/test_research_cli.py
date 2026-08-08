"""ticknet-research v2 CLI 的严格 spec、show、compare 与 token 测试。"""

from pathlib import Path

import pytest
import yaml

from ticknet.research.cli import build_parser, main
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def _spec() -> ExperimentSpec:
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


def test_cli_parser_has_expected_subcommands_and_requires_locked_token(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    captured = capsys.readouterr()
    for command in ("run", "show", "compare", "agent-step"):
        assert command in captured.out
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "locked-test",
                "--predictions",
                "x.parquet",
                "--id",
                "EXP-X",
                "--reason",
                "test",
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
