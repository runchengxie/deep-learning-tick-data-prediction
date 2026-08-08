"""ticknet-research CLI 测试：解析、policy 拒绝、show/compare。"""

from pathlib import Path

import pytest
import yaml

from ticknet.research.cli import build_parser, main
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.spec import ExperimentResult, ExperimentSpec


def _write_spec(tmp_path: Path, **overrides) -> Path:
    values = {
        "hypothesis": "降低回归损失权重应改善横截面排序",
        "experiment_type": "ablation",
        "base_config": "configs/nextday.yaml",
        "config_overrides": {"regression_loss_weight": 0.2},
        "seeds": [0, 1],
        "entry_point": "ticknet-nextday-train",
        "stage": "screening",
    }
    values.update(overrides)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding="utf-8")
    return path


def _seed_registry(tmp_path: Path, experiment_id: str) -> Path:
    registry_path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(registry_path)
    spec = ExperimentSpec(
        hypothesis="test",
        experiment_type="ablation",
        base_config="configs/nextday.yaml",
        seeds=(0,),
    )
    result = ExperimentResult(
        experiment_id=experiment_id,
        spec=spec,
        status="completed",
        git_sha="abc",
        dataset_fingerprint="fp",
        per_seed_metrics=[{"daily_rank_ic_mean": 0.02}],
        artifact_dir="/tmp",
    )
    registry.record_experiment(experiment_id, result, spec)
    registry.record_metrics(experiment_id, 0, {"daily_rank_ic_mean": 0.02})
    registry.close()
    return registry_path


def test_cli_parser_has_expected_subcommands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    captured = capsys.readouterr()
    for command in ("run", "show", "compare", "agent-step"):
        assert command in captured.out


def test_cli_agent_step_runs_a_research_round(tmp_path, capsys):
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


def test_cli_run_rejects_policy_violation(tmp_path):
    spec_path = _write_spec(
        tmp_path,
        config_overrides={"test_end": "2025-12-31"},
    )
    with pytest.raises(SystemExit, match="POLICY_REJECTED"):
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


def test_cli_show_reports_missing_experiment(tmp_path):
    with pytest.raises(SystemExit, match="找不到实验"):
        main(
            [
                "--root",
                str(tmp_path),
                "--registry",
                str(tmp_path / "registry.sqlite"),
                "--artifacts",
                str(tmp_path / "artifacts"),
                "show",
                "--id",
                "EXP-MISSING",
            ]
        )


def test_cli_show_returns_registered_experiment(tmp_path, capsys):
    registry_path = _seed_registry(tmp_path, "EXP-SEED")
    main(
        [
            "--root",
            str(tmp_path),
            "--registry",
            str(registry_path),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "show",
            "--id",
            "EXP-SEED",
        ]
    )
    captured = capsys.readouterr()
    assert "EXP-SEED" in captured.out


def test_cli_compare_lists_metrics(tmp_path, capsys):
    registry_path = _seed_registry(tmp_path, "EXP-A")
    main(
        [
            "--root",
            str(tmp_path),
            "--registry",
            str(registry_path),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "compare",
            "--ids",
            "EXP-A",
        ]
    )
    captured = capsys.readouterr()
    assert "EXP-A" in captured.out
    assert "daily_rank_ic_mean" in captured.out
