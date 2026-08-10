"""多周期 validation CLI 测试。"""

from argparse import Namespace
from pathlib import Path

import pytest

from ticknet.nextday import horizon_cli
from ticknet.nextday.config import NextDayConfig


def test_training_config_arguments_only_apply_explicit_overrides(tmp_path: Path) -> None:
    arguments = Namespace(
        config=tmp_path / "training.yaml",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_dir=None,
        device="cuda",
        num_workers=4,
        verify_data_checksums=False,
    )
    assert horizon_cli._training_config_arguments(arguments) == [
        "--config",
        str(arguments.config),
        "--manifest-path",
        str(arguments.manifest_path),
        "--device",
        "cuda",
        "--num-workers",
        "4",
        "--no-verify-data-checksums",
    ]


def test_main_loads_config_and_runs_validation(tmp_path: Path, monkeypatch) -> None:
    config = NextDayConfig(
        manifest_path=str(tmp_path / "manifest.json"),
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        evaluate_test=False,
    )
    loaded_arguments = []
    calls = []

    def fake_load_config(arguments):
        loaded_arguments.append(arguments)
        return config

    def fake_evaluate(training_config, sidecar, **kwargs):
        calls.append((training_config, sidecar, kwargs))
        return {}

    monkeypatch.setattr(horizon_cli, "load_config", fake_load_config)
    monkeypatch.setattr(horizon_cli, "evaluate_validation_horizons", fake_evaluate)
    horizon_cli.main(
        [
            "--config",
            str(tmp_path / "training.yaml"),
            "--sidecar",
            str(tmp_path / "targets.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seeds",
            "2",
            "3",
            "--horizons",
            "1",
            "5",
            "--source-revision",
            "abc123",
        ]
    )

    assert loaded_arguments == [["--config", str(tmp_path / "training.yaml")]]
    assert calls[0][0] is config
    assert calls[0][1] == tmp_path / "targets.json"
    assert calls[0][2]["seeds"] == [2, 3]
    assert calls[0][2]["horizons"] == [1, 5]
    assert calls[0][2]["source_revision"] == "abc123"


def test_main_rejects_test_enabled_config(tmp_path: Path, monkeypatch) -> None:
    config = NextDayConfig(manifest_path="manifest.json", evaluate_test=True)
    monkeypatch.setattr(horizon_cli, "load_config", lambda _arguments: config)
    with pytest.raises(ValueError, match="evaluate_test=False"):
        horizon_cli.main(
            [
                "--config",
                str(tmp_path / "training.yaml"),
                "--sidecar",
                str(tmp_path / "targets.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
