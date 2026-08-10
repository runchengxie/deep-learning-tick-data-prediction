"""Colab CLI 与 rclone 无人值守调度测试。"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import colab_multi_horizon_job as colab_job
from scripts import run_colab_nextday as colab_runner
from scripts.colab_multi_horizon_job import _drive_path
from scripts.run_colab_nextday import (
    REMOTE_RCLONE_CONFIG,
    _build_committed_wheel,
    _colab_command,
    _ensure_secret_outside_repository,
    _require_executable,
    build_job_spec,
)


def _arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        drive_root="deep-learning-tick-data-prediction",
        rclone_remote="gdrive",
        seeds=[0, 1, 2],
        horizons=[1, 3, 5],
        inference_batch_size=128,
        session="ticknet-test",
        gpu="T4",
        config=tmp_path / "config.yaml",
        rclone_config=tmp_path / "rclone.conf",
        timeout=60.0,
    )


def test_job_spec_preserves_checkpoint_signature_paths(tmp_path: Path) -> None:
    spec = build_job_spec(_arguments(tmp_path), "abc123")
    assert spec["feature_local"] == "/content/nextday-raw-200"
    assert spec["checkpoint_local"] == (
        "/content/drive/MyDrive/deep-learning-tick-data-prediction/ticknet-runs/raw-200-capacity_1m"
    )
    assert spec["rclone_config"] == REMOTE_RCLONE_CONFIG
    assert "token" not in json.dumps(spec, ensure_ascii=False).lower()
    assert spec["source_revision"] == "abc123"
    assert spec["seeds"] == [0, 1, 2]


def test_colab_commands_pin_oauth_provider() -> None:
    assert _colab_command("colab", "sessions") == [
        "colab",
        "--auth=oauth2",
        "sessions",
    ]


def test_executable_falls_back_to_user_local_bin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / ".local" / "bin" / "colab"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(colab_runner.shutil, "which", lambda _name: None)

    assert _require_executable("colab", home=tmp_path) == str(executable)


def test_colab_rclone_copy_uses_ubuntu_compatible_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        captured.append(command)

    monkeypatch.setattr(colab_job, "_run", fake_run)
    colab_job._rclone_copy("source", "destination", env={})

    assert captured[0][:3] == ["rclone", "copy", "source"]
    assert "--metadata" not in captured[0]


def test_wheel_build_uses_committed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    staging = tmp_path / "staging"
    repository.mkdir()
    staging.mkdir()
    commands: list[tuple[list[str], Path | None]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> None:
        commands.append((command, cwd))
        if command[:2] == ["git", "archive"]:
            Path(command[command.index("--output") + 1]).touch()
        if "build" in command:
            output_dir = Path(command[command.index("--out-dir") + 1])
            output_dir.mkdir()
            (output_dir / "ticknet.whl").touch()

    monkeypatch.setattr(colab_runner, "_run", fake_run)
    monkeypatch.setattr(colab_runner.shutil, "unpack_archive", lambda *_args: None)

    wheel = _build_committed_wheel("uv", repository, staging, "abc123")

    assert wheel == staging / "dist" / "ticknet.whl"
    assert commands[0][0][-1] == "abc123"
    assert commands[0][1] == repository
    assert commands[1][1] == staging / "source"


def test_secret_must_live_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    secret = repository / "rclone.conf"
    secret.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="不能放在 Git 仓库内"):
        _ensure_secret_outside_repository(secret, repository)

    external = tmp_path / "external-rclone.conf"
    external.write_text("secret", encoding="utf-8")
    _ensure_secret_outside_repository(external, repository)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "deep-learning-tick-data-prediction/ticknet-data",
            "gdrive:deep-learning-tick-data-prediction/ticknet-data",
        ),
        ("folder/file.pt", "gdrive:folder/file.pt"),
    ],
)
def test_drive_path_accepts_safe_relative_paths(path: str, expected: str) -> None:
    assert _drive_path("gdrive", path) == expected


@pytest.mark.parametrize("path", ["/absolute/path", "../escape", "folder/../../escape"])
def test_drive_path_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="安全的相对路径"):
        _drive_path("gdrive", path)
