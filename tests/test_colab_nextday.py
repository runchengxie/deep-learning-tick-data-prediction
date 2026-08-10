"""Colab CLI 与 rclone 无人值守调度测试。"""

import json
import subprocess
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
    _dry_run_plan,
    _ensure_secret_outside_repository,
    _require_executable,
    _session_exists,
    _should_stop_owned_session,
    _validate_lifecycle_arguments,
    _validate_session_selection,
    build_job_spec,
)
from ticknet.nextday.train import load_config


def _arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        workflow="multi-horizon-validation",
        drive_root="deep-learning-tick-data-prediction",
        rclone_remote="gdrive",
        seeds=[0, 1, 2],
        horizons=[1, 3, 5],
        inference_batch_size=128,
        session="ticknet-test",
        gpu="T4",
        config=tmp_path / "config.yaml",
        rclone_config=tmp_path / "rclone.conf",
        local_output_dir=tmp_path / "output",
        timeout=60.0,
        keep_session=False,
        keep_on_failure=False,
        reuse_session=False,
        dry_run=False,
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


def test_h5_training_spec_uses_independent_run_directory(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "h5-train"
    arguments.seeds = [0]

    spec = build_job_spec(arguments, "abc123")

    assert spec["workflow"] == "h5-train"
    assert spec["checkpoint_name"] == "raw-200-dual-head-capacity_1m-h5"
    assert spec["checkpoint_remote"].endswith("raw-200-capacity_1m-h5")
    assert spec["checkpoint_local"] == spec["output_local"]
    assert spec["output_remote"].endswith("raw-200-capacity_1m-h5")
    assert spec["seeds"] == [0]


def test_h5_training_config_keeps_test_locked() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(
        [
            "--config",
            str(repository_root / "configs" / "nextday-raw-200-capacity-1m-h5.yaml"),
        ]
    )

    assert config.target_horizon == 5
    assert config.target_sidecar_path == ("/content/nextday-raw-200-targets-v1/horizon-labels.json")
    assert config.checkpoint_name == "raw-200-dual-head-capacity_1m-h5"
    assert config.evaluate_test is False


def test_colab_commands_pin_oauth_provider() -> None:
    assert _colab_command("colab", "sessions") == [
        "colab",
        "--auth=oauth2",
        "sessions",
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("[ticknet-test] endpoint | Hardware: T4 | Variant: GPU", True),
        ("[colab] Session 'ticknet-test' not found.", False),
    ],
)
def test_session_exists_parses_colab_status(
    output: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colab_runner,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    assert _session_exists("colab", "ticknet-test") is expected


def test_session_exists_rejects_unknown_status_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colab_runner,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "unexpected", ""),
    )
    with pytest.raises(RuntimeError, match="无法识别"):
        _session_exists("colab", "ticknet-test")


@pytest.mark.parametrize(
    ("exists", "reuse_session", "message"),
    [
        (True, False, "已存在"),
        (False, True, "不存在"),
    ],
)
def test_session_selection_requires_explicit_ownership(
    exists: bool,
    reuse_session: bool,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_session_selection(
            session="ticknet-test",
            exists=exists,
            reuse_session=reuse_session,
        )


@pytest.mark.parametrize(
    ("exists", "reuse_session"),
    [(False, False), (True, True)],
)
def test_session_selection_accepts_unambiguous_request(
    exists: bool,
    reuse_session: bool,
) -> None:
    _validate_session_selection(
        session="ticknet-test",
        exists=exists,
        reuse_session=reuse_session,
    )


def test_keep_session_and_keep_on_failure_are_mutually_exclusive(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.keep_session = True
    arguments.keep_on_failure = True
    with pytest.raises(ValueError, match="不能同时使用"):
        _validate_lifecycle_arguments(arguments)


@pytest.mark.parametrize(
    ("keep_session", "keep_on_failure", "succeeded", "expected"),
    [
        (False, False, True, True),
        (False, False, False, True),
        (False, True, True, True),
        (False, True, False, False),
        (True, False, True, False),
        (True, False, False, False),
    ],
)
def test_owned_session_stop_policy(
    tmp_path: Path,
    keep_session: bool,
    keep_on_failure: bool,
    succeeded: bool,
    expected: bool,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.keep_session = keep_session
    arguments.keep_on_failure = keep_on_failure
    assert _should_stop_owned_session(arguments, succeeded=succeeded) is expected


@pytest.mark.parametrize(
    ("reuse_session", "keep_session", "keep_on_failure", "expected_lifecycle"),
    [
        (False, False, False, ["colab", "--auth=oauth2", "stop"]),
        (False, True, False, ["lifecycle", "keep-session", "ticknet-test"]),
        (False, False, True, ["lifecycle", "stop-on-success", "keep-on-failure"]),
        (True, False, False, ["lifecycle", "keep-reused-session", "ticknet-test"]),
    ],
)
def test_dry_run_plan_describes_session_lifecycle(
    tmp_path: Path,
    reuse_session: bool,
    keep_session: bool,
    keep_on_failure: bool,
    expected_lifecycle: list[str],
) -> None:
    arguments = _arguments(tmp_path)
    arguments.reuse_session = reuse_session
    arguments.keep_session = keep_session
    arguments.keep_on_failure = keep_on_failure
    plan = _dry_run_plan(arguments, colab="colab", revision="abc123")

    assert plan[0][2:] == ["status", "-s", "ticknet-test"]
    assert any(command[: len(expected_lifecycle)] == expected_lifecycle for command in plan)
    assert any("rm" in command and REMOTE_RCLONE_CONFIG in command for command in plan)
    assert any("new" in command for command in plan) is not reuse_session


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


def test_h5_training_invokes_each_requested_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._train_h5(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "seeds": [0, 2],
        }
    )

    assert [command[-1] for command in captured] == ["0", "2"]
    assert all("ticknet.nextday.train" in command for command in captured)


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


def _mock_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_exists: bool,
    fail_exec: bool = False,
    fail_secret_upload: bool = False,
) -> tuple[Namespace, list[list[str]]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    arguments = _arguments(tmp_path)
    arguments.config.write_text("evaluate_test: false\n", encoding="utf-8")
    arguments.rclone_config.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if fail_secret_upload and "upload" in command and REMOTE_RCLONE_CONFIG in command:
            raise subprocess.CalledProcessError(1, command)
        if fail_exec and "exec" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_build(
        uv: str,
        repository_root: Path,
        staging: Path,
        revision: str,
    ) -> Path:
        wheel = staging / "ticknet.whl"
        wheel.touch()
        return wheel

    monkeypatch.setattr(colab_runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(colab_runner, "_require_executable", lambda name: name)
    monkeypatch.setattr(colab_runner, "_source_revision", lambda _root: "abc123")
    monkeypatch.setattr(colab_runner, "_require_clean_revision", lambda _root: None)
    monkeypatch.setattr(colab_runner, "_session_exists", lambda *_args: session_exists)
    monkeypatch.setattr(colab_runner, "_build_committed_wheel", fake_build)
    monkeypatch.setattr(colab_runner, "_run", fake_run)
    return arguments, commands


def test_reused_session_is_never_created_or_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, commands = _mock_orchestrator(
        tmp_path,
        monkeypatch,
        session_exists=True,
    )
    arguments.reuse_session = True

    colab_runner.run(arguments)

    assert not any("new" in command for command in commands)
    assert not any("stop" in command for command in commands)
    assert any("exec" in command for command in commands)
    assert any("rm" in command and REMOTE_RCLONE_CONFIG in command for command in commands)


@pytest.mark.parametrize(
    ("keep_session", "keep_on_failure", "expects_stop"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_owned_success_follows_lifecycle_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep_session: bool,
    keep_on_failure: bool,
    expects_stop: bool,
) -> None:
    arguments, commands = _mock_orchestrator(
        tmp_path,
        monkeypatch,
        session_exists=False,
    )
    arguments.keep_session = keep_session
    arguments.keep_on_failure = keep_on_failure

    colab_runner.run(arguments)

    assert any("new" in command for command in commands)
    assert any("stop" in command for command in commands) is expects_stop
    assert any("rm" in command and REMOTE_RCLONE_CONFIG in command for command in commands)


def test_keep_on_failure_preserves_owned_session_and_removes_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, commands = _mock_orchestrator(
        tmp_path,
        monkeypatch,
        session_exists=False,
        fail_exec=True,
    )
    arguments.keep_on_failure = True

    with pytest.raises(subprocess.CalledProcessError):
        colab_runner.run(arguments)

    assert any("new" in command for command in commands)
    assert not any("stop" in command for command in commands)
    assert any("rm" in command and REMOTE_RCLONE_CONFIG in command for command in commands)


def test_partial_secret_upload_still_triggers_cleanup_and_ephemeral_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, commands = _mock_orchestrator(
        tmp_path,
        monkeypatch,
        session_exists=False,
        fail_secret_upload=True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        colab_runner.run(arguments)

    assert any("rm" in command and REMOTE_RCLONE_CONFIG in command for command in commands)
    assert any("stop" in command for command in commands)


def test_existing_session_without_reuse_is_rejected_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, commands = _mock_orchestrator(
        tmp_path,
        monkeypatch,
        session_exists=True,
    )

    with pytest.raises(RuntimeError, match="--reuse-session"):
        colab_runner.run(arguments)

    assert not commands
