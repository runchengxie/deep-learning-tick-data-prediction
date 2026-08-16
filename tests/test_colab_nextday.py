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
    _default_config,
    _dry_run_plan,
    _ensure_secret_outside_repository,
    _remote_wheel_path,
    _require_executable,
    _session_exists,
    _should_stop_owned_session,
    _validate_downloaded_summary,
    _validate_lifecycle_arguments,
    _validate_session_selection,
    build_job_spec,
)
from ticknet.nextday.train import load_config


def _arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        workflow="multi-horizon-validation",
        matrix_cell="1m-raw200",
        drive_root="deep-learning-tick-data-prediction",
        rclone_remote="gdrive",
        seeds=[0, 1, 2],
        horizons=[1, 3, 5],
        inference_batch_size=128,
        benchmark_batches=100,
        warmup_batches=5,
        batch_sizes=[2, 4, 8, 16, 32],
        num_workers=[2, 4, 8, 16],
        effective_batch_size=32,
        training_epochs=None,
        evaluate_test=True,
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


def test_capacity_benchmark_spec_uses_raw1000_and_gpu_specific_output(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "capacity-benchmark"
    arguments.gpu = "A100"

    spec = build_job_spec(arguments, "abc123")

    assert spec["workflow"] == "capacity-benchmark"
    assert spec["feature_remote"].endswith("nextday-raw-1000-preflight-202101-top100")
    assert spec["feature_local"] == "/content/nextday-raw-1000-preflight-202101-top100"
    assert spec["output_remote"].endswith("capacity_100m/benchmarks/a100")
    assert spec["expected_parameter_count"] == 100_817_575
    assert spec["projected_train_samples"] == 75_000
    assert spec["requested_gpu"] == "A100"


def test_raw1000_training_spec_uses_full_dataset_and_resumable_directory(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "raw1000-train"
    arguments.gpu = "A100"
    arguments.seeds = [0]

    spec = build_job_spec(arguments, "abc123")

    assert spec["workflow"] == "raw1000-train"
    assert spec["feature_remote"].endswith("nextday-raw-1000-pilot-2021-2025-top100")
    assert spec["feature_local"] == "/content/nextday-raw-1000-pilot-2021-2025-top100"
    assert spec["checkpoint_name"] == "raw-1000-top100-dual-head-capacity_100m"
    assert spec["checkpoint_remote"].endswith("capacity_100m/training")
    assert spec["checkpoint_local"] == spec["output_local"]
    assert spec["output_remote"].endswith("capacity_100m/training")
    assert spec["projected_train_samples"] == 70_805
    assert spec["seeds"] == [0]


@pytest.mark.parametrize(
    ("cell", "checkpoint_name", "parameter_count"),
    [
        ("1m-raw200", "raw-200-top100-dual-head-capacity_1m-matrix", 1_033_383),
        ("1m-raw1000", "raw-1000-top100-dual-head-capacity_1m-matrix", 1_033_383),
        ("100m-raw200", "raw-200-top100-dual-head-capacity_100m-matrix", 100_817_575),
    ],
)
def test_capacity_matrix_training_spec_uses_shared_top100_dataset(
    tmp_path: Path,
    cell: str,
    checkpoint_name: str,
    parameter_count: int,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "capacity-matrix-train"
    arguments.matrix_cell = cell
    arguments.gpu = "A100"

    spec = build_job_spec(arguments, "abc123")

    assert spec["matrix_cell"] == cell
    assert spec["feature_remote"].endswith("nextday-raw-1000-pilot-2021-2025-top100")
    assert spec["feature_local"] == "/content/nextday-raw-1000-pilot-2021-2025-top100"
    assert spec["checkpoint_name"] == checkpoint_name
    assert spec["checkpoint_remote"].endswith(f"capacity-matrix/{cell}")
    assert spec["checkpoint_local"] == spec["output_local"]
    assert spec["expected_parameter_count"] == parameter_count
    assert spec["projected_train_samples"] == 70_805
    assert _default_config("capacity-matrix-train", cell).name == (
        f"nextday-capacity-matrix-{cell}.yaml"
    )


def test_eventstream_capacity_spec_uses_month_pack_and_exact_model_size(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-capacity-benchmark"
    arguments.gpu = "A100"

    spec = build_job_spec(arguments, "abc123")

    assert spec["feature_remote"].endswith("eventstream-top400-h5-fold0-benchmark-202101")
    assert spec["feature_local"] == "/content/ticknet-eventstream/top400-h5-fold0"
    assert spec["output_remote"].endswith("capacity100m-fold0/benchmarks/a100")
    assert spec["expected_parameter_count"] == 100_604_180
    assert spec["projected_train_samples"] == 40_000


def test_eventstream_recent_capacity_spec_uses_2025_pack(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-recent-capacity-benchmark"
    arguments.gpu = "A100"

    spec = build_job_spec(arguments, "abc123")

    assert spec["feature_remote"].endswith("eventstream-top400-h5-recent-benchmark-202508")
    assert spec["feature_local"] == "/content/ticknet-eventstream/top400-h5-recent"
    assert spec["output_remote"].endswith("capacity100m-recent/benchmarks/a100")
    assert spec["expected_parameter_count"] == 100_604_180
    assert spec["projected_train_samples"] == 42_000


def test_eventstream_recent_training_uses_one_materialized_seed(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-recent-train"
    arguments.gpu = "A100"
    arguments.seeds = [0]
    arguments.training_epochs = 1
    arguments.evaluate_test = False

    spec = build_job_spec(arguments, "abc1234")

    assert spec["feature_remote"].endswith("eventstream-top400-h5-recent-materialized/seed0")
    assert spec["feature_local"] == "/content/ticknet-eventstream/materialized/recent"
    assert spec["checkpoint_local"] == spec["output_local"]
    assert spec["checkpoint_remote"] == spec["output_remote"]
    assert spec["expected_parameter_count"] == 100_604_180
    assert spec["projected_train_samples"] == 120_000
    assert spec["training_epochs"] == 1
    assert spec["evaluate_test"] is False


def test_eventstream_training_requires_one_seed(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-recent-train"

    with pytest.raises(ValueError, match="一个 seed"):
        _validate_lifecycle_arguments(arguments)


def test_eventstream_recent_sweep_spec_projects_full_training_window(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-recent-batch-size-sweep"
    arguments.gpu = "A100"
    arguments.batch_sizes = [8, 16, 32, 64]
    arguments.effective_batch_size = 64

    spec = build_job_spec(arguments, "abc123")

    assert spec["feature_remote"].endswith("eventstream-top400-h5-recent-benchmark-202508")
    assert spec["output_remote"].endswith("capacity100m-recent/batch-size-sweep/a100")
    assert spec["batch_sizes"] == [8, 16, 32, 64]
    assert spec["effective_batch_size"] == 64
    assert spec["expected_parameter_count"] == 100_604_180
    assert spec["projected_train_samples"] == 120_000


def test_eventstream_recent_input_profile_spec_scans_workers(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "eventstream-recent-input-profile"
    arguments.gpu = "A100"
    arguments.num_workers = [2, 4, 8, 16]
    arguments.effective_batch_size = 64

    spec = build_job_spec(arguments, "abc123")

    assert spec["feature_remote"].endswith("eventstream-top400-h5-recent-benchmark-202508")
    assert spec["output_remote"].endswith("capacity100m-recent/input-profile/a100")
    assert spec["num_workers"] == [2, 4, 8, 16]
    assert spec["effective_batch_size"] == 64
    assert spec["expected_parameter_count"] == 100_604_180
    assert spec["projected_train_samples"] == 120_000


def test_batch_size_sweep_spec_keeps_effective_batch_and_separate_output(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.workflow = "batch-size-sweep"
    arguments.gpu = "A100"
    arguments.benchmark_batches = 50

    spec = build_job_spec(arguments, "abc123")

    assert spec["workflow"] == "batch-size-sweep"
    assert spec["feature_remote"].endswith("nextday-raw-1000-preflight-202101-top100")
    assert spec["output_remote"].endswith("capacity_100m/batch-size-sweep/a100")
    assert spec["batch_sizes"] == [2, 4, 8, 16, 32]
    assert spec["effective_batch_size"] == 32
    assert spec["benchmark_batches"] == 50


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


def test_raw1000_training_config_keeps_test_locked_and_uses_sweep_batch() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(
        [
            "--config",
            str(repository_root / "configs" / "nextday-raw-1000-top100-capacity-100m.yaml"),
        ]
    )

    assert config.manifest_path == (
        "/content/nextday-raw-1000-pilot-2021-2025-top100/manifest.json"
    )
    assert config.train_end == "2023-12-31"
    assert config.val_end == "2024-12-31"
    assert config.test_start == "2025-01-01"
    assert config.batch_size == 32
    assert config.gradient_accumulation_steps == 1
    assert config.resume is True
    assert config.evaluate_test is False


def test_colab_commands_pin_oauth_provider() -> None:
    assert _colab_command("colab", "sessions") == [
        "colab",
        "--auth=oauth2",
        "sessions",
    ]


def test_remote_wheel_path_preserves_valid_distribution_filename() -> None:
    wheel = Path("deep_learning_tick_data_prediction-0.2.0-py3-none-any.whl")
    assert _remote_wheel_path(wheel) == (
        "/content/deep_learning_tick_data_prediction-0.2.0-py3-none-any.whl"
    )


def test_downloaded_summary_must_confirm_revision_and_success(tmp_path: Path) -> None:
    spec = {
        "workflow": "h5-train",
        "source_revision": "abc123",
    }
    summary_path = tmp_path / "colab-run-summary.json"
    summary_path.write_text(
        json.dumps({**spec, "status": "complete"}),
        encoding="utf-8",
    )
    _validate_downloaded_summary(tmp_path, spec)

    summary_path.write_text(
        json.dumps({**spec, "status": "failed", "error": "remote failure"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="remote failure"):
        _validate_downloaded_summary(tmp_path, spec)

    summary_path.write_text(
        json.dumps({**spec, "status": "complete", "source_revision": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source_revision 不匹配"):
        _validate_downloaded_summary(tmp_path, spec)


def test_eventstream_summary_confirms_seed_locked_and_oos_status(tmp_path: Path) -> None:
    spec = {
        "workflow": "eventstream-recent-train",
        "source_revision": "abc1234",
        "seeds": [0],
        "evaluate_test": False,
    }
    summary_path = tmp_path / "colab-run-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **spec,
                "status": "complete",
                "test_status": "locked_not_accessed",
                "oos_status": "not_evaluated",
            }
        ),
        encoding="utf-8",
    )
    _validate_downloaded_summary(tmp_path, spec)

    summary_path.write_text(
        json.dumps(
            {
                **spec,
                "status": "complete",
                "test_status": "locked_not_accessed",
                "oos_status": "evaluated",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="OOS 状态"):
        _validate_downloaded_summary(tmp_path, spec)


def test_downloaded_matrix_summary_must_confirm_cell(tmp_path: Path) -> None:
    spec = {
        "workflow": "capacity-matrix-train",
        "source_revision": "abc123",
        "matrix_cell": "1m-raw200",
    }
    summary_path = tmp_path / "colab-run-summary.json"
    summary_path.write_text(
        json.dumps({**spec, "status": "complete", "matrix_cell": "1m-raw1000"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="matrix_cell 不匹配"):
        _validate_downloaded_summary(tmp_path, spec)


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


@pytest.mark.parametrize(
    "workflow",
    [
        "capacity-benchmark",
        "batch-size-sweep",
        "eventstream-capacity-benchmark",
        "eventstream-recent-capacity-benchmark",
        "eventstream-recent-batch-size-sweep",
        "eventstream-recent-input-profile",
    ],
)
def test_capacity_workflows_stage_features_without_target_sidecar(
    workflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies: list[tuple[str, str]] = []
    monkeypatch.setattr(
        colab_job,
        "_rclone_copy",
        lambda source, destination, **_kwargs: copies.append((source, destination)),
    )

    colab_job._stage_inputs(
        {
            "workflow": workflow,
            "rclone_remote": "gdrive",
            "feature_remote": "project/raw1000",
            "feature_local": "/content/raw1000",
        },
        {},
    )

    assert copies == [("gdrive:project/raw1000", "/content/raw1000")]


@pytest.mark.parametrize("workflow", ["raw1000-train", "capacity-matrix-train"])
def test_raw_training_stages_features_and_resumable_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    copies: list[tuple[str, str]] = []
    monkeypatch.setattr(
        colab_job,
        "_rclone_copy",
        lambda source, destination, **_kwargs: copies.append((source, destination)),
    )
    monkeypatch.setattr(
        colab_job,
        "_remote_directory_exists",
        lambda _source, **_kwargs: True,
    )

    colab_job._stage_inputs(
        {
            "workflow": workflow,
            "rclone_remote": "gdrive",
            "feature_remote": "project/data/raw1000",
            "feature_local": "/content/raw1000",
            "checkpoint_remote": "project/runs/raw1000/training",
            "checkpoint_local": str(tmp_path / "checkpoints"),
        },
        {},
    )

    assert copies == [
        ("gdrive:project/data/raw1000", "/content/raw1000"),
        (
            "gdrive:project/runs/raw1000/training",
            str(tmp_path / "checkpoints"),
        ),
    ]


def test_eventstream_training_restores_materialized_data_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies: list[tuple[str, str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        colab_job,
        "_rclone_copy",
        lambda source, destination, **kwargs: copies.append(
            (source, destination, tuple(kwargs.get("exclude", ())))
        ),
    )
    monkeypatch.setattr(
        colab_job,
        "_remote_directory_exists",
        lambda _source, **_kwargs: True,
    )

    colab_job._stage_inputs(
        {
            "workflow": "eventstream-recent-train",
            "rclone_remote": "gdrive",
            "feature_remote": "project/data/materialized/seed0",
            "feature_local": "/content/materialized",
            "checkpoint_remote": "project/runs/eventstream/training",
            "checkpoint_local": str(tmp_path / "checkpoints"),
            "evaluate_test": False,
        },
        {},
    )

    assert copies == [
        (
            "gdrive:project/data/materialized/seed0",
            "/content/materialized",
            ("shards/oos-*/**", "shards/monitor_oos-*/**"),
        ),
        (
            "gdrive:project/runs/eventstream/training",
            str(tmp_path / "checkpoints"),
            (),
        ),
    ]


def test_eventstream_training_verifies_cache_and_preserves_oos_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )
    spec = {
        "workflow": "eventstream-recent-train",
        "feature_local": "/content/materialized",
        "output_local": str(tmp_path / "output"),
        "training_config": "/content/config.yaml",
        "seeds": [0],
        "source_revision": "abc1234",
        "expected_parameter_count": 100_604_180,
        "training_epochs": 1,
        "evaluate_test": False,
    }

    colab_job._execute_workflow(spec)

    assert any("ticknet.eventstream.materialized" in command for command in captured)
    preflight = next(
        command for command in captured if "ticknet.eventstream.materialized" in command
    )
    assert preflight.count("--partition") == 3
    train_command = next(command for command in captured if "ticknet.eventstream.train" in command)
    assert train_command[train_command.index("--seed") + 1] == "0"
    assert train_command[train_command.index("--epochs") + 1] == "1"
    assert train_command[train_command.index("--expected-parameter-count") + 1] == "100604180"
    assert "--no-evaluate-test" in train_command


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

    colab_job._train_nextday(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "seeds": [0, 2],
        }
    )

    assert [command[-1] for command in captured] == ["0", "2"]
    assert all("ticknet.nextday.train" in command for command in captured)


def test_capacity_benchmark_invokes_audited_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._benchmark_capacity(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "benchmark_batches": 100,
            "warmup_batches": 5,
            "expected_parameter_count": 100_817_575,
            "projected_train_samples": 75_000,
            "source_revision": "abc123",
            "requested_gpu": "T4",
        }
    )

    command = captured[0]
    assert "ticknet.nextday.benchmark" in command
    assert command[command.index("--expected-parameter-count") + 1] == "100817575"
    assert command[command.index("--projected-train-samples") + 1] == "75000"
    assert command[command.index("--requested-gpu") + 1] == "T4"


def test_eventstream_capacity_benchmark_invokes_eventstream_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._benchmark_eventstream(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "benchmark_batches": 100,
            "warmup_batches": 5,
            "expected_parameter_count": 100_604_180,
            "source_revision": "abc123",
            "requested_gpu": "A100",
        }
    )

    command = captured[0]
    assert "ticknet.eventstream.benchmark" in command
    assert command[command.index("--expected-parameter-count") + 1] == "100604180"
    assert "--projected-train-samples" not in command


def test_eventstream_batch_sweep_invokes_eventstream_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._sweep_eventstream_batch_sizes(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "batch_sizes": [8, 16, 32, 64],
            "effective_batch_size": 64,
            "benchmark_batches": 50,
            "warmup_batches": 5,
            "expected_parameter_count": 100_604_180,
            "projected_train_samples": 120_000,
            "source_revision": "abc123",
            "requested_gpu": "A100",
        }
    )

    command = captured[0]
    assert "ticknet.eventstream.benchmark_sweep" in command
    batch_index = command.index("--batch-sizes")
    assert command[batch_index + 1 : batch_index + 5] == ["8", "16", "32", "64"]
    assert command[command.index("--effective-batch-size") + 1] == "64"
    assert command[command.index("--projected-train-samples") + 1] == "120000"


def test_eventstream_input_profile_invokes_audited_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._profile_eventstream_input(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "num_workers": [2, 4, 8, 16],
            "effective_batch_size": 64,
            "benchmark_batches": 50,
            "warmup_batches": 5,
            "expected_parameter_count": 100_604_180,
            "projected_train_samples": 120_000,
            "source_revision": "abc123",
            "requested_gpu": "A100",
        }
    )

    command = captured[0]
    assert "ticknet.eventstream.input_profile" in command
    worker_index = command.index("--num-workers")
    assert command[worker_index + 1 : worker_index + 5] == ["2", "4", "8", "16"]
    assert command[command.index("--effective-batch-size") + 1] == "64"
    assert command[command.index("--projected-train-samples") + 1] == "120000"


def test_batch_size_sweep_invokes_audited_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        colab_job,
        "_run",
        lambda command, **_kwargs: captured.append(command),
    )

    colab_job._sweep_batch_sizes(
        {
            "output_local": str(tmp_path / "output"),
            "training_config": "/content/config.yaml",
            "batch_sizes": [2, 4, 8, 16, 32],
            "effective_batch_size": 32,
            "benchmark_batches": 50,
            "warmup_batches": 5,
            "expected_parameter_count": 100_817_575,
            "projected_train_samples": 75_000,
            "source_revision": "abc123",
            "requested_gpu": "A100",
        }
    )

    command = captured[0]
    assert "ticknet.nextday.benchmark_sweep" in command
    batch_index = command.index("--batch-sizes")
    assert command[batch_index + 1 : batch_index + 6] == ["2", "4", "8", "16", "32"]
    assert command[command.index("--effective-batch-size") + 1] == "32"
    assert command[command.index("--batches") + 1] == "50"


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
    monkeypatch.setattr(colab_runner, "_validate_downloaded_summary", lambda *_args: None)
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
