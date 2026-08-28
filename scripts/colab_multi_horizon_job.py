"""在已分配的 Colab VM 内执行无 Drive mount 的次日模型任务。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

SPEC_PATH = Path("/content/ticknet-colab-job.json")
EVENTSTREAM_BENCHMARK_WORKFLOWS = frozenset(
    {
        "eventstream-capacity-benchmark",
        "eventstream-recent-capacity-benchmark",
        "eventstream-recent-batch-size-sweep",
        "eventstream-recent-input-profile",
    }
)
EVENTSTREAM_LABEL_SCALE_WORKFLOWS = frozenset(
    {"eventstream-recent-label-scale-train", "eventstream-rolling-label-scale-train"}
)
EVENTSTREAM_TRAIN_WORKFLOWS = (
    frozenset({"eventstream-recent-train", "eventstream-rolling-train"})
    | EVENTSTREAM_LABEL_SCALE_WORKFLOWS
)
EVENTSTREAM_EMBEDDING_WORKFLOWS = frozenset({"eventstream-recent-export-embeddings"})
EVENTSTREAM_JOINT_WORKFLOWS = frozenset({"eventstream-recent-joint-finetune"})
EVENTSTREAM_PREDICTION_WORKFLOWS = frozenset({"eventstream-rolling-export-predictions"})
EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS = frozenset(
    {"eventstream-recent-gradient-audit", "eventstream-rolling-gradient-audit"}
)
EVENTSTREAM_CHECKPOINT_SYNC_SECONDS = 180.0


class _StopSignal(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _drive_path(remote: str, path: str) -> str:
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Drive 路径必须是安全的相对路径：{path}")
    return f"{remote}:{normalized.as_posix()}"


def _ensure_rclone() -> None:
    if shutil.which("rclone"):
        return
    _run(["apt-get", "update", "-qq"])
    _run(["apt-get", "install", "-y", "-qq", "rclone"])


def _rclone_copy(
    source: str,
    destination: str,
    *,
    env: dict[str, str],
    exclude: tuple[str, ...] = (),
) -> None:
    command = [
        "rclone",
        "copy",
        source,
        destination,
        "--checkers",
        "16",
        "--transfers",
        "8",
        "--fast-list",
        "--stats",
        "30s",
    ]
    for pattern in exclude:
        command.extend(("--exclude", pattern))
    _run(command, env=env)


def _remote_directory_exists(source: str, *, env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["rclone", "lsf", source, "--max-depth", "1"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        return True
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if "directory not found" in output.lower():
        return False
    raise subprocess.CalledProcessError(
        result.returncode,
        result.args,
        output=result.stdout,
        stderr=result.stderr,
    )


def _stage_inputs(spec: dict[str, Any], env: dict[str, str]) -> None:
    remote = str(spec["rclone_remote"])
    workflow = str(spec["workflow"])
    if workflow in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS:
        excluded_partitions = (
            "shards/train-*/**",
            "shards/oos-*/**",
            "shards/monitor_validation-*/**",
            "shards/monitor_oos-*/**",
        )
    elif workflow in EVENTSTREAM_PREDICTION_WORKFLOWS:
        excluded_partitions = (
            "shards/train-*/**",
            "shards/monitor_validation-*/**",
            "shards/monitor_oos-*/**",
        )
    elif workflow in EVENTSTREAM_TRAIN_WORKFLOWS and not spec["evaluate_test"]:
        excluded_partitions = ("shards/oos-*/**", "shards/monitor_oos-*/**")
    else:
        excluded_partitions = ()
    _rclone_copy(
        _drive_path(remote, str(spec["feature_remote"])),
        str(spec["feature_local"]),
        env=env,
        exclude=excluded_partitions,
    )
    if workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS:
        _rclone_copy(
            _drive_path(remote, str(spec["target_overlay_remote"])),
            str(spec["target_overlay_local"]),
            env=env,
        )
    if workflow in EVENTSTREAM_JOINT_WORKFLOWS:
        _rclone_copy(
            _drive_path(remote, str(spec["joint_cache_remote"])),
            str(spec["joint_cache_local"]),
            env=env,
        )
        checkpoint_root = Path(str(spec["checkpoint_local"]))
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        seed = int(spec["seeds"][0])
        checkpoint_name = f"{spec['checkpoint_name']}.seed{seed}.best.pt"
        _run(
            [
                "rclone",
                "copyto",
                _drive_path(remote, f"{spec['checkpoint_remote']}/{checkpoint_name}"),
                str(checkpoint_root / checkpoint_name),
            ],
            env=env,
        )
        output_remote = _drive_path(remote, str(spec["output_remote"]))
        if _remote_directory_exists(output_remote, env=env):
            _rclone_copy(output_remote, str(spec["output_local"]), env=env)
        return
    if workflow in EVENTSTREAM_EMBEDDING_WORKFLOWS:
        training_manifest_root = Path(str(spec["training_manifest_local"]))
        training_manifest_root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "rclone",
                "copyto",
                _drive_path(
                    remote,
                    f"{spec['training_manifest_remote']}/manifest.json",
                ),
                str(training_manifest_root / "manifest.json"),
            ],
            env=env,
        )
        checkpoint_root = Path(str(spec["checkpoint_local"]))
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        seed = int(spec["seeds"][0])
        checkpoint_name = f"{spec['checkpoint_name']}.seed{seed}.best.pt"
        _run(
            [
                "rclone",
                "copyto",
                _drive_path(
                    remote,
                    f"{spec['checkpoint_remote']}/{checkpoint_name}",
                ),
                str(checkpoint_root / checkpoint_name),
            ],
            env=env,
        )
        return
    if workflow in EVENTSTREAM_PREDICTION_WORKFLOWS | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS:
        checkpoint_root = Path(str(spec["checkpoint_local"]))
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        seed = int(spec["seeds"][0])
        checkpoint_name = f"{spec['checkpoint_name']}.seed{seed}.best.pt"
        _run(
            [
                "rclone",
                "copyto",
                _drive_path(remote, f"{spec['checkpoint_remote']}/{checkpoint_name}"),
                str(checkpoint_root / checkpoint_name),
            ],
            env=env,
        )
        return
    if workflow in {"multi-horizon-validation", "h5-train"}:
        _rclone_copy(
            _drive_path(remote, str(spec["target_remote"])),
            str(spec["target_local"]),
            env=env,
        )
    if (
        workflow
        in {
            "capacity-benchmark",
            "batch-size-sweep",
        }
        | EVENTSTREAM_BENCHMARK_WORKFLOWS
    ):
        return
    checkpoint_root = Path(str(spec["checkpoint_local"]))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if workflow == "multi-horizon-validation":
        for seed in spec["seeds"]:
            name = f"{spec['checkpoint_name']}.seed{int(seed)}.best.pt"
            _run(
                [
                    "rclone",
                    "copyto",
                    _drive_path(remote, f"{spec['checkpoint_remote']}/{name}"),
                    str(checkpoint_root / name),
                ],
                env=env,
            )
    elif workflow in (
        {"h5-train", "raw1000-train", "capacity-matrix-train"} | EVENTSTREAM_TRAIN_WORKFLOWS
    ):
        checkpoint_remote = _drive_path(remote, str(spec["checkpoint_remote"]))
        if _remote_directory_exists(checkpoint_remote, env=env):
            _rclone_copy(checkpoint_remote, str(checkpoint_root), env=env)
    else:
        raise ValueError(f"未知 workflow：{workflow}")


def _install_project(spec: dict[str, Any]) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pyarrow>=15",
            "pyyaml>=6",
            "polars>=1.0",
            "scikit-learn>=1.3",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(spec["wheel"]),
        ]
    )


def _evaluate(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "ticknet.nextday.horizon_cli",
        "--config",
        str(spec["training_config"]),
        "--sidecar",
        str(Path(str(spec["target_local"])) / "horizon-labels.json"),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *(str(seed) for seed in spec["seeds"]),
        "--horizons",
        *(str(horizon) for horizon in spec["horizons"]),
        "--inference-batch-size",
        str(spec["inference_batch_size"]),
        "--source-revision",
        str(spec["source_revision"]),
    ]
    _run(command)


def _train_nextday(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in spec["seeds"]:
        _run(
            [
                sys.executable,
                "-m",
                "ticknet.nextday.train",
                "--config",
                str(spec["training_config"]),
                "--seed",
                str(int(seed)),
            ]
        )


def _verify_eventstream_materialized(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ticknet.eventstream.materialized",
        "verify",
        "--root",
        str(spec["feature_local"]),
        "--output",
        str(output_dir / "materialized-preflight.json"),
    ]
    if not spec["evaluate_test"]:
        for partition in ("train", "validation", "monitor_validation"):
            command.extend(("--partition", partition))
    _run(command)


def _train_eventstream(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in spec["seeds"]:
        command = [
            sys.executable,
            "-m",
            "ticknet.eventstream.train",
            "--config",
            str(spec["training_config"]),
            "--seed",
            str(int(seed)),
            "--source-revision",
            str(spec.get("experiment_source_revision") or spec["source_revision"]),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--evaluate-test" if spec["evaluate_test"] else "--no-evaluate-test",
        ]
        if spec.get("training_epochs") is not None:
            command.extend(("--epochs", str(int(spec["training_epochs"]))))
        if spec.get("day_supervision_mode") is not None:
            command.extend(
                (
                    "--day-supervision-mode",
                    str(spec["day_supervision_mode"]),
                    "--checkpoint-dir",
                    str(spec["output_local"]),
                    "--checkpoint-name",
                    str(spec["checkpoint_name"]),
                )
            )
        if spec.get("day_loss_weight") is not None:
            command.extend(("--day-loss-weight", str(float(spec["day_loss_weight"]))))
        _run(command)


def _audit_eventstream_gradients(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec["seeds"][0])
    checkpoint = (
        Path(str(spec["checkpoint_local"])) / f"{spec['checkpoint_name']}.seed{seed}.best.pt"
    )
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.gradient_audit",
            "run",
            "--config",
            str(spec["training_config"]),
            "--materialized-root",
            str(spec["feature_local"]),
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            str(spec["expected_gradient_checkpoint_sha256"]),
            "--output",
            str(output_dir / "gradient-audit.json"),
            "--partition",
            "validation",
            "--batches",
            str(int(spec["audit_batches"])),
            "--batch-size",
            "8",
            "--device",
            "cuda",
            "--source-revision",
            str(spec["source_revision"]),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
        ]
    )


def _export_eventstream_embeddings(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec["seeds"][0])
    checkpoint = (
        Path(str(spec["checkpoint_local"])) / f"{spec['checkpoint_name']}.seed{seed}.best.pt"
    )
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.embedding",
            "--close-cache",
            str(spec["feature_local"]),
            "--checkpoint",
            str(checkpoint),
            "--training-manifest-root",
            str(spec["training_manifest_local"]),
            "--model",
            "capacity100m",
            "--output",
            str(output_dir),
            "--device",
            "cuda",
            "--batch-size",
            str(int(spec["embedding_batch_size"])),
            "--num-workers",
            str(min(max(int(value) for value in spec["num_workers"]), 4)),
            "--allow-oos",
            "--source-revision",
            str(spec["source_revision"]),
        ]
    )


def _export_eventstream_materialized_predictions(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    seed = int(spec["seeds"][0])
    checkpoint = (
        Path(str(spec["checkpoint_local"])) / f"{spec['checkpoint_name']}.seed{seed}.best.pt"
    )
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.materialized_predictions",
            "score",
            "--checkpoint",
            str(checkpoint),
            "--materialized-root",
            str(spec["feature_local"]),
            "--model",
            "capacity100m",
            "--output",
            str(output_dir),
            "--device",
            "cuda",
            "--batch-size",
            str(int(spec["embedding_batch_size"])),
            "--num-workers",
            str(min(max(int(value) for value in spec["num_workers"]), 4)),
            "--partition",
            "validation",
            "--partition",
            "oos",
            "--allow-oos",
            "--source-revision",
            str(spec["source_revision"]),
        ]
    )


def _train_eventstream_joint(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec["seeds"][0])
    checkpoint = (
        Path(str(spec["checkpoint_local"])) / f"{spec['checkpoint_name']}.seed{seed}.best.pt"
    )
    command = [
        sys.executable,
        "-m",
        "ticknet.eventstream.joint",
        "--config",
        str(spec["training_config"]),
        "--cache",
        str(spec["joint_cache_local"]),
        "--close-cache",
        str(spec["feature_local"]),
        "--pretrained-checkpoint",
        str(checkpoint),
        "--expected-pretrained-sha256",
        str(spec["expected_pretrained_sha256"]),
        "--seed",
        str(seed),
        "--output",
        str(output_dir),
        "--source-revision",
        str(spec["source_revision"]),
        "--allow-oos",
    ]
    if spec.get("training_epochs") is not None:
        command.extend(("--epochs", str(int(spec["training_epochs"]))))
    _run(command)


def _benchmark_capacity(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.nextday.benchmark",
            "--config",
            str(spec["training_config"]),
            "--output",
            str(output_dir / "capacity-benchmark.json"),
            "--batches",
            str(int(spec["benchmark_batches"])),
            "--warmup-batches",
            str(int(spec["warmup_batches"])),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--projected-train-samples",
            str(int(spec["projected_train_samples"])),
            "--source-revision",
            str(spec["source_revision"]),
            "--requested-gpu",
            str(spec["requested_gpu"]),
        ]
    )


def _benchmark_eventstream(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.benchmark",
            "--config",
            str(spec["training_config"]),
            "--output",
            str(output_dir / "capacity-benchmark.json"),
            "--batches",
            str(int(spec["benchmark_batches"])),
            "--warmup-batches",
            str(int(spec["warmup_batches"])),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--source-revision",
            str(spec["source_revision"]),
            "--requested-gpu",
            str(spec["requested_gpu"]),
        ]
    )


def _sweep_batch_sizes(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.nextday.benchmark_sweep",
            "--config",
            str(spec["training_config"]),
            "--output-dir",
            str(output_dir),
            "--batch-sizes",
            *(str(int(batch_size)) for batch_size in spec["batch_sizes"]),
            "--effective-batch-size",
            str(int(spec["effective_batch_size"])),
            "--batches",
            str(int(spec["benchmark_batches"])),
            "--warmup-batches",
            str(int(spec["warmup_batches"])),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--projected-train-samples",
            str(int(spec["projected_train_samples"])),
            "--source-revision",
            str(spec["source_revision"]),
            "--requested-gpu",
            str(spec["requested_gpu"]),
        ]
    )


def _sweep_eventstream_batch_sizes(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.benchmark_sweep",
            "--config",
            str(spec["training_config"]),
            "--output-dir",
            str(output_dir),
            "--batch-sizes",
            *(str(int(batch_size)) for batch_size in spec["batch_sizes"]),
            "--effective-batch-size",
            str(int(spec["effective_batch_size"])),
            "--batches",
            str(int(spec["benchmark_batches"])),
            "--warmup-batches",
            str(int(spec["warmup_batches"])),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--projected-train-samples",
            str(int(spec["projected_train_samples"])),
            "--source-revision",
            str(spec["source_revision"]),
            "--requested-gpu",
            str(spec["requested_gpu"]),
        ]
    )


def _profile_eventstream_input(spec: dict[str, Any]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "ticknet.eventstream.input_profile",
            "--config",
            str(spec["training_config"]),
            "--output-dir",
            str(output_dir),
            "--num-workers",
            *(str(int(workers)) for workers in spec["num_workers"]),
            "--effective-batch-size",
            str(int(spec["effective_batch_size"])),
            "--batches",
            str(int(spec["benchmark_batches"])),
            "--warmup-batches",
            str(int(spec["warmup_batches"])),
            "--expected-parameter-count",
            str(int(spec["expected_parameter_count"])),
            "--projected-train-samples",
            str(int(spec["projected_train_samples"])),
            "--source-revision",
            str(spec["source_revision"]),
            "--requested-gpu",
            str(spec["requested_gpu"]),
        ]
    )


def _write_summary(
    spec: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
) -> None:
    output_dir = Path(str(spec["output_local"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status,
        "workflow": spec["workflow"],
        "source_revision": spec["source_revision"],
        "experiment_source_revision": (
            spec.get("experiment_source_revision") or spec["source_revision"]
        ),
        "seeds": spec["seeds"],
        "output_remote": spec["output_remote"],
        "test_status": "locked_not_accessed",
        "oos_status": (
            "evaluated"
            if (
                spec["workflow"] in EVENTSTREAM_EMBEDDING_WORKFLOWS
                or spec["workflow"] in EVENTSTREAM_JOINT_WORKFLOWS
                or spec["workflow"] in EVENTSTREAM_PREDICTION_WORKFLOWS
                or (spec["workflow"] in EVENTSTREAM_TRAIN_WORKFLOWS and spec["evaluate_test"])
            )
            else "not_evaluated"
        ),
    }
    if spec.get("matrix_cell") is not None:
        summary["matrix_cell"] = spec["matrix_cell"]
    if spec.get("eventstream_fold_id") is not None:
        summary["eventstream_fold_id"] = spec["eventstream_fold_id"]
    if spec.get("day_supervision_mode") is not None:
        summary["day_supervision_mode"] = spec["day_supervision_mode"]
    if spec.get("day_loss_weight") is not None:
        summary["day_loss_weight"] = spec["day_loss_weight"]
    if error is not None:
        summary["error"] = error
    (output_dir / "colab-run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _upload_output(spec: dict[str, Any], env: dict[str, str]) -> None:
    _rclone_copy(
        str(spec["output_local"]),
        _drive_path(str(spec["rclone_remote"]), str(spec["output_remote"])),
        env=env,
    )


def _sync_eventstream_checkpoints_once(spec: dict[str, Any], env: dict[str, str]) -> None:
    output_dir = Path(str(spec["output_local"]))
    if not output_dir.is_dir():
        return
    _rclone_copy(
        str(output_dir),
        _drive_path(str(spec["rclone_remote"]), str(spec["output_remote"])),
        env=env,
        exclude=("*.tmp",),
    )
    print("事件流 checkpoint 已同步到远端。", flush=True)


def _checkpoint_sync_loop(
    spec: dict[str, Any],
    env: dict[str, str],
    stop: _StopSignal,
    interval_seconds: float,
) -> None:
    while not stop.wait(interval_seconds):
        try:
            _sync_eventstream_checkpoints_once(spec, env)
        except Exception as error:
            print(f"事件流 checkpoint 周期同步失败，将继续重试：{error}", file=sys.stderr)


@contextmanager
def _periodic_eventstream_checkpoint_sync(
    spec: dict[str, Any], env: dict[str, str]
) -> Iterator[None]:
    if spec["workflow"] not in EVENTSTREAM_TRAIN_WORKFLOWS:
        yield
        return
    stop = threading.Event()
    worker = threading.Thread(
        target=_checkpoint_sync_loop,
        args=(spec, env, stop, EVENTSTREAM_CHECKPOINT_SYNC_SECONDS),
        name="eventstream-checkpoint-sync",
        daemon=True,
    )
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def _execute_workflow(spec: dict[str, Any]) -> None:
    if spec["workflow"] == "multi-horizon-validation":
        _evaluate(spec)
    elif spec["workflow"] in {"h5-train", "raw1000-train", "capacity-matrix-train"}:
        _train_nextday(spec)
    elif spec["workflow"] == "capacity-benchmark":
        _benchmark_capacity(spec)
    elif spec["workflow"] == "batch-size-sweep":
        _sweep_batch_sizes(spec)
    elif spec["workflow"] == "eventstream-recent-batch-size-sweep":
        _sweep_eventstream_batch_sizes(spec)
    elif spec["workflow"] == "eventstream-recent-input-profile":
        _profile_eventstream_input(spec)
    elif spec["workflow"] in EVENTSTREAM_BENCHMARK_WORKFLOWS:
        _benchmark_eventstream(spec)
    elif spec["workflow"] in EVENTSTREAM_TRAIN_WORKFLOWS:
        _verify_eventstream_materialized(spec)
        _train_eventstream(spec)
    elif spec["workflow"] in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS:
        _audit_eventstream_gradients(spec)
    elif spec["workflow"] in EVENTSTREAM_EMBEDDING_WORKFLOWS:
        _export_eventstream_embeddings(spec)
    elif spec["workflow"] in EVENTSTREAM_PREDICTION_WORKFLOWS:
        from ticknet.eventstream.materialized import verify_materialized_dataset

        verify_materialized_dataset(
            Path(str(spec["feature_local"])),
            partitions=("validation", "oos"),
        )
        _export_eventstream_materialized_predictions(spec)
    elif spec["workflow"] in EVENTSTREAM_JOINT_WORKFLOWS:
        _train_eventstream_joint(spec)
    else:
        raise ValueError(f"未知 workflow：{spec['workflow']}")


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rclone_config = Path(str(spec["rclone_config"]))
    if not rclone_config.is_file():
        raise FileNotFoundError(f"缺少临时 rclone 配置：{rclone_config}")
    rclone_config.chmod(0o600)
    env = {**os.environ, "RCLONE_CONFIG": str(rclone_config)}
    try:
        try:
            _ensure_rclone()
            _install_project(spec)
            _stage_inputs(spec, env)
            with _periodic_eventstream_checkpoint_sync(spec, env):
                _execute_workflow(spec)
        except Exception as error:
            if spec["workflow"] in {
                "h5-train",
                "raw1000-train",
                "capacity-matrix-train",
                "capacity-benchmark",
                "batch-size-sweep",
                "eventstream-capacity-benchmark",
                "eventstream-recent-capacity-benchmark",
                "eventstream-recent-batch-size-sweep",
                "eventstream-recent-input-profile",
                "eventstream-recent-train",
                "eventstream-rolling-train",
                "eventstream-recent-export-embeddings",
                "eventstream-recent-joint-finetune",
                "eventstream-rolling-export-predictions",
                "eventstream-recent-gradient-audit",
                "eventstream-rolling-gradient-audit",
                "eventstream-recent-label-scale-train",
                "eventstream-rolling-label-scale-train",
            }:
                _write_summary(spec, status="failed", error=str(error))
                try:
                    _upload_output(spec, env)
                except Exception as upload_error:
                    print(f"失败产物同步失败：{upload_error}", file=sys.stderr)
            raise
        _write_summary(spec, status="complete")
        _upload_output(spec, env)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "workflow": spec["workflow"],
                    "source_revision": spec["source_revision"],
                    "output_remote": spec["output_remote"],
                    "test_status": "locked_not_accessed",
                    "oos_status": (
                        "evaluated"
                        if (
                            spec["workflow"] in EVENTSTREAM_EMBEDDING_WORKFLOWS
                            or spec["workflow"] in EVENTSTREAM_JOINT_WORKFLOWS
                            or spec["workflow"] in EVENTSTREAM_PREDICTION_WORKFLOWS
                            or (
                                spec["workflow"] in EVENTSTREAM_TRAIN_WORKFLOWS
                                and spec["evaluate_test"]
                            )
                        )
                        else "not_evaluated"
                    ),
                },
                ensure_ascii=False,
            )
        )
    finally:
        rclone_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
