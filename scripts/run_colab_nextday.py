"""从 Linux 开发机无人值守调度 Colab 次日模型任务。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "nextday-raw-200-capacity-1m.yaml"
DEFAULT_H5_CONFIG = REPOSITORY_ROOT / "configs" / "nextday-raw-200-capacity-1m-h5.yaml"
DEFAULT_100M_BENCHMARK_CONFIG = (
    REPOSITORY_ROOT / "configs" / "nextday-raw-1000-top100-capacity-100m-benchmark.yaml"
)
DEFAULT_100M_TRAIN_CONFIG = (
    REPOSITORY_ROOT / "configs" / "nextday-raw-1000-top100-capacity-100m.yaml"
)
CAPACITY_MATRIX_CONFIGS = {
    cell: REPOSITORY_ROOT / "configs" / f"nextday-capacity-matrix-{cell}.yaml"
    for cell in ("1m-raw200", "1m-raw1000", "100m-raw200")
}
CAPACITY_MATRIX_CHECKPOINTS = {
    "1m-raw200": "raw-200-top100-dual-head-capacity_1m-matrix",
    "1m-raw1000": "raw-1000-top100-dual-head-capacity_1m-matrix",
    "100m-raw200": "raw-200-top100-dual-head-capacity_100m-matrix",
}
DEFAULT_EVENTSTREAM_BENCHMARK_CONFIG = (
    REPOSITORY_ROOT / "configs" / "eventstream-h5-fold0-capacity100m-colab.yaml"
)
DEFAULT_EVENTSTREAM_RECENT_BENCHMARK_CONFIG = (
    REPOSITORY_ROOT / "configs" / "eventstream-h5-recent-capacity100m-colab.yaml"
)
DEFAULT_EVENTSTREAM_RECENT_TRAIN_CONFIG = (
    REPOSITORY_ROOT / "configs" / "eventstream-h5-recent-capacity100m-materialized-colab.yaml"
)
DEFAULT_EVENTSTREAM_JOINT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "eventstream-joint-recent-capacity100m.yaml"
)
EVENTSTREAM_CHECKPOINT_SHA256_BY_SEED = {
    0: "8632e62bdf4f27383e299c3ff676876d8a1969f6d69ec66a7ce43da24f5255e9",
    1: "edc423d89bbd2a681383d04ec1c3ae22961b2c944c449f0572f5416f31de19ed",
    2: "013e2bd1281830100bbf15f673bd8b1cb8ff08951ea48eae9f81922b1eebd4f6",
}
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
DAY_SUPERVISION_MODES = ("all", "last", "tail_weighted")
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
EVENTSTREAM_GRADIENT_AUDIT_CHECKPOINTS = {
    ("recent", 0): EVENTSTREAM_CHECKPOINT_SHA256_BY_SEED[0],
    ("fold-54-oos-202511", 0): ("d753041016d71e668a46624585f7bc7fb68f67200fee3c5b463c28b26f1c11cd"),
}
EVENTSTREAM_FOLD_PATTERN = re.compile(r"^fold-\d{2}-oos-\d{6}$")
JOB_SCRIPT = REPOSITORY_ROOT / "scripts" / "colab_multi_horizon_job.py"
REMOTE_WHEEL = "/content/<wheel-filename>"
REMOTE_CONFIG = "/content/ticknet-nextday-config.yaml"
REMOTE_RCLONE_CONFIG = "/content/ticknet-rclone.conf"
REMOTE_SPEC = "/content/ticknet-colab-job.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用官方 colab CLI 和 rclone 运行无人值守次日模型任务",
    )
    parser.add_argument(
        "--workflow",
        choices=(
            "multi-horizon-validation",
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
            "eventstream-rolling-export-predictions",
            "eventstream-recent-export-embeddings",
            "eventstream-recent-joint-finetune",
            "eventstream-recent-gradient-audit",
            "eventstream-rolling-gradient-audit",
            "eventstream-recent-label-scale-train",
            "eventstream-rolling-label-scale-train",
        ),
        default="multi-horizon-validation",
    )
    parser.add_argument("--session", default="ticknet-multi-horizon")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="T4")
    parser.add_argument(
        "--matrix-cell",
        choices=tuple(CAPACITY_MATRIX_CONFIGS),
        default="1m-raw200",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--rclone-config",
        type=Path,
        default=Path("~/.config/rclone/rclone.conf").expanduser(),
    )
    parser.add_argument("--rclone-remote", default="gdrive")
    parser.add_argument("--drive-root", default="deep-learning-tick-data-prediction")
    parser.add_argument(
        "--eventstream-fold-id",
        help="滚动事件流训练的折标识，例如 fold-54-oos-202511",
    )
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--benchmark-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--num-workers", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--training-epochs", type=int)
    parser.add_argument(
        "--experiment-source-revision",
        help="恢复旧 checkpoint 时使用的原实验代码版本，默认使用当前提交",
    )
    parser.add_argument("--audit-batches", type=int, default=16)
    parser.add_argument(
        "--day-supervision-mode",
        choices=DAY_SUPERVISION_MODES,
        default="all",
        help="日级标签的监督位置，仅用于标签尺度训练",
    )
    parser.add_argument(
        "--day-loss-weight",
        type=float,
        help="日级任务损失权重，仅用于标签尺度训练",
    )
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=14_400.0)
    retention = parser.add_mutually_exclusive_group()
    retention.add_argument(
        "--keep-session",
        action="store_true",
        help="无论成功失败都保留本次新建的 session",
    )
    retention.add_argument(
        "--keep-on-failure",
        action="store_true",
        help="成功后关闭本次新建的 session，失败时保留现场",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="只复用同名现有 session，runner 不负责关闭它",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _require_executable(name: str, *, home: Path | None = None) -> str:
    executable = shutil.which(name)
    if executable is None:
        user_executable = (home or Path.home()) / ".local" / "bin" / name
        if user_executable.is_file() and os.access(user_executable, os.X_OK):
            return str(user_executable)
    if executable is None:
        raise FileNotFoundError(f"找不到命令：{name}")
    return executable


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def _colab_command(colab: str, *arguments: str) -> list[str]:
    return [colab, "--auth=oauth2", *arguments]


def _remote_wheel_path(wheel: Path) -> str:
    return f"/content/{wheel.name}"


def _validate_lifecycle_arguments(arguments: argparse.Namespace) -> None:
    if arguments.keep_session and arguments.keep_on_failure:
        raise ValueError("--keep-session 与 --keep-on-failure 不能同时使用")
    if arguments.benchmark_batches < 1 or arguments.warmup_batches < 0:
        raise ValueError("--benchmark-batches 应为正整数，--warmup-batches 不能为负数")
    if arguments.effective_batch_size < 1:
        raise ValueError("--effective-batch-size 应为正整数")
    if arguments.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size 应为正整数")
    if not arguments.batch_sizes or len(set(arguments.batch_sizes)) != len(arguments.batch_sizes):
        raise ValueError("--batch-sizes 不能为空且不能重复")
    if any(
        batch_size < 1 or arguments.effective_batch_size % batch_size
        for batch_size in arguments.batch_sizes
    ):
        raise ValueError("--batch-sizes 必须为能整除 effective batch 的正整数")
    if not arguments.num_workers or len(set(arguments.num_workers)) != len(arguments.num_workers):
        raise ValueError("--num-workers 不能为空且不能重复")
    if any(workers < 0 for workers in arguments.num_workers):
        raise ValueError("--num-workers 不能为负数")
    if arguments.training_epochs is not None and arguments.training_epochs < 1:
        raise ValueError("--training-epochs 应为正整数")
    if arguments.experiment_source_revision is not None:
        revision = arguments.experiment_source_revision
        if arguments.workflow not in EVENTSTREAM_TRAIN_WORKFLOWS:
            raise ValueError("--experiment-source-revision 只适用于事件流训练")
        if not revision.isascii() or len(revision) < 7:
            raise ValueError("--experiment-source-revision 应为至少 7 位 ASCII 标识")
    if arguments.audit_batches < 8 or arguments.audit_batches > 16:
        raise ValueError("--audit-batches 应在 8 至 16 之间")
    _validate_eventstream_arguments(arguments)


def _validate_eventstream_fold_id(value: str | None) -> str:
    if value is None or EVENTSTREAM_FOLD_PATTERN.fullmatch(value) is None:
        raise ValueError("--eventstream-fold-id 应使用 fold-NN-oos-YYYYMM 格式")
    return value


def _gradient_audit_checkpoint_sha256(arguments: argparse.Namespace) -> str:
    fold_id = (
        _validate_eventstream_fold_id(arguments.eventstream_fold_id)
        if arguments.workflow == "eventstream-rolling-gradient-audit"
        else "recent"
    )
    key = (fold_id, int(arguments.seeds[0]))
    if key not in EVENTSTREAM_GRADIENT_AUDIT_CHECKPOINTS:
        supported = sorted(EVENTSTREAM_GRADIENT_AUDIT_CHECKPOINTS)
        raise ValueError(f"梯度审计 checkpoint 身份未登记：{key}，可用 {supported}")
    return EVENTSTREAM_GRADIENT_AUDIT_CHECKPOINTS[key]


def _validate_eventstream_arguments(arguments: argparse.Namespace) -> None:
    single_seed_workflows = (
        EVENTSTREAM_TRAIN_WORKFLOWS
        | EVENTSTREAM_EMBEDDING_WORKFLOWS
        | EVENTSTREAM_JOINT_WORKFLOWS
        | EVENTSTREAM_PREDICTION_WORKFLOWS
        | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
    )
    if arguments.workflow in single_seed_workflows and len(arguments.seeds) != 1:
        raise ValueError("事件流正式任务每次只接受一个 seed")
    oos_workflows = (
        EVENTSTREAM_EMBEDDING_WORKFLOWS
        | EVENTSTREAM_JOINT_WORKFLOWS
        | EVENTSTREAM_PREDICTION_WORKFLOWS
    )
    if arguments.workflow in oos_workflows and not arguments.evaluate_test:
        raise ValueError("完整 embedding、联合微调和预测导出必须显式保留 OOS 读取授权")
    if (
        arguments.workflow in EVENTSTREAM_JOINT_WORKFLOWS
        and arguments.seeds[0] not in EVENTSTREAM_CHECKPOINT_SHA256_BY_SEED
    ):
        supported = sorted(EVENTSTREAM_CHECKPOINT_SHA256_BY_SEED)
        raise ValueError(f"联合微调 seed 应为 {supported} 之一")
    if arguments.workflow in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS:
        if arguments.evaluate_test:
            raise ValueError("梯度审计只读取 validation，请使用 --no-evaluate-test")
        _gradient_audit_checkpoint_sha256(arguments)
    if arguments.workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS and arguments.seeds != [0]:
        raise ValueError("标签尺度第一轮只允许运行 seed 0")
    day_loss_weight = getattr(arguments, "day_loss_weight", None)
    if day_loss_weight is not None and day_loss_weight < 0:
        raise ValueError("--day-loss-weight 不能为负数")
    if (
        day_loss_weight is not None
        and arguments.workflow not in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
    ):
        raise ValueError("--day-loss-weight 只用于标签尺度训练")
    if (
        arguments.day_supervision_mode != "all"
        and arguments.workflow not in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
    ):
        raise ValueError("--day-supervision-mode 只用于标签尺度训练")
    rolling_workflows = {
        "eventstream-rolling-train",
        "eventstream-rolling-export-predictions",
        "eventstream-rolling-gradient-audit",
        "eventstream-rolling-label-scale-train",
    }
    if arguments.workflow in rolling_workflows:
        _validate_eventstream_fold_id(arguments.eventstream_fold_id)
    elif arguments.eventstream_fold_id is not None:
        raise ValueError("--eventstream-fold-id 只用于事件流滚动训练、预测导出或梯度审计")


def _default_config(
    workflow: str,
    matrix_cell: str = "1m-raw200",
    eventstream_fold_id: str | None = None,
) -> Path:
    if workflow == "capacity-matrix-train":
        return CAPACITY_MATRIX_CONFIGS[matrix_cell]
    if workflow in {
        "eventstream-rolling-train",
        "eventstream-rolling-export-predictions",
        "eventstream-rolling-gradient-audit",
        "eventstream-rolling-label-scale-train",
    }:
        fold_id = _validate_eventstream_fold_id(eventstream_fold_id)
        suffix = (
            "capacity100m-label-z-materialized-colab.yaml"
            if workflow == "eventstream-rolling-label-scale-train"
            else "capacity100m-materialized-colab.yaml"
        )
        return REPOSITORY_ROOT / "configs" / f"eventstream-h5-{fold_id}-{suffix}"
    return {
        "multi-horizon-validation": DEFAULT_CONFIG,
        "h5-train": DEFAULT_H5_CONFIG,
        "raw1000-train": DEFAULT_100M_TRAIN_CONFIG,
        "capacity-benchmark": DEFAULT_100M_BENCHMARK_CONFIG,
        "batch-size-sweep": DEFAULT_100M_BENCHMARK_CONFIG,
        "eventstream-capacity-benchmark": DEFAULT_EVENTSTREAM_BENCHMARK_CONFIG,
        "eventstream-recent-capacity-benchmark": (DEFAULT_EVENTSTREAM_RECENT_BENCHMARK_CONFIG),
        "eventstream-recent-batch-size-sweep": (DEFAULT_EVENTSTREAM_RECENT_BENCHMARK_CONFIG),
        "eventstream-recent-input-profile": (DEFAULT_EVENTSTREAM_RECENT_BENCHMARK_CONFIG),
        "eventstream-recent-train": DEFAULT_EVENTSTREAM_RECENT_TRAIN_CONFIG,
        "eventstream-recent-export-embeddings": DEFAULT_EVENTSTREAM_RECENT_TRAIN_CONFIG,
        "eventstream-recent-joint-finetune": DEFAULT_EVENTSTREAM_JOINT_CONFIG,
        "eventstream-recent-gradient-audit": DEFAULT_EVENTSTREAM_RECENT_TRAIN_CONFIG,
        "eventstream-recent-label-scale-train": (
            REPOSITORY_ROOT
            / "configs"
            / "eventstream-h5-recent-capacity100m-label-z-materialized-colab.yaml"
        ),
    }[workflow]


def _session_exists(colab: str, session: str) -> bool:
    result = _run(
        _colab_command(colab, "status", "-s", session),
        check=False,
        capture_output=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if f"Session '{session}' not found." in output:
        return False
    if f"[{session}] " in output:
        return True
    raise RuntimeError(f"无法识别 colab status 输出：{output.strip()}")


def _validate_session_selection(
    *,
    session: str,
    exists: bool,
    reuse_session: bool,
) -> None:
    if exists and not reuse_session:
        raise RuntimeError(
            f"Colab session '{session}' 已存在；如需复用，请显式传入 --reuse-session"
        )
    if not exists and reuse_session:
        raise RuntimeError(f"Colab session '{session}' 不存在，无法使用 --reuse-session")


def _should_stop_owned_session(
    arguments: argparse.Namespace,
    *,
    succeeded: bool,
) -> bool:
    if arguments.keep_session:
        return False
    return not (arguments.keep_on_failure and not succeeded)


def _ensure_secret_outside_repository(secret: Path, repository_root: Path) -> None:
    resolved_secret = secret.expanduser().resolve()
    resolved_repository = repository_root.resolve()
    try:
        resolved_secret.relative_to(resolved_repository)
    except ValueError:
        return
    raise ValueError("rclone 配置包含刷新凭据，不能放在 Git 仓库内")


def _source_revision(repository_root: Path) -> str:
    result = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
    )
    return result.stdout.strip()


def _require_clean_revision(repository_root: Path) -> None:
    result = _run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Colab job 只接受已提交的干净 revision")


def _build_committed_wheel(
    uv: str,
    repository_root: Path,
    staging: Path,
    revision: str,
) -> Path:
    source_archive = staging / "source.tar"
    source_dir = staging / "source"
    source_dir.mkdir()
    _run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output",
            str(source_archive),
            revision,
        ],
        cwd=repository_root,
    )
    shutil.unpack_archive(str(source_archive), str(source_dir), "tar")
    wheel_dir = staging / "dist"
    _run([uv, "build", "--wheel", "--out-dir", str(wheel_dir)], cwd=source_dir)
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"预期生成一个 wheel，实际为 {wheels}")
    return wheels[0]


def _eventstream_training_paths(
    arguments: argparse.Namespace,
    drive_root: str,
) -> tuple[str, str, str, str]:
    def add_day_weight_suffix(name: str) -> str:
        if getattr(arguments, "day_loss_weight", None) is None:
            return name
        weight = f"{arguments.day_loss_weight:g}".replace("-", "m").replace(".", "p")
        return f"{name}-day-weight-{weight}"

    seed = int(arguments.seeds[0])
    if arguments.workflow in {
        "eventstream-rolling-train",
        "eventstream-rolling-export-predictions",
        "eventstream-rolling-gradient-audit",
        "eventstream-rolling-label-scale-train",
    }:
        fold_id = _validate_eventstream_fold_id(arguments.eventstream_fold_id)
        run_name = f"eventstream-top400-h5-capacity100m-{fold_id}"
        feature_remote = (
            f"{drive_root}/ticknet-data/eventstream-top400-h5-rolling/"
            f"{fold_id}/materialized/seed{seed}"
        )
        feature_local = f"/content/ticknet-eventstream/materialized/{fold_id}"
        if arguments.workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS:
            run_name = f"{run_name}-label-z"
            if arguments.day_supervision_mode != "all":
                mode = arguments.day_supervision_mode.replace("_", "-")
                run_name = f"{run_name}-day-{mode}"
            run_name = add_day_weight_suffix(run_name)
        return run_name, run_name, feature_remote, feature_local
    run_name = "eventstream-top400-h5-capacity100m-recent"
    feature_remote = (
        f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-materialized/seed{seed}"
    )
    feature_local = "/content/ticknet-eventstream/materialized/recent"
    if arguments.workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS:
        run_name = f"{run_name}-label-z"
        if arguments.day_supervision_mode != "all":
            mode = arguments.day_supervision_mode.replace("_", "-")
            run_name = f"{run_name}-day-{mode}"
        run_name = add_day_weight_suffix(run_name)
    return run_name, run_name, feature_remote, feature_local


def _eventstream_target_overlay_paths(
    arguments: argparse.Namespace,
    drive_root: str,
) -> tuple[str, str]:
    if arguments.workflow not in EVENTSTREAM_LABEL_SCALE_WORKFLOWS:
        raise ValueError("当前工作流不使用日级标签覆盖层")
    fold = (
        _validate_eventstream_fold_id(arguments.eventstream_fold_id)
        if arguments.workflow == "eventstream-rolling-label-scale-train"
        else "recent"
    )
    seed = int(arguments.seeds[0])
    return (
        f"{drive_root}/ticknet-data/eventstream-top400-h5-target-overlays/{fold}/seed{seed}",
        f"/content/ticknet-eventstream/target-overlay/{fold}/seed{seed}",
    )


def _eventstream_job_paths(
    arguments: argparse.Namespace,
    drive_root: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    run_name, checkpoint_name, feature_remote, feature_local = _eventstream_training_paths(
        arguments, drive_root
    )
    if arguments.workflow in EVENTSTREAM_PREDICTION_WORKFLOWS:
        seed = int(arguments.seeds[0])
        checkpoint_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        checkpoint_local = "/content/ticknet-eventstream/checkpoint"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/predictions/seed{seed}"
        output_local = f"/content/ticknet-results/{run_name}/predictions/seed{seed}"
    elif arguments.workflow in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS:
        seed = int(arguments.seeds[0])
        checkpoint_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        checkpoint_local = "/content/ticknet-eventstream/checkpoint"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/gradient-audit/seed{seed}"
        output_local = f"/content/ticknet-results/{run_name}/gradient-audit/seed{seed}"
    else:
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        output_local = f"/content/ticknet-results/{run_name}/training"
        checkpoint_remote = output_remote
        checkpoint_local = output_local
    return (
        run_name,
        checkpoint_name,
        feature_remote,
        feature_local,
        checkpoint_remote,
        checkpoint_local,
        output_remote,
        output_local,
    )


def build_job_spec(arguments: argparse.Namespace, source_revision: str) -> dict[str, Any]:
    drive_root = arguments.drive_root.strip("/")
    workflow = arguments.workflow
    if workflow == "multi-horizon-validation":
        run_name = "raw-200-capacity_1m"
        checkpoint_name = "raw-200-dual-head-capacity_1m"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/multi-horizon-validation-2024"
        output_local = "/content/ticknet-results/multi-horizon-validation-2024"
        feature_remote = f"{drive_root}/ticknet-data/nextday-raw-200"
        feature_local = "/content/nextday-raw-200"
    elif workflow == "h5-train":
        run_name = "raw-200-capacity_1m-h5"
        checkpoint_name = "raw-200-dual-head-capacity_1m-h5"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}"
        output_local = f"/content/drive/MyDrive/{output_remote}"
        feature_remote = f"{drive_root}/ticknet-data/nextday-raw-200"
        feature_local = "/content/nextday-raw-200"
    elif workflow == "raw1000-train":
        run_name = "raw-1000-top100-capacity_100m"
        checkpoint_name = "raw-1000-top100-dual-head-capacity_100m"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        output_local = f"/content/drive/MyDrive/{output_remote}"
        feature_remote = f"{drive_root}/ticknet-data/nextday-raw-1000-pilot-2021-2025-top100"
        feature_local = "/content/nextday-raw-1000-pilot-2021-2025-top100"
    elif workflow == "capacity-matrix-train":
        matrix_cell = arguments.matrix_cell
        run_name = f"raw-1000-top100-capacity-matrix/{matrix_cell}"
        checkpoint_name = CAPACITY_MATRIX_CHECKPOINTS[matrix_cell]
        output_remote = f"{drive_root}/ticknet-runs/{run_name}"
        output_local = f"/content/drive/MyDrive/{output_remote}"
        feature_remote = f"{drive_root}/ticknet-data/nextday-raw-1000-pilot-2021-2025-top100"
        feature_local = "/content/nextday-raw-1000-pilot-2021-2025-top100"
    elif workflow in {"capacity-benchmark", "batch-size-sweep"}:
        run_name = "raw-1000-top100-capacity_100m"
        checkpoint_name = "raw-1000-top100-dual-head-capacity_100m"
        gpu_label = arguments.gpu.lower()
        if workflow == "capacity-benchmark":
            output_remote = f"{drive_root}/ticknet-runs/{run_name}/benchmarks/{gpu_label}"
            output_local = f"/content/ticknet-results/{run_name}/{gpu_label}"
        else:
            output_remote = f"{drive_root}/ticknet-runs/{run_name}/batch-size-sweep/{gpu_label}"
            output_local = f"/content/ticknet-results/{run_name}/batch-size-sweep/{gpu_label}"
        feature_remote = f"{drive_root}/ticknet-data/nextday-raw-1000-preflight-202101-top100"
        feature_local = "/content/nextday-raw-1000-preflight-202101-top100"
    elif workflow in EVENTSTREAM_BENCHMARK_WORKFLOWS:
        if workflow == "eventstream-capacity-benchmark":
            run_name = "eventstream-top400-h5-capacity100m-fold0"
            checkpoint_name = "eventstream-top400-h5-capacity100m-fold0"
            feature_remote = (
                f"{drive_root}/ticknet-data/eventstream-top400-h5-fold0-benchmark-202101"
            )
            feature_local = "/content/ticknet-eventstream/top400-h5-fold0"
        else:
            run_name = "eventstream-top400-h5-capacity100m-recent"
            checkpoint_name = "eventstream-top400-h5-capacity100m-recent"
            feature_remote = (
                f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-benchmark-202508"
            )
            feature_local = "/content/ticknet-eventstream/top400-h5-recent"
        gpu_label = arguments.gpu.lower()
        if workflow == "eventstream-recent-batch-size-sweep":
            output_remote = f"{drive_root}/ticknet-runs/{run_name}/batch-size-sweep/{gpu_label}"
            output_local = f"/content/ticknet-results/{run_name}/batch-size-sweep/{gpu_label}"
        elif workflow == "eventstream-recent-input-profile":
            output_remote = f"{drive_root}/ticknet-runs/{run_name}/input-profile/{gpu_label}"
            output_local = f"/content/ticknet-results/{run_name}/input-profile/{gpu_label}"
        else:
            output_remote = f"{drive_root}/ticknet-runs/{run_name}/benchmarks/{gpu_label}"
            output_local = f"/content/ticknet-results/{run_name}/{gpu_label}"
    elif workflow in (
        EVENTSTREAM_TRAIN_WORKFLOWS
        | EVENTSTREAM_PREDICTION_WORKFLOWS
        | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
    ):
        (
            run_name,
            checkpoint_name,
            feature_remote,
            feature_local,
            checkpoint_remote,
            checkpoint_local,
            output_remote,
            output_local,
        ) = _eventstream_job_paths(
            arguments,
            drive_root,
        )
    elif workflow in EVENTSTREAM_EMBEDDING_WORKFLOWS:
        seed = int(arguments.seeds[0])
        run_name = "eventstream-top400-h5-capacity100m-recent"
        checkpoint_name = "eventstream-top400-h5-capacity100m-recent"
        feature_remote = f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-close-cache"
        feature_local = "/content/ticknet-eventstream/close-cache/recent"
        training_manifest_remote = (
            f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-materialized/seed{seed}"
        )
        training_manifest_local = "/content/ticknet-eventstream/materialized/recent"
        checkpoint_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        checkpoint_local = "/content/ticknet-eventstream/checkpoint"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/embeddings/seed{seed}"
        output_local = f"/content/ticknet-results/{run_name}/embeddings/seed{seed}"
    elif workflow in EVENTSTREAM_JOINT_WORKFLOWS:
        seed = int(arguments.seeds[0])
        run_name = "eventstream-top400-h5-capacity100m-recent"
        checkpoint_name = "eventstream-top400-h5-capacity100m-recent"
        feature_remote = f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-close-cache"
        feature_local = "/content/ticknet-eventstream/close-cache/recent"
        joint_cache_remote = (
            f"{drive_root}/ticknet-data/eventstream-top400-h5-recent-joint-cache-v1"
        )
        joint_cache_local = "/content/ticknet-eventstream/joint-cache/recent"
        checkpoint_remote = f"{drive_root}/ticknet-runs/{run_name}/training"
        checkpoint_local = "/content/ticknet-eventstream/checkpoint"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/joint-finetune/seed{seed}"
        output_local = f"/content/ticknet-results/{run_name}/joint-finetune/seed{seed}"
    else:
        raise ValueError(f"未知 workflow：{workflow}")
    run_root = f"{drive_root}/ticknet-runs/{run_name}"
    if workflow not in (
        EVENTSTREAM_EMBEDDING_WORKFLOWS
        | EVENTSTREAM_JOINT_WORKFLOWS
        | EVENTSTREAM_PREDICTION_WORKFLOWS
        | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
    ):
        checkpoint_remote = (
            output_remote
            if workflow in {"raw1000-train", "capacity-matrix-train"} | EVENTSTREAM_TRAIN_WORKFLOWS
            else run_root
        )
        checkpoint_local = (
            output_local
            if workflow in {"raw1000-train", "capacity-matrix-train"} | EVENTSTREAM_TRAIN_WORKFLOWS
            else f"/content/drive/MyDrive/{run_root}"
        )
    return {
        "workflow": workflow,
        "rclone_remote": arguments.rclone_remote.rstrip(":"),
        "rclone_config": REMOTE_RCLONE_CONFIG,
        "feature_remote": feature_remote,
        "target_remote": f"{drive_root}/ticknet-data/nextday-raw-200-targets-v1",
        "checkpoint_remote": checkpoint_remote,
        "output_remote": output_remote,
        "feature_local": feature_local,
        "target_local": "/content/nextday-raw-200-targets-v1",
        "checkpoint_local": checkpoint_local,
        "training_manifest_remote": (
            training_manifest_remote if workflow in EVENTSTREAM_EMBEDDING_WORKFLOWS else None
        ),
        "training_manifest_local": (
            training_manifest_local if workflow in EVENTSTREAM_EMBEDDING_WORKFLOWS else None
        ),
        "joint_cache_remote": (
            joint_cache_remote if workflow in EVENTSTREAM_JOINT_WORKFLOWS else None
        ),
        "joint_cache_local": (
            joint_cache_local if workflow in EVENTSTREAM_JOINT_WORKFLOWS else None
        ),
        "target_overlay_remote": (
            _eventstream_target_overlay_paths(arguments, drive_root)[0]
            if workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
            else None
        ),
        "target_overlay_local": (
            _eventstream_target_overlay_paths(arguments, drive_root)[1]
            if workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
            else None
        ),
        "expected_pretrained_sha256": (
            EVENTSTREAM_CHECKPOINT_SHA256_BY_SEED[seed]
            if workflow in EVENTSTREAM_JOINT_WORKFLOWS
            else None
        ),
        "expected_gradient_checkpoint_sha256": (
            _gradient_audit_checkpoint_sha256(arguments)
            if workflow in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
            else None
        ),
        "output_local": output_local,
        "checkpoint_name": checkpoint_name,
        "eventstream_fold_id": (
            arguments.eventstream_fold_id
            if workflow
            in {
                "eventstream-rolling-train",
                "eventstream-rolling-export-predictions",
                "eventstream-rolling-gradient-audit",
                "eventstream-rolling-label-scale-train",
            }
            else None
        ),
        "matrix_cell": arguments.matrix_cell if workflow == "capacity-matrix-train" else None,
        "training_config": REMOTE_CONFIG,
        "wheel": REMOTE_WHEEL,
        "seeds": list(arguments.seeds),
        "horizons": list(arguments.horizons),
        "inference_batch_size": arguments.inference_batch_size,
        "embedding_batch_size": arguments.embedding_batch_size,
        "benchmark_batches": arguments.benchmark_batches,
        "warmup_batches": arguments.warmup_batches,
        "batch_sizes": list(arguments.batch_sizes),
        "num_workers": list(arguments.num_workers),
        "effective_batch_size": arguments.effective_batch_size,
        "training_epochs": arguments.training_epochs,
        "audit_batches": arguments.audit_batches,
        "day_supervision_mode": (
            arguments.day_supervision_mode
            if workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
            else None
        ),
        "day_loss_weight": (
            getattr(arguments, "day_loss_weight", None)
            if workflow in EVENTSTREAM_LABEL_SCALE_WORKFLOWS
            else None
        ),
        "evaluate_test": arguments.evaluate_test,
        "expected_parameter_count": (
            1_033_383
            if workflow == "capacity-matrix-train" and arguments.matrix_cell.startswith("1m-")
            else 100_604_180
            if workflow
            in EVENTSTREAM_BENCHMARK_WORKFLOWS
            | EVENTSTREAM_TRAIN_WORKFLOWS
            | EVENTSTREAM_EMBEDDING_WORKFLOWS
            | EVENTSTREAM_JOINT_WORKFLOWS
            | EVENTSTREAM_PREDICTION_WORKFLOWS
            | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
            else 100_817_575
        ),
        "projected_train_samples": (
            arguments.audit_batches * 8
            if workflow in EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
            else 120_000
            if workflow
            in {
                "eventstream-recent-batch-size-sweep",
                "eventstream-recent-input-profile",
                "eventstream-recent-export-embeddings",
                "eventstream-recent-joint-finetune",
                "eventstream-rolling-export-predictions",
            }
            | EVENTSTREAM_TRAIN_WORKFLOWS
            else 42_000
            if workflow == "eventstream-recent-capacity-benchmark"
            else 40_000
            if workflow == "eventstream-capacity-benchmark"
            else 70_805
            if workflow in {"raw1000-train", "capacity-matrix-train"}
            else 75_000
        ),
        "requested_gpu": arguments.gpu,
        "source_revision": source_revision,
        "experiment_source_revision": (arguments.experiment_source_revision or source_revision),
    }


def _validate_downloaded_summary(
    output_dir: Path,
    spec: dict[str, Any],
) -> None:
    summary_path = output_dir / "colab-run-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Colab job 缺少 colab-run-summary.json 完成标记")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fields = ["workflow", "source_revision"]
    if spec.get("experiment_source_revision") is not None:
        fields.append("experiment_source_revision")
    if spec.get("matrix_cell") is not None:
        fields.append("matrix_cell")
    if spec.get("eventstream_fold_id") is not None:
        fields.append("eventstream_fold_id")
    if spec.get("day_supervision_mode") is not None:
        fields.append("day_supervision_mode")
    if spec.get("day_loss_weight") is not None:
        fields.append("day_loss_weight")
    for field in fields:
        if summary.get(field) != spec[field]:
            raise RuntimeError(
                f"Colab job {field} 不匹配：{summary.get(field)!r} != {spec[field]!r}"
            )
    if (
        spec["workflow"]
        in EVENTSTREAM_TRAIN_WORKFLOWS
        | EVENTSTREAM_EMBEDDING_WORKFLOWS
        | EVENTSTREAM_JOINT_WORKFLOWS
        | EVENTSTREAM_PREDICTION_WORKFLOWS
        | EVENTSTREAM_GRADIENT_AUDIT_WORKFLOWS
    ):
        if summary.get("seeds") != spec["seeds"]:
            raise RuntimeError("Colab 事件流任务 seed 与请求不一致")
        if summary.get("test_status") != "locked_not_accessed":
            raise RuntimeError("Colab 事件流任务没有确认 2026 locked 保持隔离")
        expected_oos = (
            "evaluated"
            if spec["workflow"]
            in EVENTSTREAM_EMBEDDING_WORKFLOWS
            | EVENTSTREAM_JOINT_WORKFLOWS
            | EVENTSTREAM_PREDICTION_WORKFLOWS
            or spec["evaluate_test"]
            else "not_evaluated"
        )
        if summary.get("oos_status") != expected_oos:
            raise RuntimeError("Colab 事件流训练的 OOS 状态与请求不一致")
    if summary.get("status") != "complete":
        error = summary.get("error", "未提供远端错误")
        raise RuntimeError(f"Colab job 未完成：{error}")


def _dry_run_plan(
    arguments: argparse.Namespace,
    *,
    colab: str,
    revision: str,
) -> list[list[str]]:
    commands = [_colab_command(colab, "status", "-s", arguments.session)]
    if arguments.reuse_session:
        commands.append(["lifecycle", "reuse-existing-session", arguments.session])
    else:
        commands.append(
            _colab_command(colab, "new", "-s", arguments.session, "--gpu", arguments.gpu)
        )
    commands.extend(
        [
            _colab_command(colab, "upload", "-s", arguments.session, "<wheel>", REMOTE_WHEEL),
            _colab_command(
                colab,
                "upload",
                "-s",
                arguments.session,
                str(arguments.config),
                REMOTE_CONFIG,
            ),
            _colab_command(
                colab,
                "upload",
                "-s",
                arguments.session,
                str(arguments.rclone_config),
                REMOTE_RCLONE_CONFIG,
            ),
            _colab_command(colab, "upload", "-s", arguments.session, "<job-spec>", REMOTE_SPEC),
            _colab_command(
                colab,
                "exec",
                "-s",
                arguments.session,
                "-f",
                str(JOB_SCRIPT),
                "--timeout",
                str(arguments.timeout),
            ),
            ["rclone", "copy", f"{arguments.rclone_remote}:<drive-output>", "<local-output>"],
            _colab_command(colab, "log", "-s", arguments.session, "-o", "<execution.ipynb>"),
            _colab_command(colab, "rm", "-s", arguments.session, REMOTE_RCLONE_CONFIG),
        ]
    )
    if arguments.reuse_session:
        commands.append(["lifecycle", "keep-reused-session", arguments.session])
    elif arguments.keep_session:
        commands.append(["lifecycle", "keep-session", arguments.session])
    elif arguments.keep_on_failure:
        commands.append(["lifecycle", "stop-on-success", "keep-on-failure"])
    else:
        commands.append(_colab_command(colab, "stop", "-s", arguments.session))
    commands.append(["revision", revision])
    return commands


def run(arguments: argparse.Namespace) -> None:
    _validate_lifecycle_arguments(arguments)
    repository_root = REPOSITORY_ROOT.resolve()
    if arguments.config is None:
        arguments.config = _default_config(
            arguments.workflow,
            arguments.matrix_cell,
            arguments.eventstream_fold_id,
        )
    arguments.config = arguments.config.expanduser().resolve()
    arguments.rclone_config = arguments.rclone_config.expanduser().resolve()
    arguments.local_output_dir = arguments.local_output_dir.expanduser().resolve()
    _ensure_secret_outside_repository(arguments.rclone_config, repository_root)
    for path in (arguments.config, arguments.rclone_config, JOB_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    colab = _require_executable("colab")
    rclone = _require_executable("rclone")
    uv = _require_executable("uv")
    revision = _source_revision(repository_root)
    spec = build_job_spec(arguments, revision)

    if arguments.dry_run:
        for command in _dry_run_plan(arguments, colab=colab, revision=revision):
            print(json.dumps(command, ensure_ascii=False))
        return

    _require_clean_revision(repository_root)
    session_exists = _session_exists(colab, arguments.session)
    _validate_session_selection(
        session=arguments.session,
        exists=session_exists,
        reuse_session=arguments.reuse_session,
    )
    _run(
        [
            rclone,
            "--config",
            str(arguments.rclone_config),
            "lsjson",
            f"{spec['rclone_remote']}:{spec['feature_remote']}/manifest.json",
        ],
        capture_output=True,
    )
    arguments.local_output_dir.mkdir(parents=True, exist_ok=True)
    session_active = session_exists
    session_owned = False
    secret_uploaded = False
    succeeded = False
    with tempfile.TemporaryDirectory(prefix="ticknet-colab-") as temporary:
        staging = Path(temporary)
        spec_path = staging / "job.json"
        wheel = _build_committed_wheel(uv, repository_root, staging, revision)
        spec["wheel"] = _remote_wheel_path(wheel)
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            if not arguments.reuse_session:
                _run(
                    _colab_command(
                        colab,
                        "new",
                        "-s",
                        arguments.session,
                        "--gpu",
                        arguments.gpu,
                    )
                )
                session_active = True
                session_owned = True
            uploads = (
                (wheel, spec["wheel"]),
                (arguments.config, REMOTE_CONFIG),
                (arguments.rclone_config, REMOTE_RCLONE_CONFIG),
                (spec_path, REMOTE_SPEC),
            )
            for local_path, remote_path in uploads:
                if remote_path == REMOTE_RCLONE_CONFIG:
                    secret_uploaded = True
                _run(
                    _colab_command(
                        colab,
                        "upload",
                        "-s",
                        arguments.session,
                        str(local_path),
                        remote_path,
                    )
                )
            _run(
                _colab_command(
                    colab,
                    "exec",
                    "-s",
                    arguments.session,
                    "-f",
                    str(JOB_SCRIPT),
                    "--timeout",
                    str(arguments.timeout),
                )
            )
            _run(
                [
                    rclone,
                    "--config",
                    str(arguments.rclone_config),
                    "copy",
                    f"{spec['rclone_remote']}:{spec['output_remote']}",
                    str(arguments.local_output_dir),
                    "--checksum",
                ]
            )
            _validate_downloaded_summary(arguments.local_output_dir, spec)
            succeeded = True
        finally:
            if session_active:
                _run(
                    _colab_command(
                        colab,
                        "log",
                        "-s",
                        arguments.session,
                        "-o",
                        str(arguments.local_output_dir / "execution.ipynb"),
                    ),
                    check=False,
                )
                if secret_uploaded:
                    _run(
                        _colab_command(
                            colab,
                            "rm",
                            "-s",
                            arguments.session,
                            REMOTE_RCLONE_CONFIG,
                        ),
                        check=False,
                    )
                if session_owned and _should_stop_owned_session(
                    arguments,
                    succeeded=succeeded,
                ):
                    _run(
                        _colab_command(colab, "stop", "-s", arguments.session),
                        check=False,
                    )
                elif session_owned:
                    reason = "任务失败" if not succeeded else "--keep-session"
                    print(f"保留 Colab session '{arguments.session}'：{reason}")
                elif arguments.reuse_session:
                    print(f"复用的 Colab session '{arguments.session}' 保持运行")


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
