"""从 Linux 开发机无人值守调度 Colab 次日模型任务。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "nextday-raw-200-capacity-1m.yaml"
DEFAULT_H5_CONFIG = REPOSITORY_ROOT / "configs" / "nextday-raw-200-capacity-1m-h5.yaml"
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
        choices=("multi-horizon-validation", "h5-train"),
        default="multi-horizon-validation",
    )
    parser.add_argument("--session", default="ticknet-multi-horizon")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="T4")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--rclone-config",
        type=Path,
        default=Path("~/.config/rclone/rclone.conf").expanduser(),
    )
    parser.add_argument("--rclone-remote", default="gdrive")
    parser.add_argument("--drive-root", default="deep-learning-tick-data-prediction")
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--inference-batch-size", type=int, default=128)
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


def build_job_spec(arguments: argparse.Namespace, source_revision: str) -> dict[str, Any]:
    drive_root = arguments.drive_root.strip("/")
    workflow = arguments.workflow
    if workflow == "multi-horizon-validation":
        run_name = "raw-200-capacity_1m"
        checkpoint_name = "raw-200-dual-head-capacity_1m"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}/multi-horizon-validation-2024"
        output_local = "/content/ticknet-results/multi-horizon-validation-2024"
    elif workflow == "h5-train":
        run_name = "raw-200-capacity_1m-h5"
        checkpoint_name = "raw-200-dual-head-capacity_1m-h5"
        output_remote = f"{drive_root}/ticknet-runs/{run_name}"
        output_local = f"/content/drive/MyDrive/{output_remote}"
    else:
        raise ValueError(f"未知 workflow：{workflow}")
    run_root = f"{drive_root}/ticknet-runs/{run_name}"
    return {
        "workflow": workflow,
        "rclone_remote": arguments.rclone_remote.rstrip(":"),
        "rclone_config": REMOTE_RCLONE_CONFIG,
        "feature_remote": f"{drive_root}/ticknet-data/nextday-raw-200",
        "target_remote": f"{drive_root}/ticknet-data/nextday-raw-200-targets-v1",
        "checkpoint_remote": run_root,
        "output_remote": output_remote,
        "feature_local": "/content/nextday-raw-200",
        "target_local": "/content/nextday-raw-200-targets-v1",
        "checkpoint_local": f"/content/drive/MyDrive/{run_root}",
        "output_local": output_local,
        "checkpoint_name": checkpoint_name,
        "training_config": REMOTE_CONFIG,
        "wheel": REMOTE_WHEEL,
        "seeds": list(arguments.seeds),
        "horizons": list(arguments.horizons),
        "inference_batch_size": arguments.inference_batch_size,
        "source_revision": source_revision,
    }


def _validate_downloaded_summary(
    output_dir: Path,
    spec: dict[str, Any],
) -> None:
    summary_path = output_dir / "colab-run-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Colab job 缺少 colab-run-summary.json 完成标记")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field in ("workflow", "source_revision"):
        if summary.get(field) != spec[field]:
            raise RuntimeError(
                f"Colab job {field} 不匹配：{summary.get(field)!r} != {spec[field]!r}"
            )
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
        arguments.config = DEFAULT_H5_CONFIG if arguments.workflow == "h5-train" else DEFAULT_CONFIG
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
