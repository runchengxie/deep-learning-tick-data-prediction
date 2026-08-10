"""从 Linux 开发机无人值守调度 Colab 多周期 validation。"""

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
JOB_SCRIPT = REPOSITORY_ROOT / "scripts" / "colab_multi_horizon_job.py"
REMOTE_WHEEL = "/content/ticknet-job.whl"
REMOTE_CONFIG = "/content/ticknet-nextday-config.yaml"
REMOTE_RCLONE_CONFIG = "/content/ticknet-rclone.conf"
REMOTE_SPEC = "/content/ticknet-colab-job.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用官方 colab CLI 和 rclone 运行无人值守多周期 validation",
    )
    parser.add_argument("--session", default="ticknet-multi-horizon")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="T4")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    parser.add_argument("--keep-session", action="store_true")
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
    run_root = f"{drive_root}/ticknet-runs/raw-200-capacity_1m"
    return {
        "rclone_remote": arguments.rclone_remote.rstrip(":"),
        "rclone_config": REMOTE_RCLONE_CONFIG,
        "feature_remote": f"{drive_root}/ticknet-data/nextday-raw-200",
        "target_remote": f"{drive_root}/ticknet-data/nextday-raw-200-targets-v1",
        "checkpoint_remote": run_root,
        "output_remote": f"{run_root}/multi-horizon-validation-2024",
        "feature_local": "/content/nextday-raw-200",
        "target_local": "/content/nextday-raw-200-targets-v1",
        "checkpoint_local": (
            "/content/drive/MyDrive/deep-learning-tick-data-prediction/"
            "ticknet-runs/raw-200-capacity_1m"
        ),
        "output_local": "/content/ticknet-results/multi-horizon-validation-2024",
        "checkpoint_name": "raw-200-dual-head-capacity_1m",
        "training_config": REMOTE_CONFIG,
        "wheel": REMOTE_WHEEL,
        "seeds": list(arguments.seeds),
        "horizons": list(arguments.horizons),
        "inference_batch_size": arguments.inference_batch_size,
        "source_revision": source_revision,
    }


def _dry_run_plan(
    arguments: argparse.Namespace,
    *,
    colab: str,
    revision: str,
) -> list[list[str]]:
    return [
        _colab_command(colab, "new", "-s", arguments.session, "--gpu", arguments.gpu),
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
        _colab_command(colab, "stop", "-s", arguments.session),
        ["revision", revision],
    ]


def run(arguments: argparse.Namespace) -> None:
    repository_root = REPOSITORY_ROOT.resolve()
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
    session_created = False
    with tempfile.TemporaryDirectory(prefix="ticknet-colab-") as temporary:
        staging = Path(temporary)
        spec_path = staging / "job.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        wheel = _build_committed_wheel(uv, repository_root, staging, revision)
        try:
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
            session_created = True
            uploads = (
                (wheel, REMOTE_WHEEL),
                (arguments.config, REMOTE_CONFIG),
                (arguments.rclone_config, REMOTE_RCLONE_CONFIG),
                (spec_path, REMOTE_SPEC),
            )
            for local_path, remote_path in uploads:
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
        finally:
            if session_created:
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
                if not arguments.keep_session:
                    _run(
                        _colab_command(colab, "stop", "-s", arguments.session),
                        check=False,
                    )


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
