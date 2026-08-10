"""在已分配的 Colab VM 内执行无 Drive mount 的次日模型任务。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SPEC_PATH = Path("/content/ticknet-colab-job.json")


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
    _rclone_copy(
        _drive_path(remote, str(spec["feature_remote"])),
        str(spec["feature_local"]),
        env=env,
    )
    _rclone_copy(
        _drive_path(remote, str(spec["target_remote"])),
        str(spec["target_local"]),
        env=env,
    )
    checkpoint_root = Path(str(spec["checkpoint_local"]))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    workflow = str(spec["workflow"])
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
    elif workflow == "h5-train":
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
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
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


def _train_h5(spec: dict[str, Any]) -> None:
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
        "seeds": spec["seeds"],
        "output_remote": spec["output_remote"],
        "test_status": "locked_not_accessed",
    }
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


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rclone_config = Path(str(spec["rclone_config"]))
    if not rclone_config.is_file():
        raise FileNotFoundError(f"缺少临时 rclone 配置：{rclone_config}")
    rclone_config.chmod(0o600)
    env = {**os.environ, "RCLONE_CONFIG": str(rclone_config)}
    try:
        _ensure_rclone()
        _install_project(spec)
        _stage_inputs(spec, env)
        try:
            if spec["workflow"] == "multi-horizon-validation":
                _evaluate(spec)
            elif spec["workflow"] == "h5-train":
                _train_h5(spec)
            else:
                raise ValueError(f"未知 workflow：{spec['workflow']}")
        except Exception as error:
            if spec["workflow"] == "h5-train":
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
                },
                ensure_ascii=False,
            )
        )
    finally:
        rclone_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
