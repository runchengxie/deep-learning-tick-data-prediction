"""Colab 训练入口。

先在 Colab 主内核中挂载 Google Drive，再运行本脚本：

    from google.colab import drive
    drive.mount("/content/drive")
    !python scripts/run_colab.py --protocol setup2 --k 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_LOCAL_DATA_DIR = Path("/content/DeepLOB/data")
META_FILENAMES = ("FI2010_normalised_meta.json", "FI2010_meta.json")


def _resolve_meta_path(data_dir: Path) -> Path:
    """优先使用标准名称，同时兼容已有的简写元数据文件名。"""
    for filename in META_FILENAMES:
        path = data_dir / filename
        if path.is_file():
            return path
    return data_dir / META_FILENAMES[0]


def _files_match(source: Path, destination: Path) -> bool:
    """按文件大小和修改时间判断本地暂存文件是否可以复用。"""
    if not destination.is_file():
        return False
    source_stat = source.stat()
    destination_stat = destination.stat()
    return (
        source_stat.st_size == destination_stat.st_size
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
    )


def _copy_to_local(source: Path, local_dir: Path) -> Path:
    """把一个 Drive 文件原子复制到 Colab 本地盘，并复用完整的已有副本。"""
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / source.name
    if source.resolve() == destination.resolve():
        return source
    if _files_match(source, destination):
        print(f"复用本地数据：{destination}")
        return destination

    temporary = destination.with_name(f".{destination.name}.copying")
    temporary.unlink(missing_ok=True)
    required_bytes = source.stat().st_size
    free_bytes = shutil.disk_usage(local_dir).free
    if free_bytes < required_bytes:
        raise SystemExit(
            f"Colab 本地盘空间不足。复制 {source.name} 需要 "
            f"{required_bytes / 1024**3:.2f} GiB，当前可用 {free_bytes / 1024**3:.2f} GiB。"
        )

    print(
        f"正在把 {source.name} 从 Google Drive 复制到 Colab 本地盘 "
        f"({required_bytes / 1024**3:.2f} GiB)…",
        flush=True,
    )
    started_at = time.perf_counter()
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    duration = time.perf_counter() - started_at
    throughput = required_bytes / 1024**2 / max(duration, 1e-9)
    print(f"复制完成：{destination}｜{duration:.1f}s｜{throughput:.1f} MiB/s")
    return destination


def _stage_data_files(data_path: Path, meta_path: Path, local_dir: Path) -> tuple[Path, Path]:
    """把训练数据和元数据暂存到 Colab 本地盘。"""
    return (
        _copy_to_local(data_path, local_dir),
        _copy_to_local(meta_path, local_dir),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 Colab 上运行 DeepLOB")
    parser.add_argument(
        "--protocol",
        choices=["setup1", "setup2"],
        default="setup2",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=[10, 20, 30, 50, 100],
        default=10,
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=DEFAULT_DRIVE_ROOT,
    )
    parser.add_argument(
        "--local-data-dir",
        type=Path,
        default=DEFAULT_LOCAL_DATA_DIR,
        help="训练数据在 Colab 本地盘的暂存目录",
    )
    parser.add_argument(
        "--no-local-copy",
        action="store_true",
        help="直接读取 Google Drive，仅供排查或本地盘空间不足时使用",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    drive_root = args.drive_root.expanduser()
    if not drive_root.is_dir():
        raise SystemExit(
            "Google Drive 尚未挂载。请先在 Colab 单元格中运行\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        )

    data_dir = drive_root / "DeepLOB" / "data"
    checkpoint_dir = drive_root / "DeepLOB" / "checkpoints"
    data_path = data_dir / "FI2010_normalised.npy"
    meta_path = _resolve_meta_path(data_dir)
    missing = [path for path in (data_path, meta_path) if not path.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(
            f"缺少以下文件：\n{joined}\n"
            "元数据也可以使用兼容名称 FI2010_meta.json。\n"
            "请先用 scripts/convert_fi2010.py 转换官方数据，再上传到 Drive。"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_local_copy:
        data_path, meta_path = _stage_data_files(
            data_path,
            meta_path,
            args.local_data_dir.expanduser(),
        )
    else:
        print("已关闭本地复制，本次训练会直接读取 Google Drive。")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(REPOSITORY_ROOT),
        ],
        check=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("未检测到 CUDA，本次会使用 CPU，训练速度会明显变慢。")

    command = [
        sys.executable,
        "-m",
        "deeplob.train",
        "--config",
        str(REPOSITORY_ROOT / "configs" / "colab.yaml"),
        "--data-path",
        str(data_path),
        "--meta-path",
        str(meta_path),
        "--protocol",
        args.protocol,
        "--k",
        str(args.k),
        "--device",
        device,
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    print(f"训练结束，结果保存在 {checkpoint_dir}")


if __name__ == "__main__":
    main()
