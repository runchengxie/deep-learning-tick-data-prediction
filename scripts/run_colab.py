"""Colab 训练入口。

先在 Colab 主内核中挂载 Google Drive，再运行本脚本：

    from google.colab import drive
    drive.mount("/content/drive")
    !python scripts/run_colab.py --protocol setup2 --k 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive")
META_FILENAMES = ("FI2010_normalised_meta.json", "FI2010_meta.json")


def _resolve_meta_path(data_dir: Path) -> Path:
    """优先使用标准名称，同时兼容已有的简写元数据文件名。"""
    for filename in META_FILENAMES:
        path = data_dir / filename
        if path.is_file():
            return path
    return data_dir / META_FILENAMES[0]


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
