"""下载范围应保留本次 seed 与共享凭证，排除其他 seed。"""

import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_colab_nextday as runner


def test_training_download_filters_only_requested_seed(tmp_path: Path) -> None:
    args = Namespace(rclone_config=tmp_path / "rclone.conf", local_output_dir=tmp_path)
    spec = {
        "workflow": "eventstream-recent-train",
        "seeds": [1, 2],
        "rclone_remote": "gdrive",
        "output_remote": "runs/training",
    }
    command = runner._output_download_command(args, spec, "rclone")
    assert command[-6:] == [
        "--filter",
        "+ **.seed1.*",
        "--filter",
        "+ **.seed2.*",
        "--filter",
        "- **.seed*.*",
    ]
    assert "--checksum" in command


def test_non_training_download_retains_existing_scope(tmp_path: Path) -> None:
    args = Namespace(rclone_config=tmp_path / "rclone.conf", local_output_dir=tmp_path)
    spec = {
        "workflow": "eventstream-export-embeddings",
        "seeds": [1],
        "rclone_remote": "gdrive",
        "output_remote": "runs/embeddings/seed1",
    }
    assert "--filter" not in runner._output_download_command(args, spec, "rclone")


def test_real_rclone_excludes_other_seeds_but_keeps_shared_files(tmp_path: Path) -> None:
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("rclone is not installed")
    source = tmp_path / "source"
    source.mkdir()
    names = [
        "model.seed0.best.pt",
        "model.seed1.best.pt",
        "model.seed10.last.pt",
        "model.seed2.last.pt",
        "job_summary.json",
        "metrics.seed1.json",
    ]
    for name in names:
        (source / name).write_text(name)
    target = tmp_path / "target"
    args = Namespace(rclone_config=Path("/dev/null"), local_output_dir=target)
    spec = {
        "workflow": "eventstream-recent-train",
        "seeds": [1, 2],
        "rclone_remote": "unused",
        "output_remote": "unused",
    }
    command = runner._output_download_command(args, spec, rclone)
    command[4] = str(source)
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert {p.name for p in target.iterdir()} == {
        "model.seed1.best.pt",
        "model.seed2.last.pt",
        "job_summary.json",
        "metrics.seed1.json",
    }
