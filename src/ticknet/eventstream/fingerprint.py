"""事件流训练数据的确定性指纹。

对训练用到的所有 pack 日文件 + 共享 label 文件做 SHA-256 摘要，任一字节变化
都会改变指纹。写入训练签名，供 resume 与后续 locked test 校验数据一致性。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from ticknet.eventstream.config import PACK_ROOT, day_pack_paths

_CHUNK = 1 << 20  # 1 MiB


def file_sha256(path: str | Path) -> str:
    """流式读取文件计算 SHA-256（大文件也保持内存有界）。"""
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(
    days: list[int],
    root: Path = PACK_ROOT,
    label_path: Path | None = None,
) -> str:
    """pack 日文件 + 可选 label 文件的确定性指纹。"""
    root = Path(root)
    sorted_days = sorted({int(d) for d in days})
    packs: list[dict] = []
    for day in sorted_days:
        day_files: dict[str, str | None] = {}
        for kind, path in day_pack_paths(day, root).items():
            day_files[kind] = file_sha256(path) if path.exists() else None
        packs.append({"day": day, "files": day_files})
    label = (
        file_sha256(label_path) if label_path is not None and Path(label_path).exists() else None
    )
    return _canonical_sha256({"days": sorted_days, "packs": packs, "label": label})


def git_sha(repo_root: Path | None = None) -> str:
    """当前仓库 HEAD，解析失败返回 'unknown'。"""
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        head = result.stdout.strip()
        return head if head else "unknown"
    except Exception:
        return "unknown"
