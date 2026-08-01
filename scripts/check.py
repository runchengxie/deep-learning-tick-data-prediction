"""运行本地质量门禁，供人工执行和 pre-push hook 调用。"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _project_python() -> Path:
    candidates = (
        REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe",
        REPOSITORY_ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        '缺少项目虚拟环境。请先创建 .venv 并安装开发依赖： python -m pip install -e ".[dev]"'
    )


def main() -> None:
    python = _project_python()
    checks = (
        ("Ruff 静态检查", [python, "-m", "ruff", "check", "."]),
        ("Ruff 格式检查", [python, "-m", "ruff", "format", "--check", "."]),
        ("ty 类型检查", [python, "-m", "ty", "check"]),
        (
            "pytest 与覆盖率",
            [python, "-m", "pytest", "--cov", "--cov-report=term-missing"],
        ),
        ("冒烟检查", [python, "scripts/smoke_test.py"]),
    )

    for label, command in checks:
        print(f"\n==> {label}", flush=True)
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)

    print("\n全部本地质量门禁通过。")


if __name__ == "__main__":
    main()
