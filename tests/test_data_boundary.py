from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
BANNED_DATA_RUNTIME_IMPORTS = (
    "market_data_platform",
    "tushare",
    "rqdatac",
)
BANNED_DATA_RUNTIME_DEPENDENCIES = {
    "market-data-platform",
    "tushare",
    "rqdatac",
}


def _imported_names(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.lineno, node.module or ""))
    return imported


def _project_dependencies() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^dependencies\s*=\s*(\[.*?^\])", text)
    assert match is not None
    dependencies = ast.literal_eval(match.group(1))
    return {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in dependencies
    }


def test_data_boundary_document_exists_and_names_the_two_owners() -> None:
    document = ROOT / "docs" / "architecture" / "data-boundary.md"
    text = document.read_text(encoding="utf-8")

    assert "market-data-platform" in text
    assert "ticknet" in text
    assert "canonical" in text
    assert "window" in text
    assert "label" in text


def test_ticknet_does_not_import_platform_or_provider_runtimes() -> None:
    violations = [
        f"{path.relative_to(ROOT)}:{line}:{name}"
        for path in sorted((ROOT / "src" / "ticknet").rglob("*.py"))
        for line, name in _imported_names(path)
        if name.startswith(BANNED_DATA_RUNTIME_IMPORTS)
    ]

    assert violations == [], (
        "ticknet consumes published/canonical assets through adapters and must not own "
        "market-data-platform business code or provider SDK access:\n" + "\n".join(violations)
    )


def test_project_dependencies_exclude_platform_and_provider_runtimes() -> None:
    forbidden = _project_dependencies() & BANNED_DATA_RUNTIME_DEPENDENCIES

    assert forbidden == set()


def test_nextday_sources_do_not_reenter_legacy_snapshot_facade() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "src" / "ticknet").rglob("*.py")):
        if path.name == "raw_snapshot.py":
            continue
        for line, name in _imported_names(path):
            if name == "ticknet.nextday.raw_snapshot":
                violations.append(f"{path.relative_to(ROOT)}:{line}")

    assert violations == [], (
        "nextday implementation must import split modules directly; keep raw_snapshot "
        "for external compatibility only:\n" + "\n".join(violations)
    )
