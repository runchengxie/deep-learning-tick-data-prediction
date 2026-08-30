from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_data_boundary_document_exists_and_names_the_two_owners() -> None:
    document = ROOT / "docs" / "architecture" / "data-boundary.md"
    text = document.read_text(encoding="utf-8")

    assert "market-data-platform" in text
    assert "ticknet" in text
    assert "canonical" in text
    assert "window" in text
    assert "label" in text


def test_ticknet_does_not_import_market_data_platform_business_code() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith("market_data_platform") for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert violations == [], (
        "ticknet should consume published/canonical data through an adapter, not import "
        "market-data-platform business code:\n" + "\n".join(violations)
    )
