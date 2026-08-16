from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
STYLE_RULES = {
    "双引号": re.compile(r'[“”"]'),
    "强调": re.compile(r"\*\*"),
    "分号": re.compile(r"[；;]"),
    "破折号": re.compile(r"——|—"),
    "先否定再转折": re.compile(r"(?:不是|并非).{0,50}而是"),
    "中文旁的半角括号": re.compile(r"[\u3400-\u9fff]\(|\)[\u3400-\u9fff]"),
}


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw_line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not raw_line.startswith(("    ", "\t")):
            lines.append((line_number, INLINE_CODE.sub("", raw_line)))
    return lines


def test_internal_markdown_links_exist() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        for line_number, line in _prose_lines(path):
            for _, raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip().strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}: {target}")
    assert not failures, "内部 Markdown 链接目标不存在:\n" + "\n".join(failures)


def test_chinese_document_style() -> None:
    failures: list[str] = []
    frozen_reports = ROOT / "docs" / "reports"
    for path in _markdown_files():
        if path.is_relative_to(frozen_reports):
            continue
        for line_number, line in _prose_lines(path):
            line = MARKDOWN_LINK.sub(r"\1", line)
            for name, pattern in STYLE_RULES.items():
                if pattern.search(line):
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}: {name}")
    assert not failures, "中文文档风格检查失败:\n" + "\n".join(failures)
