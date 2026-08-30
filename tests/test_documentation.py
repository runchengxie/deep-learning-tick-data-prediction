from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
ARCHIVAL_SOURCE_START = "<!-- archival-source:start -->"
ARCHIVAL_SOURCE_END = "<!-- archival-source:end -->"
STYLE_RULES = {
    "双引号": re.compile(r'[“”"]'),
    "强调": re.compile(r"\*\*"),
    "分号": re.compile(r"[；;]"),
    "破折号": re.compile(r"——|—"),
    "先否定再转折": re.compile(r"(?:不是|并非).{0,50}而是"),
    "中文旁的半角括号": re.compile(r"[\u3400-\u9fff]\(|\)[\u3400-\u9fff]"),
}


def _markdown_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        *sorted(
            path
            for path in (ROOT / "docs").rglob("*.md")
            if "superpowers" not in path.relative_to(ROOT / "docs").parts
        ),
    ]


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    in_archival_source = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw_line.strip() == ARCHIVAL_SOURCE_START:
            in_archival_source = True
            continue
        if raw_line.strip() == ARCHIVAL_SOURCE_END:
            in_archival_source = False
            continue
        if in_archival_source:
            continue
        if raw_line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not raw_line.startswith(("    ", "\t")):
            lines.append((line_number, INLINE_CODE.sub("", raw_line)))
    return lines


def test_archival_source_markers_are_balanced() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        if text.count(ARCHIVAL_SOURCE_START) != text.count(ARCHIVAL_SOURCE_END):
            failures.append(str(path.relative_to(ROOT)))
    assert not failures, "归档原文标记不成对:\n" + "\n".join(failures)


def test_external_comparison_archival_source_is_unchanged() -> None:
    path = ROOT / "docs" / "research" / "external-l2-research-comparison.md"
    text = path.read_text(encoding="utf-8")
    archived = text.split(f"{ARCHIVAL_SOURCE_START}\n", maxsplit=1)[1]
    archived = archived.split(ARCHIVAL_SOURCE_END, maxsplit=1)[0]
    digest = hashlib.sha256(archived.encode()).hexdigest()
    assert digest == "c9b84597e6e94c2d7d44eaa51079f497cbc31a761fc31aa48c5972f1e4103c27"


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
