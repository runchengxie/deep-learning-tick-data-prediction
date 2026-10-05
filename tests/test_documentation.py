from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
ARCHIVAL_SOURCE_START = "<!-- archival-source:start -->"
ARCHIVAL_SOURCE_END = "<!-- archival-source:end -->"
PRESERVED_SOURCE_START = "<!-- preserved-source:start -->"
PRESERVED_SOURCE_END = "<!-- preserved-source:end -->"
CJK_PROSE = re.compile(r"[\u3400-\u9fff]")
ENGLISH_DOCUMENTS = {
    "README.md",
    "AGENTS.md",
    "MIGRATION-STATUS.md",
    "docs/index.md",
    "docs/documentation-index.md",
    "docs/project-status.md",
    "docs/model-catalog.md",
    "docs/reproduction-audit.md",
    "docs/architecture/data-boundary.md",
    "docs/dev/colab-cli-automation.md",
    "docs/dev/development-guide.md",
    "docs/operations/development-guide.md",
    "docs/operations/systemd-workflows.md",
    "docs/references/README.md",
    "docs/references/agentx-paper-notes.md",
    "docs/references/debang-minute-gru-notes.md",
    "docs/references/deeplob-paper-notes.md",
    "docs/dev/historical-colab-snapshots.md",
    "docs/nextday/cross-sectional-prediction.md",
    "docs/nextday/eventstream.md",
    "docs/nextday/h5-rolling-eventstream-roadmap.md",
    "docs/nextday/hardware-constraints-and-experiment-roadmap.md",
    "docs/nextday/multi-horizon-data-expansion-roadmap.md",
    "docs/nextday/nextday-100m-raw1000-benchmark.md",
    "docs/nextday/raw-200-end-to-end-pipeline.md",
    "docs/nextday/raw-data-expansion-roadmap.md",
    "docs/reports/multi-horizon-decision-2026-08-10/source-inspection.md",
    "docs/research/experiment-log.md",
    "docs/research/eventstream-gradient-audit.md",
    "docs/research/eventstream-label-scale.md",
    "docs/research/eventstream-signal-trading-diagnostics.md",
    "docs/research/external-l2-research-comparison.md",
    "docs/research/historical-data-eligibility-2026-08-27.md",
    "docs/research/m3-eventstream-representation.md",
    "docs/research/opening-coverage-inventory-2026-08-27.md",
    "docs/research/resource-strategy-and-pilot-gates.md",
    "docs/research/shanghai-opening-contract-audit-2026-08-27.md",
    "docs/research/topk-agentx-m0-research-contract.md",
    "docs/research/topk-agentx-m1-portfolio-evaluator.md",
    "docs/research/topk-agentx-m2a-deterministic-loop.md",
    "docs/research/topk-agentx-m2b-locked-approval.md",
    "docs/research/topk-agentx-m2c-executors-comparison.md",
    "docs/research/topk-agentx-m2d-registry-context.md",
    "docs/research/topk-agentx-m3-topk-diagnostics.md",
    "docs/research/topk-agentx-research-roadmap.md",
}


def _markdown_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "MIGRATION-STATUS.md",
        *sorted(
            path
            for path in (ROOT / "docs").rglob("*.md")
            if "superpowers" not in path.relative_to(ROOT / "docs").parts
        ),
    ]


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_fence = False
    source_end_markers = {
        ARCHIVAL_SOURCE_START: ARCHIVAL_SOURCE_END,
        PRESERVED_SOURCE_START: PRESERVED_SOURCE_END,
    }
    active_source_end: str | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if active_source_end is not None:
            if active_source_end in raw_line:
                raw_line = raw_line.split(active_source_end, maxsplit=1)[1]
                active_source_end = None
            else:
                continue
        for source_start, source_end in source_end_markers.items():
            if source_start in raw_line:
                before, after_start = raw_line.split(source_start, maxsplit=1)
                if source_end in after_start:
                    raw_line = before + after_start.split(source_end, maxsplit=1)[1]
                else:
                    raw_line = before
                    active_source_end = source_end
                break
        if active_source_end is not None:
            continue
        if raw_line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not raw_line.startswith(("    ", "\t")):
            lines.append((line_number, INLINE_CODE.sub("", raw_line)))
    return lines


def test_archival_source_markers_are_balanced() -> None:
    failures: list[str] = []
    marker_pairs = (
        (ARCHIVAL_SOURCE_START, ARCHIVAL_SOURCE_END),
        (PRESERVED_SOURCE_START, PRESERVED_SOURCE_END),
    )
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for start, end in marker_pairs:
            if text.count(start) != text.count(end):
                failures.append(f"{path.relative_to(ROOT)}: {start} / {end}")
    assert not failures, "归档原文标记不成对:\n" + "\n".join(failures)


def test_external_comparison_archival_source_is_unchanged() -> None:
    path = ROOT / "docs" / "research" / "external-l2-research-comparison.md"
    text = path.read_text(encoding="utf-8")
    archived = text.split(f"{ARCHIVAL_SOURCE_START}\n", maxsplit=1)[1]
    archived = archived.split(ARCHIVAL_SOURCE_END, maxsplit=1)[0]
    digest = hashlib.sha256(archived.encode()).hexdigest()
    assert digest == "c9b84597e6e94c2d7d44eaa51079f497cbc31a761fc31aa48c5972f1e4103c27"


def test_repository_identity_is_quant_deep_learning() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    migration_status = (ROOT / "MIGRATION-STATUS.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = pyproject.split("[project]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]
    scripts = pyproject.split("[project.scripts]", maxsplit=1)[1].split("\n[", maxsplit=1)[0]

    assert "# Quant Deep Learning" in readme
    assert re.search(r'(?m)^name = "quant-deep-learning"$', project)
    assert "Canonical owner: `quant-deep-learning`" in migration_status
    assert "historical migration decision" in migration_status.lower()
    assert re.search(r"(?m)^ticknet-eventstream-train =", scripts)
    assert re.search(r"(?m)^ticknet-research =", scripts)


def test_fi2010_reproduction_is_a_first_party_package() -> None:
    root = ROOT / "src" / "ticknet" / "fi2010"
    scripts = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert root.joinpath("core.py").is_file()
    assert root.joinpath("train.py").is_file()
    assert not (ROOT / "legacy").exists()
    assert 'ticknet-fi2010-train = "ticknet.fi2010.train:main"' in scripts
    assert 'ticknet-fi2010-convert = "ticknet.fi2010.scripts.convert_fi2010:main"' in scripts


def test_mkdocs_site_configuration() -> None:
    config = ROOT / "mkdocs.yml"
    landing_page = ROOT / "docs" / "index.md"

    assert config.is_file()
    assert landing_page.is_file()
    settings = config.read_text(encoding="utf-8")
    assert "name: material" in settings
    assert "site_url: https://runchengxie.github.io/quant-deep-learning/" in settings
    assert "  - Home: index.md" in settings
    assert "Project status: project-status.md" in settings
    assert "Architecture:" in settings


def test_unlicensed_broker_report_is_not_published() -> None:
    report = ROOT / "docs" / "references" / "德邦证券_分钟数据GRU选股策略初探.pdf"
    assert not report.exists()


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


def test_maintained_documentation_is_written_in_english() -> None:
    failures: list[str] = []
    for relative_path in sorted(ENGLISH_DOCUMENTS):
        path = ROOT / relative_path
        for line_number, line in _prose_lines(path):
            line = MARKDOWN_LINK.sub(r"\1", line)
            if CJK_PROSE.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}")
    message = (
        "Maintained explanatory prose must be English "
        "(use preserved-source markers for quotations):\n" + "\n".join(failures)
    )
    assert not failures, message


def test_english_inventory_covers_all_maintained_markdown() -> None:
    maintained = {path.relative_to(ROOT).as_posix() for path in _markdown_files()}
    excluded = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "docs" / "superpowers").rglob("*.md")
    }
    assert maintained - excluded == ENGLISH_DOCUMENTS
