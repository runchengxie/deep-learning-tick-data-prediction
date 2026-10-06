from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_ci_runs_on_main_push_and_pull_request() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "  push:" in workflow
    assert "    branches: [main]" in workflow
    assert "  pull_request:" in workflow
    assert "group: quant-deep-learning-ci-" in workflow


def test_public_ci_runs_static_checks_tests_and_coverage() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in (
        "uv run --locked --extra dev ruff check",
        "uv run --locked --extra dev ruff format --check",
        "uv run --locked --extra dev ty check",
        "uv run --locked --extra dev nbqa ruff",
        "uv run --locked --extra dev pytest --cov",
        "scripts/ci_hygiene.py",
        "scripts/smoke_test.py",
    ):
        assert command in workflow


def test_public_site_build_runs_for_pull_requests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "  pull_request:" in workflow
    assert "npm test && npm run build:pages" in workflow
    assert "actions/upload-pages-artifact" in workflow
