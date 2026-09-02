from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_ci_runs_on_main_push_and_pull_request() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "  push:" in workflow
    assert "    branches: [main]" in workflow
    assert "  pull_request:" in workflow


def test_public_ci_runs_static_checks_tests_and_coverage() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for command in (
        "uv run --locked --extra dev ruff check",
        "uv run --locked --extra dev ruff format --check",
        "uv run --locked --extra dev ty check",
        "uv run --locked --extra dev pytest --cov",
    ):
        assert command in workflow
