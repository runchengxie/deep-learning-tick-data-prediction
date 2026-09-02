# Public CI Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the public CI workflow to validate pull requests and `main` pushes with the repository's existing Ruff, `ty`, pytest, and coverage checks.

**Architecture:** Keep one lightweight workflow for pull requests and `main` pushes. It installs the locked development environment, runs static checks and the synthetic CPU test suite, and reports coverage without adding a new threshold. GPU, training, slow, and real-data jobs remain manual or scheduled.

**Tech Stack:** GitHub Actions, uv, Python 3.11, Ruff, ty, pytest, pytest-cov.

**Spec:** `AGENTS.md` and `docs/dev/development-guide.md` quality-gate requirements.

## Global Constraints

- CI must not require private credentials, private submodules, real market data, or GPU hardware.
- CI uses the locked development dependencies declared by `pyproject.toml` and `uv.lock`.
- Coverage is reported but does not introduce a new minimum threshold.
- The public policy remains documented in `AGENTS.md`.

---

### Task 1: Add workflow regression coverage

**Files:**
- Create: `tests/test_public_ci_workflow.py`

- [x] **Step 1: Write the failing test**

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_public_ci_workflow.py -q`

Expected: FAIL because the current workflow has no `push`, `ty`, or pytest commands.

### Task 2: Extend the public workflow

**Files:**
- Modify: `.github/workflows/ci.yml`

- [x] **Step 1: Add the `main` push trigger and locked dev environment commands**

- [x] **Step 2: Run the workflow regression test**

Run: `pytest tests/test_public_ci_workflow.py -q`

Expected: PASS.

### Task 3: Document the CI boundary

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/dev/development-guide.md`

- [x] **Step 1: State that public CI runs on PRs and `main` pushes**
- [x] **Step 2: State that coverage is reported without a new CI threshold**
- [x] **Step 3: State that GPU, training, slow, and real-data checks remain manual or scheduled**
- [x] **Step 4: Run documentation and workflow tests**

Run: `uv run --locked --extra dev pytest tests/test_public_ci_workflow.py tests/test_documentation.py -q`

Expected: PASS.

### Task 4: Run the repository quality gate

- [x] **Step 1: Run:** `uv run --locked --extra dev ruff check .`
- [x] **Step 2: Run:** `uv run --locked --extra dev ruff format --check .`
- [x] **Step 3: Run:** `uv run --locked --extra dev ty check`
- [x] **Step 4: Run:** `uv run --locked --extra dev pytest --cov --cov-report=term-missing`
- [x] **Step 5: Review the final diff and commit the changes**
