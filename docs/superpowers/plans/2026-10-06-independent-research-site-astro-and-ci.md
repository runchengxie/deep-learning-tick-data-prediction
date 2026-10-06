# Independent Research Site and CI-First Quality Plan

> **For agentic workers:** Implement this plan task-by-task. Preserve research evidence and keep user data outside Git.

**Goal:** Publish Quant Deep Learning as an independent, chart-supported Astro research site and move repository quality enforcement from local pre-commit hooks into CI.

**Architecture:** Astro renders an explicit public allowlist of Markdown and reviewed result snapshots into one GitHub Pages artifact. The existing Python package remains independent of Astro and upstream/downstream repositories. Charts are static and include accessible value tables. CI owns file hygiene, static analysis, tests, site builds, and public-output safety checks; developers can run those commands manually without installed Git hooks.

**Tech Stack:** Astro 7 static output, semantic HTML/CSS charts, Node, Python 3.12/uv, Ruff, ty, pytest, nbqa, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-06-quant-deep-learning-independent-research-site-design.md`.

## Global Constraints

- The canonical dependency direction is `quant-market-data-platform` → this repository → `quant-backtest-runtime` through published data and artifacts.
- Do not add Python dependencies on `quant-platform` or `quant-research`.
- Public pages use reviewed allowlisted evidence only; CI must not train models or read private data.
- Preserve existing model/research code, CLI names, paper facts, and public evidence caveats.
- Preserve existing public GitHub Pages URLs where practical and publish one Astro artifact.
- English is the maintained language for project documentation and site prose.
- No developer machine hooks are required; CI is the quality gate.

## Review Focus

- Public-output allowlisting excludes plans, private runbooks, raw data, predictions, checkpoints, and local paths.
- Old documentation paths and heading anchors remain usable or have explicit compatibility routes.
- Evidence status, date/sample scope, and limitations remain adjacent to published results.
- Static pages render useful text/tables when JavaScript or chart payloads are unavailable.
- CI hygiene checks report errors without rewriting contributor files.

### Task 1: Move current ownership/status document and reconcile boundaries

**Files:** `MIGRATION-STATUS.md`, `docs/architecture/repository-boundaries.md`, `README.md`, `docs/index.md`, `AGENTS.md`, `tests/test_documentation.py`.

- [ ] Rename the current document into the architecture docs as the single current ownership contract.
- [ ] Preserve the prior migration decision as history and update ownership to the standalone model-research repo with market-data upstream and backtest-runtime downstream.
- [ ] Update internal links and documentation tests; retain a small root compatibility page only if needed for existing URL stability.

### Task 2: Build the Astro site and publish approved study content

**Files:** `web/astro.config.mjs`, `web/package.json`, `web/package-lock.json`, `web/src/pages/**`, `web/src/layouts/**`, `web/src/content/**`, `web/public/**`, `.github/workflows/pages.yml`, route/publication tests.

- [ ] Create a static Astro/React site under `/quant-deep-learning/` with overview, studies, model catalog, methods, and technical docs routes.
- [ ] Add an explicit public content registry; exclude `docs/superpowers/**`, operations runbooks, internal snapshots, raw/private artifacts, and unreviewed records.
- [ ] Render existing English Markdown through Astro and preserve headings, internal links, and base-path behavior.
- [ ] Add evidence-first study cards and representative study visualizations only from reviewed checked-in summaries; show status, sample window, metric, limitations, and table equivalents.
- [ ] Replace MkDocs Pages build/deploy with Astro build and public-output route/privacy checks.

### Task 3: Remove MkDocs and pre-commit after CI parity

**Files:** `mkdocs.yml`, `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`, `scripts/check.py`, `.github/workflows/ci.yml`, `tests/test_public_ci_workflow.py`, current contributor instructions.

- [ ] Add CI file-hygiene checks for changed files: large additions, conflict markers, JSON/TOML/YAML syntax, private-key markers, trailing whitespace, and final newlines.
- [ ] Run nbqa Ruff in CI in addition to existing Ruff, ty, pytest/coverage, compile, and smoke checks.
- [ ] Remove MkDocs and pre-commit from development dependencies, configurations, and current setup instructions once their responsibilities are covered by Astro and CI.
- [ ] Retain a manual `uv run` quality command; do not install Git hooks.

### Task 4: Validate and deliver

- [ ] Run Python CI-equivalent checks, documentation tests, Astro unit/build/static verification, and available browser smoke tests.
- [ ] Confirm Pages contains only the Astro static artifact and the Python package imports without site dependencies.
- [ ] Commit, push, and update the design PR with implementation; report required checks and any remaining deployment setting outside Git.
