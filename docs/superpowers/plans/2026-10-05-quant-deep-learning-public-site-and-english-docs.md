# Quant Deep Learning Public Site and English Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reactivate the repository as `quant-deep-learning`, make its maintained explanatory documentation English-only, and publish a MkDocs Material showcase and documentation site through GitHub Pages.

**Architecture:** Keep the existing `ticknet` Python namespace and console commands stable while updating distribution metadata and public project identity. MkDocs Material builds one static site from the repository documentation; GitHub Actions builds it for pull requests and deploys the default branch using GitHub Pages artifacts. Ownership and study documentation will describe the intended artifact boundary, while code migration and artifact-schema implementation remain in a separate follow-on plan.

**Tech Stack:** Python 3.10+, existing `uv` lockfile, MkDocs, MkDocs Material, GitHub Actions Pages artifacts, pytest documentation checks.

**Spec:** `docs/superpowers/specs/2026-10-05-quant-deep-learning-reorganization-design.md`

## Global Constraints

- The project runs its model research workflows without `quant-platform`.
- Integrate systems through versioned prediction, signal, and model-manifest artifacts, with no required Python dependency between the model project and `quant-platform`.
- All maintained explanatory Markdown in the project, including the root README, `docs/`, and contributor instructions, will be written in clear English.
- Preserve dated observations, metrics, assumptions, failed experiments, sample boundaries, provenance, and uncertainty during translation.
- Code, CLI names, paths, schemas, bibliographic titles, direct quotations, and original research source files remain unchanged when translation would corrupt an identifier or evidence.
- CI must not train models or require private data, credentials, or GPU resources.
- The chip-layer source report itself stays out of the public repository unless redistribution rights are verified.
- Keep `ticknet` as a compatibility namespace; do not perform a broad Python package or CLI rename in this plan.

## Review Focus

- Historical values, dates, and caveats can drift during translation; verify translated prose against the source and retain a checked source-to-translation inventory.
- The archival excerpt in `docs/research/external-l2-research-comparison.md` is hash-pinned; preserve its bytes and existing hash test.
- Relative Markdown links can break when navigation labels or document names change; run the repository link checker and a strict MkDocs build.
- GitHub Pages deployment permissions or branch conditions can publish from the wrong event; test workflow structure and deploy only from the default branch.
- Changing the Python distribution name can break editable installs or console scripts; keep script entry points stable and verify the built metadata and lockfile.

---

### Task 1: Re-establish the public project identity

**Files:**
- Modify: `README.md`
- Modify: `MIGRATION-STATUS.md`
- Modify: `AGENTS.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: current repository metadata, current documentation, and the ownership boundaries in the spec.
- Produces: public identity `quant-deep-learning`; project distribution name `quant-deep-learning`; unchanged `ticknet.*` imports and existing `ticknet-*` console entry points; migration status describing the prior migration decision as historical and superseded.

- [ ] **Step 1: Add failing identity assertions** in `tests/test_documentation.py` for the project title, `[project].name`, and migration status. Parse `pyproject.toml` with Python's `tomllib`; assert that README and migration status identify `quant-deep-learning` as the current canonical model-research repository and that existing console-script names remain present.
- [ ] **Step 2: Run the targeted test and verify it fails** with the current migration-only status and old project metadata.

  Run: `uv run --locked --extra dev pytest tests/test_documentation.py -q`

  Expected: FAIL on the new identity assertions.
- [ ] **Step 3: Update the identity documents and package metadata.** Rewrite `README.md`, `AGENTS.md`, and `MIGRATION-STATUS.md` in English. Preserve the prior migration commit and decision as historical context, state the new owner and rationale, and update `[project].name` and its description in `pyproject.toml`. Do not rename imports or console scripts.
- [ ] **Step 4: Regenerate and inspect the lockfile.** Run `uv lock`, then confirm the lockfile records the new root package name and no unrelated dependency upgrades.
- [ ] **Step 5: Run targeted checks.** Run `uv run --locked --extra dev pytest tests/test_documentation.py -q` and `git diff --check`. The TOML metadata assertion and successful locked environment sync verify the renamed package metadata without adding a new build dependency.

  Expected: identity and link tests pass; distribution metadata names `quant-deep-learning`; all `ticknet-*` scripts remain unchanged.
- [ ] **Step 6: Commit** as `docs: reactivate quant deep learning project identity`.

### Task 2: Establish MkDocs navigation and the showcase landing page

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: Task 1 project identity and existing Markdown paths.
- Produces: one strict-buildable MkDocs Material site with `docs/index.md` as the home page and explicit navigation to project status, architecture, model catalog, next-day research, research records, operations, and reproduction notes.

- [ ] **Step 1: Add failing configuration assertions** that `mkdocs.yml` uses the Material theme, sets the canonical future site URL `https://runchengxie.github.io/quant-deep-learning/`, has an explicit navigation tree, and points to `docs/index.md`.
- [ ] **Step 2: Run the targeted test and verify it fails** because the MkDocs configuration and landing page do not exist.
- [ ] **Step 3: Add MkDocs Material to the existing development extra** in `pyproject.toml` using a bounded version range compatible with Python 3.10, then run `uv lock` and inspect the dependency diff.
- [ ] **Step 4: Create `mkdocs.yml` and `docs/index.md`.** The landing page must state project scope, current evidence stage, independent operation, artifact-based integration, and links to setup, architecture, research tracks, and study documentation. Use built-in MkDocs Material features only; do not add custom JavaScript or an interactive model demo.
- [ ] **Step 5: Build the site strictly.** Run `uv run --locked --extra dev mkdocs build --strict`.

  Expected: exit code 0, with all configured pages found and no warnings.
- [ ] **Step 6: Commit** as `docs: add MkDocs project showcase`.

### Task 3: Translate core and maintainer documentation

**Files:**
- Modify: `README.md`, `AGENTS.md`, `MIGRATION-STATUS.md`
- Modify: `docs/README.md`, `docs/index.md`, `docs/project-status.md`, `docs/model-catalog.md`, `docs/reproduction-audit.md`
- Modify: `docs/architecture/data-boundary.md`, `docs/dev/colab-cli-automation.md`, `docs/dev/development-guide.md`
- Modify: `docs/operations/development-guide.md`, `docs/operations/systemd-workflows.md`
- Modify: `docs/references/README.md`, `docs/references/agentx-paper-notes.md`, `docs/references/debang-minute-gru-notes.md`, `docs/references/deeplob-paper-notes.md`
- Modify: `legacy/notebooks/README.md`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: Task 1 identity, Task 2 MkDocs navigation, and each document's current content.
- Produces: clear English prose with the same technical meaning, runnable commands, links, metrics, evidence dates, and limitations.

- [ ] **Step 1: Extend the documentation test** to reject CJK prose outside fenced code, inline-code spans, and content between `<!-- preserved-source:start -->` and `<!-- preserved-source:end -->` markers. Keep the existing archival markers and source digest assertion unchanged.
- [ ] **Step 2: Run the targeted test and verify it fails** on the current Chinese prose.
- [ ] **Step 3: Translate the listed files in coherent sections.** Preserve code blocks, paths, command names, citation titles, numerical results, evidence status, and source links. Translate explanations and headings; do not translate source artifacts or alter archived-source marker contents.
- [ ] **Step 4: Run documentation validation.** Run `uv run --locked --extra dev pytest tests/test_documentation.py -q` and `uv run --locked --extra dev mkdocs build --strict`.

  Expected: no CJK explanatory prose, all local links resolve, and the archival source digest is unchanged.
- [ ] **Step 5: Commit** as `docs: translate core project documentation to English`.

### Task 4: Translate next-day and model research documentation

**Files:**
- Modify: `docs/nextday/cross-sectional-prediction.md`
- Modify: `docs/nextday/eventstream.md`
- Modify: `docs/nextday/h5-rolling-eventstream-roadmap.md`
- Modify: `docs/nextday/hardware-constraints-and-experiment-roadmap.md`
- Modify: `docs/nextday/multi-horizon-data-expansion-roadmap.md`
- Modify: `docs/nextday/nextday-100m-raw1000-benchmark.md`
- Modify: `docs/nextday/raw-200-end-to-end-pipeline.md`
- Modify: `docs/nextday/raw-data-expansion-roadmap.md`
- Modify: `docs/reports/multi-horizon-decision-2026-08-10/source-inspection.md`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: Task 3's English-only prose test and current research sources.
- Produces: English next-day/model documentation that retains experiment chronology and does not upgrade a pilot or candidate into a validated result.

- [ ] **Step 1: Translate each listed document** while preserving dates, sample periods, splits, metrics, units, failed experiments, commands, paths, and statements of uncertainty. Keep fenced output and literal configuration examples unchanged.
- [ ] **Step 2: Run documentation validation.** Run `uv run --locked --extra dev pytest tests/test_documentation.py -q` and `uv run --locked --extra dev mkdocs build --strict`.

  Expected: the English-only and link checks pass; all research values and evidence labels remain traceable to the source history.
- [ ] **Step 3: Commit** as `docs: translate next-day research documentation`.

### Task 5: Translate research records and preserve historical evidence

**Files:**
- Modify: all maintained Markdown under `docs/research/`
- Modify: `docs/superpowers/specs/2026-08-25-m3-eventstream-representation-design.md`
- Modify: `docs/superpowers/specs/2026-08-26-market-simulator-design.md`
- Modify: `docs/superpowers/specs/2026-08-30-l2-exchange-sequence-ordering-design.md`
- Modify: existing maintained plans under `docs/superpowers/plans/`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: the source-language test from Task 3 and the existing archival markers/digest.
- Produces: English research prose without changing frozen JSON baselines or rewriting the protected archived comparison excerpt.

- [ ] **Step 1: Extend the documentation file inventory** in `tests/test_documentation.py` to include maintained Markdown under `docs/superpowers/`, then translate research Markdown under `docs/research/` and existing human-authored specs and plans under `docs/superpowers/`. Preserve all experiment dates, identities, assumptions, sample scope, metrics, costs, failures, decisions, provenance, and uncertainty. Wrap any preserved non-English source quotation in the exact `preserved-source` markers from Task 3; keep stable identifiers intact.
- [ ] **Step 2: Keep generated and frozen records stable.** Do not edit `docs/baselines/*.json`, `docs/reports/*/*.json`, source PDFs, or the contents between `ARCHIVAL_SOURCE_START` and `ARCHIVAL_SOURCE_END` markers.
- [ ] **Step 3: Run documentation validation.** Run `uv run --locked --extra dev pytest tests/test_documentation.py -q` and `uv run --locked --extra dev mkdocs build --strict`.

  Expected: English-only check passes for maintained prose, internal links resolve, all protected archival digests remain unchanged, and MkDocs emits no warnings.
- [ ] **Step 4: Commit** as `docs: translate research records to English`.

### Task 6: Add pull-request documentation validation and Pages deployment

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_public_ci_workflow.py`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Consumes: Task 2 strict MkDocs build command and Task 3 documentation validation.
- Produces: pull-request CI that checks documentation-only changes and a Pages workflow that builds on pull requests but deploys only the default branch via `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages`.

- [ ] **Step 1: Add failing workflow assertions** for docs path filters, strict MkDocs build on pull requests, Pages permissions (`contents: read`, `pages: write`, `id-token: write` only where required), artifact upload, and deployment guarded by the default-branch push event.
- [ ] **Step 2: Run the workflow tests and verify they fail** because docs paths and a Pages workflow are not configured.
- [ ] **Step 3: Update CI path filters** so Markdown, `mkdocs.yml`, and documentation tests trigger PR quality validation. Keep model training and private data out of the workflow.
- [ ] **Step 4: Add the Pages workflow.** Build the strict MkDocs site on pull requests without deploying. On pushes to the default branch, configure Pages, upload only the generated `site/` artifact, and deploy that artifact with least-privilege permissions and concurrency protection.
- [ ] **Step 5: Run workflow and documentation checks.** Run `uv run --locked --extra dev pytest tests/test_public_ci_workflow.py tests/test_documentation.py -q`, `uv run --locked --extra dev mkdocs build --strict`, and `git diff --check`.

  Expected: workflow tests confirm PR validation and default-branch-only deployment; strict site build and local documentation checks pass.
- [ ] **Step 6: Commit** as `ci: publish MkDocs site with GitHub Pages`.

### Task 7: Verify the full documentation release and prepare the rename

**Files:**
- Modify: `README.md`
- Modify: `mkdocs.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/pages.yml`
- Test: `tests/test_documentation.py`, `tests/test_public_ci_workflow.py`

**Interfaces:**
- Consumes: completed documentation, strict site build, CI, and Pages workflow.
- Produces: a PR-ready branch with canonical `quant-deep-learning` URLs and an explicit post-merge rename checklist. This task does not perform the remote repository rename.

- [ ] **Step 1: Search maintained files for stale public identity and URLs.** Update project title, badges, workflow concurrency group, package-facing references, and site URL to `quant-deep-learning`. Preserve legacy repository names only in historical migration context.
- [ ] **Step 2: Run the complete local quality gate** with `pre-commit run --all-files` and `python scripts/check.py`, plus `uv run --locked --extra dev mkdocs build --strict`.

  Expected: all commands exit 0. Record environment, command, exit code, and any checks unavailable locally in the PR.
- [ ] **Step 3: Review the complete diff** for translation drift, altered evidence, stale links, accidental source/report inclusion, and changes outside this plan.
- [ ] **Step 4: Commit** as `docs: finalize quant deep learning public site`.
- [ ] **Step 5: After the PR is merged, rename the GitHub repository** to `quant-deep-learning`, update the local clone's `origin` URL, and verify the old URL redirects, the Pages URL loads, and default-branch deployment succeeds. Record the verification in the PR or release notes.

## Follow-on plans required by the spec

This plan deliberately delivers the public identity and documentation site as a complete, reviewable slice. Before moving or deleting implementation in other repositories, write a separate plan that inventories current `quant-platform` and `quant-research` owners, defines exact versioned artifact schemas and tests, and sequences provider-before-consumer PRs. The chip-layer study should be included in that migration plan only after its source citation, public redistribution status, and reproducibility scope are verified.
