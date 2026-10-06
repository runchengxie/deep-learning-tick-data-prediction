# Historical Data Eligibility Implementation Plan

> For agentic workers: use the `superpowers:subagent-driven-development` or `superpowers:executing-plans` skill. Steps use checkbox syntax for tracking.

**Goal:** Build an eligibility manifest for historical raw L2 stock-days from 2021 through 2025, freeze 2026, and distinguish the primary Shenzhen dataset from Shanghai research data.

**Architecture:** Add a pure data-eligibility module on top of the existing `CoverageRow`. Eligibility depends only on file coverage, stock coverage, pre-open orders, and the trading-day range. Do not encode an unverified Shanghai lag into the rules. The CLI outputs JSON, CSV, and stratified counts.

**Tech stack:** Python 3.10, dataclasses, argparse, csv, json, and pytest.

**Spec:** `docs/research/opening-coverage-inventory-2026-08-27.md`.

## Global Constraints

- Data from 2026 is excluded from the historical eligibility manifest.
- The primary Shenzhen dataset requires all three file types, stock coverage in all three, and pre-open orders.
- Shanghai research data uses the same file-completeness rules but must be marked as having an uncalibrated lag.
- Do not change the matcher, event-stream packing format, or default Shanghai lag.
- Synthetic tests must not depend on the 6 TB disk.

### Task 1: Eligibility Classification

**Files:**
- Create: `src/ticknet/simulator/eligibility.py`
- Test: `tests/test_historical_eligibility.py`

- [ ] Write tests for the primary Shenzhen dataset, Shanghai research data, missing files, missing stocks, exclusion of 2026, and days without pre-open orders.
- [ ] Run the tests and confirm they fail because the module does not exist.
- [ ] Implement `EligibilityRow`, `classify_coverage`, and `summarize_eligibility`.
- [ ] Run the focused pytest suite.
- [ ] Commit as `feat: add historical data eligibility rules`.

### Task 2: Eligibility Manifest CLI

**Files:**
- Create: `scripts/build_historical_data_manifest.py`
- Modify: `tests/test_historical_eligibility.py`
- Modify: `docs/documentation-index.md`

- [ ] Write tests for JSON, CSV, and stratified summaries, and confirm they fail.
- [ ] Implement `--raw-root`, `--json-output`, `--csv-output`, `--start-year`, `--end-year`, and `--limit-days`.
- [ ] Run focused pytest and `--help`.
- [ ] Commit as `feat: add historical data manifest CLI`.

### Task 3: Research Record and Quality Gates

**Files:**
- Create: `docs/research/historical-data-eligibility-2026-08-27.md`
- Modify: `docs/project-status.md`

- [ ] Document the 2021–2025 eligibility definition and the boundaries between the primary Shenzhen dataset and Shanghai research data.
- [ ] Record general context supported by public exchange interface documentation and gaps specific to this dataset.
- [ ] Run focused pytest, pre-commit, `scripts/check.py`, and `git diff --check`.
- [ ] Push, create and merge a PR into `main`, remove the task branch and worktree, and verify that `main` is synchronized.
