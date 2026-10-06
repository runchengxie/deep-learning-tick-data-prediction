# Opening Coverage Inventory Implementation Plan

> For agentic workers: use the `superpowers:subagent-driven-development` or `superpowers:executing-plans` skill. Steps use checkbox syntax for tracking.

**Goal:** Build a raw L2 pre-open coverage inventory across trading days, stocks, and data batches. Provide reusable report inputs for Shanghai lag strata and remaining failed samples.

**Architecture:** Add a standalone coverage scanner that reads each daily `order_preopen` file, aggregates pre-open orders by stock, and joins them with order, trade, and snapshot data. Reports include JSON, CSV, and summaries by year, month, market, and batch. Keep the production matcher and default Shanghai lag unchanged.

**Tech stack:** Python 3.10, PyArrow Parquet, dataclasses, argparse, csv, json, and pytest.

**Spec:** `docs/research/shanghai-opening-contract-audit-2026-08-27.md` and the raw L2 file layout.

## Global Constraints

- Real-data scans read raw L2 only; do not commit generated data products to Git.
- Synthetic tests use temporary Parquet files and do not depend on the external drive.
- Define an opening trade as a trade row with `time_ms <= 0`.
- Distinguish file presence, stock presence, and whether opening trades are present.
- Do not change the matcher, event-stream packing format, or default Shanghai lag.

### Task 1: Coverage Functions and Parquet Scanner

**Files:**
- Create: `src/ticknet/simulator/coverage.py`
- Test: `tests/test_opening_coverage.py`

- [ ] First write tests proving that pre-open files, stock records, the three related file types, and `time_ms <= 0` trade counts are tracked independently.
- [ ] Run focused pytest and confirm the tests fail because the module does not exist.
- [ ] Implement `CoverageRow`, `scan_preopen_coverage`, and `summarize_coverage` with batched Parquet reads; support daily and monthly snapshot files.
- [ ] Run focused pytest and confirm synthetic Parquet tests pass.
- [ ] Commit as `feat: add raw opening coverage scanner`.

### Task 2: Coverage Report CLI and Stratified Summaries

**Files:**
- Create: `scripts/audit_opening_coverage.py`
- Modify: `tests/test_opening_coverage.py`
- Modify: `docs/documentation-index.md`

- [ ] First write CLI tests for JSON, CSV, and summaries by year, month, market, and batch; confirm they fail.
- [ ] Implement `--raw-root`, `--json-output`, `--csv-output`, and optional `--limit-days`.
- [ ] Run focused pytest and `--help`.
- [ ] Commit as `feat: add opening coverage report CLI`.

### Task 3: Real-Data Scan and Research Record

**Files:**
- Create: `docs/research/opening-coverage-inventory-2026-08-27.md`
- Modify: `docs/project-status.md`

- [ ] Generate a report from the 6 TB raw root into `/tmp`; do not add CSV or JSON outputs to Git.
- [ ] Record daily file totals, stock-days, missing files, missing stocks, opening-trade gaps, pre-open order counts, and lag-input ranges for each stratum.
- [ ] State clearly that this is a coverage audit and does not prove that the ten-level identity ledger is complete.
- [ ] Commit as `docs: record raw opening coverage inventory`.

### Task 4: Quality Gates and PR Completion

- [ ] Run focused pytest, `pre-commit run --all-files`, `python scripts/check.py`, and `git diff --check`.
- [ ] Push `agent/opening-coverage-inventory`, create a PR, and merge it into `main`.
- [ ] Remove the merged local and remote branches and worktree; prune and confirm that `main` is clean and synchronized.
