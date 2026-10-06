# Shanghai Opening Contract Audit Implementation Plan

> For agentic workers: use the `superpowers:subagent-driven-development` or `superpowers:executing-plans` skill. Steps use checkbox syntax for tracking.

**Goal:** Audit opening-data contracts across stocks and dates to investigate Shanghai event clocks, pre-open order coverage, and differences in the opening ten-level quantities.

**Architecture:** Extend the existing `opening_ledger` with sample-coverage states, candidate event-lag scans, and order-level difference traces. The report CLI accepts explicit samples and writes JSON and CSV. It does not modify the matcher or put a Shanghai lag into the default configuration.

**Tech stack:** Python 3.10, dataclasses, PyArrow Parquet, pytest, argparse, csv, and json.

**Spec:** `docs/nextday/eventstream.md`, `src/ticknet/eventstream/config.py`, and the current raw L2 file layout.

## Global Constraints

- Tests use synthetic Parquet and do not depend on the 6 TB disk.
- Prices use raw L2 scaled integer cents; quantities are in shares.
- Candidate lags cover the inclusive range from `-200ms` through `200ms`, with a default step of `10ms`.
- Exclude `not_comparable` samples from the match rate over comparable samples.
- Do not change the matcher, event-stream packing format, or default Shanghai lag.

### Task 1: Coverage States and Candidate-Lag Functions

**Files:**
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] Test the distinction between a missing pre-open file, a file with no record for the stock, pre-open stock records, and closed order/trade coverage.
- [x] Test selection of the best candidate lag based on ten-level match and the fewest identity gaps; break ties by choosing the smaller absolute lag.
- [x] Confirm the tests fail before the new interface is implemented.
- [x] Implement coverage states, candidate-lag scans, and result summaries without changing the default behavior of `audit_opening_day`.
- [x] Run the tests.

### Task 2: Shanghai Sample Reports and Difference Traces

**Files:**
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Create: `scripts/audit_shanghai_contract.py`
- Test: `tests/test_shanghai_contract_cli.py`

- [x] Test that reports save each sample's date, stock, pre-open coverage, best lag, match state, per-level quantity differences, and identity gaps.
- [x] Test that a selected price level can list the remaining order quantities, trade volumes, and cancel volumes contributing to that level.
- [x] Confirm the tests fail before the CLI is implemented.
- [x] Implement explicit `--sample`, `--raw-root`, `--lag-min`, `--lag-max`, `--lag-step`, `--json-output`, and `--csv-output` arguments.
- [x] Preserve reasons for non-comparable samples instead of treating them as matcher failures.
- [x] Run tests and verify `--help`.

### Task 3: Real-Data Exploration

**Files:**
- Create: `docs/research/shanghai-opening-contract-audit-2026-08-27.md`

- [x] Run cross-year samples from Shenzhen and Shanghai; record pre-open stock coverage and the distribution of best lags.
- [x] Trace price level 777 for `600000 @ 20220615`, separating trade-boundary effects, cancel semantics, and raw-data omissions.
- [x] Record real results, the sample list, commands, and limitations in the research document.

### Task 4: Quality Gates and PR

**Files:**
- Modify: `docs/project-status.md`
- Modify: `docs/nextday/eventstream.md`

- [x] Update the status and event-stream documentation with the audit entry point and latest findings.
- [x] Run `pre-commit run --all-files`.
- [x] Run `python scripts/check.py`.
- [ ] Commit, push, create a PR, and merge it into `main`.
- [ ] After the merge, remove the local and remote branch and worktree; verify that `main` is clean and synchronized.
