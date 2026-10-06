# Opening Ledger Audit Implementation Plan

> For agentic workers: use the `superpowers:subagent-driven-development` or `superpowers:executing-plans` skill. Steps use checkbox syntax for tracking.

**Goal:** Build a reproducible audit of pre-open order identities across stocks and trading days, and test whether pre-open orders, trades, and cancels can reconstruct the ten-level book before the open.

**Architecture:** Add a pure audit module under `simulator`. Start with pre-open orders to calculate remaining quantities, apply pre-open trades and cancels, aggregate the best ten price levels, and compare them with the first complete continuous-session snapshot. The CLI reads explicit samples from raw L2 data and writes a JSON summary; it does not connect an unverified ledger directly to matching replay.

**Tech stack:** Python 3.10, dataclasses, PyArrow Parquet, pytest, and argparse.

**Spec:** The raw L2 file contract in `docs/nextday/eventstream.md` and `src/ticknet/eventstream/config.py`.

## Global Constraints

- Tests use synthetic data only; they do not depend on the 6 TB disk or external services.
- Prices are represented as raw L2 scaled integer cents; quantities are in shares.
- The pre-open ledger uses orders, trades, and cancels with `time_ms < 0`. Continuous-session events at `time_ms == 0` remain for later replay.
- Audit results distinguish exact matches, non-comparable samples, unknown trade IDs, unknown cancel IDs, and quantity mismatches.
- Do not change the structure returned by `day_input_files()` or the event-stream packing format.

### Task 1: Pure Order-Level Pre-Open Ledger

**Files:**
- Create: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] Test that pre-open buy and sell orders produce the correct remaining quantity and top ten levels after trades and partial cancels.
- [x] Test that unknown trade IDs, unknown cancel IDs, and trades exceeding the remaining quantity are reported rather than silently ignored.
- [x] Run the tests and confirm the expected failure while the module is absent.
- [x] Implement `audit_opening_ledger(orders, trades, cancels, snapshot_levels)` and return an immutable audit result.
- [x] Run the tests and verify that fields retain the raw L2 units.

### Task 2: Read Real Parquet Samples

**Files:**
- Modify: `src/ticknet/simulator/opening_ledger.py`
- Test: `tests/test_opening_ledger.py`

- [x] Test reading a specified stock-day from synthetic `order_preopen`, `trades`, and `snapshot` Parquet files, including selection of the first complete non-negative snapshot.
- [x] Confirm the read-interface tests fail before implementation.
- [x] Implement `audit_opening_day(day, ticker, raw_root)` using `day_preopen_file()`, `day_input_files()`, and the existing snapshot parser.
- [x] Deduct pre-open trades from their matching orders using `BuyID` and `SellID`; deduct pre-open cancels using `OrderID`; retain counts and quantities for unknown identities.
- [x] Run tests for stock and date filtering, price units, and incomplete snapshot levels.

### Task 3: Cross-Stock and Cross-Date CLI

**Files:**
- Create: `scripts/audit_opening_ledger.py`
- Test: `tests/test_opening_ledger_cli.py`

- [x] Test multiple `--sample YYYYMMDD:TICKER` inputs, JSON output, and summaries for exact, mismatched, non-comparable, and identity-gap outcomes.
- [x] Confirm the tests fail before the CLI exists.
- [x] Implement explicit sample selection and `--raw-root`; avoid default scans across terabytes of data.
- [x] Report per-sample differences across ten price levels, unknown IDs, and order coverage; calculate aggregate rates only over comparable samples.
- [x] Run real audits on Shenzhen samples from 2021, 2023, and 2025, plus at least one Shanghai sample.

### Task 4: Documentation, Quality Gates, and PR

**Files:**
- Modify: `docs/project-status.md`
- Modify: `docs/nextday/eventstream.md`

- [x] Record the real-audit command and results, including sample scope, data dates, and limitations.
- [x] Run `pre-commit run --all-files`.
- [x] Run `python scripts/check.py`.
- [ ] Commit, push, create a PR, and merge it into `main`.
- [ ] After the merge, remove the branch and worktree; verify that `main` is clean and synchronized with `origin/main`.
