# L2 Audit and Eventstream Task Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the raw L2 data audit and make the eventstream day-task loss weight configurable for a controlled validation experiment.

**Architecture:** Keep L2 auditing read-only with reports written outside the repository. Add one explicit `day_loss_weight` field to the eventstream configuration, include it in experiment signatures, and pass it through training and gradient-audit loss calculation without changing existing defaults.

**Tech Stack:** Python, PyTorch, pytest, existing raw L2 Parquet readers, existing Colab runner.

**Spec:** `docs/project-status.md`, `docs/research/eventstream-gradient-audit.md`, `docs/research/eventstream-label-scale.md`.

## Global Constraints

- Do not read or train on the 2026 locked interval.
- Keep the default task weights unchanged: stream 1.0, otype 0.5, reg 1.0, day 1.0.
- L2 audit commands must use explicit raw-data roots and write reports outside the repository.
- The first task-weight experiment is validation-only and must not open OOS evaluation.

### Task 1: Expand the L2 audit evidence

**Files:**
- Create: `/tmp/deep-learning-l2-audit-2026-08-28.json`
- Create: `/tmp/deep-learning-l2-audit-2026-08-28.csv`

- [x] Run the cached raw opening coverage scanner over a bounded multi-year sample.
- [x] Run explicit Shanghai opening-lag/ledger samples across available years.
- [x] Summarize coverage gaps, exact matches, and unresolved Shanghai lag behavior without modifying production configs.

### Task 2: Add configurable day-task loss weight

**Files:**
- Modify: `src/ticknet/eventstream/model.py`
- Modify: `src/ticknet/eventstream/train.py`
- Modify: `src/ticknet/eventstream/gradient_audit.py`
- Test: `tests/test_eventstream_model.py`
- Test: `tests/test_eventstream_train.py`

- [x] Add `day_loss_weight: float = 1.0` to `EventstreamConfig` and validate it is non-negative.
- [x] Pass the value to `compute_loss`, multiply only the day component, and record it in the experiment signature and audit output.
- [x] Add tests for default compatibility, zero weight, and invalid negative weight.

### Task 3: Run the controlled validation experiment

**Files:**
- Create: `/tmp/deep-learning-task-weight-dry-run.json`

- [x] Validate the new runner configuration with day weights `0.5`, `1.0`, and `2.0`.
- [ ] Run the smallest existing eventstream validation workflow with `evaluate_test=false` and separate output directories; this requires launching a Colab session.
- [ ] Compare validation Rank IC and day-gradient contribution against the existing z-label/all-position baseline after the remote run.

### Task 4: Verify and document

- [x] Run targeted tests for model/config/training behavior.
- [x] Run Ruff and ty.
- [ ] Record the audit and experiment outcome in the research log only after the commands produce complete artifacts.
