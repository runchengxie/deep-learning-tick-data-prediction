# L2 Exchange Sequence Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve optional exchange channel/sequence metadata and use it safely for same-timestamp simulator ordering.

**Architecture:** Add a pure ordering helper, extend simulator event metadata, and make real-data loading schema-aware for optional sequence columns. Cross-channel order falls back to source order and is explicitly marked non-total.

**Tech Stack:** Python, dataclasses, PyArrow, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-l2-exchange-sequence-ordering-design.md`

## Global Constraints

- Time remains the cross-channel coordinate.
- Do not invent a cross-channel exchange total order.
- Files without sequence metadata keep compatible behavior.
- Snapshot events remain after order/cancel events at the same timestamp.

---

### Task 1: Add pure ordering semantics

**Files:**
- Create: `src/ticknet/simulator/ordering.py`
- Modify: `src/ticknet/simulator/pack.py`
- Test: `tests/test_simulator_ordering.py`

- [ ] Write failing tests for alias detection, single-channel sequence ordering, cross-channel fallback, and no-sequence compatibility.
- [ ] Run tests and confirm the missing ordering module fails first.
- [ ] Add optional event ordering metadata and pure ordering helpers.
- [ ] Use the helper in synthetic pack construction.
- [ ] Run focused tests.

### Task 2: Preserve optional ordering metadata from real Parquet

**Files:**
- Modify: `src/ticknet/simulator/realdata.py`
- Modify: `tests/test_realdata_correctness.py`

- [ ] Add a synthetic Parquet test with `ChannelNo` and `ApplSeqNum` in reverse source order at one timestamp.
- [ ] Make `_read_order_events` inspect Parquet schema and project optional ordering columns when present.
- [ ] Assign deterministic source indices and use the shared event sorter.
- [ ] Populate `SimulatorPack.ordering_provenance`.
- [ ] Run real-data focused tests where PyArrow is available.

### Task 3: Document and verify

**Files:**
- Modify: `docs/research/historical-data-eligibility-2026-08-27.md`

- [ ] Document that exchange sequence is used when available and cross-channel order remains source-order fallback.
- [ ] Run pure ordering tests.
- [ ] Run the simulator/realdata test slice where dependencies are available.
- [ ] Run Ruff/format/ty on changed files where available.
