# M3-inspired EventStream Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional LOB-prefix, causal session-anchor, and hybrid VQ representations to the existing event-stream Transformer without changing legacy defaults.

**Architecture:** Keep the 80-dimensional event tensor and public dataset tuple contract unchanged. Prefix mode inserts a synthetic state token into the existing sequence positions, while VQ remains a model-side residual branch. Materialized datasets record only the representation switches that alter sampled tensors.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyArrow, pytest, YAML.

**Spec:** `docs/superpowers/specs/2026-08-25-m3-eventstream-representation-design.md`

## Global Constraints

- Existing configurations with all new switches disabled must preserve legacy sample shapes and model parameter counts.
- No future snapshot or trade may be used to construct a prefix or session anchor.
- `N_FEATURES=80`, `N_STREAMS=4`, and `N_ORDER_TYPES=12` remain unchanged.
- The materialized array schema remains unchanged.
- Matching-engine and simulator work is outside this PR.

---

### Task 1: Add causal LOB prefix and session anchors

**Files:**
- Modify: `src/ticknet/eventstream/dataset.py`
- Modify: `tests/test_eventstream.py`

**Interfaces:**
- Produces: `ORDER_TYPE_LOB_PREFIX: int = 11`
- Produces: `L2WindowDataset(..., use_lob_prefix: bool = False, use_session_anchors: bool = False)`
- Produces: `_session_open_reference(order, trade, snap, positions, prev_close_cent) -> tuple[float, bool]`
- Produces: `_build_lob_prefix_features(snap, consumed_snapshots, positions, prev_close_cent, use_session_anchors) -> np.ndarray`

- [ ] **Step 1: Write failing prefix tests**

Add tests that create a window after the first snapshot and assert that prefix mode keeps `x`, `sid`, `oid`, target and mask shapes unchanged, places `oid=11` at position zero, predicts the first real event from the prefix, and uses the latest strictly prior snapshot.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_eventstream.py -q`

Expected: FAIL because `use_lob_prefix`, `ORDER_TYPE_LOB_PREFIX`, and prefix construction do not exist.

- [ ] **Step 3: Implement minimal prefix sampling**

Update `_resolve_window_entries` so prefix mode needs `seq_len` real events rather than `seq_len + 1`. Build the synthetic prefix from the snapshot position strictly before `start`. Keep all returned tensor shapes unchanged and shift real events by one position.

- [ ] **Step 4: Write failing causal-anchor tests**

Add cases where the first trade occurs before and after the sampled boundary. Assert that the earlier window cannot use a future first trade and that the availability flag changes only after the anchor becomes observable.

- [ ] **Step 5: Run the focused test and verify RED**

Run: `pytest tests/test_eventstream.py -q`

Expected: FAIL on missing session-anchor behavior.

- [ ] **Step 6: Implement causal session anchors**

Use already-consumed stream positions to select the earliest observed valid trade, snapshot, or non-cancel order price. Encode mid/open and mid/previous-close offsets in prefix feature slots 5 and 6, plus availability flags in 7 and 8.

- [ ] **Step 7: Run the focused test and verify GREEN**

Run: `pytest tests/test_eventstream.py -q`

Expected: PASS.

### Task 2: Add configuration and materialized-data identity

**Files:**
- Modify: `src/ticknet/eventstream/train.py`
- Modify: `src/ticknet/eventstream/materialized.py`
- Modify: `tests/test_eventstream_materialized.py`
- Modify: `configs/eventstream.yaml`

**Interfaces:**
- Produces config fields: `use_lob_prefix`, `use_session_anchors`, `use_vq`, `vq_codebook_size`, `vq_dim`, `vq_loss_weight`
- Materialized contract keys: `use_lob_prefix`, `use_session_anchors`

- [ ] **Step 1: Write failing config and materialization tests**

Add tests that reject `use_session_anchors=True` when prefix mode is disabled, verify representation switches are persisted in materialized contracts, verify source and materialized samples are identical with prefix mode enabled, and reject a training config whose representation switches differ from the materialized contract.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_eventstream_materialized.py tests/test_eventstream.py -q`

Expected: FAIL because the new config and contract fields do not exist.

- [ ] **Step 3: Implement configuration plumbing**

Add the six fields to `EventstreamConfig`, validation, `to_dict` identity, raw dataset construction, monitor datasets, and model construction call sites. Pass only prefix/session switches into `L2WindowDataset`.

- [ ] **Step 4: Implement materialized identity**

Record prefix/session switches in `_materialization_contract`, pass them through `build_source_datasets`, and check them in `assert_materialized_compatible`. Preserve legacy `seeded_fixed_window_v1` when prefix mode is off and use `seeded_fixed_window_v2` when it is on.

- [ ] **Step 5: Update example config**

Add all new fields with disabled/default values to `configs/eventstream.yaml`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `pytest tests/test_eventstream_materialized.py tests/test_eventstream.py -q`

Expected: PASS.

### Task 3: Add optional hybrid VQ residual branch

**Files:**
- Modify: `src/ticknet/eventstream/model.py`
- Create: `tests/test_eventstream_vq.py`
- Modify: `src/ticknet/eventstream/train.py`

**Interfaces:**
- Produces: `VectorQuantizer(nn.Module)`
- Extends: `L2FoundationModel(..., use_vq=False, vq_codebook_size=1024, vq_dim=64)`
- Extends: `build_eventstream_model(name, *, use_vq=False, vq_codebook_size=1024, vq_dim=64)`
- Extends: `compute_loss(..., vq_loss_weight=0.0)`

- [ ] **Step 1: Write failing VQ tests**

Create tests that assert the default smoke model has the legacy parameter count/state shapes, a VQ-enabled model returns finite `vq_loss` and integer `vq_codes`, prefix/pad positions do not contribute to VQ loss, and changing `vq_loss_weight` changes total loss by the expected regularizer amount while leaving the four task components unchanged.

- [ ] **Step 2: Run VQ tests and verify RED**

Run: `pytest tests/test_eventstream_vq.py -q`

Expected: FAIL because VQ classes and arguments do not exist.

- [ ] **Step 3: Implement minimal VectorQuantizer**

Encode event core features `[0,1,2,3,4]`, compute nearest codebook entries by squared Euclidean distance, use straight-through quantization, and return codebook plus commitment loss with beta `0.25`.

- [ ] **Step 4: Add VQ residual to the model**

Project quantized vectors to `d_model` and add them to the normal embedding. Exclude positions with `sid==0` so padding and the synthetic prefix are ignored.

- [ ] **Step 5: Add VQ regularization to training loss**

Keep `compute_loss_components` unchanged. Add `out['vq_loss'] * vq_loss_weight` only in `compute_loss`, expose a `vq` metric, and pass the configured weight from training.

- [ ] **Step 6: Run VQ tests and verify GREEN**

Run: `pytest tests/test_eventstream_vq.py -q`

Expected: PASS.

### Task 4: Preserve checkpoint compatibility and experiment identity

**Files:**
- Modify: `src/ticknet/eventstream/train.py`
- Modify: relevant checkpoint/config tests under `tests/`

**Interfaces:**
- Extends: `_checkpoint_matches_experiment` default normalization for all new fields

- [ ] **Step 1: Write failing compatibility test**

Add a test with a legacy experiment dictionary that lacks the new fields and assert it matches a current default-disabled expected signature.

- [ ] **Step 2: Run the focused test and verify RED**

Run the checkpoint/config-focused pytest target containing the new test.

Expected: FAIL because missing fields are not normalized.

- [ ] **Step 3: Normalize legacy experiment dictionaries**

Set defaults for `use_lob_prefix=False`, `use_session_anchors=False`, `use_vq=False`, `vq_codebook_size=1024`, `vq_dim=64`, and `vq_loss_weight=0.25` before identity comparison.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same target.

Expected: PASS.

### Task 5: Documentation and full verification

**Files:**
- Modify: `docs/nextday/eventstream.md`
- Modify: `docs/model-catalog.md`

**Interfaces:**
- Documents the new flags, causal boundary, VQ role, and explicit statement that no real-data performance claim exists yet.

- [ ] **Step 1: Update user-facing documentation**

Document the representation switches, the reserved prefix order-type id, the causal session-anchor rule, materialization identity, and the experiment interpretation. State that the feature is an unvalidated candidate until real rolling-window experiments are run.

- [ ] **Step 2: Run repository quality gates**

Run:

```bash
pre-commit run --all-files
python scripts/check.py
```

Expected: both commands pass.

- [ ] **Step 3: Review the diff for scope and compatibility**

Confirm no matching-engine code, no simulator data-contract change, no raw-ID retention change, no locked 2026 data access, and no numerical performance claims were added.

- [ ] **Step 4: Commit and open the PR**

Use a feature commit message such as `feat: add M3-inspired eventstream representations`, then open a PR against `main` with the focused and full verification results.