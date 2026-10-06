# M3-inspired EventStream Representation Implementation Plan

Goal: Add optional LOB prefixes, fixed causal session anchors, and a Hybrid VQ representation to the existing event-stream Transformer without changing legacy defaults. Include these options in experiment identity and materialization contracts.

Architecture constraints: Retain the 80-dimensional event tensor and existing dataset tuple contract. A prefix occupies one input position in the sequence. VQ is a model-side residual branch on continuous event embeddings. A matching engine and closed-loop market simulation are out of scope for this PR.

Tech stack: Python 3.10+, NumPy, PyTorch, PyArrow, pytest, and YAML.

Design: `docs/superpowers/specs/2026-08-25-m3-eventstream-representation-design.md`

## Global Constraints

- With all new options disabled, retain legacy sample shapes, model parameter counts, and checkpoint behavior.
- Prefixes and session anchors must not read snapshots or trades after the window boundary.
- Keep `N_FEATURES=80`, `N_STREAMS=4`, and `N_ORDER_TYPES=12` unchanged.
- Keep the materialized array schema unchanged.
- Do not read the locked 2026 data partition or add performance claims based on real data.

## Task 1: LOB Prefix and Fixed Session Anchors

Files:

- `src/ticknet/eventstream/dataset.py`
- `tests/test_eventstream_m3_repr.py`

Implementation steps:

- [x] First write failing tests for prefix tensor shape, target alignment, strict use of pre-boundary snapshots, and anchor causality.
- [x] Confirm RED: the previous implementation lacks `use_lob_prefix`, `ORDER_TYPE_LOB_PREFIX`, and the prefix construction interface.
- [x] Implement `ORDER_TYPE_LOB_PREFIX = 11` without changing the public tensor shape.
- [x] Build the prefix from the last snapshot before the window begins; do not read future snapshots.
- [x] Choose the fixed session anchor as the earliest price from a trade or snapshot last observed before the boundary. Prefer a trade when timestamps tie. Once selected, the anchor remains unchanged for later windows.
- [x] Before an anchor appears, use the prior close as a numeric fallback and set its availability flag to 0.

## Task 2: Configuration and Materialization Contract

Files:

- `src/ticknet/eventstream/train.py`
- `src/ticknet/eventstream/materialized.py`
- `tests/test_eventstream_materialized.py`
- `configs/eventstream.yaml`

Implementation steps:

- [x] Add `use_lob_prefix`, `use_session_anchors`, `use_vq`, `vq_codebook_size`, `vq_dim`, and `vq_loss_weight` to `EventstreamConfig`.
- [x] Require LOB prefixes when `use_session_anchors=True`.
- [x] Pass representation options consistently to raw train, validation, OOS, and monitor datasets.
- [x] Record prefix and anchor options in the materialization contract. Use `seeded_fixed_window_v2` with prefixes; retain v1 for the legacy contract.
- [x] Add a tensor-by-tensor consistency test between source and materialized samples in prefix mode.
- [x] Update the example YAML.

## Task 3: Hybrid VQ

Files:

- `src/ticknet/eventstream/model.py`
- `src/ticknet/eventstream/train.py`
- `tests/test_eventstream_vq.py`

Implementation steps:

- [x] First write failing tests for parameter-shape compatibility with VQ disabled, code outputs, prefix/padding masks, and VQ loss weight.
- [x] Confirm RED: the existing model constructor and loss interface do not accept VQ parameters.
- [x] Encode core event behavior with `[dt_log, price_bps, qty_log, side, is_cancel]`.
- [x] Implement quantization with a nearest-neighbor codebook, straight-through estimator, codebook loss, and commitment loss.
- [x] Project quantized values to `d_model` and add them as a residual to the continuous embedding. Padding and prefixes with `sid==0` do not participate in VQ.
- [x] Keep the four-term task contract in `compute_loss_components`; add VQ regularization only in `compute_loss`.

## Task 4: Compatibility Chain

Files:

- `src/ticknet/eventstream/train.py`
- Frozen embeddings, fixed-window caches, and other checkpoint consumers
- Related tests

Implementation steps:

- [x] Normalize legacy checkpoints that lack the new fields to defaults with all new options disabled.
- [x] Inspect every `build_eventstream_model` checkpoint consumer and ensure VQ checkpoints can reconstruct the model from the experiment signature.
- [x] Check the input contract between fixed end-of-day windows and prefix mode; bind representation identity where needed.
- [x] Retain default compatibility paths for old checkpoints, materialized manifests, and close caches.

## Task 5: Documentation and Verification

Files:

- `docs/nextday/eventstream.md`
- `docs/model-catalog.md`
- PR description

Implementation steps:

- [x] Document the new configuration, causal boundaries, VQ role, and experiment interpretation.
- [x] State clearly that the new representation has not been performance-tested on real rolling windows.
- [x] Run PR CI and confirm Python 3.10, Python 3.12, Ruff, formatting, ty, pytest, coverage, and dependency audits pass.
- [x] Review the final diff and confirm it contains no matching engine, simulator raw-ID contract, locked 2026 data access, or unverified performance claims.
- [x] Update the draft PR for review and include RED/GREEN verification evidence.
