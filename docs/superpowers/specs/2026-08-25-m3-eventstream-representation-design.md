# M3-inspired EventStream Representation Design

## Purpose

This change adds three optional representation mechanisms to the existing L2 event-stream Transformer so they can be evaluated as controlled ablations for the next-day cross-sectional task:

1. a causal LOB state prefix at the beginning of each sampled event window
2. fixed session-anchor context carried by that prefix alongside the existing rolling-mid coordinates
3. an optional hybrid vector-quantized event representation added as a residual to the continuous event embedding

The change does not add a matching engine or a closed-loop market simulator. Those belong to a separate subsystem and should be evaluated only after the representation experiments show repeatable value.

## Compatibility constraints

Default behavior must stay bit-for-bit compatible at the tensor-contract and model-shape level for existing experiments:

- `N_FEATURES` remains 80.
- `N_STREAMS` remains 4.
- `N_ORDER_TYPES` remains 12.
- Existing model presets keep the same parameter count when VQ is disabled.
- Existing dataset samples keep the same tuple layout and tensor shapes.
- Existing materialized datasets remain valid for configurations with all new switches disabled.
- Old checkpoints are accepted when the new configuration fields resolve to their default disabled values.

## LOB prefix

### Encoding

When `use_lob_prefix` is enabled, the first input position is a synthetic market-state token. It uses:

- stream id `0`, which is already reserved for pad/eos
- order-type id `11`, which is currently unused by real order mappings and becomes `ORDER_TYPE_LOB_PREFIX`
- the existing 80-dimensional feature vector

The book portion of the feature vector is built from the latest snapshot strictly before the sampled window start. No snapshot after the window boundary may be used. If no prior snapshot exists, the prefix uses an empty book with the previous close as the price fallback and marks the LOB as unavailable.

The synthetic prefix is a valid causal input position and predicts the first real event. With a configured sequence length `L`, prefix mode uses one prefix plus `L - 1` real event inputs and keeps one additional real event as the final next-event target. Output tensor shapes remain `(L, ...)`.

### Prefix feature semantics

Normal event rows keep their current meanings. The special prefix row reuses event-only slots for state metadata because its reserved order-type id makes the semantics unambiguous:

- feature 5: current book mid relative to the causally known fixed session anchor, in the existing `bps / 100` scale
- feature 6: current book mid relative to previous close, in the existing `bps / 100` scale
- feature 7: `1.0` when a causal session anchor is available, else `0.0`
- feature 8: `1.0` when a prior LOB snapshot is available, else `0.0`

Snapshot-derived L1/L10 spread, imbalance, depth, weighted-price and order-count features continue to occupy their existing positions.

## Session anchors

`use_session_anchors` requires `use_lob_prefix`.

The fixed session anchor is causal. At a window boundary, consider only valid trade prices and snapshot last prices already consumed before that boundary. Select the earliest observed candidate by timestamp, preferring a trade when timestamps tie. Once the first candidate has appeared, every later window resolves to the same session anchor because later observations cannot precede it.

Before any trade or snapshot price has been observed, use previous close as a numerical fallback and set the session-anchor availability flag to zero. Orders are not used as the fixed anchor because an unexecuted quote is a weaker definition of session price than a trade or exchange snapshot.

Auction-period trades and snapshots are eligible because they are part of the already observed event history. A future first trade or snapshot must never be used for an earlier window.

The existing rolling-mid event-price normalization remains unchanged. The prefix therefore gives the model a fixed day-level coordinate while every event retains the local microstructure coordinate that has already been validated in the project.

## Hybrid VQ event representation

`use_vq` adds a learned residual branch without changing the continuous event encoder.

The VQ branch consumes the five core behavioral fields:

- `dt_log`
- `price_bps`
- `qty_log`
- `side`
- `is_cancel`

A small encoder maps those values to `vq_dim`. Each encoded vector is assigned to the nearest entry in a codebook of `vq_codebook_size` vectors. The straight-through quantized vector is projected to `d_model` and added to the normal continuous, stream and order-type embeddings.

The model output exposes `vq_loss`, defined as codebook reconstruction loss plus a commitment term with beta `0.25`. `compute_loss` adds this regularizer with configurable `vq_loss_weight`. The existing four task components returned by `compute_loss_components` remain unchanged so the gradient-audit task contract stays stable.

Padding positions do not contribute to the VQ regularizer. Prefix positions are excluded from VQ because they represent state rather than trader events.

## Configuration

`EventstreamConfig` gains:

- `use_lob_prefix: bool = False`
- `use_session_anchors: bool = False`
- `use_vq: bool = False`
- `vq_codebook_size: int = 1024`
- `vq_dim: int = 64`
- `vq_loss_weight: float = 0.25`

Validation rules:

- `use_session_anchors` requires `use_lob_prefix`.
- `vq_codebook_size >= 2`.
- `vq_dim >= 2`.
- `vq_loss_weight >= 0`.

All fields are part of experiment identity and materialization identity where they affect the input tensors. The LOB/session switches are stored in the materialized contract. VQ settings do not alter materialized arrays but remain part of the training experiment signature.

## Materialized data

The array schema remains unchanged because the prefix is encoded inside the existing sequence positions and VQ is model-side.

Materialized contracts record:

- `use_lob_prefix`
- `use_session_anchors`

Source datasets are rebuilt with those switches. Compatibility checks reject a materialized dataset produced with different representation switches.

The sampling policy version becomes `seeded_fixed_window_v2` only when prefix mode is enabled because the required number of real events per fixed window changes by one. Legacy materialized manifests with disabled representation switches retain `seeded_fixed_window_v1` semantics.

## Testing

The PR must cover these behaviors with synthetic data:

- the prefix uses the latest snapshot strictly before the window and never a future snapshot
- prefix mode keeps all public sample tensor shapes unchanged
- the prefix target is the first real event and the following targets remain aligned
- the fixed session anchor is causal, does not change after first observation, and exposes availability flags
- materialized windows exactly match canonical source windows with representation switches enabled
- incompatible materialized representation switches are rejected
- VQ returns finite codes and loss, ignores prefix/pad positions, and leaves the default model parameter count unchanged
- VQ regularization participates in `compute_loss` only when enabled
- default configurations preserve legacy dataset and checkpoint behavior

## Follow-up boundary

A matching engine requires a separate simulation data contract that preserves raw order identifiers and queue priority. The current prediction pack intentionally discards those identifiers after deriving age features. That simulator work should be proposed in a separate design and PR rather than coupled to this representation experiment.
