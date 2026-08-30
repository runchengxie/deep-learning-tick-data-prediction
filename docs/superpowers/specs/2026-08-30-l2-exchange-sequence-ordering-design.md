# L2 Exchange Sequence Ordering Design

## Goal

Use exchange sequence metadata when it is present in real L2 order data, while preserving explicit fallback semantics when exact ordering cannot be recovered.

## Constraints

- `time_ms` remains the cross-channel time coordinate.
- A sequence is used to reorder a same-timestamp bucket only when every non-snapshot event in that bucket has a sequence and the bucket belongs to at most one observed channel.
- A multi-channel same-timestamp bucket preserves source row order; the simulator must not invent a cross-channel exchange total order.
- Snapshots remain after order/cancel events at the same timestamp.
- Existing files without sequence columns keep timestamp/source-order behavior.

## Event metadata

`SimulatorEvent` gains optional `channel`, `sequence`, and `source_index`. `SimulatorPack` gains an `ordering_provenance` mapping. This metadata is simulator-only and does not alter model feature contracts.

## Column detection

The loader recognizes common aliases such as `ChannelNo`, `Channel`, `ApplSeqNum`, `BizIndex`, `SeqNum`, and `Sequence`. Detection is schema-driven; optional columns are read only when present.

## Provenance modes

- `timestamp_then_channel_sequence`: sequence is available and only one channel is observed in the loaded event stream.
- `timestamp_then_source_order_cross_channel`: sequence exists, but multiple channels are observed so cross-channel source order is retained.
- `timestamp_fallback`: no usable exchange sequence was loaded.

Every mode records `cross_channel_total_order=false`.

## Replay behavior

`load_day_pack` assigns source indices in input order, combines pre-open and continuous orders, then uses the ordering helper before inserting snapshots. The helper is also used by synthetic `build_simulator_pack`, so test and real-data packs share deterministic ordering semantics.

## Testing

Pure tests cover alias detection, same-channel sequence ordering, cross-channel source-order fallback, snapshot ordering, and no-sequence compatibility. Real-data synthetic Parquet tests verify optional sequence columns survive the loader and appear in pack provenance.
