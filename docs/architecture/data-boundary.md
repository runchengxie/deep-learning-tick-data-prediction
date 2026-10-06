# TickNet Data Boundary

This repository consumes published market data from `quant-market-data-platform`. The market-data platform owns reusable market-data assets.

## Ownership

`quant-market-data-platform` owns:

- Provider API integrations, raw landing, field normalization, and canonical schemas
- General quality checks for duplicates, missing values, ordering, and outliers
- Data versions, provenance, quality receipts, and published assets

`ticknet` in this repository owns:

- Explicit adapters from canonical L2 and event-stream data to model inputs
- Model-specific order-book normalization, windowing, and feature embeddings
- Horizon labels, leakage checks, and training/validation splits
- Tensor materialization, model training, replay, and evaluation

## Dependency direction

```text
quant-market-data-platform
  provider API -> ingest -> raw -> standardize -> canonical -> quality/provenance -> published asset

quant-deep-learning
  published asset -> adapter -> model window/features/labels -> train/evaluate
```

Cross-repository inputs enter through published files, schemas, receipts, or explicit adapters in this repository. Reusable cleaning rules belong in `quant-market-data-platform`. Normalization, windows, features, and labels that exist only for model inputs remain here.

Current examples include `ticknet.eventstream.canonical_adapter`, `ticknet.nextday.snapshot_features`, and `ticknet.nextday.snapshot_io`. These modules consume and transform platform assets. They do not redefine ownership of raw or canonical assets.

## Enforced checks

`tests/test_data_boundary.py` enforces this dependency direction in pytest. `scripts/check.py` runs the full pytest suite, so local pre-push checks and CI both exercise these rules.

The checks have two parts:

- `src/ticknet` must not directly import `market_data_platform`, `tushare`, or `rqdatac`.
- Runtime dependencies must not declare `quant-market-data-platform`, `tushare`, or `rqdatac`.

The first rule prevents model code from depending on platform implementation or provider SDKs. The second prevents provider runtime dependencies from entering this project and gradually assigning it data-ingestion responsibilities. If a separate schema-only distribution is introduced, review it independently and update the allowlist explicitly.
