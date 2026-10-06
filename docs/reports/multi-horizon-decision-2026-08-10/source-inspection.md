# Source inspection for the multi-horizon decision

This is the historical evidence snapshot used for the 2026-08-10 multi-horizon decision. The single-label manifest limitation described here has since been fixed. The multi-horizon label sidecar and `return_end_date` split purge are documented in [the data expansion roadmap](../../nextday/multi-horizon-data-expansion-roadmap.md).

## Inspected behavior

- `snapshot_targets.py` creates next-trading-day open-to-close returns in excess of the benchmark.
- `dataset.py` and `io.py` used manifest v1, with one `label_date`, `label`, and `target_return` per sample.
- A sample entered a split only when its `trading_date` and `label_date` belonged to that same split.
- Metrics included daily cross-sectional Rank IC and an uncosted long-short spread based on the same return vector.
- Multi-horizon labels require an explicit `return_end_date` for each horizon and purging at split boundaries.

## Files inspected

The historical review covered the target-generation, dataset, I/O, split, and metric implementations available at commit `4d2c4f3`. This note records the state observed on 2026-08-10; it is not a description of the current implementation.
