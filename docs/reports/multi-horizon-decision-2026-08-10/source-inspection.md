# Source inspection at `4d2c4f3`

- `snapshot_targets.py` defines the raw-200 target as the next trading day's open-to-close return minus the benchmark return for that day.
- `dataset.py` and `io.py` use manifest format version 1 with one `label_date`, `label`, and `target_return` per sample.
- `splits.py` assigns a sample only when `trading_date` and `label_date` are inside the same split.
- `metrics.py` computes daily cross-sectional Rank IC and an uncosted long-short spread from one target-return vector.
- A multi-horizon extension therefore needs an explicit `return_end_date` for each horizon and boundary purging against that date.

Inspected files:

- `src/ticknet/nextday/snapshot_targets.py`
- `src/ticknet/nextday/dataset.py`
- `src/ticknet/nextday/splits.py`
- `src/ticknet/nextday/metrics.py`
- `src/ticknet/nextday/io.py`
