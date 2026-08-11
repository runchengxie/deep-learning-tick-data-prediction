# Source inspection at `4d2c4f3`

这是 2026-08-10 多周期决策的输入证据，属于历史快照。文中描述的缺陷（单标签 manifest）此后已经修复，多周期标签侧车与 `return_end_date` 边界 purge 见 [docs/nextday/multi-horizon-data-expansion-roadmap.md](../../nextday/multi-horizon-data-expansion-roadmap.md)。

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
