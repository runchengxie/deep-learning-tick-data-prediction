# Raw Order-Book Data Expansion Roadmap

## Decision

The five-year raw datasets, multi-horizon labels, raw-1000 Top-100 run, and controlled four-cell, three-seed capacity matrix are complete. The latest matrix selected `1M/raw-200` as the only candidate; further expansion of model capacity and event windows is paused. See the [multi-horizon data expansion roadmap](multi-horizon-data-expansion-roadmap.md) for the full results.

In the earlier 2024 Top-100 raw-200 pilot, the 1,033,383-parameter model achieved best validation Rank IC values of `0.02145`, `0.02054`, and `0.01893` for seeds 0, 1, and 2. The mean was `0.02031`, with a sample standard deviation of `0.00127`. That pilot used a different stock sample from the later fixed Top-100 matrix and is retained only as a stage record.

The formal working set covers 2021–2025, a dynamic Top-400 universe, and the final 200 valid snapshot events per stock-day. The split in `configs/nextday.yaml` uses 2021–2023 for training, 2024 for validation, and keeps 2025 locked. Data generation must not read or calculate model metrics for 2025.

## Available resources (inventory on 2026-08-09)

| Resource | Inventory |
|---|---:|
| 2021 monthly snapshot files | 12 files, 157.48 GiB |
| 2022 monthly snapshot files | 12 files, 174.18 GiB |
| 2023 monthly snapshot files | 12 files, 168.74 GiB |
| 2024 monthly snapshot files | 12 files, 182.16 GiB |
| 2025 monthly snapshot files | 12 files, 201.58 GiB |
| Canonical monthly files | All 60 present, 884.14 GiB total |
| Free space on remote NVMe | 610 GiB |
| Free space on raw-data disk | 761 GiB |
| Google Drive plan | 200 GB, upgraded on 2026-08-10 |

The expected maximum is about 480,000 stock-days. The compact float16 raw-200 working set is approximately 7–8 GiB. It can be written to remote NVMe first and uploaded to Drive after audit. Raw Parquet remains on the data disk.

## Stage 0: Freeze the data contract

The formal configuration is `configs/nextday-raw.yaml`. Keep these fields fixed while generating the dataset:

- `start_date: 2021-01-01` and `end_date: 2025-12-31`
- `top_n: 400`
- `scan_start_time_ms: 18000000` and `signal_time_ms: 19500000`
- `chunks_per_sample: 2` and `chunk_size: 100`
- `min_valid_events: 200`
- `storage_dtype: float16`
- The dynamic universe uses only the prior 20 days of turnover, with at least 15 valid observations.

Any change requires a new output directory and dataset fingerprint. Never overwrite the formal working set.

## Stage 1: Single-month Top-400 preflight

Run the isolated configuration first:

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-200-preflight.yaml
```

Output is written to `data/nextday-raw-200-preflight-202101-top400/`. Acceptance criteria:

- `manifest.json`, `data-audit.json`, and every shard are present.
- Minimum, median, and maximum universe sizes are audited; the median is 400 stocks.
- `written_samples > 0`, with no duplicate stock-days.
- Every shard SHA-256 matches the manifest.
- `last_event_timestamp <= signal_timestamp`.
- `NextDayShardDataset` reads samples with shape `2 × 100 × 40`.

The preflight validates the engineering path only; it does not support a model-performance conclusion.

## Stage 2: Generate five years of raw-200

After the preflight passes, run:

```bash
mkdir -p logs
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw.yaml \
  > logs/prepare-nextday-raw-200.log 2>&1
```

The destination is `data/nextday-raw-200/`. The 2024 Top-100 pilot took about 35 minutes. Budget 8–14 hours for five years of Top-400 data because the stock universe is larger. Do not start another job against the same output directory.

`raw_snapshot` atomically replaces individual shards, but writes the complete manifest only after all samples finish. Monthly resume is not supported. Shards left by an interrupted process do not constitute a complete dataset; check that no job is still running before restarting. Raw-1000 was generated into a separate directory. Before generating another large working set, implement monthly materialization and manifest merging to avoid rescanning data.

Check progress with:

```bash
pgrep -af "ticknet-nextday-prepare-snapshot"
du -sh data/nextday-raw-200
find data/nextday-raw-200/shards -maxdepth 1 -name "part-*.npy" | wc -l
tail -n 50 logs/prepare-nextday-raw-200.log
```

## Stage 3: Integrity and coverage audit

After generation:

1. Verify the manifest fingerprint and every shard SHA-256.
2. Summarize sample counts and stock coverage by year, month, and trading day.
3. Summarize `missing_snapshot`, `insufficient_events`, `invalid_lob_rows`, and daily-backup fallback months.
4. Build train, validation, and test datasets using `configs/nextday.yaml`; confirm both trading and label dates fall within their respective intervals.
5. Sample-check normalized value ranges, class proportions, and continuous targets.
6. Save an aggregate-only audit summary. Do not commit real data or the full manifest to Git.

Minimum acceptance criteria:

- Every month with targets from 2021 through 2025 contains samples.
- Repeated integrity checks produce the same fingerprint.
- Train, validation, and test dates do not overlap, and cross-boundary labels are purged.
- Each day has enough eligible stocks to calculate cross-sectional Top-400 metrics.
- Fallback months and file counts are recorded.

## Stage 4: Upload to Drive

After the local audit passes, upload the compact working set:

```bash
rclone copy data/nextday-raw-200 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200 \
  --checksum --transfers 4 --checkers 8 --progress
```

Verify the upload with `rclone check`. Long-term Drive storage should contain only the current working set, configuration, checkpoints, and results. The raw-1000 Top-100 set was generated and evaluated separately; raw-500 was not started. Rotate large working sets to avoid retaining both formal and temporary copies.

## Stage 5: Data-first model comparison (complete)

With data, dates, and training budget fixed, compare:

1. An aggregated Logistic or HGB baseline.
2. The 86,775-parameter raw-200 model.
3. The 1,033,383-parameter raw-200 model.

Use at least seeds 0, 1, and 2 on validation. Report the mean, sample standard deviation, monthly IC, and proportion of positive-IC months. Keep 2025 locked until the model, hyperparameters, and seed list are frozen. Do not select a model using test results.

Continue capacity expansion only if the 1M model consistently outperforms the 86k model across multiple seeds and months, with gains across most months rather than a few extreme days. Otherwise retain the smaller model.

## Stage 6: Window expansion (stopped)

The following were initial launch gates. Later runs generated raw-1000 Top-100 and completed three-seed comparisons for `1M/raw-200`, `1M/raw-1000`, `100M/raw-200`, and `100M/raw-1000`. The window main effect was near zero and the capacity main effect was negative. No larger or longer formal working set is being generated.

| Working set | Estimated five-year storage | Launch gate |
|---|---:|---|
| raw-200 Top-400 | About 8 GiB | Generated |
| raw-500 Top-100 | About 5 GiB | First test for a longer-window gain |
| raw-500 Top-400 | About 20 GiB | Stable Top-100 gain |
| raw-1000 Top-400 | About 40 GiB | Not started; Top-100 did not show stable gain |

At each stage, compare against a minute-level model using the same dates, universe, and labels. The current stop condition has been met. Reconsider window expansion only after proposing a new data or model mechanism and passing a low-cost experiment.
