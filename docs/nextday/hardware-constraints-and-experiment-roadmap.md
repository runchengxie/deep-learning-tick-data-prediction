# Hardware Constraints and Staged Experiment Roadmap

## Objective

The research question is whether same-day ticks, ten-level order-book snapshots, orders, and trades can reliably predict next-trading-day cross-sectional relative returns.

Minute-level microstructure features, precomputed embeddings, and raw-event models all derive from tick data. Start with low-cost representations to measure signal strength, then test whether raw sequences add value.

## Current engineering priorities

As of 2026-08-16, the minute HGB, TCN, and GRU models and the raw-book and event-stream infrastructure were implemented. The raw-book 2×2, three-seed capacity matrix was complete and supports stopping further expansion. The current sequence is:

```text
verify real seed-0 materialization of fixed event-stream windows
  → run event-stream recent-fold capacity100m seed 0
  → add seeds 1 and 2 only if the gate passes
  → connect frozen embeddings to M4
  → run probe150m capacity ablation only after a clear gain
```

The raw-book model encodes 100-event chunks with a shared DeepLOB and aggregates them with a GRU, producing a continuous excess-return score and three-class probabilities. It supports raw-200 and raw-1000 windows and approximately 1M and 100M parameters. Only `1M/raw-200` remains a candidate.

The later sections preserve early hardware inventories, stage designs, and throughput estimates for auditing historical resource decisions. Current status and task order are maintained in [project status](../project-status.md).

## Resource snapshot

The following reflects read-only host checks and short benchmarks on 2026-08-04.

### Compute resources

| Resource | State | Implication |
|---|---|---|
| CPU | Intel i5-4690K, 4 cores / 4 threads | Suitable for streaming preprocessing, tree models, and small neural networks |
| Memory | 31 GiB RAM plus 31 GiB swap | Do not load a full month of book data; stream row groups or batches |
| GPU | GeForce GTX 970 using the nouveau driver | `cuda_available=False`; treat the host as having no usable training GPU |
| Internal disk | About 619 GB free on NVMe | Store filtered training data, caches, checkpoints, and results |
| Data disk | 6 TB USB HDD, about 890 GB free | Keep raw data here; suitable for sequential scans, not random training reads |
| Google Drive | Upgraded to 200 GB | Store pilot/formal raw-200 working sets, checkpoints, and results; keep raw data local |

The GTX 970 is not part of the primary plan. Maintaining a separate driver, CUDA, and PyTorch stack for an old 4 GB card would raise reproduction costs without fitting the desired batch. Use Colab or an on-demand cloud GPU for GPU training.

### Data resources

Raw ten-level snapshots are stored at:

```text
/mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2/snapshot
```

The 60 canonical monthly files for 2021–2025 are present and total about 884 GiB. The snapshot directory also contains about 206 GiB of 2026 monthly files, daily files, and repair archives, for roughly 1.1 TiB total. Schemas match across five sampled years and contain 75 fields, including `AskPrice1~10`, `AskVolume1~10`, `BidPrice1~10`, and `BidVolume1~10`. `time_ms` is milliseconds since 09:30; 14:55 is `19_500_000`. A check of `000001` found 4,702 valid snapshots before 14:55 on one trading day. Its final `Price` matched the daily-bar close.

The same disk has 60 monthly order files for 2021–2025 (about 1.2 TB) and 60 trade files (about 1.5 TB). `order_preopen` is about 24 GB. `_incoming` is about 320 GB and includes some 2026 daily files not yet in canonical monthly directories. The event-stream track owns individual orders and trades; see [eventstream.md](eventstream.md). Any 2026 incremental data requires its own data contract, coverage audit, and model-input design. Do not mix it directly into existing checkpoints.

The minute microstructure cache is at:

```text
/mnt/data/hdd6t/quant-data-lake/derived/level2_minute_cache/v1
```

It covers 2021–2026 and is about 122 GB. It contains 11 snapshot, 11 order, and 11 trade features per minute; the combined 33-column representation also includes a validity indicator for each modality. This cache already aggregates raw ticks to minute bars and is the most useful starting point on the available hardware.

## Measured training throughput

Short local CPU benchmarks:

| Model | Input | Throughput | One epoch for 400 stocks × 250 days |
|---|---|---:|---:|
| Chunked DeepLOB | 10 × 100 book events | About 8.5 stock-days/s | About 3.3 hours |
| Single-layer GRU | 240 × 33 minute features | About 61 stock-days/s | About 27 minutes |

These estimates exclude USB reads, Parquet decompression, validation, and checkpoint I/O. Chunked DeepLOB takes about 9.8 hours per epoch over three years; ten continuous epochs would take several days. It is not suitable as the first-stage primary model.

## Overall compute strategy

```text
USB HDD: raw snapshot/order/trade and existing minute cache
  → sequentially scan each source file once
NVMe: compact minute sequences, daily features, or raw-book shards for the dynamic universe
  →
Local CPU: data audit, Logistic Regression, tree models, small-sample training
  →
Colab or cloud GPU: small TCN/GRU, raw-book incremental tests, one-time embedding extraction
  →
Drive: retain only current working sets, embeddings, checkpoints, and results
```

Do not repeatedly scan the 884 GiB canonical snapshot files during training, or randomly read large numbers of stock-day files from the USB HDD. Preprocess sequentially by year or month and write a small number of large training shards.

## Dataset-size choices

Estimates use 400 stocks and about 1,250 trading days.

| Representation | Approximate five-year storage | Use |
|---|---:|---|
| Daily aggregated features | Hundreds of MB | Logistic Regression, LightGBM, and data audit |
| Last 60 minutes × 33 features | About 4 GB | Recommended primary model |
| Full 240-minute day × 33 features | About 16 GB | Test the value of the full day |
| Last 200 book events × 40 features | About 8 GB in float16 | Raw-book end-to-end track |
| Last 500 book events × 40 features | About 20 GB in float16 | Raw-book expansion |
| Last 1,000 book events × 40 features | About 40 GB in float16 | Later expansion, only after earlier gates pass |
| Daily 64-dimensional embedding | About 128 MB | Multi-day hierarchical models and repeated tuning |

## Shared research protocol

### Samples and universe

One sample represents one stock and one input trading day. Build a dynamic universe using only information known before the signal time:

- Require at least 120 listed trading days.
- Require the stock to be tradable that day with complete data.
- Select the top 400 using trailing 20-day turnover or liquidity.
- Lag universe statistics by at least one day; do not use next-day information.
- Retain delisted, suspended, and historically inactive stocks to reduce survivorship bias.

The initial signal time is 14:55. The minute model uses the 60 nodes from 13:55 through the minute before 14:55. The raw-book model uses the final events satisfying `time_ms <= 19_500_000`.

### Targets

The continuous target is next-day open-to-close return:

```text
next_return = next_close / next_open - 1
target_return = next_return - csi_all_a_next_return
```

Primary evaluation uses daily Rank IC on continuous `target_return`. Three-class labels are used only to train classification models:

- Bottom 20% each day: down class.
- Middle 60%: neutral class.
- Top 20%: up class.
- Keep equal returns in the same class at quantile boundaries; do not split ties by ticker.

### Time splits

The initial formal experiment uses:

```text
Train:      2021-01 through 2023-12
Validation: 2024-01 through 2024-12
Test:       2025-01 through 2025-12
```

Engineering smoke runs may use only 2024 Q1, but they cannot support a strategy-performance claim. Split by complete trading days and purge samples whose labels cross a boundary. Fit normalization, universe thresholds, and feature selection using training dates only.

## Staged research plan

### Internal control A: audit minute-cache data

Confirm that the existing cache can produce leakage-free samples:

- Read yearly snapshot, order, and trade Parquet directly.
- Join modalities by `date,ticker,minute`.
- Build the historical dynamic Top-400 universe.
- Extract the last 60 minutes available before the signal time.
- Generate next-day open-to-close excess returns.
- Report sample manifests, missingness, class distribution, and date coverage.

Acceptance requires inputs at or before the signal time; adjacent input and label trading days; disjoint splits with boundary labels removed; one sample per stock-day; missingness summarized by date, stock, and industry; and documented label coverage, suspension, and price-limit handling.

Suggested outputs:

```text
data/nextday-minute/
  manifest.json
  samples-2021.parquet
  samples-2022.parquet
  ...
results/data-audit/
  coverage.json
  missingness.parquet
  label-distribution.json
```

### Internal control B: low-cost baselines

Aggregate 60-minute sequences into daily features and train, in order:

1. Majority-class and zero-score baselines.
2. Logistic Regression.
3. HistGradientBoosting or LightGBM.

Train on local CPU first. Determine whether microstructure information has consistent direction out of sample. Proceed to more complex models only when mean daily validation Rank IC is positive, gains are not concentrated in a few months, score-group returns are broadly monotonic, and small changes to universe, signal time, or label thresholds do not reverse the result. These gates decide whether further compute is warranted; they do not replace locked-test results.

### Research extension A: 60-minute TCN

Recommended starting architecture:

```text
60 × 33 minute-level microstructure features
  → normalization fitted on training dates
  → 3–4 lightweight TCN blocks
  → 64-dimensional intraday vector
  → continuous score or three-class prediction
```

Compare snapshot-only, snapshot+order, all three modalities, 60 versus 240 minutes, and TCN versus a small GRU. Keep the model below about 500,000 parameters. Use Colab for individual tuning runs and multiple seeds for the final configuration.

Retain this model as a useful control only if TCN consistently beats the aggregated baseline on validation, Rank IC/group returns/stability agree, and adding order or trade features yields reproducible improvement.

### Raw-book end-to-end model

The raw-book model tests whether minute aggregation loses information useful for next-day prediction. Data preparation, training, and gates are in the [raw-200 end-to-end pipeline](raw-200-end-to-end-pipeline.md); this section preserves the initial design.

The formal experiment began with a controlled 2024 pilot: dynamic Top-100 stocks, the final 200 book events, 2024 H1 training, Q3 validation, and Q4 locked test. Configurations are `configs/nextday-raw-pilot.yaml` and `configs/nextday-pilot.yaml`; the million-parameter variant is `configs/nextday-raw-1m-pilot.yaml`. The pilot required data audit, Logistic baseline, and a 100-batch Colab throughput run before allocating an end-to-end budget. Daily metrics require at least 80 stocks. Since 2024 Q4 has already informed development, it is not an unseen test for new model selection.

The initial full configuration was 2021–2025, dynamic Top-400, and the last 200 book events, split into two 100-event DeepLOB blocks. Later experiments skipped raw-500 and directly compared raw-200/raw-1000 and 1M/100M on the same fixed Top-100 sample in a four-cell, three-seed matrix. The capacity main effect was negative and the window main effect near zero, triggering the stop gate. Full results are in the [multi-horizon and data expansion roadmap](multi-horizon-data-expansion-roadmap.md).

Stop raw-book capacity and window expansion. Retain existing raw models as controls against minute features.

### Research extension B: encode once, train across days

After selecting a single-day encoder, freeze a version and generate daily embeddings once:

```text
daily minute or order-book sequence
  → fixed intraday encoder
  → 64-dimensional embedding
  → recent 5-, 10-, or 20-day TCN/GRU
  → next-day cross-sectional score
```

For five years and 400 stocks, 64-dimensional float32 embeddings require about 128 MB. A multi-day model can then be iterated locally without repeatedly encoding raw ticks.

## Evaluation and stop rules

Every stage should report daily Rank IC mean, median, standard deviation, and positive share; monthly Rank IC and market-regime breakdowns; Macro F1, balanced accuracy, MCC, and Brier score; quantile portfolio returns, turnover, and approximate costs; stability by stock, industry, size, and liquidity; and variation across seeds and walk-forward folds.

Stop model expansion if training metrics improve without validation Rank IC gains; a month or seed change reverses the result; raw-book models do not consistently beat the minute TCN; returns concentrate in untradable price-limit, post-suspension, or very-low-liquidity stocks; or reasonable costs erase grouped returns.

## Current engineering sequence

The planned minute-cache adapter, dynamic universe, minute baseline, TCN, GRU, raw working sets, and event-stream packing are implemented. Next:

1. Start M4 and compare HGB with LambdaMART on the same data fingerprint.
2. Complete real seed-0 materialization, upload, and short-resume verification for event-stream fixed windows.
3. Run recent-fold `capacity100m` seed 0; add seeds 1 and 2 only if gates pass.
4. Freeze a passing checkpoint as an embedding and connect it to M4 under the same downstream contract.
5. Run a `probe150m` capacity ablation only if the 100M signal or frozen-embedding gain passes its gate.
6. Keep 2026 locked under the research protocol and explicit approval.

For each step, add synthetic-data tests before running a small real-data sample. Store real outputs under Git-ignored `data/` and `results/`; commit code, configuration, schemas, and aggregate audits only.
