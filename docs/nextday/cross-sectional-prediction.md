# Next-Day Cross-Sectional Prediction

## Research question

This experiment track is independent of the FI-2010 paper reproduction. The raw order-book pipeline supports raw-200 and raw-1000 windows. The current raw-200 candidate works as follows:

```text
one stock × one input trading day
  final 200 ten-level order-book events before the signal time
    two 100-event chunks
      DeepLOB encodes each chunk
        GRU aggregates the intraday chunk sequence
          continuous excess-return score + down/neutral/up probabilities
```

The default label ranks next-trading-day returns cross-sectionally: the bottom 20% maps to class 0, the middle 60% to class 1, and the top 20% to class 2. Ties at the quantile boundaries remain neutral, rather than being split arbitrarily by ticker. Fixed return thresholds are also supported. When a benchmark return is supplied, labels and Rank IC use excess returns.

Each stock-day contributes one sample. The implementation does not duplicate every tick from a day as a separate sample sharing the same label.

The end-to-end track trains directly on raw ticks. For five years and a dynamic Top-400 universe, raw-200 keeps the float16 working set near 8 GB. Raw-1000 and a 100M-parameter model have also been implemented and compared under controlled conditions; current evidence favors the smaller `1M/raw-200` model. Minute-level microstructure models remain as lower-cost controls. See the [hardware and staged experiment roadmap](hardware-constraints-and-experiment-roadmap.md) for resource budgets and expansion gates.

The default raw-200 architecture uses 16 convolution channels, 32 channels in each Inception branch, a 64-dimensional intraday embedding, and a 64-dimensional daily GRU, for 86,775 parameters. `configs/nextday-raw-1m-pilot.yaml` provides an isolated capacity experiment with widths 32, 64, 320, and 192, for 1,033,383 parameters. It writes checkpoints to a separate directory and does not overwrite the default baseline. Training and inference checkpoints record the full architecture signature.

## Boundary with the paper reproduction

The FI-2010 reproduction, including the `ticknet-fi2010-train` entry point, `FI2010WindowDataset`, and Setups 1 and 2, is archived under `legacy/` and is no longer part of the primary track. The current pipeline lives in `ticknet.nextday` and is trained with `ticknet-nextday-train`. FI-2010 does not provide reliable stock and trading-day boundaries, so it cannot be used to create these next-day labels.

## Input features

Upstream event arrays are computed in `float32`. Training shards may be stored as `float16` or `float32`, with shape `events × 40`; the dataset converts them back to `float32` before model input. The 40 columns must follow the ten-level order-book layout:

```text
ask_price_1, ask_size_1, bid_price_1, bid_size_1,
ask_price_2, ask_size_2, bid_price_2, bid_size_2,
...
ask_price_10, ask_size_10, bid_price_10, bid_size_10
```

Event arrays must:

- Include only data available before the signal time.
- Use consistent price, quantity, and book-level units.
- Contain no NaN or infinite values.
- Fit any learned normalization parameters using training dates only.
- Avoid test-period corporate-action information and full-sample statistics.

The current real-data adapter uses fixed transformations that require no fitting. Prices become basis-point changes relative to the first midpoint in the selected window, divided by 100. Sizes become `log1p(size) / 16`; values are then clipped to fixed bounds. This preserves within-window price changes without using validation- or test-period statistics. Transformation parameters are recorded in the manifest.

## Raw-file adapter

If monthly Shanghai and Shenzhen snapshot Parquet files are available locally, run:

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw.yaml
```

The adapter uses trailing 20-day turnover known before the prior day to build a historical dynamic Top-400 liquidity universe, selects the final 200 valid ten-level book states from candidate snapshots between 14:30 and 14:55, computes next-day stock open-to-close return in excess of the CSI All Share return, skips irrelevant ticker row groups by month, and writes large float16 shards, a manifest, and `data-audit.json`.

This is the historical diagnostic target for the raw-sequence model. The formal M3 Top-K HGB input uses the separate `configs/nextday-minute-formal-2025-v2.yaml` configuration. Its target is the holding return from the T+1 open to the T+2 open associated with the signal on T, classified against concurrent CSI All Share open-to-open returns. The dynamic Top-400 universe uses only turnover known before T. T+1 suspension, limit-up, and limit-down states are recorded separately. Holdings removed from the universe but not yet sellable continue to receive state rows with `in_universe=false`. Formal splits also require `return_end_date` to remain within the same period.

Materialize minute features and run the formal baseline:

```bash
uv run python scripts/materialize_minute_features.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --output results/m3-formal-minute-features-v2-202107

uv run python scripts/run_minute_baseline.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --materialized-features results/m3-formal-minute-features-v2-202107 \
  --evaluate-test \
  --save-predictions results/predictions-hgb-top400-open2open-2025-v2.parquet \
  --output results/nextday-minute-formal-2025-v2.json
```

The first command writes Parquet atomically by month and records source identity, target keys, per-month resource statistics, and shard SHA-256 hashes. Re-running after interruption resumes from complete months. The second command accepts only a manifest covering every target month and emits a formal data fingerprint and prediction metadata for registration through `import_predictions`. Candidates without an L2 minute window remain in the Top-400 with all-NaN features, which HGB supports natively; they must not be silently dropped.

M3 v1 materialized 14 of 60 months. July–December 2025 contained 49,600 rows, of which 192 lacked minute features; feature coverage was 99.61%. When 2021 processing resumed, 48–50% of candidates in January–May lacked all three modalities. Daily raw order files confirmed that Shenzhen alone was covered from 2021-01-04 through 2021-06-04; Shanghai data began on June 7. v1 was stopped and retained. v2 restarts materialization at the first complete month, `2021-07`.

The contiguous-array extractor produced an identical SHA-256 for the July 2025 shard compared with the previous implementation. Runtime fell from 100.3 to 72.4 seconds, and peak memory from about 2.26 to 1.69 GB. The other five months took 62.0–82.6 seconds each, with batch peak memory under 1.72 GB. These results validate full-run resource usage, resumability, and numerical consistency only; they do not indicate predictive performance.

The corrected local v2 smoke run covered the dynamic Top-20 universe from 2024-01-02 through 2024-01-12. It wrote 178 of 180 target rows; two stock-days lacked usable snapshots. All 178 written samples had 200 valid ticks. It read 44 and skipped 386 of 430 row groups, and removed 1,017 invalid book rows from candidate windows. This validates the data pipeline only, not model quality.

### Generic event-manifest input

For a different vendor format, use the vendor-neutral event-manifest entry point, `scripts/prepare_nextday.py`. It requires three inputs.

Daily bars CSV:

```csv
symbol,trading_date,open,close
000001.SZ,2024-01-02,9.41,9.53
000001.SZ,2024-01-03,9.55,9.49
```

Trading calendar, one date per line:

```text
2024-01-02
2024-01-03
```

Event JSONL manifest:

```json
{"symbol":"000001.SZ","trading_date":"2024-01-02","features_path":"events/000001.SZ-2024-01-02.npy","last_event_timestamp":"2024-01-02T14:54:59+08:00","signal_timestamp":"2024-01-02T14:55:00+08:00"}
```

`features_path` is resolved relative to the JSONL file. This interface deliberately avoids a vendor-specific raw format. An upstream adapter can produce stock-day arrays and JSONL manifests from monthly Parquet, databases, or object storage. An optional benchmark CSV has `trading_date,return` columns; returns are decimals, so `0.005` means 0.5%.

To generate two 100-event chunks:

```powershell
python scripts/prepare_nextday.py `
  --daily-bars data/daily-bars.csv `
  --calendar data/calendar.txt `
  --events-manifest data/events.jsonl `
  --benchmark data/benchmark.csv `
  --output-dir data/nextday `
  --label-method cross_sectional `
  --min-cross-section 100 `
  --chunks-per-sample 2 `
  --chunk-size 100 `
  --samples-per-shard 512
```

When a generic-manifest sample has fewer than 200 events, the adapter left-pads it by repeating that day's first valid book state; `valid_events` in the manifest retains the true event count. The primary raw-snapshot configuration instead sets `min_valid_events` to 200, removes invalid rows first, and omits stock-days that cannot provide a complete window.

Output layout:

```text
data/nextday/
  manifest.json
  shards/
    part-00000.npy
    part-00001.npy
```

Each shard has shape `samples × chunks × time × 40`. NPY files can be memory-mapped, so training does not need to load the full dataset at once. `manifest.json` stores the stock, input date, label date, return, class, signal timestamp, valid event count, and shard location for each sample. Each shard records its size and SHA-256. The `dataset_fingerprint` covers sample metadata and all shard hashes.

## Date splits and leakage control

The formal working set uses three non-overlapping date ranges: training in 2021–2023, validation in 2024, and locked test in 2025.

```yaml
train_start: "2021-01-01"
train_end: "2023-12-31"
val_start: "2024-01-01"
val_end: "2024-12-31"
test_start: "2025-01-01"
test_end: "2025-12-31"
```

Both input and label dates must be in the same split. If the final training date's label falls in validation, the sample is removed automatically. Every stock on a trading day receives the same date-based assignment; stocks are never randomly split across sets.

The upstream universe must also enforce historical point-in-time eligibility, including listing date, suspension, price limits, delisting, and liquidity screens. The current code does not reconstruct a historical universe from today's stock list.

## Training and baselines

After editing `configs/nextday.yaml`, train with:

```powershell
ticknet-nextday-train --config configs/nextday.yaml
```

Checkpoints and training history are written to `checkpoint_dir`. Resume verifies the experiment signature, including dates, model, learning rate, and `dataset_fingerprint`, and stops on configuration or data conflicts. By default, training sequentially verifies every shard SHA-256 at startup.

The formal configuration sets `evaluate_test: false`. Training then computes validation metrics only, and the result JSON has `test: null`. After freezing the validation-period model, configuration, and seed set, evaluate all best checkpoints once with the pure evaluation entry point:

```bash
ticknet-nextday-evaluate \
  --seeds 0 1 2 3 4 \
  --config configs/nextday.yaml
```

Before calculating test metrics, the evaluator verifies that all checkpoints exist and checks dates, data fingerprints, model, and training configuration. It does not create an optimizer, read a `last` checkpoint, or enter a training loop. Results go to `locked_test.<checkpoint_name>.seeds0-1-2-3-4.json`, with per-seed metrics, cross-seed mean, and sample standard deviation. Do not select seeds using test results.

The training entry point retains `evaluate_test` for smoke-test workflows. Use the pure evaluator above for formal locked tests so an unfinished training run cannot resume accidentally.

After training, convert one stock's raw `N × 40` snapshots before the signal time into a next-day signal:

```bash
ticknet-nextday-predict \
  --checkpoint checkpoints-nextday/raw-200-dual-head.seed0.best.pt \
  --manifest data/nextday-raw-200/manifest.json \
  --events-npy data/today-000001.npy \
  --input-format raw \
  --device cpu
```

Output includes a standardized continuous `score`, `expected_excess_return` mapped back to the return scale, three class probabilities, and a direction ID. For cross-sectional trading, rank same-day stock scores. A single-stock return estimate requires separate calibration and is not a promised return.

Run the aggregated-feature Logistic Regression baseline first:

```powershell
python scripts/run_nextday_baseline.py `
  --config configs/nextday.yaml `
  --output results/nextday-baseline.json
```

The baseline follows `evaluate_test` too. Use `--evaluate-test` only after the formal configuration is frozen. For each stock-day it calculates the mean, standard deviation, final value, and endpoint change for each of the 40 event features. It then fits a `StandardScaler` on training data only and trains a class-balanced Logistic Regression model. This is the minimum control for deciding whether raw-sequence modeling adds value; it does not replace richer microstructure features or a LightGBM baseline.

## Evaluation

The dual-head model jointly trains classification cross-entropy and Smooth L1 regression on standardized continuous returns. Results include accuracy, Macro/Weighted F1, balanced accuracy, MCC, per-class precision and recall, multiclass Brier score, daily Rank IC mean/standard deviation/ICIR, and the uncosted daily return spread between the highest- and lowest-scoring groups.

The long-short spread excludes fees, slippage, price limits, market impact, and short-sale constraints. It is not a tradable backtest. Model selection defaults to daily validation Rank IC. The formal test is explicitly unlocked through `ticknet-nextday-evaluate`, after training and model selection, for fixed best checkpoints only.

## Data volume and Colab

With float16 storage, 400 stocks, about 1,250 trading days, and 200 events per stock-day require roughly 8 GB for features alone. The same range with 500 and 1,000 events requires about 20 GB and 40 GB. Sharding enables batch reads in Colab but does not reduce the raw data volume.

Keep full tick and order-book data on an external drive, NAS, or object store. Generate only compact shards needed for the current study. Copy the experiment shards sequentially to `/content`; use Drive for the working set, checkpoints, and results. Avoid random training reads through a mounted Drive path. See the [Colab CLI automation guide](../dev/colab-cli-automation.md) for the current runner.

Twelve trading days are enough to validate the engineering workflow. An initial cross-sectional study should use at least 120–250 trading days; a formal out-of-sample experiment should cover two or three years and multiple market regimes.

Use local CPU for one-time raw-tick extraction and data audit, and a Colab GPU for end-to-end training. The existing `level2_minute_cache` provides a low-cost control to test whether raw ticks add value.

The controlled 2024 pilot used a dynamic Top-100 universe and the last 200 order-book events:

```bash
ticknet-nextday-prepare-snapshot --config configs/nextday-raw-pilot.yaml
python scripts/run_nextday_baseline.py --config configs/nextday-pilot.yaml \
  --output results/nextday-pilot-baseline.json
```

`configs/nextday-pilot.yaml` uses 2024 H1 for training, Q3 for validation, and Q4 for testing. `evaluate_test: false` is the default, so neither the baseline nor DeepLOB model-selection stage emits test metrics. Daily cross-sectional metrics require at least 80 stocks; days below 80% coverage of the dynamic universe are excluded. For the million-parameter development run:

```bash
ticknet-nextday-train --config configs/nextday-raw-1m-pilot.yaml --seed 0
```

Run seeds 0, 1, and 2, and compare capacity using validation only. 2024 Q4 has already informed pilot results and cannot be treated as an unseen test set for selecting the million-parameter model. If a canonical monthly file is corrupt, the converter falls back only when daily snapshots fully cover the target trading day; `data-audit.json` records fallback months and file counts. It stops if daily backups are incomplete.

## Current status and next step

The current version implements the monthly snapshot adapter, dual-head chunk encoder, AMP, gradient accumulation, checkpoint resume, and Colab handoff. The five-year raw-200 set, raw-1000 Top-100 set, multi-horizon evaluation, and four-cell three-seed capacity matrix are complete. See the [raw-data expansion roadmap](raw-data-expansion-roadmap.md) and [multi-horizon roadmap](multi-horizon-data-expansion-roadmap.md). The event-stream track handles stored orders and trades; see [eventstream.md](eventstream.md).

Only `1M/raw-200` remains a candidate. Next, freeze the three-seed aggregation method, checkpoints, and one-time test acceptance criteria before deciding whether to unlock 2025. A multi-day model can cache daily embeddings and train on them without repeatedly encoding raw ticks.
