# L2 Event-Stream Modeling

This track losslessly packs order, trade, and snapshot events, then uses a causal Transformer for next-event tasks and daily signal output. Code lives in `ticknet.eventstream`. As of 2026-08-22, the three-seed 100M recent-fold run, frozen-embedding downstream comparison, three-seed joint end-to-end run, multi-task gradient audit, label-scale experiments, and supervision-position experiments were complete.

## Data contract

`ticknet-eventstream-pack` converts each day's three streams into integer mirrors and resolves linked IDs during packing, so later reads do not need the raw files. Each day produces:

- `orders_{day}.bin`, sorted by stock, time, and OrderID.
- `trades_{day}.bin`, sorted by stock, time, and DealID.
- `snaps_{day}.bin`, sorted by stock and time.
- `index_{day}.npz`, containing per-stock stream offsets, lengths, and prior close.

Cancel events are joined to original orders to calculate order age and original quantity. Trades are joined to the arrival time of both resting orders using buyer and seller IDs. Unresolved links use `AGE_UNKNOWN_MS = -1`. Packed values remain integers; normalization happens in the data loader, so feature transformations do not require repacking.

### Opening identity-ledger audit

`ticknet.simulator.opening_ledger` audits the pre-open order ledger independently of event-stream packing. It reads `order_preopen`, then combines orders, trades, and cancels within the event window corresponding to the first continuous-auction snapshot. Remaining quantity is calculated by order ID and compared with the top ten book levels. Run `scripts/audit_opening_ledger.py` with an explicit `--sample YYYYMMDD:TICKER` to avoid scanning a full raw L2 block accidentally.

Cross-stock and cross-date Shenzhen samples support an event-clock mapping of `snapshot time_ms + 140ms`. Shanghai has no single fixed offset and is not shifted by default. If a pre-open file is missing or does not contain the requested stock, the sample is marked `not_comparable` and excluded from precision. Among 13 samples on 2026-08-27, 9 matched all ten levels at the best lag; precision was 81.8% over 11 comparable samples. Shanghai's best lags were `0`, `70`, `90`, `120`, and `150ms`; none should be promoted to a market-wide constant. This audit has not changed production packing or added unverified Shanghai rules to the matching engine.

## Dataset and features

`ticknet.eventstream.dataset` merges the three streams by time. A sample is a contiguous event window for one stock on one trading day. Its 80 features include event intervals, basis-point changes from rolling midpoint, quantities, side, cancel/order age, L1 spread and imbalance, ten-level prices and sizes, traded value, time-of-day phase, and auction flags.

Targets cover three tasks:

- Next stream type: pad, snapshot, order, or trade.
- Next order type from the source `OrderType` vocabulary.
- Daily signal looked up by stock and date from an external label table; the label may be missing.

## Model and training

`ticknet.eventstream.model` implements a causal Transformer with rotary position embeddings. Presets include `smoke`, `probe25m`, `probe50m`, `capacity100m`, and `probe150m`. `capacity100m` has a 960-dimensional hidden state, 9 Transformer blocks, 15 attention heads, a 3,840-dimensional FFN, and 100,604,180 parameters. Attention uses PyTorch scaled-dot-product attention.

Train with `ticknet-eventstream-train`. Each epoch trains next-event tasks on training windows; validation computes daily Rank IC for the daily output. Early stopping follows `selection_metric`. The trainer saves best and last checkpoints plus a JSON history. Resume validates the experiment signature and dataset fingerprint.

### Optional M3-inspired representations

Three controlled experiment switches default to off. `use_lob_prefix` adds a book-state token at the window start using only snapshots before the boundary. `use_session_anchors` adds fixed, causal intraday price coordinates. `use_vq` quantizes core event behaviors and adds them as a residual to continuous event embeddings. Rolling-midpoint coordinates remain. Existing configurations retain the 80-dimensional input, parameter count, and checkpoint behavior by default.

Prefix/session-anchor options are recorded in materialized dataset and close-cache contracts. VQ options are included in checkpoint identity. Training, frozen-embedding export, prediction export/materialization, and gradient audit reconstruct the model from those contracts. Joint fine-tuning still uses the older close-window contract and explicitly rejects checkpoints with these new representations. Design, causal boundaries, experiment order, and evidence limits are in the [M3-inspired representation study](../research/m3-eventstream-representation.md).

`ticknet-eventstream-prepare-horizon-labels` converts next-day multi-horizon labels into H3/H5 wide tables. `trading_date`, `entry_date`, and `return_end_date` must all belong to train, validation, or OOS; labels crossing a boundary are removed. `ticknet-eventstream-benchmark` runs forward and backward passes plus AdamW updates on a real pack and reports throughput, memory, and per-seed runtime. It does not read validation or OOS.

## Prediction export and signal contract

`ticknet-eventstream-export-predictions` joins daily scores with formal open-to-following-open returns, tradability states, and the dynamic universe. It emits prediction Parquet conforming to `ticknet.research.prediction_contract`, for registration through `import_predictions` or direct use by `topk_cost_sweep`. Candidate rows use model scores. State rows use score `0.0` and track whether an existing holding is tradable.

Export a formal prediction artifact to the canonical `alpha-research` signal table:

```bash
ticknet-research-export-alpha-signals \
  --predictions artifacts/predictions.parquet \
  --output artifacts/signals.parquet \
  --model-version eventstream-v1 \
  --feature-set-id l2-clean-v1
```

The adapter uses `trading_date` as `signal_date`, never the future `label_date`. The original prediction Parquet remains compatible with `portfolio-backtester`'s prediction input. The canonical signal Parquet supports alpha-research IC, rolling validation, and downstream evidence workflows.

## Recent-fold configuration

The baseline is `configs/eventstream.yaml`; the 2021 infrastructure fold uses `configs/eventstream-h5-fold0-capacity100m.yaml`. The recent fold uses `configs/eventstream-h5-recent-capacity100m.yaml`:

```text
train       2025-08 through 2025-10
validation  2025-11
OOS         2025-12
locked      from 2026
```

All 103 trading days from August through December 2025 have been packed into about 313.11 GiB, with no partial files. Configuration and labels do not read the locked 2026 period.

The adjacent fold `fold-54-oos-202511` uses July–September 2025 for training, October for validation, and November for OOS. Its local configuration is `configs/eventstream-h5-fold-54-oos-202511-capacity100m.yaml`; the remote materialized configuration is `configs/eventstream-h5-fold-54-oos-202511-capacity100m-materialized-colab.yaml`. Remote data, checkpoints, and results are isolated by fold ID.

## Input benchmark and optimization

After the A100 capacity benchmark, sweep physical batch sizes 8, 16, 32, and 64 on the same August 2025 pack, holding effective batch size at 64:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-batch-size-sweep \
  --session ticknet-eventstream-h5-recent-sweep-a100 \
  --gpu A100 \
  --batch-sizes 8 16 32 64 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/batch-size-sweep/a100
```

Each size records throughput and memory independently; an OOM at one size does not block later sizes. The benchmark reads training packs only. Profile Dataset, preloaded GPU batches, and end-to-end throughput separately:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-input-profile \
  --session ticknet-eventstream-h5-recent-input-a100 \
  --gpu A100 \
  --num-workers 2 4 8 16 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/input-profile/a100
```

On 2026-08-12, A100 measurements identified the old Dataset implementation as the main input bottleneck. Preloaded GPU batches achieved 238.79 samples/s; the old Dataset reached only 18.19 samples/s end-to-end with 16 workers because it re-merged and processed each stock's full-day stream for every 512-event window.

The optimized implementation binary-searches the target event and merges only the nearby 513 events. Nine real windows matched the old implementation element by element; per-sample construction was 15.6–18.9× faster. End-to-end throughput reached 149.40 samples/s with 8 workers. Sixteen workers fell to 140.73 samples/s because the A100 runtime was limited to 12 CPU cores. Recent-fold configuration therefore fixes `num_workers: 8`.

For 120,000 training samples and 20 epochs, the projection is 13.39 minutes per epoch, up to 4.46 hours per seed and 13.39 hours for three sequential seeds. It excludes validation and checkpoint I/O; use actual training logs and early stopping for formal runtime.

## Formal training results and capacity gates

As of 2026-08-19, no local CUDA GPU was available. `rclone about gdrive:` reported 200 GiB total, 145.292 GiB used, and 53.305 GiB free. The complete five-month pack is about 313.11 GiB and does not fit on Drive or a Colab temporary disk.

Formal training therefore uses fixed-window materialization. For each seed, local processing samples training windows once and stores the 80-dimensional model inputs, next-event targets, daily labels, and valid positions. The manifest binds the five-month source inventory, source revision, dates, seed, sampling parameters, and SHA-256 for every tensor file. Training verifies every file first and stops on content drift or mismatched seed, dates, or revision.

`eventstream-recent-train` handles the latest fold; `eventstream-rolling-train` handles additional folds. Rolling runs require `--eventstream-fold-id` such as `fold-54-oos-202511`; remote paths and run summaries include that ID. Each workflow downloads one seed's materialized training set, resumes its checkpoint, verifies allowed files, and starts 100M training. Both successful and failed runs return best/last checkpoints, history, results, preflight report, and run summary.

The short-resume check downloads only train, validation, and H3-validation shards, runs one epoch, and does not expose OOS files to the runtime. Formal continuation uses the same source revision, runs to the epoch limit, and downloads OOS only for final evaluation. Code and synthetic tests cover tensor equality before/after materialization, tamper rejection, and resume over one or two epochs. Real seeds 0–2 completed source verification, materialization, remote training, checkpoint return, and OOS evaluation.

| Seed | Best epoch | H5 validation Rank IC | H5 OOS Rank IC | Training time |
|---:|---:|---:|---:|---:|
| 0 | 4 | 0.04345 | 0.05879 | 82.9 min |
| 1 | 6 | 0.09403 | 0.03730 | 116.3 min |
| 2 | 5 | 0.08029 | 0.03291 | 102.7 min |

Mean validation Rank IC was 0.07259; mean OOS Rank IC was 0.04300. All three were positive, passing the 100M signal gate. H3 monitoring was also positive on validation and OOS for all seeds. No 2026 data entered training or evaluation.

### Storage manifest and preflight

`ticknet-eventstream-storage-readiness` audits source data. Its manifest builder reads the daily universes for the five months, assigns each day to train, validation, or OOS, and records byte size, MD5, and SHA-256 for all 412 pack files and label artifacts. It stops if a universe includes 2026, a date does not belong to exactly one split, a pack is missing, or source fingerprints differ.

Run in the main workspace containing the complete local artifacts:

```bash
ticknet-eventstream-storage-readiness build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1 \
  --universe artifacts/eventstream-h5-recent-fold/202508/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202509/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202510/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202511/universe.json \
  --universe artifacts/eventstream-h5-recent-fold/202512/universe.json \
  --artifact fold-labels/manifest.json=artifacts/eventstream-h5-recent-fold/fold-labels/manifest.json \
  --artifact fold-labels/h3.parquet=artifacts/eventstream-h5-recent-fold/fold-labels/h3.parquet \
  --artifact fold-labels/h5.parquet=artifacts/eventstream-h5-recent-fold/fold-labels/h5.parquet \
  --output artifacts/eventstream-h5-recent-fold/storage-manifest.json
```

Manifest generation reads each complete pack sequentially to calculate content hashes and should run once after data is frozen. It records paths, sizes, hashes, date contracts, and aggregate statistics; it does not contain stock lists or market data.

The manifest also records commands for verifying a direct remote copy of a full pack, for benchmark packs and future storage migrations. The remote must expose at least MD5 or SHA-256:

```bash
rclone lsjson remote:ticknet-data/eventstream-h5-recent \
  --recursive --files-only --hash \
  > artifacts/eventstream-h5-recent-fold/remote-listing.json

ticknet-eventstream-storage-readiness verify-direct-remote \
  --manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --listing artifacts/eventstream-h5-recent-fold/remote-listing.json
```

Before copying a full pack, check space for the dataset, temporary files, and checkpoints. The default reserves 5% over data size plus 20 GiB:

```bash
ticknet-eventstream-storage-readiness check-full-copy-capacity \
  --manifest /content/storage-manifest.json \
  --path /content
```

After staging, verify each file's contents:

```bash
ticknet-eventstream-storage-readiness verify-staged \
  --manifest /content/storage-manifest.json \
  --root /content/ticknet-eventstream/top400-h5-recent
```

Full-pack commands are for audit and benchmarking. Formal 100M training uses the fixed-window cache. Its materializer writes atomically by month and supports resume; the formal workflow chains cache-manifest verification, checkpoint resume, training, and artifact return.

The recent fold selects checkpoints on H5 and uses H3 only for monitoring. All three seeds passed these gates: H5 daily Rank IC is positive on validation and OOS; the fingerprint, training history, best/last checkpoints, and validation/OOS outputs are complete; 2026 was not read; and OOS did not change this round's configuration.

The frozen-representation comparison uses next-day open-to-following-open downstream labels. Event-stream H5 labels train the encoder; the downstream task remains the project's daily Top-K trading question. HGB and LambdaMART compare minute features, frozen embeddings, and their combination.

`probe150m` is currently only an implemented model preset. Frozen embeddings and joint training both produced Rank IC signals, but the three-seed joint run did not pass the top-hit-rate and cost-adjusted active-return gates. First-round checks of signal half-life, trading rules, available risk exposures, multi-task gradients, label scale, and supervision position are complete. Daily cross-sectional z-scored labels improved validation and OOS Rank IC on two folds; last-position-only and linear tail weighting did not beat the all-position baseline. Next, test daily-task weighting, then decide whether to implement a cost-aware ranking objective. The initial 150M experiment waits for a stable training-mechanism gain across two folds. The raw-book capacity/window matrix is stopped; this track does not restart raw-200 or raw-1000 expansion.

### Multi-task gradient audit

`ticknet-eventstream-gradient-audit` calculates gradients on a fixed validation batch for next-stream type, next-order type, continuous-value regression, and H5 daily return, with respect to the shared Transformer. It compares seed-0 initialization with the best checkpoint and records loss, gradient norms and ratios, pairwise cosine similarities, and input fingerprints.

Colab workflows `eventstream-recent-gradient-audit` and `eventstream-rolling-gradient-audit` download only validation shards and checkpoints with registered SHA-256. They exclude training, OOS, monitoring partitions, and the locked 2026 period. Full contracts, gates, and commands are in the [event-stream gradient audit](../research/eventstream-gradient-audit.md). Two-fold seed-0 label-scale results, the supervision-position contract, and formal conclusions are in the [label-scale study](../research/eventstream-label-scale.md).

### Joint end-to-end experiment

`ticknet-eventstream-joint-cache` creates a compact cache from the frozen E2 stock-day intersection. It stores 120-dimensional minute features, three-class labels, ranking returns, portfolio evaluation targets, and references to the shared close-window cache. Event arrays remain in the existing 6.17 GiB cache; they are not duplicated. The manifest binds the minute materialization fingerprint, close-cache fingerprint, dates, and evaluation configuration, and records SHA-256 for each Parquet file.

`ticknet-eventstream-joint-train` loads the requested seed's `capacity100m` best checkpoint. The scheduler fixes filenames and SHA-256 for seeds 0–2 and passes the seed explicitly. The model takes the final valid-event hidden state from the close window, encodes 120 aggregated minute features with a minute tower, concatenates both representations, and predicts three classes. Ranking score is up probability minus down probability. The first round trains only new layers; later stages update the Transformer with `backbone_lr` and the minute tower/classification head with `head_lr`.

Build the formal compact cache locally:

```bash
ticknet-eventstream-joint-cache build \
  --minute-config configs/nextday-minute-formal-2025-v2.yaml \
  --minute-features results/m3-formal-minute-features-v2-202107 \
  --comparison-config configs/embedding-frozen-recent-2025.yaml \
  --close-cache artifacts/eventstream-h5-recent-fold/daily-close-cache \
  --output artifacts/eventstream-h5-recent-fold/joint-feature-cache-v1 \
  --source-revision "$(git rev-parse HEAD)"
```

Run one seed remotely, explicitly allowing access to the previously approved December 2025 OOS period:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-joint-finetune \
  --session ticknet-eventstream-joint-seed0 \
  --gpu A100 \
  --seeds 0 \
  --timeout 14400 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/joint-finetune/seed0
```

The remote workflow downloads the shared close cache, joint cache, and requested seed checkpoint, then resumes existing outputs before training unfinished epochs. Results include daily validation/OOS Rank IC, NDCG, precision, cost-adjusted Top-K return, turnover, prediction Parquet, checkpoint, and run summary. 2026 remains isolated.

The formal cache contains 22,409 training, 6,963 validation, and 8,125 OOS samples, totaling 17,948,094 bytes. It shares the exact stock-day intersection, labels, and evaluation configuration with frozen E2. Fingerprint: `e4f54a62e4be3f36ac0693db59ebcdb120cd753d2dc36415b8686adaa13c1bb6`. Five local files match their Drive copies. Predictions for all three seeds match on stock, date, label, and row counts: 6,963 validation and 8,125 OOS rows.

Seed 0 trained for at most five epochs with early-stopping patience 2. With the Transformer frozen, epoch 1 validation Rank IC was 0.04430. Unfreezing the backbone raised it to 0.05784 at epoch 2. Epochs 3 and 4 scored 0.01457 and 0.03628, followed by early stopping. Final evaluation used the epoch-2 best checkpoint.

| Method | Validation Rank IC | OOS Rank IC | OOS `NDCG@100` | OOS `Precision@100` | OOS Top-100 mean daily cost-adjusted active return | OOS mean daily one-way turnover |
|---|---:|---:|---:|---:|---:|---:|
| HGB E0 minute features | 0.01808 | 0.04010 | 0.53424 | 0.26810 | -11.51 bp | 62.89% |
| HGB frozen E2, seed 0 | 0.02462 | 0.04333 | 0.53277 | 0.27048 | -7.39 bp | 64.99% |
| HGB frozen E2, three-seed mean predictions | 0.02833 | 0.05701 | 0.54450 | 0.26667 | -4.80 bp | 60.91% |
| Joint end-to-end, seed 0 | 0.05784 | 0.06296 | 0.54452 | 0.24762 | -9.26 bp | 49.91% |
| Joint end-to-end, seed 1 | 0.07694 | 0.05492 | 0.53985 | 0.21667 | -14.40 bp | 56.74% |
| Joint end-to-end, seed 2 | 0.04272 | 0.07407 | 0.55083 | 0.25333 | -5.34 bp | 42.56% |
| Joint end-to-end, three-seed mean | 0.05917 | 0.06398 | 0.54507 | 0.23921 | -9.67 bp | 49.74% |

Across joint seeds, validation Rank IC was `0.05917 ± 0.01400`, and OOS Rank IC was `0.06398 ± 0.00785`; all three OOS values were positive. OOS `NDCG@100` was `0.54507 ± 0.00450`, and mean one-way daily turnover was `49.74% ± 5.79%`. `Precision@100` was `0.23921 ± 0.01611`. Mean daily cost-adjusted active return was `-9.67 ± 3.71 bp`, negative for all three seeds.

Follow-up work on signal half-life, additional windows, staggered H5 holdings, rank smoothing, and known risk exposures is complete. EMA and return-difference turnover gates reduced turnover, but cost-adjusted active-return signs did not repeat across two consecutive OOS windows. The current decision is `HOLD`; training-mechanism ablations are next, and 150M remains deferred. See [event-stream signal and trading diagnostics](../research/eventstream-signal-trading-diagnostics.md).

### Fixed-window materialization and formal training

After building the source manifest, materialize seed 0 locally where the full pack is stored:

```bash
ticknet-eventstream-materialize build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --output artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --source-revision "$(git rev-parse HEAD)"

ticknet-eventstream-materialize verify \
  --root artifacts/eventstream-h5-recent-fold/materialized/seed0
```

Materialization resumes month by month after verifying SHA-256 for existing shards. Upload a verified directory to its seed-specific location:

```bash
rclone --config ~/.config/rclone/rclone.conf copy \
  artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/eventstream-top400-h5-recent-materialized/seed0 \
  --checksum
```

The first remote run trains one formal epoch without OOS access, to verify checkpoint return and resume:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```

For rolling folds where only materialized arrays are available, recover stock identity locally and run checkpoint inference remotely. The two manifests bind materialized and source fingerprints and the 2026 lock boundary:

```bash
ticknet-eventstream-materialized-predictions keys \
  --config configs/eventstream-h5-fold-54-oos-202511-capacity100m-materialized-colab.yaml \
  --storage-manifest artifacts/eventstream-fold-54/storage-manifest.json \
  --materialized-root artifacts/eventstream-fold-54/materialized/seed0 \
  --output artifacts/eventstream-fold-54/sample-keys \
  --allow-oos

ticknet-eventstream-materialized-predictions score \
  --checkpoint artifacts/eventstream-fold-54/training/seed0/best.pt \
  --materialized-root artifacts/eventstream-fold-54/materialized/seed0 \
  --model capacity100m \
  --output artifacts/eventstream-fold-54/predictions/seed0 \
  --device cuda \
  --allow-oos \
  --source-revision "$(git rev-parse HEAD)"
```

`ticknet-eventstream-signal-diagnostics` joins stock identity, scores, H1–H10 label sidecars, and daily bars. It reports signal half-life, 27 trading rules, five H5 cohorts, dynamic costs, and risk exposures. See the research diagnostics page for command options and artifacts.

### Frozen embeddings and downstream comparison

Three checkpoints share a single close-window cache. It stores the final 512 events before the close for each stock-day, with model inputs and stock-day keys only. It contains no random training windows and does not vary by seed:

```bash
ticknet-eventstream-close-cache build \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --pack-root /mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/top400-h5-v1 \
  --output artifacts/eventstream-h5-recent-fold/daily-close-cache \
  --seq-len 512 \
  --min-events 256 \
  --source-revision "$(git rev-parse HEAD)"

ticknet-eventstream-close-cache verify \
  --root artifacts/eventstream-h5-recent-fold/daily-close-cache
```

The shared cache was generated, fully checked, and uploaded. It contains 39,903 stock-days in five shards totaling 6,619,831,094 bytes (about 6.17 GiB), with fingerprint `59577182c8124c312de0591059c67e55d472511ca77753403ce77afbf8f109f4`. Remotely, each seed loads its corresponding training-cache manifest and checkpoint to export 960-dimensional vectors. Full export reads the previously approved December 2025 OOS period, so the scheduler retains explicit OOS authorization. Export one seed at a time:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-export-embeddings \
  --session ticknet-eventstream-embedding-seed0 \
  --gpu A100 \
  --seeds 0 \
  --embedding-batch-size 16 \
  --local-output-dir artifacts/eventstream-h5-recent-fold/embeddings/seed0
```

The scheduler downloads only the shared close cache, the requested seed's training manifest, and its best checkpoint. It returns embeddings, manifest, run summary, and Colab execution record, then checks seed, source revision, OOS status, and 2026 isolation. To run directly in an existing CUDA environment:

```bash
ticknet-eventstream-export-embeddings \
  --close-cache artifacts/eventstream-h5-recent-fold/daily-close-cache \
  --checkpoint artifacts/eventstream-h5-recent-fold/training/seed0/eventstream-top400-h5-capacity100m-recent.seed0.best.pt \
  --training-manifest-root artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --model capacity100m \
  --device cuda \
  --allow-oos \
  --output artifacts/eventstream-h5-recent-fold/embeddings/seed0 \
  --source-revision "$(git rev-parse HEAD)"
```

All three seeds were exported at source revision `449b843c83d7494ae7a396d658792eaa664ab2eb`. Local manifest verification and file-by-file Drive checks passed. Each embedding contains 39,903 rows, with identical stock-day keys and order.

| Seed | Embedding fingerprint | File bytes |
|---:|---|---:|
| 0 | `a4d67c5f06a3147d036a43700bcc88bd2e5b47b74c934a4255192255f0435b36` | 146,103,015 |
| 1 | `850ed79795d34b8e040bacad174abc3ba4b4942f865b6b9f560439fcef78530a` | 145,899,441 |
| 2 | `c51bceff90a52982b57fd1c9c4999fed4e27285993a8c00a38379e444f61e43a` | 145,939,261 |

Run downstream comparisons after seeds 1 and 2 are available:

```bash
ticknet-embedding-compare \
  --minute-config configs/nextday-minute-formal-2025-v2.yaml \
  --minute-features results/m3-formal-minute-features-v2-202107 \
  --comparison-config configs/embedding-frozen-recent-2025.yaml \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed0 \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed1 \
  --embedding artifacts/eventstream-h5-recent-fold/embeddings/seed2 \
  --output results/embedding-frozen-recent-2025
```

The comparison includes three input sets for HGB and LambdaMART: minute features, embeddings, and their combination. Metrics include Rank IC, `NDCG@50/100`, `Precision@50/100`, cost-adjusted Top-K returns, turnover, and monthly stability. Risk-exposure analysis requires a Parquet file with `trading_date`, `symbol`, `industry`, `size`, `liquidity`, and `volatility`; otherwise the result records `unavailable`.

The `FEAT-EMB-FROZEN-001` comparison used 22,409 training, 6,963 validation, and 8,125 OOS samples. Event-stream coverage of recent-fold minute candidates was 96.69%. Validation covered 18 evaluation days and OOS 21. E1 and E2 use three downstream seeds each and average prediction scores.

| Downstream model | Input | Validation Rank IC | OOS Rank IC | OOS `NDCG@100` | OOS `Precision@100` | OOS Top-100 mean daily cost-adjusted active return | OOS mean daily one-way turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| HGB | E0 minute features | 0.01808 | 0.04010 | 0.53424 | 0.26810 | -11.51 bp | 62.89% |
| HGB | E1 embedding | 0.02647 | 0.01966 | 0.52349 | 0.25905 | -13.14 bp | 64.30% |
| HGB | E2 combined | 0.02833 | 0.05701 | 0.54450 | 0.26667 | -4.80 bp | 60.91% |
| LambdaMART | E0 minute features | -0.04334 | 0.00766 | 0.52153 | 0.30143 | -0.73 bp | 48.95% |
| LambdaMART | E1 embedding | -0.01117 | 0.03414 | 0.53030 | 0.29286 | 15.53 bp | 52.57% |
| LambdaMART | E2 combined | -0.05081 | 0.01389 | 0.52695 | 0.31143 | 6.91 bp | 52.32% |

HGB E2 single-seed OOS Rank IC values were 0.04333, 0.05644, and 0.05912, each above E0's 0.04010. The three-seed mean prediction gained 0.01691 paired OOS Rank IC; it beat E0 on 16 of 21 days, with a daily-bootstrap 95% interval of 0.00596–0.02851. The validation paired gain was 0.01025, with an interval still crossing zero. HGB E2 shows a relatively stable representation gain within this fold, while cost-adjusted active return remains negative.

LambdaMART E2 varied substantially across three seeds and two months. E1's OOS cost-adjusted active return was positive, while validation was -15.34 bp; treat it as a lead requiring replication. Risk-exposure inputs were unavailable, so industry, size, volatility, and liquidity diagnostics are marked `unavailable`. Results are in `results/embedding-frozen-recent-2025/comparison.json`, fingerprint `56a7689048e539963a217c92221e8cddf1ce472526115411d5478a4a6d18dc00`.

Current candidates are frozen E2, HGB, and joint training. The joint three-seed comparison uses the same stock-days, labels, and metrics and treats current E2 as the direct control. The recent fold has only one validation and one OOS month, so add risk exposures and more time windows before drawing a broad conclusion. Keep 150M deferred.

The formal training command below resumes the same-revision checkpoint and evaluates December 2025 OOS after training:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-formal-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 20 \
  --evaluate-test \
  --timeout 21600 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```
