# Experiment Log

This log records dated experiments, decisions, and material evidence. See the [research roadmap](topk-agentx-research-roadmap.md) for the current state and [resource strategy](resource-strategy-and-pilot-gates.md) for compute gates. Detailed protocols and artifact identities remain in the linked experiment reports.

## 2026-08-29: CPU validation path without a GPU

Added `ticknet.research.cpu_validation` to separate interface defects from GPU-training defects. A controlled NumPy linear model produces formal predictions, alpha-research signals, and fixed-K long-only portfolio results. Tests cover train/test date separation, signal conversion, and differential summaries: all 22 focused tests and Ruff passed.

Inspected the first row group of `order_20260424.parquet` (1,048,576 of 313,199,561 rows). The platform profiler maps `SecuCode` to `ticker` and `OrderTime` to `time_ms`. The sample contained 883 non-positive prices and 206,009 repeated `OrderID` rows; timestamps were ordered. Repeated IDs may represent valid cancel/update events or a loading issue. Inspect `OrderType` and `BizIndex` before deciding; do not drop them yet.

Added `digest_portfolio_backtester_result()` and `compare_portfolio_digests()`. They compare engine-neutral returns, costs, turnover, drawdown, and daily fields without requiring both projects to share PyTorch.

## 2026-08-07: Google Drive organization

Moved project folders within Drive using reversible metadata operations only; no files were deleted or uploaded/downloaded. The project now uses the top-level folder `deep-learning-tick-data-prediction/` with `ticknet-data/` (raw-200 pilot and smoke sets), `ticknet-runs/` (five-seed runs, locked test, baselines, and local evidence), `fi2010-reproduction/`, `code-legacy-v1/`, the pre-test backup, and the default Colab and AI Studio folders. Current folders use `ticknet` or semantic names; legacy package names were retained inside the old code snapshot so it remains runnable. Checkpoint files were renamed from `deeplob.setup2.*.pt` to `ticknet.setup2.*.pt`.

The raw-200 A-share pilot covered 23,515 samples in 2024. The logistic baseline had validation Rank IC about 0.015 and MCC 0.059. The five-seed deep model had locked-test Rank IC `0.0075 ± 0.015`, MCC `0.070 ± 0.010`, and Macro F1 `0.335 ± 0.028`: weak, seed-unstable, and not clearly better than baseline. A 1,033,383-parameter capacity run used the same data and training protocol as the 86,775-parameter baseline. Drive operations used remote-host rclone because the local `gdrive:` OAuth token expires; private credential details are intentionally not stored here.

## 2026-08-07: Minute TCN versus aggregate-feature HGB

Compared unaggregated minute sequences with aggregate features under the same universe, labels, split, and 60-minute window. The 2024 L2 minute cache held 24,188 samples in 12 shards (174 MB), shaped `samples x 60 x 30` float32. HGB used 120 aggregate features. Twenty-five of 30 minute columns had missing values (11,251 samples affected); medians were fit on training data, and short windows were padded before filling. Splits were train 11,595 (Jan–Jun), validation 6,293 (Jul–Sep), and test 6,000 (Oct–Dec).

The TCN used four dilated causal-convolution layers (kernel 3), 64 channels, weight normalization, and two output heads. It stopped within 12 epochs; the best checkpoint was epoch 7 (validation Rank IC 0.058). HGB used `HistGradientBoosting`, `max_iter=500`, and `leaf=31`.

| Metric | HGB validation | TCN validation | HGB test | TCN test, seed 0 |
|---|---:|---:|---:|---:|
| Daily Rank IC mean | 0.0325 | 0.0579 | 0.0111 | -0.0081 |
| Macro F1 | 0.313 | 0.224 | 0.361 | 0.214 |
| MCC | 0.107 | 0.041 | 0.131 | 0.032 |

Across three test seeds, TCN Rank IC was `0.0093 ± 0.016`, Macro F1 `0.253 ± 0.033`, MCC `0.060 ± 0.024`, and balanced accuracy `0.358 ± 0.014`; HGB scored 0.0111, 0.361, 0.131, and 0.379 respectively. TCN's validation ranking advantage (0.047–0.058 versus HGB 0.032) did not generalize; HGB is the steadier minute baseline. Deliverables included the sharded L2 pipeline, median filling and padding, and `ticknet-minute-tcn-train` / `-evaluate`. The cache is far smaller than the 23 GB annual raw parquet; CPU training took about 6.6 minutes per seed.

## 2026-08-08: Multi-year minute HGB robustness

Using 120 aggregate features, `max_iter=500`, a dynamic top-100 universe, and 60-minute windows, trained on all years before each test year, validated on its first half, and tested on its second half. Configurations and results are `configs/nextday-minute-rolling-{2022,2023,2024,2025}.yaml` and `results/nextday-minute-rolling-{2022,2023,2024,2025}.json`.

| Test year | OOS Rank IC | Days | Test MCC | Train samples |
|---:|---:|---:|---:|---:|
| 2022 | 0.0218 | 124 | 0.060 | 19,559 |
| 2023 | 0.0326 | 123 | 0.110 | 43,734 |
| 2024 | 0.0353 | 124 | 0.113 | 67,930 |
| 2025 | 0.0304 | 125 | 0.081 | 92,118 |

All four OOS years were positive (Rank IC 0.022–0.035): a real but weak signal, not evidence of a profitable net strategy. `_build_samples_by_year` reduced peak memory from over 20 GB to about 9 GB. Storing minute features as float32 halved their memory again without changing metrics.

## 2026-08-08: Cost-adjusted long-short evaluation

Evaluated daily equal-weight top/bottom deciles from `results/predictions-rolling-2025.parquet` over 125 trading days in 2025 H2. One-way commission-plus-impact costs were 0, 3, 5, 10, or 20 bp; sells also paid 0.05% stamp duty. Daily turnover was 83%.

| One-way cost | Gross annualized | Net annualized | Net Sharpe | Mean daily cost |
|---:|---:|---:|---:|---:|
| 0 bp | +27.9% | +27.9% | 0.88 | 0.04% |
| 3 bp | +27.9% | +15.7% | 0.49 | 0.09% |
| 5 bp | +27.9% | +7.5% | 0.24 | 0.13% |
| 10 bp | +27.9% | -12.8% | -0.40 | 0.21% |
| 20 bp | +27.9% | -53.6% | -1.68 | 0.38% |

Breakeven cost was about 5–6 bp, below the assumed feasible A-share cost (usually at least 10 bp including impact). Rebalancing every five days reduced turnover to 17% and mean daily cost to 4.3 bp, but gross annualized return fell to -30% and net return to -40.8%. The signal was short-horizon daily momentum and not tradable at these costs. This is the historical quantile long-short diagnostic; the current Top-K series uses fixed-K long-only, open-to-open returns, and stock-level fills in `ticknet.research.portfolio` (see the [M1 evaluator](topk-agentx-m1-portfolio-evaluator.md)).

### Same-day audit of the IC/spread discrepancy

The 2025 prediction audit reported daily Rank IC 0.030 and IC IR 0.211. The best day contributed 25.7% of spread; the best five days contributed 121% and the best ten 212%. Median daily spread was 0.026%, decile monotonicity was 0.41, and monthly IC was positive in five of six months (October 2025: -0.015). Thus a few extreme days inflated mean spread while typical-day returns were small. Added `ticknet.research.audit`, `ticknet-research audit`, and five tests; the diagnostics provide an observation interface for a future Evaluation Agent.

## 2026-08-08: AgentX-style research loop

Added a machine-callable research system before adding more agent autonomy. `ticknet.research` contains experiment specifications, deterministic policy and budget checks, locked-test protocol, a single runner, SQLite registry, prediction audit, and locked-test approval. `agents/` contains the client abstraction, context builder, brainstorm, critic, and orchestrator. The one-way flow is Context → Brainstorm → Critic → Policy → Runner → Audit → Registry. The CLI exposes run/show/compare/audit/approval/locked-test/agent-step commands. Python determines metrics and decisions; agents cannot change `test_end`; every proposal declares a falsification condition; negative findings are retained; test data remain physically isolated.

All 115 tests passed (16 new research tests), and an end-to-end `agent-step` trained and registered `EXP-AUTO-TCN2`. A policy violation blocked an unauthorized proposal. Developer Agent, SGPO/Harness Evolution, and live DeepSeek/OpenAI calls remained unimplemented; provider interfaces were reserved.

## 2026-08-08: Agent-driven rebalance-frequency hypothesis

The first LLM-driven brainstorm proposed that a 2–3 day rebalance might balance turnover and return. `ticknet-research agent-step` created `EXP-LLM-ROUND1` (`git_sha=86bb130`) and evaluated 2025 H2 at 10 bp one-way cost:

| Rebalance interval | Turnover | Gross annualized | Net annualized | Net Sharpe |
|---:|---:|---:|---:|---:|
| 1 day | 0.83 | +38.1% | -12.8% | -0.40 |
| 2 days | 0.42 | -41.6% | -67.0% | -2.02 |
| 3 days | 0.29 | -54.6% | -72.2% | -2.50 |
| 5 days | 0.17 | -30.2% | -40.8% | -1.23 |
| 10 days | 0.09 | -9.0% | -14.6% | -0.49 |

The hypothesis was rejected: at 2–3 days, gross return collapsed faster than turnover. The signal behaved like daily momentum and was incompatible with the cost structure. The automated loop reached the same cost-bottleneck diagnosis independently and stored the result as structured evidence.

## 2026-08-12: Event-stream A100 input-pipeline optimization

On the August 2025 Top-400 training pack, measured DataLoader-only, preloaded GPU-only, and end-to-end throughput for the 100,604,180-parameter `capacity100m` causal Transformer. Only August 2025 training data were read; fingerprint `705445378f0fc5842ce80bcfa41a01cdd10236198c5683f108f0ba146f8c3b82`. Validation, OOS, and locked 2026 data were not accessed.

The old GPU-only rate was 235.28 samples/s. End-to-end rates with 2/4/8/16 workers were 4.62/9.53/13.23/18.19 samples/s. At 120,000 samples and 20 epochs, even the best setting projected 36.65 hours per seed. Each 512-event sample previously re-merged and sorted all daily order/trade/snapshot events and built 80 features. August contained 8,097 `(day, ticker)` pairs and about 2.455 billion events; uint32 merge indexes would take about 9.15 GiB. Simulated LRU hit rates over 20 shuffled epochs were only 5.45% with 8 GiB total cache and 22.15% with 32 GiB, so LRU was rejected.

The optimized dataset binary-searches three sorted streams, locates the merge rank, and stably merges only 513 nearby events while carrying forward the preceding valid snapshot midpoint. Exhaustive synthetic checks and nine windows across three real days matched the old output element by element. Per-sample construction became 15.6–18.9x faster. On a 12-core A100 runtime, GPU-only throughput was 238.79 samples/s; end-to-end rates for 2/4/8/16 workers were 65.24/123.71/149.40/140.73 samples/s, with projected 20-epoch runs of 10.22/5.39/4.46/4.74 hours per seed. Eight workers were selected: 8.21x over the old best and about 34x over the initial 4.4 samples/s. Three seeds projected to 13.39 hours before validation/checkpoint/early-stop overhead. Replacing the existing RoPE causal-attention/FFN Transformer with a Hugging Face wrapper would not address this input bottleneck.

## 2026-08-16: Raw-1000 Top-100, 100M three-seed training

Ran seeds 0–2 under the frozen [`nextday-raw-1000-top100-capacity-100m.yaml`](https://github.com/runchengxie/quant-deep-learning/blob/main/configs/nextday-raw-1000-top100-capacity-100m.yaml) contract. The model had 100,817,575 parameters and predicted next-day open-to-close excess return. Training used 2021–2023, validation used 2024, and daily mean Rank IC selected checkpoints. The workset had 118,078 samples: 70,805 train and 23,472 validation; fingerprint `f8a17e63d0716f9e48fd05f9a269bb61cea5bff81e9a7acf90c4a42e47505e5c`.

| Seed | Best / actual epoch | Validation Rank IC | Rank ICIR | Macro F1 | MCC | Time |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 / 17 | 0.03295 | 0.19748 | 0.37910 | 0.09666 | 66.70 min |
| 1 | 20 / 28 | 0.03340 | 0.23013 | 0.38944 | 0.11766 | 109.65 min |
| 2 | 12 / 20 | 0.02822 | 0.18163 | 0.36553 | 0.10205 | 78.27 min |

Mean validation Rank IC was 0.03152 (sample SD 0.00287; range 0.02822–0.03340), using 4.24 GPU-hours total. This exceeded the prior 1M/raw-200 mean of 0.02031 by 0.01121 (55.2%), but the runs also differed in capacity, window, universe, learning rate, and minimum daily cross-section. Checkpoints were selected on the same 2024 validation period. This is not an isolated capacity effect, OOS significance, or tradable-return result. The 2025 test remained locked: result `test` fields were null and run status was `locked_not_accessed`; 23,512 was only the test metadata row count.

## 2026-08-16: Top-100 capacity/window 2×2, three seeds

Held Top-100 universe, 2021–2023 train, 2024 validation, next-day open-to-close excess-return target, batch 32, learning rate 0.0001, patience 8, and daily Rank IC selection fixed. The 1M and 100M models had 1,033,383 and 100,817,575 parameters. `raw-200` was a zero-copy view of the final two 100-event chunks from the same `raw-1000` mmap workset. All four cells therefore shared fingerprint, 70,805 train samples, 23,472 validation samples, and 241 valid validation days. New cells used source `e2465c0`; the existing 100M/raw-1000 seeds used `56f99d9` (seed 0) and `95a3a90` (seeds 1–2). The older config's missing `input_last_chunks` means its default full raw-1000 view.

| Capacity/window | Seed validation Rank ICs | Mean ± sample SD (range) | GPU hours |
|---|---|---:|---:|
| 1M / raw-200 | 0.03792, 0.03815, 0.03638 | 0.03748 ± 0.00096 (0.03638–0.03815) | 0.72 |
| 1M / raw-1000 | 0.03784, 0.03305, 0.03500 | 0.03530 ± 0.00241 (0.03305–0.03784) | 0.89 |
| 100M / raw-200 | 0.03202, 0.02412, 0.02605 | 0.02740 ± 0.00412 (0.02412–0.03202) | 1.04 |
| 100M / raw-1000 | 0.03295, 0.03340, 0.02822 | 0.03152 ± 0.00287 (0.02822–0.03340) | 4.24 |

The nine new runs' best/actual epochs, Rank ICIR, Macro F1, MCC, validation long-short means, and training times were recorded in their run artifacts. The three 1M/raw-200 rows were respectively `9/17, 0.21095, 0.36924, 0.08023, 0.00604, 13.75 min`; `12/20, 0.23970, 0.37857, 0.08376, 0.00316, 15.54 min`; and `10/18, 0.25074, 0.37254, 0.08423, 0.00388, 13.92 min`. The 1M/raw-1000 rows were `6/14, 0.21150, 0.38324, 0.10277, 0.00494, 15.52 min`; `13/21, 0.22400, 0.38334, 0.09617, 0.00430, 23.17 min`; and `5/13, 0.19339, 0.38204, 0.09241, 0.00432, 14.48 min`. The 100M/raw-200 rows were `2/10, 0.17677, 0.35937, 0.06946, 0.00404, 16.25 min`; `5/13, 0.15957, 0.34294, 0.07794, 0.00140, 20.91 min`; and `8/16, 0.13955, 0.35886, 0.07763, 0.00342, 25.52 min`.

Across cells, capacity effects were -0.01008 at raw-200 and -0.00377 at raw-1000; window effects were -0.00219 at 1M and +0.00413 at 100M. The mean main effects were -0.00693 for capacity and +0.00097 for window, with +0.00631 interaction. `1M/raw-200` exceeded `100M/raw-1000` by 0.00596 and had the best, least variable result. Three seeds and shared validation selection limit inference; these are controlled validation findings, not significance or OOS conclusions. All 12 result files had null test values and `locked_not_accessed` status. Keep 1M/raw-200 as the sole candidate; freeze aggregation, checkpoint, one-shot test gates, and report format before considering 2025 test access.

## 2026-08-16: M3 early order-feed coverage audit and v2 start

M3 v1 intended to materialize January 2021–December 2025. Recovery completed 14/60 months and 114,400 candidates; 19,827 had all three modality features missing. Missing rates were about 48–50% per month from January–May 2021, 9.36% in June, 0.19% in July, and 0.08% in August. Direct inspection of 118 daily order files on the 6 TB data disk found that the 101 sessions from 2021-01-04 through 2021-06-04 contained Shenzhen-prefixed symbols but no Shanghai-prefixed symbols. Shanghai appeared on 2021-06-07 with 1,896 symbols. Snapshot and trade sources did contain Shanghai; the gap was isolated to the order feed. Annual minute-cache boundaries matched daily raw files, so feature extraction did not introduce the gap. No complete alternate feed was found.

Using v1 would create exchange and time bias. Earlier 2022–2025 minute HGB rolling results remain engineering history, with the 2022 fold most affected, but no longer count as formal M3 decision evidence. Configuration changed to `configs/nextday-minute-formal-2025-v2.yaml`, starting at the first complete month, 2021-07, with 2025 H1/H2 validation/test and output `results/m3-formal-minute-features-v2-202107`. Keep v1 outputs and manifests as audit records.

## 2026-08-16: M3 v2 formal Top-K cost diagnosis

V2 materialized 54 monthly shards from 2021-07 through 2025-12: 436,800 candidates, 436,256 complete three-modality rows, and 544 all-NaN rows (99.88% complete). Manifest identity, SHA-256, row count, and date boundaries passed. Runtime was 4,240.2 seconds, peak memory about 1.71 GB. Materialization identity: `a005bb9525dbbdc1f266c75561f82cb71f43631270253f46081d9d293c0bf45c`; manifest fingerprint: `3ee9871248666a105c7b21c7281767d5b6b43a3b272306152f16360d183bd9c0`.

HGB used 339,600 train, 46,000 validation, and 49,600 test samples. Validation/test Rank IC were 0.08091/0.06994; test Rank ICIR 0.53953, Macro F1 0.32129, MCC 0.13208. The formal prediction had 51,489 rows (49,600 candidates plus 1,889 dynamic-universe status rows), exactly 400 candidates on each of 124 evaluation days. Prediction SHA-256: `bdeb2cbe7de8b894fd246ff56c31e49e510c0c38eac115687047c096a9a2a45d`; data fingerprint: `6ca055086c8885bcb866da01af4481a95d58c1334ed3fe0b574cca5b66dcbb7a`.

`PRED-HGB-400-OPEN2OPEN-001` had positive monthly IC in all six months; 61.29% of days had positive IC. The top five days contributed 42.21% of long-short spread, below M3's 50% limit. `TRD-TOPK-400-001` tested 64 combinations: K=25/50/75/100, buffer=0/10/25/50, and one-way cost=5/10/15/20 bp. All used identical 124 dates. At 10 bp plus 5 bp sell stamp duty, all 16 candidates had negative net active return versus equal-weight Top-400: `NO_TRADEABLE_REGION`. `K=100, buffer=50` had the best absolute net return (12.15 bp/day, Sharpe 1.41, 41.87% one-way daily turnover) but -4.75 bp/day active return and only one positive month. Its active breakeven one-way cost was 4.33 bp, highest in the grid; absolute breakeven was 24.51 bp, partly explained by broad-market gains and not standalone alpha. Buffer reduced turnover and improved absolute net return but created no stable net active edge. Do not start `TRD-BUFFER-400-001`; M3 is complete. M4 compares HGB and LambdaMART under the same fingerprint with NDCG, precision, and exposure diagnostics.

## 2026-08-17: Event-stream 100M recent-fold three seeds

Trained `capacity100m` on August–October 2025, validated November, and evaluated December OOS. Each model had 100,604,180 parameters and 512 merged order/trade/snapshot events per input. Per seed, the training cache held 120,000 fixed windows; validation had 5,807 stock-days and OOS 6,966. H5 daily Rank IC selected checkpoints; H3 was monitoring only.

| Seed | Best epoch | H5 validation Rank IC | H5 OOS | H3 validation | H3 OOS | Time |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0.04345 | 0.05879 | 0.04095 | 0.05101 | 82.9 min |
| 1 | 6 | 0.09403 | 0.03730 | 0.06093 | 0.03371 | 116.3 min |
| 2 | 5 | 0.08029 | 0.03291 | 0.05834 | 0.03954 | 102.7 min |

H5 validation mean was 0.07259 (sample SD 0.02615); OOS mean 0.04300 (SD 0.01385). All six values were positive, meeting the pre-set 100M signal gate; all H3 monitoring values were also positive. Seed cache fingerprints were `5a7d9216c7b4a8f680ef8a22ca760b482b6ccd38f6a8df587bd7deb44f445314`, `db771015c5069f5ac0d9dfd953fbc09f9e02d98985b3af62da24794b7b36aaca`, and `99b70c05813702f9da7c77e4ac1f516a5ecb48255c9c1039ddbaf2c66cec4fff`; only fixed training windows differed. Best/last checkpoints, histories, results, run summaries, and recovery contracts were saved. Locked 2026 data were not accessed.

Next, test frozen representation increment rather than infer tradability. `FEAT-EMB-FROZEN-001` uses one shared late-day window cache and exports three 960-dimensional embeddings for HGB and LambdaMART comparisons (minute features, embedding, and combined). A shared cache was completed at source revision `3138f55a0dd82e28b077e90d6b14c582113da441`: 39,903 stock-days from August–December (23,250 train, 7,756 validation, 8,897 OOS), five shards totaling 6,619,831,094 bytes (about 6.17 GiB), fingerprint `59577182c8124c312de0591059c67e55d472511ca77753403ce77afbf8f109f4`. File checksums, shapes, date boundaries, stock-day uniqueness, and 2026 isolation passed. Keep one remote cache; download per-seed manifests and best checkpoints.

## 2026-08-17: Frozen event embeddings and downstream increment

At source revision `449b843c83d7494ae7a396d658792eaa664ab2eb`, exported the 960-dimensional hidden state at each stock-day's last valid event from three 100M checkpoints on Colab A100. Each embedding had 39,903 rows, covering 103 sessions and 944 historically selected stocks. Keys, shard row counts, and order matched across seeds; local manifest validation and Drive file checks passed.

| Seed | Checkpoint SHA-256 | Embedding fingerprint | Bytes |
|---:|---|---|---:|
| 0 | `8632e62bdf4f27383e299c3ff676876d8a1969f6d69ec66a7ce43da24f5255e9` | `a4d67c5f06a3147d036a43700bcc88bd2e5b47b74c934a4255192255f0435b36` | 146,103,015 |
| 1 | `edc423d89bbd2a681383d04ec1c3ae22961b2c944c449f0572f5416f31de19ed` | `850ed79795d34b8e040bacad174abc3ba4b4942f865b6b9f560439fcef78530a` | 145,899,441 |
| 2 | `013e2bd1281830100bbf15f673bd8b1cb8ff08951ea48eae9f81922b1eebd4f6` | `c51bceff90a52982b57fd1c9c4999fed4e27285993a8c00a38379e444f61e43a` | 145,939,261 |

`FEAT-EMB-FROZEN-001` fixed August–October train, November validation, and December OOS. E0 used aggregate minute features; E1 frozen embeddings; E2 their concatenation. Downstream models were independently trained per seed, and only scores were averaged. The common subset had 22,409 train, 6,963 validation, and 8,125 OOS samples: 96.69% coverage of the recent-fold minute candidates, with 18 validation and 21 OOS evaluation days.

| Model | Input | Validation IC | OOS IC | OOS NDCG@100 | OOS Precision@100 | OOS net active bp/day | OOS one-way turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| HGB | E0 minute | 0.01808 | 0.04010 | 0.53424 | 0.26810 | -11.51 | 62.89% |
| HGB | E1 mean of three seed scores | 0.02647 | 0.01966 | 0.52349 | 0.25905 | -13.14 | 64.30% |
| HGB | E2 mean of three seed scores | 0.02833 | 0.05701 | 0.54450 | 0.26667 | -4.80 | 60.91% |
| LambdaMART | E0 minute | -0.04334 | 0.00766 | 0.52153 | 0.30143 | -0.73 | 48.95% |
| LambdaMART | E1 mean of three seed scores | -0.01117 | 0.03414 | 0.53030 | 0.29286 | 15.53 | 52.57% |
| LambdaMART | E2 mean of three seed scores | -0.05081 | 0.01389 | 0.52695 | 0.31143 | 6.91 | 52.32% |

HGB E2 single-seed OOS ICs were 0.04333, 0.05644, and 0.05912, exceeding E0 by 0.00323, 0.01634, and 0.01903. Mean-score paired OOS increment was 0.01691; E2 won on 16/21 days; daily bootstrap 95% CI was 0.00596–0.02851. Validation increment was 0.01025 (CI -0.00505–0.02536). The rank increment repeated over three seeds and two months, but Precision@100 dipped and net active return remained negative. LambdaMART E2 single-seed increments were -0.01399, 0.01387, and 0.02864; E1's 15.53 bp/day OOS net active return came with -15.34 bp/day validation and weak month stability. Risk exposures were unavailable. Retain frozen E2 and HGB as candidates, allow one-seed `FEAT-EVENTSTREAM-JOINT-001`, and require another window plus exposure checks before M5 completion. Do not promote LambdaMART; `probe150m` remains paused. Comparison fingerprint `56a7689048e539963a217c92221e8cddf1ce472526115411d5478a4a6d18dc00`; file SHA-256 `ce14306242884547525b95b34035c929c7db38d3da3be2d4aebfc9131d1d088b`. Locked 2026 data were not used.

## 2026-08-18: Joint event-stream and minute-feature seed 0

`FEAT-EVENTSTREAM-JOINT-001` fixed the frozen-E2 stock-day intersection, labels, dates, and evaluation. Its lightweight cache stored 120-dimensional minute features, classification and ranking targets, portfolio targets, and relative shard/row references into the shared event cache; it did not duplicate the 6.17 GiB event arrays. It held 22,409 train, 6,963 validation, and 8,125 OOS samples, plus 40,274 portfolio-target rows (17,948,094 bytes). Fingerprint `e4f54a62e4be3f36ac0693db59ebcdb120cd753d2dc36415b8686adaa13c1bb6`; contract SHA-256 `c50c1f44bd13074d15ed71e2da893216707ce3349705012ed0ee7db9ad69b410`. Five local cache files matched Drive.

The first scheduled run stopped before model loading because the seed-0 checkpoint SHA had accidentally been set to seed 2's `013e2bd1281830100bbf15f673bd8b1cb8ff08951ea48eae9f81922b1eebd4f6`. Identity validation correctly refused to continue. PR #80 corrected it to seed 0's `8632e62bdf4f27383e299c3ff676876d8a1969f6d69ec66a7ce43da24f5255e9` and added an exact identity test. After Python 3.10/3.12 and dependency-audit gates passed, the fix merged; the A100 session and staged inputs were reused.

The model loaded the seed-0 `capacity100m` checkpoint and concatenated the 960-dimensional last-valid-event state with the minute tower. It had 100,899,607 parameters. Epoch 1 froze the Transformer; subsequent epochs used `1e-5` backbone and `3e-4` new-layer learning rates. Validation IC selected checkpoints; patience was 2.

| Epoch | Transformer | Train loss | Validation IC | NDCG@100 | Precision@100 | Time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Frozen | 0.91846 | 0.04430 | 0.52686 | 0.25611 | 55.2 s |
| 2 | Updated | 0.90262 | 0.05784 | 0.52342 | 0.25833 | 128.5 s |
| 3 | Updated | 0.88303 | 0.01457 | 0.52438 | 0.27500 | 135.9 s |
| 4 | Updated | 0.85736 | 0.03628 | 0.52488 | 0.27611 | 136.2 s |

Epoch 2 was best. Later training loss kept falling as validation IC declined, indicating rapid overfit after unfreezing. Versus HGB E0, HGB frozen-E2 seed 0, HGB frozen-E2 mean, and joint seed 0, respectively, validation IC was 0.01808/0.02462/0.02833/0.05784; OOS IC 0.04010/0.04333/0.05701/0.06296; OOS NDCG 0.53424/0.53277/0.54450/0.54452; Precision 0.26810/0.27048/0.26667/0.24762; net active return -11.51/-7.39/-4.80/-9.26 bp/day; turnover 62.89/64.99/60.91/49.91%. Joint seed 0 gained 0.01963 OOS IC over frozen seed 0 and 0.00595 over the three-seed mean, but lost 4.46 bp/day of net active return versus the mean and had lower Precision. Higher cross-sectional rank correlation had not become a stronger head portfolio. Source `da01954b22a1a1506c9e91f8558fcd80bf8184e8`; best checkpoint SHA `585321c251e664982f8e1066d8dfff50d97d15b42cbabfe07d1cb91261b91106`; last SHA `7f2f188ddb81bfe83e03a233db4ec6c519848eec3a091b488e2d6767bf9e2c19`; result JSON SHA `a3d38beef87e47a25e85dd64141de2ebe1ce37305049074100650e1020004a82`. Seven final Drive files matched local copies; locked 2026 was not used. Decision: `EXTEND`; run seeds 1–2, exposure, and another window before consideration. Prioritize the trading objective over 150M capacity; keep `probe150m` paused.

## 2026-08-18: Joint end-to-end three-seed replication

PR #82 pinned each seed's pretrained checkpoint filename and SHA and passed the seed explicitly; unknown seeds fail before GPU allocation. Python 3.10, Python 3.12, and dependency audit passed. Source revision `92426f67060e7ebb24cb3400ada6aa8af38ae804`. Seed 1's first A100 kernel disconnected before cache copy or training; execution history was retained, remote rclone config removed, and the session reused successfully. Seed 2 completed in a new session; both sessions were stopped.

All predictions shared stock, date, label, and row keys (6,963 validation and 8,125 OOS rows). Seeds 1 and 2 used checkpoint SHAs `edc423d89bbd2a681383d04ec1c3ae22961b2c944c449f0572f5416f31de19ed` and `013e2bd1281830100bbf15f673bd8b1cb8ff08951ea48eae9f81922b1eebd4f6`.

| Seed | Best epoch | Validation IC | OOS IC | OOS NDCG@100 | OOS Precision@100 | Gross active bp/day | Net active bp/day | Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0.05784 | 0.06296 | 0.54452 | 0.24762 | 3.10 | -9.26 | 49.91% |
| 1 | 1 | 0.07694 | 0.05492 | 0.53985 | 0.21667 | -0.33 | -14.40 | 56.74% |
| 2 | 1 | 0.04272 | 0.07407 | 0.55083 | 0.25333 | 5.19 | -5.34 | 42.56% |
| Mean | — | 0.05917 | 0.06398 | 0.54507 | 0.23921 | 2.65 | -9.67 | 49.74% |
| Population SD | — | 0.01400 | 0.00785 | 0.00450 | 0.01611 | 2.28 | 3.71 | 5.79% |

OOS IC was positive for all seeds and exceeded frozen-HGB's three-seed mean of 0.05701. NDCG was similar to frozen E2's 0.54450, but Precision was lower and net active return was negative for all seeds. Ranking signal replicated; the trading gate did not pass. Seed-1 best/last/result SHAs: `35853b3ff263454c92617c6dfe9f569eb158ec0387f43f63275ef9137ca18290`, `c9324cf6e5d88b85ee4bf6f93c98ddc291471904566cc3aa207959e3016edc4b`, `cde7c2eed3f25d82ef3619c9634b24bf00e1f3842a95e7e234b0dfc772e9dcf0`; seed-2: `5703ee8f52f0a77a607679673064fbbbfd3dc5d4995ebc9686d06efa1c1aadd0`, `b2cfb6fecfadf6d49144b451082532cfcbffddd65ec45bb8d3c65af36f6d0604`, `704ae634d4c0a80d5f87e156d837b352994c9e46fc3c96c6da99d4febe07a756`. Each seed's seven formal files matched Drive. Locked 2026 data were not used. Decision remained `EXTEND`: add exposure and another window before a small ranking-loss or cost-aware checkpoint experiment; keep `probe150m` paused.

## 2026-08-19: Adjacent rolling-fold 100M seed 0

`fold-54-oos-202511` trained July–September 2025, validated October, and tested November. A short-recovery checkpoint continued formal training; epoch 11 was best on H5 validation IC, and patience 4 stopped at epoch 15. Recovery took 11,168.5 seconds. Source `222facb9f643006f0fa8647bb0cee8b7ab4b9306`; data fingerprint `596daa34cfe2a44ad94f884db95d9ce164fd6aff38e05fad61ce1869cc8e9403`.

| Target | Validation IC | Validation IC IR | Validation extreme spread | OOS IC | OOS IC IR | OOS extreme spread |
|---|---:|---:|---:|---:|---:|---:|
| H5 | 0.08735 | 0.71629 | 0.01780 | 0.03305 | 0.29471 | -0.00512 |
| H3 monitor | 0.04840 | 0.35953 | 0.00575 | 0.04231 | 0.36987 | -0.00259 |

H5 had 4,683 validation and 5,807 OOS samples; H3 had 5,464 and 6,578. All four ICs were positive, while both OOS extreme-group spreads were negative. The adjacent fold repeated a cross-sectional ranking signal, but head-portfolio construction still needed work. Six files (best/last checkpoints, result, history, run summary, and materialization preflight) matched Drive and local copies. Best SHA `d753041016d71e668a46624585f7bc7fb68f67200fee3c5b463c28b26f1c11cd`; last `866c0be48277c0f97afaf5ceb87e691bf78e2c8522e15df88b6b5106fdcb8e03`; result `8f073c0c64fad23fb73bfe23baa169046a9184247f5f4909e0b683ed5f88bbf8`. Locked 2026 was absent. Decision: reuse existing predictions for half-life, staggered H5 cohorts, smoothing, entry thresholds, and risk attribution before deciding on seeds 1–2.

## 2026-08-19: External L2 project comparison

Classified suggestions in the [external comparison](external-l2-research-comparison.md) as repository facts, external-project claims, mechanism hypotheses, or decisions. Corrected the event-stream label description: formal H5 is continuous stock return minus concurrent CSI All Share return, applied as a daily scalar at valid event positions. No risk-neutral residual-z label was present. A scalar multi-task loss does not reveal shared-backbone task strength; use gradient norms and angles.

The validation queue was `EVT-HALFLIFE-001`, `TRD-STAGGERED-H5-001`, `TRD-RANK-EMA-001`, `RISK-ATTR-001`, `EVT-GRAD-AUDIT-001`, `EVT-LABEL-SCALE-001`, and `EVT-SUPERVISION-POSITION-001`. Reuse predictions for the first four; change one training mechanism at a time for the last three. Keep `probe150m` paused.

## 2026-08-19: Event-stream half-life and trading conversion

Stock-level predictions for adjacent-fold seed 0 matched the materialization sidecar row by row (4,683 validation, 5,807 OOS); checkpoint SHA `d753041016d71e668a46624585f7bc7fb68f67200fee3c5b463c28b26f1c11cd`. The recent-fold joint-model seed 0 contributed 6,963 validation and 8,125 OOS rows. These differ structurally: one is a direct H5 head, the other an H5-pretrained backbone with minute H1 head, so portfolio differences are not a pure same-model rolling replication.

`EVT-HALFLIFE-001` used shared H1–H10 label sidecars. November OOS IC rose from -0.00131 (H1) to 0.03301 (H5) and 0.06571 (H10); Top-100 active returns were -7.63, -45.34, and -39.16 bp. December ICs were 0.02043, 0.09956, and 0.10191, with active returns 16.09, 111.05, and 214.23 bp. Neither window showed clear IC decay through H10, but head returns did not repeat across windows.

`TRD-RANK-EMA-001` selected a 0.5 rank EMA, a 5 bp expected-return hurdle for turnover, and no absolute entry threshold from 27 October-validation rules. November turnover fell 36.79%→13.90%, while net active return moved -12.80→-13.02 bp. December turnover fell 43.45%→8.90%; net active return improved -23.77→-10.99 bp. Both remained negative. A validation-calibrated zero-profit entry threshold produced all cash in the next fold and failed transfer.

`TRD-STAGGERED-H5-001` opened a 20%-capital H5 cohort daily, yielding 20% one-way turnover. Raw ranks returned -13.90 bp in November and +14.34 bp in December after fixed costs; EMA plus hurdle returned -15.48/+17.12 bp. Repricing with square-root impact on 100m capital and 20-day average volume gave -15.61/+17.06 bp, still opposite in sign.

Size, liquidity, and volatility attribution completed. Selected-rule mean z exposures in November were 0.475, 0.374, and 0.011; December 0.567, 0.282, and -0.495, all within one standard deviation. Dated industry classification was unavailable. The top five dates contributed 63.56% and 51.04% of absolute active volatility, above the 50% gate. Decision `HOLD`: turnover passed, but cross-window positive net active return, at least 1 bp improvement in both windows, date dispersion, and industry attribution did not all pass. Pause adjacent-fold seeds 1–2 and `probe150m`; proceed with gradient, label-scale, and supervision-position tests. See the [diagnostic report](eventstream-signal-trading-diagnostics.md).

## 2026-08-19: Event-stream multi-task gradient audit

`EVT-GRAD-AUDIT-001` selected 16 evenly spaced fixed batches of eight samples from November 2025 validation in the recent fold and October validation in the adjacent fold. For seed-0 initialization and each best checkpoint, it measured four tasks' gradient norms, norm ratios, and pairwise cosine similarities over 100,584,960 shared-backbone parameters; output heads were excluded.

| Fold | Initial day-task gradient ratio, median | Best-checkpoint ratio, median | Best epoch | Persistent negative task pair across batches |
|---|---:|---:|---:|---|
| Recent | 0.61608 | 0.01969 | 4 | None |
| `fold-54-oos-202511` | 0.65568 | 0.03927 | 11 | `reg__day`, `stream__day` |

Both trained day-task ratios were below the pre-registered 0.1 gate. Conflict occurred in the adjacent fold but did not repeat in the recent fold, so no cross-fold task-weight change was justified. Decision: `day_gradient_weak`; next run `EVT-LABEL-SCALE-001`. Result fingerprints: `2fd3064238b10476a2ddb2a5e54a5155e77b78ab369b7126377866770eb28ccd`, `7ec93b77258d108b673992cd1776e28d652b146cb425a60c8be15e9181bcfe12`; cross-fold decision `9bdef3aad8f9be28f80b0236bfc90f093ce1f3b509d03afaf486f136e1140bbf`; source `3e28f04755a881cb72697db2fc50bba031c9f5b0`. OOS and locked 2026 were not accessed.

The label-scale experiment changed only H5 targets in train: winsorize each valid daily cross-section at median ± five raw MADs, then standardize by its winsorized mean and population SD. Boundary samples without H5 labels remained masked by `day_valid=0`. Event windows, validation, OOS, and H3 monitoring labels were unchanged. Begin with seed 0 in both folds; add seeds 1–2 only if validation, OOS, and extreme-group spread all improve in both.

## 2026-08-20: Event-stream label-scale seed-0 two-fold result

`EVT-LABEL-SCALE-001` completed seed 0 in both folds. Only train used daily cross-sectional z labels; validation, OOS, H3 monitoring, windows, model capacity, and task weights were unchanged. Training identity used source `1b4c0f163d1f0aab2c930468f20eecaafe8b60f3`.

| Fold | Train label | Best epoch | Validation IC | OOS IC | Validation extreme spread | OOS extreme spread |
|---|---|---:|---:|---:|---:|---:|
| Recent | Raw H5 return | 4 | 0.04345 | 0.05879 | -0.38744% | 0.34105% |
| Recent | Daily cross-sectional z | 4 | 0.11747 | 0.07446 | 1.27275% | 0.08885% |
| `fold-54-oos-202511` | Raw H5 return | 11 | 0.08735 | 0.03305 | 1.78031% | -0.51240% |
| `fold-54-oos-202511` | Daily cross-sectional z | 8 | 0.13534 | 0.07755 | 2.68055% | 0.48367% |

Validation and OOS IC improved in both folds; all four H3 monitoring ICs improved. The adjacent-fold H5 OOS extreme spread changed from negative to positive. Recent-fold H5 OOS spread remained positive but fell versus raw labels. During the adjacent-fold recovery, Colab reclaimed the VM while the task still appeared active; epochs 2–4 had not reached Drive. PR #91 added event-stream checkpoint synchronization every 180 seconds; PR #92 allowed new scheduling code to resume strictly signed old checkpoints. Training resumed from epoch 1, peaked at epoch 8, and stopped at epoch 12. Scheduler source `35f90d722e6dacd98cd9d0608d6fa3c3c7737b3e`; best, last, history, result, and summary synced locally and to Drive.

The pre-registered gate required validation, OOS, and extreme spread to improve in both folds. Recent-fold OOS extreme spread did not pass, so seeds 1–2 were not run. Decision: `EXTEND_TO_SUPERVISION_POSITION`. Using all-position z labels as the comparator, test only last-position and tail-weighted supervision; first select on seed-0 validation in both folds, then expose OOS for the selected mode. See the [label-scale report](eventstream-label-scale.md).

## 2026-08-22: Event-stream supervision-position seed-0 result

`EVT-SUPERVISION-POSITION-001` reused recent-fold z-label `all` results and trained only `last` and `tail_weighted`, both seed 0 at source `41290ff056fb318d37ce44ba89bcbf31453c07f3`. Data fingerprint: `5a7d9216c7b4a8f680ef8a22ca760b482b6ccd38f6a8df587bd7deb44f445314`; z-label fingerprint: `7f8223c0581e08115b18c19e36756e8431e2e2b395231e5a047860eb5ae53832`.

| Mode | Best epoch | Validation IC | Extreme spread | H3 monitoring IC | Epochs | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `all` | 4 | 0.11747 | 1.27275% | 0.09230 | — | — |
| `last` | 7 | 0.07802 | 0.92692% | 0.06055 | 11 | 7,723.7 s |
| `tail_weighted` | 7 | 0.11289 | 1.10797% | 0.08667 | 11 | 7,579.4 s |

Relative to `all`, `last` fell by 0.03945 IC and 0.34583 percentage points of extreme spread; `tail_weighted` fell by 0.00458 and 0.16478 points. Both failed the requirement to improve in both folds, so the experiment stopped without running the adjacent fold. OOS and locked 2026 remained inaccessible. Best/last/result SHAs for `last`: `6148eb9d4b8d83134c625e7af0570069f733e852ffd5bbc33dddcb7aecf26b5b`, `eb14826d9080cff3c466334735fc584adb921af17faa66c93a0b382fe7050f7e`, `28d1420ae275ee28c02d255afa03fbfd5bb6495d2710363bb34adea7622a0c04`; for `tail_weighted`: `3f6a4a5956631d85f74aae43ea7a3b13a61015bcca8e86306ca8879418a0a819`, `b74b141603667680afe8717e874b5a2fd0f08f3025eb6e4cf0e37028b250b2e2`, `b463d172fc470abf63eb3dd4fedd6c41c3f01d219925e3786069fe9ffb5e4fdd`. Each six-file result directory matched Drive. Decision: `KEEP_ALL`; continue supervising all valid positions. Next test only day-task loss weight; consider cost-aware ranking only if it also fails to show stable increment. Keep `probe150m` paused. The contract and artifact identities are in the [label-scale report](eventstream-label-scale.md).
