# M3 Top-K, Buffer, and Cost-Matrix Diagnostics

## Current conclusion

M3 v2 completed formal minute-feature materialization, HGB training, prediction registration, and a 64-cell Top-K cost matrix. HGB daily Rank IC was 0.06994 in the second half of 2025, with positive monthly IC in all six months. The matrix used the same 124 evaluation days, exactly 400 candidates per day, and explicit cannot-buy, cannot-sell, and out-of-universe holding states.

Formal decision: `NO_TRADEABLE_REGION`. At 10 bp one-way cost plus 5 bp sell stamp duty, all 16 K/buffer candidates had negative cost-adjusted active return. `K=100, buffer=50` had the highest absolute net return: 12.15 bp daily and net Sharpe 1.41. The Top-400 equal-weight benchmark did better; this portfolio's mean daily net active return was -4.75 bp, with only one positive month out of six.

Across all strategies, the highest break-even one-way cost for active returns was about 4.33 bp (`K=100, buffer=50`), below the 10 bp decision cost. This portfolio's absolute-return break-even cost was about 24.51 bp, but that includes the broad market's rise and is not standalone evidence of predictive alpha. Buffers reduced turnover and improved absolute net return, but did not produce a stable gain over the equal-weight benchmark.

An earlier Top-100 smoke run also yielded `NO_TRADEABLE_REGION`, but lacked formal trading states and a Registry fingerprint; retain it as engineering history only.

## Diagnostic input contract

`topk_cost_sweep` accepts exactly one of two sources:

- `predictions_path`: development smoke tests only. The output records source-file SHA-256 but has no Registry data fingerprint.
- `source_experiment_id`: select a unique seed and artifact from Registry. Before reading, verify state, path, SHA-256, and locked-date boundary, then materialize the file into this run's isolated seed directory. Propagate the source experiment's fingerprint to M3.

`evaluation_mode: formal` additionally requires the Registry source, `require_tradability: true`, `require_universe_membership: true` with daily counts checked against `expected_universe_size`, `missing_holding_policy: error`, `target_return_contract: next_open_to_following_open`, and a buffer grid containing 0 as a fixed comparison.

Formal mode validates prediction content and metadata instead of trusting prose in the upstream spec. `import_predictions` validates Parquet and metadata before registering it as a Registry prediction artifact. `topk_cost_sweep` rechecks the Registry `dataset_fingerprint` at consumption time. Required metadata:

- `ticknet.dataset_fingerprint`
- `ticknet.target_return_contract=next_open_to_following_open`
- `ticknet.universe_contract=lagged_turnover_top_n`
- `ticknet.tradability_contract=next_open_suspension_one_price_limit`
- `ticknet.suspended_mark_policy=previous_close`

Required columns are `symbol`, `trading_date`, `label_date`, `return_end_date`, `target_return`, `score`, `can_buy`, `can_sell`, and `in_universe`. `return_end_date` must be after `label_date`, and each `label_date` maps to one return-end date. Every date has exactly `expected_universe_size` rows with `in_universe=true`. Out-of-universe state rows may record whether a removed holding can be sold next day. Audit, ranking, and universe benchmarks ignore state rows; the portfolio state machine uses them to exit or retain existing holdings.

First register a formal prediction:

```yaml
experiment_type: prediction_export
executor: import_predictions
inputs:
  predictions_path: results/predictions-hgb-top400-open2open.parquet
  evaluation_mode: formal
  target_return_contract: next_open_to_following_open
  expected_universe_size: 400
```

## Formal input generation

`configs/nextday-minute-formal-2025-v2.yaml` fixes formal HGB inputs to July 2021–2024 training, first-half 2025 validation, and second-half 2025 output. The daily universe uses turnover from the prior 20 trading days and retains exactly 400 stocks. Model supervision is stock T+1 open-to-T+2 open return minus the concurrent benchmark; prediction `target_return` stores the unadjusted stock holding return for portfolio accounting. Suspensions are marked using the latest valid prior close; one-price limit-up is unbuyable and one-price limit-down is unsellable. Candidates without minute windows remain with all-NaN features for HGB's missing-value branch and are marked `feature_available=false`.

The v1 configuration began in January 2021. When materialization resumed, daily order files from 2021-01-04 through 2021-06-04 contained only Shenzhen tickers beginning with `0` or `3`, not Shanghai tickers beginning with `6`. The first 101 trading days lacked Shanghai orders; both markets appeared from June 7. Snapshot and trade sources had Shanghai coverage, so strict three-modality alignment left Shanghai candidate features empty. The missing source orders cannot be recovered. v1 stopped at 14 of 60 months, retaining its directory and manifest as audit evidence.

Materialize the approximately 110 GB source cache by month into resumable aggregate features:

```bash
uv run python scripts/materialize_minute_features.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --output results/m3-formal-minute-features-v2-202107
```

Each month enters the manifest only after atomic Parquet write and SHA-256. Re-running verifies and skips complete months; use `--period YYYY-MM` for one-month diagnostics. The manifest binds v2 target stock-days, the ordered 30 raw minute features, window parameters, and size/mtime for 15 annual three-modality source files. Identity drift blocks resume. The formal fingerprint hashes loaded 120-dimensional aggregate features, labels, and tradability state values.

After all 54 months, run HGB:

```bash
uv run python scripts/run_minute_baseline.py \
  --config configs/nextday-minute-formal-2025-v2.yaml \
  --materialized-features results/m3-formal-minute-features-v2-202107 \
  --evaluate-test \
  --save-predictions results/predictions-hgb-top400-open2open-2025-v2.parquet \
  --output results/nextday-minute-formal-2025-v2.json
```

## Formal data evidence

- The v2 manifest is `complete`: 54 shards and 436,800 rows; 436,256 have complete features and 544 remain all-NaN, for 99.88% feature coverage. All shard SHA-256 checks passed; there are no temporary or partial files. Materialization took 4,240.2 seconds and peaked near 1.71 GB RAM.
- Splits contain 339,600 training, 46,000 validation, and 49,600 test samples. Validation Rank IC is 0.08091; test Rank IC 0.06994, Macro F1 0.32129, and MCC 0.13208.
- Prediction has 51,489 rows: 49,600 candidates and 1,889 state rows. Each of 124 dates has 400 candidates; 863 rows are unbuyable and 849 unsellable. Prediction SHA-256: `bdeb2cbe7de8b894fd246ff56c31e49e510c0c38eac115687047c096a9a2a45d`.
- Formal data fingerprint: `6ca055086c8885bcb866da01af4481a95d58c1334ed3fe0b574cca5b66dcbb7a`. Predictions are registered as `PRED-HGB-400-OPEN2OPEN-001`; the cost matrix as `TRD-TOPK-400-001`.
- The v1 daily panel spans 2021-01-04 to 2025-12-29: 1,210 complete signal days with 400 candidates each, 484,000 candidate labels total. Universe coverage was complete; early order-source market coverage was not.
- A further 13,329 out-of-universe state rows were generated. Candidate and state rows record 3,282 suspensions, 264 one-price limit-ups, and 209 one-price limit-downs.
- A single-day extraction for 2025-07-01 requested 400 candidates: 399 had complete three-modality minute rows and one used the all-NaN path. It read 6 relevant row groups and skipped 835 unrelated groups using date metadata.
- Materialization completed July–December 2025: six months and 49,600 candidates, with 49,408 feature-complete and 192 all-NaN rows (99.61% coverage). Monthly missing counts were 7, 33, 15, 112, 11, and 14. Do not silently shrink Top-400 because October 2025 has more missing features.
- L2 extraction changed from per-minute Python objects to contiguous stock-day arrays with vectorized deduplication, tail selection, and three-modality alignment. An independent July 2025 rerun produced identical Parquet SHA-256; runtime fell from 100.3 to 72.4 seconds and peak memory from about 2.26 to 1.69 GB. Other months took 62.0–82.6 seconds, with batch peaks under 1.72 GB. The cumulative manifest peak remains 2.26 GB from the older implementation, consistent with recording the full-run maximum.
- v1 manifest remains `in_progress` at 14/60 months: 114,400 rows, of which 94,573 have features and 19,827 are all-NaN. All 14 shards passed identity, SHA-256, row count, date boundary, and Parquet metadata checks; no partial files remain.
- v1 missing counts for January–May 2021 were 3,969/8,000, 2,911/6,000, 4,447/9,200, 4,061/8,400, and 3,437/7,200. June missed 786/8,400; July and August missed 17/8,800 and 7/8,800. The discontinuity matches Shanghai order files beginning June 7.
- Formal HGB rejects incomplete manifests and lists missing months. Re-running a month verifies identity and shard SHA-256 before skipping it.
- Daily panels are explicitly capped at 2025-12-31. Prediction contracts add `return_end_date`, preventing 2025 samples from using locked 2026 returns.

Together, these checks establish executable data semantics, resource limits, resumability, and missing-data behavior. v2 avoids v1's early order-coverage gap and supports the formal M3 conclusion.

## Tradeable-region criteria

By default, evaluate each K/buffer at 10 bp one-way cost. A candidate must satisfy all conditions:

1. At least 60 fully comparable evaluation days.
2. Net Sharpe above zero.
3. Mean daily return after this strategy's costs exceeds the same-day equal-weight universe return before costs.
4. Cost-adjusted daily excess return is positive in at least half of months.
5. The five largest absolute gross-excess-return days contribute no more than 50% of total absolute gross excess return.
6. A non-zero buffer lowers turnover versus buffer 0, and saved costs cover the gross-return change.

Criterion 3 uses a pre-cost equal-weight benchmark and is intentionally strict. If a truly tradable no-signal portfolio is implemented later, compare it under identical states and costs while retaining this gate so benchmark turnover cannot hide weak alpha.

Each K/buffer also reports two analytic cost thresholds:

- `absolute_return_breakeven_per_side_bps`: one-way cost where absolute net return reaches zero.
- `active_return_breakeven_per_side_bps`: one-way cost where return versus equal-weight universe reaches zero.

The second tests whether predictive value covers trading costs. The first is affected by broad market direction and is not standalone evidence of model alpha.

## Artifacts and formal spec

Each run writes `source-predictions.parquet` (materialized input after SHA-256 verification), `topk-sweep.json` (M1 summaries plus M3 diagnostics), `m3-diagnostic.json` (matrix, thresholds, candidate ranking, break-even costs, source identity, final state), and per-combination `topk/k*.buffer*.cost*/` summaries, daily rows, holdings, and trades.

```yaml
experiment_type: cost_analysis
executor: topk_cost_sweep
inputs:
  source_experiment_id: PRED-HGB-400-OPEN2OPEN-001
  source_seed: 0
  artifact_name: predictions
  evaluation_mode: formal
  target_return_contract: next_open_to_following_open
  top_k: [25, 50, 75, 100]
  exit_buffer: [0, 10, 25, 50]
  cost_bps: [5, 10, 15, 20]
  decision_cost_bps: 10
  sell_stamp_tax_bps: 5
  min_symbols_per_day: 400
  require_tradability: true
  require_universe_membership: true
  missing_holding_policy: error
  expected_universe_size: 400
```

Run through the unified entry point:

```bash
ticknet-research \
  --registry results/registry.sqlite \
  --artifacts research/experiments \
  run --spec path/to/m3-formal.yaml --id TRD-TOPK-400-001
```

## 2025 engineering smoke evidence

Input: `results/predictions-rolling-2025.parquet`, SHA-256 `ee2f8c2c4dd1be56c45138f8e6ca5de48539a8a899c5f9ea3dbcc4c15867eeba`. Output: `results/m3-topk-smoke-2025/TRD-TOPK-SMOKE-001/`.

| Metric | Result |
|---|---:|
| Complete portfolios | 64 |
| Comparable days | 91 |
| Tradeable regions at 10 bp | 0 |
| K=25, buffer=50 mean daily one-way turnover | 25.42% |
| K=25, buffer=50 mean cost-adjusted active return | -1.75 bp |
| K=25, buffer=50 active-return break-even one-way cost | 6.55 bp |
| K=25, buffer=0 mean daily one-way turnover | 69.22% |
| K=25, buffer=50 net-return improvement vs buffer 0 | 10.57 bp/day |

Engineering gate `diagnostic.grid.validated_combinations >= 64` passed and Runner returned `EXTEND`. Here `EXTEND` means the diagnostics pipeline is complete, not that the strategy passed. Trading status is `diagnostic.decision.status=NO_TRADEABLE_REGION`.

## Conclusion and next steps

M3 is complete: v2's 54-month materialization, formal HGB, prediction registration, and full cost matrix are done. No candidate passed at 10 bp, so do not run `TRD-BUFFER-400-001`, which depends on a qualifying region. M4 compares HGB with LambdaMART on the same fingerprint, checks whether checkpoint prediction targets mismatch Top-K ranking, and adds NDCG, precision, and risk-exposure diagnostics. Current evidence does not support expensive order-book pretraining or a neural ranking loss.

Formal experiments use separate Registry `results/m3-formal-registry-v2.sqlite` and artifact directory `results/m3-formal-experiments-v2/`. The older `results/registry.sqlite` remains read-only historical evidence.

The existing `results/predictions-rolling-2025.parquet` was rejected by the formal validator because it lacks `can_buy`, `can_sell`, and `in_universe`. This deterministic failure boundary prevents missing tradability data from being disguised as universally tradable input.
