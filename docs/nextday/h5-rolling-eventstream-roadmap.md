# H5 Rank IC and Event-Stream Rolling Roadmap

## Decision summary

This research series trains on five-day returns and selects checkpoints by daily cross-sectional validation Rank IC@5D. IC@3D monitors whether the signal appears earlier; IC@1D is diagnostic only. Each complete rolling fold contains three months of training, one month of validation, and one month of out-of-sample (OOS) evaluation.

2021–2024 form the historical research period. Because 2025 has already informed cost audits, model comparisons, and hypothesis formation, it is visible development and rolling-validation data. 2026 is locked and must not be read until a separate approval and the protocol's full-sample gate are satisfied.

The most recent fold uses:

```text
train       2025-08-01 to 2025-10-31
validation  2025-11-01 to 2025-11-30
OOS         2025-12-01 to 2025-12-31
locked      from 2026-01-01
```

The 2021 fold remains an infrastructure and cross-year throughput baseline. Event density is higher in 2025, so formal runtime estimates use measured throughput on the 2025 pack.

## Fixed experiment contract

```text
Signal time: before 14:55 on day T
Entry: open on T+1
Exit: close on T+H, H ∈ {1, 3, 5}
Continuous target: stock return minus concurrent CSI All Share return
Primary training target: H=5 continuous return, SmoothL1 regression
Auxiliary targets: next-event and order-type cross-entropy
Primary selection metric: daily cross-sectional validation Rank IC@5D
Monitoring: Rank IC@3D, monthly positive-IC share, Newey-West t-statistic, five non-overlapping groups
Diagnostic: Rank IC@1D
```

Rank IC is calculated across stocks within each signal date, then averaged across dates. It is not used directly as the initial loss because random event-window batches do not contain a complete daily cross-section. After establishing an H5 regression baseline, evaluate pairwise or listwise ranking losses with day-level batches.

For H5 leakage control, `trading_date`, `entry_date`, and `return_end_date` must all belong to the same split. Validation selects checkpoints; OOS is reported once per fold. The period January 2021 through December 2025 produces 56 complete folds, ending with `fold-55-oos-202512`. The rolling plan excludes 2026.

## 2025 data audit

Order, trade, and snapshot inputs are complete for August–December 2025, covering 103 trading days. Raw data stays on the Linux data disk and is not uploaded to Drive.

| Month | Trading days | Raw bytes |
|---|---:|---:|
| 2025-08 | 21 | 102,662,250,487 |
| 2025-09 | 22 | 107,672,938,318 |
| 2025-10 | 17 | 77,376,235,803 |
| 2025-11 | 20 | 89,256,118,925 |
| 2025-12 | 23 | 100,099,208,577 |
| Total | 103 | 477,066,752,110 |

Preflight, daily universes, rolling plan, and labels are under `artifacts/eventstream-h5-recent-fold/`. H3 retained 22,058, 6,578, and 7,732 labels for train, validation, and OOS. H5 retained 21,266, 5,807, and 6,966. Labels crossing split boundaries were removed.

The five-month Top-400 pack is complete: 412 files and 313.11 GiB, with daily indexes for all 103 days from 2025-08-01 through 2025-12-31 and no partial files. The August 2025 benchmark pack passed audit; its remote copy was removed on 2026-08-18, while the complete local source remains on the 6 TB data disk. On 2026-08-19, `rclone about gdrive:` reported 200 GiB total, 145.292 GiB used, and 53.305 GiB free. The complete five-month pack does not fit on Drive or a Colab temporary disk.

Formal training therefore uses fixed-window caches. Locally, the cache samples train, validation, OOS, and monitoring windows by configuration and seed, preserving float32 features and original labels. Each seed requires about 25 GiB and fits the available Drive capacity. The cache manifest binds source files, dates, configuration, seed, array specifications, and per-file SHA-256 hashes. `run_colab_nextday.py` supports preflight, checkpoint recovery, failed-run artifact return, and OOS access control.

The first additional window was `fold-54-oos-202511`: July–September 2025 training, October validation, and November OOS. July's three raw streams cover 23 trading days and 92,573,146,416 bytes. The dynamic universe contained 389–397 stocks daily; H3/H5 labels passed split purging. July packing, cache materialization, remote verification, a short resume test, and formal seed 0 training are complete.

Seed 0 used 100,604,180 parameters. Its best H5 validation Rank IC was 0.08735 at epoch 11; early stopping occurred after epoch 15. H5 OOS Rank IC was 0.03305. H3 validation and OOS Rank IC were 0.04840 and 0.04231. The extreme-group H5 return spread fell from 0.01780 on validation to -0.00512 OOS. Follow-up work should examine signal decay, staggered holding periods, and top-group portfolio rules. Dataset fingerprint: `596daa34cfe2a44ad94f884db95d9ce164fd6aff38e05fad61ce1869cc8e9403`. The locked 2026 period was not accessed.

## Stage progress

| Stage | Status | Result |
|---|---|---|
| R0 | Complete | Planned 56 folds from 2021-01 through 2025-12; excluded 2026 |
| D0 | Complete | Five months of raw data, preflight, and universes verified; source manifest fingerprint fixed |
| F0 | Complete | Missing prior closes fall back per stock to the latest valid positive value; tests cover nulls, NaN, and missing columns |
| D1 | Complete | All four files for 2025-08-01 present; random reads and H5 label coverage passed |
| D2 | Complete | Audited and uploaded the 21-day August 2025 pack |
| B0 | Complete | A100 input analysis optimized to 149.40 samples/s; 20 epochs projected at 4.46 hours per seed |
| D3 | Complete | Packed September–December 2025; all 103 daily indexes complete |
| S0a | Complete | Added monthly logical-storage manifest, direct remote-file checks, capacity checks, and on-disk verification; fixed the 103-day window and blocked 2026 |
| S0b | Complete | Three-seed fixed-window caches materialized, uploaded, and passed a real resume test |
| T0 | Complete | Trained and evaluated the most recent fold, seed 0 |
| T1 | Complete | Seeds 1 and 2 complete; all three validation and OOS Rank IC values are positive |
| E0 | Complete | Three-seed frozen embeddings, HGB/LambdaMART comparisons, and joint training complete |
| R1 | Seed 0 complete | Packed and materialized `fold-54-oos-202511`, verified remotely, and completed formal training; H5 validation and OOS Rank IC are positive |
| C0 | Deferred | Await half-life, staggered holding, top-group, and risk-exposure evidence before deciding on `probe150m` |
| L0 | Sealed | Evaluate 2026 once after protocol gates and approval; do not use it to retune this research round |

## Reproducible commands

Create a rolling plan that excludes 2026:

```bash
python -m ticknet.nextday.rolling \
  --start-month 2021-01 \
  --end-month 2025-12 \
  --target-horizon 5 \
  --output artifacts/eventstream-h5-recent-fold/rolling-plan.json
```

Generate H3 and H5 labels for the most recent fold:

```bash
ticknet-eventstream-prepare-horizon-labels \
  --sidecar artifacts/eventstream-h5-fold0/horizon-sidecar/horizon-labels.json \
  --feature-manifest data/nextday-raw-200/manifest.json \
  --output-dir artifacts/eventstream-h5-recent-fold/fold-labels \
  --horizons 3 5 \
  --train-start 2025-08-01 --train-end 2025-10-31 \
  --val-start 2025-11-01 --val-end 2025-11-30 \
  --test-start 2025-12-01 --test-end 2025-12-31
```

Pack a trading day:

```bash
python -m ticknet.eventstream.pack \
  --days 20250801 \
  --universe artifacts/eventstream-h5-recent-fold/202508/universe.json \
  --pack-root $QUANT_DATA_ROOT/derived/l2_eventstream/top400-h5-v1
```

The first day took 311 seconds wall-clock, used 24.8 GiB peak memory, and peaked at 224 KiB swap. The 396-stock universe produced 60,012,903 orders, 33,284,058 trades, and 1,876,631 snapshots; the pack was 2,742,987,628 bytes. The dataset produced 2,000 training samples with labels for all 396 stocks. Three random reads each returned finite `512 × 80` features.

After uploading the August 2025 pack, run an A100 capacity benchmark:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-capacity-benchmark \
  --session ticknet-eventstream-h5-recent-100m-a100 \
  --gpu A100 \
  --benchmark-batches 100 --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/benchmarks/a100
```

## Completion criteria

- Rolling plan, universe, labels, and packs are bound to source fingerprints or deterministic file sets.
- 2025 may inform development; training and tuning paths must not read 2026.
- Validation alone selects checkpoints; OOS must not influence the same research series.
- Report daily and monthly IC@5D, Newey-West statistics, and non-overlapping five-day samples; monitor IC@3D separately.
- Keep real data, stock lists, daily packs, checkpoints, and predictions out of Git. Commit only code, configuration, and aggregate audits.
