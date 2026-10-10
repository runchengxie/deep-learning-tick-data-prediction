# Multi-Horizon Labels and Data Expansion Roadmap

Reading update, 2026-10-10. This record preserves the plans and evidence available at its research stage. References to upcoming seed-0 training, M3 materialization, gradient checks, or label-scale work are historical plans, not the current task list. Later experiments and formal diagnostics are recorded in [Project status](../project-status.md). Follow that entry point for current decisions and locked-period rules.

## Current decision

As of 2026-08-10, the five-year Top-400 raw-200 working set was complete: 470,815 stock-day samples occupying about 7.2 GiB. Seeds 0–2 of the 1,033,383-parameter model completed training on the 2024 validation period. Best daily Rank IC values were 0.02145, 0.02054, and 0.01893, respectively; the mean was 0.02031 and the sample standard deviation across seeds was 0.00127. The 2025 test period remained locked.

Google Drive had been upgraded to 200 GB, enough for raw-200, multi-horizon labels, raw-1000, and experiment artifacts. Stage B passed: on 2024 validation, the three one-day models had mean H=5 IC of 0.07750, ensemble IC of 0.08264, Newey–West t-statistic 3.11, positive IC in 83.3% of months, and a minimum IC of 0.07324 across five non-overlapping groups. A standalone H=5 seed 0 did not show a stable gain over the one-day model. H=1 therefore remains the primary target, with H=3 and H=5 used for monitoring.

The 2026-08-11 benchmark measured 21.13 samples/s on T4 and 80.23 samples/s on A100, a 3.80× speedup. Peak reserved memory was 2.40 and 2.35 GiB, respectively. The capacity gate passed, and A100 was selected for formal 100M training. The five-year raw-1000 Top-100 pilot was generated and uploaded with 118,078 samples, including 70,805 training samples from 2021–2023. The A100 batch sweep selected physical batch 32.

As of 2026-08-16, a controlled 2×2 attribution matrix using a fixed Top-100 sample set and training contract was complete, with three seeds per cell. The 2024 validation daily Rank IC results were:

| Model capacity | raw-200 | raw-1000 |
|---|---:|---:|
| 1M | 0.03748 ± 0.00096 | 0.03530 ± 0.00241 |
| 100M | 0.02740 ± 0.00412 | 0.03152 ± 0.00287 |

Increasing capacity from 1M to 100M reduced the mean by 0.01008 for raw-200 and 0.00377 for raw-1000. Increasing the window from raw-200 to raw-1000 reduced the 1M mean by 0.00219 and increased the 100M mean by 0.00413. Averaged across the other factor, the capacity main effect was -0.00693, the window main effect +0.00097, and the interaction +0.00631. The interaction reflects a partial offset of the 100M capacity penalty by the longer window; `100M/raw-1000` still trailed the best `1M/raw-200` cell by 0.00596. Under the current contract, do not expand model size or event window further. The next gate is to freeze aggregation across `1M/raw-200` seeds, checkpoints, and one-time test acceptance criteria. The 2025 test remains locked.

## Immutable target contract

```text
Signal: final N snapshots before 14:55 on day T
Entry: open on trading day T+1
Exit: close on trading day T+H, H ∈ {1, 3, 5}
Target: stock return minus CSI All Share return over the same entry and exit times
Classification: 20th/80th cross-sectional percentiles of samples available on T
```

The label sidecar stores `entry_date` and `return_end_date`. A sample is eligible only when `trading_date`, `entry_date`, and `return_end_date` all fall within the same train, validation, or test interval. H=1 reuses the existing manifest target and return as a backward-compatible control. H=3 and H=5 are generated from the same daily bars and benchmark prices.

Each sidecar is bound to the source feature `dataset_fingerprint`. A change to the return contract, quantiles, stock sample, or source data requires a new output directory; never overwrite old labels. The feature NPY shards remain unchanged, avoiding another copy of the 7.2 GiB working set.

## Stages and gates

| Stage | Action | Status | Result |
|---|---|---|---|
| A. Label sidecar | Generate H=1/3/5 Parquet labels and a versioned JSON contract | Complete | H=1 matches the old manifest; boundary purge passes |
| B. Fixed-model evaluation | Evaluate existing three checkpoints on 2024 validation IC@1D/3D/5D | Complete | H=5 monitoring gate passed |
| C. Standalone H=5 model | Keep raw-200 and 1M architecture; change only the target | Seed 0 complete | No stable gain; retain H=1 as primary target |
| D. raw-500 | Try Top-100, then Top-400 | Skipped | Use raw-1000 directly as a boundary experiment |
| E. raw-1000 | Generate Top-100 and complete capacity matrix | Complete | Longer window had no stable gain; do not expand to Top-400 |
| F. Full-day tick pilot | Measure size, throughput, and random reads | Moved to event-stream track | Packing and input benchmark complete at this stage; later three-seed training is recorded in Project status |

Capacity attribution used a 2×2 matrix with the same stock sample, dates, targets, optimizer, and evaluation contract. `1M/raw-200` was the control; `100M/raw-200` changed capacity only; `1M/raw-1000` changed window only; `100M/raw-1000` changed both. All four cells completed three seeds. The other three cells reused the raw-1000 Top-100 working set used by `100M/raw-1000`; the raw-200 view selected the final two 100-snapshot chunks per sample, avoiding a second working-set copy. All results shared the same fingerprint, 70,805 training samples, 23,472 validation samples, and 241 valid validation days. The matrix selects `1M/raw-200` as the sole candidate for the next gate. Earlier `1M/raw-200` results using a different stock sample are not used for capacity attribution.

Stage B accesses 2024 validation only. Evaluate the 2025 test once, after horizon, return handling, architecture, and seed list are frozen. Since five-day labels overlap, report daily results, one-in-five non-overlapping samples, monthly results, and uncertainty from Newey–West or block bootstrap methods.

## Recent execution commands

On the remote host, after syncing the code, generate H=1/3/5 labels:

```bash
.venv/bin/ticknet-nextday-prepare-horizon-labels \
  --manifest data/nextday-raw-200/manifest.json \
  --basic-root $QUANT_DATA_ROOT/raw/cn_a_share_level2/basic \
  --benchmark-path $QUANT_DATA_ROOT/reference/cn_market_reference/csi_all_a_000985.CSI.parquet \
  --output-dir data/nextday-raw-200-targets-v1 \
  --horizons 1 3 5 \
  --min-cross-section 100
```

Expected output is well below 1 GiB. After audit, upload labels without re-uploading the NPY features:

```bash
rclone copy data/nextday-raw-200-targets-v1 \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200-targets-v1 \
  --checksum --transfers 4 --checkers 8 --progress
```

Select a sidecar in the training configuration with:

```yaml
target_sidecar_path: ./data/nextday-raw-200-targets-v1/horizon-labels.json
target_horizon: 5
```

Stage B does not retrain for H=3 or H=5. It evaluates existing one-day checkpoint scores against alternate validation targets to plot IC decay. Stage C uses the training configuration above. Its seed-0 entry point is:

```bash
python scripts/run_colab_nextday.py \
  --workflow h5-train \
  --seeds 0 \
  --keep-on-failure \
  --session ticknet-h5-seed0 \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m-h5/seed0
```

Use `configs/nextday-raw-200-capacity-1m-h5.yaml`, changing only the target horizon and isolated artifact directory. Run seeds 1 and 2 only if seed 0 shows gains in 2024 validation H=5 IC, monthly stability, and cost-adjusted Top-K returns. Keep 2025 locked.

## Drive capacity gate: 200 GB to 400 GB

Estimate peak usage as:

```text
existing formal data + new working set + upload/verification copies + checkpoints/results + 25% safety margin
```

For the 200 GB plan, keep stable usage below 120 GB and planned peak below 150 GB. Upgrade to 400 GB before generating or uploading the full dataset if any of these hold:

- Planned peak exceeds 150 GB.
- A one-month full-day tick pilot exceeds 20 GB, implying over 100 GB for five months.
- Two formal working sets over 60 GB each must be retained concurrently.
- Other Drive data leaves less than 50 GB available to the project.

Raw-1000 was expected to require 35–40 GiB. With raw-200, labels, and checkpoints, the 200 GB plan had adequate headroom. The 400 GB decision point is after measuring one month of full-day ticks and before generating five months.

## Colab Pro+ gate

Base the resource decision on continuous runtime per seed, not the sum across seeds:

| Projected time per seed at batch 100 | Resource decision |
|---:|---|
| Under 8 hours | Keep current Colab plan; run seeds separately and save checkpoints |
| 8–12 hours | Prefer A100 or pay-as-you-go compute units; Pro+ is optional |
| 12–20 hours | Pro+ may help; require resume support and isolated outputs |
| Over 20 hours | Do not depend on one Colab session; use a dedicated cloud GPU or local RTX 5090 |

Top-400 1M raw-200 took about 51–68 minutes per seed, so Pro+ was unnecessary. The fixed Top-100 matrix took 13.75–15.54 minutes for three seeds of 1M raw-200. Benchmark raw-1000 at batch 100 first; reconsider the subscription only if one seed projects to over eight hours. Pro+ does not guarantee access to A100.

## raw-1000 Top-100 and 100M benchmark

Create an isolated single-month preflight:

```bash
.venv/bin/ticknet-nextday-prepare-snapshot \
  --config configs/nextday-raw-1000-preflight.yaml
```

Source snapshots arrive about every three seconds. The raw-200 window beginning at 14:30 usually contains only 500 events, so raw-1000 scanning begins at 13:30 and retains only the last 1,000 valid events before 14:55. The earlier scan start is only an extraction boundary; extra events do not enter the sample.

Audit `manifest.json`, `data-audit.json`, shard sizes, and checksums before uploading to:

```text
gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-1000-preflight-202101-top100
```

Run `scripts/run_colab_nextday.py --workflow capacity-benchmark` separately on T4 and A100. Each run uses five warmup batches and 100 measured batches and shuts down its ephemeral runtime afterward. Compare actual GPU name, samples/s, peak reserved GiB, and projected hours per seed for 75,000 samples. If the estimate is under eight hours with at least 20% memory headroom, generate the five-year `configs/nextday-raw-1000-top100.yaml`; otherwise reduce effective batch size or model width first.

Formal training used `configs/nextday-raw-1000-top100-capacity-100m.yaml`, selecting checkpoints by 2024 validation daily Rank IC and early stopping with patience 8. Best epochs were 9, 20, and 12; runs lasted 17, 28, and 20 epochs. Full metrics, fingerprints, source revisions, and interpretation are in [experiment-log.md](../research/experiment-log.md).

The remaining three matrix cells were trained through `scripts/run_colab_nextday.py --workflow capacity-matrix-train`, using `--matrix-cell` values `1m-raw200`, `1m-raw1000`, and `100m-raw200`. All three configurations used batch 32, learning rate 0.0001, patience 8, seeds 0–2, and 2024 validation checkpoint selection. The nine new runs used source revision `e2465c0`; the existing `100M/raw-1000` seeds used a compatible frozen contract. Per-seed metrics, effect decomposition, and interpretation are in [experiment-log.md](../research/experiment-log.md). The 2025 test remained locked throughout.

## Deliverables

- `horizon-labels.json`: contract, source feature fingerprint, horizons, row counts, and Parquet SHA-256.
- `labels.parquet`: stock, signal date, entry date, exit date, horizon, return, and class label.
- Three-seed IC@1D/3D/5D summary, monthly results, and non-overlapping-sample results.
- Per-stage raw-500/raw-1000 audits, 100-batch benchmarks, and continue/stop decisions.
- Measured size of a one-month full-day tick pilot and the projected five-month peak.

Keep real market data, label Parquet, checkpoints, and notebook HTML in artifact storage or Drive. Git contains only code, configuration, aggregate reports, and audit summaries without per-stock records.
