# Project Status

This page records project facts verifiable from code, tests, or experiment artifacts as of 2026-08-29. Roadmaps describe research plans, while the experiment log preserves historical figures. This page summarizes status. When status changes, update it first and then check the root README and relevant topic documents.

## Capabilities and progress

### Chip-summary engineering addition (2026-10-07)

The [chip age HGB experiment](research/chip-age-hgb.md) implements lagged daily
cost/age summaries, paired price/chip/combined HGB comparisons, optional verified
minute-feature fusion, and formal cost-evaluation artifacts. Synthetic tests
cover the numerical recurrence, factor adjustment, date isolation, and prediction
contract. No real-data training result is recorded; the older numerical findings
below are unchanged. A published daily asset with verified VWAP, turnover units,
and point-in-time adjustment factors is required before real training.

| Track | Existing capability | Evidence at the status date | Status |
|---|---|---|---|
| FI-2010 reproduction | DeepLOB training, evaluation, text conversion, and tests | `src/ticknet/fi2010/` | Maintained separate research track |
| Raw order book | raw-200 and raw-1000 shards, roughly 1M- and 100M-parameter models, multi-horizon labels, and inference | Controlled four-cell, three-seed Top-100 matrix complete | Candidate narrowed |
| Minute aggregates | HGB, TCN, and GRU, rolling yearly evaluation, prediction audits, and cost evaluation | M3 v2 2025 H2 Rank IC was 0.06994; the formal cost matrix found no viable region | Formal conclusion recorded |
| L2 event stream | Lossless three-stream packing, causal Transformer, training recovery, closing cache, frozen embeddings, joint fine-tuning, multi-task gradient audits, and label overlays | Keep z-score labels; last-position and tail-weighted supervision did not beat all-position supervision | Daily task-weight ablation next |
| AgentX research loop | Proposals, critiques, allowlisted executors, registry, audits, comparisons, and locked-test approval | M0 through M3 complete; M4 and M5 trading-conversion diagnostics implemented | Training-mechanism ablations next |

Automated tests use synthetic data and cover FI-2010, next-day, minute, event-stream, and research-loop functionality. `scripts/smoke_test.py` checks the DeepLOB forward pass, gradients, parameter count, and FI-2010 dataset windows without real data.

## Research findings

Minute HGB had positive daily Rank IC in four historical rolling OOS years from 2022 through 2025, ranging roughly from 0.02 to 0.035. The daily order files from 2021-01-04 through 2021-06-04 contained no Shanghai stocks, while all rolling configurations began training in 2021. These results therefore have an early-market coverage bias, strongest in the 2022 fold. They remain historical engineering baselines; the formal decision uses the M3 v2 rerun below. The historical baseline had about 83% daily turnover, negative net annual return at 10 bp one-way cost, and breakeven cost around 5–6 bp. The minute TCN's validation advantage did not carry through to test.

Validation Rank IC for the raw-order-book Top-100 four-cell matrix:

| Capacity and window | Three-seed mean and standard deviation |
|---|---:|
| `1M/raw-200` | `0.03748 ± 0.00096` |
| `1M/raw-1000` | `0.03530 ± 0.00241` |
| `100M/raw-200` | `0.02740 ± 0.00412` |
| `100M/raw-1000` | `0.03152 ± 0.00287` |

The capacity main effect was `-0.00693`, the window main effect was `+0.00097`, and the interaction was `+0.00631`. Retain `1M/raw-200` as the sole candidate and pause further capacity and window expansion. This conclusion is based on validation; the series' locked test has not been opened.

### Event-stream 100M recent fold

Seeds 0, 1, and 2 completed on the recent fold. Each model has 100,604,180 parameters. Training covers August through October 2025, validation is November 2025, and OOS is December 2025.

| Seed | Best epoch | Validation Rank IC | OOS Rank IC | Training time |
|---:|---:|---:|---:|---:|
| 0 | 4 | 0.04345 | 0.05879 | 82.9 min |
| 1 | 6 | 0.09403 | 0.03730 | 116.3 min |
| 2 | 5 | 0.08029 | 0.03291 | 102.7 min |

The validation three-seed mean was 0.07259 with sample standard deviation 0.02615. The OOS mean was 0.04300 with sample standard deviation 0.01385. Both means and every seed direction were positive, passing the 100M signal gate. H3 monitoring results were also all positive. Checkpoints, histories, results, run summaries, and cache fingerprints remain in local experiment artifacts.

Seed 0 also completed on adjacent fold `fold-54-oos-202511`. Training covers July through September 2025, validation is October, and OOS is November. H5 validation Rank IC was 0.08735 and OOS was 0.03305. H3 monitoring was 0.04840 and 0.04231, respectively, so all four directions were positive. H5 extreme-group return spread was 0.01780 on validation and -0.00512 OOS. This fold further supports a cross-sectional ranking signal while again showing that top-group returns did not improve alongside it. Formal training used source revision `222facb9f643006f0fa8647bb0cee8b7ab4b9306` and data fingerprint `596daa34cfe2a44ad94f884db95d9ce164fd6aff38e05fad61ce1869cc8e9403`. The 2026 locked period was not read.

The local machine has no available CUDA GPU. On 2026-08-19, `rclone about gdrive:` reported 200 GiB total, 145.292 GiB used, and 53.305 GiB free. A shared closing-window cache was generated locally, fully verified, and uploaded. It contains 39,903 stock-days across five shards, totaling 6,619,831,094 bytes (about 6.17 GiB), with fingerprint `59577182c8124c312de0591059c67e55d472511ca77753403ce77afbf8f109f4`. The 960-dimensional embeddings from three checkpoints were exported and verified file by file locally and remotely. Each has 39,903 rows, with identical stock-day keys and order. Together they total 437,941,717 bytes (about 417.65 MiB).

### Frozen embeddings

`FEAT-EMB-FROZEN-001` is complete. E0 uses minute features, E1 uses frozen embeddings, and E2 concatenates the two. The comparison has 22,409 training samples, 6,963 validation samples, and 8,125 OOS samples. Event-stream coverage of the minute candidate is 96.69%. A downstream model is trained separately for each checkpoint, then downstream predictions are averaged. Embedding vectors stay in their respective coordinate spaces.

For HGB, E0 Rank IC was 0.01808 on validation and 0.04010 on December 2025 OOS. E2's three seeds were 0.02462, 0.02988, and 0.02323 on validation, and 0.04333, 0.05644, and 0.05912 OOS. Mean predictions across seeds had Rank IC 0.02833 on validation and 0.05701 OOS. The paired OOS increment was 0.01691; E2 beat E0 on 16 of 21 days, with a daily-bootstrap 95% interval from 0.00596 to 0.02851. `NDCG@100` rose from 0.53424 to 0.54450, while `Precision@100` declined slightly from 0.26810 to 0.26667. Mean daily Top-100 active return after costs improved from -11.51 bp to -4.80 bp but remained negative. Daily one-way turnover declined from 62.89% to 60.91%.

LambdaMART E2 varied substantially across seeds and months. Mean three-seed validation Rank IC changed from -0.04334 to -0.05081, while OOS changed from 0.00766 to 0.01389. E1 mean predictions had OOS Rank IC 0.03414 and daily Top-100 active return after costs of 15.53 bp, but validation active return was -15.34 bp. Current evidence supports retaining frozen representations and the HGB-fusion candidate. Trading value, cross-window stability, and risk attribution need more evidence. Sector, size, volatility, and liquidity exposure inputs were unavailable and marked `unavailable`.

### Joint event-stream training

All three seeds for `FEAT-EVENTSTREAM-JOINT-001` are complete. Its lightweight cache contains 22,409 training samples, 6,963 validation samples, and 8,125 OOS samples. The cache is 17,948,094 bytes with fingerprint `e4f54a62e4be3f36ac0693db59ebcdb120cd753d2dc36415b8686adaa13c1bb6`. It stores only 120-dimensional minute features, targets, and row indices into the shared closing cache; it does not copy the 6.17 GiB event arrays. All five files were checked against the Drive copy. The three seeds agree on stock, date, label, and row count.

Each joint model loaded the matching seed's 100M best checkpoint. Best epochs for seeds 0, 1, and 2 were 2, 1, and 1. Validation Rank IC was `0.05917 ± 0.01400`, and OOS Rank IC was `0.06398 ± 0.00785`; all three OOS values were positive. OOS `NDCG@100` was `0.54507 ± 0.00450`, and `Precision@100` was `0.23921 ± 0.01611`. Mean daily Top-100 one-way turnover was `49.74% ± 5.79%`. Mean daily active return after costs was `-9.67 ± 3.71bp`, negative for all three seeds. Cross-sectional correlation passed the repeatability check, but top-group hit rate and after-cost returns did not meet the trading gate. Defer `probe150m` until there is evidence on additional time windows, risk exposures, and trading-objective improvements.

### Trading diagnostics and training ablations

Event-stream signal-decay and trading-conversion diagnostics are complete. Neither window showed clear IC decay through H10. The October validation-selected ranking EMA of `0.5`, combined with a 5 bp turnover-difference rule, reduced H1 one-way turnover for November and December from 36.79% and 43.45% to 13.90% and 8.90%. After-cost active returns remained -13.02 bp and -10.99 bp. Fixed-cost active returns for five staggered H5 holdings were -15.48 bp and 17.12 bp, opposite in direction across the two windows. Dynamic liquidity costs did not change direction. Size, liquidity, and volatility exposures were within one standard deviation. A dated local sector-classification file was unavailable. Decision: `HOLD`; adjacent-fold seeds 1 and 2 and `probe150m` remain paused.

`EVT-GRAD-AUDIT-001` audited 16 fixed validation batches on both the recent and adjacent folds. Median daily-task gradient ratios on the shared trunk fell from 0.61608 and 0.65568 at initialization to 0.01969 and 0.03927 at the best checkpoint, below the 0.1 threshold on both folds. No task pair had persistent negative correlation on both folds. The formal next step was `EVT-LABEL-SCALE-001`, a seed-0 comparison using daily cross-sectional winsorized z-score labels.

`EVT-LABEL-SCALE-001` completed seed 0 on both folds. On the recent fold, validation and OOS Rank IC increased from 0.04345 and 0.05879 to 0.11747 and 0.07446. On the adjacent fold, they increased from 0.08735 and 0.03305 to 0.13534 and 0.07755. Adjacent-fold OOS extreme-group return spread improved from -0.51240% to 0.48367%, while recent-fold spread declined from 0.34105% to 0.08885%. The preregistered requirement for improvement in all metrics across both folds did not pass, so seeds 1 and 2 were not run.

`EVT-SUPERVISION-POSITION-001` completed recent-fold seed-0 validation. Rank IC for `last` and `tail_weighted` supervision was 0.07802 and 0.11289, below 0.11747 for `all`. Their extreme-group spreads were 0.92692% and 1.10797%, also below 1.27275% for `all`. Both candidates failed the requirement to improve on both folds at the first fold, so the adjacent fold stopped and OOS remained closed. Formal decision: `KEEP_ALL`. The next step is to test daily task weighting. Full results are in the [event-stream label-scale study](research/eventstream-label-scale.md).

### AgentX M3 and trading conversion

AgentX M3 v1 completed 14 of 60 months. Newly added shards from January through May 2021 had about 48–50% of candidates missing all three feature modalities. An audit of 118 daily order files from the first half of 2021 confirmed that the first 101 trading days contained only Shenzhen stocks. Shanghai stocks first appeared on 2021-06-07. Existing raw files cannot recover the missing coverage. Keep the v1 result directory as an audit record.

M3 v2 starts in July 2021 and materialized 436,800 candidates across 54 months, with 99.88% complete-feature coverage. HGB validation Rank IC was 0.08091 in H1 2025. Rank IC was 0.06994 across 124 evaluation days in H2 2025, with positive monthly IC in all six months. The formal prediction is `PRED-HGB-400-OPEN2OPEN-001`, with data fingerprint `6ca055086c8885bcb866da01af4481a95d58c1334ed3fe0b574cca5b66dcbb7a`.

`TRD-TOPK-400-001` completed a 64-cell formal matrix and returned `NO_TRADEABLE_REGION`. At 10 bp one-way cost, the best absolute net return was for `K=100`, `buffer=50`: daily net return 12.15 bp and net Sharpe 1.41. However, daily net active return versus the Top-400 equal-weight benchmark was -4.75 bp, and only one month was positive. The highest breakeven one-way cost among portfolios was about 4.33 bp, below the 10 bp decision cost. Positive absolute return includes the broad market's concurrent rise and cannot be attributed to model alpha alone.

The raw-order-book four-cell matrix is no longer expanding. Event-stream experiments do not change that decision.

## Separate date-access protocols

The repository has two experiment protocols created at different times. Do not mix them:

| Research series | Development and validation | Locked period |
|---|---|---|
| Earlier raw-order-book capacity experiments | Train 2021–2023; validate 2024 | 2025 |
| Current AgentX and event-stream research | 2025 has been used for development and rolling validation | 2026 |

Each experiment's configuration and `ResearchProtocol` determine which dates it may access. Documentation about a locked test must name the applicable protocol.

## Current work order

1. Keep z-score labels and `all` supervision as the new baseline.
2. Audit daily-task gradient strength at the z-label best checkpoint and define candidate task weights and stopping thresholds.
3. Run seed-0 validation for daily-task weighting first. Add the adjacent fold and OOS only if the candidate passes.
4. If task weighting does not add value, design a small cost-aware ranking-objective experiment.
5. Add sector exposure attribution after dated sector-classification data is available.
6. Reconsider `probe150m` budget and its seed-0 gate only after after-cost active returns improve across windows.
7. Record new conclusions in the [experiment log](research/experiment-log.md) and update this status page.

## CPU validation path (2026-08-29)

To validate interfaces without an A100, the project added the small-sample `ticknet.research.cpu_validation` path:

```text
train rows -> NumPy linear predictions -> formal prediction -> alpha-research signal -> fixed-K portfolio
```

It validates data interfaces, date isolation, signal conversion, and backtest output. It does not replace formal PyTorch training. `compare_portfolio_evaluations()` and `compare_portfolio_digests()` produce stable summaries of returns, drawdown, costs, turnover, and daily detail. `digest_portfolio_backtester_result()` converts the `portfolio-backtester` five-tuple to the same summary format so the repositories can run differential comparisons without sharing a PyTorch environment.

A real raw-order file, `order_20260424.parquet`, was found locally. It has 313,199,561 rows and 299 row groups. A CPU check of its first 1,048,576 rows found one `TradingDay` value (`20260424`), 883 non-positive prices, and 206,009 rows with duplicate `OrderID`; timestamps were not out of order. The platform quality checker now maps `SecuCode` to `ticker` and `OrderTime` to `time_ms`, leaving `missing_columns` empty in the sample report. A full scan was deferred to avoid unnecessary I/O.

## L2 opening order-identity ledger audit

Historical-data admission is handled by `scripts/build_historical_data_manifest.py`. The 2026 period is locked. Eligible Shenzhen stock-days from 2021 through 2025 are primary data, while eligible Shanghai stock-days are limited to lag research. See [historical raw L2 eligibility](research/historical-data-eligibility-2026-08-27.md) for the admission rules and the distinction between general A-share properties and data-source-specific behavior.

Raw L2 coverage is audited with `scripts/audit_opening_coverage.py`. It scans daily `order_preopen` files and counts pre-open orders, orders, trades, and snapshots by trading day and stock, along with opening trade volume where `time_ms <= 0`. The first real-file smoke result and counting rules are in the [raw L2 opening coverage inventory](research/opening-coverage-inventory-2026-08-27.md). Full reports should be written outside the repository.

The new pre-open order-identity auditor is `ticknet.simulator.opening_ledger`. It processes pre-open orders and orders, trades, and cancellations in the opening settlement window. It subtracts fills and cancellations from remaining quantity by order ID, aggregates the top ten price levels, and compares them level by level with the first complete continuous-auction snapshot. Results distinguish exact matches, quantity or price differences, incomparable snapshots, missing pre-open files, stocks without pre-open records, unknown trade identities, unknown cancellation identities, and over-subtraction.

The first `time_ms=0` snapshot cannot be treated as a direct aggregation of remaining pre-open orders. Shenzhen samples require mapping the snapshot to the event-clock settlement window `0 + 140ms`. Shanghai has no fixed offset reusable across dates. Its default is `0ms`, and the command-line tool allows an explicit lag sweep.

The first cross-stock, cross-date audit used 6 TB of raw L2 data on 2026-08-27. It covered 13 stock-days in 2021 through 2025:

| Outcome | Count |
|---|---:|
| Exact match at all ten levels using the best lag | 9 |
| Comparable, with book-quantity or price differences | 2 |
| Incomplete input or no pre-open record for the stock | 2 |
| Exact matches among comparable samples | `9/11 = 81.8%` |

All six Shenzhen samples matched exactly at their best lag, which was `140ms` for each. Best lags for Shanghai stock `600000` were `0`, `70`, `90`, `120`, and `150ms`. On `2022-06-15`, the ninth bid level at price 777 was short by 3,900 shares at `0ms`. Order `255949` was a 3,900-share order with event time `150ms`. Including it through a `150ms` cutoff made all ten levels match. This supports an event-time boundary explanation and does not currently support an error in cancellation quantity or matching rules.

The pre-open files for Shanghai stock `600000` on `2021-01-04` and `2021-03-01` did not contain the stock, although opening trades were present, so these samples were marked incomparable. They cannot be used to evaluate the matcher. A `-140ms` to `210ms` sweep found no single reusable Shanghai offset. The 2025 sample improved at `+140ms`, while older samples remained inconsistent.

Other comparable Shanghai differences were 100 shares missing at best bid for `600000` on `2023-05-12`, and 2,100 extra shares at best bid plus 600 shares missing at the fifth bid level on `2024-12-13`. Neither case had unknown trade identities, unknown cancellation identities, or over-subtraction. Best lags were `70ms` and `120ms`, suggesting at least date- or file-batch-level timing differences in Shanghai.

Reproduction command:

```bash
python scripts/audit_opening_ledger.py \
  --raw-root $QUANT_DATA_ROOT/raw/cn_a_share_level2 \
  --sample 20210104:000001 \
  --sample 20210104:600000 \
  --sample 20220615:600000 \
  --sample 20250613:600000 \
  --output /tmp/opening-ledger-audit.json
```

The current conclusion is that the pre-open order-identity chain can reconstruct the opening ten-level book in the covered Shenzhen samples. It does not show that inputs are complete for every market and date. The opening ledger remains an audit tool and does not directly change event-stream packing or matching replay. Next, expand Shanghai samples where a pre-open stock record exists, summarize best lag by file batch, and investigate whether single-level quantity differences come from snapshot aggregation time, auction-trade boundaries, or missing source fields.

## Evidence index

- [AgentX research roadmap](research/topk-agentx-research-roadmap.md)
- [Historical experiment log](research/experiment-log.md)
- [Raw-order-book multi-horizon and capacity roadmap](nextday/multi-horizon-data-expansion-roadmap.md)
- [100M and raw-1000 controlled matrix](nextday/nextday-100m-raw1000-benchmark.md)
- [Event-stream rolling roadmap](nextday/h5-rolling-eventstream-roadmap.md)
- [External L2 project comparison](research/external-l2-research-comparison.md)
- [Formal M3 Top-K diagnostics](research/topk-agentx-m3-topk-diagnostics.md)
- [Event-stream signal-decay and trading-conversion diagnostics](research/eventstream-signal-trading-diagnostics.md)
- [Event-stream multi-task gradient audit](research/eventstream-gradient-audit.md)
- [Event-stream label-scale study](research/eventstream-label-scale.md)
