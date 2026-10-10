# Event-Stream Signal Half-Life and Trading Diagnostics

Reading update, 2026-10-10. This record preserves the plans and evidence available at its research stage. References to upcoming seed-0 training, M3 materialization, gradient checks, or label-scale work are historical plans, not the current task list. Later experiments and formal diagnostics are recorded in [Project status](../project-status.md). Follow that entry point for current decisions and locked-period rules.

## Conclusion

`EVT-HALFLIFE-001`, `TRD-STAGGERED-H5-001`, `TRD-RANK-EMA-001`, and `RISK-ATTR-001` are complete. Final decision: `HOLD`.

Rank EMA and the expected-return rebalancing threshold materially reduced daily turnover. Five staggered H5 cohorts held one fifth of capital each and fixed daily one-way turnover near 20%. These changes improved cost-adjusted active return in December 2025, but the improvement did not repeat in November; the two consecutive OOS windows had opposite directions. An absolute entry threshold came close to holding all cash in the next window, so there is not yet a portable threshold.

Evidence does not support adding adjacent-fold seeds 1 and 2 or starting a 150M model. Next, run single-variable checks of task gradients, label scale, and supervision position.

## Inputs and selection boundary

The adjacent fold uses the direct H5 daily output of `capacity100m`, seed 0: training July–September 2025, validation October, and OOS November. Stock-level predictions contain 4,683 validation rows and 5,807 OOS rows. Checkpoint SHA-256: `d753041016d71e668a46624585f7bc7fb68f67200fee3c5b463c28b26f1c11cd`.

The recent fold reuses existing predictions from joint-model seed 0: 6,963 November validation rows and 8,125 December OOS rows. The joint model uses an H5-pretrained event-stream backbone and minute features; its classification supervision comes from H1 returns on formal minute samples. These folds used their respective candidate models at the time. Cross-window results test whether trading rules transfer; they are not a pure rolling reproduction of one fixed model architecture.

Multi-horizon returns share label-sidecar fingerprint `d33fdce896062ab438a4c95b9a594b653ff366cca65d0a73e01bdccd680aa6d3`. The contract uses a close signal on T, enters at the T+1 open, exits at the T+H close, and subtracts the concurrent CSI All Share return. Locked 2026 was not used for prediction export or diagnostics.

Rules were selected only on October validation from the earlier fold. November and December OOS were used only to check acceptance criteria. Daily percentile ranks were first mapped to expected H5 excess returns by isotonic regression on validation, then evaluated with absolute entry and expected-return-difference rebalancing thresholds. Each fold fits its own validation mapping; raw scores from different models are not directly compared.

## H1–H10 signal behavior

Top-group return is the mean active return of Top-100 over the full holding period. Extreme-group spread compares the top and bottom deciles. Values are in basis points (bp), except Rank IC.

| OOS window | Horizon | Rank IC | Extreme-group spread | Top-100 active return |
|---|---:|---:|---:|---:|
| Nov 2025 | H1 | -0.00131 | -40.19 | -7.63 |
| Nov 2025 | H3 | 0.03236 | -31.80 | -17.84 |
| Nov 2025 | H5 | 0.03301 | -55.65 | -45.34 |
| Nov 2025 | H7 | 0.04298 | -28.03 | -47.11 |
| Nov 2025 | H10 | 0.06571 | 23.08 | -39.16 |
| Dec 2025 | H1 | 0.02043 | -53.34 | 16.09 |
| Dec 2025 | H3 | 0.07588 | 33.61 | 60.75 |
| Dec 2025 | H5 | 0.09956 | 74.34 | 111.05 |
| Dec 2025 | H7 | 0.09729 | 100.42 | 164.01 |
| Dec 2025 | H10 | 0.10191 | 60.39 | 214.23 |

Neither window shows clear IC decay within H10. November full-universe IC rises with horizon while Top-100 active return remains negative. December develops positive extreme-group and Top-100 returns from H3 onward. H5 does not appear too long, but top-group performance is not stable across windows.

Full outputs also retain daily IC, extreme-group spread, `NDCG@100`, `Precision@100`, and non-overlapping cumulative returns grouped by entry date.

## Rank EMA, rebalancing threshold, and cash

The 27-rule matrix compares raw ranks, EMA levels `0.5` and `0.2`, three expected-return-difference thresholds, and entry thresholds of none, 0, and 5 bp. Daily portfolios use H1 returns, 10 bp fixed one-way cost, and an extra 5 bp stamp duty on sells.

October validation selected the simpler rule: rank EMA `0.5`, a 5 bp expected-return rebalancing threshold, and no absolute entry threshold. Ties favor the rule with fewer parameters.

| OOS window | Rule | Mean daily one-way turnover | Mean daily cost-adjusted active return | Dynamic-cost active return |
|---|---|---:|---:|---:|
| Nov 2025 | Raw rank | 36.79% | -12.80 bp | -14.80 bp |
| Nov 2025 | EMA + rebalancing gate | 13.90% | -13.02 bp | -14.68 bp |
| Dec 2025 | Raw rank | 43.45% | -23.77 bp | -25.41 bp |
| Dec 2025 | EMA + rebalancing gate | 8.90% | -10.99 bp | -12.22 bp |

The new rule reduced turnover to 37.78% and 20.48% of raw-rank turnover. November cost-adjusted return fell by 0.21 bp; December improved by 12.78 bp. Both windows remained negative.

Entry thresholds did not yield a stable rule. On recent-fold validation, the calibrated H5 expected-return ceiling for raw ranks was still below zero. A zero-return entry gate therefore held all cash in December. The absolute threshold depends heavily on the calibration window and cannot yet be frozen.

## Five staggered H5 cohorts

Each day opens one cohort equal to 20% of total capital. A cohort buys at the T+1 open and exits at the T+5 close. With five cohorts in rotation, one expires each day. Daily active return is the contribution of the cohort expiring that day, scaled by its 20% capital allocation.

| OOS window | Rule | Mean daily one-way turnover | Fixed-cost active return/day | Fixed-cost cumulative active return | Dynamic-cost active return/day |
|---|---|---:|---:|---:|---:|
| Nov 2025 | Raw rank | 20.00% | -13.90 bp | -2.07% | -14.05 bp |
| Nov 2025 | EMA + rebalancing gate | 20.00% | -15.48 bp | -2.30% | -15.61 bp |
| Dec 2025 | Raw rank | 20.00% | 14.34 bp | 3.05% | 14.28 bp |
| Dec 2025 | EMA + rebalancing gate | 20.00% | 17.12 bp | 3.65% | 17.06 bp |

Staggering addresses full-portfolio daily turnover and preserves the H5 signal in December. November remained negative, so the cross-window directional gate failed. Each H5 cohort exits at the fifth-day close; the next cohort starts at the following open. Since existing labels do not cover the overnight return in this interval, the daily H1 portfolio counts return-difference turnover, while H5 cohorts are charged for each full buy/sell.

## Dynamic costs and risk exposures

Dynamic cost is a capacity scenario for a 100 million RMB portfolio. It adds square-root market impact based on participation in traded value to fixed cost, with a 10 bp coefficient and 50 bp one-way impact cap. Twenty-day average traded value is `close × volume × 100`, because local volume is recorded in lots. Missing liquidity receives the 50 bp cap. Stock-level coverage ranges from 96.72% to 99.90%.

Dynamic costs did not change the sign of either window's results. Top-100 stocks were liquid enough that incremental impact was small for staggered H5 portfolios.

Average exposures under EMA plus rebalancing gate:

| OOS window | Size z-score | Liquidity z-score | Volatility z-score |
|---|---:|---:|---:|
| Nov 2025 | 0.475 | 0.374 | 0.011 |
| Dec 2025 | 0.567 | 0.282 | -0.495 |

The portfolio is mildly tilted toward larger, more liquid stocks, and toward lower volatility in December. All exposures are within one standard deviation. Local data has no dated industry-classification file, so industry attribution is marked `unavailable`; ticker or exchange boards are not used as substitutes.

The top five dates contributed 63.56% and 51.04% of absolute active-return variation in November and December, respectively, exceeding the 50% limit. Return-date concentration also failed.

## Decision and next work

The run passed turnover reduction and available style-exposure gates. It failed these gates:

- Cost-adjusted active return was not positive in both consecutive OOS windows.
- Improvement over raw ranks was not at least 1 bp in both windows.
- The top five dates did not contribute less than 50% of absolute active-return variation.
- Industry exposure could not be checked.

Retain `capacity100m` and joint-model artifacts, but pause adjacent-fold seeds 1 and 2 and `probe150m`. Next experiments:

1. `EVT-GRAD-AUDIT-001`: compare shared-backbone gradient norms and angles by task.
2. `EVT-LABEL-SCALE-001`: compare raw returns with daily cross-sectional winsorized z labels.
3. `EVT-SUPERVISION-POSITION-001`: compare all-position, final-position, and tail-weighted supervision.
4. Add seeds 1 and 2 only if a single seed improves both validation and adjacent-fold OOS.

## Reproduction and artifacts

Use `ticknet-eventstream-signal-diagnostics`. It validates stock identity, prediction shards, checkpoints, label sidecars, and source fingerprints, then writes half-life curves, the 27-rule policy matrix, H5 cohorts, dynamic costs, exposures, and gate decisions.

Final local artifacts are under `artifacts/eventstream-portfolio-diagnostics/signal-trading-v4`:

| File | SHA-256 |
|---|---|
| `summary.json` | `21cdac3a32ba92dcc756103888883c713fefa9c3cfaaf766d8a19a72afb10bcb` |
| `half-life.json` | `06ae26d050b178adee64c844208b2e79c2bc7c9a991bd7d299fb682684ae7048` |
| `policy-matrix.json` | `872909a3735d3991096a7aea40fb2f4fe0bfbb42f6a07a920245a8a2dce53b50` |
| `staggered-h5.json` | `487984807c12214c69500f53fb5cd5a9cee47cc10efcfe5fa32710b54c5a73ab` |
| `policy-matrix.parquet` | `0d4c396db02ac5b7e86d7656d2b52d9578e47b1334c4fb32ea6e2de0b5f35677` |
| `half-life.svg` | `fbb3db63bc7b23cf60bf5c58746a0ed741df293d51dd831506c666e3accdc705` |

`signal-trading-v1` through `signal-trading-v3` predate unit, tie-selection, artifact-path, and SVG-determinism fixes. Do not cite them as formal results.
