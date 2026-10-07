# Lagged Age-Layer Chip Summaries and HGB

## Status and question

The first CPU experiment path compares lagged ordinary price features, age-layer
chip summaries, and their combination on identical formal candidates and date
splits. It can also compare existing materialized minute features with and without
chip summaries. Engineering checks use synthetic inputs. No real training result
or independent reproduction of the broker's reported performance is claimed.

The question is whether estimated cross-day holding costs improve Top-K returns
after costs, not only whole-universe Rank IC. Run this comparison before building
a distribution CNN-GRU or retraining the event-stream trunk.

## Source and deliberate differences

The source is Huatai Securities, 2026-06-02, AI series 105, bibliographic title
`基于筹码分层结构的端到端AI因子`, especially pages 5-7 (construction),
9-11 (summaries and ablations), and 13-14 (combination and portfolio evaluation).
The user supplied a local scan; the report is not redistributed here.

The report estimates survival with turnover and models a daily triangular
acquisition-price distribution around VWAP. It processes 30-day, four-channel,
32-price-bin distributions with CNN-GRU and an IC objective. This first experiment
instead trains HGB on summaries, uses the repository's three-class formal targets,
and keeps its open-to-following-open execution and 10 bp one-way cost plus 5 bp
sell stamp duty. It is an inspired experiment, not an exact reproduction.

## Data contract

Consume a published long-table Parquet asset from the market-data platform.
Provider downloads, reusable normalization, and turnover-unit conversion remain
upstream. The model package does not import provider SDKs or platform modules.

| Column | Required meaning |
|---|---|
| `symbol` | Same stable security identifier as the formal target data |
| `trading_date` | Arrow `date32`; one unique row per security and market session |
| `open`, `high`, `low`, `close`, `vwap` | Positive raw prices in one currency and unit; VWAP within low/high |
| `volume` | Nonnegative shares, not lots |
| `turnover` | Fraction of the specified circulating-share base, within [0, 1] |
| `adj_factor` | Positive factor whose consecutive ratios describe corporate actions available as of each session |

Required Parquet schema metadata:

```text
ticknet.chip_daily_contract = raw_ohlcv_shares_fractional_turnover_v1
ticknet.adjustment_contract = point_in_time_ratio
```

Declare metadata only after verifying the units and point-in-time factor semantics.
The producer should document the circulating-share denominator and asset version.
Do not silently substitute close for VWAP or clip percentage-point turnover into
the accepted range. Turnover above 100% needs a separately specified model of
survival; this first contract rejects it.

Include explicit carried-price, zero-volume, zero-turnover rows for suspended
sessions. A missing stock session resets its state and warmup. The union of daily
asset dates and target signal dates supplies the session calendar; it must cover
all intervening exchange sessions. A missing session for the entire asset cannot
be detected without an independent exchange calendar and remains an input limit.

The reader checks date min/max statistics before loading any price columns.
It skips wholly locked row groups and rejects groups spanning the 2026 boundary,
or groups without usable date statistics. Publish date-separated row groups;
an Arrow output predicate alone does not guarantee physical locked-data isolation.
The CLI fingerprints only included research bars rather than hashing locked price
payloads. The target configuration must also end before 2026, including its return
end dates. The original 2026 locked period remains sealed.

## State and features

New mass replaces the fraction specified by turnover. Survivors age by one
trading session. The four non-overlapping groups are ages 1-2, 3-10, 11-100, and
101 onward; this explicitly resolves the report's overlapping day-100 notation.
All groups share one normalization, retaining their relative mass.

The implementation uses an internal log-price grid with 1% log spacing anchored
to the first observed adjusted VWAP. It integrates triangular mass over bin edges,
retains 100 recent cohorts, and accumulates older mass and its first age moment.
It rejects prices outside the wide internal grid rather than clipping their mass.
It does not implement the report's 32-bin relative-price tensor or square-root
preprocessing. Those belong to the later CNN-GRU study.

The 18 chip features are profit ratio, mean and peak relative cost, concentration,
entropy, mean age, and four groups' mass, profitable mass, and cost-weighted mass.
Layer profit and cost quantities are unconditional contributions; their sums
recover the corresponding total. The cost bin containing the close is treated as
neutral when counting profits, avoiding artificial profitability from rounding.
Concentration and entropy depend on this fixed grid and should not be compared
numerically with different histogram conventions.

Five ordinary controls are daily open-to-close return, range divided by close,
VWAP divided by close, turnover, and log volume. This is a small price control,
not a claim to outperform Alpha158 or the established minute baseline. Use
`--minute-features` for the direct incremental comparison with that baseline.

Initial mass is placed in the first day's observed distribution at age 1.
Features require 300 consecutive sessions by default. Each split reports the
maximum surviving initial mass so that low-turnover initialization bias stays
visible. Repeat with longer warmup before drawing numerical conclusions.

For a signal at 14:55 on T, every chip and ordinary feature uses the state at the
immediately preceding market session, T-1. Missing or insufficient-history rows
are retained as NaNs, keeping candidate pools paired. Missing rows are not evidence
that the feature is neutral; review coverage before accepting a comparison.

## Run

Use a machine-local copy of an existing formal minute YAML with resolved asset
paths. The sample configuration in `configs/nextday-minute-formal-2025-v2.yaml`
describes the formal historical split but contains the original Linux paths.
Keep machine-local settings outside Git. Store outputs in the owning project's
configured data directory, including when running from a task worktree.

```powershell
uv run python scripts/run_chip_baseline.py `
  --config "$env:CONFIG_ROOT/quant-deep-learning/chip-formal.yaml" `
  --daily-bars "$env:DATA_ROOT/quant-market-data-platform/assets/chip-daily.parquet" `
  --output-dir "$env:DATA_ROOT/quant-deep-learning/experiments/chip-age-hgb/seed-0"
```

The roots above must be resolved from the workspace settings or supplied by the
launcher. No provider asset exists merely because this example names a path.
For the minute comparison, add `--minute-features` with a complete existing minute
feature manifest. Its loader checks the original source identity and target keys.
Add `--evaluate-test` only after freezing settings on validation; it evaluates
the configured pre-2026 development OOS split, not an unseen locked test.

Use a new, empty output directory for each experiment; existing artifacts are
never overwritten. HGB uses identical fixed iteration/leaf settings across feature sets and disables
random internal early stopping. Tune settings on date-held-out validation only.
Keep all candidates and off-universe holding-status rows needed by the formal
portfolio evaluator. Each evaluation exports a validated prediction Parquet,
portfolio daily/holdings/trades artifacts, and a summary. Model checkpoints and
input/target fingerprints and actual training-source digests are retained outside Git.
Prediction artifacts distinguish model feature availability from chip availability.
Saved pickle files are local
training artifacts; load only ones produced by a trusted run.

## Decision criteria and next stages

Before promoting a feature, compare paired daily Rank IC, Top-K cost-adjusted
active returns, turnover, monthly direction, feature coverage, and return-date
concentration. Use additional rolling windows and fixed independent seeds;
bootstrap by date rather than individual stocks. Examine industry, size,
liquidity, and volatility exposures where dated data exists. A single positive
month or better Rank IC is insufficient.

If summaries add stable information, test an unsplit distribution, the four-age
distribution CNN-GRU, and removal of price-position and temporal information.
Only then consider fusion with frozen event representations or order-size layers.
Large orders are a transaction-size proxy, not verified investor identities.

Code: [chip features](../../src/ticknet/nextday/chip_features.py),
[HGB comparison](../../src/ticknet/nextday/chip_baseline.py), and
[run entry point](../../scripts/run_chip_baseline.py).
