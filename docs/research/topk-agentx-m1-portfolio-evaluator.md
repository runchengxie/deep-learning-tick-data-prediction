# M1 Top-K Long-Only Portfolio Evaluator

## Summary

`ticknet.research.portfolio` is now a model-agnostic portfolio evaluator. HGB, LambdaMART, TCN, DeepLOB, and future AgentX executors can share fixed-K selection, rank buffers, rebalance thresholds, tradability constraints, costs, and stability metrics when they emit the same prediction contract.

The historical `portfolio_quantile` long-short backtest remains available, but its `mode` is explicitly marked `legacy_quantile_long_short_diagnostic`. It is not the formal Top-K long-only strategy for the new research series.

## Input contract

Prediction Parquet must contain:

| Field | Meaning |
|---|---|
| `symbol` | Stock code; unique within each `label_date` |
| `trading_date` | Signal date T |
| `label_date` | Rebalance date at the T+1 open |
| `score` | Cross-sectional score using only information available by T |
| `target_return` | Formal target: T+1 open to T+2 open |
| `in_universe` | Member of the dynamic candidate universe; required for formal runs |

Optional `can_buy` and `can_sell` fields must be provided together. Without them, smoke tests assume tradability. Formal runs require `--require-tradability --missing-holding-policy error`. Formal mode also requires `in_universe`: only `true` rows enter ranking, IC, and universe-baseline metrics. A `false` row tracks an existing holding and cannot become a new buy candidate.

The evaluator never filters candidates by future return availability. If a selected holding has a missing `target_return`, evaluation fails to avoid implicit look-ahead selection.

## Portfolio state machine

Each day, positions are determined in this order:

1. Force-retain existing holdings that cannot be sold.
2. Retain old holdings that meet `min_position_score` and remain within `top_k + exit_buffer`; without an absolute threshold, check rank only.
3. Compare remaining old holdings with buyable new candidates. Replace an old holding only when the new score exceeds it by at least `min_score_gap`.
4. With `allow_cash=true`, each selected stock still targets weight `1 / top_k`; slots below the absolute score threshold remain cash.
5. Restore target equal weights where possible. Buy/sell restrictions set weight bounds and cannot be bypassed through implicit reweighting.
6. At the end of the holding period, let weights drift with realized returns. Trades needed to restore target weights the next day count toward turnover and cost.

In smoke tests, an old holding absent from the dynamic universe may be explicitly liquidated with reason `universe_exit`. Formal evaluation uses missing-holding policy `error`: upstream data must provide an `in_universe=false` state row for a removed holding. If it is tradable, record `universe_exit`; otherwise force-retain until a later state permits exit. This distinguishes universe rotation from missing data.

## Costs and metrics

Trade details use changes in target weight:

```text
buy_cost  = buy_notional  * per_side_bps
sell_cost = sell_notional * (per_side_bps + sell_stamp_tax_bps)
net_return = gross_return - buy_cost - sell_cost
```

Initial entry has buy notional 1 and is explicitly charged. Daily output separately records buy, sell, and mean one-way turnover, so every buffer-driven change can be traced to stock-level trades.

Summary metrics include gross and net daily/annualized return, volatility, Sharpe, cumulative return, maximum drawdown; realized Top-K overlap, return versus the full-universe baseline, and Rank IC within selected holdings; monthly cumulative return, cost-adjusted return versus equal-weight universe, and positive-day share; absolute-return and absolute-excess contributions from the most extreme 1, 5, and 10 days; holding count, net/gross exposure, maximum weight, and HHI concentration; and cash weight and day-to-day changes.

Absolute score thresholds should use expected returns calibrated on validation. Raw score scales differ across models, so the same numeric threshold should not be reused without calibration. The [event-stream signal and trading diagnostics](eventstream-signal-trading-diagnostics.md) records calibration and cross-window checks.

## CLI and artifacts

Example:

```bash
python scripts/evaluate_cost_adjusted.py \
  --predictions results/predictions.parquet \
  --top-k 50 \
  --exit-buffer 20 \
  --min-score-gap 0.05 \
  --cost-bps 10 \
  --stamp-tax-bps 5 \
  --require-tradability \
  --missing-holding-policy error \
  --output-dir results/topk-k50-buffer20-cost10
```

The output directory contains `summary.json` (portfolio, cost, ranking, stability, and risk summaries), `daily.parquet` (daily turnover, return, costs, exposures, and Top-K metrics), `holdings.parquet` (daily stocks, ranks, scores, weights, return contribution, and retention reason), and `trades.parquet` (each weight change, side, reason, notional, and cost).

Without `--top-k`, the CLI retains compatibility with the historical quantile long-short mode. New experiments and AgentX must not use that legacy mode as a Top-K success gate.

## Engineering smoke test

The frozen 2025 HGB Top-100 prediction rows were evaluated with K=50, buffer=20, 10 bp one-way cost, and 5 bp sell stamp duty. The 125-day run produced 125 daily rows, 6,250 holding rows, and 7,958 trade rows, with about 28.6% mean daily one-way turnover.

That file used the historical next-open-to-same-close target and has no tradability fields. This run verifies implementation and artifacts only; it is not a return result under M0's new open-to-open trading contract.
