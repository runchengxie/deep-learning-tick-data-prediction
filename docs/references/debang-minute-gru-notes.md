# Minute-Data GRU Stock-Selection Strategy: Reading Notes

> Source information
> - Original title: <!-- preserved-source:start -->德邦证券：基于分钟数据的 GRU 模型在选股策略中的应用初探<!-- preserved-source:end -->
> - Series: Debang Securities quantitative research, Machine Learning Series No. 6, in-depth report
> - Date: 2024-07-01
> - Analyst: <!-- preserved-source:start -->肖承志<!-- preserved-source:end --> (S0120521080003)
> - Source PDF: cited but not included in this public repository because redistribution rights have not been established
> - Training and backtest data: minute data from 2018-01-01 through 2024-06-21; factor and portfolio statistics from 2019-01-01 through 2024-06-21

These notes are based on the full text extracted from the source PDF. They summarize the main findings, follow the report's structure to describe its methods, results, and limitations, and conclude with implications for this project's minute-data research.

## Main findings

- The method is simple: normalize one day's 240 minute bars, each with seven price and volume features, and feed them to a GRU with 32 hidden units to predict next-day open-to-open return. The report uses no factor engineering or style neutralization.
- Mean daily factor Rank IC is 7.5% (about 8.0% before 2022 and 6.9% afterward). Cumulative IC shows no clear decay.
- A 500-stock equal-weight long portfolio, benchmarked against CSI 1000 and charged 3 per mille for each side, reports 22.45% annualized excess return using daily open execution. Results are highly sensitive to rebalance frequency and slippage. The annualized impact of slippage is about 4.87%–9.54%; rebalance-frequency differences are about 6%–10%. Daily VWAP execution reduces annualized excess return to 12.91%.
- Weekly VWAP index enhancement, fully neutralized by sector and style, reports excess returns of +7.26% for CSI 300 (IR 1.93), +7.58% for CSI 500 (IR 1.75), and +8.86% for CSI 1000 (IR 1.83), with tracking error below 5%.
- The factor favors low liquidity, low volatility, low valuation, and high profitability. Its size correlation is only -0.03. The report recognizes that these style exposures contribute to both excess returns and transaction-cost sensitivity.

## 1. Model and method

GRU is a simplified LSTM that retains the update and reset gates. The report describes it as more efficient to train, with comparable or better performance. It makes no complex A-share-specific changes and uses basic market data to test GRU's ability to extract information:

- **Input:** one day of 240 minute bars with seven features: open, high, low, close, volume, turnover value, and trade count. Features are normalized over the 240 intraday steps.
- **Architecture:** `Input_size=7`, `Hidden_size=32`, `Bias=False`. The 32 hidden states are averaged to produce one scalar.
- **Target:** next-trading-day open-to-open return, standardized cross-sectionally each day.
- **Training:** Adam (`lr=0.001`), MSE, batch size 4096. A rolling ten-month window is split 4:1 into eight months of training and two months of validation. The model is retrained monthly and used for one month.
- **Universe:** all stocks, excluding ST and *ST stocks, stocks listed for less than one year, and suspended or untradeable stocks.

Features are normalized only through intraday time, and the target only cross-sectionally. There is no sector or size neutralization. This minimally engineered baseline therefore exposes its style tilts directly.

## 2. GRU factor results

- **IC:** mean daily Rank IC 7.5%, about 8.0% before 2022 and 6.9% afterward. Cumulative IC is stable without a pronounced drawdown.
- **Deciles:** decile returns are monotonically ordered. The bottom decile averages -0.34% per day; the top decile averages +0.24% per day. Long and short contributions are balanced, with the short side weaker.
- **Style:** size correlation is only -0.03. The factor has clear negative exposure to liquidity and volatility and positive exposure to valuation and profitability. It favors illiquid, low-volatility, low-valuation, and high-profitability stocks.
- **Sector:** exposures are broadly balanced. Scores are slightly higher for steel and banks and lower for communications and non-bank financials. Since 2023, dispersion has widened: media, consumer services, computers, and communications fell, while steel and banks rose.

## 3. Long-only portfolios and index enhancement

The long-only backtest covers 2019-01-01 through 2024-06-21. It excludes ST stocks, recent listings, and suspensions, uses CSI 1000 as the benchmark, holds 500 stocks equally, and charges 3 per mille on both sides. Turnover limits are 5% one-way daily, 25% weekly, and unlimited monthly. Execution uses the open or all-day VWAP.

| Rebalance | Open excess return, annualized | VWAP excess return, annualized | IR (open / VWAP) | Calmar (open / VWAP) | Monthly win rate (open / VWAP) |
|---|---:|---:|---:|---:|---:|
| Daily | 22.45% | 12.91% | 3.17 / 1.91 | 2.23 / 1.19 | 86% / 79% |
| Weekly | 16.95% | 9.81% | 2.40 / 1.44 | 1.53 / 0.81 | 82% / 76% |
| Monthly | 11.79% | 6.92% | 1.76 / 1.08 | 1.14 / 0.65 | 77% / 73% |

- **Open versus VWAP:** open execution outperforms over the full period, with an annualized difference of 4.87%–9.54%, which indicates the reported slippage range.
- **Rebalance frequency:** before 2020, more frequent rebalancing performed clearly better. Since 2023, its contribution weakened and sometimes turned negative.
- **2024:** equal weighting with small-cap exposure had a large drawdown during the February risk event, and returns weakened during the current year.

The index-enhancement portfolio limits absolute sector deviation to below 1%, style deviation to below 0.01 standard deviations, requires 80% constituent coverage, rebalances weekly with 15% two-way turnover, and executes at VWAP. Reported excess returns are +7.26% for CSI 300 (IR 1.93, Calmar 1.68, max drawdown 4.33%), +7.58% for CSI 500 (IR 1.81), and +8.86% for CSI 1000 (IR 1.83, Calmar 1.35). Excess return rises as benchmark constituent size decreases, but sensitivity to small-cap risk events also rises.

## 4. Limitations and independent assessment

### Limitations acknowledged in the report

- It uses only one day's minute bars, which contain limited information, and barely transforms the target. The report says long-short capability could improve.
- The low-volatility and low-liquidity preference has a material trading impact. Reducing these style tilts is proposed as a research direction.
- Suggested improvements include richer inputs, target processing to change style exposure, and style constraints in the loss function.
- Risks include historical patterns ceasing to hold, overfitting, imperfect repeatability from random seeds, and idealized execution at the open or VWAP.

### Independent assessment from this project's perspective

- Excess returns on the order of 22% combine open-price execution, high turnover, and equal-weight small-cap exposure. They should not be treated as tradable net returns. VWAP execution with neutralized index enhancement, around 7%–9%, is a more realistic comparison.
- The 7.5% IC uses raw open-to-open returns without residualization or neutralization. It is not directly comparable with this project's minute-line residual daily IC of 0.02–0.035.
- Factor returns have meaningful low-volatility and low-liquidity style beta. The small-cap risk event in February 2024 and negative contribution from frequent rebalancing since 2023 are direct evidence that trading and style exposures erode the signal. This aligns directionally with this project's cost-sensitive minute signal and breakeven cost of about 5–6 bp.

## 5. Implications and actionable work

1. **Add a GRU sequence baseline.** The project's minute path currently pairs HGB over aggregates with TCN over sequences. The report's setup can be implemented as a GRU baseline using seven minute features, intraday sequence normalization, next-day open-to-open returns, and rolling ten-month training with monthly retraining. Compare HGB, TCN, and GRU under the same walk-forward and cost framework.
2. **Align definitions before comparing results.** For reproduction, use this project's target definition, such as Barra residual returns or a neutralized target, and run net-return evaluation through `research/portfolio.py`. Quoting the report's 22% excess return or 7.5% IC directly would overstate comparability.
3. **Reuse the cost and rebalance sensitivity framework.** The report's open-versus-VWAP, daily/weekly/monthly, and annualized-slippage comparisons complement this project's turnover and cost diagnostics.
4. **Use neutralized index enhancement as the more honest reading.** The report's CSI 300/500/1000 enhancements (IR around 1.7–1.9) remove style exposures and provide a better cross-project comparison anchor than the long-only portfolio.
5. **Consider the proposed improvements.** Style constraints in the loss function align with this project's target governance and labels. They could be tested if the GRU line proceeds.

## 6. Reproduction proposal

- The project has L2 and minute data from 2021 onward. It could rerun normalized minute bars through a sequence model to next-day open-to-open prediction, but should use residualized labels, train on 2021–2023, validate on 2024, lock 2025, and calculate costs using the project's Top-K protocol.
- For a first version, reuse the `minute_tcn` data pipeline and change only the model to GRU. This minimizes confounding and avoids repeating the TCN pattern of strong validation that failed to generalize to test.
