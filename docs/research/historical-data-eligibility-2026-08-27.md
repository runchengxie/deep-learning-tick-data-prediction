# Historical Raw L2 Data Eligibility

## Decision

Exclude 2026 data from historical event-stream reconstruction and model datasets for now. Build a stock-day eligibility manifest for 2021–2025. Eligible Shenzhen stock-days form the primary dataset; eligible Shanghai stock-days are research data.

Both primary and research samples require `order_preopen`, `order`, `trades`, and `snapshot` files, the target stock to appear in all three linked event files, and a pre-open order. Shenzhen primary data uses the verified +140 ms audit configuration. Shanghai is retained only when coverage criteria pass; do not assign it a fixed lag. It still requires stock-day time audits.

## Evidence

For 2026, 70 `order_preopen` daily files were found. Fifty-five dates lacked corresponding order and trade files but had snapshots. This is a data-lake partition gap, not evidence of a matching-rule problem. The remaining 15 dates had all three file paths.

A real smoke scan of 2021-01-04 contained 2,335 Shenzhen stock-days. Of these, 2,283 had an opening trade and passed current primary-data eligibility; 52 were excluded because linked stock records were missing.

Shanghai's best lag is not a market constant. Observations across dates and stocks include 0, 20, 90, 150, 240, 280, 370, and 380 ms. For stock `600000`, all ten levels matched on 2023-05-12 at 240 ms and on 2024-12-13 at 280 ms.

## Market-wide issue or dataset-specific issue?

A-share Level-2 research generally needs a time and semantic contract across call auctions, continuous-auction snapshots, and tick events. Public Shenzhen materials describe Level-2 as including ten-level snapshots and tick data, while trading rules distinguish opening call auction, continuous trading, and closing call auction. The Shanghai market-data gateway STEP interface also distinguishes snapshots and tick data and defines generation times for order or trade messages.

Auditing opening data is therefore a general market-data engineering requirement. That does not mean every A-share tick dataset is equally poor.

This dataset has several specific gaps:

- Entire periods of 2026 order and trade files are missing.
- Early 2021 Shanghai pre-open order coverage is incomplete.
- Shenzhen has a stable +140 ms offset across sampled records, indicating a consistent clock transformation somewhere in the data supply chain.
- Shanghai lag varies by stock-day and sometimes stock, so the Shenzhen configuration cannot be reused directly.
- For some snapshots, `Volume` and `DealNum` are not fully explained by the same event boundary that best matches the book.

The checked samples do not indicate widespread random corruption of 2021–2025 order or trade data. The more likely issue is that file coverage, timestamp labels, and aggregation windows lack a unified contract.

## Event ordering boundary

When raw order Parquet retains exchange sequence fields such as `ChannelNo`, `ApplSeqNum`, or `BizIndex`, the simulator preserves them in `SimulatorEvent`. Events with the same `time_ms` are reordered by sequence only when all have sequence values and belong to a single channel. A snapshot remains after same-millisecond orders and cancels.

When multiple channels occur in the same millisecond, the simulator preserves file source order and records `cross_channel_total_order=false` in `SimulatorPack.ordering_provenance`. If the source has no sequence fields, it records `timestamp_fallback`. Timestamps remain the cross-channel time coordinate; sequence values strengthen local ordering only where evidence supports it. Vendor files must not be assumed to provide a globally ordered exchange tape.

## Usage boundaries

Eligible Shenzhen stock-days from 2021–2025 may be used for primary event-stream research with the eligibility manifest and verified time configuration. Eligible Shanghai stock-days from 2022–2025 may be used to study lag and data contracts, but must not share the Shenzhen matching configuration. Exclude 2026 until order and trade files are complete and a new manifest is built.

## Command

```bash
PYTHONPATH=src .venv/bin/python scripts/build_historical_data_manifest.py \
  --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
  --json-output /tmp/historical-manifest.json \
  --csv-output /tmp/historical-manifest.csv
```

Eligibility rules are implemented in `src/ticknet/simulator/eligibility.py`; coverage scanning is in `src/ticknet/simulator/coverage.py`.

## External rules references

- [Shenzhen Stock Exchange investor service: trading hours and call auction](https://www.szse.cn/www/investor/knowledge/stock/deal/t20190626_568129.html)
- [Shenzhen Stock Exchange next-generation trading system FAQ: Level-2 snapshots and tick data](https://www.szse.cn/www/marketServices/technicalservice/introduce/P020180328467244590967.pdf)
- [Shanghai Stock Exchange IS120 market-data gateway STEP interface specification](https://www.sse.com.cn/services/tradingtech/development/c/10816478/files/51a3e4c6b92345689c682448582c019d.pdf)
