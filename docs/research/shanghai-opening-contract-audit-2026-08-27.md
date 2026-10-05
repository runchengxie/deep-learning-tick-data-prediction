# Shanghai Opening Data-Contract Audit

## Audit objective

Using the 6 TB raw L2 dataset, this audit checks whether `order_preopen` covers the target stock, scans candidate offsets between snapshot time and order/trade event time, and rebuilds the first complete continuous-auction snapshot from the order-identity ledger.

The audit is diagnostic only. It does not modify the matching engine or write Shanghai's best observed lag into production configuration.

## Method

Each stock-day's order, trade, and snapshot Parquet files are read once. Candidate event lags range from -200 ms to +200 ms in 10 ms steps. For each candidate, orders, trades, and cancels at or before `snapshot_time_ms + lag` are included in the order-level ledger. The top ten price levels and quantities are aggregated and compared with the snapshot.

Candidate lags are ranked by exact-match status, number of missing identities, quantity differences across the ten levels, and absolute lag. This ordering supports diagnosis; it does not prove the actual exchange protocol.

Run:

```bash
python scripts/audit_shanghai_contract.py \
  --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
  --sample 20210104:000001 \
  --sample 20210104:600000 \
  --sample 20210301:000001 \
  --sample 20210301:600000 \
  --sample 20220615:000001 \
  --sample 20220615:600000 \
  --sample 20230512:000001 \
  --sample 20230512:600000 \
  --sample 20241213:000001 \
  --sample 20241213:600000 \
  --sample 20250303:600000 \
  --sample 20250613:000001 \
  --sample 20250613:600000 \
  --trace 20220615:600000:bid:777 \
  --json-output /tmp/shanghai-contract-audit.json \
  --csv-output /tmp/shanghai-contract-audit.csv
```

## Results

| Result | Count |
|---|---:|
| Total samples | 13 |
| Exact matches at best lag | 9 |
| Comparable samples with differences | 2 |
| Incomplete inputs | 2 |
| Exact-match rate among comparable samples | `9/11 = 81.8%` |

All six Shenzhen samples had a best lag of `140ms` and matched all ten levels. Best lags for Shanghai stock `600000` were:

| Trading date | Best lag | Result |
|---|---:|---|
| 2022-06-15 | `150ms` | Exact match |
| 2023-05-12 | `70ms` | Bid level 1 short by 100 shares |
| 2024-12-13 | `120ms` | Bid level 1 over by 2,100 shares; level 5 short by 600 shares |
| 2025-03-03 | `0ms` | Exact match |
| 2025-06-13 | `90ms` | Exact match |

For `600000` on 2021-01-04 and 2021-03-01, the pre-open files existed but had no record for the stock. Its trades already appeared during the opening auction, so these samples are marked `opening_trade_gap` and excluded from the match rate.

## The 3,900-share case

For `600000` on 2022-06-15, with the default `0ms` cutoff, bid price 777 at level 9 had 76,700 shares in the reconstructed book versus 80,600 in the snapshot, a difference of exactly 3,900 shares.

Order-level tracing found order `255949`: price 777, buy side, original quantity 3,900 shares, event time `150ms`. It was neither traded nor canceled. Extending the cutoff from `0ms` to `150ms` included the order, brought level 9 to 80,600 shares, and matched all ten levels.

The strongest current explanation is a boundary difference between snapshot and order-event timestamps. This case does not support an incorrect cancel-quantity interpretation or matching-engine price rule.

## Conclusions and limitations

The Shenzhen `+140ms` result was consistent in this sample and can remain the default for Shenzhen audits. Shanghai's best lag varies across dates and cannot currently be generalized to a market-wide constant. Its default `0ms` is a conservative diagnostic setting; reports must retain the candidate-lag distribution.

This was an explicit sample audit, not a full-market, all-date census. Next, sample Shanghai stock-days with pre-open records, group best lags by source-file batch, and separately inspect the timestamp provenance of snapshot, order, and trade files.
