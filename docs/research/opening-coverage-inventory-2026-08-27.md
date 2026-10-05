# Raw L2 Pre-Open Coverage Inventory

## Audit objective

This work adds a raw L2 coverage scanner at trading-day and stock level. It distinguishes pre-open file presence, whether a stock has pre-open orders, whether linked files exist, whether the stock appears in each linked file, and whether an opening trade exists. The scanner audits data only; it does not change the matching engine or Shanghai's default lag.

## Counting rules

Pre-open order counts and volume sum non-cancel rows in `order_preopen`. An opening trade is a `trades` row with `time_ms <= 0`; opening volume sums its `Volume`. Snapshot discovery supports daily files, monthly files, and daily filenames containing hyphens.

The batch field uses the month directory containing a pre-open file as a reproducible proxy for the raw-file batch. The data lake has no separate batch metadata, so this field does not claim to represent a vendor batch.

## Run the audit

Use the project environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_opening_coverage.py \
  --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
  --json-output /tmp/opening-coverage.json \
  --csv-output /tmp/opening-coverage.csv
```

Use `--limit-days` to scan in chunks. Write outputs under `/tmp` or to a non-repository location on the external disk, and do not commit them to Git.

For a full audit, set `--index-path` to cache coverage results. The first scan still reads the linked raw files; later runs reuse trading days whose file sizes and modification times have not changed. Use `--refresh-index` to force a rescan.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_opening_coverage.py \
  --raw-root /mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2 \
  --index-path /mnt/data/hdd6t/quant-data-lake/projects/level2-coverage-index.json \
  --json-output /tmp/opening-coverage.json \
  --csv-output /tmp/opening-coverage.csv
```

## First real-data smoke result

The first run scanned the 2021-01-04 pre-open file from the 6 TB raw L2 dataset with `--limit-days 1`:

| Metric | Count |
|---|---:|
| Pre-open stock-days | 2,335 |
| Stock-days with an opening trade | 2,283 |
| Stock-days with all three linked files | 2,335 |
| Stocks present in all three files | 2,283 |
| Pre-open order rows | 1,112,321 |
| Pre-open order volume | 4,675,764,535 |
| Opening trade rows | 200,882 |
| Opening trade volume | 451,384,210 |

This day's scan took about 26 seconds. Most time was spent reading the large linked order, trade, and monthly snapshot files. Scan the full 1,282 pre-open files in date ranges to avoid holding resources for a single long-running job. The coverage index spreads the I/O cost across future audits, but does not reduce the cost of the first read of raw orders and trades.

## Code boundary

`ticknet.simulator.coverage` reads linked files once per day. Order ticker-presence checks scan only target pre-open stocks and stop when all targets are found. Trades and snapshots are filtered by target stock and date before aggregation, avoiding repeated reads of a daily file for each stock.

Complete files do not prove a complete identity chain. A stock's presence in all three files does not establish that every opening order can be traced by identity. Further work should sample Shanghai stock-days from this inventory and use lag scans and order-level traces to analyze lag by year, month, file batch, and stock.

## Next steps

Finish the full coverage inventory, then scan lags from -200 ms to +200 ms for Shanghai stock-days with pre-open records. A lag should enter market configuration only if it is stable across dates, materially improves ten-level matching, and also reconciles both `Volume` and `DealNum`.
