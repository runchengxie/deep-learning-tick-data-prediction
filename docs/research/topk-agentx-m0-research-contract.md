# M0 Research Contract and Data-Access Audit

This page records the data-access boundaries frozen at M0 completion. M0–M2 are complete; current progress is in [project status](../project-status.md).

## Decision

The Top-K and AgentX research series uses protocol `topk-agentx-v1`:

- 2021–2024 are research-development data; 2025 is seen rolling-validation data.
- Since 2025 has informed new hypotheses, it is not an unseen locked test for this series.
- All data from 2026-01-01 onward is locked and must not be read by the Research Runner.
- As of 2026-08-08, 2026 data was fully aligned only through 2026-04-24 (73 trading days), below the 120-day minimum for final confirmation. The locked period remains sealed; no formal evaluation is authorized.

The versioned configuration is `configs/research-protocol-topk-v1.yaml`. The default `ResearchProtocol` uses the same boundary and can also load versioned YAML. References to a locked 2025 period in older experiments describe the historical protocol at that time.

## Audit of 2026 data availability

The audit read filenames, Parquet schemas, metadata, and date boundaries only. It did not read 2026 returns or run models. Data root: `/mnt/data/hdd6t/quant-data-lake`.

| Source | Available 2026 range | Finding |
|---|---|---|
| Raw monthly snapshots | 202601, 202602, 202603 | Metadata readable for all three months |
| Raw daily snapshots | 2026-04-01–04-24, 06-08–06-30, 07-01–07-24 | May missing; gaps after April |
| Minute snapshot cache | 2026-01-05–04-24, 73 days | Same current upper bound as label inputs |
| Minute order cache | 2026-01-05–04-24, 73 days | Same current upper bound as label inputs |
| Minute trade cache | 2026-01-05–04-24, 73 days | Same current upper bound as label inputs |
| Raw minbar | 2026-01-05 09:31–2026-04-24 14:57 | Metadata contains 88,506,750 rows |
| Open, close, volume wide tables | 2016-01-04–2026-04-24 | All three tables share this exact upper bound |
| Daily-basic partitions | 2026-01-05–04-24, 73 days | Available for liquidity universe and state joins |
| CSI All Share proxy benchmark | Through 2026-04-24 | Open and close available for concurrent returns |

Each of the three Q1 2026 minute caches has 56 trading days. The raw monthly snapshot files for January, February, and March contain 448,705,364, 303,096,683, and 478,486,263 rows; their Parquet metadata is readable.

### Access rules

The available data suggests 2026 can serve as a final out-of-sample period, but it is too short to unlock. Rules:

1. M1–M8 use manifests whose latest trading day is no later than 2025-12-31.
2. Data engineering may complete the 2026 dataset, but it cannot inform feature selection, model selection, or threshold tuning.
3. Request one independent locked-test approval only after snapshots, minute caches, daily bars, trading states, and benchmark align for at least 120 trading days.
4. Approval binds protocol version, experiment ID, checkpoint SHA-256, prediction SHA-256, and dataset fingerprint.

## Trading contract

The unified configuration is `configs/topk-portfolio-v1.yaml`.

### Universe and signal

- Select a dynamic Top-400 universe using a trailing 20-day turnover proxy available before the signal time.
- Require at least 15 valid historical observations.
- Dynamic Top-100 is for smoke tests only and cannot support formal trading conclusions.
- Models emit scores at 14:55 on day T.

### Execution and returns

The formal daily strategy is:

```text
Score at 14:55 on T
  → rebalance at the T+1 open
  → hold until the next rebalance at the T+2 open
  → measure return from T+1 open to T+2 open
```

This aligns continuing holdings, constituent turnover, and the return interval. The benchmark uses the concurrent open-to-open CSI All Share proxy return.

Existing next-open-to-same-close labels remain diagnostic. A strategy using them must assume it enters at each open and exits at each close, charging full round-trip costs on all positions. Differences in adjacent Top-K membership alone are not a valid proxy for all transaction costs.

### Portfolio and costs

- Equal-weight, long-only portfolio.
- `K = 25 / 50 / 75 / 100`.
- Exit buffer `0 / 10 / 25 / 50`.
- One-way commission and impact `5 / 10 / 15 / 20 bp`.
- Sell stamp duty `5 bp`.
- Include initial entry costs.

New positions must be tradable at the T+1 open; do not assume fills for suspensions or limit-up/down stocks. If an existing holding cannot be sold, keep it in the portfolio. Do not silently remove it. Data without trading-state fields is smoke-test-only, not formal evidence.

## Frozen baselines

The baseline manifest is [baselines/topk-agentx-v1.json](../baselines/topk-agentx-v1.json), captured from commit `bc9949825ff1a37dbfc2e4e96bcd91c46221333c`. It records file sizes, SHA-256 hashes, and key historical metrics.

Frozen local evidence includes 2022–2025 rolling minute HGB results, 2025 OOS prediction rows, the 2024 three-seed minute TCN locked-test summary, and the aggregated Logistic baseline from the raw-200 pilot. The complete five-seed raw-200 DeepLOB locked-test result exists only in the external Drive history; the repository has no corresponding verifiable result file, so no SHA-256 is fabricated. Add it to the baseline manifest only after retrieving and verifying it locally.

## M0 acceptance

Research and locked dates are separated in a versioned protocol; the default `ResearchProtocol` deterministically rejects 2026 manifests; universe, signal, execution, return, cost, and untradable rules are fixed; key local artifacts have SHA-256 records; and the incomplete 2026 locked period remains sealed. M0 is complete. M1 implements the fixed-K long-only evaluation engine.
