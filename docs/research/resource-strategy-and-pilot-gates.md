# Research Resource Strategy and Pilot Gates

This page records resource-allocation principles for constrained compute. The early plan assumed Colab Pro and a 100 GB Google Drive plan; Drive was upgraded to 200 GB on 2026-08-10. The minute, raw-book, and event-stream tracks have completed infrastructure or controlled experiments. Current state is in [project status](../project-status.md).

## Principles

Prioritize experiments that can change a research decision:

- Use low-cost baselines first to test whether features contain stable information.
- Fix data, labels, and training contracts; change one major factor at a time.
- Set continuation gates on a small number of seeds and stop when they fail.
- Make checkpoints and intermediate artifacts resumable to avoid repeating work after session loss.
- Run a locked test once, after the model, seed list, and acceptance criteria are fixed.

## Resources and intended use

| Resource | State in 2026-08 | Main use |
|---|---|---|
| Colab GPU | T4 or A100 by session | Neural-network training, throughput benchmarks, and input profiling |
| Google Drive | 200 GB | Current training datasets, checkpoints, and results |
| Remote NVMe | Compact data and temporary artifacts | Materialization, verification, and Colab staging |
| 6 TB data disk | Several TB of raw market data and minute caches | Sequential scans and long-term storage, not random training reads |
| Local CPU | 4 cores, 31 GiB RAM | Data audits, tree models, and small-sample checks |

Keep only currently needed processed data on Drive. Retain raw market data on the data disk. Large event-stream packs exceed 200 GB and require 400 GB Drive, GCS, or monthly streaming staging.

## Verified capacity

Estimates use about 400 stocks and five years of trading days.

| Representation | Approximate size | Current conclusion |
|---|---:|---|
| Last 60 minutes × 33 features | About 4 GB | Suitable for minute-sequence models |
| Full 240-minute day × 33 features | About 16 GB | Test the value of full-day information |
| raw-200 Top-400 | About 7.2 GiB | Generated and evaluated |
| raw-1000 Top-100 | Below the Top-400 estimate | Generated and included in the four-cell matrix |
| Daily 64-dimensional embedding | About 128 MB | Reusable for multi-day models |
| Five months of 2025 event-stream packs | 313.11 GiB | Generated; exceeds the 200 GB Drive capacity |

## Applying resource gates

```text
local sequential scan and materialization
  → integrity and leakage audit
  → low-cost baseline or short throughput test
  → seed 0 under a fixed contract
  → add seeds 1 and 2 only if gates pass
  → request locked-test access only after freezing the candidate
```

The raw-book capacity/window matrix met its stop condition. `1M/raw-200` had the best validation result and lowest seed variation; 100M parameters and raw-1000 did not produce stable gains. Event-stream input profiling is complete; the next high-cost task is formal seed 0 on the recent fold. AgentX M3 should finish minute-feature materialization before formal Top-K cost diagnostics.

## Run records

Save source revision, data fingerprint, configuration, seed, wall-clock duration, peak resources, metrics, and the continue/stop decision for every run. Keep training artifacts in Git-ignored directories or remote storage. Commit aggregate conclusions and reproducible commands only.

Date-specific results are in the [experiment log](experiment-log.md). Full stop rules are in the [hardware and staged experiment roadmap](../nextday/hardware-constraints-and-experiment-roadmap.md).
