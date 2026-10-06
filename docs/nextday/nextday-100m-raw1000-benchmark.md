# 100M Parameters × raw-1000 Top-100 Capacity Benchmark

This report records throughput and capacity measurements from 2026-08-11. A later controlled 2×2, three-seed matrix found no stable validation gain from either the 100M model or the raw-1000 window. `1M/raw-200` remains the only candidate. See the [multi-horizon and data expansion roadmap](multi-horizon-data-expansion-roadmap.md) for the current research decision.

## Findings

The 100,817,575-parameter `ChunkedDeepLOB` completed T4 and A100 training benchmarks on the same raw-1000 Top-100 preflight dataset. A single A100 session also swept physical batch sizes 2, 4, 8, 16, and 32. None ran out of memory. The best result used physical batch 32, gradient accumulation 1, and effective batch 32: 370.10 samples/s with 6.49 GiB peak reserved memory.

The initial capacity comparison measured 80.23 samples/s on A100, 3.80× the T4 throughput. Extrapolating 30 epochs over an estimated 75,000 training samples gave 7.79 hours per seed on A100 and 29.58 hours on T4 at physical batch 2. Batch 32 was 4.73× faster than batch 2 in the sweep, reducing the 75,000-sample estimate to 1.69 hours per seed. Recomputing with the final pilot count of 70,805 training samples gives 1.59 hours per seed, or about 4.78 GPU hours for three seeds. A100 was the preferred device for a full 100M run; T4 was suitable for smoke tests or smaller models. Checkpoints and resume support were required.

## Device comparison

| Metric | T4 | A100 |
|---|---:|---:|
| GPU | Tesla T4 | NVIDIA A100-SXM4-40GB |
| Measured batches / samples | 100 / 200 | 100 / 200 |
| Throughput (samples/s) | 21.13 | 80.23 |
| Training time for 100 batches (s) | 9.47 | 2.49 |
| Peak allocated memory (GiB) | 2.19 | 2.18 |
| Peak reserved memory (GiB) | 2.40 | 2.35 |
| Total GPU memory (GiB) | 14.56 | 39.49 |
| Estimated epoch for 75k samples (min) | 59.16 | 15.58 |
| Estimated 30 epochs per seed (h) | 29.58 | 7.79 |
| Estimated GPU hours for three seeds | 88.73 | 23.37 |

Both runs used source revision `09988b7cd3ff722ff075bef241e570f9178684e7`, dataset fingerprint `5b7ec4f0aac03847a29a43ac8266c60b16d076e940e0e30cf8dc3227af947406`, and 100,817,575 parameters. Each input contained `10 × 100 × 40` values (1,000 snapshots). The physical batch was 2, gradient accumulation was 16, and effective batch was 32. AMP, classification and regression losses, backpropagation, and AdamW updates were enabled. Neither the visible 2025 development period nor the locked 2026 period was accessed.

## A100 batch-size sweep

The sweep held effective batch size at 32 and used five warmup batches plus 50 measured batches at each physical batch size. GPU, parameter count, and dataset fingerprint matched the capacity benchmark.

| Physical batch | Accumulation | Samples/s | Relative to batch 2 | Peak allocated GiB | Peak reserved GiB | Hours for 75k × 30 epochs / seed | Hours for 70,805 × 30 epochs / seed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 78.22 | 1.00× | 2.18 | 2.35 | 7.99 | 7.54 |
| 4 | 8 | 138.83 | 1.77× | 2.39 | 2.53 | 4.50 | 4.25 |
| 8 | 4 | 227.10 | 2.90× | 2.80 | 3.09 | 2.75 | 2.60 |
| 16 | 2 | 303.54 | 3.88× | 3.87 | 4.46 | 2.06 | 1.94 |
| 32 | 1 | 370.10 | 4.73× | 5.97 | 6.49 | 1.69 | 1.59 |

The sweep used source revision `2891c4d37461c7cc13de1338f0951cecc40dbccc`. From session creation to artifact retrieval and shutdown, the runner took about 177 seconds. The five measured loops totaled about 11.44 seconds; the rest covered wheel building, runtime creation, dependency installation, data transfer, and artifact collection. The Colab CLI did not report account compute-unit usage, so this record reports only verifiable wall-clock and GPU measurements.

## Data preflight

Source snapshots arrive about every three seconds. The raw-200 scan from 14:30 to 14:55 contained a median of 500 snapshots and a maximum of 503, too few for raw-1000. The scan therefore began at 13:30, while extraction still selected only the final 1,000 valid events before the 14:55 signal time.

For the corrected January 2021 Top-100 preflight, 1,947 of 2,000 requested samples were written: 16 lacked snapshots and 37 had too few events. Every written sample had exactly 1,000 valid events. Class counts were 390, 1,170, and 387. Of these, 1,848 samples remained strictly within the January training split; month-end labels crossing the split boundary were purged. The float16 NPY shard contained 155,760,128 bytes (about 150 MiB). Local SHA-256 validation and `rclone check --checksum` against Drive both passed.

## Limits of the estimates

The 30-epoch projections exclude per-epoch validation, checkpoint writes, Colab staging, resource wait time, and early stopping. The original benchmark estimated 75,000 training samples before data generation finished; results above also include the final count of 70,805. Each sweep setting ran 50 physical batches, so the number of samples and optimizer steps varied by setting. Throughput includes each setting's actual optimizer cadence and is useful for capacity screening. A complete first epoch is still needed to estimate formal training time.

Raw JSON, execution notebooks, and completion markers remain in Drive and the Linux artifact directory. Per-sample data and credentials are not stored in Git.
