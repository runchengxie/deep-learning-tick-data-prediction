# Raw-200 End-to-End Pipeline

This guide describes the workflow from local processing of ten-level snapshots to locked-test evaluation. Resource gates are documented in the [resource strategy](../research/resource-strategy-and-pilot-gates.md). This page preserves the raw-200 pilot execution record; the [project status](../project-status.md) is the source of current status.

## 1. Goal and status

### Goal

For each stock-day, use the final 200 ten-level order-book snapshots available before 14:55. Predict next-day open-to-close excess return and probabilities for down, neutral, and up classes. A shared DeepLOB encoder processes two 100-event chunks; a GRU aggregates the chunks before two prediction heads. The default model has 86,775 parameters; the capacity variant has 1,033,383. Both fit a single-GPU Colab session.

### Status

- The `raw_snapshot.py` preparation path, `train.py` trainer, and raw configurations are implemented. Real training results exist for raw-200, raw-1000, approximately 1M parameters, and 100M parameters.
- Five-year Top-400 raw-200 and Top-100 raw-1000 datasets have been generated. A controlled four-cell, three-seed matrix selected `1M/raw-200` as the sole candidate. Further capacity and window expansion stopped; full results are in the [multi-horizon data expansion roadmap](multi-horizon-data-expansion-roadmap.md).
- Data preparation runs locally; Colab is used for training. The current orchestration entry point is `scripts/run_colab_nextday.py`.
- The local CPU processes about 8.5 stock-days per second. One epoch of the three-year chunked DeepLOB workload takes about 9.8 hours, so formal training requires Colab or a cloud GPU.

## 2. Local data preparation

Training data is built locally from the external drive. Drive stores only the filtered float16 working set, configuration, checkpoints, and results. Raw Parquet files are not uploaded.

Run the pilot preparation:

```bash
.venv/bin/ticknet-nextday-prepare-snapshot --config configs/nextday-raw-pilot.yaml
```

Important parameters in `configs/nextday-raw-pilot.yaml`:

- `start_date` and `end_date`: the 2024 calendar year.
- `scan_start_time_ms`: read events starting at 14:30.
- `signal_time_ms`: 14:55. No later data may enter a sample.
- `chunks_per_sample: 2` and `chunk_size: 100`: divide 200 events into two 100-event chunks.
- `min_valid_events: 200`: discard stock-days with fewer than 200 valid events.
- `top_n: 100`: select 100 stocks dynamically each day, using only information available before the signal time.
- `storage_dtype: float16`: compact storage for the Colab working set.
- `samples_per_shard: 2048`: shard size.

Outputs include `manifest.json` (shard paths, sample rows, SHA-256 hashes, and dataset fingerprint), float16 arrays at `shards/part-*.npy` with layout `samples × chunks × time × 40`, and `data-audit.json` (universe coverage, extraction statistics, and label distribution).

Accept the prepared dataset only when:

- All input timestamps are at or before the 14:55 signal time.
- The input and label dates are adjacent trading days, and date splits do not overlap.
- Each stock-day contributes at most one sample.
- Every shard SHA-256 and manifest fingerprint is complete and can be verified in Colab.

## 3. Stage gates (historical execution sequence)

```text
Local data preparation
  → audit the dataset
  → Logistic baseline on local CPU (confirm the features carry information)
  → Colab throughput and 100-batch run (confirm budget and resume)
  → pilot training: 2024 H1 train / Q3 validation / Q4 locked test
  → compare fixed seeds on validation and freeze the configuration
  → unlock the test for one evaluation only
```

### 3.1 Logistic baseline

Run the Logistic baseline before the deep model. It checks that the shards contain predictive information and provides an initial leakage check.

### 3.2 Colab throughput and 100-batch run

Colab automation is handled by `scripts/run_colab_nextday.py`. Before a formal run, use `--dry-run` to check the session, configuration, data directory, and output directory. The following command shows the single-seed `1M/raw-200` training entry point; the test period remains locked:

```bash
python scripts/run_colab_nextday.py \
  --workflow capacity-matrix-train \
  --matrix-cell 1m-raw200 \
  --seeds 0 \
  --session ticknet-raw200-1m-seed0 \
  --gpu A100 \
  --keep-on-failure \
  --local-output-dir artifacts/raw200-1m/seed0
```

Confirm that the GPU is available (`torch.cuda.is_available()` is true), the working set copies from Drive to `/content` with a matching fingerprint, training resumes from a checkpoint, and the checkpoint is written back to Drive.

### 3.3 Pilot training and locked test

- Train on 2024 H1, validate on Q3, and keep Q4 locked.
- Compare a fixed set of random seeds on validation and freeze the configuration.
- `EVALUATE_LOCKED_TEST` unlocks the test only when an explicit confirmation string is supplied; evaluate it once.
- Report mean and standard deviation of test Rank IC and Macro F1 across seeds. Do not select a seed using test results.

### 3.4 One-million-parameter capacity experiment

`configs/nextday-raw-1m-pilot.yaml` changes only model capacity while reusing the raw-200 pilot inputs, labels, date splits, and training hyperparameters.

| Architecture setting | 86k baseline | 1.03M capacity variant |
|---|---:|---:|
| Convolution width | 16 | 32 |
| Inception branch width | 32 | 64 |
| Chunk embedding | 64 | 320 |
| Day-level GRU hidden size | 64 | 192 |
| Total parameters | 86,775 | 1,033,383 |

An independent YAML file freezes the architecture and data contract. Run seeds 0, 1, and 2 separately through the Colab CLI. 2024 Q4 was already used for pilot results and is development evidence, not a new locked test. Compare validation Rank IC, Macro F1, training time, and cross-seed variation. Stop expanding capacity if training metrics improve without validation gains.

## 4. Colab operating conventions

1. Copy NPY files to `/content` before training; do not perform random reads through a mounted Drive path.
2. Store checkpoints on Drive so interrupted sessions can resume.
3. Evaluate locked tests through the pure evaluation entry point; it does not create an optimizer or load a `last` checkpoint.
4. Use only shards whose SHA-256 values have been checked. Stop if fingerprints differ.

## 5. Stop conditions

Any stop condition means capacity expansion should pause; that itself is a valid research result. See the evaluation and stop rules in the [hardware and experiment roadmap](hardware-constraints-and-experiment-roadmap.md).

## 6. Current actions

1. Freeze the three `1M/raw-200` checkpoints, seed aggregation method, and acceptance metrics.
2. State the trigger and pass criteria for the 2025 locked test in advance.
3. Evaluate 2025 once the gate is met. Do not use test results to select a new model.

The overall sequence is in [project status](../project-status.md). Relevant references are `configs/nextday-raw-pilot.yaml`, `configs/nextday-pilot.yaml`, `configs/nextday-raw-1m-pilot.yaml`, the [Colab CLI guide](../dev/colab-cli-automation.md), and the [raw-data expansion roadmap](raw-data-expansion-roadmap.md). The former interactive entry point is preserved at `examples/historical-workflows/nextday_end_to_end.py`.
