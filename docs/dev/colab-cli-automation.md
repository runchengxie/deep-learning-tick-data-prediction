# Unattended Colab Runs with the CLI

The Linux development machine schedules code, data, and experiment artifacts. Colab provides temporary GPU compute. The supported entry point is the Python CLI. Former notebook snapshots are retained as text under `docs/archive/historical-workflows/`; their maintained replacements are listed in the [migration guide](historical-colab-snapshots.md).

## Boundaries

- Linux holds Git worktrees, raw-200 data, local archives, and Colab and rclone OAuth credentials.
- A Colab VM holds only the data, wheel, temporary credentials, and run output needed for one session.
- Google Drive stores training data, checkpoints, and experiment artifacts shared between devices.
- `rclone.conf` contains refresh credentials and must stay outside the repository.
- Earlier raw-order-book capacity experiments keep the 2025 test period locked. Current event-stream and AgentX series use 2025 for development and lock 2026 through their research protocol. Automation supports multi-horizon evaluation, standalone H=5, capacity matrices, and event-stream benchmarks.

As of 2026-08-19, the raw-order-book four-cell, three-seed matrix, event-stream input benchmarks, recent-fold formal three-seed training, and adjacent-fold seed 0 had completed. Commands are retained here for reproduction. The current additional formal workflow is a two-fold multi-task gradient audit.

## Install on the Linux host

PyPI metadata for Colab CLI 0.6.0 does not pin Google's fork of the kernel client. After installing it, replace the PyPI 1.x package with the version from Google's lockfile. Otherwise, `colab exec` cannot find `KernelClient`:

```bash
uv tool install --force google-colab-cli==0.6.0 \
  --with 'jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git@f18e982c3265df5e923aa9def101ab3fd737e139'
```

Check OAuth and rclone. Both commands should run without interactive input:

```bash
colab --auth=oauth2 sessions
rclone --config ~/.config/rclone/rclone.conf \
  lsjson gdrive:deep-learning-tick-data-prediction/ticknet-data/nextday-raw-200/manifest.json
```

The runner also searches `~/.local/bin` in non-login SSH sessions, so it does not depend on an interactive shell's `PATH`. Colab's Ubuntu apt sources provide an older rclone, so commands inside the VM use compatible options.

## Python entry point

Use the three best checkpoints and evaluate only 2024 validation:

```bash
ticknet-nextday-evaluate-horizons \
  --config configs/nextday-raw-200-capacity-1m.yaml \
  --sidecar /content/nextday-raw-200-targets-v1/horizon-labels.json \
  --output-dir /content/ticknet-results/multi-horizon-validation-2024 \
  --seeds 0 1 2 \
  --horizons 1 3 5 \
  --source-revision "$(git rev-parse HEAD)"
```

The configuration's `manifest_path` and `checkpoint_dir` must match the original checkpoint signature. The runner uses rclone to sync data into those paths and does not change checkpoint matching rules.

## Linux scheduling entry point

Event-stream training downloads filter filenames containing `.seedN.` according to the current `--seeds` selection. Shared summaries are downloaded as usual. Checkpoints and results for other seeds stay in their original Drive directory. `--dry-run` shows the same filter. Existing local copies are not deleted automatically; review their contents, references, and recovery needs before cleanup.

First run a dry run that does not request a GPU:

```bash
python scripts/run_colab_nextday.py \
  --dry-run \
  --session ticknet-multi-horizon \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/latest
```

After reviewing the command, run it:

```bash
python scripts/run_colab_nextday.py \
  --session ticknet-multi-horizon \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/$(date +%Y%m%d-%H%M%S)
```

Standalone Stage C H=5 seed-0 training does not require a notebook:

```bash
python scripts/run_colab_nextday.py \
  --workflow h5-train \
  --seeds 0 \
  --keep-on-failure \
  --session ticknet-h5-seed0 \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m-h5/seed0
```

`h5-train` defaults to `configs/nextday-raw-200-capacity-1m-h5.yaml`. It restores an existing same-name checkpoint from Drive to its fixed path, so the same seed can resume after interruption. After every training run or failure, the runner attempts to sync the checkpoint, history, result, and `colab-run-summary.json` back to Drive.

### Five-year raw-1000 Top-100 training

The 100M model uses A100 and batch size 32. It trains only on 2021–2023, selects checkpoints using 2024 validation, and does not evaluate the 2025 test period. All three seeds are complete. The command below reproduces seed 0:

```bash
python scripts/run_colab_nextday.py \
  --workflow raw1000-train \
  --seeds 0 \
  --keep-on-failure \
  --session ticknet-100m-raw1000-seed0 \
  --gpu A100 \
  --local-output-dir artifacts/raw-1000-top100-capacity_100m/training
```

`raw1000-train` defaults to `configs/nextday-raw-1000-top100-capacity-100m.yaml`. It downloads the full 8.84 GiB working set from Drive and restores a checkpoint already present in the same training directory before starting. Use the same command with `--seeds` changed for seeds 1 and 2. None of the three checkpoint selections read the 2025 test period.

### Capacity and window matrix

The 2×2 capacity/window matrix shares the Top-100 working set above. For the raw-200 cell, `input_last_chunks: 2` reads only the last two 100-event chunks per sample. Stock-days, labels, and underlying data fingerprints are unchanged, and shards are not copied. All four cells and three seeds are complete. The command below reproduces seed 0 for `1M/raw-200`:

```bash
python scripts/run_colab_nextday.py \
  --workflow capacity-matrix-train \
  --matrix-cell 1m-raw200 \
  --seeds 0 \
  --keep-on-failure \
  --session ticknet-matrix-1m-raw200-seed0 \
  --gpu A100 \
  --local-output-dir artifacts/raw-1000-top100-capacity-matrix/1m-raw200
```

`--matrix-cell` also accepts `1m-raw1000` and `100m-raw200`. The three cells read `configs/nextday-capacity-matrix-1m-raw200.yaml`, `configs/nextday-capacity-matrix-1m-raw1000.yaml`, and `configs/nextday-capacity-matrix-100m-raw200.yaml`, respectively. They use batch size 32, learning rate 0.0001, patience 8, and 2024 validation for checkpoint selection. The 2025 test remains locked. Each cell uses a separate Drive directory and supports recovery.

### 100M benchmark

First compare T4 and A100 on a one-month raw-1000 preflight using the same source revision:

```bash
python scripts/run_colab_nextday.py \
  --workflow capacity-benchmark \
  --session ticknet-100m-raw1000-t4 \
  --gpu T4 \
  --benchmark-batches 100 \
  --warmup-batches 5 \
  --local-output-dir artifacts/raw-1000-top100-capacity_100m/benchmarks/t4
```

Use `A100` for `--gpu`, the session, and the final output-directory component to obtain a comparable result. The default configuration is `configs/nextday-raw-1000-top100-capacity-100m-benchmark.yaml`; the exact parameter count is 100,817,575. The benchmark runs AMP forward and backward passes and AdamW updates. It does not access validation or test data. An early estimate based on 75,000 training samples was recalculated using the actual 70,805 samples after the data was complete.

### Initial Top-400 full-day event-stream H5 fold

The first Top-400 full-day event-stream H5 fold uses a separate workflow. Drive needs the January 2021 benchmark pack and fold-level H5 labels. A100 runs the event-stream model with exactly 100,604,180 parameters:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-capacity-benchmark \
  --session ticknet-eventstream-h5-100m-a100 \
  --gpu A100 \
  --benchmark-batches 100 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-fold0/benchmarks/a100
```

The default configuration is `configs/eventstream-h5-fold0-capacity100m-colab.yaml`. The workflow builds only the January 2021 training set and does not read April 2021 validation or May 2021 OOS data.

The 2021 result is an infrastructure-throughput baseline only. For the formal recent fold, upload the August 2025 pack and run the same benchmark:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-capacity-benchmark \
  --session ticknet-eventstream-h5-recent-100m-a100 \
  --gpu A100 \
  --benchmark-batches 100 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/benchmarks/a100
```

The recent-fold workflow defaults to `configs/eventstream-h5-recent-capacity100m-colab.yaml` and accesses only the August 2025 training pack. November 2025 validation, December 2025 OOS, and the 2026 locked period are excluded from the benchmark.

### Recent-fold formal training

Formal recent-fold training uses a fixed materialized-window directory. Run one seed at a time. First use one epoch to validate recovery without reading OOS:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-train \
  --session ticknet-eventstream-h5-recent-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/training/seed0
```

Before training, the workflow checks the materialization manifest and every SHA-256 for permitted shards, then restores the same seed's checkpoint from Drive. The short run excludes OOS and H3 OOS shards. Success or failure syncs the checkpoint, history, result, materialization preflight, and `colab-run-summary.json` back. After the short run passes, set `--training-epochs` to `20`, use a new session name, and add `--evaluate-test`.

### Additional rolling fold

Additional rolling folds use separate paths and record the fold identifier in the run summary. This command starts a short recovery check for seed 0 of `fold-54-oos-202511`:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-rolling-train \
  --eventstream-fold-id fold-54-oos-202511 \
  --session ticknet-eventstream-fold54-seed0-a100 \
  --gpu A100 \
  --seeds 0 \
  --training-epochs 1 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-fold54/training/seed0
```

The default configuration is resolved from the fold identifier. A new fold requires its fixed-date configuration to be committed first. The runner rejects path separators and identifiers that do not match `fold-NN-oos-YYYYMM`.

### Multi-task gradient audit

The multi-task gradient audit reads validation only. Both the recent and adjacent fold use seed 0, 16 batches, and a best checkpoint with a registered SHA-256. Recent-fold command:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-gradient-audit \
  --session ticknet-gradient-audit-recent-seed0 \
  --gpu A100 \
  --seeds 0 \
  --audit-batches 16 \
  --no-evaluate-test \
  --timeout 7200 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/recent-seed0
```

For the adjacent fold, use workflow `eventstream-rolling-gradient-audit` and add `--eventstream-fold-id fold-54-oos-202511`. The workflow stages only validation shards and one checkpoint. Training, OOS, monitoring partitions, and the 2026 locked period are not copied to Colab. See the [event-stream multi-task gradient audit](../research/eventstream-gradient-audit.md) for decision thresholds.

### Event-stream input profiling

If throughput does not increase with physical batch size, measure DataLoader and GPU performance separately using the same August 2025 pack and scan worker counts:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-input-profile \
  --session ticknet-eventstream-h5-recent-input-a100 \
  --gpu A100 \
  --num-workers 2 4 8 16 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/input-profile/a100
```

Output records DataLoader-only, GPU-only with preloaded batches, and true end-to-end throughput. Select worker count by end-to-end throughput. This workflow does not read validation, OOS, or 2026 locked data.

The optimized run on 2026-08-12 selected eight workers. DataLoader-only throughput was 140.48 samples/s, end-to-end throughput 149.40 samples/s, and GPU-only throughput 238.79 samples/s. Extrapolated to 120,000 samples and 20 epochs, each seed takes about 4.46 hours. The formal recent configuration uses `num_workers: 8`.

The recent event-stream sweep reuses the same August 2025 staged data and tests batch sizes 8, 16, 32, and 64 in one A100 session. Estimates use all 120,000 training samples across three months:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-batch-size-sweep \
  --session ticknet-eventstream-h5-recent-sweep-a100 \
  --gpu A100 \
  --batch-sizes 8 16 32 64 \
  --effective-batch-size 64 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/eventstream-h5-recent-fold/batch-size-sweep/a100
```

Before formal training, a single A100 session can scan physical batch sizes 2, 4, 8, 16, and 32 with effective batch size 32, five warmup batches, and 50 measured batches per size. If one size runs out of memory, it records the failure and continues. The best successful throughput is selected:

```bash
python scripts/run_colab_nextday.py \
  --workflow batch-size-sweep \
  --session ticknet-100m-batch-sweep-a100 \
  --gpu A100 \
  --batch-sizes 2 4 8 16 32 \
  --effective-batch-size 32 \
  --benchmark-batches 50 \
  --warmup-batches 5 \
  --keep-on-failure \
  --local-output-dir artifacts/raw-1000-top100-capacity_100m/batch-size-sweep/a100
```

The runner:

1. Requires the current worktree to be committed and clean, and records the exact commit.
2. Queries for a session with the same name. It must not exist unless `--reuse-session` is explicit.
3. Uses `git archive` to build a wheel for that commit in a temporary directory without modifying the worktree. It creates a named Colab GPU runtime only when a new session is needed.
4. Uploads the wheel, fixed training configuration, job specification, and temporary `rclone.conf`.
5. Downloads workflow-specific data from Drive. Multi-horizon and H=5 use raw-200 and sidecar labels. The 100M benchmark downloads raw-1000 preflight data, the January 2021 event-stream pack, or the August 2025 recent pack.
6. Runs multi-horizon validation, standalone H=5 training, formal raw-1000 training, or the selected 100M benchmark.
7. Syncs JSON and Parquet results to Drive, then to the Linux artifact directory.
8. Exports a CLI execution notebook, deletes the temporary rclone configuration, and applies the session lifecycle policy.

## Session lifecycle

- Default: create an ephemeral session and close it after success or failure.
- `--keep-on-failure`: close the session on success and retain it after failure for debugging.
- `--keep-session`: retain this newly created session after either success or failure.
- `--reuse-session`: require an existing same-name session and reuse it. The runner never closes a reused session.

`--keep-session` and `--keep-on-failure` are mutually exclusive. If a same-name session exists without `--reuse-session`, the runner refuses before uploading files. If `--reuse-session` is supplied but the session does not exist, it also refuses. All modes delete the temporary rclone configuration uploaded by this run.

Retain a session after failure:

```bash
python scripts/run_colab_nextday.py \
  --keep-on-failure \
  --session ticknet-multi-horizon \
  --gpu T4 \
  --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/debug
```

Reuse a retained runtime:

```bash
python scripts/run_colab_nextday.py \
  --reuse-session \
  --session ticknet-multi-horizon \
  --local-output-dir artifacts/raw-200-capacity_1m/cli-runs/reuse
```

A retained VM continues consuming compute units. Explicitly stop it when it is no longer needed:

```bash
colab --auth=oauth2 stop -s ticknet-multi-horizon
```

## Why raw-200 is not uploaded directly

Colab upload is suitable for small files such as wheels, YAML, and JSON. It sends files through the Jupyter Contents API and base64-encodes binary data. Raw-200 is about 7.2 GB, so direct upload increases memory use, request size, and retry cost. Downloading directly from Drive inside Colab with rclone supports checksums, parallel transfer, and repeatable runs.

## Credential security

- Do not commit `rclone.conf`, Colab tokens, or session metadata.
- The runner rejects an rclone configuration stored inside the Git repository.
- Job specifications record remote names and paths, never token values.
- Colab's rclone configuration has mode 600 and is deleted immediately after the job.
- Colab reclaims temporary disk after the runtime stops.

## Run artifacts

Drive paths:

```text
deep-learning-tick-data-prediction/
  ticknet-runs/raw-200-capacity_1m/multi-horizon-validation-2024/
  ticknet-runs/raw-200-capacity_1m-h5/
```

The Linux `--local-output-dir` contains:

```text
multi_horizon_validation_2024.json
daily_rank_ic_2024.parquet
validation_scores_2024.parquet
execution.ipynb
```

The H=5 training directory also contains each seed's last and best checkpoint, history, result, and `colab-run-summary.json`.

The 100M benchmark's GPU-specific directory contains `capacity-benchmark.json`, `colab-run-summary.json`, and `execution.ipynb`. JSON records the actual GPU, exact parameter count, data fingerprint, throughput, peak GPU memory, and single- and three-seed estimates for 75,000 training samples.

The batch-size sweep directory contains a `batch-NN.json` for each size, a `batch-size-sweep.json` summary, `colab-run-summary.json`, and `execution.ipynb`. The summary records gradient accumulation, throughput, memory, speedup relative to the smallest successful batch, and the final selection.

The event-stream input-profile directory contains `gpu-only.json`, per-worker data-only and end-to-end JSON, `input-profile.json`, `colab-run-summary.json`, and `execution.ipynb`.

Execution history comes from official Colab logs, so a person does not need to open or save the notebook manually.

## Resuming event-stream VQ training

Existing event-stream checkpoints were trained without VQ. `EventstreamConfig.init_checkpoint` can warm-start a new experiment with VQ enabled: trunk weights load as-is, VQ modules (`vq_encoder`, `vector_quantizer`, `vq_proj`) are initialized randomly, and the training configuration sets `use_vq: true`.

Warm start and `resume` precedence: if this experiment's own last checkpoint exists, resume it first. Otherwise load `init_checkpoint`. If neither exists, train from scratch.

Recommended low-cost path: first validate the entire pipeline locally or on a free-tier GPU using a small model and a few days of packed data. Confirm `vq_loss` converges, then use A100 with the `capacity100m` configuration. A100 costs about 15 compute units per hour.
