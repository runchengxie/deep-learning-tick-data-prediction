# Event-Stream Multi-Task Gradient Audit

`EVT-GRAD-AUDIT-001` asks whether the daily-return task effectively updates the shared event-stream Transformer backbone. Its result selects the next training experiment without simultaneously changing label scale, supervision position, and task weights.

## Questions and protocol

The model learns next stream type, next order type, continuous-value regression, and H5 daily return. The daily loss currently applies at every valid event position, and all tasks share a Transformer backbone. The audit asks whether the daily gradient is substantially weaker and whether it consistently conflicts with the three generative tasks. Gradient audits describe update strength and direction; they do not replace Rank IC, NDCG, precision, or cost-adjusted return evaluation.

Seed 0 is audited on the recent fold's November 2025 validation split and the adjacent `fold-54-oos-202511` October 2025 validation split. For each fold, select 16 evenly spaced fixed batches of 8 validation samples (128 total). Save sample indices, dataset fingerprint, and full batch-tensor fingerprint. Identical inputs, weights, and source should produce identical results.

Calculate gradients only for shared-backbone parameters, excluding `head_stream`, `head_otype`, `head_reg`, and `head_day`. Use formal loss weights:

| Task | Objective | Weight |
|---|---|---:|
| `stream` | Next stream-type cross-entropy | 1.0 |
| `otype` | Next order-type cross-entropy | 0.5 |
| `reg` | Next-event continuous-value Smooth L1 | 1.0 |
| `day` | H5 daily-return Smooth L1 | 1.0 |

Record weighted and unweighted losses, backbone gradient norms and shares, the daily-gradient norm divided by the median of the three generative-task norms, six pairwise gradient cosine similarities, sample count, valid event positions, valid daily labels, and trading date. Summaries include mean, standard deviation, quantiles, extrema, and negative-cosine share. Initialization and best-checkpoint runs use the same batches with dropout and other training-time randomness disabled.

## Preregistered decision gates

After both folds complete:

1. If the median daily-gradient ratio at both best checkpoints is at most 0.1, run `EVT-LABEL-SCALE-001`.
2. If both folds show the same conflict pair with median cosine at most -0.1 and negative-cosine share at least 75%, run `EVT-SUPERVISION-POSITION-001` and review task weights.
3. If daily gradient strength is normal and no conflict persists, proceed directly to `EVT-SUPERVISION-POSITION-001`.

The label-scale experiment uses seed 0 only, comparing raw H5 returns with daily cross-sectional winsorized z labels on both folds. Add seeds 1 and 2 only if validation, OOS, and top-group metrics improve on both folds.

## Results

Both folds used 16 fixed batches of 8. Source revision: `3e28f04755a881cb72697db2fc50bba031c9f5b0`. Neither locked 2026 nor either fold's OOS entered the runtime.

| Fold | Initialization median daily-gradient ratio | Best-checkpoint median ratio | Best epoch | Persistent negative-cosine pairs |
|---|---:|---:|---:|---|
| Recent fold | 0.61608 | 0.01969 | 4 | None |
| `fold-54-oos-202511` | 0.65568 | 0.03927 | 11 | `reg__day`, `stream__day` |

At initialization, the daily task gradient was similar in scale to the generative tasks. At the best checkpoint, it fell to about 2% and 4% of their median on the two folds, below the 0.1 gate. On the adjacent fold, median `reg__day` cosine was -0.34875 (81.25% negative); `stream__day` median cosine was -0.10432 (75% negative). The recent fold did not show the same conflicts, so evidence did not support changing task weights first.

Result fingerprints: recent fold `2fd3064238b10476a2ddb2a5e54a5155e77b78ab369b7126377866770eb28ccd`; adjacent fold `7ec93b77258d108b673992cd1776e28d652b146cb425a60c8be15e9181bcfe12`; cross-fold decision `9bdef3aad8f9be28f80b0236bfc90f093ce1f3b509d03afaf486f136e1140bbf`. Decision: `day_gradient_weak`; run `EVT-LABEL-SCALE-001`. That experiment is complete; see the [event-stream label-scale study](eventstream-label-scale.md).

## Label-scale experiment contract

`ticknet-eventstream-target-overlay` creates a compact training-label overlay for an existing materialized cache. Event tensors, sampled windows, validation, OOS, and H3 monitoring labels remain unchanged. Only the H5 daily target in training batches changes; validation and OOS continue to use raw H5 returns.

For each training date with H5 labels, clip values to the cross-sectional median ± 5 raw MAD, then standardize using the clipped cross-sectional mean and population standard deviation. Samples without H5 labels because of split boundaries remain masked by `day_valid=0`. The overlay binds the original materialized fingerprint, H5-label SHA-256, daily statistics, and each monthly file's SHA-256. Recent and adjacent folds use separate overlays and training output directories.

## Reproduction

Before running, commit the current worktree and ensure it is clean. Audit the recent fold:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-gradient-audit \
  --session ticknet-gradient-audit-recent-seed0 \
  --gpu A100 --seeds 0 --audit-batches 16 \
  --no-evaluate-test --timeout 7200 --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/recent-seed0
```

Audit the adjacent fold:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-rolling-gradient-audit \
  --eventstream-fold-id fold-54-oos-202511 \
  --session ticknet-gradient-audit-fold54-seed0 \
  --gpu A100 --seeds 0 --audit-batches 16 \
  --no-evaluate-test --timeout 7200 --keep-on-failure \
  --local-output-dir artifacts/eventstream-gradient-audit/fold54-seed0
```

Both workflows stage only validation shards and a seed-0 best checkpoint with registered SHA-256. They exclude training, OOS, monitoring partitions, and locked 2026.

Create the cross-fold decision after results return locally:

```bash
ticknet-eventstream-gradient-audit decide \
  --audit artifacts/eventstream-gradient-audit/recent-seed0/gradient-audit.json \
  --audit artifacts/eventstream-gradient-audit/fold54-seed0/gradient-audit.json \
  --output artifacts/eventstream-gradient-audit/decision.json
```

Build the recent-fold label overlay:

```bash
ticknet-eventstream-target-overlay build \
  --config configs/eventstream-h5-recent-capacity100m.yaml \
  --storage-manifest artifacts/eventstream-h5-recent-fold/storage-manifest.json \
  --materialized-root artifacts/eventstream-h5-recent-fold/materialized/seed0 \
  --output artifacts/eventstream-label-scale/recent-seed0/target-overlay \
  --source-revision "$(git rev-parse HEAD)"
```

Verify and upload the compact overlay:

```bash
rclone --config ~/.config/rclone/rclone.conf copy \
  artifacts/eventstream-label-scale/recent-seed0/target-overlay \
  gdrive:deep-learning-tick-data-prediction/ticknet-data/eventstream-top400-h5-target-overlays/recent/seed0 \
  --checksum
```

Run a short resume check without OOS access:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-label-scale-train \
  --session ticknet-label-scale-recent-seed0 \
  --gpu A100 --seeds 0 --training-epochs 1 \
  --no-evaluate-test --timeout 7200 --keep-on-failure \
  --local-output-dir artifacts/eventstream-label-scale/recent-seed0/training
```

After it passes, set `--training-epochs 20` and use `--evaluate-test`. For the adjacent fold, use `eventstream-rolling-label-scale-train` and add `--eventstream-fold-id fold-54-oos-202511`. Keep checkpoints and results in separate directories.

Label-scale training improved validation and OOS Rank IC on both folds but did not improve the recent-fold OOS extreme-group spread. The preregistered gate was not fully met, so seeds 1 and 2 were paused. The next experiment is `EVT-SUPERVISION-POSITION-001`.
