# M2c: Prediction Export, Experiment Comparison, and Walk-Forward Robustness

M2c added three deterministic executors: materialize registered predictions from Registry, compare experiments across seeds, and aggregate completed experiments into walk-forward robustness evidence. They consume existing artifacts and metrics; they do not retrain models or broaden locked-data permissions.

## Prediction export boundary

`export_predictions` handles only prediction artifacts already registered in Registry. It does not provide generic inference for arbitrary checkpoints: model inputs, checkpoints, and inference commands do not yet share a universal contract.

1. `source_experiment_id` must exist and be `completed`, `frozen`, or `locked_tested`.
2. The artifact for the requested seed and name must be unique, and its file SHA-256 must match Registry.
3. Parquet must contain `symbol`, `trading_date`, `label_date`, `target_return`, and `score`.
4. Copy original bytes into the new experiment's isolated seed directory and verify SHA-256 again.
5. The Runner rechecks date protocol and runs Audit on the copy; export cannot bypass locked-test checks.

Minimal spec:

```yaml
experiment_type: prediction_export
executor: export_predictions
inputs:
  source_experiment_id: EXP-SOURCE
  source_seed: 0
  artifact_name: predictions
seeds: [0]
```

Checkpoint inference remains the responsibility of fixed training or inference executors. After M4 selects a ranking-model library and standard command, consider adding model-specific export.

## Multi-seed experiment comparison

`compare_experiments` reads at least two completed experiments. For each declared metric, it reports every seed's value, mean, sample standard deviation, minimum and maximum; raw mean difference from baseline (`delta_vs_baseline_mean`); direction-normalized improvement where positive always means better (`improvement_vs_baseline_mean`); and matched-seed count, paired raw difference, and direction-normalized paired improvement.

By default, compared experiments must share a non-empty dataset fingerprint. Use `walk_forward_robustness` for different time windows. A justified diagnostic may set `require_same_fingerprint: false`; output retains each source fingerprint and must not be described as a controlled ablation.

Metrics default to higher-is-better. Explicitly mark Brier score, error, drawdown, and other lower-is-better metrics:

```yaml
experiment_type: comparison
executor: compare_experiments
inputs:
  experiment_ids: [EXP-BASE, EXP-CANDIDATE]
  baseline_id: EXP-BASE
  metrics:
    - validation.daily_rank_ic_mean
    - validation.brier_score
  metric_directions:
    validation.brier_score: lower
seeds: [0]
```

Equivalent CLI:

```bash
ticknet-research compare \
  --ids EXP-BASE EXP-CANDIDATE \
  --baseline EXP-BASE \
  --metrics validation.daily_rank_ic_mean validation.brier_score \
  --lower-is-better validation.brier_score
```

Output is `comparison.json`, retaining each source state, Evaluation decision, and data fingerprint. Its own dataset fingerprint is a stable aggregate hash over source experiment IDs and their fingerprints; it must not masquerade as one source experiment's fingerprint.

## Walk-forward robustness

`walk_forward_robustness` treats each source experiment as one time window, aggregates seeds within each window, then reports mean, sample standard deviation, minimum, maximum, and worst window across windows. By default, require at least three windows with distinct, non-empty fingerprints so repeated registration of one split cannot pose as rolling validation.

Worst-window selection respects metric direction: the lowest window mean for Rank IC, highest for Brier. For gates, higher-is-better metrics generally constrain `window_min`; lower-is-better metrics constrain `window_max`.

```yaml
experiment_type: robustness
executor: walk_forward_robustness
inputs:
  experiment_ids: [EXP-W22, EXP-W23, EXP-W24]
  metrics:
    - validation.daily_rank_ic_mean
    - validation.brier_score
  metric_directions:
    validation.brier_score: lower
  minimum_windows: 3
seeds: [0]
```

Output is `walk-forward.json`. The executor aggregates registered windows but does not generate splits. Source experiments remain responsible for window boundaries, purging, and label protocols.

## Deterministic failures and remaining work

Unknown or incomplete experiments, missing metrics, duplicate IDs, invalid metric direction, too few windows, duplicate fingerprints, missing artifacts, or changed checksums fail deterministically; the Runner records the failed run. `train_ranker` remains explicitly unsupported and never falls back to another training command. See [M2a](topk-agentx-m2a-deterministic-loop.md).

M2d completed Registry-to-ResearchContext feedback; see [M2d](topk-agentx-m2d-registry-context.md).
