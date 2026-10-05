# M2a: Deterministic AgentX Experiment Loop

## What changed

M2a replaced arbitrary proposal entry points and a generic training runner with a controlled loop:

```text
ExperimentSpec v2
  → reserve experiment
  → Critic / Policy
  → typed executor
  → artifact contract
  → prediction Audit
  → Registry v2
  → metric gates
  → KEEP / EXTEND / DISCARD
```

This improves experiment semantics and evidence reliability. It does not establish a new model-return result. This page records the M2a release state. M2b, M2c, and M2d are now complete; see the [overall research roadmap](topk-agentx-research-roadmap.md).

## ExperimentSpec v2

Each spec must declare `executor`, `inputs`, `primary_metrics`, structured `success_gates`, `artifact_contract`, `budget`, `parent_id`, `novelty_signature`, and `stage`. YAML and LLM JSON use the same strict parser. Unknown fields and legacy `entry_point` fail validation.

```yaml
hypothesis: A Top-50 buffer can retain positive net returns after 10 bp costs
objective: Run a fixed portfolio-cost sweep on existing predictions
experiment_type: cost_analysis
executor: topk_cost_sweep
inputs:
  predictions_path: results/predictions.parquet
  top_k: [50]
  exit_buffer: [20]
  cost_bps: [10]
seeds: [0]
primary_metrics:
  - topk.k50.buffer20.cost10.net.sharpe
success_gates:
  - metric: topk.k50.buffer20.cost10.net.sharpe
    operator: gt
    threshold: 0.0
artifact_contract: [resolved_spec, resolved_config, stdout, stderr, result, run_manifest, topk_sweep]
budget:
  timeout_seconds: 600
  max_seeds: 1
rationale: Validate portfolio rules without retraining a model
falsification_condition: Net Sharpe at 10 bp one-way cost is not greater than zero
novelty_signature: top50-buffer20-cost10-v1
stage: screening
```

## Executors

At M2a release, supported executors were `train_nextday` (fixed next-day training entry), `train_minute_tcn` (fixed minute-TCN entry), `audit_predictions` (audit only, no training), `topk_cost_sweep` (calls M1's `evaluate_topk_portfolio()` across K, buffer, and cost), and `compare_experiments` (reads registered metrics and creates a basic comparison artifact).

M2c later added `import_predictions`, `export_predictions`, `walk_forward_robustness`, and a fuller `compare_experiments`. `train_ranker` still explicitly reports that it is unimplemented. The executor catalog is jointly constrained by `ticknet.research.spec` and `ticknet.research.executors`.

## Artifacts and Registry v2

The Runner reserves an experiment ID before Critic and Policy run. Each experiment and seed has a separate directory; training executors receive isolated checkpoint paths and names. Core artifacts include resolved spec/configuration, environment and Git state, stdout/stderr, result, run manifest, and executor-declared predictions, checkpoints, Audit, or cost details.

Registry enforces uniqueness for experiments, runs, metrics, reviews, and artifacts; parent experiments must exist. Nested numeric metrics are recursively flattened to dotted paths. Artifacts are registered with streaming SHA-256 and file size. Rejected, timed-out, and failed runs retain their terminal state and reason. Existing artifact directories or experiment IDs cannot be overwritten.

## Mandatory Audit, gates, and locked boundary

When a training executor returns a prediction artifact, the Runner first verifies prediction dates do not enter the protocol's locked period, then runs prediction Audit. Explicit `audit_predictions` and `topk_cost_sweep` inputs use the same protocol check. Nested Audit metrics and anomalies are written to Registry.

Evaluation computes each gate across seed means. A missing metric or any failed gate yields `DISCARD`; passing all screening gates yields `EXTEND`; passing robustness and release gates yields `KEEP`. Natural-language `falsification_condition` explains the decision, but only structured gates determine it.

At M2a release, `APPROVED` was still a static token. M2b replaced that historical limitation with content-bound, one-time approval. See [M2b locked approval](topk-agentx-m2b-locked-approval.md).

## Run and acceptance

```bash
ticknet-research run --spec path/to/experiment.yaml --id EXP-TOPK-001
ticknet-research show --id EXP-TOPK-001
ticknet-research compare --ids EXP-BASE-001 EXP-TOPK-001
```

Synthetic end-to-end tests cover cost analysis without training, automatic Audit of training predictions, rejection of locked predictions, recursive metric and artifact registration, KEEP/EXTEND/DISCARD, duplicate IDs, arbitrary entry points, and artifact conflicts.

## Subsequent milestones

1. M2b implemented content-bound one-time locked-test approval.
2. M2c implemented prediction import/export, multi-seed comparisons, and walk-forward summaries.
3. M2d implemented deterministic construction and replay of Registry-derived ResearchContext.
4. A fixed `train_ranker` entry remains unimplemented; current research does not depend on it.
