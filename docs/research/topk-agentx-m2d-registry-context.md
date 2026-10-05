# M2d: Registry-Driven ResearchContext

M2d removed the need for `agent-step` to receive manually supplied anomaly and prediction arguments. Brainstorm and Critic now consume one versioned context built from Registry, containing baselines, recent experiments, failure reasons, Evaluation decisions, Audit anomalies, parent relationships, available executors, budget, and data permissions.

This creates a deterministic feedback layer; it does not yet provide M8's multi-candidate research intelligence.

## Context sources

`ResearchContextBuilder` reads Registry only. It does not scan the source repository, market-data files, or locked predictions. Context includes:

- Baseline experiment ID, status, decision, data fingerprint, mean primary metrics, and mean of all registered metrics.
- Recent experiments' parent, hypothesis, executor, status, decision, error, and primary metrics.
- Negative results with `failed` or `rejected` state or Evaluation decision `DISCARD`.
- Anomalies from `audit_anomalies` reviews, with source experiment and decision.
- Full-history `novelty_signature` values to block duplicate experiments.
- Implemented executors, allowed experiment types, and current compute budget.
- Fingerprints already observed in Registry and locked-test access fixed to false.

Automatic baseline selection prefers the latest `KEEP`, otherwise the latest `EXTEND`. `--baseline-id` can select one explicitly, but it must be completed with `KEEP` or `EXTEND`. A discarded, failed, or still-running experiment cannot be an inherited baseline.

If a source experiment has a unique seed-0 prediction artifact, context records its Registry experiment ID, seed, artifact name, and path. For a template responding to an extreme-return anomaly, the loop first invokes `export_predictions`, checks the Registry checksum, and lets the Runner run Audit. A path does not expand permissions. All prediction inputs still undergo date and locked-test checks; the Registry export path additionally binds the source checksum.

## Replayable fingerprint

ResearchContext uses schema version 1. All fields are encoded with stable JSON before SHA-256; nondeterministic values such as current time are excluded. The same Registry state, question, baseline choice, and budget produce the same fingerprint.

After a proposal is registered, the Orchestrator writes a `research_context` review containing the full context and fingerprint. Critic review, ExperimentSpec, and final result can therefore be traced to one input snapshot. If Brainstorm changes the context, the Orchestrator rejects the run before reservation. Without an explicit experiment ID, it selects the first unused `EXP-AUTO-NNNN` from Registry, so restarting the CLI does not reuse `EXP-AUTO-0001`.

Preview context:

```bash
ticknet-research \
  --registry results/registry.sqlite \
  context \
  --question "Can a Top-K buffer cover 10 bp one-way costs?" \
  --baseline-id EXP-BASE \
  --recent-limit 10 \
  --compute-budget-hours 4
```

Run one Agent step with the same inputs:

```bash
ticknet-research \
  --registry results/registry.sqlite \
  agent-step \
  --question "Can a Top-K buffer cover 10 bp one-way costs?" \
  --baseline-id EXP-BASE \
  --recent-limit 10 \
  --compute-budget-hours 4
```

Output includes `context_fingerprint`. `--anomaly` and `--predictions` are no longer manually injected by the CLI; evidence must first be registered in Registry.

## Brainstorm and Critic constraints

Brainstorm checks all historical novelty signatures in both template and LLM paths. A duplicate candidate fails before experiment reservation, so no new Registry experiment is created. Templates skip anomaly signatures already handled.

Critic independently checks that the experiment type is allowed by context; the executor is implemented (`train_ranker` cannot be represented as available); the novelty signature is new; the ExperimentSpec timeout fits the compute budget; and existing semantic routing and ExperimentSpec validation pass.

Policy, Runner, and locked approval retain final authority. ResearchContext or LLM output cannot add executors, change data dates, or issue a locked-test token.

## Current limits

- Registry stores dataset fingerprints but cannot infer exact date ranges, so context does not invent observed dates.
- Audit anomalies have no separate resolved lifecycle. Existing novelty signatures prevent duplicate handling experiments; anomalies remain historical evidence.
- M8's multi-candidate generation, ranking, relevant-history retrieval, and causal-chain review are not implemented.
- `train_ranker` remains unimplemented; see [M2a](topk-agentx-m2a-deterministic-loop.md).

The M2 deterministic loop is complete. M3 uses existing prediction artifacts for Top-K, buffer, and cost-sensitivity diagnostics.
