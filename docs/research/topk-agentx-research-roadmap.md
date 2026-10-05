# Top-K Tradable Prediction and AgentX Research Roadmap

## Objective

This roadmap joins model research and the AgentX research system into an auditable execution chain. It asks whether intraday microstructure improves net returns for the most liquid 50–100 stocks; whether short-horizon order-book representations improve next-day and multi-day cross-sectional forecasts; and whether AgentX can propose, execute, audit, and accumulate reliable research. Experiments proceed from low-cost falsification to higher-cost work. A stage that misses its gate does not unlock dependent work. Historical records remain in the [experiment log](experiment-log.md); this page is the execution and status entry point.

## Evidence to date

As of 2026-08-22:

- Minute aggregate-feature HGB had Rank IC around 0.022–0.035 in four 2022–2025 rolling OOS years. The signal was weak but positive; missing Shanghai order-feed records in H1 2021 biased early coverage.
- In 2025 H2, daily top/bottom deciles turned over about 83%. At 10 bp one-way cost net return was negative; breakeven was about 5–6 bp. The top five days contributed more than all gross spread. Minute TCN validation gains did not generalize; aggregate HGB was steadier.
- A controlled three-seed Top-100 raw-book 2×2 matrix selected `1M/raw-200` (validation Rank IC `0.03748 ± 0.00096`). Neither 100M capacity nor a raw-1000 window produced stable gains.
- Recent-fold event-stream 100M H5 validation ICs were 0.04345, 0.09403, 0.08029 (mean 0.07259); December 2025 OOS was 0.05879, 0.03730, 0.03291 (mean 0.04300). All were positive and passed the pre-set signal gate. Adjacent-fold seed 0 had H5 validation/OOS IC 0.08735/0.03305 and H3 monitoring 0.04840/0.04231; OOS extreme-group spread was -0.00512.
- Frozen event representation with HGB E2 had single-seed OOS IC 0.04333, 0.05644, 0.05912 versus E0 0.04010. Mean-score OOS IC was 0.05701; paired daily increment 0.01691 (bootstrap 95% CI 0.00596–0.02851). Top-100 net active return improved from -11.51 to -4.80 bp/day but remained negative. LambdaMART gains did not repeat across seeds/months.
- Joint end-to-end three-seed results were validation IC `0.05917 ± 0.01400` and December OOS IC `0.06398 ± 0.00785`. OOS NDCG@100 was `0.54507 ± 0.00450`, Precision@100 `0.23921 ± 0.01611`, net active return `-9.67 ± 3.71 bp/day`, and one-way turnover `49.74% ± 5.79%`.
- Half-life/trading diagnostics found no clear IC decay through H10. Rank EMA plus a 5 bp turnover hurdle reduced November/December one-way turnover to 13.90%/8.90%, but net active return remained -13.02/-10.99 bp/day. Staggered H5 cohorts returned -15.48 and +17.12 bp/day in the two months, with no repeated sign.
- Gradient, label-scale, and supervision-position audits are complete. Daily cross-sectional z labels raised validation/OOS IC to 0.11747/0.07446 in the recent fold and 0.13534/0.07755 in the adjacent fold, but recent-fold OOS extreme spread fell from 0.34105% to 0.08885%; the two-fold gate failed. `last` and `tail_weighted` supervision scored validation IC 0.07802 and 0.11289, both below `all` at 0.11747. Decision: `KEEP_ALL`; adjacent-fold and OOS checks remain closed.
- AgentX M0–M2 are complete: typed execution, prediction import/export, multi-seed comparison, walk-forward summary, registry context, and one-shot locked-test approval are implemented.
- M3 v1 stopped after 14/60 months because early order coverage was incomplete. M3 v2 materialized 54 months and 436,800 candidates with 99.88% complete features. HGB's 2025 H2 Rank IC was 0.06994 and monthly IC was positive in all six months, but its 64-combination Top-K matrix returned `NO_TRADEABLE_REGION`: every candidate had negative net active return at 10 bp one-way cost.

M3's stop decision is final. Recent-fold M4/M5 comparisons show positive rank increments for HGB input fusion and joint training, but not cost-adjusted tradability. Half-life, trade gates, available style exposures, gradients, label scale, and supervision position have been checked. Next is a day-task-weight ablation. Raw-book expansion, adjacent-fold seeds 1–2, and the 150M event-stream ablation remain paused.

## Validation queue from the external comparison

The [external L2 comparison](external-l2-research-comparison.md) separates repository facts, external claims, mechanism hypotheses, and decisions. The adopted queue is complete:

1. [x] `EVT-HALFLIFE-001`: `IC(1)`–`IC(10)` and portfolio returns grouped by entry date.
2. [x] `TRD-STAGGERED-H5-001`: daily full rebalance versus five staggered H5 cohorts.
3. [x] `TRD-RANK-EMA-001`: rank EMA, absolute entry threshold, turnover hurdle, and cash.
4. [x] `RISK-ATTR-001`: size, liquidity, volatility; industry awaits local classifications.
5. [x] `EVT-GRAD-AUDIT-001`: shared-backbone task gradient norms and angles.
6. [x] `EVT-LABEL-SCALE-001`: raw return versus daily winsorized cross-sectional z labels.
7. [x] `EVT-SUPERVISION-POSITION-001`: all positions, last position, and tail-weighted supervision.

The first four are detailed in the [signal half-life and trading diagnostics](eventstream-signal-trading-diagnostics.md); the others are in the [gradient audit](eventstream-gradient-audit.md) and [label-scale study](eventstream-label-scale.md). Supervision position did not add value. Next change only day-task weight. Capacity expansion follows target validation, not the other way around.

## Research principles

### Optimize a tradeable portfolio before model-only metrics

The primary target is a long-only Top-K portfolio from a dynamic liquidity universe. Cross-sectional Rank IC, Macro F1, and long-short spread are diagnostics, not substitutes for net portfolio results. Formal reports include NDCG@50/100, Precision@50/100, Top-K excess return and hit rate, turnover, costs, net return and Sharpe, monthly stability, extreme-day contribution, winsorization sensitivity, and industry/size/volatility/liquidity exposures.

### Separate research, validation, and final confirmation

2025 has already been used for cost audits, model comparisons, and hypothesis generation. It is development/rolling-validation evidence, not an unseen final test for new methods. M0 audited 2026 availability, but only 73 aligned sessions were available versus 120 required. The next research series must choose a genuinely unseen 2026 interval or run a prospective paper test beginning with the next complete tradeable interval. Keep `locked_start` as `TBD` until the audit; do not select a convenient fixed date.

### Change one main mechanism at a time

Separate architecture, labels, loss, data windows, and portfolio rules. Each experiment needs a fixed comparator, primary metrics, falsification condition, and stop gate. Register negative results too.

### Freeze features before increasing model complexity

First test new order-book embeddings with HGB or LambdaMART. Only stable frozen-feature gains can unlock joint fine-tuning, neural ranking loss, or multi-day deep models.

## Milestones

| Milestone | Scope | Status |
|---|---|---|
| M0 | Reset research contract | Complete |
| M1 | Top-K evaluation core | Complete |
| M2 | Deterministic AgentX loop | Complete |
| M3 | No-retraining portfolio/cost diagnosis | Complete; no tradeable region |
| M4 | HGB and LambdaMART ranking baselines | In progress; rolling-year gate remains |
| M5 | Event-stream representation and frozen embeddings | In progress; cross-window/net-return gates remain |
| M6 | Neural Top-K loss | Not started; entry criteria apply |
| M7 | Multi-day/multi-horizon model | Not started; entry criteria apply |
| M8 | AgentX research intelligence | Not started |
| M9 | Developer Agent and Harness Evolution | Paused |

Use only `Not started`, `In progress`, `Complete`, or `Stopped`. Move to `In progress` when work starts; mark complete only after linking experiment IDs, evidence, and decision. Do not use a roadmap row as a substitute for a result.

## M0: Reset the research contract — complete (2026-08-08)

Audited local/remote 2026 minute, snapshot, daily, and trading-status availability; defined `research_end`, `validation_end`, and `locked_start`; classified 2025 as seen development evidence; fixed signal time, execution price, return interval, and unavailable-security handling; set the historical dynamic liquidity Top-400 universe with Top-100 smoke path; fixed long-only K=25/50/75/100 and one-way costs of 5/10/15/20 bp; specified initial entry, sell tax, impact, halts, and limit moves; fingerprinted HGB/TCN/raw-200 baselines; and versioned `ResearchProtocol` rather than hard-coding dates.

Acceptance: protocol answers whether each date/file is eligible for research, validation, or confirmation; repeated evaluation is deterministic; no 2025 result is described as unseen. Records: [M0 contract](topk-agentx-m0-research-contract.md), `configs/research-protocol-topk-v1.yaml`, `configs/topk-portfolio-v1.yaml`, and `../baselines/topk-agentx-v1.json`. Locked 2026 data were archived, but only 73 sessions were fully aligned (120 required). Formal daily return changed to next-open-to-following-open so turnover and return periods match.

## M1: Top-K long-only evaluation core — complete (2026-08-08)

The model-neutral evaluator reads symbol, trading/label dates, score, and target return from prediction Parquet; supports fixed K, equal-weight long-only, rank buffers, minimum turnover hurdle/score gap, dynamic membership, missing next-day holdings, daily holdings/trades/turnover/gross/cost/net detail, ranking/monthly/extreme-day/exposure metrics, and synthetic regression tests. Core logic is `src/ticknet/research/portfolio.py`; `scripts/evaluate_cost_adjusted.py` is a thin CLI. `research` does not import `nextday` training code; both consume stable artifact contracts.

Acceptance: CLI/tests/AgentX reuse one evaluator; long-only is distinct from legacy long-short; zero cost means net equals gross; buffer turnover is explainable from daily trades. Completed contract: `can_buy`, `can_sell`, strict missing-holding rules, post-trade weight drift; M2 calls `evaluate_topk_portfolio()` from typed `topk_cost_sweep`. See the [M1 evaluator and usage contract](topk-agentx-m1-portfolio-evaluator.md). The old quantile long-short path remains `legacy_quantile_long_short_diagnostic`.

## M2: Deterministic AgentX loop — complete

ExperimentSpec v2 records objective, executor, inputs, config overrides, primary metrics, success gates, falsification condition, artifact contract, budget, parent, novelty signature, and stage. Arbitrary `entry_point` strings were removed; the Runner dispatches only allowlisted executors. `train_nextday`, `train_minute_tcn`, `export_predictions`, `audit_predictions`, `topk_cost_sweep`, `walk_forward_robustness`, and `compare_experiments` are implemented. `train_ranker` is deferred until M4 chooses its library and command. `data_audit` and `cost_analysis` must invoke deterministic implementations, never silently train a model.

Each experiment/seed has its own directory with resolved spec/config, git/worktree state, dataset fingerprint, stdout/stderr/exit/runtime, checksummed checkpoints/results/predictions, Evaluation, and exceptions. Registry stores recursive validation/test/Top-K/cost metrics, uniqueness/foreign-key constraints, pre-execution records, failures, paths/checksums, paired seed comparisons, and duplicate-run rejection. Evaluation emits `KEEP`, `EXTEND`, or `DISCARD`; structured gates are computable; audit anomalies flow into subsequent context. Locked approval binds spec, checkpoint bundle, predictions, data fingerprint, and one-time approval. Static approval tokens were removed.

**M2a — deterministic loop (2026-08-08).** `ExperimentSpec.from_dict()` rejects unknown fields and legacy arbitrary entries. An allowlisted but unimplemented entry fails explicitly. Exported predictions are rechecked against date protocol and audit, preventing artifacts from bypassing the locked boundary. Registry v2 stores recursive metrics, failures, reviews, paths, sizes, and streaming SHA-256. Full compare, ranker, export, walk-forward, and context feedback were not all complete at this substage.

**M2b — locked approval (2026-08-08).** `approve-locked-test` and `locked-test` are separate steps. Only a completed release experiment with `KEEP`, a fingerprint, and registered `best_checkpoint` can receive approval. A random bearer token is shown once; Registry stores only its SHA-256. Approval binds the spec, every seed's checkpoint bundle, locked predictions, and data fingerprint. Token consumption is atomic before audit; success, failure, and replay have deterministic outcomes. Changed predictions/checkpoints invalidate approval.

**M2c — executors and comparison (2026-08-09).** `export_predictions` materializes only a checksum-matched Registry artifact and rechecks dates/audit. Comparison reports seed distributions, mean baseline delta, paired same-seed deltas, and direction-normalized values where positive means better. Walk-forward requires at least three windows with different data fingerprints and reports direction-aware variation and worst window. Registry-to-context feedback remained for M2d.

**M2d — registry context (2026-08-09).** `ResearchContextBuilder` selects KEEP/EXTEND baselines and summarizes recent experiments, parents, failure reasons, Evaluation decisions, audit anomalies, metrics, fingerprints, and all-history novelty signatures. Identical Registry states yield stable SHA-256. The Orchestrator stores the full snapshot as a `research_context` review consumed by Brainstorm and Critic. Both reject historical duplicates; Critic checks allowed actions, available executors, and compute budget. Duplicate proposals fail before reservation and do not pollute Registry. Context confers no locked-test permission and does not infer dates from fingerprints. M2's deterministic acceptance is complete; `train_ranker` remains unimplemented.

Acceptance scenarios are: cost analysis creates a report without training; training exports predictions, audits, stores nested metrics, and evaluates; unauthorized entry, locked data, duplicate IDs, and artifact conflicts fail deterministically.

## M3: No-retraining Top-K and cost diagnosis — complete; stopped

The initial Top-100 prediction was an engineering smoke test; formal decisions used dynamic Top-400 under the M0 contract. Implementation added Cartesian cost sweeps with same-date validation and deterministic sweet-spot decisions; prediction artifact state/uniqueness/checksum/date/fingerprint binding; formal trading-state and missing-holding requirements; Parquet metadata/candidate-count validation; status rows for out-of-universe holdings; and dynamic Top-400 HGB predictions with T+1 to T+2 open returns, trading state, missing-feature retention, fingerprint, and `return_end_date` leakage checks. Real daily full-range audit and one-day L2 smoke passed.

M3 v2 enforces 54 atomic monthly Parquets, source identity, shard SHA-256, resumable materialization, and complete manifest. Of 436,800 candidates, 436,256 had complete features. Registered prediction: `PRED-HGB-400-OPEN2OPEN-001`. `TRD-TOPK-SMOKE-001` passed a 64-cell engineering smoke; `TRD-TOPK-400-001` completed the formal 64-cell matrix and found no candidate meeting the 10 bp cost gate. Best absolute result (`K=100`, buffer 50) still had -4.75 bp/day net active return versus Top-400. Best active breakeven one-way cost was about 4.33 bp. Therefore dependent `TRD-BUFFER-400-001` was not started. Full limitations and results are in the [M3 diagnostics](topk-agentx-m3-topk-diagnostics.md).

The original entry gate required a reasonable-cost Top-K improvement over no-signal, gains not concentrated in one month/extreme days, and buffer turnover savings not outweighed by gross-return loss. The negative matrix does not block investigating ranking-target mismatch in M4, but it does block expensive order-book pretraining.

## M4: HGB and LambdaMART cross-sectional ranking baselines — in progress

Question: is the bottleneck pointwise prediction versus Top-K objective mismatch? Keep minute aggregate features, dates, stock intersection, and labels identical; retain HGB pointwise; provide per-trading-day LightGBM `LGBMRanker` groups and nonnegative cross-sectional relevance grades; evaluate NDCG@50/100, generate the same prediction schema, and reuse M1 costs. Ranker dependency is pinned in `pyproject.toml` and `uv.lock`. Recent-fold config: `configs/embedding-frozen-recent-2025.yaml`, train August–October 2025, validate November, OOS December.

LambdaMART E0 validation/OOS IC was -0.04334/0.00766; E1 three-seed score mean -0.01117/0.03414; E2 -0.05081/0.01389. E1 net active return was +15.53 bp/day OOS but -15.34 bp/day validation. Two months do not show stable ranking superiority. M4 remains in progress until rolling-year evidence is added using longer event caches. The gate is consistent Top-K ranking or net-return improvement across windows, without changed universe/label/score coverage, at feasible runtime and memory. If LambdaMART does not beat HGB, investigate feature information before implementing a neural ranking loss. Planned IDs: `MDL-RANK-SMOKE-001`, `MDL-RANK-ROLLING-001`.

## M5: Event-stream representation and frozen embeddings — in progress

The first stage used `configs/eventstream-h5-recent-capacity100m.yaml`: August–October 2025 train, November validation, December OOS, with 2026 locked. `capacity100m` has 100,604,180 parameters; input packing/A100 throughput were measured. As of 2026-08-19 there was no local CUDA GPU. `rclone about gdrive:` reported 200 GiB total, 145.292 used, 53.305 free; a five-month pack needed about 313.11 GiB. Training therefore used fixed-window caches of about 25 GiB per seed. Three recent-fold seeds and adjacent-fold seed 0 completed with cache/checkpoint/result fingerprints checked.

Completed sequence: infrastructure/materialization/remote checks/recovery; recent-fold seed 0 and then seeds 1–2 after the gate; shared late-day cache; fixed-checkpoint embeddings into HGB/LambdaMART; paired increments; lightweight joint cache and resumable Colab workflow; joint seeds 0–2 on the identical stock-days; adjacent-fold seed 0; trading/half-life/exposure; gradient/label-scale/supervision ablations. Remaining decision: whether cross-window trading evidence justifies adjacent seeds 1–2 or `probe150m` (currently no).

Embedding protocol takes the final 512 events before close for each stock-day and stores the final valid 960-dimensional state. Manifests bind source data, checkpoint SHA, training-cache fingerprint, code revision, date, symbol, anchor, and schema. Train a downstream model per checkpoint; aggregate metrics or prediction scores, never average vector dimensions across seed spaces. Event packs overlap M3 daily Top-400 candidates by about 368–397 names (about 96%). E0/E1/E2 use identical stock-day intersections; first-round evidence applies only to this subset.

The frozen comparison used 22,409 train, 6,963 validation, and 8,125 OOS samples (96.69% minute-candidate coverage). HGB E2 seed validation ICs were 0.02462, 0.02988, 0.02323 and OOS 0.04333, 0.05644, 0.05912; mean-score validation/OOS IC was 0.02833/0.05701 versus E0 0.01808/0.04010. This passed the small frozen-increment gate, not full M5. `FEAT-EVENTSTREAM-JOINT-001` then reused the same stock-days, dates, labels, and evaluator. Its cache stored features/targets and shard references, not a second event array; source revisions were `da01954b22a1a1506c9e91f8558fcd80bf8184e8` (seed 0) and `92426f67060e7ebb24cb3400ada6aa8af38ae804` (seeds 1–2). All keys and row counts matched; each seed's seven formal files matched Drive; locked 2026 data were excluded. Best epochs were 2/1/1. Validation IC `0.05917 ± 0.01400`; OOS `0.06398 ± 0.00785`; NDCG@100 `0.54507 ± 0.00450`; Precision@100 `0.23921 ± 0.01611`; one-way turnover `49.74% ± 5.79%`; net active return `-9.67 ± 3.71 bp/day`. IC was positive for all seeds, net active return negative for all.

The half-life/trading decision remains `HOLD`: EMA and hurdle reduced turnover, staggered H5 was positive in December and negative in November, and date concentration persisted. Next is day-task weighting. Keep adjacent seeds 1–2 and `probe150m` paused. Full findings: [event-stream trading diagnostics](eventstream-signal-trading-diagnostics.md).

### Labels, sampling, and gates

Pretraining uses A-share data, not FI-2010 weights. Candidate auxiliary targets include midpoint direction/return over the next 10/50/100/500 book events, next 1/5/30-minute return, spread/imbalance changes, and impact recovery. Every auxiliary label must follow its input window; normalization, universe, and split may use training-period knowledge only.

Once permitted by the frozen-increment gate, compare a single late-day embedding, multiple 10:00/11:30/14:00/14:55 anchors, and minute-feature/embedding concatenation. Cache embeddings once with encoder checkpoint, input protocol, date, symbol, and schema provenance. Downstream matrix: E0 minute aggregates, E1 frozen representation, E2 concatenation, E3 multi-anchor; use the same selected ranker.

M5 completion requires E2/E3 to beat E0 across multiple windows, consistent improvement in at least one of Top-K net return, NDCG, and monthly stability, and no symbol/date/future leakage explanation. Only then allow joint fine-tuning. Otherwise retain aggregate features and stop fine-tuning/capacity expansion. The raw-book 2×2 matrix already stopped raw-200/raw-1000 expansion. Planned IDs: `FEAT-EVENTSTREAM-100M-SEED0-001`, `FEAT-EVENTSTREAM-100M-ROBUSTNESS-001`, `FEAT-EMB-FROZEN-001`, `FEAT-EVENTSTREAM-JOINT-001`, `FEAT-EMB-MULTI-ANCHOR-001`, `FEAT-EVENTSTREAM-150M-ABLATION-001`.

## M6: Neural Top-K ranking loss — not started

Entry requires M4 evidence that ranking objectives matter or M5 stable embedding increment. Add `DayBatchSampler` grouped by `label_date` so each step sees same-day cross-sectional relations; implement pairwise logistic ranking and weight pairs near the Top-K boundary. Retain Smooth L1 as a return-scale anchor and demote classification to auxiliary. Add LambdaRank/differentiable NDCG only after the first version.

Compare pointwise loss, pairwise only, pointwise + pairwise, and ordinary versus Top-K-boundary-weighted pairs. Require consistent multi-seed/multi-window gains in Top-K net return and ranking metrics, not training loss alone, with controlled memory/runtime. Candidate form:

```text
L = lambda_point * SmoothL1
  + lambda_class * CrossEntropy
  + lambda_rank * TopKPairwiseLoss
```

## M7: Multi-day embeddings and horizons — not started

Entry requires a demonstrated daily embedding increment and frozen cache contract. Candidate model: 5/10/20-day embedding sequence into a small TCN/GRU, with next-day and forward-five-day heads; add standard daily price, volatility, liquidity, industry, and size factors to test whether embeddings add information or proxy exposures.

Define executable returns precisely. For five-day targets, specify start/end prices and benchmark; do not train a weekly target by merely holding daily scores. Purge labels crossing split boundaries and embargo by label length; account for overlapping targets with block bootstrap or robust errors. Compare same-day embedding→next day, 5/10/20-day sequences→next day or forward five days, daily factors alone, and factors + embeddings. Stop if the five-day target does not repeatedly beat the daily-factor baseline, turnover falls faster than gross return, or results depend on one sequence length/year.

## M8: AgentX research intelligence — not started

Entry requires stable M1/M2 evaluators, executors, artifacts, and Registry. Build context from current baseline; recent KEEP/EXTEND/DISCARD decisions; parent DAG/failures; audit anomalies; available executors/data/budget; and seen-date/locked permissions.

Brainstorm should produce 3–5 candidates per round marked `ready`, `probe_first`, or `backlog`, ranked by objective fit, evidence, feasibility, cost, risk, and novelty. Each states mechanism, observable, comparator, and falsification condition. Critic checks duplication, leakage, budget, executor availability, and computable metrics. LLM judgments advise only; Policy and statistical gates decide. Evaluation classifies each causal link `verified`, `broken`, or `unclear`; metric gains with unclear mechanism are not automatic success. Feed decisions and negative lessons into the next context.

Before real LLMs, replay fixed cases and validate structured outputs. LLMs cannot expand executor/data/locked permissions. Store model and prompt versions, context digest, and raw output. Acceptance: discover prior failures without repeating them, choose an executable candidate without manual config, feed audit/evaluation into next round, and fail safely on malformed output.

## M9: Developer Agent and Harness Evolution — paused

Only open code changes after M8 is stable. Use an isolated worktree, explicit file/module allowlist, expected mechanism/observable, required tests, diff/time/retry budgets, and prohibition on changing splits, locked protocol, or historical artifacts. Require Ruff, `ty`, pytest, smoke, and human diff review; the Developer Agent cannot merge or approve locked tests.

Do not start SGPO with the current insufficient trace set. First collect replayable success, failure, rejection, and platform-failure traces, then add a fixed replay set, paired old/new harness replay, semantic/task/file coverage and safety scores, single-subagent admission rules, and no-op regression behavior with retained failure reasons. Trace count alone is not sufficient; artifacts, metrics, and decisions must replay deterministically.

## Milestone execution record template

```yaml
milestone: M1
status: in_progress
owner: human-and-codex
hypothesis: ""
baseline: ""
change: ""
primary_metrics: []
success_gates: []
falsification_condition: ""
data_protocol_version: ""
experiment_ids: []
artifacts: []
decision: null
next_action: ""
```

Record actual commands, result paths, metrics, exceptions, and KEEP/EXTEND/DISCARD decision. “Training succeeded” or “tests passed” alone is not an experiment report.

## Next actions

M0–M3 are complete. Recent-fold frozen and joint three-seed M4/M5 comparisons are complete. Priorities:

1. Label-scale work is complete; z labels improved both folds' validation/OOS IC, but recent-fold extreme spread regressed. Do not add seeds 1–2 yet.
2. Supervision-position checks are complete; keep `all`, keep OOS closed.
3. Audit daily gradient strength at the z-label best checkpoint; pre-register day-task weight candidates and seed-0 validation gates.
4. If task weighting adds no increment, consider a small cost-aware ranking objective.
5. Add industry attribution when dated classifications are available; first-pass size/liquidity/volatility checks are complete.
6. Revisit `probe150m` only after cross-window net active-return gains are stable and budget/seed-0 gates are specified.

M6/M7 retain their entry gates. Raw-book capacity/window expansion remains stopped.
