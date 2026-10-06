# AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems

> Paper information
> - Authors: AgentX Team (62 authors)
> - Organization: Kuaishou
> - arXiv: [2606.26859v2](https://arxiv.org/abs/2606.26859), cs.AI, 2026-06-26
> - Type: Industrial technical report

---

## Abstract

The paper describes a shift from engineer-dependent, manual recommender-system iteration toward an industrial research loop. A structural execution bottleneck remains: engineers still generate hypotheses, change production code, launch A/B tests, and attribute online results. Innovation therefore scales with headcount and individual productivity rather than compounding with accumulated evidence, compute, and experiment knowledge.

The authors propose AgentX, a production multi-agent system that autonomously generates, implements, evaluates, and learns from recommender-system experiments. It coordinates four connected stages:

- **Brainstorm Agent:** combines historical experiments, system architecture, data analysis, and external research into ranked, actionable proposals.
- **Developing Agent:** turns each proposal into production-ready code through repository-grounded generation and reliability checks.
- **Evaluation Agent:** safely launches experiments, applies guardrail-veto A/B decisions, and records both successful and failed outcomes as structured knowledge.
- **Harness Evolution (SGPO):** converts execution traces into semantic gradient updates that improve the agents, moving the system from automation toward self-improvement.

In a three-week deployment in Kuaishou's main-feed and local-services recommendations, three AgentX workers turned 374 ideas into 10 launchable results. Worker throughput doubled each week through self-evolution. The paper reports eightfold concurrency, 3.7 times the business value of human engineers, a 0.561% increase in user time spent, and more than RMB 100 million in annualized revenue.

The same loop was extended to model research for autonomous paper reproduction, module ablation, and cross-paper architecture combinations.

---

## I. Introduction and motivation

The paper argues that recommender-system iteration is limited less by model capability than by a human-dependent idea-to-launch loop. Each hypothesis, production-code change, A/B test, and attribution task requires manual work. Innovation speed is bounded by engineer headcount multiplied by individual productivity instead of compounding as evidence accumulates.

AgentX aims to replace this serial human process with a parallel, compounding, evolving loop. The paper evaluates three research questions:

- RQ1: Does AgentX shorten the idea-to-rollout cycle?
- RQ2: Does each worker produce more rollout-level results?
- RQ3: Do AgentX launches produce measurable online improvements?

## II. Related work

The paper distinguishes two lines of work:

- LLM-driven AutoML uses language models for hyperparameter or architecture search, but is usually offline, limited to one task, and driven by a fixed reward.
- Agentic automation for recommender systems applies agents to recommendation workflows, but often does not close the loop in production.

AgentX differs in three ways. Its reward comes from delayed online A/B feedback, which offline metrics cannot replace. An execution trace alone is not enough to judge success; guardrails and human review are required. The optimization objective itself changes with business needs. A production agent loop must therefore run continuously rather than as a one-off job.

## III. Multi-agent design framework

Industrial recommendation agents differ from agents for mathematics, code, or sandboxed ML benchmarks in three important ways:

1. The reward is delayed online A/B feedback, not an offline metric.
2. A deployment must pass guardrails and human review before a score is available.
3. The objective changes with business goals, traffic composition, and platform constraints.

The loop is divided into four stages. The first three carry intent through online validation. The fourth uses execution traces to improve the loop itself.

| Stage | Responsibility |
|---|---|
| Brainstorm | Turn an ambiguous request into a small ranked set of actionable proposals through bounded exploration and evidence-weighted generation |
| Developing | Use repository-grounded generation and a verification-oriented implementation loop to turn a proposal into production code |
| Evaluation | Manage launch and traffic allocation, apply guardrail-veto A/B decisions, and preserve online outcomes |
| Harness Evolution | Update subagent instructions from accumulated traces (SGPO), admitting changes only through paired replay |

A shared Data Layer persists artifacts. The Knowledge Base stores practice and experience, Agent Data Management stores experiment reports, and the Monitoring Platform continuously observes system health.

## IV. Brainstorm Agent

Brainstorm determines what downstream agents can implement, evaluate, and launch. It narrows an ambiguous optimization request to a small ranked set of actionable experiments instead of producing a long list of plausible-sounding ideas. Each proposal needs production evidence, must stay within the permitted change area, and must be specific enough to implement, launch, and diagnose.

### IV.1 The ambiguity problem

The input is intentionally underspecified, while the output must be operationally precise. Unbounded exploration can invent unavailable signals or features and cross scope boundaries. Excessive conservatism returns only familiar local variations. The design uses explicit boundaries plus evidence aggregation. It normalizes a request into an intake boundary containing the primary objective, allowed business scope, prohibited changes, known guardrails, requested outputs, and unresolved questions. This makes uncertainty visible.

### IV.2 Bounded proposal exploration

Ideas are generated in batches rather than as one free-form answer. Each candidate has one of three maturity states:

- **Ready to implement:** has a concrete objective, a named place in the pipeline, a plausible target path, and enough evidence for review.
- **Probe first:** promising, but needs a specified data check, source, or pilot.
- **Moonshot backlog:** depends on future infrastructure, data, or model capability.

The batch loop proceeds by residual search. After each round, rejected directions, historical duplicates, constraint violations, and covered mechanisms enter an avoid set. The next round searches the remaining space rather than repeating the same ideas.

### IV.3 Evidence-weighted proposal generation

Candidates draw on different evidence sources, combined using weights `α_k(q,c)`:

| Source | Use | When to give it greater weight |
|---|---|---|
| Experiment KB | Historical launch reviews, failure lessons, and business definitions | The candidate depends on novelty, prior experience, or business semantics |
| System KB | Architecture, features, pipelines, DSLs, and code scope | Feasibility depends on the implementation path |
| Data Analysis | Metric definitions, SQL, and offline statistics | The candidate depends on data shape or segment behavior |
| Model Research | Structured facts about mechanisms from papers | The candidate depends on external novelty or recent academic evidence |

Candidate score:

```text
S(c|q) = λo·O + λb·B + λf·F + λh·H + λe·E(c|q) − λr·R(c)
```

`O` is objective alignment, `B` business-semantic validity, `F` implementation feasibility, `H` handoff completeness, `E` weighted evidence, and `R` risk (duplicate direction, unresolved core signal, excessive scope, or unsafe trade-off).

The Experiment KB stores historical launch reviews, business definitions, conclusions, and lessons. It records context as well as success or failure: target scenario, affected users, metric changes, launch decisions, and follow-up diagnosis. This helps prevent Brainstorm from rediscovering known failures or misusing business concepts.

The System KB is a structured domain wiki with schema, wiki, and raw-source layers, maintained through an ingest-query-lint lifecycle. Agents retrieve by module, feature, pipeline, and implementation boundary instead of relying on broad keyword search.

Data Analysis provides empirical support for candidates that depend on data patterns. It guards against overfitting anecdotes or proposing changes based on superficial metric movements.

Model Research turns papers into actionable proposal knowledge. Each paper is decomposed into typed claims (question, hypothesis, method, findings, and limitations, each with evidence strength), architecture components, and cross-paper relations (`extend`, `contradict`, `parallel`, and `apply`). Production baselines and feature contracts use the same schema. Hard training constraints, such as streaming online updates, no epochs, and prohibitions on freezing and early stopping, constrain the search space at its source.

### IV.4 Validation and implementation handoff

Generated proposals pass admission checks for primary-objective alignment, business semantics, user constraints, model-score semantics, implementation feasibility, historical overlap, A/B parameter feasibility, and maturity consistency. Only validated ready proposals reach a human approval gate. The review is focused (`approve`, `revise`, `defer`, or `reject`) rather than patching every weak candidate.

Brainstorm does not write production code. Each approved idea produces exactly one formal experiment record, one source manifest, and one handoff plan covering intended behavior, required signals, expected metric direction, guardrails, and known implementation boundaries. Brainstorm owns ambiguity resolution, evidence weighting, and proposal admission. Developing owns code and repository-level verification.

## V. Developing Agent

Developing turns an approved proposal into a verifiable code artifact through two parallel tracks:

- **Online policy track:** changes production code so it can serve real traffic safely without silent failure.
- **Offline model track:** runs training experiments and produces conclusions reliable enough to accumulate in the knowledge base.

The paper requires four conditions for a trustworthy conclusion: the implementation matches the claimed causal mechanism and expected observables; an isolated expert panel reaches a supermajority; deterministic code extracts metrics from raw logs rather than an LLM interpreting them; and any AUC gain has a validated causal attribution chain. A gain without attribution is a stop signal.

### V.1 Production-code reliability

The coding contract is stricter than syntactic correctness. Code must preserve proposal intent, stay within approved boundaries, use verified repository primitives, pass local and integration checks, and leave a reviewable change. The main repository-specific failure modes are:

- **Attribute hallucination:** inventing fields in user, context, or item feature schemas (high severity).
- **DSL misuse:** guessing a ranking-DSL operator or parameter contract incorrectly.
- **Harness-pattern violations:** changing the wrong queue, omitting registration, or bypassing a required safety mode.

### V.2 Repository-grounded code generation

Two grounding sources are used:

1. A project-specific knowledge base records change patterns, registration conventions, feature-switch rules, and accepted patch examples.
2. A case toolbox contains deterministic tools and checkers that agents must use to verify facts.

The key rule is to query feature attributes before using them. A schema-query tool returns available fields. Agents must call it before reading an attribute so field names are verified facts rather than model guesses.

### V.3 Verification-oriented implementation loop

The staged loop abstracts a proposal into an implementation plan, decomposes it into atomic requirements, assembles a patch, and runs deterministic checks. Failure feedback gives targeted repair instructions instead of regenerating the whole change. It has two verification layers:

- **Accuracy loop:** compare the implementation with the plan. Additional repair rounds count as a quality cost.
- **Dry-run pipeline:** compile and run integration checks. A clean trace passes Dryrun once.

### V.4 Quality score

The paper defines an eight-dimension weighted reliability score:

```text
Q_code = 0.06s1 + 0.12s2 + 0.22s3 + 0.08s4 + 0.06s5 + 0.18s6 + 0.18s7 + 0.10s8
```

| Dimension | Meaning | Weight |
|---|---|---:|
| N1 | C++ syntax-sugar violations | 6% |
| N2 | Harness-pattern violations | 12% |
| N3 | Attribute hallucination | 22% |
| N4 | DSL-check repair | 8% |
| N5 | C++ syntax-check repair | 6% |
| N6 | Correctness-loop iterations | 18% |
| N7 | Human intervention (hard gate) | 18% |
| N8 | Dryrun pass | 10% |

The three severity-S dimensions (attribute hallucination, correctness-loop overhead, and human intervention) account for 58% of the weight and most directly threaten autonomous production reliability. N7 is a hard binary gate: any human intervention scores zero.

### V.5 Model-development track

The workflow is `policy → (code ↔ verify) ∥ experts×N → exec → final_review`, with at most three rewrites.

- **Policy:** declares not just what to change, but the causal mechanism and expected observables (`tf.print`, `tf.summary`) that distinguish healthy from unhealthy behavior.
- **Verify:** checks that the git diff matches the policy's semantic direction and that each declared observable appears in the diff.
- **Expert agents:** run in physical isolation with private knowledge bases. A Python vote counter requires a `⌈2N/3⌉` supermajority; LLMs do not vote.
- **Exec:** a pure-Python state machine extracts metrics from training logs using regular expressions, without an LLM.

**Falsifiable attribution:** final review adjudicates each link in the claimed causal chain as `verified`, `broken`, or `unclear`, then Python folds the results deterministically. All verified links yield `CLEAR`. Any broken or unclear link yields `UNCLEAR`. An AUC gain with `UNCLEAR` attribution is a stop signal.

The paper illustrates this with RankMixer. Round 1 reproduced multiplicative gating `x·tanh(Vo·x)` and gained `+0.0003` AUC, but the claimed gate activation stayed near zero. Glorot initialization made `Vo·x` close to zero, so `tanh` was close to zero; the gate output vanished and blocked gradients. Without causal-chain checks this would be recorded as a success. With them, attribution is `UNCLEAR` and the result is not recorded as a success. Round 2 added a residual, `x·(1+tanh(Vo·x))`, and gained `+0.0022` AUC with every causal link verified.

For platform-failure robustness, a pure-function classifier reads log headers, tails, and FATAL lines, then maps them to reason codes. Deterministic errors such as NaN gradients or feature-table conflicts are abandoned immediately. Transient failures such as stalled logs or infrastructure interruption are retried once. LLM gateway failures rotate gateways. A watchdog daemon applies the same policy to all active runs.

## VI. Evaluation Agent

Evaluation decides whether code should be `KEEP`, `EXTEND`, or `DISCARD`, or be returned as a negative lesson. It converts noisy, delayed, partially observable traffic into a trustworthy reward signal.

### VI.1 Real-world rewards

Offline proxies and introspection are insufficient for recommendation systems. A policy may improve an internal score while harming long-term user experience, so online A/B feedback is treated as the authoritative reward for system iteration.

### VI.2 Safe deployment and traffic allocation

- Map each experiment to the correct business domain, world, and split factor, then allocate mutually exclusive traffic buckets.
- Bind account-level experiments by UID, device-side experiments by device ID, and use UID-first binding for mixed populations.
- When assignment is uncertain, run a pre-experiment balance check and select traffic groups with the smallest baseline differences.
- Route parameter changes through an engineering allowlist. Use a canary path for configuration rollout, observe the minimum gray-release window, and only then expand to full traffic.

### VI.3 Guardrail-veto A/B decisions

Three guardrail principles aim to control false negatives:

1. Use business-specific core metrics and veto thresholds for each business domain.
2. Use a composite economic trade-off score that weights multiple objectives. A negative value in one guardrail metric does not automatically veto.
3. Treat thresholds as attention signals rather than direct blockers. Triggered thresholds escalate to human review; hard blocks are reserved for severe deterioration.

The output is structured: `KEEP`, `EXTEND`, or `DISCARD`, with main effects, guardrail status, statistical method, observation window, and caveats.

### VI.4 Negative-result assetization

Most production experiments do not succeed. Record each failure's cause, such as missing significance, a worsened guardrail, traffic mismatch, implementation details, or business-context mismatch. Index it by pipeline stage, business objective, affected users, and content segment so Brainstorm can retrieve it before proposing new work.

## VII. Harness Evolution (SGPO)

Production experiment analysis can explain whether a recommendation policy worked, but not why an upstream agent failed. SGPO turns execution traces into subagent-prompt updates and admits them only through paired replay.

Under the constrained harness, the production-safe version updates one subagent's harness specification at a time (instructions, verification rules, output contract, and tool-use discipline) while holding others fixed. This makes each update reviewable and gives old-versus-new replay a meaningful comparison.

### VII.1 SGPO-I: conversation traces

- Sample traces from a trace pool and extract rubrics from the initial query.
- An evaluation agent writes a natural-language loss report, producing semantic gradient `g` that diagnoses missing constraints, weak step ordering, underspecified evidence, and incomplete downstream contracts.
- A refinement agent turns the gradient into a local harness edit to produce candidate `h'`.
- Run paired replay on the same tasks for old and new harnesses. Accept only if `ΔJ` exceeds a threshold and safety checks pass. Otherwise make no change and archive the failure mode.

In the paper's example, the brainstorm subagent evolved over five rounds, raising its replay score from 75.15% to 98.00%. The key accepted edits were concrete contract changes: make the task contract explicit before generating ideas and require each proposal to expose its business causal chain. Generic prompt polishing did not achieve this.

SGPO is not required to improve monotonically. Scores may improve steadily, saturate early, regress temporarily, or recover from noise. Paired replay is intended to expose regressions before a change enters the production harness.

### VII.2 SGPO-II: code replay

Evidence changes from conversation traces to historical merge-request replays. After filtering, the pool retains human-approved code changes that remain active in the current repository. The five-dimension score weights semantic correctness 40%, requirement coverage 25%, file coverage 20%, default safety 10%, and style consistency 5%. Acceptance requires weighted score at least 4.0 and semantic correctness at least 4.

In the paper's complex asynchronous-module example, the weighted score rose from 2.60 to 4.90 (88%) as the harness accumulated constraints from project-specific failures. A regression example fell from 3.67 to 1.80; paired-replay admission prevented the regression from entering the accepted harness.

### VII.3 Evolution of the model-research exploration pipeline

The paper describes a three-phase exhaustive loop:

- **Phase 1:** create one reproduction proposal per paper, rank by `ΔAUC` against the production baseline, and send the top 16 to Phase 2.
- **Phase 2:** isolate each ablatable module and compare the original with an LLM-inferred alternative using strict isolation. A module is considered effective if at least 24 of 32 experiments have positive `ΔAUC`.
- **Phase 3:** combine findings across papers. Regular rounds graft the highest-`Δ` confirmed module onto top-K lineages. Challenger rounds, every four rounds, apply effective modules to papers ranked 17–32 to preserve diversity.

Memory-guided pruning works at several levels. Paper-level pruning removes all modules from a paper when its central premise is falsified. Module-level pruning removes a module from combinations when ablation shows no independent contribution. Saturation detection stops when duplicate novelty signatures exceed 80% for two consecutive rounds.

The experience flywheel appends every training round to an append-only event log; the knowledge base is a rebuildable view. Lessons are organized as `anti_patterns` (failure mode plus log and diff patterns) and `playbook` (successful recipes, recorded when `ΔAUC > 0.001`). Each candidate lesson passes two gates: at least two independent runs and an adversarial review by an agent whose sole job is to falsify the claimed causal mechanism. Surviving lessons become `confirmed`; disputed ones become `contested` and are injected with a warning; falsified lessons are permanently excluded.

> Comparison with AgenticRecTune's Skillhub: distilling lessons directly from A/B outcomes does not explain why a change worked. An unexplained lesson is a liability because it may generalize poorly or encode a spurious correlation. Requiring a verified causal explanation before promotion extends falsifiable attribution from one experiment to the knowledge base.

## VIII. Experiments

### VIII.1 Production measurements and loop performance

The three-week deployment used three workers in two settings: main feed and local services.

| Setting | Ideas | Idea pass | Code and launch | Positive evaluation | LR |
|---|---:|---:|---:|---:|---:|
| Main feed | 361 | 27.7% | 95.0% | 8.4% | 8 |
| Local services | 13 | 46.1% | 83.3% | 40.0% | 2 |
| Overall | 374 | 28.34% | 94.3% | 9.9% | 10 |

The funnel had 374 ideas, 106 passes (28.3%), 100 launches (94.3%), and 10 positive evaluations (9.9%).

Rejection reasons among 268 rejected ideas were highly asymmetric. Platform and infrastructure constraints accounted for about 91.4%: parameter-resource conflicts 64.7% (the target parameter was already occupied by a holdout configuration, an in-flight experiment, or another traffic world), prerequisite enable flags off 6.4%, hard-constraint or allowlist violations 7.5%, and missing user or item attributes 14.5%. Actual agent errors accounted for 8.7%. The bottleneck was operational, not algorithmic. Fixing category-A gaps could recover about two-thirds of lost ideas.

More than 95% of coding failures were also infrastructure-related. DSL or force-enable wiring errors accounted for 35%, C++ or MaTX compilation constraints 20%, if/else structure 15%, and genuine algorithm errors less than 5%.

**RQ1, shorter cycle:** The serial chain became a parallel pipeline. Three workers ran 374 ideas through the loop in three weeks. Concurrent experiments were 12 per worker versus 1.5 per engineer, an eightfold increase. Throughput was close to linear in worker count.

**RQ2, more rollout results:** AgentX averaged 3.3 LRs per worker. Per-idea hit rate was lower than for senior engineers (2.7% versus 5.1%, or 0.53×), because human ideas were hand-filtered. At worker-week granularity, however, cumulative app-time gain was 0.0623% versus 0.0167%, or 3.7×. The system trades scarce-resource precision for automation at scale.

| Per worker-week metric | AgentX | Engineer | Ratio |
|---|---:|---:|---:|
| Concurrent experiments | 12 | 1.5 | 8× |
| LR count | 1.1 | 0.08 | 13.8× |
| Cumulative app-time gain | 0.0623% | 0.0167% | 3.7× |
| Launch rate per idea | 2.7% | 5.1% | 0.53× |

**RQ3, measurable online gains:** Ten LRs increased cumulative main-feed user time spent by 0.561%. Local-services recommendations produced more than RMB 100 million in annualized revenue. During the three-week self-evolution period, concurrent experiments rose from 15 to 60 (4×), idea pass rate from 15% to 45% (3×), and weekly LRs from 2 to 5 (2.5×).

### VIII.2 Model-iteration funnel (independent research)

Of 113 candidate papers, 214 experiments were dispatched automatically and 180 completed (84.1%). Of completed runs, 104 beat baselines on public datasets (57.8%). Twenty-two entered online A/B (21.2% of the 104), five evaluated positively (22.7% of 22), and two became LRs (40% of five).

Three observations:

1. **Expand, then prune.** First explore 214 combinations. Then filter by offline completion, offline performance, and business-launch criteria, reducing online A/B load by nearly an order of magnitude.
2. **Interpret the 57.8% against a simplified baseline.** Removing business features and serving constraints does not show an improvement over the full production system. It identifies a potentially integrable module or architecture.
3. **The 22-to-5-to-2 funnel is the tightest stage.** Even when offline gain exists, the joint online-positive and launch-review requirement shrinks candidates by about an order of magnitude. The online review gate is the main limit on model-side LR volume.

The two LR models increased live-stream viewing time by 0.865%.

### VIII.3 Showcase I: end-to-end experimentation (PCV case)

Two loop iterations show feedback-driven redesign:

- **Loop 1, direct PCV boosting:** Brainstorm selected PCV boosting from five candidates. Developing implemented `S1 = Br·(1+βP)`. Online effects were weak and unstable: person-level time +0.034%, user time +0.021%, active devices -0.023%, and users aged 18–30 -0.032%. Evaluation found that increasing all high-PCV content without a quality gate admitted low-quality bait content.
- **Loop 2, constrained PCV ranking:** `S2 = Bd·(1+β(u)·G(P))`, where `G(P)=max(P−τ,0)` is a quality gate, `β(u)` is an activity-aware dynamic weight, and `Bd` is a duration-oriented base score. Online user time increased 0.071%, real-show increased 0.118%, and experience guardrails remained stable. The two loops produced reusable lessons: direct PCV boosting is noisy, while adding a quality gate, activity awareness, and a duration anchor made it more reliable.

### VIII.4 Showcase II: co-evolution with expert agents

In local services, a recommendation-decision expert agent ran user-level diagnosis and returned natural-language control suggestions. AgentX Brainstorm assessed feasibility and selected a CPM-boost action. Developing implemented a UV-level CPM boost, and Evaluation measured a 4.7% revenue increase.

The expert agent supplied domain diagnosis, such as inferring car-ad demand from a driver's occupation and vehicle searches. AgentX converted it into deployable, fine-grained atomic actions. The two systems co-evolved in both directions.

## IX. Conclusion

- AgentX changes recommender-system iteration from linear manual effort into a compounding, engineerable lever.
- The three-week production study reported roughly order-of-magnitude gains in throughput and cumulative online impact.
- Engineers shift from executing each experiment to deciding which questions are worth exploring and designing more effective agent systems.
- Trace data connects the two: every underlying system improvement can be inherited by later experiments.

## Relevance to this repository

At the time these notes were prepared, `deep-learning-tick-data-prediction` had several foundations related to AgentX, but also important differences.

### Existing foundations

- Strict out-of-time splits, a locked test set, and multiple seeds provide authoritative offline reward signals.
- `AGENTS.md` requires code agents to run Ruff, `ty`, pytest, and a smoke check after changes, analogous to a Developing Agent harness.
- Shared Rank IC, after-cost evaluation, dataset fingerprints, and experiment configuration records support structured evaluation.
- The [reproduction audit](../reproduction-audit.md) distinguishes engineering facts, paper facts, and real training results.

### Important differences and cautions

1. **Feedback signal:** AgentX continually produces new online A/B data; this repository uses fixed historical data. Repeated access to the same 2024 validation set could efficiently train an agent to overfit it. Use rolling OOS evaluation (2021–2023 inner CV, 2024 research holdout, 2025 locked test) and physical data isolation.
2. **Enforce permissions in code, not prompts.** Agents should never have direct locked-test access. Human approval should permit a one-time evaluation.
3. **Prefer determinism before LLM judgment.** Use deterministic Python for metric extraction, statistical tests, and A/B decisions. LLMs should propose hypotheses and explain results.
4. **Require a falsification condition.** Every proposal should declare `falsification_condition` so the process remains scientific rather than degrading into AutoML.
5. **Assetize negative results.** Failed experiments, such as TCN failing to beat HGB or gains disappearing after costs, are knowledge assets. Record them to prevent repetition.

### Recommended implementation order at the time

1. Prediction artifact and audit: save validation prediction details and explain the discrepancy between near-zero IC and high spread.
2. ExperimentSpec, runner, and registry: unify experiment entry points and SQLite experiment memory.
3. Research protocol and physical locked-test isolation.
4. Config-only Brainstorm and Critic agents.
5. Developer Agent with code-change access, worktrees, tests, and diff review.
6. Consider SGPO and Harness Evolution only after these foundations.
