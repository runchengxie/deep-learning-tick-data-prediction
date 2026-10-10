# Documentation Index

> status: active
> owner: quant-deep-learning
> audience: human and agent
> last_verified: 2026-09-06
> source_of_truth: yes
> superseded_by: n/a

This directory contains project status, topic guides, research records, and frozen artifacts. Start with the status page, then follow the topics relevant to your work. This distinguishes current conclusions from milestone records and historical snapshots.

## Suggested reading path

New readers can start with the root README, then read:

1. [Project status](project-status.md) for current capabilities, research conclusions, data access, and work order.
2. [Model catalog](model-catalog.md) for model methods, strengths, limits, and research status.
3. [Cross-sectional prediction](nextday/cross-sectional-prediction.md) for samples, labels, date splits, and evaluation rules.
4. [AgentX research contract](research/topk-agentx-m0-research-contract.md) for the current 2025 development and 2026 locked-period boundary.
5. [AgentX research roadmap](research/topk-agentx-research-roadmap.md) for the research design and M0–M9 status.
6. [External L2 project comparison](research/external-l2-research-comparison.md) for comparisons, factual checks, and experiments adopted from related projects.
7. [Top-K diagnostics](research/topk-agentx-m3-topk-diagnostics.md) for the completed formal Top-K diagnostics.
8. [Event-stream trading diagnostics](research/eventstream-signal-trading-diagnostics.md) for signal decay, staggered holding, and trading gates.
9. [Event-stream gradient audit](research/eventstream-gradient-audit.md) for the multi-task gradient audit, decision gate, and run instructions.
10. [Event-stream label-scale study](research/eventstream-label-scale.md) for label-scale and supervision-position experiments.
11. [Operations development guide](operations/development-guide.md) for module boundaries, test scope, and quality checks.
12. [System workflows](operations/systemd-workflows.md) for historical TickNet system workflows and their cleanup.

Use `project-status.md` and the main roadmap for current state. M0, M1, M2a–M2d documents preserve the design and findings at each milestone; later work may have extended them. `reports/` and `baselines/` contain frozen artifacts.

## Concepts and exploration

[Orders, books and flow](concepts/orders-books-and-flow.md) defines market-data events, MBP/MBO, order-flow measurements and venue-specific reconstruction boundaries. The [interactive explorer](https://runchengxie.github.io/quant-deep-learning/concepts/order-book/) illustrates them with synthetic charts and event replay.

## By topic

| Document | Contents |
|---|---|
| [Project status](project-status.md) | Current capabilities, findings, data access, and next steps |
| [Model catalog](model-catalog.md) | Model inputs, methods, strengths, limitations, and research status |

### Next-day research

Next-day cross-sectional prediction, including sample rules, data processing, training entry points, and staged roadmaps.

| Document | Contents |
|---|---|
| [Cross-sectional prediction](nextday/cross-sectional-prediction.md) | Main protocol for data, adapters, splits, training, and evaluation |
| [Raw-200 end-to-end pipeline](nextday/raw-200-end-to-end-pipeline.md) | Raw-order-book workflow and current candidate |
| [Event-stream guide](nextday/eventstream.md) | Lossless L2 packing, causal Transformer, and prediction export |
| [Raw-data expansion roadmap](nextday/raw-data-expansion-roadmap.md) | Five-year raw-data generation, auditing, and expansion |
| [Multi-horizon expansion roadmap](nextday/multi-horizon-data-expansion-roadmap.md) | 1/3/5-day labels, capacity gates, raw-1000, and full-day tick data |
| [100M raw-1000 benchmark](nextday/nextday-100m-raw1000-benchmark.md) | 100M-parameter capacity benchmark and A100 batch sweep |
| [H5 rolling event-stream roadmap](nextday/h5-rolling-eventstream-roadmap.md) | H5 Rank IC target, 3/1/1 rolling protocol, and full-day event-data pilot |
| [Hardware and experiment roadmap](nextday/hardware-constraints-and-experiment-roadmap.md) | Hardware limits, research conventions, and staged roadmap |

### Research records

Top-K tradable portfolios and the AgentX automated quantitative research loop.

| Document | Contents |
|---|---|
| [Chip age HGB](research/chip-age-hgb.md) | Lagged cost/age summaries, published daily contract, paired HGB and minute-feature experiments; no real result yet |
| [AgentX research roadmap](research/topk-agentx-research-roadmap.md) | Roadmap, evidence, shared principles, and M0–M9 status |
| [M0 research contract](research/topk-agentx-m0-research-contract.md) | Research contract, data-access audit, and trading conventions |
| [M1 portfolio evaluator](research/topk-agentx-m1-portfolio-evaluator.md) | Input contract and cost formulas for the Top-K long-only evaluator |
| [M2a deterministic loop](research/topk-agentx-m2a-deterministic-loop.md) | ExperimentSpec v2 and deterministic loop at M2a completion |
| [M2b locked-test approval](research/topk-agentx-m2b-locked-approval.md) | One-time manual approval and controlled use of locked tests |
| [M2c executor comparison](research/topk-agentx-m2c-executors-comparison.md) | Prediction export, multi-seed comparison, and walk-forward robustness |
| [M2d registry context](research/topk-agentx-m2d-registry-context.md) | Reproducible ResearchContext built from the registry |
| [M3 Top-K diagnostics](research/topk-agentx-m3-topk-diagnostics.md) | Inputs, results, and gates for formal Top-K cost diagnostics |
| [Event-stream trading diagnostics](research/eventstream-signal-trading-diagnostics.md) | Signal decay, H5 staggered holding, smoothing, dynamic costs, and risk exposures |
| [Event-stream gradient audit](research/eventstream-gradient-audit.md) | Gradient strength, direction, audit contract, and gates for four event-stream tasks |
| [Event-stream label scale](research/eventstream-label-scale.md) | Label scale, supervision-position experiments, and formal decision |
| [Resource strategy and pilot gates](research/resource-strategy-and-pilot-gates.md) | Compute limits and gate-based experiment principles |
| [External L2 comparison](research/external-l2-research-comparison.md) | Related L2 research, factual corrections, mechanisms to test, and work order |
| [Shanghai opening contract audit](research/shanghai-opening-contract-audit-2026-08-27.md) | Shanghai opening-order coverage, event timing, and ten-level ledger audit |
| [Opening coverage inventory](research/opening-coverage-inventory-2026-08-27.md) | Pre-open raw L2 coverage and related-file completeness |
| [Historical data eligibility](research/historical-data-eligibility-2026-08-27.md) | Admission boundaries for historical raw L2 data from 2021 to 2025 |
| [Experiment log](research/experiment-log.md) | Dated results for TCN comparisons, rolling validation, costs, audits, and AgentX |

### Development and operations

| Document | Contents |
|---|---|
| [Development guide](dev/development-guide.md) | Module boundaries, test scope, quality gates, and dependency management |
| [Colab CLI automation](dev/colab-cli-automation.md) | Unattended training and evaluation entry points for Colab |
| [System workflows](operations/systemd-workflows.md) | Historical local TickNet workflows, paths, and cleanup records |

### Reproduction and artifacts

| Path | Contents |
|---|---|
| [Reproduction audit](reproduction-audit.md) | DeepLOB on FI-2010; maintained separately under `src/ticknet/fi2010/` |
| [Top-K AgentX v1 baseline](baselines/topk-agentx-v1.json) | Frozen M0 artifact manifest with SHA-256 and metrics |
| [Multi-horizon source inspection](reports/multi-horizon-decision-2026-08-10/source-inspection.md) | Historical snapshot from 2026-08-10; its findings were incorporated into the multi-horizon roadmap |

### Papers and reading notes

Paper sources and reading notes organized by section are in [references](references/README.md). They provide research provenance. Current implementations and experiment status remain in the status page, topic guides, and roadmaps.

## Where to record research evidence

Documents serve different reading needs. Keep numerical conclusions consistent by following these conventions:

- `project-status.md` contains only the current summary and status.
- The current-evidence section of [the AgentX roadmap](research/topk-agentx-research-roadmap.md) contains the latest real results.
- [The multi-horizon roadmap](nextday/multi-horizon-data-expansion-roadmap.md) contains multi-horizon and capacity experiments.
- [The experiment log](research/experiment-log.md) records dated historical results with full figures.

Date capacity, disk use, and Drive quota measurements because they change. Keep historical artifacts in `reports/` and `baselines/` frozen instead of rewriting them to match current status.

## Writing conventions

Write explanatory prose in clear English. Keep commands, configuration keys, module names, and metrics in inline code. Verify command options with `--help` before documenting them.
