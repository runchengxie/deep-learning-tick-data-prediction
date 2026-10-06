# Quant Deep Learning: Independent Research and Publication Site

Date: 2026-10-06
Status: Approved for implementation
Audience: quantitative researchers and general technical readers
Related repositories: `quant-market-data-platform`, `quant-backtest-runtime`, `quant-platform`, `quant-research`, and `quant-market-research`

## 1. Intent and Approved Direction

The user wants Quant Deep Learning to operate as a relatively independent model-research project, with a clear upstream data provider and downstream backtest consumer. They also want its public Pages site to explain research and results in plain language, use substantial visual evidence, and serve both technically curious readers and quantitative researchers.

The user approved this direction and later authorized the implementation work. The user selected both general technical readers and quantitative researchers as audiences.

## 2. Current State

- The repository already has a static MkDocs Material site and a GitHub Pages workflow.
- Maintained Markdown is English, but the current homepage mainly directs readers to long technical and research documents rather than presenting a visual research overview.
- `MIGRATION-STATUS.md` is at the root and still assigns generic portfolio, risk, backtest, replay, matching, and execution simulation to `quant-platform`, and private experiment governance and promotion evidence to `quant-research`.
- The current data-boundary guide already defines `quant-market-data-platform` as the owner of published market data.
- The project contains its own `ticknet.research` code for model-specific research workflows, alongside the separate private `quant-research` repository. Their overlapping concepts need an explicit capability inventory before any code is moved or removed.
- `quant-backtest-runtime` owns the backtest job protocol, state, workers, resource controls, result validation, and publishing tools. Its repository uses reusable backtest algorithms and execution simulation from `quant-platform`.
- `quant-market-research` demonstrates an Astro site with React/ECharts charts, reviewed static research data, topic pages, summaries, and explicit evidence limitations. Its pages place a plain-language finding beside sample scope, figures, interpretation, and remaining checks.
- Workspace inventory found three Astro projects under `/home/richard/code`: `quant-market-research`, `quant-intel-platform/web`, and `quant-intel-pages`. They use MkDocs differently: `quant-market-research` renders its public document routes through Astro while CI also builds a strict, allowlisted MkDocs reference site outside the deployed artifact; `quant-intel-platform` deploys MkDocs output under `/docs/` beside its Astro site; `quant-intel-pages` has no MkDocs build. Astro therefore does not make MkDocs automatically redundant; the deciding question is whether MkDocs still owns a distinct deployed documentation product or a required validation step.

## 3. Repository and Dependency Boundaries

### 3.1 Intended direction

```text
quant-market-data-platform
  owns ingestion, canonical data, versions, quality receipts, and published assets
              │
              │ published data contracts and assets
              ▼
quant-deep-learning
  owns model-specific representations, labels, models, training, inference,
  model diagnostics, reproducible study records, and prediction artifacts
              │
              │ versioned prediction/signal artifacts and model manifest
              ▼
quant-backtest-runtime
  owns backtest job orchestration, state, workers, resource control,
  result validation, and publication of completed runs
              │
              └── may call quant-platform's reusable backtest/execution mechanisms
```

### 3.2 Independence requirements

- A clean environment must be able to install Quant Deep Learning and run its public synthetic-data checks without installing `quant-platform`, `quant-research`, `quant-market-data-platform`, or `quant-backtest-runtime` as Python packages.
- Real research workflows consume published market-data assets through documented files, schemas, receipts, or explicit adapters. They do not import provider implementation code.
- The project exports model outputs and a manifest through versioned, language-neutral artifacts. It does not import a backtest-runtime or platform implementation to produce those artifacts.
- `quant-backtest-runtime` is the first-class downstream integration target for submitted or scheduled backtest jobs. It can consume predictions from this project and from unrelated model providers.
- `quant-platform` remains an optional provider of reusable backtest, portfolio, risk, replay, matching, impact, and execution mechanisms. The runtime may use those mechanisms internally; that does not create a dependency from Quant Deep Learning to `quant-platform`.
- `quant-research` may optionally receive evidence for private review, locked-period approval, or promotion workflows. Training and public model evaluation in this repository must remain usable without it.
- `ticknet.research` remains responsible for the model-specific proposal, execution, audit, and reproducibility functions needed to run this repository's studies independently. Before changing that code, inventory its overlap with `quant-research` and record which capabilities are public model tooling, optional private governance, or duplicated implementations. Do not bulk-move or delete code as part of this site project.

### 3.3 Artifact-contract gate

Before wiring scheduled work across repositories, define and test stable versions for the published market-data input and the model output artifacts. The output contract should identify at least the model and version, data snapshot and provenance, prediction date, instrument key, prediction value, eligibility/status fields, and feature/label contract. Keep portfolio weights, orders, fills, and runtime state outside the model-prediction schema.

The site may render curated results without waiting for production artifact integration. It must not present an example file or an unimplemented schema as a production contract.

## 4. Ownership Document

Move `MIGRATION-STATUS.md` to `docs/architecture/repository-boundaries.md` and use the title “Repository Ownership and Integration Boundaries.” Keep a short dated history of the earlier migration decision, then make the current ownership table and artifact directions the main content.

Update root README and documentation links, the MkDocs navigation, and documentation tests to use the new path. Do not retain a second copy of the same ownership policy in the root. The current file's `quant-platform` and `quant-research` assignments must be reconciled with the intended `quant-backtest-runtime` consumer boundary above before the rename is considered complete.

## 5. Publication-Site Options

### Option A: Expand MkDocs Material

Keep MkDocs as the single renderer, reorganize its home and navigation, and add audited static plots or custom chart components.

- Advantage: smallest migration; current English Markdown and Pages deployment remain in their existing toolchain.
- Cost: research-focused layouts, interactive charts, and a unified editorial data model require increasingly custom MkDocs extensions and theme overrides.

### Option B: Use Astro for the unified public site (recommended)

Use Astro for research overviews, study pages, search, and public technical documentation. Keep Markdown as source material, add explicit publication metadata, use React/ECharts only for charts that benefit from interaction, and generate one static Pages artifact. The target should be Astro-only for rendering and publishing. A temporary MkDocs build may remain as a migration parity check, then be removed once document coverage, route compatibility, and link checks live in the Astro toolchain.

- Advantage: one navigation system, one visual language, and flexible chart-rich research pages; this follows the proven direction of `quant-market-research`.
- Cost: migrate MkDocs routes, anchors, link checks, search, and publication filters. Existing URLs and all material research caveats need compatibility checks.

### Option C: Astro research front end plus a separate MkDocs site

Build the visual research experience in Astro and keep technical documentation in MkDocs at a distinct route.

- Advantage: appropriate when the documentation is a substantial, independently useful manual with its own navigation, release cadence, or readers, as in `quant-intel-platform` today.
- Cost: two navigation and styling systems remain. That ongoing split is not justified for this repository if Astro can render the research material and technical reference as one coherent site.

Select Option B for the final target. Workspace examples show that Astro and MkDocs can coexist when MkDocs owns a separate published manual, but coexistence should solve a real ownership need rather than follow from habit. For this repository, keep Markdown as source, migrate the public allowlist and technical reference into Astro, and remove the MkDocs build after equivalent coverage and route checks are in place. Do not keep two public sites or a permanent duplicate build as the settled design.

## 6. Site Information Architecture

The site is static and research-first. Suggested primary routes are:

- `/`: project overview, how the research pipeline works, current evidence summary, and links to active studies.
- `/studies/`: a catalog of studies by input family and evidence stage.
- `/studies/<study-id>/`: a study narrative with question, conclusion, sample scope, figures, methods, limitations, and source links.
- `/models/`: model families and comparable evaluation summaries, with explicit notes where samples or metrics do not support direct comparisons.
- `/methods/`: definitions for dates/splits, labels, ranking metrics, costs, and artifact handling.
- Existing technical documentation paths: preserved routes into detailed developer, model, and data-boundary documentation. The old MkDocs home at `/` becomes the research homepage; its documentation directory is available at `/documentation/`.

The catalog should show a short conclusion, evidence stage, sample dates, central metric, and most important open question. Initial study records should clearly distinguish:

- raw order-book validation candidates;
- minute-model baselines and their cost results;
- event-stream model OOS evidence and separate trading-conversion findings;
- FI-2010 DeepLOB paper reproduction, which is not evidence for A-share next-day prediction;
- the chip-layer proposal, which must remain “planned” until an independent implementation and evaluation exist.

Avoid a generic “all results are positive” hero. Surface the current status and include negative, mixed, and inconclusive results in the same discovery paths as favorable results.

## 7. Study Page Reading Pattern

Each study page should answer these questions in order:

1. What question did the study ask?
2. What is the current answer, and how strong is the evidence?
3. Which data, sample dates, universe, model, target, and split produced it?
4. What do the figures show? Explain axes, units, baselines, and uncertainty in everyday language.
5. What remains untested, failed, or inconclusive?
6. How can a technical reader inspect the method, code, configuration, or source record?

Use progressive disclosure: place plain-language findings and central figures first; provide definitions and detailed methods nearby; link to full run records and technical documentation rather than duplicating every table on the overview. Keep limitations adjacent to the claims they qualify.

## 8. Visual and Chart Language

Borrow the research-reading patterns used by `quant-market-research` without importing its application or copying its study content:

- a clear evidence-first overview;
- cards that summarize the topic, evidence stage, data range, current observation, open check, and next step;
- a consistent page header, navigation, typography, spacing, and light/dark theme;
- study pages that explain a chart in plain language before showing secondary diagnostics;
- direct paths from a headline result to data scope, method, and limitations.

Charts should support interpretation, not decoration. Use charts where they answer a research question: out-of-time results by fold/seed, paired model comparisons, cost sensitivity, cross-sectional ranking behavior, event-stream signal decay, or label/model diagnostics. Every chart must declare metric and units, sample scope, time split, comparison baseline, source, and any uncertainty interval that was actually computed. Do not manufacture confidence intervals or smooth away seed/fold disagreement.

Interactive charts should:

- remain usable on narrow screens and support keyboard focus where interaction exists;
- provide text captions and a static table or downloadable reviewed data for the values shown;
- load chart code only on pages that use it;
- retain meaningful rendered output when scripts fail or are disabled.

## 9. Public Data and Evidence Boundary

- Pages deployment consumes only an explicit allowlist of public study records and derived result snapshots.
- Do not publish raw L2, per-security predictions, checkpoints, full runtime outputs, private experiment locators, credentials, machine paths, or unreviewed internal notes.
- Keep full runs and retained datasets under the configured data root or approved external storage. Publish only small, reviewed, purpose-built summaries needed for figures.
- Each public snapshot records a stable study ID, schema version, source commit or artifact digest, generation/review date, data range, and fields permitted for publication.
- A static-site verification step rejects unlisted documents/data and known private paths or fields; it also checks that each chart's source and labels exist.
- Missing, stale, or incomplete evidence must be shown as unknown or pending; never substitute zero or a completed-looking placeholder.

## 10. Framework, Routes, and Build

- Use Astro's static output with the GitHub Pages base path `/quant-deep-learning/`.
- Prefer static, accessible SVG or HTML charts backed by reviewed snapshots and provide the values in an adjacent table. Add client-side chart code only when an interaction answers a real reader question.
- Keep study prose in Markdown or MDX with validated frontmatter for title, study ID, evidence status, date range, summary, public data references, and related pages.
- Render only documents admitted by a publication allowlist. Agent plans, internal design records, raw artifacts, and operational runbooks are not implicitly public pages.
- Publish one Astro-generated Pages artifact. During migration, the MkDocs output may be built into a temporary directory for parity checks, but it is not a second public artifact. Test the old route map and link/anchor compatibility; do not silently strand existing deep links.
- CI builds the static site and runs route, metadata, link, source-data, accessibility-oriented markup, and publication-boundary checks. It does not train models or read private data.

## 11. Migration and Delivery Stages

1. Update the repository-boundary document and define an ownership matrix that reflects the user's upstream/data/downstream direction.
2. Inventory current public documents, generated artifacts, route/anchor usage, and the exact set of evidence suitable for publication.
3. Define the minimal public study metadata and reviewed-result snapshot schema; create no charts from unreviewed runtime outputs.
4. Build an Astro shell and migrate the overview, catalog, and two representative studies: one event-stream result and one cost or minute-model result. Preserve access to all technical docs during this stage.
5. Verify text, charts, responsive layout, accessible table equivalents, routes, publication filters, and GitHub Pages base-path behavior.
6. Migrate remaining approved study pages and technical Markdown; move document coverage, strict-link, and publication checks into the Astro toolchain, then remove the MkDocs dependency and build after parity and link compatibility checks pass.
7. Document the production prediction/signal contract as a separate cross-repository design and implement it provider/consumer-first through independent PRs.

## 12. Non-Goals and Safety Constraints

- Do not combine the site redesign with broad source-code migrations from `quant-platform` or `quant-research`.
- Do not make any of these repositories a required Python dependency of `quant-deep-learning`.
- Do not publish private experimental outputs merely to make the site appear comprehensive.
- Do not relabel candidates, pilots, historical correlations, or paper reproductions as validated A-share trading results.
- Do not replace existing study evidence with newly calculated values inside a visual redesign. Corrections require their own provenance and review.
- Do not create a live inference or trading service as part of this static-site project.

## 13. Acceptance Criteria for the Follow-On Implementation Plan

- A dependency graph and ownership table show the market-data provider upstream and backtest runtime downstream; direct project dependencies on `quant-platform` and `quant-research` are absent.
- The renamed ownership document is the single current source for repository boundaries; the earlier migration remains clearly historical.
- Both general and quantitative readers can enter from the homepage and reach a concise conclusion, evidence, limitations, and detailed methods without encountering conflicting summaries.
- Every published study and chart identifies its evidence status and data/sample scope.
- Every public data file is allowlisted, versioned, reviewed, small enough for a static site, and traceable to its source.
- Existing public documentation links and anchors remain accessible or have explicit static compatibility routes.
- The site builds as one static Pages artifact under the repository project prefix, and CI enforces privacy/publication boundaries without accessing private data or training models.
- The UI and chart pages receive visual checks on desktop and mobile, including script-disabled or missing-data states.

## 14. Review Notes

This design is based on the Quant Deep Learning README, status and boundary documents, previous MkDocs/Pages configuration, repository instructions, and a workspace inventory of Astro projects and their Pages workflows. `quant-market-research` provides a public research-design reference: its [overview](https://runchengxie.github.io/quant-market-research/) and [low-turnover report](https://runchengxie.github.io/quant-market-research/research/factors/low-turnover/) pair plain-language findings with scope, figures, and caveats. The implementation now follows this design in the current PR.
