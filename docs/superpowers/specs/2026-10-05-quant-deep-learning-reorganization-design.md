# Quant Deep Learning Repository Reorganization

## Purpose

Re-establish the former `deep-learning-tick-data-prediction` repository as the canonical public home for independent end-to-end quantitative deep-learning research. The project should run its model research workflows without `quant-platform`, while allowing downstream systems to consume versioned prediction and signal artifacts.

The repository is currently public and already unarchived. Its latest `main` commit marks it migration-only. This design supersedes that direction after an auditable ownership and compatibility transition; it does not assume that all implementation currently present in `quant-platform` or `quant-research` can be copied back unchanged.

## Goals

- Give model-specific data representations, training, inference, evaluation, and study implementations one canonical repository.
- Keep generic replay, matching, execution simulation, portfolio construction, risk, and backtesting in `quant-platform`.
- Keep private experiment governance, locked-test approval, and promotion evidence in `quant-research`.
- Integrate systems through versioned prediction, signal, and model-manifest artifacts, with no required Python dependency between the model project and `quant-platform`.
- Rename the public repository to `quant-deep-learning` and publish one MkDocs Material site through GitHub Pages and GitHub Actions.
- Make maintained explanatory Markdown documentation English-only, while preserving source identifiers, code, bibliographic titles, and original source material where translation would alter evidence.
- Incorporate the chip-layer research as a documented study and implementation track without redistributing copyrighted broker research PDFs absent explicit redistribution permission.

## Non-goals

- Moving generic market microstructure or backtesting infrastructure into the model repository.
- Making `quant-platform` depend on this repository.
- Publishing private credentials, raw vendor data, checkpoints, or generated runtime state.
- Treating existing research results as validated merely because documentation is translated or reorganized.
- Reproducing a broker report verbatim or publishing its PDF without a confirmed license.

## Repository boundaries and artifact flow

The deep-learning repository owns model-specific representations and transformations, model architectures, training and inference, model-specific diagnostics, reproducible study definitions, and the conversion of predictions into documented signal artifacts.

`quant-market-data-platform` remains the owner of market-data ingestion and publication. The model repository may consume its published data contracts without taking ownership of ingestion.

`quant-platform` remains strategy-agnostic and owns generic portfolio, backtest, risk, replay, matching, impact, and execution simulation capabilities. It may consume versioned prediction or signal artifacts without importing model code.

`quant-research` remains the owner of private experiment governance, evidence controls, locked-test access, and promotion decisions. Public model research documentation may link to public contracts, but must not expose private reproduction locators or credentials.

Before code ownership changes, document and version the artifact contract, including schema identity, model/version identity, feature-set identity, event or prediction time, symbols, eligibility flags where applicable, and provenance. Consumers must reject unsupported schema versions clearly. The exact schema and compatibility policy are implementation-plan work following this design.

## Naming and public identity

Use `quant-deep-learning` as the repository name and project title. Retain `ticknet` as a compatibility namespace for the existing LOB, event-stream, and next-day implementation during the transition. New namespaces should reflect model families or studies where useful; a broad package rename is not part of the initial reactivation.

The GitHub repository rename is a remote operation. First merge the preparatory changes through the repository's normal PR process. Then rename the repository, verify GitHub's redirect, update the local `origin` URL and documentation links, and verify that Pages and Actions use the renamed repository. Do not assume unarchive is needed: GitHub currently reports the repository as unarchived.

## Documentation and research content

All maintained explanatory Markdown in the project, including the root README, `docs/`, and contributor instructions, will be written in clear English. Code, CLI names, paths, schemas, bibliographic titles, direct quotations, and original research source files remain unchanged when translation would corrupt an identifier or evidence. Preserve dated observations, metrics, assumptions, failed experiments, sample boundaries, provenance, and uncertainty during translation. Do not turn research notes into promotional summaries.

MkDocs Material will serve both as the project showcase and the documentation site. The landing page will explain the project scope, research tracks, evidence stage, artifact integration, and links to setup, architecture, study, and reproduction documentation. Navigation will be generated from a reviewed `mkdocs.yml`; strict builds will catch broken references. GitHub Actions will build on pull requests and deploy the default branch to GitHub Pages using the official Pages workflow permissions and artifacts. CI must not train models or require private data, credentials, or GPU resources.

The chip-layer study will have a source citation, a method summary in original wording, assumptions, a reproduction plan, implementation links, and a results page that distinguishes report claims from independently observed results. The source report itself stays out of the public repository unless redistribution rights are verified.

## Migration sequence

1. Inventory the target repository, archived migration snapshot, `quant-platform`, and `quant-research`; map duplicate modules and identify the active implementation and evidence owner for each capability.
2. Specify artifact schemas and the compatibility boundary with consumers.
3. Reactivate project documentation and contribution metadata, establish the English documentation inventory, then implement the showcase site and Pages workflow.
4. Bring model-specific code back in controlled slices, add the chip-layer study, and preserve compatibility adapters only with explicit removal conditions.
5. Remove superseded copies from provider repositories only after the replacement is merged and consumer compatibility is demonstrated through their normal PRs.
6. Rename the GitHub repository after preparatory PRs are merged; verify redirects, Pages, Actions, and remotes.

Each cross-repository migration must be reviewed against that repository's own `AGENTS.md`, data contract, and release workflow. Provider changes must land before consumers rely on them. Raw data and generated artifacts remain outside Git.

## Quality and acceptance criteria

- Repository ownership is documented consistently in the README, architecture docs, and migration record.
- A clean environment can install and run the standalone synthetic-data validation path without installing `quant-platform`.
- Artifact schemas have versioning, provenance, and compatibility checks, with tests for valid and invalid examples.
- All maintained explanatory Markdown is English, links remain navigable, and technical identifiers and research evidence remain intact.
- MkDocs builds in strict mode in CI; the showcase has a clear landing page and useful navigation.
- The Pages workflow deploys only from the intended default branch and needs no private credentials or model-training resources.
- The chip-layer study clearly separates source-reported methods/results from independently reproduced evidence and does not redistribute an unlicensed report.
- Existing required project checks pass for each implementation slice; CI and documentation build results are recorded in the corresponding PR.
- After repository rename, the old GitHub URL redirects and the new canonical URL is used by the clone, Pages site, badges, and documentation.

## Risks and mitigations

- Duplicate implementations may have diverged. Inventory first and migrate in small slices; do not bulk-copy code or delete provider code until consumers move.
- Existing research claims can be distorted by translation. Preserve exact values, dates, assumptions, caveats, and source links; review translated notes against their originals.
- Pages publication can expose unintended files. Publish only the MkDocs site tree, inspect navigation and generated output, and keep raw data and private locators excluded.
- Repository rename can break hard-coded links or local remotes. Use GitHub redirects as a bridge, update canonical URLs, and verify all site and workflow references after the rename.
- A public chip-layer implementation can imply endorsement or reproduction of a source report. Keep attribution clear, write original explanatory prose, and publish no report PDF without permission.

## Decisions deferred to the implementation plan

- Exact artifact schema fields, serialization format, and compatibility guarantees.
- Which duplicate implementation is canonical for each model family after comparing current code and tests.
- Whether the initial Pages release includes interactive charts or only static generated research summaries.
- Repository license and citation metadata after inspecting current licenses and source rights.
- Exact order and PR boundaries for multi-repository code moves.
