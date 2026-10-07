# Quant Deep Learning

Quant Deep Learning is an independent research project on end-to-end deep-learning models for structured market data. It began with a DeepLOB reproduction and now focuses on next-day cross-sectional prediction from Chinese A-share data, using order-book snapshots, minute-level inputs, and L2 event streams.

The project owns model-specific representations, training, inference, evaluation, and study records. It can run without `quant-platform` or `quant-research`. `quant-market-data-platform` is the upstream provider; `quant-backtest-runtime` is the downstream artifact consumer. Neither side needs to import this repository's Python modules. See the [repository boundaries](docs/architecture/repository-boundaries.md).

## Research tracks

- Raw order-book models use the ten-level snapshot available at a prediction time.
- Minute models test lower-cost aggregated price and volume inputs.
- Event-stream models encode order, trade, and snapshot events with causal sequence models.
- Research tooling records experiment identity, cost evaluation, prediction audits, and controlled access to locked data.
- Chip-layer research has a CPU HGB experiment path for lagged age-layer summaries and paired minute-feature comparisons. [Its source and contract](docs/research/chip-age-hgb.md) distinguish synthetic engineering checks from real training and broker reproduction. This repository does not redistribute broker reports without permission.

These tracks do not share the same evidence stage. See [Project status](docs/project-status.md) for the dated results, limits, and current research questions.

## Research boundaries

The FI-2010 DeepLOB paper reproduction is maintained as an independent research track in `ticknet.fi2010`. It does not establish that next-day stock ranking works. Results from `ticknet.nextday` and `ticknet.eventstream` are separate from the FI-2010 reproduction.

The main research path uses trading-day splits and evaluates cross-sectional ranking on out-of-time data. It also examines turnover and transaction costs. A positive Rank IC alone does not establish that a signal is tradable. Do not describe results as reproduced or validated unless the documented evidence supports that claim.

## Quick start

The public synthetic-data checks do not require private market data, a GPU, or `quant-platform`. Python 3.10 or later is supported.

```bash
uv sync --locked --extra dev
uv run python scripts/check.py
```

The check script runs Ruff, formatting checks, `ty`, pytest with coverage, and synthetic DeepLOB/FI-2010 smoke checks. CI is the required quality gate; no Git hooks need to be installed. The full [development guide](docs/operations/development-guide.md) describes the test and data boundaries.

On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1`. After changing command entry points in `pyproject.toml`, reinstall the project in editable mode so the environment's script launchers are refreshed.

## Documentation

- [Research presentation site](https://runchengxie.github.io/quant-deep-learning/) presents current evidence in plain language with sample scope, charts, limitations, and links to technical records.
- [Browse technical documentation](https://runchengxie.github.io/quant-deep-learning/documentation/) or [search public records](https://runchengxie.github.io/quant-deep-learning/search/).

- [Project status](docs/project-status.md) summarizes dated capabilities, evidence, and open work.
- [Model catalog](docs/model-catalog.md) compares model inputs, methods, strengths, and limitations.
- [Cross-sectional prediction](docs/nextday/cross-sectional-prediction.md) defines samples, labels, date splits, training, and evaluation.
- [Event-stream guide](docs/nextday/eventstream.md) describes event packing, causal training, and prediction export.
- [Data boundary](docs/architecture/data-boundary.md) describes which system owns each data transformation.
- [Repository boundaries](docs/architecture/repository-boundaries.md) defines the upstream/downstream integration and artifact boundary.
- [Documentation index](docs/documentation-index.md) links to topic guides, research records, and operating notes.
- [Reproduction audit](docs/reproduction-audit.md) records the scope and checks for the separate FI-2010 research track.

## Repository layout

```text
src/ticknet/             Shared training utilities and compatibility models
src/ticknet/nextday/     Next-day labels, datasets, minute models, and raw-book models
src/ticknet/eventstream/ L2 event packing, causal Transformer, and prediction export
src/ticknet/research/    Proposals, execution, audits, registry, and research agents
scripts/                 Human-run data preparation and local check entry points
tests/                   Automated checks that do not require private market data
configs/                 Local and Colab configurations
docs/                    Project status, technical guides, roadmaps, and research records
docs/references/         Papers and reading notes
src/ticknet/fi2010/      FI-2010 DeepLOB reproduction package
docs/archive/            Historical workflow text; active code lives in src and scripts
```

The Python import namespace remains `ticknet` for compatibility. Current repository ownership is recorded in [Repository Boundaries](docs/architecture/repository-boundaries.md). Contributor and agent rules are in [AGENTS.md](AGENTS.md).
