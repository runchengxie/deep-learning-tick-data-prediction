# Quant Deep Learning

Quant Deep Learning is an independent research project for end-to-end deep-learning models on structured market data. It began as a DeepLOB reproduction and now focuses on next-day cross-sectional prediction from Chinese A-share market data, including order-book snapshots, minute-level inputs, and L2 event streams.

The project owns model-specific representations, training, inference, evaluation, and study records. It can run without `quant-platform`. Downstream systems can consume versioned prediction and signal artifacts without importing this repository's Python modules.

## Research tracks

- Raw order-book models use the ten-level snapshot available at a prediction time.
- Minute models test lower-cost aggregated price and volume inputs.
- Event-stream models encode order, trade, and snapshot events with causal sequence models.
- Research tooling records experiment identity, cost evaluation, prediction audits, and controlled access to locked data.
- Chip-layer research is a planned study track. Its source and independent reproduction status will be documented separately. This repository does not redistribute broker reports without permission.

These tracks do not share the same evidence stage. See [Project status](docs/project-status.md) for the dated results, limits, and current research questions.

## Research boundaries

The archived FI-2010 work is kept under `legacy/` as a reproduction reference. It does not establish that next-day stock ranking works. Results from `ticknet.nextday` and `ticknet.eventstream` are separate from the FI-2010 paper reproduction.

The main research path uses trading-day splits and evaluates cross-sectional ranking on out-of-time data. It also examines turnover and transaction costs. A positive Rank IC alone does not establish that a signal is tradable. Do not describe results as reproduced or validated unless the documented evidence supports that claim.

## Quick start

The public synthetic-data checks do not require private market data, a GPU, or `quant-platform`. Python 3.10 or later is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/check.py
```

The check script runs Ruff, formatting checks, `ty`, pytest with coverage, and a FI-2010 compatibility-model smoke check. The full [development guide](docs/operations/development-guide.md) describes the test and data boundaries.

On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1`. After changing command entry points in `pyproject.toml`, reinstall the project in editable mode so the environment's script launchers are refreshed.

## Documentation

- [Project status](docs/project-status.md) summarizes dated capabilities, evidence, and open work.
- [Model catalog](docs/model-catalog.md) compares model inputs, methods, strengths, and limitations.
- [Cross-sectional prediction](docs/nextday/cross-sectional-prediction.md) defines samples, labels, date splits, training, and evaluation.
- [Event-stream guide](docs/nextday/eventstream.md) describes event packing, causal training, and prediction export.
- [Data boundary](docs/architecture/data-boundary.md) describes which system owns each data transformation.
- [Documentation index](docs/README.md) links to topic guides, research records, and operating notes.
- [Reproduction audit](docs/reproduction-audit.md) records the scope and checks for the archived FI-2010 work.

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
legacy/                  Archived FI-2010 reproduction reference
```

The Python import namespace remains `ticknet` for compatibility. Repository ownership and boundaries are recorded in [MIGRATION-STATUS.md](MIGRATION-STATUS.md). Contributor and agent rules are in [AGENTS.md](AGENTS.md).
