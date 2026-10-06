# Development and Maintenance

## Module boundaries

The core package uses the standard `src` layout:

| Module | Responsibility |
|---|---|
| `ticknet.model` | FI-2010-compatible DeepLOB network and model factory |
| `ticknet.dataset` | FI-2010-compatible tensor constants and synthetic-data helpers |
| `ticknet.train` | Training utilities shared by A-share research and FI-2010 reproduction (`set_seed`, `resolve_device`, `f1_metrics`) |
| `ticknet.fi2010` | FI-2010 dataset conversion/loading, DeepLOB training, Colab staging, and plotting |
| `ticknet.nextday` | Next-day labels, date splits, shard reading, chunked models, cross-sectional metrics, and training, including minute HGB, TCN, and GRU baselines and the raw-order-book path |
| `ticknet.eventstream` | Lossless L2 event packing, causal Transformer training, and prediction export |
| `ticknet.research` | Experiment workflow, including ExperimentSpec v2, typed executors, strategy and locked-period isolation, registry, prediction audits, and deterministic evaluation |

The FI-2010 paper reproduction is a maintained, separate research track. Its reusable code is in `src/ticknet/fi2010/`, configuration is in `configs/fi2010-colab.yaml`, and automated tests are collected in the main test suite. Its results remain distinct from A-share next-day evidence.

Most scripts are human-run entry points and orchestration. Reusable data contracts, model computation, and evaluation belong under `src/ticknet/`. A few scripts still orchestrate long Colab jobs; future refactoring should move reusable parts into the core package gradually.

Common scripts:

| Script | Purpose |
|---|---|
| `smoke_test.py` | Quick model check for FI-2010-compatible DeepLOB |
| `prepare_nextday.py` | Convert stock-day bars and event manifests into next-day prediction NPY shards |
| `run_nextday_baseline.py` | Logistic Regression baseline over aggregated intraday features |
| `run_minute_baseline.py` | HGB baseline over minute aggregates, with multi-year rolling validation and prediction export |
| `prepare_minute_shards.py` | Split minute sequences into `samples × time × features` shards for temporal models |
| `materialize_minute_features.py` | Atomically materialize formal monthly minute aggregates |
| `evaluate_cost_adjusted.py` | Thin entry point for Top-K long-only cost evaluation; also supports historical quantile long-short diagnostics |

## Colab and notebook boundaries

All active Colab workflows use Python entry points. The top-level `notebooks/` directory has been removed. Old notebooks were converted to Python snapshots under `examples/historical-workflows/` only to preserve early interactive workflows.

| Former notebook capability | Current Python entry point |
|---|---|
| Next-day model training, recovery, and locked evaluation | `ticknet.nextday.train`, `ticknet-nextday-train`, `ticknet-nextday-evaluate` |
| Multi-horizon validation | `ticknet.nextday.horizon_cli`, `ticknet-nextday-evaluate-horizons` |
| Colab sessions, data staging, and artifact retrieval | `scripts/run_colab_nextday.py` |
| Remote Colab jobs | `scripts/colab_multi_horizon_job.py` |

Old notebooks are retired, and their Python snapshots are not execution entry points. CLI contract tests, `tests/test_horizon_cli.py`, and `tests/test_colab_nextday.py` cover active entry points. Production code and documentation must not depend on retired notebook files.

The compatibility path that excluded one training fold based on `folds.npy` has been removed. It did not match FI-2010's prebuilt split semantics and added configuration branches and leakage risk. Training stops when metadata is missing from real data.

## Configuration

The `Config` dataclass holds defaults. YAML overrides defaults, then command-line arguments override YAML. Unknown YAML fields raise an error so spelling mistakes cannot be silently ignored.

Command-line options use hyphens, such as `--data-path`. YAML keys use underscores, such as `data_path`.

Inspect complete command-line help:

```powershell
ticknet-nextday-train --help
ticknet-minute-tcn-train --help
ticknet-research --help
```

## Test scope

Tests are organized by workflow and use synthetic data. They do not require real market data, Google Drive, or the complete FI-2010 dataset.

The main quality gate includes FI-2010 synthetic-data tests for the five prediction horizons and label mapping, 40-feature inputs, Setup 1 and Setup 2 selection, non-overlapping training/validation source rows, text conversion, partition metadata, and streaming NPY writes. Tests use synthetic fixtures; real FI-2010 data is not required.

Next-day cross-sectional tests cover:

- Trading calendars, adjacent-day labels, and cross-sectional three-class cut points
- Leakage checks between signal time and label date, plus purge of boundary-crossing samples
- Chunked DeepLOB, intraday GRU, two-head outputs, and sharded dataset indexing
- Continuous scores and three-class metrics including Macro F1, MCC, Brier, and daily Rank IC
- Gradient accumulation, AMP, checkpoint recovery, and experiment-signature conflicts
- Inference from raw `N × 40` snapshots to scores and directional probabilities
- Logistic Regression over aggregated intraday features
- Monthly Shanghai/Shenzhen snapshot Parquet adapters, including dynamic universes, candidate-window selection, shard fingerprints, and row-group skipping
- YAML and command-line overrides, plus CPU device selection
- Multi-horizon label sidecars and leakage prevention through the end of the return period

Minute workflow tests cover:

- The minute HGB data pipeline and L2 and Tushare feature sources
- NaN median fill, short-window padding, and validation for minute-sequence shards
- Minute TCN and GRU models, sharded datasets, training entry points, and after-cost backtests
- Prediction audits for IC, deciles, extreme-day contributions, and winsorization

Event-stream tests cover:

- Lossless packing of three raw streams, ID-link resolution, and daily indexes
- Event-window sampling, multi-task prediction heads, and daily signal heads
- Training, checkpoint recovery, dataset fingerprints, and prediction-export contracts

CLI contract tests read all command declarations in `pyproject.toml`, import each target function, and run `--help`. Tests therefore catch differences between declarations and implementations when entry points are added, removed, or moved.

Documentation tests check maintained Markdown links and English prose. Fenced code, inline code, and explicitly marked preserved source text are exempt from the language check. The archival comparison excerpt also has a pinned content hash.

Research workflow tests cover:

- Strict ExperimentSpec v2 parsing, allowlisted executors, structured metric gates, and artifact contracts
- Programmatic isolation of locked-test periods for manifests, explicit prediction inputs, and predictions generated by training
- Two-step locked approval and consumption, binding by content SHA-256, keeping raw tokens out of the database, and replay rejection
- Recursive metrics, uniqueness, parent experiments, failure states, and artifact SHA-256 in SQLite Registry v2
- Brainstorm, Critic, orchestration, mandatory Audit, and deterministic `KEEP`, `EXTEND`, and `DISCARD` decisions
- Fixed-K long-only portfolios, rank buffers, ineligible securities, weight drift, costs, and detailed artifacts
- Prediction-artifact checksum materialization, multi-seed baseline differences, and direction-normalized paired improvements
- Walk-forward aggregation across windows with different data fingerprints, metric direction, and worst-window selection
- Baseline selection from Registry to ResearchContext, return paths for failures and audits, stable fingerprints, and novelty-replay rejection
- Shared context for Brainstorm and Critic, budget and executor limits, and context-review snapshots

The smoke script checks the DeepLOB forward pass, softmax, gradients, parameter count, and FI-2010 dataset windows using synthetic data. It does not read real data. `scripts/check.py` includes it in the manual Python quality checks.

## Quality gates

GitHub Actions runs the public quality gate on pull requests and pushes to `main`. It uses locked development dependencies to run Ruff, formatting, `ty`, pytest with coverage, and Python compilation. Coverage is reported, but remote CI currently has no additional minimum threshold. Full training, slow tests, real-data checks, and GPU checks are run manually as needed.

Run the local gate with:

```bash
python scripts/check.py
```

The script runs:

```bash
ruff check .
ruff format --check .
ty check
python -m pytest --cov --cov-report=term-missing
python scripts/smoke_test.py
```

Ruff checks pycodestyle, Pyflakes, import ordering, modern syntax, common defects, comprehensions, pytest style, and simplification rules. `RUF001`, `RUF002`, and `RUF003` are disabled only because the existing source contains Chinese punctuation in strings, docstrings, and comments.

The global unresolved-import ignore was removed from `ty`. Historical Colab Python snapshots retain a scoped override because local environments generally do not include `google.colab`.

The required quality gate runs in GitHub Actions for pull requests and updates to `main`. No local hooks are installed. Run the documented `uv run` commands before requesting a PR check if you want the same feedback locally. The gate includes file hygiene, Ruff, formatting, notebook lint, type checking, pytest with coverage, smoke checks, and a static Astro build. Tests fail if core-package branch coverage is below 80%.

## Dependencies

Runtime and development dependencies are declared in `pyproject.toml`. Install the development environment with:

```bash
python -m pip install -e ".[dev]"
```

The repository commits `uv.lock`. After changing dependencies, run:

```bash
uv lock
```

After modifying CLI entry points in `pyproject.toml`, reinstall the editable project. An existing `.venv` does not automatically create new command scripts.

## Follow-up refactoring

The highest-priority refactoring opportunities are:

1. Extract shared epoch training and evaluation loops from `train.py`, `train_tcn.py`, and `train_gru.py`.
2. Split large Registry, Spec, and portfolio-evaluation modules in `ticknet.research` by validation, persistence, and evaluation responsibility.
3. Publish shared date, metric, and atomic-write helpers as stable interfaces instead of relying on private functions in other modules.
4. Narrow orchestration responsibilities in `run_colab_nextday.py` and `colab_multi_horizon_job.py`, moving testable job specifications and execution logic into the core package.
5. Add structured logs and run-state monitoring for long training jobs.

Core modules currently vary in size. Trainers repeat substantial loop logic, and the research workflow and some orchestration scripts have multiple responsibilities. Refactor in the order above. Keep CLI, configuration, and artifact contracts stable during each change, and remove complexity exceptions after the corresponding refactoring.

The FI-2010 training loop and CLI live in `ticknet.fi2010.train`; converter, plotting, Colab, and smoke entry points live in `ticknet.fi2010.scripts`. Shared random-seed, device, and classification-metric helpers remain in `ticknet.train`. `ticknet.research` is decoupled from `nextday` through CLI names and YAML configuration and does not directly import its implementation.
