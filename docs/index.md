# Quant Deep Learning

Independent end-to-end deep-learning research for structured market data.

Quant Deep Learning develops representations, models, training, inference, and research evaluation for order-book snapshots, minute data, and L2 event streams. The project can run without `quant-platform` or `quant-research`. It consumes published market-data assets upstream and is designed to send versioned prediction and signal artifacts to `quant-backtest-runtime` downstream.

## Research tracks

| Track | Scope | Evidence stage |
|---|---|---|
| Raw order book | Snapshot models over raw multi-level order-book inputs | A 1M-parameter, 200-step model remains a validation-stage candidate. Locked-test evaluation has not been triggered. |
| Minute data | HGB, TCN, and GRU baselines with rolling evaluation and cost analysis | The current M3 v2 cost matrix is recorded as a formal research result. Its reported Rank IC does not establish positive net returns after the target transaction cost. |
| L2 event stream | Lossless event packing and causal Transformer models | The 100M-parameter recent-fold runs show positive H5 OOS Rank IC across three seeds. Top-100 net returns remain a separate unresolved question. |
| Chip-layer representations | Neural models over estimated market chip distributions | Proposed research track. No independent reproduction result is claimed here. |

Evidence summary above reflects the project status record dated 2026-08-29. Consult [Project status](project-status.md) for sample definitions, exact figures, caveats, and later updates before relying on a result.

## Research workflow

```text
published market data
        ↓
model-specific representation and features
        ↓
training, validation, and walk-forward evaluation
        ↓
prediction and signal artifacts
        ↓
independent portfolio, risk, or backtest consumers
```

The model project owns model-specific computation and diagnostics. `quant-market-data-platform` owns shared data publication. `quant-backtest-runtime` consumes model artifacts and may use generic algorithms from `quant-platform`. `quant-research` remains an optional private evidence and approval system. See [Repository Boundaries](architecture/repository-boundaries.md).

Artifact schemas and compatibility guarantees are being specified before implementation ownership moves between repositories. Until then, treat existing copies in other repositories as migration-era implementations. See [Repository Boundaries](architecture/repository-boundaries.md) for the current ownership contract and migration history.

## Get started

- Install the project and run public synthetic-data checks with the [development guide](operations/development-guide.md).
- Read the [model catalog](model-catalog.md) for model inputs and known limitations.
- Follow the [cross-sectional prediction protocol](nextday/cross-sectional-prediction.md) for samples, labels, date splits, and evaluation.
- Review [event-stream models](nextday/eventstream.md) for L2 input contracts and prediction export.
- Browse the [documentation index](documentation-index.md) for research notes, operations, and source material.

## Evidence standards

FI-2010 remains a separate DeepLOB paper-reproduction track in `ticknet.fi2010`. It cannot establish next-day stock-ranking performance. All current claims must state their sample dates, split, metric definition, costs where relevant, provenance, and limitations. A positive ranking metric alone does not establish that a strategy is tradable.
