# Repository Boundaries

Quant Deep Learning is an independent model-research project. It can prepare model inputs, train and evaluate models, and export prediction artifacts without installing `quant-platform` or `quant-research`.

## Ownership

| Repository | Owns | Integration with this project |
|---|---|---|
| `quant-market-data-platform` | Provider ingestion, canonical schemas, quality receipts, versioned and published market-data assets | Upstream data through documented files and schemas; this project owns model-specific adapters and labels |
| `quant-deep-learning` | Model-specific representations, training, inference, evaluation, and research records | Produces versioned prediction and signal artifacts |
| `quant-backtest-runtime` | Backtest task protocol, execution state, workers, resource controls, artifact checks, and runtime publication | Downstream consumer of prediction/signal artifacts; does not import model implementations |
| `quant-platform` | Reusable portfolio, risk, backtest, replay, matching, and execution algorithms | Optional algorithm provider used by the runtime; not a dependency of this repository |
| `quant-research` | Private experiment governance, locked-test approval, evidence registry, and promotion policy | Optional external control plane; not required to train, predict, or evaluate here |

## Data flow

```text
quant-market-data-platform
       │ versioned published data
       ▼
quant-deep-learning
  adapt → represent → train → validate → predict
       │ prediction / signal artifact + manifest
       ▼
quant-backtest-runtime
  validate → schedule → execute → publish results
       │
       └── may use generic algorithms from quant-platform
```

No direct Python dependency or repository import is required along these boundaries. The producer and consumer exchange versioned files and manifests. This keeps the model code usable in a standalone environment and lets the runtime evolve independently.

## Current implementation status

This repository owns its model-specific training, inference, and evaluation code. It already has an alpha-signal export adapter, but a stable cross-repository prediction, signal, and model-manifest schema is still being specified. Until that contract is implemented and verified by both producer and consumer, treat existing copies in other repositories as migration-era implementations and do not assume wire compatibility.

The market-data platform remains the source for shared published data. Do not copy provider ingestion, credentials, raw market data, or consumer runtime state into this repository. Store experiment data and checkpoints outside Git under the configured project data root.

## Boundary checks

Automated tests ensure this package does not import provider, platform, research, or runtime implementations and does not declare them as required dependencies. Adapters in this repository may depend on documented data formats and artifact schemas; they must not depend on another repository's internal Python modules.
