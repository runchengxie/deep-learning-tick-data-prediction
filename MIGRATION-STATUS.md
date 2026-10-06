# Migration and Ownership Status

**Canonical owner: `quant-deep-learning`**

This repository is being reactivated as the public home for independent end-to-end quantitative deep-learning research. The former migration-only policy on the current `main` branch is a historical decision and is superseded by the repository reorganization approved for this branch. This ownership change takes effect when the reactivation change is merged.

## Historical migration decision

- The earlier migration moved model research toward `quant-research` and microstructure integrations toward `quant-platform`.
- The migration review used source commit `a56f2a08a9ba9f0062aaac0b98f1cd86ee5602f1`.
- Commit `d077446` later marked this repository migration-only and directed new work to those repositories.
- `quant-research` retains a historical snapshot and inventory for comparison.

The earlier migration did not remove all duplicate implementations. The current transition restores one canonical owner for model-specific representations, architectures, training, inference, evaluation, and study documentation. Generic replay, matching, execution simulation, portfolio construction, risk, and backtesting remain in `quant-platform`. Private experiment governance, locked-test access, and promotion evidence remain in `quant-research`.

## Integration boundary

The model project must run without `quant-platform`. Consumers integrate through versioned prediction, signal, and model-manifest artifacts rather than importing model implementation modules. Exact schemas and compatibility behavior will be specified before code is moved between repositories.

## Transition policy

Until the new ownership and compatibility contracts are implemented, treat existing copies in `quant-platform` and `quant-research` as migration-era code. Do not delete or bulk-copy implementations. Inventory their current state, establish artifact contracts, and migrate in reviewable slices. Remove provider copies only after the replacement is merged and consumers have been verified.

The repository is public and currently unarchived. Repository rename to `quant-deep-learning` and GitHub Pages publication are follow-up operations after their preparatory changes are merged.
