# Historical Python Snapshots

Historical Colab snapshots are retained as text under `docs/archive/historical-workflows/`.
They preserve early interactive flows and parameter assembly, not runnable project
entry points. The former `examples/historical-workflows/` directory has been retired.
Do not move its notebook-cell code directly into `src` or `scripts`: it includes
old repository paths, fixed historical experiment parameters, and Colab-only side
effects. Its maintained capabilities already have the replacements below.

| File | Historical purpose | Current replacement |
|---|---|---|
| [End-to-end snapshot](../archive/historical-workflows/nextday_end_to_end.py.txt) | raw-200 pilot training, recovery, and locked evaluation | `ticknet-nextday-train`, `ticknet-nextday-evaluate`, [Colab script](../../scripts/run_colab_nextday.py) |
| [Multi-horizon snapshot](../archive/historical-workflows/nextday_multi_horizon_validation.py.txt) | 2024 validation across multiple horizons and plots | `ticknet-nextday-evaluate-horizons`, `scripts/run_colab_nextday.py --workflow multi-horizon-validation` |
| [FI-2010 snapshot](../archive/historical-workflows/colab_fi2010.py.txt) | FI-2010 reproduction | `ticknet-fi2010-colab`, [package entry point](../../src/ticknet/fi2010/scripts/run_colab.py) |

Python modules and automated tests now cover training, multi-horizon evaluation, date permissions, data staging, and artifact retrieval. To reproduce the current workflows, start with the [development guide](development-guide.md) and [Colab CLI automation](colab-cli-automation.md).
