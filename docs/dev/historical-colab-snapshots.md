# Historical Python Snapshots

This directory contains historical Python snapshots retained for reference. They were converted from old Colab notebooks to preserve early interactive flows and parameter assembly. They are not active project entry points.

| File | Historical purpose | Current replacement |
|---|---|---|
| `nextday_end_to_end.py` | raw-200 pilot training, recovery, and locked evaluation | `ticknet-nextday-train`, `ticknet-nextday-evaluate`, `scripts/run_colab_nextday.py` |
| `nextday_multi_horizon_validation.py` | 2024 validation across multiple horizons and plots | `ticknet-nextday-evaluate-horizons`, `scripts/run_colab_nextday.py --workflow multi-horizon-validation` |
| `colab_fi2010.py` | FI-2010 reproduction | `ticknet-fi2010-colab` |

Python modules and automated tests now cover training, multi-horizon evaluation, date permissions, data staging, and artifact retrieval. To reproduce the current workflows, start with the [development guide](development-guide.md) and [Colab CLI automation](colab-cli-automation.md).
