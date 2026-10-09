"""Read-only trial accounting from a consistent registry snapshot."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ticknet.research.registry import ExperimentRegistry


def export_trial_accounting(
    registry: ExperimentRegistry, *, family_by_experiment: Mapping[str, str]
) -> dict[str, Any]:
    if any(
        not isinstance(value, str) or not value.strip() for value in family_by_experiment.values()
    ):
        raise ValueError("family identifiers must be nonempty strings")
    with sqlite3.connect(registry.database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        experiments = connection.execute(
            "SELECT experiment_id,status,stage,evaluation_decision "
            "FROM experiments ORDER BY experiment_id"
        ).fetchall()
        runs = connection.execute(
            "SELECT experiment_id,seed,status,exit_code FROM runs ORDER BY experiment_id,seed"
        ).fetchall()
    identities = {row["experiment_id"] for row in experiments}
    unknown = set(family_by_experiment) - identities
    if unknown:
        raise ValueError(f"family mapping contains unknown experiments: {sorted(unknown)}")
    rows = [
        {
            **dict(row),
            "family_id": family_by_experiment.get(row["experiment_id"]),
            "runs": [dict(run) for run in runs if run["experiment_id"] == row["experiment_id"]],
        }
        for row in experiments
    ]
    return {
        "schema_version": "ticknet.trial-accounting.v1",
        "experiment_count": len(rows),
        "unfinished_experiments": sum(row["status"] in {"proposed", "running"} for row in rows),
        "states": dict(Counter(row["status"] for row in rows)),
        "unmapped_experiments": sorted(identities - set(family_by_experiment)),
        "experiments": rows,
        "corrected_significance": None,
    }
