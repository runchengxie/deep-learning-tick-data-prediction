"""Consume versioned public synthetic artifacts without importing their producer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import polars as pl

_NAMES = {"null_signal", "industry_confound", "delayed_revision", "partial_fill", "sequence_gap"}
_DTYPES = {"object", "str", "float64", "int64", "bool"}


def load_scenario(manifest: Path) -> tuple[dict[str, pl.DataFrame], dict[str, object]]:
    manifest = Path(manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "quant.assurance-scenario.v1"
        or payload.get("name") not in _NAMES
        or type(payload.get("seed")) is not int
        or payload["seed"] < 0
        or not isinstance(payload.get("truth"), dict)
        or not isinstance(payload.get("inventory"), list)
        or not payload["inventory"]
    ):
        raise ValueError("invalid assurance scenario manifest")
    frames = {}
    for item in payload["inventory"]:
        if not isinstance(item, dict):
            raise ValueError("invalid inventory entry")
        name, relative, dtypes = item.get("name"), item.get("path"), item.get("dtypes")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            or relative != f"{name}.csv"
            or name in frames
            or not isinstance(dtypes, dict)
            or not dtypes
            or not all(isinstance(col, str) and dtype in _DTYPES for col, dtype in dtypes.items())
        ):
            raise ValueError("invalid inventory identity or types")
        source = (manifest.parent / relative).resolve()
        if not source.is_relative_to(manifest.parent):
            raise ValueError("scenario path escapes root")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError("scenario input hash mismatch")
        types = {
            "object": pl.String,
            "str": pl.String,
            "float64": pl.Float64,
            "int64": pl.Int64,
            "bool": pl.Boolean,
        }
        frame = pl.read_csv(
            source, schema_overrides={col: types[dtype] for col, dtype in dtypes.items()}
        )
        if set(frame.columns) != set(dtypes) or len(frame) != item.get("rows"):
            raise ValueError("scenario shape mismatch")
        frames[name] = frame
    return frames, payload["truth"]
