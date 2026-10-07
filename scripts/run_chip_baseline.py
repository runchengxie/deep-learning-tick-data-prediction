"""Run lagged age-layer chip HGB comparisons using published daily data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ticknet.nextday.chip_baseline import (
    FEATURE_SCHEMAS,
    experiment_fingerprint,
    load_daily_chip_bars,
    run_chip_comparison,
    safe_json,
)
from ticknet.nextday.chip_features import build_lagged_chip_features
from ticknet.nextday.minute_baseline import (
    MinuteExtractionReport,
    build_target_bundle,
    load_minute_baseline_config,
)
from ticknet.nextday.minute_materialization import load_materialized_minute_features
from ticknet.nextday.splits import parse_date
from ticknet.research.protocol import ResearchProtocol
from ticknet.research.registry import file_sha256


def _input_identity(
    config_path: Path,
    bars: list[Any],
    targets: list[Any],
    warmup_days: int,
    minute_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    # Hash one included bar at a time; a multi-year asset must not be copied
    # into a second full list of dictionaries and one enormous JSON string.
    digest = hashlib.sha256()
    for bar in sorted(bars, key=lambda item: (item.symbol, item.trading_date)):
        digest.update(json.dumps(asdict(bar), sort_keys=True, default=str).encode())
        digest.update(b"\n")
    identity = {
        "daily_content_fingerprint": digest.hexdigest(),
        "config_sha256": file_sha256(config_path),
        "warmup_days": warmup_days,
        "feature_contract": "lag1_chip_age_summary_v1",
        "feature_schemas": FEATURE_SCHEMAS,
        "minute": minute_receipt,
        "targets": [
            (
                target.symbol,
                target.trading_date,
                target.label_date,
                target.return_end_date,
                target.label,
                target.target_return,
                target.portfolio_return,
                target.benchmark_return,
                target.can_buy,
                target.can_sell,
                target.in_universe,
            )
            for target in targets
        ],
    }
    return identity


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Existing formal minute target YAML"
    )
    parser.add_argument(
        "--daily-bars", type=Path, required=True, help="Published chip daily Parquet"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-days", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--minute-features", type=Path, help="Optional verified minute manifest")
    parser.add_argument(
        "--evaluate-test", action="store_true", help="Evaluate seen pre-2026 OOS only"
    )
    args = parser.parse_args(argv)
    config = load_minute_baseline_config(args.config)
    if not config.formal:
        raise ValueError("chip baseline requires the formal next-open-to-following-open contract")
    if (
        max(parse_date(config.end_date), config.date_split().test.end)
        >= ResearchProtocol().locked_begin
    ):
        raise ValueError("chip configuration reaches the locked period")
    bars = load_daily_chip_bars(args.daily_bars, end_date=parse_date(config.end_date))
    bundle = build_target_bundle(config)
    candidates = [target for target in bundle.targets if target.in_universe]
    keys = [(target.trading_date, target.symbol) for target in candidates]
    features = build_lagged_chip_features(bars, keys, warmup_days=args.warmup_days)
    minute = None
    minute_receipt = None
    if args.minute_features:
        loaded = load_materialized_minute_features(
            config, candidates, args.minute_features, MinuteExtractionReport()
        )
        minute = {(item.trading_date, item.symbol): item.features for item in loaded.samples}
        minute_receipt = loaded.summary()
    identity = _input_identity(args.config, bars, bundle.targets, args.warmup_days, minute_receipt)
    fingerprint = experiment_fingerprint(identity)
    result = run_chip_comparison(
        bundle.targets,
        features,
        config.date_split(),
        args.output_dir,
        fingerprint=fingerprint,
        top_n=config.top_n,
        top_k=args.top_k,
        min_symbols_per_day=config.min_symbols_per_day,
        seed=config.seed,
        max_iter=args.max_iter,
        min_samples_leaf=args.min_samples_leaf,
        evaluate_test=args.evaluate_test,
        minute_features=minute,
    )
    result["provenance"] = {key: value for key, value in identity.items() if key != "targets"}
    result["settings"] = {
        "early_stopping": False,
        "max_iter": args.max_iter,
        "min_samples_leaf": args.min_samples_leaf,
        "top_k": args.top_k,
    }
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
    )
    result["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    source_files = [
        Path(__file__).resolve(),
        repository / "src/ticknet/nextday/chip_features.py",
        repository / "src/ticknet/nextday/chip_baseline.py",
    ]
    result["source_sha256"] = {
        path.relative_to(repository).as_posix(): file_sha256(path) for path in source_files
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(safe_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(safe_json(result), indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
