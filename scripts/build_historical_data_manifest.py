"""生成 2021 至 2025 raw L2 历史数据准入清单。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

from ticknet.simulator.coverage import scan_preopen_coverage
from ticknet.simulator.eligibility import (
    EligibilityRow,
    classify_coverage,
    eligibility_row_dict,
    summarize_eligibility,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--limit-days", type=int, default=None)
    return parser


def _write_csv(path: Path, rows: list[dict]) -> None:
    field_names = [field.name for field in fields(EligibilityRow)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    coverage_rows = scan_preopen_coverage(
        arguments.raw_root,
        limit_days=arguments.limit_days,
    )
    eligibility_rows = [
        classify_coverage(
            row,
            start_year=arguments.start_year,
            end_year=arguments.end_year,
        )
        for row in coverage_rows
    ]
    rows = [eligibility_row_dict(row) for row in eligibility_rows]
    payload = {
        "summary": summarize_eligibility(eligibility_rows),
        "rows": rows,
    }
    arguments.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(arguments.csv_output, rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
