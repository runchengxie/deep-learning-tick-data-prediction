"""生成 raw L2 盘前订单覆盖清单。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

from ticknet.simulator.coverage import (
    CoverageRow,
    coverage_row_dict,
    load_or_build_coverage_index,
    scan_preopen_coverage,
    summarize_coverage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="覆盖索引 JSON 路径；已存在且源文件未变化时复用索引",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="强制重新扫描 index-path 覆盖的交易日",
    )
    return parser


def _write_csv(path: Path, rows: list[dict]) -> None:
    field_names = [field.name for field in fields(CoverageRow)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.index_path is None:
        coverage_rows = scan_preopen_coverage(arguments.raw_root, limit_days=arguments.limit_days)
    else:
        coverage_rows = load_or_build_coverage_index(
            arguments.raw_root,
            arguments.index_path,
            limit_days=arguments.limit_days,
            force=arguments.refresh_index,
        )
    rows = [coverage_row_dict(row) for row in coverage_rows]
    payload = {"summary": summarize_coverage(coverage_rows), "rows": rows}
    arguments.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(arguments.csv_output, rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
