"""根据 raw L2 覆盖状态构造历史数据准入清单。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .coverage import CoverageRow


@dataclass(frozen=True)
class EligibilityRow:
    """一个股票日的历史数据准入结果。"""

    day: int
    ticker: str
    year: int
    month: str
    market: str
    batch: str
    primary_eligible: bool
    shanghai_research_eligible: bool
    lag_calibrated: bool
    exclusion_reasons: tuple[str, ...]


def classify_coverage(
    row: CoverageRow,
    *,
    start_year: int = 2021,
    end_year: int = 2025,
) -> EligibilityRow:
    """将覆盖行分类为深市主数据或沪市研究数据。"""
    reasons: list[str] = []
    if not start_year <= row.year <= end_year:
        reasons.append("outside_historical_window")
    if not row.preopen_file_present or not row.preopen_ticker_present:
        reasons.append("preopen_missing")
    if row.preopen_order_count == 0:
        reasons.append("preopen_order_missing")
    if not all(
        (
            row.order_file_present,
            row.trades_file_present,
            row.snapshot_file_present,
        )
    ):
        reasons.append("related_file_missing")
    if not all(
        (
            row.order_ticker_present,
            row.trades_ticker_present,
            row.snapshot_ticker_present,
        )
    ):
        reasons.append("related_ticker_missing")
    complete = not reasons
    primary_eligible = complete and row.market == "shenzhen"
    shanghai_research_eligible = complete and row.market == "shanghai"
    lag_calibrated = primary_eligible
    if shanghai_research_eligible:
        reasons.append("lag_not_calibrated")
    return EligibilityRow(
        day=row.day,
        ticker=row.ticker,
        year=row.year,
        month=row.month,
        market=row.market,
        batch=row.batch,
        primary_eligible=primary_eligible,
        shanghai_research_eligible=shanghai_research_eligible,
        lag_calibrated=lag_calibrated,
        exclusion_reasons=tuple(reasons),
    )


def summarize_eligibility(rows: Sequence[EligibilityRow]) -> dict:
    """汇总总量和市场分层准入结果。"""
    summary = {
        "total_rows": len(rows),
        "primary_eligible": sum(row.primary_eligible for row in rows),
        "shanghai_research_eligible": sum(row.shanghai_research_eligible for row in rows),
        "by_market": {},
        "by_year": {},
        "by_batch": {},
    }
    for dimension in ("market", "year", "batch"):
        groups: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "samples": 0,
                "primary_eligible": 0,
                "shanghai_research_eligible": 0,
                "excluded": 0,
            }
        )
        for row in rows:
            group = groups[str(getattr(row, dimension))]
            group["samples"] += 1
            group["primary_eligible"] += int(row.primary_eligible)
            group["shanghai_research_eligible"] += int(row.shanghai_research_eligible)
            group["excluded"] += int(
                not row.primary_eligible and not row.shanghai_research_eligible
            )
        summary[f"by_{dimension}"] = dict(sorted(groups.items()))
    return summary


def eligibility_row_dict(row: EligibilityRow) -> dict:
    """返回适合 JSON 或 CSV 的字段字典。"""
    result = asdict(row)
    result["exclusion_reasons"] = "|".join(row.exclusion_reasons)
    return result
