"""生成按月滚动的 3/1/1 多周期实验计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ticknet.nextday.splits import DateRange, WalkForwardSplit

MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def parse_month(value: str) -> date:
    """把 YYYY-MM 解析为该月首日。"""
    if not MONTH_PATTERN.fullmatch(value):
        raise ValueError(f"月份应使用 YYYY-MM 格式，收到 {value!r}")
    try:
        return date.fromisoformat(f"{value}-01")
    except ValueError as error:
        raise ValueError(f"无效月份：{value}") from error


def shift_month(month: date, offset: int) -> date:
    """将月首移动 offset 个月。"""
    absolute = month.year * 12 + month.month - 1 + offset
    year, zero_based_month = divmod(absolute, 12)
    return date(year, zero_based_month + 1, 1)


def month_end(month: date) -> date:
    return shift_month(month, 1) - timedelta(days=1)


@dataclass(frozen=True)
class RollingMonthWindow:
    """一个训练、验证和 OOS 月度窗口。"""

    fold_index: int
    target_horizon: int
    train: DateRange
    val: DateRange
    test: DateRange

    @property
    def fold_id(self) -> str:
        return f"fold-{self.fold_index:02d}-oos-{self.test.start:%Y%m}"

    def split(self) -> WalkForwardSplit:
        return WalkForwardSplit(train=self.train, val=self.val, test=self.test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "target_horizon": self.target_horizon,
            "train_start": self.train.start.isoformat(),
            "train_end": self.train.end.isoformat(),
            "val_start": self.val.start.isoformat(),
            "val_end": self.val.end.isoformat(),
            "oos_start": self.test.start.isoformat(),
            "oos_end": self.test.end.isoformat(),
            "purge_rule": "signal_entry_return_end_must_share_split",
        }


def build_rolling_windows(
    start_month: str,
    end_month: str,
    *,
    train_months: int = 3,
    validation_months: int = 1,
    oos_months: int = 1,
    step_months: int = 1,
    target_horizon: int = 5,
) -> list[RollingMonthWindow]:
    """生成包含首尾月份的滚动计划，窗口按 step_months 向后移动。"""
    start = parse_month(start_month)
    end = parse_month(end_month)
    if start > end:
        raise ValueError("start_month 不能晚于 end_month")
    dimensions = (train_months, validation_months, oos_months, step_months, target_horizon)
    if any(value < 1 for value in dimensions):
        raise ValueError("窗口月数、步长和 target_horizon 都应为正整数")

    total_months = train_months + validation_months + oos_months
    last_start = shift_month(end, -(total_months - 1))
    if last_start < start:
        raise ValueError(f"月份范围不足以容纳 {total_months} 个月的完整窗口")

    windows: list[RollingMonthWindow] = []
    cursor = start
    while cursor <= last_start:
        train_last = shift_month(cursor, train_months - 1)
        val_first = shift_month(cursor, train_months)
        val_last = shift_month(val_first, validation_months - 1)
        test_first = shift_month(val_first, validation_months)
        test_last = shift_month(test_first, oos_months - 1)
        windows.append(
            RollingMonthWindow(
                fold_index=len(windows),
                target_horizon=target_horizon,
                train=DateRange(cursor, month_end(train_last)),
                val=DateRange(val_first, month_end(val_last)),
                test=DateRange(test_first, month_end(test_last)),
            )
        )
        cursor = shift_month(cursor, step_months)
    return windows


def build_rolling_plan(
    start_month: str,
    end_month: str,
    *,
    train_months: int = 3,
    validation_months: int = 1,
    oos_months: int = 1,
    step_months: int = 1,
    target_horizon: int = 5,
) -> dict[str, Any]:
    windows = build_rolling_windows(
        start_month,
        end_month,
        train_months=train_months,
        validation_months=validation_months,
        oos_months=oos_months,
        step_months=step_months,
        target_horizon=target_horizon,
    )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "mode": "rolling_month_3_1_1",
        "start_month": start_month,
        "end_month": end_month,
        "train_months": train_months,
        "validation_months": validation_months,
        "oos_months": oos_months,
        "step_months": step_months,
        "target_horizon": target_horizon,
        "purge_rule": "signal_entry_return_end_must_share_split",
        "folds": [window.to_dict() for window in windows],
    }
    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    plan["plan_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def write_rolling_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(plan, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成按月滚动的 H5 训练/验证/OOS 计划")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=1)
    parser.add_argument("--oos-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    plan = build_rolling_plan(
        arguments.start_month,
        arguments.end_month,
        train_months=arguments.train_months,
        validation_months=arguments.validation_months,
        oos_months=arguments.oos_months,
        step_months=arguments.step_months,
        target_horizon=arguments.target_horizon,
    )
    output = arguments.output.expanduser().resolve()
    write_rolling_plan(output, plan)
    print(json.dumps({"output": str(output), **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
