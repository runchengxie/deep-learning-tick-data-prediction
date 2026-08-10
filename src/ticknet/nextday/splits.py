"""按完整交易日划分次日预测数据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"日期应使用 YYYY-MM-DD 格式，收到 {value!r}") from error


@dataclass(frozen=True)
class DateRange:
    """包含首尾日期的时间区间。"""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"日期区间起点 {self.start} 晚于终点 {self.end}")

    @classmethod
    def from_strings(cls, start: str, end: str) -> DateRange:
        return cls(parse_date(start), parse_date(end))

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end


@dataclass(frozen=True)
class WalkForwardSplit:
    """训练、验证和测试的非重叠日期区间。"""

    train: DateRange
    val: DateRange
    test: DateRange

    def __post_init__(self) -> None:
        if not self.train.end < self.val.start:
            raise ValueError("训练区间必须早于验证区间且不能重叠")
        if not self.val.end < self.test.start:
            raise ValueError("验证区间必须早于测试区间且不能重叠")

    @classmethod
    def from_strings(
        cls,
        *,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
    ) -> WalkForwardSplit:
        return cls(
            train=DateRange.from_strings(train_start, train_end),
            val=DateRange.from_strings(val_start, val_end),
            test=DateRange.from_strings(test_start, test_end),
        )

    def range_for(self, split: str) -> DateRange:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split 应为 train、val 或 test，收到 {split}")
        return getattr(self, split)

    def assign(
        self,
        trading_date: date,
        label_date: date,
        return_end_date: date | None = None,
    ) -> str | None:
        """仅在输入日、进入日和收益结束日同属一个区间时分配样本。

        收益周期跨越切分边界的样本会被清除。旧的一日标签没有单独的
        ``return_end_date``，此时收益结束日等于 ``label_date``。
        """
        if label_date <= trading_date:
            raise ValueError("label_date 必须晚于 trading_date")
        end_date = label_date if return_end_date is None else return_end_date
        if end_date < label_date:
            raise ValueError("return_end_date 不能早于 label_date")
        for name in ("train", "val", "test"):
            interval = self.range_for(name)
            if (
                interval.contains(trading_date)
                and interval.contains(label_date)
                and interval.contains(end_date)
            ):
                return name
        return None
