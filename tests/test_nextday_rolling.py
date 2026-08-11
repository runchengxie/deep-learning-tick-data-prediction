"""H5 月度 3/1/1 rolling 计划测试。"""

from datetime import date

import pytest

from ticknet.nextday.rolling import build_rolling_plan, build_rolling_windows, parse_month


def test_build_rolling_windows_generates_3_1_1_months() -> None:
    windows = build_rolling_windows("2021-01", "2021-07", target_horizon=5)

    assert len(windows) == 3
    first = windows[0]
    assert first.fold_id == "fold-00-oos-202105"
    assert first.train.start == date(2021, 1, 1)
    assert first.train.end == date(2021, 3, 31)
    assert first.val.start == date(2021, 4, 1)
    assert first.val.end == date(2021, 4, 30)
    assert first.test.start == date(2021, 5, 1)
    assert first.test.end == date(2021, 5, 31)
    assert windows[-1].test.start == date(2021, 7, 1)


def test_2021_to_2024_plan_keeps_2025_out_of_rolling_oos() -> None:
    plan = build_rolling_plan("2021-01", "2024-12", target_horizon=5)

    assert len(plan["folds"]) == 44
    assert plan["folds"][-1]["oos_end"] == "2024-12-31"
    assert plan["purge_rule"] == "signal_entry_return_end_must_share_split"
    assert len(plan["plan_fingerprint"]) == 64


@pytest.mark.parametrize("value", ["202101", "2021-1", "bad", "2021-13"])
def test_parse_month_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match=r"月份|无效"):
        parse_month(value)


def test_rolling_plan_requires_enough_months() -> None:
    with pytest.raises(ValueError, match="不足"):
        build_rolling_windows("2021-01", "2021-04")
