"""用合成预测数据验证成本回测脚本逻辑。"""

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.evaluate_cost_adjusted import main as run_backtest


def _make_synthetic(path: Path) -> None:
    rng = np.random.RandomState(0)
    symbols = [f"{600000 + i:06d}" for i in range(100)]
    rows = []
    for day_offset in range(10):
        label_date = date(2024, 1, 2) + timedelta(days=day_offset)
        for symbol in symbols:
            score = float(rng.randn())
            target_return = float(0.01 * score + 0.001 * rng.randn())
            rows.append(
                {
                    "symbol": symbol,
                    "trading_date": label_date.isoformat(),
                    "label_date": label_date.isoformat(),
                    "target_return": target_return,
                    "score": score,
                    "prob_up": 0.4,
                    "prob_neutral": 0.3,
                    "prob_down": 0.3,
                }
            )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_synthetic_backtest_runs_and_returns_sane_values() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.parquet"
        _make_synthetic(path)
        result = run_backtest(
            [
                "--predictions",
                str(path),
                "--quantile",
                "0.1",
                "--cost-bps",
                "10",
                "--min-symbols-per-day",
                "50",
            ]
        )
        assert result is not None
        assert result["evaluated_dates"] == 10
        assert result["mean_turnover"] >= 0.0
        assert result["gross"]["annualized"] > 0.0
        assert result["net"]["annualized"] < result["gross"]["annualized"]
        assert result["net"]["mean_daily"] < result["gross"]["mean_daily"]


def test_zero_cost_backtest_net_equals_gross() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.parquet"
        _make_synthetic(path)
        result = run_backtest(
            [
                "--predictions",
                str(path),
                "--quantile",
                "0.1",
                "--cost-bps",
                "0",
                "--stamp-tax",
                "0",
                "--min-symbols-per-day",
                "50",
            ]
        )
        assert result is not None
        assert result["gross"]["mean_daily"] == pytest.approx(result["net"]["mean_daily"])


def test_insufficient_daily_cross_section_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.parquet"
        _make_synthetic(path)
        with pytest.raises(SystemExit, match="没有可评估的交易日"):
            run_backtest(
                [
                    "--predictions",
                    str(path),
                    "--quantile",
                    "0.1",
                    "--cost-bps",
                    "10",
                    "--min-symbols-per-day",
                    "500",
                ]
            )


def test_topk_cli_writes_auditable_artifacts(tmp_path) -> None:
    path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "topk"
    _make_synthetic(path)
    result = run_backtest(
        [
            "--predictions",
            str(path),
            "--top-k",
            "10",
            "--exit-buffer",
            "5",
            "--cost-bps",
            "0",
            "--stamp-tax-bps",
            "0",
            "--min-symbols-per-day",
            "50",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert result["mode"] == "topk_long_only"
    assert result["policy"]["top_k"] == 10
    assert result["net"] == result["gross"]
    assert pq.read_table(output_dir / "daily.parquet").num_rows == 10
    assert pq.read_table(output_dir / "holdings.parquet").num_rows == 100
