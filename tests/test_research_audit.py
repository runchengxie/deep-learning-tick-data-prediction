"""预测审计模块测试：合成数据验证诊断逻辑。"""

from datetime import date, timedelta

import numpy as np

from ticknet.research.audit import PredictionTable, audit_predictions


def _make_table(
    *,
    days: int = 20,
    symbols_per_day: int = 100,
    signal_strength: float = 0.3,
    seed: int = 0,
    extreme_outlier: bool = False,
) -> PredictionTable:
    rng = np.random.RandomState(seed)
    symbols: list[str] = []
    label_dates: list[date] = []
    returns: list[float] = []
    scores: list[float] = []
    start = date(2024, 1, 2)
    for day in range(days):
        label_date = start + timedelta(days=day)
        for _ in range(symbols_per_day):
            symbol = f"{600000 + rng.randint(0, 100000):06d}"
            score = float(rng.randn())
            return_value = signal_strength * score + 0.02 * rng.randn()
            symbols.append(symbol)
            label_dates.append(label_date)
            returns.append(return_value)
            scores.append(score)
    if extreme_outlier:
        label_dates[0] = start
        returns[0] = 0.5
        scores[0] = 3.0
    return PredictionTable(
        symbols=np.asarray(symbols),
        label_dates=np.asarray(label_dates),
        target_returns=np.asarray(returns, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
    )


def test_audit_strong_signal_has_high_ic_and_monotonic_deciles():
    table = _make_table(signal_strength=0.5, seed=1)
    report = audit_predictions(
        table,
        min_symbols_per_day=50,
        portfolio_quantile=0.1,
    )
    assert report.daily_ic_mean > 0.3
    assert report.decile_monotonicity > 0.8
    assert report.spread_mean > 0
    assert report.positive_days_ratio > 0.8


def test_audit_extreme_outlier_triggers_tail_anomaly():
    table = _make_table(signal_strength=0.0, seed=2, extreme_outlier=True)
    report = audit_predictions(
        table,
        min_symbols_per_day=50,
        portfolio_quantile=0.1,
    )
    types = {anomaly["type"] for anomaly in report.anomalies}
    assert "tail_return_concentration" in types
    assert report.top_1_day_contribution > 0.05


def test_audit_weak_signal_low_monotonicity():
    table = _make_table(signal_strength=0.0, seed=3)
    report = audit_predictions(
        table,
        min_symbols_per_day=50,
        portfolio_quantile=0.1,
    )
    assert report.daily_ic_mean < 0.1
    assert abs(report.decile_monotonicity) < 0.6


def test_audit_report_structure_and_monthly_ic():
    table = _make_table(signal_strength=0.4, seed=4, days=25)
    report = audit_predictions(table, min_symbols_per_day=50)
    data = report.to_dict()
    assert "monthly_ic" in data
    assert data["positive_month_ratio"] >= 0
    assert len(data["decile_returns"]) == 10
    assert data["daily_ic_ir"] > 0


def test_audit_from_parquet_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = _make_table(signal_strength=0.3, seed=5)
    path = tmp_path / "predictions.parquet"
    pq.write_table(
        pa.table(
            {
                "symbol": table.symbols,
                "trading_date": [str(value) for value in table.label_dates],
                "label_date": [str(value) for value in table.label_dates],
                "target_return": table.target_returns,
                "score": table.scores,
                "prob_up": np.full(len(table.scores), 0.4),
                "prob_neutral": np.full(len(table.scores), 0.3),
                "prob_down": np.full(len(table.scores), 0.3),
            }
        ),
        path,
    )
    loaded = PredictionTable.from_parquet(path)
    assert loaded.symbols.shape == table.symbols.shape
    report = audit_predictions(loaded, min_symbols_per_day=50)
    assert report.daily_ic_mean > 0.2


def test_audit_excludes_dynamic_universe_status_rows():
    table = PredictionTable(
        symbols=np.asarray(["A", "B", "OLD"]),
        label_dates=np.asarray([date(2025, 1, 3)] * 3),
        target_returns=np.asarray([0.01, 0.02, 99.0]),
        scores=np.asarray([1.0, 2.0, 999.0]),
        in_universe=np.asarray([True, True, False]),
    )
    assert table.group_by_date() == {date(2025, 1, 3): [0, 1]}
