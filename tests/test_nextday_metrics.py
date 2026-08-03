"""次日横截面评估指标测试。"""

from datetime import date

import numpy as np
import pytest

from deeplob.nextday.metrics import evaluate_predictions


def test_perfect_daily_ranking_has_unit_rank_ic():
    labels = np.array([0, 1, 1, 2] * 2)
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.3, 0.6, 0.1],
            [0.1, 0.6, 0.3],
            [0.0, 0.1, 0.9],
        ]
        * 2
    )
    returns = np.array([-0.02, -0.01, 0.01, 0.02] * 2)
    dates = [date(2024, 1, 3)] * 4 + [date(2024, 1, 4)] * 4
    metrics = evaluate_predictions(
        labels,
        probabilities,
        returns,
        dates,
        min_symbols_per_day=4,
        portfolio_quantile=0.25,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["daily_rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["daily_long_short_return_mean"] == pytest.approx(0.04)
    assert metrics["evaluated_dates"] == 2


def test_small_dates_are_excluded_from_cross_sectional_metrics():
    metrics = evaluate_predictions(
        np.array([0, 2]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]),
        np.array([-0.01, 0.01]),
        [date(2024, 1, 3)] * 2,
        min_symbols_per_day=3,
    )
    assert np.isnan(metrics["daily_rank_ic_mean"])
    assert metrics["evaluated_dates"] == 0


def test_continuous_score_head_controls_cross_sectional_ranking():
    metrics = evaluate_predictions(
        np.array([1, 1, 1, 1]),
        np.full((4, 3), 1 / 3),
        np.array([-0.02, -0.01, 0.01, 0.02]),
        [date(2024, 1, 3)] * 4,
        scores=np.array([-2.0, -1.0, 1.0, 2.0]),
        min_symbols_per_day=4,
        portfolio_quantile=0.25,
    )
    assert metrics["daily_rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["daily_long_short_return_mean"] == pytest.approx(0.04)


def test_prediction_shapes_are_validated():
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_predictions(
            np.array([0, 1]),
            np.zeros((2, 2)),
            np.zeros(2),
            [date(2024, 1, 3)] * 2,
        )
