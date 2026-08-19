"""事件流信号半衰期、排名平滑和 H5 错峰持有诊断。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import ndcg_score

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.nextday.horizon_labels import HorizonTarget, load_horizon_sidecar
from ticknet.research.portfolio import (
    CostModel,
    PortfolioEvaluation,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
)


@dataclass(frozen=True)
class SignalRow:
    """一个分区内的股票日分数。"""

    partition: str
    trading_date: date
    symbol: str
    score: float


@dataclass(frozen=True)
class PolicyCandidate:
    """验证期选择后可原样迁移到下一滚动折的交易规则。"""

    ema_alpha: float
    min_expected_return: float | None
    min_replacement_gap: float

    @property
    def name(self) -> str:
        threshold = (
            "none" if self.min_expected_return is None else f"{self.min_expected_return:.4f}"
        )
        return f"ema-{self.ema_alpha:.2f}_entry-{threshold}_gap-{self.min_replacement_gap:.4f}"


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    return date.fromisoformat(text)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return value


def _require_complete_manifest(root: Path, *, mode: str) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("mode") != mode:
        raise ValueError(f"产物未完成或模式不匹配：{root}")
    return manifest


def load_materialized_signal_rows(
    keys_root: str | Path,
    scores_root: str | Path,
) -> tuple[list[SignalRow], dict[str, Any]]:
    """核对物化身份侧车与预测分片，并按行号恢复股票级分数。"""
    keys_dir = Path(keys_root).expanduser().resolve()
    scores_dir = Path(scores_root).expanduser().resolve()
    keys_manifest = _require_complete_manifest(
        keys_dir,
        mode="eventstream_materialized_sample_keys",
    )
    scores_manifest = _require_complete_manifest(
        scores_dir,
        mode="eventstream_materialized_day_predictions",
    )
    keys_contract = keys_manifest["contract"]
    scores_contract = scores_manifest["contract"]
    compared = ("materialized_dataset_fingerprint", "source_dataset_fingerprint", "locked_start")
    if any(keys_contract.get(key) != scores_contract.get(key) for key in compared):
        raise ValueError("物化身份侧车与预测分数的来源合同不同")

    key_info = keys_manifest["artifact"]
    keys_path = keys_dir / str(key_info["path"])
    if (
        not keys_path.is_file()
        or keys_path.stat().st_size != int(key_info["bytes"])
        or file_sha256(keys_path) != key_info["sha256"]
    ):
        raise ValueError("物化身份侧车缺失或内容漂移")
    key_table = pq.read_table(
        keys_path,
        columns=["partition", "row_index", "trading_day", "symbol"],
    )
    keys: dict[tuple[str, int], tuple[int, str]] = {}
    for row in key_table.to_pylist():
        key = (str(row["partition"]), int(row["row_index"]))
        if key in keys:
            raise ValueError(f"物化身份侧车行号重复：{key}")
        keys[key] = (int(row["trading_day"]), str(row["symbol"]))

    signals: list[SignalRow] = []
    for artifact in scores_manifest["artifacts"]:
        partition = str(artifact["partition"])
        path = scores_dir / str(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or file_sha256(path) != artifact["sha256"]
        ):
            raise ValueError(f"物化预测分片缺失或内容漂移：{path}")
        table = pq.read_table(path, columns=["row_index", "trading_day", "score"])
        for row in table.to_pylist():
            key = (partition, int(row["row_index"]))
            identity = keys.get(key)
            if identity is None or identity[0] != int(row["trading_day"]):
                raise ValueError(f"物化预测无法与股票身份逐行对齐：{key}")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"物化预测包含非有限分数：{key}")
            signals.append(
                SignalRow(
                    partition=partition,
                    trading_date=_as_date(identity[0]),
                    symbol=identity[1],
                    score=score,
                )
            )
    if len(signals) != len(keys) or len(signals) != int(scores_manifest["totals"]["rows"]):
        raise ValueError("物化预测、股票身份和 manifest 行数不同")
    return signals, {
        "key_dataset_fingerprint": keys_manifest["dataset_fingerprint"],
        "score_dataset_fingerprint": scores_manifest["dataset_fingerprint"],
        "source_dataset_fingerprint": scores_contract["source_dataset_fingerprint"],
        "checkpoint_sha256": scores_contract["checkpoint_sha256"],
    }


def load_joint_signal_rows(paths: Mapping[str, str | Path]) -> list[SignalRow]:
    """读取联合模型 validation 与 test 预测，保留股票日和连续分数。"""
    signals: list[SignalRow] = []
    seen: set[tuple[str, date, str]] = set()
    aliases = {"val": "validation", "validation": "validation", "test": "oos", "oos": "oos"}
    for raw_partition, raw_path in paths.items():
        if raw_partition not in aliases:
            raise ValueError(f"联合预测分区无效：{raw_partition}")
        partition = aliases[raw_partition]
        path = Path(raw_path).expanduser().resolve()
        table = pq.read_table(path)
        required = {"trading_day", "symbol", "score"}
        if not required.issubset(table.column_names):
            raise ValueError(f"联合预测缺少字段：{sorted(required - set(table.column_names))}")
        for row in table.select(sorted(required)).to_pylist():
            key = (partition, _as_date(row["trading_day"]), str(row["symbol"]))
            if key in seen:
                raise ValueError(f"联合预测股票日重复：{key}")
            seen.add(key)
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"联合预测包含非有限分数：{key}")
            signals.append(SignalRow(partition, key[1], key[2], score))
    if {row.partition for row in signals} != {"validation", "oos"}:
        raise ValueError("联合预测必须同时包含 validation 和 oos")
    return sorted(signals, key=lambda row: (row.partition, row.trading_date, row.symbol))


def split_signal_rows(rows: list[SignalRow]) -> tuple[list[SignalRow], list[SignalRow]]:
    validation = [row for row in rows if row.partition == "validation"]
    oos = [row for row in rows if row.partition == "oos"]
    if not validation or not oos:
        raise ValueError("信号必须同时包含 validation 和 oos")
    if max(row.trading_date for row in validation) >= min(row.trading_date for row in oos):
        raise ValueError("validation 与 oos 日期范围重叠或倒置")
    return validation, oos


def load_horizon_maps(
    manifest_path: str | Path,
    *,
    source_dataset_fingerprint: str,
    horizons: tuple[int, ...] = tuple(range(1, 11)),
    required_keys: set[tuple[str, date]] | None = None,
) -> tuple[dict[int, dict[tuple[str, date], HorizonTarget]], str]:
    """读取并校验多周期标签，返回按股票日索引的目标。"""
    result: dict[int, dict[tuple[str, date], HorizonTarget]] = {}
    fingerprint = ""
    for index, horizon in enumerate(horizons):
        loaded = load_horizon_sidecar(
            manifest_path,
            horizon=horizon,
            source_dataset_fingerprint=source_dataset_fingerprint,
            verify_checksum=index == 0,
        )
        if fingerprint and loaded.sidecar_fingerprint != fingerprint:
            raise ValueError("多周期标签读取期间 fingerprint 发生变化")
        fingerprint = loaded.sidecar_fingerprint
        result[horizon] = (
            loaded.records
            if required_keys is None
            else {key: target for key, target in loaded.records.items() if key in required_keys}
        )
    return result, fingerprint


def _rank_correlation(scores: np.ndarray, returns: np.ndarray) -> float:
    if scores.size < 2 or np.std(scores) == 0 or np.std(returns) == 0:
        return math.nan
    score_order = np.argsort(scores, kind="mergesort")
    return_order = np.argsort(returns, kind="mergesort")
    score_ranks = np.empty(scores.size, dtype=np.float64)
    return_ranks = np.empty(returns.size, dtype=np.float64)
    score_ranks[score_order] = np.arange(scores.size)
    return_ranks[return_order] = np.arange(returns.size)
    return float(np.corrcoef(score_ranks, return_ranks)[0, 1])


def _group_signals(rows: list[SignalRow]) -> dict[date, list[SignalRow]]:
    grouped: dict[date, list[SignalRow]] = defaultdict(list)
    for row in rows:
        grouped[row.trading_date].append(row)
    return grouped


def half_life_diagnostics(
    signals: list[SignalRow],
    horizon_maps: dict[int, dict[tuple[str, date], HorizonTarget]],
    *,
    top_k: int = 100,
    min_symbols_per_day: int = 100,
) -> dict[str, Any]:
    """计算 H1 至 H10 的 IC、极端组收益差和非重叠累计收益。"""
    grouped = _group_signals(signals)
    output: dict[str, Any] = {}
    for horizon, targets in sorted(horizon_maps.items()):
        daily: list[dict[str, Any]] = []
        for trading_date, day_rows in sorted(grouped.items()):
            aligned = [(row, targets.get((row.symbol, trading_date))) for row in day_rows]
            aligned = [(row, target) for row, target in aligned if target is not None]
            if len(aligned) < min_symbols_per_day:
                continue
            ranked = sorted(aligned, key=lambda pair: (-pair[0].score, pair[0].symbol))
            scores = np.asarray([row.score for row, _target in aligned], dtype=np.float64)
            returns = np.asarray(
                [target.target_return for _row, target in aligned], dtype=np.float64
            )
            tail = max(1, len(ranked) // 10)
            top = ranked[:tail]
            bottom = ranked[-tail:]
            selected = ranked[: min(top_k, len(ranked))]
            selected_targets = [target for _row, target in selected]
            labels = np.asarray([target.label for _row, target in aligned], dtype=np.int64)
            relevance = labels.astype(np.float64)
            ndcg = float(ndcg_score(relevance[None, :], scores[None, :], k=top_k))
            daily.append(
                {
                    "trading_date": trading_date.isoformat(),
                    "return_end_date": selected_targets[0].return_end_date.isoformat(),
                    "symbols": len(aligned),
                    "rank_ic": _rank_correlation(scores, returns),
                    "extreme_spread": float(
                        np.mean([target.target_return for _row, target in top])
                        - np.mean([target.target_return for _row, target in bottom])
                    ),
                    "top_k_raw_return": float(
                        np.mean([target.raw_return for target in selected_targets])
                    ),
                    "top_k_benchmark_return": float(
                        np.mean([target.benchmark_return for target in selected_targets])
                    ),
                    "top_k_active_return": float(
                        np.mean([target.target_return for target in selected_targets])
                    ),
                    "ndcg_at_k": ndcg,
                    "precision_at_k": float(
                        np.mean([target.label == 2 for target in selected_targets])
                    ),
                }
            )
        if not daily:
            raise ValueError(f"horizon={horizon} 没有满足横截面门槛的日期")
        finite_ics = [float(row["rank_ic"]) for row in daily if math.isfinite(row["rank_ic"])]
        anchors: list[dict[str, float | int]] = []
        for anchor in range(horizon):
            rows = daily[anchor::horizon]
            if not rows:
                continue
            anchors.append(
                {
                    "anchor": anchor,
                    "cohorts": len(rows),
                    "raw_cumulative": float(
                        np.prod([1.0 + float(row["top_k_raw_return"]) for row in rows]) - 1.0
                    ),
                    "benchmark_cumulative": float(
                        np.prod([1.0 + float(row["top_k_benchmark_return"]) for row in rows]) - 1.0
                    ),
                    "active_cumulative": float(
                        np.prod([1.0 + float(row["top_k_active_return"]) for row in rows]) - 1.0
                    ),
                }
            )
        output[str(horizon)] = {
            "evaluated_dates": len(daily),
            "mean_rank_ic": float(np.mean(finite_ics)),
            "rank_ic_ir": (
                float(np.mean(finite_ics) / np.std(finite_ics, ddof=1))
                if len(finite_ics) > 1 and np.std(finite_ics, ddof=1) > 0
                else math.nan
            ),
            "positive_rank_ic_ratio": float(np.mean(np.asarray(finite_ics) > 0)),
            "mean_extreme_spread": float(np.mean([row["extreme_spread"] for row in daily])),
            "mean_top_k_active_return": float(
                np.mean([row["top_k_active_return"] for row in daily])
            ),
            "mean_ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in daily])),
            "mean_precision_at_k": float(np.mean([row["precision_at_k"] for row in daily])),
            "non_overlapping_anchors": anchors,
            "daily": daily,
        }
    return output


def percentile_ranks(
    signals: list[SignalRow],
    *,
    ema_alpha: float,
) -> dict[tuple[date, str], float]:
    """先做每日横截面百分位排名，再按股票做相邻交易日 EMA。"""
    if not 0 < ema_alpha <= 1:
        raise ValueError("ema_alpha 应在 (0, 1] 内")
    grouped = _group_signals(signals)
    output: dict[tuple[date, str], float] = {}
    previous: dict[str, tuple[int, float]] = {}
    for day_index, (trading_date, rows) in enumerate(sorted(grouped.items())):
        ordered = sorted(rows, key=lambda row: (row.score, row.symbol))
        denominator = max(1, len(ordered) - 1)
        for rank, row in enumerate(ordered):
            current = rank / denominator
            old = previous.get(row.symbol)
            adjusted = (
                ema_alpha * current + (1.0 - ema_alpha) * old[1]
                if old is not None and old[0] == day_index - 1
                else current
            )
            output[(trading_date, row.symbol)] = adjusted
            previous[row.symbol] = (day_index, adjusted)
    return output


def calibrated_scores(
    validation_signals: list[SignalRow],
    oos_signals: list[SignalRow],
    h5_targets: dict[tuple[str, date], HorizonTarget],
    *,
    ema_alpha: float,
) -> tuple[dict[tuple[date, str], float], dict[tuple[date, str], float], dict[str, Any]]:
    """仅用 validation 把排名映射为 H5 预期超额收益，再应用到 OOS。"""
    validation_ranks = percentile_ranks(validation_signals, ema_alpha=ema_alpha)
    oos_ranks = percentile_ranks(oos_signals, ema_alpha=ema_alpha)
    fit_rows = [
        (rank, h5_targets.get((symbol, trading_date)))
        for (trading_date, symbol), rank in validation_ranks.items()
    ]
    fit_rows = [(rank, target) for rank, target in fit_rows if target is not None]
    if len(fit_rows) < 100:
        raise ValueError("validation 可校准的 H5 样本不足 100")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    x = np.asarray([rank for rank, _target in fit_rows], dtype=np.float64)
    y = np.asarray([target.target_return for _rank, target in fit_rows], dtype=np.float64)
    model.fit(x, y)

    def transform(values: dict[tuple[date, str], float]) -> dict[tuple[date, str], float]:
        keys = list(values)
        predicted = model.predict(np.asarray([values[key] for key in keys], dtype=np.float64))
        return dict(zip(keys, (float(value) for value in predicted), strict=True))

    validation_expected = transform(validation_ranks)
    oos_expected = transform(oos_ranks)
    return (
        validation_expected,
        oos_expected,
        {
            "ema_alpha": ema_alpha,
            "validation_samples": len(fit_rows),
            "rank_min": float(np.min(x)),
            "rank_max": float(np.max(x)),
            "target_mean": float(np.mean(y)),
            "calibrated_min": float(np.min(list(validation_expected.values()))),
            "calibrated_max": float(np.max(list(validation_expected.values()))),
        },
    )


def make_portfolio_predictions(
    signals: list[SignalRow],
    scores: dict[tuple[date, str], float],
    targets: dict[tuple[str, date], HorizonTarget],
) -> list[PortfolioPrediction]:
    predictions: list[PortfolioPrediction] = []
    for row in signals:
        key = (row.trading_date, row.symbol)
        target = targets.get((row.symbol, row.trading_date))
        if key not in scores or target is None:
            continue
        predictions.append(
            PortfolioPrediction(
                symbol=row.symbol,
                trading_date=row.trading_date,
                label_date=target.return_end_date,
                score=scores[key],
                target_return=target.target_return,
            )
        )
    if not predictions:
        raise ValueError("信号与收益标签没有交集")
    return predictions


def evaluate_policy(
    signals: list[SignalRow],
    expected_scores: dict[tuple[date, str], float],
    h1_targets: dict[tuple[str, date], HorizonTarget],
    candidate: PolicyCandidate,
    *,
    top_k: int = 100,
    min_symbols_per_day: int = 100,
    per_side_bps: float = 10.0,
    sell_stamp_tax_bps: float = 5.0,
) -> PortfolioEvaluation:
    return evaluate_topk_portfolio(
        make_portfolio_predictions(signals, expected_scores, h1_targets),
        policy=PortfolioPolicy(
            top_k=top_k,
            min_score_gap=candidate.min_replacement_gap,
            min_position_score=candidate.min_expected_return,
            allow_cash=candidate.min_expected_return is not None,
            min_symbols_per_day=min_symbols_per_day,
        ),
        cost_model=CostModel(
            per_side_bps=per_side_bps,
            sell_stamp_tax_bps=sell_stamp_tax_bps,
        ),
    )


def staggered_h5_evaluation(
    signals: list[SignalRow],
    expected_scores: dict[tuple[date, str], float],
    h5_targets: dict[tuple[str, date], HorizonTarget],
    *,
    top_k: int = 100,
    min_expected_return: float | None = None,
    min_symbols_per_day: int = 100,
    per_side_bps: float = 10.0,
    sell_stamp_tax_bps: float = 5.0,
) -> dict[str, Any]:
    """每天建立一个 1/5 资金的 H5 cohort，并在第五个交易日收回。"""
    grouped = _group_signals(signals)
    rows: list[dict[str, Any]] = []
    buy_rate = per_side_bps / 10_000.0
    sell_rate = (per_side_bps + sell_stamp_tax_bps) / 10_000.0
    for cohort_index, (trading_date, day_signals) in enumerate(sorted(grouped.items())):
        aligned = [
            (
                row,
                expected_scores.get((trading_date, row.symbol)),
                h5_targets.get((row.symbol, trading_date)),
            )
            for row in day_signals
        ]
        aligned = [
            (row, score, target)
            for row, score, target in aligned
            if score is not None and target is not None
        ]
        if len(aligned) < min_symbols_per_day:
            continue
        ranked = sorted(aligned, key=lambda item: (-float(item[1]), item[0].symbol))
        if min_expected_return is not None:
            ranked = [item for item in ranked if float(item[1]) >= min_expected_return]
        selected = ranked[:top_k]
        exposure = len(selected) / top_k / 5.0
        raw_return = (
            exposure * float(np.mean([item[2].raw_return for item in selected]))
            if selected
            else 0.0
        )
        benchmark_return = (
            exposure * float(np.mean([item[2].benchmark_return for item in selected]))
            if selected
            else 0.0
        )
        active_return = raw_return - benchmark_return
        transaction_cost = exposure * (buy_rate + sell_rate)
        return_end_date = (
            selected[0][2].return_end_date if selected else aligned[0][2].return_end_date
        )
        rows.append(
            {
                "cohort": cohort_index,
                "sleeve": cohort_index % 5,
                "trading_date": trading_date.isoformat(),
                "return_end_date": return_end_date.isoformat(),
                "positions": len(selected),
                "gross_exposure": exposure,
                "cash_weight": 0.2 - exposure,
                "raw_return_contribution": raw_return,
                "benchmark_return_contribution": benchmark_return,
                "active_return_contribution": active_return,
                "transaction_cost": transaction_cost,
                "net_return_contribution": raw_return - transaction_cost,
                "net_active_return_contribution": active_return - transaction_cost,
                "symbols": [item[0].symbol for item in selected],
            }
        )
    if not rows:
        raise ValueError("H5 错峰持有没有可评估 cohort")
    return {
        "contract": "daily_one_fifth_capital_next_open_to_h5_close",
        "evaluated_cohorts": len(rows),
        "date_range": [rows[0]["trading_date"], rows[-1]["trading_date"]],
        "mean_positions": float(np.mean([row["positions"] for row in rows])),
        "mean_gross_exposure": float(np.mean([row["gross_exposure"] for row in rows]) * 5),
        "mean_one_way_turnover": float(np.mean([row["gross_exposure"] for row in rows])),
        "mean_active_return": float(np.mean([row["active_return_contribution"] for row in rows])),
        "mean_net_active_return": float(
            np.mean([row["net_active_return_contribution"] for row in rows])
        ),
        "cumulative_raw_return": float(
            np.prod([1.0 + row["raw_return_contribution"] for row in rows]) - 1.0
        ),
        "cumulative_benchmark_return": float(
            np.prod([1.0 + row["benchmark_return_contribution"] for row in rows]) - 1.0
        ),
        "cumulative_net_return": float(
            np.prod([1.0 + row["net_return_contribution"] for row in rows]) - 1.0
        ),
        "cumulative_net_active_return": float(
            np.prod([1.0 + row["net_active_return_contribution"] for row in rows]) - 1.0
        ),
        "sleeves": [
            {
                "sleeve": sleeve,
                "cohorts": sum(row["sleeve"] == sleeve for row in rows),
                "cumulative_net_active_return": float(
                    np.prod(
                        [
                            1.0 + 5.0 * row["net_active_return_contribution"]
                            for row in rows
                            if row["sleeve"] == sleeve
                        ]
                    )
                    - 1.0
                ),
            }
            for sleeve in range(5)
        ],
        "rows": rows,
    }


def compact_portfolio_summary(evaluation: PortfolioEvaluation) -> dict[str, Any]:
    """保留矩阵选模所需的可比较指标。"""
    summary = evaluation.summary
    return {
        "evaluated_dates": summary["evaluated_dates"],
        "mean_selected_rank_ic": summary["ranking"]["mean_selected_rank_ic"],
        "mean_active_return": summary["ranking"]["mean_active_return"],
        "mean_net_active_return": float(
            np.mean([row["net_active_return"] for row in evaluation.daily])
        ),
        "cumulative_net_return": summary["net"]["cumulative_return"],
        "mean_one_way_turnover": summary["turnover"]["mean_one_way"],
        "mean_transaction_cost": summary["turnover"]["mean_transaction_cost"],
        "mean_cash_weight": summary["risk_exposure"]["mean_cash_weight"],
        "top_5_absolute_active_contribution": summary["extreme_days"][
            "top_5_absolute_active_contribution"
        ],
    }
