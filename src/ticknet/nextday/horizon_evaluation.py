"""固定既有 checkpoint，只在 2024 validation 上评估多周期 Rank IC。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from ticknet.nextday.config import NextDayConfig
from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.horizon_labels import HorizonTarget, load_horizon_sidecar
from ticknet.nextday.metrics import _rank_correlation
from ticknet.nextday.model import build_nextday_model
from ticknet.nextday.train import (
    _checkpoint_matches_experiment,
    _checkpoint_paths,
    _environment,
    _experiment_signature,
    _json_safe,
    _load_checkpoint,
)
from ticknet.train import resolve_device

VALIDATION_START = date(2024, 1, 1)
VALIDATION_END = date(2024, 12, 31)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def newey_west_mean(values: Sequence[float], *, lag: int) -> dict[str, float | int]:
    """返回样本均值的 Bartlett-kernel Newey-West 标准误和 t 值。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values 必须是非空一维序列")
    if not np.all(np.isfinite(array)):
        raise ValueError("values 必须全部有限")
    if lag < 0:
        raise ValueError("lag 不能为负数")
    effective_lag = min(int(lag), array.size - 1)
    centered = array - np.mean(array)
    long_run_variance = float(np.dot(centered, centered) / array.size)
    for offset in range(1, effective_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / array.size)
        weight = 1.0 - offset / (effective_lag + 1)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / array.size)
    mean = float(np.mean(array))
    return {
        "lag": effective_lag,
        "standard_error": standard_error,
        "t_stat": mean / standard_error if standard_error > 0 else math.nan,
    }


def daily_rank_ic(
    scores: Mapping[tuple[str, date], float],
    targets: Sequence[HorizonTarget],
    *,
    min_symbols_per_day: int,
) -> list[dict[str, Any]]:
    """按信号日计算横截面 Rank IC，并保留每日横截面规模。"""
    if min_symbols_per_day < 2:
        raise ValueError("min_symbols_per_day 至少为 2")
    targets_by_date: dict[date, list[HorizonTarget]] = defaultdict(list)
    for target in targets:
        if (target.symbol, target.trading_date) in scores:
            targets_by_date[target.trading_date].append(target)

    rows: list[dict[str, Any]] = []
    for signal_date in sorted(targets_by_date):
        day_targets = sorted(targets_by_date[signal_date], key=lambda row: row.symbol)
        if len(day_targets) < min_symbols_per_day:
            continue
        day_scores = np.asarray(
            [scores[(row.symbol, row.trading_date)] for row in day_targets],
            dtype=np.float64,
        )
        day_returns = np.asarray([row.target_return for row in day_targets], dtype=np.float64)
        rank_ic = _rank_correlation(day_scores, day_returns)
        if math.isfinite(rank_ic):
            rows.append(
                {
                    "signal_date": signal_date,
                    "symbols": len(day_targets),
                    "rank_ic": rank_ic,
                }
            )
    return rows


def summarize_daily_ic(
    rows: Sequence[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, Any]:
    """汇总全期、月度、Newey-West 和非重叠 phase 结果。"""
    if horizon < 1:
        raise ValueError("horizon 应为正整数")
    if not rows:
        raise ValueError("没有可汇总的每日 Rank IC")
    ordered = sorted(rows, key=lambda row: row["signal_date"])
    values = np.asarray([float(row["rank_ic"]) for row in ordered], dtype=np.float64)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else math.nan

    monthly_values: dict[str, list[float]] = defaultdict(list)
    for row in ordered:
        signal_date = row["signal_date"]
        if not isinstance(signal_date, date):
            raise TypeError("signal_date 应为 date")
        monthly_values[signal_date.strftime("%Y-%m")].append(float(row["rank_ic"]))
    monthly = []
    for month in sorted(monthly_values):
        month_values = np.asarray(monthly_values[month], dtype=np.float64)
        monthly.append(
            {
                "month": month,
                "dates": int(month_values.size),
                "mean": float(np.mean(month_values)),
                "std": (float(np.std(month_values, ddof=1)) if month_values.size > 1 else math.nan),
            }
        )
    positive_months = sum(float(row["mean"]) > 0 for row in monthly)

    phases = []
    for phase in range(horizon):
        phase_values = values[phase::horizon]
        if phase_values.size == 0:
            continue
        phases.append(
            {
                "phase": phase,
                "dates": int(phase_values.size),
                "mean": float(np.mean(phase_values)),
                "std": (float(np.std(phase_values, ddof=1)) if phase_values.size > 1 else math.nan),
            }
        )
    phase_means = np.asarray([row["mean"] for row in phases], dtype=np.float64)

    return {
        "dates": int(values.size),
        "daily_rank_ic_mean": mean,
        "daily_rank_ic_std": standard_deviation,
        "daily_rank_ic_ir": (
            mean / standard_deviation
            if math.isfinite(standard_deviation) and standard_deviation > 0
            else math.nan
        ),
        "newey_west": newey_west_mean(values.tolist(), lag=horizon - 1),
        "monthly_stability": {
            "months": len(monthly),
            "positive_months": positive_months,
            "positive_month_ratio": positive_months / len(monthly),
            "monthly": monthly,
        },
        "non_overlapping": {
            "stride_trading_days": horizon,
            "phases": phases,
            "all_phase_means_nonnegative": bool(np.all(phase_means >= 0)),
            "min_phase_mean": float(np.min(phase_means)),
            "max_phase_mean": float(np.max(phase_means)),
        },
    }


def _validate_request(
    training_config: NextDayConfig,
    seeds: Sequence[int],
    horizons: Sequence[int],
    inference_batch_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    training_config.validate()
    if training_config.evaluate_test:
        raise ValueError("多周期评估必须保持 evaluate_test=False")
    if training_config.target_sidecar_path is not None or training_config.target_horizon != 1:
        raise ValueError("必须使用原始 H=1 训练配置校验既有 checkpoint")
    if (
        date.fromisoformat(training_config.val_start) != VALIDATION_START
        or date.fromisoformat(training_config.val_end) != VALIDATION_END
    ):
        raise ValueError("本入口只允许评估 2024-01-01 至 2024-12-31 validation")
    selected_seeds = tuple(int(seed) for seed in seeds)
    selected_horizons = tuple(sorted(int(horizon) for horizon in horizons))
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds 必须非空且不能重复")
    if not selected_horizons or any(horizon < 1 for horizon in selected_horizons):
        raise ValueError("horizons 必须包含正整数")
    if len(set(selected_horizons)) != len(selected_horizons):
        raise ValueError("horizons 不能重复")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size 应为正整数")
    return selected_seeds, selected_horizons


def _load_validated_checkpoints(
    config: NextDayConfig,
    seeds: Sequence[int],
    *,
    dataset_fingerprint: str,
) -> list[tuple[int, Path, dict[str, Any]]]:
    checkpoints = []
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        _stem, _last_path, best_path, _history_path, _result_path = _checkpoint_paths(seed_config)
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到 seed {seed} 的最佳 checkpoint：{best_path}")
        checkpoint = _load_checkpoint(best_path, torch.device("cpu"))
        expected = _experiment_signature(seed_config, dataset_fingerprint)
        if not _checkpoint_matches_experiment(checkpoint, expected):
            raise ValueError(f"{best_path} 的实验配置与原始 H=1 训练配置不同")
        checkpoints.append((seed, best_path, checkpoint))
    return checkpoints


def _infer_all_seeds(
    config: NextDayConfig,
    dataset: NextDayShardDataset,
    checkpoints: Sequence[tuple[int, Path, dict[str, Any]]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    models = []
    score_batches: dict[int, list[np.ndarray]] = {seed: [] for seed, _path, _cp in checkpoints}
    for seed, _path, checkpoint in checkpoints:
        model = build_nextday_model(
            chunks_per_sample=dataset.chunks_per_sample,
            chunk_size=dataset.chunk_size,
            conv_channels=config.conv_channels,
            inception_channels=config.inception_channels,
            intraday_embedding_size=config.intraday_embedding_size,
            day_hidden_size=config.day_hidden_size,
            day_layers=config.day_layers,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append((seed, model))

    with torch.no_grad():
        for features, _labels, _targets in loader:
            features = features.to(device, non_blocking=True)
            for seed, model in models:
                scores = model(features).score.detach().cpu().numpy()
                score_batches[seed].append(scores)
    result = {seed: np.concatenate(batches) for seed, batches in score_batches.items()}
    if any(values.shape != (len(dataset),) for values in result.values()):
        raise RuntimeError("推理分数长度与 validation 数据集不一致")
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("checkpoint 产生了非有限连续分数")
    return result


def _validation_targets(
    sidecar_path: str | Path,
    *,
    horizon: int,
    dataset_fingerprint: str,
    available_score_keys: set[tuple[str, date]],
    verify_checksum: bool,
) -> tuple[list[HorizonTarget], dict[str, int], str, str]:
    sidecar = load_horizon_sidecar(
        sidecar_path,
        horizon=horizon,
        source_dataset_fingerprint=dataset_fingerprint,
        verify_checksum=verify_checksum,
    )
    signal_period = [
        target
        for target in sidecar.records.values()
        if VALIDATION_START <= target.trading_date <= VALIDATION_END
    ]
    purged = [
        target
        for target in signal_period
        if not (
            VALIDATION_START <= target.entry_date <= VALIDATION_END
            and VALIDATION_START <= target.return_end_date <= VALIDATION_END
        )
    ]
    valid = [
        target
        for target in signal_period
        if VALIDATION_START <= target.entry_date <= VALIDATION_END
        and VALIDATION_START <= target.return_end_date <= VALIDATION_END
    ]
    missing_score_keys = {
        (target.symbol, target.trading_date)
        for target in valid
        if (target.symbol, target.trading_date) not in available_score_keys
    }
    if missing_score_keys:
        example = sorted(missing_score_keys)[0]
        raise ValueError(f"多周期标签无法与 validation 模型分数对齐，例如 {example}")
    valid.sort(key=lambda row: (row.trading_date, row.symbol))
    counts = {
        "signal_period": len(signal_period),
        "purged_at_validation_boundary": len(purged),
        "evaluated_samples": len(valid),
    }
    return valid, counts, sidecar.sidecar_fingerprint, sidecar.return_contract


def _cross_seed_summary(
    models: Mapping[str, dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    means = np.asarray(
        [models[f"seed_{seed}"]["daily_rank_ic_mean"] for seed in seeds],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(means)),
        "std": float(np.std(means, ddof=1)) if means.size > 1 else 0.0,
        "all_seed_means_positive": bool(np.all(means > 0)),
    }


def _decision_gate(horizon_result: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    models = horizon_result["models"]
    all_seed_means_positive = all(
        models[f"seed_{seed}"]["daily_rank_ic_mean"] > 0 for seed in seeds
    )
    majority_months_each_seed = all(
        models[f"seed_{seed}"]["monthly_stability"]["positive_month_ratio"] > 0.5 for seed in seeds
    )
    non_overlap_does_not_reverse = all(
        models[f"seed_{seed}"]["non_overlapping"]["all_phase_means_nonnegative"] for seed in seeds
    )
    return {
        "definition": (
            "每个 seed 的全期均值为正、正 IC 月份占比超过 50%，且每个非重叠 phase 均值不为负"
        ),
        "all_seed_means_positive": all_seed_means_positive,
        "majority_months_each_seed": majority_months_each_seed,
        "non_overlap_does_not_reverse": non_overlap_does_not_reverse,
        "meets_roadmap_gate": (
            all_seed_means_positive and majority_months_each_seed and non_overlap_does_not_reverse
        ),
    }


def evaluate_validation_horizons(
    training_config: NextDayConfig,
    sidecar_path: str | Path,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    horizons: Sequence[int] = (1, 3, 5),
    output_dir: str | Path,
    inference_batch_size: int = 128,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """固定模型评估 2024 validation；本函数从不创建 test dataset，也不训练模型。"""
    started_at = time.perf_counter()
    selected_seeds, selected_horizons = _validate_request(
        training_config,
        seeds,
        horizons,
        inference_batch_size,
    )
    if training_config.manifest_path is None:  # pragma: no cover - validate 已覆盖
        raise ValueError("manifest_path 不能为空")
    device = resolve_device(training_config.device)
    validation_dataset = NextDayShardDataset(
        training_config.manifest_path,
        date_split=training_config.date_split(),
        split="val",
        verify_checksums=training_config.verify_data_checksums,
        input_last_chunks=training_config.input_last_chunks,
    )
    checkpoints = _load_validated_checkpoints(
        training_config,
        selected_seeds,
        dataset_fingerprint=validation_dataset.dataset_fingerprint,
    )
    scores_by_seed = _infer_all_seeds(
        training_config,
        validation_dataset,
        checkpoints,
        device=device,
        batch_size=inference_batch_size,
    )

    keys = [(record.symbol, record.trading_date) for record in validation_dataset.records]
    score_maps = {
        seed: dict(zip(keys, values.tolist(), strict=True))
        for seed, values in scores_by_seed.items()
    }
    ensemble_values = np.mean(np.stack(list(scores_by_seed.values())), axis=0)
    ensemble_scores = dict(zip(keys, ensemble_values.tolist(), strict=True))
    available_score_keys = set(keys)

    output_root = Path(output_dir).expanduser().resolve()
    summary_path = output_root / "multi_horizon_validation_2024.json"
    scores_path = output_root / "validation_scores_2024.parquet"
    daily_path = output_root / "daily_rank_ic_2024.parquet"
    score_columns: dict[str, Any] = {
        "symbol": [key[0] for key in keys],
        "signal_date": pa.array([key[1] for key in keys], type=pa.date32()),
    }
    for seed in selected_seeds:
        score_columns[f"seed_{seed}"] = pa.array(scores_by_seed[seed], type=pa.float32())
    score_columns["ensemble"] = pa.array(ensemble_values, type=pa.float32())
    _atomic_parquet(scores_path, pa.table(score_columns))

    horizon_results: dict[str, Any] = {}
    daily_output_rows: list[dict[str, Any]] = []
    target_fingerprint: str | None = None
    return_contract: str | None = None
    for horizon_index, horizon in enumerate(selected_horizons):
        targets, counts, current_fingerprint, current_contract = _validation_targets(
            sidecar_path,
            horizon=horizon,
            dataset_fingerprint=validation_dataset.dataset_fingerprint,
            available_score_keys=available_score_keys,
            verify_checksum=horizon_index == 0,
        )
        if target_fingerprint is not None and current_fingerprint != target_fingerprint:
            raise ValueError("不同 horizon 的标签侧车指纹不一致")
        target_fingerprint = current_fingerprint
        return_contract = current_contract

        model_summaries: dict[str, Any] = {}
        named_scores = {
            **{f"seed_{seed}": score_maps[seed] for seed in selected_seeds},
            "ensemble": ensemble_scores,
        }
        for model_name, model_scores in named_scores.items():
            rows = daily_rank_ic(
                model_scores,
                targets,
                min_symbols_per_day=training_config.min_symbols_per_day,
            )
            model_summaries[model_name] = summarize_daily_ic(rows, horizon=horizon)
            daily_output_rows.extend(
                {
                    "model": model_name,
                    "horizon": horizon,
                    **row,
                }
                for row in rows
            )
        horizon_result = {
            "samples": counts,
            "models": model_summaries,
            "cross_seed_daily_rank_ic_mean": _cross_seed_summary(model_summaries, selected_seeds),
        }
        horizon_result["roadmap_gate"] = _decision_gate(horizon_result, selected_seeds)
        horizon_results[str(horizon)] = horizon_result

    _atomic_parquet(
        daily_path,
        pa.table(
            {
                "model": [row["model"] for row in daily_output_rows],
                "horizon": pa.array([row["horizon"] for row in daily_output_rows], type=pa.int16()),
                "signal_date": pa.array(
                    [row["signal_date"] for row in daily_output_rows], type=pa.date32()
                ),
                "symbols": pa.array([row["symbols"] for row in daily_output_rows], type=pa.int32()),
                "rank_ic": pa.array(
                    [row["rank_ic"] for row in daily_output_rows], type=pa.float64()
                ),
            }
        ),
    )

    checkpoint_rows = [
        {
            "seed": seed,
            "path": str(path),
            "sha256": _file_sha256(path),
            "best_epoch": int(checkpoint["epoch"]),
            "best_selection_value": float(checkpoint["best_selection_value"]),
        }
        for seed, path, checkpoint in checkpoints
    ]
    result = {
        "mode": "fixed_best_checkpoint_multi_horizon_validation",
        "validation_period": {
            "start": VALIDATION_START.isoformat(),
            "end": VALIDATION_END.isoformat(),
        },
        "test_status": "locked_not_accessed",
        "training_status": "not_run",
        "config": asdict(training_config),
        "seeds": list(selected_seeds),
        "horizons": list(selected_horizons),
        "inference_batch_size": inference_batch_size,
        "source_revision": source_revision,
        "environment": _environment(device),
        "dataset_fingerprint": validation_dataset.dataset_fingerprint,
        "target_fingerprint": target_fingerprint,
        "target_return_contract": return_contract,
        "validation_score_samples": len(validation_dataset),
        "checkpoints": checkpoint_rows,
        "results": horizon_results,
        "duration_seconds": time.perf_counter() - started_at,
        "artifacts": {
            "summary": str(summary_path),
            "scores": str(scores_path),
            "daily_rank_ic": str(daily_path),
        },
    }
    safe_result = _json_safe(result)
    _atomic_json(summary_path, safe_result)
    print(
        json.dumps(
            {
                "mode": safe_result["mode"],
                "test_status": safe_result["test_status"],
                "validation_score_samples": safe_result["validation_score_samples"],
                "summary": safe_result["artifacts"]["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return safe_result
