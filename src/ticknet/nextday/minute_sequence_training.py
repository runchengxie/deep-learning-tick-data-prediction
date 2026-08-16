"""分钟 TCN 与 GRU 共用的训练、恢复和 locked test 流程。"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ticknet.dataset import NUM_CLASSES
from ticknet.nextday.config import LOCKED_TEST_AGGREGATE_METRICS, NextDayConfig
from ticknet.nextday.metrics import evaluate_predictions
from ticknet.nextday.minute_tcn import MinuteShardDataset
from ticknet.nextday.train import (
    _atomic_json,
    _atomic_torch_save,
    _checkpoint_paths,
    _environment,
    _experiment_signature,
    _json_safe,
    _load_checkpoint,
)
from ticknet.train import resolve_device, set_seed

MinuteModelFactory = Callable[[int], nn.Module]
ConfigT = TypeVar("ConfigT", bound=NextDayConfig)


@dataclass
class _TrainingProgress:
    start_epoch: int = 0
    best_selection_value: float = -math.inf
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TrainingRuntime:
    device: torch.device
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    classification_criterion: nn.CrossEntropyLoss
    regression_criterion: nn.SmoothL1Loss
    target_mean: float
    target_std: float
    use_amp: bool


def make_minute_dataloaders(
    config: NextDayConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """按统一日期切分构造分钟序列训练、验证和测试加载器。"""
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")
    date_split = config.date_split()
    datasets = tuple(
        MinuteShardDataset(
            config.manifest_path,
            date_split=date_split,
            split=split,
            verify_checksums=config.verify_data_checksums and split == "train",
        )
        for split in ("train", "val", "test")
    )
    loaders = [
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=index == 0,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        for index, dataset in enumerate(datasets)
    ]
    return loaders[0], loaders[1], loaders[2]


def _class_weights(
    dataset: MinuteShardDataset,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    counts = np.bincount([record.label for record in dataset.records], minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"训练集缺少类别：类别计数为 {counts.tolist()}")
    weights = len(dataset) / (NUM_CLASSES * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def evaluate_minute_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    min_symbols_per_day: int,
    portfolio_quantile: float,
) -> dict[str, Any]:
    """使用分钟数据契约评估具有分类和回归双输出的序列模型。"""
    dataset = dataloader.dataset
    if not isinstance(dataset, MinuteShardDataset):
        raise TypeError("分钟评估需要 MinuteShardDataset")
    probability_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, labels, _target_returns in dataloader:
            features = features.to(device, non_blocking=True)
            output = model(features)
            probability_batches.append(torch.softmax(output.logits, dim=1).cpu().numpy())
            score_batches.append(output.score.cpu().numpy())
            label_batches.append(labels.numpy())
    if not probability_batches:
        raise ValueError("评估数据集为空")
    return evaluate_predictions(
        np.concatenate(label_batches),
        np.concatenate(probability_batches),
        dataset.target_returns,
        dataset.label_dates,
        scores=np.concatenate(score_batches),
        min_symbols_per_day=min_symbols_per_day,
        portfolio_quantile=portfolio_quantile,
    )


def _require_minute_dataset(dataloader: DataLoader, *, role: str) -> MinuteShardDataset:
    dataset = dataloader.dataset
    if not isinstance(dataset, MinuteShardDataset):
        raise TypeError(f"{role}数据集类型无效")
    return dataset


def _build_runtime(
    config: NextDayConfig,
    dataset: MinuteShardDataset,
    device: torch.device,
    model_factory: MinuteModelFactory,
) -> _TrainingRuntime:
    model = model_factory(dataset.num_features).to(device)
    target_mean = float(np.mean(dataset.target_returns))
    target_std = float(np.std(dataset.target_returns))
    if not math.isfinite(target_std) or target_std <= 1e-12:
        raise ValueError("训练集目标收益没有有效方差")
    use_amp = config.amp and device.type == "cuda"
    return _TrainingRuntime(
        device=device,
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        ),
        scaler=torch.amp.GradScaler(device.type, enabled=use_amp),
        classification_criterion=nn.CrossEntropyLoss(
            weight=_class_weights(dataset, config.class_weighting, device)
        ),
        regression_criterion=nn.SmoothL1Loss(beta=0.5),
        target_mean=target_mean,
        target_std=target_std,
        use_amp=use_amp,
    )


def _restore_progress(
    config: NextDayConfig,
    runtime: _TrainingRuntime,
    last_path: Path,
    signature: dict[str, Any],
) -> _TrainingProgress:
    if not config.resume or not last_path.exists():
        return _TrainingProgress()
    checkpoint = _load_checkpoint(last_path, runtime.device)
    if checkpoint.get("experiment") != signature:
        raise ValueError(f"{last_path} 的实验配置与本次运行不同")
    runtime.model.load_state_dict(checkpoint["model"])
    runtime.optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        runtime.scaler.load_state_dict(checkpoint["scaler"])
    progress = _TrainingProgress(
        start_epoch=int(checkpoint["epoch"]),
        best_selection_value=float(checkpoint["best_selection_value"]),
        epochs_without_improvement=int(checkpoint["epochs_without_improvement"]),
        history=list(checkpoint.get("history", [])),
    )
    print(f"从第 {progress.start_epoch} 个 epoch 后继续训练：{last_path}")
    return progress


def _train_epoch(
    config: NextDayConfig,
    runtime: _TrainingRuntime,
    train_loader: DataLoader,
) -> dict[str, float]:
    epoch_started_at = time.perf_counter()
    runtime.model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    sample_count = 0
    runtime.optimizer.zero_grad(set_to_none=True)
    for batch_index, (features, labels, target_returns) in enumerate(train_loader):
        features = features.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)
        target_returns = target_returns.to(runtime.device, non_blocking=True)
        normalized_targets = (target_returns - runtime.target_mean) / runtime.target_std
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=runtime.use_amp,
        ):
            output = runtime.model(features)
            classification_loss = runtime.classification_criterion(output.logits, labels)
            regression_loss = runtime.regression_criterion(output.score, normalized_targets)
            loss = (
                config.classification_loss_weight * classification_loss
                + config.regression_loss_weight * regression_loss
            )
        runtime.scaler.scale(loss / config.gradient_accumulation_steps).backward()
        should_step = (batch_index + 1) % config.gradient_accumulation_steps == 0 or (
            batch_index + 1 == len(train_loader)
        )
        if should_step:
            runtime.scaler.step(runtime.optimizer)
            runtime.scaler.update()
            runtime.optimizer.zero_grad(set_to_none=True)
        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_classification_loss += classification_loss.item() * batch_size
        total_regression_loss += regression_loss.item() * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("训练数据集为空")
    return {
        "train_loss": total_loss / sample_count,
        "train_classification_loss": total_classification_loss / sample_count,
        "train_regression_loss": total_regression_loss / sample_count,
        "epoch_seconds": time.perf_counter() - epoch_started_at,
    }


def _update_selection(
    progress: _TrainingProgress,
    *,
    epoch: int,
    selection_value: Any,
) -> bool:
    value = float(selection_value) if selection_value is not None else math.nan
    comparable = value if math.isfinite(value) else -math.inf
    improved = epoch == 0 or comparable > progress.best_selection_value
    if improved:
        progress.best_selection_value = comparable
        progress.epochs_without_improvement = 0
    else:
        progress.epochs_without_improvement += 1
    return improved


def _checkpoint_state(
    runtime: _TrainingRuntime,
    progress: _TrainingProgress,
    signature: dict[str, Any],
    *,
    epoch: int,
) -> dict[str, Any]:
    return {
        "model": runtime.model.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "scaler": runtime.scaler.state_dict(),
        "epoch": epoch + 1,
        "best_selection_value": progress.best_selection_value,
        "epochs_without_improvement": progress.epochs_without_improvement,
        "experiment": signature,
        "history": progress.history,
        "target_normalization": {"mean": runtime.target_mean, "std": runtime.target_std},
    }


def _run_training_epochs(
    config: NextDayConfig,
    runtime: _TrainingRuntime,
    train_loader: DataLoader,
    val_loader: DataLoader,
    progress: _TrainingProgress,
    signature: dict[str, Any],
    *,
    last_path: Path,
    best_path: Path,
    history_path: Path,
) -> None:
    if progress.epochs_without_improvement >= config.patience:
        print("checkpoint 已达到 early stopping 条件，跳过后续训练。")
        return
    for epoch in range(progress.start_epoch, config.epochs):
        losses = _train_epoch(config, runtime, train_loader)
        validation = evaluate_minute_model(
            runtime.model,
            val_loader,
            runtime.device,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )
        improved = _update_selection(
            progress,
            epoch=epoch,
            selection_value=validation[config.selection_metric],
        )
        record = {
            "epoch": epoch + 1,
            **losses,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        progress.history.append(record)
        state = _checkpoint_state(runtime, progress, signature, epoch=epoch)
        if improved:
            _atomic_torch_save(best_path, state)
        _atomic_torch_save(last_path, state)
        _atomic_json(history_path, _json_safe(progress.history))
        print(
            f"epoch {epoch + 1:03d}｜训练损失 {record['train_loss']:.4f}｜"
            f"验证 macro F1 {float(validation['macro_f1']):.4f}｜"
            f"验证 Rank IC {float(validation['daily_rank_ic_mean']):.4f}"
        )
        if progress.epochs_without_improvement >= config.patience:
            print(f"验证指标连续 {config.patience} 个 epoch 未提升，停止训练。")
            break


def train_minute_sequence_model(
    config: NextDayConfig,
    model_factory: MinuteModelFactory,
) -> dict[str, Any]:
    """训练一种分钟序列模型，并按既有契约写 checkpoint 和结果 JSON。"""
    started_at = time.perf_counter()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = make_minute_dataloaders(config, device=device)
    train_dataset = _require_minute_dataset(train_loader, role="训练")
    val_dataset = _require_minute_dataset(val_loader, role="评估")
    test_dataset = _require_minute_dataset(test_loader, role="评估")
    runtime = _build_runtime(config, train_dataset, device, model_factory)
    _stem, last_path, best_path, history_path, result_path = _checkpoint_paths(config)
    signature = _experiment_signature(config, train_dataset.dataset_fingerprint)
    progress = _restore_progress(config, runtime, last_path, signature)
    _run_training_epochs(
        config,
        runtime,
        train_loader,
        val_loader,
        progress,
        signature,
        last_path=last_path,
        best_path=best_path,
        history_path=history_path,
    )

    best = _load_checkpoint(best_path, device)
    runtime.model.load_state_dict(best["model"])
    test_metrics = None
    if config.evaluate_test:
        test_metrics = evaluate_minute_model(
            runtime.model,
            test_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )
    result = {
        "config": asdict(config),
        "samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "environment": _environment(device),
        "duration_seconds": time.perf_counter() - started_at,
        "dataset_fingerprint": train_dataset.dataset_fingerprint,
        "best_selection_value": progress.best_selection_value,
        "test": test_metrics,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "history": str(history_path),
        "result_file": str(result_path),
    }
    safe_result = _json_safe(result)
    _atomic_json(result_path, safe_result)
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))
    return safe_result


def _selected_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(seed) for seed in seeds)
    if not selected:
        raise ValueError("locked test 至少需要一个随机种子")
    if len(set(selected)) != len(selected):
        raise ValueError("locked test 随机种子不能重复")
    return selected


def _load_locked_checkpoints(
    config: NextDayConfig,
    selected_seeds: tuple[int, ...],
    dataset_fingerprint: str,
) -> list[tuple[int, Path, dict[str, Any]]]:
    checkpoints: list[tuple[int, Path, dict[str, Any]]] = []
    for seed in selected_seeds:
        seed_config = replace(config, seed=seed)
        _stem, _last_path, best_path, _history_path, _result_path = _checkpoint_paths(seed_config)
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到 seed {seed} 的最佳 checkpoint：{best_path}")
        checkpoint = _load_checkpoint(best_path, torch.device("cpu"))
        expected_signature = _experiment_signature(seed_config, dataset_fingerprint)
        if checkpoint.get("experiment") != expected_signature:
            raise ValueError(f"{best_path} 的实验配置与 locked test 配置不同")
        checkpoints.append((seed, best_path, checkpoint))
    return checkpoints


def _evaluate_locked_checkpoints(
    config: NextDayConfig,
    checkpoints: list[tuple[int, Path, dict[str, Any]]],
    test_loader: DataLoader,
    test_dataset: MinuteShardDataset,
    device: torch.device,
    model_factory: MinuteModelFactory,
) -> list[dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    for seed, best_path, checkpoint in checkpoints:
        set_seed(seed)
        model = model_factory(test_dataset.num_features).to(device)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate_minute_model(
            model,
            test_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )
        per_seed.append(
            {
                "seed": seed,
                "best_epoch": int(checkpoint["epoch"]),
                "best_selection_value": float(checkpoint["best_selection_value"]),
                "best_checkpoint": str(best_path),
                "test": metrics,
            }
        )
        del model
    return per_seed


def _aggregate_locked_test_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in LOCKED_TEST_AGGREGATE_METRICS:
        values = np.asarray([row["test"][metric] for row in per_seed], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }
    return aggregate


def evaluate_best_minute_checkpoints(
    config: NextDayConfig,
    seeds: Sequence[int],
    model_factory: MinuteModelFactory,
) -> dict[str, Any]:
    """校验并评估一种分钟序列模型的固定最佳 checkpoint。"""
    started_at = time.perf_counter()
    config.validate()
    selected_seeds = _selected_seeds(seeds)
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")
    device = resolve_device(config.device)
    test_dataset = MinuteShardDataset(
        config.manifest_path,
        date_split=config.date_split(),
        split="test",
        verify_checksums=config.verify_data_checksums,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    checkpoints = _load_locked_checkpoints(
        config,
        selected_seeds,
        test_dataset.dataset_fingerprint,
    )
    per_seed = _evaluate_locked_checkpoints(
        config,
        checkpoints,
        test_loader,
        test_dataset,
        device,
        model_factory,
    )
    seed_label = "-".join(str(seed) for seed in selected_seeds)
    result_path = (
        Path(config.checkpoint_dir) / f"locked_test.{config.checkpoint_name}.seeds{seed_label}.json"
    )
    result = {
        "mode": "best_checkpoint_locked_test",
        "config": asdict(config),
        "seeds": list(selected_seeds),
        "samples": {"test": len(test_dataset)},
        "environment": _environment(device),
        "duration_seconds": time.perf_counter() - started_at,
        "dataset_fingerprint": test_dataset.dataset_fingerprint,
        "per_seed": per_seed,
        "aggregate": _aggregate_locked_test_metrics(per_seed),
        "result_file": str(result_path),
    }
    safe_result = _json_safe(result)
    _atomic_json(result_path, safe_result)
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))
    return safe_result


def _build_config_parser(
    defaults: dict[str, Any],
    *,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config")
    for name, value in defaults.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction)
        elif isinstance(value, int):
            parser.add_argument(option, type=int)
        elif isinstance(value, float):
            parser.add_argument(option, type=float)
        else:
            parser.add_argument(option)
    parser.set_defaults(**defaults)
    return parser


def load_minute_config(
    config_type: type[ConfigT],
    *,
    description: str,
    argv: list[str] | None = None,
) -> ConfigT:
    """合并分钟模型的默认值、YAML 和 CLI 参数，并执行模型专属校验。"""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)
    values = asdict(config_type())
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        valid_names = {item.name for item in fields(config_type)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)
    parser = _build_config_parser(values, description=description)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = config_type(**arguments)
    config.validate()
    return config
