"""分块 DeepLOB 次日横截面方向训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ticknet.dataset import NUM_CLASSES
from ticknet.nextday.dataset import NextDayShardDataset
from ticknet.nextday.metrics import evaluate_predictions
from ticknet.nextday.model import build_nextday_model
from ticknet.nextday.splits import WalkForwardSplit
from ticknet.train import resolve_device, set_seed

SELECTION_METRICS = {"daily_rank_ic_mean", "macro_f1", "balanced_accuracy", "mcc"}
LOCKED_TEST_AGGREGATE_METRICS = (
    "daily_rank_ic_mean",
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "brier_score",
    "daily_long_short_return_mean",
)


@dataclass
class NextDayConfig:
    """一次固定日期区间实验的配置。"""

    manifest_path: str | None = None
    train_start: str = "2021-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2024-12-31"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0
    patience: int = 8
    seed: int = 0
    num_workers: int = 0
    device: str = "cpu"
    resume: bool = True
    evaluate_test: bool = False
    verify_data_checksums: bool = True
    checkpoint_dir: str = "./checkpoints-nextday"
    checkpoint_name: str = "chunked-ticknet"
    intraday_embedding_size: int = 64
    day_hidden_size: int = 64
    day_layers: int = 1
    dropout: float = 0.1
    class_weighting: str = "balanced"
    selection_metric: str = "daily_rank_ic_mean"
    min_symbols_per_day: int = 20
    portfolio_quantile: float = 0.1
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 0.5
    gradient_accumulation_steps: int = 1
    amp: bool = True

    def validate(self) -> None:
        if not self.manifest_path:
            raise ValueError("manifest_path 不能为空")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs、batch_size 和 patience 应为正整数")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr 应为正数，weight_decay 不能为负数")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")
        if self.intraday_embedding_size < 1 or self.day_hidden_size < 1 or self.day_layers < 1:
            raise ValueError("模型隐藏维度和层数应为正整数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 应在 [0, 1) 内")
        if self.class_weighting not in {"none", "balanced"}:
            raise ValueError("class_weighting 应为 none 或 balanced")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric 应为 {sorted(SELECTION_METRICS)} 中的一个")
        if self.min_symbols_per_day < 2:
            raise ValueError("min_symbols_per_day 至少为 2")
        if not 0 < self.portfolio_quantile <= 0.5:
            raise ValueError("portfolio_quantile 应在 (0, 0.5] 内")
        if self.classification_loss_weight < 0 or self.regression_loss_weight < 0:
            raise ValueError("两个损失权重不能为负数")
        if self.classification_loss_weight + self.regression_loss_weight <= 0:
            raise ValueError("至少一个损失权重必须为正数")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 应为正整数")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 应为 cpu 或 cuda")
        self.date_split()

    def date_split(self) -> WalkForwardSplit:
        return WalkForwardSplit.from_strings(
            train_start=self.train_start,
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=self.test_start,
            test_end=self.test_end,
        )


def make_dataloaders(
    config: NextDayConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")
    date_split = config.date_split()
    datasets = tuple(
        NextDayShardDataset(
            config.manifest_path,
            date_split=date_split,
            split=split,
            verify_checksums=config.verify_data_checksums and split == "train",
        )
        for split in ("train", "val", "test")
    )
    loaders = []
    for index, dataset in enumerate(datasets):
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=index == 0,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=config.num_workers > 0,
            )
        )
    return loaders[0], loaders[1], loaders[2]


def _class_weights(dataset: NextDayShardDataset, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "none":
        return torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    counts = np.bincount([record.label for record in dataset.records], minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"训练集缺少类别：类别计数为 {counts.tolist()}")
    weights = len(dataset) / (NUM_CLASSES * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    min_symbols_per_day: int,
    portfolio_quantile: float,
) -> dict[str, Any]:
    dataset = dataloader.dataset
    if not isinstance(dataset, NextDayShardDataset):
        raise TypeError("次日评估需要 NextDayShardDataset")
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


def _atomic_torch_save(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
    }


def _experiment_signature(config: NextDayConfig, dataset_fingerprint: str) -> dict[str, Any]:
    signature = asdict(config)
    for name in (
        "epochs",
        "resume",
        "evaluate_test",
        "verify_data_checksums",
        "device",
        "num_workers",
    ):
        signature.pop(name)
    signature["dataset_fingerprint"] = dataset_fingerprint
    return signature


def _checkpoint_paths(config: NextDayConfig) -> tuple[str, Path, Path, Path, Path]:
    root = Path(config.checkpoint_dir)
    stem = f"{config.checkpoint_name}.seed{config.seed}"
    return (
        stem,
        root / f"{stem}.last.pt",
        root / f"{stem}.best.pt",
        root / f"train_history.{stem}.json",
        root / f"result.{stem}.json",
    )


def _aggregate_locked_test_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in LOCKED_TEST_AGGREGATE_METRICS:
        values = np.asarray([row["test"][metric] for row in per_seed], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }
    return aggregate


def evaluate_best_checkpoints(
    config: NextDayConfig,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """只读取固定的最佳 checkpoint，并一次性评估 locked test。"""
    started_at = time.perf_counter()
    config.validate()
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_seeds:
        raise ValueError("locked test 至少需要一个随机种子")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("locked test 随机种子不能重复")
    if config.manifest_path is None:
        raise ValueError("manifest_path 不能为空")

    device = resolve_device(config.device)
    test_dataset = NextDayShardDataset(
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

    checkpoints: list[tuple[int, Path, dict[str, Any]]] = []
    for seed in selected_seeds:
        seed_config = replace(config, seed=seed)
        _stem, _last_path, best_path, _history_path, _result_path = _checkpoint_paths(seed_config)
        if not best_path.is_file():
            raise FileNotFoundError(f"找不到 seed {seed} 的最佳 checkpoint：{best_path}")
        checkpoint = _load_checkpoint(best_path, torch.device("cpu"))
        expected_signature = _experiment_signature(
            seed_config,
            test_dataset.dataset_fingerprint,
        )
        if checkpoint.get("experiment") != expected_signature:
            raise ValueError(f"{best_path} 的实验配置与 locked test 配置不同")
        checkpoints.append((seed, best_path, checkpoint))

    per_seed: list[dict[str, Any]] = []
    for seed, best_path, checkpoint in checkpoints:
        set_seed(seed)
        model = build_nextday_model(
            chunks_per_sample=test_dataset.chunks_per_sample,
            chunk_size=test_dataset.chunk_size,
            intraday_embedding_size=config.intraday_embedding_size,
            day_hidden_size=config.day_hidden_size,
            day_layers=config.day_layers,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(
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


def train(config: NextDayConfig) -> dict[str, Any]:
    """训练分块模型并在完整测试日期区间上评估。"""
    started_at = time.perf_counter()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = make_dataloaders(config, device=device)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, NextDayShardDataset):
        raise TypeError("训练数据集类型无效")

    model = build_nextday_model(
        chunks_per_sample=train_dataset.chunks_per_sample,
        chunk_size=train_dataset.chunk_size,
        intraday_embedding_size=config.intraday_embedding_size,
        day_hidden_size=config.day_hidden_size,
        day_layers=config.day_layers,
        dropout=config.dropout,
    ).to(device)
    classification_criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_dataset, config.class_weighting, device)
    )
    regression_criterion = nn.SmoothL1Loss(beta=0.5)
    target_mean = float(np.mean(train_dataset.target_returns))
    target_std = float(np.std(train_dataset.target_returns))
    if not math.isfinite(target_std) or target_std <= 1e-12:
        raise ValueError("训练集目标收益没有有效方差")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    _stem, last_path, best_path, history_path, result_path = _checkpoint_paths(config)
    signature = _experiment_signature(config, train_dataset.dataset_fingerprint)

    start_epoch = 0
    best_selection_value = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if config.resume and last_path.exists():
        checkpoint = _load_checkpoint(last_path, device)
        if checkpoint.get("experiment") != signature:
            raise ValueError(f"{last_path} 的实验配置与本次运行不同")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_selection_value = float(checkpoint["best_selection_value"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        print(f"从第 {start_epoch} 个 epoch 后继续训练：{last_path}")

    can_continue = epochs_without_improvement < config.patience
    if not can_continue:
        print("checkpoint 已达到 early stopping 条件，跳过后续训练。")

    epoch_range = range(start_epoch, config.epochs) if can_continue else range(0)
    for epoch in epoch_range:
        epoch_started_at = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_regression_loss = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (features, labels, target_returns) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_returns = target_returns.to(device, non_blocking=True)
            normalized_targets = (target_returns - target_mean) / target_std
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(features)
                classification_loss = classification_criterion(output.logits, labels)
                regression_loss = regression_criterion(output.score, normalized_targets)
                loss = (
                    config.classification_loss_weight * classification_loss
                    + config.regression_loss_weight * regression_loss
                )
            scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % config.gradient_accumulation_steps == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * features.shape[0]
            total_classification_loss += classification_loss.item() * features.shape[0]
            total_regression_loss += regression_loss.item() * features.shape[0]
            sample_count += features.shape[0]
        if sample_count == 0:
            raise ValueError("训练数据集为空")

        validation = evaluate(
            model,
            val_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )
        raw_selection = validation[config.selection_metric]
        selection_value = float(raw_selection) if raw_selection is not None else math.nan
        comparable_selection = selection_value if math.isfinite(selection_value) else -math.inf
        improved = epoch == 0 or (comparable_selection > best_selection_value)
        if improved:
            best_selection_value = comparable_selection
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / sample_count,
            "train_classification_loss": total_classification_loss / sample_count,
            "train_regression_loss": total_regression_loss / sample_count,
            "epoch_seconds": time.perf_counter() - epoch_started_at,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_selection_value": best_selection_value,
            "epochs_without_improvement": epochs_without_improvement,
            "experiment": signature,
            "history": history,
            "target_normalization": {"mean": target_mean, "std": target_std},
        }
        if improved:
            _atomic_torch_save(best_path, state)
        _atomic_torch_save(last_path, state)
        _atomic_json(history_path, _json_safe(history))
        print(
            f"epoch {epoch + 1:03d}｜训练损失 {record['train_loss']:.4f}｜"
            f"验证 macro F1 {float(validation['macro_f1']):.4f}｜"
            f"验证 Rank IC {float(validation['daily_rank_ic_mean']):.4f}"
        )
        if epochs_without_improvement >= config.patience:
            print(f"验证指标连续 {config.patience} 个 epoch 未提升，停止训练。")
            break

    best = _load_checkpoint(best_path, device)
    model.load_state_dict(best["model"])
    test_metrics = None
    if config.evaluate_test:
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
            portfolio_quantile=config.portfolio_quantile,
        )
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    if not isinstance(val_dataset, NextDayShardDataset) or not isinstance(
        test_dataset, NextDayShardDataset
    ):
        raise TypeError("评估数据集类型无效")
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
        "best_selection_value": best_selection_value,
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


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 tick/LOB 到次日横截面方向模型")
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


def load_config(argv: list[str] | None = None) -> NextDayConfig:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)
    values = asdict(NextDayConfig())
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        valid_names = {item.name for item in fields(NextDayConfig)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)
    parser = _build_parser(values)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = NextDayConfig(**arguments)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    train(load_config(argv))


def evaluate_main(argv: list[str] | None = None) -> None:
    probe = argparse.ArgumentParser(
        description="用固定 best checkpoint 一次性评估次日 locked test",
    )
    probe.add_argument("--seeds", nargs="+", type=int, required=True)
    arguments, remaining = probe.parse_known_args(argv)
    evaluate_best_checkpoints(load_config(remaining), arguments.seeds)


if __name__ == "__main__":
    main()
