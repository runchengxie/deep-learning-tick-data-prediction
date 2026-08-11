"""分钟序列 GRU 的次日横截面方向训练入口。

与 ``train_tcn.py`` 复用同一套标签、日期切分、横截面指标和训练流程，输入是未
聚合的 ``T x features`` 分钟序列（``MinuteShardDataset``），模型换成
``MinuteGRU``。用于和聚合特征 HGB 基线及 TCN 做三路同口径受控对比。
"""

from __future__ import annotations

import argparse
import json
import math
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

from ticknet.nextday.config import NextDayConfig
from ticknet.nextday.minute_gru import MinuteShardDataset, build_minute_gru
from ticknet.nextday.train import (
    _atomic_json,
    _atomic_torch_save,
    _checkpoint_paths,
    _environment,
    _experiment_signature,
    _json_safe,
    _load_checkpoint,
)
from ticknet.nextday.train_tcn import (
    _aggregate_locked_test_metrics,
    _build_parser,
    _class_weights,
    evaluate,
    make_tcn_dataloaders,
)
from ticknet.train import resolve_device, set_seed


@dataclass
class MinuteGRUConfig(NextDayConfig):
    """分钟 GRU 实验配置，继承通用字段并追加 GRU 结构超参。"""

    gru_hidden_size: int = 64
    gru_layers: int = 2

    def validate(self) -> None:
        super().validate()
        if self.gru_hidden_size < 1 or self.gru_layers < 1:
            raise ValueError("GRU 隐藏维度和层数应为正整数")


def train(config: MinuteGRUConfig) -> dict[str, Any]:
    """训练分钟 GRU 并在完整测试日期区间上评估。"""
    started_at = time.perf_counter()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = make_tcn_dataloaders(config, device=device)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, MinuteShardDataset):
        raise TypeError("训练数据集类型无效")

    model = build_minute_gru(
        num_features=train_dataset.num_features,
        hidden_size=config.gru_hidden_size,
        num_layers=config.gru_layers,
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
    if not isinstance(val_dataset, MinuteShardDataset) or not isinstance(
        test_dataset, MinuteShardDataset
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


def load_config(argv: list[str] | None = None) -> MinuteGRUConfig:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)
    values = asdict(MinuteGRUConfig())
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        valid_names = {item.name for item in fields(MinuteGRUConfig)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)
    parser = _build_parser(values)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = MinuteGRUConfig(**arguments)
    config.validate()
    return config


def evaluate_best_checkpoints(
    config: MinuteGRUConfig,
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
        model = build_minute_gru(
            num_features=test_dataset.num_features,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_layers,
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


def main(argv: list[str] | None = None) -> None:
    train(load_config(argv))


def evaluate_main(argv: list[str] | None = None) -> None:
    probe = argparse.ArgumentParser(
        description="用固定 best checkpoint 一次性评估分钟 GRU locked test",
    )
    probe.add_argument("--seeds", nargs="+", type=int, required=True)
    arguments, remaining = probe.parse_known_args(argv)
    evaluate_best_checkpoints(load_config(remaining), arguments.seeds)


if __name__ == "__main__":
    main()
