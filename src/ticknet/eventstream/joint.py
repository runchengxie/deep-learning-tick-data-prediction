"""联合微调事件流 Transformer 与分钟特征塔。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader, Dataset

from ticknet.eventstream.close_cache import load_close_cache_manifest, verify_close_cache
from ticknet.eventstream.fingerprint import file_sha256, git_sha
from ticknet.eventstream.joint_cache import load_joint_cache_manifest
from ticknet.eventstream.model import CONFIGS, build_eventstream_model
from ticknet.research.portfolio import (
    CostModel,
    PortfolioPolicy,
    PortfolioPrediction,
    evaluate_topk_portfolio,
)
from ticknet.train import resolve_device, set_seed

PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class JointConfig:
    """一次联合微调实验的模型和训练参数。"""

    model: str = "capacity100m"
    seed: int = 0
    epochs: int = 5
    batch_size: int = 8
    backbone_lr: float = 1e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 2
    freeze_backbone_epochs: int = 1
    minute_hidden: int = 128
    fusion_hidden: int = 256
    dropout: float = 0.1
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    gradient_accumulation_steps: int = 1
    min_symbols_per_day: int = 350
    relevance_levels: int = 5
    top_ks: tuple[int, ...] = (50, 100)
    cost_bps: float = 10.0
    sell_stamp_tax_bps: float = 5.0
    resume: bool = True

    def validate(self) -> None:
        if self.model not in CONFIGS:
            raise ValueError(f"model 应为 {sorted(CONFIGS)} 之一")
        positive = (
            self.epochs,
            self.batch_size,
            self.patience,
            self.minute_hidden,
            self.fusion_hidden,
            self.gradient_accumulation_steps,
            self.min_symbols_per_day,
            self.relevance_levels,
        )
        if any(value < 1 for value in positive):
            raise ValueError("训练轮数、批量和网络维度应为正整数")
        if self.freeze_backbone_epochs < 0 or self.num_workers < 0:
            raise ValueError("冻结轮数和 num_workers 不能为负数")
        if self.backbone_lr <= 0 or self.head_lr <= 0 or self.weight_decay < 0:
            raise ValueError("学习率应为正数，weight_decay 不能为负数")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 应位于 [0, 1) 区间")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 应为 cpu 或 cuda")
        if not self.top_ks or any(value < 1 for value in self.top_ks):
            raise ValueError("top_ks 必须包含正整数")
        if self.min_symbols_per_day < max(self.top_ks):
            raise ValueError("min_symbols_per_day 不能小于最大的 Top-K")
        if self.relevance_levels < 2:
            raise ValueError("relevance_levels 至少为 2")
        if self.cost_bps < 0 or self.sell_stamp_tax_bps < 0:
            raise ValueError("交易成本不能为负数")


def load_joint_config(path: Path) -> JointConfig:
    with Path(path).open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("联合微调配置应为 YAML 对象")
    known = set(JointConfig.__dataclass_fields__)
    if unknown := {str(key) for key in raw} - known:
        raise ValueError(f"联合微调配置包含未知字段：{sorted(unknown)}")
    values = dict(raw)
    if "top_ks" in values:
        values["top_ks"] = tuple(int(value) for value in values["top_ks"])
    config = JointConfig(**values)
    config.validate()
    return config


def _fixed_list_matrix(column: pa.ChunkedArray, dimension: int) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type) or array.type.list_size != dimension:
        raise ValueError("分钟特征维度与联合缓存 manifest 不一致")
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        len(array), dimension
    )


def _normalizer(table: pa.Table, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    values = _fixed_list_matrix(table["minute_features"], feature_count)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(sums, counts, out=np.zeros(feature_count), where=counts > 0)
    centered = np.where(finite, values - means, 0.0)
    variances = np.divide(
        np.square(centered).sum(axis=0, dtype=np.float64),
        counts,
        out=np.ones(feature_count),
        where=counts > 0,
    )
    scales = np.sqrt(variances)
    scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    return means.astype(np.float32), scales.astype(np.float32)


class JointDataset(Dataset[tuple[torch.Tensor, ...]]):
    """用轻量行号索引共享尾盘缓存，不复制事件数组。"""

    def __init__(
        self,
        cache_root: Path,
        close_root: Path,
        partition: str,
        means: np.ndarray,
        scales: np.ndarray,
    ):
        manifest = load_joint_cache_manifest(cache_root, verify_files=False)
        record = next(row for row in manifest["artifacts"] if row["partition"] == partition)
        table = pq.read_table(Path(cache_root) / str(record["path"]))
        feature_count = int(manifest["contract"]["feature_count"])
        values = _fixed_list_matrix(table["minute_features"], feature_count)
        values = np.where(np.isfinite(values), values, means)
        values = np.clip((values - means) / scales, -10.0, 10.0).astype(np.float32)
        available = np.asarray(table["feature_available"].to_numpy(), dtype=np.float32)[:, None]
        self.minute = np.concatenate((values, available), axis=1)
        self.days = np.asarray(table["trading_day"].to_numpy(), dtype=np.int32)
        self.symbols = tuple(str(value) for value in table["symbol"].to_pylist())
        self.label_dates = tuple(table["label_date"].to_pylist())
        self.labels = np.asarray(table["label"].to_numpy(), dtype=np.int64)
        self.returns = np.asarray(table["ranking_target_return"].to_numpy(), dtype=np.float64)
        self.shards = tuple(str(value) for value in table["close_shard"].to_pylist())
        self.rows = np.asarray(table["close_row"].to_numpy(), dtype=np.int64)
        self.close_root = Path(close_root)
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def _shard_arrays(self, shard: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if shard not in self._arrays:
            root = self.close_root / shard
            self._arrays[shard] = tuple(
                np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("x", "sid", "oid")
            )
        return self._arrays[shard]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        x, sid, oid = self._shard_arrays(self.shards[index])
        row = int(self.rows[index])
        return (
            torch.from_numpy(np.array(x[row], copy=True)),
            torch.from_numpy(np.array(sid[row], dtype=np.int64, copy=True)),
            torch.from_numpy(np.array(oid[row], dtype=np.int64, copy=True)),
            torch.from_numpy(self.minute[index].copy()),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            torch.tensor(float(self.returns[index]), dtype=torch.float64),
            torch.tensor(int(self.days[index]), dtype=torch.int32),
            torch.tensor(index, dtype=torch.long),
        )


class JointEventstreamModel(nn.Module):
    """将收盘事件表征和同日分钟特征拼接后预测三分类信号。"""

    def __init__(self, config: JointConfig, feature_count: int):
        super().__init__()
        self.eventstream = build_eventstream_model(config.model)
        hidden = CONFIGS[config.model].d_model
        self.minute_tower = nn.Sequential(
            nn.Linear(feature_count + 1, config.minute_hidden),
            nn.GELU(),
            nn.LayerNorm(config.minute_hidden),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden + config.minute_hidden, config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        sid: torch.Tensor,
        oid: torch.Tensor,
        minute: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.eventstream.backbone(x, sid, oid)
        lengths = (sid != 0).sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("联合微调输入包含空事件窗口")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        event = hidden[rows, lengths - 1]
        return self.classifier(torch.cat((event, self.minute_tower(minute)), dim=1))


def load_pretrained_backbone(
    model: JointEventstreamModel,
    checkpoint_path: Path,
    *,
    model_name: str,
    seed: int,
    expected_sha256: str,
) -> dict[str, Any]:
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("预训练 checkpoint 的 SHA-256 与配置不一致")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("experiment"), dict):
        raise ValueError("预训练 checkpoint 缺少实验签名")
    experiment = checkpoint["experiment"]
    if experiment.get("model") != model_name or int(experiment.get("seed", -1)) != seed:
        raise ValueError("预训练 checkpoint 的模型或 seed 与配置不一致")
    if any(
        bool(experiment.get(name, False))
        for name in ("use_lob_prefix", "use_session_anchors", "use_vq")
    ):
        raise ValueError("联合微调暂不支持 M3-inspired 事件流表征")
    model.eventstream.load_state_dict(checkpoint["model"])
    return {
        "sha256": actual_sha256,
        "epoch": int(checkpoint.get("epoch", -1)),
        "selection_value": float(checkpoint.get("best_selection_value", math.nan)),
        "source_revision": str(experiment.get("source_revision", "")),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    score_ranks = _average_ranks(scores)
    return_ranks = _average_ranks(returns)
    if np.std(score_ranks) == 0 or np.std(return_ranks) == 0:
        return math.nan
    return float(np.corrcoef(score_ranks, return_ranks)[0, 1])


def _relevance(returns: np.ndarray, levels: int) -> np.ndarray:
    order = np.argsort(returns, kind="mergesort")
    ranks = np.empty(len(returns), dtype=np.int64)
    ranks[order] = np.arange(len(returns))
    return np.minimum(levels - 1, ranks * levels // len(returns))


def _ranking_metrics(
    dataset: JointDataset,
    scores: np.ndarray,
    config: JointConfig,
) -> dict[str, Any]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, day in enumerate(dataset.days):
        grouped[int(day)].append(index)
    daily: list[dict[str, Any]] = []
    for day, indices in sorted(grouped.items()):
        if len(indices) < config.min_symbols_per_day:
            raise ValueError(f"联合评估股票池低于 min_symbols_per_day：{len(indices)}")
        selected = np.asarray(indices)
        day_scores = scores[selected]
        day_returns = dataset.returns[selected]
        relevance = _relevance(day_returns, config.relevance_levels)
        predicted = np.argsort(-day_scores, kind="mergesort")
        realized = np.argsort(-day_returns, kind="mergesort")
        row: dict[str, Any] = {
            "trading_date": str(day),
            "symbols": len(indices),
            "rank_ic": _rank_ic(day_scores, day_returns),
        }
        for top_k in config.top_ks:
            k = min(top_k, len(indices))
            row[f"ndcg_at_{top_k}"] = float(
                ndcg_score(relevance[None, :], day_scores[None, :], k=k)
            )
            row[f"precision_at_{top_k}"] = float(len(set(predicted[:k]) & set(realized[:k])) / k)
        daily.append(row)
    values = np.asarray([row["rank_ic"] for row in daily], dtype=np.float64)
    metrics: dict[str, Any] = {
        "dates": len(daily),
        "symbols_per_day_min": min(row["symbols"] for row in daily),
        "symbols_per_day_max": max(row["symbols"] for row in daily),
        "daily_rank_ic_mean": float(np.mean(values)),
        "daily_rank_ic_std": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
    }
    for top_k in config.top_ks:
        for name in ("ndcg", "precision"):
            key = f"{name}_at_{top_k}"
            metrics[key] = float(np.mean([row[key] for row in daily]))
    return {"summary": metrics, "daily": daily}


def _portfolio_metrics(
    dataset: JointDataset,
    scores: np.ndarray,
    targets_path: Path,
    config: JointConfig,
) -> dict[str, Any]:
    table = pq.read_table(targets_path)
    targets: dict[tuple[date, str], dict[str, Any]] = {}
    for row in table.to_pylist():
        targets[(row["trading_date"], str(row["symbol"]))] = row
    scores_by_key = {
        (date(int(str(day)[:4]), int(str(day)[4:6]), int(str(day)[6:])), symbol): float(score)
        for day, symbol, score in zip(dataset.days, dataset.symbols, scores, strict=True)
    }
    selected_by_day: dict[date, set[str]] = defaultdict(set)
    for trading_date, symbol in scores_by_key:
        selected_by_day[trading_date].add(symbol)
    predictions: list[PortfolioPrediction] = []
    previous: set[str] = set()
    for trading_date, current in sorted(selected_by_day.items()):
        for symbol in sorted(current | (previous - current)):
            target = targets.get((trading_date, symbol))
            if target is None:
                continue
            predictions.append(
                PortfolioPrediction(
                    symbol=symbol,
                    trading_date=trading_date,
                    label_date=target["label_date"],
                    score=scores_by_key.get((trading_date, symbol), 0.0),
                    target_return=float(target["portfolio_return"]),
                    can_buy=bool(target["can_buy"]),
                    can_sell=bool(target["can_sell"]),
                    tradability_known=True,
                    in_universe=symbol in current,
                    universe_membership_known=True,
                )
            )
        previous = current
    result: dict[str, Any] = {}
    for top_k in config.top_ks:
        evaluation = evaluate_topk_portfolio(
            predictions,
            policy=PortfolioPolicy(
                top_k=top_k,
                min_symbols_per_day=config.min_symbols_per_day,
                require_tradability=True,
                require_universe_membership=True,
            ),
            cost_model=CostModel(
                per_side_bps=config.cost_bps,
                sell_stamp_tax_bps=config.sell_stamp_tax_bps,
            ),
        )
        summary = dict(evaluation.summary)
        summary["ranking"] = dict(summary["ranking"])
        summary["ranking"]["mean_net_active_return"] = float(
            np.mean([row["net_active_return"] for row in evaluation.daily])
        )
        result[str(top_k)] = summary
    return result


@torch.no_grad()
def predict(
    model: JointEventstreamModel,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    dataset = loader.dataset
    if not isinstance(dataset, JointDataset):
        raise TypeError("联合预测需要 JointDataset")
    scores = np.empty(len(dataset), dtype=np.float64)
    for x, sid, oid, minute, _label, _returns, _day, indices in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                x.to(device, non_blocking=True),
                sid.to(device, non_blocking=True),
                oid.to(device, non_blocking=True),
                minute.to(device, non_blocking=True),
            )
        probabilities = logits.float().softmax(dim=1)
        batch_scores = probabilities[:, 2] - probabilities[:, 0]
        scores[indices.numpy()] = batch_scores.cpu().numpy()
    return scores


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _signature(
    config: JointConfig,
    cache_manifest: dict[str, Any],
    close_manifest: dict[str, Any],
    checkpoint_sha256: str,
    source_revision: str,
) -> dict[str, Any]:
    values = asdict(config)
    for name in ("epochs", "resume", "device", "num_workers", "amp"):
        values.pop(name)
    return {
        **values,
        "top_ks": list(config.top_ks),
        "joint_cache_fingerprint": cache_manifest["dataset_fingerprint"],
        "close_cache_fingerprint": close_manifest["dataset_fingerprint"],
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
    }


def _write_predictions(path: Path, dataset: JointDataset, scores: np.ndarray) -> None:
    table = pa.table(
        {
            "trading_day": pa.array(dataset.days, type=pa.int32()),
            "symbol": dataset.symbols,
            "label_date": pa.array(dataset.label_dates, type=pa.date32()),
            "score": pa.array(scores, type=pa.float64()),
            "target_return": pa.array(dataset.returns, type=pa.float64()),
            "label": pa.array(dataset.labels, type=pa.int8()),
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _train_one_epoch(
    model: JointEventstreamModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    gradient_accumulation_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_function = nn.CrossEntropyLoss()
    loss_sum = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        x, sid, oid, minute, labels, _returns, _day, _indices = batch
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                x.to(device, non_blocking=True),
                sid.to(device, non_blocking=True),
                oid.to(device, non_blocking=True),
                minute.to(device, non_blocking=True),
            )
            loss = loss_function(logits, labels.to(device, non_blocking=True))
        scaler.scale(loss / gradient_accumulation_steps).backward()
        should_step = (
            batch_index + 1
        ) % gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * x.shape[0]
        sample_count += x.shape[0]
    if sample_count == 0:
        raise ValueError("联合微调训练分区为空")
    return loss_sum / sample_count


def train_joint(
    config: JointConfig,
    *,
    cache_root: Path,
    close_root: Path,
    pretrained_checkpoint: Path,
    expected_pretrained_sha256: str,
    output_root: Path,
    source_revision: str,
    allow_oos: bool,
) -> dict[str, Any]:
    """用同 seed 的预训练主干运行最近折联合端到端实验。"""
    started = time.perf_counter()
    config.validate()
    if not allow_oos:
        raise ValueError("联合实验读取 OOS 分区前必须显式传入 --allow-oos")
    if not source_revision or source_revision == "unknown":
        raise ValueError("联合微调需要有效的源码 revision")
    set_seed(config.seed)
    device = resolve_device(config.device)
    cache_manifest = load_joint_cache_manifest(cache_root)
    verify_close_cache(close_root)
    close_manifest = load_close_cache_manifest(close_root)
    contract = cache_manifest["contract"]
    if contract["close_cache_fingerprint"] != close_manifest["dataset_fingerprint"]:
        raise ValueError("联合特征缓存与尾盘缓存指纹不一致")
    comparison = contract["comparison_config"]
    if str(comparison["oos_end"]) >= "2026-01-01":
        raise ValueError("联合微调数据不能进入 2026 locked 区间")
    feature_count = int(contract["feature_count"])
    train_record = next(row for row in cache_manifest["artifacts"] if row["partition"] == "train")
    train_table = pq.read_table(Path(cache_root) / str(train_record["path"]))
    means, scales = _normalizer(train_table, feature_count)
    normalization_sha256 = hashlib.sha256(means.tobytes() + scales.tobytes()).hexdigest()
    datasets = {
        name: JointDataset(cache_root, close_root, name, means, scales) for name in PARTITIONS
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        for name, dataset in datasets.items()
    }
    model = JointEventstreamModel(config, feature_count).to(device)
    pretrained = load_pretrained_backbone(
        model,
        pretrained_checkpoint,
        model_name=config.model,
        seed=config.seed,
        expected_sha256=expected_pretrained_sha256,
    )
    signature = _signature(
        config,
        cache_manifest,
        close_manifest,
        expected_pretrained_sha256,
        source_revision,
    )
    signature["normalization_sha256"] = normalization_sha256
    backbone_parameters = list(model.eventstream.parameters())
    head_parameters = [
        *model.minute_tower.parameters(),
        *model.classifier.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr},
            {"params": head_parameters, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"eventstream-joint-{config.model}.seed{config.seed}"
    last_path = output_root / f"{stem}.last.pt"
    best_path = output_root / f"{stem}.best.pt"
    history_path = output_root / f"train-history.{stem}.json"
    result_path = output_root / f"result.{stem}.json"
    start_epoch = 0
    best_value = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if config.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint.get("experiment") != signature:
            raise ValueError("联合微调 checkpoint 与本次实验合同不同")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        best_value = float(checkpoint["best_selection_value"])
        stale_epochs = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        print(f"从第 {start_epoch} 个 epoch 后继续联合微调")
    if stale_epochs < config.patience:
        for epoch in range(start_epoch, config.epochs):
            epoch_started = time.perf_counter()
            backbone_enabled = epoch >= config.freeze_backbone_epochs
            for parameter in backbone_parameters:
                parameter.requires_grad_(backbone_enabled)
            train_loader = loaders["train"]
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                use_amp=use_amp,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            validation_scores = predict(model, loaders["val"], device, use_amp=use_amp)
            validation = _ranking_metrics(datasets["val"], validation_scores, config)["summary"]
            value = float(validation["daily_rank_ic_mean"])
            comparable = value if math.isfinite(value) else -math.inf
            improved = epoch == 0 or comparable > best_value
            if improved:
                best_value = comparable
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "backbone_trainable": backbone_enabled,
                    "train_loss": train_loss,
                    "validation": validation,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_selection_value": best_value,
                "epochs_without_improvement": stale_epochs,
                "experiment": signature,
                "history": history,
            }
            if improved:
                _atomic_torch(
                    best_path,
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_selection_value": best_value,
                        "experiment": signature,
                    },
                )
            _atomic_torch(last_path, state)
            _atomic_json(history_path, history)
            print(f"epoch {epoch + 1:03d}｜训练损失 {train_loss:.4f}｜验证 Rank IC {value:.4f}")
            if stale_epochs >= config.patience:
                print(f"验证指标连续 {config.patience} 个 epoch 未提升，停止训练")
                break
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    targets_path = Path(cache_root) / str(cache_manifest["portfolio_targets"]["path"])
    evaluations: dict[str, Any] = {}
    for partition in ("val", "test"):
        scores = predict(model, loaders[partition], device, use_amp=use_amp)
        prediction_path = output_root / f"predictions.{stem}.{partition}.parquet"
        _write_predictions(prediction_path, datasets[partition], scores)
        evaluations[partition] = {
            "ranking": _ranking_metrics(datasets[partition], scores, config),
            "portfolio": _portfolio_metrics(datasets[partition], scores, targets_path, config),
            "predictions": {
                "path": prediction_path.name,
                "sha256": file_sha256(prediction_path),
                "rows": len(datasets[partition]),
            },
        }
    result = {
        "mode": "eventstream_joint_finetune",
        "config": {**asdict(config), "top_ks": list(config.top_ks)},
        "experiment": signature,
        "pretrained": pretrained,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "samples": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": int(best["epoch"]),
        "best_selection_value": float(best["best_selection_value"]),
        "evaluation": evaluations,
        "duration_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "test_status": "evaluated",
        "locked_status": "2026_not_accessed",
        "result_file": result_path.name,
    }
    _atomic_json(result_path, result)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))
    return _json_safe(result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行事件流与分钟特征联合微调")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--close-cache", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-pretrained-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--allow-oos", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args(argv)
    config = load_joint_config(arguments.config)
    overrides = {
        name: value
        for name, value in (("seed", arguments.seed), ("epochs", arguments.epochs))
        if value is not None
    }
    if overrides:
        config = JointConfig(**{**asdict(config), **overrides})
    train_joint(
        config,
        cache_root=arguments.cache.expanduser().resolve(),
        close_root=arguments.close_cache.expanduser().resolve(),
        pretrained_checkpoint=arguments.pretrained_checkpoint.expanduser().resolve(),
        expected_pretrained_sha256=arguments.expected_pretrained_sha256,
        output_root=arguments.output.expanduser().resolve(),
        source_revision=arguments.source_revision or git_sha(Path.cwd()),
        allow_oos=arguments.allow_oos,
    )


if __name__ == "__main__":
    main()
