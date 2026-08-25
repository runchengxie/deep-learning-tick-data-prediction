"""事件流基础模型训练入口（沿用 nextday 训练约定）。

YAML 配置 -> ``EventstreamConfig`` -> 训练/验证/测试。每 epoch 在训练窗口集上做
多任务下一事件预测（stream/otype/reg + day 头），验证集按日算 day 头 Rank IC，
按 selection_metric 早停，保存 best/last checkpoint 与历史 JSON。resume 会校验
实验签名（含数据集指纹）是否一致。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ticknet.eventstream.config import PACK_ROOT, day_is_packed
from ticknet.eventstream.dataset import L2WindowDataset
from ticknet.eventstream.fingerprint import dataset_fingerprint, file_sha256
from ticknet.eventstream.materialized import (
    MaterializedWindowDataset,
    assert_materialized_compatible,
    load_materialized_manifest,
)
from ticknet.eventstream.model import (
    CONFIGS,
    DAY_SUPERVISION_MODES,
    DAY_SUPERVISION_WEIGHT_VERSION,
    build_eventstream_model,
    compute_loss,
)
from ticknet.train import resolve_device, set_seed


class EventstreamConfig:
    """一次事件流预训练实验的配置。"""

    __slots__ = (
        "amp",
        "batch_size",
        "checkpoint_dir",
        "checkpoint_name",
        "day_supervision_mode",
        "days",
        "device",
        "epochs",
        "eval_tickers",
        "evaluate_test",
        "gradient_accumulation_steps",
        "label_path",
        "lr",
        "materialized_root",
        "materialized_source_revision",
        "min_events",
        "min_symbols_per_day",
        "model",
        "monitor_label_path",
        "monitor_name",
        "num_workers",
        "pack_root",
        "patience",
        "resume",
        "samples_per_day",
        "seed",
        "selection_metric",
        "seq_len",
        "source_revision",
        "target_overlay_root",
        "test_end",
        "test_start",
        "train_end",
        "train_start",
        "use_lob_prefix",
        "use_session_anchors",
        "use_vq",
        "val_end",
        "val_start",
        "vq_codebook_size",
        "vq_dim",
        "vq_loss_weight",
        "weight_decay",
    )

    def __init__(
        self,
        *,
        pack_root: str = str(PACK_ROOT),
        label_path: str = "",
        monitor_label_path: str = "",
        monitor_name: str = "",
        train_start: int = 0,
        train_end: int = 0,
        val_start: int = 0,
        val_end: int = 0,
        test_start: int = 0,
        test_end: int = 0,
        days: tuple[int, ...] = (),
        model: str = "probe25m",
        seq_len: int = 512,
        min_events: int = 256,
        samples_per_day: int = 2000,
        eval_tickers: int = 200,
        evaluate_test: bool = True,
        day_supervision_mode: str = "all",
        use_lob_prefix: bool = False,
        use_session_anchors: bool = False,
        use_vq: bool = False,
        vq_codebook_size: int = 1024,
        vq_dim: int = 64,
        vq_loss_weight: float = 0.25,
        epochs: int = 20,
        batch_size: int = 8,
        lr: float = 3e-4,
        materialized_root: str = "",
        materialized_source_revision: str = "",
        target_overlay_root: str = "",
        weight_decay: float = 0.1,
        patience: int = 4,
        seed: int = 0,
        num_workers: int = 0,
        device: str = "cpu",
        amp: bool = True,
        gradient_accumulation_steps: int = 1,
        resume: bool = True,
        checkpoint_dir: str = "./checkpoints-eventstream",
        checkpoint_name: str = "eventstream",
        selection_metric: str = "daily_rank_ic_mean",
        source_revision: str = "",
        min_symbols_per_day: int = 20,
    ):
        self.pack_root = pack_root
        self.label_path = label_path
        self.monitor_label_path = monitor_label_path
        self.monitor_name = monitor_name
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.days = days
        self.model = model
        self.seq_len = seq_len
        self.min_events = min_events
        self.samples_per_day = samples_per_day
        self.eval_tickers = eval_tickers
        self.evaluate_test = evaluate_test
        self.day_supervision_mode = day_supervision_mode
        self.use_lob_prefix = bool(use_lob_prefix)
        self.use_session_anchors = bool(use_session_anchors)
        self.use_vq = bool(use_vq)
        self.vq_codebook_size = int(vq_codebook_size)
        self.vq_dim = int(vq_dim)
        self.vq_loss_weight = float(vq_loss_weight)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.materialized_root = materialized_root
        self.materialized_source_revision = materialized_source_revision
        self.target_overlay_root = target_overlay_root
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.num_workers = num_workers
        self.device = device
        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.resume = resume
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.selection_metric = selection_metric
        self.source_revision = source_revision
        self.min_symbols_per_day = min_symbols_per_day

    def validate(self) -> None:
        if self.model not in CONFIGS:
            raise ValueError(f"model 应为 {sorted(CONFIGS)} 之一")
        if self.seq_len < 2 or self.min_events < 1 or self.samples_per_day < 1:
            raise ValueError("seq_len、min_events 和 samples_per_day 应为正整数")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs、batch_size 和 patience 应为正整数")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr 应为正数，weight_decay 不能为负数")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps 应为正整数")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 应为 cpu 或 cuda")
        if self.day_supervision_mode not in DAY_SUPERVISION_MODES:
            raise ValueError(f"day_supervision_mode 应为 {sorted(DAY_SUPERVISION_MODES)} 之一")
        if self.use_session_anchors and not self.use_lob_prefix:
            raise ValueError("use_session_anchors 需要同时启用 use_lob_prefix")
        if self.vq_codebook_size < 2:
            raise ValueError("vq_codebook_size 至少为 2")
        if self.vq_dim < 2:
            raise ValueError("vq_dim 至少为 2")
        if self.vq_loss_weight < 0:
            raise ValueError("vq_loss_weight 不能为负数")
        if self.min_symbols_per_day < 2:
            raise ValueError("min_symbols_per_day 至少为 2")
        if self.monitor_label_path and not self.monitor_name:
            raise ValueError("提供 monitor_label_path 时必须提供 monitor_name")
        if self.target_overlay_root and not self.materialized_root:
            raise ValueError("target_overlay_root 只能与 materialized_root 一起使用")
        if self.materialized_source_revision and not self.materialized_root:
            raise ValueError("materialized_source_revision 只能与 materialized_root 一起使用")
        if self.monitor_name and (
            not self.monitor_name.isascii() or not self.monitor_name.replace("_", "").isalnum()
        ):
            raise ValueError("monitor_name 只能包含 ASCII 字母、数字和下划线")
        if (
            not self.materialized_root
            and not self.days
            and not (0 < self.train_start < self.train_end)
        ):
            raise ValueError("需要显式 days 或有效的 train_start/train_end 区间")
        if self.source_revision and (
            not self.source_revision.isascii() or len(self.source_revision) < 7
        ):
            raise ValueError("source_revision 应为至少 7 位 ASCII 标识")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> EventstreamConfig:
        unknown = set(raw) - set(cls.__slots__)
        if unknown:
            raise ValueError(f"配置包含未知字段：{sorted(unknown)}")
        data = dict(raw)
        if isinstance(data.get("days"), list):
            data["days"] = tuple(data["days"])
        return cls(**data)


def list_packed_days(start: int, end: int, root: Path) -> list[int]:
    days: list[int] = []
    for f in sorted(root.glob("index_*.npz")):
        d = int(f.stem.split("_")[1])
        if start <= d <= end and day_is_packed(d, root):
            days.append(d)
    return days


def _resolve_days(config: EventstreamConfig, root: Path) -> list[int]:
    if config.days:
        return list(config.days)
    return list_packed_days(config.train_start, config.train_end, root)


def _window_dataset_kwargs(config: EventstreamConfig) -> dict[str, bool]:
    return {
        "use_lob_prefix": config.use_lob_prefix,
        "use_session_anchors": config.use_session_anchors,
    }


def make_dataloaders(
    config: EventstreamConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    if config.materialized_root:
        root = Path(config.materialized_root)
        manifest = load_materialized_manifest(root)
        assert_materialized_compatible(manifest, config)
        train_ds = MaterializedWindowDataset(
            root,
            "train",
            target_overlay_root=(
                Path(config.target_overlay_root) if config.target_overlay_root else None
            ),
        )
        val_ds = MaterializedWindowDataset(root, "validation")
        test_ds = MaterializedWindowDataset(root, "oos") if config.evaluate_test else None
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
            )
            if test_ds is not None
            else None
        )
        return train_loader, val_loader, test_loader

    root = Path(config.pack_root)
    label_path = Path(config.label_path) if config.label_path else None
    train_ds = L2WindowDataset(
        _resolve_days(config, root),
        seq_len=config.seq_len,
        min_events=config.min_events,
        samples_per_day=config.samples_per_day,
        root=root,
        label_path=label_path,
        seed=config.seed,
        **_window_dataset_kwargs(config),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    val_loader = test_loader = None
    if config.val_start:
        val_days = list_packed_days(config.val_start, config.val_end, root)
        val_ds = L2WindowDataset(
            val_days,
            seq_len=config.seq_len,
            min_events=config.min_events,
            root=root,
            label_path=label_path,
            eval_mode=True,
            eval_tickers=config.eval_tickers,
            **_window_dataset_kwargs(config),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
    if config.test_start and config.evaluate_test:
        test_days = list_packed_days(config.test_start, config.test_end, root)
        test_ds = L2WindowDataset(
            test_days,
            seq_len=config.seq_len,
            min_events=config.min_events,
            root=root,
            label_path=label_path,
            eval_mode=True,
            eval_tickers=0,
            **_window_dataset_kwargs(config),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
    return train_loader, val_loader, test_loader


def make_monitor_dataloaders(
    config: EventstreamConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader | None, DataLoader | None]:
    """用同一批确定性收盘窗口加载只监控标签，不参与 checkpoint 选择。"""
    if config.materialized_root:
        root = Path(config.materialized_root)
        validation = DataLoader(
            MaterializedWindowDataset(root, "monitor_validation"),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        oos = (
            DataLoader(
                MaterializedWindowDataset(root, "monitor_oos"),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=device.type == "cuda",
            )
            if config.evaluate_test
            else None
        )
        return validation, oos
    if not config.monitor_label_path:
        return None, None
    root = Path(config.pack_root)
    label_path = Path(config.monitor_label_path)

    def build(start: int, end: int, eval_tickers: int) -> DataLoader | None:
        if not start:
            return None
        dataset = L2WindowDataset(
            list_packed_days(start, end, root),
            seq_len=config.seq_len,
            min_events=config.min_events,
            root=root,
            label_path=label_path,
            eval_mode=True,
            eval_tickers=eval_tickers,
            **_window_dataset_kwargs(config),
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

    return (
        build(config.val_start, config.val_end, config.eval_tickers),
        build(config.test_start, config.test_end, 0),
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "daily_rank_ic_mean": math.nan,
        "daily_rank_ic_std": math.nan,
        "ic_ir": math.nan,
        "daily_spread_mean": math.nan,
        "n_days": 0,
        "n_samples": 0,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(values.size)
        out[order] = np.arange(values.size)
        return out

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


@torch.no_grad()
def evaluate_rank_ic(
    model: nn.Module,
    dataloader: DataLoader | None,
    device: torch.device,
    *,
    min_symbols_per_day: int,
) -> dict[str, Any]:
    """day 头在最后一个有效位置（收盘全量上下文）的逐日 Spearman Rank IC。"""
    if dataloader is None:
        return _empty_metrics()
    dataset = dataloader.dataset
    if not isinstance(dataset, (L2WindowDataset, MaterializedWindowDataset)):
        raise TypeError("事件流评估需要原始或物化窗口数据集")
    if len(dataset) == 0:
        return _empty_metrics()
    preds: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    day_ids: list[np.ndarray] = []
    model.eval()
    use_amp = torch.cuda.is_available()
    for batch in dataloader:
        x, sid, oid, _, _, _, tgt_day, day_valid, valid, day = (
            b.to(device, non_blocking=True) for b in batch
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x, sid, oid)
        last = valid.sum(-1).clamp(min=1).long() - 1
        score = out["day"].float().gather(1, last[:, None]).squeeze(1)
        keep = day_valid > 0
        preds.append(score[keep].cpu().numpy())
        labs.append(tgt_day[keep].cpu().numpy())
        day_ids.append(day[keep].cpu().numpy())
    model.train()
    if not preds:
        return {
            **dict.fromkeys(
                ("daily_rank_ic_mean", "daily_rank_ic_std", "ic_ir", "daily_spread_mean"), math.nan
            ),
            "n_days": 0,
            "n_samples": 0,
        }
    pred = np.concatenate(preds)
    lab = np.concatenate(labs)
    day = np.concatenate(day_ids)

    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p, val, d in zip(pred, lab, day, strict=True):
        grouped[int(d)].append((float(p), float(val)))
    daily_ics: list[float] = []
    spreads: list[float] = []
    for rows in grouped.values():
        if len(rows) < min_symbols_per_day:
            continue
        ps = np.asarray([r[0] for r in rows])
        ls = np.asarray([r[1] for r in rows])
        ic = _spearman(ps, ls)
        if math.isfinite(ic):
            daily_ics.append(ic)
        order = np.argsort(ps, kind="mergesort")
        tail = max(1, int(np.floor(len(ps) * 0.1)))
        spreads.append(float(ls[order[-tail:]].mean() - ls[order[:tail]].mean()))
    mean = float(np.mean(daily_ics)) if daily_ics else math.nan
    std = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else math.nan
    return {
        "daily_rank_ic_mean": mean,
        "daily_rank_ic_std": std,
        "ic_ir": float(mean / std) if math.isfinite(std) and std > 0 else math.nan,
        "daily_spread_mean": float(np.mean(spreads)) if spreads else math.nan,
        "n_days": len(daily_ics),
        "n_samples": len(pred),
    }


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


def _experiment_signature(
    config: EventstreamConfig,
    fingerprint: str,
    monitor_label_fingerprint: str | None = None,
) -> dict[str, Any]:
    signature = config.to_dict()
    for name in (
        "epochs",
        "resume",
        "device",
        "num_workers",
        "amp",
        "gradient_accumulation_steps",
        "evaluate_test",
    ):
        signature.pop(name)
    if not config.monitor_label_path and not config.materialized_root:
        signature.pop("monitor_label_path")
        signature.pop("monitor_name")
    if not config.target_overlay_root:
        signature.pop("target_overlay_root")
    if not config.materialized_source_revision:
        signature.pop("materialized_source_revision")
    signature["dataset_fingerprint"] = fingerprint
    signature["day_supervision_weight_version"] = DAY_SUPERVISION_WEIGHT_VERSION
    if monitor_label_fingerprint is not None:
        signature["monitor_label_fingerprint"] = monitor_label_fingerprint
    return signature


def _checkpoint_matches_experiment(checkpoint: dict[str, Any], expected: dict[str, Any]) -> bool:
    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, dict):
        return False
    normalized = dict(experiment)
    normalized.setdefault("day_supervision_mode", "all")
    normalized.setdefault("day_supervision_weight_version", DAY_SUPERVISION_WEIGHT_VERSION)
    normalized.setdefault("use_lob_prefix", False)
    normalized.setdefault("use_session_anchors", False)
    normalized.setdefault("use_vq", False)
    normalized.setdefault("vq_codebook_size", 1024)
    normalized.setdefault("vq_dim", 64)
    normalized.setdefault("vq_loss_weight", 0.25)
    return normalized == expected


def _checkpoint_paths(config: EventstreamConfig) -> tuple[str, Path, Path, Path, Path]:
    root = Path(config.checkpoint_dir)
    stem = f"{config.checkpoint_name}.seed{config.seed}"
    return (
        stem,
        root / f"{stem}.last.pt",
        root / f"{stem}.best.pt",
        root / f"train_history.{stem}.json",
        root / f"result.{stem}.json",
    )


def train(
    config: EventstreamConfig,
    *,
    expected_parameter_count: int | None = None,
) -> dict[str, Any]:
    """训练事件流基础模型并在验证/测试区间上评估 day 头 Rank IC。"""
    started_at = time.perf_counter()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = make_dataloaders(config, device=device)
    monitor_val_loader, monitor_test_loader = make_monitor_dataloaders(config, device=device)

    root = Path(config.pack_root)
    if config.materialized_root:
        materialized_manifest = load_materialized_manifest(Path(config.materialized_root))
        assert_materialized_compatible(materialized_manifest, config)
        fingerprint = str(materialized_manifest["dataset_fingerprint"])
    else:
        fingerprint = dataset_fingerprint(
            _resolve_days(config, root),
            root=root,
            label_path=Path(config.label_path) if config.label_path else None,
        )
    monitor_label_fingerprint = (
        file_sha256(config.monitor_label_path)
        if config.monitor_label_path and not config.materialized_root
        else None
    )
    signature = _experiment_signature(config, fingerprint, monitor_label_fingerprint)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, (L2WindowDataset, MaterializedWindowDataset)):
        raise TypeError("训练数据集类型无效")
    target_overlay_fingerprint = (
        train_dataset.target_overlay_fingerprint
        if isinstance(train_dataset, MaterializedWindowDataset)
        else None
    )
    if target_overlay_fingerprint is not None:
        signature["target_overlay_fingerprint"] = target_overlay_fingerprint

    model = build_eventstream_model(
        config.model,
        use_vq=config.use_vq,
        vq_codebook_size=config.vq_codebook_size,
        vq_dim=config.vq_dim,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise ValueError(f"事件流参数量不匹配：{parameter_count} != {expected_parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    _stem, last_path, best_path, history_path, result_path = _checkpoint_paths(config)
    start_epoch = 0
    best_selection_value = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if config.resume and last_path.exists():
        checkpoint = _load_checkpoint(last_path, device)
        if not _checkpoint_matches_experiment(checkpoint, signature):
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
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            x, sid, oid, tgt_sid, tgt_oid, tgt_reg, tgt_day, day_valid, valid, _ = (
                b.to(device, non_blocking=True) for b in batch
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x, sid, oid)
                loss, _metrics = compute_loss(
                    out,
                    tgt_sid,
                    tgt_oid,
                    tgt_reg,
                    tgt_day,
                    day_valid,
                    valid,
                    day_supervision_mode=config.day_supervision_mode,
                    vq_loss_weight=config.vq_loss_weight,
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
            total_loss += loss.item() * x.shape[0]
            sample_count += x.shape[0]
        if sample_count == 0:
            raise ValueError("训练数据集为空")

        validation = evaluate_rank_ic(
            model, val_loader, device, min_symbols_per_day=config.min_symbols_per_day
        )
        monitor_validation = evaluate_rank_ic(
            model,
            monitor_val_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
        )
        raw_selection = validation[config.selection_metric]
        selection_value = float(raw_selection) if raw_selection is not None else math.nan
        comparable = selection_value if math.isfinite(selection_value) else -math.inf
        improved = epoch == 0 or comparable > best_selection_value
        if improved:
            best_selection_value = comparable
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / sample_count,
            "epoch_seconds": time.perf_counter() - epoch_started_at,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        if monitor_val_loader is not None:
            record.update(
                {
                    f"val_{config.monitor_name}_{key}": value
                    for key, value in monitor_validation.items()
                }
            )
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
        }
        if improved:
            _atomic_torch_save(best_path, state)
        _atomic_torch_save(last_path, state)
        _atomic_json(history_path, _json_safe(history))
        print(
            f"epoch {epoch + 1:03d}｜训练损失 {record['train_loss']:.4f}｜"
            f"验证 Rank IC {float(validation['daily_rank_ic_mean']):.4f}"
        )
        if epochs_without_improvement >= config.patience:
            print(f"验证指标连续 {config.patience} 个 epoch 未提升，停止训练。")
            break

    best = _load_checkpoint(best_path, device)
    model.load_state_dict(best["model"])
    val_metrics = evaluate_rank_ic(
        model, val_loader, device, min_symbols_per_day=config.min_symbols_per_day
    )
    test_metrics = (
        evaluate_rank_ic(model, test_loader, device, min_symbols_per_day=config.min_symbols_per_day)
        if config.evaluate_test
        else None
    )
    monitor_val_metrics = evaluate_rank_ic(
        model,
        monitor_val_loader,
        device,
        min_symbols_per_day=config.min_symbols_per_day,
    )
    monitor_test_metrics = (
        evaluate_rank_ic(
            model,
            monitor_test_loader,
            device,
            min_symbols_per_day=config.min_symbols_per_day,
        )
        if config.evaluate_test
        else None
    )
    dataset_types = (L2WindowDataset, MaterializedWindowDataset)
    val_count = 0
    if val_loader is not None:
        if not isinstance(val_loader.dataset, dataset_types):
            raise TypeError("验证数据集类型无效")
        val_count = len(val_loader.dataset)
    test_count = 0
    if test_loader is not None:
        if not isinstance(test_loader.dataset, dataset_types):
            raise TypeError("测试数据集类型无效")
        test_count = len(test_loader.dataset)
    result = {
        "mode": "eventstream_train",
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "samples": {"train": len(train_dataset), "val": val_count, "test": test_count},
        "environment": _environment(device),
        "duration_seconds": time.perf_counter() - started_at,
        "dataset_fingerprint": fingerprint,
        "target_overlay_fingerprint": target_overlay_fingerprint,
        "monitor_label_fingerprint": monitor_label_fingerprint,
        "best_epoch": int(best["epoch"]),
        "best_selection_value": float(best["best_selection_value"]),
        "val": val_metrics,
        "test": test_metrics,
        "test_status": "evaluated" if config.evaluate_test else "not_evaluated",
        "monitor": (
            None
            if monitor_val_loader is None
            else {
                "name": config.monitor_name,
                "label_path": config.monitor_label_path,
                "val": monitor_val_metrics,
                "test": monitor_test_metrics,
            }
        ),
        "result_file": str(result_path),
    }
    safe_result = _json_safe(result)
    _atomic_json(result_path, safe_result)
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))
    return safe_result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--day-supervision-mode",
        choices=sorted(DAY_SUPERVISION_MODES),
    )
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--expected-parameter-count", type=int)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("事件流配置应为 YAML 对象")
    overrides = {
        "seed": args.seed,
        "epochs": args.epochs,
        "source_revision": args.source_revision,
        "evaluate_test": args.evaluate_test,
        "day_supervision_mode": args.day_supervision_mode,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_name": args.checkpoint_name,
    }
    for name, value in overrides.items():
        if value is not None:
            raw[name] = value
    config = EventstreamConfig.from_mapping(dict(raw))
    train(config, expected_parameter_count=args.expected_parameter_count)


if __name__ == "__main__":
    main()
