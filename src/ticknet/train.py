"""DeepLOB 训练、评估和实验入口。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ticknet.dataset import (
    K_TO_LABEL_COLUMN,
    NUM_CLASSES,
    WINDOW_SIZE,
    FI2010WindowDataset,
    RandomLOBDataset,
)
from ticknet.model import build_model


@dataclass
class Config:
    """一次训练或一组 Setup 1 训练所需的配置。"""

    dataset: str = "random"
    data_path: str | None = None
    meta_path: str | None = None
    protocol: str = "setup2"
    setup1_cfs: list[int] = field(default_factory=lambda: list(range(1, 10)))
    k: int = 10
    epochs: int = 3
    batch_size: int = 32
    lr: float = 0.01
    eps: float = 1.0
    patience: int = 20
    seed: int = 0
    val_frac: float = 0.2
    resume: bool = True
    num_workers: int = 0
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "ticknet"
    device: str = "cpu"

    def validate(self) -> None:
        if self.dataset not in {"random", "fi2010"}:
            raise ValueError(f"dataset 应为 random 或 fi2010，收到 {self.dataset}")
        if self.protocol not in {"setup1", "setup2"}:
            raise ValueError(f"protocol 应为 setup1 或 setup2，收到 {self.protocol}")
        if self.dataset == "fi2010" and (not self.data_path or not self.meta_path):
            raise ValueError("dataset=fi2010 需要 data_path 和 meta_path")
        if not self.setup1_cfs or any(cf not in range(1, 10) for cf in self.setup1_cfs):
            raise ValueError("setup1_cfs 只能包含 1 至 9")
        if len(set(self.setup1_cfs)) != len(self.setup1_cfs):
            raise ValueError("setup1_cfs 不能包含重复值")
        if self.k not in K_TO_LABEL_COLUMN:
            raise ValueError(f"k 应为 {list(K_TO_LABEL_COLUMN)} 中的一个")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs、batch_size 和 patience 应为正整数")
        if self.lr <= 0 or self.eps <= 0:
            raise ValueError("lr 和 eps 应为正数")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")
        if not 0 < self.val_frac < 1:
            raise ValueError("val_frac 应在 0 和 1 之间")


class Metrics(TypedDict):
    """分类评估指标。"""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: list[float]
    per_class_recall: list[float]


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """解析运行设备，并在 CUDA 不可用时给出清楚提示。"""
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"device 应为 cpu 或 cuda，收到 {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        print("未检测到 CUDA，将使用 CPU。")
        return torch.device("cpu")
    return torch.device(requested)


def f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Metrics:
    """计算准确率、F1，以及各类别的精确率和召回率。"""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
    }


def make_dataloaders(
    config: Config,
    *,
    device: torch.device,
    test_cf: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试 DataLoader。"""
    if config.dataset == "random":
        datasets = (
            RandomLOBDataset(num_samples=512, seed=config.seed),
            RandomLOBDataset(num_samples=128, seed=config.seed + 1),
            RandomLOBDataset(num_samples=128, seed=config.seed + 2),
        )
    else:
        if config.data_path is None or config.meta_path is None:
            raise ValueError("FI-2010 数据路径和元数据路径不能为空")
        common = {
            "data_path": config.data_path,
            "meta_path": config.meta_path,
            "k": config.k,
            "window_size": WINDOW_SIZE,
            "protocol": config.protocol,
            "test_cf": test_cf,
            "val_frac": config.val_frac,
        }
        datasets = (
            FI2010WindowDataset(split="train", **common),
            FI2010WindowDataset(split="val", **common),
            FI2010WindowDataset(split="test", **common),
        )

    return (
        DataLoader(
            datasets[0],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
        DataLoader(
            datasets[1],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
        DataLoader(
            datasets[2],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Metrics:
    """评估模型并返回分类指标。"""
    model.eval()
    true_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(features).argmax(dim=1)
            true_batches.append(labels.cpu().numpy())
            predicted_batches.append(predictions.cpu().numpy())
    if not true_batches:
        raise ValueError("评估数据集为空")
    return f1_metrics(
        np.concatenate(true_batches),
        np.concatenate(predicted_batches),
    )


def _run_tag(config: Config, test_cf: int | None) -> str:
    if config.dataset == "random":
        return f"smoke.seed{config.seed}"
    if config.protocol == "setup1":
        return f"setup1.cf{test_cf}.k{config.k}"
    return f"setup2.k{config.k}"


def _experiment_signature(config: Config, test_cf: int | None) -> dict[str, Any]:
    values = asdict(config)
    for key in {
        "epochs",
        "resume",
        "device",
        "num_workers",
        "setup1_cfs",
        "checkpoint_dir",
        "checkpoint_name",
    }:
        values.pop(key)
    values["test_cf"] = test_cf
    return values


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train(config: Config, *, test_cf: int | None = None) -> dict[str, Any]:
    """训练一次模型，保存最近状态和验证集最佳状态。"""
    started_at = time.perf_counter()
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, validation_loader, test_loader = make_dataloaders(
        config,
        device=device,
        test_cf=test_cf,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps)

    run_tag = _run_tag(config, test_cf)
    checkpoint_root = Path(config.checkpoint_dir)
    stem = f"{config.checkpoint_name}.{run_tag}"
    last_path = checkpoint_root / f"{stem}.last.pt"
    best_path = checkpoint_root / f"{stem}.best.pt"
    history_path = checkpoint_root / f"train_history.{run_tag}.json"
    result_path = checkpoint_root / f"result.{run_tag}.json"
    signature = _experiment_signature(config, test_cf)

    start_epoch = 0
    best_validation_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if config.resume and last_path.exists():
        checkpoint = _load_checkpoint(last_path, device)
        if checkpoint.get("experiment") != signature:
            raise ValueError(f"{last_path} 的实验配置与本次运行不同，请更换 checkpoint_name 或目录")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_validation_accuracy = float(checkpoint["best_validation_accuracy"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        print(f"从第 {start_epoch} 个 epoch 后继续训练：{last_path}")

    for epoch in range(start_epoch, config.epochs):
        epoch_started_at = time.perf_counter()
        model.train()
        total_loss = 0.0
        sample_count = 0
        training_started_at = time.perf_counter()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.shape[0]
            sample_count += features.shape[0]

        training_seconds = time.perf_counter() - training_started_at
        train_loss = total_loss / sample_count
        validation_started_at = time.perf_counter()
        validation_metrics = evaluate(model, validation_loader, device)
        validation_seconds = time.perf_counter() - validation_started_at
        epoch_seconds = time.perf_counter() - epoch_started_at
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "training_samples_per_second": sample_count / training_seconds,
            **{f"val_{name}": value for name, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        validation_accuracy = float(validation_metrics["accuracy"])
        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_validation_accuracy": best_validation_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
            "experiment": signature,
            "history": history,
        }
        if improved:
            _save_checkpoint(best_path, state)
        _save_checkpoint(last_path, state)
        _write_json(history_path, history)

        print(
            f"epoch {epoch + 1:03d}｜训练损失 {train_loss:.4f}｜"
            f"验证准确率 {validation_accuracy:.4f}｜"
            f"验证 macro F1 {float(validation_metrics['macro_f1']):.4f}｜"
            f"训练 {training_seconds:.1f}s｜验证 {validation_seconds:.1f}s｜"
            f"{sample_count / training_seconds:.0f} samples/s"
        )
        if epochs_without_improvement >= config.patience:
            print(f"验证准确率连续 {config.patience} 个 epoch 未提升，停止训练。")
            break

    if not best_path.exists():
        raise RuntimeError("没有生成最佳模型检查点")
    best_checkpoint = _load_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    result = {
        "run_tag": run_tag,
        "config": asdict(config),
        "test_cf": test_cf,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
        },
        "duration_seconds": time.perf_counter() - started_at,
        "best_validation_accuracy": best_validation_accuracy,
        "test": test_metrics,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "history": str(history_path),
        "result_file": str(result_path),
    }
    _write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_setup1(config: Config) -> dict[str, Any]:
    """运行论文 Setup 1 的锚定前向九折实验。"""
    results = [train(config, test_cf=cf) for cf in config.setup1_cfs]
    macro_f1 = [float(result["test"]["macro_f1"]) for result in results]
    accuracy = [float(result["test"]["accuracy"]) for result in results]
    summary = {
        "protocol": "setup1",
        "k": config.k,
        "cfs": config.setup1_cfs,
        "per_cf": results,
        "mean_macro_f1": float(np.mean(macro_f1)),
        "std_macro_f1": float(np.std(macro_f1)),
        "mean_accuracy": float(np.mean(accuracy)),
        "std_accuracy": float(np.std(accuracy)),
    }
    output = Path(config.checkpoint_dir) / f"setup1_summary.k{config.k}.json"
    _write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练和评估 DeepLOB")
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=["random", "fi2010"])
    parser.add_argument("--data-path")
    parser.add_argument("--meta-path")
    parser.add_argument("--protocol", choices=["setup1", "setup2"])
    parser.add_argument("--setup1-cfs", type=int, nargs="+")
    parser.add_argument("--k", type=int, choices=[10, 20, 30, 50, 100])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-frac", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.set_defaults(**defaults)
    return parser


def load_config(argv: list[str] | None = None) -> Config:
    """按 Config 默认值、YAML、命令行的顺序合并配置。"""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)

    values: dict[str, Any] = asdict(Config())
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        valid_names = {item.name for item in fields(Config)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)

    parser = _build_parser(values)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = Config(**arguments)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    config = load_config(argv)
    if config.dataset == "fi2010" and config.protocol == "setup1":
        run_setup1(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
