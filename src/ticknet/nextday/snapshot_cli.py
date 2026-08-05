"""snapshot 数据准备 CLI 与编排：拼接股票池、标签、盘口提取并写出分片。

从 ``raw_snapshot.py`` 拆出顶层编排逻辑。依赖 ``snapshot_config`` / ``snapshot_io`` /
``snapshot_targets``，是子模块的汇聚点。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ticknet.nextday.io import write_sharded_dataset
from ticknet.nextday.snapshot_config import ExtractionReport, SnapshotPreparationConfig
from ticknet.nextday.snapshot_io import _write_report, iter_snapshot_samples, load_market_panels
from ticknet.nextday.snapshot_targets import build_snapshot_targets


def prepare_snapshot_dataset(
    config: SnapshotPreparationConfig,
) -> tuple[Path, dict[str, Any]]:
    """执行股票池、标签、原始盘口提取并写出 Colab 可搬运分片。"""
    config.validate()
    open_panel, close_panel, volume_panel = load_market_panels(config.basic_root)
    targets, universe = build_snapshot_targets(config, open_panel, close_panel, volume_panel)
    if not targets:
        raise ValueError("指定日期和股票池没有生成任何次日标签")
    report = ExtractionReport()
    manifest = write_sharded_dataset(
        iter_snapshot_samples(config, targets, report),
        config.output_dir,
        chunks_per_sample=config.chunks_per_sample,
        chunk_size=config.chunk_size,
        samples_per_shard=config.samples_per_shard,
        storage_dtype=config.storage_dtype,
        metadata={
            "source": "cn_a_share_level2_snapshot",
            "signal_time_ms": config.signal_time_ms,
            "scan_start_time_ms": config.scan_start_time_ms,
            "min_valid_events": config.min_valid_events,
            "normalization": {
                "price": "first_selected_mid_relative_bps",
                "price_scale_bps": config.price_scale_bps,
                "volume": "log1p",
                "volume_log_scale": config.volume_log_scale,
                "clip": config.normalized_clip,
            },
        },
    )
    selected_counts = [len(symbols) for symbols in universe.values()]
    audit: dict[str, Any] = {
        "config": asdict(config),
        "extraction": asdict(report),
        "universe": {
            "dates": len(selected_counts),
            "minimum": min(selected_counts, default=0),
            "median": float(np.median(selected_counts)) if selected_counts else 0.0,
            "maximum": max(selected_counts, default=0),
        },
        "manifest": str(manifest),
    }
    _write_report(Path(config.output_dir) / "data-audit.json", audit)
    return manifest, audit


def _build_parser(defaults: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从沪深十档 snapshot 月度 Parquet 生成端到端 DeepLOB 分片"
    )
    parser.add_argument("--config")
    for name, value in defaults.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, int):
            parser.add_argument(option, type=int)
        elif isinstance(value, float):
            parser.add_argument(option, type=float)
        else:
            parser.add_argument(option)
    parser.set_defaults(**defaults)
    return parser


def load_snapshot_config(argv: list[str] | None = None) -> SnapshotPreparationConfig:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    probe_args, _ = probe.parse_known_args(argv)
    values = asdict(
        SnapshotPreparationConfig(
            snapshot_root="",
            basic_root="",
            benchmark_path="",
            output_dir="",
        )
    )
    if probe_args.config:
        with open(probe_args.config, encoding="utf-8") as file:
            file_values = yaml.safe_load(file) or {}
        if not isinstance(file_values, dict):
            raise SystemExit("snapshot YAML 根节点应为对象")
        valid_names = {item.name for item in fields(SnapshotPreparationConfig)}
        unknown = set(file_values) - valid_names
        if unknown:
            raise SystemExit(f"snapshot YAML 含未知字段：{sorted(unknown)}")
        values.update(file_values)
    parser = _build_parser(values)
    arguments = vars(parser.parse_args(argv))
    arguments.pop("config", None)
    config = SnapshotPreparationConfig(**arguments)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    manifest, audit = prepare_snapshot_dataset(load_snapshot_config(argv))
    print(f"已写入 {audit['extraction']['written_samples']:,} 个端到端股票日样本")
    print(f"数据清单：{manifest}")
