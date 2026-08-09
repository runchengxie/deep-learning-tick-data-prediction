"""研究协议（Research Protocol）：锁定测试集的程序级隔离。

对应 AgentX 论文"权限由程序控制而非语言控制"的核心原则。即使 Agent 试图把
manifest 指向协议声明的锁定测试期，``ResearchProtocol`` 也会在
Runner 执行前拦截，防止样本外信息污染研究过程。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from ticknet.research.policy import PolicyViolation


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


@dataclass(frozen=True)
class ResearchProtocol:
    """定义 research 允许的日期边界与 locked 测试范围。"""

    research_end: str = "2024-12-31"
    validation_end: str = "2025-12-31"
    locked_start: str = "2026-01-01"
    protocol_version: str = "topk-agentx-v1"

    def __post_init__(self) -> None:
        if not self.protocol_version.strip():
            raise ValueError("protocol_version 不能为空")
        research_end = _parse_date(self.research_end)
        validation_end = _parse_date(self.validation_end)
        locked_begin = _parse_date(self.locked_start)
        if not research_end < validation_end < locked_begin:
            raise ValueError("日期边界必须满足 research_end < validation_end < locked_start")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResearchProtocol:
        """从版本化 YAML 读取研究协议，并拒绝未知字段。"""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise PolicyViolation(f"研究协议不存在: {source}")
        with source.open(encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        if not isinstance(values, dict):
            raise PolicyViolation(f"研究协议根节点应为对象: {source}")
        allowed = {"protocol_version", "research_end", "validation_end", "locked_start"}
        unknown = set(values) - allowed
        if unknown:
            raise PolicyViolation(f"研究协议包含未知字段: {sorted(unknown)}")
        try:
            arguments: dict[str, Any] = {
                "protocol_version": str(values["protocol_version"]),
                "research_end": str(values["research_end"]),
                "validation_end": str(values["validation_end"]),
                "locked_start": str(values["locked_start"]),
            }
        except KeyError as error:
            raise PolicyViolation(f"研究协议缺少字段: {error.args[0]}") from error
        return cls(**arguments)

    @property
    def research_end_date(self) -> date:
        return _parse_date(self.research_end)

    @property
    def validation_end_date(self) -> date:
        return _parse_date(self.validation_end)

    @property
    def locked_begin(self) -> date:
        return _parse_date(self.locked_start)

    def manifest_max_trading_date(self, manifest_path: str | Path) -> date:
        """从 manifest 的 samples 中取最大 trading_date。"""
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise PolicyViolation(f"manifest 不存在: {path}")
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise PolicyViolation(f"manifest 根节点应为对象: {path}")
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise PolicyViolation(f"manifest 缺少 samples: {path}")
        max_date = None
        for sample in samples:
            if not isinstance(sample, dict) or "trading_date" not in sample:
                continue
            value = _parse_date(str(sample["trading_date"]))
            if max_date is None or value > max_date:
                max_date = value
        if max_date is None:
            raise PolicyViolation(f"manifest 中无有效 trading_date: {path}")
        return max_date

    def assert_research_safe(self, manifest_path: str | Path) -> None:
        """若 manifest 包含 locked 日期，抛 PolicyViolation。"""
        max_trading_date = self.manifest_max_trading_date(manifest_path)
        if max_trading_date >= self.locked_begin:
            raise PolicyViolation(
                f"协议 {self.protocol_version} 禁止读取锁定测试期："
                f"manifest 最大交易日 {max_trading_date.isoformat()} "
                f"≥ locked_start {self.locked_start}"
            )

    def assert_predictions_safe(self, predictions_path: str | Path) -> None:
        """阻止 audit/cost executor 通过预测 Parquet 绕过 manifest 隔离。"""
        path = Path(predictions_path).expanduser().resolve()
        if not path.is_file():
            raise PolicyViolation(f"预测明细不存在: {path}")
        schema = pq.read_schema(path)
        date_columns = [name for name in ("label_date", "return_end_date") if name in schema.names]
        if not date_columns and "trading_date" in schema.names:
            date_columns = ["trading_date"]
        if not date_columns:
            raise PolicyViolation(f"预测明细缺少 label_date/trading_date: {path}")
        table = pq.read_table(path, columns=date_columns)
        if table.num_rows == 0:
            raise PolicyViolation(f"预测明细为空: {path}")
        try:
            maximum = max(
                _parse_date(str(value))
                for name in date_columns
                for value in table[name].to_pylist()
            )
        except ValueError as error:
            raise PolicyViolation(f"预测明细日期无效: {path}") from error
        if maximum >= self.locked_begin:
            raise PolicyViolation(
                f"协议 {self.protocol_version} 禁止读取锁定测试期："
                f"预测明细最大交易日 {maximum.isoformat()} ≥ locked_start {self.locked_start}"
            )
