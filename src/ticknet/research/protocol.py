"""研究协议（Research Protocol）：锁定测试集的程序级隔离。

对应 AgentX 论文"权限由程序控制而非语言控制"的核心原则。即使 Agent 试图把
manifest 指向包含锁定测试期（如 2025）的数据，``ResearchProtocol`` 也会在
Runner 执行前拦截，防止样本外信息污染研究过程。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ticknet.research.policy import PolicyViolation


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


@dataclass(frozen=True)
class ResearchProtocol:
    """定义 research 允许的日期边界与 locked 测试范围。"""

    research_cutoff: str = "2024-12-31"
    locked_start: str = "2025-01-01"

    def __post_init__(self) -> None:
        _parse_date(self.research_cutoff)
        _parse_date(self.locked_start)

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
                f"manifest 含锁定测试期数据：最大交易日 {max_trading_date.isoformat()} "
                f"≥ locked_start {self.locked_start}"
            )
