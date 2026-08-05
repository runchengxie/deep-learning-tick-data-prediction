"""snapshot 数据准备配置与共享常量。

从 ``raw_snapshot.py`` 拆出，集中管理十档盘口特征列定义、动态股票池配置与
提取质量计数。被 ``snapshot_features`` / ``snapshot_targets`` / ``snapshot_io`` /
``snapshot_cli`` 共享，自身不依赖其他子模块。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ticknet.nextday.splits import parse_date

RAW_FEATURE_COLUMNS = tuple(
    name
    for level in range(1, 11)
    for name in (
        f"AskPrice{level}",
        f"AskVolume{level}",
        f"BidPrice{level}",
        f"BidVolume{level}",
    )
)
PRICE_INDICES = np.asarray(
    [index for index in range(len(RAW_FEATURE_COLUMNS)) if index % 2 == 0],
    dtype=np.int64,
)
VOLUME_INDICES = np.asarray(
    [index for index in range(len(RAW_FEATURE_COLUMNS)) if index % 2 == 1],
    dtype=np.int64,
)
SHANGHAI_SHENZHEN_STOCK = re.compile(r"^(?:000|001|002|003|300|301|600|601|603|605|688|689)\d{3}$")


@dataclass(frozen=True)
class DailyPanel:
    """日期 × 股票的宽表日线矩阵。"""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.dates), len(self.symbols)):
            raise ValueError("日线矩阵形状与日期、股票轴不一致")
        if tuple(sorted(set(self.dates))) != self.dates:
            raise ValueError("日线日期必须严格递增")
        if tuple(sorted(set(self.symbols))) != self.symbols:
            raise ValueError("日线股票代码必须严格递增")


@dataclass(frozen=True)
class SnapshotPreparationConfig:
    """真实 snapshot 端到端数据准备配置。"""

    snapshot_root: str
    basic_root: str
    benchmark_path: str
    output_dir: str
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    signal_time_ms: int = 19_500_000
    scan_start_time_ms: int = 18_000_000
    chunks_per_sample: int = 2
    chunk_size: int = 100
    min_valid_events: int = 150
    top_n: int = 400
    min_history_days: int = 120
    liquidity_lookback_days: int = 20
    min_liquidity_observations: int = 15
    lower_quantile: float = 0.2
    upper_quantile: float = 0.8
    min_cross_section: int = 20
    samples_per_shard: int = 2048
    storage_dtype: str = "float16"
    price_scale_bps: float = 100.0
    volume_log_scale: float = 16.0
    normalized_clip: float = 32.0

    def validate(self) -> None:
        start = parse_date(self.start_date)
        end = parse_date(self.end_date)
        if end < start:
            raise ValueError("end_date 不能早于 start_date")
        for field_name in ("snapshot_root", "basic_root", "benchmark_path", "output_dir"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} 不能为空")
        if not 0 <= self.scan_start_time_ms < self.signal_time_ms:
            raise ValueError("扫描起点必须早于信号时点且不能为负数")
        if self.chunks_per_sample < 1 or self.chunk_size < 1:
            raise ValueError("chunks_per_sample 和 chunk_size 应为正整数")
        total_events = self.chunks_per_sample * self.chunk_size
        if not 1 <= self.min_valid_events <= total_events:
            raise ValueError("min_valid_events 必须位于 1 和总事件数之间")
        if self.top_n < 1 or self.min_history_days < 1:
            raise ValueError("top_n 和 min_history_days 应为正整数")
        if self.liquidity_lookback_days < 1:
            raise ValueError("liquidity_lookback_days 应为正整数")
        if not 1 <= self.min_liquidity_observations <= self.liquidity_lookback_days:
            raise ValueError("min_liquidity_observations 超出流动性窗口")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype 应为 float16 或 float32")
        if self.samples_per_shard < 1:
            raise ValueError("samples_per_shard 应为正整数")
        if self.price_scale_bps <= 0 or self.volume_log_scale <= 0:
            raise ValueError("固定归一化尺度必须为正数")
        if self.normalized_clip <= 0:
            raise ValueError("normalized_clip 必须为正数")


@dataclass
class ExtractionReport:
    """原始盘口提取期间累积的数据质量计数。"""

    requested_targets: int = 0
    written_samples: int = 0
    missing_snapshot: int = 0
    insufficient_events: int = 0
    invalid_lob_rows: int = 0
    scanned_row_groups: int = 0
    skipped_row_groups: int = 0
    monthly_file_errors: int = 0
    daily_fallback_files: int = 0
    daily_fallback_months: list[str] = field(default_factory=list)


def _yyyymmdd(value: object) -> date:
    raw = str(value).replace("-", "")
    if len(raw) != 8:
        raise ValueError(f"无效交易日：{value!r}")
    return datetime.strptime(raw, "%Y%m%d").date()


def _valid_stock_symbols(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(name for name in names if SHANGHAI_SHENZHEN_STOCK.fullmatch(name)))
