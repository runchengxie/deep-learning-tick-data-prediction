"""snapshot 数据准备的公开入口与兼容转发层。

实现已按职责拆分到 ``snapshot_config`` / ``snapshot_features`` / ``snapshot_targets`` /
``snapshot_io`` / ``snapshot_cli`` 五个子模块。本模块仅做再导出，保持对历史 import
路径（``ticknet.nextday.raw_snapshot``）与 ``pyproject.toml`` CLI 入口的兼容。
"""

from __future__ import annotations

from ticknet.nextday.labels import DailyBar, NextDayTarget
from ticknet.nextday.snapshot_cli import (
    load_snapshot_config,
    main,
    prepare_snapshot_dataset,
)
from ticknet.nextday.snapshot_config import (
    PRICE_INDICES,
    RAW_FEATURE_COLUMNS,
    SHANGHAI_SHENZHEN_STOCK,
    VOLUME_INDICES,
    DailyPanel,
    ExtractionReport,
    SnapshotPreparationConfig,
)
from ticknet.nextday.snapshot_features import (
    build_dynamic_universe,
    normalize_lob_events,
    valid_lob_event_rows,
)
from ticknet.nextday.snapshot_io import (
    iter_snapshot_samples,
    load_market_panels,
    read_wide_daily_panel,
)
from ticknet.nextday.snapshot_targets import (
    build_snapshot_targets,
    read_benchmark_open_close_returns,
)

__all__ = [
    "PRICE_INDICES",
    "RAW_FEATURE_COLUMNS",
    "SHANGHAI_SHENZHEN_STOCK",
    "VOLUME_INDICES",
    "DailyBar",
    "DailyPanel",
    "ExtractionReport",
    "NextDayTarget",
    "SnapshotPreparationConfig",
    "build_dynamic_universe",
    "build_snapshot_targets",
    "iter_snapshot_samples",
    "load_market_panels",
    "load_snapshot_config",
    "main",
    "normalize_lob_events",
    "prepare_snapshot_dataset",
    "read_benchmark_open_close_returns",
    "read_wide_daily_panel",
    "valid_lob_event_rows",
]
