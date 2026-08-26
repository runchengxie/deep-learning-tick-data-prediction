"""L2 事件流无损打包与数据集的数据契约。

设计原则：打包产物是原始 parquet 字段的*无损整数镜像*，不做量化、不做特征
工程。所有归一化（log/bps/z-score）都在数据加载器里完成，改动无需重打包。

逐条保留的原始字段：
    - orders: time_ms, price(分), volume(股), OrderType(raw), LastPrice(分)
    - trades: time_ms, price(分), volume(股), Side(raw), bsflag(raw)
    - snapshot: time_ms, last price, cum volume/turnover/dealnum（差分 + 日基准），
      总买卖深度、加权买卖价、十档 (price, volume, order count) x 买卖双侧
从 ID 关联提炼并新增：
    - 撤单事件 -> 被撤订单年龄 + 原始量
    - 成交 -> 买卖双方挂单年龄
刻意丢弃：
    - 每条事件的 ticker（移入逐日索引）、TradingDay（文件名）
    - OrderID/DealID/BuyID/SellID 原始值（任意标签；关联信息已提炼为年龄特征）
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 可用环境变量 TICKNET_RAW_L2_ROOT 指向其他数据位置（如挂载点或子集拷贝）
RAW_L2_ROOT = Path(
    os.environ.get("TICKNET_RAW_L2_ROOT", "/mnt/data/hdd6t/quant-data-lake/raw/cn_a_share_level2")
)
PACK_ROOT = Path("/mnt/data/hdd6t/quant-data-lake/derived/l2_eventstream/v2")

# ms 相对 09:30:00；集合竞价成交在原始数据里以 -300_000 记录。
SESSION_START_MS = -300_000
MARKET_END_MS = 19_620_000  # 14:57 截断，与 frame/* 约定一致
AGE_UNKNOWN_MS = -1  # 关联目标在流外（例如盘前订单簿）

N_LEVELS = 10

# --- 打包记录布局（little-endian，无 padding） -------------------------------

ORDER_DTYPE = np.dtype(
    [
        ("time_ms", "<i4"),
        ("price", "<i4"),  # 分
        ("volume", "<i4"),  # 股
        ("order_type", "<i2"),  # raw OrderType（SZ: +-1/2/3/11/12/13；SH: 0/10/-1/-11）
        ("last_price", "<i4"),  # 分，raw LastPrice
        ("cancel_age_ms", "<i4"),  # 仅撤单：now - 原订单时间；否则 0 / -1 未知
        ("cancel_orig_vol", "<i4"),  # 仅撤单：原订单量
    ],
    align=False,
)  # itemsize 26

TRADE_DTYPE = np.dtype(
    [
        ("time_ms", "<i4"),
        ("price", "<i4"),  # 分
        ("volume", "<i4"),  # 股
        ("side", "<i1"),  # raw Side
        ("bsflag", "<i1"),  # raw bsflag
        ("buy_age_ms", "<i4"),  # 成交时间 - 买单到达时间；-1 未知
        ("sell_age_ms", "<i4"),
    ],
    align=False,
)  # itemsize 22

SNAP_DTYPE = np.dtype(
    [
        ("time_ms", "<i4"),
        ("last", "<i4"),  # 分
        ("d_volume", "<i4"),  # 累计成交量相对上一快照差分（股）
        ("d_turnover", "<i8"),  # 累计成交额差分（分）
        ("d_dealnum", "<i4"),
        ("total_bidvol", "<f4"),  # 当前总深度（个别股票可能超 i32）
        ("total_askvol", "<f4"),
        ("wbid", "<i4"),  # 加权买价，分
        ("wask", "<i4"),
        ("bid_px", "<i4", (N_LEVELS,)),
        ("ask_px", "<i4", (N_LEVELS,)),
        ("bid_vol", "<i4", (N_LEVELS,)),
        ("ask_vol", "<i4", (N_LEVELS,)),
        ("bid_cnt", "<u2", (N_LEVELS,)),
        ("ask_cnt", "<u2", (N_LEVELS,)),
    ],
    align=False,
)  # itemsize 214

STREAMS = ("order", "trade", "snap")
STREAM_DTYPES = {"order": ORDER_DTYPE, "trade": TRADE_DTYPE, "snap": SNAP_DTYPE}


def day_pack_paths(day: int, root: Path = PACK_ROOT) -> dict[str, Path]:
    d = int(day)
    return {
        "order": root / f"orders_{d}.bin",
        "trade": root / f"trades_{d}.bin",
        "snap": root / f"snaps_{d}.bin",
        "index": root / f"index_{d}.npz",
    }


def day_is_packed(day: int, root: Path = PACK_ROOT) -> bool:
    return all(p.exists() for p in day_pack_paths(day, root).values())


def day_input_files(day: int, raw_root: Path = RAW_L2_ROOT) -> dict[str, Path]:
    """原始输入路径：order/trades 按日，snapshot 按月整文件。"""
    d = str(int(day))
    iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    month = d[:6]
    return {
        "order": raw_root / "order" / month / f"order_{iso}.parquet",
        "trades": raw_root / "trades" / month / f"trades_{iso}.parquet",
        "snap": raw_root / "snapshot" / f"snapshot_{month}.parquet",
    }
