"""L2 事件流窗口采样器。

一个样本 = 某 (ticker, day) 连续时间窗内，order/trade/snapshot 三流按时间归并
后的事件序列。所有归一化（log/bps）在这里基于原始整数完成，可改不重打包。

特征布局（F=80，不适用处零填充）：
    0  dt_log        log1p(距上一事件毫秒 / 1000)
    1  price_bps     事件价相对滚动中间价（bps/100）
    2  qty_log       委托/成交量，快照 d_volume
    3  side          +1 买 / -1 卖 / 0 无
    4  is_cancel
    5  age1_log      撤单年龄或成交买侧挂单年龄（秒）
    6  age2_log      成交卖侧挂单年龄（秒）
    7  origvol_log   被撤订单原始量
    8  amount_log    成交额 / 快照 d_turnover（分）
    9  spread_bps    快照 L1 价差（bps/100）
    10 imb1          快照 L1 失衡
    11 micro_bias    快照微价-中间价偏差（bps/100）
    12-21 bid_px_bps / 22-31 ask_px_bps / 32-41 bid_vol_log / 42-51 ask_vol_log
    52-61 bid_cnt_log / 62-71 ask_cnt_log
    72 totbid_log 73 totask_log 74 wbid_bps 75 wask_bps
    76 dealnum_log 77 tod_sin 78 tod_cos 79 is_auction
目标：
    下一事件流类型（0 pad/eos, 1 snap, 2 order, 3 trade）
    下一订单类型 id（vocab）、下一 price_bps / dt_log / qty_log
    日标签：由外部 label parquet 提供（day, ticker）的标量，可缺省
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ticknet.eventstream.config import PACK_ROOT, STREAM_DTYPES, day_pack_paths

N_FEATURES = 80
BPS_SCALE = 100.0  # 以 bps/100 存储使量级 O(1)

# raw OrderType -> embedding id（0 = 其他/pad）
ORDER_TYPE_VOCAB = {0: 1, 10: 2, 1: 3, 2: 4, 3: 5, 11: 6, 12: 7, 13: 8, -1: 9, -11: 10}
N_ORDER_TYPES = 12

STREAM_SNAP, STREAM_ORDER, STREAM_TRADE = 1, 2, 3


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x.astype(np.float32), 0.0))


class L2WindowDataset(Dataset):
    """训练模式：随机窗口，股票按事件数比例采样。

    评估模式（``eval_mode=True``）：每个有标签的 (ticker, day) 一个确定性样本，
    取当天最后 ``seq_len`` 个事件（收盘全量上下文），可选 ``eval_tickers`` 抽样。
    """

    def __init__(
        self,
        days: list[int],
        seq_len: int = 4096,
        min_events: int = 1024,
        samples_per_day: int = 4000,
        root: Path = PACK_ROOT,
        label_path: Path | None = None,
        seed: int = 0,
        eval_mode: bool = False,
        eval_tickers: int = 0,
    ):
        self.root = Path(root)
        self.seq_len = int(seq_len)
        self.eval_mode = bool(eval_mode)
        self.rng = np.random.default_rng(seed)
        self.days: list[int] = []
        self.index: dict[int, dict] = {}
        self.mmaps: dict[int, dict] = {}

        labels: dict[int, dict[str, float]] = {}
        if label_path is not None and Path(label_path).exists():
            import pyarrow.parquet as pq

            table = pq.read_table(label_path)
            names = table.column_names
            if "value" in names:
                day_col, tick_cols = "value", [c for c in names if c != "value"]
            else:
                day_col = "__index_level_0__" if "__index_level_0__" in names else names[0]
                tick_cols = [c for c in names if c != day_col]
            day_values = table.column(day_col).to_pylist()
            for row in range(table.num_rows):
                day = int(day_values[row])
                mapping: dict[str, float] = {}
                for ticker in tick_cols:
                    value = table.column(ticker)[row].as_py()
                    if value is not None and np.isfinite(float(value)):
                        mapping[str(ticker)] = float(value)
                labels[day] = mapping
        elif label_path is not None:
            print(
                f"[dataset] WARNING: label file missing ({label_path}), day-label loss will be zero"
            )

        entries: list[tuple[int, int]] = []
        n_labeled = 0
        for day in days:
            paths = day_pack_paths(day, self.root)
            if not all(p.exists() for p in paths.values()):
                continue
            idx = np.load(paths["index"], allow_pickle=False)
            total = idx["o_len"] + idx["t_len"] + idx["s_len"]
            ok = np.where(total >= min_events)[0]
            if len(ok) == 0:
                continue
            self.days.append(day)
            self.index[day] = {k: idx[k] for k in idx.files}
            tk_str = idx["tickers"].astype(str)
            day_labels = labels.get(int(day), {})
            lab = np.array([day_labels.get(str(tk), np.nan) for tk in tk_str], dtype=np.float32)
            self.index[day]["label"] = lab
            n_labeled += int(np.isfinite(lab).sum())
            if self.eval_mode:
                cand = ok[np.isfinite(lab[ok])]
                if eval_tickers and len(cand) > eval_tickers:
                    cand = np.sort(
                        np.random.default_rng(int(day)).choice(
                            cand, size=eval_tickers, replace=False
                        )
                    )
                entries.extend((day, int(t)) for t in cand)
            else:
                weights = total[ok].astype(np.float64)
                weights /= weights.sum()
                picks = self.rng.choice(ok, size=samples_per_day, p=weights)
                entries.extend((day, int(t)) for t in picks)
        if not entries:
            raise RuntimeError(f"no packed days found under {self.root}")
        self.entries = entries
        n_total = sum(len(v["label"]) for v in self.index.values())
        print(
            f"[dataset] {'eval' if self.eval_mode else 'train'} "
            f"{len(self.days)} days, {len(entries)} samples, day-label coverage "
            f"{n_labeled}/{n_total} ({n_labeled / max(n_total, 1):.1%})"
        )

    def _get_mmaps(self, day: int) -> dict:
        mmaps = self.mmaps.get(day)
        if mmaps is None:
            paths = day_pack_paths(day, self.root)
            mmaps = {
                stream: np.memmap(paths[stream], dtype=STREAM_DTYPES[stream], mode="r")
                for stream in ("order", "trade", "snap")
            }
            self.mmaps[day] = mmaps
        return mmaps

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        day, tk = self.entries[index]
        idx = self.index[day]
        mmaps = self._get_mmaps(day)
        order = mmaps["order"][idx["o_off"][tk] : idx["o_off"][tk] + idx["o_len"][tk]]
        trade = mmaps["trade"][idx["t_off"][tk] : idx["t_off"][tk] + idx["t_len"][tk]]
        snap = mmaps["snap"][idx["s_off"][tk] : idx["s_off"][tk] + idx["s_len"][tk]]
        prev_close_cent = float(idx["prev_close"][tk]) * 100.0  # 元 -> 分

        feats, stream_id, otype_id = _merge_and_featurize(order, trade, snap, prev_close_cent)
        n = len(stream_id)
        length = self.seq_len
        need = length + 1  # 需要下一个事件作目标
        if n < need:
            start = 0
        elif self.eval_mode:
            start = n - need  # 全量日上下文：收盘前的最后事件
        else:
            start = int(self.rng.integers(0, n - need + 1))
        end = min(start + need, n)

        x = np.zeros((length, N_FEATURES), dtype=np.float32)
        sid = np.zeros(length + 1, dtype=np.int64)
        oid = np.zeros(length + 1, dtype=np.int64)
        tgt_reg = np.zeros((length, 3), dtype=np.float32)  # next price_bps, dt_log, qty_log
        valid = np.zeros(length, dtype=np.float32)

        span = min(length, end - start - 1)
        window = slice(start, start + span)
        x[:span] = feats[window]
        sid[: span + 1] = stream_id[start : start + span + 1]
        oid[: span + 1] = otype_id[start : start + span + 1]
        nxt = slice(start + 1, start + span + 1)
        tgt_reg[:span, 0] = feats[nxt, 1]
        tgt_reg[:span, 1] = feats[nxt, 0]
        tgt_reg[:span, 2] = feats[nxt, 2]
        valid[:span] = 1.0

        lab = float(idx["label"][tk])
        day_valid = 1.0 if np.isfinite(lab) else 0.0
        tgt_day = lab if day_valid else 0.0

        return (
            torch.from_numpy(x),
            torch.from_numpy(sid[:length]),
            torch.from_numpy(oid[:length]),
            torch.from_numpy(sid[1 : length + 1]),
            torch.from_numpy(oid[1 : length + 1]),
            torch.from_numpy(tgt_reg),
            torch.tensor(tgt_day, dtype=torch.float32),
            torch.tensor(day_valid, dtype=torch.float32),
            torch.from_numpy(valid),
            torch.tensor(day, dtype=torch.int64),
        )


def _merge_and_featurize(order, trade, snap, prev_close_cent: float):
    """三流按时间归并并计算模型特征。"""
    n_o, n_t, n_s = len(order), len(trade), len(snap)
    n = n_o + n_t + n_s
    times = np.concatenate(
        [
            order["time_ms"].astype(np.int64),
            trade["time_ms"].astype(np.int64),
            snap["time_ms"].astype(np.int64),
        ]
    )
    stream = np.concatenate(
        [
            np.full(n_o, STREAM_ORDER, np.int64),
            np.full(n_t, STREAM_TRADE, np.int64),
            np.full(n_s, STREAM_SNAP, np.int64),
        ]
    )
    sorted_idx = np.argsort(times, kind="stable")
    times = times[sorted_idx]
    stream = stream[sorted_idx]

    # 滚动中间价：来自快照（首个快照之前回退 prev_close）
    mid_src = np.zeros(n, dtype=np.float64)
    if n_s:
        bid1 = snap["bid_px"][:, 0].astype(np.float64)
        ask1 = snap["ask_px"][:, 0].astype(np.float64)
        mid_s = np.where((bid1 > 0) & (ask1 > 0), (bid1 + ask1) / 2.0, np.maximum(bid1, ask1))
        mid_s = np.where(mid_s > 0, mid_s, np.nan)
        mid_src[n_o + n_t :] = mid_s
    mid_sorted = mid_src[sorted_idx]
    filled = mid_sorted.copy()
    has = ~np.isnan(filled) & (filled > 0)
    filled[~has] = 0.0
    idx_last = np.maximum.accumulate(np.where(has, np.arange(n), -1))
    ref = np.where(
        idx_last >= 0,
        filled[np.maximum(idx_last, 0)],
        prev_close_cent if prev_close_cent > 0 else np.nan,
    )
    ref = np.where(
        np.isfinite(ref) & (ref > 0),
        ref,
        np.nanmax(ref[np.isfinite(ref)]) if np.isfinite(ref).any() else 1.0,
    )

    def bps(px: np.ndarray, r: np.ndarray) -> np.ndarray:
        out = (px.astype(np.float64) / r - 1.0) * 1e4 / BPS_SCALE
        return np.clip(np.nan_to_num(out, nan=0.0), -50.0, 50.0).astype(np.float32)

    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    otype = np.zeros(n, dtype=np.int64)

    dt = np.diff(times, prepend=times[0])
    feats[:, 0] = _log1p(dt / 1000.0)
    tod = (times / 1000.0) / 19620.0
    feats[:, 77] = np.sin(2 * np.pi * tod).astype(np.float32)
    feats[:, 78] = np.cos(2 * np.pi * tod).astype(np.float32)
    feats[:, 79] = (times < 0).astype(np.float32)

    inv = np.empty(n, dtype=np.int64)
    inv[sorted_idx] = np.arange(n)
    pos_o, pos_t, pos_s = inv[:n_o], inv[n_o : n_o + n_t], inv[n_o + n_t :]

    if n_o:
        r = ref[pos_o]
        ot = order["order_type"].astype(np.int64)
        feats[pos_o, 1] = bps(order["price"], r)
        feats[pos_o, 2] = _log1p(order["volume"])
        is_cancel = np.isin(ot, (-1, -11))
        side = np.where(is_cancel, np.where(ot == -1, 1, -1), np.where(ot >= 10, -1, 1))
        feats[pos_o, 3] = side.astype(np.float32)
        feats[pos_o, 4] = is_cancel.astype(np.float32)
        age = np.where(order["cancel_age_ms"] > 0, order["cancel_age_ms"] / 1000.0, 0.0)
        feats[pos_o, 5] = _log1p(age)
        feats[pos_o, 7] = _log1p(order["cancel_orig_vol"])
        for raw, oid_ in ORDER_TYPE_VOCAB.items():
            otype[pos_o[ot == raw]] = oid_

    if n_t:
        r = ref[pos_t]
        feats[pos_t, 1] = bps(trade["price"], r)
        feats[pos_t, 2] = _log1p(trade["volume"])
        feats[pos_t, 3] = np.where(trade["bsflag"] == 1, 1.0, -1.0).astype(np.float32)
        feats[pos_t, 5] = _log1p(np.maximum(trade["buy_age_ms"], 0) / 1000.0)
        feats[pos_t, 6] = _log1p(np.maximum(trade["sell_age_ms"], 0) / 1000.0)
        feats[pos_t, 8] = _log1p(trade["price"].astype(np.float64) * trade["volume"])

    if n_s:
        r = ref[pos_s]
        feats[pos_s, 1] = bps(snap["last"], r)
        feats[pos_s, 2] = np.sign(snap["d_volume"]) * _log1p(np.abs(snap["d_volume"]))
        feats[pos_s, 8] = _log1p(np.abs(snap["d_turnover"]))
        bid1 = snap["bid_px"][:, 0].astype(np.float64)
        ask1 = snap["ask_px"][:, 0].astype(np.float64)
        bv1 = snap["bid_vol"][:, 0].astype(np.float64)
        av1 = snap["ask_vol"][:, 0].astype(np.float64)
        ok = (bid1 > 0) & (ask1 > 0)
        feats[pos_s, 9] = np.where(ok, (ask1 - bid1) / r * 1e4 / BPS_SCALE, 0.0).astype(np.float32)
        depth = bv1 + av1
        feats[pos_s, 10] = np.where(depth > 0, (bv1 - av1) / np.maximum(depth, 1), 0.0).astype(
            np.float32
        )
        micro = np.where(depth > 0, (ask1 * bv1 + bid1 * av1) / np.maximum(depth, 1), 0.0)
        feats[pos_s, 11] = np.where(ok, (micro / r - 1.0) * 1e4 / BPS_SCALE, 0.0).astype(np.float32)
        for level in range(10):
            feats[pos_s, 12 + level] = bps(snap["bid_px"][:, level], r)
            feats[pos_s, 22 + level] = bps(snap["ask_px"][:, level], r)
            feats[pos_s, 32 + level] = _log1p(snap["bid_vol"][:, level])
            feats[pos_s, 42 + level] = _log1p(snap["ask_vol"][:, level])
            feats[pos_s, 52 + level] = _log1p(snap["bid_cnt"][:, level])
            feats[pos_s, 62 + level] = _log1p(snap["ask_cnt"][:, level])
        feats[pos_s, 72] = _log1p(snap["total_bidvol"])
        feats[pos_s, 73] = _log1p(snap["total_askvol"])
        feats[pos_s, 74] = bps(snap["wbid"], r)
        feats[pos_s, 75] = bps(snap["wask"], r)
        feats[pos_s, 76] = _log1p(snap["d_dealnum"])

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, stream, otype
