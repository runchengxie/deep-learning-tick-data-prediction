"""事件流 day 头预测 -> 正式 Top-K prediction artifact。

把事件流基础模型的 day 头分数与 ``nextday.formal_targets`` 生成的
open-to-following-open 收益 / 可交易状态 / 动态 universe 合并，产出符合
``research.prediction_contract`` 正式契约的预测 parquet（含 metadata），
可直接被 ``import_predictions`` 导入 registry，或被 ``topk_cost_sweep`` 消费。

候选行（in_universe）分数来自事件流模型；状态行（in_universe=False，用于持仓
可交易跟踪）分数记为 0.0（不会被选中排序）。

用法：
    python -m ticknet.eventstream.export \
        --checkpoint checkpoints-eventstream/eventstream.seed0.best.pt \
        --model probe25m --days 20210104 20210105 \
        --pack-root /mnt/.../l2_eventstream/v2 --label-path /mnt/.../label.parquet \
        --basic-root /mnt/.../cn_a_share_level2/basic \
        --benchmark /mnt/.../benchmark_open.parquet --top-n 400 \
        --start 2021-01-04 --end 2021-01-05 --out predictions.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ticknet.eventstream.config import PACK_ROOT, STREAM_DTYPES, day_pack_paths
from ticknet.eventstream.dataset import N_FEATURES, _merge_and_featurize
from ticknet.eventstream.fingerprint import dataset_fingerprint
from ticknet.eventstream.model import build_eventstream_model
from ticknet.nextday.formal_targets import (
    build_formal_next_open_targets,
    load_formal_market_panels,
)
from ticknet.nextday.snapshot_config import SnapshotPreparationConfig
from ticknet.nextday.splits import parse_date
from ticknet.research.prediction_contract import (
    attach_formal_prediction_metadata,
    validate_formal_prediction_artifact,
)
from ticknet.train import resolve_device


@torch.no_grad()
def score_day_head(
    model,
    days: list[int],
    root: Path,
    *,
    seq_len: int,
    min_events: int,
    device: torch.device,
) -> dict[tuple[int, str], float]:
    """对每个 (day, ticker) 的收盘全量上下文输出 day 头分数。"""
    root = Path(root)
    result: dict[tuple[int, str], float] = {}
    model.eval()
    for day in days:
        paths = day_pack_paths(day, root)
        if not all(p.exists() for p in paths.values()):
            continue
        index = np.load(paths["index"], allow_pickle=False)
        total = index["o_len"] + index["t_len"] + index["s_len"]
        tickers = index["tickers"].astype(str)
        mmaps = {
            stream: np.memmap(paths[stream], dtype=STREAM_DTYPES[stream], mode="r")
            for stream in ("order", "trade", "snap")
        }
        for ticker_index in np.where(total >= min_events)[0]:
            ticker_index = int(ticker_index)
            offset = index["o_off"][ticker_index]
            order = mmaps["order"][offset : offset + index["o_len"][ticker_index]]
            offset = index["t_off"][ticker_index]
            trade = mmaps["trade"][offset : offset + index["t_len"][ticker_index]]
            offset = index["s_off"][ticker_index]
            snap = mmaps["snap"][offset : offset + index["s_len"][ticker_index]]
            feats, sid, oid = _merge_and_featurize(
                order, trade, snap, float(index["prev_close"][ticker_index]) * 100.0
            )
            n = len(sid)
            start = max(0, n - (seq_len + 1))
            span = min(seq_len, n - start - 1)
            x = np.zeros((1, seq_len, N_FEATURES), dtype=np.float32)
            x_sid = np.zeros((1, seq_len), dtype=np.int64)
            x_oid = np.zeros((1, seq_len), dtype=np.int64)
            x[0, :span] = feats[start : start + span]
            x_sid[0, :span] = sid[start : start + span]
            x_oid[0, :span] = oid[start : start + span]
            prediction = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(x_sid).to(device),
                torch.from_numpy(x_oid).to(device),
            )
            result[(int(day), str(tickers[ticker_index]))] = float(
                prediction["day"][0, span - 1].item()
            )
    model.train()
    return result


def export_predictions(
    *,
    checkpoint: str | Path,
    model_name: str,
    days: list[int],
    root: Path,
    label_path: str | Path | None,
    seq_len: int,
    min_events: int,
    basic_root: str | Path,
    benchmark_path: str | Path,
    top_n: int,
    start_date: str,
    end_date: str,
    out_path: str | Path,
    device: str = "cpu",
    liquidity_lookback_days: int = 20,
    min_liquidity_observations: int = 15,
) -> tuple[Path, dict]:
    """导出正式 Top-K 预测 artifact，并校验契约。"""
    started_at = time.perf_counter()
    device_resolved = resolve_device(device)
    model = build_eventstream_model(model_name).to(device_resolved)
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    days = [int(d) for d in days]

    scores = score_day_head(
        model, days, root, seq_len=seq_len, min_events=min_events, device=device_resolved
    )
    del model

    panels = load_formal_market_panels(basic_root, end_date=parse_date(end_date))
    config = SnapshotPreparationConfig(
        snapshot_root=str(basic_root),
        basic_root=str(basic_root),
        benchmark_path=str(benchmark_path),
        output_dir=str(Path(out_path).parent),
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        min_history_days=1,
        liquidity_lookback_days=liquidity_lookback_days,
        min_liquidity_observations=min_liquidity_observations,
    )
    targets, _universe, _report = build_formal_next_open_targets(config, panels)

    rows: list[dict] = []
    missing_scores = 0
    for target in targets:
        key = (int(target.trading_date.strftime("%Y%m%d")), target.symbol)
        if target.in_universe:
            if key not in scores:
                missing_scores += 1
                continue
            score = float(scores[key])
        else:
            score = 0.0
        rows.append(
            {
                "symbol": target.symbol,
                "trading_date": target.trading_date.isoformat(),
                "label_date": target.label_date.isoformat(),
                "return_end_date": target.return_end_date.isoformat(),
                "target_return": float(target.target_return),
                "score": score,
                "can_buy": bool(target.can_buy),
                "can_sell": bool(target.can_sell),
                "in_universe": bool(target.in_universe),
            }
        )
    if missing_scores:
        print(f"[export] {missing_scores} 个正式候选缺少模型预测，已跳过")

    fingerprint = dataset_fingerprint(days, root, Path(label_path) if label_path else None)
    table = pa.Table.from_pylist(rows)
    table = attach_formal_prediction_metadata(table, dataset_fingerprint=fingerprint)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    report = validate_formal_prediction_artifact(out, expected_universe_size=top_n)
    print(
        f"[export] {out} rows={report.row_count} candidates={report.candidate_row_count} "
        f"dates={report.label_date_count} ({time.perf_counter() - started_at:.0f}s)"
    )
    return out, report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="probe25m")
    parser.add_argument("--days", type=int, nargs="+", required=True)
    parser.add_argument("--pack-root", default=str(PACK_ROOT))
    parser.add_argument("--label-path", default="")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--min-events", type=int, default=256)
    parser.add_argument("--basic-root", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--top-n", type=int, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    parser.add_argument("--min-liquidity-observations", type=int, default=15)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export_predictions(
        checkpoint=args.checkpoint,
        model_name=args.model,
        days=args.days,
        root=Path(args.pack_root),
        label_path=Path(args.label_path) if args.label_path else None,
        seq_len=args.seq_len,
        min_events=args.min_events,
        basic_root=args.basic_root,
        benchmark_path=args.benchmark,
        top_n=args.top_n,
        start_date=args.start,
        end_date=args.end,
        out_path=args.out,
        device=args.device,
        liquidity_lookback_days=args.liquidity_lookback,
        min_liquidity_observations=args.min_liquidity_observations,
    )


if __name__ == "__main__":
    main()
