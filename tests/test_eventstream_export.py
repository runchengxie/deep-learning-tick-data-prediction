"""事件流 day 头预测导出为正式 Top-K artifact 的测试。"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ticknet.eventstream.export import export_predictions
from ticknet.eventstream.model import build_eventstream_model
from ticknet.research.portfolio import load_portfolio_predictions
from ticknet.research.prediction_contract import validate_formal_prediction_artifact

PANEL_DATES = [20201231, 20210104, 20210105, 20210106]
SYMBOL = "600000"


def _write_wide(path: Path, values: list[float]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [{"value": day, SYMBOL: value} for day, value in zip(PANEL_DATES, values, strict=True)]
        ),
        path,
    )


def _make_basic(tmp_path: Path) -> Path:
    basic = tmp_path / "basic"
    basic.mkdir(parents=True)
    prices = [10.0] * len(PANEL_DATES)
    volumes = [1000.0] * len(PANEL_DATES)
    for name in ("open", "high", "low", "close"):
        _write_wide(basic / f"{name}_data.parquet", prices)
    _write_wide(basic / "volume_data.parquet", volumes)
    _write_wide(basic / "st_data.parquet", [0.0] * len(PANEL_DATES))
    return basic


def _make_benchmark(tmp_path: Path) -> Path:
    path = tmp_path / "benchmark.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"trade_date": str(day), "open": 1000.0} for day in PANEL_DATES]),
        path,
    )
    return path


def _make_checkpoint(path: Path) -> None:
    model = build_eventstream_model("smoke")
    torch.save({"model": model.state_dict()}, path)


def _export_args(packed_day, tmp_path, ckpt, out):
    return {
        "checkpoint": ckpt,
        "model_name": "smoke",
        "days": [packed_day["day"]],
        "root": Path(packed_day["pack_root"]),
        "label_path": Path(packed_day["label_path"]),
        "seq_len": 8,
        "min_events": 2,
        "basic_root": _make_basic(tmp_path),
        "benchmark_path": _make_benchmark(tmp_path),
        "top_n": 1,
        "start_date": "2021-01-04",
        "end_date": "2021-01-06",
        "out_path": out,
        "device": "cpu",
        "liquidity_lookback_days": 1,
        "min_liquidity_observations": 1,
    }


class TestExport:
    def test_exports_formal_artifact(self, packed_day, tmp_path):
        ckpt = tmp_path / "best.pt"
        _make_checkpoint(ckpt)
        out = tmp_path / "predictions.parquet"
        _out, report = export_predictions(**_export_args(packed_day, tmp_path, ckpt, out))
        assert report["candidate_row_count"] == 1
        assert report["row_count"] == 1
        assert report["expected_universe_size"] == 1
        assert report["label_date_count"] == 1

        metadata = dict(pq.read_metadata(out).metadata)
        assert metadata[b"ticknet.target_return_contract"] == b"next_open_to_following_open"
        assert b"ticknet.dataset_fingerprint" in metadata

    def test_artifact_consumable_by_portfolio(self, packed_day, tmp_path):
        ckpt = tmp_path / "best.pt"
        _make_checkpoint(ckpt)
        out = tmp_path / "predictions.parquet"
        export_predictions(**_export_args(packed_day, tmp_path, ckpt, out))
        predictions = load_portfolio_predictions(out)
        assert len(predictions) == 1
        assert predictions[0].symbol == SYMBOL
        assert predictions[0].in_universe is True
        assert predictions[0].label_date > predictions[0].trading_date
        report = validate_formal_prediction_artifact(out, expected_universe_size=1)
        assert report.candidate_row_count == 1

    def test_export_without_scored_day_raises(self, packed_day, tmp_path):
        # 评分日与信号区间不匹配 -> 没有任何可导出目标 -> 报错
        ckpt = tmp_path / "best.pt"
        _make_checkpoint(ckpt)
        out = tmp_path / "predictions.parquet"
        with pytest.raises(ValueError, match="没有生成"):
            export_predictions(
                checkpoint=ckpt,
                model_name="smoke",
                days=[20210105],
                root=Path(packed_day["pack_root"]),
                label_path=Path(packed_day["label_path"]),
                seq_len=8,
                min_events=2,
                basic_root=_make_basic(tmp_path),
                benchmark_path=_make_benchmark(tmp_path),
                top_n=1,
                start_date="2021-01-05",
                end_date="2021-01-05",
                out_path=out,
                device="cpu",
                liquidity_lookback_days=1,
                min_liquidity_observations=1,
            )
