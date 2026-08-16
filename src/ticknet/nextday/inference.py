"""加载已训练双头模型并把逐 tick 十档盘口转换为次日信号。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ticknet.nextday.config import DEFAULT_CONV_CHANNELS, DEFAULT_INCEPTION_CHANNELS
from ticknet.nextday.dataset import manifest_fingerprint
from ticknet.nextday.io import pack_events
from ticknet.nextday.model import ChunkedDeepLOB, build_nextday_model
from ticknet.nextday.raw_snapshot import normalize_lob_events, valid_lob_event_rows


@dataclass(frozen=True)
class NextDaySignal:
    """单个股票日的模型输出。"""

    score: float
    expected_excess_return: float
    probabilities: tuple[float, float, float]
    direction: int


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - 兼容旧 PyTorch
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 根节点应为对象")
    return cast(dict[str, Any], checkpoint)


class NextDayPredictor:
    """从 checkpoint 和训练数据契约恢复端到端推理。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        manifest_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device 应为 cpu 或 cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 推理，但当前 PyTorch 没有可用 CUDA")
        self.device = torch.device(device)
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError("manifest 根节点应为对象")
        computed_fingerprint = manifest_fingerprint(manifest)
        stored_fingerprint = manifest.get("dataset_fingerprint")
        if stored_fingerprint is not None and stored_fingerprint != computed_fingerprint:
            raise ValueError("manifest dataset_fingerprint 与内容不一致")
        self.dataset_fingerprint = stored_fingerprint or computed_fingerprint
        self.source_chunks_per_sample = int(manifest["chunks_per_sample"])
        self.chunk_size = int(manifest["chunk_size"])
        self.source_total_events = self.source_chunks_per_sample * self.chunk_size
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("manifest metadata 应为对象")
        normalization = metadata.get("normalization")
        self.normalization = normalization if isinstance(normalization, dict) else None
        self.min_valid_events = int(metadata.get("min_valid_events", 1))
        if not 1 <= self.min_valid_events <= self.source_total_events:
            raise ValueError("manifest min_valid_events 超出窗口范围")

        checkpoint = _load_checkpoint(
            Path(checkpoint_path).expanduser().resolve(),
            self.device,
        )
        experiment = checkpoint.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("checkpoint 缺少 experiment 配置")
        input_last_chunks = int(experiment.get("input_last_chunks", 0))
        if not 0 <= input_last_chunks <= self.source_chunks_per_sample:
            raise ValueError("checkpoint input_last_chunks 与 manifest 不兼容")
        self.chunks_per_sample = input_last_chunks or self.source_chunks_per_sample
        self.total_events = self.chunks_per_sample * self.chunk_size
        checkpoint_fingerprint = experiment.get("dataset_fingerprint")
        if (
            checkpoint_fingerprint is not None
            and checkpoint_fingerprint != self.dataset_fingerprint
        ):
            raise ValueError("checkpoint 与 manifest 的数据指纹不一致")
        self.model: ChunkedDeepLOB = build_nextday_model(
            chunks_per_sample=self.chunks_per_sample,
            chunk_size=self.chunk_size,
            conv_channels=int(experiment.get("conv_channels", DEFAULT_CONV_CHANNELS)),
            inception_channels=int(
                experiment.get("inception_channels", DEFAULT_INCEPTION_CHANNELS)
            ),
            intraday_embedding_size=int(experiment["intraday_embedding_size"]),
            day_hidden_size=int(experiment["day_hidden_size"]),
            day_layers=int(experiment["day_layers"]),
            dropout=float(experiment["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        target_normalization = checkpoint.get("target_normalization")
        if not isinstance(target_normalization, dict):
            raise ValueError("checkpoint 缺少 target_normalization")
        self.target_mean = float(target_normalization["mean"])
        self.target_std = float(target_normalization["std"])
        if not math.isfinite(self.target_mean) or not math.isfinite(self.target_std):
            raise ValueError("目标归一化参数不是有限值")
        if self.target_std <= 0:
            raise ValueError("目标收益标准差必须为正数")

    def _predict_normalized(self, events: np.ndarray) -> NextDaySignal:
        normalized = np.asarray(events, dtype=np.float32)
        packed, _valid_events = pack_events(
            normalized,
            chunks_per_sample=self.chunks_per_sample,
            chunk_size=self.chunk_size,
        )
        features = torch.from_numpy(packed).unsqueeze(0).unsqueeze(2).to(self.device)
        with torch.inference_mode():
            output = self.model(features)
            probabilities = torch.softmax(output.logits, dim=1)[0].cpu().numpy()
            score = float(output.score[0].cpu())
        return NextDaySignal(
            score=score,
            expected_excess_return=score * self.target_std + self.target_mean,
            probabilities=(
                float(probabilities[0]),
                float(probabilities[1]),
                float(probabilities[2]),
            ),
            direction=int(np.argmax(probabilities)),
        )

    def predict_normalized(self, events: np.ndarray) -> NextDaySignal:
        """预测已按 manifest 契约归一化的 ``N × 40`` tick。"""
        return self._predict_normalized(events)

    def predict_raw_snapshot(self, raw_events: np.ndarray) -> NextDaySignal:
        """预测原始价格/数量排列的 ``N × 40`` snapshot tick。"""
        if self.normalization is None:
            raise ValueError("manifest 未记录原始 snapshot 归一化契约")
        events = np.asarray(raw_events)
        selected = events[valid_lob_event_rows(events)][-self.source_total_events :]
        if selected.shape[0] < self.min_valid_events:
            raise ValueError(
                f"有效盘口事件不足：至少需要 {self.min_valid_events}，实际为 {selected.shape[0]}"
            )
        normalized = normalize_lob_events(
            selected,
            price_scale_bps=float(self.normalization["price_scale_bps"]),
            volume_log_scale=float(self.normalization["volume_log_scale"]),
            clip=float(self.normalization["clip"]),
        )
        return self._predict_normalized(normalized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用逐 tick 十档盘口输出下一交易日信号")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events-npy", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--input-format",
        choices=["raw", "normalized"],
        default="raw",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    events = np.load(args.events_npy, allow_pickle=False)
    predictor = NextDayPredictor(
        args.checkpoint,
        args.manifest,
        device=args.device,
    )
    signal = (
        predictor.predict_raw_snapshot(events)
        if args.input_format == "raw"
        else predictor.predict_normalized(events)
    )
    print(json.dumps(asdict(signal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
