"""端到端次日信号推理契约测试。"""

import json

import numpy as np
import pytest
import torch

from ticknet.nextday.inference import NextDayPredictor, main
from ticknet.nextday.model import build_nextday_model


def _raw_events(rows: int) -> np.ndarray:
    events = []
    for row_index in range(rows):
        row = []
        mid = 100 + row_index * 0.01
        for level in range(1, 11):
            row.extend((mid + level, 100 + level, mid - level, 101 + level))
        events.append(row)
    return np.asarray(events, dtype=np.float32)


def _artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "chunks_per_sample": 1,
                "chunk_size": 20,
                "metadata": {
                    "normalization": {
                        "price_scale_bps": 100.0,
                        "volume_log_scale": 16.0,
                        "clip": 32.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    model = build_nextday_model(
        chunks_per_sample=1,
        chunk_size=20,
        intraday_embedding_size=8,
        day_hidden_size=8,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "experiment": {
                "intraday_embedding_size": 8,
                "day_hidden_size": 8,
                "day_layers": 1,
                "dropout": 0.0,
            },
            "target_normalization": {"mean": 0.001, "std": 0.02},
        },
        checkpoint,
    )
    return checkpoint, manifest


def test_predictor_returns_score_probabilities_and_direction(tmp_path):
    checkpoint, manifest = _artifacts(tmp_path)
    predictor = NextDayPredictor(checkpoint, manifest)
    signal = predictor.predict_raw_snapshot(_raw_events(25))

    assert sum(signal.probabilities) == pytest.approx(1.0)
    assert signal.direction == int(np.argmax(signal.probabilities))
    assert signal.expected_excess_return == pytest.approx(signal.score * 0.02 + 0.001)


def test_predictor_filters_invalid_rows_before_selecting_the_tail(tmp_path):
    checkpoint, manifest = _artifacts(tmp_path)
    predictor = NextDayPredictor(checkpoint, manifest)
    raw = _raw_events(21)
    invalid = raw.copy()
    invalid[-1, 0] = 0.0

    expected = predictor.predict_raw_snapshot(raw[:-1])
    actual = predictor.predict_raw_snapshot(invalid)
    assert actual.score == pytest.approx(expected.score)
    assert actual.probabilities == pytest.approx(expected.probabilities)


def test_predictor_restores_frontend_widths_from_checkpoint(tmp_path):
    checkpoint, manifest = _artifacts(tmp_path)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_nextday_model(
        chunks_per_sample=1,
        chunk_size=20,
        conv_channels=8,
        inception_channels=16,
        intraday_embedding_size=8,
        day_hidden_size=8,
    )
    content["model"] = model.state_dict()
    content["experiment"]["conv_channels"] = 8
    content["experiment"]["inception_channels"] = 16
    torch.save(content, checkpoint)

    predictor = NextDayPredictor(checkpoint, manifest)
    assert predictor.model.intraday_encoder.conv1[0].out_channels == 8
    assert predictor.model.intraday_encoder.inception.branch_3[0].out_channels == 16


def test_prediction_cli_accepts_normalized_npy(tmp_path, capsys):
    checkpoint, manifest = _artifacts(tmp_path)
    events = tmp_path / "events.npy"
    np.save(events, np.zeros((20, 40), dtype=np.float32))
    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--events-npy",
            str(events),
            "--input-format",
            "normalized",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert len(output["probabilities"]) == 3
    assert output["direction"] in {0, 1, 2}
