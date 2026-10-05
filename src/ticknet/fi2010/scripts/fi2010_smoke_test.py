"""Run a local synthetic-data smoke check for the FI-2010 reproduction.

运行方式：

    ticknet-fi2010-smoke-test
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ticknet.fi2010.core import (
    K_TO_LABEL_COLUMN,
    NUM_CLASSES,
    NUM_FEATURES,
    TOTAL_COLUMNS,
    WINDOW_SIZE,
    FI2010WindowDataset,
    get_dummy_batch,
)


def check_forward_pass() -> None:
    import torch
    import torch.nn.functional as F

    from ticknet.model import build_model

    model = build_model()
    features, _ = get_dummy_batch(batch_size=8)
    logits = model(features)
    probabilities = F.softmax(logits, dim=1)
    assert features.shape == (8, 1, WINDOW_SIZE, NUM_FEATURES)
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
    print("Passed: forward-pass shapes and softmax probabilities")


def check_fi2010_dataset() -> None:
    segment_length = 500
    segments = [
        {"cf": 7, "role": "train", "start": 0, "end": segment_length},
        {
            "cf": 7,
            "role": "test",
            "start": segment_length,
            "end": segment_length * 2,
        },
        {
            "cf": 8,
            "role": "test",
            "start": segment_length * 2,
            "end": segment_length * 3,
        },
        {
            "cf": 9,
            "role": "test",
            "start": segment_length * 3,
            "end": segment_length * 4,
        },
    ]
    rows = segment_length * 4
    rng = np.random.default_rng(1)
    data = np.zeros((rows, TOTAL_COLUMNS), dtype=np.float32)
    data[:, :NUM_FEATURES] = rng.standard_normal(
        (rows, NUM_FEATURES),
        dtype=np.float32,
    )
    labels = np.resize(np.array([1, 2, 3], dtype=np.float32), rows)
    for column in K_TO_LABEL_COLUMN.values():
        data[:, column] = labels

    with tempfile.TemporaryDirectory() as directory:
        data_path = Path(directory) / "fi2010.npy"
        meta_path = Path(directory) / "fi2010_meta.json"
        np.save(data_path, data)
        meta_path.write_text(
            json.dumps({"rows": rows, "segments": segments}),
            encoding="utf-8",
        )
        for horizon in K_TO_LABEL_COLUMN:
            with FI2010WindowDataset(
                str(data_path),
                str(meta_path),
                k=horizon,
                split="train",
                protocol="setup2",
            ) as dataset:
                features, label = dataset[0]
                assert features.shape == (1, WINDOW_SIZE, NUM_FEATURES)
                assert label in {0, 1, 2}
    print("Passed: dataset windows and labels for all five horizons")


def main() -> None:
    argparse.ArgumentParser(
        description="Run synthetic FI-2010 dataset and DeepLOB compatibility checks."
    ).parse_args()
    check_forward_pass()
    check_fi2010_dataset()
    print("All FI-2010 smoke checks passed.")


if __name__ == "__main__":
    main()
