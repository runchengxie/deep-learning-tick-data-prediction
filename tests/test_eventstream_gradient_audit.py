"""事件流梯度审计跨折决策门槛。"""

from __future__ import annotations

import json
from pathlib import Path

from ticknet.eventstream.gradient_audit import _canonical_sha256, decide_gradient_audits


def _write_audit(
    path: Path,
    *,
    ratio: float,
    cosine: float,
    negative_fraction: float,
    dataset_fingerprint: str,
) -> None:
    result = {
        "schema_version": 1,
        "mode": "eventstream_gradient_audit",
        "status": "complete",
        "materialized": {"dataset_fingerprint": dataset_fingerprint},
        "states": {
            "best_checkpoint": {
                "summary": {
                    "day_to_generative_median_gradient_norm": {"median": ratio},
                    "cosines": {
                        "stream__day": {
                            "median": cosine,
                            "negative_fraction": negative_fraction,
                        },
                        "otype__day": {"median": 0.1, "negative_fraction": 0.2},
                        "reg__day": {"median": 0.2, "negative_fraction": 0.1},
                    },
                }
            }
        },
    }
    result["result_fingerprint"] = _canonical_sha256(result)
    path.write_text(json.dumps(result), encoding="utf-8")


def test_decision_selects_label_scale_when_day_gradient_is_weak(tmp_path: Path) -> None:
    paths = [tmp_path / "recent.json", tmp_path / "rolling.json"]
    for index, path in enumerate(paths):
        _write_audit(
            path,
            ratio=0.05 + index * 0.01,
            cosine=0.2,
            negative_fraction=0.1,
            dataset_fingerprint=str(index) * 64,
        )

    result = decide_gradient_audits(paths)

    assert result["decision"] == "day_gradient_weak"
    assert result["next_experiment"] == "EVT-LABEL-SCALE-001"


def test_decision_selects_supervision_and_weight_review_for_shared_conflict(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "recent.json", tmp_path / "rolling.json"]
    for index, path in enumerate(paths):
        _write_audit(
            path,
            ratio=0.5,
            cosine=-0.2,
            negative_fraction=0.8,
            dataset_fingerprint=str(index) * 64,
        )

    result = decide_gradient_audits(paths)

    assert result["decision"] == "persistent_task_conflict"
    assert result["shared_persistent_negative_pairs"] == ["stream__day"]
    assert result["next_experiment"] == ("EVT-SUPERVISION-POSITION-001_WITH_TASK_WEIGHT_REVIEW")


def test_decision_selects_supervision_position_when_gradients_are_normal(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "recent.json", tmp_path / "rolling.json"]
    for index, path in enumerate(paths):
        _write_audit(
            path,
            ratio=0.5,
            cosine=-0.05,
            negative_fraction=0.6,
            dataset_fingerprint=str(index) * 64,
        )

    result = decide_gradient_audits(paths)

    assert result["decision"] == "gradient_strength_normal_without_persistent_conflict"
    assert result["next_experiment"] == "EVT-SUPERVISION-POSITION-001"
