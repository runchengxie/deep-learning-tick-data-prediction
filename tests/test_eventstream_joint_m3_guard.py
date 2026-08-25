"""联合微调对尚未支持的 M3-inspired 表征必须显式拒绝。"""

from pathlib import Path

import pytest
import torch

from ticknet.eventstream.fingerprint import file_sha256
from ticknet.eventstream.joint import (
    JointConfig,
    JointEventstreamModel,
    load_pretrained_backbone,
)


@pytest.mark.parametrize(
    "representation",
    [
        {
            "use_lob_prefix": True,
            "use_session_anchors": True,
            "use_vq": False,
        },
        {
            "use_lob_prefix": False,
            "use_session_anchors": False,
            "use_vq": True,
        },
    ],
)
def test_joint_rejects_m3_representation_checkpoint(
    tmp_path: Path,
    representation: dict[str, bool],
) -> None:
    config = JointConfig(model="smoke", seed=0)
    model = JointEventstreamModel(config, feature_count=2)
    checkpoint_path = tmp_path / "m3-prefix.pt"
    torch.save(
        {
            "model": model.eventstream.state_dict(),
            "experiment": {
                "model": "smoke",
                "seed": 0,
                "source_revision": "abcdef0",
                **representation,
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="M3-inspired"):
        load_pretrained_backbone(
            model,
            checkpoint_path,
            model_name="smoke",
            seed=0,
            expected_sha256=file_sha256(checkpoint_path),
        )
