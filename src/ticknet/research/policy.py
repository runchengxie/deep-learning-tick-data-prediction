"""实验策略：Agent/提案被允许改动什么，由确定性代码裁决。

对应 AgentX 论文中"Policy 是笨但绝对可靠的"分层：LLM 只是提案者，
这里才是最终决定权。禁止改动任何与数据切分、测试集、checkpoint 相关的字段，
防止 Agent 无意或有意地污染样本外评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ticknet.research.spec import ExperimentSpec

if TYPE_CHECKING:
    from ticknet.research.protocol import ResearchProtocol


class PolicyViolation(Exception):
    """策略违规，提案被拒绝。"""


@dataclass(frozen=True)
class ResearchPolicy:
    """研究策略配置：允许/禁止字段、种子上限、阶段白名单。"""

    allowed_config_fields: frozenset[str] = frozenset(
        {
            "lr",
            "weight_decay",
            "dropout",
            "classification_loss_weight",
            "regression_loss_weight",
            "conv_channels",
            "inception_channels",
            "intraday_embedding_size",
            "day_hidden_size",
            "day_layers",
            "hidden_channels",
            "tcn_layers",
            "kernel_size",
            "batch_size",
            "epochs",
            "patience",
        }
    )
    forbidden_config_fields: frozenset[str] = frozenset(
        {
            "manifest_path",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
            "evaluate_test",
            "checkpoint_dir",
            "verify_data_checksums",
            "device",
        }
    )
    max_screening_seeds: int = 3
    allowed_stages: frozenset[str] = frozenset({"screening", "robustness", "release"})

    def validate(self, spec: ExperimentSpec) -> None:
        forbidden = set(spec.config_overrides) & self.forbidden_config_fields
        if forbidden:
            raise PolicyViolation(f"禁止修改这些参数: {sorted(forbidden)}")
        unknown = set(spec.config_overrides) - self.allowed_config_fields
        if unknown:
            raise PolicyViolation(f"不允许修改这些参数: {sorted(unknown)}")
        if spec.stage not in self.allowed_stages:
            raise PolicyViolation(f"stage 应为 {sorted(self.allowed_stages)} 之一")
        if len(spec.seeds) > self.max_screening_seeds:
            raise PolicyViolation(
                f"实验预算超限：seeds {len(spec.seeds)} 超过上限 {self.max_screening_seeds}"
            )
        spec.validate()

    def validate_manifest(
        self,
        manifest_path: str | Path,
        protocol: ResearchProtocol,
    ) -> None:
        """校验 manifest 不含锁定测试期数据。"""
        protocol.assert_research_safe(manifest_path)
