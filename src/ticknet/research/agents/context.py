"""Brainstorm 与 Critic 共用的版本化研究上下文快照。"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ticknet.research.spec import IMPLEMENTED_EXECUTORS

CONTEXT_SCHEMA_VERSION = 1

DEFAULT_ALLOWED_ACTIONS = (
    "ablation",
    "robustness",
    "baseline",
    "feature_addition",
    "architecture",
    "data_audit",
    "cost_analysis",
    "prediction_export",
    "comparison",
)


@dataclass
class ResearchContext:
    """一次可序列化、可重放的 Agent 输入快照。"""

    research_question: str
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    recent_experiments: list[dict[str, Any]] = field(default_factory=list)
    open_anomalies: list[dict[str, Any]] = field(default_factory=list)
    historical_failures: list[dict[str, Any]] = field(default_factory=list)
    seen_novelty_signatures: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ACTIONS))
    available_executors: list[str] = field(default_factory=lambda: sorted(IMPLEMENTED_EXECUTORS))
    data_access: dict[str, Any] = field(
        default_factory=lambda: {
            "locked_test_access": False,
            "locked_test_requires_one_time_approval": True,
            "known_dataset_fingerprints": [],
        }
    )
    compute_budget_hours: float = 4.0
    schema_version: int = CONTEXT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError(f"不支持的 ResearchContext schema: {self.schema_version}")
        if not isinstance(self.research_question, str) or not self.research_question.strip():
            raise ValueError("research_question 不能为空")
        if (
            isinstance(self.compute_budget_hours, bool)
            or not isinstance(self.compute_budget_hours, (int, float))
            or not math.isfinite(self.compute_budget_hours)
            or self.compute_budget_hours <= 0
        ):
            raise ValueError("compute_budget_hours 必须为正数")
        if (
            not self.allowed_actions
            or any(
                not isinstance(value, str) or not value.strip() for value in self.allowed_actions
            )
            or len(set(self.allowed_actions)) != len(self.allowed_actions)
        ):
            raise ValueError("allowed_actions 必须非空且不能重复")
        if (
            not self.available_executors
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.available_executors
            )
            or len(set(self.available_executors)) != len(self.available_executors)
        ):
            raise ValueError("available_executors 必须非空且不能重复")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.seen_novelty_signatures
        ) or len(set(self.seen_novelty_signatures)) != len(self.seen_novelty_signatures):
            raise ValueError("seen_novelty_signatures 不能重复")
        if not isinstance(self.data_access, dict):
            raise ValueError("data_access 必须为对象")
        if self.data_access.get("locked_test_access") is not False:
            raise ValueError("ResearchContext 不能授予 locked_test_access")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "baseline_summary": self.baseline_summary,
            "recent_experiments": self.recent_experiments,
            "open_anomalies": self.open_anomalies,
            "historical_failures": self.historical_failures,
            "seen_novelty_signatures": self.seen_novelty_signatures,
            "allowed_actions": self.allowed_actions,
            "available_executors": self.available_executors,
            "data_access": self.data_access,
            "compute_budget_hours": self.compute_budget_hours,
        }

    @property
    def context_fingerprint(self) -> str:
        payload = json.dumps(
            self._payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **deepcopy(self._payload()),
            "context_fingerprint": self.context_fingerprint,
        }

    def to_prompt(self) -> str:
        self.validate()
        lines = [
            f"上下文指纹：{self.context_fingerprint}",
            f"研究问题：{self.research_question}",
            "基线摘要："
            + json.dumps(self.baseline_summary or "无", ensure_ascii=False, sort_keys=True),
            "近期实验：",
        ]
        if not self.recent_experiments:
            lines.append("- 无")
        for experiment in self.recent_experiments[:10]:
            lines.append(
                f"- {experiment.get('experiment_id', '?')} | "
                f"{experiment.get('status', '?')}/"
                f"{experiment.get('evaluation_decision') or '无决策'} | "
                f"{experiment.get('hypothesis', '?')} | "
                f"novelty={experiment.get('novelty_signature') or '?'}"
            )
        lines.append("历史失败与否定结论：")
        if not self.historical_failures:
            lines.append("- 无")
        for failure in self.historical_failures[:10]:
            lines.append(
                f"- {failure.get('experiment_id', '?')}: {failure.get('reason', '未记录原因')}"
            )
        lines.append("待解释异常：")
        if not self.open_anomalies:
            lines.append("- 无")
        for anomaly in self.open_anomalies[:10]:
            lines.append(
                f"- {anomaly.get('type', '?')} ({anomaly.get('severity', '?')}) "
                f"source={anomaly.get('source_experiment_id', '?')}: "
                f"{anomaly.get('detail', '?')}"
            )
        lines.append(f"可用动作：{', '.join(self.allowed_actions)}")
        lines.append(f"可用 executor：{', '.join(self.available_executors)}")
        lines.append(
            "数据权限：" + json.dumps(self.data_access, ensure_ascii=False, sort_keys=True)
        )
        visible_signatures = self.seen_novelty_signatures[:50]
        lines.append(
            "已见 novelty_signature："
            + (", ".join(visible_signatures) if visible_signatures else "无")
            + f"（共 {len(self.seen_novelty_signatures)} 个）"
        )
        lines.append(f"算力预算：{self.compute_budget_hours} 小时")
        return "\n".join(lines)
