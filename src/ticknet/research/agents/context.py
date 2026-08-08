"""研究上下文：Brainstorm Agent 的标准输入。

对应 AgentX 论文 ResearchContext：研究问题、基线证据、近期实验、异常发现、
可用动作、算力预算。Agent 只看这个上下文，不直接读整个仓库或测试集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchContext:
    """一次 brainstorm 的输入快照。"""

    research_question: str
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    recent_experiments: list[dict[str, Any]] = field(default_factory=list)
    open_anomalies: list[dict[str, Any]] = field(default_factory=list)
    allowed_actions: list[str] = field(
        default_factory=lambda: [
            "ablation",
            "robustness",
            "baseline",
            "feature_addition",
            "architecture",
            "data_audit",
            "cost_analysis",
        ]
    )
    compute_budget_hours: float = 4.0

    def to_prompt(self) -> str:
        lines = [
            f"研究问题：{self.research_question}",
            f"基线摘要：{self.baseline_summary or '无'}",
            "近期实验：",
        ]
        for experiment in self.recent_experiments[-10:]:
            lines.append(f"- {experiment.get('hypothesis', '?')}: {experiment.get('status', '?')}")
        lines.append("待解释异常：")
        for anomaly in self.open_anomalies[-5:]:
            lines.append(
                f"- {anomaly.get('type', '?')} ({anomaly.get('severity', '?')}): "
                f"{anomaly.get('detail', '?')}"
            )
        lines.append(f"可用动作：{', '.join(self.allowed_actions)}")
        lines.append(f"算力预算：{self.compute_budget_hours} 小时")
        return "\n".join(lines)
