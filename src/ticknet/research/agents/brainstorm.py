"""Brainstorm Agent：把研究上下文转成可执行的 ExperimentSpec。

对应 AgentX 论文的提案生成与证据加权。第一版模板模式从当前异常和基线
推导下一步实验；LLM 模式让模型从上下文生成结构化提案 JSON。无论哪种，
输出都必须是 ExperimentSpec，之后由 Policy 裁决。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from ticknet.research.agents.client import LLMClient
from ticknet.research.agents.context import ResearchContext
from ticknet.research.spec import ExperimentSpec

_SYSTEM_PROMPT = (
    "你是量化研究助手。根据给定研究上下文，输出一个可执行实验提案，"
    "只返回 JSON，不要解释。JSON 字段：hypothesis, rationale, "
    "falsification_condition, experiment_type, base_config, config_overrides, "
    "primary_metric, expected_direction, seeds。禁止修改任何日期或测试集相关字段。"
)


class BrainstormAgent:
    """生成实验提案的 Agent。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        default_base_config: str = "configs/nextday-pilot.yaml",
    ) -> None:
        self.llm = llm
        self.default_base_config = default_base_config

    def propose(self, context: ResearchContext) -> ExperimentSpec:
        """根据上下文生成一个实验提案。"""
        if type(self.llm).__name__ == "TemplateClient":
            return self._template_propose(context)
        return self._llm_propose(context)

    def _template_propose(self, context: ResearchContext) -> ExperimentSpec:
        """模板策略：优先解释 open_anomalies，否则做损失/结构消融。"""
        if context.open_anomalies:
            anomaly = context.open_anomalies[0]
            return self._anomaly_experiment(anomaly)
        return self._with_entry_point(
            ExperimentSpec(
                hypothesis="降低回归损失权重应改善横截面排序",
                experiment_type="ablation",
                base_config=self.default_base_config,
                config_overrides={"regression_loss_weight": 0.2},
                seeds=(0,),
                rationale="从当前基线出发的默认受控消融",
                falsification_condition="若多 seed 下 Rank IC 无稳定改善则否定",
            )
        )

    def _with_entry_point(self, spec: ExperimentSpec) -> ExperimentSpec:
        """根据 base_config 推断训练入口，避免 experiment_type 无对应入口。"""
        if "tcn" in spec.base_config:
            return replace(spec, entry_point="ticknet-minute-tcn-train")
        return replace(spec, entry_point="ticknet-nextday-train")

    def _anomaly_experiment(self, anomaly: dict[str, Any]) -> ExperimentSpec:
        anomaly_type = anomaly.get("type", "")
        if anomaly_type == "tail_return_concentration":
            return self._with_entry_point(
                ExperimentSpec(
                    hypothesis="极端收益驱动了 spread，剔除后信号是否仍存在",
                    experiment_type="data_audit",
                    base_config=self.default_base_config,
                    config_overrides={},
                    seeds=(0,),
                    rationale=anomaly.get("detail", "极端日贡献偏高"),
                    falsification_condition="若剔除极端日后 spread 显著下降且 IC 不变，"
                    "则信号由极端值驱动",
                )
            )
        if anomaly_type == "weak_decile_monotonicity":
            return self._with_entry_point(
                ExperimentSpec(
                    hypothesis="排序信号非线性，可能需要更激进的分组",
                    experiment_type="robustness",
                    base_config=self.default_base_config,
                    config_overrides={"portfolio_quantile": 0.05},
                    seeds=(0,),
                    rationale="decile 单调性弱，试探更小分位",
                    falsification_condition="若更小分位下 IC 无改善则否定非线性假设",
                )
            )
        return self._with_entry_point(
            ExperimentSpec(
                hypothesis="成本后收益为负，检查调仓频率影响",
                experiment_type="cost_analysis",
                base_config=self.default_base_config,
                config_overrides={},
                seeds=(0,),
                rationale="成本是当前主要瓶颈",
                falsification_condition="若降频无法转正则确认不可交易",
            )
        )

    def _llm_propose(self, context: ResearchContext) -> ExperimentSpec:
        output = self.llm.generate(
            _SYSTEM_PROMPT,
            context.to_prompt(),
            temperature=0.3,
        )
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM 输出中没有 JSON: {output[:500]}")
        values = json.loads(match.group(0))
        seeds = tuple(int(seed) for seed in values.get("seeds", [0]))
        spec = ExperimentSpec(
            hypothesis=str(values["hypothesis"]),
            rationale=str(values.get("rationale", "")),
            falsification_condition=str(values.get("falsification_condition", "")),
            experiment_type=str(values["experiment_type"]),
            base_config=str(values.get("base_config", self.default_base_config)),
            config_overrides=dict(values.get("config_overrides", {})),
            primary_metric=str(values.get("primary_metric", "daily_rank_ic_mean")),
            expected_direction=str(values.get("expected_direction", "increase")),
            seeds=seeds,
        )
        spec.validate()
        return spec
