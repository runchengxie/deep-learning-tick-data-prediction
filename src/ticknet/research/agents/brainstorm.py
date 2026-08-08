"""Brainstorm Agent：只生成 ExperimentSpec v2，不直接选择命令。"""

from __future__ import annotations

import json
import re
from typing import Any

from ticknet.research.agents.client import LLMClient
from ticknet.research.agents.context import ResearchContext
from ticknet.research.spec import ExperimentSpec, MetricGate

_SYSTEM_PROMPT = (
    "你是量化研究助手。只返回 ExperimentSpec v2 JSON。必须包含 hypothesis、objective、"
    "experiment_type、executor、inputs、config_overrides、primary_metrics、success_gates、"
    "artifact_contract、budget、falsification_condition、novelty_signature 和 seeds。"
    "executor 只能使用程序白名单，禁止返回 entry_point、命令、日期或测试集覆盖字段。"
)


class BrainstormAgent:
    """根据受控 ResearchContext 生成一个结构化实验。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        default_base_config: str = "configs/nextday-pilot.yaml",
    ) -> None:
        self.llm = llm
        self.default_base_config = default_base_config

    def propose(self, context: ResearchContext) -> ExperimentSpec:
        context.validate()
        if type(self.llm).__name__ == "TemplateClient":
            spec = self._template_propose(context)
        else:
            spec = self._llm_propose(context)
        if spec.novelty_signature in set(context.seen_novelty_signatures):
            raise ValueError(f"历史中已存在 novelty_signature: {spec.novelty_signature}")
        return spec

    def _template_propose(self, context: ResearchContext) -> ExperimentSpec:
        seen = set(context.seen_novelty_signatures)
        for anomaly in context.open_anomalies:
            candidate = self._anomaly_experiment(anomaly, context)
            if candidate.novelty_signature not in seen:
                return candidate
        executor = "train_minute_tcn" if "tcn" in self.default_base_config else "train_nextday"
        baseline_metrics = context.baseline_summary.get("metrics", {})
        if not isinstance(baseline_metrics, dict):
            baseline_metrics = {}
        baseline = float(
            baseline_metrics.get(
                "best_selection_value",
                context.baseline_summary.get("best_selection_value", 0.0),
            )
        )
        return ExperimentSpec(
            hypothesis="降低回归损失权重应改善横截面排序",
            objective="比较回归损失权重 0.2 与当前基线的验证排序指标",
            experiment_type="ablation",
            executor=executor,
            base_config=self.default_base_config,
            config_overrides={"regression_loss_weight": 0.2},
            seeds=(0,),
            primary_metrics=("best_selection_value",),
            success_gates=(MetricGate("best_selection_value", "gt", baseline),),
            rationale="从当前基线出发的默认受控消融",
            falsification_condition="多 seed 的 best_selection_value 均值不高于当前基线则否定",
            novelty_signature="ablation-regression-loss-weight-0.2",
        )

    def _anomaly_experiment(
        self,
        anomaly: dict[str, Any],
        context: ResearchContext,
    ) -> ExperimentSpec:
        anomaly_type = str(anomaly.get("type", ""))
        predictions = str(
            anomaly.get("predictions_path") or context.baseline_summary.get("predictions_path", "")
        )
        source_experiment_id = str(anomaly.get("source_experiment_id", ""))
        if anomaly_type == "tail_return_concentration":
            source_seed = anomaly.get("prediction_source_seed")
            novelty_signature = (
                f"audit-tail-return-concentration-{source_experiment_id}"
                if source_experiment_id
                else "audit-tail-return-concentration"
            )
            if source_experiment_id and isinstance(source_seed, int):
                return ExperimentSpec(
                    hypothesis="极端收益是否驱动了已有 spread",
                    objective="安全物化已登记预测并复查极端日与 winsorize 审计",
                    experiment_type="prediction_export",
                    executor="export_predictions",
                    inputs={
                        "source_experiment_id": source_experiment_id,
                        "source_seed": source_seed,
                        "artifact_name": str(
                            anomaly.get("prediction_artifact_name", "predictions")
                        ),
                    },
                    seeds=(0,),
                    primary_metrics=("audit.top_5_day_contribution",),
                    success_gates=(MetricGate("audit.top_5_day_contribution", "lt", 0.5),),
                    artifact_contract=(
                        "resolved_config",
                        "stdout",
                        "stderr",
                        "result",
                        "run_manifest",
                        "predictions",
                        "audit",
                    ),
                    rationale=str(anomaly.get("detail", "极端日贡献偏高")),
                    falsification_condition="top 5 日贡献不低于 50% 则否定收益稳定性",
                    novelty_signature=novelty_signature,
                )
            return ExperimentSpec(
                hypothesis="极端收益是否驱动了已有 spread",
                objective="对现有预测执行确定性极端日与 winsorize 审计",
                experiment_type="data_audit",
                executor="audit_predictions",
                inputs={"predictions_path": predictions},
                seeds=(0,),
                primary_metrics=("audit.top_5_day_contribution",),
                success_gates=(MetricGate("audit.top_5_day_contribution", "lt", 0.5),),
                artifact_contract=(
                    "resolved_config",
                    "stdout",
                    "stderr",
                    "result",
                    "run_manifest",
                    "audit",
                ),
                rationale=str(anomaly.get("detail", "极端日贡献偏高")),
                falsification_condition="top 5 日贡献不低于 50% 则否定收益稳定性",
                novelty_signature=novelty_signature,
            )
        return ExperimentSpec(
            hypothesis="Top-K 缓冲能否在真实成本下保留正净收益",
            objective="运行 K=50、buffer=20、单边 10bp 的 long-only 成本评估",
            experiment_type="cost_analysis",
            executor="topk_cost_sweep",
            inputs={
                "predictions_path": predictions,
                "top_k": [50],
                "exit_buffer": [20],
                "cost_bps": [10],
            },
            seeds=(0,),
            primary_metrics=("topk.k50.buffer20.cost10.net.sharpe",),
            success_gates=(MetricGate("topk.k50.buffer20.cost10.net.sharpe", "gt", 0.0),),
            artifact_contract=(
                "resolved_config",
                "stdout",
                "stderr",
                "result",
                "run_manifest",
                "topk_sweep",
            ),
            rationale=str(anomaly.get("detail", "成本是当前主要瓶颈")),
            falsification_condition="单边 10bp 下净 Sharpe 不大于 0 则否定可交易性",
            novelty_signature=(
                f"topk-cost-{anomaly_type or 'unknown'}-{source_experiment_id}"
                if source_experiment_id
                else f"topk-cost-{anomaly_type or 'unknown'}"
            ),
        )

    def _llm_propose(self, context: ResearchContext) -> ExperimentSpec:
        output = self.llm.generate(_SYSTEM_PROMPT, context.to_prompt(), temperature=0.3)
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM 输出中没有 JSON: {output[:500]}")
        values = json.loads(match.group(0))
        if not isinstance(values, dict):
            raise ValueError("LLM ExperimentSpec 根节点应为对象")
        values.setdefault("base_config", self.default_base_config)
        return ExperimentSpec.from_dict(values)
