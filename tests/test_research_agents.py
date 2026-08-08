"""Brainstorm、Critic 与强制 Runner/Evaluation 编排测试。"""

import json

import pytest

from ticknet.research.agents.brainstorm import BrainstormAgent
from ticknet.research.agents.client import TemplateClient, make_client
from ticknet.research.agents.context import ResearchContext
from ticknet.research.agents.context_builder import ContextBuildError, ResearchContextBuilder
from ticknet.research.agents.critic import CriticAgent
from ticknet.research.agents.orchestrator import ResearchOrchestrator
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentResult, ExperimentSpec, MetricGate


def test_make_client_template_and_unknown() -> None:
    assert isinstance(make_client("template"), TemplateClient)
    with pytest.raises(ValueError, match="未知 provider"):
        make_client("unknown_provider")


def test_template_client_echoes_user_prompt() -> None:
    client = TemplateClient()
    assert client.generate("system", "hello", temperature=0.0) == "hello"


def test_brainstorm_template_default_proposal_uses_typed_executor() -> None:
    spec = BrainstormAgent(TemplateClient()).propose(ResearchContext(research_question="测试问题"))
    assert spec.executor == "train_nextday"
    assert spec.config_overrides == {"regression_loss_weight": 0.2}
    spec.validate()


def test_brainstorm_anomaly_routes_to_audit_not_training() -> None:
    context = ResearchContext(
        research_question="测试",
        baseline_summary={"predictions_path": "predictions.parquet"},
        open_anomalies=[
            {
                "type": "tail_return_concentration",
                "severity": "high",
                "detail": "top 5 日贡献 121%",
            }
        ],
    )
    spec = BrainstormAgent(TemplateClient()).propose(context)
    assert spec.experiment_type == "data_audit"
    assert spec.executor == "audit_predictions"
    spec.validate()


def test_brainstorm_llm_path_parses_v2_json() -> None:
    class FakeClient(TemplateClient):
        def generate(self, system_prompt, user_prompt, *, temperature=0.0):
            return json.dumps(
                {
                    "hypothesis": "h",
                    "objective": "o",
                    "rationale": "r",
                    "falsification_condition": "指标不改善则否定",
                    "experiment_type": "ablation",
                    "executor": "train_nextday",
                    "config_overrides": {"lr": 0.0005},
                    "seeds": [0, 1],
                    "primary_metrics": ["validation.daily_rank_ic_mean"],
                    "success_gates": [
                        {
                            "metric": "validation.daily_rank_ic_mean",
                            "operator": "gt",
                            "threshold": 0.0,
                        }
                    ],
                    "artifact_contract": [
                        "resolved_spec",
                        "resolved_config",
                        "stdout",
                        "stderr",
                        "result",
                        "run_manifest",
                    ],
                    "budget": {"timeout_seconds": 60, "max_seeds": 2},
                    "novelty_signature": "llm-test",
                }
            )

    spec = BrainstormAgent(FakeClient()).propose(ResearchContext(research_question="q"))
    assert spec.executor == "train_nextday"
    assert spec.seeds == (0, 1)


def _complete_spec(
    *,
    experiment_type: str = "ablation",
    executor: str = "train_nextday",
    falsification_condition: str = "指标不改善则否定",
    novelty_signature: str = "critic-test",
    parent_id: str | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis="h",
        objective="o",
        experiment_type=experiment_type,
        executor=executor,
        base_config="base.yaml",
        seeds=(0,),
        success_gates=(MetricGate("best_selection_value", "gt", 0.0),),
        primary_metrics=("best_selection_value",),
        rationale="r",
        falsification_condition=falsification_condition,
        parent_id=parent_id,
        novelty_signature=novelty_signature,
    )


def test_critic_rejects_missing_falsification_and_semantic_mismatch() -> None:
    critique = CriticAgent().review(_complete_spec(falsification_condition=""))
    assert not critique.approved
    assert any("falsification" in issue for issue in critique.issues)

    mismatch = _complete_spec(experiment_type="cost_analysis", executor="train_nextday")
    critique = CriticAgent().review(mismatch)
    assert not critique.approved
    assert any("topk_cost_sweep" in issue for issue in critique.issues)

    mismatch = _complete_spec(experiment_type="robustness", executor="train_nextday")
    critique = CriticAgent().review(mismatch)
    assert not critique.approved
    assert any("walk_forward_robustness" in issue for issue in critique.issues)


def test_critic_approves_complete_spec() -> None:
    assert CriticAgent().review(_complete_spec()).approved


def test_brainstorm_and_critic_reject_seen_novelty_and_context_budget() -> None:
    context = ResearchContext(
        research_question="q",
        seen_novelty_signatures=["ablation-regression-loss-weight-0.2", "critic-test"],
        compute_budget_hours=0.5,
    )
    with pytest.raises(ValueError, match="novelty_signature"):
        BrainstormAgent(TemplateClient()).propose(context)

    critique = CriticAgent().review(_complete_spec(), context)
    assert not critique.approved
    assert any("novelty_signature" in issue for issue in critique.issues)
    assert any("算力预算" in issue for issue in critique.issues)


def _record_context_experiment(
    registry: ExperimentRegistry,
    experiment_id: str,
    *,
    status: str,
    decision: str | None,
    novelty_signature: str,
    fingerprint: str | None,
    parent_id: str | None = None,
) -> None:
    spec = _complete_spec(
        novelty_signature=novelty_signature,
        parent_id=parent_id,
    )
    registry.record_experiment(
        experiment_id,
        ExperimentResult(
            experiment_id=experiment_id,
            spec=spec,
            status=status,
            git_sha="abc",
            dataset_fingerprint=fingerprint,
            per_seed_metrics=[],
            artifact_dir=f"/tmp/{experiment_id}",
            evaluation_decision=decision,
        ),
        spec,
    )


def test_context_builder_replays_registry_decisions_failures_and_anomalies(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _record_context_experiment(
        registry,
        "EXP-BASE",
        status="completed",
        decision="KEEP",
        novelty_signature="baseline-v1",
        fingerprint="fp-base",
    )
    registry.record_metrics("EXP-BASE", 0, {"best_selection_value": 0.03})
    predictions = tmp_path / "predictions.parquet"
    predictions.write_bytes(b"registered prediction fixture")
    registry.record_artifact("EXP-BASE", 0, "predictions", predictions)
    registry.record_review(
        "EXP-BASE",
        "audit_anomalies",
        "RECORDED",
        {
            "anomalies": [
                {
                    "type": "tail_return_concentration",
                    "severity": "high",
                    "detail": "top 5 日贡献过高",
                }
            ]
        },
    )
    _record_context_experiment(
        registry,
        "EXP-FAILED",
        status="failed",
        decision=None,
        novelty_signature="failed-lr-probe",
        fingerprint=None,
        parent_id="EXP-BASE",
    )
    registry.update_experiment("EXP-FAILED", status="failed", error="executor timeout")

    builder = ResearchContextBuilder(registry)
    context = builder.build("盘口信号能否覆盖成本", compute_budget_hours=2.0)
    replay = builder.build("盘口信号能否覆盖成本", compute_budget_hours=2.0)

    assert context.to_dict() == replay.to_dict()
    assert context.baseline_summary["experiment_id"] == "EXP-BASE"
    assert context.baseline_summary["metrics"]["best_selection_value"] == 0.03
    assert context.baseline_summary["predictions_path"] == str(predictions)
    assert context.recent_experiments[0]["experiment_id"] == "EXP-FAILED"
    assert context.recent_experiments[0]["parent_id"] == "EXP-BASE"
    assert context.historical_failures[0]["reason"] == "executor timeout"
    assert context.open_anomalies[0]["source_experiment_id"] == "EXP-BASE"
    assert context.open_anomalies[0]["predictions_path"] == str(predictions)
    assert context.open_anomalies[0]["prediction_source_seed"] == 0
    assert context.seen_novelty_signatures == ["failed-lr-probe", "baseline-v1"]
    assert context.data_access["locked_test_access"] is False
    assert "train_ranker" not in context.available_executors
    assert context.context_fingerprint in context.to_prompt()
    proposal = BrainstormAgent(TemplateClient()).propose(context)
    assert proposal.executor == "export_predictions"
    assert proposal.inputs["source_experiment_id"] == "EXP-BASE"
    with pytest.raises(ContextBuildError, match="compute_budget_hours"):
        builder.build("q", compute_budget_hours=float("nan"))
    registry.close()


def _make_harness(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "samples": [
                    {"trading_date": "2024-01-02", "symbol": "600000"},
                    {"trading_date": "2024-06-30", "symbol": "600001"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text(f"manifest_path: {manifest}\n", encoding="utf-8")


def _orchestrator(tmp_path, registry):
    _make_harness(tmp_path)
    mock = tmp_path / "mock.py"
    mock.write_text(
        "import json\nprint(json.dumps({'best_selection_value': 0.01}))\n",
        encoding="utf-8",
    )
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        command_overrides={"train_nextday": ["python", str(mock)]},
    )
    return ResearchOrchestrator(
        registry,
        brainstorm=BrainstormAgent(TemplateClient(), default_base_config="base.yaml"),
        critic=CriticAgent(),
        runner=runner,
    )


def test_orchestrator_runs_evaluation_and_records_critic_review(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    context = ResearchContext(research_question="q")
    step = _orchestrator(tmp_path, registry).research_step(
        context,
        experiment_id="EXP-AUTO-TEST",
    )
    assert step.status == "completed"
    assert step.result["evaluation_decision"] == "EXTEND"
    assert step.context_fingerprint == context.context_fingerprint
    reviews = {row["review_type"]: row for row in registry.get_reviews("EXP-AUTO-TEST")}
    assert {"research_context", "critic", "evaluation"} <= set(reviews)
    snapshot = json.loads(reviews["research_context"]["payload_json"])
    assert snapshot["context_fingerprint"] == context.context_fingerprint

    feedback = ResearchContextBuilder(registry).build("q")
    repeated = _orchestrator(tmp_path, registry).research_step(
        feedback,
        experiment_id="EXP-REPEATED",
    )
    assert repeated.status == "brainstorm_failed"
    assert "novelty_signature" in repeated.error
    assert registry.get_experiment("EXP-REPEATED") is None
    registry.close()


def test_orchestrator_duplicate_id_is_deterministically_rejected(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    orchestrator = _orchestrator(tmp_path, registry)
    context = ResearchContext(research_question="q")
    assert orchestrator.research_step(context, experiment_id="EXP-DUP").status == "completed"
    second = orchestrator.research_step(context, experiment_id="EXP-DUP")
    assert second.status == "reservation_failed"
    registry.close()


def test_orchestrator_auto_id_continues_across_instances(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    context = ResearchContext(research_question="q")
    first = _orchestrator(tmp_path, registry).research_step(context)
    second = _orchestrator(tmp_path, registry).research_step(context)
    assert first.result["experiment_id"] == "EXP-AUTO-0001"
    assert second.result["experiment_id"] == "EXP-AUTO-0002"
    registry.close()
