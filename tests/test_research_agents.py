"""研究 Agent 测试：Brainstorm / Critic / Orchestrator / LLM 抽象。"""

import pytest

from ticknet.research.agents.brainstorm import BrainstormAgent
from ticknet.research.agents.client import TemplateClient, make_client
from ticknet.research.agents.context import ResearchContext
from ticknet.research.agents.critic import CriticAgent
from ticknet.research.agents.orchestrator import ResearchOrchestrator
from ticknet.research.registry import ExperimentRegistry
from ticknet.research.runner import ExperimentRunner
from ticknet.research.spec import ExperimentSpec


def test_make_client_template_and_unknown():
    assert isinstance(make_client("template"), TemplateClient)
    with pytest.raises(ValueError, match="未知 provider"):
        make_client("unknown_provider")


def test_template_client_echoes_user_prompt():
    client = TemplateClient()
    output = client.generate("system", "hello", temperature=0.0)
    assert output == "hello"


def test_brainstorm_template_default_proposal():
    agent = BrainstormAgent(TemplateClient())
    context = ResearchContext(research_question="测试问题")
    spec = agent.propose(context)
    assert spec.experiment_type == "ablation"
    assert spec.config_overrides == {"regression_loss_weight": 0.2}
    spec.validate()


def test_brainstorm_template_uses_anomaly():
    agent = BrainstormAgent(TemplateClient())
    context = ResearchContext(
        research_question="测试",
        open_anomalies=[
            {
                "type": "tail_return_concentration",
                "severity": "high",
                "detail": "top 5 日贡献 121%",
            }
        ],
    )
    spec = agent.propose(context)
    assert spec.experiment_type == "data_audit"
    assert "极端" in spec.hypothesis


def test_brainstorm_llm_path_parses_json():
    class FakeClient(TemplateClient):
        def generate(self, system_prompt, user_prompt, *, temperature=0.0):
            return (
                '{"hypothesis": "h", "rationale": "r", '
                '"falsification_condition": "f", "experiment_type": "ablation", '
                '"config_overrides": {"lr": 0.0005}, "seeds": [0, 1], '
                '"primary_metric": "daily_rank_ic_mean", "expected_direction": "increase"}'
            )

    agent = BrainstormAgent(FakeClient())
    spec = agent.propose(ResearchContext(research_question="q"))
    assert spec.experiment_type == "ablation"
    assert spec.config_overrides == {"lr": 0.0005}
    assert spec.seeds == (0, 1)


def test_critic_rejects_missing_falsification():
    critic = CriticAgent()
    spec = ExperimentSpec(
        hypothesis="h",
        experiment_type="ablation",
        base_config="configs/nextday.yaml",
        seeds=(0,),
        falsification_condition="",
    )
    critique = critic.review(spec)
    assert not critique.approved
    assert any("falsification" in issue for issue in critique.issues)


def test_critic_approves_complete_spec():
    critic = CriticAgent()
    spec = ExperimentSpec(
        hypothesis="h",
        experiment_type="ablation",
        base_config="configs/nextday.yaml",
        seeds=(0,),
        rationale="r",
        falsification_condition="若 IC 无改善则否定",
    )
    critique = critic.review(spec)
    assert critique.approved


def _make_harness(tmp_path):
    """创建含合法 base_config + manifest 的临时测试环境。"""
    import json

    manifest_dir = tmp_path / "data"
    manifest_dir.mkdir()
    manifest = manifest_dir / "manifest.json"
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
    base_config = tmp_path / "base.yaml"
    base_config.write_text(f"manifest_path: {manifest}\n", encoding="utf-8")
    return manifest


def test_orchestrator_full_loop_with_mock_runner(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _manifest = _make_harness(tmp_path)
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        entry_points={
            "ticknet-nextday-train": ["python", "-c", "print('{\\\"daily_rank_ic_mean\\\": 0.01}')"]
        },
    )
    orchestrator = ResearchOrchestrator(
        registry,
        brainstorm=BrainstormAgent(TemplateClient(), default_base_config="base.yaml"),
        critic=CriticAgent(),
        runner=runner,
    )
    step = orchestrator.research_step(
        ResearchContext(research_question="q"),
        experiment_id="EXP-AUTO-TEST",
    )
    assert step.status == "completed"
    assert step.result is not None
    registry.close()


def test_orchestrator_records_critic_review(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    _manifest = _make_harness(tmp_path)
    runner = ExperimentRunner(
        registry,
        repository_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        entry_points={"ticknet-nextday-train": ["python", "-c", "print('{}')"]},
    )
    orchestrator = ResearchOrchestrator(
        registry,
        brainstorm=BrainstormAgent(TemplateClient(), default_base_config="base.yaml"),
        critic=CriticAgent(),
        runner=runner,
    )
    step = orchestrator.research_step(
        ResearchContext(research_question="q"),
        experiment_id="EXP-AUTO-REV",
    )
    assert step.status == "completed"
    registry.close()
