import importlib

from tests.test_research import _spec
from ticknet.research.registry import ExperimentRegistry


def test_export_counts_states_seeds_and_missing_families_without_artifacts(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    for identity, status in [
        ("rejected", "failed"),
        ("unfinished", "running"),
        ("done", "completed"),
    ]:
        registry.create_experiment(
            identity,
            _spec(),
            status=status,
            git_sha="synthetic",
            artifact_dir=str(tmp_path / identity),
        )
    registry.start_run("unfinished", 0)
    module = importlib.import_module("ticknet.research.trial_accounting")
    report = module.export_trial_accounting(registry, family_by_experiment={"done": "family-a"})
    assert report["schema_version"] == "ticknet.trial-accounting.v1"
    assert report["experiment_count"] == 3
    assert report["unfinished_experiments"] == 1
    assert report["unmapped_experiments"] == ["rejected", "unfinished"]
    assert report["experiments"][2]["runs"][0]["status"] == "running"
    assert report["corrected_significance"] is None
    assert "artifact_dir" not in str(report)
    registry.close()


def test_accounting_cli_exports_explicit_family_mapping(tmp_path, capsys):
    import json

    from ticknet.research.cli import main

    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    registry.create_experiment(
        "attempt",
        _spec(),
        status="proposed",
        git_sha="synthetic",
        artifact_dir=str(tmp_path / "absent"),
    )
    registry.close()
    mapping = tmp_path / "families.json"
    mapping.write_text(json.dumps({"attempt": "family-a"}), encoding="utf-8")
    main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "export-trial-accounting",
            "--family-map",
            str(mapping),
        ]
    )
    assert json.loads(capsys.readouterr().out)["experiments"][0]["family_id"] == "family-a"
