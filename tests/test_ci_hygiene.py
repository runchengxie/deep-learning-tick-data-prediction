from pathlib import Path

from scripts.ci_hygiene import inspect_file


def test_ci_hygiene_accepts_valid_text_json_toml_and_yaml(tmp_path: Path) -> None:
    samples = {
        "note.md": "A clean note.\n",
        "state.json": '{"status": "ready"}\n',
        "project.toml": "[project]\nname = 'example'\n",
        "workflow.yml": "jobs:\n  check:\n    runs-on: ubuntu-latest\n",
    }
    for filename, content in samples.items():
        target = tmp_path / filename
        target.write_text(content, encoding="utf-8")
        assert inspect_file(target) == []


def test_ci_hygiene_rejects_whitespace_conflicts_and_bad_json(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text('{\n"x": 1, \n' + "<" * 7 + " branch\n}\n", encoding="utf-8")
    errors = inspect_file(target)
    assert any("trailing whitespace" in error for error in errors)
    assert any("merge-conflict" in error for error in errors)
    assert any("invalid JSON" in error for error in errors)


def test_ci_hygiene_rejects_large_files_and_private_key_markers(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")
    assert any("1 MiB" in error for error in inspect_file(large))

    private = tmp_path / "credential.txt"
    marker = "-" * 5 + "BEGIN PRIVATE KEY" + "-" * 5
    private.write_text(f"{marker}\nnot a real key\n{marker}\n", encoding="utf-8")
    assert any("private-key" in error for error in inspect_file(private))
