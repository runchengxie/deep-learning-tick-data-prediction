from scripts.run_colab import _resolve_meta_path


def test_resolve_meta_path_accepts_existing_short_name(tmp_path):
    short_path = tmp_path / "FI2010_meta.json"
    short_path.touch()

    assert _resolve_meta_path(tmp_path) == short_path


def test_resolve_meta_path_prefers_standard_name(tmp_path):
    standard_path = tmp_path / "FI2010_normalised_meta.json"
    short_path = tmp_path / "FI2010_meta.json"
    standard_path.touch()
    short_path.touch()

    assert _resolve_meta_path(tmp_path) == standard_path
