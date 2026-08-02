from scripts import run_colab
from scripts.run_colab import _resolve_meta_path, _stage_data_files


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


def test_stage_data_files_copies_to_local_directory(tmp_path):
    drive_dir = tmp_path / "drive"
    local_dir = tmp_path / "local"
    drive_dir.mkdir()
    data_path = drive_dir / "FI2010_normalised.npy"
    meta_path = drive_dir / "FI2010_meta.json"
    data_path.write_bytes(b"npy-data")
    meta_path.write_text('{"rows": 1}', encoding="utf-8")

    local_data, local_meta = _stage_data_files(data_path, meta_path, local_dir)

    assert local_data == local_dir / data_path.name
    assert local_meta == local_dir / meta_path.name
    assert local_data.read_bytes() == b"npy-data"
    assert local_meta.read_text(encoding="utf-8") == '{"rows": 1}'


def test_stage_data_files_reuses_matching_local_copy(tmp_path, monkeypatch):
    drive_dir = tmp_path / "drive"
    local_dir = tmp_path / "local"
    drive_dir.mkdir()
    data_path = drive_dir / "FI2010_normalised.npy"
    meta_path = drive_dir / "FI2010_normalised_meta.json"
    data_path.write_bytes(b"npy-data")
    meta_path.write_text('{"rows": 1}', encoding="utf-8")
    _stage_data_files(data_path, meta_path, local_dir)

    def fail_copy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("不应重复复制未变化的文件")

    monkeypatch.setattr(run_colab.shutil, "copy2", fail_copy)
    local_data, local_meta = _stage_data_files(data_path, meta_path, local_dir)

    assert local_data == local_dir / data_path.name
    assert local_meta == local_dir / meta_path.name
