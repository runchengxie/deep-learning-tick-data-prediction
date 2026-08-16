"""事件流正式训练存储清单与预检测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ticknet.eventstream.config import day_pack_paths
from ticknet.eventstream.storage_readiness import (
    build_storage_manifest,
    check_full_copy_capacity,
    verify_direct_remote_listing,
    verify_staged_dataset,
)
from ticknet.eventstream.storage_readiness import main as readiness_main


def _config(path: Path, day: int) -> Path:
    path.write_text(
        "\n".join(
            [
                f"train_start: {day}",
                f"train_end: {day}",
                f"val_start: {day + 1}",
                f"val_end: {day + 1}",
                f"test_start: {day + 2}",
                f"test_end: {day + 2}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _copy_day(source_root: Path, target_root: Path, source_day: int, target_day: int) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for kind, source in day_pack_paths(source_day, source_root).items():
        target = day_pack_paths(target_day, target_root)[kind]
        target.write_bytes(source.read_bytes())


def _universe(path: Path, days: list[int], fingerprint: str = "a" * 64) -> Path:
    path.write_text(
        json.dumps(
            {
                "days": len(days),
                "source_dataset_fingerprint": fingerprint,
                "universes": {str(day): ["000001"] for day in days},
            }
        ),
        encoding="utf-8",
    )
    return path


def _storage_inputs(
    tmp_path: Path,
    packed_day: dict,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    day = int(packed_day["day"])
    pack_root = tmp_path / "logical" / "pack"
    for offset in range(3):
        _copy_day(packed_day["pack_root"], pack_root, day, day + offset)
    universe = _universe(tmp_path / "universe.json", [day, day + 1, day + 2])
    h3 = tmp_path / "logical" / "fold-labels" / "h3.parquet"
    h5 = tmp_path / "logical" / "fold-labels" / "h5.parquet"
    labels_manifest = tmp_path / "logical" / "fold-labels" / "manifest.json"
    h3.parent.mkdir(parents=True)
    h3.write_bytes(b"h3-labels")
    h5.write_bytes(b"h5-labels")
    labels_manifest.write_text('{"status":"complete"}', encoding="utf-8")
    return (
        _config(tmp_path / "config.yaml", day),
        pack_root,
        universe,
        {
            "fold-labels/manifest.json": labels_manifest,
            "fold-labels/h3.parquet": h3,
            "fold-labels/h5.parquet": h5,
        },
    )


def _storage_fixture(tmp_path: Path, packed_day: dict) -> tuple[dict, Path]:
    config, pack_root, universe, artifacts = _storage_inputs(tmp_path, packed_day)
    manifest = build_storage_manifest(
        config_path=config,
        pack_root=pack_root,
        universe_paths=[universe],
        artifacts=artifacts,
        locked_start=int(packed_day["day"]) + 3,
    )
    return manifest, tmp_path / "logical"


def test_build_storage_manifest_binds_months_splits_and_hashes(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    manifest, _root = _storage_fixture(tmp_path, packed_day)

    assert manifest["status"] == "complete"
    assert manifest["totals"]["days"] == 3
    assert manifest["totals"]["pack_files"] == 12
    assert manifest["totals"]["artifact_files"] == 3
    assert manifest["contract"]["splits"] == {
        "train": [int(packed_day["day"])],
        "validation": [int(packed_day["day"]) + 1],
        "oos": [int(packed_day["day"]) + 2],
    }
    first = manifest["months"][0]["files"][0]
    assert set(first["hashes"]) == {"sha256", "md5"}
    assert len(manifest["inventory_sha256"]) == 64


def test_build_storage_manifest_rejects_missing_pack_file(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    day = int(packed_day["day"])
    universe = _universe(tmp_path / "universe.json", [day, day + 1, day + 2])
    artifacts = {}
    for name in ("manifest.json", "h3.parquet", "h5.parquet"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[f"fold-labels/{name}"] = path
    with pytest.raises(FileNotFoundError, match="缺少 pack 文件"):
        build_storage_manifest(
            config_path=_config(tmp_path / "config.yaml", day),
            pack_root=packed_day["pack_root"],
            universe_paths=[universe],
            artifacts=artifacts,
            locked_start=day + 3,
        )


def test_staged_preflight_detects_content_drift(tmp_path: Path, packed_day: dict) -> None:
    manifest, root = _storage_fixture(tmp_path, packed_day)
    report = verify_staged_dataset(manifest, root)
    assert report["files"] == 15

    first_path = root / manifest["months"][0]["files"][0]["path"]
    first_path.write_bytes(first_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="不一致"):
        verify_staged_dataset(manifest, root)


def test_direct_remote_preflight_requires_complete_matching_listing(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    manifest, _root = _storage_fixture(tmp_path, packed_day)
    records = [
        *[record for month in manifest["months"] for record in month["files"]],
        *manifest["artifacts"],
    ]
    listing = [
        {
            "Path": record["path"],
            "Size": record["bytes"],
            "Hashes": {"md5": record["hashes"]["md5"]},
        }
        for record in records
    ]
    assert verify_direct_remote_listing(manifest, listing)["status"] == "complete"

    listing.pop()
    with pytest.raises(FileNotFoundError, match="远端缺少"):
        verify_direct_remote_listing(manifest, listing)


def test_full_copy_capacity_preflight_rejects_insufficient_disk(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    manifest, _root = _storage_fixture(tmp_path, packed_day)
    required = int(manifest["totals"]["bytes"])
    with pytest.raises(RuntimeError, match="运行盘不足"):
        check_full_copy_capacity(
            manifest,
            tmp_path,
            reserve_bytes=1,
            headroom_ratio=1.0,
            available_bytes=required,
        )
    report = check_full_copy_capacity(
        manifest,
        tmp_path,
        reserve_bytes=1,
        headroom_ratio=1.0,
        available_bytes=required + 1,
    )
    assert report["status"] == "complete"


def test_cli_verifies_staged_dataset_and_writes_report(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    manifest, root = _storage_fixture(tmp_path, packed_day)
    manifest_path = tmp_path / "storage-manifest.json"
    output_path = tmp_path / "staged-preflight.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness_main(
        [
            "verify-staged",
            "--manifest",
            str(manifest_path),
            "--root",
            str(root),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["inventory_sha256"] == manifest["inventory_sha256"]


def test_cli_build_writes_complete_storage_manifest(
    tmp_path: Path,
    packed_day: dict,
) -> None:
    config, pack_root, universe, artifacts = _storage_inputs(tmp_path, packed_day)
    output_path = tmp_path / "storage-manifest.json"
    arguments = [
        "build",
        "--config",
        str(config),
        "--pack-root",
        str(pack_root),
        "--universe",
        str(universe),
    ]
    for logical_path, source in artifacts.items():
        arguments.extend(["--artifact", f"{logical_path}={source}"])
    arguments.extend(
        [
            "--locked-start",
            str(int(packed_day["day"]) + 3),
            "--output",
            str(output_path),
        ]
    )

    readiness_main(arguments)

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["totals"]["files"] == 15
