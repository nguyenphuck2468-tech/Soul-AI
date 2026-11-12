import hashlib
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.inspect_complete_package import inspect_package


def test_inventory_hashes_model_files_without_loading_them(tmp_path: Path):
    archive = tmp_path / "Soul_AI_Complete_Package.zip"
    weights = b"not a real torch file; inspection must only hash bytes"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifacts/g31/g31_full_trained.pt", weights)
        zf.writestr("artifacts/phase1_compressed/manifest.json", '{"schema_version":1}')
    report = inspect_package(archive)
    model = next(item for item in report["inventory"] if item["path"].endswith(".pt"))
    assert model["sha256"] == hashlib.sha256(weights).hexdigest()
    assert model["json_validated"] is False
    assert report["summary"]["files"] == 2


def test_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.json", "{}")
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        inspect_package(archive)


def test_rejects_case_insensitive_duplicate_paths(tmp_path: Path):
    archive = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Artifacts/skill.json", "{}")
        zf.writestr("artifacts/SKILL.json", "{}")
    with pytest.raises(ValueError, match="case-insensitive ZIP path collision"):
        inspect_package(archive)


def test_rejects_file_used_as_parent_directory(tmp_path: Path):
    archive = tmp_path / "parent.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifacts/phase1", b"file")
        zf.writestr("artifacts/phase1/skill.json", "{}")
    with pytest.raises(ValueError, match="file/directory path collision"):
        inspect_package(archive)


def test_rejects_duplicate_json_keys(tmp_path: Path):
    archive = tmp_path / "duplicate-json.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_package(archive)


def test_rejects_too_many_entries(tmp_path: Path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one.bin", b"1")
        zf.writestr("two.bin", b"2")
    with patch("scripts.inspect_complete_package.MAX_ARCHIVE_ENTRIES", 1):
        with pytest.raises(ValueError, match="too many ZIP entries"):
            inspect_package(archive)


def test_rejects_oversized_expansion(tmp_path: Path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("payload.bin", b"12345")
    with patch("scripts.inspect_complete_package.MAX_TOTAL_UNCOMPRESSED_BYTES", 4):
        with pytest.raises(ValueError, match="total uncompressed size limit"):
            inspect_package(archive)


def test_rejects_symlink_entries(tmp_path: Path):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("artifacts/link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, "target")
    with pytest.raises(ValueError, match="symlink entry refused"):
        inspect_package(archive)
