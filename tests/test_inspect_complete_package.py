import hashlib
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import inspect_complete_package as inspector
from scripts.inspect_complete_package import inspect_package, main


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


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "artifacts/model:stream.pt",
        "artifacts/CON.json",
        "artifacts/CON .json",
        "artifacts/trailing.",
        "artifacts/trailing ",
    ],
)
def test_rejects_windows_special_path_aliases(tmp_path: Path, unsafe_name: str):
    archive = tmp_path / "windows-path.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(unsafe_name, b"unsafe")
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


def test_rejects_casefolded_directory_prefix_aliases_in_both_orders(tmp_path: Path):
    entry_orders = (
        (("Game/Mods/", b""), ("game/mods/example.json", b"{}")),
        (("game/mods/example.json", b"{}"), ("Game/Mods/", b"")),
    )
    for index, entries in enumerate(entry_orders):
        archive = tmp_path / f"directory-prefix-alias-{index}.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for name, payload in entries:
                zf.writestr(name, payload)
        with pytest.raises(ValueError, match="case-insensitive ZIP directory prefix collision"):
            inspect_package(archive)


def test_rejects_unicode_normalized_directory_prefix_aliases(tmp_path: Path):
    archive = tmp_path / "unicode-directory-prefix-alias.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifacts/Caf\u00e9/", b"")
        zf.writestr("artifacts/cafe\u0301/skill.json", "{}")
    with pytest.raises(ValueError, match="case-insensitive ZIP directory prefix collision"):
        inspect_package(archive)


def test_rejects_oversized_archive_before_hashing(tmp_path: Path):
    archive = tmp_path / "oversized-archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1}')
    with patch("scripts.inspect_complete_package.MAX_ARCHIVE_BYTES", 1):
        with patch(
            "scripts.inspect_complete_package.hashlib.sha256",
            side_effect=AssertionError("oversized archive was hashed"),
        ):
            with pytest.raises(ValueError, match="inspection size limit"):
                inspect_package(archive)


def test_rejects_duplicate_json_keys(tmp_path: Path):
    archive = tmp_path / "duplicate-json.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_package(archive)


def test_rejects_deeply_nested_json_as_a_package_validation_error(tmp_path: Path):
    archive = tmp_path / "deeply-nested-json.zip"
    payload = "[" * 10000 + "0" + "]" * 10000
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", payload)
    with pytest.raises(ValueError, match="invalid JSON file manifest.json"):
        inspect_package(archive)



def test_rejects_unicode_canonical_file_parent_collision(tmp_path: Path):
    archive = tmp_path / "unicode-parent.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifacts/caf\u00e9", b"file")
        zf.writestr("artifacts/cafe\u0301/child.json", "{}")
    with pytest.raises(ValueError, match="file/directory path collision"):
        inspect_package(archive)


def test_rejects_unicode_canonical_equivalent_paths(tmp_path: Path):
    archive = tmp_path / "unicode-collision.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifacts/caf\u00e9.json", "{}")
        zf.writestr("artifacts/cafe\u0301.json", "{}")
    with pytest.raises(ValueError, match="case-insensitive ZIP path collision"):
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



def test_rejects_archive_mutation_during_inspection(tmp_path: Path):
    archive = tmp_path / "changing.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1}')
    original_read = zipfile.ZipExtFile.read
    mutated = False

    def append_during_read(member, size=-1):
        nonlocal mutated
        result = original_read(member, size)
        if not mutated:
            with archive.open("ab") as changed:
                changed.write(b"changed while inspecting")
            mutated = True
        return result

    with patch.object(zipfile.ZipExtFile, "read", append_during_read):
        with pytest.raises(ValueError, match="archive changed during inspection"):
            inspect_package(archive)


def test_cli_refuses_to_overwrite_input_archive(tmp_path: Path):
    archive = tmp_path / "Soul_AI_Complete_Package.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1}')
    original = archive.read_bytes()
    with patch(
        "sys.argv",
        ["inspect_complete_package.py", str(archive), "--output", str(archive)],
    ):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2
    assert archive.read_bytes() == original


def test_cli_refuses_output_symlink_to_input_archive(tmp_path: Path):
    archive = tmp_path / "Soul_AI_Complete_Package.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1}')
    alias = tmp_path / "inventory.json"
    alias.symlink_to(archive)
    with patch(
        "sys.argv",
        ["inspect_complete_package.py", str(archive), "--output", str(alias)],
    ):
        with pytest.raises(SystemExit) as error:
            main()
    assert error.value.code == 2
    assert zipfile.is_zipfile(archive)

def test_cli_preserves_previous_report_if_atomic_replace_fails(tmp_path: Path):
    archive = tmp_path / "Soul_AI_Complete_Package.zip"
    output = tmp_path / "inventory.json"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", '{"schema_version":1}')
    output.write_text("previous report\\n", encoding="utf-8")
    with patch(
        "sys.argv",
        ["inspect_complete_package.py", str(archive), "--output", str(output)],
    ), patch.object(inspector.os, "replace", side_effect=OSError("simulated replace failure")):
        with pytest.raises(OSError, match="simulated replace failure"):
            main()
    assert output.read_text(encoding="utf-8") == "previous report\\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "Soul_AI_Complete_Package.zip", "inventory.json"
    ]

