#!/usr/bin/env python3
"""Inspect a SOUL AI artifact ZIP without extracting or deserializing model files.

The output is an inventory and integrity report, not proof that a model is safe to execute
or that the package's claimed training metrics are reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_ARCHIVE_ENTRIES = 100_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 32 * 1024**2
CHUNK_SIZE = 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_name(info: zipfile.ZipInfo) -> tuple[str, bool]:
    raw = info.filename.replace(chr(92), "/")
    path = PurePosixPath(raw)
    is_directory = info.is_dir() or raw.endswith("/")
    if (
        not path.parts
        or raw.startswith("/")
        or chr(0) in raw
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"unsafe ZIP path: {info.filename}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK:
        raise ValueError(f"symlink entry refused: {info.filename}")
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"special ZIP entry refused: {info.filename}")
    if mode == stat.S_IFDIR and not is_directory:
        raise ValueError(f"directory mode/name mismatch: {info.filename}")
    if mode == stat.S_IFREG and is_directory:
        raise ValueError(f"file mode/name mismatch: {info.filename}")
    return "/".join(path.parts), is_directory


def inspect_package(archive_path: Path) -> dict[str, object]:
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise ValueError(f"not a readable ZIP archive: {archive_path}")

    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    seen_folded: set[str] = set()
    files_folded: set[str] = set()
    directories = 0
    total_uncompressed = 0
    archive_hash = hashlib.sha256()

    with archive_path.open("rb") as raw_archive:
        for chunk in iter(lambda: raw_archive.read(CHUNK_SIZE), b""):
            archive_hash.update(chunk)

    with zipfile.ZipFile(archive_path) as zf:
        entries = zf.infolist()
        if not entries:
            raise ValueError("empty ZIP archive")
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(
                f"too many ZIP entries: {len(entries)} (limit {MAX_ARCHIVE_ENTRIES})"
            )

        validated: list[tuple[zipfile.ZipInfo, str, bool]] = []
        for info in entries:
            name, is_directory = _validate_name(info)
            folded = name.casefold()
            if name in seen or folded in seen_folded:
                raise ValueError(f"duplicate or case-insensitive ZIP path collision: {info.filename}")
            parents = ["/".join(PurePosixPath(name).parts[:i]) for i in range(1, len(PurePosixPath(name).parts))]
            conflict = next((parent.casefold() for parent in parents if parent.casefold() in files_folded), None)
            if conflict is not None:
                raise ValueError(f"file/directory path collision: {info.filename} is below {conflict}")
            if not is_directory and any(existing.startswith(folded + "/") for existing in seen_folded):
                raise ValueError(f"file/directory path collision: {info.filename} replaces a directory")
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP entry refused: {info.filename}")
            if not is_directory:
                if info.file_size < 0 or info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                    raise ValueError(f"ZIP member exceeds size limit: {info.filename}")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise ValueError("ZIP archive exceeds total uncompressed size limit")
                files_folded.add(folded)
            else:
                directories += 1
            seen.add(name)
            seen_folded.add(folded)
            validated.append((info, name, is_directory))

        for info, name, is_directory in validated:
            if is_directory:
                inventory.append({"path": name, "kind": "directory"})
                continue
            digest = hashlib.sha256()
            json_bytes = bytearray() if name.lower().endswith(".json") else None
            actual_size = 0
            with zf.open(info) as source:
                while chunk := source.read(CHUNK_SIZE):
                    actual_size += len(chunk)
                    if actual_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                        raise ValueError(f"actual ZIP member size exceeds limit: {name}")
                    digest.update(chunk)
                    if json_bytes is not None:
                        if len(json_bytes) + len(chunk) > MAX_JSON_BYTES:
                            raise ValueError(f"JSON manifest exceeds inspection limit: {name}")
                        json_bytes.extend(chunk)
            if actual_size != info.file_size:
                raise ValueError(f"ZIP member size mismatch: {name}")
            if json_bytes is not None:
                try:
                    json.loads(bytes(json_bytes), object_pairs_hook=_reject_duplicate_keys)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"invalid JSON file {name}: {error}") from error
            inventory.append({
                "path": name,
                "kind": "file",
                "bytes": actual_size,
                "sha256": digest.hexdigest(),
                "json_validated": json_bytes is not None,
            })

    files = [item for item in inventory if item["kind"] == "file"]
    return {
        "schema_version": 1,
        "archive": archive_path.name,
        "archive_sha256": archive_hash.hexdigest(),
        "summary": {
            "entries": len(inventory),
            "files": len(files),
            "directories": directories,
            "uncompressed_bytes": total_uncompressed,
        },
        "inventory": inventory,
        "limitations": [
            "Model weights and NumPy archives are hashed but never loaded or deserialized.",
            "JSON syntax and duplicate keys are checked; package-specific manifest hash references are not interpreted.",
            "This inventory does not reproduce training metrics or establish model safety.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="SOUL AI artifact ZIP")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON inventory report")
    args = parser.parse_args()
    try:
        report = inspect_package(args.archive)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(2, f"Package inspection failed: {error}\n")
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
