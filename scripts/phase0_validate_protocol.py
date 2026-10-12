"""Fail-closed validation for the exploratory Phase 0 protocol draft.

This checks repository configuration only. It never constructs MiniCraft,
loads a checkpoint, or accesses train/dev/Confirm seeds.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "phase0_protocol_draft.yaml"
SEEDS_PATH = ROOT / "configs" / "seeds.yaml"


class ProtocolValidationError(ValueError):
    """Raised when protocol configuration is unsafe or inconsistent."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{label} must be a mapping")
    return value


def _range(value: Any, label: str) -> tuple[int, int]:
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in value)
            or value[0] > value[1]):
        raise ProtocolValidationError(f"{label} must be an increasing [min, max] range")
    return value[0], value[1]


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            return _mapping(yaml.safe_load(stream), str(path))
    except FileNotFoundError as exc:
        raise ProtocolValidationError(f"missing protocol file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ProtocolValidationError(f"invalid YAML in {path}: {exc}") from exc


def validate_protocol(protocol_path: Path = PROTOCOL_PATH,
                      seeds_path: Path = SEEDS_PATH) -> dict[str, Any]:
    """Validate the current draft and return a summary without touching seeds."""
    protocol = _load(protocol_path)
    seeds = _load(seeds_path)

    if protocol.get("status") != "DRAFT_NOT_LOCKED":
        raise ProtocolValidationError(
            "Phase 0 protocol must remain explicitly DRAFT_NOT_LOCKED until review")

    protocol_splits = _mapping(protocol.get("data_splits"), "protocol.data_splits")
    seed_splits = _mapping(seeds.get("seeds"), "seeds")
    ranges: dict[str, tuple[int, int]] = {}
    for split in ("train", "dev"):
        protocol_split = _mapping(
            protocol_splits.get(split), f"protocol.data_splits.{split}")
        seed_split = _mapping(seed_splits.get(split), f"seeds.{split}")
        ranges[split] = _range(protocol_split.get("range"), f"{split} range")
        if ranges[split] != _range(seed_split.get("range"), f"seeds.{split}.range"):
            raise ProtocolValidationError(
                f"{split} range differs between protocol and configs/seeds.yaml")
        if protocol_split.get("allowed_for") in (None, []):
            raise ProtocolValidationError(f"{split} split has no allowed uses")

    if ranges["train"][1] >= ranges["dev"][0]:
        raise ProtocolValidationError("train and dev ranges overlap")

    confirm = _mapping(protocol_splits.get("confirm"), "protocol.data_splits.confirm")
    seed_confirm = _mapping(seed_splits.get("confirm"), "seeds.confirm")
    for label, mapping, allowed in (
        ("protocol.data_splits.confirm", confirm, {"range", "status"}),
        ("seeds.confirm", seed_confirm, {"range", "status", "description", "note"}),
    ):
        unexpected = set(mapping) - allowed
        if unexpected:
            raise ProtocolValidationError(
                f"{label} contains unsupported Confirm allocation fields: "
                f"{sorted(unexpected)}")
    if confirm.get("range") is not None or seed_confirm.get("range") is not None:
        raise ProtocolValidationError("Confirm must remain unset until protocol approval")
    if confirm.get("status") != "NOT_DEFINED_OR_ACCESSED":
        raise ProtocolValidationError(
            "protocol Confirm status must remain NOT_DEFINED_OR_ACCESSED")
    if seed_confirm.get("status") not in {"CHƯA CHỐT", "NOT_DEFINED_OR_ACCESSED"}:
        raise ProtocolValidationError("configs/seeds.yaml must not grant Confirm access")

    metrics = _mapping(protocol.get("metrics"), "protocol.metrics")
    if metrics.get("thresholds") != (
            "TO_LOCK_AFTER_PROTOCOL_REVIEW; no threshold encoded here"):
        raise ProtocolValidationError("numeric Phase 0 thresholds must remain unset")
    provenance = _mapping(protocol.get("provenance"), "protocol.provenance")
    if provenance.get("phase5_thresholds_or_test_seeds") != "NOT_DEFINED_OR_ACCESSED":
        raise ProtocolValidationError("test seeds and Phase 5 thresholds must remain unset")

    return {
        "status": protocol["status"],
        "train_range": list(ranges["train"]),
        "dev_range": list(ranges["dev"]),
        "confirm_range": None,
        "confirm_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    args = parser.parse_args()
    try:
        summary = validate_protocol(args.protocol, args.seeds)
    except ProtocolValidationError as exc:
        parser.error(str(exc))
    print(
        "Phase 0 protocol validation passed: "
        f"{summary['status']}; train={summary['train_range']}, "
        f"dev={summary['dev_range']}, Confirm=unset."
    )


if __name__ == "__main__":
    main()
