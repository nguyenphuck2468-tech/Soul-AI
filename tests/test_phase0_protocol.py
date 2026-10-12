from pathlib import Path

import pytest

from scripts.phase0_validate_protocol import (
    ProtocolValidationError,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_phase0_protocol_is_fail_closed():
    summary = validate_protocol()

    assert summary == {
        "status": "DRAFT_NOT_LOCKED",
        "train_range": [0, 999],
        "dev_range": [1000, 1999],
        "confirm_range": None,
        "confirm_accessed": False,
    }


def test_validator_rejects_confirm_range(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        (ROOT / "configs" / "phase0_protocol_draft.yaml").read_text(
            encoding="utf-8"
        ).replace(
            "confirm: {range: null, status:",
            "confirm: {range: [2000, 2001], status:",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolValidationError, match="Confirm must remain unset"):
        validate_protocol(protocol, ROOT / "configs" / "seeds.yaml")


def test_validator_rejects_protocol_seed_drift(tmp_path):
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(
        (ROOT / "configs" / "seeds.yaml").read_text(encoding="utf-8")
        .replace("range: [1000, 1999]", "range: [1001, 1999]"),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolValidationError, match="dev range differs"):
        validate_protocol(ROOT / "configs" / "phase0_protocol_draft.yaml", seeds)


def test_validator_rejects_confirm_allocation_alias(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        (ROOT / "configs" / "phase0_protocol_draft.yaml").read_text(
            encoding="utf-8"
        ).replace(
            "confirm: {range: null, status:",
            "confirm: {range: null, seed_list: [2000], status:",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolValidationError, match="allocation fields"):
        validate_protocol(protocol, ROOT / "configs" / "seeds.yaml")
