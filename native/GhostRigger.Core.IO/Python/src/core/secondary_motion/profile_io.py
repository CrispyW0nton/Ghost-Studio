"""Deterministic profile IO and raw Odyssey dangly-array verification."""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.core.project.secondary_motion_profile import SecondaryMotionProfile


ODYSSEY_MDL_DATA_BASE = 12
DANGLY_HEADER_SIZE = 28
MAX_DANGLY_VERTICES = 1_000_000


@dataclass(frozen=True)
class DanglyPayloadVerification:
    valid: bool
    constraints: tuple[float, ...] = ()
    rest_positions: tuple[tuple[float, float, float], ...] = ()
    anchor_indices: tuple[int, ...] = ()
    errors: tuple[str, ...] = ()


def _profile(value: SecondaryMotionProfile | Mapping[str, Any]) -> SecondaryMotionProfile:
    return value if isinstance(value, SecondaryMotionProfile) else SecondaryMotionProfile.from_dict(value)


def serialize_secondary_motion_profile(value: SecondaryMotionProfile | Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON suitable for hashing and merge-safe registry files."""

    return (json.dumps(_profile(value).to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def deserialize_secondary_motion_profile(value: bytes | bytearray | memoryview | str) -> SecondaryMotionProfile:
    raw = bytes(value).decode("utf-8") if not isinstance(value, str) else value
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("Secondary-motion profile must be a JSON object.")
    return SecondaryMotionProfile.from_dict(parsed)


def write_secondary_motion_profile(value: SecondaryMotionProfile | Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a deterministic standalone profile artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(serialize_secondary_motion_profile(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return target


def read_secondary_motion_profile(path: str | Path) -> SecondaryMotionProfile:
    return deserialize_secondary_motion_profile(Path(path).read_bytes())


def verify_raw_dangly_payload(
    mdl: bytes | bytearray | memoryview,
    *,
    header_abs: int,
    vertex_count: int,
    base_offset: int = ODYSSEY_MDL_DATA_BASE,
) -> DanglyPayloadVerification:
    """Read back a binary dangly header without relying on PyKotor's offset bug.

    Retail stores constraints in the 0..255 range. They are normalized here to
    0..1. A zero constraint is the fixed-anchor semantic; all nonzero values are
    movable. Both the constraint and local rest-position arrays are mandatory.
    """

    raw = bytes(mdl)
    errors: list[str] = []
    if not 0 <= int(vertex_count) <= MAX_DANGLY_VERTICES:
        return DanglyPayloadVerification(False, errors=("vertex_count_out_of_bounds",))
    if header_abs < 0 or header_abs + DANGLY_HEADER_SIZE > len(raw):
        return DanglyPayloadVerification(False, errors=("dangly_header_out_of_bounds",))

    constraint_rel, count, count2, displacement, tightness, period, rest_rel = struct.unpack_from(
        "<IIIfffI", raw, header_abs
    )
    if count != count2:
        errors.append("constraint_count_pair_mismatch")
    if count != vertex_count:
        errors.append("vertex_count_mismatch")
    if not all(math.isfinite(value) for value in (displacement, tightness, period)):
        errors.append("non_finite_dangly_parameters")

    constraint_abs = int(base_offset) + int(constraint_rel)
    rest_abs = int(base_offset) + int(rest_rel)
    constraint_end = constraint_abs + int(count) * 4
    rest_end = rest_abs + int(count) * 12
    if constraint_rel <= 0 or constraint_abs < 0 or constraint_end > len(raw):
        errors.append("constraint_array_out_of_bounds")
    if rest_rel <= 0 or rest_abs < 0 or rest_end > len(raw):
        errors.append("rest_array_out_of_bounds")
    if errors:
        return DanglyPayloadVerification(False, errors=tuple(dict.fromkeys(errors)))

    stored_constraints = struct.unpack_from(f"<{count}f", raw, constraint_abs) if count else ()
    if any(not math.isfinite(value) or value < 0.0 or value > 255.0 for value in stored_constraints):
        errors.append("invalid_constraint_value")
    constraints = tuple(float(value) / 255.0 for value in stored_constraints)
    rest_positions = tuple(
        struct.unpack_from("<fff", raw, rest_abs + index * 12)
        for index in range(count)
    )
    if any(not all(math.isfinite(component) for component in position) for position in rest_positions):
        errors.append("non_finite_rest_position")
    anchors = tuple(index for index, value in enumerate(stored_constraints) if abs(float(value)) <= 1e-6)
    return DanglyPayloadVerification(
        valid=not errors,
        constraints=constraints,
        rest_positions=rest_positions,
        anchor_indices=anchors,
        errors=tuple(dict.fromkeys(errors)),
    )


__all__ = [
    "DANGLY_HEADER_SIZE",
    "DanglyPayloadVerification",
    "deserialize_secondary_motion_profile",
    "read_secondary_motion_profile",
    "serialize_secondary_motion_profile",
    "verify_raw_dangly_payload",
    "write_secondary_motion_profile",
]
