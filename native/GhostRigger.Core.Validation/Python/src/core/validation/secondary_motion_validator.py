"""Fail-closed validation and write-promotion gates for secondary motion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from src.core.project.secondary_motion_profile import (
    CURRENT_SECONDARY_MOTION_PROFILE_VERSION,
    SECONDARY_MOTION_BACKENDS,
    SECONDARY_MOTION_PROFILE_SCHEMA,
    SECONDARY_MOTION_RUNTIME_STATES,
    SecondaryMotionProfile,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_RESREF = re.compile(r"^[A-Za-z0-9_]{1,16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROHIBITED_KEY_PARTS = ("address", "pointer", "offset", "runtime_key")
_SOLVER_RANGES = {
    "gravity": (0.0, 20.0),
    "damping": (0.0, 50.0),
    "animation_follow": (0.0, 1.0),
    "collision_retention": (0.0, 1.0),
    "speed": (0.25, 4.0),
    "length_iterations": (1.0, 12.0),
    "bending_iterations": (0.0, 12.0),
}


@dataclass(frozen=True)
class SecondaryMotionValidationIssue:
    severity: str
    code: str
    message: str
    target: str = ""


@dataclass(frozen=True)
class SecondaryMotionValidationReport:
    issues: tuple[SecondaryMotionValidationIssue, ...]
    can_export_profile: bool
    can_enable_runtime_writes: bool
    runtime_state: str


def _finite_vector(value: Any, size: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value
    )


def _walk_keys(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            target = f"{path}.{key}" if path else str(key)
            yield target, str(key)
            yield from _walk_keys(item, target)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{path}.{index}" if path else str(index))


def validate_secondary_motion_profile(
    value: SecondaryMotionProfile | Mapping[str, Any],
    *,
    target_kind: str = "character",
) -> SecondaryMotionValidationReport:
    try:
        profile = value if isinstance(value, SecondaryMotionProfile) else SecondaryMotionProfile.from_dict(value)
    except (TypeError, ValueError) as exc:
        issue = SecondaryMotionValidationIssue("blocking", "invalid_profile_contract", str(exc))
        return SecondaryMotionValidationReport((issue,), False, False, "capture_only")

    data = profile.to_dict()
    issues: list[SecondaryMotionValidationIssue] = []

    def add(severity: str, code: str, message: str, target: str = "") -> None:
        issues.append(SecondaryMotionValidationIssue(severity, code, message, target))

    if profile.schema != SECONDARY_MOTION_PROFILE_SCHEMA or profile.schema_version != CURRENT_SECONDARY_MOTION_PROFILE_VERSION:
        add("blocking", "unsupported_profile_schema", "Secondary-motion profile schema is unsupported.", "schema")
    if not _IDENTIFIER.fullmatch(profile.profile_id):
        add("blocking", "invalid_profile_id", "Profile ID must be a stable 1-96 character identifier.", "profile_id")
    if profile.target_game not in {"K1", "K2"}:
        add("blocking", "invalid_target_game", "Target game must be K1 or K2.", "target_game")
    if profile.backend_capability_version != 1:
        add("blocking", "unsupported_backend_capability", "Runtime capability version is unsupported.", "backend_capability_version")

    for path, key in _walk_keys(data):
        folded = key.casefold()
        if any(part in folded for part in _PROHIBITED_KEY_PARTS):
            add("blocking", "unsafe_runtime_identity_field", "Profiles cannot store addresses, pointers, offsets, or runtime keys.", path)

    identity = profile.model_identity
    resrefs = [str(item).strip().lower() for item in identity.get("model_resrefs") or ()]
    if profile.regions and not resrefs:
        add("blocking", "missing_model_resref", "A profile with regions needs at least one exact model resref.", "model_identity.model_resrefs")
    if len(resrefs) > 1:
        add("blocking", "ambiguous_model_identity", "Runtime capability v1 requires exactly one model resref.", "model_identity.model_resrefs")
    for resref in resrefs:
        if not _RESREF.fullmatch(resref):
            add("blocking", "invalid_model_resref", "Model resrefs must be 1-16 engine-safe characters.", "model_identity.model_resrefs")
    for key in ("mdl_sha256", "mdx_sha256", "hierarchy_fingerprint"):
        value_text = str(identity.get(key) or "").lower()
        if profile.regions and (not _SHA256.fullmatch(value_text) or set(value_text) == {"0"}):
            add("blocking", "missing_exact_model_identity", f"{key} must be an exact SHA-256 value.", f"model_identity.{key}")
    if profile.regions and not str(identity.get("root_name") or "").strip():
        add("blocking", "missing_model_root", "The final KOTOR hierarchy root name is required.", "model_identity.root_name")

    region_ids: set[str] = set()
    mesh_backends: dict[str, str] = {}
    total_drivers = 0
    runtime_ordinals: set[int] = set()
    if len(profile.regions) > 9:
        add("blocking", "region_capacity_exceeded", "Runtime capability v1 supports at most 9 regions.", "regions")
    for index, region in enumerate(profile.regions):
        target = f"regions.{index}"
        region_id = str(region.get("region_id") or "")
        backend = str(region.get("backend") or "")
        if not _IDENTIFIER.fullmatch(region_id):
            add("blocking", "invalid_region_id", "Every region needs a stable identifier.", f"{target}.region_id")
        elif region_id in region_ids:
            add("blocking", "duplicate_region_id", f"Region ID {region_id!r} is duplicated.", f"{target}.region_id")
        region_ids.add(region_id)
        if backend not in SECONDARY_MOTION_BACKENDS:
            add("blocking", "invalid_region_backend", "Region backend must be runtime_pbd, native_dangly, or rigid.", f"{target}.backend")
        for mesh in region.get("mesh_membership") or ():
            mesh_name = str(mesh).strip().casefold()
            if not mesh_name:
                continue
            previous = mesh_backends.get(mesh_name)
            if previous and previous != backend and {previous, backend} != {"rigid"}:
                add("blocking", "mesh_backend_conflict", f"Mesh {mesh!r} is assigned to conflicting backends.", f"{target}.mesh_membership")
            elif previous and previous != backend:
                add("blocking", "mesh_backend_conflict", f"Mesh {mesh!r} is assigned to conflicting backends.", f"{target}.mesh_membership")
            mesh_backends[mesh_name] = backend

        drivers = list(region.get("drivers") or ())
        total_drivers += len(drivers)
        if backend == "runtime_pbd" and not drivers:
            add("blocking", "runtime_region_has_no_drivers", "Runtime PBD regions need a named driver chain.", f"{target}.drivers")
        if len(drivers) > 3:
            add("blocking", "chain_capacity_exceeded", "Runtime capability v1 supports at most 3 drivers per region.", f"{target}.drivers")
        driver_names: set[str] = set()
        for driver_index, driver in enumerate(drivers):
            driver_target = f"{target}.drivers.{driver_index}"
            name = str(driver.get("name") or "").strip()
            if not name or name.casefold() in driver_names:
                add("blocking", "invalid_driver_name", "Driver names must be present and unique within a region.", f"{driver_target}.name")
            driver_names.add(name.casefold())
            if not str(driver.get("parent_name") or "").strip():
                add("blocking", "missing_driver_parent", "Every driver needs its expected named parent.", f"{driver_target}.parent_name")
            for field_name, size in (("rest_position", 3), ("rest_quaternion", 4), ("rest_scale", 3)):
                if not _finite_vector(driver.get(field_name), size):
                    add("blocking", "invalid_driver_rest_transform", f"{field_name} must contain {size} finite values.", f"{driver_target}.{field_name}")
            quaternion = driver.get("rest_quaternion")
            if _finite_vector(quaternion, 4):
                norm = math.sqrt(sum(float(item) * float(item) for item in quaternion))
                if abs(norm - 1.0) > 0.01:
                    add("blocking", "invalid_driver_rest_quaternion", "Driver rest quaternions must be unit length.", f"{driver_target}.rest_quaternion")
            scale = driver.get("rest_scale")
            if _finite_vector(scale, 3):
                numeric_scale = [float(item) for item in scale]
                if min(numeric_scale) < 0.01 or max(numeric_scale) > 100.0 or max(numeric_scale) - min(numeric_scale) > 1e-4:
                    add("blocking", "invalid_driver_rest_scale", "Runtime drivers require a finite uniform scale from 0.01 to 100.", f"{driver_target}.rest_scale")
            if backend == "runtime_pbd":
                try:
                    ordinal = int(driver.get("offline_preorder"))
                except (TypeError, ValueError):
                    ordinal = -1
                if ordinal < 0 or ordinal >= 128:
                    add("blocking", "missing_driver_preorder", "Runtime drivers require a bounded offline pre-order index.", f"{driver_target}.offline_preorder")
                elif ordinal in runtime_ordinals:
                    add("blocking", "duplicate_driver_preorder", "Runtime driver pre-order indices must be unique.", f"{driver_target}.offline_preorder")
                runtime_ordinals.add(ordinal)
        colliders = list(region.get("colliders") or ())
        if len(colliders) > 16:
            add("blocking", "collider_capacity_exceeded", "Runtime capability v1 supports at most 16 colliders per region.", f"{target}.colliders")
        solver = dict(region.get("solver") or {})
        for key, limits in _SOLVER_RANGES.items():
            if key not in solver:
                continue
            try:
                numeric = float(solver[key])
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isfinite(numeric) or not limits[0] <= numeric <= limits[1]:
                add("blocking", "solver_parameter_out_of_range", f"{key} must remain between {limits[0]} and {limits[1]}.", f"{target}.solver.{key}")
    if total_drivers > 9:
        add("blocking", "write_capacity_exceeded", "Runtime capability v1 supports at most 9 driver writes per actor.", "regions")

    runtime = profile.runtime
    state = str(runtime.get("state") or "capture_only").lower()
    requested_write = bool(runtime.get("write_enabled")) or state == "write_enabled"
    if state not in SECONDARY_MOTION_RUNTIME_STATES:
        add("blocking", "invalid_runtime_state", "Runtime state must be capture_only, calibrated, or write_enabled.", "runtime.state")
        state = "capture_only"
    exact_evidence = all(
        bool(runtime.get(key))
        for key in ("model_identity_validated", "live_tree_validated", "visible_test_accepted")
    )
    lifecycle_ok = target_kind != "placeable" or bool(runtime.get("placeable_lifecycle_validated"))
    if requested_write and not exact_evidence:
        add("write_blocking", "runtime_write_evidence_incomplete", "Runtime writes remain disabled until exact read-only model and visible validation gates pass.", "runtime")
    if requested_write and target_kind == "placeable" and not lifecycle_ok:
        add("write_blocking", "placeable_runtime_lifecycle_unproven", "Placeable PBD remains capture-only until animated-instance and destruction/reload lifecycle proof passes.", "runtime")
    if requested_write and any(str(region.get("backend")) == "runtime_pbd" for region in profile.regions) and not str(runtime.get("capture_session") or ""):
        add("write_blocking", "runtime_capture_session_missing", "A bounded read-only calibration session is required before writes.", "runtime.capture_session")

    export_ok = not any(issue.severity == "blocking" for issue in issues)
    writes_ok = export_ok and requested_write and exact_evidence and lifecycle_ok and not any(
        issue.severity == "write_blocking" for issue in issues
    )
    return SecondaryMotionValidationReport(tuple(issues), export_ok, writes_ok, state)


__all__ = [
    "SecondaryMotionValidationIssue",
    "SecondaryMotionValidationReport",
    "validate_secondary_motion_profile",
]
