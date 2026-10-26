"""Qt-free discovery, presets, preview, and profile compilation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from src.core.project.secondary_motion_profile import SecondaryMotionProfile
from src.core.validation.secondary_motion_validator import validate_secondary_motion_profile


FIXED_STEP_SECONDS = 1.0 / 120.0
MAX_SUBSTEPS = 6

SECONDARY_MOTION_PRESETS: dict[str, dict[str, Any]] = {
    "robe_skirt": {"semantic_type": "robe", "backend": "runtime_pbd", "gravity": 4.25, "damping": 4.5, "animation_follow": 0.65, "collision_retention": 0.2},
    "sleeve": {"semantic_type": "sleeve", "backend": "runtime_pbd", "gravity": 4.25, "damping": 6.5, "animation_follow": 0.45, "collision_retention": 0.2},
    "cape": {"semantic_type": "cape", "backend": "runtime_pbd", "gravity": 4.25, "damping": 4.5, "animation_follow": 0.55, "collision_retention": 0.2},
    "ponytail_braid": {"semantic_type": "ponytail", "backend": "runtime_pbd", "gravity": 4.25, "damping": 6.5, "animation_follow": 0.45, "collision_retention": 0.2},
    "hair_lock": {"semantic_type": "front_hair_lock", "backend": "runtime_pbd", "gravity": 4.25, "damping": 6.5, "animation_follow": 0.45, "collision_retention": 0.2},
    "banner": {"semantic_type": "banner", "backend": "runtime_pbd", "gravity": 4.25, "damping": 4.5, "animation_follow": 0.35, "collision_retention": 0.2},
    "cable_hose": {"semantic_type": "cable", "backend": "runtime_pbd", "gravity": 4.25, "damping": 5.5, "animation_follow": 0.35, "collision_retention": 0.2},
    "native_dangly": {"semantic_type": "custom", "backend": "native_dangly"},
    "rigid": {"semantic_type": "custom", "backend": "rigid"},
}


def runtime_capability_descriptor() -> dict[str, Any]:
    """Single authoring/UI source for the accepted v1 runtime bounds."""

    return {
        "schema": "ghostrigger.secondary_motion_runtime_capabilities.v1",
        "capability_version": 1,
        "fixed_step_hz": 120,
        "max_substeps": MAX_SUBSTEPS,
        "max_actors": 8,
        "max_regions_per_actor": 9,
        "max_drivers_per_region": 3,
        "max_driver_writes_per_actor": 9,
        "max_colliders_per_region": 16,
        "max_profile_bytes": 262_144,
        "solver_ranges": {
            "gravity": [0.0, 20.0],
            "damping": [0.0, 50.0],
            "animation_follow": [0.0, 1.0],
            "collision_retention": [0.0, 1.0],
            "speed": [0.25, 4.0],
        },
        "write_policy": "model_specific_evidence_required",
        "pfbnm_reference_is_hardcoded": True,
    }


def discover_driver_chains(nodes: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    """Return deterministic named leaf chains without assuming stock bone names."""

    records = [dict(node) for node in nodes]
    parents = {str(row.get("name") or ""): str(row.get("parent_name") or "") for row in records if row.get("name")}
    children: dict[str, list[str]] = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)
    for values in children.values():
        values.sort(key=str.casefold)
    leaves = sorted((name for name in parents if not children.get(name)), key=str.casefold)
    chains: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for leaf in leaves:
        backwards = [leaf]
        cursor = parents.get(leaf, "")
        while cursor and cursor in parents and len(backwards) < 3:
            backwards.append(cursor)
            cursor = parents.get(cursor, "")
        chain = tuple(reversed(backwards))
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return tuple(chains)


def _vec_add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (float(left[0]) + float(right[0]), float(left[1]) + float(right[1]), float(left[2]) + float(right[2]))


def _vec_sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (float(left[0]) - float(right[0]), float(left[1]) - float(right[1]), float(left[2]) - float(right[2]))


def _vec_scale(value: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return (float(value[0]) * scalar, float(value[1]) * scalar, float(value[2]) * scalar)


def _length(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


class SecondaryMotionPreview:
    """Small deterministic position-based preview using the accepted step order."""

    def __init__(
        self,
        rest_positions: Sequence[Sequence[float]],
        *,
        pinned_indices: set[int] | frozenset[int] | None = None,
        gravity: Sequence[float] = (0.0, 0.0, -4.25),
        damping: float = 4.5,
        length_iterations: int = 4,
        bending_iterations: int = 1,
        colliders: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.rest_positions = tuple(tuple(float(item) for item in value[:3]) for value in rest_positions)
        if any(len(value) != 3 or not all(math.isfinite(item) for item in value) for value in self.rest_positions):
            raise ValueError("Preview rest positions must contain three finite values.")
        self.pinned_indices = frozenset(int(item) for item in (pinned_indices or set()))
        if any(index < 0 or index >= len(self.rest_positions) for index in self.pinned_indices):
            raise ValueError("Pinned preview index is out of bounds.")
        self.gravity = tuple(float(item) for item in gravity[:3])
        self.damping = max(0.0, float(damping))
        self.length_iterations = max(1, min(12, int(length_iterations)))
        self.bending_iterations = max(0, min(12, int(bending_iterations)))
        self.colliders = tuple(deepcopy(dict(item)) for item in colliders)
        self._segment_lengths = tuple(
            _length(_vec_sub(self.rest_positions[index], self.rest_positions[index - 1]))
            for index in range(1, len(self.rest_positions))
        )
        self._bend_lengths = tuple(
            _length(_vec_sub(self.rest_positions[index], self.rest_positions[index - 2]))
            for index in range(2, len(self.rest_positions))
        )
        self.reset()

    @property
    def positions(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self._positions)

    @property
    def velocities(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self._velocities)

    def reset(self) -> None:
        self._positions = list(self.rest_positions)
        self._velocities = [(0.0, 0.0, 0.0) for _ in self.rest_positions]
        self._accumulator = 0.0
        self.step_count = 0

    def advance(self, elapsed_seconds: float) -> int:
        elapsed = max(0.0, min(float(elapsed_seconds), FIXED_STEP_SECONDS * MAX_SUBSTEPS))
        self._accumulator += elapsed
        steps = 0
        while self._accumulator + 1e-12 >= FIXED_STEP_SECONDS and steps < MAX_SUBSTEPS:
            self.step(FIXED_STEP_SECONDS)
            self._accumulator -= FIXED_STEP_SECONDS
            steps += 1
        return steps

    def step(self, dt: float = FIXED_STEP_SECONDS) -> None:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Preview step must be a positive finite duration.")
        previous = list(self._positions)
        velocity_decay = math.exp(-self.damping * dt)
        for index in range(len(self._positions)):
            if index in self.pinned_indices:
                self._positions[index] = self.rest_positions[index]
                self._velocities[index] = (0.0, 0.0, 0.0)
                continue
            velocity = _vec_add(_vec_scale(self._velocities[index], velocity_decay), _vec_scale(self.gravity, dt))
            self._velocities[index] = velocity
            self._positions[index] = _vec_add(self._positions[index], _vec_scale(velocity, dt))

        for _ in range(self.length_iterations):
            for index, wanted in enumerate(self._segment_lengths, start=1):
                self._project_distance(index - 1, index, wanted)
        for _ in range(self.bending_iterations):
            for index, wanted in enumerate(self._bend_lengths, start=2):
                self._project_distance(index - 2, index, wanted)
        self._project_collisions()
        for index in range(len(self._positions)):
            if index in self.pinned_indices:
                self._positions[index] = self.rest_positions[index]
                self._velocities[index] = (0.0, 0.0, 0.0)
            else:
                self._velocities[index] = _vec_scale(_vec_sub(self._positions[index], previous[index]), 1.0 / dt)
        self.step_count += 1

    def _project_distance(self, left: int, right: int, wanted: float) -> None:
        delta = _vec_sub(self._positions[right], self._positions[left])
        distance = _length(delta)
        if distance <= 1e-9:
            return
        correction = _vec_scale(delta, (distance - wanted) / distance)
        left_pinned = left in self.pinned_indices
        right_pinned = right in self.pinned_indices
        if left_pinned and not right_pinned:
            self._positions[right] = _vec_sub(self._positions[right], correction)
        elif right_pinned and not left_pinned:
            self._positions[left] = _vec_add(self._positions[left], correction)
        elif not left_pinned and not right_pinned:
            half = _vec_scale(correction, 0.5)
            self._positions[left] = _vec_add(self._positions[left], half)
            self._positions[right] = _vec_sub(self._positions[right], half)

    def _project_collisions(self) -> None:
        for collider in self.colliders:
            kind = str(collider.get("shape") or "").lower()
            if kind == "plane":
                normal = tuple(float(item) for item in collider.get("normal") or (0.0, 0.0, 1.0))
                normal_length = _length(normal)
                if normal_length <= 1e-9:
                    continue
                normal = _vec_scale(normal, 1.0 / normal_length)
                distance = float(collider.get("distance") or 0.0)
                for index, position in enumerate(self._positions):
                    if index in self.pinned_indices:
                        continue
                    penetration = distance - sum(position[axis] * normal[axis] for axis in range(3))
                    if penetration > 0.0:
                        self._positions[index] = _vec_add(position, _vec_scale(normal, penetration))
            elif kind == "sphere":
                center = tuple(float(item) for item in collider.get("center") or (0.0, 0.0, 0.0))
                radius = max(0.0, float(collider.get("radius") or 0.0))
                for index, position in enumerate(self._positions):
                    if index in self.pinned_indices:
                        continue
                    delta = _vec_sub(position, center)
                    distance = _length(delta)
                    if 1e-9 < distance < radius:
                        self._positions[index] = _vec_add(center, _vec_scale(delta, radius / distance))


def compile_secondary_motion_profile(
    value: SecondaryMotionProfile | Mapping[str, Any],
    *,
    target_kind: str = "character",
) -> dict[str, Any]:
    """Compile names and rest evidence into guarded selectors; never emit keys."""

    profile = value if isinstance(value, SecondaryMotionProfile) else SecondaryMotionProfile.from_dict(value)
    report = validate_secondary_motion_profile(profile, target_kind=target_kind)
    if not report.can_export_profile:
        codes = ", ".join(issue.code for issue in report.issues if issue.severity == "blocking")
        raise ValueError(f"Secondary-motion profile is not compilable: {codes}")
    output = profile.to_dict()
    identity = deepcopy(dict(output.get("model_identity") or {}))
    identity["target_kind"] = str(target_kind or "character")
    output["model_identity"] = identity
    output["compiled"] = {
        "schema": "ghostrigger.secondary_motion_compiled_profile.v1",
        "capability_version": profile.backend_capability_version,
        "model_selector": {
            "model_resrefs": list(profile.model_identity.get("model_resrefs") or ()),
            "mdl_sha256": str(profile.model_identity.get("mdl_sha256") or ""),
            "mdx_sha256": str(profile.model_identity.get("mdx_sha256") or ""),
            "hierarchy_fingerprint": str(profile.model_identity.get("hierarchy_fingerprint") or ""),
            "root_name": str(profile.model_identity.get("root_name") or ""),
            "target_kind": str(target_kind or "character"),
        },
        "regions": [
            {
                "region_id": str(region.get("region_id") or ""),
                "backend": str(region.get("backend") or "rigid"),
                "drivers": [
                    {
                        "name": str(driver.get("name") or ""),
                        "parent_name": str(driver.get("parent_name") or ""),
                        "rest_position": list(driver.get("rest_position") or ()),
                        "rest_quaternion": list(driver.get("rest_quaternion") or ()),
                        "rest_scale": list(driver.get("rest_scale") or ()),
                        "offline_preorder": driver.get("offline_preorder"),
                    }
                    for driver in region.get("drivers") or ()
                ],
                "colliders": deepcopy(list(region.get("colliders") or ())),
                "solver": deepcopy(dict(region.get("solver") or {})),
            }
            for region in profile.regions
        ],
    }
    runtime = deepcopy(dict(output.get("runtime") or {}))
    runtime["state"] = "write_enabled" if report.can_enable_runtime_writes else "capture_only"
    runtime["write_enabled"] = bool(report.can_enable_runtime_writes)
    output["runtime"] = runtime
    return output


def apply_read_only_calibration(
    value: SecondaryMotionProfile | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    target_kind: str = "character",
) -> dict[str, Any]:
    """Record an exact capture PASS as calibrated evidence without enabling writes."""

    profile = value if isinstance(value, SecondaryMotionProfile) else SecondaryMotionProfile.from_dict(value)
    capture = deepcopy(dict(evidence or {}))
    if capture.get("schema") != "kotor_patch_manager.cloth_dynamics_k2.profile_capture.v1":
        raise ValueError("Read-only calibration evidence schema is unsupported.")
    if not bool(capture.get("passed")) or int(capture.get("writes") or 0) != 0:
        raise ValueError("Calibration requires a passing capture with writes=0.")
    if str(capture.get("profile_id") or "") != profile.profile_id:
        raise ValueError("Calibration profile ID does not match the authoring profile.")
    model_resrefs = [str(item).casefold() for item in profile.model_identity.get("model_resrefs") or ()]
    if str(capture.get("model_resref") or "").casefold() not in model_resrefs:
        raise ValueError("Calibration model does not match the exact profile model identity.")
    driver_count = sum(len(region.get("drivers") or ()) for region in profile.regions)
    if int(capture.get("expected_nodes") or -1) != driver_count:
        raise ValueError("Calibration driver count does not match the compiled profile.")

    output = profile.to_dict()
    validation_evidence = deepcopy(dict(output.get("validation_evidence") or {}))
    validation_evidence["read_only_capture"] = capture
    output["validation_evidence"] = validation_evidence
    runtime = deepcopy(dict(output.get("runtime") or {}))
    runtime.update({
        "state": "calibrated",
        "capture_session": str(capture.get("capture_session") or ""),
        "model_identity_validated": True,
        "live_tree_validated": True,
        "visible_test_accepted": False,
        "write_enabled": False,
    })
    if target_kind == "placeable":
        runtime["placeable_lifecycle_validated"] = False
    output["runtime"] = runtime
    return output


__all__ = [
    "FIXED_STEP_SECONDS",
    "MAX_SUBSTEPS",
    "SECONDARY_MOTION_PRESETS",
    "SecondaryMotionPreview",
    "apply_read_only_calibration",
    "compile_secondary_motion_profile",
    "discover_driver_chains",
    "runtime_capability_descriptor",
]
