"""Headless Map Studio contract for animated sky traffic.

Sky traffic is editor-authored like an Unreal actor, but it is not a KOTOR GIT
placeable.  The future compiler must turn these records into room-local
MDL/MDX animation nodes under one of Odyssey's automatically started
``animloop1``/``animloop2``/``animloop3`` clips.

This module owns only stable KMAP data, validation, and deterministic linear
sampling for viewport paths/arrows.  It does not claim an engine writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .authored_module_project import AuthoredModuleProject


Vec3 = tuple[float, float, float]

SKY_TRAFFIC_SCHEMA = "ghostrigger.map_studio.sky_traffic.v1"
SKY_TRAFFIC_COMPILER_TARGET = "room_mdl_animation"
SKY_TRAFFIC_RUNTIME_CONTAINER = "room_mdl_mdx"
SKY_TRAFFIC_PROJECT_EXTRA_KEY = "sky_traffic"
SKY_TRAFFIC_LOOP_SLOTS = ("animloop1", "animloop2", "animloop3")
SKY_TRAFFIC_FACING_MODES = ("path_tangent", "fixed", "preserve_model")
SKY_TRAFFIC_INTERPOLATIONS = ("linear",)

_RESREF_RE = re.compile(r"^[a-z0-9_]+$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class SkyTrafficControlPoint:
    """One stable, room-local path point."""

    point_id: str
    position: Vec3
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthoredSkyTraffic:
    """One editor actor that will eventually compile into a room animation."""

    traffic_id: str
    room_resref: str
    model_resref: str
    control_points: tuple[SkyTrafficControlPoint, ...]
    name: str = "Sky Traffic"
    animation_name: str = "animloop1"
    model_animation_name: str = ""
    loop: bool = True
    closed_path: bool = False
    duration_seconds: float | None = 30.0
    speed_units_per_second: float | None = None
    facing_mode: str = "path_tangent"
    fixed_facing_degrees: float = 0.0
    position_offset: Vec3 = (0.0, 0.0, 0.0)
    altitude_offset: float = 0.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0
    hidden_reset_seconds: float = 0.0
    interpolation: str = "linear"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compiler_target(self) -> str:
        return SKY_TRAFFIC_COMPILER_TARGET

    @property
    def is_git_placement(self) -> bool:
        return False


@dataclass(frozen=True)
class SkyTrafficValidationIssue:
    severity: str
    code: str
    message: str
    traffic_id: str = ""
    field_name: str = ""


@dataclass(frozen=True)
class SkyTrafficValidation:
    ok: bool
    issues: tuple[SkyTrafficValidationIssue, ...] = ()
    warnings: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkyTrafficSample:
    traffic_id: str
    time_seconds: float
    normalized_time: float
    distance: float
    segment_index: int
    position: Vec3
    travel_direction: Vec3
    facing_direction: Vec3 | None
    visibility_alpha: float
    hidden_for_reset: bool


@dataclass(frozen=True)
class SkyTrafficPreviewArrow:
    position: Vec3
    travel_direction: Vec3
    facing_direction: Vec3 | None
    normalized_distance: float


@dataclass(frozen=True)
class SkyTrafficPreview:
    traffic_id: str
    enabled: bool
    compiler_target: str
    path_points: tuple[Vec3, ...]
    arrows: tuple[SkyTrafficPreviewArrow, ...]
    path_length: float
    duration_seconds: float


def _vec3(value: Any, *, default: Vec3 = (math.nan, math.nan, math.nan)) -> Vec3:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"), value.get("z"))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return default
    return default


def _finite_vec3(value: Vec3) -> bool:
    return len(value) == 3 and all(math.isfinite(float(channel)) for channel in value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _normalise_resref(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalise_loop_slot(value: Any) -> str:
    text = str(value or "animloop1").strip().lower().replace(" ", "")
    return {"1": "animloop1", "2": "animloop2", "3": "animloop3"}.get(text, text)


def _normalise_facing_mode(value: Any) -> str:
    text = str(value or "path_tangent").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tangent": "path_tangent",
        "path": "path_tangent",
        "travel": "path_tangent",
        "yaw": "fixed",
        "model": "preserve_model",
        "none": "preserve_model",
    }
    return aliases.get(text, text)


def new_sky_traffic_id() -> str:
    """Return a stable ID to serialize once into KMAP."""

    return f"skytraffic_{uuid.uuid4().hex}"


def _legacy_sky_traffic_id(source: dict[str, Any]) -> str:
    """Derive a deterministic ID for pre-ID drafts instead of changing per load."""

    seed = {
        "room": source.get("room_resref") or source.get("room"),
        "model": source.get("model_resref") or source.get("model"),
        "path": source.get("path") or source.get("path_points") or source.get("control_points"),
        "name": source.get("name"),
    }
    encoded = json.dumps(seed, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"skytraffic_legacy_{hashlib.sha1(encoded).hexdigest()[:16]}"


def _control_point_from_value(value: Any, index: int) -> SkyTrafficControlPoint:
    if isinstance(value, SkyTrafficControlPoint):
        return value
    if isinstance(value, dict):
        position = _vec3(value.get("position", value.get("point")))
        point_id = str(value.get("id") or value.get("point_id") or f"point_{index + 1:02d}").strip()
        metadata = dict(value.get("metadata")) if isinstance(value.get("metadata"), dict) else {}
    else:
        position = _vec3(value)
        point_id = f"point_{index + 1:02d}"
        metadata = {}
    return SkyTrafficControlPoint(point_id=point_id, position=position, metadata=metadata)


def _control_point_payload(point: SkyTrafficControlPoint) -> dict[str, Any]:
    return {
        "id": point.point_id,
        "position": [float(point.position[0]), float(point.position[1]), float(point.position[2])],
        "metadata": dict(point.metadata),
    }


def normalise_authored_sky_traffic(data: Any) -> AuthoredSkyTraffic:
    """Parse one KMAP-compatible dictionary into the stable headless contract."""

    source = dict(data) if isinstance(data, dict) else {}
    path = dict(source.get("path")) if isinstance(source.get("path"), dict) else {}
    timing = dict(source.get("timing")) if isinstance(source.get("timing"), dict) else {}
    facing = dict(source.get("facing")) if isinstance(source.get("facing"), dict) else {}
    visibility = dict(source.get("visibility")) if isinstance(source.get("visibility"), dict) else {}
    points_value = path.get("control_points", source.get("control_points", source.get("path_points", ())))
    points = tuple(_control_point_from_value(value, index) for index, value in enumerate(tuple(points_value or ())))

    has_duration = "duration_seconds" in timing or "duration_seconds" in source
    has_speed = "speed_units_per_second" in timing or "speed_units_per_second" in source or "speed" in source
    duration_value = timing.get("duration_seconds", source.get("duration_seconds")) if has_duration else None
    speed_value = timing.get("speed_units_per_second", source.get("speed_units_per_second", source.get("speed"))) if has_speed else None
    if not has_duration and not has_speed:
        duration_value = 30.0

    known_fields = {
        "schema",
        "compiler_target",
        "runtime_container",
        "git_placement",
        "id",
        "traffic_id",
        "name",
        "room",
        "room_resref",
        "model",
        "model_resref",
        "path",
        "control_points",
        "path_points",
        "timing",
        "duration_seconds",
        "speed_units_per_second",
        "speed",
        "loop",
        "closed_path",
        "facing",
        "visibility",
        "fade_in_seconds",
        "fade_out_seconds",
        "hidden_reset_seconds",
        "facing_mode",
        "fixed_facing_degrees",
        "offset",
        "position_offset",
        "altitude",
        "altitude_offset",
        "animation_name",
        "model_animation_name",
        "interpolation",
        "enabled",
        "metadata",
    }
    metadata = dict(source.get("metadata")) if isinstance(source.get("metadata"), dict) else {}
    unknown_fields = {key: value for key, value in source.items() if key not in known_fields}
    if unknown_fields:
        metadata["_unknown_kmap_fields"] = unknown_fields

    offset = source.get("position_offset", source.get("offset", (0.0, 0.0, 0.0)))
    fixed_facing = _optional_float(facing.get("fixed_degrees", source.get("fixed_facing_degrees", 0.0)))
    altitude = _optional_float(source.get("altitude_offset", source.get("altitude", 0.0)))
    fade_in = _optional_float(visibility.get("fade_in_seconds", source.get("fade_in_seconds", 0.0)))
    fade_out = _optional_float(visibility.get("fade_out_seconds", source.get("fade_out_seconds", 0.0)))
    hidden_reset = _optional_float(
        visibility.get("hidden_reset_seconds", source.get("hidden_reset_seconds", 0.0))
    )
    return AuthoredSkyTraffic(
        traffic_id=str(source.get("id") or source.get("traffic_id") or _legacy_sky_traffic_id(source)).strip(),
        name=str(source.get("name") or "Sky Traffic").strip()[:64],
        room_resref=_normalise_resref(source.get("room_resref", source.get("room"))),
        model_resref=_normalise_resref(source.get("model_resref", source.get("model"))),
        control_points=points,
        animation_name=_normalise_loop_slot(source.get("animation_name", "animloop1")),
        model_animation_name=str(source.get("model_animation_name") or "").strip(),
        loop=_bool(timing.get("loop", source.get("loop")), True),
        closed_path=_bool(path.get("closed", source.get("closed_path")), False),
        duration_seconds=_optional_float(duration_value),
        speed_units_per_second=_optional_float(speed_value),
        facing_mode=_normalise_facing_mode(facing.get("mode", source.get("facing_mode"))),
        fixed_facing_degrees=0.0 if fixed_facing is None else fixed_facing,
        position_offset=_vec3(offset, default=(0.0, 0.0, 0.0)),
        altitude_offset=0.0 if altitude is None else altitude,
        fade_in_seconds=0.0 if fade_in is None else fade_in,
        fade_out_seconds=0.0 if fade_out is None else fade_out,
        hidden_reset_seconds=0.0 if hidden_reset is None else hidden_reset,
        interpolation=str(path.get("interpolation", source.get("interpolation", "linear")) or "linear").strip().lower(),
        enabled=_bool(source.get("enabled"), True),
        metadata=metadata,
    )


def authored_sky_traffic_to_kmap(traffic: AuthoredSkyTraffic) -> dict[str, Any]:
    """Serialize one actor without pretending it belongs in GIT."""

    metadata = dict(traffic.metadata)
    unknown_fields = dict(metadata.pop("_unknown_kmap_fields", {}) or {})
    payload = dict(unknown_fields)
    payload.update(
        {
            "schema": SKY_TRAFFIC_SCHEMA,
            "compiler_target": SKY_TRAFFIC_COMPILER_TARGET,
            "runtime_container": SKY_TRAFFIC_RUNTIME_CONTAINER,
            "git_placement": False,
            "id": traffic.traffic_id,
            "name": traffic.name,
            "room_resref": traffic.room_resref,
            "model_resref": traffic.model_resref,
            "path": {
                "interpolation": traffic.interpolation,
                "closed": bool(traffic.closed_path),
                "control_points": [_control_point_payload(point) for point in traffic.control_points],
            },
            "timing": {
                "loop": bool(traffic.loop),
                "duration_seconds": traffic.duration_seconds,
                "speed_units_per_second": traffic.speed_units_per_second,
            },
            "facing": {
                "mode": traffic.facing_mode,
                "fixed_degrees": float(traffic.fixed_facing_degrees),
            },
            "visibility": {
                "fade_in_seconds": float(traffic.fade_in_seconds),
                "fade_out_seconds": float(traffic.fade_out_seconds),
                "hidden_reset_seconds": float(traffic.hidden_reset_seconds),
            },
            "position_offset": [float(value) for value in traffic.position_offset],
            "altitude_offset": float(traffic.altitude_offset),
            "animation_name": traffic.animation_name,
            "model_animation_name": traffic.model_animation_name,
            "enabled": bool(traffic.enabled),
            "metadata": metadata,
        }
    )
    return payload


def authored_sky_traffic_from_kmap(data: Any) -> AuthoredSkyTraffic:
    return normalise_authored_sky_traffic(data)


def authored_sky_traffic_list_to_kmap(items: Iterable[AuthoredSkyTraffic]) -> list[dict[str, Any]]:
    return [authored_sky_traffic_to_kmap(item) for item in tuple(items or ())]


def authored_sky_traffic_list_from_kmap(rows: Iterable[Any]) -> tuple[AuthoredSkyTraffic, ...]:
    return tuple(normalise_authored_sky_traffic(row) for row in tuple(rows or ()))


def read_authored_project_sky_traffic(project: "AuthoredModuleProject") -> tuple[AuthoredSkyTraffic, ...]:
    """Read normalized actors from ``AuthoredModuleProject.extra``."""

    extra = dict(getattr(project, "extra", {}) or {})
    stored = extra.get(SKY_TRAFFIC_PROJECT_EXTRA_KEY, ())
    if isinstance(stored, dict):
        stored = stored.get("items", (stored,))
    return authored_sky_traffic_list_from_kmap(tuple(stored or ()))


def write_authored_project_sky_traffic(
    project: "AuthoredModuleProject",
    items: Iterable[AuthoredSkyTraffic | dict[str, Any]],
    *,
    validate: bool = True,
) -> "AuthoredModuleProject":
    """Return a replaced project with normalized KMAP rows in ``extra``."""

    normalized = tuple(
        item if isinstance(item, AuthoredSkyTraffic) else normalise_authored_sky_traffic(item)
        for item in tuple(items or ())
    )
    if validate:
        room_resrefs = {
            _normalise_resref(
                room.normalised_resref() if callable(getattr(room, "normalised_resref", None)) else getattr(room, "room_resref", "")
            )
            for room in tuple(getattr(project, "rooms", ()) or ())
        }
        validation = validate_authored_sky_traffic_collection(normalized, room_resrefs=room_resrefs)
        if not validation.ok:
            raise ValueError("; ".join(validation.blocking_issues))
    extra = dict(getattr(project, "extra", {}) or {})
    extra[SKY_TRAFFIC_PROJECT_EXTRA_KEY] = authored_sky_traffic_list_to_kmap(normalized)
    return replace(project, extra=extra)


def create_authored_sky_traffic(
    *,
    room_resref: Any,
    model_resref: Any,
    control_points: Iterable[Any],
    traffic_id: str = "",
    name: str = "Sky Traffic",
    animation_name: str = "animloop1",
    model_animation_name: str = "",
    loop: bool = True,
    closed_path: bool = False,
    duration_seconds: float | None = None,
    speed_units_per_second: float | None = None,
    facing_mode: str = "path_tangent",
    fixed_facing_degrees: float = 0.0,
    position_offset: Any = (0.0, 0.0, 0.0),
    altitude_offset: float = 0.0,
    fade_in_seconds: float = 0.0,
    fade_out_seconds: float = 0.0,
    hidden_reset_seconds: float = 0.0,
    enabled: bool = True,
    metadata: dict[str, Any] | None = None,
) -> AuthoredSkyTraffic:
    """Create and validate one new actor with a UUID-backed stable ID."""

    if duration_seconds is None and speed_units_per_second is None:
        duration_seconds = 30.0
    traffic = normalise_authored_sky_traffic(
        {
            "id": traffic_id or new_sky_traffic_id(),
            "name": name,
            "room_resref": room_resref,
            "model_resref": model_resref,
            "control_points": tuple(control_points or ()),
            "animation_name": animation_name,
            "model_animation_name": model_animation_name,
            "loop": loop,
            "closed_path": closed_path,
            "duration_seconds": duration_seconds,
            "speed_units_per_second": speed_units_per_second,
            "facing_mode": facing_mode,
            "fixed_facing_degrees": fixed_facing_degrees,
            "position_offset": position_offset,
            "altitude_offset": altitude_offset,
            "fade_in_seconds": fade_in_seconds,
            "fade_out_seconds": fade_out_seconds,
            "hidden_reset_seconds": hidden_reset_seconds,
            "enabled": enabled,
            "metadata": {"source": "map_studio:sky_traffic", **dict(metadata or {})},
        }
    )
    validation = validate_authored_sky_traffic(traffic)
    if not validation.ok:
        raise ValueError("; ".join(validation.blocking_issues))
    return traffic


def _issue(
    issues: list[SkyTrafficValidationIssue],
    traffic: AuthoredSkyTraffic,
    code: str,
    message: str,
    field_name: str,
    *,
    warning: bool = False,
) -> None:
    issues.append(
        SkyTrafficValidationIssue(
            severity="warning" if warning else "error",
            code=code,
            message=message,
            traffic_id=traffic.traffic_id,
            field_name=field_name,
        )
    )


def _validation(issues: Iterable[SkyTrafficValidationIssue]) -> SkyTrafficValidation:
    rows = tuple(issues)
    warnings = tuple(issue.message for issue in rows if issue.severity == "warning")
    blocking = tuple(issue.message for issue in rows if issue.severity != "warning")
    return SkyTrafficValidation(ok=not blocking, issues=rows, warnings=warnings, blocking_issues=blocking)


def validate_authored_sky_traffic(
    traffic: AuthoredSkyTraffic,
    *,
    room_resrefs: set[str] | None = None,
) -> SkyTrafficValidation:
    """Validate editor data against the future room-animation compiler boundary."""

    issues: list[SkyTrafficValidationIssue] = []
    label = traffic.name or traffic.traffic_id or "Sky Traffic"
    if not traffic.traffic_id or len(traffic.traffic_id) > 80 or not _ID_RE.fullmatch(traffic.traffic_id):
        _issue(issues, traffic, "SKY_TRAFFIC_ID_INVALID", f"{label} requires a stable ID using letters, numbers, '.', ':', '_' or '-'.", "traffic_id")
    for field_name, value in (("room_resref", traffic.room_resref), ("model_resref", traffic.model_resref)):
        if not value or len(value) > 16 or not _RESREF_RE.fullmatch(value):
            _issue(issues, traffic, "SKY_TRAFFIC_RESREF_INVALID", f"{label} {field_name} must be a valid KOTOR resref of 16 characters or fewer.", field_name)
    if room_resrefs is not None and traffic.room_resref not in {_normalise_resref(value) for value in room_resrefs}:
        _issue(issues, traffic, "SKY_TRAFFIC_ROOM_MISSING", f"{label} targets missing room {traffic.room_resref or '(missing)'}.", "room_resref")
    if traffic.animation_name not in SKY_TRAFFIC_LOOP_SLOTS:
        _issue(issues, traffic, "SKY_TRAFFIC_LOOP_SLOT_INVALID", f"{label} animation_name must be animloop1, animloop2, or animloop3 for automatic room playback.", "animation_name")
    if traffic.model_animation_name and len(traffic.model_animation_name) > 32:
        _issue(issues, traffic, "SKY_TRAFFIC_MODEL_ANIMATION_INVALID", f"{label} model animation name must be 32 characters or fewer.", "model_animation_name")
    if not traffic.loop:
        _issue(issues, traffic, "SKY_TRAFFIC_NON_LOOP_UNSUPPORTED", f"{label} is one-shot, but this contract only targets automatically looping room MDL animations.", "loop")
    if traffic.interpolation not in SKY_TRAFFIC_INTERPOLATIONS:
        _issue(issues, traffic, "SKY_TRAFFIC_INTERPOLATION_UNSUPPORTED", f"{label} interpolation {traffic.interpolation!r} is not supported; use linear.", "interpolation")
    if traffic.facing_mode not in SKY_TRAFFIC_FACING_MODES:
        _issue(issues, traffic, "SKY_TRAFFIC_FACING_UNSUPPORTED", f"{label} facing mode {traffic.facing_mode!r} is unsupported.", "facing_mode")
    if not math.isfinite(float(traffic.fixed_facing_degrees)):
        _issue(issues, traffic, "SKY_TRAFFIC_FACING_INVALID", f"{label} fixed facing angle must be finite.", "fixed_facing_degrees")
    if not _finite_vec3(traffic.position_offset) or not math.isfinite(float(traffic.altitude_offset)):
        _issue(issues, traffic, "SKY_TRAFFIC_OFFSET_INVALID", f"{label} position and altitude offsets must be finite.", "position_offset")
    visibility_values = (
        ("fade_in_seconds", traffic.fade_in_seconds),
        ("fade_out_seconds", traffic.fade_out_seconds),
        ("hidden_reset_seconds", traffic.hidden_reset_seconds),
    )
    for field_name, value in visibility_values:
        if not math.isfinite(float(value)) or float(value) < 0.0:
            _issue(
                issues,
                traffic,
                "SKY_TRAFFIC_VISIBILITY_TIMING_INVALID",
                f"{label} {field_name} must be a non-negative finite number.",
                field_name,
            )

    point_ids: set[str] = set()
    if len(traffic.control_points) < 2:
        _issue(issues, traffic, "SKY_TRAFFIC_PATH_TOO_SHORT", f"{label} requires at least two path control points.", "control_points")
    for point in traffic.control_points:
        if not point.point_id or point.point_id in point_ids:
            _issue(issues, traffic, "SKY_TRAFFIC_POINT_ID_INVALID", f"{label} path control-point IDs must be non-empty and unique.", "control_points")
        point_ids.add(point.point_id)
        if not _finite_vec3(point.position):
            _issue(issues, traffic, "SKY_TRAFFIC_POINT_INVALID", f"{label} path point {point.point_id or '(missing)'} must contain finite XYZ coordinates.", "control_points")
    if len(traffic.control_points) >= 2 and sky_traffic_path_length(traffic) <= 1.0e-8:
        _issue(issues, traffic, "SKY_TRAFFIC_PATH_ZERO_LENGTH", f"{label} path must cover a non-zero distance.", "control_points")

    duration = traffic.duration_seconds
    speed = traffic.speed_units_per_second
    duration_ok = duration is not None and math.isfinite(float(duration)) and float(duration) > 0.0
    speed_ok = speed is not None and math.isfinite(float(speed)) and float(speed) > 0.0
    if duration is not None and not duration_ok:
        _issue(issues, traffic, "SKY_TRAFFIC_DURATION_INVALID", f"{label} duration must be a positive finite number.", "duration_seconds")
    if speed is not None and not speed_ok:
        _issue(issues, traffic, "SKY_TRAFFIC_SPEED_INVALID", f"{label} speed must be a positive finite number.", "speed_units_per_second")
    if duration_ok and speed_ok:
        _issue(issues, traffic, "SKY_TRAFFIC_TIMING_CONFLICT", f"{label} must use duration or speed, not both.", "timing")
    if not duration_ok and not speed_ok:
        _issue(issues, traffic, "SKY_TRAFFIC_TIMING_MISSING", f"{label} requires either a positive duration or speed.", "timing")
    effective_duration = sky_traffic_effective_duration(traffic)
    if (
        effective_duration > 0.0
        and traffic.fade_in_seconds + traffic.fade_out_seconds > effective_duration
    ):
        _issue(
            issues,
            traffic,
            "SKY_TRAFFIC_VISIBILITY_OVERLAP",
            f"{label} fade-in and fade-out windows exceed its motion duration.",
            "visibility",
        )
    if traffic.loop and not traffic.closed_path and len(traffic.control_points) >= 2:
        first = traffic.control_points[0].position
        last = traffic.control_points[-1].position
        reset_is_hidden = traffic.fade_out_seconds > 0.0 and traffic.hidden_reset_seconds > 0.0
        if (
            _finite_vec3(first)
            and _finite_vec3(last)
            and _length(_sub(last, first)) > 1.0e-5
            and not reset_is_hidden
        ):
            _issue(
                issues,
                traffic,
                "SKY_TRAFFIC_LOOP_RESET",
                f"{label} uses an open path; the room loop resets from its last point to its first at the animation boundary.",
                "closed_path",
                warning=True,
            )
    return _validation(issues)


def validate_authored_sky_traffic_collection(
    items: Iterable[AuthoredSkyTraffic],
    *,
    room_resrefs: set[str] | None = None,
) -> SkyTrafficValidation:
    """Validate IDs and shared room animation-slot periods across all actors."""

    traffic = tuple(items or ())
    issues: list[SkyTrafficValidationIssue] = []
    seen_ids: set[str] = set()
    periods: dict[tuple[str, str], tuple[float, AuthoredSkyTraffic]] = {}
    for item in traffic:
        issues.extend(validate_authored_sky_traffic(item, room_resrefs=room_resrefs).issues)
        if item.traffic_id in seen_ids:
            _issue(issues, item, "SKY_TRAFFIC_ID_DUPLICATE", f"Duplicate sky-traffic ID: {item.traffic_id}.", "traffic_id")
        seen_ids.add(item.traffic_id)
        if not item.enabled:
            continue
        period = sky_traffic_effective_period(item)
        key = (item.room_resref, item.animation_name)
        previous = periods.get(key)
        if previous is None:
            periods[key] = (period, item)
        elif period > 0.0 and previous[0] > 0.0 and not math.isclose(period, previous[0], rel_tol=1.0e-6, abs_tol=1.0e-6):
            _issue(
                issues,
                item,
                "SKY_TRAFFIC_LOOP_SLOT_PERIOD_CONFLICT",
                f"Sky traffic in room {item.room_resref} shares {item.animation_name} but requests incompatible periods {previous[0]:.6g}s and {period:.6g}s.",
                "animation_name",
            )
    return _validation(issues)


def _add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _mul(value: Vec3, scalar: float) -> Vec3:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def _length(value: Vec3) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _normalise(value: Vec3, fallback: Vec3 = (1.0, 0.0, 0.0)) -> Vec3:
    length = _length(value)
    return _mul(value, 1.0 / length) if length > 1.0e-12 else fallback


def _offset_position(traffic: AuthoredSkyTraffic, position: Vec3) -> Vec3:
    return (
        position[0] + traffic.position_offset[0],
        position[1] + traffic.position_offset[1],
        position[2] + traffic.position_offset[2] + traffic.altitude_offset,
    )


def _segments(traffic: AuthoredSkyTraffic) -> tuple[tuple[int, Vec3, Vec3, float], ...]:
    points = [point.position for point in traffic.control_points if _finite_vec3(point.position)]
    if traffic.closed_path and len(points) >= 2 and _length(_sub(points[-1], points[0])) > 1.0e-9:
        points.append(points[0])
    rows: list[tuple[int, Vec3, Vec3, float]] = []
    for index in range(max(0, len(points) - 1)):
        start, end = points[index], points[index + 1]
        length = _length(_sub(end, start))
        if length > 1.0e-12:
            rows.append((index, start, end, length))
    return tuple(rows)


def sky_traffic_path_length(traffic: AuthoredSkyTraffic) -> float:
    return sum(segment[3] for segment in _segments(traffic))


def sky_traffic_effective_duration(traffic: AuthoredSkyTraffic) -> float:
    if traffic.duration_seconds is not None and math.isfinite(float(traffic.duration_seconds)) and float(traffic.duration_seconds) > 0.0:
        return float(traffic.duration_seconds)
    if traffic.speed_units_per_second is not None and math.isfinite(float(traffic.speed_units_per_second)) and float(traffic.speed_units_per_second) > 0.0:
        return sky_traffic_path_length(traffic) / float(traffic.speed_units_per_second)
    return 0.0


def sky_traffic_effective_period(traffic: AuthoredSkyTraffic) -> float:
    """Return motion plus the hidden reset interval for one complete loop."""

    duration = sky_traffic_effective_duration(traffic)
    hidden = max(0.0, float(traffic.hidden_reset_seconds)) if math.isfinite(float(traffic.hidden_reset_seconds)) else 0.0
    return duration + hidden


def _visibility_alpha(traffic: AuthoredSkyTraffic, motion_time: float, duration: float) -> float:
    if duration <= 0.0:
        return 0.0
    fade_in = max(0.0, float(traffic.fade_in_seconds))
    fade_out = max(0.0, float(traffic.fade_out_seconds))
    alpha = 1.0
    if fade_in > 0.0:
        alpha = min(alpha, max(0.0, min(1.0, motion_time / fade_in)))
    if fade_out > 0.0:
        alpha = min(alpha, max(0.0, min(1.0, (duration - motion_time) / fade_out)))
    return alpha


def _sample_distance(traffic: AuthoredSkyTraffic, distance: float) -> tuple[Vec3, Vec3, int]:
    segments = _segments(traffic)
    if not segments:
        position = traffic.control_points[0].position if traffic.control_points and _finite_vec3(traffic.control_points[0].position) else (0.0, 0.0, 0.0)
        return _offset_position(traffic, position), (1.0, 0.0, 0.0), 0
    total = sum(segment[3] for segment in segments)
    remaining = max(0.0, min(float(distance), total))
    for segment_index, start, end, length in segments:
        if remaining <= length or (segment_index, start, end, length) == segments[-1]:
            fraction = max(0.0, min(1.0, remaining / length))
            position = _add(start, _mul(_sub(end, start), fraction))
            return _offset_position(traffic, position), _normalise(_sub(end, start)), segment_index
        remaining -= length
    last = segments[-1]
    return _offset_position(traffic, last[2]), _normalise(_sub(last[2], last[1])), last[0]


def _facing_direction(traffic: AuthoredSkyTraffic, travel_direction: Vec3) -> Vec3 | None:
    if traffic.facing_mode == "preserve_model":
        return None
    if traffic.facing_mode == "fixed":
        radians = math.radians(float(traffic.fixed_facing_degrees))
        return (math.cos(radians), math.sin(radians), 0.0)
    return travel_direction


def sample_sky_traffic(traffic: AuthoredSkyTraffic, time_seconds: float) -> SkyTrafficSample:
    """Sample a deterministic room-local transform without Qt or renderer state."""

    duration = sky_traffic_effective_duration(traffic)
    period = sky_traffic_effective_period(traffic)
    path_length = sky_traffic_path_length(traffic)
    requested = float(time_seconds)
    if duration <= 0.0:
        effective_time = 0.0
        normalized_time = 0.0
    elif traffic.loop:
        cycle_time = requested % period if period > 0.0 else 0.0
        hidden_for_reset = cycle_time >= duration
        effective_time = min(cycle_time, duration)
        normalized_time = effective_time / duration
    else:
        hidden_for_reset = False
        effective_time = max(0.0, min(requested, duration))
        normalized_time = effective_time / duration
    if duration <= 0.0:
        hidden_for_reset = True
    distance = path_length * normalized_time
    position, travel_direction, segment_index = _sample_distance(traffic, distance)
    return SkyTrafficSample(
        traffic_id=traffic.traffic_id,
        time_seconds=effective_time,
        normalized_time=normalized_time,
        distance=distance,
        segment_index=segment_index,
        position=position,
        travel_direction=travel_direction,
        facing_direction=_facing_direction(traffic, travel_direction),
        visibility_alpha=(
            0.0
            if hidden_for_reset
            else _visibility_alpha(traffic, effective_time, duration)
        ),
        hidden_for_reset=hidden_for_reset,
    )


def build_sky_traffic_preview(
    traffic: AuthoredSkyTraffic,
    *,
    path_sample_count: int = 33,
    arrow_count: int = 8,
) -> SkyTrafficPreview:
    """Return deterministic polyline and arrow data for any viewport backend."""

    sample_count = max(2, int(path_sample_count))
    arrows_wanted = max(0, int(arrow_count))
    path_length = sky_traffic_path_length(traffic)
    points = tuple(
        _sample_distance(traffic, path_length * (index / (sample_count - 1)))[0]
        for index in range(sample_count)
    )
    arrows: list[SkyTrafficPreviewArrow] = []
    for index in range(arrows_wanted):
        fraction = (index + 0.5) / arrows_wanted
        position, travel_direction, _segment_index = _sample_distance(traffic, path_length * fraction)
        arrows.append(
            SkyTrafficPreviewArrow(
                position=position,
                travel_direction=travel_direction,
                facing_direction=_facing_direction(traffic, travel_direction),
                normalized_distance=fraction,
            )
        )
    return SkyTrafficPreview(
        traffic_id=traffic.traffic_id,
        enabled=bool(traffic.enabled),
        compiler_target=SKY_TRAFFIC_COMPILER_TARGET,
        path_points=points,
        arrows=tuple(arrows),
        path_length=path_length,
        duration_seconds=sky_traffic_effective_duration(traffic),
    )


authored_sky_traffic_payload = authored_sky_traffic_to_kmap


__all__ = [
    "AuthoredSkyTraffic",
    "SKY_TRAFFIC_COMPILER_TARGET",
    "SKY_TRAFFIC_FACING_MODES",
    "SKY_TRAFFIC_INTERPOLATIONS",
    "SKY_TRAFFIC_LOOP_SLOTS",
    "SKY_TRAFFIC_PROJECT_EXTRA_KEY",
    "SKY_TRAFFIC_RUNTIME_CONTAINER",
    "SKY_TRAFFIC_SCHEMA",
    "SkyTrafficControlPoint",
    "SkyTrafficPreview",
    "SkyTrafficPreviewArrow",
    "SkyTrafficSample",
    "SkyTrafficValidation",
    "SkyTrafficValidationIssue",
    "authored_sky_traffic_from_kmap",
    "authored_sky_traffic_list_from_kmap",
    "authored_sky_traffic_list_to_kmap",
    "authored_sky_traffic_payload",
    "authored_sky_traffic_to_kmap",
    "build_sky_traffic_preview",
    "create_authored_sky_traffic",
    "new_sky_traffic_id",
    "normalise_authored_sky_traffic",
    "read_authored_project_sky_traffic",
    "sample_sky_traffic",
    "sky_traffic_effective_duration",
    "sky_traffic_effective_period",
    "sky_traffic_path_length",
    "validate_authored_sky_traffic",
    "validate_authored_sky_traffic_collection",
    "write_authored_project_sky_traffic",
]
