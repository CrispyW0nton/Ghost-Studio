"""Geometry-backed facial range-of-motion and LIP timeline validation.

Controller changes and changed pixels are not sufficient proof that a face
works.  This module skins the actual mouth vertices through all 16 KOTOR talk
poses and measures visible deformation, aperture, closure, and shape coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


UPPER_LIP_BONES = frozenset({"f_um_g"})
LOWER_LIP_BONES = frozenset({"f_rlm_g", "f_llm_g"})
MOUTH_BONES = frozenset(
    {
        "f_jaw_g",
        "f_rlm_g",
        "f_llm_g",
        "f_um_g",
        "f_lmc_g",
        "f_rmc_g",
    }
)


@dataclass(frozen=True)
class FacialRangeSample:
    shape_index: int
    mean_mouth_displacement: float
    max_mouth_displacement: float
    aperture: float
    aperture_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_index": self.shape_index,
            "mean_mouth_displacement": self.mean_mouth_displacement,
            "max_mouth_displacement": self.max_mouth_displacement,
            "aperture": self.aperture,
            "aperture_delta": self.aperture_delta,
        }


@dataclass
class FacialRangeReport:
    model_name: str
    ok: bool = True
    samples: dict[int, FacialRangeSample] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    upper_vertex_count: int = 0
    lower_vertex_count: int = 0
    mouth_vertex_count: int = 0
    active_shape_count: int = 0
    mouth_scale: float = 0.0
    movement_floor: float = 0.0
    aperture_floor: float = 0.0
    talk_animation: str = ""
    talk_animation_owner: str = ""
    shape_times: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ok": self.ok,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "upper_vertex_count": self.upper_vertex_count,
            "lower_vertex_count": self.lower_vertex_count,
            "mouth_vertex_count": self.mouth_vertex_count,
            "active_shape_count": self.active_shape_count,
            "mouth_scale": self.mouth_scale,
            "movement_floor": self.movement_floor,
            "aperture_floor": self.aperture_floor,
            "talk_animation": self.talk_animation,
            "talk_animation_owner": self.talk_animation_owner,
            "shape_times": list(self.shape_times),
            "samples": {
                str(index): sample.to_dict()
                for index, sample in sorted(self.samples.items())
            },
        }


@dataclass
class LipTimelineReport:
    name: str
    duration: float
    keyframe_count: int
    ok: bool = True
    unique_shape_count: int = 0
    shape_counts: dict[int, int] = field(default_factory=dict)
    shape_durations: dict[int, float] = field(default_factory=dict)
    neutral_duration: float = 0.0
    active_duration_fraction: float = 0.0
    max_keyframe_gap: float = 0.0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.duration,
            "keyframe_count": self.keyframe_count,
            "ok": self.ok,
            "unique_shape_count": self.unique_shape_count,
            "shape_counts": {
                str(index): value
                for index, value in sorted(self.shape_counts.items())
            },
            "shape_durations": {
                str(index): value
                for index, value in sorted(self.shape_durations.items())
            },
            "neutral_duration": self.neutral_duration,
            "active_duration_fraction": self.active_duration_fraction,
            "max_keyframe_gap": self.max_keyframe_gap,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AudioLipSyncReport:
    lip_duration: float
    audio_duration: float
    duration_delta: float
    tolerance: float
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "lip_duration": self.lip_duration,
            "audio_duration": self.audio_duration,
            "duration_delta": self.duration_delta,
            "tolerance": self.tolerance,
        }


def audit_audio_lip_sync(
    *,
    lip_duration: float,
    audio_duration: float,
) -> AudioLipSyncReport:
    """Reject voice media whose playable duration cannot own the LIP clock."""

    lip_seconds = max(0.0, float(lip_duration))
    audio_seconds = max(0.0, float(audio_duration))
    tolerance = max(0.05, lip_seconds * 0.02)
    delta = audio_seconds - lip_seconds
    return AudioLipSyncReport(
        lip_duration=lip_seconds,
        audio_duration=audio_seconds,
        duration_delta=delta,
        tolerance=tolerance,
        ok=bool(lip_seconds > 0.0 and abs(delta) <= tolerance),
    )


def _weighted_centroid(
    positions: Any,
    weights: Any,
    mask: Any,
) -> Any:
    import numpy as np

    selected = np.asarray(weights[mask], dtype=float)
    total = float(selected.sum())
    if total <= 1.0e-12:
        return np.asarray(positions[mask], dtype=float).mean(axis=0)
    return np.average(
        np.asarray(positions[mask], dtype=float),
        axis=0,
        weights=selected,
    )


def score_facial_range(
    *,
    model_name: str,
    shape_positions: Mapping[int, Any],
    upper_weights: Sequence[float],
    lower_weights: Sequence[float],
    mouth_weights: Sequence[float],
    weight_threshold: float = 0.15,
) -> FacialRangeReport:
    """Score already-skinned vertex positions for all KOTOR talk shapes."""

    import numpy as np

    report = FacialRangeReport(model_name=str(model_name or "head"))
    normalized = {
        int(index): np.asarray(positions, dtype=float)
        for index, positions in shape_positions.items()
    }
    if 0 not in normalized:
        report.ok = False
        report.failures.append("neutral_shape_missing")
        return report
    neutral = normalized[0]
    if neutral.ndim != 2 or neutral.shape[1] < 3 or len(neutral) == 0:
        report.ok = False
        report.failures.append("invalid_vertex_positions")
        return report
    upper = np.asarray(upper_weights, dtype=float)
    lower = np.asarray(lower_weights, dtype=float)
    mouth = np.asarray(mouth_weights, dtype=float)
    if any(len(values) != len(neutral) for values in (upper, lower, mouth)):
        report.ok = False
        report.failures.append("weight_vertex_count_mismatch")
        return report
    upper_mask = upper >= max(0.45, float(weight_threshold))
    lower_mask = lower >= max(0.45, float(weight_threshold))
    mouth_mask = mouth >= float(weight_threshold)
    report.upper_vertex_count = int(upper_mask.sum())
    report.lower_vertex_count = int(lower_mask.sum())
    report.mouth_vertex_count = int(mouth_mask.sum())
    if not report.upper_vertex_count:
        report.failures.append("upper_lip_vertices_missing")
    if not report.lower_vertex_count:
        report.failures.append("lower_lip_vertices_missing")
    if not report.mouth_vertex_count:
        report.failures.append("mouth_vertices_missing")
    if report.failures:
        report.ok = False
        return report

    mouth_neutral = neutral[mouth_mask, :3]
    extent = mouth_neutral.max(axis=0) - mouth_neutral.min(axis=0)
    report.mouth_scale = float(np.linalg.norm(extent))
    report.movement_floor = max(1.0e-6, report.mouth_scale * 0.005)
    report.aperture_floor = max(0.001, report.mouth_scale * 0.05)
    neutral_upper = _weighted_centroid(neutral, upper, upper_mask)
    neutral_lower = _weighted_centroid(neutral, lower, lower_mask)
    neutral_aperture = float(np.linalg.norm(neutral_upper - neutral_lower))

    for index in range(16):
        positions = normalized.get(index)
        if positions is None or positions.shape != neutral.shape:
            report.failures.append(f"shape_{index:02d}_missing")
            continue
        displacement = np.linalg.norm(
            positions[mouth_mask, :3] - mouth_neutral,
            axis=1,
        )
        upper_center = _weighted_centroid(positions, upper, upper_mask)
        lower_center = _weighted_centroid(positions, lower, lower_mask)
        aperture = float(np.linalg.norm(upper_center - lower_center))
        sample = FacialRangeSample(
            shape_index=index,
            mean_mouth_displacement=(
                float(displacement.mean()) if displacement.size else 0.0
            ),
            max_mouth_displacement=(
                float(displacement.max()) if displacement.size else 0.0
            ),
            aperture=aperture,
            aperture_delta=aperture - neutral_aperture,
        )
        report.samples[index] = sample

    report.active_shape_count = sum(
        1
        for index, sample in report.samples.items()
        if index != 0
        and sample.mean_mouth_displacement >= report.movement_floor
    )
    if report.active_shape_count < 8:
        report.failures.append("frozen_facial_skin")
    ah = report.samples.get(3)
    oh = report.samples.get(4)
    closed = report.samples.get(11)
    # Surface-transfer and binary float rounding can put an otherwise visible
    # opening a few micrometres below this heuristic floor.  Keep a tightly
    # bounded 0.2% tolerance, report the near miss, and continue to reject
    # materially undersized AH/OH shapes.
    aperture_tolerance = max(1.0e-5, report.aperture_floor * 0.002)
    if (
        ah is None
        or ah.aperture_delta + aperture_tolerance < report.aperture_floor
    ):
        report.failures.append("ah_aperture_too_small")
    elif ah.aperture_delta < report.aperture_floor:
        report.warnings.append("ah_aperture_near_floor")
    if (
        oh is None
        or oh.aperture_delta + aperture_tolerance < report.aperture_floor
    ):
        report.failures.append("oh_aperture_too_small")
    elif oh.aperture_delta < report.aperture_floor:
        report.warnings.append("oh_aperture_near_floor")
    if (
        ah is not None
        and closed is not None
        and closed.aperture_delta >= ah.aperture_delta * 0.5
    ):
        report.failures.append("mpb_does_not_close")
    report.failures = list(dict.fromkeys(report.failures))
    report.ok = not report.failures
    return report


def _weights_for_bones(
    skin_node: Any,
    bone_names: set[str] | frozenset[str],
) -> Any:
    import numpy as np

    palette = [
        str(name or "").casefold()
        for name in list(getattr(skin_node, "bone_map", ()) or ())
    ]
    slots = {
        index
        for index, name in enumerate(palette)
        if name in bone_names
    }
    output = np.zeros(len(getattr(skin_node, "vertices", ()) or ()), dtype=float)
    for vertex_index, row in enumerate(
        list(getattr(skin_node, "skin_data", ()) or ())
    ):
        output[vertex_index] = sum(
            float(getattr(influence, "weight", 0.0) or 0.0)
            for influence in list(getattr(row, "influences", ()) or ())
            if int(getattr(influence, "bone_index", -1)) in slots
        )
    return output


def _talk_shape_times(animation: Any) -> tuple[float, ...]:
    nodes = getattr(animation, "nodes", ()) or ()
    if isinstance(nodes, Mapping):
        nodes = nodes.values()
    for node in nodes:
        for controller in getattr(node, "controllers", ()) or ():
            try:
                times = tuple(
                    float(value)
                    for value in controller.get("times", ())
                )
                values = tuple(controller.get("values", ()))
            except (AttributeError, TypeError, ValueError):
                continue
            if len(times) >= 16 and len(values) >= 16:
                return times[:16]
    length = max(0.0, float(getattr(animation, "length", 0.0) or 0.0))
    if length > 0.0:
        return tuple(length * index / 15.0 for index in range(16))
    return tuple(index / 30.0 for index in range(16))


def audit_head_facial_range(model: Any) -> FacialRangeReport:
    """Skin *model* through its local/inherited talk slots and score the face."""

    import numpy as np

    model_name = str(getattr(model, "name", "") or "head")
    empty = FacialRangeReport(model_name=model_name)
    if model is None:
        empty.ok = False
        empty.failures.append("no_model")
        return empty
    try:
        from src.core.animation.animation_engine import AnimationEngine
        from src.core.rendering.skeleton_render_data import (
            cpu_skin_positions,
            extract_skinning_arrays,
        )
    except ImportError:  # pragma: no cover - embedded package route
        from core.animation.animation_engine import AnimationEngine  # type: ignore
        from core.rendering.skeleton_render_data import (  # type: ignore
            cpu_skin_positions,
            extract_skinning_arrays,
        )

    skin_node = None
    for node in model.all_nodes() if hasattr(model, "all_nodes") else ():
        palette = {
            str(name or "").casefold()
            for name in list(getattr(node, "bone_map", ()) or ())
        }
        if (
            list(getattr(node, "vertices", ()) or ())
            and list(getattr(node, "skin_data", ()) or ())
            and UPPER_LIP_BONES <= palette
            and LOWER_LIP_BONES <= palette
        ):
            skin_node = node
            break
    if skin_node is None:
        empty.ok = False
        empty.failures.append("facial_skin_missing")
        return empty

    engine = AnimationEngine(model)
    if not engine.play("talk", loop=False, blend=False):
        empty.ok = False
        empty.failures.append("talk_animation_missing")
        return empty
    animation = engine.current_animation
    if animation is None:
        empty.ok = False
        empty.failures.append("talk_animation_missing")
        return empty
    shape_times = _talk_shape_times(animation)
    vertices = np.asarray(skin_node.vertices, dtype=float)
    skinning = extract_skinning_arrays(
        skin_node,
        len(vertices),
        skeleton_id=id(model),
    )
    positions_by_shape: dict[int, Any] = {}
    for index, time_value in enumerate(shape_times):
        pose = engine.evaluate(time_value)
        try:
            setattr(pose, "_gr_animation_source_model_id", id(model))
            setattr(
                pose,
                "_gr_animation_source_model_name",
                model_name,
            )
            setattr(pose, "_gr_animation_name", "talk")
        except Exception:
            pass
        positions_by_shape[index] = np.asarray(
            cpu_skin_positions(
                skin_node,
                vertices,
                skinning,
                pose,
                model=model,
            ),
            dtype=float,
        )
    report = score_facial_range(
        model_name=model_name,
        shape_positions=positions_by_shape,
        upper_weights=_weights_for_bones(skin_node, UPPER_LIP_BONES),
        lower_weights=_weights_for_bones(skin_node, LOWER_LIP_BONES),
        mouth_weights=_weights_for_bones(skin_node, MOUTH_BONES),
    )
    report.talk_animation = str(getattr(animation, "name", "") or "talk")
    owner = getattr(animation, "_gr_source_model_name", None)
    report.talk_animation_owner = str(
        owner
        or getattr(model, "supermodel", "")
        or model_name
    )
    report.shape_times = shape_times
    return report


def audit_lip_timeline(
    lip_data: Any,
    *,
    name: str = "",
) -> LipTimelineReport:
    """Measure active/silent coverage in a parsed LIP V1.0 timeline."""

    try:
        from src.core.animation.facial_performance import lip_duration
    except ImportError:  # pragma: no cover - embedded package route
        from core.animation.facial_performance import lip_duration  # type: ignore

    duration = lip_duration(lip_data)
    frames = sorted(
        list(getattr(lip_data, "keyframes", ()) or ()),
        key=lambda frame: float(getattr(frame, "time", 0.0) or 0.0),
    )
    report = LipTimelineReport(
        name=str(name or getattr(lip_data, "source_path", "") or "lip"),
        duration=duration,
        keyframe_count=len(frames),
    )
    if not frames:
        report.ok = False
        report.failures.append("no_keyframes")
        return report
    shape_counts: dict[int, int] = {}
    shape_durations: dict[int, float] = {}
    max_gap = 0.0
    for index, frame in enumerate(frames):
        shape = max(0, min(15, int(getattr(frame, "shape", 0) or 0)))
        start = max(0.0, float(getattr(frame, "time", 0.0) or 0.0))
        end = (
            max(start, float(getattr(frames[index + 1], "time", start) or start))
            if index + 1 < len(frames)
            else max(start, duration)
        )
        gap = max(0.0, end - start)
        max_gap = max(max_gap, gap)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        shape_durations[shape] = shape_durations.get(shape, 0.0) + gap
    report.shape_counts = shape_counts
    report.shape_durations = {
        shape: round(value, 6)
        for shape, value in shape_durations.items()
    }
    report.unique_shape_count = len(shape_counts)
    report.neutral_duration = round(shape_durations.get(0, 0.0), 6)
    active_duration = max(0.0, duration - report.neutral_duration)
    report.active_duration_fraction = round(
        active_duration / duration if duration > 0.0 else 0.0,
        6,
    )
    report.max_keyframe_gap = round(max_gap, 6)
    if not any(shape != 0 for shape in shape_counts):
        report.failures.append("no_active_visemes")
    if report.unique_shape_count < 3:
        report.warnings.append("low_shape_variety")
    if duration <= 0.0:
        report.failures.append("invalid_duration")
    report.ok = not report.failures
    return report


__all__ = [
    "AudioLipSyncReport",
    "FacialRangeReport",
    "FacialRangeSample",
    "LipTimelineReport",
    "LOWER_LIP_BONES",
    "MOUTH_BONES",
    "UPPER_LIP_BONES",
    "audit_audio_lip_sync",
    "audit_head_facial_range",
    "audit_lip_timeline",
    "score_facial_range",
]
