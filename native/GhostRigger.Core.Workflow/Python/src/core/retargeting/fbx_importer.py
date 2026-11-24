"""FBX Animation Importer - Retarget Workbench Production Path.

Backend: Blender 4.2 headless (``bpy.ops.import_scene.fbx``).
Pipeline: FBX -> Blender Headless -> Sampled JSON -> SourceSkeletonClip.
Output: ``SourceSkeletonClip`` for animation retargeting.

IMPORTANT: this is the production animation backend. Keep core logic stable.
IMPORTANT: this module is not using Autodesk FBX SDK; see ``fbx_backend.py`` for
optional SDK probing and backend selection.
IMPORTANT: this is an animation-only pipeline, not generic mesh/geometry import.

The importer remains backend-neutral for tests and future tool integrations. A
real FBX backend must provide evaluated global matrices; if none is configured,
the public import function fails explicitly instead of pretending raw channels
are safe enough.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .coordinate import UE_UNIT_SCALE_TO_METERS, UE_X_FORWARD_Y_RIGHT_Z_UP
from .source_animation import (
    SourcePose,
    SourceSkeletonClip,
    SourceSkeletonNode,
    Transform,
    hemisphere_continuity_xyzw,
)


FRAME0_REFERENCE_WARNING = "No complete FBX bind/default pose found; using frame 0 as source reference pose."
BLENDER_FBX_IMPORT_AXIS_SYSTEM = "blender_fbx_import_z_up"
BLENDER_FBX_IMPORT_UNIT_SCALE_TO_METERS = 1.0
BLENDER_FBX_IMPORT_HANDEDNESS = "right-handed"


class FbxImportUnavailableError(RuntimeError):
    """Raised when no supported FBX import backend is available."""


class FbxImportError(ValueError):
    """Raised when an FBX source cannot be imported deterministically."""


@dataclass(frozen=True)
class FbxBackendNode:
    """Minimal backend-neutral node record used by the importer."""

    name: str
    parent_name: Optional[str] = None


@dataclass(frozen=True)
class FbxBackendClip:
    """Minimal backend-neutral animation stack record."""

    name: str
    duration_seconds: float


class FbxBackendScene(Protocol):
    """Protocol expected from a project-supported FBX backend scene."""

    source_path: str
    nodes: Sequence[FbxBackendNode]
    clips: Sequence[FbxBackendClip]
    axis_system: Optional[str]
    unit_scale_to_meters: Optional[float]
    handedness: Optional[str]

    def evaluate_global_transform(self, node_name: str, time_seconds: float, clip_name: str) -> Iterable[Iterable[float]]:
        ...


class FbxBackend(Protocol):
    """Protocol for backend objects that can load an FBX scene."""

    def load_scene(self, path: str) -> FbxBackendScene:
        ...


@dataclass
class BlenderFbxBackendScene:
    """FBX animation scene extracted through GhostRigger's Blender bridge."""

    source_path: str
    nodes: Sequence[FbxBackendNode]
    clips: Sequence[FbxBackendClip]
    curves: Dict[str, list[tuple[float, np.ndarray]]]
    rest_matrices: Dict[str, np.ndarray]
    axis_system: str = BLENDER_FBX_IMPORT_AXIS_SYSTEM
    unit_scale_to_meters: float = BLENDER_FBX_IMPORT_UNIT_SCALE_TO_METERS
    handedness: str = BLENDER_FBX_IMPORT_HANDEDNESS
    warnings: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def evaluate_global_transform(self, node_name: str, time_seconds: float, clip_name: str) -> np.ndarray:
        samples = self.curves.get(node_name) or []
        if not samples:
            rest = self.bind_global_transform(node_name)
            return np.asarray(rest if rest is not None else np.eye(4), dtype=np.float64)
        wanted = float(time_seconds)
        for sample_time, matrix in samples:
            if abs(sample_time - wanted) <= 1e-6:
                return np.asarray(matrix, dtype=np.float64)
        nearest_time, nearest_matrix = min(samples, key=lambda item: abs(item[0] - wanted))
        return np.asarray(nearest_matrix, dtype=np.float64)

    def bind_global_transform(self, node_name: str) -> np.ndarray | None:
        matrix = self.rest_matrices.get(node_name)
        if matrix is None:
            return None
        return np.asarray(matrix, dtype=np.float64)


@dataclass
class BlenderFbxBackend:
    """Production FBX source backend using headless Blender evaluated transforms."""

    blender_executable: str | Path | None = None
    extraction_root: Path | None = None
    timeout: int = 300
    extraction_runner: Callable[..., dict[str, Any]] | None = None

    def load_scene(
        self,
        path: str,
        *,
        clip_name: str | None = None,
        sample_rate: float | None = None,
    ) -> BlenderFbxBackendScene:
        source = Path(path)
        if not source.exists():
            raise FbxImportError(f"FBX source file not found: {source}")
        payload = self._extract_payload(source, clip_name=clip_name)
        if not payload.get("success"):
            errors = [str(error) for error in (payload.get("errors") or []) if str(error).strip()]
            message = "; ".join(errors) if errors else "Blender FBX extraction failed."
            if "Blender 4.2 LTS was not found" in message:
                raise FbxImportUnavailableError(message)
            raise FbxImportError(message)
        return _scene_from_blender_payload(payload, source_path=str(source))

    def _extract_payload(self, source: Path, *, clip_name: str | None) -> dict[str, Any]:
        from .blender_animation_injection import run_blender_animation_extraction

        runner = self.extraction_runner or run_blender_animation_extraction
        output_json = self._output_json_path(source, clip_name=clip_name)
        return runner(
            source_fbx=source,
            output_json=output_json,
            action_name=str(clip_name or ""),
            frame_step=1,
            blender_executable=Path(self.blender_executable) if self.blender_executable else None,
            timeout=int(self.timeout),
        )

    def _output_json_path(self, source: Path, *, clip_name: str | None) -> Path:
        root = self.extraction_root
        if root is None:
            env_root = os.environ.get("GHOSTRIGGER_FBX_IMPORT_CACHE")
            root = Path(env_root) if env_root else Path(tempfile.gettempdir()) / "ghostrigger_fbx_import"
        root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{source.resolve()}|{source.stat().st_mtime_ns}|{source.stat().st_size}|{clip_name or ''}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        return root / f"{source.stem}_{digest}_source_clip.json"


def classify_source_node_name(name: str) -> str:
    """Classify a UE/FBX source node name without discarding it."""

    raw = str(name or "").strip()
    lowered = raw.lower().replace(" ", "_")
    if lowered in {"root", "scene", "armature"}:
        return "root"
    if (
        "twist" in lowered
        or "roll" in lowered
        or "forearm_twist" in lowered
        or "upperarm_twist" in lowered
        or "thigh_twist" in lowered
        or "calf_twist" in lowered
    ):
        return "twist"
    if (
        lowered.startswith("ik_")
        or "_ik" in lowered
        or "ik_hand" in lowered
        or "ik_foot" in lowered
        or "effector" in lowered
        or "pole" in lowered
    ):
        return "ik"
    if (
        "helper" in lowered
        or "socket" in lowered
        or "weapon" in lowered
        or "attach" in lowered
        or "virtual" in lowered
        or lowered.startswith("vb_")
        or lowered.startswith("vb")
    ):
        return "helper"
    return "deform"


def import_ue_fbx_animation_clip(
    path: str,
    *,
    clip_name: Optional[str] = None,
    sample_rate: float = 30.0,
    reference_pose: str | float | None = None,
    backend: Optional[FbxBackend] = None,
) -> SourceSkeletonClip:
    """Import and sample a UE/FBX skeleton animation clip."""

    backend = backend or _default_backend()
    scene = _load_backend_scene(backend, str(path), clip_name=clip_name, sample_rate=sample_rate)
    nodes = _topological_nodes(scene.nodes)
    if not nodes:
        raise FbxImportError("FBX file contains no source skeleton nodes.")
    selected_clip = _select_clip(scene.clips, clip_name)
    if selected_clip.duration_seconds < 0:
        raise FbxImportError(f"FBX animation stack '{selected_clip.name}' has negative duration.")
    sample_times = _sample_times(selected_clip.duration_seconds, sample_rate)

    warnings: List[str] = list(getattr(scene, "warnings", None) or [])
    axis_system = getattr(scene, "axis_system", None) or UE_X_FORWARD_Y_RIGHT_Z_UP
    unit_scale = getattr(scene, "unit_scale_to_meters", None)
    if unit_scale is None:
        unit_scale = UE_UNIT_SCALE_TO_METERS
        warnings.append("FBX unit scale metadata unavailable; assuming Unreal centimeters (0.01 meters).")
    handedness = getattr(scene, "handedness", None) or "left-handed"

    global_by_time: Dict[float, Dict[str, np.ndarray]] = {}
    previous_quat_by_node: Dict[str, tuple[float, float, float, float]] = {}
    sampled_poses: List[SourcePose] = []
    for time_value in sample_times:
        globals_for_time: Dict[str, np.ndarray] = {}
        for node in nodes:
            matrix = np.asarray(
                scene.evaluate_global_transform(node.name, time_value, selected_clip.name),
                dtype=np.float64,
            )
            if matrix.shape != (4, 4):
                raise FbxImportError(f"Backend returned non-4x4 global matrix for node '{node.name}'.")
            globals_for_time[node.name] = matrix
        global_by_time[time_value] = globals_for_time
        pose = _pose_from_global_matrices(nodes, globals_for_time, time_value)
        for node_name, transform in pose.global_transforms.items():
            fixed = hemisphere_continuity_xyzw(transform.rotation, previous_quat_by_node.get(node_name))
            previous_quat_by_node[node_name] = fixed
            pose.global_transforms[node_name] = Transform(
                position=transform.position,
                rotation=fixed,
                scale=transform.scale,
            )
        sampled_poses.append(pose)

    rest_global_matrices = _reference_global_matrices(
        scene,
        nodes,
        selected_clip.name,
        reference_pose,
        global_by_time,
        warnings,
    )
    rest_pose = _pose_from_global_matrices(nodes, rest_global_matrices, 0.0)
    source_nodes = [
        SourceSkeletonNode(
            name=node.name,
            parent_name=node.parent_name,
            index=index,
            rest_local=rest_pose.local_transforms[node.name],
            rest_global=rest_pose.global_transforms[node.name],
            classification=classify_source_node_name(node.name),
        )
        for index, node in enumerate(nodes)
    ]

    return SourceSkeletonClip(
        source_path=str(path),
        clip_name=selected_clip.name,
        duration_seconds=float(selected_clip.duration_seconds),
        sample_rate=float(sample_rate),
        nodes=source_nodes,
        rest_pose=rest_pose,
        sampled_poses=sampled_poses,
        axis_system=axis_system,
        unit_scale_to_meters=float(unit_scale),
        handedness=handedness,
        import_warnings=warnings,
        available_clips=_available_clip_entries(scene, fallback_clips=scene.clips),
    )


def _default_backend() -> FbxBackend:
    return BlenderFbxBackend()


def _load_backend_scene(
    backend: FbxBackend,
    path: str,
    *,
    clip_name: str | None,
    sample_rate: float,
) -> FbxBackendScene:
    try:
        return backend.load_scene(path, clip_name=clip_name, sample_rate=sample_rate)  # type: ignore[call-arg]
    except TypeError as exc:
        text = str(exc)
        if "unexpected keyword" not in text and "positional" not in text:
            raise
    return backend.load_scene(path)


STATIC_SAMPLE_EPSILON = 1e-6


def _check_blender_sample_not_static(
    payload: dict[str, Any],
    curves: Dict[str, list[tuple[float, np.ndarray]]],
    action_name: str,
    warnings: List[str],
) -> None:
    """Fail when every sampled pose of a multi-frame clip is identical.

    Blender 4.4+ slotted actions sample the rest pose on every frame when the
    action slot is not bound; the resulting MDL ships frozen animations and the
    FBX/MDL parity gate cannot see it (both sides are equally static). A clip is
    accepted as a genuine static hold only when the extraction script reports
    that the action itself has no keyed transform motion.
    """

    multi_frame = [samples for samples in curves.values() if len(samples) >= 2]
    if not multi_frame:
        return
    spread = 0.0
    for samples in multi_frame:
        first = samples[0][1]
        for _time_value, matrix in samples[1:]:
            spread = max(spread, float(np.max(np.abs(matrix - first))))
            if spread > STATIC_SAMPLE_EPSILON:
                return
    key_motion = payload.get("action_key_motion")
    if key_motion is not None and float(key_motion) <= STATIC_SAMPLE_EPSILON:
        warnings.append(f"FBX clip '{action_name}' is a static hold: the action has no keyed motion.")
        return
    frame_count = max(len(samples) for samples in multi_frame)
    raise FbxImportError(
        f"Static FBX sample: all {frame_count} sampled poses of clip '{action_name}' are identical "
        f"(Blender {payload.get('blender_version') or 'unknown'}, action key motion "
        f"{key_motion if key_motion is not None else 'not reported'}). The action did not drive the "
        "armature - check that scripts/blender_extract_ue5_animation.py binds the action slot "
        "(Blender 4.4+)."
    )


def _scene_from_blender_payload(payload: dict[str, Any], *, source_path: str) -> BlenderFbxBackendScene:
    source_bones = [str(name) for name in (payload.get("source_bones") or []) if str(name).strip()]
    parent_map = {
        str(name): (str(parent) if parent is not None else None)
        for name, parent in (payload.get("bone_parents") or {}).items()
    }
    if not source_bones:
        source_bones = list(parent_map)
    nodes = [FbxBackendNode(name=name, parent_name=parent_map.get(name)) for name in source_bones]
    if not nodes:
        raise FbxImportError("Blender FBX extraction produced no armature bones.")

    curves_payload = payload.get("curves") or {}
    curves: Dict[str, list[tuple[float, np.ndarray]]] = {}
    for node_name in source_bones:
        entries = curves_payload.get(node_name) or []
        samples: list[tuple[float, np.ndarray]] = []
        for entry in entries:
            try:
                time_value = float(entry.get("time_seconds", 0.0))
                samples.append((time_value, _matrix_from_blender_entry(entry)))
            except Exception as exc:
                raise FbxImportError(f"Invalid Blender FBX matrix sample for node '{node_name}': {exc}") from exc
        samples.sort(key=lambda item: item[0])
        curves[node_name] = samples

    rest_matrices: Dict[str, np.ndarray] = {}
    for node_name, rest in (payload.get("rest_world") or {}).items():
        matrix = rest.get("matrix") if isinstance(rest, dict) else None
        if matrix is not None:
            rest_matrices[str(node_name)] = _as_4x4_matrix(matrix)

    duration = _duration_from_blender_payload(payload, curves)
    action_name = str(payload.get("action_name") or "FBXAction")
    warnings = [
        "FBX animation imported through Blender; transforms are evaluated after Blender's FBX axis/unit conversion."
    ]
    _check_blender_sample_not_static(payload, curves, action_name, warnings)
    log_path = payload.get("log_path")
    metadata = {
        "backend": "blender",
        "armature_name": payload.get("armature_name"),
        "actions": _actions_from_blender_payload(payload),
        "frame_start": payload.get("frame_start"),
        "frame_end": payload.get("frame_end"),
        "frame_count": payload.get("frame_count"),
        "fps": payload.get("fps"),
        "mesh_count": payload.get("mesh_count"),
        "log_path": log_path,
    }
    return BlenderFbxBackendScene(
        source_path=str(source_path),
        nodes=nodes,
        clips=[FbxBackendClip(action_name, duration)],
        curves=curves,
        rest_matrices=rest_matrices,
        warnings=warnings,
        metadata=metadata,
    )


def _available_clip_entries(
    scene: FbxBackendScene,
    *,
    fallback_clips: Sequence[FbxBackendClip],
) -> list[dict[str, Any]]:
    metadata = getattr(scene, "metadata", None) or {}
    actions = metadata.get("actions") if isinstance(metadata, dict) else None
    entries: list[dict[str, Any]] = []
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = str(action.get("name") or "").strip()
            if not name:
                continue
            entry = dict(action)
            entry["name"] = name
            entries.append(entry)
    if entries:
        return entries
    return [
        {
            "name": str(clip.name),
            "duration_seconds": float(clip.duration_seconds),
        }
        for clip in (fallback_clips or [])
    ]


def _actions_from_blender_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return []
    fps = _safe_float(payload.get("fps"), fallback=30.0)
    out: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = str(action.get("name") or "").strip()
        if not name:
            continue
        frame_start = _safe_float(action.get("frame_start"), fallback=0.0)
        frame_end = _safe_float(action.get("frame_end"), fallback=frame_start)
        duration = max(0.0, (frame_end - frame_start) / fps) if fps > 0 else 0.0
        out.append(
            {
                "name": name,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_count": int(max(0.0, frame_end - frame_start + 1.0)),
                "duration_seconds": duration,
            }
        )
    return out


def _safe_float(value: Any, *, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(result):
        return float(fallback)
    return result


def _matrix_from_blender_entry(entry: dict[str, Any]) -> np.ndarray:
    matrix = entry.get("matrix")
    if matrix is not None:
        return _as_4x4_matrix(matrix)
    location = entry.get("location_xyz") or (0.0, 0.0, 0.0)
    rotation = entry.get("rotation_wxyz") or (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = (float(value) for value in list(rotation)[:4])
    return Transform(
        position=tuple(float(value) for value in list(location)[:3]),  # type: ignore[arg-type]
        rotation=(x, y, z, w),
    ).to_matrix()


def _as_4x4_matrix(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {value.shape}")
    return value


def _duration_from_blender_payload(
    payload: dict[str, Any],
    curves: Dict[str, list[tuple[float, np.ndarray]]],
) -> float:
    times = [time_value for samples in curves.values() for time_value, _matrix in samples]
    if times:
        return max(0.0, max(times))
    try:
        frame_start = float(payload.get("frame_start") or 0.0)
        frame_end = float(payload.get("frame_end") or frame_start)
        fps = float(payload.get("fps") or 30.0)
        return max(0.0, (frame_end - frame_start) / fps)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _select_clip(clips: Sequence[FbxBackendClip], clip_name: Optional[str]) -> FbxBackendClip:
    available = list(clips or [])
    if not available:
        raise FbxImportError("FBX file contains a skeleton but no animation stack.")
    if clip_name:
        wanted = str(clip_name).lower()
        for clip in available:
            if clip.name.lower() == wanted:
                return clip
        raise FbxImportError(
            f"FBX file does not contain animation stack '{clip_name}'. "
            f"Available stacks: {', '.join(clip.name for clip in available)}."
        )
    if len(available) > 1:
        raise FbxImportError(
            "FBX file contains multiple animation stacks: "
            f"{', '.join(clip.name for clip in available)}. Pass clip_name explicitly."
        )
    return available[0]


def _topological_nodes(nodes: Sequence[FbxBackendNode]) -> List[FbxBackendNode]:
    by_name: Dict[str, FbxBackendNode] = {}
    for node in nodes:
        key = node.name.lower()
        if key in by_name:
            raise FbxImportError(
                f"Source skeleton has duplicate node name '{node.name}'; cannot build deterministic retarget mapping."
            )
        by_name[key] = node

    ordered: List[FbxBackendNode] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: FbxBackendNode) -> None:
        key = node.name.lower()
        if key in permanent:
            return
        if key in temporary:
            raise FbxImportError(f"Source skeleton hierarchy has a cycle at '{node.name}'.")
        temporary.add(key)
        if node.parent_name:
            parent = by_name.get(node.parent_name.lower())
            if parent is None:
                raise FbxImportError(f"Source skeleton parent '{node.parent_name}' for '{node.name}' is missing.")
            visit(parent)
        temporary.remove(key)
        permanent.add(key)
        ordered.append(node)

    for node in nodes:
        visit(node)
    return ordered


def _sample_times(duration_seconds: float, sample_rate: float) -> List[float]:
    duration = max(0.0, float(duration_seconds))
    rate = float(sample_rate or 30.0)
    if rate <= 0.0:
        raise FbxImportError("sample_rate must be positive.")
    frame_count = int(math.floor(duration * rate + 1e-7))
    times = {0.0, duration}
    for frame in range(frame_count + 1):
        value = min(duration, frame / rate)
        times.add(round(value, 10))
    return sorted(times)


def _pose_from_global_matrices(
    nodes: Sequence[FbxBackendNode],
    globals_for_time: Dict[str, np.ndarray],
    time_seconds: float,
) -> SourcePose:
    global_transforms: Dict[str, Transform] = {}
    local_transforms: Dict[str, Transform] = {}
    for node in nodes:
        global_matrix = globals_for_time[node.name]
        global_transforms[node.name] = Transform.from_matrix(global_matrix)
        if node.parent_name:
            parent_matrix = globals_for_time[node.parent_name]
            local_matrix = np.linalg.inv(parent_matrix) @ global_matrix
        else:
            local_matrix = global_matrix
        local_transforms[node.name] = Transform.from_matrix(local_matrix)
    return SourcePose(
        time_seconds=float(time_seconds),
        global_transforms=global_transforms,
        local_transforms=local_transforms,
    )


def _reference_global_matrices(
    scene: FbxBackendScene,
    nodes: Sequence[FbxBackendNode],
    clip_name: str,
    reference_pose: str | float | None,
    sampled_globals: Dict[float, Dict[str, np.ndarray]],
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    if isinstance(reference_pose, (int, float)):
        return {
            node.name: np.asarray(
                scene.evaluate_global_transform(node.name, float(reference_pose), clip_name),
                dtype=np.float64,
            )
            for node in nodes
        }

    for method_name in ("bind_global_transform", "default_global_transform"):
        method = getattr(scene, method_name, None)
        if method is None:
            continue
        matrices: Dict[str, np.ndarray] = {}
        complete = True
        for node in nodes:
            value = method(node.name)
            if value is None:
                complete = False
                break
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4):
                complete = False
                break
            matrices[node.name] = matrix
        if complete:
            return matrices

    warnings.append(FRAME0_REFERENCE_WARNING)
    first_time = min(sampled_globals)
    return sampled_globals[first_time]
