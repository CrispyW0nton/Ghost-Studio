"""FBX export for GhostRigger retargeted Unity test assets.

Day 4 keeps FBX writing behind a headless Blender subprocess.  GhostRigger
serializes a deterministic intermediate JSON; Blender 4.0+ LTS builds the scene
and uses its production FBX exporter.

References:
  - Blender FBX operator:
    https://docs.blender.org/api/current/bpy.ops.export_scene.html
  - Blender command-line background execution:
    https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html
  - Autodesk FBX axis systems:
    https://help.autodesk.com/cloudhelp/2020/ENU/FBX-API-Reference/cpp_ref/class_fbx_axis_system.html
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np

from .baker import bake_retargeted_clip, compute_bind_offsets, max_quat_component_delta
from .coordinate_normalizer import BindPoseRegistry, CoordinateNormalizer, compose_matrix, matrix_to_quat_wxyz
from .mesh_loader import load_kotor_skinned_mesh
from .mesh_rebinder import RebindOptions, ReboundMesh, SourceMesh, rebind_mesh_to_target_skeleton
from .sampler import DEFAULT_CORPUS_ROOT, SampledClip, load_fixture_model, sample_clip_to_fixed_rate
from .skeleton_aligner import AlignedSkeleton, aligned_skeleton_to_registry, compute_local_from_world
from .skeleton_renamer import RenameSpec, _resolve_repo_root, load_rename_spec, validate_rename_spec
from src.unreal.animation_retargeting import build_bone_map
from src.unreal.quinn import _read_binary_fbx, load_quinn_skeleton_asset, unreal_skeleton_model


# The Workflow payload lives below native/GhostRigger.Core.Workflow/Python, so a
# fixed parents[3] pointed at that package directory, where neither the Blender
# script nor the Day 4 fixtures exist.  Share the renamer's repository-root
# resolver (GHOSTRIGGER_NATIVE_REPO_ROOT, then an ancestor/cwd scan).
REPO_ROOT = _resolve_repo_root()
DEFAULT_EXPORT_DIR = REPO_ROOT / "exports" / "retargets" / "day4"
DEFAULT_DAY45_EXPORT_DIR = REPO_ROOT / "exports" / "retargets" / "day4_5"
DEFAULT_LOG_DIR = REPO_ROOT / "Logs" / "retargeting"
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "blender_fbx_export.py"


class FBXExportFailure(RuntimeError):
    """Raised when Blender export or roundtrip validation fails."""


@dataclass
class FBXExportOptions:
    """User-facing export options for the Day 4 FBX path."""

    source_model_id: str
    source_supermodel: str
    target_skeleton_id: str
    clip_names: List[str]
    output_path: Path
    fbx_version: Literal["FBX201600", "FBX202000"] = "FBX201600"
    axis_forward: Literal["X", "Y", "Z", "-X", "-Y", "-Z"] = "-Y"
    axis_up: Literal["X", "Y", "Z"] = "Z"
    global_scale: float = 1.0
    apply_unit_scale: bool = True
    add_leaf_bones: bool = False
    primary_bone_axis: Literal["X", "Y", "Z", "-X", "-Y", "-Z"] = "Y"
    secondary_bone_axis: Literal["X", "Y", "Z", "-X", "-Y", "-Z"] = "X"
    bake_anim: bool = True
    bake_anim_use_all_bones: bool = True
    bake_anim_use_nla_strips: bool = False
    bake_anim_use_all_actions: bool = False
    bake_anim_force_startend_keying: bool = True
    bake_anim_step: float = 1.0
    bake_anim_simplify_factor: float = 0.0
    use_mesh_modifiers: bool = False
    mesh_smooth_type: Literal["OFF", "FACE", "EDGE"] = "FACE"
    use_tspace: bool = True
    use_armature_deform_only: bool = True
    keep_intermediate: bool = True
    run_roundtrip_validation: bool = True
    blender_executable: Optional[Path] = None


@dataclass
class ClipManifestEntry:
    clip_name: str
    source_supermodel: str
    frame_count: int
    fps: float
    duration_seconds: float
    sampling_strategy: str = "fixed_60fps_slerp"


@dataclass
class FBXExportManifest:
    """Sidecar data written next to every retargeted FBX."""

    fbx_path: Path
    fbx_version: str
    fbx_sha256: str
    blender_version: str
    export_timestamp: str
    source_mesh_resref: str
    source_skeleton_id: str
    aligned_skeleton_id: str
    bone_map_version: str
    clip_inventory: List[ClipManifestEntry]
    bind_pose_validation: Dict[str, float]
    axis_system: Dict[str, str]
    roundtrip_metrics: Dict[str, Any]
    intermediate_path: Path
    manifest_path: Path
    schema_version: str = "1.0"

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("fbx_path", "intermediate_path", "manifest_path"):
            payload[key] = str(payload[key])
        return payload


def _key(name: str) -> str:
    return str(name or "").strip().lower()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_deterministic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    path.write_text(text + "\n", encoding="utf-8")


# Minimum supported Blender version. Any 4.x release >= 4.0 works; the previous
# hardcoded 4.2 requirement was relaxed so standard installs (4.0, 4.1, 4.2,
# 4.3, ...) are all accepted.
MINIMUM_BLENDER_VERSION = (4, 0, 0)
# scripts/blender_fbx_export.py uses the legacy ``Action.fcurves`` API, which
# Blender 5.0 removed (slotted actions/channelbags). Under 5.x the script
# raised and the background process then hung until the 300 s timeout, so 5.x
# is rejected up front and discovery prefers the newest 4.x install.
MAXIMUM_BLENDER_MAJOR = 4


def _blender_install_roots() -> list[Path]:
    """Return Blender install search roots, in priority order.

    Mirrors the locations used by the official Windows installers and the
    Blender launcher (Program Files, Program Files (x86), and the per-user
    Programs folder under LOCALAPPDATA).
    """
    roots: list[Path] = []
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if program_files:
        roots.append(Path(program_files) / "Blender Foundation")
    if program_files_x86:
        roots.append(Path(program_files_x86) / "Blender Foundation")
    if local_app_data:
        roots.append(Path(local_app_data) / "Programs" / "Blender")
    return roots


def _parse_version_tuple(text: str) -> tuple[int, ...]:
    """Extract a numeric version tuple from text such as a folder/version line."""
    parts = re.findall(r"\d+", text)
    return tuple(int(p) for p in parts) if parts else ()


def _discover_blender_executables() -> list[Path]:
    """Return Blender executables found in common install locations, newest first."""
    found: list[tuple[tuple[int, ...], Path]] = []
    for root in _blender_install_roots():
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            exe = child / "blender.exe"
            if exe.exists():
                found.append((_parse_version_tuple(child.name), exe))
    # Supported majors first (newest first), then newer unsupported majors;
    # missing-version entries sort last.
    found.sort(
        key=lambda item: (
            bool(item[0]) and item[0][0] <= MAXIMUM_BLENDER_MAJOR,
            item[0],
        ),
        reverse=True,
    )
    return [exe for _, exe in found]


def _version_meets_minimum(version_text: str) -> bool:
    parsed = list(_parse_version_tuple(version_text))
    if not parsed:
        return False
    while len(parsed) < len(MINIMUM_BLENDER_VERSION):
        parsed.append(0)
    return tuple(parsed) >= MINIMUM_BLENDER_VERSION


def find_blender_executable(explicit: str | Path | None = None) -> Path:
    """Resolve a Blender 4.0+ executable.

    Resolution order:
      1. ``explicit`` (caller-supplied ``FBXExportOptions.blender_executable``)
      2. ``GHOSTRIGGER_BLENDER_PATH`` environment variable
      3. Common Windows install locations (newest version first)
      4. ``blender`` on ``PATH`` (``shutil.which``)
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("GHOSTRIGGER_BLENDER_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend(_discover_blender_executables())
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FBXExportFailure(
        "Blender 4.0 or newer was not found. Install Blender 4.0+ to a standard "
        "location (C:\\Program Files\\Blender Foundation\\, "
        "C:\\Program Files (x86)\\Blender Foundation\\, or "
        "%LOCALAPPDATA%\\Programs\\Blender\\), or set the GHOSTRIGGER_BLENDER_PATH "
        "environment variable, or pass FBXExportOptions.blender_executable."
    )


def blender_version(blender_executable: str | Path | None = None) -> str:
    exe = find_blender_executable(blender_executable)
    result = subprocess.run(
        [str(exe), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise FBXExportFailure(f"Blender version check failed:\n{result.stdout}\n{result.stderr}")
    first = (result.stdout or "").splitlines()[0].strip()
    if not _version_meets_minimum(first):
        raise FBXExportFailure(
            f"Blender {MINIMUM_BLENDER_VERSION[0]}.{MINIMUM_BLENDER_VERSION[1]}+ is required, found: {first}"
        )
    parsed = _parse_version_tuple(first)
    if parsed and parsed[0] > MAXIMUM_BLENDER_MAJOR:
        raise FBXExportFailure(
            f"Blender {MAXIMUM_BLENDER_MAJOR}.x is required (the export script uses the legacy "
            f"Action.fcurves API removed in 5.0), found: {first}"
        )
    return first


def _export_bone_names(aligned: AlignedSkeleton) -> list[str]:
    return [
        name for name in aligned.bone_names
        if _key(name) != "skm_quinn_simple"
    ]


def _export_parent_map(aligned: AlignedSkeleton, bone_names: Iterable[str]) -> dict[str, str | None]:
    allowed = {_key(name) for name in bone_names}
    parents: dict[str, str | None] = {}
    for name in bone_names:
        key = _key(name)
        parent = _key(aligned.bone_parents.get(key) or "")
        parents[key] = parent if parent in allowed else None
    return parents


def _export_local_from_world(aligned: AlignedSkeleton, bone_names: list[str]) -> Dict[str, np.ndarray]:
    parents = _export_parent_map(aligned, bone_names)
    world = {_key(name): np.asarray(aligned.bind_world[_key(name)], dtype=np.float64) for name in bone_names}
    return compute_local_from_world(world, parents)


def _matrix_to_list_map(mats: Dict[str, np.ndarray], bone_names: Iterable[str]) -> dict[str, list[list[float]]]:
    return {
        _key(name): np.asarray(mats[_key(name)], dtype=np.float64).tolist()
        for name in bone_names
    }


def _clip_basis_payload(clip: SampledClip, bone_names: list[str], bind_local: Dict[str, np.ndarray]) -> Dict[str, Any]:
    clip_index = {_key(name): idx for idx, name in enumerate(clip.bone_names)}
    translations = np.zeros((clip.frame_count, len(bone_names), 3), dtype=np.float64)
    rotations = np.zeros((clip.frame_count, len(bone_names), 4), dtype=np.float64)
    scales = np.ones((clip.frame_count, len(bone_names), 3), dtype=np.float64)
    rotations[:, :, 0] = 1.0

    for bone_idx, bone_name in enumerate(bone_names):
        key = _key(bone_name)
        src_idx = clip_index.get(key)
        inv_bind = np.linalg.inv(np.asarray(bind_local[key], dtype=np.float64))
        for frame in range(clip.frame_count):
            if src_idx is None:
                local = np.asarray(bind_local[key], dtype=np.float64)
            else:
                local = compose_matrix(
                    clip.positions[frame, src_idx, :],
                    clip.rotations[frame, src_idx, :],
                    clip.scales[frame, src_idx, :],
                )
            basis = inv_bind @ local
            translations[frame, bone_idx, :] = basis[:3, 3]
            rotations[frame, bone_idx, :] = matrix_to_quat_wxyz(basis)
            scales[frame, bone_idx, :] = (
                np.linalg.norm(basis[:3, 0]),
                np.linalg.norm(basis[:3, 1]),
                np.linalg.norm(basis[:3, 2]),
            )

    return {
        "name": clip.clip_name,
        "fps": float(clip.fps),
        "frame_count": int(clip.frame_count),
        "duration_seconds": float(clip.duration_s),
        "source_supermodel": clip.resolved_clip_source,
        "bone_names": [_key(name) for name in bone_names],
        "translations": translations.tolist(),
        "rotations": rotations.tolist(),
        "scales": scales.tolist(),
    }


def build_intermediate_representation(
    rebound_mesh: ReboundMesh,
    baked_clips: List[SampledClip],
    aligned_skeleton: AlignedSkeleton,
    options: FBXExportOptions,
) -> Dict[str, Any]:
    """Build deterministic JSON consumed by Blender.

    The export skeleton deliberately excludes the Unreal asset wrapper
    ``SKM_Quinn_Simple`` because the mesh weights are indexed against Quinn's
    89 real bones.  The top-level FBX armature object carries the wrapper role.
    """

    bone_names = _export_bone_names(aligned_skeleton)
    bind_world = {_key(name): np.asarray(aligned_skeleton.bind_world[_key(name)], dtype=np.float64) for name in bone_names}
    bind_local = _export_local_from_world(aligned_skeleton, bone_names)
    parents = _export_parent_map(aligned_skeleton, bone_names)
    target_names = list(rebound_mesh.transplant_metadata.get("target_bone_names", bone_names))
    if [_key(name) for name in target_names] != [_key(name) for name in bone_names]:
        target_names = bone_names

    return {
        "schema_version": "1.0",
        "created_for": "GhostRigger Day 4 FBX export",
        "documentation": {
            "blender_fbx_operator": "https://docs.blender.org/api/current/bpy.ops.export_scene.html",
            "autodesk_axis_system": "https://help.autodesk.com/cloudhelp/2020/ENU/FBX-API-Reference/cpp_ref/class_fbx_axis_system.html",
        },
        "options": _options_to_dict(options),
        "skeleton": {
            "skeleton_id": aligned_skeleton.skeleton_id,
            "base_target_id": aligned_skeleton.base_target_id,
            "source_id": aligned_skeleton.source_id,
            "bone_names": [_key(name) for name in bone_names],
            "bone_parents": parents,
            "bind_world": _matrix_to_list_map(bind_world, bone_names),
            "bind_local": _matrix_to_list_map(bind_local, bone_names),
        },
        "mesh": {
            "name": rebound_mesh.name,
            "mesh_node_name": rebound_mesh.transplant_metadata.get("source_mesh_node", rebound_mesh.name),
            "positions": np.asarray(rebound_mesh.positions, dtype=np.float64).tolist(),
            "normals": np.asarray(rebound_mesh.normals, dtype=np.float64).tolist(),
            "uvs": np.asarray(rebound_mesh.uvs, dtype=np.float64).tolist(),
            "faces": np.asarray(rebound_mesh.faces, dtype=np.int64).tolist(),
            "bone_indices": np.asarray(rebound_mesh.bone_indices, dtype=np.int64).tolist(),
            "bone_weights": np.asarray(rebound_mesh.bone_weights, dtype=np.float64).tolist(),
            "bone_name_lookup": {str(i): _key(name) for i, name in enumerate(target_names)},
        },
        "clips": [_clip_basis_payload(clip, bone_names, bind_local) for clip in baked_clips],
    }


def _combine_source_meshes(meshes: list[SourceMesh], name: str) -> SourceMesh:
    """Combine KOTOR skin nodes into one FBX mesh while preserving bone indices."""

    if not meshes:
        raise FBXExportFailure("At least one SourceMesh is required")
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        pos = np.asarray(mesh.positions, dtype=np.float64)
        positions.append(pos)
        normals.append(np.asarray(mesh.normals, dtype=np.float64))
        uvs.append(np.asarray(mesh.uvs, dtype=np.float64))
        indices.append(np.asarray(mesh.bone_indices, dtype=np.int32))
        weights.append(np.asarray(mesh.bone_weights, dtype=np.float32))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        offset += pos.shape[0]
    all_positions = np.vstack(positions)
    bbox_diag = float(np.linalg.norm(np.max(all_positions, axis=0) - np.min(all_positions, axis=0))) or 1.0
    combined = SourceMesh(
        name=name,
        positions=all_positions,
        normals=np.vstack(normals),
        uvs=np.vstack(uvs),
        bone_indices=np.vstack(indices),
        bone_weights=np.vstack(weights),
        faces=np.vstack(faces),
        source_bind_world=dict(meshes[0].source_bind_world),
        bbox_diagonal=bbox_diag,
        source_bone_names=list(meshes[0].source_bone_names),
        source_bone_index=dict(meshes[0].source_bone_index),
        local_bone_map=[],
        local_to_global_bone_indices=[],
        mesh_node_name="+".join(mesh.mesh_node_name for mesh in meshes),
        metadata={
            "combined_from": [
                {
                    "mesh_node_name": mesh.mesh_node_name,
                    "vertex_count": int(mesh.positions.shape[0]),
                    "face_count": int(mesh.faces.shape[0]),
                }
                for mesh in meshes
            ],
            "source_registry": meshes[0].metadata.get("source_registry"),
            "model_name": meshes[0].metadata.get("model_name", name),
            "supermodel": meshes[0].metadata.get("supermodel", ""),
            "mdl_path": meshes[0].metadata.get("mdl_path", ""),
        },
    )
    setattr(combined, "source_registry", getattr(meshes[0], "source_registry", None))
    return combined


def load_pmbam_combined_source_mesh() -> SourceMesh:
    """Load PMBAM's three stock skin nodes as one Day 4.5 export mesh."""

    mesh_path = DEFAULT_CORPUS_ROOT / "k1" / "pmbam.mdl"
    return _combine_source_meshes(
        [
            load_kotor_skinned_mesh(mesh_path, "Torso"),
            load_kotor_skinned_mesh(mesh_path, "LArm"),
            load_kotor_skinned_mesh(mesh_path, "RArm"),
        ],
        "PMBAM",
    )


def _matrix_last_row_is_affine(matrix: np.ndarray) -> bool:
    return np.allclose(np.asarray(matrix, dtype=np.float64)[3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8)


def validate_ground_truth(aurora_data: dict) -> list[str]:
    """Validate the Day 4.5 v6 Phase 0 skeleton contract."""

    errors: list[str] = []
    bones = aurora_data.get("bones", {})
    for name, bone in bones.items():
        matrix = bone.get("bind_world_matrix_4x4")
        if matrix is None:
            errors.append(f"Bone '{name}' missing bind_world_matrix_4x4")
            continue
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            errors.append(f"Bone '{name}' matrix not 4x4")
            continue
        if not _matrix_last_row_is_affine(arr):
            errors.append(f"Bone '{name}' matrix last row is not [0, 0, 0, 1]")
        det = float(np.linalg.det(arr[:3, :3]))
        if abs(det) <= 1e-6:
            errors.append(f"Bone '{name}' matrix is degenerate (det={det:g})")

    for name, bone in bones.items():
        parent = _key(bone.get("parent") or "")
        if parent and parent not in bones:
            errors.append(f"Bone '{name}' references missing parent '{parent}'")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            errors.append(f"Parent chain cycle at '{name}'")
            return
        visiting.add(name)
        parent = _key(bones.get(name, {}).get("parent") or "")
        if parent in bones:
            visit(parent)
        visiting.remove(name)
        visited.add(name)

    for name in bones:
        visit(name)
    return errors


def _registry_to_aurora_payload(registry: BindPoseRegistry, rename_spec: RenameSpec) -> dict[str, Any]:
    helper_keys = set(rename_spec.helper_bones_non_deform)
    bones: dict[str, Any] = {}
    for name in registry.bone_names:
        key = _key(name)
        bind_world = np.asarray(registry.world_matrix(key), dtype=np.float64)
        local = np.asarray(registry.local_matrix(key), dtype=np.float64)
        bones[key] = {
            "name": key,
            "original_name": str(name),
            "parent": _key(registry.parent_key(key)),
            "local_translation": local[:3, 3].tolist(),
            "local_rotation_quat_wxyz": matrix_to_quat_wxyz(local).tolist(),
            "bind_world_matrix_4x4": bind_world.tolist(),
            "use_deform": key not in helper_keys,
            "is_helper": key in helper_keys,
        }
    return {
        "skeleton_id": registry.skeleton_id,
        "preservation_policy": "NATIVE_AURORA_BIND_POSE",
        "bones": bones,
    }


def _mesh_vertex_weights_by_name(source_mesh: SourceMesh) -> dict[str, list[dict[str, float]]]:
    names = [str(name).lower() for name in source_mesh.source_bone_names]
    groups: dict[str, list[dict[str, float]]] = {}
    indices = np.asarray(source_mesh.bone_indices, dtype=np.int64)
    weights = np.asarray(source_mesh.bone_weights, dtype=np.float64)
    for vertex_index in range(indices.shape[0]):
        for slot in range(indices.shape[1]):
            bone_index = int(indices[vertex_index, slot])
            weight = float(weights[vertex_index, slot])
            if bone_index < 0 or bone_index >= len(names) or weight <= 1e-8:
                continue
            groups.setdefault(names[bone_index], []).append(
                {"vertex_index": int(vertex_index), "weight": weight}
            )
    return groups


def _clip_curves_payload(clip: SampledClip, registry: BindPoseRegistry) -> dict[str, Any]:
    clip_bone_index = {_key(name): idx for idx, name in enumerate(clip.bone_names)}
    curves: dict[str, dict[str, list[Any]]] = {}
    for bone_name in registry.bone_names:
        key = _key(bone_name)
        source_idx = clip_bone_index.get(key)
        if source_idx is None:
            continue
        position_keys = []
        orientation_keys = []
        for frame in range(clip.frame_count):
            blender_frame = frame + 1
            position_keys.append([blender_frame, clip.positions[frame, source_idx, :].astype(float).tolist()])
            orientation_keys.append([blender_frame, clip.rotations[frame, source_idx, :].astype(float).tolist()])
        curves[key] = {"position": position_keys, "orientation": orientation_keys}
    return {
        "name": clip.clip_name,
        "fps": float(clip.fps),
        "frame_count": int(clip.frame_count),
        "duration_seconds": float(clip.duration_s),
        "source_supermodel": clip.resolved_clip_source,
        "sampling_strategy": "fixed_30fps_delta_from_rest",
        "curves": curves,
    }


def build_intermediate_representation_day45(
    source_mesh: SourceMesh,
    sampled_clips: List[SampledClip],
    source_registry: BindPoseRegistry,
    rename_spec: RenameSpec,
    options: FBXExportOptions,
) -> Dict[str, Any]:
    """Build deterministic JSON for the v6 renamed-Aurora FBX path."""

    aurora_skeleton = _registry_to_aurora_payload(source_registry, rename_spec)
    errors = validate_ground_truth(aurora_skeleton)
    errors.extend(validate_rename_spec(rename_spec, aurora_skeleton["bones"].keys()))
    if errors:
        raise FBXExportFailure("Day 4.5 v6 intermediate validation failed: " + "; ".join(errors))

    return {
        "schema_version": "4.0.0-day4_5_v6",
        "schema": "day4_5_rename_only_kotorblender_pattern",
        "created_for": "GhostRigger Day 4.5 v6 renamed Aurora FBX export",
        "documentation": {
            "blender_fbx_operator": "https://docs.blender.org/api/current/bpy.ops.export_scene.html",
            "fbx_bind_pose": "https://help.autodesk.com/view/FBX/2020/ENU/",
            "autodesk_axis_system": "https://help.autodesk.com/cloudhelp/2020/ENU/FBX-API-Reference/cpp_ref/class_fbx_axis_system.html",
        },
        "metadata": {
            "source_model": options.source_model_id,
            "source_supermodel": options.source_supermodel,
            "preservation_policy": "NATIVE_AURORA_BIND_POSE",
            "implementation_pattern": "bone.matrix = bind_world_matrix_4x4; action curves are delta-from-rest",
        },
        "options": _options_to_dict(options),
        "aurora_skeleton": aurora_skeleton,
        "rename_spec": rename_spec.as_payload(),
        "mesh": {
            "name": source_mesh.name,
            "mesh_node_name": source_mesh.mesh_node_name,
            "positions": np.asarray(source_mesh.positions, dtype=np.float64).tolist(),
            "normals": np.asarray(source_mesh.normals, dtype=np.float64).tolist(),
            "uvs": np.asarray(source_mesh.uvs, dtype=np.float64).tolist(),
            "faces": np.asarray(source_mesh.faces, dtype=np.int64).tolist(),
            "vertex_weights": _mesh_vertex_weights_by_name(source_mesh),
        },
        "animation_clips": [_clip_curves_payload(clip, source_registry) for clip in sampled_clips],
    }


def _options_to_dict(options: FBXExportOptions) -> Dict[str, Any]:
    payload = asdict(options)
    payload["output_path"] = str(options.output_path)
    payload["blender_executable"] = str(options.blender_executable) if options.blender_executable else None
    return payload


def _run_blender(
    args: list[str],
    *,
    timeout: int = 300,
    log_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    # stdin=DEVNULL: under the KotorMCP stdio server a worker thread holds a
    # blocking read on the inherited protocol pipe, and a Blender child that
    # inherits that handle stalls until the timeout on Windows. Children must
    # never share the MCP protocol stdin.
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    target_log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    target_log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    log_path = target_log_dir / f"blender_day4_{stamp}.log"
    log_path.write_text(
        "COMMAND:\n" + " ".join(args) + "\n\nSTDOUT:\n" + (result.stdout or "") + "\n\nSTDERR:\n" + (result.stderr or ""),
        encoding="utf-8",
    )
    return result


def _fbx_axis_system(path: Path) -> Dict[str, str]:
    """Read FBX GlobalSettings axis metadata from Blender's binary FBX.

    Autodesk's FBX SDK exposes axis systems with up/front/sign properties.
    Blender writes the same metadata under ``GlobalSettings/Properties70``.
    """

    axis_names = {0: "X", 1: "Y", 2: "Z"}
    values: Dict[str, int] = {}
    try:
        roots = _read_binary_fbx(path)
    except Exception:
        return {}
    global_settings = next((node for node in roots if node.name == "GlobalSettings"), None)
    props = global_settings.child("Properties70") if global_settings is not None else None
    if props is None:
        return {}
    for prop in props.children_named("P"):
        if len(prop.props) >= 5 and prop.props[0] in {
            "UpAxis",
            "UpAxisSign",
            "FrontAxis",
            "FrontAxisSign",
            "CoordAxis",
            "CoordAxisSign",
        }:
            values[str(prop.props[0])] = int(prop.props[4])
    up = axis_names.get(values.get("UpAxis", -1), "")
    front = axis_names.get(values.get("FrontAxis", -1), "")
    if up and values.get("UpAxisSign", 1) < 0:
        up = "-" + up

    # Blender's FBX writer follows Autodesk's axis-system metadata, where the
    # coordinate axis sign participates in handedness.  For the MayaZUp/Max
    # style system documented by Autodesk (+Z up, -Y forward, right-handed),
    # Blender 4.2 writes FrontAxis=Y, FrontAxisSign=+1, CoordAxis=X,
    # CoordAxisSign=-1.  Interpret that combination as the original -Y forward
    # exporter declaration.
    if (
        values.get("UpAxis") == 2
        and values.get("UpAxisSign", 1) == 1
        and values.get("FrontAxis") == 1
        and values.get("FrontAxisSign", 1) == 1
        and values.get("CoordAxis") == 0
        and values.get("CoordAxisSign") == -1
    ):
        front = "-Y"
    elif front and values.get("FrontAxisSign", 1) < 0:
        front = "-" + front
    return {"axis_up": up.lstrip("-"), "axis_forward": front}


def roundtrip_validate(
    fbx_path: Path,
    intermediate: Dict[str, Any],
    *,
    blender_executable: str | Path | None = None,
    log_dir: Path | None = None,
) -> Dict[str, Any]:
    """Re-import the FBX through Blender and compare structural counts."""

    exe = find_blender_executable(blender_executable)
    validation_path = Path(fbx_path).with_suffix(".roundtrip.json")
    cmd = [
        str(exe),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--validate",
        str(fbx_path),
        "--validation-output",
        str(validation_path),
    ]
    result = _run_blender(cmd, timeout=300, log_dir=log_dir)
    if result.returncode != 0:
        raise FBXExportFailure(f"Blender roundtrip validation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not validation_path.exists():
        raise FBXExportFailure(f"Blender validation did not write {validation_path}")

    observed = json.loads(validation_path.read_text(encoding="utf-8"))
    if "aurora_skeleton" in intermediate:
        expected_bones = (
            len(intermediate["aurora_skeleton"]["bones"])
            + len(intermediate.get("rename_spec", {}).get("twist_leaves", []))
            + len(intermediate.get("rename_spec", {}).get("helper_leaves", []))
        )
    else:
        expected_bones = len(intermediate["skeleton"]["bone_names"])
    expected_vertices = len(intermediate["mesh"]["positions"])
    clip_payloads = intermediate.get("animation_clips") or intermediate.get("clips") or []
    expected_frames = int(clip_payloads[0]["frame_count"]) if clip_payloads else 0
    expected_axis = {
        "axis_up": intermediate["options"]["axis_up"],
        "axis_forward": intermediate["options"]["axis_forward"],
    }
    max_frame0 = 0.0
    if intermediate.get("clips") and observed.get("frame0_rotations"):
        expected_rot = np.asarray(intermediate["clips"][0]["rotations"][0], dtype=np.float64)
        observed_by_name = observed["frame0_rotations"]
        for idx, bone_name in enumerate(intermediate["clips"][0]["bone_names"]):
            if bone_name not in observed_by_name:
                continue
            delta = max_quat_component_delta(expected_rot[idx], np.asarray(observed_by_name[bone_name], dtype=np.float64))
            max_frame0 = max(max_frame0, float(delta))

    observed_axis = _fbx_axis_system(Path(fbx_path))
    metrics = {
        "validation_path": str(validation_path),
        "bone_count_expected": expected_bones,
        "bone_count_observed": int(observed.get("bone_count", -1)),
        "bone_count_match": int(observed.get("bone_count", -1)) == expected_bones,
        "vertex_count_expected": expected_vertices,
        "vertex_count_observed": int(observed.get("vertex_count", -1)),
        "vertex_count_match": int(observed.get("vertex_count", -1)) == expected_vertices,
        "frame_count_expected": expected_frames,
        "frame_count_observed": int(observed.get("frame_count", -1)),
        "frame_count_match": int(observed.get("frame_count", -1)) == expected_frames,
        "leaf_bones": list(observed.get("leaf_bones", [])),
        "no_leaf_bones_added": len(observed.get("leaf_bones", [])) == 0,
        "axis_system_expected": expected_axis,
        "axis_system_observed": observed_axis,
        "axis_system_match": observed_axis == expected_axis,
        "frame0_rotation_max_delta": max_frame0,
    }
    failures = [
        key for key in (
            "bone_count_match",
            "vertex_count_match",
            "frame_count_match",
            "no_leaf_bones_added",
            "axis_system_match",
        )
        if not metrics[key]
    ]
    if failures:
        raise FBXExportFailure(f"FBX roundtrip validation failed gates: {failures}; metrics={metrics}")
    if max_frame0 > 1e-4:
        raise FBXExportFailure(f"Frame 0 rotation roundtrip delta {max_frame0:g} exceeds 1e-4")
    return metrics


def build_manifest(
    rebound_mesh: ReboundMesh,
    baked_clips: List[SampledClip],
    aligned_skeleton: AlignedSkeleton,
    options: FBXExportOptions,
    *,
    fbx_sha256: str,
    blender_version_text: str,
    roundtrip_metrics: Dict[str, Any],
    intermediate_path: Path,
) -> FBXExportManifest:
    clip_entries = [
        ClipManifestEntry(
            clip_name=clip.clip_name,
            source_supermodel=clip.resolved_clip_source,
            frame_count=clip.frame_count,
            fps=clip.fps,
            duration_seconds=clip.duration_s,
        )
        for clip in baked_clips
    ]
    alignment = rebound_mesh.transplant_metadata.get("skeleton_prealignment", {}).get("summary") or {}
    return FBXExportManifest(
        fbx_path=options.output_path,
        fbx_version=options.fbx_version,
        fbx_sha256=fbx_sha256,
        blender_version=blender_version_text,
        export_timestamp=datetime.now(timezone.utc).isoformat(),
        source_mesh_resref=options.source_model_id,
        source_skeleton_id=aligned_skeleton.source_id,
        aligned_skeleton_id=aligned_skeleton.skeleton_id,
        bone_map_version=hashlib.sha256(json.dumps(rebound_mesh.transplant_metadata.get("index_remap", {}), sort_keys=True).encode("utf-8")).hexdigest(),
        clip_inventory=clip_entries,
        bind_pose_validation={
            "alignment_validation_max_drift": float(alignment.get("validation_max_drift", 0.0) or 0.0),
            "weight_conservation_max_drift": float(rebound_mesh.transplant_metadata.get("weight_conservation_max_drift", 0.0) or 0.0),
            "normal_unit_max_drift": float(rebound_mesh.transplant_metadata.get("normal_unit_max_drift", 0.0) or 0.0),
        },
        axis_system={
            "axis_up": options.axis_up,
            "axis_forward": options.axis_forward,
            "autodesk_predefined": "MayaZUp/Max: +Z up, -Y forward, right-handed",
        },
        roundtrip_metrics=roundtrip_metrics,
        intermediate_path=intermediate_path,
        manifest_path=options.output_path.with_suffix(".manifest.json"),
    )


def export_to_fbx(
    rebound_mesh: ReboundMesh,
    baked_clips: List[SampledClip],
    aligned_skeleton: AlignedSkeleton,
    options: FBXExportOptions,
    *,
    log_dir: Path | None = None,
) -> FBXExportManifest:
    """Export a rebound mesh, aligned skeleton, and baked clips to one FBX."""

    if rebound_mesh.aligned_skeleton is None and aligned_skeleton is None:
        raise FBXExportFailure("A Day 4 FBX export requires an aligned skeleton")
    if not baked_clips:
        raise FBXExportFailure("At least one baked clip is required")
    if not BLENDER_SCRIPT.exists():
        raise FBXExportFailure(f"Blender export script not found: {BLENDER_SCRIPT}")

    blender_exe = find_blender_executable(options.blender_executable)
    version = blender_version(blender_exe)
    output_path = Path(options.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f"{output_path.stem}_intermediate.json")

    intermediate = build_intermediate_representation(rebound_mesh, baked_clips, aligned_skeleton, options)
    write_json_deterministic(intermediate_path, intermediate)

    cmd = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--intermediate",
        str(intermediate_path),
        "--output",
        str(output_path),
        "--options",
        json.dumps(_options_to_dict(options), sort_keys=True),
    ]
    result = _run_blender(cmd, timeout=300, log_dir=log_dir)
    if result.returncode != 0:
        raise FBXExportFailure(f"Blender export failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not output_path.exists():
        raise FBXExportFailure(f"Blender claimed success but no FBX was written at {output_path}")

    fbx_sha = compute_sha256(output_path)
    roundtrip_metrics = (
        roundtrip_validate(output_path, intermediate, blender_executable=blender_exe, log_dir=log_dir)
        if options.run_roundtrip_validation
        else {}
    )
    manifest = build_manifest(
        rebound_mesh,
        baked_clips,
        aligned_skeleton,
        options,
        fbx_sha256=fbx_sha,
        blender_version_text=version,
        roundtrip_metrics=roundtrip_metrics,
        intermediate_path=intermediate_path,
    )
    write_json_deterministic(manifest.manifest_path, manifest.to_json_dict())
    return manifest


def build_day4_pmbam_g1a1_asset(
    output_path: str | Path | None = None,
    *,
    fps: float = 60.0,
    run_roundtrip_validation: bool = True,
) -> tuple[ReboundMesh, list[SampledClip], AlignedSkeleton, FBXExportOptions]:
    """Build the locked Day 4 pmbam + g1a1 + aligned Quinn export inputs."""

    normalizer = CoordinateNormalizer()
    source_model = load_fixture_model("pmbam")
    target_model = unreal_skeleton_model(load_quinn_skeleton_asset())
    source_registry = normalizer.normalize_aurora_bind(source_model, "kotor_pmbam")
    raw_target_registry = normalizer.normalize_ue5_bind(target_model, "ue5_quinn")
    report = build_bone_map(source_model, target_model)
    mesh_path = DEFAULT_CORPUS_ROOT / "k1" / "pmbam.mdl"
    mesh = load_kotor_skinned_mesh(mesh_path)
    rebound = rebind_mesh_to_target_skeleton(
        mesh,
        "kotor_pmbam",
        "ue5_quinn",
        report.mapping,
        raw_target_registry,
        normalizer,
        RebindOptions(),
    )
    if rebound.aligned_skeleton is None:
        raise FBXExportFailure("Rebind did not produce an aligned skeleton")

    aligned_registry = aligned_skeleton_to_registry(rebound.aligned_skeleton)
    sampled = sample_clip_to_fixed_rate("pmbam", "g1a1", fps=fps)
    offsets = compute_bind_offsets(source_registry, aligned_registry, report.mapping)
    baked = bake_retargeted_clip(sampled, source_registry, aligned_registry, report.mapping, offsets)
    out = Path(output_path) if output_path else DEFAULT_EXPORT_DIR / "pmbam__g1a1__to__quinn_aligned.fbx"
    options = FBXExportOptions(
        source_model_id="pmbam",
        source_supermodel=str(getattr(source_model, "supermodel", "") or ""),
        target_skeleton_id=rebound.target_skeleton_id,
        clip_names=["g1a1"],
        output_path=out,
        run_roundtrip_validation=run_roundtrip_validation,
    )
    return rebound, [baked], rebound.aligned_skeleton, options


def export_day4_pmbam_g1a1(
    output_path: str | Path | None = None,
    *,
    fps: float = 60.0,
    run_roundtrip_validation: bool = True,
    log_dir: str | Path | None = None,
) -> FBXExportManifest:
    rebound, baked, aligned, options = build_day4_pmbam_g1a1_asset(
        output_path,
        fps=fps,
        run_roundtrip_validation=run_roundtrip_validation,
    )
    return export_to_fbx(
        rebound,
        baked,
        aligned,
        options,
        log_dir=Path(log_dir) if log_dir is not None else None,
    )


def build_day45_pmbam_g1a1_asset(
    output_path: str | Path | None = None,
    *,
    fps: float = 30.0,
    run_roundtrip_validation: bool = True,
) -> tuple[SourceMesh, list[SampledClip], BindPoseRegistry, RenameSpec, FBXExportOptions]:
    """Build the Day 4.5 v6 pmbam + g1a1 renamed-Aurora Unity export inputs."""

    normalizer = CoordinateNormalizer()
    source_model = load_fixture_model("pmbam")
    source_registry = normalizer.normalize_aurora_bind(source_model, "kotor_pmbam")
    rename_spec = load_rename_spec()
    rename_errors = validate_rename_spec(rename_spec, source_registry.bone_names)
    if rename_errors:
        raise FBXExportFailure("Day 4.5 v6 rename spec invalid: " + "; ".join(rename_errors))
    mesh = load_pmbam_combined_source_mesh()
    sampled = sample_clip_to_fixed_rate("pmbam", "g1a1", fps=fps)
    out = Path(output_path) if output_path else DEFAULT_DAY45_EXPORT_DIR / "pmbam__g1a1__day4_5_v6.fbx"
    options = FBXExportOptions(
        source_model_id="pmbam",
        source_supermodel=str(getattr(source_model, "supermodel", "") or ""),
        target_skeleton_id="kotor_pmbam_renamed_ue5_native_pose",
        clip_names=["g1a1"],
        output_path=out,
        use_armature_deform_only=False,
        run_roundtrip_validation=run_roundtrip_validation,
    )
    return mesh, [sampled], source_registry, rename_spec, options


def export_day45_pmbam_g1a1(
    output_path: str | Path | None = None,
    *,
    fps: float = 30.0,
    run_roundtrip_validation: bool = True,
) -> FBXExportManifest:
    """Export Day 4.5 v6's self-contained renamed-Aurora FBX."""

    mesh, sampled_clips, source_registry, rename_spec, options = build_day45_pmbam_g1a1_asset(
        output_path,
        fps=fps,
        run_roundtrip_validation=run_roundtrip_validation,
    )
    blender_exe = find_blender_executable(options.blender_executable)
    version = blender_version(blender_exe)
    output_path = Path(options.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f"{output_path.stem}_intermediate.json")
    intermediate = build_intermediate_representation_day45(mesh, sampled_clips, source_registry, rename_spec, options)
    write_json_deterministic(intermediate_path, intermediate)

    cmd = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--intermediate",
        str(intermediate_path),
        "--output",
        str(output_path),
        "--options",
        json.dumps(_options_to_dict(options), sort_keys=True),
    ]
    result = _run_blender(cmd, timeout=300)
    if result.returncode != 0:
        raise FBXExportFailure(f"Blender Day 4.5 export failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    if not output_path.exists():
        raise FBXExportFailure(f"Blender claimed success but no FBX was written at {output_path}")

    fbx_sha = compute_sha256(output_path)
    roundtrip_metrics = (
        roundtrip_validate(output_path, intermediate, blender_executable=blender_exe)
        if options.run_roundtrip_validation
        else {}
    )
    manifest = FBXExportManifest(
        fbx_path=output_path,
        fbx_version=options.fbx_version,
        fbx_sha256=fbx_sha,
        blender_version=version,
        export_timestamp=datetime.now(timezone.utc).isoformat(),
        source_mesh_resref=options.source_model_id,
        source_skeleton_id="kotor_pmbam",
        aligned_skeleton_id=options.target_skeleton_id,
        bone_map_version=hashlib.sha256(
            json.dumps(rename_spec.rename_pairs, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        clip_inventory=[
            ClipManifestEntry(
                clip_name=clip.clip_name,
                source_supermodel=clip.resolved_clip_source,
                frame_count=clip.frame_count,
                fps=clip.fps,
                duration_seconds=clip.duration_s,
                sampling_strategy="fixed_30fps_delta_from_rest",
            )
            for clip in sampled_clips
        ],
        bind_pose_validation={
            "aurora_bone_count": float(len(source_registry.bone_names)),
            "twist_leaf_count": float(len(rename_spec.twist_leaves)),
            "helper_leaf_count": float(len(rename_spec.helper_leaves)),
            "weight_conservation_max_drift": float(
                np.max(np.abs(np.sum(mesh.bone_weights.astype(np.float64), axis=1) - 1.0))
            ),
        },
        axis_system={
            "axis_up": options.axis_up,
            "axis_forward": options.axis_forward,
            "autodesk_predefined": "MayaZUp/Max: +Z up, -Y forward, right-handed",
        },
        roundtrip_metrics=roundtrip_metrics,
        intermediate_path=intermediate_path,
        manifest_path=output_path.with_suffix(".manifest.json"),
        schema_version="4.0.0-day4_5_v6",
    )
    write_json_deterministic(manifest.manifest_path, manifest.to_json_dict())
    return manifest
