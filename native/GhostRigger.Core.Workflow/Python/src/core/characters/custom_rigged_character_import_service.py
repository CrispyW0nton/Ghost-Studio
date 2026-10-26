"""Foreign FBX import and self-contained Odyssey model assembly.

The service keeps the imported deform hierarchy authoritative.  It never calls
the native Character Builder template-rig path and never substitutes stock
humanoid node names.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.core.project.custom_rigged_character_project import (
    AnimationMapping,
    CustomRiggedCharacterProject,
    MaterialAssignment,
    SourceAsset,
    sha256_file,
)
from src.core.retargeting.fbx_importer import import_ue_fbx_animation_clip
from src.core.retargeting.source_animation import SourceSkeletonClip, Transform
from src.core.validation.custom_rigged_character_validator import (
    AnimationClipSnapshot,
    AnimationTrackSnapshot,
    CustomRiggedCharacterSnapshot,
    MaterialSnapshot,
    RigNodeSnapshot,
    normalized_influences,
    quaternion_continuity,
)
from src.converters.blender_fbx_mesh_importer import import_fbx_mesh_with_blender

from .custom_rigged_character_build_service import suggest_semantic_mapping


CUSTOM_SKIN_PALETTE_LIMIT = 12
_ANIMATION_POSITION_DELTA_EPSILON = 1.0e-5
_RUNTIME_HEIGHT_NODE_KEY = "rootjoint"
_BLENDER_TO_KOTOR = np.asarray(
    ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    dtype=np.float64,
)
_KOTOR_TO_BLENDER = np.linalg.inv(_BLENDER_TO_KOTOR)


@dataclass
class ImportedSkeleton:
    armature_name: str
    root_names: list[str]
    nodes: list[RigNodeSnapshot]
    local_transforms: dict[str, Transform]
    global_transforms: dict[str, Transform]
    fingerprint: str
    animation_fingerprint: str
    axis_conversion: str = "blender_xyz_to_kotor_xz_minus_y"
    available_root_choices: list[str] = field(default_factory=list)
    selection_required: bool = False


@dataclass
class CustomRiggedCharacterImportResult:
    source_model: Any
    skeleton: ImportedSkeleton
    snapshot: CustomRiggedCharacterSnapshot
    action_inventory: list[dict[str, Any]] = field(default_factory=list)
    clips: dict[str, SourceSkeletonClip] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    project: CustomRiggedCharacterProject | None = None


def detect_runtime_height_offset(skeleton: ImportedSkeleton) -> tuple[float, str]:
    """Recover the engine-height correction proved by the Borhek runtime.

    Arbitrary source rigs commonly keep the creature's native height on a
    ``root_joint`` immediately below their authoring root.  Odyssey omits that
    height while evaluating a creature, so the self-contained model must mirror
    it on ``heightdummy`` in the base hierarchy.  Every animation keeps the same
    helper node/edge but intentionally omits a position controller: Odyssey
    position animation keys are deltas added to the base transform, so repeating
    the height there would lift the creature twice while any clip is playing.
    The value is taken from the converted, scaled local bind transform; source
    files are never modified.
    """

    for node in skeleton.nodes:
        key = "".join(character for character in str(node.name).casefold() if character.isalnum())
        if key != _RUNTIME_HEIGHT_NODE_KEY:
            continue
        transform = skeleton.local_transforms.get(node.name)
        if transform is None:
            continue
        value = float(transform.position[2])
        if math.isfinite(value) and value > 1.0e-6:
            return value, str(node.name)
    return 0.0, ""


def _basis_for_axis_conversion(mode: str) -> np.ndarray:
    return np.eye(4, dtype=np.float64) if str(mode) == "identity_z_up" else _BLENDER_TO_KOTOR


def _kotor_matrix(
    matrix: Sequence[Sequence[float]],
    basis: np.ndarray = _BLENDER_TO_KOTOR,
) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64)
    if source.shape != (4, 4):
        raise ValueError(f"Expected 4x4 FBX matrix, got {source.shape}.")
    inverse = np.linalg.inv(basis)
    return basis @ source @ inverse


def _kotor_transform(
    transform: Transform,
    *,
    scale: float = 1.0,
    basis: np.ndarray = _BLENDER_TO_KOTOR,
) -> Transform:
    converted = Transform.from_matrix(_kotor_matrix(transform.to_matrix(), basis))
    return Transform(
        position=tuple(float(value) * float(scale) for value in converted.position),
        rotation=converted.rotation,
        scale=converted.scale,
    )


def _matrix_transform(
    matrix: Sequence[Sequence[float]],
    *,
    scale: float = 1.0,
    basis: np.ndarray = _BLENDER_TO_KOTOR,
) -> Transform:
    converted = Transform.from_matrix(_kotor_matrix(matrix, basis))
    return Transform(
        position=tuple(float(value) * float(scale) for value in converted.position),
        rotation=converted.rotation,
        scale=converted.scale,
    )


def _skeleton_fingerprint(nodes: Iterable[RigNodeSnapshot]) -> str:
    rows = [
        f"{node.name}|{node.parent}|{','.join(f'{value:.9g}' for value in node.bind_matrix)}"
        for node in nodes
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _animation_rig_fingerprint(
    rows: Iterable[tuple[str, str, Transform]],
) -> str:
    values = []
    for name, parent, transform in rows:
        matrix = np.asarray(transform.to_matrix(), dtype=np.float64).reshape(16)
        values.append(
            f"{str(name or '')}|{str(parent or '')}|"
            + ",".join(f"{float(value):.7g}" for value in matrix)
        )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def imported_skeleton_from_model(
    model: Any,
    *,
    scale: float = 1.0,
    selected_root: str = "",
) -> ImportedSkeleton:
    armatures = list(getattr(model, "_gr_fbx_armature_objects", []) or [])
    if not armatures:
        raise ValueError("The FBX contains no armature hierarchy.")
    choices: list[tuple[Mapping[str, Any], str, str]] = []
    for candidate in armatures:
        candidate_bones = [row for row in candidate.get("bones") or () if isinstance(row, Mapping)]
        candidate_names = {str(row.get("name") or "") for row in candidate_bones}
        roots = [
            str(row.get("name") or "")
            for row in candidate_bones
            if str(row.get("name") or "")
            and str(row.get("parent") or "") not in candidate_names
        ]
        for root_name in roots:
            label = f"{str(candidate.get('name') or 'Armature')} :: {root_name}"
            choices.append((candidate, root_name, label))
    if not choices:
        raise ValueError("The FBX armature contains no coherent skeleton root.")
    requested = str(selected_root or "").strip()
    matched = [
        row for row in choices
        if requested and (requested.casefold() == row[1].casefold() or requested.casefold() == row[2].casefold())
    ]
    if requested and len(matched) != 1:
        raise ValueError(
            f"Selected skeleton root '{requested}' is missing or ambiguous. Available choices: "
            + ", ".join(row[2] for row in choices)
        )
    selected = matched[0] if matched else choices[0]
    armature, chosen_root, _chosen_label = selected
    selection_required = not requested and len(choices) > 1
    axis_conversion = str(
        getattr(model, "_gr_fbx_axis_conversion", "blender_xyz_to_kotor_xz_minus_y")
    )
    axis_basis = _basis_for_axis_conversion(axis_conversion)
    all_bones = [row for row in armature.get("bones") or () if isinstance(row, Mapping)]
    children: dict[str, list[str]] = {}
    by_name = {str(row.get("name") or ""): row for row in all_bones}
    for row in all_bones:
        children.setdefault(str(row.get("parent") or ""), []).append(str(row.get("name") or ""))
    retained: set[str] = set()
    pending = [chosen_root]
    while pending:
        name = pending.pop()
        if name in retained:
            continue
        retained.add(name)
        pending.extend(children.get(name, ()))
    bones = [row for row in all_bones if str(row.get("name") or "") in retained]
    if not bones:
        raise ValueError("The selected FBX armature contains no bones.")
    global_matrices: dict[str, np.ndarray] = {}
    parent_by_name: dict[str, str] = {}
    use_deform: dict[str, bool] = {}
    for row in bones:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        matrix = row.get("matrix_world")
        if matrix is None:
            raise ValueError(f"FBX bone '{name}' has no rest matrix.")
        global_matrices[name] = _kotor_matrix(matrix, axis_basis)
        parent_by_name[name] = str(row.get("parent") or "")
        use_deform[name] = bool(row.get("use_deform", True))
    local: dict[str, Transform] = {}
    global_transforms: dict[str, Transform] = {}
    snapshots: list[RigNodeSnapshot] = []
    for name in global_matrices:
        parent_name = parent_by_name.get(name, "")
        world = global_matrices[name]
        parent_world = global_matrices.get(parent_name)
        local_matrix = np.linalg.inv(parent_world) @ world if parent_world is not None else world
        local_transform = Transform.from_matrix(local_matrix)
        world_transform = Transform.from_matrix(world)
        local[name] = Transform(
            position=tuple(value * scale for value in local_transform.position),
            rotation=local_transform.rotation,
            scale=local_transform.scale,
        )
        global_transforms[name] = Transform(
            position=tuple(value * scale for value in world_transform.position),
            rotation=world_transform.rotation,
            scale=world_transform.scale,
        )
        flat_world = tuple(float(value) for value in world.reshape(16))
        snapshots.append(RigNodeSnapshot(
            name=name,
            parent=parent_name,
            exported=True,
            deform=use_deform.get(name, True),
            kind="bone" if use_deform.get(name, True) else "control",
            transform=tuple(float(value) for value in local_matrix.reshape(16)),
            scale=tuple(float(value) for value in local_transform.scale),
            bind_matrix=flat_world,
            expected_bind_matrix=flat_world,
        ))
    roots = [node.name for node in snapshots if not node.parent]
    animation_fingerprint = _animation_rig_fingerprint(
        (node.name, node.parent, local[node.name]) for node in snapshots
    )
    return ImportedSkeleton(
        armature_name=str(armature.get("name") or "Armature"),
        root_names=roots,
        nodes=snapshots,
        local_transforms=local,
        global_transforms=global_transforms,
        fingerprint=_skeleton_fingerprint(snapshots),
        animation_fingerprint=animation_fingerprint,
        axis_conversion=axis_conversion,
        available_root_choices=[row[2] for row in choices],
        selection_required=selection_required,
    )


def _mesh_nodes(model: Any) -> list[Any]:
    return [
        node for node in (model.all_nodes() if hasattr(model, "all_nodes") else [])
        if bool(getattr(node, "is_mesh", False) or getattr(node, "is_skin", False))
        and list(getattr(node, "vertices", []) or [])
    ]


def _authored_vertex_count(node: Any) -> int:
    vertices = list(getattr(node, "vertices", []) or [])
    source_indices = list(getattr(node, "_gr_source_vertex_indices", []) or [])
    if len(source_indices) == len(vertices):
        return len({int(value) for value in source_indices})
    return len(vertices)


def _vertex_influences(model: Any) -> list[list[tuple[str, float]]]:
    result: list[list[tuple[str, float]]] = []
    for node in _mesh_nodes(model):
        palette = [str(value) for value in list(getattr(node, "bone_map", []) or [])]
        skins = list(getattr(node, "skin_data", []) or [])
        for index in range(len(list(getattr(node, "vertices", []) or []))):
            skin = skins[index] if index < len(skins) else None
            row: list[tuple[str, float]] = []
            for influence in list(getattr(skin, "influences", []) or []):
                bone_index = int(getattr(influence, "bone_index", -1))
                if 0 <= bone_index < len(palette):
                    row.append((palette[bone_index], float(getattr(influence, "weight", 0.0))))
            result.append(row)
    return result


def _resolve_material_texture(
    project: CustomRiggedCharacterProject,
    model: Any,
    node: Any,
) -> Path:
    embedded = Path(str(getattr(node, "_gr_source_texture", "") or ""))
    if embedded.is_file():
        return embedded
    names = [
        Path(str(getattr(node, "texture", "") or "").replace("\\", "/")).stem,
        str(getattr(node, "name", "") or ""),
    ]
    folders: list[Path] = []
    if project.texture_folder:
        folders.append(project.resolve_path(project.texture_folder))
    source_text = str(getattr(model, "metadata", {}).get("external_import", {}).get("source_path", "") or "")
    source_path = Path(source_text)
    if source_text:
        folders.extend((source_path.parent, source_path.parent / f"{source_path.stem}.fbm"))
    extensions = (".tga", ".tpc", ".png", ".dds", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")
    for folder in folders:
        if not folder.is_dir():
            continue
        lookup = {path.name.casefold(): path for path in folder.iterdir() if path.is_file()}
        for name in names:
            for extension in extensions:
                candidate = lookup.get(f"{name}{extension}".casefold())
                if candidate is not None:
                    return candidate
    return Path()


def _material_snapshots(
    model: Any,
    project: CustomRiggedCharacterProject,
) -> list[MaterialSnapshot]:
    result: list[MaterialSnapshot] = []
    for node in _mesh_nodes(model):
        name = str(getattr(node, "name", "mesh") or "mesh")
        raw_texture = str(getattr(node, "texture", "") or "")
        resref = Path(raw_texture.replace("\\", "/")).stem.lower()[:16]
        source = _resolve_material_texture(project, model, node)
        width = height = 0
        has_alpha = False
        source_format = source.suffix.lower().lstrip(".") if source.is_file() else Path(raw_texture).suffix.lower().lstrip(".")
        if source.is_file():
            try:
                if source.suffix.casefold() == ".tpc":
                    header = source.read_bytes()[:16]
                    if len(header) >= 12:
                        width = int.from_bytes(header[8:10], "little")
                        height = int.from_bytes(header[10:12], "little")
                else:
                    from PIL import Image

                    with Image.open(source) as image:
                        width, height = image.size
                        has_alpha = "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255
            except Exception:
                pass
        assignment = next(
            (value for value in project.material_assignments if value.material_name == name),
            None,
        )
        result.append(MaterialSnapshot(
            material_name=name,
            texture_resref=resref,
            source_format=source_format,
            texture_size=(width, height),
            uvs=tuple(tuple(float(value) for value in uv[:2]) for uv in list(getattr(node, "uvs", []) or [])),
            wrap_mode=assignment.wrap_mode if assignment else "repeat",
            has_alpha=has_alpha,
            alpha_mode=assignment.alpha_mode if assignment else ("blend" if has_alpha else "opaque"),
            source_texture=str(source) if source.is_file() else "",
        ))
    return result


def _clip_snapshot(
    clip: SourceSkeletonClip,
    axis_conversion: str,
    *,
    source_key: str = "",
    loop: bool = False,
) -> AnimationClipSnapshot:
    basis = _basis_for_axis_conversion(axis_conversion)
    tracks: list[AnimationTrackSnapshot] = []
    for node in clip.nodes:
        positions = []
        rotations = []
        scales = []
        for pose in clip.sampled_poses:
            transform = pose.local_transforms.get(node.name)
            if transform is None:
                continue
            converted = _kotor_transform(transform, basis=basis)
            positions.append(converted.position)
            rotations.append(converted.rotation)
            if any(abs(value - 1.0) > 1.0e-5 for value in converted.scale):
                scales.append(converted.scale)
        tracks.append(AnimationTrackSnapshot(
            node_name=node.name,
            positions=tuple(positions),
            rotations=tuple(rotations),
            scales=tuple(scales),
        ))
    root_name = next((node.name for node in clip.nodes if not node.parent_name), "")
    root_positions = tuple(
        _kotor_transform(pose.local_transforms[root_name], basis=basis).position
        for pose in clip.sampled_poses
        if root_name in pose.local_transforms
    )
    clip_fingerprint = _animation_rig_fingerprint(
        (
            node.name,
            node.parent_name,
            _kotor_transform(node.rest_local, basis=basis),
        )
        for node in clip.nodes
    )
    return AnimationClipSnapshot(
        name=source_key or clip.clip_name,
        duration=clip.duration_seconds,
        tracks=tuple(tracks),
        loop=bool(loop),
        root_positions=root_positions,
        source_skeleton_fingerprint=clip_fingerprint,
    )


class CustomRiggedCharacterImportService:
    """Import source files through the stable Blender mesh/animation bridges."""

    def import_project(
        self,
        project: CustomRiggedCharacterProject,
        *,
        sample_animations: bool = True,
        sample_rate: float = 30.0,
    ) -> CustomRiggedCharacterImportResult:
        source = project.resolve_path(project.primary_fbx.path)
        if not source.is_file():
            raise FileNotFoundError(f"Source FBX not found: {source}")
        source_hash = sha256_file(source)
        if project.primary_fbx.sha256 and project.primary_fbx.sha256 != source_hash:
            raise ValueError("Source FBX bytes changed since this project recorded its hash.")
        project.primary_fbx.sha256 = source_hash
        game = "K2" if project.target_game == "K2" else "K1"
        try:
            from src.core.geometry.model_data import GameVersion
        except ImportError:  # pragma: no cover
            from core.geometry.model_data import GameVersion  # type: ignore
        mesh_model = import_fbx_mesh_with_blender(
            source,
            model_name=project.resource_name or source.stem,
            game_version=GameVersion.K2 if game == "K2" else GameVersion.K1,
            supermodel="NULL",
            classification="character",
            axis_conversion=(
                "identity_z_up"
                if str(project.import_coordinate_system.get("source_up") or "auto").upper() in {"+Z", "Z"}
                else "blender_xyz_to_kotor_xz_minus_y"
                if str(project.import_coordinate_system.get("source_up") or "auto").upper() in {"+Y", "Y", "-Y"}
                else "auto"
            ),
        )
        skeleton = imported_skeleton_from_model(
            mesh_model,
            scale=project.global_scale,
            selected_root=project.selected_skeleton_root,
        )
        secondary_identity = dict(project.secondary_motion.get("model_identity") or {})
        secondary_identity.update({
            "model_resrefs": [str(project.resource_name or source.stem).strip().lower()],
            "hierarchy_fingerprint": skeleton.fingerprint,
            "root_name": skeleton.root_names[0] if skeleton.root_names else "",
            "target_kind": "custom_character",
        })
        project.secondary_motion["model_identity"] = secondary_identity
        runtime_state = dict(project.secondary_motion.get("runtime") or {})
        runtime_state.update({"state": "capture_only", "write_enabled": False})
        project.secondary_motion["runtime"] = runtime_state
        runtime_height_offset, runtime_height_source = detect_runtime_height_offset(skeleton)
        project.runtime_height_offset = runtime_height_offset
        project.runtime_height_source = runtime_height_source
        if not project.selected_skeleton_root and len(skeleton.available_root_choices) == 1:
            project.selected_skeleton_root = skeleton.root_names[0]
        if not project.export_nodes:
            project.export_nodes = {node.name: True for node in skeleton.nodes}

        primary_inventory = [dict(value) for value in list(getattr(mesh_model, "_gr_fbx_actions", []) or [])]
        action_inventory: list[dict[str, Any]] = []
        clips: dict[str, SourceSkeletonClip] = {}
        warnings: list[str] = []
        sources: list[SourceAsset] = [project.primary_fbx, *project.external_animation_assets]
        used_keys: set[str] = set()
        for source_index, asset in enumerate(sources):
            animation_source = project.resolve_path(asset.path)
            if not animation_source.is_file():
                warnings.append(f"Animation source not found: {animation_source}")
                continue
            actual_hash = sha256_file(animation_source)
            if asset.sha256 and asset.sha256.lower() != actual_hash:
                warnings.append(f"Animation source bytes changed: {animation_source.name}")
                continue
            asset.sha256 = actual_hash
            if source_index == 0:
                inventory = primary_inventory
            else:
                try:
                    inventory_clip = import_ue_fbx_animation_clip(
                        str(animation_source), sample_rate=sample_rate
                    )
                except Exception as exc:
                    warnings.append(f"Could not inspect {animation_source.name}: {exc}")
                    continue
                inventory = [dict(value) for value in list(inventory_clip.available_clips or [])]
            for entry in inventory:
                action_name = str(entry.get("name") or "").strip()
                if not action_name:
                    continue
                source_key = action_name
                if source_key in used_keys:
                    source_key = f"{animation_source.stem}::{action_name}"
                suffix = 2
                base_key = source_key
                while source_key in used_keys:
                    source_key = f"{base_key}#{suffix}"
                    suffix += 1
                used_keys.add(source_key)
                annotated = dict(entry)
                annotated.update({
                    "name": source_key,
                    "source_action_name": action_name,
                    "source_path": str(animation_source),
                    "source_sha256": actual_hash,
                    "external": source_index > 0,
                })
                action_inventory.append(annotated)
                if not sample_animations:
                    continue
                try:
                    clip = import_ue_fbx_animation_clip(
                        str(animation_source), clip_name=action_name, sample_rate=sample_rate
                    )
                except Exception as exc:
                    warnings.append(f"Could not sample animation '{source_key}': {exc}")
                    continue
                clips[source_key] = clip

        existing_mapping = {value.source_name: value for value in project.animation_mappings}
        for entry in action_inventory:
            source_key = str(entry.get("name") or "").strip()
            action_name = str(entry.get("source_action_name") or source_key).strip()
            if not source_key or source_key in existing_mapping:
                continue
            category, alias = suggest_semantic_mapping(action_name)
            assignment = "vanilla_behavior_alias" if alias else "unassigned"
            project.animation_mappings.append(AnimationMapping(
                source_name=source_key,
                assignment=assignment,
                exported_name=alias,
                confirmed=False,
                loop=category in {"primary_idle", "walk", "run"},
                bake_rate=sample_rate,
                source_path=str(entry.get("source_path") or source),
                source_sha256=str(entry.get("source_sha256") or source_hash),
                advanced={"source_action_name": action_name},
            ))

        materials = _material_snapshots(mesh_model, project)
        if not project.material_assignments:
            project.material_assignments = [MaterialAssignment(
                material_name=value.material_name,
                texture_resref=value.texture_resref,
                source_texture=value.source_texture,
                source_sha256=sha256_file(value.source_texture) if value.source_texture else "",
                output_format="TGA",
                wrap_mode=value.wrap_mode,
                alpha_mode=value.alpha_mode,
            ) for value in materials]
        else:
            by_name = {value.material_name: value for value in project.material_assignments}
            for material in materials:
                assignment = by_name.get(material.material_name)
                if assignment is None:
                    assignment = MaterialAssignment(
                        material_name=material.material_name,
                        texture_resref=material.texture_resref,
                        output_format="TGA",
                        wrap_mode=material.wrap_mode,
                        alpha_mode=material.alpha_mode,
                    )
                    project.material_assignments.append(assignment)
                if material.source_texture and not assignment.source_texture:
                    assignment.source_texture = material.source_texture
                    assignment.source_sha256 = sha256_file(material.source_texture)
        known_texture_paths = {asset.path.casefold() for asset in project.texture_assets if asset.path}
        for assignment in project.material_assignments:
            if assignment.source_texture and assignment.source_texture.casefold() not in known_texture_paths:
                resolved_texture = project.resolve_path(assignment.source_texture)
                project.texture_assets.append(SourceAsset(
                    path=assignment.source_texture,
                    sha256=assignment.source_sha256 or (sha256_file(resolved_texture) if resolved_texture.is_file() else ""),
                    role=f"texture:{assignment.texture_resref or assignment.material_name}",
                    required=True,
                ))
                known_texture_paths.add(assignment.source_texture.casefold())
        vertices = [vertex for node in _mesh_nodes(mesh_model) for vertex in list(getattr(node, "vertices", []) or [])]
        lowest = min((float(vertex[2]) for vertex in vertices), default=None)
        bb_min = tuple(float(value) for value in getattr(mesh_model, "bb_min", (0, 0, 0)) or (0, 0, 0))
        bb_max = tuple(float(value) for value in getattr(mesh_model, "bb_max", (0, 0, 0)) or (0, 0, 0))
        dimensions = tuple(abs(bb_max[index] - bb_min[index]) * project.global_scale for index in range(3))
        # ``selected_skeleton_root`` may be the user-facing "Armature :: root"
        # choice token. Geometry analysis must use the actual imported node.
        root_name = skeleton.root_names[0] if skeleton.root_names else ""
        root_height = skeleton.global_transforms.get(root_name, Transform()).position[2] if root_name else None
        snapshot = CustomRiggedCharacterSnapshot(
            nodes=[
                RigNodeSnapshot(
                    **{
                        **node.__dict__,
                        "exported": bool(project.export_nodes.get(node.name, node.exported)),
                    }
                )
                for node in skeleton.nodes
            ],
            vertex_influences=_vertex_influences(mesh_model),
            animations=[
                _clip_snapshot(
                    clip,
                    skeleton.axis_conversion,
                    source_key=source_key,
                    loop=bool(next(
                        (value.loop for value in project.animation_mappings if value.source_name == source_key),
                        False,
                    )),
                )
                for source_key, clip in clips.items()
            ],
            materials=materials,
            dimensions=dimensions,
            lowest_contact_height=lowest * project.global_scale if lowest is not None else None,
            root_height=root_height,
            runtime_height_offset=runtime_height_offset,
            runtime_height_source=runtime_height_source,
            source_unit_scale=1.0,
            source_forward=(
                "+Y" if str(project.import_coordinate_system.get("source_forward") or "auto").lower() == "auto"
                else str(project.import_coordinate_system.get("source_forward"))
            ),
            expected_forward="+Y",
            skeleton_fingerprint=skeleton.animation_fingerprint,
            available_skeleton_roots=tuple(skeleton.available_root_choices),
            skeleton_selection_required=skeleton.selection_required,
        )
        summary = {
            "mesh_count": len(_mesh_nodes(mesh_model)),
            "root_name": root_name or "Multiple roots — choose one",
            "bone_count": len(skeleton.nodes),
            "skinned_vertex_count": sum(
                _authored_vertex_count(node)
                for node in _mesh_nodes(mesh_model)
                if bool(getattr(node, "is_skin", False))
            ),
            "animation_count": len(action_inventory),
            "texture_count": len(materials),
            "attention_count": len(warnings),
            "source_sha256": source_hash,
            "skeleton_fingerprint": skeleton.fingerprint,
            "animation_rig_fingerprint": skeleton.animation_fingerprint,
            "available_skeleton_roots": list(skeleton.available_root_choices),
            "skeleton_selection_required": skeleton.selection_required,
            "runtime_height_offset": runtime_height_offset,
            "runtime_height_source": runtime_height_source,
        }
        project.last_import_summary = dict(summary)
        return CustomRiggedCharacterImportResult(
            source_model=mesh_model,
            skeleton=skeleton,
            snapshot=snapshot,
            action_inventory=action_inventory,
            clips=clips,
            summary=summary,
            warnings=warnings,
            project=project,
        )


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(values[index]) for index in range(4))
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / length, y / length, z / length, w / length)


def _quat_conjugate(value: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = _normalized_quaternion(value)
    return (-x, -y, -z, w)


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return _normalized_quaternion((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ))


def _quat_rotate(rotation: Sequence[float], point: Sequence[float]) -> tuple[float, float, float]:
    q = _normalized_quaternion(rotation)
    px, py, pz = point[:3]
    vector_q = (float(px), float(py), float(pz), 0.0)
    # Avoid normalization inside the vector multiplication path.
    x, y, z, w = q
    ix = w * vector_q[0] + y * vector_q[2] - z * vector_q[1]
    iy = w * vector_q[1] + z * vector_q[0] - x * vector_q[2]
    iz = w * vector_q[2] + x * vector_q[1] - y * vector_q[0]
    iw = -x * vector_q[0] - y * vector_q[1] - z * vector_q[2]
    return (
        ix * w + iw * -x + iy * -z - iz * -y,
        iy * w + iw * -y + iz * -x - ix * -z,
        iz * w + iw * -z + ix * -y - iy * -x,
    )


def _static_controller(kind: str, values: Sequence[float]) -> dict[str, Any]:
    return {
        "type": 8 if kind == "position" else 20,
        "name": kind,
        "columns": 3 if kind == "position" else 4,
        "times": [0.0],
        "values": [[float(value) for value in values]],
    }


def _make_node(name: str, position=(0, 0, 0), rotation=(0, 0, 0, 1), *, flags: int) -> Any:
    from src.core.geometry.model_data import ModelNode

    node = ModelNode(
        name=str(name)[:32], flags=int(flags),
        position=tuple(float(value) for value in position),
        rotation=_normalized_quaternion(rotation),
    )
    node.controllers = [
        _static_controller("position", node.position),
        _static_controller("orientation", node.rotation),
    ]
    return node


def _placement_transform(
    project: CustomRiggedCharacterProject,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pivot = list(project.pivot_offset or (0.0, 0.0, 0.0)) + [0.0, 0.0, 0.0]
    angle = math.radians(float(project.forward_rotation_degrees)) * 0.5
    return (
        (
            float(pivot[0]),
            float(pivot[1]),
            float(project.runtime_height_offset) + float(project.ground_offset) + float(pivot[2]),
        ),
        (0.0, 0.0, math.sin(angle), math.cos(angle)),
    )


def _link(parent: Any, child: Any) -> None:
    child.parent = parent
    parent.children.append(child)


def _triangle_bones(node: Any, triangle: Sequence[int]) -> set[int]:
    skins = list(getattr(node, "skin_data", []) or [])
    result: set[int] = set()
    for vertex_index in triangle:
        if int(vertex_index) >= len(skins):
            continue
        for influence in list(getattr(skins[int(vertex_index)], "influences", []) or []):
            if float(getattr(influence, "weight", 0.0)) > 0.0:
                result.add(int(getattr(influence, "bone_index", -1)))
    return {value for value in result if value >= 0}


def _palette_groups(node: Any, limit: int) -> list[list[int]]:
    groups: list[dict[str, Any]] = []
    for triangle_index, triangle in enumerate(list(getattr(node, "faces", []) or [])):
        bones = _triangle_bones(node, triangle)
        if len(bones) > limit:
            raise ValueError(f"Triangle {triangle_index} in '{node.name}' uses {len(bones)} bones; limit is {limit}.")
        candidates = [
            (len(group["bones"] | bones), index)
            for index, group in enumerate(groups)
            if len(group["bones"] | bones) <= limit
        ]
        if candidates:
            _size, group_index = min(candidates)
            group = groups[group_index]
        else:
            group = {"bones": set(), "triangles": []}
            groups.append(group)
        group["bones"].update(bones)
        group["triangles"].append(triangle_index)
    return [group["triangles"] for group in groups]


def _uv_tile_offset(uvs: Sequence[Sequence[float]]) -> tuple[float, float]:
    if not uvs:
        return (0.0, 0.0)
    minimum_u = min(float(uv[0]) for uv in uvs)
    maximum_u = max(float(uv[0]) for uv in uvs)
    minimum_v = min(float(uv[1]) for uv in uvs)
    maximum_v = max(float(uv[1]) for uv in uvs)
    if maximum_u - minimum_u <= 1.0 + 1.0e-5 and maximum_v - minimum_v <= 1.0 + 1.0e-5:
        return (-float(math.floor(minimum_u)), -float(math.floor(minimum_v)))
    return (0.0, 0.0)


def _inverse_bind_arrays(model: Any, skin: Any) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float]], list[str]]:
    by_name = {str(node.name).casefold(): node for node in model.all_nodes()}
    skin_position, skin_rotation = skin.world_transform()
    qbone: list[tuple[float, float, float, float]] = []
    tbone: list[tuple[float, float, float]] = []
    missing: list[str] = []
    for name in list(getattr(skin, "bone_map", []) or []):
        bone = by_name.get(str(name).casefold())
        if bone is None:
            missing.append(str(name))
            qbone.append((1.0, 0.0, 0.0, 0.0))
            tbone.append((0.0, 0.0, 0.0))
            continue
        bone_position, bone_rotation = bone.world_transform()
        inverse_rotation = _quat_conjugate(bone_rotation)
        relative_rotation = _quat_mul(inverse_rotation, skin_rotation)
        relative_translation = _quat_rotate(inverse_rotation, (
            skin_position[0] - bone_position[0],
            skin_position[1] - bone_position[1],
            skin_position[2] - bone_position[2],
        ))
        qbone.append((relative_rotation[3], relative_rotation[0], relative_rotation[1], relative_rotation[2]))
        tbone.append(relative_translation)
    return qbone, tbone, missing


def _skin_part(
    source: Any,
    triangle_indices: Sequence[int],
    part_index: int,
    parent: Any,
    project: CustomRiggedCharacterProject,
) -> Any:
    from src.core.geometry.model_data import BoneWeight, NodeFlags, VertexSkinData

    faces = list(getattr(source, "faces", []) or [])
    source_vertex_indices = list(
        getattr(source, "_gr_source_vertex_indices", []) or []
    )
    used: list[int] = []
    remap: dict[int, int] = {}
    source_to_part: dict[int, int] = {}
    for triangle_index in triangle_indices:
        for raw_index in faces[int(triangle_index)]:
            raw_index = int(raw_index)
            source_index = (
                int(source_vertex_indices[raw_index])
                if raw_index < len(source_vertex_indices)
                else raw_index
            )
            part_vertex = source_to_part.get(source_index)
            if part_vertex is None:
                part_vertex = len(used)
                source_to_part[source_index] = part_vertex
                used.append(raw_index)
            remap[raw_index] = part_vertex
    source_palette = [str(value) for value in list(getattr(source, "bone_map", []) or [])]
    source_skins = list(getattr(source, "skin_data", []) or [])
    palette_indices = sorted({
        int(getattr(influence, "bone_index", -1))
        for vertex_index in used if vertex_index < len(source_skins)
        for influence in list(getattr(source_skins[vertex_index], "influences", []) or [])
        if float(getattr(influence, "weight", 0.0)) > 0.0
    })
    if len(palette_indices) > CUSTOM_SKIN_PALETTE_LIMIT:
        raise ValueError(f"Split part of '{source.name}' still uses {len(palette_indices)} bones.")
    local_palette = {source_index: local_index for local_index, source_index in enumerate(palette_indices)}
    node = _make_node(
        f"{str(source.name)[:23]}_{part_index}",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
    )
    scale = float(project.global_scale)
    node.vertices = [tuple(float(value) * scale for value in source.vertices[index]) for index in used]
    node.normals = [tuple(float(value) for value in source.normals[index]) for index in used] if getattr(source, "normals", None) else []
    source_uvs = list(getattr(source, "uvs", []) or [])
    offset_u, offset_v = _uv_tile_offset(source_uvs)
    node.uvs = [
        (float(source_uvs[index][0]) + offset_u, float(source_uvs[index][1]) + offset_v)
        for index in used
    ] if source_uvs else []
    # Preserve the source-node orientation contract. Blender/DCC UVs are
    # previewed with this flag disabled, and the binary writer then performs
    # the one required conversion into KOTOR MDX orientation.
    node.uv_v_flip = bool(getattr(source, "uv_v_flip", True))
    node.faces = [tuple(remap[int(raw)] for raw in faces[int(index)]) for index in triangle_indices]
    node.bone_map = [source_palette[index] for index in palette_indices]
    node.skin_data = []
    max_influences = int(project.skin_repair_settings.get("max_influences", 4) or 4)
    for source_vertex in used:
        raw = []
        if source_vertex < len(source_skins):
            for influence in list(getattr(source_skins[source_vertex], "influences", []) or []):
                source_bone = int(getattr(influence, "bone_index", -1))
                if source_bone in local_palette and source_bone < len(source_palette):
                    raw.append((source_palette[source_bone], float(getattr(influence, "weight", 0.0))))
        clean = normalized_influences(raw, max_influences=max_influences)
        name_to_source = {source_palette[index]: index for index in palette_indices}
        row = VertexSkinData(influences=[
            BoneWeight(bone_index=local_palette[name_to_source[name]], weight=weight)
            for name, weight in clean
        ])
        row.normalize()
        node.skin_data.append(row)
    texture = str(getattr(source, "texture", "") or source.name)
    node.texture = Path(texture.replace("\\", "/")).stem.lower()[:16]
    assignment = next((value for value in project.material_assignments if value.material_name == source.name), None)
    if assignment and assignment.texture_resref:
        node.texture = assignment.texture_resref[:16]
    node.texture_names = [node.texture]
    node.tex_count = 1
    node.diffuse = tuple(getattr(source, "diffuse", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0))
    node.ambient = (1.0, 1.0, 1.0)
    node.render = True
    node.has_shadow = True
    node.vertex_space = 1
    setattr(node, "_gr_vertices_in_kotor_world", True)
    setattr(node, "_gr_custom_rig_authority", True)
    node.compute_bounds()
    _link(parent, node)
    return node


def _controller(kind: str, times: Sequence[float], values: Sequence[Sequence[float]]) -> dict[str, Any]:
    return {
        "type": 8 if kind == "position" else 20,
        "name": kind,
        "columns": 3 if kind == "position" else 4,
        "times": [float(value) for value in times],
        "values": [[float(component) for component in row] for row in values],
    }


def _exported_skeleton_records(
    project: CustomRiggedCharacterProject,
    imported: CustomRiggedCharacterImportResult,
) -> list[RigNodeSnapshot]:
    return [
        record for record in imported.skeleton.nodes
        if bool(project.export_nodes.get(record.name, True))
    ]


def _absorbed_source_root(records: Sequence[RigNodeSnapshot]) -> str:
    """Return the single foreign root that Odyssey's model root replaces.

    The proven Borhek model does not add ``godnode`` below ``c_borhek``.  Its
    index-zero authoring root *becomes* ``c_borhek`` and its children move
    below the standard ``heightdummy -> cutscenedummy`` edge.  Keeping both
    roots changes the animation tree the creature runtime resolves.
    """

    roots = [record.name for record in records if not str(record.parent or "")]
    return roots[0] if len(roots) == 1 else ""


def _export_parent_name(
    record: RigNodeSnapshot,
    *,
    records_by_name: Mapping[str, RigNodeSnapshot],
    exported_names: set[str],
    absorbed_root: str,
) -> str:
    """Resolve the closest retained parent, stopping at the absorbed root."""

    parent_name = str(record.parent or "")
    visited: set[str] = set()
    while parent_name and parent_name not in visited:
        if parent_name == absorbed_root:
            return ""
        if parent_name in exported_names:
            return parent_name
        visited.add(parent_name)
        parent = records_by_name.get(parent_name)
        parent_name = str(parent.parent or "") if parent is not None else ""
    return ""


def _has_meaningful_position_delta(
    values: Sequence[Sequence[float]],
) -> bool:
    return any(
        abs(float(component)) > _ANIMATION_POSITION_DELTA_EPSILON
        for row in values
        for component in row[:3]
    )


def _build_animation(
    project: CustomRiggedCharacterProject,
    imported: CustomRiggedCharacterImportResult,
    mapping: AnimationMapping,
) -> Any:
    from src.core.geometry.model_data import Animation, ModelNode, NodeFlags

    clip = imported.clips[mapping.source_name]
    basis = _basis_for_axis_conversion(imported.skeleton.axis_conversion)
    ordered_skeleton = _exported_skeleton_records(project, imported)
    absorbed_root = _absorbed_source_root(ordered_skeleton)
    records_by_name = {record.name: record for record in imported.skeleton.nodes}
    exported_names = {record.name for record in ordered_skeleton}
    nodes: dict[str, Any] = {}
    trim_start = max(0.0, float(mapping.trim_start or 0.0))
    trim_end = float(mapping.trim_end) if mapping.trim_end is not None else float(clip.duration_seconds)
    trim_end = min(float(clip.duration_seconds), trim_end)
    if trim_end <= trim_start:
        raise ValueError(f"Animation '{mapping.source_name}' has an empty trim range.")
    selected_poses = [
        pose for pose in clip.sampled_poses
        if trim_start - 1.0e-7 <= float(pose.time_seconds) <= trim_end + 1.0e-7
    ]
    if not selected_poses:
        raise ValueError(f"Animation '{mapping.source_name}' has no sampled poses inside its trim range.")
    source_span = max(1.0e-9, trim_end - trim_start)
    target_span = (
        float(mapping.retime_duration)
        if mapping.retime_duration is not None and float(mapping.retime_duration) > 0.0
        else source_span / max(float(mapping.playback_speed), 1.0e-6)
    )
    time_scale = target_span / source_span
    times = [max(0.0, float(pose.time_seconds) - trim_start) * time_scale for pose in selected_poses]
    for record in ordered_skeleton:
        rest = imported.skeleton.local_transforms[record.name]
        node = ModelNode(
            name=(project.resource_name if record.name == absorbed_root else record.name)[:32],
            flags=int(NodeFlags.HEADER),
            position=rest.position, rotation=rest.rotation,
        )
        positions: list[tuple[float, float, float]] = []
        rotations: list[tuple[float, float, float, float]] = []
        for pose in selected_poses:
            source = pose.local_transforms.get(record.name)
            if source is None:
                source = next((value.rest_local for value in clip.nodes if value.name == record.name), None)
            if source is None:
                continue
            converted = _kotor_transform(source, scale=project.global_scale, basis=basis)
            delta = tuple(converted.position[index] - rest.position[index] for index in range(3))
            if not record.parent and mapping.root_motion in {"in_place", "analysis_only"}:
                delta = (0.0, 0.0, 0.0)
            positions.append(delta)
            rotations.append(converted.rotation)
        # Blender-baked actions contain sub-micrometre translation noise on
        # otherwise rotation-only bones.  The working Borhek patch writes no
        # position track for those bones; Odyssey should inherit their base
        # local translation instead of evaluating a redundant controller.
        if positions and _has_meaningful_position_delta(positions):
            node.controllers.append(_controller("position", times[:len(positions)], positions))
        if rotations:
            rotations, _flips = quaternion_continuity(rotations)
            node.controllers.append(_controller("orientation", times[:len(rotations)], rotations))
        nodes[record.name] = node
    model_root = (
        nodes[absorbed_root]
        if absorbed_root
        else ModelNode(name=project.resource_name[:32], flags=int(NodeFlags.HEADER))
    )
    # Keep the proven Borhek helper node/edge in every animation, but do not
    # repeat its base translation as an animation controller.  Odyssey position
    # keys are deltas added to the base-local transform; a copied +Z key would
    # therefore double the correction whenever an animation plays.
    height = ModelNode(name="heightdummy", flags=int(NodeFlags.HEADER))
    cutscene = ModelNode(name="cutscenedummy", flags=int(NodeFlags.HEADER))
    _link(model_root, height)
    _link(height, cutscene)
    for record in ordered_skeleton:
        if record.name == absorbed_root:
            continue
        parent_name = _export_parent_name(
            record,
            records_by_name=records_by_name,
            exported_names=exported_names,
            absorbed_root=absorbed_root,
        )
        parent = nodes.get(parent_name) if parent_name else cutscene
        _link(parent or cutscene, nodes[record.name])
    return Animation(
        name=mapping.exported_name[:32],
        length=target_span,
        transition_time=float(mapping.transition_time),
        anim_root=project.resource_name[:32],
        events=[],
        nodes=[
            model_root,
            height,
            cutscene,
            *[nodes[value.name] for value in ordered_skeleton if value.name != absorbed_root],
        ],
    )


def build_self_contained_odyssey_model(
    project: CustomRiggedCharacterProject,
    imported: CustomRiggedCharacterImportResult,
) -> tuple[Any, list[dict[str, Any]]]:
    """Convert the imported foreign hierarchy/skins/actions into one model."""

    from src.core.geometry.model_data import GameVersion, KotorModel, ModelClassification, NodeFlags

    if not project.resource_name:
        raise ValueError("A KOTOR resource name is required.")
    ordered_skeleton = _exported_skeleton_records(project, imported)
    absorbed_root = _absorbed_source_root(ordered_skeleton)
    records_by_name = {record.name: record for record in imported.skeleton.nodes}
    exported_names = {record.name for record in ordered_skeleton}
    absorbed_transform = (
        imported.skeleton.local_transforms[absorbed_root]
        if absorbed_root
        else None
    )
    root = _make_node(
        project.resource_name,
        position=absorbed_transform.position if absorbed_transform is not None else (0.0, 0.0, 0.0),
        rotation=absorbed_transform.rotation if absorbed_transform is not None else (0.0, 0.0, 0.0, 1.0),
        flags=int(NodeFlags.HEADER),
    )
    placement_position, placement_rotation = _placement_transform(project)
    height = _make_node(
        "heightdummy",
        position=placement_position,
        rotation=placement_rotation,
        flags=int(NodeFlags.HEADER),
    )
    cutscene = _make_node("cutscenedummy", flags=int(NodeFlags.HEADER))
    _link(root, height)
    _link(height, cutscene)
    bone_nodes: dict[str, Any] = {}
    for record in ordered_skeleton:
        if record.name == absorbed_root:
            continue
        transform = imported.skeleton.local_transforms[record.name]
        node = _make_node(record.name, transform.position, transform.rotation, flags=int(NodeFlags.HEADER))
        bone_nodes[record.name] = node
    for record in ordered_skeleton:
        if record.name == absorbed_root:
            continue
        node = bone_nodes.get(record.name)
        if node is None:
            continue
        parent_name = _export_parent_name(
            record,
            records_by_name=records_by_name,
            exported_names=exported_names,
            absorbed_root=absorbed_root,
        )
        _link(bone_nodes.get(parent_name) if parent_name else cutscene, node)
    camera = _make_node("camerahook", position=(0.0, 1.1, 2.3), flags=int(NodeFlags.HEADER))
    _link(root, camera)
    model = KotorModel(
        name=project.resource_name[:32], supermodel="NULL", classification="character",
        game_version=GameVersion.K2 if project.target_game == "K2" else GameVersion.K1,
        model_type=int(ModelClassification.CHARACTER), anim_scale=1.0, root_node=root,
    )
    split_report: list[dict[str, Any]] = []
    for source in _mesh_nodes(imported.source_model):
        if not bool(getattr(source, "is_skin", False)):
            continue
        groups = _palette_groups(source, CUSTOM_SKIN_PALETTE_LIMIT)
        parts = [_skin_part(source, group, index, height, project) for index, group in enumerate(groups)]
        if absorbed_root:
            for part in parts:
                part.bone_map = [
                    project.resource_name if name == absorbed_root else name
                    for name in part.bone_map
                ]
        split_report.append({
            "source_mesh": str(source.name),
            "source_vertices": _authored_vertex_count(source),
            "source_triangles": len(source.faces),
            "parts": [
                {"name": part.name, "vertices": len(part.vertices), "triangles": len(part.faces), "palette": len(part.bone_map)}
                for part in parts
            ],
        })
    for skin in [node for node in model.all_nodes() if bool(getattr(node, "is_skin", False))]:
        skin.qbone_list, skin.tbone_list, missing = _inverse_bind_arrays(model, skin)
        if missing:
            raise ValueError(f"Skin '{skin.name}' refers to missing exported bones: {', '.join(missing)}")
        setattr(skin, "_gr_kotor_inverse_bind_qt", True)
    for mapping in project.animation_mappings:
        if not mapping.confirmed or mapping.assignment == "unassigned" or not mapping.exported_name:
            continue
        if mapping.source_name not in imported.clips:
            raise ValueError(f"Mapped animation '{mapping.source_name}' has not been sampled.")
        model.animations.append(_build_animation(project, imported, mapping))
    model.metadata = dict(getattr(model, "metadata", {}) or {})
    model.metadata["custom_rigged_character"] = {
        "project_id": project.project_id,
        "builder_mode": project.builder_mode,
        "foreign_hierarchy_preserved": True,
        "absorbed_source_root": absorbed_root,
        "native_humanoid_template_used": False,
        "source_sha256": project.primary_fbx.sha256,
        "skeleton_fingerprint": imported.skeleton.fingerprint,
        "runtime_height_offset": float(project.runtime_height_offset),
        "runtime_height_source": str(project.runtime_height_source or ""),
        "split_report": split_report,
    }
    model.compute_bounds()
    return model, split_report


__all__ = [
    "CUSTOM_SKIN_PALETTE_LIMIT",
    "CustomRiggedCharacterImportResult",
    "CustomRiggedCharacterImportService",
    "ImportedSkeleton",
    "build_self_contained_odyssey_model",
    "detect_runtime_height_offset",
    "imported_skeleton_from_model",
]
