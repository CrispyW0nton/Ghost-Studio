"""Prepare a DCC-friendly module OBJ without backdrop-only geometry.

This module is owned by ``GhostRigger.Core.IO`` because it defines export-time
geometry selection and compaction.  The active model-conversion pipeline is
package-local, so this file intentionally has no root ``src`` counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

from src.core.geometry.model_data import KotorModel
from src.io.export_control import TextureSidecarResult, check_export_cancelled
from src.io.fbx.fbx_exporter import merge_selected_scene_objects


DEFAULT_BACKGROUND_FACE_AREA = 100_000.0
_SKY_TEXTURE_PATTERN = re.compile(r"(?:^|[_-])sky(?:box)?[a-z0-9_-]*$", re.IGNORECASE)


@dataclass(frozen=True)
class CleanModuleObjSummary:
    """Counts reported to the UI after a clean module OBJ export."""

    scene_objects: int
    source_faces: int
    exported_faces: int
    removed_skybox_faces: int
    removed_background_faces: int
    materials: int
    texture_sidecars: TextureSidecarResult = field(default_factory=TextureSidecarResult)


def _material_texture(material: Any) -> str:
    raw = getattr(material, "texture_clean", "") or getattr(material, "texture", "") or ""
    return str(raw).strip()


def _is_skybox_material(material: Any) -> bool:
    return bool(_SKY_TEXTURE_PATTERN.search(_material_texture(material)))


def _triangle_area(vertices: list[tuple[float, float, float]], face: Any) -> float:
    if len(face) != 3:
        return 0.0
    try:
        a, b, c = (vertices[int(index)] for index in face)
    except (IndexError, TypeError, ValueError):
        return 0.0
    ab = tuple(float(b[index]) - float(a[index]) for index in range(3))
    ac = tuple(float(c[index]) - float(a[index]) for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(component * component for component in cross))


def _is_oversized_flat_background(
    vertices: list[tuple[float, float, float]],
    face: Any,
    *,
    minimum_area: float,
) -> bool:
    if _triangle_area(vertices, face) < float(minimum_area):
        return False
    try:
        z_values = [float(vertices[int(index)][2]) for index in face]
    except (IndexError, TypeError, ValueError):
        return False
    return max(z_values) - min(z_values) <= 1e-4


def validate_module_scene_objects(
    scene_objects: Any,
    *,
    is_cancelled=None,
) -> list[Any]:
    """Materialize loaded module rooms and require a runtime model for each."""

    objects = list(scene_objects or [])
    validated_objects: list[Any] = []
    missing_rooms: list[str] = []
    for index, item in enumerate(objects):
        check_export_cancelled(is_cancelled)
        runtime_model = (
            item
            if isinstance(item, KotorModel)
            else (getattr(item, "metadata", {}) or {}).get("_runtime_model")
        )
        if not isinstance(runtime_model, KotorModel):
            missing_rooms.append(str(getattr(item, "name", "") or f"Room {index + 1}"))
            continue
        validated_objects.append(item)

    if missing_rooms:
        raise ValueError(
            "Cannot export the complete map because these rooms are not loaded: "
            f"{', '.join(missing_rooms)}. Reload the module layout and try again."
        )

    check_export_cancelled(is_cancelled)
    return validated_objects


def prepare_clean_module_obj_model(
    scene_objects: Any,
    *,
    name: str = "module_clean",
    background_face_area: float = DEFAULT_BACKGROUND_FACE_AREA,
    is_cancelled=None,
) -> tuple[KotorModel, CleanModuleObjSummary]:
    """Merge a loaded module and remove skybox/backdrop-only faces.

    The merged source expands vertices per polygon corner.  Compaction is still
    performed explicitly so excluded faces leave no orphaned skybox vertices in
    the OBJ file.
    """

    validated_objects = validate_module_scene_objects(
        scene_objects,
        is_cancelled=is_cancelled,
    )
    merged = merge_selected_scene_objects(
        validated_objects,
        name=name,
        is_cancelled=is_cancelled,
    )
    mesh = merged.root_node
    if mesh is None:
        raise ValueError("The loaded module contains no exportable mesh geometry.")

    material_slots = list(getattr(mesh, "_gr_fbx_material_slots", None) or [mesh])
    source_faces = list(getattr(mesh, "faces", None) or [])
    face_materials = list(getattr(mesh, "face_mats", None) or [])
    kept: list[tuple[Any, int]] = []
    removed_skybox = 0
    removed_background = 0

    for face_index, face in enumerate(source_faces):
        if face_index % 256 == 0:
            check_export_cancelled(is_cancelled)
        material_index = int(face_materials[face_index]) if face_index < len(face_materials) else 0
        material_index = max(0, min(material_index, len(material_slots) - 1))
        if _is_skybox_material(material_slots[material_index]):
            removed_skybox += 1
            continue
        if _is_oversized_flat_background(
            mesh.vertices,
            face,
            minimum_area=background_face_area,
        ):
            removed_background += 1
            continue
        kept.append((face, material_index))

    if not kept:
        raise ValueError("Removing skybox and background geometry left no map faces to export.")

    used_materials = sorted({material_index for _face, material_index in kept})
    material_remap = {old: new for new, old in enumerate(used_materials)}
    compact_slots = [material_slots[index] for index in used_materials]
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    lightmap_uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    compact_face_materials: list[int] = []

    for face_index, (source_face, source_material) in enumerate(kept):
        if face_index % 256 == 0:
            check_export_cancelled(is_cancelled)
        output_face = []
        for raw_index in source_face:
            vertex_index = int(raw_index)
            output_face.append(len(vertices))
            vertices.append(tuple(float(value) for value in mesh.vertices[vertex_index][:3]))
            normals.append(
                tuple(float(value) for value in mesh.normals[vertex_index][:3])
                if vertex_index < len(mesh.normals)
                else (0.0, 0.0, 1.0)
            )
            uvs.append(
                tuple(float(value) for value in mesh.uvs[vertex_index][:2])
                if vertex_index < len(mesh.uvs)
                else (0.0, 0.0)
            )
            lightmap_uvs.append(
                tuple(float(value) for value in mesh.uvs_lm[vertex_index][:2])
                if vertex_index < len(mesh.uvs_lm)
                else (0.0, 0.0)
            )
        faces.append(tuple(output_face))
        compact_face_materials.append(material_remap[source_material])

    check_export_cancelled(is_cancelled)
    mesh.vertices = vertices
    mesh.normals = normals
    mesh.uvs = uvs
    mesh.uvs_lm = lightmap_uvs
    mesh.faces = faces
    mesh.face_mats = compact_face_materials
    mesh._gr_fbx_material_slots = compact_slots
    mesh.texture_names = [_material_texture(slot) for slot in compact_slots]
    mesh.tex_count = len(compact_slots)
    mesh.texture = mesh.texture_names[0] if mesh.texture_names else ""
    mesh.compute_bounds()
    merged.compute_bounds()

    return merged, CleanModuleObjSummary(
        scene_objects=len(validated_objects),
        source_faces=len(source_faces),
        exported_faces=len(faces),
        removed_skybox_faces=removed_skybox,
        removed_background_faces=removed_background,
        materials=len(compact_slots),
    )


__all__ = [
    "CleanModuleObjSummary",
    "DEFAULT_BACKGROUND_FACE_AREA",
    "prepare_clean_module_obj_model",
    "validate_module_scene_objects",
]
