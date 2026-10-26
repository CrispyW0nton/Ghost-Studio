"""Renderer-neutral mesh/material extraction for lightweight WGPU drawing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable


@dataclass(frozen=True)
class TextureRenderData:
    texture_id: str
    name: str
    source: object | None
    source_revision: tuple[int, int, int]


@dataclass(frozen=True)
class MaterialRenderData:
    material_id: str
    name: str
    diffuse_texture_id: str
    diffuse_texture_path: str
    diffuse_texture_data: TextureRenderData | None
    lightmap_texture_id: str
    lightmap_texture_path: str
    lightmap_texture_data: TextureRenderData | None
    base_color_rgba: tuple[float, float, float, float]
    alpha_mode: str
    alpha_cutoff: float
    blend_mode: str
    sprite_alpha_source: int
    sprite_glow: float
    double_sided: bool
    unlit: bool
    has_transparency: bool
    source_revision: tuple[int, int, int, int]


@dataclass(frozen=True)
class MeshRenderData:
    mesh_id: int
    source: object
    positions: object
    normals: object | None
    uvs0: object | None
    uvs1: object | None
    indices: object | None
    material: MaterialRenderData
    material_color: tuple[float, float, float, float]
    world_matrix: object
    source_revision: tuple[int, int, int]
    bone_indices: object | None = None
    bone_weights: object | None = None
    max_influences: int = 0
    skeleton_id: int = 0
    skin_revision: int = 0
    bind_shape_matrix: object | None = None
    is_skinned: bool = False
    skinning_cpu_fallback: bool = False
    skinning_warning: str = ""


class ScopedAnimationPoseSet:
    """Renderer-facing collection of independently evaluated character poses."""

    _gr_animation_pose_set = True

    def __init__(self, poses_by_character: dict[str, object] | None = None):
        self.poses_by_character = {
            str(character_id): pose
            for character_id, pose in (poses_by_character or {}).items()
            if str(character_id) and pose is not None
        }
        self.time = max((float(getattr(pose, "time", 0.0) or 0.0) for pose in self.poses_by_character.values()), default=0.0)
        self.nodes = {}

    def __bool__(self) -> bool:
        return bool(self.poses_by_character)

    def active_poses(self) -> tuple[object, ...]:
        return tuple(self.poses_by_character.values())

    def pose_for_node(self, node) -> object | None:
        if node is None:
            return None
        node_object_id = _scene_object_id_for_node(node)
        if node_object_id:
            direct = self.poses_by_character.get(node_object_id)
            if direct is not None:
                return direct
        node_import_id = _scene_import_id_for_node(node)
        if node_import_id:
            for pose in self.poses_by_character.values():
                if str(getattr(pose, "_gr_animation_scene_import_id", "") or "") == node_import_id:
                    return pose
        node_source_id = _source_model_id_for_node(node)
        # A scoped collection is intentionally stricter than a legacy single
        # pose.  Identity-free module geometry must never inherit whichever
        # character happens to be first in ``poses_by_character``.  Runtime
        # actor descendants still resolve their inherited object/import/source
        # identity through the parent-chain helpers above.
        if not node_object_id and not node_import_id and not node_source_id:
            return None
        for pose in self.poses_by_character.values():
            if _animation_pose_matches_node_identity(node, pose, node_object_id=node_object_id, node_import_id=node_import_id, node_source_id=node_source_id):
                return pose
        return None


def iter_mesh_render_data(
    model,
    *,
    anim_pose=None,
    anim_base_pose=None,
    textures: dict | None = None,
    allow_cpu_skinning: bool = True,
    vbo_builder=None,
) -> Iterable[MeshRenderData]:
    """Yield mesh draw data without storing renderer resources on scene nodes."""

    if model is None:
        return []

    import numpy as np

    textures = textures or {}
    nodes = _model_nodes(model)
    rows: list[MeshRenderData] = []
    for node in nodes:
        if not _node_is_renderable_mesh(node):
            continue
        node_anim_pose = _effective_animation_pose_for_node(node, anim_pose)
        try:
            (
                positions,
                normals,
                uvs0,
                uvs1,
                indices,
                vbo_bone_indices,
                vbo_bone_weights,
                world_matrix,
            ) = _extract_node_arrays(node, anim_pose=node_anim_pose, vbo_builder=vbo_builder)
        except Exception:
            continue
        if positions is None or len(positions) == 0:
            continue
        diffuse_name = _clean_tex_name(getattr(node, "texture", "") or "")
        diffuse_source = textures.get(diffuse_name.lower()) or textures.get(diffuse_name)
        texture_v_flip = bool(getattr(diffuse_source, "_gr_gpu_uv_v_flip", True))
        node_v_flip = bool(getattr(node, "uv_v_flip", True))
        if uvs0 is not None and node_v_flip and not texture_v_flip:
            # WGPU/PyGFX shaders apply the KotOR V conversion. Match ModernGL
            # by pre-flipping loose bottom-left DCC atlases to cancel it.
            uvs0 = np.asarray(uvs0, dtype=np.float32).copy()
            uvs0[:, 1] = 1.0 - uvs0[:, 1]
        skinning = _extract_skinning(
            node,
            len(positions),
            skeleton_id=id(model),
            bone_indices=vbo_bone_indices,
            bone_weights=vbo_bone_weights,
        )
        skinning_cpu_fallback = False
        is_bas_attachment_skin = bool(
            getattr(node, "_gr_bas_attachment_layer", False)
            and getattr(node, "is_skin", False)
            and getattr(skinning, "is_skinned", False)
        )
        if is_bas_attachment_skin and anim_pose is None:
            skinning = SimpleNamespace(
                is_skinned=False,
                bone_indices=None,
                bone_weights=None,
                max_influences=0,
                skeleton_id=0,
                skin_revision=getattr(skinning, "skin_revision", 0),
                bind_shape_matrix=None,
                warning="BAS attachment skin rendered as socket-local preview mesh",
            )
        elif is_bas_attachment_skin:
            try:
                from src.core.rendering.skeleton_render_data import cpu_skin_vbo_arrays

                skinned_positions, skinned_normals = cpu_skin_vbo_arrays(
                    node,
                    positions,
                    normals,
                    skinning,
                    node_anim_pose,
                    model=model,
                    anim_base_pose=anim_base_pose,
                )
                if skinned_positions is not positions:
                    positions = skinned_positions
                    normals = skinned_normals
                    skinning_cpu_fallback = True
                    skinning = SimpleNamespace(
                        is_skinned=False,
                        bone_indices=None,
                        bone_weights=None,
                        max_influences=0,
                        skeleton_id=0,
                        skin_revision=getattr(skinning, "skin_revision", 0),
                        bind_shape_matrix=None,
                        warning="BAS attachment skin rendered as socket-local CPU-skinned preview mesh",
                    )
            except Exception:
                pass
        if allow_cpu_skinning and anim_pose is not None and getattr(skinning, "is_skinned", False):
            try:
                from src.core.rendering.skeleton_render_data import cpu_skin_vbo_arrays

                skinned_positions, skinned_normals = cpu_skin_vbo_arrays(
                    node,
                    positions,
                    normals,
                    skinning,
                    node_anim_pose,
                    model=model,
                    anim_base_pose=anim_base_pose,
                )
                if skinned_positions is not positions:
                    positions = skinned_positions
                    normals = skinned_normals
                    skinning_cpu_fallback = True
            except Exception:
                pass
        material = _material_data(node, textures)
        source_revision = (*_node_revision(node), int(node_v_flip and texture_v_flip))
        if getattr(skinning, "is_skinned", False):
            skin_lbs_input_mode = 1 if node_anim_pose is not None else 0
            source_revision = (
                *source_revision,
                int(getattr(skinning, "skin_revision", 0) or 0),
                skin_lbs_input_mode,
            )
        if skinning_cpu_fallback:
            source_revision = (*source_revision, int(round(float(getattr(node_anim_pose, "time", 0.0) or 0.0) * 1000.0)))
        rows.append(
            MeshRenderData(
                mesh_id=id(node),
                source=node,
                positions=np.asarray(positions, dtype=np.float32),
                normals=np.asarray(normals, dtype=np.float32) if normals is not None else None,
                uvs0=np.asarray(uvs0, dtype=np.float32) if uvs0 is not None else None,
                uvs1=np.asarray(uvs1, dtype=np.float32) if uvs1 is not None else None,
                indices=np.asarray(indices, dtype=np.uint32) if indices is not None else None,
                material=material,
                material_color=material.base_color_rgba,
                world_matrix=np.asarray(world_matrix, dtype=np.float32),
                source_revision=source_revision,
                bone_indices=getattr(skinning, "bone_indices", None),
                bone_weights=getattr(skinning, "bone_weights", None),
                max_influences=int(getattr(skinning, "max_influences", 0) or 0),
                skeleton_id=int(getattr(skinning, "skeleton_id", 0) or 0),
                skin_revision=int(getattr(skinning, "skin_revision", 0) or 0),
                bind_shape_matrix=getattr(skinning, "bind_shape_matrix", None),
                is_skinned=bool(getattr(skinning, "is_skinned", False)),
                skinning_cpu_fallback=bool(skinning_cpu_fallback),
                skinning_warning=str(getattr(skinning, "warning", "") or ""),
            )
        )
    return rows


def _model_nodes(model) -> list:
    if hasattr(model, "all_nodes"):
        try:
            return list(model.all_nodes())
        except Exception:
            pass
    if hasattr(model, "mesh_nodes"):
        try:
            return list(model.mesh_nodes())
        except Exception:
            pass
    return list(getattr(model, "nodes", []) or [])


def _node_is_embedded_walkmesh(node) -> bool:
    """Return whether *node* is collision/pathing geometry, not a visual mesh.

    Stock room MDLs carry their AABB walkmesh alongside the visible room
    geometry.  The loader marks these nodes with ``vertex_space=2`` and the
    original AABB flag, but retained/composed scenes may preserve only one of
    those signals.  Keep every renderer on the same conservative contract:
    embedded walkmeshes are presented only by the dedicated walkmesh overlay.
    """
    if node is None:
        return False
    if bool(getattr(node, "is_aabb", False)):
        return True
    try:
        if int(getattr(node, "flags", 0) or 0) & 0x0200:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return int(getattr(node, "vertex_space", 0) or 0) == 2
    except (TypeError, ValueError):
        return False


def _node_is_renderable_mesh(node) -> bool:
    if node is None:
        return False
    if bool(getattr(node, "_gr_hidden", False)):
        return False
    if getattr(node, "render", True) is False:
        return False
    if bool(getattr(node, "is_saber", False)):
        return False
    if _node_is_embedded_walkmesh(node):
        return False
    return bool(getattr(node, "vertices", getattr(node, "verts", [])) and getattr(node, "faces", []))


def _node_has_renderable_rigid_surface(node) -> bool:
    """Return whether a rigid MDL node explicitly describes visible geometry.

    Odyssey's ``_g`` suffix is used both by hidden deformation helpers and by
    visible rigid body segments on several KOTOR 2 droids.  The authored render
    flag, diffuse texture, and UV channel together are stronger evidence than
    that naming convention alone.
    """
    if node is None or bool(getattr(node, "is_skin", False)):
        return False
    if getattr(node, "render", True) is False:
        return False
    texture = _clean_tex_name(getattr(node, "texture", "") or "")
    if not texture or texture == "****":
        return False
    return bool(getattr(node, "uvs", []) or [])


def _node_vertices_are_model_world(node) -> bool:
    if bool(getattr(node, "_gr_vertices_in_kotor_world", False)):
        return True
    try:
        return int(getattr(node, "vertex_space", 0) or 0) == 1
    except Exception:
        return False


def _extract_node_arrays(node, *, anim_pose=None, vbo_builder=None):
    import numpy as np

    is_skin = bool(getattr(node, "is_skin", False))
    is_bas_attachment = bool(getattr(node, "_gr_bas_attachment_layer", False))
    bas_root = _bas_attachment_root_for_node(node) if is_bas_attachment else None
    vertices_are_model_world = _node_vertices_are_model_world(node)
    world_pos, world_orient = _node_world_transform(node, anim_pose=anim_pose)
    world_matrix = node_world_matrix(node, anim_pose=anim_pose)
    if vertices_are_model_world and not is_bas_attachment:
        world_pos = (0.0, 0.0, 0.0)
        world_orient = (0.0, 0.0, 0.0, 1.0)
        world_matrix = np.eye(4, dtype=np.float32)
    if is_skin and bas_root is not None:
        # BAS skin attachments must stay out of the body palette, but their
        # bind-shape should remain local to the attachment root.  Keeping large
        # negative skin-local vertices plus a compensating full-node matrix makes
        # heads look deformed as more layers are added.
        world_pos, world_orient = _bas_attachment_local_transform(node, bas_root)
        world_matrix = node_world_matrix(bas_root, anim_pose=anim_pose)
    if vbo_builder is not None:
        skin_can_lbs = bool(
            anim_pose is not None
            and is_skin
            and (not is_bas_attachment or bas_root is not None)
            and getattr(node, "bone_map", None)
            and getattr(node, "skin_data", None)
        )
        vbo_world_pos = world_pos
        vbo_world_orient = world_orient
        if not is_skin or (is_bas_attachment and bas_root is None):
            vbo_world_pos = (0.0, 0.0, 0.0)
            vbo_world_orient = (0.0, 0.0, 0.0, 1.0)
        elif is_bas_attachment and not is_skin:
            vbo_world_pos = (0.0, 0.0, 0.0)
            vbo_world_orient = (0.0, 0.0, 0.0, 1.0)
        apply_skin_node_transform_for_bind = not skin_can_lbs
        if is_bas_attachment and is_skin:
            apply_skin_node_transform_for_bind = anim_pose is None
        cache_key = (
            _skinned_lbs_vbo_cache_key(
                node,
                vbo_builder,
                apply_skin_node_transform_for_bind=apply_skin_node_transform_for_bind,
            )
            if skin_can_lbs
            else _static_vbo_cache_key(node, vbo_builder)
        )
        cached_vbo = _get_skinned_lbs_vbo_cache(node, cache_key)
        if cached_vbo is not None:
            return (*cached_vbo, world_matrix)
        vdata, idx_arr = vbo_builder(
            node,
            vbo_world_pos,
            vbo_world_orient,
            anim_pose_node=None,
            apply_skin_node_transform_for_bind=apply_skin_node_transform_for_bind,
        )
        if vdata is not None:
            positions = np.asarray(vdata[:, 0:3], dtype=np.float32)
            normals = np.asarray(vdata[:, 3:6], dtype=np.float32) if vdata.shape[1] >= 6 else None
            uvs0 = np.asarray(vdata[:, 6:8], dtype=np.float32) if vdata.shape[1] >= 8 else None
            uvs1 = np.asarray(vdata[:, 8:10], dtype=np.float32) if vdata.shape[1] >= 10 else None
            indices = np.asarray(idx_arr, dtype=np.uint32) if idx_arr is not None and len(idx_arr) else None
            normals = smooth_render_normals(positions, normals, indices)
            bone_indices = None
            bone_weights = None
            if getattr(node, "is_skin", False) and vdata.shape[1] >= 22:
                bone_indices = np.rint(vdata[:, 14:18]).astype(np.uint16)
                bone_weights = np.asarray(vdata[:, 18:22], dtype=np.float32)
            if cache_key is not None:
                _set_skinned_lbs_vbo_cache(
                    node,
                    cache_key,
                    (positions, normals, uvs0, uvs1, indices, bone_indices, bone_weights),
                )
            return positions, normals, uvs0, uvs1, indices, bone_indices, bone_weights, world_matrix

    verts = np.asarray(getattr(node, "vertices", getattr(node, "verts", [])) or [], dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        return None, None, None, None, None, None, None, world_matrix
    normals = np.asarray(getattr(node, "normals", []) or [], dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) != len(verts):
        normals = None
    uvs0 = _node_uv_array(node, "uvs", len(verts))
    uvs1 = _node_uv_array(node, "uvs_lm", len(verts))
    faces = getattr(node, "faces", []) or []
    indices = np.asarray([int(i) for face in faces for i in tuple(face)[:3]], dtype=np.uint32)
    normals = smooth_render_normals(verts, normals, indices)
    return verts, normals, uvs0, uvs1, indices, None, None, world_matrix


def _skinned_lbs_vbo_cache_key(node, vbo_builder, *, apply_skin_node_transform_for_bind: bool):
    skin_data = getattr(node, "skin_data", None)
    composite_offset = getattr(node, "_composite_nonskin_offset", None)
    if composite_offset is not None:
        try:
            composite_offset = tuple(round(float(v), 6) for v in tuple(composite_offset)[:3])
        except Exception:
            composite_offset = None
    return (
        id(vbo_builder),
        _node_revision(node),
        len(skin_data or []),
        id(skin_data),
        tuple(getattr(node, "bone_map", []) or ()),
        composite_offset,
        bool(getattr(node, "_gr_bas_attachment_layer", False)),
        bool(apply_skin_node_transform_for_bind),
    )


def _static_vbo_cache_key(node, vbo_builder):
    if bool(getattr(node, "is_skin", False)):
        return None
    return (
        id(vbo_builder),
        "static",
        _node_revision(node),
        len(getattr(node, "vertices", getattr(node, "verts", [])) or []),
        len(getattr(node, "faces", []) or []),
        id(getattr(node, "vertices", getattr(node, "verts", [])) or None),
        id(getattr(node, "faces", []) or None),
        tuple(getattr(node, "bone_map", []) or ()),
        bool(getattr(node, "_gr_bas_attachment_layer", False)),
    )


def _get_skinned_lbs_vbo_cache(node, cache_key):
    if cache_key is None:
        return None
    try:
        cached = getattr(node, "_gr_render_lbs_vbo_cache", None)
    except Exception:
        return None
    if not isinstance(cached, dict) or cached.get("key") != cache_key:
        return None
    return cached.get("arrays")


def _set_skinned_lbs_vbo_cache(node, cache_key, arrays) -> None:
    if cache_key is None:
        return
    try:
        setattr(node, "_gr_render_lbs_vbo_cache", {"key": cache_key, "arrays": arrays})
    except Exception:
        pass


def smooth_render_normals(
    positions,
    normals=None,
    indices=None,
    *,
    crease_degrees: float = 62.0,
    position_epsilon: float = 1.0e-5,
):
    """Return render normals smoothed across compatible coincident vertices.

    KotOR binary meshes often duplicate positions at UV/lightmap seams. Averaging
    compatible duplicate normals removes artificial D3D faceting while the crease
    test preserves intentional hard edges, which is the smoothing-group behavior
    WGPU needs before buffer upload.
    """

    import numpy as np

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
        return normals

    authored = _coerce_normal_array(normals, len(pos))
    face_accum = _area_weighted_normal_accum(pos, indices)
    base = authored if authored is not None else face_accum
    base = _normalize_rows(base, fallback=face_accum)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    scale = 1.0 / max(float(position_epsilon), 1.0e-9)
    for idx, vertex in enumerate(pos):
        key = tuple(int(round(float(component) * scale)) for component in vertex[:3])
        buckets.setdefault(key, []).append(idx)

    cos_crease = math.cos(math.radians(max(0.0, min(180.0, float(crease_degrees)))))
    smoothed = base.copy()
    weights = np.linalg.norm(face_accum, axis=1)
    weights = np.where(weights > 1.0e-8, weights, 1.0)
    for members in buckets.values():
        if len(members) <= 1:
            continue
        member_normals = base[members]
        for member in members:
            ref = base[member]
            dots = member_normals @ ref
            compatible = [members[i] for i, dot in enumerate(dots) if float(dot) >= cos_crease]
            if len(compatible) <= 1:
                continue
            accum = np.zeros(3, dtype=np.float64)
            for other in compatible:
                accum += base[other].astype(np.float64) * float(weights[other])
            length = float(np.linalg.norm(accum))
            if length > 1.0e-8:
                smoothed[member] = (accum / length).astype(np.float32)

    return _normalize_rows(smoothed, fallback=base).astype(np.float32)


def _coerce_normal_array(normals, count: int):
    import numpy as np

    if normals is None:
        return None
    try:
        arr = np.asarray(normals, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) != count:
        return None
    arr = arr[:, :3]
    if not np.all(np.isfinite(arr)):
        arr = arr.copy()
        arr[~np.all(np.isfinite(arr), axis=1)] = 0.0
    if np.count_nonzero(np.linalg.norm(arr, axis=1) > 1.0e-8) == 0:
        return None
    return arr


def _area_weighted_normal_accum(positions, indices):
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    accum = np.zeros((len(pos), 3), dtype=np.float64)
    if indices is None:
        tri_indices = np.arange(len(pos), dtype=np.uint32)
    else:
        tri_indices = np.asarray(indices, dtype=np.uint32).reshape(-1)
    usable = (len(tri_indices) // 3) * 3
    for i0, i1, i2 in tri_indices[:usable].reshape((-1, 3)):
        if int(i0) >= len(pos) or int(i1) >= len(pos) or int(i2) >= len(pos):
            continue
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        v0 = pos[int(i0)]
        v1 = pos[int(i1)]
        v2 = pos[int(i2)]
        normal = np.cross(v1 - v0, v2 - v0)
        if float(np.dot(normal, normal)) <= 1.0e-18:
            continue
        accum[int(i0)] += normal
        accum[int(i1)] += normal
        accum[int(i2)] += normal
    return accum.astype(np.float32)


def _normalize_rows(values, *, fallback=None):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr
    fallback_arr = np.asarray(fallback, dtype=np.float32) if fallback is not None else None
    for idx in range(len(arr)):
        row = arr[idx]
        length = float(np.linalg.norm(row))
        if length <= 1.0e-8 and fallback_arr is not None and idx < len(fallback_arr):
            row = fallback_arr[idx]
            length = float(np.linalg.norm(row))
        if length <= 1.0e-8:
            arr[idx] = (0.0, 0.0, 1.0)
        else:
            arr[idx] = row / length
    return arr


def _extract_skinning(
    node,
    vertex_count: int,
    *,
    skeleton_id: int = 0,
    bone_indices=None,
    bone_weights=None,
):
    if bool(getattr(node, "_gr_bas_attachment_layer", False)):
        if not bool(getattr(node, "is_skin", False)):
            return SimpleNamespace(
                bone_indices=None,
                bone_weights=None,
                max_influences=0,
                skeleton_id=skeleton_id,
                skin_revision=0,
                bind_shape_matrix=None,
                is_skinned=False,
                warning="BAS attachment layers follow sockets outside body skinning",
            )
        palette_model = bas_attachment_palette_model_for_node(node)
        skeleton_id = id(palette_model) if palette_model is not None else skeleton_id
    try:
        from src.core.rendering.skeleton_render_data import extract_skinning_arrays

        return extract_skinning_arrays(
            node,
            vertex_count,
            skeleton_id=skeleton_id,
            bone_indices=bone_indices,
            bone_weights=bone_weights,
        )
    except Exception:
        return SimpleNamespace(
            bone_indices=None,
            bone_weights=None,
            max_influences=0,
            skeleton_id=skeleton_id,
            skin_revision=0,
            bind_shape_matrix=None,
            is_skinned=False,
            warning="skin adapter unavailable",
        )


def bas_attachment_palette_model_for_node(node):
    """Return an attachment-local pseudo-model for BAS head/gear skin palettes."""

    root = _bas_attachment_root_for_node(node)
    if root is None:
        return None
    cached = getattr(root, "_gr_bas_attachment_palette_model_cache", None)
    if cached is not None:
        return cached
    nodes = _bas_attachment_subtree_nodes(root)
    if not nodes:
        return None
    model = SimpleNamespace(
        name=str(getattr(root, "name", "") or "bas_attachment"),
        supermodel=str(getattr(root, "supermodel", "") or ""),
        anim_scale=float(getattr(root, "anim_scale", 1.0) or 1.0),
        root_node=root,
        all_nodes=lambda: list(nodes),
        _gr_bas_attachment_palette_model=True,
    )
    try:
        setattr(root, "_gr_bas_attachment_palette_model_cache", model)
    except Exception:
        pass
    return model


def mesh_model_matrix_for_node(node, *, anim_pose=None):
    """Return the world/model matrix a renderer should apply to one mesh node."""

    if _node_vertices_are_model_world(node) and not bool(getattr(node, "_gr_bas_attachment_layer", False)):
        import numpy as np

        return np.eye(4, dtype=np.float32)
    anim_pose = _effective_animation_pose_for_node(node, anim_pose)
    if bool(getattr(node, "_gr_bas_attachment_layer", False)):
        root = _bas_attachment_root_for_node(node)
        if root is not None and bool(getattr(node, "is_skin", False)):
            return node_world_matrix(root, anim_pose=anim_pose)
    return node_world_matrix(node, anim_pose=anim_pose)


def _effective_animation_pose_for_node(node, anim_pose):
    node_anim_pose = animation_pose_for_node(node, anim_pose)
    node_anim_pose = _bas_attachment_effective_pose_for_node(node, node_anim_pose)
    if node_anim_pose is not None and not animation_pose_applies_to_node(node, node_anim_pose):
        return None
    return node_anim_pose


def _bas_attachment_subtree_nodes(root) -> list:
    nodes = []
    stack = [root]
    visited = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        nodes.append(current)
        stack.extend(reversed(getattr(current, "children", []) or []))
    return nodes


def texture_image_to_rgba8(texture_data: TextureRenderData | None) -> tuple[int, int, bytes] | None:
    """Return bottom-up RGBA8 bytes for WGPU upload.

    The viewport TextureCache and ResourceManager already normalize KotOR TPC/TGA
    images to bottom-up orientation. WGPU samples with the same UV convention as
    the ModernGL path, so this adapter does not flip rows during upload.
    """

    if texture_data is None or texture_data.source is None:
        return None
    try:
        image = texture_data.source
        rgba = image.convert("RGBA") if hasattr(image, "convert") else None
        if rgba is None:
            return None
        width, height = rgba.size
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height), rgba.tobytes()
    except Exception:
        return None


def _node_uv_array(node, attr: str, count: int):
    import numpy as np

    raw = getattr(node, attr, []) or []
    if not raw:
        return np.full((count, 2), 0.5, dtype=np.float32)
    try:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.full((count, 2), 0.5, dtype=np.float32)
        arr = arr[:, :2]
        if len(arr) < count:
            arr = np.vstack([arr, np.full((count - len(arr), 2), 0.5, dtype=np.float32)])
        arr = arr[:count]
        if attr == "uvs" and not bool(getattr(node, "uv_v_flip", True)):
            # WGPU's current shader applies the KotOR/D3D V flip
            # unconditionally. Imported DCC meshes keep bottom-left UVs, so
            # pre-flip them here to make the net WGPU path match ModernGL.
            arr = arr.copy()
            arr[:, 1] = 1.0 - arr[:, 1]
        return arr
    except Exception:
        return np.full((count, 2), 0.5, dtype=np.float32)


def _material_data(node, textures: dict) -> MaterialRenderData:
    diffuse_name = _clean_tex_name(getattr(node, "texture", "") or "")
    lightmap_name = _clean_tex_name(
        getattr(node, "_gr_baked_lightmap_preview_name", "")
        or getattr(node, "lightmap", "")
        or ""
    )
    diffuse_texture = _texture_data(diffuse_name, textures)
    lightmap_texture = None
    if _node_has_lightmap(node, lightmap_name):
        lightmap_texture = _texture_data(lightmap_name, textures)
    base_color = _material_color(node)
    alpha_cutoff = _clamp01(float(getattr(node, "txi_alpha_test", 0.0) or 0.0) or 0.5)
    alpha_mode = _alpha_mode(node)
    blend_mode = _blend_mode(node)
    sprite_alpha_source = _sprite_alpha_source(node)
    sprite_glow = _sprite_glow(node)
    double_sided = bool(
        getattr(node, "is_dangly", False)
        or int(getattr(node, "transparency_hint", 0) or 0) in (1, 2)
        or alpha_mode == "BLEND"
    )
    material_id = "|".join(
        [
            str(id(node)),
            diffuse_name,
            lightmap_name,
            alpha_mode,
            blend_mode,
            f"{alpha_cutoff:.3f}",
            str(sprite_alpha_source),
            f"{sprite_glow:.3f}",
            str(double_sided),
        ]
    )
    material_rev = (
        int(getattr(node, "_gr_revision", 0) or 0),
        id(diffuse_texture.source) if diffuse_texture is not None else 0,
        id(lightmap_texture.source) if lightmap_texture is not None else 0,
        hash((base_color, alpha_mode, blend_mode, alpha_cutoff, sprite_alpha_source, round(sprite_glow, 3), double_sided)),
    )
    return MaterialRenderData(
        material_id=material_id,
        name=str(getattr(node, "name", "") or material_id),
        diffuse_texture_id=diffuse_texture.texture_id if diffuse_texture is not None else "",
        diffuse_texture_path=diffuse_name,
        diffuse_texture_data=diffuse_texture,
        lightmap_texture_id=lightmap_texture.texture_id if lightmap_texture is not None else "",
        lightmap_texture_path=lightmap_name,
        lightmap_texture_data=lightmap_texture,
        base_color_rgba=base_color,
        alpha_mode=alpha_mode,
        alpha_cutoff=alpha_cutoff,
        blend_mode=blend_mode,
        sprite_alpha_source=sprite_alpha_source,
        sprite_glow=sprite_glow,
        double_sided=double_sided,
        unlit=(
            any(abs(float(c)) > 1e-6 for c in tuple(getattr(node, "selfillum", (0.0, 0.0, 0.0)) or ())[:3])
            or sprite_glow > 0.001
        ),
        has_transparency=alpha_mode in {"MASK", "CUTOUT", "BLEND"} or base_color[3] < 0.999,
        source_revision=material_rev,
    )


def _texture_data(name: str, textures: dict) -> TextureRenderData | None:
    if not name:
        return None
    source = textures.get(name.lower()) or textures.get(name)
    if source is None:
        return TextureRenderData(
            texture_id=name.lower(),
            name=name,
            source=None,
            source_revision=(0, 0, 0),
        )
    width, height = getattr(source, "size", (0, 0))
    revision = (id(source), int(width or 0), int(height or 0))
    return TextureRenderData(
        texture_id=f"{name.lower()}:{revision[0]}",
        name=name.lower(),
        source=source,
        source_revision=revision,
    )


def _node_has_lightmap(node, lightmap_name: str) -> bool:
    if not lightmap_name:
        return False
    if len(getattr(node, "uvs_lm", []) or []) <= 0:
        return False
    return bool(
        getattr(node, "has_lightmap", False)
        or getattr(node, "_gr_baked_lightmap_preview_name", "")
        or int(getattr(node, "tex_count", 1) or 1) >= 2
    )


def _alpha_mode(node) -> str:
    raw_node_alpha = getattr(node, "alpha", None)
    node_alpha = _clamp01(float(1.0 if raw_node_alpha is None else raw_node_alpha))
    txi_blend = int(getattr(node, "txi_blending", 0) or 0)
    alpha_test = float(getattr(node, "txi_alpha_test", 0.0) or 0.0)
    transparency_hint = int(getattr(node, "transparency_hint", 0) or 0)
    sprite_alpha = _sprite_alpha_source(node)
    if txi_blend == 2 or (txi_blend == 0 and transparency_hint > 0):
        return "MASK"
    if (
        node_alpha < 0.999
        or float(getattr(node, "txi_wateralpha", 1.0) or 1.0) < 0.999
        or bool(getattr(node, "txi_decal", False))
        or txi_blend in (1, 3)
        or sprite_alpha
    ):
        return "BLEND"
    return "OPAQUE"


def _blend_mode(node) -> str:
    explicit = str(getattr(node, "_gr_sprite_render_mode", "") or "").lower()
    if explicit == "additive":
        return "ADDITIVE"
    if explicit == "lighten":
        return "LIGHTEN"
    if explicit in {"opaque", "cutout", "blend"}:
        return "ALPHA"
    txi_blend = int(getattr(node, "txi_blending", 0) or 0)
    if _sprite_alpha_source(node) and _sprite_glow(node) > 0.001:
        return "LIGHTEN"
    if txi_blend == 1:
        return "ADDITIVE"
    if txi_blend == 3:
        return "LIGHTEN"
    return "ALPHA"


def _sprite_alpha_source(node) -> int:
    source = str(getattr(node, "_gr_sprite_alpha_source", "") or "").lower()
    if source in {"luminance", "brightness", "matte", "black_key"}:
        return 1
    if _has_sprite_material_override(node):
        return 0
    if _is_saber_hilt(node):
        return 0
    text = f"{getattr(node, 'name', '')} {getattr(node, 'texture', '')}".lower()
    return 1 if any(token in text for token in ("saber", "sabre", "lsabre", "blade", "glow", "flare", "beam")) else 0


def _sprite_glow(node) -> float:
    explicit = getattr(node, "_gr_sprite_glow", None)
    if explicit is not None:
        try:
            return max(0.0, min(4.0, float(explicit)))
        except Exception:
            return 0.0
    if _is_saber_hilt(node):
        return 0.0
    text = f"{getattr(node, 'name', '')} {getattr(node, 'texture', '')}".lower()
    return 1.6 if any(token in text for token in ("saber", "sabre", "lsabre", "blade", "glow", "flare", "beam")) else 0.0


def _is_saber_hilt(node) -> bool:
    if str(getattr(node, "_gr_sprite_category", "") or "").lower() == "hilt":
        return True
    name = str(getattr(node, "name", "") or "").lower()
    texture = str(getattr(node, "texture", "") or "").lower()
    if texture.startswith(("w_lghtsbr", "w_shortsbr", "w_dblsbr")):
        return True
    return name.startswith(("lghtsbr", "lshandle")) or "handle" in name


def _has_sprite_material_override(node) -> bool:
    return bool(
        str(getattr(node, "_gr_sprite_category", "") or "").strip()
        or str(getattr(node, "_gr_sprite_render_mode", "") or "").strip()
        or getattr(node, "_gr_sprite_glow", None) is not None
        or str(getattr(node, "_gr_sprite_alpha_source", "") or "").strip()
    )


def _node_world_transform(node, *, anim_pose=None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    bas_root = _bas_attachment_root_for_node(node)
    if bas_root is not None:
        try:
            return _bas_attachment_world_transform(node, bas_root, anim_pose=anim_pose)
        except Exception:
            pass
    if anim_pose is not None:
        try:
            return _animated_node_world_transform(node, anim_pose)
        except Exception:
            pass
    try:
        wp, wo = node.world_transform()
        return tuple(float(v) for v in wp[:3]), tuple(float(v) for v in wo[:4])
    except Exception:
        try:
            wp = node.world_position()
            return tuple(float(v) for v in wp[:3]), (0.0, 0.0, 0.0, 1.0)
        except Exception:
            # Lightweight runtime/test nodes do not always implement the model
            # convenience methods.  Compose their explicit local transform
            # through the parent chain instead of dropping socket placement.
            try:
                from src.core.geometry.model_data import _quat_mul, _quat_normalize_bind, _quat_rotate

                parent = getattr(node, "parent", None)
                local_pos = tuple(float(v) for v in getattr(node, "position", (0.0, 0.0, 0.0))[:3])
                local_rot = tuple(float(v) for v in getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0))[:4])
                local_rot = tuple(_quat_normalize_bind(local_rot))
                if parent is None:
                    return local_pos, local_rot
                parent_pos, parent_rot = _node_world_transform(parent, anim_pose=anim_pose)
                offset = _quat_rotate(parent_rot, local_pos)
                return (
                    float(parent_pos[0]) + float(offset[0]),
                    float(parent_pos[1]) + float(offset[1]),
                    float(parent_pos[2]) + float(offset[2]),
                ), tuple(_quat_mul(parent_rot, local_rot))
            except Exception:
                return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)


def node_world_matrix(node, *, anim_pose=None):
    import numpy as np

    pos, quat = _node_world_transform(node, anim_pose=anim_pose)
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    length_sq = x * x + y * y + z * z + w * w
    if length_sq > 1e-9:
        inv_len = 1.0 / math.sqrt(length_sq)
        x, y, z, w = x * inv_len, y * inv_len, z * inv_len, w * inv_len
    else:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), float(pos[0])],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), float(pos[1])],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), float(pos[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix


def _bas_attachment_root_for_node(node):
    current = node
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if bool(getattr(current, "_gr_bas_attachment_root", False)):
            return current
        current = getattr(current, "parent", None)
    return None


def _bas_attachment_effective_pose_for_node(node, anim_pose):
    if node is None or anim_pose is None or not bool(getattr(node, "_gr_bas_attachment_layer", False)):
        return anim_pose
    root = _bas_attachment_root_for_node(node)
    if root is None:
        return anim_pose
    try:
        pose_source_id = int(getattr(anim_pose, "_gr_animation_source_model_id", 0) or 0)
        attachment_source_id = int(getattr(root, "_gr_bas_attachment_source_model_id", 0) or 0)
        if pose_source_id and attachment_source_id and pose_source_id == attachment_source_id:
            return anim_pose
    except Exception:
        pass
    slot = str(getattr(root, "_gr_bas_attachment_slot", "") or "").strip().lower()
    if slot != "head":
        return anim_pose
    source_model = getattr(root, "_gr_bas_attachment_source_model_ref", None)
    anim_name = str(getattr(anim_pose, "_gr_animation_name", "") or "").strip()
    if source_model is None or not anim_name:
        return anim_pose
    time_value = float(getattr(anim_pose, "time", 0.0) or 0.0)
    cache_key = (id(anim_pose), id(source_model), anim_name.lower(), round(time_value, 6))
    cached = getattr(root, "_gr_bas_attachment_effective_pose_cache", None)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        pose = cached.get("pose")
        if pose is not None:
            return pose
    try:
        from src.core.animation.animation_engine import AnimationEngine

        engine = AnimationEngine(source_model)
        if not engine.play(anim_name, loop=True, blend=False):
            return anim_pose
        current = getattr(engine, "current_animation", None)
        length = float(getattr(current, "length", 0.0) or 0.0)
        sample_time = time_value
        if length > 0.0:
            sample_time = sample_time % length
        pose = engine.evaluate(sample_time)
        try:
            pose.nodes = {
                str(name).lower(): pose_node
                for name, pose_node in (getattr(pose, "nodes", {}) or {}).items()
                if _bas_head_attachment_local_pose_node_allowed_name(str(name))
            }
        except Exception:
            pass
        setattr(pose, "_gr_animation_source_model_id", id(source_model))
        setattr(pose, "_gr_animation_source_model_name", str(getattr(source_model, "name", "") or ""))
        setattr(pose, "_gr_animation_name", anim_name)
        setattr(pose, "_gr_bas_socket_pose", anim_pose)
        try:
            setattr(root, "_gr_bas_attachment_effective_pose_cache", {"key": cache_key, "pose": pose})
        except Exception:
            pass
        return pose
    except Exception:
        return anim_pose


def _bas_attachment_socket_node(bas_root):
    socket_name = str(getattr(bas_root, "_gr_bas_socket_name", "") or "").lower()
    body_root = getattr(bas_root, "parent", None)
    if body_root is None or not socket_name:
        return None
    if str(getattr(body_root, "name", "") or "").lower() == socket_name:
        return body_root
    stack = [body_root]
    visited = {id(bas_root)}
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if str(getattr(current, "name", "") or "").lower() == socket_name:
            return current
        for child in reversed(getattr(current, "children", []) or []):
            if bool(getattr(child, "_gr_bas_attachment_root", False)):
                continue
            stack.append(child)
    return None


def _bas_attachment_world_transform(node, bas_root, *, anim_pose=None, transform_cache=None):
    from src.core.geometry.model_data import (
        _quat_mul,
        _quat_normalize,
        _quat_normalize_bind,
        _quat_rotate,
    )

    # Retained renderers can ask for every node in the same detachable BAS head
    # during one frame.  The uncached, renderer-neutral path below deliberately
    # rebuilds the complete root chain, but doing that for each eye/face/skin
    # node repeatedly rescans the same head pose and socket hierarchy.  Keep the
    # cache caller-owned so animation-pose and scene changes invalidate it at the
    # next frame boundary without persistent scene state.
    if transform_cache is not None:
        state_key = ("bas_attachment_root_pose", id(bas_root), id(anim_pose))
        state = transform_cache.get(state_key)
        if not (
            isinstance(state, dict)
            and state.get("root") is bas_root
            and state.get("pose") is anim_pose
        ):
            socket = _bas_attachment_socket_node(bas_root)
            socket_pose = getattr(anim_pose, "_gr_bas_socket_pose", anim_pose)
            if socket is not None:
                socket_wp, socket_wo = _node_world_transform(socket, anim_pose=socket_pose)
                socket_base = (
                    tuple(float(value) for value in socket_wp[:3]),
                    tuple(float(value) for value in socket_wo[:4]),
                )
            else:
                socket_base = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            state = {
                "root": bas_root,
                "pose": anim_pose,
                "socket_base": socket_base,
                "pose_mode": _bas_attachment_pose_mode_for_root(bas_root, anim_pose),
                "bind_states": {},
            }
            transform_cache[state_key] = state

        # Validate the BAS chain while arranging it root-first.  If a malformed
        # attachment does not actually reach its tagged root, retain the legacy
        # full-chain fallback below instead of caching a different transform.
        chain = []
        current = node
        visited: set[int] = set()
        while current is not None:
            current_id = id(current)
            if current_id in visited or len(chain) > 512:
                break
            visited.add(current_id)
            chain.append(current)
            if current is bas_root:
                break
            current = getattr(current, "parent", None)

        if chain and chain[-1] is bas_root:
            chain.reverse()
            pose_mode = str(state["pose_mode"] or "")
            bind_states = state["bind_states"]
            (base_position, base_orientation) = state["socket_base"]
            wx, wy, wz = base_position
            parent_orientation = base_orientation
            start_index = 0

            # A cached bind-state is the exact root-to-ancestor prefix needed by
            # the original full-chain calculation.  It is intentionally distinct
            # from that ancestor's leaf result because Odyssey's pure-X bind
            # quaternion normalization differs for ancestors and leaves.
            for index in range(len(chain) - 2, -1, -1):
                cached_prefix = bind_states.get(id(chain[index]))
                if cached_prefix is None:
                    continue
                (cached_position, cached_orientation) = cached_prefix
                wx, wy, wz = cached_position
                parent_orientation = cached_orientation
                start_index = index + 1
                break

            for chain_node in chain[start_index:-1]:
                pose_node = None
                if pose_mode == "source_local" or (
                    pose_mode == "inherited_head_local"
                    and _bas_inherited_head_pose_node_allowed(chain_node)
                ):
                    pose_node = _pose_node_for_transform(chain_node, anim_pose)
                if pose_node is not None:
                    lx, ly, lz = getattr(
                        pose_node,
                        "position",
                        getattr(chain_node, "position", (0.0, 0.0, 0.0)),
                    )
                    if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)):
                        lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
                    rot = list(
                        getattr(
                            pose_node,
                            "rotation",
                            getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)),
                        )
                    )
                    if not all(math.isfinite(value) for value in rot):
                        rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
                else:
                    lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
                    rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
                node_rot = _quat_normalize_bind(rot)
                rx, ry, rz = _quat_rotate(parent_orientation, (lx, ly, lz))
                wx += rx
                wy += ry
                wz += rz
                parent_orientation = _quat_mul(parent_orientation, node_rot)
                bind_states[id(chain_node)] = (
                    (float(wx), float(wy), float(wz)),
                    tuple(float(value) for value in parent_orientation[:4]),
                )

            leaf = chain[-1]
            pose_node = None
            if pose_mode == "source_local" or (
                pose_mode == "inherited_head_local"
                and _bas_inherited_head_pose_node_allowed(leaf)
            ):
                pose_node = _pose_node_for_transform(leaf, anim_pose)
            if pose_node is not None:
                lx, ly, lz = getattr(
                    pose_node,
                    "position",
                    getattr(leaf, "position", (0.0, 0.0, 0.0)),
                )
                if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)):
                    lx, ly, lz = getattr(leaf, "position", (0.0, 0.0, 0.0))
                rot = list(
                    getattr(
                        pose_node,
                        "rotation",
                        getattr(leaf, "rotation", (0.0, 0.0, 0.0, 1.0)),
                    )
                )
                if not all(math.isfinite(value) for value in rot):
                    rot = list(getattr(leaf, "rotation", (0.0, 0.0, 0.0, 1.0)))
            else:
                lx, ly, lz = getattr(leaf, "position", (0.0, 0.0, 0.0))
                rot = list(getattr(leaf, "rotation", (0.0, 0.0, 0.0, 1.0)))
            node_rot = _quat_normalize(rot)
            rx, ry, rz = _quat_rotate(parent_orientation, (lx, ly, lz))
            wx += rx
            wy += ry
            wz += rz
            parent_orientation = _quat_mul(parent_orientation, node_rot)

            length_sq = sum(float(value) * float(value) for value in parent_orientation[:4])
            if length_sq > 1e-9 and abs(length_sq - 1.0) > 1e-4:
                scale = 1.0 / math.sqrt(length_sq)
                parent_orientation = [float(value) * scale for value in parent_orientation[:4]]
            return (
                (float(wx), float(wy), float(wz)),
                tuple(float(value) for value in parent_orientation[:4]),
            )

    socket = _bas_attachment_socket_node(bas_root)
    socket_pose = getattr(anim_pose, "_gr_bas_socket_pose", anim_pose)
    if socket is not None:
        socket_wp, socket_wo = _node_world_transform(socket, anim_pose=socket_pose)
        wx, wy, wz = socket_wp
        parent_orientation = list(socket_wo)
    else:
        wx = wy = wz = 0.0
        parent_orientation = [0.0, 0.0, 0.0, 1.0]

    chain = []
    current = node
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited or len(chain) > 512:
            break
        visited.add(id(current))
        chain.append(current)
        if current is bas_root:
            break
        current = getattr(current, "parent", None)
    chain.reverse()

    pose_mode = _bas_attachment_pose_mode_for_root(bas_root, anim_pose)
    last_i = len(chain) - 1
    for index, chain_node in enumerate(chain):
        is_leaf = index == last_i
        pose_node = None
        if pose_mode == "source_local" or (pose_mode == "inherited_head_local" and _bas_inherited_head_pose_node_allowed(chain_node)):
            pose_node = _pose_node_for_transform(chain_node, anim_pose)
        if pose_node is not None:
            lx, ly, lz = getattr(pose_node, "position", getattr(chain_node, "position", (0.0, 0.0, 0.0)))
            if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)):
                lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
            rot = list(getattr(pose_node, "rotation", getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0))))
            if not all(math.isfinite(v) for v in rot):
                rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
        else:
            lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
            rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
        node_rot = _quat_normalize(rot) if is_leaf else _quat_normalize_bind(rot)
        rx, ry, rz = _quat_rotate(parent_orientation, (lx, ly, lz))
        wx += rx
        wy += ry
        wz += rz
        parent_orientation = _quat_mul(parent_orientation, node_rot)

    length_sq = sum(float(v) * float(v) for v in parent_orientation[:4])
    if length_sq > 1e-9 and abs(length_sq - 1.0) > 1e-4:
        scale = 1.0 / math.sqrt(length_sq)
        parent_orientation = [float(v) * scale for v in parent_orientation[:4]]
    return (float(wx), float(wy), float(wz)), tuple(float(v) for v in parent_orientation[:4])


def _bas_attachment_pose_mode_for_root(bas_root, anim_pose) -> str:
    if bas_root is None or anim_pose is None:
        return ""
    try:
        pose_source_id = int(getattr(anim_pose, "_gr_animation_source_model_id", 0) or 0)
        attachment_source_id = int(getattr(bas_root, "_gr_bas_attachment_source_model_id", 0) or 0)
        if pose_source_id and attachment_source_id and pose_source_id == attachment_source_id:
            return "source_local"
    except Exception:
        pass
    pose_source_name = str(getattr(anim_pose, "_gr_animation_source_model_name", "") or "").strip().lower()
    attachment_source_name = str(getattr(bas_root, "_gr_bas_attachment_source_model_name", "") or "").strip().lower()
    if pose_source_name and attachment_source_name and pose_source_name == attachment_source_name:
        return "source_local"
    if _bas_head_attachment_has_inherited_pose_tracks(bas_root, anim_pose):
        return "inherited_head_local"
    return ""


def _bas_attachment_pose_applies_to_root(bas_root, anim_pose) -> bool:
    return bool(_bas_attachment_pose_mode_for_root(bas_root, anim_pose))


def _bas_head_attachment_has_inherited_pose_tracks(bas_root, anim_pose) -> bool:
    if bas_root is None or anim_pose is None:
        return False
    slot = str(getattr(bas_root, "_gr_bas_attachment_slot", "") or "").strip().lower()
    if slot != "head":
        return False
    pose_names = {str(name or "").lower() for name in (getattr(anim_pose, "nodes", {}) or {}).keys()}
    if not pose_names:
        return False
    head_track_names = set()
    stack = [bas_root]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        name = str(getattr(current, "name", "") or "").lower()
        if name:
            head_track_names.add(name)
        stack.extend(getattr(current, "children", []) or [])
    if not head_track_names:
        return False
    return bool(pose_names & head_track_names)


def _bas_inherited_head_pose_node_allowed(node) -> bool:
    name = str(getattr(node, "name", "") or "").strip().lower()
    return _bas_head_attachment_local_pose_node_allowed_name(name)


def _bas_head_attachment_local_pose_node_allowed_name(name: str) -> bool:
    name = str(name or "").strip().lower()
    if not name:
        return False
    if name in {"necklwr_g", "neck_g", "hturn_g", "head_g", "talkdummy"}:
        return True
    if name.startswith("f_"):
        return True
    if name.startswith("eye"):
        return True
    if name.startswith("teeth") or name.startswith("tongue"):
        return True
    return False


def _pose_node_for_transform(node, anim_pose):
    if node is None or anim_pose is None:
        return None
    anim_pose = animation_pose_for_node(node, anim_pose)
    if anim_pose is None:
        return None
    if not animation_pose_applies_to_node(node, anim_pose):
        return None
    pose_nodes = getattr(anim_pose, "nodes", {}) or {}
    pose_nodes_by_index = getattr(anim_pose, "nodes_by_index", {}) or {}
    duplicate_node_names = set(getattr(anim_pose, "duplicate_node_names", set()) or set())
    try:
        pose_node = pose_nodes_by_index.get(int(getattr(node, "index", -1)))
    except Exception:
        pose_node = None
    if pose_node is not None:
        return pose_node
    node_name_key = str(getattr(node, "name", "") or "").lower()
    if node_name_key and node_name_key not in duplicate_node_names:
        return pose_nodes.get(node_name_key)
    return None


def _scene_object_id_for_node(node) -> str:
    current = node
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        object_id = str(getattr(current, "_gr_scene_object_id", "") or "")
        if object_id:
            return object_id
        current = getattr(current, "parent", None)
    return ""


def _scene_import_id_for_node(node) -> str:
    current = node
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        import_id = str(getattr(current, "_gr_scene_import_id", "") or "")
        if import_id:
            return import_id
        current = getattr(current, "parent", None)
    return ""


def _source_model_id_for_node(node) -> int:
    current = node
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        for attr in ("_gr_bas_attachment_source_model_id", "_gr_source_model_id", "_gr_runtime_source_model_id"):
            try:
                source_id = int(getattr(current, attr, 0) or 0)
            except Exception:
                source_id = 0
            if source_id:
                return source_id
        current = getattr(current, "parent", None)
    return 0


def runtime_source_model_for_node(node):
    """Return the runtime actor model that owns ``node``'s skin palette.

    PIE actors live inside a much larger resident map model.  Their qBone/tBone
    rows and DFS indices still belong to the original character model, so a
    renderer must not infer a palette from the map hierarchy.  The reference is
    stored on the runtime wrapper and resolved through ancestors to avoid
    retaining a duplicate strong reference on every copied Odyssey node.
    """

    current = node
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        source_model = getattr(current, "_gr_runtime_source_model_ref", None)
        if source_model is not None:
            return source_model
        current = getattr(current, "parent", None)
    return None


def animation_pose_for_node(node, anim_pose):
    """Return the single character pose allowed to drive ``node``."""

    if node is None or anim_pose is None:
        return None
    if bool(getattr(anim_pose, "_gr_animation_pose_set", False)):
        selector = getattr(anim_pose, "pose_for_node", None)
        return selector(node) if callable(selector) else None
    return anim_pose


def animation_pose_applies_to_node(node, anim_pose) -> bool:
    """Return whether an animation pose is allowed to drive this runtime node."""

    if node is None or anim_pose is None:
        return False
    scoped_pose = animation_pose_for_node(node, anim_pose)
    if scoped_pose is not anim_pose:
        return scoped_pose is not None
    return _animation_pose_matches_node_identity(node, anim_pose)


def _animation_pose_matches_node_identity(
    node,
    anim_pose,
    *,
    node_object_id: str | None = None,
    node_import_id: str | None = None,
    node_source_id: int | None = None,
) -> bool:
    pose_object_id = str(getattr(anim_pose, "_gr_animation_scene_object_id", "") or "")
    if pose_object_id:
        node_object_id = _scene_object_id_for_node(node) if node_object_id is None else node_object_id
        if node_object_id and node_object_id != pose_object_id:
            return False
        if node_object_id:
            return True
    pose_import_id = str(getattr(anim_pose, "_gr_animation_scene_import_id", "") or "")
    if pose_import_id:
        node_import_id = _scene_import_id_for_node(node) if node_import_id is None else node_import_id
        if node_import_id and node_import_id != pose_import_id:
            return False
        if node_import_id:
            return True
    try:
        pose_source_id = int(getattr(anim_pose, "_gr_animation_source_model_id", 0) or 0)
    except Exception:
        pose_source_id = 0
    node_source_id = _source_model_id_for_node(node) if node_source_id is None else node_source_id
    if pose_source_id and node_source_id and pose_source_id != node_source_id:
        return False
    return True


def _bas_attachment_local_transform(node, bas_root):
    from src.core.geometry.model_data import (
        _quat_mul,
        _quat_normalize,
        _quat_normalize_bind,
        _quat_rotate,
    )

    wx = wy = wz = 0.0
    parent_orientation = [0.0, 0.0, 0.0, 1.0]
    chain = []
    current = node
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited or len(chain) > 512:
            break
        visited.add(id(current))
        chain.append(current)
        if current is bas_root:
            break
        current = getattr(current, "parent", None)
    chain.reverse()

    last_i = len(chain) - 1
    for index, chain_node in enumerate(chain):
        is_leaf = index == last_i
        lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
        rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
        node_rot = _quat_normalize(rot) if is_leaf else _quat_normalize_bind(rot)
        rx, ry, rz = _quat_rotate(parent_orientation, (lx, ly, lz))
        wx += rx
        wy += ry
        wz += rz
        parent_orientation = _quat_mul(parent_orientation, node_rot)

    length_sq = sum(float(v) * float(v) for v in parent_orientation[:4])
    if length_sq > 1e-9 and abs(length_sq - 1.0) > 1e-4:
        scale = 1.0 / math.sqrt(length_sq)
        parent_orientation = [float(v) * scale for v in parent_orientation[:4]]
    return (float(wx), float(wy), float(wz)), tuple(float(v) for v in parent_orientation[:4])


def _animated_world_cache_key_from_chain(chain) -> tuple:
    key_parts = []
    for chain_node in chain:
        try:
            position = tuple(round(float(v), 6) for v in tuple(getattr(chain_node, "position", (0.0, 0.0, 0.0)))[:3])
        except Exception:
            position = (0.0, 0.0, 0.0)
        try:
            rotation = tuple(round(float(v), 6) for v in tuple(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))[:4])
        except Exception:
            rotation = (0.0, 0.0, 0.0, 1.0)
        try:
            source_position = tuple(
                round(float(v), 6)
                for v in tuple(getattr(chain_node, "_gr_scene_source_position", position))[:3]
            )
        except Exception:
            source_position = position
        key_parts.append(
            (
                id(chain_node),
                int(getattr(chain_node, "_gr_revision", 0) or 0),
                bool(getattr(chain_node, "_gr_scene_object_root", False)),
                position,
                source_position,
                rotation,
            )
        )
    return tuple(key_parts)


def _animated_pose_cache_stamp(anim_pose) -> tuple:
    def _stamp_value(name: str, fallback=0.0):
        try:
            return round(float(getattr(anim_pose, name, fallback) or fallback), 6)
        except Exception:
            return fallback

    return (
        _stamp_value("time"),
        _stamp_value("current_time"),
        _stamp_value("_gr_animation_time"),
        int(getattr(anim_pose, "_gr_revision", 0) or 0),
        str(getattr(anim_pose, "_gr_animation_name", "") or ""),
    )


def _animated_node_world_transform(node, anim_pose) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    import math

    from src.core.geometry.model_data import (
        _quat_mul,
        _quat_normalize,
        _quat_normalize_bind,
        _quat_rotate,
    )

    cache = getattr(anim_pose, "_gr_mesh_world_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(anim_pose, "_gr_mesh_world_cache", cache)
        except Exception:
            pass
    chain = []
    current = node
    visited: set[int] = set()
    while current is not None:
        node_id = id(current)
        if node_id in visited or len(chain) > 512:
            break
        visited.add(node_id)
        chain.append(current)
        current = getattr(current, "parent", None)
    chain.reverse()
    cache_key = (id(node), _animated_pose_cache_stamp(anim_pose), _animated_world_cache_key_from_chain(chain))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    wx = wy = wz = 0.0
    parent_orientation = [0.0, 0.0, 0.0, 1.0]
    last_i = len(chain) - 1
    for index, chain_node in enumerate(chain):
        is_leaf = index == last_i
        pose_node = _pose_node_for_transform(chain_node, anim_pose)
        if pose_node is not None:
            lx, ly, lz = getattr(pose_node, "position", getattr(chain_node, "position", (0.0, 0.0, 0.0)))
            if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)):
                lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
            if bool(getattr(chain_node, "_gr_scene_object_root", False)):
                scene_pos = tuple(float(v) for v in getattr(chain_node, "position", (0.0, 0.0, 0.0))[:3])
                source_pos = tuple(float(v) for v in getattr(chain_node, "_gr_scene_source_position", scene_pos)[:3])
                lx = scene_pos[0] + (float(lx) - source_pos[0])
                ly = scene_pos[1] + (float(ly) - source_pos[1])
                lz = scene_pos[2] + (float(lz) - source_pos[2])
            rot = list(getattr(pose_node, "rotation", getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0))))
            if not all(math.isfinite(v) for v in rot):
                rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
            if is_leaf:
                length_sq = rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3]
                node_rot = [rot[0] / math.sqrt(length_sq), rot[1] / math.sqrt(length_sq), rot[2] / math.sqrt(length_sq), rot[3] / math.sqrt(length_sq)] if length_sq > 1e-9 else [0.0, 0.0, 0.0, 1.0]
            else:
                node_rot = _quat_normalize_bind(rot)
        else:
            lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
            rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
            node_rot = _quat_normalize(rot) if is_leaf else _quat_normalize_bind(rot)

        rx, ry, rz = _quat_rotate(parent_orientation, (lx, ly, lz))
        wx += rx
        wy += ry
        wz += rz
        parent_orientation = _quat_mul(parent_orientation, node_rot)

    if not (math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz)):
        wp, wo = node.world_transform()
        result = (tuple(float(v) for v in wp[:3]), tuple(float(v) for v in wo[:4]))
    else:
        wo = tuple(parent_orientation)
        length_sq = sum(float(v) * float(v) for v in wo[:4])
        if length_sq > 1e-9 and abs(length_sq - 1.0) > 1e-4:
            scale = 1.0 / math.sqrt(length_sq)
            wo = tuple(float(v) * scale for v in wo[:4])
        result = ((float(wx), float(wy), float(wz)), tuple(float(v) for v in wo[:4]))
    cache[cache_key] = result
    return result


def _material_color(node) -> tuple[float, float, float, float]:
    raw = getattr(node, "diffuse", (0.72, 0.74, 0.76)) or (0.72, 0.74, 0.76)
    try:
        r, g, b = (float(raw[0]), float(raw[1]), float(raw[2]))
    except Exception:
        r, g, b = (0.72, 0.74, 0.76)
    raw_alpha = getattr(node, "alpha", None)
    alpha = float(1.0 if raw_alpha is None else raw_alpha)
    alpha *= float(getattr(node, "txi_wateralpha", 1.0) or 1.0)
    return (_clamp01(r), _clamp01(g), _clamp01(b), _clamp01(alpha))


def _node_revision(node) -> tuple[int, ...]:
    return (
        len(getattr(node, "vertices", getattr(node, "verts", [])) or []),
        len(getattr(node, "faces", []) or []),
        int(getattr(node, "_gr_revision", 0) or 0),
        int(getattr(node, "vertex_space", 0) or 0),
        1 if bool(getattr(node, "_gr_vertices_in_kotor_world", False)) else 0,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_tex_name(value: str) -> str:
    text = str(value or "").replace("\x00", "").strip().lower()
    if text in {"", "null", "none"}:
        return ""
    for ext in (".tga", ".tpc", ".dds", ".png", ".bmp"):
        if text.endswith(ext):
            text = text[: -len(ext)]
            break
    return text
