"""Animation deformation validation for rigged character models (T2532).

A fit can look perfect in bind pose while inherited animations destroy the
mesh (Rancor fixture: skeleton placed well, ``ctaunt`` stretched the arms into
spikes).  This validator provides a headless, deterministic gate that skins
the bound mesh with linear blend skinning against sampled animation poses and
flags the classic failure signatures:

- ``exploded_vertices``: vertex displacement far beyond model height.
- ``edge_stretch_spike``: triangle edge stretched > 3x its bind length.
- ``limb_collapse``: triangle edge compressed below 15% of bind length
  (normal joint bending compresses ~20-40%; 85% compression is collapse).
- ``non_finite_vertices``: NaN/inf coming out of the skin matrix.
- ``missing_bone_transforms``: bone_map names with no world transform in the
  evaluated pose (dropped palette slots — the T2530 class of bug).

Conventions (verified against the pipeline):
- Quaternions are XYZW, W-last.
- Skin-node vertices arriving here are the post-fit world/bind positions.
- ``qbone_list``/``tbone_list`` may be compact palette arrays on freshly bound
  imported payloads, or full node-indexed Odyssey arrays on native MDLs.  The
  validator therefore skins through ``MatrixPaletteUploader`` instead of
  reimplementing qBone/tBone semantics.
- Animations author absolute parent-local position/orientation controllers;
  ``evaluate_aurora_animation_pose`` composes them to world by FK.

LBS in this convention (world-space bind, per influence i)::

    local  = R_bind_i^T @ (v_bind - t_bind_i)      # into bone space
    v_anim = R_anim_i @ local + t_anim_i           # out at animated pose
    v'     = sum_i w_i * v_anim_i

Pure numpy, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging

log = logging.getLogger(__name__)

# Thresholds tuned so normal joint bending passes (elbows/knees compress
# 20-40% and stretch outer skin ~1.5x) while spikes and collapse fail.
EDGE_STRETCH_SPIKE_RATIO = 3.0
EDGE_COLLAPSE_RATIO = 0.15
DISPLACEMENT_HEIGHT_FACTOR = 1.5
EDGE_STRETCH_ABSOLUTE_HEIGHT_FACTOR = 0.25
MAX_FAILED_EDGE_FRACTION = 0.005  # 0.5% of edges may be degenerate noise


@dataclass
class DeformationSampleReport:
    animation: str = ""
    time: float = 0.0
    max_edge_stretch: float = 1.0
    min_edge_ratio: float = 1.0
    stretched_edge_fraction: float = 0.0
    collapsed_edge_fraction: float = 0.0
    max_displacement: float = 0.0
    displacement_bound: float = 0.0
    bind_floor_z: float = 0.0
    minimum_deformed_z: float = 0.0
    floor_penetration: float = 0.0
    non_finite_count: int = 0
    missing_bones: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "animation": self.animation,
            "time": self.time,
            "ok": self.ok,
            "max_edge_stretch": self.max_edge_stretch,
            "min_edge_ratio": self.min_edge_ratio,
            "stretched_edge_fraction": self.stretched_edge_fraction,
            "collapsed_edge_fraction": self.collapsed_edge_fraction,
            "max_displacement": self.max_displacement,
            "displacement_bound": self.displacement_bound,
            "bind_floor_z": self.bind_floor_z,
            "minimum_deformed_z": self.minimum_deformed_z,
            "floor_penetration": self.floor_penetration,
            "non_finite_count": self.non_finite_count,
            "missing_bones": list(self.missing_bones),
            "failures": list(self.failures),
        }


@dataclass
class DeformationValidationReport:
    ok: bool = True
    model_name: str = ""
    animations_checked: int = 0
    samples: List[DeformationSampleReport] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "model_name": self.model_name,
            "animations_checked": self.animations_checked,
            "failures": list(self.failures),
            "message": self.message,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _quat_to_matrix(q: Sequence[float]) -> "Any":
    """XYZW quaternion -> 3x3 rotation matrix (numpy)."""
    import numpy as np

    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _skin_rows_to_arrays(
    skin_data: Sequence[Any],
    vertex_count: int,
    slot_count: int,
) -> Tuple["Any", "Any"]:
    """VertexSkinData rows -> (weights[N,4], indices[N,4]) numpy arrays."""
    import numpy as np

    weights = np.zeros((vertex_count, 4), dtype=np.float64)
    indices = np.zeros((vertex_count, 4), dtype=np.int64)
    for row_index in range(min(vertex_count, len(skin_data))):
        row = skin_data[row_index]
        influences = list(getattr(row, "influences", []) or [])[:4]
        for slot, influence in enumerate(influences):
            bone_index = int(getattr(influence, "bone_index", 0))
            if 0 <= bone_index < slot_count:
                weights[row_index, slot] = float(getattr(influence, "weight", 0.0))
                indices[row_index, slot] = bone_index
    totals = weights.sum(axis=1, keepdims=True)
    safe = totals > 1.0e-9
    weights = np.where(safe, weights / np.where(safe, totals, 1.0), weights)
    return weights, indices


def _edges_from_faces(faces: Sequence[Sequence[int]], vertex_count: int) -> "Any":
    import numpy as np

    edges = set()
    for face in faces:
        if len(face) < 3:
            continue
        try:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
        except Exception:
            continue
        if max(a, b, c) >= vertex_count or min(a, b, c) < 0:
            continue
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v) if u < v else (v, u))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def _model_height(model: Any) -> float:
    lo_z = None
    hi_z = None
    for node in model.all_nodes() if hasattr(model, "all_nodes") else []:
        for vert in list(getattr(node, "vertices", []) or []):
            if len(vert) < 3:
                continue
            z = float(vert[2])
            lo_z = z if lo_z is None else min(lo_z, z)
            hi_z = z if hi_z is None else max(hi_z, z)
    if lo_z is None or hi_z is None:
        return 0.0
    return max(0.0, hi_z - lo_z)


def _model_diagonal(model: Any) -> float:
    import math

    mins = [None, None, None]
    maxs = [None, None, None]
    for node in model.all_nodes() if hasattr(model, "all_nodes") else []:
        for vert in list(getattr(node, "vertices", []) or []):
            if len(vert) < 3:
                continue
            for axis in range(3):
                value = float(vert[axis])
                mins[axis] = value if mins[axis] is None else min(mins[axis], value)
                maxs[axis] = value if maxs[axis] is None else max(maxs[axis], value)
    if any(value is None for value in mins) or any(value is None for value in maxs):
        return 0.0
    dx = float(maxs[0]) - float(mins[0])
    dy = float(maxs[1]) - float(mins[1])
    dz = float(maxs[2]) - float(mins[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _resolve_pose_transform(
    pose: Any,
    name: str,
    lookup: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    """Fetch (position, rotation) for a bone name from an evaluated pose."""
    transforms = getattr(pose, "world_transforms_by_node", {}) or {}
    entry = transforms.get(name)
    if entry is None and lookup is not None:
        actual = lookup.get(name.strip().lower())
        if actual is not None:
            entry = transforms.get(actual)
    if entry is None:
        return None
    return (
        tuple(float(v) for v in entry.position),
        tuple(float(v) for v in entry.rotation),
    )


def _skin_vertices_with_palette(
    *,
    uploader: Any,
    mesh: Any,
    pose: Any,
    anim_base_pose: Any,
    verts: Any,
    weights: Any,
    indices: Any,
) -> Any:
    """Skin vertices through the renderer's shared MatrixPaletteUploader."""
    import numpy as np

    uploader.compute_skin_node_palette(
        mesh,
        pose,
        anim_base_pose=anim_base_pose,
    )
    palette = uploader.as_numpy_array()
    if palette is None or len(palette) == 0:
        return np.array(verts, dtype=np.float64, copy=True)

    bone_count = int(palette.shape[0])
    vertices_h = np.ones((verts.shape[0], 4), dtype=np.float64)
    vertices_h[:, :3] = verts[:, :3]
    skinned = np.zeros((verts.shape[0], 4), dtype=np.float64)
    weight_total = np.zeros((verts.shape[0],), dtype=np.float64)
    for influence in range(min(4, weights.shape[1])):
        slot_weights = weights[:, influence]
        slot_indices = indices[:, influence]
        valid = (
            (slot_weights > 1.0e-9)
            & (slot_indices >= 0)
            & (slot_indices < bone_count)
        )
        if not bool(np.any(valid)):
            continue
        matrices = palette[slot_indices[valid]].astype(np.float64, copy=False)
        moved = np.einsum(
            "nij,nj->ni",
            matrices,
            vertices_h[valid],
            optimize=True,
        )
        skinned[valid] += slot_weights[valid, None] * moved
        weight_total[valid] += slot_weights[valid]

    missing = weight_total <= 1.0e-9
    if bool(np.any(missing)):
        skinned[missing] = vertices_h[missing]
    return skinned[:, :3]


def _palette_pose_from_evaluated_pose(pose: Any) -> Any:
    """Adapt EvaluatedAuroraPose into the pose shape MatrixPaletteUploader uses."""

    nodes: Dict[str, Any] = {}
    for raw_name, transform in dict(getattr(pose, "local_transforms_by_node", {}) or {}).items():
        name = str(raw_name or "")
        if not name:
            continue
        nodes[name.lower()] = SimpleNamespace(
            name=name,
            position=tuple(float(value) for value in getattr(transform, "position", (0.0, 0.0, 0.0))),
            rotation=tuple(float(value) for value in getattr(transform, "rotation", (0.0, 0.0, 0.0, 1.0))),
        )
    return SimpleNamespace(
        time=float(getattr(pose, "time", 0.0) or 0.0),
        nodes=nodes,
    )


def validate_animation_deformation(
    model: Any,
    *,
    animations: Optional[Sequence[str]] = None,
    samples_per_animation: int = 3,
    max_animations: int = 4,
    edge_stretch_ratio: float = EDGE_STRETCH_SPIKE_RATIO,
    edge_collapse_ratio: float = EDGE_COLLAPSE_RATIO,
    displacement_height_factor: float = DISPLACEMENT_HEIGHT_FACTOR,
) -> DeformationValidationReport:
    """Skin the bound mesh against sampled animation poses and score it.

    ``model`` must be a rigged model whose skin nodes carry ``skin_data``
    (VertexSkinData rows), ``bone_map`` (slot names), ``qbone_list``/
    ``tbone_list`` (bind world rot/pos per slot), and world-space vertices —
    exactly what ``bind_imported_meshes_to_skeleton`` produces.
    """
    import numpy as np

    try:
        from ..animation.animation_engine import evaluate_aurora_animation_pose
        from ..animation.gpu_skinning import MAX_BONES, MatrixPaletteUploader
    except ImportError:  # pragma: no cover
        from src.core.animation.animation_engine import (  # type: ignore
            evaluate_aurora_animation_pose,
        )
        from src.core.animation.gpu_skinning import (  # type: ignore
            MAX_BONES,
            MatrixPaletteUploader,
        )

    report = DeformationValidationReport(
        model_name=str(getattr(model, "name", "") or ""),
    )
    if model is None or getattr(model, "root_node", None) is None:
        report.ok = False
        report.message = "No model to validate."
        report.failures.append("no_model")
        return report

    anim_blocks = list(getattr(model, "animations", []) or [])
    if animations:
        wanted = {str(name).strip().lower() for name in animations}
        anim_blocks = [
            block for block in anim_blocks
            if str(getattr(block, "name", "") or "").strip().lower() in wanted
        ]
    anim_blocks = anim_blocks[:max_animations]
    if not anim_blocks:
        report.message = "No animations available to validate (skipped)."
        return report

    # Collect skinned meshes.
    skinned: List[Any] = []
    for node in model.all_nodes():
        if not bool(getattr(node, "is_skin", False)):
            continue
        if not list(getattr(node, "vertices", []) or []):
            continue
        if not list(getattr(node, "skin_data", []) or []):
            continue
        if not list(getattr(node, "bone_map", []) or []):
            continue
        skinned.append(node)
    if not skinned:
        report.message = "No skinned meshes to validate (skipped)."
        return report

    height = _model_height(model)
    diagonal = _model_diagonal(model)
    model_scale = max(height, diagonal * 0.6)
    displacement_bound = max(1.0, model_scale * displacement_height_factor)
    edge_stretch_absolute = max(0.25, model_scale * EDGE_STRETCH_ABSOLUTE_HEIGHT_FACTOR)

    # Case-insensitive node-name lookup for pose transforms.
    name_lookup = {
        str(getattr(node, "name", "") or "").strip().lower():
            str(getattr(node, "name", "") or "")
        for node in model.all_nodes()
    }

    # Pre-compute per-mesh static data.
    mesh_data = []
    for mesh in skinned:
        verts = np.asarray(
            [tuple(float(c) for c in v[:3]) for v in mesh.vertices],
            dtype=np.float64,
        )
        slot_names = [str(n or "").strip() for n in list(mesh.bone_map)]
        slot_count = len(slot_names)
        weights, indices = _skin_rows_to_arrays(
            list(mesh.skin_data), verts.shape[0], slot_count,
        )
        uploader = MatrixPaletteUploader(max_bones=max(int(MAX_BONES), slot_count))
        uploader.build_inverse_bind_pose(model)
        edges = _edges_from_faces(list(getattr(mesh, "faces", []) or []), verts.shape[0])
        bind_edge_len = (
            np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
            if edges.shape[0] else np.zeros(0)
        )
        mesh_data.append({
            "mesh": mesh,
            "verts": verts,
            "slot_names": slot_names,
            "weights": weights,
            "indices": indices,
            "edges": edges,
            "bind_edge_len": bind_edge_len,
            "uploader": uploader,
            "use_animation_base_bind": bool(getattr(
                mesh,
                "_gr_use_animation_base_bind_for_preview",
                False,
            )),
        })

    # The rendered skin, not a foot-bone pivot, defines contact with the floor.
    # Retargeted clips can keep every foot node above its bind pivot while still
    # pulling the weighted body through the stage.  Use the lowest bind vertex
    # as the immutable floor reference and compare every sampled deformed mesh
    # against that exact plane.
    bind_floor_z = min(
        float(data["verts"][:, 2].min())
        for data in mesh_data
        if data["verts"].size
    )

    for block in anim_blocks:
        anim_name = str(getattr(block, "name", "") or "?")
        length = float(getattr(block, "length", 0.0) or 0.0)
        if length <= 0.0:
            times = [0.0]
        else:
            count = max(1, int(samples_per_animation))
            times = [
                length * (step + 1) / float(count + 1)
                for step in range(count)
            ]
        base_pose = None
        base_palette_pose = None
        if any(bool(data.get("use_animation_base_bind")) for data in mesh_data):
            try:
                base_pose = evaluate_aurora_animation_pose(model, block, 0.0)
                base_palette_pose = _palette_pose_from_evaluated_pose(base_pose)
            except Exception as exc:
                log.warning(
                    "deformation validation: base pose evaluation failed for %s: %s",
                    anim_name, exc,
                )
        for t in times:
            sample = DeformationSampleReport(
                animation=anim_name,
                time=float(t),
                bind_floor_z=bind_floor_z,
                minimum_deformed_z=float("inf"),
            )
            try:
                pose = evaluate_aurora_animation_pose(model, block, float(t))
            except Exception as exc:
                sample.failures.append("pose_evaluation_failed")
                log.warning(
                    "deformation validation: pose evaluation failed for %s@%.3f: %s",
                    anim_name, t, exc,
                )
                report.samples.append(sample)
                report.ok = False
                continue

            palette_pose = _palette_pose_from_evaluated_pose(pose)
            for data in mesh_data:
                slot_names = data["slot_names"]
                missing: List[str] = []
                for bone_name in slot_names:
                    resolved = _resolve_pose_transform(pose, bone_name, name_lookup)
                    if resolved is None:
                        missing.append(bone_name)
                        continue
                if missing:
                    sample.missing_bones = sorted(set(sample.missing_bones) | set(missing))

                verts = data["verts"]
                weights = data["weights"]
                indices = data["indices"]
                try:
                    deformed = _skin_vertices_with_palette(
                        uploader=data["uploader"],
                        mesh=data["mesh"],
                        pose=palette_pose,
                        anim_base_pose=base_palette_pose if data["use_animation_base_bind"] else None,
                        verts=verts,
                        weights=weights,
                        indices=indices,
                    )
                except Exception as exc:
                    sample.failures.append("palette_skinning_failed")
                    log.warning(
                        "deformation validation: palette skinning failed for %s@%.3f/%s: %s",
                        anim_name,
                        t,
                        getattr(data["mesh"], "name", "?"),
                        exc,
                    )
                    deformed = verts

                finite_mask = np.isfinite(deformed).all(axis=1)
                non_finite = int((~finite_mask).sum())
                sample.non_finite_count += non_finite
                safe_deformed = np.where(finite_mask[:, None], deformed, verts)
                if safe_deformed.size:
                    sample.minimum_deformed_z = min(
                        sample.minimum_deformed_z,
                        float(safe_deformed[:, 2].min()),
                    )

                displacement = np.linalg.norm(safe_deformed - verts, axis=1)
                max_disp = float(displacement.max()) if displacement.size else 0.0
                sample.max_displacement = max(sample.max_displacement, max_disp)
                sample.displacement_bound = displacement_bound

                edges = data["edges"]
                if edges.shape[0]:
                    deformed_len = np.linalg.norm(
                        safe_deformed[edges[:, 0]] - safe_deformed[edges[:, 1]],
                        axis=1,
                    )
                    bind_len = data["bind_edge_len"]
                    valid = bind_len > 1.0e-9
                    ratio = np.ones_like(deformed_len)
                    ratio[valid] = deformed_len[valid] / bind_len[valid]
                    sample.max_edge_stretch = max(
                        sample.max_edge_stretch, float(ratio.max()),
                    )
                    sample.min_edge_ratio = min(
                        sample.min_edge_ratio, float(ratio.min()),
                    )
                    stretch_delta = deformed_len - bind_len
                    stretched = float((
                        (ratio > edge_stretch_ratio)
                        & (stretch_delta > edge_stretch_absolute)
                    ).mean())
                    collapsed = float((ratio < edge_collapse_ratio).mean())
                    sample.stretched_edge_fraction = max(
                        sample.stretched_edge_fraction, stretched,
                    )
                    sample.collapsed_edge_fraction = max(
                        sample.collapsed_edge_fraction, collapsed,
                    )

            if not np.isfinite(sample.minimum_deformed_z):
                sample.minimum_deformed_z = bind_floor_z
            sample.floor_penetration = max(
                0.0,
                bind_floor_z - sample.minimum_deformed_z,
            )

            if sample.non_finite_count:
                sample.failures.append("non_finite_vertices")
            if sample.missing_bones:
                sample.failures.append("missing_bone_transforms")
            if sample.max_displacement > displacement_bound:
                sample.failures.append("exploded_vertices")
            if sample.stretched_edge_fraction > MAX_FAILED_EDGE_FRACTION:
                sample.failures.append("edge_stretch_spike")
            if sample.collapsed_edge_fraction > MAX_FAILED_EDGE_FRACTION:
                sample.failures.append("limb_collapse")

            report.samples.append(sample)
            if sample.failures:
                report.ok = False

    report.animations_checked = len(anim_blocks)
    failure_names = sorted({
        failure
        for sample in report.samples
        for failure in sample.failures
    })
    report.failures.extend(failure_names)
    if report.ok:
        report.message = (
            f"Deformation OK across {report.animations_checked} animation(s), "
            f"{len(report.samples)} sample(s)."
        )
    else:
        report.message = (
            f"Deformation FAILED: {', '.join(failure_names)} across "
            f"{report.animations_checked} animation(s)."
        )
    return report
