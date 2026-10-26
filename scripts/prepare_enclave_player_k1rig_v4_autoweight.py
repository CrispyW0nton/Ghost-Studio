"""Bind the aligned high-fidelity player to the exact K1 action skeleton.

Unlike the rejected global weight-name transfer, this aligns the undeformed
player surface to the canonical K1 rest pose and computes fresh bone-heat skin
weights on that exact armature.  Geometry, UVs, and material coordinates are
preserved while the inverse bind and deformation basis become native K1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


ALIGN_BONES = (
    "pelvis_g",
    "lthigh_g",
    "lshin_g",
    "lfoot_g",
    "rthigh_g",
    "rshin_g",
    "rfoot_g",
    "torso_g",
    "torsoUpr_g",
    "lcollar_g",
    "lbicep_g",
    "LbicepL_g",
    "Lforearm_g",
    "Lhand_g",
    "rcollar_g",
    "rbicep_g",
    "RbicepL_g",
    "Rforearm_g",
    "Rhand_g",
)


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def _world_head(armature, name: str) -> np.ndarray:
    value = armature.matrix_world @ armature.data.bones[name].head_local
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[Matrix, float, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    scale = float(singular.sum() / max(np.square(source_zero).sum(), 1.0e-12))
    translation = target_center - scale * (rotation @ source_center)
    matrix = Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            matrix[row][column] = float(scale * rotation[row, column])
        matrix[row][3] = float(translation[row])
    transformed = (scale * (rotation @ source.T)).T + translation
    rms = float(np.sqrt(np.mean(np.sum(np.square(transformed - target), axis=1))))
    return matrix, scale, rms


def _bounds(mesh) -> dict[str, list[float]]:
    points = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    low = [min(float(point[index]) for point in points) for index in range(3)]
    high = [max(float(point[index]) for point in points) for index in range(3)]
    return {
        "min": low,
        "max": high,
        "dimensions": [high[index] - low[index] for index in range(3)],
    }


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.player), use_anim=False)
    player_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    player_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(player_armatures) != 1 or len(player_meshes) != 1:
        raise RuntimeError("Prepared player must contain one armature and one mesh")
    player_armature = player_armatures[0]
    player_mesh = player_meshes[0]

    bpy.ops.import_scene.fbx(filepath=str(args.reference), use_anim=False)
    targets = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj != player_armature]
    if len(targets) != 1:
        raise RuntimeError("Exact K1 reference must contain one armature")
    target_armature = targets[0]
    reference_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj != player_mesh]
    missing = [
        name for name in ALIGN_BONES
        if name not in player_armature.data.bones or name not in target_armature.data.bones
    ]
    if missing:
        raise RuntimeError(f"Alignment bones are missing: {missing}")

    source_points = np.vstack([_world_head(player_armature, name) for name in ALIGN_BONES])
    target_points = np.vstack([_world_head(target_armature, name) for name in ALIGN_BONES])
    alignment, alignment_scale, alignment_rms = _similarity(source_points, target_points)

    # Bake the aligned surface into target-armature world space before asking
    # Blender to calculate a fresh, coherent inverse bind and weight field.
    aligned_world = alignment @ player_mesh.matrix_world
    aligned_positions = [aligned_world @ vertex.co for vertex in player_mesh.data.vertices]
    for modifier in list(player_mesh.modifiers):
        if modifier.type == "ARMATURE":
            player_mesh.modifiers.remove(modifier)
    player_mesh.parent = None
    player_mesh.matrix_parent_inverse = Matrix.Identity(4)
    player_mesh.matrix_world = Matrix.Identity(4)
    for vertex, position in zip(player_mesh.data.vertices, aligned_positions):
        vertex.co = position
    for group in list(player_mesh.vertex_groups):
        player_mesh.vertex_groups.remove(group)

    for mesh in reference_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)
    bpy.data.objects.remove(player_armature, do_unlink=True)
    target_armature.animation_data_clear()

    # Root helpers drive placement, not skin deformation.
    for name in ("cutscenedummy", "rootdummy"):
        bone = target_armature.data.bones.get(name)
        if bone:
            bone.use_deform = False

    bpy.ops.object.select_all(action="DESELECT")
    player_mesh.select_set(True)
    target_armature.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    bpy.context.view_layer.update()

    deform_groups = {group.name: group.index for group in player_mesh.vertex_groups}
    unweighted = []
    maximum_sum_error = 0.0
    maximum_influences = 0
    for vertex in player_mesh.data.vertices:
        weights = [membership.weight for membership in vertex.groups if membership.weight > 0.0]
        if not weights:
            unweighted.append(vertex.index)
            continue
        maximum_influences = max(maximum_influences, len(weights))
        maximum_sum_error = max(maximum_sum_error, abs(sum(weights) - 1.0))
    if unweighted:
        raise RuntimeError(f"Bone heat left unweighted player vertices: {unweighted[:20]}")

    player_mesh.name = "SK_KotorPlayer_K1Rig"
    player_mesh.data.name = "SK_KotorPlayer_K1Rig"
    target_armature.name = "KotorPlayer_K1_ActionRig"
    bpy.context.view_layer.update()
    bounds = _bounds(player_mesh)

    bpy.ops.object.select_all(action="DESELECT")
    target_armature.select_set(True)
    player_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_armature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="ROOT",
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )

    payload = {
        "schema": "ghoststudio.enclave-player-k1rig-autoweight.v4",
        "player": str(args.player),
        "exact_action_reference": str(args.reference),
        "output": str(args.output),
        "bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "target_bones": len(target_armature.data.bones),
        "alignment_scale": alignment_scale,
        "alignment_rms_m": alignment_rms,
        "mesh_bounds_m": bounds,
        "uv_layers": len(player_mesh.data.uv_layers),
        "weight_groups": sorted(deform_groups),
        "unweighted_vertices": len(unweighted),
        "maximum_weight_sum_error": maximum_sum_error,
        "maximum_influences_per_vertex": maximum_influences,
        "weight_policy": "blender_bone_heat_on_exact_k1_action_skeleton",
        "exact_action_skeleton_preserved": True,
    }
    if (
        payload["bytes"] < 1024
        or payload["unweighted_vertices"] != 0
        or alignment_rms > 0.15
        or len(deform_groups) < 15
    ):
        raise RuntimeError(f"Exact K1 auto-weight preparation failed: {payload}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
