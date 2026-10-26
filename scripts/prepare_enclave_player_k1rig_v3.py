"""Transfer the high-fidelity player skin onto the exact K1 action skeleton.

The earlier v2 preparation globally aligned the player mesh to a canonical
armature and then reused its weights.  A global fit cannot preserve the bind
relationship when individual rest bones differ, which produced visibly wrong
locomotion despite valid animation tracks.  This version performs a weighted
per-bone rest-space skin transfer onto the same normalized K1 skeleton that
authored the exported actions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


WEIGHT_FALLBACKS = {
    "L_CalfTwist01": "lshin_g",
    "L_CalfTwist02": "lshin_g",
    "L_CalfTwist02_end": "lshin_g",
    "L_ThighTwist01": "lthigh_g",
    "L_ThighTwist02": "lthigh_g",
    "L_ThighTwist02_end": "lthigh_g",
    "R_CalfTwist01": "rshin_g",
    "R_CalfTwist02": "rshin_g",
    "R_CalfTwist02_end": "rshin_g",
    "R_ThighTwist01": "rthigh_g",
    "R_ThighTwist02": "rthigh_g",
    "R_ThighTwist02_end": "rthigh_g",
    "player_spine_detail": "torsoUpr_g",
    "NeckTwist01": "torsoUpr_g",
    "NeckTwist02": "torsoUpr_g",
    "Head": "torsoUpr_g",
    "Head_end": "torsoUpr_g",
    "L_ForearmTwist01": "Lforearm_g",
    "L_ForearmTwist02": "Lforearm_g",
    "L_ForearmTwist02_end": "Lforearm_g",
    "R_ForearmTwist01": "Rforearm_g",
    "R_ForearmTwist02": "Rforearm_g",
    "R_ForearmTwist02_end": "Rforearm_g",
    "L_UpperarmTwist02_end": "LbicepL_g",
    "R_UpperarmTwist02_end": "RbicepL_g",
    "L_Hand_end": "Lhand_g",
    "R_Hand_end": "Rhand_g",
}


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _bounds(mesh: bpy.types.Object) -> dict[str, list[float]]:
    points = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    low = [min(float(point[index]) for point in points) for index in range(3)]
    high = [max(float(point[index]) for point in points) for index in range(3)]
    return {
        "min": low,
        "max": high,
        "dimensions": [high[index] - low[index] for index in range(3)],
    }


def _bone_world(armature: bpy.types.Object, name: str) -> Matrix:
    return armature.matrix_world @ armature.data.bones[name].matrix_local


def main() -> int:
    args = _args()
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(args.player), use_anim=False)
    player_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    player_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(player_armatures) != 1 or len(player_meshes) != 1:
        raise RuntimeError(
            f"Expected one prepared player armature and mesh, got {len(player_armatures)} / {len(player_meshes)}"
        )
    player_armature = player_armatures[0]
    player_mesh = player_meshes[0]

    bpy.ops.import_scene.fbx(filepath=str(args.reference), use_anim=False)
    target_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj != player_armature]
    if len(target_armatures) != 1:
        raise RuntimeError(f"Expected one exact K1 reference armature, got {len(target_armatures)}")
    target_armature = target_armatures[0]
    target_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj != player_mesh]
    target_bones = {bone.name for bone in target_armature.data.bones}
    source_bones = {bone.name for bone in player_armature.data.bones}

    source_groups = {group.index: group.name for group in player_mesh.vertex_groups}
    group_mapping: dict[str, str] = {}
    for source_name in source_groups.values():
        if source_name in target_bones:
            group_mapping[source_name] = source_name
        elif source_name in WEIGHT_FALLBACKS and WEIGHT_FALLBACKS[source_name] in target_bones:
            group_mapping[source_name] = WEIGHT_FALLBACKS[source_name]

    missing_weight_groups = sorted(
        name for name in source_groups.values() if name in source_bones and name not in group_mapping
    )
    if missing_weight_groups:
        raise RuntimeError(f"Unmapped skinned player groups: {missing_weight_groups}")

    transfer_matrices: dict[tuple[str, str], Matrix] = {}
    for source_name, target_name in group_mapping.items():
        if source_name not in source_bones:
            continue
        transfer_matrices[(source_name, target_name)] = (
            _bone_world(target_armature, target_name)
            @ _bone_world(player_armature, source_name).inverted_safe()
        )

    vertex_positions: list[Vector] = []
    vertex_weights: list[dict[str, float]] = []
    weight_sum_errors: list[float] = []
    unweighted_vertices: list[int] = []
    for vertex in player_mesh.data.vertices:
        source_world = player_mesh.matrix_world @ vertex.co
        accumulated = Vector((0.0, 0.0, 0.0))
        mapped_weights: dict[str, float] = {}
        total = 0.0
        for membership in vertex.groups:
            source_name = source_groups.get(membership.group)
            target_name = group_mapping.get(source_name or "")
            transform = transfer_matrices.get((source_name or "", target_name or ""))
            weight = float(membership.weight)
            if not target_name or transform is None or weight <= 0.0:
                continue
            accumulated += (transform @ source_world) * weight
            mapped_weights[target_name] = mapped_weights.get(target_name, 0.0) + weight
            total += weight
        if total <= 1.0e-8:
            unweighted_vertices.append(vertex.index)
            vertex_positions.append(source_world.copy())
            vertex_weights.append({})
            continue
        vertex_positions.append(accumulated / total)
        normalized = {name: value / total for name, value in mapped_weights.items()}
        vertex_weights.append(normalized)
        weight_sum_errors.append(abs(sum(normalized.values()) - 1.0))

    if unweighted_vertices:
        raise RuntimeError(f"Player contains unweighted vertices: {unweighted_vertices[:20]}")

    # Store transferred vertices in world space, then bind them to the exact
    # K1 reference armature while preserving their world positions.
    player_mesh.matrix_world = Matrix.Identity(4)
    for vertex, position in zip(player_mesh.data.vertices, vertex_positions):
        vertex.co = position
    for group in list(player_mesh.vertex_groups):
        player_mesh.vertex_groups.remove(group)
    new_groups = {
        name: player_mesh.vertex_groups.new(name=name)
        for name in sorted({name for weights in vertex_weights for name in weights})
    }
    for vertex_index, weights in enumerate(vertex_weights):
        for name, weight in weights.items():
            new_groups[name].add([vertex_index], weight, "REPLACE")

    for modifier in list(player_mesh.modifiers):
        if modifier.type == "ARMATURE":
            player_mesh.modifiers.remove(modifier)
    player_mesh.parent = target_armature
    player_mesh.matrix_parent_inverse = target_armature.matrix_world.inverted_safe()
    modifier = player_mesh.modifiers.new(name="Exact K1 Action Skeleton", type="ARMATURE")
    modifier.object = target_armature
    modifier.use_vertex_groups = True

    for mesh in target_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)
    bpy.data.objects.remove(player_armature, do_unlink=True)
    target_armature.animation_data_clear()
    player_mesh.name = "SK_KotorPlayer_K1Rig"
    player_mesh.data.name = "SK_KotorPlayer_K1Rig"
    target_armature.name = "KotorPlayer_K1_ActionRig"
    bpy.context.view_layer.update()

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

    bounds = _bounds(player_mesh)
    payload = {
        "schema": "ghoststudio.enclave-player-k1rig.v3",
        "player": str(args.player),
        "exact_action_reference": str(args.reference),
        "output": str(args.output),
        "bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "source_bones": len(source_bones),
        "target_bones": len(target_bones),
        "mapped_weight_groups": group_mapping,
        "weighted_vertices": len(vertex_positions),
        "unweighted_vertices": len(unweighted_vertices),
        "maximum_weight_sum_error": max(weight_sum_errors) if weight_sum_errors else math.inf,
        "mesh_bounds_m": bounds,
        "uv_layers": len(player_mesh.data.uv_layers),
        "exact_action_skeleton_preserved": True,
        "rest_transfer_policy": "weighted_target_rest_world_times_inverse_source_rest_world",
    }
    height = bounds["dimensions"][1]
    if (
        payload["bytes"] < 1024
        or payload["unweighted_vertices"] != 0
        or payload["maximum_weight_sum_error"] > 1.0e-5
        or not 1.4 <= height <= 2.0
    ):
        raise RuntimeError(f"Exact K1 rest transfer failed validation: {payload}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
