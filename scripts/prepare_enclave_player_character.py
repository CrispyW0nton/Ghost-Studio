"""Pose, normalize, and rename the high-fidelity player for KOTOR animation import."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


BONE_RENAMES = {
    "Root": "cutscenedummy",
    "Hip": "rootdummy",
    "Pelvis": "pelvis_g",
    "L_Thigh": "lthigh_g",
    "L_Calf": "lshin_g",
    "L_Foot": "lfoot_g",
    "L_Foot_end": "lfootT_g",
    "R_Thigh": "rthigh_g",
    "R_Calf": "rshin_g",
    "R_Foot": "rfoot_g",
    "R_Foot_end": "rfootT_g",
    "Waist": "torso_g",
    "Spine01": "torsoUpr_g",
    "Spine02": "player_spine_detail",
    "L_Clavicle": "lcollar_dum",
    "L_Upperarm": "lcollar_g",
    "L_UpperarmTwist01": "lbicep_g",
    "L_UpperarmTwist02": "LbicepL_g",
    "L_Forearm": "Lforearm_g",
    "L_Hand": "Lhand_g",
    "R_Clavicle": "rcollar_dum",
    "R_Upperarm": "rcollar_g",
    "R_UpperarmTwist01": "rbicep_g",
    "R_UpperarmTwist02": "RbicepL_g",
    "R_Forearm": "Rforearm_g",
    "R_Hand": "Rhand_g",
}


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-height-m", type=float, default=1.78)
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _measured_kotor_arm_slope(reference: Path) -> float:
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(reference), use_anim=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("KOTOR reference must contain one armature")
    bones = armatures[0].data.bones
    measurements = []
    for shoulder_name, hand_name in (("lcollar_g", "Lhand_g"), ("rcollar_g", "Rhand_g")):
        shoulder = bones.get(shoulder_name)
        hand = bones.get(hand_name)
        if not shoulder or not hand:
            raise RuntimeError(f"KOTOR reference is missing {shoulder_name}/{hand_name}")
        horizontal = abs(float(hand.head_local.x - shoulder.head_local.x))
        vertical = abs(float(hand.head_local.y - shoulder.head_local.y))
        measurements.append(math.degrees(math.atan2(vertical, max(horizontal, 1.0e-6))))
    return sum(measurements) / len(measurements)


def _world_height(meshes: list) -> tuple[float, float, float]:
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = min(float(point.z) for point in points)
    high = max(float(point.z) for point in points)
    return high - low, low, high


def _assign_material(mesh, texture_path: Path) -> None:
    material = mesh.data.materials[0] if mesh.data.materials else bpy.data.materials.new("M_KotorPlayer")
    if not mesh.data.materials:
        mesh.data.materials.append(material)
    material.name = "M_KotorPlayer"
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(texture_path), check_existing=True)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.68
    links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])


def main() -> int:
    args = _args()
    kotor_arm_slope_degrees = _measured_kotor_arm_slope(args.reference)
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(args.source), use_anim=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    skinned_meshes = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and (obj.parent in armatures or any(mod.type == "ARMATURE" for mod in obj.modifiers))
    ]
    if len(armatures) != 1 or len(skinned_meshes) != 1:
        raise RuntimeError(f"Expected one player armature and mesh, got {len(armatures)} / {len(skinned_meshes)}")
    armature = armatures[0]
    player_mesh = skinned_meshes[0]
    removed = []
    for obj in list(bpy.context.scene.objects):
        if obj not in {armature, player_mesh}:
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)

    _assign_material(player_mesh, args.texture)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    player_mesh.select_set(False)
    bpy.ops.object.mode_set(mode="POSE")
    for bone_name, hand_name in (
        ("L_Upperarm", "L_Hand"),
        ("R_Upperarm", "R_Hand"),
    ):
        pose_bone = armature.pose.bones.get(bone_name)
        hand_bone = armature.data.bones.get(hand_name)
        rest_bone = armature.data.bones.get(bone_name)
        if not pose_bone or not hand_bone or not rest_bone:
            raise RuntimeError(f"Player skeleton is missing {bone_name}/{hand_name}")
        shoulder = rest_bone.head_local.copy()
        current = hand_bone.head_local - shoulder
        horizontal = Vector((current.x, current.y, 0.0))
        desired = horizontal + Vector((0.0, 0.0, -horizontal.length * math.tan(math.radians(kotor_arm_slope_degrees))))
        delta = current.normalized().rotation_difference(desired.normalized())
        pose_bone.matrix = (
            Matrix.Translation(shoulder)
            @ delta.to_matrix().to_4x4()
            @ Matrix.Translation(-shoulder)
            @ pose_bone.matrix
        )
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = player_mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_object.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if len(evaluated_mesh.vertices) != len(player_mesh.data.vertices):
        raise RuntimeError("Player armature evaluation changed the vertex count")
    to_local = player_mesh.matrix_world.inverted_safe()
    for target, posed in zip(player_mesh.data.vertices, evaluated_mesh.vertices):
        target.co = to_local @ (evaluated_object.matrix_world @ posed.co)
    evaluated_object.to_mesh_clear()
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="EDIT")
    for old_name, new_name in BONE_RENAMES.items():
        bone = armature.data.edit_bones.get(old_name)
        if not bone:
            raise RuntimeError(f"Player skeleton is missing required bone {old_name}")
        bone.name = new_name
    edit_bones = armature.data.edit_bones
    for side in ("L", "R"):
        prefix = side if side == "L" else "R"
        lower = side.casefold()
        edit_bones[f"{lower}collar_dum"].parent = edit_bones["torsoUpr_g"]
        edit_bones[f"{lower}collar_g"].parent = edit_bones[f"{lower}collar_dum"]
        edit_bones[f"{lower}bicep_g"].parent = edit_bones[f"{lower}collar_g"]
        edit_bones[f"{prefix}bicepL_g"].parent = edit_bones[f"{lower}bicep_g"]
        edit_bones[f"{prefix}forearm_g"].parent = edit_bones[f"{prefix}bicepL_g"]
        edit_bones[f"{prefix}hand_g"].parent = edit_bones[f"{prefix}forearm_g"]
    bpy.ops.object.mode_set(mode="OBJECT")

    original_height, _, _ = _world_height([player_mesh])
    scale_factor = args.target_height_m / max(original_height, 1.0e-6)
    armature.scale = tuple(float(value) * scale_factor for value in armature.scale)
    bpy.context.view_layer.update()
    final_height, final_low, _ = _world_height([player_mesh])
    armature.location.z -= final_low
    bpy.context.view_layer.update()
    final_height, final_low, final_high = _world_height([player_mesh])

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    player_mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
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
    if not args.output.is_file() or args.output.stat().st_size < 1024:
        raise RuntimeError("Player FBX export failed")
    payload = {
        "schema": "ghoststudio.enclave-player-preparation.v1",
        "source": str(args.source),
        "reference": str(args.reference),
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "pose_mode_used": True,
        "kotor_arm_slope_degrees": kotor_arm_slope_degrees,
        "target_height_m": args.target_height_m,
        "final_height_m": final_height,
        "final_min_z": final_low,
        "final_max_z": final_high,
        "bone_count": len(armature.data.bones),
        "bone_renames": BONE_RENAMES,
        "removed_scene_helpers": removed,
        "uv_layers": len(player_mesh.data.uv_layers),
        "material": player_mesh.data.materials[0].name if player_mesh.data.materials else "",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "output": str(args.output), "final_height_m": final_height, "kotor_arm_slope_degrees": kotor_arm_slope_degrees}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
