"""Bind the posed high-fidelity player mesh to an exact canonical K1 human rig."""

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
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _armatures() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]


def _meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _world_head(armature: bpy.types.Object, bone_name: str) -> np.ndarray:
    bone = armature.data.bones[bone_name]
    value = armature.matrix_world @ bone.head_local
    return np.array([value.x, value.y, value.z], dtype=np.float64)


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


def _merge_group(mesh: bpy.types.Object, source_name: str, target_name: str) -> None:
    source = mesh.vertex_groups.get(source_name)
    if source is None or source_name == target_name:
        return
    target = mesh.vertex_groups.get(target_name) or mesh.vertex_groups.new(name=target_name)
    weights: list[tuple[int, float]] = []
    source_index = source.index
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            if membership.group == source_index and membership.weight > 0.0:
                weights.append((vertex.index, float(membership.weight)))
                break
    for vertex_index, weight in weights:
        target.add([vertex_index], weight, "ADD")
    mesh.vertex_groups.remove(source)


def _bounds(mesh: bpy.types.Object) -> dict[str, list[float]]:
    points = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    low = [min(float(point[index]) for point in points) for index in range(3)]
    high = [max(float(point[index]) for point in points) for index in range(3)]
    return {
        "min": low,
        "max": high,
        "dimensions": [high[index] - low[index] for index in range(3)],
    }


def _assign_material(mesh: bpy.types.Object, texture_path: Path) -> None:
    material = mesh.data.materials[0] if mesh.data.materials else bpy.data.materials.new("M_KotorPlayer")
    if not mesh.data.materials:
        mesh.data.materials.append(material)
    material.name = "M_KotorPlayer"
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(texture_path), check_existing=True)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.62
    links.new(image.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])


def main() -> int:
    args = _args()
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(args.player), use_anim=False)
    player_armatures = _armatures()
    player_meshes = _meshes()
    if len(player_armatures) != 1 or len(player_meshes) != 1:
        raise RuntimeError(f"Expected one posed player rig and mesh, got {len(player_armatures)} / {len(player_meshes)}")
    player_armature = player_armatures[0]
    player_mesh = player_meshes[0]

    bpy.ops.import_scene.fbx(filepath=str(args.reference), use_anim=False)
    reference_armatures = [armature for armature in _armatures() if armature != player_armature]
    if len(reference_armatures) != 1:
        raise RuntimeError(f"Expected one canonical K1 reference armature, got {len(reference_armatures)}")
    reference_armature = reference_armatures[0]
    canonical_armature_object_name = reference_armature.name
    reference_meshes = [mesh for mesh in _meshes() if mesh != player_mesh]

    missing = [name for name in ALIGN_BONES if name not in player_armature.data.bones or name not in reference_armature.data.bones]
    if missing:
        raise RuntimeError(f"Alignment bones are missing: {missing}")
    source_points = np.vstack([_world_head(player_armature, name) for name in ALIGN_BONES])
    target_points = np.vstack([_world_head(reference_armature, name) for name in ALIGN_BONES])
    alignment, alignment_scale, alignment_rms = _similarity(source_points, target_points)
    player_mesh.matrix_world = alignment @ player_mesh.matrix_world

    canonical_bones = {bone.name for bone in reference_armature.data.bones}
    for source_name, target_name in WEIGHT_FALLBACKS.items():
        _merge_group(player_mesh, source_name, target_name)
    unsupported = sorted(group.name for group in player_mesh.vertex_groups if group.name not in canonical_bones)
    for group_name in unsupported:
        _merge_group(player_mesh, group_name, "torsoUpr_g")

    for modifier in list(player_mesh.modifiers):
        if modifier.type == "ARMATURE":
            player_mesh.modifiers.remove(modifier)
    world_matrix = player_mesh.matrix_world.copy()
    player_mesh.parent = reference_armature
    player_mesh.matrix_parent_inverse = reference_armature.matrix_world.inverted_safe()
    player_mesh.matrix_world = world_matrix
    modifier = player_mesh.modifiers.new(name="K1 Canonical Rig", type="ARMATURE")
    modifier.object = reference_armature
    modifier.use_vertex_groups = True

    _assign_material(player_mesh, args.texture)
    for mesh in reference_meshes:
        bpy.data.objects.remove(mesh, do_unlink=True)
    bpy.data.objects.remove(player_armature, do_unlink=True)
    reference_armature.animation_data_clear()
    player_mesh.name = "SK_KotorPlayer_K1Rig"
    player_mesh.data.name = "SK_KotorPlayer_K1Rig"

    remaining_groups = sorted(group.name for group in player_mesh.vertex_groups if group.name not in canonical_bones)
    if remaining_groups:
        raise RuntimeError(f"Player still contains unsupported weight groups: {remaining_groups}")
    if len(player_mesh.data.uv_layers) != 1:
        raise RuntimeError(f"Expected one preserved player UV layer, got {len(player_mesh.data.uv_layers)}")

    bpy.context.view_layer.update()
    bounds = _bounds(player_mesh)
    bpy.ops.object.select_all(action="DESELECT")
    reference_armature.select_set(True)
    player_mesh.select_set(True)
    bpy.context.view_layer.objects.active = reference_armature
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
        "schema": "ghoststudio.enclave-player-k1rig.v2",
        "player": str(args.player),
        "canonical_reference": str(args.reference),
        "output": str(args.output),
        "bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "canonical_bone_count": len(reference_armature.data.bones),
        "canonical_armature_object_name": canonical_armature_object_name,
        "alignment_bones": list(ALIGN_BONES),
        "alignment_scale": alignment_scale,
        "alignment_rms_m": alignment_rms,
        "mesh_bounds_m": bounds,
        "uv_layers": len(player_mesh.data.uv_layers),
        "unsupported_weight_groups": remaining_groups,
        "exact_k1_rig_preserved": True,
    }
    if payload["bytes"] < 1024 or alignment_rms > 0.25:
        raise RuntimeError(f"K1 player preparation failed validation: {payload}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
