"""Convert Ghost Studio's rigged GLB interchange into a binary Unreal FBX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--action", default="")
    return parser.parse_args(argv)


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0


def _action_evidence() -> list[dict]:
    evidence = []
    for action in sorted(bpy.data.actions, key=lambda item: item.name.casefold()):
        start, end = action.frame_range
        slots = list(getattr(action, "slots", []) or [])
        evidence.append({
            "name": action.name,
            "frame_start": float(start),
            "frame_end": float(end),
            "slot_count": len(slots),
            "fcurve_count": len(getattr(action, "fcurves", []) or []),
        })
    return evidence


def _skin_bone_parented_meshes(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
) -> list[dict[str, str]]:
    """Convert rigid glTF bone children into Unreal-safe skinned meshes."""

    converted: list[dict[str, str]] = []
    previous_pose_position = armature.data.pose_position
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    armature_inverse = armature.matrix_world.inverted()
    for mesh in meshes:
        if mesh.parent is not armature or mesh.parent_type != "BONE":
            continue
        bone_name = str(mesh.parent_bone or "")
        if not bone_name or armature.data.bones.get(bone_name) is None:
            raise RuntimeError(
                f"Rigid mesh {mesh.name} references missing parent bone {bone_name!r}"
            )
        mesh.data.transform(armature_inverse @ mesh.matrix_world)
        mesh.parent = armature
        mesh.parent_type = "OBJECT"
        mesh.parent_bone = ""
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        mesh.vertex_groups.clear()
        group = mesh.vertex_groups.new(name=bone_name)
        group.add(range(len(mesh.data.vertices)), 1.0, "REPLACE")
        modifier = mesh.modifiers.new(name="GhostStudioRigidAttachmentSkin", type="ARMATURE")
        modifier.object = armature
        converted.append({"mesh": mesh.name, "bone": bone_name})
    armature.data.pose_position = previous_pose_position
    bpy.context.view_layer.update()
    return converted


def main() -> int:
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    actions = _action_evidence()
    if not armatures:
        raise RuntimeError("Ghost GLB did not produce an armature in Blender")
    if not meshes:
        raise RuntimeError("Ghost GLB did not produce any meshes in Blender")
    if not actions:
        raise RuntimeError("Ghost GLB did not produce any animation actions in Blender")
    rigid_attachment_skins = _skin_bone_parented_meshes(armatures[0], meshes)

    selected_action = None
    requested_action = str(args.action or "").strip().casefold()
    if requested_action:
        selected_action = next(
            (action for action in bpy.data.actions if action.name.casefold() == requested_action),
            None,
        )
        if selected_action is None:
            raise RuntimeError(f"Ghost GLB does not contain requested action: {args.action}")
        for armature in armatures:
            if armature.animation_data is None:
                armature.animation_data_create()
            armature.animation_data.action = selected_action
        start, end = selected_action.frame_range
        bpy.context.scene.frame_start = int(round(start))
        bpy.context.scene.frame_end = int(round(end))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="ROOT",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=selected_action is None,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="COPY",
        embed_textures=True,
    )
    if not args.output.is_file() or args.output.stat().st_size < 1024:
        raise RuntimeError(f"Blender did not create a valid FBX: {args.output}")

    report = {
        "schema": "ghoststudio.blender-unreal-fbx-normalization.v1",
        "source": str(args.input),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "armatures": [
            {"name": obj.name, "bones": len(obj.data.bones)} for obj in armatures
        ],
        "meshes": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "materials": [slot.material.name if slot.material else "" for slot in obj.material_slots],
            }
            for obj in meshes
        ],
        "rigid_attachment_skins": rigid_attachment_skins,
        "actions": actions,
        "selected_action": selected_action.name if selected_action else "",
        "scene_fps": float(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
