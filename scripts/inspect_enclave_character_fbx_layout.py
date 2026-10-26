"""Print world-space mesh bounds for a normalized enclave character FBX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rest-pose", action="store_true")
    return parser.parse_args(argv)


def _world_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    corners = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return (
        Vector(tuple(min(corner[index] for corner in corners) for index in range(3))),
        Vector(tuple(max(corner[index] for corner in corners) for index in range(3))),
    )


def _parent_chain(obj: bpy.types.Object) -> list[str]:
    chain = []
    parent = obj.parent
    while parent is not None:
        chain.append(parent.name)
        parent = parent.parent
    return chain


def main() -> int:
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if args.input.suffix.casefold() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)
    else:
        bpy.ops.import_scene.fbx(filepath=str(args.input), automatic_bone_orientation=False)
    if args.rest_pose:
        for armature in (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"):
            armature.data.pose_position = "REST"
        bpy.context.view_layer.update()
    for action in sorted(bpy.data.actions, key=lambda item: item.name.casefold()):
        slots = [
            (slot.identifier, slot.target_id_type)
            for slot in getattr(action, "slots", [])
        ]
        channel_count = 0
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for bag in getattr(strip, "channelbags", []):
                    channel_count += len(getattr(bag, "fcurves", []))
        print(
            "ACTION",
            action.name,
            "range=",
            tuple(round(float(value), 3) for value in action.frame_range),
            "slots=",
            slots,
            "channels=",
            channel_count,
        )
    for armature in sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda item: item.name.casefold(),
    ):
        for bone_name in (
            "rootdummy",
            "pelvis_g",
            "headhook",
            "head_g",
            "bas_head_head",
            "bas_head_eyeLA",
        ):
            bone = armature.data.bones.get(bone_name)
            if bone is None:
                continue
            head = armature.matrix_world @ bone.head_local
            tail = armature.matrix_world @ bone.tail_local
            print(
                "BONE",
                bone_name,
                "head=",
                tuple(round(float(value), 5) for value in head),
                "tail=",
                tuple(round(float(value), 5) for value in tail),
            )
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name.casefold()):
        if obj.type != "MESH":
            continue
        lower, upper = _world_bounds(obj)
        raw_points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        raw_lower = Vector(
            tuple(min(point[index] for point in raw_points) for index in range(3))
        )
        raw_upper = Vector(
            tuple(max(point[index] for point in raw_points) for index in range(3))
        )
        print(
            "MESH",
            obj.name,
            "parent=",
            obj.parent.name if obj.parent else "",
            "parent_type=",
            obj.parent_type,
            "parent_bone=",
            obj.parent_bone,
            "parent_chain=",
            _parent_chain(obj),
            "modifiers=",
            [(modifier.type, getattr(modifier, "object", None).name if getattr(modifier, "object", None) else "") for modifier in obj.modifiers],
            "origin=",
            tuple(round(float(value), 5) for value in obj.matrix_world.translation),
            "min=",
            tuple(round(float(value), 5) for value in lower),
            "max=",
            tuple(round(float(value), 5) for value in upper),
            "raw_min=",
            tuple(round(float(value), 5) for value in raw_lower),
            "raw_max=",
            tuple(round(float(value), 5) for value in raw_upper),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
