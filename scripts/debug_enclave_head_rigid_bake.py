"""Isolate rigid head rest-space baking math for the enclave BAS pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _bounds(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    data = evaluated.to_mesh()
    try:
        points = [evaluated.matrix_world @ vertex.co for vertex in data.vertices]
    finally:
        evaluated.to_mesh_clear()
    return {
        "min": [min(float(point[i]) for point in points) for i in range(3)],
        "max": [max(float(point[i]) for point in points) for i in range(3)],
    }


def main():
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    head = next(obj for obj in bpy.context.scene.objects if obj.name.casefold() == "head")
    armature.data.pose_position = "REST"
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data.action = None
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    before = _bounds(head)
    world = head.matrix_world.copy()
    head.data = head.data.copy()
    head.data.transform(armature.matrix_world.inverted_safe() @ world)
    head.parent = armature
    head.parent_type = "OBJECT"
    head.parent_bone = ""
    head.matrix_parent_inverse = Matrix.Identity(4)
    head.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    after_bake = _bounds(head)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature.data.edit_bones.new("bas_head_debug")
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 0.025, 0.0)
    bone_name = bone.name
    bpy.ops.object.mode_set(mode="OBJECT")
    head.vertex_groups.clear()
    group = head.vertex_groups.new(name=bone_name)
    group.add(range(len(head.data.vertices)), 1.0, "REPLACE")
    modifier = head.modifiers.new(name="DebugArmature", type="ARMATURE")
    modifier.object = armature
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    after_modifier = _bounds(head)
    payload = {
        "before": before,
        "after_bake": after_bake,
        "after_modifier": after_modifier,
        "stored_world": [float(value) for row in world for value in row],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
