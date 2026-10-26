"""Render a player mesh using a native KOTOR FBX action without sidecars.

This is a visual diagnostic gate.  The player must already be bound to the
same exact KOTOR skeleton as the action FBX.  We copy the native Blender FBX
action directly so no quaternion reconstruction or Unreal conversion can hide
an axis/rest-pose error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(values)


def _look_at(camera, point: Vector) -> None:
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def _world_bounds(objects) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return low, high


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.player), use_anim=False, automatic_bone_orientation=False)
    target_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    target_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(target_armatures) != 1 or len(target_meshes) != 1:
        raise RuntimeError("Player FBX must contain exactly one armature and one mesh")
    target = target_armatures[0]

    bpy.ops.import_scene.fbx(filepath=str(args.action), automatic_bone_orientation=False)
    sources = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj != target]
    if len(sources) != 1:
        raise RuntimeError("Action FBX must contain exactly one armature")
    source = sources[0]
    action = source.animation_data.action if source.animation_data else None
    if action is None:
        raise RuntimeError("Action FBX has no native animation")

    target_names = set(target.data.bones.keys())
    source_names = set(source.data.bones.keys())
    if target_names != source_names:
        raise RuntimeError(
            f"Skeleton names differ: target-only={sorted(target_names-source_names)}, "
            f"source-only={sorted(source_names-target_names)}"
        )
    rest_differences = {}
    for name in sorted(target_names):
        left = target.data.bones[name].matrix_local
        right = source.data.bones[name].matrix_local
        difference = max(abs(float(left[row][column] - right[row][column])) for row in range(4) for column in range(4))
        if difference > 1.0e-5:
            rest_differences[name] = difference

    target.animation_data_create()
    target.animation_data.action = action
    for obj in list(bpy.context.scene.objects):
        if obj == source or (obj.type == "MESH" and obj not in target_meshes):
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Native KOTOR Action Audit")
    scene.world.color = (0.055, 0.065, 0.08)

    low, high = _world_bounds(target_meshes)
    center = (low + high) * 0.5
    dimensions = high - low
    bpy.ops.object.camera_add(location=center + Vector((3.2, 0.0, 3.6)))
    camera = bpy.context.object
    camera.data.lens = 58.0
    _look_at(camera, center)
    scene.camera = camera

    start = int(round(action.frame_range[0]))
    end = int(round(action.frame_range[1]))
    samples = sorted({start, start + max(1, (end - start) // 4), start + max(1, (end - start) // 2)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renders = []
    for frame in samples:
        scene.frame_set(frame)
        destination = args.output_dir / f"native_action_frame_{frame:04d}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        renders.append({"frame": frame, "path": str(destination), "bytes": destination.stat().st_size})

    receipt = {
        "schema": "ghoststudio.enclave.player-native-action-audit.v6",
        "player": str(args.player),
        "action": str(args.action),
        "action_name": action.name,
        "skeleton_bones": len(target_names),
        "rest_matrix_differences": rest_differences,
        "rest_matrices_exact": not rest_differences,
        "mesh_dimensions_m": [float(dimensions.x), float(dimensions.y), float(dimensions.z)],
        "renders": renders,
        "all_checks_passed": not rest_differences and all(item["bytes"] >= 10_000 for item in renders),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
