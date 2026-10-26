"""Render selected frames from one action embedded in a player multi-take FBX."""

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
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--action-name-contains", default="")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(values)


def _look_at(camera, point: Vector) -> None:
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx), use_anim=True, automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise RuntimeError(
            "animation FBX must contain one armature and at least one mesh; "
            f"found {len(armatures)} armatures and {len(meshes)} meshes"
        )
    candidates = [
        action for action in bpy.data.actions
        if (
            not args.frame_count
            or int(round(action.frame_range[1] - action.frame_range[0] + 1)) == args.frame_count
        )
        and (
            not args.action_name_contains
            or args.action_name_contains.lower() in action.name.lower()
        )
    ]
    candidates.sort(key=lambda action: action.name)
    if not candidates or not 0 <= args.candidate_index < len(candidates):
        raise RuntimeError(
            f"expected candidate {args.candidate_index} among {args.frame_count or 'any'}-frame "
            f"actions, found {[a.name for a in candidates]}"
        )
    armature = armatures[0]
    armature.animation_data_create()
    selected_action = candidates[args.candidate_index]
    armature.animation_data.action = selected_action

    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (low + high) * 0.5
    height = max(high.z - low.z, 1.0)
    bpy.ops.object.camera_add(location=center + Vector((height * 1.35, -height * 2.5, height * 0.2)))
    camera = bpy.context.object
    camera.data.lens = 62.0
    _look_at(camera, center)

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 700
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = int(round(selected_action.frame_range[0]))
    end = int(round(selected_action.frame_range[1]))
    samples = sorted({start, start + (end - start) // 4, start + (end - start) // 2, end})
    renders = []
    for frame in samples:
        scene.frame_set(frame)
        destination = args.output_dir / f"frame_{frame:04d}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        renders.append({"frame": frame, "path": str(destination), "bytes": destination.stat().st_size})
    receipt = {
        "schema": "ghoststudio.enclave.player-multitake-visual-audit.v1",
        "fbx": str(args.fbx),
        "action": selected_action.name,
        "frame_range": [start, end],
        "renders": renders,
        "all_checks_passed": all(item["bytes"] > 10_000 for item in renders),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
