"""Render a neutral character FBX for visual rest-pose comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lod-zero-only", action="store_true")
    return parser.parse_args(values)


def look_at(camera, point: Vector) -> None:
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> int:
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx), use_anim=False, automatic_bone_orientation=False)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if args.lod_zero_only:
        meshes = [
            obj for obj in meshes
            if obj.name.endswith("_0")
            and not obj.name.startswith("*")
            and "_cap_" not in obj.name
            and not obj.name.endswith("_off_0")
        ]
        visible = set(meshes)
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH" and obj not in visible:
                obj.hide_render = True
    if not meshes:
        raise RuntimeError("Reference FBX has no renderable meshes")
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (low + high) * 0.5
    height = max(high.z - low.z, 1.0)
    width = max(high.x - low.x, 1.0)
    distance = max(height, width) * 2.2
    bpy.ops.object.camera_add(location=center + Vector((distance * 0.10, -distance, height * 0.05)))
    camera = bpy.context.object
    camera.data.lens = 58.0
    look_at(camera, center)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 700
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)
    receipt = {
        "schema": "ghoststudio.character-reference-pose-render.v1",
        "fbx": str(args.fbx),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "mesh_count": len(meshes),
        "bounds": {"minimum": list(low), "maximum": list(high)},
        "all_checks_passed": args.output.stat().st_size > 10_000,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
