"""Report evaluated per-mesh bounds for a GLB/FBX character frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    return parser.parse_args(argv)


def main() -> int:
    args = _args()
    if args.input.suffix.casefold() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)
    else:
        bpy.ops.import_scene.fbx(filepath=str(args.input), automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if bpy.data.actions:
        action = sorted(bpy.data.actions, key=lambda item: item.name.casefold())[0]
        for armature in armatures:
            armature.data.pose_position = "POSE"
            armature.animation_data_create()
            armature.animation_data.action = action
    bpy.context.scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or "shadow" in obj.name.casefold():
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
        lower = [min(point[index] for point in points) for index in range(3)]
        upper = [max(point[index] for point in points) for index in range(3)]
        center = [(lower[index] + upper[index]) * 0.5 for index in range(3)]
        rows.append(
            {
                "name": obj.name,
                "lower": lower,
                "upper": upper,
                "center": center,
                "extent": [upper[index] - lower[index] for index in range(3)],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"meshes": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "meshes": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
