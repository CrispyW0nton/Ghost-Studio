"""Dump Blender-imported world-space vertices for named mesh registration audits."""

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
    parser.add_argument("--mesh", action="append", default=[])
    parser.add_argument("--rest-pose", action="store_true")
    parser.add_argument("--clear-actions", action="store_true")
    return parser.parse_args(argv)


def _import(path: Path) -> None:
    suffix = path.suffix.casefold()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path), import_pack_images=True)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset: {path}")


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import(args.input)
    if args.clear_actions:
        for obj in bpy.context.scene.objects:
            if obj.animation_data is not None:
                obj.animation_data.action = None
    if args.rest_pose:
        for obj in bpy.context.scene.objects:
            if obj.type == "ARMATURE":
                obj.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    requested = {name.casefold() for name in args.mesh}
    rows = []
    for obj in sorted(bpy.context.scene.objects, key=lambda value: value.name.casefold()):
        if obj.type != "MESH" or (requested and obj.name.casefold() not in requested):
            continue
        world_vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
        rows.append(
            {
                "name": obj.name,
                "parent": obj.parent.name if obj.parent else "",
                "matrix_world": [float(value) for row in obj.matrix_world for value in row],
                "vertices_world": [[float(value) for value in vertex] for vertex in world_vertices],
                "vertices_local": [[float(value) for value in vertex.co] for vertex in obj.data.vertices],
            }
        )
    missing = sorted(requested - {row["name"].casefold() for row in rows})
    if missing:
        raise LookupError(f"Missing requested mesh(es): {missing}")
    payload = {"schema": "ghostrigger.blender-mesh-registration.v1", "input": str(args.input), "meshes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mesh_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
