#!/usr/bin/env python3
"""Dump Blender-imported UV loops and material image bindings for one mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


def _import_asset(path: Path) -> None:
    suffix = path.suffix.casefold()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported input: {path}")


def main() -> int:
    args_after_separator = []
    if "--" in __import__("sys").argv:
        args_after_separator = __import__("sys").argv[
            __import__("sys").argv.index("--") + 1 :
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(args_after_separator)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_asset(args.input)
    candidates = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.casefold() == args.mesh.casefold()
    ]
    if not candidates:
        candidates = [
            obj for obj in bpy.context.scene.objects
            if obj.type == "MESH" and args.mesh.casefold() in obj.name.casefold()
        ]
    if not candidates:
        raise LookupError(f"No mesh matching {args.mesh!r} in {args.input}")
    obj = candidates[0]
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    loops = []
    if uv_layer is not None:
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                uv = uv_layer.data[loop_index].uv
                loops.append(
                    {
                        "polygon": int(polygon.index),
                        "loop": int(loop_index),
                        "vertex": int(mesh.loops[loop_index].vertex_index),
                        "uv": [float(uv.x), float(uv.y)],
                    }
                )
    materials = []
    for slot in obj.material_slots:
        material = slot.material
        images = []
        if material is not None and material.use_nodes:
            images = [
                str(node.image.filepath)
                for node in material.node_tree.nodes
                if node.type == "TEX_IMAGE" and node.image is not None
            ]
        materials.append({"name": material.name if material else "", "images": images})
    payload = {
        "input": str(args.input),
        "object": obj.name,
        "vertex_count": len(mesh.vertices),
        "polygon_count": len(mesh.polygons),
        "uv_layer": uv_layer.name if uv_layer is not None else "",
        "loops": loops,
        "materials": materials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "loops"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
