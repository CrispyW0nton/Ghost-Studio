"""Measure rest/animation stability and UV/material fidelity across BAS interchange files.

This is a read-only diagnostic.  It imports each supplied GLB/FBX into a clean
Blender scene, samples evaluated mesh bounds, and emits deterministic JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _evaluated_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector] | None:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if not mesh.vertices:
            return None
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        lower = Vector(tuple(min(point[index] for point in points) for index in range(3)))
        upper = Vector(tuple(max(point[index] for point in points) for index in range(3)))
        return lower, upper
    finally:
        evaluated.to_mesh_clear()


def _scene_bounds() -> dict[str, object]:
    lowers: list[Vector] = []
    uppers: list[Vector] = []
    mesh_rows: list[dict[str, object]] = []
    for obj in sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name.casefold(),
    ):
        bounds = _evaluated_bounds(obj)
        if bounds is None:
            continue
        lower, upper = bounds
        lowers.append(lower)
        uppers.append(upper)
        mesh_rows.append({
            "name": obj.name,
            "min": [round(float(value), 6) for value in lower],
            "max": [round(float(value), 6) for value in upper],
        })
    if not lowers:
        return {"valid": False, "mesh_bounds": mesh_rows}
    lower = Vector(tuple(min(value[index] for value in lowers) for index in range(3)))
    upper = Vector(tuple(max(value[index] for value in uppers) for index in range(3)))
    size = upper - lower
    values = tuple(lower) + tuple(upper) + tuple(size)
    return {
        "valid": _finite(values),
        "min": [round(float(value), 6) for value in lower],
        "max": [round(float(value), 6) for value in upper],
        "size": [round(float(value), 6) for value in size],
        "diagonal": round(float(size.length), 6),
        "mesh_bounds": mesh_rows,
    }


def _uv_record(obj: bpy.types.Object) -> dict[str, object]:
    layers = []
    for layer in obj.data.uv_layers:
        values = [(float(loop.uv.x), float(loop.uv.y)) for loop in layer.data]
        layers.append({
            "name": layer.name,
            "count": len(values),
            "min": [round(min(value[index] for value in values), 6) for index in range(2)] if values else [],
            "max": [round(max(value[index] for value in values), 6) for index in range(2)] if values else [],
            "finite": all(_finite(value) for value in values),
        })
    materials = []
    for slot in obj.material_slots:
        material = slot.material
        images = []
        if material and material.use_nodes:
            images = sorted({
                str(node.image.filepath or node.image.name)
                for node in material.node_tree.nodes
                if node.type == "TEX_IMAGE" and node.image
            })
        materials.append({
            "name": material.name if material else "",
            "images": images,
        })
    return {
        "name": obj.name,
        "vertex_count": len(obj.data.vertices),
        "polygon_count": len(obj.data.polygons),
        "parent": obj.parent.name if obj.parent else "",
        "parent_type": obj.parent_type,
        "parent_bone": obj.parent_bone,
        "vertex_groups": sorted(group.name for group in obj.vertex_groups),
        "modifiers": [modifier.type for modifier in obj.modifiers],
        "uv_layers": layers,
        "materials": materials,
    }


def _sample_frames() -> list[int]:
    actions = list(bpy.data.actions)
    frames = {int(bpy.context.scene.frame_start), int(bpy.context.scene.frame_end), 0}
    for action in actions:
        start, end = action.frame_range
        frames.update({int(round(start)), int(round((start + end) * 0.5)), int(round(end))})
    return sorted(frames)


def _import(path: Path) -> None:
    if path.suffix.casefold() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path), import_pack_images=True)
    else:
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)


def _audit(path: Path) -> dict[str, object]:
    _reset_scene()
    _import(path)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    actions = sorted(bpy.data.actions, key=lambda item: item.name.casefold())

    for armature in armatures:
        armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    rest_bounds = _scene_bounds()

    for armature in armatures:
        armature.data.pose_position = "POSE"
    samples = []
    for frame in _sample_frames():
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        bounds = _scene_bounds()
        rest_diagonal = float(rest_bounds.get("diagonal", 0.0) or 0.0)
        diagonal = float(bounds.get("diagonal", 0.0) or 0.0)
        samples.append({
            "frame": frame,
            "bounds": bounds,
            "diagonal_vs_rest": round(diagonal / rest_diagonal, 6) if rest_diagonal > 1e-9 else None,
        })

    return {
        "path": str(path),
        "armatures": [{
            "name": obj.name,
            "bone_count": len(obj.data.bones),
            "world_matrix": [[round(float(value), 6) for value in row] for row in obj.matrix_world],
        } for obj in armatures],
        "actions": [{
            "name": action.name,
            "frame_range": [round(float(value), 3) for value in action.frame_range],
        } for action in actions],
        "rest_bounds": rest_bounds,
        "pose_samples": samples,
        "meshes": [_uv_record(obj) for obj in sorted(meshes, key=lambda item: item.name.casefold())],
    }


def main() -> int:
    args = _args()
    report = {
        "schema": "ghoststudio.enclave-character-pipeline-audit.v1",
        "inputs": [_audit(path.resolve()) for path in args.input],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "inputs": len(report["inputs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
