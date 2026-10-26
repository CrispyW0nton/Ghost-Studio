"""Measure the actual open neck loops in normalized enclave humanoid FBXs.

The earlier band-centre audit intentionally used broad surface bands, which is
useful for export-regression detection but can be biased by collars, lekku, and
shoulder geometry.  This diagnostic finds connected boundary loops on the
separate head and torso meshes and chooses the nearest pair.  Their centroid
delta is the cosmetic seam offset a viewer can actually perceive.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _reset_scene() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for collection in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _import_rest(path: Path) -> None:
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data.action = None
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def _mesh_by_role(role: str) -> bpy.types.Object:
    exact = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.casefold() == role
    ]
    candidates = exact or [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and role in obj.name.casefold()
    ]
    if not candidates:
        raise RuntimeError(f"No {role!r} mesh in imported scene")
    return max(candidates, key=lambda obj: len(obj.data.vertices))


def _boundary_components(obj: bpy.types.Object) -> list[dict[str, object]]:
    counts: Counter[tuple[int, int]] = Counter()
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for index, vertex in enumerate(vertices):
            other = vertices[(index + 1) % len(vertices)]
            counts[tuple(sorted((int(vertex), int(other))))] += 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (left, right), count in counts.items():
        if count != 1:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = []
    unvisited = set(adjacency)
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        indices = {seed}
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in indices:
                    continue
                indices.add(neighbor)
                unvisited.discard(neighbor)
                stack.append(neighbor)
        if len(indices) < 3:
            continue
        points = [obj.matrix_world @ obj.data.vertices[index].co for index in indices]
        center = tuple(statistics.median(point[axis] for point in points) for axis in range(3))
        components.append(
            {
                "vertices": len(indices),
                "center": center,
                "extent": [
                    max(point[axis] for point in points)
                    - min(point[axis] for point in points)
                    for axis in range(3)
                ],
            }
        )
    return components


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _measure(path: Path) -> dict[str, object]:
    _reset_scene()
    _import_rest(path)
    head = _mesh_by_role("head")
    torso = _mesh_by_role("torso")
    head_loops = _boundary_components(head)
    torso_loops = _boundary_components(torso)
    if not head_loops or not torso_loops:
        raise RuntimeError(f"No compatible boundary loops in {path}")
    head_loop, torso_loop = min(
        (
            (head_loop, torso_loop)
            for head_loop in head_loops
            for torso_loop in torso_loops
        ),
        key=lambda pair: _distance(pair[0]["center"], pair[1]["center"]),
    )
    delta = [
        head_loop["center"][axis] - torso_loop["center"][axis]
        for axis in range(3)
    ]
    return {
        "head_mesh": head.name,
        "torso_mesh": torso.name,
        "head_boundary_count": len(head_loops),
        "torso_boundary_count": len(torso_loops),
        "head_neck_loop": head_loop,
        "torso_neck_loop": torso_loop,
        "head_minus_torso": {axis: delta[index] for index, axis in enumerate("xyz")},
        "centroid_distance": _distance(head_loop["center"], torso_loop["center"]),
    }


def main() -> None:
    args = _arguments()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rows = []
    for character in receipt["characters"]:
        if not character.get("bas_head_attachment_skins"):
            continue
        row = {"utc_resref": character["utc_resref"]}
        try:
            row.update(_measure(Path(character["base_fbx"])))
            row["success"] = True
        except Exception as exc:  # keep the whole roster audit inspectable
            row.update({"success": False, "error": str(exc)})
        rows.append(row)
    payload = {
        "schema": "ghoststudio.enclave-neck-boundary-alignment.v1",
        "character_count": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "success_count": payload["success_count"]}))


if __name__ == "__main__":
    main()
