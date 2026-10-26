"""Measure source-vs-FBX humanoid head/neck centerline alignment.

Run inside Blender in background mode.  This is a read-only diagnostic: it
imports each source GLB and normalized FBX into a clean scene, samples the
authored rest pose, and compares the horizontal center of the lowest head
vertices with the highest torso vertices.  A source/FBX delta identifies an
export-space regression; a matching offset identifies authored hook placement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import bpy


def _arguments() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _reset_scene() -> None:
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # glTF actions may carry fake users, so remove them explicitly before the
    # generic orphan cleanup.  Otherwise a later character can inherit stale
    # animation data during a multi-character audit.
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def _clear_animation_and_set_rest() -> None:
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data.action = None
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def _import(path: Path) -> None:
    suffix = path.suffix.casefold()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    else:
        raise ValueError(f"Unsupported asset: {path}")
    _clear_animation_and_set_rest()


def _world_points(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]


def _mesh_by_role(role: str) -> bpy.types.Object:
    candidates = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.casefold() == role
    ]
    if not candidates:
        candidates = [
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and role in obj.name.casefold()
        ]
    if not candidates:
        raise RuntimeError(f"No {role!r} mesh in imported scene")
    return max(candidates, key=lambda obj: len(obj.data.vertices))


def _points_for_vertex_groups(
    obj: bpy.types.Object, group_names: set[str]
) -> list[tuple[float, float, float]]:
    group_indices = {
        group.index
        for group in obj.vertex_groups
        if group.name.casefold() in group_names
    }
    if not group_indices:
        return []
    points = []
    for vertex in obj.data.vertices:
        if any(
            membership.group in group_indices and membership.weight > 0.01
            for membership in vertex.groups
        ):
            points.append(tuple(obj.matrix_world @ vertex.co))
    return points


def _median_center(points: list[tuple[float, float, float]]) -> dict[str, float]:
    return {
        "x": statistics.median(point[0] for point in points),
        "y": statistics.median(point[1] for point in points),
        "z": statistics.median(point[2] for point in points),
    }


def _measure_point_sets(
    head_points: list[tuple[float, float, float]],
    torso_points: list[tuple[float, float, float]],
) -> dict[str, object]:
    combined = head_points + torso_points
    axis_ranges = [
        max(point[index] for point in combined) - min(point[index] for point in combined)
        for index in range(3)
    ]
    vertical_axis = max(range(3), key=lambda index: axis_ranges[index])
    head_axis = [point[vertical_axis] for point in head_points]
    torso_axis = [point[vertical_axis] for point in torso_points]
    head_min, head_max = min(head_axis), max(head_axis)
    torso_min, torso_max = min(torso_axis), max(torso_axis)
    if abs(head_min - torso_max) <= abs(head_max - torso_min):
        head_cutoff = head_min + (head_max - head_min) * 0.10
        torso_cutoff = torso_max - (torso_max - torso_min) * 0.035
        head_band = [
            point for point in head_points if point[vertical_axis] <= head_cutoff
        ]
        torso_band = [
            point for point in torso_points if point[vertical_axis] >= torso_cutoff
        ]
        seam = "head_min_to_torso_max"
    else:
        head_cutoff = head_max - (head_max - head_min) * 0.10
        torso_cutoff = torso_min + (torso_max - torso_min) * 0.035
        head_band = [
            point for point in head_points if point[vertical_axis] >= head_cutoff
        ]
        torso_band = [
            point for point in torso_points if point[vertical_axis] <= torso_cutoff
        ]
        seam = "head_max_to_torso_min"
    head_center = _median_center(head_band)
    torso_center = _median_center(torso_band)
    return {
        "vertical_axis": "xyz"[vertical_axis],
        "seam": seam,
        "head_band_vertices": len(head_band),
        "torso_band_vertices": len(torso_band),
        "head_center": head_center,
        "torso_center": torso_center,
        "head_minus_torso": {
            axis: head_center[axis] - torso_center[axis] for axis in ("x", "y", "z")
        },
    }


def _measure() -> dict[str, object]:
    try:
        head = _mesh_by_role("head")
        torso = _mesh_by_role("torso")
        result = _measure_point_sets(_world_points(head), _world_points(torso))
        result.update(
            {
                "measurement_mode": "separate_meshes",
                "head_mesh": head.name,
                "torso_mesh": torso.name,
            }
        )
        return result
    except RuntimeError:
        # Unreal's FBX exporter combines skeletal mesh sections into one mesh,
        # but preserves the rigid KOTOR head and torso vertex groups.  Recover
        # the same semantic point sets from those groups for a bind-space
        # round-trip comparison.
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if not meshes:
            raise
        combined = max(meshes, key=lambda obj: len(obj.data.vertices))
        head_points = _points_for_vertex_groups(
            combined, {"bas_head_head", "head_g"}
        )
        torso_points = _points_for_vertex_groups(
            combined, {"torso_g", "torsoupr_g"}
        )
        if not head_points or not torso_points:
            raise RuntimeError(
                f"Combined mesh {combined.name!r} has no semantic head/torso groups"
            )
        result = _measure_point_sets(head_points, torso_points)
        result.update(
            {
                "measurement_mode": "combined_vertex_groups",
                "head_mesh": combined.name,
                "torso_mesh": combined.name,
            }
        )
        return result


def main() -> None:
    args = _arguments()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rows = []
    for character in receipt["characters"]:
        source = Path(character["source_glb"])
        normalized = Path(character["base_fbx"])
        row = {"utc_resref": character["utc_resref"]}
        for lane, path in (("source", source), ("fbx", normalized)):
            _reset_scene()
            _import(path)
            row[lane] = _measure()
        row["fbx_minus_source_head_offset"] = {
            axis: row["fbx"]["head_minus_torso"][axis]
            - row["source"]["head_minus_torso"][axis]
            for axis in ("x", "y", "z")
        }
        rows.append(row)
    payload = {
        "schema": "ghoststudio.enclave-neck-alignment-audit.v1",
        "character_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"character_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
