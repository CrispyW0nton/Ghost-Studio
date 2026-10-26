"""Compare animated source and normalized-FBX head/neck alignment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utc", action="append", required=True)
    parser.add_argument("--action", default="pause1")
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for collection in (
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _clear_actions() -> None:
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data.action = None


def _slot(action: bpy.types.Action, obj: bpy.types.Object):
    identifier = f"OB{obj.name}"
    return next(
        (slot for slot in getattr(action, "slots", []) if slot.identifier == identifier),
        None,
    )


def _assign_source(action_name: str) -> bpy.types.Action:
    action = next(
        item for item in bpy.data.actions if item.name.casefold() == action_name.casefold()
    )
    _clear_actions()
    for obj in bpy.context.scene.objects:
        slot = _slot(action, obj)
        if slot is None:
            continue
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "POSE"
    return action


def _assign_fbx() -> bpy.types.Action:
    armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    armature_slot = f"OB{armature.name}".casefold()
    candidates = [
        item
        for item in bpy.data.actions
        if any(
            slot.identifier.casefold().startswith(armature_slot)
            for slot in getattr(item, "slots", [])
        )
    ]
    if not candidates:
        raise RuntimeError(f"No FBX action authored for armature {armature.name}")
    action = max(candidates, key=lambda item: item.frame_range[1] - item.frame_range[0])
    _clear_actions()
    armature.animation_data_create()
    armature.animation_data.action = action
    slot = _slot(action, armature)
    if slot is not None:
        armature.animation_data.action_slot = slot
    armature.data.pose_position = "POSE"
    return action


def _mesh(role: str) -> bpy.types.Object:
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
        raise RuntimeError(f"No {role} mesh")
    return max(candidates, key=lambda obj: len(obj.data.vertices))


def _evaluated_points(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return [tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def _alignment() -> dict[str, object]:
    head = _mesh("head")
    torso = _mesh("torso")
    head_points = _evaluated_points(head)
    torso_points = _evaluated_points(torso)
    combined = head_points + torso_points
    ranges = [
        max(point[index] for point in combined) - min(point[index] for point in combined)
        for index in range(3)
    ]
    vertical = max(range(3), key=lambda index: ranges[index])
    head_values = [point[vertical] for point in head_points]
    torso_values = [point[vertical] for point in torso_points]
    hmin, hmax = min(head_values), max(head_values)
    tmin, tmax = min(torso_values), max(torso_values)
    if abs(hmin - tmax) <= abs(hmax - tmin):
        hcut = hmin + (hmax - hmin) * 0.10
        tcut = tmax - (tmax - tmin) * 0.035
        head_band = [point for point in head_points if point[vertical] <= hcut]
        torso_band = [point for point in torso_points if point[vertical] >= tcut]
    else:
        hcut = hmax - (hmax - hmin) * 0.10
        tcut = tmin + (tmax - tmin) * 0.035
        head_band = [point for point in head_points if point[vertical] >= hcut]
        torso_band = [point for point in torso_points if point[vertical] <= tcut]
    head_center_x = statistics.median(point[0] for point in head_band)
    torso_center_x = statistics.median(point[0] for point in torso_band)
    return {
        "vertical_axis": "xyz"[vertical],
        "head_center_x": head_center_x,
        "torso_center_x": torso_center_x,
        "head_minus_torso_x": head_center_x - torso_center_x,
    }


def _sample(path: Path, action_name: str, source: bool) -> dict[str, object]:
    _reset()
    if source:
        bpy.ops.import_scene.gltf(filepath=str(path), import_pack_images=True)
        action = _assign_source(action_name)
    else:
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
        action = _assign_fbx()
    start = int(round(action.frame_range[0]))
    end = int(round(action.frame_range[1]))
    frames = sorted({start, end, (start + end) // 2, start + (end - start) // 4, start + 3 * (end - start) // 4})
    samples = []
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples.append({"frame": frame, **_alignment()})
    return {"action": action.name, "frame_range": [start, end], "samples": samples}


def main() -> None:
    args = _args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    wanted = {value.casefold() for value in args.utc}
    rows = []
    for character in receipt["characters"]:
        utc = character["utc_resref"]
        if utc.casefold() not in wanted:
            continue
        source_path = Path(character["source_glb"])
        fbx_path = Path(character["base_fbx"]).parent.parent / "normalized_actions" / f"A_{utc}_{args.action}.fbx"
        source = _sample(source_path, args.action, True)
        fbx = _sample(fbx_path, args.action, False)
        comparisons = []
        for source_sample, fbx_sample in zip(source["samples"], fbx["samples"]):
            comparisons.append(
                {
                    "source_frame": source_sample["frame"],
                    "fbx_frame": fbx_sample["frame"],
                    "source_head_center_x": source_sample["head_center_x"],
                    "fbx_head_center_x": fbx_sample["head_center_x"],
                    "source_torso_center_x": source_sample["torso_center_x"],
                    "fbx_torso_center_x": fbx_sample["torso_center_x"],
                    "source_x": source_sample["head_minus_torso_x"],
                    "fbx_x": fbx_sample["head_minus_torso_x"],
                    "fbx_minus_source_x": fbx_sample["head_minus_torso_x"]
                    - source_sample["head_minus_torso_x"],
                }
            )
        rows.append({"utc_resref": utc, "source": source, "fbx": fbx, "comparisons": comparisons})
    payload = {
        "schema": "ghoststudio.enclave-animated-head-alignment-audit.v1",
        "action": args.action,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
