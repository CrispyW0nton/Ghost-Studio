"""Print Blender's imported hierarchy/action layout for a rigid animated GLB."""

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
    return parser.parse_args(argv)


def main() -> int:
    args = _args()
    bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)
    animated_objects = [
        obj for obj in bpy.context.scene.objects if obj.type in {"EMPTY", "MESH"}
    ]
    cwalk = next(
        (action for action in bpy.data.actions if action.name.casefold() == "cwalk"),
        None,
    )
    reassignment_diffs = []
    if cwalk is not None:
        original = {}
        for frame in (0, 4, 8, 16):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            original[frame] = {
                obj.name: obj.matrix_world.copy() for obj in animated_objects
            }
        for obj in animated_objects:
            if obj.animation_data is not None:
                obj.animation_data.action = None
        for obj in animated_objects:
            slot = next(
                (
                    slot
                    for slot in cwalk.slots
                    if slot.identifier == f"OB{obj.name}"
                ),
                None,
            )
            if slot is None:
                continue
            obj.animation_data_create()
            obj.animation_data.action = cwalk
            obj.animation_data.action_slot = slot
        for frame in (0, 4, 8, 16):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            for obj in animated_objects:
                before = original[frame][obj.name]
                after = obj.matrix_world
                delta = max(
                    abs(before[row][column] - after[row][column])
                    for row in range(4)
                    for column in range(4)
                )
                if delta > 1.0e-6:
                    reassignment_diffs.append(
                        {"frame": frame, "object": obj.name, "max_matrix_delta": delta}
                    )
    objects = []
    for obj in bpy.context.scene.objects:
        animation = obj.animation_data
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "parent_type": obj.parent_type,
                "parent_bone": obj.parent_bone,
                "action": animation.action.name if animation and animation.action else None,
                "matrix_local": [list(row) for row in obj.matrix_local],
            }
        )
    actions = []
    for action in bpy.data.actions:
        slots = []
        for slot in getattr(action, "slots", []):
            channelbag = action.layers[0].strips[0].channelbag(slot)
            slots.append(
                {
                    "identifier": slot.identifier,
                    "target_id_type": slot.target_id_type,
                    "fcurves": [
                        {
                            "path": curve.data_path,
                            "index": curve.array_index,
                            "keys": len(curve.keyframe_points),
                        }
                        for curve in (channelbag.fcurves if channelbag else [])
                    ],
                }
            )
        actions.append(
            {
                "name": action.name,
                "frame_range": list(action.frame_range),
                "slots": slots,
            }
        )
    payload = {
        "objects": objects,
        "actions": actions,
        "reassignment_diffs": reassignment_diffs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "objects": len(objects),
        "actions": len(actions),
        "reassignment_diffs": len(reassignment_diffs),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
