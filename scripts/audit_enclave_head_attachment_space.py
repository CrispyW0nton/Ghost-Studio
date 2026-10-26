"""Audit KOTOR BAS head attachment hierarchy against the body armature."""

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


def _vec(value) -> list[float]:
    return [round(float(axis), 8) for axis in value]


def main() -> int:
    args = _args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input), import_pack_images=True)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    rows = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name.casefold()):
        if obj.name.casefold() not in {
            "headhook", "neck_g", "necklwr_g", "hturn_g", "head_g",
            "rootdummy__head", "torso_g__head", "torsoupr_g__head",
        }:
            continue
        rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else "",
                "parent_type": obj.parent_type,
                "parent_bone": obj.parent_bone,
                "local_location": _vec(obj.location),
                "world_location": _vec(obj.matrix_world.translation),
                "local_matrix": [round(float(v), 8) for row in obj.matrix_local for v in row],
                "world_matrix": [round(float(v), 8) for row in obj.matrix_world for v in row],
            }
        )
    bones = []
    for armature in armatures:
        for bone in armature.data.bones:
            bones.append(
                {
                    "armature": armature.name,
                    "name": bone.name,
                    "parent": bone.parent.name if bone.parent else "",
                    "head_local": _vec(bone.head_local),
                    "tail_local": _vec(bone.tail_local),
                }
            )
    payload = {
        "schema": "ghostrigger.enclave-head-attachment-audit.v1",
        "input": str(args.input),
        "objects": rows,
        "bones": bones,
        "actions": [
            {
                "name": action.name,
                "frame_range": [float(action.frame_range[0]), float(action.frame_range[1])],
                "slots": [
                    {
                        "identifier": slot.identifier,
                        "target_id_type": slot.target_id_type,
                    }
                    for slot in getattr(action, "slots", [])
                ],
            }
            for action in bpy.data.actions
        ],
        "active_actions": [
            {
                "object": obj.name,
                "action": obj.animation_data.action.name
                if obj.animation_data and obj.animation_data.action
                else "",
                "slot": obj.animation_data.action_slot.identifier
                if obj.animation_data
                and obj.animation_data.action
                and obj.animation_data.action_slot
                else "",
            }
            for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name.casefold())
            if obj.animation_data is not None
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
