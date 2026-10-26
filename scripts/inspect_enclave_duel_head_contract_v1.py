"""Read-only Blender inspection of canonical rigid-head and source object hierarchies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    values = list(__import__("sys").argv)
    values = values[values.index("--") + 1 :] if "--" in values else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fbx", type=Path, required=True)
    parser.add_argument("--source-fbx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def _import(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def _matrix_translation(matrix) -> list[float]:
    return [float(value) for value in matrix.translation]


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    target_objects = _import(args.base_fbx)
    target = next(obj for obj in target_objects if obj.type == "ARMATURE")
    target_rows = []
    for bone in target.data.bones:
        if bone.name.startswith("bas_head_") or "head" in bone.name.casefold() or "neck" in bone.name.casefold():
            target_rows.append(
                {
                    "name": bone.name,
                    "parent": bone.parent.name if bone.parent else None,
                    "rest_translation": _matrix_translation(bone.matrix_local),
                    "use_deform": bool(bone.use_deform),
                }
            )

    source_objects = _import(args.source_fbx)
    source = next(obj for obj in source_objects if obj.type == "ARMATURE")
    action_frames = [
        obj.animation_data.action.frame_range[:]
        for obj in source_objects
        if obj.animation_data and obj.animation_data.action
    ]
    start = int(min(frame[0] for frame in action_frames))
    end = int(max(frame[1] for frame in action_frames))
    source_rows = []
    for obj in source_objects:
        if obj.type != "MESH":
            continue
        if obj.name.casefold() not in {
            "head", "mesh01", "eyela", "eyelıd", "eyelid", "eyera", "eyerıd",
            "teethlower", "teethupper", "tongue",
        } and not any(token in obj.name.casefold() for token in ("head", "eye", "teeth", "tongue", "mesh01")):
            continue
        samples = []
        for frame in (start, (start + end) // 2, end):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            samples.append({"frame": frame, "world_translation": _matrix_translation(obj.matrix_world)})
        source_rows.append(
            {
                "name": obj.name,
                "parent": obj.parent.name if obj.parent else None,
                "parent_type": obj.parent_type,
                "parent_bone": obj.parent_bone,
                "local_translation": _matrix_translation(obj.matrix_local),
                "samples": samples,
            }
        )
    payload = {
        "target_armature_world_translation": _matrix_translation(target.matrix_world),
        "source_armature_world_translation": _matrix_translation(source.matrix_world),
        "target_bones": target_rows,
        "source_head_objects": source_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MCPSTUDIO_HEAD_CONTRACT=" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
