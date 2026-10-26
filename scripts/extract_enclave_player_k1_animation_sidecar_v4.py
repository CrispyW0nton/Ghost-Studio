"""Sample a real KOTOR humanoid FBX action into rest-relative bone deltas.

The UE 5.6 FBX animation-only import path used by the Enclave project creates
track names and frame counts but drops every key.  This sidecar keeps the
authentic Ghost Studio/KOTOR action and lets Unreal's AnimationDataController
author the keys explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    return parser.parse_args(values)


def _quat_xyzw(quaternion) -> list[float]:
    value = quaternion.normalized()
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def _continuous(previous: list[float] | None, current: list[float]) -> list[float]:
    if previous is None:
        return current
    dot = sum(a * b for a, b in zip(previous, current))
    return [-value for value in current] if dot < 0.0 else current


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=str(args.source),
        automatic_bone_orientation=False,
    )
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one action armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("KOTOR action FBX has no active action")

    start = int(round(action.frame_range[0]))
    end = int(round(action.frame_range[1]))
    fps = float(bpy.context.scene.render.fps) / max(
        1.0, float(bpy.context.scene.render.fps_base)
    )
    bones = [bone.name for bone in armature.data.bones]
    rest_local = {}
    for bone in armature.data.bones:
        rest_local[bone.name] = (
            bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
            if bone.parent
            else bone.matrix_local.copy()
        )

    tracks = {name: {"rotations_xyzw": []} for name in bones}
    previous = {name: None for name in bones}
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for pose_bone in armature.pose.bones:
            pose_local = (
                pose_bone.parent.matrix.inverted_safe() @ pose_bone.matrix
                if pose_bone.parent
                else pose_bone.matrix.copy()
            )
            delta = rest_local[pose_bone.name].inverted_safe() @ pose_local
            rotation = _continuous(previous[pose_bone.name], _quat_xyzw(delta.to_quaternion()))
            previous[pose_bone.name] = rotation
            tracks[pose_bone.name]["rotations_xyzw"].append(rotation)

    moving = []
    for name, track in tracks.items():
        rotations = track["rotations_xyzw"]
        if rotations:
            anchor = rotations[0]
            max_delta = max(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(anchor, sample)))
                for sample in rotations
            )
        else:
            max_delta = 0.0
        track["max_quaternion_delta"] = max_delta
        if max_delta > 1.0e-4:
            moving.append(name)

    payload = {
        "schema": "ghoststudio.enclave.kotor-player-animation-sidecar.v4",
        "clip": args.clip,
        "source": str(args.source),
        "armature": armature.name,
        "action": action.name,
        "fps": fps,
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "bones": bones,
        "moving_bones": moving,
        "tracks": tracks,
        "translation_policy": "preserve_unreal_reference_pose",
        "rotation_policy": "blender_rest_relative_local_delta",
        "all_checks_passed": len(moving) >= 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("MCPSTUDIO_RESULT=" + json.dumps({
        "output": str(args.output),
        "clip": args.clip,
        "frames": payload["frame_count"],
        "bones": len(bones),
        "moving_bones": len(moving),
        "all_checks_passed": payload["all_checks_passed"],
    }, sort_keys=True))
    if not payload["all_checks_passed"]:
        raise RuntimeError("KOTOR sidecar contains too few moving bones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
