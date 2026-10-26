"""Read-only unit-space comparison for KOTOR duel and meditation FBXs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


BONES = (
    "PFBIM",
    "PMBIM",
    "cutscenedummy",
    "rootdummy",
    "pelvis_g",
    "torso_g",
    "torsoUpr_g",
    "rhand_g",
    "Rhand_g",
    "lhand_g",
    "Lhand_g",
    "rfoot_g",
    "lfoot_g",
    "headhook",
    "Hturn_g",
    "head_g",
    "bas_head_head",
)


def _args() -> argparse.Namespace:
    values = list(__import__("sys").argv)
    values = values[values.index("--") + 1 :] if "--" in values else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def _vec(value) -> list[float]:
    return [float(component) for component in value]


def _inspect(path: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=str(path),
        use_anim=True,
        automatic_bone_orientation=False,
    )
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"{path}: expected one armature, got {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action:
        start = int(action.frame_range[0])
        end = int(action.frame_range[1])
        frames = sorted({start, (start + end) // 2, end})
    else:
        frames = [0]
    samples = {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        sample = {}
        for name in BONES:
            pose = armature.pose.bones.get(name)
            rest = armature.data.bones.get(name)
            if not pose or not rest:
                continue
            parent = pose.parent.name if pose.parent else None
            sample[name] = {
                "parent": parent,
                "rest_translation": _vec(rest.matrix_local.translation),
                "basis_location": _vec(pose.location),
                "basis_scale": _vec(pose.scale),
                "pose_translation": _vec(pose.matrix.translation),
                "pose_scale": _vec(pose.matrix.to_scale()),
            }
        samples[str(frame)] = sample
    location_curve_max = {}
    if action:
        for curve in action.fcurves:
            if not curve.data_path.endswith(".location"):
                continue
            values = [abs(float(point.co.y)) for point in curve.keyframe_points]
            location_curve_max[curve.data_path] = max(
                location_curve_max.get(curve.data_path, 0.0),
                max(values, default=0.0),
            )
    return {
        "path": str(path),
        "armature": armature.name,
        "armature_location": _vec(armature.location),
        "armature_scale": _vec(armature.scale),
        "armature_world_scale": _vec(armature.matrix_world.to_scale()),
        "action": action.name if action else None,
        "frames": frames,
        "samples": samples,
        "location_curve_max": location_curve_max,
    }


def main() -> int:
    args = _args()
    payload = {
        "schema": "ghoststudio.enclave.duel-pose-units-diagnostic.v2",
        "files": [_inspect(path) for path in args.fbx],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MCPSTUDIO_RESULT=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
