"""Compare full and Unreal-compatible duel FBX head-track contracts."""

import argparse
import json
from pathlib import Path

import bpy


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def vec(value):
    return [round(float(item), 6) for item in value]


def inspect(path: Path):
    reset()
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"{path}: expected one armature, got {len(armatures)}")
    armature = armatures[0]
    actions = list(bpy.data.actions)
    start = int(bpy.context.scene.frame_start)
    end = int(bpy.context.scene.frame_end)
    sample_frames = sorted({start, (start + end) // 2, end})
    head_bones = [
        bone.name
        for bone in armature.data.bones
        if bone.name.casefold().startswith("bas_head_")
        or "headhook" in bone.name.casefold()
    ]
    samples = {}
    armature.data.pose_position = "POSE"
    for frame in sample_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples[str(frame)] = {
            name: {
                "location": vec(armature.pose.bones[name].matrix.translation),
                "scale": vec(armature.pose.bones[name].matrix.to_scale()),
            }
            for name in head_bones
            if name in armature.pose.bones
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "armature": armature.name,
        "bone_count": len(armature.data.bones),
        "head_bones": head_bones,
        "actions": [action.name for action in actions],
        "frame_range": [start, end],
        "samples": samples,
        "objects": [
            {"name": obj.name, "type": obj.type, "parent": obj.parent.name if obj.parent else None}
            for obj in bpy.context.scene.objects
        ],
    }


options = args()
payload = {
    "schema": "ghoststudio.enclave.duel-fbx-head-track-audit.v1",
    "files": [inspect(path) for path in options.fbx],
}
options.out.parent.mkdir(parents=True, exist_ok=True)
options.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
