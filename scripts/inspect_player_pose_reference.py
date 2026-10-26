"""Inspect source player and KOTOR reference skeleton rest directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def _reset():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _inspect(path: Path) -> dict:
    _reset()
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature in {path}, got {len(armatures)}")
    armature = armatures[0]
    return {
        "path": str(path),
        "armature": armature.name,
        "dimensions": [round(float(v), 6) for v in armature.dimensions],
        "armature_location": [round(float(v), 6) for v in armature.location],
        "armature_rotation": [round(float(v), 6) for v in armature.rotation_euler],
        "armature_scale": [round(float(v), 6) for v in armature.scale],
        "mesh_dimensions": {obj.name: [round(float(v), 6) for v in obj.dimensions] for obj in meshes},
        "mesh_transforms": {
            obj.name: {
                "location": [round(float(v), 6) for v in obj.location],
                "rotation": [round(float(v), 6) for v in obj.rotation_euler],
                "scale": [round(float(v), 6) for v in obj.scale],
                "parent": obj.parent.name if obj.parent else "",
            }
            for obj in meshes
        },
        "bones": [
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else "",
                "head": [round(float(v), 6) for v in bone.head_local],
                "tail": [round(float(v), 6) for v in bone.tail_local],
            }
            for bone in armature.data.bones
        ],
    }


def main() -> int:
    args = _args()
    payload = {"player": _inspect(args.player), "reference": _inspect(args.reference)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "player_bones": len(payload["player"]["bones"]), "reference_bones": len(payload["reference"]["bones"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
