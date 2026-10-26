"""Audit Player.fbx and Jedi Academy reference/locomotion FBXs in Blender."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--animation", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(values)


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def inspect(path: Path) -> dict:
    clear()
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bounds = []
    for mesh in meshes:
        for corner in mesh.bound_box:
            point = mesh.matrix_world @ __import__("mathutils").Vector(corner)
            bounds.append([float(point.x), float(point.y), float(point.z)])
    actions = []
    for action in bpy.data.actions:
        try:
            curve_count = len(action.fcurves)
        except AttributeError:
            curve_count = 0
        actions.append({
            "name": action.name,
            "frame_range": [float(action.frame_range[0]), float(action.frame_range[1])],
            "curve_count": curve_count,
        })
    armature_records = []
    for armature in armatures:
        roots = [bone.name for bone in armature.data.bones if bone.parent is None]
        armature_records.append({
            "name": armature.name,
            "bone_count": len(armature.data.bones),
            "roots": roots,
            "bones": [bone.name for bone in armature.data.bones],
            "parents": {
                bone.name: bone.parent.name if bone.parent else None
                for bone in armature.data.bones
            },
        })
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "objects": len(bpy.context.scene.objects),
        "meshes": [{"name": mesh.name, "vertices": len(mesh.data.vertices)} for mesh in meshes],
        "armatures": armature_records,
        "actions": actions,
        "bounds": {
            "minimum": [min(item[axis] for item in bounds) for axis in range(3)] if bounds else None,
            "maximum": [max(item[axis] for item in bounds) for axis in range(3)] if bounds else None,
        },
    }


def main() -> int:
    args = arguments()
    records = [inspect(args.player), inspect(args.reference)]
    records.extend(inspect(path) for path in args.animation)
    payload = {
        "schema": "ghoststudio.jedi-academy-player-source-audit.v1",
        "records": records,
        "all_checks_passed": all(record["armatures"] for record in records)
            and bool(records[0]["meshes"])
            and bool(records[1]["meshes"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
