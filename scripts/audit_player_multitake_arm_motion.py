"""Audit per-bone motion ranges in a Blender-imported player FBX.

Run this script through Blender in background mode.  It is intentionally
read-only and reports whether locomotion takes contain meaningful upper-body
motion instead of silently leaving the character in its bind pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


AUDIT_BONES = (
    "rbicep_g",
    "lbicep_g",
    "Rforearm_g",
    "Lforearm_g",
    "rthigh_g",
    "lthigh_g",
)


def _curve_range(curve) -> float:
    values = [float(point.co.y) for point in curve.keyframe_points]
    return max(values) - min(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(script_args)

    bpy.ops.import_scene.fbx(filepath=str(args.fbx.resolve()))
    actions = []
    for action in bpy.data.actions:
        per_bone = {}
        for bone in AUDIT_BONES:
            needle = f'pose.bones["{bone}"]'
            curves = [curve for curve in action.fcurves if needle in curve.data_path]
            per_bone[bone] = {
                "curve_count": len(curves),
                "maximum_channel_range": max((_curve_range(curve) for curve in curves), default=0.0),
            }
        actions.append(
            {
                "name": action.name,
                "frame_range": [float(value) for value in action.frame_range],
                "curve_count": len(action.fcurves),
                "bones": per_bone,
            }
        )

    payload = {
        "schema": "ghoststudio.player-multitake-motion-audit.v1",
        "fbx": str(args.fbx.resolve()),
        "actions": actions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
