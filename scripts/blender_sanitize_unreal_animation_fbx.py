"""Re-export one FBX as an armature-only Unreal animation clip.

This preserves the imported armature action while excluding BAS mesh and empty
attachment objects.  In particular, an animated ``headhook`` EMPTY must not be
serialized as a bone track when targeting a corrected merged-head skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = _args()
    if not args.source.is_file():
        raise RuntimeError(f"Missing animation FBX: {args.source}")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=str(args.source), use_anim=True)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("Imported FBX has no armature action")
    start, end = action.frame_range
    bpy.context.scene.frame_start = int(round(start))
    bpy.context.scene.frame_end = int(round(end))
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="ROOT",
        use_armature_deform_only=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )
    payload = {
        "schema": "ghoststudio.unreal-animation-fbx-sanitize.v1",
        "source": str(args.source),
        "output": str(args.output),
        "armature": armature.name,
        "action": action.name,
        "bone_count": len(armature.data.bones),
        "frame_start": int(round(start)),
        "frame_end": int(round(end)),
        "excluded_object_count": len(bpy.context.scene.objects) - 1,
        "output_bytes": args.output.stat().st_size,
    }
    if payload["output_bytes"] < 1024:
        raise RuntimeError(f"Sanitized FBX is invalid: {payload}")
    print("MCP_UNREAL_ANIMATION_FBX_SANITIZE=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
