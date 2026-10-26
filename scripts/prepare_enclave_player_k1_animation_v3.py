import argparse
import json
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    return parser.parse_args(bpy.app.driver_namespace.get("argv_after_double_dash", []))


def real_args():
    import sys

    values = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    return parser.parse_args(values)


args = real_args()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(args.source), automatic_bone_orientation=False)

armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
if len(armatures) != 1:
    raise RuntimeError(f"Expected one source armature, found {len(armatures)}")
armature = armatures[0]
if armature.animation_data is None or armature.animation_data.action is None:
    raise RuntimeError("Source FBX armature has no active animation")
action = armature.animation_data.action

# The canonical Unreal Skeleton was originally created from an FBX whose mesh
# node became a synthetic root above the armature object. Recreate that exact
# two-node hierarchy so the legacy FBX animation importer accepts the clip.
root = bpy.data.objects.new("SK_dan13_ahlan_BAS_skin", None)
bpy.context.scene.collection.objects.link(root)
armature.parent = root

# The earlier export included every helper object and mesh. Interchange created
# nominal tracks for them but persisted zero keys. Export exactly the canonical
# armature and its active baked action so Unreal receives real bone tracks.
for obj in bpy.context.scene.objects:
    obj.select_set(False)
root.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
start, end = action.frame_range
bpy.context.scene.frame_start = int(round(start))
bpy.context.scene.frame_end = int(round(end))
args.output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.fbx(
    filepath=str(args.output),
    check_existing=False,
    use_selection=True,
    object_types={"ARMATURE", "EMPTY"},
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_UNITS",
    axis_forward="-Z",
    axis_up="Y",
    add_leaf_bones=False,
    primary_bone_axis="Y",
    secondary_bone_axis="X",
    armature_nodetype="ROOT",
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

changing_curves = 0
for curve in action.fcurves:
    values = [round(point.co.y, 7) for point in curve.keyframe_points]
    if len(set(values)) > 1:
        changing_curves += 1
payload = {
    "schema": "mcpstudio.enclave.kotor-player-animation-source.v3",
    "clip": args.clip,
    "source": str(args.source),
    "output": str(args.output),
    "output_bytes": args.output.stat().st_size,
    "armature": armature.name,
    "bones": len(armature.data.bones),
    "action": action.name,
    "frame_range": [float(start), float(end)],
    "fcurves": len(action.fcurves),
    "changing_curves": changing_curves,
    "all_checks_passed": bool(args.output.is_file() and changing_curves > 0),
}
receipt = args.output.with_suffix(".json")
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print("MCPSTUDIO_RESULT=" + json.dumps(payload, sort_keys=True))
if not payload["all_checks_passed"]:
    raise RuntimeError("Clean KOTOR animation source contains no animated curves")
