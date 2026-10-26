import bpy
import json
import sys


args = sys.argv[sys.argv.index("--") + 1 :]
path = args[args.index("--fbx") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)

actions = []
for action in bpy.data.actions:
    curves = list(action.fcurves)
    keyed_points = sum(len(curve.keyframe_points) for curve in curves)
    changing_curves = 0
    for curve in curves:
        values = [round(point.co.y, 7) for point in curve.keyframe_points]
        if len(set(values)) > 1:
            changing_curves += 1
    actions.append(
        {
            "name": action.name,
            "frame_range": list(action.frame_range),
            "fcurves": len(curves),
            "keyed_points": keyed_points,
            "changing_curves": changing_curves,
            "sample_paths": [curve.data_path for curve in curves[:12]],
        }
    )

payload = {
    "path": path,
    "actions": actions,
    "armatures": [
        {
            "name": obj.name,
            "action": obj.animation_data.action.name
            if obj.animation_data and obj.animation_data.action
            else None,
        }
        for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE"
    ],
}
print("MCPSTUDIO_RESULT=" + json.dumps(payload, sort_keys=True))
