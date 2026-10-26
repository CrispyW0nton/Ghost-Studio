import bpy
import json
import os
import sys


def argv_value(flag):
    args = sys.argv[sys.argv.index("--") + 1 :]
    return args[args.index(flag) + 1]


def inspect_fbx(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    return {
        "path": path,
        "armatures": [
            {
                "object_name": obj.name,
                "data_name": obj.data.name,
                "bones": [bone.name for bone in obj.data.bones],
                "parents": {bone.name: (bone.parent.name if bone.parent else None) for bone in obj.data.bones},
            }
            for obj in arms
        ],
    }


player = inspect_fbx(argv_value("--player"))
reference = inspect_fbx(argv_value("--reference"))
player_bones = set(player["armatures"][0]["bones"])
reference_bones = set(reference["armatures"][0]["bones"])
payload = {
    "player": player,
    "reference": reference,
    "player_only": sorted(player_bones - reference_bones),
    "reference_only": sorted(reference_bones - player_bones),
}
print("MCPSTUDIO_RESULT=" + json.dumps(payload, sort_keys=True))
