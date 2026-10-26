"""Print the imported hierarchy around the KOTOR head socket for one FBX."""

from __future__ import annotations

import json
import sys

import bpy


argv = sys.argv[sys.argv.index("--") + 1 :]
source = argv[0]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.fbx(filepath=source, use_anim=True)
armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
action = armature.animation_data.action if armature.animation_data else None
rows = []
for bone in armature.data.bones:
    name = bone.name.casefold()
    if name in {"headhook", "torsoupr_g", "torso_g", "head_g", "hturn_g", "neck_g", "necklwr_g"} or (
        bone.parent and bone.parent.name.casefold() == "headhook"
    ):
        rows.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "use_deform": bool(bone.use_deform),
                "children": [child.name for child in bone.children],
            }
        )
curve_bones = sorted(
    {
        curve.data_path.split('pose.bones["', 1)[1].split('"]', 1)[0]
        for curve in (action.fcurves if action else [])
        if 'pose.bones["' in curve.data_path
    }
)
print(
    "MCP_DUEL_FBX_HIERARCHY="
    + json.dumps(
        {
            "source": source,
            "armature": armature.name,
            "bone_count": len(armature.data.bones),
            "rows": rows,
            "action": action.name if action else None,
            "curve_bones_containing_head": [name for name in curve_bones if "head" in name.casefold()],
            "curve_bone_count": len(curve_bones),
            "head_named_objects": [
                {
                    "name": obj.name,
                    "type": obj.type,
                    "parent": obj.parent.name if obj.parent else None,
                    "animation": (
                        obj.animation_data.action.name
                        if obj.animation_data and obj.animation_data.action
                        else None
                    ),
                }
                for obj in bpy.context.scene.objects
                if "head" in obj.name.casefold()
            ],
        },
        sort_keys=True,
    )
)
