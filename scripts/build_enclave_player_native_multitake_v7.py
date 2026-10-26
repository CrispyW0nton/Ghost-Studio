"""Package the high-fidelity player with native KOTOR animation takes.

The prior Unreal sidecar path rebuilt rotations in a second coordinate basis
and produced invalid poses.  This exporter retains the exact KOTOR skeleton,
copies the native FBX actions without recomputing transforms, and emits one
multi-take FBX suitable for a mesh+animation Unreal import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--pause", type=Path, required=True)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--talk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def _matrix_error(left, right) -> float:
    return max(
        abs(float(left[row][column] - right[row][column]))
        for row in range(4)
        for column in range(4)
    )


def _import_action(path: Path, target, name: str):
    before = set(bpy.context.scene.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    imported = set(bpy.context.scene.objects) - before
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"{path} did not contain one action armature")
    source = armatures[0]
    source_action = source.animation_data.action if source.animation_data else None
    if source_action is None:
        raise RuntimeError(f"{path} has no active action")

    target_names = set(target.data.bones.keys())
    source_names = set(source.data.bones.keys())
    target_only = sorted(target_names - source_names)
    source_only = sorted(source_names - target_names)
    if target_only:
        raise RuntimeError(
            f"{path} is missing target bones: {target_only}; source-only={source_only}"
        )
    errors = {
        bone_name: _matrix_error(
            target.data.bones[bone_name].matrix_local,
            source.data.bones[bone_name].matrix_local,
        )
        for bone_name in sorted(target_names)
    }
    errors = {key: value for key, value in errors.items() if value > 1.0e-5}
    if errors:
        raise RuntimeError(f"{path} rest matrices differ from the bound player: {errors}")

    action = source_action.copy()
    action.name = name
    action.use_fake_user = True
    frame_start = int(round(action.frame_range[0]))
    frame_end = int(round(action.frame_range[1]))
    curves = len(action.fcurves)
    keyed_curves = sum(1 for curve in action.fcurves if len(curve.keyframe_points) >= 2)
    if frame_end <= frame_start or keyed_curves < 8:
        raise RuntimeError(f"{path} action is not populated")

    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    # The FBX importer creates helper-node actions in addition to the armature
    # action.  If they remain, Blender's all-actions FBX export emits dozens of
    # bogus takes.  Retain only the explicit armature-action copy above.
    for imported_action in list(set(bpy.data.actions) - before_actions):
        if imported_action != action:
            bpy.data.actions.remove(imported_action)
    return action, {
        "name": name,
        "source": str(path),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "curves": curves,
        "keyed_curves": keyed_curves,
        "rest_matrices_exact": True,
        "ignored_source_only_bones": source_only,
    }


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=str(args.player),
        use_anim=False,
        automatic_bone_orientation=False,
    )
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("Bound player must contain exactly one armature and mesh")
    target = armatures[0]

    takes = []
    actions = []
    for path, name in (
        (args.pause, "A_KotorPlayer_pause1"),
        (args.walk, "A_KotorPlayer_walk"),
        (args.run, "A_KotorPlayer_run"),
        (args.talk, "A_KotorPlayer_talk"),
    ):
        action, record = _import_action(path, target, name)
        actions.append(action)
        takes.append(record)

    target.animation_data_create()
    target.animation_data.action = actions[0]

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    meshes[0].select_set(True)
    bpy.context.view_layer.objects.active = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
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
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0,
        path_mode="COPY",
        embed_textures=True,
    )

    payload = {
        "schema": "ghoststudio.enclave.player-native-multitake.v8",
        "player": str(args.player),
        "output": str(args.output),
        "bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "skeleton_bones": len(target.data.bones),
        "mesh_vertices": len(meshes[0].data.vertices),
        "uv_layers": len(meshes[0].data.uv_layers),
        "takes": takes,
        "transform_policy": "native_kotor_fbx_actions_no_sidecar_reconstruction",
        "all_checks_passed": (
            args.output.is_file()
            and args.output.stat().st_size >= 1_000_000
            and len(target.data.bones) == 45
            and len(takes) == 4
            and all(item["keyed_curves"] >= 8 for item in takes)
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
