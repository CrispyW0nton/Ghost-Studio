"""Build the Enclave player multi-take FBX with a genuine KOTOR run.

The validated v7 player mesh/skeleton remains authoritative.  Its accepted
pause, walk, and talk takes are preserved byte-semantically as Blender actions.
The native KOTOR run is sampled from the normalized Odyssey skeleton and its
rest-relative local deltas are conjugated into the player's bone bases.  This
avoids both the invalid ASCII-FBX interchange and the earlier Unreal-side
quaternion reconstruction path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


ALIGN_BONES = (
    "pelvis_g", "lthigh_g", "lshin_g", "lfoot_g",
    "rthigh_g", "rshin_g", "rfoot_g", "torso_g", "torsoUpr_g",
    "lcollar_g", "lbicep_g", "LbicepL_g", "Lforearm_g", "Lhand_g",
    "rcollar_g", "rbicep_g", "RbicepL_g", "Rforearm_g", "Rhand_g",
)


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-v7", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def _local_rest(bone) -> Matrix:
    return (
        bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
        if bone.parent else bone.matrix_local.copy()
    )


def _rotation_mapping(source, target) -> tuple[Matrix, float, float]:
    source_points = np.array(
        [[float(v) for v in source.data.bones[name].head_local] for name in ALIGN_BONES],
        dtype=np.float64,
    )
    target_points = np.array(
        [[float(v) for v in target.data.bones[name].head_local] for name in ALIGN_BONES],
        dtype=np.float64,
    )
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_zero = source_points - source_center
    target_zero = target_points - target_center
    covariance = source_zero.T @ target_zero
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    scale = float(singular.sum() / max(np.square(source_zero).sum(), 1.0e-12))
    transformed = (scale * (rotation @ source_zero.T)).T + target_center
    rms = float(np.sqrt(np.mean(np.sum(np.square(transformed - target_points), axis=1))))
    matrix = Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            matrix[row][column] = float(rotation[row, column])
    return matrix, scale, rms


def _preserved_actions(actions: list[bpy.types.Action]) -> dict[str, bpy.types.Action]:
    by_length: dict[int, list[bpy.types.Action]] = {}
    for action in actions:
        length = int(round(action.frame_range[1] - action.frame_range[0] + 1))
        by_length.setdefault(length, []).append(action)
    if len(by_length.get(201, [])) != 1 or len(by_length.get(16, [])) != 1:
        raise RuntimeError(f"v7 take contract changed: {sorted((k, len(v)) for k, v in by_length.items())}")
    if len(by_length.get(33, [])) != 2:
        raise RuntimeError("v7 must contain its accepted walk and duplicated run takes")
    pause = by_length[201][0]
    talk = by_length[16][0]
    walk = sorted(by_length[33], key=lambda item: item.name)[0]
    pause.name = "A_KotorPlayer_pause1"
    walk.name = "A_KotorPlayer_walk"
    talk.name = "A_KotorPlayer_talk"
    for action in (pause, walk, talk):
        action.use_fake_user = True
    return {"pause1": pause, "walk": walk, "talk": talk}


def _key_retargeted_run(target, source, source_action) -> tuple[bpy.types.Action, dict]:
    missing = [
        name for name in ALIGN_BONES
        if name not in source.data.bones or name not in target.data.bones
    ]
    if missing:
        raise RuntimeError(f"run retarget alignment bones are missing: {missing}")

    basis, basis_scale, basis_rms = _rotation_mapping(source, target)
    shared = sorted(set(target.data.bones.keys()) & set(source.data.bones.keys()))
    source_rest = {name: _local_rest(source.data.bones[name]) for name in shared}
    conversions = {}
    for name in shared:
        source_global = source.data.bones[name].matrix_local.to_quaternion().to_matrix().to_4x4()
        target_global = target.data.bones[name].matrix_local.to_quaternion().to_matrix().to_4x4()
        conversions[name] = target_global.inverted_safe() @ basis @ source_global

    run = bpy.data.actions.new("A_KotorPlayer_run")
    run.use_fake_user = True
    target.animation_data_create()
    target.animation_data.action = run
    source.animation_data.action = source_action
    start = int(round(source_action.frame_range[0]))
    end = int(round(source_action.frame_range[1]))
    previous: dict[str, Quaternion | None] = {name: None for name in target.pose.bones.keys()}

    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for pose_bone in target.pose.bones:
            name = pose_bone.name
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.location = Vector((0.0, 0.0, 0.0))
            pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            pose_bone.scale = Vector((1.0, 1.0, 1.0))
            if name in shared:
                source_pose = source.pose.bones[name]
                source_local = (
                    source_pose.parent.matrix.inverted_safe() @ source_pose.matrix
                    if source_pose.parent else source_pose.matrix.copy()
                )
                source_delta = source_rest[name].inverted_safe() @ source_local
                conversion = conversions[name]
                target_delta = conversion @ source_delta @ conversion.inverted_safe()
                location, rotation, scale = target_delta.decompose()
                location *= basis_scale
                prior = previous[name]
                if prior is not None and prior.dot(rotation) < 0.0:
                    rotation.negate()
                rotation.normalize()
                previous[name] = rotation.copy()
                pose_bone.location = location
                pose_bone.rotation_quaternion = rotation
                pose_bone.scale = scale
            pose_bone.keyframe_insert("location", frame=frame, group=name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=name)
            pose_bone.keyframe_insert("scale", frame=frame, group=name)

    moving_bones = []
    for pose_bone in target.pose.bones:
        curves = [curve for curve in run.fcurves if f'pose.bones["{pose_bone.name}"]' in curve.data_path]
        maximum = max(
            (
                max(point.co.y for point in curve.keyframe_points)
                - min(point.co.y for point in curve.keyframe_points)
            )
            for curve in curves
        ) if curves else 0.0
        if maximum > 1.0e-4:
            moving_bones.append(pose_bone.name)

    upper = {"lbicep_g", "rbicep_g", "Lforearm_g", "Rforearm_g", "Lhand_g", "Rhand_g"}
    if len(moving_bones) < 15 or not upper.issubset(moving_bones) or basis_rms >= 0.2:
        raise RuntimeError(
            f"retargeted run failed motion gate: moving={moving_bones}, basis_rms={basis_rms}"
        )
    return run, {
        "source": str(source_action.name),
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "shared_bones": len(shared),
        "moving_bones": moving_bones,
        "upper_body_bones_animated": sorted(upper),
        "basis_similarity_scale": basis_scale,
        "basis_fit_rms_target_units": basis_rms,
        "rotation_policy": "source_rest_delta_conjugated_into_validated_player_bone_basis",
        "translation_policy": "source_local_delta_rotated_and_proportion_scaled",
    }


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(
        filepath=str(args.player_v7),
        use_anim=True,
        automatic_bone_orientation=False,
    )
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("v7 player must contain exactly one armature and one mesh")
    target = armatures[0]
    imported_v7_actions = list(set(bpy.data.actions) - before_actions)
    preserved = _preserved_actions(imported_v7_actions)
    keep = set(preserved.values())
    for action in imported_v7_actions:
        if action not in keep:
            bpy.data.actions.remove(action)

    before_objects = set(bpy.context.scene.objects)
    before_source_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(
        filepath=str(args.native_run),
        automatic_bone_orientation=False,
    )
    imported_objects = set(bpy.context.scene.objects) - before_objects
    sources = [obj for obj in imported_objects if obj.type == "ARMATURE"]
    if len(sources) != 1:
        raise RuntimeError(f"native run must contain one source armature, found {len(sources)}")
    source = sources[0]
    source_action = source.animation_data.action if source.animation_data else None
    if source_action is None:
        raise RuntimeError("native run has no active action")
    run, run_record = _key_retargeted_run(target, source, source_action)

    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(set(bpy.data.actions) - before_source_actions):
        if action != run:
            bpy.data.actions.remove(action)

    final_actions = [preserved["pause1"], preserved["walk"], run, preserved["talk"]]
    target.animation_data_create()
    target.animation_data.action = preserved["pause1"]
    for action in final_actions:
        action.use_fake_user = True

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
        "player_v7": str(args.player_v7),
        "native_run": str(args.native_run),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size if args.output.is_file() else 0,
        "skeleton_bones": len(target.data.bones),
        "mesh_vertices": len(meshes[0].data.vertices),
        "preserved_takes": {
            name: [float(action.frame_range[0]), float(action.frame_range[1])]
            for name, action in preserved.items()
        },
        "retargeted_run": run_record,
        "all_checks_passed": bool(
            args.output.is_file()
            and args.output.stat().st_size >= 1_000_000
            and len(target.data.bones) == 45
            and len(final_actions) == 4
            and len(run_record["moving_bones"]) >= 15
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
