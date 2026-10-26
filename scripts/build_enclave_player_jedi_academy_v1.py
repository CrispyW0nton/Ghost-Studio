"""Pose Player.fbx to Jedi Academy and retarget its core movement library.

The Jedi Academy Kyle rig is the bind-pose and animation authority.  The
high-fidelity Player mesh remains the render mesh.  This script deliberately
uses Blender Pose Mode to turn the Player T-pose into the mapped Jedi Academy
A-pose before applying that pose as the new rest pose.  Animation is then
transferred as source-rest-relative global rotation deltas, so unlike the
older name-only retarget it does not assume that the two rigs share a bind
orientation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


PLAYER_TO_JA = {
    "Hip": "pelvis",
    "L_Thigh": "lfemurYZ",
    "L_Calf": "ltibia",
    "L_Foot": "ltalus",
    "R_Thigh": "rfemurYZ",
    "R_Calf": "rtibia",
    "R_Foot": "rtalus",
    "Waist": "lower_lumbar",
    "Spine01": "upper_lumbar",
    "Spine02": "thoracic",
    "NeckTwist01": "cervical",
    "Head": "cranium",
    "L_Clavicle": "lclavical",
    "L_Upperarm": "lhumerus",
    "L_Forearm": "lradius",
    "L_Hand": "lhand",
    "R_Clavicle": "rclavical",
    "R_Upperarm": "rhumerus",
    "R_Forearm": "rradius",
    "R_Hand": "rhand",
}

# Limbs and torso landmarks produce a stable similarity basis without relying
# on either rig's helper/end bones.
ALIGN_PLAYER_BONES = (
    "Hip", "L_Thigh", "L_Calf", "L_Foot", "R_Thigh", "R_Calf", "R_Foot",
    "Waist", "Spine01", "Spine02", "NeckTwist01", "Head",
    "L_Upperarm", "L_Forearm", "L_Hand", "R_Upperarm", "R_Forearm", "R_Hand",
)


@dataclass(frozen=True)
class Clip:
    name: str
    chunk_start: int
    chunk_end: int
    source_start: int
    source_frames: int
    source_fps: float

    @property
    def local_start(self) -> int:
        return self.source_start - self.chunk_start + 1

    @property
    def output_frames(self) -> int:
        # Include both endpoints so loops retain their closing source sample.
        duration = max((self.source_frames - 1) / self.source_fps, 0.0)
        return max(2, int(round(duration * 30.0)) + 1)


CLIPS = (
    Clip("idle", 14193, 15220, 14793, 150, 20.0),
    Clip("walk", 20312, 21312, 20438, 26, 20.0),
    Clip("run", 12170, 13179, 12305, 26, 40.0),
    Clip("walk_back", 20312, 21312, 20667, 29, 20.0),
    Clip("run_back", 12170, 13179, 12367, 10, 20.0),
    Clip("strafe_left", 15221, 16229, 16131, 13, 10.0),
    Clip("strafe_right", 15221, 16229, 16144, 13, 10.0),
    Clip("turn", 18242, 19308, 18666, 12, 20.0),
    Clip("jump", 7087, 8126, 7951, 8, 20.0),
    Clip("land", 8127, 9130, 9048, 5, 20.0),
)


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--animation-dir", type=Path, required=True)
    parser.add_argument("--output-fbx", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def import_fbx(path: Path, *, use_anim: bool) -> tuple[list[bpy.types.Object], list[bpy.types.Action]]:
    before_objects = set(bpy.context.scene.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(
        filepath=str(path),
        use_anim=use_anim,
        automatic_bone_orientation=False,
    )
    return (
        list(set(bpy.context.scene.objects) - before_objects),
        list(set(bpy.data.actions) - before_actions),
    )


def single_armature(objects: list[bpy.types.Object], label: str) -> bpy.types.Object:
    values = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(values) != 1:
        raise RuntimeError(f"{label} must contain one armature, found {len(values)}")
    return values[0]


def world_bone_head(armature: bpy.types.Object, name: str) -> np.ndarray:
    point = armature.matrix_world @ armature.data.bones[name].head_local
    return np.asarray(tuple(point), dtype=np.float64)


def similarity_source_to_target(
    source: bpy.types.Object,
    target: bpy.types.Object,
) -> tuple[Matrix, float, float]:
    source_points = np.asarray(
        [world_bone_head(source, PLAYER_TO_JA[name]) for name in ALIGN_PLAYER_BONES],
        dtype=np.float64,
    )
    target_points = np.asarray(
        [world_bone_head(target, name) for name in ALIGN_PLAYER_BONES],
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


def pose_target_to_reference(
    target: bpy.types.Object,
    source: bpy.types.Object,
    source_to_target: Matrix,
) -> dict:
    """Use Blender Pose Mode to turn the Player T-pose into JA's A-pose."""
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    del source_to_target
    before = {}
    reference_drop = {}
    # Player is already upright and proportionally correct.  The actual bind
    # mismatch that blocks locomotion is its horizontal T-pose arms.  Match the
    # Jedi Academy reference's per-side A-pose drop while preserving Player's
    # authored forward/up basis, torso, leg stance, bone lengths, and rolls.
    for target_name, source_name in (
        ("L_Upperarm", "lhumerus"),
        ("R_Upperarm", "rhumerus"),
    ):
        target_bone = target.pose.bones[target_name]
        current = (target_bone.tail - target_bone.head).normalized()
        before[target_name] = current.copy()
        source_vector = source.data.bones[source_name].vector.normalized()
        source_horizontal = math.sqrt(source_vector.x ** 2 + source_vector.y ** 2)
        drop = abs(math.atan2(source_vector.z, max(source_horizontal, 1.0e-8)))
        # Jedi Academy's reference is a relaxed A-pose, not an arm-at-sides
        # stance.  Clamp only against malformed reference data.
        drop = min(max(drop, math.radians(25.0)), math.radians(60.0))
        reference_drop[target_name] = math.degrees(drop)
        horizontal = Vector((current.x, current.y, 0.0)).normalized()
        desired = horizontal * math.cos(drop) + Vector((0.0, 0.0, -math.sin(drop)))
        correction = current.rotation_difference(desired.normalized())
        rest_matrix = target.data.bones[target_name].matrix_local.copy()
        corrected_rotation = correction.to_matrix() @ rest_matrix.to_3x3()
        corrected = corrected_rotation.to_4x4()
        corrected.translation = rest_matrix.translation
        target_bone.matrix_basis = rest_matrix.inverted_safe() @ corrected
        bpy.context.view_layer.update()
    # This is the explicit Pose Mode operation requested by the user.  It also
    # updates the skinned mesh bind data to the accepted A-pose.
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()

    achieved_drop = {}
    for target_name in ("L_Upperarm", "R_Upperarm"):
        direction = target.data.bones[target_name].vector.normalized()
        horizontal = math.sqrt(direction.x ** 2 + direction.y ** 2)
        achieved_drop[target_name] = math.degrees(
            abs(math.atan2(direction.z, max(horizontal, 1.0e-8)))
        )
    errors = {
        name: abs(achieved_drop[name] - reference_drop[name])
        for name in reference_drop
    }
    return {
        "operation": "Blender Pose Mode -> Apply Pose as Rest Pose",
        "posed_bones": ["L_Upperarm", "R_Upperarm"],
        "reference_arm_drop_degrees": reference_drop,
        "achieved_arm_drop_degrees": achieved_drop,
        "maximum_direction_error_degrees": max(errors.values()),
        "direction_error_degrees": errors,
        "upperarm_drop_degrees": {
            name: math.degrees(before[name].angle(target.data.bones[name].vector.normalized()))
            for name in ("L_Upperarm", "R_Upperarm")
        },
    }


def chunk_path(animation_dir: Path, clip: Clip) -> Path:
    value = animation_dir / f"_humanoid_frames_{clip.chunk_start:05d}_{clip.chunk_end:05d}.fbx"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def skeleton_action(actions: list[bpy.types.Action]) -> bpy.types.Action:
    candidates = []
    for action in actions:
        try:
            curve_count = len(action.fcurves)
        except AttributeError:
            curve_count = 0
        if curve_count > 100:
            candidates.append(action)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one JA skeleton action, found {[a.name for a in candidates]}")
    return candidates[0]


def target_depth(target: bpy.types.Object) -> dict[str, int]:
    return {
        bone.name: len(list(bone.parent_recursive))
        for bone in target.data.bones
    }


def retarget_clip(
    target: bpy.types.Object,
    source: bpy.types.Object,
    source_action: bpy.types.Action,
    clip: Clip,
    source_to_target: Matrix,
    similarity_scale: float,
) -> tuple[bpy.types.Action, dict]:
    source.animation_data_create()
    source.animation_data.action = source_action
    target.animation_data_create()
    action = bpy.data.actions.new(f"A_JAPlayer_{clip.name}")
    action.use_fake_user = True
    target.animation_data.action = action
    depth = target_depth(target)
    ordered = sorted(PLAYER_TO_JA, key=lambda name: depth[name])
    source_rest_rotation = {
        source_name: source.data.bones[source_name].matrix_local.to_quaternion()
        for source_name in PLAYER_TO_JA.values()
    }
    target_rest_rotation = {
        target_name: target.data.bones[target_name].matrix_local.to_quaternion()
        for target_name in PLAYER_TO_JA
    }
    source_rest_pelvis = source.data.bones["pelvis"].head_local.copy()
    basis_q = source_to_target.to_quaternion()
    prior: dict[str, Quaternion | None] = {
        pose_bone.name: None for pose_bone in target.pose.bones
    }
    scene = bpy.context.scene
    for output_frame in range(1, clip.output_frames + 1):
        seconds = (output_frame - 1) / 30.0
        source_frame = clip.local_start + seconds * clip.source_fps
        source_frame = min(source_frame, clip.local_start + clip.source_frames - 1)
        base = int(math.floor(source_frame))
        scene.frame_set(base, subframe=source_frame - base)
        bpy.context.view_layer.update()

        # Reset the destination to its new JA-aligned rest pose.
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.location = Vector((0.0, 0.0, 0.0))
            pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            pose_bone.scale = Vector((1.0, 1.0, 1.0))
        bpy.context.view_layer.update()

        for target_name in ordered:
            source_name = PLAYER_TO_JA[target_name]
            source_pose_q = source.pose.bones[source_name].matrix.to_quaternion()
            source_delta_q = source_pose_q @ source_rest_rotation[source_name].inverted()
            mapped_delta_q = basis_q @ source_delta_q @ basis_q.inverted()
            desired_global_q = mapped_delta_q @ target_rest_rotation[target_name]
            desired_global_q.normalize()
            previous = prior[target_name]
            if previous is not None and previous.dot(desired_global_q) < 0.0:
                desired_global_q.negate()
            prior[target_name] = desired_global_q.copy()
            destination = target.pose.bones[target_name]
            matrix = desired_global_q.to_matrix().to_4x4()
            matrix.translation = destination.matrix.translation
            if target_name == "Hip":
                source_delta = source.pose.bones["pelvis"].matrix.translation - source_rest_pelvis
                mapped_delta = source_to_target.to_3x3() @ source_delta
                # Gameplay clips are in-place.  Preserve only vertical pelvis
                # response (including jump/land) and discard authored travel.
                matrix.translation.z = (
                    target.data.bones["Hip"].head_local.z
                    + mapped_delta.z * similarity_scale
                )
            destination.matrix = matrix
            bpy.context.view_layer.update()

        for pose_bone in target.pose.bones:
            pose_bone.keyframe_insert("location", frame=output_frame, group=pose_bone.name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=output_frame, group=pose_bone.name)
            pose_bone.keyframe_insert("scale", frame=output_frame, group=pose_bone.name)

    moving = []
    amplitudes = {}
    for name in target.pose.bones.keys():
        curves = [curve for curve in action.fcurves if f'pose.bones["{name}"]' in curve.data_path]
        amplitude = max(
            (
                max(float(point.co.y) for point in curve.keyframe_points)
                - min(float(point.co.y) for point in curve.keyframe_points)
            )
            for curve in curves
        ) if curves else 0.0
        amplitudes[name] = amplitude
        if amplitude > 1.0e-4:
            moving.append(name)
    upper = ("L_Upperarm", "L_Forearm", "L_Hand", "R_Upperarm", "R_Forearm", "R_Hand")
    return action, {
        "name": clip.name,
        "source_global_start": clip.source_start,
        "source_frames": clip.source_frames,
        "source_fps": clip.source_fps,
        "output_frames": clip.output_frames,
        "output_fps": 30,
        "in_place": True,
        "moving_bones": moving,
        "upper_body_amplitudes": {name: amplitudes[name] for name in upper},
        "upper_body_animated": sum(amplitudes[name] > 1.0e-4 for name in upper),
    }


def remove_imported(objects: list[bpy.types.Object], actions: list[bpy.types.Action]) -> None:
    for obj in objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def mesh_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [mesh.matrix_world @ Vector(corner) for mesh in meshes for corner in mesh.bound_box]
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return low, high


def main() -> int:
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    player_objects, player_actions = import_fbx(args.player, use_anim=False)
    target = single_armature(player_objects, "Player")
    meshes = [obj for obj in player_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("Player has no skinned mesh")
    for action in player_actions:
        bpy.data.actions.remove(action)

    reference_objects, reference_actions = import_fbx(args.reference, use_anim=False)
    reference = single_armature(reference_objects, "Jedi Academy reference")
    missing = [
        f"{target_name}->{source_name}"
        for target_name, source_name in PLAYER_TO_JA.items()
        if target_name not in target.data.bones or source_name not in reference.data.bones
    ]
    if missing:
        raise RuntimeError(f"retarget map is incomplete: {missing}")
    source_to_target, similarity_scale, basis_rms = similarity_source_to_target(reference, target)
    pose_receipt = pose_target_to_reference(target, reference, source_to_target)

    remove_imported(reference_objects, reference_actions)
    actions = []
    action_receipts = []
    for clip in CLIPS:
        source_objects, source_actions = import_fbx(chunk_path(args.animation_dir, clip), use_anim=True)
        source = single_armature(source_objects, clip.name)
        action, record = retarget_clip(
            target,
            source,
            skeleton_action(source_actions),
            clip,
            source_to_target,
            similarity_scale,
        )
        actions.append(action)
        action_receipts.append(record)
        removable = [item for item in source_actions if item != action]
        remove_imported(source_objects, removable)

    target.animation_data_create()
    target.animation_data.action = next(action for action in actions if action.name.endswith("_idle"))
    for action in actions:
        action.use_fake_user = True
    bpy.context.scene.render.fps = 30

    # Keep a Blender source artifact so the Pose Mode result is directly
    # inspectable and future retarget changes do not have to start from FBX.
    args.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = target
    args.output_fbx.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(args.output_fbx),
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
    low, high = mesh_bounds(meshes)
    critical_actions = {"idle", "walk", "run", "jump", "land"}
    receipt = {
        "schema": "ghoststudio.enclave.player-jedi-academy.v1",
        "player": str(args.player),
        "jedi_academy_reference": str(args.reference),
        "output_fbx": str(args.output_fbx),
        "output_blend": str(args.output_blend),
        "output_bytes": args.output_fbx.stat().st_size if args.output_fbx.is_file() else 0,
        "source_to_target_similarity_scale": similarity_scale,
        "source_to_target_basis_rms": basis_rms,
        "pose": pose_receipt,
        "actions": action_receipts,
        "mesh_bounds": {"minimum": list(low), "maximum": list(high)},
        "all_checks_passed": bool(
            args.output_fbx.is_file()
            and args.output_fbx.stat().st_size > 1_000_000
            and args.output_blend.is_file()
            and pose_receipt["maximum_direction_error_degrees"] < 2.0
            and all(
                record["upper_body_animated"] >= 4
                for record in action_receipts
                if record["name"] in critical_actions
            )
            and len(actions) == len(CLIPS)
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
