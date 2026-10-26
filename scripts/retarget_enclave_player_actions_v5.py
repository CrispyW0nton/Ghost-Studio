"""Retarget normalized K1 actions onto the high-fidelity player's own rig.

The player keeps its original, correctly skinned armature.  KOTOR motion is
mapped in armature space and then converted back to rest-relative local bone
deltas for deterministic Unreal AnimationDataController authoring.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion


ALIGN_BONES = (
    "pelvis_g",
    "lthigh_g",
    "lshin_g",
    "lfoot_g",
    "rthigh_g",
    "rshin_g",
    "rfoot_g",
    "torso_g",
    "torsoUpr_g",
    "lcollar_g",
    "lbicep_g",
    "LbicepL_g",
    "Lforearm_g",
    "Lhand_g",
    "rcollar_g",
    "rbicep_g",
    "RbicepL_g",
    "Rforearm_g",
    "Rhand_g",
)


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="clip=path.fbx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(values)


def _continuous(previous: list[float] | None, value: Quaternion) -> list[float]:
    value.normalize()
    current = [float(value.x), float(value.y), float(value.z), float(value.w)]
    if previous is not None and sum(a * b for a, b in zip(previous, current)) < 0.0:
        current = [-item for item in current]
    return current


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


def _local_rest(bone) -> Matrix:
    return (
        bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
        if bone.parent
        else bone.matrix_local.copy()
    )


def _parse_sources(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"Invalid --source value: {value}")
        clip, path = value.split("=", 1)
        result.append((clip.strip(), Path(path.strip())))
    return result


def _retarget(player_path: Path, clip: str, source_path: Path, output: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(player_path), use_anim=False)
    target_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(target_armatures) != 1:
        raise RuntimeError(f"Player must contain one armature, found {len(target_armatures)}")
    target = target_armatures[0]
    target.animation_data_clear()

    bpy.ops.import_scene.fbx(filepath=str(source_path), automatic_bone_orientation=False)
    source_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj != target]
    if len(source_armatures) != 1:
        raise RuntimeError(f"K1 action must contain one armature, found {len(source_armatures)}")
    source = source_armatures[0]
    action = source.animation_data.action if source.animation_data else None
    if action is None:
        raise RuntimeError(f"K1 action has no active animation: {source_path}")

    missing = [
        name for name in ALIGN_BONES
        if name not in source.data.bones or name not in target.data.bones
    ]
    if missing:
        raise RuntimeError(f"Retarget alignment bones are missing: {missing}")
    basis, basis_scale, basis_rms = _rotation_mapping(source, target)
    frame_start = int(round(action.frame_range[0]))
    frame_end = int(round(action.frame_range[1]))
    fps = float(bpy.context.scene.render.fps) / max(
        float(bpy.context.scene.render.fps_base), 1.0
    )
    target_bones = list(target.data.bones)
    shared = {bone.name for bone in target_bones if bone.name in source.data.bones}
    rest_local = {bone.name: _local_rest(bone) for bone in target_bones}
    source_rest_local = {
        name: _local_rest(source.data.bones[name]) for name in shared
    }
    local_basis = {}
    for name in shared:
        source_global_rotation = source.data.bones[name].matrix_local.to_quaternion().to_matrix().to_4x4()
        target_global_rotation = target.data.bones[name].matrix_local.to_quaternion().to_matrix().to_4x4()
        local_basis[name] = (
            target_global_rotation.inverted_safe()
            @ basis
            @ source_global_rotation
        )
    tracks = {bone.name: {"rotations_xyzw": []} for bone in target_bones}
    previous = {bone.name: None for bone in target_bones}

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for bone in target_bones:
            if bone.name in shared:
                source_pose = source.pose.bones[bone.name]
                source_pose_local = (
                    source_pose.parent.matrix.inverted_safe() @ source_pose.matrix
                    if source_pose.parent
                    else source_pose.matrix.copy()
                )
                source_delta_local = (
                    source_rest_local[bone.name].inverted_safe() @ source_pose_local
                )
                conversion = local_basis[bone.name]
                delta_local = (
                    conversion @ source_delta_local @ conversion.inverted_safe()
                )
            else:
                delta_local = Matrix.Identity(4)
            rotation = _continuous(previous[bone.name], delta_local.to_quaternion())
            previous[bone.name] = rotation
            tracks[bone.name]["rotations_xyzw"].append(rotation)

    moving = []
    for name, track in tracks.items():
        rotations = track["rotations_xyzw"]
        anchor = rotations[0]
        maximum = max(
            math.sqrt(sum((left - right) ** 2 for left, right in zip(anchor, sample)))
            for sample in rotations
        )
        track["max_quaternion_delta"] = maximum
        if maximum > 1.0e-4:
            moving.append(name)

    payload = {
        "schema": "ghoststudio.enclave.kotor-player-retarget-sidecar.v5",
        "clip": clip,
        "player": str(player_path),
        "source": str(source_path),
        "source_action": action.name,
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_end - frame_start + 1,
        "bones": [bone.name for bone in target_bones],
        "shared_source_bones": sorted(shared),
        "moving_bones": moving,
        "tracks": tracks,
        "basis_rotation_rows": [[float(basis[row][column]) for column in range(3)] for row in range(3)],
        "basis_similarity_scale": basis_scale,
        "basis_fit_rms_target_units": basis_rms,
        "translation_policy": "preserve_high_fidelity_player_reference_pose",
        "rotation_policy": "source_armature_global_delta_conjugated_into_player_basis",
        # The high-fidelity rig intentionally keeps its own finger/twist detail
        # bones; the 25 shared K1 body-chain bones are the authoritative motion
        # bridge.  Requiring source finger names would reject a correct target.
        "all_checks_passed": len(shared) >= 24 and len(moving) >= 8 and basis_rms < 0.2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if not payload["all_checks_passed"]:
        raise RuntimeError(f"Player action retarget failed validation: {payload}")
    return {
        "clip": clip,
        "output": str(output),
        "frames": payload["frame_count"],
        "bones": len(payload["bones"]),
        "shared_bones": len(shared),
        "moving_bones": len(moving),
        "basis_fit_rms_target_units": basis_rms,
        "all_checks_passed": True,
    }


def main() -> int:
    args = _args()
    records = []
    for clip, source in _parse_sources(args.source):
        if not source.is_file():
            raise RuntimeError(f"Missing K1 source action: {source}")
        records.append(_retarget(args.player, clip, source, args.output_dir / f"{clip}.json"))
    payload = {
        "schema": "ghoststudio.enclave.kotor-player-retarget.v5",
        "player": str(args.player),
        "records": records,
        "all_checks_passed": bool(records) and all(record["all_checks_passed"] for record in records),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
