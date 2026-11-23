"""Blender-side UE5 FBX animation extraction for Sprint 3 R3.A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--action", default="")
    parser.add_argument("--frame-step", type=int, default=1)
    return parser.parse_args(argv)


def quat_wxyz(quat):
    return [float(quat.w), float(quat.x), float(quat.y), float(quat.z)]


def matrix_rows(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def capture_rest_pose_bases(armature):
    """Capture source-bone rest bases before animated frame sampling."""

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    try:
        bpy.ops.object.mode_set(mode="POSE")
        for pose_bone in armature.pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    bases = {}
    for bone in armature.data.bones:
        world = armature.matrix_world @ bone.matrix_local
        bases[bone.name] = {
            "world_matrix_at_rest": matrix_rows(world),
            "data_bone_matrix": matrix_rows(bone.matrix_local),
            "head": [float(v) for v in bone.head_local],
            "tail": [float(v) for v in bone.tail_local],
            "rotation_wxyz": quat_wxyz(world.to_quaternion()),
            "location_xyz": [float(v) for v in world.translation],
        }
    return bases


STATIC_POSE_EPSILON = 1e-6
TRANSFORM_DATA_PATHS = ("location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale")


def pick_action(armature, action_name: str):
    if action_name:
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise RuntimeError(f"Action not found: {action_name}")
        return action
    if armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    if bpy.data.actions:
        return bpy.data.actions[0]
    raise RuntimeError("No action found in FBX")


def bind_action(armature, action):
    """Assign ``action`` to the armature, including its slot on Blender 4.4+.

    Blender 4.4+ uses slotted actions: assigning ``animation_data.action`` alone
    leaves ``action_slot`` empty when no slot identifier matches the object, and
    the armature then evaluates its rest pose on every frame. FBX takes import
    with per-take slot names, so the slot must be bound explicitly.
    """

    animation_data = armature.animation_data_create()
    animation_data.action = action
    if not hasattr(animation_data, "action_slot"):
        return None  # Blender < 4.4: legacy actions bind directly.
    slot = animation_data.action_slot
    if slot is not None:
        return slot
    candidates = list(getattr(animation_data, "action_suitable_slots", None) or [])
    if not candidates:
        candidates = [
            candidate
            for candidate in action.slots
            if getattr(candidate, "target_id_type", "OBJECT") in {"OBJECT", "UNSPECIFIED"}
        ]
    if not candidates:
        candidates = list(action.slots)
    if not candidates:
        raise RuntimeError(f"Action '{action.name}' has no slots to bind to '{armature.name}'")
    preferred = [candidate for candidate in candidates if candidate.identifier == f"OB{armature.name}"]
    animation_data.action_slot = (preferred or candidates)[0]
    if animation_data.action_slot is None:
        raise RuntimeError(f"Could not bind a slot of action '{action.name}' to '{armature.name}'")
    return animation_data.action_slot


def action_fcurves(action, slot):
    """Return the action's F-curves for ``slot`` (slotted 4.4+/5.x) or legacy ``action.fcurves``."""

    if slot is not None and hasattr(action, "layers"):
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                channelbag = strip.channelbag(slot) if hasattr(strip, "channelbag") else None
                if channelbag is not None:
                    curves.extend(channelbag.fcurves)
        return curves
    return list(getattr(action, "fcurves", None) or [])


def action_key_motion(action, slot) -> float:
    """Largest keyed value range across the action's transform F-curves (0 = static action)."""

    spread = 0.0
    for fcurve in action_fcurves(action, slot):
        path = str(fcurve.data_path)
        if not path.endswith(TRANSFORM_DATA_PATHS):
            continue
        values = [float(point.co[1]) for point in fcurve.keyframe_points]
        if len(values) >= 2:
            spread = max(spread, max(values) - min(values))
    return spread


def sampled_pose_spread(curves) -> float:
    """Largest change of any bone world-matrix entry from the clip's first sampled frame."""

    spread = 0.0
    for samples in curves.values():
        if len(samples) < 2:
            continue
        first = samples[0]["matrix"]
        for sample in samples[1:]:
            matrix = sample["matrix"]
            for row in range(4):
                for col in range(4):
                    spread = max(spread, abs(matrix[row][col] - first[row][col]))
    return spread


def action_entries():
    entries = []
    for action in bpy.data.actions:
        entries.append(
            {
                "name": action.name,
                "frame_start": float(action.frame_range[0]),
                "frame_end": float(action.frame_range[1]),
                "frame_count": int(round(action.frame_range[1] - action.frame_range[0] + 1)),
            }
        )
    return entries


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx))
    bpy.context.view_layer.update()

    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("FBX import produced no armature")
    armature = armatures[0]
    action = pick_action(armature, args.action)
    slot = bind_action(armature, action)
    key_motion = action_key_motion(action, slot)

    frame_start = int(round(action.frame_range[0]))
    frame_end = int(round(action.frame_range[1]))
    step = max(1, int(args.frame_step))
    fps = float(bpy.context.scene.render.fps) / max(1.0, float(bpy.context.scene.render.fps_base))

    source_bones = [bone.name for bone in armature.data.bones]
    parents = {bone.name: bone.parent.name if bone.parent else None for bone in armature.data.bones}
    rest_pose_bases = capture_rest_pose_bases(armature)
    rest_world = {}
    for bone in armature.data.bones:
        world = armature.matrix_world @ bone.matrix_local
        rest_world[bone.name] = {
            "matrix": matrix_rows(world),
            "rotation_wxyz": quat_wxyz(world.to_quaternion()),
            "location_xyz": [float(v) for v in world.translation],
        }

    frames = []
    curves = {name: [] for name in source_bones}
    for frame in range(frame_start, frame_end + 1, step):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frames.append(frame)
        for pose_bone in armature.pose.bones:
            world = armature.matrix_world @ pose_bone.matrix
            loc, rot, _scale = world.decompose()
            curves[pose_bone.name].append(
                {
                    "frame": int(frame),
                    "time_seconds": float((frame - frame_start) / fps) if fps else 0.0,
                    "rotation_wxyz": quat_wxyz(rot),
                    "location_xyz": [float(loc.x), float(loc.y), float(loc.z)],
                    "matrix": matrix_rows(world),
                }
            )

    pose_spread = sampled_pose_spread(curves)
    if len(frames) >= 2 and key_motion > STATIC_POSE_EPSILON and pose_spread <= STATIC_POSE_EPSILON:
        raise RuntimeError(
            f"Static sample: action '{action.name}' has keyed motion ({key_motion:.6g}) but all "
            f"{len(frames)} sampled poses are identical (Blender {bpy.app.version_string}, slot "
            f"{getattr(slot, 'identifier', None)!r}). The action is not driving the armature."
        )

    meshes = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]
    payload = {
        "success": True,
        "source_fbx": str(args.fbx),
        "blender_version": bpy.app.version_string,
        "armature_name": armature.name,
        "action_name": action.name,
        "action_slot": getattr(slot, "identifier", None),
        "action_key_motion": key_motion,
        "sampled_pose_spread": pose_spread,
        "actions": action_entries(),
        "source_bone_count": len(source_bones),
        "source_bones": source_bones,
        "bone_parents": parents,
        "rest_world": rest_world,
        "rest_pose_bases": rest_pose_bases,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": max(0, len(frames)),
        "frames": frames,
        "fps": fps,
        "mesh_count": len(meshes),
        "meshes": meshes,
        "curves": curves,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[UE5_ANIM_EXTRACT] {args.json}")


if __name__ == "__main__":
    main()
