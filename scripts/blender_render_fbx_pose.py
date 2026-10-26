"""Render deterministic source-FBX pose sheets for retarget visual validation.

This deliberately renders the authored FBX before any Ghost Rigger conversion so
source and Aurora output can be compared at matching normalized clip times.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fractions", default="0,0.1667,0.3333,0.5,0.6667,0.8333,1")
    return parser.parse_args(argv)


def _look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("FBX contains no renderable mesh bounds")
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def _pose_points(armature: bpy.types.Object) -> list[Vector]:
    return [
        armature.matrix_world @ point
        for bone in armature.pose.bones
        for point in (bone.head, bone.tail)
    ]


def _point_bounds(points: list[Vector]) -> tuple[Vector, Vector]:
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def _actor_axes(armature: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    bones = armature.pose.bones
    left = armature.matrix_world @ bones["upperarm_l"].head
    right = armature.matrix_world @ bones["upperarm_r"].head
    pelvis = armature.matrix_world @ bones["pelvis"].head
    head = armature.matrix_world @ bones["head"].head
    actor_right = (right - left).normalized()
    actor_up = (head - pelvis).normalized()
    actor_forward = actor_right.cross(actor_up).normalized()
    actor_up = actor_forward.cross(actor_right).normalized()
    return actor_right, actor_up, actor_forward


def _make_pose_curves(
    armature: bpy.types.Object,
    material: bpy.types.Material,
) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for bone in armature.pose.bones:
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail
        curve = bpy.data.curves.new(f"pose_{bone.name}", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = max(0.0025, (tail - head).length * 0.035)
        curve.bevel_resolution = 2
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (*head, 1.0)
        spline.points[1].co = (*tail, 1.0)
        curve.materials.append(material)
        obj = bpy.data.objects.new(f"pose_{bone.name}", curve)
        bpy.context.scene.collection.objects.link(obj)
        objects.append(obj)
    return objects


def main() -> None:
    args = _args()
    fractions = [max(0.0, min(1.0, float(value))) for value in args.fractions.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx))
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("FBX must contain an armature for visual proof")

    action = armatures[0].animation_data.action if armatures[0].animation_data else None
    if action is None and bpy.data.actions:
        action = bpy.data.actions[0]
        armatures[0].animation_data_create()
        armatures[0].animation_data.action = action
    if action is None:
        raise RuntimeError("FBX contains no animation action")
    frame_start, frame_end = (float(value) for value in action.frame_range)

    world = scene.world or bpy.data.worlds.new("SourcePoseWorld")
    scene.world = world
    world.color = (0.025, 0.025, 0.025)
    # Blender 5.0 exposes Eevee as ``BLENDER_EEVEE`` again, while 4.x uses
    # ``BLENDER_EEVEE_NEXT``.  Keep the read-only proof renderer portable
    # across both supported generations instead of failing before capture.
    engine_ids = {item.identifier for item in scene.bl_rna.properties["render"].fixed_type.properties["engine"].enum_items}
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT" in engine_ids
        else "BLENDER_EEVEE"
    )
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    light_data = bpy.data.lights.new("SourcePoseKey", "AREA")
    light_data.energy = 1200.0
    light_data.shape = "DISK"
    light_data.size = 5.0
    light = bpy.data.objects.new("SourcePoseKey", light_data)
    scene.collection.objects.link(light)

    pose_material = bpy.data.materials.new("SourcePoseBones")
    pose_material.diffuse_color = (0.1, 0.75, 1.0, 1.0)
    pose_material.use_nodes = True
    bsdf = pose_material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.35, 0.8, 1.0)
    bsdf.inputs["Emission Color"].default_value = (0.02, 0.25, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.0

    camera_data = bpy.data.cameras.new("SourcePoseCamera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("SourcePoseCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for fraction in fractions:
        frame = int(round(frame_start + ((frame_end - frame_start) * fraction)))
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        pose_objects = [] if meshes else _make_pose_curves(armatures[0], pose_material)
        pose_points = _pose_points(armatures[0])
        bounds_min, bounds_max = _world_bounds(meshes) if meshes else _point_bounds(pose_points)
        center = (bounds_min + bounds_max) * 0.5
        extent = bounds_max - bounds_min
        scale = max(float(extent.x), float(extent.y), float(extent.z), 0.1)
        light.location = center + Vector((-scale, -scale, scale * 1.5))
        _look_at(light, center)
        actor_right, actor_up, actor_forward = _actor_axes(armatures[0])
        views = {
            "front": -actor_forward,
            "side": actor_right,
            "three_quarter": (-actor_forward + actor_right + (actor_up * 0.2)).normalized(),
        }
        for view_name, direction in views.items():
            camera.location = center + (direction * scale * 3.0)
            _look_at(camera, center)
            camera.data.ortho_scale = scale * 1.25
            scene.render.filepath = str(
                args.output_dir / f"source_{view_name}_frame_{frame:04d}_t_{fraction:.4f}.png"
            )
            bpy.ops.render.render(write_still=True)
        for obj in pose_objects:
            curve = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.curves.remove(curve)


if __name__ == "__main__":
    main()
