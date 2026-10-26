"""Render visual acceptance frames for the retargeted high-fidelity player."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


def _args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(values)


def _look_at(camera, point: Vector) -> None:
    camera.rotation_euler = (point - camera.location).to_track_quat("-Z", "Y").to_euler()


def _apply_pose(armature, sidecar: dict, frame_index: int) -> None:
    desired_world = {}
    for bone in armature.data.bones:
        rest_local = (
            bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
            if bone.parent
            else bone.matrix_local.copy()
        )
        values = sidecar["tracks"][bone.name]["rotations_xyzw"][frame_index]
        delta = Quaternion((values[3], values[0], values[1], values[2])).to_matrix().to_4x4()
        pose_local = rest_local @ delta
        desired = (
            desired_world[bone.parent.name] @ pose_local
            if bone.parent
            else pose_local
        )
        desired_world[bone.name] = desired
        armature.pose.bones[bone.name].matrix = desired
    bpy.context.view_layer.update()


def main() -> int:
    args = _args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(args.player), use_anim=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("Expected one player armature and mesh")
    armature = armatures[0]
    points = [mesh.matrix_world @ Vector(corner) for mesh in meshes for corner in mesh.bound_box]
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    center = (low + high) * 0.5
    dimensions = high - low

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Player Retarget Audit World")
    scene.world.color = (0.055, 0.065, 0.08)

    camera_location = (
        center + Vector((3.2, 0.0, 3.6))
        if dimensions.y > dimensions.z
        else center + Vector((4.2, -3.2, 0.4))
    )
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.lens = 58.0
    _look_at(camera, center)
    scene.camera = camera

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = (
        ("idle_mid", "pause1", 100),
        ("walk_a", "walk", 7),
        ("walk_b", "walk", 23),
    )
    rendered = []
    for label, clip, frame_index in samples:
        sidecar = json.loads((args.sidecar_dir / f"{clip}.json").read_text(encoding="utf-8"))
        _apply_pose(armature, sidecar, min(frame_index, sidecar["frame_count"] - 1))
        destination = args.output_dir / f"player_retarget_v5_{label}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        rendered.append({"label": label, "path": str(destination), "bytes": destination.stat().st_size})
    receipt = {
        "schema": "ghoststudio.enclave.player-retarget-visual.v5",
        "renders": rendered,
        "all_checks_passed": all(item["bytes"] >= 10_000 for item in rendered),
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
