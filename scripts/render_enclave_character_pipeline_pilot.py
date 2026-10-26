"""Render comparable BAS/direct/normalized character files for visual diagnosis."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--animate-source",
        action="store_true",
        help="Render the supplied BAS/GLB source with its canonical walk action.",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=("front", "back", "left", "right"),
        help="Review view to render; may be repeated (default: back).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Optional explicit animation frame instead of the action midpoint.",
    )
    parser.add_argument(
        "--action",
        help="Optional exact action name for animated comparison renders.",
    )
    parser.add_argument(
        "--flip-uv-v",
        action="store_true",
        help="Diagnostic only: invert imported UV0 V before rendering.",
    )
    return parser.parse_args(argv)


def _reset() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _import(path: Path) -> None:
    if path.suffix.casefold() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path), import_pack_images=True)
    else:
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)


def _texture_directory(path: Path) -> Path:
    current = path.parent
    for _ in range(4):
        candidate = current / "textures"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return path.parent


def _material_key(name: str) -> str:
    value = re.sub(r"[._]\d{3}$", "", name, flags=re.IGNORECASE)
    return value.casefold()


def _relink_materials(path: Path) -> None:
    textures = {
        item.stem.casefold(): item
        for item in _texture_directory(path).glob("*.png")
    }
    for material in bpy.data.materials:
        key = _material_key(material.name)
        texture_path = textures.get(key)
        if texture_path is None:
            candidates = [item for name, item in textures.items() if key.startswith(name)]
            texture_path = candidates[0] if len(candidates) == 1 else None
        material.use_nodes = True
        nodes = material.node_tree.nodes
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")
        if principled is None:
            principled = nodes.new("ShaderNodeBsdfPrincipled")
        material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Roughness"].default_value = 0.65
        if texture_path is not None:
            image = bpy.data.images.load(str(texture_path), check_existing=True)
            sample = next(
                (
                    node
                    for node in nodes
                    if node.type == "TEX_IMAGE"
                    and node.image
                    and Path(bpy.path.abspath(node.image.filepath)).resolve() == texture_path.resolve()
                ),
                None,
            )
            if sample is None:
                sample = nodes.new("ShaderNodeTexImage")
            sample.image = image
            material.node_tree.links.new(sample.outputs["Color"], principled.inputs["Base Color"])


def _visible_meshes() -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if (
            "shadow" in obj.name.casefold()
            or obj.name.casefold() == "tongue"
            or len(obj.data.polygons) == 0
        ):
            obj.hide_render = True
            continue
        obj.hide_render = False
        meshes.append(obj)
    return meshes


def _flip_uv_v() -> None:
    for mesh in bpy.data.meshes:
        if not mesh.uv_layers:
            continue
        layer = mesh.uv_layers.active
        if layer is None:
            continue
        for loop in layer.data:
            loop.uv.y = 1.0 - float(loop.uv.y)


def _bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            points.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    lower = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    upper = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return lower, upper


def _point_camera(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def _add_area(name: str, location: tuple[float, float, float], energy: float, size: float) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    _point_camera(obj, Vector((0.0, 0.0, 0.8)))


def _canonical_action(path: Path) -> bpy.types.Action:
    if _ARGS.action:
        exact = next(
            (
                action
                for action in bpy.data.actions
                if action.name.casefold() == _ARGS.action.casefold()
            ),
            None,
        )
        if exact is None:
            raise RuntimeError(f"Action {_ARGS.action!r} not found in {path}")
        return exact
    preferred = "cwalk" if "ambdrd" in path.stem.casefold() else "walk"
    return next(
        (action for action in bpy.data.actions if action.name.casefold() == preferred),
        next(
            (action for action in bpy.data.actions if preferred in action.name.casefold()),
            sorted(bpy.data.actions, key=lambda item: item.name.casefold())[0],
        ),
    )


def _action_slot(action: bpy.types.Action, obj: bpy.types.Object):
    """Return the Blender 4.4+ action slot authored for ``obj``."""

    identifier = f"OB{obj.name}"
    return next(
        (
            slot
            for slot in getattr(action, "slots", [])
            if slot.identifier == identifier
        ),
        None,
    )


def _assign_action_to_authored_objects(action: bpy.types.Action) -> int:
    """Apply a multi-object glTF/FBX action without cross-assigning slots.

    KOTOR BAS heads are authored as an object hierarchy while the body is an
    armature. Assigning the same action blindly to every armature evaluates
    only the body's slot and produces a misleading source/export comparison.
    """

    assigned = 0
    for obj in bpy.context.scene.objects:
        slot = _action_slot(action, obj)
        if slot is None:
            continue
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot
        assigned += 1
    if assigned:
        return assigned

    # Blender's FBX importer commonly creates one single-slot action per root
    # and does not preserve the original object identifier. In that case the
    # selected armature action is unambiguous.
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) == 1:
        armature = armatures[0]
        armature.animation_data_create()
        armature.animation_data.action = action
        return 1
    return 0


def _render(
    path: Path,
    output: Path,
    animated: bool,
    view: str,
    explicit_frame: int | None,
) -> None:
    _reset()
    _import(path)
    _relink_materials(path)
    if _ARGS.flip_uv_v:
        _flip_uv_v()
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    for armature in armatures:
        armature.data.pose_position = "POSE" if animated else "REST"
    if animated and bpy.data.actions:
        action = _canonical_action(path)
        print(f"Animating {path.name} with action {action.name}", flush=True)
        assigned = _assign_action_to_authored_objects(action)
        if not assigned:
            raise RuntimeError(
                f"Action {action.name!r} has no compatible object slots in {path}"
            )
        start, end = action.frame_range
        bpy.context.scene.frame_set(
            explicit_frame if explicit_frame is not None else int(round((start + end) * 0.5))
        )
    else:
        bpy.context.scene.frame_set(0)

    # Ghost Studio's BAS interchange is Y-up.  Match the runtime actor's
    # authored -90 degree roll so the review image is an upright front view.
    interchange_root = bpy.data.objects.new("AuditInterchangeRoot", None)
    bpy.context.scene.collection.objects.link(interchange_root)
    for obj in list(bpy.context.scene.objects):
        if obj is interchange_root or obj.parent is not None:
            continue
        obj.parent = interchange_root
        obj.matrix_parent_inverse = Matrix.Identity(4)
    interchange_root.rotation_euler.x = math.radians(-90.0)
    interchange_root.rotation_euler.z = {
        "back": 0.0,
        "front": math.pi,
        "left": math.pi * 0.5,
        "right": -math.pi * 0.5,
    }[view]
    bpy.context.view_layer.update()

    meshes = _visible_meshes()
    lower, upper = _bounds(meshes)
    center = (lower + upper) * 0.5
    size = upper - lower

    scene = bpy.context.scene
    # Blender 5 renamed the Eevee identifier back to BLENDER_EEVEE; retain
    # compatibility with the 4.2/4.5 runners used by older Ghost Studio jobs.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(output)
    scene.world.color = (0.025, 0.025, 0.035)

    camera_data = bpy.data.cameras.new("AuditCamera")
    camera = bpy.data.objects.new("AuditCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 58.0
    vertical = max(float(size.z), 0.1)
    horizontal = max(float(size.x), 0.1)
    distance = max(vertical / (2.0 * math.tan(math.radians(34.0) * 0.5)), horizontal * 1.4)
    camera.location = center + Vector((0.0, -max(distance, float(size.y) * 1.2), vertical * 0.03))
    _point_camera(camera, center)

    light_scale = max(vertical, horizontal, 1.0)
    _add_area("Key", tuple(center + Vector((-light_scale, -light_scale, light_scale))), 900.0, light_scale)
    _add_area("Fill", tuple(center + Vector((light_scale, -0.2 * light_scale, 0.4 * light_scale))), 500.0, light_scale)
    _add_area("Rim", tuple(center + Vector((0.0, light_scale, light_scale))), 700.0, light_scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    global _ARGS
    args = _args()
    _ARGS = args
    views = args.view or ["back"]
    for path in args.input:
        animated = args.animate_source or "normalized_actions" in {
            part.casefold() for part in path.parts
        }
        suffix = "animated" if animated else "rest"
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem)
        parent = re.sub(r"[^A-Za-z0-9_]+", "_", path.parent.name)
        for view in views:
            output = args.output_dir / f"{parent}__{safe}__{suffix}__{view}.png"
            _render(path.resolve(), output.resolve(), animated, view, args.frame)
            print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
