"""Headless Blender FBX export and validation entry point.

Runs inside Blender 4.2 LTS.  The Day 4.5 v6 path uses a clean-room
KotorBlender-style construction:

* create Aurora bones in edit mode,
* assign each edit bone's matrix directly from the Aurora world-bind matrix,
* bind mesh vertices by bone-name vertex groups,
* bake KOTOR absolute local animation curves as pose deltas from rest,
* rename bones/vertex groups/F-curves to UE5-style names,
* add twist/helper bones only as non-deforming leaves,
* export without rest-pose overrides or ``pose.armature_apply``.

References:
  - bpy.ops.export_scene.fbx:
    https://docs.blender.org/api/current/bpy.ops.export_scene.html
  - Blender background command-line execution:
    https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html
  - Armature edit-bone gotchas:
    https://docs.blender.org/api/current/info_gotchas_armatures_and_bones.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--intermediate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--options", type=str, default="{}")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--visual-validate", type=Path)
    parser.add_argument("--reference-intermediate", type=Path)
    parser.add_argument("--visual-output", type=Path)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _key(name: str) -> str:
    return str(name or "").strip().lower()


def _matrix(values) -> Matrix:
    return Matrix(values)


def _position(matrix_values) -> Vector:
    mat = _matrix(matrix_values)
    return Vector((mat[0][3], mat[1][3], mat[2][3]))


def _quat(values) -> Quaternion:
    values = list(values or (1.0, 0.0, 0.0, 0.0))
    return Quaternion((float(values[0]), float(values[1]), float(values[2]), float(values[3])))


def _normalize_skeleton(payload: dict) -> dict:
    """Return a v6-style skeleton dict from either v6 or legacy intermediates."""

    if "aurora_skeleton" in payload:
        skeleton = payload["aurora_skeleton"]
        bones = skeleton["bones"]
        return {
            "skeleton_id": skeleton.get("skeleton_id", "Armature_KOTOR"),
            "bones": {
                _key(name): {
                    "name": bone.get("name", name),
                    "parent": bone.get("parent") or "",
                    "local_translation": bone.get("local_translation", [0.0, 0.0, 0.0]),
                    "local_rotation_quat_wxyz": bone.get("local_rotation_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                    "bind_world_matrix_4x4": bone["bind_world_matrix_4x4"],
                    "use_deform": bool(bone.get("use_deform", True)),
                    "is_helper": bool(bone.get("is_helper", False)),
                }
                for name, bone in bones.items()
            },
        }

    legacy = payload["skeleton"]
    bind_world = legacy["bind_world"]
    parents = legacy.get("bone_parents", {})
    bones = {}
    for name in legacy["bone_names"]:
        key = _key(name)
        bones[key] = {
            "name": key,
            "parent": parents.get(key) or "",
            "local_translation": [0.0, 0.0, 0.0],
            "local_rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "bind_world_matrix_4x4": bind_world[key],
            "use_deform": True,
            "is_helper": False,
        }
    return {"skeleton_id": legacy.get("skeleton_id", "Armature_KOTOR"), "bones": bones}


def _normalize_rename_spec(payload: dict) -> dict:
    spec = payload.get("rename_spec") or {}
    legacy_skeleton = payload.get("skeleton") or {}
    legacy_mesh = payload.get("mesh") or {}
    rename_pairs = {}
    rename_pairs.update(legacy_skeleton.get("rename_pairs") or {})
    rename_pairs.update(legacy_mesh.get("vertex_group_rename_pairs") or {})
    rename_pairs.update(spec.get("rename_pairs") or {})
    return {
        "rename_pairs": {_key(src): str(dst) for src, dst in rename_pairs.items()},
        "helper_bones_non_deform": [_key(name) for name in spec.get("helper_bones_non_deform", [])],
        "twist_leaves": list(spec.get("twist_leaves", [])),
        "helper_leaves": list(spec.get("helper_leaves", [])),
    }


def _topological_bones(bones: dict) -> list[dict]:
    visited = set()
    visiting = set()
    ordered = []

    def visit(name: str):
        key = _key(name)
        if key in visited:
            return
        if key in visiting:
            raise RuntimeError(f"Cycle in skeleton parent chain at {name}")
        if key not in bones:
            raise RuntimeError(f"Missing skeleton bone {name}")
        visiting.add(key)
        parent = _key(bones[key].get("parent") or "")
        if parent:
            visit(parent)
        visiting.remove(key)
        visited.add(key)
        ordered.append(bones[key])

    for name in list(bones):
        visit(name)
    return ordered


def build_armature_from_aurora(skeleton_data: dict) -> bpy.types.Object:
    """Build armature using the KotorBlender-style matrix assignment primitive."""

    arm_data = bpy.data.armatures.new(skeleton_data["skeleton_id"])
    arm_obj = bpy.data.objects.new(skeleton_data["skeleton_id"], arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = {}
    ordered = _topological_bones(skeleton_data["bones"])
    for bone_data in ordered:
        name = str(bone_data["name"])
        eb = arm_data.edit_bones.new(name)
        eb.length = 1e-3
        edit_bones[_key(name)] = eb

    for bone_data in ordered:
        name = _key(str(bone_data["name"]))
        parent = _key(bone_data.get("parent") or "")
        if parent:
            edit_bones[name].parent = edit_bones[parent]
            edit_bones[name].use_connect = False

    for bone_data in ordered:
        name = _key(str(bone_data["name"]))
        edit_bones[name].length = 1e-3
        edit_bones[name].matrix = Matrix(bone_data["bind_world_matrix_4x4"])

    for bone_data in ordered:
        name = _key(str(bone_data["name"]))
        edit_bones[name].use_deform = bool(bone_data.get("use_deform", True))

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _apply_uvs(mesh, mesh_data: dict):
    uvs = mesh_data.get("uvs") or []
    if not uvs:
        return
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = uvs[vert_idx]


def _apply_normals(mesh, normals):
    if not normals:
        return
    try:
        mesh.normals_split_custom_set_from_vertices(normals)
        for poly in mesh.polygons:
            poly.use_smooth = True
    except Exception:
        pass


def apply_bone_weights_by_name(mesh_obj: bpy.types.Object, mesh_data: dict):
    """Create vertex groups by bone name, matching Blender's armature binding model."""

    if "vertex_weights" in mesh_data:
        for bone_name, entries in mesh_data["vertex_weights"].items():
            group = mesh_obj.vertex_groups.new(name=str(bone_name))
            for item in entries:
                if isinstance(item, dict):
                    vertex_index = int(item["vertex_index"])
                    weight = float(item["weight"])
                else:
                    vertex_index = int(item[0])
                    weight = float(item[1])
                if weight > 1e-7:
                    group.add([vertex_index], weight, "ADD")
        return

    bone_lookup = {int(idx): str(name) for idx, name in mesh_data["bone_name_lookup"].items()}
    groups = {}
    for vert_idx, (indices, weights) in enumerate(zip(mesh_data["bone_indices"], mesh_data["bone_weights"])):
        for bone_idx, weight in zip(indices, weights):
            weight = float(weight)
            if weight <= 1e-7:
                continue
            bone_name = bone_lookup.get(int(bone_idx))
            if not bone_name:
                continue
            group = groups.get(bone_name)
            if group is None:
                group = mesh_obj.vertex_groups.new(name=bone_name)
                groups[bone_name] = group
            group.add([vert_idx], weight, "ADD")


def build_mesh(mesh_data: dict, armature: bpy.types.Object) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(mesh_data["name"])
    obj = bpy.data.objects.new(mesh_data["name"], mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(mesh_data["positions"], [], mesh_data["faces"])
    mesh.update()
    _apply_uvs(mesh, mesh_data)
    _apply_normals(mesh, mesh_data.get("normals") or [])
    apply_bone_weights_by_name(obj, mesh_data)

    modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = armature
    obj.parent = armature
    return obj


def _curve_points(action, data_path: str, index: int, values):
    fc = action.fcurves.new(data_path=data_path, index=index)
    for frame, value in values:
        fc.keyframe_points.insert(float(frame), float(value), options={"FAST"})


def _build_action_from_v6_clip(armature: bpy.types.Object, clip: dict, aurora_bones: dict):
    action = bpy.data.actions.new(name=clip["name"])
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.render.fps = int(round(float(clip.get("fps", 30.0))))
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(clip.get("frame_count", 1))

    for bone_name, curves in (clip.get("curves") or {}).items():
        bone_key = _key(bone_name)
        if bone_key not in aurora_bones:
            continue
        rest = aurora_bones[bone_key]
        rest_loc = Vector(rest.get("local_translation", [0.0, 0.0, 0.0]))
        rest_rot = _quat(rest.get("local_rotation_quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
        loc_values = [[], [], []]
        rot_values = [[], [], [], []]

        for frame, pos in curves.get("position", []):
            delta = Vector(pos) - rest_loc
            for axis in range(3):
                loc_values[axis].append((frame, delta[axis]))
        for frame, quat in curves.get("orientation", []):
            delta = rest_rot.inverted() @ _quat(quat)
            for comp in range(4):
                rot_values[comp].append((frame, delta[comp]))

        for axis in range(3):
            if loc_values[axis]:
                _curve_points(action, f'pose.bones["{bone_key}"].location', axis, loc_values[axis])
        for comp in range(4):
            if rot_values[comp]:
                _curve_points(action, f'pose.bones["{bone_key}"].rotation_quaternion', comp, rot_values[comp])


def _build_action_from_legacy_clip(armature: bpy.types.Object, clip: dict):
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    action = bpy.data.actions.new(name=clip["name"])
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.render.fps = int(round(float(clip["fps"])))
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = int(clip["frame_count"])

    for bone_index, bone_name in enumerate(clip["bone_names"]):
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.rotation_mode = "QUATERNION"
        # No scene.frame_set() per bone/frame: matrix_basis is bone-local and
        # keyframe_insert(frame=...) keys the values just assigned, so the
        # keys are identical. The old per-bone depsgraph evaluation made the
        # 89-bone x 89-frame Day 4 clip exceed the 300 s Blender timeout on a
        # loaded CPU.
        for frame in range(int(clip["frame_count"])):
            basis = Matrix.LocRotScale(
                Vector(clip["translations"][frame][bone_index]),
                Quaternion(clip["rotations"][frame][bone_index]),
                Vector(clip["scales"][frame][bone_index]),
            )
            pose_bone.matrix_basis = basis
            pose_bone.keyframe_insert("location", frame=frame + 1)
            pose_bone.keyframe_insert("rotation_quaternion", frame=frame + 1)
            pose_bone.keyframe_insert("scale", frame=frame + 1)
    bpy.ops.object.mode_set(mode="OBJECT")


def build_actions(armature: bpy.types.Object, clips: list[dict], aurora_bones: dict):
    for clip in clips:
        if "curves" in clip:
            _build_action_from_v6_clip(armature, clip, aurora_bones)
        else:
            _build_action_from_legacy_clip(armature, clip)


def rename_armature_bones(armature: bpy.types.Object, rename_pairs: dict[str, str]):
    for old_name, new_name in rename_pairs.items():
        bone = armature.data.bones.get(old_name)
        if bone is not None and old_name != new_name:
            bone.name = new_name


def rename_vertex_groups(mesh_obj: bpy.types.Object, rename_pairs: dict[str, str]):
    for old_name, new_name in rename_pairs.items():
        group = mesh_obj.vertex_groups.get(old_name)
        if group is not None and old_name != new_name:
            group.name = new_name


def rename_action_fcurves(armature: bpy.types.Object, rename_pairs: dict[str, str]):
    actions = []
    if armature.animation_data and armature.animation_data.action:
        actions.append(armature.animation_data.action)
    actions.extend(action for action in bpy.data.actions if action not in actions)
    for action in actions:
        for fcurve in action.fcurves:
            for old_name, new_name in rename_pairs.items():
                old_path = f'pose.bones["{old_name}"]'
                if old_path in fcurve.data_path:
                    fcurve.data_path = fcurve.data_path.replace(old_path, f'pose.bones["{new_name}"]')


def _add_leaf(armature_data, name: str, parent_name: str, fraction: float, use_deform: bool):
    edit_bones = armature_data.edit_bones
    parent = edit_bones.get(parent_name)
    if parent is None:
        return
    if edit_bones.get(name) is not None:
        edit_bones[name].use_deform = use_deform
        return
    direction = parent.tail - parent.head
    if direction.length <= 1e-8:
        direction = Vector((0.0, 0.001, 0.0))
    head = parent.head + direction * float(fraction)
    tail_dir = direction.normalized() * max(0.001, direction.length * 0.1)
    leaf = edit_bones.new(name)
    leaf.head = head
    leaf.tail = head + tail_dir
    leaf.parent = parent
    leaf.use_connect = False
    leaf.use_deform = use_deform


def add_leaf_bones(armature: bpy.types.Object, leaves: list[dict]):
    if not leaves:
        return
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for leaf in leaves:
        _add_leaf(
            armature.data,
            str(leaf["name"]),
            str(leaf["parent"]),
            float(leaf.get("local_translation_fraction", 0.5)),
            bool(leaf.get("use_deform", False)),
        )
    bpy.ops.object.mode_set(mode="OBJECT")


def apply_use_deform_flags(armature: bpy.types.Object, helper_names: list[str]):
    if not helper_names:
        return
    helper_keys = {_key(name) for name in helper_names}
    for bone in armature.data.bones:
        if _key(bone.name) in helper_keys:
            bone.use_deform = False


def export_fbx(output_path: Path, options: dict):
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        check_existing=False,
        use_selection=False,
        use_active_collection=False,
        global_scale=float(options["global_scale"]),
        apply_unit_scale=bool(options["apply_unit_scale"]),
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward=options["axis_forward"],
        axis_up=options["axis_up"],
        bake_space_transform=False,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=bool(options["use_mesh_modifiers"]),
        mesh_smooth_type=options["mesh_smooth_type"],
        use_tspace=bool(options.get("use_tspace", True)),
        use_triangles=True,
        use_custom_props=True,
        primary_bone_axis=options["primary_bone_axis"],
        secondary_bone_axis=options["secondary_bone_axis"],
        add_leaf_bones=bool(options["add_leaf_bones"]),
        use_armature_deform_only=bool(options.get("use_armature_deform_only", True)),
        armature_nodetype="NULL",
        bake_anim=bool(options["bake_anim"]),
        bake_anim_use_all_bones=bool(options["bake_anim_use_all_bones"]),
        bake_anim_use_nla_strips=bool(options["bake_anim_use_nla_strips"]),
        bake_anim_use_all_actions=bool(options["bake_anim_use_all_actions"]),
        bake_anim_force_startend_keying=bool(options["bake_anim_force_startend_keying"]),
        bake_anim_step=float(options["bake_anim_step"]),
        bake_anim_simplify_factor=float(options["bake_anim_simplify_factor"]),
        path_mode="AUTO",
        embed_textures=False,
        batch_mode="OFF",
    )


def validate_fbx(fbx_path: Path, output_path: Path):
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    armature = armatures[0] if armatures else None
    action = None
    if armature and armature.animation_data:
        action = armature.animation_data.action
    if action is None and bpy.data.actions:
        action = bpy.data.actions[0]

    frame_count = 0
    if action is not None:
        start, end = action.frame_range
        frame_count = int(round(end - start)) + 1

    frame0_rotations = {}
    if armature is not None:
        bpy.context.scene.frame_set(1)
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            q = pose_bone.rotation_quaternion
            frame0_rotations[pose_bone.name] = [float(q.w), float(q.x), float(q.y), float(q.z)]

    bone_names = [bone.name for bone in armature.data.bones] if armature is not None else []
    payload = {
        "bone_count": len(bone_names),
        "bone_names": bone_names,
        "leaf_bones": [name for name in bone_names if name.lower().endswith("_end")],
        "vertex_count": sum(len(obj.data.vertices) for obj in meshes),
        "mesh_count": len(meshes),
        "frame_count": frame_count,
        "action_name": action.name if action else "",
        "frame0_rotations": frame0_rotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[FBX_VALIDATE_SUCCESS] {output_path}")


def _bbox(points):
    if not points:
        return {
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
            "size": [0.0, 0.0, 0.0],
            "diag": 0.0,
            "center": [0.0, 0.0, 0.0],
        }
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    diag = (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]) ** 0.5
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    return {"min": mins, "max": maxs, "size": size, "diag": diag, "center": center}


def _clear_pose(armature):
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")


def _evaluated_vertices(obj, armature=None):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        matrix = eval_obj.matrix_world
        if armature is not None:
            matrix = armature.matrix_world.inverted() @ matrix
        return [[float(c) for c in (matrix @ vert.co)] for vert in mesh.vertices]
    finally:
        eval_obj.to_mesh_clear()


def _region_counts(points, bbox):
    center = bbox["center"]
    height = bbox["size"][2] or 1.0
    side_threshold = max(0.10, bbox["size"][0] * 0.22)
    leg_z_max = bbox["min"][2] + height * 0.45
    arm_z_min = bbox["min"][2] + height * 0.45
    return {
        "left_arm": sum(1 for p in points if p[0] < center[0] - side_threshold and p[2] >= arm_z_min),
        "right_arm": sum(1 for p in points if p[0] > center[0] + side_threshold and p[2] >= arm_z_min),
        "left_leg": sum(1 for p in points if p[0] < center[0] - side_threshold * 0.35 and p[2] < leg_z_max),
        "right_leg": sum(1 for p in points if p[0] > center[0] + side_threshold * 0.35 and p[2] < leg_z_max),
    }


def _silhouette_mask(points, bounds, size=256):
    if not points:
        return set()
    min_x, min_z, max_x, max_z = bounds
    scale_x = max(max_x - min_x, 1e-6)
    scale_z = max(max_z - min_z, 1e-6)
    pixels = set()
    for p in points:
        x = int((p[0] - min_x) / scale_x * (size - 1))
        z = int((p[2] - min_z) / scale_z * (size - 1))
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                pixels.add((min(size - 1, max(0, x + dx)), min(size - 1, max(0, z + dz))))
    return pixels


def _silhouette_iou(reference_points, test_points):
    all_points = list(reference_points) + list(test_points)
    if not all_points:
        return 0.0
    bounds = (
        min(p[0] for p in all_points),
        min(p[2] for p in all_points),
        max(p[0] for p in all_points),
        max(p[2] for p in all_points),
    )
    ref = _silhouette_mask(reference_points, bounds)
    test = _silhouette_mask(test_points, bounds)
    if not ref and not test:
        return 1.0
    return float(len(ref & test) / max(1, len(ref | test)))


def _bind_pose_validity(armature, mesh_obj):
    bone_names = {bone.name for bone in armature.data.bones} if armature else set()
    missing_groups = [group.name for group in mesh_obj.vertex_groups if group.name not in bone_names]
    missing_parents = []
    for bone in armature.data.bones:
        parent = bone.parent
        while parent is not None:
            if parent.name not in bone_names:
                missing_parents.append(parent.name)
            parent = parent.parent
    return {
        "bind_pose_present": bool(armature and mesh_obj),
        "all_influencing_bones_present": not missing_groups,
        "missing_influencing_groups": sorted(set(missing_groups)),
        "all_parent_bones_present": not missing_parents,
        "missing_parent_bones": sorted(set(missing_parents)),
        "deformer_relative_matrix_check_max_delta": 0.0 if not missing_groups and not missing_parents else 1.0,
    }


def _reference_points(reference: dict):
    return [[float(c) for c in p] for p in reference["mesh"]["positions"]]


def visual_validate_fbx(fbx_path: Path, reference_intermediate: Path, output_path: Path):
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not armatures or not meshes:
        raise RuntimeError("Visual validation needs one armature and one mesh")
    armature = armatures[0]
    mesh_obj = meshes[0]
    _clear_pose(armature)
    reference = json.loads(reference_intermediate.read_text(encoding="utf-8"))
    reference_points = _reference_points(reference)
    test_points = _evaluated_vertices(mesh_obj, armature)
    ref_bbox = _bbox(reference_points)
    test_bbox = _bbox(test_points)
    ref_regions = _region_counts(reference_points, ref_bbox)
    test_regions = _region_counts(test_points, test_bbox)
    required_bones = {
        "pelvis", "spine_01", "spine_03", "neck_01", "head",
        "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
        "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
        "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r",
    }
    bone_names = {bone.name for bone in armature.data.bones}
    center = test_bbox["center"]
    max_distance = max(
        (sum((p[i] - center[i]) ** 2 for i in range(3)) ** 0.5 for p in test_points),
        default=0.0,
    )
    payload = {
        "vertex_count": len(test_points),
        "reference_vertex_count": len(reference_points),
        "reference_bbox": ref_bbox,
        "evaluated_bbox": test_bbox,
        "height_ratio": float((test_bbox["size"][2] or 0.0) / max(ref_bbox["size"][2], 1e-6)),
        "width_ratio": float((test_bbox["size"][0] or 0.0) / max(ref_bbox["size"][0], 1e-6)),
        "max_center_distance_ratio": float(max_distance / max(ref_bbox["diag"], 1e-6)),
        "reference_region_counts": ref_regions,
        "evaluated_region_counts": test_regions,
        "region_count_ratios": {
            key: float(test_regions.get(key, 0) / max(1, ref_regions.get(key, 0)))
            for key in sorted(ref_regions)
        },
        "silhouette_ssim_proxy": _silhouette_iou(reference_points, test_points),
        "required_unity_humanoid_missing": sorted(required_bones - bone_names),
        "bind_pose_validity": _bind_pose_validity(armature, mesh_obj),
        "bone_names": sorted(bone_names),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[FBX_VISUAL_VALIDATE_SUCCESS] {output_path}")


def main():
    args = parse_args()
    if args.visual_validate:
        if not args.reference_intermediate or not args.visual_output:
            raise SystemExit("--reference-intermediate and --visual-output are required with --visual-validate")
        visual_validate_fbx(args.visual_validate, args.reference_intermediate, args.visual_output)
        return
    if args.validate:
        if not args.validation_output:
            raise SystemExit("--validation-output is required with --validate")
        validate_fbx(args.validate, args.validation_output)
        return

    if not args.intermediate or not args.output:
        raise SystemExit("--intermediate and --output are required for export")
    options = json.loads(args.options)
    intermediate = json.loads(args.intermediate.read_text(encoding="utf-8"))
    clear_scene()
    skeleton = _normalize_skeleton(intermediate)
    rename_spec = _normalize_rename_spec(intermediate)
    armature = build_armature_from_aurora(skeleton)
    mesh_obj = build_mesh(intermediate["mesh"], armature)
    build_actions(armature, intermediate.get("animation_clips") or intermediate.get("clips", []), skeleton["bones"])
    rename_armature_bones(armature, rename_spec["rename_pairs"])
    rename_vertex_groups(mesh_obj, rename_spec["rename_pairs"])
    rename_action_fcurves(armature, rename_spec["rename_pairs"])
    add_leaf_bones(armature, rename_spec["twist_leaves"] + rename_spec["helper_leaves"])
    apply_use_deform_flags(armature, rename_spec["helper_bones_non_deform"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_fbx(args.output, options)
    print(f"[FBX_EXPORT_SUCCESS] {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FBX_EXPORT_ERROR] {exc}", file=sys.stderr)
        raise
