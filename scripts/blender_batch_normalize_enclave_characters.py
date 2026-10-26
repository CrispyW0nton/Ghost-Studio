"""Normalize every canonical enclave character GLB into Unreal binary FBXs."""

from __future__ import annotations

import argparse
import bmesh
import json
import math
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--roster",
        type=Path,
        help="Optional canonical roster path for isolated pilot export roots.",
    )
    parser.add_argument(
        "--glb-receipt",
        type=Path,
        help="Optional subset GLB receipt used for isolated validation runs.",
    )
    parser.add_argument(
        "--utc",
        action="append",
        help="Optional UTC resref filter; may be repeated for targeted rebuilds.",
    )
    return parser.parse_args(argv)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "character"


def _reset_scene() -> None:
    # Selection-based deletion skips hidden or excluded startup-file objects.
    # Those helpers then contaminate every subsequent character export (the
    # user's Blender startup scene included hidden Icosphere meshes).  Remove
    # datablocks directly so each character starts from a truly empty scene.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # glTF actions commonly carry a fake user.  A users==0-only cleanup leaves
    # those clips resident between characters, so the next import receives
    # .001 action names and the normalizer can silently bake the previous
    # character's animation into the current skeleton/head hook.
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for collection in (
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1.0


def _select_all() -> None:
    bpy.ops.object.select_all(action="SELECT")


def _remove_rig_display_helpers() -> list[str]:
    """Remove Blender-only custom bone shapes before FBX export.

    The glTF importer creates an ``Icosphere`` mesh used solely as a pose-bone
    custom display shape.  It has no KOTOR material or gameplay geometry and
    must never become an Unreal skeletal-mesh section.
    """

    helper_objects = {
        pose_bone.custom_shape
        for armature in bpy.context.scene.objects
        if armature.type == "ARMATURE"
        for pose_bone in armature.pose.bones
        if pose_bone.custom_shape is not None
    }
    names = sorted(obj.name for obj in helper_objects)
    for armature in bpy.context.scene.objects:
        if armature.type != "ARMATURE":
            continue
        for pose_bone in armature.pose.bones:
            if pose_bone.custom_shape in helper_objects:
                pose_bone.custom_shape = None
    for obj in helper_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    return names


def _strip_malformed_bas_tongue_faces() -> list[str]:
    """Keep tongue bind clusters but remove the malformed visible surface.

    The original Odyssey tongue skin is required during glTF import because its
    first-skin bind frame keeps the shared humanoid head/body armature stable,
    and the same clusters must survive FBX export.  Its polygons are malformed
    in Blender, however, and stretch into a vertical spike.  Remove faces only;
    vertices, deform weights, and bind clusters remain available to the exporter.
    """

    malformed = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.casefold() == "tongue"
    ]
    names = sorted(obj.name for obj in malformed)
    for obj in malformed:
        editable = bmesh.new()
        editable.from_mesh(obj.data)
        bmesh.ops.delete(editable, geom=list(editable.faces), context="FACES_ONLY")
        editable.to_mesh(obj.data)
        editable.free()
        obj.data.update()
    return names


def _convert_uv_origin_for_fbx() -> list[str]:
    """Convert Blender's glTF UV origin once for the FBX interchange boundary."""

    converted: list[str] = []
    for mesh in bpy.data.meshes:
        if not mesh.uv_layers:
            continue
        for layer in mesh.uv_layers:
            for loop in layer.data:
                loop.uv.y = 1.0 - float(loop.uv.y)
        converted.append(mesh.name)
    return sorted(converted)


def _skin_bone_parented_meshes(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
) -> list[dict[str, str]]:
    """Bake rigid glTF bone children into the shared Unreal skin.

    Blender's FBX exporter preserves these meshes as rigid bone children, but
    Unreal's skeletal importer drops their object-space offset while combining
    them into the skeletal mesh.  KOTOR heads, eyes, teeth, and tongues then
    land at the character origin.  Baking the evaluated rest placement into
    armature-local vertices and weighting each rigid mesh to its authored
    parent bone preserves both its position and its animation in Unreal.
    """

    converted: list[dict[str, str]] = []
    previous_pose_position = armature.data.pose_position
    _clear_object_actions(list(bpy.context.scene.objects))
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    armature_inverse = armature.matrix_world.inverted()
    for mesh in meshes:
        if mesh.parent is not armature or mesh.parent_type != "BONE":
            continue
        bone_name = str(mesh.parent_bone or "")
        if not bone_name or armature.data.bones.get(bone_name) is None:
            raise RuntimeError(
                f"Rigid mesh {mesh.name} references missing parent bone {bone_name!r}"
            )

        armature_local = armature_inverse @ mesh.matrix_world
        mesh.data.transform(armature_local)
        mesh.parent = armature
        mesh.parent_type = "OBJECT"
        mesh.parent_bone = ""
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        # The glTF importer can leave transform actions on rigid facial parts
        # (notably Yoda/Vandar eyelids and eyes).  Once the part is weighted to
        # its authored parent bone, retaining that object action applies the
        # same motion twice and throws the part away from the face in FBX.
        mesh.animation_data_clear()

        mesh.vertex_groups.clear()
        group = mesh.vertex_groups.new(name=bone_name)
        group.add(range(len(mesh.data.vertices)), 1.0, "REPLACE")
        modifier = mesh.modifiers.new(name="GhostStudioRigidAttachmentSkin", type="ARMATURE")
        modifier.object = armature
        converted.append({"mesh": mesh.name, "bone": bone_name})
    armature.data.pose_position = previous_pose_position
    bpy.context.view_layer.update()
    return converted


def _attachment_descendants(root: bpy.types.Object) -> list[bpy.types.Object]:
    ordered: list[bpy.types.Object] = []

    def visit(node: bpy.types.Object) -> None:
        ordered.append(node)
        for child in sorted(node.children, key=lambda item: item.name.casefold()):
            visit(child)

    visit(root)
    return ordered


def _mesh_world_center(mesh: bpy.types.Object, world_matrix: Matrix) -> Vector:
    points = [world_matrix @ vertex.co for vertex in mesh.data.vertices]
    return Vector(
        tuple(
            (min(point[axis] for point in points) + max(point[axis] for point in points))
            * 0.5
            for axis in range(3)
        )
    )


def _preserve_authored_attachment_frames(
    attachment_meshes: list[bpy.types.Object],
    authored_world: dict[bpy.types.Object, Matrix],
    rest_world: dict[bpy.types.Object, Matrix],
) -> dict[bpy.types.Object, Vector]:
    """Keep each facial section in the complete authored head-relative frame.

    KOTOR BAS heads contain a transform-node hierarchy whose child mesh data
    does not share the head mesh's origin.  Changing the body armature from
    pose to rest moves and rescales the head and its facial sections by
    different amounts.  A center-only correction leaves the eyes and lids in
    the wrong scale/rotation frame even when their bounds centers appear close.
    Map every authored facial object through the same full transform that maps
    the authored head onto the corrected rest head before rigid skinning.
    """

    exact_heads = [mesh for mesh in attachment_meshes if mesh.name.casefold() == "head"]
    if not exact_heads:
        return {}
    head = max(exact_heads, key=lambda item: len(item.data.vertices))
    head_frame_delta = rest_world[head] @ authored_world[head].inverted_safe()
    corrections: dict[bpy.types.Object, Vector] = {}
    for mesh in attachment_meshes:
        if mesh is head:
            corrections[mesh] = Vector((0.0, 0.0, 0.0))
            continue
        previous_world = rest_world[mesh]
        target_world = head_frame_delta @ authored_world[mesh]
        current_center = _mesh_world_center(mesh, rest_world[mesh])
        target_center = _mesh_world_center(mesh, target_world)
        correction = target_center - current_center
        if correction.length > 0.05:
            raise RuntimeError(
                f"Unsafe BAS facial offset correction {correction.length:.6f} m "
                f"on {mesh.name}"
            )
        # Preserve the entire authored transform frame, not merely its center.
        # ``previous_world`` remains explicit here so the receipt continues to
        # describe the visible translation introduced by this correction.
        rest_world[mesh] = target_world
        corrections[mesh] = correction
    return corrections


def _boundary_components(
    mesh: bpy.types.Object,
    world_matrix: Matrix,
) -> list[dict[str, object]]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    for polygon in mesh.data.polygons:
        vertices = list(polygon.vertices)
        for index, vertex in enumerate(vertices):
            other = vertices[(index + 1) % len(vertices)]
            edge_counts[tuple(sorted((int(vertex), int(other))))] += 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (left, right), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    components: list[dict[str, object]] = []
    unvisited = set(adjacency)
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        indices = {seed}
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in indices:
                    continue
                indices.add(neighbor)
                unvisited.discard(neighbor)
                stack.append(neighbor)
        if len(indices) < 3:
            continue
        points = [world_matrix @ mesh.data.vertices[index].co for index in indices]
        center = Vector(
            tuple(statistics.median(point[axis] for point in points) for axis in range(3))
        )
        components.append({"vertices": len(indices), "center": center})
    return components


def _head_neck_boundary_correction(
    attachment_meshes: list[bpy.types.Object],
    meshes: list[bpy.types.Object],
    rest_world: dict[bpy.types.Object, Matrix],
) -> Vector:
    """Return the small 3D translation that aligns the actual open neck loops.

    Broad surface-band centres are biased by collars, shoulders, and Twi'lek
    lekku.  Both the detachable head and body torso preserve an open boundary
    loop at the neck, so the nearest pair is the authoritative seam.  Aligning
    their centroids corrects the source-authored sub-centimeter/centimeter
    offset without reshaping either mesh or changing animation timing.
    """

    exact_heads = [mesh for mesh in attachment_meshes if mesh.name.casefold() == "head"]
    head_candidates = exact_heads or [
        mesh for mesh in attachment_meshes if "head" in mesh.name.casefold()
    ]
    exact_torsos = [mesh for mesh in meshes if mesh.name.casefold() == "torso"]
    torso_candidates = exact_torsos or [
        mesh for mesh in meshes if "torso" in mesh.name.casefold()
    ]
    if not head_candidates or not torso_candidates:
        return Vector((0.0, 0.0, 0.0))

    head = max(head_candidates, key=lambda item: len(item.data.vertices))
    torso = max(torso_candidates, key=lambda item: len(item.data.vertices))
    head_loops = _boundary_components(head, rest_world[head])
    torso_loops = _boundary_components(torso, torso.matrix_world.copy())
    if not head_loops or not torso_loops:
        return Vector((0.0, 0.0, 0.0))
    head_loop, torso_loop = min(
        (
            (head_loop, torso_loop)
            for head_loop in head_loops
            for torso_loop in torso_loops
        ),
        key=lambda pair: (pair[0]["center"] - pair[1]["center"]).length,
    )
    correction = torso_loop["center"] - head_loop["center"]
    if correction.length > 0.025:
        raise RuntimeError(
            f"Unsafe BAS head seam correction {correction.length:.6f} m on {head.name}"
        )
    return correction


def _bake_head_attachment_into_armature(
    armature: bpy.types.Object,
    meshes: list[bpy.types.Object],
    required: list[str],
    actions_by_name: dict[str, bpy.types.Action],
) -> tuple[list[dict[str, str]], dict[str, bpy.types.Action]]:
    """Merge the BAS head object tree into the body's skeletal contract.

    Ghost Studio correctly evaluates KOTOR heads as a transform hierarchy
    parented to the body's ``headhook`` attachment.  Blender's FBX exporter,
    however, otherwise serializes that hierarchy beside the body armature.
    Unreal then promotes the object nodes to a second pseudo-skeleton and its
    FBX basis conversion introduces a visible lateral neck seam.  Bake each
    visible head/face mesh into armature-local rest space, give it an identity
    rigid bone, and bake every source animation at 30 fps.  Rest placement and
    facial animation therefore use one unambiguous skeletal space in Unreal.
    """

    roots = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name.casefold() == "headhook"
        and obj.parent is armature
        and obj.parent_type == "BONE"
        and obj.parent_bone
    ]
    if not roots:
        return [], actions_by_name
    if len(roots) != 1:
        raise RuntimeError(
            f"Expected one BAS headhook on {armature.name}, found {len(roots)}"
        )

    root = roots[0]
    nodes = _attachment_descendants(root)
    node_set = set(nodes)
    attachment_meshes = [obj for obj in nodes if obj.type == "MESH"]
    if not attachment_meshes:
        raise RuntimeError(f"{armature.name} headhook has no visible attachment meshes")

    # Capture the authored visible registration before switching the body
    # armature to REST. The BAS hierarchy moves different facial mesh origins
    # by different amounts at that boundary even when no action is active.
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    authored_attachment_world = {
        mesh: mesh.matrix_world.copy()
        for mesh in attachment_meshes
    }

    # Capture the authored bind state, not the glTF importer's currently active
    # (usually ``walk``) action.  Baking that active frame into mesh vertices is
    # the exact source of the apparent off-centre headhook in Unreal.
    _clear_object_actions(list(bpy.context.scene.objects))
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    rest_world = {obj: obj.matrix_world.copy() for obj in nodes}
    seam_correction_vector = _head_neck_boundary_correction(
        attachment_meshes, meshes, rest_world
    )
    seam_correction = Matrix.Translation(tuple(seam_correction_vector))
    rest_world = {
        obj: seam_correction @ matrix for obj, matrix in rest_world.items()
    }
    facial_offset_corrections = _preserve_authored_attachment_frames(
        attachment_meshes,
        authored_attachment_world,
        rest_world,
    )
    armature_inverse = armature.matrix_world.inverted_safe()
    original_bone_names = [bone.name for bone in armature.data.bones]

    bone_name_by_mesh: dict[bpy.types.Object, str] = {}
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    skeleton_root = armature.data.edit_bones.get("cutscenedummy")
    if skeleton_root is None:
        skeleton_root = next(
            (bone for bone in armature.data.edit_bones if bone.parent is None),
            None,
        )
    if skeleton_root is None:
        raise RuntimeError(f"{armature.name} has no root bone for BAS attachments")
    for mesh in attachment_meshes:
        desired = f"bas_head_{mesh.name}"
        if armature.data.edit_bones.get(desired) is not None:
            desired = f"{desired}__rigid"
        bone = armature.data.edit_bones.new(desired)
        # Rigid attachment vertices are baked into armature-local rest space.
        # Keep their deform bones at identity so neither Blender FBX nor Unreal
        # can reapply the headhook rest translation as a second bind offset.
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.025, 0.0)
        bone.length = 0.025
        # One FBX skeleton must have one root.  Keeping rigid facial bones as
        # additional roots makes Blender split the animation stack per root
        # and Unreal sees incomplete facial clips.
        bone.parent = skeleton_root
        bone.use_connect = False
        bone_name_by_mesh[mesh] = bone.name
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.animation_data_create()

    source_actions = {
        name: actions_by_name[name.casefold()]
        for name in required
    }
    for name, action in source_actions.items():
        action.name = f"__GhostSource_{name}"

    participants = [
        obj
        for obj in bpy.context.scene.objects
        if obj is armature or obj in node_set
    ]
    baked_by_name: dict[str, bpy.types.Action] = {}
    for name in required:
        source_action = source_actions[name]
        baked = bpy.data.actions.new(name=name)
        start = int(math.floor(source_action.frame_range[0]))
        end = int(math.ceil(source_action.frame_range[1]))
        for frame in range(start, end + 1):
            armature.data.pose_position = "POSE"
            _assign_source_action(participants, source_action)
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            # Preserve authored local pose deltas. Reapplying global pose
            # matrices bone-by-bone through an already parented hierarchy
            # compounds parent transforms and produces the characteristic
            # elongated/rubber limbs seen in Unreal.
            body_pose = {
                bone_name: armature.pose.bones[bone_name].matrix_basis.copy()
                for bone_name in original_bone_names
            }
            depsgraph = bpy.context.evaluated_depsgraph_get()
            attachment_world = {
                mesh: seam_correction
                @ mesh.evaluated_get(depsgraph).matrix_world.copy()
                for mesh in attachment_meshes
            }

            armature.animation_data.action = baked
            bpy.context.scene.frame_set(frame)
            for bone_name, matrix_basis in body_pose.items():
                armature.pose.bones[bone_name].matrix_basis = matrix_basis
            bpy.context.view_layer.update()
            for mesh, world_matrix in attachment_world.items():
                armature.pose.bones[bone_name_by_mesh[mesh]].matrix = (
                    armature_inverse
                    @ world_matrix
                    @ rest_world[mesh].inverted_safe()
                    @ armature.matrix_world
                )
            bpy.context.view_layer.update()
            for pose_bone in armature.pose.bones:
                pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
                pose_bone.keyframe_insert(
                    "rotation_quaternion", frame=frame, group=pose_bone.name
                )
                pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)
        baked_by_name[name.casefold()] = baked

    _clear_object_actions(participants)
    armature.animation_data.action = None
    bpy.context.scene.frame_set(0)
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()

    converted: list[dict[str, str]] = []
    for mesh in attachment_meshes:
        mesh.data = mesh.data.copy()
        mesh.data.transform(armature_inverse @ rest_world[mesh])
        mesh.animation_data_clear()
        mesh.parent = armature
        mesh.parent_type = "OBJECT"
        mesh.parent_bone = ""
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        mesh.vertex_groups.clear()
        bone_name = bone_name_by_mesh[mesh]
        group = mesh.vertex_groups.new(name=bone_name)
        group.add(range(len(mesh.data.vertices)), 1.0, "REPLACE")
        modifier = mesh.modifiers.new(
            name="GhostStudioBASHeadAttachmentSkin", type="ARMATURE"
        )
        modifier.object = armature
        converted.append(
            {
                "mesh": mesh.name,
                "bone": bone_name,
                "neck_seam_correction_m": list(seam_correction_vector),
                "neck_seam_correction_distance_m": seam_correction_vector.length,
                "facial_offset_correction_m": list(
                    facial_offset_corrections.get(mesh, Vector((0.0, 0.0, 0.0)))
                ),
            }
        )

    for obj in reversed(nodes):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in source_actions.values():
        if action.users == 0:
            bpy.data.actions.remove(action)
    # Keep the base mesh export in an explicit rest pose.  The action exporter
    # opts back into POSE mode per clip below; leaving the final sampled facial
    # frame active here would otherwise be serialized as the FBX bind pose.
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    return converted, baked_by_name


def _action_slot(action, obj: bpy.types.Object):
    identifier = f"OB{obj.name}"
    return next(
        (slot for slot in getattr(action, "slots", []) if slot.identifier == identifier),
        None,
    )


def _clear_object_actions(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.animation_data is not None:
            obj.animation_data.action = None


def _assign_source_action(objects: list[bpy.types.Object], action) -> None:
    for obj in objects:
        slot = _action_slot(action, obj)
        if slot is None:
            if obj.animation_data is not None:
                obj.animation_data.action = None
            continue
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot


def _rigid_hierarchy_to_armature(
    meshes: list[bpy.types.Object],
    required: list[str],
) -> tuple[bpy.types.Object, list[dict[str, str]]]:
    """Convert an articulated glTF object hierarchy into an Unreal skeleton.

    KOTOR droids are authored as rigid mesh nodes rather than weighted skin
    sections.  Blender therefore imports their canonical GLB as animated
    objects and no Armature.  Unreal needs one skeletal hierarchy, so this
    samples each original object action onto pose bones, then rigid-weights
    every visible part to the bone representing its authored node.
    """

    nodes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type in {"EMPTY", "MESH"}
    ]
    node_set = set(nodes)
    if not nodes or not meshes:
        raise RuntimeError("Rigid hierarchy conversion requires nodes and meshes")

    source_actions = {
        name: next(
            (action for action in bpy.data.actions if action.name.casefold() == name.casefold()),
            None,
        )
        for name in required
    }
    missing = [name for name, action in source_actions.items() if action is None]
    if missing:
        raise RuntimeError(f"Rigid hierarchy actions missing: {missing}")

    _clear_object_actions(nodes)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    rest_world = {obj: obj.matrix_world.copy() for obj in nodes}

    armature_data = bpy.data.armatures.new("GhostStudioRigidSkeleton")
    armature = bpy.data.objects.new("GhostStudioRigidSkeleton", armature_data)
    bpy.context.scene.collection.objects.link(armature)
    armature.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for obj in nodes:
        bone = armature_data.edit_bones.new(obj.name)
        rest = rest_world[obj]
        rotation = rest.to_3x3().normalized().to_4x4()
        rotation.translation = rest.translation
        bone.matrix = rotation
        bone.length = 0.05
        edit_bones[obj] = bone
    for obj, bone in edit_bones.items():
        if obj.parent in node_set:
            bone.parent = edit_bones[obj.parent]
            bone.use_connect = False
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.animation_data_create()

    # Preserve the canonical names for the baked armature actions.
    for name, action in source_actions.items():
        action.name = f"__GhostSource_{name}"

    for name, source_action in source_actions.items():
        baked = bpy.data.actions.new(name=name)
        armature.animation_data.action = baked
        _assign_source_action(nodes, source_action)
        start = int(math.floor(source_action.frame_range[0]))
        end = int(math.ceil(source_action.frame_range[1]))
        for frame in range(start, end + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            evaluated = {
                obj: obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
                for obj in nodes
            }
            for obj in nodes:
                pose_bone = armature.pose.bones.get(obj.name)
                if pose_bone is None:
                    continue
                # Blender edit bones always use a Y-axis bone basis, which can
                # differ from the authored KOTOR object pivot even when their
                # origins and hierarchy agree.  Apply the sampled object delta
                # to Blender's actual rest bone rather than treating the
                # object matrix as the pose bone matrix directly.
                source_delta = evaluated[obj] @ rest_world[obj].inverted_safe()
                pose_bone.matrix = (
                    armature.matrix_world.inverted()
                    @ source_delta
                    @ armature.matrix_world
                    @ armature.data.bones[obj.name].matrix_local
                )
            bpy.context.view_layer.update()
            for pose_bone in armature.pose.bones:
                pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
                pose_bone.keyframe_insert("rotation_quaternion", frame=frame, group=pose_bone.name)
                pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)

    _clear_object_actions(nodes)
    armature.animation_data.action = None
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    converted = []
    for mesh in meshes:
        bone_name = mesh.name
        if armature.data.bones.get(bone_name) is None:
            raise RuntimeError(f"Rigid mesh {mesh.name} has no generated skeleton bone")
        mesh.data = mesh.data.copy()
        mesh.data.transform(rest_world[mesh])
        mesh.animation_data_clear()
        mesh.parent = armature
        mesh.parent_type = "OBJECT"
        mesh.parent_bone = ""
        mesh.matrix_parent_inverse = Matrix.Identity(4)
        mesh.matrix_basis = Matrix.Identity(4)
        mesh.vertex_groups.clear()
        group = mesh.vertex_groups.new(name=bone_name)
        group.add(range(len(mesh.data.vertices)), 1.0, "REPLACE")
        modifier = mesh.modifiers.new(name="GhostStudioRigidHierarchySkin", type="ARMATURE")
        modifier.object = armature
        converted.append({"mesh": mesh.name, "bone": bone_name})

    for obj in nodes:
        if obj.type == "EMPTY":
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        if action.name.startswith("__GhostSource_"):
            bpy.data.actions.remove(action)
    bpy.context.view_layer.update()
    return armature, converted


def _export_base(path: Path) -> None:
    # Base FBXs are skeleton/mesh bind assets, never animation snapshots.
    # Rigid-hierarchy characters and bone-parented attachments can otherwise
    # retain the final sampled action pose even after the action is unassigned,
    # which Unreal then serializes as a malformed reference pose.
    _clear_object_actions(list(bpy.context.scene.objects))
    bpy.context.scene.frame_set(0)
    for armature in (
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ):
        for pose_bone in armature.pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)
        armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    _select_all()
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="ROOT",
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )


def _material_family(name: str) -> str:
    return re.sub(r"[._]\d{3}$", "", str(name), flags=re.IGNORECASE)


def _upstream_image_node(socket, visited=None):
    visited = set() if visited is None else visited
    if socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        node = link.from_node
        if node.as_pointer() in visited:
            continue
        visited.add(node.as_pointer())
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node
        for input_socket in node.inputs:
            image_node = _upstream_image_node(input_socket, visited)
            if image_node is not None:
                return image_node
    return None


def _principled_base_color_image_node(material):
    if not material.use_nodes or material.node_tree is None:
        return None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        base_color = node.inputs.get("Base Color")
        image_node = _upstream_image_node(base_color)
        if image_node is not None:
            return image_node
    return None


def _principled_base_color_image(material):
    image_node = _principled_base_color_image_node(material)
    return image_node.image if image_node is not None else None


def _materialize_base_color_images(texture_root: Path) -> list[str]:
    """Give every imported glTF material family a real PNG before FBX export.

    Blender can display packed glTF images whose names have no extension, but
    its FBX exporter omits those image links for rigid head attachments.  The
    result keeps the facial UVs while silently binding the body image to every
    head section in Unreal.  Materialize and rebind each actual family first;
    object-space rig conversion must never rewrite UV or sampler identity.
    """

    if texture_root.exists():
        shutil.rmtree(texture_root)
    texture_root.mkdir(parents=True, exist_ok=True)
    family_images = {}
    staged: list[str] = []
    for material in sorted(bpy.data.materials, key=lambda item: item.name.casefold()):
        image_node = _principled_base_color_image_node(material)
        if image_node is None:
            continue
        image = image_node.image
        family = _material_family(material.name)
        destination = texture_root / f"{_safe(family)}.png"
        if family.casefold() not in family_images:
            image.filepath_raw = str(destination)
            image.file_format = "PNG"
            image.save()
            family_images[family.casefold()] = image
            staged.append(str(destination))
        image_node.image = family_images[family.casefold()]
    if not staged:
        raise RuntimeError("No Principled base-color images were available to materialize")
    return staged


def _stage_unreal_base_color_textures(base_fbx: Path, texture_root: Path) -> list[str]:
    """Give Blender's extensionless FBX media stable PNG names for Unreal.

    Blender writes embedded media into ``<model>.fbm`` without filename
    extensions.  Unreal's automated texture importer therefore cannot select
    a translator for those files.  The bytes are already PNG and already have
    the corrected GLB orientation, so copy (do not re-encode or flip) only the
    base-color payloads to a deterministic per-character texture directory.
    """

    media_root = base_fbx.with_suffix(".fbm")
    if texture_root.exists():
        shutil.rmtree(texture_root)
    texture_root.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    suffix = "_BaseColor"
    expected_families = {
        _material_family(material.name).casefold()
        for material in bpy.data.materials
        if _principled_base_color_image(material) is not None
    }
    if media_root.is_dir():
        for source in sorted(media_root.iterdir()):
            if not source.is_file() or source.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            material_name = source.name
            if material_name.casefold().endswith(suffix.casefold()):
                material_name = material_name[: -len(suffix)]
            family = _material_family(material_name)
            if expected_families and family.casefold() not in expected_families:
                continue
            destination = texture_root / f"{_safe(family)}.png"
            if not destination.exists():
                shutil.copyfile(source, destination)
                staged.append(str(destination))
    else:
        # Depending on the material composition Blender may embed every image
        # directly in the FBX and omit the .fbm directory.  Save the imported
        # GLB image buffer without transforming UVs or pixels.
        for image in sorted(bpy.data.images, key=lambda item: item.name.casefold()):
            image_name = Path(image.name).stem
            if not image_name.casefold().endswith(suffix.casefold()):
                continue
            material_name = _material_family(image_name[: -len(suffix)])
            destination = texture_root / f"{_safe(material_name)}.png"
            image.filepath_raw = str(destination)
            image.file_format = "PNG"
            image.save()
            staged.append(str(destination))
    if not staged:
        # A fully skeletalized BAS head no longer needs Blender to serialize
        # its old object hierarchy.  In that case FBX can embed the packed
        # glTF images directly and omit both ``.fbm`` media and the generated
        # ``_BaseColor`` image suffix.  Resolve the actual Principled BSDF
        # base-color image per material and save that packed buffer under the
        # same deterministic material contract used by the Unreal importer.
        for material in sorted(bpy.data.materials, key=lambda item: item.name.casefold()):
            if not material.use_nodes or material.node_tree is None:
                continue
            image = _principled_base_color_image(material)
            if image is None:
                continue
            destination = texture_root / f"{_safe(_material_family(material.name))}.png"
            if destination.exists():
                continue
            image.filepath_raw = str(destination)
            image.file_format = "PNG"
            image.save()
            staged.append(str(destination))
    # Unreal binds by the FBX material-slot family, so make sure every actual
    # Principled base-color family has one deterministic staged PNG.
    for material in sorted(bpy.data.materials, key=lambda item: item.name.casefold()):
        image = _principled_base_color_image(material)
        if image is None:
            continue
        material_key = _material_family(material.name)
        destination = texture_root / f"{_safe(material_key)}.png"
        if destination.exists():
            if str(destination) not in staged:
                staged.append(str(destination))
            continue
        image.filepath_raw = str(destination)
        image.file_format = "PNG"
        image.save()
        staged.append(str(destination))
    if not staged:
        raise RuntimeError(f"No base-color textures found in {media_root}")
    return staged


def _export_action(path: Path, action, armatures: list) -> None:
    for armature in armatures:
        armature.data.pose_position = "POSE"
        if armature.animation_data is None:
            armature.animation_data_create()
        armature.animation_data.action = action
        slot = _action_slot(action, armature)
        if slot is not None:
            armature.animation_data.action_slot = slot
    start, end = action.frame_range
    bpy.context.scene.frame_start = int(round(start))
    bpy.context.scene.frame_end = int(round(end))
    # Animation FBXs carry only the armature.  Exporting the BAS attachment
    # empties makes Unreal reinterpret animated helpers such as ``headhook``
    # as bone tracks even though they are not part of the corrected merged-head
    # skeleton.  Meshes/materials belong to the base FBX and are unnecessary
    # for animation import as well.
    bpy.ops.object.select_all(action="DESELECT")
    for armature in armatures:
        armature.select_set(True)
    bpy.context.view_layer.objects.active = armatures[0]
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        check_existing=False,
        use_selection=True,
        object_types={"ARMATURE"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="ROOT",
        # Animation-only FBXs target the already-imported Unreal skeleton.
        # KOTOR socket/helper leaves such as ``headhook`` are not deforming
        # bones and corrected merged-head skeletons intentionally omit them.
        # Exporting those leaves as animation tracks makes Unreal reject an
        # otherwise compatible clip ("Unable to retrieve bone index").
        # Ancestor bones required by deforming descendants remain included by
        # Blender's deform-only hierarchy export.
        use_armature_deform_only=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )


def main() -> int:
    args = _args()
    roster_path = (
        args.roster
        if args.roster is not None
        else args.export_root / "enclave_interior_roster_v1.json"
    )
    glb_receipt_path = (
        args.glb_receipt
        if args.glb_receipt is not None
        else args.export_root / "enclave_interior_character_glb_v1.json"
    )
    roster = json.loads(roster_path.read_text(encoding="utf-8"))["creatures"]
    glb_exports = json.loads(glb_receipt_path.read_text(encoding="utf-8"))["exports"]
    if args.utc:
        requested = {str(value).casefold() for value in args.utc}
        glb_exports = [
            row
            for row in glb_exports
            if str(row["utc_resref"]).casefold() in requested
        ]
        found = {str(row["utc_resref"]).casefold() for row in glb_exports}
        missing_requested = sorted(requested - found)
        if missing_requested:
            raise RuntimeError(
                f"Requested UTC resrefs missing from GLB receipt: {missing_requested}"
            )
    required_by_utc: dict[str, list[str]] = {}
    for row in roster:
        required_by_utc.setdefault(
            str(row["utc_resref"]).casefold(),
            [str(name) for name in row["animation_names"]],
        )

    records = []
    for index, exported in enumerate(glb_exports, start=1):
        utc_resref = str(exported["utc_resref"])
        source = Path(exported["asset"])
        required = (
            [str(exported["animation"])]
            if args.glb_receipt is not None and exported.get("animation")
            else required_by_utc[utc_resref.casefold()]
        )
        _reset_scene()
        bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True)
        removed_rig_helpers = _remove_rig_display_helpers()
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        actions_by_name = {action.name.casefold(): action for action in bpy.data.actions}
        missing = [name for name in required if name.casefold() not in actions_by_name]
        rigid_attachment_skins: list[dict[str, str]] = []
        bas_head_attachment_skins: list[dict[str, str]] = []
        rigid_hierarchy_skins: list[dict[str, str]] = []
        if not armatures and meshes and not missing:
            armature, rigid_hierarchy_skins = _rigid_hierarchy_to_armature(meshes, required)
            armatures = [armature]
            actions_by_name = {action.name.casefold(): action for action in bpy.data.actions}
            missing = [name for name in required if name.casefold() not in actions_by_name]
        if len(armatures) != 1 or not meshes or missing:
            raise RuntimeError(
                f"{utc_resref} normalization contract failed: "
                f"armatures={len(armatures)} meshes={len(meshes)} missing={missing}"
            )
        if not rigid_hierarchy_skins:
            bas_head_attachment_skins, baked_actions = (
                _bake_head_attachment_into_armature(
                    armatures[0], meshes, required, actions_by_name
                )
            )
            if baked_actions:
                actions_by_name.update(baked_actions)
            rigid_attachment_skins = _skin_bone_parented_meshes(armatures[0], meshes)

        character_root = args.export_root / _safe(utc_resref)
        normalized = character_root / "normalized"
        normalized_actions = character_root / "normalized_actions"
        normalized.mkdir(parents=True, exist_ok=True)
        normalized_actions.mkdir(parents=True, exist_ok=True)
        base_path = normalized / f"SK_{_safe(utc_resref)}_BAS_Unreal.fbx"
        debug_head = next(
            (obj for obj in meshes if obj.name.casefold() == "head"), None
        )
        if debug_head is not None:
            raw_points = [debug_head.matrix_world @ vertex.co for vertex in debug_head.data.vertices]
            print(
                "[BAS head bind]",
                utc_resref,
                "pose=",
                armatures[0].data.pose_position,
                "action=",
                armatures[0].animation_data.action.name
                if armatures[0].animation_data and armatures[0].animation_data.action
                else "",
                "x=",
                (
                    round(min(float(point.x) for point in raw_points), 6),
                    round(max(float(point.x) for point in raw_points), 6),
                ),
                flush=True,
            )
        material_source_root = character_root / "_fbx_texture_sources"
        materialized_texture_files = _materialize_base_color_images(
            material_source_root
        )
        stripped_malformed_meshes = _strip_malformed_bas_tongue_faces()
        uv_origin_converted_meshes = _convert_uv_origin_for_fbx()
        _export_base(base_path)
        texture_files = _stage_unreal_base_color_textures(
            base_path, character_root / "textures"
        )
        action_records = []
        for animation_name in required:
            action = actions_by_name[animation_name.casefold()]
            action_path = normalized_actions / f"A_{_safe(utc_resref)}_{_safe(animation_name)}.fbx"
            _export_action(action_path, action, armatures)
            if not action_path.is_file() or action_path.stat().st_size < 1024:
                raise RuntimeError(f"{utc_resref}:{animation_name} did not produce a valid FBX")
            action_records.append({
                "name": animation_name,
                "asset": str(action_path),
                "bytes": action_path.stat().st_size,
                "frame_start": int(round(action.frame_range[0])),
                "frame_end": int(round(action.frame_range[1])),
            })
        if not base_path.is_file() or base_path.stat().st_size < 1024:
            raise RuntimeError(f"{utc_resref} did not produce a valid base FBX")
        records.append({
            "utc_resref": utc_resref,
            "source_glb": str(source),
            "base_fbx": str(base_path),
            "base_bytes": base_path.stat().st_size,
            "texture_files": texture_files,
            "materialized_texture_files": materialized_texture_files,
            "armature": armatures[0].name,
            "bones": len(armatures[0].data.bones),
            "mesh_names": [obj.name for obj in meshes],
            "removed_rig_display_helpers": removed_rig_helpers,
            "stripped_malformed_meshes": stripped_malformed_meshes,
            "uv_origin_converted_meshes": uv_origin_converted_meshes,
            "bas_head_attachment_skins": bas_head_attachment_skins,
            "rigid_attachment_skins": rigid_attachment_skins,
            "rigid_hierarchy_skins": rigid_hierarchy_skins,
            "animations": action_records,
        })
        print(f"[{index}/{len(glb_exports)}] {utc_resref}", flush=True)

    receipt = {
        "schema": "ghoststudio.enclave-interior-unreal-fbx-normalization.v1",
        "character_count": len(records),
        "all_characters_valid": len(records) == len(glb_exports),
        "characters": records,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "receipt": str(args.receipt),
        "character_count": len(records),
        "all_characters_valid": receipt["all_characters_valid"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
