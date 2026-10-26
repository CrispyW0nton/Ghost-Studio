"""Bake KOTOR duel clips onto the canonical BAS merged-head Unreal skeleton.

The legacy normalized duel FBXs contain a 45-bone body armature plus an
animated object hierarchy for the detachable head.  Animation-only FBXs that
strip those objects leave the canonical ``bas_head_*`` bones at reference pose,
which detaches the head during combat.  This converter samples both contracts:

* body pose-bone local deltas -> matching canonical body bones;
* visible head-object world deltas -> canonical rigid ``bas_head_*`` bones.

Only animation FBXs and a JSON receipt are written.  Source KOTOR exports and
canonical skeletal-mesh FBXs remain read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix


CLIPS = ("c2a1", "c2a2", "c2a3", "c2d1", "c2d2", "c2p1")


def _args() -> argparse.Namespace:
    values = list(__import__("sys").argv)
    values = values[values.index("--") + 1 :] if "--" in values else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--utc", required=True, choices=("dan13_jedi001", "dan13_jedim"))
    parser.add_argument("--base-fbx", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=list(CLIPS))
    parser.add_argument(
        "--head-mode",
        choices=("source-world", "inherit-parent"),
        default="source-world",
        help="Bake source object motion or retain rigid head bones under cutscenedummy.",
    )
    parser.add_argument(
        "--normalize-cutscenedummy",
        action="store_true",
        help="Remove the legacy 100x bridge scale/translation from cutscenedummy.",
    )
    parser.add_argument(
        "--unreal-root-scale-compensation",
        type=float,
        default=1.0,
        help=(
            "Authored scale for cutscenedummy animation keys before Blender's FBX "
            "unit conversion. Use 0.01 for Unreal animation-only imports, where "
            "the FBX exporter otherwise serializes this bridge track at scale 100."
        ),
    )
    return parser.parse_args(values)


def _import_fbx(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    return [obj for obj in bpy.context.scene.objects if obj not in before]


def _armature(objects: list[bpy.types.Object]) -> bpy.types.Object:
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    return armatures[0]


def _clear_actions(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        if obj.animation_data is not None:
            obj.animation_data.action = None


def _restore_actions(assignments: dict[bpy.types.Object, bpy.types.Action]) -> None:
    for obj, action in assignments.items():
        obj.animation_data_create()
        obj.animation_data.action = action


def _head_mesh_for_bone(
    source_objects: list[bpy.types.Object], bone_name: str
) -> bpy.types.Object | None:
    mesh_name = bone_name.removeprefix("bas_head_").removesuffix("__rigid")
    exact = [
        obj
        for obj in source_objects
        if obj.type == "MESH" and obj.name.casefold() == mesh_name.casefold()
    ]
    if exact:
        return exact[0]
    # Blender may suffix duplicate imports; accept only a unique stem match.
    candidates = [
        obj
        for obj in source_objects
        if obj.type == "MESH" and obj.name.split(".", 1)[0].casefold() == mesh_name.casefold()
    ]
    return candidates[0] if len(candidates) == 1 else None


def _export_animation(path: Path, armature: bpy.types.Object, action: bpy.types.Action) -> None:
    armature.data.pose_position = "POSE"
    armature.animation_data_create()
    start, end = action.frame_range
    bpy.context.scene.frame_start = int(round(start))
    bpy.context.scene.frame_end = int(round(end))
    # FBX serializes the currently evaluated pose as each bone node's default
    # transform before it samples animation.  Leaving the duel action active
    # therefore turns the final combat frame into the imported reference pose.
    # Keep only the baked action, evaluate a neutral pose for the FBX defaults,
    # and let the exporter's all-actions pass assign/sample the action afterward.
    for obj in bpy.context.scene.objects:
        if obj.animation_data is not None:
            obj.animation_data.action = None
    for candidate in list(bpy.data.actions):
        if candidate != action:
            bpy.data.actions.remove(candidate)
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    bpy.context.scene.frame_set(int(round(start)))
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    path.parent.mkdir(parents=True, exist_ok=True)
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
        use_armature_deform_only=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="AUTO",
        embed_textures=False,
    )


def _convert_one(args: argparse.Namespace, clip: str) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    target_objects = _import_fbx(args.base_fbx)
    target = _armature(target_objects)
    target.name = f"TARGET_{args.utc}"
    for obj in target_objects:
        if obj is not target:
            obj.name = f"TARGET_{obj.name}"

    source_path = args.source_dir / f"A_{args.utc}_{clip}.fbx"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_objects = _import_fbx(source_path)
    source = _armature(source_objects)

    assignments = {
        obj: obj.animation_data.action
        for obj in source_objects
        if obj.animation_data is not None and obj.animation_data.action is not None
    }
    if source not in assignments:
        raise RuntimeError(f"{source_path.name} has no body-armature action")
    starts = [action.frame_range[0] for action in assignments.values()]
    ends = [action.frame_range[1] for action in assignments.values()]
    start = int(math.floor(min(starts)))
    end = int(math.ceil(max(ends)))

    # Source-file rest/bind state for the animated head object hierarchy.
    _clear_actions(source_objects)
    source.data.pose_position = "REST"
    bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()
    head_bones = [bone.name for bone in target.data.bones if bone.name.startswith("bas_head_")]
    source_mesh_by_bone = {
        bone_name: _head_mesh_for_bone(source_objects, bone_name)
        for bone_name in head_bones
    }
    missing_head = [name for name, mesh in source_mesh_by_bone.items() if mesh is None]
    if missing_head:
        raise RuntimeError(f"Missing source head meshes for bones: {missing_head}")
    source_rest_world = {
        bone_name: mesh.matrix_world.copy()
        for bone_name, mesh in source_mesh_by_bone.items()
        if mesh is not None
    }
    target_body_bones = [
        bone.name for bone in target.data.bones if not bone.name.startswith("bas_head_")
    ]
    body_bones = [name for name in target_body_bones if source.pose.bones.get(name) is not None]
    missing_body = [name for name in target_body_bones if source.pose.bones.get(name) is None]
    if len(body_bones) < 40:
        raise RuntimeError(f"Only {len(body_bones)} canonical body bones matched source")

    baked = bpy.data.actions.new(name=f"A_{args.utc}_{clip}_HeadFixed")
    target.animation_data_create()
    target.animation_data.action = baked
    target.data.pose_position = "POSE"
    # Source head objects are evaluated in the source armature's world basis.
    # Conjugating them through the target armature (whose FBX import scale can
    # differ by 100x) turns centimetres into metres and launches the rigid head
    # bones far away in Unreal.  First derive the animation delta entirely in
    # source-armature space; both canonical skeletons share the same local KOTOR
    # bone basis, so that delta can then be assigned directly to the target.
    source_inverse = source.matrix_world.inverted_safe()
    head_translation_lengths: list[float] = []
    head_scale_errors: list[float] = []
    head_basis_translation_lengths: list[float] = []
    cutscene_scale_errors: list[float] = []
    cutscene_translation_lengths: list[float] = []

    for frame in range(start, end + 1):
        _restore_actions(assignments)
        source.data.pose_position = "POSE"
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        body_local = {
            name: source.pose.bones[name].matrix_basis.copy()
            for name in body_bones
        }
        depsgraph = bpy.context.evaluated_depsgraph_get()
        head_world = {
            bone_name: mesh.evaluated_get(depsgraph).matrix_world.copy()
            for bone_name, mesh in source_mesh_by_bone.items()
            if mesh is not None
        }

        target.animation_data.action = baked
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)
        for name, matrix_basis in body_local.items():
            target.pose.bones[name].matrix_basis = matrix_basis
        if args.normalize_cutscenedummy:
            cutscene = target.pose.bones.get("cutscenedummy")
            if cutscene is None:
                raise RuntimeError("Canonical target is missing cutscenedummy")
            # Legacy animation-only FBXs carry the scene-unit conversion on
            # cutscenedummy (scale 100 plus a 100x local translation), whereas
            # the canonical BAS skeletal mesh and its working meditation clips
            # keep that bridge node at unit scale.  Normalize just this bridge
            # node; all authored combat rotations below it remain unchanged.
            cutscene.location = cutscene.location * 0.01
            compensation = args.unreal_root_scale_compensation
            cutscene.scale = (compensation, compensation, compensation)
        cutscene = target.pose.bones.get("cutscenedummy")
        if cutscene is not None:
            cutscene_translation_lengths.append(cutscene.location.length)
            cutscene_scale = cutscene.scale
            cutscene_scale_errors.append(
                max(
                    abs(cutscene_scale.x - args.unreal_root_scale_compensation),
                    abs(cutscene_scale.y - args.unreal_root_scale_compensation),
                    abs(cutscene_scale.z - args.unreal_root_scale_compensation),
                )
            )
        bpy.context.view_layer.update()
        if args.head_mode == "inherit-parent":
            # Canonical BAS rigid-head bones are already parented to
            # cutscenedummy.  Their skinned vertices use bind transforms at
            # the armature origin, so rebaking legacy object-world channels
            # applies cutscenedummy twice and launches the head during combat.
            # Identity local tracks preserve the bind offsets while inheriting
            # the animated parent exactly, matching the working meditation
            # contract.
            for bone_name in head_bones:
                target.pose.bones[bone_name].matrix_basis = Matrix.Identity(4)
        else:
            for bone_name, world_matrix in head_world.items():
                target.pose.bones[bone_name].matrix = (
                    source_inverse
                    @ world_matrix
                    @ source_rest_world[bone_name].inverted_safe()
                    @ source.matrix_world
                )
        bpy.context.view_layer.update()
        for bone_name in head_bones:
            pose_matrix = target.pose.bones[bone_name].matrix
            head_translation_lengths.append(pose_matrix.translation.length)
            head_basis_translation_lengths.append(
                target.pose.bones[bone_name].matrix_basis.translation.length
            )
            # Validate each rigid head bone's authored local scale.  When the
            # Unreal root compensation is 0.01, the component-space pose scale
            # intentionally inherits 0.01 here and is restored to 1.0 by the
            # FBX exporter's 100x unit conversion.
            scale = target.pose.bones[bone_name].matrix_basis.to_scale()
            head_scale_errors.append(max(abs(scale.x - 1.0), abs(scale.y - 1.0), abs(scale.z - 1.0)))
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
            pose_bone.keyframe_insert(
                "rotation_quaternion", frame=frame, group=pose_bone.name
            )
            pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)

    output = args.output_dir / f"A_{args.utc}_{clip}_HeadFixed.fbx"
    reference_rest = {
        bone.name: bone.matrix_local.copy()
        for bone in target.data.bones
    }
    _export_animation(output, target, baked)
    exported_objects = _import_fbx(output)
    exported_armature = _armature(exported_objects)
    shared_rest_bones = sorted(set(reference_rest) & set(exported_armature.data.bones.keys()))
    rest_translation_errors = [
        (
            reference_rest[name].translation
            - exported_armature.data.bones[name].matrix_local.translation
        ).length
        for name in shared_rest_bones
    ]
    return {
        "clip": clip,
        "source": str(source_path),
        "output": str(output),
        "frames": [start, end],
        "frame_count": end - start + 1,
        "body_bones_baked": len(body_bones),
        "missing_non_deform_body_bones": missing_body,
        "head_bones_baked": len(head_bones),
        "head_bones": head_bones,
        "head_mode": args.head_mode,
        "cutscenedummy_normalized": args.normalize_cutscenedummy,
        "unreal_root_scale_compensation": args.unreal_root_scale_compensation,
        "cutscenedummy_translation_max_m": max(cutscene_translation_lengths, default=0.0),
        "cutscenedummy_scale_error_max": max(cutscene_scale_errors, default=0.0),
        "head_translation_max_m": max(head_translation_lengths, default=0.0),
        "head_basis_translation_max_m": max(head_basis_translation_lengths, default=0.0),
        "head_scale_error_max": max(head_scale_errors, default=0.0),
        "export_rest_bones_compared": len(shared_rest_bones),
        "export_rest_translation_error_max_m": max(rest_translation_errors, default=float("inf")),
        "bytes": output.stat().st_size,
    }


def main() -> int:
    args = _args()
    rows = [_convert_one(args, clip) for clip in args.clips]
    payload = {
        "schema": "ghoststudio.enclave.duel-headfixed-fbx.v1",
        "utc": args.utc,
        "canonical_base": str(args.base_fbx),
        "clips": rows,
        "passed": all(
            row["body_bones_baked"] >= 40
            and row["head_bones_baked"] >= 1
            and row["head_translation_max_m"] < 5.0
            and row["export_rest_bones_compared"] >= 40
            and row["export_rest_translation_error_max_m"] < 0.001
            and (
                row["head_mode"] != "inherit-parent"
                or row["head_basis_translation_max_m"] < 0.001
            )
            and (
                not row["cutscenedummy_normalized"]
                or (
                    row["cutscenedummy_scale_error_max"] < 0.001
                    and row["cutscenedummy_translation_max_m"] < 0.01
                )
            )
            and row["head_scale_error_max"] < 0.05
            and row["bytes"] > 100_000
            for row in rows
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MCPSTUDIO_DUEL_HEADFIX=" + json.dumps(payload))
    if not payload["passed"]:
        raise RuntimeError(f"Duel head-fix acceptance failed: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
