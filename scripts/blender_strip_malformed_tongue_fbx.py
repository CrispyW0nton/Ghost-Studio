"""Strip malformed tongue faces from a verified normalized character FBX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _reset() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _strip_tongue_faces() -> list[str]:
    tongues = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.casefold() == "tongue"
    ]
    for tongue in tongues:
        editable = bmesh.new()
        editable.from_mesh(tongue.data)
        bmesh.ops.delete(editable, geom=list(editable.faces), context="FACES_ONLY")
        editable.to_mesh(tongue.data)
        editable.free()
        tongue.data.update()
    return sorted(obj.name for obj in tongues)


def _flip_uv_v_for_fbx_interchange() -> list[str]:
    converted: list[str] = []
    for mesh in bpy.data.meshes:
        if not mesh.uv_layers:
            continue
        for layer in mesh.uv_layers:
            for loop in layer.data:
                loop.uv.y = 1.0 - float(loop.uv.y)
        converted.append(mesh.name)
    return sorted(converted)


def main() -> int:
    args = _args()
    _reset()
    bpy.ops.import_scene.fbx(
        filepath=str(args.input.resolve()),
        automatic_bone_orientation=False,
    )
    stripped = _strip_tongue_faces()
    if not stripped:
        raise RuntimeError("No tongue mesh was found in the normalized FBX")
    uv_converted = _flip_uv_v_for_fbx_interchange()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(args.output.resolve()),
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
    if not args.output.is_file() or args.output.stat().st_size < 1024:
        raise RuntimeError("Tongue-cleaned FBX export failed")
    print(json.dumps({
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "stripped_meshes": stripped,
        "uv_origin_converted_meshes": uv_converted,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
