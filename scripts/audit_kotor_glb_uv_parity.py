#!/usr/bin/env python3
"""Compare stock KOTOR MDL UV0 data with a Ghost Studio glTF/GLB export.

This diagnostic is intentionally read-only.  It reports whether the exported
UVs and triangle indices still match the stock MDL without a destructive
pre-flip at the glTF interchange boundary.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from pygltflib import GLTF2
from pykotor.extract.installation import Installation
from pykotor.resource.formats.mdl import read_mdl
from pykotor.resource.type import ResourceType


_COMPONENT_FORMAT = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
_TYPE_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _accessor_rows(gltf: GLTF2, accessor_index: int) -> list[tuple[Any, ...]]:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    if blob is None:
        raise ValueError("The audit currently requires a binary .glb file.")
    component_format = _COMPONENT_FORMAT[int(accessor.componentType)]
    width = _TYPE_WIDTH[str(accessor.type)]
    component_size = struct.calcsize("<" + component_format)
    row_size = component_size * width
    stride = int(view.byteStride or row_size)
    start = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    unpack = struct.Struct("<" + component_format * width).unpack_from
    return [unpack(blob, start + index * stride) for index in range(int(accessor.count))]


def _stock_meshes(install: Installation, resref: str) -> dict[str, Any]:
    mdl_resource = install.resource(resref, ResourceType.MDL)
    mdx_resource = install.resource(resref, ResourceType.MDX)
    if mdl_resource is None or mdx_resource is None:
        raise FileNotFoundError(f"Could not load {resref}.mdl/.mdx from the installation.")
    model = read_mdl(mdl_resource.data, source_ext=mdx_resource.data)
    return {
        str(node.name).casefold(): node.mesh
        for node in model.all_nodes()
        if node.mesh is not None and node.mesh.vertex_positions and node.mesh.faces
    }


def _mapped_face_uv(face: Any) -> tuple[int, int, int]:
    verts = (int(face.v1), int(face.v2), int(face.v3))
    tverts = (int(face.t1), int(face.t2), int(face.t3))
    return tuple(tvert if tvert >= 0 else vert for tvert, vert in zip(tverts, verts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, type=Path)
    parser.add_argument("--resref", required=True)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stock = _stock_meshes(Installation(args.game), args.resref)
    gltf = GLTF2().load_binary(str(args.glb))
    report: dict[str, Any] = {
        "game": str(args.game),
        "resref": args.resref,
        "glb": str(args.glb),
        "meshes": [],
    }

    for mesh in gltf.meshes or []:
        name = str(mesh.name or "")
        stock_mesh = stock.get(name.casefold())
        if stock_mesh is None or not mesh.primitives:
            continue
        primitive = mesh.primitives[0]
        texcoord_index = getattr(primitive.attributes, "TEXCOORD_0", None)
        if texcoord_index is None:
            continue
        exported_uvs = _accessor_rows(gltf, int(texcoord_index))
        exported_indices = [int(row[0]) for row in _accessor_rows(gltf, int(primitive.indices))]
        stock_uvs = [(float(uv.x), float(uv.y)) for uv in (stock_mesh.vertex_uv1 or [])]
        stock_indices = [
            int(value)
            for face in stock_mesh.faces
            for value in (int(face.v1), int(face.v2), int(face.v3))
        ]
        expected_uvs = stock_uvs
        uv_error = [
            max(abs(float(actual[0]) - expected[0]), abs(float(actual[1]) - expected[1]))
            for actual, expected in zip(exported_uvs, expected_uvs)
        ]
        mapped_face_uvs = [_mapped_face_uv(face) for face in stock_mesh.faces]
        geometry_face_indices = [
            (int(face.v1), int(face.v2), int(face.v3)) for face in stock_mesh.faces
        ]
        report["meshes"].append(
            {
                "name": name,
                "stock_vertex_count": len(stock_mesh.vertex_positions),
                "stock_uv_count": len(stock_uvs),
                "exported_uv_count": len(exported_uvs),
                "stock_index_count": len(stock_indices),
                "exported_index_count": len(exported_indices),
                "uv_max_error": max(uv_error, default=None),
                "uv_match": len(exported_uvs) == len(expected_uvs)
                and max(uv_error, default=0.0) <= 1e-6,
                "index_match": exported_indices == stock_indices,
                "has_distinct_face_uv_indices": mapped_face_uvs != geometry_face_indices,
            }
        )

    report["all_matched"] = bool(report["meshes"]) and all(
        row["uv_match"] and row["index_match"] for row in report["meshes"]
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["all_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
