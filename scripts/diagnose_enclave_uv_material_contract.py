"""Compare KOTOR source UV/material facts with a composed BAS GLB.

This is a deliberately narrow visual-export diagnostic.  It checks exact UV0
values (not aggregate sums), material-to-texture bindings, and composed head
attachment facts for one enclave character at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mcp.start_kotormcp_stdio import _python_roots  # noqa: E402

for _root in reversed(_python_roots(ROOT)):
    _text = str(_root)
    if _root.exists() and _text not in sys.path:
        sys.path.insert(0, _text)

from kotormcp.tools.ghostrigger import (  # noqa: E402
    _apply_primary_skin_texture,
    _compose_bas_character,
)
from kotormcp.tools.ghostrigger_tools import _load_ghostrigger_model  # noqa: E402
from pygltflib import GLTF2  # noqa: E402


_COMPONENT_FORMAT = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", required=True)
    parser.add_argument("--head", default="")
    parser.add_argument("--body-texture", default="")
    parser.add_argument("--head-texture", default="")
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _all_nodes(model: Any) -> list[Any]:
    method = getattr(model, "all_nodes", None)
    return list(method()) if callable(method) else list(getattr(model, "nodes", []) or [])


def _mesh_nodes(model: Any) -> list[Any]:
    return [
        node
        for node in _all_nodes(model)
        if list(getattr(node, "vertices", []) or [])
        and list(getattr(node, "faces", []) or [])
        and not bool(getattr(node, "_gr_bas_geometry_replaced_by_attachment", False))
    ]


def _accessor_values(gltf: GLTF2, accessor_index: int) -> list[tuple[float | int, ...]]:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    fmt, component_size = _COMPONENT_FORMAT[accessor.componentType]
    width = _TYPE_COMPONENTS[accessor.type]
    packed_size = component_size * width
    stride = int(view.byteStride or packed_size)
    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    return [
        struct.unpack_from("<" + fmt * width, blob, offset + row * stride)
        for row in range(accessor.count)
    ]


def _max_uv_error(actual: list[tuple[float, float]], expected: list[tuple[float, float]]) -> float:
    if len(actual) != len(expected):
        return math.inf
    return max(
        (max(abs(float(a[0]) - float(e[0])), abs(float(a[1]) - float(e[1]))) for a, e in zip(actual, expected)),
        default=0.0,
    )


def _texture_for_material(gltf: GLTF2, material_index: int | None) -> dict[str, Any]:
    if material_index is None:
        return {"material": "", "image": "", "uri_prefix": ""}
    material = gltf.materials[material_index]
    texture_info = getattr(getattr(material, "pbrMetallicRoughness", None), "baseColorTexture", None)
    if texture_info is None:
        return {"material": str(material.name or ""), "image": "", "uri_prefix": ""}
    texture = gltf.textures[texture_info.index]
    image = gltf.images[texture.source]
    return {
        "material": str(material.name or ""),
        "image": str(image.name or ""),
        "uri_prefix": str(image.uri or "")[:32],
    }


def main() -> int:
    args = _args()
    body = _load_ghostrigger_model("K1", args.body)
    _apply_primary_skin_texture(body, args.body_texture)
    composed = body
    bas_report: dict[str, Any] = {}
    if args.head:
        head = _load_ghostrigger_model("K1", args.head)
        _apply_primary_skin_texture(head, args.head_texture)
        composed, bas_report = _compose_bas_character(body, head, name=args.glb.stem)

    gltf = GLTF2().load_binary(str(args.glb))
    source_by_name: dict[str, list[Any]] = defaultdict(list)
    for node in _mesh_nodes(composed):
        source_by_name[str(getattr(node, "name", "") or "").casefold()].append(node)

    rows: list[dict[str, Any]] = []
    occurrence: dict[str, int] = defaultdict(int)
    for gltf_node in gltf.nodes:
        if gltf_node.mesh is None:
            continue
        mesh = gltf.meshes[gltf_node.mesh]
        key = str(mesh.name or gltf_node.name or "").casefold()
        slot = occurrence[key]
        occurrence[key] += 1
        candidates = source_by_name.get(key, [])
        source = candidates[slot] if slot < len(candidates) else None
        primitive = mesh.primitives[0]
        actual_uv = (
            _accessor_values(gltf, primitive.attributes.TEXCOORD_0)
            if primitive.attributes.TEXCOORD_0 is not None
            else []
        )
        source_uv = list(getattr(source, "uvs", []) or []) if source is not None else []
        source_faces = list(getattr(source, "faces", []) or []) if source is not None else []
        source_face_uvs = list(getattr(source, "face_uvs", []) or []) if source is not None else []
        face_uv_mismatch_count = 0
        if len(source_face_uvs) == len(source_faces):
            for face, face_uv in zip(source_faces, source_face_uvs):
                if tuple(int(value) for value in face) != tuple(int(value) for value in face_uv):
                    face_uv_mismatch_count += 1
        direct = [(float(uv[0]), float(uv[1])) for uv in source_uv]
        flipped = [(float(uv[0]), 1.0 - float(uv[1])) for uv in source_uv]
        texture = str(
            getattr(source, "texture_clean", "")
            or getattr(source, "texture", "")
            or ""
        ) if source is not None else ""
        rows.append(
            {
                "node": str(mesh.name or gltf_node.name or ""),
                "occurrence": slot,
                "source_found": source is not None,
                "source_texture": texture,
                "source_uv_count": len(source_uv),
                "source_face_uv_count": len(source_face_uvs),
                "face_uv_mismatch_count": face_uv_mismatch_count,
                "glb_uv_count": len(actual_uv),
                "max_error_direct": _max_uv_error(actual_uv, direct),
                "max_error_v_flipped": _max_uv_error(actual_uv, flipped),
                "material_binding": _texture_for_material(gltf, primitive.material),
                "is_skin": bool(getattr(source, "is_skin", False)) if source is not None else False,
                "attachment_slot": str(getattr(source, "_gr_bas_attachment_slot", "") or "") if source is not None else "",
            }
        )

    report = {
        "schema": "ghoststudio.enclave-uv-material-contract.v1",
        "body": args.body,
        "head": args.head,
        "body_texture_override": args.body_texture,
        "head_texture_override": args.head_texture,
        "glb": str(args.glb.resolve()),
        "bas_report": bas_report,
        "source_mesh_count": sum(len(items) for items in source_by_name.values()),
        "glb_mesh_count": len(rows),
        "rows": rows,
        "failures": [
            row
            for row in rows
            if not row["source_found"]
            or row["source_uv_count"] != row["glb_uv_count"]
            or row["max_error_v_flipped"] > 1.0e-6
            or row["source_texture"].casefold() != row["material_binding"]["material"].casefold()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "source_mesh_count": report["source_mesh_count"],
        "glb_mesh_count": report["glb_mesh_count"],
        "failure_count": len(report["failures"]),
    }, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
