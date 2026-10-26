"""Audit raw and BAS-composed KOTOR facial mesh bounds and skin metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _bounds(vertices) -> dict[str, list[float]]:
    rows = [tuple(float(value) for value in vertex[:3]) for vertex in vertices or []]
    if not rows:
        return {"minimum": [], "maximum": []}
    return {
        "minimum": [min(row[axis] for row in rows) for axis in range(3)],
        "maximum": [max(row[axis] for row in rows) for axis in range(3)],
    }


def _node_row(node) -> dict:
    skin_data = list(getattr(node, "skin_data", None) or [])
    used_bones: set[int] = set()
    for vertex in skin_data:
        for influence in list(getattr(vertex, "influences", None) or []):
            if float(getattr(influence, "weight", 0.0) or 0.0) > 1e-6:
                used_bones.add(int(getattr(influence, "bone_index", -1)))
    bone_map = list(getattr(node, "bone_map", None) or [])
    bone_node_indices = [
        int(value) for value in list(getattr(node, "bone_node_indices", None) or [])
    ]
    qbones = [
        [float(value) for value in row]
        for row in list(getattr(node, "qbone_list", None) or [])
    ]
    tbones = [
        [float(value) for value in row]
        for row in list(getattr(node, "tbone_list", None) or [])
    ]
    return {
        "name": str(getattr(node, "name", "") or ""),
        "parent": str(getattr(getattr(node, "parent", None), "name", "") or ""),
        "position": [float(value) for value in getattr(node, "position", (0, 0, 0))],
        "rotation": [float(value) for value in getattr(node, "rotation", (0, 0, 0, 1))],
        "vertex_count": len(list(getattr(node, "vertices", None) or [])),
        "bounds": _bounds(getattr(node, "vertices", None)),
        "is_skin": bool(getattr(node, "is_skin", False)),
        "skin_vertex_count": len(skin_data),
        "bone_map": [str(value or "") for value in bone_map],
        "bone_node_indices": bone_node_indices,
        "qbone_count": len(qbones),
        "tbone_count": len(tbones),
        "qbone_used": [
            qbones[index] if 0 <= index < len(qbones) else []
            for index in bone_node_indices
        ],
        "tbone_used": [
            tbones[index] if 0 <= index < len(tbones) else []
            for index in bone_node_indices
        ],
        "used_bone_slots": sorted(used_bones),
        "used_bones": [
            str(bone_map[index] or "") if 0 <= index < len(bone_map) else ""
            for index in sorted(used_bones)
        ],
        "vertices": [
            {
                "position": [float(value) for value in vertex[:3]],
                "influences": [
                    {
                        "slot": int(getattr(influence, "bone_index", -1)),
                        "bone": (
                            str(bone_map[int(getattr(influence, "bone_index", -1))] or "")
                            if 0 <= int(getattr(influence, "bone_index", -1)) < len(bone_map)
                            else ""
                        ),
                        "weight": float(getattr(influence, "weight", 0.0) or 0.0),
                    }
                    for influence in list(getattr(skin_data[index], "influences", None) or [])
                    if float(getattr(influence, "weight", 0.0) or 0.0) > 1e-6
                ] if index < len(skin_data) else [],
            }
            for index, vertex in enumerate(list(getattr(node, "vertices", None) or []))
        ],
        "render": bool(getattr(node, "render", True)),
        "texture": str(getattr(node, "texture", "") or ""),
    }


def _facial_rows(model) -> list[dict]:
    tokens = ("head", "eye", "lid", "teeth", "tongue", "gum", "jaw")
    return [
        _node_row(node)
        for node in model.all_nodes()
        if any(token in str(getattr(node, "name", "") or "").casefold() for token in tokens)
        and bool(getattr(node, "vertices", None))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _bootstrap()

    from kotormcp.tools.ghostrigger import _compose_bas_character, _get_services, _locate_and_parse

    services = _get_services()
    _, body = _locate_and_parse(args.body, "k1", str(args.game_root), services)
    _, head = _locate_and_parse(args.head, "k1", str(args.game_root), services)
    composed, report = _compose_bas_character(body, head, name="bas_inner_geometry_audit")
    payload = {
        "schema": "ghostrigger.kotor-bas-inner-geometry-audit.v1",
        "body": args.body,
        "head": args.head,
        "body_facial": _facial_rows(body),
        "head_facial": _facial_rows(head),
        "composed_facial": _facial_rows(composed),
        "composition_report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "head_nodes": len(payload["head_facial"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
