"""Export a stock KOTOR model as a hierarchy-baked OBJ for visual audits.

This deliberately bypasses the animated glTF/FBX hierarchy.  Comparing this
static reference with an animated interchange export isolates whether a visual
registration failure originates in source model loading or in hierarchy/skin
serialization.
"""

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
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def _vec(values) -> list[float]:
    return [round(float(value), 8) for value in list(values)[:3]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=("k1", "k2"), required=True)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--resref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    _bootstrap()
    from src.converters.mesh_converter import OBJExporter
    from src.core.assets.resource_manager import ResourceManager

    manager = ResourceManager()
    configured = (
        manager.set_k1_dir(str(args.game_root))
        if args.game == "k1"
        else manager.set_k2_dir(str(args.game_root))
    )
    if not configured:
        raise RuntimeError(f"Could not index {args.game} installation: {args.game_root}")
    model = manager.load_model_strict(args.resref, args.game.upper())
    if model is None:
        raise RuntimeError(f"Could not load {args.resref} from {args.game}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = OBJExporter().export(
        model,
        str(args.output),
        tex_cache=None,
        export_rigging=False,
    )
    if not args.output.is_file() or args.output.stat().st_size < 1024:
        raise RuntimeError(f"Static OBJ export failed: {result!r}")

    facial_terms = ("head", "eye", "lid", "teeth", "tooth", "tongue", "gum", "jaw")
    rows = []
    for node in model.all_nodes():
        name = str(getattr(node, "name", "") or "")
        if not any(term in name.casefold() for term in facial_terms):
            continue
        world_position, world_rotation = OBJExporter._node_bind_world_transform(node)
        world_vertices = OBJExporter._node_bind_world_verts(node)
        rows.append(
            {
                "name": name,
                "parent": str(getattr(getattr(node, "parent", None), "name", "") or ""),
                "is_skin": bool(getattr(node, "is_skin", False)),
                "render": bool(getattr(node, "render", True)),
                "texture": str(getattr(node, "texture", "") or ""),
                "local_position": _vec(getattr(node, "position", (0.0, 0.0, 0.0))),
                "world_position": _vec(world_position),
                "world_rotation": [round(float(value), 8) for value in world_rotation],
                "vertex_count": len(list(getattr(node, "vertices", ()) or ())),
                "uv_count": len(list(getattr(node, "uvs", ()) or ())),
                "face_count": len(list(getattr(node, "faces", ()) or ())),
                "world_vertices": [
                    [round(float(value), 8) for value in vertex]
                    for vertex in world_vertices
                ],
                "world_bounds_min": [
                    round(min(float(vertex[axis]) for vertex in world_vertices), 8)
                    if world_vertices else 0.0
                    for axis in range(3)
                ],
                "world_bounds_max": [
                    round(max(float(vertex[axis]) for vertex in world_vertices), 8)
                    if world_vertices else 0.0
                    for axis in range(3)
                ],
            }
        )

    receipt_path = args.receipt or args.output.with_suffix(".audit.json")
    payload = {
        "schema": "ghostrigger.kotor-static-obj-audit.v1",
        "game": args.game,
        "resref": args.resref,
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "nodes": rows,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "receipt": str(receipt_path), "nodes": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
