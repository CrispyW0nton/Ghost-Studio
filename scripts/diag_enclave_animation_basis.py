"""Inspect bind-versus-animation transform semantics for one BAS character.

This is a read-only diagnostic used by the enclave character export gate.  It
prints the primary body, resolved supermodel, composed BAS, and selected
animation controller values for the same named nodes so transform-basis bugs
can be isolated before Unreal import.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _node_map(model) -> dict[str, list]:
    rows: dict[str, list] = {}
    for node in model.all_nodes():
        rows.setdefault(str(node.name or "").casefold(), []).append(node)
    return rows


def _animation_node_map(animation) -> dict[str, list]:
    rows: dict[str, list] = {}
    for node in animation.nodes or []:
        rows.setdefault(str(node.name or "").casefold(), []).append(node)
    return rows


def _controller(node, controller_type: int):
    for controller in node.controllers or []:
        if int(controller.get("type", -1)) == controller_type:
            return controller
    return None


def _quat_dot(left, right) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _quat_angle_degrees(left, right) -> float:
    left_mag = math.sqrt(max(1.0e-30, _quat_dot(left, left)))
    right_mag = math.sqrt(max(1.0e-30, _quat_dot(right, right)))
    dot = abs(_quat_dot(left, right) / (left_mag * right_mag))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _sample_rows(controller) -> list[dict]:
    if not controller:
        return []
    times = list(controller.get("times", []) or [])
    values = list(controller.get("values", []) or [])
    if not values:
        return []
    indices = sorted({0, len(values) // 2, len(values) - 1})
    return [
        {
            "index": index,
            "time": float(times[index]) if index < len(times) else None,
            "value": [float(value) for value in values[index]],
        }
        for index in indices
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--head", default="")
    parser.add_argument("--animation", default="walk")
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _bootstrap()

    from kotormcp.tools.ghostrigger import (
        _compose_bas_character,
        _get_services,
        _locate_and_parse,
        _prepare_selected_fbx_animations,
        _resolve_fbx_base_skeleton,
    )

    services = _get_services()
    _body_path, body = _locate_and_parse(args.body, "k1", str(args.game_root), services)
    base = _resolve_fbx_base_skeleton(body, "k1", str(args.game_root), services)
    head = None
    composed = body
    if args.head:
        _head_path, head = _locate_and_parse(args.head, "k1", str(args.game_root), services)
        composed, _report = _compose_bas_character(body, head, name=f"diag_{args.body}")
        if base is not None:
            setattr(composed, "_gr_fbx_base_skeleton_model", base)
    prepared = _prepare_selected_fbx_animations(
        composed,
        {"animation_names": [args.animation]},
        game="k1",
        game_path=str(args.game_root),
        services=services,
        supplemental_models=(head,) if head is not None else (),
    )
    animation = next(
        item for item in prepared.animations
        if str(item.name or "").casefold() == args.animation.casefold()
    )

    names = args.node or [
        "rootdummy", "pelvis", "torso_g", "lcollar_g", "lbicep_g",
        "lforearm_g", "lhand", "rcollar_g", "rbicep_g", "rforearm_g", "rhand",
    ]
    models = {
        "body": _node_map(body),
        "base": _node_map(base) if base is not None else {},
        "composed": _node_map(composed),
        "prepared": _node_map(prepared),
    }
    animation_nodes = _animation_node_map(animation)
    report = {
        "body": args.body,
        "head": args.head,
        "body_supermodel": str(getattr(body, "supermodel", "") or ""),
        "base_name": str(getattr(base, "name", "") or "") if base is not None else "",
        "animation": args.animation,
        "animation_source": dict(getattr(prepared, "_gr_fbx_animation_selection", {}) or {}),
        "skins": [],
        "nodes": [],
    }
    prepared_nodes = list(prepared.all_nodes())
    for skin_node in prepared_nodes:
        if not bool(getattr(skin_node, "is_skin", False)):
            continue
        report["skins"].append({
            "name": str(getattr(skin_node, "name", "") or ""),
            "dfs_index": prepared_nodes.index(skin_node),
            "bone_map": list(getattr(skin_node, "bone_map", []) or []),
            "bone_node_indices": list(getattr(skin_node, "bone_node_indices", []) or []),
            "qbone_count": len(getattr(skin_node, "qbone_list", []) or []),
            "tbone_count": len(getattr(skin_node, "tbone_list", []) or []),
            "original_dfs_index": getattr(skin_node, "_gr_original_dfs_index", None),
        })
    for authored_name in names:
        key = authored_name.casefold()
        row = {"name": authored_name, "models": {}, "animation_occurrences": []}
        for label, mapping in models.items():
            row["models"][label] = [
                {
                    "position": [float(value) for value in node.position],
                    "rotation": [float(value) for value in node.rotation],
                    "parent": str(getattr(getattr(node, "parent", None), "name", "") or ""),
                }
                for node in mapping.get(key, [])
            ]
        for occurrence, anim_node in enumerate(animation_nodes.get(key, [])):
            orientation = _controller(anim_node, 20)
            bind_rows = row["models"]["prepared"]
            bind_rotation = bind_rows[occurrence]["rotation"] if occurrence < len(bind_rows) else None
            samples = _sample_rows(orientation)
            for sample in samples:
                if bind_rotation is not None:
                    sample["angle_from_bind_degrees"] = _quat_angle_degrees(
                        bind_rotation,
                        sample["value"],
                    )
            row["animation_occurrences"].append({
                "occurrence": occurrence,
                "orientation": samples,
                "position": _sample_rows(_controller(anim_node, 8)),
            })
        report["nodes"].append(row)

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
