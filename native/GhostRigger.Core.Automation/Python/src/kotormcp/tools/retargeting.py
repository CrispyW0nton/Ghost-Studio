"""Animation retargeting MCP tools.

These tools expose GhostRigger's existing KotOR and Unreal/Quinn retargeting
helpers through read-only MCP contracts first.  The initial surface is
intentionally conservative: inspect skeletons, list animation availability, and
build mapping reports.  Actual animation baking/export can build on these
contracts once the mapping reports are stable.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kotormcp.utils import json_content


_NULL_REFS = {"", "null", "none"}
_MAX_MANUAL_MAPPING = 512
_MAX_BONE_NAME = 128
_MIN_FPS = 1.0
_MAX_FPS = 120.0
_DAY4_OUTPUT_NAME = "pmbam__g1a1__to__quinn_aligned.fbx"
_DAY4_GATES = (
    "bone_count_match",
    "vertex_count_match",
    "frame_count_match",
    "no_leaf_bones_added",
    "axis_system_match",
)


def _error(payload: Dict[str, Any], max_chars: int = 32_000) -> Dict[str, Any]:
    """Return a JSON error payload flagged as an MCP tool error."""
    return {**json_content(payload, max_chars=max_chars), "isError": True}


def _manual_mapping(value: Any) -> Optional[Dict[str, str]]:
    """Validate the optional manual bone-name overrides."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("manual_mapping must be an object of bone-name strings")
    if len(value) > _MAX_MANUAL_MAPPING:
        raise ValueError(f"manual_mapping exceeds {_MAX_MANUAL_MAPPING} entries")
    for key, target in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(target, str)
            or not key
            or len(key) > _MAX_BONE_NAME
            or len(target) > _MAX_BONE_NAME
        ):
            raise ValueError(
                f"manual_mapping keys and values must be non-empty strings of at most {_MAX_BONE_NAME} characters"
            )
    return dict(value)


def get_tools() -> List[Dict[str, Any]]:
    """Return retargeting tool definitions."""
    return [
        {
            "name": "ghostrigger_get_retarget_skeleton_info",
            "description": (
                "Return skeleton/node summary for an Aurora/KotOR model or the bundled "
                "Unreal Quinn target skeleton. Read-only; useful before retargeting."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_type": {
                        "type": "string",
                        "enum": ["aurora_model", "unreal_quinn"],
                        "default": "aurora_model",
                    },
                    "game": {"type": "string", "description": "k1/k2 for Aurora models"},
                    "resref": {
                        "type": "string",
                        "description": "Aurora MDL resref or absolute path when target_type=aurora_model",
                    },
                    "game_path": {"type": "string", "description": "Optional install path override"},
                    "skeleton_dir": {
                        "type": "string",
                        "description": "Optional directory containing SKM_Quinn_Simple skeleton assets",
                    },
                    "include_nodes": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "ghostrigger_build_retarget_map",
            "description": (
                "Build a deterministic source-to-target bone mapping report using "
                "GhostRigger's existing retargeting rules. Supports Aurora/KotOR "
                "models and the bundled Unreal Quinn target skeleton. Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["aurora_model", "unreal_quinn"],
                        "default": "aurora_model",
                    },
                    "target_type": {
                        "type": "string",
                        "enum": ["aurora_model", "unreal_quinn"],
                        "default": "unreal_quinn",
                    },
                    "source_game": {"type": "string"},
                    "source_resref": {"type": "string"},
                    "source_game_path": {"type": "string"},
                    "target_game": {"type": "string"},
                    "target_resref": {"type": "string"},
                    "target_game_path": {"type": "string"},
                    "skeleton_dir": {
                        "type": "string",
                        "description": "Optional directory containing SKM_Quinn_Simple skeleton assets",
                    },
                    "manual_mapping": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "maxLength": 128},
                        "maxProperties": 512,
                        "description": "Optional source-name to target-name overrides (at most 512 pairs, names at most 128 characters)",
                    },
                },
            },
        },
        {
            "name": "ghostrigger_list_retarget_animations",
            "description": (
                "List local and inherited Aurora/KotOR animations for a model, walking "
                "the supermodel chain with the same first-match-wins rule used by "
                "GhostRigger's animation resolver. Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "k1/k2"},
                    "resref": {"type": "string", "description": "Aurora MDL resref or absolute path"},
                    "game_path": {"type": "string", "description": "Optional install path override"},
                    "include_chain": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "default": 500},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "ghostrigger_export_unity_fbx",
            "description": (
                "Export a retargeted Unity-ready FBX through the Day 4 pipeline. "
                "Sprint 1 currently supports pmbam + g1a1 -> aligned Quinn. "
                "Returns paths and validation summaries only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_model_resref": {"type": "string", "default": "pmbam"},
                    "target_skeleton_id": {"type": "string", "default": "ue5_quinn"},
                    "clip_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["g1a1"],
                    },
                    "output_dir": {"type": "string"},
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "False refuses to replace an existing FBX, manifest, intermediate or roundtrip file",
                    },
                    "options_overrides": {
                        "type": "object",
                        "properties": {
                            "fps": {"type": "number", "minimum": 1, "maximum": 120},
                        },
                        "additionalProperties": False,
                    },
                },
            },
        },
    ]


def _load_aurora_model(resref: str, game: Optional[str], game_path: Optional[str] = None) -> Tuple[Any, str]:
    from kotormcp.adapters import (  # noqa: PLC0415
        CompositeModelLocator,
        FileSystemModelLocator,
        InstallationModelLocator,
        MDLBinaryParserAdapter,
        get_default_registry,
    )

    registry = get_default_registry()
    locator = CompositeModelLocator([
        FileSystemModelLocator(),
        InstallationModelLocator(registry),
    ])
    parser = MDLBinaryParserAdapter()
    path_label, mdl_bytes, mdx_bytes = locator.locate(resref, game, game_path)
    return parser.parse(mdl_bytes, mdx_bytes, path_label), path_label


def _load_quinn_model(skeleton_dir: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    from src.unreal.quinn import load_quinn_skeleton_asset, unreal_skeleton_model  # noqa: PLC0415

    asset = load_quinn_skeleton_asset(Path(skeleton_dir) if skeleton_dir else None)
    model = unreal_skeleton_model(asset)
    return model, {
        "name": asset.name,
        "source": asset.source,
        "bone_count": asset.bone_count,
        "bone_map_path": str(asset.bone_map_path),
        "fbx_path": str(asset.fbx_path),
        "texture_count": len(asset.texture_paths),
    }


def _model_nodes(model: Any) -> list:
    return list(model.all_nodes()) if callable(getattr(model, "all_nodes", None)) else []


def _node_name(node: Any) -> str:
    return str(getattr(node, "name", "") or "")


def _node_rows(model: Any, *, include_nodes: bool) -> List[Dict[str, Any]]:
    if not include_nodes:
        return []
    rows: List[Dict[str, Any]] = []
    for node in _model_nodes(model):
        parent = getattr(node, "parent", None)
        children = list(getattr(node, "children", []) or [])
        rows.append({
            "name": _node_name(node),
            "parent": _node_name(parent) if parent is not None else "",
            "children": [_node_name(child) for child in children],
            "flags": int(getattr(node, "flags", 0) or 0),
            "is_mesh": bool(getattr(node, "is_mesh", False)),
            "is_skin": bool(getattr(node, "is_skin", False)),
            "is_dummy": bool(getattr(node, "is_dummy", False)),
            "position": [float(v) for v in (getattr(node, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))],
            "rotation_xyzw": [float(v) for v in (getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))],
        })
    return rows


def _skeleton_summary(
    model: Any,
    *,
    source: str,
    skeleton_type: str,
    include_nodes: bool = True,
    asset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    nodes = _model_nodes(model)
    mesh_nodes = list(model.mesh_nodes()) if callable(getattr(model, "mesh_nodes", None)) else []
    bone_nodes = list(model.bone_nodes()) if callable(getattr(model, "bone_nodes", None)) else []
    return {
        "name": str(getattr(model, "name", "") or ""),
        "source": source,
        "skeleton_type": skeleton_type,
        "supermodel": str(getattr(model, "supermodel", "") or ""),
        "node_count": len(nodes),
        "mesh_node_count": len(mesh_nodes),
        "bone_node_count": len(bone_nodes),
        "animation_count": len(getattr(model, "animations", []) or []),
        "animations": [str(getattr(anim, "name", "") or "") for anim in (getattr(model, "animations", []) or [])],
        "asset": asset or {},
        "nodes": _node_rows(model, include_nodes=include_nodes),
    }


async def handle_get_retarget_skeleton_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    target_type = str(arguments.get("target_type") or "aurora_model")
    include_nodes = bool(arguments.get("include_nodes", True))
    try:
        if target_type == "unreal_quinn":
            model, asset = _load_quinn_model(arguments.get("skeleton_dir"))
            return json_content(_skeleton_summary(
                model,
                source=asset.get("source", "SKM_Quinn_Simple"),
                skeleton_type="unreal_quinn",
                include_nodes=include_nodes,
                asset=asset,
            ), max_chars=120_000)

        resref = str(arguments.get("resref") or "").strip()
        if not resref:
            return json_content({"error": "resref is required for target_type=aurora_model."})
        model, path_label = _load_aurora_model(
            resref,
            arguments.get("game"),
            arguments.get("game_path"),
        )
        return json_content(_skeleton_summary(
            model,
            source=path_label,
            skeleton_type="aurora_model",
            include_nodes=include_nodes,
        ), max_chars=120_000)
    except Exception as exc:
        return json_content({"error": str(exc)})


def _load_mapping_model(prefix: str, arguments: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    model_type = str(arguments.get(f"{prefix}_type") or "aurora_model")
    if model_type == "unreal_quinn":
        model, asset = _load_quinn_model(arguments.get("skeleton_dir"))
        return model, {"type": "unreal_quinn", **asset}

    resref = str(arguments.get(f"{prefix}_resref") or "").strip()
    if not resref:
        raise ValueError(f"{prefix}_resref is required when {prefix}_type=aurora_model")
    model, path_label = _load_aurora_model(
        resref,
        arguments.get(f"{prefix}_game"),
        arguments.get(f"{prefix}_game_path"),
    )
    return model, {
        "type": "aurora_model",
        "resref": resref,
        "game": arguments.get(f"{prefix}_game") or "",
        "source": path_label,
    }


async def handle_build_retarget_map(arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        manual = _manual_mapping(arguments.get("manual_mapping"))
        source_model, source_meta = _load_mapping_model("source", arguments)
        target_model, target_meta = _load_mapping_model("target", arguments)

        if source_meta.get("type") == "unreal_quinn" or target_meta.get("type") == "unreal_quinn":
            from src.unreal.animation_retargeting import build_bone_map  # noqa: PLC0415
        else:
            from src.core.animation_retargeting import build_bone_map  # noqa: PLC0415

        report = build_bone_map(source_model, target_model, manual_mapping=manual)
        payload = asdict(report)
        payload["source"] = source_meta
        payload["target"] = target_meta
        payload["matched_count"] = report.matched_count
        if hasattr(report, "derived_count"):
            payload["derived_count"] = report.derived_count
        payload["coverage"] = {
            "source_mapped_ratio": (
                report.matched_count / max(1, report.matched_count + len(report.missing_source))
            ),
            "target_direct_or_derived_ratio": (
                (report.matched_count + len(getattr(report, "derived_target", ()) or ()))
                / max(
                    1,
                    report.matched_count
                    + len(getattr(report, "derived_target", ()) or ())
                    + len(report.missing_target),
                )
            ),
        }
        return json_content(payload, max_chars=120_000)
    except Exception as exc:
        return _error({"error": str(exc)})


def _animation_names(model: Any) -> List[str]:
    return [str(getattr(anim, "name", "") or "") for anim in (getattr(model, "animations", []) or [])]


async def handle_list_retarget_animations(arguments: Dict[str, Any]) -> Dict[str, Any]:
    resref = str(arguments.get("resref") or "").strip()
    game = arguments.get("game")
    include_chain = bool(arguments.get("include_chain", True))
    limit = int(arguments.get("limit") or 500)
    if not resref:
        return json_content({"error": "resref is required."})
    try:
        model, path_label = _load_aurora_model(resref, game, arguments.get("game_path"))
        chain: List[Dict[str, Any]] = []
        seen_anim_names: set[str] = set()
        visited: set[str] = set()
        current_model = model
        current_ref = resref
        while current_model is not None:
            current_key = str(current_ref or getattr(current_model, "name", "") or "").lower()
            if current_key in visited:
                chain.append({"resref": current_ref, "cycle": True})
                break
            visited.add(current_key)
            local_names = _animation_names(current_model)
            inherited_at_this_level = []
            for name in local_names:
                key = name.lower()
                if key not in seen_anim_names:
                    seen_anim_names.add(key)
                    inherited_at_this_level.append(name)
            chain.append({
                "resref": current_ref,
                "model_name": str(getattr(current_model, "name", "") or ""),
                "source": path_label if len(chain) == 0 else f"supermodel:{current_ref}",
                "supermodel": str(getattr(current_model, "supermodel", "") or ""),
                "local_animation_count": len(local_names),
                "new_animation_count": len(inherited_at_this_level),
                "new_animations": inherited_at_this_level[:limit],
            })
            if not include_chain:
                break
            next_ref = str(getattr(current_model, "supermodel", "") or "").strip()
            if next_ref.lower() in _NULL_REFS:
                break
            current_ref = next_ref
            current_model, _path = _load_aurora_model(current_ref, game, arguments.get("game_path"))
            path_label = _path

        all_names: List[str] = []
        seen = set()
        for entry in chain:
            for name in entry.get("new_animations", []):
                key = str(name).lower()
                if key not in seen:
                    seen.add(key)
                    all_names.append(str(name))
                if len(all_names) >= limit:
                    break
            if len(all_names) >= limit:
                break
        return json_content({
            "resref": resref,
            "game": game,
            "source": chain[0]["source"] if chain else "",
            "chain": chain,
            "available_animation_count": len(seen_anim_names),
            "available_animations": all_names,
            "truncated": len(all_names) < len(seen_anim_names),
        })
    except Exception as exc:
        return json_content({"error": str(exc)})


async def handle_export_unity_fbx(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Run the locked Day 4 pmbam+g1a1 Unity FBX export pipeline.

    Inputs are bounded: only the Day 4 source/target/clip triple, an fps in
    [1, 120], and an optional ``overwrite=False`` that refuses to replace any
    existing export file.  ``validation_status`` is derived from the Blender
    roundtrip gates rather than asserted.
    """

    try:
        source = str(arguments.get("source_model_resref") or "pmbam").strip().lower()
        target = str(arguments.get("target_skeleton_id") or "ue5_quinn").strip().lower()
        clips = arguments.get("clip_names") or ["g1a1"]
        if not isinstance(clips, list):
            return _error({"ok": False, "error": "clip_names must be an array."})
        clips = [str(clip).strip().lower() for clip in clips]
        if source != "pmbam" or target not in {"ue5_quinn", "quinn"} or clips != ["g1a1"]:
            return _error({
                "ok": False,
                "error": "Sprint 1 Day 4 exporter currently supports only source_model_resref='pmbam', target_skeleton_id='ue5_quinn', clip_names=['g1a1'].",
            })

        overrides = arguments.get("options_overrides")
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict) or set(overrides) - {"fps"}:
            return _error({"ok": False, "error": "options_overrides accepts only fps."})
        try:
            fps = float(overrides.get("fps", 60.0))
        except (TypeError, ValueError):
            return _error({"ok": False, "error": "fps must be a number."})
        if not (_MIN_FPS <= fps <= _MAX_FPS):  # also rejects NaN
            return _error({"ok": False, "error": f"fps must be between {_MIN_FPS:g} and {_MAX_FPS:g}."})

        output_dir = Path(arguments.get("output_dir") or "exports/retargets/day4")
        output_path = output_dir / _DAY4_OUTPUT_NAME
        planned = [
            output_path,
            output_path.with_suffix(".manifest.json"),
            output_path.with_suffix(".roundtrip.json"),
            output_path.with_name(f"{output_path.stem}_intermediate.json"),
        ]
        if arguments.get("overwrite", True) is False:
            existing = [str(path) for path in planned if path.exists()]
            if existing:
                return _error({"ok": False, "error": "Refusing to overwrite existing export files.", "existing": existing})

        from src.core.retargeting.fbx_exporter import export_day4_pmbam_g1a1  # noqa: PLC0415

        manifest = export_day4_pmbam_g1a1(output_path, fps=fps, log_dir=output_dir / "logs")
        payload = manifest.to_json_dict()
        metrics = dict(payload.get("roundtrip_metrics") or {})
        failed_gates = [gate for gate in _DAY4_GATES if metrics.get(gate) is not True]
        validation_status = "PASS" if metrics and not failed_gates else "FAIL"
        summary = {
            "vertex_count": metrics.get("vertex_count_observed"),
            "bone_count": metrics.get("bone_count_observed"),
            "clip_count": len(payload["clip_inventory"]),
            "fbx_sha256": payload["fbx_sha256"],
            "fps": fps,
            "roundtrip_metrics": metrics,
            "failed_gates": failed_gates,
        }
        result = {
            "ok": validation_status == "PASS",
            "fbx_path": payload["fbx_path"],
            "manifest_path": payload["manifest_path"],
            "intermediate_path": payload["intermediate_path"],
            "validation_status": validation_status,
            "summary": summary,
        }
        if validation_status != "PASS":
            return _error(result)
        return json_content(result, max_chars=32_000)
    except Exception as exc:
        import subprocess  # noqa: PLC0415

        if isinstance(exc, subprocess.TimeoutExpired):
            # The default message embeds the full Blender command line.
            return _error({"ok": False, "error": f"Blender subprocess timed out after {exc.timeout:g} s"})
        return _error({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
