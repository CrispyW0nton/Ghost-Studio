"""GhostRigger FBX sidecar metadata for KOTOR/DCC handoff."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from src.core.geometry.model_data import CharacterMode, detect_character_mode


SCHEMA_VERSION = 1

KOTOR_HOOK_ROLES: dict[str, str] = {
    "head_g": "head_attachment_geometry",
    "headhook": "head_attachment_socket",
    "maskhook": "mask_attachment_socket",
    "gogglehook": "goggle_attachment_socket",
    "lhand_g": "left_hand_attachment",
    "rhand_g": "right_hand_attachment",
    "lhand": "left_hand_bone",
    "rhand": "right_hand_bone",
    "camerahook": "camera_anchor",
    "foot_l": "left_foot_alignment",
    "foot_r": "right_foot_alignment",
    "talkdummy": "facial_talk_anchor",
}


def sidecar_path_for_fbx(fbx_path: str | Path) -> Path:
    """Return the canonical GhostRigger metadata path for an FBX file."""
    target = Path(fbx_path)
    return target.with_suffix(".ghostrigger.json")


def inspect_fbx_skin_objects(asset_path: Path) -> dict[str, Any]:
    """Return lightweight ASCII FBX diagnostics for engine import contracts."""
    if asset_path.suffix.lower() != ".fbx" or not asset_path.exists():
        return {"checked": False, "reason": "not_fbx"}
    text = asset_path.read_text(encoding="utf-8", errors="ignore")
    skin_ids = re.findall(r'\bDeformer:\s+(\d+),\s+"[^"]*",\s+"Skin"', text)
    cluster_ids = re.findall(r'\bDeformer:\s+(\d+),\s+"[^"]*",\s+"Cluster"', text)
    legacy_cluster_ids = re.findall(r'\bSubDeformer:\s+(\d+),\s+"[^"]*",\s+"Cluster"', text)
    texture_ids = re.findall(r'\bTexture:\s+(\d+),', text)
    skeleton_attrs = re.findall(
        r'\bNodeAttribute:\s+\d+,\s+"[^"]*",\s+"(?:LimbNode|Limb|Skeleton)"',
        text,
    )
    limb_models = re.findall(r'\bModel:\s+\d+,\s+"[^"]*",\s+"LimbNode"', text)
    bind_poses = re.findall(r'\bPose:\s+\d+,\s+"[^"]*",\s+"BindPose"', text)
    pose_nodes = re.findall(r'\bPoseNode:\s+{', text)
    transform_count = len(re.findall(r'\bTransform:\s+\*16\s+{', text))
    transform_link_count = len(re.findall(r'\bTransformLink:\s+\*16\s+{', text))
    wrap_u_count = len(re.findall(r'\bP:\s+"WrapModeU"', text))
    wrap_v_count = len(re.findall(r'\bP:\s+"WrapModeV"', text))
    animation_stacks = re.findall(r'\bAnimationStack:\s+\d+,', text)
    animation_layers = re.findall(r'\bAnimationLayer:\s+\d+,', text)
    animation_curves = re.findall(r'\bAnimationCurve:\s+\d+,', text)

    object_counts = Counter(skin_ids + cluster_ids)
    duplicate_ids = {object_id: count for object_id, count in object_counts.items() if count > 1}
    skin_contract_ok = (
        not skin_ids
        or (
            bool(cluster_ids)
            and bool(skeleton_attrs)
            and bool(bind_poses)
            and transform_count >= len(cluster_ids)
            and transform_link_count >= len(cluster_ids)
        )
    )
    texture_contract_ok = (
        not texture_ids
        or (
            wrap_u_count >= len(texture_ids)
            and wrap_v_count >= len(texture_ids)
        )
    )

    return {
        "checked": True,
        "skin_deformers": len(skin_ids),
        "clusters": len(cluster_ids),
        "legacy_subdeformer_clusters": len(legacy_cluster_ids),
        "skeleton_node_attributes": len(skeleton_attrs),
        "limb_node_models": len(limb_models),
        "bind_poses": len(bind_poses),
        "pose_nodes": len(pose_nodes),
        "cluster_transforms": transform_count,
        "cluster_transform_links": transform_link_count,
        "textures": len(texture_ids),
        "texture_wrap_u": wrap_u_count,
        "texture_wrap_v": wrap_v_count,
        "animation_stacks": len(animation_stacks),
        "animation_layers": len(animation_layers),
        "animation_curves": len(animation_curves),
        "skin_contract_ok": skin_contract_ok,
        "texture_contract_ok": texture_contract_ok,
        "duplicate_object_ids": duplicate_ids,
        "ok": not duplicate_ids and not legacy_cluster_ids and skin_contract_ok and texture_contract_ok,
    }


def build_kotor_fbx_manifest(
    model: Any,
    fbx_path: str | Path,
    *,
    source_path: str = "",
    game: str = "",
    resref: str = "",
    exported_mesh_names: list[str] | None = None,
    fbx_diagnostics: dict[str, Any] | None = None,
    exporter_backend: str = "",
    fbx_format: str = "",
    target_engine: str = "unreal",
    compatibility_profile: str = "standard",
    pbr: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sidecar preserving KOTOR semantics that FBX cannot represent."""
    target = Path(fbx_path)
    nodes = _all_nodes(model)
    node_names = {str(getattr(node, "name", "") or "") for node in nodes}
    mesh_nodes = [node for node in nodes if _is_mesh_like(node)]
    exported_set = set(exported_mesh_names or [])
    exported_meshes = [
        node
        for node in mesh_nodes
        if not exported_set or str(getattr(node, "name", "") or "") in exported_set
    ]
    animations = list(getattr(model, "animations", []) or [])
    mode = _detect_mode(model)
    skin_summary = _skin_summary(mesh_nodes, node_names)
    diagnostics = fbx_diagnostics or inspect_fbx_skin_objects(target)
    profile = str(compatibility_profile or "standard").strip().lower()
    engine = str(target_engine or "unreal").strip().lower()
    meter_scale_profile = profile in {"unity", "unreal", "3ds_max"}
    animation_selection = _animation_selection_metadata(model)
    manifest = {
        "schema": "ghostrigger.kotor_fbx_manifest.v1",
        "schema_version": SCHEMA_VERSION,
        "tool": "ghostrigger.fbx_exporter",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "game": game or _game_name(model),
            "resref": resref or str(getattr(model, "name", "") or ""),
            "path": source_path,
            "model_name": str(getattr(model, "name", "") or ""),
            "supermodel": str(getattr(model, "supermodel", "") or ""),
            "classification": _safe_int(getattr(model, "model_type", 0), 0),
            "character_mode": mode.value if hasattr(mode, "value") else str(mode),
        },
        "fbx": {
            "path": str(target),
            "sidecar_path": str(sidecar_path_for_fbx(target)),
            "format": fbx_format or ("FBX 7.4 ASCII" if target.exists() else ""),
            "exporter_backend": exporter_backend,
            "compatibility_profile": profile,
            "diagnostics": diagnostics,
            **diagnostics,
        },
        "coordinate_system": {
            "source": "KOTOR object/bind space",
            "fbx_axis": "Z-up, -Y forward, +X right",
            "unit_scale": (
                "1 KOTOR unit declared as 1 meter (100 centimeters)"
                if meter_scale_profile else
                "1 KOTOR unit declared as 1 centimeter (legacy standard profile)"
            ),
            "uv_v_flipped_for_dcc": True,
            "notes": [
                "KOTOR-specific node semantics are stored in this manifest, not inferred from FBX node type alone.",
                "Use qbone/tbone metadata when rebuilding exact Odyssey skin bind data.",
            ],
        },
        "counts": {
            "nodes": len(nodes),
            "mesh_nodes": len(mesh_nodes),
            "exported_mesh_nodes": len(exported_meshes),
            "skin_mesh_nodes": len([node for node in mesh_nodes if bool(getattr(node, "is_skin", False))]),
            "vertices": sum(len(getattr(node, "vertices", []) or []) for node in exported_meshes),
            "faces": sum(len(getattr(node, "faces", []) or []) for node in exported_meshes),
            "animations": len(animations),
            "hooks": len(_hook_nodes(nodes)),
            "missing_skin_bones": len(skin_summary["missing_bones"]),
        },
        "nodes": [_node_entry(node) for node in nodes],
        "hooks": _hook_nodes(nodes),
        "materials": [_material_entry(node) for node in exported_meshes],
        "meshes": [_mesh_entry(node) for node in exported_meshes],
        "skeleton": {
            "supermodel": str(getattr(model, "supermodel", "") or ""),
            "bone_nodes": [_node_entry(node) for node in nodes if _is_skeleton_node(node)],
            "bone_count": len([node for node in nodes if _is_skeleton_node(node)]),
            "skin": skin_summary,
        },
        "animations": [_animation_entry(anim) for anim in animations],
        "kotor_semantics": {
            "node_flags_preserved": True,
            "texture_slots": ["diffuse", "lightmap", "bump_map", "texture_names"],
            "hook_roles": KOTOR_HOOK_ROLES,
            "supermodel_required_for_animation_inheritance": bool(str(getattr(model, "supermodel", "") or "").strip()),
            "roundtrip_notes": [
                "Do not reorder KOTOR skeleton nodes when reimporting into GhostRigger.",
                "Preserve bone_map order, qbone_list, and tbone_list for MDL/MDX export.",
                "FBX can be consumed by Unreal, while this manifest preserves Odyssey-only authoring state.",
            ],
        },
        "handoff": _engine_handoff(engine, profile),
    }
    if pbr is not None:
        manifest["pbr"] = _manifest_safe(pbr)
    if animation_selection is not None:
        manifest["animation_selection"] = animation_selection
    manifest[engine] = manifest["handoff"]
    manifest["validation"] = _validation_summary(manifest)
    return manifest


def write_kotor_fbx_manifest(
    model: Any,
    fbx_path: str | Path,
    *,
    source_path: str = "",
    game: str = "",
    resref: str = "",
    exported_mesh_names: list[str] | None = None,
    fbx_diagnostics: dict[str, Any] | None = None,
    exporter_backend: str = "",
    fbx_format: str = "",
    target_engine: str = "unreal",
    compatibility_profile: str = "standard",
    pbr: Mapping[str, Any] | None = None,
) -> Path:
    """Write and return the canonical FBX sidecar path."""
    manifest = build_kotor_fbx_manifest(
        model,
        fbx_path,
        source_path=source_path,
        game=game,
        resref=resref,
        exported_mesh_names=exported_mesh_names,
        fbx_diagnostics=fbx_diagnostics,
        exporter_backend=exporter_backend,
        fbx_format=fbx_format,
        target_engine=target_engine,
        compatibility_profile=compatibility_profile,
        pbr=pbr,
    )
    path = sidecar_path_for_fbx(fbx_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _all_nodes(model: Any) -> list[Any]:
    try:
        return list(model.all_nodes())
    except Exception:
        return []


def _game_name(model: Any) -> str:
    game = getattr(model, "game_version", "")
    return str(getattr(game, "name", game) or "")


def _detect_mode(model: Any) -> CharacterMode:
    try:
        return detect_character_mode(model)
    except Exception:
        return CharacterMode.AMBIGUOUS


def _node_entry(node: Any) -> dict[str, Any]:
    name = str(getattr(node, "name", "") or "")
    parent = getattr(node, "parent", None)
    return {
        "name": name,
        "parent": str(getattr(parent, "name", "") or "") if parent is not None else "",
        "flags": _safe_int(getattr(node, "flags", 0), 0),
        "type": str(getattr(node, "type_label", "") or _node_type(node)),
        "position": _float_list(getattr(node, "position", (0.0, 0.0, 0.0)), 3),
        "rotation_xyzw": _float_list(getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0)), 4),
        "render": bool(getattr(node, "render", True)),
        "is_mesh": bool(getattr(node, "is_mesh", False)),
        "is_skin": bool(getattr(node, "is_skin", False)),
        "is_hook": name.lower() in KOTOR_HOOK_ROLES,
        "hook_role": KOTOR_HOOK_ROLES.get(name.lower(), ""),
    }


def _mesh_entry(node: Any) -> dict[str, Any]:
    skin_data = list(getattr(node, "skin_data", []) or [])
    influence_counts = [len(getattr(row, "influences", []) or []) for row in skin_data]
    return {
        "name": str(getattr(node, "name", "") or ""),
        "type": str(getattr(node, "type_label", "") or _node_type(node)),
        "vertices": len(getattr(node, "vertices", []) or []),
        "faces": len(getattr(node, "faces", []) or []),
        "normals": len(getattr(node, "normals", []) or []),
        "tangents": len(getattr(node, "tangents", []) or []),
        "uv_sets": {
            "diffuse": len(getattr(node, "uvs", []) or []),
            "lightmap": len(getattr(node, "uvs_lm", []) or []),
            "uv2": len(getattr(node, "uvs_2", []) or []),
            "uv3": len(getattr(node, "uvs_3", []) or []),
        },
        "skin": {
            "is_skin": bool(getattr(node, "is_skin", False)),
            "bone_map": [str(item or "") for item in (getattr(node, "bone_map", []) or [])],
            "qbone_count": len(getattr(node, "qbone_list", []) or []),
            "tbone_count": len(getattr(node, "tbone_list", []) or []),
            "weighted_vertices": len([count for count in influence_counts if count > 0]),
            "max_influences_per_vertex": max(influence_counts) if influence_counts else 0,
        },
    }


def _material_entry(node: Any) -> dict[str, Any]:
    texture_names = [str(item or "") for item in (getattr(node, "texture_names", []) or [])]
    return {
        "mesh": str(getattr(node, "name", "") or ""),
        "diffuse": str(getattr(node, "texture_clean", "") or getattr(node, "texture", "") or ""),
        "lightmap": str(getattr(node, "lightmap", "") or ""),
        "bump_map": str(getattr(node, "bump_map", "") or getattr(node, "txi_bumpmaptexture", "") or ""),
        "texture_names": texture_names,
        "tex_count": _safe_int(getattr(node, "tex_count", len(texture_names) or 1), 1),
        "face_material_slots": sorted({_safe_int(item, 0) for item in (getattr(node, "face_mats", []) or [])}),
        "txi": {
            "blending": _safe_int(getattr(node, "txi_blending", 0), 0),
            "cube": bool(getattr(node, "txi_cube", False)),
            "proceduretype": str(getattr(node, "txi_proceduretype", "") or ""),
            "envmaptexture": str(getattr(node, "txi_envmaptexture", "") or ""),
            "bumpmaptexture": str(getattr(node, "txi_bumpmaptexture", "") or ""),
            "islightmap": bool(getattr(node, "txi_islightmap", False)),
            "isbumpmap": bool(getattr(node, "txi_isbumpmap", False)),
            "clamp_s": bool(getattr(node, "txi_clamp_s", False)),
            "clamp_t": bool(getattr(node, "txi_clamp_t", False)),
            "alpha_test": _safe_float(getattr(node, "txi_alpha_test", 0.5), 0.5),
        },
    }


def _skin_summary(mesh_nodes: list[Any], node_names: set[str]) -> dict[str, Any]:
    skin_meshes = []
    all_missing: set[str] = set()
    duplicate_maps: dict[str, dict[str, int]] = {}
    over_limit_vertices: dict[str, list[int]] = {}
    unnormalized_vertices: dict[str, list[int]] = {}
    for node in mesh_nodes:
        if not bool(getattr(node, "is_skin", False)):
            continue
        bone_map = [str(item or "") for item in (getattr(node, "bone_map", []) or [])]
        counts = Counter(item for item in bone_map if item)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        if duplicates:
            duplicate_maps[str(getattr(node, "name", "") or "")] = duplicates
        missing = sorted(name for name in counts if name not in node_names)
        all_missing.update(missing)
        skin_rows = list(getattr(node, "skin_data", []) or [])
        vertex_count = len(getattr(node, "vertices", []) or [])
        over_limit = []
        unnormalized = []
        zero_weight = []
        for index, row in enumerate(skin_rows):
            influences = list(getattr(row, "influences", []) or [])
            positive = [item for item in influences if _safe_float(getattr(item, "weight", 0.0), 0.0) > 0.0]
            if not positive:
                zero_weight.append(index)
            if len(positive) > 4:
                over_limit.append(index)
            total = sum(_safe_float(getattr(item, "weight", 0.0), 0.0) for item in positive)
            if positive and abs(total - 1.0) > 0.01:
                unnormalized.append(index)
        if over_limit:
            over_limit_vertices[str(getattr(node, "name", "") or "")] = over_limit[:64]
        if unnormalized:
            unnormalized_vertices[str(getattr(node, "name", "") or "")] = unnormalized[:64]
        skin_meshes.append(
            {
                "mesh": str(getattr(node, "name", "") or ""),
                "bone_map": bone_map,
                "bone_map_unique": list(counts.keys()),
                "missing_bones": missing,
                "duplicate_bone_map_entries": duplicates,
                "qbone_count": len(getattr(node, "qbone_list", []) or []),
                "tbone_count": len(getattr(node, "tbone_list", []) or []),
                "vertex_count": vertex_count,
                "vertex_rows": len(skin_rows),
                "weighted_vertices": len(skin_rows) - len(zero_weight),
                "zero_weight_vertices": zero_weight[:64],
            }
        )
    return {
        "skin_meshes": skin_meshes,
        "missing_bones": sorted(all_missing),
        "duplicate_bone_map_entries": duplicate_maps,
        "vertices_over_kotor_influence_limit": over_limit_vertices,
        "vertices_with_unnormalized_weights": unnormalized_vertices,
        "kotor_influence_limit": 4,
    }


def _animation_entry(anim: Any) -> dict[str, Any]:
    nodes = list(getattr(anim, "nodes", []) or [])
    controller_types: set[str] = set()
    keyed_nodes = []
    for node in nodes:
        controllers = list(getattr(node, "controllers", []) or [])
        for controller in controllers:
            controller_types.add(str(controller.get("type", "") if isinstance(controller, dict) else ""))
        keyed_nodes.append(
            {
                "name": str(getattr(node, "name", "") or ""),
                "controller_count": len(controllers),
                "controller_types": sorted(
                    str(controller.get("type", "") if isinstance(controller, dict) else "")
                    for controller in controllers
                ),
            }
        )
    return {
        "name": str(getattr(anim, "name", "") or ""),
        "length": _safe_float(getattr(anim, "length", 0.0), 0.0),
        "transition_time": _safe_float(getattr(anim, "transition_time", 0.0), 0.0),
        "anim_root": str(getattr(anim, "anim_root", "") or ""),
        "node_count": len(nodes),
        "controller_types": sorted(item for item in controller_types if item),
        "events": [
            {"time": _safe_float(getattr(event, "time", 0.0), 0.0), "name": str(getattr(event, "name", "") or "")}
            for event in (getattr(anim, "events", []) or [])
        ],
        "nodes": keyed_nodes,
    }


def _hook_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    hooks = []
    for node in nodes:
        name = str(getattr(node, "name", "") or "")
        role = KOTOR_HOOK_ROLES.get(name.lower())
        if role:
            hooks.append({"name": name, "role": role, "parent": str(getattr(getattr(node, "parent", None), "name", "") or "")})
    return hooks


def _validation_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    errors = []
    skin = manifest.get("skeleton", {}).get("skin", {})
    if skin.get("missing_bones"):
        warnings.append({"code": "missing_skin_bones", "message": "Some skin bone_map entries do not exist as FBX/KOTOR nodes."})
    if skin.get("vertices_over_kotor_influence_limit"):
        errors.append({"code": "too_many_skin_influences", "message": "Some vertices exceed KOTOR's 4-influence skin limit."})
    for row in skin.get("skin_meshes", []):
        mesh_name = str(row.get("mesh", "") or "<unnamed skin>")
        vertex_count = int(row.get("vertex_count", 0) or 0)
        vertex_rows = int(row.get("vertex_rows", 0) or 0)
        if vertex_rows != vertex_count:
            errors.append({
                "code": "skin_vertex_row_mismatch",
                "message": (
                    f"{mesh_name} has {vertex_rows} skin rows for "
                    f"{vertex_count} vertices."
                ),
            })
        if row.get("zero_weight_vertices"):
            errors.append({
                "code": "zero_weight_skin_vertices",
                "message": f"{mesh_name} contains zero-weight skin rows.",
            })
    fbx_diag = manifest.get("fbx", {}).get("diagnostics", {})
    if fbx_diag.get("checked") and not fbx_diag.get("ok", True):
        warnings.append({"code": "fbx_contract_warning", "message": "FBX text diagnostics reported import-contract issues."})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _animation_selection_metadata(model: Any) -> dict[str, Any] | None:
    """Normalize optional selected-take provenance for the FBX sidecar.

    Animation selection belongs to the workflow layer.  Core.IO only records
    the small, JSON-safe report stamped on the prepared export model, keeping
    the exporter independent from supermodel/resource resolution.
    """
    raw = getattr(model, "_gr_fbx_animation_selection", None)
    if raw is None:
        return None
    if is_dataclass(raw):
        raw = asdict(raw)
    elif not isinstance(raw, Mapping) and hasattr(raw, "__dict__"):
        raw = vars(raw)
    if not isinstance(raw, Mapping):
        return None

    payload = {str(key): _manifest_safe(value) for key, value in raw.items()}

    def first(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return default

    selected = _selection_names(first("selected", "selected_animation_names", default=[]))
    requested = _selection_names(
        first("requested", "requested_animation_names", default=selected)
    )
    embedded = _selection_names(
        first("embedded", "embedded_animation_names", default=selected)
    )
    if not selected and embedded:
        selected = list(embedded)
    missing = _selection_names(
        first("missing", "missing_animation_names", default=[])
    )
    sources = first(
        "sources",
        "source_by_animation",
        "source_models",
        "source",
        default={},
    )
    scales = first(
        "scales",
        "scale_by_animation",
        "cumulative_scales",
        "scale",
        default={},
    )
    scopes = first("scopes", "source_scopes", default={})
    contributing_models = first(
        "contributing_models",
        "contributors",
        default={},
    )
    sets = first("sets", "animation_sets", default=[])
    return {
        "selected": selected,
        "requested": requested,
        "embedded": embedded,
        "missing": missing,
        "sources": _manifest_safe(sources),
        "scales": _manifest_safe(scales),
        "scopes": _manifest_safe(scopes),
        "contributing_models": _manifest_safe(contributing_models),
        "sets": _manifest_safe(sets),
    }


def _selection_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item) for item in values if str(item).strip()]


def _manifest_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _manifest_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _manifest_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_manifest_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _manifest_safe(vars(value))
    return str(value)


def _engine_handoff(target_engine: str, compatibility_profile: str) -> dict[str, Any]:
    target = str(target_engine or "unreal").strip().lower()
    if target == "unity":
        from src.core.export.pbr_materials import shader_contract

        unity_shader = str(shader_contract("unity")["name"])
        return {
            "target": target,
            "compatibility_profile": compatibility_profile,
            "recommended_import": {
                "scale_factor": 1.0,
                "convert_units": True,
                "bake_axis_conversion": True,
                "import_animations": True,
                "animation_compression": "Off",
                "resample_curves": False,
                "import_materials": True,
                "material_naming": "By Base Texture Name",
                "normal_handling": "Import when exported tangents are present; otherwise calculate in Unity.",
                "material_shader": unity_shader,
                "base_color_space": "sRGB",
                "data_texture_space": "linear",
            },
            "notes": [
                "Keep the FBX and its textures folder together inside Assets so relative texture references resolve.",
                "The Unity profile declares one KOTOR coordinate unit as one meter and uses linear animation curves with continuous Euler branches.",
                "GhostRigger writes BaseColor, Normal, AO, ORM, and Unity MetallicSmoothness maps plus their native Lit shader bindings.",
            ],
        }
    if target == "3ds_max":
        return {
            "target": target,
            "compatibility_profile": compatibility_profile,
            "recommended_import": {
                "system_unit_conversion": "Automatic",
                "axis_conversion": "Z-up source; no manual root rotation",
                "animation": True,
                "materials": True,
            },
            "notes": [
                "Allow the FBX importer to convert the declared meter-scale system unit to the active 3ds Max system unit.",
                "Texture color can vary with 3ds Max OCIO/gamma policy; the FBX carries file links but not user color-space overrides.",
            ],
        }
    if target == "unreal":
        from src.core.export.pbr_materials import shader_contract

        unreal_shader = str(shader_contract("unreal")["name"])
        unreal_profile = str(compatibility_profile or "standard").strip().lower() == "unreal"
        return {
            "target": target,
            "compatibility_profile": compatibility_profile,
            "recommended_import": {
                "skeletal_mesh": True,
                "import_mesh": True,
                "import_animations": True,
                "animation_take_policy": "Import each selected embedded FBX take as a separate Animation Sequence.",
                "sample_rate_fps": 30,
                "animation_interpolation": "Linear",
                "import_materials": True,
                "import_textures": True,
                "import_meshes_in_bone_hierarchy": True,
                "preserve_smoothing_groups": True,
                "normal_import_method": "Import Normals and Tangents when tangents are present; otherwise let Unreal compute tangents.",
                "convert_scene_unit": unreal_profile,
                "use_t0_as_ref_pose": False,
                "update_skeleton_reference_pose": False,
                "preserve_native_kotor_skeleton": True,
                "automatic_quinn_retarget": False,
                "material_shader": unreal_shader,
                "base_color_srgb": True,
                "normal_srgb": False,
                "orm_srgb": False,
                "normal_compression": "Normalmap/BC5",
            },
            "notes": [
                "Keep the FBX and its textures folder together and enable texture import so relative PNG references resolve.",
                "Bind the generated ORM channels as R=Ambient Occlusion, G=Roughness, B=Metallic on an Unreal Default Lit material.",
                "Enable Import Meshes in Bone Hierarchy so attached eyes, eyelids, teeth, tongue, and other rigid child meshes are imported as geometry instead of converted to bones.",
                (
                    "The Unreal profile declares one KOTOR coordinate unit as one meter and writes fixed 30 fps linear animation curves with continuous Euler branches."
                    if unreal_profile else
                    "The standard profile retains GhostRigger's legacy centimeter declaration; choose the Unreal profile for meter-scale units and linear engine curves."
                ),
                "Selected animation takes are embedded in this single FBX and import as separate Animation Sequences on the same native KOTOR skeleton.",
                "GhostRigger preserves the Odyssey hierarchy and exact bone names; it does not silently retarget the export to Quinn or the Unreal mannequin.",
                "Use Unreal IK Rig/Retargeter after import when Quinn or mannequin animation compatibility is required.",
                "Enable Use T0 As Ref Pose only when the first animation frame is intentionally the model's bind/reference pose.",
                "Keep the GhostRigger manifest with the FBX so inherited-supermodel sources and KOTOR round-trip semantics remain available.",
            ],
        }
    return {
        "target": target,
        "compatibility_profile": compatibility_profile,
        "recommended_import": {
            "import_mesh": True,
            "import_animations": True,
        },
        "notes": [
            "Use the GhostRigger manifest for KOTOR-specific rebuild and round-trip metadata that FBX cannot represent.",
        ],
    }


def _is_mesh_like(node: Any) -> bool:
    return bool(getattr(node, "is_mesh", False) or getattr(node, "is_skin", False))


def _is_skeleton_node(node: Any) -> bool:
    return bool(getattr(node, "is_dummy", False) or not _is_mesh_like(node))


def _node_type(node: Any) -> str:
    if bool(getattr(node, "is_skin", False)):
        return "skin"
    if bool(getattr(node, "is_mesh", False)):
        return "trimesh"
    return "dummy"


def _float_list(values: Any, count: int) -> list[float]:
    result = [_safe_float(item, 0.0) for item in list(values or [])[:count]]
    while len(result) < count:
        result.append(1.0 if count == 4 and len(result) == 3 else 0.0)
    return result


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
