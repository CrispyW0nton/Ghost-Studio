"""
GhostRigger-specific MCP tools.

Architecture (Khononov, "Balancing Coupling in Software Design"):
  • Each handler depends on:
      - ModelLocatorPort  (Contract Coupling: find bytes)
      - ModelParserPort   (Contract Coupling: parse bytes)
      - ModelAnalyzer     (Domain service: extract structured info)
  • The adapters (FileSystemModelLocator, MDLBinaryParserAdapter, etc.)
    live in adapters.py — all pykotor / filesystem specifics are there.
  • Handlers receive a pre-built container of collaborators via
    _get_services(), which composes a default adapter chain on first call.
  • Result types (ModelInfo, AuditResult) are immutable data contracts
    defined in ports.py — tools serialize them to JSON; nothing leaks
    the internal model object outside this module.

Changes from v2.9:
  • Removed duplicate _locate_mdl / _load_model helpers (Connascence of
    Algorithm eliminated — now handled by CompositeModelLocator +
    MDLBinaryParserAdapter)
  • Removed scattered sys.path.insert() calls (intrusive path hacking moved
    to MDLBinaryParserAdapter)
  • resolve_game_safe() now delegates to the injected registry
  • handle_model_info() and handle_audit() share ModelAnalyzer (single source
    of truth for model-interrogation logic)
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kotormcp.ports import (
    AuditResult,
    InstallationRegistryPort,
    ModelInfo,
    ModelLocatorPort,
    ModelParserPort,
)
from kotormcp.utils import json_content

# ── Service container ─────────────────────────────────────────────────────────

class _Services:
    """
    Lightweight container wiring the default adapter chain.

    Using a container instead of module-level globals means tests can
    swap individual services without patching globals (lower Distance,
    lower coupling strength for test code).
    """

    def __init__(
        self,
        locator: ModelLocatorPort,
        parser: ModelParserPort,
        analyzer,  # ModelAnalyzer — not typed to avoid circular import
        registry: InstallationRegistryPort,
        texture_cache_factory=None,
    ):
        self.locator = locator
        self.parser = parser
        self.analyzer = analyzer
        self.registry = registry
        self.texture_cache_factory = texture_cache_factory


_default_services: Optional[_Services] = None


def _get_services() -> _Services:
    """Build the default service composition on first call (lazy singleton)."""
    global _default_services
    if _default_services is None:
        from kotormcp.adapters import (  # noqa: PLC0415
            CompositeModelLocator,
            FileSystemModelLocator,
            InstallationModelLocator,
            MDLBinaryParserAdapter,
            ModelAnalyzer,
            InstallationTextureCache,
            get_default_registry,
        )
        registry = get_default_registry()
        locator = CompositeModelLocator([
            FileSystemModelLocator(),
            InstallationModelLocator(registry),
        ])
        parser = MDLBinaryParserAdapter()
        analyzer = ModelAnalyzer()
        def texture_cache_factory(game_alias, explicit_path=None):
            resolved = registry.resolve(game_alias)
            if resolved is None:
                raise ValueError(f"Unsupported game alias for texture export: {game_alias}")
            return InstallationTextureCache(registry.load(resolved, explicit_path))

        _default_services = _Services(
            locator,
            parser,
            analyzer,
            registry,
            texture_cache_factory=texture_cache_factory,
        )
    return _default_services


def _reset_services() -> None:
    """Force a rebuild of the service container (used in tests)."""
    global _default_services
    _default_services = None


# ── Tool definitions ──────────────────────────────────────────────────────────

def get_tools() -> List[Dict[str, Any]]:
    """Return GhostRigger-specific tool definitions."""
    return [
        {
            "name": "ghostrigger_source_identity",
            "description": (
                "Read-only: report the exact source bytes an export would read "
                "(model/head MDL+MDX via the export locator, the model the debug "
                "session would load, or the fixed Day 4 retarget corpus) as SHA-256 "
                "and sizes, plus the chitin.key hash of the installation. "
                "Writes nothing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["model", "debug_session", "retarget_day4"]},
                    "resref": {"type": "string", "maxLength": 1024},
                    "head_resref": {"type": "string", "maxLength": 1024},
                    "game": {"type": "string", "enum": ["k1", "k2"]},
                    "game_path": {"type": "string", "maxLength": 1024},
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "ghostrigger_open_model",
            "description": (
                "Open a KotOR MDL/MDX model in the GhostRigger 3D viewport. "
                "Pass resref (without extension) or an absolute file path. "
                "Optionally provide game/game_path for installation-relative lookup."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": "MDL resource name (e.g. n_sithpraet) or absolute file path",
                    },
                    "game": {
                        "type": "string",
                        "description": "Optional game alias (k1/k2)",
                    },
                    "game_path": {
                        "type": "string",
                        "description": "Optional absolute path to KotOR installation",
                    },
                },
                "required": ["resref"],
            },
        },
        {
            "name": "ghostrigger_render_model",
            "description": (
                "Render a KotOR MDL model to a PNG image and return the file path. "
                "Useful for quick visual inspection. Requires a GPU/OpenGL context."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": (
                            "Body/full-model MDL resource name or absolute file path. "
                            "When head_resref is supplied this is the BAS body."
                        ),
                    },
                    "head_resref": {
                        "type": "string",
                        "description": (
                            "Optional detachable head MDL. Ghost Studio BAS attaches it "
                            "to the body's animated headhook before Unreal FBX export."
                        ),
                    },
                    "body_texture": {
                        "type": "string",
                        "description": (
                            "Optional UTC/UTI-resolved body texture override, such as PMBI03."
                        ),
                    },
                    "head_texture": {
                        "type": "string",
                        "description": (
                            "Optional heads.2da-resolved head texture override."
                        ),
                    },
                    "width": {"type": "integer", "default": 512, "minimum": 16, "maximum": 4096},
                    "height": {"type": "integer", "default": 512, "minimum": 16, "maximum": 4096},
                    "azimuth": {"type": "number", "default": 45, "minimum": -360, "maximum": 360},
                    "elevation": {"type": "number", "default": 30, "minimum": -90, "maximum": 90},
                    "output_path": {"type": "string"},
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": "False creates output_path exclusively and fails if it exists",
                    },
                    "game": {"type": "string"},
                    "game_path": {"type": "string"},
                },
                "required": ["resref"],
            },
        },
        {
            "name": "ghostrigger_model_info",
            "description": (
                "Return structured information about a KotOR MDL model: "
                "node count, mesh nodes, bone list, UV presence, bounding box, "
                "animations, classification, and supermodel."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": "MDL resource name or absolute file path",
                    },
                    "game": {"type": "string"},
                    "game_path": {"type": "string"},
                },
                "required": ["resref"],
            },
        },
        {
            "name": "ghostrigger_list_game_models",
            "description": (
                "List MDL models available in a KotOR installation, with optional "
                "name filter. Returns up to `limit` results."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "k1 or k2"},
                    "game_path": {"type": "string"},
                    "filter": {"type": "string", "description": "Substring filter"},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["game"],
            },
        },
        {
            "name": "ghostrigger_audit",
            "description": (
                "Run a quick integrity check on a KotOR MDL model: "
                "counts nodes, detects UV/normal mismatches, validates bounding box."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": "MDL resource name or absolute file path",
                    },
                    "game": {"type": "string"},
                    "game_path": {"type": "string"},
                },
                "required": ["resref"],
            },
        },
        {
            "name": "ghostrigger_export_model_for_unity",
            "description": (
                "Export a KotOR MDL/MDX model from game data or a file path into "
                "a Unity project Assets folder as FBX, with a GhostRigger metadata "
                "JSON sidecar. Use this for repeatable GhostRigger -> Unity MCP "
                "asset transfer tests."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": "MDL resource name or absolute file path",
                    },
                    "game": {
                        "type": "string",
                        "description": "Game alias: k1 or k2",
                    },
                    "game_path": {
                        "type": "string",
                        "description": "Optional absolute path to KotOR installation",
                    },
                    "unity_project": {
                        "type": "string",
                        "description": "Absolute Unity project root",
                    },
                    "asset_subdir": {
                        "type": "string",
                        "default": "Assets/KotorImported/GhostRigger",
                        "description": "Unity-project-relative output folder",
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Optional output filename stem; defaults to resref",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "False refuses to replace an existing asset or its "
                            ".ghostrigger.json/.pbr.json sidecars"
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": ["fbx"],
                        "default": "fbx",
                    },
                    "export_rigging": {
                        "type": "boolean",
                        "default": True,
                        "description": "Write rigging JSON sidecars next to the FBX",
                    },
                    "export_textures": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Extract referenced KOTOR textures and generate Unity Lit "
                            "PBR material maps beside the FBX"
                        ),
                    },
                    "animation_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Effective local or inherited animation set names to embed as "
                            "Unity-compatible FBX takes. An explicit empty array exports a "
                            "mesh/rig-only FBX; omitting this property preserves legacy local clips."
                        ),
                    },
                },
                "required": ["resref", "game", "unity_project"],
            },
        },
        {
            "name": "ghostrigger_export_model_for_unreal",
            "description": (
                "Export a KotOR MDL/MDX model for Unreal Engine as FBX or as a "
                "rigged GLB interchange source. "
                "Selected effective animation sets are materialized from the supermodel "
                "chain and embedded as independent animation takes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {
                        "type": "string",
                        "description": "MDL resource name or absolute file path",
                    },
                    "game": {
                        "type": "string",
                        "description": "Game alias: k1 or k2",
                    },
                    "game_path": {
                        "type": "string",
                        "description": "Optional absolute path to the KotOR installation",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Complete destination .fbx or .glb path",
                    },
                    "export_dir": {
                        "type": "string",
                        "description": "Destination folder; filename defaults to the model resref",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "False refuses to replace an existing asset or its "
                            ".ghostrigger.json/.pbr.json sidecars"
                        ),
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Optional filename stem when export_dir is used",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["fbx", "glb"],
                        "default": "fbx",
                        "description": (
                            "FBX is the direct Unreal transfer. GLB is the robust "
                            "BAS/skin/animation source for binary-FBX normalization."
                        ),
                    },
                    "head_resref": {
                        "type": "string",
                        "description": (
                            "Optional detachable BAS head MDL attached to the body's "
                            "animated headhook before export."
                        ),
                    },
                    "body_texture": {
                        "type": "string",
                        "description": "Optional UTC/UTI-resolved body texture override.",
                    },
                    "head_texture": {
                        "type": "string",
                        "description": "Optional heads.2da-resolved head texture override.",
                    },
                    "export_rigging": {
                        "type": "boolean",
                        "default": True,
                        "description": "Write rigging JSON sidecars next to the FBX",
                    },
                    "export_textures": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Extract referenced game textures as PNG sidecars and bind "
                            "the FBX material references to their relative paths"
                        ),
                    },
                    "animation_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Effective local or inherited animation set names to embed as "
                            "Unreal-compatible FBX takes. An explicit empty array exports a "
                            "mesh/rig-only FBX; omitting this property preserves legacy local clips."
                        ),
                    },
                },
                "required": ["resref", "game"],
            },
        },
        {
            "name": "ghostrigger_validate_unity_import",
            "description": (
                "Build a validation manifest for a GhostRigger-exported Unity asset. "
                "Pass the GhostRigger transfer metadata sidecar and Unity-side import "
                "facts (clips, renderer types, material count, skin/bindpose counts) "
                "to get stable pass/warning/error diagnostics."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "transfer_metadata_path": {
                        "type": "string",
                        "description": "Path to the *.ghostrigger.json transfer sidecar",
                    },
                    "unity_summary": {
                        "type": "object",
                        "description": "Unity import facts: asset_path, clips, renderers, warnings, errors",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to write the validation manifest JSON",
                    },
                },
                "required": ["transfer_metadata_path", "unity_summary"],
            },
        },
        {
            "name": "ghostrigger_run_malak_unity_smoke",
            "description": (
                "Run the Unity MCP Ghost import smoke for a GhostRigger model "
                "(defaults: the legacy Malak main-menu fixture). Talks only to the "
                "authenticated loopback bridge named by <unity_project>/Library/"
                "UnityMcpGhost/bridge-identity.json (owner-only bearer token, exact "
                "project and editor IDs). Refuses while any open scene is unsaved or "
                "untitled, in play mode, compiling, importing or with a busy bridge "
                "queue; opens the scene, optionally refreshes assets, verifies the "
                "instance/Animator/renderers, captures before/after screenshots and "
                "restores the prior scene setup. dry_run only reports readiness. The "
                "report is created exclusively at output_path."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "unity_project": {
                        "type": "string",
                        "description": "Absolute Unity project root (its Library/UnityMcpGhost holds the bridge identity and token)",
                    },
                    "scene_path": {
                        "type": "string",
                        "description": "Unity scene asset path (Assets/...unity)",
                    },
                    "asset_path": {
                        "type": "string",
                        "description": "GhostRigger FBX asset path (Assets/...fbx)",
                    },
                    "instance_name": {
                        "type": "string",
                        "description": "Expected GameObject instance name in the scene",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional absolute report path, created exclusively (its folder must exist)",
                    },
                    "screenshot_dir": {
                        "type": "string",
                        "description": "Optional absolute existing empty folder; the two screenshots are copied into it exclusively",
                    },
                    "screenshot_delay": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 1.0,
                    },
                    "screenshot_width": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 1920,
                        "default": 640,
                    },
                    "screenshot_height": {
                        "type": "integer",
                        "minimum": 64,
                        "maximum": 1080,
                        "default": 360,
                    },
                    "refresh_assets": {
                        "type": "boolean",
                        "default": True,
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                    },
                    "operation_id": {
                        "type": "string",
                        "description": "Optional idempotency prefix for the bridge mutations ([A-Za-z0-9-], at most 64)",
                    },
                    "expected_port": {
                        "type": "integer",
                        "minimum": 1024,
                        "maximum": 65535,
                        "description": "Optional: refuse unless the bridge identity still names this port (the listener the caller verified)",
                    },
                    "expected_editor_instance_id": {
                        "type": "string",
                        "description": "Optional: refuse unless the bridge identity still names this editor instance",
                    },
                },
                "required": ["unity_project"],
                "additionalProperties": False,
            },
        },
    ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _locate_and_parse(
    resref: str,
    game: Optional[str],
    game_path: Optional[str],
    svc: _Services,
) -> Tuple[str, Any]:
    """
    Locate MDL bytes and parse to a model object.

    Returns (path_label, model).
    Raises FileNotFoundError or Exception on failure.
    """
    path_label, mdl_bytes, mdx_bytes = svc.locator.locate(resref, game, game_path)
    model = svc.parser.parse(mdl_bytes, mdx_bytes, path_label)
    return path_label, model


def _export_fbx_for_unity(
    model: Any,
    out_path: Path,
    export_rigging: bool,
    *,
    tex_cache=None,
) -> bool:
    """Small seam for tests around the real FBX exporter."""
    try:
        from src.converters.mesh_converter import FBXExporter  # noqa: PLC0415
    except ImportError:                                      # pragma: no cover - MCP path shim
        from converters.mesh_converter import FBXExporter     # type: ignore  # noqa: PLC0415

    return FBXExporter().export(
        model,
        str(out_path),
        tex_cache=tex_cache,
        export_rigging=export_rigging,
        base_skeleton_model=getattr(model, "_gr_fbx_base_skeleton_model", None),
        compatibility_profile="unity",
    )


def _export_fbx_for_unreal(
    model: Any,
    out_path: Path,
    export_rigging: bool,
    *,
    tex_cache=None,
) -> bool:
    """Small seam for tests around the Unreal-compatible FBX exporter."""
    try:
        from src.converters.mesh_converter import FBXExporter  # noqa: PLC0415
    except ImportError:                                      # pragma: no cover - MCP path shim
        from converters.mesh_converter import FBXExporter     # type: ignore  # noqa: PLC0415

    return FBXExporter().export(
        model,
        str(out_path),
        tex_cache=tex_cache,
        export_rigging=export_rigging,
        base_skeleton_model=getattr(model, "_gr_fbx_base_skeleton_model", None),
        compatibility_profile="unreal",
    )


def _export_glb_for_unreal(
    model: Any,
    out_path: Path,
    export_rigging: bool,
    *,
    tex_cache=None,
) -> bool:
    """Write the lossless BAS/skin/animation interchange used before Unreal FBX."""
    try:
        from src.converters.mesh_converter import GLTFExporter  # noqa: PLC0415
    except ImportError:                                      # pragma: no cover - MCP path shim
        from converters.mesh_converter import GLTFExporter   # type: ignore  # noqa: PLC0415

    return GLTFExporter().export(
        model,
        str(out_path),
        binary=True,
        tex_cache=tex_cache,
        export_rigging=export_rigging,
    )


def _apply_primary_skin_texture(model: Any, texture_resref: str) -> int:
    """Apply an explicit UTC/UTI texture variation to rendered skin sections."""
    texture = str(texture_resref or "").strip()
    if model is None or not texture or texture.lower() in {"null", "none", "****"}:
        return 0
    try:
        nodes = list(model.all_nodes())
    except Exception:
        nodes = list(getattr(model, "nodes", []) or [])
    changed = 0
    for node in nodes:
        if not bool(getattr(node, "is_skin", False)):
            continue
        if not list(getattr(node, "vertices", []) or []):
            continue
        setattr(node, "texture", texture)
        texture_names = list(getattr(node, "texture_names", []) or [])
        if texture_names:
            texture_names[0] = texture
        else:
            texture_names = [texture]
        setattr(node, "texture_names", texture_names)
        if hasattr(node, "tex_count"):
            setattr(node, "tex_count", max(1, int(getattr(node, "tex_count", 0) or 0)))
        changed += 1
    return changed


def _compose_bas_character(
    body_model: Any,
    head_model: Any,
    *,
    name: str,
) -> tuple[Any, Dict[str, Any]]:
    """Build the same headhook-based BAS composition used by the Ghost Studio UI."""
    try:
        from src.systems.bas.preview_composer import (  # noqa: PLC0415
            build_bas_preview_model,
            prepare_bas_composed_export_model,
        )
    except ImportError:                                      # pragma: no cover - payload shim
        from systems.bas.preview_composer import (  # type: ignore  # noqa: PLC0415
            build_bas_preview_model,
            prepare_bas_composed_export_model,
        )

    preview = build_bas_preview_model(
        body_model=body_model,
        attachment_models={"head": head_model},
        name=name,
    )
    composed, report = prepare_bas_composed_export_model(
        preview,
        require_unique_body_names=True,
    )
    composed.name = name
    return composed, dict(report)


def _artifact_stamp(path: Path) -> Tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _texture_sidecar_inventory(
    output_path: Path,
    *,
    previous_manifest_stamp: Tuple[int, int] | None = None,
    require_changed_manifest: bool = False,
) -> List[Dict[str, Any]]:
    """Return evidence only for sidecars generated by the current FBX export."""
    pbr_manifest = output_path.with_suffix(".pbr.json")
    try:
        if (
            require_changed_manifest
            and _artifact_stamp(pbr_manifest) == previous_manifest_stamp
        ):
            return []
        manifest = json.loads(pbr_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    root = output_path.parent.resolve()
    channel_paths = {
        str(channel.get("path") or "")
        for material in manifest.get("materials", [])
        if isinstance(material, dict)
        for channel in dict(material.get("channels") or {}).values()
        if isinstance(channel, dict) and str(channel.get("path") or "")
    }
    rows: List[Dict[str, Any]] = []
    for relative_path in sorted(channel_paths, key=str.casefold):
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.suffix.lower() != ".png" or not path.is_file():
            continue
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({
            "name": path.stem,
            "path": str(path),
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return rows


def _glb_pbr_inventory(output_path: Path) -> Dict[str, Any]:
    """Return digest-backed evidence for PBR materials embedded in one GLB."""
    digest = sha256()
    with output_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with output_path.open("rb") as stream:
        header = stream.read(12)
        chunk_header = stream.read(8)
        if len(header) != 12 or len(chunk_header) != 8:
            raise ValueError("GLB header is truncated")
        magic, version, total_length = struct.unpack("<III", header)
        json_length, json_type = struct.unpack("<II", chunk_header)
        if magic != 0x46546C67 or version != 2 or json_type != 0x4E4F534A:
            raise ValueError("GLB does not contain a valid glTF 2.0 JSON chunk")
        document = json.loads(stream.read(json_length).decode("utf-8"))

    materials: List[Dict[str, Any]] = []
    for material in document.get("materials", []):
        if not isinstance(material, dict):
            continue
        pbr = dict(material.get("pbrMetallicRoughness") or {})
        extras = dict(material.get("extras") or {})
        contract = dict(extras.get("ghostriggerPbr") or {})
        alpha_contract = dict(contract.get("alpha") or {})
        materials.append({
            "name": str(material.get("name") or ""),
            "shader": str(contract.get("shader") or ""),
            "alpha_mode": str(material.get("alphaMode") or "OPAQUE"),
            "blend_operation": str(
                alpha_contract.get("blend_operation") or material.get("alphaMode") or "OPAQUE"
            ),
            "base_color_texture": "baseColorTexture" in pbr,
            "metallic_roughness_texture": "metallicRoughnessTexture" in pbr,
            "normal_texture": "normalTexture" in material,
            "occlusion_texture": "occlusionTexture" in material,
        })
    return {
        "path": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "declared_bytes": total_length,
        "image_count": len(document.get("images", [])),
        "material_count": len(materials),
        "materials": materials,
    }


class _McpAnimationResourceAdapter:
    """Expose the MCP locator/parser pair as a supermodel-chain loader."""

    revision = 0

    def __init__(self, services: _Services, game_path: Optional[str]) -> None:
        self._services = services
        self._game_path = game_path

    def load_model(self, resref: str, game: str) -> Any:
        try:
            _path_label, model = _locate_and_parse(
                resref,
                game,
                self._game_path,
                self._services,
            )
            return model
        except Exception:
            return None

    def load_model_strict(self, resref: str, game: str) -> Any:
        return self.load_model(resref, game)


def _resolve_fbx_base_skeleton(
    model: Any,
    game: Optional[str],
    game_path: Optional[str],
    services: _Services,
) -> Any:
    """Resolve and attach the immediate supermodel used for FBX bind poses."""
    supermodel = str(getattr(model, "supermodel", "") or "").strip()
    if not supermodel or supermodel.lower() in {"null", "none", "****"}:
        return None
    try:
        _base_path, base_skeleton = _locate_and_parse(
            supermodel,
            game,
            game_path,
            services,
        )
    except Exception:
        # qBone/tBone is the bounded fallback. The exported validation
        # manifest will still report any unresolved palette entries.
        return None
    setattr(model, "_gr_fbx_base_skeleton_model", base_skeleton)
    return base_skeleton


def _requested_animation_names(arguments: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    """Preserve omitted-vs-empty selection semantics from the MCP request."""
    if "animation_names" not in arguments or arguments.get("animation_names") is None:
        return None
    raw_names = arguments.get("animation_names")
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if not isinstance(raw_names, (list, tuple)):
        raise ValueError("animation_names must be an array of strings.")
    if any(not isinstance(name, str) for name in raw_names):
        raise ValueError("animation_names must contain only strings.")
    return tuple(raw_names)


def _prepare_selected_fbx_animations(
    model: Any,
    arguments: Dict[str, Any],
    *,
    game: str,
    game_path: Optional[str],
    services: _Services,
    supplemental_models: Tuple[Any, ...] = (),
) -> Any:
    """Materialize exactly the requested effective clips before FBX writing."""
    selected_names = _requested_animation_names(arguments)

    try:
        from src.core.animation.fbx_animation_selection import (  # noqa: PLC0415
            prepare_fbx_animation_export_model,
        )
    except ImportError:                                      # pragma: no cover - MCP path shim
        from core.animation.fbx_animation_selection import (  # type: ignore  # noqa: PLC0415
            prepare_fbx_animation_export_model,
        )

    base_skeleton_model = getattr(model, "_gr_fbx_base_skeleton_model", None)
    prepared = prepare_fbx_animation_export_model(
        model,
        selected_names,
        game=game,
        resource_manager=_McpAnimationResourceAdapter(services, game_path),
        base_skeleton_model=base_skeleton_model,
        supplemental_models=tuple(supplemental_models or ()),
    )
    if base_skeleton_model is not None:
        # Keep the already-resolved bind-pose model shared rather than retaining
        # a second deep-copied supermodel tree on the prepared export candidate.
        setattr(prepared, "_gr_fbx_base_skeleton_model", base_skeleton_model)
    return prepared


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_open_model(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Open model in GhostRigger viewport via IPC (if running) or return info."""
    resref = arguments.get("resref", "")
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    svc = _get_services()

    try:
        path_label, mdl_bytes, _ = svc.locator.locate(resref, game, game_path)
        size = len(mdl_bytes)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})

    # Try to open via IPC (GhostRigger may be running)
    try:
        import requests  # noqa: PLC0415
        resp = requests.post(
            "http://127.0.0.1:7001/api/open_mdl",
            json={
                "version": "1.0",
                "sender": "KotorMCP",
                "action": "open_mdl",
                "payload": {"resref": resref, "module_dir": ""},
            },
            timeout=1.0,
        )
        ipc_ok = resp.json().get("status") == "ok"
    except Exception:
        ipc_ok = False

    return json_content({
        "status": "ok",
        "resref": resref,
        "path": path_label,
        "size": size,
        "ipc_sent": ipc_ok,
        "message": (
            "Opened in GhostRigger viewport" if ipc_ok
            else "IPC not available; model located at path"
        ),
    })


_RENDER_MIN_SIZE = 16
_RENDER_MAX_SIZE = 4096
_UNSAFE_STEM_CHARS = set('/\\:*?"<>|\x00')


def _render_view_arguments(arguments: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Validate render size and camera angles (bounded, finite)."""
    import math  # noqa: PLC0415

    try:
        width = int(arguments.get("width", 512))
        height = int(arguments.get("height", 512))
        azimuth = float(arguments.get("azimuth", 45))
        elevation = float(arguments.get("elevation", 30))
    except (TypeError, ValueError) as exc:
        raise ValueError("width/height must be integers and azimuth/elevation numbers") from exc
    for label, value in (("width", width), ("height", height)):
        if not (_RENDER_MIN_SIZE <= value <= _RENDER_MAX_SIZE):
            raise ValueError(f"{label} must be between {_RENDER_MIN_SIZE} and {_RENDER_MAX_SIZE}")
    if not (math.isfinite(azimuth) and -360.0 <= azimuth <= 360.0):
        raise ValueError("azimuth must be a finite angle between -360 and 360")
    if not (math.isfinite(elevation) and -90.0 <= elevation <= 90.0):
        raise ValueError("elevation must be a finite angle between -90 and 90")
    return width, height, azimuth, elevation


def _safe_output_stem(stem: str) -> bool:
    """Return True for a plain filename stem with no path components."""
    text = str(stem or "")
    return (
        bool(text)
        and len(text) <= 128
        and text not in {".", ".."}
        and not any(ch in _UNSAFE_STEM_CHARS for ch in text)
    )


def _existing_export_artifacts(asset_path: Path) -> List[str]:
    """Return existing primary/sidecar paths an engine export would replace."""
    candidates = (
        asset_path,
        asset_path.with_suffix(".ghostrigger.json"),
        asset_path.with_suffix(".pbr.json"),
    )
    return [str(path) for path in candidates if path.exists()]


def _flag_tool_errors(handler):
    """Mark ``{"error": ...}`` payloads as MCP tool errors (isError)."""
    import functools  # noqa: PLC0415

    @functools.wraps(handler)
    async def wrapped(arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = await handler(arguments)
        try:
            payload = json.loads(raw.get("text", ""))
        except (TypeError, ValueError, AttributeError):
            return raw
        if isinstance(payload, dict) and payload.get("error"):
            return {**raw, "isError": True}
        return raw

    return wrapped


async def handle_render_model(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Render MDL to PNG and return file path."""
    import os  # noqa: PLC0415

    resref = arguments.get("resref", "")
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    try:
        width, height, azimuth, elevation = _render_view_arguments(arguments)
    except ValueError as exc:
        return json_content({"error": str(exc)})
    output_path: Optional[str] = arguments.get("output_path")
    overwrite = arguments.get("overwrite", True) is not False
    if output_path is not None and not overwrite and os.path.exists(output_path):
        return json_content({"error": "Refusing to overwrite existing render output."})
    svc = _get_services()

    try:
        path_label, model = _locate_and_parse(resref, game, game_path, svc)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})
    except Exception as exc:
        return json_content({"error": f"Failed to parse MDL: {exc}"})

    try:
        # Bootstrap sys.path so relative imports (..core.model_data) resolve.
        # The gui package uses "from ..core.model_data import …" which requires
        # the project root (parent of src/) to be on sys.path so that
        # "src.gui" and "src.core" are addressable as proper sub-packages.
        import sys  # noqa: PLC0415
        _project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        # MCP servers are long-lived, so drop stale aliases before importing
        # the canonical core/adapter modules.
        for _mod_name in (
            "src.adapters.rendering.moderngl_legacy_bridge",
            "src.adapters.rendering.moderngl_scene_helpers",
            "src.core.rendering.gpu_scene_helpers",
            "src.core.rendering.frame_core.renderer",
        ):
            _mod = sys.modules.get(_mod_name)
            _target = str(getattr(_mod, "_target", "") or "")
            if _target.startswith("src.gui."):
                sys.modules.pop(_mod_name, None)
        from src.adapters.rendering.moderngl_legacy_bridge import GpuRenderer
        from src.adapters.rendering.moderngl_scene_helpers import render_model_autoframe
        from src.core.camera.arcball_camera import ArcBallCamera
        from src.core.graphics.tpc import _is_tpc_data, _load_tpc_bytes

        # ── Build texture dict from model nodes + game library ─────────────────
        # Collect all texture names referenced by the model
        tex_names: list = []
        seen_tex: set = set()
        try:
            all_nodes_fn = getattr(model, 'all_nodes', None)
            nodes = list(all_nodes_fn()) if all_nodes_fn else []
            for _n in nodes:
                if not getattr(_n, 'is_mesh', False):
                    continue
                for attr in ('texture', 'lightmap', 'bump_map',
                             'txi_envmaptexture', 'txi_specularcolour'):
                    _t = str(getattr(_n, attr, '') or '').strip().lower()
                    if _t and _t not in ('null', '', 'none') and _t not in seen_tex:
                        tex_names.append(_t)
                        seen_tex.add(_t)
                # Also check texture_names list (multi-texture nodes)
                for _tn in getattr(_n, 'texture_names', []):
                    _t = str(_tn or '').strip().lower()
                    if _t and _t not in ('null', '', 'none') and _t not in seen_tex:
                        tex_names.append(_t)
                        seen_tex.add(_t)
        except Exception:
            pass

        textures: dict = {}
        _lib = getattr(svc, 'library', None) or getattr(svc, 'lib', None)
        # Try to find game library from service
        if _lib is None:
            try:
                from src.resources.game_library import GameLibrary  # noqa: PLC0415
                _lib = GameLibrary()
                if game_path and os.path.isdir(game_path):
                    _lib.scan(game_path)
            except Exception:
                pass
        if _lib is not None:
            _game_tag = 'K2' if (game or '').upper() in ('K2', 'TSL', 'KOTOR2') else 'K1'
            for tname in tex_names:
                try:
                    raw = _lib.get_texture_data(tname, _game_tag)
                    if raw:
                        img_tex = None
                        if _is_tpc_data(raw):
                            img_tex = _load_tpc_bytes(raw)
                        else:
                            try:
                                import io as _io
                                from PIL import Image as _PILImg  # noqa: PLC0415
                                img_tex = _PILImg.open(_io.BytesIO(raw)).convert('RGBA')
                            except Exception:
                                pass
                        if img_tex is not None:
                            textures[tname] = img_tex
                except Exception:
                    pass

        # MDL headers may contain oversized placeholder bounds. Use the same
        # renderable, world-space geometry bounds as the rendering backend.
        from src.core.rendering.gpu_scene_helpers import _compute_model_bounds

        camera = ArcBallCamera()
        camera.azimuth = azimuth
        camera.elevation = elevation
        render_bounds = _compute_model_bounds(model)
        camera.frame_bounds(
            tuple(render_bounds[f"min_{axis}"] for axis in "xyz"),
            tuple(render_bounds[f"max_{axis}"] for axis in "xyz"),
        )

        renderer = GpuRenderer()
        # The current renderer owns a standalone GPU context. Forcing CPU here
        # disables that context; this renderer no longer has a PIL fallback.
        img = renderer.render(model, camera, width, height, textures=textures)

        if img is None:
            return json_content({"error": "GPU renderer unavailable; no image was produced"})

        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "audit_output"
            )
            os.makedirs(output_dir, exist_ok=True)
            # Sanitise resref for use as a filename
            safe_ref = os.path.basename(resref).replace(os.sep, "_")
            output_path = os.path.join(
                output_dir,
                f"mcp_render_{safe_ref}_{int(azimuth)}az_{int(elevation)}el.png",
            )

        if overwrite:
            img.save(output_path)
        else:
            with open(output_path, "xb") as handle:
                img.save(handle, format="PNG")
        return json_content({
            "status": "ok",
            "resref": resref,
            "output_path": os.path.abspath(output_path),
            "width": width,
            "height": height,
            "backend": renderer.perf.get("backend", "unknown"),
        })
    except FileExistsError:
        return json_content({"error": "Refusing to overwrite existing render output."})
    except Exception as exc:
        # No traceback: it disclosed local paths and module layout to MCP clients.
        return json_content({"error": f"Render failed: {type(exc).__name__}: {str(exc)[:500]}"})


async def handle_model_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured model metadata as a ModelInfo contract."""
    resref = arguments.get("resref", "")
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    svc = _get_services()

    try:
        path_label, model = _locate_and_parse(resref, game, game_path, svc)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})
    except Exception as exc:
        return json_content({"error": f"Failed to parse MDL: {exc}"})

    try:
        info: ModelInfo = svc.analyzer.model_info(model, resref, path_label)
        # Convert frozen dataclass to dict for JSON serialisation
        d = {
            "resref": info.resref,
            "path": info.path,
            "node_count": info.node_count,
            "mesh_node_count": info.mesh_node_count,
            "total_vertices": info.total_vertices,
            "total_faces": info.total_faces,
            "bone_count": info.bone_count,
            "bones": info.bones,
            "animations": info.animations,
            "bounding_box": {
                "min": info.bounding_box_min,
                "max": info.bounding_box_max,
            },
            "classification": info.classification,
            "supermodel": info.supermodel,
        }
        return json_content(d)
    except Exception as exc:
        return json_content({"error": f"Info extraction failed: {exc}"})


async def handle_list_game_models(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List MDL resources in an installation."""
    game_alias = arguments.get("game")
    game_path = arguments.get("game_path")
    name_filter = (arguments.get("filter") or "").lower()
    limit = int(arguments.get("limit", 50))
    svc = _get_services()

    game = svc.registry.resolve(game_alias)
    if game is None:
        return json_content({"error": "Specify game (k1/k2)."})

    try:
        installation = svc.registry.load(game, game_path)
        results = []
        for entry in installation.iter_resources("all"):
            if entry.restype != "MDL":
                continue
            if name_filter and name_filter not in entry.resref.lower():
                continue
            results.append({
                "resref": entry.resref,
                "source": entry.source,
                "size": entry.size,
            })
            if len(results) >= limit:
                break
        return json_content({"count": len(results), "models": results})
    except Exception as exc:
        return json_content({"error": str(exc)})


async def handle_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Quick integrity check on a model — returns an AuditResult contract."""
    resref = arguments.get("resref", "")
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    svc = _get_services()

    try:
        path_label, model = _locate_and_parse(resref, game, game_path, svc)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})
    except Exception as exc:
        return json_content({"error": f"Failed to parse MDL: {exc}"})

    try:
        result: AuditResult = svc.analyzer.audit(model, resref)
        return json_content({
            "resref": result.resref,
            "status": result.status,
            "node_count": result.node_count,
            "mesh_node_count": result.mesh_node_count,
            "bounding_box_ok": result.bounding_box_ok,
            "issues": result.issues,
            "warnings": result.warnings,
        })
    except Exception as exc:
        return json_content({"error": f"Audit failed: {exc}"})


async def handle_export_model_for_unity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Export a KotOR model into a Unity project asset folder."""
    resref = str(arguments.get("resref", "") or "").strip()
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    unity_project_raw = str(arguments.get("unity_project", "") or "").strip()
    asset_subdir = str(
        arguments.get("asset_subdir") or "Assets/KotorImported/GhostRigger"
    )
    output_name = str(arguments.get("output_name", "") or "").strip() or None
    fmt = str(arguments.get("format") or "fbx").lower().lstrip(".")
    export_rigging = bool(arguments.get("export_rigging", True))
    export_textures = bool(arguments.get("export_textures", True))
    svc = _get_services()

    if not resref:
        return json_content({"error": "resref is required."})
    if not game:
        return json_content({"error": "game is required."})
    if not unity_project_raw:
        return json_content({"error": "unity_project is required."})
    if fmt != "fbx":
        return json_content({"error": f"Unsupported Unity transfer format: {fmt}"})

    try:
        path_label, model = _locate_and_parse(resref, game, game_path, svc)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})
    except Exception as exc:
        return json_content({"error": f"Failed to parse MDL: {exc}"})

    _resolve_fbx_base_skeleton(model, game, game_path, svc)
    try:
        model = _prepare_selected_fbx_animations(
            model,
            arguments,
            game=str(game),
            game_path=game_path,
            services=svc,
        )
    except Exception as exc:
        return json_content({"error": f"Animation selection failed: {exc}"})

    try:
        texture_cache = None
        if export_textures:
            factory = getattr(svc, "texture_cache_factory", None)
            if callable(factory):
                texture_cache = factory(game, game_path)

        try:
            from src.core.export.unity_export_bridge import (  # noqa: PLC0415
                build_output_paths,
                export_model_for_unity,
            )
        except ImportError:                                      # pragma: no cover - MCP path shim
            from core.export.unity_export_bridge import (  # type: ignore  # noqa: PLC0415
                build_output_paths,
                export_model_for_unity,
            )

        unity_exporter = (
            partial(_export_fbx_for_unity, tex_cache=texture_cache)
            if texture_cache is not None
            else _export_fbx_for_unity
        )

        expected_asset_path, _metadata_path = build_output_paths(
            Path(unity_project_raw),
            asset_subdir,
            output_name or resref,
            fmt,
        )
        if arguments.get("overwrite", True) is False:
            existing = _existing_export_artifacts(expected_asset_path)
            if existing:
                return json_content({
                    "error": "Refusing to overwrite existing export files.",
                    "existing": existing,
                })
        previous_pbr_stamp = _artifact_stamp(expected_asset_path.with_suffix(".pbr.json"))
        result = export_model_for_unity(
            model,
            game=str(game).upper(),
            resref=resref,
            asset_name=output_name,
            unity_project=Path(unity_project_raw),
            asset_subdir=asset_subdir,
            extension=fmt,
            export_rigging=export_rigging,
            exporter=unity_exporter,
            source_path=path_label,
        )
        asset_path = Path(result["asset"])
        texture_sidecars = (
            _texture_sidecar_inventory(
                asset_path,
                previous_manifest_stamp=previous_pbr_stamp,
                require_changed_manifest=True,
            )
            if export_textures
            else []
        )
        result["textures"] = {
            "requested": export_textures,
            "sidecars": texture_sidecars,
            "resolution": (
                texture_cache.summary()
                if texture_cache is not None
                and callable(getattr(texture_cache, "summary", None))
                else {}
            ),
        }
        return json_content(result)
    except Exception as exc:
        return json_content({"error": f"Unity export failed: {exc}"})


async def handle_export_model_for_unreal(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Export a KotOR model with the Unreal FBX or rigged-GLB profile."""
    resref = str(arguments.get("resref", "") or "").strip()
    head_resref = str(arguments.get("head_resref", "") or "").strip()
    body_texture = str(arguments.get("body_texture", "") or "").strip()
    head_texture = str(arguments.get("head_texture", "") or "").strip()
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    output_path_raw = str(arguments.get("output_path", "") or "").strip()
    export_dir_raw = str(arguments.get("export_dir", "") or "").strip()
    output_name = str(arguments.get("output_name", "") or "").strip()
    transfer_format = str(arguments.get("format", "fbx") or "fbx").strip().lower()
    export_rigging = bool(arguments.get("export_rigging", True))
    export_textures = bool(arguments.get("export_textures", True))
    svc = _get_services()

    if not resref:
        return json_content({"error": "resref is required."})
    if not game:
        return json_content({"error": "game is required."})
    if not output_path_raw and not export_dir_raw:
        return json_content({"error": "output_path or export_dir is required."})
    if transfer_format not in {"fbx", "glb"}:
        return json_content({"error": "format must be 'fbx' or 'glb'."})

    try:
        path_label, model = _locate_and_parse(resref, game, game_path, svc)
    except FileNotFoundError as exc:
        return json_content({"error": str(exc)})
    except Exception as exc:
        return json_content({"error": f"Failed to parse MDL: {exc}"})

    base_skeleton_model = _resolve_fbx_base_skeleton(model, game, game_path, svc)
    texture_overrides = {
        "body": {
            "resref": body_texture,
            "sections_changed": _apply_primary_skin_texture(model, body_texture),
        },
        "head": {"resref": head_texture, "sections_changed": 0},
    }
    head_model = None
    bas_report: Dict[str, Any] = {}
    if head_resref:
        try:
            head_path_label, head_model = _locate_and_parse(
                head_resref,
                game,
                game_path,
                svc,
            )
        except FileNotFoundError as exc:
            return json_content({"error": f"BAS head was not found: {exc}"})
        except Exception as exc:
            return json_content({"error": f"Failed to parse BAS head: {exc}"})
        texture_overrides["head"]["sections_changed"] = _apply_primary_skin_texture(
            head_model,
            head_texture,
        )
        try:
            composed_name = output_name or Path(resref).stem
            model, bas_report = _compose_bas_character(
                model,
                head_model,
                name=composed_name,
            )
            bas_report.update({
                "mode": "headless_body",
                "body_resref": resref,
                "head_resref": head_resref,
                "head_source": head_path_label,
            })
            if base_skeleton_model is not None:
                setattr(model, "_gr_fbx_base_skeleton_model", base_skeleton_model)
        except Exception as exc:
            return json_content({"error": f"BAS composition failed: {exc}"})
    try:
        model = _prepare_selected_fbx_animations(
            model,
            arguments,
            game=str(game),
            game_path=game_path,
            services=svc,
            supplemental_models=(head_model,) if head_model is not None else (),
        )
    except Exception as exc:
        return json_content({"error": f"Animation selection failed: {exc}"})

    if output_path_raw:
        output_path = Path(output_path_raw)
        expected_suffix = f".{transfer_format}"
        if output_path.suffix.lower() != expected_suffix:
            # Previously rewritten silently, which could replace a different file.
            return json_content({
                "error": f"output_path must end with {expected_suffix} for format '{transfer_format}'."
            })
    else:
        stem = output_name or Path(resref).stem
        if not _safe_output_stem(stem):
            return json_content({"error": "output_name must be a plain file stem."})
        output_path = Path(export_dir_raw) / f"{stem}.{transfer_format}"
    if arguments.get("overwrite", True) is False:
        existing = _existing_export_artifacts(output_path)
        if existing:
            return json_content({
                "error": "Refusing to overwrite existing export files.",
                "existing": existing,
            })

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        texture_cache = None
        if export_textures:
            factory = getattr(svc, "texture_cache_factory", None)
            if not callable(factory):
                raise RuntimeError("texture export provider is unavailable")
            texture_cache = factory(game, game_path)
        exporter = (
            _export_glb_for_unreal
            if transfer_format == "glb"
            else _export_fbx_for_unreal
        )
        previous_pbr_stamp = _artifact_stamp(output_path.with_suffix(".pbr.json"))
        if not exporter(model, output_path, export_rigging, tex_cache=texture_cache):
            raise RuntimeError(f"exporter returned False for {output_path}")
        if not output_path.exists():
            raise RuntimeError(f"exporter did not create {output_path}")
    except Exception as exc:
        return json_content({"error": f"Unreal export failed: {exc}"})

    manifest_path = output_path.with_suffix(".ghostrigger.json")
    texture_sidecars = (
        _texture_sidecar_inventory(
            output_path,
            previous_manifest_stamp=previous_pbr_stamp,
            require_changed_manifest=True,
        )
        if export_textures and transfer_format == "fbx"
        else []
    )
    embedded_pbr = (
        _glb_pbr_inventory(output_path)
        if export_textures and transfer_format == "glb"
        else {}
    )
    texture_resolution = (
        texture_cache.summary()
        if texture_cache is not None and callable(getattr(texture_cache, "summary", None))
        else {}
    )
    animations = [
        str(getattr(animation, "name", "") or "")
        for animation in (getattr(model, "animations", None) or ())
    ]
    return json_content({
        "status": "ok",
        "asset": str(output_path),
        "manifest": str(manifest_path) if manifest_path.exists() else "",
        "compatibility_profile": (
            "unreal_glb_intermediate" if transfer_format == "glb" else "unreal"
        ),
        "format": transfer_format,
        "bas": bas_report,
        "texture_overrides": texture_overrides,
        "animations": animations,
        "animation_selection": dict(
            getattr(model, "_gr_fbx_animation_selection", None) or {}
        ),
        "textures": {
            "requested": export_textures,
            "sidecars": texture_sidecars,
            "embedded": embedded_pbr,
            "resolution": texture_resolution,
        },
        "counts": {
            "animations": len(animations),
            "texture_sidecars": len(texture_sidecars),
            "embedded_pbr_materials": int(embedded_pbr.get("material_count", 0)),
        },
        "source": {
            "game": str(game).upper(),
            "resref": resref,
            "path": path_label,
        },
    })


async def handle_validate_unity_import(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Unity-side import facts against a GhostRigger transfer sidecar."""
    transfer_path = str(arguments.get("transfer_metadata_path", "") or "").strip()
    unity_summary = arguments.get("unity_summary") or {}
    output_raw = str(arguments.get("output_path", "") or "").strip()
    if not transfer_path:
        return json_content({"error": "transfer_metadata_path is required."})
    if not isinstance(unity_summary, dict):
        return json_content({"error": "unity_summary must be an object."})

    try:
        try:
            from src.core.export.unity_import_validator import validate_unity_import_file  # noqa: PLC0415
        except ImportError:                                      # pragma: no cover - MCP path shim
            from core.export.unity_import_validator import validate_unity_import_file  # type: ignore  # noqa: PLC0415

        manifest = validate_unity_import_file(
            Path(transfer_path),
            unity_summary,
            Path(output_raw) if output_raw else None,
        )
        return json_content(manifest)
    except Exception as exc:
        return json_content({"error": f"Unity import validation failed: {exc}"})


_UNITY_SMOKE_ARGUMENTS = frozenset({
    "unity_project", "scene_path", "asset_path", "instance_name", "output_path",
    "screenshot_dir", "screenshot_delay", "screenshot_width", "screenshot_height",
    "refresh_assets", "dry_run", "operation_id", "expected_port", "expected_editor_instance_id",
})


def _unity_smoke_error(message: str, code: str) -> Dict[str, Any]:
    return {**json_content({"error": message[:500], "code": code[:64]}), "isError": True}


async def handle_run_malak_unity_smoke(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Run the authenticated Unity MCP Ghost import smoke (defaults: Malak fixture).

    The endpoint and token come only from the project's bridge identity; there
    is no host/port argument. Failures are flagged isError without tracebacks.
    """
    try:
        try:
            from src.core.special import unity_malak_smoke as smoke  # noqa: PLC0415
        except ImportError:  # pragma: no cover - MCP path shim
            from core.special import unity_malak_smoke as smoke  # type: ignore  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - broken install
        return _unity_smoke_error(f"Unity smoke module unavailable: {type(exc).__name__}", "MODULE_UNAVAILABLE")

    args = arguments if isinstance(arguments, dict) else {}
    unknown = sorted(set(args) - _UNITY_SMOKE_ARGUMENTS)
    if unknown:
        return _unity_smoke_error(
            f"Unsupported argument(s): {', '.join(unknown)} (the bridge endpoint comes from the project's bridge identity)",
            "INVALID_ARGUMENT",
        )

    def text(name: str, default: Optional[str] = None) -> Optional[str]:
        value = args.get(name, default)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value

    def flag(name: str, default: bool) -> bool:
        value = args.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value

    try:
        output_raw = text("output_path")
        shots_raw = text("screenshot_dir")
        report = smoke.run_malak_main_menu_smoke(
            unity_project=text("unity_project") or "",
            scene_path=text("scene_path", smoke.DEFAULT_SCENE_PATH),
            asset_path=text("asset_path", smoke.DEFAULT_ASSET_PATH),
            instance_name=text("instance_name", smoke.DEFAULT_INSTANCE_NAME),
            output_path=output_raw,
            screenshot_dir=shots_raw,
            screenshot_delay=args.get("screenshot_delay", 1.0),
            screenshot_width=args.get("screenshot_width", 640),
            screenshot_height=args.get("screenshot_height", 360),
            refresh_assets=flag("refresh_assets", True),
            dry_run=flag("dry_run", False),
            operation_id=text("operation_id"),
            expected_port=args.get("expected_port"),
            expected_editor_instance_id=text("expected_editor_instance_id"),
        )
    except smoke.UnityBridgeError as exc:
        return _unity_smoke_error(str(exc), exc.code)
    except FileExistsError as exc:
        return _unity_smoke_error(str(exc), "OUTPUT_EXISTS")
    except ValueError as exc:
        return _unity_smoke_error(str(exc), "INVALID_ARGUMENT")
    except Exception as exc:  # noqa: BLE001 - no traceback or environment leaks
        return _unity_smoke_error(f"Unity smoke failed: {type(exc).__name__}: {str(exc)[:300]}", "SMOKE_FAILED")

    if report.get("dry_run") or not output_raw:
        return json_content(report, max_chars=120_000)
    # The full report is on disk; answer with a bounded summary and its hash.
    # The verdict is "smoke_ok" (not "ok"): a failing smoke is a completed run.
    summary = {
        key: report.get(key)
        for key in ("schema_version", "tool", "status", "dry_run", "operation_id", "scene_path",
                    "asset_path", "instance_name", "selected_clip", "bridge", "restore", "aborted")
    }
    summary["smoke_ok"] = bool(report.get("ok"))
    summary["issues"] = [item.get("code") for item in report.get("issues", [])]
    summary["warnings"] = [item.get("code") for item in report.get("warnings", [])]
    summary["screenshots"] = {
        name: (shot.get("copy") if isinstance(shot, dict) else None)
        for name, shot in (report.get("screenshots") or {}).items()
    }
    summary["report_path"] = output_raw
    summary["report_sha256"] = sha256(Path(output_raw).read_bytes()).hexdigest()
    return json_content(summary)


# ── Backward-compatible helper ────────────────────────────────────────────────

def resolve_game_safe(label: Optional[str]):
    """Resolve game label without raising (backward-compatible shim)."""
    return _get_services().registry.resolve(label)


# ── Source identity (read-only; used by Kaigen preview/apply wrappers) ─────────

_IDENTITY_MAX_FILE = 256 * 1024 * 1024


def _bytes_identity(data: Optional[bytes]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def _file_identity(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        size = path.stat().st_size
        if size > _IDENTITY_MAX_FILE:
            raise ValueError(f"Identity source exceeds bound: {path.name}")
        info.update(_bytes_identity(path.read_bytes()) or {})
    return info


def _install_identity(game_path: Optional[str]) -> Dict[str, Any]:
    if not game_path:
        return {}
    return {"chitin_key": _file_identity(Path(game_path) / "chitin.key")}


def _model_identity(resref: str, game: Optional[str], game_path: Optional[str], svc: _Services) -> Dict[str, Any]:
    path_label, mdl_bytes, mdx_bytes = svc.locator.locate(resref, game, game_path)
    return {
        "resref": resref,
        "path": str(path_label),
        "mdl": _bytes_identity(mdl_bytes),
        "mdx": _bytes_identity(mdx_bytes or b""),
    }


async def handle_source_identity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Hash the exact inputs an export would read, without writing anything."""
    mode = str(arguments.get("mode") or "")
    resref = str(arguments.get("resref") or "").strip()
    game = arguments.get("game")
    game_path = arguments.get("game_path")
    try:
        if mode == "model":
            if not resref or not game:
                raise ValueError("resref and game are required for mode=model")
            svc = _get_services()
            head_resref = str(arguments.get("head_resref") or "").strip()
            payload = {
                "mode": mode,
                "model": _model_identity(resref, game, game_path, svc),
                "head": _model_identity(head_resref, game, game_path, svc) if head_resref else None,
                "install": _install_identity(game_path),
            }
        elif mode == "debug_session":
            if not resref or not game_path:
                raise ValueError("resref and game_path are required for mode=debug_session")
            from kotormcp.tools.debug_skinning import session_source_identity  # noqa: PLC0415

            # Same resolution as ghostrigger_debug_load_model (the export
            # locator), from a throwaway probe; the live session is untouched.
            payload = {
                "mode": mode,
                **session_source_identity(resref, str(game_path)),
                "install": _install_identity(game_path),
            }
        elif mode == "retarget_day4":
            from src.core.retargeting import fbx_exporter, sampler  # noqa: PLC0415
            from src.core.retargeting.skeleton_renamer import DEFAULT_RENAME_MAP  # noqa: PLC0415
            from src.unreal.quinn import load_quinn_skeleton_asset  # noqa: PLC0415

            corpus = Path(sampler.DEFAULT_CORPUS_ROOT) / "k1"
            files = sorted(corpus.glob("*.md[lx]")) if corpus.is_dir() else []
            if len(files) > 512:
                raise ValueError("Day 4 corpus exceeds bound")
            asset = load_quinn_skeleton_asset()
            try:
                blender = str(fbx_exporter.find_blender_executable())
            except Exception as exc:  # pragma: no cover - depends on host
                blender = f"unavailable: {type(exc).__name__}"
            payload = {
                "mode": mode,
                "corpus_root": str(corpus),
                "corpus": [_file_identity(path) | {"path": path.name} for path in files],
                "blender_script": _file_identity(Path(fbx_exporter.BLENDER_SCRIPT)),
                "rename_map": _file_identity(Path(DEFAULT_RENAME_MAP)),
                "quinn": {
                    "bone_map": _file_identity(Path(asset.bone_map_path)),
                    "fbx": _file_identity(Path(asset.fbx_path)),
                },
                "blender_executable": blender,
            }
        else:
            raise ValueError("mode must be model, debug_session or retarget_day4")
        return json_content(payload, max_chars=120_000)
    except Exception as exc:
        return {
            **json_content({"error": f"{type(exc).__name__}: {str(exc)[:500]}"}),
            "isError": True,
        }


# Engine/render exports flag their {"error": ...} payloads as MCP errors.
handle_render_model = _flag_tool_errors(handle_render_model)
handle_export_model_for_unity = _flag_tool_errors(handle_export_model_for_unity)
handle_export_model_for_unreal = _flag_tool_errors(handle_export_model_for_unreal)
