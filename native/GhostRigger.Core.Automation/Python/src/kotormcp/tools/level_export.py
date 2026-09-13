"""Explicit saved-KMAP export for MCP clients; no desktop control implied."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import anyio

from kotormcp.utils import json_content
from src.core.level.kmap_serializer import KMapSerializer
from src.core.level.level_export_bridge import LevelExportOptions
from src.io.level_scene_export import export_level_scene


def get_tools():
    return [{
        "name": "ghoststudio_export_level",
        "description": (
            "Export all enabled visible rooms and resolved authored placed models from a SAVED "
            "Ghost Studio .kmap into one static .fbx or .obj plus materials, textures and a JSON "
            "manifest. Room placement is baked once. Save the Level Editor scene first: this "
            "tool does not click the desktop or read unsaved edits. Requires the matching KOTOR "
            "installation for stock assets. Not a playable .mod, animation or shader conversion. "
            "Output must be outside the game installation. Existing output is preserved unless "
            "overwrite=true is explicitly requested. Returns real output size and SHA-256."
        ),
        "inputSchema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "kmap_path": {"type": "string", "description": "Absolute path to the saved .kmap."},
                "output_path": {"type": "string", "description": "Absolute destination ending in .fbx or .obj."},
                "game_dir": {"type": "string", "description": "Absolute matching KOTOR 1/2 installation directory."},
                "asset_dir": {"type": "string", "description": "Optional directory of custom MDL/MDX, templates, 2DA and texture resources required by this saved level; immediate files only."},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["kmap_path", "output_path", "game_dir"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": False, "openWorldHint": False},
    }]


def _absolute_path(arguments, name):
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value).resolve()


def export_saved_level(arguments):
    """Resolve explicitly named input assets and invoke the shared IO owner."""
    from src.core.assets.resource_manager import ResourceManager

    if set(arguments) - {"kmap_path", "output_path", "game_dir", "asset_dir", "overwrite"}:
        raise ValueError("Unknown level-export argument")
    overwrite = arguments.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    source = _absolute_path(arguments, "kmap_path")
    target = _absolute_path(arguments, "output_path")
    game_dir = _absolute_path(arguments, "game_dir")
    if source.suffix.lower() != ".kmap" or not source.is_file():
        raise ValueError("kmap_path must name an existing saved .kmap")
    if target.suffix.lower() not in {".fbx", ".obj"}:
        raise ValueError("output_path must end in .fbx or .obj")
    if not game_dir.is_dir():
        raise ValueError("game_dir does not exist")
    if target.is_relative_to(game_dir):
        raise ValueError("Export outside the source game installation")
    if target.exists() and not overwrite:
        raise ValueError("Output exists; choose another name or explicitly set overwrite=true")
    project = KMapSerializer.load(source)
    game = str(project.game).upper()
    if game not in {"K1", "K2"}:
        raise ValueError("Saved KMAP must identify game K1 or K2")
    resources = []
    if arguments.get("asset_dir") is not None:
        asset_dir = _absolute_path(arguments, "asset_dir")
        if not asset_dir.is_dir():
            raise ValueError("asset_dir does not exist")
        extensions = {"mdl", "mdx", "utp", "utc", "utd", "2da", "tga", "tpc"}
        for path in sorted(asset_dir.iterdir()):
            kind = path.suffix.lower().lstrip(".")
            if path.is_file() and kind in extensions:
                resources.append((path.stem, kind, path.read_bytes()))
    manager = ResourceManager()
    setter = manager.set_k1_dir if game == "K1" else manager.set_k2_dir
    if not setter(str(game_dir)):
        raise ValueError(f"Could not load the {game} installation at game_dir")
    result = export_level_scene(project, target, LevelExportOptions(overwrite=overwrite),
                                resource_manager=manager, template_resources=tuple(resources),
                                protected_roots=(game_dir,))
    payload = asdict(result)
    payload.update({"source_kmap": str(source), "scope": "saved_kmap_static_geometry",
                    "desktop_control": False, "unsaved_edits_included": False})
    if result.ok:
        with target.open("rb") as stream:
            payload["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        payload["bytes"] = target.stat().st_size
        payload["geometry"] = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))["geometry"]
    return payload


async def handle_export_level(arguments):
    try:
        payload = await anyio.to_thread.run_sync(export_saved_level, dict(arguments))
    except Exception as exc:
        payload = {"ok": False, "code": "level_export_failed", "message": str(exc)}
    response = json_content(payload)
    response["isError"] = not payload["ok"]
    return response
