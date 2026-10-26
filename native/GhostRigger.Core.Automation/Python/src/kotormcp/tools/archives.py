"""Archive operations: list_archive, extract_resource.

Ported from OldRepublicDevs/KotorMCP (Tools/KotorMCP/src/kotormcp/tools/archives.py)
and adapted to GhostRigger's port-contract layer (ports.py / adapters.py).

Design note (Constantine, §11 — coupling):
  Both tools use data coupling only: each receives a plain dict and returns
  a JSON payload.  The extract tool is the FIRST write-capable tool in
  GhostRigger KotorMCP; it writes to a path-validated location on disk.

Tools exposed:
  kotor_list_archive      — list KEY/BIF/RIM/ERF/MOD contents (read-only)
  kotor_extract_resource  — extract a resource to disk (write: disk output)

Extraction contract (2026-09-30, Kaigen game_data group):
  * ``game_path`` names the exact installation; it is indexed afresh for every
    call (the game-keyed registry cache could serve another directory or an
    index that predates Override changes).
  * The response names the location that supplied the bytes (override,
    modules, chitin, ...), the container file relative to the installation,
    the size and the SHA-256. ``dry_run`` reports exactly that and writes
    nothing.
  * ``output_dir`` writes ``<resref>.<ext>`` inside that existing directory;
    the legacy ``output_path`` stays confined under PYKOTOR_EXTRACT_DIR (or the
    working directory). Files are created exclusively unless ``overwrite`` is
    true. Errors are valid JSON and flagged ``isError``.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from kotormcp.schemas import ListArchiveInput
from kotormcp.state import load_installation, resolve_game
from kotormcp.utils.formatting import json_content
from kotormcp.utils.path_safety import get_extract_base, resolve_and_validate_under_base

MAX_EXTRACT_BYTES = 64 * 1024 * 1024
_RESREF = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
_RESTYPE = re.compile(r"^\.?[A-Za-z0-9]{1,8}$")
# Locations a single lookup may be restricted to (pykotor SearchLocation names).
EXTRACT_SOURCES = (
    "OVERRIDE", "MODULES", "CHITIN", "TEXTURES_TPA", "TEXTURES_TPB", "TEXTURES_TPC",
    "TEXTURES_GUI", "MUSIC", "SOUND", "VOICE", "LIPS", "RIMS",
)
DEFAULT_EXTRACT_ORDER = ("OVERRIDE", "MODULES", "CHITIN")
_EXTRACT_KEYS = {
    "game", "resref", "restype", "game_path", "source", "output_dir", "output_path", "dry_run", "overwrite",
}


def _parse_restype(restype_str: str):
    """Parse restype string to ResourceType (None when unknown or unavailable)."""
    try:
        from pykotor.resource.type import ResourceType  # type: ignore[import]
        ext = restype_str.strip().lstrip(".").lower()
        restype = ResourceType.from_extension(ext)
    except ImportError:
        return None
    if restype is None or int(getattr(restype, "type_id", -1)) < 0 or not getattr(restype, "extension", ""):
        return None
    return restype


def _error(message: str) -> Dict[str, Any]:
    return {**json_content({"error": str(message)[:500]}), "isError": True}


def get_tools() -> List[Dict[str, Any]]:
    """Return tool definitions for archive operations."""
    return [
        {
            "name": "kotor_list_archive",
            "description": (
                "Use when you need to list contents of a KEY/BIF/RIM/ERF/MOD file "
                "with pagination. Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to archive file"},
                    "key_file": {"type": "string", "description": "Path to KEY file (for BIF listing)"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "kotor_extract_resource",
            "description": (
                "Resolve resref.restype in one KOTOR installation (Override, modules, chitin, or the one "
                "location named by 'source') and report the location, container file, size and SHA-256. "
                "dry_run writes nothing; otherwise writes <resref>.<ext> into an existing output_dir "
                "(or the legacy output_path under PYKOTOR_EXTRACT_DIR) exclusively unless overwrite is "
                "true. Never writes the installation. [destructiveHint: writes to disk]"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "enum": ["k1", "k2"], "description": "Game alias: k1 or k2"},
                    "resref": {"type": "string", "pattern": _RESREF.pattern, "description": "Resource reference name"},
                    "restype": {"type": "string", "pattern": _RESTYPE.pattern, "description": "Resource type extension"},
                    "game_path": {
                        "type": "string",
                        "maxLength": 1024,
                        "description": "Absolute installation directory (contains chitin.key); indexed afresh per call",
                    },
                    "source": {
                        "type": "string",
                        "enum": list(EXTRACT_SOURCES),
                        "description": "Optional: resolve only from this location (default order OVERRIDE, MODULES, CHITIN)",
                    },
                    "output_dir": {
                        "type": "string",
                        "maxLength": 1024,
                        "description": "Existing absolute directory; the file is written as <resref>.<ext>",
                    },
                    "output_path": {
                        "type": "string",
                        "maxLength": 1024,
                        "description": "Legacy output file or directory, confined under PYKOTOR_EXTRACT_DIR (or the cwd)",
                    },
                    "dry_run": {"type": "boolean", "default": False, "description": "Resolve and report only"},
                    "overwrite": {"type": "boolean", "default": False, "description": "Replace an existing output file"},
                },
                "required": ["game", "resref", "restype"],
                "additionalProperties": False,
            },
        },
    ]


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_list_archive(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List archive contents with pagination."""
    inp = ListArchiveInput.model_validate(arguments)

    try:
        from pykotor.tools.archives import list_bif, list_erf, list_key, list_rim  # type: ignore[import]
    except ImportError as e:
        return json_content({"error": f"pykotor not available: {e}"})

    path = Path(inp.file_path)
    if not path.exists():
        return json_content({"error": f"Archive file not found: {path}"})

    suffix = path.suffix.lower()
    items: List[Dict[str, Any]] = []
    try:
        if suffix == ".key":
            bif_files, resources = list_key(path)
            for resref, restype_ext, bif_index, res_index in resources:
                items.append({
                    "resref": resref,
                    "type": restype_ext,
                    "bif_index": bif_index,
                    "res_index": res_index,
                    "bif_file": bif_files[bif_index] if bif_index < len(bif_files) else None,
                })
        elif suffix == ".bif":
            key_path = Path(inp.key_file) if inp.key_file else path.parent / "chitin.key"
            for ar in list_bif(path, key_path=key_path if key_path.exists() else None):
                items.append({"resref": str(ar.resref), "type": ar.restype.extension, "size": ar.size})
        elif suffix == ".rim":
            for ar in list_rim(path):
                items.append({"resref": str(ar.resref), "type": ar.restype.extension, "size": ar.size})
        elif suffix in (".erf", ".mod", ".sav", ".hak"):
            for ar in list_erf(path):
                items.append({"resref": str(ar.resref), "type": ar.restype.extension, "size": ar.size})
        else:
            return json_content({
                "error": f"Unsupported archive type: {suffix}. Use .key, .bif, .rim, .erf, .mod, .sav, .hak",
            })
    except Exception as e:
        return json_content({"error": str(e)})

    limit = getattr(inp, "limit", 50) or 50
    offset = getattr(inp, "offset", 0) or 0
    total = len(items)
    page = items[offset: offset + limit]
    has_more = total > offset + limit
    next_offset = offset + len(page) if has_more else None
    return json_content({
        "total": total,
        "count": len(page),
        "offset": offset,
        "items": page,
        "has_more": has_more,
        "next_offset": next_offset,
    })


def _installation_root(value: Any, game_alias: str) -> Path:
    text = str(value or "").strip()
    if not text or len(text) > 1024:
        raise ValueError("game_path must be an absolute installation directory")
    requested = Path(text)
    if not requested.is_absolute():
        raise ValueError("game_path must be an absolute installation directory")
    root = requested.resolve(strict=True)
    if not root.is_dir() or requested.is_symlink():
        raise ValueError("game_path must be an existing installation directory")
    chitin = root / "chitin.key"
    executable = root / ("swkotor2.exe" if game_alias == "k2" else "swkotor.exe")
    if not chitin.is_file() or chitin.is_symlink() or not executable.is_file():
        raise ValueError(f"game_path is not a {game_alias.upper()} installation (chitin.key and {executable.name})")
    return root


def _location_label(root: Optional[Path], filepath: Any) -> Optional[str]:
    if filepath is None:
        return None
    file = Path(str(filepath))
    if root is not None:
        try:
            return file.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
    return file.name


def _resolve_resource(game: Any, game_path: Optional[Path], resref: str, restype: Any, source: Optional[str]):
    """Return {location, file, data} for the first location that has the resource, or None.

    Locations are tried one at a time in order, so the reported location is
    the one that actually supplied the bytes.
    """
    locations = [source] if source else list(DEFAULT_EXTRACT_ORDER)
    if game_path is None:
        adapter = load_installation(game)
        inner = getattr(adapter, "_inst", None)
        if inner is None:  # non-pykotor adapter: no per-location detail
            entry = adapter.get_resource(resref, restype.extension, order=[source] if source else None)
            if entry is None:
                return None
            return {"location": (source or "installation").lower(), "file": None, "data": bytes(entry.data or b"")}
        game_path = Path(str(inner.path())).resolve()
    from pykotor.extract.installation import Installation, SearchLocation  # type: ignore[import]

    installation = Installation(game_path)
    for name in locations:
        result = installation.resource(resref, restype, order=[getattr(SearchLocation, name)])
        if result is None:
            continue
        data = result.data() if callable(getattr(result, "data", None)) else getattr(result, "data", b"")
        return {
            "location": name.lower(),
            "file": _location_label(game_path, getattr(result, "filepath", None)),
            "data": bytes(data or b""),
            "game_path": str(game_path),
        }
    return None


def _write_exclusive(target: Path, data: bytes, overwrite: bool) -> None:
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError("Output file already exists; pass overwrite=true to replace it")
        if not target.is_file() or target.is_symlink():
            raise ValueError("Output path is not a regular file")
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return
    with target.open("xb") as handle:
        handle.write(data)


async def handle_extract_resource(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a resource in one installation, report its identity and optionally write it."""
    arguments = dict(arguments or {})
    unknown = sorted(set(arguments) - _EXTRACT_KEYS)
    if unknown:
        return _error(f"Unsupported argument: {', '.join(unknown)}")
    game_alias = str(arguments.get("game") or "").strip().lower()
    if game_alias not in {"k1", "k2"}:
        return _error("Unknown game alias. Use k1 or k2.")
    game = resolve_game(game_alias)
    if game is None:
        return _error("Unknown game alias. Use k1 or k2.")
    resref = arguments.get("resref")
    if not isinstance(resref, str) or not _RESREF.fullmatch(resref):
        return _error("resref must be 1-16 letters, digits, '_' or '-'")
    resref = resref.lower()
    raw_type = arguments.get("restype")
    if not isinstance(raw_type, str) or not _RESTYPE.fullmatch(raw_type):
        return _error("restype must be a resource type extension")
    restype = _parse_restype(raw_type)
    if restype is None:
        return _error(f"Unknown resource type: {raw_type}")
    source = arguments.get("source")
    if source is not None:
        source = str(source).strip().upper()
        if source not in EXTRACT_SOURCES:
            return _error(f"Unknown resource source: {arguments.get('source')}")
    for flag in ("dry_run", "overwrite"):
        if flag in arguments and not isinstance(arguments[flag], bool):
            return _error(f"{flag} must be a boolean")
    dry_run = arguments.get("dry_run") is True
    overwrite = arguments.get("overwrite") is True
    output_dir = arguments.get("output_dir")
    output_path = arguments.get("output_path")
    if not dry_run and (output_dir is None) == (output_path is None):
        return _error("Provide exactly one of output_dir or output_path (or dry_run=true)")

    try:
        game_path = (
            None if arguments.get("game_path") in (None, "") else _installation_root(arguments["game_path"], game_alias)
        )
        found = _resolve_resource(game, game_path, resref, restype, source)
        if found is None:
            where = source.lower() if source else "override, modules or chitin"
            return _error(f"Resource {resref}.{restype.extension} not found in {where}.")
        data = found["data"]
        if len(data) > MAX_EXTRACT_BYTES:
            return _error("Resource exceeds the 64 MiB extraction bound")
        report: Dict[str, Any] = {
            "status": "ok",
            "game": game_alias,
            "resref": resref,
            "restype": restype.extension.lower(),
            "location": found["location"],
            "source_file": found["file"],
            "game_path": found.get("game_path"),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "dry_run": dry_run,
        }
        if dry_run:
            return json_content(report)

        filename = f"{resref}.{restype.extension.lower()}"
        if output_dir is not None:
            requested = Path(str(output_dir))
            if not requested.is_absolute():
                return _error("output_dir must be an absolute directory")
            base = requested.resolve(strict=True)
            if not base.is_dir() or requested.is_symlink():
                return _error("output_dir must be an existing directory")
            if game_path is not None and (base == game_path or game_path in base.parents):
                return _error("output_dir must be outside the game installation")
            out_path = resolve_and_validate_under_base(base / filename, base, allow_nonexistent=True)
        else:
            out_path = Path(str(output_path))
            ext = f".{restype.extension.lower()}"
            if out_path.suffix.lower() != ext:
                out_path = out_path / filename if out_path.is_dir() else out_path.with_suffix(ext)
            try:
                out_path = resolve_and_validate_under_base(out_path, get_extract_base(), allow_nonexistent=True)
            except ValueError as e:
                return _error(f"Output path rejected: {e}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive(out_path, data, overwrite)
        if hashlib.sha256(out_path.read_bytes()).hexdigest() != report["sha256"]:
            return _error("Extracted file failed readback")
        report["path"] = str(out_path)
        return json_content(report)
    except FileExistsError as e:
        return _error(str(e))
    except Exception as e:
        return _error(f"{type(e).__name__}: {e}")
