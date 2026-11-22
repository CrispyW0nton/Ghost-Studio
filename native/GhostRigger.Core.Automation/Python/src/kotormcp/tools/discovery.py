"""
Resource discovery tools: list, describe, find, search.

Architecture note (Khononov, "Balancing Coupling in Software Design"):
  Tools interact with KotOR installations through the InstallationPort
  contract (Contract Coupling).  The old _iter_resources() free function
  that reached into pykotor Installation internals has been replaced by
  installation.iter_resources() — all pykotor knowledge lives in adapters.py.

  _iter_resources() is kept as a thin shim for any internal callers (e.g.
  ghostrigger.handle_list_game_models) — it delegates to the port.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kotormcp.schemas import (
    DescribeResourceInput,
    FindResourceInput,
    ListResourcesInput,
    SearchResourcesInput,
)
from kotormcp.state import load_installation, resolve_game
from kotormcp.utils import json_content


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_pykotor():
    """Return pykotor symbols or raise ImportError with helpful message."""
    try:
        from pykotor.resource.type import ResourceType  # noqa: PLC0415
        from pykotor.extract.installation import SearchLocation  # noqa: PLC0415
        return ResourceType, SearchLocation
    except ImportError as exc:
        raise ImportError("pykotor is required for resource discovery.") from exc


GFF_HEAVY_EXTS = {
    "gff", "bic", "utc", "utd", "ute", "uti", "utp", "uts", "utm",
    "utt", "utw", "are", "git", "ifo", "dlg", "fac", "jrl",
}


def _parse_resource_types(raw: Optional[List[str]]):
    if not raw:
        return set()
    ResourceType, _ = _safe_pykotor()
    parsed = set()
    for token in raw:
        value = token.strip().upper()
        if not value:
            continue
        if value in ResourceType.__members__:
            parsed.add(ResourceType[value])
            continue
        cleaned = value.lstrip(".").lower()
        try:
            parsed.add(ResourceType.from_extension(cleaned))
        except (ValueError, KeyError):
            pass  # ignore unknown types
    return parsed


def _resource_snapshot(source: str, resource) -> Dict[str, Any]:
    try:
        identifier = resource.identifier()
        return {
            "resref": identifier.lower_resname if hasattr(identifier, "lower_resname") else resource.resname(),
            "type": resource.restype().name,
            "extension": resource.restype().extension,
            "size": resource.size(),
            "source": source,
        }
    except Exception as exc:
        return {"source": source, "error": str(exc)}


def _resolve_module_name(installation, alias: str) -> Optional[str]:
    alias_lower = alias.lower()
    try:
        modules = installation.modules_list()
        lookup = {name.lower(): name for name in modules}
        if alias_lower in lookup:
            return lookup[alias_lower]
        for candidate in modules:
            if alias_lower in candidate.lower():
                return candidate
    except Exception:
        pass
    return None


def _iter_resources(installation, location: str, module_filter: Optional[str]) -> Iterator[Tuple[str, Any]]:
    """
    Thin compatibility shim: delegate to the InstallationPort.iter_resources() contract.

    The installation argument may be either an InstallationPort (new path) or a
    raw pykotor Installation object (legacy path) — we handle both so existing
    callers don't need to change.
    """
    from kotormcp.ports import InstallationPort as _Port  # noqa: PLC0415

    if isinstance(installation, _Port):
        # New path: delegate to port contract (ResourceEntry objects)
        for entry in installation.iter_resources(location, module_filter):
            yield entry.source, _ResourceEntryProxy(entry)
        return

    # Legacy path: raw pykotor Installation — keep old logic intact
    lowered = location.lower()
    if lowered == "auto":
        lowered = "all"

    if lowered in {"override", "all"}:
        try:
            for r in installation.override_resources():
                yield "override", r
        except Exception:
            pass

    if lowered in {"core", "all"}:
        try:
            for r in installation.core_resources():
                yield "core", r
        except Exception:
            pass

    if lowered.startswith("module:"):
        _, module_alias = lowered.split(":", 1)
        resolved = _resolve_module_name(installation, module_alias)
        if resolved:
            try:
                for r in installation.module_resources(resolved):
                    yield f"module:{resolved}", r
            except Exception:
                pass
        return

    if lowered in {"modules", "all"}:
        try:
            for module_name in installation.modules_list():
                if module_filter and module_filter.lower() not in module_name.lower():
                    continue
                try:
                    for r in installation.module_resources(module_name):
                        yield f"module:{module_name}", r
                except Exception:
                    pass
        except Exception:
            pass

    if lowered in {"chitin", "bif", "all"}:
        try:
            for r in installation.chitin_resources():
                yield "chitin", r
        except Exception:
            pass


class _ResourceEntryProxy:
    """Adapts a ResourceEntry to the pykotor resource object interface used by snapshot helpers."""

    def __init__(self, entry):
        self._entry = entry

    def resname(self):
        return self._entry.resref

    def restype(self):
        return _ResourceTypeProxy(self._entry.restype, self._entry.extension)

    def size(self):
        return self._entry.size

    @property
    def data(self):
        return self._entry.data

    def identifier(self):
        return self


class _ResourceTypeProxy:
    def __init__(self, name: str, extension: str):
        self.name = name
        self.extension = extension

    def __eq__(self, other):
        if isinstance(other, _ResourceTypeProxy):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


def _summarize_gff(data: bytes) -> Dict[str, Any]:
    try:
        from pykotor.resource.formats.gff.gff_auto import read_gff  # noqa: PLC0415
        gff = read_gff(BytesIO(data))
        root = gff.root
        fields = []
        for label, field_type, value in root:
            preview: Any = value
            if isinstance(value, bytes):
                preview = f"<bytes:{len(value)}>"
            elif hasattr(value, "__len__") and len(str(value)) > 120:
                preview = f"{str(value)[:117]}..."
            fields.append({"label": label, "type": field_type.name, "preview": str(preview)})
        return {"struct_id": root.struct_id, "field_count": len(root), "fields": fields[:20]}
    except Exception as exc:
        return {"error": str(exc)}


def _summarize_2da(data: bytes) -> Dict[str, Any]:
    try:
        from pykotor.resource.formats.twoda.twoda_auto import read_2da  # noqa: PLC0415
        table = read_2da(data)
        rows = []
        for idx in range(min(table.get_height(), 10)):
            rows.append({"row": idx, "values": {h: table.get_cell_safe(idx, h, "") for h in table.get_headers()}})
        return {"columns": table.get_headers(), "row_count": table.get_height(), "sample": rows}
    except Exception as exc:
        return {"error": str(exc)}


def _describe_resource(result) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "resref": (getattr(result, "resref", None) or
                   (result.resname() if callable(getattr(result, "resname", None)) else result.resname)),
        "type": (result.restype if isinstance(result.restype, str) else
                 (result.restype() if callable(result.restype) else result.restype).name),
        "bytes": len(result.data),
    }
    try:
        summary["source"] = str(getattr(result, "filepath", getattr(result, "source", "")))
    except Exception:
        pass
    data = result.data
    ext = summary.get("type", "").lower()
    if ext in GFF_HEAVY_EXTS:
        summary["analysis"] = _summarize_gff(data)
    elif ext == "2da":
        summary["analysis"] = _summarize_2da(data)
    else:
        summary["analysis"] = {"size": len(data), "head": data[:64].hex()}
    return summary


# ── Tool definitions ──────────────────────────────────────────────────────────

def get_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "listResources",
            "description": "List resources from override/modules/chitin with optional filters. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "k1 or k2"},
                    "location": {"type": "string", "description": "override, modules, core, chitin, all"},
                    "moduleFilter": {"type": "string"},
                    "resourceTypes": {"type": "array", "items": {"type": "string"}},
                    "resrefQuery": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["game"],
            },
        },
        {
            "name": "describeResource",
            "description": "Summarize a single resource (GFF, TLK, 2DA) using resolution order. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string"},
                    "resref": {"type": "string"},
                    "restype": {"type": "string"},
                    "order": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["game", "resref", "restype"],
            },
        },
        {
            "name": "kotor_find_resource",
            "description": "Find first match for a resref or see all locations. Supports glob (e.g. 203tel*). Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string"},
                    "query": {"type": "string"},
                    "all_locations": {"type": "boolean", "default": True},
                },
                "required": ["game", "query"],
            },
        },
        {
            "name": "kotor_search_resources",
            "description": "Search resource names by regex. Paginated. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string"},
                    "pattern": {"type": "string"},
                    "location": {"type": "string", "default": "all"},
                    "limit": {"type": "integer", "default": 50},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["game", "pattern"],
            },
        },
    ]


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_list_resources(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = ListResourcesInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game parameter (k1/k2)."})
    try:
        installation = load_installation(game)
    except Exception as exc:
        return json_content({"error": str(exc)})

    location = getattr(inp, "location", "all") or "all"
    module_filter = getattr(inp, "moduleFilter", None)
    type_filters = _parse_resource_types(getattr(inp, "resourceTypes", None))
    resref_query = (getattr(inp, "resrefQuery", None) or "").lower()
    limit = getattr(inp, "limit", 50) or 50
    offset = getattr(inp, "offset", 0) or 0

    results: List[Dict[str, Any]] = []
    skipped = 0
    for source, resource in _iter_resources(installation, location, module_filter):
        try:
            name = resource.resname().lower()
            rtype = resource.restype()
        except Exception:
            continue
        if resref_query and resref_query not in name:
            continue
        if type_filters and rtype not in type_filters:
            continue
        if skipped < offset:
            skipped += 1
            continue
        results.append(_resource_snapshot(source, resource))
        if len(results) >= limit:
            break

    has_more = len(results) == limit
    return json_content({
        "count": len(results),
        "offset": offset,
        "items": results,
        "has_more": has_more,
        "next_offset": (offset + len(results)) if has_more else None,
    })


async def handle_describe_resource(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = DescribeResourceInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game parameter (k1/k2)."})
    try:
        ResourceType, SearchLocation = _safe_pykotor()
        installation = load_installation(game)
        restype = ResourceType.from_extension(inp.restype.lstrip(".").lower())
        if str(restype) == "INVALID":
            return json_content({"error": f"Unknown resource type: {inp.restype}"})
        order = [
            SearchLocation.OVERRIDE,
            SearchLocation.MODULES,
            SearchLocation.CHITIN,
        ]
        result = installation.get_resource(inp.resref, inp.restype.lstrip(".").lower(),
                                           order=["override", "modules", "chitin"])
        if result is None:
            return json_content({"error": f"{inp.resref}.{restype.extension} not found."})
        return json_content(_describe_resource(result))
    except Exception as exc:
        return json_content({"error": str(exc)})


async def handle_find_resource(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = FindResourceInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game parameter (k1/k2)."})
    try:
        ResourceType, SearchLocation = _safe_pykotor()
        installation = load_installation(game)
        query = inp.query.strip()
        # Simple implementation: search all locations for matching resources
        matches = []
        for source, resource in _iter_resources(installation, "all", None):
            name = resource.resname().lower()
            if "*" in query or "?" in query:
                import fnmatch  # noqa: PLC0415
                if fnmatch.fnmatch(name, query.lower().split(".")[0]):
                    matches.append(_resource_snapshot(source, resource))
            else:
                q_lower = query.lower()
                if "." in q_lower:
                    q_name, q_ext = q_lower.rsplit(".", 1)
                    if name == q_name and resource.restype().extension.lower() == q_ext:
                        matches.append(_resource_snapshot(source, resource))
                elif name == q_lower:
                    matches.append(_resource_snapshot(source, resource))
            if not getattr(inp, "all_locations", True) and matches:
                break
        return json_content({"count": len(matches), "matches": matches[:200]})
    except Exception as exc:
        return json_content({"error": str(exc)})


async def handle_search_resources(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = SearchResourcesInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game parameter (k1/k2)."})
    try:
        pattern_re = re.compile(inp.pattern, re.IGNORECASE)
    except re.error as exc:
        return json_content({"error": f"Invalid regex: {exc}"})
    try:
        installation = load_installation(game)
    except Exception as exc:
        return json_content({"error": str(exc)})

    location = getattr(inp, "location", "all") or "all"
    limit = getattr(inp, "limit", 50) or 50
    offset = getattr(inp, "offset", 0) or 0

    items: List[Dict[str, Any]] = []
    skipped = 0
    for source, resource in _iter_resources(installation, location, None):
        try:
            name = resource.resname()
        except Exception:
            continue
        if not pattern_re.search(name):
            continue
        if skipped < offset:
            skipped += 1
            continue
        items.append(_resource_snapshot(source, resource))
        if len(items) >= limit:
            break

    has_more = len(items) == limit
    return json_content({
        "count": len(items),
        "offset": offset,
        "items": items,
        "has_more": has_more,
        "next_offset": (offset + len(items)) if has_more else None,
    })
