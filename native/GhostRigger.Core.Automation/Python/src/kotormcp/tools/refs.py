"""Reference and plot tools: list_references, find_referrers, describe_dlg/jrl.

Ported from OldRepublicDevs/KotorMCP (Tools/KotorMCP/src/kotormcp/tools/refs.py)
and adapted to GhostRigger's port-contract layer (ports.py / adapters.py).

Design note (Constantine, §11 — Integration Coupling):
  All six tools are Transform-Analysis style: each has a single input domain
  (resref + game) and emits a JSON payload.  No tool name encodes the call
  context (Discord bot, VS Code, CI pipeline) — names are data addresses.

Tools exposed:
  kotor_list_references        — outbound refs from any GFF resource
  kotor_find_referrers         — what references a given resref / tag / script
  kotor_find_strref_referrers  — what resources use a TLK strref
  kotor_describe_dlg           — DLG structure: entry/reply counts, script refs
  kotor_describe_jrl           — JRL summary: category & entry counts
  kotor_describe_resource_refs — generic GFF reference extractor
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List

from kotormcp.schemas import (
    ListReferencesInput,
    FindReferrersInput,
    FindStrrefReferrersInput,
    DescribeDlgInput,
    DescribeJrlInput,
    DescribeResourceRefsInput,
)
from kotormcp.state import load_installation, resolve_game
from kotormcp.utils.formatting import json_content


def _parse_restype(restype: str):
    """Parse restype string to ResourceType (pykotor)."""
    try:
        from pykotor.resource.type import ResourceType  # type: ignore[import]
        value = restype.strip().upper()
        if value in ResourceType.__members__:
            return ResourceType[value]
        cleaned = value.lstrip(".").lower()
        return ResourceType.from_extension(cleaned)
    except ImportError:
        return None


def _canonical_order():
    """Return canonical search order (override → modules → chitin)."""
    try:
        from pykotor.tools.finder import canonical_search_order  # type: ignore[import]
        return canonical_search_order()
    except ImportError:
        return None



def _err(msg: str) -> Dict[str, Any]:
    """Return a canonical error result in {"type":"text","text":"<json>"} format.

    Uses json.dumps() so special characters in msg cannot break JSON parsing.
    """
    return {"type": "text", "text": json.dumps({"error": msg})}

def get_tools() -> List[Dict[str, Any]]:
    """Return tool definitions for reference and plot analysis (read-only).

    Architecture note (Khononov, Contract Coupling):
      All tools depend on InstallationPort.get_resource() and the stable
      pykotor.tools.references API.  No volatile implementation details leak
      across this boundary.
    """
    return [
        {
            "name": "kotor_list_references",
            "description": (
                "Use when tracing what a resource references: list outbound refs "
                "(scripts, conversations, tags, template resrefs) by field path. "
                "Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "resref": {"type": "string", "description": "Resource reference name"},
                    "restype": {"type": "string", "description": "Resource type (e.g. DLG, UTC)"},
                    "path": {"type": "string", "description": "Optional installation path override"},
                },
                "required": ["game", "resref", "restype"],
            },
        },
        {
            "name": "kotor_find_referrers",
            "description": (
                "Use when you need to find which resources reference a script resref, tag, "
                "conversation, or resref. Use module_root to narrow scope; expensive over full "
                "install. Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "value": {"type": "string", "description": "Script resref, tag, conversation resref, or resref to search for"},
                    "reference_kind": {
                        "type": "string",
                        "enum": ["script", "tag", "conversation", "resref"],
                        "default": "resref",
                        "description": "Kind of reference",
                    },
                    "path": {"type": "string", "description": "Optional installation path override"},
                    "module_root": {"type": "string", "description": "Limit search to this module"},
                    "partial_match": {"type": "boolean", "default": False},
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, match is case-sensitive (PyKotor default: case-insensitive).",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["game", "value"],
            },
        },
        {
            "name": "kotor_find_strref_referrers",
            "description": (
                "Use when you need to find which resources use a TLK strref "
                "(TLK/2DA Find References parity). Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "strref": {"type": "integer", "minimum": 0, "description": "TLK string reference ID"},
                    "path": {"type": "string", "description": "Optional installation path override"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "required": ["game", "strref"],
            },
        },
        {
            "name": "kotor_describe_dlg",
            "description": (
                "Use when you need DLG structure: entry/reply counts and script/condition refs. "
                "Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "resref": {"type": "string", "description": "DLG resource reference"},
                    "path": {"type": "string", "description": "Optional installation path override"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "kotor_describe_jrl",
            "description": (
                "Use when you need a JRL (journal) summary: categories and entries. "
                "Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "resref": {"type": "string", "description": "JRL resource reference (e.g. global)"},
                    "path": {"type": "string", "description": "Optional installation path override"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "kotor_describe_resource_refs",
            "description": (
                "Use when you need a reference summary for any GFF resource "
                "(scripts, conversations, tags, template resrefs). Read-only."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "Game alias: k1 or k2"},
                    "resref": {"type": "string", "description": "Resource reference name"},
                    "restype": {"type": "string", "description": "Resource type (e.g. UTC, ARE)"},
                    "path": {"type": "string", "description": "Optional installation path override"},
                },
                "required": ["game", "resref", "restype"],
            },
        },
    ]


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_list_references(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """List outbound references from a single GFF resource."""
    inp = ListReferencesInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.resource.formats.gff import read_gff  # type: ignore[import]
        from pykotor.tools.references import extract_references  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        restype = _parse_restype(inp.restype)
        if restype is None:
            return _err(f"Unknown restype: {inp.restype}")

        entry = installation.get_resource(inp.resref, restype.name)
        if entry is None:
            return _err(f"Resource {inp.resref}.{inp.restype} not found.")

        gff = read_gff(bytes(entry.data))
        file_type = restype.extension.upper()
        refs = extract_references(gff, file_type)
        out = [{"ref_kind": r.ref_kind, "value": r.value, "field_path": r.field_path} for r in refs]
        return json_content({"resref": inp.resref, "restype": inp.restype, "references": out, "count": len(out)})
    except Exception as e:
        return _err(str(e))


async def handle_find_referrers(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Find resources that reference the given value."""
    inp = FindReferrersInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.tools.references import find_referrers  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        # Unwrap the adapter to get the raw installation object for find_referrers
        raw_inst = getattr(installation, "_installation", installation)
        kind = (getattr(inp, "reference_kind", None) or "resref").lower()
        if kind not in ("script", "tag", "conversation", "resref"):
            kind = "resref"
        results = find_referrers(
            raw_inst,
            inp.value,
            reference_kind=kind,
            module_root=getattr(inp, "module_root", None) or None,
            partial_match=getattr(inp, "partial_match", False),
            file_types=None,
        )
        total = len(results)
        offset = getattr(inp, "offset", 0) or 0
        limit = getattr(inp, "limit", 100) or 100
        start = min(offset, total)
        end = min(start + limit, total)
        page = results[start:end]
        items = [
            {
                "resref": r.file_resource.resname(),
                "restype": r.file_resource.restype().name,
                "field_path": r.field_path,
                "matched_value": r.matched_value,
                "filepath": str(r.file_resource.filepath()),
            }
            for r in page
        ]
        return json_content({
            "count": len(items),
            "offset": start,
            "total": total,
            "has_more": end < total,
            "items": items,
        })
    except Exception as e:
        return _err(str(e))


async def handle_find_strref_referrers(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Find resources that reference a TLK strref."""
    inp = FindStrrefReferrersInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.tools.reference_finder import find_field_value_references  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        raw_inst = getattr(installation, "_installation", installation)
        results = find_field_value_references(
            raw_inst,
            str(inp.strref),
            partial_match=False,
            case_sensitive=True,
            file_pattern=None,
            file_types=None,
        )
        total = len(results)
        offset = getattr(inp, "offset", 0) or 0
        limit = getattr(inp, "limit", 100) or 100
        start = min(offset, total)
        end = min(start + limit, total)
        page = results[start:end]
        items = [
            {
                "resref": r.file_resource.resname(),
                "restype": r.file_resource.restype().name,
                "field_path": r.field_path,
                "matched_value": r.matched_value,
                "filepath": str(r.file_resource.filepath()),
            }
            for r in page
        ]
        return json_content({
            "count": len(items),
            "offset": start,
            "total": total,
            "has_more": end < total,
            "items": items,
        })
    except Exception as e:
        return _err(str(e))


def _dlg_summary(gff: Any) -> Dict[str, Any]:
    """Build a short DLG summary from root (entry/reply counts, script refs)."""
    try:
        from pykotor.resource.formats.gff.gff_data import GFFList  # type: ignore[import]
        try:
            from pykotor.tools.references import extract_references
            reference_scope = "pykotor full extractor"
        except ImportError:
            # Ghost's installed PyKotor omits tools.references. Traverse the
            # already-parsed typed DLG tree; this is not another binary reader.
            from types import SimpleNamespace
            from pykotor.resource.formats.gff import GFFFieldType
            def extract_references(document, _resource_type):
                rows = []
                scripts = {"Script", "Script2", "Active", "Active2", "EndConversation", "EndConverAbort"}
                def visit(node):
                    for label, kind, value in node:
                        if kind == GFFFieldType.Struct:
                            visit(value)
                        elif kind == GFFFieldType.List:
                            for child in value:
                                visit(child)
                        elif label in scripts and str(value):
                            rows.append(SimpleNamespace(ref_kind="script", value=str(value)))
                        elif label == "Conversation" and str(value):
                            rows.append(SimpleNamespace(ref_kind="conversation", value=str(value)))
                visit(document.root)
                return rows
            reference_scope = "typed DLG script/condition/conversation fields only"
        root = gff.root
        entry_list = root.get("EntryList")
        reply_list = root.get("ReplyList")
        entry_count = len(entry_list) if isinstance(entry_list, GFFList) else 0
        reply_count = len(reply_list) if isinstance(reply_list, GFFList) else 0
        refs = extract_references(gff, "DLG")
        scripts = [r.value for r in refs if r.ref_kind == "script"]
        conversations = [r.value for r in refs if r.ref_kind == "conversation"]
        return {
            "entry_count": entry_count,
            "reply_count": reply_count,
            "script_refs": list(dict.fromkeys(scripts)),
            "conversation_refs": list(dict.fromkeys(conversations)),
            "reference_count": len(refs),
            "reference_scope": reference_scope,
        }
    except Exception as e:
        return {"error": str(e)}


def _jrl_summary(gff: Any) -> Dict[str, Any]:
    """Build a short JRL summary (categories, entry count)."""
    try:
        from pykotor.resource.formats.gff.gff_data import GFFList  # type: ignore[import]
        root = gff.root
        categories = root.get("Categories")
        entry_list = root.get("EntryList")
        cat_count = len(categories) if isinstance(categories, GFFList) else 0
        entry_count = len(entry_list) if isinstance(entry_list, GFFList) else 0
        return {"category_count": cat_count, "entry_count": entry_count}
    except Exception as e:
        return {"error": str(e)}


async def handle_describe_dlg(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Describe a DLG resource (entries, replies, script/conversation refs)."""
    inp = DescribeDlgInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.resource.formats.gff import read_gff  # type: ignore[import]
        from pykotor.resource.type import ResourceType  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        entry = installation.get_resource(inp.resref, "DLG")
        if entry is None:
            return _err(f"DLG {inp.resref} not found.")
        gff = read_gff(bytes(entry.data))
        summary = _dlg_summary(gff)
        summary["resref"] = inp.resref
        return json_content(summary)
    except Exception as e:
        return _err(str(e))


async def handle_describe_jrl(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Describe a JRL resource (categories, entries)."""
    inp = DescribeJrlInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.resource.formats.gff import read_gff  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        entry = installation.get_resource(inp.resref, "JRL")
        if entry is None:
            return _err(f"JRL {inp.resref} not found.")
        gff = read_gff(bytes(entry.data))
        summary = _jrl_summary(gff)
        summary["resref"] = inp.resref
        return json_content(summary)
    except Exception as e:
        return _err(str(e))


async def handle_describe_resource_refs(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Generic GFF reference summary (extract_references)."""
    inp = DescribeResourceRefsInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return _err("Unknown game alias. Use k1 or k2.")

    try:
        from pykotor.resource.formats.gff import read_gff  # type: ignore[import]
        from pykotor.tools.references import extract_references  # type: ignore[import]
    except ImportError as e:
        return _err(f"pykotor not available: {e}")

    try:
        installation = load_installation(game, getattr(inp, "path", None))
        restype = _parse_restype(inp.restype)
        if restype is None:
            return _err(f"Unknown restype: {inp.restype}")

        entry = installation.get_resource(inp.resref, restype.name)
        if entry is None:
            return _err(f"Resource {inp.resref}.{inp.restype} not found.")

        gff = read_gff(bytes(entry.data))
        file_type = restype.extension.upper()
        refs = extract_references(gff, file_type)
        out = [{"ref_kind": r.ref_kind, "value": r.value, "field_path": r.field_path} for r in refs]
        return json_content({"resref": inp.resref, "restype": inp.restype, "references": out, "count": len(out)})
    except Exception as e:
        return _err(str(e))
