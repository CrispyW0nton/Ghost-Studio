"""Clean-room GhostScripter MCP name compatibility for GhostStudio.

Only public GhostScripter interoperability contracts are represented here.
Compatibility aliases are exposed only when GhostStudio has an owning service
with equivalent intent, validation, and semantic readback.  Binary writers are
in-memory by default.  The one install operation is additionally gated by an
explicit workspace, an explicit game root, confirmation, and conflict policy.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

from kotormcp.state import load_installation, resolve_game
from kotormcp.utils import json_content


COMPATIBILITY_TOOL_NAME = "ghostrigger_ghostscripter_compatibility"


@dataclass(frozen=True)
class LegacyCompatibility:
    """One public GhostScripter README command mapped to a GhostStudio owner."""

    legacy_name: str
    category: str
    status: str
    owner: str
    equivalent: str
    callable_name: str = ""
    note: str = ""

    @property
    def exposed(self) -> bool:
        return bool(self.callable_name)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exposed"] = self.exposed
        return payload


def _alias(
    name: str,
    category: str,
    equivalent: str,
    *,
    owner: str = "GhostRigger.Core.Automation",
    note: str = "",
) -> LegacyCompatibility:
    return LegacyCompatibility(name, category, "callable_alias", owner, equivalent, name, note)


def _existing(name: str, category: str, *, note: str = "") -> LegacyCompatibility:
    return LegacyCompatibility(
        name,
        category,
        "native_name",
        "GhostRigger.Core.Automation",
        name,
        name,
        note,
    )


def _service(name: str, category: str, equivalent: str, *, note: str) -> LegacyCompatibility:
    return LegacyCompatibility(
        name,
        category,
        "core_service_only",
        "GhostRigger.Core.Workflow",
        equivalent,
        "",
        note,
    )


def _partial(name: str, category: str, equivalent: str, *, note: str) -> LegacyCompatibility:
    return LegacyCompatibility(name, category, "partial", "GhostRigger.Core.Automation", equivalent, "", note)


# README accounting: 59 primary names plus the documented journalOverview alias
# equals the advertised 60-tool surface.  The README separately mentions four
# unprefixed aliases; those already exist natively in GhostStudio and are listed
# by compatibility_report() as declared_extra_aliases rather than double-counted.
LEGACY_GHOSTSCRIPTER_COMPATIBILITY: tuple[LegacyCompatibility, ...] = (
    # Installation and discovery (7)
    _alias("gsDetectInstallations", "installation_discovery", "detectInstallations"),
    _alias("gsLoadInstallation", "installation_discovery", "loadInstallation"),
    _alias("gsListResources", "installation_discovery", "listResources"),
    _alias("gsDescribeResource", "installation_discovery", "describeResource"),
    _alias("searchResources", "installation_discovery", "kotor_search_resources"),
    _alias(
        "searchAll",
        "installation_discovery",
        "kotor_search_resources",
        note="GhostStudio fixes the search location to all resource sources.",
    ),
    _alias(
        "listResType",
        "installation_discovery",
        "listResources",
        note="The compatibility adapter maps restype to listResources.resourceTypes.",
    ),
    # Reading game formats (16 + one documented alias)
    _alias("readGFF", "read_formats", "kotor_read_gff"),
    _alias("readDLG", "read_formats", "kotor_read_gff(restype=dlg)"),
    _alias("readTwoDA", "read_formats", "kotor_read_2da"),
    _alias("readTLK", "read_formats", "kotor_read_tlk"),
    _alias("readJournal", "read_formats", "kotor_read_gff(restype=jrl)"),
    _existing("journalOverview", "read_formats", note="Already part of the native KotorMCP surface."),
    _alias(
        "readSSF",
        "read_formats",
        "src.core.scripting.data_authoring.SoundSetDocument",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Accepts base64 SSF bytes or resolves an installed resource and preserves every named and unnamed retail StrRef entry.",
    ),
    _alias(
        "readLIP",
        "read_formats",
        "src.core.scripting.data_authoring.LipDocument",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Accepts base64 LIP bytes or resolves an installed resource and returns its viseme timeline.",
    ),
    _alias("readPTH", "read_formats", "kotor_read_gff(restype=pth)"),
    _alias(
        "readLTR",
        "read_formats",
        "PyKotor read_ltr",
        note="Returns one complete 28-character probability block selected by a zero-, one-, or two-character Markov context.",
    ),
    _alias("readGUI", "read_formats", "kotor_read_gff(restype=gui)"),
    _alias(
        "readSave",
        "read_formats",
        "kotor_list_archive + kotor_list_saves",
        note="Returns a bounded decoded save overview, folder manifest, and SAVEGAME.sav resource inventory without modifying the save.",
    ),
    _alias(
        "readNCS",
        "read_formats",
        "src.core.scripting.reference.inspect_ncs",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Accepts base64 NCS bytes or resolves an installed resource; returns authoritative disassembly and recovery proof.",
    ),
    _alias(
        "readVIS",
        "read_formats",
        "PyKotor read_vis",
        note="Returns every room and its directed visible-room set from base64 or an installed VIS resource.",
    ),
    _alias("readIFO", "read_formats", "kotor_read_gff(restype=ifo)"),
    _alias(
        "readWAV",
        "read_formats",
        "PyKotor read_wav + Map Studio audio adapter",
        note="Returns decoded KOTOR WAV wrapper, codec, rate, channel, duration, and payload hash metadata without returning unbounded audio bytes.",
    ),
    _alias(
        "readTXI",
        "read_formats",
        "GhostRigger texture/TXI pipeline",
        note="Reads standalone TXI or embedded TPC metadata and returns all parsed non-null texture directives.",
    ),
    # Targeted lookups (7)
    _alias("twoDALookup", "targeted_lookup", "kotor_lookup_2da"),
    _alias("moduleOverview", "targeted_lookup", "kotor_describe_module"),
    _alias(
        "nwscriptSignature",
        "targeted_lookup",
        "src.core.scripting.reference.NWScriptReferenceService.function",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
    ),
    _alias(
        "nwscriptCategories",
        "targeted_lookup",
        "src.core.scripting.reference.NWScriptReferenceService.categories",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
    ),
    _alias(
        "searchNWScript",
        "targeted_lookup",
        "src.core.scripting.reference.NWScriptReferenceService.search_functions",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
    ),
    _alias(
        "getNWScriptDB",
        "targeted_lookup",
        "src.core.scripting.reference.NWScriptReferenceService",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="The compatibility response is paginated to keep MCP context bounded.",
    ),
    _alias(
        "pathfindRoute",
        "targeted_lookup",
        "PyKotor read_pth + bounded GhostStudio A* adapter",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.IO",
        note=(
            "Preserves GhostScripter's PTH node-index/XY-snap response contract by default; "
            "mode=wok_pie explicitly selects GhostStudio's deterministic PIE face route."
        ),
    ),
    # Writing game formats (8): bounded encoders plus an explicit install gate.
    _alias(
        "writeGFF",
        "write_formats",
        "src.core.scripting.blueprint_authoring.BlueprintGFFDocument.to_bytes/save",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Typed GFF interchange is loss-preserving; ambiguous legacy fields require allowLossy=true.",
    ),
    _alias(
        "writeDLG",
        "write_formats",
        "src.core.scripting.studio.ScriptingStudioService.dialogue_bytes",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Builds the legacy dialogue DTO through GhostStudio validation and semantic readback.",
    ),
    _alias(
        "writeTwoDA",
        "write_formats",
        "src.core.scripting.data_authoring.TwoDADocument.to_bytes",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Supports legacy text V2.0 and binary V2.b responses with table readback.",
    ),
    _alias(
        "writeERF",
        "write_formats",
        "src.core.scripting.packaging.NarrativePackagingService.build_archive",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Builds in an isolated owned temporary output and verifies every packed resource exactly.",
    ),
    _alias(
        "writeSSF",
        "write_formats",
        "src.core.scripting.data_authoring.SoundSetDocument.to_bytes",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Preserves named slots and bounded unnamed retail-tail entries; optional installs use writeOverride's gate.",
    ),
    _alias(
        "writeLIP",
        "write_formats",
        "src.core.scripting.data_authoring.LipDocument.to_bytes",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Validates ordered viseme keyframes and performs PyKotor readback before returning bytes.",
    ),
    _alias(
        "writePTH",
        "write_formats",
        "core.modules.authored_module_pathing.build_authored_pth_bytes",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Scene",
        note="Uses Map Studio's authored path graph contract when packaged and verifies the resulting PTH graph.",
    ),
    _alias(
        "writeOverride",
        "write_formats",
        "src.core.scripting.packaging.NarrativePackagingService.stage_override/install_override",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Requires explicit workspace_root, game_root, confirm_install=true, and an explicit conflict policy; installs are staged, verified, backed up, and receipted.",
    ),
    # Script tools (3)
    _alias(
        "compileScript",
        "scripts",
        "src.core.scripting.studio.ScriptingStudioService.compile_script",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="In-memory compiler output is parsed back before base64 is returned; retail game proof remains separate.",
    ),
    _alias(
        "decompileScript",
        "scripts",
        "src.core.scripting.studio.ScriptingStudioService.decompile_ncs",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Disassembly is authoritative; reconstructed source reports exact-recompile status.",
    ),
    _alias(
        "compileSummary",
        "scripts",
        "src.core.scripting.studio.ScriptingStudioService.compile_script diagnostics",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
    ),
    # Patching (1)
    _alias(
        "twoDAChangesINI",
        "patching",
        "src.core.scripting.data_authoring.TwoDADocument.export_changes_ini",
        owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow",
        note="Accepts original and modified 2DA payloads as base64 and refuses unsafe deletes.",
    ),
    # Composite game objects (17)
    _alias("getResource", "composite", "get_resource + legacy format adapter", note="Preserves format-specific legacy top-level keys and adds bounded source/hash evidence."),
    _alias("getQuest", "composite", "global.jrl query + bounded script/dialogue resolver", note="Preserves quest_id/name/states and optional scripts/dialogues under legacy flags."),
    _alias("getNpc", "composite", "getBlueprint(restype=utc)", note="Returns a typed bounded UTC tree plus common identity, localized text, script, and conversation references."),
    _alias("getCreature", "composite", "getBlueprint(restype=utc)", note="Returns a typed bounded UTC tree plus common identity, localized text, script, and conversation references."),
    _alias("getScript", "composite", "get_resource(nss/ncs) + inspect_ncs", note="Returns available NSS source and authoritative NCS inspection in one bounded response."),
    _alias("getArea", "composite", "get_resource(are/git)", note="Returns paired ARE/GIT trees, localized area identity, and bounded instance-list counts."),
    _alias("getModule", "composite", "module resource query + module.ifo adapter", note="Preserves module identity, entry point, areas, scripts, fields, sources, and optional GIT counts."),
    _alias("getDoor", "composite", "getBlueprint(restype=utd)", note="Returns a typed bounded UTD tree plus common localized, script, conversation, and inventory references."),
    _alias("getPlaceable", "composite", "getBlueprint(restype=utp)", note="Returns a typed bounded UTP tree plus common localized, script, conversation, and inventory references."),
    _alias("getItem", "composite", "getBlueprint(restype=uti)", note="Returns a typed bounded UTI tree plus common localized and inventory metadata; cross-table semantic interpretation remains explicit."),
    _alias("getEncounter", "composite", "getBlueprint(restype=ute)", note="Returns a typed bounded UTE tree plus script and spawn-list structure."),
    _alias("getTrigger", "composite", "getBlueprint(restype=utt)", note="Returns a typed bounded UTT tree plus script and transition references."),
    _alias("getWaypoint", "composite", "getBlueprint(restype=utw)", note="Returns a typed bounded UTW tree plus localized and map-note metadata."),
    _alias("getStore", "composite", "getBlueprint(restype=utm)", note="Returns a typed bounded UTM tree plus localized and inventory-list metadata."),
    _alias("getSound", "composite", "getBlueprint(restype=uts)", note="Returns a typed bounded UTS tree plus sound and script references."),
    _alias("getFaction", "composite", "get_resource(fac)", note="Returns a bounded FAC field tree and all top-level faction/reputation list counts."),
    _alias("getBlueprint", "composite", "src.core.scripting.blueprint_authoring.BlueprintGFFDocument", owner="GhostRigger.Core.Automation + GhostRigger.Core.Workflow", note="Auto-detects KOTOR blueprint type when unambiguous and returns a bounded typed field tree with preservation metadata."),
)


DECLARED_EXTRA_ALIASES: tuple[str, ...] = (
    "detectInstallations",
    "loadInstallation",
    "listResources",
    "describeResource",
)


_DIRECT_TARGETS: dict[str, str] = {
    "gsDetectInstallations": "detectInstallations",
    "gsLoadInstallation": "loadInstallation",
    "gsListResources": "listResources",
    "gsDescribeResource": "describeResource",
    "searchResources": "kotor_search_resources",
    "searchAll": "kotor_search_resources",
    "listResType": "listResources",
    "readGFF": "kotor_read_gff",
    "readDLG": "kotor_read_gff",
    "readTwoDA": "kotor_read_2da",
    "readTLK": "kotor_read_tlk",
    "readJournal": "kotor_read_gff",
    "readPTH": "kotor_read_gff",
    "readGUI": "kotor_read_gff",
    "readIFO": "kotor_read_gff",
    "twoDALookup": "kotor_lookup_2da",
    "moduleOverview": "kotor_describe_module",
}


_BLUEPRINT_COMPOSITE_TYPES: dict[str, str] = {
    "getNpc": "utc",
    "getCreature": "utc",
    "getDoor": "utd",
    "getPlaceable": "utp",
    "getItem": "uti",
    "getEncounter": "ute",
    "getTrigger": "utt",
    "getWaypoint": "utw",
    "getStore": "utm",
    "getSound": "uts",
}


_SERVICE_ALIASES: frozenset[str] = frozenset(
    {
        "readLTR",
        "readNCS",
        "readSave",
        "readSSF",
        "readLIP",
        "readVIS",
        "readWAV",
        "readTXI",
        "pathfindRoute",
        "nwscriptSignature",
        "nwscriptCategories",
        "searchNWScript",
        "getNWScriptDB",
        "compileScript",
        "decompileScript",
        "compileSummary",
        "twoDAChangesINI",
        "getScript",
        "getResource",
        "getQuest",
        "getArea",
        "getModule",
        "getFaction",
        "getBlueprint",
        "writeGFF",
        "writeDLG",
        "writeTwoDA",
        "writeERF",
        "writeSSF",
        "writeLIP",
        "writePTH",
        "writeOverride",
        "ghoststudio_restore_override",
        *_BLUEPRINT_COMPOSITE_TYPES,
    }
)


def compatibility_report() -> dict[str, Any]:
    """Return the entire README-derived inventory as JSON-serializable data."""

    counts = Counter(row.status for row in LEGACY_GHOSTSCRIPTER_COMPATIBILITY)
    categories = Counter(row.category for row in LEGACY_GHOSTSCRIPTER_COMPATIBILITY)
    exposed = sum(row.exposed for row in LEGACY_GHOSTSCRIPTER_COMPATIBILITY)
    status_counts = {
        status: counts.get(status, 0)
        for status in ("callable_alias", "native_name", "core_service_only", "partial", "missing")
    }
    return {
        "source_contract": "GhostScripter README public names only (clean-room)",
        "advertised_tool_count": 60,
        "inventory_count": len(LEGACY_GHOSTSCRIPTER_COMPATIBILITY),
        "callable_legacy_count": exposed,
        "not_exposed_count": len(LEGACY_GHOSTSCRIPTER_COMPATIBILITY) - exposed,
        "status_counts": status_counts,
        "category_counts": dict(sorted(categories.items())),
        "declared_extra_aliases": list(DECLARED_EXTRA_ALIASES),
        "safety_policy": (
            "Format writers are bounded, validated, and read back in memory. Override installation additionally "
            "requires an explicit safe workspace, explicit game root, confirmation, and conflict policy."
        ),
        "retail_proof": "No compatibility alias alone proves that KOTOR 1 or KOTOR 2 accepted a built resource.",
        "tools": [row.to_dict() for row in LEGACY_GHOSTSCRIPTER_COMPATIBILITY],
    }


def _specialized_read_schema(restype: str, *, default_resref: str = "") -> dict[str, Any]:
    properties: dict[str, Any] = {
        "game": {"type": "string", "description": "Game alias: k1 or k2"},
        "resref": {"type": "string", "description": f"Resource reference for the {restype.upper()} resource"},
        "field_paths": {"type": "array", "items": {"type": "string"}},
        "max_depth": {"type": "integer", "default": 4},
        "max_fields": {"type": "integer", "default": 200},
    }
    required = ["game"] if default_resref else ["game", "resref"]
    if default_resref:
        properties["resref"]["default"] = default_resref
    return {"type": "object", "properties": properties, "required": required}


def _service_tool_definitions() -> list[dict[str, Any]]:
    game = {"type": "string", "description": "Game alias: k1 or k2", "default": "k2"}
    source_properties = {
        "game": game,
        "resref": {"type": "string", "default": "script"},
        "source": {"type": "string", "description": "NWScript source text"},
        "include_dirs": {"type": "array", "items": {"type": "string"}},
    }
    install_guard = {
        "workspace_root": {
            "type": "string",
            "description": "Existing absolute project/workspace folder used for the owned, verified Override stage.",
        },
        "game_root": {
            "type": "string",
            "description": "Existing absolute KOTOR game root containing chitin.key and the matching executable.",
        },
        "confirm_install": {
            "type": "boolean",
            "description": "Must be literal true before GhostStudio may mutate the selected game installation.",
        },
        "on_conflict": {
            "type": "string",
            "enum": ["block", "backup"],
            "default": "block",
            "description": "Block replacement by default, or explicitly back up and replace a differing Override file.",
        },
    }
    return [
        {
            "name": COMPATIBILITY_TOOL_NAME,
            "description": "Return the clean-room GhostScripter-to-GhostStudio automation compatibility registry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Optional status filter"},
                    "category": {"type": "string", "description": "Optional category filter"},
                },
            },
        },
        {
            "name": "readNCS",
            "description": "Legacy-compatible NCS inspection using authoritative disassembly and exact-recompile proof.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "script"},
                    "ncs_base64": {"type": "string", "description": "Optional NCS bytes; otherwise resolve resref.ncs from the game"},
                    "include_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readSSF",
            "description": "Read all named and unnamed KOTOR sound-set StrRef entries from base64 or an installed SSF resource.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "soundset"},
                    "ssf_base64": {"type": "string", "description": "Optional SSF bytes; otherwise resolve resref.ssf from the game"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readLIP",
            "description": "Read a KOTOR viseme timeline from base64 or an installed LIP resource.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "voice"},
                    "lip_base64": {"type": "string", "description": "Optional LIP bytes; otherwise resolve resref.lip from the game"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readLTR",
            "description": "Inspect one complete Markov probability block from a KOTOR LTR without returning the unbounded full 3-gram table.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "humanm"},
                    "ltr_base64": {"type": "string", "description": "Optional LTR bytes; otherwise resolve resref.ltr from the game"},
                    "context": {"type": "string", "default": "", "description": "Zero, one, or two lowercase LTR characters selecting the probability block"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readVIS",
            "description": "Read the complete directed room-visibility graph from base64 or an installed VIS resource.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "module"},
                    "vis_base64": {"type": "string", "description": "Optional VIS bytes; otherwise resolve resref.vis from the game"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readWAV",
            "description": "Inspect KOTOR WAV wrapper and decoded stream metadata without returning unbounded audio samples.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "sound"},
                    "wav_base64": {"type": "string", "description": "Optional WAV bytes; otherwise resolve resref.wav from the game"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readTXI",
            "description": "Read standalone TXI metadata or the TXI embedded in a TPC and return every parsed non-null directive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "texture"},
                    "txi_base64": {"type": "string", "description": "Optional standalone TXI bytes; otherwise resolve TXI then TPC metadata"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "readSave",
            "description": "Read one save folder summary plus a bounded SAVEGAME.sav inventory; never modifies save files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "save_folder": {"type": "string", "description": "Optional exact save folder path"},
                    "save_name": {"type": "string", "description": "Optional save/folder name substring when discovering saves"},
                    "save_root": {"type": "string", "description": "Optional folder containing save directories"},
                    "resource_limit": {"type": "integer", "default": 250, "minimum": 1, "maximum": 1000},
                },
                "required": ["game"],
            },
        },
        {
            "name": "pathfindRoute",
            "description": (
                "Find the bounded A* shortest route on an area's KOTOR PTH graph. Endpoints may be PTH node "
                "indices or world-space X/Y coordinates, which snap to the nearest node. Returns the legacy "
                "node-index path, distance, and waypoint contract. Set mode=wok_pie to explicitly use Map "
                "Studio PIE walkable-face routing instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "maxLength": 16, "description": "Area/resource resref"},
                    "mode": {
                        "type": "string",
                        "enum": ["pth", "wok_pie"],
                        "default": "pth",
                        "description": "Legacy PTH routing by default; explicit Map Studio PIE WOK routing when requested",
                    },
                    "pth_base64": {"type": "string", "description": "Optional PTH bytes; otherwise resolve resref.pth from the game"},
                    "start_index": {"type": "integer", "minimum": 0, "description": "Starting PTH node index"},
                    "end_index": {"type": "integer", "minimum": 0, "description": "Destination PTH node index"},
                    "start_x": {"type": "number", "description": "World X; used with start_y to snap to the nearest PTH node"},
                    "start_y": {"type": "number", "description": "World Y; used with start_x"},
                    "end_x": {"type": "number", "description": "World X; used with end_y to snap to the nearest PTH node"},
                    "end_y": {"type": "number", "description": "World Y; used with end_x"},
                    "wok_base64": {"type": "string", "description": "Optional WOK bytes for mode=wok_pie; otherwise resolve resref.wok"},
                    "start": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "PIE mode start XYZ"},
                    "destination": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "PIE mode destination XYZ"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "getResource",
            "description": (
                "Legacy universal KOTOR resource reader. Preserves GhostScripter's format-specific top-level "
                "keys while adding bounded GhostStudio source/hash evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "maxLength": 16},
                    "type": {"type": "string", "description": "Resource extension such as utc, dlg, 2da, nss, or ncs"},
                    "restype": {"type": "string", "description": "Accepted GhostStudio alias for type"},
                    "format": {"type": "string", "default": "auto"},
                    "field_paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["game", "resref", "type"],
            },
        },
        {
            "name": "getQuest",
            "description": (
                "Return one journal quest with resolved state text, optional referenced scripts, and an optional "
                "bounded DLG-reference scan using GhostScripter's response contract."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "questId": {"type": "string"},
                    "quest_id": {"type": "string"},
                    "tag": {"type": "string"},
                    "includeScripts": {"type": "boolean", "default": True},
                    "includeDialogues": {"type": "boolean", "default": False},
                    "include_scripts": {"type": "boolean"},
                    "include_dlg": {"type": "boolean"},
                },
                "required": ["game", "questId"],
            },
        },
        {
            "name": "getModule",
            "description": (
                "Return module.ifo identity, entry point, areas, scripts, optional per-area GIT counts, and "
                "bounded GhostStudio module-source evidence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "module_id": {"type": "string", "maxLength": 16},
                    "module_root": {"type": "string", "maxLength": 16},
                    "include_git": {"type": "boolean", "default": False},
                },
                "required": ["game", "module_id"],
            },
        },
        {
            "name": "getScript",
            "description": "Return installed NSS source and authoritative NCS disassembly/recovery evidence together in one bounded response.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "include_source": {"type": "boolean", "default": True},
                    "include_disassembly": {"type": "boolean", "default": True},
                    "max_text_chars": {"type": "integer", "default": 250000, "minimum": 1000, "maximum": 750000},
                    "include_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "getArea",
            "description": "Return paired ARE/GIT GFF trees, localized area name, and top-level gameplay instance counts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "module_id": {"type": "string", "description": "Optional module capsule context when an area resref is shared"},
                    "max_depth": {"type": "integer", "default": 5, "minimum": 1, "maximum": 12},
                    "max_fields": {"type": "integer", "default": 300, "minimum": 1, "maximum": 1000},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "getFaction",
            "description": "Return a bounded FAC GFF tree plus faction/reputation list counts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "repute"},
                    "max_depth": {"type": "integer", "default": 6, "minimum": 1, "maximum": 12},
                    "max_fields": {"type": "integer", "default": 500, "minimum": 1, "maximum": 1000},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "getBlueprint",
            "description": "Auto-detect or read a typed KOTOR blueprint and return bounded preservation-safe field rows and common references.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "type": {"type": "string", "description": "Legacy alias for restype"},
                    "restype": {"type": "string", "description": "Optional exact blueprint type such as utc, utp, utd, or uti"},
                    "max_fields": {"type": "integer", "default": 300, "minimum": 1, "maximum": 1000},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "nwscriptSignature",
            "description": "Return one K1/K2 NWScript function signature from GhostStudio's compiler definitions.",
            "inputSchema": {
                "type": "object",
                "properties": {"game": game, "name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "nwscriptCategories",
            "description": "List GhostStudio's searchable NWScript function categories.",
            "inputSchema": {"type": "object", "properties": {"game": game}},
        },
        {
            "name": "searchNWScript",
            "description": "Search K1/K2 NWScript function names, signatures, and descriptions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "query": {"type": "string", "default": ""},
                    "category": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            },
        },
        {
            "name": "getNWScriptDB",
            "description": "Return a bounded page of K1/K2 NWScript functions and constants.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 200},
                },
            },
        },
        {
            "name": "compileScript",
            "description": "Compile NWScript in memory, parse the NCS back, and return base64 bytes plus diagnostics.",
            "inputSchema": {"type": "object", "properties": source_properties, "required": ["game", "resref", "source"]},
        },
        {
            "name": "decompileScript",
            "description": "Recover NWScript from base64 or installed NCS and report whether recompilation is byte-exact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "script"},
                    "ncs_base64": {"type": "string"},
                },
                "required": ["game", "resref"],
            },
        },
        {
            "name": "compileSummary",
            "description": "Validate and compile NWScript without returning bytecode; reports compiler/readback diagnostics.",
            "inputSchema": {"type": "object", "properties": source_properties, "required": ["game", "resref", "source"]},
        },
        {
            "name": "twoDAChangesINI",
            "description": "Generate a conservative TSLPatcher changes.ini diff from two base64 2DA payloads; unsafe deletes are refused.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "original_2da_base64": {"type": "string"},
                    "modified_2da_base64": {"type": "string"},
                },
                "required": ["table_name", "original_2da_base64", "modified_2da_base64"],
            },
        },
        {
            "name": "writeGFF",
            "description": "Write a complete typed GhostScripter GFF document, or explicitly opt into lossy legacy-field inference.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document": {"type": "object"},
                    "fileType": {"type": "string"},
                    "file_type": {"type": "string"},
                    "fields": {"type": "object"},
                    "allowLossy": {"type": "boolean", "default": False},
                    "allow_lossy": {"type": "boolean", "default": False},
                },
                "oneOf": [
                    {"required": ["document"]},
                    {"required": ["fileType", "fields", "allowLossy"]},
                    {"required": ["file_type", "fields", "allow_lossy"]},
                ],
            },
        },
        {
            "name": "writeDLG",
            "description": "Write a legacy GhostScripter dialogue DTO through GhostStudio's validated DLG service.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string", "default": "dialogue"},
                    "dialogue": {"type": "object"},
                },
                "required": ["game", "dialogue"],
            },
        },
        {
            "name": "writeTwoDA",
            "description": "Write legacy columns/rows as KOTOR 2DA V2.0 text or V2.b binary with semantic readback.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resref": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "object"}},
                    "format": {"type": "string", "enum": ["text", "binary"], "default": "text"},
                    "edits": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["resref", "columns", "rows"],
            },
        },
        {
            "name": "writeERF",
            "description": "Pack legacy resource rows into a verified ERF, MOD, or SAV archive and return it in memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "object"}},
                    "archive_type": {"type": "string", "enum": ["ERF ", "MOD ", "SAV "], "default": "MOD "},
                },
                "required": ["files"],
            },
        },
        {
            "name": "writeLIP",
            "description": "Write an ordered KOTOR LIP V1.0 viseme timeline and return verified bytes in memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "duration": {"type": "number"},
                    "keyframes": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["duration", "keyframes"],
            },
        },
        {
            "name": "writeSSF",
            "description": "Write a KOTOR SSF V1.1 including bounded unnamed retail-tail entries; optional installs use the explicit Override gate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "slots": {"type": "object"},
                    "unknown_slots": {},
                    "write_override": {"type": "boolean", "default": False},
                    **install_guard,
                },
                "required": ["game", "resref", "slots"],
            },
        },
        {
            "name": "writePTH",
            "description": "Write a validated KOTOR PTH path graph and optionally install through the explicit Override gate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "points": {"type": "array", "items": {"type": "object"}},
                    "write_override": {"type": "boolean", "default": False},
                    **install_guard,
                },
                "required": ["game", "resref", "points"],
            },
        },
        {
            "name": "writeOverride",
            "description": "Stage, verify, and explicitly install one resource into KOTOR Override with conflict backup and a receipt.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "restype": {"type": "string"},
                    "data_b64": {"type": "string"},
                    **install_guard,
                    "backup_root": {
                        "type": "string",
                        "description": (
                            "Optional existing absolute folder outside game_root for the install transaction, "
                            "conflict backups and install receipt (default: game_root/GhostStudioBackups)."
                        ),
                    },
                    "expect_existing": {
                        "type": "string",
                        "description": (
                            "Optional compare-and-swap guard from a preview: 'absent' or the SHA-256 the "
                            "current Override file must have; any other state refuses the install."
                        ),
                    },
                },
                "required": [
                    "game", "resref", "restype", "data_b64",
                    "workspace_root", "game_root", "confirm_install",
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "ghoststudio_restore_override",
            "description": (
                "Undo one writeOverride install: only when the Override file still has the installed SHA-256, "
                "save its bytes under backup_root (outside the game root), then write the prior bytes back "
                "or remove a file the install created. Requires confirm_restore=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": game,
                    "resref": {"type": "string"},
                    "restype": {"type": "string"},
                    "game_root": install_guard["game_root"],
                    "backup_root": {
                        "type": "string",
                        "description": "Existing absolute folder outside game_root for the pre-restore backup and receipt.",
                    },
                    "expect_current": {
                        "type": "string",
                        "description": "SHA-256 the current Override file must have (the installed bytes).",
                    },
                    "prior_data_b64": {
                        "type": "string",
                        "description": "Prior bytes to write back (base64); exactly one of this or remove_file.",
                    },
                    "remove_file": {
                        "type": "boolean",
                        "description": "true when the install created the file: remove it instead of writing prior bytes.",
                    },
                    "confirm_restore": {
                        "type": "boolean",
                        "description": "Must be literal true before GhostStudio may change a game installation.",
                    },
                },
                "required": [
                    "game", "resref", "restype", "game_root", "backup_root",
                    "expect_current", "confirm_restore",
                ],
                "additionalProperties": False,
            },
        },
    ]


def get_tools(native_tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build alias definitions from native schemas plus bounded service tools."""

    native_by_name = {str(row.get("name", "")): row for row in native_tools}
    definitions: list[dict[str, Any]] = []
    specialized = {
        "readDLG": _specialized_read_schema("dlg"),
        "readJournal": _specialized_read_schema("jrl", default_resref="global"),
        "readPTH": _specialized_read_schema("pth"),
        "readGUI": _specialized_read_schema("gui"),
        "readIFO": _specialized_read_schema("ifo", default_resref="module"),
    }
    for legacy_name, target_name in _DIRECT_TARGETS.items():
        target = native_by_name.get(target_name)
        if target is None:
            raise RuntimeError(f"Legacy compatibility target is not registered: {target_name}")
        definition = dict(target)
        definition["name"] = legacy_name
        definition["description"] = f"Legacy GhostScripter compatibility alias for {target_name}. " + str(
            target.get("description", "")
        )
        if legacy_name in specialized:
            definition["inputSchema"] = specialized[legacy_name]
        elif legacy_name == "listResType":
            definition["inputSchema"] = {
                "type": "object",
                "properties": {
                    "game": {"type": "string"},
                    "restype": {"type": "string"},
                    "location": {"type": "string", "default": "all"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["game", "restype"],
            }
        definitions.append(definition)
    service_definitions = _service_tool_definitions()
    definitions.extend(service_definitions)
    blueprint_definition = next(row for row in service_definitions if row["name"] == "getBlueprint")
    for legacy_name, restype in _BLUEPRINT_COMPOSITE_TYPES.items():
        definition = dict(blueprint_definition)
        definition["name"] = legacy_name
        definition["description"] = (
            f"Legacy GhostScripter {legacy_name} compatibility reader for a typed .{restype} blueprint. "
            "Returns bounded fields and common references without modifying the resource."
        )
        schema = dict(blueprint_definition["inputSchema"])
        properties = {key: dict(value) for key, value in schema["properties"].items()}
        properties["restype"] = {"type": "string", "const": restype, "default": restype}
        schema["properties"] = properties
        definition["inputSchema"] = schema
        definitions.append(definition)
    return definitions


def resolve_direct_alias(name: str, arguments: Mapping[str, Any]) -> tuple[str, Dict[str, Any]] | None:
    """Resolve an alias to a native command and adapt only documented-safe defaults."""

    target = _DIRECT_TARGETS.get(name)
    if target is None:
        return None
    adapted = dict(arguments)
    if name == "searchAll":
        adapted["location"] = "all"
    elif name == "listResType":
        restype = str(adapted.pop("restype", "")).strip()
        adapted["resourceTypes"] = [restype] if restype else []
    elif name in {"readDLG", "readPTH", "readGUI", "readIFO", "readJournal"}:
        adapted["restype"] = {
            "readDLG": "dlg",
            "readPTH": "pth",
            "readGUI": "gui",
            "readIFO": "ifo",
            "readJournal": "jrl",
        }[name]
        if name == "readJournal":
            adapted.setdefault("resref", "global")
        elif name == "readIFO":
            adapted.setdefault("resref", "module")
    return target, adapted


def handles_service_alias(name: str) -> bool:
    return name == COMPATIBILITY_TOOL_NAME or name in _SERVICE_ALIASES


def _function_payload(row: object) -> dict[str, Any]:
    return {
        "routine_id": int(getattr(row, "routine_id", -1)),
        "name": str(getattr(row, "name", "")),
        "return_type": str(getattr(row, "return_type", "")),
        "signature": str(getattr(row, "signature", "")),
        "description": str(getattr(row, "description", "")),
        "category": str(getattr(row, "category", "")),
        "game": str(getattr(row, "game", "")),
    }


def _constant_payload(row: object) -> dict[str, Any]:
    return {
        "name": str(getattr(row, "name", "")),
        "datatype": str(getattr(row, "datatype", "")),
        "value": str(getattr(row, "value", "")),
        "game": str(getattr(row, "game", "")),
    }


def _diagnostics(rows: Iterable[object]) -> list[dict[str, Any]]:
    return [row.to_dict() if hasattr(row, "to_dict") else {"message": str(row)} for row in rows]


def _ncs_payload(arguments: Mapping[str, Any]) -> bytes:
    encoded = str(arguments.get("ncs_base64", "") or "").strip()
    if encoded:
        return base64.b64decode(encoded, validate=True)
    game = resolve_game(str(arguments.get("game", "")))
    if game is None:
        raise ValueError("game must be k1 or k2")
    resref = str(arguments.get("resref", "") or "").strip()
    if not resref:
        raise ValueError("resref is required when ncs_base64 is omitted")
    entry = load_installation(game).get_resource(resref, "ncs")
    if entry is None:
        raise FileNotFoundError(f"{resref}.ncs was not found in the selected installation")
    return bytes(entry.data)


def _format_payload(arguments: Mapping[str, Any], restype: str) -> bytes:
    encoded = str(arguments.get(f"{restype}_base64", "") or "").strip()
    if encoded:
        return base64.b64decode(encoded, validate=True)
    game = resolve_game(str(arguments.get("game", "")))
    if game is None:
        raise ValueError("game must be k1 or k2")
    resref = str(arguments.get("resref", "") or "").strip()
    if not resref:
        raise ValueError(f"resref is required when {restype}_base64 is omitted")
    entry = load_installation(game).get_resource(resref, restype)
    if entry is None:
        raise FileNotFoundError(f"{resref}.{restype} was not found in the selected installation")
    return bytes(entry.data)


def _json_value(value: Any) -> Any:
    """Convert PyKotor metadata values to bounded JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    text = str(value)
    return (text, False) if len(text) <= limit else (text[:limit], True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ltr_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.ltr import LTR, read_ltr

    payload = _format_payload(arguments, "ltr")
    document = read_ltr(payload)
    context = str(arguments.get("context", "") or "").casefold()
    if len(context) > 2 or any(character not in LTR.CHARACTER_SET for character in context):
        raise ValueError("LTR context must contain zero, one, or two characters from abcdefghijklmnopqrstuvwxyz'-")
    if not context:
        block = document._singles  # noqa: SLF001 - PyKotor exposes no read-only block accessor.
        table = "singles"
    elif len(context) == 1:
        block = document._doubles[LTR.CHARACTER_SET.index(context)]  # noqa: SLF001
        table = "doubles"
    else:
        block = document._triples[LTR.CHARACTER_SET.index(context[0])][LTR.CHARACTER_SET.index(context[1])]  # noqa: SLF001
        table = "triples"
    probabilities = [
        {
            "character": character,
            "start": block.get_start(character),
            "middle": block.get_middle(character),
            "end": block.get_end(character),
        }
        for character in LTR.CHARACTER_SET
    ]
    return {
        "resref": str(arguments.get("resref", "humanm")),
        "context": context,
        "table": table,
        "character_set": LTR.CHARACTER_SET,
        "character_count": len(LTR.CHARACTER_SET),
        "block_count": {"singles": 1, "doubles": 28, "triples": 784},
        "probabilities": probabilities,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_vis_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.vis import read_vis

    payload = _format_payload(arguments, "vis")
    document = read_vis(payload)
    rooms = sorted(document.all_rooms())
    visibility = {
        room: [candidate for candidate in rooms if document.get_visible(room, candidate)]
        for room in rooms
    }
    return {
        "resref": str(arguments.get("resref", "module")),
        "room_count": len(rooms),
        "directed_edge_count": sum(len(rows) for rows in visibility.values()),
        "rooms": rooms,
        "visibility": visibility,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_wav_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.wav import read_wav

    payload = _format_payload(arguments, "wav")
    document = read_wav(payload)
    encoding = document.get_encoding_enum()
    duration = (len(document.data) / document.bytes_per_sec) if document.bytes_per_sec > 0 else None
    return {
        "resref": str(arguments.get("resref", "sound")),
        "wav_type": getattr(document.wav_type, "name", str(document.wav_type)),
        "audio_format": getattr(document.audio_format, "name", str(document.audio_format)),
        "encoding": encoding.name if encoding is not None else f"UNKNOWN_{document.encoding}",
        "encoding_code": int(document.encoding),
        "channels": int(document.channels),
        "sample_rate": int(document.sample_rate),
        "bits_per_sample": int(document.bits_per_sample),
        "bytes_per_second": int(document.bytes_per_sec),
        "block_align": int(document.block_align),
        "duration_seconds": duration,
        "is_pcm": bool(document.is_pcm()),
        "is_adpcm": bool(document.is_adpcm()),
        "is_mp3": bool(document.is_mp3()),
        "decoded_audio_byte_count": len(document.data),
        "decoded_audio_sha256": hashlib.sha256(document.data).hexdigest(),
        "source_byte_count": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_txi_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.txi import read_txi

    encoded = str(arguments.get("txi_base64", "") or "").strip()
    source_restype = "txi"
    if encoded:
        payload = base64.b64decode(encoded, validate=True)
        document = read_txi(payload)
        raw_text = payload.decode("utf-8", errors="replace")
    else:
        game = resolve_game(str(arguments.get("game", "")))
        if game is None:
            raise ValueError("game must be k1 or k2")
        resref = str(arguments.get("resref", "") or "").strip()
        if not resref:
            raise ValueError("resref is required when txi_base64 is omitted")
        installation = load_installation(game)
        entry = installation.get_resource(resref, "txi")
        if entry is not None:
            payload = bytes(entry.data)
            document = read_txi(payload)
            raw_text = payload.decode("utf-8", errors="replace")
        else:
            from pykotor.resource.formats.tpc import read_tpc

            entry = installation.get_resource(resref, "tpc")
            if entry is None:
                raise FileNotFoundError(f"Neither {resref}.txi nor embedded {resref}.tpc TXI metadata was found")
            payload = bytes(entry.data)
            document = read_tpc(payload).txi
            raw_text = str(document)
            source_restype = "tpc"
    directives = {
        key: _json_value(value)
        for key, value in vars(document.get_features()).items()
        if value is not None
    }
    return {
        "resref": str(arguments.get("resref", "texture")),
        "source_restype": source_restype,
        "embedded_in_tpc": source_restype == "tpc",
        "empty": bool(document.empty()),
        "directive_count": len(directives),
        "directives": directives,
        "raw_text": raw_text,
        "source_byte_count": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_save_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.erf import read_erf

    from kotormcp.tools.game_test import _discover_saves, _save_summary

    game = resolve_game(str(arguments.get("game", "")))
    if game is None:
        raise ValueError("game must be k1 or k2")
    installation = load_installation(game)
    requested_folder = str(arguments.get("save_folder", "") or "").strip()
    if requested_folder:
        folder = Path(requested_folder).expanduser().resolve()
        if not (folder / "SAVEGAME.sav").is_file():
            raise FileNotFoundError(f"SAVEGAME.sav was not found in {folder}")
        summary = _save_summary(folder)
    else:
        saves = _discover_saves(
            Path(installation.path()),
            explicit_root=str(arguments.get("save_root", "") or "").strip() or None,
        )
        needle = str(arguments.get("save_name", "") or "").strip().casefold()
        if needle:
            summary = next(
                (
                    row
                    for row in saves
                    if needle
                    in " ".join(
                        str(row.get(key, ""))
                        for key in ("folder_name", "save_name", "area_name", "last_module", "pc_name")
                    ).casefold()
                ),
                None,
            )
        else:
            summary = saves[0] if saves else None
        if summary is None:
            suffix = f" matching {needle!r}" if needle else ""
            raise FileNotFoundError(f"No KOTOR save was found{suffix}")
        folder = Path(str(summary["folder"])).resolve()

    savegame_path = folder / "SAVEGAME.sav"
    archive = read_erf(savegame_path)
    resources = [
        {"resref": str(resource.resref), "restype": str(resource.restype.extension).casefold()}
        for resource in archive
    ]
    limit = min(1000, max(1, int(arguments.get("resource_limit", 250) or 250)))
    files = []
    for path in sorted((row for row in folder.iterdir() if row.is_file()), key=lambda row: row.name.casefold()):
        try:
            files.append({"name": path.name, "byte_count": path.stat().st_size})
        except OSError:
            files.append({"name": path.name, "byte_count": None})
    summary = dict(summary)
    summary["folder"] = str(folder)
    return {
        "game": installation.game_name(),
        "summary": summary,
        "file_count": len(files),
        "files_truncated": len(files) > 250,
        "files": files[:250],
        "savegame_byte_count": savegame_path.stat().st_size,
        "savegame_sha256": _sha256_file(savegame_path),
        "resource_count": len(resources),
        "resource_limit": limit,
        "resources_truncated": len(resources) > limit,
        "resources": resources[:limit],
    }


def _bounded_installed_or_base64_payload(arguments: Mapping[str, Any], restype: str) -> bytes:
    """Resolve one compatibility input without permitting an unbounded binary."""

    encoded = arguments.get(f"{restype}_base64")
    if encoded not in (None, ""):
        return _decode_compat_base64(encoded, f"{restype}_base64")
    game_id = resolve_game(str(arguments.get("game", "")))
    if game_id is None:
        raise ValueError("game must be k1 or k2")
    resref = _compat_resref(arguments.get("resref"))
    entry = load_installation(game_id).get_resource(resref, restype)
    if entry is None:
        raise FileNotFoundError(f"{resref}.{restype} was not found in the selected installation")
    payload = bytes(entry.data)
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError(f"{resref}.{restype} exceeds the 64 MiB compatibility safety limit")
    return payload


def _pathfind_wok_pie_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.bwm import read_bwm
    from src.math.walkmesh_runtime import WalkmeshRuntimeIndex

    payload = _bounded_installed_or_base64_payload(arguments, "wok")
    start_values = tuple(float(value) for value in arguments.get("start", ()))
    destination_values = tuple(float(value) for value in arguments.get("destination", ()))
    if len(start_values) != 3 or len(destination_values) != 3:
        raise ValueError("start and destination must each contain exactly three coordinates")
    bwm = read_bwm(payload)
    vertices: list[tuple[float, float, float]] = []
    vertex_indexes: dict[tuple[float, float, float], int] = {}

    def vertex_index(value: Any) -> int:
        point = (float(value.x), float(value.y), float(value.z))
        if point not in vertex_indexes:
            vertex_indexes[point] = len(vertices)
            vertices.append(point)
        return vertex_indexes[point]

    faces = [
        SimpleNamespace(
            v1=vertex_index(face.v1),
            v2=vertex_index(face.v2),
            v3=vertex_index(face.v3),
            surface=int(face.material),
            adj1=-1,
            adj2=-1,
            adj3=-1,
        )
        for face in bwm.faces
    ]
    runtime = WalkmeshRuntimeIndex(
        SimpleNamespace(verts=vertices, faces=faces),
        game=str(arguments.get("game", "K2") or "K2"),
    )
    start = runtime.sample_at(start_values[0], start_values[1], start_values[2])
    destination = runtime.sample_at(destination_values[0], destination_values[1], destination_values[2])
    route = runtime.route(start.face_index, destination.face_index, destination.position) if start and destination else ()
    return {
        "game": str(arguments.get("game", "")).upper(),
        "resref": str(arguments.get("resref", "room")),
        "start": list(start_values),
        "destination": list(destination_values),
        "start_sample": (
            {"position": list(start.position), "face_index": start.face_index, "surface_id": start.surface_id}
            if start is not None
            else None
        ),
        "destination_sample": (
            {"position": list(destination.position), "face_index": destination.face_index, "surface_id": destination.surface_id}
            if destination is not None
            else None
        ),
        "connected": bool(route),
        "route_point_count": len(route),
        "route": [list(point) for point in route],
        "walkable_face_count": len(runtime.walkable_faces),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "proof_scope": "Deterministic GhostStudio PIE face routing; retail KOTOR runtime proof remains a separate game test.",
    }


def _pathfind_pth_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve GhostScripter's public PTH A* route and XY-snap contract."""

    import heapq

    from pykotor.resource.generics.pth import read_pth

    game = _compat_game(arguments.get("game"))
    resref = _compat_resref(arguments.get("resref"))
    payload = _bounded_installed_or_base64_payload(arguments, "pth")
    pth = read_pth(payload)
    node_count = len(pth)
    if node_count == 0:
        raise ValueError(f"pathfindRoute: no Path_Points found in {resref}.pth")
    if node_count > 200_000:
        raise ValueError("pathfindRoute: PTH exceeds the 200,000-node safety limit")

    points: list[dict[str, float]] = []
    for index, point in enumerate(pth):
        x = float(point.x)
        y = float(point.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"pathfindRoute: PTH node {index} has non-finite coordinates")
        points.append({"x": x, "y": y})

    raw_edges = tuple(getattr(pth, "_connections", ()))
    if len(raw_edges) > 1_000_000:
        raise ValueError("pathfindRoute: PTH exceeds the 1,000,000-edge safety limit")
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for edge in raw_edges:
        source = int(edge.source)
        target = int(edge.target)
        # Retail PTH data can contain stale connection records. They cannot be
        # traversed safely, so retain valid directed graph semantics and ignore
        # only records that address no decoded Path_Point.
        if 0 <= source < node_count and 0 <= target < node_count:
            adjacency[source].append(target)

    def distance(first: int, second: int) -> float:
        dx = points[first]["x"] - points[second]["x"]
        dy = points[first]["y"] - points[second]["y"]
        return math.hypot(dx, dy)

    def nearest_node(x: float, y: float) -> tuple[int, float]:
        best_index = 0
        best_distance = math.inf
        for index, point in enumerate(points):
            candidate = math.hypot(point["x"] - x, point["y"] - y)
            if candidate < best_distance:
                best_index = index
                best_distance = candidate
        return best_index, round(best_distance, 4)

    start_snap: dict[str, Any] | None = None
    end_snap: dict[str, Any] | None = None
    if arguments.get("start_index") is not None:
        start_index = _strict_integer(arguments.get("start_index"), "start_index")
    elif arguments.get("start_x") is not None and arguments.get("start_y") is not None:
        start_x = _finite_number(arguments.get("start_x"), "start_x")
        start_y = _finite_number(arguments.get("start_y"), "start_y")
        start_index, snap_distance = nearest_node(start_x, start_y)
        start_snap = {
            "snapped_to": start_index,
            "snap_distance": snap_distance,
            "from_x": start_x,
            "from_y": start_y,
        }
    else:
        raise ValueError("pathfindRoute: provide 'start_index' OR ('start_x' + 'start_y')")

    if arguments.get("end_index") is not None:
        end_index = _strict_integer(arguments.get("end_index"), "end_index")
    elif arguments.get("end_x") is not None and arguments.get("end_y") is not None:
        end_x = _finite_number(arguments.get("end_x"), "end_x")
        end_y = _finite_number(arguments.get("end_y"), "end_y")
        end_index, snap_distance = nearest_node(end_x, end_y)
        end_snap = {
            "snapped_to": end_index,
            "snap_distance": snap_distance,
            "from_x": end_x,
            "from_y": end_y,
        }
    else:
        raise ValueError("pathfindRoute: provide 'end_index' OR ('end_x' + 'end_y')")

    if not 0 <= start_index < node_count:
        raise ValueError(f"pathfindRoute: start_index {start_index} out of range [0, {node_count - 1}]")
    if not 0 <= end_index < node_count:
        raise ValueError(f"pathfindRoute: end_index {end_index} out of range [0, {node_count - 1}]")

    path: list[int]
    if start_index == end_index:
        path = [start_index]
        total_distance = 0.0
    else:
        open_set: list[tuple[float, int]] = [(0.0, start_index)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_index: 0.0}
        visited: set[int] = set()
        found = False
        while open_set:
            _score, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current == end_index:
                found = True
                break
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                candidate_score = g_score[current] + distance(current, neighbor)
                if candidate_score < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = candidate_score
                    heapq.heappush(open_set, (candidate_score + distance(neighbor, end_index), neighbor))
        if not found:
            raise ValueError(f"pathfindRoute: no path found from {start_index} to {end_index}")
        path = []
        current = end_index
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start_index)
        path.reverse()
        total_distance = sum(distance(path[index], path[index + 1]) for index in range(len(path) - 1))

    result: dict[str, Any] = {
        "game": game,
        "resref": resref,
        "start_index": start_index,
        "end_index": end_index,
        "path": path,
        "step_count": len(path),
        "total_distance": round(total_distance, 4),
        "waypoints": [points[index] for index in path],
    }
    if start_snap is not None:
        result["start_nearest"] = start_snap
    if end_snap is not None:
        result["end_nearest"] = end_snap
    return result


def _pathfind_route_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(arguments.get("mode", "pth") or "pth").strip().casefold()
    if mode == "pth":
        return _pathfind_pth_payload(arguments)
    if mode == "wok_pie":
        return _pathfind_wok_pie_payload(arguments)
    raise ValueError("pathfindRoute mode must be 'pth' or 'wok_pie'")


def _resource_entry_type(entry: Any) -> str:
    extension = str(getattr(entry, "extension", "") or "").strip().casefold().lstrip(".")
    if extension:
        return extension
    return str(getattr(entry, "restype", "") or "").strip().casefold().lstrip(".")


def _legacy_dialogue_resource_payload(payload: bytes, installation: Any) -> dict[str, Any]:
    from pykotor.resource.generics.dlg import DLGReply, read_dlg

    dialogue = read_dlg(payload)
    entries = dialogue.all_entries(as_sorted=True)
    replies = dialogue.all_replies(as_sorted=True)
    node_limit = 5_000

    def link_payload(link: Any) -> dict[str, Any]:
        node = getattr(link, "node", None)
        active1 = str(getattr(link, "active1", ""))
        active2 = str(getattr(link, "active2", ""))
        return {
            "index": int(getattr(node, "list_index", -1)),
            "is_reply": isinstance(node, DLGReply),
            "is_child": bool(getattr(link, "is_child", False)),
            "active_script": "" if active1 == "****" else active1,
            "active_script2": "" if active2 == "****" else active2,
            "link_comment": str(getattr(link, "comment", "")),
            "display_inactive": bool(getattr(link, "display_inactive", False)),
        }

    def node_payload(node: Any, *, reply: bool) -> dict[str, Any]:
        text_value = getattr(node, "text", None)
        row: dict[str, Any] = {
            "text": _legacy_gff_value(text_value, installation),
            "strref": int(getattr(text_value, "stringref", -1)),
        }
        if reply:
            row["listener"] = str(getattr(node, "listener", ""))
        else:
            row["speaker"] = str(getattr(node, "speaker", ""))
        for key, attribute in (
            ("script1", "script1"),
            ("script2", "script2"),
            ("vo_resref", "vo_resref"),
            ("sound", "sound"),
        ):
            value = str(getattr(node, attribute, ""))
            if value and value != "****":
                row[key] = value
        row["branches"] = [link_payload(link) for link in list(getattr(node, "links", ()))[:5_000]]
        return row

    result: dict[str, Any] = {
        "entry_count": len(entries),
        "reply_count": len(replies),
        "starters": [link_payload(link) for link in list(dialogue.starters)[:5_000]],
        "entries": [node_payload(node, reply=False) for node in entries[:node_limit]],
        "replies": [node_payload(node, reply=True) for node in replies[:node_limit]],
        "skippable": bool(dialogue.skippable),
        "delay_entry": int(dialogue.delay_entry),
        "delay_reply": int(dialogue.delay_reply),
        "ambient_track": str(dialogue.ambient_track),
        "animated_cut": int(dialogue.animated_cut),
        "camera_model": str(dialogue.camera_model),
        "conversation_type": int(getattr(dialogue.conversation_type, "value", dialogue.conversation_type)),
        "computer_type": int(getattr(dialogue.computer_type, "value", dialogue.computer_type)),
        "old_hit_check": bool(dialogue.old_hit_check),
        "unequip_items": bool(dialogue.unequip_items),
        "unequip_h_item": bool(dialogue.unequip_hands),
        "word_count": int(dialogue.word_count),
    }
    end_script = str(dialogue.on_end)
    abort_script = str(dialogue.on_abort)
    if end_script and end_script != "****":
        result["end_script"] = end_script
    if abort_script and abort_script != "****":
        result["abort_script"] = abort_script
    result["ghoststudio_dialogue_truncated"] = (
        len(entries) > node_limit or len(replies) > node_limit or len(dialogue.starters) > 5_000
    )
    return result


def _get_resource_compat_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    game = _compat_game(arguments.get("game"))
    game_id = resolve_game(game)
    if game_id is None:
        raise ValueError("game must be K1 or K2")
    resref = _compat_resref(arguments.get("resref"))
    restype = str(arguments.get("type", arguments.get("restype", "")) or "").strip().casefold().lstrip(".")
    if not restype or len(restype) > 16 or not re.fullmatch(r"[a-z0-9_]+", restype):
        raise ValueError("getResource: 'type' is required and must be a resource extension")
    installation = load_installation(game_id)
    entry = installation.get_resource(resref, restype)
    if entry is None:
        raise FileNotFoundError(f"Resource not found: {resref}.{restype} in {game}")
    payload = bytes(entry.data)
    response: dict[str, Any] = {
        "game": game,
        "resref": resref,
        "type": restype,
        "size_bytes": len(payload),
        "ghoststudio": {
            "restype": restype,
            "source": str(getattr(entry, "source", "")),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bounded_response": True,
        },
    }
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        response["parse_error"] = "Resource exceeds the 64 MiB compatibility parsing safety limit."
        return response
    try:
        if restype == "2da":
            from pykotor.resource.formats.twoda import read_2da

            table = read_2da(payload)
            columns = list(table.get_headers())
            rows = []
            for index in range(min(table.get_height(), 100)):
                row = {"__label": table.get_label(index)}
                for column in columns:
                    row[column] = table.get_cell_safe(index, column, "****")
                rows.append(row)
            response.update(
                {
                    "columns": columns,
                    "row_count": table.get_height(),
                    "rows": {
                        "columns": columns,
                        "total_rows": table.get_height(),
                        "offset": 0,
                        "returned": len(rows),
                        "rows": rows,
                    },
                }
            )
        elif restype == "dlg":
            response.update(_legacy_dialogue_resource_payload(payload, installation))
        elif restype in {
            "utc", "utp", "utd", "uti", "uts", "utt", "utw", "ute", "utm", "are", "git", "jrl",
            "ifo", "fac", "bic", "gff", "pth", "gui", "res", "gic", "nfo",
        }:
            from pykotor.resource.formats.gff import read_gff

            response["fields"] = _legacy_gff_value(read_gff(payload).root, installation, budget=[20_000])
        elif restype == "nss":
            source, truncated = _bounded_text(payload.decode("latin-1", errors="replace"), 750_000)
            response["source"] = source
            response["source_truncated"] = truncated
        elif restype == "ncs":
            from src.core.scripting.reference import inspect_ncs

            inline_payload = payload[:512 * 1024]
            try:
                inspected = inspect_ncs(payload, game=game, resref=resref)
                if inspected.recovered_source:
                    source, truncated = _bounded_text(inspected.recovered_source, 750_000)
                    response["decompiled_source"] = source
                    response["decompiled_source_truncated"] = truncated
                    response["decompiler_used"] = True
                else:
                    response["compiled_base64"] = base64.b64encode(inline_payload).decode("ascii")
                    response["note"] = "No source reconstruction was available; compiled NCS bytes are returned."
                response["ghoststudio"]["ncs_instruction_count"] = inspected.instruction_count
                response["ghoststudio"]["ncs_exact_recompile"] = inspected.exact_recompile
            except Exception as exc:
                response["compiled_base64"] = base64.b64encode(inline_payload).decode("ascii")
                response["note"] = "NCS source recovery was unavailable; compiled bytes are returned."
                response["ghoststudio"]["ncs_inspection_error"] = str(exc)
            if len(payload) > len(inline_payload):
                response["compiled_base64_truncated"] = True
                response["ghoststudio"]["inline_binary_limit"] = len(inline_payload)
        elif restype == "tlk":
            from pykotor.resource.formats.tlk import read_tlk

            table = read_tlk(payload)
            samples = []
            for index, row in enumerate(table.entries[:50]):
                if row.text.strip():
                    samples.append({"strref": index, "text": row.text[:80]})
                    if len(samples) == 3:
                        break
            response.update(
                {
                    "type": "TLK",
                    "entry_count": len(table.entries),
                    "language_id": int(table.language.value),
                    "language_name": table.language.name,
                    "sample_entries": samples,
                }
            )
        else:
            try:
                from pykotor.resource.formats.gff import read_gff

                response["fields"] = _legacy_gff_value(read_gff(payload).root, installation, budget=[10_000])
                response["parsed_as"] = "GFF"
            except Exception:
                response["raw_base64"] = base64.b64encode(payload[:4096]).decode("ascii")
                response["note"] = "Unknown format; first 4KB returned as base64."
    except Exception as exc:
        response["parse_error"] = str(exc)
    return response


def _get_quest_compat_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.gff import read_gff

    game = _compat_game(arguments.get("game"))
    game_id = resolve_game(game)
    if game_id is None:
        raise ValueError("game must be K1 or K2")
    quest_id = str(
        arguments.get("questId", arguments.get("quest_id", arguments.get("tag", ""))) or ""
    ).strip()
    if not quest_id:
        raise ValueError("getQuest: 'questId' is required")
    include_scripts = bool(arguments.get("includeScripts", arguments.get("include_scripts", True)))
    include_dialogues = bool(arguments.get("includeDialogues", arguments.get("include_dlg", False)))
    installation = load_installation(game_id)
    journal_entry = installation.get_resource("global", "jrl")
    if journal_entry is None:
        raise FileNotFoundError(f"global.jrl not found in {game}")
    journal_payload = bytes(journal_entry.data)
    if len(journal_payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("global.jrl exceeds the 64 MiB compatibility safety limit")
    root = read_gff(journal_payload).root
    categories = _legacy_list_rows(root, installation, "Categories")
    category = next(
        (row for row in categories if str(row.get("Tag", "")).casefold() == quest_id.casefold()),
        None,
    )
    if category is None:
        raise FileNotFoundError(f"Quest '{quest_id}' not found in global.jrl")

    states = []
    script_resrefs: list[str] = []
    for raw in category.get("EntryList", []) if isinstance(category.get("EntryList"), list) else []:
        if not isinstance(raw, Mapping):
            continue
        state: dict[str, Any] = {
            "id": raw.get("ID", raw.get("Id", 0)),
            "text": raw.get("Text", "") or "",
            "end": bool(raw.get("End", False)),
        }
        for label in (
            "script", "script1", "script2", "scriptAbort", "scriptToRun", "runScript",
            "Script", "OnAccept", "OnFail", "OnEnd", "OnAssign", "SetFlag",
        ):
            value = raw.get(label)
            if value and value != "****":
                state[label] = value
                script_resrefs.append(str(value).casefold())
        states.append(state)
    response: dict[str, Any] = {
        "game": game,
        "quest_id": str(category.get("Tag", quest_id)),
        "name": category.get("Name", "") or "",
        "state_count": len(states),
        "states": states,
        "ghoststudio": {
            "journal_source": str(getattr(journal_entry, "source", "")),
            "journal_sha256": hashlib.sha256(journal_payload).hexdigest(),
            "category_count": len(categories),
            "bounded_response": True,
        },
    }

    unique_scripts = list(dict.fromkeys(script_resrefs))
    if include_scripts:
        scripts: dict[str, str] = {}
        total_chars = 0
        for script in unique_scripts[:200]:
            source_entry = installation.get_resource(script, "nss")
            if source_entry is not None:
                source, _truncated = _bounded_text(bytes(source_entry.data).decode("latin-1", errors="replace"), 250_000)
            else:
                compiled_entry = installation.get_resource(script, "ncs")
                if compiled_entry is None:
                    source = "[not found]"
                else:
                    try:
                        from src.core.scripting.reference import inspect_ncs

                        inspected = inspect_ncs(bytes(compiled_entry.data), game=game, resref=script)
                        source = inspected.recovered_source or f"[compiled NCS — {len(compiled_entry.data)} bytes]"
                    except Exception:
                        source = f"[compiled NCS — {len(compiled_entry.data)} bytes]"
                    source, _truncated = _bounded_text(source, 250_000)
            remaining = max(0, 1_000_000 - total_chars)
            scripts[script] = source[:remaining]
            total_chars += len(scripts[script])
            if total_chars >= 1_000_000:
                response["ghoststudio"]["scripts_truncated"] = True
                break
        response["scripts"] = scripts

    if include_dialogues:
        references: list[dict[str, Any]] = []
        examined = 0
        seen: set[str] = set()
        iterator = getattr(installation, "iter_resources", None)
        if callable(iterator) and unique_scripts:
            resources_seen = 0
            for resource in iterator("all"):
                resources_seen += 1
                if resources_seen > 50_000:
                    response["ghoststudio"]["dialogue_resource_scan_truncated"] = True
                    break
                if _resource_entry_type(resource) != "dlg":
                    continue
                dlg_resref = str(getattr(resource, "resref", "") or "").casefold()
                if not dlg_resref or dlg_resref in seen:
                    continue
                seen.add(dlg_resref)
                examined += 1
                if examined > 50:
                    break
                try:
                    dlg_root = read_gff(bytes(resource.data)).root
                    hits = []
                    for collection in ("EntryList", "ReplyList"):
                        for index, row in enumerate(_legacy_list_rows(dlg_root, installation, collection)):
                            for label in ("Script", "Script2", "ActionParam1", "Active"):
                                value = row.get(label)
                                if value and str(value).casefold() in unique_scripts:
                                    hits.append(f"entry #{index}: {value!r}")
                    if hits:
                        references.append({"dlg_resref": dlg_resref, "references": hits[:5]})
                        if len(references) == 20:
                            break
                except Exception:
                    continue
        response["dialogues_referencing_quest"] = references
        response["ghoststudio"]["dialogues_examined"] = examined
        response["ghoststudio"]["dialogue_scan_limit"] = 50
        response["ghoststudio"]["dialogue_resource_scan_limit"] = 50_000
    return response


def _module_resource_entries(
    installation: Any,
    module_id: str,
) -> tuple[list[Any], list[str], bool, int, Counter[str]]:
    iterator = getattr(installation, "iter_resources", None)
    if not callable(iterator):
        return [], [], False, 0, Counter()
    entries = []
    sources: list[str] = []
    truncated = False
    resource_count = 0
    type_breakdown: Counter[str] = Counter()
    for index, entry in enumerate(iterator(f"module:{module_id}")):
        if index >= 100_000:
            truncated = True
            break
        resource_count += 1
        restype = _resource_entry_type(entry)
        type_breakdown[restype] += 1
        if restype in {"ifo", "git"}:
            entries.append(entry)
        source = str(getattr(entry, "source", ""))
        if source and source not in sources:
            sources.append(source)
    return entries, sources, truncated, resource_count, type_breakdown


def _module_capsule_paths(installation: Any, module_id: str, sources: Sequence[str]) -> list[Path]:
    path_getter = getattr(installation, "path", None)
    if not callable(path_getter):
        return []
    modules_root = (Path(str(path_getter())) / "Modules").resolve()
    names = []
    for source in sources:
        value = str(source)
        if value.casefold().startswith("module:"):
            value = value.split(":", 1)[1]
        if value:
            names.append(Path(value).name)
    names.extend((f"{module_id}.mod", f"{module_id}.rim", f"{module_id}_s.rim", f"{module_id}_dlg.erf"))
    paths = []
    seen: set[str] = set()
    for name in names:
        candidate = (modules_root / name).resolve()
        identity = str(candidate).casefold()
        if identity in seen or not candidate.is_relative_to(modules_root):
            continue
        seen.add(identity)
        if candidate.is_file() and candidate.suffix.casefold() in {".mod", ".rim", ".erf"}:
            paths.append(candidate)
    return paths


def _module_capsule_entry(
    installation: Any,
    module_id: str,
    sources: Sequence[str],
    resref: str,
    restype: str,
) -> Any | None:
    from pykotor.extract.capsule import Capsule
    from pykotor.resource.type import ResourceType

    resource_type = ResourceType.from_extension(restype.casefold().lstrip("."))
    for path in _module_capsule_paths(installation, module_id, sources):
        try:
            payload = Capsule(path).resource(resref, resource_type)
        except Exception:
            continue
        if payload is not None:
            return SimpleNamespace(
                resref=resref,
                restype=restype.upper(),
                extension=restype.casefold(),
                data=bytes(payload),
                source=str(path),
            )
    return None


def _get_module_compat_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.gff import read_gff

    game = _compat_game(arguments.get("game"))
    game_id = resolve_game(game)
    if game_id is None:
        raise ValueError("game must be K1 or K2")
    module_id = _compat_resref(arguments.get("module_id", arguments.get("module_root")), label="module_id")
    include_git = bool(arguments.get("include_git", False))
    installation = load_installation(game_id)
    entries, module_sources, resource_scan_truncated, resource_count, type_breakdown = _module_resource_entries(
        installation, module_id
    )
    ifo_entry = next(
        (
            entry
            for entry in entries
            if _resource_entry_type(entry) == "ifo" and str(getattr(entry, "resref", "")).casefold() == "module"
        ),
        None,
    )
    if ifo_entry is None or not bytes(getattr(ifo_entry, "data", b"")):
        ifo_entry = _module_capsule_entry(installation, module_id, module_sources, "module", "ifo")
    if ifo_entry is None:
        ifo_entry = installation.get_resource("module", "ifo")
    if ifo_entry is None:
        raise FileNotFoundError(f"module.ifo not found in the '{module_id}' capsule")
    ifo_payload = bytes(ifo_entry.data)
    if len(ifo_payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("module.ifo exceeds the 64 MiB compatibility safety limit")
    root = read_gff(ifo_payload).root
    area_rows = _legacy_list_rows(root, installation, "Mod_Area_list")
    if not area_rows:
        area_rows = _legacy_list_rows(root, installation, "Area_list")
    areas = [
        str(row.get("Area_Name", row.get("AreaName", "")))
        for row in area_rows
        if row.get("Area_Name", row.get("AreaName"))
    ]
    module_script_labels = (
        "Mod_OnAcquirItem", "Mod_OnActivateItem", "Mod_OnClientEntr", "Mod_OnClientLeav", "Mod_OnHeartbeat",
        "Mod_OnModLoad", "Mod_OnModStart", "Mod_OnPlrDeath", "Mod_OnPlrDying", "Mod_OnPlrLvlUp",
        "Mod_OnPlrRest", "Mod_OnPlrRespawn", "Mod_OnUnAqreItem", "Mod_OnUsrDefined", "OnAcquireItem",
        "OnActivateItem", "OnClientEnter", "OnClientLeave", "OnHeartbeat", "OnLoad", "OnModuleStart",
        "OnPlayerDeath", "OnPlayerDying", "OnPlayerLevelUp", "OnPlayerRest", "OnPlayerRespawn",
        "OnUnacquireItem", "OnUserDefined",
    )
    scripts = _legacy_script_fields(root, installation, module_script_labels)
    area_summaries: list[dict[str, Any]] = []
    if include_git:
        for area in areas[:1_000]:
            git_entry = next(
                (
                    entry
                    for entry in entries
                    if _resource_entry_type(entry) == "git"
                    and str(getattr(entry, "resref", "")).casefold() == area.casefold()
                ),
                None,
            )
            if git_entry is None or not bytes(getattr(git_entry, "data", b"")):
                git_entry = _module_capsule_entry(installation, module_id, module_sources, area, "git")
            if git_entry is None:
                git_entry = installation.get_resource(area, "git")
            if git_entry is None:
                area_summaries.append({"area": area, "error": "GIT not found"})
                continue
            try:
                git_root = read_gff(bytes(git_entry.data)).root
                summary: dict[str, Any] = {"area": area}
                for key, label in {
                    "creatures": "Creature List", "doors": "Door List", "placeables": "Placeable List",
                    "waypoints": "Waypoint List", "triggers": "TriggerList", "stores": "StoreList",
                    "sounds": "SoundList", "encounters": "EncounterList",
                }.items():
                    raw_list = _legacy_root_raw(git_root, label)
                    summary[key] = len(raw_list) if raw_list is not None else 0
                area_summaries.append(summary)
            except Exception as exc:
                area_summaries.append({"area": area, "error": str(exc)})

    def first_value(*labels: str) -> Any:
        return _legacy_root_value(root, installation, *labels)

    fallback_source = str(getattr(ifo_entry, "source", ""))
    resolved_sources = [str(path) for path in _module_capsule_paths(installation, module_id, module_sources)]
    return {
        "game": game,
        "module_id": module_id,
        "module_sources": resolved_sources or module_sources or ([fallback_source] if fallback_source else []),
        "mod_name": first_value("Mod_Name", "ModName"),
        "tag": first_value("Mod_Tag", "Tag"),
        "vo_id": first_value("Mod_VO_ID", "VO_ID"),
        "entry_area": first_value("Mod_Entry_Area", "EntryArea"),
        "entry_x": first_value("Mod_Entry_X", "EntryX"),
        "entry_y": first_value("Mod_Entry_Y", "EntryY"),
        "entry_z": first_value("Mod_Entry_Z", "EntryZ"),
        "entry_dir_x": first_value("Mod_Entry_Dir_X", "EntryDirX"),
        "entry_dir_y": first_value("Mod_Entry_Dir_Y", "EntryDirY"),
        "areas": areas,
        "area_summaries": area_summaries,
        "scripts": scripts,
        "fields": _legacy_gff_value(root, installation, budget=[20_000]),
        "ghoststudio": {
            "ifo_source": str(getattr(ifo_entry, "source", "")),
            "ifo_byte_count": len(ifo_payload),
            "ifo_sha256": hashlib.sha256(ifo_payload).hexdigest(),
            "resource_count": resource_count,
            "type_breakdown": dict(sorted(type_breakdown.items())),
            "resource_scan_truncated": resource_scan_truncated,
            "bounded_response": True,
        },
    }


def _get_script_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.reference import inspect_ncs

    game_id = resolve_game(str(arguments.get("game", "")))
    if game_id is None:
        raise ValueError("game must be k1 or k2")
    game = str(arguments.get("game", "K2") or "K2")
    resref = str(arguments.get("resref", "") or "").strip()
    if not resref:
        raise ValueError("resref is required")
    installation = load_installation(game_id)
    nss_entry = installation.get_resource(resref, "nss")
    ncs_entry = installation.get_resource(resref, "ncs")
    if nss_entry is None and ncs_entry is None:
        raise FileNotFoundError(f"Neither {resref}.nss nor {resref}.ncs was found")
    limit = min(750_000, max(1_000, int(arguments.get("max_text_chars", 250_000) or 250_000)))
    response: dict[str, Any] = {
        "game": game.upper(),
        "resref": resref,
        "nss_found": nss_entry is not None,
        "ncs_found": ncs_entry is not None,
        "source_type": "not_found",
    }
    if nss_entry is not None:
        source = bytes(nss_entry.data).decode("utf-8", errors="replace")
        shown_source, truncated = _bounded_text(source, limit)
        response["source_type"] = "nss_source"
        if bool(arguments.get("include_source", True)):
            response["source"] = shown_source
        response["nss"] = {
            "source": shown_source if bool(arguments.get("include_source", True)) else "",
            "source_truncated": truncated if bool(arguments.get("include_source", True)) else False,
            "byte_count": len(nss_entry.data),
            "sha256": hashlib.sha256(bytes(nss_entry.data)).hexdigest(),
            "source_location": str(getattr(nss_entry, "source", "")),
        }
    if ncs_entry is not None:
        payload = bytes(ncs_entry.data)
        compiled: dict[str, Any] = {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_location": str(getattr(ncs_entry, "source", "")),
        }
        try:
            inspected = inspect_ncs(
                payload,
                game=game,
                resref=resref,
                include_dirs=tuple(arguments.get("include_dirs", ()) or ()),
            )
            disassembly, disassembly_truncated = _bounded_text(inspected.disassembly, limit)
            recovered, recovered_truncated = _bounded_text(inspected.recovered_source, limit)
            compiled.update(
                {
                    "instruction_count": inspected.instruction_count,
                    "disassembly": disassembly if bool(arguments.get("include_disassembly", True)) else "",
                    "disassembly_truncated": disassembly_truncated if bool(arguments.get("include_disassembly", True)) else False,
                    "recovered_source": recovered if bool(arguments.get("include_source", True)) else "",
                    "recovered_source_truncated": recovered_truncated if bool(arguments.get("include_source", True)) else False,
                    "exact_recompile": inspected.exact_recompile,
                    "recompile_error": inspected.recompile_error,
                }
            )
            if nss_entry is None and inspected.recovered_source:
                response["source_type"] = "ncs_decompiled"
                if bool(arguments.get("include_source", True)):
                    response["source"] = recovered
            elif nss_entry is None:
                response["source_type"] = "ncs_binary_only"
            response["analysis"] = {
                "instruction_count": inspected.instruction_count,
                "exact_recompile": inspected.exact_recompile,
                "recompile_error": inspected.recompile_error,
            }
        except Exception as exc:
            compiled["inspection_error"] = str(exc).strip() or exc.__class__.__name__
            if nss_entry is None:
                response["source_type"] = "ncs_binary_only"
                response["note"] = "Script is compiled NCS and source recovery did not succeed."
        response["ncs"] = compiled
    return response


def _gff_component_payload(entry: Any, *, max_depth: int, max_fields: int) -> tuple[dict[str, Any], Any]:
    from pykotor.resource.formats.gff import read_gff
    from kotormcp.tools.gffdata import _gff_struct_to_dict

    payload = bytes(entry.data)
    gff = read_gff(payload)
    lists = {
        str(label): len(value)
        for label, field_type, value in gff.root
        if getattr(field_type, "name", "") == "List"
    }
    return (
        {
            "content_type": getattr(gff.content, "name", str(gff.content)),
            "source": str(getattr(entry, "source", "")),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "top_level_list_counts": lists,
            "root": _gff_struct_to_dict(gff.root, max_depth=max_depth, max_fields=max_fields),
        },
        gff,
    )


def _get_area_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    game_id = resolve_game(str(arguments.get("game", "")))
    if game_id is None:
        raise ValueError("game must be k1 or k2")
    resref = str(arguments.get("resref", "") or "").strip()
    if not resref:
        raise ValueError("resref is required")
    installation = load_installation(game_id)
    are_entry = installation.get_resource(resref, "are")
    if are_entry is None:
        raise FileNotFoundError(f"{resref}.are was not found")
    max_depth = min(12, max(1, int(arguments.get("max_depth", 5) or 5)))
    max_fields = min(1000, max(1, int(arguments.get("max_fields", 300) or 300)))
    are, are_gff = _gff_component_payload(are_entry, max_depth=max_depth, max_fields=max_fields)
    git_entry = installation.get_resource(resref, "git")
    git = None
    git_gff = None
    if git_entry is not None:
        git, git_gff = _gff_component_payload(git_entry, max_depth=max_depth, max_fields=max_fields)
    name_payload = None
    name = are_gff.root.get_locstring("Name", None)
    if name is not None:
        name_payload = _localized_field_payload(name, installation)
    are_root = are_gff.root
    response: dict[str, Any] = {
        "game": str(arguments.get("game", "")).upper(),
        "resref": resref,
        "module_id": str(arguments.get("module_id", arguments.get("moduleId", "")) or "") or None,
        "tag": are_gff.root.get_string("Tag", ""),
        "localized_name": name_payload,
        "are": are,
        "git_found": git is not None,
        "git": git,
        "instance_list_counts": git["top_level_list_counts"] if git is not None else {},
    }
    response.update(
        {
            "area_name": _legacy_root_value(are_root, installation, "Name", "AreaName"),
            "tileset": _legacy_root_value(are_root, installation, "Tileset"),
            "flags": _legacy_root_value(are_root, installation, "Flags"),
            "camera_style": _legacy_root_value(are_root, installation, "CameraStyle"),
            "default_envmap": _legacy_root_value(are_root, installation, "DefaultEnvMap"),
            "ambient_sound": _legacy_root_value(are_root, installation, "AmbientSndDay", "AmbientSnd"),
            "ambient_volume": _legacy_root_value(are_root, installation, "AmbientSndDayVol", "AmbientSndVol"),
            "ambient_battle_sound": _legacy_root_value(are_root, installation, "AmbientSndBat", "EnvAudio"),
            "music_standard": _legacy_root_value(are_root, installation, "MusicDay", "MusicStandard"),
            "music_battle": _legacy_root_value(are_root, installation, "MusicBattle"),
            "music_delay": _legacy_root_value(are_root, installation, "MusicDelay"),
            "fog_enabled": _legacy_root_value(are_root, installation, "SunFogOn", "FogOn"),
            "fog_near": _legacy_root_value(are_root, installation, "SunFogNear", "FogNear"),
            "fog_far": _legacy_root_value(are_root, installation, "SunFogFar", "FogFar"),
            "fog_color": _legacy_root_value(are_root, installation, "SunFogColor"),
            "sun_ambient_color": _legacy_root_value(are_root, installation, "SunAmbientColor"),
            "sun_diffuse_color": _legacy_root_value(are_root, installation, "SunDiffuseColor"),
            "dynamic_light_color": _legacy_root_value(are_root, installation, "DynAmbientColor"),
            "shadows": _legacy_root_value(are_root, installation, "SunShadows"),
            "grass_texture": _legacy_root_value(are_root, installation, "Grass_TexName"),
            "grass_density": _legacy_root_value(are_root, installation, "Grass_Density"),
            "grass_size": _legacy_root_value(are_root, installation, "Grass_QuadSize"),
            "wind_power": _legacy_root_value(are_root, installation, "WindPower"),
            "unescapable": _legacy_root_value(are_root, installation, "Unescapable"),
            "disable_transit": _legacy_root_value(are_root, installation, "DisableTransit"),
            "weather": _legacy_root_value(are_root, installation, "WeatherID"),
            "chance_rain": _legacy_root_value(are_root, installation, "ChanceRain"),
            "chance_snow": _legacy_root_value(are_root, installation, "ChanceSnow"),
            "chance_lightning": _legacy_root_value(are_root, installation, "ChanceLightning"),
            "stealth_xp_enabled": _legacy_root_value(are_root, installation, "StealthXPEnabled"),
            "stealth_xp_loss": _legacy_root_value(are_root, installation, "StealthXPLoss"),
            "stealth_xp_max": _legacy_root_value(are_root, installation, "StealthXPMax"),
            "no_rest": _legacy_root_value(are_root, installation, "NoRest"),
            "scripts": _legacy_script_fields(are_root, installation, ("OnEnter", "OnExit", "OnHeartbeat", "OnUserDefined")),
            "rooms": [],
            "creatures": [],
            "doors": [],
            "placeables": [],
            "waypoints": [],
            "triggers": [],
            "stores": [],
            "sounds": [],
            "encounters": [],
            "cameras": [],
            "use_templates": None,
            "current_weather": None,
            "weather_started": None,
            "parse_errors": [],
        }
    )

    lyt_entry = installation.get_resource(resref, "lyt")
    if lyt_entry is not None:
        try:
            for line in bytes(lyt_entry.data).decode("latin-1", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) < 4 or parts[0].casefold() in {
                    "beginlayout", "donelayout", "roomcount", "trackcount", "obstaclecount", "doorhookcount",
                }:
                    continue
                try:
                    response["rooms"].append(
                        {"room": parts[0], "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3])}
                    )
                except ValueError:
                    continue
        except Exception as exc:
            response["parse_errors"].append(f"LYT parse error: {exc}")

    if git_gff is not None:
        git_root = git_gff.root

        def instance_rows(label: str) -> list[dict[str, Any]]:
            rows = []
            for raw in _legacy_list_rows(git_root, installation, label):
                row: dict[str, Any] = {}
                tag = raw.get("Tag") or raw.get("tag")
                reference = raw.get("TemplateResRef") or raw.get("ResRef") or raw.get("resref")
                if tag:
                    row["tag"] = tag
                if reference:
                    row["resref"] = reference
                if any(raw.get(key) is not None for key in ("XPosition", "YPosition", "ZPosition")):
                    row.update(
                        {
                            "x": raw.get("XPosition"),
                            "y": raw.get("YPosition"),
                            "z": raw.get("ZPosition"),
                        }
                    )
                rows.append(row)
            return rows

        for key, label in {
            "creatures": "Creature List",
            "doors": "Door List",
            "placeables": "Placeable List",
            "waypoints": "Waypoint List",
            "triggers": "TriggerList",
            "stores": "StoreList",
            "sounds": "SoundList",
            "encounters": "EncounterList",
        }.items():
            response[key] = instance_rows(label)
        response["use_templates"] = _legacy_root_value(git_root, installation, "UseTemplates")
        response["current_weather"] = _legacy_root_value(git_root, installation, "CurrentWeather")
        response["weather_started"] = _legacy_root_value(git_root, installation, "WeatherStarted")

        area_properties = _legacy_root_raw(git_root, "AreaProperties")
        if area_properties is not None:
            for result_key, labels in {
                "ambient_sound": ("AmbientSndDay",),
                "ambient_volume": ("AmbientSndDayVol",),
                "ambient_battle_sound": ("EnvAudio",),
                "music_standard": ("MusicDay",),
                "music_battle": ("MusicBattle",),
                "music_delay": ("MusicDelay",),
            }.items():
                if response[result_key] is None:
                    response[result_key] = _legacy_root_value(area_properties, installation, *labels)

        for raw in _legacy_list_rows(git_root, installation, "CameraList"):
            camera: dict[str, Any] = {}
            if raw.get("CameraID") is not None:
                camera["camera_id"] = raw["CameraID"]
            for source, target in (
                ("XPosition", "x"), ("YPosition", "y"), ("ZPosition", "z"), ("Pitch", "pitch")
            ):
                if raw.get(source) is not None:
                    camera[target] = raw[source]
            response["cameras"].append(camera)
    return response


def _get_faction_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    game_id = resolve_game(str(arguments.get("game", "")))
    if game_id is None:
        raise ValueError("game must be k1 or k2")
    resref = str(arguments.get("resref", "repute") or "repute").strip()
    installation = load_installation(game_id)
    entry = installation.get_resource(resref, "fac")
    if entry is None:
        raise FileNotFoundError(f"{resref}.fac was not found")
    max_depth = min(12, max(1, int(arguments.get("max_depth", 6) or 6)))
    max_fields = min(1000, max(1, int(arguments.get("max_fields", 500) or 500)))
    component, gff = _gff_component_payload(entry, max_depth=max_depth, max_fields=max_fields)
    fields = _legacy_gff_value(gff.root, installation, budget=[10_000])
    factions: list[dict[str, Any]] = []
    for index, row in enumerate(_legacy_list_rows(gff.root, installation, "FactionList")):
        name = row.get("FactionName") or row.get("Name") or f"faction_{index}"
        reputation_rows = row.get("RepList", [])
        reputation = []
        if isinstance(reputation_rows, list):
            for relation in reputation_rows:
                if not isinstance(relation, Mapping) or relation.get("FactionID") is None:
                    continue
                reputation.append(
                    {
                        "target_faction_index": int(relation["FactionID"]),
                        "reputation": int(relation["FactionRep"]) if relation.get("FactionRep") is not None else None,
                    }
                )
        factions.append({"index": index, "name": str(name), "rep_entries": reputation})
    return {
        "game": str(arguments.get("game", "")).upper(),
        "resref": resref,
        "factions": factions,
        "count": len(factions),
        "fields": fields,
        "faction": component,
        "relation_list_counts": component["top_level_list_counts"],
    }


_BLUEPRINT_TYPE_ORDER: tuple[str, ...] = (
    "utc", "utp", "utd", "uti", "ute", "utt", "utw", "utm", "uts",
    "bic", "btc", "btp", "btd", "bti", "bte", "btt", "btm",
)


def _localized_field_payload(value: Any, installation: Any) -> dict[str, Any]:
    stringref = int(getattr(value, "stringref", -1))
    text = ""
    if stringref >= 0:
        try:
            text = str(installation.talktable_string(stringref))
        except Exception:
            text = ""
    return {
        "stringref": stringref,
        "resolved_text": text,
        "localized_data": _json_value(value.to_dict()) if hasattr(value, "to_dict") else str(value),
    }


def _legacy_gff_value(value: Any, installation: Any, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Convert GFF values to the bounded, convenient shape legacy composites exposed.

    The typed field rows returned elsewhere in the response remain the lossless
    evidence.  This view intentionally resolves ResRefs and localized strings
    into the simple values expected by GhostScripter composite consumers.
    """

    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import ResRef
    from pykotor.resource.formats.gff import GFFList, GFFStruct

    if budget is None:
        budget = [10_000]
    if budget[0] <= 0:
        return "<compatibility view truncated>"
    budget[0] -= 1
    if depth > 8:
        return "<compatibility depth limit>"
    if isinstance(value, LocalizedString):
        inline = value.get(0, 0, use_fallback=True)
        if inline:
            return inline
        if value.stringref >= 0:
            try:
                return str(installation.talktable_string(int(value.stringref)))
            except Exception:
                return ""
        return ""
    if isinstance(value, ResRef):
        return str(value)
    if isinstance(value, GFFStruct):
        result: dict[str, Any] = {}
        for label, _field_type, child in value:
            if budget[0] <= 0:
                result["_truncated"] = True
                break
            result[str(label)] = _legacy_gff_value(child, installation, depth=depth + 1, budget=budget)
        return result
    if isinstance(value, GFFList):
        result = []
        for child in value:
            if budget[0] <= 0:
                break
            result.append(_legacy_gff_value(child, installation, depth=depth + 1, budget=budget))
        return result
    return _json_value(value)


def _legacy_root_value(root: Any, installation: Any, *labels: str, default: Any = None) -> Any:
    for label in labels:
        try:
            if root.exists(label):
                return _legacy_gff_value(root.value(label), installation)
        except Exception:
            continue
    return default


def _legacy_root_raw(root: Any, *labels: str) -> Any:
    for label in labels:
        try:
            if root.exists(label):
                return root.value(label)
        except Exception:
            continue
    return None


def _legacy_script_fields(root: Any, installation: Any, labels: Sequence[str] | None = None) -> dict[str, Any]:
    wanted = set(labels or ())
    scripts: dict[str, Any] = {}
    for label, _field_type, value in root:
        text = str(label)
        if wanted and text not in wanted:
            continue
        if not wanted and not (text.startswith("Script") or text.startswith("On")):
            continue
        parsed = _legacy_gff_value(value, installation)
        if parsed not in (None, "", "****", 0, False):
            scripts[text] = parsed
    return scripts


def _legacy_blueprint_common(root: Any, installation: Any, restype: str) -> dict[str, Any]:
    return {
        "type": restype,
        "tag": _legacy_root_value(root, installation, "Tag", "tag"),
        "template_resref": _legacy_root_value(root, installation, "TemplateResRef", "ResRef"),
        "comment": _legacy_root_value(root, installation, "Comment"),
        "name": _legacy_root_value(root, installation, "LocalizedName", "Name", "FirstName"),
        "description": _legacy_root_value(root, installation, "Description", "DescIdentified"),
        "scripts": _legacy_script_fields(root, installation),
    }


def _legacy_lock_payload(root: Any, installation: Any) -> dict[str, Any]:
    return {
        "locked": _legacy_root_value(root, installation, "Locked"),
        "key_required": _legacy_root_value(root, installation, "KeyRequired"),
        "key_name": _legacy_root_value(root, installation, "KeyName"),
        "lock_dc": _legacy_root_value(root, installation, "LockDC"),
        "open_lock_dc": _legacy_root_value(root, installation, "OpenLockDC"),
    }


def _legacy_trap_payload(root: Any, installation: Any, *, include_type: bool = True) -> dict[str, Any]:
    result = {
        "detectable": _legacy_root_value(root, installation, "TrapDetectable"),
        "disarmable": _legacy_root_value(root, installation, "TrapDisarmable"),
        "trap_dc": _legacy_root_value(root, installation, "TrapDetectDC"),
        "disarm_dc": _legacy_root_value(root, installation, "DisarmDC"),
    }
    if include_type:
        result["trap_type"] = _legacy_root_value(root, installation, "TrapType")
    return result


def _legacy_list_rows(root: Any, installation: Any, label: str) -> list[dict[str, Any]]:
    raw = _legacy_root_raw(root, label)
    if raw is None:
        return []
    rows = _legacy_gff_value(raw, installation, budget=[10_000])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _legacy_creature_payload(payload: bytes, root: Any, installation: Any) -> dict[str, Any]:
    from pykotor.resource.generics.utc import read_utc

    utc = read_utc(payload)
    first = _legacy_root_value(root, installation, "FirstName") or ""
    last = _legacy_root_value(root, installation, "LastName") or ""
    classes = [
        {
            "class_id": row.class_id,
            "level": row.class_level,
            "powers": list(row.powers),
        }
        for row in utc.classes
    ]
    equipment = {
        slot.name.casefold(): {
            "resref": str(item.resref),
            "droppable": bool(item.droppable),
            "infinite": bool(item.infinite),
        }
        for slot, item in utc.equipment.items()
    }
    skill_names = (
        "computer_use", "demolitions", "stealth", "awareness",
        "persuade", "repair", "security", "treat_injury",
    )
    scripts: dict[str, str] = {}
    for label, attribute in {
        "ScriptSpawn": "on_spawn",
        "ScriptDeath": "on_death",
        "ScriptOnNotice": "on_notice",
        "ScriptAttacked": "on_attacked",
        "ScriptDamaged": "on_damaged",
        "ScriptEndRound": "on_end_round",
        "ScriptHeartbeat": "on_heartbeat",
        "ScriptOnBlocked": "on_blocked",
        "ScriptDialogue": "on_dialog",
        "ScriptEndDialogu": "on_end_dialog",
        "ScriptDisturbed": "on_disturbed",
        "ScriptRested": "on_rested",
        "ScriptSpellAt": "on_spell",
        "ScriptUserDefine": "on_user_defined",
    }.items():
        value = str(getattr(utc, attribute))
        if value and value != "****":
            scripts[label] = value
    return {
        "tag": utc.tag,
        "name": " ".join(part for part in (str(first), str(last)) if part) or None,
        "race": utc.race_id,
        "subrace": utc.subrace_id,
        "gender": utc.gender_id,
        "classes": classes,
        "str": utc.strength,
        "dex": utc.dexterity,
        "con": utc.constitution,
        "int_": utc.intelligence,
        "wis": utc.wisdom,
        "cha": utc.charisma,
        "hp": utc.current_hp,
        "base_hp": utc.hp,
        "max_hp": utc.max_hp,
        "ac": utc.natural_ac,
        "appearance": utc.appearance_id,
        "faction": utc.faction_id,
        "conversation": str(utc.conversation),
        "equipment": equipment,
        "feats": list(utc.feats),
        "skills": {name: int(getattr(utc, name)) for name in skill_names},
        "scripts": scripts,
    }


def _legacy_named_blueprint_payload(
    name: str,
    restype: str,
    payload: bytes,
    root: Any,
    installation: Any,
) -> dict[str, Any]:
    """Return additive top-level keys matching legacy composite object views."""

    if name == "getCreature":
        return _legacy_creature_payload(payload, root, installation)
    if name == "getNpc":
        result = {
            "utc_fields": _legacy_gff_value(root, installation, budget=[10_000]),
            "scripts": _legacy_script_fields(
                root,
                installation,
                (
                    "ScriptHeartbeat", "ScriptOnNotice", "ScriptSpellAt", "ScriptAttacked",
                    "ScriptDamaged", "ScriptDisturbed", "ScriptEndRound", "ScriptEndDialogue",
                    "ScriptDialogue", "ScriptSpawn", "ScriptDeath", "ScriptUserDefined",
                ),
            ),
        }
        appearance = _legacy_root_value(root, installation, "Appearance_Type", "AppearanceType")
        if isinstance(appearance, int) and appearance >= 0:
            result["appearance_row"] = appearance
            entry = installation.get_resource("appearance", "2da")
            if entry is not None:
                try:
                    from pykotor.resource.formats.twoda import read_2da

                    result["appearance_label"] = read_2da(bytes(entry.data)).get_cell_safe(
                        appearance, "label", f"row {appearance}"
                    )
                except Exception as exc:
                    result["appearance_note"] = str(exc)
        faction = _legacy_root_value(root, installation, "FactionID")
        if isinstance(faction, int) and faction >= 0:
            result["faction_row"] = faction
            entry = installation.get_resource("repute", "2da")
            if entry is not None:
                try:
                    from pykotor.resource.formats.twoda import read_2da

                    result["faction"] = read_2da(bytes(entry.data)).get_cell_safe(
                        faction, "label", f"faction {faction}"
                    )
                except Exception:
                    pass
        conversation = _legacy_root_value(root, installation, "Conversation")
        if conversation and conversation != "****":
            result["dialogue_resref"] = conversation
            dialogue_entry = installation.get_resource(str(conversation), "dlg")
            if dialogue_entry is not None:
                try:
                    from pykotor.resource.formats.gff import read_gff

                    dialogue_root = read_gff(bytes(dialogue_entry.data)).root
                    entries = _legacy_list_rows(dialogue_root, installation, "EntryList")
                    replies = _legacy_list_rows(dialogue_root, installation, "ReplyList")
                    starters = _legacy_list_rows(dialogue_root, installation, "StartingList")
                    opening = ""
                    if starters:
                        first_index = starters[0].get("Index", 0)
                        if isinstance(first_index, int) and 0 <= first_index < len(entries):
                            opening = str(entries[first_index].get("Text", "") or "")[:200]
                    result["dialogue_summary"] = {
                        "entry_count": len(entries),
                        "reply_count": len(replies),
                        "opening_line": opening,
                    }
                except Exception as exc:
                    result["dialogue_error"] = str(exc)
        return result
    if name == "getDoor":
        result = {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "local_ident": _legacy_root_value(root, installation, "LocName"),
            "generic_type": _legacy_root_value(root, installation, "GenericType"),
            "animation_state": _legacy_root_value(root, installation, "AnimationState"),
            "lock": _legacy_lock_payload(root, installation),
            "trap": _legacy_trap_payload(root, installation),
            "scripts": _legacy_script_fields(
                root,
                installation,
                (
                    "OnClick", "OnClosed", "OnDamaged", "OnDeath", "OnDisarm", "OnFailToOpen",
                    "OnHeartbeat", "OnLock", "OnMeleeAttacked", "OnOpen", "OnSpellCastAt",
                    "OnTrapTriggered", "OnUnlock", "OnUserDefined", "LinkedTo", "LinkedToFlags",
                ),
            ),
        }
        conversation = _legacy_root_value(root, installation, "Conversation")
        if conversation and conversation != "****":
            result["conversation"] = conversation
        return result
    if name == "getPlaceable":
        inventory = []
        for row in _legacy_list_rows(root, installation, "ItemList"):
            item = row.get("InventoryRes") or row.get("ResRef") or row.get("resref")
            if item:
                inventory.append(item)
        result = {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "local_ident": _legacy_root_value(root, installation, "LocName"),
            "appearance": _legacy_root_value(root, installation, "Appearance"),
            "static": _legacy_root_value(root, installation, "Static"),
            "useable": _legacy_root_value(root, installation, "Useable"),
            "lock": _legacy_lock_payload(root, installation),
            "trap": _legacy_trap_payload(root, installation),
            "inventory": inventory,
            "scripts": _legacy_script_fields(
                root,
                installation,
                (
                    "OnClick", "OnClosed", "OnDamaged", "OnDeath", "OnDisarm", "OnHeartbeat",
                    "OnInvDisturbed", "OnLock", "OnMeleeAttacked", "OnOpen", "OnSpellCastAt",
                    "OnTrapTriggered", "OnUnlock", "OnUsed", "OnUserDefined",
                ),
            ),
        }
        conversation = _legacy_root_value(root, installation, "Conversation")
        if conversation and conversation != "****":
            result["conversation"] = conversation
        return result
    if name == "getItem":
        properties = []
        property_names = (
            "PropertyName", "Subtype", "CostTable", "CostValue", "Param1", "Param1Value",
            "ChanceAppear", "UpgradeType",
        )
        for row in _legacy_list_rows(root, installation, "PropertiesList"):
            properties.append({key: row[key] for key in property_names if key in row})
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "base_item": _legacy_root_value(root, installation, "BaseItem"),
            "stack_size": _legacy_root_value(root, installation, "StackSize"),
            "cost": _legacy_root_value(root, installation, "Cost"),
            "add_cost": _legacy_root_value(root, installation, "AddCost"),
            "stolen": _legacy_root_value(root, installation, "Stolen"),
            "identified": _legacy_root_value(root, installation, "Identified"),
            "charges": _legacy_root_value(root, installation, "Charges"),
            "upgrade_level": _legacy_root_value(root, installation, "UpgradeLevel"),
            "name": _legacy_root_value(root, installation, "LocalizedName", "Name"),
            "description": _legacy_root_value(root, installation, "DescIdentified", "Description"),
            "properties": properties,
            "scripts": _legacy_script_fields(root, installation, ("OnActivated", "OnHeartbeat", "OnUserDefined")),
        }
    if name == "getEncounter":
        spawn_list = []
        for row in _legacy_list_rows(root, installation, "CreatureList"):
            creature = row.get("ResRef") or row.get("resref")
            if creature:
                spawn_list.append(
                    {
                        "resref": creature,
                        "cr": float(row["CR"]) if row.get("CR") is not None else None,
                        "single_spawn": bool(row.get("SingleSpawn")),
                    }
                )
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "active": _legacy_root_value(root, installation, "Active"),
            "difficulty": _legacy_root_value(root, installation, "DifficultyIndex", "Difficulty"),
            "faction": _legacy_root_value(root, installation, "Faction"),
            "spawn_list": spawn_list,
            "scripts": _legacy_script_fields(
                root, installation, ("OnEntered", "OnExhausted", "OnExit", "OnHeartbeat", "OnSpawn", "OnUserDefined")
            ),
        }
    if name == "getTrigger":
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "trap_type": _legacy_root_value(root, installation, "TrapType"),
            "trap_one_shot": _legacy_root_value(root, installation, "TrapOneShot"),
            "linked_to": _legacy_root_value(root, installation, "LinkedTo"),
            "trap": _legacy_trap_payload(root, installation, include_type=False),
            "scripts": _legacy_script_fields(
                root,
                installation,
                ("ScriptHeartbeat", "ScriptOnEnter", "ScriptOnExit", "ScriptUserDefine", "OnTrapTriggered", "OnDisarm", "OnClick"),
            ),
        }
    if name == "getWaypoint":
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "name": _legacy_root_value(root, installation, "LocalizedName", "LocName"),
            "x": _legacy_root_value(root, installation, "XPosition"),
            "y": _legacy_root_value(root, installation, "YPosition"),
            "z": _legacy_root_value(root, installation, "ZPosition"),
            "dir_x": _legacy_root_value(root, installation, "XOrientation"),
            "dir_y": _legacy_root_value(root, installation, "YOrientation"),
            "has_map_note": _legacy_root_value(root, installation, "HasMapNote"),
            "map_note_enabled": _legacy_root_value(root, installation, "MapNoteEnabled"),
            "map_note": _legacy_root_value(root, installation, "MapNote"),
        }
    if name == "getStore":
        inventory = []
        for row in _legacy_list_rows(root, installation, "ItemList"):
            item = row.get("InventoryRes") or row.get("ResRef") or row.get("resref")
            if item:
                inventory.append({"resref": item, "infinite": bool(row.get("Infinite"))})
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "name": _legacy_root_value(root, installation, "LocName"),
            "mark_up": _legacy_root_value(root, installation, "MarkUp"),
            "mark_down": _legacy_root_value(root, installation, "MarkDown"),
            "buy_sell_flag": _legacy_root_value(root, installation, "BuySellFlag"),
            "inventory": inventory,
            "scripts": _legacy_script_fields(root, installation, ("OnOpenStore",)),
        }
    if name == "getSound":
        sounds = []
        for row in _legacy_list_rows(root, installation, "Sounds"):
            sound = row.get("Sound") or row.get("ResRef") or row.get("resref")
            if sound:
                sounds.append(sound)
        return {
            "tag": _legacy_root_value(root, installation, "Tag"),
            "active": _legacy_root_value(root, installation, "Active"),
            "continuous": _legacy_root_value(root, installation, "Continuous"),
            "looping": _legacy_root_value(root, installation, "Looping"),
            "positional": _legacy_root_value(root, installation, "Positional"),
            "volume": _legacy_root_value(root, installation, "Volume"),
            "pitch_variation": _legacy_root_value(root, installation, "PitchVariation"),
            "min_distance": _legacy_root_value(root, installation, "MinDistance"),
            "max_distance": _legacy_root_value(root, installation, "MaxDistance"),
            "x": _legacy_root_value(root, installation, "XPosition"),
            "y": _legacy_root_value(root, installation, "YPosition"),
            "z": _legacy_root_value(root, installation, "ZPosition"),
            "sounds": sounds,
        }
    return {}


def _legacy_generic_blueprint_payload(restype: str, root: Any, installation: Any) -> dict[str, Any]:
    """Add the public getBlueprint convenience fields without losing typed rows."""

    result = _legacy_blueprint_common(root, installation, restype)
    fields_by_type: dict[str, Mapping[str, tuple[str, ...]]] = {
        "utc": {
            "first_name": ("FirstName",), "last_name": ("LastName",), "appearance": ("Appearance_Type",),
            "gender": ("Gender",), "race": ("Race",), "subrace": ("SubraceIndex",), "faction": ("FactionID",),
            "hp": ("HitPoints",), "hp_current": ("CurrentHitPoints",), "hp_max": ("MaxHitPoints",),
            "str": ("Str",), "dex": ("Dex",), "con": ("Con",), "int": ("Int",), "wis": ("Wis",),
            "cha": ("Cha",), "conversation": ("Conversation",),
        },
        "uti": {
            "base_item": ("BaseItem",), "cost": ("Cost",), "add_cost": ("AddCost",), "stolen": ("Stolen",),
            "identified": ("Identified",), "charges": ("Charges",), "stack_size": ("StackSize",),
            "model_part1": ("ModelPart1",), "model_part2": ("ModelPart2",), "model_part3": ("ModelPart3",),
            "texture_var": ("TextureVar",), "upgrade_level": ("UpgradeLevel",),
        },
        "utp": {
            "appearance": ("Appearance",), "conversation": ("Conversation",), "faction": ("Faction",),
            "hp": ("HP",), "hp_current": ("CurrentHP",), "hardness": ("Hardness",), "fort_save": ("Fort",),
            "locked": ("Locked",), "key_required": ("KeyRequired",), "key_name": ("KeyName",),
            "has_inventory": ("HasInventory",), "static": ("Static",),
        },
        "utd": {
            "appearance": ("AppearanceID",), "conversation": ("Conversation",), "faction": ("Faction",),
            "hp": ("HP",), "hp_current": ("CurrentHP",), "hardness": ("Hardness",), "fort_save": ("Fort",),
            "locked": ("Locked",), "key_required": ("KeyRequired",), "key_name": ("KeyName",),
            "generic_type": ("GenericType",), "linked_to": ("LinkedTo",), "linked_to_flags": ("LinkedToFlags",),
        },
        "ute": {
            "active": ("Active",), "difficulty": ("DifficultyIndex",), "faction": ("Faction",),
            "max_creatures": ("MaxCreatures",), "reset_time": ("ResetTime",), "spawn_option": ("SpawnOption",),
        },
        "utm": {
            "mark_up": ("MarkUp",), "mark_down": ("MarkDown",), "black_market": ("BlackMarket",), "id": ("ID",),
        },
        "uts": {
            "active": ("Active",), "looping": ("Looping",), "positional": ("Positional",), "priority": ("Priority",),
            "elevation": ("Elevation",), "max_distance": ("MaxDistance",), "min_distance": ("MinDistance",),
            "random_range": ("RandomRangeX", "RandomRange"),
        },
        "utt": {
            "faction": ("Faction",), "type": ("Type",), "cursor": ("Cursor",),
            "highlight_height": ("HighlightHeight",), "linked_to": ("LinkedTo",), "linked_to_flags": ("LinkedToFlags",),
        },
        "utw": {
            "appearance": ("Appearance",), "linked_to": ("LinkedTo",), "map_note": ("MapNote",),
            "map_note_enabled": ("MapNoteEnabled",), "has_map_note": ("HasMapNote",),
        },
    }
    for key, labels in fields_by_type.get(restype, {}).items():
        value = _legacy_root_value(root, installation, *labels)
        if value is not None:
            result[key] = value
    list_fields = {
        "utc": {"class_list": "ClassList", "equip_item_list": "Equip_ItemList", "item_list": "ItemList"},
        "uti": {"properties": "PropertiesList"},
        "utp": {"item_list": "ItemList"},
        "ute": {"creature_list": "CreatureList"},
        "utm": {"item_list": "ItemList"},
        "uts": {"sound_list": "Sounds"},
    }
    for key, label in list_fields.get(restype, {}).items():
        result[key] = _legacy_list_rows(root, installation, label)
    return {key: value for key, value in result.items() if value is not None or key in {"tag", "name", "description"}}


def _get_blueprint_payload(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.blueprint_authoring import BLUEPRINT_RESOURCE_TYPES, BlueprintGFFDocument

    game_id = resolve_game(str(arguments.get("game", "")))
    if game_id is None:
        raise ValueError("game must be k1 or k2")
    resref = str(arguments.get("resref", "") or "").strip()
    if not resref:
        raise ValueError("resref is required")
    requested_type = _BLUEPRINT_COMPOSITE_TYPES.get(
        name,
        str(arguments.get("restype", arguments.get("type", "")) or "").strip().casefold().lstrip("."),
    )
    if requested_type and requested_type not in BLUEPRINT_RESOURCE_TYPES:
        raise ValueError(f"Unsupported KOTOR blueprint type: {requested_type}")
    installation = load_installation(game_id)
    candidates: list[tuple[str, Any]] = []
    for restype in ((requested_type,) if requested_type else _BLUEPRINT_TYPE_ORDER):
        entry = installation.get_resource(resref, restype)
        if entry is not None:
            candidates.append((restype, entry))
    if not candidates:
        expected = f".{requested_type}" if requested_type else " under any known blueprint type"
        raise FileNotFoundError(f"{resref}{expected} was not found")
    if len(candidates) > 1:
        return {
            "game": str(arguments.get("game", "")).upper(),
            "resref": resref,
            "ambiguous": True,
            "error": "This resref exists under multiple blueprint types; specify restype to select one without guessing.",
            "candidates": [
                {
                    "restype": restype,
                    "byte_count": len(entry.data),
                    "sha256": hashlib.sha256(bytes(entry.data)).hexdigest(),
                    "source": str(getattr(entry, "source", "")),
                }
                for restype, entry in candidates
            ],
        }
    restype, entry = candidates[0]
    payload = bytes(entry.data)
    document = BlueprintGFFDocument.load(payload)
    summary = document.summary()
    all_fields = document.fields()
    limit = min(1000, max(1, int(arguments.get("max_fields", 300) or 300)))
    fields = [row.as_row() for row in all_fields[:limit]]
    identity_names = {
        "tag", "templateresref", "comment", "appearance_type", "baseitem", "conversation",
        "paletteid", "portraitid", "factionid", "plot", "static", "useable",
    }
    identity = {
        row.label: row.edit_text
        for row in all_fields
        if row.parent_path == "$" and row.editable and row.label.casefold() in identity_names
    }
    resource_references = [
        {"path": row.path, "label": row.label, "resref": row.edit_text}
        for row in all_fields
        if row.field_type == "ResRef" and row.edit_text.strip() and row.edit_text.strip() != "****"
    ][:200]
    script_references = [
        row
        for row in resource_references
        if "script" in row["label"].casefold() or row["label"].casefold().startswith("on")
    ]
    conversation_references = [
        row for row in resource_references if row["label"].casefold() in {"conversation", "dialog", "dialogue"}
    ]
    localized_fields = []
    for row in all_fields:
        if row.field_type != "LocalizedString" or len(localized_fields) >= 100:
            continue
        try:
            localized_fields.append(
                {"path": row.path, "label": row.label, **_localized_field_payload(document.value(row.path), installation)}
            )
        except Exception as exc:
            localized_fields.append({"path": row.path, "label": row.label, "error": str(exc)})
    list_counts = [
        {"path": row.path, "label": row.label, "struct_count": row.child_count}
        for row in all_fields
        if row.field_type == "List"
    ][:100]
    response = {
        "game": str(arguments.get("game", "")).upper(),
        "resref": resref,
        "restype": restype,
        "ambiguous": False,
        "source": str(getattr(entry, "source", "")),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "embedded_content_type": summary.content_type,
        "embedded_resource_type": summary.resource_type,
        "content_type_matches_extension": summary.resource_type == restype,
        "root_struct_id": summary.root_struct_id,
        "field_count": summary.field_count,
        "editable_field_count": summary.editable_field_count,
        "field_limit": limit,
        "fields_truncated": len(all_fields) > limit,
        "fields": fields,
        "identity": identity,
        "localized_fields": localized_fields,
        "resource_references": resource_references,
        "script_references": script_references,
        "conversation_references": conversation_references,
        "list_counts": list_counts,
        "preservation": "The complete parsed GFF graph remains intact; only this MCP response is field-limited.",
    }
    from pykotor.resource.formats.gff import read_gff

    root = read_gff(payload).root
    response["legacy_fields"] = _legacy_gff_value(root, installation, budget=[10_000])
    response.update(_legacy_generic_blueprint_payload(restype, root, installation))
    if name in _BLUEPRINT_COMPOSITE_TYPES:
        response.update(_legacy_named_blueprint_payload(name, restype, payload, root, installation))
    return response


_COMPAT_MAX_BINARY_BYTES = 64 * 1024 * 1024
_COMPAT_RESREF = re.compile(r"^[a-z0-9_]{1,16}$")


def _compat_game(value: object) -> str:
    game = str(value or "K2").strip().upper()
    if game in {"K1", "1", "KOTOR", "KOTOR1"}:
        return "K1"
    if game in {"K2", "2", "TSL", "KOTOR2"}:
        return "K2"
    raise ValueError("game must be K1 or K2")


def _compat_resref(value: object, *, label: str = "resref") -> str:
    text = str(value or "").strip().lower()
    if not _COMPAT_RESREF.fullmatch(text):
        raise ValueError(f"{label} must use 1-16 lowercase letters, numbers, or underscores")
    return text


def _strict_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError as exc:
            raise ValueError(f"{label} must be a base-10 integer") from exc
    raise ValueError(f"{label} must be an integer")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _decode_compat_base64(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required and must be base64 text")
    if len(value) > ((_COMPAT_MAX_BINARY_BYTES + 2) // 3) * 4 + 4:
        raise ValueError(f"{label} exceeds the 64 MiB compatibility safety limit")
    try:
        payload = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid base64") from exc
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError(f"{label} exceeds the 64 MiB compatibility safety limit")
    return payload


def _typed_gff_struct(document: object, *, path: str, depth: int, state: list[int]):
    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import ResRef
    from pykotor.resource.formats.gff import GFFFieldType, GFFList, GFFStruct
    from utility.common.geometry import Vector3, Vector4

    if depth > 64:
        raise ValueError("typed GFF exceeds the 64-level nesting safety limit")
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must be a struct object")
    if document.get("truncated"):
        raise ValueError(f"{path} is truncated and cannot be written")
    struct_id = _strict_integer(document.get("struct_id"), f"{path}.struct_id")
    if not -1 <= struct_id <= 0xFFFFFFFF:
        raise ValueError(f"{path}.struct_id is outside the uint32/-1 range")
    fields = document.get("fields")
    if not isinstance(fields, list):
        raise ValueError(f"{path}.fields must be an ordered array")
    struct = GFFStruct(struct_id)
    seen: set[str] = set()
    integer_ranges = {
        GFFFieldType.UInt8: (0, 0xFF),
        GFFFieldType.Int8: (-0x80, 0x7F),
        GFFFieldType.UInt16: (0, 0xFFFF),
        GFFFieldType.Int16: (-0x8000, 0x7FFF),
        GFFFieldType.UInt32: (0, 0xFFFFFFFF),
        GFFFieldType.Int32: (-0x80000000, 0x7FFFFFFF),
        GFFFieldType.UInt64: (0, 0xFFFFFFFFFFFFFFFF),
        GFFFieldType.Int64: (-0x8000000000000000, 0x7FFFFFFFFFFFFFFF),
    }
    integer_setters = {
        GFFFieldType.UInt8: struct.set_uint8,
        GFFFieldType.Int8: struct.set_int8,
        GFFFieldType.UInt16: struct.set_uint16,
        GFFFieldType.Int16: struct.set_int16,
        GFFFieldType.UInt32: struct.set_uint32,
        GFFFieldType.Int32: struct.set_int32,
        GFFFieldType.UInt64: struct.set_uint64,
        GFFFieldType.Int64: struct.set_int64,
    }
    for index, raw_field in enumerate(fields):
        state[0] += 1
        if state[0] > 100_000:
            raise ValueError("typed GFF exceeds the 100,000-field safety limit")
        field_path = f"{path}.fields[{index}]"
        if not isinstance(raw_field, Mapping):
            raise ValueError(f"{field_path} must be an object")
        label = raw_field.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{field_path}.label must be a nonempty string")
        try:
            encoded_label = label.encode("ascii", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{field_path}.label must be ASCII") from exc
        if len(encoded_label) > 16:
            raise ValueError(f"{field_path}.label exceeds GFF's 16-byte limit")
        if label in seen:
            raise ValueError(f"{path} contains duplicate field label {label!r}")
        seen.add(label)
        type_name = raw_field.get("type")
        if not isinstance(type_name, str) or type_name not in GFFFieldType.__members__:
            raise ValueError(f"{field_path}.type is not a supported GFF type")
        field_type = GFFFieldType[type_name]
        if "type_id" in raw_field and _strict_integer(raw_field["type_id"], f"{field_path}.type_id") != int(field_type):
            raise ValueError(f"{field_path}.type_id conflicts with {type_name}")
        value = raw_field.get("value")
        if field_type in integer_ranges:
            parsed = _strict_integer(value, f"{field_path}.value")
            minimum, maximum = integer_ranges[field_type]
            if not minimum <= parsed <= maximum:
                raise ValueError(f"{field_path}.value is outside {type_name}'s range")
            integer_setters[field_type](label, parsed)
        elif field_type in {GFFFieldType.Single, GFFFieldType.Double}:
            parsed = _finite_number(value, f"{field_path}.value")
            (struct.set_single if field_type == GFFFieldType.Single else struct.set_double)(label, parsed)
        elif field_type == GFFFieldType.String:
            if not isinstance(value, str):
                raise ValueError(f"{field_path}.value must be a string")
            struct.set_string(label, value)
        elif field_type == GFFFieldType.ResRef:
            if not isinstance(value, str) or len(value) > 16:
                raise ValueError(f"{field_path}.value must be a string no longer than 16 characters")
            struct.set_resref(label, ResRef(value))
        elif field_type == GFFFieldType.LocalizedString:
            if not isinstance(value, Mapping) or not isinstance(value.get("substrings"), list):
                raise ValueError(f"{field_path}.value must contain stringref and a substrings array")
            stringref = _strict_integer(value.get("stringref"), f"{field_path}.value.stringref")
            if not -1 <= stringref <= 0xFFFFFFFE:
                raise ValueError(f"{field_path}.value.stringref is outside the KOTOR StrRef range")
            substrings: dict[int, str] = {}
            for sub_index, raw_substring in enumerate(value["substrings"]):
                sub_path = f"{field_path}.value.substrings[{sub_index}]"
                if not isinstance(raw_substring, Mapping) or not isinstance(raw_substring.get("text"), str):
                    raise ValueError(f"{sub_path} must contain an integer id and string text")
                substring_id = _strict_integer(raw_substring.get("id"), f"{sub_path}.id")
                if not 0 <= substring_id <= 0xFFFFFFFF or substring_id in substrings:
                    raise ValueError(f"{sub_path}.id is duplicated or outside the uint32 range")
                substrings[substring_id] = raw_substring["text"]
            struct.set_locstring(label, LocalizedString(stringref, substrings))
        elif field_type == GFFFieldType.Binary:
            if not isinstance(value, Mapping) or value.get("encoding") != "base64":
                raise ValueError(f"{field_path}.value must use base64 encoding")
            struct.set_binary(label, _decode_compat_base64(value.get("data"), f"{field_path}.value.data"))
        elif field_type == GFFFieldType.Struct:
            struct.set_struct(label, _typed_gff_struct(value, path=f"{field_path}.value", depth=depth + 1, state=state))
        elif field_type == GFFFieldType.List:
            if not isinstance(value, list):
                raise ValueError(f"{field_path}.value must be a struct array")
            children = GFFList()
            for child_index, child in enumerate(value):
                children.append(_typed_gff_struct(child, path=f"{field_path}.value[{child_index}]", depth=depth + 1, state=state))
            struct.set_list(label, children)
        elif field_type in {GFFFieldType.Vector3, GFFFieldType.Vector4}:
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_path}.value must be a vector object")
            x = _finite_number(value.get("x"), f"{field_path}.value.x")
            y = _finite_number(value.get("y"), f"{field_path}.value.y")
            z = _finite_number(value.get("z"), f"{field_path}.value.z")
            if field_type == GFFFieldType.Vector3:
                struct.set_vector3(label, Vector3(x, y, z))
            else:
                struct.set_vector4(label, Vector4(x, y, z, _finite_number(value.get("w"), f"{field_path}.value.w")))
    return struct


def _lossy_gff_struct(fields: Mapping[str, Any], *, depth: int = 0, state: list[int] | None = None):
    from pykotor.resource.formats.gff import GFFList, GFFStruct

    counter = state or [0]
    if depth > 64:
        raise ValueError("legacy GFF fields exceed the 64-level nesting safety limit")
    struct = GFFStruct(0 if depth else -1)
    for label, value in fields.items():
        counter[0] += 1
        if counter[0] > 100_000:
            raise ValueError("legacy GFF fields exceed the 100,000-field safety limit")
        if not isinstance(label, str) or not label or len(label.encode("ascii", errors="strict")) > 16:
            raise ValueError("legacy GFF labels must be nonempty ASCII strings of at most 16 bytes")
        if isinstance(value, bool):
            struct.set_uint8(label, int(value))
        elif isinstance(value, str):
            struct.set_string(label, value)
        elif isinstance(value, int):
            if not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"legacy integer field {label!r} is outside the UInt32 range")
            struct.set_uint32(label, value)
        elif isinstance(value, float):
            struct.set_single(label, _finite_number(value, label))
        elif isinstance(value, Mapping):
            struct.set_struct(label, _lossy_gff_struct(value, depth=depth + 1, state=counter))
        elif isinstance(value, list):
            children = GFFList()
            for index, child in enumerate(value):
                if not isinstance(child, Mapping):
                    raise ValueError(f"legacy GFF list {label!r}[{index}] must be a struct object")
                children.append(_lossy_gff_struct(child, depth=depth + 1, state=counter))
            struct.set_list(label, children)
        else:
            raise ValueError(f"legacy GFF field {label!r} has unsupported inferred type {type(value).__name__}")
    return struct


def _write_gff_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.formats.gff import GFF, GFFContent
    from src.core.scripting.blueprint_authoring import BlueprintGFFDocument

    document = arguments.get("document")
    has_fields = "fields" in arguments
    if document is not None and has_fields:
        raise ValueError("provide either document or legacy fields, not both")
    if document is not None:
        if not isinstance(document, Mapping):
            raise ValueError("document must be a typed GFF object")
        if document.get("schema") != "ghostscripter.gff.typed.v1" or document.get("complete") is not True:
            raise ValueError("document must be a complete ghostscripter.gff.typed.v1 object")
        if document.get("file_version") != "V3.2":
            raise ValueError("KOTOR typed GFF documents must use file_version V3.2")
        file_type = document.get("file_type")
        if not isinstance(file_type, str) or len(file_type) != 4:
            raise ValueError("document.file_type must be an exact four-character content tag")
        requested = arguments.get("fileType", arguments.get("file_type"))
        if requested is not None and str(requested).ljust(4)[:4] != file_type:
            raise ValueError("fileType conflicts with document.file_type")
        content = GFFContent(file_type)
        if document.get("content") is not None and document.get("content") != content.name:
            raise ValueError("document.content conflicts with document.file_type")
        root = _typed_gff_struct(document.get("root"), path="root", depth=0, state=[0])
        fidelity = "lossless_typed"
        extra = {"schema": document.get("schema"), "content": content.name}
    else:
        raw_type = arguments.get("fileType", arguments.get("file_type", ""))
        file_type = str(raw_type).ljust(4)[:4]
        if not re.fullmatch(r"[A-Za-z0-9_ ]{4}", file_type):
            raise ValueError("fileType must be a recognized four-character ASCII GFF tag")
        content = GFFContent(file_type)
        fields = arguments.get("fields")
        if not isinstance(fields, Mapping):
            raise ValueError("legacy fields must be an object")
        if arguments.get("allowLossy", arguments.get("allow_lossy", False)) is not True:
            raise ValueError("ambiguous legacy fields require literal allowLossy=true")
        root = _lossy_gff_struct(fields)
        fidelity = "lossy_legacy"
        extra = {"warning": "Legacy fields used inferred String/UInt32/Single/Struct/List types."}
    gff = GFF(content)
    gff.root = root
    payload = BlueprintGFFDocument.from_gff(gff).to_bytes()
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("serialized GFF exceeds the 64 MiB compatibility safety limit")
    return {
        **extra,
        "file_type": file_type,
        "fidelity": fidelity,
        "size_bytes": len(payload),
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def _dialogue_node_arrays(dialogue: object) -> tuple[list[object], list[object]]:
    from pykotor.resource.generics.dlg import DLGEntry, DLGReply

    entries: dict[int, object] = {}
    replies: dict[int, object] = {}
    pending = list(tuple(getattr(dialogue, "starters", ()) or ()))
    seen_links: set[int] = set()
    seen_nodes: set[int] = set()
    while pending:
        link = pending.pop(0)
        if id(link) in seen_links:
            continue
        seen_links.add(id(link))
        node = getattr(link, "node", None)
        if node is None:
            continue
        if id(node) not in seen_nodes:
            seen_nodes.add(id(node))
            target = entries if isinstance(node, DLGEntry) else replies if isinstance(node, DLGReply) else None
            if target is None:
                raise ValueError(f"unsupported DLG node type: {node.__class__.__name__}")
            index = int(getattr(node, "list_index", -1))
            if index < 0 or index in target:
                index = max(target, default=-1) + 1
            target[index] = node
        pending.extend(tuple(getattr(node, "links", ()) or ()))
    return [entries[index] for index in sorted(entries)], [replies[index] for index in sorted(replies)]


def _apply_legacy_dialogue_node(node: object, raw: Mapping[str, Any]) -> None:
    from pykotor.common.language import Gender, Language, LocalizedString
    from src.core.scripting.dialogue_contract import apply_dialogue_node_fields

    if "strref" in raw or "text" in raw:
        stringref = _strict_integer(raw.get("strref", -1), "dialogue node strref")
        if not -1 <= stringref <= 0xFFFFFFFE:
            raise ValueError("dialogue node strref is outside the KOTOR StrRef range")
        localized = LocalizedString(stringref)
        if "text" in raw:
            localized.set_data(Language.ENGLISH, Gender.MALE, str(raw.get("text") or ""))
        node.text = localized
    mapping = {
        "speaker": "speaker",
        "listener": "listener",
        "script1": "script1",
        "script2": "script2",
        "vo_resref": "vo_resref",
        "sound": "sound",
    }
    changes = {target: raw[source] for source, target in mapping.items() if source in raw}
    if changes:
        apply_dialogue_node_fields(node, changes)


def _legacy_dialogue_links(raw_rows: object, targets: Sequence[object], *, label: str) -> list[object]:
    from pykotor.resource.generics.dlg import DLGLink
    from src.core.scripting.dialogue_contract import apply_dialogue_link_fields

    if not isinstance(raw_rows, list):
        raise ValueError(f"{label} must be an array")
    links: list[object] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        target_index = _strict_integer(raw.get("index", -1), f"{label}[{index}].index")
        if not 0 <= target_index < len(targets):
            raise ValueError(f"{label}[{index}].index is outside the target node array")
        link = DLGLink(targets[target_index], index)
        changes: dict[str, Any] = {}
        if "is_child" in raw:
            changes["is_child"] = bool(raw["is_child"])
        elif "is_reply" in raw:
            changes["is_child"] = bool(raw["is_reply"])
        for source, target in {
            "active_script": "active1",
            "active_script2": "active2",
            "link_comment": "comment",
            "display_inactive": "display_inactive",
        }.items():
            if source in raw:
                changes[target] = raw[source]
        if changes:
            apply_dialogue_link_fields(link, changes)
        links.append(link)
    return links


def _patch_dialogue_display_inactive(payload: bytes, dialogue_dto: Mapping[str, Any], game: str) -> bytes:
    """Patch the K2 link field PyKotor's current DLG model does not emit."""

    if game != "K2":
        return payload
    from pykotor.resource.formats.gff import read_gff
    from src.core.scripting.blueprint_authoring import BlueprintGFFDocument

    gff = read_gff(payload)
    touched = False

    def patch_rows(struct_rows: object, dto_rows: object) -> None:
        nonlocal touched
        if not isinstance(dto_rows, list):
            return
        for index, raw in enumerate(dto_rows):
            if index >= len(struct_rows) or not isinstance(raw, Mapping) or "display_inactive" not in raw:
                continue
            struct_rows[index].set_uint8("DisplayInactive", int(bool(raw["display_inactive"])))
            touched = True

    if "starters" in dialogue_dto:
        patch_rows(gff.root.get_list("StartingList"), dialogue_dto.get("starters"))
    entries = dialogue_dto.get("entries")
    if isinstance(entries, list):
        entry_structs = gff.root.get_list("EntryList")
        for index, raw in enumerate(entries):
            if index < len(entry_structs) and isinstance(raw, Mapping) and "branches" in raw:
                patch_rows(entry_structs[index].get_list("RepliesList"), raw.get("branches"))
    replies = dialogue_dto.get("replies")
    if isinstance(replies, list):
        reply_structs = gff.root.get_list("ReplyList")
        for index, raw in enumerate(replies):
            if index < len(reply_structs) and isinstance(raw, Mapping) and "branches" in raw:
                patch_rows(reply_structs[index].get_list("EntriesList"), raw.get("branches"))
    return BlueprintGFFDocument.from_gff(gff).to_bytes() if touched else payload


def _write_dialogue_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.generics.dlg import DLG, DLGEntry, DLGReply
    from src.core.scripting.dialogue_contract import apply_dialogue_settings
    from src.core.scripting.studio import DialogueDocument, ScriptingStudioService, dialogue_structure_summary

    game = _compat_game(arguments.get("game", "K1"))
    resref = _compat_resref(arguments.get("resref", "dialogue"))
    dto = arguments.get("dialogue")
    if not isinstance(dto, Mapping):
        raise ValueError("dialogue must be a JSON object")
    service = ScriptingStudioService()
    fidelity = dto.get("source_fidelity") if "source_fidelity" in dto else None
    if fidelity is not None:
        if not isinstance(fidelity, Mapping) or fidelity.get("schema") != "ghostscripter.dlg.source-gff.v1":
            raise ValueError("source_fidelity uses an unsupported schema")
        source = _decode_compat_base64(fidelity.get("binary_base64"), "source_fidelity.binary_base64")
        if not source.startswith(b"DLG V3.2"):
            raise ValueError("source_fidelity payload is not a KOTOR DLG V3.2 resource")
        document = service.dialogue_from_bytes(source, game=game, resref=resref, origin="legacy_ghostscripter")
        dialogue = document.dialogue
        existing_entries, existing_replies = _dialogue_node_arrays(dialogue)
    else:
        document = DialogueDocument(resref=resref, game=game, dialogue=DLG(), origin="legacy_ghostscripter")
        dialogue = document.dialogue
        existing_entries, existing_replies = [], []

    root_changes: dict[str, Any] = {}
    for source, target in {
        "end_script": "on_end",
        "abort_script": "on_abort",
        "skippable": "skippable",
        "delay_entry": "delay_entry",
        "delay_reply": "delay_reply",
        "ambient_track": "ambient_track",
        "animated_cut": "animated_cut",
        "camera_model": "camera_model",
        "conversation_type": "conversation_type",
        "computer_type": "computer_type",
        "old_hit_check": "old_hit_check",
        "unequip_items": "unequip_items",
        "unequip_h_item": "unequip_hands",
        "word_count": "word_count",
    }.items():
        if source in dto:
            root_changes[target] = dto[source]
    if root_changes:
        apply_dialogue_settings(dialogue, root_changes)

    def build_nodes(key: str, node_class: type, existing: list[object]) -> list[object]:
        if key not in dto:
            return existing if fidelity is not None else []
        raw_nodes = dto.get(key)
        if not isinstance(raw_nodes, list):
            raise ValueError(f"dialogue.{key} must be an array")
        nodes: list[object] = []
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping):
                raise ValueError(f"dialogue.{key}[{index}] must be an object")
            node = existing[index] if index < len(existing) else node_class()
            node.list_index = index
            _apply_legacy_dialogue_node(node, raw)
            nodes.append(node)
        return nodes

    entries = build_nodes("entries", DLGEntry, existing_entries)
    replies = build_nodes("replies", DLGReply, existing_replies)
    raw_entries = dto.get("entries") if isinstance(dto.get("entries"), list) else None
    raw_replies = dto.get("replies") if isinstance(dto.get("replies"), list) else None
    if raw_entries is not None:
        for index, raw in enumerate(raw_entries):
            if "branches" in raw:
                entries[index].links = _legacy_dialogue_links(raw["branches"], replies, label=f"entries[{index}].branches")
    if raw_replies is not None:
        for index, raw in enumerate(raw_replies):
            if "branches" in raw:
                replies[index].links = _legacy_dialogue_links(raw["branches"], entries, label=f"replies[{index}].branches")
    if "starters" in dto:
        dialogue.starters = _legacy_dialogue_links(dto.get("starters"), entries, label="starters")
    elif fidelity is None:
        dialogue.starters = []

    payload, diagnostics = service.dialogue_bytes(document)
    blocking = [row.message for row in diagnostics if row.blocking]
    if not payload or blocking:
        raise ValueError("Cannot serialize invalid DLG: " + "; ".join(blocking or ["no payload was produced"]))
    payload = _patch_dialogue_display_inactive(payload, dto, game)
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("serialized DLG exceeds the 64 MiB compatibility safety limit")
    readback = service.dialogue_from_bytes(payload, game=game, resref=resref, origin="compatibility_readback")
    summary = dialogue_structure_summary(readback.dialogue)
    expected = dialogue_structure_summary(dialogue)
    if summary != expected:
        raise ValueError(f"DLG readback changed graph structure: {expected} -> {summary}")
    return {
        "game": game,
        "size_bytes": len(payload),
        "entry_count": summary["entries"],
        "reply_count": summary["replies"],
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def _twoda_text_token(value: str) -> str:
    text = str(value)
    if not text:
        return "****"
    if any(character in text for character in "\r\n\t\0"):
        raise ValueError("2DA text cells cannot contain tabs, newlines, or NUL characters")
    if '"' in text:
        raise ValueError("2DA text cells containing quote characters are not safely representable")
    return f'"{text}"' if any(character.isspace() for character in text) else text


def _twoda_text_bytes(document: object) -> bytes:
    headers = tuple(getattr(document, "headers"))
    labels = tuple(getattr(document, "labels"))
    lines = ["2DA V2.0", "", "\t" + "\t".join(_twoda_text_token(header) for header in headers)]
    for index, label in enumerate(labels):
        row = document.row(index)
        lines.append(_twoda_text_token(label) + "\t" + "\t".join(_twoda_text_token(row[header]) for header in headers))
    return ("\n".join(lines) + "\n").encode("latin-1", errors="strict")


def _parse_twoda_text(payload: bytes) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    text = payload.decode("latin-1")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "2DA V2.0" or lines[1] != "":
        raise ValueError("2DA V2.0 readback header is invalid")

    def decode_token(token: str) -> str:
        value = token.strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value

    headers = tuple(decode_token(value) for value in lines[2].lstrip("\t").split("\t"))
    labels: list[str] = []
    rows: list[tuple[str, ...]] = []
    for line_number, line in enumerate(lines[3:], start=4):
        if not line.strip():
            continue
        tokens = line.split("\t")
        if len(tokens) != len(headers) + 1:
            raise ValueError(f"2DA V2.0 readback row {line_number} has the wrong cell count")
        labels.append(decode_token(tokens[0]))
        rows.append(tuple(decode_token(value) for value in tokens[1:]))
    return headers, tuple(labels), tuple(rows)


def _write_twoda_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.data_authoring import TwoDADocument

    resref = _compat_resref(arguments.get("resref"))
    columns = arguments.get("columns")
    rows = arguments.get("rows")
    if not isinstance(columns, list) or not columns or len(columns) > 1_000:
        raise ValueError("columns must be a nonempty array with at most 1,000 entries")
    if not all(isinstance(column, str) and column.strip() and not any(character.isspace() for character in column) for column in columns):
        raise ValueError("every 2DA column must be a nonempty string")
    if len(set(columns)) != len(columns):
        raise ValueError("2DA column names must be unique")
    if not isinstance(rows, list) or len(rows) > 100_000:
        raise ValueError("rows must be an array with at most 100,000 entries")
    if len(rows) * len(columns) > 2_000_000:
        raise ValueError("2DA input exceeds the 2,000,000-cell compatibility safety limit")
    labels: list[str] = []
    values: list[list[str]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or "label" not in raw:
            raise ValueError(f"rows[{index}] must be an object containing label")
        unknown = set(raw).difference({"label", *columns})
        if unknown:
            raise ValueError(f"rows[{index}] contains unknown column(s): {', '.join(sorted(map(str, unknown)))}")
        row_label = str(raw["label"])
        if not row_label:
            raise ValueError(f"rows[{index}].label cannot be empty")
        labels.append(row_label)
        values.append([
            "****" if raw.get(column) in {None, ""} else str(raw.get(column, "****"))
            for column in columns
        ])
    document = TwoDADocument(columns, labels, values)
    warnings: list[str] = []
    edits = arguments.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("edits must be an array")
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping) or not {"row", "column", "value"} <= set(edit):
            raise ValueError(f"edits[{index}] must contain row, column, and value")
        column = str(edit["column"])
        if column not in document.headers:
            warnings.append(f"Column '{column}' was not found; edit skipped.")
            continue
        row_key = edit["row"]
        if isinstance(row_key, bool):
            raise ValueError(f"edits[{index}].row cannot be boolean")
        if isinstance(row_key, int):
            row_index = row_key
            if not 0 <= row_index < document.row_count:
                warnings.append(f"Row index {row_index} out of range; edit skipped.")
                continue
        else:
            try:
                row_index = document.labels.index(str(row_key))
            except ValueError:
                warnings.append(f"Row label '{row_key}' not found; edit skipped.")
                continue
        document.set_cell(row_index, column, "****" if edit["value"] in {None, ""} else edit["value"])
    output_format = str(arguments.get("format", "text") or "text").strip().lower()
    if output_format == "binary":
        payload = document.to_bytes()
        readback = TwoDADocument.load(payload).snapshot()
        if readback != document.snapshot():
            raise ValueError("binary 2DA semantic readback did not match the authored table")
    elif output_format == "text":
        payload = _twoda_text_bytes(document)
        readback = _parse_twoda_text(payload)
        expected = (document.headers, document.labels, document.snapshot().rows)
        if readback != expected:
            raise ValueError("text 2DA semantic readback did not match the authored table")
    else:
        raise ValueError("format must be text or binary")
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("serialized 2DA exceeds the 64 MiB compatibility safety limit")
    response: dict[str, Any] = {
        "resref": resref,
        "format": output_format,
        "row_count": document.row_count,
        "column_count": document.column_count,
        "size_bytes": len(payload),
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }
    if warnings:
        response["warnings"] = warnings
    return response


def _write_erf_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.packaging import NarrativePackagingService, PackageResource, inspect_narrative_archive

    raw_files = arguments.get("files")
    if not isinstance(raw_files, list) or not raw_files or len(raw_files) > 50_000:
        raise ValueError("files must be a nonempty array with at most 50,000 resources")
    resources: list[PackageResource] = []
    total_bytes = 0
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise ValueError(f"files[{index}] must be an object")
        data = _decode_compat_base64(raw.get("data_b64"), f"files[{index}].data_b64")
        total_bytes += len(data)
        if total_bytes > _COMPAT_MAX_BINARY_BYTES:
            raise ValueError("archive inputs exceed the 64 MiB compatibility safety limit")
        resources.append(PackageResource(str(raw.get("resref", "")), str(raw.get("type", "")), data))
    raw_type = str(arguments.get("archive_type", "MOD ") or "MOD ").upper().ljust(4)[:4]
    if raw_type not in {"ERF ", "MOD ", "SAV "}:
        raise ValueError("archive_type must be ERF , MOD , or SAV ")
    archive_type = raw_type.strip()
    with tempfile.TemporaryDirectory(prefix="ghoststudio-legacy-erf-") as temporary:
        output = Path(temporary) / f"compat.{archive_type.lower()}"
        result = NarrativePackagingService.build_archive(resources, output, archive_type=archive_type)
        blocking = [row.message for row in result.issues if row.blocking]
        if not result.ok or blocking:
            raise ValueError("archive build failed: " + "; ".join(blocking or ["output was not committed"]))
        payload = output.read_bytes()
        if len(payload) > _COMPAT_MAX_BINARY_BYTES:
            raise ValueError("serialized archive exceeds the 64 MiB compatibility safety limit")
        inspection = inspect_narrative_archive(output)
        actual = {(row.resref, row.restype): row.data for row in inspection.resources}
        expected = {(row.resref, row.restype): row.data for row in resources}
        if actual != expected:
            raise ValueError("archive exact resource readback failed")
    return {
        "archive_type": archive_type,
        "file_count": len(inspection.resources),
        "size_bytes": len(payload),
        "files": [row.filename for row in inspection.resources],
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def _validated_existing_root(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    requested = Path(text)
    if not requested.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    current = requested
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} cannot traverse a symbolic link")
        current = current.parent
    root = requested.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be an existing safe directory")
    if root == Path(root.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return root


def _separate_trees(first: Path, second: Path) -> bool:
    return not (first == second or first in second.parents or second in first.parents)


def _validated_game_root(arguments: Mapping[str, Any], game: str) -> Path:
    game_root = _validated_existing_root(arguments.get("game_root"), label="game_root")
    executable = game_root / ("swkotor.exe" if game == "K1" else "swkotor2.exe")
    chitin = game_root / "chitin.key"
    if not executable.is_file() or executable.is_symlink() or not chitin.is_file() or chitin.is_symlink():
        raise ValueError(f"game_root does not contain the expected safe {game} executable and chitin.key")
    return game_root


def _validated_backup_root(value: object, game_root: Path) -> Path:
    backup_root = _validated_existing_root(value, label="backup_root")
    if not _separate_trees(backup_root, game_root):
        raise ValueError("backup_root and game_root must be separate directory trees")
    return backup_root


def _expected_state(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if text != "absent" and not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be 'absent' or a SHA-256 hex digest")
    return text


def _install_override_resource(
    arguments: Mapping[str, Any],
    *,
    resource: object,
    backup_root: object = None,
    expect_existing: object = None,
) -> dict[str, Any]:
    from src.core.scripting.packaging import NarrativePackagingService

    if arguments.get("confirm_install") is not True:
        raise ValueError("confirm_install must be literal true before GhostStudio may change a game installation")
    game = _compat_game(arguments.get("game", "K1"))
    workspace_root = _validated_existing_root(arguments.get("workspace_root"), label="workspace_root")
    game_root = _validated_existing_root(arguments.get("game_root"), label="game_root")
    if not _separate_trees(workspace_root, game_root):
        raise ValueError("workspace_root and game_root must be separate directory trees")
    game_root = _validated_game_root(arguments, game)
    external = None if backup_root in (None, "") else _validated_backup_root(backup_root, game_root)
    expected = None if expect_existing in (None, "") else _expected_state(expect_existing, "expect_existing")
    policy = str(arguments.get("on_conflict", "block") or "block").strip().lower()
    if policy not in {"block", "backup"}:
        raise ValueError("on_conflict must be block or backup")
    stage_path = (
        workspace_root
        / ".ghoststudio"
        / "automation-override-stages"
        / game.lower()
        / f"{resource.resref}_{resource.restype}"
    )
    stage = NarrativePackagingService.stage_override(
        (resource,),
        stage_path,
        game=game,
        replace_owned=True,
    )
    blocking = [row.message for row in stage.issues if row.blocking]
    if not stage.ok or blocking:
        raise ValueError("Override staging failed: " + "; ".join(blocking or ["stage was not committed"]))
    installed = NarrativePackagingService.install_override(
        stage.stage_path,
        game_root,
        on_conflict=policy,
        backup_root=external,
        expected_existing=None if expected is None else {resource.filename: expected},
    )
    blocking = [row.message for row in installed.issues if row.blocking]
    if not installed.ok or blocking:
        raise ValueError("Override install failed: " + "; ".join(blocking or ["install was not committed"]))
    destination = game_root / "Override" / resource.filename
    if not destination.is_file() or destination.is_symlink() or destination.read_bytes() != resource.data:
        raise ValueError("Override destination failed exact post-install readback")
    return {
        "game": game,
        "resref": resource.resref,
        "restype": resource.restype,
        "path": str(destination),
        "stage_path": stage.stage_path,
        "stage_manifest": stage.manifest_path,
        "backup_path": installed.backup_path,
        "receipt_path": installed.receipt_path,
        "backup_location": "game_root" if external is None else "external",
        "expected_existing": expected,
        "installed": list(installed.installed),
        "skipped_identical": list(installed.skipped_identical),
        "conflict_policy": policy,
        "size_bytes": len(resource.data),
        "sha256": resource.sha256,
        "written": True,
        "engine_proof": "not_recorded",
    }


def _write_override_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.packaging import PackageResource

    game = _compat_game(arguments.get("game", "K1"))
    data = _decode_compat_base64(arguments.get("data_b64"), "data_b64")
    resource = PackageResource(
        _compat_resref(arguments.get("resref")),
        str(arguments.get("restype", "")),
        data,
    )
    return _install_override_resource(
        {**dict(arguments), "game": game},
        resource=resource,
        backup_root=arguments.get("backup_root"),
        expect_existing=arguments.get("expect_existing"),
    )


def _restore_override_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Compare-and-swap undo of one writeOverride install (prior bytes or removal)."""

    from src.core.scripting.packaging import NarrativePackagingService, PackageResource

    if arguments.get("confirm_restore") is not True:
        raise ValueError("confirm_restore must be literal true before GhostStudio may change a game installation")
    game = _compat_game(arguments.get("game", "K1"))
    game_root = _validated_game_root(arguments, game)
    backup_root = _validated_backup_root(arguments.get("backup_root"), game_root)
    expected = _expected_state(arguments.get("expect_current"), "expect_current")
    if expected == "absent":
        raise ValueError("expect_current must be the installed SHA-256")
    has_prior = "prior_data_b64" in arguments and arguments.get("prior_data_b64") is not None
    remove = arguments.get("remove_file") is True
    if has_prior == remove:
        raise ValueError("Provide exactly one of prior_data_b64 or remove_file=true")
    prior = None
    if has_prior:
        raw = arguments.get("prior_data_b64")
        # A zero-byte prior file is legal; _decode_compat_base64 requires text.
        prior = b"" if raw == "" else _decode_compat_base64(raw, "prior_data_b64")
    resource = PackageResource(_compat_resref(arguments.get("resref")), str(arguments.get("restype", "")), b"")
    result = NarrativePackagingService.restore_override_file(
        game_root,
        resource.resref,
        resource.restype,
        expected_current_sha256=expected,
        backup_root=backup_root,
        prior_data=prior,
    )
    blocking = [row.message for row in result.issues if row.blocking]
    if not result.ok or blocking:
        raise ValueError("Override restore failed: " + "; ".join(blocking or ["restore was not committed"]))
    destination = game_root / "Override" / resource.filename
    if prior is None:
        if destination.exists():
            raise ValueError("Override file is still present after removal")
    elif not destination.is_file() or destination.is_symlink() or destination.read_bytes() != prior:
        raise ValueError("Override destination failed exact post-restore readback")
    return {
        "game": game,
        "resref": resource.resref,
        "restype": resource.restype,
        "path": str(destination),
        "action": result.action,
        "pre_restore_sha256": result.pre_restore_sha256,
        "restored_sha256": result.restored_sha256,
        "backup_path": result.backup_path,
        "receipt_path": result.receipt_path,
        "written": True,
        "engine_proof": "not_recorded",
    }


def _write_lip_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.data_authoring import LipDocument, LipKeyframeRecord

    duration = _finite_number(arguments.get("duration"), "duration")
    if duration < 0:
        raise ValueError("duration cannot be negative")
    raw_frames = arguments.get("keyframes")
    if not isinstance(raw_frames, list) or len(raw_frames) > 1_000_000:
        raise ValueError("keyframes must be an array with at most 1,000,000 entries")
    shape_names = {name: index for index, name in enumerate(LipDocument.shape_names())}
    frames: list[LipKeyframeRecord] = []
    previous = -1.0
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping) or "time" not in raw or "shape" not in raw:
            raise ValueError(f"keyframes[{index}] must contain time and shape")
        time = _finite_number(raw["time"], f"keyframes[{index}].time")
        if time < 0 or time <= previous:
            raise ValueError("LIP keyframe times must be nonnegative and strictly ascending")
        if time > duration:
            raise ValueError(f"keyframes[{index}] occurs after the declared duration")
        shape = raw["shape"]
        if isinstance(shape, bool):
            raise ValueError(f"keyframes[{index}].shape cannot be boolean")
        if isinstance(shape, int):
            shape_index = shape
        elif isinstance(shape, str):
            name = shape.strip().upper()
            if name in shape_names:
                shape_index = shape_names[name]
            else:
                shape_index = LipDocument.shape_for_phoneme(name)
                if shape_index == 0 and name not in {" ", "-", "NEUTRAL"}:
                    raise ValueError(f"keyframes[{index}].shape is not a recognized LIP shape or ARPAbet phoneme")
        else:
            raise ValueError(f"keyframes[{index}].shape must be an integer or name")
        if not 0 <= shape_index <= 15:
            raise ValueError(f"keyframes[{index}].shape must be from 0 to 15")
        frames.append(LipKeyframeRecord(time, shape_index))
        previous = time
    document = LipDocument(duration, frames)
    payload = document.to_bytes()
    readback = LipDocument.load(payload)
    readback_blocking = [row.message for row in readback.validate() if row.blocking]
    same_timeline = (
        abs(readback.duration - document.duration) <= 1.0e-5
        and len(readback.keyframes) == len(document.keyframes)
        and all(
            abs(actual.time - expected.time) <= 1.0e-5 and actual.shape == expected.shape
            for actual, expected in zip(readback.keyframes, document.keyframes, strict=True)
        )
    )
    if readback_blocking or not same_timeline:
        raise ValueError("LIP semantic readback did not match the authored timeline")
    return {
        "size_bytes": len(payload),
        "keyframe_count": len(frames),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _write_ssf_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from src.core.scripting.data_authoring import SoundSetDocument
    from src.core.scripting.packaging import PackageResource

    game = _compat_game(arguments.get("game", "K1"))
    resref = _compat_resref(arguments.get("resref"))
    raw_slots = arguments.get("slots")
    if not isinstance(raw_slots, Mapping):
        raise ValueError("slots must be an object mapping slot names or indices to StrRefs")
    raw_unknown = arguments.get("unknown_slots")
    unknown_rows: list[tuple[int, int]] = []
    if raw_unknown is not None:
        if isinstance(raw_unknown, Mapping):
            iterator = raw_unknown.items()
        elif isinstance(raw_unknown, list):
            parsed: list[tuple[object, object]] = []
            for index, row in enumerate(raw_unknown):
                if not isinstance(row, Mapping) or "index" not in row or "strref" not in row:
                    raise ValueError(f"unknown_slots[{index}] must contain index and strref")
                parsed.append((row["index"], row["strref"]))
            iterator = parsed
        else:
            raise ValueError("unknown_slots must be an object or an array of index/strref rows")
        for raw_index, raw_stringref in iterator:
            index = _strict_integer(raw_index, "unknown slot index")
            stringref = _strict_integer(raw_stringref, f"unknown slot {index} StrRef")
            if not 28 <= index <= 4095:
                raise ValueError("unknown SSF slot indices must be from 28 to the safety limit 4095")
            if not -1 <= stringref <= 0xFFFFFFFE:
                raise ValueError(f"unknown SSF slot {index} StrRef is outside the supported range")
            unknown_rows.append((index, stringref))
    entry_count = max(40, max((index + 1 for index, _value in unknown_rows), default=40))
    document = SoundSetDocument([-1] * entry_count)
    for raw_slot, raw_stringref in raw_slots.items():
        stringref = _strict_integer(raw_stringref, f"slot {raw_slot!r} StrRef")
        if not -1 <= stringref <= 0xFFFFFFFE:
            raise ValueError(f"slot {raw_slot!r} StrRef is outside the supported range")
        if isinstance(raw_slot, bool):
            raise ValueError("boolean SSF slot keys are invalid")
        if isinstance(raw_slot, int) or (isinstance(raw_slot, str) and raw_slot.strip().isdigit()):
            slot: int | str = _strict_integer(raw_slot, "slot index")
        elif isinstance(raw_slot, str):
            slot = raw_slot.strip().upper()
        else:
            raise ValueError("SSF slot keys must be integers or canonical names")
        document.set_slot(slot, stringref)
    for index, stringref in unknown_rows:
        document.set_unknown_entry(index, stringref)
    payload = document.to_bytes()
    readback = SoundSetDocument.load(payload)
    if readback.snapshot() != document.snapshot():
        raise ValueError("SSF semantic readback did not preserve every named and unnamed entry")
    response: dict[str, Any] = {
        "game": game,
        "resref": resref,
        "size_bytes": len(payload),
        "slot_count": 28,
        "assigned_slot_count": sum(1 for value in document.named_stringrefs if value != -1),
        "entry_count": len(document.stringrefs),
        "data": base64.b64encode(payload).decode("ascii"),
    }
    if arguments.get("write_override") is True:
        response["write_override"] = _install_override_resource(
            {**dict(arguments), "game": game},
            resource=PackageResource(resref, "ssf", payload),
        )
    return response


def _write_pth_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    from pykotor.resource.generics.pth import read_pth
    from src.core.scripting.packaging import PackageResource

    game = _compat_game(arguments.get("game", "K1"))
    resref = _compat_resref(arguments.get("resref"))
    raw_points = arguments.get("points")
    if not isinstance(raw_points, list) or not raw_points or len(raw_points) > 1_000_000:
        raise ValueError("points must be a nonempty array with at most 1,000,000 entries")
    points: list[tuple[float, float]] = []
    connections: list[tuple[int, int]] = []
    seen_connections: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ValueError(f"points[{index}] must be an object")
        x = _finite_number(raw.get("x", 0.0), f"points[{index}].x")
        y = _finite_number(raw.get("y", 0.0), f"points[{index}].y")
        raw_connections = raw.get("connections", [])
        if not isinstance(raw_connections, list):
            raise ValueError(f"points[{index}].connections must be an array")
        points.append((x, y))
        for edge_index, raw_target in enumerate(raw_connections):
            target = _strict_integer(raw_target, f"points[{index}].connections[{edge_index}]")
            if not 0 <= target < len(raw_points):
                raise ValueError(f"points[{index}].connections[{edge_index}] is outside the point array")
            edge = (index, target)
            if index == target or edge in seen_connections:
                raise ValueError("PTH self-links and duplicate links are not valid authored path edges")
            seen_connections.add(edge)
            connections.append(edge)
            if len(connections) > 2_000_000:
                raise ValueError("PTH input exceeds the 2,000,000-edge compatibility safety limit")
    if len(points) > 1 and not connections:
        raise ValueError("a PTH graph with multiple points requires at least one directed connection")
    try:
        try:
            from src.core.modules.authored_module_pathing import (
                AuthoredPathConnection,
                AuthoredPathGraph,
                AuthoredPathPoint,
                build_authored_pth_bytes,
                validate_authored_path_graph,
            )
        except ImportError:
            from core.modules.authored_module_pathing import (  # type: ignore[no-redef]
                AuthoredPathConnection,
                AuthoredPathGraph,
                AuthoredPathPoint,
                build_authored_pth_bytes,
                validate_authored_path_graph,
            )
        graph = AuthoredPathGraph(
            points=tuple(AuthoredPathPoint(f"point_{index}", x, y) for index, (x, y) in enumerate(points)),
            connections=tuple(AuthoredPathConnection(source, target) for source, target in connections),
            metadata={"source": "legacy_ghostscripter_compatibility"},
        )
        validation = validate_authored_path_graph(graph)
        if not validation.ok:
            raise ValueError("; ".join(validation.blocking_issues))
        payload = bytes(build_authored_pth_bytes(graph))
    except ImportError:
        from pykotor.resource.generics.pth import PTH, bytes_pth

        graph = PTH()
        for x, y in points:
            graph.add(x, y)
        for source, target in connections:
            graph.connect(source, target)
        payload = bytes(bytes_pth(graph))
    if len(payload) > _COMPAT_MAX_BINARY_BYTES:
        raise ValueError("serialized PTH exceeds the 64 MiB compatibility safety limit")
    readback = read_pth(payload)
    if len(readback) != len(points):
        raise ValueError("PTH readback changed the point count")
    if any(
        abs(actual.x - expected[0]) > 1.0e-5 or abs(actual.y - expected[1]) > 1.0e-5
        for actual, expected in zip(readback, points, strict=True)
    ):
        raise ValueError("PTH readback changed point coordinates")
    actual_connections = []
    for source in range(len(readback)):
        actual_connections.extend((source, edge.target) for edge in readback.outgoing(source))
    if actual_connections != connections:
        raise ValueError("PTH readback changed directed connections or their order")
    response: dict[str, Any] = {
        "game": game,
        "resref": resref,
        "point_count": len(points),
        "connection_count": len(connections),
        "size_bytes": len(payload),
        "format": "PTH-GFF",
        "data": base64.b64encode(payload).decode("ascii"),
    }
    if arguments.get("write_override") is True:
        response["write_override"] = _install_override_resource(
            {**dict(arguments), "game": game},
            resource=PackageResource(resref, "pth", payload),
        )
    return response


async def handle_service_alias(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle safe compatibility commands backed by Qt-free scripting services."""

    if name == COMPATIBILITY_TOOL_NAME:
        report = compatibility_report()
        status = str(arguments.get("status", "") or "").strip()
        category = str(arguments.get("category", "") or "").strip()
        if status or category:
            report["tools"] = [
                row
                for row in report["tools"]
                if (not status or row["status"] == status) and (not category or row["category"] == category)
            ]
            report["filtered_count"] = len(report["tools"])
        return json_content(report, max_chars=150_000)

    try:
        from src.core.scripting.reference import NWScriptReferenceService, inspect_ncs

        game = str(arguments.get("game", "K2") or "K2")
        writer_handlers = {
            "writeGFF": _write_gff_payload,
            "writeDLG": _write_dialogue_payload,
            "writeTwoDA": _write_twoda_payload,
            "writeERF": _write_erf_payload,
            "writeLIP": _write_lip_payload,
            "writeSSF": _write_ssf_payload,
            "writePTH": _write_pth_payload,
            "writeOverride": _write_override_payload,
            "ghoststudio_restore_override": _restore_override_payload,
        }
        if name in writer_handlers:
            response = writer_handlers[name](arguments)
            return json_content(response, max_chars=max(250_000, _COMPAT_MAX_BINARY_BYTES * 2))
        if name == "nwscriptSignature":
            row = NWScriptReferenceService.function(str(arguments.get("name", "")), game=game)
            return json_content({"found": row is not None, "function": _function_payload(row) if row else None})
        if name == "nwscriptCategories":
            rows = NWScriptReferenceService.categories(game)
            return json_content({"game": game.upper(), "count": len(rows), "categories": list(rows)})
        if name == "searchNWScript":
            limit = min(200, max(1, int(arguments.get("limit", 50) or 50)))
            rows = NWScriptReferenceService.search_functions(
                str(arguments.get("query", "") or ""),
                game=game,
                category=str(arguments.get("category", "") or ""),
                limit=limit,
            )
            return json_content({"game": game.upper(), "count": len(rows), "functions": [_function_payload(row) for row in rows]})
        if name == "getNWScriptDB":
            offset = max(0, int(arguments.get("offset", 0) or 0))
            limit = min(200, max(1, int(arguments.get("limit", 100) or 100)))
            functions = NWScriptReferenceService.functions(game)
            constants = NWScriptReferenceService.constants(game)
            return json_content(
                {
                    "game": game.upper(),
                    "offset": offset,
                    "limit": limit,
                    "function_count": len(functions),
                    "constant_count": len(constants),
                    "functions": [_function_payload(row) for row in functions[offset : offset + limit]],
                    "constants": [_constant_payload(row) for row in constants[offset : offset + limit]],
                },
                max_chars=150_000,
            )
        if name == "readNCS":
            payload = _ncs_payload(arguments)
            row = inspect_ncs(
                payload,
                game=game,
                resref=str(arguments.get("resref", "script") or "script"),
                include_dirs=tuple(arguments.get("include_dirs", ()) or ()),
            )
            return json_content(
                {
                    "game": row.game,
                    "resref": row.resref,
                    "byte_count": row.byte_count,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "instruction_count": row.instruction_count,
                    "disassembly": row.disassembly,
                    "recovered_source": row.recovered_source,
                    "exact_recompile": row.exact_recompile,
                    "recompile_error": row.recompile_error,
                },
                max_chars=750_000,
            )

        if name == "readLTR":
            return json_content(_read_ltr_payload(arguments), max_chars=150_000)
        if name == "readVIS":
            return json_content(_read_vis_payload(arguments), max_chars=500_000)
        if name == "readWAV":
            return json_content(_read_wav_payload(arguments), max_chars=100_000)
        if name == "readTXI":
            return json_content(_read_txi_payload(arguments), max_chars=500_000)
        if name == "readSave":
            return json_content(_read_save_payload(arguments), max_chars=500_000)
        if name == "pathfindRoute":
            return json_content(_pathfind_route_payload(arguments), max_chars=250_000)
        if name == "getResource":
            return json_content(_get_resource_compat_payload(arguments), max_chars=1_750_000)
        if name == "getQuest":
            return json_content(_get_quest_compat_payload(arguments), max_chars=1_750_000)
        if name == "getModule":
            return json_content(_get_module_compat_payload(arguments), max_chars=1_750_000)
        if name == "getScript":
            return json_content(_get_script_payload(arguments), max_chars=1_750_000)
        if name == "getArea":
            return json_content(_get_area_payload(arguments), max_chars=1_500_000)
        if name == "getFaction":
            return json_content(_get_faction_payload(arguments), max_chars=1_000_000)
        if name == "getBlueprint" or name in _BLUEPRINT_COMPOSITE_TYPES:
            return json_content(_get_blueprint_payload(name, arguments), max_chars=1_500_000)

        if name in {"readSSF", "readLIP"}:
            from src.core.scripting.data_authoring import LipDocument, SoundSetDocument

            restype = "ssf" if name == "readSSF" else "lip"
            payload = _format_payload(arguments, restype)
            if name == "readSSF":
                document = SoundSetDocument.load(payload)
                names = document.slot_names()
                slots = [
                    {
                        "index": index,
                        "name": names[index] if index < len(names) else None,
                        "named": index < len(names),
                        "strref": stringref,
                    }
                    for index, stringref in enumerate(document.stringrefs)
                ]
                response = {
                    "resref": str(arguments.get("resref", "soundset")),
                    "slot_count": len(slots),
                    "named_slot_count": len(names),
                    "unnamed_slot_count": max(0, len(slots) - len(names)),
                    "table_offset": document.table_offset,
                    "header_padding_byte_count": len(document.header_padding),
                    "slots": slots,
                }
            else:
                document = LipDocument.load(payload)
                shapes = document.shape_names()
                response = {
                    "resref": str(arguments.get("resref", "voice")),
                    "duration": document.duration,
                    "keyframe_count": len(document.keyframes),
                    "keyframes": [
                        {
                            "time": frame.time,
                            "shape": frame.shape,
                            "shape_name": shapes[frame.shape] if 0 <= frame.shape < len(shapes) else "UNKNOWN",
                        }
                        for frame in document.keyframes
                    ],
                }
            response["byte_count"] = len(payload)
            response["sha256"] = hashlib.sha256(payload).hexdigest()
            return json_content(response, max_chars=250_000)

        if name in {"compileScript", "compileSummary", "decompileScript"}:
            from src.core.scripting.studio import ScriptDocument, ScriptingStudioService

            service = ScriptingStudioService()
            resref = str(arguments.get("resref", "script") or "script")
            if name == "decompileScript":
                payload = _ncs_payload(arguments)
                document, diagnostics = service.decompile_ncs(payload, game=game, resref=resref)
                return json_content(
                    {
                        "game": document.game,
                        "resref": document.resref,
                        "source": document.source,
                        "disassembly": document.disassembly,
                        "exact_recompile": document.recovered_source_exact,
                        "source_sha256": hashlib.sha256(document.source.encode("utf-8")).hexdigest(),
                        "ncs_sha256": hashlib.sha256(payload).hexdigest(),
                        "diagnostics": _diagnostics(diagnostics),
                    },
                    max_chars=750_000,
                )
            document = ScriptDocument(resref=resref, game=game, source=str(arguments.get("source", "") or ""))
            result = service.compile_script(document, include_dirs=tuple(arguments.get("include_dirs", ()) or ()))
            response: dict[str, Any] = {
                "ok": result.ok,
                "game": result.game,
                "resref": result.resref,
                "compiler": result.compiler,
                "readback_ok": result.readback_ok,
                "byte_count": len(result.ncs_bytes),
                "sha256": hashlib.sha256(result.ncs_bytes).hexdigest() if result.ncs_bytes else "",
                "diagnostics": _diagnostics(result.diagnostics),
            }
            if name == "compileScript" and result.ncs_bytes:
                response["ncs_base64"] = base64.b64encode(result.ncs_bytes).decode("ascii")
            return json_content(response, max_chars=max(100_000, len(response.get("ncs_base64", "")) + 10_000))

        if name == "twoDAChangesINI":
            from src.core.scripting.data_authoring import TwoDADocument

            original = TwoDADocument.load(base64.b64decode(str(arguments["original_2da_base64"]), validate=True))
            modified = TwoDADocument.load(base64.b64decode(str(arguments["modified_2da_base64"]), validate=True))
            changes = modified.export_changes_ini(original, str(arguments["table_name"]))
            return json_content(
                {
                    "table_name": str(arguments["table_name"]),
                    "changes_ini": changes,
                    "sha256": hashlib.sha256(changes.encode("utf-8")).hexdigest(),
                },
                max_chars=250_000,
            )
    except Exception as exc:
        payload = json_content({"error": str(exc).strip() or exc.__class__.__name__, "legacy_tool": name})
        if name in _GAME_INSTALL_WRITERS:
            payload["isError"] = True
        return payload

    raise ValueError(f"Unknown GhostScripter compatibility tool: {name}")


# Game-install mutators report refusals and failures as MCP isError results.
_GAME_INSTALL_WRITERS = frozenset({"writeOverride", "ghoststudio_restore_override"})


__all__ = [
    "COMPATIBILITY_TOOL_NAME",
    "DECLARED_EXTRA_ALIASES",
    "LEGACY_GHOSTSCRIPTER_COMPATIBILITY",
    "LegacyCompatibility",
    "compatibility_report",
    "get_tools",
    "handle_service_alias",
    "handles_service_alias",
    "resolve_direct_alias",
]
