"""Clean-room KOTOR script and dialogue authoring workflows.

This module is deliberately Qt-free.  It owns editable document contracts,
PyKotor-backed NSS/NCS compilation, DLG read/write validation, and transactional
staging.  The Scripting Suite window only presents these services.

The checks here prove compiler and structural readback, not retail-engine
execution.  A KOTOR 1/2 game proof is still required before a resource can be
described as engine-proven.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4


_RESREF_PATTERN = re.compile(r"^[a-z0-9_]{1,16}$")
_LINE_PATTERN = re.compile(r"(?:line|at)\s+(\d+)(?::(\d+))?", re.IGNORECASE)
_DIALOGUE_DISPLAY_TEXT_CACHE_ATTRIBUTE = "_ghostrigger_dialogue_display_text"
_DIALOGUE_DISPLAY_INACTIVE_PRESENT_ATTRIBUTE = "_ghostrigger_display_inactive_present"
_DIALOGUE_DISPLAY_INACTIVE_FIELD = "DisplayInactive"


def _check_nwscript_provider(game):
    """Apply Ghost Studio's PyKotor compiler fix, then prove JSR linking and ACTION order."""
    from src.core.scripting.pykotor_ncs_compiler_fix import verify_nwscript_compiler

    verify_nwscript_compiler(game)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _game_key(game: object) -> str:
    value = str(game or "K2").strip().upper()
    if value in {"K1", "1", "KOTOR", "KOTOR1"}:
        return "K1"
    if value in {"K2", "2", "TSL", "KOTOR2"}:
        return "K2"
    raise ValueError("Target game must be K1 or K2.")


def _pykotor_game(game: object):
    from pykotor.common.misc import Game

    return Game.K1 if _game_key(game) == "K1" else Game.K2


def normalise_script_resref(value: object, *, fallback: str = "") -> str:
    """Return a conservative KOTOR resource identifier without hiding errors."""

    text = Path(str(value or "").strip()).stem.lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if not text:
        text = str(fallback or "").strip().lower()
    return text[:16]


def _resref_diagnostic(resref: object, *, kind: str) -> "StudioDiagnostic | None":
    value = str(resref or "").strip().lower()
    if _RESREF_PATTERN.fullmatch(value):
        return None
    return StudioDiagnostic(
        severity="blocking",
        code="narrative.invalid_resref",
        message=f"{kind} resref must use 1-16 lowercase letters, numbers, or underscores.",
        resource=value,
    )


def _atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(bytes(data))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _atomic_write_text(path: str | Path, text: str) -> Path:
    return _atomic_write_bytes(path, str(text).encode("utf-8"))


def _quiet_bytes_gff(gff: object) -> bytes:
    """Serialize through PyKotor while containing its current validation prints."""

    from pykotor.resource.formats.gff import bytes_gff

    with redirect_stdout(io.StringIO()):
        return bytes(bytes_gff(gff))


def _quiet_read_dlg(data: bytes):
    """Read DLG bytes without leaking PyKotor's current field debug prints."""

    from pykotor.resource.generics.dlg import read_dlg

    with redirect_stdout(io.StringIO()):
        return read_dlg(bytes(data))


def _ordered_dialogue_links(links: Iterable[object]) -> list[object]:
    """Return links in the exact order used by PyKotor's DLG writer."""

    return sorted(
        tuple(links or ()),
        key=lambda link: (
            int(getattr(link, "list_index", -1)) == -1,
            int(getattr(link, "list_index", -1)),
        ),
    )


def _dialogue_link_gff_pairs(dialogue: object, gff: object) -> Iterable[tuple[object, object, str]]:
    """Pair every DLGLink with its typed GFF struct without relying on object hashes.

    PyKotor retains each reachable node's original ``list_index`` when reading an
    imported DLG, but its graph helpers omit detached raw EntryList/ReplyList
    records.  Source GFFs with those editor remnants must therefore be paired by
    original index.  Canonical writer output is compact and remains positional.
    """

    from pykotor.resource.formats.gff import GFFList

    def pair_list(
        links: Iterable[object],
        structs: object,
        path: str,
    ) -> Iterable[tuple[object, object, str]]:
        ordered = _ordered_dialogue_links(links)
        if len(ordered) != len(structs):
            raise ValueError(
                f"DLG/GFF link count mismatch at {path}: model={len(ordered)}, gff={len(structs)}"
            )
        for index, link in enumerate(ordered):
            yield link, structs[index], f"{path}[{index}]"

    def pair_nodes(
        nodes: Iterable[object],
        structs: object,
        path: str,
    ) -> Iterable[tuple[object, object, str]]:
        ordered = tuple(nodes)
        if len(ordered) == len(structs):
            for index, node in enumerate(ordered):
                yield node, structs[index], f"{path}[{index}]"
            return

        seen_indices: set[int] = set()
        for node in ordered:
            index = int(getattr(node, "list_index", -1))
            if index < 0 or index >= len(structs) or index in seen_indices:
                raise ValueError(
                    f"DLG/GFF node index mismatch at {path}: "
                    f"list_index={index}, model={len(ordered)}, gff={len(structs)}"
                )
            seen_indices.add(index)
            yield node, structs[index], f"{path}[{index}]"

    root = gff.root
    starters = root.acquire("StartingList", GFFList())
    yield from pair_list(getattr(dialogue, "starters", ()), starters, "StartingList")

    entries = tuple(dialogue.all_entries(as_sorted=True))
    entry_structs = root.acquire("EntryList", GFFList())
    for entry, entry_struct, entry_path in pair_nodes(entries, entry_structs, "EntryList"):
        link_structs = entry_struct.acquire("RepliesList", GFFList())
        yield from pair_list(
            getattr(entry, "links", ()),
            link_structs,
            f"{entry_path}.RepliesList",
        )

    replies = tuple(dialogue.all_replies(as_sorted=True))
    reply_structs = root.acquire("ReplyList", GFFList())
    for reply, reply_struct, reply_path in pair_nodes(replies, reply_structs, "ReplyList"):
        link_structs = reply_struct.acquire("EntriesList", GFFList())
        yield from pair_list(
            getattr(reply, "links", ()),
            link_structs,
            f"{reply_path}.EntriesList",
        )


def _hydrate_dialogue_display_inactive(dialogue: object, gff: object) -> None:
    """Hydrate PyKotor's omitted DisplayInactive BYTE onto each DLGLink."""

    for link, link_struct, _path in _dialogue_link_gff_pairs(dialogue, gff):
        present = bool(link_struct.exists(_DIALOGUE_DISPLAY_INACTIVE_FIELD))
        setattr(link, _DIALOGUE_DISPLAY_INACTIVE_PRESENT_ATTRIBUTE, present)
        setattr(
            link,
            "display_inactive",
            bool(link_struct.acquire(_DIALOGUE_DISPLAY_INACTIVE_FIELD, 0)) if present else False,
        )


def _patch_dialogue_display_inactive(dialogue: object, gff: object, game: object) -> None:
    """Post-patch K2 DisplayInactive as a typed BYTE after PyKotor dismantling.

    False fields that were absent in the source stay absent for vanilla-like
    output.  A field that existed in an imported DLG remains explicit, allowing
    an edit from true to false to overwrite the imported value losslessly.
    """

    if _game_key(game) != "K2":
        return
    for link, link_struct, _path in _dialogue_link_gff_pairs(dialogue, gff):
        value = bool(getattr(link, "display_inactive", False))
        source_present = bool(
            getattr(link, _DIALOGUE_DISPLAY_INACTIVE_PRESENT_ATTRIBUTE, False)
        )
        if value or source_present:
            link_struct.set_uint8(_DIALOGUE_DISPLAY_INACTIVE_FIELD, int(value))
        elif link_struct.exists(_DIALOGUE_DISPLAY_INACTIVE_FIELD):
            link_struct.remove(_DIALOGUE_DISPLAY_INACTIVE_FIELD)


def _verify_dialogue_display_inactive(dialogue: object, gff: object, game: object) -> None:
    """Require K2 DisplayInactive values to survive as typed GFF BYTE fields."""

    if _game_key(game) != "K2":
        return
    from pykotor.resource.formats.gff import GFFFieldType

    for link, link_struct, path in _dialogue_link_gff_pairs(dialogue, gff):
        expected = bool(getattr(link, "display_inactive", False))
        expected_present = expected or bool(
            getattr(link, _DIALOGUE_DISPLAY_INACTIVE_PRESENT_ATTRIBUTE, False)
        )
        actual_present = bool(link_struct.exists(_DIALOGUE_DISPLAY_INACTIVE_FIELD))
        if actual_present != expected_present:
            raise ValueError(
                f"{_DIALOGUE_DISPLAY_INACTIVE_FIELD} presence changed at {path}: "
                f"expected={expected_present}, actual={actual_present}"
            )
        if not actual_present:
            continue
        field_type = link_struct.what_type(_DIALOGUE_DISPLAY_INACTIVE_FIELD)
        if field_type is not GFFFieldType.UInt8:
            raise ValueError(
                f"{_DIALOGUE_DISPLAY_INACTIVE_FIELD} at {path} is {field_type}, expected UInt8"
            )
        actual = bool(link_struct.get_uint8(_DIALOGUE_DISPLAY_INACTIVE_FIELD))
        if actual != expected:
            raise ValueError(
                f"{_DIALOGUE_DISPLAY_INACTIVE_FIELD} changed at {path}: "
                f"expected={expected}, actual={actual}"
            )


@dataclass(frozen=True)
class StudioDiagnostic:
    severity: str
    code: str
    message: str
    resource: str = ""
    line: int | None = None
    column: int | None = None
    fix_hint: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity.strip().lower() in {"blocking", "error"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "resource": self.resource,
            "line": self.line,
            "column": self.column,
            "fix_hint": self.fix_hint,
        }


@dataclass
class ScriptDocument:
    document_id: str = field(default_factory=lambda: f"script_{uuid4().hex}")
    resref: str = "new_script"
    game: str = "K2"
    source: str = "void main()\n{\n    // Add KOTOR gameplay actions here.\n}\n"
    source_path: str = ""
    origin: str = "new"
    dirty: bool = True
    last_compiled_path: str = ""
    last_compiled_sha256: str = ""
    disassembly: str = ""
    decompiled_from_sha256: str = ""
    recovered_source_exact: bool = False
    recovery_error: str = ""

    def __post_init__(self) -> None:
        self.game = _game_key(self.game)
        self.resref = normalise_script_resref(self.resref, fallback="new_script")

    @property
    def display_name(self) -> str:
        return f"{self.resref}.nss"


@dataclass
class DialogueDocument:
    document_id: str = field(default_factory=lambda: f"dialogue_{uuid4().hex}")
    resref: str = "new_dialogue"
    game: str = "K2"
    dialogue: Any = None
    source_path: str = ""
    source_bytes: bytes = b""
    source_structure: dict[str, int] = field(default_factory=dict)
    origin: str = "new"
    dirty: bool = True

    def __post_init__(self) -> None:
        self.game = _game_key(self.game)
        self.resref = normalise_script_resref(self.resref, fallback="new_dialogue")

    @property
    def display_name(self) -> str:
        return f"{self.resref}.dlg"


@dataclass(frozen=True)
class ScriptCompileResult:
    resref: str
    game: str
    ncs_bytes: bytes = b""
    diagnostics: tuple[StudioDiagnostic, ...] = ()
    compiler: str = "PyKotor compile_nss"
    readback_ok: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.ncs_bytes and self.readback_ok and not any(row.blocking for row in self.diagnostics))


@dataclass(frozen=True)
class NarrativeResource:
    resref: str
    restype: str
    data: bytes
    source_document_id: str = ""

    @property
    def filename(self) -> str:
        return f"{self.resref}.{self.restype.lower()}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class NarrativeBuildResult:
    output_dir: str
    game: str
    resources: tuple[NarrativeResource, ...] = ()
    diagnostics: tuple[StudioDiagnostic, ...] = ()
    manifest_path: str = ""
    committed: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.committed and not any(row.blocking for row in self.diagnostics))

    def resource_tuples(self, *, runtime_only: bool = True) -> tuple[tuple[str, str, bytes], ...]:
        rows = self.resources
        if runtime_only:
            rows = tuple(row for row in rows if row.restype.lower() in {"ncs", "dlg"})
        return tuple((row.resref, row.restype.lower(), bytes(row.data)) for row in rows)


def _narrative_output_ownership_diagnostic(output: Path) -> StudioDiagnostic | None:
    """Refuse to replace a nonempty directory not owned by this build service."""

    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        return StudioDiagnostic(
            "blocking",
            "narrative.output_not_owned",
            f"Narrative build output is not a directory: {output}",
            fix_hint="Choose an empty folder or an existing GhostStudio narrative-build folder.",
        )
    try:
        entries = tuple(output.iterdir())
    except Exception as access_error:
        return StudioDiagnostic(
            "blocking",
            "narrative.output_not_owned",
            f"GhostStudio cannot inspect the narrative output safely: {output}. {access_error}",
            fix_hint="Check destination permissions or choose an empty build folder.",
        )
    if not entries:
        return None
    manifest_path = output / "narrative-build.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = None
    allowed_names = {"narrative-build.json", "resources"}
    unknown_names = sorted(entry.name for entry in entries if entry.name not in allowed_names)
    if not isinstance(manifest, dict) or manifest.get("file_type") != "GhostStudioNarrativeBuild":
        return StudioDiagnostic(
            "blocking",
            "narrative.output_not_owned",
            f"Refusing to replace nonempty output that is not a GhostStudio narrative build: {output}",
            fix_hint="Choose an empty folder or move unrelated files out of this destination.",
        )
    if unknown_names:
        return StudioDiagnostic(
            "blocking",
            "narrative.output_not_owned",
            "Refusing to replace a narrative output containing unrelated root items: "
            + ", ".join(unknown_names),
            fix_hint="Move unrelated files out of the build folder, then build again.",
        )
    resources_path = output / "resources"
    if resources_path.exists() and not resources_path.is_dir():
        return StudioDiagnostic(
            "blocking",
            "narrative.output_not_owned",
            f"Narrative resources path is not a directory: {resources_path}",
            fix_hint="Repair or choose a different GhostStudio narrative-build folder.",
        )
    return None


def _promote_build_directory(staging: Path, output: Path) -> tuple[bool, tuple[StudioDiagnostic, ...]]:
    """Swap a complete staged build into place and restore the prior build on failure."""

    backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    previous_moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            previous_moved = True
        os.replace(staging, output)
    except Exception as promotion_error:
        if previous_moved:
            try:
                os.replace(backup, output)
            except Exception as rollback_error:
                return False, (
                    StudioDiagnostic(
                        "blocking",
                        "narrative.build_rollback_failed",
                        "Narrative build promotion failed and GhostStudio could not restore the previous output. "
                        f"The previous build is preserved at {backup}. Promotion error: {promotion_error}. "
                        f"Rollback error: {rollback_error}.",
                        fix_hint=f"Close programs using the build folder, then restore or inspect {backup}.",
                    ),
                )
        failure_message = (
            "Narrative build promotion failed; the previous output was restored and no partial new build is active. "
            if previous_moved
            else "Narrative build promotion failed before the live output changed; no partial build is active. "
        )
        return False, (
            StudioDiagnostic(
                "blocking",
                "narrative.build_promotion_failed",
                failure_message + str(promotion_error),
                fix_hint="Close programs using the build folder and build again.",
            ),
        )

    diagnostics: list[StudioDiagnostic] = []
    if previous_moved and backup.exists():
        try:
            shutil.rmtree(backup)
        except Exception as cleanup_error:
            diagnostics.append(
                StudioDiagnostic(
                    "warning",
                    "narrative.previous_build_cleanup_failed",
                    f"The new build is active, but the previous backup could not be removed: {backup}. {cleanup_error}",
                    fix_hint=f"Remove {backup} after closing programs that may be using it.",
                )
            )
    return True, tuple(diagnostics)


def _remember_dialogue_display_text(node: object, localized_string: object, value: str) -> str:
    """Remember the exact text presented for a LocalizedString-backed node.

    Dialogue inspectors submit every visible field together.  A TLK-backed
    LocalizedString cannot reproduce its resolved display text without the game
    TLK, so retaining this transient pair lets the subsequent setter distinguish
    an unchanged inspector value from an intentional literal-text replacement.
    PyKotor's DLG writer serializes explicit node fields and ignores this private
    presentation cache.
    """

    try:
        setattr(
            node,
            _DIALOGUE_DISPLAY_TEXT_CACHE_ATTRIBUTE,
            (localized_string, str(value)),
        )
    except Exception:
        pass
    return str(value)


def dialogue_node_text(node: object, *, tlk_lookup: Any = None) -> str:
    text = getattr(node, "text", None)
    if text is None:
        return ""
    stringref = int(getattr(text, "stringref", -1) or -1)
    if stringref >= 0 and callable(tlk_lookup):
        try:
            resolved = str(tlk_lookup(stringref) or "")
            if resolved:
                return _remember_dialogue_display_text(node, text, resolved)
        except Exception:
            pass
    try:
        from pykotor.common.language import Gender, Language

        embedded = text.get(Language.ENGLISH, Gender.MALE, use_fallback=True)
        if embedded:
            return _remember_dialogue_display_text(node, text, str(embedded))
    except Exception:
        pass
    for value in dict(getattr(text, "_substrings_internal", {}) or {}).values():
        if value:
            return _remember_dialogue_display_text(node, text, str(value))
    fallback = f"<TLK {stringref}>" if stringref >= 0 else ""
    return _remember_dialogue_display_text(node, text, fallback)


def set_dialogue_node_text(node: object, value: str) -> None:
    from pykotor.common.language import LocalizedString

    submitted = str(value or "")
    current = getattr(node, "text", None)
    cached = getattr(node, _DIALOGUE_DISPLAY_TEXT_CACHE_ATTRIBUTE, None)
    cache_matches_current = (
        isinstance(cached, tuple)
        and len(cached) == 2
        and cached[0] is current
    )
    if cache_matches_current:
        if submitted == str(cached[1]):
            return
    elif current is not None and submitted == dialogue_node_text(node):
        return

    replacement = LocalizedString.from_english(submitted)
    node.text = replacement
    _remember_dialogue_display_text(node, replacement, submitted)


def _walk_dialogue_links(dialogue: object) -> Iterable[tuple[object, object, int]]:
    """Yield each reachable link/node pair once while tolerating child cycles."""

    pending: list[tuple[object, int]] = [
        (link, 0) for link in tuple(getattr(dialogue, "starters", ()) or ())
    ]
    seen_links: set[int] = set()
    cursor = 0
    while cursor < len(pending):
        link, depth = pending[cursor]
        cursor += 1
        if id(link) in seen_links:
            continue
        seen_links.add(id(link))
        node = getattr(link, "node", None)
        yield link, node, depth
        if node is None:
            continue
        for child in tuple(getattr(node, "links", ()) or ()):
            pending.append((child, depth + 1))


def dialogue_structure_summary(dialogue: object) -> dict[str, int]:
    links = list(_walk_dialogue_links(dialogue))
    nodes = {id(node) for _link, node, _depth in links if node is not None}
    from pykotor.resource.generics.dlg import DLGEntry, DLGReply

    return {
        "starters": len(tuple(getattr(dialogue, "starters", ()) or ())),
        "links": len(links),
        "nodes": len(nodes),
        "entries": len({id(node) for _link, node, _depth in links if isinstance(node, DLGEntry)}),
        "replies": len({id(node) for _link, node, _depth in links if isinstance(node, DLGReply)}),
    }


def _gff_unknown_fingerprint(source_struct: object, canonical_struct: object, path: str = "root") -> dict[str, str]:
    """Return unknown GFF fields and values retained outside PyKotor's DLG model."""

    result: dict[str, str] = {}
    canonical_fields = {label: (field_type, value) for label, field_type, value in canonical_struct}
    for label, field_type, value in source_struct:
        field_path = f"{path}.{label}"
        canonical = canonical_fields.get(label)
        if canonical is None or canonical[0] != field_type:
            result[field_path] = f"{getattr(field_type, 'name', field_type)}:{value!r}"
            continue
        canonical_value = canonical[1]
        type_name = str(getattr(field_type, "name", field_type))
        if type_name == "Struct":
            result.update(_gff_unknown_fingerprint(value, canonical_value, field_path))
        elif type_name == "List":
            for index, child in enumerate(value):
                if index >= len(canonical_value):
                    result[f"{field_path}[{index}]"] = f"Struct:{child!r}"
                    continue
                result.update(
                    _gff_unknown_fingerprint(child, canonical_value[index], f"{field_path}[{index}]")
                )
    return result


def _overlay_canonical_gff(target_struct: object, canonical_struct: object, path: str = "root") -> None:
    """Overlay known canonical fields while retaining unknown source fields."""

    for label, field_type, canonical_value in canonical_struct:
        type_name = str(getattr(field_type, "name", field_type))
        existing_type = target_struct.what_type(label) if target_struct.exists(label) else None
        if existing_type != field_type:
            target_struct._fields[label] = deepcopy(canonical_struct._fields[label])  # noqa: SLF001 - PyKotor has no generic setter
            continue
        if type_name == "Struct":
            _overlay_canonical_gff(target_struct.get_struct(label), canonical_value, f"{path}.{label}")
            continue
        if type_name == "List":
            target_list = target_struct.get_list(label)
            if len(target_list) != len(canonical_value):
                target_struct._fields[label] = deepcopy(canonical_struct._fields[label])  # noqa: SLF001
                continue
            for index, canonical_child in enumerate(canonical_value):
                _overlay_canonical_gff(target_list[index], canonical_child, f"{path}.{label}[{index}]")
            continue
        target_struct._fields[label] = deepcopy(canonical_struct._fields[label])  # noqa: SLF001


def _preserve_imported_dialogue_fields(document: DialogueDocument, canonical_gff: object) -> tuple[bytes, dict[str, str]]:
    """Write a topology-stable imported DLG without discarding unknown GFF data."""

    from pykotor.resource.formats.gff import read_gff
    from pykotor.resource.generics.dlg import dismantle_dlg, read_dlg

    original_gff = read_gff(document.source_bytes)
    original_dialogue = read_dlg(document.source_bytes)
    _hydrate_dialogue_display_inactive(original_dialogue, original_gff)
    original_canonical = dismantle_dlg(original_dialogue, _pykotor_game(document.game))
    _patch_dialogue_display_inactive(original_dialogue, original_canonical, document.game)
    unknown_before = _gff_unknown_fingerprint(original_gff.root, original_canonical.root)
    current_summary = dialogue_structure_summary(document.dialogue)
    source_summary = document.source_structure or dialogue_structure_summary(original_dialogue)
    if current_summary != source_summary and unknown_before:
        raise ValueError(
            "This imported DLG contains unknown engine fields and its graph topology changed. "
            "GhostStudio blocked the write instead of discarding unmapped data; edit node properties only, "
            "or create an authored copy before changing the graph."
        )
    if current_summary != source_summary:
        return _quiet_bytes_gff(canonical_gff), {}
    preserved_gff = deepcopy(original_gff)
    _overlay_canonical_gff(preserved_gff.root, canonical_gff.root)
    payload = _quiet_bytes_gff(preserved_gff)
    written_gff = read_gff(payload)
    written_dialogue = read_dlg(payload)
    _hydrate_dialogue_display_inactive(written_dialogue, written_gff)
    written_canonical = dismantle_dlg(written_dialogue, _pykotor_game(document.game))
    _patch_dialogue_display_inactive(written_dialogue, written_canonical, document.game)
    unknown_after = _gff_unknown_fingerprint(written_gff.root, written_canonical.root)
    missing_or_changed = {
        key: value for key, value in unknown_before.items() if unknown_after.get(key) != value
    }
    if missing_or_changed:
        sample = ", ".join(list(missing_or_changed)[:4])
        raise ValueError(f"Unknown DLG fields did not survive the preservation write: {sample}")
    return payload, unknown_before


def imported_dialogue_unknown_fields(document: DialogueDocument) -> dict[str, str]:
    """Describe unmapped fields that make imported topology edits unsafe."""

    if not document.source_bytes:
        return {}
    from pykotor.resource.formats.gff import read_gff
    from pykotor.resource.generics.dlg import dismantle_dlg, read_dlg

    source_gff = read_gff(document.source_bytes)
    source_dialogue = read_dlg(document.source_bytes)
    _hydrate_dialogue_display_inactive(source_dialogue, source_gff)
    canonical = dismantle_dlg(source_dialogue, _pykotor_game(document.game))
    _patch_dialogue_display_inactive(source_dialogue, canonical, document.game)
    return _gff_unknown_fingerprint(source_gff.root, canonical.root)


class ScriptingStudioService:
    """Own script/dialogue IO, validation, compilation, and staged builds."""

    def new_script(self, *, game: str = "K2", resref: str = "new_script") -> ScriptDocument:
        return ScriptDocument(game=game, resref=resref)

    def load_script(self, path: str | Path, *, game: str = "K2") -> ScriptDocument:
        source_path = Path(path)
        data = source_path.read_bytes()
        try:
            source = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            source = data.decode("latin-1")
        return ScriptDocument(
            resref=source_path.stem,
            game=game,
            source=source,
            source_path=str(source_path),
            origin="local_file",
            dirty=False,
        )

    def script_from_bytes(
        self,
        data: bytes,
        *,
        game: str,
        resref: str,
        origin: str = "game_resource",
    ) -> ScriptDocument:
        payload = bytes(data or b"")
        try:
            source = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            source = payload.decode("latin-1")
        return ScriptDocument(resref=resref, game=game, source=source, origin=origin, dirty=False)

    def save_script(self, document: ScriptDocument, path: str | Path | None = None) -> Path:
        target_value = str(path or document.source_path or "").strip()
        if not target_value:
            raise ValueError("Choose an NSS file path before saving this script.")
        target = Path(target_value)
        issue = _resref_diagnostic(document.resref, kind="Script")
        if issue is not None:
            raise ValueError(issue.message)
        _atomic_write_text(target, document.source)
        document.source_path = str(target)
        document.resref = normalise_script_resref(target.stem, fallback=document.resref)
        document.dirty = False
        return target

    def validate_script(self, document: ScriptDocument) -> tuple[StudioDiagnostic, ...]:
        diagnostics: list[StudioDiagnostic] = []
        issue = _resref_diagnostic(document.resref, kind="Script")
        if issue is not None:
            diagnostics.append(issue)
        if not document.source.strip():
            diagnostics.append(
                StudioDiagnostic("blocking", "script.empty", "Script source is empty.", document.resref)
            )
        if "__NCS_BYTECODE__" in document.source or "__END_NCS_BYTECODE__" in document.source:
            diagnostics.append(
                StudioDiagnostic(
                    "blocking",
                    "script.embedded_bytecode_blocked",
                    "Authored NSS cannot contain an embedded NCS bytecode block.",
                    document.resref,
                    fix_hint="Remove the embedded bytecode block and compile auditable NWScript source.",
                )
            )
        if "void main" not in document.source and "int StartingConditional" not in document.source:
            diagnostics.append(
                StudioDiagnostic(
                    "warning",
                    "script.entrypoint_not_obvious",
                    "No main() or StartingConditional() declaration was found by the quick check; compilation is authoritative.",
                    document.resref,
                )
            )
        return tuple(diagnostics)

    def compile_native_script(
        self, document: ScriptDocument, *, compiler_path: Path, source_path: Path, output_dir: Path,
    ) -> ScriptCompileResult:
        """Compile a staged source with PyKotor's fingerprinted external compiler provider.

        The caller supplies a qualified compiler and its target-game nwscript.nss.
        source_path may have sibling include files. A fresh output directory keeps
        failed compilations from accepting stale bytecode left by another call.
        """
        from pykotor.resource.formats.ncs import read_ncs
        from pykotor.resource.formats.ncs.compilers import ExternalNCSCompiler

        diagnostics = list(self.validate_script(document))
        backend = "PyKotor ExternalNCSCompiler"
        payload = b""
        if not any(row.blocking for row in diagnostics):
            try:
                source_path = Path(source_path).resolve(strict=True)
                if source_path.name.casefold() != (document.resref + ".nss").casefold():
                    raise ValueError("Staged NSS filename does not match the document.")
                if source_path.read_text(encoding="utf-8-sig") != document.source:
                    raise ValueError("Staged NSS differs from the requested document.")
                compiler = ExternalNCSCompiler(Path(compiler_path).resolve(strict=True))
                backend += " / " + compiler.get_info().name
                output_dir = Path(output_dir).resolve()
                output_dir.mkdir(exist_ok=False)
                target = output_dir / (document.resref + ".ncs")
                stdout, stderr = compiler.compile_script(source_path, target, _pykotor_game(document.game))
                if stderr.strip() or not target.is_file():
                    raise ValueError((stderr.strip() or stdout.strip() or "Native compiler produced no NCS.")[:1000])
                candidate = target.read_bytes()
                if candidate[:9] != b"NCS V1.0\x42" or int.from_bytes(candidate[9:13], "big") != len(candidate):
                    raise ValueError("Native compiler produced an invalid NCS header or size.")
                if not read_ncs(candidate).instructions:
                    raise ValueError("Native compiler produced no instructions.")
                payload = candidate
            except (OSError, RuntimeError, ValueError) as exc:
                diagnostics.append(StudioDiagnostic("blocking", "script.native_compile_failed", str(exc), document.resref))
        return ScriptCompileResult(document.resref, document.game, ncs_bytes=payload, compiler=backend,
                                   diagnostics=tuple(diagnostics), readback_ok=bool(payload))

    def compile_script(
        self,
        document: ScriptDocument,
        *,
        include_dirs: Sequence[str | Path] = (),
    ) -> ScriptCompileResult:
        diagnostics = list(self.validate_script(document))
        if any(row.blocking for row in diagnostics):
            return ScriptCompileResult(document.resref, document.game, diagnostics=tuple(diagnostics))
        try:
            from pykotor.resource.formats.ncs import bytes_ncs, read_ncs
            from src.core.scripting.pykotor_ncs_compiler_fix import compile_nss_verified

            compiled = compile_nss_verified(
                document.source,
                _pykotor_game(document.game),
                library_lookup=[Path(value) for value in include_dirs] or None,
            )
            payload = bytes(bytes_ncs(compiled))
            if not payload.startswith(b"NCS V1.0"):
                raise ValueError("Compiler output does not contain the KOTOR NCS V1.0 header.")
            read_ncs(payload)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            match = _LINE_PATTERN.search(message)
            diagnostics.append(
                StudioDiagnostic(
                    "blocking",
                    "script.compile_failed",
                    message,
                    document.resref,
                    int(match.group(1)) if match else None,
                    int(match.group(2)) if match and match.group(2) else None,
                    "Open the diagnostic location, correct the NWScript source, and compile again.",
                )
            )
            return ScriptCompileResult(document.resref, document.game, diagnostics=tuple(diagnostics))
        diagnostics.append(
            StudioDiagnostic(
                "info",
                "script.compiler_readback_passed",
                f"Compiled {len(payload)} NCS bytes and parsed the result back successfully. Retail KOTOR proof remains required.",
                document.resref,
            )
        )
        if document.game == "K2":
            diagnostics.append(
                StudioDiagnostic(
                    "warning",
                    "script.k2_dialect_subset",
                    "Compilation used PyKotor's bundled K2 definitions. Vanilla nwscript.nss dialect fingerprinting is still required for uncommon late K2 actions.",
                    document.resref,
                )
            )
        return ScriptCompileResult(
            document.resref,
            document.game,
            ncs_bytes=payload,
            diagnostics=tuple(diagnostics),
            readback_ok=True,
        )

    def decompile_ncs(self, data: bytes, *, game: str, resref: str) -> tuple[ScriptDocument, tuple[StudioDiagnostic, ...]]:
        payload = bytes(data or b"")
        from src.core.scripting.reference import inspect_ncs

        inspection = inspect_ncs(payload, game=game, resref=resref)
        source = inspection.recovered_source
        if not source.strip():
            source = (
                "// GhostStudio could not reconstruct editable NWScript from this NCS.\n"
                "// Use the authoritative Disassembly tab for inspection.\n"
            )
        document = ScriptDocument(
            resref=resref,
            game=game,
            source=source,
            origin="decompiled_ncs",
            dirty=True,
            disassembly=inspection.disassembly,
            decompiled_from_sha256=hashlib.sha256(payload).hexdigest(),
            recovered_source_exact=inspection.exact_recompile,
            recovery_error=inspection.recompile_error,
        )
        if inspection.exact_recompile:
            diagnostic = StudioDiagnostic(
                "info",
                "script.decompile_exact_recompile",
                "Recovered source recompiled to bytes identical to the imported NCS. The instruction listing remains authoritative.",
                document.resref,
            )
        else:
            detail = f" Recompile check: {inspection.recompile_error}" if inspection.recompile_error else ""
            diagnostic = StudioDiagnostic(
                "warning",
                "script.decompile_requires_review",
                "Recovered source is not byte-identical to the imported NCS and must not replace it without review."
                + detail,
                document.resref,
            )
        return document, (diagnostic,)

    def new_dialogue(self, *, game: str = "K2", resref: str = "new_dialogue") -> DialogueDocument:
        from pykotor.common.language import LocalizedString
        from pykotor.resource.generics.dlg import DLG, DLGEntry, DLGLink

        dialogue = DLG()
        entry = DLGEntry()
        entry.speaker = "OWNER"
        entry.text = LocalizedString.from_english("New dialogue line")
        dialogue.starters.append(DLGLink(entry))
        return DialogueDocument(resref=resref, game=game, dialogue=dialogue)

    def load_dialogue(self, path: str | Path, *, game: str = "K2") -> DialogueDocument:
        source_path = Path(path)
        document = self.dialogue_from_bytes(
            source_path.read_bytes(), game=game, resref=source_path.stem, origin="local_file"
        )
        document.source_path = str(source_path)
        return document

    def dialogue_from_bytes(
        self,
        data: bytes,
        *,
        game: str,
        resref: str,
        origin: str = "game_resource",
    ) -> DialogueDocument:
        payload = bytes(data or b"")
        dialogue = _quiet_read_dlg(payload)
        from pykotor.resource.formats.gff import read_gff

        _hydrate_dialogue_display_inactive(dialogue, read_gff(payload))
        return DialogueDocument(
            resref=resref,
            game=game,
            dialogue=dialogue,
            source_bytes=payload,
            source_structure=dialogue_structure_summary(dialogue),
            origin=origin,
            dirty=False,
        )

    def dialogue_topology_requires_editable_copy(self, document: DialogueDocument) -> bool:
        """Whether a topology edit would discard fields GhostStudio cannot map."""

        return bool(imported_dialogue_unknown_fields(document))

    def make_editable_dialogue_copy(
        self,
        document: DialogueDocument,
        *,
        resref: str,
        source_path: str | Path = "",
    ) -> DialogueDocument:
        """Clone known DLG data into a new authored document without touching source.

        Unknown imported GFF fields remain protected in ``document.source_bytes``.
        The returned document intentionally has no imported-byte overlay, so graph
        topology may be changed and saved to a distinct user-selected path.
        """

        target = Path(source_path) if str(source_path or "").strip() else None
        if target is not None and document.source_path:
            try:
                same_path = target.resolve(strict=False) == Path(document.source_path).resolve(strict=False)
            except OSError:
                same_path = str(target) == str(document.source_path)
            if same_path:
                raise ValueError("Choose a new path for the authored copy; the imported DLG remains protected.")
        return DialogueDocument(
            resref=normalise_script_resref(resref, fallback=f"{document.resref}_copy"),
            game=document.game,
            dialogue=deepcopy(document.dialogue),
            source_path=str(target) if target is not None else "",
            source_bytes=b"",
            source_structure=dialogue_structure_summary(document.dialogue),
            origin="authored_copy",
            dirty=True,
        )

    def validate_dialogue(self, document: DialogueDocument) -> tuple[StudioDiagnostic, ...]:
        diagnostics: list[StudioDiagnostic] = []
        issue = _resref_diagnostic(document.resref, kind="Dialogue")
        if issue is not None:
            diagnostics.append(issue)
        dialogue = document.dialogue
        if dialogue is None:
            diagnostics.append(
                StudioDiagnostic("blocking", "dialogue.missing_model", "Dialogue document has no DLG model.", document.resref)
            )
            return tuple(diagnostics)
        summary = dialogue_structure_summary(dialogue)
        if not summary["starters"]:
            diagnostics.append(
                StudioDiagnostic(
                    "blocking",
                    "dialogue.no_starter",
                    "Dialogue needs at least one starting entry.",
                    document.resref,
                )
            )
        for link, node, _depth in _walk_dialogue_links(dialogue):
            if node is None:
                diagnostics.append(
                    StudioDiagnostic(
                        "blocking",
                        "dialogue.broken_link",
                        "A dialogue link has no target node.",
                        document.resref,
                    )
                )
            for field_name in ("script1", "script2"):
                value = str(getattr(node, field_name, "") or "").strip().lower() if node is not None else ""
                if value and not _RESREF_PATTERN.fullmatch(value):
                    diagnostics.append(
                        StudioDiagnostic(
                            "blocking",
                            "dialogue.invalid_script_resref",
                            f"{field_name} uses invalid script resref '{value}'.",
                            document.resref,
                        )
                    )
            for field_name in ("active1", "active2"):
                value = str(getattr(link, field_name, "") or "").strip().lower()
                if value and not _RESREF_PATTERN.fullmatch(value):
                    diagnostics.append(
                        StudioDiagnostic(
                            "blocking",
                            "dialogue.invalid_condition_resref",
                            f"{field_name} uses invalid condition resref '{value}'.",
                            document.resref,
                        )
                    )
        from src.core.scripting.dialogue_contract import validate_dialogue_authoring

        existing = {(row.code, row.message) for row in diagnostics}
        for issue in validate_dialogue_authoring(dialogue, game=document.game):
            identity = (issue.code, issue.message)
            if identity in existing:
                continue
            existing.add(identity)
            diagnostics.append(
                StudioDiagnostic(
                    issue.severity,
                    issue.code,
                    issue.message,
                    document.resref,
                    fix_hint=f"Review DLG field '{issue.field}' on {issue.object_id}." if issue.field else "",
                )
            )
        return tuple(diagnostics)

    def dialogue_bytes(self, document: DialogueDocument) -> tuple[bytes, tuple[StudioDiagnostic, ...]]:
        diagnostics = list(self.validate_dialogue(document))
        if any(row.blocking for row in diagnostics):
            return b"", tuple(diagnostics)
        before = dialogue_structure_summary(document.dialogue)
        try:
            from pykotor.resource.formats.gff import read_gff
            from pykotor.resource.generics.dlg import dismantle_dlg

            canonical_gff = dismantle_dlg(document.dialogue, _pykotor_game(document.game))
            _patch_dialogue_display_inactive(document.dialogue, canonical_gff, document.game)
            unknown_fields: dict[str, str] = {}
            if document.source_bytes:
                payload, unknown_fields = _preserve_imported_dialogue_fields(document, canonical_gff)
            else:
                payload = _quiet_bytes_gff(canonical_gff)
            written_gff = read_gff(payload)
            _verify_dialogue_display_inactive(document.dialogue, written_gff, document.game)
            readback = _quiet_read_dlg(payload)
            _hydrate_dialogue_display_inactive(readback, written_gff)
            after = dialogue_structure_summary(readback)
            if before != after:
                raise ValueError(f"DLG structural readback changed the graph: before={before}, after={after}")
        except Exception as exc:
            diagnostics.append(
                StudioDiagnostic(
                    "blocking",
                    "dialogue.write_readback_failed",
                    str(exc).strip() or exc.__class__.__name__,
                    document.resref,
                    fix_hint="Correct broken graph links or invalid node properties before saving.",
                )
            )
            return b"", tuple(diagnostics)
        diagnostics.append(
            StudioDiagnostic(
                "info",
                "dialogue.structural_readback_passed",
                f"DLG graph readback passed ({after['nodes']} nodes, {after['links']} links). Retail KOTOR proof remains required.",
                document.resref,
            )
        )
        if unknown_fields:
            diagnostics.append(
                StudioDiagnostic(
                    "info",
                    "dialogue.unknown_fields_preserved",
                    f"Preserved {len(unknown_fields)} unknown imported GFF field(s) while applying topology-stable edits.",
                    document.resref,
                )
            )
        return payload, tuple(diagnostics)

    def save_dialogue(self, document: DialogueDocument, path: str | Path | None = None) -> tuple[Path, tuple[StudioDiagnostic, ...]]:
        target_value = str(path or document.source_path or "").strip()
        if not target_value:
            raise ValueError("Choose a DLG file path before saving this dialogue.")
        target = Path(target_value)
        payload, diagnostics = self.dialogue_bytes(document)
        if not payload:
            return target, diagnostics
        _atomic_write_bytes(target, payload)
        document.source_path = str(target)
        document.resref = normalise_script_resref(target.stem, fallback=document.resref)
        document.source_bytes = payload
        document.dirty = False
        return target, diagnostics

    def build(
        self,
        documents: Sequence[ScriptDocument | DialogueDocument],
        output_dir: str | Path,
        *,
        game: str | None = None,
        include_dirs: Sequence[str | Path] = (),
    ) -> NarrativeBuildResult:
        docs = tuple(documents or ())
        target_game = _game_key(game or (docs[0].game if docs else "K2"))
        diagnostics: list[StudioDiagnostic] = []
        resources: list[NarrativeResource] = []
        seen: dict[tuple[str, str], bytes] = {}

        for document in docs:
            if _game_key(document.game) != target_game:
                diagnostics.append(
                    StudioDiagnostic(
                        "blocking",
                        "narrative.mixed_target_games",
                        f"{document.display_name} targets {document.game}; this build targets {target_game}.",
                        document.resref,
                    )
                )
                continue
            if isinstance(document, ScriptDocument):
                source_resource = NarrativeResource(
                    document.resref,
                    "nss",
                    document.source.encode("utf-8"),
                    document.document_id,
                )
                compile_result = self.compile_script(document, include_dirs=include_dirs)
                diagnostics.extend(compile_result.diagnostics)
                candidates = [source_resource]
                if compile_result.ok:
                    candidates.append(
                        NarrativeResource(
                            document.resref,
                            "ncs",
                            compile_result.ncs_bytes,
                            document.document_id,
                        )
                    )
            else:
                payload, rows = self.dialogue_bytes(document)
                diagnostics.extend(rows)
                candidates = (
                    [NarrativeResource(document.resref, "dlg", payload, document.document_id)] if payload else []
                )
            for resource in candidates:
                key = (resource.resref, resource.restype.lower())
                prior = seen.get(key)
                if prior is not None and prior != resource.data:
                    diagnostics.append(
                        StudioDiagnostic(
                            "blocking",
                            "narrative.resource_collision",
                            f"Two documents produce different bytes for {resource.filename}.",
                            resource.resref,
                        )
                    )
                    continue
                if prior is None:
                    seen[key] = resource.data
                    resources.append(resource)

        output = Path(output_dir)
        if not docs:
            diagnostics.append(
                StudioDiagnostic("blocking", "narrative.no_documents", "There are no open documents to build.")
            )
        if any(row.blocking for row in diagnostics):
            return NarrativeBuildResult(str(output), target_game, tuple(resources), tuple(diagnostics))

        ownership_diagnostic = _narrative_output_ownership_diagnostic(output)
        if ownership_diagnostic is not None:
            diagnostics.append(ownership_diagnostic)
            return NarrativeBuildResult(str(output), target_game, tuple(resources), tuple(diagnostics))

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=str(output.parent)))
        final_manifest = output / "narrative-build.json"
        try:
            staged_resources = staging / "resources"
            staged_resources.mkdir(parents=True, exist_ok=True)
            for resource in resources:
                (staged_resources / resource.filename).write_bytes(resource.data)
            manifest = {
                "file_type": "GhostStudioNarrativeBuild",
                "schema_version": 1,
                "created_at": _utc_now(),
                "game": target_game,
                "engine_proof": "not_recorded",
                "resources": [
                    {
                        "resref": row.resref,
                        "restype": row.restype.upper(),
                        "filename": row.filename,
                        "sha256": row.sha256,
                        "byte_count": len(row.data),
                        "source_document_id": row.source_document_id,
                    }
                    for row in resources
                ],
            }
            manifest_path = staging / "narrative-build.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            committed, promotion_diagnostics = _promote_build_directory(staging, output)
            diagnostics.extend(promotion_diagnostics)
        except Exception as staging_error:
            committed = False
            diagnostics.append(
                StudioDiagnostic(
                    "blocking",
                    "narrative.build_staging_failed",
                    f"Narrative build staging failed before the live output changed: {staging_error}",
                    fix_hint="Check destination permissions and available disk space, then build again.",
                )
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return NarrativeBuildResult(
            str(output),
            target_game,
            tuple(resources),
            tuple(diagnostics),
            str(final_manifest) if committed else "",
            committed,
        )


__all__ = [
    "DialogueDocument",
    "NarrativeBuildResult",
    "NarrativeResource",
    "ScriptCompileResult",
    "ScriptDocument",
    "ScriptingStudioService",
    "StudioDiagnostic",
    "dialogue_node_text",
    "dialogue_structure_summary",
    "normalise_script_resref",
    "set_dialogue_node_text",
]
