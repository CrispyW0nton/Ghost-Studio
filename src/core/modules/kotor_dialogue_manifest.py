"""Lossless, engine-neutral manifests for Odyssey DLG conversation graphs.

The serializer deliberately preserves KOTOR conditions and scripts as data.
It never executes arbitrary NWScript; a target runtime must provide an
allowlisted evaluator/action adapter for the operations it understands.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Mapping

SCHEMA = "ghoststudio.kotor-dialogue-manifest.v2"
TLKLookup = Callable[[int], str]


def _resref(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() == "none" else text


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _floating(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _localized_text(value: Any, tlk_lookup: TLKLookup | None) -> tuple[str, int]:
    try:
        stringref = int(getattr(value, "stringref", -1))
    except (TypeError, ValueError):
        stringref = -1
    if stringref >= 0 and callable(tlk_lookup):
        try:
            text = str(tlk_lookup(stringref) or "")
        except Exception:
            text = ""
        if text:
            return text, stringref
    substrings = dict(getattr(value, "_substrings_internal", {}) or {})
    if 0 in substrings:
        return str(substrings[0] or ""), stringref
    if substrings:
        return str(next(iter(substrings.values())) or ""), stringref
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            from pykotor.common.language import Gender, Language

            text = getter(Language.ENGLISH, Gender.MALE, use_fallback=True)
            if text:
                return str(text), stringref
        except Exception:
            pass
    return "", stringref


def _parameters(owner: Any, prefix: str) -> list[int | str]:
    values: list[int | str] = [
        _integer(getattr(owner, f"{prefix}_param{index}", 0), 0)
        for index in range(1, 6)
    ]
    values.append(str(getattr(owner, f"{prefix}_param6", "") or ""))
    return values if any(values[:5]) or values[5] else []


def _actions(node: Any) -> list[dict[str, Any]]:
    rows = []
    for slot in ("script1", "script2"):
        resref = _resref(getattr(node, slot, ""))
        if resref:
            rows.append(
                {
                    "slot": slot,
                    "resref": resref,
                    "parameters": _parameters(node, slot),
                }
            )
    return rows


def _conditions(link: Any) -> list[dict[str, Any]]:
    rows = []
    for slot in ("active1", "active2"):
        resref = _resref(getattr(link, slot, ""))
        if resref:
            rows.append(
                {
                    "slot": slot,
                    "resref": resref,
                    "negated": bool(getattr(link, f"{slot}_not", False)),
                    "parameters": _parameters(link, slot),
                }
            )
    return rows


def _node_kind(node: Any) -> str:
    return "entry" if node.__class__.__name__ == "DLGEntry" else "reply"


def _node_id(node: Any, counts: dict[str, int], used: set[str]) -> str:
    kind = _node_kind(node)
    index = _integer(getattr(node, "list_index", -1), -1)
    base = f"{kind}:{index}" if index >= 0 else f"{kind}:runtime:{counts[kind]}"
    counts[kind] += 1
    result = base
    suffix = 2
    while result in used:
        result = f"{base}:{suffix}"
        suffix += 1
    used.add(result)
    return result


def _node_row(
    node: Any,
    node_id: str,
    *,
    tlk_lookup: TLKLookup | None,
    voice_assets: Mapping[str, str],
) -> dict[str, Any]:
    text, stringref = _localized_text(getattr(node, "text", None), tlk_lookup)
    voice_resref = _resref(getattr(node, "vo_resref", ""))
    sound_resref = _resref(getattr(node, "sound", ""))
    # Retail alien/droid dialogue often names a missing per-line streamed VO
    # while providing the actually playable generic Sound ResRef. Prefer the
    # per-line performance only when it resolved; otherwise retain KOTOR's
    # canonical generic vocalization instead of silently dropping audio.
    voice_asset_resref = (
        voice_resref
        if voice_resref and voice_resref.casefold() in voice_assets
        else sound_resref
        if sound_resref and sound_resref.casefold() in voice_assets
        else voice_resref or sound_resref
    )
    animations = [
        {
            "participant": str(getattr(animation, "participant", "") or ""),
            "animation_id": _integer(getattr(animation, "animation_id", 0), 0),
        }
        for animation in tuple(getattr(node, "animations", ()) or ())
    ]
    return {
        "id": node_id,
        "kind": _node_kind(node),
        "list_index": _integer(getattr(node, "list_index", -1), -1),
        "speaker": str(getattr(node, "speaker", "") or ""),
        "listener": str(getattr(node, "listener", "") or ""),
        "text": text,
        "stringref": stringref,
        "sound_resref": sound_resref,
        "voice_resref": voice_resref,
        "voice_asset_resref": voice_asset_resref,
        "voice_asset": str(voice_assets.get(voice_asset_resref.casefold(), "")) if voice_asset_resref else "",
        "delay_ms": _integer(getattr(node, "delay", -1), -1),
        "wait_flags": _integer(getattr(node, "wait_flags", 0), 0),
        "unskippable": bool(getattr(node, "unskippable", False)),
        "camera": {
            "angle": _integer(getattr(node, "camera_angle", 0), 0),
            "id": _integer(getattr(node, "camera_id", -1), -1),
            "animation": _integer(getattr(node, "camera_anim", -1), -1),
            "fov": _floating(getattr(node, "camera_fov", 0.0), 0.0),
            "height_offset": _floating(getattr(node, "camera_height", 0.0), 0.0),
            "target_height_offset": _floating(getattr(node, "target_height", 0.0), 0.0),
            "effect": _integer(getattr(node, "camera_effect", -1), -1),
        },
        "fade": {
            "type": _integer(getattr(node, "fade_type", 0), 0),
            "delay": _floating(getattr(node, "fade_delay", 0.0), 0.0),
            "length": _floating(getattr(node, "fade_length", 0.0), 0.0),
            "color": [
                _floating(getattr(getattr(node, "fade_color", None), channel, 0.0), 0.0)
                for channel in ("r", "g", "b")
            ],
        },
        "animations": animations,
        "actions": _actions(node),
        "quest": {
            "tag": str(getattr(node, "quest", "") or ""),
            "entry": _integer(getattr(node, "quest_entry", 0), 0),
        },
        "plot": {
            "index": _integer(getattr(node, "plot_index", -1), -1),
            "xp_percentage": _floating(getattr(node, "plot_xp_percentage", 0.0), 0.0),
        },
        "emotion_id": _integer(getattr(node, "emotion_id", 0), 0),
        "facial_id": _integer(getattr(node, "facial_id", 0), 0),
    }


def build_kotor_dialogue_manifest(
    dialogue: Any,
    *,
    game: str,
    conversation_resref: str,
    tlk_lookup: TLKLookup | None = None,
    voice_assets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Serialize every reachable DLG entry, reply, link, and control field."""

    assets = {str(key).casefold(): str(value) for key, value in (voice_assets or {}).items()}
    used_nodes: set[str] = set()
    used_links: set[str] = set()
    counts = {"entry": 0, "reply": 0}
    node_ids: dict[int, str] = {}
    link_ids: dict[int, str] = {}
    node_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    starters: list[str] = []
    pending: deque[tuple[Any, str, int]] = deque(
        (link, "starter", index)
        for index, link in enumerate(tuple(getattr(dialogue, "starters", ()) or ()))
    )
    expanded_nodes: set[int] = set()
    seen_links: set[int] = set()

    while pending:
        link, source_id, link_index = pending.popleft()
        if link is None or id(link) in seen_links:
            continue
        seen_links.add(id(link))
        link_base = f"{source_id}:link:{link_index}"
        link_id = link_base
        suffix = 2
        while link_id in used_links:
            link_id = f"{link_base}:{suffix}"
            suffix += 1
        used_links.add(link_id)
        link_ids[id(link)] = link_id
        if source_id == "starter":
            starters.append(link_id)

        node = getattr(link, "node", None)
        target_id = ""
        if node is not None:
            target_id = node_ids.get(id(node), "")
            if not target_id:
                target_id = _node_id(node, counts, used_nodes)
                node_ids[id(node)] = target_id
                node_rows.append(
                    _node_row(
                        node,
                        target_id,
                        tlk_lookup=tlk_lookup,
                        voice_assets=assets,
                    )
                )
        link_rows.append(
            {
                "id": link_id,
                "source": source_id,
                "target": target_id,
                "condition_logic": "OR" if bool(getattr(link, "logic", False)) else "AND",
                "conditions": _conditions(link),
                "display_inactive": bool(getattr(link, "display_inactive", False)),
                "is_child": bool(getattr(link, "is_child", False)),
                "comment": str(getattr(link, "comment", "") or ""),
            }
        )
        if node is None or id(node) in expanded_nodes:
            continue
        expanded_nodes.add(id(node))
        for child_index, child in enumerate(tuple(getattr(node, "links", ()) or ())):
            pending.append((child, target_id, child_index))

    missing_voice = sorted(
        {
            row["voice_asset_resref"]
            for row in node_rows
            if row["kind"] == "entry"
            and row["text"]
            and row["voice_asset_resref"]
            and not row["voice_asset"]
        },
        key=str.casefold,
    )
    return {
        "schema": SCHEMA,
        "source": {
            "game": str(game or "K1"),
            "conversation_resref": str(conversation_resref or "").casefold(),
            "format": "Odyssey DLG",
        },
        "conversation": {
            "skippable": bool(getattr(dialogue, "skippable", False)),
            "conversation_type": _integer(getattr(dialogue, "conversation_type", 0), 0),
            "computer_type": _integer(getattr(dialogue, "computer_type", 0), 0),
            "animated_cut": _integer(getattr(dialogue, "animated_cut", 0), 0),
            "ambient_track": _resref(getattr(dialogue, "ambient_track", "")),
            "camera_model": _resref(getattr(dialogue, "camera_model", "")),
            "on_end": _resref(getattr(dialogue, "on_end", "")),
            "on_abort": _resref(getattr(dialogue, "on_abort", "")),
            "unequip_hands": bool(getattr(dialogue, "unequip_hands", False)),
            "unequip_items": bool(getattr(dialogue, "unequip_items", False)),
        },
        "starters": starters,
        "nodes": node_rows,
        "links": link_rows,
        "validation": {
            "node_count": len(node_rows),
            "entry_count": sum(row["kind"] == "entry" for row in node_rows),
            "reply_count": sum(row["kind"] == "reply" for row in node_rows),
            "link_count": len(link_rows),
            "voiced_entry_count": sum(
                row["kind"] == "entry"
                and bool(row["text"])
                and bool(row["voice_asset_resref"])
                for row in node_rows
            ),
            "missing_voice_asset_resrefs": missing_voice,
            "broken_link_ids": [row["id"] for row in link_rows if not row["target"]],
        },
    }
