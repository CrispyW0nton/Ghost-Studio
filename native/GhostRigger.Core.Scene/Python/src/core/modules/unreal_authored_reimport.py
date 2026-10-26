"""Three-way reconciliation of source components and existing Map Studio edits."""
from __future__ import annotations
import copy
import json

_MISSING = object()


def merge_authored_source(baseline: dict, local: dict, incoming: dict, decisions=None):
    """Return a proposal and explicit conflicting property paths, without mutation.

    Lists of surface topology are atomic. Room identity is stable across order
    changes. Placements and lights belong to Ghost after the first import.
    """
    decisions = decisions or {}
    conflicts = []

    def clone(value):
        return _MISSING if value is _MISSING else copy.deepcopy(value)

    def merge(old, current, new, path):
        if current == old:
            return clone(new)
        if new == old or current == new:
            return clone(current)
        if all(isinstance(v, dict) for v in (old, current, new)):
            result = {}
            for key in sorted(old.keys() | current.keys() | new.keys()):
                value = merge(old.get(key, _MISSING), current.get(key, _MISSING), new.get(key, _MISSING), path + (key,))
                if value is not _MISSING:
                    result[key] = value
            return result
        key = json.dumps(path, separators=(",", ":"))
        conflicts.append(key)
        return clone(new if decisions.get(key) == "accept_source" else current)

    def rooms(payload):
        result = {}
        for room in payload["rooms"]:
            key = room["room_resref"]
            if key in result:
                raise ValueError("Duplicate component identity in authored source.")
            result[key] = room
        return result

    result = copy.deepcopy(incoming)
    old, current, new = rooms(baseline), rooms(local), rooms(incoming)
    merged_rooms = merge(old, current, new, ("rooms",))
    # Preserve native source order, followed by retained local objects.
    result["rooms"] = [merged_rooms[key] for key in dict.fromkeys([*new, *current]) if key in merged_rooms]
    for key in local.keys() | baseline.keys() | incoming.keys():
        if key == "rooms":
            continue
        if key in {"placements", "lights"}:
            result[key] = copy.deepcopy(local.get(key, incoming.get(key)))
            continue
        value = merge(baseline.get(key, _MISSING), local.get(key, _MISSING), incoming.get(key, _MISSING), (key,))
        if value is _MISSING:
            result.pop(key, None)
        else:
            result[key] = value
    result["game_tested"] = False
    return result, tuple(conflicts)
