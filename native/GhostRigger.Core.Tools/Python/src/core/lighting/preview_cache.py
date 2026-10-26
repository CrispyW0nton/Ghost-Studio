"""Small in-memory cache for live lightmap previews."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from hashlib import sha1
import json
from typing import Any


class LightmapPreviewCache:
    def __init__(self, max_entries: int = 12) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, object] = OrderedDict()

    def make_key(self, mesh: object, settings: object, lights: list[object] | tuple[object, ...]) -> str:
        payload = {
            "mesh": getattr(mesh, "name", "") or id(mesh),
            "uv": getattr(settings, "selected_uv_channel", 1),
            "preview_resolution": getattr(settings, "preview_resolution", 128),
            "settings": self._settings_dict(settings),
            "lights": [
                {
                    "name": getattr(light, "name", ""),
                    "type": getattr(light, "type", getattr(light, "light_kind", "point")),
                    "position": tuple(getattr(light, "position", (0.0, 0.0, 0.0))),
                    "direction": tuple(getattr(light, "direction", (0.0, -1.0, -1.0))),
                    "color": tuple(getattr(light, "color", getattr(light, "light_color", (1.0, 1.0, 1.0)))),
                    "intensity": getattr(light, "intensity", getattr(light, "light_multiplier", 1.0)),
                    "radius": getattr(light, "radius", getattr(light, "light_radius", 5.0)),
                    "enabled": getattr(light, "enabled", getattr(light, "light_enabled", True)),
                    # Both change the bake (ambient-only lights skip N.L; shadow flags gate occlusion), so a
                    # toggled light must not reuse a cached preview.
                    "ambient_only": getattr(light, "ambient_only", getattr(light, "light_ambient_only", False)),
                    "casts_shadows": getattr(light, "casts_shadows", getattr(light, "light_shadow", True)),
                }
                for light in lights
            ],
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return sha1(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> object | None:
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: object) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def _settings_dict(self, settings: object) -> dict[str, Any]:
        try:
            data = asdict(settings)
        except Exception:
            data = dict(getattr(settings, "__dict__", {}) or {})
        data.pop("warnings", None)
        return data


__all__ = ("LightmapPreviewCache",)
