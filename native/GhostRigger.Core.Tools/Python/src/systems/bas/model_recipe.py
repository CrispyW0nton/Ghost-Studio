"""Persistence helpers for Body Attachment System model recipes."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BAS_MODEL_RECIPE_SCHEMA = "ghostrigger.bas.model"
BAS_MODEL_RECIPE_VERSION = 1
BAS_SLOT_ORDER = (
    "body",
    "head",
    "mask",
    "goggles",
    "left_hand",
    "right_hand",
    "left_weapon",
    "belt",
    "right_weapon",
)
BAS_SOCKET_BY_SLOT = {
    "head": "headhook",
    "mask": "MaskHook",
    "goggles": "GoggleHook",
    "left_hand": "lhand",
    "right_hand": "rhand",
    "left_weapon": "lhand",
    "belt": "pelvis_g",
    "right_weapon": "rhand",
}
BAS_SLOT_LABELS = {
    "body": "BODY",
    "head": "HEAD",
    "mask": "MASK",
    "goggles": "GOGGLES",
    "left_hand": "L. HAND",
    "right_hand": "R. HAND",
    "left_weapon": "L. Weapon",
    "belt": "BELT",
    "right_weapon": "R. Wep",
}


def default_bas_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def safe_bas_recipe_stem(*parts: object) -> str:
    raw = "_".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
    raw = re.sub(r"[^a-z0-9_.-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_.-")
    return raw or "bas_model"


def bas_model_identity(model: Any, *, fallback: str = "") -> dict[str, str]:
    name = str(getattr(model, "name", "") or fallback or "").strip()
    resref = str(getattr(model, "_gr_source_resref", "") or name or fallback or "").strip()
    game = str(
        getattr(model, "_gr_source_game", "") or getattr(model, "game_version", "") or ""
    ).strip()
    if "." in game:
        game = game.rsplit(".", 1)[-1]
    return {
        "resref": resref,
        "name": name or resref,
        "game": game.upper(),
        "supermodel": str(getattr(model, "supermodel", "") or "").strip(),
    }


def build_bas_model_recipe(
    *,
    body_model: Any,
    attachment_models: dict[str, Any] | None = None,
    attachment_resrefs: dict[str, str] | None = None,
    attachment_transforms: dict[str, dict[str, Any]] | None = None,
    game: str = "",
    build_name: str = "",
    mode: str = "headless_body",
) -> dict[str, Any]:
    attachment_models = attachment_models or {}
    attachment_resrefs = attachment_resrefs or {}
    attachment_transforms = attachment_transforms or {}
    body = bas_model_identity(body_model, fallback="body")
    game_tag = str(game or body.get("game") or "").strip().upper()
    body["game"] = game_tag or body.get("game", "")

    layers: list[dict[str, Any]] = [
        {
            "slot": "body",
            "label": BAS_SLOT_LABELS["body"],
            "state": "base",
            "enabled": True,
            "socket": "",
            "resref": body["resref"],
            "model_name": body["name"],
            "game": body["game"],
        }
    ]
    mode_key = str(mode or "headless_body").strip().lower()
    if mode_key not in {"headless_body", "full_body"}:
        mode_key = "headless_body"

    for slot in BAS_SLOT_ORDER:
        if slot == "body":
            continue
        resref = str(attachment_resrefs.get(slot, "") or "").strip()
        model = attachment_models.get(slot)
        model_info = bas_model_identity(model, fallback=resref) if model is not None else {}
        state = (
            "socket"
            if slot in {"left_hand", "right_hand"} or (slot == "head" and mode_key == "full_body")
            else ("attached" if resref or model is not None else "empty")
        )
        layers.append(
            {
                "slot": slot,
                "label": BAS_SLOT_LABELS.get(slot, slot),
                "state": state,
                "enabled": state == "attached",
                "socket": BAS_SOCKET_BY_SLOT.get(slot, ""),
                "resref": resref or model_info.get("resref", ""),
                "model_name": model_info.get("name", ""),
                "game": model_info.get("game", "") or game_tag,
                "transform": normalize_bas_layer_transform(attachment_transforms.get(slot)),
            }
        )

    clean_build_name = str(build_name or "").strip()
    attachment_parts = [
        layer["resref"] for layer in layers if layer["state"] == "attached" and layer["resref"]
    ]
    recipe_stem = (
        safe_bas_recipe_stem(clean_build_name)
        if clean_build_name
        else safe_bas_recipe_stem(game_tag, body["resref"], *attachment_parts)
    )
    character_preset = getattr(body_model, "_gr_bas_character_preset", None)
    recipe = {
        "schema": BAS_MODEL_RECIPE_SCHEMA,
        "version": BAS_MODEL_RECIPE_VERSION,
        "saved_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "game": game_tag,
        "recipe_id": recipe_stem,
        "display_name": clean_build_name or f"{body['name']} BAS",
        "build_name": clean_build_name or recipe_stem,
        "mode": mode_key,
        "body": body,
        "layers": layers,
        "attachments": {
            slot: str(resref)
            for slot, resref in sorted(attachment_resrefs.items())
            if str(resref or "").strip()
        },
        "runtime": {
            "body_animation_owner": "body",
            "attachment_transform_mode": "socket_follower",
            "attachment_skinning": "isolated_from_body_palette",
            "body_mode": mode_key,
            "notes": "BAS JSON stores a lightweight recipe. Source MDL/MDX assets stay referenced by game/resref.",
        },
    }
    if isinstance(character_preset, dict) and character_preset.get("utc_resref"):
        recipe["character_preset"] = {
            "game": str(character_preset.get("game") or game_tag),
            "utc_resref": str(character_preset.get("utc_resref") or ""),
            "display_name": str(character_preset.get("display_name") or ""),
            "body_game": str(character_preset.get("body_game") or body.get("game") or game_tag),
            "head_game": str(character_preset.get("head_game") or ""),
            "slot_games": {
                str(slot): str(game)
                for slot, game in dict(character_preset.get("slot_games") or {}).items()
                if str(game or "").upper() in {"K1", "K2"}
            },
            "slot_models": {
                str(slot): str(resref)
                for slot, resref in dict(character_preset.get("slot_models") or {}).items()
                if str(resref or "").strip()
            },
            "nonvisual_slots": {
                str(slot): str(resref)
                for slot, resref in dict(character_preset.get("nonvisual_slots") or {}).items()
                if str(resref or "").strip()
            },
            "body_texture": str(character_preset.get("body_texture") or ""),
            "head_texture": str(character_preset.get("head_texture") or ""),
            "equipment_items": {
                str(slot): str(resref)
                for slot, resref in dict(character_preset.get("equipment_items") or {}).items()
                if str(resref or "").strip()
            },
        }
    return recipe


def save_bas_model_recipe(recipe: dict[str, Any], directory: Path | None = None) -> Path:
    target_dir = directory or default_bas_models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_bas_recipe_stem(
        recipe.get("recipe_id") or recipe.get("display_name") or "bas_model"
    )
    path = target_dir / f"{stem}.json"
    path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def normalize_bas_layer_transform(
    transform: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    transform = transform or {}

    def values(key: str, fallback: tuple[float, ...]) -> list[float]:
        raw = transform.get(key, fallback)
        try:
            result = [float(value) for value in list(raw)[: len(fallback)]]
        except Exception:
            result = []
        while len(result) < len(fallback):
            result.append(float(fallback[len(result)]))
        return result

    return {
        "position": values("position", (0.0, 0.0, 0.0)),
        "rotation": values("rotation", (0.0, 0.0, 0.0, 1.0)),
        "scale": values("scale", (1.0, 1.0, 1.0)),
    }


def is_bas_model_recipe(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema") == BAS_MODEL_RECIPE_SCHEMA


def load_bas_model_recipe(path: str | Path) -> dict[str, Any]:
    recipe_path = Path(path)
    data = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not is_bas_model_recipe(data):
        raise ValueError(f"{recipe_path.name} is not a BAS model recipe.")
    return data
