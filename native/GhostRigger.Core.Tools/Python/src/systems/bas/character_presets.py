"""Installed-game character presets for the Body Attachment System.

The preset catalog is derived from the effective UTC, UTI, and 2DA resources
in each configured KOTOR installation.  It intentionally stores only logical
resource references: loading, preview composition, and export remain owned by
the existing BAS workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

_GAMES = ("K1", "K2")
_BAS_EQUIPMENT_SLOTS = {
    "HEAD": "mask",
    "LEFT_HAND": "left_weapon",
    "RIGHT_HAND": "right_weapon",
    "BELT": "belt",
}


def _clean(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "" if text in {"", "****", "none", "null"} else text


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class BasCharacterPreset:
    """One effective installed UTC resolved to BAS model layers."""

    game: str
    utc_resref: str
    display_name: str
    tag: str
    appearance_id: int
    appearance_label: str
    modeltype: str
    body_resref: str
    body_game: str = ""
    body_texture: str = ""
    head_resref: str = ""
    head_game: str = ""
    head_texture: str = ""
    slot_models: tuple[tuple[str, str], ...] = ()
    slot_games: tuple[tuple[str, str], ...] = ()
    equipment_items: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.game}:{self.utc_resref}"

    @property
    def composition_mode(self) -> str:
        return "headless_body" if self.modeltype == "B" else "full_body"

    @property
    def loadable(self) -> bool:
        return bool(self.body_resref and (self.modeltype != "B" or self.head_resref))

    @property
    def slot_model_map(self) -> dict[str, str]:
        return dict(self.slot_models)

    @property
    def slot_game_map(self) -> dict[str, str]:
        return dict(self.slot_games)

    @property
    def equipment_item_map(self) -> dict[str, str]:
        return dict(self.equipment_items)

    @property
    def label(self) -> str:
        name = self.display_name or self.tag or self.utc_resref
        return f"{name} [{self.game}] - {self.utc_resref}"

    @property
    def summary(self) -> str:
        def source_suffix(source_game: str) -> str:
            source = str(source_game or self.game).upper()
            return f" [{source} asset]" if source and source != self.game else ""

        layers = [f"Body: {self.body_resref or 'unresolved'}{source_suffix(self.body_game)}"]
        if self.head_resref:
            layers.append(f"Head: {self.head_resref}{source_suffix(self.head_game)}")
        slot_games = self.slot_game_map
        for slot, model in self.slot_models:
            layers.append(
                f"{slot.replace('_', ' ').title()}: {model}"
                f"{source_suffix(slot_games.get(slot, ''))}"
            )
        return " · ".join(layers)


@dataclass(frozen=True)
class BasCharacterPresetCatalog:
    """All resolved character presets from the configured game installs."""

    presets: tuple[BasCharacterPreset, ...] = ()
    errors: tuple[str, ...] = ()
    counts_by_game: Mapping[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.presets

    def for_game(self, game: str = "") -> tuple[BasCharacterPreset, ...]:
        tag = str(game or "").strip().upper()
        if tag not in _GAMES:
            return self.presets
        return tuple(preset for preset in self.presets if preset.game == tag)

    def get(self, game: str, utc_resref: str) -> BasCharacterPreset | None:
        key = f"{str(game or '').strip().upper()}:{_clean(utc_resref)}"
        return next((preset for preset in self.presets if preset.key == key), None)


def _read_twoda(manager: Any, name: str, game: str):
    try:
        from src.core.assets.resource_manager import RES_2DA
        from src.core.templates.twoda import TwoDA

        payload = manager.get_strict(name, RES_2DA, game)
        return TwoDA.from_bytes(payload, name) if payload else None
    except Exception:
        log.debug("BAS character presets: could not read %s.2da for %s", name, game, exc_info=True)
        return None


def _table_row(table: Any, index: int):
    if table is None or index < 0:
        return None
    try:
        return table[index] if index < len(table) else None
    except Exception:
        return None


def _game_install(manager: Any, game: str) -> Any:
    getter = getattr(manager, "get_k1" if game == "K1" else "get_k2", None)
    try:
        return getter() if callable(getter) else None
    except Exception:
        return None


def _game_resrefs(manager: Any, game: str, resource_type: int) -> tuple[str, ...]:
    install = _game_install(manager, game)
    if install is None or not hasattr(install, "list_resrefs"):
        return ()
    try:
        return tuple(
            sorted(
                {_clean(value) for value in install.list_resrefs(resource_type) if _clean(value)}
            )
        )
    except Exception:
        log.debug("BAS character presets: could not enumerate %s resources", game, exc_info=True)
        return ()


def _load_tlk(manager: Any, game: str):
    """Read one installation's dialog.tlk for user-facing UTC names."""

    install = _game_install(manager, game)
    game_dir = Path(str(getattr(install, "game_dir", "") or ""))
    if not game_dir.is_dir():
        return None
    try:
        tlk_path = next(
            (
                path
                for path in game_dir.iterdir()
                if path.is_file() and path.name.casefold() == "dialog.tlk"
            ),
            None,
        )
        if tlk_path is None:
            return None
        from pykotor.resource.formats.tlk import read_tlk

        return read_tlk(tlk_path)
    except Exception:
        log.debug("BAS character presets: dialog.tlk read failed for %s", game, exc_info=True)
        return None


def _localized_name(value: Any, tlk: Any, fallback: str) -> str:
    stringref = _safe_int(getattr(value, "stringref", -1), -1)
    if stringref >= 0 and tlk is not None:
        try:
            entry = tlk.get(stringref)
            text = str(getattr(entry, "text", "") or "").strip()
            if text:
                return text.replace("\r", " ").replace("\n", " ")
        except Exception:
            pass
    try:
        substrings = dict(value.to_dict().get("substrings", {}))
        text = next((str(item).strip() for item in substrings.values() if str(item).strip()), "")
        if text:
            return text.replace("\r", " ").replace("\n", " ")
    except Exception:
        pass
    return fallback


def _strict_texture_exists(manager: Any, texture: str, game: str) -> bool:
    name = _clean(texture)
    if not name:
        return False
    try:
        from src.core.assets.resource_manager import RES_TGA, RES_TPC

        return bool(
            manager.get_strict(name, RES_TPC, game) or manager.get_strict(name, RES_TGA, game)
        )
    except Exception:
        return False


def _texture_with_variation(
    manager: Any,
    prefix: str,
    variation: int,
    game: str,
) -> str:
    clean_prefix = _clean(prefix)
    if not clean_prefix:
        return ""
    suffix = max(1, int(variation or 1))
    candidate = f"{clean_prefix}{suffix:02d}"
    if suffix != 1 and not _strict_texture_exists(manager, candidate, game):
        candidate = f"{clean_prefix}01"
    return candidate


def _read_uti(manager: Any, resref: str, game: str):
    name = _clean(resref)
    if not name:
        return None
    try:
        from pykotor.resource.generics.uti import construct_uti, read_uti
        from src.core.assets.resource_manager import RES_UTI

        payload = manager.get_strict(name, RES_UTI, game)
        if not payload:
            return None
        try:
            return read_uti(payload)
        except Exception:
            # A small number of retail templates contain an UNSET language ID
            # in an unrelated localized string. The engine still consumes the
            # scalar UTI fields, so retry with an instance-local tolerant
            # locstring reader rather than mutating PyKotor globally.
            from types import MethodType

            from pykotor.common.language import Language, LocalizedString
            from pykotor.resource.formats.gff.io_gff import GFFBinaryReader

            def read_tolerant_locstring(reader):
                locstring = LocalizedString.from_invalid()
                reader.skip(4)
                locstring.stringref = reader.read_uint32(max_neg1=True)
                for _index in range(reader.read_uint32()):
                    string_id = reader.read_uint32()
                    language, gender = LocalizedString.substring_pair(string_id)
                    length = reader.read_uint32()
                    encoding = "cp1252" if language == Language.UNSET else language.get_encoding()
                    locstring.set_data(
                        language,
                        gender,
                        reader.read_string(length, encoding=encoding),
                    )
                return locstring

            gff_reader = GFFBinaryReader(payload)
            gff_reader._reader.read_locstring = MethodType(  # noqa: SLF001
                read_tolerant_locstring,
                gff_reader._reader,  # noqa: SLF001
            )
            return construct_uti(gff_reader.load())
    except Exception:
        log.debug("BAS character presets: could not read %s:%s.uti", game, name, exc_info=True)
        return None


def _game_candidates(preferred_game: str) -> tuple[str, str]:
    preferred = str(preferred_game or "K1").strip().upper()
    preferred = preferred if preferred in _GAMES else "K1"
    return preferred, "K1" if preferred == "K2" else "K2"


def _read_uti_with_game(manager: Any, resref: str, game: str) -> tuple[Any, str]:
    for source_game in _game_candidates(game):
        uti = _read_uti(manager, resref, source_game)
        if uti is not None:
            return uti, source_game
    return None, ""


def _model_source_game(
    resref: str,
    preferred_game: str,
    models_by_game: Mapping[str, set[str]],
) -> str:
    name = _clean(resref)
    if not name:
        return ""
    if not any(models_by_game.values()):
        return _game_candidates(preferred_game)[0]
    for source_game in _game_candidates(preferred_game):
        if name in models_by_game.get(source_game, set()):
            return source_game
    return ""


def _appearance_layers(
    manager: Any,
    *,
    game: str,
    utc: Any,
    appearance: Any,
    heads: Any,
    baseitems_by_game: Mapping[str, Any],
    models_by_game: Mapping[str, set[str]],
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    appearance_id = _safe_int(getattr(utc, "appearance_id", -1), -1)
    row = _table_row(appearance, appearance_id)
    if row is None:
        return {
            "appearance_label": "",
            "modeltype": "",
            "body_resref": "",
            "body_texture": "",
            "head_resref": "",
            "head_texture": "",
        }, [f"appearance.2da row {appearance_id} is unavailable"]

    modeltype = str(row.get("modeltype", "") or "").strip().upper()
    appearance_label = str(row.get("label", "") or "").strip()
    body_resref = ""
    body_texture = ""
    head_resref = ""
    head_texture = ""

    if modeltype == "B":
        bodyvar = "a"
        texture_variation = 1
        try:
            from pykotor.common.misc import EquipmentSlot

            armor_item = getattr(utc, "equipment", {}).get(EquipmentSlot.ARMOR)
        except Exception:
            armor_item = None
        armor_resref = _clean(getattr(armor_item, "resref", ""))
        armor_uti, armor_game = (
            _read_uti_with_game(manager, armor_resref, game) if armor_resref else (None, "")
        )
        if armor_uti is not None:
            baseitems = baseitems_by_game.get(armor_game)
            base_row = _table_row(baseitems, _safe_int(getattr(armor_uti, "base_item", -1), -1))
            resolved_bodyvar = _clean(base_row.get("bodyvar", "")) if base_row is not None else ""
            if resolved_bodyvar:
                bodyvar = resolved_bodyvar
            texture_variation = max(1, _safe_int(getattr(armor_uti, "texture_variation", 1), 1))
        body_resref = _clean(row.get(f"model{bodyvar}", "")) or _clean(row.get("modela", ""))
        evil_column = f"tex{bodyvar}evil"
        alignment = _safe_int(getattr(utc, "alignment", 50), 50)
        texture_prefix = _clean(row.get(evil_column, "")) if alignment <= 25 else ""
        texture_prefix = texture_prefix or _clean(row.get(f"tex{bodyvar}", ""))
        body_texture_game = _model_source_game(body_resref, game, models_by_game) or game
        body_texture = _texture_with_variation(
            manager,
            texture_prefix,
            texture_variation,
            body_texture_game,
        )

        head_index = _safe_int(row.get("normalhead", -1), -1)
        head_row = _table_row(heads, head_index)
        if head_row is not None:
            head_resref = _clean(head_row.get("head", ""))
            alignment = _safe_int(getattr(utc, "alignment", 50), 50)
            texture_column = ""
            if alignment < 10:
                texture_column = "headtexvvve"
            elif alignment < 20:
                texture_column = "headtexvve"
            elif alignment < 30:
                texture_column = "headtexve"
            elif alignment < 40:
                texture_column = "headtexe"
            elif game == "K2" and "alttexture" in [str(value).lower() for value in heads.columns]:
                texture_column = "alttexture"
            head_texture = _clean(head_row.get(texture_column, "")) if texture_column else ""
            head_texture_game = _model_source_game(head_resref, game, models_by_game) or game
            if head_texture and not _strict_texture_exists(
                manager,
                head_texture,
                head_texture_game,
            ):
                head_texture = ""
    else:
        body_resref = _clean(row.get("race", "")) or _clean(row.get("modela", ""))
        body_texture = _clean(row.get("racetex", ""))

    if modeltype == "B" and not head_resref:
        # Some retail rows (notably K2's Toydarian row) are marked B even
        # though every model column and race point at one self-contained
        # creature model. Treat those as full-body instead of demanding a
        # detachable humanoid head.
        race_resref = _clean(row.get("race", ""))
        if race_resref and race_resref == body_resref:
            modeltype = "F"

    if not body_resref:
        warnings.append(f"appearance.2da row {appearance_id} has no usable body model")
    if modeltype == "B" and not head_resref:
        warnings.append("body-part appearance has no detachable heads.2da model")
    return {
        "appearance_label": appearance_label,
        "modeltype": modeltype,
        "body_resref": body_resref,
        "body_texture": body_texture,
        "head_resref": head_resref,
        "head_texture": head_texture,
    }, warnings


def _variation_model(default_model: str, variation: int) -> str:
    model = _clean(default_model)
    if not model:
        return ""
    suffix = max(1, int(variation or 1))
    if "001" in model:
        return model.replace("001", f"{suffix:03d}")
    return model


def _variation_model_candidates(default_model: str, variation: int) -> tuple[str, ...]:
    model = _clean(default_model)
    if not model:
        return ()
    suffix = max(1, int(variation or 1))
    candidates = [_variation_model(model, suffix)]
    stem, separator, trailing = model.rpartition("_")
    if separator and trailing.isdigit():
        candidates.append(f"{stem}_{suffix:03d}")
        candidates.append(f"{stem}_{suffix:0{len(trailing)}d}")
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _resolve_installed_model(
    candidates: tuple[str, ...],
    preferred_game: str,
    models_by_game: Mapping[str, set[str]],
) -> tuple[str, str]:
    if not candidates:
        return "", ""
    if not any(models_by_game.values()):
        return candidates[0], _game_candidates(preferred_game)[0]
    for source_game in _game_candidates(preferred_game):
        installed = models_by_game.get(source_game, set())
        for candidate in candidates:
            if candidate in installed:
                return candidate, source_game
    return candidates[0], ""


def _equipment_layers(
    manager: Any,
    *,
    game: str,
    utc: Any,
    baseitems_by_game: Mapping[str, Any],
    models_by_game: Mapping[str, set[str]],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    list[str],
]:
    slot_models: dict[str, str] = {}
    slot_games: dict[str, str] = {}
    item_rows: list[tuple[str, str]] = []
    warnings: list[str] = []
    equipment = getattr(utc, "equipment", {}) or {}
    for equipment_slot, item in equipment.items():
        slot_name = str(getattr(equipment_slot, "name", equipment_slot) or "").strip().upper()
        item_resref = _clean(getattr(item, "resref", ""))
        if not item_resref:
            continue
        item_rows.append((slot_name.lower(), item_resref))
        bas_slot = _BAS_EQUIPMENT_SLOTS.get(slot_name)
        if not bas_slot:
            continue
        uti, uti_game = _read_uti_with_game(manager, item_resref, game)
        if uti is None:
            warnings.append(f"{slot_name}: {item_resref}.uti is unavailable")
            continue
        variation = max(1, _safe_int(getattr(uti, "model_variation", 1), 1))
        if slot_name == "HEAD":
            candidates = (f"i_mask_{variation:03d}",)
        else:
            baseitems = baseitems_by_game.get(uti_game)
            base_row = _table_row(baseitems, _safe_int(getattr(uti, "base_item", -1), -1))
            default_model = base_row.get("defaultmodel", "") if base_row is not None else ""
            candidates = _variation_model_candidates(default_model, variation)
            if not candidates and slot_name == "BELT":
                candidates = (f"i_belt_{variation:03d}",)
        model, model_game = _resolve_installed_model(candidates, uti_game or game, models_by_game)
        if model and model_game:
            slot_models[bas_slot] = model
            slot_games[bas_slot] = model_game
        elif model:
            warnings.append(f"{slot_name}: canonical model {model} is not installed")

    return (
        tuple(
            (slot, slot_models[slot])
            for slot in ("mask", "left_weapon", "belt", "right_weapon")
            if slot in slot_models
        ),
        tuple(
            (slot, slot_games[slot])
            for slot in ("mask", "left_weapon", "belt", "right_weapon")
            if slot in slot_games
        ),
        tuple(sorted(item_rows)),
        warnings,
    )


def _resolve_preset(
    manager: Any,
    *,
    game: str,
    resref: str,
    utc_bytes: bytes,
    appearance: Any,
    heads: Any,
    baseitems_by_game: Mapping[str, Any],
    tlk: Any,
    models_by_game: Mapping[str, set[str]],
) -> BasCharacterPreset:
    from pykotor.resource.generics.utc import read_utc

    utc = read_utc(utc_bytes)
    tag = str(getattr(utc, "tag", "") or "").strip()
    display_name = _localized_name(getattr(utc, "first_name", None), tlk, tag or resref)
    layers, warnings = _appearance_layers(
        manager,
        game=game,
        utc=utc,
        appearance=appearance,
        heads=heads,
        baseitems_by_game=baseitems_by_game,
        models_by_game=models_by_game,
    )
    slot_models, slot_games, equipment_items, equipment_warnings = _equipment_layers(
        manager,
        game=game,
        utc=utc,
        baseitems_by_game=baseitems_by_game,
        models_by_game=models_by_game,
    )
    warnings.extend(equipment_warnings)
    body_resref = layers["body_resref"]
    head_resref = layers["head_resref"]
    body_game = _model_source_game(body_resref, game, models_by_game)
    head_game = _model_source_game(head_resref, game, models_by_game)
    if body_resref and not body_game:
        warnings.append(f"canonical body model {body_resref} is not installed")
        body_resref = ""
    if head_resref and not head_game:
        warnings.append(f"canonical head model {head_resref} is not installed")
        head_resref = ""
    return BasCharacterPreset(
        game=game,
        utc_resref=resref,
        display_name=display_name,
        tag=tag,
        appearance_id=_safe_int(getattr(utc, "appearance_id", -1), -1),
        appearance_label=layers["appearance_label"],
        modeltype=layers["modeltype"],
        body_resref=body_resref,
        body_game=body_game,
        body_texture=layers["body_texture"],
        head_resref=head_resref,
        head_game=head_game,
        head_texture=layers["head_texture"],
        slot_models=slot_models,
        slot_games=slot_games,
        equipment_items=equipment_items,
        warnings=tuple(warnings),
    )


def build_bas_character_preset_catalog(manager: Any) -> BasCharacterPresetCatalog:
    """Resolve every effective installed K1/K2 UTC into a BAS preset."""

    if manager is None:
        return BasCharacterPresetCatalog()
    try:
        from src.core.assets.resource_manager import RES_UTC
    except Exception as exc:
        return BasCharacterPresetCatalog(errors=(str(exc),))

    model_rows = list(getattr(manager, "list_models", lambda _game="all": [])("all") or ())
    models_by_game = {
        game: {
            _clean(name)
            for name, owner in model_rows
            if str(owner or "").upper() == game and _clean(name)
        }
        for game in _GAMES
    }
    presets: list[BasCharacterPreset] = []
    errors: list[str] = []
    counts: dict[str, int] = {}
    for game in _GAMES:
        utc_resrefs = _game_resrefs(manager, game, RES_UTC)
        if not utc_resrefs:
            continue
        appearance = _read_twoda(manager, "appearance", game)
        heads = _read_twoda(manager, "heads", game)
        baseitems = _read_twoda(manager, "baseitems", game)
        if appearance is None or baseitems is None:
            errors.append(f"{game}: appearance.2da or baseitems.2da is unavailable")
            continue
        tlk = _load_tlk(manager, game)
        other_game = "K1" if game == "K2" else "K2"
        baseitems_by_game = {
            game: baseitems,
            other_game: _read_twoda(manager, "baseitems", other_game),
        }
        game_count = 0
        for resref in utc_resrefs:
            try:
                payload = manager.get_strict(resref, RES_UTC, game)
                if not payload:
                    errors.append(f"{game}:{resref}.utc is unavailable")
                    continue
                preset = _resolve_preset(
                    manager,
                    game=game,
                    resref=resref,
                    utc_bytes=payload,
                    appearance=appearance,
                    heads=heads,
                    baseitems_by_game=baseitems_by_game,
                    tlk=tlk,
                    models_by_game=models_by_game,
                )
                if preset.loadable:
                    presets.append(preset)
                    game_count += 1
                else:
                    reason = (
                        "no complete body/head composition exists in either configured game"
                        if preset.body_resref
                        else "no body model exists in either configured game"
                    )
                    errors.append(f"{game}:{resref}.utc omitted: {reason}")
            except Exception as exc:
                errors.append(f"{game}:{resref}.utc: {exc}")
        counts[game] = game_count

    presets.sort(key=lambda value: (value.display_name.casefold(), value.game, value.utc_resref))
    return BasCharacterPresetCatalog(
        presets=tuple(presets),
        errors=tuple(errors),
        counts_by_game=counts,
    )


def apply_bas_texture_override(model: Any, texture: str) -> int:
    """Apply one UTC-resolved creature texture to the model's visible skin."""

    override = _clean(texture)
    if model is None or not override:
        return 0
    try:
        nodes = list(model.all_nodes())
    except Exception:
        nodes = list(getattr(model, "nodes", []) or [])
    visible = [
        node
        for node in nodes
        if list(getattr(node, "vertices", []) or []) and bool(getattr(node, "is_mesh", False))
    ]
    targets = [node for node in visible if bool(getattr(node, "is_skin", False))] or visible
    changed = 0
    for node in targets:
        current = _clean(getattr(node, "texture", ""))
        if not current:
            continue
        setattr(node, "texture", override)
        texture_names = list(getattr(node, "texture_names", []) or [])
        if texture_names:
            texture_names[0] = override
        else:
            texture_names = [override]
        setattr(node, "texture_names", texture_names)
        changed += 1
    if changed:
        setattr(model, "_gr_bas_texture_override", override)
    return changed


__all__ = [
    "BasCharacterPreset",
    "BasCharacterPresetCatalog",
    "apply_bas_texture_override",
    "build_bas_character_preset_catalog",
]
