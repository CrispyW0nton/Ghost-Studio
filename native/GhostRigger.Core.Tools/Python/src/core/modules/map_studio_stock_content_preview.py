"""Stock KOTOR content preview for Map Studio.

Loads real game geometry into the Map Studio viewport preview model:

* Stock LYT rooms (``project.rooms`` -> room MDL/MDX from the game install).
* Gameplay instance geometry (creature UTC, placeable UTP, door UTD template
  resrefs resolved through appearance.2da / placeables.2da / genericdoors.2da).

The output mirrors ``build_authored_module_preview_model``: one KotorModel
whose root children are per-room / per-instance group nodes positioned in
world space, with flattened world-baked mesh children.  Flattening bakes the
source model's nested dummy/trimesh transforms into group-local vertices so
the existing 2-level hover walker and renderer stay correct.

KOTOR contract: this preview is a read-only visualization of source game
resources.  It never mutates source module data and its meshes are tagged
``_gr_map_studio_stock_mesh`` so authoring/export flows can tell stock
reference geometry apart from authored geometry.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("map_studio_stock_preview")

# Standard Aurora resource-type ids (verified against module_save_pipeline and
# the proven GUI library-panel path).  NOTE: resource_manager.py's own RES_UTC
# (2023) and RES_UTD (2038) constants are wrong for chitin.key lookups — do
# not swap these for those.
RES_2DA = 2017
RES_MDL = 2002
RES_MDX = 3008
RES_UTI = 2025
RES_UTC = 2027
RES_DLG = 2029
RES_UTT = 2032
RES_UTP = 2044
RES_UTD = 2042
RES_UTM = 2051

_RESOURCE_TYPE_NAMES = {
    RES_2DA: "2da",
    RES_MDL: "mdl",
    RES_MDX: "mdx",
    RES_UTI: "uti",
    RES_UTC: "utc",
    RES_DLG: "dlg",
    RES_UTT: "utt",
    RES_UTP: "utp",
    RES_UTD: "utd",
    RES_UTM: "utm",
}


def _resource_type_name(value: object) -> str:
    """Return one stable extension for int or string resource-type values."""

    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        numeric = -1
    if numeric in _RESOURCE_TYPE_NAMES:
        return _RESOURCE_TYPE_NAMES[numeric]
    return str(value or "").strip().lower().lstrip(".")


def _import_model_data() -> Any:
    try:
        from src.core.geometry import model_data as md  # type: ignore

        return md
    except Exception:
        from core.geometry import model_data as md  # type: ignore

        return md


def _vec3(value: object) -> tuple[float, float, float]:
    try:
        seq = tuple(value)  # type: ignore[arg-type]
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def _read_gff_fields(data: bytes) -> Any | None:
    try:
        from src.formats.gff_reader import read_gff  # type: ignore
    except Exception:
        try:
            from formats.gff_reader import read_gff  # type: ignore
        except Exception:
            return None
    try:
        return read_gff(bytes(data))
    except Exception:
        return None


def _parse_2da(data: bytes, name: str) -> Any | None:
    try:
        from src.core.templates.twoda import TwoDA  # type: ignore
    except Exception:
        try:
            from core.templates.twoda import TwoDA  # type: ignore
        except Exception:
            return None
    try:
        return TwoDA.from_bytes(bytes(data), name=name)
    except Exception:
        return None


class TemplateModelResolver:
    """Resolve GIT template resrefs (UTC/UTP/UTD) to renderable MDL resrefs.

    Chains (KOTOR contract):

    * Creature:  UTC ``Appearance_Type`` -> appearance.2da row.  ``modeltype``
      'B' means body-part model (column ``modela``); anything else means
      full-body model (column ``race``, fallback ``modela``) plus the exact
      instance texture named by ``racetex`` when present.
    * Placeable: UTP ``Appearance`` -> placeables.2da ``modelname``.
    * Door:      UTD ``GenericType`` (fallback ``Appearance`` for K2) ->
      genericdoors.2da ``modelname``.
    * Store/merchant (UTM): no geometry by design — physical bodies come from
      the creature referencing the store.

    All failures resolve to ``""`` so callers keep their abstract-marker
    fallback and never crash a preview build.
    """

    def __init__(
        self,
        resource_manager: Any,
        game: str = "K1",
        *,
        template_resources: Any = (),
        placeable_rows: Any = (),
    ) -> None:
        self._manager = resource_manager
        self._game = str(game or "K1").upper()
        self._tables: dict[str, Any | None] = {}
        self._models: dict[tuple[str, str], str] = {}
        self._resource_overrides: dict[tuple[str, str], bytes] = {}
        for item in tuple(template_resources or ()):
            try:
                resref, restype, data = item
            except (TypeError, ValueError):
                continue
            key = (str(resref or "").strip().lower(), _resource_type_name(restype))
            if key[0] and key[1] and data:
                self._resource_overrides[key] = bytes(data)
        self._placeable_rows: dict[str, Any] = {}
        for row in tuple(placeable_rows or ()):
            if isinstance(row, dict):
                resref = str(row.get("resref") or row.get("template_resref") or "").strip().lower()
            else:
                resref = str(
                    getattr(row, "resref", "") or getattr(row, "template_resref", "") or ""
                ).strip().lower()
            if resref:
                self._placeable_rows[resref] = row

    def _resource_bytes(self, resref: str, res_type: object) -> bytes | None:
        key = (str(resref or "").strip().lower(), _resource_type_name(res_type))
        override = self._resource_overrides.get(key)
        if override is not None:
            return override
        manager = self._manager
        if manager is None:
            return None
        try:
            getter = getattr(manager, "get_strict", None)
            data = (
                getter(key[0], res_type, self._game)
                if callable(getter)
                else manager.get(key[0], res_type, self._game)
            )
        except Exception:
            return None
        return bytes(data) if data else None

    def _table(self, name: str) -> Any | None:
        key = name.lower()
        if key in self._tables:
            return self._tables[key]
        table = None
        raw = self._resource_bytes(key, RES_2DA)
        if raw:
            table = _parse_2da(raw, key)
            if table is None:
                log.warning("Could not parse %s.2da for %s", key, self._game)
        self._tables[key] = table
        return table

    def _template_bytes(self, resref: str, res_type: int) -> bytes | None:
        return self._resource_bytes(resref, res_type)

    def model_resource_bytes(self, model_resref: str) -> tuple[bytes, bytes] | None:
        """Return project/library MDL+MDX overrides for one resolved model."""

        mdl = self._resource_bytes(model_resref, RES_MDL)
        if not mdl:
            return None
        return (mdl, self._resource_bytes(model_resref, RES_MDX) or b"")

    def _placeable_row_appearance(self, utp_resref: str) -> int:
        row = self._placeable_rows.get(str(utp_resref or "").strip().lower())
        if row is None:
            return -1
        if isinstance(row, dict):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            value = row.get("appearance_id", metadata.get("appearance_id"))
        else:
            value = getattr(row, "appearance_id", None)
            metadata = getattr(row, "metadata", {})
            if value is None and isinstance(metadata, dict):
                value = metadata.get("appearance_id")
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _cached(self, kind: str, resref: str) -> str | None:
        return self._models.get((kind, str(resref).strip().lower()))

    def _store(self, kind: str, resref: str, model: str) -> str:
        clean = str(model or "").strip().lower()
        if clean in {"", "****", "null"}:
            clean = ""
        self._models[(kind, str(resref).strip().lower())] = clean
        return clean

    def creature_model(self, utc_resref: str) -> str:
        cached = self._cached("creature", utc_resref)
        if cached is not None:
            return cached
        raw = self._template_bytes(utc_resref, RES_UTC)
        if not raw:
            return self._store("creature", utc_resref, "")
        gff = _read_gff_fields(raw)
        if gff is None:
            return self._store("creature", utc_resref, "")
        try:
            row_index = int(gff.get("Appearance_Type", gff.get("Appearance", -1)))
        except Exception:
            row_index = -1
        table = self._table("appearance")
        if table is None or row_index < 0 or row_index >= len(table):
            return self._store("creature", utc_resref, "")
        modeltype = str(table.get(row_index, "modeltype", "") or "").strip().upper()
        if modeltype == "B":
            model = table.get(row_index, "modela", "")
        else:
            model = table.get(row_index, "race", "") or table.get(row_index, "modela", "")
        return self._store("creature", utc_resref, model)

    def creature_body_texture(self, utc_resref: str) -> str:
        """Resolve the retail full-body ``racetex`` instance override.

        K2's creature appearance path reads ``RaceTex`` directly for non-B
        model types.  It does not append the B-body ``01`` variation suffix.
        Body-part appearances remain empty here because their texture depends
        on equipped armor/baseitems data that this template-only resolver does
        not yet own.
        """

        cached = self._cached("creature_body_texture", utc_resref)
        if cached is not None:
            return cached
        raw = self._template_bytes(utc_resref, RES_UTC)
        gff = _read_gff_fields(raw) if raw else None
        if gff is None:
            return self._store("creature_body_texture", utc_resref, "")
        try:
            row_index = int(gff.get("Appearance_Type", gff.get("Appearance", -1)))
        except Exception:
            row_index = -1
        table = self._table("appearance")
        if table is None or row_index < 0 or row_index >= len(table):
            return self._store("creature_body_texture", utc_resref, "")
        modeltype = str(table.get(row_index, "modeltype", "") or "").strip().upper()
        if modeltype == "B":
            return self._store("creature_body_texture", utc_resref, "")
        return self._store(
            "creature_body_texture",
            utc_resref,
            table.get(row_index, "racetex", ""),
        )

    def creature_head_model(self, utc_resref: str) -> str:
        """Resolve the separate head MDL for body-part (modeltype B) creatures.

        The appearance.2da ``normalhead`` cell indexes heads.2da, whose
        ``head`` column names the head model that the engine attaches at the
        body's ``headhook``.  Full-body appearances resolve to ``""``.
        """

        cached = self._cached("creature_head", utc_resref)
        if cached is not None:
            return cached
        raw = self._template_bytes(utc_resref, RES_UTC)
        gff = _read_gff_fields(raw) if raw else None
        if gff is None:
            return self._store("creature_head", utc_resref, "")
        try:
            row_index = int(gff.get("Appearance_Type", gff.get("Appearance", -1)))
        except Exception:
            row_index = -1
        table = self._table("appearance")
        if table is None or row_index < 0 or row_index >= len(table):
            return self._store("creature_head", utc_resref, "")
        modeltype = str(table.get(row_index, "modeltype", "") or "").strip().upper()
        if modeltype != "B":
            return self._store("creature_head", utc_resref, "")
        try:
            head_row = int(str(table.get(row_index, "normalhead", "") or "").strip())
        except (TypeError, ValueError):
            return self._store("creature_head", utc_resref, "")
        heads = self._table("heads")
        if heads is None or head_row < 0 or head_row >= len(heads):
            return self._store("creature_head", utc_resref, "")
        return self._store("creature_head", utc_resref, heads.get(head_row, "head", ""))

    def head_model_for_placement_kind(self, kind: str, template_resref: str) -> str:
        clean_kind = str(kind or "").strip().lower()
        resref = str(template_resref or "").strip()
        if resref and clean_kind in {"creature", "utc"}:
            return self.creature_head_model(resref)
        return ""

    def body_texture_for_placement_kind(self, kind: str, template_resref: str) -> str:
        clean_kind = str(kind or "").strip().lower()
        resref = str(template_resref or "").strip()
        if resref and clean_kind in {"creature", "utc"}:
            return self.creature_body_texture(resref)
        return ""

    def placeable_model(self, utp_resref: str) -> str:
        cached = self._cached("placeable", utp_resref)
        if cached is not None:
            return cached
        raw = self._template_bytes(utp_resref, RES_UTP)
        gff = _read_gff_fields(raw) if raw else None
        try:
            row_index = int(gff.get("Appearance", -1)) if gff is not None else -1
        except Exception:
            row_index = -1
        if row_index < 0:
            # A just-authored library asset can be available to Map Studio
            # before its UTP is injected into the ResourceManager.  The row's
            # typed appearance id still follows the same placeables.2da chain.
            row_index = self._placeable_row_appearance(utp_resref)
        table = self._table("placeables")
        if table is None or row_index < 0 or row_index >= len(table):
            return self._store("placeable", utp_resref, "")
        return self._store("placeable", utp_resref, table.get(row_index, "modelname", ""))

    def door_model(self, utd_resref: str) -> str:
        cached = self._cached("door", utd_resref)
        if cached is not None:
            return cached
        raw = self._template_bytes(utd_resref, RES_UTD)
        if not raw:
            return self._store("door", utd_resref, "")
        gff = _read_gff_fields(raw)
        if gff is None:
            return self._store("door", utd_resref, "")
        try:
            row_index = int(gff.get("GenericType", gff.get("Appearance", -1)))
        except Exception:
            row_index = -1
        table = self._table("genericdoors")
        if table is None or row_index < 0 or row_index >= len(table):
            return self._store("door", utd_resref, "")
        return self._store("door", utd_resref, table.get(row_index, "modelname", ""))

    def weapon_damage_dice(self, uti_resref: str, strength_modifier: int = 0) -> Any | None:
        """Resolve an equipped item's baseitems.2da damage dice.

        Returns a ``MapStudioPIEDamageDice`` for a weapon (``numdice`` × d
        ``dietoroll`` from baseitems.2da, plus Strength for a melee weapon), or
        ``None`` for a non-weapon (armor/misc) or any resolution failure so the
        caller keeps its generic fallback. Fully defensive by contract.
        """

        raw = self._template_bytes(uti_resref, RES_UTI)
        if not raw:
            return None
        gff = _read_gff_fields(raw)
        if gff is None:
            return None
        try:
            base_item = int(gff.get("BaseItem", -1))
        except Exception:
            return None
        table = self._table("baseitems")
        if table is None or base_item < 0 or base_item >= len(table):
            return None
        numdice = str(table.get(base_item, "numdice", "") or "").strip()
        dietoroll = str(table.get(base_item, "dietoroll", "") or "").strip()
        if numdice in ("", "****") and dietoroll in ("", "****"):
            return None  # not a damage-dealing weapon (armor, plot item, etc.)
        ranged = str(table.get(base_item, "rangedweapon", "") or "").strip() not in ("", "****", "0")
        try:
            from .map_studio_pie_combat import derive_pie_weapon_damage_dice
        except Exception:
            return None
        return derive_pie_weapon_damage_dice(
            {"numdice": numdice, "dietoroll": dietoroll},
            strength_modifier=int(strength_modifier),
            ranged=ranged,
        )

    def ambient_music_resref(self, track: int) -> str:
        """Resolve an ``ambientmusic.2da`` row to its streaming music resref.

        KOTOR area music is script-selected by row index (``MusicBackgroundChangeDay``);
        this maps that literal row to the ``resource`` column so PIE can report
        (and, later, play) the area's ambient track. Returns ``""`` on any
        non-resolvable row. Fully defensive by contract.
        """

        try:
            row = int(track)
        except (TypeError, ValueError):
            return ""
        if row < 0:
            return ""
        table = self._table("ambientmusic")
        if table is None or row >= len(table):
            return ""
        return str(table.get(row, "resource", "") or "").strip()

    def armor_class_bonus(self, uti_resref: str) -> tuple[int, int] | None:
        """Resolve an equipped item's baseitems.2da armor class and max-Dex cap.

        Returns ``(base_ac, max_dex)`` for an AC-granting item (baseac > 0) — the
        armor bonus and the Dexterity cap it imposes — or ``None`` for a
        non-armor item or any failure, so the caller keeps uncapped Dex and no
        armor bonus. Fully defensive by contract.
        """

        raw = self._template_bytes(uti_resref, RES_UTI)
        if not raw:
            return None
        gff = _read_gff_fields(raw)
        if gff is None:
            return None
        try:
            base_item = int(gff.get("BaseItem", -1))
        except Exception:
            return None
        table = self._table("baseitems")
        if table is None or base_item < 0 or base_item >= len(table):
            return None
        baseac = str(table.get(base_item, "baseac", "") or "").strip()
        if baseac in ("", "****", "0"):
            return None
        try:
            base_ac = int(float(baseac))
        except (TypeError, ValueError):
            return None
        if base_ac <= 0:
            return None
        dexbonus = str(table.get(base_item, "dexbonus", "") or "").strip()
        try:
            max_dex = int(float(dexbonus)) if dexbonus not in ("", "****") else 99
        except (TypeError, ValueError):
            max_dex = 99
        return (base_ac, max_dex)

    def weapon_critical(self, uti_resref: str) -> tuple[int, int] | None:
        """Resolve an equipped weapon's baseitems.2da crit threat and multiplier.

        Returns ``(threat, multiplier)`` where ``threat`` is how many top d20
        rolls threaten (``critthreat``: 1 => only 20, 2 => 19-20) and
        ``multiplier`` is ``crithitmult``. Returns ``None`` for a non-weapon or
        any failure so the caller keeps the d20 baseline (threat 1, x2).
        """

        raw = self._template_bytes(uti_resref, RES_UTI)
        if not raw:
            return None
        gff = _read_gff_fields(raw)
        if gff is None:
            return None
        try:
            base_item = int(gff.get("BaseItem", -1))
        except Exception:
            return None
        table = self._table("baseitems")
        if table is None or base_item < 0 or base_item >= len(table):
            return None
        numdice = str(table.get(base_item, "numdice", "") or "").strip()
        dietoroll = str(table.get(base_item, "dietoroll", "") or "").strip()
        if numdice in ("", "****") and dietoroll in ("", "****"):
            return None  # not a damage-dealing weapon
        threat_raw = str(table.get(base_item, "critthreat", "") or "").strip()
        mult_raw = str(table.get(base_item, "crithitmult", "") or "").strip()
        try:
            threat = int(float(threat_raw)) if threat_raw not in ("", "****") else 1
        except (TypeError, ValueError):
            threat = 1
        try:
            multiplier = int(float(mult_raw)) if mult_raw not in ("", "****") else 2
        except (TypeError, ValueError):
            multiplier = 2
        return (max(1, threat), max(1, multiplier))

    def weapon_damage_type(self, uti_resref: str) -> str | None:
        """Resolve an equipped weapon's KOTOR damage type from baseitems.2da.

        Returns the label for the weapon's `damageflags` (e.g. "Slashing",
        "Energy") via the authoritative nwscript `DAMAGE_TYPE_*` bit map, or
        ``None`` for a non-weapon / any failure.
        """

        raw = self._template_bytes(uti_resref, RES_UTI)
        if not raw:
            return None
        gff = _read_gff_fields(raw)
        if gff is None:
            return None
        try:
            base_item = int(gff.get("BaseItem", -1))
        except Exception:
            return None
        table = self._table("baseitems")
        if table is None or base_item < 0 or base_item >= len(table):
            return None
        numdice = str(table.get(base_item, "numdice", "") or "").strip()
        dietoroll = str(table.get(base_item, "dietoroll", "") or "").strip()
        if numdice in ("", "****") and dietoroll in ("", "****"):
            return None  # not a damage-dealing weapon
        flags_raw = str(table.get(base_item, "damageflags", "") or "").strip()
        if flags_raw in ("", "****"):
            return None
        try:
            from .map_studio_pie_combat import pie_damage_type_label
        except Exception:
            return None
        return pie_damage_type_label(flags_raw)

    def weapon_feat_category(self, uti_resref: str) -> str:
        """Classify an equipped weapon for Weapon Focus/Specialization feats.

        Returns "lightsaber" (baseitems.2da `powereditem`), "melee" (non-powered
        non-ranged), "blaster" (ranged), or "" for a non-weapon/failure. Only
        the unambiguous lightsaber/melee categories drive feat bonuses (a
        blaster's pistol-vs-rifle focus feat can't be split from these columns).
        """

        raw = self._template_bytes(uti_resref, RES_UTI)
        if not raw:
            return ""
        gff = _read_gff_fields(raw)
        if gff is None:
            return ""
        try:
            base_item = int(gff.get("BaseItem", -1))
        except Exception:
            return ""
        table = self._table("baseitems")
        if table is None or base_item < 0 or base_item >= len(table):
            return ""
        numdice = str(table.get(base_item, "numdice", "") or "").strip()
        if numdice in ("", "****"):
            return ""  # not a weapon
        powered = str(table.get(base_item, "powereditem", "") or "").strip()
        if powered not in ("", "****", "0"):
            return "lightsaber"
        ranged = str(table.get(base_item, "rangedweapon", "") or "").strip() not in ("", "****", "0")
        return "blaster" if ranged else "melee"

    def store_model(self, utm_resref: str) -> str:
        """UTM stores are non-spatial and have no geometry by design."""

        return ""

    def model_for_placement_kind(self, kind: str, template_resref: str) -> str:
        clean_kind = str(kind or "").strip().lower()
        resref = str(template_resref or "").strip()
        if not resref:
            return ""
        if clean_kind in {"creature", "utc"}:
            return self.creature_model(resref)
        if clean_kind in {"placeable", "utp"}:
            return self.placeable_model(resref)
        if clean_kind in {"door", "utd"}:
            return self.door_model(resref)
        return ""


def _quat_rotate(quat: tuple[float, float, float, float], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    qx, qy, qz, qw = quat
    vx, vy, vz = vec
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _bearing_quat(bearing: float) -> tuple[float, float, float, float]:
    import math

    half = float(bearing or 0.0) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compose two xyzw quaternions without importing a GUI/math adapter."""

    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _model_node_named(source_model: Any, name: str) -> Any | None:
    """Find one node by case-insensitive name in a loaded Kotor model."""

    target = str(name or "").strip().lower()
    if not target:
        return None
    root = getattr(source_model, "root_node", None)
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        if str(getattr(node, "name", "") or "").strip().lower() == target:
            return node
        stack.extend(tuple(getattr(node, "children", ()) or ()))
    return None


def _flattened_mesh_nodes(
    md: Any,
    source_model: Any,
    group_node: Any,
    *,
    group_resref: str,
    role: str,
    rotation: tuple[float, float, float, float] | None = None,
    pre_transform: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None = None,
    override_texture: str = "",
) -> list[Any]:
    """Bake a loaded KotorModel's nested meshes into flat group-local nodes.

    ``world_transform()`` gives each mesh node's model-space transform; we bake
    vertices/normals into group-local space so the preview stays a strict
    root -> group -> mesh two-level tree (hover walker + renderer contract).
    ``rotation`` applies an extra group-level rotation (creature bearing).
    ``pre_transform`` applies a socket offset — position plus quaternion, e.g.
    the body's ``headhook`` — after the mesh's own model-space bake and before
    the bearing rotation, so grafted head models sit where the engine puts them.
    ``override_texture`` is copy-only instance material state; source MDLs are
    never mutated when multiple appearance rows share one full-body model.
    """

    flattened: list[Any] = []
    root = getattr(source_model, "root_node", None)
    if root is None:
        return flattened
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(tuple(getattr(node, "children", ()) or ()))
        vertices = tuple(getattr(node, "vertices", ()) or ())
        faces = tuple(getattr(node, "faces", ()) or ())
        if not vertices or not faces:
            continue
        if not bool(getattr(node, "render", True)):
            continue
        flags = int(getattr(node, "flags", 0) or 0)
        if flags & int(md.NodeFlags.AABB):
            continue
        try:
            (wx, wy, wz), wq = node.world_transform()
        except Exception:
            (wx, wy, wz), wq = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        is_skin = bool(flags & int(md.NodeFlags.SKIN))
        baked_vertices: list[tuple[float, float, float]] = []
        baked_normals: list[tuple[float, float, float]] = []
        normals = tuple(getattr(node, "normals", ()) or ())
        for index, vertex in enumerate(vertices):
            local = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            if is_skin:
                point = (local[0] + wx, local[1] + wy, local[2] + wz)
            else:
                rotated = _quat_rotate(wq, local)
                point = (rotated[0] + wx, rotated[1] + wy, rotated[2] + wz)
            if pre_transform is not None:
                hook_position, hook_rotation = pre_transform
                point = _quat_rotate(hook_rotation, point)
                point = (point[0] + hook_position[0], point[1] + hook_position[1], point[2] + hook_position[2])
            if rotation is not None:
                point = _quat_rotate(rotation, point)
            baked_vertices.append(point)
            if index < len(normals):
                normal = (float(normals[index][0]), float(normals[index][1]), float(normals[index][2]))
                if not is_skin:
                    normal = _quat_rotate(wq, normal)
                if pre_transform is not None:
                    normal = _quat_rotate(pre_transform[1], normal)
                if rotation is not None:
                    normal = _quat_rotate(rotation, normal)
                baked_normals.append(normal)
        if len(baked_normals) != len(baked_vertices):
            baked_normals = [(0.0, 0.0, 1.0)] * len(baked_vertices)
        uvs = [tuple(float(v) for v in uv[:2]) for uv in tuple(getattr(node, "uvs", ()) or ())]
        face_uvs = [tuple(int(index) for index in row) for row in (getattr(node, "face_uvs", ()) or ())]
        independent_uvs = len(face_uvs) == len(faces) and all(
            len(row) == 3 and all(0 <= index < len(uvs) for index in row) for row in face_uvs
        )
        if not independent_uvs:
            face_uvs = []
        if not independent_uvs and len(uvs) != len(baked_vertices):
            uvs = [(0.0, 0.0)] * len(baked_vertices)
        texture_override = str(override_texture or "").strip().lower()
        if texture_override in {"null", "none", "****"}:
            texture_override = ""
        texture = texture_override or str(getattr(node, "texture", "") or "")
        clean_faces = [tuple(int(v) for v in face[:3]) for face in faces]
        face_mats = list(getattr(node, "face_mats", ()) or ())
        if len(face_mats) != len(clean_faces):
            face_mats = [0] * len(clean_faces)
        texture_names = [str(name) for name in tuple(getattr(node, "texture_names", ()) or ())]
        if texture_override:
            if texture_names:
                texture_names[0] = texture_override
            else:
                texture_names = [texture_override]
        source_alpha = getattr(node, "alpha", None)
        mesh_node = md.ModelNode(
            name=str(getattr(node, "name", "") or f"{group_resref}_{role}"),
            flags=int(md.NodeFlags.MESH),
            vertices=baked_vertices,
            normals=baked_normals,
            uvs=uvs,
            face_uvs=face_uvs,
            faces=clean_faces,
            face_mats=face_mats,
            texture=texture,
            texture_names=texture_names or ([texture] if texture else []),
            tex_count=max(1, int(getattr(node, "tex_count", 1) or 1)),
            lightmap=str(getattr(node, "lightmap", "") or ""),
            has_lightmap=bool(getattr(node, "has_lightmap", False)),
            diffuse=tuple(float(v) for v in tuple(getattr(node, "diffuse", (0.8, 0.8, 0.8)))[:3]),
            ambient=tuple(float(v) for v in tuple(getattr(node, "ambient", (0.2, 0.2, 0.2)))[:3]),
            # Alpha 0 is meaningful Odyssey state, not a missing value. Door
            # models such as DOR_LKO04 carry an untextured ``trans`` helper at
            # alpha 0; coercing it through ``or 1.0`` turns that invisible
            # helper into a large white fallback-texture panel.
            alpha=float(1.0 if source_alpha is None else source_alpha),
        )
        # Flattening changes only geometry space. Preserve the source material
        # contract so doors, foliage, glass, glows, and other stock assets keep
        # the same render classification as their original MDL nodes.
        for material_attr in (
            "specular",
            "shininess",
            "selfillum",
            "transparency_hint",
            "beaming",
            "background_geometry",
            "rotate_texture",
            "animate_uv",
            "uv_dir_x",
            "uv_dir_y",
            "uv_jitter",
            "uv_jitter_speed",
            "dirt_enabled",
            "dirt_texture",
            "dirt_coord_space",
            "hide_in_holograms",
        ):
            if hasattr(node, material_attr):
                setattr(mesh_node, material_attr, getattr(node, material_attr))
        uvs_lm = [tuple(float(v) for v in uv[:2]) for uv in tuple(getattr(node, "uvs_lm", ()) or ())]
        if len(uvs_lm) == len(baked_vertices):
            mesh_node.uvs_lm = uvs_lm
        mesh_node.parent = group_node
        setattr(mesh_node, "_gr_map_studio_room_resref", group_resref)
        # Indexed role: GModeler face edits must identify WHICH flattened
        # surface a hover hit; the index matches the imported-mesh surface
        # order because both walk the source model with the same traversal.
        setattr(mesh_node, "_gr_map_studio_mesh_role", f"{role}_{len(flattened)}")
        setattr(mesh_node, "_gr_map_studio_stock_mesh", True)
        if texture_override:
            setattr(mesh_node, "_gr_instance_texture_override", texture_override)
        try:
            mesh_node.compute_bounds()
        except Exception:
            pass
        flattened.append(mesh_node)
    return flattened


def _flattened_emitter_nodes(
    md: Any,
    source_model: Any,
    group_node: Any,
    *,
    group_resref: str,
    role: str,
    rotation: tuple[float, float, float, float] | None = None,
) -> list[Any]:
    """Copy source emitters into the combined preview in group-local space.

    Map Studio flattens mesh transforms for picking, but emitter nodes carry
    simulation state rather than triangles.  Their model-space transform is
    therefore baked onto a lightweight child while the authored placement
    group continues to own world translation and later transform edits.
    """

    flattened: list[Any] = []
    root = getattr(source_model, "root_node", None)
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        stack.extend(tuple(getattr(node, "children", ()) or ()))
        if not bool(getattr(node, "is_emitter", False)):
            continue
        try:
            position, orientation = node.world_transform()
        except Exception:
            position, orientation = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        position = tuple(float(value) for value in tuple(position)[:3])
        orientation = tuple(float(value) for value in tuple(orientation)[:4])
        if rotation is not None:
            position = _quat_rotate(rotation, position)
            orientation = _quat_multiply(rotation, orientation)
        emitter_node = md.ModelNode(
            name=str(getattr(node, "name", "") or f"{group_resref}_{role}_emitter"),
            flags=int(getattr(node, "flags", int(md.NodeFlags.EMITTER)) or int(md.NodeFlags.EMITTER)),
            position=position,
            rotation=orientation,
        )
        emitter_node.parent = group_node
        emitter_node.emitter_params = deepcopy(dict(getattr(node, "emitter_params", {}) or {}))
        emitter_node.controllers = deepcopy(list(getattr(node, "controllers", ()) or ()))
        setattr(emitter_node, "_gr_map_studio_room_resref", group_resref)
        setattr(emitter_node, "_gr_map_studio_emitter_role", f"{role}_{len(flattened)}")
        setattr(emitter_node, "_gr_map_studio_stock_emitter", True)
        flattened.append(emitter_node)
    return flattened


@dataclass(frozen=True)
class StockContentPreviewResult:
    """Result of merging stock rooms + instance geometry into a preview model."""

    room_count: int = 0
    instance_count: int = 0
    mesh_count: int = 0
    emitter_count: int = 0
    resolved_placement_ids: tuple[str, ...] = field(default_factory=tuple)
    unresolved_placement_ids: tuple[str, ...] = field(default_factory=tuple)
    placement_models: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_kotor_model_from_bytes(mdl: bytes, mdx: bytes = b"", *, resref: str = "") -> Any | None:
    """Decode an MDL/MDX pair supplied by a project/library resource bundle."""

    clean = str(resref or "").strip().lower()
    if not mdl:
        return None
    try:
        from src.core.game.kotor_loader import load_model_from_bytes  # type: ignore
    except Exception:
        try:
            from core.game.kotor_loader import load_model_from_bytes  # type: ignore
        except Exception as exc:
            log.warning("Map Studio stock model: kotor_loader import failed: %s", exc)
            return None
    try:
        model = load_model_from_bytes(bytes(mdl), bytes(mdx or b""))
        if model is None:
            log.warning("Map Studio model bytes returned no model for %s.", clean or "project resource")
        return model
    except Exception as exc:
        log.warning("Map Studio model bytes failed for %s: %s", clean or "project resource", exc)
        return None


def _load_kotor_model(resource_manager: Any, resref: str, game: str) -> Any | None:
    clean = str(resref or "").strip().lower()
    if not clean or clean == "null" or resource_manager is None:
        return None
    try:
        strict = getattr(resource_manager, "get_strict", None)
        mdl = strict(clean, RES_MDL, game) if callable(strict) else resource_manager.get_mdl(clean, game)
        if not mdl:
            log.warning("Map Studio stock model: %s.mdl not found in %s game resources.", clean, game)
            return None
        mdx = (
            strict(clean, RES_MDX, game)
            if callable(strict)
            else resource_manager.get_mdx(clean, game)
        ) or b""
    except Exception as exc:
        log.warning("Map Studio stock model: resource lookup failed for %s (%s): %s", clean, game, exc)
        return None
    return load_kotor_model_from_bytes(bytes(mdl), bytes(mdx), resref=clean)


def load_stock_kotor_model(resource_manager: Any, resref: str, game: str) -> Any | None:
    """Public alias: load one stock KOTOR MDL/MDX model by resref."""

    return _load_kotor_model(resource_manager, resref, game)


def append_stock_content_to_preview_root(
    md: Any,
    root: Any,
    *,
    rooms: tuple[Any, ...] = (),
    placements: tuple[Any, ...] = (),
    resource_manager: Any = None,
    game: str = "K1",
    model_loader: Any = None,
    resolver: Any = None,
) -> StockContentPreviewResult:
    """Append stock room + instance group nodes to a preview-model root.

    ``rooms``: KMAP ``RoomInstance`` rows (duck-typed: ``model_resref``,
    ``transform.position``, ``visible``, ``room_id``, ``name``).
    ``placements``: ``AuthoredGameplayPlacementRow`` rows (duck-typed:
    ``kind``, ``template_resref``, ``position``, ``bearing``, ``is_spatial``).
    ``model_loader``: optional ``(resref, game) -> KotorModel|None`` override
    for tests; defaults to ResourceManager MDL/MDX + ``load_model_from_bytes``.
    """

    warnings: list[str] = []
    game_tag = str(game or "K1").upper()
    loader = model_loader or (lambda resref, g=game_tag: _load_kotor_model(resource_manager, resref, g))
    room_count = 0
    instance_count = 0
    mesh_count = 0
    emitter_count = 0
    resolved_placement_ids: list[str] = []
    unresolved_placement_ids: list[str] = []
    placement_models: list[tuple[str, str]] = []
    model_cache: dict[str, Any | None] = {}

    def _model_for(resref: str) -> Any | None:
        key = str(resref or "").strip().lower()
        if not key:
            return None
        if key not in model_cache:
            model_cache[key] = loader(key)
        return model_cache[key]

    for room in tuple(rooms or ()):
        if not bool(getattr(room, "visible", True)):
            continue
        resref = str(getattr(room, "model_resref", "") or "").strip().lower()
        if not resref or resref == "null":
            continue
        source_model = _model_for(resref)
        if source_model is None:
            warnings.append(f"Stock room {resref} could not be loaded from the {game_tag} game resources.")
            continue
        transform = getattr(room, "transform", None)
        position = _vec3(getattr(transform, "position", (0.0, 0.0, 0.0)))
        group = md.ModelNode(
            name=resref,
            flags=int(md.NodeFlags.HEADER),
            position=position,
        )
        group.parent = root
        setattr(group, "_gr_map_studio_room_resref", resref)
        setattr(group, "_gr_map_studio_stock_room", True)
        setattr(group, "_gr_map_studio_room_id", str(getattr(room, "room_id", "") or ""))
        meshes = _flattened_mesh_nodes(md, source_model, group, group_resref=resref, role="stock_room")
        emitters = _flattened_emitter_nodes(
            md, source_model, group, group_resref=resref, role="stock_room"
        )
        if not meshes and not emitters:
            warnings.append(f"Stock room {resref} has no renderable mesh or emitter nodes.")
            continue
        group.children.extend((*meshes, *emitters))
        root.children.append(group)
        room_count += 1
        mesh_count += len(meshes)
        emitter_count += len(emitters)

    active_resolver = resolver
    if active_resolver is None and resource_manager is not None:
        active_resolver = TemplateModelResolver(resource_manager, game_tag)

    for placement in tuple(placements or ()):
        if not bool(getattr(placement, "is_spatial", True)):
            continue
        kind = str(getattr(placement, "kind", "") or "").strip().lower()
        if kind not in {"creature", "placeable", "door", "sky_traffic", "entry_point"}:
            continue
        template_resref = str(getattr(placement, "template_resref", "") or "").strip()
        resolver_template_resref = (
            str(getattr(placement, "creature_source_template_resref", "") or "").strip()
            if kind == "creature"
            else ""
        ) or template_resref
        placement_id = str(getattr(placement, "placement_id", "") or f"{kind}:{template_resref}")
        direct_model_kind = kind in {"sky_traffic", "entry_point"}
        if not template_resref or (active_resolver is None and not direct_model_kind):
            unresolved_placement_ids.append(placement_id)
            continue
        # Sky traffic and the IFO player start are viewport-only direct model
        # references. They bypass UTP/UTC/UTD template lookup; neither is a GIT
        # placement even though both must behave like editable scene objects.
        model_resref = (
            str(getattr(placement, "model_resref", "") or template_resref)
            if direct_model_kind
            else str(active_resolver.model_for_placement_kind(kind, resolver_template_resref) or "")
        )
        if not model_resref:
            warnings.append(f"{kind.replace('_', ' ').title()} {template_resref} has no resolvable model; keeping marker only.")
            unresolved_placement_ids.append(placement_id)
            continue
        source_model = _model_for(model_resref)
        if source_model is None:
            warnings.append(f"{kind.replace('_', ' ').title()} {template_resref} model {model_resref} could not be loaded.")
            unresolved_placement_ids.append(placement_id)
            continue
        group = md.ModelNode(
            name=f"{kind}_{template_resref}",
            flags=int(md.NodeFlags.HEADER),
            position=_vec3(getattr(placement, "position", (0.0, 0.0, 0.0))),
        )
        group.parent = root
        setattr(group, "_gr_map_studio_room_resref", placement_id)
        setattr(group, "_gr_map_studio_stock_instance", True)
        setattr(group, "_gr_map_studio_placement_id", placement_id)
        setattr(group, "_gr_map_studio_placement_kind", kind)
        setattr(group, "_gr_map_studio_template_resref", template_resref)
        setattr(group, "_gr_map_studio_model_resref", model_resref)
        rotation = _bearing_quat(float(getattr(placement, "bearing", 0.0) or 0.0))
        body_texture_resref = ""
        body_texture_for_kind = getattr(active_resolver, "body_texture_for_placement_kind", None)
        if kind == "creature" and callable(body_texture_for_kind):
            try:
                body_texture_resref = str(
                    body_texture_for_kind(kind, resolver_template_resref) or ""
                ).strip().lower()
            except Exception as exc:
                warnings.append(
                    f"Creature {template_resref} body texture could not be resolved: {exc}"
                )
        if body_texture_resref:
            setattr(group, "_gr_map_studio_body_texture_resref", body_texture_resref)
        meshes = _flattened_mesh_nodes(
            md,
            source_model,
            group,
            group_resref=placement_id,
            role=f"stock_{kind}",
            rotation=rotation,
            override_texture=body_texture_resref,
        )
        emitters = _flattened_emitter_nodes(
            md,
            source_model,
            group,
            group_resref=placement_id,
            role=f"stock_{kind}",
            rotation=rotation,
        )
        if not meshes and not emitters:
            warnings.append(
                f"{kind.replace('_', ' ').title()} {template_resref} model {model_resref} "
                "has no renderable meshes or emitters."
            )
            unresolved_placement_ids.append(placement_id)
            continue
        head_for_kind = getattr(active_resolver, "head_model_for_placement_kind", None)
        head_resref = str(getattr(placement, "head_model_resref", "") or "").strip().lower()
        if kind == "creature" and callable(head_for_kind):
            # Body-part appearances (modeltype B) carry no head geometry: the
            # engine grafts appearance.2da normalhead -> heads.2da at the
            # body's headhook.  Mirror that so previews aren't headless.
            head_resref = str(head_for_kind(kind, resolver_template_resref) or "")
        if head_resref and kind in {"creature", "entry_point"}:
            head_model = _model_for(head_resref)
            hook = _model_node_named(source_model, "headhook")
            if head_model is None:
                warnings.append(
                    f"{kind.replace('_', ' ').title()} {template_resref} head {head_resref} could not be loaded; the body previews headless."
                )
            elif hook is None:
                warnings.append(
                    f"{kind.replace('_', ' ').title()} {template_resref} body {model_resref} has no headhook node; head {head_resref} skipped."
                )
            else:
                try:
                    hook_position, hook_rotation = hook.world_transform()
                except Exception:
                    hook_position, hook_rotation = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
                head_meshes = _flattened_mesh_nodes(
                    md,
                    head_model,
                    group,
                    group_resref=placement_id,
                    role=f"stock_{kind}_head",
                    rotation=rotation,
                    pre_transform=(
                        tuple(float(value) for value in tuple(hook_position)[:3]),
                        tuple(float(value) for value in tuple(hook_rotation)[:4]),
                    ),
                )
                meshes.extend(head_meshes)
        for mesh in meshes:
            setattr(mesh, "_gr_map_studio_placement_id", placement_id)
            setattr(mesh, "_gr_map_studio_placement_kind", kind)
        for emitter in emitters:
            setattr(emitter, "_gr_map_studio_placement_id", placement_id)
            setattr(emitter, "_gr_map_studio_placement_kind", kind)
        group.children.extend((*meshes, *emitters))
        root.children.append(group)
        instance_count += 1
        mesh_count += len(meshes)
        emitter_count += len(emitters)
        resolved_placement_ids.append(placement_id)
        placement_models.append((placement_id, model_resref))

    return StockContentPreviewResult(
        room_count=room_count,
        instance_count=instance_count,
        mesh_count=mesh_count,
        emitter_count=emitter_count,
        resolved_placement_ids=tuple(resolved_placement_ids),
        unresolved_placement_ids=tuple(unresolved_placement_ids),
        placement_models=tuple(placement_models),
        warnings=tuple(warnings),
    )


def build_map_studio_combined_preview_model(
    *,
    authored_model: Any = None,
    project_name: str = "map_studio_preview",
    game: str = "K1",
    rooms: tuple[Any, ...] = (),
    placements: tuple[Any, ...] = (),
    resource_manager: Any = None,
    model_loader: Any = None,
    resolver: Any = None,
    resource_revision: object = "",
) -> tuple[Any | None, StockContentPreviewResult]:
    """Merge authored preview geometry with stock rooms + instance geometry.

    Returns ``(model, stock_result)``.  ``model`` is ``None`` when neither
    authored nor stock geometry produced meshes.  The authored model's room
    group nodes are re-used as-is (tags intact); stock content is appended as
    sibling groups.  A fresh ``_gr_map_studio_preview_key`` is derived from
    the authored key plus the stock composition so the viewport reloads when
    either side changes.
    """

    md = _import_model_data()
    has_stock = bool(rooms) or bool(placements)
    if not has_stock:
        return authored_model, StockContentPreviewResult()

    if authored_model is not None:
        root = getattr(authored_model, "root_node", None)
        model = authored_model
        if root is None:
            return authored_model, StockContentPreviewResult()
    else:
        root = md.ModelNode(name=str(project_name or "map_studio_preview"), flags=int(md.NodeFlags.HEADER))
        model = md.KotorModel(
            name=str(project_name or "map_studio_preview"),
            supermodel="NULL",
            classification="area",
            game_version=md.GameVersion.K2 if str(game).upper() == "K2" else md.GameVersion.K1,
            model_type=int(md.ModelClassification.EFFECT),
            root_node=root,
        )
        model.disable_fog = True
        setattr(model, "_gr_map_studio_preview_model", True)

    result = append_stock_content_to_preview_root(
        md,
        root,
        rooms=tuple(rooms or ()),
        placements=tuple(placements or ()),
        resource_manager=resource_manager,
        game=game,
        model_loader=model_loader,
        resolver=resolver,
    )
    if result.mesh_count <= 0 and result.emitter_count <= 0 and authored_model is None:
        return None, result

    import hashlib as _hashlib

    stock_signature = "|".join(
        sorted(
            [
                f"room:{str(getattr(room, 'model_resref', '') or '')}:{_vec3(getattr(getattr(room, 'transform', None), 'position', (0, 0, 0)))}:{bool(getattr(room, 'visible', True))}"
                for room in tuple(rooms or ())
            ]
            + [
                f"inst:{str(getattr(p, 'kind', '') or '')}:{str(getattr(p, 'template_resref', '') or '')}:{_vec3(getattr(p, 'position', (0, 0, 0)))}:{float(getattr(p, 'bearing', 0.0) or 0.0):.4f}"
                for p in tuple(placements or ())
            ]
        )
    )
    base_key = str(getattr(model, "_gr_map_studio_preview_key", "") or "authored:none")
    resolved_signature = "|".join(f"{placement_id}:{model_resref}" for placement_id, model_resref in result.placement_models)
    combined_key = _hashlib.sha1(
        f"{base_key}|{stock_signature}|{resolved_signature}|resources:{resource_revision}".encode("utf-8")
    ).hexdigest()
    setattr(model, "_gr_map_studio_preview_key", combined_key)
    setattr(model, "_gr_map_studio_resolved_placement_ids", result.resolved_placement_ids)
    setattr(model, "_gr_map_studio_unresolved_placement_ids", result.unresolved_placement_ids)
    try:
        model.compute_bounds()
    except Exception:
        pass
    return model, result
