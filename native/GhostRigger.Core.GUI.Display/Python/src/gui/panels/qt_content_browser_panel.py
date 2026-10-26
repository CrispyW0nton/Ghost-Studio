"""Unified Qt Content Browser for GhostRigger assets."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from PySide6 import QtCore, QtWidgets

from src.gui.qt_lib.assets.qt_theme import icon
from src.gui.qt_lib.panels.qt_library_panel import (
    MODEL_CATEGORY_ORDER,
    MODEL_SUBCATEGORY_ORDER,
    content_browser_metadata_for_resref,
    enrich_library_rows,
    infer_model_category,
)


ASSET_TYPES = ("All", "Model", "Animation", "Texture", "Blueprint", "Module", "Scene")


_READABLE_RESREF_OVERRIDES = {
    "plc_backpack": "Backpack",
    "plc_bodyranc": "Rancor Corpse",
    "plc_brokndrd": "Broken Droid",
    "plc_fccage": "Force Cage",
    "plc_koltank": "Kolto Tank",
    "plc_lndspdr": "Landspeeder",
    "plc_oilpudle": "Oil Puddle",
    "plc_pwrcond": "Power Conduit",
    "plc_rakatflg": "Rakatan Flag",
    "plc_rnepillr": "Ruined Pillar",
    "plc_sithsarc": "Sith Sarcophagus",
    "plc_starmap": "Star Map",
    "plc_stmvent": "Steam Vent",
    "plc_wookcrps": "Wookiee Corpse",
}

_READABLE_PREFIXES = (
    "g_w_", "g_i_", "g_a_", "iw_", "ia_", "uti_", "plc_", "dor_", "door_", "w_", "i_",
    "a_", "n_", "c_", "p_", "s_", "l_", "fx_", "v_",
)

_READABLE_WORDS = {
    "acd": "Acid",
    "ban": "Banner",
    "blstr": "Blaster",
    "btnpnl": "Button Panel",
    "cjar": "Ceramic Jar",
    "comp": "Computer",
    "crps": "Corpse",
    "dbl": "Double",
    "drd": "Droid",
    "fx": "FX",
    "holo": "Hologram",
    "hvy": "Heavy",
    "jnk": "Junk",
    "lght": "Light",
    "lghtsbr": "Lightsaber",
    "pnl": "Panel",
    "rakat": "Rakatan",
    "rfl": "Rifle",
    "sbr": "Saber",
    "spchunk": "Ship Chunk",
    "spc": "Space",
    "swy": "Swoop",
    "vbr": "Vibro",
}

_DISPLAY_MEMBER_NAMES = (
    (("bastila",), "Bastila"),
    (("carth",), "Carth"),
    (("mission",), "Mission"),
    (("zaalbar",), "Zaalbar"),
    (("canderous", "p_cand"), "Canderous"),
    (("jolee",), "Jolee"),
    (("juhani",), "Juhani"),
    (("hk47", "hk_47"), "HK-47"),
    (("t3m4", "t3_m4", "t3m3"), "T3-M4"),
    (("kreia",), "Kreia"),
    (("atton",), "Atton"),
    (("baodur", "bao_dur", "bao-dur", "bao"), "Bao-Dur"),
    (("handmaiden",), "Handmaiden"),
    (("disciple", "mical"), "Disciple"),
    (("visas",), "Visas"),
    (("mira",), "Mira"),
    (("hanharr",), "Hanharr"),
    (("mandalore", "mandra"), "Mandalore"),
    (("g0t0", "goto"), "G0-T0"),
    (("atris",), "Atris"),
)

_SPECIES_SINGULAR = {
    "Aqualish": "Aqualish",
    "Bith": "Bith",
    "Devaronians": "Devaronian",
    "Duros": "Duros",
    "Gamorreans": "Gamorrean",
    "Gand": "Gand",
    "Gran": "Gran",
    "Ithorians": "Ithorian",
    "Quarren": "Quarren",
    "Rakata": "Rakata",
    "Rodians": "Rodian",
    "Selkath": "Selkath",
    "Sullustans": "Sullustan",
    "Trandoshans": "Trandoshan",
    "Tusken Raiders": "Tusken Raider",
    "Twi'leks": "Twi'lek",
    "Weequay": "Weequay",
    "Wookiees": "Wookiee",
    "Yoda Species": "Yoda Species",
}


def _display_name_from_library_row(row: dict, category: str, resref: str, metadata: dict[str, str]) -> str:
    game_name = _game_metadata_display_name(row, resref)
    if game_name:
        return game_name
    encoded = _encoded_display_name(category, resref, metadata)
    if encoded:
        return encoded
    candidates = (
        row.get("display_name"),
        row.get("localized_name"),
        row.get("name_label"),
        row.get("area_label") if category == "Modules" else "",
        row.get("area_name") if category == "Modules" else "",
        row.get("placeable_tag"),
        row.get("door_tag"),
        row.get("item_tag"),
        row.get("template_name"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text.lower() != resref.lower():
            return _humanize_asset_name(text)
    return _humanize_asset_name(resref)


def _game_metadata_display_name(row: dict, resref: str) -> str:
    source = str(row.get("metadata_source") or "")
    if not any(token in source for token in ("2da", "UTI", "UTP", "UTD")):
        return ""
    for key in ("item_display_name", "localized_name", "name_label", "template_name"):
        text = str(row.get(key) or "").strip()
        if not text or text.lower() == resref.lower():
            continue
        label = _humanize_asset_name(text)
        outfit_variant = _outfit_display_variant(row)
        if outfit_variant:
            return _with_variant(label, outfit_variant)
        return label
    return ""


def _outfit_display_variant(row: dict) -> str:
    pieces = [
        str(row.get("outfit_gender") or "").strip(),
        str(row.get("outfit_size") or "").strip(),
    ]
    return " ".join(piece for piece in pieces if piece)


def _encoded_display_name(category: str, resref: str, metadata: dict[str, str]) -> str:
    r = (resref or "").lower()
    if category == "Modules":
        return str(metadata.get("area") or "").strip()
    if category == "Player Characters":
        return _player_display_name(r, metadata)
    if category == "Party Members":
        return _party_display_name(r, metadata)
    if category in {"Commoners", "NPCs", "Level Assets", "Misc Models"}:
        species = _species_gender_display_name(r, metadata)
        if species:
            return species
    if category == "Commoners":
        commoner = str(metadata.get("commoner_type") or "").strip()
        return _with_variant(f"{commoner} Commoner".strip(), _letter_or_number_variant(r))
    if category == "NPCs":
        return _npc_display_name(r, metadata)
    if category == "Droids":
        return _droid_display_name(r, metadata)
    if category == "Creatures":
        creature = str(metadata.get("creature_type") or "").strip()
        return _with_variant(creature, _letter_or_number_variant(r))
    if category == "Supermodels":
        supermodel = str(metadata.get("supermodel_type") or "").strip()
        return f"{supermodel} Supermodel".strip()
    if category == "Turrets":
        turret = str(metadata.get("turret_type") or "").strip()
        return f"{turret} Turret".strip()
    if category == "Holograms":
        holo = str(metadata.get("hologram_type") or "").strip()
        return f"{holo} Hologram".strip()
    if category == "Engine Items":
        return _with_variant(str(metadata.get("engine_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Environment":
        return _with_variant(str(metadata.get("environment_type") or "").strip(), _letter_or_number_variant(r))
    if category == "GUI":
        return _with_variant(str(metadata.get("gui_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Level Assets":
        asset_type = str(metadata.get("level_asset_type") or "").strip()
        return _with_variant(asset_type, _letter_or_number_variant(r))
    if category == "Minigame":
        return _with_variant(str(metadata.get("minigame_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Menus":
        return _with_variant(str(metadata.get("menu_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Visual FX":
        fx_type = str(metadata.get("fx_type") or "").strip()
        return _with_variant(f"{fx_type} FX".strip(), _letter_or_number_variant(r))
    if category == "Visuals":
        visual_type = str(metadata.get("visual_type") or "").strip()
        return _with_variant(f"{visual_type} Visual".strip(), _letter_or_number_variant(r))
    if category == "Planets":
        return _with_variant(str(metadata.get("planet_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Stunts":
        return _with_variant(str(metadata.get("stunt_type") or "").strip(), _letter_or_number_variant(r))
    if category == "Templates":
        return _with_variant(str(metadata.get("template_type") or "").strip() + " Template", _letter_or_number_variant(r))
    return ""


def _player_display_name(resref: str, metadata: dict[str, str]) -> str:
    r = resref[3:] if resref.startswith("k1_") else resref
    if not r.startswith(("pm", "pf")):
        return ""
    gender = str(metadata.get("player_gender") or ("Male" if r.startswith("pm") else "Female"))
    part = str(metadata.get("player_part") or "")
    class_code = str(metadata.get("player_class") or "").replace("Class ", "").strip()
    variant = str(metadata.get("player_variant") or "").strip()
    pieces = ["Player", gender]
    if part and part != "Model":
        pieces.append(part)
    if class_code:
        pieces.append(class_code)
    if variant and not variant.startswith("Head "):
        pieces.append(variant)
    elif variant.startswith("Head "):
        pieces.append(variant.replace("Head ", ""))
    return " ".join(pieces)


def _party_display_name(resref: str, metadata: dict[str, str]) -> str:
    member = str(metadata.get("party_member") or "").strip()
    if not member:
        for tokens, label in _DISPLAY_MEMBER_NAMES:
            if any(token in resref for token in tokens):
                member = label
                break
    if not member:
        return ""
    part = str(metadata.get("party_model_part") or "").strip()
    if not part:
        if re.search(r"h\d*$", resref):
            part = "Head"
        elif re.search(r"b[a-z]?$", resref):
            part = "Body"
        else:
            part = "Model"
    variant = _party_variant(resref, part)
    return _with_variant(f"{member} {part}".strip(), variant)


def _npc_display_name(resref: str, metadata: dict[str, str]) -> str:
    species = _species_gender_display_name(resref, metadata)
    if species:
        return species
    variant = _npc_variant(resref)
    if any(token in resref for token in ("sithappr", "sith_app", "apprent")):
        return _with_variant("Sith Apprentice", variant)
    if any(token in resref for token in ("sithsold", "sithsoldier")):
        return _with_variant("Sith Soldier", variant)
    if any(token in resref for token in ("sithoff", "sith_off")):
        return _with_variant("Sith Officer", variant)
    if any(token in resref for token in ("sithass", "assassin")):
        return _with_variant("Sith Assassin", variant)
    if any(token in resref for token in ("jedmast", "jedi_master", "jedimaster", "master")):
        return _with_variant("Jedi Master", variant)
    if any(token in resref for token in ("padawan",)):
        return _with_variant("Jedi Padawan", variant)
    if any(token in resref for token in ("swoop", "gadon", "gendar", "brejik", "deadeye")):
        return _with_variant("Swoop Gang Member", variant)
    if any(token in resref for token in ("blackvulkar", "vulkar")):
        return _with_variant("Black Vulkar Gang Member", variant)
    if "bek" in resref:
        return _with_variant("Hidden Bek Gang Member", variant)
    role = str(metadata.get("npc_role") or "").strip()
    if role:
        return _with_variant(role, variant)
    faction = str(metadata.get("npc_faction") or "").strip()
    if faction and faction != "Named":
        part = str(metadata.get("npc_model_part") or "").strip()
        return _with_variant(f"{faction} {part}".strip(), variant)
    return ""


def _droid_display_name(resref: str, metadata: dict[str, str]) -> str:
    if "hk47" in resref or "hk_47" in resref:
        return "HK-47"
    if "t3m4" in resref or "t3_m4" in resref:
        return "T3-M4"
    if "g0t0" in resref or "goto" in resref:
        return "G0-T0"
    droid_type = str(metadata.get("droid_type") or "").strip()
    return _with_variant(f"{droid_type} Droid".strip(), _letter_or_number_variant(resref))


def _species_gender_display_name(resref: str, metadata: dict[str, str]) -> str:
    species = str(metadata.get("npc_species") or metadata.get("creature_type") or "").strip()
    if not species:
        for token, label in (
            ("wookie", "Wookiee"),
            ("wook", "Wookiee"),
            ("twilek", "Twi'lek"),
            ("twi", "Twi'lek"),
            ("selkath", "Selkath"),
            ("rakata", "Rakata"),
            ("rodian", "Rodian"),
            ("jawa", "Jawa"),
            ("ithorian", "Ithorian"),
            ("trandoshan", "Trandoshan"),
        ):
            if token in resref:
                species = label
                break
    species = _SPECIES_SINGULAR.get(species, species)
    if not species:
        return ""
    gender = ""
    if resref.endswith("f") or "_f" in resref or "female" in resref:
        gender = "Female"
    elif resref.endswith("m") or "_m" in resref or "male" in resref:
        gender = "Male"
    return " ".join(part for part in (species, gender) if part)


def _party_variant(resref: str, part: str) -> str:
    if part == "Head":
        match = re.search(r"h(\d+)$", resref)
        if match:
            return f"Head {int(match.group(1)):02d}"
    if part == "Body":
        match = re.search(r"b([a-z])$", resref)
        if match:
            return match.group(1).upper()
    return _letter_or_number_variant(resref)


def _letter_or_number_variant(resref: str) -> str:
    base = resref.lower().strip()
    match = re.search(r"(?:_|-)([a-z])(\d*)$", base)
    if match:
        letter, number = match.groups()
        if number:
            return f"{letter.upper()}{number}"
        if letter not in {"m", "f"}:
            return letter.upper()
    match = re.search(r"(\d+)$", base)
    if match:
        return f"{int(match.group(1)):02d}"
    return ""


def _npc_variant(resref: str) -> str:
    separated = _letter_or_number_variant(resref)
    if separated:
        return separated
    match = re.search(r"(\d+)(h|b)$", resref)
    if match:
        part = "Head" if match.group(2) == "h" else "Body"
        return f"{int(match.group(1)):02d} {part}"
    match = re.search(r"(?:swoopgang|blackvulkar|vulkar|bek)([a-z])$", resref)
    if match:
        return match.group(1).upper()
    return ""


def _with_variant(label: str, variant: str) -> str:
    label = label.strip()
    variant = variant.strip()
    if not label:
        return ""
    if not variant or label.endswith(f" {variant}"):
        return label
    return f"{label} {variant}"


def _humanize_asset_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix, label in _READABLE_RESREF_OVERRIDES.items():
        if lowered.startswith(prefix):
            suffix = lowered[len(prefix):]
            number = suffix if suffix.isdigit() else ""
            return f"{label} {number}".strip()
    for prefix in _READABLE_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            lowered = text.lower()
            break
    text = text.replace("_", " ").replace("-", " ")
    text = "".join(" " + char if index and char.isupper() and not text[index - 1].isspace() else char for index, char in enumerate(text))
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    words = [word for word in re.split(r"\s+", text) if word]
    readable = [_READABLE_WORDS.get(word.lower(), word) for word in words]
    return " ".join(word.upper() if word.lower() in {"fx", "gui", "hk", "t3", "tsf"} else word[:1].upper() + word[1:] for word in readable)


@dataclass(slots=True)
class ContentAssetDescriptor:
    """Small UI descriptor that keeps source data lossless for callers."""

    asset_type: str
    name: str
    display_name: str = ""
    game: str = ""
    category: str = ""
    source: str = ""
    row: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def searchable_text(self) -> str:
        values = [
            self.asset_type,
            self.name,
            self.display_name,
            self.game,
            self.category,
            self.source,
            *self.tags,
            *(str(value) for value in self.metadata.values()),
        ]
        return " ".join(value for value in values if value).lower()


def descriptor_from_library_row(row: dict) -> ContentAssetDescriptor:
    item = dict(row)
    category = str(item.get("category") or infer_model_category(str(item.get("resref", ""))))
    asset_type = "Module" if category == "Modules" else "Blueprint" if category == "Templates" else "Model"
    source = str(item.get("source", ""))
    name = str(item.get("resref", ""))
    taxonomy_metadata = content_browser_metadata_for_resref(name, category)
    subcategory = str(item.get("subcategory") or taxonomy_metadata.get("subcategory") or "")
    if subcategory:
        taxonomy_metadata["subcategory"] = subcategory
    metadata = {
        "area": item.get("area_label") or item.get("area_name") or "",
        "module": item.get("module_code") or "",
        "class": item.get("model_class") or "",
        "template": item.get("item_template_resref") or item.get("placeable_template_resref") or "",
        "tag": item.get("item_tag") or item.get("placeable_tag") or "",
        "door template": item.get("door_template_resref") or "",
        "door tag": item.get("door_tag") or "",
        "door type": item.get("door_generic_type") or "",
        "open lock dc": item.get("door_open_lock_dc") or "",
        "key": item.get("door_key_name") or "",
        "linked to": item.get("door_linked_to") or "",
        "base item": item.get("item_baseitem") or "",
        "bodyvar": item.get("item_bodyvar") or item.get("appearance_bodyvar") or "",
        "item": item.get("item_display_name") or "",
        "model variation": item.get("item_model_variation") or "",
        "appearance": item.get("placeable_appearance") or "",
        "appearance label": item.get("appearance_label") or "",
        "appearance row": item.get("appearance_row") or "",
        "appearance model": item.get("appearance_model_column") or "",
        "outfit": " ".join(
            part for part in (
                str(item.get("outfit_gender") or "").strip(),
                str(item.get("outfit_size") or "").strip(),
            )
            if part
        ),
        "metadata source": item.get("metadata_source") or "",
        **taxonomy_metadata,
    }
    return ContentAssetDescriptor(
        asset_type=asset_type,
        name=name,
        display_name=_display_name_from_library_row(item, category, name, metadata),
        game=str(item.get("game", "")),
        category=category,
        source=source,
        row=item,
        metadata=metadata,
        tags=tuple(
            str(value)
            for value in (
                category,
                subcategory,
                *taxonomy_metadata.values(),
                item.get("item_template_resref", ""),
                item.get("placeable_template_resref", ""),
                item.get("door_template_resref", ""),
                item.get("item_tag", ""),
                item.get("placeable_tag", ""),
                item.get("door_tag", ""),
                item.get("location", ""),
            )
            if value
        ),
    )


def descriptor_from_animation_entry(entry: dict) -> ContentAssetDescriptor:
    item = dict(entry)
    model = str(item.get("model") or item.get("model_name") or "")
    anim = str(item.get("animation") or item.get("anim_name") or "")
    game = str(item.get("game", ""))
    source = str(item.get("source", ""))
    object_name = str(item.get("object_name", ""))
    resref = str(item.get("resref", ""))
    return ContentAssetDescriptor(
        asset_type="Animation",
        name=anim or model,
        display_name=_humanize_asset_name(anim or model),
        game=game,
        category="Animation",
        source=source,
        row=item,
        metadata={
            "model": model,
            "object": object_name,
            "resref": resref,
            "frames": item.get("frames", ""),
            "length": item.get("length", ""),
            "source": source,
        },
        tags=tuple(str(value) for value in ("animation", model, object_name, resref, source) if value),
    )


class QtContentAssetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, asset: ContentAssetDescriptor):
        model_or_meta = str(asset.metadata.get("model") or asset.metadata.get("area") or asset.category or "")
        super().__init__([
            asset.display_name or asset.name,
            asset.name,
            asset.asset_type,
            asset.game,
            asset.category,
            model_or_meta,
        ])
        self.asset = asset
        self.setData(0, QtCore.Qt.UserRole, asset)


class QtContentBrowserPanel(QtWidgets.QWidget):
    """Unreal-style browser that owns models, animations, and future asset rows."""

    loadRequested = QtCore.Signal(str, str)
    primarySceneLoadRequested = QtCore.Signal(dict)
    addToCurrentSceneRequested = QtCore.Signal(dict)
    extractRequested = QtCore.Signal(dict)
    assetActionRequested = QtCore.Signal(str, dict)
    retargetSourceRequested = QtCore.Signal(dict)
    retargetTargetRequested = QtCore.Signal(dict)
    characterBuilderRequested = QtCore.Signal(dict)
    levelEditorNewRequested = QtCore.Signal(dict)
    levelEditorImportRequested = QtCore.Signal(dict)
    batchRequested = QtCore.Signal(str, list)
    scanRequested = QtCore.Signal()
    deepScanRequested = QtCore.Signal()
    libraryActionRequested = QtCore.Signal(str)
    loadConnectedMapChanged = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("contentBrowser")
        self._library_rows: list[dict] = []
        self._scene_animation_entries: list[dict] = []
        self._scanned_animation_entries: list[dict] = []
        self._assets: list[ContentAssetDescriptor] = []
        self._active_nav: tuple[str, str] = ("type", "All")
        self._deferred_asset_generation = 0
        self._deferred_filter_generation = 0
        self._splitter_user_adjusted = False
        self._splitter_layout_applied = False
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setOpaqueResize(True)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        root.addWidget(self.splitter, 1)

        self.nav_tree = QtWidgets.QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setObjectName("contentBrowserNavigation")
        self.nav_tree.setMinimumWidth(88)
        self.nav_tree.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.nav_tree.itemSelectionChanged.connect(self._on_navigation_changed)

        self.details = QtWidgets.QWidget()
        self.details.setObjectName("contentBrowserDetails")
        self.details.setMinimumWidth(72)
        self.details.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(5)
        self.detail_title = QtWidgets.QLabel("Select an asset")
        self.detail_title.setProperty("heading", True)
        self.detail_text = QtWidgets.QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumBlockCount(80)
        details_layout.addWidget(self.detail_title)
        details_layout.addWidget(self.detail_text, 1)
        self._build_action_buttons(details_layout)

        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setObjectName("contentBrowserSidebar")
        self.sidebar.setMinimumWidth(112)
        self.sidebar.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(3, 3, 3, 3)
        sidebar_layout.setSpacing(5)
        self.sidebar_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.sidebar_splitter.setObjectName("contentBrowserSidebarSplitter")
        self.sidebar_splitter.setChildrenCollapsible(False)
        self.sidebar_splitter.setOpaqueResize(True)
        self.sidebar_splitter.addWidget(self.nav_tree)
        self.sidebar_splitter.addWidget(self.details)
        self.sidebar_splitter.setStretchFactor(0, 3)
        self.sidebar_splitter.setStretchFactor(1, 2)
        self.sidebar_splitter.setSizes([320, 180])
        sidebar_layout.addWidget(self.sidebar_splitter, 1)
        self.splitter.addWidget(self.sidebar)

        center = QtWidgets.QWidget()
        self.asset_area = center
        center.setMinimumWidth(96)
        center.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)
        self._build_filters(center_layout)
        self.asset_view = QtWidgets.QTreeWidget()
        self.asset_view.setObjectName("contentBrowserAssets")
        self.asset_view.setHeaderLabels(["Display Name", "Asset Name", "Type", "Game", "Category", "Meta"])
        self.asset_view.setSortingEnabled(True)
        self.asset_view.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.asset_view.setRootIsDecorated(False)
        self.asset_view.setAlternatingRowColors(True)
        self.asset_view.header().setStretchLastSection(False)
        self.asset_view.header().setMinimumSectionSize(32)
        self.asset_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.asset_view.setDragEnabled(True)
        self.asset_view.itemDoubleClicked.connect(lambda _item, _column: self._activate_selected())
        self.asset_view.itemSelectionChanged.connect(self._update_details)
        self.asset_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.asset_view.customContextMenuRequested.connect(self._show_context_menu)
        center_layout.addWidget(self.asset_view, 1)
        self.count_label = QtWidgets.QLabel("")
        center_layout.addWidget(self.count_label)
        self.splitter.addWidget(center)

        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("No game directory set")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        status_row.addWidget(self.status_label, 1)
        self.scan_anims_button = QtWidgets.QPushButton("Scan Animations")
        self.scan_anims_button.setProperty("compact", True)
        self._make_status_button_shrinkable(self.scan_anims_button)
        self.scan_anims_button.clicked.connect(lambda: self.libraryActionRequested.emit("Scan Animations"))
        status_row.addWidget(self.scan_anims_button)
        self.refresh_anims_button = QtWidgets.QPushButton("Refresh Animations")
        self.refresh_anims_button.setProperty("compact", True)
        self._make_status_button_shrinkable(self.refresh_anims_button)
        self.refresh_anims_button.clicked.connect(lambda: self.libraryActionRequested.emit("Refresh"))
        status_row.addWidget(self.refresh_anims_button)
        for label, fmt in (("Batch OBJ", "obj"), ("Batch ASCII", "ascii"), ("Batch TGA", "tga")):
            button = QtWidgets.QPushButton(label)
            button.setProperty("compact", True)
            self._make_status_button_shrinkable(button)
            button.clicked.connect(lambda _checked=False, f=fmt: self.batchRequested.emit(f, self.visible_rows()))
            status_row.addWidget(button)
        self.deep_button = QtWidgets.QPushButton("Deep Scan")
        self.scan_button = QtWidgets.QPushButton("Scan")
        self.scan_button.setProperty("accent", True)
        self.deep_button.setProperty("compact", True)
        self.scan_button.setProperty("compact", True)
        self._make_status_button_shrinkable(self.deep_button)
        self._make_status_button_shrinkable(self.scan_button)
        self.deep_button.clicked.connect(self.deepScanRequested.emit)
        self.scan_button.clicked.connect(self.scanRequested.emit)
        status_row.addWidget(self.deep_button)
        status_row.addWidget(self.scan_button)
        root.addLayout(status_row)

        self._rebuild_navigation()
        self._apply_filter()
        QtCore.QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _make_status_button_shrinkable(self, button: QtWidgets.QPushButton) -> None:
        button.setMinimumWidth(0)
        button.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)

    def _build_filters(self, layout: QtWidgets.QVBoxLayout) -> None:
        search_row = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search assets")
        self.search_edit.textChanged.connect(self._apply_filter)
        clear = QtWidgets.QPushButton("x")
        clear.setProperty("compact", True)
        clear.setMinimumWidth(0)
        clear.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        clear.clicked.connect(self.search_edit.clear)
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(clear)
        layout.addLayout(search_row)

        filters = QtWidgets.QGridLayout()
        filters.setHorizontalSpacing(5)
        filters.setVerticalSpacing(3)
        self.type_filter = QtWidgets.QComboBox()
        self.type_filter.addItems(ASSET_TYPES)
        self.type_filter.currentTextChanged.connect(self._apply_filter)
        self.game_filter = QtWidgets.QComboBox()
        self.game_filter.addItems(["All", "K1", "K2"])
        self.game_filter.currentTextChanged.connect(self._apply_filter)
        self.source_filter = QtWidgets.QComboBox()
        self.source_filter.addItem("All Sources")
        self.source_filter.currentTextChanged.connect(self._apply_filter)
        self.tag_filter = QtWidgets.QComboBox()
        self.tag_filter.addItems([
            "All Tags",
            "Player Characters",
            *[
                f"Player Characters / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Player Characters"]
            ],
            "Party Members",
            *[
                f"Party Members / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Party Members"]
            ],
            "Commoners",
            *[
                f"Commoners / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Commoners"]
            ],
            "NPCs",
            *[
                f"NPCs / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["NPCs"]
            ],
            "Droids",
            *[
                f"Droids / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Droids"]
            ],
            "Turrets",
            *[
                f"Turrets / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Turrets"]
            ],
            "Creatures",
            *[
                f"Creatures / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Creatures"]
            ],
            "Holograms",
            *[
                f"Holograms / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Holograms"]
            ],
            "Supermodels",
            *[
                f"Supermodels / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Supermodels"]
            ],
            "Modules",
            *[
                f"Modules / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Modules"]
            ],
            "Level Assets",
            *[
                f"Level Assets / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Level Assets"]
            ],
            "Environment",
            *[
                f"Environment / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Environment"]
            ],
            "Skyboxes",
            *[
                f"Skyboxes / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Skyboxes"]
            ],
            "Minigame",
            *[
                f"Minigame / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Minigame"]
            ],
            "Menus",
            *[
                f"Menus / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Menus"]
            ],
            "GUI",
            *[
                f"GUI / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["GUI"]
            ],
            "Placeables",
            *[
                f"Placeables / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Placeables"]
            ],
            "Doors",
            "Doors / Taris",
            "Doors / Dantooine",
            "Doors / Tatooine",
            "Doors / Kashyyyk",
            "Doors / Manaan",
            "Doors / Korriban",
            "Doors / Leviathan",
            "Doors / Star Forge",
            "Doors / Rakata",
            "Doors / Yavin",
            "Doors / Endar Spire",
            "Doors / Ebon Hawk",
            "Doors / Peragus",
            "Doors / Telos",
            "Doors / Harbinger",
            "Doors / Nar Shaddaa",
            "Doors / Dxun",
            "Doors / Onderon",
            "Doors / Malachor",
            "Doors / Ravager",
            "Doors / Droid Planet",
            "Doors / Force Fields",
            "Doors / Generic Doors",
            "Doors / Unknown Doors",
            "Engine Items",
            *[
                f"Engine Items / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Engine Items"]
            ],
            "Armor",
            "Armor / Clothing",
            "Armor / Jedi Robes",
            "Armor / Light Armor",
            "Armor / Medium Armor",
            "Armor / Heavy Armor",
            "Armor / Environmental Suits",
            "Armor / Disguises",
            "Armor / Misc Armor",
            "Inventory",
            "Inventory / Security Spikes",
            "Inventory / Computer Spikes",
            "Inventory / Parts",
            "Inventory / Mines",
            "Inventory / Spikes",
            "Inventory / Traps",
            "Inventory / Medkits",
            "Inventory / Masks",
            "Inventory / Implants",
            "Inventory / Gauntlets",
            "Inventory / Armbands",
            "Inventory / Droid Items",
            "Inventory / Belts",
            "Inventory / Stims",
            "Inventory / Adrenal Stims",
            "Inventory / Combat Shots",
            "Inventory / Credits",
            "Inventory / Upgrades",
            "Inventory / Datapads",
            "Inventory / Pazaak",
            "Inventory / Quest Items",
            "Inventory / Misc Items",
            "Weapons",
            "Weapons / Grenades",
            "Weapons / Lightsabers",
            "Weapons / Double-Bladed Lightsabers",
            "Weapons / Short Lightsabers",
            "Weapons / Lightsaber Crystals",
            "Weapons / Vibroblades",
            "Weapons / Double-Bladed Melee",
            "Weapons / Blasters",
            "Weapons / Heavy Blasters",
            "Weapons / Blaster Rifles",
            "Weapons / Heavy Weapons",
            "Weapons / Creature Weapons",
            "Weapons / Single-Handed Melee",
            "Weapons / Two-Handed Weapons",
            "Weapons / Misc Weapons",
            "Visual FX",
            *[
                f"Visual FX / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Visual FX"]
            ],
            "Visuals",
            *[
                f"Visuals / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Visuals"]
            ],
            "Planets",
            *[
                f"Planets / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Planets"]
            ],
            "Misc Models",
            *[
                f"Misc Models / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Misc Models"]
            ],
            "Stunts",
            *[
                f"Stunts / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Stunts"]
            ],
            "Uncategorised",
            "Templates",
            *[
                f"Templates / {subcategory}"
                for subcategory in MODEL_SUBCATEGORY_ORDER["Templates"]
            ],
            "Current Model",
        ])
        self.tag_filter.currentTextChanged.connect(self._apply_filter)
        self.recency_filter = QtWidgets.QComboBox()
        self.recency_filter.addItems(["Any Time", "Recent First"])
        self.recency_filter.currentTextChanged.connect(self._apply_filter)
        self.compat_filter = QtWidgets.QComboBox()
        self.compat_filter.addItems(["All Compatibility", "Current Game", "Cross-Game"])
        self.compat_filter.currentTextChanged.connect(self._apply_filter)
        for label, combo in (
            ("Asset Type", self.type_filter),
            ("Game", self.game_filter),
            ("Source", self.source_filter),
            ("Tags", self.tag_filter),
            ("Updated", self.recency_filter),
            ("Compatibility", self.compat_filter),
        ):
            self._make_combo_shrinkable(combo)
            column = len([None for _ in range(filters.count())]) % 3
            row = filters.count() // 3
            filters.addLayout(self._labeled_filter(label, combo), row, column)
        for column in range(3):
            filters.setColumnStretch(column, 1)
        layout.addLayout(filters)

        self.module_load_scope = QtWidgets.QComboBox()
        self.module_load_scope.setObjectName("contentBrowserModuleLoadScope")
        self.module_load_scope.addItem("Selected room only", False)
        self.module_load_scope.addItem("Entire connected map", True)
        self.module_load_scope.setToolTip(
            "Controls module rooms opened or added to the main scene. "
            "Selected room only uses less memory. Entire connected map loads all "
            "rooms in the module layout at their connected positions. "
            "Applies to the next load; existing scene objects are unchanged."
        )
        self._make_combo_shrinkable(self.module_load_scope)
        self.module_load_scope.currentIndexChanged.connect(
            lambda _index: self.loadConnectedMapChanged.emit(self.module_load_scope.currentData() is True)
        )
        layout.addLayout(self._labeled_filter("Module loading", self.module_load_scope))

    def set_load_connected_map(self, enabled: bool) -> None:
        with QtCore.QSignalBlocker(self.module_load_scope):
            self.module_load_scope.setCurrentIndex(1 if enabled is True else 0)

    def _labeled_filter(self, text: str, combo: QtWidgets.QComboBox) -> QtWidgets.QVBoxLayout:
        wrapper = QtWidgets.QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(2)
        label = QtWidgets.QLabel(text)
        label.setBuddy(combo)
        label.setMinimumWidth(0)
        combo.setAccessibleName(text)
        wrapper.addWidget(label)
        wrapper.addWidget(combo)
        return wrapper

    def _make_combo_shrinkable(self, combo: QtWidgets.QComboBox) -> None:
        combo.setMinimumWidth(0)
        combo.setMinimumContentsLength(1)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)

    def _build_action_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        self.primary_button = self._compact_action_button("Open")
        self.primary_button.setProperty("accent", True)
        self.primary_button.clicked.connect(self._activate_selected)
        layout.addWidget(self.primary_button)

        grid = QtWidgets.QGridLayout()
        self.preview_button = self._compact_action_button("Preview")
        self.stop_button = self._compact_action_button("Stop")
        self.apply_button = self._compact_action_button("Apply Animation")
        self.extract_button = self._compact_action_button("Extract")
        self.level_button = self._compact_action_button("Add to Scene")
        self.inspect_button = self._compact_action_button("Inspect")
        self.export_button = self._compact_action_button("Export")
        self.source_button = self._compact_action_button("Retarget Source")
        self.target_button = self._compact_action_button("Retarget Target")
        buttons = [
            self.preview_button,
            self.stop_button,
            self.apply_button,
            self.extract_button,
            self.level_button,
            self.inspect_button,
            self.export_button,
            self.source_button,
            self.target_button,
        ]
        for index, button in enumerate(buttons):
            button.setProperty("compact", True)
            grid.addWidget(button, index // 2, index % 2)
        self.preview_button.clicked.connect(lambda: self.libraryActionRequested.emit("Preview"))
        self.stop_button.clicked.connect(lambda: self.libraryActionRequested.emit("Stop"))
        self.apply_button.clicked.connect(lambda: self.libraryActionRequested.emit("Load"))
        self.extract_button.clicked.connect(self.extract_selected)
        self.level_button.clicked.connect(self.import_selected_to_level)
        self.inspect_button.clicked.connect(lambda: self.libraryActionRequested.emit("Inspect"))
        self.export_button.clicked.connect(lambda: self.libraryActionRequested.emit("Export"))
        self.source_button.clicked.connect(lambda: self._emit_retarget("source"))
        self.target_button.clicked.connect(lambda: self._emit_retarget("target"))
        layout.addLayout(grid)

    def _compact_action_button(self, text: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setProperty("compact", True)
        button.setProperty("_gr_full_text", text)
        button.setToolTip(text)
        button.setMinimumWidth(0)
        button.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        return button

    def set_rows(self, rows: list[dict]) -> None:
        self._library_rows = enrich_library_rows(rows)
        self._rebuild_assets()

    def set_rows_deferred(self, rows: list[dict], batch_size: int = 350) -> None:
        self._deferred_asset_generation += 1
        generation = self._deferred_asset_generation
        self._library_rows = [dict(row) for row in rows]
        self._assets = []
        self.asset_view.clear()
        self.count_label.setText(f"Preparing {len(self._library_rows)} asset(s)...")
        self.status_label.setText("Preparing Content Browser assets...")

        total = len(self._library_rows)
        batch = max(50, int(batch_size or 350))
        index = 0

        def build_next_batch() -> None:
            nonlocal index
            if generation != self._deferred_asset_generation:
                return
            end = min(total, index + batch)
            self._assets.extend(
                descriptor_from_library_row(row)
                for row in self._library_rows[index:end]
            )
            index = end
            self.count_label.setText(f"Preparing {index}/{total} asset(s)...")
            if index < total:
                QtCore.QTimer.singleShot(0, build_next_batch)
                return
            self._assets.extend(
                descriptor_from_animation_entry(entry)
                for entry in self._scene_animation_entries
            )
            self._assets.extend(
                descriptor_from_animation_entry(entry)
                for entry in self._scanned_animation_entries
            )
            self._rebuild_sources()
            self._rebuild_navigation()
            self._apply_filter()

        QtCore.QTimer.singleShot(0, build_next_batch)

    def set_animation_entries(self, entries: list[dict]) -> None:
        self.set_scene_animation_entries(entries)

    def set_scene_animation_entries(self, entries: list[dict]) -> None:
        self._scene_animation_entries = [dict(entry) for entry in entries]
        self._rebuild_assets()

    def set_scanned_animation_entries(self, entries: list[dict]) -> None:
        self._scanned_animation_entries = [dict(entry) for entry in entries]
        self._rebuild_assets()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def selected_asset(self) -> Optional[ContentAssetDescriptor]:
        item = self.asset_view.currentItem()
        return getattr(item, "asset", None) if item else None

    def selected_row(self) -> Optional[dict]:
        asset = self.selected_asset()
        if asset is None or asset.asset_type == "Animation":
            return None
        return dict(asset.row)

    def selected_entry(self) -> Optional[dict]:
        asset = self.selected_asset()
        if asset is None or asset.asset_type != "Animation":
            return None
        return dict(asset.row)

    def visible_assets(self) -> list[ContentAssetDescriptor]:
        assets = []
        for index in range(self.asset_view.topLevelItemCount()):
            asset = getattr(self.asset_view.topLevelItem(index), "asset", None)
            if asset is not None:
                assets.append(asset)
        return assets

    def visible_rows(self) -> list[dict]:
        rows = []
        for asset in self.visible_assets():
            if asset.asset_type != "Animation" and asset.row.get("resref"):
                rows.append(dict(asset.row))
        return rows

    def select_asset_type(self, asset_type: str) -> None:
        text = asset_type if asset_type in ASSET_TYPES else "All"
        self.type_filter.setCurrentText(text)
        self._select_navigation("type", text)

    def load_selected(self) -> None:
        row = self.selected_row()
        if row:
            self.loadRequested.emit(str(row.get("resref", "")), str(row.get("game", "")))

    def extract_selected(self) -> None:
        row = self.selected_row()
        if row:
            self.extractRequested.emit(row)

    def open_selected_as_new_scene(self) -> None:
        row = self.selected_row()
        if row:
            self.primarySceneLoadRequested.emit(row)

    def add_selected_to_current_scene(self) -> None:
        row = self.selected_row()
        if row:
            self.addToCurrentSceneRequested.emit(row)

    def open_selected_in_new_level_editor(self) -> None:
        row = self.selected_row()
        if row:
            self.levelEditorNewRequested.emit(row)

    def import_selected_to_level(self) -> None:
        row = self.selected_row()
        if row:
            self.levelEditorImportRequested.emit(row)

    def open_selected_in_character_builder(self) -> None:
        row = self.selected_row()
        if row:
            self.characterBuilderRequested.emit(row)

    def request_selected_asset_action(self, action: str) -> None:
        row = self.selected_row()
        if row:
            self.assetActionRequested.emit(action, row)

    def apply_ghost_theme(self, theme) -> None:
        return None

    def apply_ghost_layout(self, layout) -> None:
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        spacing = layout.spacing_value("panelSpacing", 5)
        for widget in (self, self.sidebar, self.details):
            widget_layout = widget.layout()
            if widget_layout is not None:
                widget_layout.setSpacing(spacing)
        self.splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
        self.sidebar_splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
        if not self._splitter_user_adjusted and not self._splitter_layout_applied:
            self._apply_initial_splitter_sizes()

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self._splitter_user_adjusted = True

    def _apply_initial_splitter_sizes(self) -> None:
        if self._splitter_user_adjusted or self._splitter_layout_applied:
            return
        width = max(1, self.splitter.width())
        if width < 120:
            return
        sidebar = max(136, min(300, int(width * 0.34)))
        center = max(96, width - sidebar)
        self.splitter.setSizes([sidebar, center])
        height = max(1, self.sidebar_splitter.height())
        if height >= 120:
            details = max(96, min(180, int(height * 0.36)))
            self.sidebar_splitter.setSizes([max(120, height - details), details])
        self._splitter_layout_applied = True

    def _rebuild_assets(self) -> None:
        self._assets = [
            *(descriptor_from_library_row(row) for row in self._library_rows),
            *(descriptor_from_animation_entry(entry) for entry in self._scene_animation_entries),
            *(descriptor_from_animation_entry(entry) for entry in self._scanned_animation_entries),
        ]
        self._rebuild_sources()
        self._rebuild_navigation()
        self._apply_filter()

    def _rebuild_sources(self) -> None:
        current = self.source_filter.currentText() if hasattr(self, "source_filter") else "All Sources"
        self.source_filter.blockSignals(True)
        self.source_filter.clear()
        self.source_filter.addItem("All Sources")
        sources = sorted({asset.source for asset in self._assets if asset.source})
        self.source_filter.addItems(sources)
        index = self.source_filter.findText(current)
        self.source_filter.setCurrentIndex(index if index >= 0 else 0)
        self.source_filter.blockSignals(False)

    def _rebuild_navigation(self) -> None:
        self.nav_tree.blockSignals(True)
        self.nav_tree.clear()
        all_item = QtWidgets.QTreeWidgetItem(["All Assets"])
        all_item.setData(0, QtCore.Qt.UserRole, ("type", "All"))
        self.nav_tree.addTopLevelItem(all_item)
        for asset_type in ASSET_TYPES[1:]:
            item = QtWidgets.QTreeWidgetItem([asset_type])
            item.setData(0, QtCore.Qt.UserRole, ("type", asset_type))
            self.nav_tree.addTopLevelItem(item)
        categories = sorted({asset.category for asset in self._assets if asset.category}, key=self._category_sort_key)
        if categories:
            folders = QtWidgets.QTreeWidgetItem(["Folders / Categories"])
            folders.setData(0, QtCore.Qt.UserRole, ("type", "All"))
            self.nav_tree.addTopLevelItem(folders)
            for category in categories:
                child = QtWidgets.QTreeWidgetItem([category])
                child.setData(0, QtCore.Qt.UserRole, ("category", category))
                folders.addChild(child)
                subcategories = sorted(
                    {
                        str(asset.metadata.get("subcategory") or "")
                        for asset in self._assets
                        if asset.category == category and asset.metadata.get("subcategory")
                    },
                    key=lambda value, cat=category: self._subcategory_sort_key(cat, value),
                )
                for subcategory in subcategories:
                    subchild = QtWidgets.QTreeWidgetItem([subcategory])
                    subchild.setData(0, QtCore.Qt.UserRole, ("subcategory", f"{category}\0{subcategory}"))
                    child.addChild(subchild)
            folders.setExpanded(True)
        self.nav_tree.blockSignals(False)
        self._select_navigation(*self._active_nav)

    def _select_navigation(self, key: str, value: str) -> None:
        target = (key, value)
        for item in self._walk_nav_items():
            if item.data(0, QtCore.Qt.UserRole) == target:
                self.nav_tree.setCurrentItem(item)
                return

    def _walk_nav_items(self):
        for index in range(self.nav_tree.topLevelItemCount()):
            root = self.nav_tree.topLevelItem(index)
            yield from self._walk_nav_branch(root)

    def _walk_nav_branch(self, item: QtWidgets.QTreeWidgetItem):
        yield item
        for child_index in range(item.childCount()):
            yield from self._walk_nav_branch(item.child(child_index))

    def _category_sort_key(self, category: str) -> tuple[int, str]:
        try:
            return (MODEL_CATEGORY_ORDER.index(category), category)
        except ValueError:
            return (len(MODEL_CATEGORY_ORDER), category)

    def _subcategory_sort_key(self, category: str, subcategory: str) -> tuple[int, str]:
        order = MODEL_SUBCATEGORY_ORDER.get(category, ())
        try:
            return (order.index(subcategory), subcategory)
        except ValueError:
            return (len(order), subcategory)

    def _on_navigation_changed(self) -> None:
        item = self.nav_tree.currentItem()
        data = item.data(0, QtCore.Qt.UserRole) if item is not None else ("type", "All")
        if not isinstance(data, tuple) or len(data) != 2:
            data = ("type", "All")
        self._active_nav = (str(data[0]), str(data[1]))
        if self._active_nav[0] == "type":
            self.type_filter.blockSignals(True)
            self.type_filter.setCurrentText(self._active_nav[1] if self._active_nav[1] in ASSET_TYPES else "All")
            self.type_filter.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self) -> None:
        if not hasattr(self, "asset_view"):
            return
        needle = self.search_edit.text().strip().lower()
        asset_type = self.type_filter.currentText()
        game = self.game_filter.currentText()
        source = self.source_filter.currentText()
        tag = self.tag_filter.currentText()
        compatibility = self.compat_filter.currentText()
        nav_key, nav_value = self._active_nav

        filtered = [
            asset
            for asset in self._assets
            if self._asset_matches_filters(
                asset,
                needle=needle,
                asset_type=asset_type,
                game=game,
                source=source,
                tag=tag,
                compatibility=compatibility,
                nav_key=nav_key,
                nav_value=nav_value,
            )
        ]
        if len(filtered) > 750:
            self._apply_filter_deferred(filtered)
            return
        self._replace_asset_items(filtered, resize_columns=True)

    def _asset_matches_filters(
        self,
        asset: ContentAssetDescriptor,
        *,
        needle: str,
        asset_type: str,
        game: str,
        source: str,
        tag: str,
        compatibility: str,
        nav_key: str,
        nav_value: str,
    ) -> bool:
        if asset_type != "All" and asset.asset_type != asset_type:
            return False
        if game != "All" and asset.game != game:
            return False
        if source != "All Sources" and asset.source != source:
            return False
        if nav_key == "category" and asset.category != nav_value:
            return False
        if nav_key == "subcategory":
            nav_category, _, nav_subcategory = nav_value.partition("\0")
            if asset.category != nav_category or asset.metadata.get("subcategory") != nav_subcategory:
                return False
        if tag != "All Tags" and not self._matches_tag(asset, tag):
            return False
        if compatibility == "Current Game" and game != "All" and asset.game and asset.game != game:
            return False
        if compatibility == "Cross-Game" and asset.game not in {"", "K1", "K2"}:
            return False
        return not (needle and needle not in asset.searchable_text)

    def _replace_asset_items(self, assets: list[ContentAssetDescriptor], *, resize_columns: bool) -> None:
        self.asset_view.setSortingEnabled(False)
        self.asset_view.clear()
        for asset in assets:
            item = QtContentAssetItem(asset)
            item.setIcon(0, self._asset_icon(asset))
            self.asset_view.addTopLevelItem(item)
        self.asset_view.setSortingEnabled(True)
        self.asset_view.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.count_label.setText(f"{self.asset_view.topLevelItemCount()} asset(s) shown")
        if resize_columns:
            for column in range(self.asset_view.columnCount()):
                self.asset_view.resizeColumnToContents(column)
        self._update_details()

    def _apply_filter_deferred(self, assets: list[ContentAssetDescriptor], batch_size: int = 300) -> None:
        self._deferred_filter_generation += 1
        generation = self._deferred_filter_generation
        self.asset_view.setSortingEnabled(False)
        self.asset_view.clear()
        self.count_label.setText(f"Showing 0/{len(assets)} asset(s)...")
        total = len(assets)
        batch = max(50, int(batch_size or 300))
        index = 0

        def add_next_batch() -> None:
            nonlocal index
            if generation != self._deferred_filter_generation:
                return
            end = min(total, index + batch)
            items = []
            for asset in assets[index:end]:
                item = QtContentAssetItem(asset)
                item.setIcon(0, self._asset_icon(asset))
                items.append(item)
            self.asset_view.addTopLevelItems(items)
            index = end
            self.count_label.setText(f"Showing {index}/{total} asset(s)...")
            if index < total:
                QtCore.QTimer.singleShot(0, add_next_batch)
                return
            self.asset_view.setSortingEnabled(True)
            self.asset_view.sortByColumn(0, QtCore.Qt.AscendingOrder)
            self.count_label.setText(f"{total} asset(s) shown")
            self._update_details()

        QtCore.QTimer.singleShot(0, add_next_batch)

    def _matches_tag(self, asset: ContentAssetDescriptor, tag: str) -> bool:
        if " / " in tag:
            category, subcategory = tag.split(" / ", 1)
            return (
                asset.category.lower() == category.lower()
                and str(asset.metadata.get("subcategory", "")).lower() == subcategory.lower()
            )
        haystack = " ".join([asset.category, asset.source, *asset.tags]).lower()
        mapping = {
            "Player Characters": "player characters",
            "Party Members": "party members",
            "Commoners": "commoners",
            "NPCs": "npcs",
            "Droids": "droids",
            "Turrets": "turrets",
            "Creatures": "creatures",
            "Holograms": "holograms",
            "Supermodels": "supermodels",
            "Modules": "module",
            "Level Assets": "level assets",
            "Environment": "environment",
            "Skyboxes": "skyboxes",
            "Minigame": "minigame",
            "Menus": "menus",
            "GUI": "gui",
            "Placeables": "placeables",
            "Doors": "doors",
            "Engine Items": "engine items",
            "Armor": "armor",
            "Armor / Clothing": "clothing",
            "Armor / Jedi Robes": "jedi robes",
            "Armor / Light Armor": "light armor",
            "Armor / Medium Armor": "medium armor",
            "Armor / Heavy Armor": "heavy armor",
            "Armor / Environmental Suits": "environmental suits",
            "Armor / Disguises": "disguises",
            "Armor / Misc Armor": "misc armor",
            "Inventory": "inventory",
            "Inventory / Security Spikes": "security spikes",
            "Inventory / Computer Spikes": "computer spikes",
            "Inventory / Parts": "parts",
            "Inventory / Mines": "mines",
            "Inventory / Spikes": "spikes",
            "Inventory / Traps": "traps",
            "Inventory / Medkits": "medkits",
            "Inventory / Masks": "masks",
            "Inventory / Implants": "implants",
            "Inventory / Gauntlets": "gauntlets",
            "Inventory / Armbands": "armbands",
            "Inventory / Droid Items": "droid items",
            "Inventory / Belts": "belts",
            "Inventory / Stims": "stims",
            "Inventory / Adrenal Stims": "adrenal stims",
            "Inventory / Combat Shots": "combat shots",
            "Inventory / Credits": "credits",
            "Inventory / Upgrades": "upgrades",
            "Inventory / Datapads": "datapads",
            "Inventory / Pazaak": "pazaak",
            "Inventory / Quest Items": "quest items",
            "Inventory / Misc Items": "misc items",
            "Weapons": "weapons",
            "Weapons / Grenades": "grenades",
            "Weapons / Lightsabers": "lightsabers",
            "Weapons / Double-Bladed Lightsabers": "double-bladed lightsabers",
            "Weapons / Short Lightsabers": "short lightsabers",
            "Weapons / Lightsaber Crystals": "lightsaber crystals",
            "Weapons / Vibroblades": "vibroblades",
            "Weapons / Double-Bladed Melee": "double-bladed melee",
            "Weapons / Blasters": "blasters",
            "Weapons / Heavy Blasters": "heavy blasters",
            "Weapons / Blaster Rifles": "blaster rifles",
            "Weapons / Heavy Weapons": "heavy weapons",
            "Weapons / Creature Weapons": "creature weapons",
            "Weapons / Single-Handed Melee": "single-handed melee",
            "Weapons / Two-Handed Weapons": "two-handed weapons",
            "Weapons / Misc Weapons": "misc weapons",
            "Placeables / Containers": "containers",
            "Placeables / Computers & Panels": "computers & panels",
            "Placeables / Doors & Transitions": "doors & transitions",
            "Placeables / Furniture": "furniture",
            "Placeables / Lights & VFX": "lights & vfx",
            "Placeables / Traps & Hazards": "traps & hazards",
            "Placeables / Environmental Props": "environmental props",
            "Placeables / Misc Placeables": "misc placeables",
            "Doors / Taris": "taris",
            "Doors / Dantooine": "dantooine",
            "Doors / Tatooine": "tatooine",
            "Doors / Kashyyyk": "kashyyyk",
            "Doors / Manaan": "manaan",
            "Doors / Korriban": "korriban",
            "Doors / Leviathan": "leviathan",
            "Doors / Star Forge": "star forge",
            "Doors / Rakata": "rakata",
            "Doors / Yavin": "yavin",
            "Doors / Endar Spire": "endar spire",
            "Doors / Ebon Hawk": "ebon hawk",
            "Doors / Peragus": "peragus",
            "Doors / Telos": "telos",
            "Doors / Harbinger": "harbinger",
            "Doors / Nar Shaddaa": "nar shaddaa",
            "Doors / Dxun": "dxun",
            "Doors / Onderon": "onderon",
            "Doors / Malachor": "malachor",
            "Doors / Ravager": "ravager",
            "Doors / Droid Planet": "droid planet",
            "Doors / Force Fields": "force fields",
            "Doors / Generic Doors": "generic doors",
            "Doors / Unknown Doors": "unknown doors",
            "Visual FX": "visual fx",
            "Visuals": "visuals",
            "Planets": "planets",
            "Misc Models": "misc models",
            "Stunts": "stunts",
            "Uncategorised": "uncategorised",
            "Templates": "template",
            "Current Model": "current model",
        }
        return mapping.get(tag, tag).lower() in haystack

    def _asset_icon(self, asset: ContentAssetDescriptor):
        if asset.asset_type == "Animation":
            return icon("anims", 16)
        if asset.asset_type == "Module":
            return icon("cat_module", 16)
        if asset.asset_type == "Blueprint":
            return icon("skeleton", 16)
        if asset.category in {"Creatures", "Droids", "Turrets", "Holograms"}:
            return icon("cat_creature", 16)
        if asset.category in {"NPCs", "Party Members", "Player Characters", "Commoners", "Supermodels"}:
            return icon("cat_character", 16)
        if asset.category in {"Armor", "Inventory", "Weapons", "Placeables", "Item/Armor/Weapons"}:
            return icon("cat_item", 16)
        if asset.category in {
            "Environment", "Doors", "Engine Items", "Visual FX", "Visuals", "Planets",
            "Misc Models", "Stunts", "Skyboxes", "Minigame", "Menus", "GUI",
            "Level Assets", "Uncategorised",
        }:
            return icon("cat_other", 16)
        return icon("library", 16)

    def _update_details(self) -> None:
        asset = self.selected_asset()
        if asset is None:
            self.detail_title.setText("Select an asset")
            self.detail_text.setPlainText("")
            self._set_action_state(None)
            return
        self.detail_title.setText(asset.display_name or asset.name)
        lines = [
            f"Asset Name: {asset.name}",
            f"Type: {asset.asset_type}",
            f"Game: {asset.game or 'Any'}",
            f"Category: {asset.category or 'Uncategorized'}",
        ]
        if asset.source:
            lines.append(f"Source: {asset.source}")
        for key, value in asset.metadata.items():
            if value not in ("", None):
                lines.append(f"{key.title()}: {value}")
        self.detail_text.setPlainText("\n".join(lines))
        self._set_action_state(asset)

    def _set_action_state(self, asset: Optional[ContentAssetDescriptor]) -> None:
        is_animation = asset is not None and asset.asset_type == "Animation"
        has_model_row = asset is not None and asset.asset_type != "Animation" and bool(asset.row.get("resref"))
        self.primary_button.setText("Preview" if is_animation else "Add")
        self.primary_button.setEnabled(asset is not None)
        self.preview_button.setEnabled(is_animation)
        self.stop_button.setEnabled(True)
        self.apply_button.setEnabled(is_animation)
        self.export_button.setEnabled(is_animation)
        self.inspect_button.setEnabled(asset is not None)
        for button in (self.extract_button, self.level_button, self.source_button, self.target_button):
            button.setEnabled(has_model_row)

    def _asset_supports_character_builder(self, asset: ContentAssetDescriptor) -> bool:
        return asset.asset_type == "Model" and asset.category in {"Player Characters", "NPCs"}

    def _activate_selected(self) -> None:
        asset = self.selected_asset()
        if asset is None:
            return
        if asset.asset_type == "Animation":
            self.libraryActionRequested.emit("Preview")
            return
        row = asset.row
        if row.get("resref"):
            self.addToCurrentSceneRequested.emit(dict(row))

    def _context_action(
        self,
        menu: QtWidgets.QMenu,
        text: str,
        icon_name: str,
        action_id: str,
    ):
        action = menu.addAction(icon(icon_name, 22), text)
        action.setData(action_id)
        return action

    def _build_model_context_menu(self, asset: ContentAssetDescriptor) -> tuple[QtWidgets.QMenu, dict[str, object]]:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("QMenu { icon-size: 22px; }")
        actions: dict[str, object] = {}
        actions["open_new_scene"] = self._context_action(menu, "Open as New Scene", "new_scene", "open_new_scene")
        actions["add_current_scene"] = self._context_action(menu, "Add to Current Scene", "scene_add", "add_current_scene")
        actions["level_new"] = self._context_action(menu, "Open In Level Editor (New)", "level_editor_new", "level_new")
        actions["level_existing"] = self._context_action(
            menu,
            "Add to Level Editor (Existing Level)",
            "level_editor_add",
            "level_existing",
        )
        if self._asset_supports_character_builder(asset):
            actions["character_builder_new"] = self._context_action(
                menu,
                "Open in Character Builder (New)",
                "charbuilder",
                "character_builder_new",
            )
        menu.addSeparator()
        asset_menu = QtWidgets.QMenu("Asset Actions", menu)
        asset_menu.setIcon(icon("asset_actions", 22))
        menu.addMenu(asset_menu)
        menu._gr_asset_actions_menu = asset_menu
        asset_menu.setStyleSheet("QMenu { icon-size: 22px; }")
        actions["extract_mdl_mdx"] = self._context_action(
            asset_menu,
            "Extract (.MDL/.MDX)",
            "extract_mdl_mdx",
            "extract_mdl_mdx",
        )
        actions["extract_fbx_openfbx"] = self._context_action(
            asset_menu,
            "Extract (.FBX) (openfbx)",
            "extract_fbx_openfbx",
            "extract_fbx_openfbx",
        )
        actions["extract_fbx_autodesk"] = self._context_action(
            asset_menu,
            "Extract (.FBX) (autodesk if installed)",
            "extract_fbx_autodesk",
            "extract_fbx_autodesk",
        )
        menu.addSeparator()
        actions["retarget_source"] = self._context_action(
            menu,
            "Send to Retarget Workbench (Source)",
            "anims",
            "retarget_source",
        )
        actions["retarget_target"] = self._context_action(
            menu,
            "Send to Retarget Workbench (Target)",
            "rig",
            "retarget_target",
        )
        return menu, actions

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.asset_view.itemAt(pos)
        if item is not None:
            self.asset_view.setCurrentItem(item)
        asset = self.selected_asset()
        if asset is None:
            return
        menu = QtWidgets.QMenu(self)
        if asset.asset_type == "Animation":
            preview_action = menu.addAction("Preview Animation")
            stop_action = menu.addAction("Stop Preview")
            load_action = menu.addAction("Load in Current Animations")
            export_action = menu.addAction("Export Animation")
            chosen = menu.exec(self.asset_view.mapToGlobal(pos))
            if chosen is preview_action:
                self.libraryActionRequested.emit("Preview")
            elif chosen is stop_action:
                self.libraryActionRequested.emit("Stop")
            elif chosen is load_action:
                self.libraryActionRequested.emit("Load")
            elif chosen is export_action:
                self.libraryActionRequested.emit("Export")
            return
        menu, actions = self._build_model_context_menu(asset)
        chosen = menu.exec(self.asset_view.mapToGlobal(pos))
        action_id = chosen.data() if chosen is not None else ""
        if action_id == "open_new_scene":
            self.open_selected_as_new_scene()
        elif action_id == "add_current_scene":
            self.add_selected_to_current_scene()
        elif action_id == "level_new":
            self.open_selected_in_new_level_editor()
        elif action_id == "level_existing":
            self.import_selected_to_level()
        elif action_id == "character_builder_new":
            self.open_selected_in_character_builder()
        elif action_id == "extract_mdl_mdx":
            self.extract_selected()
        elif action_id in {"extract_fbx_openfbx", "extract_fbx_autodesk"}:
            self.request_selected_asset_action(str(action_id))
        elif action_id == "retarget_source":
            self._emit_retarget("source")
        elif action_id == "retarget_target":
            self._emit_retarget("target")

    def _emit_retarget(self, role: str) -> None:
        row = self.selected_row()
        if not row:
            return
        if role == "source":
            self.retargetSourceRequested.emit(row)
        else:
            self.retargetTargetRequested.emit(row)
