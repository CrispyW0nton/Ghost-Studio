"""Focused contracts for installed-game BAS character preset mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _configure_native_python_roots() -> None:
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for item in reversed(_python_roots(ROOT)):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)


def _table_bytes(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> bytes:
    from src.core.scripting.data_authoring import TwoDADocument

    return TwoDADocument(
        columns,
        tuple(str(index) for index in range(len(rows))),
        rows,
    ).to_bytes()


class _Install:
    def __init__(self, rows: dict[int, tuple[str, ...]]) -> None:
        self.rows = rows
        self.game_dir = ""

    def list_resrefs(self, resource_type: int):
        return self.rows.get(int(resource_type), ())


class _PresetManager:
    revision = 41

    def __init__(self) -> None:
        from pykotor.common.language import LocalizedString
        from pykotor.common.misc import EquipmentSlot, Game, InventoryItem, ResRef
        from pykotor.resource.generics.utc import UTC, bytes_utc
        from pykotor.resource.generics.uti import UTI, bytes_uti
        from src.core.assets.resource_manager import RES_UTC, RES_UTI

        carth = UTC()
        carth.resref = ResRef("k_carth")
        carth.tag = "Carth"
        carth.first_name = LocalizedString.from_english("Carth Onasi")
        carth.appearance_id = 0
        carth.alignment = 50
        carth.equipment[EquipmentSlot.ARMOR] = InventoryItem(ResRef("carth_armor"))
        carth.equipment[EquipmentSlot.RIGHT_HAND] = InventoryItem(ResRef("carth_blaster"))
        carth.equipment[EquipmentSlot.HEAD] = InventoryItem(ResRef("carth_mask"))

        drexl = UTC()
        drexl.resref = ResRef("c_drexlf")
        drexl.tag = "Drexl"
        drexl.first_name = LocalizedString.from_english("Drexl")
        drexl.appearance_id = 0

        armor = UTI()
        armor.resref = ResRef("carth_armor")
        armor.base_item = 0
        armor.texture_variation = 2
        blaster = UTI()
        blaster.resref = ResRef("carth_blaster")
        blaster.base_item = 1
        blaster.model_variation = 3
        mask = UTI()
        mask.resref = ResRef("carth_mask")
        mask.base_item = 2
        mask.model_variation = 2

        self.resources = {
            ("K1", "k_carth", RES_UTC): bytes_utc(carth, Game.K1),
            ("K2", "c_drexlf", RES_UTC): bytes_utc(drexl, Game.K2),
            ("K1", "carth_armor", RES_UTI): bytes_uti(armor, Game.K1),
            ("K1", "carth_blaster", RES_UTI): bytes_uti(blaster, Game.K1),
            ("K1", "carth_mask", RES_UTI): bytes_uti(mask, Game.K1),
        }
        self.tables = {
            ("K1", "appearance"): _table_bytes(
                (
                    "label",
                    "modeltype",
                    "modela",
                    "modelb",
                    "texa",
                    "texb",
                    "normalhead",
                    "race",
                    "racetex",
                ),
                (
                    (
                        "Carth",
                        "B",
                        "p_carthbase",
                        "p_carthbb",
                        "p_cartha",
                        "p_carthb",
                        "0",
                        "****",
                        "****",
                    ),
                ),
            ),
            ("K1", "heads"): _table_bytes(
                ("head", "headtexe"),
                (("p_carthh", "p_carthdark"),),
            ),
            ("K1", "baseitems"): _table_bytes(
                ("bodyvar", "defaultmodel"),
                (("b", "****"), ("****", "w_blstrpstl_001"), ("****", "i_mask_001")),
            ),
            ("K2", "appearance"): _table_bytes(
                ("label", "modeltype", "modela", "normalhead", "race", "racetex"),
                (("Drexl", "F", "****", "****", "c_drexlf", "c_drexlf01"),),
            ),
            ("K2", "heads"): _table_bytes(("head",), ()),
            ("K2", "baseitems"): _table_bytes(("bodyvar", "defaultmodel"), (("****", "****"),)),
        }
        self._k1 = _Install({RES_UTC: ("k_carth",)})
        self._k2 = _Install({RES_UTC: ("c_drexlf",)})

    def get_k1(self):
        return self._k1

    def get_k2(self):
        return self._k2

    def list_models(self, game: str = "all"):
        rows = [
            ("p_carthbb", "K1"),
            ("p_carthh", "K1"),
            ("w_blstrpstl_003", "K1"),
            ("i_mask_002", "K1"),
            ("c_drexlf", "K2"),
        ]
        return rows if game == "all" else [row for row in rows if row[1] == game]

    def get_strict(self, name, resource_type, game):
        from src.core.assets.resource_manager import RES_2DA, RES_TPC

        key = (str(game).upper(), str(name).lower())
        if int(resource_type) == RES_2DA:
            return self.tables.get(key)
        if int(resource_type) == RES_TPC and key in {
            ("K1", "p_carthb02"),
            ("K1", "p_candb02"),
            ("K1", "p_canddark"),
        }:
            return b"texture"
        return self.resources.get((key[0], key[1], int(resource_type)))


class _CrossGamePresetManager(_PresetManager):
    """K2 legacy UTCs whose canonical visible assets are split across games."""

    def __init__(self) -> None:
        super().__init__()
        from pykotor.common.language import LocalizedString
        from pykotor.common.misc import EquipmentSlot, Game, InventoryItem, ResRef
        from pykotor.resource.generics.utc import UTC, bytes_utc
        from pykotor.resource.generics.uti import UTI, bytes_uti
        from src.core.assets.resource_manager import RES_UTC, RES_UTI

        canderous = UTC()
        canderous.resref = ResRef("k2_cand")
        canderous.tag = "Cand"
        canderous.first_name = LocalizedString.from_english("Canderous")
        canderous.appearance_id = 1
        canderous.alignment = 25
        canderous.equipment[EquipmentSlot.ARMOR] = InventoryItem(ResRef("cand_armor"))
        canderous.equipment[EquipmentSlot.RIGHT_HAND] = InventoryItem(ResRef("cand_blaster"))

        pike_guard = UTC()
        pike_guard.resref = ResRef("k2_pike_guard")
        pike_guard.tag = "PikeGuard"
        pike_guard.first_name = LocalizedString.from_english("Pike Guard")
        pike_guard.appearance_id = 2
        pike_guard.equipment[EquipmentSlot.RIGHT_HAND] = InventoryItem(ResRef("pike_weapon"))

        broken = UTC()
        broken.resref = ResRef("k1_broken")
        broken.tag = "Broken"
        broken.first_name = LocalizedString.from_english("Broken Template")
        broken.appearance_id = 1

        headless = UTC()
        headless.resref = ResRef("k1_headless")
        headless.tag = "Headless"
        headless.first_name = LocalizedString.from_english("Incomplete Headless Template")
        headless.appearance_id = 2

        mislabeled_full = UTC()
        mislabeled_full.resref = ResRef("k2_mislabeled_full")
        mislabeled_full.tag = "MislabeledFull"
        mislabeled_full.first_name = LocalizedString.from_english("Mislabeled Full Body")
        mislabeled_full.appearance_id = 3

        cand_blaster = UTI()
        cand_blaster.resref = ResRef("cand_blaster")
        cand_blaster.base_item = 1
        cand_blaster.model_variation = 3
        cand_armor = UTI()
        cand_armor.resref = ResRef("cand_armor")
        cand_armor.base_item = 0
        cand_armor.texture_variation = 2
        pike_weapon = UTI()
        pike_weapon.resref = ResRef("pike_weapon")
        pike_weapon.base_item = 0
        pike_weapon.model_variation = 1

        self.resources.update(
            {
                ("K2", "k2_cand", RES_UTC): bytes_utc(canderous, Game.K2),
                ("K2", "k2_pike_guard", RES_UTC): bytes_utc(pike_guard, Game.K2),
                ("K1", "k1_broken", RES_UTC): bytes_utc(broken, Game.K1),
                ("K1", "k1_headless", RES_UTC): bytes_utc(headless, Game.K1),
                ("K2", "k2_mislabeled_full", RES_UTC): bytes_utc(mislabeled_full, Game.K2),
                ("K1", "cand_blaster", RES_UTI): bytes_uti(cand_blaster, Game.K1),
                ("K2", "cand_armor", RES_UTI): bytes_uti(cand_armor, Game.K2),
                ("K2", "pike_weapon", RES_UTI): bytes_uti(pike_weapon, Game.K2),
            }
        )
        self.tables[("K1", "appearance")] = _table_bytes(
            (
                "label",
                "modeltype",
                "modela",
                "modelb",
                "texa",
                "texb",
                "normalhead",
                "race",
                "racetex",
            ),
            (
                (
                    "Carth",
                    "B",
                    "p_carthbase",
                    "p_carthbb",
                    "p_cartha",
                    "p_carthb",
                    "0",
                    "****",
                    "****",
                ),
                ("Broken", "F", "****", "****", "****", "****", "****", "****", "****"),
                (
                    "Headless",
                    "B",
                    "p_carthbb",
                    "p_carthbb",
                    "p_carthb",
                    "p_carthb",
                    "99",
                    "****",
                    "****",
                ),
            ),
        )
        self.tables[("K2", "appearance")] = _table_bytes(
            ("label", "modeltype", "modela", "texa", "normalhead", "race", "racetex"),
            (
                ("Drexl", "F", "****", "****", "****", "c_drexlf", "c_drexlf01"),
                ("Canderous", "B", "p_candbb", "p_candb", "0", "****", "****"),
                ("PikeGuard", "F", "****", "****", "****", "c_drexlf", "c_drexlf01"),
                ("MislabeledFull", "B", "c_drexlf", "****", "****", "c_drexlf", "c_drexlf01"),
            ),
        )
        self.tables[("K2", "heads")] = _table_bytes(
            ("head", "headtexve"),
            (("p_candh", "p_canddark"),),
        )
        self.tables[("K2", "baseitems")] = _table_bytes(
            ("bodyvar", "defaultmodel"),
            (("a", "w_forcepike_01"),),
        )
        self._k1 = _Install({RES_UTC: ("k_carth", "k1_broken", "k1_headless")})
        self._k2 = _Install(
            {RES_UTC: ("c_drexlf", "k2_cand", "k2_pike_guard", "k2_mislabeled_full")}
        )

    def list_models(self, game: str = "all"):
        rows = super().list_models("all") + [
            ("p_candbb", "K1"),
            ("p_candh", "K1"),
            ("w_forcepike_001", "K2"),
        ]
        return rows if game == "all" else [row for row in rows if row[1] == game]


def test_catalog_resolves_k1_humanoid_and_k2_full_body_from_game_data() -> None:
    _configure_native_python_roots()
    from src.systems.bas.character_presets import build_bas_character_preset_catalog

    catalog = build_bas_character_preset_catalog(_PresetManager())

    assert catalog.counts_by_game == {"K1": 1, "K2": 1}
    carth = catalog.get("K1", "k_carth")
    assert carth is not None
    assert carth.display_name == "Carth Onasi"
    assert carth.composition_mode == "headless_body"
    assert carth.body_resref == "p_carthbb"
    assert carth.body_texture == "p_carthb02"
    assert carth.head_resref == "p_carthh"
    assert carth.slot_model_map == {
        "mask": "i_mask_002",
        "right_weapon": "w_blstrpstl_003",
    }
    assert carth.equipment_item_map["armor"] == "carth_armor"

    drexl = catalog.get("K2", "c_drexlf")
    assert drexl is not None
    assert drexl.composition_mode == "full_body"
    assert drexl.body_resref == "c_drexlf"
    assert drexl.head_resref == ""


def test_catalog_resolves_cross_game_legacy_characters_with_layer_provenance() -> None:
    _configure_native_python_roots()
    from src.systems.bas.character_presets import build_bas_character_preset_catalog

    catalog = build_bas_character_preset_catalog(_CrossGamePresetManager())

    canderous = catalog.get("K2", "k2_cand")
    assert canderous is not None
    assert canderous.loadable
    assert canderous.body_resref == "p_candbb"
    assert canderous.body_game == "K1"
    assert canderous.body_texture == "p_candb02"
    assert canderous.head_resref == "p_candh"
    assert canderous.head_game == "K1"
    assert canderous.head_texture == "p_canddark"
    assert canderous.slot_model_map == {"right_weapon": "w_blstrpstl_003"}
    assert canderous.slot_game_map == {"right_weapon": "K1"}
    assert not canderous.warnings


def test_catalog_normalizes_retail_weapon_model_padding() -> None:
    _configure_native_python_roots()
    from src.systems.bas.character_presets import build_bas_character_preset_catalog

    catalog = build_bas_character_preset_catalog(_CrossGamePresetManager())
    pike_guard = catalog.get("K2", "k2_pike_guard")
    assert pike_guard is not None
    assert pike_guard.slot_model_map == {"right_weapon": "w_forcepike_001"}
    assert pike_guard.slot_game_map == {"right_weapon": "K2"}
    assert not pike_guard.warnings


def test_catalog_omits_templates_without_a_model_in_either_game() -> None:
    _configure_native_python_roots()
    from src.systems.bas.character_presets import build_bas_character_preset_catalog

    catalog = build_bas_character_preset_catalog(_CrossGamePresetManager())
    assert catalog.get("K1", "k1_broken") is None
    assert catalog.get("K1", "k1_headless") is None
    mislabeled_full = catalog.get("K2", "k2_mislabeled_full")
    assert mislabeled_full is not None
    assert mislabeled_full.composition_mode == "full_body"
    assert all(preset.loadable for preset in catalog.presets)


def test_uti_reader_recovers_scalar_fields_when_a_localized_string_is_invalid(monkeypatch) -> None:
    _configure_native_python_roots()
    import pykotor.resource.generics.uti as uti_module
    from src.systems.bas.character_presets import _read_uti

    manager = _CrossGamePresetManager()

    def reject_standard_reader(_payload):
        raise ValueError("No encoding defined for language: UNSET")

    monkeypatch.setattr(uti_module, "read_uti", reject_standard_reader)
    recovered = _read_uti(manager, "cand_blaster", "K1")

    assert recovered is not None
    assert recovered.base_item == 1
    assert recovered.model_variation == 3


def test_panel_character_mode_filters_games_and_emits_selected_preset() -> None:
    _configure_native_python_roots()
    from PySide6 import QtWidgets
    from src.gui.qt_lib.panels.qt_body_attachment_panel import QtBodyAttachmentPanel
    from src.systems.bas.character_presets import build_bas_character_preset_catalog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = QtBodyAttachmentPanel()
    assert panel.character_preset_group.isHidden()

    catalog = build_bas_character_preset_catalog(_PresetManager())
    panel._handle_character_catalog_ready(catalog, 41)
    panel.set_mode("character_preset")
    assert not panel.character_preset_group.isHidden()
    assert panel.character_combo.count() == 2
    assert panel.apply_character_preset_button.isEnabled()

    panel.character_game_combo.setCurrentIndex(panel.character_game_combo.findData("K2"))
    assert panel.character_combo.count() == 1
    assert panel.selected_character_preset().game == "K2"
    emitted: list[object] = []
    panel.characterPresetRequested.connect(emitted.append)
    panel._emit_character_preset()
    assert emitted and emitted[0].utc_resref == "c_drexlf"


def test_main_bas_workflow_replaces_old_layers_with_preset_layers() -> None:
    _configure_native_python_roots()
    from PySide6 import QtWidgets
    from src.gui.qt_lib.panels.qt_body_attachment_panel import QtBodyAttachmentPanel
    from src.gui.windows.application_core.shared.bas_workflow import BasWorkflowMixin
    from src.systems.bas.character_presets import BasCharacterPreset

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class _Harness(BasWorkflowMixin):
        def __init__(self) -> None:
            self.body_attachment_panel = QtBodyAttachmentPanel()
            self._bas_attachments = {"mask": SimpleNamespace(name="old")}
            self._bas_attachment_resrefs = {"mask": "old"}
            self._bas_attachment_transforms = {"mask": {}}
            self._bas_body_model = None
            self._current_model = None
            self._current_head_model = None
            self._current_attachment_model = None
            self._bas_active_build_name = ""
            self._bas_mode = "headless_body"
            self.models = {
                "p_carthbb": SimpleNamespace(name="p_carthbb", all_nodes=lambda: []),
                "p_carthh": SimpleNamespace(name="p_carthh", all_nodes=lambda: []),
                "w_blstrpstl_003": SimpleNamespace(name="w_blstrpstl_003", all_nodes=lambda: []),
            }

        def _load_bas_attachment_model(self, resref, *, game="", strict=False):
            assert game == "K1" and strict
            return self.models.get(resref)

        def _adopt_bas_body_model(self, model, resref, game):
            self._bas_body_model = model
            self._current_model = model
            self.body_attachment_panel.set_body_model(model, resref=resref, game=game)

        def _rebuild_bas_preview(self):
            return "BAS preview updated."

        def _refresh_bas_animation_panel_after_layer_change(self, _slot):
            return None

    preset = BasCharacterPreset(
        game="K1",
        utc_resref="k_carth",
        display_name="Carth Onasi",
        tag="Carth",
        appearance_id=0,
        appearance_label="Carth",
        modeltype="B",
        body_resref="p_carthbb",
        head_resref="p_carthh",
        slot_models=(("right_weapon", "w_blstrpstl_003"),),
        equipment_items=(("right_hand", "carth_blaster"),),
    )
    harness = _Harness()
    harness._handle_bas_character_preset_requested(preset)

    assert harness._bas_attachment_resrefs == {
        "right_weapon": "w_blstrpstl_003",
        "head": "p_carthh",
    }
    assert harness.body_attachment_panel.selected_mode() == "character_preset"
    assert harness.body_attachment_panel.layer_rows()[1][:2] == ("HEAD", "p_carthh")
    assert harness._bas_body_model._gr_bas_character_preset["utc_resref"] == "k_carth"
    assert (
        "Loaded canonical K1 character Carth Onasi"
        in harness.body_attachment_panel.status.toPlainText()
    )


def test_main_bas_workflow_loads_k2_canderous_from_explicit_k1_layer_sources() -> None:
    _configure_native_python_roots()
    from PySide6 import QtWidgets
    from src.gui.qt_lib.panels.qt_body_attachment_panel import QtBodyAttachmentPanel
    from src.gui.windows.application_core.shared.bas_workflow import BasWorkflowMixin
    from src.systems.bas.character_presets import BasCharacterPreset

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class _Harness(BasWorkflowMixin):
        def __init__(self) -> None:
            self.body_attachment_panel = QtBodyAttachmentPanel()
            self._bas_attachments = {}
            self._bas_attachment_resrefs = {}
            self._bas_attachment_transforms = {}
            self._bas_body_model = None
            self._current_model = None
            self._current_head_model = None
            self._current_attachment_model = None
            self._bas_active_build_name = ""
            self._bas_mode = "headless_body"
            self.load_calls: list[tuple[str, str, bool]] = []
            self.models = {
                ("K1", "p_candbb"): SimpleNamespace(
                    name="p_candbb", _gr_source_game="K1", all_nodes=lambda: []
                ),
                ("K1", "p_candh"): SimpleNamespace(
                    name="p_candh", _gr_source_game="K1", all_nodes=lambda: []
                ),
                ("K1", "w_hvrptbltr_002"): SimpleNamespace(
                    name="w_hvrptbltr_002", _gr_source_game="K1", all_nodes=lambda: []
                ),
            }

        def _load_bas_attachment_model(self, resref, *, game="", strict=False):
            self.load_calls.append((resref, game, strict))
            return self.models.get((game, resref))

        def _adopt_bas_body_model(self, model, resref, game):
            self._bas_body_model = model
            self._current_model = model
            self.body_attachment_panel.set_body_model(model, resref=resref, game=game)

        def _rebuild_bas_preview(self):
            return "BAS preview updated."

        def _refresh_bas_animation_panel_after_layer_change(self, _slot):
            return None

    preset = BasCharacterPreset(
        game="K2",
        utc_resref="p_cand",
        display_name="Canderous",
        tag="Cand",
        appearance_id=5,
        appearance_label="Party_NPC_Cand",
        modeltype="B",
        body_resref="p_candbb",
        body_game="K1",
        head_resref="p_candh",
        head_game="K1",
        slot_models=(("right_weapon", "w_hvrptbltr_002"),),
        slot_games=(("right_weapon", "K1"),),
        equipment_items=(("right_hand", "g_w_hvrptbltr002"),),
    )
    harness = _Harness()

    harness._handle_bas_character_preset_requested(preset)

    assert harness.load_calls == [
        ("p_candbb", "K1", True),
        ("w_hvrptbltr_002", "K1", True),
        ("p_candh", "K1", True),
    ]
    assert harness._bas_attachment_resrefs == {
        "right_weapon": "w_hvrptbltr_002",
        "head": "p_candh",
    }
    assert harness._bas_body_model._gr_bas_character_preset["game"] == "K2"
    assert harness._bas_body_model._gr_bas_character_preset["body_game"] == "K1"
    assert (
        "Loaded canonical K2 character Canderous"
        in harness.body_attachment_panel.status.toPlainText()
    )


def test_main_bas_workflow_keeps_engine_only_inventory_from_breaking_creatures() -> None:
    _configure_native_python_roots()
    from PySide6 import QtWidgets
    from src.gui.qt_lib.panels.qt_body_attachment_panel import QtBodyAttachmentPanel
    from src.gui.windows.application_core.shared.bas_workflow import BasWorkflowMixin
    from src.systems.bas.character_presets import BasCharacterPreset

    from src.core.geometry import model_data as md

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    body = md.KotorModel(
        name="turret",
        root_node=md.ModelNode(name="turret", flags=int(md.NodeFlags.HEADER)),
    )
    weapon = md.KotorModel(
        name="w_blstrpstl_001",
        root_node=md.ModelNode(name="blaster", flags=int(md.NodeFlags.HEADER)),
    )
    weapon._gr_source_resref = "w_blstrpstl_001"

    class _Harness(BasWorkflowMixin):
        def __init__(self) -> None:
            self.body_attachment_panel = QtBodyAttachmentPanel()
            self._bas_attachments = {}
            self._bas_attachment_resrefs = {}
            self._bas_attachment_transforms = {}
            self._bas_body_model = None
            self._current_model = None
            self._current_head_model = None
            self._current_attachment_model = None
            self._bas_active_build_name = ""
            self._bas_mode = "full_body"
            self.preview = None

        def _load_bas_attachment_model(self, resref, *, game="", strict=False):
            assert game == "K1" and strict
            return {"turret": body, "w_blstrpstl_001": weapon}.get(resref)

        def _adopt_bas_body_model(self, model, resref, game):
            self._bas_body_model = model
            self._current_model = model

        def _apply_bas_preview_to_viewport(self, preview, *, previous_preview=None):
            self.preview = preview

        def _refresh_bas_animation_panel_after_layer_change(self, _slot):
            return None

    preset = BasCharacterPreset(
        game="K1",
        utc_resref="g_turretbl001",
        display_name="Blaster Turret",
        tag="turret",
        appearance_id=0,
        appearance_label="Turret",
        modeltype="F",
        body_resref="turret",
        slot_models=(("right_weapon", "w_blstrpstl_001"),),
        equipment_items=(("right_hand", "g_w_blstrpstl01"),),
    )
    harness = _Harness()

    harness._handle_bas_character_preset_requested(preset)

    assert harness.preview is not None
    assert harness._bas_attachments == {}
    assert harness._bas_attachment_resrefs == {}
    assert body._gr_bas_character_preset["equipment_items"] == {
        "right_hand": "g_w_blstrpstl01"
    }
    assert body._gr_bas_character_preset["nonvisual_slots"] == {
        "right_weapon": "w_blstrpstl_001"
    }
    status = harness.body_attachment_panel.status.toPlainText()
    assert "Loaded canonical K1 character Blaster Turret" in status
    assert "Non-visual inventory kept as metadata" in status


def test_recipe_persists_cross_game_character_preset_provenance() -> None:
    _configure_native_python_roots()
    from src.systems.bas.model_recipe import build_bas_model_recipe

    body = SimpleNamespace(
        name="p_candbb",
        _gr_source_resref="p_candbb",
        _gr_source_game="K1",
        supermodel="pmbb",
        _gr_bas_character_preset={
            "game": "K2",
            "utc_resref": "p_cand",
            "display_name": "Canderous",
            "body_game": "K1",
            "head_game": "K1",
            "slot_games": {"head": "K1", "right_weapon": "K1"},
            "slot_models": {"right_weapon": "w_hvrptbltr_002"},
            "nonvisual_slots": {},
            "body_texture": "p_candbb01",
            "head_texture": "",
            "equipment_items": {"right_hand": "g_w_hvrptbltr002"},
        },
    )
    head = SimpleNamespace(
        name="p_candh",
        _gr_source_resref="p_candh",
        _gr_source_game="K1",
        supermodel="pmbb",
    )

    recipe = build_bas_model_recipe(
        body_model=body,
        attachment_models={"head": head},
        attachment_resrefs={"head": "p_candh"},
        game="K1",
        mode="headless_body",
    )

    assert recipe["body"]["game"] == "K1"
    assert recipe["character_preset"] == {
        "game": "K2",
        "utc_resref": "p_cand",
        "display_name": "Canderous",
        "body_game": "K1",
        "head_game": "K1",
        "slot_games": {"head": "K1", "right_weapon": "K1"},
        "slot_models": {"right_weapon": "w_hvrptbltr_002"},
        "nonvisual_slots": {},
        "body_texture": "p_candbb01",
        "head_texture": "",
        "equipment_items": {"right_hand": "g_w_hvrptbltr002"},
    }
