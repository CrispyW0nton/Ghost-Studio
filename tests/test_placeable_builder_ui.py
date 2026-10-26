"""Focused UI and main-shell contracts for the Placeable Builder workbench."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from PySide6 import QtCore, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]


def _qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_placeable_builder_is_a_dedicated_service_facing_workbench() -> None:
    _qapp()
    from src.gui.qt_lib.windows.qt_placeable_builder import PLACEABLE_CATEGORIES, QtPlaceableBuilderWindow

    window = QtPlaceableBuilderWindow()
    try:
        assert window.windowTitle() == "GhostStudio — Placeable Builder"
        assert [window.inspector_tabs.tabText(index) for index in range(window.inspector_tabs.count())] == [
            "Identity",
            "Visual",
            "Secondary Motion",
            "Interaction",
            "Scripts / Conversation",
            "Particles",
            "Resources",
            "Readiness",
        ]
        assert tuple(window.category_combo.itemData(index) for index in range(window.category_combo.count())) == PLACEABLE_CATEGORIES
        assert "not one UTP switch" in window.puzzle_note_label.text()
        assert "scripts, tags, keys/items" in window.puzzle_note_label.text()
        assert window.preview_viewport.viewport_role == "main"
        assert window.preview_viewport.map_studio_authoring_chrome_enabled is False
        assert window.preview_viewport.viewport_map_studio_modeling_tabs is None
        assert window.inspector_tabs.tabBar().usesScrollButtons() is True
        assert window.inspector_tabs.tabBar().expanding() is False
        assert window.inspector_tabs.tabBar().elideMode() == QtCore.Qt.ElideNone
        assert window.secondary_motion_editor.runtime_status.text().startswith("Safe authoring mode")
        assert [action.text() for action in window.placeable_toolbar.actions() if not action.isSeparator()] == [
            "New",
            "Clone",
            "Open",
            "Save to Library",
            "Export Game Bundle…",
            "Validate",
            "Open Library Folder",
        ]
        assert all(
            not action.icon().isNull()
            for action in (
                window.new_action,
                window.clone_action,
                window.open_action,
                window.save_library_action,
                window.export_utp_action,
                window.validate_action,
                window.open_library_folder_action,
            )
        )
    finally:
        window.mark_clean()
        window.close()


def test_placeable_particle_tab_is_honest_and_actionable() -> None:
    _qapp()
    from src.gui.qt_lib.windows.qt_placeable_builder import QtPlaceableBuilderWindow

    window = QtPlaceableBuilderWindow()
    try:
        assert window.particle_effects_table.horizontalHeaderItem(1).text() == "X (m)"
        assert window.findChild(QtWidgets.QScrollArea, "placeableParticlesScroll").horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
        assert "bakes these emitters" in window.particle_export_warning.text()
        assert "Map Studio" in window.particle_export_warning.text()
        assert "same resources are added" in window.particle_export_warning.text()
        assert window.remove_particle_effect_button.isEnabled() is False
        assert window.reset_particle_offsets_button.isEnabled() is False

        document = window.current_document()
        document.setdefault("metadata", {})["particle_effects"] = [
            {
                "game": "K2",
                "model": "plc_holo01",
                "node": "emitter01",
                "offset": [1.0, 2.0, 3.0],
            }
        ]
        window.set_document(document)
        assert window.particle_effects_summary.text() == "1 emitter attachment(s)."
        window.particle_effects_table.selectRow(0)
        assert window.remove_particle_effect_button.isEnabled() is True
        assert window.reset_particle_offsets_button.isEnabled() is True
        window._reset_selected_particle_offsets()
        assert window.current_document()["metadata"]["particle_effects"][0]["offset"] == [0.0, 0.0, 0.0]
    finally:
        window.mark_clean()
        window.close()


def test_particle_game_bundle_manifest_keeps_runtime_mapping_evidence() -> None:
    source = (
        ROOT
        / "native/GhostRigger.Core.Tools/Python/src/core/tools/placeable_builder_tool_service.py"
    ).read_text(encoding="utf-8")

    assert '"appearance_id": particle_build.appearance_id' in source
    assert '"model_resref": particle_build.model_resref' in source
    assert '"source_model_resref": particle_build.source_model_resref' in source
    assert '"emitter_count": particle_build.emitter_count' in source


def test_placeable_builder_round_trips_domain_shape_without_dropping_evidence_or_unknown_fields() -> None:
    _qapp()
    from src.gui.qt_lib.windows.qt_placeable_builder import QtPlaceableBuilderWindow

    base_template = {
        "scheme": "game_resource",
        "game": "k2",
        "resref": "plc_box1",
        "restype": "UTP",
        "layer": "models_bif",
        "metadata": {"source": "vanilla"},
    }
    mdl_address = {
        "scheme": "game_resource",
        "game": "k2",
        "resref": "plc_box1",
        "restype": "MDL",
        "layer": "models_bif",
        "metadata": {"source": "vanilla"},
    }
    document = {
        "file_type": "ghostrigger.placeable_asset",
        "schema_version": 1,
        "asset_id": "placeable_0123456789abcdef0123456789abcdef",
        "game": "K2",
        "template_resref": "gr_box01",
        "tag": "gr_box01",
        "display_name": "Reusable Cargo Box",
        "description": "A reusable locked container.",
        "comment": "UI round-trip fixture",
        "category": "container",
        "visual_source": "stock",
        "appearance_id": 12,
        "gameplay": {
            "useable": True,
            "has_inventory": True,
            "inventory_items": ["g_i_progspike01"],
            "lockable": True,
            "locked": True,
            "maximum_hp": 15,
            "current_hp": 15,
            "future_gameplay_field": "preserve-me",
        },
        "scripts": {"on_used": "k_gr_box_use", "on_force_power": "k_gr_force"},
        "resources": {"mdl": mdl_address, "mdx": None, "pwk": None, "textures": []},
        "base_template": base_template,
        "base_evidence": {"sha256": "a" * 64, "field_count": 42},
        "appearance_evidence": {"model_resref": "plc_box1", "verified": True},
        "metadata": {"future_metadata": {"preserve": True}},
        "future_top_level": "preserve-me-too",
    }
    window = QtPlaceableBuilderWindow()
    try:
        window.set_document(document)
        output = window.current_document()
        assert output["asset_id"] == document["asset_id"]
        assert output["category"] == "container"
        assert output["gameplay"]["future_gameplay_field"] == "preserve-me"
        assert output["scripts"]["on_force_power"] == "k_gr_force"
        assert output["base_evidence"] == document["base_evidence"]
        assert output["appearance_evidence"] == document["appearance_evidence"]
        assert output["future_top_level"] == "preserve-me-too"
        assert output["resources"]["mdl"] == mdl_address
        assert output["base_template"] == base_template
        assert output["metadata"]["future_metadata"] == {"preserve": True}
        assert output["secondary_motion"]["schema"] == "ghostrigger.secondary_motion_profile.v1"
        assert output["secondary_motion"]["runtime"]["write_enabled"] is False

        window.display_name_edit.setText("Updated Cargo Box")
        assert window.current_document()["display_name"] == "Updated Cargo Box"
    finally:
        window.mark_clean()
        window.close()


def test_placeable_builder_library_search_readiness_and_completion_signals(tmp_path: Path) -> None:
    _qapp()
    from src.gui.qt_lib.windows.qt_placeable_builder import QtPlaceableBuilderWindow

    window = QtPlaceableBuilderWindow()
    saved: list[dict] = []
    changed: list[str] = []
    try:
        root = tmp_path / "PlaceableLibrary"
        window.set_library_root(root)
        assert window.placeable_library_root == root
        window.set_library_rows(
            [
                {
                    "game": "K2",
                    "resref": "gr_terminal",
                    "label": "Security Terminal",
                    "category": "Placeables",
                    "subcategory": "Terminal",
                    "confidence": "authored_unproven",
                    "metadata": {"document_valid": True},
                },
                {
                    "game": "K1",
                    "resref": "gr_crate",
                    "label": "Supply Crate",
                    "category": "Placeables",
                    "subcategory": "Container",
                },
            ]
        )
        assert window.library_proxy_model.rowCount() == 2
        window.library_search_edit.setText("terminal")
        assert window.library_proxy_model.rowCount() == 1
        assert window.library_proxy_model.index(0, 0).data() == "Security Terminal"
        window.library_game_filter.setCurrentText("K1")
        assert window.library_proxy_model.rowCount() == 0
        window.library_game_filter.setCurrentText("All games")

        issue = SimpleNamespace(severity="warning", code="appearance_unproven", message="Needs 2DA evidence", fix_hint="Attach proof")
        report = SimpleNamespace(
            issues=(issue,),
            document_valid=True,
            utp_export_ready=True,
            structural_evidence_ready=False,
            engine_ready=False,
        )
        window.set_readiness(report)
        assert window.inspector_tabs.currentIndex() == 0
        assert window.readiness_labels["library"].text() == "Ready"
        assert "Blocked" in window.readiness_labels["module"].text()
        assert "Not proven" in window.readiness_labels["game"].text()
        assert window.validation_issues_tree.topLevelItemCount() == 1
        window.set_readiness(report, reveal=True)
        assert window.inspector_tabs.currentIndex() == 7

        window.saveToLibraryRequested.connect(saved.append)
        window.libraryChanged.connect(changed.append)
        window.request_save_to_library()
        assert len(saved) == 1
        window.accept_library_save()
        assert changed == [str(root)]
        assert window.dirty is False
    finally:
        window.mark_clean()
        window.close()


def test_main_shell_has_lazy_placeable_builder_route_and_dedicated_icon() -> None:
    chrome = (
        ROOT
        / "native/GhostRigger.Core.GUI.Display/Python/src/gui/windows/application_core/shared/window_chrome.py"
    ).read_text(encoding="utf-8")
    routing = (
        ROOT
        / "native/GhostRigger.Core.GUI.Display/Python/src/gui/windows/application_core/shared/viewport_tools.py"
    ).read_text(encoding="utf-8")
    facade = (ROOT / "native/GhostRigger.Core.GUI.Display/Python/src/gui/qt_lib.py").read_text(encoding="utf-8")
    assert 'self._icon(MAIN_ACTION_ICON_KEYS["placeable_builder"]), "Open Placeable Builder...", self' in chrome
    assert "self.placeable_builder_action.triggered.connect(self._open_placeable_builder_window)" in chrome
    assert "tools_menu.addAction(self.placeable_builder_action)" in chrome
    assert '"CommandStripPlaceableBuilderButton"' in chrome
    assert "def _open_placeable_builder_window(self):" in routing
    assert "from src.gui.qt_lib.windows.qt_placeable_builder import QtPlaceableBuilderWindow" in routing
    assert '"qt_placeable_builder"' in facade
    source_icon = ROOT / "src/gui/icons/placeable_builder.svg"
    runtime_icon = ROOT / "native/GhostRigger.Native.Core.Host/RuntimePayload/src/gui/icons/placeable_builder.svg"
    assert source_icon.read_bytes() == runtime_icon.read_bytes()


def test_main_shell_lazy_route_opens_and_reuses_placeable_workbench(tmp_path: Path) -> None:
    _qapp()
    from src.gui.windows.application_core.shared.viewport_tools import ViewportToolsMixin

    class Host(ViewportToolsMixin, QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.app_root = tmp_path
            self.settings_data = {"viewport_navigation_profile": "maya"}

    host = Host()
    try:
        host._open_placeable_builder_window()
        first = host.placeable_builder_window
        assert first.isVisible()
        assert first.placeable_library_root == tmp_path / "Saved" / "PlaceableLibrary"
        assert first.preview_viewport.map_studio_authoring_chrome_enabled is False
        host._open_placeable_builder_window()
        assert host.placeable_builder_window is first
    finally:
        window = getattr(host, "placeable_builder_window", None)
        if window is not None:
            window.mark_clean()
            window.close()
        host.close()


def test_placeable_builder_controller_saves_real_library_document_and_utp(tmp_path: Path) -> None:
    _qapp()
    from src.gui.qt_lib.windows.qt_placeable_builder import QtPlaceableBuilderWindow
    from src.gui.qt_lib.windows.qt_placeable_builder_controller import QtPlaceableBuilderController

    window = QtPlaceableBuilderWindow()
    controller = QtPlaceableBuilderController(window, library_root=tmp_path, parent=window)
    changed: list[str] = []
    window.libraryChanged.connect(changed.append)
    try:
        initial_id = window.current_document()["asset_id"]
        assert initial_id.startswith("placeable_")
        window.template_resref_edit.setText("gr_crate")
        window.tag_edit.setText("gr_crate")
        window.display_name_edit.setText("Reusable Supply Crate")
        window.appearance_id_spin.setValue(4)

        result = controller.save_document(window.current_document())

        assert result.ok is True, result.messages
        assert (tmp_path / "gr_crate.ghostplaceable.json").is_file()
        assert (tmp_path / "gr_crate.utp").is_file()
        assert changed == [str(tmp_path)]
        assert any(row["resref"] == "gr_crate" for row in window.library_rows())
        assert window.readiness_labels["game"].text().startswith("Not proven")
        assert window.dirty is False
    finally:
        window.mark_clean()
        window.close()
