"""Content Browser room scope and saved user preference regressions."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6 import QtWidgets

from src.gui.panels.qt_content_browser_panel import QtContentBrowserPanel
from src.gui.windows.application_core.shared import resource_loading
from src.gui.dialogs.qt_settings_dialog import load_settings


class BrowserHost(resource_loading.ResourceLoadingMixin):
    def __init__(self, settings, path=None, objects=()):
        self.settings_data = settings
        self.settings_path = path
        self.scene_manager = SimpleNamespace(get_scene_objects=lambda: objects)
        self.loads = []
        self._model_worker_is_running = lambda: False
        self._current_game = "K2"
        self._resource_manager = object()

    def _start_resource_load(self, resref, game, import_action=""):
        self.loads.append((resref, game, import_action))

    def _log(self, *args):
        pass


@pytest.mark.parametrize("preference", [None, False, "false"])
@pytest.mark.parametrize("add,existing", [(False, False), (False, True), (True, True)])
def test_room_default_never_expands_map(monkeypatch, preference, add, existing):
    host = BrowserHost({"content_browser_load_connected_map": preference}, objects=[object()] if existing else [])
    def unexpected_expansion(**kwargs):
        pytest.fail("Selected-room mode must not resolve the full map")
    monkeypatch.setattr(resource_loading, "resolve_module_room_placements", unexpected_expansion)
    row = {"game": "K2", "resref": "001ebo1", "category": "Modules"}
    action = host._add_content_browser_model_to_current_scene if add else host._load_content_browser_primary_scene_model
    action(row)
    assert host.loads == [("001ebo1", "K2", "add" if add or existing else "clear")]


@pytest.mark.parametrize("add", [False, True])
def test_connected_map_keeps_all_layout_rooms(monkeypatch, add):
    host = BrowserHost({"content_browser_load_connected_map": True})
    host._model_worker_is_running = lambda: False
    host._current_game = "K2"
    host._resource_manager = object()
    host._show_progress_toast = lambda *args: None
    rooms = [SimpleNamespace(module_root="001ebo", area_label="Ebon Hawk", resref=name) for name in ("001ebo1", "001ebo2")]
    monkeypatch.setattr(resource_loading, "resolve_module_room_placements", lambda **kwargs: rooms)
    monkeypatch.setattr(resource_loading.QtCore.QTimer, "singleShot", lambda delay, callback: callback())
    loaded = []
    host._load_module_rooms_on_ui_thread = lambda placements, action: loaded.append((placements, action))
    action = host._add_content_browser_model_to_current_scene if add else host._load_content_browser_primary_scene_model
    action({"game": "K2", "resref": "001ebo1", "category": "Modules"})
    assert loaded == [(tuple(rooms), "add" if add else "clear")]
    assert host.loads == []


def test_scope_control_saves_restores_and_switches_back(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = QtContentBrowserPanel()
    path = tmp_path / "settings.json"
    host = BrowserHost({"unrelated_setting": "preserved"}, path)
    host._settings_dialog = SimpleNamespace(settings=dict(host.settings_data))
    panel.loadConnectedMapChanged.connect(host._set_content_browser_load_connected_map)
    assert panel.module_load_scope.currentText() == "Selected room only"
    panel.module_load_scope.setCurrentIndex(1)
    saved = load_settings(path)
    assert saved == {"unrelated_setting": "preserved", "content_browser_load_connected_map": True}
    assert host._settings_dialog.settings == saved
    reopened = QtContentBrowserPanel()
    reopened.set_load_connected_map(saved["content_browser_load_connected_map"])
    assert reopened.module_load_scope.currentText() == "Entire connected map"
    panel.module_load_scope.setCurrentIndex(0)
    assert load_settings(path)["content_browser_load_connected_map"] is False
    panel.deleteLater()
    reopened.deleteLater()
    app.processEvents()


def test_double_click_module_uses_scope_control(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = QtContentBrowserPanel()
    host = BrowserHost({}, tmp_path / "settings.json")
    panel.addToCurrentSceneRequested.connect(host._add_content_browser_model_to_current_scene)
    panel.set_rows([{"game": "K2", "resref": "001ebo1", "category": "Modules"}])
    panel.select_asset_type("Module")
    panel.asset_view.setCurrentItem(panel.asset_view.topLevelItem(0))
    panel.asset_view.itemDoubleClicked.emit(panel.asset_view.currentItem(), 0)
    assert host.loads == [("001ebo1", "K2", "add")]
    panel.deleteLater()
    app.processEvents()


def test_preference_save_failure_keeps_session_choice(tmp_path):
    host = BrowserHost({}, tmp_path / "missing" / "settings.json")
    messages = []
    host._log = lambda *args: messages.append(args)
    host._set_content_browser_load_connected_map(True)
    assert host.settings_data["content_browser_load_connected_map"] is True
    assert messages and messages[0][1] == "error"
