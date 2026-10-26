"""Dedicated Qt workbench for authoring reusable KOTOR placeable assets.

The window deliberately owns presentation and interaction only.  Persistence,
UTP serialization, vanilla-structure evidence, dependency resolution, and game
proof remain services exposed through the public request signals below.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from src.core.project.secondary_motion_profile import default_secondary_motion_profile
from src.gui.panels.qt_secondary_motion_editor import QtSecondaryMotionEditor

from src.core.rendering.renderer_settings import RendererSettings
from src.core.rendering.viewport_navigation import DEFAULT_VIEWPORT_NAVIGATION_PROFILE
from src.gui.qt_lib.viewports.qt_viewport import QtMainViewportWidget


PLACEABLE_CATEGORIES = ("container", "terminal", "puzzle", "interactive", "decor")

SCRIPT_HOOKS = (
    ("On Used", "on_used"),
    ("On Open", "on_open"),
    ("On Closed", "on_closed"),
    ("On Open Failed", "on_open_failed"),
    ("On Damaged", "on_damaged"),
    ("On Death", "on_death"),
    ("On Heartbeat", "on_heartbeat"),
    ("On Inventory Changed", "on_inventory"),
    ("On Lock", "on_lock"),
    ("On Unlock", "on_unlock"),
    ("On User Defined", "on_user_defined"),
    ("On Disarm", "on_disarm"),
    ("On Trap Triggered", "on_trap_triggered"),
)

LIBRARY_ROW_ROLE = int(QtCore.Qt.UserRole) + 1
LIBRARY_GAME_ROLE = LIBRARY_ROW_ROLE + 1
LIBRARY_CATEGORY_ROLE = LIBRARY_ROW_ROLE + 2
LIBRARY_SEARCH_ROLE = LIBRARY_ROW_ROLE + 3


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _resource_resref(value: object) -> str:
    row = _mapping(value)
    if row:
        return str(row.get("resref") or row.get("path") or "")
    return str(value or "")


def _resource_address(game: str, resref: str, restype: str) -> dict[str, Any] | None:
    text = str(resref or "").strip()
    if not text:
        return None
    return {
        "scheme": "project_resource",
        "game": str(game or "").lower() or None,
        "module_id": None,
        "resref": text.lower(),
        "restype": str(restype or "").upper() or None,
        "layer": "placeable_library",
        "path": None,
        "object_id": None,
        "fragment": None,
        "metadata": {},
    }


class _PlaceableLibraryFilterProxyModel(QtCore.QSortFilterProxyModel):
    """Fast multi-field filter for stock and authored placeable rows."""

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._query = ""
        self._game = ""
        self._category = ""
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(QtCore.Qt.CaseInsensitive)

    def set_placeable_filters(self, *, query: str, game: str, category: str) -> None:
        values = (str(query or "").strip().lower(), str(game or "").upper(), str(category or "").lower())
        if values == (self._query, self._game, self._category):
            return
        begin_change = getattr(self, "beginFilterChange", None)
        if callable(begin_change):
            begin_change()
        self._query, self._game, self._category = values
        end_change = getattr(self, "endFilterChange", None)
        if callable(end_change):
            end_change(QtCore.QSortFilterProxyModel.Direction.Rows)
        else:  # pragma: no cover - compatibility with older PySide6 builds
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        if self._query and self._query not in str(index.data(LIBRARY_SEARCH_ROLE) or ""):
            return False
        if self._game in {"K1", "K2"} and str(index.data(LIBRARY_GAME_ROLE) or "").upper() != self._game:
            return False
        if self._category and str(index.data(LIBRARY_CATEGORY_ROLE) or "").lower() != self._category:
            return False
        return True


_PARTICLE_LIBRARY_CACHE: dict[tuple[str, str], list] = {}


def _load_particle_templates(app_root: Path, game: str) -> list:
    """Load (and memoize) the scanned emitter template library for one game."""
    from src.core.particles.emitter_library import (
        library_cache_path,
        load_library,
        resolve_library_root,
    )

    root = resolve_library_root(Path(app_root))
    key = (str(root), str(game).upper())
    cached = _PARTICLE_LIBRARY_CACHE.get(key)
    if cached is not None:
        return cached
    templates = load_library(library_cache_path(root, game))
    _PARTICLE_LIBRARY_CACHE[key] = templates
    return templates


class _ParticleAssetPickerDialog(QtWidgets.QDialog):
    """Modal picker over the scanned K1/K2 emitter template library."""

    def __init__(self, parent: QtWidgets.QWidget, game: str, templates: list):
        super().__init__(parent)
        self.setObjectName("placeableParticleAssetPicker")
        self.setWindowTitle(f"Add Particle Effect — {game} Library")
        self.resize(560, 620)
        self._game = str(game).upper()
        self._templates = templates
        self.selected_model: str = ""
        self.selected_nodes: list[str] = []
        self.whole_effect: bool = False

        layout = QtWidgets.QVBoxLayout(self)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setObjectName("placeableParticlePickerSearch")
        self.search_edit.setPlaceholderText(
            "Search model, emitter, texture, update, or blend…"
        )
        self.search_edit.textChanged.connect(self._rebuild_tree)
        layout.addWidget(self.search_edit)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setObjectName("placeableParticlePickerTree")
        self.tree.setHeaderLabels(["Source", "Info"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.itemExpanded.connect(self._populate_expanded_model)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        layout.addWidget(self.tree, 1)

        self.selection_label = QtWidgets.QLabel("Choose one source model or one or more of its emitters.")
        self.selection_label.setObjectName("placeableParticlePickerSelectionSummary")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        hint = QtWidgets.QLabel(
            "Select emitters, or select a model row and use Add Entire Effect to attach the complete "
            "authored effect (e.g. plc_holoXXX planet holograms keep their full ring lattice)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QtWidgets.QHBoxLayout()
        self.add_selected_button = QtWidgets.QPushButton("Add Selected Emitters")
        self.add_selected_button.setObjectName("placeableParticlePickerAddSelected")
        self.add_selected_button.clicked.connect(lambda: self._accept(whole=False))
        buttons.addWidget(self.add_selected_button)
        self.add_all_button = QtWidgets.QPushButton("Add Entire Effect")
        self.add_all_button.setObjectName("placeableParticlePickerAddAll")
        self.add_all_button.clicked.connect(lambda: self._accept(whole=True))
        buttons.addWidget(self.add_all_button)
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        self._rebuild_tree()

    @staticmethod
    def _template_haystack(template) -> str:
        defn = template.definition
        return " ".join(
            (
                template.model,
                template.node,
                str(defn.get("texture", "")),
                str(defn.get("update", "")),
                str(defn.get("blend", "")),
            )
        ).lower()

    def _add_emitter_children(self, model_item: QtWidgets.QTreeWidgetItem, rows: list) -> None:
        model_item.takeChildren()
        for template in rows:
            defn = template.definition
            info = f"{defn.get('update', '')}/{defn.get('blend', '')} {defn.get('texture', '')}"
            child = QtWidgets.QTreeWidgetItem([template.node, info])
            child.setData(0, QtCore.Qt.UserRole, ("emitter", (template.model, template.node)))
            model_item.addChild(child)

    def _rebuild_tree(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            by_model: dict[str, list] = {}
            for template in self._templates:
                if query and query not in self._template_haystack(template):
                    continue
                by_model.setdefault(template.model, []).append(template)
            for model_name in sorted(by_model):
                rows = by_model[model_name]
                model_item = QtWidgets.QTreeWidgetItem([model_name, f"{len(rows)} emitters"])
                model_item.setData(0, QtCore.Qt.UserRole, ("model", model_name))
                if query:
                    self._add_emitter_children(model_item, rows)
                else:
                    model_item.addChild(QtWidgets.QTreeWidgetItem(["Expand to view emitters", ""]))
                self.tree.addTopLevelItem(model_item)
            if query:
                self.tree.expandToDepth(0)
        finally:
            self.tree.setUpdatesEnabled(True)
        self._selection_changed()

    def _populate_expanded_model(self, item: QtWidgets.QTreeWidgetItem) -> None:
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload or payload[0] != "model":
            return
        first = item.child(0)
        if first is not None and first.data(0, QtCore.Qt.UserRole) is None:
            model_name = str(payload[1])
            self._add_emitter_children(
                item, [row for row in self._templates if row.model == model_name]
            )

    @staticmethod
    def _item_model(item: QtWidgets.QTreeWidgetItem) -> str:
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload:
            return ""
        return str(payload[1] if payload[0] == "model" else payload[1][0])

    def _selection_changed(self) -> None:
        selected = self.tree.selectedItems()
        current = self.tree.currentItem()
        if current is None and selected:
            current = selected[0]
        active_model = self._item_model(current) if current is not None else ""
        if active_model:
            blocker = QtCore.QSignalBlocker(self.tree)
            for item in selected:
                if self._item_model(item) != active_model:
                    item.setSelected(False)
            del blocker
            selected = self.tree.selectedItems()
        emitter_count = sum(
            1 for item in selected
            if (item.data(0, QtCore.Qt.UserRole) or (None,))[0] == "emitter"
        )
        self.add_selected_button.setEnabled(bool(active_model and emitter_count))
        self.add_all_button.setEnabled(bool(active_model))
        if active_model:
            detail = f"{emitter_count} emitter(s) selected" if emitter_count else "entire effect available"
            self.selection_label.setText(f"{self._game}:{active_model} — {detail}.")
        else:
            self.selection_label.setText("Choose one source model or one or more of its emitters.")

    def _double_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        payload = item.data(0, QtCore.Qt.UserRole)
        if not payload:
            return
        self._accept(whole=payload[0] == "model")

    def _accept(self, *, whole: bool) -> None:
        model_name = ""
        nodes: list[str] = []
        for item in self.tree.selectedItems():
            payload = item.data(0, QtCore.Qt.UserRole)
            if not payload:
                continue
            kind, ref = payload
            if kind == "model":
                model_name = model_name or str(ref)
            else:
                item_model, item_node = ref
                model_name = model_name or str(item_model)
                nodes.append(str(item_node))
        if not model_name or (not whole and not nodes):
            self.selection_label.setText("Select at least one emitter, or choose Add Entire Effect.")
            return
        self.selected_model = model_name
        self.selected_nodes = [] if whole else nodes
        self.whole_effect = whole or not nodes
        self.accept()


def _pick_particle_effects(parent: QtWidgets.QWidget, game: str, resource_manager) -> list[dict[str, Any]]:
    """Run the picker and return portable effect records with baked transforms."""
    from src.core.particles.emitter_library import build_effect_records

    app_root = Path(getattr(parent.parent(), "app_root", Path.cwd()) or Path.cwd())
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        templates = _load_particle_templates(app_root, game)
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    if not templates:
        raise RuntimeError(
            f"No scanned {game} emitter library found. Run Scan Game Libraries in the Particle Editor first."
        )
    dialog = _ParticleAssetPickerDialog(parent, game, templates)
    if dialog.exec() != QtWidgets.QDialog.Accepted or not dialog.selected_model:
        return []

    if resource_manager is not None:
        source_model = None
        try:
            source_model = resource_manager.load_model(dialog.selected_model, game)
        except Exception:
            source_model = None
        if source_model is not None:
            return build_effect_records(
                source_model,
                game,
                dialog.selected_model,
                node_names=None if dialog.whole_effect else dialog.selected_nodes,
            )

    # No resource manager: fall back to definition-only records at the origin.
    wanted = None if dialog.whole_effect else {name.lower() for name in dialog.selected_nodes}
    records: list[dict[str, Any]] = []
    for template in templates:
        if template.model != dialog.selected_model:
            continue
        if wanted is not None and template.node.lower() not in wanted:
            continue
        records.append({
            "game": str(game).upper(),
            "model": template.model,
            "node": template.node,
            "definition": deepcopy(template.definition),
            "base_position": [0.0, 0.0, 1.0],
            "base_rotation": [0.0, 0.0, 0.0, 1.0],
            "offset": [0.0, 0.0, 0.0],
        })
    return records


class QtPlaceableBuilderWindow(QtWidgets.QMainWindow):
    """Standalone workbench for a project Placeable Library asset."""

    newRequested = QtCore.Signal()
    cloneRequested = QtCore.Signal(object)
    openRequested = QtCore.Signal()
    saveToLibraryRequested = QtCore.Signal(object)
    exportUtpRequested = QtCore.Signal(object)
    validateRequested = QtCore.Signal(object)
    openLibraryFolderRequested = QtCore.Signal(str)
    refreshLibraryRequested = QtCore.Signal()
    libraryAssetActivated = QtCore.Signal(object)
    libraryChanged = QtCore.Signal(str)
    documentChanged = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("GhostStudio — Placeable Builder")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setMinimumSize(1040, 680)
        self.setObjectName("placeableBuilderWindow")
        self.setProperty("ghostLayoutId", "placeableBuilder")

        self._library_root = Path()
        self._library_rows: list[dict[str, Any]] = []
        self._document_passthrough: dict[str, Any] = {}
        self._base_template_passthrough: dict[str, Any] | None = None
        self._resource_passthrough: dict[str, Any] = {}
        self._scripts_passthrough: dict[str, Any] = {}
        self._readiness: dict[str, Any] = {}
        self._resource_rows: list[dict[str, Any]] = []
        self._particle_effects: list[dict[str, Any]] = []
        self._resource_manager: Any = None
        self._updating_document = False
        self._dirty = False
        self._renderer_settings = RendererSettings.from_settings(getattr(parent, "settings_data", {}) or {})
        self._navigation_profile = DEFAULT_VIEWPORT_NAVIGATION_PROFILE

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._connect_document_controls()
        self._reset_document(emit_request=False)

        theme_manager = getattr(parent, "theme_manager", None)
        layout_manager = getattr(parent, "layout_manager", None)
        if theme_manager is not None:
            theme_manager.register_theme_aware_widget(self)
            self.apply_ghost_theme(theme_manager.current_theme or theme_manager.get_theme())
        if layout_manager is not None:
            layout_manager.layoutChanged.connect(self.apply_ghost_layout)
            self.apply_ghost_layout(layout_manager.current_layout or layout_manager.get_layout())
        else:
            self.resize(1360, 820)

    # ------------------------------------------------------------------
    # Window construction

    def _workspace_icon(self, name: str, fallback: QtWidgets.QStyle.StandardPixmap) -> QtGui.QIcon:
        provider = getattr(self.parent(), "_icon", None)
        if callable(provider):
            icon = provider(name, 18)
            if icon is not None and not icon.isNull():
                return icon
        return QtWidgets.QApplication.style().standardIcon(fallback)

    def _build_actions(self) -> None:
        self.new_action = QtGui.QAction(
            self._workspace_icon("new_scene", QtWidgets.QStyle.SP_FileIcon), "New", self
        )
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setStatusTip("Start a new reusable placeable asset")
        self.new_action.triggered.connect(self.new_placeable)

        self.clone_action = QtGui.QAction(
            self._workspace_icon("duplicate", QtWidgets.QStyle.SP_FileDialogNewFolder), "Clone", self
        )
        self.clone_action.setStatusTip("Clone this placeable through the library service")
        self.clone_action.triggered.connect(self.clone_placeable)

        self.open_action = QtGui.QAction(
            self._workspace_icon("open", QtWidgets.QStyle.SP_DialogOpenButton), "Open", self
        )
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setStatusTip("Open a stock UTP or authored Placeable Library asset")
        self.open_action.triggered.connect(self._request_open)

        self.save_library_action = QtGui.QAction(
            self._workspace_icon("save", QtWidgets.QStyle.SP_DialogSaveButton), "Save to Library", self
        )
        self.save_library_action.setShortcut("Ctrl+S")
        self.save_library_action.setStatusTip("Save the human-readable asset into the Placeable Library")
        self.save_library_action.triggered.connect(self.request_save_to_library)

        self.export_utp_action = QtGui.QAction(
            self._workspace_icon("export", QtWidgets.QStyle.SP_DialogSaveButton), "Export Game Bundle…", self
        )
        self.export_utp_action.setStatusTip(
            "Build a verified UTP + MDL/MDX + particle texture + placeables.2da Override bundle"
        )
        self.export_utp_action.triggered.connect(self.request_export_utp)

        self.validate_action = QtGui.QAction(
            self._workspace_icon("diag", QtWidgets.QStyle.SP_MessageBoxInformation), "Validate", self
        )
        self.validate_action.setStatusTip("Check document, UTP, dependency, and structural-evidence readiness")
        self.validate_action.triggered.connect(self.request_validate)

        self.open_library_folder_action = QtGui.QAction(
            self._workspace_icon("library", QtWidgets.QStyle.SP_DirOpenIcon), "Open Library Folder", self
        )
        self.open_library_folder_action.triggered.connect(self._request_open_library_folder)

        self.refresh_library_action = QtGui.QAction(
            self._workspace_icon("refresh", QtWidgets.QStyle.SP_BrowserReload), "Refresh Library", self
        )
        self.refresh_library_action.triggered.connect(lambda _checked=False: self.refreshLibraryRequested.emit())

        self.frame_preview_action = QtGui.QAction(
            self._workspace_icon("viewport_frame", QtWidgets.QStyle.SP_DesktopIcon), "Frame Preview", self
        )
        self.frame_preview_action.triggered.connect(lambda: self.preview_viewport.frame_all())

        self.close_action = QtGui.QAction("Close", self)
        self.close_action.setShortcut("Ctrl+W")
        self.close_action.triggered.connect(self.close)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        for action in (
            self.new_action,
            self.clone_action,
            self.open_action,
            None,
            self.save_library_action,
            self.export_utp_action,
            None,
            self.open_library_folder_action,
            None,
            self.close_action,
        ):
            file_menu.addSeparator() if action is None else file_menu.addAction(action)
        build_menu = self.menuBar().addMenu("Build")
        build_menu.addAction(self.validate_action)
        build_menu.addAction(self.export_utp_action)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.refresh_library_action)
        view_menu.addAction(self.frame_preview_action)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Placeable Builder", self)
        toolbar.setObjectName("placeableBuilderToolbar")
        toolbar.setMovable(False)
        for action in (
            self.new_action,
            self.clone_action,
            self.open_action,
            None,
            self.save_library_action,
            self.export_utp_action,
            self.validate_action,
            None,
            self.open_library_folder_action,
        ):
            toolbar.addSeparator() if action is None else toolbar.addAction(action)
        self.placeable_toolbar = toolbar
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

    def _build_central(self) -> None:
        self.workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.workspace_splitter.setObjectName("placeableBuilderWorkspaceSplitter")
        self.workspace_splitter.setChildrenCollapsible(False)
        self.library_panel = self._build_library_panel()
        self.preview_panel = self._build_preview_panel()
        self.inspector_tabs = self._build_inspector_tabs()
        self.workspace_splitter.addWidget(self.library_panel)
        self.workspace_splitter.addWidget(self.preview_panel)
        self.workspace_splitter.addWidget(self.inspector_tabs)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setStretchFactor(2, 0)
        self.setCentralWidget(self.workspace_splitter)

    def _build_library_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("placeableLibraryPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Placeable Library")
        title.setObjectName("placeableLibraryHeading")
        header.addWidget(title)
        header.addStretch(1)
        refresh = QtWidgets.QToolButton()
        refresh.setDefaultAction(self.refresh_library_action)
        refresh.setObjectName("placeableLibraryRefreshButton")
        header.addWidget(refresh)
        layout.addLayout(header)

        self.library_root_label = QtWidgets.QLabel("Library folder not connected")
        self.library_root_label.setObjectName("placeableLibraryRootLabel")
        self.library_root_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.library_root_label.setWordWrap(True)
        layout.addWidget(self.library_root_label)

        self.library_search_edit = QtWidgets.QLineEdit()
        self.library_search_edit.setObjectName("placeableLibrarySearchEdit")
        self.library_search_edit.setPlaceholderText("Search name, resref, tag, or category…")
        self.library_search_edit.setClearButtonEnabled(True)
        self.library_search_edit.textChanged.connect(self._rebuild_library_view)
        layout.addWidget(self.library_search_edit)

        filters = QtWidgets.QHBoxLayout()
        self.library_game_filter = QtWidgets.QComboBox()
        self.library_game_filter.setObjectName("placeableLibraryGameFilter")
        self.library_game_filter.addItems(["All games", "K1", "K2"])
        self.library_game_filter.currentIndexChanged.connect(self._rebuild_library_view)
        filters.addWidget(self.library_game_filter)
        self.library_category_filter = QtWidgets.QComboBox()
        self.library_category_filter.setObjectName("placeableLibraryCategoryFilter")
        self.library_category_filter.addItem("All categories", "")
        for category in PLACEABLE_CATEGORIES:
            self.library_category_filter.addItem(category.title(), category)
        self.library_category_filter.currentIndexChanged.connect(self._rebuild_library_view)
        filters.addWidget(self.library_category_filter)
        layout.addLayout(filters)

        self.library_tree = QtWidgets.QTreeView()
        self.library_tree.setObjectName("placeableLibraryTree")
        self.library_tree.setRootIsDecorated(False)
        self.library_tree.setAlternatingRowColors(True)
        self.library_tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.library_tree.setSortingEnabled(True)
        self.library_model = QtGui.QStandardItemModel(self)
        self.library_model.setHorizontalHeaderLabels(["Placeable", "Game", "Type", "Readiness"])
        self.library_proxy_model = _PlaceableLibraryFilterProxyModel(self)
        self.library_proxy_model.setSourceModel(self.library_model)
        self.library_tree.setModel(self.library_proxy_model)
        self.library_tree.doubleClicked.connect(lambda _index: self.activate_selected_library_asset())
        self.library_tree.activated.connect(lambda _index: self.activate_selected_library_asset())
        self.library_tree.selectionModel().selectionChanged.connect(lambda _selected, _deselected: self._library_selection_changed())
        layout.addWidget(self.library_tree, 1)
        return panel

    def _build_preview_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("placeablePreviewPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("In-Engine Asset Preview")
        label.setObjectName("placeablePreviewHeading")
        header.addWidget(label)
        header.addStretch(1)
        frame_button = QtWidgets.QToolButton()
        frame_button.setDefaultAction(self.frame_preview_action)
        frame_button.setObjectName("placeablePreviewFrameButton")
        header.addWidget(frame_button)
        layout.addLayout(header)

        self.preview_viewport = QtMainViewportWidget(self, map_studio_authoring_chrome=False)
        self.preview_viewport.setObjectName("placeableBuilderPreviewViewport")
        self.preview_viewport.set_renderer_settings(self._renderer_settings)
        self.preview_viewport.set_navigation_profile(self._navigation_profile)
        self.preview_viewport.statusMessage.connect(lambda message: self.statusBar().showMessage(message, 4000))
        layout.addWidget(self.preview_viewport, 1)

        self.preview_status_label = QtWidgets.QLabel(
            "Choose a stock UTP or custom model resource to preview. Preview is not engine proof."
        )
        self.preview_status_label.setObjectName("placeablePreviewStatusLabel")
        self.preview_status_label.setWordWrap(True)
        layout.addWidget(self.preview_status_label)
        return panel

    def _build_inspector_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget(self)
        tabs.setObjectName("placeableBuilderInspectorTabs")
        pages = (
            (self._build_identity_tab(), "Identity"),
            (self._build_visual_tab(), "Visual"),
            (self._build_secondary_motion_tab(), "Secondary Motion"),
            (self._build_interaction_tab(), "Interaction"),
            (self._build_scripts_tab(), "Scripts / Conversation"),
            (self._build_particles_tab(), "Particles"),
            (self._build_resources_tab(), "Resources"),
            (self._build_readiness_tab(), "Readiness"),
        )
        for page, label in pages:
            index = tabs.addTab(page, label)
            tabs.setTabToolTip(index, label)
        # The inspector is intentionally narrow beside the viewport.  Preserve
        # readable tab names and scroll the strip instead of reducing every tab
        # to an ambiguous one- or two-letter ellipsis.
        tabs.tabBar().setUsesScrollButtons(True)
        tabs.tabBar().setExpanding(False)
        tabs.tabBar().setElideMode(QtCore.Qt.ElideNone)
        return tabs

    def _build_secondary_motion_tab(self) -> QtWidgets.QWidget:
        self.secondary_motion_editor = QtSecondaryMotionEditor(
            self,
            target_kind="placeable",
            target_game="K2",
        )
        self.secondary_motion_editor.profileChanged.connect(self._document_edited)
        return self.secondary_motion_editor

    @staticmethod
    def _scroll_tab(content: QtWidgets.QWidget, name: str) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName(name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _form(parent: QtWidgets.QWidget) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout(parent)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return form

    def _build_identity_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        form = self._form(content)
        self.display_name_edit = QtWidgets.QLineEdit()
        self.display_name_edit.setObjectName("placeableDisplayNameEdit")
        self.template_resref_edit = QtWidgets.QLineEdit()
        self.template_resref_edit.setObjectName("placeableTemplateResrefEdit")
        self.template_resref_edit.setMaxLength(16)
        self.tag_edit = QtWidgets.QLineEdit()
        self.tag_edit.setObjectName("placeableTagEdit")
        self.game_combo = QtWidgets.QComboBox()
        self.game_combo.setObjectName("placeableGameCombo")
        self.game_combo.addItems(["K1", "K2"])
        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.setObjectName("placeableCategoryCombo")
        for category in PLACEABLE_CATEGORIES:
            self.category_combo.addItem(category.title(), category)
        self.description_edit = QtWidgets.QPlainTextEdit()
        self.description_edit.setObjectName("placeableDescriptionEdit")
        self.description_edit.setPlaceholderText("What this reusable asset is for in a level")
        self.description_edit.setMaximumBlockCount(200)
        self.comment_edit = QtWidgets.QLineEdit()
        self.comment_edit.setObjectName("placeableCommentEdit")
        form.addRow("Display name", self.display_name_edit)
        form.addRow("Template resref", self.template_resref_edit)
        form.addRow("Tag", self.tag_edit)
        form.addRow("Target game", self.game_combo)
        form.addRow("Category", self.category_combo)
        form.addRow("Description", self.description_edit)
        form.addRow("Author comment", self.comment_edit)
        self.puzzle_note_label = QtWidgets.QLabel(
            "Puzzle is an authoring category, not one UTP switch. Build puzzle behavior by composing "
            "scripts, tags, keys/items, inventory state, and conversation resources."
        )
        self.puzzle_note_label.setObjectName("placeablePuzzleCompositionNote")
        self.puzzle_note_label.setWordWrap(True)
        form.addRow("Puzzle behavior", self.puzzle_note_label)
        return self._scroll_tab(content, "placeableIdentityScroll")

    def _build_visual_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        form = self._form(content)
        self.visual_source_combo = QtWidgets.QComboBox()
        self.visual_source_combo.setObjectName("placeableVisualSourceCombo")
        self.visual_source_combo.addItem("Stock appearance (placeables.2da)", "stock")
        self.visual_source_combo.addItem("Custom MDL / MDX / PWK resources", "custom")
        self.base_template_edit = QtWidgets.QLineEdit()
        self.base_template_edit.setObjectName("placeableBaseTemplateEdit")
        self.base_template_edit.setPlaceholderText("Known-loadable base UTP resref")
        self.base_template_edit.setMaxLength(16)
        self.appearance_id_spin = QtWidgets.QSpinBox()
        self.appearance_id_spin.setObjectName("placeableAppearanceIdSpin")
        self.appearance_id_spin.setRange(-1, 65535)
        self.appearance_id_spin.setSpecialValueText("Not set")
        self.appearance_model_hint_edit = QtWidgets.QLineEdit()
        self.appearance_model_hint_edit.setObjectName("placeableAppearanceModelHintEdit")
        self.appearance_model_hint_edit.setPlaceholderText("Resolved model resref (preview hint only)")
        self.visual_contract_label = QtWidgets.QLabel()
        self.visual_contract_label.setObjectName("placeableVisualContractLabel")
        self.visual_contract_label.setWordWrap(True)
        form.addRow("Visual source", self.visual_source_combo)
        form.addRow("Base UTP", self.base_template_edit)
        form.addRow("Appearance row", self.appearance_id_spin)
        form.addRow("Resolved model", self.appearance_model_hint_edit)
        form.addRow("Proof contract", self.visual_contract_label)
        self.visual_source_combo.currentIndexChanged.connect(self._update_visual_contract_text)
        self._update_visual_contract_text()
        return self._scroll_tab(content, "placeableVisualScroll")

    def _build_interaction_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)

        behavior = QtWidgets.QGroupBox("Behavior")
        behavior.setObjectName("placeableBehaviorGroup")
        behavior_layout = QtWidgets.QGridLayout(behavior)
        self.static_check = QtWidgets.QCheckBox("Static")
        self.useable_check = QtWidgets.QCheckBox("Useable")
        self.has_inventory_check = QtWidgets.QCheckBox("Has inventory")
        self.party_interact_check = QtWidgets.QCheckBox("Party interact")
        self.plot_check = QtWidgets.QCheckBox("Plot")
        self.min1_hp_check = QtWidgets.QCheckBox("Cannot fall below 1 HP")
        self.not_blastable_check = QtWidgets.QCheckBox("Not blastable")
        for index, check in enumerate(
            (
                self.static_check,
                self.useable_check,
                self.has_inventory_check,
                self.party_interact_check,
                self.plot_check,
                self.min1_hp_check,
                self.not_blastable_check,
            )
        ):
            behavior_layout.addWidget(check, index // 2, index % 2)
        root.addWidget(behavior)

        durability = QtWidgets.QGroupBox("Durability")
        durability.setObjectName("placeableDurabilityGroup")
        durability_form = self._form(durability)
        self.maximum_hp_spin = QtWidgets.QSpinBox()
        self.maximum_hp_spin.setRange(-32768, 32767)
        self.current_hp_spin = QtWidgets.QSpinBox()
        self.current_hp_spin.setRange(-32768, 32767)
        self.hardness_spin = QtWidgets.QSpinBox()
        self.hardness_spin.setRange(0, 255)
        durability_form.addRow("Maximum HP", self.maximum_hp_spin)
        durability_form.addRow("Current HP", self.current_hp_spin)
        durability_form.addRow("Hardness", self.hardness_spin)
        root.addWidget(durability)

        lock = QtWidgets.QGroupBox("Lock & Key")
        lock.setObjectName("placeableLockGroup")
        lock_form = self._form(lock)
        self.lockable_check = QtWidgets.QCheckBox("Lockable")
        self.locked_check = QtWidgets.QCheckBox("Starts locked")
        lock_flags = QtWidgets.QWidget()
        lock_flags_layout = QtWidgets.QHBoxLayout(lock_flags)
        lock_flags_layout.setContentsMargins(0, 0, 0, 0)
        lock_flags_layout.addWidget(self.lockable_check)
        lock_flags_layout.addWidget(self.locked_check)
        self.key_required_check = QtWidgets.QCheckBox("Key required")
        self.auto_remove_key_check = QtWidgets.QCheckBox("Consume key")
        key_flags = QtWidgets.QWidget()
        key_flags_layout = QtWidgets.QHBoxLayout(key_flags)
        key_flags_layout.setContentsMargins(0, 0, 0, 0)
        key_flags_layout.addWidget(self.key_required_check)
        key_flags_layout.addWidget(self.auto_remove_key_check)
        self.key_name_edit = QtWidgets.QLineEdit()
        self.key_name_edit.setMaxLength(16)
        self.lock_dc_spin = QtWidgets.QSpinBox()
        self.lock_dc_spin.setRange(0, 255)
        self.unlock_dc_spin = QtWidgets.QSpinBox()
        self.unlock_dc_spin.setRange(0, 255)
        lock_form.addRow("State", lock_flags)
        lock_form.addRow("Key policy", key_flags)
        lock_form.addRow("Key resref", self.key_name_edit)
        lock_form.addRow("Lock DC", self.lock_dc_spin)
        lock_form.addRow("Open Lock DC", self.unlock_dc_spin)
        root.addWidget(lock)

        inventory = QtWidgets.QGroupBox("Inventory")
        inventory.setObjectName("placeableInventoryGroup")
        inventory_layout = QtWidgets.QVBoxLayout(inventory)
        self.inventory_items_edit = QtWidgets.QPlainTextEdit()
        self.inventory_items_edit.setObjectName("placeableInventoryItemsEdit")
        self.inventory_items_edit.setPlaceholderText("One UTI resref per line")
        inventory_layout.addWidget(self.inventory_items_edit)
        root.addWidget(inventory)

        trap = QtWidgets.QGroupBox("Trap")
        trap.setObjectName("placeableTrapGroup")
        trap_form = self._form(trap)
        self.trap_detectable_check = QtWidgets.QCheckBox("Detectable")
        self.trap_disarmable_check = QtWidgets.QCheckBox("Disarmable")
        self.trap_one_shot_check = QtWidgets.QCheckBox("One shot")
        trap_flags = QtWidgets.QWidget()
        trap_flags_layout = QtWidgets.QHBoxLayout(trap_flags)
        trap_flags_layout.setContentsMargins(0, 0, 0, 0)
        for check in (self.trap_detectable_check, self.trap_disarmable_check, self.trap_one_shot_check):
            trap_flags_layout.addWidget(check)
        self.trap_detect_dc_spin = QtWidgets.QSpinBox()
        self.trap_detect_dc_spin.setRange(0, 255)
        self.trap_disarm_dc_spin = QtWidgets.QSpinBox()
        self.trap_disarm_dc_spin.setRange(0, 255)
        self.trap_type_spin = QtWidgets.QSpinBox()
        self.trap_type_spin.setRange(0, 255)
        self.trap_flag_spin = QtWidgets.QSpinBox()
        self.trap_flag_spin.setRange(0, 255)
        trap_form.addRow("State", trap_flags)
        trap_form.addRow("Detect DC", self.trap_detect_dc_spin)
        trap_form.addRow("Disarm DC", self.trap_disarm_dc_spin)
        trap_form.addRow("Trap type", self.trap_type_spin)
        trap_form.addRow("Trap flag", self.trap_flag_spin)
        root.addWidget(trap)
        root.addStretch(1)
        return self._scroll_tab(content, "placeableInteractionScroll")

    def _build_scripts_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        conversation = QtWidgets.QGroupBox("Conversation")
        conversation.setObjectName("placeableConversationGroup")
        conversation_form = self._form(conversation)
        self.conversation_edit = QtWidgets.QLineEdit()
        self.conversation_edit.setObjectName("placeableConversationResrefEdit")
        self.conversation_edit.setMaxLength(16)
        conversation_form.addRow("DLG resref", self.conversation_edit)
        root.addWidget(conversation)

        scripts = QtWidgets.QGroupBox("Script Hooks")
        scripts.setObjectName("placeableScriptHooksGroup")
        scripts_form = self._form(scripts)
        self.script_edits: dict[str, QtWidgets.QLineEdit] = {}
        for label, hook in SCRIPT_HOOKS:
            edit = QtWidgets.QLineEdit()
            edit.setMaxLength(16)
            edit.setObjectName(f"placeableScript_{hook}")
            edit.setPlaceholderText("NCS resref")
            self.script_edits[hook] = edit
            scripts_form.addRow(label, edit)
        root.addWidget(scripts)
        note = QtWidgets.QLabel(
            "Terminals, containers, and puzzles become interactive through these hooks plus tag, key/item, "
            "inventory, and conversation contracts. Validate every referenced NCS, DLG, and UTI before module export."
        )
        note.setObjectName("placeableScriptCompositionNote")
        note.setWordWrap(True)
        root.addWidget(note)
        root.addStretch(1)
        return self._scroll_tab(content, "placeableScriptsScroll")

    def _build_particles_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        group = QtWidgets.QGroupBox("Attached Particle Effects")
        group.setObjectName("placeableParticleEffectsGroup")
        group_layout = QtWidgets.QVBoxLayout(group)

        self.particle_effects_table = QtWidgets.QTableWidget(0, 4)
        self.particle_effects_table.setObjectName("placeableParticleEffectsTable")
        self.particle_effects_table.setHorizontalHeaderLabels(
            ["Effect source", "X (m)", "Y (m)", "Z (m)"]
        )
        particle_header = self.particle_effects_table.horizontalHeader()
        particle_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for column in (1, 2, 3):
            particle_header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        self.particle_effects_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.particle_effects_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.particle_effects_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.particle_effects_table.itemSelectionChanged.connect(self._update_particle_action_state)
        self.particle_effects_table.setMinimumHeight(180)
        group_layout.addWidget(self.particle_effects_table)

        self.particle_effects_summary = QtWidgets.QLabel("No particle effects attached.")
        self.particle_effects_summary.setObjectName("placeableParticleEffectsSummary")
        group_layout.addWidget(self.particle_effects_summary)

        buttons = QtWidgets.QGridLayout()
        self.add_particle_effect_button = QtWidgets.QPushButton("Add From Game Library...")
        self.add_particle_effect_button.setObjectName("placeableAddParticleEffectButton")
        self.add_particle_effect_button.clicked.connect(self._open_particle_asset_picker)
        buttons.addWidget(self.add_particle_effect_button, 0, 0, 1, 2)
        self.remove_particle_effect_button = QtWidgets.QPushButton("Remove Selected")
        self.remove_particle_effect_button.setObjectName("placeableRemoveParticleEffectButton")
        self.remove_particle_effect_button.clicked.connect(self._remove_selected_particle_effects)
        buttons.addWidget(self.remove_particle_effect_button, 1, 0)
        self.reset_particle_offsets_button = QtWidgets.QPushButton("Reset Selected Offsets")
        self.reset_particle_offsets_button.setObjectName("placeableResetParticleOffsetsButton")
        self.reset_particle_offsets_button.clicked.connect(self._reset_selected_particle_offsets)
        buttons.addWidget(self.reset_particle_offsets_button, 1, 1)
        buttons.setColumnStretch(0, 1)
        buttons.setColumnStretch(1, 1)
        group_layout.addLayout(buttons)
        root.addWidget(group)

        note = QtWidgets.QLabel(
            "Attach emitters from the current target game's scanned library — a single emitter or a whole effect "
            "such as the Ebon Hawk planet holograms (plc_holoXXX). Effects are stored in the placeable "
            "document, grafted onto the preview model with their authored transforms, and simulate live "
            "in the preview viewport. Offsets shift each emitter relative to its authored position."
        )
        note.setObjectName("placeableParticleEffectsNote")
        note.setWordWrap(True)
        root.addWidget(note)
        export_limit = QtWidgets.QGroupBox("Game export")
        export_limit.setObjectName("placeableParticleExportLimitationGroup")
        export_limit_layout = QtWidgets.QVBoxLayout(export_limit)
        self.particle_export_warning = QtWidgets.QLabel(
            "Export Game Bundle bakes these emitters into a new retail KOTOR MDL/MDX, copies their textures, "
            "adds a collision-safe placeables.2da row, patches the UTP Appearance value, and reads everything back. "
            "When this saved placeable is used in Map Studio, the same resources are added to the map package automatically."
        )
        self.particle_export_warning.setObjectName("placeableParticleExportWarning")
        self.particle_export_warning.setWordWrap(True)
        export_limit_layout.addWidget(self.particle_export_warning)
        root.addWidget(export_limit)

        open_editor = QtWidgets.QPushButton("Open Particle Editor…")
        open_editor.setObjectName("placeableOpenParticleEditorButton")
        open_editor.clicked.connect(self._open_particle_editor)
        root.addWidget(open_editor, 0, QtCore.Qt.AlignLeft)
        root.addStretch(1)
        self._update_particle_action_state()
        scroll = self._scroll_tab(content, "placeableParticlesScroll")
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return scroll

    def _refresh_particle_effects_table(self) -> None:
        table = self.particle_effects_table
        table.blockSignals(True)
        try:
            table.setRowCount(len(self._particle_effects))
            for row, record in enumerate(self._particle_effects):
                label = f"{record.get('game', '')}:{record.get('model', '')}:{record.get('node', '')}"
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
                offset = list(record.get("offset") or (0.0, 0.0, 0.0))
                for axis in range(3):
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(-1000.0, 1000.0)
                    spin.setDecimals(3)
                    spin.setSingleStep(0.05)
                    spin.setValue(float(offset[axis]))
                    spin.valueChanged.connect(
                        lambda value, r=row, a=axis: self._on_particle_offset_changed(r, a, value)
                    )
                    table.setCellWidget(row, 1 + axis, spin)
        finally:
            table.blockSignals(False)
        self._update_particle_action_state()

    def _update_particle_action_state(self) -> None:
        rows = {index.row() for index in self.particle_effects_table.selectedIndexes()}
        has_selection = bool(rows)
        self.remove_particle_effect_button.setEnabled(has_selection)
        self.reset_particle_offsets_button.setEnabled(has_selection)
        count = len(self._particle_effects)
        self.particle_effects_summary.setText(
            "No particle effects attached." if not count else f"{count} emitter attachment(s)."
        )

    def _reset_selected_particle_offsets(self) -> None:
        rows = sorted({index.row() for index in self.particle_effects_table.selectedIndexes()})
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._particle_effects):
                self._particle_effects[row]["offset"] = [0.0, 0.0, 0.0]
        self._refresh_particle_effects_table()
        self._document_edited()

    def _open_particle_editor(self) -> None:
        opener = getattr(self.parent(), "_open_particle_editor_window", None)
        if callable(opener):
            opener()
        else:
            self.statusBar().showMessage("Open Particle Editor from the main Tools menu.", 6000)

    def _on_particle_offset_changed(self, row: int, axis: int, value: float) -> None:
        if self._updating_document or row >= len(self._particle_effects):
            return
        offset = list(self._particle_effects[row].get("offset") or [0.0, 0.0, 0.0])
        while len(offset) < 3:
            offset.append(0.0)
        offset[axis] = float(value)
        self._particle_effects[row]["offset"] = offset
        self._document_edited()

    def _remove_selected_particle_effects(self) -> None:
        rows = sorted({index.row() for index in self.particle_effects_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        choice = QtWidgets.QMessageBox.question(
            self,
            "Remove Particle Attachments?",
            f"Remove {len(rows)} selected emitter attachment(s) from this placeable?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if choice != QtWidgets.QMessageBox.Yes:
            return
        for row in rows:
            if 0 <= row < len(self._particle_effects):
                del self._particle_effects[row]
        self._refresh_particle_effects_table()
        self._document_edited()

    def _open_particle_asset_picker(self) -> None:
        game = self.game_combo.currentText().strip().upper() or "K2"
        try:
            records = _pick_particle_effects(self, game, self._resource_manager)
        except Exception as exc:
            self.statusBar().showMessage(f"Particle library unavailable: {exc}", 8000)
            return
        if not records:
            return
        self._particle_effects.extend(records)
        self._refresh_particle_effects_table()
        start_row = len(self._particle_effects) - len(records)
        if start_row >= 0:
            self.particle_effects_table.selectRow(start_row)
        self._document_edited()
        self.statusBar().showMessage(
            f"Attached {len(records)} emitter(s). They simulate live in the preview.", 8000
        )

    def _build_resources_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        refs = QtWidgets.QGroupBox("Custom Visual References")
        refs.setObjectName("placeableCustomResourcesGroup")
        refs_form = self._form(refs)
        self.mdl_resref_edit = QtWidgets.QLineEdit()
        self.mdl_resref_edit.setMaxLength(16)
        self.mdx_resref_edit = QtWidgets.QLineEdit()
        self.mdx_resref_edit.setMaxLength(16)
        self.pwk_resref_edit = QtWidgets.QLineEdit()
        self.pwk_resref_edit.setMaxLength(16)
        self.texture_resrefs_edit = QtWidgets.QPlainTextEdit()
        self.texture_resrefs_edit.setPlaceholderText("One texture resref or resref.tga / .tpc / .txi per line")
        refs_form.addRow("MDL resref", self.mdl_resref_edit)
        refs_form.addRow("MDX resref", self.mdx_resref_edit)
        refs_form.addRow("PWK resref", self.pwk_resref_edit)
        refs_form.addRow("Textures", self.texture_resrefs_edit)
        root.addWidget(refs)

        note = QtWidgets.QLabel(
            "The library document stores lightweight resource addresses only. Model, texture, walkmesh, script, "
            "dialog, and item bytes remain external resources and must resolve during module preflight."
        )
        note.setObjectName("placeableResourceAddressNote")
        note.setWordWrap(True)
        root.addWidget(note)
        self.resource_tree = QtWidgets.QTreeWidget()
        self.resource_tree.setObjectName("placeableResourceDependencyTree")
        self.resource_tree.setHeaderLabels(["Resource", "Type", "Source", "Status"])
        self.resource_tree.setRootIsDecorated(False)
        self.resource_tree.setAlternatingRowColors(True)
        root.addWidget(self.resource_tree, 1)
        return self._scroll_tab(content, "placeableResourcesScroll")

    def _build_readiness_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        self.readiness_labels: dict[str, QtWidgets.QLabel] = {}
        rows = (
            ("library", "Library-ready", "The reusable JSON asset passes document validation."),
            ("module", "Bundle-ready", "UTP, model, appearance mapping, and dependencies pass export preflight."),
            ("game", "Manual game proof", "A packaged module has spawned this placeable in KOTOR."),
        )
        for key, title, explanation in rows:
            group = QtWidgets.QGroupBox(title)
            group.setObjectName(f"placeableReadiness_{key}")
            group_layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel("Not checked")
            label.setObjectName(f"placeableReadinessStatus_{key}")
            label.setWordWrap(True)
            group_layout.addWidget(label)
            detail = QtWidgets.QLabel(explanation)
            detail.setWordWrap(True)
            group_layout.addWidget(detail)
            root.addWidget(group)
            self.readiness_labels[key] = label
        self.evidence_status_label = QtWidgets.QLabel(
            "Vanilla structural evidence has not been attached. Parser acceptance is not engine proof."
        )
        self.evidence_status_label.setObjectName("placeableStructuralEvidenceStatus")
        self.evidence_status_label.setWordWrap(True)
        root.addWidget(self.evidence_status_label)
        self.validation_issues_tree = QtWidgets.QTreeWidget()
        self.validation_issues_tree.setObjectName("placeableValidationIssuesTree")
        self.validation_issues_tree.setHeaderLabels(["Severity", "Check", "Message"])
        self.validation_issues_tree.setRootIsDecorated(False)
        self.validation_issues_tree.setAlternatingRowColors(True)
        root.addWidget(self.validation_issues_tree, 1)
        return self._scroll_tab(content, "placeableReadinessScroll")

    def _build_statusbar(self) -> None:
        self.statusBar().setObjectName("placeableBuilderStatusBar")
        self.statusBar().showMessage("Placeable Builder ready. Create or open an asset to begin.")

    # ------------------------------------------------------------------
    # Public service-facing API

    def set_library_root(self, root: str | Path) -> None:
        self._library_root = Path(root) if str(root or "") else Path()
        self.library_root_label.setText(str(self._library_root) if str(self._library_root) else "Library folder not connected")
        self.library_root_label.setToolTip(str(self._library_root) if str(self._library_root) else "")

    def library_root(self) -> Path:
        return self._library_root

    @property
    def placeable_library_root(self) -> Path:
        """Stable path consumed by the main shell and Map Studio refresh bridge."""

        return self._library_root

    def notify_library_changed(self) -> None:
        """Publish a completed save/import/delete performed by the owning service."""

        self.libraryChanged.emit(str(self._library_root))

    def accept_library_save(self, document: object | None = None) -> None:
        """Service callback after an atomic library save has succeeded."""

        if document is not None:
            self.set_document(document, mark_clean=True)
        else:
            self.mark_clean()
        self.notify_library_changed()
        self.statusBar().showMessage("Placeable saved to the library.", 5000)

    def set_library_rows(self, rows: object) -> None:
        self._library_rows = [_mapping(row) for row in (rows or ())]
        self._populate_library_model()
        self._rebuild_library_view()

    def library_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._library_rows))

    def selected_library_row(self) -> dict[str, Any] | None:
        selected = self.library_tree.selectionModel().selectedRows(0)
        if not selected:
            return None
        source_index = self.library_proxy_model.mapToSource(selected[0])
        row = source_index.data(LIBRARY_ROW_ROLE)
        return deepcopy(row) if isinstance(row, dict) else None

    def activate_selected_library_asset(self) -> None:
        row = self.selected_library_row()
        if row is not None:
            self.libraryAssetActivated.emit(row)

    def set_document(self, document: object, *, mark_clean: bool = True) -> None:
        data = _mapping(document)
        self._updating_document = True
        try:
            self._document_passthrough = deepcopy(data)
            self._resource_passthrough = deepcopy(_mapping(data.get("resources")))
            self._scripts_passthrough = deepcopy(_mapping(data.get("scripts")))
            self._base_template_passthrough = deepcopy(_mapping(data.get("base_template"))) or None
            self.display_name_edit.setText(str(data.get("display_name") or ""))
            self.template_resref_edit.setText(str(data.get("template_resref") or ""))
            self.tag_edit.setText(str(data.get("tag") or ""))
            self.game_combo.setCurrentText(str(data.get("game") or "K2").upper())
            self.secondary_motion_editor.set_profile(
                _mapping(data.get("secondary_motion"))
                or default_secondary_motion_profile(target_game=str(data.get("game") or "K2"))
            )
            self._set_combo_data(self.category_combo, str(data.get("category") or "decor").lower())
            self.description_edit.setPlainText(str(data.get("description") or ""))
            self.comment_edit.setText(str(data.get("comment") or ""))
            self._set_combo_data(self.visual_source_combo, str(data.get("visual_source") or "stock").lower())
            self.base_template_edit.setText(_resource_resref(data.get("base_template")))
            appearance_id = data.get("appearance_id")
            self.appearance_id_spin.setValue(int(appearance_id) if appearance_id is not None else -1)
            appearance_evidence = _mapping(data.get("appearance_evidence"))
            metadata = _mapping(data.get("metadata"))
            self.appearance_model_hint_edit.setText(
                str(appearance_evidence.get("model_resref") or metadata.get("model_resref_hint") or "")
            )

            gameplay = _mapping(data.get("gameplay"))
            self.static_check.setChecked(bool(gameplay.get("static")))
            self.useable_check.setChecked(bool(gameplay.get("useable", True)))
            self.has_inventory_check.setChecked(bool(gameplay.get("has_inventory")))
            self.party_interact_check.setChecked(bool(gameplay.get("party_interact")))
            self.plot_check.setChecked(bool(gameplay.get("plot")))
            self.min1_hp_check.setChecked(bool(gameplay.get("min1_hp")))
            self.not_blastable_check.setChecked(bool(gameplay.get("not_blastable")))
            self.maximum_hp_spin.setValue(int(gameplay.get("maximum_hp", 1)))
            self.current_hp_spin.setValue(int(gameplay.get("current_hp", gameplay.get("maximum_hp", 1))))
            self.hardness_spin.setValue(int(gameplay.get("hardness", 0)))
            self.lockable_check.setChecked(bool(gameplay.get("lockable")))
            self.locked_check.setChecked(bool(gameplay.get("locked")))
            self.key_required_check.setChecked(bool(gameplay.get("key_required")))
            self.auto_remove_key_check.setChecked(bool(gameplay.get("auto_remove_key")))
            self.key_name_edit.setText(str(gameplay.get("key_name") or ""))
            self.lock_dc_spin.setValue(int(gameplay.get("lock_dc", 0)))
            self.unlock_dc_spin.setValue(int(gameplay.get("unlock_dc", 0)))
            self.inventory_items_edit.setPlainText("\n".join(str(item) for item in gameplay.get("inventory_items") or ()))
            self.trap_detectable_check.setChecked(bool(gameplay.get("trap_detectable")))
            self.trap_disarmable_check.setChecked(bool(gameplay.get("trap_disarmable")))
            self.trap_one_shot_check.setChecked(bool(gameplay.get("trap_one_shot")))
            self.trap_detect_dc_spin.setValue(int(gameplay.get("trap_detect_dc", 0)))
            self.trap_disarm_dc_spin.setValue(int(gameplay.get("trap_disarm_dc", 0)))
            self.trap_type_spin.setValue(int(gameplay.get("trap_type", 0)))
            self.trap_flag_spin.setValue(int(gameplay.get("trap_flag", 0)))
            self.conversation_edit.setText(str(gameplay.get("conversation_resref") or ""))
            scripts = _mapping(data.get("scripts"))
            for hook, edit in self.script_edits.items():
                edit.setText(str(scripts.get(hook) or ""))

            resources = _mapping(data.get("resources"))
            self.mdl_resref_edit.setText(_resource_resref(resources.get("mdl")))
            self.mdx_resref_edit.setText(_resource_resref(resources.get("mdx")))
            self.pwk_resref_edit.setText(_resource_resref(resources.get("pwk")))
            texture_lines = []
            for address in resources.get("textures") or ():
                row = _mapping(address)
                resref = str(row.get("resref") or row.get("path") or "")
                restype = str(row.get("restype") or "").lower()
                texture_lines.append(f"{resref}.{restype}" if resref and restype else resref)
            self.texture_resrefs_edit.setPlainText("\n".join(line for line in texture_lines if line))
            metadata = _mapping(data.get("metadata"))
            self._particle_effects = [
                dict(record) for record in (metadata.get("particle_effects") or [])
                if isinstance(record, dict)
            ]
            self._refresh_particle_effects_table()
            self._update_visual_contract_text()
        finally:
            self._updating_document = False
        if mark_clean:
            self.mark_clean()
        else:
            self._set_dirty(True)

    def current_document(self) -> dict[str, Any]:
        data = deepcopy(self._document_passthrough)
        data.update(
            {
                "file_type": str(data.get("file_type") or "ghostrigger.placeable_asset"),
                "schema_version": int(data.get("schema_version") or 2),
                "asset_id": str(data.get("asset_id") or ""),
                "game": self.game_combo.currentText().upper(),
                "template_resref": self.template_resref_edit.text().strip().lower(),
                "tag": self.tag_edit.text().strip(),
                "display_name": self.display_name_edit.text().strip(),
                "description": self.description_edit.toPlainText().strip(),
                "comment": self.comment_edit.text().strip(),
                "category": str(self.category_combo.currentData() or "decor"),
                "visual_source": str(self.visual_source_combo.currentData() or "stock"),
                "appearance_id": None if self.appearance_id_spin.value() < 0 else self.appearance_id_spin.value(),
            }
        )

        gameplay = deepcopy(_mapping(data.get("gameplay")))
        gameplay.update(
            {
                "static": self.static_check.isChecked(),
                "useable": self.useable_check.isChecked(),
                "has_inventory": self.has_inventory_check.isChecked(),
                "inventory_items": self._nonempty_lines(self.inventory_items_edit.toPlainText()),
                "lockable": self.lockable_check.isChecked(),
                "locked": self.locked_check.isChecked(),
                "key_required": self.key_required_check.isChecked(),
                "key_name": self.key_name_edit.text().strip().lower(),
                "auto_remove_key": self.auto_remove_key_check.isChecked(),
                "unlock_dc": self.unlock_dc_spin.value(),
                "lock_dc": self.lock_dc_spin.value(),
                "trap_detectable": self.trap_detectable_check.isChecked(),
                "trap_detect_dc": self.trap_detect_dc_spin.value(),
                "trap_disarmable": self.trap_disarmable_check.isChecked(),
                "trap_disarm_dc": self.trap_disarm_dc_spin.value(),
                "trap_flag": self.trap_flag_spin.value(),
                "trap_one_shot": self.trap_one_shot_check.isChecked(),
                "trap_type": self.trap_type_spin.value(),
                "maximum_hp": self.maximum_hp_spin.value(),
                "current_hp": self.current_hp_spin.value(),
                "hardness": self.hardness_spin.value(),
                "plot": self.plot_check.isChecked(),
                "min1_hp": self.min1_hp_check.isChecked(),
                "not_blastable": self.not_blastable_check.isChecked(),
                "party_interact": self.party_interact_check.isChecked(),
                "conversation_resref": self.conversation_edit.text().strip().lower(),
            }
        )
        data["gameplay"] = gameplay

        scripts = deepcopy(self._scripts_passthrough)
        for hook, edit in self.script_edits.items():
            value = edit.text().strip().lower()
            if value:
                scripts[hook] = value
            else:
                scripts.pop(hook, None)
        data["scripts"] = scripts

        game = data["game"]
        resources = deepcopy(self._resource_passthrough)
        resources.update(
            {
                "mdl": self._edited_resource_address(resources.get("mdl"), game, self.mdl_resref_edit.text(), "MDL"),
                "mdx": self._edited_resource_address(resources.get("mdx"), game, self.mdx_resref_edit.text(), "MDX"),
                "pwk": self._edited_resource_address(resources.get("pwk"), game, self.pwk_resref_edit.text(), "PWK"),
                "textures": self._edited_texture_addresses(
                    resources.get("textures") or (),
                    game,
                    self._nonempty_lines(self.texture_resrefs_edit.toPlainText()),
                ),
            }
        )
        data["resources"] = resources
        self.secondary_motion_editor.set_target_game(game)
        data["secondary_motion"] = self.secondary_motion_editor.profile()
        base_resref = self.base_template_edit.text().strip()
        if base_resref:
            existing = deepcopy(self._base_template_passthrough or {})
            if str(existing.get("resref") or "").lower() != base_resref.lower():
                existing = _resource_address(game, base_resref, "UTP") or {}
                existing["scheme"] = "game_resource"
                existing["layer"] = "base_game"
            data["base_template"] = existing
        else:
            data["base_template"] = None
        metadata = deepcopy(_mapping(data.get("metadata")))
        model_hint = self.appearance_model_hint_edit.text().strip().lower()
        if model_hint:
            metadata["model_resref_hint"] = model_hint
        else:
            metadata.pop("model_resref_hint", None)
        if self._particle_effects:
            metadata["particle_effects"] = deepcopy(self._particle_effects)
        else:
            metadata.pop("particle_effects", None)
        data["metadata"] = metadata
        return data

    def set_readiness(self, report: object, *, reveal: bool = False) -> None:
        if isinstance(report, Mapping):
            snapshot = dict(report)
        else:
            snapshot = {
                "issues": list(getattr(report, "issues", ()) or ()),
                "document_valid": bool(getattr(report, "document_valid", False)),
                "utp_export_ready": bool(getattr(report, "utp_export_ready", False)),
                "structural_evidence_ready": bool(getattr(report, "structural_evidence_ready", False)),
                "engine_ready": bool(getattr(report, "engine_ready", False)),
            }
        snapshot.setdefault("issues", [])
        self._readiness = snapshot
        document_valid = bool(snapshot.get("document_valid"))
        utp_ready = bool(snapshot.get("utp_export_ready"))
        structural_ready = bool(snapshot.get("structural_evidence_ready"))
        engine_ready = bool(snapshot.get("engine_ready"))
        self.readiness_labels["library"].setText("Ready" if document_valid else "Blocked — resolve document issues")
        self.readiness_labels["module"].setText(
            "Ready for module dependency preflight" if utp_ready and structural_ready else "Blocked — UTP or structural proof incomplete"
        )
        self.readiness_labels["game"].setText(
            "Confirmed in KOTOR" if engine_ready else "Not proven — package, install, warp, and spawn in game"
        )
        self.evidence_status_label.setText(
            "Vanilla/base UTP structure and appearance mapping evidence are attached."
            if structural_ready
            else "Vanilla structural evidence is incomplete. Parser acceptance is not engine proof."
        )
        self.validation_issues_tree.clear()
        for issue in snapshot.get("issues") or ():
            row = _mapping(issue)
            if row:
                severity = str(row.get("severity") or "info")
                code = str(row.get("code") or "")
                message = str(row.get("message") or issue)
                fix_hint = str(row.get("fix_hint") or "")
            else:
                severity = str(getattr(issue, "severity", "info"))
                code = str(getattr(issue, "code", ""))
                message = str(getattr(issue, "message", issue))
                fix_hint = str(getattr(issue, "fix_hint", ""))
            item = QtWidgets.QTreeWidgetItem([severity.title(), code, message])
            if fix_hint:
                item.setToolTip(2, fix_hint)
            self.validation_issues_tree.addTopLevelItem(item)
        if reveal:
            self.inspector_tabs.setCurrentWidget(self.inspector_tabs.widget(7))
        self.statusBar().showMessage("Placeable validation updated.", 4000)

    def readiness_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._readiness)

    def set_resource_rows(self, rows: object) -> None:
        self._resource_rows = [_mapping(row) for row in (rows or ())]
        self.resource_tree.clear()
        for row in self._resource_rows:
            resource = str(row.get("label") or row.get("resref") or row.get("path") or "")
            restype = str(row.get("restype") or row.get("resource_type") or "")
            source = str(row.get("source") or row.get("layer") or "")
            status = str(row.get("status") or ("Resolved" if row.get("resolved") else "Unresolved"))
            self.resource_tree.addTopLevelItem(QtWidgets.QTreeWidgetItem([resource, restype, source, status]))

    def set_preview_model(
        self,
        model: object,
        *,
        texture_dir: str = "",
        resource_manager: object = None,
        game: str = "",
        message: str = "",
    ) -> None:
        if resource_manager is not None:
            self._resource_manager = resource_manager
            self.preview_viewport.set_resource_manager(resource_manager, game or self.game_combo.currentText())
        self.preview_viewport.load_model(model, texture_dir)
        self.preview_status_label.setText(
            message
            or ("Preview loaded. This confirms editor rendering only; validate and prove the exported UTP in game."
                if model is not None else "No preview model is loaded.")
        )

    def set_renderer_settings(self, settings: RendererSettings | dict | None) -> None:
        self._renderer_settings = (
            settings if isinstance(settings, RendererSettings) else RendererSettings.from_settings(settings or {})
        )
        self.preview_viewport.set_renderer_settings(self._renderer_settings)

    def set_navigation_profile(self, profile: object) -> None:
        self._navigation_profile = profile or DEFAULT_VIEWPORT_NAVIGATION_PROFILE
        self.preview_viewport.set_navigation_profile(self._navigation_profile)

    def mark_clean(self) -> None:
        self._set_dirty(False)

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Action and presentation helpers

    def new_placeable(self) -> None:
        self._reset_document(emit_request=True)

    def clone_placeable(self) -> None:
        self.cloneRequested.emit(self.current_document())
        self.statusBar().showMessage("Clone requested. The library service will assign a new stable asset ID.", 5000)

    def request_save_to_library(self) -> None:
        self.saveToLibraryRequested.emit(self.current_document())
        self.statusBar().showMessage("Save to Placeable Library requested.", 4000)

    def request_export_utp(self) -> None:
        self.exportUtpRequested.emit(self.current_document())
        self.statusBar().showMessage("Game-bundle export requested; target-game resource preflight must pass first.", 5000)

    def request_validate(self) -> None:
        self.validateRequested.emit(self.current_document())
        self.statusBar().showMessage("Placeable validation requested.", 4000)

    def _request_open(self) -> None:
        self.openRequested.emit()

    def _request_open_library_folder(self) -> None:
        self.openLibraryFolderRequested.emit(str(self._library_root))

    def _reset_document(self, *, emit_request: bool) -> None:
        blank = {
            "file_type": "ghostrigger.placeable_asset",
            "schema_version": 2,
            "asset_id": "",
            "game": "K2",
            "template_resref": "",
            "tag": "",
            "display_name": "",
            "description": "",
            "comment": "",
            "category": "decor",
            "visual_source": "stock",
            "appearance_id": None,
            "gameplay": {"useable": True, "maximum_hp": 1, "current_hp": 1},
            "scripts": {},
            "resources": {"mdl": None, "mdx": None, "pwk": None, "textures": []},
            "base_template": None,
            "base_evidence": None,
            "appearance_evidence": None,
            "secondary_motion": default_secondary_motion_profile(target_game="K2"),
            "metadata": {},
        }
        self.set_document(blank, mark_clean=True)
        self.set_readiness({"issues": [], "document_valid": False, "utp_export_ready": False, "structural_evidence_ready": False, "engine_ready": False})
        self.inspector_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("New placeable. Choose a base UTP or stock appearance, then validate.")
        if emit_request:
            self.newRequested.emit()

    def _connect_document_controls(self) -> None:
        controls: list[QtCore.QObject] = [
            self.display_name_edit,
            self.template_resref_edit,
            self.tag_edit,
            self.game_combo,
            self.category_combo,
            self.description_edit,
            self.comment_edit,
            self.visual_source_combo,
            self.base_template_edit,
            self.appearance_id_spin,
            self.appearance_model_hint_edit,
            self.inventory_items_edit,
            self.conversation_edit,
            self.mdl_resref_edit,
            self.mdx_resref_edit,
            self.pwk_resref_edit,
            self.texture_resrefs_edit,
            self.maximum_hp_spin,
            self.current_hp_spin,
            self.hardness_spin,
            self.lock_dc_spin,
            self.unlock_dc_spin,
            self.trap_detect_dc_spin,
            self.trap_disarm_dc_spin,
            self.trap_type_spin,
            self.trap_flag_spin,
            self.key_name_edit,
            *self.script_edits.values(),
            self.static_check,
            self.useable_check,
            self.has_inventory_check,
            self.party_interact_check,
            self.plot_check,
            self.min1_hp_check,
            self.not_blastable_check,
            self.lockable_check,
            self.locked_check,
            self.key_required_check,
            self.auto_remove_key_check,
            self.trap_detectable_check,
            self.trap_disarmable_check,
            self.trap_one_shot_check,
        ]
        for control in controls:
            if isinstance(control, QtWidgets.QLineEdit):
                control.textEdited.connect(self._document_edited)
            elif isinstance(control, QtWidgets.QPlainTextEdit):
                control.textChanged.connect(self._document_edited)
            elif isinstance(control, QtWidgets.QComboBox):
                control.currentIndexChanged.connect(self._document_edited)
            elif isinstance(control, QtWidgets.QAbstractSpinBox):
                control.valueChanged.connect(self._document_edited)
            elif isinstance(control, QtWidgets.QAbstractButton):
                control.toggled.connect(self._document_edited)

    def _document_edited(self, *_args) -> None:
        if self._updating_document:
            return
        self._set_dirty(True)
        self.documentChanged.emit(self.current_document())

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        title = "GhostStudio — Placeable Builder"
        self.setWindowTitle(f"{title} *" if self._dirty else title)

    def _update_visual_contract_text(self, *_args) -> None:
        custom = self.visual_source_combo.currentData() == "custom"
        if custom:
            text = (
                "Custom visuals require paired MDL/MDX resources, the matching placeables.2da appearance mapping, "
                "PWK where applicable, texture dependencies, and byte/source evidence before module export."
            )
        else:
            text = (
                "Stock visuals use a proven placeables.2da row plus a known-loadable base UTP. "
                "The appearance-to-model mapping still requires source evidence."
            )
        self.visual_contract_label.setText(text)

    def _rebuild_library_view(self, *_args) -> None:
        search = self.library_search_edit.text().strip().lower()
        wanted_game = self.library_game_filter.currentText()
        wanted_category = str(self.library_category_filter.currentData() or "").lower()
        self.library_proxy_model.set_placeable_filters(
            query=search,
            game=wanted_game,
            category=wanted_category,
        )

    def _populate_library_model(self) -> None:
        self.library_model.removeRows(0, self.library_model.rowCount())
        for row in self._library_rows:
            game = str(row.get("game") or "").upper()
            category = str(row.get("subcategory") or row.get("placeable_category") or "").lower()
            metadata = _mapping(row.get("metadata"))
            if bool(row.get("engine_ready") or metadata.get("engine_ready")):
                readiness = "Game-proven"
            elif bool(row.get("structural_evidence") or metadata.get("structural_evidence_ready")):
                readiness = "Structure-proven"
            elif bool(metadata.get("document_valid")):
                readiness = "Library-ready"
            else:
                readiness = str(row.get("confidence") or "Unproven").replace("_", " ").title()
            label = str(row.get("label") or row.get("display_name") or row.get("resref") or "Unnamed placeable")
            search_text = " ".join(
                str(row.get(key) or "")
                for key in ("label", "display_name", "resref", "template_resref", "tag", "subcategory", "category")
            ).lower()
            items = [QtGui.QStandardItem(value) for value in (label, game, category.title(), readiness)]
            items[0].setData(deepcopy(row), LIBRARY_ROW_ROLE)
            items[0].setData(game, LIBRARY_GAME_ROLE)
            items[0].setData(category, LIBRARY_CATEGORY_ROLE)
            items[0].setData(search_text, LIBRARY_SEARCH_ROLE)
            warning = str(row.get("warning") or "")
            if warning:
                items[0].setToolTip(warning)
                items[3].setToolTip(warning)
            for item in items:
                item.setEditable(False)
            self.library_model.appendRow(items)
        for column in range(4):
            self.library_tree.resizeColumnToContents(column)

    def _library_selection_changed(self) -> None:
        row = self.selected_library_row()
        if row is None:
            return
        self.statusBar().showMessage(
            f"Selected {row.get('label') or row.get('resref') or 'placeable'}. Double-click or press Enter to open.", 4000
        )

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(value, QtCore.Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _nonempty_lines(text: str) -> list[str]:
        return [line.strip().lower() for line in str(text or "").splitlines() if line.strip()]

    @staticmethod
    def _texture_address(game: str, value: str) -> dict[str, Any]:
        path = str(value or "").strip()
        stem, dot, suffix = path.rpartition(".")
        restype = suffix.upper() if dot and suffix.lower() in {"tga", "tpc", "txi"} else "TGA"
        resref = stem if dot and suffix.lower() in {"tga", "tpc", "txi"} else path
        return _resource_address(game, resref, restype) or {}

    @staticmethod
    def _edited_resource_address(existing: object, game: str, resref: str, restype: str) -> dict[str, Any] | None:
        wanted = str(resref or "").strip().lower()
        if not wanted:
            return None
        row = _mapping(existing)
        if str(row.get("resref") or "").lower() == wanted and str(row.get("restype") or "").upper() == restype:
            return deepcopy(row)
        return _resource_address(game, wanted, restype)

    @classmethod
    def _edited_texture_addresses(cls, existing: object, game: str, values: list[str]) -> list[dict[str, Any]]:
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for address in existing or ():
            row = _mapping(address)
            key = (str(row.get("resref") or "").lower(), str(row.get("restype") or "").upper())
            if key[0] and key[1]:
                indexed[key] = row
        output: list[dict[str, Any]] = []
        for value in values:
            generated = cls._texture_address(game, value)
            key = (str(generated.get("resref") or "").lower(), str(generated.get("restype") or "").upper())
            output.append(deepcopy(indexed.get(key) or generated))
        return output

    def apply_ghost_theme(self, theme) -> None:
        hook = getattr(self.preview_viewport, "apply_ghost_theme", None)
        if callable(hook):
            hook(theme)
        self.update()

    def apply_ghost_layout(self, layout) -> None:
        self.resize(layout.main_width, layout.main_height)
        self.workspace_splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
        library = layout.panel("library")
        inspector = layout.panel("properties")
        inspector_width = max(
            inspector.preferred_width,
            5 * layout.spacing_value("tabWidth", 78),
        )
        preview_width = layout.viewport.preferred_width
        self.library_panel.setMinimumWidth(library.min_width)
        self.inspector_tabs.setMinimumWidth(inspector_width)
        self.preview_panel.setMinimumWidth(layout.viewport.min_width)
        self.preview_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding
        )
        minimum_width = (
            library.min_width
            + layout.viewport.min_width
            + inspector_width
            + (2 * self.workspace_splitter.handleWidth())
        )
        self.setMinimumWidth(max(self.minimumWidth(), minimum_width))
        self.workspace_splitter.setSizes([library.preferred_width, preview_width, inspector_width])
        toolbar_layout = layout.toolbar("main")
        self.placeable_toolbar.setIconSize(QtCore.QSize(toolbar_layout.icon_size, toolbar_layout.icon_size))
        self.placeable_toolbar.setMinimumHeight(toolbar_layout.height)
        input_height = layout.spacing_value("inputHeight", 24)
        for widget in [
            *self.findChildren(QtWidgets.QLineEdit),
            *self.findChildren(QtWidgets.QComboBox),
            *self.findChildren(QtWidgets.QSpinBox),
            *self.findChildren(QtWidgets.QDoubleSpinBox),
        ]:
            widget.setMinimumHeight(input_height)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._dirty:
            event.accept()
            return
        choice = QtWidgets.QMessageBox.question(
            self,
            "Unsaved Placeable",
            "This placeable has unsaved changes. Save it to the Placeable Library before closing?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if choice == QtWidgets.QMessageBox.Save:
            self.request_save_to_library()
            event.ignore()
        elif choice == QtWidgets.QMessageBox.Discard:
            event.accept()
        else:
            event.ignore()


__all__ = ["PLACEABLE_CATEGORIES", "QtPlaceableBuilderWindow"]
