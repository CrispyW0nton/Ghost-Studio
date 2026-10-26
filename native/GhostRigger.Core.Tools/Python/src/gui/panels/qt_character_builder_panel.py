"""Qt character builder panels and window for GhostRigger.

M2 (T201–T207) rewrote :class:`QtCharacterBuilderWindow` as a proper
AccuRig-style HUD: a ``QMainWindow`` with a top toolbar (mode switcher
+ camera presets + tool toggles), a horizontal splitter containing the
left :class:`QtWorkflowRail` / centre viewport stack / right
:class:`QtInspectorPanel`, and the :class:`QtBottomStrip` docked at
the bottom (validation banner, anim scrubber, stats, export log).

M5 (T501–T506) is progressively replacing the legacy five-tab
:class:`QtCharacterBuilderPanel` with the new workflow service in
:mod:`src.core.headless_body_workflow`.  T501 (this task) wires the
real *Load Body* path; later tasks fill in check / rig / export.

Roadmap: knowledge_base/roadmap/02_roadmap_2026_05.md §M2 + §M5.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from src.gui.qt_lib.panels.qt_bottom_strip import QtBottomStrip
from src.gui.qt_lib.panels.qt_inspector_panel import QtInspectorPanel
from src.gui.qt_lib.panels.qt_properties_panel import QtPropertiesPanel
from src.gui.qt_lib.panels.qt_head_builder_workspace import (
    QtHeadBuilderAssetTree,
    QtHeadBuilderEvidencePanel,
    QtHeadBuilderProperties,
)
from src.gui.qt_lib.assets.qt_theme import (
    C,
    apply_theme,
    heading,
    make_horizontal_overflow_area,
    make_scrollable_panel,
    update_legacy_palette,
)
from src.gui.qt_lib.panels.qt_workflow_rail import QtWorkflowRail
from src.gui.panels.qt_secondary_motion_editor import QtSecondaryMotionEditor
from src.systems.bas.attachment_alignment import default_bas_attachment_transform
from src.systems.bas.attachment_catalog import repair_bas_body_texture_references
from src.systems.bas.head_resolution import normalize_bas_model_resref, resolve_bas_head_resref
from src.systems.bas.preview_composer import (
    bas_slot_for_preview_socket,
    bas_socket_for_slot,
    build_bas_preview_model,
)

from PySide6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)


def _issue_field(issue: Any, field: str, default: str = "") -> str:
    """Return one display field from a ValidationIssue-like object."""
    if isinstance(issue, dict):
        value = issue.get(field, default)
    else:
        value = getattr(issue, field, default)
    if field == "severity":
        value = getattr(value, "value", value)
    if value is None:
        return default
    return str(value)


def _issue_slot_text(issue: Any) -> str:
    value = (
        issue.get("slot", "")
        if isinstance(issue, dict) else
        getattr(issue, "slot", "")
    )
    value = getattr(value, "value", value)
    return "" if value is None else str(value)


def _attachment_type_from_resref(resref: str) -> str:
    name = str(resref or "").lower()
    if "lghtsbr" in name or "saber" in name:
        return "lightsaber"
    if name.startswith(("w_blstr", "w_rfl", "w_bow")):
        return "blaster"
    if name.startswith("w_"):
        return "weapon"
    if name.startswith(("i_mask", "ia_", "g_i_mask")):
        return "headgear"
    return "item"


class _ValidationIssueTableModel(QtCore.QAbstractTableModel):
    """Small table model for the M9/T902 validation report dialog."""

    HEADERS = ("Severity", "Code", "Message", "Slot", "Node")

    def __init__(self, issues: list[Any], parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._issues = list(issues or [])

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._issues)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role != QtCore.Qt.DisplayRole or orientation != QtCore.Qt.Horizontal:
            return None
        if 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._issues)):
            return None
        issue = self._issues[index.row()]
        if role == QtCore.Qt.UserRole:
            return issue
        if role != QtCore.Qt.DisplayRole:
            return None
        if isinstance(issue, str):
            return issue if index.column() == 2 else ""
        columns = (
            _issue_field(issue, "severity"),
            _issue_field(issue, "code"),
            _issue_field(issue, "message"),
            _issue_slot_text(issue),
            _issue_field(issue, "node"),
        )
        if 0 <= index.column() < len(columns):
            return columns[index.column()]
        return None

    def issue_at(self, row: int) -> Any:
        if 0 <= row < len(self._issues):
            return self._issues[row]
        return None


class QtValidationReportDialog(QtWidgets.QDialog):
    """Full validation report dialog with a jump-to-node action."""

    jumpRequested = QtCore.Signal(str)

    def __init__(self, issues: list[Any], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Validation Report")
        self.resize(780, 420)
        self._model = _ValidationIssueTableModel(issues, self)
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._table = QtWidgets.QTableView(self)
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            2,
            QtWidgets.QHeaderView.Stretch,
        )
        self._table.doubleClicked.connect(lambda _idx: self._emit_jump())
        layout.addWidget(self._table, 1)

        if self._model.rowCount() > 0:
            self._table.selectRow(0)
        else:
            self._table.setToolTip("No validation issues have been reported.")

        button_row = QtWidgets.QHBoxLayout()
        self._jump_btn = QtWidgets.QPushButton("Jump to Bone")
        self._jump_btn.setToolTip("Select the issue's node in the viewport.")
        self._jump_btn.clicked.connect(self._emit_jump)
        button_row.addWidget(self._jump_btn)
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        selection = self._table.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(lambda *_args: self._refresh_jump_state())
        self._refresh_jump_state()

    def _selected_issue(self) -> Any:
        selection = self._table.selectionModel()
        if selection is None:
            return None
        rows = selection.selectedRows()
        if not rows:
            return None
        return self._model.issue_at(rows[0].row())

    def _selected_node(self) -> str:
        return _issue_field(self._selected_issue(), "node")

    def _refresh_jump_state(self) -> None:
        self._jump_btn.setEnabled(bool(self._selected_node()))

    def _emit_jump(self) -> None:
        node = self._selected_node()
        if node:
            self.jumpRequested.emit(node)


# ── CharacterMode wiring (pykotor-safe) ─────────────────────────────────────
# ``src.core.__init__`` eagerly imports the loader stack (pykotor).  We
# isolate the failure so the window still loads when those deps are
# missing — the mode switcher simply renders with disabled buttons.
try:
    from src.core.geometry.model_data import CharacterMode
    _CHARACTER_MODE_AVAILABLE = True
except Exception:                                       # pragma: no cover
    CharacterMode = None                                # type: ignore[assignment]
    _CHARACTER_MODE_AVAILABLE = False


# ── QtViewportWidget import (also pykotor-safe) ─────────────────────────────
# ``qt_viewport`` pulls in viewport_core which is heavy.  When it
# fails (typical in unit-test sandboxes), we fall back to a labelled
# placeholder QWidget so the rest of the shell still composes.
try:
    from src.gui.qt_lib.viewports.qt_viewport import QtCharacterBuilderViewportWidget
    _VIEWPORT_AVAILABLE = True
except Exception as _vp_exc:                            # pragma: no cover
    _VIEWPORT_AVAILABLE = False
    _VIEWPORT_IMPORT_ERROR = f"{type(_vp_exc).__name__}: {_vp_exc}"
    QtCharacterBuilderViewportWidget = None             # type: ignore[assignment]


def _import_model_data():
    try:
        from src.core.geometry.model_data import CharacterScene
    except ImportError:
        from core.geometry.model_data import CharacterScene  # type: ignore
    return CharacterScene


def _import_scene_io():
    try:
        from src.core.geometry.model_data import SceneIO
    except ImportError:
        from core.geometry.model_data import SceneIO  # type: ignore
    return SceneIO


# ── QSettings keys (M2 / T207) ──────────────────────────────────────────────
# Centralised so renames are one-place changes.
_QSETTINGS_ORG  = "GhostRigger"
_QSETTINGS_APP  = "CharacterBuilder"

_QSK_GEOMETRY        = "window/geometry"
_QSK_WINDOW_STATE    = "window/state"
_QSK_SPLITTER_SIZES  = "window/splitter_sizes"
_QSK_LAST_MODE       = "window/last_mode"


class QtCharacterBuilderPanel(QtWidgets.QWidget):
    """Compact launcher panel embedded in the main window's right-pane tabs.

    The original M0 implementation was a five-tab placeholder
    (Assembly / Selection / Transform / Rig / Export) full of dead
    buttons that did nothing.  M2 introduced the proper full-window
    Character Builder (:class:`QtCharacterBuilderWindow`) and M5
    completes the migration by replacing the dead tabs with a thin
    launcher that opens the real builder.

    Public attributes preserved for backward compatibility with
    ``qt_main_window.py``:
      * ``game_combo``   — K1/K2 selector (still used by the main shell)

    Signals:
      * ``launchRequested()`` — emitted when the user clicks the
        "Open Character Builder…" button.  The main window connects this
        to its existing builder-window action; if no listener connects,
        the panel opens the window itself.
    """

    launchRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._builder_window: Optional[QtWidgets.QMainWindow] = None
        self._build()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(heading("Character Builder"))
        title_row.addStretch(1)
        # K1/K2 game-version selector (preserved so qt_main_window
        # keeps its setCurrentText() / currentTextChanged() bindings).
        self.game_combo = QtWidgets.QComboBox()
        self.game_combo.addItems(["K1", "K2"])
        self.game_combo.setToolTip("Active KOTOR game version")
        title_row.addWidget(QtWidgets.QLabel("Game:"))
        title_row.addWidget(self.game_combo)
        root.addLayout(title_row)

        # Brief explanation of the new workflow.
        blurb = QtWidgets.QLabel(
            "The full Character Builder opens in its own window.  It hosts the\n"
            "AccuRig-style HUD (joint dots, mini-thumbnail, snap-view, weight\n"
            "heat-map) and the five-step KOTOR character export workflow."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color:{C.get('text2', '#888')}; padding:2px 0;")
        root.addWidget(blurb)

        # The five workflow steps as a read-only summary so the user
        # can see what the builder will guide them through.
        steps_label = QtWidgets.QLabel("Character Builder workflow:")
        steps_label.setStyleSheet(
            f"color:{C.get('gold', '#FFD700')}; font-weight:bold; padding-top:6px;"
        )
        root.addWidget(steps_label)

        steps_list = QtWidgets.QListWidget()
        steps_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        steps_list.setFocusPolicy(QtCore.Qt.NoFocus)
        steps_list.setFrameShape(QtWidgets.QFrame.NoFrame)
        steps_list.setStyleSheet(
            f"QListWidget {{ background:{C.get('bg2', '#1a1a1a')}; "
            f"               color:{C.get('text1', '#ddd')}; "
            f"               border:1px solid {C.get('bg3', '#222')}; }}"
        )
        for i, label in enumerate([
            "1. Choose Base + Load Mesh",
            "2. Assign Skeleton",
            "3. Assign Animations",
            "4. Preview",
            "5. Export MDL",
        ], start=1):
            QtWidgets.QListWidgetItem(label, steps_list)
        root.addWidget(steps_list, 1)
        self.workflow_steps = steps_list

        # Launch button — opens the real Character Builder window.
        self.launch_button = QtWidgets.QPushButton("Open Character Builder…")
        self.launch_button.setToolTip(
            "Open the AccuRig-style Character Builder window for the active mode"
        )
        self.launch_button.clicked.connect(self._on_launch_clicked)
        root.addWidget(self.launch_button)

    # ── Slots ────────────────────────────────────────────────────────
    @QtCore.Slot()
    def _on_launch_clicked(self) -> None:
        # Emit first so the host window can intercept (e.g. to reuse an
        # already-open builder instance).
        self.launchRequested.emit()
        if self.receivers(self.launchRequested) > 1:
            # Host took over; nothing else to do.
            return
        # No listener — open a window owned by this panel.
        if self._builder_window is None:
            try:
                self._builder_window = QtCharacterBuilderWindow(self)
            except Exception as exc:                        # pragma: no cover
                log.exception("Failed to open Character Builder window")
                QtWidgets.QMessageBox.critical(
                    self, "Character Builder",
                    f"Could not open the Character Builder window:\n\n{exc}",
                )
                return
        self._builder_window.show()
        self._builder_window.raise_()
        self._builder_window.activateWindow()


def _split_failure_dialog_copy(result) -> "tuple[str, str] | None":
    """Map a Node Splitter hard-fail result to (dialog_title, dialog_body).

    P5-min (T2514): the two anatomical-split hard-fail paths (T2512 D-4 missing
    donor; palette overflow) must surface as actionable dialogs, not
    tracebacks or silent status-strip lines.  Pure function (no widgets) so the
    dialog copy is unit-testable headlessly.  Returns ``None`` for every other
    result (success and soft/info outcomes keep the existing status-strip UX).
    """
    if not isinstance(result, dict) or result.get("ok", False):
        return None
    code = str(result.get("code") or "")
    if code == "missing_donor":
        return (
            "Node Splitter — Base Skeleton Required",
            "Character Builder requires a base skeleton (weight donor) to "
            "split a skinned mesh into KOTOR-sized bone regions.\n\n"
            "Please select a KOTOR base skeleton (step 1, Inspector → "
            "\"KOTOR Base Skeleton (weight donor)\") before splitting.",
        )
    if code == "palette_overflow":
        detail = str(result.get("message") or "")
        return (
            "Node Splitter — Export Blocked (Palette Overflow)",
            "A split region still needs more than 16 bones, so this mesh "
            "cannot be exported as a KOTOR skin node.\n\n"
            f"{detail}\n\n"
            "Try re-partitioning (different donor / cleaner weight painting) "
            "or reducing the skinning complexity of the affected area.",
        )
    return None


class QtCharacterBuilderWindow(QtWidgets.QMainWindow):
    """AccuRig-style Character Builder window shell (M2 / T201).

    Layout (audit §4.1)::

        ┌─ TOP TOOLBAR ────────────────────────────────────────────────────┐
        │ [Mode: Headless | Head | Humanoid | Creature]  [K1 | K2]        │
        │ [Front][Back][L][R][T][B][Persp][Ortho]  [Sym][Snap][Validate]  │
        ├──────────────┬─────────────────────────────────────┬─────────────┤
        │ LEFT RAIL    │ CENTER VIEWPORT (QtViewportWidget)  │ RIGHT INSPECTOR
        │ (workflow)   │                                     │ (mode-aware)│
        ├──────────────┴─────────────────────────────────────┴─────────────┤
        │ BOTTOM STRIP: validation • scrubber • stats • log               │
        └──────────────────────────────────────────────────────────────────┘

    Wiring (the controller role):
      * Rail.stepSelected      → Inspector.set_step
      * ModeToolbar.modeChanged → Rail.set_mode + scene.set_mode(locked=True)
      * Scene mode changes      → push to Rail + Properties + mode toolbar
      * Window geometry, splitter sizes, last mode persisted via QSettings
        ("GhostRigger" / "CharacterBuilder") on close (T207).
    """

    # Re-emitted to outside listeners (e.g. qt_main_window status bar)
    # whenever the user picks a different CharacterMode.
    modeChanged = QtCore.Signal(object)
    # Navigation only: the foreign-rig product remains an independent window.
    customBuilderRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        CharacterScene = _import_model_data()
        self.scene = CharacterScene(game_version="K1")
        if _CHARACTER_MODE_AVAILABLE and CharacterMode is not None:
            try:
                self.scene.mode = CharacterMode.HEADLESS_BODY
                self.scene.mode_locked = False
            except Exception:                              # pragma: no cover
                log.debug("Could not seed initial CharacterMode", exc_info=True)
        self._scene_path = ""
        # ``_mode_actions`` maps CharacterMode → QAction so the toolbar
        # can be driven both by user clicks AND programmatic updates
        # (e.g. when scene.mode is restored from QSettings on startup).
        self._mode_actions: dict = {}
        # Prevents echo loops when the scene pushes a mode change back
        # to the toolbar.
        self._suppress_mode_signal = False
        # M5 / T503 — AcuRig instance shared between "Place Guides" and
        # "Generate Skeleton" so user-locked guide overrides survive
        # across the two clicks.  Lazily populated by the body-rig slot.
        self._acurig: Optional[Any] = None
        self._legacy_acurig_enabled = False
        self._body_guides: dict[str, Any] = {}
        self._body_guide_history: Optional[Any] = None
        # Durable workflow state lives in scene metadata; runtime AcuRig and
        # model objects stay out of the human-readable .ghostrig payload.
        self._rig_session: Optional[Any] = None
        self._restore_rig_session_from_scene()
        # M12 / T1202 — selected KOTOR skeleton template for imported
        # OBJ/FBX bodies.  Options are provided by the Qt-free picker
        # service and mirrored into the right inspector.
        self._skeleton_template_options: list[Any] = []
        self._skeleton_template_options_by_key: dict[str, Any] = {}
        self._installed_skeleton_template_rows_by_game: dict[str, list[dict[str, str]]] = {}
        self._selected_skeleton_template_key = ""
        self._selected_skeleton_template_model: Optional[Any] = None
        self._manual_fit_scale: float = 1.0
        self._manual_fit_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._manual_fit_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._resource_manager: Optional[Any] = None
        self._resource_manager_games: set[str] = set()
        self._preview_attachment_path: str = ""
        self._bas_preview_attachments: dict[str, tuple[Any, str]] = {}
        self._bas_preview_body: Any = None
        self._animation_engine: Optional[Any] = None
        self._animation_last_tick: Optional[float] = None
        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._tick_preview_animation)
        # M9 / T901 — live validation is intentionally debounced so
        # guide drags and slider-like controls do not spam the workflow
        # service while still refreshing the export banner promptly.
        self._live_validation_timer = QtCore.QTimer(self)
        self._live_validation_timer.setSingleShot(True)
        self._live_validation_timer.setInterval(200)
        self._live_validation_timer.timeout.connect(self._run_live_validation)
        self._last_validation_result: Optional[Any] = None

        self.setObjectName("QtCharacterBuilderWindow")
        self.setWindowTitle("Ghost-Studio - Character Builder")
        self.resize(1280, 800)
        apply_theme(self)

        self._build_toolbars()
        self._build_central()
        self._build_bottom_strip()
        self._build_menubar()
        self._connect_signals()
        from src.gui.qt_lib.controllers.head_builder_controller import (
            QtHeadBuilderController,
        )

        self.head_builder_controller = QtHeadBuilderController(self)
        self._restore_settings()
        self._sync_from_scene()
        self._update_title()
        self._refresh_skeleton_template_options()
        theme_manager = getattr(parent, "theme_manager", None)
        layout_manager = getattr(parent, "layout_manager", None)
        if theme_manager is not None:
            theme_manager.register_theme_aware_widget(self)
            theme = theme_manager.current_theme or theme_manager.get_theme()
            if getattr(theme, "is_native", lambda: False)():
                self.apply_native_theme()
            else:
                self.apply_ghost_theme(theme)
        if layout_manager is not None:
            self.apply_ghost_layout(layout_manager.current_layout or layout_manager.get_layout())

    def set_renderer_settings(self, settings: object) -> None:
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_renderer_settings"):
            viewport.set_renderer_settings(settings)

    def open_mode(self, mode: object) -> object:
        """Select one authoring mode without replacing the current scene.

        This is the public lifecycle entry used by the main shell.  Reopening
        the Head Builder therefore reuses the existing Character Builder
        window, viewport, controller state, and dirty scene.
        """
        if not _CHARACTER_MODE_AVAILABLE or CharacterMode is None:
            raise RuntimeError("Character modes are unavailable in this build")
        if isinstance(mode, CharacterMode):
            target = mode
        else:
            key = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
            facial_mode = key in {
                "facial_performance_head",
                "advanced_facial_head",
                "custom_facial_head",
            }
            if facial_mode:
                self._set_head_facial_output_mode("custom_patch_curves")
            aliases = {
                "native_kotor_head": "HEAD",
                "facial_performance_head": "HEAD",
                "advanced_facial_head": "HEAD",
                "custom_facial_head": "HEAD",
                "head_builder": "HEAD",
                "custom_head": "HEAD",
                "modular_head": "HEAD",
                "head": "HEAD",
                "native_kotor_character": "HEADLESS_BODY",
                "headless_body": "HEADLESS_BODY",
                "body": "HEADLESS_BODY",
                "humanoid": "HUMANOID",
                "creature": "CREATURE",
                "supermodel": "SUPERMODEL",
            }
            enum_name = aliases.get(key, key.upper())
            try:
                target = CharacterMode[enum_name]
            except KeyError as exc:
                raise ValueError(f"Unknown Character Builder mode: {mode!r}") from exc
            if target == CharacterMode.HEAD and not facial_mode:
                setter = getattr(self, "_set_head_facial_output_mode", None)
                if callable(setter):
                    setter("vanilla_lip")
        self._apply_mode(target, locked=True, source="public_entry")
        return target

    def _set_head_facial_output_mode(self, mode: str) -> None:
        """Configure vanilla or patch-required output for the Head Builder."""

        normalized = (
            "custom_patch_curves"
            if str(mode).strip().lower() == "custom_patch_curves"
            else "vanilla_lip"
        )
        self._head_facial_output_mode = normalized
        workspace = getattr(self, "head_builder_properties", None)
        setter = getattr(workspace, "set_facial_performance_mode", None)
        if callable(setter):
            setter(normalized == "custom_patch_curves")

    def set_legacy_acurig_enabled(self, enabled: bool) -> None:
        """Opt into the experimental AcuRig body-generation path.

        The normal Character Builder export workflow uses the selected native
        KOTOR template through ``apply_template_rig``.  AcuRig remains available
        only as an explicit legacy/experimental diagnostic path.
        """
        self._legacy_acurig_enabled = bool(enabled)

    def _require_legacy_acurig_enabled(self, action_label: str) -> bool:
        """Return True only when the legacy AcuRig path has been opted into."""
        if bool(getattr(self, "_legacy_acurig_enabled", False)):
            return True
        message = (
            f"{action_label} uses the legacy/experimental AcuRig path and is "
            "disabled by default. Use Build KOTOR Skeleton to bind the selected "
            "native KOTOR template for game export."
        )
        if hasattr(self.inspector, "set_body_rig_status"):
            try:
                self.inspector.set_body_rig_status(message, kind="warning")
            except Exception:                              # pragma: no cover
                log.exception("inspector.set_body_rig_status failed")
        if hasattr(self, "bottom_strip"):
            self.bottom_strip.set_validation(
                "warning",
                "LEGACY_ACURIG_DISABLED",
                issues=[message],
            )
        self.statusBar().showMessage(message, 7000)
        return False

    def apply_ghost_theme(self, theme) -> None:
        if getattr(theme, "is_native", lambda: False)():
            self.apply_native_theme()
            return
        update_legacy_palette(theme)
        self.setStyleSheet(self._character_builder_theme_stylesheet(theme))
        themed: set[int] = set()
        for widget in self.findChildren(QtWidgets.QWidget):
            widget_id = id(widget)
            if widget_id in themed:
                continue
            themed.add(widget_id)
            hook = getattr(widget, "apply_ghost_theme", None)
            if callable(hook):
                hook(theme)

    def apply_native_theme(self) -> None:
        self.setStyleSheet("")
        for widget in self.findChildren(QtWidgets.QWidget):
            widget.setStyleSheet("")
        for widget in self.findChildren(QtWidgets.QWidget):
            hook = getattr(widget, "apply_native_theme", None)
            if callable(hook):
                hook()

    @staticmethod
    def _character_builder_theme_stylesheet(theme) -> str:
        """Return window-scoped styling for palette-only Character Builder themes."""

        c = theme.color
        m = theme.metric
        radius = m("border.radius", 3)
        input_height = m("input.height", max(18, m("button.height", 28) - 8))
        button_padding_x = m("button.paddingX", m("button.paddingH", 10))
        button_padding_y = m("button.paddingY", m("button.paddingV", 5))
        return f"""
        QMainWindow#QtCharacterBuilderWindow,
        QMainWindow#QtCharacterBuilderWindow QWidget {{
            background: {c('window.background')};
            color: {c('window.text', c('text.primary'))};
        }}
        QMainWindow#QtCharacterBuilderWindow QMenuBar,
        QMainWindow#QtCharacterBuilderWindow QMenu,
        QMainWindow#QtCharacterBuilderWindow QToolBar,
        QMainWindow#QtCharacterBuilderWindow QStatusBar {{
            background: {c('toolbar.background')};
            color: {c('text.primary')};
            border: 0;
        }}
        QMainWindow#QtCharacterBuilderWindow QToolBar#CharacterBuilderToolbar,
        QMainWindow#QtCharacterBuilderWindow QToolBar#CharacterBuilderToolbarContents,
        QMainWindow#QtCharacterBuilderWindow QScrollArea#CharacterBuilderToolbarScroll {{
            background: {c('toolbar.background')};
            border: 0;
        }}
        QMainWindow#QtCharacterBuilderWindow QSplitter::handle,
        QMainWindow#QtCharacterBuilderWindow QMainWindow::separator {{
            background: {c('panel.border')};
        }}
        QMainWindow#QtCharacterBuilderWindow QGroupBox {{
            color: {c('text.primary')};
            border: 1px solid {c('groupbox.border')};
            border-radius: {radius}px;
            margin-top: {m('groupbox.margin', 8)}px;
            padding-top: {m('groupbox.margin', 8)}px;
        }}
        QMainWindow#QtCharacterBuilderWindow QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            color: {c('groupbox.title')};
        }}
        QMainWindow#QtCharacterBuilderWindow QListWidget,
        QMainWindow#QtCharacterBuilderWindow QTextEdit,
        QMainWindow#QtCharacterBuilderWindow QPlainTextEdit,
        QMainWindow#QtCharacterBuilderWindow QTreeWidget,
        QMainWindow#QtCharacterBuilderWindow QTableWidget,
        QMainWindow#QtCharacterBuilderWindow QTableView,
        QMainWindow#QtCharacterBuilderWindow QTabWidget::pane {{
            background: {c('table.background', c('viewport.background'))};
            color: {c('table.text', c('text.primary'))};
            border: 1px solid {c('panel.border')};
        }}
        QMainWindow#QtCharacterBuilderWindow QHeaderView::section {{
            background: {c('table.headerBackground')};
            color: {c('table.headerText')};
            border: 1px solid {c('table.grid', c('panel.border'))};
            padding: 4px;
        }}
        QMainWindow#QtCharacterBuilderWindow QLineEdit,
        QMainWindow#QtCharacterBuilderWindow QComboBox,
        QMainWindow#QtCharacterBuilderWindow QDoubleSpinBox,
        QMainWindow#QtCharacterBuilderWindow QSpinBox {{
            background: {c('input.background')};
            color: {c('input.text')};
            border: 1px solid {c('input.border')};
            border-radius: {radius}px;
            padding: 4px 6px;
            min-height: {input_height}px;
        }}
        QMainWindow#QtCharacterBuilderWindow QLineEdit:focus,
        QMainWindow#QtCharacterBuilderWindow QComboBox:focus,
        QMainWindow#QtCharacterBuilderWindow QDoubleSpinBox:focus,
        QMainWindow#QtCharacterBuilderWindow QSpinBox:focus {{
            border-color: {c('input.focusBorder')};
        }}
        QMainWindow#QtCharacterBuilderWindow QComboBox QAbstractItemView {{
            background: {c('panel.backgroundAlt', c('panel.altBackground'))};
            color: {c('text.primary')};
            selection-background-color: {c('selection.background')};
            selection-color: {c('selection.text')};
        }}
        QMainWindow#QtCharacterBuilderWindow QPushButton,
        QMainWindow#QtCharacterBuilderWindow QToolButton {{
            background: {c('button.background')};
            color: {c('button.text')};
            border: 1px solid {c('panel.border')};
            border-radius: {radius}px;
            padding: {button_padding_y}px {button_padding_x}px;
            min-height: {m('button.height', 28)}px;
            min-width: {m('button.minWidth', 76)}px;
        }}
        QMainWindow#QtCharacterBuilderWindow QPushButton:hover,
        QMainWindow#QtCharacterBuilderWindow QToolButton:hover {{
            background: {c('button.hover')};
            color: {c('accent.primary')};
        }}
        QMainWindow#QtCharacterBuilderWindow QPushButton:checked,
        QMainWindow#QtCharacterBuilderWindow QToolButton:checked {{
            background: {c('button.checked')};
            color: {c('button.checkedText', c('button.text'))};
            border-color: {c('accent.primary')};
        }}
        QMainWindow#QtCharacterBuilderWindow QPushButton:disabled,
        QMainWindow#QtCharacterBuilderWindow QToolButton:disabled {{
            background: {c('button.disabledBackground')};
            color: {c('button.disabledText', c('text.disabled'))};
            border-color: {c('panel.border')};
        }}
        QMainWindow#QtCharacterBuilderWindow QPushButton[accent="true"],
        QMainWindow#QtCharacterBuilderWindow QToolButton[accent="true"] {{
            background: {c('accent.primary')};
            color: {c('button.accentText')};
            border-color: {c('accent.primary')};
        }}
        QMainWindow#QtCharacterBuilderWindow QTabBar::tab {{
            background: {c('tab.background')};
            color: {c('tab.text')};
            border: 1px solid {c('panel.border')};
            padding: {m('tab.paddingY', m('tab.padding', 4))}px {m('tab.paddingX', 12)}px;
        }}
        QMainWindow#QtCharacterBuilderWindow QTabBar::tab:selected {{
            background: {c('tab.selectedBackground')};
            color: {c('tab.selectedText')};
            border-color: {c('accent.primary')};
        }}
        QMainWindow#QtCharacterBuilderWindow QScrollBar:vertical,
        QMainWindow#QtCharacterBuilderWindow QScrollBar:horizontal {{
            background: {c('scrollbar.background')};
            border: 0;
        }}
        QMainWindow#QtCharacterBuilderWindow QScrollBar::handle:vertical,
        QMainWindow#QtCharacterBuilderWindow QScrollBar::handle:horizontal {{
            background: {c('scrollbar.handle')};
            border-radius: {radius}px;
            min-height: 24px;
            min-width: 24px;
        }}
        """

    def apply_ghost_layout(self, layout) -> None:
        toolbar = layout.toolbar("main")
        self._toolbar.setFixedHeight(toolbar.height)
        self._toolbar.setIconSize(QtCore.QSize(toolbar.icon_size, toolbar.icon_size))
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly if toolbar.button_mode == "iconOnly" else QtCore.Qt.ToolButtonTextOnly if toolbar.button_mode == "textOnly" else QtCore.Qt.ToolButtonTextBesideIcon)
        self._toolbar_scroll.setFixedHeight(max(toolbar.height + 10, toolbar.height))
        self._splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
        self.rail.setMinimumWidth(max(220, layout.panel("library").min_width // 2))
        self.inspector.setMinimumWidth(max(360, layout.panel("properties").min_width))
        self._splitter.setSizes([
            max(240, layout.panel("library").preferred_width // 2),
            max(layout.viewport.preferred_width, layout.viewport.min_width),
            max(380, layout.panel("properties").preferred_width),
        ])
        for widget in [*self.findChildren(QtWidgets.QComboBox), *self.findChildren(QtWidgets.QSpinBox), *self.findChildren(QtWidgets.QDoubleSpinBox)]:
            widget.setMinimumHeight(layout.spacing_value("inputHeight", 24))
        for widget in (
            getattr(self, "rail", None),
            getattr(self, "inspector", None),
            getattr(self, "properties", None),
            getattr(self, "bottom_strip", None),
            getattr(self, "head_builder_assets", None),
            getattr(self, "head_builder_properties", None),
            getattr(self, "head_builder_evidence", None),
        ):
            hook = getattr(widget, "apply_ghost_layout", None)
            if callable(hook):
                hook(layout)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_toolbars(self) -> None:
        """Top toolbar — mode switcher (T205) + game + camera presets."""
        toolbar_shell = QtWidgets.QToolBar("Character Builder Toolbar", self)
        toolbar_shell.setObjectName("CharacterBuilderToolbar")
        toolbar_shell.setMovable(False)
        toolbar_shell.setFloatable(False)
        toolbar_shell.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)

        toolbar = QtWidgets.QToolBar("Character Builder Toolbar Contents", self)
        toolbar.setObjectName("CharacterBuilderToolbarContents")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        toolbar.setFixedHeight(32)
        toolbar_scroll = make_horizontal_overflow_area(
            toolbar,
            "CharacterBuilderToolbarScroll",
            height=48,
            parent=toolbar_shell,
        )
        toolbar_shell.addWidget(toolbar_scroll)
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar_shell)
        self._toolbar_shell = toolbar_shell
        self._toolbar_scroll = toolbar_scroll
        self._toolbar = toolbar

        brand = QtWidgets.QLabel("  GHOSTRIGGER AUTORIG  ")
        brand.setObjectName("CharacterBuilderToolbarBrand")
        brand.setStyleSheet(
            f"color:{C.get('accent', '#00FF7A')}; "
            "font-weight:800; font-size:10pt; letter-spacing:0px;"
        )
        toolbar.addWidget(brand)

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Character type: "))
        self._character_type_combo = QtWidgets.QComboBox()
        self._character_type_combo.addItem("Native KOTOR Character", "native_kotor_character")
        self._character_type_combo.addItem("Custom Rigged Character", "custom_rigged_character")
        self._character_type_combo.setToolTip(
            "Switch to the independent custom-rig workflow without changing this native KOTOR character."
        )
        self._character_type_combo.currentIndexChanged.connect(self._on_character_type_changed)
        toolbar.addWidget(self._character_type_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Mode: "))

        # Exclusive QToolButtons wired to authoring CharacterMode values (T205).
        self._mode_action_group = QtGui.QActionGroup(self)
        self._mode_action_group.setExclusive(True)

        mode_specs = [
            ("HEADLESS_BODY", "Headless"),
            ("HEAD",          "Head"),
            ("HUMANOID",      "Humanoid"),
            ("CREATURE",      "Creature"),
        ]
        for mode_name, label in mode_specs:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            mode_obj = None
            if _CHARACTER_MODE_AVAILABLE and CharacterMode is not None:
                try:
                    mode_obj = CharacterMode[mode_name]
                except KeyError:                            # pragma: no cover
                    mode_obj = None
            if mode_obj is not None:
                self._mode_actions[mode_obj] = action
                action.setData(mode_obj)
            else:
                # Disable the button when the enum is unavailable, but
                # keep it visible so the layout stays stable.
                action.setEnabled(False)
            action.triggered.connect(self._on_mode_action_triggered)
            self._mode_action_group.addAction(action)
            toolbar.addAction(action)

        toolbar.addSeparator()

        # Game version selector.
        toolbar.addWidget(QtWidgets.QLabel(" Game: "))
        self._game_combo = QtWidgets.QComboBox()
        self._game_combo.addItems(["K1", "K2"])
        self._game_combo.setCurrentText(getattr(self.scene, "game_version", "K1"))
        self._game_combo.currentTextChanged.connect(self._on_game_changed)
        toolbar.addWidget(self._game_combo)

        toolbar.addSeparator()

        # Camera preset buttons — placeholders that emit through the
        # viewport widget when available.  Kept here so the AccuRig
        # toolbar shape is established for M4 work.
        for preset, tooltip in [
            ("Front",  "Camera: front"),
            ("Back",   "Camera: back"),
            ("L",      "Camera: left"),
            ("R",      "Camera: right"),
            ("T",      "Camera: top"),
            ("B",      "Camera: bottom"),
            ("Persp",  "Camera: perspective"),
            ("Ortho",  "Camera: orthographic"),
        ]:
            act = QtGui.QAction(preset, self)
            act.setToolTip(tooltip)
            act.triggered.connect(lambda _checked=False, p=preset: self._on_camera_preset(p))
            toolbar.addAction(act)

        toolbar.addSeparator()

        # Tool toggles.
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Rig: "))

        self._rig_transform_action_group = QtGui.QActionGroup(self)
        self._rig_transform_action_group.setExclusive(True)
        self._rig_transform_actions: dict[str, QtGui.QAction] = {}
        for key, label, tooltip in (
            ("select", "Select", "Select joints, bones, and imported mesh handles."),
            ("translate", "Move", "Move selected rig guides or fitted mesh handles."),
            ("rotate", "Rotate", "Rotate the selected rig or fit handle."),
            ("transform", "Transform", "Use the universal transform/scale handle."),
        ):
            action = QtGui.QAction(label, self)
            action.setObjectName(f"CharacterBuilderRigToolAction_{key}")
            action.setCheckable(True)
            action.setToolTip(tooltip)
            action.setData(key)
            action.triggered.connect(
                lambda _checked=False, tool_key=key: self._run_character_builder_transform_action(tool_key)
            )
            self._rig_transform_action_group.addAction(action)
            self._rig_transform_actions[key] = action
            toolbar.addAction(action)
        self._rig_transform_actions["select"].setChecked(True)

        toolbar.addSeparator()

        self._symmetry_action = QtGui.QAction("Symmetry", self)
        self._symmetry_action.setCheckable(True)
        self._symmetry_action.setChecked(True)
        self._symmetry_action.setToolTip("Mirror placement across X")
        self._symmetry_action.toggled.connect(self._on_joint_symmetry_toggled)
        toolbar.addAction(self._symmetry_action)

        self._bones_action = QtGui.QAction("Bones", self)
        self._bones_action.setObjectName("CharacterBuilderBonesToggleAction")
        self._bones_action.setCheckable(True)
        self._bones_action.setToolTip("Show or hide the selected base skeleton and generated bones")
        self._bones_action.toggled.connect(self._on_bones_toggled)
        toolbar.addAction(self._bones_action)

        self._weights_action = QtGui.QAction("Weights", self)
        self._weights_action.setObjectName("CharacterBuilderWeightsToggleAction")
        self._weights_action.setCheckable(True)
        self._weights_action.setToolTip("Show or hide selected-bone weight heat-map preview")
        self._weights_action.toggled.connect(self._on_weights_toggled)
        toolbar.addAction(self._weights_action)

        self._joints_action = QtGui.QAction("Joints", self)
        self._joints_action.setObjectName("CharacterBuilderJointDotsToggleAction")
        self._joints_action.setCheckable(True)
        self._joints_action.setChecked(True)
        self._joints_action.setToolTip("Show or hide rig guide/joint handles")
        self._joints_action.toggled.connect(self._on_joint_dots_toggled)
        toolbar.addAction(self._joints_action)

        self._snap_action = QtGui.QAction("Snap", self)
        self._snap_action.setCheckable(True)
        self._snap_action.setToolTip("Snap pins to mesh surface")
        toolbar.addAction(self._snap_action)

        self._undo_guide_action = QtGui.QAction("Undo Guide", self)
        self._undo_guide_action.setShortcut(QtGui.QKeySequence.Undo)
        self._undo_guide_action.setEnabled(False)
        self._undo_guide_action.setToolTip("Undo the last body guide drag")
        self._undo_guide_action.triggered.connect(self._on_undo_body_guide_requested)
        toolbar.addAction(self._undo_guide_action)

        self._redo_guide_action = QtGui.QAction("Redo Guide", self)
        self._redo_guide_action.setShortcut(QtGui.QKeySequence.Redo)
        self._redo_guide_action.setEnabled(False)
        self._redo_guide_action.setToolTip("Redo the last undone body guide drag")
        self._redo_guide_action.triggered.connect(self._on_redo_body_guide_requested)
        toolbar.addAction(self._redo_guide_action)

        validate_action = QtGui.QAction("Validate", self)
        validate_action.setToolTip("Run validation now (results appear in bottom banner)")
        validate_action.triggered.connect(self._on_validate_requested)
        toolbar.addAction(validate_action)

        self._head_toolbar_separator = toolbar.addSeparator()
        self._head_toolbar_separator.setVisible(False)
        self._head_toolbar_actions: list[QtGui.QAction] = []
        for key, label, tooltip, shortcut in (
            ("new", "New", "New Custom Head project", ""),
            ("open", "Open", "Open .ghosthead.json project", ""),
            ("save", "Save", "Save Custom Head project", ""),
            ("undo", "Undo", "Undo the last project command", ""),
            ("redo", "Redo", "Redo the last undone command", ""),
            ("import", "Import", "Import and audit OBJ or FBX head art", ""),
            ("validate", "Validate", "Run structural and binary preflight", ""),
            ("export", "Export", "Export and reload verified MDL/MDX", ""),
            ("prepare", "Prepare Test", "Prepare a read-only install preview", ""),
            ("restore", "Restore", "Restore pre-test game files", ""),
            ("help", "Help", "Show the Custom Head workflow guide", ""),
        ):
            action = QtGui.QAction(label, self)
            action.setObjectName(f"HeadBuilderToolbarAction_{key}")
            action.setToolTip(tooltip)
            action.setData(key)
            if shortcut:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(
                lambda _checked=False, action_key=key: (
                    self._dispatch_head_toolbar_action(action_key)
                )
            )
            action.setVisible(False)
            toolbar.addAction(action)
            self._head_toolbar_actions.append(action)
            if key == "undo":
                self._head_undo_action = action
                action.setEnabled(False)
            elif key == "redo":
                self._head_redo_action = action
                action.setEnabled(False)

    def _build_central(self) -> None:
        """Central widget — horizontal splitter: rail / viewport / inspector."""
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setObjectName("CharacterBuilderSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)

        # Left workflow and project provenance.
        left_holder = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left_holder)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.rail = QtWorkflowRail(self)
        self.rail.setMinimumWidth(220)
        left_layout.addWidget(self.rail, 1)
        self.head_builder_assets = QtHeadBuilderAssetTree(self)
        self.head_builder_assets.setVisible(False)
        left_layout.addWidget(self.head_builder_assets, 1)
        splitter.addWidget(left_holder)

        # Centre — viewport stack so future modes can swap previews
        # (e.g. dual-orthographic for Body / Head, single-perspective
        # for Creature).
        viewport_holder = QtWidgets.QWidget()
        viewport_layout = QtWidgets.QVBoxLayout(viewport_holder)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)

        self._viewport_stack = QtWidgets.QStackedWidget()
        if _VIEWPORT_AVAILABLE and QtCharacterBuilderViewportWidget is not None:
            try:
                self.viewport = QtCharacterBuilderViewportWidget(self)
            except Exception as exc:                       # pragma: no cover
                log.warning("QtCharacterBuilderWindow: viewport init failed: %s", exc)
                self.viewport = self._make_viewport_placeholder(str(exc))
        else:
            err = locals().get("_VIEWPORT_IMPORT_ERROR", "viewport unavailable")
            self.viewport = self._make_viewport_placeholder(err)
        self._viewport_stack.addWidget(self.viewport)
        viewport_layout.addWidget(self._viewport_stack, 1)
        viewport_holder.setMinimumWidth(400)
        splitter.addWidget(viewport_holder)

        # Right inspector (T203) — split into two stacked sub-panels:
        # the contextual step inspector on top and a properties panel
        # below (reuses the existing M1/T105 widget for the
        # CharacterMode badge + model stats).
        right_holder = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_holder)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_split.setChildrenCollapsible(False)
        self.inspector = QtInspectorPanel(self)
        right_split.addWidget(self.inspector)
        # Character Studio edits character/node properties only. Module room,
        # wall, NULL-mesh, and WOK browsing belongs to Map/Module Studio.
        self.properties = QtPropertiesPanel(self, module_browser_enabled=False)
        right_split.addWidget(self.properties)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)
        self.head_builder_properties = QtHeadBuilderProperties(self)
        self._right_stack = QtWidgets.QStackedWidget(self)
        self.character_authoring_tabs = QtWidgets.QTabWidget(self)
        self.character_authoring_tabs.setObjectName("characterBuilderAuthoringTabs")
        self.character_authoring_tabs.addTab(right_split, "Rig & Deformation")
        self.secondary_motion_editor = QtSecondaryMotionEditor(
            self,
            target_kind="native_character",
            target_game=str(getattr(self.scene, "game_version", "K1") or "K1"),
        )
        metadata = getattr(self.scene, "metadata", None)
        if isinstance(metadata, dict) and isinstance(metadata.get("secondary_motion"), dict):
            self.secondary_motion_editor.set_profile(metadata["secondary_motion"])
        self.secondary_motion_editor.profileChanged.connect(self._on_secondary_motion_profile_changed)
        self.character_authoring_tabs.addTab(self.secondary_motion_editor, "Secondary Motion")
        self._right_stack.addWidget(self.character_authoring_tabs)
        self._right_stack.addWidget(self.head_builder_properties)
        right_layout.addWidget(self._right_stack, 1)

        right_holder.setMinimumWidth(260)
        splitter.addWidget(right_holder)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 700, 340])

        self._splitter = splitter
        self.setCentralWidget(splitter)

    def _make_viewport_placeholder(self, message: str) -> QtWidgets.QWidget:
        """Build a labelled placeholder used when the real viewport fails."""
        placeholder = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(placeholder)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        title = QtWidgets.QLabel("Viewport unavailable")
        title.setStyleSheet(
            f"color:{C.get('gold', '#FFD700')}; font-weight:bold; font-size:11pt;"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        detail = QtWidgets.QLabel(message)
        detail.setStyleSheet(f"color:{C.get('text2', '#888')}; font-size:9pt;")
        detail.setAlignment(QtCore.Qt.AlignCenter)
        detail.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(detail)
        placeholder.setStyleSheet(f"background:{C.get('bg2', '#1a1a1a')};")
        return placeholder

    def _build_bottom_strip(self) -> None:
        """Bottom strip (T204) hosted as a fixed dock at the bottom edge."""
        self.bottom_strip = QtBottomStrip(self)
        self.head_builder_evidence = QtHeadBuilderEvidencePanel(self)
        self._bottom_stack = QtWidgets.QStackedWidget(self)
        self._bottom_stack.addWidget(self.bottom_strip)
        self._bottom_stack.addWidget(self.head_builder_evidence)
        dock = QtWidgets.QDockWidget("Status", self)
        dock.setObjectName("CharacterBuilderBottomDock")
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QtWidgets.QWidget())     # hide title bar
        dock.setWidget(
            make_scrollable_panel(
                self._bottom_stack,
                "CharacterBuilderBottomDockScroll",
                dock,
            )
        )
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self._bottom_dock = dock

    def _build_menubar(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_action = QtGui.QAction("New Scene", self)
        self._file_new_action = new_action
        new_action.setShortcut(QtGui.QKeySequence.New)
        new_action.triggered.connect(self._new_active_document)

        open_action = QtGui.QAction("Open Scene...", self)
        self._file_open_action = open_action
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self._open_active_document)

        save_action = QtGui.QAction("Save Scene", self)
        self._file_save_action = save_action
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self._save_active_document)

        save_as_action = QtGui.QAction("Save Scene As...", self)
        self._file_save_as_action = save_as_action
        save_as_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(
            lambda: self._save_active_document(save_as=True)
        )

        close_action = QtGui.QAction("Close", self)
        close_action.triggered.connect(self.close)

        file_menu.addAction(new_action)
        file_menu.addSeparator()
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(close_action)

    # ── Signal plumbing ──────────────────────────────────────────────────

    @staticmethod
    def _is_head_builder_mode(mode: object) -> bool:
        value = (
            getattr(mode, "value", None)
            or getattr(mode, "name", "")
            or str(mode or "")
        )
        return str(value).strip().casefold() == "head"

    def _update_head_builder_workspace(self, mode: object) -> None:
        active = self._is_head_builder_mode(mode)
        self.head_builder_assets.setVisible(active)
        self._right_stack.setCurrentIndex(1 if active else 0)
        self._bottom_stack.setCurrentIndex(1 if active else 0)
        self._head_toolbar_separator.setVisible(active)
        for action in self._head_toolbar_actions:
            action.setVisible(active)
        self._file_new_action.setText(
            "New Custom Head Project" if active else "New Scene"
        )
        self._file_open_action.setText(
            "Open Custom Head Project…" if active else "Open Scene..."
        )
        self._file_save_action.setText(
            "Save Custom Head Project" if active else "Save Scene"
        )
        self._file_save_as_action.setText(
            "Save Custom Head Project As…" if active else "Save Scene As..."
        )
        if active and hasattr(self, "head_builder_controller"):
            self.head_builder_controller.refresh()

    def _new_active_document(self) -> None:
        if self._is_head_builder_mode(getattr(self.scene, "mode", None)):
            self._dispatch_head_toolbar_action("new")
        else:
            self._new_scene()

    def _open_active_document(self) -> None:
        if self._is_head_builder_mode(getattr(self.scene, "mode", None)):
            self._dispatch_head_toolbar_action("open")
        else:
            self._open_scene()

    def _save_active_document(self, *, save_as: bool = False) -> bool:
        if self._is_head_builder_mode(getattr(self.scene, "mode", None)):
            controller = getattr(self, "head_builder_controller", None)
            if controller is None:
                return False
            if save_as:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save Custom Head Project As",
                    "",
                    "Ghost Head Project (*.ghosthead.json)",
                )
                return bool(path) and controller.save(path)
            return controller.save()
        return self._save_scene(save_as=save_as)

    @QtCore.Slot(int)
    def _on_workflow_step_selected(self, step_number: int) -> None:
        mode = getattr(self.scene, "mode", None)
        if self._is_head_builder_mode(mode) and hasattr(
            self,
            "head_builder_controller",
        ):
            self.head_builder_controller.set_step(step_number)
        else:
            self.inspector.set_step(step_number)

    def _dispatch_head_toolbar_action(self, action_key: str) -> None:
        controller = getattr(self, "head_builder_controller", None)
        if controller is None:
            return
        key = str(action_key or "")
        if key == "new":
            controller.execute_action(
                "new_project",
                self.head_builder_properties.project_payload(),
            )
        elif key == "open":
            self.head_builder_properties._choose_open_project()
        elif key == "save":
            controller.save()
        elif key == "undo":
            controller.undo()
        elif key == "redo":
            controller.redo()
        elif key == "import":
            controller.execute_action(
                "import_art",
                self.head_builder_properties.import_payload(),
            )
        elif key == "validate":
            controller.execute_action("run_preflight", {})
        elif key == "export":
            controller.execute_action("export_binary", {})
        elif key == "prepare":
            controller.execute_action("prepare_install", {})
        elif key == "restore":
            controller.execute_action("restore_install", {})
        elif key == "help":
            QtWidgets.QMessageBox.information(
                self,
                "Custom KOTOR Head Builder",
                "Follow the numbered workflow from Project + Game through Safe "
                "Retail Test. Each completed step records evidence. The final "
                "retail pass is only accepted after you explicitly confirm the "
                "observer checklist and attach evidence.",
            )

    def _connect_signals(self) -> None:
        self.rail.stepSelected.connect(self._on_workflow_step_selected)
        self.inspector.exportRequested.connect(self._on_export_requested)
        self.inspector.loadRequested.connect(self._on_load_model_requested)
        if hasattr(self.inspector, "fitAdjustmentChanged"):
            self.inspector.fitAdjustmentChanged.connect(
                self._on_fit_adjustment_changed)
        if hasattr(self.inspector, "fitAdjustmentResetRequested"):
            self.inspector.fitAdjustmentResetRequested.connect(
                self._on_fit_adjustment_reset_requested)
        if hasattr(self.inspector, "refitToSelectedBaseRequested"):
            self.inspector.refitToSelectedBaseRequested.connect(
                self._on_refit_to_selected_base_requested)
        self.inspector.validateRequested.connect(self._on_validate_requested)
        self.inspector.checkModelRequested.connect(self._on_check_model_requested)
        self.bottom_strip.bannerClicked.connect(self._on_validation_banner_clicked)
        # M4 HUD QoL: wire the inspector's overlay controls to the
        # viewport instead of leaving them as passive surface widgets.
        self.inspector.symmetryToggled.connect(self._on_joint_symmetry_toggled)
        self.inspector.jointOpacityChanged.connect(self._on_joint_opacity_changed)
        self.inspector.jointSizeChanged.connect(self._on_joint_size_changed)
        # M5 / T503 — body-rig action buttons.
        if hasattr(self.inspector, "placeGuidesRequested"):
            self.inspector.placeGuidesRequested.connect(
                self._on_place_body_guides_requested)
        if hasattr(self.inspector, "generateSkeletonRequested"):
            self.inspector.generateSkeletonRequested.connect(
                self._on_generate_skeleton_requested)
        # M12 / T1202 — skeleton template picker + apply flow.
        if hasattr(self.inspector, "skeletonTemplateSelected"):
            self.inspector.skeletonTemplateSelected.connect(
                self._on_skeleton_template_selected)
        if hasattr(self.inspector, "browseSkeletonTemplateRequested"):
            self.inspector.browseSkeletonTemplateRequested.connect(
                self._on_browse_skeleton_template_requested)
        if hasattr(self.inspector, "applySkeletonTemplateRequested"):
            self.inspector.applySkeletonTemplateRequested.connect(
                self._on_apply_skeleton_template_requested)
        if hasattr(self.inspector, "splitMeshNodesRequested"):
            self.inspector.splitMeshNodesRequested.connect(
                self._on_split_mesh_nodes_requested)
        # M5 / T504 — hand-rig action buttons.
        if hasattr(self.inspector, "placeHandGuidesRequested"):
            self.inspector.placeHandGuidesRequested.connect(
                self._on_place_hand_guides_requested)
        if hasattr(self.inspector, "handMaskChanged"):
            self.inspector.handMaskChanged.connect(
                self._on_hand_mask_changed)
        # M5 / T505 — check-actor preview animations.
        if hasattr(self.inspector, "playPreviewAnimationRequested"):
            self.inspector.playPreviewAnimationRequested.connect(
                self._on_play_preview_animation_requested)
        if hasattr(self.inspector, "stopPreviewAnimationRequested"):
            self.inspector.stopPreviewAnimationRequested.connect(
                self._on_stop_preview_animation_requested)
        if hasattr(self.inspector, "refreshPreviewAnimationsRequested"):
            self.inspector.refreshPreviewAnimationsRequested.connect(
                self._on_refresh_preview_animations_requested)
        if hasattr(self.inspector, "browsePreviewAttachmentRequested"):
            self.inspector.browsePreviewAttachmentRequested.connect(
                self._on_browse_preview_attachment_requested)
        if hasattr(self.inspector, "attachPreviewAttachmentRequested"):
            self.inspector.attachPreviewAttachmentRequested.connect(
                self._on_attach_preview_attachment_requested)
        bas_panel = getattr(self.inspector, "body_attachment_panel", None)
        if bas_panel is not None:
            bas_panel.attachRequested.connect(self._on_bas_panel_attach_requested)
            bas_panel.clearRequested.connect(self._on_bas_panel_clear_requested)
            bas_panel.characterPresetRequested.connect(self._on_bas_character_preset_requested)
            bas_panel.slotSelected.connect(self._ensure_cb_bas_attachment_catalog)
            bas_panel.catalogRefreshRequested.connect(self._ensure_cb_bas_attachment_catalog)
            QtCore.QTimer.singleShot(0, self._ensure_cb_bas_attachment_catalog)
        # M12 / T1204 — mode-aware motion assignment replaces the
        # placeholder Add Motions action.
        if hasattr(self.inspector, "assignMotionsRequested"):
            self.inspector.assignMotionsRequested.connect(
                self._on_assign_motions_requested)
        if hasattr(self.inspector, "romTestRequested"):
            self.inspector.romTestRequested.connect(
                self._on_run_rom_test_requested)
        # M6 / T602 — Head Facial Palette.
        if hasattr(self.inspector, "headFacialBoneSelected"):
            self.inspector.headFacialBoneSelected.connect(
                self._on_head_facial_bone_selected)
        if hasattr(self.inspector, "rigHeadRequested"):
            self.inspector.rigHeadRequested.connect(
                self._on_rig_head_requested)
        if hasattr(self.inspector, "rigFaceRequested"):
            self.inspector.rigFaceRequested.connect(
                self._on_rig_face_requested)
        # M6 / T603 — Viseme Test Panel.
        if hasattr(self.inspector, "applyVisemeRequested"):
            self.inspector.applyVisemeRequested.connect(
                self._on_apply_viseme_requested)
        # M6 / T604 — Phoneme Calibration Panel.
        if hasattr(self.inspector, "calibratePhonemeRequested"):
            self.inspector.calibratePhonemeRequested.connect(
                self._on_calibrate_phoneme_requested)
        # M6 / T605 — Head-mode camera preset request.
        if hasattr(self.inspector, "headCameraPresetRequested"):
            self.inspector.headCameraPresetRequested.connect(
                self._on_head_camera_preset_requested)
        # M12 / T1203 — viewport joint-dot drags become AcuRig guide
        # overrides, so the next Generate Skeleton uses the edited pins.
        if hasattr(self.viewport, "nodeMoved"):
            self.viewport.nodeMoved.connect(self._on_viewport_node_moved)
        if hasattr(self.viewport, "rigTransformMarkingMenuRequested"):
            self.viewport.rigTransformMarkingMenuRequested.connect(
                self._open_character_builder_transform_marking_menu)
        if hasattr(self.viewport, "rigToolsMarkingMenuRequested"):
            self.viewport.rigToolsMarkingMenuRequested.connect(
                self._open_character_builder_rig_marking_menu)
        # When the user picks a different mode in the properties panel
        # (M1/T105), echo it through the toolbar so the two stay in sync.
        if hasattr(self.properties, "characterModeChanged"):
            self.properties.characterModeChanged.connect(
                self._on_properties_mode_changed)

    # ── Toolbar slots ────────────────────────────────────────────────────

    @QtCore.Slot(int)
    def _on_character_type_changed(self, index: int) -> None:
        if self._character_type_combo.itemData(index) != "custom_rigged_character":
            return
        self._character_type_combo.blockSignals(True)
        self._character_type_combo.setCurrentIndex(0)
        self._character_type_combo.blockSignals(False)
        self.customBuilderRequested.emit()

    @QtCore.Slot()
    def _on_mode_action_triggered(self) -> None:
        if self._suppress_mode_signal:
            return
        action = self.sender()
        if not isinstance(action, QtGui.QAction):
            return
        mode = action.data()
        self._apply_mode(mode, locked=True, source="toolbar")

    @QtCore.Slot(str)
    def _on_game_changed(self, game: str) -> None:
        if not game:
            return
        self.scene.game_version = game
        if hasattr(self, "secondary_motion_editor"):
            self.secondary_motion_editor.set_target_game(game)
        self.scene.dirty = True
        self._refresh_skeleton_template_options()
        self._update_title()
        self._schedule_live_validation("game_changed")

    @QtCore.Slot(object)
    def _on_secondary_motion_profile_changed(self, profile: object) -> None:
        metadata = getattr(self.scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self.scene, "metadata", metadata)
        metadata["secondary_motion"] = dict(profile) if isinstance(profile, dict) else {}
        self.scene.dirty = True
        self._schedule_live_validation("secondary_motion_changed")

    @QtCore.Slot(object)
    def _on_properties_mode_changed(self, mode) -> None:
        """Forward overrides from the right-side properties panel."""
        # ``None`` is the panel's '(Auto)' sentinel — unlock the scene.
        if mode is None:
            if hasattr(self.scene, "unlock_mode"):
                self.scene.unlock_mode()
            self._sync_from_scene()
            self._schedule_live_validation("mode_unlocked")
            return
        self._apply_mode(mode, locked=True, source="properties")

    @QtCore.Slot(str)
    def _on_camera_preset(self, preset: str) -> None:
        # Placeholder — wired to the viewport in M4.  We expose the
        # call site here so the toolbar is fully populated.
        viewport = getattr(self, "viewport", None)
        if viewport is None or not hasattr(viewport, "set_camera_preset"):
            log.debug("Camera preset '%s' requested (no viewport hook)", preset)
            return
        try:
            viewport.set_camera_preset(preset)              # type: ignore[attr-defined]
        except Exception as exc:                            # pragma: no cover
            log.warning("Camera preset '%s' failed: %s", preset, exc)

    def _set_rig_transform_action_checked(self, key: str) -> None:
        action = getattr(self, "_rig_transform_actions", {}).get(str(key or ""))
        if action is None:
            return
        action.blockSignals(True)
        try:
            action.setChecked(True)
        finally:
            action.blockSignals(False)

    def _run_character_builder_transform_action(self, key: str) -> None:
        """Apply the Character Builder transform tool through viewport state."""
        tool_key = str(key or "select").strip().lower()
        viewport = getattr(self, "viewport", None)
        if tool_key == "select":
            if viewport is not None and hasattr(viewport, "set_viewport_selection_mode"):
                viewport.set_viewport_selection_mode("any")
            self._set_rig_transform_action_checked("select")
            self.statusBar().showMessage("Rig tool: Select.", 4000)
            return
        mode_by_key = {"translate": 1, "rotate": 2, "transform": 3}
        mode = mode_by_key.get(tool_key, 1)
        if viewport is not None:
            renderer = getattr(viewport, "_renderer", None)
            if (
                hasattr(viewport, "toggle_gimbal")
                and not bool(getattr(renderer, "show_gimbal", True))
            ):
                viewport.toggle_gimbal(True)
            if hasattr(viewport, "set_gimbal_mode"):
                viewport.set_gimbal_mode(mode)
            if hasattr(viewport, "set_viewport_selection_mode"):
                viewport.set_viewport_selection_mode("any")
        checked_key = tool_key if tool_key in mode_by_key else "translate"
        self._set_rig_transform_action_checked(checked_key)
        label = "Move" if mode == 1 else "Rotate" if mode == 2 else "Transform"
        self.statusBar().showMessage(f"Rig tool: {label}.", 4000)

    def _set_viewport_toggle_state(
        self,
        action_name: str,
        setter_name: str,
        enabled: bool,
    ) -> None:
        viewport = getattr(self, "viewport", None)
        if viewport is not None:
            setter = getattr(viewport, setter_name, None)
            if callable(setter):
                setter(bool(enabled))
        action = getattr(self, action_name, None)
        if action is not None and action.isChecked() != bool(enabled):
            action.blockSignals(True)
            try:
                action.setChecked(bool(enabled))
            finally:
                action.blockSignals(False)

    @QtCore.Slot(bool)
    def _on_bones_toggled(self, enabled: bool) -> None:
        self._set_viewport_toggle_state("_bones_action", "toggle_bones", enabled)
        self.statusBar().showMessage(
            "Bones visible." if enabled else "Bones hidden.",
            3000,
        )

    @QtCore.Slot(bool)
    def _on_weights_toggled(self, enabled: bool) -> None:
        self._set_viewport_toggle_state("_weights_action", "toggle_weight_heatmap", enabled)
        self.statusBar().showMessage(
            "Weight heat-map visible." if enabled else "Weight heat-map hidden.",
            3000,
        )

    @QtCore.Slot(bool)
    def _on_joint_dots_toggled(self, enabled: bool) -> None:
        self._set_viewport_toggle_state("_joints_action", "toggle_joint_dots", enabled)
        self.statusBar().showMessage(
            "Joint handles visible." if enabled else "Joint handles hidden.",
            3000,
        )

    @QtCore.Slot(bool)
    def _on_joint_symmetry_toggled(self, enabled: bool) -> None:
        """Mirror the shared Symmetry toggle into toolbar, inspector, and viewport."""
        enabled = bool(enabled)
        action = getattr(self, "_symmetry_action", None)
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            try:
                action.setChecked(enabled)
            finally:
                action.blockSignals(False)
        inspector = getattr(self, "inspector", None)
        if inspector is not None and hasattr(inspector, "set_symmetry_enabled"):
            try:
                inspector.set_symmetry_enabled(enabled)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_symmetry_enabled failed")
        viewport = getattr(self, "viewport", None)
        if viewport is None or not hasattr(viewport, "set_joint_symmetry"):
            return
        try:
            viewport.set_joint_symmetry(enabled)
        except Exception:                                   # pragma: no cover
            log.exception("viewport.set_joint_symmetry failed")

    @QtCore.Slot(float)
    def _on_joint_opacity_changed(self, value: float) -> None:
        """Mirror the inspector opacity slider into joint-dot alpha."""
        viewport = getattr(self, "viewport", None)
        if viewport is None or not hasattr(viewport, "set_joint_dot_opacity"):
            return
        try:
            viewport.set_joint_dot_opacity(float(value))
        except Exception:                                   # pragma: no cover
            log.exception("viewport.set_joint_dot_opacity failed")

    @QtCore.Slot(float)
    def _on_joint_size_changed(self, value: float) -> None:
        """Map inspector 0..1 size values onto the viewport's 2..16 px dots."""
        viewport = getattr(self, "viewport", None)
        if viewport is None or not hasattr(viewport, "set_joint_dot_size"):
            return
        try:
            clamped = max(0.0, min(1.0, float(value)))
            viewport.set_joint_dot_size(round(2 + clamped * 14))
        except Exception:                                   # pragma: no cover
            log.exception("viewport.set_joint_dot_size failed")

    def _build_character_builder_transform_marking_menu(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMenu:
        """Build the plain-RMB Character Builder rig transform marking menu."""
        menu = QtWidgets.QMenu(parent or self)
        menu.setObjectName("characterBuilderTransformMarkingMenu")
        frame = QtWidgets.QFrame(menu)
        frame.setObjectName("characterBuilderTransformMarkingRadial")
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        for key, label, row, column in (
            ("select", "Select", 1, 1),
            ("translate", "Move", 0, 1),
            ("rotate", "Rotate", 1, 0),
            ("transform", "Transform", 1, 2),
        ):
            action = QtGui.QAction(label, menu)
            action.setObjectName(f"characterBuilderTransformMarkingAction_{key}")
            action.setData(key)
            action.triggered.connect(
                lambda _checked=False, tool_key=key: self._run_character_builder_transform_action(tool_key)
            )
            action.triggered.connect(menu.close)
            button = QtWidgets.QToolButton(frame)
            button.setObjectName(f"characterBuilderTransformMarkingButton_{key}")
            button.setDefaultAction(action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            button.setMinimumWidth(92)
            layout.addWidget(button, row, column)
            menu.addAction(action)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setObjectName("characterBuilderTransformMarkingRadialWidgetAction")
        widget_action.setDefaultWidget(frame)
        menu.insertAction(menu.actions()[0] if menu.actions() else None, widget_action)
        menu.insertSeparator(menu.actions()[1] if len(menu.actions()) > 1 else None)
        return menu

    def _open_character_builder_transform_marking_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = self._build_character_builder_transform_marking_menu(self)
        menu.exec(global_pos)

    def _build_character_builder_rig_marking_menu(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QMenu:
        """Build the Shift+RMB rigging/deformation marking menu."""
        menu = QtWidgets.QMenu(parent or self)
        menu.setObjectName("characterBuilderRigMarkingMenu")

        quick_frame = QtWidgets.QFrame(menu)
        quick_frame.setObjectName("characterBuilderRigMarkingQuickRadial")
        quick_layout = QtWidgets.QGridLayout(quick_frame)
        quick_layout.setContentsMargins(8, 8, 8, 8)
        quick_layout.setHorizontalSpacing(6)
        quick_layout.setVerticalSpacing(6)
        for key, label, row, column, handler in (
            ("bones", "Bones", 0, 1, self._toggle_bones_from_marking_menu),
            ("symmetry", "Symmetry", 1, 0, self._toggle_symmetry_from_marking_menu),
            ("joints", "Joints", 1, 1, self._toggle_joints_from_marking_menu),
            ("weights", "Weights", 1, 2, self._toggle_weights_from_marking_menu),
            ("center_pivot", "Center Pivot", 2, 0, self._center_pivot_from_marking_menu),
            ("refit", "Re-fit", 2, 1, self._on_refit_to_selected_base_requested),
            ("freeze_transforms", "Freeze", 2, 2, self._freeze_transform_from_marking_menu),
            ("validate", "Validate", 3, 1, self._on_validate_requested),
        ):
            action = QtGui.QAction(label, menu)
            action.setObjectName(f"characterBuilderRigMarkingQuickAction_{key}")
            action.triggered.connect(lambda _checked=False, slot=handler: slot())
            action.triggered.connect(menu.close)
            button = QtWidgets.QToolButton(quick_frame)
            button.setObjectName(f"characterBuilderRigMarkingQuickButton_{key}")
            button.setDefaultAction(action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            button.setMinimumWidth(98)
            quick_layout.addWidget(button, row, column)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setObjectName("characterBuilderRigMarkingQuickWidgetAction")
        widget_action.setDefaultWidget(quick_frame)
        menu.addAction(widget_action)

        menu.addSection("Rig Checks")
        for key, label, handler in (
            ("reset_fit", "Reset Fit Controls", self._on_fit_adjustment_reset_requested),
            ("apply_skeleton", "Apply Base Skeleton", self._on_apply_skeleton_template_requested),
            ("build_skeleton", "Build Skeleton + Weights", self._on_generate_skeleton_requested),
            ("rom", "Range of Motion Test", self._on_run_rom_test_requested),
            ("refresh_anims", "Refresh Preview Motions", self._on_refresh_preview_animations_requested),
        ):
            action = QtGui.QAction(label, menu)
            action.setObjectName(f"characterBuilderRigMarkingAction_{key}")
            action.setData(key)
            action.triggered.connect(lambda _checked=False, slot=handler: slot())
            menu.addAction(action)
        return menu

    def _open_character_builder_rig_marking_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = self._build_character_builder_rig_marking_menu(self)
        menu.exec(global_pos)

    def _toggle_bones_from_marking_menu(self, _checked: bool = False) -> None:
        action = getattr(self, "_bones_action", None)
        self._on_bones_toggled(not bool(action.isChecked()) if action is not None else True)

    def _toggle_weights_from_marking_menu(self, _checked: bool = False) -> None:
        action = getattr(self, "_weights_action", None)
        self._on_weights_toggled(not bool(action.isChecked()) if action is not None else True)

    def _toggle_joints_from_marking_menu(self, _checked: bool = False) -> None:
        action = getattr(self, "_joints_action", None)
        self._on_joint_dots_toggled(not bool(action.isChecked()) if action is not None else True)

    def _toggle_symmetry_from_marking_menu(self, _checked: bool = False) -> None:
        action = getattr(self, "_symmetry_action", None)
        self._on_joint_symmetry_toggled(not bool(action.isChecked()) if action is not None else True)

    def _center_pivot_from_marking_menu(self) -> None:
        viewport = getattr(self, "viewport", None)
        center = getattr(viewport, "center_pivot_to_selection", None)
        if callable(center) and center():
            self.statusBar().showMessage("Pivot centered on selected bounds.", 4000)
            return
        self.statusBar().showMessage("Select a mesh or object before centering its pivot.", 5000)

    def _freeze_transform_from_marking_menu(self) -> None:
        viewport = getattr(self, "viewport", None)
        freeze = getattr(viewport, "freeze_selected_transform", None)
        if callable(freeze) and freeze():
            self.statusBar().showMessage("Selected mesh transforms frozen.", 4000)
            return
        self.statusBar().showMessage("Select a mesh node before freezing transforms.", 5000)

    @staticmethod
    def _body_workflow_module():
        try:
            from core.characters import headless_body_workflow as _wf
        except ImportError:                                 # pragma: no cover
            from src.core.characters import headless_body_workflow as _wf  # type: ignore
        return _wf

    @staticmethod
    def _head_workflow_module():
        try:
            from core.characters import head_workflow as _wf
        except ImportError:                                 # pragma: no cover
            from src.core.characters import head_workflow as _wf  # type: ignore
        return _wf

    def _workflow_module(self):
        """Return the workflow service that owns the active authoring mode."""
        if self._is_scene_mode("head"):
            return self._head_workflow_module()
        return self._body_workflow_module()

    @staticmethod
    def _rig_session_module():
        try:
            from core.characters import rig_session as _rig
        except ImportError:                                 # pragma: no cover
            from src.core.characters import rig_session as _rig  # type: ignore
        return _rig

    def _restore_rig_session_from_scene(self):
        """Restore the durable rig graph attached to the active scene."""
        _rig = self._rig_session_module()
        metadata = getattr(self.scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self.scene, "metadata", metadata)
        try:
            self._rig_session = _rig.RigSession.restore_from_metadata(metadata)
        except (TypeError, ValueError):
            log.warning("Invalid RigSession metadata; starting a new session", exc_info=True)
            self._rig_session = _rig.RigSession()
        self._rig_session.store_in_metadata(metadata)
        return self._rig_session

    def _sync_rig_session_metadata(self, *, mark_dirty: bool = True) -> None:
        metadata = getattr(self.scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self.scene, "metadata", metadata)
        session = self._rig_session or self._restore_rig_session_from_scene()
        session.store_in_metadata(metadata)
        if mark_dirty:
            self.scene.dirty = True

    def _start_rig_stage(self, stage: str, *, cancellable: bool = False) -> None:
        session = self._rig_session or self._restore_rig_session_from_scene()
        session.start_stage(stage, cancellable=cancellable)
        self._sync_rig_session_metadata()

    def _fail_rig_stage(self, stage: str, message: str) -> None:
        session = self._rig_session or self._restore_rig_session_from_scene()
        session.fail_stage(stage, message)
        self._sync_rig_session_metadata()

    def _complete_rig_stage(self, stage: str, artifact: dict[str, Any]) -> None:
        session = self._rig_session or self._restore_rig_session_from_scene()
        session.complete_stage(stage, artifact)
        self._sync_rig_session_metadata()

    @staticmethod
    def _serialize_rig_guides(guides: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Copy editable guide values without retaining runtime guide objects."""
        records: dict[str, dict[str, Any]] = {}
        for raw_name, guide in sorted((guides or {}).items(), key=lambda item: str(item[0])):
            name = str(raw_name or getattr(guide, "name", "") or "").strip().lower()
            position = getattr(guide, "position", None)
            try:
                coords = [float(value) for value in tuple(position)[:3]]
            except (TypeError, ValueError):
                continue
            if len(coords) != 3 or not name:
                continue
            colour = getattr(guide, "colour", (255, 200, 0))
            try:
                colour_values = [int(value) for value in tuple(colour)[:3]]
            except (TypeError, ValueError):
                colour_values = [255, 200, 0]
            records[name] = {
                "position": coords,
                "bone_parent": getattr(guide, "bone_parent", None),
                "locked": bool(getattr(guide, "locked", False)),
                "mirror_of": getattr(guide, "mirror_of", None),
                "colour": (colour_values + [255, 200, 0])[:3],
            }
        return records

    def _body_landmark_artifact(self, source: str) -> dict[str, Any]:
        guides = dict(self._body_guides or {})
        if self._acurig is not None and hasattr(self._acurig, "get_all_guides"):
            try:
                guides = dict(self._acurig.get_all_guides() or guides)
            except Exception:                               # pragma: no cover
                log.debug("Could not snapshot AcuRig guides", exc_info=True)
        records = self._serialize_rig_guides(guides)
        return {
            "kind": "body_landmarks",
            "source": str(source or "manual"),
            "guide_count": len(records),
            "guides": records,
        }

    def _finger_artifact(
        self,
        source: str,
        *,
        guides: Optional[dict[str, Any]] = None,
        masked_bones: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if guides is None and self._acurig is not None and hasattr(self._acurig, "get_all_guides"):
            try:
                guides = dict(self._acurig.get_all_guides() or {})
            except Exception:                               # pragma: no cover
                guides = {}
        hand_names = {
            "lforearm", "lhand", "lfinger01",
            "rforearm", "rhand", "rfinger01",
        }
        serialized = self._serialize_rig_guides(dict(guides or {}))
        serialized = {name: row for name, row in serialized.items() if name in hand_names}
        if masked_bones is None:
            mask = getattr(self._acurig, "mask", None)
            values = getattr(mask, "masked_bones", []) if mask is not None else []
            masked_bones = list(values() if callable(values) else values or [])
        return {
            "kind": "finger_landmarks",
            "source": str(source or "manual"),
            "guide_count": len(serialized),
            "guides": serialized,
            "masked_bones": sorted(str(value) for value in (masked_bones or [])),
        }

    def _restore_body_guides_from_rig_session(self) -> bool:
        """Rebuild lightweight AcuRig guide objects from saved session data."""
        _rig = self._rig_session_module()
        session = self._rig_session or self._restore_rig_session_from_scene()
        body_state = session.state(_rig.RigStage.BODY_LANDMARKS)
        if (
            not body_state.has_preserved_output
            or body_state.status in {_rig.RigStageStatus.PENDING, _rig.RigStageStatus.STALE}
        ):
            self._body_guides = {}
            self._acurig = None
            return False
        body_artifact = body_state.artifact
        records = body_artifact.get("guides") if isinstance(body_artifact, dict) else None
        if not isinstance(records, dict) or not records:
            self._body_guides = {}
            self._acurig = None
            return False
        records = dict(records)
        finger_state = session.state(_rig.RigStage.FINGERS)
        finger_artifact = (
            finger_state.artifact
            if finger_state.has_preserved_output
            and finger_state.status not in {_rig.RigStageStatus.PENDING, _rig.RigStageStatus.STALE}
            else {}
        )
        finger_records = (
            finger_artifact.get("guides")
            if isinstance(finger_artifact, dict) else None
        )
        if isinstance(finger_records, dict):
            records.update(finger_records)
        try:
            try:
                from src.autorig.accurig import AcuRig, RigGuide
            except ImportError:                            # pragma: no cover
                from autorig.accurig import AcuRig, RigGuide  # type: ignore
            guides = {}
            for raw_name, record in records.items():
                if not isinstance(record, dict):
                    continue
                position = tuple(float(value) for value in list(record.get("position") or ())[:3])
                if len(position) != 3:
                    continue
                name = str(raw_name or "").strip().lower()
                guides[name] = RigGuide(
                    name=name,
                    position=position,
                    bone_parent=record.get("bone_parent"),
                    locked=bool(record.get("locked", False)),
                    mirror_of=record.get("mirror_of"),
                    colour=tuple(int(value) for value in list(record.get("colour") or (255, 200, 0))[:3]),
                )
            if not guides:
                return False
            acurig = AcuRig()
            acurig._guides = guides
            masked = (
                list(finger_artifact.get("masked_bones") or [])
                if isinstance(finger_artifact, dict) else []
            )
            for bone in masked:
                acurig.mask.mask(str(bone))
            self._acurig = acurig
            self._body_guides = dict(guides)
            if hasattr(getattr(self, "inspector", None), "set_hand_masked_bones"):
                self.inspector.set_hand_masked_bones(masked)
            self._push_body_guides_to_viewport()
            return True
        except Exception:                                  # pragma: no cover
            log.exception("Could not restore saved RigSession landmarks")
            return False

    def _ensure_body_guide_history(self):
        _wf = self._body_workflow_module()
        if self._body_guide_history is None:
            self._body_guide_history = _wf.BodyGuideEditHistory()
        return self._body_guide_history

    def _refresh_body_guide_undo_actions(self) -> None:
        history = self._body_guide_history
        can_undo = bool(getattr(history, "can_undo", False))
        can_redo = bool(getattr(history, "can_redo", False))
        if hasattr(self, "_undo_guide_action"):
            self._undo_guide_action.setEnabled(can_undo)
        if hasattr(self, "_redo_guide_action"):
            self._redo_guide_action.setEnabled(can_redo)

    def _push_body_guides_to_viewport(self) -> None:
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_acurig_guides"):
            try:
                viewport.set_acurig_guides(self._body_guides)
            except Exception:                               # pragma: no cover
                log.exception("viewport.set_acurig_guides failed")

    def _apply_body_guide_history_result(self, result) -> None:
        if not getattr(result, "ok", False):
            if hasattr(self.inspector, "set_body_rig_status"):
                self.inspector.set_body_rig_status(
                    getattr(result, "message", "Guide edit unavailable."),
                    kind="warning",
                )
            self.statusBar().showMessage(getattr(result, "message", ""), 5000)
            self._refresh_body_guide_undo_actions()
            return
        self._body_guides = dict(getattr(result, "guides", {}) or {})
        self._push_body_guides_to_viewport()
        self._complete_rig_stage(
            "body_landmarks",
            self._body_landmark_artifact(str(getattr(result, "code", "history") or "history")),
        )
        if hasattr(self.inspector, "set_body_rig_status"):
            self.inspector.set_body_rig_status(result.message, kind="ok")
        try:
            self.scene.dirty = True
        except Exception:                                  # pragma: no cover
            pass
        self.statusBar().showMessage(result.message, 4000)
        self._refresh_body_guide_undo_actions()
        self._update_title()
        self._schedule_live_validation("body_guide_history")

    @QtCore.Slot()
    def _on_undo_body_guide_requested(self) -> None:
        """Undo the latest AccuRig guide edit."""
        _wf = self._body_workflow_module()
        result = _wf.undo_body_guide_edit(
            self._acurig,
            self._ensure_body_guide_history(),
        )
        self._apply_body_guide_history_result(result)

    @QtCore.Slot()
    def _on_redo_body_guide_requested(self) -> None:
        """Redo the latest undone AccuRig guide edit."""
        _wf = self._body_workflow_module()
        result = _wf.redo_body_guide_edit(
            self._acurig,
            self._ensure_body_guide_history(),
        )
        self._apply_body_guide_history_result(result)

    @QtCore.Slot(object)
    def _on_viewport_node_moved(self, node) -> None:
        """Persist body joint-dot drags as AcuRig guide overrides."""
        if self._acurig is None:
            return
        _wf = self._body_workflow_module()

        result = _wf.update_body_guide_from_node(
            self._acurig,
            node,
            auto_mirror=bool(
                getattr(getattr(self, "viewport", None), "joint_symmetry_enabled", False)
            ),
        )
        if not getattr(result, "ok", False):
            return
        self._body_guide_history = _wf.record_body_guide_edit(
            self._ensure_body_guide_history(),
            result,
        )

        self._body_guides = dict(getattr(result, "guides", {}) or {})
        self._push_body_guides_to_viewport()
        self._complete_rig_stage(
            "body_landmarks",
            self._body_landmark_artifact("viewport_edit"),
        )

        if hasattr(self.inspector, "set_body_rig_status"):
            try:
                self.inspector.set_body_rig_status(
                    result.message,
                    kind="ok",
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_body_rig_status failed")

        try:
            self.scene.dirty = True
        except Exception:                                  # pragma: no cover
            pass
        self._refresh_body_guide_undo_actions()
        self._update_title()
        self._schedule_live_validation("viewport_node_moved")

    # ── Inspector slots ──────────────────────────────────────────────────

    @QtCore.Slot()
    def _on_load_model_requested(self) -> None:
        """Load custom art through the active Head or Body workflow.

        Opens a file picker scoped to the formats the
        active workflow accepts, invokes the owning service, and reports
        the result through the bottom-strip validation banner. Head mode
        accepts geometry-only external art without requiring a body
        skeleton; donor selection and transplant remain later Head Builder
        stages.

        Mode-mismatch handling: when the auto-detector says the file
        looks like a Head / Creature / Supermodel rather than a
        Headless Body, the user is prompted to either:
          • switch the active mode to match the detected file (and
            keep the load), or
          • cancel the load (which leaves the slot assigned but warns
            in the banner).
        """
        if self._is_scene_mode("supermodel"):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Complete character?",
                "Supermodel mode is for KOTOR's separate body + head preview "
                "workflow.\n\nIs this file a complete all-in-one character "
                "mesh with the head already attached?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer == QtWidgets.QMessageBox.No:
                self._on_load_composite_requested()
                return
            if _CHARACTER_MODE_AVAILABLE and CharacterMode is not None:
                try:
                    self._apply_mode(
                        CharacterMode.HEADLESS_BODY,
                        locked=True,
                        source="supermodel_complete_character_load",
                    )
                except Exception:                          # pragma: no cover
                    log.debug("Could not switch complete character load mode",
                              exc_info=True)

        is_head_mode = self._is_scene_mode("head")
        try:
            _wf = self._workflow_module()
        except Exception as exc:                            # pragma: no cover
            log.exception("Could not import active Character Builder workflow")
            self.bottom_strip.set_validation(
                "error", "LOAD_UNAVAILABLE",
                issues=[f"Workflow service unavailable: {exc}"],
            )
            return

        if not is_head_mode and not self._selected_skeleton_template_model:
            message = (
                "Choose a KOTOR base skeleton before loading the custom mesh. "
                "GhostRigger uses that base to auto-scale and orient the import."
            )
            if hasattr(self.inspector, "set_skeleton_template_status"):
                self.inspector.set_skeleton_template_status(message, kind="warning")
            self.bottom_strip.set_validation(
                "warning", "BASE_SKELETON_REQUIRED", issues=[message]
            )
            self.statusBar().showMessage(message, 7000)
            try:
                self.inspector.set_step(1)
            except Exception:
                pass
            return

        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Custom Head Art" if is_head_mode else "Load Body Model",
            "",
            _wf.load_file_filter(),
        )
        if not path:
            return

        gv = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
             getattr(self.scene, "game_version", "K1")
        self._start_rig_stage("source")
        if is_head_mode:
            external_art = Path(path).suffix.lower() in {
                ".fbx", ".obj", ".gltf", ".glb", ".ply", ".stl",
            }
            result = _wf.load_head(
                path,
                self.scene,
                game_version=gv,
                allow_mode_correction=external_art,
            )
        else:
            fit_label = self._selected_skeleton_template_fit_label()
            result = _wf.load_body(
                path,
                self.scene,
                game_version=gv,
                fit_reference_model=self._selected_skeleton_template_model,
                fit_reference_label=fit_label,
                expected_mode=getattr(self.scene, "mode", None),
            )

        # ── Mode mismatch — offer to switch ──────────────────────────
        if result.code == "mode_mismatch" and result.detected_mode is not None:
            detected_label = getattr(result.detected_mode, "display_name",
                                     str(result.detected_mode))
            current_mode = getattr(self.scene, "mode", None)
            current_label = getattr(current_mode, "display_name",
                                    str(current_mode or "current mode"))
            answer = QtWidgets.QMessageBox.question(
                self,
                "Wrong character mode?",
                f"This file looks like a {detected_label} model, not a "
                f"{current_label}.\n\nSwitch the Character Builder to "
                f"{detected_label} mode and keep this file loaded?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self._apply_mode(result.detected_mode,
                                 locked=True, source="load_model_autoswitch")
                self.bottom_strip.set_validation(
                    "info", "LOADED",
                    issues=[f"Loaded {result.resref}; mode → {detected_label}"],
                )
                self.statusBar().showMessage(
                    f"Loaded {result.resref} (mode switched to {detected_label})",
                    5000,
                )
                self._on_model_loaded_into_scene(result)
                return
            # User declined the switch — keep the slot, warn the banner.
            self.bottom_strip.set_validation(
                "warning", "MODE_MISMATCH",
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            self._on_model_loaded_into_scene(result)
            return

        # ── Hard failures ────────────────────────────────────────────
        if not result.ok:
            self._fail_rig_stage("source", result.message)
            self.bottom_strip.set_validation(
                "error", result.code.upper(),
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            return

        # ── Happy path ───────────────────────────────────────────────
        self.bottom_strip.set_validation(
            "info", "LOADED",
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)
        self._on_model_loaded_into_scene(result)

    def _is_scene_mode(self, value: str) -> bool:
        mode = getattr(self.scene, "mode", None)
        mode_value = (
            getattr(mode, "value", None)
            or getattr(mode, "name", "")
            or str(mode or "")
        ).lower()
        return mode_value == value.lower()

    def _on_load_composite_requested(self) -> None:
        """Workflow Step 1 for M7 Supermodel mode: load body + head."""
        try:
            from core.workflow import composite_workflow as _cw
            from core.characters import head_workflow as _head_wf
            from core.characters import headless_body_workflow as _body_wf
        except ImportError:                                 # pragma: no cover
            try:
                from src.core.workflow import composite_workflow as _cw       # type: ignore
                from src.core.characters import head_workflow as _head_wf       # type: ignore
                from src.core.characters import headless_body_workflow as _body_wf  # type: ignore
            except Exception as exc:
                log.exception("Could not import composite workflow")
                self.bottom_strip.set_validation(
                    "error", "COMPOSITE_UNAVAILABLE",
                    issues=[f"Composite workflow unavailable: {exc}"],
                )
                return

        body_path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Body Model",
            "",
            _body_wf.load_file_filter(),
        )
        if not body_path:
            return

        head_path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Head Model",
            "",
            _head_wf.load_file_filter(),
        )
        if not head_path:
            return

        gv = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
             getattr(self.scene, "game_version", "K1")
        self._start_rig_stage("source")
        result = _cw.load_composite(
            self.scene,
            body_path=body_path,
            head_path=head_path,
            game_version=gv,
            build_preview=True,
        )

        issues = [result.message]
        snap = getattr(result, "snap", None)
        if snap is not None:
            if getattr(snap, "message", ""):
                issues.append(snap.message)
            issues.extend(list(getattr(snap, "warnings", []) or []))

        if not result.ok:
            self._fail_rig_stage("source", result.message)
            self.bottom_strip.set_validation(
                "error", (result.code or "composite").upper(),
                issues=issues,
            )
            self.statusBar().showMessage(result.message, 6000)
            self._sync_from_scene()
            self._update_title()
            return

        self.bottom_strip.set_validation(
            "info", "COMPOSITE_LOADED", issues=issues,
        )
        self._complete_rig_stage(
            "source",
            {
                "kind": "composite_source",
                "body_path": str(body_path),
                "head_path": str(head_path),
                "game_version": str(gv),
            },
        )
        self.statusBar().showMessage(result.message, 5000)
        self._sync_from_scene()
        try:
            preview_model = getattr(snap, "preview_model", None)
            body_model = getattr(getattr(result, "body_result", None), "model", None)
            model = preview_model or body_model
            if (model is not None
                    and hasattr(self, "viewport")
                    and hasattr(self.viewport, "load_model")):
                self.viewport.load_model(model)
        except Exception:                                    # pragma: no cover
            log.exception("Failed to push composite preview into viewport")
        self._update_title()
        self._schedule_live_validation("composite_loaded")

    def _on_model_loaded_into_scene(self, result) -> None:
        """Post-load housekeeping shared by every load branch.

        Pushes the new model into the viewport, refreshes the workflow
        rail, and marks the scene dirty so File→Save offers to write.
        """
        self._complete_rig_stage(
            "source",
            {
                "kind": "source_model",
                "source_path": str(getattr(result, "source_path", "") or ""),
                "resref": str(getattr(result, "resref", "") or ""),
                "game_version": str(getattr(self.scene, "game_version", "K1") or "K1"),
            },
        )
        # Sync rail / properties panel with the (possibly auto-updated)
        # CharacterMode now reflected in the scene.
        self._sync_from_scene()
        # Push the model into the viewport so the user sees it.
        try:
            if (result.model is not None
                    and hasattr(self, "viewport")
                    and hasattr(self.viewport, "load_model")):
                self._load_model_in_viewport_with_textures(
                    result.model,
                    source_path=str(getattr(result, "source_path", "") or ""),
                    prompt=True,
                )
                if (
                    not self._is_scene_mode("head")
                    and
                    self._selected_skeleton_template_model is not None
                    and hasattr(self.viewport, "set_external_skeleton")
                ):
                    self.viewport.set_external_skeleton(
                        self._selected_skeleton_template_model,
                        fit_to_model=False,
                    )
                if hasattr(self.viewport, "clear_acurig_guides"):
                    self.viewport.clear_acurig_guides()
        except Exception:                                    # pragma: no cover
            log.exception("Failed to push loaded model into the viewport")
        self._body_guides = {}
        self._body_guide_history = None
        self._refresh_body_guide_undo_actions()
        self._manual_fit_scale = 1.0
        self._manual_fit_rotation = (0.0, 0.0, 0.0)
        self._manual_fit_translation = (0.0, 0.0, 0.0)
        if hasattr(self.inspector, "set_fit_adjustment"):
            try:
                self.inspector.set_fit_adjustment(
                    scale=1.0,
                    rotation_degrees=(0.0, 0.0, 0.0),
                    translation=(0.0, 0.0, 0.0),
                    emit=False,
                )
            except Exception:
                log.exception("inspector.set_fit_adjustment failed")
        self._push_import_fit_report_to_inspector(result.model)
        self._refresh_skeleton_template_options()
        self._refresh_motion_assignment_state()
        self._update_title()
        self._schedule_live_validation("model_loaded")

    def _extract_import_fit_report(self, model: Any) -> Optional[dict[str, Any]]:
        """Return the auto-fit report persisted on an imported external mesh."""
        metadata = getattr(model, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        report = metadata.get("kotor_fit_report")
        if isinstance(report, dict):
            return report
        normalization = metadata.get("kotor_normalization")
        if isinstance(normalization, dict):
            nested = normalization.get("fit_report")
            if isinstance(nested, dict):
                return nested
        return None

    def _push_import_fit_report_to_inspector(self, model: Any = None) -> None:
        """Synchronize Character Builder inspector fit evidence from *model*."""
        if model is None:
            _entry, model = self._body_model_for_fit_adjustment()
        report = self._extract_import_fit_report(model)
        if hasattr(self.inspector, "set_import_fit_report"):
            try:
                self.inspector.set_import_fit_report(report)
            except Exception:                                  # pragma: no cover
                log.exception("inspector.set_import_fit_report failed")
        viewport = getattr(self, "viewport", None)
        if viewport is None:
            return
        overlay = None
        if isinstance(report, dict):
            fitted = report.get("fitted_visual_overlay")
            visual = report.get("visual_overlay")
            overlay = fitted if isinstance(fitted, dict) else visual if isinstance(visual, dict) else None
        try:
            if overlay is not None and hasattr(viewport, "set_character_fit_overlay"):
                viewport.set_character_fit_overlay(overlay)
            elif hasattr(viewport, "clear_character_fit_overlay"):
                viewport.clear_character_fit_overlay()
        except Exception:                                      # pragma: no cover
            log.exception("viewport fit overlay sync failed")

    def _selected_skeleton_template_fit_label(self) -> str:
        selected_option = self._skeleton_template_options_by_key.get(
            str(self._selected_skeleton_template_key or "")
        )
        if selected_option is None:
            return ""
        return str(
            self._option_field(selected_option, "source_resref", "")
            or self._option_field(selected_option, "resref", "")
            or self._option_field(selected_option, "name", "")
            or ""
        )

    def _skeleton_template_requested_resref(self, option: Any) -> str:
        return str(
            self._option_field(option, "resref", "")
            or self._option_field(option, "name", "")
            or ""
        ).strip()

    def _skeleton_template_source_resref(self, option: Any) -> str:
        return str(
            self._option_field(option, "source_resref", "")
            or self._skeleton_template_requested_resref(option)
            or ""
        ).strip()

    def _skeleton_template_status_label(self, option: Any) -> str:
        requested = self._skeleton_template_requested_resref(option)
        source = self._skeleton_template_source_resref(option)
        if source and requested and source.lower() != requested.lower():
            return f"{source} (requested target {requested})"
        return source or requested or str(self._option_field(option, "name", "") or "")

    def _external_import_source_path(self, model: Any, entry: Any = None) -> str:
        metadata = getattr(model, "metadata", None)
        if isinstance(metadata, dict):
            external = metadata.get("external_import")
            if isinstance(external, dict):
                path = str(external.get("source_path") or "")
                if path:
                    return path
        return str(getattr(entry, "source_path", "") or "")

    def _body_model_for_fit_adjustment(self) -> tuple[Any, Any]:
        try:
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.geometry import model_data as _md           # type: ignore
        entry = self.scene.get(_md.PartSlot.HEADLESS_BODY)
        model = getattr(entry, "model", None) if entry is not None else None
        return entry, model

    def _model_render_bounds_for_template_check(
        self,
        model: Any,
    ) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        nodes = []
        try:
            nodes = list(model.all_nodes()) if model is not None and hasattr(model, "all_nodes") else []
        except Exception:
            nodes = []
        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]
        found = False
        for node in nodes:
            if not getattr(node, "vertices", None):
                continue
            if bool(getattr(node, "_gr_hidden", False)):
                continue
            for vertex in list(getattr(node, "vertices", []) or []):
                try:
                    x, y, z = float(vertex[0]), float(vertex[1]), float(vertex[2])
                except Exception:
                    continue
                mins[0] = min(mins[0], x)
                mins[1] = min(mins[1], y)
                mins[2] = min(mins[2], z)
                maxs[0] = max(maxs[0], x)
                maxs[1] = max(maxs[1], y)
                maxs[2] = max(maxs[2], z)
                found = True
        if not found:
            return None
        return (tuple(mins), tuple(maxs))

    @staticmethod
    def _bounds_diag(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
        return sum((float(bounds[1][axis]) - float(bounds[0][axis])) ** 2 for axis in range(3)) ** 0.5

    @staticmethod
    def _bounds_center(bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float]:
        return tuple((float(bounds[0][axis]) + float(bounds[1][axis])) * 0.5 for axis in range(3))

    def _mesh_bounds_match_template_for_binding(self, mesh_model: Any, template_model: Any) -> bool:
        mesh_bounds = self._model_render_bounds_for_template_check(mesh_model)
        template_bounds = self._model_render_bounds_for_template_check(template_model)
        if mesh_bounds is None or template_bounds is None:
            return True
        mesh_diag = self._bounds_diag(mesh_bounds)
        template_diag = self._bounds_diag(template_bounds)
        if mesh_diag <= 1.0e-6 or template_diag <= 1.0e-6:
            return True
        scale_ratio = mesh_diag / template_diag
        mesh_center = self._bounds_center(mesh_bounds)
        template_center = self._bounds_center(template_bounds)
        center_delta = sum((mesh_center[axis] - template_center[axis]) ** 2 for axis in range(3)) ** 0.5
        return 0.55 <= scale_ratio <= 1.80 and center_delta <= max(template_diag * 0.45, 0.25)

    def _ensure_template_fitted_body_for_binding(
        self,
        mesh_model: Any,
        entry: Any,
        template_model: Any,
        option: Any,
        game: str,
    ) -> Any:
        """Re-fit the source import before binding if the scene holds raw OBJ space."""
        if self._mesh_bounds_match_template_for_binding(mesh_model, template_model):
            return mesh_model
        source_path = self._external_import_source_path(mesh_model, entry)
        if not source_path or not os.path.isfile(source_path):
            return mesh_model
        try:
            from core.characters import headless_body_workflow as _wf
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.characters import headless_body_workflow as _wf  # type: ignore
            from src.core.geometry import model_data as _md  # type: ignore

        fit_override = {}
        if hasattr(self.inspector, "selected_fit_override"):
            try:
                fit_override = self.inspector.selected_fit_override()
            except Exception:
                fit_override = {}
        fit_label = self._skeleton_template_status_label(option)
        result = _wf.load_body(
            source_path,
            self.scene,
            game_version=game,
            allow_mode_correction=True,
            fit_reference_model=template_model,
            fit_reference_label=fit_label,
            fit_override=fit_override,
            expected_mode=getattr(self.scene, "mode", None),
        )
        if not result.ok:
            log.warning(
                "Character Builder: pre-bind template refit failed for %s: %s",
                source_path,
                getattr(result, "message", ""),
            )
            return mesh_model
        refreshed = self.scene.get(_md.PartSlot.HEADLESS_BODY)
        fitted_model = getattr(refreshed, "model", None) if refreshed is not None else None
        if fitted_model is not None:
            self.bottom_strip.set_log_tail("pre-bind fit refreshed from selected KOTOR base")
            return fitted_model
        return mesh_model

    @QtCore.Slot(float, float, float, float, float, float, float)
    def _on_fit_adjustment_changed(
        self,
        scale: float,
        rx: float,
        ry: float,
        rz: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> None:
        """Apply manual scale/orientation/translation correction after auto-fit."""
        _entry, model = self._body_model_for_fit_adjustment()
        if model is None:
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(
                    "Load a custom mesh before adjusting fit.",
                    kind="warning",
                )
            return

        old_scale = max(0.01, float(self._manual_fit_scale or 1.0))
        new_scale = max(0.01, float(scale or 1.0))
        old_rot = tuple(float(v or 0.0) for v in self._manual_fit_rotation)
        new_rot = (float(rx or 0.0), float(ry or 0.0), float(rz or 0.0))
        delta_rot = tuple(new_rot[i] - old_rot[i] for i in range(3))
        old_translation = tuple(float(v or 0.0) for v in self._manual_fit_translation)
        new_translation = (float(tx or 0.0), float(ty or 0.0), float(tz or 0.0))
        delta_translation = tuple(new_translation[i] - old_translation[i] for i in range(3))
        delta_scale = new_scale / old_scale
        if (
            abs(delta_scale - 1.0) < 1e-6
            and all(abs(v) < 1e-6 for v in delta_rot)
            and all(abs(v) < 1e-6 for v in delta_translation)
        ):
            return

        try:
            from core.characters import headless_body_workflow as _wf
        except ImportError:                                 # pragma: no cover
            from src.core.characters import headless_body_workflow as _wf  # type: ignore

        result = _wf.apply_external_model_fit_adjustment(
            model,
            rotation_delta_degrees=delta_rot,
            scale_delta=delta_scale,
            translation_delta=delta_translation,
        )
        if not bool(result.get("ok")):
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(
                    str(result.get("message") or "Fit adjustment did not apply."),
                    kind="warning",
                )
            return

        self._manual_fit_scale = new_scale
        self._manual_fit_rotation = new_rot
        self._manual_fit_translation = new_translation
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "refresh_model_geometry"):
            viewport.refresh_model_geometry()
            if hasattr(viewport, "frame_all"):
                viewport.frame_all()
        try:
            self.scene.dirty = True
        except Exception:
            pass
        if hasattr(self.inspector, "set_fit_adjustment_status"):
            self.inspector.set_fit_adjustment_status(
                f"Fit adjusted: {new_scale * 100:.0f}%, pos {new_translation[0]:.3f}/{new_translation[1]:.3f}/{new_translation[2]:.3f}, rot {new_rot[0]:.1f}/{new_rot[1]:.1f}/{new_rot[2]:.1f}.",
                kind="ok",
            )
        self._update_title()
        self._schedule_live_validation("manual_fit_adjusted")

    @QtCore.Slot()
    def _on_fit_adjustment_reset_requested(self) -> None:
        """Reset the manual-fit controls for the next imported mesh."""
        self._manual_fit_scale = 1.0
        self._manual_fit_rotation = (0.0, 0.0, 0.0)
        self._manual_fit_translation = (0.0, 0.0, 0.0)
        if hasattr(self.inspector, "set_fit_adjustment"):
            self.inspector.set_fit_adjustment(
                scale=1.0,
                rotation_degrees=(0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0),
                emit=False,
            )
        if hasattr(self.inspector, "set_fit_adjustment_status"):
            self.inspector.set_fit_adjustment_status(
                "Fit controls reset. Reload the mesh to discard applied corrections.",
                kind="info",
            )
        self._push_import_fit_report_to_inspector()

    @QtCore.Slot()
    def _on_refit_to_selected_base_requested(self) -> None:
        """Reload the original external mesh and auto-fit to the selected base."""
        entry, model = self._body_model_for_fit_adjustment()
        if model is None:
            message = "Load a custom mesh before re-fitting to the selected base."
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(message, kind="warning")
            self.statusBar().showMessage(message, 6000)
            return
        if self._selected_skeleton_template_model is None:
            message = "Choose a KOTOR base skeleton before re-fitting the custom mesh."
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(message, kind="warning")
            self.statusBar().showMessage(message, 6000)
            return

        source_path = self._external_import_source_path(model, entry)
        if not source_path or not os.path.isfile(source_path):
            message = (
                "Cannot re-fit because the original external mesh path is missing. "
                "Load the custom mesh again."
            )
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(message, kind="warning")
            self.statusBar().showMessage(message, 7000)
            return

        try:
            from src.core.characters import headless_body_workflow as _wf
        except Exception:
            try:
                from core.characters import headless_body_workflow as _wf  # type: ignore
            except Exception as exc:                         # pragma: no cover
                message = f"Workflow service unavailable: {exc}"
                if hasattr(self.inspector, "set_fit_adjustment_status"):
                    self.inspector.set_fit_adjustment_status(message, kind="error")
                self.statusBar().showMessage(message, 7000)
                return

        gv = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
             getattr(self.scene, "game_version", "K1")
        fit_label = self._selected_skeleton_template_fit_label()
        fit_override = {}
        if hasattr(self.inspector, "selected_fit_override"):
            try:
                fit_override = self.inspector.selected_fit_override()
            except Exception:
                log.exception("inspector.selected_fit_override failed")
        result = _wf.load_body(
            source_path,
            self.scene,
            game_version=gv,
            fit_reference_model=self._selected_skeleton_template_model,
            fit_reference_label=fit_label,
            fit_override=fit_override,
            expected_mode=getattr(self.scene, "mode", None),
        )
        if not result.ok:
            message = str(result.message or "Re-fit failed.")
            if hasattr(self.inspector, "set_fit_adjustment_status"):
                self.inspector.set_fit_adjustment_status(message, kind="error")
            self.bottom_strip.set_validation(
                "error",
                str(result.code or "REFIT_FAILED").upper(),
                issues=[message],
            )
            self.statusBar().showMessage(message, 7000)
            return

        self._on_model_loaded_into_scene(result)
        if hasattr(self.inspector, "set_fit_adjustment_status"):
            self.inspector.set_fit_adjustment_status(
                f"Re-fit to {fit_label or 'selected KOTOR base'} completed from the original import.",
                kind="ok",
            )
        self.bottom_strip.set_validation(
            "info",
            "REFIT_COMPLETE",
            issues=[f"Re-fit external mesh to {fit_label or 'selected KOTOR base'}."],
        )
        self.statusBar().showMessage("Re-fit to selected base completed.", 6000)

    def _load_model_in_viewport_with_textures(
        self,
        model: Any,
        *,
        source_path: str = "",
        prompt: bool = False,
    ) -> None:
        """Load a model and resolve external texture folders for OBJ/FBX/glTF."""
        dirs = self._resolve_external_texture_dirs(model, source_path, prompt=prompt)
        self.viewport.load_model(model, extra_texture_dirs=dirs)

    def _resolve_external_texture_dirs(
        self,
        model: Any,
        source_path: str,
        *,
        prompt: bool,
    ) -> list[str]:
        try:
            from core.characters import headless_body_workflow as _wf
        except ImportError:                                 # pragma: no cover
            from src.core.characters import headless_body_workflow as _wf  # type: ignore

        metadata = getattr(self.scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self.scene, "metadata", metadata)
        stored = [
            str(path)
            for path in list(metadata.get("external_texture_dirs", []) or [])
            if path and os.path.isdir(str(path))
        ]
        candidates = _wf.candidate_texture_dirs(source_path)
        dirs: list[str] = []
        seen_dirs: set[str] = set()
        for directory in stored + candidates:
            key = os.path.normcase(os.path.abspath(directory)) if directory else ""
            if directory and os.path.isdir(directory) and key not in seen_dirs:
                seen_dirs.add(key)
                dirs.append(directory)

        _wf.reconcile_external_texture_names(model, dirs)
        report = _wf.texture_resolution_report(model, dirs)
        names = list(report.get("expected", []) or [])
        missing = list(report.get("missing", []) or [])
        if names and missing and prompt:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Locate texture folder",
                str(Path(source_path).resolve().parent) if source_path else "",
                QtWidgets.QFileDialog.ShowDirsOnly,
            )
            if chosen and os.path.isdir(chosen):
                chosen_key = os.path.normcase(os.path.abspath(chosen))
                if chosen_key not in seen_dirs:
                    seen_dirs.add(chosen_key)
                    dirs.insert(0, chosen)
                _wf.reconcile_external_texture_names(model, dirs)
                report = _wf.texture_resolution_report(model, dirs)
                missing = list(report.get("missing", []) or [])

        metadata["external_texture_dirs"] = dirs
        metadata["external_texture_report"] = report
        if names:
            if missing:
                self.bottom_strip.set_log_tail(
                    f"missing texture(s): {', '.join(missing[:3])}"
                )
            elif report.get("found_count"):
                self.bottom_strip.set_log_tail(
                    f"textures: {int(report.get('found_count', 0))} found"
                )
        return dirs

    def _ensure_game_resource_manager(self, game: str = "") -> Optional[Any]:
        """Index the configured game install and wire it to animation/texture systems."""
        raw_game = game or getattr(self.scene, "game_version", "K1") or "K1"
        try:
            from src.core.characters.headless_body_workflow import normalize_kotor_game_tag
        except ImportError:                              # pragma: no cover
            from core.characters.headless_body_workflow import normalize_kotor_game_tag  # type: ignore
        game_key = normalize_kotor_game_tag(raw_game)
        if self._resource_manager is None:
            try:
                from src.core.assets.resource_manager import get_manager
            except ImportError:                             # pragma: no cover
                from core.assets.resource_manager import get_manager  # type: ignore
            self._resource_manager = get_manager()

        if game_key not in self._resource_manager_games:
            try:
                from src.resources.game_detector import detect_kotor_dirs
            except ImportError:                             # pragma: no cover
                from resources.game_detector import detect_kotor_dirs  # type: ignore
            k1_dir, k2_dir = detect_kotor_dirs(prefer_config=True)
            if game_key == "K1" and k1_dir:
                if self._resource_manager.set_k1_dir(k1_dir):
                    self._resource_manager_games.add("K1")
            elif game_key == "K2" and k2_dir:
                if self._resource_manager.set_k2_dir(k2_dir):
                    self._resource_manager_games.add("K2")

        try:
            from src.core.animation.animation_engine import SuperModelResolver
        except ImportError:                                 # pragma: no cover
            from core.animation.animation_engine import SuperModelResolver  # type: ignore
        if getattr(SuperModelResolver, "_resource_manager", None) is not self._resource_manager:
            try:
                SuperModelResolver.clear_cache()
            except Exception:                               # pragma: no cover
                log.debug("SuperModelResolver.clear_cache failed", exc_info=True)
        SuperModelResolver.configure(self._resource_manager)

        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_resource_manager"):
            try:
                viewport.set_resource_manager(self._resource_manager, game_key)
            except Exception:                               # pragma: no cover
                log.debug("viewport.set_resource_manager failed", exc_info=True)
        return self._resource_manager

    def _body_model_for_preview(self) -> Optional[Any]:
        try:
            from src.core.geometry.model_data import PartSlot
        except ImportError:                                 # pragma: no cover
            from core.geometry.model_data import PartSlot  # type: ignore
        entry = self.scene.get(PartSlot.HEADLESS_BODY)
        return getattr(entry, "model", None) if entry is not None else None

    # ── M12 / T1202 — KOTOR skeleton template picker ────────────────────

    @staticmethod
    def _option_field(option: Any, name: str, default: Any = "") -> Any:
        if isinstance(option, dict):
            return option.get(name, default)
        return getattr(option, name, default)

    def _installed_skeleton_template_rows(self, game: str) -> list[dict[str, str]]:
        """Return installed KOTOR MDLs for the base-skeleton picker."""
        try:
            from src.core.characters.headless_body_workflow import normalize_kotor_game_tag
        except ImportError:                              # pragma: no cover
            from core.characters.headless_body_workflow import normalize_kotor_game_tag  # type: ignore
        game_key = normalize_kotor_game_tag(game)
        cached = self._installed_skeleton_template_rows_by_game.get(game_key)
        if cached is not None:
            return list(cached)

        rows: list[dict[str, str]] = []
        try:
            try:
                from core.characters import character_builder as _cb
                from core.game.kotor_install import KotorInstallation  # type: ignore
            except ImportError:                                  # pragma: no cover
                from src.core.characters import character_builder as _cb      # type: ignore
                from src.core.game.kotor_install import KotorInstallation  # type: ignore

            root = _cb._detect_game_dir(game_key)
            if root and os.path.isdir(root):
                inst = KotorInstallation(root)
                for resref in inst.list_models():
                    name = str(resref or "").strip().lower()
                    if not name:
                        continue
                    rows.append({
                        "resref": name,
                        "name": name,
                        "source": "installation",
                        "path": f"installation:{name}.mdl",
                    })
        except Exception:
            log.debug("Could not scan installed skeleton template rows", exc_info=True)

        self._installed_skeleton_template_rows_by_game[game_key] = rows
        return list(rows)

    def _load_skeleton_template_model(self, option: Any) -> Optional[Any]:
        """Load the selected KOTOR skeleton reference from game data."""
        try:
            from core.characters import character_builder as _cb
        except ImportError:                                 # pragma: no cover
            from src.core.characters import character_builder as _cb    # type: ignore

        source = str(self._option_field(option, "source", ""))
        game = str(self._option_field(option, "game", "") or
                   getattr(self.scene, "game_version", "K1"))
        part = str(self._option_field(option, "part", "body") or "body")
        requested_resref = str(self._option_field(option, "resref", "") or
                               self._option_field(option, "name", "") or "")
        source_resref = str(self._option_field(option, "source_resref", "") or "")
        resref = str(source_resref or requested_resref)
        path = str(self._option_field(option, "path", "") or "")

        if source == "bundled":
            return _cb.load_template(game=game, part=part)

        if path and not path.startswith("installation:") and os.path.isfile(path):
            try:
                from core.game.kotor_loader import load_model_from_file  # type: ignore
            except ImportError:                                  # pragma: no cover
                from src.core.game.kotor_loader import load_model_from_file  # type: ignore
            return load_model_from_file(path)

        if resref:
            return _cb.load_game_skeleton_source(resref, game=game)
        return None

    def _viewport_external_skeleton_model(self) -> Optional[Any]:
        """Return the reference skeleton currently visible in the viewport."""
        viewport = getattr(self, "viewport", None)
        renderer = getattr(viewport, "_renderer", None)
        model = getattr(renderer, "_ext_skeleton", None)
        return model if model is not None else None

    def _option_from_loaded_skeleton_template(
        self,
        model: Any,
        *,
        fallback_key: str = "",
    ) -> dict[str, Any]:
        """Build an apply option from an already-loaded KOTOR base skeleton."""
        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        requested = (
            str(getattr(model, "_gr_requested_resref", "") or "")
            or str(getattr(model, "_gr_target_resref", "") or "")
            or str(getattr(model, "name", "") or "")
            or "selected_kotor_base"
        ).strip().lower()
        source = (
            str(getattr(model, "_gr_source_resref", "") or "")
            or str(getattr(model, "_gr_variant_source_resref", "") or "")
            or requested
        ).strip().lower()
        key = str(fallback_key or f"loaded:{str(game).lower()}:{requested}").strip()
        option = {
            "key": key,
            "source": "loaded",
            "game": str(game or "K1"),
            "part": "body",
            "name": requested,
            "resref": requested,
            "source_resref": source,
            "path": f"loaded:{source}.mdl",
            "description": "Already loaded KOTOR base skeleton from the viewport.",
        }
        if key:
            self._skeleton_template_options_by_key[key] = option
            self._selected_skeleton_template_key = key
        return option

    def _typed_skeleton_template_option(self, key: str) -> Optional[dict[str, Any]]:
        """Build a temporary installed-model option from a typed resref."""
        raw = str(key or "")
        if not raw.startswith("typed:"):
            return None
        resref = raw[6:].strip().lower()
        clean = "".join(ch for ch in resref if ch.isalnum() or ch == "_")
        if not clean or clean != resref or len(resref) > 16:
            return None
        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        source_resref = resref
        warnings: list[str] = []
        try:
            try:
                from core.animation_retargeting import skeleton_template_picker as _picker
            except ImportError:                              # pragma: no cover
                from src.core.animation_retargeting import skeleton_template_picker as _picker  # type: ignore
            installed = [
                str(row.get("resref") or "").strip().lower()
                for row in self._installed_skeleton_template_rows(str(game or "K1"))
            ]
            source_resref = _picker.resolve_model_variant_source_resref(
                resref,
                installed,
            )
            if source_resref != resref:
                warnings.append(
                    f"'{resref}' is treated as an appearance/texture variant; "
                    f"loading base MDL '{source_resref}' for the KOTOR node DAG."
                )
        except Exception:
            log.debug("Could not resolve typed skeleton variant", exc_info=True)
        option = {
            "key": f"game:{str(game).lower()}:{resref}:typed",
            "source": "installation",
            "game": str(game or "K1"),
            "part": "body",
            "name": resref,
            "resref": resref,
            "source_resref": source_resref,
            "path": f"installation:{source_resref}.mdl",
            "description": "Typed KOTOR model resref from the configured installation.",
            "warnings": warnings,
            "metadata": {
                "requested_resref": resref,
                "source_resref": source_resref,
                "variant_resolution": (
                    "npc_numbered_variant_base"
                    if source_resref != resref else
                    "exact"
                ),
            },
        }
        self._skeleton_template_options_by_key[str(option["key"])] = option
        if not any(
            str(self._option_field(opt, "key", "")) == option["key"]
            for opt in self._skeleton_template_options
        ):
            self._skeleton_template_options.insert(0, option)
        return option

    def _refresh_skeleton_template_options(self) -> None:
        """Refresh the body-rig template picker for the current game."""
        try:
            from core.animation_retargeting import skeleton_template_picker as _picker
        except ImportError:                                 # pragma: no cover
            try:
                from src.core.animation_retargeting import skeleton_template_picker as _picker  # type: ignore
            except Exception as exc:
                log.exception("Could not import skeleton_template_picker")
                if hasattr(self.inspector, "set_skeleton_template_status"):
                    self.inspector.set_skeleton_template_status(
                        f"Skeleton picker unavailable: {exc}",
                        kind="error",
                    )
                return

        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        game_models = self._installed_skeleton_template_rows(game)
        part = (
            "creature" if self._is_scene_mode("creature") else
            "head" if self._is_scene_mode("head") else
            "body"
        )
        result = _picker.list_skeleton_templates(
            game=game,
            part=part,
            game_models=game_models,
            max_results=8000,
        )
        options = list(getattr(result, "options", []) or [])
        self._skeleton_template_options = options
        self._skeleton_template_options_by_key = {
            str(self._option_field(option, "key", "")): option
            for option in options
            if str(self._option_field(option, "key", ""))
        }

        if hasattr(self.inspector, "set_skeleton_template_status"):
            kind = "ok" if options else "warning"
            self.inspector.set_skeleton_template_status(
                getattr(result, "message", "") or (
                    f"{len(options)} skeleton template(s) available."
                    if options else "No KOTOR skeleton templates found."
                ),
                kind=kind,
            )

        if hasattr(self.inspector, "set_skeleton_template_options"):
            try:
                self.inspector.set_skeleton_template_options(options)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_skeleton_template_options failed")

    @QtCore.Slot()
    def _on_browse_skeleton_template_requested(self) -> None:
        """Let the user select a specific KOTOR MDL as the base skeleton."""
        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        initial_dir = ""
        try:
            try:
                from core.characters import character_builder as _cb
            except ImportError:                              # pragma: no cover
                from src.core.characters import character_builder as _cb  # type: ignore
            initial_dir = str(_cb._detect_game_dir(game) or "")
        except Exception:
            initial_dir = ""

        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose KOTOR base skeleton MDL",
            initial_dir,
            "KOTOR model (*.mdl);;All files (*.*)",
        )
        if not path:
            return

        abs_path = os.path.abspath(path)
        name = Path(abs_path).stem.lower()
        key = f"file:{os.path.normcase(abs_path)}"
        option = {
            "key": key,
            "source": "file",
            "game": str(game or "K1"),
            "part": "body",
            "name": name,
            "resref": name,
            "source_resref": name,
            "path": abs_path,
            "description": "User-selected KOTOR MDL base skeleton.",
            "warnings": [],
        }

        self._skeleton_template_options_by_key[key] = option
        self._skeleton_template_options = [
            opt for opt in self._skeleton_template_options
            if str(self._option_field(opt, "key", "")) != key
        ]
        self._skeleton_template_options.insert(0, option)
        if hasattr(self.inspector, "set_skeleton_template_options"):
            self.inspector.set_skeleton_template_options(
                self._skeleton_template_options
            )
        if hasattr(self.inspector, "set_selected_skeleton_template_key"):
            self.inspector.set_selected_skeleton_template_key(key, emit=False)
        self._on_skeleton_template_selected(key)

    @QtCore.Slot(str)
    def _on_skeleton_template_selected(self, key: str) -> None:
        """Preview the selected template skeleton over the loaded mesh."""
        self._selected_skeleton_template_key = str(key or "")
        option = self._skeleton_template_options_by_key.get(
            self._selected_skeleton_template_key
        )
        if option is None:
            option = self._typed_skeleton_template_option(
                self._selected_skeleton_template_key
            )
            if option is not None:
                self._selected_skeleton_template_key = str(
                    self._option_field(option, "key", "")
                )
        if option is None:
            return

        label = self._skeleton_template_status_label(option) or str(
            self._option_field(option, "name", "") or key
        )

        template_model = self._load_skeleton_template_model(option)
        if template_model is None:
            self._selected_skeleton_template_model = None
            if hasattr(self.inspector, "set_skeleton_template_status"):
                self.inspector.set_skeleton_template_status(
                    f"Could not load {label} from the configured KOTOR install. "
                    "Set the game directory or choose an installed MDL path.",
                    kind="warning",
                )
            return
        self._selected_skeleton_template_model = template_model
        requested_resref = str(self._option_field(option, "resref", "") or "")
        source_resref = str(self._option_field(option, "source_resref", "") or requested_resref)
        if requested_resref:
            setattr(template_model, "_gr_requested_resref", requested_resref)
            setattr(template_model, "_gr_target_resref", requested_resref)
        if source_resref and source_resref != requested_resref:
            setattr(template_model, "_gr_source_resref", source_resref)
            setattr(template_model, "_gr_variant_source_resref", source_resref)
            setattr(template_model, "_gr_variant_resolution", "npc_numbered_variant_base")
        try:
            from core.characters import character_builder as _cb
        except ImportError:                                 # pragma: no cover
            from src.core.characters import character_builder as _cb  # type: ignore
        try:
            _cb.log_character_builder_event(
                "ui.skeleton_template_selected",
                key=self._selected_skeleton_template_key,
                option={
                    "source": self._option_field(option, "source", ""),
                    "game": self._option_field(option, "game", ""),
                    "part": self._option_field(option, "part", ""),
                    "resref": self._option_field(option, "resref", ""),
                    "source_resref": self._option_field(option, "source_resref", ""),
                    "path": self._option_field(option, "path", ""),
                },
                template=_cb.summarize_model_for_character_builder(template_model),
            )
        except Exception:
            log.debug("Character Builder template selection diagnostic failed", exc_info=True)

        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_external_skeleton"):
            try:
                viewport.set_external_skeleton(template_model, fit_to_model=False)
            except Exception:                               # pragma: no cover
                log.exception("viewport.set_external_skeleton failed")

        if hasattr(self.inspector, "set_skeleton_template_status"):
            self.inspector.set_skeleton_template_status(
                f"Using {label} as the base skeleton. Imported meshes will fit to it.",
                kind="ok",
            )

    @QtCore.Slot()
    def _on_apply_skeleton_template_requested(self) -> None:
        """Attach the selected KOTOR template skeleton to the loaded body."""
        key = self._selected_skeleton_template_key
        if not key and hasattr(self.inspector, "selected_skeleton_template_key"):
            key = self.inspector.selected_skeleton_template_key()
        option = self._skeleton_template_options_by_key.get(str(key or ""))
        if option is None:
            option = self._typed_skeleton_template_option(str(key or ""))
            if option is not None:
                key = str(self._option_field(option, "key", ""))
                self._selected_skeleton_template_key = key
                self._selected_skeleton_template_model = None
        if option is None:
            loaded_template = (
                self._selected_skeleton_template_model
                or self._viewport_external_skeleton_model()
            )
            if loaded_template is not None:
                self._selected_skeleton_template_model = loaded_template
                option = self._option_from_loaded_skeleton_template(
                    loaded_template,
                    fallback_key=str(key or ""),
                )
        if option is None:
            message = "Choose a KOTOR skeleton template before applying."
            if hasattr(self.inspector, "set_skeleton_template_status"):
                self.inspector.set_skeleton_template_status(message, kind="warning")
            self.statusBar().showMessage(message, 5000)
            return

        try:
            from core.characters import character_builder as _cb
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.characters import character_builder as _cb    # type: ignore
            from src.core.geometry import model_data as _md           # type: ignore

        entry = self.scene.get(_md.PartSlot.HEADLESS_BODY)
        mesh_model = getattr(entry, "model", None) if entry is not None else None
        if mesh_model is None:
            message = "Load an OBJ, FBX, glTF, or MDL body before applying a skeleton."
            if hasattr(self.inspector, "set_skeleton_template_status"):
                self.inspector.set_skeleton_template_status(message, kind="warning")
            self.bottom_strip.set_validation(
                "warning", "NO_BODY_MESH", issues=[message]
            )
            self.statusBar().showMessage(message, 6000)
            return

        game = str(self._option_field(option, "game", "") or
                   getattr(self.scene, "game_version", "K1"))
        template_model = (
            self._selected_skeleton_template_model
            or self._viewport_external_skeleton_model()
        )
        if template_model is None:
            template_model = self._load_skeleton_template_model(option)
            self._selected_skeleton_template_model = template_model
        try:
            _cb.log_character_builder_event(
                "ui.apply_skeleton.requested",
                key=str(key or ""),
                option={
                    "source": self._option_field(option, "source", ""),
                    "game": self._option_field(option, "game", ""),
                    "part": self._option_field(option, "part", ""),
                    "resref": self._option_field(option, "resref", ""),
                    "source_resref": self._option_field(option, "source_resref", ""),
                    "path": self._option_field(option, "path", ""),
                },
                template=_cb.summarize_model_for_character_builder(template_model),
                body=_cb.summarize_model_for_character_builder(mesh_model),
                manual_fit={
                    "scale": float(self._manual_fit_scale or 1.0),
                    "rotation_degrees": tuple(float(v or 0.0) for v in self._manual_fit_rotation),
                    "translation": tuple(float(v or 0.0) for v in self._manual_fit_translation),
                },
            )
        except Exception:
            log.debug("Character Builder apply-skeleton diagnostic failed", exc_info=True)
        mesh_model = self._ensure_template_fitted_body_for_binding(
            mesh_model,
            entry,
            template_model,
            option,
            game,
        )
        self._start_rig_stage("skeleton")
        result = _cb.apply_template_rig(
            mesh_model,
            template_model,
            game=game,
            scale_mode="manual",
            scale_factor=1.0,
        )

        if not bool(result.get("ok")):
            message = str(result.get("message") or "Template skeleton apply failed.")
            self._fail_rig_stage("skeleton", message)
            if hasattr(self.inspector, "set_skeleton_template_status"):
                self.inspector.set_skeleton_template_status(message, kind="error")
            self.bottom_strip.set_validation(
                "error", "SKELETON_TEMPLATE", issues=[message]
            )
            self.statusBar().showMessage(message, 6000)
            return

        rigged_model = result.get("model")
        resref = getattr(entry, "resref", "") if entry is not None else ""
        source_path = getattr(entry, "source_path", "") if entry is not None else ""
        self.scene.assign(
            _md.PartSlot.HEADLESS_BODY,
            rigged_model,
            resref=resref,
            game_version=game,
            source_path=source_path,
        )
        self._body_guides = {}
        self._body_guide_history = None
        self._refresh_body_guide_undo_actions()

        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "load_model"):
            try:
                self._load_model_in_viewport_with_textures(
                    rigged_model,
                    source_path=source_path,
                    prompt=False,
                )
                if hasattr(viewport, "clear_external_skeleton"):
                    viewport.clear_external_skeleton()
                if hasattr(viewport, "clear_acurig_guides"):
                    viewport.clear_acurig_guides()
            except Exception:                               # pragma: no cover
                log.exception("Failed to refresh viewport after template apply")

        warnings = list(result.get("warnings") or [])
        message = str(result.get("message") or "Template skeleton applied.")
        self._complete_rig_stage(
            "skeleton",
            {
                "kind": "native_kotor_template",
                "template_key": str(key or ""),
                "template_resref": str(self._option_field(option, "resref", "") or ""),
                "game_version": str(game),
                "warnings": [str(value) for value in warnings],
            },
        )
        if hasattr(self.inspector, "set_skeleton_template_status"):
            self.inspector.set_skeleton_template_status(message, kind="ok")
        self._push_import_fit_report_to_inspector(rigged_model)
        self.bottom_strip.set_validation(
            "info",
            "SKELETON_TEMPLATE",
            issues=[message] + warnings,
        )
        self.statusBar().showMessage(message, 6000)
        self._update_title()
        self._schedule_live_validation("skeleton_template_applied")

    @QtCore.Slot()
    def _on_split_mesh_nodes_requested(self) -> None:
        """Split imported mesh islands before binding the KOTOR skeleton.

        P5-min (T2514): the button also handles SKINNED meshes now — over-
        palette skinned nodes are split anatomically with weight remap
        (T2512), using the selected base skeleton as the weight donor.  The
        two hard-fail paths surface as actionable dialogs, not tracebacks.
        """

        _wf = self._body_workflow_module()
        result = _wf.split_imported_mesh_nodes(
            self.scene,
            respect_skinned="split_with_weight_remap",
            reference_model=getattr(self, "_selected_skeleton_template_model", None),
        )
        dialog_copy = _split_failure_dialog_copy(result)
        if dialog_copy is not None:
            QtWidgets.QMessageBox.critical(self, dialog_copy[0], dialog_copy[1])
        message = str(result.get("message") or "Node Splitter finished.")
        ok = bool(result.get("ok"))
        kind = "ok" if ok and int(result.get("split_nodes", 0) or 0) else "info"
        if not ok:
            kind = "error"
        if hasattr(self.inspector, "set_node_splitter_status"):
            self.inspector.set_node_splitter_status(message, kind=kind)
        # Mirror to the skeleton-template status only for unskinned island
        # splits (its original purpose) — mirroring every result printed the
        # same line twice in the Build Skeleton box (T2514 manual-test finding).
        if (
            hasattr(self.inspector, "set_skeleton_template_status")
            and ok
            and int(result.get("source_nodes", 0) or 0) > 0
        ):
            self.inspector.set_skeleton_template_status(message, kind=kind)
        try:
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.geometry import model_data as _md  # type: ignore
        entry = self.scene.get(_md.PartSlot.HEADLESS_BODY)
        model = getattr(entry, "model", None) if entry is not None else None
        source_path = getattr(entry, "source_path", "") if entry is not None else ""
        viewport = getattr(self, "viewport", None)
        if model is not None and viewport is not None and hasattr(viewport, "load_model"):
            try:
                self._load_model_in_viewport_with_textures(
                    model,
                    source_path=source_path,
                    prompt=False,
                )
            except Exception:                               # pragma: no cover
                log.exception("Failed to refresh viewport after node split")
        if hasattr(self, "bottom_strip"):
            self.bottom_strip.set_validation(
                "info" if ok else "error",
                "NODE_SPLITTER",
                issues=[message],
            )
        self.statusBar().showMessage(message, 6000)
        self._schedule_live_validation("node_splitter")

    @QtCore.Slot()
    def _on_validate_requested(self) -> None:
        """Workflow Step 7 (Validate Scene) — M5 / T506.

        Runs :func:`headless_body_workflow.validate_for_export` and
        pushes the result into the inspector's validation tally + the
        bottom-strip banner.  Replaces the M2 synthetic-clean stub.
        """
        self._run_validation(reason="manual", update_status=True)

    def _schedule_live_validation(self, reason: str = "") -> None:
        """Debounce validation after scene mutations (M9 / T901)."""
        timer = getattr(self, "_live_validation_timer", None)
        if timer is None:
            return
        timer.setProperty("reason", reason or "scene_mutation")
        timer.start()

    @QtCore.Slot()
    def _run_live_validation(self) -> None:
        """Timer callback for live export-readiness validation."""
        timer = getattr(self, "_live_validation_timer", None)
        reason = str(timer.property("reason") if timer is not None else "live")
        self._run_validation(reason=reason or "live", update_status=False)

    def _run_validation(self, *, reason: str, update_status: bool) -> Any:
        """Run the workflow validation service and refresh UI surfaces."""
        try:
            _wf = self._workflow_module()
        except Exception as exc:                            # pragma: no cover
            log.exception("Could not import active Character Builder workflow")
            self.bottom_strip.set_validation(
                "error", "VALIDATE_UNAVAILABLE",
                issues=[f"Workflow service unavailable: {exc}"],
            )
            return None

        validator = (
            getattr(_wf, "validate_for_export_head", None)
            if self._is_scene_mode("head")
            else getattr(_wf, "validate_for_export", None)
        )
        if not callable(validator):
            self.bottom_strip.set_validation(
                "error",
                "VALIDATE_UNAVAILABLE",
                issues=["The active workflow does not provide an export validator."],
            )
            return None
        result = validator(self.scene, strict=True)
        self._last_validation_result = result

        # Push detailed tally + Export-button-enable state into the
        # inspector's validate page.
        if hasattr(self.inspector, "set_validate_for_export_result"):
            try:
                self.inspector.set_validate_for_export_result(result)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_validate_for_export_result failed")

        # Banner severity follows the workflow's code.
        if result.code == "blocked":
            severity = "error"
        elif result.code == "warnings_only":
            severity = "warning"
        else:
            severity = "clean"

        self.bottom_strip.set_validation(
            severity, result.code.upper(),
            issues=result.issues,
        )
        if update_status:
            self.statusBar().showMessage(result.message, 6000)
        else:
            log.debug("Live validation refreshed after %s: %s", reason, result.code)
        return result

    def _format_validation_issue_lines(
        self,
        issues: list[Any],
        limit: int = 12,
    ) -> list[str]:
        """Build concise issue lines for modal details panes."""
        lines: list[str] = []
        for issue in list(issues or [])[:max(0, int(limit))]:
            if isinstance(issue, str):
                lines.append(issue)
                continue
            sev = _issue_field(issue, "severity").upper() or "ISSUE"
            code = _issue_field(issue, "code") or "VALIDATION"
            node = _issue_field(issue, "node")
            message = _issue_field(issue, "message")
            target = f" [{node}]" if node else ""
            lines.append(f"{sev} {code}{target}: {message}")
        remaining = len(list(issues or [])) - len(lines)
        if remaining > 0:
            lines.append(f"... plus {remaining} more issue(s).")
        return lines

    def _confirm_pre_export_validation(self) -> tuple[bool, bool]:
        """T904 gate: block errors; ask before exporting with warnings."""
        result = self._run_validation(reason="pre_export_gate", update_status=False)
        if result is None:
            return False, False

        if (
            int(getattr(result, "error_count", 0) or 0) > 0
            or not bool(getattr(result, "ok", False))
        ):
            message = getattr(result, "message", "Export blocked by validation.")
            if hasattr(self.inspector, "set_export_status"):
                try:
                    self.inspector.set_export_status(message, kind="error")
                except Exception:                           # pragma: no cover
                    log.exception("inspector.set_export_status failed")
            self.statusBar().showMessage(message, 6000)
            return False, False

        warnings = int(getattr(result, "warning_count", 0) or 0)
        if warnings <= 0:
            return True, False

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Export Warnings")
        box.setText(f"Validation found {warnings} warning(s).")
        box.setInformativeText(
            "Warnings may still produce a usable MDL, but they are worth "
            "reviewing before testing in KOTOR."
        )
        details = "\n".join(
            self._format_validation_issue_lines(
                list(getattr(result, "issues", []) or [])
            )
        )
        if details:
            box.setDetailedText(details)
        export_btn = box.addButton(
            "Export anyway",
            QtWidgets.QMessageBox.AcceptRole,
        )
        box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(export_btn)
        box.exec()
        if box.clickedButton() is not export_btn:
            self.statusBar().showMessage(
                "Export cancelled; warnings left for review.",
                5000,
            )
            return False, False
        return True, True

    @QtCore.Slot()
    def _on_validation_banner_clicked(self) -> None:
        """Open the full validation report from the bottom-strip banner."""
        issues = self.bottom_strip.issues()
        if not issues:
            result = self._last_validation_result or self._run_validation(
                reason="banner_clicked",
                update_status=False,
            )
            issues = (
                list(getattr(result, "issues", []) or [])
                if result is not None else
                []
            )

        dialog = QtValidationReportDialog(issues, self)
        dialog.jumpRequested.connect(self._on_validation_report_jump_requested)
        dialog.exec()

    @QtCore.Slot(str)
    def _on_validation_report_jump_requested(self, node_name: str) -> None:
        """Select the validation issue target node in the viewport."""
        node_name = str(node_name or "").strip()
        if not node_name:
            return
        viewport = getattr(self, "viewport", None)
        node = self._find_viewport_node(node_name)
        if viewport is not None and node is not None and hasattr(viewport, "set_selected_node"):
            try:
                viewport.set_selected_node(node)
                self.statusBar().showMessage(
                    f"Selected validation target: {node_name}",
                    4000,
                )
                return
            except Exception:                               # pragma: no cover
                log.exception("viewport.set_selected_node failed for %s", node_name)
        self.statusBar().showMessage(f"Validation target not visible: {node_name}", 5000)

    def _find_viewport_node(self, node_name: str) -> Any:
        """Best-effort node lookup against the currently previewed model."""
        needle = str(node_name or "").strip().lower()
        if not needle:
            return None
        model = getattr(getattr(self, "viewport", None), "model", None)
        if model is None:
            return None
        try:
            nodes = model.all_nodes() if hasattr(model, "all_nodes") else []
        except Exception:                                  # pragma: no cover
            nodes = []
        for node in list(nodes or []):
            if str(getattr(node, "name", "") or "").lower() == needle:
                return node
        return None

    @QtCore.Slot()
    def _on_check_model_requested(self) -> None:
        """Workflow Step 2 (Check Model) — M5 / T502.

        Runs :func:`headless_body_workflow.check_model` and projects
        the result into the bottom-strip validation banner.  Severity
        colour and summary text are computed inside the service so the
        Qt code stays a thin adapter.
        """
        if self._is_scene_mode("supermodel"):
            try:
                from core.workflow import composite_workflow as _cw
            except ImportError:                             # pragma: no cover
                try:
                    from src.core.workflow import composite_workflow as _cw  # type: ignore
                except Exception as exc:
                    log.exception("Could not import composite_workflow")
                    self.bottom_strip.set_validation(
                        "error", "CHECK_UNAVAILABLE",
                        issues=[f"Composite workflow unavailable: {exc}"],
                    )
                    return

            result = _cw.check_composite(self.scene, strict=False)
            self.bottom_strip.set_validation(
                result.banner_key,
                result.summary,
                issues=result.issues,
            )
            if hasattr(self.inspector, "set_check_model_result"):
                try:
                    self.inspector.set_check_model_result(result)
                except Exception:                           # pragma: no cover
                    log.exception("inspector.set_check_model_result failed")
            self.statusBar().showMessage(result.message, 6000)
            return

        try:
            _wf = self._workflow_module()
        except Exception as exc:                            # pragma: no cover
            log.exception("Could not import active Character Builder workflow")
            self.bottom_strip.set_validation(
                "error", "CHECK_UNAVAILABLE",
                issues=[f"Workflow service unavailable: {exc}"],
            )
            return

        checker = (
            getattr(_wf, "check_head", None)
            if self._is_scene_mode("head")
            else getattr(_wf, "check_model", None)
        )
        if not callable(checker):
            self.bottom_strip.set_validation(
                "error",
                "CHECK_UNAVAILABLE",
                issues=["The active workflow does not provide a model check."],
            )
            return
        result = checker(self.scene)
        # Store full issues list so a future banner-click can drill into
        # the report (UX hook documented in qt_bottom_strip.py).
        self.bottom_strip.set_validation(
            result.banner_key,
            result.summary,
            issues=result.issues,
        )
        # Push the issue table into the inspector so the user can
        # triage findings without leaving the workflow.
        if hasattr(self.inspector, "set_check_model_result"):
            try:
                self.inspector.set_check_model_result(result)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_check_model_result failed")
        # Brief status-bar tail with the per-severity tally.
        if result.error_count or result.warning_count or result.info_count:
            self.statusBar().showMessage(
                f"Check Model: {result.error_count} error(s), "
                f"{result.warning_count} warning(s), "
                f"{result.info_count} info ({len(result.codes)} unique code(s))",
                6000,
            )
        else:
            self.statusBar().showMessage("Check Model: all good", 4000)

    @QtCore.Slot()
    def _on_export_requested(self) -> None:
        """Workflow Step 7 (Export…) — M5 / T506.

        Opens the modal :class:`QtExportDialog`; on Accept, runs the
        validation gate again, then dispatches
        :func:`headless_body_workflow.export_scene` with the user-
        selected formats / output directory / sidecar option.
        Per-format outcomes are surfaced in the inspector status line
        and the bottom-strip banner.

        Replaces the M2 "Export — implementation pending (M10)" status
        stub with the real workflow.  Per-format MDL/MDX/FBX/glTF/OBJ
        binary writers are routed through the per-mode workflow service.
        Supermodel mode uses the composite exporter so FBX/glTF contain
        the head parented under the body's headhook.
        """
        _wf = self._workflow_module()
        try:
            from src.gui.qt_lib.dialogs.qt_export_dialog import QtExportDialog
        except Exception:                                   # pragma: no cover
            try:
                from src.gui.qt_lib.dialogs.qt_export_dialog import QtExportDialog
            except Exception as exc:
                log.exception("Could not import QtExportDialog")
                self.bottom_strip.set_validation(
                    "error", "EXPORT_UNAVAILABLE",
                    issues=[f"Export dialog unavailable: {exc}"],
                )
                return

        # Derive a sensible default resref from the active primary slot for the
        # dialog's read-only hint label.
        md = None
        try:
            from src.core.geometry import model_data as md  # noqa: WPS433 - lazy on purpose
        except Exception:                                   # pragma: no cover
            try:
                from core.geometry import model_data as md  # type: ignore  # noqa: WPS433
            except Exception:
                md = None
        initial_resref = ""
        if md is not None:
            slot = (
                md.PartSlot.HEAD_SHELL
                if self._is_scene_mode("head")
                else md.PartSlot.HEADLESS_BODY
            )
            entry = self.scene.get(slot)
            if entry is not None:
                initial_resref = (entry.resref or "").lower() or ""

        default_formats = getattr(_wf, "default_export_formats_for_mode", None)
        initial_formats = (
            default_formats(self.scene)
            if callable(default_formats)
            else ("kotor",)
        )
        dlg = QtExportDialog(
            self,
            default_dir=getattr(self, "_last_export_dir", ""),
            initial_resref=initial_resref,
            initial_formats=initial_formats,
            initial_write_sidecar=True,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage("Export cancelled.", 3000)
            return

        formats = dlg.selected_formats()
        out_dir = dlg.output_dir()
        write_sidecar = dlg.write_sidecar()
        profile_getter = getattr(dlg, "fbx_compatibility_profile", None)
        fbx_compatibility_profile = (
            str(profile_getter() or "standard") if callable(profile_getter) else "standard"
        )
        tex_cache_getter = getattr(getattr(self, "viewport", None), "tex_cache", None)
        export_tex_cache = tex_cache_getter() if callable(tex_cache_getter) else tex_cache_getter
        fbx_animation_names = None
        animation_resource_manager = None
        if "fbx" in formats and md is not None:
            body_entry = self.scene.get(md.PartSlot.HEADLESS_BODY)
            body_model = getattr(body_entry, "model", None) if body_entry is not None else None
            head_entry = self.scene.get(md.PartSlot.HEAD_SHELL)
            head_model = getattr(head_entry, "model", None) if head_entry is not None else None
            game = str(getattr(self.scene, "game_version", "") or "K1").upper()
            animation_resource_manager = self._ensure_game_resource_manager(game)
            try:
                self._sync_motion_controls_to_scene(_wf)
            except Exception:
                log.debug("Could not sync motion controls before FBX animation selection", exc_info=True)
            supermodel = str(getattr(body_model, "supermodel", "") or "").strip()
            base_skeleton_model = None
            if body_model is not None and supermodel.lower() not in {"", "null", "none", "****"}:
                if animation_resource_manager is not None:
                    try:
                        loader = getattr(animation_resource_manager, "load_model_strict", None)
                        if not callable(loader):
                            loader = animation_resource_manager.load_model
                        base_skeleton_model = loader(supermodel, game)
                        setattr(body_model, "_gr_fbx_base_skeleton_model", base_skeleton_model)
                    except Exception:
                        log.warning("Could not resolve Character Builder FBX supermodel %s", supermodel, exc_info=True)
            if body_model is not None:
                try:
                    from src.core.animation.fbx_animation_selection import list_fbx_animation_sets
                    from src.gui.qt_lib.dialogs.qt_fbx_animation_selection_dialog import (
                        QtFbxAnimationSelectionDialog,
                    )

                    rows = list_fbx_animation_sets(
                        body_model,
                        game=game,
                        resource_manager=animation_resource_manager,
                        base_skeleton_model=base_skeleton_model,
                        supplemental_models=((head_model,) if head_model is not None else ()),
                    )
                    initial_names = [
                        str(getattr(anim, "name", "") or "")
                        for anim in list(getattr(body_model, "animations", []) or [])
                        if str(getattr(anim, "name", "") or "")
                    ]
                    current_animation = getattr(
                        getattr(self, "_animation_engine", None),
                        "current_animation",
                        None,
                    )
                    current_name = str(getattr(current_animation, "name", "") or "")
                    if current_name and current_name.lower() not in {
                        name.lower() for name in initial_names
                    }:
                        initial_names.append(current_name)
                    animation_dialog = QtFbxAnimationSelectionDialog(
                        rows,
                        self,
                        profile=fbx_compatibility_profile,
                        initial_selected_names=tuple(initial_names),
                        current_animation_name=current_name,
                    )
                    if animation_dialog.exec() != QtWidgets.QDialog.Accepted:
                        self.statusBar().showMessage("Export cancelled.", 3000)
                        return
                    fbx_animation_names = tuple(
                        animation_dialog.selected_animation_names()
                    )
                except Exception as exc:
                    log.exception("Could not prepare Character Builder FBX animation selection")
                    self.bottom_strip.set_validation(
                        "error",
                        "FBX_ANIMATION_SELECTION",
                        issues=[str(exc)],
                    )
                    return
        # Remember the chosen folder for the next invocation.
        self._last_export_dir = out_dir
        can_export, skip_validation = self._confirm_pre_export_validation()
        if not can_export:
            return

        self._start_rig_stage("export")
        if self._is_scene_mode("supermodel"):
            try:
                from core.workflow import composite_workflow as _cw  # noqa: WPS433
            except Exception:                               # pragma: no cover
                from src.core.workflow import composite_workflow as _cw  # type: ignore
            result = _cw.export_composite_scene(
                self.scene,
                formats=formats,
                out_dir=out_dir,
                write_sidecar=write_sidecar,
                skip_validation=skip_validation,
                fbx_compatibility_profile=fbx_compatibility_profile,
                tex_cache=export_tex_cache,
                fbx_animation_names=fbx_animation_names,
                animation_resource_manager=animation_resource_manager,
            )
        elif self._is_scene_mode("head"):
            result = _wf.export_head_scene(
                self.scene,
                formats=formats,
                out_dir=out_dir,
                write_sidecar=write_sidecar,
                skip_validation=skip_validation,
            )
        else:
            result = _wf.export_scene(
                self.scene,
                formats=formats,
                out_dir=out_dir,
                write_sidecar=write_sidecar,
                skip_validation=skip_validation,
                fbx_compatibility_profile=fbx_compatibility_profile,
                tex_cache=export_tex_cache,
                fbx_animation_names=fbx_animation_names,
                animation_resource_manager=animation_resource_manager,
            )

        export_artifact = {
            "kind": "character_export",
            "output_dir": str(out_dir),
            "requested_formats": [str(value) for value in formats],
            "write_sidecar": bool(write_sidecar),
            "fbx_compatibility_profile": fbx_compatibility_profile,
            "fbx_animation_names": (
                None if fbx_animation_names is None else list(fbx_animation_names)
            ),
            "sidecar_path": str(result.sidecar_path or ""),
            "formats": [
                {
                    "label": str(getattr(row, "label", "") or ""),
                    "ok": bool(getattr(row, "ok", False)),
                    "message": str(getattr(row, "message", "") or ""),
                }
                for row in list(result.formats or [])
            ],
        }
        head_binary_pending = (
            self._is_scene_mode("head")
            and any(
                str(getattr(row, "code", "") or "") == "not_implemented"
                for row in list(result.formats or [])
            )
        )
        if result.ok and not head_binary_pending:
            self._complete_rig_stage("export", export_artifact)
        else:
            self._fail_rig_stage("export", result.message)

        # Inspector status line + bottom-strip banner.
        if hasattr(self.inspector, "set_export_status"):
            try:
                self.inspector.set_export_status(
                    result.message,
                    kind=(
                        "warning"
                        if head_binary_pending
                        else "ok" if result.ok else "error"
                    ),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_export_status failed")

        # Banner severity: blocked / no_body / all_failed → error;
        # any successful row (sidecar OK or a future format OK) → info.
        if head_binary_pending:
            severity = "warning"
            tag = "HEAD_BINARY_PENDING"
        elif not result.ok:
            severity = "error"
            tag = (result.code or "export").upper()
        else:
            severity = "info"
            tag = "EXPORTED"

        # Compose a multi-line issue list summarising per-format outcomes.
        issues: list = []
        for row in result.formats:
            issues.append(f"{row.label}: {row.message}")
        if result.sidecar_path:
            issues.append(f"Sidecar JSON: {result.sidecar_path}")

        self.bottom_strip.set_validation(severity, tag, issues=issues)
        self.statusBar().showMessage(result.message, 6000)

        # ── Creature Package Generation ────────────────────────────────
        # When the MDL export succeeds, automatically generate the full
        # creature installation package (appearance.2da + UTC + spawn script)
        # so the creature can be spawned in-game as an enemy.
        if (
            result.ok
            and not self._is_scene_mode("head")
            and any("mdl" in f.lower() for f in formats)
        ):
            try:
                self._generate_creature_package(out_dir, result)
            except Exception as exc:
                log.exception("Creature package generation failed")
                self.statusBar().showMessage(
                    f"Creature package generation failed: {exc}", 5000)

    def _generate_creature_package(self, out_dir: str, export_result) -> None:
        """Generate appearance.2da, UTC, spawn script, and readme alongside
        the exported MDL.  This wraps the creature_package_builder module.
        """
        try:
            from src.core.characters.creature_package_builder import (
                CreatureSpec, build_creature_package,
            )
        except ImportError:
            from core.characters.creature_package_builder import (
                CreatureSpec, build_creature_package,
            )

        # Determine the resref from the scene
        md = None
        try:
            from src.core.geometry import model_data as md
        except Exception:
            try:
                from core.geometry import model_data as md  # type: ignore
            except Exception:
                md = None

        resref = ""
        if md is not None:
            entry = self.scene.get(md.PartSlot.HEADLESS_BODY)
            if entry is not None:
                resref = (entry.resref or "").lower()

        if not resref:
            return  # Can't generate without a resref

        # Find the exported MDL path
        mdl_path = Path(out_dir) / f"{resref}.mdl"
        if not mdl_path.exists():
            return

        # Build creature spec
        display_name = resref.replace("c_", "").replace("_", " ").title()
        spec = CreatureSpec(
            resref=resref,
            display_name=display_name,
            app_type="S",  # creature animations (crun, cwalk)
            faction_id=1,  # hostile
            level=5,
            max_hp=45,
            str_stat=14,
            dex_stat=10,
            con_stat=12,
            game_version="K2",
        )

        # Try to read the game's appearance.2da for row appending
        existing_2da = None
        try:
            if hasattr(self, "_resource_manager") and self._resource_manager:
                for game_key in ("K2", "K1"):
                    data = self._resource_manager.get_resource(
                        "appearance", "2da", game_key)
                    if data:
                        existing_2da = data
                        break
        except Exception:
            pass

        pkg_result = build_creature_package(
            spec,
            out_dir,
            mdl_path=mdl_path,
            existing_appearance_2da=existing_2da,
        )

        msg = (f"Creature package generated: appearance row "
               f"{pkg_result.appearance_row}, {len(pkg_result.files_written)} files")
        self.statusBar().showMessage(msg, 5000)
        log.info(msg)

    # ── M5 / T503 — Body-rig step slots ──────────────────────────────────

    @QtCore.Slot()
    def _on_place_body_guides_requested(self) -> None:
        """Place AcuRig humanoid guides on the loaded body model.

        Wraps :func:`headless_body_workflow.place_body_guides` and pushes
        the result into the inspector status label, the bottom-strip
        banner, and (on success) refreshes the viewport so the joint-dot
        HUD picks up the newly-placed guides.  The created
        :class:`AcuRig` instance is kept on ``self._acurig`` so the
        subsequent *Generate Skeleton* click reuses it (preserving any
        user-locked guide overrides).
        """
        if not self._require_legacy_acurig_enabled("Place Body Guides"):
            return

        _wf = self._body_workflow_module()

        self._start_rig_stage("body_landmarks")
        result = _wf.place_body_guides(
            self.scene,
            snap_to_bones=True,
            acurig=self._acurig,
        )

        # Inspector status label — colour-coded per kind.
        if hasattr(self.inspector, "set_body_rig_status"):
            try:
                self.inspector.set_body_rig_status(
                    result.message,
                    kind=("ok" if result.ok else "error"),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_body_rig_status failed")

        if not result.ok:
            self._fail_rig_stage("body_landmarks", result.message)
            self.bottom_strip.set_validation(
                "error", "PLACE_GUIDES",
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            return

        # Persist the AcuRig instance for the next click.
        self._acurig = result.acurig
        self._body_guides = dict(result.guides or {})
        self._body_guide_history = None
        self._refresh_body_guide_undo_actions()
        self._complete_rig_stage(
            "body_landmarks",
            self._body_landmark_artifact("place_body_guides"),
        )

        # Refresh viewport joint-dot overlay by re-loading the body.
        try:
            body = _wf._get_body_model(self.scene)
            if (body is not None
                    and hasattr(self, "viewport")
                    and hasattr(self.viewport, "load_model")):
                self.viewport.load_model(body)
                if hasattr(self.viewport, "set_acurig_guides"):
                    self.viewport.set_acurig_guides(self._body_guides)
        except Exception:                                    # pragma: no cover
            log.exception("Failed to refresh viewport after place_body_guides")

        self.bottom_strip.set_validation(
            "info", "GUIDES_PLACED",
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)

    @QtCore.Slot()
    def _on_generate_skeleton_requested(self) -> None:
        """Build the skeleton + heat-map weights on the body model.

        Wraps :func:`headless_body_workflow.generate_skeleton`, forwards
        the cached :class:`AcuRig` instance (so user-edited guides are
        respected), pushes status into the inspector + bottom strip, and
        refreshes the viewport with the freshly-rigged model on success.
        """
        if not self._require_legacy_acurig_enabled("Create New Skeleton"):
            return

        _wf = self._body_workflow_module()

        self._start_rig_stage("skeleton")
        result = _wf.generate_skeleton(
            self.scene,
            acurig=self._acurig,
            smooth_iterations=2,
        )

        if hasattr(self.inspector, "set_body_rig_status"):
            try:
                self.inspector.set_body_rig_status(
                    result.message,
                    kind=("ok" if result.ok else "error"),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_body_rig_status failed")

        if not result.ok:
            self._fail_rig_stage("skeleton", result.message)
            severity_code = (result.code or "skeleton").upper()
            self.bottom_strip.set_validation(
                "error", severity_code,
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            return

        # Push the rigged body back into the viewport.
        try:
            body = _wf._get_body_model(self.scene)
            if (body is not None
                    and hasattr(self, "viewport")
                    and hasattr(self.viewport, "load_model")):
                self.viewport.load_model(body)
                if hasattr(self.viewport, "set_acurig_guides"):
                    guides = (
                        self._acurig.get_all_guides()
                        if self._acurig is not None
                        and hasattr(self._acurig, "get_all_guides")
                        else self._body_guides
                    )
                    self._body_guides = dict(guides or {})
                    self.viewport.set_acurig_guides(self._body_guides)
        except Exception:                                    # pragma: no cover
            log.exception("Failed to refresh viewport after generate_skeleton")

        # Mark the scene dirty so File → Save offers to persist.
        self._complete_rig_stage(
            "skeleton",
            {
                "kind": "legacy_acurig_generated",
                "bone_count": int(result.bone_count),
                "guide_count": len(self._body_guides),
            },
        )
        self._complete_rig_stage(
            "weights",
            {
                "kind": "legacy_acurig_auto_skin",
                "vertices_skinned": int(result.vertices_skinned),
                "weighting_method": str(result.weighting_method or ""),
            },
        )
        try:
            self.scene.dirty = True
        except Exception:                                    # pragma: no cover
            pass

        self.bottom_strip.set_validation(
            "info", "SKELETON_GENERATED",
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)
        self._update_title()
        self._schedule_live_validation("skeleton_generated")

    # ── M5 / T504 — Hand-rig step slots ──────────────────────────────────

    @QtCore.Slot()
    def _on_place_hand_guides_requested(self) -> None:
        """Refresh AcuRig hand-subset guides and sync the mask checkboxes.

        Wraps :func:`headless_body_workflow.place_hand_guides` and
        pushes the result into the inspector status label + bottom-strip
        banner.  The :class:`AcuRig` instance is cached on
        ``self._acurig`` (shared with T503) so subsequent mask toggles
        and the next body-rig pass keep working on the same instance.
        """
        if not self._require_legacy_acurig_enabled("Rebuild Hand Guides"):
            return

        _wf = self._body_workflow_module()

        self._start_rig_stage("fingers")
        result = _wf.place_hand_guides(
            self.scene,
            acurig=self._acurig,
            snap_to_bones=True,
        )

        if hasattr(self.inspector, "set_hand_rig_status"):
            try:
                self.inspector.set_hand_rig_status(
                    result.message,
                    kind=("ok" if result.ok else "error"),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_hand_rig_status failed")

        if not result.ok:
            self._fail_rig_stage("fingers", result.message)
            self.bottom_strip.set_validation(
                "error", (result.code or "hand_guides").upper(),
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            return

        # Persist the AcuRig instance so subsequent mask toggles share it.
        self._acurig = result.acurig
        self._complete_rig_stage(
            "fingers",
            self._finger_artifact(
                "place_hand_guides",
                guides=dict(result.guides or {}),
                masked_bones=list(result.masked_bones or []),
            ),
        )

        # Push the current mask state into the checkbox column so the UI
        # reflects whatever AcuRig already had set.
        if hasattr(self.inspector, "set_hand_masked_bones"):
            try:
                self.inspector.set_hand_masked_bones(result.masked_bones)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_hand_masked_bones failed")

        # Refresh viewport joint-dot overlay.
        try:
            body = _wf._get_body_model(self.scene)
            if (body is not None
                    and hasattr(self, "viewport")
                    and hasattr(self.viewport, "load_model")):
                self.viewport.load_model(body)
        except Exception:                                    # pragma: no cover
            log.exception("Failed to refresh viewport after place_hand_guides")

        self.bottom_strip.set_validation(
            "info", "HAND_GUIDES",
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)

    @QtCore.Slot(str, bool)
    def _on_hand_mask_changed(self, bone: str, checked: bool) -> None:
        """One of the per-bone mask checkboxes was toggled.

        Recomputes the full masked-bone set from the current checkbox
        state and forwards it to
        :func:`headless_body_workflow.apply_hand_masks` so AcuRig's
        :class:`BoneMask` mirrors the UI.
        """
        if not self._require_legacy_acurig_enabled("Hand weight mask edits"):
            return

        _wf = self._body_workflow_module()

        if self._acurig is None:
            # User toggled a checkbox before clicking *Place Hand Guides*.
            # Surface a friendly status instead of silently failing.
            if hasattr(self.inspector, "set_hand_rig_status"):
                self.inspector.set_hand_rig_status(
                    "Click Place Hand Guides first.",
                    kind="warning",
                )
            return

        # Recover the *full* set of intended-masked bones from the
        # current checkbox state, not just the single bone that
        # triggered the signal — keeps AcuRig in sync even if multiple
        # signals fire in quick succession.
        checkboxes = getattr(self.inspector, "_hand_mask_checkboxes", {}) or {}
        masked_now: list = [
            name for name, cb in checkboxes.items() if cb.isChecked()
        ]
        # Override with the freshly-toggled state in case the checkbox
        # widget hasn't latched yet (defensive).
        if checked and bone not in masked_now:
            masked_now.append(bone)
        elif (not checked) and bone in masked_now:
            masked_now = [b for b in masked_now if b != bone]

        result = _wf.apply_hand_masks(
            self.scene,
            acurig=self._acurig,
            masked_bones=masked_now,
        )

        if hasattr(self.inspector, "set_hand_rig_status"):
            try:
                self.inspector.set_hand_rig_status(
                    result.message,
                    kind=("ok" if result.ok else "warning"),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_hand_rig_status failed")

        if result.ok:
            self._complete_rig_stage(
                "fingers",
                self._finger_artifact(
                    "hand_mask_edit",
                    masked_bones=list(result.masked_bones or []),
                ),
            )
        else:
            self._fail_rig_stage("fingers", result.message)

        # Re-sync checkbox column with the canonical AcuRig state — in
        # case ``apply_hand_masks`` snapped to a slightly different set.
        if result.ok and hasattr(self.inspector, "set_hand_masked_bones"):
            try:
                self.inspector.set_hand_masked_bones(result.masked_bones)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_hand_masked_bones failed")
        if result.ok:
            self._schedule_live_validation("hand_mask_changed")

    # ── M12 / T1204 — Motion assignment ────────────────────────────────

    def _refresh_motion_assignment_state(self) -> None:
        """Mirror workflow motion state into the inspector controls."""
        try:
            _wf = self._body_workflow_module()
        except Exception:
            return

        result = _wf.motion_assignment_options(self.scene)
        if hasattr(self.inspector, "set_motion_assignment"):
            try:
                self.inspector.set_motion_assignment(
                    source=result.source,
                    supermodel=result.supermodel,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment failed")
        if hasattr(self.inspector, "set_motion_assignment_status"):
            kind = "ok" if result.ok else "warning"
            try:
                self.inspector.set_motion_assignment_status(
                    result.message, kind=kind,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment_status failed")

    @QtCore.Slot()
    def _on_assign_motions_requested(self) -> None:
        """Apply the selected KOTOR motion source to the current body."""
        _wf = self._body_workflow_module()

        source = "model"
        if hasattr(self.inspector, "selected_motion_source"):
            source = self.inspector.selected_motion_source()
        supermodel = ""
        if hasattr(self.inspector, "selected_motion_supermodel"):
            supermodel = self.inspector.selected_motion_supermodel()

        result = _wf.assign_motion_source(
            self.scene,
            source,
            supermodel=supermodel,
        )
        kind = "ok" if result.ok else "warning"
        if result.code in ("no_body", "unknown_source"):
            kind = "error"

        if hasattr(self.inspector, "set_motion_assignment_status"):
            try:
                self.inspector.set_motion_assignment_status(
                    result.message, kind=kind,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment_status failed")

        self.bottom_strip.set_validation(
            "info" if result.ok else "warning",
            (result.code or "motion").upper(),
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)

        if result.ok:
            self._on_refresh_preview_animations_requested()
            self._update_title()
            self._schedule_live_validation("motions_assigned")

    def _sync_motion_controls_to_scene(self, workflow_module=None) -> Optional[Any]:
        """Apply the inspector's motion dropdowns before library/preview queries."""
        try:
            _wf = workflow_module or self._body_workflow_module()
        except Exception:                                  # pragma: no cover
            return None

        source = "model"
        if hasattr(self.inspector, "selected_motion_source"):
            source = self.inspector.selected_motion_source()
        supermodel = ""
        if hasattr(self.inspector, "selected_motion_supermodel"):
            supermodel = self.inspector.selected_motion_supermodel()

        result = _wf.assign_motion_source(
            self.scene,
            source,
            supermodel=supermodel,
        )
        if hasattr(self.inspector, "set_motion_assignment_status"):
            try:
                kind = "ok" if result.ok else "warning"
                if result.code in ("no_body", "unknown_source"):
                    kind = "error"
                self.inspector.set_motion_assignment_status(
                    result.message,
                    kind=kind,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment_status failed")
        return result

    @QtCore.Slot()
    def _on_run_rom_test_requested(self) -> None:
        """Assign and run the generated range-of-motion preview."""
        try:
            _wf = self._body_workflow_module()
        except Exception as exc:                            # pragma: no cover
            log.exception("Could not import headless_body_workflow")
            self.bottom_strip.set_validation(
                "error", "ROM_UNAVAILABLE",
                issues=[f"ROM workflow unavailable: {exc}"],
            )
            return

        viewport = getattr(self, "viewport", None)
        result = _wf.run_rom_test(self.scene, viewport=viewport)
        kind = "ok" if result.ok else "error"

        if hasattr(self.inspector, "set_motion_assignment"):
            try:
                self.inspector.set_motion_assignment(
                    source=getattr(_wf, "MOTION_SOURCE_ROM", "generated_rom"),
                    supermodel="",
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment failed")
        if hasattr(self.inspector, "set_motion_assignment_status"):
            try:
                self.inspector.set_motion_assignment_status(
                    result.message,
                    kind=kind,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_motion_assignment_status failed")

        preview = _wf.available_preview_animations(self.scene)
        if hasattr(self.inspector, "set_preview_animations"):
            try:
                self.inspector.set_preview_animations(
                    preview.available,
                    preview.missing,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_animations failed")
        if hasattr(self.inspector, "set_preview_status"):
            try:
                self.inspector.set_preview_status(result.message, kind=kind)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_status failed")

        if result.ok:
            frames = max(1, int(round((result.length or 4.0) * 30)))
            try:
                self.bottom_strip.set_frame_range(0, frames)
                self.bottom_strip.set_current_frame(0)
                self.bottom_strip.set_playing(True)
            except Exception:                               # pragma: no cover
                log.exception("bottom_strip ROM scrubber update failed")

        self.bottom_strip.set_validation(
            "info" if result.ok else "error",
            "ROM_RUNNING" if result.ok else (result.code or "rom").upper(),
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 5000)
        if result.ok:
            self._refresh_motion_assignment_state()
            self._schedule_live_validation("rom_test")

    # ── M5 / T505 — Check-Actor step slots ───────────────────────────────

    @QtCore.Slot()
    def _on_refresh_preview_animations_requested(self) -> None:
        """Re-enumerate preview animations on the body model.

        Calls :func:`headless_body_workflow.available_preview_animations`
        and pushes the available / missing split into the inspector
        dropdown.  Also surfaces a status banner so the user knows
        whether the standard set (walk / idle / talk) is present.
        """
        _wf = self._body_workflow_module()

        self._ensure_game_resource_manager()
        self._sync_motion_controls_to_scene(_wf)

        result = _wf.available_preview_animations(self.scene)
        library = (
            _wf.available_animation_library(self.scene)
            if hasattr(_wf, "available_animation_library")
            else result
        )

        if hasattr(self.inspector, "set_preview_animations"):
            try:
                self.inspector.set_preview_animations(
                    result.available, result.missing,
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_animations failed")
        if hasattr(self.inspector, "set_animation_library"):
            try:
                self.inspector.set_animation_library(
                    library.available,
                    library.missing,
                    message=library.message,
                    diagnostics=getattr(library, "diagnostics", []),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_animation_library failed")

        # Pick the status kind based on the workflow code.
        if result.code == "no_body":
            kind = "error"
        elif result.code == "no_animations":
            kind = "warning"
        else:
            kind = ("ok" if result.available else "warning")

        if hasattr(self.inspector, "set_preview_status"):
            try:
                self.inspector.set_preview_status(result.message, kind=kind)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_status failed")

        msg = library.message if library.available else result.message
        self.statusBar().showMessage(msg, 5000)

    @QtCore.Slot(str)
    def _on_play_preview_animation_requested(self, anim_name: str) -> None:
        """Dispatch a preview animation to the viewport.

        Wraps :func:`headless_body_workflow.play_preview_animation`,
        passing the live viewport widget so its
        ``set_animation_pose`` is invoked on the chosen
        :class:`Animation`.
        """
        _wf = self._body_workflow_module()

        self._ensure_game_resource_manager()
        self._sync_motion_controls_to_scene(_wf)
        result = self._start_preview_animation(anim_name)
        if result is None:
            result = _wf.play_preview_animation(
                self.scene, anim_name, viewport=getattr(self, "viewport", None),
            )

        if hasattr(self.inspector, "set_preview_status"):
            try:
                self.inspector.set_preview_status(
                    result.message,
                    kind=("ok" if result.ok else "error"),
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_status failed")

        if not result.ok:
            self.bottom_strip.set_validation(
                "warning", (result.code or "preview").upper(),
                issues=[result.message],
            )
            self.statusBar().showMessage(result.message, 6000)
            return

        self.bottom_strip.set_validation(
            "info", "PREVIEW_PLAYING",
            issues=[result.message],
        )
        self.statusBar().showMessage(result.message, 4000)

    @QtCore.Slot()
    def _on_stop_preview_animation_requested(self) -> None:
        """Halt the currently-playing preview animation.

        Wraps :func:`headless_body_workflow.stop_preview_animation`,
        which dispatches ``viewport.set_animation_pose(None)`` per the
        existing viewport contract.
        """
        _wf = self._body_workflow_module()

        viewport = getattr(self, "viewport", None)
        timer = getattr(self, "_animation_timer", None)
        if timer is not None:
            timer.stop()
        self._animation_last_tick = None
        engine = getattr(self, "_animation_engine", None)
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        result = _wf.stop_preview_animation(viewport=viewport)

        if hasattr(self.inspector, "set_preview_status"):
            try:
                self.inspector.set_preview_status(
                    result.message, kind="info",
                )
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_preview_status failed")

        self.statusBar().showMessage(result.message, 4000)

    def _start_preview_animation(self, anim_name: str) -> Optional[Any]:
        """Start real AnimationEngine playback for local or inherited clips."""
        body = self._body_model_for_preview()
        if body is None:
            return None
        self._ensure_game_resource_manager()
        try:
            from src.core.animation.animation_engine import AnimationEngine
        except ImportError:                                 # pragma: no cover
            from core.animation.animation_engine import AnimationEngine  # type: ignore
        engine = AnimationEngine(body)
        if not engine.play(str(anim_name or ""), loop=True, blend=False):
            return None
        self._animation_engine = engine
        self._animation_last_tick = None
        anim = engine.current_animation
        length = float(getattr(anim, "length", 0.0) or 0.0) if anim else 0.0
        base_pose = engine.evaluate(0.0)
        pose = base_pose
        preview_anim_name = str(getattr(anim, "name", anim_name) if anim else anim_name)
        try:
            setattr(base_pose, "_gr_animation_source_model_id", id(body))
            setattr(base_pose, "_gr_animation_source_model_name", str(getattr(body, "name", "") or ""))
            setattr(base_pose, "_gr_animation_name", preview_anim_name)
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_animation_pose"):
            if hasattr(viewport, "set_anim_base_pose"):
                viewport.set_anim_base_pose(base_pose)
            viewport.set_animation_pose(
                pose,
                name=preview_anim_name,
                time=0.0,
                length=length,
            )
        try:
            from core.characters import character_builder as _cb
        except ImportError:                                 # pragma: no cover
            from src.core.characters import character_builder as _cb  # type: ignore
        try:
            _cb.log_character_builder_event(
                "ui.preview_animation.started",
                animation=str(getattr(anim, "name", anim_name) if anim else anim_name),
                length=length,
                body=_cb.summarize_model_for_character_builder(body),
            )
        except Exception:
            log.debug("Character Builder preview diagnostic failed", exc_info=True)
        self._animation_timer.start()
        _wf = self._body_workflow_module()
        return _wf.CheckActorResult(
            ok=True,
            playing=str(getattr(anim, "name", anim_name) if anim else anim_name),
            length=length,
            message=f"Playing '{getattr(anim, 'name', anim_name)}' ({length:.2f}s).",
            code="playing",
        )

    def _tick_preview_animation(self) -> None:
        engine = getattr(self, "_animation_engine", None)
        if engine is None or not getattr(engine, "is_playing", False):
            self._animation_timer.stop()
            self._animation_last_tick = None
            return
        now = time.perf_counter()
        if self._animation_last_tick is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1.0 / 60.0, min(now - self._animation_last_tick, 0.25))
        self._animation_last_tick = now
        still_playing = engine.advance(dt)
        anim = engine.current_animation
        pose = engine.evaluate()
        length = float(getattr(anim, "length", 0.0) or 0.0) if anim else 0.0
        name = str(getattr(anim, "name", "") or "")
        body = self._body_model_for_preview()
        try:
            setattr(pose, "_gr_animation_source_model_id", id(body) if body is not None else 0)
            setattr(pose, "_gr_animation_source_model_name", str(getattr(body, "name", "") or ""))
            setattr(pose, "_gr_animation_name", name)
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "set_animation_pose"):
            viewport.set_animation_pose(
                pose,
                name=name,
                time=engine.current_time,
                length=length,
            )
        if length > 0:
            try:
                frame = int(max(0.0, min(engine.current_time / length, 1.0)) * length * 30)
                self.bottom_strip.set_current_frame(frame)
                self.bottom_strip.set_playing(True)
            except Exception:
                pass
        if not still_playing:
            self._animation_timer.stop()
            self._animation_last_tick = None

    @QtCore.Slot()
    def _on_browse_preview_attachment_requested(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose KOTOR weapon or equipment MDL",
            "",
            "KOTOR model (*.mdl);;All files (*.*)",
        )
        if not path:
            return
        self._preview_attachment_path = os.path.abspath(path)
        if hasattr(self.inspector, "set_preview_attachment_source"):
            self.inspector.set_preview_attachment_source(
                path=self._preview_attachment_path,
            )
        if hasattr(self.inspector, "set_preview_attachment_status"):
            self.inspector.set_preview_attachment_status(
                f"Selected {Path(path).name}. Click Attach Preview.",
                kind="ok",
            )

    @QtCore.Slot(str, str, str)
    def _on_attach_preview_attachment_requested(
        self,
        socket: str,
        resref: str,
        path: str,
    ) -> None:
        try:
            from core.assets import asset_preview as _ap
        except ImportError:                                 # pragma: no cover
            from src.core.assets import asset_preview as _ap        # type: ignore

        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        manager = self._ensure_game_resource_manager(game)
        item_model = None
        item_path = str(path or self._preview_attachment_path or "")
        clean_resref = normalize_bas_model_resref(resref)
        if item_path and os.path.isfile(item_path):
            try:
                from core.game.kotor_loader import load_model_from_file
            except ImportError:                              # pragma: no cover
                from src.core.game.kotor_loader import load_model_from_file  # type: ignore
            item_model = load_model_from_file(item_path)
            clean_resref = clean_resref or normalize_bas_model_resref(Path(item_path).stem)
        elif manager is not None and clean_resref:
            if bas_slot_for_preview_socket(socket, clean_resref) == "head":
                try:
                    try:
                        from src.core.geometry import model_data as _md
                    except ImportError:                  # pragma: no cover
                        from core.geometry import model_data as _md  # type: ignore
                    body_model = self.scene.get_model(_md.PartSlot.HEADLESS_BODY)
                except Exception:
                    body_model = None
                resolution = resolve_bas_head_resref(
                    requested=clean_resref,
                    body_model=body_model,
                    manager=manager,
                    game=str(game or "K1").upper(),
                )
                clean_resref = resolution.resolved_resref or clean_resref
            item_model = manager.load_model(clean_resref, str(game or "K1").upper())

        bas_slot = bas_slot_for_preview_socket(socket, clean_resref)
        bas_socket = bas_socket_for_slot(bas_slot) if bas_slot else str(socket or "rhand")

        spec = _ap.AttachmentSpec(
            item_model=item_model,
            item_resref=clean_resref,
            item_path=item_path,
            socket=bas_socket,
            attachment_type=_attachment_type_from_resref(clean_resref),
        )
        result = _ap.attach_item_to_preview(self.scene, spec)
        kind = "ok" if result.ok else "error"
        if hasattr(self.inspector, "set_preview_attachment_status"):
            self.inspector.set_preview_attachment_status(result.message, kind=kind)
        self.bottom_strip.set_validation(
            "info" if result.ok else "warning",
            (result.code or "attachment").upper(),
            issues=[result.message] + list(getattr(result, "warnings", []) or []),
        )
        self.statusBar().showMessage(result.message, 6000)
        if result.ok:
            self._show_attachment_preview_model(
                body=getattr(result, "body_model", None),
                item=getattr(result, "item_model", None),
                socket_name=bas_socket,
                bas_slot=bas_slot,
                item_resref=clean_resref,
            )
            self._schedule_live_validation("preview_attachment")

    def _show_attachment_preview_model(
        self,
        *,
        body: Any,
        item: Any,
        socket_name: str,
        bas_slot: str,
        item_resref: str,
    ) -> None:
        if body is None or item is None:
            return
        try:
            slot = str(bas_slot or bas_slot_for_preview_socket(socket_name, item_resref) or "").strip()
            if not slot:
                return
            label = str(item_resref or getattr(item, "name", "") or "")
            self._bas_preview_attachments[slot] = (item, label)
            self._rebuild_cb_bas_preview(body)
            bas_panel = getattr(self.inspector, "body_attachment_panel", None)
            if bas_panel is not None:
                bas_panel.set_slot_model(slot, item, resref=label)
            if hasattr(self.inspector, "set_preview_attachment_status"):
                self.inspector.set_preview_attachment_status(
                    f"BAS preview attached {label or 'attachment'} to {bas_socket_for_slot(slot)}.",
                    kind="ok",
                )
        except Exception:                                  # pragma: no cover
            log.exception("Could not build attachment preview model")

    def _rebuild_cb_bas_preview(
        self,
        body: Any,
        *,
        skip_unsupported: bool = False,
    ) -> None:
        """Compose the body with every attached BAS layer and present it."""

        self._bas_preview_body = body
        attachment_models = {slot: item for slot, (item, _resref) in self._bas_preview_attachments.items()}
        attachment_transforms = {
            slot: default_bas_attachment_transform(slot, resref)
            for slot, (_item, resref) in self._bas_preview_attachments.items()
        }
        preview = build_bas_preview_model(
            body_model=body,
            attachment_models=attachment_models,
            attachment_transforms=attachment_transforms,
            name=f"{getattr(body, 'name', 'body')}_bas_preview",
            skip_unsupported=skip_unsupported,
        )
        skipped_rows = tuple(
            getattr(preview, "_gr_bas_skipped_attachments", ()) or ()
        )
        self._bas_last_skipped_attachments = skipped_rows
        skipped_slots = {
            str(row.get("slot", ""))
            for row in skipped_rows
            if isinstance(row, dict)
        }
        setattr(self.scene, "preview_model", preview)
        metadata = getattr(self.scene, "metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("body_attachment_system", {})
            metadata["body_attachment_system"].update({
                "active": bool(set(attachment_models) - skipped_slots),
                "preview_owner": "character_builder",
                "attachments": {slot: resref for slot, (_item, resref) in self._bas_preview_attachments.items()},
                "layers": [
                    {
                        "slot": slot,
                        "socket": bas_socket_for_slot(slot),
                        "resref": resref,
                        "enabled": slot not in skipped_slots,
                        "reason": (
                            "model has no compatible visual socket"
                            if slot in skipped_slots else ""
                        ),
                    }
                    for slot, (_item, resref) in self._bas_preview_attachments.items()
                ],
            })
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "load_model"):
            viewport.load_model(preview)

    def _cb_bas_body_model(self) -> Any:
        """Resolve the current base body for BAS preview composition."""

        if self._bas_preview_body is not None:
            return self._bas_preview_body
        try:
            from src.core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from core.geometry import model_data as _md        # type: ignore
        for slot_name in ("HEADLESS_BODY", "FULL_BODY"):
            part = getattr(_md.PartSlot, slot_name, None)
            if part is None:
                continue
            try:
                model = self.scene.get_model(part)
            except Exception:
                model = None
            if model is not None:
                return model
        return None

    def _ensure_cb_bas_attachment_catalog(self, *_args) -> None:
        """Populate the embedded BAS panel from the installed games once."""

        panel = getattr(self.inspector, "body_attachment_panel", None)
        if panel is None:
            return
        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        manager = None
        for game_tag in ("K1", "K2"):
            manager = self._ensure_game_resource_manager(game_tag) or manager
        # Indexing both installs must not leave the viewport targeting the
        # other game's texture/supermodel context.
        manager = self._ensure_game_resource_manager(game) or manager
        if manager is None:
            return
        revision = getattr(manager, "revision", None)
        revision_getter = getattr(panel, "attachment_catalog_revision", None)
        current_revision = revision_getter() if callable(revision_getter) else None
        catalog_ready = panel.attachment_catalog() is not None and (
            revision is None or current_revision == int(revision)
        )
        if not catalog_ready:
            try:
                from src.systems.bas.attachment_catalog import build_bas_attachment_catalog

                catalog = build_bas_attachment_catalog(manager)
            except Exception:                               # pragma: no cover
                log.debug("Character Builder BAS catalog build failed", exc_info=True)
            else:
                if not catalog.empty:
                    panel.set_attachment_catalog(catalog, revision=revision)
                    panel.set_status(
                        "BAS catalog ready: "
                        f"{len(catalog.entries('head'))} K1/K2 head choices and "
                        f"{len(catalog.entries('body'))} headless body choices."
                    )
        if panel.selected_mode() == "character_preset" and hasattr(
            panel, "load_character_preset_catalog"
        ):
            panel.load_character_preset_catalog(manager, revision=revision)

    def _on_bas_character_preset_requested(self, preset) -> None:
        """Replace the Character Studio preview with one canonical UTC build."""

        panel = getattr(self.inspector, "body_attachment_panel", None)
        game = str(getattr(preset, "game", "") or "").strip().upper()
        body_resref = normalize_bas_model_resref(getattr(preset, "body_resref", ""))
        body_game = str(getattr(preset, "body_game", "") or game).strip().upper()
        if panel is None or game not in {"K1", "K2"} or not body_resref:
            if panel is not None:
                panel.set_status("The selected character preset has no loadable body model.")
            return
        manager = self._ensure_game_resource_manager(game)
        if manager is None:
            panel.set_status(f"Configure the {game} installation before loading this character.")
            return

        def _load_model(resref: str, source_game: str):
            source_manager = self._ensure_game_resource_manager(source_game) or manager
            try:
                if hasattr(source_manager, "load_model_strict"):
                    return source_manager.load_model_strict(resref, source_game)
                return source_manager.load_model(resref, source_game)
            except Exception:
                return None

        body = _load_model(body_resref, body_game)
        if body is None:
            panel.set_status(f"Could not load canonical body {body_game}:{body_resref}.")
            return
        requested_layers: dict[str, str] = dict(getattr(preset, "slot_model_map", {}) or {})
        requested_layer_games: dict[str, str] = dict(
            getattr(preset, "slot_game_map", {}) or {}
        )
        head_resref = normalize_bas_model_resref(getattr(preset, "head_resref", ""))
        if head_resref:
            requested_layers["head"] = head_resref
            requested_layer_games["head"] = str(
                getattr(preset, "head_game", "") or game
            ).strip().upper()
        loaded_layers: dict[str, Any] = {}
        missing: list[str] = []
        for slot, resref in requested_layers.items():
            layer_game = str(requested_layer_games.get(slot) or game).upper()
            model = _load_model(resref, layer_game)
            if model is None:
                missing.append(f"{slot}: {layer_game}:{resref}")
                continue
            loaded_layers[slot] = model
        if head_resref and "head" not in loaded_layers:
            head_game = str(requested_layer_games.get("head") or game).upper()
            panel.set_status(
                f"Preset was not applied because canonical head {head_game}:{head_resref} could not be loaded."
            )
            return

        from src.systems.bas.character_presets import apply_bas_texture_override

        apply_bas_texture_override(body, getattr(preset, "body_texture", ""))
        if "head" in loaded_layers:
            apply_bas_texture_override(loaded_layers["head"], getattr(preset, "head_texture", ""))
        setattr(
            body,
            "_gr_bas_character_preset",
            {
                "game": game,
                "utc_resref": str(getattr(preset, "utc_resref", "") or ""),
                "display_name": str(getattr(preset, "display_name", "") or ""),
                "body_game": body_game,
                "head_game": str(getattr(preset, "head_game", "") or game).upper(),
                "slot_games": dict(requested_layer_games),
                "slot_models": dict(getattr(preset, "slot_model_map", {}) or {}),
                "body_texture": str(getattr(preset, "body_texture", "") or ""),
                "head_texture": str(getattr(preset, "head_texture", "") or ""),
                "equipment_items": dict(getattr(preset, "equipment_item_map", {}) or {}),
            },
        )
        try:
            try:
                from src.core.geometry import model_data as _md
            except ImportError:                             # pragma: no cover
                from core.geometry import model_data as _md  # type: ignore
            composition_mode = str(
                getattr(preset, "composition_mode", "headless_body") or "headless_body"
            )
            if composition_mode == "full_body":
                character_mode = _md.CharacterMode.FULL_BODY
                part_slot = _md.PartSlot.FULL_BODY
            else:
                character_mode = _md.CharacterMode.HEADLESS_BODY
                part_slot = _md.PartSlot.HEADLESS_BODY
            self.scene.set_mode(character_mode, locked=True)
            self.scene.assign(
                part_slot,
                body,
                resref=body_resref,
                game_version=body_game,
            )
            self.scene.game_version = body_game
            game_index = self._game_combo.findText(body_game) if hasattr(self, "_game_combo") else -1
            if game_index >= 0:
                self._game_combo.blockSignals(True)
                self._game_combo.setCurrentIndex(game_index)
                self._game_combo.blockSignals(False)
            self._bas_preview_body = body
            self._bas_preview_attachments = {
                slot: (model, requested_layers[slot]) for slot, model in loaded_layers.items()
            }
            self._body_guides = {}
            self._rebuild_cb_bas_preview(body, skip_unsupported=True)
            skipped_rows = tuple(
                getattr(self, "_bas_last_skipped_attachments", ()) or ()
            )
            skipped_slots = {
                str(row.get("slot", ""))
                for row in skipped_rows
                if isinstance(row, dict)
            }
            for slot in skipped_slots:
                self._bas_preview_attachments.pop(slot, None)
            preset_metadata = getattr(body, "_gr_bas_character_preset", None)
            if isinstance(preset_metadata, dict):
                preset_metadata["nonvisual_slots"] = {
                    slot: requested_layers.get(slot, "")
                    for slot in sorted(skipped_slots)
                    if requested_layers.get(slot, "")
                }
        except Exception as exc:                            # pragma: no cover
            log.exception("Character Builder canonical BAS preset failed")
            panel.set_status(f"Could not apply canonical {game} preset: {exc}")
            return

        self._sync_from_scene()
        for slot in (
            "head", "mask", "goggles", "left_hand", "right_hand",
            "left_weapon", "belt", "right_weapon",
        ):
            if slot != "body":
                panel.clear_slot_model(slot)
        panel.set_composition_mode(composition_mode)
        panel.set_mode("character_preset")
        panel.set_body_model(body, resref=body_resref, game=body_game)
        for slot, model in loaded_layers.items():
            panel.set_slot_model(slot, model, resref=requested_layers[slot])
        warnings = list(getattr(preset, "warnings", ()) or ())
        if missing:
            warnings.append("Missing optional layers: " + ", ".join(missing))
        if skipped_rows:
            warnings.append(
                "Non-visual inventory kept as metadata: "
                + ", ".join(
                    f"{row.get('slot', 'item')} ({row.get('socket', 'socket')} unavailable)"
                    for row in skipped_rows
                    if isinstance(row, dict)
                )
            )
        suffix = f" Warnings: {'; '.join(warnings[:3])}" if warnings else ""
        source_games = {
            source
            for source in (body_game, *requested_layer_games.values())
            if source in {"K1", "K2"} and source != game
        }
        source_note = (
            f" Uses {', '.join(sorted(source_games))} legacy assets."
            if source_games else ""
        )
        panel.set_status(
            f"Loaded canonical {game} character {getattr(preset, 'display_name', '') or getattr(preset, 'utc_resref', '')}: "
            f"{getattr(preset, 'summary', '')}.{source_note}{suffix}"
        )
        self._schedule_live_validation("bas_character_preset")

    def _on_bas_panel_attach_requested(self, slot: str, resref: str) -> None:
        """Attach a catalog item from the embedded BAS panel to the preview."""

        slot = str(slot or "").strip().lower()
        panel = getattr(self.inspector, "body_attachment_panel", None)
        if slot in {"left_hand", "right_hand"}:
            if panel is not None:
                panel.set_status("Hand slots are sockets; attach items through the weapon slots.")
            return
        resref_clean = normalize_bas_model_resref(resref)
        if not resref_clean:
            if panel is not None:
                panel.set_status("No attachment model selected.")
            return
        game = self._game_combo.currentText() if hasattr(self, "_game_combo") else \
            getattr(self.scene, "game_version", "K1")
        selected_game_getter = getattr(panel, "selected_model_game", None)
        selected_game = str(selected_game_getter() or "").upper() if callable(selected_game_getter) else ""
        target_game = selected_game if selected_game in {"K1", "K2"} else str(game or "K1").upper()
        manager = self._ensure_game_resource_manager(target_game)
        if manager is None:
            if panel is not None:
                panel.set_status("Configure the KOTOR game folders to load attachment models.")
            return
        if slot == "body":
            try:
                item_model = (
                    manager.load_model_strict(resref_clean, target_game)
                    if hasattr(manager, "load_model_strict")
                    else manager.load_model(resref_clean, target_game)
                )
            except Exception:
                item_model = None
            if item_model is None:
                if panel is not None:
                    panel.set_status(f"Could not load {target_game}:{resref_clean}.")
                return
            repair_bas_body_texture_references(
                item_model,
                manager=manager,
                game=target_game,
                resref=resref_clean,
            )
            try:
                try:
                    from src.core.geometry import model_data as _md
                except ImportError:                         # pragma: no cover
                    from core.geometry import model_data as _md  # type: ignore
                self.scene.set_mode(_md.CharacterMode.HEADLESS_BODY, locked=True)
                self.scene.assign(
                    _md.PartSlot.HEADLESS_BODY,
                    item_model,
                    resref=resref_clean,
                    game_version=target_game,
                )
                self.scene.game_version = target_game
                self._bas_preview_body = item_model
                self._body_guides = {}
                self._rebuild_cb_bas_preview(item_model)
            except Exception as exc:                        # pragma: no cover
                log.exception("Character Builder BAS body switch failed")
                if panel is not None:
                    panel.set_status(f"Could not use {target_game}:{resref_clean}: {exc}")
                return
            self._sync_from_scene()
            if panel is not None:
                panel.set_body_model(item_model, resref=resref_clean, game=target_game)
                panel.set_mode("headless_body")
                panel.set_status(
                    f"Using {target_game}:{resref_clean} as the BAS body; "
                    "existing attachment layers were preserved."
                )
            self._schedule_live_validation("bas_body")
            return
        body = self._cb_bas_body_model()
        if body is None:
            if panel is not None:
                panel.set_status("Load or build a body first (steps 1-3), then attach preview items.")
            return
        if slot == "head":
            resolution = resolve_bas_head_resref(
                requested=resref_clean,
                body_model=body,
                manager=manager,
                game=target_game,
            )
            resref_clean = resolution.resolved_resref or resref_clean
        try:
            item_model = (
                manager.load_model_strict(resref_clean, target_game)
                if selected_game and hasattr(manager, "load_model_strict")
                else manager.load_model(resref_clean, target_game)
            )
        except Exception:
            item_model = None
        if item_model is None:
            if panel is not None:
                panel.set_status(f"Could not load {resref_clean}.")
            return
        self._bas_preview_attachments[slot] = (item_model, resref_clean)
        try:
            self._rebuild_cb_bas_preview(body)
        except Exception as exc:                            # pragma: no cover
            log.exception("Character Builder BAS preview rebuild failed")
            self._bas_preview_attachments.pop(slot, None)
            if panel is not None:
                panel.set_status(f"BAS preview failed: {exc}")
            return
        if panel is not None:
            panel.set_slot_model(slot, item_model, resref=resref_clean)
            attached = ", ".join(resref for _slot, (_item, resref) in sorted(self._bas_preview_attachments.items()))
            panel.set_status(f"BAS preview updated: {attached}.")
        self._schedule_live_validation("preview_attachment")

    def _on_bas_panel_clear_requested(self, slot: str) -> None:
        slot = str(slot or "").strip().lower()
        panel = getattr(self.inspector, "body_attachment_panel", None)
        if slot not in self._bas_preview_attachments:
            if panel is not None:
                panel.set_status(f"{slot.replace('_', ' ')} is already empty.")
            return
        self._bas_preview_attachments.pop(slot, None)
        if panel is not None:
            panel.clear_slot_model(slot)
        body = self._cb_bas_body_model()
        if body is not None:
            try:
                self._rebuild_cb_bas_preview(body)
            except Exception:                               # pragma: no cover
                log.exception("Character Builder BAS preview rebuild failed after clear")
        if panel is not None:
            panel.set_status(f"Cleared {slot.replace('_', ' ')}.")

    # ── M6 / T602 — Head Facial Palette slots ────────────────────────────

    @QtCore.Slot(str)
    def _on_head_facial_bone_selected(self, bone_name: str) -> None:
        """Forward a Head Facial Palette click to viewport + status bar.

        For M6 we surface the selection on the status bar and the
        bottom-strip info banner; M7 / M8 will wire this to the M4
        joint-dot HUD so the matching dot highlights in the viewport.
        """
        if not bone_name:
            return
        msg = f"Facial bone: {bone_name}"
        # Surface on the bottom strip (informational, not blocking).
        try:
            self.bottom_strip.set_validation(
                "info", "FACIAL_BONE_SELECTED",
                issues=[msg],
            )
        except Exception:                                   # pragma: no cover
            log.exception("bottom_strip.set_validation failed for "
                          "head facial bone selection")
        self.statusBar().showMessage(msg, 4000)
        # Best-effort: highlight on the joint-dot HUD when the viewport
        # exposes a per-joint selector (M4 / T402 surface).
        viewport = getattr(self, "viewport", None)
        if viewport is not None and hasattr(viewport, "highlight_joint"):
            try:
                viewport.highlight_joint(bone_name)
            except Exception:                               # pragma: no cover
                log.exception("viewport.highlight_joint failed for %s",
                              bone_name)

    @QtCore.Slot()
    def _on_rig_head_requested(self) -> None:
        """Run the M6 / T601 Head Rig step from the Inspector palette."""
        try:
            from core.characters import head_workflow as _hw
        except ImportError:                                 # pragma: no cover
            from src.core.characters import head_workflow as _hw       # type: ignore
        # Parent body is None for stand-alone head edits; the
        # supermodel-mode window will pass scene.headless_body model.
        result = _hw.rig_head(self.scene, parent_body=None)
        kind = "ok" if result.ok else "error"
        try:
            self.bottom_strip.set_validation(
                kind, (result.code or "rig_head").upper(),
                issues=[result.message],
            )
        except Exception:                                   # pragma: no cover
            log.exception("bottom_strip.set_validation failed for rig_head")
        self.statusBar().showMessage(result.message, 5000)
        if result.ok:
            self._schedule_live_validation("head_rigged")

    @QtCore.Slot()
    def _on_rig_face_requested(self) -> None:
        """Run the M6 / T601 Face Rig step from the Inspector palette."""
        try:
            from core.characters import head_workflow as _hw
        except ImportError:                                 # pragma: no cover
            from src.core.characters import head_workflow as _hw       # type: ignore
        result = _hw.rig_face(self.scene)
        kind = "ok" if result.ok else "warning"
        try:
            self.bottom_strip.set_validation(
                kind, (result.code or "rig_face").upper(),
                issues=[result.message],
            )
        except Exception:                                   # pragma: no cover
            log.exception("bottom_strip.set_validation failed for rig_face")
        self.statusBar().showMessage(result.message, 5000)
        if result.ok:
            self._schedule_live_validation("face_rigged")

    @QtCore.Slot(int)
    def _on_apply_viseme_requested(self, viseme_index: int) -> None:
        """Apply a LIPShape viseme to the head's facial bones (M6 / T603).

        Forwards to :func:`head_workflow.apply_viseme`.  The result
        (``(ok, message)``) is surfaced through the inspector's viseme
        status line, the bottom-strip banner, and the main-window
        status bar so the user gets consistent feedback regardless of
        which surface they're watching.
        """
        try:
            from core.characters import head_workflow as _hw
        except ImportError:                                 # pragma: no cover
            from src.core.characters import head_workflow as _hw       # type: ignore
        try:
            # Xaria and normal modular heads inherit ``talk`` from their
            # supermodel; configure that resolver before evaluating a slot.
            self._ensure_game_resource_manager()
            ok, message = _hw.apply_viseme(
                self.scene,
                int(viseme_index),
                viewport=getattr(self, "viewport", None),
            )
        except Exception as exc:                            # pragma: no cover
            log.exception("apply_viseme failed for viseme=%s", viseme_index)
            ok, message = False, f"apply_viseme raised: {exc}"

        kind = "ok" if ok else "warning"

        # 1. Inspector status line.
        if hasattr(self.inspector, "set_viseme_status"):
            try:
                self.inspector.set_viseme_status(message, kind=kind)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_viseme_status failed")

        # 2. Bottom-strip banner.
        try:
            self.bottom_strip.set_validation(
                kind, f"VISEME_{int(viseme_index):02d}",
                issues=[message],
            )
        except Exception:                                   # pragma: no cover
            log.exception("bottom_strip.set_validation failed for viseme")

        # 3. Main status bar.
        try:
            self.statusBar().showMessage(message, 5000)
        except Exception:                                   # pragma: no cover
            pass
        if ok:
            self._schedule_live_validation("viseme_previewed")

    @QtCore.Slot(str, int)
    def _on_calibrate_phoneme_requested(
        self, phoneme_label: str, viseme_index: int,
    ) -> None:
        """Persist a phoneme→viseme calibration (M6 / T604).

        Forwards to :func:`head_workflow.calibrate_phoneme` which
        stashes the mapping on ``scene.head_phoneme_calibration`` for
        the M9 persistence pass to consume.  Result is surfaced via
        inspector status, bottom-strip banner, and status bar.
        """
        try:
            from core.characters import head_workflow as _hw
        except ImportError:                                 # pragma: no cover
            from src.core.characters import head_workflow as _hw       # type: ignore
        try:
            ok, message = _hw.calibrate_phoneme(
                self.scene, str(phoneme_label), int(viseme_index)
            )
        except Exception as exc:                            # pragma: no cover
            log.exception(
                "calibrate_phoneme failed for label=%r viseme=%s",
                phoneme_label, viseme_index,
            )
            ok, message = False, f"calibrate_phoneme raised: {exc}"

        kind = "ok" if ok else "warning"

        # 1. Inspector status line.
        if hasattr(self.inspector, "set_phoneme_status"):
            try:
                self.inspector.set_phoneme_status(message, kind=kind)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_phoneme_status failed")

        # 2. Bottom-strip banner.  Code segment is the phoneme label
        # with non-identifier chars stripped so the banner code key
        # stays parseable (e.g. "PHONEME_AH_OPEN_VOWEL").
        safe_label = "".join(
            ch if ch.isalnum() else "_"
            for ch in str(phoneme_label).strip().upper()
        ).strip("_")
        try:
            self.bottom_strip.set_validation(
                kind, f"PHONEME_{safe_label or 'UNKNOWN'}",
                issues=[message],
            )
        except Exception:                                   # pragma: no cover
            log.exception("bottom_strip.set_validation failed for phoneme")

        # 3. Main status bar.
        try:
            self.statusBar().showMessage(message, 5000)
        except Exception:                                   # pragma: no cover
            pass

    @QtCore.Slot()
    def _on_head_camera_preset_requested(self) -> None:
        """Apply the Head-mode camera preset to the viewport (M6 / T605).

        Forwards to :meth:`QtViewportWidget.apply_head_camera_preset`.
        Result is surfaced through the bottom-strip banner and the
        main status bar.  Silently no-ops when the viewport lacks the
        method (e.g. in lightweight test envs).
        """
        if not hasattr(self.viewport, "apply_head_camera_preset"):
            return                                          # pragma: no cover
        try:
            ok, message = self.viewport.apply_head_camera_preset()
        except Exception as exc:                            # pragma: no cover
            log.exception("apply_head_camera_preset failed")
            ok, message = False, f"head camera preset raised: {exc}"

        kind = "ok" if ok else "warning"
        try:
            self.bottom_strip.set_validation(
                kind, "HEAD_CAMERA_PRESET", issues=[message],
            )
        except Exception:                                   # pragma: no cover
            log.exception(
                "bottom_strip.set_validation failed for head camera preset"
            )
        try:
            self.statusBar().showMessage(message, 5000)
        except Exception:                                   # pragma: no cover
            pass

    # ── Mode-application helper (T205) ───────────────────────────────────

    def _apply_mode(self, mode, *, locked: bool, source: str) -> None:
        """Apply *mode* to scene + rail + inspector + properties panel.

        Parameters
        ----------
        mode    : :class:`CharacterMode` value.  Pass ``None`` to clear
                  any override and re-derive from the scene.
        locked  : When True, locks the scene's mode (toolbar/UI override
                  semantics).  False is used during initial restore.
        source  : Short tag used in log messages for traceability.
        """
        if mode is None:
            return
        # Push into the scene (no-op if the scene doesn't have set_mode).
        if hasattr(self.scene, "set_mode"):
            try:
                self.scene.set_mode(mode, locked=locked)
            except Exception:                              # pragma: no cover
                log.exception("scene.set_mode failed from %s", source)
        # Rebuild rail content.
        self.rail.set_mode(mode)
        self._update_head_builder_workspace(mode)
        # M6 / T602 — also tell the inspector so it can swap the
        # Face-Rig page between legacy controls and the Head Facial
        # Palette.  Guarded with hasattr() because the inspector
        # method ships in M6.
        if hasattr(self.inspector, "set_active_mode"):
            try:
                self.inspector.set_active_mode(mode)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_active_mode failed from %s",
                              source)
        # M6 / T605 — auto-apply the Head camera preset when switching
        # into HEAD mode so the viewport frames the head canonically.
        # Duck-typed detection (matches inspector.set_active_mode).
        self._maybe_apply_head_camera_preset(mode, source=source)
        # Push into properties panel without echoing the signal back.
        if hasattr(self.properties, "set_character_mode"):
            self.properties.set_character_mode(mode, from_scene=True)
        # Echo mode in toolbar buttons.
        self._reflect_mode_in_toolbar(mode)
        self.modeChanged.emit(mode)
        self._update_title()
        self._schedule_live_validation(f"mode_{source}")
        log.info("Character Builder mode → %s (source=%s, locked=%s)",
                 getattr(mode, "name", mode), source, locked)

    def _maybe_apply_head_camera_preset(self, mode, *, source: str) -> None:
        """Apply :func:`head_workflow.head_camera_preset` when *mode* is HEAD.

        M6 / T605.  Duck-typed HEAD detection (``.value`` /``.name``/
        ``str(mode)``) so we don't bind to the pykotor-importing
        :mod:`core` package.  Silently no-ops when the viewport lacks
        the apply method (lightweight test envs).
        """
        if mode is None:
            return
        mode_value = (
            getattr(mode, "value", None)
            or getattr(mode, "name", "")
            or str(mode or "")
        ).lower()
        if mode_value != "head":
            return
        if not hasattr(self.viewport, "apply_head_camera_preset"):
            return
        try:
            self.viewport.apply_head_camera_preset()
        except Exception:                                   # pragma: no cover
            log.exception(
                "Auto-apply of head camera preset failed from %s", source
            )

    def _reflect_mode_in_toolbar(self, mode) -> None:
        """Tick the matching toolbar action without firing handlers."""
        action = self._mode_actions.get(mode)
        if action is None:
            return
        self._suppress_mode_signal = True
        try:
            action.setChecked(True)
        finally:
            self._suppress_mode_signal = False

    def _sync_from_scene(self) -> None:
        """Push the scene's current mode out to rail / inspector / panel."""
        mode = getattr(self.scene, "mode", None)
        self.rail.set_mode(mode)
        self._update_head_builder_workspace(mode)
        # M6 / T602 — keep inspector face-rig page in sync with mode.
        if hasattr(self.inspector, "set_active_mode"):
            try:
                self.inspector.set_active_mode(mode)
            except Exception:                               # pragma: no cover
                log.exception("inspector.set_active_mode failed from _sync_from_scene")
        # M6 / T605 — also auto-apply Head camera preset when syncing
        # into HEAD mode on scene-restore.
        self._maybe_apply_head_camera_preset(mode, source="_sync_from_scene")
        if hasattr(self.properties, "set_character_mode"):
            self.properties.set_character_mode(mode, from_scene=True)
        self._reflect_mode_in_toolbar(mode)
        if hasattr(self.scene, "game_version"):
            self._game_combo.blockSignals(True)
            try:
                self._game_combo.setCurrentText(self.scene.game_version or "K1")
            finally:
                self._game_combo.blockSignals(False)
        bas_panel = getattr(self.inspector, "body_attachment_panel", None)
        if bas_panel is not None:
            body = self._cb_bas_body_model()
            if body is not None:
                bas_panel.set_body_model(body)
        self._refresh_motion_assignment_state()

    def _capture_scene_session_metadata(self) -> None:
        """Persist UI-only rigging state before SceneIO serialises metadata."""
        self._sync_rig_session_metadata(mark_dirty=False)
        metadata = getattr(self.scene, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self.scene, "metadata", metadata)
        if self._selected_skeleton_template_key:
            metadata["skeleton_template_key"] = self._selected_skeleton_template_key

        _entry, model = self._body_model_for_fit_adjustment()
        model_metadata = getattr(model, "metadata", {}) if model is not None else {}
        manual_state = (
            model_metadata.get("manual_fit_adjustment")
            if isinstance(model_metadata, dict) else None
        )
        if not manual_state:
            manual_state = {
                "scale": float(self._manual_fit_scale or 1.0),
                "rotation_degrees": tuple(float(v or 0.0) for v in self._manual_fit_rotation),
                "translation": tuple(float(v or 0.0) for v in self._manual_fit_translation),
            }
        if (
            abs(float(manual_state.get("scale", 1.0) or 1.0) - 1.0) > 1e-6
            or any(abs(float(v or 0.0)) > 1e-6 for v in manual_state.get("rotation_degrees", ()))
            or any(abs(float(v or 0.0)) > 1e-6 for v in manual_state.get("translation", ()))
        ):
            metadata["manual_fit_adjustment"] = {
                "scale": float(manual_state.get("scale", 1.0) or 1.0),
                "rotation_degrees": [
                    float(v or 0.0)
                    for v in list(manual_state.get("rotation_degrees", (0.0, 0.0, 0.0)))[:3]
                ],
                "translation": [
                    float(v or 0.0)
                    for v in list(manual_state.get("translation", (0.0, 0.0, 0.0)))[:3]
                ],
            }

    def _rehydrate_scene_models_from_sources(self) -> list[str]:
        """Reload saved source files so opening a .ghostrig scene is visible."""
        try:
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.geometry import model_data as _md           # type: ignore

        messages: list[str] = []
        saved_entries = list(getattr(self.scene, "slots", {}).items())
        for slot, entry in saved_entries:
            source_path = str(getattr(entry, "source_path", "") or "")
            if not source_path:
                continue
            source_path = os.path.abspath(os.path.expanduser(source_path))
            slot_label = getattr(slot, "value", str(slot))
            if not os.path.isfile(source_path):
                messages.append(f"{slot_label}: source missing ({source_path})")
                continue
            gv = str(
                getattr(entry, "game_version", "")
                or getattr(self.scene, "game_version", "K1")
                or "K1"
            )
            try:
                if slot == _md.PartSlot.HEAD_SHELL:
                    try:
                        from core.characters import head_workflow as _head_wf
                    except ImportError:                      # pragma: no cover
                        from src.core.characters import head_workflow as _head_wf  # type: ignore
                    result = _head_wf.load_head(
                        source_path,
                        self.scene,
                        game_version=gv,
                        allow_mode_correction=True,
                    )
                elif slot == _md.PartSlot.HEADLESS_BODY:
                    try:
                        from core.characters import headless_body_workflow as _body_wf
                    except ImportError:                      # pragma: no cover
                        from src.core.characters import headless_body_workflow as _body_wf  # type: ignore
                    result = _body_wf.load_body(
                        source_path,
                        self.scene,
                        game_version=gv,
                        allow_mode_correction=True,
                    )
                else:
                    result = self._load_generic_scene_slot_model(slot, entry, source_path, gv)
            except Exception as exc:                         # pragma: no cover - loader-specific
                log.exception("Scene restore failed for %s", source_path)
                messages.append(f"{slot_label}: load failed ({exc})")
                continue

            ok = bool(getattr(result, "ok", False))
            model = getattr(result, "model", None)
            if ok or model is not None:
                messages.append(str(getattr(result, "message", "") or f"{slot_label}: loaded"))
            else:
                messages.append(str(getattr(result, "message", "") or f"{slot_label}: not loaded"))
        return messages

    def _load_generic_scene_slot_model(
        self,
        slot: Any,
        entry: Any,
        source_path: str,
        game_version: str,
    ) -> Any:
        """Reload non-body/non-head slots from direct MDL-style sources."""
        try:
            from core.game.kotor_loader import load_model_from_file
        except ImportError:                                 # pragma: no cover
            from src.core.game.kotor_loader import load_model_from_file  # type: ignore

        model = load_model_from_file(source_path)
        resref = str(getattr(entry, "resref", "") or Path(source_path).stem)
        self.scene.assign(
            slot,
            model,
            resref=resref,
            game_version=game_version,
            source_path=source_path,
        )

        return type("SceneSlotLoadResult", (), {
            "ok": True,
            "code": "loaded",
            "model": model,
            "message": f"Loaded {resref} ({Path(source_path).name})",
        })()

    def _restore_manual_fit_from_metadata(self) -> None:
        """Apply saved import fit to a freshly reloaded source mesh."""
        metadata = getattr(self.scene, "metadata", {}) or {}
        state = metadata.get("manual_fit_adjustment") if isinstance(metadata, dict) else None
        if not isinstance(state, dict):
            self._manual_fit_scale = 1.0
            self._manual_fit_rotation = (0.0, 0.0, 0.0)
            self._manual_fit_translation = (0.0, 0.0, 0.0)
            if hasattr(self.inspector, "set_fit_adjustment"):
                self.inspector.set_fit_adjustment(
                    scale=1.0,
                    rotation_degrees=(0.0, 0.0, 0.0),
                    translation=(0.0, 0.0, 0.0),
                    emit=False,
                )
            return
        scale = float(state.get("scale", 1.0) or 1.0)
        rotation = tuple(
            float(v or 0.0)
            for v in list(state.get("rotation_degrees", (0.0, 0.0, 0.0)))[:3]
        )
        translation = tuple(
            float(v or 0.0)
            for v in list(state.get("translation", (0.0, 0.0, 0.0)))[:3]
        )
        rotation = (rotation + (0.0, 0.0, 0.0))[:3]
        translation = (translation + (0.0, 0.0, 0.0))[:3]

        _entry, model = self._body_model_for_fit_adjustment()
        if model is not None and (
            abs(scale - 1.0) > 1e-6
            or any(abs(v) > 1e-6 for v in rotation)
            or any(abs(v) > 1e-6 for v in translation)
        ):
            try:
                from core.characters import headless_body_workflow as _wf
            except ImportError:                             # pragma: no cover
                from src.core.characters import headless_body_workflow as _wf  # type: ignore
            _wf.apply_external_model_fit_adjustment(
                model,
                rotation_delta_degrees=rotation,
                scale_delta=scale,
                translation_delta=translation,
            )
        self._manual_fit_scale = scale
        self._manual_fit_rotation = rotation
        self._manual_fit_translation = translation
        if hasattr(self.inspector, "set_fit_adjustment"):
            self.inspector.set_fit_adjustment(
                scale=scale,
                rotation_degrees=rotation,
                translation=translation,
                emit=False,
            )

    def _primary_scene_entry_for_viewport(self) -> Optional[Any]:
        """Return the best loaded scene slot to show after opening a scene."""
        try:
            from core.geometry import model_data as _md
        except ImportError:                                 # pragma: no cover
            from src.core.geometry import model_data as _md           # type: ignore
        slots = getattr(self.scene, "slots", {}) or {}
        for slot in (
            _md.PartSlot.HEADLESS_BODY,
            _md.PartSlot.HEAD_SHELL,
            _md.PartSlot.BODY_VARIANT,
            _md.PartSlot.OTHER,
        ):
            entry = slots.get(slot)
            if entry is not None and getattr(entry, "model", None) is not None:
                return entry
        for entry in slots.values():
            if getattr(entry, "model", None) is not None:
                return entry
        return None

    def _load_primary_scene_model_in_viewport(self) -> bool:
        """Display the primary rehydrated model after File -> Open Scene."""
        entry = self._primary_scene_entry_for_viewport()
        if entry is None:
            return False
        model = getattr(entry, "model", None)
        if model is None:
            return False
        try:
            self._load_model_in_viewport_with_textures(
                model,
                source_path=str(getattr(entry, "source_path", "") or ""),
                prompt=False,
            )
            if hasattr(self.viewport, "frame_all"):
                self.viewport.frame_all()
            return True
        except Exception:                                  # pragma: no cover
            log.exception("Failed to display restored scene model")
            return False

    def _load_scene_from_path(self, path: str) -> list[str]:
        """Load a .ghostrig file, rehydrate models, and update the builder."""
        SceneIO = _import_scene_io()
        self.scene = SceneIO.load(path, load_models=False)
        self._rig_session = None
        self._restore_rig_session_from_scene()
        self._scene_path = path
        messages = self._rehydrate_scene_models_from_sources()
        self._restore_manual_fit_from_metadata()
        self._sync_from_scene()
        shown = self._load_primary_scene_model_in_viewport()
        self._refresh_skeleton_template_options()
        self._body_guide_history = None
        self._restore_body_guides_from_rig_session()
        self._refresh_body_guide_undo_actions()
        if hasattr(self.scene, "mark_clean"):
            self.scene.mark_clean()
        else:
            self.scene.dirty = False
        self._update_title()
        if not shown:
            messages.append("No renderable source model could be restored.")
        return messages

    # ── Scene I/O (preserved from pre-M2) ────────────────────────────────

    def _confirm_discard_or_save(self, prompt: str) -> bool:
        if not getattr(self.scene, "dirty", False):
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved Changes",
            prompt,
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            return self._save_scene()
        return True

    @QtCore.Slot()
    def _new_scene(self) -> None:
        if not self._confirm_discard_or_save(
            "The current scene has unsaved changes. Save before creating a new scene?"
        ):
            return
        CharacterScene = _import_model_data()
        game_version = getattr(self.scene, "game_version", "K1")
        self.scene = CharacterScene(game_version=game_version)
        self._scene_path = ""
        self._rig_session = None
        self._restore_rig_session_from_scene()
        self._acurig = None
        self._body_guides = {}
        self._body_guide_history = None
        self._refresh_body_guide_undo_actions()
        self.statusBar().showMessage("New scene created", 3000)
        self._sync_from_scene()
        self._update_title()

    @QtCore.Slot()
    def _open_scene(self) -> None:
        if not self._confirm_discard_or_save("Save current scene before opening another?"):
            return
        SceneIO = _import_scene_io()
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Character Scene",
            "",
            f"GhostRigger Scene (*{SceneIO.EXTENSION});;All files (*.*)",
        )
        if not path:
            return
        try:
            messages = self._load_scene_from_path(path)
            self.statusBar().showMessage(f"Scene loaded: {os.path.basename(path)}", 4000)
            if messages:
                self.bottom_strip.set_validation(
                    "info",
                    "SCENE_LOADED",
                    issues=messages[:6],
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open Failed", str(exc))

    @QtCore.Slot()
    def _save_scene(self, *, save_as: bool = False) -> bool:
        SceneIO = _import_scene_io()
        path = self._scene_path
        if not path or save_as:
            path, _selected = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Character Scene",
                "",
                f"GhostRigger Scene (*{SceneIO.EXTENSION});;All files (*.*)",
            )
            if not path:
                return False
            if not path.endswith(SceneIO.EXTENSION):
                path += SceneIO.EXTENSION
        try:
            self._capture_scene_session_metadata()
            SceneIO.save(self.scene, path)
            self._scene_path = path
            self.statusBar().showMessage(f"Scene saved: {os.path.basename(path)}", 4000)
            self._update_title()
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(exc))
            return False

    def _update_title(self) -> None:
        name = getattr(self.scene, "character_name", "") or ""
        if not name and self._scene_path:
            name = os.path.splitext(os.path.basename(self._scene_path))[0]
        dirty_marker = " *" if getattr(self.scene, "dirty", False) else ""
        mode = getattr(self.scene, "mode", None)
        mode_suffix = ""
        if mode is not None:
            mode_label = getattr(mode, "display_name", None) or getattr(mode, "name", None) or str(mode)
            mode_suffix = f" [{mode_label}]"
        suffix = f" - {name}" if name else ""
        self.setWindowTitle(
            f"Ghost-Studio - Character Builder{suffix}{mode_suffix}{dirty_marker}"
        )

    # ── QSettings persistence (T207) ─────────────────────────────────────

    @staticmethod
    def _settings() -> QtCore.QSettings:
        return QtCore.QSettings(_QSETTINGS_ORG, _QSETTINGS_APP)

    def _restore_settings(self) -> None:
        """Restore window geometry / dock state / splitter / last mode."""
        s = self._settings()
        geom = s.value(_QSK_GEOMETRY)
        if isinstance(geom, (QtCore.QByteArray, bytes, bytearray)):
            self.restoreGeometry(QtCore.QByteArray(geom))
        state = s.value(_QSK_WINDOW_STATE)
        if isinstance(state, (QtCore.QByteArray, bytes, bytearray)):
            self.restoreState(QtCore.QByteArray(state))
        sizes = s.value(_QSK_SPLITTER_SIZES)
        if isinstance(sizes, (list, tuple)) and len(sizes) >= 3:
            try:
                self._splitter.setSizes([int(x) for x in sizes])
            except (TypeError, ValueError):                # pragma: no cover
                pass

        # Restore the last-active mode and apply it (unlocked — the user
        # can still let auto-detect take over by re-loading a model).
        last_mode_name = s.value(_QSK_LAST_MODE)
        if isinstance(last_mode_name, str) and last_mode_name and _CHARACTER_MODE_AVAILABLE:
            try:
                restored = CharacterMode[last_mode_name]
            except KeyError:                               # pragma: no cover
                restored = None
            if restored is not None:
                self._apply_mode(restored, locked=False, source="qsettings")

    def _save_settings(self) -> None:
        s = self._settings()
        s.setValue(_QSK_GEOMETRY, self.saveGeometry())
        s.setValue(_QSK_WINDOW_STATE, self.saveState())
        s.setValue(_QSK_SPLITTER_SIZES, list(self._splitter.sizes()))
        mode = getattr(self.scene, "mode", None)
        mode_name = getattr(mode, "name", "") or ""
        s.setValue(_QSK_LAST_MODE, mode_name)
        s.sync()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        controller = getattr(self, "head_builder_controller", None)
        if (
            controller is not None
            and controller.requires_save_prompt
            and not controller.confirm_discard_or_save(
                "The Custom Head project has unsaved changes. Save before closing?"
            )
        ):
            event.ignore()
            return
        if not self._confirm_discard_or_save(
            "The scene has unsaved changes. Save before closing?"
        ):
            event.ignore()
            return
        try:
            self._save_settings()
        except Exception:                                  # pragma: no cover
            log.exception("Failed to persist QSettings")
        event.accept()
