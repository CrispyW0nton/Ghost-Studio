"""GhostRigger Map Studio Level Editor window."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
from time import perf_counter
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from src.core.level import KMapProject, KMapSerializer, LevelScene, LevelTransform, resolve_project_texture_path
from src.core.modules.authored_module_export import authored_module_smoke_summary_lines
from src.core.modules.module_editor_controller import MapStudioTextureCloneCancelled, ModuleEditorController
from src.core.modules.module_editor_model import ModuleEditorModel
from src.core.modules.map_studio_tool_action_dispatch import (
    MapStudioToolActionContext,
    execute_map_studio_tool_belt_action,
    resolve_map_studio_tool_belt_action,
)
from src.gui.panels.module_editor.blueprints_tab import BlueprintsTab
from src.gui.panels.module_editor.builder_tab import BuilderTab
from src.gui.panels.module_editor.environment_tab import MapStudioEnvironmentTab
from src.gui.panels.module_editor.environment_kit_browser import EnvironmentKitBrowser
from src.gui.panels.module_editor.export_panel import ModuleExportPanel
from src.core.modules.map_studio_hover_context import map_studio_hover_context_summary
from src.core.modules.map_studio_texture_paint import TexturePaintSession, decode_image_rgba
from src.core.modules.map_studio_marking_menu_registry import (
    map_studio_marking_menu_action,
    map_studio_marking_menu_tree_for_hover,
)
from src.gui.panels.module_editor.component_marking_menu import MapStudioComponentMarkingMenu
from src.gui.dialogs.map_studio_obj_room_import_dialog import MapStudioObjRoomImportDialog
from src.gui.dialogs.map_studio_panorama_skybox_dialog import MapStudioPanoramaSkyboxDialog
from src.gui.panels.module_editor.texture_browser_dialog import MapStudioTextureBrowserDialog
from src.gui.panels.module_editor.texture_paint_tab import MapStudioTexturePaintTab
from src.core.modules.authored_imported_mesh import (
    ImportedMeshRoomPrimitive,
    ImportedMeshSurface,
    imported_mesh_surface_index_for_role,
    imported_mesh_surface_role,
    resolve_imported_mesh_face_target,
)
from src.core.modules.map_studio_live_topology_session import MapStudioLiveTopologySession
from src.core.modules.map_studio_multi_cut import (
    MultiCutSession,
    MultiCutSettings,
    anchor_from_surface_hit,
)
from src.gui.panels.module_editor.module_editor_asset_browser import ModuleEditorAssetBrowser
from src.gui.panels.module_editor.module_editor_outliner import ModuleEditorOutliner, authored_primitive_item_id
from src.gui.panels.module_editor.module_editor_properties import MapStudioPIEContextPanel, ModuleEditorPropertiesPanel
from src.gui.panels.module_editor.module_editor_toolbar import ModuleEditorToolbar
from src.gui.panels.module_editor.module_editor_viewport_panel import ModuleEditorViewportPanel
from src.gui.panels.module_editor.porter_tab import PorterTab
from src.gui.panels.module_editor.placement_tab import PlacementTab
from src.gui.panels.module_editor.terrain_kit_browser import TerrainKitBrowser
from src.gui.panels.module_editor.readiness_panel import ModuleReadinessPanel
from src.gui.panels.module_editor.rooms_tab import RoomsTab
from src.gui.panels.module_editor.validation_panel import ModuleValidationPanel
from src.gui.panels.module_editor.walkmesh_tab import WalkmeshTab
from src.gui.panels.module_editor.workflow_panel import MapStudioWorkflowPanel
from src.gui.qt_lib.assets.qt_theme import make_horizontal_overflow_area, make_scrollable_panel
from src.core.rendering.renderer_settings import RendererSettings
from src.core.rendering.viewport_navigation import DEFAULT_VIEWPORT_NAVIGATION_PROFILE


_MAP_STUDIO_VALIDATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-validation",
)

_MAP_STUDIO_TERRAIN_CATALOG_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-terrain-catalog",
)

_MAP_STUDIO_ENVIRONMENT_THUMBNAIL_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-environment-thumbnail",
)

_MAP_STUDIO_PIE_ACTOR_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-pie-actors",
)

# Play may briefly join a body/head prewarm that is already at its publication
# boundary.  Keep this bounded so a long pure-Python MDL parse never freezes
# the editor; an unfinished actor falls back to the simulation marker for that
# run instead of launching a duplicate parse on the GUI thread.
_MAP_STUDIO_PIE_PLAYER_PREWARM_WAIT_SECONDS = 0.20

_MAP_STUDIO_TEXTURE_CLONE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-texture-clone",
)

_MAP_STUDIO_OBJ_IMPORT_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-obj-import",
)

_MAP_STUDIO_SKYBOX_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="map-studio-skybox",
)

# NPC idle poses are intentionally sampled below the player/render cadence.
# A stock area such as 207TEL can retain 32 animated creatures (roughly 2,800
# DAG nodes); evaluating every skeleton in one 30 Hz burst consumed about
# 17 ms on the reference fixture before the renderer did any work.  Twelve
# samples per second remains smooth at the viewport's practical frame rate,
# while round-robin cohorts distribute that work instead of producing a large
# every-other-frame hitch.
_MAP_STUDIO_PIE_CREATURE_POSE_HZ = 12.0
# The PIE timer targets 60 Hz.  Pose work may make a real frame longer than
# that, but feeding the longer frame time back into the pose quota creates a
# self-paced catch-up spiral (more poses -> slower frame -> still more poses).
# Keep the quota tied to one intended timer slice; each actor separately keeps
# its real elapsed time until its round-robin turn.
_MAP_STUDIO_PIE_CREATURE_SCHEDULER_HZ = 60.0


def _stamp_map_studio_pie_actor_pose(pose: Any, actor: Any, animation_name: str) -> Any:
    """Attach the actor/clip identity required by retained BAS head rendering.

    The viewport row also carries ``animation_name``, but renderer-neutral BAS
    pose resolution consumes the pose itself.  Keeping both in sync lets a
    detachable head sample its own inherited hierarchy while the body pose
    continues to place the animated ``headhook`` socket.
    """

    if pose is None or actor is None:
        return pose
    source_model = getattr(actor, "source_model", None)
    setattr(pose, "_gr_animation_scene_object_id", str(getattr(actor, "actor_id", "") or ""))
    setattr(pose, "_gr_animation_source_model_id", id(source_model))
    setattr(pose, "_gr_animation_source_model_name", str(getattr(source_model, "name", "") or ""))
    setattr(pose, "_gr_animation_name", str(animation_name or ""))
    return pose


class _MapStudioWorkflowStack(QtWidgets.QStackedWidget):
    """Compact tab-compatible workflow stack sized from only its active page.

    Map Studio historically put eight workflows in a narrow ``QTabWidget``.
    Besides clipping the tab labels, ``QTabWidget`` reports the largest child
    as every page's preferred size.  The Builder and Environment pages could
    therefore force horizontal overflow while the much smaller Rooms or WOK
    pages were active.  This stack preserves the small tab API used by the
    window while leaving workflow choice to the accessible combo box above it.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tab_labels: list[str] = []
        self._tab_tooltips: dict[int, str] = {}
        self._height_sync_pending = False
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setMinimumHeight(0)
        # QStackedLayout normally contributes the largest hidden page to the
        # container's minimum size.  In Map Studio that made the compact Paint
        # page inherit the very tall Builder page and exposed thousands of
        # pixels of empty vertical scroll.  The stack reports the active page's
        # size explicitly, so its internal layout must not reintroduce the
        # hidden-page constraint.
        layout = self.layout()
        if layout is not None:
            layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.currentChanged.connect(self._active_page_changed)

    def addTab(self, widget: QtWidgets.QWidget, label: str) -> int:  # noqa: N802 - QTabWidget compatibility
        index = self.addWidget(widget)
        self._tab_labels.append(str(label))
        widget.setMinimumWidth(0)
        widget.installEventFilter(self)
        return index

    def tabText(self, index: int) -> str:  # noqa: N802 - QTabWidget compatibility
        return self._tab_labels[index] if 0 <= index < len(self._tab_labels) else ""

    def setTabToolTip(self, index: int, text: str) -> None:  # noqa: N802 - QTabWidget compatibility
        if 0 <= index < self.count():
            self._tab_tooltips[index] = str(text)

    def tabToolTip(self, index: int) -> str:  # noqa: N802 - QTabWidget compatibility
        return self._tab_tooltips.get(index, "")

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        page = self.currentWidget()
        if page is None:
            return QtCore.QSize(300, 200)
        hint = page.sizeHint()
        # The rail owns width.  Only the active page contributes height.
        return QtCore.QSize(min(320, max(0, hint.width())), max(0, hint.height()))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        page = self.currentWidget()
        height = 0 if page is None else max(0, page.minimumSizeHint().height())
        return QtCore.QSize(0, height)

    def _active_page_changed(self, _index: int) -> None:
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
        self._queue_active_page_height_sync()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is self.currentWidget() and event.type() == QtCore.QEvent.LayoutRequest:
            self._queue_active_page_height_sync()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._queue_active_page_height_sync()

    def _queue_active_page_height_sync(self) -> None:
        if self._height_sync_pending:
            return
        self._height_sync_pending = True
        QtCore.QTimer.singleShot(0, self._sync_active_page_height)

    def _sync_active_page_height(self) -> None:
        self._height_sync_pending = False
        page = self.currentWidget()
        if page is None:
            return
        page_layout = page.layout()
        if page_layout is not None:
            page_layout.invalidate()
        wanted = max(1, page.minimumSizeHint().height(), page.sizeHint().height())
        if page.hasHeightForWidth():
            wanted = max(wanted, page.heightForWidth(max(1, self.width())))
        if self.maximumHeight() != wanted:
            self.setMaximumHeight(wanted)
        if self.height() != wanted:
            self.resize(self.width(), wanted)
        self.updateGeometry()


def _build_map_studio_geometry_validation_snapshot(project_data: dict[str, Any]) -> dict[str, Any]:
    """Run expensive, read-only readiness checks away from the Qt thread."""

    started = perf_counter()
    project = KMapSerializer.from_dict(project_data)
    controller = ModuleEditorController(ModuleEditorModel(project=project))
    readiness_result = controller.authored_module_readiness()
    # Terrain overlay validation serializes the generated WOK so it can audit
    # the raw perimeter records.  Keep that work in this existing validation
    # worker; doing it directly from mouse release caused a visible hitch even
    # for the default 17x17 terrain patch.
    terrain_walkability_overlay = controller.authored_terrain_walkability_overlay()
    return {
        "readiness": readiness_result.readiness,
        "issues": controller.validate(readiness_result=readiness_result),
        "walkmesh_status": controller.authored_walkmesh_status(),
        "walkmesh_room_surfaces": controller.authored_walkmesh_room_surface_choices(),
        "terrain_walkability_overlay": terrain_walkability_overlay,
        "terrain_room_choices": controller.authored_terrain_room_choices(),
        "elapsed_ms": (perf_counter() - started) * 1000.0,
    }


class _MapStudioGameProofDialog(QtWidgets.QDialog):
    """Collect manual KOTOR smoke-test proof before marking a module tested."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        proof_manifest_path: str = "",
        package_resource_summary: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record Map Studio Game Proof")
        self.setModal(True)
        self.setObjectName("mapStudioGameProofDialog")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.proof_manifest_edit = QtWidgets.QLineEdit(proof_manifest_path)
        self.proof_manifest_edit.setObjectName("mapStudioProofManifestLineEdit")
        proof_browse = QtWidgets.QPushButton("Browse...")
        proof_browse.setObjectName("mapStudioProofManifestBrowseButton")
        proof_row = QtWidgets.QHBoxLayout()
        proof_row.addWidget(self.proof_manifest_edit, 1)
        proof_row.addWidget(proof_browse)
        form.addRow("Proof manifest", proof_row)

        self.package_resource_label = QtWidgets.QLabel(
            package_resource_summary
            or "Package inventory: stage or install the authored module before recording proof."
        )
        self.package_resource_label.setObjectName("mapStudioProofPackageResourceSummaryLabel")
        self.package_resource_label.setWordWrap(True)
        layout.addWidget(self.package_resource_label)

        self.evidence_edit = QtWidgets.QLineEdit()
        self.evidence_edit.setObjectName("mapStudioProofEvidenceLineEdit")
        evidence_browse = QtWidgets.QPushButton("Browse...")
        evidence_browse.setObjectName("mapStudioProofEvidenceBrowseButton")
        evidence_row = QtWidgets.QHBoxLayout()
        evidence_row.addWidget(self.evidence_edit, 1)
        evidence_row.addWidget(evidence_browse)
        form.addRow("Screenshot/video", evidence_row)

        self.tester_edit = QtWidgets.QLineEdit()
        self.tester_edit.setObjectName("mapStudioProofTesterLineEdit")
        form.addRow("Tester", self.tester_edit)

        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setObjectName("mapStudioProofNotesEdit")
        self.notes_edit.setMaximumHeight(90)
        form.addRow("Notes", self.notes_edit)

        checks_box = QtWidgets.QGroupBox("KOTOR in-game acceptance checks")
        checks_box.setObjectName("mapStudioProofChecksGroupBox")
        checks_layout = QtWidgets.QVBoxLayout(checks_box)
        self.module_loads_box = QtWidgets.QCheckBox("`warp` loads the generated module in KOTOR")
        self.module_loads_box.setObjectName("mapStudioProofModuleLoadsCheckBox")
        self.module_identity_box = QtWidgets.QCheckBox("Loaded module identity matches the authored resref")
        self.module_identity_box.setObjectName("mapStudioProofModuleIdentityCheckBox")
        self.player_floor_box = QtWidgets.QCheckBox("Player appears on the generated floor, not in void")
        self.player_floor_box.setObjectName("mapStudioProofPlayerFloorCheckBox")
        self.placeable_visible_box = QtWidgets.QCheckBox("Authored/test placeable appears where expected")
        self.placeable_visible_box.setObjectName("mapStudioProofPlaceableVisibleCheckBox")
        self.walkable_floor_box = QtWidgets.QCheckBox("Player can walk across the generated floor")
        self.walkable_floor_box.setObjectName("mapStudioProofWalkableFloorCheckBox")
        self.transition_pathing_box = QtWidgets.QCheckBox("Transitions and pathing behave sanely in the loaded module")
        self.transition_pathing_box.setObjectName("mapStudioProofTransitionPathingCheckBox")
        self.no_inherited_box = QtWidgets.QCheckBox("No inherited vanilla geometry or scripted movers appear")
        self.no_inherited_box.setObjectName("mapStudioProofNoInheritedContentCheckBox")
        self.allow_missing_evidence_box = QtWidgets.QCheckBox("Record incomplete attempt if evidence file is missing")
        self.allow_missing_evidence_box.setObjectName("mapStudioProofAllowMissingEvidenceCheckBox")
        for widget in (
            self.module_loads_box,
            self.module_identity_box,
            self.player_floor_box,
            self.placeable_visible_box,
            self.walkable_floor_box,
            self.transition_pathing_box,
            self.no_inherited_box,
            self.allow_missing_evidence_box,
        ):
            checks_layout.addWidget(widget)
        layout.addWidget(checks_box)

        self.plcaa_checks_box = QtWidgets.QGroupBox("Custom plcaa end-to-end checks")
        self.plcaa_checks_box.setObjectName("mapStudioPlcaaProofChecksGroupBox")
        plcaa_layout = QtWidgets.QVBoxLayout(self.plcaa_checks_box)
        plcaa_check_specs = (
            ("texture_paint_box", "Painted textures are visible on the staged surfaces in KOTOR", "mapStudioProofTexturePaintCheckBox"),
            ("terrain_walkmesh_box", "Sculpted terrain and its generated WOK work in KOTOR", "mapStudioProofTerrainWalkmeshCheckBox"),
            ("staged_assets_box", "Placed assets match their Map Studio position and orientation", "mapStudioProofStagedAssetsCheckBox"),
            ("enemy_hostile_box", "Placed enemy spawns and attacks the player", "mapStudioProofEnemyHostileCheckBox"),
            ("npc_roam_box", "Placed friendly NPC spawns and free-roams", "mapStudioProofNpcRoamCheckBox"),
            ("terminal_box", "Placed terminal can be used and performs its configured action", "mapStudioProofTerminalCheckBox"),
            ("container_box", "Placed container opens and contains its configured inventory", "mapStudioProofContainerCheckBox"),
            ("puzzle_box", "The staged 1-2-3 puzzle unlocks its reward door", "mapStudioProofPuzzleCheckBox"),
            ("animated_door_box", "Placed animated door opens and closes correctly", "mapStudioProofAnimatedDoorCheckBox"),
            ("configured_transition_box", "Configured door/trigger transition reaches its destination", "mapStudioProofConfiguredTransitionCheckBox"),
            ("player_start_match_box", "Player start position and facing match Map Studio", "mapStudioProofPlayerStartMatchCheckBox"),
        )
        for attribute, label, object_name in plcaa_check_specs:
            widget = QtWidgets.QCheckBox(label)
            widget.setObjectName(object_name)
            setattr(self, attribute, widget)
            plcaa_layout.addWidget(widget)
        layout.addWidget(self.plcaa_checks_box)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.setObjectName("mapStudioProofButtons")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        proof_browse.clicked.connect(self._browse_proof_manifest)
        evidence_browse.clicked.connect(self._browse_evidence)
        self.proof_manifest_edit.textChanged.connect(self._sync_plcaa_check_visibility)
        self._sync_plcaa_check_visibility()

    def values(self) -> dict[str, Any]:
        return {
            "proof_manifest_path": self.proof_manifest_edit.text().strip(),
            "evidence_path": self.evidence_edit.text().strip(),
            "tester": self.tester_edit.text().strip(),
            "notes": self.notes_edit.toPlainText().strip(),
            "module_loads_in_game": self.module_loads_box.isChecked(),
            "module_identity_matches_authored_resref": self.module_identity_box.isChecked(),
            "player_spawns_on_floor": self.player_floor_box.isChecked(),
            "test_placeable_visible": self.placeable_visible_box.isChecked(),
            "player_can_walk_on_floor": self.walkable_floor_box.isChecked(),
            "transition_pathing_sanity_confirmed": self.transition_pathing_box.isChecked(),
            "no_inherited_base_game_geometry_or_scripted_movers": self.no_inherited_box.isChecked(),
            "texture_paint_visible_in_game": self.texture_paint_box.isChecked(),
            "terrain_sculpt_and_generated_walkmesh_work_in_game": self.terrain_walkmesh_box.isChecked(),
            "placed_assets_match_editor_staging": self.staged_assets_box.isChecked(),
            "enemy_spawns_hostile": self.enemy_hostile_box.isChecked(),
            "npc_spawns_and_free_roams": self.npc_roam_box.isChecked(),
            "terminal_operates": self.terminal_box.isChecked(),
            "container_opens_with_inventory": self.container_box.isChecked(),
            "puzzle_sequence_unlocks_door": self.puzzle_box.isChecked(),
            "animated_door_operates": self.animated_door_box.isChecked(),
            "configured_transition_operates": self.configured_transition_box.isChecked(),
            "player_start_position_and_facing_match": self.player_start_match_box.isChecked(),
            "allow_missing_evidence": self.allow_missing_evidence_box.isChecked(),
        }

    def _sync_plcaa_check_visibility(self, _text: str = "") -> None:
        """Show the detailed gate only for the custom plcaa proof manifest."""

        module_root = ""
        path = self.proof_manifest_edit.text().strip()
        if path:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    module_root = str(payload.get("module_root") or "").strip().lower()
            except Exception:
                module_root = ""
        self.plcaa_checks_box.setVisible(module_root == "plcaa")

    def _browse_proof_manifest(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Map Studio proof manifest",
            self.proof_manifest_edit.text().strip(),
            "Proof manifest (*.json);;All files (*.*)",
        )
        if path:
            self.proof_manifest_edit.setText(path)

    def _browse_evidence(self) -> None:
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select KOTOR screenshot or video evidence",
            self.evidence_edit.text().strip(),
            "Evidence (*.png *.jpg *.jpeg *.bmp *.mp4 *.mov *.mkv);;All files (*.*)",
        )
        if path:
            self.evidence_edit.setText(path)


class _MapStudioLytResourceDialog(QtWidgets.QDialog):
    """Choose an indexed area layout, preferring its complete stock module."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] = ()) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Stock Module / LYT")
        self.setModal(True)
        self.setObjectName("mapStudioLytResourceDialog")
        self.resize(620, 460)
        self._rows = [dict(row) for row in rows]
        self._filtered_rows: list[dict[str, Any]] = []

        root = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setObjectName("mapStudioLytResourceSearchLineEdit")
        self.search_edit.setPlaceholderText("Filter resrefs")
        self.game_combo = QtWidgets.QComboBox(self)
        self.game_combo.setObjectName("mapStudioLytResourceGameComboBox")
        self.game_combo.addItems(["All Games", "K1", "K2"])
        controls.addWidget(self.search_edit, 1)
        controls.addWidget(self.game_combo)
        root.addLayout(controls)

        self.resource_list = QtWidgets.QListWidget(self)
        self.resource_list.setObjectName("mapStudioLytResourceListWidget")
        self.resource_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        root.addWidget(self.resource_list, 1)

        self.detail_label = QtWidgets.QLabel(self)
        self.detail_label.setObjectName("mapStudioLytResourceDetailLabel")
        self.detail_label.setWordWrap(True)
        root.addWidget(self.detail_label)

        self.hydration_label = QtWidgets.QLabel(
            "When a matching module capsule is installed, Map Studio loads the complete area: "
            "ARE/GIT/IFO, player start, gameplay objects, room models, textures, and lightmaps. "
            "If no capsule can be matched, it will say explicitly that only the LYT layout was loaded."
        )
        self.hydration_label.setObjectName("mapStudioLytResourceHydrationLabel")
        self.hydration_label.setWordWrap(True)
        root.addWidget(self.hydration_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        buttons.setObjectName("mapStudioLytResourceButtons")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.search_edit.textChanged.connect(self._apply_filter)
        self.game_combo.currentTextChanged.connect(self._apply_filter)
        self.resource_list.itemSelectionChanged.connect(self._update_detail)
        self.resource_list.itemDoubleClicked.connect(lambda _item: self.accept())
        self._apply_filter()

    def selected_row(self) -> dict[str, Any] | None:
        item = self.resource_list.currentItem()
        if item is None:
            return None
        row = item.data(QtCore.Qt.UserRole)
        return dict(row) if isinstance(row, dict) else None

    def accept(self) -> None:
        if self.selected_row() is None:
            QtWidgets.QMessageBox.information(self, "Load Stock Module / LYT", "Select an area resource to load.")
            return
        super().accept()

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        game_filter = self.game_combo.currentText().strip()
        self.resource_list.clear()
        self._filtered_rows = []
        for row in self._rows:
            game = str(row.get("game", "") or "").upper()
            resref = str(row.get("resref", "") or "").lower()
            if game_filter in {"K1", "K2"} and game != game_filter:
                continue
            if query and query not in resref and query not in game.lower():
                continue
            self._filtered_rows.append(row)
            item = QtWidgets.QListWidgetItem(self._row_label(row))
            item.setData(QtCore.Qt.UserRole, dict(row))
            item.setToolTip(self._row_detail(row))
            self.resource_list.addItem(item)
        if self.resource_list.count() > 0:
            self.resource_list.setCurrentRow(0)
        self._update_detail()

    def _update_detail(self) -> None:
        row = self.selected_row()
        if row is None:
            if self._rows:
                self.detail_label.setText("No indexed LYT resource matches the current filter.")
            else:
                self.detail_label.setText("No indexed LYT resources were found in the configured game directories.")
            return
        self.detail_label.setText(self._row_detail(row))

    @staticmethod
    def _row_label(row: dict[str, Any]) -> str:
        game = str(row.get("game", "") or "?").upper()
        resref = str(row.get("resref", "") or "<unknown>").lower()
        room_count = int(row.get("room_count", 0) or 0)
        doorhook_count = int(row.get("doorhook_count", 0) or 0)
        return f"[{game}] {resref}    {room_count} room(s), {doorhook_count} doorhook(s)"

    @staticmethod
    def _row_detail(row: dict[str, Any]) -> str:
        game = str(row.get("game", "") or "?").upper()
        resref = str(row.get("resref", "") or "<unknown>").lower()
        source = str(row.get("source", "") or "configured game resources")
        room_count = int(row.get("room_count", 0) or 0)
        doorhook_count = int(row.get("doorhook_count", 0) or 0)
        return f"{game}:{resref}.lyt from {source} - {room_count} room(s), {doorhook_count} doorhook(s)"


class _MapStudioLaunchHandoffDialog(QtWidgets.QDialog):
    """Show the exact manual warp-test handoff before opening KOTOR."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        warp_command: str,
        launcher_path: str = "",
        proof_manifest_path: str = "",
        proof_recording_script_path: str = "",
        launch_helper_command: str = "",
        package_resource_summary: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map Studio Warp Test Handoff")
        self.setModal(True)
        root = QtWidgets.QVBoxLayout(self)

        self.warning_label = QtWidgets.QLabel(
            "Launching KOTOR is not game proof. After the game opens, run the exact warp command, "
            "verify spawn/walk/placeables in-game, then record proof with screenshot or video evidence."
        )
        self.warning_label.setObjectName("mapStudioLaunchHandoffWarningLabel")
        self.warning_label.setWordWrap(True)
        root.addWidget(self.warning_label)

        self.package_resource_label = QtWidgets.QLabel(
            package_resource_summary
            or "Package inventory: stage or install the authored module before launch handoff."
        )
        self.package_resource_label.setObjectName("mapStudioLaunchPackageResourceSummaryLabel")
        self.package_resource_label.setWordWrap(True)
        root.addWidget(self.package_resource_label)

        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.warp_command_edit = QtWidgets.QLineEdit(warp_command)
        self.warp_command_edit.setObjectName("mapStudioLaunchWarpCommandLineEdit")
        self.warp_command_edit.setReadOnly(True)
        copy_warp_button = QtWidgets.QPushButton("Copy")
        copy_warp_button.setObjectName("mapStudioLaunchCopyWarpCommandButton")
        warp_row = QtWidgets.QHBoxLayout()
        warp_row.addWidget(self.warp_command_edit, 1)
        warp_row.addWidget(copy_warp_button)
        form.addRow("Run this exact KOTOR console command", warp_row)

        self.launcher_path_edit = QtWidgets.QLineEdit(launcher_path)
        self.launcher_path_edit.setObjectName("mapStudioLaunchScriptPathLineEdit")
        self.launcher_path_edit.setReadOnly(True)
        form.addRow("Launch script", self.launcher_path_edit)

        self.proof_manifest_edit = QtWidgets.QLineEdit(proof_manifest_path)
        self.proof_manifest_edit.setObjectName("mapStudioLaunchProofManifestLineEdit")
        self.proof_manifest_edit.setReadOnly(True)
        form.addRow("Proof manifest", self.proof_manifest_edit)

        self.proof_recorder_edit = QtWidgets.QLineEdit(proof_recording_script_path)
        self.proof_recorder_edit.setObjectName("mapStudioLaunchProofRecorderLineEdit")
        self.proof_recorder_edit.setReadOnly(True)
        form.addRow("Proof recorder", self.proof_recorder_edit)

        self.helper_command_edit = QtWidgets.QPlainTextEdit(launch_helper_command)
        self.helper_command_edit.setObjectName("mapStudioLaunchHelperCommandEdit")
        self.helper_command_edit.setReadOnly(True)
        self.helper_command_edit.setMaximumHeight(64)
        form.addRow("CLI helper", self.helper_command_edit)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(
            "Open Launcher" if launcher_path else "Open Proof Folder"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        copy_warp_button.clicked.connect(
            lambda: QtGui.QGuiApplication.clipboard().setText(self.warp_command_edit.text())
        )


class _MapStudioPackageWizardDialog(QtWidgets.QDialog):
    """Review authored-module package targets before staging or installing."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        mode: str,
        readiness: object | None,
        output_dir: str = "",
        game_modules_dir: str = "",
        dry_run: bool = True,
    ) -> None:
        super().__init__(parent)
        self._readiness = readiness
        self._mode = str(mode or "stage").strip().lower()
        self.setWindowTitle("Map Studio Package Wizard")
        self.setModal(True)
        self.setObjectName("mapStudioPackageWizardDialog")

        root = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel(self._summary_text())
        self.summary_label.setObjectName("mapStudioPackageWizardSummaryLabel")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.module_root_edit = QtWidgets.QLineEdit(str(getattr(readiness, "module_root", "") or ""))
        self.module_root_edit.setObjectName("mapStudioPackageWizardModuleRootLineEdit")
        self.module_root_edit.setReadOnly(True)
        form.addRow("Module root", self.module_root_edit)

        self.game_edit = QtWidgets.QLineEdit(str(getattr(readiness, "game", "") or ""))
        self.game_edit.setObjectName("mapStudioPackageWizardGameLineEdit")
        self.game_edit.setReadOnly(True)
        form.addRow("Target game", self.game_edit)

        self.capability_edit = QtWidgets.QLineEdit(str(getattr(readiness, "capability_stage", "") or "not_checked"))
        self.capability_edit.setObjectName("mapStudioPackageWizardCapabilityLineEdit")
        self.capability_edit.setReadOnly(True)
        form.addRow("Capability", self.capability_edit)

        self.output_dir_edit = QtWidgets.QLineEdit(output_dir)
        self.output_dir_edit.setObjectName("mapStudioPackageWizardOutputDirLineEdit")
        output_browse = QtWidgets.QPushButton("Browse...")
        output_browse.setObjectName("mapStudioPackageWizardOutputBrowseButton")
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(output_browse)
        form.addRow("Stage/package folder", output_row)

        self.install_check = QtWidgets.QCheckBox("Install/copy .mod to a KOTOR Modules folder after staging")
        self.install_check.setObjectName("mapStudioPackageWizardInstallCheckBox")
        self.install_check.setChecked(self._mode == "install")
        self.install_check.setEnabled(self._mode == "install")
        root.addWidget(self.install_check)

        self.modules_dir_edit = QtWidgets.QLineEdit(game_modules_dir)
        self.modules_dir_edit.setObjectName("mapStudioPackageWizardModulesDirLineEdit")
        modules_browse = QtWidgets.QPushButton("Browse...")
        modules_browse.setObjectName("mapStudioPackageWizardModulesBrowseButton")
        modules_row = QtWidgets.QHBoxLayout()
        modules_row.addWidget(self.modules_dir_edit, 1)
        modules_row.addWidget(modules_browse)
        form.addRow("KOTOR Modules folder", modules_row)

        self.dry_run_check = QtWidgets.QCheckBox("Dry run: preview validation and targets without final writes")
        self.dry_run_check.setObjectName("mapStudioPackageWizardDryRunCheckBox")
        self.dry_run_check.setChecked(bool(dry_run))
        root.addWidget(self.dry_run_check)

        self.overwrite_check = QtWidgets.QCheckBox("Back up and replace existing module package if needed")
        self.overwrite_check.setObjectName("mapStudioPackageWizardOverwriteCheckBox")
        root.addWidget(self.overwrite_check)

        self.no_partial_write_label = QtWidgets.QLabel(
            "Writes use the authored-module ExportJob staging path: package files, checklist, and proof manifest are reviewed before game-ready proof is recorded."
        )
        self.no_partial_write_label.setObjectName("mapStudioPackageWizardNoPartialWriteLabel")
        self.no_partial_write_label.setWordWrap(True)
        root.addWidget(self.no_partial_write_label)

        self.resource_table = QtWidgets.QTableWidget(0, 3)
        self.resource_table.setObjectName("mapStudioPackageWizardResourceReviewTable")
        self.resource_table.setHorizontalHeaderLabels(("Resource or reference", "Status", "Why it matters"))
        self.resource_table.verticalHeader().setVisible(False)
        self.resource_table.horizontalHeader().setStretchLastSection(True)
        self.resource_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.resource_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.resource_table.setMinimumHeight(160)
        root.addWidget(self.resource_table)
        self._populate_resource_table()

        self.proof_gate_table = QtWidgets.QTableWidget(0, 2)
        self.proof_gate_table.setObjectName("mapStudioPackageWizardProofGateTable")
        self.proof_gate_table.setHorizontalHeaderLabels(("Live KOTOR proof check", "Package gate status"))
        self.proof_gate_table.verticalHeader().setVisible(False)
        self.proof_gate_table.horizontalHeader().setStretchLastSection(True)
        self.proof_gate_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.proof_gate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.proof_gate_table.setMinimumHeight(120)
        root.addWidget(self.proof_gate_table)
        self._populate_proof_gate_table()

        self.blocking_label = QtWidgets.QLabel(self._blocking_text())
        self.blocking_label.setObjectName("mapStudioPackageWizardBlockingLabel")
        self.blocking_label.setWordWrap(True)
        root.addWidget(self.blocking_label)

        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.setObjectName("mapStudioPackageWizardButtons")
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(self._ok_text())
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        output_browse.clicked.connect(self._browse_output_dir)
        modules_browse.clicked.connect(self._browse_modules_dir)
        self.install_check.toggled.connect(self._sync_install_controls)
        self.modules_dir_edit.textChanged.connect(lambda _text: self._sync_install_controls())
        self.module_root_edit.textChanged.connect(lambda _text: self._sync_install_controls())
        self.dry_run_check.toggled.connect(lambda _checked: self._sync_install_controls())
        self._sync_install_controls()

    def values(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir_edit.text().strip(),
            "game_modules_dir": self.modules_dir_edit.text().strip(),
            "dry_run": self.dry_run_check.isChecked(),
            "install_requested": self.install_check.isChecked(),
            "overwrite": self.overwrite_check.isChecked(),
        }

    def _summary_text(self) -> str:
        action = {
            "export": "Export an authored .mod package candidate.",
            "stage": "Stage an authored .mod package, checklist, and proof manifest for a KOTOR warp test.",
            "install": "Stage and install an authored .mod package for a KOTOR warp test.",
        }.get(self._mode, "Stage an authored .mod package for a KOTOR warp test.")
        return (
            f"{action} Review ARE/GIT/IFO/LYT/VIS/PTH, room MDL/MDX/WOK, install target, "
            "and proof handoff before anything is written."
        )

    def _ok_text(self) -> str:
        if self._mode == "export":
            return "Export Candidate"
        if self._mode == "install":
            return "Stage and Install"
        return "Stage Package"

    def _blocking_text(self) -> str:
        if self._readiness is None:
            return "Package gate: readiness has not been checked. Create/open a KMAP and validate before packaging."
        blockers = tuple(str(item) for item in tuple(getattr(self._readiness, "blocking_messages", ()) or ()) if str(item).strip())
        if blockers:
            return "Package gate blockers: " + " | ".join(blockers[:3])
        export_status = str(getattr(self._readiness, "export_status", "") or "")
        return f"Package gate: {export_status or 'No blocking readiness issues reported.'}"

    def _populate_resource_table(self) -> None:
        readiness = self._readiness
        expected = tuple(getattr(readiness, "expected_runtime_resources", ()) or ()) if readiness is not None else ()
        present = {self._resource_label(key) for key in tuple(getattr(readiness, "present_runtime_resources", ()) or ())}
        missing = {self._resource_label(key) for key in tuple(getattr(readiness, "missing_runtime_resources", ()) or ())}
        rows: list[tuple[str, str, str]] = []
        for key in expected:
            label = self._resource_label(key)
            if label in present:
                status = "current in KMAP package metadata"
            elif label in missing:
                status = "will be generated/staged by package build"
            else:
                status = "expected runtime output"
            rows.append((label, status, self._resource_reason(label)))
        metadata = dict(getattr(readiness, "metadata", {}) or {}) if readiness is not None else {}
        rows.extend(self._reference_rows_from_metadata(metadata))
        if not rows:
            rows.append(("No runtime resources listed", "not ready", "Create or validate an authored module before packaging."))
        self.resource_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.resource_table.setItem(row, column, item)

    def _populate_proof_gate_table(self) -> None:
        metadata = dict(getattr(self._readiness, "metadata", {}) or {}) if self._readiness is not None else {}
        test_plan = dict(metadata.get("modder_test_plan") or {}) if isinstance(metadata.get("modder_test_plan"), dict) else {}
        checks = tuple(str(item) for item in tuple(test_plan.get("acceptance_checks") or ()) if str(item).strip())
        missing = {str(item) for item in tuple(test_plan.get("missing_acceptance_checks") or checks)}
        if not checks:
            checks = (
                "module_loads_in_game",
                "module_identity_matches_authored_resref",
                "player_spawns_on_floor",
                "test_placeable_visible",
                "player_can_walk_on_floor",
                "transition_pathing_sanity_confirmed",
                "no_inherited_base_game_geometry_or_scripted_movers",
                "screenshot_or_video_captured",
            )
            missing = set(checks)
        rows = tuple((self._proof_check_label(check), self._proof_check_status(check, missing)) for check in checks)
        self.proof_gate_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.proof_gate_table.setItem(row, column, item)

    @staticmethod
    def _proof_check_label(check: str) -> str:
        return {
            "module_loads_in_game": "`warp` loads the generated module in KOTOR",
            "module_identity_matches_authored_resref": "Loaded module identity matches the authored resref",
            "player_spawns_on_floor": "Player appears on the generated floor",
            "test_placeable_visible": "Authored/test placeable appears where expected",
            "player_can_walk_on_floor": "Player can walk across generated WOK",
            "transition_pathing_sanity_confirmed": "Transitions and PTH pathing behave sanely",
            "no_inherited_base_game_geometry_or_scripted_movers": "No inherited vanilla geometry or scripted movers appear",
            "screenshot_or_video_captured": "Screenshot or video evidence is attached",
            "texture_paint_visible_in_game": "Painted textures are visible on the staged surfaces in KOTOR",
            "terrain_sculpt_and_generated_walkmesh_work_in_game": "Sculpted terrain and its generated WOK work in KOTOR",
            "placed_assets_match_editor_staging": "Placed assets match their Map Studio position and orientation",
            "enemy_spawns_hostile": "Placed enemy spawns and attacks the player",
            "npc_spawns_and_free_roams": "Placed friendly NPC spawns and free-roams",
            "terminal_operates": "Placed terminal can be used and performs its configured action",
            "container_opens_with_inventory": "Placed container opens and contains its configured inventory",
            "puzzle_sequence_unlocks_door": "The staged 1-2-3 puzzle unlocks its reward door",
            "animated_door_operates": "Placed animated door opens and closes correctly",
            "configured_transition_operates": "Configured door/trigger transition reaches its destination",
            "player_start_position_and_facing_match": "Player start position and facing match Map Studio",
        }.get(check, check.replace("_", " "))

    @staticmethod
    def _proof_check_status(check: str, missing: set[str]) -> str:
        if check in missing:
            return "Required after staging; cannot be satisfied by package build alone"
        return "Accepted in recorded proof"

    @classmethod
    def _reference_rows_from_metadata(cls, metadata: dict[str, object]) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for ref in tuple(metadata.get("gameplay_template_references") or ()):
            if not isinstance(ref, dict):
                continue
            restype = str(ref.get("restype") or "").lower().lstrip(".")
            resref = str(ref.get("template_resref") or "").strip()
            kind = str(ref.get("kind") or "template").strip()
            if not resref or not restype:
                continue
            label = f"{kind}:{resref}.{restype}"
            status = str(ref.get("status") or ("packaged" if ref.get("packaged") else "external_or_base_game"))
            reason = str(ref.get("message") or cls._reference_reason("template", restype))
            rows.append((label, status, reason))
        for ref in tuple(metadata.get("script_references") or ()):
            if not isinstance(ref, dict):
                continue
            script = str(ref.get("script_resref") or "").strip()
            if not script:
                continue
            label = f"script:{script}.ncs"
            status = str(ref.get("status") or ("packaged" if ref.get("packaged") else "external_or_override"))
            reason = str(ref.get("message") or cls._reference_reason("script", "ncs"))
            rows.append((label, status, reason))
        for ref in tuple(metadata.get("dialog_references") or ()):
            if not isinstance(ref, dict):
                continue
            dialog = str(ref.get("dialog_resref") or "").strip()
            if not dialog:
                continue
            label = f"dialog:{dialog}.dlg"
            status = str(ref.get("status") or ("packaged" if ref.get("packaged") else "external_or_override"))
            reason = str(ref.get("message") or cls._reference_reason("dialog", "dlg"))
            rows.append((label, status, reason))
        return rows

    @staticmethod
    def _resource_label(key: object) -> str:
        if isinstance(key, tuple) and len(key) >= 2:
            return f"{str(key[0]).strip()}.{str(key[1]).strip().lower().lstrip('.')}"
        return str(key or "").strip()

    @staticmethod
    def _resource_reason(label: str) -> str:
        restype = label.rpartition(".")[2].lower()
        return {
            "are": "Area metadata used when the module loads.",
            "git": "Gameplay instances such as entry point, placeables, doors, and triggers.",
            "ifo": "Module identity and entry area metadata.",
            "pth": "Path graph anchors for entry, placements, and transitions.",
            "lyt": "Room layout membership and positions.",
            "vis": "Room visibility/culling links.",
            "wok": "Walkable/non-walkable collision surface.",
            "mdl": "Visible room model geometry.",
            "mdx": "Paired model vertex data.",
        }.get(restype, "KOTOR runtime package dependency.")

    @staticmethod
    def _reference_reason(kind: str, restype: str) -> str:
        if kind == "script":
            return "ARE/IFO script hook dependency that must resolve during the in-game smoke test."
        if kind == "dialog":
            return "Dialog/conversation dependency that must resolve during the in-game smoke test."
        return {
            "utc": "Creature template referenced by authored GIT placement data.",
            "utd": "Door template referenced by authored GIT transition data.",
            "utp": "Placeable template referenced by authored GIT placement data.",
            "utt": "Trigger template referenced by authored GIT transition data.",
            "utw": "Waypoint template referenced by authored GIT/pathing proof data.",
        }.get(restype, "Gameplay template dependency that must resolve during the in-game smoke test.")

    def _sync_install_controls(self) -> None:
        install = self.install_check.isChecked()
        self.modules_dir_edit.setEnabled(install)
        self.overwrite_check.setEnabled(install)
        destination = self._install_destination()
        exists = bool(destination and destination.exists())
        if not install:
            self.overwrite_check.setText("Back up and replace existing module package if needed")
            self.overwrite_check.setChecked(False)
            return
        if exists:
            self.overwrite_check.setText(f"Back up and replace existing {destination.name}")
        else:
            self.overwrite_check.setText("No existing .mod detected in selected Modules folder")
            self.overwrite_check.setChecked(False)

    def _install_destination(self) -> Path | None:
        modules_dir = self.modules_dir_edit.text().strip()
        module_root = self.module_root_edit.text().strip().lower()
        if not modules_dir or not module_root:
            return None
        return Path(modules_dir) / f"{module_root}.mod"

    def _accept_if_valid(self) -> None:
        if not self.output_dir_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Map Studio Package Wizard", "Choose a staging/package folder first.")
            return
        if self.install_check.isChecked():
            if not self.modules_dir_edit.text().strip():
                QtWidgets.QMessageBox.warning(self, "Map Studio Package Wizard", "Choose the target KOTOR Modules folder first.")
                return
            destination = self._install_destination()
            if destination is not None and destination.exists() and not self.dry_run_check.isChecked() and not self.overwrite_check.isChecked():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Map Studio Package Wizard",
                    f"{destination.name} already exists. Enable backup/replace or choose another Modules folder.",
                )
                return
        self.accept()

    def _browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Map Studio package staging folder", self.output_dir_edit.text().strip())
        if path:
            self.output_dir_edit.setText(path)

    def _browse_modules_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose KOTOR Modules folder", self.modules_dir_edit.text().strip())
        if path:
            self.modules_dir_edit.setText(path)


class _MapStudioNewProjectDialog(QtWidgets.QDialog):
    """Collect the KOTOR-facing identity for a new Map Studio KMAP."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        module_root: str = "grdev01",
        game: str = "K1",
        author: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Map Studio KMAP")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.hint_label = QtWidgets.QLabel(
            "Create a KMAP project with the KOTOR module root it will eventually export as. "
            "Use a resref-safe name: 16 characters or fewer, letters, numbers, and underscores."
        )
        self.hint_label.setObjectName("mapStudioNewProjectHintLabel")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.module_root_edit = QtWidgets.QLineEdit(module_root or "grdev01")
        self.module_root_edit.setObjectName("mapStudioNewProjectModuleRootLineEdit")
        self.module_root_edit.setPlaceholderText("grdev01")
        form.addRow("Module root / KMAP name", self.module_root_edit)

        self.game_combo = QtWidgets.QComboBox()
        self.game_combo.setObjectName("mapStudioNewProjectGameComboBox")
        self.game_combo.addItem("Knights of the Old Republic (K1)", "K1")
        self.game_combo.addItem("The Sith Lords (K2)", "K2")
        index = self.game_combo.findData(str(game or "K1").upper())
        self.game_combo.setCurrentIndex(index if index >= 0 else 0)
        form.addRow("Target game", self.game_combo)

        self.author_edit = QtWidgets.QLineEdit(author or "")
        self.author_edit.setObjectName("mapStudioNewProjectAuthorLineEdit")
        self.author_edit.setPlaceholderText("modder name or team")
        form.addRow("Author", self.author_edit)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        return {
            "name": self.module_root_edit.text().strip(),
            "game": str(self.game_combo.currentData() or "K1"),
            "author": self.author_edit.text().strip(),
        }


class _MapStudioToolBeltCustomizeDialog(QtWidgets.QDialog):
    """Choose which Map Studio actions appear in the session tool belt."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        actions: tuple[Any, ...] = (),
        selected_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize Map Studio Tool Belt")
        self.setModal(True)
        root = QtWidgets.QVBoxLayout(self)
        self.hint_label = QtWidgets.QLabel(
            "Choose the modeling, terrain, gameplay, and validation actions you want in the active tool belt. "
            "This custom belt is kept for the current Map Studio session."
        )
        self.hint_label.setObjectName("mapStudioToolBeltCustomizeHintLabel")
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setObjectName("mapStudioToolBeltCustomizeSearchLineEdit")
        self.search_edit.setPlaceholderText("Filter tools by name, workspace, or KOTOR guardrail")
        root.addWidget(self.search_edit)
        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setObjectName("mapStudioToolBeltCustomizeSummaryLabel")
        root.addWidget(self.summary_label)
        self.action_list = QtWidgets.QListWidget()
        self.action_list.setObjectName("mapStudioToolBeltCustomizeListWidget")
        root.addWidget(self.action_list, 1)

        selected = {str(key) for key in selected_keys}
        for action in actions:
            key = str(getattr(action, "key", "") or "")
            if not key:
                continue
            label = str(getattr(action, "label", key) or key)
            workspace = str(getattr(action, "workspace_key", "") or "builder").replace("_", " ")
            state = "usable" if bool(getattr(action, "implemented", False)) else "planned"
            item = QtWidgets.QListWidgetItem(f"{label}  [{workspace}; {state}]")
            item.setData(QtCore.Qt.UserRole, key)
            description = str(getattr(action, "description", "") or "")
            tooltip = str(getattr(action, "description", "") or "")
            guardrail = str(getattr(action, "kotor_guardrail", "") or "")
            if guardrail:
                tooltip = f"{tooltip}\nKOTOR: {guardrail}" if tooltip else f"KOTOR: {guardrail}"
            item.setData(
                QtCore.Qt.UserRole + 1,
                " ".join((key, label, workspace, state, description, guardrail)).lower(),
            )
            item.setToolTip(tooltip)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if key in selected else QtCore.Qt.Unchecked)
            self.action_list.addItem(item)
        self.action_list.itemChanged.connect(lambda _item: self._update_selection_summary())
        self.search_edit.textChanged.connect(self._filter_actions)
        self._update_selection_summary()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _filter_actions(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for row in range(self.action_list.count()):
            item = self.action_list.item(row)
            if item is None:
                continue
            haystack = str(item.data(QtCore.Qt.UserRole + 1) or "").lower()
            item.setHidden(bool(needle and needle not in haystack))
        self._update_selection_summary()

    def _update_selection_summary(self) -> None:
        selected = 0
        visible = 0
        total = self.action_list.count()
        for row in range(total):
            item = self.action_list.item(row)
            if item is None:
                continue
            if item.checkState() == QtCore.Qt.Checked:
                selected += 1
            if not item.isHidden():
                visible += 1
        self.summary_label.setText(f"{selected} selected; {visible} visible of {total} available Map Studio tools.")

    def selected_action_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for row in range(self.action_list.count()):
            item = self.action_list.item(row)
            if item is not None and item.checkState() == QtCore.Qt.Checked:
                key = str(item.data(QtCore.Qt.UserRole) or "")
                if key:
                    keys.append(key)
        return tuple(keys)


class ModuleEditorWindow(QtWidgets.QMainWindow):
    """Top-level KMAP Map Studio Level Editor window."""

    scriptingResourceEditRequested = QtCore.Signal(object)
    terrainCatalogProgress = QtCore.Signal(str)
    terrainCatalogReady = QtCore.Signal(object, str)
    terrainCatalogFailed = QtCore.Signal(str)
    environmentCatalogProgress = QtCore.Signal(str)
    environmentCatalogReady = QtCore.Signal(object, str)
    environmentCatalogFailed = QtCore.Signal(str)
    environmentThumbnailReady = QtCore.Signal(object, object, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        theme_manager: Any = None,
        layout_manager: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowTitle("Ghost-Studio Map Studio - Level Editor")
        self.controller = ModuleEditorController()
        self.theme_manager = theme_manager or getattr(parent, "theme_manager", None)
        self.layout_manager = layout_manager or getattr(parent, "layout_manager", None)
        self._last_output_dir = ""
        self._last_game_modules_dir = ""
        self._last_map_studio_install_overwrite = False
        self._library_rows: list[dict[str, Any]] = []
        self._base_library_rows: list[dict[str, Any]] = []
        self._placeable_library_rows: list[dict[str, Any]] = []
        self._custom_creature_resources: dict[str, tuple[bytes, str]] = {}
        self._placeable_library_root = ""
        self._placeable_game_resource_provider: Any = None
        self._placeable_library_game = ""
        self._placeable_library_module_root = ""
        self._map_studio_placement_thumbnail_cache: dict[tuple[Any, ...], QtGui.QPixmap | None] = {}
        self._map_studio_placement_thumbnail_waits: dict[tuple[Any, ...], int] = {}
        self._map_studio_terrain_kit_thumbnail_cache: dict[tuple[Any, ...], QtGui.QPixmap | None] = {}
        self._map_studio_terrain_kit_thumbnail_waits: dict[tuple[Any, ...], int] = {}
        self._map_studio_environment_kit_thumbnail_cache: dict[tuple[Any, ...], QtGui.QPixmap | None] = {}
        self._map_studio_environment_kit_thumbnail_waits: dict[tuple[Any, ...], int] = {}
        self._map_studio_environment_kit_thumbnail_inflight: set[tuple[Any, ...]] = set()
        self._scripting_studio_resources: tuple[tuple[str, str, bytes], ...] = ()
        self._plcaa_manual_proof_rows: list[dict[str, Any]] = []
        self._plcaa_manual_proof_build: Any = None
        self._plcaa_manual_proof_cache_key: tuple[str, int] | None = None
        self._map_studio_workspace_modes: dict[str, Any] = {}
        self._map_studio_custom_belt_keys: tuple[str, ...] = ()
        self._map_studio_tool_action_index: dict[str, Any] = {}
        self._syncing_map_studio_tool_belt_preferences = False
        self._map_studio_viewport_focus_state: dict[str, Any] | None = None
        # Loaded modules should open as a complete world. Backdrop surfaces
        # remain non-pickable, and the toolbar checkbox can hide them when a
        # modder needs an uncluttered blockout view.
        self._map_studio_show_skybox = True
        self._texture_paint_session: TexturePaintSession | None = None
        self._texture_paint_texture_id = ""
        self._texture_paint_resref = ""
        self._texture_paint_view_image: Any = None
        self._texture_paint_accepting_stroke = False
        self._texture_paint_stamp_name = ""
        self._texture_paint_stamp_size: tuple[int, int] = (0, 0)
        self._texture_paint_stamp_rgba = b""
        self._texture_paint_preview_error = ""
        self._texture_paint_upload_timer = QtCore.QTimer(self)
        self._texture_paint_upload_timer.setSingleShot(True)
        self._texture_paint_upload_timer.setInterval(16)
        self._texture_paint_upload_timer.timeout.connect(self._flush_map_studio_texture_paint_tiles)
        self._map_studio_pie_session: Any = None
        self._map_studio_pie_last_time = 0.0
        self._map_studio_pie_camera_snapshot: tuple[Any, ...] | None = None
        self._map_studio_pie_desired_camera_distance = 3.2
        self._map_studio_pie_last_resolved_camera_distance: float | None = None
        self._map_studio_pie_status_bucket = -1
        self._map_studio_pie_active_room_resref = ""
        self._map_studio_pie_scope_text = ""
        self._map_studio_pie_control_states: list[tuple[Any, bool]] = []
        self._map_studio_pie_actor: Any = None
        self._map_studio_pie_hidden_player_start_groups: list[tuple[int, Any]] = []
        self._map_studio_pie_party_actors: list[dict[str, Any]] = []
        self._map_studio_pie_animation_engine: Any = None
        self._map_studio_pie_animation_name = ""
        self._map_studio_pie_animation_run = False
        self._map_studio_pie_actor_warning = ""
        self._map_studio_pie_player_model_cache: dict[tuple[Any, ...], tuple[Any, str]] = {}
        self._map_studio_pie_player_prewarm_pending: dict[tuple[Any, ...], Event] = {}
        self._map_studio_pie_player_cache_lock = Lock()
        self._map_studio_pie_camera_turn_input = 0.0
        self._map_studio_pie_camera_turn_velocity = 0.0
        self._map_studio_pie_creature_entries: list[dict[str, Any]] = []
        self._map_studio_pie_hidden_creature_groups: list[tuple[int, Any]] = []
        self._map_studio_pie_door_entries: list[dict[str, Any]] = []
        self._map_studio_pie_hidden_door_groups: list[tuple[int, Any]] = []
        self._map_studio_pie_authoring_preview_model: Any | None = None
        self._map_studio_pie_runtime_preview_model: Any | None = None
        self._map_studio_pie_creature_summary = "creatures 0"
        self._map_studio_pie_creature_animation_budget = 0.0
        self._map_studio_pie_creature_animation_cursor = 0
        self._map_studio_pie_creature_prepare_generation = 0
        self._map_studio_pie_creature_prepare_future: Any = None
        self._map_studio_pie_creature_prepare_cancel: Event | None = None
        self._map_studio_pie_creature_prepare_preview_id = 0
        self._map_studio_pie_creature_model_cache: dict[tuple[Any, ...], Any] = {}
        self._map_studio_pie_creature_model_cache_context: tuple[Any, ...] | None = None
        self._map_studio_pie_audio_runtime: Any = None
        self._map_studio_pie_audio_summary = "audio unavailable"
        self._map_studio_pie_audio_update_bucket = -1
        self._map_studio_pie_dialogue_audio_runtime: Any = None
        self._map_studio_pie_dialogue_audio_signature: tuple[str, ...] = ()
        self._map_studio_pie_dialogue_line_signature: tuple[str, ...] = ()
        self._map_studio_pie_dialogue_line_elapsed = 0.0
        self._map_studio_pie_dialogue_line_interval = 0.0
        self._map_studio_pie_player_action_state: dict[str, Any] = {}
        self._map_studio_pie_dialogue_animation_entities: set[str] = set()
        self._map_studio_pie_dialogue_animation_policies: dict[int, Any] = {}
        self._map_studio_pie_dialogue_animation_policies_loaded = False
        self._map_studio_pie_dialogue_node_id = ""
        self._map_studio_pie_dialogue_lip_limitation_reported = False
        self._map_studio_pie_dialogue_camera_animation_signature: tuple[str, int] | None = None
        self._map_studio_pie_dialogue_camera_animation_active = False
        self._map_studio_pie_dialogue_camera_snapshot: tuple[float, float, float, float] | None = None
        self._map_studio_pie_gameplay_mode = "exploration"
        self.resource_manager: Any = None
        self._build_actions()
        self._build_menus()
        self._build_ui()
        self._connect()
        self.set_renderer_settings(RendererSettings.from_settings(getattr(parent, "settings_data", {}) or {}))
        self.set_navigation_profile(
            getattr(parent, "settings_data", {}).get("viewport_navigation_profile", DEFAULT_VIEWPORT_NAVIGATION_PROFILE)
            if parent is not None
            else DEFAULT_VIEWPORT_NAVIGATION_PROFILE
        )
        self._refresh_all()
        if self.layout_manager is not None:
            self.apply_ghost_layout(self.layout_manager.current_layout or self.layout_manager.get_layout())
        if self.theme_manager is not None:
            self.apply_ghost_theme(self.theme_manager.current_theme or self.theme_manager.get_theme())

    @property
    def project(self) -> KMapProject:
        return self.controller.project

    def _build_actions(self) -> None:
        self.new_action = QtGui.QAction("New KMAP", self)
        self.open_action = QtGui.QAction("Open KMAP...", self)
        self.save_action = QtGui.QAction("Save KMAP", self)
        self.save_as_action = QtGui.QAction("Save KMAP As...", self)
        self.import_module_action = QtGui.QAction("Import Module...", self)
        self.import_mod_file_action = QtGui.QAction("Import Module File (.mod / .rim)...", self)
        self.import_mod_file_action.setObjectName("mapStudioImportModFileAction")
        self.import_stock_module_action = QtGui.QAction("Import Stock Module (RIM)...", self)
        self.import_obj_map_action = QtGui.QAction("Import OBJ Map...", self)
        self.import_obj_map_action.setObjectName("mapStudioImportObjMapAction")
        self.import_obj_map_action.setToolTip(
            "Import a textured OBJ as editable KOTOR room geometry, explicitly review floor components, and generate its WOK."
        )
        self.convert_all_stock_rooms_action = QtGui.QAction("Make All Stock Rooms Editable", self)
        self.convert_all_stock_rooms_action.setObjectName("mapStudioConvertAllStockRoomsAction")
        self.add_room_from_module_action = QtGui.QAction("Add Room from Module...", self)
        self.add_room_from_module_action.setObjectName("mapStudioAddRoomFromModuleAction")
        self.add_room_from_module_action.setToolTip(
            "Browse rooms indexed from any .mod/.rim/.kmap and add one to this module, ready to snap into place."
        )
        self.snap_rooms_doorway_action = QtGui.QAction("Snap Rooms at Doorway...", self)
        self.snap_rooms_doorway_action.setObjectName("mapStudioSnapRoomsDoorwayAction")
        self.snap_rooms_doorway_action.setToolTip(
            "Move one room so a chosen doorway lines up exactly with a doorway on another room."
        )
        self.import_library_asset_action = QtGui.QAction("Import Selected Library Asset", self)
        self.import_texture_action = QtGui.QAction("Import Texture to Project...", self)
        self.import_texture_action.setObjectName("mapStudioImportTextureAction")
        self.export_fbx_action = QtGui.QAction("Export Whole Level as FBX...", self)
        self.export_fbx_action.setToolTip("Export all enabled visible rooms and placed models to one static FBX, with textures and a scene manifest.")
        self.export_obj_action = QtGui.QAction("Export Whole Level as OBJ...", self)
        self.export_obj_action.setToolTip("Export the level to one OBJ with separate mesh groups, an MTL, textures and a scene manifest.")
        self.export_package_action = QtGui.QAction("Export Scene Package...", self)
        self.close_action = QtGui.QAction("Close", self)
        self.undo_action = QtGui.QAction("Undo", self)
        self.undo_action.setObjectName("mapStudioUndoAction")
        self.undo_action.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.undo_action.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.redo_action = QtGui.QAction("Redo", self)
        self.redo_action.setObjectName("mapStudioRedoAction")
        self.redo_action.setShortcuts([QtGui.QKeySequence("Ctrl+R"), QtGui.QKeySequence("Ctrl+Y")])
        self.redo_action.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.delete_action = QtGui.QAction("Delete Selected", self)
        self.duplicate_action = QtGui.QAction("Duplicate Selected", self)
        self.rename_action = QtGui.QAction("Rename Selected", self)
        self.validate_action = QtGui.QAction("Validate KMAP", self)
        self.simulate_action = QtGui.QAction("Play in Editor", self)
        self.simulate_action.setObjectName("mapStudioSimulateAction")
        self.simulate_action.setCheckable(True)
        self.simulate_action.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.simulate_action.setToolTip(
            "Play in Editor (Alt+P) using the current walkmesh, player camera, creatures, and ambient sound. "
            "This is a GhostStudio simulation, not KOTOR engine proof."
        )
        self.simulate_action.setStatusTip(self.simulate_action.toolTip())
        self.simulate_action.setProperty("mapStudioPIEState", "editing")
        self.build_action = QtGui.QAction("Build Module Files", self)
        self.generate_walls_action = QtGui.QAction("Walkmesh Boundary Rules...", self)
        self.generate_walls_action.setObjectName("mapStudioWalkmeshBoundaryRulesAction")
        self.generate_walls_action.setToolTip(
            "KOTOR room WOKs are walkable floor regions. Visible walls belong in room geometry, not as vertical WOK faces."
        )
        self.paint_walkmesh_action = QtGui.QAction("Paint Walkmesh Faces", self)
        self.open_output_action = QtGui.QAction("Open Output Folder", self)
        self.help_action = QtGui.QAction("Map Studio Help", self)
        self.kmap_help_action = QtGui.QAction("KMAP Format Help", self)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        for action in (self.new_action, self.open_action, self.save_action, self.save_as_action):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_mod_file_action)
        file_menu.addAction(self.import_module_action)
        file_menu.addAction(self.import_stock_module_action)
        file_menu.addAction(self.import_obj_map_action)
        file_menu.addAction(self.convert_all_stock_rooms_action)
        file_menu.addAction(self.add_room_from_module_action)
        file_menu.addAction(self.snap_rooms_doorway_action)
        file_menu.addAction(self.import_library_asset_action)
        file_menu.addAction(self.import_texture_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_fbx_action)
        file_menu.addAction(self.export_obj_action)
        file_menu.addAction(self.export_package_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_action)

        edit_menu = self.menuBar().addMenu("Edit")
        for action in (self.undo_action, self.redo_action, self.delete_action, self.duplicate_action, self.rename_action):
            edit_menu.addAction(action)
        edit_menu.addSeparator()
        self.center_pivot_action = edit_menu.addAction("Center Pivot")
        self.center_pivot_action.setObjectName("mapStudioCenterPivotAction")
        self.center_pivot_action.triggered.connect(lambda: self._run_map_studio_tool_belt_key("center_pivot"))
        self.freeze_transform_action = edit_menu.addAction("Freeze Transforms")
        self.freeze_transform_action.setObjectName("mapStudioFreezeTransformAction")
        self.freeze_transform_action.triggered.connect(lambda: self._run_map_studio_tool_belt_key("freeze_transform"))

        view_menu = self.menuBar().addMenu("View")
        self.view_menu = view_menu
        self.outliner_action = view_menu.addAction("Show Outliner")
        self.outliner_action.setCheckable(True)
        self.outliner_action.setChecked(True)
        self.properties_action = view_menu.addAction("Show Properties")
        self.properties_action.setCheckable(True)
        self.properties_action.setChecked(True)
        self.viewport_action = view_menu.addAction("Show Viewport")
        self.viewport_action.setCheckable(True)
        self.viewport_action.setChecked(True)
        self.focus_viewport_action = view_menu.addAction("Maximize Viewport")
        self.focus_viewport_action.setObjectName("mapStudioMaximizeViewportAction")
        self.focus_viewport_action.setCheckable(True)
        self.focus_viewport_action.setShortcut(QtGui.QKeySequence("Ctrl+Space"))
        self.focus_viewport_action.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.focus_viewport_action.setToolTip(
            "Temporarily collapse both side rails and lower docks so the editor viewport fills Map Studio."
        )
        self.validation_action = view_menu.addAction("Show Validation / Output")
        self.validation_action.setCheckable(True)
        self.validation_action.setChecked(False)
        self.reset_layout_action = view_menu.addAction("Reset Layout")
        self.reset_layout_action.triggered.connect(self._reset_layout)

        tools_menu = self.menuBar().addMenu("Tools")
        for action in (
            self.validate_action,
            self.simulate_action,
            self.build_action,
            self.generate_walls_action,
            self.paint_walkmesh_action,
            self.open_output_action,
        ):
            tools_menu.addAction(action)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(self.help_action)
        help_menu.addAction(self.kmap_help_action)

    def _build_ui(self) -> None:
        shell = QtWidgets.QWidget()
        self.setCentralWidget(shell)
        root = QtWidgets.QVBoxLayout(shell)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        self.toolbar = ModuleEditorToolbar(self)
        self.toolbar_scroll = make_horizontal_overflow_area(
            self.toolbar,
            "mapStudioTopToolbarScrollArea",
            height=34,
            parent=shell,
        )
        root.addWidget(self.toolbar_scroll)
        self.map_studio_scope_label = QtWidgets.QLabel(
            "Map Studio Level Editor: KMAP terrain, rooms, walkmesh, placements, validation, staged export, install handoff, and game proof."
        )
        self.map_studio_scope_label.setObjectName("mapStudioLevelEditorScopeLabel")
        self.map_studio_scope_label.setWordWrap(True)
        root.addWidget(self.map_studio_scope_label)
        workspace_row = QtWidgets.QHBoxLayout()
        workspace_row.setContentsMargins(0, 0, 0, 0)
        workspace_row.setSpacing(6)
        self.map_studio_workspace_label = QtWidgets.QLabel("Workspace")
        self.map_studio_workspace_label.setObjectName("mapStudioWorkspaceLabel")
        self.map_studio_workspace_combo = QtWidgets.QComboBox()
        self.map_studio_workspace_combo.setObjectName("mapStudioWorkspaceComboBox")
        for mode in self.controller.map_studio_workspace_modes():
            key = str(getattr(mode, "key", "") or "")
            if not key:
                continue
            self._map_studio_workspace_modes[key] = mode
            self.map_studio_workspace_combo.addItem(str(getattr(mode, "label", key) or key), key)
        self.map_studio_workspace_guide_label = QtWidgets.QLabel("")
        self.map_studio_workspace_guide_label.setObjectName("mapStudioWorkspaceGuideLabel")
        self.map_studio_workspace_guide_label.setWordWrap(True)
        self.map_studio_open_workspace_button = QtWidgets.QPushButton("Open Workspace")
        self.map_studio_open_workspace_button.setObjectName("mapStudioOpenWorkspaceButton")
        workspace_row.addWidget(self.map_studio_workspace_label)
        workspace_row.addWidget(self.map_studio_workspace_combo)
        workspace_row.addWidget(self.map_studio_workspace_guide_label, 1)
        workspace_row.addWidget(self.map_studio_open_workspace_button)
        root.addLayout(workspace_row)
        self.map_studio_tool_belt_tabs = QtWidgets.QTabWidget()
        self.map_studio_tool_belt_tabs.setObjectName("mapStudioToolBeltTabs")
        self.map_studio_tool_belt_default_tab = QtWidgets.QWidget()
        self.map_studio_tool_belt_default_tab.setObjectName("mapStudioToolBeltDefaultTab")
        belt_row = QtWidgets.QHBoxLayout(self.map_studio_tool_belt_default_tab)
        belt_row.setContentsMargins(0, 0, 0, 0)
        belt_row.setSpacing(6)
        self.map_studio_tool_belt_label = QtWidgets.QLabel("Belt")
        self.map_studio_tool_belt_label.setObjectName("mapStudioToolBeltLabel")
        self.map_studio_tool_belt_preset_combo = QtWidgets.QComboBox()
        self.map_studio_tool_belt_preset_combo.setObjectName("mapStudioToolBeltPresetComboBox")
        for preset in self.controller.available_map_studio_tool_belt_presets():
            self.map_studio_tool_belt_preset_combo.addItem(str(getattr(preset, "label", "") or preset.key), str(preset.key))
        self.map_studio_tool_belt_widget = QtWidgets.QWidget()
        self.map_studio_tool_belt_widget.setObjectName("mapStudioToolBeltWidget")
        self.map_studio_tool_belt_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.map_studio_tool_belt_layout = QtWidgets.QHBoxLayout(self.map_studio_tool_belt_widget)
        self.map_studio_tool_belt_layout.setContentsMargins(0, 0, 0, 0)
        self.map_studio_tool_belt_layout.setSpacing(4)
        self.map_studio_tool_belt_scroll = make_horizontal_overflow_area(
            self.map_studio_tool_belt_widget,
            "mapStudioToolBeltScrollArea",
            height=34,
            parent=self.map_studio_tool_belt_default_tab,
        )
        self.map_studio_command_search_combo = QtWidgets.QComboBox()
        self.map_studio_command_search_combo.setObjectName("mapStudioCommandSearchComboBox")
        self.map_studio_command_search_combo.setEditable(True)
        self.map_studio_command_search_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.map_studio_command_search_combo.setMinimumWidth(220)
        self.map_studio_command_search_combo.setToolTip("Search and run any command-backed Map Studio tool. Shortcut: Ctrl+K.")
        self.map_studio_command_run_button = QtWidgets.QPushButton("Run")
        self.map_studio_command_run_button.setObjectName("mapStudioCommandSearchRunButton")
        self.map_studio_command_run_button.setToolTip("Run the selected Map Studio command through the shared tool action dispatcher.")
        self.map_studio_customize_tool_belt_button = QtWidgets.QPushButton("Customize Belt...")
        self.map_studio_customize_tool_belt_button.setObjectName("mapStudioCustomizeToolBeltButton")
        belt_row.addWidget(self.map_studio_tool_belt_label)
        belt_row.addWidget(self.map_studio_tool_belt_preset_combo)
        belt_row.addWidget(self.map_studio_tool_belt_scroll, 1)
        belt_row.addWidget(self.map_studio_command_search_combo)
        belt_row.addWidget(self.map_studio_command_run_button)
        belt_row.addWidget(self.map_studio_customize_tool_belt_button)
        self.map_studio_tool_belt_tabs.addTab(self.map_studio_tool_belt_default_tab, "Default")

        self.map_studio_tool_belt_custom_tab = QtWidgets.QWidget()
        self.map_studio_tool_belt_custom_tab.setObjectName("mapStudioToolBeltCustomTab")
        custom_belt_root = QtWidgets.QVBoxLayout(self.map_studio_tool_belt_custom_tab)
        custom_belt_root.setContentsMargins(0, 0, 0, 0)
        custom_belt_root.setSpacing(4)
        custom_add_row = QtWidgets.QHBoxLayout()
        custom_add_row.setContentsMargins(0, 0, 0, 0)
        custom_add_row.setSpacing(6)
        self.map_studio_custom_tool_combo = QtWidgets.QComboBox()
        self.map_studio_custom_tool_combo.setObjectName("mapStudioCustomToolComboBox")
        self.map_studio_custom_tool_combo.setEditable(True)
        self.map_studio_custom_tool_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.map_studio_custom_tool_combo.setToolTip("Search all Map Studio modeling, terrain, placement, and export tools.")
        self.map_studio_custom_tool_add_button = QtWidgets.QToolButton()
        self.map_studio_custom_tool_add_button.setObjectName("mapStudioCustomToolAddButton")
        self.map_studio_custom_tool_add_button.setText("+")
        self.map_studio_custom_tool_add_button.setToolTip("Add the selected indexed tool to the custom Map Studio tool belt.")
        custom_add_row.addWidget(QtWidgets.QLabel("Custom Tool"))
        custom_add_row.addWidget(self.map_studio_custom_tool_combo, 1)
        custom_add_row.addWidget(self.map_studio_custom_tool_add_button)
        self.map_studio_custom_tool_belt_widget = QtWidgets.QWidget()
        self.map_studio_custom_tool_belt_widget.setObjectName("mapStudioCustomToolBeltWidget")
        self.map_studio_custom_tool_belt_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.map_studio_custom_tool_belt_layout = QtWidgets.QHBoxLayout(self.map_studio_custom_tool_belt_widget)
        self.map_studio_custom_tool_belt_layout.setContentsMargins(0, 0, 0, 0)
        self.map_studio_custom_tool_belt_layout.setSpacing(4)
        self.map_studio_custom_tool_belt_scroll = make_horizontal_overflow_area(
            self.map_studio_custom_tool_belt_widget,
            "mapStudioCustomToolBeltScrollArea",
            height=34,
            parent=self.map_studio_tool_belt_custom_tab,
        )
        custom_belt_root.addLayout(custom_add_row)
        custom_belt_root.addWidget(self.map_studio_custom_tool_belt_scroll)
        self.map_studio_tool_belt_tabs.addTab(self.map_studio_tool_belt_custom_tab, "Custom +")
        root.addWidget(self.map_studio_tool_belt_tabs)
        self.map_studio_command_search_readiness_label = QtWidgets.QLabel(
            "Command readiness: choose a Map Studio tool to see capability stage, affected KOTOR resources, and export/game-proof impact."
        )
        self.map_studio_command_search_readiness_label.setObjectName("mapStudioCommandSearchReadinessLabel")
        self.map_studio_command_search_readiness_label.setWordWrap(True)
        root.addWidget(self.map_studio_command_search_readiness_label)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(True)
        root.addWidget(self.main_splitter, 1)

        left = QtWidgets.QWidget()
        left.setObjectName("mapStudioLeftAuthoringRail")
        left.setMinimumWidth(220)
        left.setMaximumWidth(16777215)
        left.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.left_authoring_rail = left
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.outliner = ModuleEditorOutliner(left)
        self.asset_browser = ModuleEditorAssetBrowser(left)
        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self.outliner, "Outliner")
        self.left_tabs.addTab(self.asset_browser, "Assets")
        self.workflow_tabs = _MapStudioWorkflowStack()
        self.workflow_tabs.setObjectName("mapStudioWorkflowStack")
        self.workflow_selector = QtWidgets.QComboBox()
        self.workflow_selector.setObjectName("mapStudioWorkflowSelector")
        self.workflow_selector.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.workflow_selector.setToolTip(
            "Choose the Map Studio workflow shown below. All workflows stay reachable in the narrow authoring rail."
        )
        self.workflow_selector.setAccessibleName("Map Studio workflow")
        self.rooms_tab = RoomsTab()
        self.placement_tab = PlacementTab()
        self.walkmesh_tab = WalkmeshTab()
        self.texture_paint_tab = MapStudioTexturePaintTab()
        self.environment_tab = MapStudioEnvironmentTab()
        self.porter_tab = PorterTab()
        self.builder_tab = BuilderTab()
        self.environment_kit_browser = EnvironmentKitBrowser()
        self.environment_kit_browser.set_assets(self.controller.available_map_studio_environment_kit_pieces())
        self.builder_tab.adopt_environment_kit_browser(self.environment_kit_browser)
        self.terrain_kit_browser = TerrainKitBrowser()
        self.terrain_kit_browser.set_assets(self.controller.available_map_studio_terrain_kit_assets())
        self.builder_tab.adopt_terrain_kit_browser(self.terrain_kit_browser)
        self._map_studio_kit_browser_game = str(getattr(self.project, "game", "K1") or "K1").upper()
        self.builder_tab.adopt_skybox_tools(
            self.environment_tab.sky_group,
            self.environment_tab.sky_traffic_group,
        )
        self.environment_tab.adopt_room_lighting_tools(self.builder_tab.roomLightingGroup)
        self.builder_tab.set_primitive_presets(self.controller.available_authored_room_presets())
        self.builder_tab.set_terrain_shape_presets(self.controller.available_authored_terrain_shape_presets())
        self.builder_tab.set_walkmesh_surfaces(self.controller.available_authored_walkmesh_surfaces())
        self.walkmesh_tab.set_walkmesh_surfaces(self.controller.available_authored_walkmesh_surfaces())
        self.builder_tab.set_composition_primitive_kinds(self.controller.available_authored_composition_primitive_kinds())
        self.builder_tab.set_gameplay_placement_kinds(self.controller.available_authored_gameplay_placement_kinds())
        self.placement_tab.set_placement_kinds(self.controller.available_authored_gameplay_placement_kinds())
        self.builder_tab.set_script_hook_fields(self.controller.authored_script_hook_field_choices())
        self.builder_tab.set_modeling_component_modes(self.controller.available_map_studio_component_modes())
        self.builder_tab.set_modeling_tools(self.controller.available_map_studio_modeling_tools())
        self.builder_tab.set_modeling_snap_modes(self.controller.available_map_studio_snap_modes())
        self.builder_tab.set_terrain_brushes(self.controller.available_map_studio_terrain_brushes())
        self.builder_tab.set_building_styles(self.controller.available_map_studio_building_styles())
        self._refresh_map_studio_tool_index()
        self.blueprints_tab = BlueprintsTab()
        self.blueprints_tab.adopt_script_hook_tools(self.builder_tab.scriptHooksGroup)
        for label, widget in (
            ("Rooms", self.rooms_tab),
            ("Place", self.placement_tab),
            ("Walkmesh (WOK)", self.walkmesh_tab),
            ("Paint", self.texture_paint_tab),
            ("Environment", self.environment_tab),
            ("Porter", self.porter_tab),
            ("Build", self.builder_tab),
            ("Data", self.blueprints_tab),
        ):
            self.workflow_tabs.addTab(widget, label)
            self.workflow_selector.addItem(label)
        self.workflow_tabs.setTabToolTip(
            self.workflow_tabs.indexOf(self.environment_tab),
            "Environment: room and ARE lighting, weather/fog, shadows, and baked lightmaps.",
        )
        self.workflow_tabs.setTabToolTip(
            self.workflow_tabs.indexOf(self.builder_tab),
            "Build: Room Building, Terrain Building, and KOTOR Skybox authoring.",
        )
        self.workflow_selector.setItemData(
            self.workflow_tabs.indexOf(self.environment_tab),
            self.workflow_tabs.tabToolTip(self.workflow_tabs.indexOf(self.environment_tab)),
            QtCore.Qt.ToolTipRole,
        )
        self.workflow_selector.currentIndexChanged.connect(self.workflow_tabs.setCurrentIndex)
        self.workflow_tabs.currentChanged.connect(self._sync_map_studio_workflow_selector)
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.addWidget(self.left_tabs)
        workflow_host = QtWidgets.QWidget()
        workflow_host.setObjectName("mapStudioWorkflowRail")
        workflow_host.setMinimumWidth(0)
        workflow_host_layout = QtWidgets.QVBoxLayout(workflow_host)
        workflow_host_layout.setContentsMargins(0, 0, 0, 0)
        workflow_host_layout.setSpacing(4)
        workflow_selector_row = QtWidgets.QHBoxLayout()
        workflow_selector_row.setContentsMargins(4, 0, 4, 0)
        workflow_label = QtWidgets.QLabel("Workflow")
        workflow_label.setBuddy(self.workflow_selector)
        workflow_selector_row.addWidget(workflow_label)
        workflow_selector_row.addWidget(self.workflow_selector, 1)
        workflow_host_layout.addLayout(workflow_selector_row)
        self.workflow_tabs_scroll = make_scrollable_panel(
            self.workflow_tabs,
            "mapStudioWorkflowTabsScrollArea",
            parent=workflow_host,
        )
        self.workflow_tabs_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        workflow_host_layout.addWidget(self.workflow_tabs_scroll, 1)
        self.workflow_dock = QtWidgets.QDockWidget("Content Browser / Workflow", self)
        self.workflow_dock.setObjectName("mapStudioWorkflowContentBrowserDock")
        self.workflow_dock.setAccessibleName("Map Studio Content Browser and Workflow")
        self.workflow_dock.setToolTip(
            "Drag this title bar to float the Content Browser, move it to another monitor, or dock it on any edge."
        )
        self.workflow_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        self.workflow_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.workflow_dock.setWidget(workflow_host)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.workflow_dock)
        self.workflow_dock_action = self.workflow_dock.toggleViewAction()
        self.workflow_dock_action.setText("Show Content Browser / Workflow")
        self.view_menu.insertAction(self.reset_layout_action, self.workflow_dock_action)
        self.workflow_dock.topLevelChanged.connect(
            lambda _floating: self._queue_map_studio_workflow_control_fit()
        )
        self.main_splitter.splitterMoved.connect(
            lambda _position, _index: self._queue_map_studio_workflow_control_fit()
        )
        self.workflow_tabs.currentChanged.connect(
            lambda _index: self._queue_map_studio_workflow_control_fit()
        )
        QtCore.QTimer.singleShot(0, self._queue_map_studio_workflow_control_fit)
        self.left_splitter.setStretchFactor(0, 1)
        self.left_splitter.setSizes([860])
        left_layout.addWidget(self.left_splitter, 1)
        self.main_splitter.addWidget(left)

        center = QtWidgets.QWidget()
        center.setObjectName("mapStudioViewportHost")
        center.setMinimumWidth(0)
        center.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewport_host = center
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport_panel = ModuleEditorViewportPanel(center)
        # The viewport canvas owns the Maya-style marking menu through its
        # input filter.  A CustomContextMenu on the whole panel also captured
        # descendant shelf buttons, making their right-click Tool Options
        # unreachable in the real application.
        self.viewport_panel.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.viewport_panel_scroll = None
        center_layout.addWidget(self.viewport_panel, 1)
        self.main_splitter.addWidget(center)

        right = QtWidgets.QWidget()
        right.setObjectName("mapStudioRightInspectorRail")
        right.setMinimumWidth(230)
        right.setMaximumWidth(16777215)
        right.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.right_inspector_rail = right
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.properties = ModuleEditorPropertiesPanel(right)
        self.pie_context_panel = MapStudioPIEContextPanel(right)
        self.workflow_panel = MapStudioWorkflowPanel(right)
        self.readiness_panel = ModuleReadinessPanel(right)
        self.export_panel = ModuleExportPanel(right)
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.setObjectName("mapStudioRightTabs")
        self.right_tabs.setMinimumWidth(0)
        self.right_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.right_tabs.addTab(self.properties, "Properties")
        pie_context_index = self.right_tabs.addTab(self.pie_context_panel, "PIE")
        self.right_tabs.setTabToolTip(
            pie_context_index,
            "Set the simulated player context and resource-driven conversation start previews.",
        )
        export_page = QtWidgets.QWidget(self.right_tabs)
        export_page.setMinimumWidth(0)
        export_page.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.map_studio_export_page = export_page
        export_layout = QtWidgets.QVBoxLayout(export_page)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.addWidget(self.workflow_panel)
        self.workflow_panel.setVisible(False)
        self.export_diagnostics_toggle = QtWidgets.QToolButton(export_page)
        self.export_diagnostics_toggle.setObjectName("mapStudioExportDiagnosticsButton")
        self.export_diagnostics_toggle.setText("Technical export diagnostics")
        self.export_diagnostics_toggle.setCheckable(True)
        self.export_diagnostics_toggle.setChecked(False)
        self.export_diagnostics_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.export_diagnostics_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        export_layout.addWidget(self.export_panel)
        export_layout.addWidget(self.export_diagnostics_toggle)
        export_layout.addWidget(self.readiness_panel)
        self.export_diagnostics_toggle.toggled.connect(self.readiness_panel.setVisible)
        self.export_diagnostics_toggle.toggled.connect(
            lambda visible: self.export_diagnostics_toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
            )
        )
        self.readiness_panel.setVisible(False)
        self.right_tabs.addTab(export_page, "Export")
        self.right_tabs_scroll = make_scrollable_panel(
            self.right_tabs,
            "mapStudioRightTabsScrollArea",
            parent=right,
        )
        self.right_tabs_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_layout.addWidget(self.right_tabs_scroll, 1)
        self.main_splitter.addWidget(right)

        self.bottom_tabs = QtWidgets.QTabWidget()
        self.validation_panel = ModuleValidationPanel()
        self.output_log = QtWidgets.QPlainTextEdit()
        self.output_log.setReadOnly(True)
        self.bottom_tabs.addTab(self.validation_panel, "Validation")
        self.bottom_tabs.addTab(self.output_log, "Output")
        self.bottom_tabs.setMinimumSize(280, 120)
        self.validation_output_dock = QtWidgets.QDockWidget("Validation / Output", self)
        self.validation_output_dock.setObjectName("mapStudioValidationOutputDock")
        self.validation_output_dock.setAccessibleName("Map Studio Validation and Output")
        self.validation_output_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        self.validation_output_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.validation_output_dock.setWidget(self.bottom_tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.validation_output_dock)
        self.tabifyDockWidget(self.workflow_dock, self.validation_output_dock)
        self.validation_output_dock.hide()
        self.statusBar().showMessage("Map Studio Level Editor ready.")
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, True)
        self.main_splitter.setSizes([260, 1380, 300])
        self.resizeDocks((self.workflow_dock,), (280,), QtCore.Qt.Orientation.Vertical)
        self._update_map_studio_workspace_guide()
        self._refresh_map_studio_tool_belt()
        self._apply_map_studio_minimal_layout()

    def _connect(self) -> None:
        self.new_action.triggered.connect(self.new_kmap)
        self.open_action.triggered.connect(self.open_kmap)
        self.save_action.triggered.connect(self.save_kmap)
        self.save_as_action.triggered.connect(self.save_kmap_as)
        self.import_module_action.triggered.connect(self.import_module)
        self.import_mod_file_action.triggered.connect(self.import_module_file)
        self.import_stock_module_action.triggered.connect(self._import_stock_module)
        self.import_obj_map_action.triggered.connect(self.import_obj_map)
        self.convert_all_stock_rooms_action.triggered.connect(self._convert_all_stock_rooms)
        self.add_room_from_module_action.triggered.connect(self._add_room_from_module)
        self.snap_rooms_doorway_action.triggered.connect(self._snap_rooms_at_doorway)
        self.import_library_asset_action.triggered.connect(self.import_selected_library_asset)
        self.import_texture_action.triggered.connect(self.import_project_texture)
        self.export_fbx_action.triggered.connect(lambda: self.export_fbx(False))
        self.export_obj_action.triggered.connect(lambda: self._export_level_geometry("obj", False))
        self.export_package_action.triggered.connect(lambda: self.build_module_files())
        self.close_action.triggered.connect(self.close)
        self.undo_action.triggered.connect(self.undo_map_studio_command)
        self.redo_action.triggered.connect(self.redo_map_studio_command)
        self.delete_action.triggered.connect(self.delete_selected)
        self.duplicate_action.triggered.connect(self.duplicate_selected)
        self.rename_action.triggered.connect(self.rename_selected)
        self.validate_action.triggered.connect(self.validate_kmap)
        self.simulate_action.triggered.connect(self.toggle_map_studio_pie)
        self.build_action.triggered.connect(self.build_module_files)
        self.generate_walls_action.triggered.connect(lambda: self._handle_tab_action("Walkmesh Boundary Rules"))
        self.paint_walkmesh_action.triggered.connect(lambda: self._handle_tab_action("Paint Face"))
        self.open_output_action.triggered.connect(self.open_output_folder)
        self.help_action.triggered.connect(lambda: self._show_help("Map Studio"))
        self.kmap_help_action.triggered.connect(lambda: self._show_help("KMAP Format"))
        self.map_studio_workspace_combo.currentIndexChanged.connect(self._handle_map_studio_workspace_changed)
        self.map_studio_open_workspace_button.clicked.connect(lambda: self._open_selected_map_studio_workspace())
        self.map_studio_tool_belt_preset_combo.currentIndexChanged.connect(self._handle_map_studio_tool_belt_preset_changed)
        self.map_studio_customize_tool_belt_button.clicked.connect(self._customize_map_studio_tool_belt)
        self.map_studio_custom_tool_add_button.clicked.connect(self._add_selected_map_studio_custom_tool)
        self.map_studio_command_run_button.clicked.connect(self._run_selected_map_studio_command_search)
        self.map_studio_command_search_combo.currentIndexChanged.connect(
            lambda _index=0: self._update_map_studio_command_search_readiness()
        )
        self.map_studio_command_search_combo.editTextChanged.connect(
            lambda _text="": self._update_map_studio_command_search_readiness()
        )
        self._connect_map_studio_tool_context_refresh_signals()
        self.map_studio_tool_belt_widget.customContextMenuRequested.connect(
            lambda pos: self._open_map_studio_tool_context_menu(self.map_studio_tool_belt_widget, pos)
        )
        self.map_studio_custom_tool_belt_widget.customContextMenuRequested.connect(
            lambda pos: self._open_map_studio_tool_context_menu(self.map_studio_custom_tool_belt_widget, pos)
        )
        self.map_studio_command_search_action = QtGui.QAction("Map Studio Command Search", self)
        self.map_studio_command_search_action.setObjectName("mapStudioCommandSearchAction")
        self.map_studio_command_search_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        self.map_studio_command_search_action.triggered.connect(self._focus_map_studio_command_search)
        self.addAction(self.map_studio_command_search_action)
        self.map_studio_universal_transform_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self)
        self.map_studio_universal_transform_shortcut.setObjectName("mapStudioUniversalTransformShortcut")
        self.map_studio_universal_transform_shortcut.activated.connect(self._activate_map_studio_universal_transform_shortcut)
        self.map_studio_translate_gizmo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("W"), self.viewport_panel)
        self.map_studio_translate_gizmo_shortcut.setObjectName("mapStudioTranslateGizmoShortcut")
        self.map_studio_translate_gizmo_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_translate_gizmo_shortcut.activated.connect(
            lambda: self.viewport_panel.set_transform_gizmo_mode("translate")
        )
        self.map_studio_rotate_gizmo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("E"), self.viewport_panel)
        self.map_studio_rotate_gizmo_shortcut.setObjectName("mapStudioRotateGizmoShortcut")
        self.map_studio_rotate_gizmo_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_rotate_gizmo_shortcut.activated.connect(
            lambda: self.viewport_panel.set_transform_gizmo_mode("rotate")
        )
        self.map_studio_scale_gizmo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self.viewport_panel)
        self.map_studio_scale_gizmo_shortcut.setObjectName("mapStudioScaleGizmoShortcut")
        self.map_studio_scale_gizmo_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_scale_gizmo_shortcut.activated.connect(
            lambda: self.viewport_panel.set_transform_gizmo_mode("scale")
        )
        self.map_studio_delete_selection_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.viewport_panel)
        self.map_studio_delete_selection_shortcut.setObjectName("mapStudioDeleteSelectionShortcut")
        self.map_studio_delete_selection_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_delete_selection_shortcut.activated.connect(self.delete_map_studio_current_selection)
        self.map_studio_outliner_delete_selection_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.outliner
        )
        self.map_studio_outliner_delete_selection_shortcut.setObjectName("mapStudioOutlinerDeleteSelectionShortcut")
        self.map_studio_outliner_delete_selection_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_outliner_delete_selection_shortcut.activated.connect(self.delete_map_studio_current_selection)
        self.map_studio_ground_snap_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_End), self.viewport_panel)
        self.map_studio_ground_snap_shortcut.setObjectName("mapStudioGroundSnapShortcut")
        self.map_studio_ground_snap_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_ground_snap_shortcut.activated.connect(self.snap_map_studio_selected_placement_to_ground)
        self.map_studio_vertex_snap_shortcut = QtGui.QShortcut(QtGui.QKeySequence("V"), self.viewport_panel)
        self.map_studio_vertex_snap_shortcut.setObjectName("mapStudioVertexSnapShortcut")
        self.map_studio_vertex_snap_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_vertex_snap_shortcut.activated.connect(
            lambda: self._activate_map_studio_modifier_shortcut("vertex_snap")
        )
        self.map_studio_transform_level_snap_shortcut = QtGui.QShortcut(QtGui.QKeySequence("J"), self.viewport_panel)
        self.map_studio_transform_level_snap_shortcut.setObjectName("mapStudioTransformLevelSnapShortcut")
        self.map_studio_transform_level_snap_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.map_studio_transform_level_snap_shortcut.activated.connect(
            lambda: self._activate_map_studio_modifier_shortcut("transform_snap_level")
        )
        # Maya shelf parity.  These shortcuts are scoped to the Map Studio
        # viewport, so they never steal input from text fields or the other
        # Ghost Studio workbenches.
        self._map_studio_maya_shortcuts: list[QtGui.QShortcut] = []
        maya_shortcut_parent = getattr(self.viewport_panel, "viewport", self.viewport_panel)
        for object_name, sequence, callback in (
            ("mapStudioMayaExtrudeShortcut", "Ctrl+E", lambda: self._run_map_studio_maya_shortcut("extrude")),
            ("mapStudioMayaBevelShortcut", "Ctrl+B", lambda: self._run_map_studio_maya_shortcut("bevel")),
            ("mapStudioMayaMultiCutShortcut", "Ctrl+X", lambda: self._run_map_studio_maya_shortcut("multi_cut")),
            ("mapStudioMayaQuadDrawShortcut", "Ctrl+Q", lambda: self._run_map_studio_maya_shortcut("quad_draw")),
            ("mapStudioMayaDuplicateSpecialShortcut", "Ctrl+Shift+D", lambda: self._run_map_studio_maya_shortcut("duplicate_special")),
            ("mapStudioMayaBridgeOrFillShortcut", "Ctrl+/", self._run_map_studio_bridge_or_fill_shortcut),
            ("mapStudioMayaRepeatLastShortcut", "G", self._repeat_last_map_studio_modeling_command),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), maya_shortcut_parent)
            shortcut.setObjectName(object_name)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._map_studio_maya_shortcuts.append(shortcut)
        self.toolbar.actionRequested.connect(self._toolbar_action)
        self.toolbar.viewModeChanged.connect(self.viewport_panel.set_view_mode)
        self.toolbar.selectionModeChanged.connect(self._handle_map_studio_edit_mode_changed)
        self.toolbar.skyboxVisibilityChanged.connect(self._set_map_studio_skybox_visible)
        self.asset_browser.importRequested.connect(self.import_library_asset)
        self.outliner.itemsSelected.connect(self._select_map_studio_items)
        self.outliner.actionRequested.connect(self._outliner_action)
        self.outliner.itemRenamed.connect(self._rename_outliner_item_inline)
        self.viewport_panel.itemSelected.connect(self.select_item)
        self.viewport_panel.itemsSelected.connect(self._select_map_studio_items)
        self.viewport_panel.transformEdited.connect(self._set_transform)
        self.viewport_panel.placementsTranslated.connect(self._translate_map_studio_placements)
        self.viewport_panel.placementRequested.connect(self._place_authored_gameplay_from_viewport)
        self.viewport_panel.placementModeExited.connect(self.placement_tab.stop_placement_mode)
        self.viewport_panel.terrainKitRequested.connect(self._place_authored_terrain_kit_from_viewport)
        self.viewport_panel.terrainKitSnapPreviewRequested.connect(
            self._preview_authored_terrain_kit_from_viewport
        )
        self.viewport_panel.environmentKitRequested.connect(self._place_authored_environment_kit_from_viewport)
        self.viewport_panel.environmentKitSnapPreviewRequested.connect(
            self._preview_authored_terrain_kit_from_viewport
        )
        self.viewport_panel.buildingRoomRequested.connect(self._build_map_studio_room_from_viewport)
        self.viewport_panel.buildingOpeningPreviewRequested.connect(
            self._preview_map_studio_opening_from_viewport
        )
        self.viewport_panel.buildingOpeningRequested.connect(self._build_map_studio_opening_from_viewport)
        self.viewport_panel.roomOutlinePointEdited.connect(self._set_authored_room_outline_point)
        self.viewport_panel.roomOutlinePointSnapPreviewRequested.connect(self.preview_authored_floor_plan_vertex_snap_candidates)
        self.viewport_panel.roomOutlinePointSnapped.connect(self.snap_authored_floor_plan_vertex)
        self.viewport_panel.roomOutlineEdgeSelected.connect(self._select_authored_room_outline_edge)
        self.viewport_panel.roomPrimitiveSelected.connect(self._select_authored_room_primitive)
        self.viewport_panel.roomPrimitivesSelected.connect(self._select_authored_room_primitives)
        self.viewport_panel.roomPrimitiveMoved.connect(self._move_authored_room_primitive)
        self.viewport_panel.roomPrimitiveRotated.connect(self._rotate_authored_room_primitive)
        self.viewport_panel.roomPrimitiveScaled.connect(self._scale_authored_room_primitive)
        self.viewport_panel.roomPrimitivesTransformCommitted.connect(self._transform_authored_room_primitives)
        self.viewport_panel.authoredRoomSnapPreviewRequested.connect(self._preview_authored_room_drag_snap)
        self.viewport_panel.sceneObjectsTranslated.connect(self._translate_map_studio_scene_objects)
        self.viewport_panel.terrainBrushFrameRequested.connect(self.apply_map_studio_viewport_terrain_brush_frame)
        self.viewport_panel.terrainBrushStrokeCommitted.connect(self.commit_map_studio_viewport_terrain_brush_stroke)
        self.viewport_panel.terrainBrushOptionsChanged.connect(self._set_map_studio_terrain_brush_options)
        self.viewport_panel.terrainBrushSelected.connect(self._select_map_studio_terrain_brush)
        self.viewport_panel.terrainBrushStrengthChanged.connect(self._set_map_studio_terrain_brush_strength)
        self.viewport_panel.terrainBrushDeltaChanged.connect(self._set_map_studio_terrain_brush_delta)
        self.viewport_panel.terrainSculptModeChanged.connect(self._set_map_studio_terrain_sculpt_enabled)
        self.viewport_panel.modeMarkingMenuRequested.connect(self._open_map_studio_mode_marking_menu)
        self.viewport_panel.toolMarkingMenuRequested.connect(self._open_map_studio_tool_marking_menu)
        self.viewport_panel.hoverContextChanged.connect(self._handle_map_studio_hover_context_changed)
        self.viewport_panel.mapStudioRoomClicked.connect(self._handle_map_studio_room_clicked)
        self.viewport_panel.mapStudioRoomsRectSelected.connect(self._handle_map_studio_rooms_rect_selected)
        self.viewport_panel.transformGizmoModeChanged.connect(self._handle_map_studio_transform_gizmo_mode_changed)
        self.viewport_panel.undoShortcutRequested.connect(self.undo_map_studio_command)
        self.viewport_panel.redoShortcutRequested.connect(self.redo_map_studio_command)
        self.viewport_panel.deleteShortcutRequested.connect(self.delete_map_studio_current_selection)
        self.viewport_panel.groundSnapShortcutRequested.connect(self.snap_map_studio_selected_placement_to_ground)
        self.viewport_panel.pieMoveInputChanged.connect(self._handle_map_studio_pie_move_input)
        self.viewport_panel.pieDestinationRequested.connect(self._set_map_studio_pie_destination)
        self.viewport_panel.pieCameraInputChanged.connect(self._handle_map_studio_pie_camera_input)
        self.viewport_panel.pieStopRequested.connect(self._stop_map_studio_pie)
        gameplay_signal = getattr(self.viewport_panel, "pieGameplayActionRequested", None)
        if gameplay_signal is not None:
            gameplay_signal.connect(self._handle_map_studio_pie_gameplay_action)
        self.viewport_panel.componentExtrudeCommitted.connect(self._commit_map_studio_component_extrude)
        self.viewport_panel.componentExtrudePreviewRequested.connect(self._preview_map_studio_component_extrude)
        self.viewport_panel.componentExtrudePreviewCancelled.connect(self._cancel_map_studio_component_preview)
        self.viewport_panel.componentBevelCommitted.connect(self._commit_map_studio_component_bevel)
        self.viewport_panel.componentBevelPreviewRequested.connect(self._preview_map_studio_component_bevel)
        self.viewport_panel.componentBevelPreviewCancelled.connect(self._cancel_map_studio_component_preview)
        self.viewport_panel.modelingToolGestureCommitted.connect(self._commit_map_studio_modeling_tool_gesture)
        self.viewport_panel.texturePaintStrokeBegan.connect(self._begin_map_studio_texture_paint_stroke)
        self.viewport_panel.texturePaintSampleRequested.connect(self._append_map_studio_texture_paint_sample)
        self.viewport_panel.texturePaintStrokeCommitted.connect(self._commit_map_studio_texture_paint_stroke)
        self.viewport_panel.texturePaintStrokeCancelled.connect(self._cancel_map_studio_texture_paint_stroke)
        self.validation_panel.issueActivated.connect(self.select_item)
        self.readiness_panel.gameTestRequested.connect(self.record_game_smoke_proof)
        self.readiness_panel.launchHandoffRequested.connect(self.open_map_studio_launch_handoff)
        self.workflow_panel.newProjectRequested.connect(self.new_kmap)
        self.workflow_panel.openProjectRequested.connect(self.open_kmap)
        self.workflow_panel.saveProjectRequested.connect(self.save_kmap)
        self.workflow_panel.renameSelectedRequested.connect(self.rename_selected)
        self.workflow_panel.duplicateSelectedRequested.connect(self.duplicate_selected)
        self.workflow_panel.deleteSelectedRequested.connect(self.delete_selected)
        self.workflow_panel.focusSelectedRequested.connect(self.viewport_panel.focus_selected)
        self.workflow_panel.builderRequested.connect(self.show_map_studio_builder)
        self.workflow_panel.geometryToolsRequested.connect(self.show_map_studio_geometry_tools)
        self.workflow_panel.starterRoomRequested.connect(self.create_map_studio_starter_room)
        self.workflow_panel.doorwayBlockoutRequested.connect(self.create_map_studio_doorway_blockout)
        self.workflow_panel.corridorRequested.connect(self.create_map_studio_corridor)
        self.workflow_panel.starterTerrainRequested.connect(self.create_map_studio_starter_terrain)
        self.workflow_panel.terrainToolsRequested.connect(self.show_map_studio_terrain_tools)
        self.workflow_panel.lightingToolsRequested.connect(self.show_map_studio_lighting_tools)
        self.workflow_panel.placementToolsRequested.connect(self.show_map_studio_placement_tools)
        self.workflow_panel.scriptToolsRequested.connect(self.show_map_studio_script_tools)
        self.workflow_panel.testPlaceableRequested.connect(self.add_map_studio_test_placeable)
        self.workflow_panel.walkmeshToolsRequested.connect(self.show_map_studio_walkmesh_tools)
        self.workflow_panel.validateRequested.connect(self.validate_kmap)
        self.workflow_panel.stageRequested.connect(lambda: self.stage_authored_module(self.export_panel.dry_run.isChecked()))
        self.workflow_panel.installRequested.connect(lambda: self.install_authored_module(self.export_panel.dry_run.isChecked()))
        self.workflow_panel.launchHandoffRequested.connect(self.open_map_studio_launch_handoff)
        self.workflow_panel.proofRequested.connect(self.record_game_smoke_proof)
        self.properties.transformChanged.connect(self._set_transform)
        self.properties.visibilityChanged.connect(lambda item_id, value: self._set_visibility(item_id, value))
        self.properties.lockChanged.connect(lambda item_id, value: self._set_locked(item_id, value))
        self.properties.propertyChanged.connect(self._set_property)
        self.properties.transitionChanged.connect(self._set_authored_gameplay_transition)
        self.properties.cameraChanged.connect(self._set_authored_gameplay_camera_properties)
        self.properties.roomLightChanged.connect(self._set_authored_room_light_properties)
        self.pie_context_panel.playerContextChanged.connect(self._set_map_studio_pie_player_context)
        self.pie_context_panel.starterOverrideChanged.connect(self._set_map_studio_pie_starter_override)
        self.pie_context_panel.previewRequested.connect(self._update_map_studio_pie_opening_preview)
        self.pie_context_panel.resetRequested.connect(self._reset_map_studio_pie_context)
        self.export_panel.exportRequested.connect(self.export_fbx)
        self.export_panel.targetGameRequested.connect(self._retarget_map_studio_export_game)
        self.export_panel.devTestModuleRequested.connect(self.stage_dev_test_module)
        self.export_panel.authoredModuleRequested.connect(self.export_authored_module)
        self.export_panel.authoredModuleStageRequested.connect(self.stage_authored_module)
        self.export_panel.authoredModuleInstallRequested.connect(self.install_authored_module)
        self.export_panel.builderFixRequested.connect(self.show_map_studio_builder)
        self.export_panel.walkmeshFixRequested.connect(self.show_map_studio_walkmesh_tools)
        self.export_panel.placementFixRequested.connect(self.show_map_studio_placement_tools)
        self.export_panel.validateRequested.connect(self.validate_kmap)
        self.export_panel.selectFixTargetRequested.connect(self._select_map_studio_export_fix_target)
        self.workflow_tabs.currentChanged.connect(self._reset_map_studio_workflow_scroll)
        for tab in (self.rooms_tab, self.walkmesh_tab, self.porter_tab, self.builder_tab, self.blueprints_tab):
            tab.actionRequested.connect(self._handle_tab_action)
        self.builder_tab.primitivePresetRequested.connect(self.create_authored_room_preset)
        self.builder_tab.roomOperationRequested.connect(self.apply_authored_room_operation)
        self.builder_tab.floorPlanExtrusionRequested.connect(self.apply_authored_floor_plan_extrusion)
        self.builder_tab.floorPlanOpeningRequested.connect(self.set_authored_floor_plan_wall_opening)
        self.builder_tab.floorPlanOpeningMarkerRequested.connect(self.create_authored_opening_transition_marker)
        self.builder_tab.floorPlanVertexSnapPreviewRequested.connect(self.preview_authored_floor_plan_vertex_snap_candidates)
        self.builder_tab.floorPlanVertexSnapRequested.connect(self.snap_authored_floor_plan_vertex)
        self.builder_tab.floorPlanVertexWeldRequested.connect(self.weld_authored_floor_plan_vertices)
        self.builder_tab.floorPlanVertexFlattenRequested.connect(self.flatten_authored_floor_plan_vertices)
        self.builder_tab.floorPlanVertexCleanupRequested.connect(self.cleanup_authored_floor_plan_vertices)
        self.builder_tab.floorPlanVertexMirrorRequested.connect(self.mirror_authored_floor_plan_vertices)
        self.builder_tab.floorPlanFaceFillRequested.connect(self.fill_authored_floor_plan_face)
        self.builder_tab.floorPlanFaceSplitRequested.connect(self.split_authored_floor_plan_face)
        self.builder_tab.floorPlanFaceTriangulateRequested.connect(self.triangulate_authored_floor_plan_face)
        self.builder_tab.floorPlanNormalsCleanupRequested.connect(self.cleanup_authored_floor_plan_normals)
        self.builder_tab.terrainOperationRequested.connect(self.apply_authored_terrain_operation)
        self.builder_tab.terrainLiveBrushFrameRequested.connect(self.preview_map_studio_terrain_sculpt_frame)
        self.builder_tab.starterRoomRequested.connect(self.create_map_studio_starter_room)
        self.builder_tab.starterTerrainRequested.connect(self.create_map_studio_starter_terrain)
        self.builder_tab.terrainCreateRequested.connect(self._create_and_focus_map_studio_terrain_patch)
        self.builder_tab.terrainDressingRequested.connect(self.show_map_studio_placement_tools)
        self.builder_tab.terrainPaintRequested.connect(self._show_map_studio_texture_paint_workflow)
        self.builder_tab.snapRoomsAtDoorwayRequested.connect(self._snap_rooms_at_doorway)
        self.builder_tab.snapRoomsToGridRequested.connect(self._snap_selected_authored_rooms_to_grid)
        self.builder_tab.buildSectionChanged.connect(self._on_map_studio_build_section_changed)
        self.builder_tab.buildingToolChanged.connect(self._set_map_studio_building_tool)
        self.builder_tab.buildingSettingsChanged.connect(self.viewport_panel.update_pascal_building_settings)
        self.builder_tab.buildingLevelCreateRequested.connect(self._add_map_studio_building_level)
        self.builder_tab.buildingLevelViewChanged.connect(self._apply_map_studio_level_presentation)
        self.builder_tab.browseVanillaRoomKitsRequested.connect(self._add_room_from_module)
        self.builder_tab.spatialPlanVisibilityChanged.connect(
            self._refresh_map_studio_spatial_design_overlay
        )
        self.builder_tab.spatialPlanAuditRequested.connect(
            self._audit_map_studio_spatial_design
        )
        for combo_name in ("terrainRoomComboBox", "terrainBrushComboBox"):
            combo = getattr(self.builder_tab, combo_name, None)
            if combo is not None:
                combo.currentIndexChanged.connect(lambda _index=0: self._sync_map_studio_terrain_brush_context())
        for control_name in (
            "terrainRadiusSpinBox",
            "terrainHardnessSpinBox",
            "terrainSmoothStrengthSpinBox",
            "terrainDeltaSpinBox",
            "terrainSculptEnabledCheckBox",
        ):
            control = getattr(self.builder_tab, control_name, None)
            if control is not None:
                signal = getattr(control, "valueChanged", None) or getattr(control, "toggled", None)
                if signal is not None:
                    signal.connect(lambda _value=None: self._sync_map_studio_terrain_brush_context())
        self.builder_tab.roomRectangularUnionRequested.connect(self.merge_authored_floor_plan_rooms)
        self.builder_tab.floorPlanBridgeRequested.connect(self.bridge_authored_floor_plan_edges)
        self.builder_tab.roomPrimitiveAddRequested.connect(self.add_authored_room_primitive)
        self.builder_tab.roomPrimitiveTransformRequested.connect(self.apply_authored_room_primitive_transform)
        self.builder_tab.roomPrimitiveDimensionsPreviewRequested.connect(self.preview_authored_room_primitive_dimensions)
        self.builder_tab.roomPrimitiveDimensionsPreviewCancelled.connect(self.cancel_authored_room_primitive_dimensions_preview)
        self.builder_tab.roomPrimitiveDimensionsRequested.connect(self.apply_authored_room_primitive_dimensions)
        self.builder_tab.roomPrimitiveStyleRequested.connect(self.apply_authored_room_primitive_style)
        self.builder_tab.roomPrimitiveRemoveRequested.connect(self.remove_authored_room_primitive)
        self.builder_tab.roomPrimitiveSeparateRequested.connect(self.separate_authored_room_primitive)
        self.builder_tab.roomStyleRequested.connect(self.apply_authored_room_style)
        self.walkmesh_tab.roomSurfaceRequested.connect(self.apply_authored_walkmesh_surface)
        self.builder_tab.roomLightRequested.connect(self.add_authored_room_light)
        self.builder_tab.moduleEntryPointRequested.connect(self.set_authored_module_entry_point)
        self.builder_tab.gameplayPlacementRequested.connect(self.add_authored_gameplay_placement)
        self.builder_tab.gameplayPlacementStatusChanged.connect(self.workflow_panel.set_active_authoring_context)
        self.placement_tab.placementModeChanged.connect(self._set_map_studio_placement_mode)
        self.placement_tab.placementRequested.connect(self.add_authored_gameplay_placement)
        self.placement_tab.selectionRequested.connect(self.select_item)
        self.placement_tab.transformRequested.connect(self._apply_placement_tab_transform)
        self.placement_tab.creatureBehaviorRequested.connect(self._apply_placement_tab_creature_behavior)
        self.placement_tab.dialogueEditorRequested.connect(self._request_creature_dialogue_editor)
        self.placement_tab.actionRequested.connect(self._handle_placement_tab_action)
        self.placement_tab.thumbnailRequested.connect(self._render_map_studio_placement_thumbnail)
        self.placement_tab.customUtcImportRequested.connect(self._import_custom_creature_utc)
        self.terrain_kit_browser.thumbnailRequested.connect(self._render_map_studio_terrain_kit_thumbnail)
        self.terrain_kit_browser.refreshVanillaRequested.connect(self._refresh_vanilla_terrain_kit_catalog)
        self.terrain_kit_browser.collectionStyleChanged.connect(
            self.builder_tab.select_building_style
        )
        self.environment_kit_browser.thumbnailRequested.connect(self._render_map_studio_environment_kit_thumbnail)
        self.environment_kit_browser.refreshVanillaRequested.connect(self._refresh_vanilla_environment_kit_catalog)
        self.environment_kit_browser.statusChanged.connect(self.workflow_panel.set_active_authoring_context)
        self.environment_kit_browser.collectionStyleChanged.connect(
            lambda selection_id, kind: self.builder_tab.select_building_style(
                str(selection_id)[6:] if str(selection_id).startswith("style:") else f"kit:{selection_id}",
                kind,
            )
        )
        self.builder_tab.buildingStyleChanged.connect(
            self.environment_kit_browser.select_building_style
        )
        self.builder_tab.buildingStyleChanged.connect(
            self.terrain_kit_browser.select_building_style
        )
        self.builder_tab.buildingStyleChanged.connect(
            self.environment_tab.set_skybox_building_style
        )
        self.terrainCatalogProgress.connect(
            lambda message: self.terrain_kit_browser.set_refreshing(True, message)
        )
        self.terrainCatalogReady.connect(self._vanilla_terrain_catalog_ready)
        self.terrainCatalogFailed.connect(self._vanilla_terrain_catalog_failed)
        self.environmentCatalogProgress.connect(
            lambda message: self.environment_kit_browser.set_refreshing(True, message)
        )
        self.environmentCatalogReady.connect(self._vanilla_environment_catalog_ready)
        self.environmentCatalogFailed.connect(self._vanilla_environment_catalog_failed)
        self.environmentThumbnailReady.connect(self._apply_map_studio_environment_kit_thumbnail)
        self.placement_tab.statusChanged.connect(self.workflow_panel.set_active_authoring_context)
        self.texture_paint_tab.importRequested.connect(self.import_project_texture)
        self.texture_paint_tab.assignRequested.connect(self._assign_map_studio_texture_paint_target)
        self.texture_paint_tab.paintEnabledChanged.connect(self._set_map_studio_texture_paint_enabled)
        self.texture_paint_tab.targetChanged.connect(self._reset_map_studio_texture_paint_session)
        self.texture_paint_tab.brushChanged.connect(self._update_map_studio_texture_paint_brush_context)
        self.texture_paint_tab.brushSourceRequested.connect(self._choose_map_studio_texture_paint_brush_source)
        self.texture_paint_tab.brushSourceCleared.connect(self._clear_map_studio_texture_paint_brush_source)
        self.texture_paint_tab.makeUsedEditableRequested.connect(self._make_used_map_textures_editable)
        self.texture_paint_tab.applyRequested.connect(self._apply_map_studio_texture_changes)
        self.environment_tab.worldSettingsRequested.connect(self._apply_map_studio_world_settings)
        self.environment_tab.lightmapApplyRequested.connect(self._apply_map_studio_surface_lightmap)
        self.environment_tab.skyboxCreateRequested.connect(self._create_map_studio_five_face_skybox)
        self.environment_tab.skyPanoramaRequested.connect(self._create_map_studio_panorama_skybox)
        self.environment_tab.skyTrafficCreateRequested.connect(self._create_map_studio_sky_traffic)
        self.builder_tab.modelingContextChanged.connect(self.workflow_panel.set_active_authoring_context)
        self.builder_tab.scriptHookRequested.connect(self.set_authored_script_hook)
        self.builder_tab.scriptEditorRequested.connect(self._request_script_editor)
        self.outliner_action.toggled.connect(lambda visible: self.outliner.setVisible(visible))
        self.properties_action.toggled.connect(lambda visible: self.properties.setVisible(visible))
        self.viewport_action.toggled.connect(lambda visible: self.viewport_panel.setVisible(visible))
        self.focus_viewport_action.toggled.connect(self._set_map_studio_viewport_maximized)
        self.validation_action.toggled.connect(self._set_map_studio_validation_output_visible)
        self.validation_output_dock.visibilityChanged.connect(
            self._sync_map_studio_validation_output_action
        )
        self._map_studio_pie_timer = QtCore.QTimer(self)
        self._map_studio_pie_timer.setObjectName("mapStudioPIETimer")
        self._map_studio_pie_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._map_studio_pie_timer.setInterval(16)
        self._map_studio_pie_timer.timeout.connect(self._tick_map_studio_pie)

    # Property/transform-scoped commands whose undo/redo can repaint just the
    # touched placement or light instead of reloading the whole map.
    _TARGETED_HISTORY_PLACEMENT_ACTIONS = frozenset({
        "map_studio.gameplay.move_placement",
        "map_studio.gameplay.rename_placement",
        "map_studio.gameplay.edit_creature_behavior",
        "map_studio.gameplay.edit_camera",
        "map_studio.gameplay.set_transition",
        "map_studio.gameplay.snap_placement_to_walkmesh",
    })
    _TARGETED_HISTORY_LIGHT_ACTIONS = frozenset({
        "map_studio.lighting.move_room_light",
        "map_studio.lighting.edit_room_light",
        "map_studio.lighting.rename_room_light",
    })

    def _refresh_after_map_studio_history(self, result) -> None:
        """Repaint only what the undone/redone command could have touched.

        History restores replace the whole KMAP snapshot, but a placement or
        light property command only changes that one object: promote its
        reverted values onto the live preview instead of reloading the map.
        Membership, topology, texture, and unknown commands keep the broad
        ``_refresh_all`` correctness fallback.
        """

        record = getattr(result, "record", None)
        action_key = str(getattr(record, "action_key", "") or "")
        metadata = dict(getattr(record, "metadata", {}) or {})
        if not tuple(getattr(record, "sidecar_patches", ()) or ()):
            if action_key in self._TARGETED_HISTORY_PLACEMENT_ACTIONS:
                placement_id = str(metadata.get("placement_id", "") or "")
                if placement_id:
                    self._refresh_map_studio_gameplay_change(
                        result.message,
                        placement_ids=(placement_id,),
                        refresh_outliner_labels=True,
                    )
                    return
            if action_key in self._TARGETED_HISTORY_LIGHT_ACTIONS:
                light_id = str(metadata.get("light_id", "") or "")
                if light_id:
                    self._refresh_map_studio_gameplay_change(
                        result.message,
                        light_ids=(light_id,),
                        refresh_markers=False,
                        refresh_placement_rows=False,
                        refresh_outliner_labels=True,
                    )
                    return
        self._refresh_all(result.message)

    def undo_map_studio_command(self) -> None:
        session = self._texture_paint_session
        if session is not None and bool(getattr(session, "stroke_active", False)):
            self._cancel_map_studio_texture_paint_stroke()
            self._log("Cancelled the active texture-paint stroke; press Undo again for earlier edits.")
            return
        try:
            result = self.controller.undo_map_studio_command()
        except Exception as exc:
            message = f"Undo blocked: {exc}"
            self.statusBar().showMessage(message, 7000)
            self._log(message)
            QtWidgets.QMessageBox.warning(self, "Undo Texture Change", message)
            self._update_map_studio_undo_redo_actions()
            return
        if result is None:
            self._update_map_studio_undo_redo_actions()
            self._log("Nothing to undo.")
            return
        if tuple(getattr(result.record, "sidecar_patches", ()) or ()):
            self._reset_map_studio_texture_paint_session()
        self._refresh_after_map_studio_history(result)
        self._reload_map_studio_texture_paint_after_history(result)

    def redo_map_studio_command(self) -> None:
        session = self._texture_paint_session
        if session is not None and bool(getattr(session, "stroke_active", False)):
            self._cancel_map_studio_texture_paint_stroke()
            self._log("Cancelled the active texture-paint stroke before Redo.")
            return
        try:
            result = self.controller.redo_map_studio_command()
        except Exception as exc:
            message = f"Redo blocked: {exc}"
            self.statusBar().showMessage(message, 7000)
            self._log(message)
            QtWidgets.QMessageBox.warning(self, "Redo Texture Change", message)
            self._update_map_studio_undo_redo_actions()
            return
        if result is None:
            self._update_map_studio_undo_redo_actions()
            self._log("Nothing to redo.")
            return
        if tuple(getattr(result.record, "sidecar_patches", ()) or ()):
            self._reset_map_studio_texture_paint_session()
        self._refresh_after_map_studio_history(result)
        self._reload_map_studio_texture_paint_after_history(result)

    def _reload_map_studio_texture_paint_after_history(self, result: object) -> None:
        """Refresh every file-backed texture touched by global Undo/Redo.

        The renderer refresh is independent of Paint mode.  Leaving Paint must
        not leave the software/GPU cache showing pixels that global Undo has
        already restored on disk.  Recreating the editable paint session is the
        only part gated on Paint mode.
        """

        record = getattr(result, "record", None)
        patches = tuple(getattr(record, "sidecar_patches", ()) or ())
        if not patches:
            return
        metadata = dict(getattr(record, "metadata", {}) or {})
        texture_id = str(metadata.get("texture_id") or "")
        metadata_resref = str(metadata.get("resref") or "").strip().lower()
        patched_paths = {Path(str(patch.path)).resolve() for patch in patches}
        patched_resrefs = {
            path.stem.lower()
            for path in patched_paths
            if path.suffix.lower() in {".tga", ".tpc", ".txi"}
        }
        if metadata_resref:
            patched_resrefs.add(metadata_resref)

        affected: dict[str, object] = {}
        for texture in tuple(getattr(self.project, "textures", ()) or ()):
            resref = str(getattr(texture, "resref", "") or "").strip().lower()
            candidate_id = str(getattr(texture, "texture_id", "") or "")
            source_value = str(getattr(texture, "path", "") or "").strip()
            source = resolve_project_texture_path(self.project, source_value) if source_value else None
            txi_value = str(dict(getattr(texture, "metadata", {}) or {}).get("txi_path") or "").strip()
            txi = resolve_project_texture_path(self.project, txi_value) if txi_value else None
            path_hit = any(
                path is not None and path.resolve() in patched_paths
                for path in (source, txi)
            )
            if candidate_id == texture_id or resref in patched_resrefs or path_hit:
                if resref:
                    affected[resref] = texture

        viewport = getattr(self.viewport_panel, "viewport", None)
        updater = getattr(viewport, "update_texture_regions", None)
        invalidator = getattr(viewport, "invalidate_texture", None)
        for resref in sorted(patched_resrefs | set(affected)):
            texture = affected.get(resref)
            source_value = str(getattr(texture, "path", "") or "").strip() if texture is not None else ""
            source = resolve_project_texture_path(self.project, source_value) if source_value else None
            if source is not None and source.is_file() and callable(updater):
                try:
                    updater(resref, self._map_studio_texture_view_image(source), None)
                except Exception as exc:
                    self._log(f"Texture history renderer refresh failed for {resref}: {exc}")
            elif callable(invalidator):
                try:
                    invalidator(resref)
                except Exception as exc:
                    self._log(f"Texture history renderer invalidation failed for {resref}: {exc}")

        paint_tab = getattr(self, "texture_paint_tab", None)
        if paint_tab is None or not paint_tab.paint_button.isChecked():
            return
        selected = paint_tab.selected_texture_id()
        wanted = texture_id if self._project_texture_for_id(texture_id) is not None else selected
        if not wanted or self._project_texture_for_id(wanted) is None:
            return
        try:
            self._load_map_studio_texture_paint_session(wanted)
        except Exception as exc:
            paint_tab.set_status(f"Texture history applied, but live preview reload failed: {exc}")

    def _update_map_studio_undo_redo_actions(self) -> None:
        undo_label = self.controller.command_history.undo_label
        redo_label = self.controller.command_history.redo_label
        self.undo_action.setEnabled(self.controller.can_undo_map_studio_command())
        self.redo_action.setEnabled(self.controller.can_redo_map_studio_command())
        self.undo_action.setText(f"Undo {undo_label}" if undo_label else "Undo")
        self.redo_action.setText(f"Redo {redo_label}" if redo_label else "Redo")

    def new_kmap(self) -> None:
        if not self._confirm_discard_or_save():
            return
        dialog = _MapStudioNewProjectDialog(
            self,
            module_root=str(getattr(self.project, "name", "") or "grdev01"),
            game=str(getattr(self.project, "game", "") or "K1"),
            author=str(getattr(self.project, "author", "") or ""),
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            project = self.controller.new_project(**dialog.values())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "New Map Studio KMAP", str(exc))
            return
        self._reset_map_studio_texture_paint_session()
        self._refresh_all(f"Created Map Studio KMAP {project.name} for {project.game}.")

    def open_kmap(self) -> None:
        if not self._confirm_discard_or_save():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open KMAP", "", "GhostRigger KMAP (*.kmap);;JSON files (*.json);;All files (*.*)")
        if not path:
            return
        try:
            self.controller.open_project(path, resource_manager=self.resource_manager)
            self._reset_map_studio_texture_paint_session()
            self._refresh_all(f"Opened {Path(path).name}.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open KMAP", str(exc))

    def save_kmap(self) -> None:
        if not self.project.path:
            self.save_kmap_as()
            return
        try:
            self.controller.save_project()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Map Studio KMAP", str(exc))
            self.statusBar().showMessage(f"Save failed: {exc}", 7000)
            return
        self._refresh_all(f"Saved {Path(self.project.path).name}.")

    def save_kmap_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save KMAP As", f"{self.project.name}.kmap", "GhostRigger KMAP (*.kmap)")
        if not path:
            return
        if not path.lower().endswith(".kmap"):
            path += ".kmap"
        try:
            self.controller.save_project(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Map Studio KMAP As", str(exc))
            self.statusBar().showMessage(f"Save As failed: {exc}", 7000)
            return
        self._refresh_all(f"Saved {Path(path).name}.")

    def import_obj_map(self) -> None:
        """Analyze, review, and atomically commit one custom OBJ room."""

        if not str(getattr(self.project, "path", "") or "").strip():
            QtWidgets.QMessageBox.information(
                self,
                "Import OBJ Map",
                "Save this KMAP first. Map textures are stored beside it as project-relative assets.",
            )
            self.save_kmap_as()
            if not str(getattr(self.project, "path", "") or "").strip():
                return
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Custom OBJ Map",
            "",
            "Wavefront OBJ maps (*.obj);;All files (*.*)",
        )
        if not path:
            return

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        lock_existing_module = isinstance(payload, dict)
        self.import_obj_map_action.setEnabled(False)
        progress = QtWidgets.QProgressDialog(
            "Reading materials, scaling geometry, and finding connected floor components…",
            "",
            0,
            0,
            self,
        )
        progress.setObjectName("mapStudioObjImportProgressDialog")
        progress.setWindowTitle("Analyze OBJ Map")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        future = _MAP_STUDIO_OBJ_IMPORT_EXECUTOR.submit(
            self.controller.prepare_obj_room_import,
            path,
            module_root=str(getattr(self.project, "name", "") or ""),
            game=str(getattr(self.project, "game", "K2") or "K2"),
            source_units="auto",
            source_up_axis="y",
            center_xy=True,
            ground_to_zero=False,
        )
        wait_loop = QtCore.QEventLoop(self)
        poll_timer = QtCore.QTimer(self)
        poll_timer.setInterval(40)

        def poll_obj_analysis() -> None:
            if future.done():
                poll_timer.stop()
                wait_loop.quit()

        poll_timer.timeout.connect(poll_obj_analysis)
        poll_timer.start()
        if not future.done():
            wait_loop.exec()
        try:
            preparation = future.result()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import OBJ Map", str(exc))
            self.statusBar().showMessage(f"OBJ analysis failed: {exc}", 8000)
            return
        finally:
            poll_timer.stop()
            poll_timer.deleteLater()
            wait_loop.deleteLater()
            progress.close()
            progress.deleteLater()
            self.import_obj_map_action.setEnabled(True)

        dialog = MapStudioObjRoomImportDialog(
            preparation,
            self,
            lock_existing_module=lock_existing_module,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Importing OBJ textures and generating the reviewed KOTOR walkmesh…")
        try:
            result = self.controller.commit_obj_room_import(
                preparation,
                selected_component_ids=dialog.selected_component_ids(),
                module_root=dialog.module_root(),
                room_resref=dialog.room_resref(),
                game=dialog.target_game(),
                target_face_budget=dialog.target_face_budget(),
                texture_max_dimension=dialog.texture_max_dimension(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import OBJ Map", str(exc))
            self.statusBar().showMessage(f"OBJ import rolled back: {exc}", 9000)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        generated_faces = int(result.walkmesh_report.get("generated_face_count", 0) or 0)
        message = (
            f"Imported {result.room_resref} with {len(result.imported_texture_resrefs)} texture(s) and "
            f"{generated_faces} structurally validated WOK face(s). Retail KOTOR traversal proof is still required."
        )
        self._reset_map_studio_texture_paint_session()
        self._refresh_all(message)

    def import_project_texture(self) -> None:
        """Import a unique, editable texture without rebuilding the viewport."""

        if not self.project.path:
            self.save_kmap_as()
            if not self.project.path:
                return
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Texture to Map Studio Project",
            "",
            "Texture images (*.png *.tga *.dds *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        try:
            asset = self.controller.import_project_texture(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Texture", str(exc))
            return
        message = (
            f"Imported {asset.resref} ({asset.width}x{asset.height}) as a unique project TGA; "
            "it will be bundled with the next module export."
        )
        self.statusBar().showMessage(message, 7000)
        self.texture_paint_tab.set_project(self.project)
        self._sync_map_studio_texture_apply_state()
        self.viewport_panel.set_project_texture_paths(self.project)
        self._refresh_map_studio_geometry_change(
            message,
            rebuild_viewport_model=False,
            refresh_scene_tree=True,
        )

    def _project_texture_for_id(self, texture_id: str):
        wanted = str(texture_id or "")
        return next(
            (texture for texture in tuple(self.project.textures or ()) if str(texture.texture_id) == wanted),
            None,
        )

    def _sync_map_studio_texture_apply_state(self) -> None:
        """Reflect controller-side draft/hash drift without polling per dab."""

        try:
            pending = tuple(self.controller.project_texture_apply_pending_resrefs())
            required = bool(pending or self.controller.has_unapplied_project_texture_changes())
        except Exception as exc:
            self._log(f"Room texture Apply state refresh failed: {exc}")
            return
        self.texture_paint_tab.set_apply_state(required, pending)

    def _reset_map_studio_texture_paint_session(self, _texture_id: str = "") -> None:
        self._texture_paint_upload_timer.stop()
        session = self._texture_paint_session
        if session is not None and session.stroke_active:
            session.cancel_stroke()
        self._texture_paint_session = None
        self._texture_paint_texture_id = ""
        self._texture_paint_resref = ""
        self._texture_paint_view_image = None
        self._texture_paint_accepting_stroke = False
        self._texture_paint_preview_error = ""
        self._update_map_studio_undo_redo_actions()

    @staticmethod
    def _map_studio_texture_view_image(source: Path):
        """Decode one authored texture into the renderer's bottom-up image."""

        width, height, rgba = decode_image_rgba(source.read_bytes())
        from PIL import Image

        top_down = Image.frombytes("RGBA", (width, height), rgba)
        bottom_up = top_down.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # This image is already converted to the renderer's upload orientation.
        # Keep this explicit for both paint sessions and history refreshes.
        bottom_up._gr_gpu_uv_v_flip = False
        return bottom_up

    def _load_map_studio_texture_paint_session(self, texture_id: str) -> TexturePaintSession:
        texture = self._project_texture_for_id(texture_id)
        if texture is None:
            raise ValueError("Choose a project texture before painting.")
        source = resolve_project_texture_path(self.project, texture.path)
        if not source.is_file():
            raise FileNotFoundError(f"Project texture payload is missing: {source}")
        width, height, rgba = decode_image_rgba(source.read_bytes())
        session = TexturePaintSession(width, height, rgba, tile_size=64, max_history=64)
        bottom_up = self._map_studio_texture_view_image(source)
        self._texture_paint_session = session
        self._texture_paint_texture_id = str(texture.texture_id)
        self._texture_paint_resref = str(texture.resref or "").lower()
        self._texture_paint_view_image = bottom_up
        updater = getattr(getattr(self.viewport_panel, "viewport", None), "update_texture_regions", None)
        if callable(updater):
            updater(self._texture_paint_resref, bottom_up, None)
        self._update_map_studio_texture_paint_brush_context(self.texture_paint_tab.current_brush())
        return session

    def _update_map_studio_texture_paint_brush_context(self, brush: object | None = None) -> None:
        """Keep the UV-aware viewport cursor synchronized with compact brush controls."""

        active_brush = brush if brush is not None else self.texture_paint_tab.current_brush()
        session = self._texture_paint_session
        texture_size = (
            (int(session.width), int(session.height))
            if session is not None
            else (1024, 1024)
        )
        setter = getattr(self.viewport_panel, "set_texture_paint_brush_context", None)
        if callable(setter):
            setter(active_brush, texture_size=texture_size, resref=self._texture_paint_resref)

    def _set_map_studio_texture_paint_enabled(self, enabled: bool) -> None:
        wanted = bool(enabled)
        if wanted:
            texture_id = self.texture_paint_tab.selected_texture_id()
            try:
                if self._texture_paint_session is None or self._texture_paint_texture_id != texture_id:
                    self._load_map_studio_texture_paint_session(texture_id)
            except Exception as exc:
                self.texture_paint_tab.set_status(str(exc))
                QtWidgets.QMessageBox.warning(self, "Texture Paint", str(exc))
                self.texture_paint_tab.stop_painting()
                return
            self.workflow_tabs.setCurrentWidget(self.texture_paint_tab)
            self._set_map_studio_toolbar_edit_mode("Texture Paint")
            albedo_index = self.toolbar.view_mode.findText("Albedo")
            if albedo_index >= 0:
                self.toolbar.view_mode.setCurrentIndex(albedo_index)
            self.workflow_panel.set_active_authoring_context(
                "Texture Paint: nearest-visible face, diffuse UV0, dirty-tile live feedback, one undo item per drag"
            )
        self.viewport_panel.set_texture_paint_interaction(wanted)
        if not wanted:
            self._texture_paint_accepting_stroke = False
        self._update_map_studio_undo_redo_actions()

    def _clear_map_studio_texture_paint_brush_source(self) -> None:
        self._texture_paint_stamp_name = ""
        self._texture_paint_stamp_size = (0, 0)
        self._texture_paint_stamp_rgba = b""
        self.texture_paint_tab.set_brush_source("")
        self.texture_paint_tab.set_status("Solid-color brush selected.")

    def _choose_map_studio_texture_paint_brush_source(self) -> None:
        name = MapStudioTextureBrowserDialog.pick_texture(
            getattr(self, "resource_manager", None),
            self,
            project=self.project,
            game=str(getattr(self.project, "game", "") or ""),
        )
        if not name:
            return
        project_texture = next(
            (texture for texture in tuple(self.project.textures or ()) if str(texture.resref).lower() == name.lower()),
            None,
        )
        try:
            if project_texture is not None and str(project_texture.path or "").strip():
                source = resolve_project_texture_path(self.project, project_texture.path)
                width, height, rgba = decode_image_rgba(source.read_bytes())
            else:
                manager = getattr(self, "resource_manager", None)
                if manager is None:
                    raise ValueError("Connect a KOTOR resource manager before choosing a game texture brush.")
                image = manager.load_texture_image(name, str(getattr(self.project, "game", "K1") or "K1"))
                if image is None:
                    raise ValueError(f"Could not decode game texture {name}.")
                from PIL import Image

                image = image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                width, height = image.size
                rgba = image.tobytes()
            if width > 256 or height > 256:
                from PIL import Image

                stamp = Image.frombytes("RGBA", (width, height), rgba)
                stamp.thumbnail((256, 256), Image.Resampling.LANCZOS)
                width, height = stamp.size
                rgba = stamp.tobytes()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Texture Paint Brush", str(exc))
            return
        self._texture_paint_stamp_name = name
        self._texture_paint_stamp_size = (int(width), int(height))
        self._texture_paint_stamp_rgba = bytes(rgba)
        self.texture_paint_tab.set_brush_source(name)
        self.texture_paint_tab.set_status(f"Brush source {name} ready ({width}x{height} stamp).")

    def _assign_map_studio_texture_paint_target(self, texture_id: str) -> None:
        texture = self._project_texture_for_id(texture_id)
        if texture is None:
            self.texture_paint_tab.set_status("Choose or import a project texture first.")
            return
        context = self.viewport_panel.current_map_studio_hover_context()
        if context is None or str(getattr(context, "component_type", "") or "") != "face":
            self.texture_paint_tab.set_status("Hover the visible face that should receive this unique texture, then assign again.")
            return
        room_resref = str(getattr(context, "room_resref", "") or "").strip().lower()
        mesh_role = str(getattr(context, "mesh_role", "") or "")
        face_index = int(getattr(context, "face_index", -1))
        if mesh_role.startswith("stock_room"):
            ok, message = self.controller.convert_stock_room_to_imported_mesh(
                room_resref=room_resref,
                resource_manager=getattr(self, "resource_manager", None),
            )
            if not ok:
                self.texture_paint_tab.set_status(message)
                return
            suffix = mesh_role.rsplit("_", 1)[-1]
            mesh_role = imported_mesh_surface_role(int(suffix) if suffix.isdigit() else 0)
        ok, message = self.controller.set_imported_mesh_room_face_texture(
            room_resref=room_resref,
            mesh_role=mesh_role,
            face_indices=(face_index,),
            texture=str(texture.resref or ""),
        )
        self.texture_paint_tab.set_status(message)
        self.statusBar().showMessage(message, 6000)
        if ok:
            self._refresh_map_studio_geometry_change(message)

    @staticmethod
    def _used_map_diffuse_resrefs(preview_model: object | None) -> tuple[str, ...]:
        """Return room diffuse materials only; gameplay actors and lightmaps stay out."""

        if preview_model is None:
            return ()
        all_nodes = getattr(preview_model, "all_nodes", None)
        nodes = tuple(all_nodes() or ()) if callable(all_nodes) else ()
        if not nodes:
            root = getattr(preview_model, "root_node", None)
            pending = [root] if root is not None else []
            collected: list[object] = []
            while pending:
                node = pending.pop()
                collected.append(node)
                pending.extend(tuple(getattr(node, "children", ()) or ()))
            nodes = tuple(collected)
        values: list[str] = []
        for node in nodes:
            if not bool(getattr(node, "is_mesh", False)):
                continue
            role = str(getattr(node, "_gr_map_studio_mesh_role", "") or "").strip().lower()
            if role.startswith("stock_") and not role.startswith("stock_room"):
                continue
            texture = str(
                getattr(node, "texture_clean", "") or getattr(node, "texture", "") or ""
            ).strip().strip("\x00").lower()
            if texture and texture not in {"null", "none", "default"}:
                values.append(texture)
        return tuple(dict.fromkeys(values))

    def _make_used_map_textures_editable(self) -> None:
        """Clone loaded-room diffuse textures as one cancellable undo command."""

        if not str(getattr(self.project, "path", "") or "").strip():
            self.save_kmap_as()
            if not str(getattr(self.project, "path", "") or "").strip():
                return
        preview = getattr(self.viewport_panel, "_room_preview_model", None)
        used = self._used_map_diffuse_resrefs(preview)
        editable = {
            str(getattr(texture, "resref", "") or "").strip().lower()
            for texture in tuple(getattr(self.project, "textures", ()) or ())
            if str(getattr(texture, "path", "") or "").strip()
            and Path(str(getattr(texture, "path", "") or "")).suffix.lower() == ".tga"
            and str(dict(getattr(texture, "metadata", {}) or {}).get("asset_kind") or "").lower()
            != "map_studio_lightmap"
        }
        wanted = tuple(resref for resref in used if resref not in editable)
        if not wanted:
            self.texture_paint_tab.set_status("Every used room diffuse texture is already editable.")
            return

        progress = QtWidgets.QProgressDialog(
            "Preparing loaded-room diffuse textures…",
            "Cancel",
            0,
            len(wanted),
            self,
        )
        progress.setObjectName("mapStudioRoomTextureCloneProgressDialog")
        progress.setWindowTitle("Clone Room Textures")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        cancel_event = Event()
        progress_updates: SimpleQueue[tuple[int, int, str]] = SimpleQueue()
        progress.canceled.connect(cancel_event.set)

        future = _MAP_STUDIO_TEXTURE_CLONE_EXECUTOR.submit(
            self.controller.clone_game_textures_for_paint,
            wanted,
            resource_manager=getattr(self, "resource_manager", None),
            progress_callback=lambda completed, total, resref: progress_updates.put(
                (int(completed), int(total), str(resref or ""))
            ),
            cancel_requested=cancel_event.is_set,
        )
        wait_loop = QtCore.QEventLoop(self)
        poll_timer = QtCore.QTimer(self)
        poll_timer.setInterval(40)

        def poll_clone_job() -> None:
            while True:
                try:
                    completed, total, resref = progress_updates.get_nowait()
                except Empty:
                    break
                progress.setMaximum(max(1, total))
                progress.setValue(max(0, min(completed, total)))
                progress.setLabelText(
                    f"Made {completed} of {total} room diffuse textures editable\n{resref}"
                )
            if progress.wasCanceled() and not cancel_event.is_set():
                cancel_event.set()
            if cancel_event.is_set() and not future.done():
                progress.setCancelButton(None)
                progress.setLabelText(
                    "Canceling after the current texture finishes…\n"
                    "Project files will be rolled back automatically."
                )
                progress.show()
            if future.done():
                poll_timer.stop()
                wait_loop.quit()

        poll_timer.timeout.connect(poll_clone_job)
        poll_timer.start()
        poll_clone_job()
        if not future.done():
            wait_loop.exec()
        poll_clone_job()

        try:
            assets = future.result()
        except MapStudioTextureCloneCancelled:
            message = "Making room diffuse textures editable was cancelled; no project textures were changed."
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 6000)
            self._log(message)
            return
        except Exception as exc:
            message = f"Could not make the loaded-room diffuse texture set editable: {exc}"
            self.texture_paint_tab.set_status(message)
            QtWidgets.QMessageBox.warning(self, "Clone Room Textures", message)
            return
        finally:
            poll_timer.stop()
            poll_timer.deleteLater()
            wait_loop.deleteLater()
            progress.close()
            progress.deleteLater()
        self.viewport_panel.set_project_texture_paths(self.project)
        self.texture_paint_tab.set_project(self.project)
        self._sync_map_studio_texture_apply_state()
        self.texture_paint_tab.set_material_inventory(used, self.project)
        message = (
            f"Made {len(assets)} used room diffuse texture(s) editable in one undoable batch. "
            "Select a room material, paint it live, then Apply Textures."
        )
        self.texture_paint_tab.set_status(message)
        self.statusBar().showMessage(message, 8000)
        self._update_map_studio_undo_redo_actions()

    def _begin_map_studio_texture_paint_stroke(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        context = values.get("context")
        material = str(getattr(context, "material", "") or "").strip().lower()
        if material != self._texture_paint_resref:
            self._texture_paint_accepting_stroke = False
            self.texture_paint_tab.set_status(
                f"Hovered face uses {material or 'no diffuse texture'}; assign {self._texture_paint_resref} to it before painting."
            )
            return
        self._texture_paint_preview_error = ""
        session = self._texture_paint_session
        if session is None:
            return
        if session.stroke_active:
            session.cancel_stroke()
        brush = replace(
            self.texture_paint_tab.current_brush(),
            stamp_name=self._texture_paint_stamp_name,
            stamp_size=self._texture_paint_stamp_size,
            stamp_rgba=self._texture_paint_stamp_rgba,
        )
        session.begin_stroke(brush)
        self._texture_paint_accepting_stroke = True

    def _append_map_studio_texture_paint_sample(self, payload: object) -> None:
        session = self._texture_paint_session
        if session is None or not session.stroke_active or not self._texture_paint_accepting_stroke:
            return
        values = dict(payload) if isinstance(payload, dict) else {}
        if bool(values.get("break_before", False)):
            session.break_stroke()
        context = values.get("context")
        if str(getattr(context, "material", "") or "").strip().lower() != self._texture_paint_resref:
            return
        uv = tuple(values.get("uv") or ())
        if len(uv) < 2:
            return
        session.append_sample((float(uv[0]), float(uv[1])), pressure=float(values.get("pressure", 1.0) or 1.0))
        if not self._texture_paint_upload_timer.isActive():
            self._texture_paint_upload_timer.start()

    def _flush_map_studio_texture_paint_tiles(self) -> None:
        """Upload at most once per display frame while pointer samples accumulate."""

        session = self._texture_paint_session
        if session is None:
            return
        self._apply_map_studio_texture_paint_tiles(session.pending_tile_payloads())

    def _publish_map_studio_texture_paint_preview(
        self,
        image: object,
        regions: object,
        *,
        finalize: bool,
    ) -> bool:
        """Call the explicit renderer contract once and expose any failure."""

        updater = getattr(getattr(self.viewport_panel, "viewport", None), "update_texture_regions", None)
        if not callable(updater):
            return True
        try:
            updater(
                self._texture_paint_resref,
                image,
                tuple(regions or ()),
                finalize=bool(finalize),
            )
        except Exception as exc:
            phase = "mipmap finalize" if finalize else "tile upload"
            detail = f"Live room-texture preview {phase} failed: {exc}"
            self._texture_paint_preview_error = detail
            message = f"{detail}. The paint stroke remains editable; fix the renderer before trusting the preview."
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 8000)
            self._log(message)
            return False
        return True

    def _apply_map_studio_texture_paint_tiles(self, tiles, *, finalize: bool = False) -> bool:
        image = self._texture_paint_view_image
        session = self._texture_paint_session
        if image is None or session is None:
            return False
        from PIL import Image

        regions: list[tuple[int, int, int, int]] = []
        for tile in tuple(tiles or ()):
            tile_image = Image.frombytes("RGBA", (int(tile.width), int(tile.height)), bytes(tile.rgba))
            tile_image = tile_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            bottom_y = int(session.height) - (int(tile.y) + int(tile.height))
            image.paste(tile_image, (int(tile.x), bottom_y))
            regions.append((int(tile.x), bottom_y, int(tile.width), int(tile.height)))
        if not regions:
            return True
        return self._publish_map_studio_texture_paint_preview(
            image,
            regions,
            finalize=bool(finalize),
        )

    def _commit_map_studio_texture_paint_stroke(self) -> None:
        session = self._texture_paint_session
        accepting = self._texture_paint_accepting_stroke
        self._texture_paint_accepting_stroke = False
        if session is None or not session.stroke_active:
            return
        if not accepting:
            session.cancel_stroke()
            return
        self._texture_paint_upload_timer.stop()
        self._flush_map_studio_texture_paint_tiles()
        result = session.end_stroke()
        if not result.changed:
            return
        if self._texture_paint_view_image is not None:
            self._publish_map_studio_texture_paint_preview(
                self._texture_paint_view_image,
                (),
                finalize=True,
            )
        texture_id = self._texture_paint_texture_id
        try:
            asset = self.controller.commit_project_texture_paint(
                texture_id,
                session,
                stroke_result=result,
            )
        except Exception as exc:
            self._reset_map_studio_texture_paint_session()
            try:
                if texture_id and self._project_texture_for_id(texture_id) is not None:
                    self._load_map_studio_texture_paint_session(texture_id)
            except Exception as reload_exc:
                self._log(f"Texture Paint rollback preview reload failed: {reload_exc}")
            message = f"Texture-paint stroke was rolled back: {exc}"
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 7000)
            self._log(message)
            return
        message = (
            f"Painted {asset.resref}: {result.pixels_changed} texel(s), "
            f"{len(result.dirty_tiles)} dirty tile(s), one undoable stroke. "
            "Apply Textures before export."
        )
        if self._texture_paint_preview_error:
            message = f"{message} Renderer warning: {self._texture_paint_preview_error}."
        self.texture_paint_tab.set_unapplied_changes(True, asset.resref)
        self.texture_paint_tab.set_status(message)
        self.statusBar().showMessage(message, 6000)
        self.setWindowTitle(f"Ghost-Studio Map Studio - Level Editor - {self.project.name} *")
        self._update_map_studio_undo_redo_actions()

    def _apply_map_studio_texture_changes(self) -> None:
        """Accept the live-painted sidecars and refresh the export/readiness gate."""

        session = self._texture_paint_session
        if session is not None and bool(getattr(session, "stroke_active", False)):
            message = "Finish or cancel the active paint stroke before applying room textures."
            self.texture_paint_tab.set_status(message)
            return
        try:
            result = self.controller.apply_project_texture_changes()
        except Exception as exc:
            message = f"Room texture changes were not applied: {exc}"
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 8000)
            self._log(message)
            QtWidgets.QMessageBox.warning(self, "Apply Textures", message)
            return
        message = str(result.get("message") or "Texture changes applied.")
        self._refresh_all(message)
        self.texture_paint_tab.set_status(message)
        self.statusBar().showMessage(message, 7000)

    def _cancel_map_studio_texture_paint_stroke(self) -> None:
        session = self._texture_paint_session
        self._texture_paint_accepting_stroke = False
        if session is None or not session.stroke_active:
            return
        self._texture_paint_upload_timer.stop()
        dirty = session.active_dirty_tiles()
        session.cancel_stroke()
        self._apply_map_studio_texture_paint_tiles(session.dirty_tile_payloads(dirty))
        self.texture_paint_tab.set_status("Texture-paint stroke cancelled.")

    def import_module(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Import module folder", "")
        if not path:
            return
        module = self.controller.add_module(Path(path).name, source_path=path)
        self._refresh_all(f"Imported module reference {module.module_name}.")

    def _convert_all_stock_rooms(self) -> None:
        """Bake every stock room to editable geometry so export covers the whole map."""

        ok, message = self.controller.convert_all_stock_rooms_to_imported_mesh(
            resource_manager=getattr(self, "resource_manager", None),
        )
        self.statusBar().showMessage(message, 8000)
        self._log(f"Map Studio: {message}")
        if not ok:
            # A silent status-bar failure here cost a whole manual test run;
            # conversion problems must be impossible to miss.
            QtWidgets.QMessageBox.warning(self, "Make All Stock Rooms Editable", message)
        self._refresh_all(message)

    def _import_stock_module(self) -> None:
        """Import a complete stock KOTOR module (RIM) into an editable project.

        Reads ARE/GIT/IFO/LYT/VIS + room MDL geometry from the module RIM and
        populates the authored module project with all placements, rooms, lights,
        and metadata — everything needed to edit and re-export the module.
        """

        # Step 1: modules directory
        modules_dir = str(getattr(self, "_last_game_modules_dir", "") or "").strip()
        if not modules_dir:
            modules_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select KOTOR Modules folder (contains .rim / .mod files)", "")
            if not modules_dir:
                return
            self._last_game_modules_dir = modules_dir

        # Step 2: module resref
        resref, ok = QtWidgets.QInputDialog.getText(
            self, "Import Stock Module", "Module resref (e.g. plcaa, 001ebo1, tad_m12aa):")
        if not ok or not resref.strip():
            return
        resref = resref.strip().lower()

        # Step 3: game tag
        game = str(getattr(self.project, "game", "") or "").upper()
        if game not in ("K1", "K2"):
            items = ("K1", "K2")
            choice, ok2 = QtWidgets.QInputDialog.getItem(
                self, "Select Game", "Target game:", items, 0, False)
            if not ok2:
                return
            game = str(choice).upper()

        # Step 4: call the controller
        ok3, message = self.controller.import_stock_module_from_rim(
            module_resref=resref,
            modules_dir=modules_dir,
            game=game,
            resource_manager=self.resource_manager,
        )

        if ok3:
            conversion_ok, conversion_message = self.controller.convert_all_stock_rooms_to_imported_mesh(
                resource_manager=self.resource_manager,
            )
            combined_message = f"{message} {conversion_message}"
            self._log(f"Map Studio: {conversion_message}")
            self.statusBar().showMessage(combined_message, 10000)
            if not conversion_ok:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Stock Room Conversion",
                    conversion_message,
                )
            self._refresh_map_studio_geometry_change(combined_message)
            # Switch to geometry workspace so the user sees the imported rooms.
            try:
                self.map_studio_workspace_combo.setCurrentIndex(
                    max(0, self.map_studio_workspace_combo.findData("geometry")))
            except Exception:
                pass
        else:
            QtWidgets.QMessageBox.warning(self, "Import Stock Module", message)

    def import_module_file(self) -> None:
        """Import a KOTOR module capsule (.mod / .rim / .erf) directly by file.

        The friendly path for custom community modules: pick the file, and the
        game (K1/K2) is auto-detected from a bundled room model.  Bundled room
        models/WOKs/textures resolve via the ResourceManager overlay, then every
        stock room is baked to editable geometry in one step.
        """

        start_dir = str(getattr(self, "_last_game_modules_dir", "") or "")
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import KOTOR module",
            start_dir,
            "KOTOR modules (*.mod *.rim *.erf);;All files (*)",
        )
        if not path:
            return
        capsule = Path(path)
        self._last_game_modules_dir = str(capsule.parent)
        resref = capsule.stem.lower()

        game = self._detect_module_game(capsule) or str(getattr(self.project, "game", "") or "").upper()
        if game not in ("K1", "K2"):
            choice, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Select Game",
                f"Could not auto-detect the game for {capsule.name}.\nTarget game:",
                ("K1", "K2"),
                0,
                False,
            )
            if not ok:
                return
            game = str(choice).upper()

        resource_manager = getattr(self, "resource_manager", None)
        ok, message = self.controller.import_stock_module_from_rim(
            module_resref=resref,
            modules_dir=str(capsule.parent),
            game=game,
            resource_manager=resource_manager,
        )
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Import Module", message)
            return
        self._log(f"Map Studio: {message}")
        # Bake stock rooms to editable geometry so they can be edited + exported.
        conv_ok, conv_message = self.controller.convert_all_stock_rooms_to_imported_mesh(
            resource_manager=resource_manager,
        )
        self._log(f"Map Studio: {conv_message}")
        self.statusBar().showMessage(f"{message}  {conv_message}", 10000)
        try:
            self.map_studio_workspace_combo.setCurrentIndex(
                max(0, self.map_studio_workspace_combo.findData("geometry")))
        except Exception:
            pass
        self._refresh_all(f"Imported module {resref} ({game}).")

    def _add_room_from_module(self) -> None:
        """Browse rooms indexed from a chosen module/kmap and add one to this map.

        Foundation of the modular map builder: pick a source .mod/.rim/.kmap,
        pick a room from its labeled, door-hook-aware catalog, and drop it into
        the current project as editable geometry (ready to snap into place).
        """

        from src.core.modules.map_studio_room_catalog import (
            build_room_catalog_from_capsule,
            build_room_catalog_from_kmap,
        )

        start_dir = str(getattr(self, "_last_game_modules_dir", "") or "")
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a source module or KMAP",
            start_dir,
            "Room sources (*.mod *.rim *.erf *.kmap);;All files (*)",
        )
        if not path:
            return
        source = Path(path)
        self._last_game_modules_dir = str(source.parent)
        if source.suffix.lower() == ".kmap":
            result = build_room_catalog_from_kmap(source)
        else:
            game = self._detect_module_game(source) or str(getattr(self.project, "game", "") or "").upper()
            result = build_room_catalog_from_capsule(source, game=game)
        entries = result.sorted_entries()
        if not entries:
            reason = result.warnings[0] if result.warnings else "no rooms were indexed."
            QtWidgets.QMessageBox.warning(self, "Add Room from Module", f"Could not index rooms from {source.name}: {reason}")
            return
        labels = [f"{e.label} [{e.connection_count} hook(s)]" for e in entries]
        choice, accepted = QtWidgets.QInputDialog.getItem(
            self, "Add Room from Module", f"Room from {source.name}:", labels, 0, False
        )
        if not accepted:
            return
        entry = entries[labels.index(str(choice))]
        resource_manager = getattr(self, "resource_manager", None)
        if resource_manager is None:
            QtWidgets.QMessageBox.warning(
                self, "Add Room from Module",
                "Connect a KOTOR game directory first so the room's model and walkmesh can be loaded.")
            return
        try:
            ok, message = self.controller.add_catalog_room_to_project(
                room_resref=entry.room_resref,
                source_path=entry.source_path,
                source_module=entry.module_resref,
                game=entry.game or str(getattr(self.project, "game", "") or "").upper(),
                resource_manager=resource_manager,
                connection_points=entry.connection_points,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Room from Module", f"Could not add room {entry.room_resref}: {exc}")
            return
        if not ok:
            QtWidgets.QMessageBox.information(self, "Add Room from Module", message)
            return
        self._log(f"Map Studio: {message}")
        self.statusBar().showMessage(message, 8000)
        try:
            self.map_studio_workspace_combo.setCurrentIndex(
                max(0, self.map_studio_workspace_combo.findData("geometry")))
        except Exception:
            pass
        self._refresh_all(f"Added room {entry.room_resref} from {entry.module_resref}.")

    def _snap_rooms_at_doorway(self) -> None:
        """Move one room so a chosen doorway lines up with a doorway on another."""

        choices = [row for row in self.controller.authored_room_doorway_choices() if int(row.get("hook_count", 0) or 0) > 0]
        if len(choices) < 2:
            QtWidgets.QMessageBox.information(
                self, "Snap Rooms at Doorway",
                "Add at least two rooms that carry doorway hooks (rooms added via 'Add Room from Module' "
                "record their door hooks) before snapping.")
            return

        def _pick_room(title: str) -> dict | None:
            labels = [f"{row['room_resref']} ({row['hook_count']} door hook(s))" for row in choices]
            choice, accepted = QtWidgets.QInputDialog.getItem(self, "Snap Rooms at Doorway", title, labels, 0, False)
            return choices[labels.index(str(choice))] if accepted else None

        def _pick_door(row: dict, title: str) -> str | None:
            doors = list(row.get("doors", ()) or ())
            if not doors:
                return None
            if len(doors) == 1:
                return doors[0]
            choice, accepted = QtWidgets.QInputDialog.getItem(self, "Snap Rooms at Doorway", title, doors, 0, False)
            return str(choice) if accepted else None

        source_row = _pick_room("Room to MOVE:")
        if source_row is None:
            return
        source_door = _pick_door(source_row, f"{source_row['room_resref']} doorway to align:")
        if source_door is None:
            return
        remaining = [row for row in choices if row["room_resref"] != source_row["room_resref"]]
        target_labels = [f"{row['room_resref']} ({row['hook_count']} door hook(s))" for row in remaining]
        target_choice, accepted = QtWidgets.QInputDialog.getItem(
            self, "Snap Rooms at Doorway", "Room to snap ONTO (stays put):", target_labels, 0, False
        )
        if not accepted:
            return
        target_row = remaining[target_labels.index(str(target_choice))]
        target_door = _pick_door(target_row, f"{target_row['room_resref']} doorway to meet:")
        if target_door is None:
            return
        ok, message = self.controller.snap_authored_rooms_at_doorway(
            source_room_resref=source_row["room_resref"],
            source_door=source_door,
            target_room_resref=target_row["room_resref"],
            target_door=target_door,
        )
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Snap Rooms at Doorway", message)
            return
        self._log(f"Map Studio: {message}")
        self.statusBar().showMessage(message, 8000)
        self._refresh_all(message)

    def _detect_module_game(self, capsule: Path) -> str:
        """Return 'K1'/'K2' from a bundled or loose room MDL pointer."""

        try:
            import struct as _struct

            from pykotor.extract.capsule import LazyCapsule
            from pykotor.resource.type import ResourceType as _RT

            def _game_from_header(data: bytes) -> str:
                if len(data) < 16:
                    return ""
                fp1 = _struct.unpack_from("<I", data, 12)[0]
                if fp1 == 4285200:
                    return "K2"
                if fp1 == 4273776:
                    return "K1"
                return ""

            for item in LazyCapsule(str(capsule)):
                if item.restype() != _RT.MDL:
                    continue
                data = bytes(item.data() or b"")
                detected = _game_from_header(data)
                if detected:
                    return detected

            # Metadata-only MODs from older toolchains keep their models in a
            # sibling Override directory.  Probe only the nearest recovered
            # bundle, reading 16-byte headers rather than loading each model.
            bundle_root = capsule.parent
            if bundle_root.name.strip().lower() in {"module", "modules"}:
                bundle_root = bundle_root.parent
            for ancestor in capsule.parents:
                if ancestor.parent.name.strip().lower() == "extracted":
                    bundle_root = ancestor
                    break
            if not (bundle_root / "chitin.key").is_file():
                for model_path in sorted(bundle_root.rglob("*.mdl"), key=lambda item: str(item).lower()):
                    try:
                        with model_path.open("rb") as stream:
                            detected = _game_from_header(stream.read(16))
                    except OSError:
                        continue
                    if detected:
                        return detected
        except Exception:
            return ""
        return ""

    def set_library_rows(self, rows: list[dict[str, Any]]) -> None:
        self._base_library_rows = [dict(row) for row in rows]
        self.refresh_placeable_library()

    def _apply_combined_library_rows(self) -> None:
        combined: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in (*self._base_library_rows, *self._placeable_library_rows, *self._plcaa_manual_proof_rows):
            value = dict(row)
            key = (
                str(value.get("game") or "").upper(),
                str(value.get("resref") or value.get("template_resref") or "").lower(),
                str(value.get("restype") or value.get("resource_type") or value.get("type") or "").lower().lstrip("."),
            )
            combined[key] = value
        self._library_rows = list(combined.values())
        self.asset_browser.set_rows(self._library_rows)
        palette = self.controller.authored_gameplay_palette_entries(self._library_rows)
        self.builder_tab.set_gameplay_palette_entries(palette)
        self.placement_tab.set_palette_entries(palette)

    def set_placeable_library_root(self, root: str | Path, *, provider: Any = None) -> None:
        """Attach the reusable Placeable Builder library to this Map Studio."""

        self._placeable_library_root = str(Path(root).resolve()) if str(root or "").strip() else ""
        self._placeable_game_resource_provider = provider
        self.refresh_placeable_library()

    def _placeable_provider(self) -> Any:
        if self._placeable_game_resource_provider is not None:
            return self._placeable_game_resource_provider
        if self.resource_manager is None:
            return None
        try:
            from src.core.resources.game_resource_provider import ResourceManagerGameResourceProvider

            return ResourceManagerGameResourceProvider(self.resource_manager)
        except Exception:
            return None

    def _authored_module_root(self) -> str:
        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        return str(payload.get("module_root") or getattr(self.project, "name", "") or "").strip().lower()

    def _plcaa_proof_build(self, base_provider: Any) -> Any:
        """Build/cache target-game plcaa proof resources without replacing the base library."""

        if self._authored_module_root() != "plcaa":
            self._plcaa_manual_proof_build = None
            self._plcaa_manual_proof_cache_key = None
            self._plcaa_manual_proof_rows = []
            return None
        game = str(getattr(self.project, "game", "") or "K1").strip().upper()
        provider_owner = self._placeable_game_resource_provider or self.resource_manager or base_provider
        cache_key = (game, id(provider_owner))
        if self._plcaa_manual_proof_build is not None and self._plcaa_manual_proof_cache_key == cache_key:
            return self._plcaa_manual_proof_build
        from src.core.workflow.plcaa_manual_proof_kit import (
            build_plcaa_manual_proof_kit_from_provider,
            plcaa_manual_proof_palette_rows,
        )

        build = build_plcaa_manual_proof_kit_from_provider(game, base_provider)
        self._plcaa_manual_proof_build = build
        self._plcaa_manual_proof_cache_key = cache_key
        self._plcaa_manual_proof_rows = [dict(row) for row in plcaa_manual_proof_palette_rows(game)]
        for issue in tuple(getattr(build, "issues", ()) or ()):
            severity = str(getattr(issue, "severity", "warning") or "warning").title()
            self._log(f"PLCaa proof kit {severity}: {getattr(issue, 'message', issue)}")
        return build

    def _map_studio_gameplay_provider(self) -> Any:
        """Return base content plus generated plcaa proof candidates when applicable."""

        base_provider = self._placeable_provider()
        build = self._plcaa_proof_build(base_provider)
        if build is None or not bool(getattr(build, "ok", False)):
            return base_provider
        from src.core.resources.game_resource_provider import CompositeGameResourceProvider
        from src.core.workflow.plcaa_manual_proof_kit import plcaa_manual_proof_in_memory_provider

        generated = plcaa_manual_proof_in_memory_provider(build)
        return CompositeGameResourceProvider(
            provider for provider in (generated, base_provider) if provider is not None
        )

    def refresh_placeable_library(self) -> tuple[dict[str, Any], ...]:
        """Refresh target-game UTP templates and authored placeable assets."""

        rows: tuple[dict[str, Any], ...] = ()
        provider = self._map_studio_gameplay_provider()
        if self._placeable_library_root or provider is not None:
            try:
                from src.core.workflow.placeable_builder_service import placeable_library_rows

                # The startup Content Browser is model-oriented; it is not a
                # trustworthy GIT template browser. Discover real UTP records
                # from the selected game provider and merge project-authored
                # Placeable Builder assets over them by typed identity.
                rows = placeable_library_rows(
                    self._placeable_library_root,
                    game=str(getattr(self.project, "game", "") or ""),
                    provider=provider,
                )
            except Exception as exc:
                self._log(f"Placeable Library refresh warning: {exc}")
        self._placeable_library_rows = [dict(row) for row in rows]
        self._placeable_library_game = str(getattr(self.project, "game", "") or "").upper()
        self._placeable_library_module_root = self._authored_module_root()
        proof_build = self._plcaa_manual_proof_build
        if proof_build is not None and bool(getattr(proof_build, "ok", False)):
            self.controller.set_authored_placeable_resources(tuple(getattr(proof_build, "resources", ()) or ()))
        elif self._placeable_library_module_root != "plcaa":
            self.controller.set_authored_placeable_resources(())
        self.controller.set_authored_placeable_preview_rows(self._placeable_library_rows)
        self._apply_combined_library_rows()
        return tuple(self._placeable_library_rows)

    def set_scripting_studio_resources(self, resources: object) -> None:
        """Stage a successful Scripting Studio build for the next Map Studio export.

        This boundary accepts immutable ``(resref, restype, bytes)`` rows (or a
        build result exposing ``resource_tuples``).  Map Studio never reaches
        into the workbench's mutable documents; only validated runtime types
        such as NCS, DLG, JRL, 2DA, LIP, SSF, and GFF blueprints are accepted.
        """

        rows: object = resources
        resource_tuples = getattr(resources, "resource_tuples", None)
        if callable(resource_tuples):
            rows = resource_tuples(runtime_only=True)
        elif not isinstance(resources, (list, tuple)) and hasattr(resources, "resources"):
            rows = tuple(
                (
                    str(getattr(item, "resref", "") or ""),
                    str(getattr(item, "restype", "") or ""),
                    bytes(getattr(item, "data", b"") or b""),
                )
                for item in tuple(getattr(resources, "resources", ()) or ())
            )
        normalized: list[tuple[str, str, bytes]] = []
        for item in tuple(rows or ()):
            try:
                resref, restype, data = item
            except (TypeError, ValueError):
                continue
            clean_ref = str(resref or "").strip().lower()[:16]
            clean_type = str(restype or "").strip().lower().lstrip(".")
            payload = bytes(data or b"")
            if clean_ref and clean_type and payload:
                normalized.append((clean_ref, clean_type, payload))
        self.controller.set_authored_scripting_resources(normalized)
        self._scripting_studio_resources = tuple(normalized)
        self._log(
            f"Staged {len(normalized)} validated Scripting Suite runtime resource(s) "
            "for the next Map Studio package."
        )

    def _authored_placeable_template_resrefs(self) -> tuple[str, ...]:
        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        placements = dict(payload.get("placements") or {})
        result: list[str] = []
        for item in tuple(placements.get("placeables") or ()):
            value = dict(item or {}) if isinstance(item, dict) else {}
            resref = str(value.get("template_resref") or value.get("resref") or "").strip().lower()
            if resref and resref not in result:
                result.append(resref)
        return tuple(result)

    def _authored_interactive_template_requests(self) -> tuple[tuple[str, str], ...]:
        """Return typed templates manually placed into the authored GIT."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        placements = dict(payload.get("placements") or {})
        result: list[tuple[str, str]] = []
        for field_name, restype in (("placeables", "UTP"), ("doors", "UTD")):
            for item in tuple(placements.get(field_name) or ()):
                value = dict(item or {}) if isinstance(item, dict) else {}
                resref = str(value.get("template_resref") or value.get("resref") or "").strip().lower()
                key = (resref, restype)
                if resref and key not in result:
                    result.append(key)
        return tuple(result)

    def _sync_placeable_library_resources_for_export(self) -> None:
        """Resolve non-core placed UTP/UTD graphs into the final MOD."""

        self.refresh_placeable_library()
        provider = self._map_studio_gameplay_provider()
        proof_build = self._plcaa_manual_proof_build
        selected = self._authored_interactive_template_requests()
        resources: list[tuple[str, str, bytes]] = list(tuple(getattr(proof_build, "resources", ()) or ()))
        issues: list[Any] = list(tuple(getattr(proof_build, "issues", ()) or ()))
        report = None
        if selected:
            from src.core.workflow.placeable_builder_service import referenced_interactive_resource_report

            report = referenced_interactive_resource_report(
                self._placeable_library_root,
                selected,
                game=str(getattr(self.project, "game", "") or ""),
                provider=provider,
            )
            resources.extend(tuple(report.resources or ()))
            issues.extend(tuple(report.issues or ()))

        merged: dict[tuple[str, str], bytes] = {}
        for resref, restype, data in resources:
            key = (str(resref or "").strip().lower(), str(restype or "").strip().lower().lstrip("."))
            payload = bytes(data or b"")
            if not key[0] or not key[1] or not payload:
                continue
            prior = merged.get(key)
            if prior is not None and prior != payload:
                raise ValueError(f"PLCaa/placeable export collision for {key[0]}.{key[1]}.")
            merged[key] = payload
        merged_resources = tuple((resref, restype, data) for (resref, restype), data in sorted(merged.items()))
        self.controller.set_authored_placeable_resources(merged_resources, issues=issues)
        for issue in issues:
            severity = str(getattr(issue, "severity", "warning") or "warning").title()
            self._log(f"Placeable Library {severity}: {getattr(issue, 'message', issue)}")
        blocking = [
            str(getattr(issue, "message", issue))
            for issue in issues
            if str(getattr(issue, "severity", "") or "").strip().lower() in {"blocking", "error"}
        ]
        if blocking or (report is not None and bool(getattr(report, "has_blocking", False))):
            messages = blocking or ["One or more referenced resources could not be resolved."]
            raise ValueError("Placeable Library export is blocked: " + " ".join(messages))

    def _sync_authored_creature_behavior_resources_for_export(self) -> None:
        """Resolve donor UTCs and compile target-game creature behavior resources."""

        all_creatures = [
            row
            for row in self.controller.authored_gameplay_placements()
            if str(getattr(row, "kind", "") or "").strip().lower() == "creature"
        ]
        authored_snapshot = self.controller._map_studio_authored_project_snapshot()
        instance_provenance = dict(
            getattr(getattr(authored_snapshot, "placements", None), "metadata", {}).get(
                "instance_provenance", {}
            )
            if authored_snapshot is not None
            else {}
        )
        for creature in all_creatures:
            provenance = dict(instance_provenance.get(str(creature.placement_id), {}) or {})
            asset_path = Path(str(provenance.get("asset_path") or ""))
            if asset_path.suffix.lower() != ".utc" or not asset_path.is_file():
                continue
            resref = str(creature.template_resref or asset_path.stem).strip().lower()
            try:
                self._custom_creature_resources[resref] = (
                    asset_path.read_bytes(),
                    str(asset_path.resolve()),
                )
            except OSError as exc:
                raise ValueError(
                    f"Custom creature UTC {resref} could not be reread from {asset_path}: {exc}"
                ) from exc
        rows = [
            row
            for row in all_creatures
            if str(getattr(row, "creature_behavior_role", "template") or "template").strip().lower() != "template"
        ]
        merged: dict[tuple[str, str], bytes] = {
            (resref, "utc"): bytes(payload)
            for resref, (payload, _path) in self._custom_creature_resources.items()
            if payload
        }
        if not rows:
            resources = tuple((resref, restype, data) for (resref, restype), data in sorted(merged.items()))
            self.controller.set_authored_creature_resources(resources)
            return
        provider = self._map_studio_gameplay_provider()
        reader = getattr(provider, "read_resource", None)
        if provider is None or not callable(reader):
            self.controller.set_authored_creature_resources(())
            raise ValueError(
                "Creature behavior export requires the target KOTOR installation/resource provider to resolve source UTC templates."
            )
        from src.core.modules.authored_creature_behavior import build_authored_creature_behavior_resources

        game = str(getattr(self.project, "game", "") or "K1").strip().upper()
        def add_resource(resref: str, restype: str, data: bytes) -> None:
            key = (str(resref or "").strip().lower(), str(restype or "").strip().lower().lstrip("."))
            payload = bytes(data or b"")
            if not key[0] or not key[1] or not payload:
                return
            prior = merged.get(key)
            if prior is not None and prior != payload:
                raise ValueError(f"Creature behavior resource collision for {key[0]}.{key[1]}.")
            merged[key] = payload

        for row in rows:
            source_template = str(getattr(row, "creature_source_template_resref", "") or "").strip().lower()
            generated_template = str(getattr(row, "creature_generated_template_resref", "") or "").strip().lower()
            if not source_template or not generated_template:
                raise ValueError(f"Creature {getattr(row, 'tag', row.placement_id)} has incomplete behavior metadata.")
            try:
                source_utc = bytes(reader(source_template, "UTC", game=game))
            except Exception as exc:
                self.controller.set_authored_creature_resources(())
                raise ValueError(
                    f"Could not resolve target-game source UTC {source_template}.utc for creature {getattr(row, 'tag', row.placement_id)}: {exc}"
                ) from exc
            build = build_authored_creature_behavior_resources(
                source_utc,
                game=game,
                template_resref=generated_template,
                instance_tag=str(getattr(row, "tag", "") or generated_template),
                faction_role=str(getattr(row, "creature_behavior_role", "neutral") or "neutral"),
                conversation_resref=str(getattr(row, "creature_conversation_resref", "") or ""),
                movement_mode=str(getattr(row, "creature_movement_mode", "stationary") or "stationary"),
            )
            for resource in build.resources:
                add_resource(*resource)
            conversation = str(getattr(row, "creature_conversation_resref", "") or "").strip().lower()
            if conversation:
                staged_reader = getattr(self.controller, "authored_scripting_resource", None)
                staged_dialogue = staged_reader(conversation, "dlg") if callable(staged_reader) else None
                if staged_dialogue is not None:
                    # It is already part of authored_project_extra_resources;
                    # do not duplicate it in the creature-owned resource set.
                    self._log(
                        f"Creature conversation {conversation}.dlg resolves from the current Scripting Studio build."
                    )
                else:
                    try:
                        add_resource(conversation, "dlg", bytes(reader(conversation, "DLG", game=game)))
                    except Exception as exc:
                        self.controller.set_authored_creature_resources(())
                        raise ValueError(
                            f"Creature conversation {conversation}.dlg could not be resolved for target game {game}: {exc}"
                        ) from exc
        resources = tuple((resref, restype, data) for (resref, restype), data in sorted(merged.items()))
        self.controller.set_authored_creature_resources(resources)
        self._log(f"Creature behavior export resolved {len(rows)} authored creature(s) into {len(resources)} UTC/script/dialog resource(s).")

    def _import_custom_creature_utc(self) -> None:
        """Add a real local UTC to the placement palette and export resource set."""

        path_text, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Custom Creature UTC",
            "",
            "KOTOR Creature Templates (*.utc);;All Files (*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            payload = path.read_bytes()
            from pykotor.resource.generics.utc import read_utc

            read_utc(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Import Custom Creature UTC",
                f"{path.name} is not a readable KOTOR UTC template:\n{exc}",
            )
            return
        from src.core.modules.authored_module_objects import normalise_resource_resref

        resref = normalise_resource_resref(path.stem)
        if not resref:
            QtWidgets.QMessageBox.warning(
                self,
                "Import Custom Creature UTC",
                "The UTC filename must contain a valid KOTOR resource name.",
            )
            return
        game = str(getattr(self.project, "game", "") or "K1").strip().upper()
        self._custom_creature_resources[resref] = (payload, str(path.resolve()))
        existing = [
            row
            for row in self._base_library_rows
            if not (
                str(row.get("resref") or "").strip().lower() == resref
                and str(row.get("restype") or "").strip().lower().lstrip(".") == "utc"
            )
        ]
        existing.append(
            {
                "game": game,
                "resref": resref,
                "template_resref": resref,
                "restype": "utc",
                "category": "Creatures",
                "source": "Custom UTC import",
                "path": str(path.resolve()),
            }
        )
        self._base_library_rows = existing
        self.controller.set_authored_creature_resources(
            tuple(
                (name, "utc", bytes(data))
                for name, (data, _source_path) in sorted(self._custom_creature_resources.items())
            )
        )
        self._apply_combined_library_rows()
        creature_index = self.placement_tab.kind_combo.findData("creature")
        if creature_index >= 0:
            self.placement_tab.kind_combo.setCurrentIndex(creature_index)
        self.placement_tab.search_edit.setText(resref)
        self.statusBar().showMessage(
            f"Imported verified custom UTC {resref}; drag it onto authored room or terrain geometry.",
            7000,
        )

    def set_renderer_settings(self, settings: RendererSettings | dict | None) -> None:
        renderer_settings = settings if isinstance(settings, RendererSettings) else RendererSettings.from_settings(settings or {})
        viewport_panel = getattr(self, "viewport_panel", None)
        if viewport_panel is not None and hasattr(viewport_panel, "set_renderer_settings"):
            viewport_panel.set_renderer_settings(renderer_settings)

    def import_selected_library_asset(self) -> None:
        row = self.asset_browser.selected_row()
        if not row:
            self.left_tabs.setCurrentWidget(self.asset_browser)
            self._log("Select an asset in the Assets tab first.")
            return
        self.import_library_asset(row)

    def import_library_asset(self, row: dict[str, Any]) -> None:
        try:
            item = self.controller.import_library_asset(row, resource_manager=self.resource_manager)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Library Asset", str(exc))
            return
        item_name = getattr(item, "name", getattr(item, "module_name", str(row.get("resref", "asset"))))
        self.left_tabs.setCurrentWidget(self.outliner)
        self._refresh_all(f"Imported library asset {item_name}.")

    def delete_selected(self) -> None:
        if self.controller.remove_selected():
            self._refresh_all("Deleted selected item.")

    def duplicate_selected(self) -> None:
        if self.controller.duplicate_selected() is not None:
            self._refresh_all("Duplicated selected item.")

    def select_map_studio_authored_context(self) -> bool:
        """Persist the current Builder-authored selection as lightweight KMAP state."""

        context = self._map_studio_tool_action_context("select")
        primitive_name = str(getattr(context, "primitive_name", "") or "").strip()
        room_resref = str(getattr(context, "room_resref", "") or "").strip()
        component_mode = "object"
        combo = getattr(self.builder_tab, "componentModeComboBox", None)
        if combo is not None:
            data = combo.currentData()
            if isinstance(data, dict):
                component_mode = str(data.get("key") or component_mode)
        try:
            selection = self.controller.set_map_studio_active_selection(
                component_mode=component_mode,
                workspace_key=str(self.map_studio_workspace_combo.currentData() or "geometry"),
                tool_key="select",
                room_resref=room_resref,
                primitive_name=primitive_name,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Select Map Studio Target", str(exc))
            return False
        target = str(selection.get("primitive_name") or selection.get("room_resref") or "Map Studio")
        self.show_map_studio_builder()
        self.statusBar().showMessage(f"Selected Map Studio authored target {target}.", 5000)
        self._log(f"Map Studio authored selection stored in KMAP: {target}.")
        return True

    def move_map_studio_authored_primitive_selection(self) -> bool:
        """Move the Builder-selected authored primitive using the visible Move X/Y/Z fields."""

        data = self._map_studio_combo_data("roomPrimitiveTransformComboBox")
        primitive_name = str(data.get("primitive_name") or "").strip()
        if not primitive_name:
            return False
        room_resref = str(data.get("room_resref") or "").strip()
        before_translation = tuple(float(value) for value in tuple(data.get("translation") or (0.0, 0.0, 0.0))[:3])
        if len(before_translation) != 3:
            before_translation = (0.0, 0.0, 0.0)
        spin_x = getattr(self.builder_tab, "primitiveTranslateXSpinBox", None)
        spin_y = getattr(self.builder_tab, "primitiveTranslateYSpinBox", None)
        spin_z = getattr(self.builder_tab, "primitiveTranslateZSpinBox", None)
        if spin_x is None or spin_y is None or spin_z is None:
            return False
        after_translation = (float(spin_x.value()), float(spin_y.value()), float(spin_z.value()))
        world_delta = tuple(after_translation[index] - before_translation[index] for index in range(3))
        if all(abs(delta) < 1e-9 for delta in world_delta):
            self.show_map_studio_builder()
            spin_x.setFocus()
            self.statusBar().showMessage("Map Studio Move ready: edit Move X/Y/Z, then click Move again to author the KMAP transform.", 5000)
            return False
        try:
            result = self.controller.move_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
                world_delta=world_delta,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Move Map Studio Primitive", str(exc))
            return False
        readiness = result.readiness
        message = f"Moved room primitive {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self.controller.model.select("")
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=((room_resref, primitive_name),),
            refresh_outlines=True,
        )
        return True

    def delete_map_studio_authored_primitive_selection(self) -> bool:
        """Delete the Builder-selected authored primitive through the KMAP controller."""

        context = self._map_studio_tool_action_context("delete_selected")
        primitive_name = str(getattr(context, "primitive_name", "") or "").strip()
        if not primitive_name:
            return False
        room_resref = str(getattr(context, "room_resref", "") or "").strip()
        try:
            result = self.controller.remove_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Map Studio Primitive", str(exc))
            return False
        readiness = result.readiness
        message = f"Deleted room primitive {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True)
        return True

    def _run_map_studio_tool_belt_key(self, key: str) -> None:
        """Execute a tool-belt action by its stable key (Edit menu entries)."""

        action = self._map_studio_tool_action_for_key(key)
        if action is None:
            self.statusBar().showMessage(f"Map Studio tool {key} is not available.", 4000)
            return
        self._handle_map_studio_tool_belt_action(action)

    def _handle_map_studio_room_clicked(self, room_resref: str, additive: bool, subtractive: bool = False) -> None:
        """Viewport object selection: Shift adds and Alt removes, like Maya."""

        clicked_id = str(room_resref or "").strip()
        entry_point = self.controller.authored_module_entry_point_preview_row()
        if clicked_id.lower() == "entry_point" and entry_point is not None:
            selected_ids = list(self.controller.model.selected_ids or ())
            if subtractive:
                selected_ids = [value for value in selected_ids if value != "entry_point"]
                self._select_map_studio_items(selected_ids)
            elif additive:
                if "entry_point" not in selected_ids:
                    selected_ids.append("entry_point")
                self._select_map_studio_items(selected_ids)
            else:
                self.select_item("entry_point")
            return
        known_placements = {
            str(getattr(row, "placement_id", "") or "").strip().lower():
            str(getattr(row, "placement_id", "") or "").strip()
            for row in (self.controller.authored_gameplay_placements() or ())
            if str(getattr(row, "placement_id", "") or "").strip()
        }
        placement_id = known_placements.get(clicked_id.lower())
        if placement_id:
            selected_ids = list(self.controller.model.selected_ids or ())
            if subtractive:
                selected_ids = [value for value in selected_ids if value != placement_id]
                self._select_map_studio_items(selected_ids)
            elif additive:
                if placement_id not in selected_ids:
                    selected_ids.append(placement_id)
                self._select_map_studio_items(selected_ids)
            else:
                self.select_item(placement_id)
            return
        resref = clicked_id.lower()
        authored_rooms = {
            str(value or "").strip().lower()
            for value in tuple(self.controller.authored_room_resrefs() or ())
            if str(value or "").strip()
        }
        if not resref:
            if not additive and not subtractive:
                self._select_map_studio_items(())
            return
        if resref in authored_rooms:
            item_id = f"authored_room:{resref}"
            selected_ids = list(self.controller.model.selected_ids or ())
            if subtractive:
                selected_ids = [value for value in selected_ids if str(value).lower() != item_id]
                self._select_map_studio_items(selected_ids)
            elif additive:
                if not any(str(value).lower() == item_id for value in selected_ids):
                    selected_ids.append(item_id)
                self._select_map_studio_items(selected_ids)
            else:
                self.select_item(item_id)
            return
        selected = list(getattr(self, "_map_studio_selected_rooms", []) or [])
        if subtractive:
            selected = [value for value in selected if value != resref]
        elif additive:
            if resref not in selected:
                selected.append(resref)
        else:
            selected = [resref]
        self._map_studio_selected_rooms = selected
        summary = ", ".join(selected) if selected else "(none)"
        self.statusBar().showMessage(f"Selected room(s): {summary} — press Delete to remove.", 5000)
        self._log(f"Map Studio room selection: {summary}")

    def _handle_map_studio_rooms_rect_selected(self, room_resrefs, additive: bool) -> None:
        """Drag-marquee selection over the viewport (plain or Ctrl LMB drag)."""

        incoming = [str(value or "").strip().lower() for value in tuple(room_resrefs or ()) if str(value or "").strip()]
        selected = list(getattr(self, "_map_studio_selected_rooms", []) or []) if additive else []
        for resref in incoming:
            if resref not in selected:
                selected.append(resref)
        authored_rooms = {
            str(value or "").strip().lower()
            for value in tuple(self.controller.authored_room_resrefs() or ())
            if str(value or "").strip()
        }
        scene_ids = [f"authored_room:{value}" if value in authored_rooms else value for value in selected]
        if scene_ids:
            if additive:
                existing = list(self.controller.model.selected_ids or ())
                scene_ids = existing + [value for value in scene_ids if value not in existing]
            self._select_map_studio_items(scene_ids)
        else:
            self._select_map_studio_items(())
        summary = ", ".join(selected) if selected else "(none)"
        self.statusBar().showMessage(
            f"Marquee selected {len(incoming)} object(s); selection: {summary} — press Delete to remove.", 6000
        )

    def _delete_map_studio_placement_ids(self, candidate_ids: list[str]) -> bool:
        """Remove authored gameplay placements matched by hover/selection ids.

        Stock preview instance groups carry their placement_id as the hover
        room_resref, so clicking a crate and pressing Delete lands here.
        """

        if "entry_point" in candidate_ids and self.controller.authored_module_entry_point_preview_row() is not None:
            try:
                self.controller.clear_authored_module_entry_point()
            except Exception as exc:
                self._log(f"Map Studio: player start could not be removed: {exc}")
                return False
            message = "Player Start removed. Module export is blocked until a new Player Start is set; Undo restores it."
            self.statusBar().showMessage(message, 8000)
            self._map_studio_selected_rooms = [
                value for value in getattr(self, "_map_studio_selected_rooms", []) if value != "entry_point"
            ]
            self.controller.model.select("")
            self._refresh_all(message)
            return True
        known = {
            str(getattr(row, "placement_id", "") or "")
            for row in (self.controller.authored_gameplay_placements() or ())
        }
        known.discard("")
        matched = [pid for pid in candidate_ids if pid in known]
        if not matched:
            return False
        try:
            # One undoable command for the whole marquee/multi delete.
            removed = list(self.controller.remove_authored_gameplay_placements(matched))
        except Exception as exc:
            self._log(f"Map Studio: placements could not be removed: {exc}")
            return False
        if not removed:
            return False
        message = f"Removed {len(removed)} placement(s): {', '.join(removed[:4])}{' ...' if len(removed) > 4 else ''}"
        self.statusBar().showMessage(message, 6000)
        self._map_studio_selected_rooms = [
            r for r in getattr(self, "_map_studio_selected_rooms", []) if r not in set(removed)
        ]
        self._refresh_all(message)
        return True

    def _delete_selected_map_studio_rooms(self) -> bool:
        selected = list(getattr(self, "_map_studio_selected_rooms", []) or [])
        if not selected:
            return False
        if self._delete_map_studio_placement_ids(selected):
            return True
        ok, message = self.controller.delete_map_studio_rooms(selected)
        self.statusBar().showMessage(message, 6000)
        self._log(f"Map Studio: {message}")
        if ok:
            self._map_studio_selected_rooms = []
            self._refresh_map_studio_geometry_change(
                message,
                refresh_outlines=True,
                refresh_terrain=True,
                refresh_room_choices=True,
                refresh_connections=True,
            )
        return True

    def _delete_selected_map_studio_primitives(self) -> bool:
        """Delete every selected terrain/environment/building object together."""

        selected = self._selected_authored_room_primitives()
        if not selected:
            return False
        try:
            removed = tuple(self.controller.remove_authored_room_primitives(selected))
        except Exception as exc:
            self._log(f"Map Studio: selected objects could not be removed: {exc}")
            return False
        if not removed:
            return False
        message = (
            f"Removed {len(removed)} selected object(s): "
            + ", ".join(name for _room, name in removed[:4])
            + (" ..." if len(removed) > 4 else "")
        )
        self.controller.model.select_many(())
        self.viewport_panel.set_selected_room_primitives(())
        self.outliner.select_ids(())
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )
        return True

    def _delete_map_studio_scene_object_selection(self) -> bool:
        """Delete selected kit pieces, placements, and Player Start together."""

        selected_ids = [
            str(value or "")
            for value in tuple(self.controller.model.selected_ids or ())
            if str(value or "")
        ]
        if not selected_ids:
            return False
        primitive_selections = tuple(
            identity
            for identity in (self._parse_map_studio_primitive_outliner_id(item_id) for item_id in selected_ids)
            if identity is not None
        )
        known_placements = {
            str(getattr(row, "placement_id", "") or "")
            for row in tuple(self.controller.authored_gameplay_placements() or ())
        }
        if self.controller.authored_module_entry_point_preview_row() is not None:
            known_placements.add("entry_point")
        placement_ids = tuple(item_id for item_id in selected_ids if item_id in known_placements)
        room_resrefs = tuple(
            item_id.split(":", 1)[1].strip().lower()
            for item_id in selected_ids
            if item_id.startswith("authored_room:") and item_id.split(":", 1)[1].strip()
        )
        if not primitive_selections and not placement_ids and not room_resrefs:
            return False
        try:
            removed_primitives, removed_placements = self.controller.remove_map_studio_scene_objects(
                primitive_selections=primitive_selections,
                placement_ids=placement_ids,
                room_resrefs=room_resrefs,
            )
        except Exception as exc:
            self._log(f"Map Studio: selected scene objects could not be removed: {exc}")
            return False
        removed_count = len(removed_primitives) + len(removed_placements)
        if not removed_count:
            return False
        self.controller.model.select_many(())
        self.outliner.select_ids(())
        self.viewport_panel.set_selected_room_primitives(())
        self.viewport_panel.set_selected_gameplay_placement_ids(())
        self.viewport_panel.set_map_studio_scene_selection_ids(())
        self._map_studio_selected_rooms = []
        message = f"Removed {removed_count} selected scene object(s). Undo restores the complete selection."
        if "entry_point" in removed_placements:
            message += " Module export is blocked until a Player Start is set."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )
        return True

    def delete_map_studio_current_selection(self) -> None:
        """Delete key: hovered component, then selected rooms, then selection.

        Hovering a face/edge/vertex (stock rooms auto-convert to editable
        geometry) deletes that component; otherwise the click/marquee room
        selection deletes as one undoable batch; then the selected authored
        primitive; then the selected scene object.
        """

        if self._delete_map_studio_component_selection():
            return
        if self._delete_map_studio_hovered_component():
            return
        if self._delete_map_studio_scene_object_selection():
            return
        # An explicit marquee/multi selection outranks the hovered object:
        # deleting five selected desks must not collapse to the one under
        # the cursor.
        if self._delete_selected_map_studio_primitives():
            return
        if self._delete_selected_map_studio_rooms():
            return
        selected_ids = [str(value or "") for value in tuple(self.controller.model.selected_ids or ()) if str(value or "")]
        if selected_ids and self._delete_map_studio_placement_ids(selected_ids):
            return
        context = self.viewport_panel.current_map_studio_hover_context()
        hovered_id = str(getattr(context, "room_resref", "") or "") if context is not None else ""
        if hovered_id and self._delete_map_studio_placement_ids([hovered_id]):
            return
        if self.delete_map_studio_authored_primitive_selection():
            return
        self.delete_selected()

    def _delete_map_studio_component_selection(self) -> bool:
        """Delete every yellow-selected component (Shift multi-select) at once."""

        panel = self.viewport_panel
        selection = panel.map_studio_component_selection()
        if not selection:
            return False
        faces: dict[tuple[str, str], list[int]] = {}
        others: list[dict] = []
        for entry in selection:
            if str(entry.get("component_type")) == "face" and int(entry.get("face_index", -1)) >= 0:
                faces.setdefault((str(entry.get("room_resref")), str(entry.get("mesh_role"))), []).append(int(entry["face_index"]))
            else:
                others.append(entry)
        handled = False
        for (room_resref, mesh_role), indices in faces.items():
            ok, message = self.controller.delete_imported_mesh_room_faces(
                room_resref=room_resref, mesh_role=mesh_role, face_indices=sorted(set(indices))
            )
            self.statusBar().showMessage(message, 6000)
            self._log(f"Map Studio: {message}")
            handled = handled or ok
        for entry in others:
            op = {"edge": "edge_delete", "vertex": "vertex_delete"}.get(str(entry.get("component_type")))
            if not op:
                continue
            kwargs = {
                "room_resref": str(entry.get("room_resref")),
                "op": op,
                "mesh_role": str(entry.get("mesh_role")),
                "face_index": int(entry.get("face_index", -1)),
            }
            if op == "vertex_delete":
                kwargs["vertex_corner"] = int(entry.get("vertex_index", 0))
            else:
                kwargs["edge_corners"] = tuple(entry.get("edge_indices") or (0, 1))
            ok, message = self.controller.apply_imported_mesh_room_component_op(**kwargs)
            self.statusBar().showMessage(message, 6000)
            handled = handled or ok
        if handled:
            panel.clear_map_studio_component_selection()
            self._refresh_map_studio_geometry_change("Deleted selected components.")
        # When nothing was deletable (e.g. selection on a non-imported room),
        # fall through so object/room/placement deletion still gets a chance.
        return handled

    def _delete_map_studio_hovered_component(self) -> bool:
        # Object mode deletes whole objects (rooms/placements), never a
        # single hovered face — that surprised the user badly.
        if str(getattr(self.viewport_panel, "_hover_component_mode", "")) == "object":
            return False
        context = self.viewport_panel.current_map_studio_hover_context()
        component = str(getattr(context, "component_type", "") or "") if context is not None else ""
        routes = {
            "face": ("face_delete", "Single Face"),
            "edge": ("edge_delete", "Single Edge"),
            "vertex": ("vertex_delete", "Single Vertex"),
        }
        route = routes.get(component)
        if route is None:
            return False
        return self._apply_component_modeling_imported_face_action(route[0], route[1])

    def _parse_map_studio_primitive_outliner_id(self, item_id: str) -> tuple[str, str] | None:
        parts = str(item_id or "").split(":", 2)
        if len(parts) != 3 or parts[0] != "authored_primitive":
            return None
        room_resref = parts[1].strip()
        primitive_name = parts[2].strip()
        if not room_resref or not primitive_name:
            return None
        return (room_resref, primitive_name)

    def _map_studio_primitive_outliner_id(self, room_resref: str, primitive_name: str) -> str:
        return authored_primitive_item_id(room_resref, primitive_name)

    def _safe_map_studio_primitive_name_for_ui(self, value: str) -> str:
        text = str(value or "").strip()
        safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
        return safe.strip("_-")[:32]

    def rename_map_studio_authored_primitive(self, room_resref: str, primitive_name: str, new_name: str) -> bool:
        updated_name = self._safe_map_studio_primitive_name_for_ui(new_name)
        if not updated_name:
            return False
        if updated_name == str(primitive_name or "").strip():
            self._select_authored_room_primitive(room_resref, primitive_name)
            return True
        try:
            self.controller.rename_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
                new_primitive_name=updated_name,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Map Studio Primitive", str(exc))
            return False
        message = f"Renamed room primitive {primitive_name} to {updated_name}; previous exports/proofs are now stale."
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=((room_resref, updated_name),),
            refresh_outlines=True,
        )
        return True

    def rename_selected(self) -> None:
        item_id = self.controller.model.selected_ids[0] if self.controller.model.selected_ids else ""
        if not item_id:
            return
        primitive_identity = self._parse_map_studio_primitive_outliner_id(item_id)
        if primitive_identity is not None:
            room_resref, primitive_name = primitive_identity
            self.placement_tab.set_selected_placement("")
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename Authored Primitive", "Name:", text=primitive_name)
            if ok and name.strip():
                self.rename_map_studio_authored_primitive(room_resref, primitive_name, name.strip())
            return
        if item_id.startswith("authored:"):
            authored = next((row for row in self.controller.authored_gameplay_placements() if getattr(row, "placement_id", "") == item_id), None)
            if authored is None:
                return
            current = str(getattr(authored, "tag", "") or getattr(authored, "template_resref", "") or item_id)
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename Authored Placement", "Name:", text=current)
            if ok and name.strip():
                try:
                    self.controller.rename_authored_gameplay_placement(item_id, tag=name.strip())
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Rename Authored Placement", str(exc))
                    return
                self._refresh_map_studio_gameplay_change(
                    "Renamed authored gameplay placement.",
                    placement_ids=(item_id,),
                    refresh_outliner_labels=True,
                )
            return
        if item_id.startswith("authored_light:"):
            light = next((row for row in self.controller.authored_room_lights() if getattr(row, "light_id", "") == item_id), None)
            if light is None:
                return
            current = str(getattr(light, "name", "") or item_id)
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename Authored Room Light", "Name:", text=current)
            if ok and name.strip():
                try:
                    self.controller.rename_authored_room_light(item_id, name=name.strip())
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Rename Authored Room Light", str(exc))
                    return
                self._refresh_map_studio_gameplay_change(
                    "Renamed authored room light.",
                    light_ids=(item_id,),
                    refresh_markers=False,
                    refresh_placement_rows=False,
                    refresh_outliner_labels=True,
                )
            return
        item = self.project.find_room(item_id) or self.project.find_module(item_id) or self.project.find_blueprint(item_id)
        if item is None:
            return
        current = getattr(item, "name", getattr(item, "module_name", ""))
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename Selected", "Name:", text=current)
        if ok and name.strip():
            self._set_property(item_id, "name", name.strip())

    def _rename_outliner_item_inline(self, item_id: str, new_name: str) -> None:
        primitive_identity = self._parse_map_studio_primitive_outliner_id(item_id)
        if primitive_identity is not None:
            room_resref, primitive_name = primitive_identity
            self.rename_map_studio_authored_primitive(room_resref, primitive_name, new_name)
            return
        self.select_item(item_id)
        if item_id.startswith("authored:"):
            try:
                self.controller.rename_authored_gameplay_placement(item_id, tag=str(new_name or "").strip())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Rename Authored Placement", str(exc))
                self._refresh_all()
                return
            self._refresh_map_studio_gameplay_change(
                "Renamed authored gameplay placement.",
                placement_ids=(item_id,),
                refresh_outliner_labels=True,
            )
            return
        if item_id.startswith("authored_light:"):
            try:
                self.controller.rename_authored_room_light(item_id, name=str(new_name or "").strip())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Rename Authored Room Light", str(exc))
                self._refresh_all()
                return
            self._refresh_map_studio_gameplay_change(
                "Renamed authored room light.",
                light_ids=(item_id,),
                refresh_markers=False,
                refresh_placement_rows=False,
                refresh_outliner_labels=True,
            )
            return
        item = self.project.find_room(item_id) or self.project.find_module(item_id) or self.project.find_blueprint(item_id)
        if item is not None and str(new_name or "").strip():
            self._set_property(item_id, "name", str(new_name or "").strip())

    def validate_kmap(self) -> None:
        issues = self.controller.validate()
        self.validation_panel.set_issues(issues)
        errors = sum(1 for issue in issues if issue.severity.lower() == "error")
        self._log(f"Validation complete: {len(issues)} issue(s), {errors} error(s).")
        self._show_map_studio_validation_output(self.validation_panel)

    def _show_map_studio_validation_output(self, page: QtWidgets.QWidget | None = None) -> None:
        """Open the normally-hidden diagnostics dock for an explicit user task."""

        if page is not None:
            self.bottom_tabs.setCurrentWidget(page)
        self._set_map_studio_validation_output_visible(True)

    def _set_map_studio_validation_output_visible(self, visible: bool) -> None:
        dock = self.validation_output_dock
        if bool(visible):
            dock.show()
            dock.raise_()
        else:
            dock.hide()

    def _sync_map_studio_validation_output_action(self, visible: bool) -> None:
        blocker = QtCore.QSignalBlocker(self.validation_action)
        self.validation_action.setChecked(bool(visible))
        del blocker

    def _selected_map_studio_workspace_key(self) -> str:
        return str(self.map_studio_workspace_combo.currentData() or "").strip()

    def _set_map_studio_workspace_combo_key(self, key: str) -> None:
        """Keep the visible workspace selector aligned with programmatic focus changes."""

        wanted = str(key or "").strip()
        index = self.map_studio_workspace_combo.findData(wanted)
        if index < 0:
            return
        if self.map_studio_workspace_combo.currentIndex() != index:
            previous = self.map_studio_workspace_combo.blockSignals(True)
            try:
                self.map_studio_workspace_combo.setCurrentIndex(index)
            finally:
                self.map_studio_workspace_combo.blockSignals(previous)
        self._update_map_studio_workspace_guide()

    def _set_map_studio_toolbar_edit_mode(self, label: str) -> None:
        """Keep the toolbar edit-mode selector aligned with explicit workspace changes."""

        combo = getattr(self.toolbar, "selection_mode", None)
        if combo is None:
            return
        wanted = str(label or "").strip()
        index = combo.findText(wanted)
        if index < 0 or combo.currentIndex() == index:
            self._sync_map_studio_edit_mode_context(wanted)
            return
        previous = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(previous)
        self._sync_map_studio_edit_mode_context(wanted)

    def _set_map_studio_skybox_visible(self, visible: bool) -> None:
        """Toggle non-selectable authored backdrop rooms in the live preview."""

        wanted = bool(visible)
        if wanted == bool(getattr(self, "_map_studio_show_skybox", False)):
            return
        self._map_studio_show_skybox = wanted
        preview = self.controller.map_studio_viewport_preview_model(
            resource_manager=getattr(self, "resource_manager", None),
            include_backdrops=wanted,
        )
        self.viewport_panel.set_authored_room_preview_model(preview)
        message = "Skybox/backdrop preview enabled." if wanted else "Skybox/backdrop preview hidden."
        self.statusBar().showMessage(message, 4000)
        self._log(message)

    def _sync_map_studio_edit_mode_context(self, label: str) -> None:
        """Refresh workflow-panel mode context from headless Map Studio mode policy."""

        context = self.controller.map_studio_edit_mode_context(label)
        self.workflow_panel.set_edit_mode_context(
            mode_label=str(getattr(context, "label", "") or label or "Object"),
            editing_target=str(getattr(context, "editing_target", "") or ""),
            kotor_guardrail=str(getattr(context, "kotor_guardrail", "") or ""),
            next_action=str(getattr(context, "next_action", "") or ""),
        )

    def _update_map_studio_workspace_guide(self) -> None:
        key = self._selected_map_studio_workspace_key()
        mode = self._map_studio_workspace_modes.get(key)
        if mode is None:
            self.map_studio_workspace_guide_label.setText(
                "Choose the Map Studio workspace for the current KOTOR level-authoring task."
            )
            return
        summary = str(getattr(mode, "summary", "") or "")
        next_action = str(getattr(mode, "next_action", "") or "")
        text = summary
        if next_action:
            text = f"{summary} Next: {next_action}" if summary else f"Next: {next_action}"
        self.map_studio_workspace_guide_label.setText(text)

    def _handle_map_studio_workspace_changed(self, _index: int) -> None:
        self._update_map_studio_workspace_guide()
        self._open_selected_map_studio_workspace(log_focus=False)

    def _open_selected_map_studio_workspace(self, *, log_focus: bool = True) -> None:
        key = self._selected_map_studio_workspace_key()
        if key == "geometry":
            self._set_map_studio_toolbar_edit_mode("Object")
            self.show_map_studio_geometry_tools()
        elif key == "terrain":
            self._set_map_studio_toolbar_edit_mode("Terrain")
            self.show_map_studio_terrain_tools()
        elif key == "walkmesh":
            self._set_map_studio_toolbar_edit_mode("Walkmesh")
            self.show_map_studio_walkmesh_tools()
        elif key == "placements":
            self._set_map_studio_toolbar_edit_mode("Placement")
            self.show_map_studio_placement_tools()
        elif key == "lighting":
            self._set_map_studio_toolbar_edit_mode("Object")
            self.show_map_studio_lighting_tools()
        elif key == "scripts":
            self._set_map_studio_toolbar_edit_mode("Object")
            self.show_map_studio_script_tools()
        elif key == "export":
            self._set_map_studio_toolbar_edit_mode("Export")
            self.right_tabs.setCurrentWidget(self.map_studio_export_page)
            self.workflow_panel.set_active_authoring_context(
                "Export + Game Proof: validate, stage/install, warp test, then record proof"
            )
            if log_focus:
                self._log("Map Studio export and game-proof workspace focused.")
        else:
            self._set_map_studio_toolbar_edit_mode("Object")
            self.left_tabs.setCurrentWidget(self.outliner)
            self.workflow_panel.set_active_authoring_context(
                "Project: KMAP identity, target game, outliner, asset browser, and save/open state"
            )
            if log_focus:
                self._log("Map Studio project workspace focused.")

    def _selected_map_studio_tool_belt_preset_key(self) -> str:
        return str(self.map_studio_tool_belt_preset_combo.currentData() or "blockout").strip() or "blockout"

    def _apply_map_studio_tool_belt_preferences_from_project(self) -> None:
        preferences = self.controller.map_studio_tool_belt_preferences()
        preset_key = str(getattr(preferences, "preset_key", "blockout") or "blockout")
        custom_keys = tuple(str(item) for item in getattr(preferences, "custom_action_keys", ()) or ())
        self._syncing_map_studio_tool_belt_preferences = True
        try:
            self._map_studio_custom_belt_keys = custom_keys
            index = self.map_studio_tool_belt_preset_combo.findData(preset_key)
            if index < 0:
                index = self.map_studio_tool_belt_preset_combo.findData("blockout")
            if index >= 0 and self.map_studio_tool_belt_preset_combo.currentIndex() != index:
                self.map_studio_tool_belt_preset_combo.setCurrentIndex(index)
        finally:
            self._syncing_map_studio_tool_belt_preferences = False

    def _persist_map_studio_tool_belt_preferences(self) -> None:
        if self._syncing_map_studio_tool_belt_preferences:
            return
        self.controller.set_map_studio_tool_belt_preferences(
            preset_key=self._selected_map_studio_tool_belt_preset_key(),
            custom_action_keys=self._map_studio_custom_belt_keys,
        )

    def _handle_map_studio_tool_belt_preset_changed(self, _index: int) -> None:
        self._persist_map_studio_tool_belt_preferences()
        self._refresh_map_studio_tool_belt()

    def _sync_map_studio_tool_belt_preset_for_edit_mode(self, label: str) -> None:
        """Show the most relevant built-in tool belt for the active edit mode."""

        current_preset = self._selected_map_studio_tool_belt_preset_key()
        if current_preset == "custom":
            return
        mode_key = str(label or "Object").strip().lower()
        preset_key = {
            "object": "blockout",
            "vertex": "component",
            "edge": "component",
            "face": "component",
            "multi-component": "maya_modeling",
            "walkmesh": "component",
            "placement": "gameplay",
            "terrain": "terrain",
            "export": "export",
        }.get(mode_key, "blockout")
        index = self.map_studio_tool_belt_preset_combo.findData(preset_key)
        if index < 0 or self.map_studio_tool_belt_preset_combo.currentIndex() == index:
            return
        previous = self.map_studio_tool_belt_preset_combo.blockSignals(True)
        try:
            self.map_studio_tool_belt_preset_combo.setCurrentIndex(index)
        finally:
            self.map_studio_tool_belt_preset_combo.blockSignals(previous)
        self._refresh_map_studio_tool_belt()

    def _refresh_map_studio_tool_index(self) -> None:
        """Populate the indexed custom-belt picker with all Map Studio tools."""

        self._map_studio_tool_action_index = {
            str(getattr(action, "key", "") or ""): action
            for action in self.controller.available_map_studio_tool_belt_actions()
            if str(getattr(action, "key", "") or "")
        }
        search_results = self.controller.map_studio_tool_command_search("", limit=0, include_planned=True)
        combo = getattr(self, "map_studio_custom_tool_combo", None)
        if combo is not None:
            previous_text = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                for result in search_results:
                    state = "usable" if bool(getattr(result, "implemented", False)) else "planned"
                    combo.addItem(f"{result.display_label} [{state}]", result.key)
                    index = combo.count() - 1
                    combo.setItemData(index, self._map_studio_command_search_tooltip(result), QtCore.Qt.ToolTipRole)
                if previous_text:
                    combo.setEditText(previous_text)
            finally:
                combo.blockSignals(False)
            completer = combo.completer()
            if completer is not None:
                completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
                completer.setFilterMode(QtCore.Qt.MatchContains)
                completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        command_combo = getattr(self, "map_studio_command_search_combo", None)
        if command_combo is not None:
            previous_text = command_combo.currentText()
            command_combo.blockSignals(True)
            try:
                command_combo.clear()
                for result in search_results:
                    if not bool(getattr(result, "implemented", False)):
                        continue
                    command_combo.addItem(result.display_label, result.key)
                    index = command_combo.count() - 1
                    command_combo.setItemData(index, self._map_studio_command_search_tooltip(result), QtCore.Qt.ToolTipRole)
                if previous_text:
                    command_combo.setEditText(previous_text)
            finally:
                command_combo.blockSignals(False)
            completer = command_combo.completer()
            if completer is not None:
                completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
                completer.setFilterMode(QtCore.Qt.MatchContains)
                completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self._update_map_studio_command_search_readiness()

    def _map_studio_command_search_tooltip(self, result: Any) -> str:
        """Format command-search metadata without owning the policy itself."""

        lines = [
            str(getattr(result, "description", "") or "").strip(),
            f"Capability: {str(getattr(result, 'capability_stage', '') or 'unknown').replace('_', ' ')}",
        ]
        resource_text = ", ".join(str(item) for item in tuple(getattr(result, "resource_impacts", ()) or ()))
        if resource_text:
            lines.append(f"Affects: {resource_text}")
        guardrail = str(getattr(result, "kotor_guardrail", "") or "").strip()
        if guardrail:
            lines.append(f"KOTOR: {guardrail}")
        readiness = str(getattr(result, "readiness_summary", "") or "").strip()
        if readiness:
            lines.append(readiness)
        return "\n".join(line for line in lines if line)

    def _map_studio_command_search_route(self, result: Any | None) -> Any | None:
        """Resolve the current dispatcher route for a command-search result."""

        if result is None:
            return None
        key = str(getattr(result, "key", "") or "").strip()
        if not key or key not in self._map_studio_tool_action_index:
            return None
        return resolve_map_studio_tool_belt_action(key, self._map_studio_tool_action_context(key))

    def _map_studio_command_search_context_tooltip(self, result: Any | None) -> str:
        """Format command-search tooltip with current route readiness appended."""

        if result is None:
            return ""
        tooltip = self._map_studio_command_search_tooltip(result)
        route = self._map_studio_command_search_route(result)
        if route is None or bool(getattr(route, "enabled", True)):
            return tooltip
        reason = str(getattr(route, "disabled_reason", "") or "").strip()
        if not reason:
            return tooltip
        return f"{tooltip}\nNot ready now: {reason}" if tooltip else f"Not ready now: {reason}"

    def _map_studio_tool_route_tooltip(self, action: Any, route: Any) -> str:
        """Format dispatcher route metadata for tool-belt buttons and menus."""

        lines = [
            str(getattr(action, "description", "") or getattr(route, "status_message", "") or "").strip(),
            f"Capability: {str(getattr(route, 'capability_stage', '') or 'unknown').replace('_', ' ')}",
        ]
        resource_text = ", ".join(str(item) for item in tuple(getattr(route, "resource_impacts", ()) or ()))
        if resource_text:
            lines.append(f"Affects: {resource_text}")
        guardrail = str(getattr(action, "kotor_guardrail", "") or "").strip()
        if guardrail:
            lines.append(f"KOTOR: {guardrail}")
        readiness = str(getattr(route, "readiness_summary", "") or "").strip()
        if readiness:
            lines.append(readiness)
        if not bool(getattr(route, "enabled", True)):
            reason = str(getattr(route, "disabled_reason", "") or "").strip()
            if reason:
                lines.append(f"Not ready: {reason}")
        return "\n".join(line for line in lines if line)

    def _map_studio_command_search_summary(self, result: Any | None) -> str:
        if result is None:
            return (
                "Command readiness: choose a Map Studio tool to see capability stage, "
                "affected KOTOR resources, and export/game-proof impact."
            )
        label = str(getattr(result, "display_label", "") or getattr(result, "label", "") or getattr(result, "key", "") or "Command")
        capability = str(getattr(result, "capability_stage", "") or "unknown").replace("_", " ")
        resource_text = ", ".join(str(item) for item in tuple(getattr(result, "resource_impacts", ()) or ())) or "none"
        readiness = str(getattr(result, "readiness_summary", "") or "").strip()
        summary = f"Command readiness: {label} | Capability: {capability} | Affects: {resource_text}. {readiness}".strip()
        route = self._map_studio_command_search_route(result)
        if route is not None and not bool(getattr(route, "enabled", True)):
            reason = str(getattr(route, "disabled_reason", "") or "").strip()
            if reason:
                summary = f"{summary} Not ready now: {reason}"
        return summary

    def _selected_map_studio_command_search_result(self) -> Any | None:
        combo = getattr(self, "map_studio_command_search_combo", None)
        if combo is None:
            return None
        key = str(combo.currentData() or "").strip()
        if key:
            matches = self.controller.map_studio_tool_command_search(key, limit=0)
            for result in matches:
                if str(getattr(result, "key", "") or "") == key:
                    return result
        query = str(combo.currentText() or "").strip()
        if not query:
            return None
        matches = self.controller.map_studio_tool_command_search(query, limit=1)
        return matches[0] if matches else None

    def _update_map_studio_command_search_readiness(self) -> None:
        label = getattr(self, "map_studio_command_search_readiness_label", None)
        if label is None:
            return
        result = self._selected_map_studio_command_search_result()
        summary = self._map_studio_command_search_summary(result)
        label.setText(summary)
        label.setToolTip(self._map_studio_command_search_context_tooltip(result) if result is not None else summary)

    def _clear_map_studio_tool_belt_layout(self, layout: QtWidgets.QLayout | None = None) -> None:
        target_layout = layout or self.map_studio_tool_belt_layout
        while target_layout.count():
            item = target_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _connect_map_studio_tool_context_refresh_signals(self) -> None:
        """Refresh command-surface readiness when visible Map Studio context changes."""

        for combo_name in (
            "roomPrimitiveTransformComboBox",
            "primitiveSurfaceComboBox",
            "roomSurfaceComboBox",
            "curveGuidePurposeComboBox",
        ):
            combo = getattr(self.builder_tab, combo_name, None)
            if combo is not None:
                combo.currentIndexChanged.connect(lambda _index=0: self._refresh_map_studio_tool_context())
        for line_name in (
            "curveGuideNameLineEdit",
        ):
            line = getattr(self.builder_tab, line_name, None)
            if line is not None:
                line.textChanged.connect(lambda _text="": self._refresh_map_studio_tool_context())
        for spin_name in (
            "duplicateSpecialCountSpinBox",
            "duplicateSpecialOffsetXSpinBox",
            "duplicateSpecialOffsetYSpinBox",
            "duplicateSpecialOffsetZSpinBox",
            "duplicateSpecialRotationZSpinBox",
            "duplicateSpecialScaleXSpinBox",
            "duplicateSpecialScaleYSpinBox",
            "duplicateSpecialScaleZSpinBox",
            "curveGuidePoint1XSpinBox",
            "curveGuidePoint1YSpinBox",
            "curveGuidePoint1ZSpinBox",
            "curveGuidePoint2XSpinBox",
            "curveGuidePoint2YSpinBox",
            "curveGuidePoint2ZSpinBox",
            "curveGuidePoint3XSpinBox",
            "curveGuidePoint3YSpinBox",
            "curveGuidePoint3ZSpinBox",
        ):
            spin = getattr(self.builder_tab, spin_name, None)
            if spin is not None:
                spin.valueChanged.connect(lambda _value=0: self._refresh_map_studio_tool_context())

    def _refresh_map_studio_tool_context(self) -> None:
        """Rebuild Map Studio command surfaces from the current Builder selection."""

        self._refresh_map_studio_tool_belt()
        self._update_map_studio_command_search_readiness()

    def _refresh_map_studio_tool_belt(self) -> None:
        self._clear_map_studio_tool_belt_layout(self.map_studio_tool_belt_layout)
        custom_layout = getattr(self, "map_studio_custom_tool_belt_layout", None)
        if custom_layout is not None:
            self._clear_map_studio_tool_belt_layout(custom_layout)
        preset_key = self._selected_map_studio_tool_belt_preset_key()
        actions = self.controller.map_studio_tool_belt_actions_for_preset(
            preset_key,
            custom_action_keys=self._map_studio_custom_belt_keys,
        )
        if not actions and preset_key == "custom":
            placeholder = QtWidgets.QLabel("Customize the belt to choose visible tools.")
            placeholder.setObjectName("mapStudioToolBeltEmptyCustomLabel")
            self.map_studio_tool_belt_layout.addWidget(placeholder)
            self.map_studio_tool_belt_layout.addStretch(1)
        else:
            self._populate_map_studio_tool_belt_layout(self.map_studio_tool_belt_layout, actions)
        if custom_layout is not None:
            custom_actions = self.controller.map_studio_tool_belt_actions_for_preset(
                "custom",
                custom_action_keys=self._map_studio_custom_belt_keys,
            )
            if custom_actions:
                self._populate_map_studio_tool_belt_layout(custom_layout, custom_actions)
            else:
                placeholder = QtWidgets.QLabel("Use + to add any indexed Map Studio tool.")
                placeholder.setObjectName("mapStudioCustomToolBeltEmptyLabel")
                custom_layout.addWidget(placeholder)
                custom_layout.addStretch(1)

    def _build_map_studio_tool_qaction(self, action: Any, *, context_menu: bool = False) -> QtGui.QAction:
        """Create a Qt action for one Map Studio tool without owning command policy."""

        key = str(getattr(action, "key", "") or "")
        label = str(getattr(action, "label", key) or key)
        route = resolve_map_studio_tool_belt_action(key, self._map_studio_tool_action_context(key))
        qaction = QtGui.QAction(label, self)
        qaction.setObjectName(
            f"mapStudioToolContextAction_{key}" if context_menu else f"mapStudioToolBeltQAction_{key}"
        )
        qaction.setData(key)
        qaction.setEnabled(bool(route.enabled) if context_menu else bool(getattr(action, "implemented", False)))
        tooltip = self._map_studio_tool_route_tooltip(action, route)
        hotkey = str(getattr(action, "hotkey", "") or "")
        if hotkey:
            tooltip = f"{tooltip}\nHotkey: {hotkey}" if tooltip else f"Hotkey: {hotkey}"
        shortcut_sequence = str(getattr(action, "shortcut_sequence", "") or "")
        shortcut_behavior = str(getattr(action, "shortcut_behavior", "") or "")
        if shortcut_sequence:
            qaction.setProperty("mapStudioShortcutSequence", shortcut_sequence)
            if context_menu:
                qaction.setShortcut(QtGui.QKeySequence(shortcut_sequence))
            shortcut_label = f"Shortcut sequence: {shortcut_sequence}"
            if shortcut_behavior:
                qaction.setProperty("mapStudioShortcutBehavior", shortcut_behavior)
                shortcut_label = f"{shortcut_label} ({shortcut_behavior.replace('_', ' ')})"
            tooltip = f"{tooltip}\n{shortcut_label}" if tooltip else shortcut_label
        if tooltip:
            qaction.setToolTip(tooltip)
            qaction.setStatusTip(tooltip)
        qaction.triggered.connect(
            lambda _checked=False, tool_action=action: self._handle_map_studio_tool_belt_action(tool_action)
        )
        return qaction

    #: Belt actions that stay visible as top-level buttons; everything else
    #: folds into grouped dropdowns so the shelf stays one clean row.
    _BELT_PINNED_KEYS = ("select", "move", "duplicate_selected", "delete_selected", "texture_paint", "paint_wok")
    _BELT_MODE_KEYS = ("object", "vertex", "edge", "face", "walkmesh", "terrain")
    _BELT_CREATE_KEYS = (
        "blockout_room",
        "create_room",
        "corridor",
        "floor",
        "plane",
        "wall",
        "cube",
        "cylinder",
        "sphere",
        "cone",
        "torus",
        "ramp",
        "stairs",
        "door_frame",
        "arch",
        "terrain_patch",
        "primitive",
    )

    def _make_map_studio_belt_tool_button(self, action: Any) -> QtWidgets.QToolButton:
        key = str(getattr(action, "key", "") or "")
        qaction = self._build_map_studio_tool_qaction(action)
        button = QtWidgets.QToolButton()
        button.setDefaultAction(qaction)
        button.setObjectName(f"mapStudioToolBeltButton_{key}")
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        button.setEnabled(bool(getattr(action, "implemented", False)))
        return button

    def _make_map_studio_belt_group(self, title: str, group_key: str, actions: list[Any]) -> QtWidgets.QToolButton:
        """One dropdown holding a group of belt tools (buttons stay findable)."""

        dropdown = QtWidgets.QToolButton()
        dropdown.setObjectName(f"mapStudioToolBeltGroup_{group_key}")
        dropdown.setText(f"{title} ▾")
        dropdown.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        dropdown.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(dropdown)
        menu.setObjectName(f"mapStudioToolBeltGroupMenu_{group_key}")
        container = QtWidgets.QWidget(menu)
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(4)
        columns = 4
        for index, action in enumerate(actions):
            button = self._make_map_studio_belt_tool_button(action)
            button.clicked.connect(menu.close)
            grid.addWidget(button, index // columns, index % columns)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setDefaultWidget(container)
        menu.addAction(widget_action)
        dropdown.setMenu(menu)
        return dropdown

    def _populate_map_studio_tool_belt_layout(self, layout: QtWidgets.QHBoxLayout, actions: tuple[Any, ...] | list[Any]) -> None:
        """Draw the belt: pinned essentials + Mode/Create/Tools dropdowns."""

        pinned: list[Any] = []
        modes: list[Any] = []
        creators: list[Any] = []
        extras: list[Any] = []
        for action in actions:
            key = str(getattr(action, "key", "") or "")
            if not key:
                continue
            if key in self._BELT_PINNED_KEYS:
                pinned.append(action)
            elif key in self._BELT_MODE_KEYS:
                modes.append(action)
            elif key in self._BELT_CREATE_KEYS:
                creators.append(action)
            else:
                extras.append(action)
        for action in pinned:
            layout.addWidget(self._make_map_studio_belt_tool_button(action))
        if creators:
            layout.addWidget(self._make_map_studio_belt_group("Create", "create", creators))
        if extras:
            layout.addWidget(self._make_map_studio_belt_group("Tools", "tools", extras))
        if modes:
            # Modes live in the toolbar's Edit Mode combo; the belt keeps them
            # reachable (and test-clickable) without eating shelf width.
            layout.addWidget(self._make_map_studio_belt_group("Mode", "mode", modes))
        layout.addStretch(1)

    def _add_map_studio_context_menu_action(self, menu: QtWidgets.QMenu, action: Any) -> None:
        """Add one dispatcher-backed Map Studio action to a context menu."""

        key = str(getattr(action, "key", "") or "")
        if not key:
            return
        menu.addAction(self._build_map_studio_tool_qaction(action, context_menu=True))

    def _open_map_studio_tool_context_menu(self, widget: QtWidgets.QWidget, pos: QtCore.QPoint) -> None:
        """Open a context command surface backed by the shared Map Studio dispatcher."""

        menu = QtWidgets.QMenu(widget)
        menu.setObjectName("mapStudioToolContextMenu")
        focus_search_action = menu.addAction("Command Search...")
        focus_search_action.setObjectName("mapStudioToolContextMenuCommandSearchAction")
        focus_search_action.triggered.connect(self._focus_map_studio_command_search)
        customize_action = menu.addAction("Customize Tool Belt...")
        customize_action.setObjectName("mapStudioToolContextMenuCustomizeAction")
        customize_action.triggered.connect(self._customize_map_studio_tool_belt)

        current_actions = self.controller.map_studio_tool_belt_actions_for_preset(
            self._selected_map_studio_tool_belt_preset_key(),
            custom_action_keys=self._map_studio_custom_belt_keys,
        )
        if current_actions:
            current_menu = menu.addMenu("Current Belt")
            current_menu.setObjectName("mapStudioToolContextMenuCurrentBeltMenu")
            for action in current_actions:
                self._add_map_studio_context_menu_action(current_menu, action)

        query = ""
        combo = getattr(self, "map_studio_command_search_combo", None)
        if combo is not None:
            query = str(combo.currentText() or "").strip()
        search_results = self.controller.map_studio_tool_command_search(query, limit=18)
        if search_results:
            search_menu = menu.addMenu("Matching Commands" if query else "Common Commands")
            search_menu.setObjectName("mapStudioToolContextMenuSearchResultsMenu")
            added: set[str] = set()
            for result in search_results:
                key = str(getattr(result, "key", "") or "")
                if not key or key in added:
                    continue
                action = self._map_studio_tool_action_index.get(key)
                if action is None:
                    continue
                self._add_map_studio_context_menu_action(search_menu, action)
                added.add(key)

        if menu.actions():
            menu.exec(widget.mapToGlobal(pos))

    def _build_map_studio_mode_marking_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        """Build the Maya-style viewport mode marking menu for Map Studio."""

        menu = QtWidgets.QMenu(parent or self)
        menu.setObjectName("mapStudioModeMarkingMenu")
        frame = QtWidgets.QFrame(menu)
        frame.setObjectName("mapStudioModeMarkingMenuRadial")
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        # Two-mode design: Component Modeling already picks
        # faces, edges, and vertices by hover, so per-component modes are
        # redundant here. Edit Mode activates component modeling; Object Mode is
        # select/transform/delete; Terrain and Placement stay as workflows.
        entries = (
            ("edit", "Multi-Component Mode", "Multi-Component", 0, 0, "mapStudioModeMarkingAction_edit", "mapStudioModeMarkingButton_edit"),
            ("object", "Object Mode", "Object", 0, 1, "mapStudioModeMarkingAction_object", "mapStudioModeMarkingButton_object"),
            ("terrain", "Terrain", "Terrain", 1, 0, "mapStudioModeMarkingAction_terrain", "mapStudioModeMarkingButton_terrain"),
            ("placement", "Placement", "Placement", 1, 1, "mapStudioModeMarkingAction_placement", "mapStudioModeMarkingButton_placement"),
        )
        for key, label, mode_label, row, column, action_name, button_name in entries:
            action = QtGui.QAction(label, menu)
            action.setObjectName(action_name)
            action.setData(key)
            action.triggered.connect(lambda _checked=False, mode_key=key: self._run_map_studio_mode_marking_action(mode_key))
            action.triggered.connect(menu.close)
            button = QtWidgets.QToolButton(frame)
            button.setObjectName(button_name)
            button.setDefaultAction(action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            button.setMinimumWidth(82)
            button.setToolTip(
                "Automatically target the nearest visible face, edge, or vertex; RMB opens the marking menu." if key == "edit"
                else f"Switch Map Studio to {mode_label} mode."
            )
            layout.addWidget(button, row, column)
            menu.addAction(action)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setObjectName("mapStudioModeMarkingRadialWidgetAction")
        widget_action.setDefaultWidget(frame)
        menu.insertAction(menu.actions()[0] if menu.actions() else None, widget_action)
        menu.insertSeparator(menu.actions()[1] if len(menu.actions()) > 1 else None)
        return menu

    def _open_map_studio_mode_marking_menu(self, global_pos: QtCore.QPoint) -> None:
        if self._open_map_studio_component_marking_menu(global_pos):
            return
        menu = self._build_map_studio_mode_marking_menu(self.viewport_panel)
        menu.exec(global_pos)

    def _handle_map_studio_hover_context_changed(self, context) -> None:
        """Echo the hovered component in the status bar without stealing focus."""

        summary = map_studio_hover_context_summary(context) if context is not None else ""
        if summary:
            self.statusBar().showMessage(f"Map Studio hover: {summary}", 2500)

    def _open_map_studio_component_marking_menu(self, global_pos: QtCore.QPoint) -> bool:
        """Open the compact component-modeling menu for the hovered target."""

        context = self.viewport_panel.current_map_studio_hover_context()
        tree = map_studio_marking_menu_tree_for_hover(context)
        if tree is None:
            return False
        menu = MapStudioComponentMarkingMenu(tree, context, self.viewport_panel)
        menu.actionSelected.connect(self._handle_map_studio_component_marking_action)
        menu.open_at(global_pos)
        return True

    def _handle_map_studio_component_marking_action(self, action_key: str, target: str) -> None:
        action = map_studio_marking_menu_action(action_key)
        if action is None:
            return
        if action_key in {"face_extrude", "edge_extrude"} and self.viewport_panel.arm_component_extrude():
            # Maya-style: the menu arms the interactive pull; drag positions it.
            return
        if action_key == "edge_bevel" and self.viewport_panel.arm_component_bevel():
            # The bevel remains a preview until the user releases the drag or
            # presses Apply in the persistent operator strip.
            return
        if action_key in {
            "face_delete",
            "face_set_texture",
            "face_extrude",
            "face_inset",
            "face_move",
            "face_flat",
            "face_flip",
            "face_split",
            "vertex_move",
            "vertex_weld",
            "vertex_delete",
            "edge_move",
            "edge_bevel",
            "edge_split",
            "edge_collapse",
            "edge_delete",
        } and self._apply_component_modeling_imported_face_action(action_key, target):
            return
        if not action.implemented:
            impacts = ", ".join(action.resource_impacts) or "editor state only"
            self.statusBar().showMessage(
                f"Map Studio {action.label} -> {target}: registered but not wired to geometry yet"
                f" (affects {impacts}). {action.kotor_guardrail}",
                6000,
            )
            self._log(f"Map Studio component-modeling action (read-only): {action.key} target={target}")
            return
        belt = self._map_studio_tool_action_for_key(action.tool_belt_key) if action.tool_belt_key else None
        if belt is not None:
            self._handle_map_studio_tool_belt_action(belt)

    def _map_studio_live_topology_source(self, payload: dict) -> ImportedMeshRoomPrimitive | None:
        """Capture one immutable preview source; repeated drags never accumulate."""

        room_resref = str(payload.get("room_resref", "") or "")
        mesh_role = str(payload.get("mesh_role", "") or "")
        key = (room_resref, mesh_role, str(payload.get("kind", "") or ""))
        cached = getattr(self, "_map_studio_topology_preview_source", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        self._cancel_map_studio_component_preview()
        rows = self.viewport_panel.preview_room_mesh_payloads(room_resref)
        surfaces: list[ImportedMeshSurface] = []
        for index, row in enumerate(rows):
            vertices = tuple(row.get("vertices", ()) or ())
            faces = tuple(row.get("faces", ()) or ())
            if not vertices or not faces:
                continue
            surfaces.append(
                ImportedMeshSurface(
                    name=str(row.get("name") or f"{room_resref}_preview_{index}"),
                    texture=str(row.get("texture") or ""),
                    vertices=vertices,
                    faces=faces,
                    uvs=tuple(row.get("uvs", ()) or ()),
                    normals=tuple(row.get("normals", ()) or ()),
                    lightmap=str(row.get("lightmap") or ""),
                    texture_names=tuple(row.get("texture_names", ()) or ()),
                    tex_count=int(row.get("tex_count", 1) or 1),
                    uvs_lm=tuple(row.get("uvs_lm", ()) or ()),
                    face_mats=tuple(int(value) for value in tuple(row.get("face_mats", ()) or ())),
                )
            )
        if not surfaces:
            return None
        primitive = ImportedMeshRoomPrimitive(room_resref=room_resref, surfaces=tuple(surfaces))
        self._map_studio_topology_preview_source = (key, primitive)
        return primitive

    def _map_studio_prepared_topology_session(
        self,
        source: ImportedMeshRoomPrimitive,
        payload: dict,
        operation: str,
    ) -> MapStudioLiveTopologySession:
        """Prepare once per gesture/option identity, then reuse for drag frames."""

        room_resref = str(payload.get("room_resref", "") or "")
        mesh_role = str(payload.get("mesh_role", "") or "")
        if operation == "face_extrude":
            face_indices = tuple(sorted({int(value) for value in tuple(payload.get("face_indices", ()) or ())}))
            world_mode = str(payload.get("axis_mode", "normal") or "normal") == "world"
            axis = tuple(float(value) for value in tuple(payload.get("axis", (0.0, 0.0, 1.0)))[:3])
            key = (
                id(source),
                operation,
                room_resref,
                mesh_role,
                face_indices,
                axis if world_mode else None,
            )
            cached = getattr(self, "_map_studio_prepared_topology_preview", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            session = MapStudioLiveTopologySession.prepare_face_extrude(
                source,
                mesh_role,
                face_indices,
                direction=axis if world_mode else None,
            )
        elif operation == "edge_extrude":
            corners = tuple(int(value) for value in tuple(payload.get("edge_corners", (0, 1)))[:2])
            axis = tuple(float(value) for value in tuple(payload.get("axis", (0.0, 0.0, 1.0)))[:3])
            tile_size = float(payload.get("tile_size", 0.0) or 0.0)
            key = (
                id(source),
                operation,
                room_resref,
                mesh_role,
                int(payload.get("face_index", -1)),
                corners,
                axis,
                tile_size,
            )
            cached = getattr(self, "_map_studio_prepared_topology_preview", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            session = MapStudioLiveTopologySession.prepare_edge_extrude(
                source,
                mesh_role,
                int(payload.get("face_index", -1)),
                corners,
                direction=axis,
                tile_size=tile_size,
            )
        elif operation == "edge_bevel":
            corners = tuple(int(value) for value in tuple(payload.get("edge_corners", (0, 1)))[:2])
            key = (
                id(source),
                operation,
                room_resref,
                mesh_role,
                int(payload.get("face_index", -1)),
                corners,
                int(payload.get("segments", 1) or 1),
                float(payload.get("profile", 0.5) or 0.0),
                str(payload.get("miter", "auto") or "auto"),
                float(payload.get("smoothing_angle_degrees", 180.0) or 0.0),
                str(payload.get("uv_mode", "preserve") or "preserve"),
                bool(payload.get("clamp_overlap", True)),
            )
            cached = getattr(self, "_map_studio_prepared_topology_preview", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            session = MapStudioLiveTopologySession.prepare_edge_bevel(
                source,
                mesh_role,
                int(payload.get("face_index", -1)),
                corners,
                segments=int(payload.get("segments", 1) or 1),
                profile=float(payload.get("profile", 0.5) or 0.0),
                miter=str(payload.get("miter", "auto") or "auto"),
                smoothing_angle_degrees=float(payload.get("smoothing_angle_degrees", 180.0) or 0.0),
                uv_mode=str(payload.get("uv_mode", "preserve") or "preserve"),
                clamp_overlap=bool(payload.get("clamp_overlap", True)),
            )
        else:
            raise ValueError(f"Unsupported prepared topology operation: {operation}")
        self._map_studio_prepared_topology_preview = (key, session)
        return session

    def _show_live_imported_surface(self, primitive: ImportedMeshRoomPrimitive, room_resref: str, mesh_role: str) -> bool:
        surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
        if surface_index < 0:
            return False
        surface = primitive.surfaces[surface_index]
        return self.viewport_panel.apply_component_mesh_preview(
            room_resref,
            mesh_role,
            vertices=surface.vertices,
            faces=surface.faces,
            normals=surface.normals,
            uvs=surface.uvs,
            uvs_lm=surface.uvs_lm,
            face_mats=surface.face_mats,
        )

    def _refresh_map_studio_imported_mesh_change(
        self,
        message: str,
        room_resref: str,
        mesh_role: str,
    ) -> bool:
        """Patch one committed mesh while preserving all other GPU residency.

        Component operators mutate exactly one imported-mesh surface.  The
        prior path rebuilt the combined authored + stock preview and routed it
        through ``load_model()``, which cleared every renderer cache and reset
        the framebuffers.  Reusing the already-resident preview (or patching
        the committed surface for immediate actions) keeps camera, texture,
        stock-room, and unrelated mesh resources alive.

        Returns ``True`` when the scoped path was available.  Missing/stale
        preview identity falls back to the full model replacement so undo and
        export truth remain controller-owned.
        """

        cached_reader = getattr(self.controller, "last_committed_imported_mesh_room", None)
        room_spec = cached_reader(room_resref) if callable(cached_reader) else None
        if room_spec is None:
            room_spec = self.controller.imported_mesh_room(room_resref)
        live_surface_count = sum(
            1 for _room_node, _mesh_node in self.viewport_panel._iter_room_preview_mesh_nodes(room_resref)
        )
        committed_surface_count = len(tuple(getattr(getattr(room_spec, "primitive", None), "surfaces", ()) or ()))
        patched = bool(
            room_spec is not None
            and committed_surface_count == live_surface_count
            and self._show_live_imported_surface(room_spec.primitive, room_resref, mesh_role)
        )
        promoted = bool(
            patched
            and self.viewport_panel.promote_component_mesh_preview(room_resref, mesh_role)
        )
        if promoted:
            self._map_studio_topology_preview_source = None
            self._map_studio_prepared_topology_preview = None
        self._refresh_map_studio_geometry_change(
            message,
            rebuild_viewport_model=not promoted,
            refresh_scene_tree=not promoted,
        )
        return promoted

    def _preview_map_studio_component_extrude(self, payload: dict) -> None:
        source = self._map_studio_live_topology_source(payload)
        if source is None:
            return
        room_resref = str(payload.get("room_resref", "") or "")
        mesh_role = str(payload.get("mesh_role", "") or "")
        distance = float(payload.get("distance", 0.0) or 0.0)
        started = perf_counter()
        try:
            operation = "edge_extrude" if str(payload.get("kind", "") or "") == "edge" else "face_extrude"
            session = self._map_studio_prepared_topology_session(source, payload, operation)
            updated = session.evaluate(distance)
            self._show_live_imported_surface(updated, room_resref, mesh_role)
        except ValueError as exc:
            self.statusBar().showMessage(f"Extrude preview: {exc}", 2500)
        finally:
            self._last_map_studio_topology_preview_ms = (perf_counter() - started) * 1000.0

    def _preview_map_studio_component_bevel(self, payload: dict) -> None:
        if str(payload.get("kind", "") or "") == "multi_edge_bevel":
            return
        source = self._map_studio_live_topology_source(payload)
        if source is None:
            return
        room_resref = str(payload.get("room_resref", "") or "")
        mesh_role = str(payload.get("mesh_role", "") or "")
        started = perf_counter()
        try:
            session = self._map_studio_prepared_topology_session(source, payload, "edge_bevel")
            updated = session.evaluate(float(payload.get("amount", 0.25) or 0.25))
            self._show_live_imported_surface(updated, room_resref, mesh_role)
        except ValueError as exc:
            self.statusBar().showMessage(f"Bevel preview: {exc}", 2500)
        finally:
            self._last_map_studio_topology_preview_ms = (perf_counter() - started) * 1000.0

    def _cancel_map_studio_component_preview(self) -> None:
        self.viewport_panel.clear_component_mesh_preview()
        self._map_studio_topology_preview_source = None
        self._map_studio_prepared_topology_preview = None

    def _commit_map_studio_component_bevel(self, payload: dict) -> None:
        multi_edges = tuple(
            tuple(int(value) for value in tuple(edge)[:2])
            for edge in tuple(payload.get("edge_vertex_indices") or ())
        )
        ok, message = self.controller.apply_imported_mesh_room_component_op(
            room_resref=str(payload.get("room_resref", "") or ""),
            op="multi_edge_bevel" if len(multi_edges) > 1 else "edge_bevel",
            mesh_role=str(payload.get("mesh_role", "") or ""),
            face_index=int(payload.get("face_index", -1)),
            edge_corners=tuple(int(value) for value in tuple(payload.get("edge_corners", (0, 1)))[:2]),
            amount=float(payload.get("amount", 0.25) or 0.25),
            segments=int(payload.get("segments", 1) or 1),
            profile=float(payload.get("profile", 0.5) or 0.0),
            miter=str(payload.get("miter", "auto") or "auto"),
            smoothing_angle_degrees=float(payload.get("smoothing_angle_degrees", 180.0) or 0.0),
            uv_mode=str(payload.get("uv_mode", "preserve") or "preserve"),
            clamp_overlap=bool(payload.get("clamp_overlap", True)),
            edge_vertex_indices=multi_edges,
        )
        self.statusBar().showMessage(message, 6000)
        self._log(f"Map Studio: {message}")
        if ok:
            self._refresh_map_studio_imported_mesh_change(
                message,
                str(payload.get("room_resref", "") or ""),
                str(payload.get("mesh_role", "") or ""),
            )
        else:
            self._cancel_map_studio_component_preview()

    def _commit_map_studio_component_extrude(self, payload: dict) -> None:
        """Land one interactive Ctrl+E extrude (single undo entry).

        Faces: extrude along the region normal, then select the new caps so
        the user can immediately pull again.  Edges: append the outward quad.
        """

        kind = str(payload.get("kind", "") or "")
        room_resref = str(payload.get("room_resref", "") or "")
        mesh_role = str(payload.get("mesh_role", "") or "")
        distance = float(payload.get("distance", 0.0) or 0.0)
        if abs(distance) <= 1.0e-4:
            self._cancel_map_studio_component_preview()
            self.statusBar().showMessage("Extrude cancelled (no drag distance).", 3000)
            return
        if kind == "face":
            face_indices = tuple(int(v) for v in tuple(payload.get("face_indices", ()) or ()))
            if not face_indices:
                self._cancel_map_studio_component_preview()
                return
            preview_source = getattr(self, "_map_studio_topology_preview_source", None)
            primitive_before = None
            if preview_source is not None:
                source_key, source_primitive = preview_source
                if tuple(source_key[:2]) == (room_resref, mesh_role):
                    primitive_before = source_primitive
            if primitive_before is None:
                room_spec = self.controller.imported_mesh_room(room_resref)
                primitive_before = getattr(room_spec, "primitive", None)
            before_count = -1
            if primitive_before is not None:
                surface_index = imported_mesh_surface_index_for_role(primitive_before, mesh_role)
                if surface_index >= 0:
                    before_count = len(primitive_before.surfaces[surface_index].faces)
            world_mode = str(payload.get("axis_mode", "normal")) == "world"
            axis = tuple(float(v) for v in tuple(payload.get("axis", (0.0, 0.0, 1.0)))[:3])
            ok, message = self.controller.extrude_imported_mesh_room_faces(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
                distance=distance,
                direction=axis if world_mode else None,
            )
            self.statusBar().showMessage(message, 6000)
            self._log(f"Map Studio: {message}")
            if not ok:
                self._cancel_map_studio_component_preview()
                return
            self._refresh_map_studio_imported_mesh_change(message, room_resref, mesh_role)
            if before_count >= 0:
                # extrude rebuilds faces as kept + caps + sides; caps start at
                # (before - len(region)) in the rebuilt list.
                cap_start = before_count - len(face_indices)
                caps = tuple(range(cap_start, cap_start + len(face_indices)))
                selected = self.viewport_panel.select_map_studio_faces(room_resref, mesh_role, caps)
                if selected:
                    self.statusBar().showMessage(
                        f"{message} New cap selected — Ctrl+E pulls again.", 6000
                    )
            return
        if kind == "edge":
            axis = tuple(float(v) for v in tuple(payload.get("axis", (0.0, 0.0, 1.0)))[:3])
            delta = (axis[0] * distance, axis[1] * distance, axis[2] * distance)
            ok, message = self.controller.apply_imported_mesh_room_component_op(
                room_resref=room_resref,
                op="edge_extrude",
                mesh_role=mesh_role,
                face_index=int(payload.get("face_index", -1)),
                edge_corners=tuple(int(v) for v in tuple(payload.get("edge_corners", (0, 1)))[:2]),
                delta=delta,
            )
            self.statusBar().showMessage(message, 6000)
            self._log(f"Map Studio: {message}")
            if ok:
                self._refresh_map_studio_imported_mesh_change(message, room_resref, mesh_role)
            else:
                self._cancel_map_studio_component_preview()

    @staticmethod
    def _map_studio_barycentric_weights(point, triangle) -> tuple[float, float, float] | None:
        if len(tuple(point or ())) < 3 or len(tuple(triangle or ())) < 3:
            return None
        a, b, c = (tuple(float(value) for value in row[:3]) for row in tuple(triangle)[:3])
        p = tuple(float(value) for value in tuple(point)[:3])
        v0 = tuple(b[index] - a[index] for index in range(3))
        v1 = tuple(c[index] - a[index] for index in range(3))
        v2 = tuple(p[index] - a[index] for index in range(3))
        d00 = sum(v0[index] * v0[index] for index in range(3))
        d01 = sum(v0[index] * v1[index] for index in range(3))
        d11 = sum(v1[index] * v1[index] for index in range(3))
        d20 = sum(v2[index] * v0[index] for index in range(3))
        d21 = sum(v2[index] * v1[index] for index in range(3))
        denominator = (d00 * d11) - (d01 * d01)
        if abs(denominator) <= 1.0e-18:
            return None
        second = ((d11 * d20) - (d01 * d21)) / denominator
        third = ((d00 * d21) - (d01 * d20)) / denominator
        return (1.0 - second - third, second, third)

    def _map_studio_component_local_point(self, entry: dict) -> tuple[float, float, float] | None:
        room = str(entry.get("room_resref") or "")
        role = str(entry.get("mesh_role") or "")
        face_index = int(entry.get("face_index", -1))
        world_triangle = tuple(entry.get("face_world_points") or ())[:3]
        weights = self._map_studio_barycentric_weights(entry.get("world_point") or (), world_triangle)
        room_spec = self.controller.imported_mesh_room(room)
        primitive = getattr(room_spec, "primitive", None)
        if primitive is None or weights is None:
            return None
        surface_index = imported_mesh_surface_index_for_role(primitive, role)
        if surface_index < 0:
            return None
        surface = primitive.surfaces[surface_index]
        if not 0 <= face_index < len(surface.faces):
            return None
        face = tuple(int(value) for value in tuple(surface.faces[face_index])[:3])
        if len(face) < 3:
            return None
        return tuple(
            sum(float(surface.vertices[face[corner]][axis]) * weights[corner] for corner in range(3))
            for axis in range(3)
        )

    @staticmethod
    def _ordered_map_studio_boundary_loop(edges) -> tuple[int, ...]:
        adjacency: dict[int, set[int]] = {}
        clean_edges = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in tuple(edges or ())
            if len(tuple(edge or ())) >= 2 and int(edge[0]) >= 0 and int(edge[1]) >= 0
        }
        for first, second in clean_edges:
            if first == second:
                return ()
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return ()
        start = min(adjacency)
        loop = [start]
        previous = None
        current = start
        while True:
            candidates = sorted(value for value in adjacency[current] if value != previous)
            if not candidates:
                return ()
            following = candidates[0]
            if following == start:
                return tuple(loop) if len(loop) == len(adjacency) else ()
            if following in loop:
                return ()
            loop.append(following)
            previous, current = current, following

    def _map_studio_baked_modeling_options(self, action_key: str) -> dict[str, Any]:
        """Return persistent options for KOTOR-safe static mesh operators.

        These operators deliberately bake their result into the imported room
        mesh.  They do not claim Maya's dependency-graph history or a live
        deformer handle, both of which would be discarded by MDL/WOK export.
        """

        defaults: dict[str, dict[str, Any]] = {
            "mirror": {"axis": "x", "center": 0.0, "duplicate": True, "merge_tolerance": 1.0e-5},
            "bridge": {"divisions": 0, "taper": 0.0, "twist_degrees": 0.0, "smooth": True},
            "boolean_a_minus_b": {"weld_tolerance": 1.0e-6},
            "multi_cut": {
                "coplanar_angle_degrees": 0.5,
                "plane_tolerance": 1.0e-4,
                "boundary_tolerance": 1.0e-7,
                "snap_tolerance": 1.0e-4,
            },
            "bend_tool": {"axis": "x", "curvature_degrees": 90.0},
            "lattice": {"offset_axis": "z", "upper_layer_offset": 0.5},
            "shrink_wrap": {"projection": "nearest_triangle", "offset": 0.0, "align_normals": False},
            "wrap": {"nearest_count": 4, "influence": 1.0, "max_distance": 0.0},
            "make_hole": {"planarity_tolerance": 1.0e-4, "boundary_tolerance": 1.0e-6},
            "merge_components": {"threshold": 0.01},
            "insert_edge_loop": {"position": 0.5},
            "quad_draw": {
                "planarity_tolerance": 0.25,
                "surface_offset": 0.0,
                "auto_weld": True,
                "weld_tolerance": 1.0e-4,
            },
        }
        key = str(action_key or "").strip()
        stored = getattr(self, "_map_studio_baked_mesh_options", None)
        if not isinstance(stored, dict):
            stored = {}
            self._map_studio_baked_mesh_options = stored
        if key not in stored:
            stored[key] = dict(defaults.get(key, {}))
        return stored[key]

    def _edit_map_studio_baked_modeling_options(self, action_key: str) -> bool:
        """Edit the honest static subset exposed by an advanced shelf tool."""

        key = str(action_key or "").strip()
        if key not in {
            "mirror",
            "bridge",
            "bend_tool",
            "lattice",
            "shrink_wrap",
            "wrap",
            "make_hole",
            "merge_components",
            "insert_edge_loop",
            "quad_draw",
            "multi_cut",
            "boolean_a_minus_b",
        }:
            return False
        options = self._map_studio_baked_modeling_options(key)
        dialog = QtWidgets.QDialog(self)
        title = {
            "mirror": "Mirror Options",
            "bridge": "Bridge Options",
            "bend_tool": "Bend Options",
            "lattice": "Lattice Options",
            "shrink_wrap": "ShrinkWrap Options",
            "wrap": "Wrap Options",
            "make_hole": "Make Hole Options",
            "merge_components": "Merge Options",
            "insert_edge_loop": "Insert Edge Loop Options",
            "quad_draw": "Quad Draw Options",
            "multi_cut": "Multi-Cut Options",
            "boolean_a_minus_b": "Difference A - B Options",
        }[key]
        dialog.setWindowTitle(title)
        dialog.setObjectName(f"mapStudio{key.title().replace('_', '')}BakedOptionsDialog")
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel(
            "This Map Studio operator bakes a static polygon result for KOTOR export; "
            "it does not create a persistent Maya dependency-graph deformer."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        controls: dict[str, Any] = {}

        def _axis_combo(value: str) -> QtWidgets.QComboBox:
            combo = QtWidgets.QComboBox(dialog)
            for axis in ("X", "Y", "Z"):
                combo.addItem(axis, axis.lower())
            index = combo.findData(str(value or "x").lower())
            combo.setCurrentIndex(max(0, index))
            return combo

        def _double(value: float, minimum: float, maximum: float, decimals: int = 4) -> QtWidgets.QDoubleSpinBox:
            spin = QtWidgets.QDoubleSpinBox(dialog)
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(int(decimals))
            spin.setValue(float(value))
            return spin

        if key == "mirror":
            controls["axis"] = _axis_combo(options["axis"])
            controls["center"] = _double(options["center"], -100000.0, 100000.0)
            controls["duplicate"] = QtWidgets.QCheckBox("Keep source and add mirrored copy", dialog)
            controls["duplicate"].setChecked(bool(options["duplicate"]))
            controls["merge_tolerance"] = _double(options["merge_tolerance"], 0.0, 10.0, 6)
            form.addRow("Mirror axis:", controls["axis"])
            form.addRow("Axis-plane center:", controls["center"])
            form.addRow("Mode:", controls["duplicate"])
            form.addRow("Seam tolerance:", controls["merge_tolerance"])
        elif key == "bridge":
            controls["divisions"] = QtWidgets.QSpinBox(dialog)
            controls["divisions"].setRange(0, 1024)
            controls["divisions"].setValue(int(options["divisions"]))
            controls["taper"] = _double(options["taper"], -0.99, 100.0, 3)
            controls["twist_degrees"] = _double(options["twist_degrees"], -3600.0, 3600.0, 2)
            controls["smooth"] = QtWidgets.QCheckBox("Blend normals across the generated strip", dialog)
            controls["smooth"].setChecked(bool(options["smooth"]))
            form.addRow("Divisions:", controls["divisions"])
            form.addRow("Taper:", controls["taper"])
            form.addRow("Twist (degrees):", controls["twist_degrees"])
            form.addRow("Normals:", controls["smooth"])
            limitation = QtWidgets.QLabel(
                "Select exactly two border edges. Divisions create real intermediate edge rows; taper and twist "
                "deform only those rows so both selected borders remain exact. The result is baked for KOTOR export."
            )
            limitation.setWordWrap(True)
            form.addRow("KOTOR-safe bridge:", limitation)
        elif key == "multi_cut":
            controls["coplanar_angle_degrees"] = _double(options["coplanar_angle_degrees"], 0.0, 45.0, 3)
            controls["plane_tolerance"] = _double(options["plane_tolerance"], 0.0, 1.0, 7)
            controls["boundary_tolerance"] = _double(options["boundary_tolerance"], 0.0, 1.0, 8)
            controls["snap_tolerance"] = _double(options["snap_tolerance"], 0.0, 1.0, 7)
            form.addRow("Coplanar angle:", controls["coplanar_angle_degrees"])
            form.addRow("Plane tolerance:", controls["plane_tolerance"])
            form.addRow("Boundary tolerance:", controls["boundary_tolerance"])
            form.addRow("Component snap tolerance:", controls["snap_tolerance"])
            subset = QtWidgets.QLabel(
                "Place two anchors across one connected coplanar triangle patch. The line previews without changing "
                "KMAP; Enter commits one undo step, Backspace removes the last anchor, and Esc clears before exiting. "
                "Chained turns, MMB slice, subdivisions, edge flow, and crease crossing remain disabled until their "
                "topology contracts are safe."
            )
            subset.setWordWrap(True)
            form.addRow("Safe interactive slice:", subset)
        elif key == "boolean_a_minus_b":
            controls["weld_tolerance"] = _double(options["weld_tolerance"], 0.0, 0.1, 8)
            form.addRow("Topology weld tolerance:", controls["weld_tolerance"])
            subset = QtWidgets.QLabel(
                "Selection order is A then B. This solid Boolean runs only when both selected surfaces are closed, "
                "consistently wound two-manifolds. Open KOTOR floors and walls are refused without changing KMAP; "
                "use the planar architectural subtraction workflow for those surfaces. Cutter-derived cap faces keep B's material."
            )
            subset.setWordWrap(True)
            form.addRow("Closed-solid contract:", subset)
        elif key == "bend_tool":
            controls["axis"] = _axis_combo(options["axis"])
            controls["curvature_degrees"] = _double(options["curvature_degrees"], -3600.0, 3600.0, 2)
            form.addRow("Length axis:", controls["axis"])
            form.addRow("Curvature (degrees):", controls["curvature_degrees"])
        elif key == "lattice":
            controls["offset_axis"] = _axis_combo(options["offset_axis"])
            controls["upper_layer_offset"] = _double(options["upper_layer_offset"], -10000.0, 10000.0)
            form.addRow("Upper cage-layer direction:", controls["offset_axis"])
            form.addRow("Upper cage-layer offset:", controls["upper_layer_offset"])
            subset = QtWidgets.QLabel("Static 2x2x2 trilinear cage; the lower Z cage layer stays fixed.")
            subset.setWordWrap(True)
            form.addRow("Static subset:", subset)
        elif key == "shrink_wrap":
            controls["projection"] = QtWidgets.QComboBox(dialog)
            controls["projection"].addItem("Nearest triangle", "nearest_triangle")
            controls["projection"].addItem("Nearest vertex", "nearest_vertex")
            controls["projection"].setCurrentIndex(
                max(0, controls["projection"].findData(options["projection"]))
            )
            controls["offset"] = _double(options["offset"], -10000.0, 10000.0)
            controls["align_normals"] = QtWidgets.QCheckBox("Align selected normals to the live surface", dialog)
            controls["align_normals"].setChecked(bool(options["align_normals"]))
            form.addRow("Projection:", controls["projection"])
            form.addRow("Surface offset:", controls["offset"])
            form.addRow("Normals:", controls["align_normals"])
        elif key == "wrap":
            controls["nearest_count"] = QtWidgets.QSpinBox(dialog)
            controls["nearest_count"].setRange(1, 64)
            controls["nearest_count"].setValue(int(options["nearest_count"]))
            controls["influence"] = _double(options["influence"], -100.0, 100.0)
            controls["max_distance"] = _double(options["max_distance"], 0.0, 100000.0)
            form.addRow("Nearest driver vertices:", controls["nearest_count"])
            form.addRow("Influence:", controls["influence"])
            form.addRow("Maximum distance (0 = unlimited):", controls["max_distance"])
            subset = QtWidgets.QLabel(
                "Make Live captures the driver baseline. Edit that live mesh, select the target, then run Wrap to bake its vertex deltas."
            )
            subset.setWordWrap(True)
            form.addRow("Workflow:", subset)
        elif key == "make_hole":
            controls["planarity_tolerance"] = _double(options["planarity_tolerance"], 0.0, 10.0, 6)
            controls["boundary_tolerance"] = _double(options["boundary_tolerance"], 0.0, 0.25, 7)
            form.addRow("Planarity tolerance:", controls["planarity_tolerance"])
            form.addRow("Boundary clearance:", controls["boundary_tolerance"])
            subset = QtWidgets.QLabel(
                "Pick the outer face first and a separate cutter face of the same mesh second. "
                "The cutter face is removed and its outline becomes the open border."
            )
            subset.setWordWrap(True)
            form.addRow("Pick order:", subset)
        elif key == "merge_components":
            controls["threshold"] = _double(options["threshold"], 0.0, 10000.0, 6)
            form.addRow("Merge distance:", controls["threshold"])
            subset = QtWidgets.QLabel(
                "Selected vertices inside the threshold merge to deterministic centroids. Exactly two selected "
                "border edges merge by their nearest endpoint pairing. UV, lightmap, normal, and material seams "
                "remain valid Odyssey records; unsafe non-manifold results are refused."
            )
            subset.setWordWrap(True)
            form.addRow("KOTOR-safe merge:", subset)
        elif key == "insert_edge_loop":
            controls["position"] = _double(options["position"], 0.001, 0.999, 3)
            controls["position"].setSingleStep(0.05)
            form.addRow("Position along selected edge:", controls["position"])
            subset = QtWidgets.QLabel(
                "Select one edge on a connected Quad Draw strip. Map Studio follows the stored logical-quad "
                "provenance and inserts one complete loop as one undoable edit. Arbitrary stock KOTOR "
                "triangulation, stale provenance, branches, and ambiguous rings are refused instead of guessed."
            )
            subset.setWordWrap(True)
            form.addRow("Safe topology contract:", subset)
        elif key == "quad_draw":
            controls["planarity_tolerance"] = _double(options["planarity_tolerance"], 0.0, 100.0, 5)
            controls["surface_offset"] = _double(options["surface_offset"], -1000.0, 1000.0, 5)
            controls["auto_weld"] = QtWidgets.QCheckBox("Reuse nearby Quad Draw vertices", dialog)
            controls["auto_weld"].setChecked(bool(options["auto_weld"]))
            controls["weld_tolerance"] = _double(options["weld_tolerance"], 0.0, 10.0, 6)
            form.addRow("Planarity tolerance:", controls["planarity_tolerance"])
            form.addRow("Live-surface offset:", controls["surface_offset"])
            form.addRow("Auto weld:", controls["auto_weld"])
            form.addRow("Weld tolerance:", controls["weld_tolerance"])
            subset = QtWidgets.QLabel(
                "Make a reference surface Live, then click four perimeter points. The editable retopology surface "
                "stays separate from the reference and is stored as ordinary triangulated KOTOR geometry."
            )
            subset.setWordWrap(True)
            form.addRow("Workflow:", subset)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return True
        for name, control in controls.items():
            if isinstance(control, QtWidgets.QComboBox):
                options[name] = str(control.currentData() or "")
            elif isinstance(control, QtWidgets.QCheckBox):
                options[name] = bool(control.isChecked())
            elif isinstance(control, QtWidgets.QSpinBox):
                options[name] = int(control.value())
            elif isinstance(control, QtWidgets.QDoubleSpinBox):
                options[name] = float(control.value())
        self.statusBar().showMessage(f"{title} saved. Run the shelf command to bake the result.", 4500)
        return True

    def _map_studio_imported_surface_in_room_space(
        self,
        surface_room_resref: str,
        mesh_role: str,
        destination_room_resref: str,
    ) -> ImportedMeshSurface | None:
        """Resolve one imported surface in another authored room's local space."""

        source_room = self.controller.imported_mesh_room(surface_room_resref)
        destination_room = self.controller.imported_mesh_room(destination_room_resref)
        primitive = getattr(source_room, "primitive", None)
        if primitive is None or destination_room is None:
            return None
        surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
        if surface_index < 0:
            return None
        surface = primitive.surfaces[surface_index]
        source_position = tuple(float(value) for value in tuple(getattr(source_room, "position", (0.0, 0.0, 0.0)))[:3])
        destination_position = tuple(
            float(value) for value in tuple(getattr(destination_room, "position", (0.0, 0.0, 0.0)))[:3]
        )
        offset = tuple(source_position[axis] - destination_position[axis] for axis in range(3))
        if all(abs(value) <= 1.0e-12 for value in offset):
            return surface
        return replace(
            surface,
            name=f"{surface.name}@{surface_room_resref}_in_{destination_room_resref}",
            vertices=tuple(
                tuple(float(point[axis]) + offset[axis] for axis in range(3)) for point in surface.vertices
            ),
        )

    def _map_studio_selected_mesh_vertex_indices(
        self,
        room_resref: str,
        mesh_role: str,
        selection,
    ) -> tuple[int, ...]:
        """Expand selected vertices, edges, or faces to stable raw vertex IDs."""

        room_spec = self.controller.imported_mesh_room(room_resref)
        primitive = getattr(room_spec, "primitive", None)
        surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role) if primitive is not None else -1
        surface = primitive.surfaces[surface_index] if surface_index >= 0 else None
        selected: set[int] = set()
        for row in tuple(selection or ()):
            if str(row.get("room_resref") or "") != room_resref or str(row.get("mesh_role") or "") != mesh_role:
                continue
            component = str(row.get("component_type") or "")
            if component == "vertex":
                index = int(row.get("mesh_vertex_index", -1))
                if index >= 0:
                    selected.add(index)
            elif component == "edge":
                selected.update(
                    int(value) for value in tuple(row.get("mesh_edge_indices") or ())[:2] if int(value) >= 0
                )
            elif component == "face" and surface is not None:
                face_index = int(row.get("face_index", -1))
                if 0 <= face_index < len(surface.faces):
                    selected.update(int(value) for value in surface.faces[face_index])
        return tuple(sorted(selected))

    def _capture_map_studio_live_wrap_driver_baseline(self) -> bool:
        """Capture the Make Live driver state used by the static Wrap bake."""

        live_surface = tuple(getattr(self.viewport_panel, "_map_studio_live_surface", ()) or ())
        if len(live_surface) < 2:
            return False
        room, role = str(live_surface[0]), str(live_surface[1])
        surface = self._map_studio_imported_surface_in_room_space(room, role, room)
        if surface is None:
            return False
        self._map_studio_live_wrap_driver_state = (room, role, surface)
        return True

    def _cancel_map_studio_multi_cut_preview(self) -> None:
        """Restore the exact pre-cut resident mesh without touching KMAP."""

        self.viewport_panel.clear_component_mesh_preview()
        self._map_studio_multi_cut_preview = None

    def _evaluate_map_studio_multi_cut_preview(self, entries) -> dict[str, Any] | None:
        """Build one non-mutating two-anchor Multi-Cut preview from viewport hits."""

        anchors_in = [dict(value) for value in tuple(entries or ())]
        if len(anchors_in) != 2:
            self._cancel_map_studio_multi_cut_preview()
            return None
        room = str(anchors_in[0].get("room_resref") or "")
        role = str(anchors_in[0].get("mesh_role") or "")
        if (
            not room
            or not role
            or any(
                str(value.get("room_resref") or "") != room
                or str(value.get("mesh_role") or "") != role
                for value in anchors_in
            )
        ):
            self.statusBar().showMessage("Multi-Cut anchors must stay on one editable room surface.", 4500)
            self._cancel_map_studio_multi_cut_preview()
            return None
        room_spec = self.controller.imported_mesh_room(room)
        primitive = getattr(room_spec, "primitive", None)
        if not isinstance(primitive, ImportedMeshRoomPrimitive):
            self.statusBar().showMessage("Multi-Cut needs an editable imported polygon surface.", 4500)
            self._cancel_map_studio_multi_cut_preview()
            return None
        surface_index = imported_mesh_surface_index_for_role(primitive, role)
        if surface_index < 0:
            self.statusBar().showMessage("Multi-Cut could not resolve the selected polygon surface.", 4500)
            self._cancel_map_studio_multi_cut_preview()
            return None
        surface = primitive.surfaces[surface_index]
        options = self._map_studio_baked_modeling_options("multi_cut")
        settings = MultiCutSettings(
            coplanar_angle_degrees=float(options["coplanar_angle_degrees"]),
            plane_tolerance=float(options["plane_tolerance"]),
            boundary_tolerance=float(options["boundary_tolerance"]),
        )
        try:
            session = MultiCutSession.begin(primitive, role, settings=settings)
            stable_anchors = []
            for entry in anchors_in:
                point = self._map_studio_component_local_point(entry)
                if point is None:
                    raise ValueError("Multi-Cut could not resolve a stable room-local pointer hit.")
                anchor = anchor_from_surface_hit(
                    surface,
                    int(entry.get("face_index", -1)),
                    point,
                    snap_tolerance=float(options["snap_tolerance"]),
                    plane_tolerance=float(options["plane_tolerance"]),
                )
                stable_anchors.append(anchor)
                session = session.add_anchor(anchor)
            evaluation = session.preview()
            if not evaluation.ok:
                raise ValueError(evaluation.diagnostics[0] if evaluation.diagnostics else "Multi-Cut is invalid.")
            if not self._show_live_imported_surface(evaluation.primitive, room, role):
                raise ValueError("Multi-Cut could not patch the resident viewport surface.")
        except ValueError as exc:
            self.statusBar().showMessage(f"Multi-Cut preview: {exc}", 5500)
            self._cancel_map_studio_multi_cut_preview()
            return None
        state = {
            "room_resref": room,
            "mesh_role": role,
            "anchors": tuple(stable_anchors),
            "settings": settings,
            "evaluation": evaluation,
        }
        self._map_studio_multi_cut_preview = state
        self.statusBar().showMessage(
            f"Multi-Cut preview crosses {len(evaluation.affected_faces)} face(s). Press Enter to commit once.",
            4500,
        )
        return state

    def _handle_map_studio_multi_cut_gesture(self, payload: dict[str, Any]) -> None:
        """Handle preview/cancel/Enter phases for the persistent Multi-Cut context."""

        phase = str(payload.get("phase") or "preview").strip().lower()
        if phase == "cancel":
            self._cancel_map_studio_multi_cut_preview()
            return
        entries = tuple(payload.get("anchors") or ())
        if phase == "preview":
            self._evaluate_map_studio_multi_cut_preview(entries)
            return
        if phase != "commit":
            self.statusBar().showMessage(f"Unknown Multi-Cut gesture phase: {phase}", 3500)
            return
        state = getattr(self, "_map_studio_multi_cut_preview", None)
        if not isinstance(state, dict):
            state = self._evaluate_map_studio_multi_cut_preview(entries)
        if not isinstance(state, dict):
            return
        evaluation = state["evaluation"]
        ok, message = self.controller.commit_imported_mesh_multi_cut(
            room_resref=str(state["room_resref"]),
            mesh_role=str(state["mesh_role"]),
            anchors=tuple(state["anchors"]),
            settings=state["settings"],
            expected_source_fingerprint=str(evaluation.source_fingerprint),
            expected_result_fingerprint=str(evaluation.result_fingerprint),
        )
        self.statusBar().showMessage(message, 6000)
        self._log(f"Map Studio: {message}")
        if ok:
            self._refresh_map_studio_imported_mesh_change(
                message,
                str(state["room_resref"]),
                str(state["mesh_role"]),
            )
            self._map_studio_multi_cut_preview = None
        else:
            self._cancel_map_studio_multi_cut_preview()

    def _commit_map_studio_modeling_tool_gesture(self, tool_key: str, payload: object) -> None:
        """Commit one persistent-tool gesture as one controller-owned undo step."""

        key = str(tool_key or "").strip()
        values = dict(payload) if isinstance(payload, dict) else {}
        kwargs: dict[str, Any] = {}
        room = role = ""
        if key == "multi_cut":
            self._handle_map_studio_multi_cut_gesture(values)
            return
        elif key == "target_weld":
            source = dict(values.get("source") or {})
            target = dict(values.get("target") or {})
            room = str(source.get("room_resref") or "")
            role = str(source.get("mesh_role") or "")
            if room != str(target.get("room_resref") or ""):
                self.statusBar().showMessage("Target Weld requires source and target in the same KOTOR room.", 4500)
                return
            kwargs = {
                "room_resref": room,
                "op": "target_weld",
                "mesh_role": role,
                "face_index": int(source.get("face_index", -1)),
                "source_vertex_index": int(source.get("mesh_vertex_index", -1)),
                "target_vertex_index": int(target.get("mesh_vertex_index", -1)),
                "target_mesh_role": str(target.get("mesh_role") or role),
            }
        elif key == "connect_components":
            selection = [dict(value) for value in tuple(values.get("selection") or ())]
            vertices = [value for value in selection if int(value.get("mesh_vertex_index", -1)) >= 0]
            if len(vertices) != 2:
                self.statusBar().showMessage("Connect needs exactly two vertices in one editable polygon patch.", 4500)
                return
            room = str(vertices[0].get("room_resref") or "")
            role = str(vertices[0].get("mesh_role") or "")
            if any(str(value.get("room_resref") or "") != room or str(value.get("mesh_role") or "") != role for value in vertices):
                self.statusBar().showMessage("Connect vertices must belong to the same editable mesh.", 4500)
                return
            kwargs = {
                "room_resref": room,
                "op": "connect_vertices",
                "mesh_role": role,
                "face_index": int(vertices[0].get("face_index", -1)),
                "first_vertex_index": int(vertices[0].get("mesh_vertex_index", -1)),
                "second_vertex_index": int(vertices[1].get("mesh_vertex_index", -1)),
            }
        elif key == "make_hole":
            outer = dict(values.get("outer") or {})
            cutter = dict(values.get("cutter") or {})
            room = str(outer.get("room_resref") or "")
            role = str(outer.get("mesh_role") or "")
            if (
                not room
                or not role
                or room != str(cutter.get("room_resref") or "")
                or role != str(cutter.get("mesh_role") or "")
            ):
                self.statusBar().showMessage(
                    "Make Hole requires two faces of the same editable polygon object.", 5000
                )
                return
            outer_face = int(outer.get("face_index", -1))
            cutter_face = int(cutter.get("face_index", -1))
            if outer_face < 0 or cutter_face < 0 or outer_face == cutter_face:
                self.statusBar().showMessage(
                    "Make Hole needs two different faces: outer face first, cutter face second.", 5000
                )
                return
            options = self._map_studio_baked_modeling_options("make_hole")
            kwargs = {
                "room_resref": room,
                "op": "make_hole",
                "mesh_role": role,
                "face_index": outer_face,
                "cutter_face_index": cutter_face,
                "make_hole_planarity_tolerance": float(options["planarity_tolerance"]),
                "make_hole_boundary_tolerance": float(options["boundary_tolerance"]),
            }
        elif key == "quad_draw":
            live = tuple(values.get("live_surface") or ())
            entries = [dict(value) for value in tuple(values.get("point_entries") or ())]
            if len(live) < 2 or len(entries) != 4:
                self.statusBar().showMessage(
                    "Quad Draw needs four ordered points on a Make Live reference surface.", 5000
                )
                return
            live_room, live_role = str(live[0]), str(live[1])
            if any(
                str(entry.get("room_resref") or "") != live_room
                or str(entry.get("mesh_role") or "") != live_role
                for entry in entries
            ):
                self.statusBar().showMessage(
                    "Quad Draw points must all project onto the active Make Live surface.", 5000
                )
                return
            local_points = tuple(self._map_studio_component_local_point(entry) for entry in entries)
            if any(point is None for point in local_points):
                self.statusBar().showMessage(
                    "Quad Draw could not resolve stable room-local points on the live surface.", 5000
                )
                return
            room_spec = self.controller.imported_mesh_room(live_room)
            primitive = getattr(room_spec, "primitive", None)
            live_surface_index = (
                imported_mesh_surface_index_for_role(primitive, live_role) if primitive is not None else -1
            )
            if primitive is None or live_surface_index < 0:
                self.statusBar().showMessage("Quad Draw's Make Live surface is no longer available.", 5000)
                return
            target_state = tuple(getattr(self, "_map_studio_quad_draw_target_state", ()) or ())
            if (
                len(target_state) == 3
                and target_state[:2] == (live_room, live_role)
                and imported_mesh_surface_index_for_role(primitive, str(target_state[2])) >= 0
            ):
                target_role = str(target_state[2])
            else:
                target_role = imported_mesh_surface_role(len(primitive.surfaces))
            source_surface = primitive.surfaces[live_surface_index]
            source_face = int(entries[0].get("face_index", -1))
            source_material = (
                int(source_surface.face_mats[source_face])
                if source_surface.face_mats and 0 <= source_face < len(source_surface.face_mats)
                else 0
            )
            normal_hint = None
            if 0 <= source_face < len(source_surface.faces) and len(source_surface.normals) == len(source_surface.vertices):
                source_vertices = tuple(int(value) for value in source_surface.faces[source_face])
                averaged = tuple(
                    sum(float(source_surface.normals[index][axis]) for index in source_vertices) / len(source_vertices)
                    for axis in range(3)
                )
                if sum(value * value for value in averaged) > 1.0e-12:
                    normal_hint = averaged
            options = self._map_studio_baked_modeling_options("quad_draw")
            surface_offset = float(options["surface_offset"])
            if normal_hint is not None and abs(surface_offset) > 1.0e-12:
                length = sum(value * value for value in normal_hint) ** 0.5
                unit_normal = tuple(value / length for value in normal_hint)
                local_points = tuple(
                    tuple(float(point[axis]) + (unit_normal[axis] * surface_offset) for axis in range(3))
                    for point in local_points
                    if point is not None
                )
            room = live_room
            role = target_role
            kwargs = {
                "room_resref": room,
                "op": "quad_draw",
                "mesh_role": role,
                "face_index": -1,
                "quad_points": tuple(point for point in local_points if point is not None),
                "quad_material": source_material,
                "quad_texture": str(source_surface.texture or ""),
                "quad_lightmap": "",
                "quad_normal_hint": normal_hint,
                "quad_planarity_tolerance": float(options["planarity_tolerance"]),
                "quad_auto_weld": bool(options["auto_weld"]),
                "quad_weld_tolerance": float(options["weld_tolerance"]),
            }
        else:
            return
        ok, message = self.controller.apply_imported_mesh_room_component_op(**kwargs)
        self.statusBar().showMessage(message, 6500)
        self._log(f"Map Studio: {message}")
        if ok:
            if key == "quad_draw":
                self._map_studio_quad_draw_target_state = (live_room, live_role, role)
            self._refresh_map_studio_imported_mesh_change(message, room, role)

    def _apply_map_studio_component_shelf_action(self, action_key: str) -> bool:
        """Prefer genuine imported-mesh component operations over floor-plan fallbacks."""

        key = str(action_key or "").strip()
        selection = list(self.viewport_panel.map_studio_component_selection() or ())
        if not selection:
            return False
        room = str(selection[0].get("room_resref") or "")
        role = str(selection[0].get("mesh_role") or "")
        if not room or not role:
            return False
        if any(str(row.get("room_resref") or "") != room for row in selection):
            self.statusBar().showMessage(
                f"{key.replace('_', ' ').title()} cannot mix components from different KOTOR rooms.", 5000
            )
            return True
        same_surface = all(str(row.get("mesh_role") or "") == role for row in selection)
        if not same_surface and key not in {"bridge", "boolean_a_minus_b"}:
            self.statusBar().showMessage(
                f"{key.replace('_', ' ').title()} needs components from one editable mesh surface.", 5000
            )
            return True
        kwargs: dict[str, Any] | None = None
        if key == "fill_hole":
            loop = self._ordered_map_studio_boundary_loop(
                row.get("mesh_edge_indices") or () for row in selection if str(row.get("component_type") or "") == "edge"
            )
            if loop:
                kwargs = {"op": "fill_boundary_loop", "face_index": -1, "loop_vertex_indices": loop}
        elif key in {"soften_edges", "harden_edges"}:
            edges = tuple(
                tuple(int(value) for value in tuple(row.get("mesh_edge_indices") or ())[:2])
                for row in selection if str(row.get("component_type") or "") == "edge"
            )
            faces = tuple(
                int(row.get("face_index", -1))
                for row in selection if str(row.get("component_type") or "") == "face" and int(row.get("face_index", -1)) >= 0
            )
            if edges:
                kwargs = {"op": key, "face_index": -1, "edge_vertex_indices": edges}
            elif faces:
                kwargs = {"op": "soften_faces" if key == "soften_edges" else "harden_faces", "face_index": faces[0], "face_indices": faces}
        elif key == "reverse_normals":
            faces = tuple(
                int(row.get("face_index", -1))
                for row in selection if str(row.get("component_type") or "") == "face" and int(row.get("face_index", -1)) >= 0
            )
            if faces:
                kwargs = {"op": "face_flip", "face_index": faces[0], "face_indices": faces}
        elif key == "insert_edge_loop":
            edge = next((row for row in selection if str(row.get("component_type") or "") == "edge"), None)
            if edge is not None:
                raw_edge = tuple(int(value) for value in tuple(edge.get("mesh_edge_indices") or ())[:2])
                if len(raw_edge) == 2 and raw_edge[0] != raw_edge[1]:
                    options = self._map_studio_baked_modeling_options(key)
                    kwargs = {
                        "op": "insert_edge_loop",
                        "face_index": int(edge.get("face_index", -1)),
                        "loop_edge_vertices": raw_edge,
                        "loop_position": float(options["position"]),
                    }
        elif key == "merge_components":
            vertices = [
                row for row in selection
                if str(row.get("component_type") or "") == "vertex"
                and int(row.get("mesh_vertex_index", -1)) >= 0
            ]
            edges = [row for row in selection if str(row.get("component_type") or "") == "edge"]
            options = self._map_studio_baked_modeling_options(key)
            if len(vertices) >= 2 and not edges:
                kwargs = {
                    "op": "merge_components",
                    "face_index": int(vertices[0].get("face_index", -1)),
                    "merge_vertex_indices": tuple(
                        sorted({int(row.get("mesh_vertex_index", -1)) for row in vertices})
                    ),
                    "merge_threshold": float(options["threshold"]),
                }
            elif len(edges) == 2 and not vertices:
                edge_indices = tuple(
                    tuple(int(value) for value in tuple(row.get("mesh_edge_indices") or ())[:2])
                    for row in edges
                )
                if all(len(edge) == 2 and edge[0] != edge[1] for edge in edge_indices):
                    kwargs = {
                        "op": "merge_components",
                        "face_index": int(edges[0].get("face_index", -1)),
                        "merge_edge_vertex_indices": edge_indices,
                        "merge_threshold": float(options["threshold"]),
                    }
        elif key == "mirror":
            options = self._map_studio_baked_modeling_options(key)
            kwargs = {
                "op": "mirror_geometry",
                "face_index": -1,
                "mirror_axis": str(options["axis"]),
                "mirror_center": float(options["center"]),
                "mirror_duplicate": bool(options["duplicate"]),
                "mirror_merge_seam_tolerance": float(options["merge_tolerance"]),
            }
        elif key == "bridge":
            edges = [row for row in selection if str(row.get("component_type") or "") == "edge"]
            if len(edges) == 2:
                first_edge = tuple(int(value) for value in tuple(edges[0].get("mesh_edge_indices") or ())[:2])
                second_edge = tuple(int(value) for value in tuple(edges[1].get("mesh_edge_indices") or ())[:2])
                if len(first_edge) == 2 and len(second_edge) == 2:
                    options = self._map_studio_baked_modeling_options(key)
                    role = str(edges[0].get("mesh_role") or role)
                    kwargs = {
                        "op": "bridge_border_edges",
                        "face_index": int(edges[0].get("face_index", -1)),
                        "first_edge_vertices": first_edge,
                        "second_edge_vertices": second_edge,
                        "target_mesh_role": str(edges[1].get("mesh_role") or role),
                        "bridge_divisions": int(options["divisions"]),
                        "bridge_taper": float(options["taper"]),
                        "bridge_twist_degrees": float(options["twist_degrees"]),
                        "bridge_smooth": bool(options["smooth"]),
                    }
        elif key == "boolean_a_minus_b":
            ordered_roles: list[str] = []
            for row in selection:
                selected_role = str(row.get("mesh_role") or "")
                if selected_role and selected_role not in ordered_roles:
                    ordered_roles.append(selected_role)
            if len(ordered_roles) == 2:
                options = self._map_studio_baked_modeling_options(key)
                role = ordered_roles[0]
                kwargs = {
                    "op": "boolean_difference_closed_solids",
                    "face_index": -1,
                    "boolean_cutter_mesh_role": ordered_roles[1],
                    "boolean_weld_tolerance": float(options["weld_tolerance"]),
                }
        elif key == "bend_tool":
            options = self._map_studio_baked_modeling_options(key)
            kwargs = {
                "op": "bend_vertices",
                "face_index": -1,
                "deform_vertex_indices": self._map_studio_selected_mesh_vertex_indices(room, role, selection),
                "deform_axis": str(options["axis"]),
                "curvature_degrees": float(options["curvature_degrees"]),
            }
        elif key == "lattice":
            options = self._map_studio_baked_modeling_options(key)
            axis_index = {"x": 0, "y": 1, "z": 2}.get(str(options["offset_axis"]), 2)
            displacement = [0.0, 0.0, 0.0]
            displacement[axis_index] = float(options["upper_layer_offset"])
            kwargs = {
                "op": "lattice_deform",
                "face_index": -1,
                "deform_vertex_indices": self._map_studio_selected_mesh_vertex_indices(room, role, selection),
                "lattice_control_deltas": ((0.0, 0.0, 0.0),) * 4 + (tuple(displacement),) * 4,
            }
        elif key == "shrink_wrap":
            live = tuple(getattr(self.viewport_panel, "_map_studio_live_surface", ()) or ())
            if len(live) >= 2:
                if (room, role) == (str(live[0]), str(live[1])):
                    self.statusBar().showMessage(
                        "ShrinkWrap target and source are the same live surface; select a different mesh to project.",
                        5000,
                    )
                    return True
                options = self._map_studio_baked_modeling_options(key)
                target = self._map_studio_imported_surface_in_room_space(str(live[0]), str(live[1]), room)
                if target is not None:
                    kwargs = {
                        "op": "shrink_wrap",
                        "face_index": -1,
                        "deform_vertex_indices": self._map_studio_selected_mesh_vertex_indices(room, role, selection),
                        "shrink_target_surface": target,
                        "shrink_projection": str(options["projection"]),
                        "shrink_offset": float(options["offset"]),
                        "shrink_align_normals": bool(options["align_normals"]),
                    }
        elif key == "wrap":
            live = tuple(getattr(self.viewport_panel, "_map_studio_live_surface", ()) or ())
            baseline = getattr(self, "_map_studio_live_wrap_driver_state", None)
            if len(live) < 2:
                self.statusBar().showMessage("Wrap needs a Make Live driver surface first.", 4500)
                return True
            if (room, role) == (str(live[0]), str(live[1])):
                self.statusBar().showMessage(
                    "Wrap driver and target are the same surface; select a different target mesh.", 5000
                )
                return True
            if not baseline or tuple(baseline[:2]) != tuple(live[:2]):
                if self._capture_map_studio_live_wrap_driver_baseline():
                    self.statusBar().showMessage(
                        "Captured the live Wrap driver baseline. Edit that driver, select the target, then run Wrap again.",
                        6000,
                    )
                else:
                    self.statusBar().showMessage("Could not capture the live Wrap driver surface.", 4500)
                return True
            live_room, live_role, driver_base_local = baseline
            driver_current_local = self._map_studio_imported_surface_in_room_space(live_room, live_role, live_room)
            if driver_current_local is None:
                self.statusBar().showMessage("The live Wrap driver is no longer available.", 4500)
                return True
            if tuple(driver_base_local.vertices) == tuple(driver_current_local.vertices):
                self.statusBar().showMessage(
                    "The live Wrap driver has not changed since Make Live; edit it before baking Wrap.", 5000
                )
                return True
            driver_deformed = self._map_studio_imported_surface_in_room_space(live_room, live_role, room)
            if driver_deformed is not None:
                if driver_current_local.vertices and driver_deformed.vertices:
                    room_offset = tuple(
                        float(driver_deformed.vertices[0][axis]) - float(driver_current_local.vertices[0][axis])
                        for axis in range(3)
                    )
                else:
                    room_offset = (0.0, 0.0, 0.0)
                driver_base = replace(
                    driver_base_local,
                    name=f"{driver_base_local.name}@make_live_baseline",
                    vertices=tuple(
                        tuple(float(point[axis]) + room_offset[axis] for axis in range(3))
                        for point in driver_base_local.vertices
                    ),
                )
                options = self._map_studio_baked_modeling_options(key)
                kwargs = {
                    "op": "wrap_deform",
                    "face_index": -1,
                    "deform_vertex_indices": self._map_studio_selected_mesh_vertex_indices(room, role, selection),
                    "wrap_driver_base": driver_base,
                    "wrap_driver_deformed": driver_deformed,
                    "wrap_nearest_count": int(options["nearest_count"]),
                    "wrap_influence": float(options["influence"]),
                    "wrap_max_distance": float(options["max_distance"]),
                }
        if kwargs is None:
            if key == "insert_edge_loop":
                self.statusBar().showMessage(
                    "Insert Edge Loop needs one selected edge on a provenance-safe Quad Draw strip.", 5000
                )
                return True
            if key == "bridge":
                self.statusBar().showMessage("Bridge needs exactly two selected border edges in one KOTOR room.", 4500)
                return True
            if key == "boolean_a_minus_b":
                self.statusBar().showMessage(
                    "Difference A - B needs exactly two closed surfaces in one room, selected in A-then-B order.", 5500
                )
                return True
            if key == "shrink_wrap":
                self.statusBar().showMessage("ShrinkWrap needs a Make Live target surface.", 4500)
                return True
            diagnostics = {
                "fill_hole": "Fill Hole needs one complete closed border-edge loop.",
                "soften_edges": "Soften Edge needs selected edges or faces on one editable surface.",
                "harden_edges": "Harden Edge needs selected edges or faces on one editable surface.",
                "reverse_normals": "Reverse needs one or more selected faces on one editable surface.",
                "merge_components": (
                    "Merge needs at least two selected vertices or exactly two selected border edges "
                    "on one editable surface."
                ),
                "bend_tool": "Bend needs selected vertices, edges, faces, or one editable surface.",
                "lattice": "Lattice needs selected vertices, edges, faces, or one editable surface.",
                "wrap": "Wrap needs a Make Live driver and a different selected target surface.",
            }
            self.statusBar().showMessage(
                diagnostics.get(key, f"{key.replace('_', ' ').title()} cannot use the current component selection."),
                5500,
            )
            return True
        ok, message = self.controller.apply_imported_mesh_room_component_op(
            room_resref=room, mesh_role=role, **kwargs
        )
        self.statusBar().showMessage(message, 6500)
        self._log(f"Map Studio: {message}")
        if ok:
            self.viewport_panel.clear_map_studio_component_selection()
            self._refresh_map_studio_imported_mesh_change(message, room, role)
        return True

    def _component_modeling_amount(
        self,
        action_key: str,
        title: str,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
    ) -> tuple[float, bool]:
        """Component-modeling amount flow: apply immediately with the remembered
        per-action amount; hold Ctrl while picking the action to type a value."""
        amounts = getattr(self, "_component_modeling_last_amounts", None)
        if amounts is None:
            amounts = {}
            self._component_modeling_last_amounts = amounts
        remembered = float(amounts.get(action_key, default))
        ctrl_held = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier)
        if not ctrl_held:
            return remembered, True
        value, accepted = QtWidgets.QInputDialog.getDouble(
            self, title, label, remembered, minimum, maximum, 2
        )
        if accepted:
            amounts[action_key] = float(value)
        return float(value), accepted

    def _apply_component_modeling_imported_face_action(self, action_key: str, target: str) -> bool:
        """Route Delete / Set Texture on a hovered face to imported-mesh geometry.

        Hovering a read-only stock room converts it to editable imported
        geometry first (one undoable command each), so any loaded game map
        can be customized directly.  Returns False when the hover target is
        not face geometry this path owns (parametric rooms keep their
        tool-belt routing).
        """

        context = self.viewport_panel.current_map_studio_hover_context()
        component_type = str(getattr(context, "component_type", "") or "") if context is not None else ""
        wanted_component = action_key.split("_", 1)[0]
        if component_type != wanted_component or component_type not in {"face", "edge", "vertex"}:
            return False
        room_resref = str(getattr(context, "room_resref", "") or "").strip().lower()
        mesh_role = str(getattr(context, "mesh_role", "") or "")
        face_index = int(getattr(context, "face_index", -1))
        if not room_resref or face_index < 0:
            return False

        if mesh_role.startswith("stock_room"):
            ok, message = self.controller.convert_stock_room_to_imported_mesh(
                room_resref=room_resref,
                resource_manager=getattr(self, "resource_manager", None),
            )
            self.statusBar().showMessage(message, 6000)
            self._log(f"Map Studio: {message}")
            if not ok:
                return True
            suffix = mesh_role.rsplit("_", 1)[-1]
            surface_index = int(suffix) if suffix.isdigit() else 0
            mesh_role = imported_mesh_surface_role(surface_index)
        elif self.controller.imported_mesh_room(room_resref) is None:
            return False

        face_indices: list[int] = [face_index]
        room_spec = self.controller.imported_mesh_room(room_resref)
        if room_spec is not None:
            try:
                face_indices = list(
                    resolve_imported_mesh_face_target(room_spec.primitive, mesh_role, face_index, target)
                )
            except (IndexError, ValueError) as exc:
                self.statusBar().showMessage(f"Cannot resolve {target}: {exc}", 5000)
                return True

        shift_held = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        if wanted_component in {"vertex", "edge"} or action_key in {"face_flat", "face_flip", "face_split"}:
            component_kwargs: dict = {
                "room_resref": room_resref,
                "op": action_key,
                "mesh_role": mesh_role,
                "face_index": face_index,
            }
            if wanted_component == "vertex":
                component_kwargs["vertex_corner"] = int(getattr(context, "vertex_index", 0))
            elif wanted_component == "edge":
                component_kwargs["edge_corners"] = tuple(getattr(context, "edge_indices", (0, 1)))
            if action_key in {"vertex_move", "edge_move"}:
                distance, accepted = self._component_modeling_amount(
                    action_key,
                    "Move Along Face Normal",
                    "Distance (meters, negative = inward):",
                    0.5,
                    -100.0,
                    100.0,
                )
                if not accepted:
                    self.statusBar().showMessage("Move cancelled.", 3000)
                    return True
                normal = tuple(float(v) for v in tuple(getattr(context, "face_normal", (0.0, 0.0, 1.0)))[:3])
                component_kwargs["delta"] = (
                    normal[0] * float(distance),
                    normal[1] * float(distance),
                    normal[2] * float(distance),
                )
            elif action_key == "vertex_weld":
                max_distance, accepted = self._component_modeling_amount(
                    action_key, "Weld Vertex", "Maximum snap distance (meters):", 0.5, 0.01, 25.0
                )
                if not accepted:
                    self.statusBar().showMessage("Weld cancelled.", 3000)
                    return True
                component_kwargs["max_distance"] = float(max_distance)
            elif action_key == "edge_bevel":
                amount, accepted = self._component_modeling_amount(
                    action_key,
                    "Bevel Edge",
                    "Chamfer width (meters):",
                    0.25,
                    0.001,
                    25.0,
                )
                if not accepted:
                    self.statusBar().showMessage("Bevel cancelled.", 3000)
                    return True
                component_kwargs["amount"] = float(amount)
            elif action_key in {"face_flat", "face_flip"}:
                component_kwargs["face_indices"] = tuple(face_indices)
            ok, message = self.controller.apply_imported_mesh_room_component_op(**component_kwargs)
            self.statusBar().showMessage(message, 6000)
            self._log(f"Map Studio: {message}")
            if ok:
                self._refresh_map_studio_imported_mesh_change(message, room_resref, mesh_role)
            return True
        if action_key == "face_delete":
            ok, message = self.controller.delete_imported_mesh_room_faces(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
            )
        elif action_key == "face_extrude":
            distance, accepted = self._component_modeling_amount(
                action_key, "Extrude Faces", "Extrude distance (meters, negative = inward):", 2.0, -100.0, 100.0
            )
            if not accepted:
                self.statusBar().showMessage("Extrude cancelled.", 3000)
                return True
            ok, message = self.controller.extrude_imported_mesh_room_faces(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
                distance=float(distance),
                point_normal=shift_held,
            )
        elif action_key == "face_inset":
            inset, accepted = self._component_modeling_amount(
                action_key, "Inset Faces", "Inset amount (meters):", 0.5, 0.01, 50.0
            )
            if not accepted:
                self.statusBar().showMessage("Inset cancelled.", 3000)
                return True
            ok, message = self.controller.inset_imported_mesh_room_faces(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
                inset=float(inset),
            )
        elif action_key == "face_move":
            distance, accepted = self._component_modeling_amount(
                action_key, "Move Faces", "Move along face normal (meters, negative = inward):", 0.5, -100.0, 100.0
            )
            if not accepted:
                self.statusBar().showMessage("Move cancelled.", 3000)
                return True
            normal = tuple(float(v) for v in tuple(getattr(context, "face_normal", (0.0, 0.0, 1.0)))[:3])
            delta = (normal[0] * float(distance), normal[1] * float(distance), normal[2] * float(distance))
            ok, message = self.controller.move_imported_mesh_room_faces(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
                delta=delta,
            )
        else:
            texture = MapStudioTextureBrowserDialog.pick_texture(
                getattr(self, "resource_manager", None),
                self,
                project=self.project,
                game=str(getattr(self.project, "game", "") or ""),
            )
            if not texture:
                self.statusBar().showMessage("Set Texture cancelled.", 3000)
                return True
            ok, message = self.controller.set_imported_mesh_room_face_texture(
                room_resref=room_resref,
                mesh_role=mesh_role,
                face_indices=face_indices,
                texture=texture,
            )
        self.statusBar().showMessage(message, 6000)
        self._log(f"Map Studio: {message}")
        if ok:
            if action_key == "face_set_texture":
                # Material-region edits may split one surface into several;
                # that changes render-node ownership and requires the safe
                # full preview composition path.
                self._refresh_map_studio_geometry_change(message)
            else:
                self._refresh_map_studio_imported_mesh_change(message, room_resref, mesh_role)
        return True

    def _run_map_studio_mode_marking_action(self, mode_key: str) -> None:
        key = str(mode_key or "object").strip().lower()
        if key == "select":
            action = self._map_studio_tool_action_for_key("select")
            if action is not None:
                self._handle_map_studio_tool_belt_action(action)
            else:
                self.select_map_studio_authored_context()
            return
        label_by_key = {
            "object": "Object",
            "edit": "Multi-Component",
            "terrain": "Terrain",
            "placement": "Placement",
            "vertex": "Vertex",
            "edge": "Edge",
            "face": "Face",
        }
        label = label_by_key.get(key, "Object")
        self._set_map_studio_toolbar_edit_mode(label)
        self._handle_map_studio_edit_mode_changed(label)
        self.select_map_studio_authored_context()

    def _map_studio_tool_action_for_key(self, key: str):
        action_key = str(key or "").strip()
        if not action_key:
            return None
        if not self._map_studio_tool_action_index:
            self._refresh_map_studio_tool_index()
        action = self._map_studio_tool_action_index.get(action_key)
        if action is not None:
            return action
        for candidate in self.controller.available_map_studio_tool_belt_actions():
            if str(getattr(candidate, "key", "") or "") == action_key:
                self._map_studio_tool_action_index[action_key] = candidate
                return candidate
        return None

    def _make_map_studio_marking_tool_action(
        self,
        key: str,
        *,
        label: str = "",
        object_prefix: str = "mapStudioToolMarkingAction",
    ) -> QtGui.QAction:
        action = self._map_studio_tool_action_for_key(key)
        if action is None:
            qaction = QtGui.QAction(label or key, self)
            qaction.setObjectName(f"{object_prefix}_{key}")
            qaction.setEnabled(False)
            return qaction
        qaction = self._build_map_studio_tool_qaction(action, context_menu=True)
        qaction.setObjectName(f"{object_prefix}_{key}")
        if label:
            qaction.setText(label)
        return qaction

    def _add_map_studio_marking_tool_action(
        self,
        menu: QtWidgets.QMenu,
        key: str,
        *,
        label: str = "",
        object_prefix: str = "mapStudioToolMarkingAction",
    ) -> QtGui.QAction:
        qaction = self._make_map_studio_marking_tool_action(key, label=label, object_prefix=object_prefix)
        menu.addAction(qaction)
        return qaction

    def _add_map_studio_planned_marking_action(self, menu: QtWidgets.QMenu, key: str, label: str) -> QtGui.QAction:
        qaction = QtGui.QAction(label, menu)
        qaction.setObjectName(f"mapStudioToolMarkingPlannedAction_{key}")
        qaction.setEnabled(False)
        qaction.setToolTip("Planned Map Studio tool; not yet backed by an authored KMAP command.")
        menu.addAction(qaction)
        return qaction

    def _build_map_studio_tool_marking_menu(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QMenu:
        """Build the Shift+right-click Maya-style Map Studio tool marking menu."""

        menu = QtWidgets.QMenu(parent or self)
        menu.setObjectName("mapStudioToolMarkingMenu")
        quick_frame = QtWidgets.QFrame(menu)
        quick_frame.setObjectName("mapStudioToolMarkingQuickRadial")
        quick_layout = QtWidgets.QGridLayout(quick_frame)
        quick_layout.setContentsMargins(8, 8, 8, 8)
        quick_layout.setHorizontalSpacing(6)
        quick_layout.setVerticalSpacing(6)
        quick_actions = (
            ("extrude", "Extrude", 0, 1, "mapStudioToolMarkingQuickAction_extrude", "mapStudioToolMarkingQuickButton_extrude"),
            ("bridge", "Bridge", 1, 0, "mapStudioToolMarkingQuickAction_bridge", "mapStudioToolMarkingQuickButton_bridge"),
            ("cut", "Cut / Multi-Cut", 1, 1, "mapStudioToolMarkingQuickAction_cut", "mapStudioToolMarkingQuickButton_cut"),
            ("weld", "Weld / Merge", 1, 2, "mapStudioToolMarkingQuickAction_weld", "mapStudioToolMarkingQuickButton_weld"),
            ("fill_hole", "Fill Hole", 2, 0, "mapStudioToolMarkingQuickAction_fill_hole", "mapStudioToolMarkingQuickButton_fill_hole"),
            ("bevel", "Bevel / Inset", 2, 2, "mapStudioToolMarkingQuickAction_bevel", "mapStudioToolMarkingQuickButton_bevel"),
        )
        for key, label, row, column, action_name, button_name in quick_actions:
            action = self._make_map_studio_marking_tool_action(
                key,
                label=label,
                object_prefix="mapStudioToolMarkingQuickAction",
            )
            action.setObjectName(action_name)
            action.triggered.connect(menu.close)
            button = QtWidgets.QToolButton(quick_frame)
            button.setObjectName(button_name)
            button.setDefaultAction(action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            button.setMinimumWidth(104)
            quick_layout.addWidget(button, row, column)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setObjectName("mapStudioToolMarkingQuickWidgetAction")
        widget_action.setDefaultWidget(quick_frame)
        menu.addAction(widget_action)
        menu.addSection("Polygon Tools")
        for key, label in (
            ("insert_edge_loop", "Insert Edge Loop"),
            ("cut_slice_insert_edges", "Slice"),
            ("triangulate", "Triangulate"),
            ("cleanup", "Cleanup"),
            ("soften_edges", "Soften Edges"),
            ("harden_edges", "Harden Edges"),
            ("reverse_normals", "Reverse Normals"),
            ("mirror", "Mirror"),
            ("separate", "Separate Shells"),
            ("combine", "Combine Meshes"),
            ("texture_paint", "Texture Paint"),
            ("paint_material", "Assign Material Intent"),
            ("paint_wok", "Paint WOK Surface"),
            ("validate", "Validate Selection"),
        ):
            self._add_map_studio_marking_tool_action(menu, key, label=label)
        boolean_menu = menu.addMenu("Boolean")
        boolean_menu.setObjectName("mapStudioToolMarkingBooleanMenu")
        for key, label in (
            ("boolean", "Boolean Tool"),
            ("boolean_a_minus_b", "A - B"),
            ("boolean_b_minus_a", "B - A"),
        ):
            self._add_map_studio_marking_tool_action(boolean_menu, key, label=label)
        terrain_menu = menu.addMenu("Terrain Brushes")
        terrain_menu.setObjectName("mapStudioToolMarkingTerrainBrushesMenu")
        for key, label in (
            ("sculpt_raise", "Raise"),
            ("sculpt_lower", "Lower"),
            ("sculpt_smooth", "Smooth"),
            ("sculpt_flatten", "Flatten"),
            ("sculpt_erase", "Erase / Reset"),
            ("sculpt_plateau", "Plateau"),
            ("sculpt_ramp", "Ramp"),
            ("sculpt_slope", "Slope"),
            ("sculpt_terrace", "Terrace"),
            ("sculpt_pinch", "Pinch"),
            ("sculpt_erode", "Erode"),
            ("sculpt_noise", "Noise"),
        ):
            self._add_map_studio_marking_tool_action(terrain_menu, key, label=label)
        uv_menu = menu.addMenu("UV / Mapping")
        uv_menu.setObjectName("mapStudioToolMarkingUvMappingMenu")
        self._add_map_studio_marking_tool_action(uv_menu, "texture_paint", label="Texture Paint / Assign Target")
        self._add_map_studio_marking_tool_action(uv_menu, "paint_material", label="Assign Material Intent")
        # Roadmap items live in the audit brief and the component-modeling panel's
        # dimmed actions — a "Planned / Missing" menu is UI noise, not a tool.
        return menu

    def _open_map_studio_tool_marking_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = self._build_map_studio_tool_marking_menu(self.viewport_panel)
        menu.exec(global_pos)

    def _focus_map_studio_command_search(self) -> None:
        """Focus the command-search field without changing the active Map Studio workspace."""

        combo = getattr(self, "map_studio_command_search_combo", None)
        if combo is None:
            return
        combo.setFocus()
        line_edit = combo.lineEdit()
        if line_edit is not None:
            line_edit.selectAll()
        self.statusBar().showMessage("Map Studio command search focused. Type a tool name and press Run.", 4000)

    def _run_selected_map_studio_command_search(self) -> None:
        """Run the selected/typed command-search action through the shared dispatcher."""

        combo = getattr(self, "map_studio_command_search_combo", None)
        if combo is None:
            return
        key = str(combo.currentData() or "").strip()
        if not key:
            query = combo.currentText().strip()
            matches = self.controller.map_studio_tool_command_search(query, limit=1)
            if matches:
                key = matches[0].key
        action = self._map_studio_tool_action_index.get(key)
        if action is None:
            self._log("Choose a Map Studio command from the command search before running it.")
            self.statusBar().showMessage("Choose a Map Studio command before running it.", 4000)
            return
        self._handle_map_studio_tool_belt_action(action)

    def _add_selected_map_studio_custom_tool(self) -> None:
        """Add the currently selected indexed tool to the custom Map Studio belt."""

        key = str(self.map_studio_custom_tool_combo.currentData() or "").strip()
        if not key:
            text = self.map_studio_custom_tool_combo.currentText().strip().lower()
            for candidate_key, action in self._map_studio_tool_action_index.items():
                label = str(getattr(action, "label", candidate_key) or candidate_key).lower()
                if text == candidate_key.lower() or text in label:
                    key = candidate_key
                    break
        if not key:
            self._log("Choose a Map Studio tool from the indexed custom tool list before adding it.")
            return
        if key not in self._map_studio_custom_belt_keys:
            self._map_studio_custom_belt_keys = (*self._map_studio_custom_belt_keys, key)
        custom_index = self.map_studio_tool_belt_preset_combo.findData("custom")
        if custom_index >= 0:
            previous = self._syncing_map_studio_tool_belt_preferences
            self._syncing_map_studio_tool_belt_preferences = True
            try:
                self.map_studio_tool_belt_preset_combo.setCurrentIndex(custom_index)
            finally:
                self._syncing_map_studio_tool_belt_preferences = previous
        self._persist_map_studio_tool_belt_preferences()
        self._refresh_map_studio_tool_belt()
        action = self._map_studio_tool_action_index.get(key)
        label = str(getattr(action, "label", key) if action is not None else key)
        self._log(f"Added {label} to the custom Map Studio tool belt.")

    def _customize_map_studio_tool_belt(self) -> None:
        all_actions = self.controller.available_map_studio_tool_belt_actions()
        selected_keys = self._map_studio_custom_belt_keys
        if not selected_keys:
            selected_keys = tuple(
                str(getattr(action, "key", "") or "") for action in all_actions if bool(getattr(action, "implemented", False))
            )
        dialog = _MapStudioToolBeltCustomizeDialog(
            self,
            actions=all_actions,
            selected_keys=selected_keys,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._map_studio_custom_belt_keys = dialog.selected_action_keys()
        custom_index = self.map_studio_tool_belt_preset_combo.findData("custom")
        if custom_index >= 0:
            self._syncing_map_studio_tool_belt_preferences = True
            try:
                self.map_studio_tool_belt_preset_combo.setCurrentIndex(custom_index)
            finally:
                self._syncing_map_studio_tool_belt_preferences = False
        self._persist_map_studio_tool_belt_preferences()
        self._refresh_map_studio_tool_belt()
        self._log("Map Studio custom tool belt saved in this KMAP.")

    def _map_studio_belt_primitive_kind(self, action_key: str) -> str:
        """Map direct shelf buttons to authored composition primitive kinds."""

        return {
            "plane": "plane",
            "cube": "cube",
            "wall": "wall",
            "ramp": "ramp",
            "stairs": "stairs",
            "cylinder": "cylinder",
            "sphere": "sphere",
            "cone": "cone",
            "torus": "torus",
            "door_frame": "door_frame",
            "arch": "arch",
        }.get(str(action_key or "").strip(), "")

    def _map_studio_belt_placement_kind(self, action_key: str) -> str:
        """Map direct shelf buttons to authored KOTOR placement kinds."""

        return {
            "placeable": "placeable",
            "creature": "creature",
            "door": "door",
            "waypoint": "waypoint",
            "trigger": "trigger",
            "encounter": "encounter",
            "sound": "sound",
            "camera": "camera",
            "store": "store",
        }.get(str(action_key or "").strip(), "")

    def _map_studio_belt_terrain_brush(self, action_key: str) -> str:
        """Map direct shelf buttons to terrain sculpt brush keys."""

        return {
            "sculpt_raise": "raise",
            "sculpt_lower": "lower",
            "sculpt_smooth": "smooth",
            "sculpt_flatten": "flatten",
            "sculpt_erase": "erase",
            "sculpt_plateau": "plateau",
            "sculpt_ramp": "ramp",
            "sculpt_slope": "slope",
            "sculpt_terrace": "terrace",
            "sculpt_pinch": "pinch",
            "sculpt_erode": "erode",
            "sculpt_noise": "noise",
        }.get(str(action_key or "").strip(), "")

    def _map_studio_combo_data(self, combo_name: str) -> dict[str, Any]:
        """Return the current data dictionary for one Builder combo."""

        combo = getattr(self.builder_tab, combo_name, None)
        if combo is None:
            return {}
        data = combo.currentData()
        if isinstance(data, dict):
            return dict(data)
        if isinstance(data, str):
            return {"room_resref": data}
        return {}

    def _map_studio_current_room_resref(self) -> str:
        """Return the best current authored room context visible in Builder."""

        for combo_name in (
            "floorPlanVertexRoomComboBox",
            "floorPlanExtrusionRoomComboBox",
            "floorPlanOpeningRoomComboBox",
            "floorPlanOpeningMarkerRoomComboBox",
            "floorPlanUnionFirstRoomComboBox",
            "floorPlanBridgeFirstRoomComboBox",
            "roomPrimitiveTransformComboBox",
        ):
            data = self._map_studio_combo_data(combo_name)
            room = str(data.get("room_resref") or "").strip()
            if room:
                return room
        choices = self.controller.authored_floor_plan_room_choices()
        if choices:
            return str(getattr(choices[0], "room_resref", "") or "")
        return ""

    def _map_studio_selected_point_indices(self) -> tuple[int, ...]:
        """Return selected Builder point indices without owning selection policy."""

        parser = getattr(self.builder_tab, "_parse_floor_plan_point_indices", None)
        if callable(parser):
            try:
                return tuple(int(index) for index in tuple(parser() or ()))
            except Exception:
                return ()
        return ()

    def _map_studio_tool_action_context(self, action_key: str) -> MapStudioToolActionContext:
        """Collect current UI selection facts for the core action dispatcher."""

        vertex_data = self._map_studio_combo_data("floorPlanVertexRoomComboBox")
        vertex_target_data = self._map_studio_combo_data("floorPlanVertexTargetRoomComboBox")
        bridge_first = self._map_studio_combo_data("floorPlanBridgeFirstRoomComboBox")
        bridge_second = self._map_studio_combo_data("floorPlanBridgeSecondRoomComboBox")
        union_first = self._map_studio_combo_data("floorPlanUnionFirstRoomComboBox")
        union_second = self._map_studio_combo_data("floorPlanUnionSecondRoomComboBox")
        opening_data = self._map_studio_combo_data("floorPlanOpeningRoomComboBox")
        primitive_data = self._map_studio_combo_data("roomPrimitiveTransformComboBox")
        primitive_surface_data = self._map_studio_combo_data("primitiveSurfaceComboBox")
        room_surface_data = self._map_studio_combo_data("roomSurfaceComboBox")
        opening_marker_data = self._map_studio_combo_data("floorPlanOpeningMarkerRoomComboBox")
        selected_points = self._map_studio_selected_point_indices()
        source_point = getattr(self.builder_tab, "floorPlanSourcePointSpinBox", None)
        target_point = getattr(self.builder_tab, "floorPlanTargetPointSpinBox", None)
        bridge_first_edge = getattr(self.builder_tab, "floorPlanBridgeFirstEdgeSpinBox", None)
        bridge_second_edge = getattr(self.builder_tab, "floorPlanBridgeSecondEdgeSpinBox", None)
        cleanup_tolerance = getattr(self.builder_tab, "floorPlanCleanupToleranceSpinBox", None)
        flatten_axis = getattr(self.builder_tab, "floorPlanFlattenAxisComboBox", None)
        mirror_axis = getattr(self.builder_tab, "floorPlanMirrorAxisComboBox", None)
        operation_combo = getattr(self.builder_tab, "roomOperationComboBox", None)
        operation_distance = getattr(self.builder_tab, "operationDistanceSpinBox", None)
        operation_edge = getattr(self.builder_tab, "operationEdgeIndexSpinBox", None)
        cut_center_x = getattr(self.builder_tab, "cutCenterXSpinBox", None)
        cut_center_y = getattr(self.builder_tab, "cutCenterYSpinBox", None)
        cut_width = getattr(self.builder_tab, "cutWidthSpinBox", None)
        cut_depth = getattr(self.builder_tab, "cutDepthSpinBox", None)
        duplicate_count = getattr(self.builder_tab, "duplicateSpecialCountSpinBox", None)
        duplicate_offset_x = getattr(self.builder_tab, "duplicateSpecialOffsetXSpinBox", None)
        duplicate_offset_y = getattr(self.builder_tab, "duplicateSpecialOffsetYSpinBox", None)
        duplicate_offset_z = getattr(self.builder_tab, "duplicateSpecialOffsetZSpinBox", None)
        duplicate_rotation_z = getattr(self.builder_tab, "duplicateSpecialRotationZSpinBox", None)
        duplicate_scale_x = getattr(self.builder_tab, "duplicateSpecialScaleXSpinBox", None)
        duplicate_scale_y = getattr(self.builder_tab, "duplicateSpecialScaleYSpinBox", None)
        duplicate_scale_z = getattr(self.builder_tab, "duplicateSpecialScaleZSpinBox", None)
        curve_name_line = getattr(self.builder_tab, "curveGuideNameLineEdit", None)
        curve_purpose_combo = getattr(self.builder_tab, "curveGuidePurposeComboBox", None)
        curve_p1x = getattr(self.builder_tab, "curveGuidePoint1XSpinBox", None)
        curve_p1y = getattr(self.builder_tab, "curveGuidePoint1YSpinBox", None)
        curve_p1z = getattr(self.builder_tab, "curveGuidePoint1ZSpinBox", None)
        curve_p2x = getattr(self.builder_tab, "curveGuidePoint2XSpinBox", None)
        curve_p2y = getattr(self.builder_tab, "curveGuidePoint2YSpinBox", None)
        curve_p2z = getattr(self.builder_tab, "curveGuidePoint2ZSpinBox", None)
        curve_p3x = getattr(self.builder_tab, "curveGuidePoint3XSpinBox", None)
        curve_p3y = getattr(self.builder_tab, "curveGuidePoint3YSpinBox", None)
        curve_p3z = getattr(self.builder_tab, "curveGuidePoint3ZSpinBox", None)
        key = str(action_key or "").strip()
        axis = "x"
        if key == "mirror_y":
            axis = "y"
        elif key == "mirror_x":
            axis = "x"
        elif key in {"cut", "split", "cut_slice_insert_edges", "insert_edge_loop"} and operation_combo is not None:
            axis = "y" if str(operation_combo.currentData() or "") == "split_y" else "x"
        elif key in {"mirror", "flatten", "grid_snap", "transform_snap_level"}:
            axis_combo = mirror_axis if key == "mirror" else flatten_axis
            if axis_combo is not None:
                axis = str(axis_combo.currentData() or "x")
        metadata: dict[str, Any] = {}
        if cleanup_tolerance is not None:
            metadata["tolerance"] = float(cleanup_tolerance.value())
        viewport_panel = getattr(self, "viewport_panel", None)
        active_modifier_getter = getattr(viewport_panel, "active_map_studio_modifier", None)
        active_modifier = active_modifier_getter() if callable(active_modifier_getter) else ""
        if active_modifier:
            metadata["active_modifier_action"] = str(active_modifier)
            metadata["active_modifier_behavior"] = "hold_modifier"
            metadata["active_modifier_source"] = "map_studio_viewport"
            metadata["active_modifier_coordinate_space"] = "viewport_interaction"
        weld_policy = getattr(self.builder_tab, "floorPlanWeldPolicyComboBox", None)
        if weld_policy is not None:
            metadata["position_policy"] = str(weld_policy.currentData() or "target")
        placement_kind_combo = getattr(self.builder_tab, "gameplayPlacementKindComboBox", None)
        placement_kind = str(placement_kind_combo.currentData() or "") if placement_kind_combo is not None else ""
        placement_template = str(getattr(getattr(self.builder_tab, "gameplayTemplateLineEdit", None), "text", lambda: "")()).strip()
        placement_tag = str(getattr(getattr(self.builder_tab, "gameplayTagLineEdit", None), "text", lambda: "")()).strip()
        placement_x = getattr(self.builder_tab, "gameplayPosXSpinBox", None)
        placement_y = getattr(self.builder_tab, "gameplayPosYSpinBox", None)
        placement_z = getattr(self.builder_tab, "gameplayPosZSpinBox", None)
        placement_bearing = getattr(self.builder_tab, "gameplayBearingSpinBox", None)
        entry_area = str(getattr(getattr(self.builder_tab, "entryPointAreaLineEdit", None), "text", lambda: "")()).strip()
        entry_x = getattr(self.builder_tab, "entryPointPosXSpinBox", None)
        entry_y = getattr(self.builder_tab, "entryPointPosYSpinBox", None)
        entry_z = getattr(self.builder_tab, "entryPointPosZSpinBox", None)
        entry_facing = getattr(self.builder_tab, "entryPointFacingSpinBox", None)
        light_room = str(getattr(getattr(self.builder_tab, "roomLightRoomLineEdit", None), "text", lambda: "")()).strip()
        light_name = str(getattr(getattr(self.builder_tab, "roomLightNameLineEdit", None), "text", lambda: "")()).strip()
        light_type_combo = getattr(self.builder_tab, "roomLightTypeComboBox", None)
        light_type = str(light_type_combo.currentData() or "point") if light_type_combo is not None else "point"
        light_x = getattr(self.builder_tab, "roomLightPosXSpinBox", None)
        light_y = getattr(self.builder_tab, "roomLightPosYSpinBox", None)
        light_z = getattr(self.builder_tab, "roomLightPosZSpinBox", None)
        light_r = getattr(self.builder_tab, "roomLightColorRSpinBox", None)
        light_g = getattr(self.builder_tab, "roomLightColorGSpinBox", None)
        light_b = getattr(self.builder_tab, "roomLightColorBSpinBox", None)
        light_radius = getattr(self.builder_tab, "roomLightRadiusSpinBox", None)
        light_intensity = getattr(self.builder_tab, "roomLightIntensitySpinBox", None)
        script_scope_combo = getattr(self.builder_tab, "scriptHookScopeComboBox", None)
        script_field_combo = getattr(self.builder_tab, "scriptHookFieldComboBox", None)
        script_scope = str(script_scope_combo.currentData() or "area") if script_scope_combo is not None else "area"
        script_field = str(
            (
                script_field_combo.currentData()
                if script_field_combo is not None and script_field_combo.currentData()
                else script_field_combo.currentText()
                if script_field_combo is not None
                else ""
            )
            or ""
        ).strip()
        script_resref = str(getattr(getattr(self.builder_tab, "scriptHookResrefLineEdit", None), "text", lambda: "")()).strip()
        wall_opening_name = str(getattr(getattr(self.builder_tab, "floorPlanOpeningNameLineEdit", None), "text", lambda: "")()).strip()
        wall_opening_edge = getattr(self.builder_tab, "floorPlanOpeningEdgeSpinBox", None)
        wall_opening_center = getattr(self.builder_tab, "floorPlanOpeningCenterSpinBox", None)
        wall_opening_width = getattr(self.builder_tab, "floorPlanOpeningWidthSpinBox", None)
        wall_opening_height = getattr(self.builder_tab, "floorPlanOpeningHeightSpinBox", None)
        wall_opening_bottom = getattr(self.builder_tab, "floorPlanOpeningBottomSpinBox", None)
        opening_marker_opening = getattr(self.builder_tab, "floorPlanOpeningMarkerNameComboBox", None)
        opening_marker_kind = getattr(self.builder_tab, "floorPlanOpeningMarkerKindComboBox", None)
        opening_marker_template = str(getattr(getattr(self.builder_tab, "floorPlanOpeningMarkerTemplateLineEdit", None), "text", lambda: "")()).strip()
        opening_marker_tag = str(getattr(getattr(self.builder_tab, "floorPlanOpeningMarkerTagLineEdit", None), "text", lambda: "")()).strip()
        opening_marker_linked_to = str(getattr(getattr(self.builder_tab, "floorPlanOpeningMarkerLinkedToLineEdit", None), "text", lambda: "")()).strip()
        opening_marker_linked_module = str(getattr(getattr(self.builder_tab, "floorPlanOpeningMarkerLinkedModuleLineEdit", None), "text", lambda: "")()).strip()
        opening_marker_target_type = getattr(self.builder_tab, "floorPlanOpeningMarkerTargetTypeComboBox", None)
        opening_marker_transition = getattr(self.builder_tab, "floorPlanOpeningMarkerTransitionDestSpinBox", None)
        terrain_context_getter = getattr(self.builder_tab, "current_terrain_brush_context", None)
        terrain_context = terrain_context_getter() if callable(terrain_context_getter) else {}
        if not isinstance(terrain_context, dict):
            terrain_context = {}
        terrain_row = getattr(self.builder_tab, "terrainRowSpinBox", None)
        terrain_column = getattr(self.builder_tab, "terrainColumnSpinBox", None)
        module_root_line = getattr(self.builder_tab, "moduleRootLineEdit", None)
        primitive_kind_combo = getattr(self.builder_tab, "compositionPrimitiveKindComboBox", None)
        primitive_name_line = getattr(self.builder_tab, "compositionPrimitiveNameLineEdit", None)
        primitive_kind_data = primitive_kind_combo.currentData() if primitive_kind_combo is not None else {}
        if not isinstance(primitive_kind_data, dict):
            primitive_kind_data = {}
        direct_primitive_keys = {
            "floor",
            "plane",
            "cube",
            "wall",
            "ramp",
            "stairs",
            "cylinder",
            "door_frame",
            "arch",
        }
        if key == "primitive":
            primitive_kind = str(primitive_kind_data.get("kind") or "").strip()
            primitive_name = str(getattr(primitive_name_line, "text", lambda: "")()).strip()
        elif key in direct_primitive_keys:
            primitive_kind = ""
            primitive_name = ""
        else:
            primitive_kind = ""
            primitive_name = str(primitive_data.get("primitive_name") or "")
        if primitive_name and "supports_walkmesh_surface" in primitive_data:
            metadata["supports_walkmesh_surface"] = bool(primitive_data.get("supports_walkmesh_surface"))
            metadata["selected_primitive_type"] = str(primitive_data.get("primitive_type") or "")
            metadata["selected_primitive_surface_name"] = str(primitive_data.get("surface_name") or "")
        move_delta = (0.0, 0.0, 0.0)
        if primitive_name:
            before_translation = tuple(float(value) for value in tuple(primitive_data.get("translation") or (0.0, 0.0, 0.0))[:3])
            if len(before_translation) != 3:
                before_translation = (0.0, 0.0, 0.0)
            move_x = getattr(self.builder_tab, "primitiveTranslateXSpinBox", None)
            move_y = getattr(self.builder_tab, "primitiveTranslateYSpinBox", None)
            move_z = getattr(self.builder_tab, "primitiveTranslateZSpinBox", None)
            if move_x is not None and move_y is not None and move_z is not None:
                after_translation = (float(move_x.value()), float(move_y.value()), float(move_z.value()))
                move_delta = tuple(after_translation[index] - before_translation[index] for index in range(3))
        if key == "paint_wok":
            surface_data = primitive_surface_data if primitive_name else room_surface_data
            surface_id = str(surface_data.get("surface_id") or primitive_data.get("surface_id") or "").strip()
            if surface_id:
                metadata["surface_id"] = surface_id
        if key == "paint_material":
            texture_line = (
                getattr(self.builder_tab, "primitiveTextureLineEdit", None)
                if primitive_name
                else getattr(self.builder_tab, "roomTextureLineEdit", None)
            )
            texture = str(getattr(texture_line, "text", lambda: "")()).strip()
            if texture:
                metadata["texture"] = texture
            surface_data = primitive_surface_data if primitive_name else room_surface_data
            surface_id = str(surface_data.get("surface_id") or primitive_data.get("surface_id") or "").strip()
            if surface_id:
                metadata["surface_id"] = surface_id
        if key == "curve_tool":
            metadata["curve_name"] = str(getattr(curve_name_line, "text", lambda: "")()).strip()
            metadata["curve_purpose"] = str(
                (
                    curve_purpose_combo.currentData()
                    if curve_purpose_combo is not None and curve_purpose_combo.currentData()
                    else "path_guide"
                )
            )
            metadata["coordinate_space"] = "kmap_world"
            metadata["points"] = (
                (
                    float(curve_p1x.value()) if curve_p1x is not None else 0.0,
                    float(curve_p1y.value()) if curve_p1y is not None else 0.0,
                    float(curve_p1z.value()) if curve_p1z is not None else 0.0,
                ),
                (
                    float(curve_p2x.value()) if curve_p2x is not None else 1.0,
                    float(curve_p2y.value()) if curve_p2y is not None else 0.5,
                    float(curve_p2z.value()) if curve_p2z is not None else 0.0,
                ),
                (
                    float(curve_p3x.value()) if curve_p3x is not None else 2.0,
                    float(curve_p3y.value()) if curve_p3y is not None else 0.5,
                    float(curve_p3z.value()) if curve_p3z is not None else 0.0,
                ),
            )
        terrain_room_resref = str(terrain_context.get("room_resref") or "").strip()
        if key.startswith("sculpt_"):
            current_room_resref = terrain_room_resref
        elif key == "opening":
            current_room_resref = str(opening_data.get("room_resref") or "").strip()
        elif key == "opening_marker":
            current_room_resref = str(opening_marker_data.get("room_resref") or "").strip()
        else:
            current_room_resref = str(
                vertex_data.get("room_resref")
                or primitive_data.get("room_resref")
                or self._map_studio_current_room_resref()
            ).strip()
        return MapStudioToolActionContext(
            module_root=str(getattr(module_root_line, "text", lambda: "")()).strip(),
            room_resref=current_room_resref,
            first_room_resref=str(bridge_first.get("room_resref") or union_first.get("room_resref") or ""),
            second_room_resref=str(bridge_second.get("room_resref") or union_second.get("room_resref") or ""),
            result_room_resref=str(
                getattr(getattr(self.builder_tab, "floorPlanUnionResultRoomLineEdit", None), "text", lambda: "")()
                if key == "combine"
                else getattr(getattr(self.builder_tab, "floorPlanBridgeResultRoomLineEdit", None), "text", lambda: "")()
                if key == "bridge"
                else getattr(getattr(self.builder_tab, "roomPrimitiveSeparateResultLineEdit", None), "text", lambda: "")()
            ).strip(),
            primitive_name=primitive_name,
            primitive_kind=primitive_kind,
            placement_kind=placement_kind,
            placement_template_resref=placement_template,
            placement_tag=placement_tag,
            placement_position=(
                float(placement_x.value()) if placement_x is not None else 0.0,
                float(placement_y.value()) if placement_y is not None else 0.0,
                float(placement_z.value()) if placement_z is not None else 0.0,
            ),
            placement_bearing=(
                math.radians(float(placement_bearing.value()))
                if placement_bearing is not None
                else 0.0
            ),
            entry_area_resref=entry_area,
            entry_position=(
                float(entry_x.value()) if entry_x is not None else 0.0,
                float(entry_y.value()) if entry_y is not None else 0.0,
                float(entry_z.value()) if entry_z is not None else 0.0,
            ),
            entry_facing=(
                math.radians(float(entry_facing.value()))
                if entry_facing is not None
                else 0.0
            ),
            light_room_resref=light_room,
            light_name=light_name,
            light_position=(
                float(light_x.value()) if light_x is not None else 0.0,
                float(light_y.value()) if light_y is not None else 0.0,
                float(light_z.value()) if light_z is not None else 2.25,
            ),
            light_color=(
                float(light_r.value()) if light_r is not None else 1.0,
                float(light_g.value()) if light_g is not None else 0.92,
                float(light_b.value()) if light_b is not None else 0.78,
            ),
            light_radius=float(light_radius.value()) if light_radius is not None else 8.0,
            light_intensity=float(light_intensity.value()) if light_intensity is not None else 1.0,
            light_type=light_type,
            script_scope=script_scope,
            script_field_name=script_field,
            script_resref=script_resref,
            wall_opening_name=wall_opening_name,
            wall_opening_edge_index=int(wall_opening_edge.value()) if wall_opening_edge is not None else 0,
            wall_opening_center_fraction=float(wall_opening_center.value()) if wall_opening_center is not None else 0.5,
            wall_opening_width=float(wall_opening_width.value()) if wall_opening_width is not None else 1.5,
            wall_opening_height=float(wall_opening_height.value()) if wall_opening_height is not None else 2.1,
            wall_opening_bottom=float(wall_opening_bottom.value()) if wall_opening_bottom is not None else 0.0,
            opening_name=str(
                (
                    opening_marker_opening.currentData()
                    if opening_marker_opening is not None and opening_marker_opening.currentData()
                    else opening_marker_opening.currentText()
                    if opening_marker_opening is not None
                    else ""
                )
                or ""
            ).strip(),
            opening_marker_kind=str(
                (
                    opening_marker_kind.currentData()
                    if opening_marker_kind is not None and opening_marker_kind.currentData()
                    else opening_marker_kind.currentText()
                    if opening_marker_kind is not None
                    else "door"
                )
                or "door"
            ),
            opening_marker_template_resref=opening_marker_template,
            opening_marker_tag=opening_marker_tag,
            opening_marker_linked_to=opening_marker_linked_to,
            opening_marker_linked_to_module=opening_marker_linked_module,
            opening_marker_linked_to_flags=(
                int(opening_marker_target_type.currentData() or 0)
                if opening_marker_target_type is not None and opening_marker_linked_to
                else 0
            ),
            opening_marker_transition_destination=int(opening_marker_transition.value()) if opening_marker_transition is not None else 0,
            point_index=int(source_point.value()) if source_point is not None else None,
            point_indices=selected_points,
            target_point_index=int(target_point.value()) if target_point is not None else None,
            target_room_resref=str(vertex_target_data.get("room_resref") or ""),
            first_edge_index=int(bridge_first_edge.value()) if bridge_first_edge is not None else None,
            second_edge_index=int(bridge_second_edge.value()) if bridge_second_edge is not None else None,
            axis=axis,
            positive_z=key != "reverse_normals",
            operation_distance=float(operation_distance.value()) if operation_distance is not None else 0.25,
            operation_edge_index=int(operation_edge.value()) if operation_edge is not None else 0,
            terrain_row_index=int(terrain_row.value()) if terrain_row is not None else 0,
            terrain_column_index=int(terrain_column.value()) if terrain_column is not None else 0,
            terrain_delta=float(terrain_context.get("delta", 0.1) or 0.1),
            terrain_radius=int(terrain_context.get("radius", 0) or 0),
            terrain_height=float(terrain_context.get("height", 0.0) or 0.0),
            terrain_iterations=int(terrain_context.get("iterations", 1) or 1),
            terrain_strength=float(terrain_context.get("strength", 0.5) or 0.5),
            cut_center=(
                float(cut_center_x.value()) if cut_center_x is not None else 0.0,
                float(cut_center_y.value()) if cut_center_y is not None else 0.0,
            ),
            cut_size=(
                float(cut_width.value()) if cut_width is not None else 1.0,
                float(cut_depth.value()) if cut_depth is not None else 1.0,
            ),
            duplicate_count=int(duplicate_count.value()) if duplicate_count is not None else 1,
            duplicate_translation_offset=(
                float(duplicate_offset_x.value()) if duplicate_offset_x is not None else 1.0,
                float(duplicate_offset_y.value()) if duplicate_offset_y is not None else 0.0,
                float(duplicate_offset_z.value()) if duplicate_offset_z is not None else 0.0,
            ),
            duplicate_rotation_offset_degrees_z=float(duplicate_rotation_z.value()) if duplicate_rotation_z is not None else 0.0,
            duplicate_scale_multiplier=(
                float(duplicate_scale_x.value()) if duplicate_scale_x is not None else 1.0,
                float(duplicate_scale_y.value()) if duplicate_scale_y is not None else 1.0,
                float(duplicate_scale_z.value()) if duplicate_scale_z is not None else 1.0,
            ),
            move_delta=move_delta,
            export_output_dir=str(getattr(self, "_last_output_dir", "") or "").strip(),
            export_dry_run=self._map_studio_export_dry_run_enabled(),
            export_overwrite=bool(getattr(self, "_last_map_studio_install_overwrite", False))
            if key == "install_module"
            else False,
            export_game_modules_dir=str(getattr(self, "_last_game_modules_dir", "") or "").strip(),
            metadata=metadata,
        )

    def _ensure_map_studio_export_output_dir(self, title: str) -> bool:
        """Prompt once for an export/staging folder when the belt route needs it."""

        if str(getattr(self, "_last_output_dir", "") or "").strip():
            return True
        path = QtWidgets.QFileDialog.getExistingDirectory(self, title, "")
        if not path:
            return False
        self._last_output_dir = path
        return True

    def _map_studio_authored_module_root_for_install(self) -> str:
        """Return the authored module root used for install overwrite checks."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        return str(payload.get("module_root") or getattr(self.project, "name", "") or "authored").strip().lower()

    def _ensure_map_studio_game_modules_dir(self) -> bool:
        """Prompt for the target KOTOR Modules folder when Install Test needs it."""

        modules_path = str(getattr(self, "_last_game_modules_dir", "") or "").strip()
        if not modules_path:
            modules_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select KOTOR Modules folder", "")
            if not modules_path:
                return False
            self._last_game_modules_dir = modules_path
        module_root = self._map_studio_authored_module_root_for_install()
        destination = Path(modules_path) / f"{module_root}.mod"
        self._last_map_studio_install_overwrite = False
        if destination.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Install Authored Module",
                f"{destination.name} already exists in the selected Modules folder.\n\n"
                "GhostRigger will create a .bak backup before replacing it. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return False
            self._last_map_studio_install_overwrite = True
        return True

    def _open_map_studio_package_wizard(self, *, mode: str, dry_run: bool) -> dict[str, object] | None:
        """Collect package/stage/install targets through one Map Studio review step."""

        readiness = None
        try:
            readiness = self.controller.authored_module_readiness().readiness
        except Exception:
            readiness = None
        dialog = _MapStudioPackageWizardDialog(
            self,
            mode=mode,
            readiness=readiness,
            output_dir=str(getattr(self, "_last_output_dir", "") or ""),
            game_modules_dir=str(getattr(self, "_last_game_modules_dir", "") or ""),
            dry_run=bool(dry_run),
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        values = dialog.values()
        self._last_output_dir = str(values.get("output_dir") or "")
        self._last_game_modules_dir = str(values.get("game_modules_dir") or "")
        self._last_map_studio_install_overwrite = bool(values.get("overwrite"))
        self.export_panel.dry_run.setChecked(bool(values.get("dry_run")))
        return values

    def _try_arm_map_studio_component_tool(self, action_key: str) -> bool:
        """Prefer Maya-style live component tools when a component is active."""

        key = str(action_key or "").strip().lower()
        if key not in {"extrude", "bevel"}:
            return False
        panel = getattr(self, "viewport_panel", None)
        if panel is None:
            return False
        selection_reader = getattr(panel, "map_studio_component_selection", None)
        selection = list(selection_reader() or ()) if callable(selection_reader) else []
        component_mode = str(getattr(panel, "_hover_component_mode", "") or "").strip().lower()
        if not selection and component_mode not in {"face", "edge"}:
            return False
        armer = getattr(panel, "arm_component_extrude" if key == "extrude" else "arm_component_bevel", None)
        if not callable(armer) or not bool(armer()):
            return False
        label = "Extrude" if key == "extrude" else "Bevel"
        self.statusBar().showMessage(
            (
                f"{label} armed on {len(selection)} selected edges. Adjust options and click Apply for one atomic edit; Esc cancels."
                if key == "bevel" and len(selection) > 1
                else f"{label} armed on the selected component. Drag the live viewport manipulator; Esc cancels."
            ),
            5000,
        )
        return True

    def _execute_map_studio_tool_belt_command(self, action_key: str) -> bool:
        """Execute a tool-belt action when the core dispatcher has a command."""

        if action_key == "texture_paint":
            self._show_map_studio_texture_paint_workflow()
            return True
        if self._try_arm_map_studio_component_tool(action_key):
            return True
        if action_key in {"stage_module", "install_module"}:
            values = self._open_map_studio_package_wizard(
                mode="install" if action_key == "install_module" else "stage",
                dry_run=self._map_studio_export_dry_run_enabled(),
            )
            if values is None:
                self._focus_map_studio_export_proof_workspace()
                self.statusBar().showMessage("Map Studio package command canceled before any files were written.", 5000)
                return False
        context = self._map_studio_tool_action_context(action_key)
        route = resolve_map_studio_tool_belt_action(action_key, context)
        duplicate_before_names: set[tuple[str, str]] = set()
        primitive_before_names: set[tuple[str, str]] | None = None
        if action_key == "duplicate_selected":
            duplicate_before_names = {
                (
                    str(getattr(row, "room_resref", "") or ""),
                    str(getattr(row, "primitive_name", "") or ""),
                )
                for row in self.controller.authored_room_primitive_transforms()
            }
        if route.command_method == "add_authored_room_primitive":
            primitive_before_names = {
                (
                    str(getattr(row, "room_resref", "") or ""),
                    str(getattr(row, "primitive_name", "") or ""),
                )
                for row in self.controller.authored_room_primitive_transforms()
            }
        if not route.command_method:
            if route.disabled_reason:
                self.statusBar().showMessage(route.disabled_reason, 5000)
                self._log(f"Map Studio action not ready: {route.disabled_reason}")
            return False
        if not route.enabled:
            self.statusBar().showMessage(route.disabled_reason or "Map Studio action is not ready.", 5000)
            self._log(f"Map Studio action not ready: {route.disabled_reason}")
            return False
        try:
            result = execute_map_studio_tool_belt_action(self.controller, action_key, context)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 6000)
            self._log(f"Map Studio action failed: {exc}")
            return False
        status_message = route.status_message or f"{route.label} complete."
        if action_key == "terrain" and isinstance(result, dict):
            status_message = str(result.get("summary") or status_message)
            next_action = str(result.get("next_action") or "").strip()
            if next_action:
                status_message = f"{status_message} Next: {next_action}"
            self.show_map_studio_terrain_tools()
        elif action_key == "walkmesh":
            summary = str(getattr(result, "summary", "") or status_message)
            next_action = str(getattr(result, "next_action", "") or "").strip()
            status_message = f"{summary} Next: {next_action}" if next_action else summary
            self.show_map_studio_walkmesh_tools()
        elif action_key == "validate":
            issues = list(result or ())
            errors = sum(1 for issue in issues if str(getattr(issue, "severity", "")).lower() == "error")
            warnings = sum(1 for issue in issues if str(getattr(issue, "severity", "")).lower() == "warning")
            status_message = f"Validation complete: {len(issues)} issue(s), {errors} error(s), {warnings} warning(s)."
            self.validation_panel.set_issues(issues)
            self._show_map_studio_validation_output(self.validation_panel)
            self._set_map_studio_workspace_combo_key("export")
        elif action_key == "script":
            scope = str(getattr(result, "scope", "") or "").strip()
            field_name = str(getattr(result, "field_name", "") or "").strip()
            script_resref = str(getattr(result, "script_resref", "") or "").strip()
            if bool(getattr(result, "removed", False)):
                status_message = f"Cleared {scope} script hook {field_name}; export, install handoff, and game proof are stale."
            else:
                status_message = (
                    f"Assigned {scope} script hook {field_name} -> {script_resref}; "
                    "export, install handoff, and game proof are stale."
                )
            self.show_map_studio_script_tools()
        elif action_key == "stage_module":
            status_message = str(getattr(result, "message", "") or status_message)
            self._last_output_dir = str(route.command_kwargs.get("output_dir") or self._last_output_dir or "")
            self._log_authored_module_stage_result(result)
            self._focus_map_studio_export_proof_workspace()
        elif action_key == "install_module":
            status_message = str(getattr(result, "message", "") or status_message)
            self._last_output_dir = str(route.command_kwargs.get("output_dir") or self._last_output_dir or "")
            self._last_game_modules_dir = str(route.command_kwargs.get("game_modules_dir") or self._last_game_modules_dir or "")
            self._log_authored_module_stage_result(result)
            if not bool(getattr(result, "ok", False)):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Install Authored Module",
                    str(getattr(result, "message", "") or "Install failed."),
                )
            self._focus_map_studio_export_proof_workspace()
        elif action_key == "launch_handoff":
            summary = str(getattr(result, "summary", "") or status_message)
            blockers = tuple(getattr(result, "blocking_messages", ()) or ())
            next_action = str(getattr(result, "next_action", "") or "").strip()
            status_message = f"{summary} {blockers[0]}" if blockers else summary
            if next_action and not blockers:
                status_message = f"{status_message} Next: {next_action}"
            self._focus_map_studio_export_proof_workspace()
            self._open_map_studio_launch_handoff_dialog_from_summary(result)
        elif action_key == "record_proof":
            summary = str(getattr(result, "summary", "") or status_message)
            blockers = tuple(getattr(result, "blocking_messages", ()) or ())
            next_action = str(getattr(result, "next_action", "") or "").strip()
            status_message = f"{summary} {blockers[0]}" if blockers else summary
            if next_action and not blockers:
                status_message = f"{status_message} Next: {next_action}"
            self._focus_map_studio_export_proof_workspace()
            self._record_game_smoke_proof_from_summary(result)
        if action_key == "universal_transform":
            overlay_setter = getattr(self.viewport_panel, "set_universal_transform_overlay", None)
            if callable(overlay_setter):
                overlay_setter(result)
            dimensions = tuple(getattr(result, "dimensions", ()) or ())
            center = tuple(getattr(result, "center", ()) or ())
            if len(dimensions) == 3:
                status_message = (
                    f"Universal Transform: {getattr(result, 'primitive_name', '')} "
                    f"W {dimensions[0]:.3f} / D {dimensions[1]:.3f} / H {dimensions[2]:.3f} m"
                )
                if len(center) == 3:
                    status_message += f"; center {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}."
            self._log(status_message)
        select_primitive_after_refresh: tuple[str, str] | None = None
        if action_key == "duplicate_selected":
            new_rows = [
                row
                for row in self.controller.authored_room_primitive_transforms()
                if (
                    str(getattr(row, "room_resref", "") or ""),
                    str(getattr(row, "primitive_name", "") or ""),
                )
                not in duplicate_before_names
            ]
            if new_rows:
                selected = new_rows[-1]
                select_primitive_after_refresh = (
                    str(getattr(selected, "room_resref", "") or ""),
                    str(getattr(selected, "primitive_name", "") or ""),
                )
        elif primitive_before_names is not None:
            new_rows = [
                row
                for row in self.controller.authored_room_primitive_transforms()
                if (
                    str(getattr(row, "room_resref", "") or ""),
                    str(getattr(row, "primitive_name", "") or ""),
                )
                not in primitive_before_names
            ]
            if new_rows:
                selected = new_rows[-1]
                select_primitive_after_refresh = (
                    str(getattr(selected, "room_resref", "") or ""),
                    str(getattr(selected, "primitive_name", "") or ""),
                )
        command_method = str(getattr(route, "command_method", "") or "")
        geometry_commands = {
            "move_authored_room_primitive",
            "add_authored_room_primitive",
            "apply_authored_terrain_brush_stroke",
            "reset_authored_room_primitive_transform",
            "zero_authored_room_primitive_pivot",
            "delete_authored_room_primitive_history",
            "center_authored_room_primitive_pivot",
            "freeze_authored_room_primitive_transform",
            "grid_snap_authored_room_primitive",
            "shrink_wrap_authored_room_primitive_to_terrain",
            "set_authored_room_primitive_style",
            "set_authored_room_walkmesh_surface",
            "apply_authored_room_style",
            "set_authored_floor_plan_wall_opening",
            "snap_authored_floor_plan_vertex",
            "grid_snap_authored_floor_plan_vertices",
            "weld_authored_floor_plan_vertices",
            "flatten_authored_floor_plan_vertices",
            "transform_snap_authored_room_primitive_level",
            "transform_snap_authored_floor_plan_vertices",
            "cleanup_authored_floor_plan_vertices",
            "mirror_authored_room_primitive_transform",
            "mirror_z_authored_terrain_heightfield",
            "bend_authored_terrain_heightfield",
            "lattice_authored_terrain_heightfield",
            "add_authored_curve_guide",
            "mirror_authored_floor_plan_vertices",
            "cleanup_authored_floor_plan_normals",
            "set_authored_room_edge_normal_policy",
            "edge_extrude_authored_floor_plan_room",
            "bevel_authored_floor_plan_room",
            "inset_authored_floor_plan_room",
            "rectangular_cut_authored_floor_plan_room",
            "axis_split_authored_floor_plan_room",
            "boolean_difference_authored_floor_plan_rooms",
            "triangulate_authored_floor_plan_face",
            "fill_authored_floor_plan_face",
            "bridge_authored_floor_plan_edges",
            "combine_authored_room_primitives",
            "merge_authored_floor_plan_rooms",
            "separate_authored_room_primitive_shells",
            "duplicate_authored_room_primitive",
            "remove_authored_room_primitive",
            "snap_authored_room_primitive_pivot_to_vertex",
        }
        no_rebuild_commands = {
            "set_map_studio_active_selection",
            "authored_terrain_status",
            "authored_walkmesh_status",
            "map_studio_universal_transform_overlay",
        }
        if command_method in geometry_commands:
            selection = (select_primitive_after_refresh,) if select_primitive_after_refresh is not None else ()
            changes_room_set = command_method in {
                "boolean_difference_authored_floor_plan_rooms",
                "bridge_authored_floor_plan_edges",
                "merge_authored_floor_plan_rooms",
            }
            refresh_room_choices = changes_room_set or command_method == "set_authored_floor_plan_wall_opening"
            self._refresh_map_studio_geometry_change(
                status_message,
                primitive_selection=selection,
                refresh_outlines=True,
                refresh_terrain=(
                    "terrain" in command_method
                    or "walkmesh" in command_method
                    or command_method == "apply_authored_room_style"
                ),
                refresh_room_choices=refresh_room_choices,
                refresh_connections=("floor_plan" in command_method or changes_room_set),
            )
        elif command_method in no_rebuild_commands:
            self._update_map_studio_undo_redo_actions()
            self._log(status_message)
        else:
            self._refresh_all(status_message)
        if select_primitive_after_refresh is not None:
            self._select_map_studio_room_primitive(*select_primitive_after_refresh)
        self.workflow_panel.set_active_authoring_context(
            route.authoring_context or route.readiness_impact or route.status_message or route.label
        )
        return True

    def _select_map_studio_room_primitive(self, room_resref: str, primitive_name: str) -> bool:
        """Select an authored composition primitive in the visible Builder controls."""

        selector = getattr(self.builder_tab, "select_room_primitive", None)
        if callable(selector):
            try:
                selected = bool(selector(room_resref, primitive_name))
                if selected:
                    self._refresh_map_studio_selected_primitive_transform_overlay()
                return selected
            except Exception:
                return False
        return False

    def _focus_map_studio_entry_point_controls(self) -> None:
        """Focus Builder controls for the authored IFO player start."""

        self.show_map_studio_placement_tools()
        area = getattr(self.builder_tab, "entryPointAreaLineEdit", None)
        if area is not None:
            area.setFocus()
            area.selectAll()
        self.workflow_panel.set_active_authoring_context(
            "Entry point: edit the module IFO player start and keep it on walkable WOK"
        )
        self._log("Map Studio entry point controls focused. Set the area resref, XYZ, and facing before validation/game proof.")

    def _select_map_studio_modeling_tool(self, tool_key: str) -> None:
        """Focus the Builder modeling tool matching a belt action."""

        combo = getattr(self.builder_tab, "modelingToolComboBox", None)
        if combo is None:
            return
        wanted = str(tool_key or "").strip()
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and str(data.get("key") or "") == wanted:
                combo.setCurrentIndex(index)
                combo.setFocus()
                return

    def _select_map_studio_component_mode(self, component_key: str) -> None:
        """Synchronize the Builder component selector with the toolbar edit mode."""

        combo = getattr(self.builder_tab, "componentModeComboBox", None)
        if combo is None:
            return
        wanted = str(component_key or "").strip().lower()
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and str(data.get("key") or "").strip().lower() == wanted:
                combo.setCurrentIndex(index)
                return

    def _select_map_studio_snap_mode(self, snap_key: str) -> None:
        """Synchronize the Builder snap selector with a tool-belt action."""

        combo = getattr(self.builder_tab, "snapModeComboBox", None)
        if combo is None:
            return
        wanted = str(snap_key or "").strip().lower()
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and str(data.get("key") or "").strip().lower() == wanted:
                combo.setCurrentIndex(index)
                return

    def _focus_map_studio_vertex_workflow(self, action_key: str) -> None:
        """Route vertex-oriented belt actions to the Builder vertex workflow."""

        key = str(action_key or "").strip()
        tool_by_action = {
            "vertex_snap": "snap_vertices",
            "grid_snap": "snap_vertices",
            "transform_snap_level": "transform_snap_level",
            "weld": "weld_vertices",
            "flatten": "flatten_vertices",
            "mirror": "mirror_footprint",
            "cleanup": "cleanup_footprint",
        }
        snap_by_action = {
            "vertex_snap": "vertex",
            "grid_snap": "grid",
            "transform_snap_level": "level",
            "weld": "vertex",
            "flatten": "grid",
            "mirror": "grid",
            "cleanup": "grid",
        }
        context_by_action = {
            "vertex_snap": (
                "Vertex snap: move one floor-plan point to another point or room handle "
                "without welding topology. Hold V previews point snapping; commit through "
                "Snap Vertex so KMAP, WOK, readiness, and export-stale state update together."
            ),
            "grid_snap": (
                "Grid snap: move selected floor-plan points to the authored Map Studio grid "
                "without welding topology. Validate room seams, WOK, staged export, and game "
                "proof after snapping."
            ),
            "weld": (
                "Weld vertices: merge selected floor-plan points into one topology point "
                "and repair room/WOK references before export."
            ),
            "flatten": "Flatten vertices: align selected points on a local X/Y line for clean walls, seams, and doorways.",
            "transform_snap_level": (
                "Transform level snap: hold J with transform active to align selected vertices or edges "
                "onto one shared X/Y/Z level before validating room seams and WOK output."
            ),
            "mirror": "Mirror vertices: mirror authored footprint points while preserving a valid convex KOTOR room boundary.",
            "cleanup": "Cleanup vertices: remove duplicate or collinear floor-plan points before MDL/WOK generation.",
        }
        log_by_action = {
            "vertex_snap": (
                "Map Studio Vertex Snap focused. This moves a point to another point; it does "
                "not merge topology. Use Weld when the points should become one vertex."
            ),
            "grid_snap": (
                "Map Studio Grid Snap focused. This moves selected floor-plan points to the "
                "grid; it does not weld topology."
            ),
            "weld": "Map Studio Weld focused. Welding merges topology and can change WOK/room face references.",
            "flatten": "Map Studio Flatten focused. Align selected points before validating room seams and WOK output.",
            "transform_snap_level": "Map Studio Transform Level Snap focused. Hold J during transform to align selected vertices/edges to one level.",
            "mirror": "Map Studio Mirror focused. Mirrored footprints still need convexity and WOK validation.",
            "cleanup": "Map Studio Cleanup focused. Cleanup removes duplicate/collinear points before export.",
        }
        self._select_map_studio_component_mode("vertex")
        self._select_map_studio_modeling_tool(tool_by_action.get(key, "snap_vertices"))
        self._select_map_studio_snap_mode(snap_by_action.get(key, "grid"))
        self.workflow_panel.set_active_authoring_context(context_by_action.get(key, context_by_action["vertex_snap"]))
        self._log(log_by_action.get(key, log_by_action["vertex_snap"]))
        tool = getattr(self.builder_tab, "floorPlanVertexRoomComboBox", None)
        if tool is not None:
            tool.setFocus()
        preview = getattr(self.builder_tab, "request_floor_plan_vertex_snap_preview", None)
        if callable(preview):
            preview()

    def _activate_map_studio_universal_transform_shortcut(self) -> None:
        """Route Ctrl+T through the Map Studio tool-belt action catalog."""

        for action in self.controller.available_map_studio_tool_belt_actions():
            if str(getattr(action, "key", "") or "") == "universal_transform":
                self._handle_map_studio_tool_belt_action(action)
                return
        self._focus_map_studio_universal_transform()

    def _activate_map_studio_modifier_shortcut(self, action_key: str) -> None:
        """Route Maya-style viewport modifier shortcuts through Map Studio tools."""

        key = str(action_key or "").strip()
        if self._execute_map_studio_tool_belt_command(key):
            return
        if key == "vertex_snap":
            self._focus_map_studio_vertex_workflow("vertex_snap")
            message = (
                "Hold V: vertex snap mode focused. Select a source and target "
                "floor-plan vertex, then drag/snap in the viewport."
            )
        elif key == "transform_snap_level":
            self._focus_map_studio_vertex_workflow("transform_snap_level")
            message = (
                "Hold J: transform level snap focused. Select two or more "
                "vertices/edges to align to a shared level."
            )
        else:
            message = f"Map Studio shortcut {key} is not mapped."
        self.workflow_panel.set_active_authoring_context(message)
        self.statusBar().showMessage(message, 5000)
        self._log(message)

    def _focus_map_studio_universal_transform(self) -> None:
        """Focus the selected-component Universal Manipulator workflow."""

        self.show_map_studio_geometry_tools()
        self._select_map_studio_component_mode("object")
        self._select_map_studio_modeling_tool("universal_transform")
        self.workflow_panel.set_active_authoring_context(
            "Universal Manipulator: Ctrl+T displays selected component bounds, gizmo handles, and exact width/depth/height for modular-kit scaling."
        )
        self.statusBar().showMessage("Map Studio Universal Manipulator active. Select a mesh/component to inspect width, depth, and height.", 5000)
        self._log("Map Studio Universal Manipulator focused. Use selected bounds for exact modular-kit dimensions.")

    def _select_map_studio_gameplay_kind(self, placement_kind: str) -> None:
        """Focus the Builder placement controls for one KOTOR resource kind."""

        combo = getattr(self.builder_tab, "gameplayPlacementKindComboBox", None)
        if combo is None:
            return
        wanted = str(placement_kind or "").strip().lower()
        for index in range(combo.count()):
            if str(combo.itemData(index) or "").strip().lower() == wanted:
                combo.setCurrentIndex(index)
                break
        search = getattr(self.builder_tab, "gameplayPaletteSearchLineEdit", None)
        if search is not None:
            search.setFocus()
            search.selectAll()
        else:
            combo.setFocus()

    def _select_map_studio_terrain_brush(self, brush_key: str) -> None:
        """Focus the Builder terrain sculpt brush matching a belt action."""

        combo = getattr(self.builder_tab, "terrainBrushComboBox", None)
        if combo is None:
            return
        wanted = str(brush_key or "").strip()
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and str(data.get("key") or "") == wanted:
                combo.setCurrentIndex(index)
                label = str(data.get("label") or wanted).strip() or wanted
                operation = str(data.get("operation") or wanted).strip() or wanted
                guardrail = str(data.get("guardrail") or "").strip()
                self.workflow_panel.set_active_authoring_context(
                    f"Terrain brush: {label}. Live strokes update dirty terrain samples only; "
                    "full MDL/WOK rebuild waits for stroke commit, validation, or export."
                )
                message = (
                    f"Map Studio terrain brush selected: {label} ({operation}). "
                    "Brush frames stay dirty-region scoped for low-latency sculpting."
                )
                if guardrail:
                    message += f" KOTOR: {guardrail}"
                self._log(message)
                self._sync_map_studio_terrain_brush_context(force_enabled=True)
                return
        self.workflow_panel.set_active_authoring_context(
            f"Terrain brush: {wanted or '(none)'} is not available in the current Map Studio tool set."
        )
        self._log(f"Map Studio terrain brush '{wanted}' is not available.")

    def _focus_map_studio_opening_marker_controls(self) -> None:
        """Focus Builder controls that convert authored openings into KOTOR transition markers."""

        self.show_map_studio_geometry_tools()
        marker_room = getattr(self.builder_tab, "floorPlanOpeningMarkerRoomComboBox", None)
        if marker_room is not None:
            marker_room.setFocus()
        self.workflow_panel.set_active_authoring_context(
            "Opening marker: create a door/trigger transition source or waypoint destination. Set Linked To, target type, and module on sources."
        )
        self._log(
            "Map Studio opening transition marker controls focused. Choose an authored opening, marker kind, template/tag, and transition destination."
        )

    def _select_authored_room_outline_edge(self, room_resref: str, edge_index: int) -> None:
        """Focus Builder edge tools after a floor-plan edge is selected in the viewport."""

        room = str(room_resref or "").strip()
        edge = int(edge_index)
        if not room or edge < 0:
            return
        self.show_map_studio_geometry_tools()
        self._select_map_studio_component_mode("edge")
        self._select_map_studio_modeling_tool("bridge")
        selector = getattr(self.builder_tab, "select_floor_plan_edge", None)
        selected = bool(selector(room, edge)) if callable(selector) else False
        context = (
            f"Edge mode: selected {room} edge {edge}. Use Bridge, Wall Opening, or Edge Extrude for KOTOR room seams."
        )
        self.workflow_panel.set_active_authoring_context(context)
        self.statusBar().showMessage(context)
        if selected:
            self._log(f"Map Studio selected floor-plan edge {edge} in {room}; Builder edge tools were synchronized.")
        else:
            self._log(
                f"Map Studio selected floor-plan edge {edge} in {room}, but Builder has no matching floor-plan room choice."
            )

    def _map_studio_export_dry_run_enabled(self) -> bool:
        """Return the current export dry-run preference from the Export panel."""

        dry_run = getattr(getattr(self, "export_panel", None), "dry_run", None)
        if dry_run is None:
            return True
        return bool(dry_run.isChecked())

    def _focus_map_studio_export_proof_workspace(self) -> None:
        """Focus the staged export/install/game-proof controls."""

        self._set_map_studio_workspace_combo_key("export")
        self.right_tabs.setCurrentWidget(self.map_studio_export_page)
        self.workflow_panel.set_active_authoring_context(
            "Export + Game Proof: validate, stage/install, warp test, then record proof"
        )

    def _select_map_studio_export_fix_target(self, target_id: str) -> None:
        """Select the authored object or entry-point controls named by export readiness."""

        target = str(target_id or "").strip()
        if not target:
            return
        if target == "entry_point":
            self._focus_map_studio_entry_point_controls()
            self.statusBar().showMessage("Focused Map Studio module entry point for the current PTH/WOK blocker.")
            return
        if target.startswith("authored:"):
            self.show_map_studio_placement_tools()
            self.select_item(target)
            try:
                self.viewport_panel.focus_selected()
            except Exception:
                pass
            self.workflow_panel.set_active_authoring_context(
                "Placement fix: move the selected authored resource onto generated walkable WOK, then Validate again."
            )
            self._log(f"Selected export blocker target {target}. Move it onto walkable WOK before staging.")
            return
        self.select_item(target)
        self._log(f"Selected export blocker target {target}.")

    def _retarget_map_studio_export_game(self, target_game: str) -> None:
        """Run the project port transaction from the Export workspace."""

        target = str(target_game or "").strip().upper()
        source = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        original_source = str(getattr(self.project, "source_game", source) or source).strip().upper()
        if target not in {"K1", "K2"}:
            self.export_panel.set_target_game(source, source_game=original_source)
            return
        if target == source:
            self.export_panel.set_target_game(target, source_game=original_source)
            return
        report = self.controller.record_port(source, target)
        if not bool(getattr(report, "ok", False)):
            self.export_panel.set_target_game(source, source_game=original_source)
            message = str(getattr(report, "message", "") or f"Could not retarget the authored KMAP to {target}.")
            QtWidgets.QMessageBox.warning(self, "Change KOTOR Target Game", message)
            self._log(message)
            return
        self._reset_map_studio_texture_paint_session()
        message = str(getattr(report, "message", "") or f"Retargeted the authored KMAP to {target}.")
        risks = tuple(str(item) for item in tuple(getattr(report, "unsupported", ()) or ()) if str(item).strip())
        self._refresh_all(message)
        self.statusBar().showMessage(message, 9000)
        for risk in risks:
            self._log(f"Target-game dependency risk: {risk}")

    def _show_map_studio_texture_paint_workflow(self) -> None:
        """Focus the paint workflow without treating it as a geometry command."""

        self.workflow_tabs.setCurrentWidget(self.texture_paint_tab)
        self.texture_paint_tab.set_project(self.project)
        self._sync_map_studio_texture_apply_state()
        self._set_map_studio_toolbar_edit_mode("Texture Paint")
        self.viewport_panel.set_map_studio_hover_probe(True, "object")
        self.texture_paint_tab.set_status(
            "Choose an editable room diffuse material, or import a project TGA and assign it to a visible room face."
        )

    def _handle_map_studio_tool_belt_action(self, action: Any) -> None:
        key = str(getattr(action, "key", "") or "")
        workspace_key = str(getattr(action, "workspace_key", "") or "")
        tool_key = str(getattr(action, "tool_key", "") or "")
        if key == "delete_selected":
            # The object-mode shelf command acts on the same complete scene
            # selection as Delete/Backspace.  Routing this through the
            # primitive dispatcher loses authored-room selections because
            # those rows do not have a composition primitive_name.
            self.delete_map_studio_current_selection()
            return
        if key == "duplicate_special_options":
            self._open_map_studio_modeling_tool_options(key)
            return
        if key in {
            "multi_cut",
            "target_weld",
            "make_hole",
            "connect_components",
            "make_live",
            "quad_draw",
            "select_triangles",
            "select_quads",
            "convert_contained_faces",
        }:
            self._run_map_studio_viewport_modeling_command(key)
            return
        if key == "texture_paint":
            self._show_map_studio_texture_paint_workflow()
            return
        if key in {
            "fill_hole",
            "mirror",
            "bridge",
            "bend_tool",
            "lattice",
            "wrap",
            "shrink_wrap",
            "soften_edges",
            "harden_edges",
            "reverse_normals",
            "insert_edge_loop",
            "merge_components",
            "boolean_a_minus_b",
        } and self._apply_map_studio_component_shelf_action(key):
            self._last_map_studio_modeling_action_key = key
            return
        if key == "combine" and self._combine_selected_authored_room_primitives():
            return
        if key == "separate" and self._separate_selected_authored_room_primitive():
            return
        if key == "plane" and str(self.map_studio_workspace_combo.currentData() or "") == "terrain":
            # In Terrain, Plane means a subdivided sculptable heightfield,
            # not the four-vertex platform used by Geometry blockout.
            self._create_and_focus_map_studio_terrain_patch()
            self.statusBar().showMessage(
                "Created a sculptable terrain plane. Drag in the viewport, then Generate Walkmesh.", 6000
            )
            return
        if key == "terrain_patch":
            self._create_and_focus_map_studio_terrain_patch()
            return
        route_context = self._map_studio_tool_action_context(key)
        route = resolve_map_studio_tool_belt_action(key, route_context)
        direct_command_actions = {
            "plane",
            "cube",
            "wall",
            "ramp",
            "stairs",
            "cylinder",
            "door_frame",
            "arch",
            "universal_transform",
            "reset_transform",
            "zero_pivot",
            "delete_history",
            "cleanup",
            "triangulate",
            "normals",
            "reverse_normals",
            "soften_edges",
            "harden_edges",
            "mirror",
            "mirror_x",
            "mirror_y",
            "mirror_z",
            "extrude",
            "bevel",
            "boolean",
            "boolean_a_minus_b",
            "boolean_b_minus_a",
            "blockout_room",
            "create_room",
            "corridor",
            "terrain_patch",
            "primitive",
            "cut",
            "split",
            "cut_slice_insert_edges",
            "insert_edge_loop",
            "fill",
            "fill_hole",
            "bridge",
            "shrink_wrap",
            "bend_tool",
            "curve_tool",
            "lattice",
            "wrap",
            "inset",
            "combine",
            "separate",
            "select",
            "move",
            "duplicate_selected",
            "delete_selected",
            "object_grid_snap",
            "object_vertex_snap",
            "center_pivot",
            "freeze_transform",
            "paint_material",
            "paint_wok",
            "duplicate_special",
            "vertex_snap",
            "grid_snap",
            "weld",
            "merge_components",
            "flatten",
            "transform_snap_level",
            "place",
            "entry_point",
            "placeable",
            "creature",
            "door",
            "waypoint",
            "trigger",
            "encounter",
            "sound",
            "camera",
            "store",
            "light",
            "script",
            "validate",
            "stage_module",
            "install_module",
            "launch_handoff",
            "record_proof",
            "opening",
            "opening_marker",
            "terrain",
            "walkmesh",
            "sculpt_raise",
            "sculpt_lower",
            "sculpt_smooth",
            "sculpt_flatten",
            "sculpt_erase",
            "sculpt_plateau",
            "sculpt_ramp",
            "sculpt_slope",
            "sculpt_terrace",
            "sculpt_pinch",
            "sculpt_erode",
            "sculpt_noise",
        }
        if key in direct_command_actions:
            if self._execute_map_studio_tool_belt_command(key):
                return
            if route.command_method:
                return
            if key == "opening_marker":
                self._focus_map_studio_opening_marker_controls()
                return
        terrain_brush = route.terrain_brush or self._map_studio_belt_terrain_brush(key)
        if terrain_brush:
            self.show_map_studio_terrain_tools()
            self._select_map_studio_terrain_brush(terrain_brush)
            self._sync_map_studio_terrain_brush_context(force_enabled=True)
            return
        primitive_kind = route.primitive_kind or self._map_studio_belt_primitive_kind(key)
        if primitive_kind:
            self.show_map_studio_geometry_tools()
            self.add_authored_room_primitive(primitive_kind, "")
            return
        placement_kind = route.placement_kind or self._map_studio_belt_placement_kind(key)
        if placement_kind:
            self.show_map_studio_placement_tools()
            self._select_map_studio_gameplay_kind(placement_kind)
            return
        if key in {
            "create_room",
            "primitive",
            "universal_transform",
            "reset_transform",
            "zero_pivot",
            "delete_history",
            "extrude",
            "bridge",
            "cut",
            "split",
            "cut_slice_insert_edges",
            "insert_edge_loop",
            "opening",
            "fill",
            "fill_hole",
            "vertex_snap",
            "grid_snap",
            "transform_snap_level",
            "weld",
            "merge_components",
            "flatten",
            "mirror",
            "mirror_x",
            "mirror_y",
            "mirror_z",
            "cleanup",
            "triangulate",
            "normals",
            "reverse_normals",
            "soften_edges",
            "harden_edges",
            "bevel",
            "boolean",
            "boolean_a_minus_b",
            "boolean_b_minus_a",
            "lattice",
            "wrap",
            "shrink_wrap",
            "duplicate_special",
            "curve_tool",
            "bend_tool",
            "combine",
            "separate",
        }:
            self.show_map_studio_geometry_tools()
            if tool_key:
                self._select_map_studio_modeling_tool(tool_key)
            if key == "universal_transform":
                self._focus_map_studio_universal_transform()
                return
            if key == "extrude":
                operation_combo = getattr(self.builder_tab, "roomOperationComboBox", None)
                if operation_combo is not None:
                    index = operation_combo.findData("edge_extrude")
                    if index >= 0:
                        operation_combo.setCurrentIndex(index)
                    operation_combo.setFocus()
            if key in {"cut", "split", "cut_slice_insert_edges", "insert_edge_loop"}:
                operation_combo = getattr(self.builder_tab, "roomOperationComboBox", None)
                if operation_combo is not None:
                    index = operation_combo.findData("split_x")
                    if index >= 0:
                        operation_combo.setCurrentIndex(index)
                    operation_combo.setFocus()
            if key in {"boolean", "boolean_a_minus_b", "boolean_b_minus_a"}:
                operation_combo = getattr(self.builder_tab, "roomOperationComboBox", None)
                if operation_combo is not None:
                    index = operation_combo.findData("rectangular_cut")
                    if index >= 0:
                        operation_combo.setCurrentIndex(index)
                    operation_combo.setFocus()
            if key == "bridge":
                tool = getattr(self.builder_tab, "floorPlanBridgeFirstRoomComboBox", None)
                if tool is not None:
                    tool.setFocus()
            if key == "opening":
                tool = getattr(self.builder_tab, "floorPlanOpeningRoomComboBox", None)
                if tool is not None:
                    tool.setFocus()
            if key == "combine":
                tool = getattr(self.builder_tab, "floorPlanUnionFirstRoomComboBox", None)
                self.workflow_panel.set_active_authoring_context(
                    "Combine Meshes: select at least two authored objects in one room. Their transforms are baked into "
                    "one polygon object while materials, UVs, normals, provenance, and disconnected shells remain intact."
                )
                self._log(
                    "Map Studio Combine Meshes is active. For rectangular room-footprint union, use the Floor Plan Union controls."
                )
                if tool is not None:
                    tool.setFocus()
            if key == "separate":
                tool = getattr(self.builder_tab, "roomPrimitiveTransformComboBox", None)
                self.workflow_panel.set_active_authoring_context(
                    "Separate Shells: select one Combined Mesh. Every disconnected polygon shell becomes an independently "
                    "selectable object in the same KOTOR room. Use Extract to Export Room only when you need a new room boundary."
                )
                self._log(
                    "Map Studio Separate Shells is active; this is polygon separation, not room extraction."
                )
                if tool is not None:
                    tool.setFocus()
            if key in {"vertex_snap", "grid_snap", "transform_snap_level", "weld", "merge_components", "flatten", "mirror", "mirror_x", "mirror_y", "mirror_z", "cleanup"}:
                self._focus_map_studio_vertex_workflow({
                    "merge_components": "weld",
                    "mirror_x": "mirror",
                    "mirror_y": "mirror",
                    "mirror_z": "mirror",
                }.get(key, key))
            return
        if workspace_key == "terrain":
            self.show_map_studio_terrain_tools()
        elif workspace_key == "walkmesh":
            self.show_map_studio_walkmesh_tools()
        elif workspace_key == "placements":
            self.show_map_studio_placement_tools()
        elif workspace_key == "lighting":
            self.show_map_studio_lighting_tools()
        elif workspace_key == "scripts":
            self.show_map_studio_script_tools()
        elif workspace_key == "export":
            self._focus_map_studio_export_proof_workspace()
        else:
            self.show_map_studio_builder()

    def show_map_studio_builder(self) -> None:
        """Focus the Builder tab inside the existing Map Studio Level Editor."""

        self._set_map_studio_workspace_combo_key("geometry")
        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        self.builder_tab.focus_build_section("room")
        self.workflow_panel.set_active_authoring_context("Room Building: primitives, floor plans, and magnetic assembly")
        self._sync_map_studio_terrain_brush_context(force_enabled=False)
        self._log("Map Studio Room Building focused.")

    def _open_map_studio_mode_from_viewport(self, mode_label: str) -> None:
        """Route the Map Studio viewport's mode belt into the owning workspace."""

        label = str(mode_label or "Object").strip() or "Object"
        self._set_map_studio_toolbar_edit_mode(label)
        self._handle_map_studio_edit_mode_changed(label)

    def _run_map_studio_maya_shortcut(self, action_key: str) -> None:
        """Run one conflict-free Maya shortcut while the viewport has focus."""

        key = str(action_key or "").strip()
        if not key:
            return
        self._run_map_studio_viewport_modeling_command(key)

    def _run_map_studio_bridge_or_fill_shortcut(self) -> None:
        """Match the user's contextual Ctrl+/ command: bridge two borders, otherwise fill."""

        selected = list(self.viewport_panel.map_studio_component_selection() or ())
        edges = [row for row in selected if str(row.get("component_type") or "") == "edge"]
        # A valid Fill Hole selection is itself a closed loop of three or more
        # border edges.  Treating every 2+ edge selection as Bridge made the
        # contextual shortcut unable to reach Fill Hole at all.
        self._run_map_studio_viewport_modeling_command("bridge" if len(edges) == 2 else "fill_hole")

    def _repeat_last_map_studio_modeling_command(self) -> None:
        """Maya G: repeat the most recently invoked repeatable modeling command."""

        key = str(getattr(self, "_last_map_studio_modeling_action_key", "") or "").strip()
        if not key:
            self.statusBar().showMessage("No Map Studio modeling command to repeat yet.", 2500)
            return
        self._run_map_studio_viewport_modeling_command(key)

    def _open_map_studio_modeling_tool_options(self, action_key: str) -> None:
        """Open persistent tool settings without running or committing the command."""

        key = str(action_key or "").strip()
        if self._edit_map_studio_baked_modeling_options(key):
            return
        if key in {"duplicate_special_options", "duplicate_special"}:
            self.show_map_studio_geometry_tools()
            control = getattr(self.builder_tab, "duplicateSpecialCountSpinBox", None)
            if control is not None:
                control.setFocus(QtCore.Qt.OtherFocusReason)
                control.selectAll()
            self.workflow_panel.set_active_authoring_context(
                "Duplicate Special Options: set copy count plus per-copy translation, Z rotation, and XYZ scale. "
                "Ctrl+Shift+D repeats the current settings on the selected object."
            )
            self.statusBar().showMessage("Duplicate Special options focused.", 3500)
            return
        if key == "bevel":
            if self.viewport_panel.arm_component_bevel():
                self.viewport_panel.bevel_width_spin.setFocus(QtCore.Qt.OtherFocusReason)
                self.viewport_panel.bevel_width_spin.selectAll()
                self.statusBar().showMessage(
                    "Bevel Tool Settings: width, segments, profile, miter, smoothing, UV mode, and overlap clamp are live.",
                    5000,
                )
            else:
                self.statusBar().showMessage("Select an editable imported-mesh edge before opening Bevel settings.", 4500)
            return
        if key == "extrude":
            self._run_map_studio_viewport_modeling_command("extrude")
            return
        action = self._map_studio_tool_action_for_key(key)
        if action is None:
            return
        route = resolve_map_studio_tool_belt_action(key, self._map_studio_tool_action_context(key))
        self.show_map_studio_geometry_tools()
        self.workflow_panel.set_active_authoring_context(
            route.authoring_context or f"{getattr(action, 'label', key)} Tool Settings"
        )
        self.statusBar().showMessage(
            f"{getattr(action, 'label', key)} settings are persistent; adjust the visible modeling controls, then work in the viewport.",
            4500,
        )

    def _run_map_studio_viewport_modeling_command(self, action_key: str) -> None:
        """Route the Map Studio-only viewport belt into the real tool dispatcher."""

        key = str(action_key or "").strip()
        if not key:
            return
        if key == "duplicate_special_options":
            self._open_map_studio_modeling_tool_options(key)
            return
        if key == "delete_history":
            component_selection = list(self.viewport_panel.map_studio_component_selection() or ())
            room = str(component_selection[0].get("room_resref") or "") if component_selection else ""
            role = str(component_selection[0].get("mesh_role") or "") if component_selection else ""
            if room:
                try:
                    self.controller.delete_authored_room_primitive_history(
                        room_resref=room,
                        primitive_name=room,
                    )
                except (ValueError, RuntimeError) as exc:
                    self.statusBar().showMessage(str(exc), 5500)
                else:
                    message = (
                        f"Deleted transient construction history for imported room {room}; "
                        "evaluated geometry and export provenance were retained."
                    )
                    self.statusBar().showMessage(message, 6000)
                    self._refresh_map_studio_imported_mesh_change(message, room, role)
                    self._last_map_studio_modeling_action_key = key
                return
        if key in {"select_triangles", "select_quads", "convert_contained_faces"}:
            selector = getattr(self.viewport_panel, key, None)
            count = int(selector() or 0) if callable(selector) else 0
            label = {
                "select_triangles": "triangle face",
                "select_quads": "quad region",
                "convert_contained_faces": "contained face",
            }[key]
            self._last_map_studio_modeling_action_key = key
            self.statusBar().showMessage(f"Selected {count} {label}{'' if count == 1 else 's'}.", 3500)
            return
        if key in {"multi_cut", "target_weld", "make_hole", "connect_components", "make_live", "quad_draw"}:
            activator = getattr(self.viewport_panel, "activate_map_studio_modeling_tool", None)
            if callable(activator) and activator(key):
                if key == "make_live":
                    self._capture_map_studio_live_wrap_driver_baseline()
                self._last_map_studio_modeling_action_key = key
                self.statusBar().showMessage(
                    f"{key.replace('_', ' ').title()} active. Work in the viewport; Enter commits where applicable and Esc exits.",
                    5000,
                )
            else:
                self.statusBar().showMessage(
                    f"{key.replace('_', ' ').title()} could not start in the current selection context.", 4500
                )
            # Persistent tools are owned by the viewport activator.  Never
            # fall through to the belt handler: it routes these same keys back
            # here and would recurse when activation is unavailable.
            return
        if key == "select":
            if not self.select_map_studio_authored_context():
                self._open_map_studio_mode_from_viewport("Object")
            return
        if key == "move":
            if not self.move_map_studio_authored_primitive_selection():
                self._open_map_studio_mode_from_viewport("Object")
            return
        if key == "duplicate_selected":
            if not self._execute_map_studio_tool_belt_command(key):
                self.duplicate_selected()
            return
        if key == "delete_selected":
            if not self._execute_map_studio_tool_belt_command(key):
                self.delete_selected()
            return
        action = self._map_studio_tool_action_for_key(key)
        if action is None:
            self.statusBar().showMessage(f"Map Studio modeling command '{key}' is unavailable.", 4000)
            return
        self._last_map_studio_modeling_action_key = key
        self._handle_map_studio_tool_belt_action(action)

    def focus_map_studio_modeling_workspace(self) -> None:
        """Open the Map Studio modeling controls from the main viewport affordance."""

        self.show_map_studio_builder()
        self._set_map_studio_toolbar_edit_mode("Object")
        self._select_map_studio_component_mode("object")
        self._select_map_studio_modeling_tool("primitive_room")
        component_combo = getattr(self.builder_tab, "componentModeComboBox", None)
        if component_combo is not None:
            component_combo.setFocus()
        self.statusBar().showMessage(
            "Map Studio Modeling ready: Object, Vertex, Edge, Face, Terrain, and Walkmesh tools author KMAP state.",
            5000,
        )
        self._log("Map Studio Modeling workspace focused from the main viewport toolbar.")

    def focus_map_studio_tutorial_workspace(self, route: str) -> None:
        """Focus a stable Map Studio workflow requested by the tutorial window."""

        key = str(route or "map_studio").strip().lower()
        if key in {"gmodeler", "modeling"}:  # legacy tutorial route remains readable
            self.show_map_studio_geometry_tools()
            self._set_map_studio_toolbar_edit_mode("Multi-Component")
            self._handle_map_studio_edit_mode_changed("Multi-Component")
        elif key == "terrain":
            self._set_map_studio_toolbar_edit_mode("Terrain")
            self._handle_map_studio_edit_mode_changed("Terrain")
        elif key == "texture_paint":
            self._show_map_studio_texture_paint_workflow()
            self._handle_map_studio_edit_mode_changed("Texture Paint")
        elif key == "game_proof":
            self._focus_map_studio_export_proof_workspace()
            self._set_map_studio_toolbar_edit_mode("Export")
        else:
            self.focus_map_studio_modeling_workspace()
        self.statusBar().showMessage(
            f"Tutorial workspace ready: {key.replace('_', ' ').title()}.",
            5000,
        )

    def _focus_map_studio_edit_mode_workspace(self, label: str) -> None:
        """Route the toolbar edit mode to the closest usable Map Studio workspace."""

        mode_key = str(label or "Object").strip().lower()
        if mode_key == "object":
            self.show_map_studio_builder()
            self._select_map_studio_component_mode("object")
            self._select_map_studio_modeling_tool("primitive_room")
            self.left_tabs.setCurrentWidget(self.outliner)
            return
        if mode_key == "vertex":
            self.show_map_studio_geometry_tools()
            self._select_map_studio_component_mode("vertex")
            self._select_map_studio_modeling_tool("weld_vertices")
            tool = getattr(self.builder_tab, "floorPlanVertexRoomComboBox", None)
            if tool is not None:
                tool.setFocus()
            return
        if mode_key == "edge":
            self.show_map_studio_geometry_tools()
            self._select_map_studio_component_mode("edge")
            self._select_map_studio_modeling_tool("bridge")
            tool = getattr(self.builder_tab, "floorPlanBridgeFirstRoomComboBox", None)
            if tool is not None:
                tool.setFocus()
            return
        if mode_key == "face":
            self.show_map_studio_geometry_tools()
            self._select_map_studio_component_mode("face")
            self._select_map_studio_modeling_tool("fill_face")
            tool = getattr(self.builder_tab, "fillFloorPlanFaceButton", None)
            if tool is not None:
                tool.setFocus()
            return
        if mode_key == "texture paint":
            self.workflow_tabs.setCurrentWidget(self.texture_paint_tab)
            self.texture_paint_tab.set_project(self.project)
            self._sync_map_studio_texture_apply_state()
            if self.texture_paint_tab.selected_texture_id():
                self.texture_paint_tab.paint_button.setChecked(True)
            else:
                self.texture_paint_tab.set_status("Import a project texture to begin Texture Paint mode.")
            return
        if mode_key == "walkmesh":
            self._select_map_studio_component_mode("walkmesh")
            self._select_map_studio_modeling_tool("paint_wok")
            self.show_map_studio_walkmesh_tools()
            return
        if mode_key == "placement":
            self.show_map_studio_placement_tools()
            return
        if mode_key == "terrain":
            self.show_map_studio_terrain_tools()
            self._select_map_studio_component_mode("terrain")
            self._select_map_studio_modeling_tool("terrain_sculpt")
            return
        if mode_key == "export":
            self._focus_map_studio_export_proof_workspace()
            return

    def _handle_map_studio_edit_mode_changed(self, mode: str) -> None:
        """Reflect the toolbar edit mode in the Map Studio workflow/readiness panel."""

        label = str(mode or "Object").strip() or "Object"
        if label != "Texture Paint" and self.texture_paint_tab.paint_button.isChecked():
            self.texture_paint_tab.stop_painting()
        self._sync_map_studio_tool_belt_preset_for_edit_mode(label)
        self._focus_map_studio_edit_mode_workspace(label)
        self._sync_map_studio_edit_mode_context(label)
        hover_modes = {
            "Object": "object",
            "Multi-Component": "",
            "Texture Paint": "object",
            "Vertex": "vertex",
            "Edge": "edge",
            "Face": "face",
            "Walkmesh": "walkmesh",
            "Terrain": "terrain",
        }
        self.viewport_panel.set_map_studio_hover_probe(label in hover_modes, hover_modes.get(label, ""))
        descriptions = {
            "Object": "select, move, duplicate, and organize rooms, placements, lights, and module objects",
            "Multi-Component": "automatically edit the nearest visible face, edge, or vertex; RMB opens the optional marking menu",
            "Texture Paint": "paint a unique project texture on the nearest visible face through diffuse UV0",
            "Vertex": "edit room and walkmesh vertices with snap, weld, flatten, mirror, and cleanup tools",
            "Edge": "edit seams, door or corridor borders, bridge edges, bevels, and rectangular cuts",
            "Face": "edit room faces, material intent, WOK surface intent, triangulation, and cleanup",
            "Walkmesh": "inspect and paint walkable, non-walkable, door, water, and transition faces",
            "Placement": "place and transform KOTOR creatures, placeables, doors, triggers, cameras, and waypoints",
            "Terrain": "sculpt terrain heightfields, ramps, plateaus, erosion, smoothing, and walkability",
            "Export": "validate, stage, install, hand off, warp-test, and record game proof",
        }
        context = f"{label} mode: {descriptions.get(label, 'author the active Map Studio selection')}"
        self.workflow_panel.set_active_authoring_context(context)
        self.statusBar().showMessage(f"Map Studio {context}", 5000)
        self._log(f"Map Studio edit mode changed: {context}")

    def show_map_studio_geometry_tools(self) -> None:
        """Focus Builder's primitive, operation, and modular room controls."""

        self._set_map_studio_workspace_combo_key("geometry")
        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        primitive = getattr(self.builder_tab, "roomPrimitivePresetComboBox", None)
        if primitive is not None:
            primitive.setFocus()
        self.workflow_panel.set_active_authoring_context(
            "Geometry: primitive rooms, extrusion, bevel/inset, rectangular cuts, boolean union, and modular room pieces"
        )
        self._log("Map Studio geometry tools focused. Use Builder to create rooms, edit primitives, apply bevels/cuts, and compose modular pieces.")

    def _reset_map_studio_workflow_scroll(self, _index: int = -1) -> None:
        """Show the top of each workflow instead of inheriting another tab's deep scroll."""

        scroll = getattr(self, "workflow_tabs_scroll", None)
        if scroll is None:
            return
        for bar in (scroll.horizontalScrollBar(), scroll.verticalScrollBar()):
            QtCore.QTimer.singleShot(0, lambda target=bar: target.setValue(target.minimum()))

    def _sync_map_studio_workflow_selector(self, index: int) -> None:
        """Keep programmatic workflow routing visible in the compact selector."""

        selector = getattr(self, "workflow_selector", None)
        if selector is not None and selector.currentIndex() != index:
            blocker = QtCore.QSignalBlocker(selector)
            selector.setCurrentIndex(index)
            del blocker
        stack = getattr(self, "workflow_tabs", None)
        if stack is not None:
            stack.updateGeometry()
        scroll = getattr(self, "workflow_tabs_scroll", None)
        if scroll is not None:
            scroll.updateGeometry()

    def _queue_map_studio_workflow_control_fit(self) -> None:
        """Elide narrow-rail controls after Qt has assigned their final widths."""

        if getattr(self, "_map_studio_workflow_fit_pending", False):
            return
        self._map_studio_workflow_fit_pending = True
        QtCore.QTimer.singleShot(0, self._fit_map_studio_workflow_controls)

    def _fit_map_studio_workflow_controls(self) -> None:
        """Keep long button labels readable instead of silently clipping them."""

        self._map_studio_workflow_fit_pending = False
        stack = getattr(self, "workflow_tabs", None)
        page = None if stack is None else stack.currentWidget()
        if page is None:
            return
        for control in page.findChildren(QtWidgets.QAbstractButton):
            if not control.isVisibleTo(page) or control.width() <= 0:
                continue
            current_text = str(control.text() or "")
            previous_rendered = control.property("_gr_workflow_rendered_text")
            full_text = control.property("_gr_workflow_full_text")
            if full_text is None or (previous_rendered is not None and current_text != previous_rendered):
                full_text = current_text
                control.setProperty("_gr_workflow_full_text", full_text)
            full_text = str(full_text or "")
            if not full_text.strip():
                continue
            padding = 18
            if isinstance(control, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                padding += control.style().pixelMetric(QtWidgets.QStyle.PM_IndicatorWidth, None, control) + 4
            elif not control.icon().isNull():
                padding += control.iconSize().width() + 4
            available = max(12, control.width() - padding)
            rendered = control.fontMetrics().elidedText(full_text, QtCore.Qt.ElideRight, available)
            if current_text != rendered:
                control.setText(rendered)
            control.setProperty("_gr_workflow_rendered_text", rendered)
            if not control.accessibleName():
                control.setAccessibleName(full_text.replace("&", ""))
            if rendered != full_text and not control.toolTip():
                control.setToolTip(full_text.replace("&", ""))

    def show_map_studio_walkmesh_tools(self) -> None:
        """Focus the existing Walkmesh tab inside the Map Studio Level Editor."""

        self._set_map_studio_workspace_combo_key("walkmesh")
        self.workflow_tabs.setCurrentWidget(self.walkmesh_tab)
        self.workflow_panel.set_active_authoring_context("Walkmesh: inspect and paint walkable/non-walkable faces")
        self._log("Map Studio Walkmesh tools focused. Use these to inspect, load, or paint walkable faces.")

    def show_map_studio_terrain_tools(self) -> None:
        """Focus Build / Terrain and enable direct viewport sculpting."""

        self._set_map_studio_workspace_combo_key("terrain")
        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        self.builder_tab.focus_build_section("terrain")
        self._select_map_studio_component_mode("terrain")
        terrain = getattr(self.builder_tab, "terrainRoomComboBox", None)
        if terrain is not None:
            terrain.setFocus()
        self.workflow_panel.set_active_authoring_context("Terrain: sculpt heightfield samples and slope/walkability")
        self._sync_map_studio_terrain_brush_context(force_enabled=True)
        self._log("Map Studio Terrain Building focused. Drag on the selected terrain surface to sculpt it.")

    def _on_map_studio_build_section_changed(self, section: str) -> None:
        """Keep viewport interaction aligned with the visible Build section."""

        key = str(section or "room").strip().lower()
        self._reset_map_studio_workflow_scroll()
        if key == "terrain":
            self.builder_tab.buildingToolButtons["select"].setChecked(True)
            self.viewport_panel.set_pascal_building_tool("select")
            self._select_map_studio_component_mode("terrain")
            self._sync_map_studio_terrain_brush_context(force_enabled=True)
            self.workflow_panel.set_active_authoring_context(
                "Terrain Building: sculpt, dress, and paint outdoor surfaces"
            )
        elif key == "skybox":
            self.builder_tab.buildingToolButtons["select"].setChecked(True)
            self.viewport_panel.set_pascal_building_tool("select")
            self._sync_map_studio_terrain_brush_context(force_enabled=False)
            self.workflow_panel.set_active_authoring_context(
                "Skybox: build a KOTOR sky dome from textures or an HDR/panorama"
            )
        else:
            self._sync_map_studio_terrain_brush_context(force_enabled=False)
            self.workflow_panel.set_active_authoring_context(
                "Room Building: draw closed wall loops, add doors/windows, and organize levels"
            )

    def _refresh_map_studio_spatial_design_overlay(self, _visible: bool | None = None) -> None:
        """Refresh only the purposeful grid plan layered over authored markers."""

        placements = tuple(self.controller.authored_gameplay_placements() or ())
        geometry = self.controller.authored_editor_marker_geometry(
            include_spatial_design=self.builder_tab.spatial_plan_overlay_enabled()
        )
        self.viewport_panel.set_authored_gameplay_markers(
            placements,
            self.controller.authored_gameplay_fallback_preview_markers(),
            geometry,
        )

    def _audit_map_studio_spatial_design(self) -> None:
        """Expose a concise designer-facing audit without hiding it in export text."""

        audit = self.controller.audit_map_studio_spatial_design()
        ledger = self.controller.map_studio_spatial_design_placement_ledger()
        self.builder_tab.set_spatial_design_context(audit, ledger)
        if audit.ok:
            message = audit.summary()
        else:
            first = next(iter(audit.blocking_issues), "Create a spatial plan for this map.")
            message = f"{audit.summary()} · {first}"
        self.workflow_panel.set_active_authoring_context(message)
        self._log(message)

    def show_map_studio_lighting_tools(self) -> None:
        """Focus Environment's authored and ARE lighting controls."""

        self._set_map_studio_workspace_combo_key("lighting")
        self.workflow_tabs.setCurrentWidget(self.environment_tab)
        name = getattr(self.builder_tab, "roomLightNameLineEdit", None)
        if name is not None:
            name.setFocus()
            name.selectAll()
        self.workflow_panel.set_active_authoring_context("Lighting: add authored room lights before lightmap/export checks")
        self._log("Map Studio lighting tools focused. Add authored room lights before staging lightmap-ready test builds.")

    def show_map_studio_placement_tools(self) -> None:
        """Focus the direct-placement workspace."""

        self._set_map_studio_workspace_combo_key("placements")
        self.workflow_dock.show()
        self.workflow_dock.raise_()
        self.workflow_tabs.setCurrentWidget(self.placement_tab)
        self.placement_tab.search_edit.setFocus()
        self.placement_tab.search_edit.selectAll()
        self.workflow_panel.set_active_authoring_context(
            "Placement: drag a KOTOR asset from the list onto the exact level surface, then use W/E to adjust."
        )
        self._log("Map Studio placement workspace focused. Drag an asset from the list and release it on the level.")

    def show_map_studio_script_tools(self) -> None:
        """Focus Data's authored module/area script-hook controls."""

        self._set_map_studio_workspace_combo_key("scripts")
        self.workflow_tabs.setCurrentWidget(self.blueprints_tab)
        script = getattr(self.builder_tab, "scriptHookResrefLineEdit", None)
        if script is not None:
            script.setFocus()
            script.selectAll()
        self.workflow_panel.set_active_authoring_context("Scripts: assign ARE/IFO script hook resrefs")
        self._log("Map Studio script-hook tools focused. Assign ARE/IFO script resrefs that resolve from the package, Override, or base game.")

    def add_map_studio_test_placeable(self) -> None:
        """Add a known-safe test placeable through the existing authored placement service."""

        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        self.add_authored_gameplay_placement(
            "placeable",
            "plc_bench",
            "map_studio_test_placeable",
            1.75,
            1.5,
            0.0,
            0.0,
        )

    def add_map_studio_camera(self) -> None:
        """Add an authored camera marker through the existing gameplay placement service."""

        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        camera_count = sum(
            1
            for placement in self.controller.authored_gameplay_placements()
            if str(getattr(placement, "kind", "") or "").lower() == "camera"
        )
        self.add_authored_gameplay_placement(
            "camera",
            "",
            str(camera_count + 1),
            0.0,
            -2.5,
            1.6,
            0.0,
        )
        self.workflow_panel.set_active_authoring_context(
            "Camera: authored camera marker added. Move it in Properties or the viewport, then validate before export."
        )

    def add_map_studio_room_light(self) -> None:
        """Add an authored room light through the room-light service."""

        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        light_count = len(tuple(self.controller.authored_room_lights() or ()))
        self.add_authored_room_light(
            "",
            f"key_light_{light_count + 1}",
            0.0,
            0.0,
            2.25,
            1.0,
            0.92,
            0.78,
            8.0,
            1.0,
            "point",
        )
        self.workflow_panel.set_active_authoring_context(
            "Lighting: authored room light added. Tune color, radius, and position before export/lightmap checks."
        )

    def set_authored_module_entry_point(
        self,
        area_resref: str,
        x: float,
        y: float,
        z: float,
        facing: float,
    ) -> None:
        """Update the authored module IFO player start from Builder controls."""

        try:
            self.controller.set_authored_module_entry_point(
                area_resref=area_resref,
                position=(x, y, z),
                facing=facing,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Set Module Entry Point", str(exc))
            return
        self._refresh_all("Updated Map Studio module entry point/player start.")

    def create_map_studio_starter_room(self) -> None:
        """Create a small authored room through the existing Builder preset path."""

        self._create_map_studio_starter_preset(
            preset_id="rectangular_dev_room",
            module_root="grdev01",
            label="starter room",
        )

    def create_map_studio_doorway_blockout(self) -> None:
        """Create a doorway-focused authored room through the Builder preset path."""

        self._create_map_studio_starter_preset(
            preset_id="doorway_blockout",
            module_root="grdoor",
            label="doorway blockout",
        )

    def create_map_studio_corridor(self) -> None:
        """Create a corridor/hall authored room through the Builder preset path."""

        self._create_map_studio_starter_preset(
            preset_id="wide_hall",
            module_root="grhall",
            label="corridor",
        )

    def create_map_studio_starter_terrain(self) -> None:
        """Create a terrain authored module through the existing Builder preset path."""

        self._create_map_studio_starter_preset(
            preset_id="terrain_heightfield",
            module_root="grterrain",
            label="terrain patch",
        )

    def _create_map_studio_starter_preset(self, *, preset_id: str, module_root: str, label: str) -> None:
        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        module_root_edit = getattr(self.builder_tab, "moduleRootLineEdit", None)
        if module_root_edit is not None:
            module_root_edit.setText(module_root)
        self._log(f"Creating Map Studio {label} from Builder preset {preset_id}.")
        self.create_authored_room_preset(preset_id, module_root)

    def build_module_files(self) -> None:
        if self._map_studio_pie_session is not None:
            self._stop_map_studio_pie()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self._last_output_dir or "")
        if not path:
            return
        try:
            result = self.controller.generate_module_files(path)
        except Exception as exc:
            message = f"Module build blocked: {exc}"
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 8000)
            self._log(message)
            QtWidgets.QMessageBox.warning(self, "Build Module", message)
            return
        self._last_output_dir = str(getattr(result, "output_dir", "") or path)
        self._log(result.message)
        if getattr(result, "module_path", ""):
            self._log(f"Package: {result.module_path}")
        if result.manifest_path:
            self._log(f"Manifest: {result.manifest_path}")
        if getattr(result, "module_root", "") and hasattr(result, "metadata"):
            for line in authored_module_smoke_summary_lines(result):
                self._log(line)
        for warning in getattr(result, "warnings", ()):
            self._log(f"Warning: {warning}")
        for issue in getattr(result, "blocking_issues", ()):
            self._log(f"Blocking: {issue}")
        self._refresh_all("Module files generated.")

    def stage_dev_test_module(self, dry_run: bool = False) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Stage grdev01 dev test module", self._last_output_dir or "")
        if not path:
            return
        try:
            result = self.controller.stage_dev_test_module(path, dry_run=dry_run)
        except Exception as exc:
            message = f"Dev-test staging blocked: {exc}"
            self.texture_paint_tab.set_status(message)
            self.statusBar().showMessage(message, 8000)
            self._log(message)
            QtWidgets.QMessageBox.warning(self, "Stage Dev Test Module", message)
            return
        self._last_output_dir = path
        self._log(result.message)
        export_result = result.export_result
        if export_result is not None:
            if export_result.module_path:
                self._log(f"Package: {export_result.module_path}")
            if export_result.manifest_path:
                self._log(f"Manifest: {export_result.manifest_path}")
        if result.checklist_path:
            self._log(f"Game-test checklist: {result.checklist_path}")
        if result.proof_manifest_path:
            self._log(f"Proof manifest: {result.proof_manifest_path}")
        if getattr(result, "launch_helper_command", ""):
            self._log(f"Launch dry-run helper: {result.launch_helper_command}")
        if getattr(result, "elevated_launch_script_path", ""):
            self._log(f"Elevated launch helper: {result.elevated_launch_script_path}")
        if getattr(result, "proof_recording_script_path", ""):
            self._log(f"Proof recorder: {result.proof_recording_script_path}")
        for warning in result.warnings:
            self._log(f"Warning: {warning}")
        for issue in result.blocking_issues:
            self._log(f"Blocking: {issue}")

    def export_authored_module(self, dry_run: bool = False) -> None:
        values = self._open_map_studio_package_wizard(mode="export", dry_run=dry_run)
        if values is None:
            return
        path = str(values.get("output_dir") or "")
        try:
            self._sync_placeable_library_resources_for_export()
            self._sync_authored_creature_behavior_resources_for_export()
            result = self.controller.export_authored_module(path, dry_run=bool(values.get("dry_run")))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Authored Module", str(exc))
            return
        self._last_output_dir = path
        self._log(result.message)
        if result.module_path:
            self._log(f"Package: {result.module_path}")
        if result.manifest_path:
            self._log(f"Manifest: {result.manifest_path}")
        for line in authored_module_smoke_summary_lines(result):
            self._log(line)
        for warning in result.warnings:
            self._log(f"Warning: {warning}")
        for issue in result.blocking_issues:
            self._log(f"Blocking: {issue}")
        self._refresh_all("Authored module export state updated.")

    def stage_authored_module(self, dry_run: bool = False) -> None:
        values = self._open_map_studio_package_wizard(mode="stage", dry_run=dry_run)
        if values is None:
            return
        path = str(values.get("output_dir") or "")
        try:
            self._sync_placeable_library_resources_for_export()
            self._sync_authored_creature_behavior_resources_for_export()
            result = self.controller.stage_authored_module(path, dry_run=bool(values.get("dry_run")))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Stage Authored Module", str(exc))
            return
        self._last_output_dir = path
        self._log_authored_module_stage_result(result)
        self._refresh_all("Authored module game-test staging updated.")

    def _log_authored_module_stage_result(self, result: Any) -> None:
        """Log staged authored-module export, checklist, and proof handoff details."""

        self._log(result.message)
        export_result = result.export_result
        if export_result is not None:
            if export_result.module_path:
                self._log(f"Package: {export_result.module_path}")
            if export_result.manifest_path:
                self._log(f"Manifest: {export_result.manifest_path}")
        if result.installed_module_path:
            self._log(f"Installed module: {result.installed_module_path}")
        if result.backup_module_path:
            self._log(f"Backup module: {result.backup_module_path}")
        if result.checklist_path:
            self._log(f"Game-test checklist: {result.checklist_path}")
        if result.proof_manifest_path:
            self._log(f"Proof manifest: {result.proof_manifest_path}")
        if export_result is not None:
            for line in authored_module_smoke_summary_lines(export_result):
                self._log(line)
        for warning in result.warnings:
            self._log(f"Warning: {warning}")
        for issue in result.blocking_issues:
            self._log(f"Blocking: {issue}")

    def install_authored_module(self, dry_run: bool = False) -> None:
        values = self._open_map_studio_package_wizard(mode="install", dry_run=dry_run)
        if values is None:
            return
        path = str(values.get("output_dir") or "")
        modules_path = str(values.get("game_modules_dir") or "")
        overwrite = bool(values.get("overwrite"))
        try:
            self._sync_placeable_library_resources_for_export()
            self._sync_authored_creature_behavior_resources_for_export()
            result = self.controller.stage_authored_module(
                path,
                dry_run=bool(values.get("dry_run")),
                game_modules_dir=modules_path,
                overwrite=overwrite,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Install Authored Module", str(exc))
            return
        self._last_output_dir = path
        self._last_game_modules_dir = modules_path
        self._log_authored_module_stage_result(result)
        if not result.ok:
            QtWidgets.QMessageBox.warning(self, "Install Authored Module", result.message)
        self._refresh_all("Authored module game-test install updated.")

    def record_game_smoke_proof(self) -> None:
        try:
            proof_defaults = self.controller.map_studio_game_proof_recording_handoff()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Record Game Smoke Proof", str(exc))
            return
        self._focus_map_studio_export_proof_workspace()
        self._record_game_smoke_proof_from_summary(proof_defaults)

    def _record_game_smoke_proof_from_summary(self, proof_defaults: Any) -> bool:
        """Open the proof dialog using the controller's proof-recording defaults."""

        blockers = tuple(getattr(proof_defaults, "blocking_messages", ()) or ())
        for blocker in blockers:
            self._log(f"Proof recording setup: {blocker}")
        for warning in tuple(getattr(proof_defaults, "warnings", ()) or ()):
            self._log(f"Warning: {warning}")
        default_manifest = str(getattr(proof_defaults, "proof_manifest_path", "") or "")
        dialog = _MapStudioGameProofDialog(
            self,
            proof_manifest_path=default_manifest,
            package_resource_summary=str(getattr(proof_defaults, "package_resource_summary", "") or ""),
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return False
        values = dialog.values()
        if not values["proof_manifest_path"]:
            QtWidgets.QMessageBox.warning(self, "Record Game Smoke Proof", "Choose the proof manifest written by the Map Studio stage action.")
            return False
        if not values["evidence_path"] and not values["allow_missing_evidence"]:
            QtWidgets.QMessageBox.warning(self, "Record Game Smoke Proof", "Choose screenshot or video evidence from the actual KOTOR test.")
            return False
        result = self.controller.record_map_studio_game_proof(**values)
        self._log(result.message)
        if getattr(result, "proof_manifest_path", ""):
            self._log(f"Proof manifest: {result.proof_manifest_path}")
        if getattr(result, "pack_manifest_path", ""):
            self._log(f"Pack manifest: {result.pack_manifest_path}")
        if getattr(result, "evidence_path", ""):
            self._log(f"Evidence: {result.evidence_path}")
        for warning in getattr(result, "warnings", ()):
            self._log(f"Warning: {warning}")
        for issue in getattr(result, "blocking_issues", ()):
            self._log(f"Blocking: {issue}")
        if not getattr(result, "ok", False):
            QtWidgets.QMessageBox.warning(self, "Record Game Smoke Proof", result.message)
            return False
        self._refresh_all("Map Studio game proof updated.")
        return True

    def open_map_studio_launch_handoff(self) -> None:
        try:
            handoff = self.controller.map_studio_launch_handoff()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Launch Handoff", str(exc))
            return
        self._focus_map_studio_export_proof_workspace()
        self._open_map_studio_launch_handoff_dialog_from_summary(handoff)

    def _open_map_studio_launch_handoff_dialog_from_summary(self, handoff: Any) -> None:
        """Open the launch/proof handoff dialog from the controller's non-mutating summary."""

        blockers = tuple(getattr(handoff, "blocking_messages", ()) or ())
        if blockers or not bool(getattr(handoff, "ready", False)):
            message = "\n".join(blockers) if blockers else "Stage or install an authored module game-test package first."
            QtWidgets.QMessageBox.information(self, "Open Launch Handoff", message)
            self._log(f"Launch handoff not ready: {message}")
            return
        launcher_path = Path(str(getattr(handoff, "launcher_path", "") or getattr(handoff, "elevated_launch_script_path", "") or ""))
        proof_path = Path(str(getattr(handoff, "proof_manifest_path", "") or ""))
        proof_recorder_path = Path(str(getattr(handoff, "proof_recording_script_path", "") or ""))
        dialog = _MapStudioLaunchHandoffDialog(
            self,
            warp_command=str(getattr(handoff, "warp_command", "") or "warp <module>"),
            launcher_path=str(launcher_path) if launcher_path.is_file() else "",
            proof_manifest_path=str(proof_path) if proof_path.is_file() else "",
            proof_recording_script_path=str(proof_recorder_path) if proof_recorder_path.is_file() else "",
            launch_helper_command=str(getattr(handoff, "launch_helper_command", "") or ""),
            package_resource_summary=str(getattr(handoff, "package_resource_summary", "") or ""),
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        if launcher_path.is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(launcher_path)))
            self._log(f"Opened launch handoff: {launcher_path}")
        elif proof_path.is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(proof_path.parent)))
            self._log(f"Opened proof folder: {proof_path.parent}")
        for warning in tuple(getattr(handoff, "warnings", ()) or ()):
            self._log(f"Warning: {warning}")
        self._log(f"Map Studio warp command: {getattr(handoff, 'warp_command', 'warp <module>')}")

    def create_authored_room_preset(self, preset_id: str, module_root: str) -> None:
        try:
            result = self.controller.create_authored_room_preset_module(preset_id=preset_id, module_root=module_root)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Create Authored Room Primitive", str(exc))
            return
        readiness = result.readiness
        message = f"Created authored module {self.project.name} from primitive preset {preset_id}."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_all(message)

    def apply_authored_room_operation(
        self,
        operation: str,
        distance: float,
        edge_index: int,
        cut_center_x: float,
        cut_center_y: float,
        cut_width: float,
        cut_depth: float,
    ) -> None:
        try:
            if operation == "rectangular_cut":
                result = self.controller.apply_authored_room_operation(
                    operation=operation,
                    center=(cut_center_x, cut_center_y),
                    size=(cut_width, cut_depth),
                )
            elif operation in {"split_x", "split_y"}:
                result = self.controller.apply_authored_room_operation(
                    operation=operation,
                    axis="x" if operation == "split_x" else "y",
                    coordinate=cut_center_x if operation == "split_x" else cut_center_y,
                )
            elif operation == "edge_extrude":
                result = self.controller.apply_authored_room_operation(
                    operation=operation,
                    distance=distance,
                    edge_index=edge_index,
                )
            else:
                result = self.controller.apply_authored_room_operation(operation=operation, distance=distance)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Room Operation", str(exc))
            return
        readiness = result.readiness
        message = f"Applied room operation {operation}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )

    def apply_authored_floor_plan_extrusion(
        self,
        room_resref: str,
        z: float,
        wall_height: float,
        include_walls: bool,
        floor_surface_id: str,
    ) -> None:
        try:
            result = self.controller.set_authored_floor_plan_extrusion(
                room_resref=room_resref,
                z=z,
                wall_height=wall_height,
                include_walls=include_walls,
                floor_surface_id=floor_surface_id,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Floor-Plan Extrusion", str(exc))
            return
        readiness = result.readiness
        message = f"Updated floor-plan extrusion for {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True)

    def set_authored_floor_plan_wall_opening(
        self,
        room_resref: str,
        name: str,
        edge_index: int,
        center_fraction: float,
        width: float,
        height: float,
        bottom: float,
    ) -> None:
        try:
            result = self.controller.apply_authored_room_operation(
                operation="wall_opening",
                room_resref=room_resref,
                name=name,
                edge_index=edge_index,
                center_fraction=center_fraction,
                width=width,
                height=height,
                bottom=bottom,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Floor-Plan Wall Opening", str(exc))
            return
        readiness = result.readiness
        opening_label = name or f"edge {edge_index}"
        message = f"Updated wall opening {opening_label} in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True)

    def create_authored_opening_transition_marker(
        self,
        room_resref: str,
        opening_name: str,
        marker_kind: str,
        template_resref: str,
        tag: str,
        linked_to: str,
        linked_to_module: str,
        linked_to_flags: int,
        transition_destination: int,
    ) -> None:
        try:
            result = self.controller.apply_authored_room_operation(
                operation="opening_transition_marker",
                room_resref=room_resref,
                opening_name=opening_name,
                marker_kind=marker_kind,
                template_resref=template_resref,
                tag=tag,
                linked_to=linked_to,
                linked_to_module=linked_to_module,
                linked_to_flags=linked_to_flags,
                transition_destination=transition_destination,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Create Opening Transition Marker", str(exc))
            return
        readiness = result.readiness
        marker_label = tag or opening_name or marker_kind
        message = f"Created {marker_kind} marker {marker_label} from wall opening; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_all(message)

    def bridge_authored_floor_plan_edges(
        self,
        first_room_resref: str,
        first_edge_index: int,
        second_room_resref: str,
        second_edge_index: int,
        result_room_resref: str,
    ) -> None:
        try:
            result = self.controller.bridge_authored_floor_plan_edges(
                first_room_resref=first_room_resref,
                first_edge_index=first_edge_index,
                second_room_resref=second_room_resref,
                second_edge_index=second_edge_index,
                result_room_resref=result_room_resref,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Bridge Floor-Plan Edges", str(exc))
            return
        readiness = result.readiness
        message = (
            f"Bridged floor-plan edge {first_edge_index} in {first_room_resref} "
            f"to edge {second_edge_index} in {second_room_resref}; previous exports/proofs are now stale."
        )
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )

    def snap_authored_floor_plan_vertex(
        self,
        room_resref: str,
        point_index: int,
        target_point_index: int,
        target_room_resref: str,
    ) -> None:
        try:
            result = self.controller.snap_authored_floor_plan_vertex(
                room_resref=room_resref,
                point_index=point_index,
                target_point_index=target_point_index,
                target_room_resref=target_room_resref,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Snap Floor-Plan Vertex", str(exc))
            return
        readiness = result.readiness
        target_label = target_room_resref or room_resref
        message = (
            f"Snapped floor-plan point {point_index} in {room_resref} to point {target_point_index} in {target_label}; "
            "previous exports/proofs are now stale."
        )
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def preview_authored_floor_plan_vertex_snap_candidates(self, room_resref: str, point_index: int) -> None:
        """Show nearest non-mutating floor-plan vertex snap candidates."""

        setter = getattr(self.builder_tab, "set_floor_plan_vertex_snap_candidates", None)
        viewport_setter = getattr(self.viewport_panel, "set_room_outline_vertex_snap_candidates", None)
        if not callable(setter) and not callable(viewport_setter):
            return
        try:
            candidates = self.controller.authored_floor_plan_vertex_snap_candidates(
                room_resref=room_resref,
                point_index=int(point_index),
                limit=4,
            )
        except Exception:
            return
        if callable(setter):
            setter(candidates)
        if callable(viewport_setter):
            viewport_setter(room_resref, int(point_index), candidates)

    def weld_authored_floor_plan_vertices(
        self,
        room_resref: str,
        point_indices: object,
        target_point_index: int,
        position_policy: str,
    ) -> None:
        try:
            result = self.controller.weld_authored_floor_plan_vertices(
                room_resref=room_resref,
                point_indices=tuple(point_indices or ()),
                target_point_index=target_point_index,
                position_policy=position_policy,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Weld Floor-Plan Vertices", str(exc))
            return
        readiness = result.readiness
        message = f"Welded floor-plan vertices in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def flatten_authored_floor_plan_vertices(
        self,
        room_resref: str,
        point_indices: object,
        axis: str,
        value: object,
    ) -> None:
        try:
            flatten_value = None if value is None else float(value)
            result = self.controller.flatten_authored_floor_plan_vertices(
                room_resref=room_resref,
                point_indices=tuple(point_indices or ()),
                axis=axis,
                value=flatten_value,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Flatten Floor-Plan Vertices", str(exc))
            return
        readiness = result.readiness
        message = f"Flattened floor-plan vertices in {room_resref} on {axis}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def cleanup_authored_floor_plan_vertices(
        self,
        room_resref: str,
        tolerance: float,
    ) -> None:
        try:
            result = self.controller.cleanup_authored_floor_plan_vertices(
                room_resref=room_resref,
                tolerance=float(tolerance),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cleanup Floor-Plan Vertices", str(exc))
            return
        readiness = result.readiness
        message = f"Cleaned redundant floor-plan vertices in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def mirror_authored_floor_plan_vertices(
        self,
        room_resref: str,
        axis: str,
    ) -> None:
        try:
            result = self.controller.mirror_authored_floor_plan_vertices(
                room_resref=room_resref,
                axis=axis,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Mirror Floor-Plan Footprint", str(exc))
            return
        readiness = result.readiness
        message = f"Mirrored floor-plan footprint in {room_resref} across local {axis}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def fill_authored_floor_plan_face(
        self,
        room_resref: str,
        point_indices: object,
    ) -> None:
        try:
            result = self.controller.fill_authored_floor_plan_face(
                room_resref=room_resref,
                point_indices=tuple(point_indices or ()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Fill Floor-Plan Face", str(exc))
            return
        readiness = result.readiness
        message = f"Filled floor-plan face loop in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def split_authored_floor_plan_face(
        self,
        room_resref: str,
        point_indices: object,
    ) -> None:
        try:
            result = self.controller.split_authored_floor_plan_face(
                room_resref=room_resref,
                point_indices=tuple(point_indices or ()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Split Floor-Plan Face", str(exc))
            return
        readiness = result.readiness
        message = f"Split floor-plan face in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def triangulate_authored_floor_plan_face(
        self,
        room_resref: str,
    ) -> None:
        try:
            result = self.controller.triangulate_authored_floor_plan_face(
                room_resref=room_resref,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Triangulate Floor-Plan Face", str(exc))
            return
        readiness = result.readiness
        message = f"Triangulated floor-plan face in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def cleanup_authored_floor_plan_normals(
        self,
        room_resref: str,
    ) -> None:
        try:
            result = self.controller.cleanup_authored_floor_plan_normals(
                room_resref=room_resref,
                positive_z=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cleanup Floor-Plan Normals", str(exc))
            return
        readiness = result.readiness
        message = f"Cleaned floor-plan normals in {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True, refresh_connections=True)

    def apply_authored_terrain_operation(
        self,
        operation: str,
        room_resref: str,
        row_index: int,
        column_index: int,
        height: float,
        delta: float,
        radius: int,
        iterations: int,
        strength: float,
    ) -> None:
        try:
            result = self.controller.apply_authored_terrain_operation(
                operation=operation,
                room_resref=room_resref,
                row_index=row_index,
                column_index=column_index,
                height=height,
                delta=delta,
                radius=radius,
                iterations=iterations,
                strength=strength,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Terrain Operation", str(exc))
            return
        readiness = result.readiness
        message = f"Applied terrain operation {operation} to {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
        )

    def preview_map_studio_terrain_sculpt_frame(
        self,
        brush: str,
        room_resref: str,
        row_index: int,
        column_index: int,
        height: float,
        delta: float,
        radius: int,
        iterations: int,
        strength: float,
    ) -> None:
        try:
            context_getter = getattr(self.builder_tab, "current_terrain_brush_context", None)
            brush_context = context_getter() if callable(context_getter) else {}
            if not isinstance(brush_context, dict):
                brush_context = {}
            performance_policy = self.controller.map_studio_viewport_performance_policy()
            frame = self.controller.prepare_map_studio_terrain_sculpt_frame(
                room_resref=room_resref,
                brush=brush,
                points=((int(row_index), int(column_index), 1.0),),
                delta=delta,
                radius=radius,
                height=height,
                iterations=iterations,
                strength=strength,
                falloff_hardness=float(brush_context.get("hardness", 0.5) or 0.5),
                budget_ms=float(getattr(performance_policy, "terrain_brush_budget_ms", 4.0) or 4.0),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Check Live Terrain Brush Frame", str(exc))
            return
        status = (
            f"Live terrain frame: {frame.performance.estimated_apply_ms:.3f} ms / "
            f"{frame.performance.budget_ms:.3f} ms, {frame.performance.affected_sample_count} sample(s), "
            f"{'ready' if frame.should_apply_live else 'too heavy'}; full MDL/WOK rebuild deferred."
        )
        if getattr(frame, "warnings", ()):
            status = f"{status} {frame.warnings[0]}"
        label = getattr(self.builder_tab, "terrainBrushStatusLabel", None)
        if label is not None:
            label.setText(status)
        self._log(status)

    def _create_and_focus_map_studio_terrain_patch(self) -> None:
        """Create a terrain patch and drop straight into terrain painting."""

        try:
            resref = self.controller.create_terrain_patch()
        except Exception as exc:
            self.statusBar().showMessage(f"Terrain patch could not be created: {exc}", 6000)
            self._log(f"Map Studio terrain patch failed: {exc}")
            return
        self._refresh_map_studio_geometry_change(
            f"Created terrain patch {resref}.",
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )
        self.show_map_studio_terrain_tools()
        combo = getattr(self.builder_tab, "terrainRoomComboBox", None)
        if combo is not None:
            index = combo.findData(resref)
            if index < 0:
                index = combo.findText(resref, QtCore.Qt.MatchContains)
            if index >= 0:
                combo.setCurrentIndex(index)
        self._select_map_studio_terrain_brush("raise")
        self._sync_map_studio_terrain_brush_context(force_enabled=True)
        self.toolbar.selection_mode.setCurrentText("Terrain")
        self.statusBar().showMessage(
            f"Terrain patch {resref} ready — drag in the viewport to paint. Pick a brush (raise/lower/smooth/flatten).",
            8000,
        )

    def _sync_map_studio_terrain_brush_context(self, *, force_enabled: bool | None = None) -> None:
        """Keep the viewport terrain brush state aligned with Builder controls."""

        context_getter = getattr(self.builder_tab, "current_terrain_brush_context", None)
        context = context_getter() if callable(context_getter) else {}
        if not isinstance(context, dict):
            context = {}
        enabled = force_enabled
        if enabled is None:
            section_tabs = getattr(self.builder_tab, "buildSectionTabs", None)
            terrain_page = getattr(self.builder_tab, "terrainBuildingPage", None)
            enabled = bool(
                self.workflow_tabs.currentWidget() is self.builder_tab
                and section_tabs is not None
                and section_tabs.currentWidget() is terrain_page
            )
        if not bool(context.get("enabled", False)):
            enabled = False
        setter = getattr(self.viewport_panel, "set_terrain_brush_interaction", None)
        if callable(setter):
            setter(
                enabled=enabled,
                room_resref=str(context.get("room_resref") or ""),
                brush=str(context.get("brush") or ""),
                row_count=int(context.get("row_count", 0) or 0),
                column_count=int(context.get("column_count", 0) or 0),
                radius=int(context.get("radius", 0) or 0),
                hardness=float(context.get("hardness", context.get("strength", 0.5)) or 0.5),
                strength=float(context.get("strength", 0.5) or 0.5),
                delta=float(context.get("delta", 0.1) or 0.1),
            )

    def _set_map_studio_terrain_brush_options(self, radius: int, hardness: float) -> None:
        """Apply viewport brush size/falloff edits to the visible Builder controls."""

        radius_spin = getattr(self.builder_tab, "terrainRadiusSpinBox", None)
        hardness_spin = getattr(self.builder_tab, "terrainHardnessSpinBox", None)
        if radius_spin is not None:
            blocked = radius_spin.blockSignals(True)
            radius_spin.setValue(max(radius_spin.minimum(), min(radius_spin.maximum(), int(radius))))
            radius_spin.blockSignals(blocked)
        if hardness_spin is not None:
            blocked = hardness_spin.blockSignals(True)
            hardness_spin.setValue(max(hardness_spin.minimum(), min(hardness_spin.maximum(), float(hardness))))
            hardness_spin.blockSignals(blocked)
        label = getattr(self.builder_tab, "terrainBrushStatusLabel", None)
        if label is not None:
            label.setText(f"Brush options: size {int(radius)}, hardness {float(hardness):.2f}.")
        self._sync_map_studio_terrain_brush_context(force_enabled=True)

    def _set_map_studio_terrain_brush_strength(self, strength: float) -> None:
        """Keep the viewport sculpt shelf and Terrain Building strength aligned."""

        spin = getattr(self.builder_tab, "terrainSmoothStrengthSpinBox", None)
        if spin is not None:
            blocked = spin.blockSignals(True)
            spin.setValue(max(spin.minimum(), min(spin.maximum(), float(strength))))
            spin.blockSignals(blocked)
        self._sync_map_studio_terrain_brush_context(force_enabled=True)

    def _set_map_studio_terrain_brush_delta(self, delta: float) -> None:
        """Keep the viewport sculpt shelf and Terrain Building height step aligned."""

        spin = getattr(self.builder_tab, "terrainDeltaSpinBox", None)
        if spin is not None:
            blocked = spin.blockSignals(True)
            spin.setValue(max(spin.minimum(), min(spin.maximum(), float(delta))))
            spin.blockSignals(blocked)
        self._sync_map_studio_terrain_brush_context(force_enabled=True)

    def _set_map_studio_terrain_sculpt_enabled(self, enabled: bool) -> None:
        """Mirror the viewport's enter/exit sculpt state into Terrain Building."""

        checkbox = getattr(self.builder_tab, "terrainSculptEnabledCheckBox", None)
        if checkbox is not None:
            blocked = checkbox.blockSignals(True)
            checkbox.setChecked(bool(enabled))
            checkbox.blockSignals(blocked)
        self._sync_map_studio_terrain_brush_context(force_enabled=bool(enabled))

    def apply_map_studio_viewport_terrain_brush_frame(self, brush: str, room_resref: str, points: object) -> None:
        """Apply one live viewport terrain sculpt frame without a full Map Studio rebuild."""

        context_getter = getattr(self.builder_tab, "current_terrain_brush_context", None)
        context = context_getter() if callable(context_getter) else {}
        if not isinstance(context, dict):
            context = {}
        try:
            performance_policy = self.controller.map_studio_viewport_performance_policy()
            result = self.controller.apply_map_studio_terrain_sculpt_frame(
                room_resref=room_resref,
                brush=brush,
                points=tuple(points or ()),
                delta=float(context.get("delta", 0.1) or 0.1),
                radius=int(context.get("radius", 0) or 0),
                height=float(context.get("height", 0.0) or 0.0),
                iterations=int(context.get("iterations", 1) or 1),
                strength=float(context.get("strength", 0.5) or 0.5),
                falloff_hardness=float(context.get("hardness", 0.5) or 0.5),
                budget_ms=float(getattr(performance_policy, "terrain_brush_budget_ms", 4.0) or 4.0),
                force=True,
            )
        except Exception as exc:
            status = f"Live terrain brush failed: {exc}"
            label = getattr(self.builder_tab, "terrainBrushStatusLabel", None)
            if label is not None:
                label.setText(status)
            self._log(status)
            return
        if result.applied:
            self.viewport_panel.apply_terrain_height_patch(
                room_resref,
                result.dirty_region_with_halo,
                result.dirty_height_patch,
                row_count=int(result.row_count),
                column_count=int(result.column_count),
            )
        frame = result.frame
        status = (
            f"Sculpting {str(brush).replace('_', ' ').title()}: "
            f"{frame.performance.affected_sample_count} terrain sample(s) updated. "
            "Release to create one undo step and refresh slope checks."
        )
        if not result.applied:
            status = result.message
        label = getattr(self.builder_tab, "terrainBrushStatusLabel", None)
        if label is not None:
            label.setText(status)

    def commit_map_studio_viewport_terrain_brush_stroke(self, brush: str, room_resref: str) -> None:
        """Commit a live terrain stroke without replacing its resident mesh."""

        started = perf_counter()
        try:
            commit = self.controller.commit_map_studio_terrain_sculpt_stroke(
                brush=brush,
                room_resref=room_resref,
            )
        except Exception as exc:
            self._log(f"Terrain brush commit failed: {exc}")
            return
        if commit is None:
            return
        message = (
            f"Committed terrain brush {brush} on {room_resref}; validating WOK/readiness in the background."
        )
        # The terrain node already contains the final dirty height/normal
        # patch.  Keep the existing overlay as the XY hit-test proxy so the
        # next stroke can begin immediately; background validation replaces it
        # with a fresh slope/perimeter result without interrupting sculpting.
        self._refresh_map_studio_geometry_change(
            message,
            rebuild_viewport_model=False,
            refresh_scene_tree=False,
            validation_delay_ms=250,
        )
        self._sync_map_studio_terrain_brush_context()
        self._last_map_studio_terrain_release_ms = (perf_counter() - started) * 1000.0

    def merge_authored_floor_plan_rooms(self, first_room_resref: str, second_room_resref: str, result_room_resref: str) -> None:
        try:
            result = self.controller.merge_authored_floor_plan_rooms(
                first_room_resref=first_room_resref,
                second_room_resref=second_room_resref,
                result_room_resref=result_room_resref,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Union Rectangular Rooms", str(exc))
            return
        readiness = result.readiness
        label = result_room_resref.strip() or first_room_resref
        message = f"Merged floor-plan rooms {first_room_resref} and {second_room_resref} into {label}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )

    def add_authored_room_primitive(self, primitive_kind: str, primitive_name: str) -> None:
        before_keys = {
            (
                str(getattr(row, "room_resref", "") or ""),
                str(getattr(row, "primitive_name", "") or ""),
            )
            for row in tuple(self.controller.authored_room_primitive_transforms() or ())
        }
        try:
            result = self.controller.add_authored_room_primitive(
                primitive_kind=primitive_kind,
                primitive_name=primitive_name,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Room Primitive", str(exc))
            return
        readiness = result.readiness
        label = primitive_name or primitive_kind
        message = f"Added room primitive {label}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        after_rows = tuple(self.controller.authored_room_primitive_transforms() or ())
        created = tuple(
            (
                str(getattr(row, "room_resref", "") or ""),
                str(getattr(row, "primitive_name", "") or ""),
            )
            for row in after_rows
            if (
                str(getattr(row, "room_resref", "") or ""),
                str(getattr(row, "primitive_name", "") or ""),
            )
            not in before_keys
        )
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=created[-1:] if created else (),
            refresh_outlines=True,
        )

    def apply_authored_room_primitive_transform(
        self,
        room_resref: str,
        primitive_name: str,
        tx: float,
        ty: float,
        tz: float,
        rot_z: float,
        sx: float,
        sy: float,
        sz: float,
        px: float,
        py: float,
        pz: float,
    ) -> None:
        try:
            result = self.controller.set_authored_room_primitive_transform(
                room_resref=room_resref,
                primitive_name=primitive_name,
                translation=(tx, ty, tz),
                rotation_degrees_z=rot_z,
                scale=(sx, sy, sz),
                pivot=(px, py, pz),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Primitive Transform", str(exc))
            return
        readiness = result.readiness
        message = f"Transformed room primitive {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=((room_resref, primitive_name),),
            refresh_outlines=True,
        )

    def apply_authored_room_primitive_dimensions(self, room_resref: str, primitive_name: str, dimensions: object) -> None:
        try:
            result = self.controller.set_authored_room_primitive_dimensions(
                room_resref=room_resref,
                primitive_name=primitive_name,
                dimensions=dimensions,
            )
        except Exception as exc:
            self.viewport_panel.clear_primitive_recipe_preview()
            self._refresh_map_studio_selected_primitive_transform_overlay()
            QtWidgets.QMessageBox.warning(self, "Apply Primitive Dimensions", str(exc))
            return
        readiness = result.readiness
        message = f"Edited room primitive dimensions for {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        # Only promote a resident preview when it was evaluated from the exact
        # values just committed.  A deferred high-topology preview may leave an
        # older lightweight mesh resident; promoting that stale mesh would make
        # the viewport disagree with the retained recipe.
        promoted = bool(
            getattr(self.controller, "last_map_studio_primitive_commit_matches_preview", False)
        ) and self.viewport_panel.promote_primitive_recipe_preview(room_resref, primitive_name)
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=((room_resref, primitive_name),),
            refresh_outlines=True,
            rebuild_viewport_model=not promoted,
        )

    def preview_authored_room_primitive_dimensions(
        self,
        room_resref: str,
        primitive_name: str,
        dimensions: object,
    ) -> None:
        """Evaluate and display one construction recipe without touching KMAP."""

        try:
            payloads = self.controller.preview_authored_room_primitive_dimensions(
                room_resref=room_resref,
                primitive_name=primitive_name,
                dimensions=dimensions,
            )
        except Exception as exc:
            # A partially typed numeric value should not interrupt modeling
            # with a modal dialog.  Keep the last valid preview resident and
            # surface the constraint in the status bar until the user edits or
            # explicitly applies the property set.
            if bool(getattr(self.controller, "last_map_studio_primitive_preview_deferred", False)):
                self.viewport_panel.clear_primitive_recipe_preview()
                self._refresh_map_studio_selected_primitive_transform_overlay()
            self.statusBar().showMessage(f"Primitive preview: {exc}", 5000)
            return
        if self.viewport_panel.apply_primitive_recipe_preview(
            room_resref,
            primitive_name,
            payloads,
        ):
            overlay = getattr(self.controller, "last_map_studio_primitive_preview_overlay", None)
            if overlay is not None:
                self.viewport_panel.set_universal_transform_overlay(overlay)
            elapsed = float(
                getattr(self.controller, "last_map_studio_primitive_preview_elapsed_ms", 0.0)
                or 0.0
            )
            self.statusBar().showMessage(
                f"Previewing {primitive_name} construction inputs ({elapsed:.2f} ms; not committed).",
                2500,
            )

    def cancel_authored_room_primitive_dimensions_preview(
        self,
        _room_resref: str = "",
        _primitive_name: str = "",
    ) -> None:
        """Restore the resident mesh before the current recipe scrub."""

        self.viewport_panel.clear_primitive_recipe_preview()
        self._refresh_map_studio_selected_primitive_transform_overlay()
        self.statusBar().showMessage("Primitive construction changes cancelled.", 2500)

    def apply_authored_room_primitive_style(self, room_resref: str, primitive_name: str, texture: str, surface_id: str) -> None:
        try:
            result = self.controller.set_authored_room_primitive_style(
                room_resref=room_resref,
                primitive_name=primitive_name,
                texture=texture,
                surface_id=surface_id,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Primitive Material + Surface", str(exc))
            return
        readiness = result.readiness
        message = f"Styled room primitive {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=((room_resref, primitive_name),),
        )

    def remove_authored_room_primitive(self, room_resref: str, primitive_name: str) -> None:
        try:
            result = self.controller.remove_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Remove Room Primitive", str(exc))
            return
        readiness = result.readiness
        message = f"Removed room primitive {primitive_name}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_outlines=True)

    def separate_authored_room_primitive(self, room_resref: str, primitive_name: str, result_room_resref: str) -> None:
        """Legacy room-boundary workflow, now labeled honestly as Extract."""

        try:
            result = self.controller.separate_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
                result_room_resref=result_room_resref,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Extract Primitive to Export Room", str(exc))
            return
        readiness = result.readiness
        message = f"Extracted room primitive {primitive_name} to its own KOTOR export room; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_terrain=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )

    def apply_authored_room_style(self, texture: str, floor_surface: str) -> None:
        try:
            result = self.controller.apply_authored_room_style(texture=texture, floor_surface=floor_surface)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Room Material + Surface", str(exc))
            return
        readiness = result.readiness
        message = "Applied room material and walkmesh surface; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message)

    def apply_authored_walkmesh_surface(self, room_resref: str, floor_surface: str) -> None:
        try:
            result = self.controller.set_authored_room_walkmesh_surface(room_resref=room_resref, floor_surface=floor_surface)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Room WOK Surface", str(exc))
            return
        readiness = result.readiness
        message = f"Applied WOK surface {floor_surface} to room {room_resref}; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(message, refresh_terrain=True)

    def add_authored_room_light(
        self,
        room_resref: str,
        name: str,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        color_r: float,
        color_g: float,
        color_b: float,
        radius: float,
        intensity: float,
        light_type: str,
    ) -> None:
        try:
            result = self.controller.add_authored_room_light(
                room_resref=room_resref,
                name=name,
                position=(pos_x, pos_y, pos_z),
                color=(color_r, color_g, color_b),
                radius=radius,
                intensity=intensity,
                light_type=light_type,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Room Light", str(exc))
            return
        readiness = result.readiness
        message = "Added authored room light; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_all(message)

    def add_authored_gameplay_placement(
        self,
        kind: str,
        template_resref: str,
        tag: str,
        x: float,
        y: float,
        z: float,
        bearing: float,
    ) -> None:
        try:
            result = self.controller.add_authored_gameplay_placement(
                kind=kind,
                template_resref=template_resref,
                tag=tag,
                position=(x, y, z),
                bearing=bearing,
                snap_to_walkmesh=(
                    self.workflow_tabs.currentWidget() is self.placement_tab
                    and bool(self.placement_tab.snap_wok_box.isChecked())
                ),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Gameplay Placement", str(exc))
            return
        if kind in {"placeable", "door"}:
            try:
                # Coordinate placement must resolve the same authored resource
                # graph as a viewport drop so its real model/effects appear
                # immediately instead of remaining a marker until export.
                self._sync_placeable_library_resources_for_export()
            except Exception as exc:
                self._log(f"Placeable preview resource warning: {exc}")
        readiness = result.readiness
        message = f"Added authored {kind} placement; previous exports/proofs are now stale."
        if readiness is not None:
            message = f"{message} Readiness: {readiness.capability_stage}."
        self._refresh_map_studio_geometry_change(
            message,
            refresh_outlines=True,
            refresh_room_choices=True,
            refresh_connections=True,
        )
        matching = [
            row
            for row in self.controller.authored_gameplay_placements()
            if str(getattr(row, "kind", "") or "") == str(kind)
        ]
        placement_id = str(getattr(matching[-1], "placement_id", "") or "") if matching else ""
        if placement_id:
            self.select_item(placement_id)

    def _set_map_studio_placement_mode(self, context: object) -> None:
        values = dict(context) if isinstance(context, dict) else {"enabled": False}
        source_game = str(values.get("game") or "").strip().upper()
        target_game = str(getattr(self.project, "game", "") or "").strip().upper()
        if bool(values.get("enabled", False)) and source_game and target_game and source_game != target_game:
            values["enabled"] = False
            QtWidgets.QMessageBox.warning(
                self,
                "Wrong Game Resource",
                f"{source_game} content cannot be placed in a {target_game} map. Choose a {target_game} template.",
            )
        self.viewport_panel.set_placement_tool_context(values)
        if bool(values.get("enabled", False)):
            self.workflow_panel.set_active_authoring_context(
                "Placement armed: click a visible level surface; Esc exits. WOK snapping keeps gameplay anchors on valid floor."
            )

    def _render_map_studio_placement_thumbnail(self, entry: object) -> None:
        """Resolve and render one visible Content Browser asset tile lazily."""

        def value(key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        manager = getattr(self, "resource_manager", None)
        game = str(value("game") or getattr(self.project, "game", "K1") or "K1").strip().upper()
        kind = str(value("kind") or "placeable").strip().lower()
        template_resref = str(value("template_resref") or "").strip().lower()
        manager_revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else 0
        authored_revision = int(getattr(self.controller, "_authored_placeable_preview_revision", 0) or 0)
        cache_key = (id(manager), manager_revision, authored_revision, game, kind, template_resref)
        if cache_key in self._map_studio_placement_thumbnail_cache:
            cached = self._map_studio_placement_thumbnail_cache[cache_key]
            detail = (
                f"{template_resref}: real KOTOR model thumbnail ready. Drag its tile onto a highlighted floor."
                if cached is not None
                else f"{template_resref}: no renderable 3D model was resolved; the asset can still be dragged by its type tile."
            )
            self.placement_tab.set_asset_thumbnail(entry, cached, detail)
            return
        if manager is None:
            self._map_studio_placement_thumbnail_cache[cache_key] = None
            self.placement_tab.set_asset_thumbnail(
                entry,
                None,
                "Connect the selected KOTOR installation to load real asset thumbnails.",
            )
            return

        try:
            model, model_resref = self.controller.map_studio_palette_preview_model(entry, manager)
            if model is None:
                raise ValueError("No renderable model is associated with this resource type.")
            textures: dict[str, Any] = {}
            texture_list = getattr(model, "texture_list", None)
            texture_names = tuple(texture_list() or ()) if callable(texture_list) else ()
            for texture_name in texture_names[:16]:
                clean_name = str(texture_name or "").strip().strip("\x00").lower()
                if not clean_name or clean_name in {"null", "none", "default"}:
                    continue
                try:
                    image = manager.load_texture_image(clean_name, game, max_size=256)
                except TypeError:
                    image = manager.load_texture_image(clean_name, game)
                if image is not None:
                    textures[clean_name] = image

            from src.adapters.rendering.moderngl_scene_helpers import render_model_autoframe

            viewport = getattr(self.viewport_panel, "viewport", None)
            thumbnail_renderer = getattr(viewport, "_gpu_renderer", None)
            active_renderer = getattr(thumbnail_renderer, "active_renderer", None)
            if active_renderer is not None:
                thumbnail_renderer = active_renderer
            backend_id = str(getattr(thumbnail_renderer, "backend_id", "") or "").strip().lower()
            if (
                thumbnail_renderer is None
                or not callable(getattr(thumbnail_renderer, "render", None))
                or "null" in backend_id
            ):
                wait_count = self._map_studio_placement_thumbnail_waits.get(cache_key, 0) + 1
                self._map_studio_placement_thumbnail_waits[cache_key] = wait_count
                if wait_count <= 12:
                    QtCore.QTimer.singleShot(
                        250,
                        lambda pending_entry=entry: self._render_map_studio_placement_thumbnail(pending_entry),
                    )
                    return
                raise ValueError("The viewport renderer is unavailable for thumbnail reuse.")

            rendered = render_model_autoframe(
                model,
                W=192,
                H=192,
                textures=textures,
                views=["diag"],
                renderer=thumbnail_renderer,
            ).get("diag")
            if rendered is None:
                raise ValueError("The model renderer did not produce an image.")
            rgba = rendered.convert("RGBA")
            width, height = rgba.size
            qimage = QtGui.QImage(
                rgba.tobytes("raw", "RGBA"),
                int(width),
                int(height),
                int(width) * 4,
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(qimage)
            if pixmap.isNull():
                raise ValueError("The rendered model thumbnail was empty.")
        except Exception as exc:
            self._map_studio_placement_thumbnail_cache[cache_key] = None
            self._log(f"Placement thumbnail warning for {template_resref or kind}: {exc}")
            self.placement_tab.set_asset_thumbnail(
                entry,
                None,
                f"{template_resref or kind}: no renderable 3D thumbnail; drag the type tile to place it.",
            )
            return

        self._map_studio_placement_thumbnail_cache[cache_key] = pixmap
        self._map_studio_placement_thumbnail_waits.pop(cache_key, None)
        if len(self._map_studio_placement_thumbnail_cache) > 192:
            oldest_key = next(iter(self._map_studio_placement_thumbnail_cache))
            self._map_studio_placement_thumbnail_cache.pop(oldest_key, None)
        self.placement_tab.set_asset_thumbnail(
            entry,
            pixmap,
            f"{template_resref} → {model_resref}: drag this tile onto a highlighted floor to place it.",
        )

    def _refresh_vanilla_terrain_kit_catalog(self) -> None:
        """Rebuild vanilla outdoor-mesh metadata without freezing the viewport."""

        if bool(getattr(self, "_terrain_catalog_scan_active", False)):
            return
        manager = getattr(self, "resource_manager", None)
        if manager is None or not bool(getattr(manager, "is_ready", lambda: False)()):
            self.terrain_kit_browser.set_refreshing(
                False,
                "Connect the target KOTOR installation before refreshing the Vanilla Terrain library.",
            )
            return
        self._terrain_catalog_scan_active = True
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        self.terrain_kit_browser.set_refreshing(
            True,
            f"Studying {game} outdoor room geometry. You can keep working while the catalog builds…",
        )

        def progress(completed: int, total: int, detail: str) -> None:
            percent = int(round((100.0 * completed / total))) if total else 0
            self.terrainCatalogProgress.emit(
                f"Vanilla Terrain {percent}% · {detail}"
            )

        def work() -> None:
            try:
                rows, path = self.controller.refresh_map_studio_vanilla_terrain_kit_assets(
                    resource_manager=manager,
                    progress=progress,
                )
            except Exception as exc:
                self.terrainCatalogFailed.emit(str(exc))
                return
            self.terrainCatalogReady.emit(rows, path)

        _MAP_STUDIO_TERRAIN_CATALOG_EXECUTOR.submit(work)

    def _vanilla_terrain_catalog_ready(self, rows: object, path: str) -> None:
        self._terrain_catalog_scan_active = False
        self._map_studio_terrain_kit_thumbnail_cache.clear()
        self._map_studio_terrain_kit_thumbnail_waits.clear()
        entries = tuple(rows or ())
        self.terrain_kit_browser.set_assets(entries)
        vanilla_count = sum(
            1
            for entry in entries
            if str((entry.get("source_kind") if isinstance(entry, dict) else getattr(entry, "source_kind", "")) or "")
            == "vanilla_game"
        )
        self.terrain_kit_browser.set_refreshing(
            False,
            f"Vanilla Terrain ready: {vanilla_count:,} locally indexed pieces. Catalog: {path}",
        )

    def _vanilla_terrain_catalog_failed(self, message: str) -> None:
        self._terrain_catalog_scan_active = False
        self.terrain_kit_browser.set_refreshing(
            False,
            f"Vanilla Terrain refresh failed: {message}",
        )

    def _refresh_vanilla_environment_kit_catalog(self) -> None:
        """Rebuild typed interior/exterior module kits without blocking authoring."""

        if bool(getattr(self, "_environment_catalog_scan_active", False)):
            return
        manager = getattr(self, "resource_manager", None)
        if manager is None or not bool(getattr(manager, "is_ready", lambda: False)()):
            self.environment_kit_browser.set_refreshing(
                False,
                "Connect the target KOTOR installation before refreshing Vanilla Environment Kits.",
            )
            return
        self._environment_catalog_scan_active = True
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        self.environment_kit_browser.set_refreshing(
            True,
            f"Classifying {game} rooms, exteriors, and LYT doorway magnets. You can keep working…",
        )

        def progress(completed: int, total: int, detail: str) -> None:
            percent = int(round((100.0 * completed / total))) if total else 0
            self.environmentCatalogProgress.emit(f"Environment Kits {percent}% · {detail}")

        def work() -> None:
            try:
                rows, path = self.controller.refresh_map_studio_vanilla_environment_kits(
                    resource_manager=manager,
                    progress=progress,
                )
            except Exception as exc:
                self.environmentCatalogFailed.emit(str(exc))
                return
            self.environmentCatalogReady.emit(rows, path)

        _MAP_STUDIO_TERRAIN_CATALOG_EXECUTOR.submit(work)

    def _sync_map_studio_kit_browsers_for_project_game(self) -> bool:
        """Reload both drag shelves when an opened KMAP changes the target game."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        if game == str(getattr(self, "_map_studio_kit_browser_game", "") or "").upper():
            return False
        self.environment_kit_browser.set_assets(
            self.controller.available_map_studio_environment_kit_pieces()
        )
        self.terrain_kit_browser.set_assets(
            self.controller.available_map_studio_terrain_kit_assets()
        )
        self._map_studio_kit_browser_game = game
        return True

    def _vanilla_environment_catalog_ready(self, rows: object, path: str) -> None:
        self._environment_catalog_scan_active = False
        self._map_studio_environment_kit_thumbnail_cache.clear()
        self._map_studio_environment_kit_thumbnail_waits.clear()
        entries = tuple(rows or ())
        self.environment_kit_browser.set_assets(entries)
        self.builder_tab.set_building_styles(self.controller.available_map_studio_building_styles())
        self.environment_kit_browser.set_refreshing(
            False,
            f"Environment Kits ready: {len(entries):,} typed room/exterior pieces. Catalog: {path}",
        )

    def _vanilla_environment_catalog_failed(self, message: str) -> None:
        self._environment_catalog_scan_active = False
        self.environment_kit_browser.set_refreshing(
            False,
            f"Environment Kit refresh failed: {message}",
        )

    def _render_map_studio_terrain_kit_thumbnail(self, entry: object) -> None:
        """Render one supplied OBJ terrain form through the active viewport backend."""

        def value(key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        asset_id = str(value("asset_id") or "").strip()
        label = str(value("label") or asset_id or "Terrain piece")
        manager = getattr(self, "resource_manager", None)
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        manager_revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else 0
        cache_key = (asset_id, game, id(manager), manager_revision)
        if cache_key in self._map_studio_terrain_kit_thumbnail_cache:
            self.terrain_kit_browser.set_asset_thumbnail(
                entry,
                self._map_studio_terrain_kit_thumbnail_cache[cache_key],
                f"{label}: drag this terrain form onto a visible level surface.",
            )
            return
        try:
            model = self.controller.map_studio_terrain_kit_preview_model(
                asset_id,
                resource_manager=manager,
            )
            if model is None:
                raise ValueError("No preview mesh was produced.")
            textures: dict[str, Any] = {}
            texture_list = getattr(model, "texture_list", None)
            for texture_name in tuple(texture_list() or ())[:16] if callable(texture_list) else ():
                clean_name = str(texture_name or "").strip().strip("\x00").lower()
                if manager is None or not clean_name or clean_name in {"null", "none", "default"}:
                    continue
                try:
                    image = manager.load_texture_image(clean_name, game, max_size=256)
                except TypeError:
                    image = manager.load_texture_image(clean_name, game)
                if image is not None:
                    textures[clean_name] = image

            viewport = getattr(self.viewport_panel, "viewport", None)
            thumbnail_renderer = getattr(viewport, "_gpu_renderer", None)
            active_renderer = getattr(thumbnail_renderer, "active_renderer", None)
            if active_renderer is not None:
                thumbnail_renderer = active_renderer
            backend_id = str(getattr(thumbnail_renderer, "backend_id", "") or "").strip().lower()
            if (
                thumbnail_renderer is None
                or not callable(getattr(thumbnail_renderer, "render", None))
                or "null" in backend_id
            ):
                wait_count = self._map_studio_terrain_kit_thumbnail_waits.get(cache_key, 0) + 1
                self._map_studio_terrain_kit_thumbnail_waits[cache_key] = wait_count
                if wait_count <= 12:
                    QtCore.QTimer.singleShot(
                        250,
                        lambda pending_entry=entry: self._render_map_studio_terrain_kit_thumbnail(pending_entry),
                    )
                    return
                raise ValueError("The viewport renderer is unavailable for thumbnail reuse.")

            from src.adapters.rendering.moderngl_scene_helpers import render_model_autoframe

            rendered = render_model_autoframe(
                model,
                W=192,
                H=192,
                textures=textures,
                views=["diag"],
                renderer=thumbnail_renderer,
            ).get("diag")
            if rendered is None:
                raise ValueError("The renderer did not produce a terrain thumbnail.")
            rgba = rendered.convert("RGBA")
            width, height = rgba.size
            qimage = QtGui.QImage(
                rgba.tobytes("raw", "RGBA"),
                int(width),
                int(height),
                int(width) * 4,
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(qimage)
            if pixmap.isNull():
                raise ValueError("The terrain thumbnail was empty.")
        except Exception as exc:
            self._map_studio_terrain_kit_thumbnail_cache[cache_key] = None
            self._log(f"Terrain Kit thumbnail warning for {asset_id}: {exc}")
            self.terrain_kit_browser.set_asset_thumbnail(
                entry,
                None,
                f"{label}: preview unavailable; the asset can still be dragged into the viewport.",
            )
            return

        self._map_studio_terrain_kit_thumbnail_cache[cache_key] = pixmap
        self._map_studio_terrain_kit_thumbnail_waits.pop(cache_key, None)
        self.terrain_kit_browser.set_asset_thumbnail(
            entry,
            pixmap,
            f"{label}: drag this terrain form onto a visible level surface.",
        )

    def _render_map_studio_environment_kit_thumbnail(self, entry: object) -> None:
        """Queue one authentic room-tile preview without blocking Map Studio."""

        def value(key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        piece_id = str(value("piece_id") or value("asset_id") or "").strip()
        label = str(value("label") or piece_id or "Environment piece")
        manager = getattr(self, "resource_manager", None)
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        manager_revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else 0
        cache_key = (piece_id, game, id(manager), manager_revision)
        if cache_key in self._map_studio_environment_kit_thumbnail_cache:
            self.environment_kit_browser.set_asset_thumbnail(
                entry,
                self._map_studio_environment_kit_thumbnail_cache[cache_key],
                f"{label}: drag into the viewport; green means a compatible doorway snap.",
            )
            return
        if cache_key in self._map_studio_environment_kit_thumbnail_inflight:
            return
        self._map_studio_environment_kit_thumbnail_inflight.add(cache_key)
        future = _MAP_STUDIO_ENVIRONMENT_THUMBNAIL_EXECUTOR.submit(
            self._build_map_studio_environment_kit_thumbnail,
            piece_id,
            game,
            manager,
        )

        def publish(done: object) -> None:
            try:
                payload = done.result()
            except Exception as exc:  # worker failure is rendered as a usable placeholder
                payload = {"error": str(exc)}
            self.environmentThumbnailReady.emit(entry, cache_key, payload)

        future.add_done_callback(publish)

    def _build_map_studio_environment_kit_thumbnail(
        self,
        piece_id: str,
        game: str,
        manager: object,
    ) -> dict[str, Any]:
        """Load and render one thumbnail entirely away from Qt's UI thread."""

        model = self.controller.map_studio_environment_kit_preview_model(
            piece_id,
            resource_manager=manager,
        )
        if model is None:
            raise ValueError("The source room model is unavailable in the connected game installation.")
        textures: dict[str, Any] = {}
        texture_list = getattr(model, "texture_list", None)
        for texture_name in tuple(texture_list() or ())[:24] if callable(texture_list) else ():
            clean_name = str(texture_name or "").strip().strip("\x00").lower()
            if manager is None or not clean_name or clean_name in {"null", "none", "default"}:
                continue
            try:
                image = manager.load_texture_image(clean_name, game, max_size=256)
            except TypeError:
                image = manager.load_texture_image(clean_name, game)
            if image is not None:
                textures[clean_name] = image

        from src.adapters.rendering.moderngl_scene_helpers import render_model_autoframe

        rendered = render_model_autoframe(
            model,
            W=192,
            H=192,
            textures=textures,
            views=["diag"],
        ).get("diag")
        if rendered is None:
            raise ValueError("The renderer did not produce an environment-piece thumbnail.")
        rgba = rendered.convert("RGBA")
        width, height = rgba.size
        return {
            "rgba": rgba.tobytes("raw", "RGBA"),
            "width": int(width),
            "height": int(height),
        }

    @QtCore.Slot(object, object, object)
    def _apply_map_studio_environment_kit_thumbnail(
        self,
        entry: object,
        cache_key: object,
        payload: object,
    ) -> None:
        """Publish a completed worker thumbnail as a GUI-owned pixmap."""

        key = tuple(cache_key or ())
        self._map_studio_environment_kit_thumbnail_inflight.discard(key)
        values = dict(payload) if isinstance(payload, dict) else {}
        piece_id = str(
            (entry.get("piece_id") if isinstance(entry, dict) else getattr(entry, "piece_id", ""))
            or ""
        )
        label = str(
            (entry.get("label") if isinstance(entry, dict) else getattr(entry, "label", ""))
            or piece_id
            or "Environment piece"
        )
        error = str(values.get("error") or "").strip()
        pixmap: QtGui.QPixmap | None = None
        if not error:
            try:
                width = int(values.get("width", 0) or 0)
                height = int(values.get("height", 0) or 0)
                rgba = bytes(values.get("rgba") or b"")
                if width <= 0 or height <= 0 or len(rgba) != width * height * 4:
                    raise ValueError("The environment-piece thumbnail buffer was invalid.")
                qimage = QtGui.QImage(
                    rgba,
                    width,
                    height,
                    width * 4,
                    QtGui.QImage.Format.Format_RGBA8888,
                ).copy()
                pixmap = QtGui.QPixmap.fromImage(qimage)
                if pixmap.isNull():
                    raise ValueError("The environment-piece thumbnail was empty.")
            except Exception as exc:
                error = str(exc)
                pixmap = None

        if error:
            self._map_studio_environment_kit_thumbnail_cache[key] = None
            self._log(f"Environment Kit thumbnail warning for {piece_id}: {error}")
            self.environment_kit_browser.set_asset_thumbnail(
                entry,
                None,
                f"{label}: preview unavailable; the piece can still be dragged into the viewport.",
            )
            return

        self._map_studio_environment_kit_thumbnail_cache[key] = pixmap
        self._map_studio_environment_kit_thumbnail_waits.pop(key, None)
        if len(self._map_studio_environment_kit_thumbnail_cache) > 192:
            oldest_key = next(iter(self._map_studio_environment_kit_thumbnail_cache))
            self._map_studio_environment_kit_thumbnail_cache.pop(oldest_key, None)
        self.environment_kit_browser.set_asset_thumbnail(
            entry,
            pixmap,
            f"{label}: drag into the viewport; green means a compatible doorway snap.",
        )

    def _place_authored_terrain_kit_from_viewport(self, payload: object) -> None:
        """Commit one surface-snapped Terrain Kit room from a content-browser drop."""

        values = dict(payload) if isinstance(payload, dict) else {}
        position = tuple(values.get("position", ()) or ())
        if len(position) < 3:
            return
        try:
            room_resref = self.controller.add_authored_terrain_kit_asset(
                asset_id=str(values.get("asset_id") or ""),
                position=position[:3],
                rotation_degrees_z=float(values.get("rotation_degrees_z", 0.0) or 0.0),
                scale=float(values.get("scale", 1.0) or 1.0),
                target_room_resref=str(values.get("room_resref") or ""),
                resource_manager=getattr(self, "resource_manager", None),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Place Terrain Piece", str(exc))
            return
        label = str(values.get("label") or values.get("asset_id") or "terrain piece")
        self._refresh_all(
            f"Placed {label} as static KOTOR terrain geometry ({room_resref}); previous exports/proofs are now stale."
        )

    def _preview_authored_terrain_kit_from_viewport(self, payload: object) -> None:
        """Resolve a disposable magnet preview during a browser drag."""

        values = dict(payload) if isinstance(payload, dict) else {}
        position = tuple(values.get("position", ()) or ())
        if len(position) < 3:
            self.viewport_panel.set_terrain_kit_snap_preview(None)
            return
        try:
            preview = self.controller.preview_authored_terrain_kit_placement(
                asset_id=str(values.get("asset_id") or ""),
                position=position[:3],
                rotation_degrees_z=float(values.get("rotation_degrees_z", 0.0) or 0.0),
                scale=float(values.get("scale", 1.0) or 1.0),
            )
        except Exception:
            preview = None
        self.viewport_panel.set_terrain_kit_snap_preview(preview)

    def _place_authored_environment_kit_from_viewport(self, payload: object) -> None:
        """Commit one surface- or doorway-snapped vanilla environment tile."""

        values = dict(payload) if isinstance(payload, dict) else {}
        position = tuple(values.get("position", ()) or ())
        if len(position) < 3:
            return
        try:
            room_resref = self.controller.add_authored_environment_kit_piece(
                piece_id=str(values.get("piece_id") or values.get("asset_id") or ""),
                position=position[:3],
                rotation_degrees_z=float(values.get("rotation_degrees_z", 0.0) or 0.0),
                scale=float(values.get("scale", 1.0) or 1.0),
                target_room_resref=str(values.get("room_resref") or ""),
                resource_manager=getattr(self, "resource_manager", None),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Place Environment Piece", str(exc))
            return
        label = str(values.get("label") or values.get("piece_id") or "environment piece")
        self._refresh_all(
            f"Placed {label} as exportable KOTOR room geometry ({room_resref}); continue from a doorway magnet."
        )

    def _place_authored_gameplay_from_viewport(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        position = tuple(values.get("position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        if len(position) < 3:
            return
        kind = str(values.get("kind", "placeable") or "placeable")
        source_game = str(values.get("game") or "").strip().upper()
        target_game = str(getattr(self.project, "game", "") or "").strip().upper()
        if source_game and target_game and source_game != target_game:
            QtWidgets.QMessageBox.warning(
                self,
                "Wrong Game Resource",
                f"{source_game} content cannot be placed in a {target_game} map.",
            )
            return
        provenance = {
            key: str(values.get(key) or "").strip()
            for key in ("game", "library_source", "asset_id", "asset_path")
            if str(values.get(key) or "").strip()
        }
        try:
            self.controller.add_authored_gameplay_placement(
                kind=kind,
                template_resref=str(values.get("template_resref", "") or ""),
                tag=str(values.get("tag", "") or ""),
                position=position[:3],
                bearing=float(values.get("bearing", 0.0) or 0.0),
                snap_to_walkmesh=bool(values.get("snap_to_walkmesh", True)),
                provenance=provenance,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Place Gameplay Object", str(exc))
            return
        if kind in {"placeable", "door"}:
            try:
                # Resolve the exact UTP/UTD provenance now so module-local
                # terminals, containers, and doors preview/package from their
                # source graph instead of an unrelated duplicate resref.
                self._sync_placeable_library_resources_for_export()
            except Exception as exc:
                self._log(f"Placeable preview resource warning: {exc}")
        rows = [
            row for row in self.controller.authored_gameplay_placements()
            if str(getattr(row, "kind", "") or "") == kind
        ]
        placement_id = str(getattr(rows[-1], "placement_id", "") or "") if rows else ""
        self._refresh_all(f"Placed authored {kind} from the viewport; previous exports/proofs are now stale.")
        if placement_id:
            self.select_item(placement_id)
        if bool(values.get("keep_placing", False)):
            self.viewport_panel.set_placement_tool_context(self.placement_tab.placement_context())

    def _set_map_studio_building_tool(self, tool: str) -> None:
        """Hand direct-wall input to the viewport while Build / Room is active."""

        settings_getter = getattr(self.builder_tab, "_building_settings", None)
        settings = settings_getter() if callable(settings_getter) else {}
        self.viewport_panel.set_pascal_building_tool(tool, settings)
        if str(tool or "").strip().lower() != "select":
            self._sync_map_studio_terrain_brush_context(force_enabled=False)
            self.viewport_panel.set_placement_tool_context({"enabled": False})

    def _add_map_studio_building_level(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        level_index = int(values.get("level_index", 0) or 0)
        try:
            self.controller.add_map_studio_building_level(
                level_index=level_index,
                name=str(values.get("name") or f"Level {level_index + 1}"),
                floor_z=float(values.get("floor_z", 0.0) or 0.0),
                floor_to_floor_height=float(values.get("floor_to_floor_height", 3.0) or 3.0),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Building Level", str(exc))
            return
        self._refresh_all(
            f"Added Level {level_index + 1}; choose Draw Walls to author rooms at its true elevation."
        )
        self.builder_tab.select_building_level(level_index)

    def _apply_map_studio_level_presentation(self, payload: object = None) -> None:
        presentation = (
            dict(payload)
            if isinstance(payload, dict)
            else dict(self.builder_tab.building_level_presentation())
        )
        room_levels = {
            room_resref: int(level.level_index)
            for level in self.controller.map_studio_building_levels()
            for room_resref in tuple(level.room_resrefs or ())
        }
        self.viewport_panel.set_pascal_building_level_presentation(presentation, room_levels)

    def _build_map_studio_room_from_viewport(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        try:
            room_resref = self.controller.add_map_studio_building_room(
                points=tuple(values.get("points") or ()),
                floor_z=float(values.get("floor_z", 0.0) or 0.0),
                wall_height=float(values.get("wall_height", 3.0) or 3.0),
                level_index=int(values.get("level_index", 0) or 0),
                level_name=str(values.get("level_name") or ""),
                style_id=str(values.get("style_id") or "plcaa_graybox"),
                architecture_archetype=str(values.get("architecture_archetype") or ""),
                include_ceiling=bool(values.get("include_ceiling", True)),
                building_kind=str(values.get("building_kind") or "interior"),
                roof_type=str(values.get("roof_type") or "none"),
                roof_pitch_degrees=float(values.get("roof_pitch_degrees", 30.0) or 30.0),
                roof_overhang=float(values.get("roof_overhang", 0.25) or 0.0),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Build Room", str(exc))
            return
        self._refresh_all(
            f"Built {room_resref} from the closed wall path; MDL/MDX/WOK and module package outputs are now stale."
        )
        self.viewport_panel.set_pascal_building_tool("walls", values)

    def _build_map_studio_opening_from_viewport(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        kind = str(values.get("opening_kind") or "door").strip().lower()
        try:
            self.controller.set_map_studio_building_opening(
                room_resref=str(values.get("room_resref") or ""),
                edge_index=int(values.get("edge_index", 0) or 0),
                opening_kind=kind,
                center_fraction=float(values.get("center_fraction", 0.5) or 0.5),
                width=float(values.get("opening_width", 1.25) or 1.25),
                height=float(values.get("opening_height", 2.2) or 2.2),
                bottom=float(values.get("bottom", 0.0) or 0.0),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, f"Add {kind.title()}", str(exc))
            return
        self._refresh_all(f"Added a {kind} opening; generated room and module outputs are now stale.")
        self.viewport_panel.set_pascal_building_tool(kind, values)

    def _preview_map_studio_opening_from_viewport(self, payload: object) -> None:
        """Resolve one live hosted-opening ghost through the headless policy."""

        values = dict(payload) if isinstance(payload, dict) else {}
        try:
            resolved = self.controller.preview_map_studio_building_opening(
                room_resref=str(values.get("room_resref") or ""),
                edge_index=int(values.get("edge_index", 0) or 0),
                opening_kind=str(values.get("opening_kind") or "door"),
                center_fraction=float(values.get("center_fraction", 0.5) or 0.5),
                width=float(values.get("width", values.get("opening_width", 1.25)) or 1.25),
                height=float(values.get("height", values.get("opening_height", 2.2)) or 2.2),
                bottom=float(values.get("bottom", 0.0) or 0.0),
            )
        except Exception as exc:
            resolved = {"valid": False, "reason": str(exc)}
        self.viewport_panel.set_pascal_building_opening_preview({**values, **dict(resolved or {})})

    def _apply_placement_tab_transform(self, placement_id: str, position: object, bearing: float) -> None:
        try:
            self.controller.set_authored_gameplay_placement_transform(
                placement_id,
                position=tuple(position or (0.0, 0.0, 0.0)),
                bearing=float(bearing),
                snap_to_walkmesh=bool(self.placement_tab.snap_wok_box.isChecked()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Transform Gameplay Object", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            "Updated authored gameplay transform; previous exports/proofs are now stale.",
            placement_ids=(placement_id,),
        )
        self.select_item(placement_id)

    def _apply_placement_tab_creature_behavior(
        self,
        placement_id: str,
        faction_role: str,
        conversation_resref: str,
        movement_mode: str,
    ) -> None:
        try:
            self.controller.set_authored_creature_behavior(
                placement_id,
                faction_role=faction_role,
                conversation_resref=conversation_resref,
                movement_mode=movement_mode,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Creature Behavior", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            "Applied selected-creature behavior intent; target-game UTC/scripts will compile during export and plcaa manual proof is stale.",
            placement_ids=(placement_id,),
        )
        self.select_item(placement_id)

    def _request_creature_dialogue_editor(self, placement_id: str, conversation_resref: str) -> None:
        """Deep-link one creature conversation into the standalone narrative workbench."""

        row = next(
            (
                item
                for item in self.controller.authored_gameplay_placements()
                if str(getattr(item, "placement_id", "") or "") == str(placement_id or "")
            ),
            None,
        )
        source_name = str(
            getattr(row, "tag", "")
            or getattr(row, "template_resref", "")
            or placement_id
            or "conversation"
        ).strip().lower()
        clean_name = "_".join(part for part in source_name.replace(":", "_").split() if part)
        clean_name = "".join(character for character in clean_name if character.isalnum() or character == "_")
        suggested_resref = (f"dlg_{clean_name}" if clean_name else "dlg_new")[:16]
        resref = str(conversation_resref or "").strip().lower()[:16]
        if not resref and row is not None:
            resref = suggested_resref
            self._apply_placement_tab_creature_behavior(
                placement_id,
                str(getattr(row, "creature_behavior_role", "neutral") or "neutral"),
                resref,
                str(getattr(row, "creature_movement_mode", "stationary") or "stationary"),
            )
        context = {
            "source": "map_studio",
            "kind": "dialogue",
            "game": str(getattr(self.project, "game", "") or "K1").strip().upper(),
            "restype": "DLG",
            "resref": resref,
            "suggested_resref": resref or suggested_resref,
            "owner_kind": "creature",
            "owner_id": str(placement_id or ""),
            "field_name": "conversation_resref",
        }
        self.scriptingResourceEditRequested.emit(context)
        self._log(
            f"Opened Scripting & Dialogue Studio for creature conversation {resref or suggested_resref}.dlg."
        )

    def _handle_placement_tab_action(self, action: str, placement_id: str) -> None:
        try:
            if action == "snap_to_walkmesh":
                self.controller.snap_authored_gameplay_placement_to_walkmesh(placement_id)
                message = "Snapped gameplay placement to generated walkable WOK."
                selected_id = placement_id
            elif action == "duplicate":
                update = self.controller.duplicate_authored_gameplay_placement(placement_id)
                message = "Duplicated gameplay placement."
                selected_id = update.placement_id
            elif action == "delete":
                self.controller.remove_authored_gameplay_placement(placement_id)
                message = "Deleted gameplay placement."
                selected_id = ""
            elif action == "focus":
                self.select_item(placement_id)
                self.viewport_panel.focus_selected()
                return
            else:
                return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Gameplay Placement", str(exc))
            return
        if action == "snap_to_walkmesh":
            # Transform-only commit: promote the moved proxy in place instead
            # of rebuilding the combined preview model.
            self._refresh_map_studio_gameplay_change(
                f"{message} Previous exports/proofs are now stale.",
                placement_ids=(placement_id,),
            )
        else:
            self._refresh_all(f"{message} Previous exports/proofs are now stale.")
        if selected_id:
            self.select_item(selected_id)

    def snap_map_studio_selected_placement_to_ground(self) -> None:
        """Unreal-style End key: drop the selected GIT instance straight down."""

        placement_id = self.viewport_panel.selected_gameplay_placement_id()
        if not placement_id:
            self.statusBar().showMessage("End: select a placed map object or animated door first.", 5000)
            return
        try:
            _update, snap = self.controller.snap_authored_gameplay_placement_to_walkmesh(
                placement_id,
                downward_only=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Snap Placement to Ground", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            f"Snapped selected placement down to walkable ground face {snap.face_index}; previous exports/proofs are now stale.",
            placement_ids=(placement_id,),
        )
        self.select_item(placement_id)

    def set_authored_script_hook(self, scope: str, field_name: str, script_resref: str) -> None:
        try:
            if str(script_resref or "").strip():
                update = self.controller.set_authored_script_hook(
                    scope=scope,
                    field_name=field_name,
                    script_resref=script_resref,
                )
                message = f"Assigned {update.scope} script hook {update.field_name} -> {update.script_resref}."
            else:
                update = self.controller.remove_authored_script_hook(
                    scope=scope,
                    field_name=field_name,
                )
                message = f"Cleared {update.scope} script hook {update.field_name}."
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Script Hook", str(exc))
            return
        self._refresh_all(f"{message} Previous exports/proofs are now stale.")

    def _request_script_editor(self, scope: str, field_name: str, script_resref: str) -> None:
        """Deep-link an ARE/IFO hook into the standalone script editor."""

        clean_scope = str(scope or "area").strip().lower()
        clean_field = str(field_name or "").strip()
        resref = str(script_resref or "").strip().lower()[:16]
        field_slug = "".join(
            character for character in clean_field.lower() if character.isalnum() or character == "_"
        )
        suggested_resref = (f"gr_{field_slug}" if field_slug else "gr_script")[:16]
        if not resref:
            resref = suggested_resref
            self.set_authored_script_hook(clean_scope, clean_field, resref)
        context = {
            "source": "map_studio",
            "kind": "script",
            "game": str(getattr(self.project, "game", "") or "K1").strip().upper(),
            "restype": "NSS",
            "resref": resref,
            "suggested_resref": resref or suggested_resref,
            "owner_kind": f"{clean_scope}_script_hook",
            "owner_id": f"{clean_scope}:{clean_field}",
            "scope": clean_scope,
            "field_name": clean_field,
        }
        self.scriptingResourceEditRequested.emit(context)
        self._log(
            f"Opened Scripting & Dialogue Studio for {clean_scope} hook {clean_field}: "
            f"{resref or suggested_resref}.nss."
        )

    def _apply_map_studio_world_settings(self, values: object) -> None:
        settings = dict(values) if isinstance(values, dict) else {}
        try:
            update = self.controller.set_authored_world_lighting_settings(settings)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply World Settings", str(exc))
            return
        self._refresh_all(f"{update.summary} Previous ARE export/game proof is now stale.")

    def _apply_map_studio_surface_lightmap(self, values: object) -> None:
        settings = dict(values) if isinstance(values, dict) else {}
        resref = str(settings.get("lightmap_resref") or "").strip().lower()
        self.environment_tab.set_lightmap_status(
            f"Baking {resref or 'selected surface'} from ARE ambient and enabled room lights..."
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            result = self.controller.apply_authored_surface_lightmap(**settings)
        except Exception as exc:
            self.environment_tab.set_lightmap_status(f"Lightmap apply failed: {exc}")
            QtWidgets.QMessageBox.warning(self, "Bake & Apply Lightmap", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        viewport = getattr(self.viewport_panel, "viewport", None)
        invalidator = getattr(viewport, "invalidate_texture", None)
        if callable(invalidator) and resref:
            try:
                invalidator(resref)
            except Exception as exc:
                self._log(f"Lightmap renderer invalidation failed for {resref}: {exc}")
        for warning in tuple(getattr(result, "warnings", ()) or ()):
            self._log(f"Lightmap: {warning}")
        sidecar = getattr(result, "sidecar", None)
        resolution = int(getattr(sidecar, "width", 0) or 0)
        self._refresh_all(
            f"Applied {resref}.tpc ({resolution}x{resolution}) to the selected room surface; "
            "module packaging and manual KOTOR lighting proof are now stale."
        )

    def _create_map_studio_five_face_skybox(self, values: object) -> None:
        settings = dict(values) if isinstance(values, dict) else {}
        try:
            _room, message = self.controller.create_authored_five_face_skybox(**settings)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Create Five-Face Skybox", str(exc))
            return
        self._map_studio_show_skybox = True
        previous = self.toolbar.show_skybox.blockSignals(True)
        try:
            self.toolbar.show_skybox.setChecked(True)
        finally:
            self.toolbar.show_skybox.blockSignals(previous)
        self._refresh_all(message)

    def _create_map_studio_panorama_skybox(self, values: object) -> None:
        """Convert a panorama off-thread, then atomically author its room and TGAs."""

        settings = dict(values) if isinstance(values, dict) else {}
        if not str(getattr(self.project, "path", "") or "").strip():
            QtWidgets.QMessageBox.information(
                self,
                "Create Skybox from Panorama / HDR",
                "Save this KMAP first. The five generated TGA textures are stored beside the project.",
            )
            self.save_kmap_as()
            if not str(getattr(self.project, "path", "") or "").strip():
                return
        source_path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose Equirectangular Sky Panorama",
            "",
            (
                "Panorama / HDR images (*.hdr *.exr *.pfm *.pic *.rgbe *.png *.jpg *.jpeg *.tif *.tiff);;"
                "HDR images (*.hdr *.exr *.pfm *.pic *.rgbe);;"
                "Standard images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*.*)"
            ),
        )
        if not source_path:
            return
        options_dialog = MapStudioPanoramaSkyboxDialog(source_path, self)
        if options_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        conversion_settings = options_dialog.settings()
        preparation_values = {
            key: settings.get(key, "")
            for key in (
                "room_resref",
                "north_texture",
                "east_texture",
                "south_texture",
                "west_texture",
                "top_texture",
            )
        }
        preparation_values.update(conversion_settings)
        progress = QtWidgets.QProgressDialog(
            "Decoding, tone-mapping, and projecting five KOTOR skybox panels…",
            "",
            0,
            0,
            self,
        )
        progress.setObjectName("mapStudioPanoramaSkyboxProgressDialog")
        progress.setWindowTitle("Convert Skybox Panorama")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        panorama_button_was_enabled = self.environment_tab.sky_panorama_button.isEnabled()
        self.environment_tab.sky_panorama_button.setEnabled(False)
        future = _MAP_STUDIO_SKYBOX_EXECUTOR.submit(
            self.controller.prepare_panorama_skybox,
            source_path,
            **preparation_values,
        )
        wait_loop = QtCore.QEventLoop(self)
        poll_timer = QtCore.QTimer(self)
        poll_timer.setInterval(40)

        def poll_skybox_conversion() -> None:
            if future.done():
                poll_timer.stop()
                wait_loop.quit()

        poll_timer.timeout.connect(poll_skybox_conversion)
        poll_timer.start()
        if not future.done():
            wait_loop.exec()
        try:
            preparation = future.result()
        except Exception as exc:
            self.environment_tab.set_skybox_status(f"Panorama conversion failed: {exc}")
            QtWidgets.QMessageBox.warning(self, "Create Skybox from Panorama / HDR", str(exc))
            self.statusBar().showMessage(f"Skybox conversion failed: {exc}", 9000)
            return
        finally:
            poll_timer.stop()
            poll_timer.deleteLater()
            wait_loop.deleteLater()
            progress.close()
            progress.deleteLater()
            self.environment_tab.sky_panorama_button.setEnabled(panorama_button_was_enabled)

        self.environment_tab.set_skybox_status(
            "Projection complete. Writing five project TGAs and the visual-only sky room…"
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            result = self.controller.commit_panorama_skybox(
                preparation,
                half_extent=float(settings.get("half_extent", 500.0)),
                bottom_z=float(settings.get("bottom_z", -500.0)),
                top_z=float(settings.get("top_z", 500.0)),
                visible_rooms=tuple(settings.get("visible_rooms") or ()),
            )
        except Exception as exc:
            self.environment_tab.set_skybox_status(f"Skybox creation rolled back: {exc}")
            QtWidgets.QMessageBox.warning(self, "Create Skybox from Panorama / HDR", str(exc))
            self.statusBar().showMessage(f"Skybox creation rolled back: {exc}", 9000)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._map_studio_show_skybox = True
        previous = self.toolbar.show_skybox.blockSignals(True)
        try:
            self.toolbar.show_skybox.setChecked(True)
        finally:
            self.toolbar.show_skybox.blockSignals(previous)
        self.environment_tab.set_skybox_status(result.message)
        self._reset_map_studio_texture_paint_session()
        self.viewport_panel.set_project_texture_paths(self.project)
        self.texture_paint_tab.set_project(self.project)
        self._refresh_all(result.message)

    def _create_map_studio_sky_traffic(self, values: object) -> None:
        settings = dict(values) if isinstance(values, dict) else {}
        try:
            _traffic, message, validation = self.controller.create_authored_sky_traffic(**settings)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Create Sky Traffic", str(exc))
            return
        for warning in tuple(getattr(validation, "warnings", ()) or ()):
            self._log(f"Sky traffic: {warning}")
        self._refresh_all(message)

    def export_fbx(self, dry_run: bool = False) -> None:
        self._export_level_geometry("fbx", dry_run)

    def _export_level_geometry(self, format_name: str, dry_run: bool = False) -> None:
        from src.core.level import LevelExportOptions
        from src.gui.qt_lib.dialogs.level_scene_export_dialog import run_level_scene_export

        label = format_name.upper()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export Whole Level as {label}", f"{self.project.name}.{format_name}",
            f"{label} files (*.{format_name})")
        if not path:
            return
        if not Path(path).suffix:
            path += f".{format_name}"
        try:
            result = run_level_scene_export(
                self, self.project, path, LevelExportOptions(dry_run=dry_run),
                resource_manager=self.resource_manager,
                template_resources=self.controller.authored_project_extra_resources())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Level Export Failed", str(exc))
            return
        self.validation_panel.set_issues(result.issues)
        self._log(result.message)
        if result.manifest_path:
            self._log(f"Manifest: {result.manifest_path}")
        for warning in result.warnings:
            self._log(warning)
        if result.ok:
            self._last_output_dir = str(Path(path).parent)
            QtWidgets.QMessageBox.information(self, "Level Export Complete", result.message)
        else:
            QtWidgets.QMessageBox.warning(self, "Level Export Failed", result.message)

    def open_output_folder(self) -> None:
        if self._last_output_dir:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._last_output_dir))
        else:
            self._log("No output folder has been generated yet.")

    def _map_studio_pie_context_settings(self) -> dict[str, Any]:
        getter = getattr(self.controller, "map_studio_pie_context_settings", None)
        if not callable(getter):
            return {
                "player_role": "normal_pc",
                "player_gender": "male",
                "dialogue_start_overrides": {},
            }
        try:
            values = getter()
        except Exception as exc:
            self._log(f"PIE conversation context could not be read: {exc}")
            return {
                "player_role": "normal_pc",
                "player_gender": "male",
                "dialogue_start_overrides": {},
            }
        return dict(values or {})

    def _update_map_studio_pie_context_settings(self, **changes: object) -> bool:
        updater = getattr(self.controller, "update_map_studio_pie_context_settings", None)
        if not callable(updater):
            updater = getattr(self.controller, "update_map_studio_pie_context", None)
        if not callable(updater):
            self._log("PIE conversation context editing is unavailable in this build.")
            return False
        try:
            updater(**changes)
        except Exception as exc:
            self._log(f"PIE conversation context could not be updated: {exc}")
            return False
        self.setWindowTitle(
            f"Ghost-Studio Map Studio - Level Editor - {self.project.name}{' *' if self.project.dirty else ''}"
        )
        self._refresh_map_studio_pie_context_panel()
        return True

    def _refresh_map_studio_pie_context_panel(self) -> None:
        panel = getattr(self, "pie_context_panel", None)
        if panel is None:
            return
        settings = self._map_studio_pie_context_settings()
        catalog_getter = getattr(self.controller, "map_studio_pie_dialogue_catalog", None)
        catalog: object = ()
        unavailable_reason = ""
        if callable(catalog_getter):
            try:
                catalog = tuple(catalog_getter() or ())
            except Exception as exc:
                unavailable_reason = f"Dialogue resources could not be inspected for PIE: {exc}"
        if not catalog and not unavailable_reason:
            unavailable_reason = (
                "No dialogue resources are available in the loaded content. A bare .lyt contains room layout only; "
                "attach or import a .mod or hydrated .kmap to tune conversation starts."
            )
        panel.set_catalog(
            catalog,
            player_role=str(settings.get("player_role") or "normal_pc"),
            player_gender=str(settings.get("player_gender") or "male"),
            overrides=settings.get("dialogue_start_overrides") or {},
            unavailable_reason=unavailable_reason,
        )
        selected_resref = ""
        combo = getattr(panel, "conversation_combo", None)
        if combo is not None and combo.count() > 0:
            selected_resref = str(combo.currentData() or "").strip().lower()
        starter_getter = getattr(panel, "current_starter_link_id", None)
        starter_link = starter_getter() if callable(starter_getter) else ""
        self._update_map_studio_pie_opening_preview(selected_resref, starter_link)

    def _update_map_studio_pie_opening_preview(self, conversation_resref: str, starter_link_id: str) -> None:
        """Resolve and display the opening NPC line for the selected conversation."""

        panel = getattr(self, "pie_context_panel", None)
        if panel is None or not hasattr(panel, "set_opening_preview"):
            return
        resref = str(conversation_resref or "").strip().lower()
        if not resref:
            panel.clear_opening_preview()
            return
        previewer = getattr(self.controller, "map_studio_pie_dialogue_preview", None)
        if not callable(previewer):
            panel.clear_opening_preview()
            return
        try:
            preview = previewer(resref, starter_link_id=str(starter_link_id or "").strip().lower())
        except Exception as exc:  # the preview must never break the panel refresh
            panel.set_opening_preview("", warning=f"Opening line could not be resolved: {exc}")
            return
        panel.set_opening_preview(
            str(preview.get("text") or ""),
            forced=bool(preview.get("forced")),
            blocked=bool(preview.get("blocked")),
            warning=str(preview.get("warning") or ""),
        )

    def _set_map_studio_pie_player_context(self, player_role: str, player_gender: str) -> None:
        if self._update_map_studio_pie_context_settings(
            player_role=str(player_role or "normal_pc"),
            player_gender=str(player_gender or "male"),
        ):
            self.statusBar().showMessage("PIE player conversation context updated; it applies on the next Play.")

    def _set_map_studio_pie_starter_override(
        self,
        conversation_resref: str,
        starter_link_id: str,
        resource_sha256: str,
    ) -> None:
        resref = str(conversation_resref or "").strip().lower()
        if not resref:
            return
        settings = self._map_studio_pie_context_settings()
        overrides = {
            str(key or "").strip().lower(): dict(value or {})
            for key, value in dict(settings.get("dialogue_start_overrides") or {}).items()
            if str(key or "").strip() and isinstance(value, dict)
        }
        link_id = str(starter_link_id or "").strip()
        if link_id:
            overrides[resref] = {
                "starter_link_id": link_id,
                "resource_sha256": str(resource_sha256 or "").strip().lower(),
            }
        else:
            overrides.pop(resref, None)
        if self._update_map_studio_pie_context_settings(dialogue_start_overrides=overrides):
            message = (
                f"PIE will preview {resref}.dlg from starting link {link_id}; Active conditions are bypassed."
                if link_id
                else f"{resref}.dlg returned to canonical Auto start selection."
            )
            self.statusBar().showMessage(message)

    def _reset_map_studio_pie_context(self) -> None:
        if self._update_map_studio_pie_context_settings(
            player_role="normal_pc",
            player_gender="male",
            global_numbers={},
            global_booleans={},
            local_booleans={},
            dialogue_start_overrides={},
        ):
            self.statusBar().showMessage("PIE conversation context reset to a clean normal-player preview.")

    def select_item(self, item_id: str) -> None:
        item_id = str(item_id or "")
        self._map_studio_selected_rooms = []
        self.viewport_panel.clear_map_studio_room_geometry_selection()
        self.viewport_panel.set_selected_gameplay_placement_ids(())
        self.viewport_panel.set_map_studio_scene_selection_ids((item_id,) if item_id else ())
        if item_id.startswith("authored_room:"):
            room_resref = str(item_id).split(":", 1)[1].strip().lower()
            self._map_studio_selected_rooms = [room_resref] if room_resref else []
            self.controller.model.select(item_id)
            self.outliner.select_id(item_id)
            self.properties.set_selection(item_id)
            self.workflow_panel.set_selection_context(f"room: {room_resref}")
            self.statusBar().showMessage(f"Selected room {room_resref} — press Delete to remove it.")
            return
        primitive_identity = self._parse_map_studio_primitive_outliner_id(item_id)
        if primitive_identity is not None:
            room_resref, primitive_name = primitive_identity
            self.controller.model.select(item_id)
            self.outliner.select_id(item_id)
            self._select_authored_room_primitive(room_resref, primitive_name)
            self.properties.set_selection(item_id)
            label = self._selected_item_label(item_id)
            self.workflow_panel.set_selection_context(label)
            self.statusBar().showMessage(f"Selected {label or primitive_name}")
            return
        self.controller.model.select(item_id)
        self.outliner.select_id(item_id)
        self.viewport_panel.select_id(item_id)
        self.placement_tab.set_selected_placement(item_id if str(item_id).startswith("authored:") else "")
        self.properties.set_selection(item_id)
        self.pie_context_panel.focus_conversation_for_owner(item_id)
        self.workflow_panel.set_selection_context(self._selected_item_label(item_id))
        self.statusBar().showMessage(f"Selected {item_id}")

    def _select_map_studio_items(self, item_ids: object) -> None:
        """Accept Maya-style extended selection from the scene outliner."""

        selected = [str(value or "") for value in tuple(item_ids or ()) if str(value or "")]
        if not selected:
            self.controller.model.select_many(())
            self.outliner.select_ids(())
            self.viewport_panel.set_selected_gameplay_placement_ids(())
            self.viewport_panel.set_selected_room_primitives(())
            self.viewport_panel.set_map_studio_scene_selection_ids(())
            self._map_studio_selected_rooms = []
            self.properties.set_selection("")
            self.workflow_panel.set_selection_context("")
            self.statusBar().showMessage("Map Studio selection cleared.", 2500)
            return
        if len(selected) == 1:
            self.select_item(selected[0])
            return
        self.viewport_panel.clear_map_studio_room_geometry_selection()
        self.controller.model.select_many(selected)
        self.placement_tab.set_selected_placement("")
        self.outliner.select_ids(selected)
        self.viewport_panel.set_map_studio_scene_selection_ids(selected)
        primitive_entries = [
            identity
            for identity in (self._parse_map_studio_primitive_outliner_id(item_id) for item_id in selected)
            if identity is not None
        ]
        self.viewport_panel.set_selected_room_primitives(primitive_entries)
        placement_ids = [
            item_id
            for item_id in selected
            if item_id == "entry_point" or str(item_id).startswith("authored:")
        ]
        self.viewport_panel.set_selected_gameplay_placement_ids(placement_ids)
        authored_room_resrefs = {
            str(value or "").strip().lower()
            for value in tuple(self.controller.authored_room_resrefs() or ())
            if str(value or "").strip()
        }
        self._map_studio_selected_rooms = [
            (item_id.split(":", 1)[1].strip().lower() if item_id.startswith("authored_room:") else item_id.lower())
            for item_id in selected
            if (
                (item_id.startswith("authored_room:") and item_id.split(":", 1)[1].strip().lower() in authored_room_resrefs)
                or item_id.lower() in authored_room_resrefs
            )
        ]
        active = selected[-1]
        self.properties.set_selection(active)
        self.pie_context_panel.focus_conversation_for_owner(active)
        if primitive_entries:
            room_resref, primitive_name = primitive_entries[-1]
            self.builder_tab.select_room_primitive(room_resref, primitive_name)
            self._refresh_map_studio_selected_primitive_transform_overlay()
        self.workflow_panel.set_selection_context(f"{len(selected)} scene objects selected")
        self.statusBar().showMessage(
            f"Selected {len(selected)} scene objects; transform, Combine Meshes, or Separate Shells is ready."
        )

    def _selected_item_label(self, item_id: str) -> str:
        if not item_id:
            return ""
        if item_id == "entry_point":
            return "Player Start (IFO module entry point)"
        if str(item_id).startswith("authored_room:"):
            return f"room: {str(item_id).split(':', 1)[1]}"
        primitive_identity = self._parse_map_studio_primitive_outliner_id(item_id)
        if primitive_identity is not None:
            room_resref, primitive_name = primitive_identity
            row = self._authored_room_primitive_row(room_resref, primitive_name)
            primitive_type = str(getattr(row, "primitive_type", "") or "primitive") if row is not None else "primitive"
            return f"{primitive_type}: {primitive_name} ({room_resref})"
        authored = next((row for row in self.controller.authored_gameplay_placements() if getattr(row, "placement_id", "") == item_id), None)
        if authored is not None:
            kind = str(getattr(authored, "kind", "resource") or "resource")
            tag = str(getattr(authored, "tag", "") or getattr(authored, "template_resref", "") or item_id)
            return f"{kind}: {tag}"
        authored_light = next((row for row in self.controller.authored_room_lights() if getattr(row, "light_id", "") == item_id), None)
        if authored_light is not None:
            name = str(getattr(authored_light, "name", "") or item_id)
            room = str(getattr(authored_light, "room_resref", "") or "")
            return f"room light: {name}" + (f" ({room})" if room else "")
        item = self.project.find_room(item_id) or self.project.find_module(item_id) or self.project.find_blueprint(item_id)
        if item is None:
            return item_id
        if hasattr(item, "module_name"):
            return f"module: {getattr(item, 'module_name', item_id)}"
        if hasattr(item, "room_id"):
            return f"room: {getattr(item, 'name', item_id)}"
        return f"blueprint: {getattr(item, 'name', item_id)}"

    def _set_map_studio_pie_authoring_enabled(self, enabled: bool) -> None:
        """Suspend authoring controls while the read-only simulation owns input."""

        if enabled:
            for control, was_enabled in reversed(self._map_studio_pie_control_states):
                try:
                    control.setEnabled(bool(was_enabled))
                except RuntimeError:
                    continue
            self._map_studio_pie_control_states = []
            return
        if self._map_studio_pie_control_states:
            return
        controls = [
            self.new_action,
            self.open_action,
            self.save_action,
            self.save_as_action,
            self.import_module_action,
            self.import_mod_file_action,
            self.import_stock_module_action,
            self.convert_all_stock_rooms_action,
            self.import_library_asset_action,
            self.import_texture_action,
            self.export_fbx_action,
            self.export_obj_action,
            self.export_package_action,
            self.undo_action,
            self.redo_action,
            self.delete_action,
            self.duplicate_action,
            self.rename_action,
            self.build_action,
            self.generate_walls_action,
            self.paint_walkmesh_action,
            self.map_studio_workspace_combo,
            self.map_studio_open_workspace_button,
            self.map_studio_tool_belt_preset_combo,
            self.map_studio_customize_tool_belt_button,
            self.map_studio_custom_tool_add_button,
            self.map_studio_command_run_button,
            self.outliner,
            self.asset_browser,
            self.workflow_tabs,
            self.properties,
            self.pie_context_panel,
            getattr(self.viewport_panel, "scene_table", None),
            getattr(self.viewport_panel, "translate_gizmo_button", None),
            getattr(self.viewport_panel, "rotate_gizmo_button", None),
            getattr(self.viewport_panel, "scale_gizmo_button", None),
            getattr(self.viewport_panel, "terrain_brush_box", None),
            getattr(self.toolbar, "selection_mode", None),
            self.map_studio_universal_transform_shortcut,
            self.map_studio_translate_gizmo_shortcut,
            self.map_studio_rotate_gizmo_shortcut,
            self.map_studio_scale_gizmo_shortcut,
            self.map_studio_delete_selection_shortcut,
            self.map_studio_ground_snap_shortcut,
            self.map_studio_vertex_snap_shortcut,
            self.map_studio_transform_level_snap_shortcut,
        ]
        controls.extend(
            button
            for key, button in getattr(self.toolbar, "buttons", {}).items()
            if key not in {"simulate", "validate"}
        )
        controls.extend(getattr(self.toolbar, "tool_belt_buttons", {}).values())
        seen: set[int] = set()
        for control in controls:
            if control is None or id(control) in seen:
                continue
            seen.add(id(control))
            try:
                was_enabled = bool(control.isEnabled())
                self._map_studio_pie_control_states.append((control, was_enabled))
                control.setEnabled(False)
            except RuntimeError:
                continue

    def _set_map_studio_pie_command_active(self, active: bool) -> None:
        """Keep the menu command and the prominent toolbar control in one PIE state."""

        active = bool(active)
        action = self.simulate_action
        tooltip = (
            "Stop Play in Editor (Alt+P) and restore the authoring camera and tools."
            if active
            else (
                "Play in Editor (Alt+P) using the current walkmesh, player camera, creatures, and ambient sound. "
                "This is a GhostStudio simulation, not KOTOR engine proof."
            )
        )
        blocked = action.blockSignals(True)
        try:
            action.setChecked(active)
            action.setText("Stop Play in Editor" if active else "Play in Editor")
            action.setIconText("Stop" if active else "Play")
            action.setProperty("mapStudioPIEState", "playing" if active else "editing")
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        finally:
            action.blockSignals(blocked)
        self.toolbar.set_simulation_active(active)

    def toggle_map_studio_pie(self, _checked: bool = False) -> None:
        """Start or stop the persistent Map Studio simulation mode."""

        if self._map_studio_pie_session is not None:
            self._stop_map_studio_pie()
        else:
            self._start_map_studio_pie()

    def _map_studio_pie_player_settings(self) -> tuple[str, str]:
        extra = dict(getattr(self.project, "extra", {}) or {})
        player_settings = dict(extra.get("pie_player") or {})
        body_resref = str(player_settings.get("body_resref") or "pmbam").strip().lower()
        head_resref = str(player_settings.get("head_resref") or "pmhc01").strip().lower()
        return body_resref, head_resref

    def _set_map_studio_pie_player_facial_detail_visible(self, visible: bool) -> None:
        """Use retail-distance facial LOD outside close dialogue framing."""

        actor = self._map_studio_pie_actor
        root = getattr(actor, "root_node", None)
        if root is None:
            return
        # These internal mouth, eyelid, and separate eyeball layers are
        # sub-pixel at the DEFAULT
        # exploration-camera distance but each costs a complete animated draw.
        # Preserve the complete head skin and restore every close facial layer
        # as soon as dialogue framing begins.
        detail_names = {
            "tongue",
            "teethua",
            "teethla",
            "eyellid",
            "eyerlid",
            "eyela",
            "eyera",
        }
        changed = False
        stack = [root]
        visited: set[int] = set()
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.extend(tuple(getattr(node, "children", ()) or ()))
            if str(getattr(node, "name", "") or "").strip().lower() not in detail_names:
                continue
            wanted_hidden = not bool(visible)
            if bool(getattr(node, "_gr_hidden", False)) != wanted_hidden:
                setattr(node, "_gr_hidden", wanted_hidden)
                changed = True
            setattr(node, "_gr_map_studio_pie_facial_detail_lod", True)
        if changed:
            preview_model = self._map_studio_pie_runtime_preview_model
            if preview_model is not None:
                setattr(
                    preview_model,
                    "_gr_classification_revision",
                    int(getattr(preview_model, "_gr_classification_revision", 0) or 0) + 1,
                )

    @staticmethod
    def _map_studio_pie_resource_revision(manager: Any) -> int:
        """Return a stable non-negative revision for cache keying."""

        try:
            return max(0, int(getattr(manager, "revision", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _map_studio_pie_player_cache_state(
        self,
    ) -> tuple[dict[tuple[Any, ...], tuple[Any, str]], dict[tuple[Any, ...], Event], Lock]:
        """Return lazily compatible cache state for older construction harnesses."""

        cache = getattr(self, "_map_studio_pie_player_model_cache", None)
        if cache is None:
            cache = {}
            self._map_studio_pie_player_model_cache = cache
        pending = getattr(self, "_map_studio_pie_player_prewarm_pending", None)
        if not isinstance(pending, dict):
            pending = {}
            self._map_studio_pie_player_prewarm_pending = pending
        cache_lock = getattr(self, "_map_studio_pie_player_cache_lock", None)
        if cache_lock is None:
            cache_lock = Lock()
            self._map_studio_pie_player_cache_lock = cache_lock
        return cache, pending, cache_lock

    def _prewarm_map_studio_pie_player_model(self) -> None:
        """Warm the composed PIE player model off-thread.

        First Play measured ~30-50 s on large modules because the player body
        and head load through the MDL reader on the Qt thread. A background
        worker prepares the reusable composed model without touching
        SuperModelResolver's process-global manager/cache; inherited animation
        setup remains on the GUI-thread Play path.
        """

        manager = getattr(self, "resource_manager", None)
        strict_loader = getattr(manager, "load_model_strict", None)
        if manager is None or not callable(strict_loader):
            return
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        body_resref, head_resref = self._map_studio_pie_player_settings()
        manager_revision = self._map_studio_pie_resource_revision(manager)
        cache_key = (manager, manager_revision, game, body_resref, head_resref)
        cache, pending, cache_lock = self._map_studio_pie_player_cache_state()
        completion = Event()
        with cache_lock:
            for prior_key in tuple(cache):
                if len(prior_key) < 2 or prior_key[0] is not manager or prior_key[1] != manager_revision:
                    cache.pop(prior_key, None)
            if cache_key in cache or cache_key in pending:
                return
            pending[cache_key] = completion

        def _worker() -> None:
            warning = ""
            actor_model = None
            try:
                body_model = strict_loader(body_resref, game)
                if body_model is None:
                    return
                head_model = None
                if head_resref:
                    try:
                        head_model = strict_loader(head_resref, game)
                    except Exception as exc:
                        warning = f"The player head could not be loaded; PIE is using the body model only ({exc})."
                actor_model = body_model
                if head_model is not None:
                    try:
                        from src.systems.bas.preview_composer import build_bas_preview_model

                        actor_model = build_bas_preview_model(
                            body_model=body_model,
                            attachment_models={"head": head_model},
                            name=f"{body_resref}_{head_resref}_pie_player",
                        )
                    except Exception as exc:
                        actor_model = body_model
                        warning = f"The player head could not be attached; PIE is using the body model only ({exc})."
            except Exception:
                pass
            finally:
                with cache_lock:
                    if (
                        actor_model is not None
                        and getattr(self, "resource_manager", None) is manager
                        and self._map_studio_pie_resource_revision(manager) == manager_revision
                    ):
                        cache[cache_key] = (actor_model, warning)
                    pending.pop(cache_key, None)
                    completion.set()

        import threading

        threading.Thread(target=_worker, name="map_studio_pie_player_prewarm", daemon=True).start()

    def _create_map_studio_pie_player_actor(self, session: Any, preview_model: Any, game: str) -> str:
        """Attach a runtime-only PMBAM/PMHC01 player with inherited clips."""

        self._map_studio_pie_actor = None
        self._map_studio_pie_animation_engine = None
        self._map_studio_pie_animation_name = ""
        manager = getattr(self, "resource_manager", None)
        if manager is None or preview_model is None:
            return "Animated player preview is unavailable because the game resource library or resident map model is missing."
        body_resref, head_resref = self._map_studio_pie_player_settings()
        strict_loader = getattr(manager, "load_model_strict", None)
        if not callable(strict_loader):
            return "Animated player preview requires target-game-strict model loading."
        # Loading pmbam + pmhc01 and their supermodel animation chains through
        # the MDL reader measured ~49 s per Play press on large-module
        # sessions; the composed actor is immutable to PIE (attachment deep
        # copies the DAG), so reuse it across Play presses.
        manager_revision = self._map_studio_pie_resource_revision(manager)
        normalized_game = str(game or "K1").strip().upper()
        cache_key = (manager, manager_revision, normalized_game, body_resref, head_resref)
        cache, pending, cache_lock = self._map_studio_pie_player_cache_state()
        with cache_lock:
            for prior_key in tuple(cache):
                if len(prior_key) < 2 or prior_key[0] is not manager or prior_key[1] != manager_revision:
                    cache.pop(prior_key, None)
            cached = cache.get(cache_key)
            pending_completion = pending.get(cache_key)
            owns_load = cached is None and pending_completion is None
            if owns_load:
                pending_completion = Event()
                pending[cache_key] = pending_completion
        warning = ""
        if cached is not None:
            actor_model, warning = cached
        elif not owns_load:
            completed = pending_completion.wait(_MAP_STUDIO_PIE_PLAYER_PREWARM_WAIT_SECONDS)
            with cache_lock:
                cached = cache.get(cache_key)
            if cached is not None:
                actor_model, warning = cached
            elif (
                getattr(self, "resource_manager", None) is not manager
                or self._map_studio_pie_resource_revision(manager) != manager_revision
            ):
                return (
                    "Game resources changed while the animated player model was preparing; "
                    "this PIE run uses the simulation marker only. Stop and Play again to load the updated player."
                )
            elif completed or pending_completion.is_set():
                return (
                    "Background player preparation did not produce a usable animated model; "
                    "this PIE run uses the simulation marker only. Stop and Play again to retry."
                )
            else:
                return (
                    "The animated player model is still preparing in the background; "
                    "this PIE run uses the simulation marker only. Stop and Play again after preparation completes."
                )
        else:
            actor_model = None
            load_error = ""
            try:
                body_model = strict_loader(body_resref, normalized_game)
            except Exception as exc:
                body_model = None
                load_error = f"Animated player body could not be loaded: {exc}"
            if body_model is None and not load_error:
                load_error = (
                    f"Animated player body {body_resref}.mdl was not found in the {normalized_game} resource library."
                )
            if body_model is not None:
                actor_model = body_model
            head_model = None
            if body_model is not None and head_resref:
                try:
                    head_model = strict_loader(head_resref, normalized_game)
                except Exception as exc:
                    warning = f"The player head could not be loaded; PIE is using the body model only ({exc})."
            if head_model is not None:
                try:
                    # Use the same BAS composition contract as Character Studio.
                    # Heads are socket-following attachment layers and must not be
                    # merged into the body skin palette as ordinary hierarchy.
                    from src.systems.bas.preview_composer import build_bas_preview_model

                    actor_model = build_bas_preview_model(
                        body_model=body_model,
                        attachment_models={"head": head_model},
                        name=f"{body_resref}_{head_resref}_pie_player",
                    )
                except Exception as exc:
                    actor_model = body_model
                    warning = f"The player head could not be attached; PIE is using the body model only ({exc})."
            with cache_lock:
                resources_unchanged = (
                    getattr(self, "resource_manager", None) is manager
                    and self._map_studio_pie_resource_revision(manager) == manager_revision
                )
                if actor_model is not None and resources_unchanged:
                    cache[cache_key] = (actor_model, warning)
                pending.pop(cache_key, None)
                pending_completion.set()
            if load_error:
                return load_error
            if not resources_unchanged:
                return (
                    "Game resources changed while the animated player model was loading; "
                    "this PIE run uses the simulation marker only. Stop and Play again to load the updated player."
                )
        if (
            getattr(self, "resource_manager", None) is not manager
            or self._map_studio_pie_resource_revision(manager) != manager_revision
        ):
            return (
                "Game resources changed before the animated player could be attached; "
                "this PIE run uses the simulation marker only. Stop and Play again to load the updated player."
            )
        actor = None
        try:
            from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
            from src.core.modules.map_studio_pie import attach_map_studio_pie_actor

            SuperModelResolver.configure(manager)
            actor = attach_map_studio_pie_actor(
                preview_model,
                actor_model,
                position=tuple(session.state.position),
                facing_radians=float(session.state.facing_radians),
                recompute_bounds=False,
            )
            if actor is None:
                return "The player MDL loaded, but its Odyssey hierarchy could not be attached to the map preview."
            engine = AnimationEngine(actor_model)
            if not engine.play("pause1", loop=True, blend=False):
                engine.play("idlepose", loop=True, blend=False)
            self._map_studio_pie_actor = actor
            self._set_map_studio_pie_player_facial_detail_visible(False)
            self._map_studio_pie_animation_engine = engine
            self._map_studio_pie_animation_name = str(
                getattr(getattr(engine, "current_animation", None), "name", "") or ""
            ).lower()
            self._map_studio_pie_animation_run = False
        except Exception as exc:
            if actor is not None:
                try:
                    actor.detach(recompute_bounds=False)
                except Exception:
                    pass
            self._map_studio_pie_actor = None
            self._map_studio_pie_animation_engine = None
            return f"Animated player setup failed: {exc}"
        return warning

    def _hide_map_studio_pie_player_start_preview(self, preview_model: Any) -> None:
        """Hide the complete authoring Player Start while its PIE actor exists.

        The authoring preview is a single tagged parent containing the body,
        head, and any attachment meshes. Removing that parent keeps the parts
        indivisible and prevents the runtime player from being drawn directly
        over its editor-only twin.
        """

        self._map_studio_pie_hidden_player_start_groups = []
        if self._map_studio_pie_actor is None:
            return
        root = getattr(preview_model, "root_node", None)
        if root is None:
            return
        original_children = list(tuple(getattr(root, "children", ()) or ()))
        hidden = [
            (index, node)
            for index, node in enumerate(original_children)
            if str(getattr(node, "_gr_map_studio_placement_kind", "") or "").strip().lower()
            == "entry_point"
            or str(getattr(node, "_gr_map_studio_placement_id", "") or "").strip().lower()
            == "entry_point"
        ]
        if not hidden:
            return
        hidden_ids = {id(node) for _index, node in hidden}
        root.children = [node for node in original_children if id(node) not in hidden_ids]
        for node in tuple(root.children or ()):
            node.parent = root
        self._map_studio_pie_hidden_player_start_groups = hidden

    def _restore_map_studio_pie_player_start_preview(self, preview_model: Any) -> None:
        """Restore the authoring Player Start after all transient PIE DAGs leave."""

        root = getattr(preview_model, "root_node", None)
        if root is not None:
            for original_index, group in self._map_studio_pie_hidden_player_start_groups:
                if any(group is child for child in tuple(root.children or ())):
                    continue
                group.parent = root
                root.children.insert(min(max(0, int(original_index)), len(root.children)), group)
        self._map_studio_pie_hidden_player_start_groups = []

    def _create_map_studio_pie_party_actors(self, session: Any, preview_model: Any, game: str) -> str:
        """Attach runtime-only companion actors that trail the player.

        Companions come from the configurable PIE party roster (UTC resrefs).
        Each is a SEPARATE retained actor (never the creature cohort), placed at
        its walkmesh-snapped follow slot and idle-posed; positions update per
        tick. Body model resolved via the same appearance path creatures use.
        Fully guarded so a resolution/load failure never breaks PIE.
        """

        self._map_studio_pie_party_actors = []
        manager = getattr(self, "resource_manager", None)
        if manager is None or preview_model is None or session is None:
            return ""
        roster = tuple(self._map_studio_pie_context_settings().get("party_roster") or ())
        follow = getattr(session, "party_follow_targets", None)
        if not roster or not callable(follow):
            return ""
        resolver = getattr(self.controller, "_map_studio_stock_template_resolver", None)
        strict_loader = getattr(manager, "load_model_strict", None)
        resolve_body = getattr(resolver, "creature_model", None)
        if resolver is None or not callable(strict_loader) or not callable(resolve_body):
            return "PIE party companions could not be resolved (resource resolver unavailable)."
        try:
            from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
            from src.core.modules.map_studio_pie import attach_map_studio_pie_actor

            SuperModelResolver.configure(manager)
        except Exception as exc:
            return f"PIE party companion actors are unavailable ({exc})."
        game_tag = str(game or "K1").strip().upper()
        targets = tuple(follow(len(roster)))
        warnings: list[str] = []
        for slot, resref in enumerate(roster, start=1):
            resref = str(resref or "").strip().lower()
            if not resref:
                continue
            try:
                body_resref = str(resolve_body(resref) or "").strip().lower()
            except Exception:
                body_resref = ""
            if not body_resref:
                warnings.append(f"companion {resref} has no resolvable body model")
                continue
            try:
                actor_model = strict_loader(body_resref, game_tag)
            except Exception as exc:
                warnings.append(f"companion {resref} body {body_resref} failed to load ({exc})")
                continue
            if actor_model is None:
                warnings.append(f"companion {resref} body {body_resref} not found")
                continue
            position = targets[slot - 1] if slot - 1 < len(targets) else tuple(session.state.position)
            try:
                actor = attach_map_studio_pie_actor(
                    preview_model,
                    actor_model,
                    position=tuple(float(v) for v in tuple(position)[:3]),
                    facing_radians=float(session.state.facing_radians),
                    recompute_bounds=False,
                )
            except Exception as exc:
                warnings.append(f"companion {resref} could not attach ({exc})")
                continue
            if actor is None:
                continue
            engine = None
            try:
                engine = AnimationEngine(actor_model)
                if not engine.play("pause1", loop=True, blend=False):
                    engine.play("idlepose", loop=True, blend=False)
            except Exception:
                engine = None
            self._map_studio_pie_party_actors.append(
                {"actor": actor, "engine": engine, "slot": slot, "resref": resref}
            )
        return "; ".join(warnings)

    def _update_map_studio_pie_party_actors(self, delta_time: float) -> None:
        """Drive each companion actor to its current follow slot with an idle pose."""

        entries = self._map_studio_pie_party_actors
        if not entries:
            return
        session = self._map_studio_pie_session
        follow = getattr(session, "party_follow_targets", None)
        if session is None or not callable(follow):
            return
        targets = tuple(follow(len(entries)))
        step = max(0.0, min(float(delta_time), 0.25))
        facing = float(session.state.facing_radians)
        for entry in entries:
            actor = entry.get("actor")
            slot = int(entry.get("slot", 0) or 0)
            if actor is None or slot < 1 or slot - 1 >= len(targets):
                continue
            position = tuple(float(v) for v in tuple(targets[slot - 1])[:3])
            try:
                actor.set_transform(position, facing)
            except Exception:
                pass
            engine = entry.get("engine")
            if engine is not None:
                try:
                    engine.advance(step)
                except Exception:
                    pass

    def _update_map_studio_pie_player_actor(self, frame: Any, delta_time: float) -> tuple[Any, ...] | None:
        """Advance locomotion and return one deferred retained-renderer row."""

        actor = self._map_studio_pie_actor
        engine = self._map_studio_pie_animation_engine
        if actor is None or engine is None:
            return None
        speed = math.sqrt(sum(float(value) * float(value) for value in tuple(frame.velocity)[:2]))
        running = bool(frame.moving and speed > 3.25)
        action_state = self._map_studio_pie_player_action_state
        current = str(getattr(getattr(engine, "current_animation", None), "name", "") or "").lower()
        if action_state.get("pie_action_active"):
            still_playing = engine.advance(max(0.0, min(float(delta_time), 0.25)))
            if not still_playing and not action_state.get("pie_action_hold"):
                self._restore_map_studio_pie_actor_action_animation("pie:player", force=True)
            current = str(getattr(getattr(engine, "current_animation", None), "name", "") or "").lower()
        else:
            wanted = "run" if running else ("walk" if frame.moving else "pause1")
            if wanted != current:
                played = engine.play(
                    wanted,
                    loop=True,
                    blend=True,
                    sync_phase=bool(current in {"walk", "run"} and wanted in {"walk", "run"}),
                )
                if not played and wanted == "pause1":
                    engine.play("idlepose", loop=True, blend=True)
                current = str(getattr(getattr(engine, "current_animation", None), "name", "") or "").lower()
            engine.advance(max(0.0, min(float(delta_time), 0.25)))
        pose = _stamp_map_studio_pie_actor_pose(engine.evaluate(), actor, current)
        actor.set_transform(tuple(frame.position), float(frame.facing_radians))
        self._map_studio_pie_animation_name = current
        self._map_studio_pie_animation_run = running
        return (
            actor.root_node,
            actor.actor_id,
            pose,
            current,
            float(getattr(engine, "current_time", 0.0) or 0.0),
            float(getattr(getattr(engine, "current_animation", None), "length", 0.0) or 0.0),
        )

    def _create_map_studio_pie_creature_actors(self, preview_model: Any, game: str) -> str:
        """Prepare retained idle actors asynchronously, then publish atomically."""

        self._cancel_map_studio_pie_creature_preparation(reset_summary=False)
        self._map_studio_pie_creature_entries = []
        self._map_studio_pie_hidden_creature_groups = []
        self._map_studio_pie_creature_summary = "creatures 0"
        self._map_studio_pie_creature_animation_budget = 0.0
        self._map_studio_pie_creature_animation_cursor = 0
        manager = getattr(self, "resource_manager", None)
        root = getattr(preview_model, "root_node", None)
        if manager is None or root is None:
            return "Creature animation preview is unavailable because the resource library or resident map model is missing."
        try:
            from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
            from src.core.modules.map_studio_pie import prepare_map_studio_pie_actor_hierarchy
            from src.core.modules.map_studio_pie_creatures import (
                build_map_studio_pie_creature_plan,
                prepare_map_studio_pie_creature_actor_artifacts,
            )
            from src.core.modules.map_studio_stock_content_preview import (
                RES_UTC,
                TemplateModelResolver,
                load_kotor_model_from_bytes,
            )
            from src.systems.bas.preview_composer import build_bas_preview_model

            controller = self.controller
            placements = controller.map_studio_authored_placements_snapshot()
            if placements is None:
                return "Creature animation preview is unavailable because this KMAP has no authored module snapshot."
            template_resources = tuple(getattr(controller, "_authored_creature_resources", ()) or ())
            resolver = getattr(controller, "_map_studio_stock_template_resolver", None)
            if (
                resolver is None
                or getattr(resolver, "_game", "") != game
                or getattr(resolver, "_manager", None) is not manager
            ):
                resolver = TemplateModelResolver(
                    manager,
                    game,
                    template_resources=template_resources,
                )
            try:
                scene_animations = controller.map_studio_scene_animation_map(manager)
            except Exception:
                scene_animations = {}
            scene_source = str(getattr(scene_animations, "source", "") or "")
            if scene_source:
                scene_hash = str(getattr(scene_animations, "source_sha256", "") or "")
                scene_script = str(getattr(scene_animations, "script_resref", "") or "")
                intent_count = len(dict(getattr(scene_animations, "intents", {}) or {}))
                self._log(
                    f"Simulation scene animations: {scene_script or '(OnEnter)'}.ncs from {scene_source}; "
                    f"SHA-256 {scene_hash[:12] or '(unavailable)'}, {intent_count} literal intent(s)."
                )
            plan = build_map_studio_pie_creature_plan(
                placements,
                resolver,
                game=game,
                utc_reader=lambda resref, _game: resolver._template_bytes(resref, RES_UTC),
                template_resources=template_resources,
                scene_animations=scene_animations,
            )
            for warning in tuple(plan.warnings):
                self._log(f"Simulation creatures: {warning}")
            if not tuple(plan.specs):
                return ""

            generation = self._map_studio_pie_creature_prepare_generation
            source_cache_owner = getattr(controller, "_map_studio_stock_model_cache", None)
            cache_context = (
                id(manager),
                int(getattr(manager, "revision", 0) or 0),
                str(game or "K1").upper(),
                id(source_cache_owner),
                int(getattr(controller, "_authored_placeable_preview_revision", 0) or 0),
            )
            if self._map_studio_pie_creature_model_cache_context != cache_context:
                self._map_studio_pie_creature_model_cache.clear()
                self._map_studio_pie_creature_model_cache_context = cache_context
            source_cache = dict(source_cache_owner or {})
            cancel_event = Event()
            self._map_studio_pie_creature_prepare_cancel = cancel_event
            future = _MAP_STUDIO_PIE_ACTOR_EXECUTOR.submit(
                prepare_map_studio_pie_creature_actor_artifacts,
                plan,
                manager,
                resolver,
                game,
                source_cache,
                dict(self._map_studio_pie_creature_model_cache),
                model_bytes_loader=load_kotor_model_from_bytes,
                model_composer=build_bas_preview_model,
                animation_engine_factory=AnimationEngine,
                hierarchy_preparer=prepare_map_studio_pie_actor_hierarchy,
                supermodel_configurer=SuperModelResolver.configure,
                cancel_requested=cancel_event.is_set,
            )
            self._map_studio_pie_creature_prepare_future = future
            self._map_studio_pie_creature_prepare_preview_id = id(preview_model)
            self._map_studio_pie_creature_summary = (
                f"creatures preparing 0/{len(plan.specs)}; "
                f"scripts {plan.suppressed_script_creature_count} deferred"
            )
            QtCore.QTimer.singleShot(
                25,
                lambda value=generation, model=preview_model: (
                    self._poll_map_studio_pie_creature_preparation(value, model)
                ),
            )
            return ""
        except Exception as exc:
            return f"Creature animation preview could not start: {exc}"

    def _cancel_map_studio_pie_creature_preparation(self, *, reset_summary: bool = True) -> None:
        """Invalidate a worker generation so Stop/reload cannot publish stale DAGs."""

        self._map_studio_pie_creature_prepare_generation = int(
            getattr(self, "_map_studio_pie_creature_prepare_generation", 0) or 0
        ) + 1
        future = getattr(self, "_map_studio_pie_creature_prepare_future", None)
        cancel_event = getattr(self, "_map_studio_pie_creature_prepare_cancel", None)
        self._map_studio_pie_creature_prepare_future = None
        self._map_studio_pie_creature_prepare_cancel = None
        self._map_studio_pie_creature_prepare_preview_id = 0
        if cancel_event is not None:
            cancel_event.set()
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        if reset_summary:
            self._map_studio_pie_creature_summary = "creatures 0"

    def _poll_map_studio_pie_creature_preparation(self, generation: int, preview_model: Any) -> None:
        """Promote only the current worker result on the Qt thread."""

        if generation != int(getattr(self, "_map_studio_pie_creature_prepare_generation", -1)):
            return
        if self._map_studio_pie_session is None:
            return
        if id(preview_model) != int(getattr(self, "_map_studio_pie_creature_prepare_preview_id", 0)):
            return
        future = getattr(self, "_map_studio_pie_creature_prepare_future", None)
        if future is None:
            return
        if not future.done():
            QtCore.QTimer.singleShot(
                25,
                lambda value=generation, model=preview_model: (
                    self._poll_map_studio_pie_creature_preparation(value, model)
                ),
            )
            return
        self._map_studio_pie_creature_prepare_future = None
        self._map_studio_pie_creature_prepare_cancel = None
        try:
            result = future.result()
        except Exception as exc:
            self._map_studio_pie_creature_summary = "creatures static; preparation failed"
            self._log(f"Simulation creature preparation failed: {exc}")
            return
        # Re-check after Future.result(): Stop/reload may have invalidated the
        # generation while the worker was finishing.
        if (
            generation != int(getattr(self, "_map_studio_pie_creature_prepare_generation", -1))
            or self._map_studio_pie_session is None
            or id(preview_model) != int(getattr(self, "_map_studio_pie_creature_prepare_preview_id", 0))
        ):
            return
        for cache_key, actor_model in tuple(result.prototype_models):
            self._map_studio_pie_creature_model_cache[cache_key] = actor_model
        while len(self._map_studio_pie_creature_model_cache) > 128:
            self._map_studio_pie_creature_model_cache.pop(next(iter(self._map_studio_pie_creature_model_cache)))
        self._promote_map_studio_pie_creature_actors(preview_model, result)

    def _promote_map_studio_pie_creature_actors(
        self,
        preview_model: Any,
        result: Any,
    ) -> None:
        """Swap flattened creatures for a complete retained batch in one publication."""

        root = getattr(preview_model, "root_node", None)
        if root is None:
            return
        from src.core.modules.map_studio_pie import (
            attach_map_studio_pie_actor,
            resolve_map_studio_pie_actor_grounding,
        )

        original_children = tuple(getattr(root, "children", ()) or ())
        creature_groups = {
            str(getattr(node, "_gr_map_studio_placement_id", "") or ""): (index, node)
            for index, node in enumerate(original_children)
            if str(getattr(node, "_gr_map_studio_placement_kind", "") or "").lower() == "creature"
        }
        pending_entries: list[dict[str, Any]] = []
        pending_wrappers: list[Any] = []
        hidden: list[tuple[int, Any]] = []
        failures = list(result.failures)
        session = self._map_studio_pie_session
        walkmesh = getattr(session, "walkmesh", None)
        config = getattr(session, "config", None)
        for prepared in tuple(result.entries):
            spec = prepared.spec
            grounding = resolve_map_studio_pie_actor_grounding(
                walkmesh,
                prepared.actor_model,
                spec.position,
                radius=0.0,
                max_step_up=float(getattr(config, "max_step_up", 0.45) or 0.45),
                max_step_down=float(getattr(config, "max_step_down", 0.75) or 0.75),
            )
            actor = attach_map_studio_pie_actor(
                preview_model,
                prepared.actor_model,
                position=grounding.surface_position,
                facing_radians=spec.facing_radians,
                actor_id=spec.render.actor_id,
                recompute_bounds=False,
                prepared_root=prepared.prepared_root,
                append_to_preview=False,
                support_plane_z=grounding.support_plane_z,
            )
            if actor is None:
                failures.append(f"{spec.tag}: prepared hierarchy could not be attached")
                continue
            pending_wrappers.append(actor.root_node)
            pending_entries.append(
                {
                    "actor": actor,
                    "engine": prepared.animation_engine,
                    "spec": spec,
                    "initial_pose": prepared.initial_pose,
                    "grounding": grounding,
                }
            )
            group = creature_groups.get(spec.placement_id)
            if group is not None and not any(group[1] is item[1] for item in hidden):
                hidden.append(group)

        hidden_nodes = {id(node) for _index, node in hidden}
        # This is the only live-scene mutation: the renderer sees either all
        # flattened actors or the complete prepared retained batch, never a
        # half-built mixture.
        root.children = [
            node for node in original_children if id(node) not in hidden_nodes
        ] + pending_wrappers
        for wrapper in pending_wrappers:
            wrapper.parent = root
        self._map_studio_pie_creature_entries = pending_entries
        self._map_studio_pie_hidden_creature_groups = sorted(hidden, key=lambda item: item[0])
        self._map_studio_pie_creature_summary = (
            f"creatures {len(pending_entries)}/{result.total_spec_count} idle; "
            f"scripts {result.suppressed_script_creature_count} deferred"
        )
        warning = self._activate_map_studio_pie_runtime_actors(
            preview_model,
            recompute_bounds=False,
            reload_model=False,
        )
        if warning:
            self._log(f"Simulation creature promotion warning: {warning}")
        self._log(
            "Simulation creatures prepared off-thread in "
            f"{result.elapsed_ms:.1f} ms (compose {result.composition_ms:.1f}, "
            f"DAG copies {result.hierarchy_copy_ms:.1f}, idle poses {result.animation_ms:.1f}; "
            f"prototype cache hits {result.prototype_cache_hits})."
        )
        if failures:
            detail = "; ".join(failures[:6])
            more = len(failures) - 6
            if more > 0:
                detail += f"; {more} more"
            self._log(
                f"Simulation creatures: {len(failures)} actor(s) kept their static authoring preview ({detail})."
            )

    def _activate_map_studio_pie_runtime_actors(
        self,
        preview_model: Any,
        *,
        recompute_bounds: bool = True,
        reload_model: bool = True,
    ) -> str:
        """Upload the map plus all transient actors once and publish initial poses."""

        viewport = getattr(self.viewport_panel, "viewport", None)
        if viewport is None or preview_model is None:
            return "The viewport could not activate retained PIE actors."
        if recompute_bounds:
            try:
                preview_model.compute_bounds()
            except Exception:
                pass
        try:
            if reload_model:
                texture_dirs = list(getattr(self.viewport_panel, "_project_texture_dirs", ()) or ())
                viewport.load_model(preview_model, extra_texture_dirs=texture_dirs)
                # ``load_model`` publishes synchronous model/selection signals.
                # Map Studio's authoring preview listeners may use those
                # signals to rehydrate the room-local source hierarchy so edit
                # mode remains source-preserving.  PIE owns a disposable copy,
                # however, and rendering that restored hierarchy turns a
                # detailed cave parcel into thousands of draw submissions.
                # Compact once more after publication and before the queued
                # renderer frame consumes the model.  The renderer caches were
                # cleared by ``load_model`` and have not rendered on this event
                # loop turn, so the first uploaded frame sees only these
                # room-local material batches.
                from src.core.modules.authored_module_preview_model import (
                    batch_authored_preview_model_for_pie_in_place,
                )

                batch_authored_preview_model_for_pie_in_place(preview_model)
                gpu_renderer = getattr(viewport, "_gpu_renderer", None)
                clear_gpu_caches = getattr(gpu_renderer, "clear_caches", None)
                if callable(clear_gpu_caches):
                    clear_gpu_caches()
                renderer = getattr(viewport, "_renderer", None)
                set_renderer_model = getattr(renderer, "set_model", None)
                if callable(set_renderer_model):
                    set_renderer_model(preview_model)
            runtime_rows: list[tuple[Any, ...]] = []
            actor = self._map_studio_pie_actor
            engine = self._map_studio_pie_animation_engine
            if actor is not None and engine is not None:
                pose = _stamp_map_studio_pie_actor_pose(
                    engine.evaluate(), actor, self._map_studio_pie_animation_name
                )
                runtime_rows.append(
                    (
                        actor.root_node,
                        actor.actor_id,
                        pose,
                        self._map_studio_pie_animation_name,
                        float(getattr(engine, "current_time", 0.0) or 0.0),
                        float(getattr(getattr(engine, "current_animation", None), "length", 0.0) or 0.0),
                    )
                )
            for entry in self._map_studio_pie_creature_entries:
                creature_actor = entry["actor"]
                creature_engine = entry["engine"]
                creature_animation_name = str(
                    getattr(getattr(creature_engine, "current_animation", None), "name", "") or ""
                )
                pose = _stamp_map_studio_pie_actor_pose(
                    entry.pop("initial_pose", None) or creature_engine.evaluate(),
                    creature_actor,
                    creature_animation_name,
                )
                runtime_rows.append(
                    (
                        creature_actor.root_node,
                        creature_actor.actor_id,
                        pose,
                        creature_animation_name,
                        float(getattr(creature_engine, "current_time", 0.0) or 0.0),
                        float(getattr(getattr(creature_engine, "current_animation", None), "length", 0.0) or 0.0),
                    )
                )
            # Door actors replace their static authoring groups before this
            # first retained-actor upload.  Publish the closed hold pose in the
            # same transaction as the player and creatures; otherwise a door
            # has no skinned runtime frame until proximity changes its state,
            # which makes it appear to pop into existence near the player.
            from src.core.modules.map_studio_pie_doors import (
                apply_map_studio_pie_door_vertical_pose_policy,
            )

            for entry in self._map_studio_pie_door_entries:
                door_actor = entry["actor"]
                door_engine = entry["engine"]
                door_animation_name = str(
                    getattr(getattr(door_engine, "current_animation", None), "name", "") or ""
                )
                pose = _stamp_map_studio_pie_actor_pose(
                    apply_map_studio_pie_door_vertical_pose_policy(
                        door_engine.evaluate(),
                        entry.get("vertical_pose_policy"),
                    ),
                    door_actor,
                    door_animation_name,
                )
                runtime_rows.append(
                    (
                        door_actor.root_node,
                        door_actor.actor_id,
                        pose,
                        door_animation_name,
                        float(getattr(door_engine, "current_time", 0.0) or 0.0),
                        float(getattr(getattr(door_engine, "current_animation", None), "length", 0.0) or 0.0),
                    )
                )
            if runtime_rows:
                viewport.set_animation_playback_active(True, "Map Studio PIE retained actors")
                viewport.update_runtime_character_frames(
                    runtime_rows,
                    camera_changed=True,
                    scene_changed=not reload_model,
                )
            elif not reload_model:
                request = getattr(viewport, "_request_render", None)
                if callable(request):
                    request(
                        fast=True,
                        reason="Map Studio PIE runtime actor structure",
                        scene=True,
                        resources=True,
                    )
            return ""
        except Exception as exc:
            return f"Retained PIE actor upload failed: {exc}"

    def _update_map_studio_pie_creature_actors(self, delta_time: float) -> tuple[tuple[Any, ...], ...]:
        """Advance staggered idle-pose cohorts, never fake NCS AI.

        Every creature keeps its own elapsed animation time, but only a
        round-robin cohort is evaluated on a given viewport tick.  The
        fractional budget targets twelve samples per actor per second when the
        viewport sustains its intended 60 Hz timer: 32 actors produce cohorts
        of six or seven.  A slow frame never expands that cohort.  Instead,
        visual NPC sampling degrades gracefully while every entry retains its
        real elapsed animation time until its round-robin turn.  This prevents
        pose work from feeding its own frame time back into a catch-up spiral.
        """

        if not self._map_studio_pie_creature_entries:
            return ()
        entries = self._map_studio_pie_creature_entries
        elapsed = max(0.0, min(float(delta_time), 0.25))
        if elapsed <= 0.0:
            return ()
        for entry in entries:
            entry["pie_animation_elapsed"] = min(
                0.25,
                float(entry.get("pie_animation_elapsed", 0.0) or 0.0) + elapsed,
            )

        actor_count = len(entries)
        scheduler_step = min(elapsed, 1.0 / _MAP_STUDIO_PIE_CREATURE_SCHEDULER_HZ)
        budget = max(
            0.0,
            float(getattr(self, "_map_studio_pie_creature_animation_budget", 0.0) or 0.0),
        ) + actor_count * scheduler_step * _MAP_STUDIO_PIE_CREATURE_POSE_HZ
        requested_count = int(budget)
        cohort_cap = max(
            1,
            math.ceil(
                actor_count
                * _MAP_STUDIO_PIE_CREATURE_POSE_HZ
                / _MAP_STUDIO_PIE_CREATURE_SCHEDULER_HZ
            ),
        )
        update_count = min(actor_count, cohort_cap, requested_count)
        # Keep only the fractional token.  Whole overdue tokens represent UI
        # time already lost and replaying them would immediately recreate the
        # feedback loop this scheduler exists to avoid.  Per-entry elapsed
        # values above remain untouched, so phase time is not tied to quota.
        self._map_studio_pie_creature_animation_budget = budget - math.floor(budget)
        if update_count <= 0:
            return ()

        cursor = int(getattr(self, "_map_studio_pie_creature_animation_cursor", 0) or 0) % actor_count
        rows: list[tuple[Any, ...]] = []
        for offset in range(update_count):
            entry = entries[(cursor + offset) % actor_count]
            actor = entry["actor"]
            engine = entry["engine"]
            step = min(0.25, float(entry.get("pie_animation_elapsed", elapsed) or elapsed))
            entry["pie_animation_elapsed"] = 0.0
            still_playing = engine.advance(step)
            if (
                entry.get("pie_action_active")
                and not still_playing
                and not entry.get("pie_action_hold")
            ):
                entity_id = str(getattr(entry.get("spec"), "placement_id", "") or "")
                self._restore_map_studio_pie_actor_action_animation(entity_id, force=True)
            animation_name = str(
                getattr(getattr(engine, "current_animation", None), "name", "") or ""
            )
            pose = _stamp_map_studio_pie_actor_pose(engine.evaluate(), actor, animation_name)
            rows.append(
                (
                    actor.root_node,
                    actor.actor_id,
                    pose,
                    animation_name,
                    float(getattr(engine, "current_time", 0.0) or 0.0),
                    float(getattr(getattr(engine, "current_animation", None), "length", 0.0) or 0.0),
                )
            )
        self._map_studio_pie_creature_animation_cursor = (cursor + update_count) % actor_count
        return tuple(rows)

    def map_studio_pie_door_diagnostics(self) -> dict[str, Any]:
        """Report the loaded module's PIE door plan for artifact/culling diagnosis.

        Editor-side diagnostic: each authored door's resolved genericdoors model,
        world position (Z reveals a 'floating' door), whether the model loaded,
        and whether an animated actor was built and matched a static group.
        """

        manager = getattr(self, "resource_manager", None)
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        result: dict[str, Any] = {"doors": [], "built_actor_count": len(getattr(self, "_map_studio_pie_door_entries", []) or [])}
        try:
            from src.core.modules.map_studio_pie_doors import build_map_studio_pie_door_plan
            from src.core.modules.map_studio_stock_content_preview import load_stock_kotor_model

            placements = self.controller.map_studio_authored_placements_snapshot()
            resolver = getattr(self.controller, "_map_studio_stock_template_resolver", None)
            if placements is None or resolver is None:
                return {**result, "reason": "no authored placements/resolver"}
            built_ids = {
                str(getattr(entry.get("spec"), "door_id", "")) for entry in (getattr(self, "_map_studio_pie_door_entries", []) or [])
            }
            for spec in build_map_studio_pie_door_plan(placements, resolver):
                model_loads = False
                if spec.model_resref and manager is not None:
                    try:
                        model_loads = load_stock_kotor_model(manager, spec.model_resref, game) is not None
                    except Exception:
                        model_loads = False
                result["doors"].append({
                    "door_id": spec.door_id,
                    "tag": spec.tag,
                    "model_resref": spec.model_resref,
                    "position": [round(float(v), 3) for v in spec.position],
                    "z": round(float(spec.position[2]), 3),
                    "can_build_actor": bool(spec.can_build_actor),
                    "model_loads": bool(model_loads),
                    "actor_built": spec.door_id in built_ids,
                })
        except Exception as exc:
            result["reason"] = f"door diagnostics failed: {exc}"
        result["door_count"] = len(result["doors"])
        result["unresolved_model_count"] = sum(1 for d in result["doors"] if not d["model_resref"])
        result["model_load_fail_count"] = sum(1 for d in result["doors"] if d["model_resref"] and not d["model_loads"])
        return result

    def _create_map_studio_pie_door_actors(self, preview_model: Any, game: str) -> str:
        """Swap each static door for an animated door actor that opens in PIE."""

        self._map_studio_pie_door_entries = []
        self._map_studio_pie_hidden_door_groups = []
        manager = getattr(self, "resource_manager", None)
        root = getattr(preview_model, "root_node", None)
        session = self._map_studio_pie_session
        if manager is None or root is None or session is None:
            return ""
        original_children = list(tuple(getattr(root, "children", ()) or ()))
        try:
            from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver
            from src.core.modules.map_studio_pie import attach_map_studio_pie_actor
            from src.core.modules.map_studio_pie_doors import (
                build_map_studio_pie_door_vertical_pose_policy,
                build_map_studio_pie_door_plan,
                door_state_clip_candidates,
                map_studio_pie_door_visual_nodes,
                play_map_studio_pie_door_clip,
                set_map_studio_pie_door_visuals_hidden,
            )
            from src.core.modules.map_studio_stock_content_preview import load_stock_kotor_model

            placements = self.controller.map_studio_authored_placements_snapshot()
            resolver = getattr(self.controller, "_map_studio_stock_template_resolver", None)
            if placements is None or resolver is None:
                return ""
            specs = tuple(spec for spec in build_map_studio_pie_door_plan(placements, resolver) if spec.can_build_actor)
            if not specs:
                return ""
            SuperModelResolver.configure(manager)
            # Build every actor off-tree first. Publication is one transaction:
            # either all successful wrappers replace their matching static
            # groups together, or the resident scene stays byte-for-byte intact.
            door_groups = [
                node
                for node in tuple(getattr(root, "children", ()) or ())
                if str(getattr(node, "_gr_map_studio_placement_kind", "") or "").lower() == "door"
            ]
            model_cache: dict[str, Any] = {}
            claimed_static_groups: set[int] = set()
            entries: list[dict[str, Any]] = []
            failures: list[str] = []
            for spec in specs:
                try:
                    source = model_cache.get(spec.model_resref)
                    if source is None:
                        source = load_stock_kotor_model(manager, spec.model_resref, game)
                        model_cache[spec.model_resref] = source
                    if source is None:
                        failures.append(f"{spec.tag or spec.door_id}: model {spec.model_resref} was unavailable")
                        continue
                    import copy as _copy

                    actor_model = _copy.deepcopy(source)
                    actor = attach_map_studio_pie_actor(
                        preview_model,
                        actor_model,
                        position=spec.position,
                        facing_radians=spec.bearing,
                        actor_id=f"__map_studio_pie_door__:{spec.door_id}",
                        recompute_bounds=False,
                        append_to_preview=False,
                        model_yaw_offset_radians=0.0,
                    )
                    if actor is None:
                        failures.append(f"{spec.tag or spec.door_id}: actor hierarchy could not be retained")
                        continue
                    # Door panels are rigid animated meshes. Their stock MDL
                    # headers use deliberately loose (-5..10 m) bounds, which
                    # are appropriate for collision but keep several occluded
                    # doors in the render queue. Let the renderer use each
                    # panel's exact uploaded bounds after the whole-actor test.
                    setattr(actor.root_node, "_gr_map_studio_pie_rigid_actor", True)
                    # Aurora door MDLs commonly include an animated ``trans``
                    # collision/transition volume with NULL material.  It is
                    # runtime data, not visible door geometry; drawing it was
                    # the dark plane users saw clipping through an open portal.
                    actor_stack = [actor.root_node]
                    actor_visited: set[int] = set()
                    while actor_stack:
                        actor_node = actor_stack.pop()
                        if id(actor_node) in actor_visited:
                            continue
                        actor_visited.add(id(actor_node))
                        actor_stack.extend(tuple(getattr(actor_node, "children", ()) or ()))
                        if (
                            str(getattr(actor_node, "name", "") or "").strip().lower() == "trans"
                            and str(getattr(actor_node, "texture", "") or "").strip().lower()
                            in {"", "null", "none"}
                        ):
                            setattr(actor_node, "_gr_hidden", True)
                            setattr(actor_node, "_gr_map_studio_pie_transition_helper", True)
                    engine = AnimationEngine(actor_model)
                    vertical_pose_policy = build_map_studio_pie_door_vertical_pose_policy(
                        actor_model,
                        None,
                    )
                    if play_map_studio_pie_door_clip(
                        engine,
                        door_state_clip_candidates(is_open=True, transitioning=True),
                        loop=False,
                    ):
                        opening_animation = getattr(engine, "current_animation", None)
                        open_pose = engine.evaluate(
                            float(getattr(opening_animation, "length", 0.0) or 0.0)
                        )
                        vertical_pose_policy = build_map_studio_pie_door_vertical_pose_policy(
                            actor_model,
                            open_pose,
                        )
                    play_map_studio_pie_door_clip(
                        engine, door_state_clip_candidates(is_open=False, transitioning=False), loop=True
                    )
                    exact = next(
                        (
                            node
                            for node in door_groups
                            if id(node) not in claimed_static_groups
                            and str(getattr(node, "_gr_map_studio_placement_id", "") or "") == spec.door_id
                        ),
                        None,
                    )
                    nearest = exact or min(
                        (node for node in door_groups if id(node) not in claimed_static_groups),
                        key=lambda node: sum(
                            (float(getattr(node, "position", (0, 0, 0))[i]) - spec.position[i]) ** 2 for i in range(3)
                        ),
                        default=None,
                    )
                    actor_visual_nodes = map_studio_pie_door_visual_nodes(actor.root_node)
                    proxy_visual_nodes = map_studio_pie_door_visual_nodes(nearest)
                    animated_visual_active = nearest is None
                    if nearest is not None:
                        claimed_static_groups.add(id(nearest))
                        # Keep the already-resident flattened door as the
                        # closed-state proxy. It renders immediately on PIE
                        # load and is compacted to one draw by the final
                        # door-only static batching pass. The full source MDL
                        # becomes visible only while it actually animates.
                        set_map_studio_pie_door_visuals_hidden(actor_visual_nodes, True)
                        set_map_studio_pie_door_visuals_hidden(proxy_visual_nodes, False)
                    entries.append(
                        {
                            "actor": actor,
                            "engine": engine,
                            "spec": spec,
                            "is_open": False,
                            "transitioning": False,
                            "vertical_pose_policy": vertical_pose_policy,
                            "proxy_group": nearest,
                            "actor_visual_nodes": actor_visual_nodes,
                            "proxy_visual_nodes": proxy_visual_nodes,
                            "animated_visual_active": animated_visual_active,
                        }
                    )
                except Exception as exc:
                    failures.append(f"{spec.tag or spec.door_id}: {exc}")

            wrappers = [entry["actor"].root_node for entry in entries]
            root.children = list(original_children) + wrappers
            for node in tuple(root.children or ()):
                node.parent = root
            self._map_studio_pie_door_entries = entries
            self._map_studio_pie_hidden_door_groups = []
            if failures:
                return "Animated door preview skipped " + "; ".join(failures[:4])
            return ""
        except Exception as exc:
            root.children = original_children
            for node in original_children:
                node.parent = root
            self._map_studio_pie_door_entries = []
            self._map_studio_pie_hidden_door_groups = []
            return f"Animated door setup failed: {exc}"

    def _set_map_studio_pie_door_animated_visual(
        self,
        entry: dict[str, Any],
        animated: bool,
    ) -> None:
        """Atomically swap one closed proxy with its animated source model."""

        wanted = bool(animated)
        if bool(entry.get("animated_visual_active", False)) == wanted:
            return
        from src.core.modules.map_studio_pie_doors import (
            set_map_studio_pie_door_visuals_hidden,
        )

        changed = set_map_studio_pie_door_visuals_hidden(
            entry.get("actor_visual_nodes"),
            not wanted,
        )
        changed = (
            set_map_studio_pie_door_visuals_hidden(
                entry.get("proxy_visual_nodes"),
                wanted,
            )
            or changed
        )
        entry["animated_visual_active"] = wanted
        if changed:
            preview_model = self._map_studio_pie_runtime_preview_model
            if preview_model is not None:
                setattr(
                    preview_model,
                    "_gr_classification_revision",
                    int(getattr(preview_model, "_gr_classification_revision", 0) or 0) + 1,
                )

    def _update_map_studio_pie_door_actors(self, frame: Any, delta_time: float) -> tuple[tuple[Any, ...], ...]:
        """Play each door's opening/opened/closing/closed clips from its state."""

        entries = self._map_studio_pie_door_entries
        if not entries:
            return ()
        from src.core.modules.map_studio_pie_doors import (
            advance_map_studio_pie_door_animation,
            apply_map_studio_pie_door_vertical_pose_policy,
        )

        states = {str(getattr(state, "entity_id", "")): bool(getattr(state, "is_open", False)) for state in tuple(getattr(frame, "door_states", ()) or ())}
        rows: list[tuple[Any, ...]] = []
        for entry in entries:
            spec = entry["spec"]
            engine = entry["engine"]
            actor = entry["actor"]
            current_open = bool(entry["is_open"])
            transitioning = bool(entry.get("transitioning", False))
            wanted_open = states.get(spec.door_id, current_open)
            # The initial closed/open hold pose was published with the actor
            # batch. Stable doors do not need their identical pose evaluated,
            # cache-invalidated, and resubmitted on every 16 ms PIE tick.
            # Continue publishing while a one-shot transition is active and
            # once more when it lands on the final held pose.
            if bool(wanted_open) == current_open and not transitioning:
                continue
            # Swap before evaluating the first transition frame so the proxy
            # and source panel are never drawn together.
            self._set_map_studio_pie_door_animated_visual(entry, True)
            animation_step = advance_map_studio_pie_door_animation(
                engine,
                wanted_open=wanted_open,
                current_open=current_open,
                transitioning=transitioning,
                delta_time=delta_time,
            )
            entry["is_open"] = animation_step.is_open
            entry["transitioning"] = animation_step.transitioning
            pose = apply_map_studio_pie_door_vertical_pose_policy(
                engine.evaluate(),
                entry.get("vertical_pose_policy"),
            )
            setattr(pose, "_gr_animation_scene_object_id", actor.actor_id)
            setattr(pose, "_gr_animation_source_model_id", id(actor.source_model))
            rows.append(
                (
                    actor.root_node,
                    actor.actor_id,
                    pose,
                    str(getattr(getattr(engine, "current_animation", None), "name", "") or ""),
                    float(getattr(engine, "current_time", 0.0) or 0.0),
                    float(getattr(getattr(engine, "current_animation", None), "length", 0.0) or 0.0),
                )
            )
            if not animation_step.is_open and not animation_step.transitioning:
                self._set_map_studio_pie_door_animated_visual(entry, False)
        return tuple(rows)

    def _remove_map_studio_pie_runtime_actors(self) -> None:
        """Detach all runtime DAGs, restore flattened authoring actors, reload once."""

        self._cancel_map_studio_pie_creature_preparation()
        player = self._map_studio_pie_actor
        creature_entries = tuple(self._map_studio_pie_creature_entries)
        door_entries = tuple(self._map_studio_pie_door_entries)
        party_entries = tuple(self._map_studio_pie_party_actors)
        actors = (
            ([player] if player is not None else [])
            + [entry["actor"] for entry in creature_entries]
            + [entry["actor"] for entry in party_entries if entry.get("actor") is not None]
            + [entry["actor"] for entry in door_entries]
        )
        preview_model = next((actor.preview_model for actor in actors if actor is not None), None)
        if preview_model is None:
            preview_model = self._map_studio_pie_runtime_preview_model
        viewport = getattr(self.viewport_panel, "viewport", None)
        if viewport is not None:
            try:
                viewport.set_animation_playback_active(False, "Map Studio PIE stopped")
            except Exception:
                pass
        for actor in actors:
            try:
                actor.detach(recompute_bounds=False)
            except Exception:
                pass
        root = getattr(preview_model, "root_node", None)
        if root is not None:
            # Door indices were captured after creature statics were hidden, so
            # unwind in reverse publication order: doors first, creatures next.
            for original_index, group in self._map_studio_pie_hidden_door_groups:
                if any(group is child for child in tuple(root.children or ())):
                    continue
                group.parent = root
                root.children.insert(min(max(0, int(original_index)), len(root.children)), group)
            for original_index, group in self._map_studio_pie_hidden_creature_groups:
                if any(group is child for child in tuple(root.children or ())):
                    continue
                group.parent = root
                root.children.insert(min(max(0, int(original_index)), len(root.children)), group)
            # The entry-point group was removed before creature/door statics,
            # so restore it last to recover the original authored child order.
            self._restore_map_studio_pie_player_start_preview(preview_model)
        reload_model = self._map_studio_pie_authoring_preview_model or preview_model
        self.viewport_panel._room_preview_model = reload_model
        if reload_model is not None:
            try:
                reload_model.compute_bounds()
            except Exception:
                pass
            if viewport is not None:
                try:
                    texture_dirs = list(getattr(self.viewport_panel, "_project_texture_dirs", ()) or ())
                    viewport.load_model(reload_model, extra_texture_dirs=texture_dirs)
                except Exception:
                    pass
        self._map_studio_pie_actor = None
        self._map_studio_pie_hidden_player_start_groups = []
        self._map_studio_pie_party_actors = []
        self._map_studio_pie_animation_engine = None
        self._map_studio_pie_animation_name = ""
        self._map_studio_pie_animation_run = False
        self._map_studio_pie_creature_entries = []
        self._map_studio_pie_hidden_creature_groups = []
        self._map_studio_pie_door_entries = []
        self._map_studio_pie_hidden_door_groups = []
        self._map_studio_pie_authoring_preview_model = None
        self._map_studio_pie_runtime_preview_model = None
        self._map_studio_pie_creature_summary = "creatures 0"
        self._map_studio_pie_creature_animation_budget = 0.0
        self._map_studio_pie_creature_animation_cursor = 0

    def _handle_map_studio_pie_camera_input(self, payload: object) -> None:
        """Apply DEFAULT camerastyle free-look limits."""

        session = self._map_studio_pie_session
        camera = getattr(getattr(self.viewport_panel, "viewport", None), "camera", None)
        if session is None or camera is None:
            return
        values = dict(payload or {}) if isinstance(payload, dict) else {}
        dx = float(values.get("orbit_x", 0.0) or 0.0)
        dy = float(values.get("orbit_y", 0.0) or 0.0)
        if dx or dy:
            camera.azimuth = (float(camera.azimuth) - (dx * 0.25)) % 360.0
            # Retail DEFAULT free-look rows allow roughly 15 degrees up and
            # 20 degrees down around the 83-degree camera pitch.
            camera.elevation = max(-13.0, min(22.0, float(camera.elevation) - (dy * 0.20)))
            set_azimuth = getattr(session, "set_camera_azimuth", None)
            if callable(set_azimuth):
                set_azimuth(float(camera.azimuth))

    def _start_map_studio_pie_ambient_audio(self, session: Any, game: str) -> str:
        """Start the bounded UTS ambient preview without claiming engine parity."""

        self._stop_map_studio_pie_ambient_audio()
        manager = getattr(self, "resource_manager", None)
        placements = self.controller.map_studio_authored_placements_snapshot()
        if manager is None or placements is None:
            self._map_studio_pie_audio_summary = "audio unavailable"
            return "Ambient UTS preview is unavailable because the resource library or authored placements are missing."
        audio = None
        try:
            from src.adapters.qt_audio.map_studio_pie_audio import MapStudioPIEAmbientAudio
            from src.core.modules.map_studio_pie_audio import build_map_studio_pie_ambient_sound_plan

            # WAV existence is checked lazily by the playback adapter.  Doing
            # a second eager pass over every clip here causes a noticeable PIE
            # startup stall on stock modules while providing no extra truth.
            plan = build_map_studio_pie_ambient_sound_plan(
                placements,
                manager,
                game,
                check_clip_resources=False,
            )
            audio = MapStudioPIEAmbientAudio(
                manager,
                game,
                self,
                seed=0,
                max_voices=32,
            )
            audio.warningRaised.connect(
                lambda message: self._log(f"Simulation ambient audio: {message}")
            )
            audio.start(plan, listener_position=session.player_eye_target())
            self._map_studio_pie_audio_runtime = audio
            self._map_studio_pie_audio_update_bucket = -1
            active_count = len(plan.active_specs)
            self._map_studio_pie_audio_summary = f"audio {min(active_count, 32)}/{len(plan.specs)} UTS"
            for warning in tuple(plan.warnings)[:12]:
                self._log(f"Simulation ambient audio: {warning.message}")
            remaining = max(0, len(plan.warnings) - 12)
            if remaining:
                self._log(f"Simulation ambient audio: {remaining} additional plan warning(s) omitted from the live log.")
            self._log(f"Simulation ambient audio: {audio.approximation_note}")
            return ""
        except Exception as exc:
            if audio is not None:
                try:
                    audio.close()
                    audio.deleteLater()
                except Exception:
                    pass
            self._map_studio_pie_audio_runtime = None
            self._map_studio_pie_audio_summary = "audio unavailable"
            return f"Ambient UTS preview could not start: {exc}"

    def _start_map_studio_pie_dialogue_audio(self, game: str) -> str:
        """Create the bounded authored DLG voice/sound playback adapter."""

        self._stop_map_studio_pie_dialogue_audio()
        manager = getattr(self, "resource_manager", None)
        if manager is None:
            return "Dialogue audio is unavailable because the resource library is missing."
        try:
            from src.adapters.qt_audio.map_studio_pie_audio import MapStudioPIEDialogueAudio

            audio = MapStudioPIEDialogueAudio(manager, game, self)
            audio.warningRaised.connect(
                lambda message: self._log(f"Simulation dialogue audio: {message}")
            )
            self._map_studio_pie_dialogue_audio_runtime = audio
            self._map_studio_pie_dialogue_audio_signature = ()
            return ""
        except Exception as exc:
            self._map_studio_pie_dialogue_audio_runtime = None
            return f"Dialogue audio preview could not start: {exc}"

    def _sync_map_studio_pie_dialogue_audio(self, gameplay: Any) -> None:
        audio = self._map_studio_pie_dialogue_audio_runtime
        if audio is None:
            return
        dialogue = getattr(gameplay, "dialogue", None)
        listening = bool(
            dialogue is not None
            and not bool(getattr(dialogue, "ended", False))
            and str(getattr(dialogue, "state", "") or "").strip().lower() == "listening"
        )
        signature = (
            str(getattr(dialogue, "current_node_id", "") or ""),
            str(getattr(dialogue, "voice_resref", "") or "").strip().lower(),
            str(getattr(dialogue, "sound_resref", "") or "").strip().lower(),
        ) if listening else ()
        if signature == self._map_studio_pie_dialogue_audio_signature:
            return
        self._map_studio_pie_dialogue_audio_signature = signature
        if not signature:
            audio.stop()
            self._reset_map_studio_pie_dialogue_line_timer()
            return
        audio.play_line(signature[1], signature[2])
        self._begin_map_studio_pie_dialogue_line_timer(dialogue, signature)

    def _reset_map_studio_pie_dialogue_line_timer(self) -> None:
        self._map_studio_pie_dialogue_line_signature = ()
        self._map_studio_pie_dialogue_line_elapsed = 0.0
        self._map_studio_pie_dialogue_line_interval = 0.0

    def _begin_map_studio_pie_dialogue_line_timer(
        self,
        dialogue: Any,
        signature: tuple[str, ...],
    ) -> None:
        from src.core.modules.map_studio_pie_dialogue import map_studio_pie_dialogue_line_interval

        audio = self._map_studio_pie_dialogue_audio_runtime
        duration = getattr(audio, "current_duration_seconds", None) if audio is not None else None
        self._map_studio_pie_dialogue_line_signature = tuple(signature)
        self._map_studio_pie_dialogue_line_elapsed = 0.0
        self._map_studio_pie_dialogue_line_interval = map_studio_pie_dialogue_line_interval(
            str(getattr(dialogue, "text", "") or ""),
            delay_seconds=int(getattr(dialogue, "delay", -1) or -1),
            audio_duration_seconds=duration,
        )

    def _advance_map_studio_pie_dialogue_line_timer(self, gameplay: Any, delta_time: float) -> None:
        """Auto-advance only after the node's scheduled editor interval.

        Audio completion never advances the graph directly.  This timer owns
        progression, extends to WAV/Qt duration when available, and freezes
        with the RTwP pause.  Exact retail WaitFlags bits and LIP timing remain
        unknown and are deliberately not fabricated here.
        """

        dialogue = getattr(gameplay, "dialogue", None)
        listening = bool(
            dialogue is not None
            and not bool(getattr(dialogue, "ended", False))
            and str(getattr(dialogue, "state", "") or "").strip().lower() == "listening"
        )
        signature = (
            str(getattr(dialogue, "current_node_id", "") or ""),
            str(getattr(dialogue, "voice_resref", "") or "").strip().lower(),
            str(getattr(dialogue, "sound_resref", "") or "").strip().lower(),
        ) if listening else ()
        if not signature:
            self._reset_map_studio_pie_dialogue_line_timer()
            return
        if signature != self._map_studio_pie_dialogue_line_signature:
            self._begin_map_studio_pie_dialogue_line_timer(dialogue, signature)

        combat = getattr(gameplay, "combat", None)
        if combat is not None and bool(getattr(combat, "paused", False)):
            return

        from src.core.modules.map_studio_pie_dialogue import map_studio_pie_dialogue_line_interval

        audio = self._map_studio_pie_dialogue_audio_runtime
        duration = getattr(audio, "current_duration_seconds", None) if audio is not None else None
        self._map_studio_pie_dialogue_line_interval = map_studio_pie_dialogue_line_interval(
            str(getattr(dialogue, "text", "") or ""),
            delay_seconds=int(getattr(dialogue, "delay", -1) or -1),
            audio_duration_seconds=duration,
        )
        self._map_studio_pie_dialogue_line_elapsed += max(0.0, min(float(delta_time), 0.25))
        if self._map_studio_pie_dialogue_line_elapsed + 1.0e-9 < self._map_studio_pie_dialogue_line_interval:
            return
        # If Qt has not reported duration yet, do not cut off a still-playing
        # authored line merely because the text fallback expired.
        if audio is not None and bool(getattr(audio, "active", False)) and duration is None:
            return
        session = self._map_studio_pie_session
        self._reset_map_studio_pie_dialogue_line_timer()
        if session is not None:
            session.continue_gameplay_dialogue()
            self._publish_map_studio_pie_gameplay_state()

    def _stop_map_studio_pie_dialogue_audio(self) -> None:
        audio = self._map_studio_pie_dialogue_audio_runtime
        self._map_studio_pie_dialogue_audio_runtime = None
        self._map_studio_pie_dialogue_audio_signature = ()
        self._reset_map_studio_pie_dialogue_line_timer()
        if audio is None:
            return
        try:
            audio.close()
        except Exception as exc:
            self._log(f"Simulation dialogue audio cleanup warning: {exc}")
        try:
            audio.deleteLater()
        except RuntimeError:
            pass

    def _update_map_studio_pie_ambient_audio(self, session: Any, frame: Any) -> None:
        """Move the editor listener at 5 Hz so audio cannot consume the render budget."""

        audio = self._map_studio_pie_audio_runtime
        if audio is None:
            return
        bucket = int(float(frame.simulation_time) * 5.0)
        if bucket == self._map_studio_pie_audio_update_bucket:
            return
        self._map_studio_pie_audio_update_bucket = bucket
        try:
            audio.set_listener_position(session.player_eye_target())
        except Exception as exc:
            self._log(f"Simulation ambient audio listener update stopped: {exc}")
            self._stop_map_studio_pie_ambient_audio()
            self._map_studio_pie_audio_summary = "audio stopped"

    def _stop_map_studio_pie_ambient_audio(self) -> None:
        """Release every transient QtMultimedia voice owned by the PIE run."""

        audio = self._map_studio_pie_audio_runtime
        self._map_studio_pie_audio_runtime = None
        self._map_studio_pie_audio_update_bucket = -1
        if audio is None:
            return
        try:
            audio.close()
        except Exception as exc:
            self._log(f"Simulation ambient audio cleanup warning: {exc}")
        try:
            audio.deleteLater()
        except RuntimeError:
            pass

    def _start_map_studio_pie(self, *, focus_viewport: bool = True) -> None:
        """Build a runtime-only WOK/camera preflight from the current KMAP.

        Manual Play keeps its keyboard-focus handoff. Focus-safe automation can
        suppress that final handoff so a non-activating Map Studio window does
        not promote itself to the foreground on Windows.
        """

        self._cancel_map_studio_component_preview()
        self._cancel_map_studio_texture_paint_stroke()
        preview_model = getattr(self.viewport_panel, "_room_preview_model", None)
        if preview_model is None:
            preview_model = getattr(getattr(self.viewport_panel, "viewport", None), "model", None)
        try:
            build = self.controller.create_map_studio_pie_session(
                preview_model=preview_model,
                resource_manager=getattr(self, "resource_manager", None),
            )
        except Exception as exc:
            self._set_map_studio_pie_command_active(False)
            QtWidgets.QMessageBox.warning(self, "Simulation Blocked", f"Map Studio could not start simulation:\n{exc}")
            return
        session = getattr(build, "session", None)
        validation = getattr(build, "validation", None)
        if session is None or not bool(getattr(validation, "ok", False)):
            issues = tuple(getattr(validation, "blocking_issues", ()) or ())
            detail = "\n".join(f"• {issue}" for issue in issues[:8]) or "The current map is not simulation-ready."
            self._set_map_studio_pie_command_active(False)
            QtWidgets.QMessageBox.warning(
                self,
                "Simulation Blocked",
                "Fix these Map Studio readiness issues before simulating:\n\n" + detail,
            )
            return
        self._map_studio_pie_authoring_preview_model = preview_model
        try:
            from src.core.modules.authored_module_preview_model import optimize_authored_preview_model_for_pie

            runtime_preview_model = optimize_authored_preview_model_for_pie(preview_model)
        except Exception as exc:
            runtime_preview_model = preview_model
            self._log(f"Simulation static batching warning: {exc}")
        self._map_studio_pie_runtime_preview_model = runtime_preview_model
        preview_model = runtime_preview_model
        batch_summary = dict(
            getattr(preview_model, "_gr_map_studio_pie_static_batch_summary", {}) or {}
        )
        if batch_summary:
            self._log(
                "Simulation static architecture batching: "
                f"{batch_summary.get('source_meshes', 0)} authored meshes -> "
                f"{batch_summary.get('runtime_batches', 0)} material batches "
                f"({batch_summary.get('draw_calls_saved', 0)} draw submissions removed)."
            )
        camera = getattr(getattr(self.viewport_panel, "viewport", None), "camera", None)
        if camera is None:
            self._set_map_studio_pie_command_active(False)
            QtWidgets.QMessageBox.warning(self, "Simulation Blocked", "The Map Studio viewport camera is not available.")
            return
        self._map_studio_pie_camera_snapshot = (
            float(camera.azimuth),
            float(camera.elevation),
            float(camera.distance),
            tuple(float(value) for value in tuple(camera.target)[:3]),
            float(getattr(camera, "fov", 45.0)),
            float(getattr(camera, "_near", 0.01)),
            float(getattr(camera, "_far", 1000.0)),
        )
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        hud_skin_warning = ""
        configure_game_hud = getattr(self.viewport_panel, "configure_map_studio_pie_game_hud", None)
        if callable(configure_game_hud):
            _body_resref, head_resref = self._map_studio_pie_player_settings()
            player_portrait_resref = f"po_{head_resref}" if head_resref else "po_pmhc01"
            hud_skin_warning = str(
                configure_game_hud(
                    getattr(self, "resource_manager", None),
                    game,
                    module_root=self._authored_module_root(),
                    player_portrait_resref=player_portrait_resref,
                )
                or ""
            )
        # K1 and K2 share the retail DEFAULT camerastyle distance/FOV.
        self._map_studio_pie_desired_camera_distance = 3.2
        camera.fov = 55.0
        self._map_studio_pie_session = session
        self._map_studio_pie_last_time = perf_counter()
        self._map_studio_pie_last_resolved_camera_distance = None
        self._map_studio_pie_status_bucket = -1
        self._map_studio_pie_active_room_resref = ""
        self._map_studio_pie_camera_turn_input = 0.0
        self._map_studio_pie_camera_turn_velocity = 0.0
        self._map_studio_pie_player_action_state = {}
        self._map_studio_pie_dialogue_animation_entities = set()
        self._map_studio_pie_dialogue_animation_policies = {}
        self._map_studio_pie_dialogue_animation_policies_loaded = False
        self._map_studio_pie_dialogue_node_id = ""
        self._map_studio_pie_dialogue_lip_limitation_reported = False
        self._map_studio_pie_dialogue_camera_animation_signature = None
        self._map_studio_pie_dialogue_camera_animation_active = False
        self._map_studio_pie_dialogue_camera_snapshot = None
        self._reset_map_studio_pie_dialogue_line_timer()
        self._map_studio_pie_gameplay_mode = "exploration"
        self._map_studio_pie_scope_text = self.map_studio_scope_label.text()
        self._set_map_studio_pie_authoring_enabled(False)
        self._set_map_studio_pie_command_active(True)
        # Keep the panel's resident-model owner aligned with the viewport while
        # PIE is active.  The panel is consulted by delayed preview/resource
        # refreshes, so leaving the authoring model here can silently replace
        # the batched runtime model after its first upload.
        self.viewport_panel._room_preview_model = runtime_preview_model
        self.viewport_panel.set_map_studio_pie_active(True)
        self._map_studio_pie_actor_warning = self._create_map_studio_pie_player_actor(session, preview_model, game)
        self._hide_map_studio_pie_player_start_preview(preview_model)
        creature_warning = self._create_map_studio_pie_creature_actors(preview_model, game)
        party_warning = self._create_map_studio_pie_party_actors(session, preview_model, game)
        if party_warning:
            self._log(f"Simulation party companions: {party_warning}")
        door_warning = self._create_map_studio_pie_door_actors(preview_model, game)
        # The viewport's authored-room wrappers may be republished while Play
        # is being assembled (for example by a queued resource refresh).
        # Compact the final disposable runtime DAG once more after every
        # transient actor swap and before its single renderer upload.  This
        # prevents the original per-panel room meshes and their material
        # batches from being submitted together.
        try:
            from src.core.modules.authored_module_preview_model import (
                batch_authored_preview_model_for_pie_in_place,
            )

            batch_authored_preview_model_for_pie_in_place(preview_model)
        except Exception as exc:
            self._log(f"Simulation final static batching warning: {exc}")
        self._sync_map_studio_pie_room_visibility(force=True)
        runtime_actor_warning = self._activate_map_studio_pie_runtime_actors(preview_model)
        audio_warning = self._start_map_studio_pie_ambient_audio(session, game)
        dialogue_audio_warning = self._start_map_studio_pie_dialogue_audio(game)
        camera.target = list(session.player_eye_target())
        camera.azimuth = (math.degrees(float(session.state.facing_radians)) + 180.0) % 360.0
        camera.elevation = 7.0
        camera.distance = self._map_studio_pie_desired_camera_distance
        session.set_camera_azimuth(float(camera.azimuth))
        session.reset_camera_collision()
        set_gameplay_state = getattr(self.viewport_panel, "set_map_studio_pie_gameplay_state", None)
        if callable(set_gameplay_state):
            set_gameplay_state(session.gameplay_snapshot())
        self.viewport_panel.set_map_studio_pie_overlay(session.overlay_geometry())
        self.map_studio_scope_label.setText(
            "Simulation — not KOTOR proof: W/S moves, Z/C strafes, A/D turns the camera, Q/E cycle targets, "
            "Enter uses the primary action, number keys choose dialogue replies, and Space pauses combat. "
            "A floor click routes the player; click a runtime actor to interact. DLG, doors, containers, and "
            "deterministic d20 combat are editor previews; arbitrary NCS and exact Odyssey AI still require export."
        )
        self._map_studio_pie_timer.start()
        if focus_viewport:
            self.viewport_panel.viewport.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self.statusBar().showMessage(
            f"Simulation started: {getattr(build, 'walkable_face_count', 0)} walkable WOK faces; "
            f"{getattr(build, 'collision_triangle_count', 0)} camera-collision triangles."
        )
        for warning in tuple(getattr(validation, "warnings", ()) or ()):
            self._log(f"Simulation warning: {warning}")
        if self._map_studio_pie_actor_warning:
            self._log(f"Simulation player warning: {self._map_studio_pie_actor_warning}")
        if creature_warning:
            self._log(f"Simulation creature warning: {creature_warning}")
        if door_warning:
            self._log(f"Simulation door warning: {door_warning}")
        if runtime_actor_warning:
            self._log(f"Simulation retained actor warning: {runtime_actor_warning}")
        if audio_warning:
            self._log(f"Simulation ambient audio warning: {audio_warning}")
        if dialogue_audio_warning:
            self._log(f"Simulation dialogue audio warning: {dialogue_audio_warning}")
        if hud_skin_warning:
            self._log(f"Simulation game-HUD skin warning: {hud_skin_warning}")

    def _stop_map_studio_pie(self) -> None:
        """Stop simulation and restore the exact authoring camera and controls."""

        if self._map_studio_pie_session is None:
            self._stop_map_studio_pie_ambient_audio()
            self._stop_map_studio_pie_dialogue_audio()
            self._set_map_studio_pie_command_active(False)
            return
        self._map_studio_pie_timer.stop()
        self._stop_map_studio_pie_ambient_audio()
        self._stop_map_studio_pie_dialogue_audio()
        self._remove_map_studio_pie_runtime_actors()
        self.viewport_panel.set_map_studio_pie_active(False)
        self.viewport_panel.set_map_studio_pie_overlay(None)
        clear_gameplay_state = getattr(self.viewport_panel, "clear_map_studio_pie_gameplay_state", None)
        if callable(clear_gameplay_state):
            clear_gameplay_state()
        else:
            set_gameplay_state = getattr(self.viewport_panel, "set_map_studio_pie_gameplay_state", None)
            if callable(set_gameplay_state):
                set_gameplay_state(None)
        camera = getattr(getattr(self.viewport_panel, "viewport", None), "camera", None)
        snapshot = self._map_studio_pie_camera_snapshot
        if camera is not None and snapshot is not None:
            camera.azimuth = snapshot[0]
            camera.elevation = snapshot[1]
            camera.distance = snapshot[2]
            camera.target = list(snapshot[3])
            camera.fov = snapshot[4]
            camera._near = snapshot[5]
            camera._far = snapshot[6]
        self._map_studio_pie_session = None
        self._map_studio_pie_camera_snapshot = None
        self._map_studio_pie_last_resolved_camera_distance = None
        self._map_studio_pie_active_room_resref = ""
        self._map_studio_pie_actor_warning = ""
        self._map_studio_pie_camera_turn_input = 0.0
        self._map_studio_pie_camera_turn_velocity = 0.0
        self._map_studio_pie_player_action_state = {}
        self._map_studio_pie_dialogue_animation_entities = set()
        self._map_studio_pie_dialogue_animation_policies = {}
        self._map_studio_pie_dialogue_animation_policies_loaded = False
        self._map_studio_pie_dialogue_node_id = ""
        self._map_studio_pie_dialogue_lip_limitation_reported = False
        self._map_studio_pie_dialogue_camera_animation_signature = None
        self._map_studio_pie_dialogue_camera_animation_active = False
        self._map_studio_pie_dialogue_camera_snapshot = None
        self._reset_map_studio_pie_dialogue_line_timer()
        self._map_studio_pie_gameplay_mode = "exploration"
        self._set_map_studio_pie_authoring_enabled(True)
        self._set_map_studio_pie_command_active(False)
        if self._map_studio_pie_scope_text:
            self.map_studio_scope_label.setText(self._map_studio_pie_scope_text)
        request = getattr(getattr(self.viewport_panel, "viewport", None), "_request_render", None)
        if callable(request):
            request(fast=True, reason="Map Studio PIE stopped", camera=True, overlay=True, hud=True)
        self.statusBar().showMessage("Simulation stopped; authoring camera and tools restored.")

    def _handle_map_studio_pie_move_input(self, payload: object) -> None:
        session = self._map_studio_pie_session
        if session is None:
            return
        values = dict(payload or {}) if isinstance(payload, dict) else {}
        camera = getattr(getattr(self.viewport_panel, "viewport", None), "camera", None)
        self._map_studio_pie_camera_turn_input = max(
            -1.0,
            min(1.0, float(values.get("camera_turn", 0.0) or 0.0)),
        )
        session.set_move_input(
            float(values.get("forward", 0.0) or 0.0),
            float(values.get("strafe", 0.0) or 0.0),
            camera_azimuth_degrees=float(getattr(camera, "azimuth", 90.0)),
            run=bool(values.get("run", False)),
        )

    @staticmethod
    def _map_studio_pie_entity_id_from_runtime_id(value: object) -> str:
        """Normalize retained actor IDs back to the registry's authored IDs."""

        text = str(value or "").strip()
        for prefix in (
            "__map_studio_pie_creature__:",
            "__map_studio_pie_door__:",
            "__map_studio_pie_placeable__:",
        ):
            if text.startswith(prefix):
                return text[len(prefix) :]
        return text

    def _publish_map_studio_pie_gameplay_state(self) -> None:
        session = self._map_studio_pie_session
        if session is None:
            return
        setter = getattr(self.viewport_panel, "set_map_studio_pie_gameplay_state", None)
        if callable(setter):
            setter(session.gameplay_snapshot())
        self.viewport_panel.set_map_studio_pie_overlay(session.overlay_geometry())

    def _handle_map_studio_pie_gameplay_action(self, payload: object) -> None:
        """Route one HUD/keyboard action through the headless PIE session."""

        session = self._map_studio_pie_session
        if session is None:
            return
        values = dict(payload or {}) if isinstance(payload, dict) else {"action": str(payload or "")}
        action = str(values.get("action") or values.get("command") or "").strip().lower()
        result: Any = None
        try:
            if action == "focus_cycle":
                result = session.cycle_gameplay_focus(int(values.get("direction", 1) or 1))
            elif action == "focus_entity":
                entity_id = self._map_studio_pie_entity_id_from_runtime_id(values.get("entity_id"))
                result = session.focus_gameplay_entity(entity_id)
            elif action in {"primary", "dialogue_continue"}:
                snapshot = session.gameplay_snapshot()
                dialogue = getattr(snapshot, "dialogue", None)
                if dialogue is not None and not bool(getattr(dialogue, "ended", False)):
                    if bool(getattr(dialogue, "can_continue", False)):
                        result = session.continue_gameplay_dialogue()
                    elif tuple(getattr(dialogue, "choices", ()) or ()):
                        self.statusBar().showMessage("Choose a numbered dialogue reply (1–9).", 4000)
                    else:
                        result = session.continue_gameplay_dialogue()
                else:
                    requested_command = str(values.get("action_command") or "").strip() or None
                    result = session.activate_gameplay_focus(requested_command)
            elif action == "dialogue_choice":
                result = session.choose_gameplay_dialogue(int(values.get("number", 0) or 0))
            elif action == "combat_toggle_pause":
                result = session.toggle_gameplay_combat_pause()
            elif action == "combat_clear_queue":
                result = session.clear_gameplay_combat_queue()
            elif action == "combat_attack":
                target_id = self._map_studio_pie_entity_id_from_runtime_id(values.get("target_id"))
                result = session.activate_gameplay_entity(target_id, "attack")
            elif action == "modal_close":
                if not session.close_gameplay_modal():
                    self._stop_map_studio_pie()
                    return
            elif action == "interact_entity":
                entity_id = self._map_studio_pie_entity_id_from_runtime_id(values.get("entity_id"))
                result = session.activate_gameplay_entity(entity_id)
            elif action == "inventory_take":
                entity_id = self._map_studio_pie_entity_id_from_runtime_id(values.get("entity_id"))
                result = session.take_gameplay_item(
                    entity_id,
                    str(values.get("resref") or ""),
                    int(values.get("quantity", 1) or 1),
                )
            elif action == "inventory_take_all":
                entity_id = self._map_studio_pie_entity_id_from_runtime_id(values.get("entity_id"))
                result = session.take_all_gameplay_items(entity_id)
            else:
                self._log(f"Simulation ignored unknown gameplay action {action or '(blank)'!r}.")
                return
        except Exception as exc:
            self._log(f"Simulation gameplay action {action or '(blank)'} failed: {exc}")
            self.statusBar().showMessage(f"Simulation action failed: {exc}", 6000)
        else:
            message = str(getattr(result, "message", "") or "")
            if message:
                self.statusBar().showMessage(message, 5000)
        self._publish_map_studio_pie_gameplay_state()

    def _map_studio_pie_actor_action_target(self, entity_id: str) -> tuple[Any, dict[str, Any]] | None:
        wanted = str(entity_id or "")
        if wanted == "pie:player":
            engine = self._map_studio_pie_animation_engine
            return (engine, self._map_studio_pie_player_action_state) if engine is not None else None
        for entry in self._map_studio_pie_creature_entries:
            spec = entry.get("spec")
            if str(getattr(spec, "placement_id", "") or "") == wanted:
                return entry.get("engine"), entry
        return None

    def _play_map_studio_pie_actor_action_animation(
        self,
        entity_id: str,
        candidates: tuple[str, ...],
        *,
        role: str,
        loop: bool,
    ) -> bool:
        target = self._map_studio_pie_actor_action_target(entity_id)
        if target is None:
            return False
        engine, state = target
        if engine is None:
            return False
        current = str(getattr(getattr(engine, "current_animation", None), "name", "") or "").lower()
        if not state.get("pie_action_active"):
            state["pie_action_restore"] = current or "pause1"
        for candidate in tuple(dict.fromkeys(str(value or "").strip().lower() for value in candidates)):
            if candidate and engine.play(candidate, loop=loop, blend=True):
                state["pie_action_active"] = True
                state["pie_action_role"] = str(role or "")
                state["pie_action_hold"] = str(role or "") == "death"
                state["pie_action_loop"] = bool(loop)
                return True
        return False

    def _restore_map_studio_pie_actor_action_animation(self, entity_id: str, *, force: bool = False) -> None:
        target = self._map_studio_pie_actor_action_target(entity_id)
        if target is None:
            return
        engine, state = target
        if engine is None or not state.get("pie_action_active"):
            return
        if state.get("pie_action_hold") and not force:
            return
        restore = str(state.get("pie_action_restore") or "pause1")
        if not engine.play(restore, loop=True, blend=True):
            engine.play("pause1", loop=True, blend=True)
        for key in ("pie_action_active", "pie_action_role", "pie_action_hold", "pie_action_loop", "pie_action_restore"):
            state.pop(key, None)

    def _apply_map_studio_pie_combat_animation_event(self, event: Any) -> None:
        role = str(getattr(event, "animation_role", "") or "").strip().lower()
        entity_id = str(getattr(event, "entity_id", "") or "")
        if not role or not entity_id:
            return
        defaults = {
            "ready": ("ready", "combat", "pause1"),
            "attack": ("c2a1", "c1a1", "c3a1", "attack1"),
            "damage": ("c2d1", "c1d1", "c3d1", "damage1", "hit"),
            "death": ("dead1", "death1", "dead", "death"),
        }
        candidates = tuple(getattr(event, "animation_candidates", ()) or ()) + defaults.get(role, ())
        if candidates and not self._play_map_studio_pie_actor_action_animation(
            entity_id,
            candidates,
            role=role,
            loop=role == "ready",
        ):
            self._log(f"Simulation actor {entity_id} has no resolvable {role} combat clip.")

    def _map_studio_pie_dialogue_animation_policy(self, animation_id: int) -> Any:
        """Resolve retail DialogAnimations flags once per PIE run."""

        if not self._map_studio_pie_dialogue_animation_policies_loaded:
            self._map_studio_pie_dialogue_animation_policies_loaded = True
            manager = getattr(self, "resource_manager", None)
            payload = b""
            if manager is not None:
                try:
                    from pykotor.resource.type import ResourceType

                    getter = getattr(manager, "get_strict", None) or getattr(manager, "get", None)
                    game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
                    if callable(getter):
                        try:
                            value = getter("dialoganimations", ResourceType.TwoDA.type_id, game)
                        except TypeError:
                            value = getter("dialoganimations", ResourceType.TwoDA.type_id)
                    else:
                        value = None
                    if isinstance(value, (bytes, bytearray, memoryview)):
                        payload = bytes(value)
                    else:
                        data = getattr(value, "data", None)
                        payload = bytes(data() if callable(data) else data or b"")
                except Exception as exc:
                    self._log(f"Simulation dialogue animation policy lookup failed: {exc}")
            try:
                from src.core.modules.map_studio_pie_dialogue import (
                    load_map_studio_pie_dialogue_animation_policies,
                )

                self._map_studio_pie_dialogue_animation_policies = (
                    load_map_studio_pie_dialogue_animation_policies(payload)
                )
            except Exception:
                self._map_studio_pie_dialogue_animation_policies = {}
        wanted = int(animation_id)
        return self._map_studio_pie_dialogue_animation_policies.get(
            wanted,
            self._map_studio_pie_dialogue_animation_policies.get(wanted % 10000),
        )

    def _sync_map_studio_pie_dialogue_animations(self, gameplay: Any) -> None:
        dialogue = getattr(gameplay, "dialogue", None)
        active = bool(
            dialogue is not None
            and not bool(getattr(dialogue, "ended", False))
            and str(getattr(dialogue, "current_node_id", "") or "")
        )
        dialogue_state = str(getattr(dialogue, "state", "") or "").strip().lower() if active else ""
        node_id = str(getattr(dialogue, "current_node_id", "") or "") if active else ""
        animation_signature = f"{node_id}:{dialogue_state}" if node_id else ""
        if animation_signature == self._map_studio_pie_dialogue_node_id:
            return
        for entity_id in tuple(self._map_studio_pie_dialogue_animation_entities):
            self._restore_map_studio_pie_actor_action_animation(entity_id, force=True)
        self._map_studio_pie_dialogue_animation_entities.clear()
        self._map_studio_pie_dialogue_node_id = animation_signature
        if not active or dialogue_state != "listening":
            return
        owner_id = str(getattr(dialogue, "owner_id", "") or "")
        registry = getattr(self._map_studio_pie_session, "entity_registry", None)
        owner = registry.by_id(owner_id) if registry is not None else None
        owner_tags = {
            "owner",
            str(getattr(owner, "tag", "") or "").strip().lower(),
            str(getattr(dialogue, "speaker_tag", "") or "").strip().lower(),
        }
        try:
            from src.core.modules.map_studio_scene_animations import scene_animation_clip_candidates
        except Exception:
            scene_animation_clip_candidates = lambda _constant: ()
        for participant, constant in tuple(getattr(dialogue, "animations", ()) or ()):
            clean_participant = str(participant or "").strip().lower()
            if clean_participant in {"player", "pc", "listener"}:
                entity_id = "pie:player"
            elif clean_participant in owner_tags or not clean_participant:
                entity_id = owner_id
            else:
                entity = next(
                    (
                        row
                        for row in tuple(getattr(registry, "entities", ()) or ())
                        if str(getattr(row, "tag", "") or "").strip().lower() == clean_participant
                    ),
                    None,
                )
                entity_id = str(getattr(entity, "entity_id", "") or "")
            # PyKotor retains the historical 10000-based DLG value; both the
            # retail table row and the existing clip resolver use its base ID.
            candidates = tuple(scene_animation_clip_candidates(int(constant) % 10000))
            policy = self._map_studio_pie_dialogue_animation_policy(int(constant))
            looping = bool(getattr(policy, "looping", False))
            fire_and_forget = bool(getattr(policy, "fire_and_forget", False))
            overlay = bool(getattr(policy, "overlay", False))
            if overlay:
                self._log(
                    f"Simulation dialogue animation {constant} requests Overlay; the retained actor API "
                    "has no layered track, so PIE presents it as a blended base clip."
                )
            if entity_id and candidates and self._play_map_studio_pie_actor_action_animation(
                entity_id,
                candidates,
                role="dialogue",
                loop=looping,
            ):
                # FireForget clips own their natural one-shot completion; all
                # other node intents are stopped/replaced at the node boundary.
                if not fire_and_forget:
                    self._map_studio_pie_dialogue_animation_entities.add(entity_id)
        # Retail keeps node AnimList body gestures and LIP facial curves on
        # separate channels.  The current retained-actor API has no facial
        # LIP track, while Character Studio's LIPPlayback mutates an authored
        # scene and is therefore unsafe to reuse here.  Preserve only explicit
        # DLG animations; never turn a voiced line into a fabricated looping
        # body-talk animation as a substitute for missing facial playback.
        if (
            str(getattr(dialogue, "voice_resref", "") or "")
            and not self._map_studio_pie_dialogue_lip_limitation_reported
        ):
            self._map_studio_pie_dialogue_lip_limitation_reported = True
            self._log(
                "Simulation dialogue facial animation: authored VO is playing and explicit DLG body "
                "animations are preserved, but retained actors do not yet expose the facial LIP curve "
                "track. PIE leaves lip motion unavailable instead of fabricating a looping body-talk clip."
            )

    def _try_map_studio_pie_dialogue_camera_animation(
        self,
        dialogue: Any,
        *,
        speaker_position: tuple[float, ...],
        listener_position: tuple[float, ...],
    ) -> bool:
        """Try the renderer's optional authored camera-animation hook first."""

        animation_id = getattr(dialogue, "camera_animation", None)
        if animation_id is None:
            self._map_studio_pie_dialogue_camera_animation_signature = None
            self._map_studio_pie_dialogue_camera_animation_active = False
            return False
        signature = (str(getattr(dialogue, "current_node_id", "") or ""), int(animation_id))
        if signature == self._map_studio_pie_dialogue_camera_animation_signature:
            return bool(self._map_studio_pie_dialogue_camera_animation_active)
        self._map_studio_pie_dialogue_camera_animation_signature = signature
        self._map_studio_pie_dialogue_camera_animation_active = False
        viewport = getattr(self.viewport_panel, "viewport", None)
        presenter = getattr(viewport, "play_map_studio_pie_dialogue_camera_animation", None)
        if not callable(presenter):
            # The current ArcBall renderer has no camera-animation track API;
            # keep the ordered attempt explicit and fall through to the placed
            # camera / angle solver instead of pretending the track played.
            return False
        try:
            active = bool(
                presenter(
                    int(animation_id),
                    speaker_position=speaker_position,
                    listener_position=listener_position,
                    field_of_view=getattr(dialogue, "camera_fov", None),
                    camera_height_offset=float(getattr(dialogue, "camera_height_offset", 0.0) or 0.0),
                    target_height_offset=float(getattr(dialogue, "target_height_offset", 0.0) or 0.0),
                )
            )
        except Exception as exc:
            self._log(f"Simulation dialogue camera animation {animation_id} fell back: {exc}")
            active = False
        self._map_studio_pie_dialogue_camera_animation_active = active
        return active

    def _update_map_studio_pie_dialogue_camera(self, gameplay: Any, camera: Any) -> None:
        """Apply animation-first, placed-camera, then angle-based framing."""

        mode = str(getattr(gameplay, "mode", "exploration") or "exploration")
        dialogue = getattr(gameplay, "dialogue", None)
        if mode != "dialogue" or dialogue is None:
            if self._map_studio_pie_gameplay_mode == "dialogue":
                self._set_map_studio_pie_player_facial_detail_visible(False)
                snapshot = self._map_studio_pie_dialogue_camera_snapshot
                if snapshot is not None:
                    camera.azimuth, camera.elevation, camera.distance, camera.fov = snapshot
                    self._map_studio_pie_desired_camera_distance = float(snapshot[2])
                else:
                    camera.fov = 55.0
                    self._map_studio_pie_desired_camera_distance = 3.2
                    camera.distance = 3.2
                self._map_studio_pie_dialogue_camera_snapshot = None
                self._map_studio_pie_dialogue_camera_animation_signature = None
                self._map_studio_pie_dialogue_camera_animation_active = False
                reset_collision = getattr(self._map_studio_pie_session, "reset_camera_collision", None)
                if callable(reset_collision):
                    reset_collision()
            self._map_studio_pie_gameplay_mode = mode
            return
        session = self._map_studio_pie_session
        registry = getattr(session, "entity_registry", None)
        owner = registry.by_id(str(getattr(dialogue, "owner_id", "") or "")) if registry is not None else None
        if owner is None:
            self._map_studio_pie_gameplay_mode = mode
            return
        player = tuple(session.state.position)
        target = tuple(getattr(owner, "position", player) or player)
        target_height_offset = float(getattr(dialogue, "target_height_offset", 0.0) or 0.0)
        camera_height_offset = float(getattr(dialogue, "camera_height_offset", 0.0) or 0.0)
        entering_dialogue = self._map_studio_pie_gameplay_mode != "dialogue"
        if entering_dialogue:
            self._set_map_studio_pie_player_facial_detail_visible(True)
            self._map_studio_pie_dialogue_camera_snapshot = (
                float(camera.azimuth),
                float(camera.elevation),
                float(camera.distance),
                float(getattr(camera, "fov", 55.0)),
            )
        if self._try_map_studio_pie_dialogue_camera_animation(
            dialogue,
            speaker_position=target,
            listener_position=player,
        ):
            if getattr(dialogue, "camera_fov", None) is not None:
                camera.fov = float(dialogue.camera_fov)
            self._map_studio_pie_gameplay_mode = mode
            return
        # Placed shots (CameraAngle 6) resolve the authored area camera; the
        # discrete angle shot table and all framing math live in the headless
        # solver so they stay testable outside Qt.
        from src.core.modules.map_studio_pie_dialogue_camera import (
            DialoguePlacedCamera,
            solve_map_studio_pie_dialogue_camera,
        )

        placed_camera = None
        camera_id = getattr(dialogue, "camera_id", None)
        if int(getattr(dialogue, "camera_angle", 0) or 0) == 6 and camera_id is not None and registry is not None:
            authored_camera = next(
                (
                    row
                    for row in registry.of_kind("camera")
                    if int((getattr(row, "metadata", {}) or {}).get("camera_id", -1)) == int(camera_id)
                ),
                None,
            )
            if authored_camera is not None:
                metadata = dict(getattr(authored_camera, "metadata", {}) or {})
                placed_camera = DialoguePlacedCamera(
                    position=tuple(
                        float(value)
                        for value in tuple(getattr(authored_camera, "position", player) or player)[:3]
                    ),
                    height=float(metadata.get("height", 0.0) or 0.0),
                    field_of_view=float(metadata.get("field_of_view", 45.0) or 45.0),
                )
        framing = solve_map_studio_pie_dialogue_camera(
            listener_position=player,
            speaker_position=target,
            camera_angle=int(getattr(dialogue, "camera_angle", 0) or 0),
            camera_fov=getattr(dialogue, "camera_fov", None),
            camera_height_offset=camera_height_offset,
            target_height_offset=target_height_offset,
            placed_camera=placed_camera,
        )
        camera.target = list(framing.target)
        camera.azimuth = framing.azimuth_deg
        camera.elevation = framing.elevation_deg
        camera.fov = framing.fov
        self._map_studio_pie_desired_camera_distance = framing.distance
        if framing.mode == "placed":
            camera.distance = framing.distance
        else:
            # An angle shot never pushes the camera further out than its authored
            # shot distance, but a player who walked closer keeps that framing.
            camera.distance = min(float(getattr(camera, "distance", framing.distance) or framing.distance), framing.distance)
        if entering_dialogue:
            reset_collision = getattr(session, "reset_camera_collision", None)
            if callable(reset_collision):
                reset_collision()
        self._map_studio_pie_gameplay_mode = mode

    def _set_map_studio_pie_destination(self, position: object) -> None:
        session = self._map_studio_pie_session
        if session is None:
            return
        try:
            point = tuple(float(value) for value in tuple(position)[:3])
        except (TypeError, ValueError):
            return
        if len(point) < 3:
            return
        if session.set_destination(point, run=True):
            self.statusBar().showMessage("Simulation route accepted; click-to-move is following connected WOK faces.")
        else:
            self.statusBar().showMessage("Simulation route rejected: destination is outside clearance or disconnected.", 5000)

    def _sync_map_studio_pie_room_visibility(self, *, force: bool = False) -> None:
        """Apply KOTOR-style room visibility when the player crosses a WOK seam."""

        session = self._map_studio_pie_session
        preview_model = self._map_studio_pie_runtime_preview_model
        if session is None or preview_model is None:
            return
        current_room = str(
            getattr(session, "current_room_resref", lambda: "")() or ""
        ).strip().lower()
        if not current_room:
            return
        if not force and current_room == self._map_studio_pie_active_room_resref:
            return
        from src.core.modules.map_studio_pie import (
            apply_map_studio_pie_room_visibility,
        )

        result = apply_map_studio_pie_room_visibility(
            preview_model,
            active_room_resref=current_room,
            connected_room_resrefs=getattr(
                session,
                "visible_room_resrefs",
                lambda: (current_room,),
            )(),
            player_position=tuple(
                getattr(getattr(session, "state", None), "position", ())
                or (0.0, 0.0, 0.0)
            ),
        )
        self._map_studio_pie_active_room_resref = current_room
        if result.changed_node_count:
            self._log(
                "Simulation room visibility: "
                f"{current_room} + {max(0, len(result.visible_room_resrefs) - 1)} "
                f"threshold/nearby room(s); {len(result.hidden_room_resrefs)} distant room(s) culled."
            )

    def _tick_map_studio_pie(self) -> None:
        """Advance fixed-step simulation without rebuilding or reloading the renderer."""

        session = self._map_studio_pie_session
        if session is None:
            return
        now = perf_counter()
        delta_time = max(0.0, now - self._map_studio_pie_last_time)
        self._map_studio_pie_last_time = now
        frame = session.advance(delta_time)
        self._sync_map_studio_pie_room_visibility()
        gameplay = getattr(frame, "gameplay", None)
        for event in tuple(getattr(frame, "events", ()) or ()):
            if str(getattr(event, "animation_role", "") or ""):
                self._apply_map_studio_pie_combat_animation_event(event)
        if gameplay is not None:
            self._sync_map_studio_pie_dialogue_animations(gameplay)
            self._sync_map_studio_pie_dialogue_audio(gameplay)
            self._advance_map_studio_pie_dialogue_line_timer(gameplay, delta_time)
        camera = getattr(getattr(self.viewport_panel, "viewport", None), "camera", None)
        camera_blocked = False
        if camera is not None:
            camera_turn = float(self._map_studio_pie_camera_turn_input)
            turn_delta = min(max(delta_time, 0.0), 0.05)
            target_turn_velocity = camera_turn * 200.0
            current_turn_velocity = float(self._map_studio_pie_camera_turn_velocity)
            # Installed K1/K2 INIs expose a 200 deg/s target, 500 deg/s^2
            # acceleration, and 2000 deg/s^2 deceleration.  Ghidra has not
            # recovered the exact Odyssey integrator, so this is an explicit
            # clean-room approach-to-target using those retail constants.
            slowing_or_reversing = (
                abs(target_turn_velocity) <= 1.0e-7
                or current_turn_velocity * target_turn_velocity < 0.0
            )
            turn_acceleration = 2000.0 if slowing_or_reversing else 500.0
            max_velocity_change = turn_acceleration * turn_delta
            velocity_error = target_turn_velocity - current_turn_velocity
            velocity_change = max(-max_velocity_change, min(max_velocity_change, velocity_error))
            current_turn_velocity += velocity_change
            if abs(current_turn_velocity) <= 1.0e-7 and abs(target_turn_velocity) <= 1.0e-7:
                current_turn_velocity = 0.0
            self._map_studio_pie_camera_turn_velocity = current_turn_velocity
            if abs(current_turn_velocity) > 1.0e-7:
                camera.azimuth = (
                    float(camera.azimuth) - (current_turn_velocity * turn_delta)
                ) % 360.0
            target = session.player_eye_target()
            camera.target = list(target)
            session.set_camera_azimuth(float(camera.azimuth))
            desired_distance = max(session.config.minimum_camera_distance, self._map_studio_pie_desired_camera_distance)
            azimuth = math.radians(float(camera.azimuth))
            elevation = math.radians(float(camera.elevation))
            cosine_elevation = math.cos(elevation)
            desired_eye = (
                target[0] + desired_distance * cosine_elevation * math.cos(azimuth),
                target[1] + desired_distance * cosine_elevation * math.sin(azimuth),
                target[2] + desired_distance * math.sin(elevation),
            )
            resolved = session.resolve_camera_distance(target, desired_eye, delta_time=delta_time)
            camera.distance = resolved
            self._map_studio_pie_last_resolved_camera_distance = resolved
            camera_blocked = resolved + 0.02 < desired_distance
            if gameplay is not None:
                self._update_map_studio_pie_dialogue_camera(gameplay, camera)
            # The selected-target plate is projected and composed at the scene
            # presentation boundary.  Updating a separate full-canvas child at
            # this 16 ms simulation cadence races QLabel pixmap publication and
            # can preserve old reticles while postponing the new world frame.
        player_runtime_row = self._update_map_studio_pie_player_actor(frame, delta_time)
        runtime_rows = list(self._update_map_studio_pie_creature_actors(delta_time))
        self._update_map_studio_pie_party_actors(delta_time)
        runtime_rows.extend(self._update_map_studio_pie_door_actors(frame, delta_time))
        if player_runtime_row is not None:
            runtime_rows.insert(0, player_runtime_row)
        if runtime_rows:
            self.viewport_panel.viewport.update_runtime_character_frames(
                runtime_rows,
                camera_changed=True,
            )
        # Any retained actor batch already requested the frame.  This includes
        # NPC-only PIE when the player model is unavailable; requesting a
        # second camera-only repaint caused a duplicate present/flicker.
        actor_rendered = bool(runtime_rows)
        self._update_map_studio_pie_ambient_audio(session, frame)
        status_bucket = int(frame.simulation_time * 10.0)
        if status_bucket != self._map_studio_pie_status_bucket:
            self._map_studio_pie_status_bucket = status_bucket
            self._publish_map_studio_pie_gameplay_state()
            motion = "BLOCKED" if frame.blocked else ("moving" if frame.moving else "idle")
            route = f" | route {len(frame.path)}" if frame.destination is not None else ""
            camera_state = " | camera blocked" if camera_blocked else ""
            animation_state = f" | anim {self._map_studio_pie_animation_name or 'unavailable'}"
            audio_state = f" | {self._map_studio_pie_audio_summary}"
            creature_state = f" | {self._map_studio_pie_creature_summary}"
            gameplay_state = f" | {getattr(gameplay, 'mode', 'exploration')}"
            self.viewport_panel.marker_summary_label.setText(
                "Simulation — not KOTOR proof | "
                f"Player {frame.position[0]:.2f}, {frame.position[1]:.2f}, {frame.position[2]:.2f} "
                f"| WOK face {frame.face_index} | {motion}{route}{camera_state}{animation_state}"
                f"{creature_state}{audio_state}{gameplay_state} | Esc closes/stops"
            )
        for event in frame.events:
            self._log(f"Simulation: {event.message}")
        if not actor_rendered:
            request = getattr(getattr(self.viewport_panel, "viewport", None), "_request_render", None)
            if callable(request):
                # The resident map scene is unchanged.  A scene-dirty request
                # forces pygfx to rebuild every room/resource bridge on each
                # 16 ms tick, which is catastrophic on stock modules such as
                # 207TEL when the optional player actor cannot be resolved.
                request(fast=True, reason="Map Studio PIE camera frame", camera=True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._confirm_discard_or_save(restore_discarded_sidecars=True):
            self._stop_map_studio_pie()
            event.accept()
        else:
            event.ignore()

    def _toolbar_action(self, action: str) -> None:
        if str(action or "").startswith("tool_belt:"):
            key = str(action).split(":", 1)[1]
            tool_action = self._map_studio_tool_action_index.get(key)
            if tool_action is None:
                tool_action = next(
                    (
                        candidate
                        for candidate in self.controller.available_map_studio_tool_belt_actions()
                        if str(getattr(candidate, "key", "") or "") == key
                    ),
                    None,
                )
            if tool_action is not None:
                self._handle_map_studio_tool_belt_action(tool_action)
            return
        mapping = {
            "new": self.new_kmap,
            "open": self.open_kmap,
            "save": self.save_kmap,
            "import_module": self.import_module,
            "add_room": lambda: self._handle_tab_action("Add Room"),
            "add_module": lambda: self._handle_tab_action("Add Module"),
            "validate": self.validate_kmap,
            "simulate": self.toggle_map_studio_pie,
            "build": self.build_module_files,
            "generate_module_files": self.build_module_files,
            "export_fbx": lambda: self.export_fbx(False),
        }
        callback = mapping.get(action)
        if callback:
            callback()

    def _indexed_lyt_resource_rows(self) -> list[dict[str, Any]]:
        try:
            from src.core.assets.resource_manager import RES_LYT
        except Exception:
            RES_LYT = 3000
        manager = getattr(self, "resource_manager", None)
        if manager is None:
            return []
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for game, getter_name in (("K1", "get_k1"), ("K2", "get_k2")):
            getter = getattr(manager, getter_name, None)
            install = getter() if callable(getter) else None
            list_resrefs = getattr(install, "list_resrefs", None)
            if not callable(list_resrefs):
                continue
            try:
                resrefs = list_resrefs(RES_LYT)
            except Exception:
                resrefs = []
            source = str(getattr(install, "game_dir", "") or getattr(install, "root", "") or "")
            for raw_resref in resrefs or ():
                resref = str(raw_resref or "").strip().lower()
                if not resref:
                    continue
                key = (game, resref)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    summary = self._lyt_resource_summary(manager.get(resref, RES_LYT, game))
                except Exception:
                    summary = {"room_count": 0, "doorhook_count": 0}
                rows.append(
                    {
                        "game": game,
                        "resref": resref,
                        "source": source or "configured game resources",
                        "room_count": int(summary.get("room_count", 0) or 0),
                        "doorhook_count": int(summary.get("doorhook_count", 0) or 0),
                    }
                )
        rows.sort(key=lambda row: (str(row.get("game", "")), str(row.get("resref", ""))))
        return rows

    def _lyt_resource_summary(self, data: bytes | bytearray | str | None) -> dict[str, int]:
        if isinstance(data, str):
            text = data
        else:
            text = bytes(data or b"").decode("latin-1", errors="replace")
        try:
            from src.core.modules import module_format as mf
        except Exception:
            from core.modules import module_format as mf  # type: ignore
        lyt = mf.LYTLayout.from_text(text)
        return {
            "room_count": len(getattr(lyt, "rooms", []) or []),
            "doorhook_count": len(getattr(lyt, "doorhooks", []) or []),
        }

    def _choose_indexed_lyt_resource(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        dialog = _MapStudioLytResourceDialog(self, rows=rows)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dialog.selected_row()

    def _open_indexed_lyt_resource_picker(self) -> None:
        rows = self._indexed_lyt_resource_rows()
        if not rows:
            QtWidgets.QMessageBox.information(
                self,
                "Load Indexed LYT",
                "No indexed LYT resources are available from the configured KOTOR game directories.",
            )
            return
        row = self._choose_indexed_lyt_resource(rows)
        if row is not None:
            self._load_indexed_lyt_resource(row)

    def _load_indexed_lyt_resource(self, row: dict[str, Any]) -> None:
        try:
            from src.core.assets.resource_manager import RES_LYT
        except Exception:
            RES_LYT = 3000
        manager = getattr(self, "resource_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(self, "Load Indexed LYT", "Game resource manager is not available.")
            return
        game = str(row.get("game", "") or "K1").upper()
        resref = str(row.get("resref", "") or "").strip().lower()
        if not resref:
            QtWidgets.QMessageBox.warning(self, "Load Indexed LYT", "Selected LYT resource is missing a resref.")
            return
        try:
            data = manager.get(resref, RES_LYT, game)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Indexed LYT", f"Could not read {game}:{resref}.lyt: {exc}")
            return
        if not data:
            QtWidgets.QMessageBox.warning(self, "Load Indexed LYT", f"{game}:{resref}.lyt could not be read.")
            return

        # The indexed picker historically stopped at LYT.  That explains the
        # deceptive state where a stock map displayed its two room models but
        # had no GIT creatures/placeables/doors/sounds, no transition metadata,
        # and no IFO player start.  Prefer the complete installed module when
        # its area resref maps directly to a capsule; retain an explicit
        # layout-only fallback for standalone LYT resources and K1 areas whose
        # module filename differs from their ARE/LYT resref.
        game_root = ""
        game_dir_getter = getattr(manager, "game_dir", None)
        if callable(game_dir_getter):
            try:
                game_root = str(game_dir_getter(game) or "").strip()
            except Exception:
                game_root = ""
        if not game_root:
            game_root = str(row.get("source", "") or "").strip()
        modules_dir = Path(game_root) / "Modules" if game_root else Path()
        module_container: Path | None = None
        if game_root and modules_dir.is_dir():
            wanted = {f"{resref}.rim", f"{resref}.mod"}
            for candidate in (modules_dir / f"{resref}.rim", modules_dir / f"{resref}.mod"):
                if candidate.is_file():
                    module_container = candidate
                    break
            if module_container is None:
                # Keep discovery correct on case-sensitive test/dev volumes;
                # the shipped Windows install is naturally case-insensitive.
                try:
                    module_container = next(
                        (candidate for candidate in modules_dir.iterdir() if candidate.name.lower() in wanted),
                        None,
                    )
                except OSError:
                    module_container = None

        if module_container is not None:
            existing_authored = bool(
                dict(getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
            )
            existing_scene = bool(
                tuple(getattr(self.project, "rooms", ()) or ())
                or tuple(getattr(self.project, "modules", ()) or ())
                or tuple(getattr(self.project, "blueprints", ()) or ())
            )
            if existing_authored or existing_scene:
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "Replace Current Map Studio Scene",
                    f"Load the complete {game}:{resref} module and replace the current Map Studio scene?\n\n"
                    "Save the current KMAP first if you need to keep it.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Cancel,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return

            before_project = deepcopy(self.project)
            started = perf_counter()
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.statusBar().showMessage(
                    f"Loading complete {game}:{resref}: ARE/GIT/IFO gameplay, player start, and module resources..."
                )
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                self.controller.new_project(
                    name=resref,
                    game=game,
                    author=str(getattr(before_project, "author", "") or ""),
                )
                ok, import_message = self.controller.import_stock_module_from_rim(
                    module_resref=resref,
                    modules_dir=str(modules_dir),
                    game=game,
                    resource_manager=manager,
                )
                if not ok:
                    self.controller.model.set_project(before_project)
                    QtWidgets.QMessageBox.warning(self, "Load Stock Module", import_message)
                    self._refresh_all(f"Full module load failed; restored the previous Map Studio scene. {import_message}")
                    return

                placement_count = len(tuple(self.controller.authored_gameplay_placements() or ()))
                self.statusBar().showMessage(
                    f"Hydrated {placement_count} gameplay object(s) and the IFO player start; "
                    "converting room geometry for editing..."
                )
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                conversion_ok, conversion_message = self.controller.convert_all_stock_rooms_to_imported_mesh(
                    resource_manager=manager,
                )
                self.statusBar().showMessage(
                    "Preparing viewport object models. Diffuse textures and baked lightmaps continue streaming after the first frame..."
                )
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                self._last_game_modules_dir = str(modules_dir)
                elapsed_ms = (perf_counter() - started) * 1000.0
                message = (
                    f"Loaded complete {game}:{resref} in {elapsed_ms / 1000.0:.2f}s: "
                    f"{placement_count} gameplay object(s), IFO player start, and editable room geometry. "
                    f"{conversion_message} Textures/lightmaps may stream for a few seconds on the first view."
                )
                if not conversion_ok:
                    QtWidgets.QMessageBox.warning(self, "Stock Room Conversion", conversion_message)
                self._refresh_all(message)
                resolved = len(tuple(self.controller.last_map_studio_resolved_placement_ids or ()))
                unresolved = len(tuple(self.controller.last_map_studio_unresolved_placement_ids or ()))
                self.statusBar().showMessage(
                    f"{message} Resolved {resolved} placed object model(s); "
                    f"{unresolved} model-bearing placement(s) use honest marker fallbacks.",
                    15000,
                )
                return
            except Exception as exc:
                self.controller.model.set_project(before_project)
                failure = f"Full module load failed; restored the previous Map Studio scene. {exc}"
                self._refresh_all(failure)
                QtWidgets.QMessageBox.warning(self, "Load Stock Module", failure)
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        if game and str(getattr(self.project, "game", "") or "").upper() != game:
            self.project.game = game
            self.controller.model.game = game
        result = self.controller.layout_service.load_lyt_bytes(
            self.project,
            bytes(data),
            module_id=self.controller.model.active_module_id,
            source_module=resref,
            source_path=f"game_resource://{game}/{resref}.lyt",
        )
        self._refresh_all(
            f"{result.message} ({game}:{resref}.lyt). Layout only: no matching installed "
            "module capsule was found, so GIT gameplay objects, door transitions, sounds, and the IFO player start "
            "were not hydrated. Use File -> Import Module File for a capsule whose filename differs from its LYT resref."
        )

    @staticmethod
    def _room_connection_hook_choice_label(hook: object) -> str:
        label = str(getattr(hook, "label", "") or getattr(hook, "hook_id", "") or "Room opening")
        kind = str(getattr(hook, "opening_kind", "door") or "door")
        intent = str(getattr(hook, "intent", "connectable") or "connectable").replace("_", " ")
        width = float(getattr(hook, "width", 0.0) or 0.0)
        height = float(getattr(hook, "height", 0.0) or 0.0)
        return f"{label} — {kind} {width:g}m x {height:g}m — {intent}"

    def _connect_authored_room_openings(self) -> None:
        audit = self.controller.authored_room_connection_audit()
        hooks = [hook for hook in tuple(getattr(audit, "hooks", ()) or ()) if bool(getattr(hook, "passable", False))]
        rooms = {str(getattr(hook, "room_resref", "") or "") for hook in hooks}
        if len(hooks) < 2 or len(rooms) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Connect Room Openings",
                "Create floor-level doorway openings on at least two authored floor-plan rooms first. "
                "Use Builder → Floor-plan wall opening, then return here to connect them.",
            )
            return
        source_labels = [self._room_connection_hook_choice_label(hook) for hook in hooks]
        source_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Connect Room Openings",
            "Opening on the room to move and align:",
            source_labels,
            0,
            False,
        )
        if not accepted:
            return
        source = hooks[source_labels.index(str(source_label))]
        targets = [
            hook
            for hook in hooks
            if str(getattr(hook, "room_resref", "") or "") != str(getattr(source, "room_resref", "") or "")
        ]
        target_labels = [self._room_connection_hook_choice_label(hook) for hook in targets]
        target_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Connect Room Openings",
            "Opening on the room that stays in place:",
            target_labels,
            0,
            False,
        )
        if not accepted:
            return
        target = targets[target_labels.index(str(target_label))]
        try:
            update = self.controller.connect_authored_room_openings(
                source_hook_id=str(getattr(source, "hook_id", "") or ""),
                target_hook_id=str(getattr(target, "hook_id", "") or ""),
                align_source=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Connect Room Openings", str(exc))
            return
        self._refresh_all(update.summary)

    def _set_authored_room_opening_intent(self) -> None:
        audit = self.controller.authored_room_connection_audit()
        hooks = [
            hook
            for hook in tuple(getattr(audit, "hooks", ()) or ())
            if not str(getattr(hook, "connected_room_resref", "") or "").strip()
            and str(getattr(hook, "opening_kind", "door") or "door").strip().lower()
            not in {"window", "backdrop", "view"}
        ]
        if not hooks:
            QtWidgets.QMessageBox.information(
                self,
                "Set Opening Intent",
                "There are no unconnected room openings to classify. Draw a doorway or drag in a vanilla room first.",
            )
            return
        labels = [self._room_connection_hook_choice_label(hook) for hook in hooks]
        selected_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Set Opening Intent",
            "Room opening:",
            labels,
            0,
            False,
        )
        if not accepted:
            return
        hook = hooks[labels.index(str(selected_label))]
        choices = (
            ("Available connector", "connectable"),
            ("Sealed — locked area-style door", "sealed"),
            ("Intentional module exit", "external"),
        )
        current_intent = str(getattr(hook, "intent", "connectable") or "connectable")
        current_index = next(
            (index for index, (_label, value) in enumerate(choices) if value == current_intent),
            0,
        )
        intent_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Set Opening Intent",
            "How should this opening behave?",
            [label for label, _value in choices],
            current_index,
            False,
        )
        if not accepted:
            return
        intent = dict(choices)[str(intent_label)]
        try:
            update = self.controller.set_authored_room_opening_intent(
                hook_id=str(getattr(hook, "hook_id", "") or ""),
                intent=intent,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Set Opening Intent", str(exc))
            return
        self._refresh_all(update.summary)

    def _show_authored_room_connection_audit(self) -> None:
        audit = self.controller.authored_room_connection_audit()
        summary = str(getattr(audit, "summary", "") or "Room connection audit is unavailable.")
        warnings = tuple(getattr(audit, "warnings", ()) or ())
        details = "\n".join(f"• {warning}" for warning in warnings[:10])
        if not details:
            details = "No opening-alignment warnings."
        QtWidgets.QMessageBox.information(
            self,
            "Room Connection Audit",
            f"{summary}\n\n{details}\n\n"
            "This proves KMAP opening alignment and symmetric VIS intent only. "
            "Export validation must still prove WOK transition structure, and KOTOR traversal requires a live warp test.",
        )

    def _snap_selected_authored_rooms_to_grid(self) -> None:
        selected = tuple(getattr(self, "_map_studio_selected_rooms", ()) or ())
        if not selected:
            QtWidgets.QMessageBox.information(
                self,
                "Snap Rooms to Grid",
                "Select one or more authored rooms in the viewport first. Ctrl/Shift-click adds rooms to the selection.",
            )
            return
        grid_size, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Snap Rooms to Grid",
            "XY grid size in meters:",
            1.0,
            0.01,
            1000.0,
            2,
        )
        if not accepted:
            return
        try:
            message = self.controller.snap_authored_rooms_to_grid(selected, grid_size=float(grid_size))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Snap Rooms to Grid", str(exc))
            return
        self._refresh_all(message)

    def _auto_arrange_authored_rooms(self) -> None:
        selected = tuple(getattr(self, "_map_studio_selected_rooms", ()) or ())
        if not selected:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Auto Arrange Rooms",
                "No rooms are selected. Auto-arrange every authored room in the module?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        spacing, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "Auto Arrange Rooms",
            "Clear spacing between room geometry in meters:",
            1.0,
            0.0,
            1000.0,
            2,
        )
        if not accepted:
            return
        try:
            message = self.controller.auto_arrange_authored_rooms(selected, spacing=float(spacing))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto Arrange Rooms", str(exc))
            return
        self._refresh_all(message)

    def _handle_tab_action(self, action: str) -> None:
        if action == "Create grdev01 Dev Room":
            result = self.controller.create_dev_test_authored_module()
            readiness = result.readiness
            message = "Created grdev01 authored module with one primitive room, generated walkmesh intent, player start, and test placeable."
            if readiness is not None:
                message = f"{message} Readiness: {readiness.capability_stage}."
            self._refresh_all(message)
            return
        if action == "Create grgold01 Golden Proof Module":
            result = self.controller.create_golden_test_authored_module()
            readiness = result.readiness
            message = (
                "Created grgold01 golden proof module with room geometry, WOK player start, "
                "placeable, waypoint, door transition intent, and NPC."
            )
            if readiness is not None:
                message = f"{message} Readiness: {readiness.capability_stage}."
            self._refresh_all(message)
            return
        if action == "Load LYT":
            self._open_indexed_lyt_resource_picker()
            return
        if action == "Load WOK":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load WOK", "", "Walkmesh files (*.wok *.dwk *.pwk *.bwm);;All files (*.*)")
            if path:
                result = self.controller.load_wok(path)
                self._refresh_all(result.message)
            return
        if action == "Save WOK":
            room = str(self.walkmesh_tab._current_room_resref() or "").strip()
            if not room:
                QtWidgets.QMessageBox.information(
                    self, "Save WOK", "Pick a room in the Walkmesh tab's Room dropdown first.")
                return
            data = self.controller.map_studio_room_walkmesh_bytes(room)
            if not data:
                QtWidgets.QMessageBox.warning(
                    self, "Save WOK", f"Room {room} has no walkmesh geometry to save.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save WOK", f"{room}.wok", "Walkmesh files (*.wok)")
            if path:
                Path(path).write_bytes(data)
                message = f"Saved {room}.wok ({len(data)} bytes)."
                self._log(f"Map Studio: {message} -> {path}")
                self.statusBar().showMessage(message, 6000)
            return
        if action == "Walkmesh Boundary Rules":
            QtWidgets.QMessageBox.information(
                self,
                "KOTOR Walkmesh Boundaries",
                "KOTOR room WOK files describe the walkable floor region and its perimeter loops. "
                "Do not bake vertical wall or ceiling triangles into the WOK; an enclosing NON_WALK slab can "
                "freeze player movement. Build visible walls with the room modeling tools, and let the floor "
                "boundary/perimeter block movement.",
            )
            return
        if action == "Generate from Selected Floor Faces":
            selection = tuple(self.viewport_panel.map_studio_component_selection() or ())
            selected_faces = tuple(
                row
                for row in selection
                if str(row.get("component_type") or "") == "face"
                and int(row.get("face_index", -1)) >= 0
            )
            if not selected_faces:
                QtWidgets.QMessageBox.information(
                    self,
                    action,
                    "Switch to Face mode and select only the imported room faces that are truly walkable floor. "
                    "Do not select roofs, tables, decorative ledges, walls, or ceilings.",
                )
                return
            rooms = {str(row.get("room_resref") or "").strip() for row in selected_faces}
            rooms.discard("")
            if len(rooms) != 1:
                QtWidgets.QMessageBox.warning(
                    self,
                    action,
                    "A reviewed floor selection must belong to exactly one imported room.",
                )
                return
            room = next(iter(rooms))
            room_spec = self.controller.imported_mesh_room(room)
            primitive = getattr(room_spec, "primitive", None)
            if not isinstance(primitive, ImportedMeshRoomPrimitive):
                QtWidgets.QMessageBox.warning(
                    self,
                    action,
                    f"Room {room} is not editable imported geometry.",
                )
                return
            if primitive.wok is not None and primitive.wok.faces:
                QtWidgets.QMessageBox.information(
                    self,
                    action,
                    f"Room {room} already has an authoritative source WOK. It was not replaced. "
                    "Use Fill Floor Faces for a reviewed coverage repair, or edit the WOK directly.",
                )
                return
            surface_faces: dict[int, tuple[int, ...]] = {}
            grouped_faces: dict[int, set[int]] = {}
            for row in selected_faces:
                role = str(row.get("mesh_role") or "")
                surface_index = imported_mesh_surface_index_for_role(primitive, role)
                if surface_index < 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        action,
                        f"Selected mesh role {role or '(unnamed)'} is not an editable surface in {room}.",
                    )
                    return
                grouped_faces.setdefault(surface_index, set()).add(int(row["face_index"]))
            surface_faces.update(
                (surface_index, tuple(sorted(face_indices)))
                for surface_index, face_indices in sorted(grouped_faces.items())
            )
            face_count = sum(len(indices) for indices in surface_faces.values())
            default_reason = (
                f"Reviewed {face_count} selected render face(s) in the Map Studio viewport as the real "
                f"walkable floor for {room}."
            )
            reason, accepted = QtWidgets.QInputDialog.getText(
                self,
                "Confirm Walkmesh Floor Intent",
                "Why are these selected faces safe to use as KOTOR floor collision?",
                QtWidgets.QLineEdit.Normal,
                default_reason,
            )
            if not accepted:
                return
            ok, message = self.controller.prepare_imported_room_walkmesh_generation_intent(
                room_resref=room,
                surface_faces=surface_faces,
                reason=str(reason or ""),
            )
            if not ok:
                QtWidgets.QMessageBox.warning(self, action, message)
                return
            generated, generation_message = self.controller.auto_generate_map_studio_walkmesh()
            combined_message = f"{message} {generation_message}"
            if not generated:
                QtWidgets.QMessageBox.warning(self, action, combined_message)
                return
            self._select_map_studio_component_mode("walkmesh")
            self._refresh_all(combined_message)
            return
        if action == "Auto Generate Walkmesh":
            try:
                ok, message = self.controller.auto_generate_map_studio_walkmesh()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Auto Generate Walkmesh", str(exc))
                return
            if not ok:
                QtWidgets.QMessageBox.information(self, "Auto Generate Walkmesh", message)
                return
            # Re-audit and show the freshly generated walkmesh overlay.
            try:
                status = self.controller.authored_walkmesh_status()
                self.walkmesh_tab.set_walkmesh_status(status)
                self.walkmesh_tab.set_room_surface_choices(self.controller.authored_walkmesh_room_surface_choices())
            except Exception:
                pass
            self._select_map_studio_component_mode("walkmesh")
            self.statusBar().showMessage(message, 8000)
            self._log(f"Map Studio: {message}")
            self._refresh_all(message)
            return
        if action == "Fill Floor Faces":
            room = str(self.walkmesh_tab._current_room_resref() or "").strip()
            rooms = [room] if room else [
                spec.normalised_resref()
                for spec in tuple(getattr(self.controller._map_studio_authored_project_snapshot(), "rooms", ()) or ())
            ]
            if not rooms:
                QtWidgets.QMessageBox.information(
                    self, "Fill Floor Faces",
                    "Load or import an authored module first; this patches imported room WOKs "
                    "with walkable faces wherever visible floor has no walkmesh coverage.")
                return
            messages: list[str] = []
            for resref in rooms:
                try:
                    ok, message = self.controller.fill_authored_room_wok_from_floors(room_resref=resref)
                except Exception as exc:
                    ok, message = False, f"Room {resref}: {exc}"
                if ok and "added" in message:
                    messages.append(message)
                elif not ok and room:
                    QtWidgets.QMessageBox.warning(self, "Fill Floor Faces", message)
                    return
            summary = " ".join(messages) if messages else "All room walkmeshes already cover their visible floors."
            self.statusBar().showMessage(summary, 8000)
            self._log(f"Map Studio: {summary}")
            self._refresh_all(summary)
            return
        if action in {"Generate Walkmesh", "Validate Walkmesh"}:
            try:
                # Authored WOK is a deterministic derivative of the current
                # room/terrain intent.  This call recompiles every room and
                # audits the resulting faces without duplicating a binary WOK
                # blob in the human-readable KMAP or adding a fake undo item.
                status = self.controller.authored_walkmesh_status()
                room_surfaces = self.controller.authored_walkmesh_room_surface_choices()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, action, str(exc))
                return

            self.walkmesh_tab.set_walkmesh_status(status)
            self.walkmesh_tab.set_room_surface_choices(room_surfaces)
            if int(getattr(status, "terrain_room_count", 0) or 0) > 0:
                self.viewport_panel.set_terrain_walkability_overlay(
                    self.controller.authored_terrain_walkability_overlay()
                )
            self._select_map_studio_component_mode("walkmesh")
            self.workflow_panel.set_active_authoring_context(
                "Walkmesh: current authored geometry compiled and audited; export will materialize the engine WOK"
            )

            if action == "Generate Walkmesh":
                prefix = "Regenerated derived WOK from current authored geometry"
            else:
                prefix = "Walkmesh validation passed" if status.ready else "Walkmesh validation blocked"
            message = (
                f"{prefix}: {int(getattr(status, 'room_count', 0) or 0)} room(s), "
                f"{int(getattr(status, 'walkable_triangle_count', 0) or 0)} walkable triangle(s), "
                f"{int(getattr(status, 'non_walk_triangle_count', 0) or 0)} blocked triangle(s), "
                f"max slope {float(getattr(status, 'max_slope_degrees', 0.0) or 0.0):.1f} deg."
            )
            self.statusBar().showMessage(message, 8000)
            self._log(f"Map Studio: {message}")
            if not status.ready:
                detail = "\n".join(
                    tuple(getattr(status, "blocking_messages", ()) or ())
                    or (str(getattr(status, "next_action", "Fix the reported WOK issues before export.")),)
                )
                QtWidgets.QMessageBox.warning(self, action, f"{message}\n\n{detail}")
            return
        if action in {"Add Room", "Add Module"}:
            if action == "Add Module":
                module = self.controller.add_module(f"Module{len(self.project.modules) + 1:03d}")
                self.select_item(module.module_id)
            else:
                room = LevelScene(self.project).add_room(f"Room{len(self.project.rooms) + 1:03d}", module_id=self.controller.model.active_module_id)
                self.select_item(room.room_id)
            self._refresh_all(f"{action} complete.")
            return
        if action == "Remove Room":
            self.delete_selected()
            return
        if action == "Duplicate Room":
            self.duplicate_selected()
            return
        if action == "Connect Room Openings":
            self._connect_authored_room_openings()
            return
        if action == "Set Opening Intent":
            self._set_authored_room_opening_intent()
            return
        if action == "Audit Room Connections":
            self._show_authored_room_connection_audit()
            return
        if action == "Auto Arrange":
            self._auto_arrange_authored_rooms()
            return
        if action == "Snap Room to Grid":
            self._snap_selected_authored_rooms_to_grid()
            return
        if action == "Save Layout":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save LYT", f"{self.project.name}.lyt", "LYT files (*.lyt)")
            if path:
                self.controller.layout_service.save_lyt_text(self.project, path)
                self._log(f"Saved layout {path}.")
            return
        if action == "Validate Module":
            self.validate_kmap()
            return
        if action in {"Generate Module Files", "Generate Manifest", "Build ERF/RIM Preview", "Build Loose Override Package"}:
            self.build_module_files()
            return
        if action in {"Port K1 to K2", "Port K2 to K1"}:
            report = self.controller.record_port("K1", "K2") if action.endswith("K2") else self.controller.record_port("K2", "K1")
            self._refresh_all(report.message)
            return
        if action == "Add Blueprint":
            blueprint = self.controller.add_blueprint(blueprint_type=self.blueprints_tab.type_combo.currentText())
            self._refresh_all(f"Added blueprint {blueprint.name}.")
            return
        if action == "Add Camera":
            self.add_map_studio_camera()
            return
        if action == "Add Light":
            self.add_map_studio_room_light()
            return
        if action == "Send to GModular":
            ok, message = self.controller.blueprint_service.send_to_gmodular(None)
            self._log(message if not ok else "Sent blueprint to GModular.")
            return
        if action in {"Focus Selected Room", "Focus in Viewport"}:
            self.viewport_panel.focus_selected()
            return
        self._log(f"{action} is available as an editor hook; backend support is experimental.")

    def _outliner_action(self, action: str, item_id: str) -> None:
        selected_ids = {
            str(value or "")
            for value in tuple(self.controller.model.selected_ids or ())
            if str(value or "")
        }
        preserve_multi_delete = action == "delete" and item_id in selected_ids and len(selected_ids) > 1
        if item_id and not preserve_multi_delete:
            self.select_item(item_id)
        if action == "delete" and self._delete_map_studio_scene_object_selection():
            return
        if item_id == "entry_point" and action == "delete":
            self._delete_map_studio_placement_ids(["entry_point"])
            return
        primitive_identity = self._parse_map_studio_primitive_outliner_id(item_id)
        if primitive_identity is not None:
            if action == "rename":
                self.rename_selected()
                return
            if action == "delete":
                self.delete_map_studio_current_selection()
                return
            if action == "duplicate":
                self._execute_map_studio_tool_belt_command("duplicate_selected")
                return
            if action == "focus_in_viewport":
                self.viewport_panel.focus_selected()
                return
            if action == "validate_selected":
                self.validate_kmap()
                return
        mapping = {
            "add_module": "Add Module",
            "add_room": "Add Room",
            "add_blueprint": "Add Blueprint",
            "add_camera": "Add Camera",
            "add_light": "Add Light",
            "delete": "Remove Room",
            "duplicate": "Duplicate Room",
            "focus_in_viewport": "Focus in Viewport",
            "validate_selected": "Validate Module",
        }
        if action == "rename":
            self.rename_selected()
        else:
            self._handle_tab_action(mapping.get(action, action))

    def _set_transform(self, item_id: str, transform: LevelTransform) -> None:
        if item_id == "entry_point":
            entry = self.controller.authored_module_entry_point()
            if entry is None or not str(getattr(entry, "area_resref", "") or "").strip():
                return
            try:
                self.controller.set_authored_module_entry_point(
                    area_resref=str(getattr(entry, "area_resref", "") or ""),
                    position=transform.position,
                    facing=float(transform.rotation[2]) if len(transform.rotation) >= 3 else float(getattr(entry, "facing", 0.0) or 0.0),
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Move Player Start", str(exc))
                return
            self._refresh_all("Moved Player Start and updated its IFO position/facing.")
            self.select_item("entry_point")
            return
        if item_id.startswith("authored_light:"):
            try:
                self.controller.set_authored_room_light_transform(
                    item_id,
                    position=transform.position,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Move Authored Room Light", str(exc))
                return
            self._refresh_map_studio_gameplay_change(
                light_ids=(item_id,),
                refresh_markers=False,
                refresh_placement_rows=False,
            )
            return
        if item_id.startswith("authored:"):
            try:
                self.controller.set_authored_gameplay_placement_transform(
                    item_id,
                    position=transform.position,
                    bearing=float(transform.rotation[2]) if len(transform.rotation) >= 3 else None,
                    snap_to_walkmesh=bool(self.placement_tab.snap_wok_box.isChecked()),
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Move Authored Gameplay Placement", str(exc))
                return
            self._refresh_map_studio_gameplay_change(placement_ids=(item_id,))
            return
        if LevelScene(self.project).set_transform(item_id, transform):
            self._refresh_all()

    def _translate_map_studio_placements(self, payload: object) -> None:
        values = dict(payload) if isinstance(payload, dict) else {}
        placement_ids = tuple(values.get("placement_ids") or ())
        delta = tuple(values.get("world_delta") or ())
        try:
            moved = self.controller.translate_authored_gameplay_placements(
                placement_ids,
                world_delta=delta,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Move Selected Objects", str(exc))
            self._refresh_all()
            return
        if not moved:
            self._refresh_all()
            return
        self._refresh_map_studio_gameplay_change(
            f"Moved {len(moved)} selected objects together; previous exports/proofs are now stale.",
            placement_ids=moved,
        )
        self.viewport_panel.set_selected_gameplay_placement_ids(moved)
        self.controller.model.select_many(moved)
        self.outliner.select_ids(moved)

    def _preview_authored_room_drag_snap(self, payload: object) -> None:
        """Resolve a live whole-room doorway magnet without changing KMAP."""

        values = dict(payload) if isinstance(payload, dict) else {}
        try:
            preview = self.controller.preview_authored_room_drag_snap(
                source_room_resref=str(values.get("source_room_resref") or ""),
                world_delta=tuple(values.get("world_delta") or ()),
            )
        except Exception:
            preview = None
        self.viewport_panel.set_authored_room_snap_preview(preview)

    def _translate_map_studio_scene_objects(self, payload: object) -> None:
        """Commit one drag spanning terrain/building pieces and placements."""

        values = dict(payload) if isinstance(payload, dict) else {}
        primitive_selections = tuple(values.get("primitive_selections") or ())
        placement_ids = tuple(values.get("placement_ids") or ())
        room_resrefs = tuple(values.get("room_resrefs") or ())
        item_ids = tuple(str(value or "") for value in tuple(values.get("item_ids") or ()) if str(value or ""))
        delta = tuple(values.get("world_delta") or ())
        room_opening_snap = values.get("room_opening_snap")
        if isinstance(room_opening_snap, dict) and bool(room_opening_snap.get("magnet_snapped", False)):
            terrain_seam = str(room_opening_snap.get("snap_kind") or "").strip().lower() == "terrain_seam"
            try:
                update = self.controller.connect_authored_room_drag_snap(room_opening_snap)
            except Exception as exc:
                self.viewport_panel.cancel_pending_room_primitive_commit_preview()
                QtWidgets.QMessageBox.warning(
                    self,
                    "Join Terrain Edges" if terrain_seam else "Snap Rooms at Doorway",
                    str(exc),
                )
                self._refresh_all()
                return
            source = str(getattr(update.source_hook, "room_resref", "") or "")
            target = str(getattr(update.target_hook, "room_resref", "") or "")
            self._refresh_all(
                (
                    f"Joined {source} to {target}; terrain heights and reciprocal WOK traversal are seamless."
                    if terrain_seam
                    else f"Snapped {source} to {target}; openings, shared door, VIS intent, and PIE portal are aligned."
                )
            )
            self._select_map_studio_items(item_ids or ((f"authored_room:{source}",) if source else ()))
            return
        try:
            moved_primitives, moved_placements = self.controller.translate_map_studio_scene_objects(
                primitive_selections=primitive_selections,
                placement_ids=placement_ids,
                room_resrefs=room_resrefs,
                world_delta=delta,
            )
        except Exception as exc:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            QtWidgets.QMessageBox.warning(self, "Move Selected Objects", str(exc))
            self._refresh_all()
            return
        moved_count = len(moved_primitives) + len(moved_placements)
        if not moved_count:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            self._refresh_all()
            return
        self.viewport_panel.promote_room_primitive_drag_preview(moved_primitives)
        self._refresh_all(f"Moved {moved_count} selected scene objects together.")
        self._select_map_studio_items(item_ids)

    def _set_authored_room_outline_point(self, room_resref: str, point_index: int, world_position: object) -> None:
        try:
            self.controller.move_authored_room_outline_point(
                room_resref=room_resref,
                point_index=int(point_index),
                world_position=tuple(world_position),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Move Authored Room Outline Point", str(exc))
            return
        self._refresh_all()

    def _select_authored_room_primitive(self, room_resref: str, primitive_name: str) -> None:
        self._select_authored_room_primitives(((room_resref, primitive_name),))

    def _selected_authored_room_primitives(self) -> list[tuple[str, str]]:
        return [
            identity
            for identity in (
                self._parse_map_studio_primitive_outliner_id(item_id)
                for item_id in tuple(self.controller.model.selected_ids or ())
            )
            if identity is not None
        ]

    def _combine_selected_authored_room_primitives(self) -> bool:
        selected = self._selected_authored_room_primitives()
        if len(selected) < 2:
            return False
        rooms = {room for room, _name in selected}
        if len(rooms) != 1:
            self.statusBar().showMessage(
                "Combine needs objects in the same authored room. Separate or move them into one room first.", 6000
            )
            return True
        room_resref = selected[0][0]
        names = [name for _room, name in selected]
        try:
            self.controller.combine_authored_room_primitives(
                room_resref=room_resref,
                primitive_names=names,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Combine Meshes", str(exc))
            return True
        combined_rows = [
            row
            for row in self.controller.authored_room_primitive_transforms()
            if str(getattr(row, "room_resref", "") or "") == room_resref
            and str(getattr(row, "primitive_type", "") or "") == "combined_mesh"
        ]
        combined_selection = (
            ((room_resref, str(combined_rows[-1].primitive_name)),)
            if combined_rows
            else ()
        )
        self._refresh_map_studio_geometry_change(
            f"Combined {len(names)} objects into one real polygon mesh; materials, UVs, normals, and disconnected shells were preserved.",
            primitive_selection=combined_selection,
        )
        return True

    def _separate_selected_authored_room_primitive(self) -> bool:
        selected = self._selected_authored_room_primitives()
        if len(selected) != 1:
            return False
        room_resref, primitive_name = selected[0]
        before_names = {
            str(getattr(row, "primitive_name", "") or "")
            for row in self.controller.authored_room_primitive_transforms()
            if str(getattr(row, "room_resref", "") or "") == room_resref
        }
        try:
            self.controller.separate_authored_room_primitive_shells(
                room_resref=room_resref,
                primitive_name=primitive_name,
                name_prefix="",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Separate Polygon Shells", str(exc))
            return True
        shell_rows = [
            row
            for row in self.controller.authored_room_primitive_transforms()
            if str(getattr(row, "room_resref", "") or "") == room_resref
            and str(getattr(row, "primitive_name", "") or "") not in before_names
        ]
        shell_selection = tuple((room_resref, str(row.primitive_name)) for row in shell_rows)
        self._refresh_map_studio_geometry_change(
            f"Separated {primitive_name} into {len(shell_rows)} connected polygon shell object(s) in the same room.",
            primitive_selection=shell_selection,
        )
        return True

    def _select_authored_room_primitives(self, entries: object) -> None:
        """Select one or more authored objects from viewport Shift-clicks."""

        selected: list[tuple[str, str]] = []
        for room_resref, primitive_name in tuple(entries or ()):
            key = (str(room_resref or "").strip(), str(primitive_name or "").strip())
            if key[0] and key[1] and key not in selected:
                selected.append(key)
        if not selected:
            self.controller.model.select_many(())
            self.outliner.select_ids(())
            self.viewport_panel.set_selected_room_primitives(())
            return
        room_resref, primitive_name = selected[-1]
        if not self.builder_tab.select_room_primitive(room_resref, primitive_name):
            return
        item_ids = [self._map_studio_primitive_outliner_id(room, name) for room, name in selected]
        self.controller.model.select_many(item_ids)
        self.outliner.select_ids(item_ids)
        self.viewport_panel.set_map_studio_scene_selection_ids(item_ids)
        self.viewport_panel.set_selected_room_primitives(selected)
        self.properties.set_selection(item_ids[-1])
        self.workflow_tabs.setCurrentWidget(self.builder_tab)
        self._refresh_map_studio_selected_primitive_transform_overlay()
        mode = self.viewport_panel.transform_gizmo_mode()
        if len(selected) == 1:
            context = self._selected_item_label(item_ids[-1])
            message = f"Selected room primitive {primitive_name}; {mode.title()} gimbal ready."
        else:
            context = f"{len(selected)} authored objects selected"
            message = (
                f"Selected {len(selected)} authored objects; {mode.title()}, Combine Meshes, and Separate Shells are ready."
            )
        self.workflow_panel.set_selection_context(context)
        self.statusBar().showMessage(message)

    def _move_authored_room_primitive(self, room_resref: str, primitive_name: str, world_delta: object) -> None:
        selection = ((room_resref, primitive_name),)
        try:
            self.controller.move_authored_room_primitive(
                room_resref=room_resref,
                primitive_name=primitive_name,
                world_delta=tuple(world_delta),
            )
        except Exception as exc:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            QtWidgets.QMessageBox.warning(self, "Move Authored Room Primitive", str(exc))
            return
        promoted = self.viewport_panel.promote_room_primitive_drag_preview(selection)
        self._refresh_map_studio_geometry_change(
            f"Moved room primitive {primitive_name}; previous exports/proofs are now stale.",
            primitive_selection=selection,
            rebuild_viewport_model=not promoted,
            refresh_scene_tree=not promoted,
        )
        if promoted:
            self._refresh_map_studio_selected_primitive_transform_overlay()

    def _authored_room_primitive_row(self, room_resref: str, primitive_name: str):
        room = str(room_resref or "").strip()
        name = str(primitive_name or "").strip()
        for row in self.controller.authored_room_primitive_transforms():
            if str(getattr(row, "primitive_name", "") or "") != name:
                continue
            if room and str(getattr(row, "room_resref", "") or "") != room:
                continue
            return row
        return None

    def _rotate_authored_room_primitive(self, room_resref: str, primitive_name: str, delta_degrees: float) -> None:
        row = self._authored_room_primitive_row(room_resref, primitive_name)
        if row is None:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            return
        selection = ((room_resref, primitive_name),)
        current = float(getattr(row, "rotation_degrees_z", 0.0) or 0.0)
        try:
            self.controller.set_authored_room_primitive_transform(
                room_resref=room_resref,
                primitive_name=primitive_name,
                rotation_degrees_z=current + float(delta_degrees),
            )
        except Exception as exc:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            QtWidgets.QMessageBox.warning(self, "Rotate Authored Room Primitive", str(exc))
            return
        promoted = self.viewport_panel.promote_room_primitive_drag_preview(selection)
        self._refresh_map_studio_geometry_change(
            f"Rotated room primitive {primitive_name}; previous exports/proofs are now stale.",
            primitive_selection=selection,
            rebuild_viewport_model=not promoted,
            refresh_scene_tree=not promoted,
        )
        if promoted:
            self._refresh_map_studio_selected_primitive_transform_overlay()

    def _scale_authored_room_primitive(self, room_resref: str, primitive_name: str, scale_multiplier: object) -> None:
        row = self._authored_room_primitive_row(room_resref, primitive_name)
        if row is None:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            return
        selection = ((room_resref, primitive_name),)
        current = tuple(float(value) for value in tuple(getattr(row, "scale", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0))[:3])
        if len(current) != 3:
            current = (1.0, 1.0, 1.0)
        multiplier = tuple(float(value) for value in tuple(scale_multiplier or (1.0, 1.0, 1.0))[:3])
        if len(multiplier) != 3:
            multiplier = (1.0, 1.0, 1.0)
        updated = tuple(max(0.01, current[index] * multiplier[index]) for index in range(3))
        try:
            self.controller.set_authored_room_primitive_transform(
                room_resref=room_resref,
                primitive_name=primitive_name,
                scale=updated,
            )
        except Exception as exc:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            QtWidgets.QMessageBox.warning(self, "Scale Authored Room Primitive", str(exc))
            return
        promoted = self.viewport_panel.promote_room_primitive_drag_preview(selection)
        self._refresh_map_studio_geometry_change(
            f"Scaled room primitive {primitive_name}; previous exports/proofs are now stale.",
            primitive_selection=selection,
            rebuild_viewport_model=not promoted,
            refresh_scene_tree=not promoted,
        )
        if promoted:
            self._refresh_map_studio_selected_primitive_transform_overlay()

    def _transform_authored_room_primitives(self, payload: object) -> None:
        data = dict(payload or {})
        selection = tuple(
            (str(room or "").strip(), str(name or "").strip())
            for room, name in tuple(data.get("selection", ()) or ())
            if str(room or "").strip() and str(name or "").strip()
        )
        if not selection:
            return
        mode = str(data.get("mode") or "translate").strip().lower()
        try:
            self.controller.transform_authored_room_primitives(
                selections=selection,
                mode=mode,
                world_delta=tuple(data.get("world_delta", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)),
                rotation_delta_degrees_z=float(data.get("rotation_delta_degrees", 0.0) or 0.0),
                scale_multiplier=tuple(data.get("scale_multiplier", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)),
                world_pivot=tuple(data.get("world_pivot", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)),
            )
        except Exception as exc:
            self.viewport_panel.cancel_pending_room_primitive_commit_preview()
            QtWidgets.QMessageBox.warning(self, "Transform Authored Room Primitives", str(exc))
            return
        promoted = self.viewport_panel.promote_room_primitive_drag_preview(selection)
        verb = {"translate": "Moved", "rotate": "Rotated", "scale": "Scaled"}.get(mode, "Transformed")
        message = f"{verb} {len(selection)} room primitives as one selection; previous exports/proofs are now stale."
        self._refresh_map_studio_geometry_change(
            message,
            primitive_selection=selection,
            rebuild_viewport_model=not promoted,
            refresh_scene_tree=not promoted,
        )
        if promoted:
            self._refresh_map_studio_selected_primitive_transform_overlay()

    def _current_map_studio_room_primitive_identity(self) -> tuple[str, str] | None:
        data = self._map_studio_combo_data("roomPrimitiveTransformComboBox")
        primitive_name = str(data.get("primitive_name") or "").strip()
        if not primitive_name:
            return None
        return (str(data.get("room_resref") or "").strip(), primitive_name)

    def _refresh_map_studio_selected_primitive_transform_overlay(self) -> None:
        identity = self._current_map_studio_room_primitive_identity()
        if identity is None:
            self.viewport_panel.set_universal_transform_overlay(None)
            return
        room_resref, primitive_name = identity
        try:
            overlay = self.controller.map_studio_universal_transform_overlay(
                room_resref=room_resref,
                primitive_name=primitive_name,
            )
        except Exception:
            self.viewport_panel.set_universal_transform_overlay(None)
            return
        self.viewport_panel.set_universal_transform_overlay(overlay)

    def _handle_map_studio_transform_gizmo_mode_changed(self, mode_key: str) -> None:
        label = {"translate": "Translate", "rotate": "Rotate", "scale": "Scale"}.get(
            str(mode_key or "").lower(),
            "Translate",
        )
        self._set_map_studio_toolbar_edit_mode("Object")
        self._select_map_studio_component_mode("object")
        self._select_map_studio_modeling_tool("universal_transform")
        self._refresh_map_studio_selected_primitive_transform_overlay()
        self.workflow_panel.set_active_authoring_context(
            f"{label} gimbal: selected authored primitives edit durable KMAP transform state; "
            "commits mark MDL/MDX/WOK/LYT/VIS/PTH/export proof stale."
        )
        self.statusBar().showMessage(f"Map Studio {label} gimbal active. Shortcuts: W translate, E rotate, R scale.", 5000)

    def _set_visibility(self, item_id: str, visible: bool) -> None:
        if item_id.startswith("authored:"):
            return
        if LevelScene(self.project).set_visibility(item_id, visible):
            self._refresh_all()

    def _set_locked(self, item_id: str, locked: bool) -> None:
        if item_id.startswith("authored:"):
            return
        if LevelScene(self.project).set_locked(item_id, locked):
            self._refresh_all()

    def _set_property(self, item_id: str, key: str, value: Any) -> None:
        if item_id.startswith("authored:"):
            if key == "name":
                try:
                    self.controller.rename_authored_gameplay_placement(item_id, tag=str(value or "").strip())
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Rename Authored Placement", str(exc))
                    return
                self._refresh_map_studio_gameplay_change(
                    "Renamed authored gameplay placement.",
                    placement_ids=(item_id,),
                    refresh_outliner_labels=True,
                )
            return
        if item_id.startswith("authored_light:"):
            if key == "name":
                try:
                    self.controller.rename_authored_room_light(item_id, name=str(value or "").strip())
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Rename Authored Room Light", str(exc))
                    return
                self._refresh_map_studio_gameplay_change(
                    "Renamed authored room light.",
                    light_ids=(item_id,),
                    refresh_markers=False,
                    refresh_placement_rows=False,
                    refresh_outliner_labels=True,
                )
            return
        item = self.project.find_room(item_id) or self.project.find_module(item_id) or self.project.find_blueprint(item_id)
        if item is None:
            return
        if key == "name" and hasattr(item, "module_name"):
            item.module_name = str(value)
        elif hasattr(item, key):
            setattr(item, key, value)
        self.project.mark_dirty()
        self._refresh_all()

    def _set_authored_gameplay_transition(
        self,
        item_id: str,
        linked_to: str,
        linked_to_module: str,
        linked_to_flags: int,
        transition_destination: int,
    ) -> None:
        if not item_id.startswith("authored:"):
            return
        try:
            self.controller.set_authored_gameplay_transition(
                item_id,
                linked_to=linked_to,
                linked_to_module=linked_to_module,
                linked_to_flags=linked_to_flags,
                transition_destination=transition_destination,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Edit Authored Transition", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            "Updated authored transition destination.",
            placement_ids=(item_id,),
        )

    def _set_authored_gameplay_camera_properties(
        self,
        item_id: str,
        camera_id: int,
        field_of_view: float,
        height: float,
        mic_range: float,
        pitch: float,
    ) -> None:
        if not item_id.startswith("authored:camera:"):
            return
        try:
            self.controller.set_authored_gameplay_camera_properties(
                item_id,
                camera_id=camera_id,
                field_of_view=field_of_view,
                height=height,
                mic_range=mic_range,
                pitch=pitch,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Edit Authored Camera", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            "Updated authored camera properties.",
            placement_ids=(item_id,),
        )

    def _set_authored_room_light_properties(self, item_id: str, settings: object) -> None:
        if not item_id.startswith("authored_light:"):
            return
        try:
            values = dict(settings) if isinstance(settings, dict) else {}
            self.controller.set_authored_room_light_properties(
                item_id,
                light_type=values.get("light_type"),
                color=values.get("color"),
                radius=values.get("radius"),
                intensity=values.get("intensity"),
                enabled=values.get("enabled"),
                casts_shadows=values.get("casts_shadows"),
                affects_diffuse=values.get("affects_diffuse"),
                affects_lightmap=values.get("affects_lightmap"),
                direction=values.get("direction"),
                cone_angle_degrees=values.get("cone_angle_degrees"),
                bake_group=values.get("bake_group"),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Edit Authored Room Light", str(exc))
            return
        self._refresh_map_studio_gameplay_change(
            "Updated authored room light properties.",
            light_ids=(item_id,),
            refresh_markers=False,
            refresh_placement_rows=False,
        )

    def _sync_map_studio_viewport_resource_manager(self) -> None:
        """Push stock plus active-project resources into the viewport cache."""

        manager = getattr(self, "resource_manager", None)
        if manager is None:
            return
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        publish_project = getattr(manager, "set_project_overlay", None)
        if callable(publish_project):
            try:
                publish_project(self.controller.authored_project_extra_resources())
            except Exception as exc:
                self._log(f"Map Studio project texture preview warning: {exc}")
        key = (id(manager), game, int(getattr(manager, "revision", 0) or 0))
        if getattr(self, "_map_studio_viewport_resource_key", None) == key:
            return
        viewport = getattr(self.viewport_panel, "viewport", None)
        set_resource_manager = getattr(viewport, "set_resource_manager", None)
        if callable(set_resource_manager):
            try:
                set_resource_manager(manager, game)
                self._map_studio_viewport_resource_key = key
            except Exception:
                pass

    def _log_map_studio_stock_preview_warnings(self) -> None:
        warnings = tuple(getattr(self.controller, "last_map_studio_stock_preview_warnings", ()) or ())
        if warnings and warnings != getattr(self, "_map_studio_last_stock_warnings", ()):
            self._map_studio_last_stock_warnings = warnings
            for warning in warnings[:8]:
                self._log(f"Map Studio stock preview: {warning}")

    def _apply_map_studio_minimal_layout(self) -> None:
        """T3001 UI cleanup: keep only map-building essentials visible.

        Duplicated or verbose strips stay constructed (automation, tests, and
        dispatch contracts depend on them) but are hidden: the scope banner,
        the Workspace row, the command-readiness banner, and the tool-belt
        tab chrome.  The toolbar edit-mode combo remains the single visible
        workspace switcher; hidden controls keep working programmatically.
        """

        for widget in (
            self.map_studio_scope_label,
            self.map_studio_workspace_label,
            self.map_studio_workspace_combo,
            self.map_studio_workspace_guide_label,
            self.map_studio_open_workspace_button,
            self.map_studio_command_search_readiness_label,
        ):
            widget.setVisible(False)
        tab_bar = self.map_studio_tool_belt_tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(False)
        try:
            self.map_studio_tool_belt_tabs.setTabVisible(1, False)
        except Exception:
            pass

    def _refresh_map_studio_geometry_validation(self, generation: int) -> None:
        """Refresh export gates after the viewport has had a chance to paint."""

        if generation != int(getattr(self, "_map_studio_geometry_refresh_generation", 0) or 0):
            return
        panel = getattr(self, "viewport_panel", None)
        interaction_fields = (
            "_component_extrude_drag",
            "_component_bevel_drag",
            "_terrain_brush_drag",
            "_terrain_brush_option_drag",
            "_room_primitive_drag",
            "_room_outline_point_drag",
            "_marker_drag",
        )
        if panel is not None and any(getattr(panel, field, None) is not None for field in interaction_fields):
            # Never let full readiness/WOK validation interrupt a live pointer
            # gesture.  The same generation retries after the interaction;
            # a newer commit supersedes this one through the guard above.
            QtCore.QTimer.singleShot(
                100,
                lambda value=generation: self._refresh_map_studio_geometry_validation(value),
            )
            return
        self._map_studio_geometry_validation_requested_generation = generation
        running = getattr(self, "_map_studio_geometry_validation_future", None)
        if running is not None and not running.done():
            # One worker is enough.  Its poller discards the stale result and
            # immediately starts the newest requested generation.
            return
        try:
            project_data = deepcopy(KMapSerializer.to_dict(self.project))
            future = _MAP_STUDIO_VALIDATION_EXECUTOR.submit(
                _build_map_studio_geometry_validation_snapshot,
                project_data,
            )
            self._map_studio_geometry_validation_future = future
            self._map_studio_geometry_validation_future_generation = generation
            QtCore.QTimer.singleShot(
                25,
                lambda value=generation: self._poll_map_studio_geometry_validation(value),
            )
        except Exception as exc:
            self._log(f"Map Studio readiness worker could not start: {exc}")

    def _poll_map_studio_geometry_validation(self, generation: int) -> None:
        """Apply a validation worker result on Qt's thread, or coalesce it."""

        if generation != int(getattr(self, "_map_studio_geometry_validation_future_generation", -1)):
            return
        future = getattr(self, "_map_studio_geometry_validation_future", None)
        if future is None:
            return
        if not future.done():
            QtCore.QTimer.singleShot(
                25,
                lambda value=generation: self._poll_map_studio_geometry_validation(value),
            )
            return

        self._map_studio_geometry_validation_future = None
        current_generation = int(getattr(self, "_map_studio_geometry_refresh_generation", 0) or 0)
        if generation != current_generation:
            requested_generation = int(
                getattr(self, "_map_studio_geometry_validation_requested_generation", -1)
            )
            if requested_generation == current_generation:
                self._refresh_map_studio_geometry_validation(current_generation)
            return
        try:
            result = dict(future.result() or {})
            readiness = result["readiness"]
            self.workflow_panel.set_state(self.project, readiness)
            self.readiness_panel.set_readiness(readiness)
            self.export_panel.set_readiness(readiness)
            self.validation_panel.set_issues(result["issues"])
            self.walkmesh_tab.set_walkmesh_status(result["walkmesh_status"])
            self.walkmesh_tab.set_room_surface_choices(result["walkmesh_room_surfaces"])
            terrain_overlay = result.get("terrain_walkability_overlay")
            self.viewport_panel.set_terrain_walkability_overlay(terrain_overlay)
            terrain_room_choices = result.get("terrain_room_choices")
            if terrain_room_choices is not None:
                self.builder_tab.set_terrain_room_choices(terrain_room_choices)
                self._sync_map_studio_terrain_brush_context()
            self._last_map_studio_geometry_validation_ms = float(result.get("elapsed_ms", 0.0) or 0.0)
        except Exception as exc:
            self._log(f"Map Studio deferred readiness refresh failed: {exc}")

    def _refresh_map_studio_geometry_change(
        self,
        message: str = "",
        *,
        primitive_selection=(),
        refresh_outlines: bool = False,
        refresh_terrain: bool = False,
        refresh_room_choices: bool = False,
        refresh_connections: bool = False,
        rebuild_viewport_model: bool = True,
        refresh_scene_tree: bool = True,
        validation_delay_ms: int = 75,
    ) -> None:
        """Refresh only the editor surfaces made stale by a geometry commit.

        ``rebuild_viewport_model=False`` is the topology hot path: the final
        live preview was already promoted in-place, so replacing the combined
        model would only evict unrelated resident resources.  The full path is
        retained as a correctness fallback for structural room/node changes.
        """

        started = perf_counter()
        self.setWindowTitle(f"Ghost-Studio Map Studio - Level Editor - {self.project.name}{' *' if self.project.dirty else ''}")
        self._update_map_studio_undo_redo_actions()
        if rebuild_viewport_model:
            self._cancel_map_studio_component_preview()
            preview = self.controller.map_studio_viewport_preview_model(
                resource_manager=getattr(self, "resource_manager", None),
                include_backdrops=bool(getattr(self, "_map_studio_show_skybox", False)),
            )
            self.viewport_panel.set_authored_room_preview_model(preview)
        else:
            self._map_studio_topology_preview_source = None
            self._map_studio_prepared_topology_preview = None
        if refresh_outlines:
            room_outlines = self.controller.authored_room_outline_geometry()
            self.viewport_panel.set_authored_room_outline_geometry(room_outlines)
        if refresh_terrain:
            self.viewport_panel.set_terrain_walkability_overlay(
                self.controller.authored_terrain_walkability_overlay()
            )
        if refresh_scene_tree:
            primitive_rows = self.controller.authored_room_primitive_transforms()
            self.builder_tab.set_room_primitives(primitive_rows)
            if refresh_room_choices:
                self.builder_tab.set_floor_plan_room_choices(self.controller.authored_floor_plan_room_choices())
                self.builder_tab.set_terrain_room_choices(self.controller.authored_terrain_room_choices())
            if refresh_connections:
                self.rooms_tab.set_connection_audit(self.controller.authored_room_connection_audit())
            self.outliner.set_project(
                self.project,
                tuple(
                    row
                    for row in (
                        self.controller.authored_module_entry_point_preview_row(),
                        *tuple(self.controller.authored_gameplay_placements() or ()),
                    )
                    if row is not None
                ),
                self.controller.authored_room_lights(),
                primitive_rows,
                self.controller.authored_room_resrefs(),
            )
            selection = tuple(primitive_selection or self.viewport_panel.selected_room_primitives())
            if selection:
                self._select_authored_room_primitives(selection)
            else:
                self._refresh_map_studio_selected_primitive_transform_overlay()
        generation = int(getattr(self, "_map_studio_geometry_refresh_generation", 0) or 0) + 1
        self._map_studio_geometry_refresh_generation = generation
        self._map_studio_geometry_validation_requested_generation = generation
        # Coalesce readiness work behind the first committed render.  A zero
        # delay competed with the renderer for the same event-loop turn and
        # made the final mouse release feel like another drag hitch.
        QtCore.QTimer.singleShot(
            max(0, int(validation_delay_ms)),
            lambda value=generation: self._refresh_map_studio_geometry_validation(value),
        )
        self._last_map_studio_geometry_refresh_ms = (perf_counter() - started) * 1000.0
        if message:
            self._log(message)

    def _refresh_map_studio_gameplay_change(
        self,
        message: str = "",
        *,
        placement_ids=(),
        light_ids=(),
        refresh_markers: bool = True,
        refresh_placement_rows: bool = True,
        refresh_outliner_labels: bool = False,
        validation_delay_ms: int = 75,
    ) -> None:
        """Refresh only the surfaces made stale by a GIT placement/light commit.

        The rendered proxy is promoted in place instead of rebuilding the
        combined preview model, mirroring the topology hot path in
        ``_refresh_map_studio_geometry_change``.  Export readiness and
        validation reuse the deferred geometry generation contract.  Scene
        selection (including multi-selection) is intentionally left
        untouched; placement add/remove and undo/redo still route through
        the broad ``_refresh_all`` correctness fallback.
        """

        started = perf_counter()
        self.setWindowTitle(f"Ghost-Studio Map Studio - Level Editor - {self.project.name}{' *' if self.project.dirty else ''}")
        self._update_map_studio_undo_redo_actions()
        placements = tuple(self.controller.authored_gameplay_placements() or ())
        lights = tuple(self.controller.authored_room_lights() or ())
        placement_rows = {str(getattr(row, "placement_id", "") or ""): row for row in placements}
        edited_placements = tuple(str(value or "") for value in tuple(placement_ids or ()) if str(value or ""))
        edited_lights = tuple(str(value or "") for value in tuple(light_ids or ()) if str(value or ""))
        # Promote committed transforms onto live preview nodes before the
        # marker table refresh so the one-time baked-bearing capture still
        # sees the pre-commit marker state.
        for placement_id in edited_placements:
            row = placement_rows.get(placement_id)
            if row is None or not bool(getattr(row, "is_spatial", True)):
                continue
            self.viewport_panel.update_authored_placement_preview_transform(
                placement_id,
                position=getattr(row, "position", None),
                bearing=getattr(row, "bearing", None),
            )
        if refresh_markers:
            self.viewport_panel.set_authored_gameplay_markers(
                placements,
                self.controller.authored_gameplay_fallback_preview_markers(),
                self.controller.authored_editor_marker_geometry(
                    include_spatial_design=self.builder_tab.spatial_plan_overlay_enabled()
                ),
            )
        self.viewport_panel.update_authored_scene_rows(
            placements,
            lights,
            item_ids=edited_placements + edited_lights,
        )
        if refresh_placement_rows:
            self.placement_tab.set_placements(placements)
        self.properties.set_project(self.project, placements, lights)
        shown = self.properties.current_item_id()
        if shown and (shown in edited_placements or shown in edited_lights):
            self.properties.set_selection(shown)
        if refresh_outliner_labels:
            for item_id in (*edited_placements, *edited_lights):
                row = placement_rows.get(item_id)
                if row is not None:
                    label = str(getattr(row, "tag", "") or getattr(row, "template_resref", "") or item_id)
                else:
                    light = next((value for value in lights if str(getattr(value, "light_id", "") or "") == item_id), None)
                    if light is None:
                        continue
                    label = str(getattr(light, "name", "") or item_id)
                self.outliner.update_item_text(item_id, label)
        generation = int(getattr(self, "_map_studio_geometry_refresh_generation", 0) or 0) + 1
        self._map_studio_geometry_refresh_generation = generation
        self._map_studio_geometry_validation_requested_generation = generation
        QtCore.QTimer.singleShot(
            max(0, int(validation_delay_ms)),
            lambda value=generation: self._refresh_map_studio_geometry_validation(value),
        )
        self._last_map_studio_gameplay_refresh_ms = (perf_counter() - started) * 1000.0
        if message:
            self._log(message)

    def _refresh_all(self, message: str = "") -> None:
        if self._map_studio_pie_session is not None:
            self._stop_map_studio_pie()
        # A broad synchronous refresh supersedes any older background
        # geometry-readiness snapshot (for example when Undo replaces KMAP).
        self._map_studio_geometry_refresh_generation = int(
            getattr(self, "_map_studio_geometry_refresh_generation", 0) or 0
        ) + 1
        self._map_studio_geometry_validation_requested_generation = -1
        project_game = str(getattr(self.project, "game", "") or "").upper()
        project_root = self._authored_module_root()
        if (
            project_game != str(getattr(self, "_placeable_library_game", "") or "").upper()
            or project_root != str(getattr(self, "_placeable_library_module_root", "") or "").lower()
        ):
            self.refresh_placeable_library()
        primitive_selection = tuple(self.viewport_panel.selected_room_primitives())
        self.setWindowTitle(f"Ghost-Studio Map Studio - Level Editor - {self.project.name}{' *' if self.project.dirty else ''}")
        self._sync_map_studio_viewport_resource_manager()
        # Warm the PIE player model cache in the background so the first Play
        # press does not block on the body/head share of MDL loading. Global
        # supermodel resolution intentionally remains on the GUI-thread path.
        # Deferred: the pure-Python MDL parse contends for the GIL, so it must
        # not overlap this refresh's own scene load.
        QtCore.QTimer.singleShot(1500, self._prewarm_map_studio_pie_player_model)
        self._update_map_studio_undo_redo_actions()
        self._apply_map_studio_tool_belt_preferences_from_project()
        self._refresh_map_studio_tool_belt()
        authored_placements = self.controller.authored_gameplay_placements()
        authored_entry_point = self.controller.authored_module_entry_point_preview_row()
        authored_editor_placements = (
            ((authored_entry_point,) if authored_entry_point is not None else ())
            + tuple(authored_placements or ())
        )
        authored_room_lights = self.controller.authored_room_lights()
        authored_room_preview_model = self.controller.map_studio_viewport_preview_model(
            resource_manager=getattr(self, "resource_manager", None),
            include_backdrops=bool(getattr(self, "_map_studio_show_skybox", False)),
        )
        authored_markers = self.controller.authored_gameplay_fallback_preview_markers()
        authored_marker_geometry = self.controller.authored_editor_marker_geometry(
            include_spatial_design=self.builder_tab.spatial_plan_overlay_enabled()
        )
        authored_room_outline_geometry = self.controller.authored_room_outline_geometry()
        self._log_map_studio_stock_preview_warnings()
        authored_terrain_walkability_overlay = self.controller.authored_terrain_walkability_overlay()
        authored_walkmesh_status = self.controller.authored_walkmesh_status()
        authored_walkmesh_room_surfaces = self.controller.authored_walkmesh_room_surface_choices()
        authored_room_primitives = self.controller.authored_room_primitive_transforms()
        authored_floor_plan_rooms = self.controller.authored_floor_plan_room_choices()
        authored_terrain_rooms = self.controller.authored_terrain_room_choices()
        authored_room_connection_audit = self.controller.authored_room_connection_audit()
        self.rooms_tab.set_connection_audit(authored_room_connection_audit)
        self.placement_tab.set_placements(authored_placements)
        self.texture_paint_tab.set_project(self.project)
        self.environment_tab.set_world_settings(self.controller.authored_world_lighting_settings())
        self.environment_tab.set_lightmap_context(
            self.controller.authored_lightmap_surface_rows(),
            project_saved=bool(str(getattr(self.project, "path", "") or "").strip()),
            light_count=len(tuple(authored_room_lights or ())),
        )
        self.environment_tab.set_skybox_context(
            module_root=str(getattr(self.project, "name", "") or ""),
            game=str(getattr(self.project, "game", "") or "K1"),
            room_resrefs=self.controller.authored_room_resrefs(),
        )
        self.environment_tab.set_sky_traffic_context(
            room_resrefs=self.controller.authored_room_resrefs(),
            traffic_count=len(tuple(self.controller.authored_sky_traffic() or ())),
        )
        if self._texture_paint_texture_id and self._project_texture_for_id(self._texture_paint_texture_id) is None:
            self._reset_map_studio_texture_paint_session()
        self.builder_tab.set_module_entry_point(self.controller.authored_module_entry_point())
        self.builder_tab.set_room_primitives(authored_room_primitives)
        self.builder_tab.set_floor_plan_room_choices(authored_floor_plan_rooms)
        self.builder_tab.set_terrain_room_choices(authored_terrain_rooms)
        self._sync_map_studio_kit_browsers_for_project_game()
        self.builder_tab.set_building_styles(self.controller.available_map_studio_building_styles())
        building_style_context = self.controller.map_studio_building_style_context()
        if building_style_context:
            self.builder_tab.select_building_style(
                str(building_style_context.get("style_id") or ""),
                str(building_style_context.get("environment_kind") or ""),
            )
        self.builder_tab.set_building_levels(self.controller.map_studio_building_levels())
        self.builder_tab.set_spatial_design_context(
            self.controller.audit_map_studio_spatial_design(),
            self.controller.map_studio_spatial_design_placement_ledger(),
        )
        self.builder_tab.set_script_hooks(self.controller.authored_script_hooks())
        self.walkmesh_tab.set_walkmesh_status(authored_walkmesh_status)
        self.walkmesh_tab.set_room_surface_choices(authored_walkmesh_room_surfaces)
        self.properties.set_project(self.project, authored_placements, authored_room_lights)
        self._refresh_map_studio_pie_context_panel()
        self.outliner.set_project(
            self.project,
            authored_editor_placements,
            authored_room_lights,
            authored_room_primitives,
            self.controller.authored_room_resrefs(),
        )
        self.viewport_panel.set_project(
            self.project,
            authored_editor_placements,
            authored_room_lights,
            authored_markers,
            authored_marker_geometry,
            authored_room_outline_geometry,
            authored_terrain_walkability_overlay,
            authored_room_preview_model,
        )
        self._apply_map_studio_level_presentation()
        self.texture_paint_tab.set_material_inventory(
            self._used_map_diffuse_resrefs(authored_room_preview_model),
            self.project,
        )
        self._sync_map_studio_terrain_brush_context()
        self._refresh_map_studio_selected_primitive_transform_overlay()
        readiness_result = self.controller.authored_module_readiness()
        self.export_panel.set_target_game(
            str(getattr(self.project, "game", "K1") or "K1"),
            source_game=str(getattr(self.project, "source_game", "") or ""),
        )
        self.workflow_panel.set_state(self.project, readiness_result.readiness)
        self.readiness_panel.set_readiness(readiness_result.readiness)
        self.export_panel.set_readiness(readiness_result.readiness)
        self.validation_panel.set_issues(
            self.controller.validate(readiness_result=readiness_result)
        )
        if primitive_selection:
            self._select_authored_room_primitives(primitive_selection)
        elif self.controller.model.selected_ids:
            # A general single selection can still use the normal inspector
            # route; authored multi-object selections are restored above and
            # are never collapsed to selected_ids[0].
            self.select_item(self.controller.model.selected_ids[-1])
        else:
            self.workflow_panel.set_selection_context("")
        if message:
            self._log(message)

    def _log(self, message: str) -> None:
        if not message:
            return
        self.output_log.appendPlainText(message)
        self.statusBar().showMessage(message)

    def _confirm_discard_or_save(self, *, restore_discarded_sidecars: bool = False) -> bool:
        if not self.project.dirty:
            return True
        result = QtWidgets.QMessageBox.question(
            self,
            "Unsaved KMAP",
            "Save changes before continuing?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
        )
        if result == QtWidgets.QMessageBox.Cancel:
            return False
        if result == QtWidgets.QMessageBox.Save:
            self.save_kmap()
            return not self.project.dirty
        if restore_discarded_sidecars:
            try:
                self.controller.discard_project_texture_sidecar_changes()
                self._reset_map_studio_texture_paint_session()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Discard Texture Changes", str(exc))
                return False
        return True

    def _set_map_studio_viewport_maximized(self, maximized: bool) -> None:
        """Toggle a reversible viewport-first Map Studio layout."""

        splitter = getattr(self, "main_splitter", None)
        left = getattr(self, "left_authoring_rail", None)
        right = getattr(self, "right_inspector_rail", None)
        if splitter is None or left is None or right is None:
            return
        chrome = tuple(
            widget
            for widget in (
                getattr(self, "toolbar_scroll", None),
                getattr(self, "map_studio_tool_belt_tabs", None),
            )
            if widget is not None
        )
        docks = tuple(
            dock
            for dock in (
                getattr(self, "workflow_dock", None),
                getattr(self, "validation_output_dock", None),
            )
            if dock is not None
        )
        if maximized:
            if self._map_studio_viewport_focus_state is None:
                self._map_studio_viewport_focus_state = {
                    "splitter_sizes": list(splitter.sizes()),
                    "left_visible": not left.isHidden(),
                    "right_visible": not right.isHidden(),
                    "chrome_visible": tuple(not widget.isHidden() for widget in chrome),
                    "dock_visible": tuple(not dock.isHidden() for dock in docks),
                }
            left.hide()
            right.hide()
            for widget in chrome:
                widget.hide()
            for dock in docks:
                dock.hide()
            splitter.setSizes([0, max(1, self.width()), 0])
            self.statusBar().showMessage(
                "Viewport maximized. Press Ctrl+Space or clear View > Maximize Viewport to restore panels.",
                6000,
            )
            return

        state = self._map_studio_viewport_focus_state or {}
        left.setVisible(bool(state.get("left_visible", True)))
        right.setVisible(bool(state.get("right_visible", True)))
        for widget, visible in zip(chrome, state.get("chrome_visible", (True,) * len(chrome))):
            widget.setVisible(bool(visible))
        for dock, visible in zip(docks, state.get("dock_visible", (True,) * len(docks))):
            dock.setVisible(bool(visible))
        sizes = [int(size) for size in state.get("splitter_sizes", ()) if int(size) >= 0]
        if len(sizes) != 3 or sum(sizes) <= 0:
            sizes = [280, 1080, 300]
        splitter.setSizes(sizes)
        self._map_studio_viewport_focus_state = None
        self.statusBar().showMessage("Map Studio panels restored.", 3000)

    def _reset_layout(self) -> None:
        if getattr(self, "focus_viewport_action", None) is not None and self.focus_viewport_action.isChecked():
            self.focus_viewport_action.setChecked(False)
        workflow_dock = getattr(self, "workflow_dock", None)
        if workflow_dock is not None:
            workflow_dock.setFloating(False)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, workflow_dock)
            workflow_dock.show()
        validation_dock = getattr(self, "validation_output_dock", None)
        if validation_dock is not None:
            validation_dock.setFloating(False)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, validation_dock)
            if workflow_dock is not None:
                self.tabifyDockWidget(workflow_dock, validation_dock)
            validation_dock.hide()
        if self.layout_manager is not None:
            self.apply_ghost_layout(self.layout_manager.get_layout())
        else:
            self.main_splitter.setSizes([260, 1380, 300])
            if workflow_dock is not None:
                self.resizeDocks((workflow_dock,), (280,), QtCore.Qt.Orientation.Vertical)

    def _show_help(self, topic: str) -> None:
        QtWidgets.QMessageBox.information(
            self,
            topic,
            "Map Studio is GhostRigger's Level Editor opened from the Module Editor icon. It works on KMAP projects and keeps terrain, rooms, walkmeshes, placements, validation, staged export, install handoff, and game-test proof in one workflow without embedding heavy mesh or texture blobs.",
        )

    def set_navigation_profile(self, profile: object) -> None:
        self.viewport_panel.set_navigation_profile(profile)

    def apply_ghost_theme(self, theme) -> None:
        if theme is None:
            return
        if getattr(theme, "is_native", lambda: False)():
            self.apply_native_theme()
            return
        stylesheet = ""
        try:
            from src.gui.libtheme.qt_stylesheet_builder import QtStylesheetBuilder

            stylesheet = QtStylesheetBuilder().build(theme)
        except Exception:
            stylesheet = ""
        if stylesheet:
            self.setStyleSheet(stylesheet)
        for widget in self.findChildren(QtWidgets.QWidget):
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

    def apply_ghost_layout(self, layout) -> None:
        if layout is None:
            return
        if not self.isMaximized():
            self.resize(layout.main_width, layout.main_height)
        self.main_splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
        self.left_authoring_rail.setMinimumWidth(
            max(200, min(240, layout.panel("library").min_width))
        )
        self.right_inspector_rail.setMinimumWidth(
            max(220, min(260, layout.panel("properties").min_width))
        )
        normal_splitter_sizes = [
            max(220, min(340, layout.panel("library").preferred_width)),
            max(layout.viewport.min_width, layout.viewport.preferred_width + 200),
            max(230, min(400, layout.panel("properties").preferred_width)),
        ]
        if self.focus_viewport_action.isChecked():
            if self._map_studio_viewport_focus_state is not None:
                self._map_studio_viewport_focus_state["splitter_sizes"] = normal_splitter_sizes
        else:
            self.main_splitter.setSizes(normal_splitter_sizes)
        if hasattr(self, "left_splitter"):
            self.left_splitter.setHandleWidth(layout.spacing_value("splitterHandleWidth", 6))
            self.left_splitter.setSizes([max(360, layout.main_height - 220)])
        workflow_dock = getattr(self, "workflow_dock", None)
        if workflow_dock is not None and not workflow_dock.isFloating():
            self.resizeDocks(
                (workflow_dock,),
                (max(220, min(360, layout.main_height // 3)),),
                QtCore.Qt.Orientation.Vertical,
            )
        bottom = layout.panel("outputLog")
        validation_dock = getattr(self, "validation_output_dock", None)
        if validation_dock is not None and validation_dock.isVisible() and not validation_dock.isFloating():
            self.resizeDocks(
                (validation_dock,),
                (max(140, min(300, bottom.preferred_height)),),
                QtCore.Qt.Orientation.Vertical,
            )
        self.toolbar.apply_ghost_layout(layout)
        try:
            from src.gui.libtheme.layout_applier import apply_menu_layout

            apply_menu_layout(layout, self)
        except Exception:
            pass
        for widget in self.findChildren(QtWidgets.QWidget):
            hook = getattr(widget, "apply_ghost_layout", None)
            if callable(hook) and widget is not self.toolbar:
                hook(layout)
