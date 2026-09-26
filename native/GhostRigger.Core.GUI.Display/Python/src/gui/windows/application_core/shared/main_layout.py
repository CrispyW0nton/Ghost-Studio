"""Main-window layout construction for GhostRigger."""

from __future__ import annotations

try:
    from PySide6 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - import gate for Qt runtime
    raise RuntimeError("PySide6 is required for the Qt shell") from exc

from src.core.scene.axis_mode import AxisMode
from src.gui.qt_lib.panels.adjust_pivot_panel import AdjustPivotPanel
from src.gui.qt_lib.panels.qt_animation_panel import QtAnimationsPanel
from src.gui.qt_lib.panels.qt_body_attachment_panel import QtBodyAttachmentPanel
from src.gui.qt_lib.panels.qt_camera_panel import QtCameraPanel
from src.gui.qt_lib.panels.qt_content_browser_panel import QtContentBrowserPanel
from src.gui.qt_lib.panels.qt_diagnostics_panel import QtDiagnosticsPanel
from src.gui.qt_lib.panels.qt_lighting_panel import QtLightingPanel
from src.gui.qt_lib.panels.qt_log_panel import QtLogPanel, QtPythonTerminalPanel
from src.gui.qt_lib.panels.qt_mesh_tools_panel import QtMeshToolsPanel
from src.gui.qt_lib.panels.qt_properties_panel import QtPropertiesPanel
from src.gui.qt_lib.panels.qt_resource_panel import QtResourceBrowserPanel, Qt2DABrowserPanel
from src.gui.qt_lib.panels.qt_rig_panel import QtRigWindow
from src.gui.qt_lib.panels.qt_scene_outliner_panel import QtSceneOutlinerPanel
from src.gui.qt_lib.panels.qt_skeleton_panel import QtSkeletonPanel
from src.gui.qt_lib.panels.qt_sprite_material_panel import QtSpriteMaterialPanel
from src.gui.qt_lib.panels.qt_texture_panel import QtTextureToolWindow
from src.core.rendering.renderer_settings import RendererSettings
from src.gui.qt_lib.viewports.qt_viewport import QtMainViewportWidget
from src.core.rendering.viewport_navigation import DEFAULT_VIEWPORT_NAVIGATION_PROFILE
from src.gui.qt_lib.windows.qt_blueprint_editor import QtBlueprintEditorWindow
from src.gui.qt_lib.windows.qt_retarget_preview_controller import QtRetargetViewportAdapter, RetargetPreviewUiController
from src.gui.qt_lib.windows.qt_retarget_workbench_controller import RetargetWorkbenchController
from src.gui.windows.application_core.application_core_lib.functions.qt_helpers import _qt_object_alive
from src.gui.windows.application_core.application_core_lib.shared.workspace_presets import (
    DEFAULT_WORKSPACE_PRESET,
    WORKSPACE_PRESETS,
    WorkspaceSwitcher,
    load_saved_workspace_preset,
    save_workspace_preset,
)
from src.gui.windows.application_core.application_core_lib.shared.undo_stack import SceneUndoStack


class MainWindowLayoutMixin:
    """Widget construction and persistent startup dock layout."""

    def _build_layout(self):
        top_shell = QtWidgets.QWidget()
        top_shell.setObjectName("ReservedTopUi")
        top_shell.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        top_layout = QtWidgets.QVBoxLayout(top_shell)
        top_layout.setContentsMargins(3, 0, 3, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(self._make_header())
        top_layout.addWidget(self._make_command_bar())
        self.reserved_top_shell = top_shell
        self.reserved_top_layout = top_layout

        reserved_toolbar = QtWidgets.QToolBar("GhostRigger Top UI", self)
        reserved_toolbar.setObjectName("ReservedTopToolbar")
        reserved_toolbar.setMovable(False)
        reserved_toolbar.setFloatable(False)
        reserved_toolbar.setAllowedAreas(QtCore.Qt.TopToolBarArea)
        reserved_toolbar.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        reserved_toolbar.setContentsMargins(0, 0, 0, 0)
        reserved_toolbar.addWidget(top_shell)
        self.reserved_top_toolbar = reserved_toolbar
        self.addToolBar(QtCore.Qt.TopToolBarArea, reserved_toolbar)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(3, 0, 3, 3)
        root.setSpacing(0)
        self.setCentralWidget(central)

        self.content_browser_panel = QtContentBrowserPanel(self)
        self.library_panel = self.content_browser_panel
        self.content_browser_panel.set_load_connected_map(
            self.settings_data.get("content_browser_load_connected_map", False)
        )
        self.content_browser_panel.loadConnectedMapChanged.connect(self._set_content_browser_load_connected_map)
        self.content_browser_panel.scanRequested.connect(self._scan_library)
        self.content_browser_panel.deepScanRequested.connect(self._scan_library)
        self.content_browser_panel.loadRequested.connect(self._start_resource_load)
        self.content_browser_panel.primarySceneLoadRequested.connect(self._load_content_browser_primary_scene_model)
        self.content_browser_panel.addToCurrentSceneRequested.connect(self._add_content_browser_model_to_current_scene)
        self.content_browser_panel.extractRequested.connect(self._extract_library_row)
        self.content_browser_panel.assetActionRequested.connect(self._handle_content_browser_asset_action)
        self.content_browser_panel.levelEditorNewRequested.connect(self._send_library_row_to_new_module_editor)
        self.content_browser_panel.levelEditorImportRequested.connect(self._send_library_row_to_module_editor)
        self.content_browser_panel.characterBuilderRequested.connect(self._send_library_row_to_character_builder)
        self.content_browser_panel.retargetSourceRequested.connect(
            lambda row: self._send_library_row_to_retarget(row, "source")
        )
        self.content_browser_panel.retargetTargetRequested.connect(
            lambda row: self._send_library_row_to_retarget(row, "target")
        )
        self.content_browser_panel.batchRequested.connect(self._batch_library_export)
        self.content_browser_panel.libraryActionRequested.connect(self._handle_animation_library_action)

        self.scene_outliner_panel = QtSceneOutlinerPanel(self)
        self.scene_outliner_panel.objectSelected.connect(self._select_scene_object)
        self.scene_outliner_panel.helperNodeSelected.connect(self._on_scene_outliner_helper_node_selected)
        self.scene_outliner_panel.lightNodeSelected.connect(self._on_scene_outliner_light_node_selected)
        self.scene_outliner_panel.objectDeleteRequested.connect(self._delete_scene_object)
        self.scene_outliner_panel.objectDuplicateRequested.connect(self._duplicate_scene_object)
        self.scene_outliner_panel.objectFocusRequested.connect(self._focus_scene_object)
        self.scene_outliner_panel.objectVisibilityChanged.connect(self._set_scene_object_visible)
        self.scene_outliner_panel.objectLockedChanged.connect(self._set_scene_object_locked)
        self.scene_outliner_panel.objectRenamed.connect(self._rename_scene_object)
        self.scene_outliner_panel.objectAddToSequenceRequested.connect(self._add_scene_object_to_sequence)
        self.skeleton_panel = QtSkeletonPanel(self)
        self.lighting_panel = QtLightingPanel(self)
        self.camera_panel = QtCameraPanel(self)
        self.sprite_materials_panel = QtSpriteMaterialPanel(self)
        self.properties_panel = QtPropertiesPanel(self, module_browser_enabled=False)
        self.module_geometry_panel = QtPropertiesPanel(self)
        self.module_geometry_panel.set_module_browser_only(True)
        self.skeleton_panel.nodeSelected.connect(self.properties_panel.show_node)
        self.rig_window = QtRigWindow(self)
        self.rig_window.rigActionRequested.connect(self._handle_rig_action)
        self.rig_panel = self.rig_window.panel
        self.texture_tool_window = QtTextureToolWindow(self)
        self.texture_panel = self.texture_tool_window.texture_panel
        self.normal_map_panel = self.texture_tool_window.normal_map_panel
        self.diagnostics_window = None
        self.diagnostics_panel = QtDiagnosticsPanel(self._get_model, self)
        self.animations_panel = QtAnimationsPanel(self)
        self.animations_panel.animationSelected.connect(self._handle_animation_selected)
        self.animations_panel.animationActionRequested.connect(self._handle_animation_action)
        self.animations_panel.animationSourceChanged.connect(self._handle_animation_source_changed)
        self.animations_panel.animationTargetChanged.connect(self._handle_animation_target_changed)
        self.animations_panel.inheritanceGameChanged.connect(self._handle_animation_inheritance_game_changed)
        self.animations_panel.inheritanceSupermodelChanged.connect(self._handle_animation_inheritance_game_changed)
        self.animations_panel.seekRequested.connect(self._handle_animation_seek)
        self.body_attachment_panel = QtBodyAttachmentPanel(self)
        self.body_attachment_panel.attachRequested.connect(self._handle_bas_attach_requested)
        self.body_attachment_panel.clearRequested.connect(self._handle_bas_clear_requested)
        self.body_attachment_panel.saveBuildRequested.connect(self._handle_bas_save_build_requested)
        self.body_attachment_panel.exportComposedRequested.connect(self._handle_bas_export_composed_requested)
        self.body_attachment_panel.modeChanged.connect(self._handle_bas_mode_changed)
        # Populate the slot combos with every attachable game model once the
        # game library is configured; the first slot click resolves it lazily.
        self.body_attachment_panel.slotSelected.connect(self._ensure_bas_attachment_catalog)
        self.body_attachment_panel.catalogRefreshRequested.connect(self._ensure_bas_attachment_catalog)
        QtCore.QTimer.singleShot(0, self._ensure_bas_attachment_catalog)
        self.animation_library_panel = self.content_browser_panel
        self._retarget_workbench_controls_connected = False
        self._unreal_source_row: Optional[dict] = None
        self._unreal_source_game = ""
        self.twoda_panel = Qt2DABrowserPanel(self)
        self.twoda_panel.refreshRequested.connect(self._refresh_twoda_panel)
        self.twoda_panel.tableSelected.connect(self._load_twoda_table)
        self.resource_panel = QtResourceBrowserPanel(self)
        self.resource_panel.scanRequested.connect(self._populate_resource_panel)
        self.resource_panel.resourceSelected.connect(self._preview_resource_row)
        self.resource_panel.resourceActivated.connect(self._activate_resource_row)
        self.module_editor_window = None
        self.mesh_tools_panel = QtMeshToolsPanel(self)
        self.adjust_pivot_panel = AdjustPivotPanel(self)
        self.blueprint_window = QtBlueprintEditorWindow(self)
        self.blueprint_panel = self.blueprint_window.panel
        self._detachable_panels: dict[str, QtWidgets.QDockWidget] = {}
        self._detachable_panel_sizes = {
            "content_browser": (760, 520),
            "scene": (360, 620),
            "properties": (420, 720),
            "animations": (380, 520),
            "nodes": (620, 700),
            "lighting": (420, 620),
            "cameras": (460, 680),
            "module_meshes": (620, 720),
            "sprite_materials": (560, 680),
            "mesh_tools": (420, 760),
            "adjust_pivot": (320, 420),
            "output_log": (760, 320),
            "python_terminal": (760, 320),
            "2das": (980, 640),
            "resources": (980, 640),
            "sequence_editor": (1180, 720),
            "diagnostics": (760, 560),
        }
        self.content_browser_dock = self._create_detachable_panel(
            "content_browser",
            "Content Browser",
            self.content_browser_panel,
            QtCore.Qt.LeftDockWidgetArea,
            scroll=False,
        )
        self.scene_dock = self._create_detachable_panel(
            "scene",
            "Scene",
            self.scene_outliner_panel,
            QtCore.Qt.LeftDockWidgetArea,
            scroll=False,
        )
        self.properties_dock = self._create_detachable_panel(
            "properties",
            "Properties",
            self.properties_panel,
            QtCore.Qt.RightDockWidgetArea,
            scroll=True,
        )
        self.animations_dock = self._create_detachable_panel(
            "animations",
            "Animation Browser",
            self.animations_panel,
            QtCore.Qt.RightDockWidgetArea,
            scroll=False,
        )
        self.body_attachment_dock = self._create_detachable_panel(
            "body_attachment",
            "Body Attachment System",
            self.body_attachment_panel,
            QtCore.Qt.RightDockWidgetArea,
            scroll=False,
        )
        self._create_detachable_panel("nodes", "Nodes", self.skeleton_panel, QtCore.Qt.LeftDockWidgetArea)
        self._create_detachable_panel("lighting", "Lighting", self.lighting_panel, QtCore.Qt.RightDockWidgetArea)
        self._create_detachable_panel("cameras", "Cameras", self.camera_panel, QtCore.Qt.RightDockWidgetArea)
        self._create_detachable_panel("module_meshes", "Module Meshes", self.module_geometry_panel, QtCore.Qt.RightDockWidgetArea)
        self._create_detachable_panel("sprite_materials", "Sprite Materials", self.sprite_materials_panel, QtCore.Qt.RightDockWidgetArea)
        self.mesh_tools_dock = self._create_detachable_panel("mesh_tools", "Mesh Tools", self.mesh_tools_panel, QtCore.Qt.RightDockWidgetArea)
        self.adjust_pivot_dock = self._create_detachable_panel("adjust_pivot", "Adjust Pivot", self.adjust_pivot_panel, QtCore.Qt.RightDockWidgetArea)
        self.log_panel = QtLogPanel(self)
        self.log_panel.setMinimumHeight(96)
        self.output_log_dock = self._create_detachable_panel(
            "output_log",
            "Output Log",
            self.log_panel,
            QtCore.Qt.BottomDockWidgetArea,
            scroll=False,
        )
        self.python_terminal_panel = QtPythonTerminalPanel(self)
        self.python_terminal_panel.setMinimumHeight(96)
        self.python_terminal_dock = self._create_detachable_panel(
            "python_terminal",
            "Python Terminal",
            self.python_terminal_panel,
            QtCore.Qt.BottomDockWidgetArea,
            scroll=False,
        )
        self._create_detachable_panel("2das", "2DA Browser", self.twoda_panel, QtCore.Qt.LeftDockWidgetArea)
        self._create_detachable_panel("resources", "Resource Browser", self.resource_panel, QtCore.Qt.LeftDockWidgetArea)
        self.diagnostics_dock = self._create_detachable_panel(
            "diagnostics",
            "Diagnostics",
            self.diagnostics_panel,
            QtCore.Qt.RightDockWidgetArea,
            scroll=False,
        )
        self._stack_content_browser_under_scene()

        self.viewport = QtMainViewportWidget(self)
        self.viewport.set_renderer_settings(RendererSettings.from_settings(self.settings_data))
        self.viewport.set_navigation_profile(
            self.settings_data.get("viewport_navigation_profile", DEFAULT_VIEWPORT_NAVIGATION_PROFILE)
        )
        self.viewport.set_axis_mode(self.settings_data.get("last_axis_mode", AxisMode.WORLD.value))
        self.settings_data["last_pivot_edit_mode"] = "affect_object_only"
        self.viewport.set_pivot_edit_mode("affect_object_only")
        self._install_editor_integration_services()
        # Global scene-level undo/redo stack (#2).
        self._scene_undo_stack = SceneUndoStack(max_size=100, parent=self)
        self._transform_snapshots: dict[int, tuple] = {}
        self._lighting_snapshot = None
        self._camera_snapshot = None
        self._applying_scene_undo = False
        self._active_workspace = DEFAULT_WORKSPACE_PRESET
        # Wire SceneUndoStack signals to refresh the global undo/redo actions.
        connect_signals = getattr(self, "_connect_scene_undo_signals", None)
        if callable(connect_signals):
            connect_signals()
        self.adjust_pivot_panel.set_pivot_mode(self.viewport.pivot_edit_mode())
        self.adjust_pivot_panel.pivotModeChanged.connect(self._set_pivot_edit_mode)
        self.adjust_pivot_panel.pivotActionRequested.connect(self._apply_pivot_action)
        self.mesh_tools_panel.set_viewport(self.viewport)
        if bool(self.settings_data.get("show_adjust_pivot_toolbox", False)):
            self.adjust_pivot_dock.show()
        self.viewport.setMinimumWidth(420)
        self.viewport.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.viewport_label = self.viewport.canvas
        self.skeleton_panel.nodeSelected.connect(self._on_skeleton_node_selected)
        self.viewport.nodeSelected.connect(self.properties_panel.show_node)
        self.viewport.nodeSelected.connect(self._on_viewport_scene_node_selected)
        self.viewport.nodeSelected.connect(self.module_geometry_panel.show_node)
        self.viewport.nodeSelected.connect(self.module_geometry_panel.select_module_mesh)
        self.viewport.meshSelectionChanged.connect(self.module_geometry_panel.select_module_meshes)
        self.viewport.nodeMoved.connect(self.properties_panel.show_node)
        self.viewport.nodeMoved.connect(self._on_viewport_scene_node_moved)
        self.viewport.sceneObjectDeleteRequested.connect(self._delete_scene_object)
        self.viewport.statusMessage.connect(self.statusBar().showMessage)
        self.viewport.renderStateChanged.connect(self._on_viewport_render_state_changed)
        self.viewport.renderStateChanged.connect(self._on_renderer_backend_status_changed)
        self.viewport.axis_mode_control.axisModeChanged.connect(self._persist_axis_mode)
        self.viewport.nodeMoved.connect(self.module_geometry_panel.show_node)
        self.viewport.nodeMoved.connect(lambda node: self._record_transform_event(node))
        self.viewport.meshVisibilityChanged.connect(self._on_viewport_mesh_visibility_changed)
        self.viewport.gpuUploadProgress.connect(self._on_viewport_gpu_upload_progress)
        self.lighting_panel.lightingModeChanged.connect(self.viewport.set_lighting_mode)
        self.lighting_panel.mapToggled.connect(self.viewport.set_texture_map_enabled)
        self.lighting_panel.lightmapSettingsChanged.connect(self.viewport.set_lightmap_settings)
        self.lighting_panel.shaderComplexityChanged.connect(self.viewport.set_shader_complexity_mode)
        self.lighting_panel.lightChanged.connect(self._on_lighting_panel_changed)
        self.lighting_panel.lightChanged.connect(lambda payload=None: self._record_lighting_event(payload))
        self.lighting_panel.lightSelected.connect(lambda node: self.viewport.set_selected_node(node, source="lighting panel"))
        self.lighting_panel.createLightRequested.connect(self._create_scene_light_object)
        self.lighting_panel.lightmapBakeRequested.connect(self._open_lightmap_baker)
        self.viewport.nodeSelected.connect(self.lighting_panel.select_light)
        self.lighting_panel.apply_preview_settings_to_viewport(self.viewport)
        self.sprite_materials_panel.spriteSelected.connect(self._on_sprite_material_selected)
        self.sprite_materials_panel.spriteRenderChanged.connect(self._on_sprite_materials_changed)
        self._sync_lighting_helper_visibility_to_viewport()
        self.camera_panel.cameraSelected.connect(lambda node: self.viewport.set_selected_node(node, source="camera panel"))
        self.camera_panel.cameraChanged.connect(self._on_camera_panel_changed)
        self.camera_panel.cameraChanged.connect(lambda: self._record_camera_event(None))
        self.camera_panel.activeCameraRequested.connect(self.viewport.switch_to_camera)
        self.camera_panel.clearActiveCameraRequested.connect(self.viewport.switch_to_perspective)
        self.camera_panel.createCameraRequested.connect(self._create_scene_camera_object)
        self.camera_panel.createFromViewRequested.connect(self._create_scene_camera_from_view)
        self.camera_panel.alignCameraToViewRequested.connect(self.viewport.align_camera_to_current_view)
        self.camera_panel.alignViewToCameraRequested.connect(self.viewport.align_view_to_camera)
        self.camera_panel.deleteCameraRequested.connect(self._delete_scene_camera_object)
        self.camera_panel.duplicateCameraRequested.connect(self._duplicate_scene_camera_object)
        self.camera_panel.renderFrameRequested.connect(self._open_render_frame_dialog)
        self.viewport.cameraChanged.connect(self._sync_camera_panel_from_viewport)
        self.viewport.activeCameraChanged.connect(lambda _node=None: self.camera_panel.refresh())
        self.viewport.cameraSelectionChanged.connect(self.camera_panel.select_camera_object)
        self.module_geometry_panel.moduleMeshesSelected.connect(self._on_module_meshes_selected_from_panel)
        self.module_geometry_panel.moduleMeshVisibilityChanged.connect(self._on_module_mesh_visibility_changed)
        self.properties_panel.positionApplied.connect(
            lambda node, _x, _y, _z: self.viewport.refresh_node_transform(node)
        )
        self.properties_panel.positionApplied.connect(
            lambda node, _x, _y, _z: self._record_transform_event(node)
        )
        self.properties_panel.rotationApplied.connect(
            lambda node, _rx, _ry, _rz: self.viewport.refresh_node_transform(node)
        )
        self.properties_panel.rotationApplied.connect(
            lambda node, _rx, _ry, _rz: self._record_transform_event(node)
        )
        self.properties_panel.scaleApplied.connect(
            lambda node, _sx, _sy, _sz: self.viewport.refresh_node_transform(node)
        )
        self.properties_panel.scaleApplied.connect(
            lambda node, _sx, _sy, _sz: self._record_transform_event(node)
        )
        self.viewport.measurementSettingsChanged.connect(self._merge_measurement_settings)
        self._apply_measurement_settings()
        self._retarget_preview_viewport = QtRetargetViewportAdapter(self.viewport, parent=self)
        self.retarget_preview_controller = RetargetPreviewUiController(
            viewport=self._retarget_preview_viewport,
            preview_action=self.preview_retarget_action,
            export_action=self.export_retarget_preview_action,
            target_model_provider=lambda: self._retarget_target_model or self._current_model,
            resource_manager_provider=lambda: self._resource_manager or self._get_resource_manager(),
            log_callback=self._log,
            status_callback=self.statusBar().showMessage,
        )
        self.retarget_workbench_controller = RetargetWorkbenchController(
            ue_to_kotor_controller=self.retarget_preview_controller,
            viewport=self._retarget_preview_viewport,
            preview_action=self.preview_retarget_action,
            export_action=self.export_retarget_preview_action,
            log_callback=self._log,
            status_callback=self.statusBar().showMessage,
        )
        self._connect_retarget_workbench_window_controls()
        self._apply_retarget_workbench_mode_status()

        viewport_toolbar_band = self._make_viewport_toolbar_band(self.viewport.take_viewport_toolbar())
        if viewport_toolbar_band is not None:
            self.reserved_top_layout.addWidget(viewport_toolbar_band)
            QtCore.QTimer.singleShot(0, self._sync_reserved_top_rows)
        root.addWidget(self.viewport, 1)
        self._install_gui_log_handler()
        self._configure_python_terminal_context()
        self.splitDockWidget(self.output_log_dock, self.python_terminal_dock, QtCore.Qt.Horizontal)

        # Compatibility placeholders for the already-migrated loading helpers.
        self.k1_dir_edit = QtWidgets.QLineEdit(str(self.settings_data.get("k1_dir") or ""))
        self.k2_dir_edit = QtWidgets.QLineEdit(str(self.settings_data.get("k2_dir") or ""))
        self.scan_button = QtWidgets.QPushButton("Scan")
        self.library_list = QtWidgets.QListWidget()
        self.library_filter = QtWidgets.QLineEdit()
        self.props_text = QtWidgets.QTextEdit()

        # Apply the saved workspace preset after the layout is fully assembled.
        QtCore.QTimer.singleShot(0, self._apply_startup_workspace)

    # ── Workspace presets (#5) ──────────────────────────────────────────

    def apply_workspace(self, preset_name: str) -> None:
        """Show/hide dock panels according to a named workspace preset."""
        preset = WORKSPACE_PRESETS.get(preset_name)
        if preset is None:
            self._log(f"Unknown workspace preset: {preset_name}", "warning")
            return
        panels = getattr(self, "_detachable_panels", {})
        visible = set(preset.get("visible_docks", []))
        hidden = set(preset.get("hidden_docks", []))

        for key in preset.get("visible_docks", []):
            if key in panels:
                self._show_workspace_dock(key)
        for key in preset.get("hidden_docks", []):
            dock = panels.get(key)
            if dock is not None and _qt_object_alive(dock):
                self._hide_workspace_dock(key)

        save_workspace_preset(preset_name)
        self._active_workspace = preset_name
        switcher = getattr(self, "workspace_switcher", None)
        if switcher is not None:
            switcher.set_current_preset(preset_name)
        self._log(f"Workspace switched to: {preset.get('label', preset_name)}", "info")

    def _apply_startup_workspace(self) -> None:
        """Restore the last-used (or default) workspace preset on startup."""
        preset_name = load_saved_workspace_preset()
        self.apply_workspace(preset_name)

    def _hide_workspace_dock(self, key: str) -> None:
        """Hide a workspace dock and sync its toggle action."""
        dock = getattr(self, "_detachable_panels", {}).get(key)
        if dock is None or not _qt_object_alive(dock):
            return
        if dock.isFloating():
            dock.setFloating(False)
        dock.hide()
        self._sync_dock_toggle_action(key, False)
        self._persist_selected_layout_dock_state()

    def _stack_content_browser_under_scene(self) -> None:
        """Keep startup browsing in the left dock column without stealing the output/log row."""
        if not all(_qt_object_alive(dock) for dock in (self.scene_dock, self.content_browser_dock)):
            return
        if self.scene_dock.isFloating() or self.content_browser_dock.isFloating():
            return
        self.splitDockWidget(self.scene_dock, self.content_browser_dock, QtCore.Qt.Vertical)
        QtCore.QTimer.singleShot(0, self._resize_startup_left_dock_stack)

    def _resize_startup_left_dock_stack(self) -> None:
        if not all(_qt_object_alive(dock) for dock in (self.scene_dock, self.content_browser_dock)):
            return
        layout = self.layout_manager.get_layout()
        scene_panel = layout.panel("scene")
        content_panel = layout.panel("contentBrowser")
        scene_height = max(scene_panel.min_height, scene_panel.preferred_height)
        content_height = max(content_panel.min_height, content_panel.preferred_height)
        left_width = max(scene_panel.min_width, scene_panel.preferred_width, content_panel.min_width, content_panel.preferred_width)
        try:
            self.resizeDocks(
                [self.scene_dock, self.content_browser_dock],
                [scene_height, content_height],
                QtCore.Qt.Vertical,
            )
            self.resizeDocks(
                [self.scene_dock, self.content_browser_dock],
                [left_width, left_width],
                QtCore.Qt.Horizontal,
            )
        except RuntimeError:
            pass
