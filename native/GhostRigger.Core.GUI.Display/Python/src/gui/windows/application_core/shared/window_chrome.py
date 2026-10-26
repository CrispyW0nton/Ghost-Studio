"""Main-window actions, menus, and top chrome widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - import gate for Qt runtime
    raise RuntimeError("PySide6 is required for the Qt shell") from exc

from src.gui.qt_lib.assets.qt_matrix_background import QtMatrixLabel, QtMatrixPanel
from src.gui.qt_lib.dialogs.qt_dialogs import (
    show_about,
    show_format_reference,
    show_ipc_info,
    show_viewport_navigation_reference,
)
from src.gui.qt_lib.dialogs.qt_settings_dialog import save_settings
from src.gui.libtheme.icon_manager import ThemeIconManager
from src.gui.qt_lib.assets.qt_theme import C
from src.gui.windows.application_core.shared.workspace_presets import (
    WorkspaceSwitcher,
    load_saved_workspace_preset,
)

# ``C`` is the shared, theme-aware compatibility palette.  It is seeded
# from ``LEGACY_MATRIX_COLORS`` and kept in sync with the active theme by
# ``update_legacy_palette(theme)`` (invoked from the theme applier), so
# every colour read below reflects the resolved theme rather than a
# frozen Matrix-default snapshot.  (Issue #11 theme-token migration.)
_GUI_DIR = Path(__file__).resolve().parents[3]
_QT_ICON_DIR = (_GUI_DIR / "icons").as_posix()
_fallback_icons = ThemeIconManager(_GUI_DIR / "icons")

# Semantic product-surface icon keys.  Keep QAction and command-strip routing
# explicit so a workbench cannot silently fall back to another product's icon.
MAIN_ACTION_ICON_KEYS: dict[str, str] = {
    "theme_editor": "theme_editor",
    "character_builder": "character_builder",
    "animation_library": "animation_library",
    "animation_browser": "animation_browser",
    "retarget_workbench": "retarget",
    "retarget_preview": "retarget_preview",
    "unreal_animator": "unreal_animator",
    "map_studio": "map_studio",
    "gui_editor": "gui_editor",
    "placeable_builder": "placeable_builder",
    "particle_editor": "particle_editor",
    "scripting_dialogue_studio": "scripting_dialogue_studio",
    "module_editor": "module_editor",
    "rigging": "rigging",
    "blueprint_editor": "blueprint_editor",
    "content_browser": "content_browser",
    "nodes": "nodes",
    "twoda_browser": "twoda_browser",
    "resource_browser": "resource_browser",
    "tutorials": "tutorials",
    "module_meshes": "module_meshes",
}

MAIN_COMMAND_STRIP_ICON_KEYS: dict[str, str] = {
    "character_builder": "character_builder",
    "map_studio": "map_studio",
    "gui_editor": "gui_editor",
    "placeable_builder": "placeable_builder",
    "particle_editor": "particle_editor",
    "scripting_dialogue_studio": "scripting_dialogue_studio",
    "module_editor": "module_editor",
    "content_browser": "content_browser",
    "animation_browser": "animation_browser",
    "nodes": "nodes",
    "module_meshes": "module_meshes",
    "twoda_browser": "twoda_browser",
    "resource_browser": "resource_browser",
}


class WindowChromeMixin:
    """Actions, menus, command strip, and Matrix header behavior."""

    def _build_actions(self):
        self.new_scene_action = QtGui.QAction(self._icon("new_scene"), "New Scene", self)
        self.new_scene_action.setShortcut("Ctrl+N")
        self.new_scene_action.triggered.connect(self._new_scene)
        self.open_scene_action = QtGui.QAction(self._icon("open"), "Open Scene...", self)
        self.open_scene_action.setShortcut("Ctrl+O")
        self.open_scene_action.triggered.connect(self._open_scene)
        self.save_scene_action = QtGui.QAction("Save Scene", self)
        self.save_scene_action.setShortcut("Ctrl+S")
        self.save_scene_action.triggered.connect(self._save_scene)
        self.save_scene_as_action = QtGui.QAction("Save Scene As...", self)
        self.save_scene_as_action.setShortcut("Ctrl+Shift+S")
        self.save_scene_as_action.triggered.connect(self._save_scene_as)
        self.close_scene_action = QtGui.QAction("Close Scene", self)
        self.close_scene_action.triggered.connect(self._close_scene)
        self.export_scene_action = QtGui.QAction("Export Scene...", self)
        self.export_scene_action.triggered.connect(self._export_scene)

        self.open_model_action = QtGui.QAction(self._icon("open"), "Open KOTOR Model (Automatic)...", self)
        self.open_model_action.setShortcut("Ctrl+Shift+M")
        self.open_model_action.setStatusTip("Automatically detect binary/ASCII MDL, KOTOR 1/2, MDX pairing, and model type")
        self.open_model_action.triggered.connect(self._open_model)

        self.open_ascii_action = QtGui.QAction("Open KOTOR Model (Automatic; alternate shortcut)...", self)
        self.open_ascii_action.setShortcut("Ctrl+Shift+O")
        self.open_ascii_action.setStatusTip("Alternate shortcut for the automatic KOTOR model importer")
        self.open_ascii_action.triggered.connect(self._open_model)
        self.clear_model_action = QtGui.QAction("Clear Model", self)
        self.clear_model_action.setShortcut("Ctrl+Shift+Delete")
        self.clear_model_action.triggered.connect(self._clear_model)
        self.import_obj_action = QtGui.QAction("Import OBJ...", self)
        self.import_obj_action.setShortcut("Ctrl+I")
        self.import_obj_action.triggered.connect(self._import_obj)
        self.import_fbx_action = QtGui.QAction("Import FBX...", self)
        self.import_fbx_action.triggered.connect(self._import_fbx)
        self.import_gltf_action = QtGui.QAction("Import GLB/GLTF...", self)
        self.import_gltf_action.triggered.connect(self._import_gltf)
        self.save_ascii_action = QtGui.QAction("Save ASCII MDL...", self)
        self.save_ascii_action.setShortcut("Ctrl+Alt+S")
        self.save_ascii_action.triggered.connect(self._save_ascii_mdl)
        self.export_binary_action = QtGui.QAction("Export Binary MDL...", self)
        self.export_binary_action.setShortcut("Ctrl+Shift+B")
        self.export_binary_action.triggered.connect(self._export_mdl_binary)
        self.export_obj_action = QtGui.QAction("Export OBJ...", self)
        self.export_obj_action.setShortcut("Ctrl+E")
        self.export_obj_action.triggered.connect(self._export_obj)
        self.export_fbx_action = QtGui.QAction("Export FBX...", self)
        self.export_fbx_action.triggered.connect(self._export_fbx)
        self.export_selected_fbx_action = QtGui.QAction("Export Selected FBX...", self)
        self.export_selected_fbx_action.triggered.connect(self._export_selected_fbx)
        self.export_full_module_fbx_action = QtGui.QAction(
            "Export Full Module as One FBX...",
            self,
        )
        self.export_full_module_fbx_action.triggered.connect(
            self._export_full_module_fbx,
        )
        self.export_full_module_obj_action = QtGui.QAction(
            "Export Full Module as One OBJ...",
            self,
        )
        self.export_full_module_obj_action.setStatusTip(
            "Export every room in the loaded module layout as one OBJ, "
            "including all renderable geometry, materials, and textures"
        )
        self.export_full_module_obj_action.triggered.connect(
            self._export_full_module_obj,
        )
        self.export_maya_room_module_obj_action = QtGui.QAction(
            "Export Maya-Friendly Module (Room Objects)...",
            self,
        )
        self.export_maya_room_module_obj_action.setStatusTip(
            "Export the loaded module as one OBJ containing one positioned, "
            "selectable mesh object per room"
        )
        self.export_maya_room_module_obj_action.triggered.connect(
            self._export_maya_room_module_obj,
        )
        self.export_clean_module_obj_action = QtGui.QAction(
            "Export Clean Full Map as OBJ...",
            self,
        )
        self.export_clean_module_obj_action.setStatusTip(
            "Export every room as one OBJ without the skybox or oversized "
            "background plane, including materials and textures"
        )
        self.export_clean_module_obj_action.triggered.connect(
            self._export_clean_module_obj,
        )
        self.export_gltf_action = QtGui.QAction("Export GLB/GLTF...", self)
        self.export_gltf_action.setShortcut("Ctrl+G")
        self.export_gltf_action.triggered.connect(self._export_gltf)
        self.export_humanoid_action = QtGui.QAction("Export Humanoid Template...", self)
        self.export_humanoid_action.triggered.connect(self._export_humanoid_template)
        self.texture_dir_action = QtGui.QAction("Set Texture Directory...", self)
        self.texture_dir_action.triggered.connect(self._set_texture_dir)
        self.settings_action = QtGui.QAction(self._icon("settings"), "Settings...", self)
        self.settings_action.setShortcut("Ctrl+Comma")
        self.settings_action.triggered.connect(self._open_settings_dialog)
        self.theme_editor_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["theme_editor"]), "Theme Editor...", self)
        self.theme_editor_action.triggered.connect(self._open_theme_editor_window)
        self.autorig_action = QtGui.QAction(self._icon("autorig"), "Auto-Rig Current Model", self)
        self.autorig_action.setShortcut("Ctrl+Shift+G")
        self.autorig_action.triggered.connect(self._quick_autorig)
        self.remove_rig_action = QtGui.QAction("Remove Rigging", self)
        self.remove_rig_action.triggered.connect(self._remove_rig)
        self.frame_all_action = QtGui.QAction("Frame All", self)
        self.frame_all_action.setShortcut("F")
        self.frame_all_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.frame_all_action.triggered.connect(lambda: self._call_viewport("frame_all"))
        self.reset_camera_action = QtGui.QAction("Reset Camera", self)
        self.reset_camera_action.setShortcut("R")
        self.reset_camera_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.reset_camera_action.triggered.connect(lambda: self._call_viewport("reset_camera"))
        self.undo_viewport_action = QtGui.QAction("Undo", self)
        self.undo_viewport_action.setShortcut("Ctrl+Z")
        self.undo_viewport_action.triggered.connect(self._combined_undo)
        self.redo_viewport_action = QtGui.QAction("Redo", self)
        self.redo_viewport_action.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z", "Ctrl+R"])
        self.redo_viewport_action.triggered.connect(self._combined_redo)
        self.wire_action = QtGui.QAction("Toggle Wireframe", self)
        self.wire_action.setShortcut("W")
        self.wire_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.wire_action.triggered.connect(lambda: self._click_viewport_button("wire_button"))
        self.bones_action = QtGui.QAction("Toggle Bones", self)
        self.bones_action.setShortcut("B")
        self.bones_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.bones_action.triggered.connect(lambda: self._click_viewport_button("bones_button"))
        self.texture_action = QtGui.QAction("Toggle Texture", self)
        self.texture_action.setShortcut("T")
        self.texture_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.texture_action.triggered.connect(lambda: self._click_viewport_button("texture_button"))
        self.grid_action = QtGui.QAction("Toggle Grid", self)
        self.grid_action.setShortcut("Alt+G")
        self.grid_action.triggered.connect(lambda: self._click_viewport_button("grid_button"))
        self.uv_action = QtGui.QAction("Open UV Viewer...", self)
        self.uv_action.triggered.connect(self._open_uv_viewer)
        self.diag_action = QtGui.QAction(self._icon("diag"), "Diagnostics...", self)
        self.diag_action.setShortcut("Ctrl+Shift+D")
        self._configure_dock_toggle_action(self.diag_action, "diagnostics", self._show_diagnostics_panel)
        self.fbx_sdk_status_action = QtGui.QAction("FBX SDK Status", self)
        self.fbx_sdk_status_action.triggered.connect(self._show_fbx_sdk_status)
        self.fbx_sdk_setup_action = QtGui.QAction("Autodesk FBX SDK Setup...", self)
        self.fbx_sdk_setup_action.triggered.connect(self._open_fbx_sdk_setup)
        self.info_action = QtGui.QAction("Model Info...", self)
        self.info_action.triggered.connect(self._show_model_info)
        self.refresh_action = QtGui.QAction("Refresh All", self)
        self.refresh_action.setShortcut("F5")
        self.refresh_action.triggered.connect(self._refresh_all)
        self.character_builder_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["character_builder"]), "Character Builder (New Window)...", self)
        self.character_builder_action.setShortcut("Ctrl+Shift+C")
        self.character_builder_action.triggered.connect(self._open_qt_character_builder_window)
        self.head_builder_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["character_builder"]),
            "Custom KOTOR Head Builder...",
            self,
        )
        self.head_builder_action.setShortcut("Ctrl+Shift+H")
        self.head_builder_action.triggered.connect(
            lambda _checked=False: self._open_qt_character_builder_window(
                "native_kotor_head"
            )
        )
        self.anims_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["animation_library"]), "Animation Library", self)
        self.anims_action.setShortcut("Ctrl+Shift+A")
        self.anims_action.triggered.connect(lambda: self._show_content_browser("Animation"))
        self.animation_browser_dock_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["animation_browser"]), "Animation Browser", self)
        self._configure_dock_toggle_action(
            self.animation_browser_dock_action,
            "animations",
            lambda: self._show_workspace_dock("animations"),
        )
        self.body_attachment_panel_action = QtGui.QAction(self._icon("body_attachment"), "Body Attachment System", self)
        self._configure_dock_toggle_action(
            self.body_attachment_panel_action,
            "body_attachment",
            lambda: self._show_workspace_dock("body_attachment"),
        )
        self.retarget_workbench_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["retarget_workbench"]), "Animation Retargeting Workbench...", self)
        self.retarget_workbench_action.setShortcut("Ctrl+Shift+A")
        self.retarget_workbench_action.triggered.connect(self._open_animation_retarget_window)
        self.load_retarget_source_clip_action = QtGui.QAction("Load UE/FBX Source Animation...", self)
        self.load_retarget_source_clip_action.triggered.connect(self._load_retarget_source_clip)
        self.load_retarget_profile_action = QtGui.QAction("Load Retarget Profile...", self)
        self.load_retarget_profile_action.triggered.connect(self._load_retarget_profile)
        self.preview_retarget_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["retarget_preview"]), "Preview Retarget", self)
        self.preview_retarget_action.setEnabled(False)
        self.preview_retarget_action.triggered.connect(self._preview_retarget_animation)
        self.export_retarget_preview_action = QtGui.QAction("Export Retarget Preview...", self)
        self.export_retarget_preview_action.setEnabled(False)
        self.export_retarget_preview_action.triggered.connect(self._export_retarget_preview)
        self.unreal_animator_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["unreal_animator"]), "Unreal Animator...", self)
        self.unreal_animator_action.setShortcut("Ctrl+Shift+U")
        self.unreal_animator_action.triggered.connect(self._open_unreal_animator_window)
        self.sequence_editor_action = QtGui.QAction(self._icon("sequence"), "Sequence Editor", self)
        self.sequence_editor_action.setShortcut("Ctrl+Alt+Q")
        self._configure_dock_toggle_action(self.sequence_editor_action, "sequence_editor", self._show_sequence_editor_dock)
        self.modules_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["map_studio"]), "Open Map Studio (KMAP Area Authoring)", self)
        self.modules_action.setStatusTip("Author new KMAP areas with room geometry, walkmeshes, and gameplay placements")
        self.modules_action.triggered.connect(self._open_map_studio_modeling_workspace)
        self.gui_editor_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["gui_editor"]),
            "Open GUI Editor (Odyssey UI)...",
            self,
        )
        self.gui_editor_action.setStatusTip(
            "Inspect and author KOTOR .gui resources in a standalone workbench with a shared PIE preview contract"
        )
        self.gui_editor_action.triggered.connect(self._open_gui_editor_window)
        self.placeable_builder_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["placeable_builder"]), "Open Placeable Builder...", self
        )
        self.placeable_builder_action.setStatusTip(
            "Design reusable containers, terminals, puzzles, interactive props, and decor for the Placeable Library"
        )
        self.placeable_builder_action.triggered.connect(self._open_placeable_builder_window)
        self.particle_editor_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["particle_editor"]), "Open Particle Editor...", self
        )
        self.particle_editor_action.setStatusTip(
            "Edit KOTOR emitter particle systems live and browse every retail emitter as a reusable template"
        )
        self.particle_editor_action.triggered.connect(self._open_particle_editor_window)
        self.scripting_dialogue_studio_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["scripting_dialogue_studio"]),
            "Open Scripting Suite...",
            self,
        )
        self.scripting_dialogue_studio_action.setShortcut("Ctrl+Shift+J")
        self.scripting_dialogue_studio_action.setStatusTip(
            "Author scripts, dialogue, quests, journals, tables, TLK/LIP/SSF, blueprints, projects, and validated packages"
        )
        self.scripting_dialogue_studio_action.triggered.connect(self._open_scripting_dialogue_studio_window)
        self.stock_module_editor_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["module_editor"]), "Open Module Editor (Stock MOD/RIM Patcher)", self)
        self.stock_module_editor_action.setStatusTip("Patch textures, walkmeshes, and objects in existing stock .mod/.rim module archives")
        self.stock_module_editor_action.triggered.connect(self._open_stock_module_editor_window)
        self.rig_window_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["rigging"]), "Open Rigging Window", self)
        self.rig_window_action.triggered.connect(self._open_rig_window)
        self.texture_tool_action = QtGui.QAction(self._icon("texture"), "Texture Tool...", self)
        self.texture_tool_action.triggered.connect(self._open_texture_tool_window)
        self.blueprint_editor_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["blueprint_editor"]), "Blueprint Editor...", self)
        self.blueprint_editor_action.triggered.connect(self._open_blueprint_editor_window)
        self.content_browser_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["content_browser"]), "Open Content Browser", self)
        self._configure_dock_toggle_action(
            self.content_browser_action,
            "content_browser",
            lambda: self._show_content_browser("All"),
        )
        self.scene_panel_action = QtGui.QAction(self._icon("scene"), "Scene Information", self)
        self._configure_dock_toggle_action(self.scene_panel_action, "scene", lambda: self._show_workspace_dock("scene"))
        self.properties_panel_action = QtGui.QAction(self._icon("props"), "Open Properties", self)
        self._configure_dock_toggle_action(
            self.properties_panel_action,
            "properties",
            lambda: self._show_workspace_dock("properties"),
        )
        self.nodes_panel_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["nodes"]), "Open Nodes Panel", self)
        self._configure_dock_toggle_action(self.nodes_panel_action, "nodes", lambda: self._show_workspace_dock("nodes"))
        self.lighting_panel_action = QtGui.QAction(self._icon("lights"), "Open Lighting Panel", self)
        self._configure_dock_toggle_action(self.lighting_panel_action, "lighting", lambda: self._show_workspace_dock("lighting"))
        self.camera_panel_action = QtGui.QAction(self._icon("cameras"), "Open Camera Panel", self)
        self._configure_dock_toggle_action(self.camera_panel_action, "cameras", lambda: self._show_workspace_dock("cameras"))
        self.create_free_camera_action = QtGui.QAction(self._icon("camera_free"), "Free Camera", self)
        self.create_free_camera_action.triggered.connect(lambda: self._create_scene_camera_object("Free Camera"))
        self.create_target_camera_action = QtGui.QAction(self._icon("camera_target"), "Target Camera", self)
        self.create_target_camera_action.triggered.connect(lambda: self._create_scene_camera_object("Target Camera"))
        self.create_cinematic_camera_action = QtGui.QAction(self._icon("camera_cinematic"), "Cinematic Camera", self)
        self.create_cinematic_camera_action.triggered.connect(lambda: self._create_scene_camera_object("Cinematic Camera"))
        self.create_point_light_action = QtGui.QAction(self._icon("light_point"), "Point Light", self)
        self.create_point_light_action.triggered.connect(lambda: self._create_scene_light_object("point"))
        self.create_spot_light_action = QtGui.QAction(self._icon("light_spot"), "Spot Light", self)
        self.create_spot_light_action.triggered.connect(lambda: self._create_scene_light_object("spot"))
        self.create_directional_light_action = QtGui.QAction(self._icon("light_directional"), "Directional Light", self)
        self.create_directional_light_action.triggered.connect(lambda: self._create_scene_light_object("directional"))
        self.create_area_light_action = QtGui.QAction(self._icon("light_area"), "Area Light", self)
        self.create_area_light_action.triggered.connect(lambda: self._create_scene_light_object("area"))
        self.create_ambient_light_action = QtGui.QAction(self._icon("light_ambient"), "Ambient Light", self)
        self.create_ambient_light_action.triggered.connect(lambda: self._create_scene_light_object("ambient"))
        self.render_frame_action = QtGui.QAction("Render Camera Still...", self)
        self.render_frame_action.triggered.connect(self._open_render_frame_dialog)
        self.twoda_panel_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["twoda_browser"]), "Open 2DA Browser", self)
        self._configure_dock_toggle_action(self.twoda_panel_action, "2das", lambda: self._show_workspace_dock("2das"))
        self.resources_panel_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["resource_browser"]), "Open Resource Browser", self)
        self._configure_dock_toggle_action(
            self.resources_panel_action,
            "resources",
            lambda: self._show_workspace_dock("resources"),
        )
        self.module_meshes_panel_action = QtGui.QAction(self._icon(MAIN_ACTION_ICON_KEYS["module_meshes"]), "Open Module Meshes", self)
        self._configure_dock_toggle_action(
            self.module_meshes_panel_action,
            "module_meshes",
            lambda: self._show_workspace_dock("module_meshes"),
        )
        self.mesh_tools_panel_action = QtGui.QAction(self._icon("mesh_tools"), "Open Mesh Tools", self)
        self._configure_dock_toggle_action(
            self.mesh_tools_panel_action,
            "mesh_tools",
            lambda: self._show_workspace_dock("mesh_tools"),
        )
        self.output_log_panel_action = QtGui.QAction(self._icon("output_log"), "Open Output Log", self)
        self._configure_dock_toggle_action(
            self.output_log_panel_action,
            "output_log",
            lambda: self._show_workspace_dock("output_log"),
        )
        self.python_terminal_panel_action = QtGui.QAction(self._icon("python_terminal"), "Open Python Terminal", self)
        self._configure_dock_toggle_action(
            self.python_terminal_panel_action,
            "python_terminal",
            lambda: self._show_workspace_dock("python_terminal"),
        )
        self.sprite_materials_panel_action = QtGui.QAction(self._icon("sprite_materials"), "Open Sprite Materials", self)
        self._configure_dock_toggle_action(
            self.sprite_materials_panel_action,
            "sprite_materials",
            lambda: self._show_workspace_dock("sprite_materials"),
        )
        self.adjust_pivot_panel_action = QtGui.QAction(self._icon("viewport_gimbal"), "Open Adjust Pivot", self)
        self._configure_dock_toggle_action(
            self.adjust_pivot_panel_action,
            "adjust_pivot",
            lambda: self._show_workspace_dock("adjust_pivot"),
        )
        self.set_mdlops_action = QtGui.QAction("Set MDLOps Path...", self)
        self.set_mdlops_action.triggered.connect(self._set_mdlops)
        self.compile_action = QtGui.QAction("Compile ASCII MDL to Binary", self)
        self.compile_action.triggered.connect(self._compile_mdlops)
        self.decompile_action = QtGui.QAction("Decompile Binary MDL", self)
        self.decompile_action.triggered.connect(self._decompile_mdlops)
        self.port_model_action = QtGui.QAction("Port Current Model (K1/K2)...", self)
        self.port_model_action.triggered.connect(self._port_current_model)
        self.generate_module_action = QtGui.QAction("Generate Module Files...", self)
        self.generate_module_action.triggered.connect(self._generate_module_files)
        self.about_module_action = QtGui.QAction("About Map Studio (KMAP Area Authoring)", self)
        self.about_module_action.triggered.connect(self._about_modular)
        self.validate_character_action = QtGui.QAction("Validate Current Character...", self)
        self.validate_character_action.triggered.connect(self._validate_current_character)
        self.ping_scripter_action = QtGui.QAction("Ping Scripting Suite Compatibility (port 7002)...", self)
        self.ping_scripter_action.triggered.connect(lambda: self._ipc_ping("GhostStudio Scripting Suite", 7002))
        self.ping_gmodular_action = QtGui.QAction("Ping GModular (port 7003)...", self)
        self.ping_gmodular_action.triggered.connect(lambda: self._ipc_ping("GModular", 7003))
        self.notify_gmodular_action = QtGui.QAction("Notify GModular: Blueprint Saved...", self)
        self.notify_gmodular_action.triggered.connect(self._ipc_notify_saved)
        self.refresh_gmodular_action = QtGui.QAction("Refresh GModular Viewport", self)
        self.refresh_gmodular_action.triggered.connect(self._ipc_refresh_gmodular)
        self.about_action = QtGui.QAction("About GhostRigger...", self)
        self.about_action.triggered.connect(lambda: show_about(self))

        self.getting_started_action = QtGui.QAction(
            self._icon(MAIN_ACTION_ICON_KEYS["tutorials"]),
            "Tutorials && Getting Started...",
            self,
        )
        self.getting_started_action.setShortcut("F1")
        self.getting_started_action.setStatusTip(
            "Learn each GhostStudio pillar and open the real workspace for the selected tutorial"
        )
        self.getting_started_action.triggered.connect(self._open_getting_started_window)

        self.quit_action = QtGui.QAction("Exit", self)
        self.quit_action.setShortcut("Alt+F4")
        self.quit_action.triggered.connect(self.close)

    def _build_menu(self):
        # Menu structure follows the standard File/Edit/View/Tools/Window/Help
        # convention. The previous Customise/Model/Modules/MDLOps/Retarget/Create/
        # IPC top-level menus were consolidated into the roles below.
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.new_scene_action)
        file_menu.addAction(self.open_scene_action)
        self.recent_scenes_menu = file_menu.addMenu("Recent Scenes")
        self._rebuild_recent_scenes_menu()
        file_menu.addAction(self.save_scene_action)
        file_menu.addAction(self.save_scene_as_action)
        file_menu.addAction(self.close_scene_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_model_action)
        file_menu.addAction(self.clear_model_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_obj_action)
        file_menu.addAction(self.import_fbx_action)
        file_menu.addAction(self.import_gltf_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_ascii_action)
        file_menu.addAction(self.export_scene_action)
        file_menu.addAction(self.export_binary_action)
        file_menu.addAction(self.export_obj_action)
        file_menu.addAction(self.export_fbx_action)
        file_menu.addAction(self.export_selected_fbx_action)
        file_menu.addAction(self.export_full_module_fbx_action)
        file_menu.addAction(self.export_full_module_obj_action)
        file_menu.addAction(self.export_maya_room_module_obj_action)
        file_menu.addAction(self.export_clean_module_obj_action)
        file_menu.addAction(self.export_gltf_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_humanoid_action)
        file_menu.addSeparator()
        file_menu.addAction(self.texture_dir_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        # Edit: replaces the old 'Customise' menu. Undo/Redo are wired to the
        # existing viewport-edit actions as placeholders; a full document
        # undo/redo stack can replace these later.
        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_viewport_action)
        edit_menu.addAction(self.redo_viewport_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.theme_editor_action)

        # View: replaces the old 'Model' menu. Display toggles and viewport
        # navigation live here; rigging/authoring tools moved to Tools.
        view_menu = self.menuBar().addMenu("View")
        for action in (
            self.wire_action,
            self.bones_action,
            self.texture_action,
            self.grid_action,
            None,
            self.frame_all_action,
            self.reset_camera_action,
            None,
            self.uv_action,
            self.info_action,
            self.refresh_action,
        ):
            self._add_menu_action(view_menu, action)

        # Tools: merges the old Tools + Create menus, plus the Retarget, MDLOps
        # and module-authoring entries that were previously separate top-level
        # menus. IPC tooling is parked at the end under a Developer separator.
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(self.autorig_action)
        tools_menu.addAction(self.remove_rig_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.modules_action)
        tools_menu.addAction(self.gui_editor_action)
        tools_menu.addAction(self.placeable_builder_action)
        tools_menu.addAction(self.particle_editor_action)
        tools_menu.addAction(self.scripting_dialogue_studio_action)
        tools_menu.addAction(self.stock_module_editor_action)
        tools_menu.addAction(self.rig_window_action)
        tools_menu.addSeparator()
        retarget_menu = tools_menu.addMenu("Animation Retargeting")
        retarget_menu.addAction(self.load_retarget_source_clip_action)
        retarget_menu.addAction(self.load_retarget_profile_action)
        retarget_menu.addAction(self.preview_retarget_action)
        retarget_menu.addAction(self.export_retarget_preview_action)
        retarget_menu.addSeparator()
        retarget_menu.addAction(self.retarget_workbench_action)
        retarget_menu.addAction(self.unreal_animator_action)
        tools_menu.addSeparator()
        mdlops_menu = tools_menu.addMenu("MDLOps")
        mdlops_menu.addAction(self.set_mdlops_action)
        mdlops_menu.addAction(self.compile_action)
        mdlops_menu.addAction(self.decompile_action)
        tools_menu.addAction(self.port_model_action)
        tools_menu.addAction(self.generate_module_action)
        tools_menu.addSeparator()
        camera_menu = tools_menu.addMenu(self._icon("cameras"), "Camera")
        camera_menu.addAction(self.create_free_camera_action)
        camera_menu.addAction(self.create_target_camera_action)
        camera_menu.addAction(self.create_cinematic_camera_action)
        light_menu = tools_menu.addMenu(self._icon("lights"), "Light")
        light_menu.addAction(self.create_point_light_action)
        light_menu.addAction(self.create_spot_light_action)
        light_menu.addAction(self.create_directional_light_action)
        light_menu.addAction(self.create_area_light_action)
        light_menu.addAction(self.create_ambient_light_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.diag_action)
        tools_menu.addAction(self.fbx_sdk_status_action)
        setup_menu = tools_menu.addMenu("Setup")
        setup_menu.addAction(self.fbx_sdk_setup_action)
        tools_menu.addAction(self.render_frame_action)
        tools_menu.addAction(self.texture_tool_action)
        tools_menu.addAction(self.blueprint_editor_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.character_builder_action)
        tools_menu.addAction(self.head_builder_action)
        tools_menu.addAction(self.validate_character_action)
        tools_menu.addSeparator()
        # Developer / IPC tooling. These low-level diagnostics should be hidden
        # behind a debug flag in the future rather than shipped on the main menu.
        developer_label = QtGui.QAction("Developer", self)
        developer_label.setSeparator(True)
        tools_menu.addAction(developer_label)
        server_action = QtGui.QAction("GhostRigger Server (port 7001) - This Program", self)
        server_action.setEnabled(False)
        tools_menu.addAction(server_action)
        tools_menu.addAction(self.ping_gmodular_action)
        tools_menu.addAction(self.notify_gmodular_action)
        tools_menu.addAction(self.refresh_gmodular_action)
        ipc_info_action = QtGui.QAction("IPC Protocol Info", self)
        ipc_info_action.triggered.connect(lambda: show_ipc_info(self))
        tools_menu.addAction(ipc_info_action)

        # Window: the dock/panel toggles that previously lived under 'Modules'.
        window_menu = self.menuBar().addMenu("Window")
        for action in (
            self.sequence_editor_action,
            None,
            self.content_browser_action,
            self.scene_panel_action,
            self.properties_panel_action,
            self.body_attachment_panel_action,
            self.nodes_panel_action,
            self.lighting_panel_action,
            self.camera_panel_action,
            self.module_meshes_panel_action,
            self.mesh_tools_panel_action,
            self.sprite_materials_panel_action,
            self.adjust_pivot_panel_action,
            self.twoda_panel_action,
            self.resources_panel_action,
            None,
            self.output_log_panel_action,
            self.python_terminal_panel_action,
        ):
            self._add_menu_action(window_menu, action)

        help_menu = self.menuBar().addMenu("Help")
        format_action = QtGui.QAction("KotOR MDL Format Reference", self)
        format_action.triggered.connect(lambda: show_format_reference(self))
        viewport_controls_action = QtGui.QAction("Viewport Navigation Controls", self)
        viewport_controls_action.triggered.connect(lambda: show_viewport_navigation_reference(self))
        help_menu.addAction(self.getting_started_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.about_module_action)
        help_menu.addAction(viewport_controls_action)
        help_menu.addAction(format_action)
        diagnostics_menu = help_menu.addMenu("Diagnostics")
        diagnostics_menu.addAction(self.fbx_sdk_status_action)

    def _build_toolbar(self):
        # The original GhostRigger top chrome is rebuilt as regular Qt widgets
        # so later panels can be swapped in without changing the host frame.
        pass

    def _icon(self, name: str, size: int = 16) -> QtGui.QIcon:
        if hasattr(self, "theme_manager"):
            return self.theme_manager.icon(name, size)
        svg = _GUI_DIR / "icons" / f"{name}.svg"
        if svg.exists():
            return QtGui.QIcon(str(svg))
        path = _GUI_DIR / "icons" / f"{name}_{size}.png"
        if path.exists():
            return QtGui.QIcon(str(path))
        fallback = _GUI_DIR / "icons" / f"{name}_24.png"
        return QtGui.QIcon(str(fallback)) if fallback.exists() else _fallback_icons.icon(name, None, size)

    def _ghost_studio_brand_icon(self, size: int = 36) -> QtGui.QIcon:
        """Return the real GhostStudio application artwork, never an initials tile."""

        root = Path(getattr(self, "app_root", "") or "")
        for candidate in (
            root / "assets" / "icons" / "icon_256x256.png",
            root / "assets" / "icons" / "ghostrigger_icon.png",
        ):
            if candidate.is_file():
                icon = QtGui.QIcon(str(candidate))
                if not icon.isNull():
                    return icon

        # A standalone GhostStudio build carries the same artwork as its
        # native executable ICON resource. QFileIconProvider asks Windows for
        # that real shell icon, so packaged builds do not need a loose PNG.
        try:
            executable = QtCore.QFileInfo(QtCore.QCoreApplication.applicationFilePath())
            icon = QtWidgets.QFileIconProvider().icon(executable)
            if not icon.isNull():
                return icon
        except Exception:
            pass
        return self._icon("logo", size)

    def _placeholder_action(self, text: str, shortcut: str = "") -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda _checked=False, label=text: self._not_migrated(label))
        return action

    def _add_menu_action(self, menu: QtWidgets.QMenu, action: Optional[QtGui.QAction]):
        if action is None:
            menu.addSeparator()
        else:
            menu.addAction(action)

    def _not_migrated(self, label: str):
        self._log(f"{label} is waiting for its Qt panel migration.", "warning")

    def _combined_undo(self) -> None:
        """Global undo: try the viewport stack first, then the scene stack."""
        viewport = getattr(self, "viewport", None)
        viewport_undo = getattr(viewport, "undo", None)
        if callable(viewport_undo):
            try:
                if viewport_undo():
                    self._update_undo_action_state()
                    return
            except Exception:
                pass
        scene_stack = getattr(self, "_scene_undo_stack", None)
        if scene_stack is not None:
            scene_stack.undo()
            self._update_undo_action_state()

    def _combined_redo(self) -> None:
        """Global redo: try the viewport stack first, then the scene stack."""
        viewport = getattr(self, "viewport", None)
        viewport_redo = getattr(viewport, "redo", None)
        if callable(viewport_redo):
            try:
                if viewport_redo():
                    self._update_undo_action_state()
                    return
            except Exception:
                pass
        scene_stack = getattr(self, "_scene_undo_stack", None)
        if scene_stack is not None:
            scene_stack.redo()
            self._update_undo_action_state()

    def _update_undo_action_state(self) -> None:
        """Refresh the enabled state / tooltip of the global undo/redo actions."""
        undo_action = getattr(self, "undo_viewport_action", None)
        redo_action = getattr(self, "redo_viewport_action", None)
        viewport = getattr(self, "viewport", None)
        scene_stack = getattr(self, "_scene_undo_stack", None)

        viewport_can_undo = bool(viewport is not None and getattr(viewport, "_undo_stack", []))
        scene_can_undo = bool(scene_stack is not None and scene_stack.can_undo())
        can_undo = viewport_can_undo or scene_can_undo

        viewport_can_redo = bool(viewport is not None and getattr(viewport, "_redo_stack", []))
        scene_can_redo = bool(scene_stack is not None and scene_stack.can_redo())
        can_redo = viewport_can_redo or scene_can_redo

        if undo_action is not None:
            undo_action.setEnabled(can_undo)
            label = ""
            if scene_stack is not None and scene_stack.undo_text():
                label = f" — {scene_stack.undo_text()}"
            undo_action.setText(f"Undo{label}")
        if redo_action is not None:
            redo_action.setEnabled(can_redo)
            label = ""
            if scene_stack is not None and scene_stack.redo_text():
                label = f" — {scene_stack.redo_text()}"
            redo_action.setText(f"Redo{label}")

    def _connect_scene_undo_signals(self) -> None:
        """Wire SceneUndoStack signals to refresh the undo/redo action state."""
        scene_stack = getattr(self, "_scene_undo_stack", None)
        if scene_stack is None:
            return
        scene_stack.can_undo_changed.connect(self._update_undo_action_state)
        scene_stack.can_redo_changed.connect(self._update_undo_action_state)

    def _make_header(self) -> QtWidgets.QFrame:
        header = QtMatrixPanel(engine=self._matrix_engine, opacity=0.55)
        header.setObjectName("HeaderBar")
        header.setFixedHeight(58)
        self.header_bar = header

        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(18, 7, 18, 7)
        layout.setSpacing(10)

        logo = QtWidgets.QLabel()
        logo.setObjectName("GhostStudioBrandIcon")
        logo.setFixedSize(38, 38)
        logo.setAlignment(QtCore.Qt.AlignCenter)
        brand_icon = self._ghost_studio_brand_icon(36)
        pix = brand_icon.pixmap(36, 36)
        if not pix.isNull():
            logo.setPixmap(pix)
            if self.windowIcon().isNull():
                self.setWindowIcon(brand_icon)
        else:
            logo.setText("GS")
            logo.setStyleSheet(f"color:{C['accent']}; font-weight:bold; font-size:16pt;")
        logo.setStyleSheet(logo.styleSheet() + "background:transparent;")
        layout.addWidget(logo)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        title = QtMatrixLabel("GHOST STUDIO")
        title.setObjectName("GhostTitle")
        subtitle = QtWidgets.QLabel("Odyssey Engine Pipeline  //  KotOR 1 & 2 TSL")
        subtitle.setObjectName("GhostSubtitle")
        subtitle.setStyleSheet("background:transparent;")
        self.header_title = title
        self.header_subtitle = subtitle
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch(1)

        meta_box = QtWidgets.QVBoxLayout()
        meta_box.setContentsMargins(0, 0, 0, 0)
        meta_box.setSpacing(2)
        self.metrics_label = QtWidgets.QLabel("")
        self.metrics_label.setObjectName("HeaderMeta")
        self.version_label = QtWidgets.QLabel(f"v{self.APP_VERSION}")
        self.version_label.setObjectName("HeaderMeta")
        self.ipc_label = QtWidgets.QLabel("IPC: port 7001 *")
        self.ipc_label.setObjectName("HeaderIpcMeta")
        for label in (self.metrics_label, self.version_label, self.ipc_label):
            label.setAlignment(QtCore.Qt.AlignRight)
            label.setStyleSheet(label.styleSheet() + "background:transparent;")
            meta_box.addWidget(label)
        layout.addLayout(meta_box)
        self._apply_matrix_bar_config(header)
        return header

    def _make_command_bar(self) -> QtWidgets.QWidget:
        host = QtWidgets.QFrame()
        host.setObjectName("CommandBarHost")
        host.setMinimumHeight(36)
        self.command_bar_host = host

        host_layout = QtWidgets.QHBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)

        bar = QtWidgets.QFrame(host)
        bar.setObjectName("CommandBar")
        bar.setMinimumHeight(36)
        self.command_bar = bar

        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(5)

        self.model_pill = QtWidgets.QLabel("// Untitled Scene")
        self.model_pill.setObjectName("ModelPill")
        self.model_pill.setMinimumWidth(170)
        self.model_pill.setMaximumWidth(260)
        self.model_pill.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.model_pill.setToolTip("Active KMAX scene.")
        layout.addWidget(self.model_pill, 0, QtCore.Qt.AlignVCenter)

        layout.addWidget(self._tool_button("New Scene  Ctrl+N", self.new_scene_action, "new_scene"))
        layout.addWidget(self._tool_button("Open Scene  Ctrl+O", self.open_scene_action, "open"))
        layout.addWidget(self._tool_button("Save  Ctrl+S", self.save_scene_action, "save"))
        layout.addWidget(self._tool_button("Auto-Rig  Ctrl+Shift+G", self.autorig_action, "autorig"))
        layout.addWidget(self._tool_button("Character Builder", self.character_builder_action, MAIN_COMMAND_STRIP_ICON_KEYS["character_builder"]))
        head_builder_button = self._tool_button(
            "Head Builder",
            self.head_builder_action,
            MAIN_COMMAND_STRIP_ICON_KEYS["character_builder"],
        )
        head_builder_button.setObjectName("CommandStripHeadBuilderButton")
        head_builder_button.setToolTip(
            "Open the Custom KOTOR Head Builder in the reusable Character Studio window"
        )
        layout.addWidget(head_builder_button)
        map_studio_button = self._tool_button("Map Studio", self.modules_action, MAIN_COMMAND_STRIP_ICON_KEYS["map_studio"])
        map_studio_button.setObjectName("CommandStripMapStudioButton")
        map_studio_button.setToolTip("Open Map Studio (KMAP Area Authoring)")
        layout.addWidget(map_studio_button)
        gui_editor_button = self._tool_button(
            "GUI Editor",
            self.gui_editor_action,
            MAIN_COMMAND_STRIP_ICON_KEYS["gui_editor"],
        )
        gui_editor_button.setObjectName("CommandStripGuiEditorButton")
        gui_editor_button.setToolTip("Open the standalone GUI Editor for KOTOR .gui resources and PIE HUD previews")
        layout.addWidget(gui_editor_button)
        placeable_builder_button = self._tool_button(
            "Placeables", self.placeable_builder_action, MAIN_COMMAND_STRIP_ICON_KEYS["placeable_builder"]
        )
        placeable_builder_button.setObjectName("CommandStripPlaceableBuilderButton")
        placeable_builder_button.setToolTip(
            "Open Placeable Builder and author reusable objects for the Map Studio Placeable Library"
        )
        layout.addWidget(placeable_builder_button)
        particle_editor_button = self._tool_button(
            "Particles", self.particle_editor_action, MAIN_COMMAND_STRIP_ICON_KEYS["particle_editor"]
        )
        particle_editor_button.setObjectName("CommandStripParticleEditorButton")
        particle_editor_button.setToolTip(
            "Open the Particle Editor: live emitter editing plus the K1/K2 retail emitter template library"
        )
        layout.addWidget(particle_editor_button)
        scripting_studio_button = self._tool_button(
            "Scripting Suite",
            self.scripting_dialogue_studio_action,
            MAIN_COMMAND_STRIP_ICON_KEYS["scripting_dialogue_studio"],
        )
        scripting_studio_button.setObjectName("CommandStripScriptingDialogueStudioButton")
        scripting_studio_button.setToolTip(
            "Open the complete Scripting Suite for narrative authoring, projects, packaging, and Map Studio handoff"
        )
        layout.addWidget(scripting_studio_button)
        module_editor_button = self._tool_button("Module Editor", self.stock_module_editor_action, MAIN_COMMAND_STRIP_ICON_KEYS["module_editor"])
        module_editor_button.setObjectName("CommandStripModuleEditorButton")
        module_editor_button.setToolTip("Open Module Editor (Stock MOD/RIM Patcher)")
        layout.addWidget(module_editor_button)
        layout.addWidget(self._tool_button("Tex Dir", self.texture_dir_action, "texture"))
        layout.addWidget(self._tool_button("Settings  Ctrl+Comma", self.settings_action, "settings", compact=True))

        import_button = self._menu_button("Import", "import", [
            self.import_obj_action,
            self.import_fbx_action,
            self.import_gltf_action,
            None,
            self.open_ascii_action,
        ])
        export_button = self._menu_button("Export", "export", [
            self.export_binary_action,
            self.export_obj_action,
            self.export_fbx_action,
            self.export_gltf_action,
            None,
            self.export_humanoid_action,
            None,
            self.save_ascii_action,
            self.compile_action,
        ])
        camera_create_button = self._menu_button("Create Camera", "camera_cinematic", [
            self.create_free_camera_action,
            self.create_target_camera_action,
            self.create_cinematic_camera_action,
        ])
        light_create_button = self._menu_button("Create Light", "light_point", [
            self.create_point_light_action,
            self.create_spot_light_action,
            self.create_directional_light_action,
            self.create_area_light_action,
            self.create_ambient_light_action,
        ])
        layout.addWidget(import_button)
        layout.addWidget(export_button)
        layout.addWidget(camera_create_button)
        layout.addWidget(light_create_button)

        layout.addStretch(1)
        layout.addWidget(self._tool_button("Content", self.content_browser_action, MAIN_COMMAND_STRIP_ICON_KEYS["content_browser"], compact=True))
        layout.addWidget(self._tool_button("Scene Information", self.scene_panel_action, "scene", compact=True))
        layout.addWidget(self._tool_button("Properties", self.properties_panel_action, "props", compact=True))
        layout.addWidget(self._tool_button("BAS", self.body_attachment_panel_action, "body_attachment", compact=True))
        layout.addWidget(self._tool_button("Sequence Editor", self.sequence_editor_action, "sequence", compact=True))
        layout.addWidget(self._tool_button("Animation Browser", self.animation_browser_dock_action, MAIN_COMMAND_STRIP_ICON_KEYS["animation_browser"], compact=True))
        layout.addWidget(self._tool_button("Nodes", self.nodes_panel_action, MAIN_COMMAND_STRIP_ICON_KEYS["nodes"], compact=True))
        layout.addWidget(self._tool_button("Lighting", self.lighting_panel_action, "lights", compact=True))
        layout.addWidget(self._tool_button("Cameras", self.camera_panel_action, "cameras", compact=True))
        layout.addWidget(self._tool_button("Module Meshes", self.module_meshes_panel_action, MAIN_COMMAND_STRIP_ICON_KEYS["module_meshes"], compact=True))
        layout.addWidget(self._tool_button("Mesh Tools", self.mesh_tools_panel_action, "mesh_tools", compact=True))
        layout.addWidget(self._tool_button("Sprite Materials", self.sprite_materials_panel_action, "sprite_materials", compact=True))
        layout.addWidget(self._tool_button("Adjust Pivot", self.adjust_pivot_panel_action, "viewport_gimbal", compact=True))
        layout.addWidget(self._tool_button("2DA Browser", self.twoda_panel_action, MAIN_COMMAND_STRIP_ICON_KEYS["twoda_browser"], compact=True))
        layout.addWidget(self._tool_button("Resource Browser", self.resources_panel_action, MAIN_COMMAND_STRIP_ICON_KEYS["resource_browser"], compact=True))
        layout.addWidget(self._tool_button("Log", self.output_log_panel_action, "output_log", compact=True))
        layout.addWidget(self._tool_button("Terminal", self.python_terminal_panel_action, "python_terminal", compact=True))
        layout.addWidget(self._tool_button("Diagnostics  Ctrl+Shift+D", self.diag_action, "diag", compact=True))

        self.visual_profile_combo = QtWidgets.QComboBox()
        self.visual_profile_combo.setObjectName("VisualProfileCombo")
        self.visual_profile_combo.setToolTip("Apply a saved visual profile layout.")
        self.visual_profile_combo.setMinimumWidth(170)
        self._populate_visual_profile_combo()
        self.visual_profile_combo.currentIndexChanged.connect(self._on_visual_profile_selected)
        bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.workspace_switcher = WorkspaceSwitcher(self, current=load_saved_workspace_preset())
        self.workspace_switcher.presetSelected.connect(self.apply_workspace)
        host_layout.addWidget(self.workspace_switcher, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        host_layout.addWidget(bar, 1)
        host_layout.addWidget(self.visual_profile_combo, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return host

    def _populate_visual_profile_combo(self) -> None:
        combo = getattr(self, "visual_profile_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            default_layout = self.layout_manager.get_layout("default")
            combo.addItem(default_layout.name or "Default", "default")
            for layout in self.layout_manager.available_layouts():
                if layout.id == "default":
                    continue
                combo.addItem(layout.name, layout.id)
            selected = self.layout_manager.settings.selected_layout
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _on_visual_profile_selected(self, _index: int) -> None:
        combo = getattr(self, "visual_profile_combo", None)
        if combo is None:
            return
        layout_id = str(combo.currentData() or "default")
        current = self.layout_manager.settings.selected_layout
        if layout_id == current:
            return
        self.layout_manager.select_layout(layout_id, window=self)
        self._sync_theme_layout_settings()
        try:
            save_settings(self.settings_path, self.settings_data)
        except Exception as exc:
            self._log(f"Could not save visual profile: {exc}", "warning")
        layout = self.layout_manager.get_layout(layout_id)
        self._log(f"Visual profile applied: {layout.name}", "success")

    def _make_viewport_toolbar_band(self, toolbar: QtWidgets.QWidget | None) -> QtWidgets.QWidget | None:
        if toolbar is None:
            return None
        band = QtWidgets.QFrame()
        band.setObjectName("ViewportToolbarBand")
        band.setFrameShape(QtWidgets.QFrame.StyledPanel)
        band.setLineWidth(1)
        band.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        root = QtWidgets.QVBoxLayout(band)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        row_host = QtWidgets.QWidget(band)
        row_host.setObjectName("ViewportToolbarDefaultRow")
        row = QtWidgets.QHBoxLayout(row_host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        toolbar.setParent(row_host)
        toolbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        row.addWidget(toolbar, 1)
        root.addWidget(row_host)
        self.viewport_toolbar_band = band
        self.viewport_toolbar_default_row = row_host
        self.viewport_toolbar_hosted_scroll = toolbar
        self._sync_viewport_toolbar_band()
        return band

    def _sync_viewport_toolbar_band(self) -> None:
        toolbar = getattr(self, "viewport_toolbar_hosted_scroll", None)
        band = getattr(self, "viewport_toolbar_band", None)
        if toolbar is None or band is None:
            return
        height = self._height_for_wrapping_widget(toolbar, max(28, toolbar.sizeHint().height()))
        toolbar.setMinimumHeight(height)
        toolbar.setMaximumHeight(height)
        row_host = getattr(self, "viewport_toolbar_default_row", None)
        if row_host is not None:
            row_host.setMinimumHeight(height)
            row_host.setMaximumHeight(height)
        band.setFixedHeight(height)
        band.updateGeometry()

    def _sync_reserved_top_rows(self) -> None:
        command_bar = getattr(self, "command_bar", None)
        if command_bar is not None:
            height = self._height_for_wrapping_widget(command_bar, max(36, command_bar.sizeHint().height()))
            command_bar.setMinimumHeight(height)
            command_bar.setMaximumHeight(height)
            host = getattr(self, "command_bar_host", None)
            if host is not None:
                host_height = max(36, height, host.sizeHint().height())
                host.setMinimumHeight(host_height)
                host.setMaximumHeight(host_height)
        self._sync_viewport_toolbar_band()
        top_shell = getattr(self, "reserved_top_shell", None)
        if top_shell is not None:
            shell_height = max(1, top_shell.sizeHint().height())
            top_shell.setMinimumHeight(shell_height)
            top_shell.setMaximumHeight(shell_height)
            top_shell.updateGeometry()
        top_toolbar = getattr(self, "reserved_top_toolbar", None)
        if top_toolbar is not None:
            toolbar_height = max(1, top_toolbar.sizeHint().height(), getattr(top_shell, "minimumHeight", lambda: 0)())
            top_toolbar.setMinimumHeight(toolbar_height)
            top_toolbar.setMaximumHeight(toolbar_height)
            top_toolbar.updateGeometry()

    @staticmethod
    def _height_for_wrapping_widget(widget: QtWidgets.QWidget, fallback: int) -> int:
        layout = widget.layout()
        width = max(1, widget.width())
        if layout is not None and layout.hasHeightForWidth():
            return max(1, layout.heightForWidth(width))
        return max(1, int(fallback))

    def _matrix_bar_settings(self, theme=None) -> dict:
        legacy = dict(self.settings_data.get("matrix_bar", {}))
        style = ""
        glyphs = ""
        font_family = ""
        image_path = ""
        if theme is not None:
            styles = getattr(theme, "styles", {})
            style = str(styles.get("matrixBar.style", "") or "")
            glyphs = str(styles.get("matrixBar.glyphs", "") or "")
            font_family = str(styles.get("matrixBar.fontFamily", "") or "")
            image_path = str(styles.get("matrixBar.imagePath", "") or "")
            try:
                crop = (
                    float(styles.get("matrixBar.cropX", 0) or 0),
                    float(styles.get("matrixBar.cropY", 0) or 0),
                    float(styles.get("matrixBar.cropW", 100) or 100),
                    float(styles.get("matrixBar.cropH", 100) or 100),
                )
            except (TypeError, ValueError):
                crop = (0.0, 0.0, 100.0, 100.0)
        else:
            crop = (0.0, 0.0, 100.0, 100.0)
        if not style:
            style = str(legacy.get("style") or ("matrix" if self.settings_data.get("matrix_background", True) else "disabled"))
        return {
            "style": style,
            "glyphs": glyphs or str(legacy.get("glyphs") or ""),
            "font_family": font_family or str(legacy.get("font_family") or ""),
            "image_path": image_path or str(legacy.get("image_path") or ""),
            "crop": crop,
        }

    def _apply_matrix_bar_config(self, panel: QtMatrixPanel, theme=None) -> None:
        cfg = self._matrix_bar_settings(theme)
        style = str(cfg.get("style") or "matrix")
        panel.set_matrix_config(
            style=style,
            glyphs=str(cfg.get("glyphs") or ""),
            font_family=str(cfg.get("font_family") or ""),
            image_path=str(cfg.get("image_path") or ""),
            crop=cfg.get("crop"),
        )

    def _apply_matrix_theme(self, theme) -> None:
        for panel in (getattr(self, "header_bar", None),):
            if panel is not None:
                self._apply_matrix_bar_config(panel, theme)
                panel.apply_ghost_theme(theme)
        if hasattr(self, "header_subtitle"):
            self.header_subtitle.setStyleSheet(
                f"color:{theme.color('matrixBar.subtext', theme.color('text.secondary'))}; background:transparent;"
            )
        for label in (getattr(self, "metrics_label", None), getattr(self, "version_label", None)):
            if label is not None:
                label.setStyleSheet(
                    f"color:{theme.color('matrixBar.metaText', theme.color('text.secondary'))}; background:transparent;"
                )
        if hasattr(self, "ipc_label"):
            self.ipc_label.setStyleSheet(
                f"color:{theme.color('matrixBar.ipcText', theme.color('accent.primary'))}; background:transparent; font-size:7pt;"
            )

    def _tool_button(
        self,
        text: str,
        action: QtGui.QAction,
        icon_name: str = "",
        accent: bool = False,
        compact: bool = False,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("CommandStripButton")
        display_text = self._command_button_label(text, action)
        button.setText("")
        button.setProperty("_gr_full_text", display_text)
        button.setProperty("_gr_ignore_layout_button_mode", True)
        if icon_name:
            button.setIcon(self._icon(icon_name, 18))
        button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        button.setIconSize(QtCore.QSize(18, 18))
        button.setFixedSize(30, 22)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        button.setProperty("accent", accent)
        button.setProperty("compact", compact)
        if action.isCheckable():
            button.setCheckable(True)
            button.setChecked(action.isChecked())
            action.toggled.connect(button.setChecked)
        button.clicked.connect(action.trigger)
        if action.shortcut():
            button.setToolTip(f"{action.text()} ({action.shortcut().toString()})")
        else:
            button.setToolTip(action.text())
        return button

    @staticmethod
    def _command_button_label(text: str, action: QtGui.QAction) -> str:
        label = str(text or action.text()).strip()
        if "  " in label:
            label = label.split("  ", 1)[0].strip()
        shortcut = action.shortcut().toString() if action.shortcut() else ""
        if shortcut and label.endswith(shortcut):
            label = label[: -len(shortcut)].strip()
        return label or str(action.text()).strip()

    def _menu_button(
        self,
        text: str,
        icon_name: str,
        actions: list[Optional[QtGui.QAction]],
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("CommandStripMenuButton")
        button.setText("")
        button.setProperty("_gr_full_text", text)
        button.setProperty("_gr_ignore_layout_button_mode", True)
        button.setIcon(self._icon(icon_name, 18))
        button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        button.setIconSize(QtCore.QSize(18, 18))
        button.setFixedSize(34, 22)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setToolTip(text)
        menu = QtWidgets.QMenu(button)
        for action in actions:
            self._add_menu_action(menu, action)
        button.setMenu(menu)
        return button

    def _separator(self) -> QtWidgets.QFrame:
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Plain)
        sep.setStyleSheet(f"color:{C['border']}; background:{C['border']};")
        sep.setFixedWidth(1)
        return sep
