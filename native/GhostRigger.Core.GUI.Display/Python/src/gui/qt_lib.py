"""Central Qt GUI library facade.

The implementation modules live in category folders under ``src.gui``. Import
through this module when code outside those folders needs GUI pieces, for
example ``src.gui.qt_lib.panels.qt_common_panels``.
"""

from __future__ import annotations

from importlib import import_module, reload
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import Iterable
import sys


_GROUPS: dict[str, tuple[str, ...]] = {
    "assets": (
        "qt_icon_manager",
        "qt_matrix_background",
        "qt_theme",
    ),
    "dialogs": (
        "unreal_conversion_dialog",
        "unreal_level_import_dialog",
        "add_model_to_scene_dialog",
        "fbx_sdk_setup_dialog",
        "qt_render_frame_dialog",
        "qt_dialogs",
        "qt_export_dialog",
        "qt_fbx_animation_selection_dialog",
        "qt_getting_started_window",
        "qt_lightmap_baker_dialog",
        "qt_settings_dialog",
        "lightmap_preview_window",
        "uv_geometry_preview_window",
        "uv_preview_window",
    ),
    "gizmo": (
        "gizmo_draw_data",
        "gizmo_mode",
        "gizmo_picker",
        "gizmo_renderer",
        "transform_controller",
        "transform_gizmo",
    ),
    "integration": (
        "editor_services",
        "tool_integration_registry",
    ),
    "camera": (
        "arcball_camera",
        "camera_controller",
        "camera_gizmo_renderer",
        "camera_manager",
        "camera_model",
        "camera_overlays",
        "camera_picker",
        "camera_presets",
        "camera_render_settings",
        "camera_rig",
        "camera_selection",
        "camera_target",
        "camera_viewport_adapter",
        "frame_renderer",
        "render_manifest",
        "render_output",
    ),
    "controllers": (
        "head_builder_controller",
    ),
    "lighting": (
        "aurora_light_adapter",
        "light_export_bridge",
        "light_gizmo_renderer",
        "light_grouping",
        "light_manager",
        "light_model",
        "light_picker",
        "render_data",
        "light_selection",
        "light_types",
        "lighting_rig_presets",
        "lighting_viewport_controller",
        "lightmap_controller",
        "lightmap_export_bridge",
        "lightmap_gpu_solver",
        "lightmap_bake_job",
        "lightmap_bake_settings",
        "lightmap_bake_worker",
        "lightmap_baker",
        "lightmap_denoiser",
        "lightmap_lighting_solver",
        "lightmap_manifest",
        "lightmap_output",
        "lightmap_padding",
        "lightmap_compare",
        "lightmap_sampler",
        "lightmap_rasterizer",
        "preview_cache",
        "raycast_backend",
        "lightmap_shadow_solver",
        "lightmap_uv_validator",
        "uv_atlas_generator",
        "uv_channel_info",
        "material_map_controller",
        "settings",
        "shader_complexity",
    ),
    "libtheme": (
        "font_manager",
        "icon_manager",
        "layout_applier",
        "layout_loader",
        "layout_manager",
        "layout_model",
        "layout_validator",
        "os_theme_detector",
        "qt_stylesheet_builder",
        "style_tokens",
        "theme_applier",
        "theme_loader",
        "theme_manager",
        "theme_model",
        "theme_registry",
        "theme_settings",
        "theme_validator",
        "theme_watcher",
    ),
    "panels": (
        "unreal_pipeline_panel",
        "qt_animation_panel",
        "qt_content_browser_panel",
        "adjust_pivot_panel",
        "axis_mode_control",
        "qt_bottom_strip",
        "qt_body_attachment_panel",
        "qt_character_builder_panel",
        "qt_common_panels",
        "qt_head_builder_workspace",
        "qt_camera_panel",
        "qt_diagnostics_panel",
        "qt_inspector_panel",
        "qt_library_panel",
        "qt_lighting_panel",
        "qt_log_panel",
        "qt_mesh_operation_options",
        "qt_mesh_selection_toolbar",
        "qt_mesh_tools_panel",
        "qt_modular_panel",
        "qt_normal_map_panel",
        "qt_properties_panel",
        "qt_resource_browser_model",
        "qt_resource_panel",
        "qt_rig_panel",
        "qt_scene_outliner_panel",
        "qt_skeleton_panel",
        "qt_sprite_material_panel",
        "qt_texture_panel",
        "qt_ue5_rig_export_panel",
        "qt_workflow_rail",
    ),
    "rendering": (
        "accel",
        "gpu_renderer",
        "hardware_info",
        "mesh_render_data",
        "direct3d_renderer",
        "moderngl_renderer",
        "null_renderer",
        "picking",
        "qt_accel",
        "qt_gpu_renderer",
        "renderer_backend",
        "renderer_capabilities",
        "renderer_factory",
        "renderer_interface",
        "renderer_performance",
        "renderer_profiler",
        "renderer_settings",
        "skeleton_render_data",
        "wgpu_renderer",
    ),
    "sequence_editor": (
        "sequence_curve_editor",
        "sequence_dopesheet",
        "sequence_editor_window",
        "sequence_outliner",
        "sequence_property_panel",
        "sequence_timeline_widget",
        "sequence_toolbar",
        "sequence_track_list_widget",
        "sequence_transport_bar",
        "sequence_viewport_panel",
    ),
    "textures": (
        "qt_tex_atlas",
        "qt_tpc_render_utils",
        "tex_atlas",
        "tpc",
        "tpc_render_utils",
        "txi",
    ),
    "viewports": (
        "qt_transform_typein_bar",
        "qt_uv_viewer",
        "qt_viewport",
        "viewport_display",
        "viewport_host",
        "viewport_navigation",
        "viewcube",
    ),
    "windows": (
        "module_editor_window",
        "progress_toast",
        "qt_blueprint_editor",
        "qt_character_builder_window",
        "qt_character_builder_mode_selector",
        "qt_custom_rigged_character_builder_window",
        "qt_custom_rigged_character_builder_controller",
        "qt_gui_editor_window",
        "qt_main_window",
        "qt_particle_editor",
        "qt_placeable_builder",
        "qt_placeable_builder_controller",
        "qt_retarget_preview_controller",
        "qt_retarget_workbench_controller",
        "qt_retarget_window",
        "qt_scripting_dialogue_studio",
        "qt_scripting_blueprint_page",
        "qt_scripting_data_pages",
        "qt_scripting_integrated_tools_page",
        "qt_scripting_project_package_pages",
        "qt_scripting_quest_builder_page",
        "qt_scripting_reference_page",
        "qt_scripting_tutorial_page",
        "qt_source_clip_preview_model",
        "qt_unreal_animator",
    ),
}

_ALIASES: dict[str, str] = {}


class _AliasLoader(Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return sys.modules.get(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        loaded = import_module(self.target)
        if getattr(module, "_loaded", None) is not None:
            loaded = reload(loaded)
        module.__dict__["_loaded"] = loaded
        module.__dict__["_target"] = self.target
        for name, value in loaded.__dict__.items():
            if name not in {"__name__", "__package__", "__spec__", "__loader__"}:
                module.__dict__[name] = value


class _AliasFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        alias_target = _ALIASES.get(fullname)
        if alias_target is None:
            return None
        return spec_from_loader(fullname, _AliasLoader(alias_target))


class _LazyModule(ModuleType):
    """Module alias that imports its grouped implementation on first use."""

    def __init__(self, alias: str, target: str) -> None:
        super().__init__(alias)
        self.__dict__["_target"] = target
        self.__dict__["_loaded"] = None

    def _load(self) -> ModuleType:
        loaded = self.__dict__.get("_loaded")
        if loaded is None:
            loaded = import_module(self.__dict__["_target"])
            self.__dict__["_loaded"] = loaded
            for name, value in loaded.__dict__.items():
                if name not in {"__name__", "__package__", "__spec__", "__loader__"}:
                    self.__dict__[name] = value
        return loaded

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))


def _make_package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = []  # type: ignore[attr-defined]
    return package


def _register_alias(alias: str, target: str) -> _LazyModule:
    _ALIASES[alias] = target
    module = _LazyModule(alias, target)
    module.__package__ = alias.rpartition(".")[0]
    module.__loader__ = _AliasLoader(target)
    module.__spec__ = spec_from_loader(alias, module.__loader__)
    sys.modules[alias] = module
    return module


def _register_group(group: str, modules: Iterable[str]) -> ModuleType:
    package_name = f"{__name__}.{group}"
    package = _make_package(package_name)
    sys.modules[package_name] = package
    globals()[group] = package

    for module_name in modules:
        alias = f"{package_name}.{module_name}"
        target = f"src.gui.{group}.{module_name}"
        module = _register_alias(alias, target)
        setattr(package, module_name, module)
        globals().setdefault(module_name, module)

    package.__all__ = list(modules)  # type: ignore[attr-defined]
    return package


__path__ = []  # type: ignore[var-annotated]

if not any(isinstance(_finder, _AliasFinder) for _finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

for _group, _modules in _GROUPS.items():
    _register_group(_group, _modules)

__all__ = [*_GROUPS.keys()]
