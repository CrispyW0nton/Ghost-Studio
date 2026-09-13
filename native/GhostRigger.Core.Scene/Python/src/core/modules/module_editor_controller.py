"""Controller coordinating KMAP project state and module-editor services."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from src.core.level import (
    KMapProject,
    KMapSerializer,
    KMapValidator,
    LevelExportBridge,
    LevelExportOptions,
    LevelScene,
    LevelTransform,
    MapStudioTextureSidecarJournal,
    MapStudioTextureSidecarPatch,
    clone_game_texture_asset,
    create_project_texture_asset,
    create_project_tpc_texture_asset,
    import_project_texture_asset,
    managed_project_texture_sidecars,
    new_kmap_project,
    project_texture_directory,
    project_texture_export_resources,
    save_texture_paint_session,
    tga_dirty_tile_byte_ranges,
)
from src.core.scene.module_scene_import import resolve_module_room_placement
from src.io.obj_room_document import load_obj_room_document
from src.io.skybox_panorama_conversion import (
    PanoramaSkyboxOptions,
    SKYBOX_PANORAMA_FACE_ORDER,
    load_and_convert_equirectangular_panorama,
)

from .module_blueprint_service import ModuleBlueprintService
from .module_builder_service import ModuleBuilderService
from .module_editor_model import MapStudioWorkspaceMode, ModuleEditorModel
from .authored_module_export import (
    AuthoredModuleExportRequest,
    AuthoredModuleGameProofRequest,
    AuthoredModuleGameProofResult,
    AuthoredModuleInstallPrepRequest,
    export_authored_module_project,
    prepare_authored_module_install,
    record_authored_module_game_proof,
)
from .authored_module_kmap_bridge import (
    TEXTURE_PAINT_UNAPPLIED_BLOCKER,
    authored_project_from_kmap_payload,
    authored_project_to_kmap_payload,
    build_kmap_authored_module_readiness,
    create_dev_test_authored_module_payload,
    create_golden_test_authored_module_payload,
    texture_paint_has_unapplied_changes,
    texture_paint_pending_resrefs,
)
from .authored_module_preview_model import build_authored_module_preview_model
from .authored_primitive_polygon_cages import (
    build_authored_primitive_polygon_cage,
    logical_topology_counts,
)
from .authored_room_composition import AuthoredRoomComposition, primitive_to_mesh
from .authored_primitive_topology_policy import (
    PrimitivePreviewDeferred,
    PrimitiveTopologySafetyError,
    enforce_primitive_topology_budget,
)
from .map_studio_pie import build_map_studio_pie_session
from .map_studio_pie_dialogue import (
    MapStudioPIEDialogueContextEvaluator,
    MapStudioPIEDialogueSession,
    build_map_studio_pie_dialogue_catalog,
)
from .map_studio_pie_entities import build_pie_entity_registry
from .map_studio_stock_content_preview import (
    RES_DLG,
    RES_UTC,
    RES_UTD,
    RES_UTI,
    RES_UTM,
    RES_UTP,
    RES_UTT,
    TemplateModelResolver,
    build_map_studio_combined_preview_model,
    load_kotor_model_from_bytes,
    load_stock_kotor_model,
)
from .map_studio_pie_resources import inspect_map_studio_pie_resource
from .stock_module_importer import import_stock_module
from .authored_obj_room_import import (
    ObjRoomAuthoringOptions,
    analyze_obj_walkmesh_candidate_components,
    apply_obj_walkmesh_component_review,
    build_obj_room_primitive,
    generate_adaptive_obj_walkmesh,
    suggested_obj_texture_resref,
)
from .authored_imported_mesh import (
    ImportedMeshRoomPrimitive,
    ImportedMeshSurface,
    append_imported_mesh_quad,
    bend_imported_mesh_vertices,
    boolean_difference_imported_mesh_surfaces,
    imported_mesh_surface_role,
    bevel_imported_mesh_edge,
    bevel_imported_mesh_edges,
    bridge_imported_mesh_border_edges,
    build_imported_mesh_primitive_from_stock_model,
    extract_imported_mesh_room_bounds,
    fill_imported_wok_from_floor_surfaces,
    generate_room_walkmesh_from_geometry,
    imported_mesh_room_is_backdrop,
    prepare_imported_mesh_for_static_runtime_rebuild,
    prepare_imported_mesh_walkmesh_generation_intent,
    collapse_imported_mesh_edge,
    delete_imported_mesh_edge_faces,
    delete_imported_mesh_faces,
    delete_imported_mesh_vertex_faces,
    extrude_imported_mesh_faces,
    flatten_imported_mesh_faces,
    flip_imported_mesh_faces,
    inset_imported_mesh_faces,
    extrude_imported_mesh_edge,
    move_imported_mesh_edge,
    move_imported_mesh_faces,
    move_imported_mesh_vertex,
    connect_imported_mesh_vertices,
    merge_imported_mesh_components,
    fill_imported_mesh_boundary_loop,
    harden_imported_mesh_edges,
    insert_imported_mesh_edge_loop,
    lattice_deform_imported_mesh_vertices,
    make_hole_in_imported_mesh_face,
    mirror_imported_mesh_geometry,
    set_imported_mesh_face_texture,
    set_imported_mesh_face_smoothing,
    soften_imported_mesh_edges,
    split_imported_mesh_edge,
    split_imported_mesh_face,
    split_imported_mesh_face_at_point,
    shrink_wrap_imported_mesh_vertices,
    weld_imported_mesh_vertex,
    weld_imported_mesh_vertices,
    wrap_deform_imported_mesh_vertices,
)
from .authored_module_project import AuthoredRoomSpec, authored_resref_blocking_issue, compile_authored_room_spec, normalise_resref
from .authored_module_layout import (
    AuthoredRoomConnectionAudit,
    AuthoredRoomOpeningIntentUpdate,
    audit_authored_room_connections,
    auto_arrange_authored_rooms as auto_arrange_authored_rooms_in_project,
    connect_authored_room_drag_snap as connect_authored_room_drag_snap_in_project,
    connect_authored_room_openings as connect_authored_room_openings_in_project,
    preview_authored_room_drag_snap as preview_authored_room_drag_snap_in_project,
    set_authored_room_opening_intent as set_authored_room_opening_intent_in_project,
    snap_authored_rooms_to_grid as snap_authored_rooms_to_grid_in_project,
)
from .authored_module_validation_projection import authored_module_readiness_validation_issues
from .authored_gameplay_palette import authored_gameplay_palette_from_library_rows
from .map_studio_modeling_tools import (
    available_map_studio_component_modes,
    available_map_studio_edit_mode_contexts,
    available_map_studio_modeling_tools,
    available_map_studio_snap_modes,
    available_map_studio_terrain_brushes,
    available_map_studio_tool_belt_actions,
    available_map_studio_tool_belt_presets,
    map_studio_edit_mode_context,
    map_studio_modeling_tool_summary,
    map_studio_tool_command_search,
    map_studio_tool_belt_actions_for_preset,
    map_studio_viewport_performance_policy,
)
from .map_studio_multi_cut import MultiCutAnchor, MultiCutSession, MultiCutSettings
from .map_studio_export_objects import map_studio_export_object_boundaries
from .map_studio_terrain_sculpt_session import (
    MapStudioTerrainSculptApplyResult,
    begin_terrain_sculpt_stroke,
    prepare_terrain_sculpt_frame_for_project,
)
from .map_studio_terrain_kit import (
    build_terrain_kit_preview_model,
    build_terrain_kit_primitive,
    module_transition_asset_for_profiles,
    scan_vanilla_terrain_kit_assets,
    terrain_kit_asset,
    terrain_kit_asset_rows,
    terrain_kit_runtime_resources,
    transform_terrain_kit_primitive,
    vanilla_terrain_kit_assets,
    write_vanilla_terrain_kit_catalog,
)
from .map_studio_environment_kits import (
    EnvironmentKitWallTarget,
    audit_environment_kit_room_occupancy,
    environment_kit_builder_style_id,
    environment_kit_collection_rows,
    environment_kit_piece,
    environment_kit_piece_rows,
    environment_kit_source_room_position,
    generated_environment_kit_boundary_magnets,
    kotor_module_world_label,
    nearest_environment_kit_snap,
    nearest_environment_kit_wall_snap,
    rebase_environment_kit_room_to_walkmesh,
    scan_vanilla_environment_kits,
    seal_environment_kit_exterior_bounds,
    trim_environment_kit_connection_overlap,
    trim_environment_kit_room_volume_overlap,
    vanilla_environment_kit_collections,
    write_vanilla_environment_kit_catalog,
)
from .map_studio_pascal_building import (
    add_pascal_building_room as add_pascal_building_room_to_project,
    available_pascal_building_styles,
    pascal_building_levels as pascal_building_levels_for_project,
    pascal_architecture_door_spec,
    pascal_architecture_profile_for_room,
    pascal_architecture_runtime_resources,
    preview_pascal_building_opening,
    scan_vanilla_pascal_building_styles,
    set_pascal_building_level as set_pascal_building_level_in_project,
    set_pascal_building_opening as set_pascal_building_opening_in_project,
    vanilla_pascal_building_styles,
    write_vanilla_pascal_style_catalog,
)
from .map_studio_spatial_design import (
    SpatialDesignPlan,
    audit_spatial_design,
    combine_marker_geometry,
    read_authored_spatial_design,
    spatial_design_marker_geometry,
    spatial_design_placement_ledger,
    write_authored_spatial_design,
)
from .map_studio_texture_paint import suggest_kotor_texture_resref, validate_kotor_texture_resref
from .map_studio_tool_belt_preferences import (
    MAP_STUDIO_TOOL_BELT_SECTION,
    normalise_map_studio_tool_belt_preferences,
)
from .map_studio_tool_contract_audit import audit_map_studio_tool_belt_contract
from .map_studio_command_history import MapStudioCommandHistory, MapStudioCommandRestoreResult
from .map_studio_curve_guides import add_authored_curve_guide as add_curve_guide_to_project
from .map_studio_curve_guides import authored_curve_guides
from .map_studio_universal_transform_overlay import build_map_studio_universal_transform_overlay
from .authored_gameplay_marker_geometry import (
    AuthoredGameplayMarkerGeometry,
    authored_gameplay_marker_geometry,
    authored_gameplay_marker_geometry_for_project,
)
from .authored_gameplay_preview import (
    authored_gameplay_preview_markers,
    authored_module_entry_point_preview_marker,
    authored_module_entry_point_preview_row,
)
from .authored_module_lighting import (
    add_authored_room_light as add_authored_room_light_to_project,
    authored_room_light_rows,
    duplicate_authored_room_light,
    parse_authored_room_light_id,
    remove_authored_room_light,
    rename_authored_room_light,
    update_authored_room_light_properties,
    update_authored_room_light_transform,
)
from .authored_module_world_lighting import (
    apply_authored_environment_style_defaults,
    authored_world_lighting_settings as read_authored_world_lighting_settings,
    default_authored_world_lighting_settings,
    update_authored_world_lighting_settings,
)
from .authored_skybox import (
    FiveFaceSkyboxSpec,
    FiveFaceSkyboxTextures,
    build_five_face_skybox_room,
)
from .authored_sky_traffic import (
    build_sky_traffic_preview,
    create_authored_sky_traffic as create_sky_traffic_actor,
    read_authored_project_sky_traffic,
    sample_sky_traffic,
    validate_authored_sky_traffic_collection,
    write_authored_project_sky_traffic,
)
from .authored_module_scripts import (
    authored_script_hook_field_choices,
    authored_script_hooks,
    remove_authored_script_hook,
    set_authored_script_hook,
)
from .authored_module_placements import (
    SUPPORTED_AUTHORED_GAMEPLAY_PLACEMENTS,
    add_authored_gameplay_placement,
    authored_gameplay_placement_rows,
    duplicate_authored_gameplay_placement,
    parse_authored_gameplay_placement_id,
    remove_authored_module_entry_point,
    remove_authored_gameplay_placement,
    rename_authored_gameplay_placement,
    snap_authored_gameplay_placement_to_walkmesh,
    update_authored_module_entry_point,
    update_authored_creature_behavior,
    update_authored_gameplay_camera_properties,
    update_authored_gameplay_placement_transform,
    update_authored_gameplay_transition,
)
from .authored_room_operations import (
    add_authored_floor_plan_opening_transition_marker,
    add_authored_room_composition_primitive,
    apply_authored_floor_plan_axis_split,
    apply_authored_floor_plan_bevel,
    apply_authored_floor_plan_boolean_difference,
    apply_authored_floor_plan_edge_extrude,
    apply_authored_floor_plan_inset,
    apply_authored_floor_plan_rectangular_cut,
    apply_authored_terrain_operation,
    apply_authored_floor_plan_rectangular_union,
    apply_authored_floor_plan_operation,
    available_authored_composition_primitive_kinds,
    authored_floor_plan_room_choices,
    authored_floor_plan_vertex_snap_candidates,
    authored_room_composition_primitive_vertex_snap_candidates,
    authored_room_composition_primitive_universal_transform,
    authored_terrain_room_choices,
    authored_room_composition_primitives,
    bridge_authored_floor_plan_edges,
    center_authored_room_composition_primitive_pivot,
    claim_authored_room_composition_floor,
    cleanup_authored_floor_plan_normals,
    cleanup_authored_floor_plan_vertices,
    combine_authored_room_composition_meshes,
    delete_authored_room_composition_primitive_history,
    group_authored_room_composition_primitives,
    duplicate_authored_room_composition_primitive,
    fill_authored_floor_plan_face,
    freeze_authored_room_composition_primitive_transform,
    flatten_authored_floor_plan_vertices,
    grid_snap_authored_floor_plan_vertices,
    grid_snap_authored_room_composition_primitive,
    mirror_authored_room_composition_primitive_transform,
    mirror_authored_floor_plan_vertices,
    move_authored_floor_plan_point,
    move_authored_room_composition_primitive,
    transform_authored_room_composition_primitives,
    rename_authored_room_composition_primitive,
    remove_authored_room_composition_primitive,
    reset_authored_room_composition_primitive_transform,
    separate_authored_room_composition_primitive,
    separate_authored_room_combined_primitive_shells,
    set_authored_floor_plan_extrusion_settings,
    set_authored_floor_plan_wall_opening,
    set_authored_room_edge_normal_policy,
    set_authored_room_composition_primitive_dimensions,
    set_authored_room_composition_primitive_style,
    set_authored_room_composition_primitive_transform,
    shrink_wrap_authored_room_composition_primitive_to_terrain,
    split_authored_floor_plan_face,
    snap_authored_room_composition_primitive_pivot_to_vertex,
    snap_authored_floor_plan_vertex_to_vertex,
    transform_snap_authored_room_composition_primitive_level,
    transform_snap_authored_floor_plan_vertices,
    triangulate_authored_floor_plan_face,
    weld_authored_floor_plan_vertices,
    zero_authored_room_composition_primitive_pivot,
)
from .authored_room_outline_geometry import AuthoredRoomOutlineGeometry, authored_room_outline_geometry_for_project
from .authored_room_presets import available_authored_room_primitive_presets, create_authored_module_from_room_preset
from .authored_room_style import update_authored_room_style
from .authored_terrain_builder import (
    TerrainHeightfieldPrimitive,
    apply_terrain_shape_preset,
    available_terrain_shape_presets,
)
from .authored_terrain_walkability_overlay import (
    AuthoredTerrainWalkabilityOverlay,
    authored_terrain_walkability_overlay_for_project,
)
from .authored_module_walkmesh import combine_authored_module_walkmesh
from .authored_walkmesh_status import AuthoredWalkmeshStatus, authored_walkmesh_status_for_project
from .authored_walkmesh_status import authored_walkmesh_room_surface_choices
from .authored_walkmesh_surfaces import authored_walkmesh_surface_palette
from .authored_walkmesh_audit import audit_authored_wok
from .dev_module_smoke import DevModuleGameProofRequest, DevModuleInstallPrepRequest, DevModuleSmokeRequest, prepare_dev_test_module_install, record_dev_module_game_proof
from .module_layout_service import ModuleLayoutService
from .module_porter_service import ModulePorterService
from .module_walkmesh_service import ModuleWalkmeshService


def _read_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


_MAP_STUDIO_PROOF_CHECK_LABELS = {
    "module_loads_in_game": "`warp` loads the generated module in KOTOR",
    "module_identity_matches_authored_resref": "Loaded module identity matches the authored resref",
    "player_spawns_on_floor": "Player appears on the generated floor, not in void",
    "test_placeable_visible": "Authored/test placeable appears where expected",
    "player_can_walk_on_floor": "Player can walk across the generated floor",
    "transition_pathing_sanity_confirmed": "Transitions and pathing behave sanely in the loaded module",
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
}


def _map_studio_proof_required_check_labels(proof_manifest_path: str | Path) -> tuple[str, ...]:
    proof = _read_json_object(proof_manifest_path) if str(proof_manifest_path or "").strip() else {}
    checks = proof.get("acceptance_checks") if isinstance(proof, dict) else ()
    if not isinstance(checks, list):
        checks = ()
    labels = []
    for check in checks:
        key = str(check or "").strip()
        if not key:
            continue
        labels.append(_MAP_STUDIO_PROOF_CHECK_LABELS.get(key, key.replace("_", " ")))
    if labels:
        return tuple(labels)
    return (
        _MAP_STUDIO_PROOF_CHECK_LABELS["module_loads_in_game"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["module_identity_matches_authored_resref"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["player_spawns_on_floor"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["test_placeable_visible"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["player_can_walk_on_floor"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["transition_pathing_sanity_confirmed"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["no_inherited_base_game_geometry_or_scripted_movers"],
        _MAP_STUDIO_PROOF_CHECK_LABELS["screenshot_or_video_captured"],
    )


def _map_studio_package_resource_inventory(
    proof_manifest_path: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the staged package inventory from KMAP metadata or proof manifest."""

    for source in (
        dict(payload or {}),
        dict((payload or {}).get("modder_test_plan") or {}) if isinstance((payload or {}).get("modder_test_plan"), dict) else {},
    ):
        inventory = source.get("package_resource_inventory")
        if isinstance(inventory, dict):
            return dict(inventory)
    proof = _read_json_object(proof_manifest_path) if str(proof_manifest_path or "").strip() else {}
    for source in (
        proof,
        dict(proof.get("package") or {}) if isinstance(proof.get("package"), dict) else {},
        dict(proof.get("modder_test_plan") or {}) if isinstance(proof.get("modder_test_plan"), dict) else {},
    ):
        inventory = source.get("package_resource_inventory") or source.get("resource_inventory")
        if isinstance(inventory, dict):
            return dict(inventory)
    return {}


def _map_studio_package_resource_summary(inventory: dict[str, Any]) -> str:
    """Return a compact modder-facing summary of the staged package inventory."""

    if not inventory:
        return "No package resource inventory is available; stage or install the authored module again."
    groups = dict(inventory.get("resource_groups") or {})
    required = len(tuple(inventory.get("required_runtime_resources") or ()))
    missing = len(tuple(inventory.get("missing_required_runtime_resources") or ()))
    readback_ok = bool(inventory.get("readback_ok"))
    installed = bool(dict(inventory.get("install") or {}).get("installed"))
    dry_run = bool(dict(inventory.get("install") or {}).get("dry_run"))
    archive_count = int(groups.get("verified_archive_resource_count") or 0)
    loose_count = int(groups.get("loose_staged_resource_count") or 0)
    install_state = "installed" if installed else ("dry-run install" if dry_run else "staged")
    return (
        f"Package inventory: {required} required runtime resource(s), {missing} missing, "
        f"{archive_count} archive readback resource(s), {loose_count} loose staged file(s), "
        f"readback {'ok' if readback_ok else 'not verified'}, {install_state}."
    )


MAP_STUDIO_MODELING_STALE_OUTPUTS = ("MDL", "MDX", "WOK", "LYT", "VIS", "PTH", ".mod")
MAP_STUDIO_MODELING_READINESS_IMPACT = "Map Studio validation, export, install handoff, and game proof are stale."
MAP_STUDIO_WORLD_LIGHTING_STALE_OUTPUTS = ("ARE", ".mod")
MAP_STUDIO_WORLD_LIGHTING_READINESS_IMPACT = (
    "World lighting and fog changed; ARE export, install handoff, and game proof are stale."
)
MAP_STUDIO_ACTIVE_SELECTION_SECTION = "map_studio_active_selection"
MAP_STUDIO_SELECTION_READINESS_IMPACT = "Map Studio selection target changed; generated resources are unchanged."


def _authored_primitive_by_identity(
    authored,
    *,
    room_resref: str,
    primitive_name: str,
):
    """Find one retained recipe without evaluating or allocating its mesh."""

    wanted_room = normalise_resref(room_resref)
    wanted_name = str(primitive_name or "").strip()
    for room in tuple(getattr(authored, "rooms", ()) or ()):
        if room.normalised_resref() != wanted_room:
            continue
        composition = room.primitive
        if not isinstance(composition, AuthoredRoomComposition):
            break
        for primitive in (composition.floor,) + tuple(composition.primitives or ()):
            base = getattr(primitive, "primitive", primitive)
            candidate_name = str(
                getattr(primitive, "name", "")
                or getattr(base, "name", "")
                or ""
            ).strip()
            if candidate_name == wanted_name:
                return primitive
        break
    raise ValueError(
        f"Room {room_resref} has no authored primitive named '{primitive_name}'."
    )


@dataclass(frozen=True)
class MapStudioLaunchHandoffSummary:
    """Non-mutating summary of the current Map Studio in-game launch handoff."""

    ready: bool
    module_root: str
    game: str
    warp_command: str
    launcher_path: str = ""
    proof_manifest_path: str = ""
    proof_recording_script_path: str = ""
    launch_helper_command: str = ""
    elevated_launch_script_path: str = ""
    installed_module_path: str = ""
    checklist_path: str = ""
    resolved_modules_dir: str = ""
    resolved_game_root_dir: str = ""
    package_resource_inventory: dict[str, Any] | None = None
    package_resource_summary: str = ""
    warnings: tuple[str, ...] = ()
    blocking_messages: tuple[str, ...] = ()
    summary: str = ""
    next_action: str = ""
    capability_stage: str = "installed_for_game_test_handoff"


@dataclass(frozen=True)
class MapStudioGameProofRecordingSummary:
    """Non-mutating defaults for recording Map Studio in-game proof."""

    ready: bool
    module_root: str
    game: str
    warp_command: str
    proof_manifest_path: str = ""
    default_evidence_path: str = ""
    required_checks: tuple[str, ...] = ()
    package_resource_inventory: dict[str, Any] | None = None
    package_resource_summary: str = ""
    warnings: tuple[str, ...] = ()
    blocking_messages: tuple[str, ...] = ()
    summary: str = ""
    next_action: str = ""
    capability_stage: str = "installed_for_game_test_recording_handoff"


class DeferredAuthoredModuleReadiness:
    """Compute the authored-module readiness projection on first access.

    Pointer-release placement/light commits ignore the readiness return
    entirely, and the Map Studio window refreshes export gates through its
    deferred background validation worker.  Computing the ~50 ms readiness
    projection synchronously inside every commit made dragging hitch, so
    the hot-path setters hand back this lazy view instead.  Attribute
    access resolves against the authored state at access time; the
    controller already caches the projection per authored revision.
    """

    __slots__ = ("_controller", "_result")

    def __init__(self, controller) -> None:
        self._controller = controller
        self._result = None

    def _resolve(self):
        if self._result is None:
            self._result = self._controller.authored_module_readiness()
        return self._result

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)


class MapStudioTextureCloneCancelled(RuntimeError):
    """Raised when a user cancels the atomic game-texture clone transaction."""


@dataclass(frozen=True)
class MapStudioObjRoomImportPreparation:
    """Immutable result of the background-safe OBJ room analysis pass.

    Parsing and topology analysis are intentionally separate from commit.  A
    Qt worker may build this value, but only the GUI thread may pass it to the
    commit method that mutates KMAP state and project texture sidecars.
    """

    source_path: str
    project_path: str
    project_texture_resrefs: tuple[str, ...]
    module_root: str
    options: ObjRoomAuthoringOptions
    document: Any
    primitive: ImportedMeshRoomPrimitive
    report: Any
    components: tuple[Any, ...]
    material_texture_resrefs: tuple[tuple[str, str], ...]
    texture_imports: tuple[tuple[str, str, str], ...]
    neutral_texture_resref: str = ""


@dataclass(frozen=True)
class MapStudioObjRoomImportResult:
    """Committed OBJ room, texture, and walkmesh summary for Map Studio UI."""

    module_root: str
    room_resref: str
    game: str
    imported_texture_resrefs: tuple[str, ...]
    selected_component_ids: tuple[str, ...]
    walkmesh_report: dict[str, Any]


@dataclass(frozen=True)
class MapStudioPanoramaSkyboxPreparation:
    """Immutable background conversion plus the KMAP state it was based on."""

    source_path: str
    source_sha256: str
    project_path: str
    project_texture_resrefs: tuple[str, ...]
    authored_state_token: tuple[Any, ...]
    room_resref: str
    face_texture_resrefs: tuple[tuple[str, str], ...]
    conversion: Any


@dataclass(frozen=True)
class MapStudioPanoramaSkyboxResult:
    """Atomic panorama-texture and authored-room commit summary."""

    room_resref: str
    texture_resrefs: tuple[str, ...]
    source_is_hdr: bool
    face_size: int
    message: str


class ModuleEditorController:
    def __init__(self, model: ModuleEditorModel | None = None) -> None:
        self.model = model or ModuleEditorModel()
        self.layout_service = ModuleLayoutService()
        self.walkmesh_service = ModuleWalkmeshService()
        self.blueprint_service = ModuleBlueprintService()
        self.builder_service = ModuleBuilderService()
        self.porter_service = ModulePorterService()
        self.validator = KMapValidator()
        self.export_bridge = LevelExportBridge(self.validator)
        self.command_history = MapStudioCommandHistory()
        self.texture_sidecar_journal = MapStudioTextureSidecarJournal()
        self._authored_placeable_resources: tuple[tuple[str, str, bytes], ...] = ()
        self._authored_placeable_resource_issues: tuple[Any, ...] = ()
        self._authored_creature_resources: tuple[tuple[str, str, bytes], ...] = ()
        self._authored_scripting_resources: tuple[tuple[str, str, bytes], ...] = ()
        self._authored_placeable_preview_rows: tuple[dict[str, Any], ...] = ()
        self._authored_placeable_preview_revision = 0
        self._map_studio_pie_context_resource_revision = 0
        self.last_map_studio_resolved_placement_ids: tuple[str, ...] = ()
        self.last_map_studio_unresolved_placement_ids: tuple[str, ...] = ()
        self.last_map_studio_preview_cache_hit = False
        self.last_map_studio_preview_elapsed_ms = 0.0
        self.last_map_studio_primitive_preview_elapsed_ms = 0.0
        self.last_map_studio_primitive_preview_overlay = None
        self.last_map_studio_primitive_preview_deferred = False
        self.last_map_studio_primitive_preview_identity = None
        self.last_map_studio_primitive_preview_dimensions = None
        self.last_map_studio_primitive_commit_matches_preview = False
        self._map_studio_combined_preview_cache: tuple[Any, Any, Any] | None = None
        self._map_studio_authored_state_revision = 0
        self._map_studio_cached_authored_project: tuple[tuple[Any, ...], Any] | None = None
        self._map_studio_cached_query_token: tuple[Any, ...] | None = None
        self._map_studio_cached_queries: dict[tuple[Any, ...], Any] = {}
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        self._last_committed_imported_mesh_room = None

    @property
    def project(self) -> KMapProject:
        return self.model.project

    def _map_studio_authored_state_token(self) -> tuple[Any, ...]:
        """Return an O(1) identity/revision token for authored KMAP state."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        return (
            id(payload),
            int(self._map_studio_authored_state_revision),
            id(self.project),
            str(getattr(self.project, "name", "") or ""),
            str(getattr(self.project, "game", "") or ""),
        )

    def _invalidate_map_studio_authored_state(self, _reason: str = "") -> None:
        """Invalidate parsed/derived preview state after an in-place mutation.

        Most authored edits replace the payload dictionary and therefore miss
        by identity automatically.  Command recording calls this method as the
        explicit guard for nested proof/PTH/texture metadata edits that retain
        the same dictionary object.  No KMAP serialization or deep traversal
        occurs on the interaction path.
        """

        self._map_studio_authored_state_revision += 1
        self._map_studio_cached_authored_project = None
        self._map_studio_cached_query_token = None
        self._map_studio_cached_queries.clear()
        self._map_studio_combined_preview_cache = None

    def _map_studio_authored_project_snapshot(self):
        """Decode the current authored payload once per identity/revision."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if payload is None:
            return None
        token = self._map_studio_authored_state_token()
        cached = self._map_studio_cached_authored_project
        if cached is not None and cached[0] == token:
            return cached[1]
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        self._map_studio_cached_authored_project = (token, authored)
        if self._map_studio_cached_query_token != token:
            self._map_studio_cached_query_token = token
            self._map_studio_cached_queries.clear()
        return authored

    def map_studio_authored_placements_snapshot(self):
        """Return an isolated copy of the current authored placements.

        PIE needs only placement rows, not the full mutable authored project.
        Copying this focused surface prevents a background preview workflow
        from mutating Scene's decoded-project cache while avoiding an expensive
        copy of room geometry and texture-paint payloads.
        """

        authored = self._map_studio_authored_project_snapshot()
        placements = getattr(authored, "placements", None) if authored is not None else None
        return deepcopy(placements) if placements is not None else None

    def map_studio_scene_animation_map(self, resource_manager: Any):
        """Tag-occurrence animation clips from the module's own OnEnter NCS.

        207TEL and similar ambient scenes assign creature animations by tag in
        their OnEnter script; this reads that intent so PIE can pose each
        creature. The imported MOD/RIM is authoritative and is checked before
        Override/game-library resources. The returned mapping reports its
        script resref, source, SHA-256, and raw constants for diagnostics.
        """

        authored = self._map_studio_authored_project_snapshot()
        from .map_studio_scene_animations import (
            MapStudioSceneAnimationMap,
            build_module_scene_animations,
        )

        self.last_map_studio_scene_animation_source = ""
        self.last_map_studio_scene_animation_sha256 = ""
        self.last_map_studio_scene_animation_script = ""
        if authored is None:
            return MapStudioSceneAnimationMap()

        script_resref, ncs_bytes, ncs_source = self._map_studio_onenter_ncs(authored, resource_manager)
        self.last_map_studio_scene_animation_script = script_resref
        if not script_resref:
            return MapStudioSceneAnimationMap()
        if not ncs_bytes:
            return MapStudioSceneAnimationMap(script_resref=script_resref)

        source_sha256 = hashlib.sha256(bytes(ncs_bytes)).hexdigest()
        try:
            result = build_module_scene_animations(
                onenter_ncs_bytes=bytes(ncs_bytes),
                script_resref=script_resref,
                source=ncs_source,
                source_sha256=source_sha256,
            )
        except Exception:
            result = MapStudioSceneAnimationMap(
                script_resref=script_resref,
                source=ncs_source,
                source_sha256=source_sha256,
            )
        self.last_map_studio_scene_animation_source = ncs_source
        self.last_map_studio_scene_animation_sha256 = source_sha256
        return result

    def _map_studio_onenter_ncs(self, authored: Any, resource_manager: Any):
        """Resolve a module's OnEnter script resref and compiled NCS bytes.

        Shared by scene-animation reading and PIE scripting-state execution. The
        imported MOD/RIM capsule is authoritative (checked first), then the
        active resource manager. Returns ``(script_resref, ncs_bytes|None,
        source)``; never raises.
        """

        from .map_studio_scene_animations import module_onenter_script_resref

        module_root = normalise_resref(getattr(authored, "module_root", "") or getattr(self.project, "name", ""))
        game = str(getattr(authored, "game", "") or self.project.game or "K1").upper()
        script_resref = ""
        # Prefer the preserved stock IFO's Mod_OnClientEntr, else the k_<root>_enter convention.
        extra = dict(getattr(authored, "extra", {}) or {})
        ifo_record = dict((extra.get("stock_resources") or {}).get("ifo") or {})
        if ifo_record.get("data"):
            try:
                import base64

                from pykotor.resource.formats.gff import read_gff

                ifo_bytes = base64.b64decode(str(ifo_record.get("data") or ""), validate=True)
                script_resref = module_onenter_script_resref(read_gff(bytes(ifo_bytes)).root)
            except Exception:
                script_resref = ""
        if not script_resref and module_root:
            script_resref = f"k_{module_root}_enter"
        if not script_resref:
            return "", None, ""
        ncs_bytes, ncs_source = self._load_module_ncs_resource(script_resref, authored, resource_manager)
        return script_resref, ncs_bytes, ncs_source

    def _load_module_ncs_resource(self, resref: str, authored: Any, resource_manager: Any):
        """Load one compiled NCS script by resref for the loaded module.

        The imported MOD/RIM capsule is authoritative (checked first), then the
        active resource manager. Shared by OnEnter reading and dialogue node
        action-script execution. Returns ``(ncs_bytes|None, source)``; never
        raises.
        """

        clean = str(resref or "").strip()
        if not clean:
            return None, ""
        game = str(getattr(authored, "game", "") or self.project.game or "K1").upper()
        extra = dict(getattr(authored, "extra", {}) or {})
        ncs_bytes = None
        ncs_source = ""
        try:
            from pykotor.extract.capsule import LazyCapsule
            from pykotor.resource.type import ResourceType as RT

            import_source = str(extra.get("import_source") or "").strip()
            source_paths: list[Path] = []
            if import_source:
                primary = Path(import_source)
                source_paths.append(primary)
                if primary.suffix.lower() == ".rim":
                    if primary.stem.lower().endswith("_s"):
                        source_paths.append(primary.with_name(f"{primary.stem[:-2]}.rim"))
                    else:
                        source_paths.append(primary.with_name(f"{primary.stem}_s.rim"))
            for source_path in tuple(dict.fromkeys(source_paths)):
                if not source_path.is_file():
                    continue
                try:
                    data = LazyCapsule(str(source_path)).resource(clean, RT.NCS)
                except Exception:
                    data = None
                if data:
                    ncs_bytes = bytes(data)
                    ncs_source = f"capsule:{source_path.resolve()}"
                    break
        except Exception:
            ncs_bytes = None

        if not ncs_bytes and resource_manager is not None:
            try:
                from pykotor.resource.type import ResourceType as RT

                getter = getattr(resource_manager, "get_strict", None) or getattr(resource_manager, "get", None)
                data = getter(clean, int(RT.NCS), game) if callable(getter) else None
                if data:
                    ncs_bytes = bytes(data)
                    ncs_source = f"resource_manager:{game}"
            except Exception:
                ncs_bytes = None
        return ncs_bytes, ncs_source

    def map_studio_pie_scripted_globals(
        self,
        resource_manager: Any,
        *,
        sandbox_numbers: dict[str, int] | None = None,
        sandbox_booleans: dict[str, bool] | None = None,
        sandbox_strings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute the module OnEnter script for PIE scripting state.

        Primary engine: the clean-room NCS virtual machine
        (``map_studio_pie_nwscript_vm``) runs the compiled script with real
        control flow, seeded with the creator sandbox so campaign-gated writes
        behave like retail (e.g. 207TEL only advances ``207TEL_Benok`` 1→2 when
        the earlier-module state is present); it also returns the scripted-event
        timeline (AssignCommand/DelayCommand closures, music, fades) for PIE.
        Fallback: the bounded literal-write reader (``map_studio_pie_scripting``)
        when the VM cannot complete — over-eager on campaign-conditional writes,
        tracked in the parity ledger. Journal writes are reported, never applied
        to campaign quest state. Never raises.
        """

        empty: dict[str, Any] = {
            "global_numbers": {},
            "global_booleans": {},
            "global_strings": {},
            "journal": (),
            "script_resref": "",
            "source": "",
            "effect_count": 0,
            "engine": "none",
            "commands": (),
            "unknown_routines": {},
        }
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return empty
        try:
            script_resref, ncs_bytes, ncs_source = self._map_studio_onenter_ncs(authored, resource_manager)
        except Exception:
            return empty
        if not ncs_bytes:
            return {**empty, "script_resref": script_resref}

        game = str(getattr(authored, "game", "") or self.project.game or "K1").upper()
        try:
            from .map_studio_pie_nwscript_vm import MapStudioPIEScriptContext, execute_ncs_script

            vm_context = MapStudioPIEScriptContext(
                game=game,
                global_numbers=dict(sandbox_numbers or {}),
                global_booleans=dict(sandbox_booleans or {}),
                global_strings=dict(sandbox_strings or {}),
            )
            vm_result = execute_ncs_script(bytes(ncs_bytes), game=game, context=vm_context)
        except Exception:
            vm_result = None
        if vm_result is not None and vm_result.completed:
            commands = tuple(
                {
                    "kind": str(command.kind),
                    "object_tag": str(command.object_tag),
                    "args": tuple(a for a in command.args if isinstance(a, (int, float, str))),
                    "delay_seconds": float(command.delay_seconds),
                    "deferred": command.saved_state is not None,
                }
                for command in vm_result.commands
            )
            return {
                "global_numbers": dict(vm_result.global_numbers),
                "global_booleans": dict(vm_result.global_booleans),
                "global_strings": dict(vm_result.global_strings),
                "journal": tuple(vm_result.journal_entries),
                "script_resref": script_resref,
                "source": ncs_source,
                "effect_count": len(vm_result.global_numbers)
                + len(vm_result.global_booleans)
                + len(vm_result.global_strings)
                + len(vm_result.journal_entries),
                "engine": "vm",
                "instructions_executed": int(vm_result.instructions_executed),
                "commands": commands,
                "local_booleans": {
                    f"{tag}:{index}": bool(value)
                    for (tag, index), value in vm_result.local_booleans.items()
                },
                "unknown_routines": dict(vm_result.unknown_routines),
            }

        from .map_studio_pie_scripting import apply_ncs_global_effects, extract_ncs_global_effects

        try:
            effects = extract_ncs_global_effects(bytes(ncs_bytes))
            numbers, booleans, strings, journal = apply_ncs_global_effects(effects)
        except Exception:
            return {**empty, "script_resref": script_resref, "source": ncs_source}
        return {
            **empty,
            "global_numbers": numbers,
            "global_booleans": booleans,
            "global_strings": strings,
            "journal": tuple((effect.name, int(effect.value)) for effect in journal),
            "script_resref": script_resref,
            "source": ncs_source,
            "effect_count": len(effects),
            "engine": "literal-reader",
        }

    def map_studio_pie_area_music(self, resource_manager: Any) -> dict[str, Any]:
        """Resolve the area's ambient music from its OnEnter script.

        KOTOR area music is not a static ARE field — the area OnEnter calls
        ``MusicBackgroundChangeDay``/``…Night`` with a literal ``ambientmusic.2da``
        row. This bounded reader recovers those tracks and resolves each to its
        streaming music resref, so PIE can report the area's ambient music
        instead of treating it as absent. Reported (not yet played); never
        raises.
        """

        from .map_studio_pie_scripting import extract_ncs_area_music

        empty: dict[str, Any] = {
            "day_track": None,
            "night_track": None,
            "battle_track": None,
            "day_resref": "",
            "night_resref": "",
            "battle_resref": "",
            "script_resref": "",
            "source": "",
        }
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return empty
        try:
            script_resref, ncs_bytes, ncs_source = self._map_studio_onenter_ncs(authored, resource_manager)
        except Exception:
            return empty
        if not ncs_bytes:
            return {**empty, "script_resref": script_resref}
        try:
            music = extract_ncs_area_music(bytes(ncs_bytes))
        except Exception:
            return {**empty, "script_resref": script_resref, "source": ncs_source}
        resolver = getattr(self, "_map_studio_stock_template_resolver", None)
        resolve = getattr(resolver, "ambient_music_resref", None)

        def _resref(track: Any) -> str:
            if track is None or not callable(resolve):
                return ""
            try:
                return str(resolve(int(track)) or "")
            except Exception:
                return ""

        return {
            "day_track": music.day_track,
            "night_track": music.night_track,
            "battle_track": music.battle_track,
            "day_resref": _resref(music.day_track),
            "night_resref": _resref(music.night_track),
            "battle_resref": _resref(music.battle_track),
            "script_resref": script_resref,
            "source": ncs_source,
        }

    def map_studio_pie_transition_validation(self, resource_manager: Any) -> dict[str, Any]:
        """Validate the module's inter-module transitions against installed modules.

        Doors/triggers carry ``LinkedToModule``/``LinkedTo``; a link to a module
        that is not installed would black-screen or crash the retail game. This
        resolves each destination against the installation's module list so a
        creator can catch broken transitions before launching. Reported only
        (PIE does not warp); never raises.
        """

        from .map_studio_pie_triggers import validate_module_transitions

        empty: dict[str, Any] = {"transitions": (), "checked": 0, "missing": 0, "available_module_count": 0}
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return empty
        try:
            context = self._map_studio_pie_resource_context()
            registry = build_pie_entity_registry(authored, template_inspector=context.template_inspector)
        except Exception:
            return empty

        transitions: list[tuple[str, str, str, str]] = []
        for entity in tuple(getattr(registry, "entities", ()) or ()):
            module = str(getattr(entity, "transition_module", "") or "").strip()
            if not module:
                continue
            kind = str(getattr(entity, "kind", "") or "").strip().lower()
            transitions.append(
                (
                    kind,
                    str(getattr(entity, "tag", "") or ""),
                    module,
                    str(getattr(entity, "transition_target", "") or ""),
                )
            )
        if not transitions:
            return empty

        available: list[str] = []
        game = str(getattr(authored, "game", "") or self.project.game or "K1").upper()
        manager = getattr(context.resolver, "_manager", None) if context.resolver is not None else None
        installation_getter = getattr(manager, "get_k2" if game == "K2" else "get_k1", None) if manager is not None else None
        try:
            installation = installation_getter() if callable(installation_getter) else None
        except Exception:
            installation = None
        if installation is not None:
            # Prefer the installation's own module enumeration; fall back to
            # scanning the Modules directory so a lazily-loaded install still
            # resolves rather than falsely reporting every link as missing.
            for attr in ("modules_list", "module_names"):
                try:
                    lister = getattr(installation, attr, None)
                    listed = lister() if callable(lister) else None
                    if listed:
                        available = [str(name) for name in (listed.keys() if isinstance(listed, dict) else listed)]
                        break
                except Exception:
                    continue
            if not available:
                try:
                    path_getter = getattr(installation, "module_path", None)
                    modules_dir = path_getter() if callable(path_getter) else None
                    if modules_dir is not None:
                        for child in Path(str(modules_dir)).glob("*.*"):
                            if child.suffix.lower() in {".mod", ".rim", ".erf"}:
                                available.append(child.name)
                except Exception:
                    pass

        checks = validate_module_transitions(transitions, available)
        return {
            "transitions": tuple(
                {
                    "source_kind": c.source_kind,
                    "tag": c.tag,
                    "module": c.module,
                    "target": c.target,
                    "exists": c.exists,  # True | False | None (unverifiable)
                }
                for c in checks
            ),
            "checked": len(checks),
            "missing": sum(1 for c in checks if c.exists is False),
            "unverified": sum(1 for c in checks if c.exists is None),
            "available_module_count": len({name for name in available if name}),
        }

    def _map_studio_cached_authored_query(self, key: tuple[Any, ...], builder):
        """Reuse an immutable/read-only projection for the current revision."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP.")
        token = self._map_studio_authored_state_token()
        if self._map_studio_cached_query_token != token:
            self._map_studio_cached_query_token = token
            self._map_studio_cached_queries.clear()
        cache_key = tuple(key)
        if cache_key not in self._map_studio_cached_queries:
            self._map_studio_cached_queries[cache_key] = builder(authored)
        return self._map_studio_cached_queries[cache_key]

    def _map_studio_cached_project_query(self, key: tuple[Any, ...], builder):
        """Cache a read-only whole-KMAP projection for the authored revision."""

        token = self._map_studio_authored_state_token()
        if self._map_studio_cached_query_token != token:
            self._map_studio_cached_query_token = token
            self._map_studio_cached_queries.clear()
        cache_key = tuple(key)
        if cache_key not in self._map_studio_cached_queries:
            self._map_studio_cached_queries[cache_key] = builder(self.project)
        return self._map_studio_cached_queries[cache_key]

    def _capture_map_studio_command_state(self):
        return self.command_history.capture(
            self.project,
            selected_ids=tuple(self.model.selected_ids),
            active_module_id=self.model.active_module_id,
            active_room_id=self.model.active_room_id,
        )

    def _restore_map_studio_command_state_without_history(self, snapshot) -> None:
        """Restore a failed file-backed transaction without moving Undo/Redo."""

        restored = KMapSerializer.from_dict(dict(getattr(snapshot, "data", {}) or {}))
        restored.path = str(getattr(snapshot, "path", "") or "")
        restored.dirty = bool(getattr(snapshot, "dirty", False))
        self.model.set_project(restored)
        self.model.selected_ids = list(getattr(snapshot, "selected_ids", ()) or ())
        self.model.active_module_id = str(getattr(snapshot, "active_module_id", "") or "")
        self.model.active_room_id = str(getattr(snapshot, "active_room_id", "") or "")
        self._invalidate_map_studio_authored_state("transaction restore")

    def _record_map_studio_command(
        self,
        *,
        action_key: str,
        label: str,
        before,
        stale_outputs: tuple[str, ...] = MAP_STUDIO_MODELING_STALE_OUTPUTS,
        readiness_impact: str = MAP_STUDIO_MODELING_READINESS_IMPACT,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        sidecar_patches: tuple[MapStudioTextureSidecarPatch, ...] = (),
    ):
        if {str(value).upper() for value in tuple(stale_outputs or ())} & {"MDL", "MDX", "WOK", "LYT", "VIS", "PTH"}:
            self._mark_stock_pth_dirty(label)
        record = self.command_history.record(
            action_key=action_key,
            label=label,
            before=before,
            after=self._capture_map_studio_command_state(),
            stale_outputs=stale_outputs,
            readiness_impact=readiness_impact,
            summary=summary,
            metadata=metadata,
            sidecar_patches=tuple(sidecar_patches or ()),
        )
        if record is not None:
            self._mark_map_studio_export_proof_stale(
                record.label,
                before=before,
                stale_outputs=tuple(stale_outputs or ()),
                readiness_impact=readiness_impact,
            )
            self.model.log(f"Undo checkpoint recorded: {record.label}.")
        # Selection context lives outside the authored module and does not
        # change geometry/readiness. Rebuilding a 352-node stock preview on
        # every click would defeat the cache and recreate the renderer-reset
        # behavior this revision model is designed to remove.
        if str(action_key or "") != "map_studio.selection.select":
            self._invalidate_map_studio_authored_state(label)
        return record

    def _mark_stock_pth_dirty(self, reason: str) -> None:
        """Invalidate an imported byte-exact PTH after an authored edit."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if not isinstance(payload, dict):
            return
        authored_extra = payload.get("extra")
        if not isinstance(authored_extra, dict):
            return
        stock_resources = authored_extra.get("stock_resources")
        if not isinstance(stock_resources, dict) or not stock_resources.get("pth"):
            return
        if authored_extra.get("stock_pth_dirty"):
            return
        authored_extra["stock_pth_dirty"] = True
        authored_extra["stock_pth_preserved"] = False
        authored_extra["stock_pth_dirty_reason"] = str(reason or "Map Studio authored state changed.")
        self.project.dirty = True

    def _mark_map_studio_export_proof_stale(
        self,
        latest_summary: str,
        *,
        before,
        stale_outputs: tuple[str, ...],
        readiness_impact: str,
    ) -> None:
        if not stale_outputs:
            return
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if not isinstance(payload, dict):
            return
        previous_payload = self._map_studio_authored_payload_from_snapshot(before)
        has_export_or_proof = any(
            str(payload.get(key) or previous_payload.get(key) or "").strip()
            for key in (
                "pack_manifest_path",
                "proof_manifest_path",
                "installed_module_path",
                "in_game_proof_evidence_path",
            )
        ) or bool(payload.get("package_resource_inventory") or previous_payload.get("package_resource_inventory"))
        if not has_export_or_proof:
            return
        # Stale stage/proof artifacts are removed, not carried forward: an
        # edit after staging means the packaged module no longer matches the
        # KMAP, so pointing at it would be dishonest readiness reporting.
        for key in (
            "pack_manifest_path",
            "proof_manifest_path",
            "checklist_path",
            "installed_module_path",
            "backup_module_path",
            "resolved_modules_dir",
            "resolved_game_root_dir",
            "launch_helper_command",
            "elevated_launch_script_path",
            "proof_recording_script_path",
            "package_resource_inventory",
            "export_job",
            "in_game_proof",
            "in_game_proof_evidence_path",
            "evidence_path",
            "game_test",
        ):
            payload.pop(key, None)
        payload["runtime_resources"] = []
        payload["game_tested"] = False
        payload["manual_proof_required"] = True
        # The bridge's payload invalidation (edited_rooms/latest_operation,
        # per-edit stale outputs) is authoritative when present; only fill a
        # generic record when the bridge produced none.
        if not isinstance(payload.get("export_proof_invalidation"), dict):
            payload["export_proof_invalidation"] = {
                "invalidates_previous_export": True,
                "invalidates_game_proof": True,
                "latest_summary": str(latest_summary or "").strip(),
                "stale_outputs": [str(output) for output in tuple(stale_outputs or ()) if str(output).strip()],
                "readiness_impact": str(readiness_impact or "").strip(),
                "next_action": "Regenerate the authored module package, reinstall it if needed, and record fresh in-game proof.",
            }
        self.project.dirty = True

    @staticmethod
    def _clear_authored_export_proof_invalidation(
        payload: dict[str, Any],
        *,
        clear_stage_metadata: bool,
    ) -> None:
        """Mark generated resources current while keeping live game proof honest."""

        for key in (
            "export_proof_invalidation",
            "in_game_proof",
            "in_game_proof_evidence_path",
            "evidence_path",
            "game_test",
        ):
            payload.pop(key, None)
        if clear_stage_metadata:
            for key in (
                "proof_manifest_path",
                "checklist_path",
                "installed_module_path",
                "backup_module_path",
                "resolved_modules_dir",
                "resolved_game_root_dir",
                "launch_helper_command",
                "elevated_launch_script_path",
                "proof_recording_script_path",
                "package_resource_inventory",
                "modder_test_plan",
                "export_job",
            ):
                payload.pop(key, None)
        payload["manual_proof_required"] = True
        payload["game_tested"] = False

    @staticmethod
    def _map_studio_authored_payload_from_snapshot(snapshot) -> dict[str, Any]:
        data = dict(getattr(snapshot, "data", {}) or {})
        payload = data.get("authored_module")
        if isinstance(payload, dict):
            return dict(payload)
        extra = data.get("extra_sections")
        if isinstance(extra, dict) and isinstance(extra.get("authored_module"), dict):
            return dict(extra["authored_module"])
        return {}

    def can_undo_map_studio_command(self) -> bool:
        return self.command_history.can_undo

    def can_redo_map_studio_command(self) -> bool:
        return self.command_history.can_redo

    def undo_map_studio_command(self) -> MapStudioCommandRestoreResult | None:
        result = self.command_history.undo()
        if result is None:
            return None
        sidecar_patches = tuple(getattr(result.record, "sidecar_patches", ()) or ())
        sidecars_applied = False
        try:
            if sidecar_patches:
                self.texture_sidecar_journal.capture(
                    self.project,
                    paths=tuple(patch.path for patch in sidecar_patches),
                )
                self.texture_sidecar_journal.apply(self.project, sidecar_patches, use_after=False)
                sidecars_applied = True
            self.model.set_project(result.project)
        except Exception:
            if sidecars_applied:
                self.texture_sidecar_journal.apply(self.project, sidecar_patches, use_after=True)
            self.command_history.redo()
            raise
        self.model.selected_ids = list(result.selected_ids)
        self.model.active_module_id = result.active_module_id
        self.model.active_room_id = result.active_room_id
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        self._authored_creature_resources = ()
        self._invalidate_map_studio_authored_state("undo")
        self.model.log(result.message)
        return result

    def redo_map_studio_command(self) -> MapStudioCommandRestoreResult | None:
        result = self.command_history.redo()
        if result is None:
            return None
        sidecar_patches = tuple(getattr(result.record, "sidecar_patches", ()) or ())
        sidecars_applied = False
        try:
            if sidecar_patches:
                self.texture_sidecar_journal.capture(
                    self.project,
                    paths=tuple(patch.path for patch in sidecar_patches),
                )
                self.texture_sidecar_journal.apply(self.project, sidecar_patches, use_after=True)
                sidecars_applied = True
            self.model.set_project(result.project)
        except Exception:
            if sidecars_applied:
                self.texture_sidecar_journal.apply(self.project, sidecar_patches, use_after=False)
            self.command_history.undo()
            raise
        self.model.selected_ids = list(result.selected_ids)
        self.model.active_module_id = result.active_module_id
        self.model.active_room_id = result.active_room_id
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        self._authored_creature_resources = ()
        self._invalidate_map_studio_authored_state("redo")
        self.model.log(result.message)
        return result

    def new_project(self, name: str = "new_level", game: str = "K1", author: str = "") -> KMapProject:
        game_key = str(game or "K1").strip().upper()
        if game_key not in {"K1", "K2"}:
            raise ValueError("Map Studio projects must target K1 or K2.")
        issue = authored_resref_blocking_issue("Map Studio module root", name)
        if issue:
            raise ValueError(issue)
        project_name = normalise_resref(name) or "new_level"
        self.discard_project_texture_sidecar_changes()
        self.model.set_project(new_kmap_project(name=project_name, game=game_key, author=str(author or "").strip()))
        self._invalidate_map_studio_authored_state("new project")
        self._invalidate_map_studio_stock_preview_resources()
        self.texture_sidecar_journal.clear()
        self.command_history.clear()
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        self._authored_creature_resources = ()
        self.model.project.dirty = True
        self.model.log(f"Created new Map Studio KMAP project {project_name} for {game_key}.")
        return self.model.project

    def _index_import_source_capsules(
        self,
        resource_manager: Any,
        source_path: str | Path,
        *,
        clear_existing: bool = True,
    ) -> tuple[Path, ...]:
        """Publish the active KMAP's preserved module capsules to resources.

        ``import_source`` is project provenance, not authored content. Reopening
        a KMAP must therefore restore the ResourceManager overlay without
        rewriting the serialized authored-module payload or marking the project
        dirty. RIM modules may split templates into a sibling ``_s.rim``; use
        the same primary-then-companion order as the live import path.
        """

        if resource_manager is None:
            return ()
        clear_overlay = getattr(resource_manager, "clear_module_overlay", None)
        if clear_existing and callable(clear_overlay):
            clear_overlay()

        raw_source = str(source_path or "").strip()
        if not raw_source:
            return ()
        source = Path(raw_source).expanduser()
        if not source.is_absolute() and str(getattr(self.project, "path", "") or "").strip():
            source = Path(self.project.path).resolve().parent / source
        if not source.is_file():
            self.model.log(f"Could not restore bundled module resources; import source is missing: {source}")
            return ()

        add_overlay = getattr(resource_manager, "add_module_overlay", None)
        if not callable(add_overlay):
            self.model.log("Could not restore bundled module resources; the active resource manager has no capsule overlay API.")
            return ()

        overlay_paths = [source]
        if source.suffix.lower() == ".rim":
            if source.stem.lower().endswith("_s"):
                companion = source.with_name(f"{source.stem[:-2]}.rim")
            else:
                companion = source.with_name(f"{source.stem}_s.rim")
            if companion.is_file():
                overlay_paths.append(companion)

        indexed: list[Path] = []
        for overlay_path in overlay_paths:
            try:
                resource_count = int(add_overlay(str(overlay_path)) or 0)
            except Exception as exc:
                self.model.log(f"Could not index bundled module resources from {overlay_path.name}: {exc}")
                continue
            indexed.append(overlay_path)
            if resource_count:
                self.model.log(f"Indexed {resource_count} bundled resource(s) from {overlay_path.name} for editing.")
        if indexed:
            self._invalidate_map_studio_stock_preview_resources()
        return tuple(indexed)

    def restore_project_import_source_overlay(self, resource_manager: Any) -> tuple[Path, ...]:
        """Restore the saved authored module's source-capsule resource layer."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        authored_extra = payload.get("extra") if isinstance(payload, dict) else None
        source = str(authored_extra.get("import_source") or "") if isinstance(authored_extra, dict) else ""
        return self._index_import_source_capsules(resource_manager, source, clear_existing=True)

    def open_project(self, path: str | Path, *, resource_manager: Any = None) -> KMapProject:
        project = KMapSerializer.load(path)
        self.discard_project_texture_sidecar_changes()
        self.model.set_project(project)
        self._invalidate_map_studio_authored_state("open project")
        self._invalidate_map_studio_stock_preview_resources()
        self.texture_sidecar_journal.clear()
        self.texture_sidecar_journal.promote(project)
        self.command_history.clear()
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        self._authored_creature_resources = ()
        if resource_manager is not None:
            self.restore_project_import_source_overlay(resource_manager)
        self.model.log(f"Opened KMAP {Path(path).name}.")
        return project

    def save_project(self, path: str | Path | None = None) -> None:
        current_path = str(getattr(self.project, "path", "") or "").strip()
        if path is not None and current_path:
            current_parent = Path(current_path).resolve().parent
            target_parent = Path(path).resolve().parent
            if current_parent != target_parent and managed_project_texture_sidecars(self.project):
                raise ValueError(
                    "Save KMAP As cannot move a project with texture sidecars to another folder yet. "
                    "Save beside the current KMAP, or use a future Move Project command that copies and rebases assets transactionally."
                )
        previous_resolved = Path(current_path).resolve() if current_path else None
        KMapSerializer.save(self.project, path)
        path_changed = previous_resolved is not None and previous_resolved != Path(self.project.path).resolve()
        if path_changed:
            self.command_history.clear()
            self.model.log("Cleared Map Studio Undo/Redo because Save As established a new project-file epoch.")
        self.texture_sidecar_journal.clear()
        self.texture_sidecar_journal.promote(self.project)
        self.model.log(f"Saved KMAP {Path(self.project.path).name}.")

    def discard_project_texture_sidecar_changes(self) -> int:
        """Restore project texture bytes to the last successful Save/Open."""

        count = self.texture_sidecar_journal.restore_baseline(self.project)
        if count:
            self.model.log(f"Discarded unsaved changes to {count} project texture sidecar(s).")
        return int(count)

    def prepare_obj_room_import(
        self,
        path: str | Path,
        *,
        module_root: str = "",
        room_resref: str = "",
        game: str = "",
        source_units: str = "auto",
        source_up_axis: str = "y",
        scale_override: float | None = None,
        center_xy: bool = True,
        ground_to_zero: bool = False,
    ) -> MapStudioObjRoomImportPreparation:
        """Parse and analyze an OBJ room without mutating the active KMAP.

        This method is safe to run in Map Studio's import worker.  It captures
        the project path and texture-resref set so the GUI-thread commit can
        reject a stale analysis instead of overwriting intervening work.
        """

        source = Path(path).expanduser().resolve()
        document = load_obj_room_document(source)
        current_root = str(module_root or getattr(self.project, "name", "") or "").strip()
        if not current_root or current_root.lower() == "new_level":
            current_root = suggested_obj_texture_resref(source.stem, "", used=set())
        current_room = str(room_resref or "").strip()
        if not current_room:
            current_room = suggested_obj_texture_resref(source.stem, "", used=set())
        root_issue = authored_resref_blocking_issue("Authored module", current_root)
        if root_issue:
            raise ValueError(root_issue)

        target_game = str(game or getattr(self.project, "game", "K2") or "K2").strip().upper()
        existing_texture_resrefs = tuple(
            sorted(
                {
                    str(getattr(texture, "resref", "") or "").strip().lower()
                    for texture in tuple(getattr(self.project, "textures", ()) or ())
                    if str(getattr(texture, "resref", "") or "").strip()
                }
            )
        )
        reserved = set(existing_texture_resrefs)
        used_material_names = tuple(
            dict.fromkeys(str(surface.material_name or "") for surface in document.surfaces)
        )
        material_rows = {str(material.name or ""): material for material in document.materials}
        source_resrefs: dict[str, str] = {}
        material_texture_resrefs: dict[str, str] = {}
        texture_imports: list[tuple[str, str, str]] = []
        unresolved_material_names: list[str] = []
        for material_name in used_material_names:
            material = material_rows.get(material_name)
            texture_path = str(getattr(material, "diffuse_texture_path", "") or "")
            resolved = str(Path(texture_path).resolve()) if texture_path and Path(texture_path).is_file() else ""
            if not resolved:
                unresolved_material_names.append(material_name)
                continue
            source_key = str(Path(resolved)).lower()
            texture_resref = source_resrefs.get(source_key)
            if not texture_resref:
                texture_resref = suggested_obj_texture_resref(
                    material_name,
                    resolved,
                    used=reserved,
                )
                reserved.add(texture_resref)
                source_resrefs[source_key] = texture_resref
                texture_imports.append((material_name, texture_resref, resolved))
            material_texture_resrefs[material_name] = texture_resref

        neutral_texture_resref = ""
        if unresolved_material_names:
            neutral_texture_resref = suggested_obj_texture_resref("obj_neutral", "", used=reserved)
            reserved.add(neutral_texture_resref)
            for material_name in unresolved_material_names:
                material_texture_resrefs[material_name] = neutral_texture_resref

        options = ObjRoomAuthoringOptions(
            room_resref=current_room,
            game=target_game,
            source_units=str(source_units or "auto"),
            source_up_axis=str(source_up_axis or "y"),
            scale_override=scale_override,
            center_xy=bool(center_xy),
            ground_to_zero=bool(ground_to_zero),
        )
        primitive, report = build_obj_room_primitive(
            document,
            options,
            material_texture_resrefs=material_texture_resrefs,
        )
        components = analyze_obj_walkmesh_candidate_components(primitive)
        if not components:
            raise ValueError(
                "The OBJ contains no up-facing floor candidates. Check its source-up axis, scale, and face winding."
            )
        return MapStudioObjRoomImportPreparation(
            source_path=str(source),
            project_path=str(getattr(self.project, "path", "") or ""),
            project_texture_resrefs=existing_texture_resrefs,
            module_root=normalise_resref(current_root),
            options=options,
            document=document,
            primitive=primitive,
            report=report,
            components=tuple(components),
            material_texture_resrefs=tuple(material_texture_resrefs.items()),
            texture_imports=tuple(texture_imports),
            neutral_texture_resref=neutral_texture_resref,
        )

    def commit_obj_room_import(
        self,
        preparation: MapStudioObjRoomImportPreparation,
        *,
        selected_component_ids: tuple[str, ...] | list[str],
        module_root: str = "",
        room_resref: str = "",
        game: str = "",
        target_face_budget: int = 1_800,
        texture_max_dimension: int = 2_048,
    ) -> MapStudioObjRoomImportResult:
        """Commit one reviewed OBJ room, textures, and WOK atomically.

        The KMAP must already have a stable path because imported pixels live
        in project-relative sidecars.  Walkmesh component selection is
        mandatory; recommendations from analysis never authorize collision by
        themselves.
        """

        project_path = str(getattr(self.project, "path", "") or "").strip()
        if not project_path:
            raise ValueError("Save the KMAP before importing an OBJ map and its texture sidecars.")
        if Path(project_path).resolve() != Path(str(preparation.project_path or "")).resolve():
            raise ValueError("The active KMAP changed after OBJ analysis. Analyze the OBJ again before importing.")
        current_texture_resrefs = tuple(
            sorted(
                {
                    str(getattr(texture, "resref", "") or "").strip().lower()
                    for texture in tuple(getattr(self.project, "textures", ()) or ())
                    if str(getattr(texture, "resref", "") or "").strip()
                }
            )
        )
        if current_texture_resrefs != tuple(preparation.project_texture_resrefs):
            raise ValueError("Project textures changed after OBJ analysis. Analyze the OBJ again to allocate safe ResRefs.")

        selected_ids = tuple(dict.fromkeys(str(value) for value in tuple(selected_component_ids or ()) if str(value)))
        if not selected_ids:
            raise ValueError("Review and select at least one floor component before importing the OBJ room.")
        clean_root = str(module_root or preparation.module_root or "").strip()
        clean_room = str(room_resref or preparation.options.room_resref or "").strip()
        target_game = str(game or preparation.options.game or "K2").strip().upper()
        root_issue = authored_resref_blocking_issue("Authored module", clean_root)
        room_issue = authored_resref_blocking_issue("OBJ room", clean_room)
        if root_issue:
            raise ValueError(root_issue)
        if room_issue:
            raise ValueError(room_issue)
        if target_game not in {"K1", "K2"}:
            raise ValueError("OBJ room target game must be K1 or K2.")
        texture_limit = int(texture_max_dimension)
        if texture_limit not in {1_024, 2_048}:
            raise ValueError("OBJ room textures must use the engine-safe 1024 or 2048 maximum dimension policy.")

        options = replace(
            preparation.options,
            room_resref=normalise_resref(clean_room),
            game=target_game,
        )
        primitive, report = build_obj_room_primitive(
            preparation.document,
            options,
            material_texture_resrefs=dict(preparation.material_texture_resrefs),
        )
        components = analyze_obj_walkmesh_candidate_components(primitive)
        reviewed = apply_obj_walkmesh_component_review(
            primitive,
            components,
            selected_component_ids=selected_ids,
            reason=(
                "Explicit Map Studio OBJ import review: only the checked up-facing connected components "
                "are authorized as floor collision."
            ),
        )
        room_primitive, walkmesh_report = generate_adaptive_obj_walkmesh(
            reviewed,
            components,
            selected_component_ids=selected_ids,
            target_face_budget=int(target_face_budget),
        )

        from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
        from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

        wok = room_primitive.wok
        if wok is None or not wok.faces:
            raise ValueError("The reviewed OBJ import did not produce a walkable WOK.")
        entry_face = wok.faces[0]
        entry_vertices = tuple(wok.verts[index] for index in (entry_face.v1, entry_face.v2, entry_face.v3))
        entry_position = tuple(
            sum(float(vertex[axis]) for vertex in entry_vertices) / 3.0
            for axis in range(3)
        )
        room_key = normalise_resref(clean_room)
        room = AuthoredRoomSpec(
            room_resref=room_key,
            primitive=room_primitive,
            visible_rooms=(room_key,),
            metadata={
                "primitive": "imported_mesh",
                "source": "map_studio:obj_room_import",
                "source_path": str(preparation.source_path),
                "source_sha256": str(report.source_sha256),
                "walkmesh_component_review": list(selected_ids),
                "walkmesh_structural_validation": str(walkmesh_report.get("structural_validation") or ""),
            },
        )
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            authored = AuthoredModuleProject(
                metadata=AuthoredModuleMetadata(
                    module_root=normalise_resref(clean_root),
                    game=target_game,
                    display_name=f"{Path(preparation.source_path).stem} custom map",
                    tag=normalise_resref(clean_root),
                    metadata={
                        "source": "map_studio:obj_room_import",
                        "source_path": str(preparation.source_path),
                    },
                ),
                rooms=(room,),
                placements=AuthoredGameplayPlacement(
                    entry_point=ModuleEntryPoint(
                        area_resref=normalise_resref(clean_root),
                        position=entry_position,
                    )
                ),
                notes=(
                    "OBJ geometry and walkmesh are structurally validated authoring output; retail KOTOR proof remains required.",
                ),
            )
        else:
            if normalise_resref(clean_root) != authored.module_root:
                raise ValueError(
                    f"This KMAP already authors module {authored.module_root}; the OBJ import must use that module ResRef."
                )
            if target_game != authored.game:
                raise ValueError(
                    f"This KMAP targets {authored.game}; create a separate KMAP before importing the room for {target_game}."
                )
            if any(existing.normalised_resref() == room_key for existing in authored.rooms):
                raise ValueError(f"Authored room {room_key} already exists. Choose a unique room ResRef.")
            authored = replace(authored, rooms=tuple(authored.rooms) + (room,))

        target_dir = project_texture_directory(self.project)
        sidecar_paths = tuple(
            path
            for _material_name, texture_resref, _source in preparation.texture_imports
            for path in (target_dir / f"{texture_resref}.tga", target_dir / f"{texture_resref}.txi")
        )
        if str(preparation.neutral_texture_resref or ""):
            neutral_target = target_dir / f"{preparation.neutral_texture_resref}.tga"
            sidecar_paths = tuple(sidecar_paths) + (neutral_target, neutral_target.with_suffix(".txi"))
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        created_paths = tuple(value for value, state in sidecar_before.states if state is None)
        imported_assets: list[Any] = []
        try:
            if str(preparation.neutral_texture_resref or ""):
                imported_assets.append(
                    create_project_texture_asset(
                        self.project,
                        resref=str(preparation.neutral_texture_resref),
                        width=2,
                        height=2,
                        rgba=bytes((160, 160, 160, 255)) * 4,
                        source="map_studio:obj_room_neutral_fallback",
                        metadata={
                            "fallback_scope": "obj_material_without_diffuse_image",
                            "source_path": str(preparation.source_path),
                        },
                    )
                )
            for _material_name, texture_resref, source_path in preparation.texture_imports:
                imported_assets.append(
                    import_project_texture_asset(
                        self.project,
                        source_path,
                        resref=texture_resref,
                        max_dimension=texture_limit,
                    )
                )
            self._store_authored_project(authored)
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        sidecar_patches = self.texture_sidecar_journal.finish(
            self.project,
            sidecar_before,
            paths=sidecar_paths,
            created_paths=created_paths,
        )
        self.model.log(
            f"Imported OBJ room {room_key}: {report.triangle_count} render triangle(s), "
            f"{walkmesh_report.get('generated_face_count', 0)} WOK face(s), "
            f"{len(imported_assets)} project texture(s)."
        )
        self._record_map_studio_command(
            action_key="map_studio.room.import_obj",
            label=f"Import OBJ room {room_key}",
            before=before,
            metadata={
                "source_path": str(preparation.source_path),
                "source_sha256": str(report.source_sha256),
                "module_root": normalise_resref(clean_root),
                "room_resref": room_key,
                "game": target_game,
                "render_triangle_count": int(report.triangle_count),
                "selected_component_ids": list(selected_ids),
                "walkmesh_report": dict(walkmesh_report),
                "texture_resrefs": [str(asset.resref) for asset in imported_assets],
                "texture_max_dimension": texture_limit,
            },
            stale_outputs=("MDL", "MDX", "WOK", "LYT", "VIS", "PTH", "ARE", "IFO", "GIT", "TGA", ".mod"),
            readiness_impact=(
                "The custom room must pass Map Studio readiness, vanilla-structural comparison, and a manual retail warp/traversal proof."
            ),
            sidecar_patches=sidecar_patches,
        )
        return MapStudioObjRoomImportResult(
            module_root=normalise_resref(clean_root),
            room_resref=room_key,
            game=target_game,
            imported_texture_resrefs=tuple(str(asset.resref) for asset in imported_assets),
            selected_component_ids=selected_ids,
            walkmesh_report=dict(walkmesh_report),
        )

    def import_project_texture(self, path: str | Path, *, resref: str = ""):
        """Import one unique editable texture beside the saved KMAP."""

        source = Path(path)
        existing = tuple(str(texture.resref or "") for texture in tuple(self.project.textures or ()))
        clean_resref = (
            validate_kotor_texture_resref(resref)
            if str(resref or "").strip()
            else suggest_kotor_texture_resref(source.name, existing)
        )
        target = project_texture_directory(self.project) / f"{clean_resref}.tga"
        sidecar_paths = (target, target.with_suffix(".txi"))
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        created_paths = tuple(value for value, state in sidecar_before.states if state is None)
        try:
            asset = import_project_texture_asset(self.project, path, resref=clean_resref)
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        sidecar_patches = self.texture_sidecar_journal.finish(
            self.project,
            sidecar_before,
            paths=sidecar_paths,
            created_paths=created_paths,
        )
        self.model.log(
            f"Imported project texture {asset.resref} ({asset.width}x{asset.height}); "
            "it will be bundled on the next authored module export."
        )
        self._record_map_studio_command(
            action_key="map_studio.texture.import",
            label=f"Import texture {asset.resref}",
            before=before,
            metadata={
                "texture_id": asset.texture_id,
                "resref": asset.resref,
                "width": asset.width,
                "height": asset.height,
            },
            stale_outputs=("TGA", "TXI", ".mod"),
            readiness_impact="Custom texture resources must be repackaged before game proof.",
            sidecar_patches=sidecar_patches,
        )
        return asset

    def clone_game_textures_for_paint(
        self,
        resrefs: tuple[str, ...] | list[str],
        *,
        resource_manager: Any,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[Any, ...]:
        """Make used game diffuse textures writable in one atomic command.

        ``progress_callback(completed, total, resref)`` runs after each clone.
        Cancellation is checked before every clone and rolls back both KMAP
        state and any sidecars created earlier in the same transaction.
        """

        clean_resrefs = tuple(
            dict.fromkeys(
                validate_kotor_texture_resref(value)
                for value in tuple(resrefs or ())
                if str(value or "").strip()
            )
        )
        if not clean_resrefs:
            return ()
        target_dir = project_texture_directory(self.project)
        sidecar_paths = tuple(
            path
            for resref in clean_resrefs
            for path in (target_dir / f"{resref}.tga", target_dir / f"{resref}.txi")
        )
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        created_paths = tuple(value for value, state in sidecar_before.states if state is None)
        try:
            created_assets: list[Any] = []
            total = len(clean_resrefs)
            for completed, resref in enumerate(clean_resrefs, start=1):
                if callable(cancel_requested) and bool(cancel_requested()):
                    raise MapStudioTextureCloneCancelled(
                        "Making used map textures editable was cancelled."
                    )
                asset = clone_game_texture_asset(
                    self.project,
                    resref,
                    resource_manager=resource_manager,
                    game=str(getattr(self.project, "game", "K1") or "K1"),
                )
                created_assets.append(asset)
                if callable(progress_callback):
                    progress_callback(completed, total, resref)
                if callable(cancel_requested) and bool(cancel_requested()):
                    raise MapStudioTextureCloneCancelled(
                        "Making used map textures editable was cancelled."
                    )
            assets = tuple(created_assets)
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        sidecar_patches = self.texture_sidecar_journal.finish(
            self.project,
            sidecar_before,
            paths=sidecar_paths,
            created_paths=created_paths,
        )
        self._record_map_studio_command(
            action_key="map_studio.texture.make_used_editable",
            label=f"Make {len(assets)} room diffuse texture(s) editable",
            before=before,
            metadata={"resrefs": [asset.resref for asset in assets]},
            stale_outputs=("TGA", "TXI", ".mod"),
            readiness_impact="Project room diffuse texture overrides must be repackaged before game proof.",
            sidecar_patches=sidecar_patches,
        )
        self.model.log(
            f"Made {len(assets)} room diffuse texture(s) editable as module-local TGA overrides."
        )
        return assets

    def authored_project_texture_resources(self) -> tuple[tuple[str, str, bytes], ...]:
        """Return custom texture payloads that will be included in the MOD."""

        return project_texture_export_resources(self.project)

    def set_authored_placeable_resources(self, resources: Any, *, issues: Any = ()) -> None:
        """Inject Placeable Library resources without coupling Scene to Workflow.

        The Placeable Builder workflow owns UTP/dependency resolution.  Map
        Studio owns the final module transaction, so the GUI hands the resolved
        byte resources across this narrow boundary before validation/export.
        """

        normalized: list[tuple[str, str, bytes]] = []
        for item in tuple(resources or ()):
            try:
                resref, restype, data = item
            except (TypeError, ValueError):
                continue
            clean_ref = str(resref or "").strip().lower()[:16]
            clean_type = str(restype or "").strip().lower().lstrip(".")
            if clean_ref and clean_type and data:
                normalized.append((clean_ref, clean_type, bytes(data)))
        next_resources = tuple(normalized)
        changed = next_resources != self._authored_placeable_resources
        self._authored_placeable_resources = next_resources
        self._authored_placeable_resource_issues = tuple(issues or ())
        if changed:
            self._invalidate_map_studio_stock_preview_resources()

    def set_authored_placeable_preview_rows(self, rows: Any) -> None:
        """Supply typed Placeable Library rows used before UTP export/injection.

        This is a preview-only bridge.  The authored UTP remains the engine
        source of truth at export; rows merely let a newly saved library asset
        follow Appearance -> placeables.2da -> modelname immediately.
        """

        normalized: list[dict[str, Any]] = []
        for row in tuple(rows or ()):
            if isinstance(row, dict):
                normalized.append(dict(row))
        next_rows = tuple(normalized)
        if next_rows != self._authored_placeable_preview_rows:
            self._authored_placeable_preview_rows = next_rows
            self._invalidate_map_studio_stock_preview_resources()

    def set_authored_creature_resources(self, resources: Any) -> None:
        """Inject generated per-instance UTC/NCS/DLG resources for final packaging."""

        normalized: list[tuple[str, str, bytes]] = []
        for item in tuple(resources or ()):
            try:
                resref, restype, data = item
            except (TypeError, ValueError):
                continue
            clean_ref = str(resref or "").strip().lower()[:16]
            clean_type = str(restype or "").strip().lower().lstrip(".")
            if clean_ref and clean_type and data:
                normalized.append((clean_ref, clean_type, bytes(data)))
        if tuple(normalized) != self._authored_creature_resources:
            self._authored_creature_resources = tuple(normalized)
            self._invalidate_map_studio_stock_preview_resources()

    def set_authored_scripting_resources(self, resources: Any) -> None:
        """Inject built Scripting Studio resources into Map Studio's export transaction.

        The scripting workbench owns source documents and compilation.  Scene
        owns only immutable, typed resource bytes that have already passed that
        workflow's validation gate.  Duplicate identities with different bytes
        are rejected here so a Map Studio build can never depend on ordering.
        """

        normalized: dict[tuple[str, str], bytes] = {}
        for item in tuple(resources or ()):
            try:
                resref, restype, data = item
            except (TypeError, ValueError):
                continue
            clean_ref = str(resref or "").strip().lower()[:16]
            clean_type = str(restype or "").strip().lower().lstrip(".")
            payload = bytes(data or b"")
            if not clean_ref or not clean_type or not payload:
                continue
            key = (clean_ref, clean_type)
            prior = normalized.get(key)
            if prior is not None and prior != payload:
                raise ValueError(
                    f"Scripting Studio resource collision for {clean_ref}.{clean_type}."
                )
            normalized[key] = payload
        next_resources = tuple(
            (resref, restype, data)
            for (resref, restype), data in sorted(normalized.items())
        )
        if next_resources != self._authored_scripting_resources:
            self._authored_scripting_resources = next_resources
            self._map_studio_pie_context_resource_revision += 1

    def authored_scripting_resource(self, resref: Any, restype: Any) -> bytes | None:
        """Return one staged workbench resource for contextual export resolution."""

        key = (
            str(resref or "").strip().lower()[:16],
            str(restype or "").strip().lower().lstrip("."),
        )
        for item_resref, item_restype, data in self._authored_scripting_resources:
            if (item_resref, item_restype) == key:
                return bytes(data)
        return None

    def _invalidate_map_studio_stock_preview_resources(self) -> None:
        """Drop only resource-derived preview caches after library changes."""

        self._authored_placeable_preview_revision += 1
        self._map_studio_combined_preview_cache = None
        for name in ("_map_studio_stock_template_resolver", "_map_studio_stock_model_cache"):
            if hasattr(self, name):
                delattr(self, name)

    def authored_project_extra_resources(self) -> tuple[tuple[str, str, bytes], ...]:
        """Return all GUI-resolved authored resources for the final MOD."""

        resources = (
            tuple(self.authored_project_texture_resources())
            + tuple(self._authored_placeable_resources)
            + tuple(self._authored_creature_resources)
            + tuple(self._authored_architecture_runtime_resources())
            + tuple(self._authored_scripting_resources)
        )
        result: list[tuple[str, str, bytes]] = []
        seen: dict[tuple[str, str], bytes] = {}
        for resref, restype, data in resources:
            key = (normalise_resref(resref), str(restype or "").strip().lower().lstrip("."))
            payload = bytes(data)
            if key in seen:
                if seen[key] != payload:
                    raise ValueError(
                        f"Authored runtime resource {key[0]}.{key[1]} has conflicting generated content."
                    )
                continue
            seen[key] = payload
            result.append((key[0], key[1], payload))
        return tuple(result)

    def _authored_architecture_runtime_resources(self) -> tuple[tuple[str, str, bytes], ...]:
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return ()
        return (
            tuple(pascal_architecture_runtime_resources(authored))
            + tuple(terrain_kit_runtime_resources(authored))
        )

    def _map_studio_template_resources(self) -> tuple[tuple[str, str, bytes], ...]:
        return (
            tuple(self._authored_placeable_resources)
            + tuple(self._authored_creature_resources)
            + tuple(self._authored_architecture_runtime_resources())
        )

    def _require_authored_creature_resources_ready(self) -> None:
        """Block a package whose GIT points at a generated UTC that is absent."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        placements = dict(payload.get("placements") or {})
        metadata = dict(placements.get("metadata") or {})
        behaviors = dict(metadata.get("creature_behaviors") or {})
        required = {
            str(dict(value).get("generated_template_resref") or "").strip().lower()
            for value in behaviors.values()
            if isinstance(value, dict)
            and str(dict(value).get("faction_role") or "template").strip().lower() != "template"
        }
        bundled = {
            str(resref or "").strip().lower()
            for resref, restype, _data in tuple(self._authored_creature_resources or ())
            if str(restype or "").strip().lower().lstrip(".") == "utc"
        }
        missing = sorted(required - bundled)
        if missing:
            raise ValueError(
                "Creature behavior export is blocked because generated UTC resources were not resolved: "
                + ", ".join(missing)
            )

    def authored_placeable_resource_issues(self) -> tuple[Any, ...]:
        return self._authored_placeable_resource_issues

    def _require_authored_placeable_resources_ready(self) -> None:
        """Keep headless exports from bypassing typed Placeable Library blockers."""

        blocking = [
            str(getattr(issue, "message", issue))
            for issue in tuple(self._authored_placeable_resource_issues or ())
            if str(getattr(issue, "severity", "") or "").strip().lower() in {"blocking", "error"}
        ]
        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        placements = dict(payload.get("placements") or {})
        referenced = {
            str(dict(item or {}).get("template_resref") or dict(item or {}).get("resref") or "").strip().lower()
            for item in tuple(placements.get("placeables") or ())
            if isinstance(item, dict)
        }
        authored_rows = {
            str(row.get("resref") or row.get("template_resref") or "").strip().lower()
            for row in tuple(self._authored_placeable_preview_rows or ())
            if str(row.get("source") or "").strip().lower() == "placeable_builder"
        }
        placement_metadata = dict(placements.get("metadata") or {})
        provenance_rows = dict(placement_metadata.get("instance_provenance") or {})
        provenance_authored = {
            str(dict(value or {}).get("template_resref") or "").strip().lower()
            for value in provenance_rows.values()
            if isinstance(value, dict)
            and str(value.get("library_source") or "").strip().lower() == "placeable_builder"
        }
        bundled_utps = {
            str(resref or "").strip().lower()
            for resref, restype, _data in tuple(self._authored_placeable_resources or ())
            if str(restype or "").strip().lower().lstrip(".") == "utp"
        }
        missing = sorted(((referenced & authored_rows) | provenance_authored) - bundled_utps)
        if missing:
            blocking.append(
                "Referenced Placeable Builder UTP resources were not resolved for export: " + ", ".join(missing)
            )
        if blocking:
            raise ValueError("Placeable Library export is blocked: " + " ".join(blocking))

    def commit_project_texture_paint(self, texture_id: str, session, *, stroke_result: Any = None):
        """Persist one crash-safe draft stroke and mark it unapplied for export."""

        sidecar_paths = managed_project_texture_sidecars(self.project, texture_id=str(texture_id))
        if not sidecar_paths:
            raise ValueError("Texture Paint target has no project sidecar path.")
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        try:
            asset = save_texture_paint_session(self.project, texture_id, session)
            payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
            if isinstance(payload, dict):
                payload = dict(payload)
                self._clear_authored_export_proof_invalidation(payload, clear_stage_metadata=True)
                payload["texture_paint_dirty"] = True
                payload["texture_paint_unapplied"] = True
                payload["texture_paint_resref"] = asset.resref
                payload["texture_paint_pending_resrefs"] = list(
                    dict.fromkeys((*texture_paint_pending_resrefs(payload), asset.resref))
                )
                self.project.extra_sections["authored_module"] = payload
            texture = next(
                (
                    item
                    for item in tuple(getattr(self.project, "textures", ()) or ())
                    if str(getattr(item, "texture_id", "") or "") == str(asset.texture_id)
                ),
                None,
            )
            if texture is not None:
                texture.metadata = {
                    **dict(getattr(texture, "metadata", {}) or {}),
                    "paint_unapplied": True,
                }
            ranges_by_path: dict[str, tuple[tuple[int, int], ...]] = {}
            dirty_tiles = tuple(getattr(stroke_result, "dirty_tiles", ()) or ())
            if dirty_tiles:
                tga_bytes = Path(asset.path).read_bytes()
                ranges_by_path[str(Path(asset.path).resolve())] = tga_dirty_tile_byte_ranges(
                    width=int(session.width),
                    height=int(session.height),
                    tile_size=int(session.tile_size),
                    dirty_tiles=dirty_tiles,
                    tga_bytes=tga_bytes,
                )
            sidecar_patches = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                ranges_by_path=ranges_by_path,
            )
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        delta_bytes = sum(patch.stored_byte_count for patch in sidecar_patches)
        self._record_map_studio_command(
            action_key="map_studio.texture.paint_stroke",
            label=f"Texture Paint Stroke {asset.resref}",
            before=before,
            stale_outputs=("TGA", "TXI", ".mod"),
            readiness_impact="Painted texture resources must be repackaged before game proof.",
            metadata={
                "texture_id": str(texture_id),
                "resref": asset.resref,
                "dirty_tile_count": len(tuple(getattr(stroke_result, "dirty_tiles", ()) or ())),
                "sidecar_delta_bytes": int(delta_bytes),
            },
            sidecar_patches=sidecar_patches,
        )
        self.model.log(
            f"Saved texture-paint draft stroke to {asset.resref}; click Apply Texture Changes before export."
        )
        return asset

    def has_unapplied_project_texture_changes(self) -> bool:
        """Return whether the KMAP contains live paint drafts not finalized by the user."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        return texture_paint_has_unapplied_changes(payload)

    @staticmethod
    def _texture_resource_hashes(
        resources: tuple[tuple[str, str, bytes], ...] | list[tuple[str, str, bytes]],
    ) -> dict[tuple[str, str], str]:
        return {
            (str(resref).strip().lower(), str(restype).strip().lower()): hashlib.sha256(data).hexdigest()
            for resref, restype, data in tuple(resources or ())
        }

    def _project_texture_applied_resource_drift(
        self,
        resources: tuple[tuple[str, str, bytes], ...] | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Return applied resource keys whose current sidecar bytes differ."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if not isinstance(payload, dict):
            return ()
        applied_records = tuple(payload.get("texture_paint_applied_resources") or ())
        tracked_textures = tuple(
            texture
            for texture in tuple(getattr(self.project, "textures", ()) or ())
            if str(dict(getattr(texture, "metadata", {}) or {}).get("paint_applied_sha256") or "").strip()
        )
        if not applied_records and not tracked_textures:
            return ()
        current_resources = self._texture_resource_hashes(
            tuple(self.authored_project_texture_resources()) if resources is None else resources
        )
        drifted: list[tuple[str, str]] = []
        for record in applied_records:
            if not isinstance(record, dict):
                continue
            key = (
                str(record.get("resref") or "").strip().lower(),
                str(record.get("restype") or "").strip().lower(),
            )
            expected = str(record.get("sha256") or "").strip().lower()
            if not key[0] or not key[1] or not expected:
                continue
            if current_resources.get(key) != expected:
                drifted.append(key)
        for texture in tracked_textures:
            metadata = dict(getattr(texture, "metadata", {}) or {})
            expected = str(metadata.get("paint_applied_sha256") or "").strip().lower()
            if not expected:
                continue
            resref = str(getattr(texture, "resref", "") or "").strip().lower()
            path_value = str(getattr(texture, "path", "") or "").strip()
            restype = Path(path_value).suffix.lower().lstrip(".") if path_value else "tga"
            key = (resref, restype)
            if current_resources.get(key) != expected:
                drifted.append(key)
        return tuple(dict.fromkeys(drifted))

    def project_texture_reapply_resrefs(self) -> tuple[str, ...]:
        """Return applied textures whose current sidecars need re-acceptance."""

        return tuple(
            dict.fromkeys(resref for resref, _restype in self._project_texture_applied_resource_drift())
        )

    def project_texture_apply_pending_resrefs(self) -> tuple[str, ...]:
        """Return paint drafts plus hash-drifted resources for the Apply All UI."""

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if not isinstance(payload, dict):
            return ()
        explicit = texture_paint_pending_resrefs(payload)
        if texture_paint_has_unapplied_changes(payload) and not explicit:
            explicit = tuple(
                dict.fromkeys(
                    str(resref or "").strip().lower()
                    for resref, _restype, _data in self.authored_project_texture_resources()
                    if str(resref or "").strip()
                )
            )
        return tuple(dict.fromkeys((*explicit, *self.project_texture_reapply_resrefs())))

    def project_texture_apply_required(self) -> bool:
        """Return whether Apply All must accept drafts or changed sidecars."""

        return bool(
            self.has_unapplied_project_texture_changes()
            or self.project_texture_reapply_resrefs()
        )

    def _require_applied_project_texture_changes(self) -> None:
        if self.has_unapplied_project_texture_changes():
            raise ValueError(TEXTURE_PAINT_UNAPPLIED_BLOCKER)
        drifted = self._project_texture_applied_resource_drift()
        if drifted:
            resref, restype = drifted[0]
            raise ValueError(
                f'Project texture "{resref}.{restype}" changed after Apply Texture Changes. '
                "Apply Textures again before export."
            )

    def apply_project_texture_changes(self) -> dict[str, Any]:
        """Finalize drafted TGA/TXI sidecars as the texture set eligible for export.

        Paint strokes remain file-backed, crash-safe drafts so live preview and
        global Undo can operate without embedding pixels in KMAP.  Apply is a
        lightweight, explicit acceptance transaction: it validates/reads the
        pending export resources, records deterministic hashes and revisions,
        and clears the export-blocking draft state.  It never changes source
        game textures, mesh UV0, or lightmap UV data.
        """

        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if not isinstance(payload, dict):
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load one first.")
        has_drafts = texture_paint_has_unapplied_changes(payload)
        has_applied_records = bool(payload.get("texture_paint_applied_resources")) or any(
            str(dict(getattr(texture, "metadata", {}) or {}).get("paint_applied_sha256") or "").strip()
            for texture in tuple(getattr(self.project, "textures", ()) or ())
        )
        if not has_drafts and not has_applied_records:
            return {
                "applied": False,
                "resource_count": 0,
                "resrefs": (),
                "message": "No unapplied texture changes are waiting.",
            }
        resources = tuple(self.authored_project_texture_resources())
        drifted_resrefs = tuple(
            dict.fromkeys(
                resref
                for resref, _restype in self._project_texture_applied_resource_drift(resources)
            )
        )
        if not has_drafts and not drifted_resrefs:
            return {
                "applied": False,
                "resource_count": 0,
                "resrefs": (),
                "message": "No unapplied texture changes are waiting.",
            }

        before = self._capture_map_studio_command_state()
        pending_resrefs = texture_paint_pending_resrefs(payload)
        if has_drafts and not pending_resrefs:
            pending_resrefs = tuple(
                dict.fromkeys(
                    str(resref or "").strip().lower()
                    for resref, _restype, _data in resources
                    if str(resref or "").strip()
                )
            )
        requested_resrefs = tuple(dict.fromkeys((*pending_resrefs, *drifted_resrefs)))
        pending_resources = tuple(
            (resref, restype, data)
            for resref, restype, data in resources
            if str(resref or "").strip().lower() in requested_resrefs
        )
        finalized_resrefs = tuple(
            dict.fromkeys(str(resref or "").strip().lower() for resref, _restype, _data in pending_resources)
        )
        if not pending_resources:
            raise ValueError(
                "Cannot apply Texture Paint changes because no project-owned TGA/TPC sidecar is available."
            )
        if requested_resrefs:
            missing = tuple(resref for resref in requested_resrefs if resref not in finalized_resrefs)
            if missing:
                raise ValueError(
                    "Cannot apply Texture Paint changes because project sidecars are missing for: "
                    + ", ".join(missing)
                )

        applied_records = [
            {
                "resref": str(resref).strip().lower(),
                "restype": str(restype).strip().lower(),
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for resref, restype, data in pending_resources
        ]
        manifest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for record in tuple(payload.get("texture_paint_applied_resources") or ()):
            if not isinstance(record, dict):
                continue
            key = (
                str(record.get("resref") or "").strip().lower(),
                str(record.get("restype") or "").strip().lower(),
            )
            if key[0] and key[1] and key[0] not in finalized_resrefs:
                manifest_by_key[key] = dict(record)
        for record in applied_records:
            manifest_by_key[(str(record["resref"]), str(record["restype"]))] = record
        manifest = list(manifest_by_key.values())
        hashes_by_resref = {
            str(item["resref"]): str(item["sha256"])
            for item in applied_records
            if str(item["restype"]).lower() in {"tga", "tpc"}
        }
        for texture in tuple(getattr(self.project, "textures", ()) or ()):
            resref = str(getattr(texture, "resref", "") or "").strip().lower()
            if resref not in finalized_resrefs:
                continue
            metadata = dict(getattr(texture, "metadata", {}) or {})
            metadata["paint_unapplied"] = False
            metadata["paint_applied_revision"] = int(metadata.get("paint_revision", 0) or 0)
            if resref in hashes_by_resref:
                metadata["paint_applied_sha256"] = hashes_by_resref[resref]
            texture.metadata = metadata

        updated_payload = dict(payload)
        updated_payload["texture_paint_dirty"] = False
        updated_payload["texture_paint_unapplied"] = False
        updated_payload["texture_paint_pending_resrefs"] = []
        updated_payload.pop("texture_paint_resref", None)
        updated_payload["texture_paint_applied_resources"] = manifest
        updated_payload["texture_paint_applied_revision"] = int(
            updated_payload.get("texture_paint_applied_revision", 0) or 0
        ) + 1
        self.project.extra_sections["authored_module"] = updated_payload
        self.project.dirty = True

        self._record_map_studio_command(
            action_key="map_studio.texture.apply_changes",
            label="Apply Texture Changes",
            before=before,
            stale_outputs=("TGA", "TXI", ".mod"),
            readiness_impact="Texture paint drafts are finalized and eligible for the next module export.",
            summary="Accepted the current project-owned texture sidecars without changing UV or lightmap data.",
            metadata={
                "resrefs": list(finalized_resrefs),
                "resource_count": len(applied_records),
                "applied_revision": updated_payload["texture_paint_applied_revision"],
            },
        )
        message = (
            f"Applied texture changes for {len(finalized_resrefs)} texture(s); "
            "the current sidecars are eligible for module export."
        )
        self.model.log(message)
        return {
            "applied": True,
            "resource_count": len(applied_records),
            "resrefs": finalized_resrefs,
            "message": message,
        }

    def add_module(self, name: str, *, source_path: str = "", game: str | None = None):
        module = LevelScene(self.project).add_module(name, source_path=source_path, game=game)
        self.model.active_module_id = module.module_id
        self.model.select(module.module_id)
        self.model.log(f"Added module {module.module_name}.")
        return module

    def remove_selected(self) -> bool:
        item_id = self.model.selected_ids[0] if self.model.selected_ids else ""
        if not item_id:
            return False
        if item_id.startswith("authored:"):
            update = self.remove_authored_gameplay_placement(item_id)
            self.model.select("")
            self.model.log(f"Deleted Map Studio {update.kind} placement {update.tag}.")
            return True
        if item_id.startswith("authored_light:"):
            update = self.remove_authored_room_light(item_id)
            self.model.select("")
            self.model.log(f"Deleted Map Studio room light {update.light.name}.")
            return True
        scene = LevelScene(self.project)
        changed = scene.remove_room(item_id) or scene.remove_module(item_id)
        if changed:
            self.model.select("")
            self.model.log(f"Deleted {item_id}.")
        return changed

    def duplicate_selected(self):
        item_id = self.model.selected_ids[0] if self.model.selected_ids else ""
        if item_id.startswith("authored:"):
            update = self.duplicate_authored_gameplay_placement(item_id)
            self.model.select(update.placement_id)
            self.model.log(f"Duplicated Map Studio {update.kind} placement {update.tag}.")
            return update
        if item_id.startswith("authored_light:"):
            update = self.duplicate_authored_room_light(item_id)
            self.model.select(update.light_id)
            self.model.log(f"Duplicated Map Studio room light {update.light.name}.")
            return update
        clone = LevelScene(self.project).duplicate_room(item_id)
        if clone is not None:
            self.model.select(clone.room_id)
            self.model.log(f"Duplicated room {clone.name}.")
        return clone

    def load_lyt(self, path: str | Path):
        return self.layout_service.load_lyt_file(self.project, path, module_id=self.model.active_module_id)

    def import_stock_module_from_rim(
        self,
        *,
        module_resref: str,
        modules_dir: str,
        game: str = "",
        resource_manager: Any = None,
    ) -> tuple[bool, str]:
        """Import a complete stock KOTOR module into an editable authored project.

        Reads ARE/GIT/IFO/LYT/VIS from the module RIM, loads room MDL geometry,
        and populates the authored module project with all placements, rooms,
        lights, and metadata.

        Returns ``(ok, message)``.
        """

        from pathlib import Path as _Path

        resref = str(module_resref or "").strip().lower()
        if not resref:
            return False, "No module resref provided."

        game_tag = str(game or self.project.game or "K1").upper()
        modules_path = _Path(str(modules_dir or "").strip())
        if not modules_path.exists():
            return False, f"Modules directory not found: {modules_path}"

        # KOTOR modules are .rim files; some are .mod (which is a RIM container).
        rim_path = modules_path / f"{resref}.rim"
        if not rim_path.exists():
            rim_path = modules_path / f"{resref}.mod"
        if not rim_path.exists():
            return False, f"Module RIM not found: {rim_path}"

        self.model.log(f"Importing stock module {resref} ({game_tag}) from {rim_path.name}...")

        # Custom community modules bundle their own room MDL/MDX/WOK + textures
        # inside the capsule rather than referencing base-game rooms. Overlay
        # them so model resolution (import + convert-to-editable) finds them.
        # RIM-based modules split gameplay templates (UTC/UTP/UTD/...) into an
        # "_s.rim" companion, so overlay that too when it exists; without it,
        # creatures/doors/placeables render as markers instead of real models.
        if resource_manager is not None and hasattr(resource_manager, "add_module_overlay"):
            self._index_import_source_capsules(resource_manager, rim_path, clear_existing=True)

            # Classic module releases put only ARE/GIT/IFO in Modules/*.mod
            # and ship the actual LYT/VIS/MDL/MDX/WOK/textures loose beside it
            # (usually in a sibling Override directory).  Index that bundle
            # lazily.  A real game installation is deliberately excluded: its
            # chitin.key/Override/Modules layers are already indexed and a
            # recursive scan there would be both redundant and expensive.
            add_loose_overlay = getattr(resource_manager, "add_loose_overlay", None)
            bundle_root = rim_path.parent
            if bundle_root.name.strip().lower() in {"module", "modules"}:
                bundle_root = bundle_root.parent
            # MediaFire/archive staging keeps each release below
            # ``Extracted/<package>`` and some releases share textures or room
            # binaries from a sibling subfolder inside that package.  Expand
            # to that one package boundary, never to the whole collection.
            for ancestor in rim_path.parents:
                if ancestor.parent.name.strip().lower() == "extracted":
                    bundle_root = ancestor
                    break
            if callable(add_loose_overlay) and not (bundle_root / "chitin.key").is_file():
                try:
                    loose_count = int(add_loose_overlay(str(bundle_root), recursive=True) or 0)
                    if loose_count:
                        self.model.log(
                            f"Indexed {loose_count} loose companion resource(s) from {bundle_root.name} for editing."
                        )
                except Exception as exc:
                    self.model.log(f"Could not index loose module companions from {bundle_root}: {exc}")
            # The manager object is intentionally reused across imports, so
            # object identity cannot reveal that its overlay bytes changed.
            self._invalidate_map_studio_stock_preview_resources()

        # Model loader closure.
        _loader_cache: dict[str, Any] = {}

        def _model_loader(resref_arg: str, game_arg: str):
            cache_key = f"{resref_arg.lower()}|{game_arg}"
            if cache_key in _loader_cache:
                return _loader_cache[cache_key]
            model = load_stock_kotor_model(resource_manager, resref_arg, game_arg)
            _loader_cache[cache_key] = model
            return model

        result = import_stock_module(
            module_resref=resref,
            game=game_tag,
            rim_path=rim_path,
            resource_provider=resource_manager,
            model_loader=_model_loader if resource_manager is not None else None,
        )

        if result.project is None:
            errors = "; ".join(result.errors) if result.errors else "Unknown import error."
            self.model.log(f"  Import FAILED: {errors}")
            return False, errors

        # Convert to KMAP payload and store.
        payload = authored_project_to_kmap_payload(result.project)
        self.project.extra_sections["authored_module"] = payload
        self.project.name = result.project.metadata.module_root
        self.project.game = game_tag
        self.project.dirty = True
        self._invalidate_map_studio_authored_state("stock module import")

        # Log summary.
        summary_parts = [f"{result.room_count} rooms"]
        for kind, count in sorted(result.placement_counts.items()):
            if count:
                summary_parts.append(f"{count} {kind}")
        self.model.log(f"  Imported: {', '.join(summary_parts)}.")
        for warning in result.warnings:
            self.model.log(f"  Warning: {warning}")

        return True, f"Imported {resref}: {', '.join(summary_parts)}."

    # ------------------------------------- Map Studio component mesh editing

    def _load_authored_project_or_raise(self):
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            raise ValueError(
                "No authored Map Studio module is stored in this KMAP. Import the stock module (File -> Import Stock Module) first."
            )
        return authored

    def _store_authored_project(self, updated) -> None:
        # Any authored mutation can invalidate the imported module's original
        # path graph. Preserve the byte-exact PTH only while the stock import
        # remains untouched; edited projects must regenerate pathing.
        updated_extra = dict(getattr(updated, "extra", {}) or {})
        stock_resources = dict(updated_extra.get("stock_resources") or {})
        if stock_resources.get("pth") and not updated_extra.get("stock_pth_dirty"):
            from dataclasses import replace as _replace

            updated_extra["stock_pth_dirty"] = True
            updated_extra["stock_pth_preserved"] = False
            updated_extra["stock_pth_dirty_reason"] = (
                "Map Studio authored state changed after stock import; regenerate PTH from current walkable geometry."
            )
            updated = _replace(updated, extra=updated_extra)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self._invalidate_map_studio_authored_state("store authored project")

    def imported_mesh_room(self, room_resref: str):
        """Return the authored imported-mesh room spec for a resref, or None."""

        try:
            authored = self._load_authored_project_or_raise()
        except ValueError:
            return None
        resref = normalise_resref(room_resref)
        for room in authored.rooms:
            if room.normalised_resref() == resref and isinstance(room.primitive, ImportedMeshRoomPrimitive):
                return room
        return None

    def last_committed_imported_mesh_room(self, room_resref: str):
        """Return the just-committed mesh room without decoding KMAP again.

        The cache is scoped to the exact payload object produced by the edit.
        Any project replacement, undo/redo, or structural mutation misses and
        callers fall back to :meth:`imported_mesh_room`.
        """

        cached = getattr(self, "_last_committed_imported_mesh_room", None)
        if not cached:
            return None
        payload, cached_resref, room = cached
        current = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        wanted = normalise_resref(room_resref)
        if payload is not current or cached_resref != wanted:
            return None
        if not isinstance(getattr(room, "primitive", None), ImportedMeshRoomPrimitive):
            return None
        return room

    def _stock_room_wok_bytes(self, resref: str, resource_manager: Any, authored: Any) -> bytes | None:
        """Fetch the stock room's WOK bytes: Override/KEY-BIF, then the import RIM."""

        getter = getattr(resource_manager, "get", None)
        if callable(getter):
            try:
                data = getter(resref, 2016, str(getattr(authored, "game", "K1") or "K1").upper())
                if data:
                    return bytes(data)
            except Exception:
                pass
        source = str((getattr(authored, "extra", {}) or {}).get("import_source") or "")
        if source:
            try:
                from pathlib import Path as _Path

                from pykotor.extract.capsule import LazyCapsule
                from pykotor.resource.type import ResourceType as RT

                rim = _Path(source)
                if rim.exists():
                    data = LazyCapsule(rim).resource(resref, RT.WOK)
                    if data:
                        return bytes(data)
            except Exception:
                pass
        return None

    def convert_stock_room_to_imported_mesh(
        self,
        *,
        room_resref: str,
        resource_manager: Any = None,
        position: Any = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Bake one stock KOTOR room into editable imported-mesh authored geometry.

        The room keeps its LYT position and original UVs; the stock KMAP room
        row is retired so the authored copy renders instead of the read-only
        preview.  Recorded as one undoable command.  ``position`` overrides the
        placement for a newly added room (used when pulling a room in from
        another module); ``extra_metadata`` records provenance such as the
        source module for later snapping.
        """

        from dataclasses import replace as _replace

        resref = normalise_resref(room_resref)
        if not resref:
            return False, "No room resref provided."
        position_override = position
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            # LYT-loaded loose rooms have no authored module yet; create a
            # minimal one so "edit any loaded map" works without a full
            # stock-module import first.
            from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
            from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

            root = normalise_resref(str(getattr(self.project, "name", "") or "") ) or resref
            authored = AuthoredModuleProject(
                metadata=AuthoredModuleMetadata(
                    module_root=root,
                    game=str(self.project.game or "K1").upper(),
                    display_name=f"{root} (edited stock map)",
                    tag=root,
                ),
                rooms=(),
                placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref=root)),
                lights=(),
            )
            self.model.log(
                f"Created authored module {root} automatically so stock room {resref} can become editable."
            )
        else:
            authored = self._load_authored_project_or_raise()
        existing = next((room for room in authored.rooms if room.normalised_resref() == resref), None)
        if existing is not None and isinstance(existing.primitive, ImportedMeshRoomPrimitive):
            return True, f"Room {resref} is already editable."
        if resource_manager is None:
            return False, "Connect a KOTOR game directory first so the stock room model can be loaded."
        before = self._capture_map_studio_command_state()
        game = str(self.project.game or "K1").upper()
        model = load_stock_kotor_model(resource_manager, resref, game)
        if model is None:
            return False, f"Stock room model {resref} could not be loaded from the {game} game resources."
        wok_bytes = self._stock_room_wok_bytes(resref, resource_manager, authored)
        primitive = build_imported_mesh_primitive_from_stock_model(
            model,
            room_resref=resref,
            source_model=resref,
            game=game,
            wok_bytes=wok_bytes,
        )
        if not primitive.surfaces:
            return False, f"Stock room model {resref} has no editable render surfaces."

        position = (0.0, 0.0, 0.0)
        stock_rows = [
            room
            for room in tuple(getattr(self.project, "rooms", ()) or ())
            if str(getattr(room, "model_resref", "") or getattr(room, "name", "") or "").strip().lower() == resref
        ]
        metadata = {"source": "stock_room_conversion", "source_model": resref}
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        # Individual backdrop surfaces keep their stable surface indices and
        # are filtered only by viewport visibility/picking. A whole room is a
        # backdrop-only room only when every render surface is backdrop and its
        # stock WOK contributes no walkable floor; mixed K2 rooms (231telsb,
        # 151harsb) must keep their real geometry and pathing.
        backdrop_only = imported_mesh_room_is_backdrop(primitive)
        metadata["backdrop_only"] = backdrop_only
        metadata["is_backdrop"] = backdrop_only  # legacy KMAP/readiness key
        metadata["backdrop_surface_indices"] = [
            index
            for index, surface in enumerate(primitive.surfaces)
            if bool(getattr(surface, "backdrop", False))
        ]
        if existing is not None:
            position = tuple(float(v) for v in tuple(existing.position or (0.0, 0.0, 0.0))[:3])
            metadata = {**dict(existing.metadata or {}), **metadata}
            metadata.pop("pie_exclude_unresolved_stock_geometry", None)
            metadata.pop("stock_geometry_issue", None)
            metadata["stock_geometry_status"] = "resolved"
            rooms = tuple(
                room
                if room.normalised_resref() != resref
                else AuthoredRoomSpec(
                    room_resref=resref,
                    primitive=primitive,
                    position=position,
                    visible_rooms=room.visible_rooms,
                    metadata=metadata,
                )
                for room in authored.rooms
            )
        else:
            if position_override is not None:
                position = tuple(float(v) for v in tuple(position_override)[:3])
            elif stock_rows:
                transform = getattr(stock_rows[0], "transform", None)
                position = tuple(
                    float(v) for v in tuple(getattr(transform, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))[:3]
                )
            else:
                position = (0.0, 0.0, 0.0)
            rooms = tuple(authored.rooms) + (
                AuthoredRoomSpec(room_resref=resref, primitive=primitive, position=position, metadata=metadata),
            )
        self._store_authored_project(_replace(authored, rooms=rooms))
        if stock_rows:
            keep = [
                room
                for room in tuple(getattr(self.project, "rooms", ()) or ())
                if str(getattr(room, "model_resref", "") or getattr(room, "name", "") or "").strip().lower() != resref
            ]
            self.project.rooms = list(keep)
        wok_note = "with its stock walkmesh" if primitive.wok is not None else "with a fallback floor walkmesh"
        self.model.log(
            f"Converted stock room {resref} into editable geometry ({len(primitive.surfaces)} surfaces, {wok_note})."
        )
        self._record_map_studio_command(
            action_key="map_studio.imported_mesh.convert",
            label=f"Make room {resref} editable",
            before=before,
            metadata={"room_resref": resref, "surface_count": len(primitive.surfaces)},
        )
        return True, f"Room {resref} is now editable ({len(primitive.surfaces)} surfaces, {wok_note})."

    def add_catalog_room_to_project(
        self,
        *,
        room_resref: str,
        source_path: str,
        source_module: str = "",
        game: str = "",
        resource_manager: Any = None,
        position: Any = None,
        connection_points: Any = None,
    ) -> tuple[bool, str]:
        """Add a catalog-indexed room from another module into the current project.

        Overlays the room's source ``.mod``/``.rim`` so its model/WOK resolve,
        then bakes it into editable imported-mesh geometry positioned at
        ``position`` (defaults just past the current module's east edge so it
        does not overlap existing rooms, ready to be snapped into place). The
        source module and the room's doorway connection points are recorded on
        the room for later doorway snapping.
        """

        from pathlib import Path as _Path

        resref = normalise_resref(room_resref)
        if not resref:
            return False, "No room resref provided."
        if resource_manager is None:
            return False, "Connect a KOTOR game directory first so the source room model can be loaded."
        capsule = _Path(str(source_path or "").strip())
        if capsule.suffix.lower() in {".mod", ".rim", ".erf"} and capsule.is_file():
            add_overlay = getattr(resource_manager, "add_module_overlay", None)
            if callable(add_overlay):
                try:
                    add_overlay(str(capsule))
                    companion = capsule.with_name(f"{capsule.stem}_s.rim")
                    if capsule.suffix.lower() == ".rim" and companion.exists():
                        add_overlay(str(companion))
                except Exception as exc:
                    self.model.log(f"Could not overlay source module {capsule.name}: {exc}")
            self._invalidate_map_studio_stock_preview_resources()
        # A room already in this project would otherwise silently no-op; report it.
        snapshot = self._map_studio_authored_project_snapshot()
        if snapshot is not None and any(room.normalised_resref() == resref for room in tuple(getattr(snapshot, "rooms", ()) or ())):
            return False, f"Room {resref} is already in this module. Rename or remove it before re-adding."
        drop_position = position if position is not None else self._next_added_room_position()
        serialised_hooks = [
            {
                "door": str(getattr(hook, "door", "") or (hook.get("door", "") if isinstance(hook, dict) else "")),
                "local_position": [float(v) for v in tuple(getattr(hook, "local_position", None) or (hook.get("local_position", (0.0, 0.0, 0.0)) if isinstance(hook, dict) else (0.0, 0.0, 0.0)))[:3]],
                "orientation": [float(v) for v in tuple(getattr(hook, "orientation", None) or (hook.get("orientation", (0.0, 0.0, 0.0, 1.0)) if isinstance(hook, dict) else (0.0, 0.0, 0.0, 1.0)))[:4]],
            }
            for hook in tuple(connection_points or ())
        ]
        return self.convert_stock_room_to_imported_mesh(
            room_resref=resref,
            resource_manager=resource_manager,
            position=drop_position,
            extra_metadata={
                "catalog_source_path": str(capsule),
                "catalog_source_module": normalise_resref(source_module) or normalise_resref(capsule.stem),
                "added_from_catalog": True,
                "connection_points": serialised_hooks,
            },
        )

    def available_map_studio_terrain_kit_assets(self) -> tuple[dict[str, Any], ...]:
        """Return portable static-geometry assets for Terrain Building."""

        return terrain_kit_asset_rows(game=str(getattr(self.project, "game", "K1") or "K1").upper())

    def available_map_studio_environment_kits(self, *, kind: str = "") -> tuple[dict[str, Any], ...]:
        """Return the trained vanilla module collections for the active game."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        return environment_kit_collection_rows(game=game, kind=kind)

    def available_map_studio_environment_kit_pieces(self, *, kind: str = "") -> tuple[dict[str, Any], ...]:
        """Return typed vanilla room/exterior tiles for the visual kit browser."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        return environment_kit_piece_rows(game=game, kind=kind)

    def available_map_studio_building_styles(self) -> tuple[dict[str, Any], ...]:
        """Return KOTOR-safe surface styles for the direct wall tool."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        return tuple(
            {
                "style_id": style.style_id,
                "label": style.label,
                "game": style.game,
                "floor_texture": style.floor_texture,
                "wall_texture": style.wall_texture,
                "ceiling_texture": style.ceiling_texture,
                "source_module": style.source_module,
                "source_room": style.source_room,
                "architecture_profile": style.architecture_profile,
                "architecture_shell_profile": style.architecture_shell_profile,
                "architecture_archetypes": tuple(
                    {
                        "archetype_id": archetype.archetype_id,
                        "label": archetype.label,
                        "shell_profile": archetype.shell_profile,
                        "recommended_wall_height_m": float(archetype.recommended_wall_height_m),
                        "recommended_floor_to_floor_m": float(archetype.recommended_floor_to_floor_m),
                        "description": archetype.description,
                        "evidence_rooms": tuple(archetype.evidence_rooms),
                    }
                    for archetype in style.architecture_archetypes
                ),
                "recommended_wall_height_m": float(style.recommended_wall_height_m),
                "recommended_floor_to_floor_m": float(style.recommended_floor_to_floor_m),
                "recommended_door_width_m": float(style.recommended_door_width_m),
                "recommended_door_height_m": float(style.recommended_door_height_m),
                "accent_textures": tuple(style.accent_textures),
                "evidence_rooms": tuple(style.evidence_rooms),
                "world_label": kotor_module_world_label(style.game, style.source_module),
                "environment_kind": next(
                    (tag for tag in tuple(style.tags) if tag in {"interior", "exterior"}),
                    "both",
                ),
                "collection_id": style.style_id[4:] if style.style_id.startswith("kit:") else "",
                "tags": tuple(style.tags),
            }
            for style in available_pascal_building_styles(game)
        )

    def map_studio_building_levels(self):
        """Return authored Pascal-style levels without changing KMAP state."""

        authored = self._map_studio_authored_project_snapshot()
        return () if authored is None else pascal_building_levels_for_project(authored)

    def refresh_map_studio_vanilla_building_styles(
        self,
        *,
        resource_manager: Any,
        progress: Any = None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Rebuild the local retail room-surface palette catalog."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        styles = scan_vanilla_pascal_building_styles(
            resource_manager,
            games=(game,),
            progress=progress,
        )
        retained = tuple(style for style in vanilla_pascal_building_styles() if style.game != game)
        path = write_vanilla_pascal_style_catalog(retained + styles)
        return self.available_map_studio_building_styles(), str(path)

    def refresh_map_studio_vanilla_environment_kits(
        self,
        *,
        resource_manager: Any,
        progress: Any = None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Retrain module collections from the configured base-game layouts."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        collections = scan_vanilla_environment_kits(
            resource_manager,
            games=(game,),
            progress=progress,
        )
        retained = tuple(
            collection for collection in vanilla_environment_kit_collections() if collection.game != game
        )
        path = write_vanilla_environment_kit_catalog(retained + collections)
        return self.available_map_studio_environment_kit_pieces(), str(path)

    def _map_studio_building_project(self):
        """Return the authored module, creating the lightweight KMAP owner when needed."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is not None:
            return authored
        from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
        from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

        root = normalise_resref(str(getattr(self.project, "name", "") or "")) or "grbuild"
        return AuthoredModuleProject(
            metadata=AuthoredModuleMetadata(
                module_root=root,
                game=str(getattr(self.project, "game", "K1") or "K1").upper(),
                display_name=f"{root} authored map",
                tag=root,
            ),
            rooms=(),
            placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref=root)),
        )

    @staticmethod
    def _with_automatic_room_walkmesh_status(
        project: Any,
        *,
        operation: str,
        room_resrefs: tuple[str, ...],
    ) -> Any:
        """Compile and audit touched room WOKs before one build action commits."""

        requested = {normalise_resref(value) for value in tuple(room_resrefs or ()) if normalise_resref(value)}
        audits: list[Any] = []
        for room in tuple(project.rooms or ()):
            room_resref = room.normalised_resref()
            if requested and room_resref not in requested:
                continue
            geometry = compile_authored_room_spec(room)
            audit = audit_authored_wok(room_resref, geometry.wok)
            if not audit.ready:
                raise ValueError(" ".join(audit.blocking_messages))
            audits.append(audit)
        if requested and {audit.room_resref for audit in audits} != requested:
            missing = ", ".join(sorted(requested - {audit.room_resref for audit in audits}))
            raise ValueError(f"Automatic walkmesh generation could not find room(s): {missing}.")
        extra = dict(project.extra or {})
        extra["walkmesh_generation_mode"] = "automatic_room_assembly"
        extra["last_walkmesh_build"] = {
            "operation": str(operation or "room_build"),
            "auto_generated": True,
            "room_resrefs": [audit.room_resref for audit in audits],
            "room_face_counts": {audit.room_resref: int(audit.face_count) for audit in audits},
            "walkable_face_counts": {audit.room_resref: int(audit.walkable_face_count) for audit in audits},
            "warning_count": sum(len(audit.warnings) for audit in audits),
            "ready": True,
        }
        return replace(project, extra=extra)

    def add_map_studio_building_level(
        self,
        *,
        level_index: int,
        name: str = "",
        floor_z: float = 0.0,
        floor_to_floor_height: float = 3.0,
    ) -> int:
        """Persist an empty or occupied level as one undoable authoring action."""

        authored = self._map_studio_building_project()
        before = self._capture_map_studio_command_state()
        updated = set_pascal_building_level_in_project(
            authored,
            level_index=int(level_index),
            name=str(name or f"Level {int(level_index) + 1}"),
            floor_z=float(floor_z),
            floor_to_floor_height=float(floor_to_floor_height),
        )
        self._store_authored_project(updated)
        self.model.log(
            f"Added {name or f'Level {int(level_index) + 1}'} at {float(floor_z):.2f} m; "
            "the level is authoring state until rooms are drawn on it."
        )
        self._record_map_studio_command(
            action_key="map_studio.building.level.create",
            label=f"Add {name or f'Level {int(level_index) + 1}'}",
            before=before,
            stale_outputs=(),
            readiness_impact="presentation-only level definition",
            metadata={
                "level_index": int(level_index),
                "name": str(name or f"Level {int(level_index) + 1}"),
                "floor_z": float(floor_z),
                "floor_to_floor_height": float(floor_to_floor_height),
            },
        )
        return int(level_index)

    def add_map_studio_building_room(
        self,
        *,
        points: Any,
        floor_z: float = 0.0,
        wall_height: float = 3.0,
        level_index: int = 0,
        level_name: str = "",
        style_id: str = "plcaa_graybox",
        architecture_archetype: str = "",
        include_ceiling: bool = True,
        floor_surface_id: int | str = 4,
        building_kind: str = "interior",
        roof_type: str = "none",
        roof_pitch_degrees: float = 30.0,
        roof_overhang: float = 0.25,
    ) -> str:
        """Create one exportable room from a closed direct-wall loop."""

        authored = self._map_studio_building_project()
        styles = {style.style_id: style for style in available_pascal_building_styles(authored.game)}
        style = styles.get(str(style_id or "")) or next(iter(styles.values()))
        archetype_id = str(architecture_archetype or "").strip().lower()
        archetype = next(
            (
                candidate
                for candidate in style.architecture_archetypes
                if candidate.archetype_id == archetype_id
            ),
            None,
        )
        if style.architecture_archetypes and archetype is None:
            archetype = style.architecture_archetypes[0]
            archetype_id = archetype.archetype_id
        shell_profile = (
            archetype.shell_profile
            if archetype is not None
            else style.architecture_shell_profile
        )
        evidence_rooms = tuple(
            dict.fromkeys(
                tuple(style.evidence_rooms)
                + (tuple(archetype.evidence_rooms) if archetype is not None else ())
            )
        )
        before = self._capture_map_studio_command_state()
        updated, room_resref = add_pascal_building_room_to_project(
            authored,
            points=points,
            floor_z=float(floor_z),
            wall_height=float(wall_height),
            level_index=int(level_index),
            level_name=str(level_name or ""),
            floor_texture=style.floor_texture,
            wall_texture=style.wall_texture,
            ceiling_texture=style.ceiling_texture,
            include_ceiling=bool(include_ceiling),
            floor_surface_id=floor_surface_id,
            style_id=style.style_id,
            style_source_module=style.source_module,
            style_source_room=style.source_room,
            architecture_profile=style.architecture_profile,
            architecture_shell_profile=shell_profile,
            architecture_archetype=archetype_id,
            architecture_accent_textures=style.accent_textures,
            architecture_evidence_rooms=evidence_rooms,
            building_kind=str(building_kind or "interior"),
            roof_type=str(roof_type or "none"),
            roof_pitch_degrees=float(roof_pitch_degrees),
            roof_overhang=float(roof_overhang),
        )
        updated = self._with_automatic_room_walkmesh_status(
            updated,
            operation="add_pascal_building_room",
            room_resrefs=(room_resref,),
        )
        environment_update = apply_authored_environment_style_defaults(updated, style.style_id)
        updated = environment_update.project
        self._store_authored_project(updated)
        created_room = next(
            (room for room in updated.rooms if room.normalised_resref() == room_resref),
            None,
        )
        created_primitive = getattr(created_room, "primitive", None)
        created_metadata = dict(getattr(created_primitive, "metadata", {}) or {})
        actual_kind = str(created_metadata.get("building_kind") or building_kind or "interior")
        roof_label = str(created_metadata.get("building_roof_type") or roof_type or "none").strip().lower()
        actual_ceiling = bool(getattr(created_primitive, "include_ceiling", include_ceiling))
        self.model.log(
            f"Built {actual_kind} room {room_resref} from a closed wall loop on "
            f"{level_name or f'Level {int(level_index) + 1}'}"
            + (f" with a {roof_label} roof." if roof_label != "none" else ".")
            + (f" {environment_update.summary}" if environment_update.applied else "")
        )
        self._record_map_studio_command(
            action_key="map_studio.building.room.create",
            label=f"Build room {room_resref}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "point_count": len(tuple(points or ())),
                "level_index": int(level_index),
                "floor_z": float(floor_z),
                "wall_height": float(wall_height),
                "style_id": style.style_id,
                "architecture_profile": style.architecture_profile,
                "architecture_shell_profile": shell_profile,
                "architecture_archetype": archetype_id,
                "include_ceiling": actual_ceiling,
                "building_kind": actual_kind,
                "roof_type": roof_label,
                "roof_pitch_degrees": float(roof_pitch_degrees),
                "roof_overhang": float(roof_overhang),
                "walkmesh_auto_generated": True,
                "environment_style_defaults_applied": bool(environment_update.applied),
                "environment_style_summary": str(environment_update.summary),
            },
        )
        return room_resref

    def set_map_studio_building_opening(
        self,
        *,
        room_resref: str,
        edge_index: int,
        opening_kind: str,
        center_fraction: float,
        width: float,
        height: float,
        bottom: float = 0.0,
    ) -> None:
        """Cut one engine-safe door or window opening into a wall edge."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            raise ValueError("Draw and close a room before adding a door or window.")
        before = self._capture_map_studio_command_state()
        updated = set_pascal_building_opening_in_project(
            authored,
            room_resref=room_resref,
            edge_index=int(edge_index),
            opening_kind=opening_kind,
            center_fraction=float(center_fraction),
            width=float(width),
            height=float(height),
            bottom=float(bottom),
        )
        updated = self._with_automatic_room_walkmesh_status(
            updated,
            operation=f"set_pascal_building_{str(opening_kind or 'door').strip().lower()}",
            room_resrefs=(normalise_resref(room_resref),),
        )
        self._store_authored_project(updated)
        kind = str(opening_kind or "door").strip().lower()
        self.model.log(f"Added {kind} opening to {normalise_resref(room_resref)} wall {int(edge_index) + 1}.")
        self._record_map_studio_command(
            action_key=f"map_studio.building.{kind}.create",
            label=f"Add {kind} to {normalise_resref(room_resref)}",
            before=before,
            metadata={
                "room_resref": normalise_resref(room_resref),
                "edge_index": int(edge_index),
                "opening_kind": kind,
                "center_fraction": float(center_fraction),
                "width": float(width),
                "height": float(height),
                "bottom": float(bottom),
                "walkmesh_auto_generated": True,
            },
        )

    def preview_map_studio_building_opening(
        self,
        *,
        room_resref: str,
        edge_index: int,
        opening_kind: str,
        center_fraction: float,
        width: float,
        height: float,
        bottom: float = 0.0,
    ) -> dict[str, Any]:
        """Resolve a hosted door/window ghost without mutating KMAP state."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return {
                "valid": False,
                "reason": "Draw and close a room before adding a door or window.",
            }
        preview = preview_pascal_building_opening(
            authored,
            room_resref=room_resref,
            edge_index=int(edge_index),
            opening_kind=opening_kind,
            center_fraction=float(center_fraction),
            width=float(width),
            height=float(height),
            bottom=float(bottom),
        )
        return {
            "room_resref": preview.room_resref,
            "edge_index": preview.edge_index,
            "opening_kind": preview.opening_kind,
            "center_fraction": preview.center_fraction,
            "width": preview.width,
            "height": preview.height,
            "bottom": preview.bottom,
            "valid": preview.valid,
            "reason": preview.reason,
        }

    def refresh_map_studio_vanilla_terrain_kit_assets(
        self,
        *,
        resource_manager: Any,
        progress: Any = None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Rebuild the local retail-terrain metadata index for the target game."""

        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        assets = scan_vanilla_terrain_kit_assets(
            resource_manager,
            games=(game,),
            outdoor_only=True,
            progress=progress,
        )
        retained = tuple(asset for asset in vanilla_terrain_kit_assets() if asset.game != game)
        path = write_vanilla_terrain_kit_catalog(retained + assets)
        return terrain_kit_asset_rows(game=game), str(path)

    def map_studio_terrain_kit_preview_model(self, asset_id: str, *, resource_manager: Any = None):
        """Build one isolated Terrain Kit model for a lazy browser thumbnail."""

        return build_terrain_kit_preview_model(
            str(asset_id or ""),
            game=str(getattr(self.project, "game", "K1") or "K1").upper(),
            resource_manager=resource_manager,
        )

    def map_studio_environment_kit_preview_model(self, piece_id: str, *, resource_manager: Any = None):
        """Load one trained vanilla tile or clipped dressing thumbnail."""

        piece = environment_kit_piece(str(piece_id or ""))
        if piece is None or resource_manager is None:
            return None
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        if piece.game != game:
            return None
        model = load_stock_kotor_model(resource_manager, piece.model_resref or piece.room_resref, game)
        if model is None or piece.role != "dressing":
            return model
        whole_room = build_imported_mesh_primitive_from_stock_model(
            model,
            room_resref="grkitpreview",
            source_model=piece.model_resref or piece.room_resref,
            game=game,
        )
        primitive = extract_imported_mesh_room_bounds(
            whole_room,
            bounds=piece.source_bounds_m,
            room_resref="grkitpreview",
            surface_names=piece.source_surface_names,
            anchor_mode=piece.anchor_mode,
            backdrop_texture=piece.backdrop_texture_resref,
            backdrop_axis=piece.backdrop_axis,
        )
        from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
        from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

        room = AuthoredRoomSpec(
            room_resref="grkitpreview",
            primitive=primitive,
            visible_rooms=("grkitpreview",),
            metadata={"source": "map_studio:environment_kit_thumbnail", "visual_only": True},
        )
        project = AuthoredModuleProject(
            metadata=AuthoredModuleMetadata(module_root="grkitpreview", game=game),
            rooms=(room,),
            placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref="grkitpreview")),
        )
        return build_authored_module_preview_model(project, include_backdrops=True).model

    def preview_authored_terrain_kit_placement(
        self,
        *,
        asset_id: str,
        position: Any,
        rotation_degrees_z: float = 0.0,
        scale: float = 1.0,
        preferred_wall_room_resref: str = "",
        excluded_wall_targets: tuple[tuple[str, int], ...] = (),
        _source_piece_override: Any = None,
    ) -> dict[str, Any]:
        """Return the surface position or nearest compatible trained-kit snap.

        All positions are authored-module world space in metres. Piece magnet
        positions are stored in piece-local space and transformed here before
        the placement transaction mutates the KMAP project.
        """

        values = tuple(position or (0.0, 0.0, 0.0))
        point = tuple(float(value) for value in values[:3])
        uniform_scale = float(scale)
        yaw_degrees = float(rotation_degrees_z)
        if len(point) < 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("Terrain kit placement requires a finite 3D surface position.")
        if not math.isfinite(uniform_scale) or uniform_scale <= 0.0:
            raise ValueError("Terrain kit scale must be a finite value greater than zero.")
        if not math.isfinite(yaw_degrees):
            raise ValueError("Terrain kit rotation must be finite.")

        result: dict[str, Any] = {
            "position": point,
            "rotation_degrees_z": yaw_degrees,
            "scale": uniform_scale,
            "magnet_snapped": False,
            "source_magnet_id": "",
            "target_magnet_id": "",
            "target_piece_id": "",
            "target_room_resref": "",
            "cursor_distance": 0.0,
            "target_is_authored_wall": False,
            "target_edge_index": -1,
            "target_center_fraction": 0.5,
            "opening_width": 0.0,
            "opening_height": 0.0,
            "opening_bottom": 0.0,
        }
        source_piece = (
            _source_piece_override
            if _source_piece_override is not None
            else environment_kit_piece(str(asset_id or ""))
        )
        authored = self._map_studio_authored_project_snapshot()
        if source_piece is None or authored is None:
            return result
        preferred_wall_room = normalise_resref(preferred_wall_room_resref)
        excluded_walls = {
            (normalise_resref(str(room_resref or "")), int(edge_index))
            for room_resref, edge_index in tuple(excluded_wall_targets or ())
        }

        targets: list[tuple[Any, tuple[float, float, float], float, float, str]] = []
        for room in tuple(authored.rooms or ()):
            room_metadata = dict(getattr(room, "metadata", {}) or {})
            primitive_metadata = dict(getattr(getattr(room, "primitive", None), "metadata", {}) or {})
            target_asset_id = str(
                room_metadata.get("environment_kit_piece_id")
                or primitive_metadata.get("environment_kit_piece_id")
                or room_metadata.get("terrain_kit_asset_id")
                or primitive_metadata.get("terrain_kit_asset_id")
                or ""
            )
            target_piece = environment_kit_piece(target_asset_id)
            room_position = tuple(getattr(room, "position", ()) or ())
            if target_piece is None or len(room_position) < 3:
                continue
            try:
                target_point = tuple(float(value) for value in room_position[:3])
                target_rotation = float(
                    room_metadata.get(
                        "environment_kit_rotation_degrees_z",
                        primitive_metadata.get(
                            "environment_kit_rotation_degrees_z",
                            room_metadata.get(
                                "terrain_kit_rotation_degrees_z",
                                primitive_metadata.get("terrain_kit_rotation_degrees_z", 0.0),
                            ),
                        ),
                    )
                    or 0.0
                )
                target_scale = float(
                    room_metadata.get(
                        "environment_kit_scale",
                        primitive_metadata.get(
                            "environment_kit_scale",
                            room_metadata.get("terrain_kit_scale", primitive_metadata.get("terrain_kit_scale", 1.0)),
                        ),
                    )
                    or 1.0
                )
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in target_point) or target_scale <= 0.0:
                continue
            targets.append(
                (
                    target_piece,
                    target_point,
                    math.radians(target_rotation),
                    target_scale,
                    room.normalised_resref(),
                )
            )
        horizontal_span = max(0.01, *(float(value) for value in source_piece.dimensions_m[:2]))
        snap = (
            nearest_environment_kit_snap(
                source_piece,
                proposed_position=point,
                proposed_yaw=math.radians(yaw_degrees),
                source_scale=uniform_scale,
                targets=tuple(targets),
                max_distance=max(0.75, min(3.0, horizontal_span * 0.12)),
            )
            if targets
            else None
        )

        # Pascal rooms expose every footprint edge as a live doorway target.
        # The retail room's own LYT hook is projected onto the nearest edge,
        # then the opening preview contract validates/clamps the exact cut.
        from .authored_room_floorplan import FloorPlanRoomPrimitive, polygon_signed_area

        def point_inside_polygon(
            candidate: tuple[float, float],
            polygon: tuple[tuple[float, float], ...],
        ) -> bool:
            inside = False
            previous = polygon[-1] if polygon else (0.0, 0.0)
            for current in polygon:
                if (current[1] > candidate[1]) != (previous[1] > candidate[1]):
                    denominator = previous[1] - current[1]
                    if abs(denominator) > 1.0e-12:
                        boundary_x = (
                            ((previous[0] - current[0]) * (candidate[1] - current[1]))
                            / denominator
                        ) + current[0]
                        if candidate[0] < boundary_x:
                            inside = not inside
                previous = current
            return inside

        wall_snap = None
        for room in tuple(authored.rooms or ()):
            primitive = getattr(room, "primitive", None)
            primitive_metadata = dict(getattr(primitive, "metadata", {}) or {})
            if not isinstance(primitive, FloorPlanRoomPrimitive):
                continue
            if str(primitive_metadata.get("source") or "").strip().lower() != "map_studio:pascal_building":
                continue
            room_key = room.normalised_resref()
            if preferred_wall_room and room_key != preferred_wall_room:
                continue
            profile = str(primitive_metadata.get("architecture_profile") or "").strip().lower()
            door_spec = pascal_architecture_door_spec(str(authored.game or "K1"), profile)
            style_id = str(primitive_metadata.get("style_id") or "").strip().lower()
            style = next(
                (
                    candidate
                    for candidate in available_pascal_building_styles(str(authored.game or "K1"))
                    if candidate.style_id.strip().lower() == style_id
                ),
                None,
            )
            opening_width = (
                float(door_spec["opening_width_m"])
                if door_spec
                else float(style.recommended_door_width_m) if style is not None else 1.8
            )
            opening_height = (
                float(door_spec["opening_height_m"])
                if door_spec
                else min(
                    float(style.recommended_door_height_m) if style is not None else 2.4,
                    float(primitive.wall_height) - 0.08,
                )
            )
            origin = tuple(float(value) for value in tuple(room.position or (0.0, 0.0, 0.0))[:3])
            points = tuple((float(value[0]), float(value[1])) for value in tuple(primitive.points or ()))
            world_points = tuple((origin[0] + value[0], origin[1] + value[1]) for value in points)
            dropped_inside_room = point_inside_polygon(
                (float(point[0]), float(point[1])),
                world_points,
            )
            ccw = polygon_signed_area(points) > 0.0
            for edge_index, local_start in enumerate(points):
                if (room_key, edge_index) in excluded_walls:
                    continue
                local_end = points[(edge_index + 1) % len(points)]
                dx = local_end[0] - local_start[0]
                dy = local_end[1] - local_start[1]
                edge_length = math.hypot(dx, dy)
                if edge_length <= opening_width + 0.04:
                    continue
                outward = (
                    (dy / edge_length, -dx / edge_length)
                    if ccw
                    else (-dy / edge_length, dx / edge_length)
                )
                wall = EnvironmentKitWallTarget(
                    room_resref=room.normalised_resref(),
                    edge_index=edge_index,
                    start=(origin[0] + local_start[0], origin[1] + local_start[1]),
                    end=(origin[0] + local_end[0], origin[1] + local_end[1]),
                    floor_z=origin[2] + float(primitive.z),
                    outward_yaw_radians=math.atan2(outward[1], outward[0]),
                    opening_width=opening_width,
                    opening_height=opening_height,
                )
                candidate = nearest_environment_kit_wall_snap(
                    source_piece,
                    proposed_position=point,
                    proposed_yaw=math.radians(yaw_degrees),
                    source_scale=uniform_scale,
                    walls=(wall,),
                    # Dropping a complete vanilla room anywhere on an authored
                    # room means "attach this room", not "occupy this floor".
                    # Search every wall in that room, while ordinary free-space
                    # drags retain the short magnetic capture radius.
                    max_distance=(
                        max(edge_length * 2.0, 8.0)
                        if dropped_inside_room or preferred_wall_room
                        else max(1.5, min(4.0, edge_length * 0.35))
                    ),
                )
                if candidate is None:
                    continue
                opening_preview = preview_pascal_building_opening(
                    authored,
                    room_resref=room.normalised_resref(),
                    edge_index=edge_index,
                    opening_kind="door",
                    center_fraction=candidate.target_center_fraction,
                    width=opening_width,
                    height=opening_height,
                    bottom=0.0,
                )
                if not opening_preview.valid:
                    continue
                delta_fraction = float(opening_preview.center_fraction) - float(candidate.target_center_fraction)
                candidate = replace(
                    candidate,
                    position=(
                        float(candidate.position[0]) + dx * delta_fraction,
                        float(candidate.position[1]) + dy * delta_fraction,
                        float(candidate.position[2]),
                    ),
                    target_center_fraction=float(opening_preview.center_fraction),
                    opening_width=float(opening_preview.width),
                    opening_height=float(opening_preview.height),
                    opening_bottom=float(opening_preview.bottom),
                )
                if wall_snap is None or candidate.cursor_distance < wall_snap.cursor_distance:
                    wall_snap = candidate
        if wall_snap is not None and (snap is None or wall_snap.cursor_distance < snap.cursor_distance):
            snap = wall_snap
        if snap is None:
            return result
        result.update(
            {
                "position": tuple(float(value) for value in snap.position),
                "rotation_degrees_z": math.degrees(float(snap.yaw_radians)) % 360.0,
                "magnet_snapped": True,
                "source_magnet_id": snap.source_magnet_id,
                "target_magnet_id": snap.target_magnet_id,
                "target_piece_id": snap.target_piece_id,
                "target_room_resref": snap.target_room_resref,
                "cursor_distance": float(snap.cursor_distance),
                "target_is_authored_wall": bool(snap.target_is_authored_wall),
                "target_edge_index": int(snap.target_edge_index),
                "target_center_fraction": float(snap.target_center_fraction),
                "opening_width": float(snap.opening_width),
                "opening_height": float(snap.opening_height),
                "opening_bottom": float(snap.opening_bottom),
            }
        )
        return result

    def add_authored_terrain_kit_asset(
        self,
        *,
        asset_id: str,
        position: Any,
        rotation_degrees_z: float = 0.0,
        scale: float = 1.0,
        target_room_resref: str = "",
        resource_manager: Any = None,
    ) -> str:
        """Surface-place a terrain mesh as one small Odyssey room.

        KOTOR's GIT placeable list cannot own arbitrary static cliff geometry.
        Explicit nonblocking dressing keeps a zero-face WOK. Traversal pieces
        may carry generated floor WOKs, while unresolved solid blockers remain
        export-blocking until their footprint is cut from the host walkmesh.
        """

        entry = terrain_kit_asset(asset_id)
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
            from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

            root = normalise_resref(str(getattr(self.project, "name", "") or "")) or "grterrain"
            authored = AuthoredModuleProject(
                metadata=AuthoredModuleMetadata(
                    module_root=root,
                    game=str(getattr(self.project, "game", "K1") or "K1").upper(),
                    display_name=f"{root} terrain map",
                    tag=root,
                ),
                rooms=(),
                placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref=root)),
            )

        placement = self.preview_authored_terrain_kit_placement(
            asset_id=entry.asset_id,
            position=position,
            rotation_degrees_z=rotation_degrees_z,
            scale=scale,
        )
        point = tuple(float(value) for value in tuple(placement["position"])[:3])
        rotation_degrees_z = float(placement["rotation_degrees_z"])
        scale = float(placement["scale"])

        used = {room.normalised_resref() for room in tuple(authored.rooms or ())}
        room_key = ""
        for ordinal in range(1, 10_000):
            candidate = f"grtk{ordinal:04d}"
            if candidate not in used:
                room_key = candidate
                break
        if not room_key:
            raise ValueError("This map has exhausted its terrain-kit room identifiers.")

        primitive = build_terrain_kit_primitive(
            entry.asset_id,
            room_key,
            str(authored.game or "K1").upper(),
            float(rotation_degrees_z),
            float(scale),
            resource_manager=resource_manager,
        )
        primitive_metadata = dict(getattr(primitive, "metadata", {}) or {})
        terrain_visual_only = bool(primitive_metadata.get("terrain_kit_visual_only", False))
        collision_status = str(
            primitive_metadata.get("terrain_kit_collision_status") or ""
        )
        collision_ready = bool(
            primitive_metadata.get("terrain_kit_collision_ready", terrain_visual_only)
        )
        collision_blocking_reason = str(
            primitive_metadata.get("terrain_kit_collision_blocking_reason") or ""
        )
        all_rooms = tuple(room.normalised_resref() for room in authored.rooms) + (room_key,)
        updated_existing = tuple(
            replace(
                room,
                visible_rooms=tuple(dict.fromkeys(tuple(room.visible_rooms or ()) + (room_key,))),
            )
            for room in authored.rooms
        )
        room = AuthoredRoomSpec(
            room_resref=room_key,
            primitive=primitive,
            position=(point[0], point[1], point[2]),
            visible_rooms=all_rooms,
            metadata={
                "primitive": "imported_mesh",
                "source": "map_studio:terrain_kit",
                "terrain_kit_asset_id": entry.asset_id,
                "terrain_kit_label": entry.label,
                "terrain_kit_source_kind": (
                    "vanilla_game" if hasattr(entry, "module_resref") else "supplied_obj"
                ),
                "terrain_kit_source_game": str(getattr(entry, "game", "") or ""),
                "terrain_kit_source_module": str(getattr(entry, "module_resref", "") or ""),
                "terrain_kit_source_room": str(getattr(entry, "room_resref", "") or ""),
                "terrain_kit_category": entry.category,
                "terrain_kit_rotation_degrees_z": float(rotation_degrees_z),
                "terrain_kit_scale": float(scale),
                "terrain_kit_target_room_resref": normalise_resref(
                    str(placement.get("target_room_resref") or target_room_resref)
                ),
                "terrain_kit_surface_target_room_resref": normalise_resref(target_room_resref),
                "terrain_kit_magnet_snapped": bool(placement.get("magnet_snapped", False)),
                "terrain_kit_source_magnet_id": str(placement.get("source_magnet_id") or ""),
                "terrain_kit_target_magnet_id": str(placement.get("target_magnet_id") or ""),
                "terrain_kit_target_piece_id": str(placement.get("target_piece_id") or ""),
                "terrain_kit_snap_cursor_distance": float(placement.get("cursor_distance", 0.0) or 0.0),
                "terrain_kit_collision_status": collision_status,
                "terrain_kit_collision_ready": collision_ready,
                "terrain_kit_collision_blocking_reason": collision_blocking_reason,
                "visual_only": terrain_visual_only,
                "surface_snapped": True,
            },
        )
        before = self._capture_map_studio_command_state()
        self._store_authored_project(replace(authored, rooms=updated_existing + (room,)))
        snap_note = (
            f"; magnet-snapped to {placement.get('target_room_resref')}"
            if placement.get("magnet_snapped", False)
            else ""
        )
        self.model.log(
            f"Placed Terrain Kit asset {entry.label} as room {room_key} at "
            f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}{snap_note}."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain_kit.place",
            label=f"Place terrain piece {entry.label}",
            before=before,
            metadata={
                "asset_id": entry.asset_id,
                "room_resref": room_key,
                "position": list(point),
                "rotation_degrees_z": float(rotation_degrees_z),
                "scale": float(scale),
                "visual_only": True,
                "magnet_snapped": bool(placement.get("magnet_snapped", False)),
                "source_magnet_id": str(placement.get("source_magnet_id") or ""),
                "target_magnet_id": str(placement.get("target_magnet_id") or ""),
                "target_piece_id": str(placement.get("target_piece_id") or ""),
                "target_room_resref": str(placement.get("target_room_resref") or ""),
            },
        )
        return room_key

    @staticmethod
    def _environment_kit_wall_portal_inset(
        authored: Any,
        *,
        target_room_resref: str,
        target_edge_index: int,
        source_portal: dict[str, Any] | None,
        source_hook: dict[str, Any] | None,
    ) -> float:
        """Project a stock threshold edge onto the authored room's inward normal."""

        if not source_portal or not source_hook:
            return 0.0
        portal_midpoint = tuple(float(value) for value in tuple(source_portal.get("midpoint") or ())[:3])
        hook_position = tuple(float(value) for value in tuple(source_hook.get("local_position") or ())[:3])
        if len(portal_midpoint) != 3 or len(hook_position) != 3:
            return 0.0
        target = next(
            (room for room in tuple(authored.rooms or ()) if room.normalised_resref() == normalise_resref(target_room_resref)),
            None,
        )
        primitive = getattr(target, "primitive", None)
        points = tuple(getattr(primitive, "points", ()) or ())
        edge = int(target_edge_index)
        if edge < 0 or edge >= len(points):
            return 0.0
        from .authored_room_floorplan import polygon_signed_area

        start = points[edge]
        end = points[(edge + 1) % len(points)]
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        length = math.hypot(dx, dy)
        if length <= 1.0e-8:
            return 0.0
        ccw = polygon_signed_area(tuple((float(point[0]), float(point[1])) for point in points)) > 0.0
        inward = (-dy / length, dx / length) if ccw else (dy / length, -dx / length)
        offset = (portal_midpoint[0] - hook_position[0], portal_midpoint[1] - hook_position[1])
        projected = offset[0] * inward[0] + offset[1] * inward[1]
        if projected < -0.01:
            raise ValueError("The selected stock doorway threshold points away from the authored room wall.")
        return min(1.5, max(0.0, projected))

    def add_authored_environment_kit_piece(
        self,
        *,
        piece_id: str,
        position: Any,
        rotation_degrees_z: float = 0.0,
        scale: float = 1.0,
        target_room_resref: str = "",
        resource_manager: Any = None,
        _excluded_wall_targets: tuple[tuple[str, int], ...] = (),
    ) -> str:
        """Place a reusable vanilla room tile as uniquely named authored geometry.

        Retail model/WOK bytes are resolved from the user's game installation,
        transformed together, and compiled under a fresh room resref. The KMAP
        keeps only source provenance, transform, and typed magnet metadata.
        """

        piece = environment_kit_piece(str(piece_id or ""))
        if piece is None or piece.role not in {"room_tile", "exterior_tile", "dressing"}:
            raise ValueError("Choose a trained vanilla room, exterior tile, or dressing piece.")
        if piece.placement_quality not in {"verified", "generated_connector"}:
            raise ValueError(
                piece.placement_quality_message
                or f"{piece.label} is awaiting geometry review and cannot be placed."
            )
        if resource_manager is None:
            raise ValueError(f"Connect the {piece.game} game installation before placing {piece.label}.")
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        if piece.game != game:
            raise ValueError(f"{piece.label} is a {piece.game} kit piece and cannot be placed in a {game} map.")

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
            from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject

            root = normalise_resref(str(getattr(self.project, "name", "") or "")) or "grmodule"
            authored = AuthoredModuleProject(
                metadata=AuthoredModuleMetadata(
                    module_root=root,
                    game=game,
                    display_name=f"{root} environment kit map",
                    tag=root,
                ),
                rooms=(),
                placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref=root)),
            )

        requested_position = tuple(float(value) for value in tuple(position or (0.0, 0.0, 0.0))[:3])
        wall_aligned = False
        if piece.anchor_mode == "wall" and target_room_resref and len(requested_position) >= 3:
            from .authored_room_floorplan import FloorPlanRoomPrimitive, polygon_signed_area

            target_key = normalise_resref(target_room_resref)
            target_room = next(
                (room for room in authored.rooms if room.normalised_resref() == target_key),
                None,
            )
            target_primitive = getattr(target_room, "primitive", None)
            if isinstance(target_primitive, FloorPlanRoomPrimitive):
                room_position = tuple(getattr(target_room, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
                points = tuple(
                    (
                        float(point[0]) + float(room_position[0]),
                        float(point[1]) + float(room_position[1]),
                    )
                    for point in target_primitive.points
                )
                best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
                for index, start in enumerate(points):
                    end = points[(index + 1) % len(points)]
                    dx, dy = end[0] - start[0], end[1] - start[1]
                    length_squared = (dx * dx) + (dy * dy)
                    if length_squared <= 1.0e-10:
                        continue
                    t = max(
                        0.0,
                        min(
                            1.0,
                            (((requested_position[0] - start[0]) * dx) + ((requested_position[1] - start[1]) * dy))
                            / length_squared,
                        ),
                    )
                    closest = (start[0] + (dx * t), start[1] + (dy * t))
                    distance = math.hypot(requested_position[0] - closest[0], requested_position[1] - closest[1])
                    if best is None or distance < best[0]:
                        best = (distance, closest, (dx, dy))
                if best is not None:
                    _distance, closest, direction = best
                    edge_length = math.hypot(direction[0], direction[1])
                    ccw = polygon_signed_area(tuple(target_primitive.points)) > 0.0
                    inward = (
                        (-direction[1] / edge_length, direction[0] / edge_length)
                        if ccw
                        else (direction[1] / edge_length, -direction[0] / edge_length)
                    )
                    local_axis = {"x": 0.0, "y": math.pi * 0.5, "-x": math.pi, "-y": -math.pi * 0.5}.get(
                        piece.local_normal_axis,
                        math.pi * 0.5,
                    )
                    rotation_degrees_z = math.degrees(math.atan2(inward[1], inward[0]) - local_axis)
                    requested_position = (
                        closest[0] + (inward[0] * 0.015),
                        closest[1] + (inward[1] * 0.015),
                        requested_position[2],
                    )
                    wall_aligned = True

        placement = self.preview_authored_terrain_kit_placement(
            asset_id=piece.piece_id,
            position=requested_position,
            rotation_degrees_z=rotation_degrees_z,
            scale=scale,
            preferred_wall_room_resref=target_room_resref,
            excluded_wall_targets=_excluded_wall_targets,
        )
        point = tuple(float(value) for value in tuple(placement["position"])[:3])
        rotation_degrees_z = float(placement["rotation_degrees_z"])
        scale = float(placement["scale"])

        used = {room.normalised_resref() for room in tuple(authored.rooms or ())}
        room_key = next((f"grkit{ordinal:04d}" for ordinal in range(1, 10_000) if f"grkit{ordinal:04d}" not in used), "")
        if not room_key:
            raise ValueError("This map has exhausted its environment-kit room identifiers.")

        model = load_stock_kotor_model(resource_manager, piece.model_resref or piece.room_resref, game)
        if model is None:
            raise ValueError(
                f"Vanilla room model {piece.model_resref or piece.room_resref} was not found in the configured {game} installation."
            )
        wok_bytes = (
            None
            if piece.role == "dressing"
            else self._stock_room_wok_bytes(piece.room_resref, resource_manager, authored)
        )
        whole_room = build_imported_mesh_primitive_from_stock_model(
            model,
            room_resref=room_key,
            source_model=piece.model_resref or piece.room_resref,
            game=game,
            wok_bytes=wok_bytes,
        )
        if piece.placement_quality == "generated_connector":
            source_wok = getattr(whole_room, "wok", None)
            if source_wok is None or not tuple(getattr(source_wok, "faces", ()) or ()):
                components = analyze_obj_walkmesh_candidate_components(whole_room)
                substantial = tuple(component for component in components if float(component.area) >= 2.0)
                if not substantial:
                    raise ValueError(
                        f"{piece.label} has no substantial reviewed floor geometry from which to generate collision."
                    )
                floor_z = min(float(component.bounds_min[2]) for component in substantial)
                selected = tuple(
                    component
                    for component in substantial
                    if float(component.bounds_max[2]) <= floor_z + 0.35
                    and float(component.bounds_max[2]) - float(component.bounds_min[2]) <= 0.50
                )
                if not selected:
                    raise ValueError(
                        f"{piece.label} has no flat floor component at its lowest reviewed elevation."
                    )
                whole_room = apply_obj_walkmesh_component_review(
                    whole_room,
                    components,
                    selected_component_ids=tuple(component.component_id for component in selected),
                    reason=(
                        "Reviewed stock-room recovery: substantial up-facing components at the "
                        "lowest shared floor elevation; preserve separate walkable islands."
                    ),
                )
                whole_room, generation = generate_room_walkmesh_from_geometry(
                    whole_room,
                    disconnected_island_policy="preserve",
                    source_wok_policy="replace",
                )
                if str(generation.get("structural_validation") or "").strip().lower() != "passed":
                    raise ValueError(
                        f"Ghost Studio could not validate generated floor collision for {piece.label}."
                    )
            whole_room = rebase_environment_kit_room_to_walkmesh(whole_room)
            generated_magnets = generated_environment_kit_boundary_magnets(
                whole_room,
                opening_width=4.0,
            )
            if not generated_magnets:
                raise ValueError(
                    f"{piece.label} has no exposed WOK boundary wide enough for a safe generated connector. "
                    "Existing closed stock walls were preserved."
                )
            piece = replace(
                piece,
                magnets=generated_magnets,
                placement_quality="generated_connector",
            )
            from .authored_module_walkmesh import prepare_environment_kit_room_walkmesh

            whole_room = prepare_environment_kit_room_walkmesh(
                whole_room,
                source_room_position=(0.0, 0.0, 0.0),
                magnets=piece.magnets,
            )
            placement = self.preview_authored_terrain_kit_placement(
                asset_id=piece.piece_id,
                position=requested_position,
                rotation_degrees_z=rotation_degrees_z,
                scale=scale,
                preferred_wall_room_resref=target_room_resref,
                excluded_wall_targets=_excluded_wall_targets,
                _source_piece_override=piece,
            )
            point = tuple(float(value) for value in tuple(placement["position"])[:3])
            rotation_degrees_z = float(placement["rotation_degrees_z"])
            scale = float(placement["scale"])
        elif piece.role != "dressing" and whole_room.wok is not None:
            from .authored_module_walkmesh import prepare_environment_kit_room_walkmesh

            source_origin = environment_kit_source_room_position(piece, resource_manager)
            whole_room = prepare_environment_kit_room_walkmesh(
                whole_room,
                source_room_position=source_origin,
                magnets=piece.magnets,
            )
        primitive = (
            extract_imported_mesh_room_bounds(
                whole_room,
                bounds=piece.source_bounds_m,
                room_resref=room_key,
                surface_names=piece.source_surface_names,
                anchor_mode=piece.anchor_mode,
                backdrop_texture=piece.backdrop_texture_resref,
                backdrop_axis=piece.backdrop_axis,
            )
            if piece.role == "dressing"
            else whole_room
        )
        if not primitive.surfaces:
            raise ValueError(f"{piece.label} has no renderable room geometry.")
        had_lightmaps = any(bool(surface.lightmap) for surface in primitive.surfaces)
        primitive = replace(
            primitive,
            room_resref=room_key,
            surfaces=tuple(replace(surface, lightmap="", uvs_lm=()) for surface in primitive.surfaces),
            wok=(replace(primitive.wok, name=room_key, raw=None) if primitive.wok is not None else None),
            metadata={
                **dict(primitive.metadata or {}),
                "source": "map_studio:environment_kit",
                "environment_kit_piece_id": piece.piece_id,
                "environment_kit_collection_id": piece.collection_id,
                "environment_kit_source_game": piece.game,
                "environment_kit_source_module": piece.module_resref,
                "environment_kit_source_room": piece.room_resref,
                "environment_kit_class_id": piece.class_id,
                "environment_kit_role": piece.role,
                "environment_kit_anchor_mode": piece.anchor_mode,
                "environment_kit_wall_aligned": wall_aligned,
                "environment_kit_rotation_degrees_z": rotation_degrees_z,
                "environment_kit_scale": scale,
                "source_lightmaps_removed_for_relighting": had_lightmaps,
            },
        )
        source_magnet_id = str(placement.get("source_magnet_id") or "")
        physical_portal_aligned = False
        connection_outward_normal: tuple[float, float] | None = None
        untransformed_source_portal = next(
            (
                dict(row or {})
                for row in tuple(dict(primitive.metadata or {}).get("walkmesh_portals") or ())
                if str(dict(row or {}).get("magnet_id") or "").strip().lower() == source_magnet_id.strip().lower()
            ),
            None,
        )
        if bool(placement.get("target_is_authored_wall", False)) and untransformed_source_portal is not None:
            target_key = normalise_resref(str(placement.get("target_room_resref") or ""))
            target_edge = int(placement.get("target_edge_index", -1))
            target_room = next(
                (candidate for candidate in authored.rooms if candidate.normalised_resref() == target_key),
                None,
            )
            target_points = tuple(getattr(getattr(target_room, "primitive", None), "points", ()) or ())
            portal_start = tuple(float(value) for value in tuple(untransformed_source_portal.get("start") or ())[:3])
            portal_end = tuple(float(value) for value in tuple(untransformed_source_portal.get("end") or ())[:3])
            if target_room is not None and 0 <= target_edge < len(target_points) and len(portal_start) == len(portal_end) == 3:
                target_start = target_points[target_edge]
                target_end = target_points[(target_edge + 1) % len(target_points)]
                target_angle = math.atan2(
                    float(target_end[1]) - float(target_start[1]),
                    float(target_end[0]) - float(target_start[0]),
                )
                portal_angle = math.atan2(portal_end[1] - portal_start[1], portal_end[0] - portal_start[0])
                candidates = (target_angle - portal_angle, target_angle + math.pi - portal_angle)
                preview_angle = math.radians(rotation_degrees_z)

                def angular_distance(value: float) -> float:
                    return abs((value - preview_angle + math.pi) % (2.0 * math.pi) - math.pi)

                target_length = math.hypot(
                    float(target_end[0]) - float(target_start[0]),
                    float(target_end[1]) - float(target_start[1]),
                )
                target_outward: tuple[float, float] | None = None
                if target_length > 1.0e-8:
                    target_area_twice = sum(
                        (float(point[0]) * float(target_points[(index + 1) % len(target_points)][1]))
                        - (float(point[1]) * float(target_points[(index + 1) % len(target_points)][0]))
                        for index, point in enumerate(target_points)
                    )
                    target_ccw = target_area_twice > 0.0
                    target_inward = (
                        (-(float(target_end[1]) - float(target_start[1])) / target_length,
                         (float(target_end[0]) - float(target_start[0])) / target_length)
                        if target_ccw
                        else ((float(target_end[1]) - float(target_start[1])) / target_length,
                              -(float(target_end[0]) - float(target_start[0])) / target_length)
                    )
                    target_outward = (-target_inward[0], -target_inward[1])

                # Aligning the threshold edge leaves two valid rotations 180
                # degrees apart.  Cursor-facing alone can choose the one whose
                # room body points back through the authored wall (506OND is a
                # concrete example), after which overlap trimming removes the
                # entire imported room.  Resolve the ambiguity from geometry:
                # the stock room's render-bounds centre must land on the
                # authored wall's outward side.
                source_vertices = tuple(
                    vertex
                    for surface in tuple(getattr(primitive, "surfaces", ()) or ())
                    if bool(getattr(surface, "render", True))
                    and not bool(getattr(surface, "backdrop", False))
                    and not bool(getattr(surface, "background_geometry", False))
                    for vertex in tuple(getattr(surface, "vertices", ()) or ())
                )
                portal_midpoint = (
                    (portal_start[0] + portal_end[0]) * 0.5,
                    (portal_start[1] + portal_end[1]) * 0.5,
                )
                body_delta: tuple[float, float] | None = None
                source_wok = getattr(primitive, "wok", None)
                source_faces = tuple(getattr(source_wok, "faces", ()) or ())
                source_wok_vertices = tuple(getattr(source_wok, "verts", ()) or ())
                try:
                    source_face_index = int(
                        untransformed_source_portal.get("source_face_index", -1)
                    )
                except (TypeError, ValueError):
                    source_face_index = -1
                if 0 <= source_face_index < len(source_faces):
                    source_face = source_faces[source_face_index]
                    indices = (
                        int(getattr(source_face, "v1", -1)),
                        int(getattr(source_face, "v2", -1)),
                        int(getattr(source_face, "v3", -1)),
                    )
                    if all(0 <= index < len(source_wok_vertices) for index in indices):
                        adjacent = tuple(source_wok_vertices[index] for index in indices)
                        body_delta = (
                            (sum(float(vertex[0]) for vertex in adjacent) / 3.0)
                            - portal_midpoint[0],
                            (sum(float(vertex[1]) for vertex in adjacent) / 3.0)
                            - portal_midpoint[1],
                        )
                if body_delta is None and source_vertices:
                    xs = tuple(float(vertex[0]) for vertex in source_vertices)
                    ys = tuple(float(vertex[1]) for vertex in source_vertices)
                    body_delta = (
                        ((min(xs) + max(xs)) * 0.5) - portal_midpoint[0],
                        ((min(ys) + max(ys)) * 0.5) - portal_midpoint[1],
                    )

                if body_delta is not None and target_outward is not None:
                    def outward_score(value: float) -> float:
                        cosine = math.cos(value)
                        sine = math.sin(value)
                        rotated = (
                            (body_delta[0] * cosine) - (body_delta[1] * sine),
                            (body_delta[0] * sine) + (body_delta[1] * cosine),
                        )
                        return (
                            (rotated[0] * target_outward[0])
                            + (rotated[1] * target_outward[1])
                        )

                    rotation_radians = max(
                        candidates,
                        key=lambda value: (outward_score(value), -angular_distance(value)),
                    )
                else:
                    rotation_radians = min(candidates, key=angular_distance)
                rotation_degrees_z = math.degrees(rotation_radians) % 360.0
                physical_portal_aligned = True
                connection_outward_normal = target_outward
        primitive = transform_terrain_kit_primitive(
            primitive,
            rotation_degrees_z=rotation_degrees_z,
            scale=scale,
        )
        runtime_graph = dict(dict(primitive.metadata or {}).get("source_runtime_graph") or {})
        if any(
            int(runtime_graph.get(key, 0) or 0) > 0
            for key in ("animation_count", "light_count", "emitter_count", "reference_count")
        ):
            primitive = prepare_imported_mesh_for_static_runtime_rebuild(
                primitive,
                reason=(
                    "Environment Kit room geometry is transformed into a reusable room-local tile; "
                    "source module model lights/runtime nodes are deliberately replaced by authored module lighting."
                ),
            )

        angle = math.radians(rotation_degrees_z)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        connection_points: list[dict[str, Any]] = []
        for magnet in piece.magnets:
            x = float(magnet.local_position[0]) * scale
            y = float(magnet.local_position[1]) * scale
            z = float(magnet.local_position[2]) * scale
            yaw = magnet.snap_facing_radians + angle
            connection_points.append(
                {
                    "door": magnet.magnet_id,
                    "kind": magnet.kind,
                    "magnet_class": magnet.magnet_class,
                    "local_position": [(x * cosine) - (y * sine), (x * sine) + (y * cosine), z],
                    "orientation": [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)],
                    "source": magnet.source,
                }
            )

        source_portal = next(
            (
                dict(row or {})
                for row in tuple(dict(primitive.metadata or {}).get("walkmesh_portals") or ())
                if str(dict(row or {}).get("magnet_id") or "").strip().lower() == source_magnet_id.strip().lower()
            ),
            None,
        )
        if physical_portal_aligned and source_portal is not None and connection_outward_normal is not None:
            primitive = trim_environment_kit_connection_overlap(
                primitive,
                portal_midpoint=tuple(float(value) for value in tuple(source_portal.get("midpoint") or ())[:3]),
                outward_normal=connection_outward_normal,
            )
            target_room_for_closure = next(
                (
                    candidate
                    for candidate in authored.rooms
                    if candidate.normalised_resref()
                    == normalise_resref(str(placement.get("target_room_resref") or ""))
                ),
                None,
            )
            target_style_for_closure = str(
                dict(getattr(getattr(target_room_for_closure, "primitive", None), "metadata", {}) or {}).get(
                    "style_id"
                )
                or ""
            ).strip().lower()
            if (
                piece.role in {"room_tile", "exterior_tile"}
                and target_style_for_closure == "architecture:k1_shadowlands"
            ):
                # A single Shadowlands LYT partition is deliberately only one
                # slice of an outdoor map.  Give its remaining WOK perimeter a
                # continuous dirt-bank continuation so it cannot expose void
                # gaps while the selected doorway stays open to the clearing.
                primitive = seal_environment_kit_exterior_bounds(
                    primitive,
                    portal_midpoint=tuple(float(value) for value in tuple(source_portal.get("midpoint") or ())[:3]),
                    portal_start=tuple(float(value) for value in tuple(source_portal.get("start") or ())[:3]),
                    portal_end=tuple(float(value) for value in tuple(source_portal.get("end") or ())[:3]),
                    portal_width=float(source_portal.get("width_m", 0.0) or 0.0),
                )
            source_portal = next(
                (
                    dict(row or {})
                    for row in tuple(dict(primitive.metadata or {}).get("walkmesh_portals") or ())
                    if str(dict(row or {}).get("magnet_id") or "").strip().lower() == source_magnet_id.strip().lower()
                ),
                None,
            )
        source_hook_row = next(
            (row for row in connection_points if str(row.get("door") or "").strip().lower() == source_magnet_id.strip().lower()),
            None,
        )
        if source_portal is not None and source_hook_row is not None:
            portal_midpoint = tuple(float(value) for value in tuple(source_portal.get("midpoint") or ())[:3])
            hook_position = tuple(float(value) for value in tuple(source_hook_row.get("local_position") or ())[:3])
            if len(portal_midpoint) == 3 and len(hook_position) == 3:
                if physical_portal_aligned:
                    target_key = normalise_resref(str(placement.get("target_room_resref") or ""))
                    target_edge = int(placement.get("target_edge_index", -1))
                    target_room = next(
                        (candidate for candidate in authored.rooms if candidate.normalised_resref() == target_key),
                        None,
                    )
                    target_primitive = getattr(target_room, "primitive", None)
                    target_points = tuple(getattr(target_primitive, "points", ()) or ())
                    if target_room is not None and 0 <= target_edge < len(target_points):
                        start = target_points[target_edge]
                        end = target_points[(target_edge + 1) % len(target_points)]
                        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
                        edge_length = math.hypot(dx, dy)
                        if edge_length > 1.0e-8:
                            portal_width = max(
                                0.05,
                                float(source_portal.get("width_m", placement.get("opening_width", 1.8)) or 1.8),
                            )
                            aligned_opening = preview_pascal_building_opening(
                                authored,
                                room_resref=target_key,
                                edge_index=target_edge,
                                opening_kind="door",
                                center_fraction=float(placement.get("target_center_fraction", 0.5) or 0.5),
                                width=portal_width,
                                height=float(placement.get("opening_height", 2.4) or 2.4),
                                bottom=float(placement.get("opening_bottom", 0.0) or 0.0),
                            )
                            if not aligned_opening.valid:
                                raise ValueError(aligned_opening.reason or "The stock room threshold does not fit this wall edge.")
                            target_origin = tuple(
                                float(value) for value in tuple(target_room.position or (0.0, 0.0, 0.0))[:3]
                            )
                            center_fraction = float(aligned_opening.center_fraction)
                            point = (
                                target_origin[0] + float(start[0]) + dx * center_fraction - portal_midpoint[0],
                                target_origin[1] + float(start[1]) + dy * center_fraction - portal_midpoint[1],
                                target_origin[2] + float(getattr(target_primitive, "z", 0.0)) - portal_midpoint[2],
                            )
                            placement = {
                                **placement,
                                "target_center_fraction": float(aligned_opening.center_fraction),
                                "opening_width": float(aligned_opening.width),
                                "opening_height": float(aligned_opening.height),
                                "opening_bottom": float(aligned_opening.bottom),
                            }
                else:
                    # Keep the retail door-hook height but place the actual WOK
                    # threshold exactly on the adjoining generated floor.
                    point = (point[0], point[1], point[2] - (portal_midpoint[2] - hook_position[2]))

        if piece.role in {"room_tile", "exterior_tile"}:
            primitive = trim_environment_kit_room_volume_overlap(
                primitive,
                position=point,
                rooms=tuple(authored.rooms or ()),
            )
            target_key = normalise_resref(
                str(placement.get("target_room_resref") or "")
            )
            occupancy = audit_environment_kit_room_occupancy(
                primitive,
                position=point,
                rooms=tuple(authored.rooms or ()),
                ignored_room_resrefs=((target_key,) if target_key else ()),
            )
            if not bool(occupancy.get("ok", False)):
                conflicts = tuple(
                    dict(row or {})
                    for row in tuple(occupancy.get("conflicts") or ())
                )
                labels = ", ".join(
                    f"{str(row.get('room_resref') or 'room')} "
                    f"({float(row.get('overlap_area_m2', 0.0)):.2f} m²)"
                    for row in conflicts
                )
                target_edge = int(placement.get("target_edge_index", -1))
                target_pair = (target_key, target_edge)
                if (
                    target_key
                    and target_edge >= 0
                    and target_pair not in set(_excluded_wall_targets)
                    and len(_excluded_wall_targets) < 32
                ):
                    # The nearest portal can be blocked by an adjacent room.
                    # Retry the same user drop against the next free wall of
                    # the intended authored room. No project mutation occurs
                    # before this audit, so the retry remains one undoable
                    # placement transaction.
                    return self.add_authored_environment_kit_piece(
                        piece_id=piece_id,
                        position=position,
                        rotation_degrees_z=rotation_degrees_z,
                        scale=scale,
                        target_room_resref=target_key,
                        resource_manager=resource_manager,
                        _excluded_wall_targets=(
                            *tuple(_excluded_wall_targets),
                            target_pair,
                        ),
                    )
                raise ValueError(
                    f"Ghost Studio could not find a free doorway for {piece.label}. "
                    f"It would overlap {labels}. Drag it toward another room wall or move a neighboring room "
                    "to create enough space; the retail walkmesh was left unchanged."
                )
            primitive = replace(
                primitive,
                metadata={
                    **dict(primitive.metadata or {}),
                    "environment_kit_occupancy_audit": occupancy,
                },
            )

        target_opening_name = ""
        if bool(placement.get("target_is_authored_wall", False)):
            target_key = normalise_resref(str(placement.get("target_room_resref") or ""))
            target_edge = int(placement.get("target_edge_index", -1))
            if not target_key or target_edge < 0:
                raise ValueError("The authored-wall snap target is no longer valid.")
            source_style_id = environment_kit_builder_style_id(
                piece.game,
                piece.module_resref,
                piece.collection_id,
            )
            source_style = next(
                (
                    candidate
                    for candidate in available_pascal_building_styles(game)
                    if candidate.style_id.strip().lower() == source_style_id.strip().lower()
                ),
                None,
            )
            source_profile = (
                str(source_style.architecture_profile or "").strip().lower()
                if source_style is not None
                else ""
            )
            target_profile = pascal_architecture_profile_for_room(authored, target_key)
            source_door_spec = pascal_architecture_door_spec(game, source_profile)
            target_door_spec = pascal_architecture_door_spec(game, target_profile)
            transition_asset_id = module_transition_asset_for_profiles(target_profile, source_profile)
            cave_profiles = {"korriban_caves_k1", "korriban_caves_k2"}
            open_cave_transition = bool(transition_asset_id) and (
                source_profile in cave_profiles or target_profile in cave_profiles
            )
            # When a measured architectural stock room meets an organic cave,
            # let the architectural side own the transition hardware.  The
            # target cave has no door contract of its own, so without this
            # adapter the retail DOR_LKO04 threshold would be left as a bare
            # rectangular hole and the UTD actor would never be authored.
            cross_style_door_spec = (
                source_door_spec
                if source_door_spec is not None
                and target_door_spec is None
                and source_profile != target_profile
                and not open_cave_transition
                else None
            )
            connection_metadata = {
                "connected_room_resref": room_key,
                "connected_opening_name": source_magnet_id,
                "connection_state": "connected",
                "walkmesh_portal": True,
                "walkmesh_portal_inset_m": 0.0
                if physical_portal_aligned
                else self._environment_kit_wall_portal_inset(
                    authored,
                    target_room_resref=target_key,
                    target_edge_index=target_edge,
                    source_portal=source_portal,
                    source_hook=source_hook_row,
                ),
                "walkmesh_portal_width_m": float((source_portal or {}).get("width_m", 0.0) or 0.0),
                "walkmesh_portal_source": "environment_kit_stock_threshold",
            }
            if transition_asset_id:
                connection_metadata.update(
                    {
                        "module_transition_asset_id": transition_asset_id,
                        "module_transition_target_profile": source_profile,
                        "module_transition_floor_required": True,
                        "module_transition_owner": "authored_target_room",
                        "module_transition_source": "environment_kit_stock_threshold",
                        "open_module_transition": open_cave_transition,
                        "cave_archway_transition": open_cave_transition,
                        "suppress_door_actor": open_cave_transition,
                    }
                )
            if cross_style_door_spec is not None:
                connection_metadata.update(
                    {
                        "cross_style_transition": True,
                        "cross_style_source_profile": source_profile,
                        "cross_style_target_profile": target_profile,
                        "door_template_resref": str(cross_style_door_spec["template_resref"]),
                        "door_model_resref": str(cross_style_door_spec["model_resref"]),
                        "door_appearance_id": int(cross_style_door_spec["appearance_id"]),
                        "door_aperture_width_m": float(cross_style_door_spec["opening_width_m"]),
                        "door_aperture_height_m": float(cross_style_door_spec["opening_height_m"]),
                        "door_outer_width_m": float(
                            cross_style_door_spec.get(
                                "frame_width_m",
                                cross_style_door_spec["opening_width_m"],
                            )
                        ),
                        "door_outer_height_m": float(
                            cross_style_door_spec.get(
                                "frame_height_m",
                                cross_style_door_spec["opening_height_m"],
                            )
                        ),
                        "architecture_role": f"{source_profile}_to_{target_profile}_doorway",
                    }
                )
            authored = set_pascal_building_opening_in_project(
                authored,
                room_resref=target_key,
                edge_index=target_edge,
                opening_kind="door",
                center_fraction=float(placement.get("target_center_fraction", 0.5) or 0.5),
                width=float(placement.get("opening_width", 1.8) or 1.8),
                height=float(placement.get("opening_height", 2.4) or 2.4),
                bottom=float(placement.get("opening_bottom", 0.0) or 0.0),
                connection_metadata=connection_metadata,
            )
            target_room_after = next(room for room in authored.rooms if room.normalised_resref() == target_key)
            target_opening_name = next(
                (
                    str(opening.name or "")
                    for opening in tuple(getattr(target_room_after.primitive, "openings", ()) or ())
                    if str(dict(opening.metadata or {}).get("connected_room_resref") or "").strip().lower() == room_key
                    and str(dict(opening.metadata or {}).get("connected_opening_name") or "").strip().lower()
                    == source_magnet_id.strip().lower()
                ),
                "",
            )
            if cross_style_door_spec is not None and target_opening_name:
                target_primitive_after = target_room_after.primitive
                target_points_after = tuple(getattr(target_primitive_after, "points", ()) or ())
                target_start = target_points_after[target_edge]
                target_end = target_points_after[(target_edge + 1) % len(target_points_after)]
                target_origin = tuple(
                    float(value)
                    for value in tuple(target_room_after.position or (0.0, 0.0, 0.0))[:3]
                )
                center_fraction = float(placement.get("target_center_fraction", 0.5) or 0.5)
                opening_bottom = float(placement.get("opening_bottom", 0.0) or 0.0)
                door_position = (
                    target_origin[0]
                    + float(target_start[0])
                    + (float(target_end[0]) - float(target_start[0])) * center_fraction,
                    target_origin[1]
                    + float(target_start[1])
                    + (float(target_end[1]) - float(target_start[1])) * center_fraction,
                    target_origin[2]
                    + float(getattr(target_primitive_after, "z", 0.0) or 0.0)
                    + opening_bottom,
                )
                door_bearing = math.atan2(
                    float(target_end[1]) - float(target_start[1]),
                    float(target_end[0]) - float(target_start[0]),
                )
                door_update = add_authored_gameplay_placement(
                    authored,
                    kind="door",
                    template_resref=str(cross_style_door_spec["template_resref"]),
                    tag=f"{target_key}_{target_opening_name}"[:32],
                    position=door_position,
                    bearing=door_bearing,
                )
                authored = door_update.project
                patched_rooms: list[AuthoredRoomSpec] = []
                for current in authored.rooms:
                    if current.normalised_resref() != target_key:
                        patched_rooms.append(current)
                        continue
                    patched_openings = tuple(
                        replace(
                            opening,
                            metadata={
                                **dict(opening.metadata or {}),
                                "door_placement_id": door_update.placement_id,
                                "cross_style_transition_actor": True,
                            },
                        )
                        if str(opening.name or "").strip() == target_opening_name
                        else opening
                        for opening in tuple(getattr(current.primitive, "openings", ()) or ())
                    )
                    patched_rooms.append(
                        replace(
                            current,
                            primitive=replace(current.primitive, openings=patched_openings),
                        )
                    )
                authored = replace(authored, rooms=tuple(patched_rooms))
                target_room_after = next(
                    room for room in authored.rooms if room.normalised_resref() == target_key
                )
            connected_rooms: list[AuthoredRoomSpec] = []
            for current in authored.rooms:
                if current.normalised_resref() != target_key:
                    connected_rooms.append(current)
                    continue
                connections = list(dict(current.metadata or {}).get("environment_kit_connections") or ())
                connections.append(
                    {
                        "room_resref": room_key,
                        "piece_id": piece.piece_id,
                        "edge_index": target_edge,
                        "center_fraction": float(placement.get("target_center_fraction", 0.5) or 0.5),
                        "source_magnet_id": str(placement.get("source_magnet_id") or ""),
                    }
                )
                connected_rooms.append(
                    replace(
                        current,
                        metadata={
                            **dict(current.metadata or {}),
                            "environment_kit_connections": connections,
                        },
                    )
                )
            authored = replace(authored, rooms=tuple(connected_rooms))

        all_rooms = tuple(room.normalised_resref() for room in authored.rooms) + (room_key,)
        existing = tuple(
            replace(room, visible_rooms=tuple(dict.fromkeys(tuple(room.visible_rooms or ()) + (room_key,))))
            for room in authored.rooms
        )
        room = AuthoredRoomSpec(
            room_resref=room_key,
            primitive=primitive,
            position=point,
            visible_rooms=all_rooms,
            metadata={
                "primitive": "imported_mesh",
                "source": "map_studio:environment_kit",
                "environment_kit_piece_id": piece.piece_id,
                "environment_kit_collection_id": piece.collection_id,
                "environment_kit_label": piece.label,
                "environment_kit_class_id": piece.class_id,
                "environment_kit_role": piece.role,
                "environment_kit_anchor_mode": piece.anchor_mode,
                "environment_kit_wall_aligned": wall_aligned,
                "environment_kit_source_game": piece.game,
                "environment_kit_source_module": piece.module_resref,
                "environment_kit_source_room": piece.room_resref,
                "environment_kit_rotation_degrees_z": rotation_degrees_z,
                "environment_kit_scale": scale,
                "environment_kit_magnet_snapped": bool(placement.get("magnet_snapped", False)),
                "environment_kit_source_magnet_id": str(placement.get("source_magnet_id") or ""),
                "environment_kit_target_magnet_id": str(placement.get("target_magnet_id") or ""),
                "environment_kit_target_piece_id": str(placement.get("target_piece_id") or ""),
                "environment_kit_target_room_resref": normalise_resref(
                    str(placement.get("target_room_resref") or target_room_resref)
                ),
                "environment_kit_target_is_authored_wall": bool(placement.get("target_is_authored_wall", False)),
                "environment_kit_target_edge_index": int(placement.get("target_edge_index", -1)),
                "environment_kit_target_center_fraction": float(
                    placement.get("target_center_fraction", 0.5) or 0.5
                ),
                "environment_kit_opening_width": float(placement.get("opening_width", 0.0) or 0.0),
                "environment_kit_opening_height": float(placement.get("opening_height", 0.0) or 0.0),
                "environment_kit_surface_target_room_resref": normalise_resref(target_room_resref),
                "connection_points": connection_points,
                "surface_snapped": True,
            },
        )
        before = self._capture_map_studio_command_state()
        updated_project = replace(authored, rooms=existing + (room,))
        target_connection_room = normalise_resref(str(placement.get("target_room_resref") or ""))
        target_connection_hook = (
            target_opening_name
            if bool(placement.get("target_is_authored_wall", False))
            else str(placement.get("target_magnet_id") or "")
        )
        if bool(placement.get("magnet_snapped", False)) and source_magnet_id and target_connection_room and target_connection_hook:
            from .authored_module_walkmesh import (
                compile_authored_room_connection_walkmeshes,
                upsert_authored_walkmesh_room_connection,
            )

            updated_project = upsert_authored_walkmesh_room_connection(
                updated_project,
                source_room_resref=room_key,
                source_hook_name=source_magnet_id,
                target_room_resref=target_connection_room,
                target_hook_name=target_connection_hook,
                connection_source="map_studio_environment_kit_drop",
            )
            walkmesh_build = compile_authored_room_connection_walkmeshes(updated_project)
            if not walkmesh_build.ready:
                raise ValueError(" ".join(walkmesh_build.blocking_issues))
            if not bool(placement.get("target_is_authored_wall", False)):
                target_piece = environment_kit_piece(str(placement.get("target_piece_id") or ""))
                source_style_id = environment_kit_builder_style_id(
                    piece.game,
                    piece.module_resref,
                    piece.collection_id,
                )
                target_style_id = (
                    environment_kit_builder_style_id(
                        target_piece.game,
                        target_piece.module_resref,
                        target_piece.collection_id,
                    )
                    if target_piece is not None
                    else ""
                )
                source_style = next(
                    (
                        candidate
                        for candidate in available_pascal_building_styles(game)
                        if candidate.style_id.strip().lower() == source_style_id.strip().lower()
                    ),
                    None,
                )
                door_spec = (
                    pascal_architecture_door_spec(game, source_style.architecture_profile)
                    if source_style is not None and source_style_id == target_style_id
                    else None
                )
                portal = next(
                    (
                        candidate
                        for candidate in walkmesh_build.portals
                        if {
                            candidate.source_room_resref,
                            candidate.target_room_resref,
                        }
                        == {room_key, target_connection_room}
                    ),
                    None,
                )
                if door_spec is not None and portal is not None:
                    source_wok = walkmesh_build.room_woks[portal.source_room_resref]
                    face = source_wok.faces[portal.source_face_index]
                    indices = (int(face.v1), int(face.v2), int(face.v3))
                    first_index = indices[int(portal.source_local_edge)]
                    second_index = indices[(int(portal.source_local_edge) + 1) % 3]
                    first = source_wok.verts[first_index]
                    second = source_wok.verts[second_index]
                    bearing = math.atan2(
                        float(second[1]) - float(first[1]),
                        float(second[0]) - float(first[0]),
                    )
                    midpoint = tuple(
                        (
                            float(portal.source_midpoint[axis])
                            + float(portal.target_midpoint[axis])
                        )
                        * 0.5
                        for axis in range(3)
                    )
                    door_update = add_authored_gameplay_placement(
                        updated_project,
                        kind="door",
                        template_resref=str(door_spec["template_resref"]),
                        tag=f"{room_key}_{target_connection_room}_door"[:32],
                        position=midpoint,
                        bearing=bearing,
                    )
                    updated_project = door_update.project
            project_extra = dict(updated_project.extra or {})
            project_extra["last_walkmesh_build"] = {
                "operation": "place_environment_kit_room",
                "auto_generated": True,
                "source_room_resref": room_key,
                "target_room_resref": target_connection_room,
                "portal_count": len(walkmesh_build.portals),
                "midpoint_gaps_m": [float(portal.midpoint_gap) for portal in walkmesh_build.portals],
                "room_face_counts": {
                    room_resref: len(tuple(wok.faces or ()))
                    for room_resref, wok in walkmesh_build.room_woks.items()
                },
                "ready": True,
            }
            updated_project = replace(updated_project, extra=project_extra)
        environment_update = apply_authored_environment_style_defaults(
            updated_project,
            environment_kit_builder_style_id(piece.game, piece.module_resref, piece.collection_id),
        )
        updated_project = environment_update.project
        self._store_authored_project(updated_project)
        self.model.log(
            f"Placed environment-kit piece {piece.label} as {room_key} at "
            f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}."
            + (f" {environment_update.summary}" if environment_update.applied else "")
        )
        self._record_map_studio_command(
            action_key="map_studio.environment_kit.place",
            label=f"Place environment piece {piece.label}",
            before=before,
            metadata={
                "piece_id": piece.piece_id,
                "collection_id": piece.collection_id,
                "room_resref": room_key,
                "position": list(point),
                "rotation_degrees_z": rotation_degrees_z,
                "scale": scale,
                "magnet_snapped": bool(placement.get("magnet_snapped", False)),
                "target_is_authored_wall": bool(placement.get("target_is_authored_wall", False)),
                "target_room_resref": str(placement.get("target_room_resref") or ""),
                "target_edge_index": int(placement.get("target_edge_index", -1)),
                "walkmesh_auto_generated": bool(
                    placement.get("magnet_snapped", False)
                    and source_magnet_id
                    and target_connection_room
                    and target_connection_hook
                ),
            },
        )
        return room_key

    def _next_added_room_position(self) -> tuple[float, float, float]:
        """A non-overlapping drop point just east of the current module bounds."""

        snapshot = self._map_studio_authored_project_snapshot()
        max_x = 0.0
        found = False
        for room in tuple(getattr(snapshot, "rooms", ()) or ()):
            values = tuple(getattr(room, "position", ()) or ())
            if len(values) >= 1:
                max_x = max(max_x, float(values[0]))
                found = True
        return (max_x + 30.0 if found else 0.0, 0.0, 0.0)

    def authored_room_doorway_choices(self) -> tuple[dict[str, Any], ...]:
        """Return rooms and their recorded door hooks for the snapping UI."""

        from .map_studio_room_snapping import authored_room_door_hooks

        snapshot = self._map_studio_authored_project_snapshot()
        rows: list[dict[str, Any]] = []
        for room in tuple(getattr(snapshot, "rooms", ()) or ()):
            hooks = authored_room_door_hooks(room)
            rows.append(
                {
                    "room_resref": room.normalised_resref(),
                    "doors": tuple(hook.door for hook in hooks),
                    "hook_count": len(hooks),
                }
            )
        return tuple(rows)

    def map_studio_building_style_context(self) -> dict[str, str]:
        """Return the authored room style that Builder should restore on open.

        KMAP persists the Pascal style on each floor-plan primitive.  Prefer the
        currently active authored room, then the first styled room, so reopening
        a Shadowlands or other trained kit cannot silently present the default
        Endar Spire controls and mismatched content shelves.
        """

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return {}
        active_room = normalise_resref(str(getattr(self.model, "active_room_id", "") or ""))
        rooms = tuple(authored.rooms or ())
        ordered = tuple(
            room for room in rooms if active_room and room.normalised_resref() == active_room
        ) + tuple(
            room for room in rooms if not active_room or room.normalised_resref() != active_room
        )
        for room in ordered:
            primitive = getattr(room, "primitive", None)
            metadata = dict(getattr(primitive, "metadata", {}) or {})
            style_id = str(metadata.get("style_id") or "").strip()
            if not style_id:
                continue
            return {
                "style_id": style_id,
                "environment_kind": str(metadata.get("building_kind") or "").strip().lower(),
                "room_resref": room.normalised_resref(),
            }
        return {}

    def snap_authored_rooms_at_doorway(
        self,
        *,
        source_room_resref: str,
        source_door: str,
        target_room_resref: str,
        target_door: str,
    ) -> tuple[bool, str]:
        """Translate one room so its door hook meets another room's. One undo step."""

        from .map_studio_room_snapping import snap_authored_room_to_room

        authored = self._load_authored_project_or_raise()
        before = self._capture_map_studio_command_state()
        try:
            result = snap_authored_room_to_room(
                authored,
                source_room_resref=normalise_resref(source_room_resref),
                source_door=source_door,
                target_room_resref=normalise_resref(target_room_resref),
                target_door=target_door,
            )
        except ValueError as exc:
            return False, str(exc)
        self._store_authored_project(result.project)
        tx = result.translation
        message = (
            f"Snapped {normalise_resref(source_room_resref)} to {normalise_resref(target_room_resref)} at their doorways "
            f"(moved {tx[0]:.2f}, {tx[1]:.2f}, {tx[2]:.2f})."
        )
        if result.warnings:
            message = f"{message} {result.warnings[0]}"
        self.model.log(f"Map Studio: {message}")
        self._record_map_studio_command(
            action_key="map_studio.rooms.snap_doorway",
            label=f"Snap {normalise_resref(source_room_resref)} to {normalise_resref(target_room_resref)}",
            before=before,
            metadata={
                "source_room": normalise_resref(source_room_resref),
                "target_room": normalise_resref(target_room_resref),
                "translation": list(tx),
            },
        )
        return True, message

    def _apply_imported_mesh_room_edit(self, *, room_resref: str, action_key: str, label: str, editor) -> tuple[bool, str]:
        from dataclasses import replace as _replace

        self._last_committed_imported_mesh_room = None
        resref = normalise_resref(room_resref)
        authored = self._load_authored_project_or_raise()
        target = next(
            (room for room in authored.rooms if room.normalised_resref() == resref),
            None,
        )
        if target is None:
            return False, f"Room {resref} is not an authored room."
        before = self._capture_map_studio_command_state()
        if not isinstance(target.primitive, ImportedMeshRoomPrimitive):
            # Component modeling is universal: primitive/floor-plan/terrain rooms bake
            # to editable mesh on their first component edit, so a translated
            # Create-menu cube can be edited just like a converted stock room.
            from dataclasses import replace as _bake_replace

            geometry = compile_authored_room_spec(target)
            meshes = (geometry.room_mesh,) + tuple(geometry.helper_meshes or ())
            surfaces = tuple(
                ImportedMeshSurface(
                    name=str(mesh.name or f"{resref}_srf{index}"),
                    texture=str(mesh.texture or ""),
                    vertices=tuple(mesh.vertices),
                    faces=tuple(mesh.faces),
                    uvs=tuple(mesh.uvs or ()),
                    normals=tuple(mesh.normals or ()),
                    diffuse=tuple(mesh.diffuse),
                    ambient=tuple(mesh.ambient),
                )
                for index, mesh in enumerate(meshes)
                if mesh.vertices and mesh.faces
            )
            if not surfaces:
                return False, f"Room {resref} has no bakeable geometry."
            baked = ImportedMeshRoomPrimitive(
                room_resref=resref,
                surfaces=surfaces,
                source_model=resref,
                game=str(self.project.game or "K1").upper(),
                wok=geometry.wok,
                metadata={"baked_from": type(target.primitive).__name__},
            )
            rooms = tuple(
                room if room is not target else _bake_replace(room, primitive=baked) for room in authored.rooms
            )
            authored = _bake_replace(authored, rooms=rooms)
            target = next(room for room in authored.rooms if room.normalised_resref() == resref)
            self.model.log(f"Baked room {resref} to an editable component-modeling mesh.")
        try:
            updated_primitive = editor(target.primitive)
        except ValueError as exc:
            return False, str(exc)
        if updated_primitive is target.primitive:
            return True, "No change."
        # If the edit changed the room's render surfaces (not just its WOK),
        # mark the render geometry edited so export rebuilds the room MDL
        # instead of preserving the original stock model. WOK-only edits (the
        # floor-fill walkmesh repair) leave surfaces untouched and keep the
        # room eligible for original-model/light preservation.
        if tuple(getattr(updated_primitive, "surfaces", ()) or ()) != tuple(getattr(target.primitive, "surfaces", ()) or ()):
            updated_metadata = dict(getattr(updated_primitive, "metadata", {}) or {})
            if not updated_metadata.get("render_geometry_edited"):
                updated_metadata["render_geometry_edited"] = True
                updated_primitive = _replace(updated_primitive, metadata=updated_metadata)
        rooms = tuple(
            room if room is not target else _replace(room, primitive=updated_primitive) for room in authored.rooms
        )
        self._store_authored_project(_replace(authored, rooms=rooms))
        updated_room = next(room for room in rooms if room.normalised_resref() == resref)
        self._last_committed_imported_mesh_room = (
            self.project.extra_sections.get("authored_module"),
            resref,
            updated_room,
        )
        self.model.log(label)
        self._record_map_studio_command(
            action_key=action_key,
            label=label,
            before=before,
            metadata={"room_resref": resref},
        )
        return True, label

    def auto_generate_map_studio_walkmesh(
        self,
        *,
        slope_max_degrees: float = 45.0,
        disconnected_island_policy: str = "reject",
        source_wok_policy: str = "preserve",
    ) -> tuple[bool, str]:
        """Derive every room's walkmesh from its geometry in one undoable step.

        The robust "Auto Generate Walkmesh" one-button path: for every imported
        room, replace its WOK with a fresh one built from the room's render
        geometry using the studied KOTOR conventions (walkable up-facing floors
        <= slope, omitted walls/unsafe slopes, dropped ceilings).  The default
        rejects disconnected generated islands; recovery tooling can request
        ``preserve`` only after reviewing that topology explicitly. Existing
        imported WOKs are preserved unless a caller explicitly requests
        destructive replacement. Primitive and terrain rooms keep their
        compiled WOK. Records one undo command when geometry changes.
        """

        from dataclasses import replace as _replace

        from .authored_imported_mesh import generate_room_walkmesh_from_geometry
        from .authored_module_walkmesh import resolve_room_wok_module_offset

        authored = self._load_authored_project_or_raise()
        before = self._capture_map_studio_command_state()
        rooms = list(authored.rooms)
        total_floor = 0
        total_wall = 0
        total_transitions = 0
        regenerated = 0
        non_imported_skipped: list[str] = []
        no_floor_skipped: list[str] = []
        source_wok_preserved: list[str] = []
        generation_blocked: list[str] = []
        density_warnings: list[str] = []
        for index, room in enumerate(rooms):
            if not isinstance(getattr(room, "primitive", None), ImportedMeshRoomPrimitive):
                non_imported_skipped.append(room.normalised_resref())
                continue
            position = tuple(float(value) for value in tuple(room.position or (0.0, 0.0, 0.0))[:3])
            wok_offset, _alignment_warning = resolve_room_wok_module_offset(room)
            source_wok_to_render_offset = tuple(wok_offset[axis] - position[axis] for axis in range(3))
            updated_primitive, report = generate_room_walkmesh_from_geometry(
                room.primitive,
                slope_max_degrees=float(slope_max_degrees),
                source_wok_to_render_offset=source_wok_to_render_offset,
                disconnected_island_policy=disconnected_island_policy,
                source_wok_policy=source_wok_policy,
            )
            if updated_primitive is room.primitive:
                blocked_reason = str(report.get("blocked_reason") or "").strip()
                if blocked_reason:
                    generation_blocked.append(blocked_reason)
                elif bool(report.get("source_wok_preserved", False)):
                    source_wok_preserved.append(room.normalised_resref())
                else:
                    no_floor_skipped.append(room.normalised_resref())
                continue
            rooms[index] = _replace(room, primitive=updated_primitive)
            regenerated += 1
            total_floor += int(report.get("walkable_floor_faces", report.get("floor_faces", 0)) or 0)
            total_wall += int(report.get("wall_faces", 0) or 0)
            total_transitions += int(report.get("mapped_transition_edges", 0) or 0)
            density_warning = str(report.get("density_warning") or "").strip()
            if density_warning:
                density_warnings.append(f"{room.normalised_resref()}: {density_warning}")
        if regenerated <= 0:
            if generation_blocked:
                return False, " ".join(generation_blocked)
            if source_wok_preserved:
                return True, (
                    f"Preserved {len(source_wok_preserved)} imported room WOK(s) without destructive render-derived "
                    "replacement. Use Fill Floor Faces for a source-seeded extension, or explicitly choose Replace "
                    "Source WOK after reviewing holes, blockers, transitions, and islands."
                )
            details: list[str] = []
            if non_imported_skipped:
                details.append(f"{len(non_imported_skipped)} room(s) are not editable imported meshes")
            if no_floor_skipped:
                details.append(f"{len(no_floor_skipped)} imported room(s) had no derivable walkable floor")
            suffix = f" ({'; '.join(details)})" if details else ""
            return False, f"Auto Generate Walkmesh made no changes{suffix}."
        self._store_authored_project(_replace(authored, rooms=tuple(rooms)))
        message = (
            f"Auto-generated walkmesh for {regenerated} room(s): {total_floor} walkable floor face(s), "
            f"{total_wall} wall/unsafe-slope face(s) omitted from collision."
        )
        if non_imported_skipped:
            message = f"{message} {len(non_imported_skipped)} non-imported room(s) kept their compiled walkmesh."
        if no_floor_skipped:
            message = f"{message} {len(no_floor_skipped)} imported room(s) had no derivable walkable floor and were unchanged."
        if source_wok_preserved:
            message = f"{message} Preserved {len(source_wok_preserved)} imported source WOK(s)."
        if total_transitions:
            message = f"{message} Preserved {total_transitions} source perimeter transition edge(s)."
        if generation_blocked:
            message = f"{message} {len(generation_blocked)} room(s) were refused by transition or structural topology checks."
        if density_warnings:
            message = f"{message} Density warning: {' '.join(density_warnings)}"
        self.model.log(f"Map Studio: {message}")
        self._record_map_studio_command(
            action_key="map_studio.walkmesh.auto_generate",
            label=f"Auto Generate Walkmesh ({regenerated} room(s))",
            before=before,
            metadata={
                "rooms": regenerated,
                "floor_faces": total_floor,
                "wall_faces": total_wall,
                "mapped_transition_edges": total_transitions,
                "generation_blocked_rooms": len(generation_blocked),
                "disconnected_island_policy": str(disconnected_island_policy),
                "source_wok_policy": str(source_wok_policy),
                "density_warning_rooms": len(density_warnings),
                "non_imported_rooms": len(non_imported_skipped),
                "no_floor_rooms": len(no_floor_skipped),
                "source_wok_preserved_rooms": len(source_wok_preserved),
            },
        )
        return True, message

    def prepare_imported_room_walkmesh_generation_intent(
        self,
        *,
        room_resref: str,
        surface_faces: dict[int, tuple[int, ...] | None],
        reason: str,
    ) -> tuple[bool, str]:
        """Persist a reviewed floor-surface/face selection before generation.

        Imported render geometry with no source WOK is intentionally not safe
        for one-click inference: roofs and furniture can be upward-facing too.
        This records the creator's explicit selection as one undoable KMAP
        operation; ``auto_generate_map_studio_walkmesh`` then applies the
        normal slope, topology, perimeter, and island gates to that selection.
        """

        resref = normalise_resref(room_resref)
        clean_reason = str(reason or "").strip()
        if not clean_reason:
            return False, "Walkmesh floor review requires a written reason."
        return self._apply_imported_mesh_room_edit(
            room_resref=resref,
            action_key="map_studio.imported_mesh.walkmesh_generation_intent",
            label=f"Review floor geometry for {resref} walkmesh generation",
            editor=lambda primitive: prepare_imported_mesh_walkmesh_generation_intent(
                primitive,
                surface_faces=surface_faces,
                reason=clean_reason,
            ),
        )

    def fill_authored_room_wok_from_floors(
        self,
        *,
        room_resref: str,
        slope_max_degrees: float = 35.0,
        z_tolerance: float = 1.5,
    ) -> tuple[bool, str]:
        """Patch one imported room's WOK with walkable faces from its visible floor.

        Repairs converted rooms whose imported WOK covers only part of the
        rendered floor (an invisible cliff in PIE and in-game).  Recorded as
        one undoable command; NON_WALK coverage is respected as intentional.
        """

        resref = normalise_resref(room_resref)
        outcome: dict[str, Any] = {}

        # Render surfaces are room-local; the WOK may be module-space or
        # room-local (or mislabeled — resolve_room_wok_module_offset audits
        # that).  frame_offset maps render coordinates into the WOK's frame:
        # module placement is wok + wok_offset and also local + position, so
        # render-in-wok-frame = local + position - wok_offset.
        from .authored_module_walkmesh import resolve_room_wok_module_offset

        authored = self._load_authored_project_or_raise()
        target_room = next((room for room in authored.rooms if room.normalised_resref() == resref), None)
        if target_room is None:
            return False, f"Room {resref} is not an authored room."
        position = tuple(float(v) for v in tuple(target_room.position or (0.0, 0.0, 0.0))[:3])
        wok_offset, _warning = resolve_room_wok_module_offset(target_room)
        frame_offset = tuple(position[axis] - wok_offset[axis] for axis in range(3))

        def _editor(primitive):
            patched, report = fill_imported_wok_from_floor_surfaces(
                primitive,
                slope_max_degrees=float(slope_max_degrees),
                z_tolerance=float(z_tolerance),
                render_to_wok_offset=frame_offset,
            )
            outcome.update(report)
            return patched

        ok, message = self._apply_imported_mesh_room_edit(
            room_resref=resref,
            action_key="map_studio.imported_mesh.wok_floor_fill",
            label=f"Fill room {resref} walkmesh from visible floor geometry",
            editor=_editor,
        )
        if not ok:
            return ok, message
        added = int(outcome.get("faces_added", 0) or 0)
        if added <= 0:
            partial = int(outcome.get("faces_partial_overlap", 0) or 0)
            unstitched = int(outcome.get("faces_unstitched", 0) or 0)
            downward = int(outcome.get("faces_downward", 0) or 0)
            if partial or unstitched or downward:
                return True, (
                    f"Room {resref}: no topology-safe floor faces were added "
                    f"({partial} partial-overlap, {unstitched} unstitched, {downward} downward-facing). "
                    "Use Auto Generate Walkmesh for a deliberate full-room replacement."
                )
            return True, (
                f"Room {resref} walkmesh already covers its visible floor "
                f"({int(outcome.get('faces_already_covered', 0) or 0)} floor face(s) checked)."
            )
        return True, (
            f"Room {resref}: added {added} walkable face(s) from visible floor geometry "
            f"({int(outcome.get('faces_already_covered', 0) or 0)} already covered, "
            f"{int(outcome.get('faces_too_steep', 0) or 0)} too steep, "
            f"{int(outcome.get('faces_partial_overlap', 0) or 0)} partial-overlap, "
            f"{int(outcome.get('faces_unstitched', 0) or 0)} unstitched, "
            f"{int(outcome.get('faces_downward', 0) or 0)} downward-facing)."
        )

    def prepare_imported_room_for_static_runtime_rebuild(
        self,
        *,
        room_resref: str,
        reason: str,
    ) -> tuple[bool, str]:
        """Acknowledge a deliberate, lossy stock-runtime-graph replacement.

        Normal stock-room conversion remains export-blocked when flattening
        drops animations, model lights, emitters, or references.  This named
        operation is the explicit opt-in for a creator who is intentionally
        authoring a new static shell; it records the discarded source counts
        and reason in KMAP and lets export compile a fresh static room MDL.
        """

        resref = normalise_resref(room_resref)
        clean_reason = str(reason or "").strip()
        if not clean_reason:
            return False, "Static runtime-graph rebuild requires a written reason."
        return self._apply_imported_mesh_room_edit(
            room_resref=resref,
            action_key="map_studio.imported_mesh.static_runtime_rebuild",
            label=f"Replace {resref} source runtime graph with a new static room graph",
            editor=lambda primitive: prepare_imported_mesh_for_static_runtime_rebuild(
                primitive,
                reason=clean_reason,
            ),
        )

    def delete_imported_mesh_room_faces(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        face_indices: tuple[int, ...] | list[int],
    ) -> tuple[bool, str]:
        indices = tuple(sorted({int(v) for v in face_indices}))
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.delete_faces",
            label=f"Delete {len(indices)} face(s) from {normalise_resref(room_resref)}",
            editor=lambda primitive: delete_imported_mesh_faces(primitive, mesh_role, indices),
        )

    def set_imported_mesh_room_face_texture(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        face_indices: tuple[int, ...] | list[int],
        texture: str,
    ) -> tuple[bool, str]:
        indices = tuple(sorted({int(v) for v in face_indices}))
        clean = str(texture or "").strip().lower()
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.set_face_texture",
            label=f"Set texture {clean} on {len(indices)} face(s) of {normalise_resref(room_resref)}",
            editor=lambda primitive: set_imported_mesh_face_texture(primitive, mesh_role, indices, clean),
        )

    def extrude_imported_mesh_room_faces(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        face_indices: tuple[int, ...] | list[int],
        distance: float,
        point_normal: bool = False,
        direction: tuple[float, float, float] | None = None,
    ) -> tuple[bool, str]:
        indices = tuple(sorted({int(v) for v in face_indices}))
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.extrude_faces",
            label=f"Extrude {len(indices)} face(s) of {normalise_resref(room_resref)} by {float(distance):.2f}m",
            editor=lambda primitive: extrude_imported_mesh_faces(
                primitive,
                mesh_role,
                indices,
                float(distance),
                point_normal=bool(point_normal),
                direction=direction,
            ),
        )

    def inset_imported_mesh_room_faces(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        face_indices: tuple[int, ...] | list[int],
        inset: float,
    ) -> tuple[bool, str]:
        indices = tuple(sorted({int(v) for v in face_indices}))
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.inset_faces",
            label=f"Inset {len(indices)} face(s) of {normalise_resref(room_resref)} by {float(inset):.2f}m",
            editor=lambda primitive: inset_imported_mesh_faces(primitive, mesh_role, indices, float(inset)),
        )

    def move_imported_mesh_room_faces(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        face_indices: tuple[int, ...] | list[int],
        delta: tuple[float, float, float],
    ) -> tuple[bool, str]:
        indices = tuple(sorted({int(v) for v in face_indices}))
        offset = tuple(float(v) for v in tuple(delta)[:3])
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.move_faces",
            label=f"Move {len(indices)} face(s) of {normalise_resref(room_resref)} by {offset}",
            editor=lambda primitive: move_imported_mesh_faces(primitive, mesh_role, indices, offset),
        )

    def commit_imported_mesh_multi_cut(
        self,
        *,
        room_resref: str,
        mesh_role: str,
        anchors: tuple[MultiCutAnchor, MultiCutAnchor],
        settings: MultiCutSettings | None = None,
        expected_source_fingerprint: str = "",
        expected_result_fingerprint: str = "",
    ) -> tuple[bool, str]:
        """Authoritatively commit one previewed two-anchor Multi-Cut segment.

        The viewport preview never mutates KMAP.  Enter calls this method,
        which rebuilds the result from the current immutable source, rejects a
        stale source or mismatched preview, and records exactly one undo item.
        """

        clean_anchors = tuple(anchors or ())
        if len(clean_anchors) != 2:
            return False, "Multi-Cut needs exactly two anchors before Enter commits the segment."
        expected_source = str(expected_source_fingerprint or "").strip()
        expected_result = str(expected_result_fingerprint or "").strip()

        def _commit(primitive: ImportedMeshRoomPrimitive) -> ImportedMeshRoomPrimitive:
            session = MultiCutSession.begin(primitive, mesh_role, settings=settings)
            if expected_source and session.source_fingerprint != expected_source:
                raise ValueError("Multi-Cut source changed after preview; clear the line and place it again.")
            for anchor in clean_anchors:
                session = session.add_anchor(anchor)
            evaluation = session.commit()
            if not evaluation.ok:
                raise ValueError(evaluation.diagnostics[0] if evaluation.diagnostics else "Multi-Cut is invalid.")
            if expected_result and evaluation.result_fingerprint != expected_result:
                raise ValueError("Multi-Cut authoritative result no longer matches the visible preview.")
            self._last_committed_multi_cut_evaluation = evaluation
            return evaluation.primitive

        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key="map_studio.imported_mesh.multi_cut",
            label=f"Multi-Cut one segment across {normalise_resref(room_resref)}",
            editor=_commit,
        )

    def apply_imported_mesh_room_component_op(
        self,
        *,
        room_resref: str,
        op: str,
        mesh_role: str,
        face_index: int,
        vertex_corner: int = -1,
        edge_corners: tuple[int, int] = (-1, -1),
        face_indices: tuple[int, ...] | list[int] = (),
        delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
        amount: float = 0.25,
        segments: int = 1,
        profile: float = 0.5,
        miter: str = "auto",
        smoothing_angle_degrees: float = 180.0,
        uv_mode: str = "preserve",
        clamp_overlap: bool = True,
        max_distance: float = 0.5,
        source_vertex_index: int = -1,
        target_vertex_index: int = -1,
        target_mesh_role: str = "",
        first_vertex_index: int = -1,
        second_vertex_index: int = -1,
        merge_vertex_indices: tuple[int, ...] | list[int] = (),
        merge_edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
        merge_threshold: float = 0.01,
        loop_vertex_indices: tuple[int, ...] | list[int] = (),
        loop_edge_vertices: tuple[int, int] | list[int] = (-1, -1),
        loop_position: float = 0.5,
        edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mirror_axis: str = "x",
        mirror_center: float | tuple[float, float, float] = 0.0,
        mirror_duplicate: bool = True,
        mirror_merge_seam_tolerance: float = 1.0e-5,
        first_edge_vertices: tuple[int, int] = (-1, -1),
        second_edge_vertices: tuple[int, int] = (-1, -1),
        bridge_divisions: int = 0,
        bridge_taper: float = 0.0,
        bridge_twist_degrees: float = 0.0,
        bridge_smooth: bool = True,
        boolean_cutter_mesh_role: str = "",
        boolean_weld_tolerance: float = 1.0e-6,
        deform_vertex_indices: tuple[int, ...] | list[int] = (),
        deform_axis: str = "x",
        curvature_degrees: float = 90.0,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        lattice_control_deltas: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = (),
        lattice_bounds_min: tuple[float, float, float] | None = None,
        lattice_bounds_max: tuple[float, float, float] | None = None,
        shrink_target_surface: ImportedMeshSurface | None = None,
        shrink_projection: str = "nearest_triangle",
        shrink_offset: float = 0.0,
        shrink_align_normals: bool = False,
        wrap_driver_base: ImportedMeshSurface | None = None,
        wrap_driver_deformed: ImportedMeshSurface | None = None,
        wrap_nearest_count: int = 4,
        wrap_influence: float = 1.0,
        wrap_max_distance: float = 0.0,
        cutter_face_index: int = -1,
        make_hole_planarity_tolerance: float = 1.0e-4,
        make_hole_boundary_tolerance: float = 1.0e-6,
        quad_points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]] = (),
        quad_material: int | None = None,
        quad_texture: str = "",
        quad_lightmap: str = "",
        quad_normal_hint: tuple[float, float, float] | None = None,
        quad_planarity_tolerance: float = 0.25,
        quad_auto_weld: bool = True,
        quad_weld_tolerance: float = 1.0e-4,
    ) -> tuple[bool, str]:
        """Apply one component-modeling edge/vertex/face operation with undo.

        Position-welded ops (move/weld/collapse/flatten) touch every seam
        copy of the affected vertices across all imported surfaces.
        """

        resref = normalise_resref(room_resref)
        face = int(face_index)
        faces = tuple(sorted({int(v) for v in face_indices})) or (face,)
        edge = tuple(int(v) for v in tuple(edge_corners)[:2])
        offset = tuple(float(v) for v in tuple(delta)[:3])
        selected_vertices = tuple(sorted({int(value) for value in tuple(deform_vertex_indices)})) or None
        operation = str(op or "")
        if operation == "shrink_wrap" and shrink_target_surface is None:
            return False, "Baked ShrinkWrap requires a Make Live target surface."
        if operation == "wrap_deform" and (wrap_driver_base is None or wrap_driver_deformed is None):
            return False, "Baked Wrap requires both the captured driver baseline and its edited surface."
        if operation == "lattice_deform" and len(tuple(lattice_control_deltas)) != 8:
            return False, "Baked Lattice requires eight 2x2x2 control-point deltas."
        editors = {
            "vertex_move": lambda p: move_imported_mesh_vertex(p, mesh_role, face, int(vertex_corner), offset),
            "vertex_weld": lambda p: weld_imported_mesh_vertex(
                p, mesh_role, face, int(vertex_corner), max_distance=float(max_distance)
            ),
            "vertex_delete": lambda p: delete_imported_mesh_vertex_faces(p, mesh_role, face, int(vertex_corner)),
            "edge_move": lambda p: move_imported_mesh_edge(p, mesh_role, face, edge, offset),
            "edge_extrude": lambda p: extrude_imported_mesh_edge(p, mesh_role, face, edge, offset),
            "edge_bevel": lambda p: bevel_imported_mesh_edge(
                p,
                mesh_role,
                face,
                edge,
                float(amount),
                segments=int(segments),
                profile=float(profile),
                miter=str(miter or "auto"),
                smoothing_angle_degrees=float(smoothing_angle_degrees),
                uv_mode=str(uv_mode or "preserve"),
                clamp_overlap=bool(clamp_overlap),
            ),
            "multi_edge_bevel": lambda p: bevel_imported_mesh_edges(
                p,
                mesh_role,
                tuple(tuple(int(value) for value in tuple(selected_edge)[:2]) for selected_edge in tuple(edge_vertex_indices)),
                float(amount),
                segments=int(segments),
                profile=float(profile),
                miter=str(miter or "auto"),
                smoothing_angle_degrees=float(smoothing_angle_degrees),
                uv_mode=str(uv_mode or "preserve"),
                clamp_overlap=bool(clamp_overlap),
            ),
            "edge_split": lambda p: split_imported_mesh_edge(p, mesh_role, face, edge),
            "edge_collapse": lambda p: collapse_imported_mesh_edge(p, mesh_role, face, edge),
            "edge_delete": lambda p: delete_imported_mesh_edge_faces(p, mesh_role, face, edge),
            "face_flat": lambda p: flatten_imported_mesh_faces(p, mesh_role, faces),
            "face_flip": lambda p: flip_imported_mesh_faces(p, mesh_role, faces),
            "face_split": lambda p: split_imported_mesh_face(p, mesh_role, face),
            "face_split_at_point": lambda p: split_imported_mesh_face_at_point(
                p, mesh_role, face, tuple(float(value) for value in tuple(point)[:3])
            ),
            "target_weld": lambda p: weld_imported_mesh_vertices(
                p,
                mesh_role,
                int(source_vertex_index),
                int(target_vertex_index),
                target_mesh_role=str(target_mesh_role or mesh_role),
            ),
            "connect_vertices": lambda p: connect_imported_mesh_vertices(
                p, mesh_role, int(first_vertex_index), int(second_vertex_index)
            ),
            "merge_components": lambda p: merge_imported_mesh_components(
                p,
                mesh_role,
                tuple(int(value) for value in tuple(merge_vertex_indices)),
                border_edges=tuple(
                    tuple(int(value) for value in tuple(edge))
                    for edge in tuple(merge_edge_vertex_indices)
                ),
                threshold=float(merge_threshold),
            ),
            "fill_boundary_loop": lambda p: fill_imported_mesh_boundary_loop(
                p, mesh_role, tuple(int(value) for value in tuple(loop_vertex_indices))
            ),
            "insert_edge_loop": lambda p: insert_imported_mesh_edge_loop(
                p,
                mesh_role,
                tuple(int(value) for value in tuple(loop_edge_vertices)[:2]),
                position=float(loop_position),
            ),
            "soften_edges": lambda p: soften_imported_mesh_edges(
                p, mesh_role, tuple(tuple(int(value) for value in edge[:2]) for edge in tuple(edge_vertex_indices))
            ),
            "harden_edges": lambda p: harden_imported_mesh_edges(
                p, mesh_role, tuple(tuple(int(value) for value in edge[:2]) for edge in tuple(edge_vertex_indices))
            ),
            "soften_faces": lambda p: set_imported_mesh_face_smoothing(p, mesh_role, faces, soften=True),
            "harden_faces": lambda p: set_imported_mesh_face_smoothing(p, mesh_role, faces, soften=False),
            "make_hole": lambda p: make_hole_in_imported_mesh_face(
                p,
                mesh_role,
                face,
                cutter_face_index=int(cutter_face_index),
                planarity_tolerance=max(0.0, float(make_hole_planarity_tolerance)),
                boundary_tolerance=max(0.0, float(make_hole_boundary_tolerance)),
            ),
            "quad_draw": lambda p: append_imported_mesh_quad(
                p,
                mesh_role,
                tuple(tuple(float(component) for component in tuple(value)[:3]) for value in tuple(quad_points)),
                material=quad_material,
                texture=str(quad_texture or ""),
                lightmap=str(quad_lightmap or ""),
                normal_hint=(
                    tuple(float(component) for component in tuple(quad_normal_hint)[:3])
                    if quad_normal_hint is not None
                    else None
                ),
                planarity_tolerance=max(0.0, float(quad_planarity_tolerance)),
                auto_weld=bool(quad_auto_weld),
                weld_tolerance=max(0.0, float(quad_weld_tolerance)),
            ),
            "mirror_geometry": lambda p: mirror_imported_mesh_geometry(
                p,
                axis=str(mirror_axis or "x"),
                center=mirror_center,
                duplicate=bool(mirror_duplicate),
                merge_seam_tolerance=float(mirror_merge_seam_tolerance),
                mesh_roles=(mesh_role,),
            ),
            "bridge_border_edges": lambda p: bridge_imported_mesh_border_edges(
                p,
                mesh_role,
                tuple(int(value) for value in tuple(first_edge_vertices)[:2]),
                tuple(int(value) for value in tuple(second_edge_vertices)[:2]),
                target_mesh_role=str(target_mesh_role or mesh_role),
                divisions=int(bridge_divisions),
                taper=float(bridge_taper),
                twist_degrees=float(bridge_twist_degrees),
                smooth=bool(bridge_smooth),
            ),
            "boolean_difference_closed_solids": lambda p: boolean_difference_imported_mesh_surfaces(
                p,
                mesh_role,
                str(boolean_cutter_mesh_role),
                weld_tolerance=max(0.0, float(boolean_weld_tolerance)),
            ),
            "bend_vertices": lambda p: bend_imported_mesh_vertices(
                p,
                mesh_role,
                selected_vertices,
                axis=str(deform_axis or "x"),
                curvature_degrees=float(curvature_degrees),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            "lattice_deform": lambda p: lattice_deform_imported_mesh_vertices(
                p,
                mesh_role,
                selected_vertices,
                control_deltas=tuple(
                    tuple(float(component) for component in tuple(delta_value)[:3])
                    for delta_value in tuple(lattice_control_deltas)
                ),
                bounds_min=lattice_bounds_min,
                bounds_max=lattice_bounds_max,
            ),
            "shrink_wrap": lambda p: shrink_wrap_imported_mesh_vertices(
                p,
                mesh_role,
                shrink_target_surface,
                selected_vertices,
                projection=str(shrink_projection or "nearest_triangle"),
                offset=float(shrink_offset),
                align_normals=bool(shrink_align_normals),
            ),
            "wrap_deform": lambda p: wrap_deform_imported_mesh_vertices(
                p,
                mesh_role,
                wrap_driver_base,
                wrap_driver_deformed,
                selected_vertices,
                nearest_count=int(wrap_nearest_count),
                influence=float(wrap_influence),
                max_distance=float(wrap_max_distance),
            ),
        }
        editor = editors.get(operation)
        if editor is None:
            return False, f"Unknown imported-mesh component op: {op!r}"
        labels = {
            "vertex_move": f"Move vertex of {resref}",
            "vertex_weld": f"Weld vertex of {resref} to nearest",
            "vertex_delete": f"Delete vertex fan of {resref}",
            "edge_move": f"Move edge of {resref}",
            "edge_extrude": f"Extrude edge of {resref}",
            "edge_bevel": (
                f"Bevel edge of {resref} by {abs(float(amount)):.3f}m "
                f"({max(1, int(segments))} segment(s), profile {float(profile):.2f})"
            ),
            "multi_edge_bevel": (
                f"Bevel {len(tuple(edge_vertex_indices))} edges of {resref} atomically by {abs(float(amount)):.3f}m "
                f"({max(1, int(segments))} segment(s), profile {float(profile):.2f})"
            ),
            "edge_split": f"Split edge of {resref}",
            "edge_collapse": f"Collapse edge of {resref}",
            "edge_delete": f"Delete edge faces of {resref}",
            "face_flat": f"Flatten {len(faces)} face(s) of {resref}",
            "face_flip": f"Flip {len(faces)} face(s) of {resref}",
            "face_split": f"Split face of {resref} at centroid",
            "face_split_at_point": f"Multi-Cut face of {resref} at pointer position",
            "target_weld": f"Target-weld vertex {int(source_vertex_index)} to {int(target_vertex_index)} in {resref}",
            "connect_vertices": f"Connect vertices {int(first_vertex_index)} and {int(second_vertex_index)} in {resref}",
            "merge_components": (
                f"Merge {len(tuple(merge_vertex_indices))} selected vertex/vertices in {resref}"
                if tuple(merge_vertex_indices)
                else f"Merge two selected border edges in {resref}"
            ),
            "fill_boundary_loop": f"Fill {len(tuple(loop_vertex_indices))}-vertex boundary loop in {resref}",
            "insert_edge_loop": (
                f"Insert edge loop at {float(loop_position):.3f} through the provenance-safe quad strip in {resref}"
            ),
            "soften_edges": f"Soften {len(tuple(edge_vertex_indices))} edge(s) in {resref}",
            "harden_edges": f"Harden {len(tuple(edge_vertex_indices))} edge(s) in {resref}",
            "soften_faces": f"Soften {len(faces)} face(s) in {resref}",
            "harden_faces": f"Harden {len(faces)} face(s) in {resref}",
            "make_hole": f"Make a hole in face {face} using face {int(cutter_face_index)} in {resref}",
            "quad_draw": f"Append one projected Quad Draw polygon to {mesh_role} in {resref}",
            "mirror_geometry": (
                f"Bake {'copy' if mirror_duplicate else 'cut'} mirror of {mesh_role} in {resref} "
                f"across {str(mirror_axis or 'x').upper()}"
            ),
            "bridge_border_edges": (
                f"Bake bridge with {int(bridge_divisions)} division(s) from {mesh_role} "
                f"to {str(target_mesh_role or mesh_role)} in {resref}"
            ),
            "boolean_difference_closed_solids": (
                f"Subtract closed surface {str(boolean_cutter_mesh_role)} from {mesh_role} in {resref}"
            ),
            "bend_vertices": (
                f"Bake {float(curvature_degrees):.2f}-degree {str(deform_axis or 'x').upper()} bend in {resref}"
            ),
            "lattice_deform": f"Bake 2x2x2 lattice deformation in {resref}",
            "shrink_wrap": (
                f"Bake {str(shrink_projection or 'nearest_triangle').replace('_', ' ')} ShrinkWrap in {resref}"
            ),
            "wrap_deform": f"Bake static driver-delta Wrap in {resref}",
        }
        return self._apply_imported_mesh_room_edit(
            room_resref=room_resref,
            action_key=f"map_studio.imported_mesh.{op}",
            label=labels[str(op)],
            editor=editor,
        )

    def convert_all_stock_rooms_to_imported_mesh(self, *, resource_manager: Any = None) -> tuple[bool, str]:
        """Bake every stock KMAP room into editable imported geometry.

        Run before export so the packaged module contains real MDL/WOK for
        every room instead of read-only stock references the export skips.
        """

        from dataclasses import replace as _replace

        rows = tuple(getattr(self.project, "rooms", ()) or ())
        resrefs: list[str] = []
        for row in rows:
            resref = str(getattr(row, "model_resref", "") or getattr(row, "name", "") or "").strip().lower()
            if resref and resref != "null" and resref not in resrefs:
                resrefs.append(resref)
        # Rooms imported from a stock module carry a placeholder floor-plan
        # primitive (real geometry lives in preview metadata) — without this
        # conversion component tools would edit, and export would ship, a 10x10
        # placeholder square instead of the actual room.
        extra = getattr(self.project, "extra_sections", {}) or {}
        if extra.get("authored_module") is not None:
            authored = self._load_authored_project_or_raise()
            for room in authored.rooms:
                if isinstance(room.primitive, ImportedMeshRoomPrimitive):
                    continue
                if str(dict(room.metadata or {}).get("source", "")) != "stock_module_import":
                    continue
                resref = room.normalised_resref()
                if resref and resref not in resrefs:
                    resrefs.append(resref)
        if not resrefs:
            return True, "No stock rooms to convert."
        converted: list[str] = []
        skipped: list[str] = []
        skipped_reasons: dict[str, str] = {}
        failures: list[str] = []
        for resref in resrefs:
            ok, message = self.convert_stock_room_to_imported_mesh(
                room_resref=resref,
                resource_manager=resource_manager,
            )
            if ok:
                converted.append(resref)
            elif "no editable render surfaces" in message or "could not be loaded" in message:
                # Dummy/reference rooms (emitters/hooks only, e.g. 001ebo17), or
                # rooms an ARE lists with no model bundled anywhere (common in
                # custom modules), have no geometry to edit; skipping them is
                # normal, not a failure.
                skipped.append(resref)
                skipped_reasons[resref] = str(message)
            else:
                failures.append(f"{resref}: {message}")
        if skipped_reasons and extra.get("authored_module") is not None:
            # A stock LYT/ARE can legally mention dummy rooms, or a community
            # module can omit optional room models supplied by another package.
            # Keep those source rows for round-trip diagnostics, but label them
            # explicitly so PIE never compiles the 10x10 import placeholder as
            # real render/collision geometry.
            authored = self._load_authored_project_or_raise()
            updated_rooms = []
            changed = False
            for room in tuple(authored.rooms or ()):
                resref = room.normalised_resref()
                reason = skipped_reasons.get(resref)
                if reason is None or isinstance(room.primitive, ImportedMeshRoomPrimitive):
                    updated_rooms.append(room)
                    continue
                metadata = dict(room.metadata or {})
                metadata.update(
                    {
                        "stock_geometry_status": "unresolved",
                        "stock_geometry_issue": reason,
                        "pie_exclude_unresolved_stock_geometry": True,
                    }
                )
                updated_rooms.append(_replace(room, metadata=metadata))
                changed = True
            if changed:
                self._store_authored_project(_replace(authored, rooms=tuple(updated_rooms)))
        summary = f"Converted {len(converted)} of {len(resrefs)} stock room(s) to editable geometry."
        if skipped:
            summary += (
                f" Skipped {len(skipped)} unresolved stock room reference(s) with no render geometry "
                f"(excluded from PIE collision): {', '.join(skipped[:4])}."
            )
        if failures:
            summary += f" Failures: {'; '.join(failures[:3])}" + (" ..." if len(failures) > 3 else "")
        self.model.log(summary)
        return not failures, summary

    def delete_map_studio_rooms(self, room_resrefs) -> tuple[bool, str]:
        """Delete whole rooms by resref: stock KMAP rows and authored rooms.

        One undoable command covers everything removed so a marquee-selected
        batch restores together.
        """

        from dataclasses import replace as _replace

        wanted = {normalise_resref(value) for value in (room_resrefs or ()) if str(value or "").strip()}
        wanted.discard("")
        if not wanted:
            return False, "No rooms selected."
        before = self._capture_map_studio_command_state()
        removed: set[str] = set()

        rows = list(getattr(self.project, "rooms", ()) or ())
        keep_rows = []
        for row in rows:
            key = str(getattr(row, "model_resref", "") or getattr(row, "name", "") or "").strip().lower()
            if key in wanted:
                removed.add(key)
            else:
                keep_rows.append(row)
        if len(keep_rows) != len(rows):
            self.project.rooms = keep_rows

        extra = getattr(self.project, "extra_sections", {}) or {}
        if extra.get("authored_module") is not None:
            authored = self._load_authored_project_or_raise()
            retained_rooms = tuple(room for room in authored.rooms if room.normalised_resref() not in wanted)
            kept = tuple(
                _replace(
                    room,
                    visible_rooms=tuple(
                        target
                        for target in tuple(room.visible_rooms or ())
                        if normalise_resref(target) not in wanted
                    ),
                )
                for room in retained_rooms
            )
            if len(kept) != len(authored.rooms):
                removed.update(
                    room.normalised_resref() for room in authored.rooms if room.normalised_resref() in wanted
                )
                updated_extra = dict(getattr(authored, "extra", {}) or {})
                updated_extra["vis_pairs"] = [
                    pair
                    for pair in list(updated_extra.get("vis_pairs") or ())
                    if len(tuple(pair or ())) >= 2
                    and normalise_resref(tuple(pair)[0]) not in wanted
                    and normalise_resref(tuple(pair)[1]) not in wanted
                ]
                self._store_authored_project(_replace(authored, rooms=kept, extra=updated_extra))

        if not removed:
            return False, f"No rooms matched: {', '.join(sorted(wanted))}."
        self.project.dirty = True
        names = ", ".join(sorted(removed))
        self.model.log(f"Deleted room(s) {names}; walkmesh, layout, and export proof are now stale.")
        self._record_map_studio_command(
            action_key="map_studio.rooms.delete",
            label=f"Delete {len(removed)} room(s)",
            before=before,
            metadata={"room_resrefs": sorted(removed)},
        )
        return True, f"Deleted room(s): {names}"

    def load_wok(self, path: str | Path, room_id: str = ""):
        target_room = room_id or self.model.active_room_id
        result = self.walkmesh_service.load_wok_file(self.project, path, room_id=target_room)
        if result.ok and target_room:
            self.model.loaded_walkmeshes[target_room] = result.wok
        return result

    def validate(self, *, readiness_result=None):
        issues = list(self.validator.validate(self.project))
        readiness_result = readiness_result or self.authored_module_readiness()
        issues.extend(
            authored_module_readiness_validation_issues(
                readiness_result.readiness,
                bridge_warnings=readiness_result.warnings,
                bridge_blocking_messages=readiness_result.blocking_messages,
            )
        )
        return issues

    def authored_module_readiness(self):
        """Return Map Studio authored-module readiness for the current KMAP."""

        return self._map_studio_cached_project_query(
            ("authored_module_readiness",), build_kmap_authored_module_readiness
        )

    def map_studio_workspace_modes(self) -> tuple[MapStudioWorkspaceMode, ...]:
        """Return the modder-facing Map Studio workspaces exposed by the Level Editor."""

        return (
            MapStudioWorkspaceMode(
                key="project",
                label="Project",
                summary="KMAP identity, target game, outliner, asset browser, and save/open state.",
                next_action="Create or open a KMAP, then choose the workspace for the current map task.",
            ),
            MapStudioWorkspaceMode(
                key="geometry",
                label="Room Geometry",
                summary="Build floors, walls, corridors, doorway blockouts, primitives, and component-mode edits: Object, Vertex, Edge, Face, Terrain, and Walkmesh.",
                next_action="Choose a modeling mode/tool, then create primitives, snap vertices, bevel/inset, cut, bridge, boolean, or shape the authored layout.",
            ),
            MapStudioWorkspaceMode(
                key="terrain",
                label="Terrain Builder",
                summary="Create terrain patches, sculpt heightfield samples, and check slope/walkability intent.",
                next_action="Create a terrain patch, choose a heightfield room, then raise/lower/smooth terrain samples.",
            ),
            MapStudioWorkspaceMode(
                key="walkmesh",
                label="Walkmesh",
                summary="Inspect WOK surfaces, walkable faces, non-walkable barriers, and traversal readiness.",
                next_action="Generate or inspect WOK faces before staging a playable module.",
            ),
            MapStudioWorkspaceMode(
                key="placements",
                label="Placements",
                summary="Place KOTOR creatures, placeables, doors, triggers, encounters, cameras, sounds, waypoints, and stores.",
                next_action="Search the game library or type a template resref, then place the resource in the viewport.",
            ),
            MapStudioWorkspaceMode(
                key="lighting",
                label="Lighting",
                summary="Author room lights and plan future lightmap coverage before export/game testing.",
                next_action="Add key/fill/ambient lights per room and validate lighting coverage in readiness.",
            ),
            MapStudioWorkspaceMode(
                key="scripts",
                label="Scripts + Transitions",
                summary="Assign ARE/IFO script hooks and configure door, trigger, or waypoint transition targets.",
                next_action="Set script resrefs or transition destination tags/modules when the map needs behavior or exits.",
            ),
            MapStudioWorkspaceMode(
                key="export",
                label="Export + Game Proof",
                summary="Validate, stage, install, open warp-test handoff, and record live KOTOR proof.",
                next_action="Validate first; only call the module game-ready after a staged install and recorded warp proof.",
            ),
        )

    def create_dev_test_authored_module(self, *, module_root: str = "grdev01"):
        """Store the editable first Map Studio dev-test module in the KMAP."""

        root = str(module_root or "grdev01").strip() or "grdev01"
        payload = create_dev_test_authored_module_payload(module_root=root, game=str(self.project.game or "K1").upper())
        self.project.extra_sections["authored_module"] = payload
        self.project.name = str(payload.get("module_root") or root)
        self.project.game = str(payload.get("game") or self.project.game or "K1").upper()
        self.project.dirty = True
        self.model.log(f"Created authored Map Studio module {self.project.name}.")
        return self.authored_module_readiness()

    def snap_authored_gameplay_placement_to_walkmesh(self, placement_id: str, *, downward_only: bool = False):
        """Move one authored placement onto generated WOK, optionally straight down like Unreal End."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update, snap = snap_authored_gameplay_placement_to_walkmesh(
            authored,
            placement_id,
            downward_only=bool(downward_only),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Snapped Map Studio {update.kind} placement {update.tag} to walkable WOK face {snap.face_index}."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.snap_placement_to_walkmesh",
            label=f"Snap {update.kind} placement {update.tag} to WOK",
            before=before,
            metadata={
                "placement_id": placement_id,
                "kind": update.kind,
                "tag": update.tag,
                "position": update.position,
                "face_index": snap.face_index,
                "surface_id": snap.surface_id,
                "horizontal_distance": snap.horizontal_distance,
                "downward_only": bool(downward_only),
            },
        )
        return update, snap

    def create_golden_test_authored_module(self, *, module_root: str = "grgold01"):
        """Store the canonical full Map Studio golden smoke module in the KMAP."""

        root = str(module_root or "grgold01").strip() or "grgold01"
        payload = create_golden_test_authored_module_payload(module_root=root, game=str(self.project.game or "K1").upper())
        self.project.extra_sections["authored_module"] = payload
        self.project.name = str(payload.get("module_root") or root)
        self.project.game = str(payload.get("game") or self.project.game or "K1").upper()
        self.project.dirty = True
        self.model.log(f"Created authored Map Studio golden module {self.project.name}.")
        return self.authored_module_readiness()

    def available_authored_room_presets(self):
        """Return named primitive room presets for the Map Studio Builder tab."""

        return available_authored_room_primitive_presets()

    def authored_room_connection_audit(self) -> AuthoredRoomConnectionAudit:
        """Return Holocron-inspired room-opening connection health for Rooms UI."""

        if self._map_studio_authored_project_snapshot() is None:
            return AuthoredRoomConnectionAudit()
        return self._map_studio_cached_authored_query(
            ("room_connection_audit",), audit_authored_room_connections
        )

    def connect_authored_room_openings(
        self,
        *,
        source_hook_id: str,
        target_hook_id: str,
        align_source: bool = True,
    ):
        """Align two authored openings, persist their link, and record one undo step."""

        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        update = connect_authored_room_openings_in_project(
            authored,
            source_hook_id,
            target_hook_id,
            align_source=bool(align_source),
        )
        self._store_authored_project(update.project)
        self.model.log(
            f"{update.summary} Rotation {update.rotation_degrees:.1f} degrees; "
            f"translation ({update.translation[0]:.3f}, {update.translation[1]:.3f}, {update.translation[2]:.3f})."
        )
        self._record_map_studio_command(
            action_key="map_studio.rooms.connect_openings",
            label=f"Connect {update.source_hook.room_resref} to {update.target_hook.room_resref}",
            before=before,
            metadata={
                "source_hook_id": update.source_hook.hook_id,
                "target_hook_id": update.target_hook.hook_id,
                "rotation_degrees": update.rotation_degrees,
                "translation": list(update.translation),
                "vis_intent_updated": True,
                "wok_transition_proof_required": True,
            },
        )
        return update

    def set_authored_room_opening_intent(
        self,
        *,
        hook_id: str,
        intent: str,
    ) -> AuthoredRoomOpeningIntentUpdate:
        """Persist connector/external/sealed intent as one undoable Map Studio action."""

        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        update = set_authored_room_opening_intent_in_project(
            authored,
            hook_id=str(hook_id or ""),
            intent=str(intent or ""),
        )
        self._store_authored_project(update.project)
        self.model.log(update.summary)
        self._record_map_studio_command(
            action_key="map_studio.rooms.set_opening_intent",
            label=f"Set {update.opening_name} to {update.intent}",
            before=before,
            metadata={
                "hook_id": update.hook_id,
                "room_resref": update.room_resref,
                "opening_name": update.opening_name,
                "opening_intent": update.intent,
                "sealed_door_placement_id": update.sealed_door_placement_id,
            },
        )
        return update

    def preview_authored_room_drag_snap(
        self,
        *,
        source_room_resref: str,
        world_delta: Any,
        snap_distance: float = 2.5,
        target_room_resref: str = "",
        target_edge_index: int = -1,
        target_opening_name: str = "",
    ) -> dict[str, object]:
        """Resolve a disposable doorway magnet for one whole-room drag."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return {
                "magnet_snapped": False,
                "source_room_resref": normalise_resref(source_room_resref),
                "reason": "Create or load an authored room first.",
            }
        preview = preview_authored_room_drag_snap_in_project(
            authored,
            source_room_resref=source_room_resref,
            world_delta=tuple(world_delta or ()),
            snap_distance=float(snap_distance),
            target_room_resref=str(target_room_resref or ""),
            target_edge_index=int(target_edge_index),
            target_opening_name=str(target_opening_name or ""),
        )
        return preview.as_payload()

    def connect_authored_room_drag_snap(self, preview: Any):
        """Commit a previewed doorway or terrain-edge magnet as one undo step."""

        values = dict(preview) if isinstance(preview, dict) else preview
        snap_kind = (
            str(values.get("snap_kind") or "doorway").strip().lower()
            if isinstance(values, dict)
            else str(getattr(values, "snap_kind", "doorway") or "doorway").strip().lower()
        )
        terrain_seam = snap_kind == "terrain_seam"
        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        update = connect_authored_room_drag_snap_in_project(authored, values)
        self._store_authored_project(update.project)
        self.model.log(update.summary if terrain_seam else f"{update.summary} Whole-room doorway magnet committed with one shared KOTOR door actor.")
        self._record_map_studio_command(
            action_key=(
                "map_studio.terrain.drag_snap_edge"
                if terrain_seam
                else "map_studio.rooms.drag_snap_opening"
            ),
            label=f"Snap {update.source_hook.room_resref} to {update.target_hook.room_resref}",
            before=before,
            metadata={
                "source_hook_id": update.source_hook.hook_id,
                "target_hook_id": update.target_hook.hook_id,
                "rotation_degrees": update.rotation_degrees,
                "translation": list(update.translation),
                "auto_cut_source": bool(dict(values or {}).get("auto_cut_source", False)) if isinstance(values, dict) else False,
                "snap_kind": snap_kind,
                "shared_runtime_door": not terrain_seam,
                "terrain_edge_welded": terrain_seam,
                "vis_intent_updated": not terrain_seam,
                "wok_auto_generated": True,
            },
        )
        return update

    def snap_authored_rooms_to_grid(self, room_resrefs, *, grid_size: float = 1.0) -> str:
        """Snap selected authored room positions to the KMAP layout grid."""

        selected = tuple(normalise_resref(value) for value in (room_resrefs or ()) if str(value or "").strip())
        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        updated = snap_authored_rooms_to_grid_in_project(authored, selected, grid_size=float(grid_size))
        self._store_authored_project(updated)
        self._record_map_studio_command(
            action_key="map_studio.rooms.snap_grid",
            label=f"Snap {len(selected)} room(s) to {float(grid_size):g}m grid",
            before=before,
            metadata={"room_resrefs": list(selected), "grid_size": float(grid_size)},
        )
        message = f"Snapped {len(selected)} authored room(s) to the {float(grid_size):g}m XY layout grid."
        self.model.log(message)
        return message

    def auto_arrange_authored_rooms(self, room_resrefs=(), *, spacing: float = 1.0) -> str:
        """Geometry-aware non-overlapping arrangement for selected/all authored rooms."""

        selected = tuple(normalise_resref(value) for value in (room_resrefs or ()) if str(value or "").strip())
        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        affected = len(selected) if selected else len(authored.rooms)
        updated = auto_arrange_authored_rooms_in_project(authored, selected, spacing=float(spacing))
        self._store_authored_project(updated)
        self._record_map_studio_command(
            action_key="map_studio.rooms.auto_arrange",
            label=f"Auto arrange {affected} room(s)",
            before=before,
            metadata={"room_resrefs": list(selected), "spacing": float(spacing), "geometry_aware": True},
        )
        scope = "selected" if selected else "all"
        message = f"Auto-arranged {affected} {scope} authored room(s) with {float(spacing):g}m geometry-aware spacing."
        self.model.log(message)
        return message

    def available_authored_terrain_shape_presets(self):
        """Return named terrain shape presets for the Map Studio Builder tab."""

        return available_terrain_shape_presets()

    def available_authored_walkmesh_surfaces(self):
        """Return named WOK surface choices for authored room floors."""

        return authored_walkmesh_surface_palette()

    def available_authored_composition_primitive_kinds(self):
        """Return primitive kinds that can be added to authored composition rooms."""

        return available_authored_composition_primitive_kinds()

    def available_authored_gameplay_placement_kinds(self):
        """Return supported authored gameplay placement kinds for Map Studio UI."""

        return SUPPORTED_AUTHORED_GAMEPLAY_PLACEMENTS

    def available_map_studio_component_modes(self):
        """Return visible object/component edit modes for Map Studio modeling."""

        return available_map_studio_component_modes()

    def available_map_studio_edit_mode_contexts(self):
        """Return visible top-level edit-mode context for Map Studio UX."""

        return available_map_studio_edit_mode_contexts()

    def map_studio_edit_mode_context(self, mode_label: str = "Object"):
        """Return the KOTOR-specific UX context for a visible edit mode."""

        return map_studio_edit_mode_context(mode_label)

    def available_map_studio_modeling_tools(self):
        """Return visible Maya-like, KOTOR-aware modeling tools."""

        return available_map_studio_modeling_tools()

    def available_map_studio_snap_modes(self):
        """Return visible snap modes for Map Studio modeling tools."""

        return available_map_studio_snap_modes()

    def available_map_studio_terrain_brushes(self):
        """Return visible terrain sculpt brushes for Map Studio."""

        return available_map_studio_terrain_brushes()

    def map_studio_viewport_performance_policy(self):
        """Return the interaction budget for smooth Map Studio viewport edits."""

        return map_studio_viewport_performance_policy()

    def available_map_studio_tool_belt_actions(self):
        """Return actions that can be shown in the Map Studio tool belt."""

        return available_map_studio_tool_belt_actions()

    def available_map_studio_tool_belt_presets(self):
        """Return built-in tool-belt presets for common map-building tasks."""

        return available_map_studio_tool_belt_presets()

    def map_studio_tool_belt_actions_for_preset(
        self,
        preset_key: str = "blockout",
        *,
        custom_action_keys: tuple[str, ...] | list[str] = (),
    ):
        """Resolve a preset or custom session action list into visible belt buttons."""

        return map_studio_tool_belt_actions_for_preset(
            preset_key,
            custom_action_keys=custom_action_keys,
        )

    def map_studio_tool_command_search(
        self,
        query: str = "",
        *,
        limit: int = 50,
        include_planned: bool = False,
    ):
        """Return searchable Map Studio command entries for command palettes and custom belts."""

        return map_studio_tool_command_search(
            query,
            limit=limit,
            include_planned=include_planned,
        )

    def map_studio_tool_belt_contract_audit(self):
        """Return a headless audit of visible Map Studio tool-belt action contracts."""

        return audit_map_studio_tool_belt_contract()

    def map_studio_tool_belt_preferences(self):
        """Return KMAP-persisted Map Studio tool-belt preferences."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get(MAP_STUDIO_TOOL_BELT_SECTION) or {})
        return normalise_map_studio_tool_belt_preferences(payload)

    def set_map_studio_tool_belt_preferences(
        self,
        *,
        preset_key: str = "blockout",
        custom_action_keys: tuple[str, ...] | list[str] = (),
    ):
        """Persist Map Studio tool-belt preferences into the current KMAP."""

        preferences = normalise_map_studio_tool_belt_preferences(
            preset_key=preset_key,
            custom_action_keys=custom_action_keys,
        )
        self.project.extra_sections[MAP_STUDIO_TOOL_BELT_SECTION] = preferences.to_kmap_section()
        self.project.dirty = True
        self.model.log(f"Updated Map Studio tool belt preference: {preferences.preset_key}.")
        return preferences

    def map_studio_active_selection(self) -> dict[str, Any]:
        """Return KMAP-persisted active Map Studio selection context."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get(MAP_STUDIO_ACTIVE_SELECTION_SECTION) or {})
        if int(payload.get("version") or 0) != 1:
            return {}
        return payload

    def set_map_studio_active_selection(
        self,
        *,
        component_mode: str = "object",
        workspace_key: str = "geometry",
        tool_key: str = "select",
        room_resref: str = "",
        primitive_name: str = "",
        selection_kind: str = "",
    ) -> dict[str, Any]:
        """Persist the active authored selection target without staling generated resources."""

        before = self._capture_map_studio_command_state()
        component = str(component_mode or "object").strip().lower() or "object"
        workspace = str(workspace_key or "geometry").strip().lower() or "geometry"
        tool = str(tool_key or "select").strip().lower() or "select"
        room = str(room_resref or "").strip()
        primitive = str(primitive_name or "").strip()
        kind = str(selection_kind or "").strip().lower()
        if not kind:
            kind = "composition_primitive" if primitive else ("authored_room" if room else "map_studio")
        payload = {
            "version": 1,
            "selection_kind": kind,
            "component_mode": component,
            "workspace_key": workspace,
            "tool_key": tool,
            "room_resref": room,
            "primitive_name": primitive,
        }
        self.project.extra_sections[MAP_STUDIO_ACTIVE_SELECTION_SECTION] = payload
        self.project.dirty = True
        label_target = primitive or room or "Map Studio"
        self.model.log(f"Selected Map Studio {kind.replace('_', ' ')} {label_target}.")
        self._record_map_studio_command(
            action_key="map_studio.selection.select",
            label=f"Select {label_target}",
            before=before,
            stale_outputs=(),
            readiness_impact=MAP_STUDIO_SELECTION_READINESS_IMPACT,
            metadata=payload,
        )
        return payload

    def map_studio_modeling_tool_summary(
        self,
        *,
        mode_key: str = "object",
        tool_key: str = "",
        snap_key: str = "grid",
    ) -> str:
        """Return UI-ready active modeling context text."""

        return map_studio_modeling_tool_summary(
            mode_key=mode_key,
            tool_key=tool_key,
            snap_key=snap_key,
        )

    def authored_curve_guides(self):
        """Return KMAP-authored curve guides for road, path, terrain, and PTH planning."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(("curve_guides",), authored_curve_guides)

    def authored_gameplay_palette_entries(self, rows, *, query: str = "", kind: str = ""):
        """Return game-library-backed resources that can seed gameplay placements."""

        return authored_gameplay_palette_from_library_rows(
            rows,
            game=str(getattr(self.project, "game", "") or ""),
            query=query,
            kind=kind,
        )

    def authored_gameplay_placements(self):
        """Return selectable authored gameplay placements for the current KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(("gameplay_placements",), authored_gameplay_placement_rows)

    def authored_module_entry_point(self):
        """Return the authored IFO player start for the current KMAP."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return None
        return authored.placements.entry_point

    def authored_module_entry_point_preview_row(self):
        """Return the selectable real-player viewport row for the IFO start."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return None
        player_settings = dict((getattr(self.project, "extra", {}) or {}).get("pie_player") or {})
        return authored_module_entry_point_preview_row(
            authored,
            body_model_resref=str(player_settings.get("body_resref") or "pmbam"),
            head_model_resref=str(player_settings.get("head_resref") or "pmhc01"),
        )

    def authored_room_lights(self):
        """Return selectable authored room lights for the current KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(("room_lights",), authored_room_light_rows)

    def authored_world_lighting_settings(self) -> dict[str, Any]:
        """Return normalized ARE world-lighting values for Map Studio controls."""

        if self._map_studio_authored_project_snapshot() is None:
            settings = default_authored_world_lighting_settings(
                str(getattr(self.project, "game", "") or "K1")
            )
            settings["available"] = False
            return settings
        settings = dict(self._map_studio_cached_authored_query(
            ("world_lighting",), read_authored_world_lighting_settings
        ))
        settings["available"] = True
        return settings

    def authored_lightmap_surface_rows(self) -> tuple[dict[str, Any], ...]:
        """List imported room surfaces that can enter the lightmap workflow."""

        try:
            authored = self._load_authored_project_or_raise()
        except ValueError:
            return ()
        rows: list[dict[str, Any]] = []
        for room in tuple(authored.rooms or ()):
            primitive = getattr(room, "primitive", None)
            if not isinstance(primitive, ImportedMeshRoomPrimitive):
                continue
            bake_records = dict(getattr(primitive, "metadata", {}) or {}).get("lightmap_bakes")
            bake_records = dict(bake_records) if isinstance(bake_records, dict) else {}
            for index, surface in enumerate(tuple(primitive.surfaces or ())):
                role = imported_mesh_surface_role(index)
                proof = dict(bake_records.get(role) or {})
                rows.append(
                    {
                        "room_resref": room.normalised_resref(),
                        "surface_index": index,
                        "surface_role": role,
                        "surface_name": str(surface.name or role),
                        "vertex_count": len(tuple(surface.vertices or ())),
                        "face_count": len(tuple(surface.faces or ())),
                        "has_uv2": len(tuple(surface.uvs_lm or ())) == len(tuple(surface.vertices or ()))
                        and bool(surface.vertices),
                        "lightmap_resref": str(surface.lightmap or ""),
                        "backdrop": bool(getattr(surface, "backdrop", False)),
                        "bake_status": str(proof.get("status") or ("original_preserved" if surface.lightmap else "not_baked")),
                        "engine_game_proof": bool(proof.get("engine_game_proof", False)),
                    }
                )
        return tuple(rows)

    def apply_authored_surface_lightmap(
        self,
        *,
        room_resref: str,
        surface_role_or_index: str | int,
        lightmap_resref: str,
        resolution: int = 64,
        include_world_ambient: bool = True,
        use_shadows: bool = True,
    ):
        """Bake one imported surface and commit its vanilla-shaped TPC atomically."""

        from dataclasses import replace as _replace

        from src.core.lighting.lightmap_bake_settings import LightmapBakeSettings
        from src.core.workflow.map_studio_lightmap_apply import apply_imported_surface_lightmap

        authored = self._load_authored_project_or_raise()
        clean_resref = validate_kotor_texture_resref(lightmap_resref)
        target_dir = project_texture_directory(self.project)
        sidecar_paths = tuple(target_dir / f"{clean_resref}{suffix}" for suffix in (".tpc", ".tga", ".txi"))
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        created_paths = tuple(value for value, state in sidecar_before.states if state is None)
        world = read_authored_world_lighting_settings(authored)
        sun_ambient = tuple(world.get("sun_ambient") or (64, 64, 64))
        dynamic_ambient = tuple(world.get("dynamic_ambient") or sun_ambient)
        background = tuple(
            (float(sun_ambient[index]) + float(dynamic_ambient[index])) / (2.0 * 255.0)
            for index in range(3)
        )
        settings = LightmapBakeSettings(
            resolution=int(resolution),
            bake_resolution=int(resolution),
            include_ambient=bool(include_world_ambient),
            background_color=background,
            use_shadows=bool(use_shadows),
            include_diffuse=True,
            output_format="tga",
            generate_manifest=False,
            preview_after_bake=False,
        )
        try:
            result = apply_imported_surface_lightmap(
                authored,
                room_resref=room_resref,
                surface_role_or_index=surface_role_or_index,
                lightmap_resref=clean_resref,
                resolution=int(resolution),
                settings=settings,
            )
            if not result.ok or result.sidecar is None:
                raise ValueError("; ".join(result.errors or ("Lightmap bake/apply failed.",)))
            sidecar = result.sidecar
            asset = create_project_tpc_texture_asset(
                self.project,
                resref=clean_resref,
                width=sidecar.width,
                height=sidecar.height,
                tpc_bytes=sidecar.tpc_bytes,
                source="map_studio:applied_lightmap",
                metadata={
                    "room_resref": sidecar.room_resref,
                    "surface_role": sidecar.surface_role,
                    "surface_index": sidecar.surface_index,
                    "tpc_sha256": sidecar.tpc_sha256,
                    "engine_game_proof": False,
                    "proof": dict(sidecar.proof),
                },
            )
            updated_extra = dict(getattr(result.project, "extra", {}) or {})
            registry = dict(updated_extra.get("applied_lightmaps") or {})
            registry[f"{sidecar.room_resref}:{sidecar.surface_role}"] = {
                **dict(sidecar.proof),
                "texture_id": asset.texture_id,
                "project_texture_resref": clean_resref,
                "resource_type": "tpc",
            }
            updated_extra["applied_lightmaps"] = registry
            updated_extra["last_lightmap_apply"] = registry[f"{sidecar.room_resref}:{sidecar.surface_role}"]
            updated = _replace(result.project, extra=updated_extra)
            self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
            self.project.name = updated.metadata.module_root
            self.project.game = updated.game
            self.project.dirty = True
            sidecar_patches = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        self.model.log(
            f"Applied {clean_resref}.tpc to {sidecar.room_resref}/{sidecar.surface_role}; "
            "the resource is vanilla-structural but still requires a manual KOTOR warp proof."
        )
        self._record_map_studio_command(
            action_key="map_studio.lightmap.apply_surface",
            label=f"Apply lightmap {clean_resref}",
            before=before,
            stale_outputs=("MDL", "MDX", "TPC", ".mod"),
            readiness_impact="Repackage the edited room and record a fresh manual KOTOR lighting proof.",
            metadata={
                "room_resref": sidecar.room_resref,
                "surface_role": sidecar.surface_role,
                "lightmap_resref": clean_resref,
                "resolution": sidecar.width,
                "tpc_sha256": sidecar.tpc_sha256,
                "engine_game_proof": False,
            },
            sidecar_patches=sidecar_patches,
        )
        return result

    def set_authored_world_lighting_settings(self, values: dict[str, Any]):
        """Apply one undoable ARE-only world-lighting edit to the KMAP."""

        before = self._capture_map_studio_command_state()
        authored = self._load_authored_project_or_raise()
        update = update_authored_world_lighting_settings(authored, values)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self._record_map_studio_command(
            action_key="map_studio.environment.world_lighting",
            label="Update World Lighting",
            before=before,
            stale_outputs=MAP_STUDIO_WORLD_LIGHTING_STALE_OUTPUTS,
            readiness_impact=MAP_STUDIO_WORLD_LIGHTING_READINESS_IMPACT,
            summary=update.summary,
            metadata={
                "profile": str(update.settings["profile"]),
                "fog_enabled": bool(update.settings["fog_enabled"]),
                "source": str(update.settings["source"]),
            },
        )
        self.model.log(f"{update.summary} Previous ARE export/game proof is now stale.")
        return update

    def authored_room_resrefs(self) -> tuple[str, ...]:
        """Return every authored LYT room resref, including visual backdrops."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("room_resrefs",),
            lambda authored: tuple(room.normalised_resref() for room in authored.rooms if room.normalised_resref()),
        )

    @staticmethod
    def _panorama_skybox_face_texture_rows(
        *,
        north_texture: str,
        east_texture: str,
        south_texture: str,
        west_texture: str,
        top_texture: str,
    ) -> tuple[tuple[str, str], ...]:
        return (
            ("north", str(north_texture or "").strip()),
            ("east", str(east_texture or "").strip()),
            ("south", str(south_texture or "").strip()),
            ("west", str(west_texture or "").strip()),
            ("top", str(top_texture or "").strip()),
        )

    @staticmethod
    def _validate_panorama_skybox_identity(
        *,
        room_resref: str,
        face_texture_rows: tuple[tuple[str, str], ...],
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        room_issue = authored_resref_blocking_issue("Skybox room", room_resref)
        if room_issue:
            raise ValueError(room_issue)
        clean_rows: list[tuple[str, str]] = []
        for face, value in face_texture_rows:
            issue = authored_resref_blocking_issue(f"Skybox {face} texture", value)
            if issue:
                raise ValueError(issue)
            clean_rows.append((str(face), normalise_resref(value)))
        texture_resrefs = tuple(value for _face, value in clean_rows)
        if len(set(texture_resrefs)) != len(texture_resrefs):
            raise ValueError("Each panorama skybox face requires a unique texture ResRef.")
        return normalise_resref(room_resref), tuple(clean_rows)

    def prepare_panorama_skybox(
        self,
        source_path: str | Path,
        *,
        room_resref: str,
        north_texture: str,
        east_texture: str,
        south_texture: str,
        west_texture: str,
        top_texture: str,
        face_size: int = 1_024,
        exposure_ev: float = 0.0,
        longitude_offset_degrees: float = 0.0,
        tone_mapper: str = "aces",
    ) -> MapStudioPanoramaSkyboxPreparation:
        """Decode and project a panorama without mutating KMAP or sidecars.

        This pass is intentionally safe for Map Studio's worker executor.  The
        returned state token and texture set let the GUI-thread commit reject
        intervening room or texture edits instead of silently overwriting them.
        """

        project_path = str(getattr(self.project, "path", "") or "").strip()
        if not project_path:
            raise ValueError("Save the KMAP before converting a panorama; generated TGAs live beside the project.")
        authored_state_token = self._map_studio_authored_state_token()
        authored = self._load_authored_project_or_raise()
        if not tuple(authored.rooms or ()):
            raise ValueError("Add or import at least one playable room before creating its skybox.")
        face_rows = self._panorama_skybox_face_texture_rows(
            north_texture=north_texture,
            east_texture=east_texture,
            south_texture=south_texture,
            west_texture=west_texture,
            top_texture=top_texture,
        )
        room_name, clean_rows = self._validate_panorama_skybox_identity(
            room_resref=room_resref,
            face_texture_rows=face_rows,
        )
        if any(room.normalised_resref() == room_name for room in authored.rooms):
            raise ValueError(f"Authored room {room_name} already exists.")
        existing_texture_resrefs = tuple(
            sorted(
                {
                    str(getattr(texture, "resref", "") or "").strip().lower()
                    for texture in tuple(getattr(self.project, "textures", ()) or ())
                    if str(getattr(texture, "resref", "") or "").strip()
                }
            )
        )
        collisions = sorted(set(existing_texture_resrefs) & {value for _face, value in clean_rows})
        if collisions:
            raise ValueError(
                "Panorama skybox texture ResRefs already exist in this project: " + ", ".join(collisions)
            )
        size = int(face_size)
        if size not in {256, 512, 1_024, 2_048}:
            raise ValueError("Map Studio skybox faces must use 256, 512, 1024, or 2048 pixels.")
        mapper = str(tone_mapper or "aces").strip().lower()
        if mapper not in {"aces", "reinhard"}:
            raise ValueError("Map Studio panorama tone mapping must use ACES or Reinhard.")

        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Skybox panorama does not exist: {source}")
        stat_before = source.stat()
        conversion = load_and_convert_equirectangular_panorama(
            source,
            options=PanoramaSkyboxOptions(
                face_size=size,
                exposure_ev=float(exposure_ev),
                longitude_offset_degrees=float(longitude_offset_degrees),
                tone_mapper=mapper,
            ),
        )
        stat_after_conversion = source.stat()
        source_bytes = source.read_bytes()
        stat_after_hash = source.stat()
        stat_key = lambda value: (int(value.st_size), int(value.st_mtime_ns))
        if not (
            stat_key(stat_before) == stat_key(stat_after_conversion) == stat_key(stat_after_hash)
        ):
            raise ValueError("The panorama changed while it was being converted. Run the conversion again.")
        if tuple(face.name for face in conversion.faces) != tuple(SKYBOX_PANORAMA_FACE_ORDER):
            raise ValueError("Panorama conversion returned an unexpected skybox face order.")
        if self._map_studio_authored_state_token() != authored_state_token:
            raise ValueError("Authored rooms changed while the panorama was being converted. Run the conversion again.")
        return MapStudioPanoramaSkyboxPreparation(
            source_path=str(source),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            project_path=project_path,
            project_texture_resrefs=existing_texture_resrefs,
            authored_state_token=authored_state_token,
            room_resref=room_name,
            face_texture_resrefs=clean_rows,
            conversion=conversion,
        )

    def _append_authored_five_face_skybox(
        self,
        *,
        room_resref: str,
        north_texture: str,
        east_texture: str,
        south_texture: str,
        west_texture: str,
        top_texture: str,
        half_extent: float,
        bottom_z: float,
        top_z: float,
        visible_rooms: tuple[str, ...],
        authoring_metadata: Mapping[str, Any] | None = None,
    ):
        authored = self._load_authored_project_or_raise()
        room_name = normalise_resref(room_resref)
        if any(room.normalised_resref() == room_name for room in authored.rooms):
            raise ValueError(f"Authored room {room_name or '(missing)'} already exists.")
        targets = tuple(
            dict.fromkeys(
                normalise_resref(value)
                for value in (visible_rooms or tuple(room.normalised_resref() for room in authored.rooms))
                if normalise_resref(value)
            )
        )
        sky_room = build_five_face_skybox_room(
            FiveFaceSkyboxSpec(
                room_resref=room_name,
                textures=FiveFaceSkyboxTextures(
                    north=north_texture,
                    east=east_texture,
                    south=south_texture,
                    west=west_texture,
                    top=top_texture,
                ),
                half_extent=float(half_extent),
                bottom_z=float(bottom_z),
                top_z=float(top_z),
                visible_rooms=targets,
                game=authored.game,
                metadata=dict(authoring_metadata or {}),
            )
        )
        updated = replace(authored, rooms=tuple(authored.rooms) + (sky_room,))
        self._store_authored_project(updated)
        return sky_room, targets

    def create_authored_five_face_skybox(
        self,
        *,
        room_resref: str,
        north_texture: str,
        east_texture: str,
        south_texture: str,
        west_texture: str,
        top_texture: str,
        half_extent: float = 500.0,
        bottom_z: float = -500.0,
        top_z: float = 500.0,
        visible_rooms: tuple[str, ...] = (),
        authoring_metadata: Mapping[str, Any] | None = None,
    ):
        """Append one undoable vanilla-style visual sky room to the KMAP."""

        before = self._capture_map_studio_command_state()
        sky_room, targets = self._append_authored_five_face_skybox(
            room_resref=room_resref,
            north_texture=north_texture,
            east_texture=east_texture,
            south_texture=south_texture,
            west_texture=west_texture,
            top_texture=top_texture,
            half_extent=half_extent,
            bottom_z=bottom_z,
            top_z=top_z,
            visible_rooms=visible_rooms,
            authoring_metadata=authoring_metadata,
        )
        room_name = sky_room.normalised_resref()
        self._record_map_studio_command(
            action_key="map_studio.environment.create_skybox",
            label=f"Create skybox room {room_name}",
            before=before,
            metadata={
                "room_resref": room_name,
                "textures": [north_texture, east_texture, south_texture, west_texture, top_texture],
                "visible_rooms": list(targets),
                "projection": "five_face_box",
            },
        )
        message = (
            f"Created five-face skybox room {room_name}; MDL/MDX/LYT/VIS/export proof are now stale."
        )
        self.model.log(message)
        return sky_room, message

    def commit_panorama_skybox(
        self,
        preparation: MapStudioPanoramaSkyboxPreparation,
        *,
        half_extent: float = 500.0,
        bottom_z: float = -500.0,
        top_z: float = 500.0,
        visible_rooms: tuple[str, ...] = (),
    ) -> MapStudioPanoramaSkyboxResult:
        """Atomically create five TGA assets and their visual-only sky room."""

        project_path = str(getattr(self.project, "path", "") or "").strip()
        if not project_path:
            raise ValueError("Save the KMAP before committing panorama skybox textures.")
        if Path(project_path).resolve() != Path(str(preparation.project_path or "")).resolve():
            raise ValueError("The active KMAP changed after panorama conversion. Convert the panorama again.")
        if self._map_studio_authored_state_token() != tuple(preparation.authored_state_token):
            raise ValueError("Authored rooms changed after panorama conversion. Convert the panorama again.")
        current_texture_resrefs = tuple(
            sorted(
                {
                    str(getattr(texture, "resref", "") or "").strip().lower()
                    for texture in tuple(getattr(self.project, "textures", ()) or ())
                    if str(getattr(texture, "resref", "") or "").strip()
                }
            )
        )
        if current_texture_resrefs != tuple(preparation.project_texture_resrefs):
            raise ValueError("Project textures changed after panorama conversion. Convert the panorama again.")
        if tuple(face for face, _resref in preparation.face_texture_resrefs) != tuple(SKYBOX_PANORAMA_FACE_ORDER):
            raise ValueError("Prepared panorama skybox faces are incomplete or out of order.")
        texture_resrefs = tuple(resref for _face, resref in preparation.face_texture_resrefs)
        if set(texture_resrefs) & set(current_texture_resrefs):
            raise ValueError("A prepared panorama skybox texture ResRef now collides with this project.")

        conversion = preparation.conversion
        target_dir = project_texture_directory(self.project)
        sidecar_paths = tuple(
            path
            for resref in texture_resrefs
            for path in (target_dir / f"{resref}.tga", target_dir / f"{resref}.txi")
        )
        before = self._capture_map_studio_command_state()
        sidecar_before = self.texture_sidecar_journal.capture(self.project, paths=sidecar_paths)
        created_paths = tuple(value for value, state in sidecar_before.states if state is None)
        common_metadata = {
            "panorama_source_path": str(preparation.source_path),
            "panorama_source_sha256": str(preparation.source_sha256),
            "panorama_source_width": int(conversion.source_width),
            "panorama_source_height": int(conversion.source_height),
            "panorama_source_is_hdr": bool(conversion.source_is_hdr),
            "panorama_face_size": int(conversion.face_size),
            "panorama_exposure_ev": float(conversion.exposure_ev),
            "panorama_longitude_offset_degrees": float(conversion.longitude_offset_degrees),
            "panorama_tone_mapper": str(conversion.tone_mapper),
            "tone_mapped_for_kotor": True,
            "engine_game_proof": False,
        }
        try:
            for face_name, texture_resref in preparation.face_texture_resrefs:
                face = conversion.face(face_name)
                create_project_texture_asset(
                    self.project,
                    resref=texture_resref,
                    width=int(face.width),
                    height=int(face.height),
                    rgba=bytes(face.rgba),
                    source="map_studio:panorama_skybox",
                    metadata={**common_metadata, "skybox_face": str(face_name)},
                    asset_kind="map_studio_panorama_skybox",
                )
            texture_map = dict(preparation.face_texture_resrefs)
            sky_room, targets = self._append_authored_five_face_skybox(
                room_resref=preparation.room_resref,
                north_texture=texture_map["north"],
                east_texture=texture_map["east"],
                south_texture=texture_map["south"],
                west_texture=texture_map["west"],
                top_texture=texture_map["top"],
                half_extent=half_extent,
                bottom_z=bottom_z,
                top_z=top_z,
                visible_rooms=visible_rooms,
                authoring_metadata={
                    **common_metadata,
                    "panorama_projection": "equirectangular_to_five_face_box",
                },
            )
        except Exception:
            rollback = self.texture_sidecar_journal.finish(
                self.project,
                sidecar_before,
                paths=sidecar_paths,
                created_paths=created_paths,
            )
            self.texture_sidecar_journal.apply(self.project, rollback, use_after=False)
            self._restore_map_studio_command_state_without_history(before)
            raise
        sidecar_patches = self.texture_sidecar_journal.finish(
            self.project,
            sidecar_before,
            paths=sidecar_paths,
            created_paths=created_paths,
        )
        room_name = sky_room.normalised_resref()
        message = (
            f"Converted {Path(preparation.source_path).name} into five {int(conversion.face_size)} px KOTOR TGA "
            f"panels and created skybox room {room_name}; manual retail KOTOR visual proof is still required."
        )
        self.model.log(message)
        self._record_map_studio_command(
            action_key="map_studio.environment.create_panorama_skybox",
            label=f"Create panorama skybox {room_name}",
            before=before,
            stale_outputs=("MDL", "MDX", "LYT", "VIS", "TGA", ".mod"),
            readiness_impact=(
                "The tone-mapped skybox textures and visual room must be repackaged and visually proven in retail KOTOR."
            ),
            metadata={
                **common_metadata,
                "room_resref": room_name,
                "texture_resrefs": list(texture_resrefs),
                "visible_rooms": list(targets),
                "projection": "equirectangular_to_five_face_box",
            },
            sidecar_patches=sidecar_patches,
        )
        return MapStudioPanoramaSkyboxResult(
            room_resref=room_name,
            texture_resrefs=texture_resrefs,
            source_is_hdr=bool(conversion.source_is_hdr),
            face_size=int(conversion.face_size),
            message=message,
        )

    def authored_sky_traffic(self):
        """Return normalized room-animation traffic intent stored in KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("sky_traffic",), read_authored_project_sky_traffic
        )

    def create_authored_sky_traffic(
        self,
        *,
        room_resref: str,
        model_resref: str,
        start: Any,
        end: Any,
        name: str = "Sky Traffic",
        animation_name: str = "animloop1",
        duration_seconds: float | None = 30.0,
        speed_units_per_second: float | None = None,
        facing_mode: str = "path_tangent",
        closed_path: bool = False,
    ):
        """Store one Unreal-like sky actor that targets a room MDL animation."""

        from dataclasses import replace as _replace

        authored = self._load_authored_project_or_raise()
        before = self._capture_map_studio_command_state()
        traffic = create_sky_traffic_actor(
            room_resref=room_resref,
            model_resref=model_resref,
            control_points=(start, end),
            name=name,
            animation_name=animation_name,
            duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
            speed_units_per_second=(
                float(speed_units_per_second) if speed_units_per_second is not None else None
            ),
            facing_mode=facing_mode,
            closed_path=bool(closed_path),
        )
        current = read_authored_project_sky_traffic(authored)
        traffic_rows = tuple(current) + (traffic,)
        validation = validate_authored_sky_traffic_collection(
            traffic_rows,
            room_resrefs={room.normalised_resref() for room in authored.rooms},
        )
        if not validation.ok:
            raise ValueError("; ".join(validation.blocking_issues))
        updated = write_authored_project_sky_traffic(authored, traffic_rows)
        updated_extra = dict(getattr(updated, "extra", {}) or {})
        updated_extra["sky_traffic_compiler_target"] = "room_mdl_animation"
        updated = _replace(updated, extra=updated_extra)
        self._store_authored_project(updated)
        self._record_map_studio_command(
            action_key="map_studio.environment.create_sky_traffic",
            label=f"Create sky traffic {traffic.name}",
            before=before,
            metadata={
                "traffic_id": traffic.traffic_id,
                "room_resref": traffic.room_resref,
                "model_resref": traffic.model_resref,
                "animation_name": traffic.animation_name,
                "duration_seconds": traffic.duration_seconds,
                "speed_units_per_second": traffic.speed_units_per_second,
                "compiler_target": traffic.compiler_target,
            },
        )
        message = (
            f"Created sky traffic {traffic.name} using {traffic.model_resref}; it previews as a model/path actor and "
            f"targets {traffic.room_resref}.{traffic.animation_name} room animation export."
        )
        self.model.log(message)
        return traffic, message, validation

    def authored_sky_traffic_marker_geometry(self) -> AuthoredGameplayMarkerGeometry:
        """Build cyan world-space paths and direction arrows for the viewport."""

        from .authored_gameplay_marker_geometry import AuthoredGameplayMarkerLine

        lines = []
        warnings: list[str] = []
        traffic_rows = tuple(self.authored_sky_traffic() or ())
        for traffic in traffic_rows:
            preview = build_sky_traffic_preview(traffic, path_sample_count=33, arrow_count=8)
            marker_id = f"sky_traffic:{traffic.traffic_id}"
            for index in range(max(0, len(preview.path_points) - 1)):
                lines.append(
                    AuthoredGameplayMarkerLine(
                        placement_id=marker_id,
                        kind="sky_traffic",
                        label=traffic.name,
                        start=preview.path_points[index],
                        end=preview.path_points[index + 1],
                        color="#55d8ff",
                        role="sky_traffic_path",
                    )
                )
            arrow_length = max(1.0, min(12.0, preview.path_length * 0.04))
            for arrow in preview.arrows:
                direction = arrow.facing_direction or arrow.travel_direction
                lines.append(
                    AuthoredGameplayMarkerLine(
                        placement_id=marker_id,
                        kind="sky_traffic",
                        label=traffic.name,
                        start=arrow.position,
                        end=tuple(
                            float(arrow.position[axis]) + (float(direction[axis]) * arrow_length)
                            for axis in range(3)
                        ),
                        color="#55d8ff",
                        role="sky_traffic_direction",
                    )
                )
            if not traffic.enabled:
                warnings.append(f"{traffic.name}: disabled sky traffic is shown for editing but will not compile.")
        return AuthoredGameplayMarkerGeometry(
            marker_count=len(traffic_rows),
            lines=tuple(lines),
            warnings=tuple(warnings),
        )

    def authored_sky_traffic_preview_rows(self):
        """Return viewport-only direct-model rows; these never enter GIT."""

        from types import SimpleNamespace
        import math as _math

        rows = []
        for traffic in tuple(self.authored_sky_traffic() or ()):
            if not traffic.enabled:
                continue
            sample = sample_sky_traffic(traffic, 0.0)
            facing = sample.facing_direction or sample.travel_direction
            rows.append(
                SimpleNamespace(
                    placement_id=f"sky_traffic:{traffic.traffic_id}",
                    kind="sky_traffic",
                    template_resref=traffic.model_resref,
                    model_resref=traffic.model_resref,
                    position=sample.position,
                    bearing=_math.atan2(float(facing[1]), float(facing[0])),
                    is_spatial=True,
                )
            )
        return tuple(rows)

    def authored_script_hook_field_choices(self):
        """Return editable ARE/IFO script hook fields for Map Studio controls."""

        return authored_script_hook_field_choices()

    def authored_script_hooks(self):
        """Return current authored script hooks for the current KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return {"area": {}, "module": {}}
        return self._map_studio_cached_authored_query(("script_hooks",), authored_script_hooks)

    def authored_gameplay_preview_markers(self):
        """Return UI-ready preview markers for authored gameplay placements."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(("gameplay_preview_markers",), authored_gameplay_preview_markers)

    def authored_gameplay_marker_geometry(self):
        """Return renderer-ready geometry for authored gameplay placement markers."""

        if self._map_studio_authored_project_snapshot() is None:
            return AuthoredGameplayMarkerGeometry()
        return self._map_studio_cached_authored_query(
            ("gameplay_marker_geometry",), authored_gameplay_marker_geometry_for_project
        )

    def authored_gameplay_fallback_preview_markers(self):
        """Return the IFO player start plus unresolved GIT model fallbacks."""

        resolved = set(tuple(self.last_map_studio_resolved_placement_ids or ()))
        placement_markers = tuple(
            marker
            for marker in self.authored_gameplay_preview_markers()
            if str(getattr(marker, "placement_id", "") or "") not in resolved
        )
        if getattr(self, "model", None) is None:
            return placement_markers
        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return placement_markers
        entry_marker = authored_module_entry_point_preview_marker(authored)
        if entry_marker is None or "entry_point" in resolved:
            return placement_markers
        return (entry_marker,) + placement_markers

    def authored_gameplay_fallback_marker_geometry(self):
        """Renderer geometry for honest unresolved-model marker fallbacks."""

        gameplay = authored_gameplay_marker_geometry(self.authored_gameplay_fallback_preview_markers())
        traffic = self.authored_sky_traffic_marker_geometry()
        return combine_marker_geometry(gameplay, traffic)

    def map_studio_spatial_design(self) -> SpatialDesignPlan | None:
        """Return the purpose-led grid plan stored with the current KMAP."""

        authored = self._map_studio_authored_project_snapshot()
        return read_authored_spatial_design(authored) if authored is not None else None

    def set_map_studio_spatial_design(self, plan: SpatialDesignPlan | None):
        """Persist a spatial plan and return its immediately useful audit."""

        authored = self._load_authored_project_or_raise()
        self._store_authored_project(write_authored_spatial_design(authored, plan))
        return audit_spatial_design(plan)

    def audit_map_studio_spatial_design(self):
        """Audit zones, routes, clearances, grid alignment, and placement intent."""

        return audit_spatial_design(self.map_studio_spatial_design())

    def map_studio_spatial_design_placement_ledger(self):
        """Return exact coordinates and rationale rows for Map Studio UI/proofs."""

        return spatial_design_placement_ledger(self.map_studio_spatial_design())

    def map_studio_spatial_design_marker_geometry(self):
        """Return world-space zone, route, and placement guides for the viewport."""

        return spatial_design_marker_geometry(self.map_studio_spatial_design())

    def authored_editor_marker_geometry(self, *, include_spatial_design: bool = True):
        """Compose ordinary authored markers with the optional spatial plan."""

        gameplay = self.authored_gameplay_fallback_marker_geometry()
        spatial = (
            self.map_studio_spatial_design_marker_geometry()
            if bool(include_spatial_design)
            else AuthoredGameplayMarkerGeometry()
        )
        return combine_marker_geometry(gameplay, spatial)

    def authored_room_outline_geometry(self):
        """Return renderer-ready outlines for authored Map Studio rooms."""

        if self._map_studio_authored_project_snapshot() is None:
            return AuthoredRoomOutlineGeometry()
        return self._map_studio_cached_authored_query(
            ("room_outline_geometry",), authored_room_outline_geometry_for_project
        )

    def authored_room_preview_model(self, *, include_backdrops: bool = False):
        """Return a live KotorModel preview for authored Map Studio render geometry."""

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return None
        # Combined preview composition appends stock groups to this model in
        # place. Cache the final combined model, never this mutable base model,
        # or a resource revision would append a second copy of every instance.
        return build_authored_module_preview_model(
            authored, include_backdrops=include_backdrops
        ).model

    def _map_studio_combined_preview_state_signature(self) -> tuple[Any, ...]:
        """Return an O(room-count) preview key without serializing KMAP.

        Payload identity covers normal immutable replacement edits; the
        explicit authored-state revision covers nested in-place proof/PTH/
        texture mutations.  Only lightweight legacy KMAP room transforms are
        enumerated because those live outside the authored payload.
        """

        room_rows = []
        for room in tuple(getattr(self.project, "rooms", ()) or ()):
            transform = getattr(room, "transform", None)

            def _components(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
                value = tuple(getattr(transform, name, default) or default)
                try:
                    return tuple(float(value[index]) for index in range(3))
                except (IndexError, TypeError, ValueError):
                    return default

            room_rows.append(
                (
                    str(getattr(room, "room_id", "") or ""),
                    str(getattr(room, "model_resref", "") or getattr(room, "name", "") or ""),
                    bool(getattr(room, "visible", True)),
                    _components("position", (0.0, 0.0, 0.0)),
                    _components("rotation", (0.0, 0.0, 0.0)),
                    _components("scale", (1.0, 1.0, 1.0)),
                )
            )
        return (self._map_studio_authored_state_token(), tuple(room_rows))

    def map_studio_viewport_preview_model(self, resource_manager=None, *, include_backdrops: bool = False):
        """Return the merged viewport preview: authored rooms plus stock KOTOR content.

        Stock LYT rooms (``project.rooms``) and spatial gameplay placements
        (creature/placeable/door template resrefs) render real game geometry
        when a game resource manager is available.  Without one, the authored
        preview is returned unchanged so headless flows keep working.
        """

        started = perf_counter()
        game = str(getattr(self.project, "game", "K1") or "K1").upper()
        entry_point_row = self.authored_module_entry_point_preview_row()
        entry_point_signature = (
            str(getattr(entry_point_row, "model_resref", "") or ""),
            str(getattr(entry_point_row, "head_model_resref", "") or ""),
            tuple(getattr(entry_point_row, "position", ()) or ()),
            float(getattr(entry_point_row, "bearing", 0.0) or 0.0),
        ) if entry_point_row is not None else ()
        cache_key = (
            id(resource_manager),
            int(getattr(resource_manager, "revision", 0) or 0) if resource_manager is not None else 0,
            game,
            bool(include_backdrops),
            int(self._authored_placeable_preview_revision),
            self._map_studio_combined_preview_state_signature(),
            entry_point_signature,
        )
        cached = self._map_studio_combined_preview_cache
        if resource_manager is not None and cached is not None and cached[0] == cache_key:
            model, stock_result = cached[1], cached[2]
            self.last_map_studio_stock_preview_warnings = tuple(stock_result.warnings)
            self.last_map_studio_resolved_placement_ids = tuple(stock_result.resolved_placement_ids)
            self.last_map_studio_unresolved_placement_ids = tuple(stock_result.unresolved_placement_ids)
            self.last_map_studio_preview_cache_hit = True
            self.last_map_studio_preview_elapsed_ms = (perf_counter() - started) * 1000.0
            return model

        self.last_map_studio_preview_cache_hit = False
        authored = self.authored_room_preview_model(include_backdrops=include_backdrops)
        # Authored-room previews already contain both generated Pascal geometry
        # and converted stock-room surfaces.  The synchronized KMAP room table
        # carries those same resrefs for LYT/VIS/export ownership; feeding them
        # back through the stock preview loader draws a second complete copy on
        # top of the authored one (z-fighting, apparent wall clipping, and
        # hundreds of avoidable PIE submissions).
        try:
            authored_room_resrefs = {
                room.normalised_resref()
                for room in self._map_studio_authored_project_snapshot().rooms
            }
        except Exception:
            authored_room_resrefs = set()

        def _preview_room_resref(room: object) -> str:
            return normalise_resref(
                getattr(room, "model_resref", "")
                or getattr(room, "name", "")
                or getattr(room, "room_id", "")
            )

        rooms = tuple(
            room
            for room in tuple(getattr(self.project, "rooms", ()) or ())
            if _preview_room_resref(room) not in authored_room_resrefs
        )
        placements = ((entry_point_row,) if entry_point_row is not None else ()) + tuple(
            self.authored_gameplay_placements() or ()
        ) + tuple(
            self.authored_sky_traffic_preview_rows() or ()
        )
        self.last_map_studio_stock_preview_warnings = ()
        self.last_map_studio_resolved_placement_ids = ()
        self.last_map_studio_unresolved_placement_ids = ()
        if resource_manager is None or (not rooms and not placements):
            self.last_map_studio_preview_elapsed_ms = (perf_counter() - started) * 1000.0
            return authored
        template_resources = self._map_studio_template_resources()
        template_resource_signature = tuple(
            (resref, restype, hashlib.sha1(bytes(data)).hexdigest())
            for resref, restype, data in template_resources
        )
        resolver = getattr(self, "_map_studio_stock_template_resolver", None)
        if (
            resolver is None
            or getattr(resolver, "_game", "") != game
            or getattr(resolver, "_manager", None) is not resource_manager
            or getattr(resolver, "_ghostrigger_template_resource_signature", ()) != template_resource_signature
        ):
            resolver = TemplateModelResolver(
                resource_manager,
                game,
                template_resources=template_resources,
                placeable_rows=self._authored_placeable_preview_rows,
            )
            setattr(resolver, "_ghostrigger_template_resource_signature", template_resource_signature)
            self._map_studio_stock_template_resolver = resolver
        cache = getattr(self, "_map_studio_stock_model_cache", None)
        if cache is None:
            cache = {}
            self._map_studio_stock_model_cache = cache

        def _loader(resref, _game=game, _manager=resource_manager, _cache=cache):
            key = (_game, str(resref or "").strip().lower())
            if key not in _cache:
                override = resolver.model_resource_bytes(key[1])
                if override is not None:
                    _cache[key] = load_kotor_model_from_bytes(*override, resref=key[1])
                else:
                    _cache[key] = load_stock_kotor_model(_manager, key[1], _game)
            return _cache[key]
        model, stock_result = build_map_studio_combined_preview_model(
            authored_model=authored,
            project_name=str(getattr(self.project, "name", "") or "map_studio_preview"),
            game=game,
            rooms=rooms,
            placements=placements,
            resource_manager=resource_manager,
            model_loader=_loader,
            resolver=resolver,
            resource_revision=self._authored_placeable_preview_revision,
        )
        self.last_map_studio_stock_preview_warnings = tuple(stock_result.warnings)
        self.last_map_studio_resolved_placement_ids = tuple(stock_result.resolved_placement_ids)
        self.last_map_studio_unresolved_placement_ids = tuple(stock_result.unresolved_placement_ids)
        self._map_studio_combined_preview_cache = (cache_key, model, stock_result)
        self.last_map_studio_preview_elapsed_ms = (perf_counter() - started) * 1000.0
        return model

    def map_studio_palette_preview_model(self, entry: object, resource_manager=None):
        """Resolve one Content Browser tile to its real KOTOR preview model."""

        def value(key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        manager = resource_manager
        kind = str(value("kind", "") or "").strip().lower()
        template_resref = str(value("template_resref", "") or "").strip().lower()
        if manager is None or kind not in {"creature", "placeable", "door"} or not template_resref:
            return None, ""
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        entry_game = str(value("game", "") or "").strip().upper()
        if entry_game and entry_game != game:
            return None, ""
        template_resources = self._map_studio_template_resources()
        template_resource_signature = tuple(
            (resref, restype, hashlib.sha1(bytes(data)).hexdigest())
            for resref, restype, data in template_resources
        )
        resolver = getattr(self, "_map_studio_stock_template_resolver", None)
        if (
            resolver is None
            or getattr(resolver, "_game", "") != game
            or getattr(resolver, "_manager", None) is not manager
            or getattr(resolver, "_ghostrigger_template_resource_signature", ()) != template_resource_signature
        ):
            resolver = TemplateModelResolver(
                manager,
                game,
                template_resources=template_resources,
                placeable_rows=self._authored_placeable_preview_rows,
            )
            setattr(resolver, "_ghostrigger_template_resource_signature", template_resource_signature)
            self._map_studio_stock_template_resolver = resolver
        model_resref = str(resolver.model_for_placement_kind(kind, template_resref) or "").strip().lower()
        if not model_resref:
            return None, ""
        cache = getattr(self, "_map_studio_stock_model_cache", None)
        if cache is None:
            cache = {}
            self._map_studio_stock_model_cache = cache
        key = (game, model_resref)
        if key not in cache:
            override = resolver.model_resource_bytes(model_resref)
            if override is not None:
                cache[key] = load_kotor_model_from_bytes(*override, resref=model_resref)
            else:
                cache[key] = load_stock_kotor_model(manager, model_resref, game)
        return cache[key], model_resref

    def map_studio_pie_context_settings(self) -> dict[str, Any]:
        """Return the normalized, preview-only dialogue sandbox stored in KMAP."""

        raw = dict((getattr(self.project, "extra_sections", {}) or {}).get("map_studio_pie_context") or {})
        role = str(raw.get("player_role") or "normal_pc").strip().lower().replace("-", "_")
        if role not in {"normal_pc", "b4d4"}:
            role = "normal_pc"
        gender = str(raw.get("player_gender") or "male").strip().lower()
        if gender not in {"male", "female"}:
            gender = "male"

        global_numbers: dict[str, int] = {}
        for key, value in dict(raw.get("global_numbers") or {}).items():
            name = str(key or "").strip()
            if not name:
                continue
            try:
                global_numbers[name] = int(value)
            except (TypeError, ValueError):
                continue
        global_booleans = {
            str(key).strip(): bool(value)
            for key, value in dict(raw.get("global_booleans") or {}).items()
            if str(key or "").strip()
        }
        global_strings = {
            str(key).strip(): str(value)
            for key, value in dict(raw.get("global_strings") or {}).items()
            if str(key or "").strip()
        }
        # Preview player build: a UTC resref whose combat stats stand in for the
        # PC in RTwP combat (the PC is a custom campaign build, so this is a
        # configurable sandbox value; empty keeps the editor proxy).
        player_combat_template = str(raw.get("player_combat_template") or "").strip().lower()[:16]
        local_booleans = {
            str(key).strip(): bool(value)
            for key, value in dict(raw.get("local_booleans") or {}).items()
            if str(key or "").strip()
        }
        dialogue_start_overrides: dict[str, dict[str, str]] = {}
        for resref, value in dict(raw.get("dialogue_start_overrides") or {}).items():
            clean_resref = str(resref or "").strip().lower()[:16]
            if not clean_resref or not isinstance(value, Mapping):
                continue
            starter_link_id = str(value.get("starter_link_id") or "").strip().lower()
            resource_sha256 = str(value.get("resource_sha256") or "").strip().lower()
            if starter_link_id:
                dialogue_start_overrides[clean_resref] = {
                    "starter_link_id": starter_link_id,
                    "resource_sha256": resource_sha256,
                }
        # Preview party roster: up to two companion UTC resrefs the creator wants
        # trailing the player in PIE (party membership is campaign state, not
        # module data, so it is a configurable sandbox value).
        party_roster: list[str] = []
        seen_party: set[str] = set()
        for value in tuple(raw.get("party_roster") or ()):
            resref = str(value or "").strip().lower()[:16]
            if resref and resref not in seen_party:
                seen_party.add(resref)
                party_roster.append(resref)
            if len(party_roster) >= 2:
                break
        return {
            "player_role": role,
            "player_gender": gender,
            "global_numbers": global_numbers,
            "global_booleans": global_booleans,
            "global_strings": global_strings,
            "local_booleans": local_booleans,
            "dialogue_start_overrides": dialogue_start_overrides,
            "party_roster": tuple(party_roster),
            "player_combat_template": player_combat_template,
        }

    def update_map_studio_pie_context(self, **changes: Any) -> dict[str, Any]:
        """Persist preview context without invalidating authored export geometry."""

        current = self.map_studio_pie_context_settings()
        current.update(changes)
        # Normalize through the same read contract before committing the section.
        prior_section = (getattr(self.project, "extra_sections", {}) or {}).get("map_studio_pie_context")
        if not isinstance(getattr(self.project, "extra_sections", None), dict):
            self.project.extra_sections = {}
        self.project.extra_sections["map_studio_pie_context"] = current
        normalized = self.map_studio_pie_context_settings()
        self.project.extra_sections["map_studio_pie_context"] = normalized
        if prior_section != normalized:
            self.project.mark_dirty()
        return deepcopy(normalized)

    def update_map_studio_pie_context_settings(self, **changes: Any) -> dict[str, Any]:
        """Compatibility spelling used by the compact PIE context panel."""

        return self.update_map_studio_pie_context(**changes)

    def _map_studio_pie_resource_context(self) -> SimpleNamespace:
        """Build shared template, DLG, item, and TLK readers for PIE surfaces."""

        resolver = getattr(self, "_map_studio_stock_template_resolver", None)
        game = str(getattr(self.project, "game", "K1") or "K1").strip().upper()
        resource_types = {
            "creature": RES_UTC,
            "door": RES_UTD,
            "placeable": RES_UTP,
            "trigger": RES_UTT,
            "store": RES_UTM,
            "item": RES_UTI,
        }
        reader = getattr(resolver, "_template_bytes", None) if resolver is not None else None

        def _tlk_lookup(strref: int) -> str:
            if resolver is None:
                return ""
            manager = getattr(resolver, "_manager", None)
            game_tag = str(getattr(resolver, "_game", game) or game).strip().upper()
            revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else 0
            cache = getattr(self, "_map_studio_pie_tlk_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                self._map_studio_pie_tlk_cache = cache
            key = (id(manager), revision, game_tag)
            if key not in cache:
                talk_table = None
                installation_getter = getattr(manager, "get_k2" if game_tag == "K2" else "get_k1", None)
                installation = installation_getter() if callable(installation_getter) else None
                game_dir = str(getattr(installation, "game_dir", "") or "")
                tlk_path = Path(game_dir) / "dialog.tlk" if game_dir else None
                if tlk_path is not None and tlk_path.is_file():
                    try:
                        from pykotor.resource.formats.tlk import read_tlk

                        talk_table = read_tlk(tlk_path)
                    except Exception:
                        talk_table = None
                cache.clear()
                cache[key] = talk_table
            talk_table = cache.get(key)
            if talk_table is None:
                return ""
            try:
                entry = talk_table.get(int(strref))
                return str(getattr(entry, "text", entry) or "")
            except Exception:
                return ""

        def _resource_bytes(kind: str, resref: str) -> bytes | None:
            restype = resource_types.get(str(kind or "").strip().lower())
            if restype is None or not callable(reader):
                return None
            raw = reader(resref, restype)
            return bytes(raw) if raw else None

        weapon_damage_resolver = getattr(resolver, "weapon_damage_dice", None) if resolver is not None else None
        armor_ac_resolver = getattr(resolver, "armor_class_bonus", None) if resolver is not None else None
        weapon_critical_resolver = getattr(resolver, "weapon_critical", None) if resolver is not None else None
        weapon_damage_type_resolver = getattr(resolver, "weapon_damage_type", None) if resolver is not None else None
        weapon_feat_category_resolver = getattr(resolver, "weapon_feat_category", None) if resolver is not None else None

        def _template_inspector(kind: str, resref: str) -> dict[str, Any]:
            raw = _resource_bytes(kind, resref)
            if raw is None:
                raise FileNotFoundError(f"{str(kind or 'template').upper()} {resref} was unavailable for PIE inspection")
            try:
                return inspect_map_studio_pie_resource(
                    kind,
                    resref,
                    raw,
                    tlk_lookup=_tlk_lookup,
                    weapon_damage_resolver=weapon_damage_resolver if kind == "creature" else None,
                    armor_ac_resolver=armor_ac_resolver if kind == "creature" else None,
                    weapon_critical_resolver=weapon_critical_resolver if kind == "creature" else None,
                    weapon_damage_type_resolver=weapon_damage_type_resolver if kind == "creature" else None,
                    weapon_feat_category_resolver=weapon_feat_category_resolver if kind == "creature" else None,
                )
            except Exception as exc:
                raise ValueError(f"{kind} {resref} could not be decoded for PIE: {exc}") from exc

        def _dialogue_loader(resref: str) -> bytes | None:
            staged = self.authored_scripting_resource(resref, "dlg")
            if staged:
                return bytes(staged)
            if not callable(reader):
                return None
            raw = reader(resref, RES_DLG)
            return bytes(raw) if raw else None

        def _item_inspector(resref: str) -> dict[str, Any]:
            return _template_inspector("item", resref)

        return SimpleNamespace(
            resolver=resolver,
            game=game,
            template_inspector=_template_inspector,
            dialogue_loader=_dialogue_loader,
            item_inspector=_item_inspector,
            tlk_lookup=_tlk_lookup,
        )

    def map_studio_pie_dialogue_catalog(self):
        """Return conversations actually referenced by the loaded module entities.

        A standalone LYT has rooms and door hooks but no GIT templates or DLG
        resources, so it intentionally returns an empty catalog until a module
        source is attached or the project is hydrated into an authored KMAP.
        """

        authored = self._map_studio_authored_project_snapshot()
        if authored is None:
            return ()
        context = self._map_studio_pie_resource_context()
        manager = getattr(context.resolver, "_manager", None) if context.resolver is not None else None
        manager_revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else 0
        cache_key = (
            "pie_dialogue_catalog",
            id(manager),
            manager_revision,
            int(self._authored_placeable_preview_revision),
            int(self._map_studio_pie_context_resource_revision),
        )

        def _build(source):
            registry = build_pie_entity_registry(source, template_inspector=context.template_inspector)
            source_path = str(
                dict(getattr(source, "extra", {}) or {}).get("import_source")
                or getattr(self.project, "path", "")
                or ""
            ).strip()
            source_name = Path(source_path).name if source_path else "loaded KMAP"
            return build_map_studio_pie_dialogue_catalog(
                registry,
                dialogue_loader=context.dialogue_loader,
                game=context.game,
                tlk_lookup=context.tlk_lookup,
                dialogue_source_label=lambda resref: f"{source_name} / {resref}.dlg",
            )

        return self._map_studio_cached_authored_query(cache_key, _build)

    def _map_studio_pie_condition_evaluator(self, settings: dict[str, Any]) -> MapStudioPIEDialogueContextEvaluator:
        """Build the bounded PIE dialogue condition evaluator from context settings.

        Shared by live PIE session construction and the compact panel's opening
        line preview so both evaluate authored Active conditions identically.
        """

        local_booleans: dict[tuple[str, int], bool] = {}
        for key, value in dict(settings.get("local_booleans") or {}).items():
            owner, separator, index = str(key or "").rpartition(":")
            if not separator or not owner:
                continue
            try:
                local_booleans[(owner, int(index))] = bool(value)
            except (TypeError, ValueError):
                continue
        return MapStudioPIEDialogueContextEvaluator(
            player_role=str(settings.get("player_role") or "normal_pc"),
            player_gender=str(settings.get("player_gender") or "male"),
            global_numbers=dict(settings.get("global_numbers") or {}),
            global_booleans=dict(settings.get("global_booleans") or {}),
            global_strings=dict(settings.get("global_strings") or {}),
            local_booleans=local_booleans,
        )

    def map_studio_pie_dialogue_preview(self, resref: str, *, starter_link_id: str = "") -> dict[str, Any]:
        """Resolve the opening NPC line a conversation shows under current PIE context.

        Runs the identical production dialogue classes the live PIE runtime uses
        (the same DLG loader, TLK, and ``MapStudioPIEDialogueContextEvaluator``),
        so the compact PIE panel can display which authored line a conversation
        opens with: canonical ``Auto (clean PIE state)`` selection, or a forced
        preview-override starting link.  It never mutates persisted settings and
        never raises for missing/undecodable resources.
        """

        clean_resref = str(resref or "").strip().lower()
        result: dict[str, Any] = {
            "resref": clean_resref,
            "resolved": False,
            "forced": False,
            "starter_link_id": "",
            "speaker_tag": "",
            "text": "",
            "blocked": False,
            "warning": "",
        }
        if not clean_resref:
            return result
        context = self._map_studio_pie_resource_context()
        loader = getattr(context, "dialogue_loader", None)
        if not callable(loader):
            result["warning"] = "No dialogue loader is available for the current project."
            return result
        try:
            payload = loader(clean_resref)
        except Exception as exc:  # resource IO must never crash the panel preview
            result["warning"] = f"DLG {clean_resref} could not be read: {exc}"
            return result
        if not payload:
            result["warning"] = f"DLG {clean_resref} was not found in the active {context.game} resources."
            return result
        payload = bytes(payload)
        settings = self.map_studio_pie_context_settings()
        forced_link = str(starter_link_id or "").strip().lower()
        if not forced_link:
            override = dict(settings.get("dialogue_start_overrides") or {}).get(clean_resref)
            if isinstance(override, Mapping):
                candidate = str(override.get("starter_link_id") or "").strip().lower()
                expected_sha = str(override.get("resource_sha256") or "").strip().lower()
                actual_sha = hashlib.sha256(payload).hexdigest()
                if candidate and (not expected_sha or expected_sha == actual_sha):
                    forced_link = candidate
        try:
            session = MapStudioPIEDialogueSession(
                payload,
                game=context.game,
                resref=clean_resref,
                owner_id="pie:preview",
                listener_id="pie:player",
                tlk_lookup=context.tlk_lookup,
                condition_evaluator=self._map_studio_pie_condition_evaluator(settings),
                starter_link_id=forced_link,
                allow_unknown_starter_assumption=False,
            )
            snapshot = session.start()
        except Exception as exc:  # malformed DLG must not crash the panel preview
            result["warning"] = f"DLG {clean_resref} could not be previewed: {exc}"
            return result
        result["resolved"] = not bool(getattr(snapshot, "blocked", False))
        result["forced"] = bool(forced_link)
        result["starter_link_id"] = forced_link
        result["speaker_tag"] = str(getattr(snapshot, "speaker_tag", "") or "")
        result["text"] = str(getattr(snapshot, "text", "") or "")
        result["blocked"] = bool(getattr(snapshot, "blocked", False))
        if result["blocked"]:
            warnings = tuple(getattr(snapshot, "warnings", ()) or ())
            result["warning"] = str(warnings[-1]) if warnings else "Dialogue has no valid starting NPC entry."
        return result

    def create_map_studio_pie_session(self, *, preview_model=None, resource_manager=None):
        """Build one read-only PIE session from the current authored snapshot.

        Session construction is intentionally explicit and one-shot: the WOK
        spatial index and room collision BVH are built once when Play starts,
        never during a paint frame.  The returned validation keeps simulation
        capability separate from KOTOR export/game-proof status.

        When ``resource_manager`` is supplied, the module's OnEnter script is read
        for literal global-variable writes (bounded NCS reader) and folded under
        the creator-configured sandbox so dialogue Active conditions see
        script-set globals; explicit creator overrides still win.
        """

        authored = self._load_authored_project_or_raise()
        # The combined WOK is a pure projection of the authored state; on
        # large converted modules recombining it measured ~9 s per Play press
        # (koq201, 9 imported rooms), so reuse the authored-revision cache.
        combined_walkmesh = self._map_studio_cached_authored_query(
            ("pie_combined_walkmesh",), combine_authored_module_walkmesh
        )
        context = self._map_studio_pie_resource_context()
        settings = self.map_studio_pie_context_settings()
        # Execute the OnEnter script so dialogue conditions read script state.
        # The creator sandbox is the PRE-ENTRY campaign state: the NCS virtual
        # machine is seeded with it and runs the script on top (campaign-gated
        # writes fire exactly when their gate state is present, like retail).
        # Only the bounded literal-reader fallback keeps the old merge where
        # creator values win over the reader's unconditional writes.
        self.last_map_studio_pie_scripted_globals = None
        journal_seed: tuple[tuple[str, int], ...] = ()
        if resource_manager is not None:
            scripted = self.map_studio_pie_scripted_globals(
                resource_manager,
                sandbox_numbers=dict(settings.get("global_numbers") or {}),
                sandbox_booleans=dict(settings.get("global_booleans") or {}),
                sandbox_strings=dict(settings.get("global_strings") or {}),
            )
            self.last_map_studio_pie_scripted_globals = scripted
            if str(scripted.get("engine") or "") == "vm":
                settings = {
                    **settings,
                    "global_numbers": dict(scripted.get("global_numbers") or {}),
                    "global_booleans": dict(scripted.get("global_booleans") or {}),
                    "global_strings": dict(scripted.get("global_strings") or {}),
                }
            elif scripted.get("global_numbers") or scripted.get("global_booleans") or scripted.get("global_strings"):
                merged_numbers = dict(scripted.get("global_numbers") or {})
                merged_numbers.update(settings.get("global_numbers") or {})
                merged_booleans = dict(scripted.get("global_booleans") or {})
                merged_booleans.update(settings.get("global_booleans") or {})
                merged_strings = dict(scripted.get("global_strings") or {})
                merged_strings.update(settings.get("global_strings") or {})
                settings = {
                    **settings,
                    "global_numbers": merged_numbers,
                    "global_booleans": merged_booleans,
                    "global_strings": merged_strings,
                }
            # Seed the runtime quest log with the OnEnter script's
            # AddJournalQuestEntry writes (reported, not campaign state).
            journal_seed = tuple(scripted.get("journal") or ())
        # A compiled-NCS loader so a dialogue node's action script can execute
        # its bounded literal global/journal writes into the shared PIE state.
        # Uses the same authored snapshot the OnEnter reader resolves the capsule
        # from, so node scripts load from the imported MOD/RIM + resource manager.
        script_loader = None
        if resource_manager is not None:
            script_authored = self._map_studio_authored_project_snapshot()
            if script_authored is not None:
                def script_loader(resref, _authored=script_authored, _rm=resource_manager):
                    data, _source = self._load_module_ncs_resource(resref, _authored, _rm)
                    return data
        condition_evaluator = self._map_studio_pie_condition_evaluator(settings)
        party_combatants = self._resolve_party_combatants(settings.get("party_roster") or (), context)
        player_combat_stats = self._resolve_player_combat_stats(settings.get("player_combat_template") or "", context)
        result = build_map_studio_pie_session(
            authored,
            preview_model=preview_model,
            combined_walkmesh=combined_walkmesh,
            template_inspector=context.template_inspector,
            dialogue_loader=context.dialogue_loader,
            item_inspector=context.item_inspector,
            tlk_lookup=context.tlk_lookup,
            dialogue_condition_evaluator=condition_evaluator,
            dialogue_start_overrides=dict(settings.get("dialogue_start_overrides") or {}),
            journal_seed=journal_seed,
            script_loader=script_loader,
            party_combatants=party_combatants,
            player_combat_stats=player_combat_stats,
        )
        # Reflect the configured preview party roster in the live follow formation.
        session = getattr(result, "session", None)
        if session is not None and hasattr(session, "set_party_follower_count"):
            session.set_party_follower_count(len(settings.get("party_roster") or ()))
        return result

    def _resolve_party_combatants(self, party_roster: Any, context: Any) -> tuple[dict[str, Any], ...]:
        """Resolve configured party companions to RTwP combat stat rows.

        Each roster UTC is inspected through the same creature projection the
        placed creatures use (HP/AC/attack/damage/crit from the 2DA chain), so a
        companion joins PIE combat as an assisting ally with faithful stats
        rather than a proxy. Defensive: a UTC that cannot be inspected is skipped.
        """

        inspector = getattr(context, "template_inspector", None)
        if not callable(inspector):
            return ()
        rows: list[dict[str, Any]] = []
        for index, resref in enumerate(tuple(party_roster or ())):
            clean = str(resref or "").strip().lower()
            if not clean:
                continue
            try:
                inspected = dict(inspector("creature", clean) or {})
            except Exception:
                continue
            rows.append(
                {
                    "entity_id": f"pie:party:{index}",
                    "display_name": str(inspected.get("display_name") or clean),
                    "max_hp": max(1, int(inspected.get("max_hp", 0) or 0) or 1),
                    "current_hp": max(1, int(inspected.get("current_hp", 0) or inspected.get("max_hp", 0) or 1)),
                    "armor_class": max(1, int(inspected.get("armor_class", 10) or 10)),
                    "attack_bonus": int(inspected.get("attack_bonus", 0) or 0),
                    "damage_min": max(1, int(inspected.get("damage_min", 1) or 1)),
                    "damage_max": max(1, int(inspected.get("damage_max", 6) or 6)),
                    "critical_threat": max(1, int(inspected.get("critical_threat", 1) or 1)),
                    "critical_multiplier": max(1, int(inspected.get("critical_multiplier", 2) or 2)),
                    "damage_type": str(inspected.get("damage_type", "Physical") or "Physical"),
                    "initiative_bonus": int(inspected.get("initiative_bonus", 0) or 0),
                }
            )
        return tuple(rows)

    def _resolve_player_combat_stats(self, template_resref: Any, context: Any):
        """Resolve the configured player-build UTC to PIE combat stats.

        The PC is a custom campaign build, so PIE uses an editor proxy by
        default; when the creator picks a representative UTC the same creature
        projection (HP/AC/attack/damage from the 2DA chain) stands in for the
        player in RTwP combat. Returns ``None`` (keep the proxy) on any failure.
        """

        resref = str(template_resref or "").strip().lower()
        inspector = getattr(context, "template_inspector", None)
        if not resref or not callable(inspector):
            return None
        try:
            inspected = dict(inspector("creature", resref) or {})
        except Exception:
            return None
        try:
            from .map_studio_pie_combat import MapStudioPIECombatStats, MapStudioPIEDamageDice
        except Exception:
            return None
        max_hp = max(1, int(inspected.get("max_hp", 0) or 0) or 1)
        current_hp = max(1, int(inspected.get("current_hp", 0) or inspected.get("max_hp", 0) or 1))
        damage_min = max(1, int(inspected.get("damage_min", 1) or 1))
        damage_max = max(damage_min, int(inspected.get("damage_max", 6) or 6))
        return MapStudioPIECombatStats(
            max_hp=max_hp,
            current_hp=min(current_hp, max_hp),
            armor_class=max(1, int(inspected.get("armor_class", 10) or 10)),
            attack_bonus=int(inspected.get("attack_bonus", 0) or 0),
            damage=MapStudioPIEDamageDice(1, max(1, damage_max - damage_min + 1), damage_min - 1),
            initiative_bonus=int(inspected.get("initiative_bonus", 0) or 0),
            critical_threat=max(1, int(inspected.get("critical_threat", 1) or 1)),
            critical_multiplier=max(1, int(inspected.get("critical_multiplier", 2) or 2)),
            damage_type=str(inspected.get("damage_type", "Physical") or "Physical"),
        )

    def authored_room_primitive_transforms(self):
        """Return editable composition primitive transform rows for the current KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("room_primitive_transforms",), authored_room_composition_primitives
        )

    def authored_room_primitive_universal_transform(self, *, room_resref: str, primitive_name: str):
        """Return exact selected primitive bounds for the Universal Manipulator."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        return authored_room_composition_primitive_universal_transform(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )

    def map_studio_universal_transform_overlay(self, *, room_resref: str, primitive_name: str):
        """Return viewport-ready Universal Manipulator overlay geometry."""

        selection = self.authored_room_primitive_universal_transform(
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        return build_map_studio_universal_transform_overlay(selection)

    def map_studio_export_object_boundaries(self):
        """Return exportable authored room/object boundaries for the current KMAP."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            return ()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        return map_studio_export_object_boundaries(authored)

    def authored_floor_plan_room_choices(self):
        """Return floor-plan rooms that can participate in Builder boolean operations."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("floor_plan_room_choices",), authored_floor_plan_room_choices
        )

    def authored_terrain_room_choices(self):
        """Return terrain rooms that can participate in Builder heightfield operations."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("terrain_room_choices",), authored_terrain_room_choices
        )

    def authored_terrain_walkability_overlay(self):
        """Return renderer-ready terrain WOK walkability feedback."""

        if self._map_studio_authored_project_snapshot() is None:
            return AuthoredTerrainWalkabilityOverlay()
        return self._map_studio_cached_authored_query(
            ("terrain_walkability_overlay",), authored_terrain_walkability_overlay_for_project
        )

    def authored_terrain_status(self):
        """Return modder-facing terrain authoring status for the current KMAP."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            return {
                "ready": False,
                "terrain_room_count": 0,
                "walkable_triangle_count": 0,
                "non_walk_triangle_count": 0,
                "max_slope_degrees": 0.0,
                "summary": "Terrain: no authored Map Studio module is loaded.",
                "next_action": "Create a Terrain Patch, then choose a brush before sculpting.",
                "warnings": (),
                "capability_stage": "previewable_status_query",
            }
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        choices = tuple(authored_terrain_room_choices(authored))
        overlay = authored_terrain_walkability_overlay_for_project(authored)
        ready = bool(choices)
        summary = (
            f"Terrain: {len(choices)} terrain room(s), {int(overlay.walkable_triangle_count)} walkable triangle(s), "
            f"{int(overlay.non_walk_triangle_count)} blocked triangle(s), max slope {float(overlay.max_slope_degrees):.1f} deg."
            if ready
            else "Terrain: no terrain heightfield rooms exist in this authored module."
        )
        return {
            "ready": ready,
            "terrain_room_count": len(choices),
            "walkable_triangle_count": int(overlay.walkable_triangle_count),
            "non_walk_triangle_count": int(overlay.non_walk_triangle_count),
            "max_slope_degrees": float(overlay.max_slope_degrees),
            "summary": summary,
            "next_action": (
                "Select a terrain room, apply a brush stroke, then validate WOK slope/walkability before staging."
                if ready
                else "Create a Terrain Patch or convert a room to terrain before sculpting."
            ),
            "warnings": tuple(overlay.warnings),
            "capability_stage": "previewable_status_query",
        }

    def authored_walkmesh_status(self):
        """Return modder-facing walkmesh status for the current KMAP."""

        if self._map_studio_authored_project_snapshot() is None:
            return AuthoredWalkmeshStatus(
                ready=False,
                summary="Walkmesh: no authored Map Studio module is loaded.",
                next_action="Create or open a KMAP with authored rooms before inspecting walkmesh.",
            )
        return self._map_studio_cached_authored_query(
            ("walkmesh_status",), authored_walkmesh_status_for_project
        )

    def authored_walkmesh_room_surface_choices(self):
        """Return authored rooms whose WOK floor surface can be edited in Walkmesh tools."""

        if self._map_studio_authored_project_snapshot() is None:
            return ()
        return self._map_studio_cached_authored_query(
            ("walkmesh_room_surface_choices",), authored_walkmesh_room_surface_choices
        )

    def create_authored_room_preset_module(self, *, preset_id: str, module_root: str = "grdev01"):
        """Store an authored module created from a named primitive room preset."""

        root = str(module_root or "grdev01").strip() or "grdev01"
        before = self._capture_map_studio_command_state()
        authored = create_authored_module_from_room_preset(
            preset_id=preset_id,
            module_root=root,
            game=str(self.project.game or "K1").upper(),
        )
        payload = authored_project_to_kmap_payload(authored)
        self.project.extra_sections["authored_module"] = payload
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        self.model.log(f"Created authored Map Studio module {self.project.name} from primitive preset {preset_id}.")
        self._record_map_studio_command(
            action_key="map_studio.room_preset.create",
            label=f"Create authored module {authored.metadata.module_root}",
            before=before,
            metadata={"preset_id": preset_id, "module_root": root},
        )
        return self.authored_module_readiness()

    def apply_authored_room_operation(self, *, operation: str, **kwargs: Any):
        """Apply a floor-plan shaping operation to the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = apply_authored_floor_plan_operation(authored, operation, **kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(f"Applied Map Studio room operation {operation}.")
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.operation",
            label=f"Apply room operation {operation}",
            before=before,
            metadata={"operation": operation, "kwargs": dict(kwargs)},
        )
        return self.authored_module_readiness()

    def rectangular_cut_authored_floor_plan_room(
        self,
        *,
        room_resref: str = "",
        center: Any,
        size: Any,
        room_resref_prefix: str | None = None,
    ):
        """Cut one authored floor-plan room with a rectangular cutter and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        cut_center = tuple(float(value) for value in tuple(center or (0.0, 0.0))[:2])
        cut_size = tuple(float(value) for value in tuple(size or (1.0, 1.0))[:2])
        if len(cut_center) != 2 or len(cut_size) != 2:
            raise ValueError("Rectangular cut requires a 2D center and size in floor-plan local space.")
        updated = apply_authored_floor_plan_rectangular_cut(
            authored,
            room_resref=room_resref,
            center=(cut_center[0], cut_center[1]),
            size=(cut_size[0], cut_size[1]),
            room_resref_prefix=room_resref_prefix,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Rectangular-cut Map Studio room {room_resref or '(first room)'} at {cut_center} size {cut_size}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.rectangular_cut",
            label=f"Rectangular cut {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "center": cut_center,
                "size": cut_size,
                "room_resref_prefix": room_resref_prefix or "",
            },
        )
        return self.authored_module_readiness()

    def boolean_difference_authored_floor_plan_rooms(
        self,
        *,
        first_room_resref: str,
        second_room_resref: str,
        result_room_resref: str = "",
    ):
        """Subtract one authored floor-plan room from another with explicit KMAP metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = apply_authored_floor_plan_boolean_difference(
            authored,
            first_room_resref=first_room_resref,
            second_room_resref=second_room_resref,
            result_room_resref=result_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Applied Map Studio boolean difference {first_room_resref} - {second_room_resref}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.boolean_difference",
            label=f"Boolean difference {first_room_resref} - {second_room_resref}",
            before=before,
            metadata={
                "first_room_resref": first_room_resref,
                "second_room_resref": second_room_resref,
                "result_room_resref": result_room_resref,
            },
        )
        return self.authored_module_readiness()

    def axis_split_authored_floor_plan_room(
        self,
        *,
        room_resref: str = "",
        axis: str,
        coordinate: float,
        room_resref_prefix: str | None = None,
    ):
        """Split one floor-plan room along a local X/Y axis with explicit KMAP metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        split_axis = str(axis or "x").strip().lower()
        split_coordinate = float(coordinate)
        updated = apply_authored_floor_plan_axis_split(
            authored,
            room_resref=room_resref,
            axis=split_axis,
            coordinate=split_coordinate,
            room_resref_prefix=room_resref_prefix,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Axis-split Map Studio room {room_resref or '(first room)'} on {split_axis}={split_coordinate:g}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.axis_split",
            label=f"Axis split {room_resref or 'room'} on {split_axis}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "axis": split_axis,
                "coordinate": split_coordinate,
                "room_resref_prefix": room_resref_prefix or "",
            },
        )
        return self.authored_module_readiness()

    def edge_extrude_authored_floor_plan_room(
        self,
        *,
        room_resref: str = "",
        edge_index: int,
        distance: float,
    ):
        """Pull one authored floor-plan room edge outward and record explicit extrusion command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        edge = int(edge_index)
        extrusion_distance = float(distance)
        updated = apply_authored_floor_plan_edge_extrude(
            authored,
            room_resref=room_resref,
            edge_index=edge,
            distance=extrusion_distance,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Extruded Map Studio room {room_resref or '(first room)'} edge {edge} by {extrusion_distance:g} m; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.edge_extrude",
            label=f"Extrude edge {edge} on {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "edge_index": edge,
                "distance": extrusion_distance,
            },
        )
        return self.authored_module_readiness()

    def inset_authored_floor_plan_room(
        self,
        *,
        room_resref: str = "",
        distance: float,
    ):
        """Inset one authored floor-plan room and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        inset_distance = float(distance)
        updated = apply_authored_floor_plan_inset(
            authored,
            room_resref=room_resref,
            distance=inset_distance,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Inset Map Studio room {room_resref or '(first room)'} by {inset_distance:g} m; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.inset",
            label=f"Inset {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "distance": inset_distance,
            },
        )
        return self.authored_module_readiness()

    def bevel_authored_floor_plan_room(
        self,
        *,
        room_resref: str = "",
        distance: float,
    ):
        """Bevel one authored floor-plan room and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        bevel_distance = float(distance)
        updated = apply_authored_floor_plan_bevel(
            authored,
            room_resref=room_resref,
            distance=bevel_distance,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Beveled Map Studio room {room_resref or '(first room)'} by {bevel_distance:g} m; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.bevel",
            label=f"Bevel {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "distance": bevel_distance,
            },
        )
        return self.authored_module_readiness()

    def set_authored_floor_plan_wall_opening(
        self,
        *,
        room_resref: str = "",
        name: str = "",
        edge_index: int = 0,
        center_fraction: float = 0.5,
        width: float = 1.5,
        height: float = 2.1,
        bottom: float = 0.0,
    ):
        """Add or replace one named wall opening on an authored floor-plan room edge."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        opening_name = str(name or "").strip() or f"opening_edge_{int(edge_index)}"
        updated = set_authored_floor_plan_wall_opening(
            authored,
            room_resref=room_resref,
            name=opening_name,
            edge_index=int(edge_index),
            center_fraction=float(center_fraction),
            width=float(width),
            height=float(height),
            bottom=float(bottom),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Set Map Studio wall opening {opening_name} on {room_resref or '(first room)'} edge {int(edge_index)}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.wall_opening",
            label=f"Set wall opening {opening_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "name": opening_name,
                "edge_index": int(edge_index),
                "center_fraction": float(center_fraction),
                "width": float(width),
                "height": float(height),
                "bottom": float(bottom),
            },
        )
        return self.authored_module_readiness()

    def add_authored_floor_plan_opening_transition_marker(
        self,
        *,
        room_resref: str = "",
        opening_name: str = "",
        edge_index: int | None = None,
        marker_kind: str = "door",
        template_resref: str = "",
        tag: str = "",
        linked_to: str = "",
        linked_to_module: str = "",
        linked_to_flags: int = 0,
        transition_destination: int = 0,
    ):
        """Create KOTOR door/trigger/waypoint transition data from one authored wall opening."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        marker_type = str(marker_kind or "door").strip().lower() or "door"
        updated = add_authored_floor_plan_opening_transition_marker(
            authored,
            room_resref=room_resref,
            opening_name=opening_name,
            edge_index=edge_index,
            marker_kind=marker_type,
            template_resref=template_resref,
            tag=tag,
            linked_to=linked_to,
            linked_to_module=linked_to_module,
            linked_to_flags=int(linked_to_flags),
            transition_destination=int(transition_destination),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Created Map Studio opening marker {tag or opening_name or '(selected opening)'} as {marker_type}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.opening_transition_marker",
            label=f"Add opening marker {tag or opening_name or marker_type}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "opening_name": opening_name,
                "edge_index": edge_index,
                "marker_kind": marker_type,
                "template_resref": template_resref,
                "tag": tag,
                "linked_to": linked_to,
                "linked_to_module": linked_to_module,
                "linked_to_flags": int(linked_to_flags),
                "transition_destination": int(transition_destination),
            },
        )
        return self.authored_module_readiness()

    def create_terrain_patch(
        self,
        *,
        room_resref: str = "",
        shape_preset_id: str = "flat",
        resolution: int = 17,
        width: float = 20.0,
        depth: float = 20.0,
        module_root: str = "",
    ):
        """Add a sculptable terrain heightfield room to the authored module.

        This is the entry point for terrain painting: it creates the room the
        terrain brush/sculpt pipeline targets.  Auto-creates the authored
        module if the KMAP has none yet.
        """

        grid = max(2, int(resolution))
        root = str(module_root or getattr(self.project, "name", "") or "grterrain").strip() or "grterrain"
        before = self._capture_map_studio_command_state()
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            authored = None
        else:
            authored = authored_project_from_kmap_payload(
                payload,
                fallback_name=str(getattr(self.project, "name", "") or "new_level"),
                fallback_game=str(getattr(self.project, "game", "") or "K1"),
            )
        base_resref = normalise_resref(room_resref) or f"{normalise_resref(root)}_terrain"
        existing = {room.normalised_resref() for room in (authored.rooms if authored else ())}
        resref = base_resref
        suffix = 1
        while resref in existing:
            suffix += 1
            resref = f"{base_resref}{suffix}"
        flat_heights = tuple(tuple(0.0 for _ in range(grid)) for _ in range(grid))
        terrain = TerrainHeightfieldPrimitive(
            room_resref=resref,
            heights=flat_heights,
            width=float(width),
            depth=float(depth),
        )
        preset = str(shape_preset_id or "flat").strip() or "flat"
        if preset != "flat":
            try:
                terrain = apply_terrain_shape_preset(terrain, preset_id=preset)
            except Exception as exc:
                self.model.log(f"Terrain shape preset {preset} skipped: {exc}")
        room = AuthoredRoomSpec(
            room_resref=resref,
            primitive=terrain,
            visible_rooms=(resref,),
            metadata={"primitive": "terrain_heightfield", "source": "terrain_patch_tool"},
        )
        if authored is None:
            from .authored_module_project import create_terrain_room_project
            from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint

            authored = create_terrain_room_project(
                module_root=root,
                game=str(self.project.game or "K1").upper(),
                display_name=f"{root} terrain",
                terrain=terrain,
                placements=AuthoredGameplayPlacement(
                    entry_point=ModuleEntryPoint(area_resref=normalise_resref(root))
                ),
            )
        else:
            from dataclasses import replace as _replace

            authored = _replace(authored, rooms=tuple(authored.rooms) + (room,))
        self._store_authored_project(authored)
        self.model.log(
            f"Created terrain patch {resref} ({grid}x{grid} grid, {preset} shape). Enter Terrain mode and paint with the sculpt brushes."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.create_patch",
            label=f"Create terrain patch {resref}",
            before=before,
            metadata={"room_resref": resref, "shape_preset_id": preset, "resolution": grid},
        )
        return resref

    def apply_authored_terrain_operation(self, *, operation: str, **kwargs: Any):
        """Apply a terrain heightfield operation to the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = apply_authored_terrain_operation(authored, operation, **kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(f"Applied Map Studio terrain operation {operation}.")
        self._record_map_studio_command(
            action_key="map_studio.terrain.operation",
            label=f"Apply terrain operation {operation}",
            before=before,
            metadata={"operation": operation, "kwargs": dict(kwargs)},
        )
        return self.authored_module_readiness()

    def apply_authored_terrain_brush_stroke(
        self,
        *,
        brush: str = "raise",
        room_resref: str = "",
        row_index: int = 0,
        column_index: int = 0,
        points: tuple[tuple[int, int, float], ...] = (),
        delta: float = 0.1,
        radius: int = 0,
        height: float = 0.0,
        iterations: int = 1,
        strength: float = 0.5,
        falloff_hardness: float = 0.5,
        preserve_boundary: bool = True,
        symmetry_axis: str = "",
    ):
        """Commit one dirty-region scoped terrain brush stroke to the authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        brush_key = str(brush or "raise").strip().lower() or "raise"
        stroke_points = tuple(points or ((int(row_index), int(column_index), 1.0),))
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        operation_kwargs: dict[str, Any] = {
            "room_resref": room_resref,
            "row_index": int(row_index),
            "column_index": int(column_index),
            "points": stroke_points,
            "delta": float(delta),
            "radius": int(radius),
            "height": float(height),
            "iterations": int(iterations),
            "strength": float(strength),
            "falloff_hardness": float(falloff_hardness),
            "preserve_boundary": bool(preserve_boundary),
        }
        if str(symmetry_axis or "").strip():
            operation_kwargs["symmetry_axis"] = str(symmetry_axis or "").strip().lower()
        updated = apply_authored_terrain_operation(authored, f"brush_stroke:{brush_key}", **operation_kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Applied Map Studio terrain brush {brush_key} to {room_resref or '(first terrain room)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.brush_stroke",
            label=f"Apply terrain brush {brush_key}",
            before=before,
            metadata={"brush": brush_key, **operation_kwargs},
        )
        return self.authored_module_readiness()

    def shrink_wrap_authored_placements_to_terrain(
        self,
        *,
        room_resref: str = "",
    ):
        """Project authored gameplay placements onto one terrain heightfield and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = apply_authored_terrain_operation(authored, "shrink_wrap", room_resref=room_resref)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Shrink-wrapped Map Studio placements to terrain {room_resref or '(first terrain room)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.shrink_wrap_placements",
            label=f"Shrink wrap placements {room_resref or 'terrain'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "target": "authored_gameplay_placements",
                "surface": "terrain_heightfield",
            },
        )
        return self.authored_module_readiness()

    def mirror_z_authored_terrain_heightfield(
        self,
        *,
        room_resref: str = "",
        center_height: float | None = None,
    ):
        """Mirror one authored terrain heightfield vertically and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        kwargs: dict[str, Any] = {"room_resref": room_resref}
        if center_height is not None:
            kwargs["center_height"] = float(center_height)
        updated = apply_authored_terrain_operation(authored, "mirror_z", **kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Mirror-Z Map Studio terrain {room_resref or '(first terrain room)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.mirror_z",
            label=f"Mirror Z terrain {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "center_height": None if center_height is None else float(center_height),
            },
        )
        return self.authored_module_readiness()

    def bend_authored_terrain_heightfield(
        self,
        *,
        room_resref: str = "",
        axis: str = "x",
        amplitude: float = 0.25,
        center: float | None = None,
        span: float | None = None,
    ):
        """Bend one authored terrain heightfield and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        bend_axis = str(axis or "x").strip().lower() or "x"
        bend_amplitude = float(amplitude)
        kwargs: dict[str, Any] = {
            "room_resref": room_resref,
            "axis": bend_axis,
            "amplitude": bend_amplitude,
        }
        if center is not None:
            kwargs["center"] = float(center)
        if span is not None:
            kwargs["span"] = float(span)
        updated = apply_authored_terrain_operation(authored, "bend", **kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Bent Map Studio terrain {room_resref or '(first terrain room)'} on {bend_axis} by {bend_amplitude:g}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.bend",
            label=f"Bend terrain {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "axis": bend_axis,
                "amplitude": bend_amplitude,
                "center": None if center is None else float(center),
                "span": None if span is None else float(span),
            },
        )
        return self.authored_module_readiness()

    def lattice_authored_terrain_heightfield(
        self,
        *,
        room_resref: str = "",
        strength: float = 1.0,
        control_deltas: Any = None,
        amplitude: float | None = None,
    ):
        """Apply one authored terrain lattice deformation and record explicit command metadata."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        lattice_strength = float(strength)
        kwargs: dict[str, Any] = {
            "room_resref": room_resref,
            "strength": lattice_strength,
        }
        if control_deltas is not None:
            kwargs["control_deltas"] = control_deltas
        elif amplitude is not None:
            kwargs["amplitude"] = float(amplitude)
        updated = apply_authored_terrain_operation(authored, "lattice", **kwargs)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Applied Map Studio terrain lattice to {room_resref or '(first terrain room)'} at strength {lattice_strength:g}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.terrain.lattice",
            label=f"Lattice terrain {room_resref or 'room'}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "strength": lattice_strength,
                "control_deltas": control_deltas,
                "amplitude": None if amplitude is None else float(amplitude),
            },
        )
        return self.authored_module_readiness()

    def add_authored_curve_guide(
        self,
        *,
        name: str = "",
        points: Any = (),
        purpose: str = "path_guide",
        room_resref: str = "",
        coordinate_space: str = "kmap_world",
        metadata: dict[str, Any] | None = None,
    ):
        """Add a Map Studio authoring curve guide to the current KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = add_curve_guide_to_project(
            authored,
            name=name,
            points=points,
            purpose=purpose,
            room_resref=room_resref,
            coordinate_space=coordinate_space,
            metadata=metadata,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        guide_name = str(updated.extra.get("last_curve_guide") or name or "curve")
        self.model.log(
            f"Added Map Studio curve guide {guide_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.curve_guide.add",
            label=f"Add curve guide {guide_name}",
            before=before,
            metadata={
                "name": name,
                "purpose": purpose,
                "room_resref": room_resref,
                "coordinate_space": coordinate_space,
            },
        )
        return self.authored_module_readiness()

    def prepare_map_studio_terrain_sculpt_frame(
        self,
        *,
        room_resref: str,
        brush: str,
        points: tuple[Any, ...] | list[Any],
        delta: float = 0.1,
        radius: int = 0,
        height: float = 0.0,
        iterations: int = 1,
        strength: float = 0.5,
        falloff_hardness: float = 0.5,
        preserve_boundary: bool = True,
        max_points_per_frame: int = 8,
        budget_ms: float = 8.0,
    ):
        """Prepare a live terrain sculpt frame without mutating the KMAP project."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        return prepare_terrain_sculpt_frame_for_project(
            authored,
            room_resref=room_resref,
            brush=brush,
            points=points,
            delta=delta,
            radius=radius,
            height=height,
            iterations=iterations,
            strength=strength,
            falloff_hardness=falloff_hardness,
            preserve_boundary=preserve_boundary,
            max_points_per_frame=max_points_per_frame,
            budget_ms=budget_ms,
        )

    def commit_map_studio_terrain_sculpt_stroke(self, *, brush: str, room_resref: str):
        """Serialize and record one undo command for a released live stroke.

        Return the stroke commit itself instead of running authored-module
        readiness here.  The Map Studio refresh that follows release owns the
        full geometry/walkability readiness pass; doing it in both places made
        a 129x129 terrain release pay that whole-grid cost twice.
        """

        session = self._terrain_sculpt_session
        before = self._terrain_sculpt_command_before
        self._terrain_sculpt_command_before = None
        self._terrain_sculpt_session = None
        commit = None
        if session is not None:
            commit = session.commit()
            updated = commit.project
            self.project.extra_sections["authored_module"] = commit.payload
            self.project.name = updated.metadata.module_root
            self.project.game = updated.game
            self.project.dirty = True
            self.model.log(
                f"Committed terrain sculpt {brush} on {room_resref}: {commit.frame_count} frame(s), "
                f"{commit.dirty_region.changed_sample_count} changed sample(s), one KMAP serialization "
                f"({commit.serialization_elapsed_ms:.3f} ms)."
            )
        if before is not None:
            self._record_map_studio_command(
                action_key="map_studio.terrain.sculpt_stroke",
                label=f"Sculpt terrain {brush} on {room_resref}",
                before=before,
                metadata={
                    "brush": brush,
                    "room_resref": room_resref,
                    "frame_count": int(getattr(session, "frame_count", 0) or 0),
                    "serialization_count": int(getattr(session, "serialization_count", 0) or 0),
                },
            )
        return commit

    def apply_map_studio_terrain_sculpt_frame(
        self,
        *,
        room_resref: str,
        brush: str,
        points: tuple[Any, ...] | list[Any],
        delta: float = 0.1,
        radius: int = 0,
        height: float = 0.0,
        iterations: int = 1,
        strength: float = 0.5,
        falloff_hardness: float = 0.5,
        preserve_boundary: bool = True,
        max_points_per_frame: int = 8,
        budget_ms: float = 8.0,
        force: bool = False,
    ) -> MapStudioTerrainSculptApplyResult:
        """Mutate one stroke-owned dirty height buffer; never serialize per frame."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        wanted_room = normalise_resref(room_resref)
        session = self._terrain_sculpt_session
        if session is None or normalise_resref(getattr(session, "room_resref", "")) != wanted_room:
            if session is not None and not bool(getattr(session, "committed", False)):
                session.cancel()
            self._terrain_sculpt_command_before = self._capture_map_studio_command_state()
            session = begin_terrain_sculpt_stroke(
                payload,
                room_resref=room_resref,
                fallback_name=str(getattr(self.project, "name", "") or "new_level"),
                fallback_game=str(getattr(self.project, "game", "") or "K1"),
            )
            self._terrain_sculpt_session = session
        result = session.apply_frame(
            brush=brush,
            points=points,
            delta=delta,
            radius=radius,
            height=height,
            iterations=iterations,
            strength=strength,
            falloff_hardness=falloff_hardness,
            preserve_boundary=preserve_boundary,
            max_points_per_frame=max_points_per_frame,
            budget_ms=budget_ms,
            force=force,
        )
        patch_region = result.dirty_region_with_halo
        patch = session.dirty_height_patch(patch_region) if result.applied else ()
        return MapStudioTerrainSculptApplyResult(
            applied=bool(result.applied),
            frame=result.frame,
            message=result.message,
            dirty_region=result.dirty_region,
            dirty_region_with_halo=patch_region,
            dirty_height_patch=patch,
            row_count=int(session.row_count),
            column_count=int(session.column_count),
            elapsed_ms=float(result.elapsed_ms),
            project_serialized=False,
        )

    def merge_authored_floor_plan_rooms(
        self,
        *,
        first_room_resref: str,
        second_room_resref: str,
        result_room_resref: str = "",
    ):
        """Union two compatible floor-plan rooms in the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = apply_authored_floor_plan_rectangular_union(
            authored,
            first_room_resref=first_room_resref,
            second_room_resref=second_room_resref,
            result_room_resref=result_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Merged Map Studio floor-plan rooms {first_room_resref} and {second_room_resref}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.merge_rooms",
            label=f"Merge {first_room_resref} and {second_room_resref}",
            before=before,
            metadata={
                "first_room_resref": first_room_resref,
                "second_room_resref": second_room_resref,
                "result_room_resref": result_room_resref,
            },
        )
        return self.authored_module_readiness()

    def combine_authored_room_primitives(
        self,
        *,
        room_resref: str,
        primitive_names: Any,
        group_name: str = "",
    ):
        """Combine selected objects into one genuine procedural polygon mesh."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = combine_authored_room_composition_meshes(
            authored,
            room_resref=room_resref,
            primitive_names=primitive_names,
            combined_name=group_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        if isinstance(primitive_names, (str, bytes)):
            text = primitive_names.decode("utf-8", errors="ignore") if isinstance(primitive_names, bytes) else primitive_names
            selected = tuple(part.strip() for part in text.split(",") if part.strip())
        else:
            selected = tuple(str(name or "").strip() for name in tuple(primitive_names or ()) if str(name or "").strip())
        self.model.log(
            f"Combined Map Studio primitives {', '.join(selected)} into one polygon mesh; "
            "source recipes, materials, UVs, normals, and disconnected shells were preserved."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.combine_meshes",
            label=f"Combine meshes {', '.join(selected)}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_names": selected,
                "combined_name": group_name,
            },
        )
        return self.authored_module_readiness()

    def group_authored_room_primitives(
        self,
        *,
        room_resref: str,
        primitive_names: Any,
        group_name: str = "",
    ):
        """Create a scene group while keeping polygon objects independent."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = group_authored_room_composition_primitives(
            authored,
            room_resref=room_resref,
            primitive_names=primitive_names,
            group_name=group_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        if isinstance(primitive_names, (str, bytes)):
            text = primitive_names.decode("utf-8", errors="ignore") if isinstance(primitive_names, bytes) else primitive_names
            selected = tuple(part.strip() for part in text.split(",") if part.strip())
        else:
            selected = tuple(str(name or "").strip() for name in tuple(primitive_names or ()) if str(name or "").strip())
        self._record_map_studio_command(
            action_key="map_studio.primitive.group",
            label=f"Group objects {', '.join(selected)}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_names": selected, "group_name": group_name},
        )
        self.model.log(f"Grouped {len(selected)} Map Studio objects; polygon topology remains independent.")
        return self.authored_module_readiness()

    def separate_authored_room_primitive_shells(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        name_prefix: str = "",
    ):
        """Split one Combined Mesh into connected polygon-shell objects."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = separate_authored_room_combined_primitive_shells(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            name_prefix=name_prefix,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        room = next((row for row in updated.rooms if row.normalised_resref() == normalise_resref(room_resref)), None)
        shell_names = tuple(
            str(value)
            for value in tuple(dict(getattr(room, "metadata", {}) or {}).get("last_separated_shell_names") or ())
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.separate_shells",
            label=f"Separate shells {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "name_prefix": name_prefix,
                "shell_names": shell_names,
            },
        )
        self.model.log(
            f"Separated {primitive_name} into {len(shell_names)} connected polygon shell object(s); "
            "previous exports/proofs are now stale."
        )
        return self.authored_module_readiness()

    def bridge_authored_floor_plan_edges(
        self,
        *,
        first_room_resref: str,
        first_edge_index: int,
        second_room_resref: str,
        second_edge_index: int,
        result_room_resref: str = "",
    ):
        """Create a connector room between two compatible floor-plan room edges."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = bridge_authored_floor_plan_edges(
            authored,
            first_room_resref=first_room_resref,
            first_edge_index=int(first_edge_index),
            second_room_resref=second_room_resref,
            second_edge_index=int(second_edge_index),
            result_room_resref=result_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Bridged Map Studio floor-plan edge {first_edge_index} in {first_room_resref} "
            f"to edge {second_edge_index} in {second_room_resref}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.bridge_edges",
            label=f"Bridge {first_room_resref}:{int(first_edge_index)} to {second_room_resref}:{int(second_edge_index)}",
            before=before,
            metadata={
                "first_room_resref": first_room_resref,
                "first_edge_index": int(first_edge_index),
                "second_room_resref": second_room_resref,
                "second_edge_index": int(second_edge_index),
                "result_room_resref": result_room_resref,
            },
        )
        return self.authored_module_readiness()

    def set_authored_floor_plan_extrusion(
        self,
        *,
        room_resref: str = "",
        z: float | None = None,
        wall_height: float | None = None,
        include_walls: bool | None = None,
        floor_surface_id: int | str | None = None,
    ):
        """Set explicit room extrusion parameters for the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        updated = set_authored_floor_plan_extrusion_settings(
            authored,
            room_resref=room_resref,
            z=z,
            wall_height=wall_height,
            include_walls=include_walls,
            floor_surface_id=floor_surface_id,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Updated Map Studio floor-plan extrusion settings for {room_resref or 'the selected room'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.set_extrusion",
            label=f"Set {room_resref or 'room'} extrusion",
            before=before,
            metadata={
                "room_resref": room_resref,
                "z": z,
                "wall_height": wall_height,
                "include_walls": include_walls,
                "floor_surface_id": floor_surface_id,
            },
        )
        return self.authored_module_readiness()

    def move_authored_room_outline_point(self, *, room_resref: str, point_index: int, world_position: Any):
        """Move one authored room outline vertex through the headless floor-plan operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = move_authored_floor_plan_point(
            authored,
            room_resref=room_resref,
            point_index=int(point_index),
            world_position=tuple(world_position),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Moved Map Studio room {room_resref or '(first room)'} outline point {int(point_index)}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.move_vertex",
            label=f"Move {room_resref or 'room'} outline point {int(point_index)}",
            before=before,
            metadata={"room_resref": room_resref, "point_index": int(point_index), "world_position": tuple(world_position)},
        )
        return self.authored_module_readiness()

    def authored_floor_plan_vertex_snap_candidates(
        self,
        *,
        room_resref: str,
        point_index: int,
        max_distance: float | None = None,
        include_same_room: bool = True,
        include_cross_room: bool = True,
        limit: int = 8,
    ):
        """Return nearest floor-plan vertex snap targets without mutating project state."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        return authored_floor_plan_vertex_snap_candidates(
            authored,
            room_resref=room_resref,
            point_index=int(point_index),
            max_distance=max_distance,
            include_same_room=bool(include_same_room),
            include_cross_room=bool(include_cross_room),
            limit=int(limit),
        )

    def authored_room_primitive_vertex_snap_candidates(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        target_primitive_name: str = "",
        max_results: int = 8,
        distance_limit: float | None = None,
    ):
        """Return nearest primitive vertex snap targets without mutating project state."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        return authored_room_composition_primitive_vertex_snap_candidates(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            target_primitive_name=target_primitive_name,
            max_results=int(max_results),
            distance_limit=distance_limit,
        )

    def snap_authored_floor_plan_vertex(
        self,
        *,
        room_resref: str,
        point_index: int,
        target_point_index: int,
        target_room_resref: str = "",
    ):
        """Snap one authored floor-plan vertex to another through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = snap_authored_floor_plan_vertex_to_vertex(
            authored,
            room_resref=room_resref,
            point_index=int(point_index),
            target_point_index=int(target_point_index),
            target_room_resref=target_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Snapped Map Studio room {room_resref or '(first room)'} point {int(point_index)} to point {int(target_point_index)}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.snap_vertex",
            label=f"Snap {room_resref or 'room'} point {int(point_index)}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "point_index": int(point_index),
                "target_point_index": int(target_point_index),
                "target_room_resref": target_room_resref,
            },
        )
        return self.authored_module_readiness()

    def grid_snap_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        point_indices: Any,
        grid_size: float = 0.1,
        axes: Any = ("x", "y"),
    ):
        """Snap authored floor-plan vertices to grid through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        axes_tuple = tuple(str(axis or "").strip().lower() for axis in tuple(axes or ("x", "y")))
        updated = grid_snap_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            point_indices=indices,
            grid_size=float(grid_size),
            axes=axes_tuple,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Grid-snapped Map Studio room {room_resref or '(first room)'} floor-plan vertices {indices} to {float(grid_size):g}m; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.grid_snap_vertices",
            label=f"Grid snap {room_resref or 'room'} vertices",
            before=before,
            metadata={"room_resref": room_resref, "point_indices": indices, "grid_size": float(grid_size), "axes": axes_tuple},
        )
        return self.authored_module_readiness()

    def weld_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        point_indices: Any,
        target_point_index: int | None = None,
        position_policy: str = "target",
    ):
        """Weld authored floor-plan vertices through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        updated = weld_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            point_indices=indices,
            target_point_index=target_point_index,
            position_policy=position_policy,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Welded Map Studio room {room_resref or '(first room)'} floor-plan vertices {indices}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.weld_vertices",
            label=f"Weld {room_resref or 'room'} vertices",
            before=before,
            metadata={
                "room_resref": room_resref,
                "point_indices": indices,
                "target_point_index": target_point_index,
                "position_policy": position_policy,
            },
        )
        return self.authored_module_readiness()

    def flatten_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        point_indices: Any,
        axis: str = "x",
        value: float | None = None,
    ):
        """Flatten authored floor-plan vertices through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        updated = flatten_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            point_indices=indices,
            axis=axis,
            value=value,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Flattened Map Studio room {room_resref or '(first room)'} floor-plan vertices {indices} on {axis}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.flatten_vertices",
            label=f"Flatten {room_resref or 'room'} vertices on {axis}",
            before=before,
            metadata={"room_resref": room_resref, "point_indices": indices, "axis": axis, "value": value},
        )
        return self.authored_module_readiness()

    def transform_snap_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        point_indices: Any,
        axis: str = "x",
        target_point_index: int | None = None,
        value: float | None = None,
        level_policy: str = "average",
    ):
        """Apply hold-J transform level snapping through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        updated = transform_snap_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            point_indices=indices,
            axis=axis,
            target_point_index=target_point_index,
            value=value,
            level_policy=level_policy,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Transform-snapped Map Studio room {room_resref or '(first room)'} floor-plan vertices {indices} on {axis}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.transform_snap_level",
            label=f"Transform snap {room_resref or 'room'} vertices on {axis}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "point_indices": indices,
                "axis": axis,
                "target_point_index": target_point_index,
                "value": value,
                "level_policy": level_policy,
            },
        )
        return self.authored_module_readiness()

    def cleanup_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        tolerance: float = 0.001,
    ):
        """Cleanup redundant authored floor-plan vertices through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = cleanup_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            tolerance=float(tolerance),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Cleaned Map Studio room {room_resref or '(first room)'} floor-plan vertices; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.cleanup_vertices",
            label=f"Clean {room_resref or 'room'} floor-plan vertices",
            before=before,
            metadata={"room_resref": room_resref, "tolerance": float(tolerance)},
        )
        return self.authored_module_readiness()

    def fill_authored_floor_plan_face(
        self,
        *,
        room_resref: str,
        point_indices: Any,
    ):
        """Fill a floor-plan point loop through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        updated = fill_authored_floor_plan_face(
            authored,
            room_resref=room_resref,
            point_indices=indices,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Filled Map Studio room {room_resref or '(first room)'} floor-plan face loop {indices}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.fill_face",
            label=f"Fill {room_resref or 'room'} floor-plan face",
            before=before,
            metadata={"room_resref": room_resref, "point_indices": indices},
        )
        return self.authored_module_readiness()

    def triangulate_authored_floor_plan_face(
        self,
        *,
        room_resref: str,
    ):
        """Triangulate a floor-plan footprint through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = triangulate_authored_floor_plan_face(
            authored,
            room_resref=room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Triangulated Map Studio room {room_resref or '(first room)'} floor-plan face; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.triangulate_face",
            label=f"Triangulate {room_resref or 'room'} floor-plan face",
            before=before,
            metadata={"room_resref": room_resref},
        )
        return self.authored_module_readiness()

    def split_authored_floor_plan_face(
        self,
        *,
        room_resref: str,
        point_indices: Any,
    ):
        """Record a selected-vertex face split through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        indices = tuple(int(index) for index in tuple(point_indices or ()))
        updated = split_authored_floor_plan_face(
            authored,
            room_resref=room_resref,
            point_indices=indices,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Split Map Studio room {room_resref or '(first room)'} floor-plan face between points {indices}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.split_face",
            label=f"Split {room_resref or 'room'} floor-plan face",
            before=before,
            metadata={"room_resref": room_resref, "point_indices": indices},
        )
        return self.authored_module_readiness()

    def cleanup_authored_floor_plan_normals(
        self,
        *,
        room_resref: str,
        positive_z: bool = True,
    ):
        """Clean floor-plan face winding through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = cleanup_authored_floor_plan_normals(
            authored,
            room_resref=room_resref,
            positive_z=bool(positive_z),
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Cleaned Map Studio room {room_resref or '(first room)'} floor-plan normals; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.cleanup_normals",
            label=f"Clean {room_resref or 'room'} floor-plan normals",
            before=before,
            metadata={"room_resref": room_resref, "positive_z": bool(positive_z)},
        )
        return self.authored_module_readiness()

    def mirror_authored_floor_plan_vertices(
        self,
        *,
        room_resref: str,
        axis: str = "x",
    ):
        """Mirror an authored floor-plan footprint through the domain operation."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = mirror_authored_floor_plan_vertices(
            authored,
            room_resref=room_resref,
            axis=axis,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Mirrored Map Studio room {room_resref or '(first room)'} floor-plan across local {axis}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.floor_plan.mirror_vertices",
            label=f"Mirror {room_resref or 'room'} floor-plan on {axis}",
            before=before,
            metadata={"room_resref": room_resref, "axis": axis},
        )
        return self.authored_module_readiness()

    def set_authored_room_primitive_transform(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        translation: Any = None,
        rotation_degrees_z: float | None = None,
        scale: Any = None,
        pivot: Any = None,
    ):
        """Set one authored composition primitive transform in the current KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = set_authored_room_composition_primitive_transform(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            translation=translation,
            rotation_degrees_z=rotation_degrees_z,
            scale=scale,
            pivot=pivot,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Transformed Map Studio room primitive {primitive_name} in {room_resref or '(first room)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.transform",
            label=f"Transform primitive {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "translation": translation,
                "rotation_degrees_z": rotation_degrees_z,
                "scale": scale,
                "pivot": pivot,
            },
        )
        return self.authored_module_readiness()

    def move_authored_room_primitive(self, *, room_resref: str, primitive_name: str, world_delta: Any):
        """Move one authored composition primitive by a viewport-authored world delta."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = move_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            world_delta=world_delta,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Moved Map Studio room primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.move",
            label=f"Move primitive {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name, "world_delta": world_delta},
        )
        return self.authored_module_readiness()

    def transform_authored_room_primitives(
        self,
        *,
        selections: Any,
        mode: str,
        world_delta: Any = (0.0, 0.0, 0.0),
        rotation_delta_degrees_z: float = 0.0,
        scale_multiplier: Any = (1.0, 1.0, 1.0),
        world_pivot: Any = (0.0, 0.0, 0.0),
    ):
        """Transform a multi-object selection in one KMAP and undo transaction."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        selection_values = tuple(selections or ())
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = transform_authored_room_composition_primitives(
            authored,
            selections=selection_values,
            mode=mode,
            world_delta=world_delta,
            rotation_delta_degrees_z=rotation_delta_degrees_z,
            scale_multiplier=scale_multiplier,
            world_pivot=world_pivot,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        mode_key = str(updated.extra.get("batch_transform_mode") or mode or "translate").strip().lower()
        selection_count = int(updated.extra.get("batch_transform_count") or len(selection_values))
        label_verb = {"translate": "Move", "rotate": "Rotate", "scale": "Scale"}.get(mode_key, "Transform")
        label = f"{label_verb} {selection_count} primitive{'s' if selection_count != 1 else ''}"
        self.model.log(
            f"{label} as one Map Studio selection; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.batch_transform",
            label=label,
            before=before,
            metadata={
                "mode": mode_key,
                "selections": tuple(updated.extra.get("batch_transform_primitives") or ()),
                "selection_count": selection_count,
                "world_delta": world_delta,
                "rotation_delta_degrees_z": float(rotation_delta_degrees_z),
                "scale_multiplier": scale_multiplier,
                "world_pivot": world_pivot,
            },
        )
        return self.authored_module_readiness()

    def grid_snap_authored_room_primitive(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        grid_size: float = 0.1,
        axes: Any = ("x", "y", "z"),
    ):
        """Snap one authored primitive pivot to the Map Studio grid."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = grid_snap_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            grid_size=grid_size,
            axes=axes,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Grid-snapped Map Studio primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.object_grid_snap",
            label=f"Object grid snap {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "grid_size": grid_size,
                "axes": tuple(axes or ()),
            },
        )
        return self.authored_module_readiness()

    def snap_authored_room_primitive_pivot_to_vertex(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        target_primitive_name: str = "",
        target_vertex_index: int | None = None,
    ):
        """Snap one authored primitive pivot to a vertex on another primitive."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = snap_authored_room_composition_primitive_pivot_to_vertex(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            target_primitive_name=target_primitive_name,
            target_vertex_index=target_vertex_index,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Vertex-snapped Map Studio primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.object_vertex_snap",
            label=f"Object vertex snap {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "target_primitive_name": target_primitive_name,
                "target_vertex_index": None if target_vertex_index is None else int(target_vertex_index),
            },
        )
        return self.authored_module_readiness()

    def transform_snap_authored_room_primitive_level(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        axis: str = "z",
        target_primitive_name: str = "",
        target_vertex_index: int | None = None,
        value: float | None = None,
    ):
        """Align one authored primitive pivot component to a target level."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = transform_snap_authored_room_composition_primitive_level(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            axis=axis,
            target_primitive_name=target_primitive_name,
            target_vertex_index=target_vertex_index,
            value=value,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Transform-snapped Map Studio primitive {primitive_name} on {axis}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.object_transform_snap_level",
            label=f"Object transform snap {primitive_name} on {axis}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "axis": axis,
                "target_primitive_name": target_primitive_name,
                "target_vertex_index": None if target_vertex_index is None else int(target_vertex_index),
                "value": value,
            },
        )
        return self.authored_module_readiness()

    def mirror_authored_room_primitive_transform(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        axis: str = "x",
        center: float = 0.0,
    ):
        """Mirror one authored primitive object placement across a coordinate plane."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = mirror_authored_room_composition_primitive_transform(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            axis=axis,
            center=center,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Mirrored Map Studio primitive {primitive_name} across {axis}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.object_mirror",
            label=f"Object mirror {primitive_name} across {axis}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "axis": axis,
                "center": float(center),
            },
        )
        return self.authored_module_readiness()

    def shrink_wrap_authored_room_primitive_to_terrain(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        terrain_room_resref: str = "",
    ):
        """Drop one authored primitive to the selected/first terrain heightfield."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = shrink_wrap_authored_room_composition_primitive_to_terrain(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            terrain_room_resref=terrain_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Shrink-wrapped Map Studio primitive {primitive_name} to terrain; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.object_shrink_wrap_to_terrain",
            label=f"Object shrink wrap {primitive_name} to terrain",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "terrain_room_resref": terrain_room_resref,
            },
        )
        return self.authored_module_readiness()

    def reset_authored_room_primitive_transform(self, *, room_resref: str, primitive_name: str):
        """Reset one authored primitive's translate/rotate/scale, retaining its pivot."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = reset_authored_room_composition_primitive_transform(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Reset Map Studio transform for primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.reset_transform",
            label=f"Reset transformations {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def center_authored_room_primitive_pivot(self, *, room_resref: str, primitive_name: str):
        """Center one authored primitive pivot while preserving visible geometry."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = center_authored_room_composition_primitive_pivot(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Centered Map Studio pivot for primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.center_pivot",
            label=f"Center pivot {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def zero_authored_room_primitive_pivot(self, *, room_resref: str, primitive_name: str):
        """Move one authored primitive pivot to local zero without moving geometry."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = zero_authored_room_composition_primitive_pivot(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Zeroed Map Studio pivot for primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.zero_pivot",
            label=f"Zero pivot {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def freeze_authored_room_primitive_transform(self, *, room_resref: str, primitive_name: str):
        """Freeze a supported authored primitive transform into its parametric shape."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = freeze_authored_room_composition_primitive_transform(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Froze Map Studio transform for primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.freeze_transform",
            label=f"Freeze transform {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def delete_authored_room_primitive_history(self, *, room_resref: str, primitive_name: str):
        """Delete transient operator history while retaining evaluated/export data."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = delete_authored_room_composition_primitive_history(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Deleted Map Studio construction history for {primitive_name}; evaluated geometry and export provenance were retained."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.delete_history",
            label=f"Delete history {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def duplicate_authored_room_primitive(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        duplicate_count: int = 1,
        translation_offset: Any = (1.0, 0.0, 0.0),
        rotation_offset_degrees_z: float = 0.0,
        scale_multiplier: Any = (1.0, 1.0, 1.0),
    ):
        """Duplicate one authored composition primitive with repeatable transform offsets."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = duplicate_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            duplicate_count=duplicate_count,
            translation_offset=translation_offset,
            rotation_offset_degrees_z=rotation_offset_degrees_z,
            scale_multiplier=scale_multiplier,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Duplicated Map Studio room primitive {primitive_name} {int(duplicate_count)} time(s); previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.duplicate_special",
            label=f"Duplicate primitive {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "duplicate_count": int(duplicate_count),
                "translation_offset": translation_offset,
                "rotation_offset_degrees_z": rotation_offset_degrees_z,
                "scale_multiplier": scale_multiplier,
            },
        )
        return self.authored_module_readiness()

    def add_authored_room_primitive(
        self,
        *,
        primitive_kind: str,
        room_resref: str = "",
        primitive_name: str = "",
        translation: Any = None,
        rotation_degrees_z: float | None = None,
        scale: Any = None,
        pivot: Any = None,
        texture: str = "",
        floor_surface: Any = None,
        module_root: str = "",
    ):
        """Append a primitive instance to an authored composition room."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        before = self._capture_map_studio_command_state()
        auto_created_module = False
        root = str(module_root or getattr(self.project, "name", "") or "grblock").strip() or "grblock"
        if payload is None:
            authored = create_authored_module_from_room_preset(
                preset_id="composition_starter_room",
                module_root=root,
                game=str(self.project.game or "K1").upper(),
            )
            auto_created_module = True
        else:
            authored = authored_project_from_kmap_payload(
                payload,
                fallback_name=str(getattr(self.project, "name", "") or "new_level"),
                fallback_game=str(getattr(self.project, "game", "") or "K1"),
            )
        if auto_created_module and str(primitive_kind or "").strip().lower() in {"floor", "plane"}:
            updated = claim_authored_room_composition_floor(
                authored,
                room_resref=room_resref,
                primitive_name=primitive_name,
                texture=texture,
                floor_surface=floor_surface,
            )
        else:
            updated = add_authored_room_composition_primitive(
                authored,
                primitive_kind=primitive_kind,
                room_resref=room_resref,
                primitive_name=primitive_name,
                translation=translation,
                rotation_degrees_z=rotation_degrees_z,
                scale=scale,
                pivot=pivot,
                texture=texture,
                floor_surface=floor_surface,
            )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Added Map Studio room primitive {primitive_kind} {primitive_name or '(auto-named)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.add",
            label=f"Add {primitive_kind} primitive",
            before=before,
            metadata={
                "primitive_kind": primitive_kind,
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "translation": translation,
                "rotation_degrees_z": rotation_degrees_z,
                "scale": scale,
                "pivot": pivot,
                "texture": texture,
                "floor_surface": floor_surface,
                "module_root": root if auto_created_module else "",
                "auto_created_module": auto_created_module,
            },
        )
        return self.authored_module_readiness()

    def set_authored_room_primitive_dimensions(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        dimensions: Any,
    ):
        """Set editable dimensions for one authored composition primitive."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        # The decoded authored project is immutable and cached by payload
        # identity/revision.  Reusing it here keeps an Apply commit to one
        # recipe evaluation plus one KMAP serialization; live scrubbing uses
        # ``preview_authored_room_primitive_dimensions`` below and performs
        # neither serialization nor command-history capture.
        authored = self._load_authored_project_or_raise()
        updated = set_authored_room_composition_primitive_dimensions(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            dimensions=dimensions,
        )
        updated_primitive = _authored_primitive_by_identity(
            updated,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        # The immutable recipe update above allocates no geometry.  Enforce the
        # absolute safety policy before KMAP serialization, history mutation,
        # or any downstream mesh evaluation.
        enforce_primitive_topology_budget(updated_primitive, operation="commit")
        preview_identity = (normalise_resref(room_resref), str(primitive_name or "").strip())
        self.last_map_studio_primitive_commit_matches_preview = bool(
            self.last_map_studio_primitive_preview_identity == preview_identity
            and self.last_map_studio_primitive_preview_dimensions == dimensions
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Edited Map Studio room primitive dimensions for {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.set_dimensions",
            label=f"Set dimensions for primitive {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name, "dimensions": dimensions},
        )
        return self.authored_module_readiness()

    def preview_authored_room_primitive_dimensions(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        dimensions: Any,
    ) -> tuple[dict[str, Any], ...]:
        """Evaluate one retained primitive recipe without mutating KMAP state.

        Maya's primitive attributes dirty one construction node and redraw the
        selected mesh in place.  This method is the Scene-owned equivalent: it
        reuses the decoded authored snapshot, replaces one immutable recipe in
        memory, and returns only that primitive's render arrays.  It creates no
        undo record, does not mark the project dirty, and never serializes the
        authored payload.
        """

        started = perf_counter()
        authored = self._load_authored_project_or_raise()
        updated = set_authored_room_composition_primitive_dimensions(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            dimensions=dimensions,
        )
        wanted_room = normalise_resref(room_resref)
        wanted_name = str(primitive_name or "").strip()
        updated_primitive = _authored_primitive_by_identity(
            updated,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        try:
            # This integer-only estimate happens before ``primitive_to_mesh``
            # allocates positions, normals, UVs, or face tuples.
            enforce_primitive_topology_budget(updated_primitive, operation="preview")
        except (PrimitivePreviewDeferred, PrimitiveTopologySafetyError):
            self.last_map_studio_primitive_preview_deferred = True
            self.last_map_studio_primitive_preview_identity = None
            self.last_map_studio_primitive_preview_dimensions = None
            self.last_map_studio_primitive_preview_overlay = None
            self.last_map_studio_primitive_preview_elapsed_ms = (perf_counter() - started) * 1000.0
            raise
        self.last_map_studio_primitive_preview_deferred = False
        for room in tuple(updated.rooms or ()):
            if room.normalised_resref() != wanted_room:
                continue
            composition = room.primitive
            if not isinstance(composition, AuthoredRoomComposition):
                break
            candidates = (composition.floor,) + tuple(composition.primitives or ())
            for primitive in candidates:
                base = getattr(primitive, "primitive", primitive)
                candidate_name = str(
                    getattr(primitive, "name", "")
                    or getattr(base, "name", "")
                    or ""
                ).strip()
                if candidate_name != wanted_name:
                    continue
                mesh = primitive_to_mesh(primitive)
                try:
                    logical_cage = build_authored_primitive_polygon_cage(
                        primitive,
                        room_resref=wanted_room,
                    )
                except TypeError:
                    logical_cage = None
                if logical_cage is not None:
                    logical_vertex_count, logical_edge_count, logical_face_count = logical_topology_counts(logical_cage)
                    topology_count_source = "retained_construction_cage"
                else:
                    logical_vertex_count = len(tuple(mesh.vertices or ()))
                    logical_face_count = len(tuple(mesh.faces or ()))
                    logical_edge_count = 0
                    topology_count_source = "legacy_render_mesh_fallback"
                metadata = dict(getattr(mesh, "metadata", {}) or {})
                faces = tuple(tuple(int(value) for value in face[:3]) for face in tuple(mesh.faces or ()))
                face_mats = tuple(
                    int(value)
                    for value in tuple(
                        metadata.get("face_mats")
                        or metadata.get("face_material_ids")
                        or ()
                    )
                )
                if len(face_mats) != len(faces):
                    face_mats = tuple(0 for _ in faces)
                result = (
                    {
                        "room_resref": wanted_room,
                        "primitive_name": wanted_name,
                        "mesh_name": str(mesh.name or wanted_name),
                        "vertices": tuple(tuple(float(value) for value in vertex[:3]) for vertex in tuple(mesh.vertices or ())),
                        "faces": faces,
                        "normals": tuple(tuple(float(value) for value in normal[:3]) for normal in tuple(mesh.normals or ())),
                        "uvs": tuple(tuple(float(value) for value in uv[:2]) for uv in tuple(mesh.uvs or ())),
                        "uvs_lm": tuple(tuple(float(value) for value in uv[:2]) for uv in tuple(metadata.get("uvs_lm") or ())),
                        "face_mats": face_mats,
                        "texture": str(mesh.texture or ""),
                    },
                )
                room_offset = tuple(float(value) for value in tuple(room.position or (0.0, 0.0, 0.0))[:3])
                if len(room_offset) < 3:
                    room_offset = (0.0, 0.0, 0.0)
                world_vertices = tuple(
                    tuple(float(vertex[index]) + room_offset[index] for index in range(3))
                    for vertex in tuple(mesh.vertices or ())
                )
                if world_vertices:
                    bounds_min = tuple(min(vertex[index] for vertex in world_vertices) for index in range(3))
                    bounds_max = tuple(max(vertex[index] for vertex in world_vertices) for index in range(3))
                    center = tuple((bounds_min[index] + bounds_max[index]) * 0.5 for index in range(3))
                    dimensions3 = tuple(bounds_max[index] - bounds_min[index] for index in range(3))
                    self.last_map_studio_primitive_preview_overlay = build_map_studio_universal_transform_overlay(
                        SimpleNamespace(
                            room_resref=wanted_room,
                            primitive_name=wanted_name,
                            primitive_type=type(base).__name__.removesuffix("Primitive").lower(),
                            coordinate_space="kmap_world_preview",
                            bounds_min=bounds_min,
                            bounds_max=bounds_max,
                            center=center,
                            dimensions=dimensions3,
                            vertex_count=logical_vertex_count,
                            face_count=logical_face_count,
                            committed_edit_stale_outputs=("MDL", "MDX", "WOK", "LYT", "VIS", "PTH", ".mod"),
                            readiness_impact=(
                                "Committing construction inputs invalidates validation, export, install handoff, and game proof."
                            ),
                            metadata={
                                "preview_only": True,
                                "construction_recipe": True,
                                "logical_edge_count": logical_edge_count,
                                "topology_count_source": topology_count_source,
                            },
                        )
                    )
                self.last_map_studio_primitive_preview_elapsed_ms = (perf_counter() - started) * 1000.0
                self.last_map_studio_primitive_preview_identity = (wanted_room, wanted_name)
                self.last_map_studio_primitive_preview_dimensions = deepcopy(dimensions)
                return result
            break
        raise ValueError(
            f"Room {room_resref} has no authored primitive named '{primitive_name}' to preview."
        )

    def set_authored_room_primitive_style(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        texture: str = "",
        surface_id: Any = None,
    ):
        """Set material and optional WOK surface intent for one composition primitive."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = set_authored_room_composition_primitive_style(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            texture=texture,
            surface_id=surface_id,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Styled Map Studio room primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.set_style",
            label=f"Style primitive {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name, "texture": texture, "surface_id": surface_id},
        )
        return self.authored_module_readiness()

    def set_authored_room_edge_normal_policy(
        self,
        *,
        room_resref: str = "",
        policy: str,
        primitive_name: str = "",
        edge_indices: Any = None,
    ):
        """Record soft/hard edge-normal intent for authored Map Studio geometry."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = set_authored_room_edge_normal_policy(
            authored,
            room_resref=room_resref,
            policy=policy,
            primitive_name=primitive_name,
            edge_indices=edge_indices,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        normal_policy = str(policy or "").strip().lower()
        label_word = "Soften" if normal_policy.startswith("soft") else "Harden"
        self.model.log(
            f"{label_word}ed Map Studio visual edge-normal policy for {room_resref or '(first room)'}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.edge_normals.set_policy",
            label=f"{label_word} edges",
            before=before,
            metadata={
                "room_resref": room_resref,
                "policy": policy,
                "primitive_name": primitive_name,
                "edge_indices": tuple(edge_indices or ()),
            },
        )
        return self.authored_module_readiness()

    def rename_authored_room_primitive(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        new_primitive_name: str,
    ):
        """Rename one authored composition primitive in the current KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = rename_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            new_primitive_name=new_primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Renamed Map Studio room primitive {primitive_name} to {new_primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.rename",
            label=f"Rename primitive {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "new_primitive_name": new_primitive_name,
            },
        )
        return self.authored_module_readiness()

    def remove_authored_room_primitive(self, *, room_resref: str, primitive_name: str):
        """Remove one authored composition primitive from the current KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = remove_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        self.model.log(
            f"Removed Map Studio room primitive {primitive_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.remove",
            label=f"Remove primitive {primitive_name}",
            before=before,
            metadata={"room_resref": room_resref, "primitive_name": primitive_name},
        )
        return self.authored_module_readiness()

    def remove_authored_room_primitives(self, selections: Any) -> tuple[tuple[str, str], ...]:
        """Delete an object multi-selection in one KMAP/Undo transaction."""

        requested = tuple(
            dict.fromkeys(
                (str(room or "").strip(), str(name or "").strip())
                for room, name in tuple(selections or ())
                if str(room or "").strip() and str(name or "").strip()
            )
        )
        if not requested:
            return ()
        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        removed: list[tuple[str, str]] = []
        for room_resref, primitive_name in requested:
            try:
                authored = remove_authored_room_composition_primitive(
                    authored,
                    room_resref=room_resref,
                    primitive_name=primitive_name,
                )
            except Exception as exc:
                self.model.log(f"Map Studio object {room_resref}/{primitive_name} could not be removed: {exc}")
                continue
            removed.append((room_resref, primitive_name))
        if not removed:
            return ()
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(authored)
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        self._record_map_studio_command(
            action_key="map_studio.primitive.remove_many",
            label=f"Remove {len(removed)} objects",
            before=before,
            metadata={"selections": removed},
        )
        self.model.log(f"Removed {len(removed)} selected Map Studio objects; previous exports/proofs are now stale.")
        return tuple(removed)

    def remove_map_studio_scene_objects(
        self,
        *,
        primitive_selections: Any = (),
        placement_ids: Any = (),
        room_resrefs: Any = (),
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        """Delete a mixed viewport/Outliner selection as one undoable edit."""

        primitives = tuple(
            dict.fromkeys(
                (str(room or "").strip(), str(name or "").strip())
                for room, name in tuple(primitive_selections or ())
                if str(room or "").strip() and str(name or "").strip()
            )
        )
        placements = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in tuple(placement_ids or ())
                if str(value or "").strip()
            )
        )
        rooms = tuple(
            dict.fromkeys(
                normalise_resref(value)
                for value in tuple(room_resrefs or ())
                if str(value or "").strip()
            )
        )
        if not primitives and not placements and not rooms:
            return (), ()
        for placement_id in placements:
            if placement_id != "entry_point":
                parse_authored_gameplay_placement_id(placement_id)
        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        removed_rooms: list[str] = []
        if rooms:
            retained_rooms = []
            for room in tuple(authored.rooms or ()):
                resref = room.normalised_resref()
                if resref in rooms:
                    removed_rooms.append(resref)
                else:
                    retained_rooms.append(room)
            if removed_rooms:
                removed_set = set(removed_rooms)
                retained_rooms = [
                    replace(
                        room,
                        visible_rooms=tuple(
                            target
                            for target in tuple(room.visible_rooms or ())
                            if normalise_resref(target) not in removed_set
                        ),
                    )
                    for room in retained_rooms
                ]
                updated_extra = dict(getattr(authored, "extra", {}) or {})
                updated_extra["vis_pairs"] = [
                    pair
                    for pair in list(updated_extra.get("vis_pairs") or ())
                    if len(tuple(pair or ())) >= 2
                    and normalise_resref(tuple(pair)[0]) not in removed_set
                    and normalise_resref(tuple(pair)[1]) not in removed_set
                ]
                authored = replace(authored, rooms=tuple(retained_rooms), extra=updated_extra)
        removed_primitives: list[tuple[str, str]] = []
        for room_resref, primitive_name in primitives:
            try:
                authored = remove_authored_room_composition_primitive(
                    authored,
                    room_resref=room_resref,
                    primitive_name=primitive_name,
                )
            except Exception as exc:
                self.model.log(f"Map Studio object {room_resref}/{primitive_name} could not be removed: {exc}")
                continue
            removed_primitives.append((room_resref, primitive_name))
        removed_placements: list[str] = []
        for placement_id in placements:
            try:
                if placement_id == "entry_point":
                    if not str(authored.placements.entry_point.area_resref or "").strip():
                        continue
                    authored = remove_authored_module_entry_point(authored).project
                else:
                    authored = remove_authored_gameplay_placement(authored, placement_id).project
            except Exception as exc:
                self.model.log(f"Map Studio placement {placement_id} could not be removed: {exc}")
                continue
            removed_placements.append(placement_id)
        if not removed_rooms and not removed_primitives and not removed_placements:
            return (), ()
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(authored)
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        removed_count = len(removed_rooms) + len(removed_primitives) + len(removed_placements)
        self._record_map_studio_command(
            action_key="map_studio.scene.remove_objects",
            label=f"Remove {removed_count} scene objects",
            before=before,
            metadata={
                "primitive_selections": tuple(removed_primitives),
                "placement_ids": tuple(removed_placements),
                "room_resrefs": tuple(removed_rooms),
            },
        )
        self.model.log(
            f"Removed {removed_count} mixed Map Studio scene objects; previous exports/proofs are now stale."
        )
        return tuple(removed_primitives), tuple(removed_placements) + tuple(
            f"authored_room:{value}" for value in removed_rooms
        )

    def separate_authored_room_primitive(
        self,
        *,
        room_resref: str,
        primitive_name: str,
        result_room_resref: str = "",
    ):
        """Separate one authored composition primitive into a new room/object boundary."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        updated = separate_authored_room_composition_primitive(
            authored,
            room_resref=room_resref,
            primitive_name=primitive_name,
            result_room_resref=result_room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(updated)
        self.project.name = updated.metadata.module_root
        self.project.game = updated.game
        self.project.dirty = True
        separated_room = str(updated.extra.get("last_room_operation", "")).rsplit(":", 1)[-1]
        self.model.log(
            f"Separated Map Studio room primitive {primitive_name} into {separated_room}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.primitive.separate",
            label=f"Separate primitive {primitive_name}",
            before=before,
            metadata={
                "room_resref": room_resref,
                "primitive_name": primitive_name,
                "result_room_resref": result_room_resref,
                "separated_room": separated_room,
            },
        )
        return self.authored_module_readiness()

    def apply_authored_room_style(self, *, texture: str = "", floor_surface: Any = 4, room_resref: str = ""):
        """Apply room texture and WOK surface intent to the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_room_style(
            authored,
            texture=texture,
            floor_surface=floor_surface,
            room_resref=room_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Applied Map Studio room style texture {update.texture}, surface {update.floor_surface_id} ({update.floor_surface_name})."
        )
        for warning in update.warnings:
            self.model.log(f"Warning: {warning}")
        self._record_map_studio_command(
            action_key="map_studio.room.set_style",
            label=f"Style {room_resref or 'room'}",
            before=before,
            metadata={"texture": texture, "floor_surface": floor_surface, "room_resref": room_resref},
        )
        return self.authored_module_readiness()

    def set_authored_room_walkmesh_surface(self, *, room_resref: str, floor_surface: Any = 4):
        """Apply only the room WOK surface from Walkmesh tools, preserving texture intent."""

        target = normalise_resref(room_resref)
        texture = ""
        for choice in self.authored_walkmesh_room_surface_choices():
            if normalise_resref(getattr(choice, "room_resref", "")) == target:
                texture = str(getattr(choice, "texture", "") or "")
                break
        return self.apply_authored_room_style(texture=texture, floor_surface=floor_surface, room_resref=room_resref)

    def map_studio_room_walkmesh_bytes(self, room_resref: str) -> bytes | None:
        """Compile one authored room and return its .wok bytes (for Save WOK).

        Uses the same geometry compile the exporter uses, so the saved file is
        exactly what would ship in the module.
        """

        target = normalise_resref(room_resref)
        if not target:
            return None
        authored = self._load_authored_project_or_raise()
        room = next(
            (r for r in authored.rooms if r.normalised_resref() == target),
            None,
        )
        if room is None:
            return None
        geometry = compile_authored_room_spec(room)
        wok = getattr(geometry, "wok", None)
        if wok is None or not getattr(wok, "faces", None):
            return None
        to_bytes = getattr(wok, "to_bytes", None)
        return to_bytes() if callable(to_bytes) else None

    def add_authored_gameplay_placement(
        self,
        *,
        kind: str,
        template_resref: str = "",
        tag: str = "",
        position: Any = (0.0, 0.0, 0.0),
        bearing: float = 0.0,
        snap_to_walkmesh: bool = False,
        provenance: dict[str, Any] | None = None,
    ):
        """Append a gameplay object placement to the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = add_authored_gameplay_placement(
            authored,
            kind=kind,
            template_resref=template_resref,
            tag=tag,
            position=position,
            bearing=bearing,
            provenance=provenance,
        )
        snap = None
        if bool(snap_to_walkmesh) and update.kind != "store":
            update, snap = snap_authored_gameplay_placement_to_walkmesh(update.project, update.placement_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Added Map Studio {update.kind} placement {update.tag} at {update.position}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.add_placement",
            label=f"Add {update.kind} placement {update.tag}",
            before=before,
            metadata={
                "kind": update.kind,
                "template_resref": update.template_resref,
                "tag": update.tag,
                "position": update.position,
                "placement_id": update.placement_id,
                "provenance": dict(provenance or {}),
                "snapped_to_walkmesh": snap is not None,
                "walkmesh_face_index": snap.face_index if snap is not None else -1,
            },
        )
        return self.authored_module_readiness()

    def set_authored_module_entry_point(
        self,
        *,
        area_resref: str = "",
        position: Any = (0.0, 0.0, 0.0),
        facing: float = 0.0,
    ):
        """Edit the module IFO entry point/player start for the authored KMAP."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        update = update_authored_module_entry_point(
            authored,
            area_resref=area_resref,
            position=position,
            facing=facing,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Updated Map Studio module entry point {update.area_resref} at {update.position}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.set_entry_point",
            label=f"Set entry point {update.area_resref}",
            before=before,
            metadata={
                "area_resref": update.area_resref,
                "position": update.position,
                "facing": update.facing,
            },
        )
        return self.authored_module_readiness()

    def clear_authored_module_entry_point(self):
        """Remove the selectable IFO player start as one undoable command."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        if not str(authored.placements.entry_point.area_resref or "").strip():
            return self.authored_module_readiness()
        update = remove_authored_module_entry_point(authored)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log("Removed Map Studio module entry point/player start; export is blocked until a new start is set.")
        self._record_map_studio_command(
            action_key="map_studio.gameplay.remove_entry_point",
            label="Remove player start",
            before=before,
            metadata={
                "position": update.position,
                "facing": update.facing,
            },
        )
        return self.authored_module_readiness()

    def add_authored_room_light(
        self,
        *,
        room_resref: str = "",
        name: str = "",
        position: Any = (0.0, 0.0, 2.25),
        color: Any = (1.0, 0.92, 0.78),
        radius: float = 8.0,
        intensity: float = 1.0,
        light_type: str = "point",
    ):
        """Append room-light authoring intent to the current authored KMAP module."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = add_authored_room_light_to_project(
            authored,
            room_resref=room_resref,
            name=name,
            position=position,
            color=color,
            radius=radius,
            intensity=intensity,
            light_type=light_type,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Added Map Studio {update.light.light_type} room light {update.light.name} in {update.light.room_resref}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.add_room_light",
            label=f"Add room light {update.light.name}",
            before=before,
            metadata={
                "light_id": update.light_id,
                "room_resref": update.light.room_resref,
                "name": update.light.name,
                "position": update.light.position,
                "color": update.light.color,
                "radius": update.light.radius,
                "intensity": update.light.intensity,
                "light_type": update.light.light_type,
            },
        )
        return self.authored_module_readiness()

    def set_authored_gameplay_placement_transform(
        self,
        placement_id: str,
        *,
        position: Any,
        bearing: float | None = None,
        snap_to_walkmesh: bool = False,
    ):
        """Move one authored gameplay placement by virtual id."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_gameplay_placement_transform(
            authored,
            placement_id,
            position=position,
            bearing=bearing,
        )
        snap = None
        if bool(snap_to_walkmesh):
            update, snap = snap_authored_gameplay_placement_to_walkmesh(update.project, placement_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Moved Map Studio {update.kind} placement {update.tag} to {update.position}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.move_placement",
            label=f"Move {update.kind} placement {update.tag}",
            before=before,
            metadata={
                "placement_id": placement_id,
                "kind": update.kind,
                "tag": update.tag,
                "position": update.position,
                "bearing": bearing,
                "snapped_to_walkmesh": snap is not None,
                "walkmesh_face_index": snap.face_index if snap is not None else -1,
            },
        )
        # Lazy: drag/property commits ignore this and the window's deferred
        # validation worker refreshes export gates off the Qt thread.
        return DeferredAuthoredModuleReadiness(self)

    def translate_authored_gameplay_placements(self, placement_ids, *, world_delta: Any) -> tuple[str, ...]:
        """Move a Maya-style multi-selection as one KMAP/Undo transaction."""

        requested = tuple(dict.fromkeys(str(value or "").strip() for value in tuple(placement_ids or ()) if str(value or "").strip()))
        delta = tuple(float(value) for value in tuple(world_delta or ())[:3])
        if not requested or len(delta) != 3:
            return ()
        for placement_id in requested:
            if placement_id != "entry_point":
                parse_authored_gameplay_placement_id(placement_id)
        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        rows = {row.placement_id: row for row in authored_gameplay_placement_rows(authored)}
        before = self._capture_map_studio_command_state()
        moved: list[str] = []
        for placement_id in requested:
            if placement_id == "entry_point":
                entry = authored.placements.entry_point
                if not str(entry.area_resref or "").strip():
                    continue
                position = tuple(float(entry.position[index]) + delta[index] for index in range(3))
                update = update_authored_module_entry_point(
                    authored,
                    area_resref=str(entry.area_resref),
                    position=position,
                    facing=float(entry.facing),
                )
                authored = update.project
                moved.append(placement_id)
                continue
            row = rows.get(placement_id)
            if row is None or not bool(row.is_spatial):
                continue
            position = tuple(float(row.position[index]) + delta[index] for index in range(3))
            update = update_authored_gameplay_placement_transform(
                authored,
                placement_id,
                position=position,
                bearing=float(row.bearing),
            )
            authored = update.project
            moved.append(placement_id)
        if not moved:
            return ()
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(authored)
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        self._record_map_studio_command(
            action_key="map_studio.gameplay.move_placements",
            label=f"Move {len(moved)} placements",
            before=before,
            metadata={"placement_ids": moved, "world_delta": delta},
        )
        self.model.log(f"Moved {len(moved)} Map Studio objects together; previous exports/proofs are now stale.")
        return tuple(moved)

    def translate_map_studio_scene_objects(
        self,
        *,
        primitive_selections: Any = (),
        placement_ids: Any = (),
        room_resrefs: Any = (),
        world_delta: Any = (0.0, 0.0, 0.0),
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        """Move a mixed authored-object selection in one KMAP/Undo command."""

        primitives = tuple(
            dict.fromkeys(
                (str(room or "").strip(), str(name or "").strip())
                for room, name in tuple(primitive_selections or ())
                if str(room or "").strip() and str(name or "").strip()
            )
        )
        placements = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in tuple(placement_ids or ())
                if str(value or "").strip()
            )
        )
        rooms = tuple(
            dict.fromkeys(
                str(value or "").strip().lower()
                for value in tuple(room_resrefs or ())
                if str(value or "").strip()
            )
        )
        delta = tuple(float(value) for value in tuple(world_delta or ())[:3])
        if (not primitives and not placements and not rooms) or len(delta) != 3:
            return (), ()
        for placement_id in placements:
            if placement_id != "entry_point":
                parse_authored_gameplay_placement_id(placement_id)
        payload = (getattr(self.project, "extra_sections", {}) or {}).get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        if rooms:
            owned_door_ids: list[str] = []
            for room in tuple(authored.rooms or ()):
                normalise = getattr(room, "normalised_resref", None)
                resref = str(
                    normalise() if callable(normalise) else getattr(room, "room_resref", "") or ""
                ).strip().lower()
                if resref not in rooms:
                    continue
                primitive = getattr(room, "primitive", None)
                for opening in tuple(getattr(primitive, "openings", ()) or ()):
                    owned_id = str(
                        dict(getattr(opening, "metadata", {}) or {}).get("door_placement_id") or ""
                    ).strip()
                    if owned_id:
                        parse_authored_gameplay_placement_id(owned_id)
                        owned_door_ids.append(owned_id)
            placements = tuple(dict.fromkeys((*placements, *owned_door_ids)))
        before = self._capture_map_studio_command_state()
        moved_rooms: list[str] = []
        if rooms:
            updated_rooms = []
            for room in tuple(authored.rooms or ()):
                normalise = getattr(room, "normalised_resref", None)
                resref = str(normalise() if callable(normalise) else getattr(room, "room_resref", "") or "").strip().lower()
                if resref in rooms:
                    origin = tuple(float(value) for value in tuple(getattr(room, "position", (0.0, 0.0, 0.0)))[:3])
                    room = replace(
                        room,
                        position=tuple(origin[index] + delta[index] for index in range(3)),
                    )
                    moved_rooms.append(resref)
                updated_rooms.append(room)
            authored = replace(authored, rooms=tuple(updated_rooms))
        moved_primitives: tuple[tuple[str, str], ...] = ()
        if primitives:
            authored = transform_authored_room_composition_primitives(
                authored,
                selections=primitives,
                mode="translate",
                world_delta=delta,
                world_pivot=(0.0, 0.0, 0.0),
            )
            moved_primitives = primitives
        rows = {row.placement_id: row for row in authored_gameplay_placement_rows(authored)}
        moved_placements: list[str] = []
        for placement_id in placements:
            if placement_id == "entry_point":
                entry = authored.placements.entry_point
                if not str(entry.area_resref or "").strip():
                    continue
                authored = update_authored_module_entry_point(
                    authored,
                    area_resref=str(entry.area_resref),
                    position=tuple(float(entry.position[index]) + delta[index] for index in range(3)),
                    facing=float(entry.facing),
                ).project
                moved_placements.append(placement_id)
                continue
            row = rows.get(placement_id)
            if row is None or not bool(row.is_spatial):
                continue
            authored = update_authored_gameplay_placement_transform(
                authored,
                placement_id,
                position=tuple(float(row.position[index]) + delta[index] for index in range(3)),
                bearing=float(row.bearing),
            ).project
            moved_placements.append(placement_id)
        if not moved_rooms and not moved_primitives and not moved_placements:
            return (), ()
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(authored)
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        moved_count = len(moved_rooms) + len(moved_primitives) + len(moved_placements)
        self._record_map_studio_command(
            action_key="map_studio.scene.move_objects",
            label=f"Move {moved_count} scene objects",
            before=before,
            metadata={
                "primitive_selections": moved_primitives,
                "placement_ids": tuple(moved_placements),
                "room_resrefs": tuple(moved_rooms),
                "world_delta": delta,
            },
        )
        self.model.log(
            f"Moved {moved_count} mixed Map Studio scene objects together; previous exports/proofs are now stale."
        )
        return moved_primitives, tuple(moved_placements) + tuple(f"authored_room:{value}" for value in moved_rooms)

    def rename_authored_gameplay_placement(self, placement_id: str, *, tag: Any):
        """Rename one authored gameplay placement by virtual id."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = rename_authored_gameplay_placement(authored, placement_id, tag=tag)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Renamed Map Studio {update.kind} placement to {update.tag}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.rename_placement",
            label=f"Rename {update.kind} placement {update.tag}",
            before=before,
            metadata={"placement_id": placement_id, "kind": update.kind, "tag": update.tag},
        )
        return update

    def set_authored_creature_behavior(
        self,
        placement_id: str,
        *,
        faction_role: Any,
        conversation_resref: Any = "",
        movement_mode: Any = "stationary",
    ):
        """Author selected-creature UTC behavior while keeping source resources immutable."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_creature_behavior(
            authored,
            placement_id,
            faction_role=faction_role,
            conversation_resref=conversation_resref,
            movement_mode=movement_mode,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self._authored_creature_resources = ()
        role = str(faction_role or "template").strip().lower()
        movement = str(movement_mode or "stationary").strip().lower()
        self.model.log(
            f"Updated Map Studio creature {update.tag}: {role}, {movement}; target-game UTC resources will be rebuilt at export."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.edit_creature_behavior",
            label=f"Edit creature behavior {update.tag}",
            before=before,
            stale_outputs=("GIT", "UTC", "NCS", ".mod"),
            readiness_impact="Generated creature templates/scripts must be rebuilt and manually proven in plcaa.",
            metadata={
                "placement_id": placement_id,
                "tag": update.tag,
                "faction_role": role,
                "conversation_resref": str(conversation_resref or "").strip().lower(),
                "movement_mode": movement,
                "template_resref": update.template_resref,
            },
        )
        return update

    def duplicate_authored_gameplay_placement(self, placement_id: str):
        """Duplicate one authored gameplay placement by virtual id."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = duplicate_authored_gameplay_placement(authored, placement_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Duplicated Map Studio {update.kind} placement {update.tag}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.duplicate_placement",
            label=f"Duplicate {update.kind} placement {update.tag}",
            before=before,
            metadata={
                "source_placement_id": placement_id,
                "placement_id": update.placement_id,
                "kind": update.kind,
                "tag": update.tag,
            },
        )
        return update

    def remove_authored_gameplay_placement(self, placement_id: str):
        """Remove one authored gameplay placement by virtual id."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = remove_authored_gameplay_placement(authored, placement_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Removed Map Studio {update.kind} placement {update.tag}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.remove_placement",
            label=f"Remove {update.kind} placement {update.tag}",
            before=before,
            metadata={"placement_id": placement_id, "kind": update.kind, "tag": update.tag},
        )
        return update

    def remove_authored_gameplay_placements(self, placement_ids) -> tuple[str, ...]:
        """Remove several placements (marquee delete) as ONE undoable command."""

        requested = [str(value or "").strip() for value in tuple(placement_ids or ()) if str(value or "").strip()]
        if not requested:
            return ()
        for placement_id in requested:
            parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        removed: list[str] = []
        labels: list[str] = []
        for placement_id in requested:
            try:
                update = remove_authored_gameplay_placement(authored, placement_id)
            except Exception as exc:
                self.model.log(f"Map Studio placement {placement_id} could not be removed: {exc}")
                continue
            authored = update.project
            removed.append(placement_id)
            labels.append(f"{update.kind} {update.tag}")
        if not removed:
            return ()
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(authored)
        self.project.name = authored.metadata.module_root
        self.project.game = authored.game
        self.project.dirty = True
        summary = ", ".join(labels[:4]) + (" ..." if len(labels) > 4 else "")
        self.model.log(
            f"Removed {len(removed)} Map Studio placement(s): {summary}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.remove_placements",
            label=f"Remove {len(removed)} placements",
            before=before,
            metadata={"placement_ids": list(removed), "labels": labels},
        )
        return tuple(removed)

    def set_authored_gameplay_camera_properties(
        self,
        placement_id: str,
        *,
        camera_id: Any | None = None,
        field_of_view: Any | None = None,
        height: Any | None = None,
        mic_range: Any | None = None,
        pitch: Any | None = None,
    ):
        """Edit selected authored camera GIT CameraList fields from Properties."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_gameplay_camera_properties(
            authored,
            placement_id,
            camera_id=camera_id,
            field_of_view=field_of_view,
            height=height,
            mic_range=mic_range,
            pitch=pitch,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Edited Map Studio camera {update.tag}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.edit_camera",
            label=f"Edit camera {update.tag}",
            before=before,
            metadata={
                "placement_id": placement_id,
                "tag": update.tag,
                "camera_id": camera_id,
                "field_of_view": field_of_view,
                "height": height,
                "mic_range": mic_range,
                "pitch": pitch,
            },
        )
        # Lazy: drag/property commits ignore this and the window's deferred
        # validation worker refreshes export gates off the Qt thread.
        return DeferredAuthoredModuleReadiness(self)

    def set_authored_gameplay_transition(
        self,
        placement_id: str,
        *,
        linked_to: Any = "",
        linked_to_module: Any = "",
        linked_to_flags: Any = 0,
        transition_destination: Any = 0,
    ):
        """Set transition destination fields on a selected authored door or trigger."""

        parse_authored_gameplay_placement_id(placement_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_gameplay_transition(
            authored,
            placement_id,
            linked_to=linked_to,
            linked_to_module=linked_to_module,
            linked_to_flags=linked_to_flags,
            transition_destination=transition_destination,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Updated Map Studio {update.kind} transition for {update.tag}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.gameplay.set_transition",
            label=f"Set {update.kind} transition {update.tag}",
            before=before,
            metadata={
                "placement_id": placement_id,
                "kind": update.kind,
                "tag": update.tag,
                "linked_to": linked_to,
                "linked_to_module": linked_to_module,
                "linked_to_flags": linked_to_flags,
                "transition_destination": transition_destination,
            },
        )
        # Lazy: drag/property commits ignore this and the window's deferred
        # validation worker refreshes export gates off the Qt thread.
        return DeferredAuthoredModuleReadiness(self)

    def set_authored_room_light_transform(self, light_id: str, *, position: Any):
        """Move one authored room light by virtual id."""

        parse_authored_room_light_id(light_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = update_authored_room_light_transform(authored, light_id, position=position)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Moved Map Studio room light {update.light.name} to {update.light.position}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.move_room_light",
            label=f"Move room light {update.light.name}",
            before=before,
            metadata={"light_id": light_id, "name": update.light.name, "position": update.light.position},
        )
        # Lazy: drag/property commits ignore this and the window's deferred
        # validation worker refreshes export gates off the Qt thread.
        return DeferredAuthoredModuleReadiness(self)

    def set_authored_room_light_properties(
        self,
        light_id: str,
        *,
        color: Any | None = None,
        radius: float | None = None,
        intensity: float | None = None,
        light_type: Any | None = None,
        enabled: Any | None = None,
        casts_shadows: Any | None = None,
        affects_diffuse: Any | None = None,
        affects_lightmap: Any | None = None,
        direction: Any | None = None,
        cone_angle_degrees: float | None = None,
        bake_group: Any = ...,
    ):
        """Edit selected authored room-light settings from the properties panel."""

        parse_authored_room_light_id(light_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        room_light_updates = {
            "color": color,
            "radius": radius,
            "intensity": intensity,
            "light_type": light_type,
            "enabled": enabled,
            "casts_shadows": casts_shadows,
            "affects_diffuse": affects_diffuse,
            "affects_lightmap": affects_lightmap,
            "direction": direction,
            "cone_angle_degrees": cone_angle_degrees,
        }
        if bake_group is not ...:
            room_light_updates["bake_group"] = bake_group
        update = update_authored_room_light_properties(
            authored,
            light_id,
            **room_light_updates,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Edited Map Studio room light {update.light.name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.edit_room_light",
            label=f"Edit room light {update.light.name}",
            before=before,
            metadata={
                "light_id": light_id,
                "name": update.light.name,
                "color": update.light.color,
                "radius": update.light.radius,
                "intensity": update.light.intensity,
                "light_type": update.light.light_type,
                "enabled": update.light.enabled,
                "casts_shadows": update.light.casts_shadows,
                "affects_diffuse": update.light.affects_diffuse,
                "affects_lightmap": update.light.affects_lightmap,
                "direction": update.light.direction,
                "cone_angle_degrees": update.light.cone_angle_degrees,
                "bake_group": update.light.bake_group,
            },
        )
        # Lazy: drag/property commits ignore this and the window's deferred
        # validation worker refreshes export gates off the Qt thread.
        return DeferredAuthoredModuleReadiness(self)

    def rename_authored_room_light(self, light_id: str, *, name: Any):
        """Rename one authored room light by virtual id."""

        parse_authored_room_light_id(light_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = rename_authored_room_light(authored, light_id, name=name)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Renamed Map Studio room light to {update.light.name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.rename_room_light",
            label=f"Rename room light {update.light.name}",
            before=before,
            metadata={"light_id": light_id, "name": update.light.name},
        )
        return update

    def duplicate_authored_room_light(self, light_id: str):
        """Duplicate one authored room light by virtual id."""

        parse_authored_room_light_id(light_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = duplicate_authored_room_light(authored, light_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Duplicated Map Studio room light {update.light.name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.duplicate_room_light",
            label=f"Duplicate room light {update.light.name}",
            before=before,
            metadata={"source_light_id": light_id, "light_id": update.light_id, "name": update.light.name},
        )
        return update

    def remove_authored_room_light(self, light_id: str):
        """Remove one authored room light by virtual id."""

        parse_authored_room_light_id(light_id)
        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = remove_authored_room_light(authored, light_id)
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Removed Map Studio room light {update.light.name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.lighting.remove_room_light",
            label=f"Remove room light {update.light.name}",
            before=before,
            metadata={"light_id": light_id, "name": update.light.name},
        )
        return update

    def set_authored_script_hook(self, *, scope: Any, field_name: Any, script_resref: Any):
        """Assign one authored module/area script hook in the current KMAP payload."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = set_authored_script_hook(
            authored,
            scope=scope,
            field_name=field_name,
            script_resref=script_resref,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Assigned Map Studio {update.scope} script hook {update.field_name} -> {update.script_resref}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.script.set_hook",
            label=f"Set {update.scope} script {update.field_name}",
            before=before,
            metadata={
                "scope": update.scope,
                "field_name": update.field_name,
                "script_resref": update.script_resref,
            },
        )
        return update

    def remove_authored_script_hook(self, *, scope: Any, field_name: Any):
        """Clear one authored module/area script hook in the current KMAP payload."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        before = self._capture_map_studio_command_state()
        update = remove_authored_script_hook(
            authored,
            scope=scope,
            field_name=field_name,
        )
        self.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(update.project)
        self.project.name = update.project.metadata.module_root
        self.project.game = update.project.game
        self.project.dirty = True
        self.model.log(
            f"Cleared Map Studio {update.scope} script hook {update.field_name}; previous exports/proofs are now stale."
        )
        self._record_map_studio_command(
            action_key="map_studio.script.remove_hook",
            label=f"Clear {update.scope} script {update.field_name}",
            before=before,
            metadata={
                "scope": update.scope,
                "field_name": update.field_name,
                "script_resref": update.script_resref,
            },
        )
        return update

    def build_preview(self, output_dir: str | Path):
        return self.builder_service.build_preview(self.project, output_dir)

    def generate_module_files(self, output_dir: str | Path):
        """Generate module files for the current project and record authored export state."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        if extra.get("authored_module") is not None:
            return self.export_authored_module(output_dir, dry_run=False)
        return self.build_preview(output_dir)

    def stage_dev_test_module(self, output_dir: str | Path, *, dry_run: bool = False, overwrite: bool = False):
        """Stage the first from-scratch Map Studio smoke module package."""

        self._require_applied_project_texture_changes()
        output_path = Path(output_dir)
        game = str(self.project.game or "K1").upper()
        return prepare_dev_test_module_install(
            DevModuleInstallPrepRequest(
                output_dir=str(output_path),
                game=game,
                dry_run=dry_run,
                overwrite=overwrite,
                smoke_request=DevModuleSmokeRequest(
                    output_dir=str(output_path),
                    game=game,
                ),
            )
        )

    def export_authored_module(self, output_dir: str | Path, *, dry_run: bool = False, overwrite: bool = False):
        """Export the authored module currently stored in the KMAP project."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        self._require_applied_project_texture_changes()
        self._require_authored_placeable_resources_ready()
        self._require_authored_creature_resources_ready()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        output_path = Path(output_dir)
        extra_resources = self.authored_project_extra_resources()
        result = export_authored_module_project(
            AuthoredModuleExportRequest(
                project=authored,
                output_dir=str(output_path),
                dry_run=dry_run,
                strict=not dry_run,
                extra_resources=extra_resources,
            )
        )
        if not dry_run and result.resources:
            runtime_resources = [f"{item.resref}.{item.restype}" for item in result.resources]
            payload = dict(payload)
            self._clear_authored_export_proof_invalidation(payload, clear_stage_metadata=True)
            payload["runtime_resources"] = runtime_resources
            payload["pack_manifest_path"] = result.manifest_path
            export_job = dict((result.metadata or {}).get("export_job") or {})
            if export_job:
                payload["export_job"] = export_job
            manifest = _read_json_object(result.manifest_path)
            authored_manifest = manifest.get("map_studio_authored_module")
            if isinstance(authored_manifest, dict):
                manifest_export_job = authored_manifest.get("export_job")
                if isinstance(manifest_export_job, dict):
                    payload["export_job"] = manifest_export_job
                test_plan = authored_manifest.get("modder_test_plan")
                if isinstance(test_plan, dict):
                    payload["modder_test_plan"] = test_plan
            self.project.extra_sections["authored_module"] = payload
            self.project.dirty = True
        self.model.log(result.message)
        return result

    def stage_authored_module(
        self,
        output_dir: str | Path,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
        game_modules_dir: str | Path = "",
        auto_detect_game_modules_dir: bool = False,
    ):
        """Stage the current authored KMAP module with a manual game-test checklist."""

        extra = getattr(self.project, "extra_sections", {}) or {}
        payload = extra.get("authored_module")
        if payload is None:
            raise ValueError("No authored Map Studio module is stored in this KMAP. Create or load an authored module first.")
        self._require_applied_project_texture_changes()
        self._require_authored_placeable_resources_ready()
        self._require_authored_creature_resources_ready()
        before = self._capture_map_studio_command_state()
        authored = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(self.project, "name", "") or "new_level"),
            fallback_game=str(getattr(self.project, "game", "") or "K1"),
        )
        output_path = Path(output_dir)
        extra_resources = self.authored_project_extra_resources()
        result = prepare_authored_module_install(
            AuthoredModuleInstallPrepRequest(
                project=authored,
                output_dir=str(output_path),
                game_modules_dir=str(game_modules_dir or ""),
                dry_run=dry_run,
                overwrite=overwrite,
                auto_detect_game_modules_dir=bool(auto_detect_game_modules_dir),
                export_request=AuthoredModuleExportRequest(
                    project=authored,
                    output_dir=str(output_path),
                    strict=True,
                    extra_resources=extra_resources,
                ),
            )
        )
        export_result = result.export_result
        recorded_package_metadata = False
        if export_result is not None and export_result.resources:
            runtime_resources = [f"{item.resref}.{item.restype}" for item in export_result.resources]
            payload = dict(payload)
            self._clear_authored_export_proof_invalidation(payload, clear_stage_metadata=True)
            payload["runtime_resources"] = runtime_resources
            payload["proof_manifest_path"] = result.proof_manifest_path
            payload["checklist_path"] = result.checklist_path
            payload["resolved_modules_dir"] = result.resolved_modules_dir
            payload["resolved_game_root_dir"] = result.resolved_game_root_dir
            payload["launch_helper_command"] = result.launch_helper_command
            payload["elevated_launch_script_path"] = result.elevated_launch_script_path
            payload["proof_recording_script_path"] = result.proof_recording_script_path
            payload["installed_module_path"] = result.installed_module_path
            payload["backup_module_path"] = result.backup_module_path
            payload["pack_manifest_path"] = export_result.manifest_path
            proof_manifest = _read_json_object(result.proof_manifest_path)
            export_job = proof_manifest.get("export_job")
            if isinstance(export_job, dict):
                payload["export_job"] = export_job
            test_plan = proof_manifest.get("modder_test_plan")
            if isinstance(test_plan, dict):
                payload["modder_test_plan"] = test_plan
            package_inventory = _map_studio_package_resource_inventory(result.proof_manifest_path, payload)
            if package_inventory:
                payload["package_resource_inventory"] = package_inventory
            self.project.extra_sections["authored_module"] = payload
            self.project.dirty = True
            recorded_package_metadata = True
        if recorded_package_metadata:
            module_root = str(payload.get("module_root") or getattr(self.project, "name", "") or "new_level")
            install_requested = bool(str(game_modules_dir or "").strip() or auto_detect_game_modules_dir)
            label = f"{'Install test module' if install_requested else 'Stage authored module'} {module_root}"
            self._record_map_studio_command(
                action_key=(
                    "map_studio.export.install_module"
                    if install_requested
                    else "map_studio.export.stage_module"
                ),
                label=label,
                before=before,
                stale_outputs=(),
                readiness_impact=(
                    "Map Studio package/proof metadata changed; geometry, WOK, LYT/VIS/PTH resources are unchanged."
                ),
                summary=(
                    "Recorded staged package, install handoff, checklist, and proof-manifest metadata in KMAP."
                ),
                metadata={
                    "module_root": module_root,
                    "dry_run": bool(dry_run),
                    "installed": install_requested,
                    "proof_manifest_path": result.proof_manifest_path,
                    "pack_manifest_path": getattr(export_result, "manifest_path", ""),
                },
            )
        self.model.log(result.message)
        return result

    def map_studio_launch_handoff(self) -> MapStudioLaunchHandoffSummary:
        """Describe the current manual KOTOR launch/proof handoff without mutating project state."""

        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        module_root = str(payload.get("module_root") or getattr(self.project, "name", "") or "").strip()
        game = str(payload.get("game") or getattr(self.project, "game", "") or "K1").strip().upper()
        warp_command = str(payload.get("warp_command") or (f"warp {module_root}" if module_root else "warp <module>")).strip()
        proof_manifest_path = str(payload.get("proof_manifest_path") or "").strip()
        proof_recording_script_path = str(payload.get("proof_recording_script_path") or "").strip()
        launch_helper_command = str(payload.get("launch_helper_command") or "").strip()
        elevated_launch_script_path = str(payload.get("elevated_launch_script_path") or "").strip()
        installed_module_path = str(payload.get("installed_module_path") or "").strip()
        checklist_path = str(payload.get("checklist_path") or "").strip()
        resolved_modules_dir = str(payload.get("resolved_modules_dir") or "").strip()
        resolved_game_root_dir = str(payload.get("resolved_game_root_dir") or "").strip()
        package_resource_inventory = _map_studio_package_resource_inventory(proof_manifest_path, payload)
        package_resource_summary = _map_studio_package_resource_summary(package_resource_inventory)

        blocking: list[str] = []
        warnings: list[str] = []
        if not payload:
            blocking.append("No authored Map Studio module has been staged for a launch handoff.")
        if proof_manifest_path:
            if not Path(proof_manifest_path).is_file():
                blocking.append(f"Proof manifest does not exist: {proof_manifest_path}")
        elif not elevated_launch_script_path:
            blocking.append("Stage or install the authored module before opening the launch handoff.")
        if elevated_launch_script_path and not Path(elevated_launch_script_path).is_file():
            warnings.append(f"Launch script is not available on disk: {elevated_launch_script_path}")
        if proof_recording_script_path and not Path(proof_recording_script_path).is_file():
            warnings.append(f"Proof recorder is not available on disk: {proof_recording_script_path}")
        if checklist_path and not Path(checklist_path).is_file():
            warnings.append(f"Game-test checklist is not available on disk: {checklist_path}")
        if installed_module_path and not Path(installed_module_path).exists():
            warnings.append(f"Installed module candidate is not present on disk: {installed_module_path}")
        if not module_root:
            warnings.append("Module root is unknown; the warp command must be checked before launch.")
        if not launch_helper_command:
            warnings.append("No CLI launch helper command was recorded; use the proof manifest and warp command manually.")
        if not package_resource_inventory:
            warnings.append("Package resource inventory is missing; stage or install the authored module again before game proof.")

        ready = len(blocking) == 0
        summary = (
            f"Launch handoff ready for {game}:{module_root or '<module>'}; run {warp_command} and record proof."
            if ready
            else "Launch handoff is not ready; stage or install an authored module package first."
        )
        next_action = (
            "Open the launcher/proof folder, run the exact KOTOR warp command, then record screenshot or video proof."
            if ready
            else "Stage .mod or Install Test from Map Studio export tools before launching."
        )
        return MapStudioLaunchHandoffSummary(
            ready=ready,
            module_root=module_root,
            game=game,
            warp_command=warp_command,
            launcher_path=elevated_launch_script_path,
            proof_manifest_path=proof_manifest_path,
            proof_recording_script_path=proof_recording_script_path,
            launch_helper_command=launch_helper_command,
            elevated_launch_script_path=elevated_launch_script_path,
            installed_module_path=installed_module_path,
            checklist_path=checklist_path,
            resolved_modules_dir=resolved_modules_dir,
            resolved_game_root_dir=resolved_game_root_dir,
            package_resource_inventory=package_resource_inventory,
            package_resource_summary=package_resource_summary,
            warnings=tuple(warnings),
            blocking_messages=tuple(blocking),
            summary=summary,
            next_action=next_action,
        )

    def map_studio_game_proof_recording_handoff(self) -> MapStudioGameProofRecordingSummary:
        """Return default proof-recording inputs without mutating project state."""

        launch = self.map_studio_launch_handoff()
        proof_manifest_path = str(getattr(launch, "proof_manifest_path", "") or "").strip()
        warnings = list(getattr(launch, "warnings", ()) or ())
        blocking = list(getattr(launch, "blocking_messages", ()) or ())
        if proof_manifest_path and not Path(proof_manifest_path).is_file():
            if not any(proof_manifest_path in item for item in blocking):
                blocking.append(f"Proof manifest does not exist: {proof_manifest_path}")
        if not proof_manifest_path:
            blocking.append("Choose the proof manifest written by Stage .mod or Install Test before recording game proof.")
        required_checks = _map_studio_proof_required_check_labels(proof_manifest_path)
        package_resource_inventory = _map_studio_package_resource_inventory(proof_manifest_path, payload=None)
        if not package_resource_inventory:
            package_resource_inventory = dict(getattr(launch, "package_resource_inventory", {}) or {})
        package_resource_summary = _map_studio_package_resource_summary(package_resource_inventory)
        if not package_resource_inventory:
            warnings.append("Package resource inventory is missing; record proof only after staging/installing the current authored module.")
        ready = len(blocking) == 0
        summary = (
            f"Proof recording ready for {launch.game}:{launch.module_root or '<module>'}; attach screenshot or video evidence."
            if ready
            else "Proof recording needs a staged Map Studio proof manifest before it can mark the module game-tested."
        )
        next_action = (
            "Select the KOTOR screenshot/video evidence, confirm the in-game checks, then record proof."
            if ready
            else "Use Stage .mod or Install Test first, or browse to an existing proof manifest in the Record Proof dialog."
        )
        return MapStudioGameProofRecordingSummary(
            ready=ready,
            module_root=launch.module_root,
            game=launch.game,
            warp_command=launch.warp_command,
            proof_manifest_path=proof_manifest_path,
            required_checks=required_checks,
            package_resource_inventory=package_resource_inventory,
            package_resource_summary=package_resource_summary,
            warnings=tuple(warnings),
            blocking_messages=tuple(blocking),
            summary=summary,
            next_action=next_action,
        )

    def record_map_studio_game_proof(
        self,
        *,
        proof_manifest_path: str | Path,
        evidence_path: str | Path,
        tester: str = "",
        notes: str = "",
        module_loads_in_game: bool = False,
        module_identity_matches_authored_resref: bool = False,
        player_spawns_on_floor: bool = False,
        test_placeable_visible: bool = False,
        player_can_walk_on_floor: bool = False,
        transition_pathing_sanity_confirmed: bool = False,
        no_inherited_base_game_geometry_or_scripted_movers: bool = False,
        texture_paint_visible_in_game: bool = False,
        terrain_sculpt_and_generated_walkmesh_work_in_game: bool = False,
        placed_assets_match_editor_staging: bool = False,
        enemy_spawns_hostile: bool = False,
        npc_spawns_and_free_roams: bool = False,
        terminal_operates: bool = False,
        container_opens_with_inventory: bool = False,
        puzzle_sequence_unlocks_door: bool = False,
        animated_door_operates: bool = False,
        configured_transition_operates: bool = False,
        player_start_position_and_facing_match: bool = False,
        allow_missing_evidence: bool = False,
    ):
        """Record in-game proof for a staged Map Studio module proof manifest."""

        before = self._capture_map_studio_command_state()
        proof_path = Path(proof_manifest_path)
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except Exception:
            proof = {}
        task = str(proof.get("task") or "").strip().upper()
        proof_filename = proof_path.name.lower()
        payload = dict((getattr(self.project, "extra_sections", {}) or {}).get("authored_module") or {})
        current_proof_manifest = str(payload.get("proof_manifest_path") or "").strip()
        invalidation = dict(payload.get("export_proof_invalidation") or {}) if isinstance(payload.get("export_proof_invalidation"), dict) else {}
        is_authored_proof = bool(proof.get("t2601_smoke_contract") or proof.get("modder_test_plan") or proof.get("package"))
        if payload and is_authored_proof:
            if invalidation:
                message = (
                    "Cannot record Map Studio game proof because the authored KMAP changed after this package/proof "
                    "manifest was staged. Regenerate the package and run a fresh in-game proof."
                )
                result = AuthoredModuleGameProofResult(
                    ok=False,
                    proof_manifest_path=str(proof_path),
                    evidence_path=str(evidence_path),
                    blocking_issues=[message],
                    message=message,
                    code="stale_proof_manifest",
                )
                self.model.log(result.message)
                return result
            if not current_proof_manifest:
                message = (
                    "Cannot record Map Studio game proof because this authored KMAP has no current staged proof "
                    "manifest. Stage the current authored module before recording live proof."
                )
                result = AuthoredModuleGameProofResult(
                    ok=False,
                    proof_manifest_path=str(proof_path),
                    evidence_path=str(evidence_path),
                    blocking_issues=[message],
                    message=message,
                    code="proof_manifest_not_current",
                )
                self.model.log(result.message)
                return result
            if current_proof_manifest and Path(current_proof_manifest) != proof_path:
                message = (
                    "Cannot record Map Studio game proof because the selected proof manifest is not the current "
                    "staged manifest for this KMAP. Stage the current authored module again."
                )
                result = AuthoredModuleGameProofResult(
                    ok=False,
                    proof_manifest_path=str(proof_path),
                    evidence_path=str(evidence_path),
                    blocking_issues=[message],
                    message=message,
                    code="proof_manifest_not_current",
                )
                self.model.log(result.message)
                return result
        common = {
            "proof_manifest_path": str(proof_path),
            "evidence_path": str(evidence_path),
            "tester": tester,
            "notes": notes,
            "module_loads_in_game": bool(module_loads_in_game),
            "player_spawns_on_floor": bool(player_spawns_on_floor),
            "test_placeable_visible": bool(test_placeable_visible),
            "player_can_walk_on_floor": bool(player_can_walk_on_floor),
            "allow_missing_evidence": bool(allow_missing_evidence),
        }
        if task == "T2601" or proof_filename.endswith("_in_game_smoke_manifest.json"):
            result = record_dev_module_game_proof(DevModuleGameProofRequest(**common))
        else:
            result = record_authored_module_game_proof(
                AuthoredModuleGameProofRequest(
                    **common,
                    module_identity_matches_authored_resref=bool(module_identity_matches_authored_resref),
                    transition_pathing_sanity_confirmed=bool(transition_pathing_sanity_confirmed),
                    no_inherited_base_game_geometry_or_scripted_movers=bool(
                        no_inherited_base_game_geometry_or_scripted_movers
                    ),
                    texture_paint_visible_in_game=bool(texture_paint_visible_in_game),
                    terrain_sculpt_and_generated_walkmesh_work_in_game=bool(
                        terrain_sculpt_and_generated_walkmesh_work_in_game
                    ),
                    placed_assets_match_editor_staging=bool(placed_assets_match_editor_staging),
                    enemy_spawns_hostile=bool(enemy_spawns_hostile),
                    npc_spawns_and_free_roams=bool(npc_spawns_and_free_roams),
                    terminal_operates=bool(terminal_operates),
                    container_opens_with_inventory=bool(container_opens_with_inventory),
                    puzzle_sequence_unlocks_door=bool(puzzle_sequence_unlocks_door),
                    animated_door_operates=bool(animated_door_operates),
                    configured_transition_operates=bool(configured_transition_operates),
                    player_start_position_and_facing_match=bool(player_start_position_and_facing_match),
                )
            )
            if getattr(result, "ok", False):
                if payload:
                    try:
                        recorded_proof = json.loads(Path(getattr(result, "proof_manifest_path", "") or proof_path).read_text(encoding="utf-8"))
                    except Exception:
                        recorded_proof = {}
                    payload["game_tested"] = True
                    payload["proof_manifest_path"] = str(getattr(result, "proof_manifest_path", "") or proof_path)
                    payload["pack_manifest_path"] = str(getattr(result, "pack_manifest_path", "") or "")
                    payload["in_game_proof_evidence_path"] = str(getattr(result, "evidence_path", "") or evidence_path)
                    for key in ("manual_proof_required", "game_test", "modder_test_plan", "export_job"):
                        if key in recorded_proof:
                            payload[key] = recorded_proof[key]
                    authored_proof = recorded_proof.get("in_game_proof")
                    authored_manifest = recorded_proof.get("map_studio_authored_module")
                    if not isinstance(authored_proof, dict) and isinstance(authored_manifest, dict):
                        authored_proof = authored_manifest.get("in_game_proof")
                    if not isinstance(authored_proof, dict):
                        try:
                            pack_proof = json.loads(Path(getattr(result, "pack_manifest_path", "") or "").read_text(encoding="utf-8"))
                        except Exception:
                            pack_proof = {}
                        pack_authored = pack_proof.get("map_studio_authored_module") if isinstance(pack_proof, dict) else {}
                        if isinstance(pack_authored, dict):
                            authored_proof = pack_authored.get("in_game_proof")
                    if isinstance(authored_proof, dict):
                        payload["in_game_proof"] = authored_proof
                    self.project.extra_sections["authored_module"] = payload
                    self.project.dirty = True
                    module_root = str(payload.get("module_root") or getattr(self.project, "name", "") or "new_level")
                    self._record_map_studio_command(
                        action_key="map_studio.export.record_game_proof",
                        label=f"Record game proof {module_root}",
                        before=before,
                        stale_outputs=(),
                        readiness_impact=(
                            "Map Studio proof metadata changed; generated MDL/MDX/WOK/LYT/VIS/PTH/.mod files are unchanged."
                        ),
                        summary="Recorded accepted in-game evidence and promoted the authored module proof state.",
                        metadata={
                            "module_root": module_root,
                            "proof_manifest_path": str(getattr(result, "proof_manifest_path", "") or proof_path),
                            "evidence_path": str(getattr(result, "evidence_path", "") or evidence_path),
                            "game_tested": True,
                        },
                    )
        self.model.log(getattr(result, "message", "Recorded Map Studio game proof."))
        return result

    def export_fbx(self, output_path: str | Path, *, dry_run: bool = False, **context):
        return self.export_bridge.export_fbx(self.project, output_path, LevelExportOptions(dry_run=dry_run), **context)

    def export_obj(self, output_path: str | Path, *, dry_run: bool = False, **context):
        return self.export_bridge.export_obj(self.project, output_path, LevelExportOptions(dry_run=dry_run), **context)

    def add_blueprint(self, name: str = "Blueprint", blueprint_type: str = "Custom", template_resref: str = ""):
        blueprint = self.blueprint_service.add_blueprint(self.project, name, blueprint_type=blueprint_type, template_resref=template_resref)
        self.model.select(blueprint.blueprint_id)
        return blueprint

    def import_library_asset(self, row: dict[str, Any], *, resource_manager: Any = None):
        """Add a game-library asset row to the editable KMAP scene."""
        asset = dict(row or {})
        resref = str(asset.get("resref") or "").strip()
        if not resref:
            raise ValueError("Library asset is missing a resref.")
        game = str(asset.get("game") or self.project.game or "K1").upper()
        category = str(asset.get("category") or "").strip()
        model_class = str(asset.get("model_class") or "").strip()
        source = str(asset.get("source") or "")
        metadata = {
            "library_asset": {
                "resref": resref,
                "game": game,
                "category": category,
                "model_class": model_class,
                "source": source,
                "area_label": asset.get("area_label", ""),
                "module_code": asset.get("module_code", ""),
                "location": asset.get("location", ""),
            }
        }
        scene = LevelScene(self.project)
        if category == "Module" or model_class.lower() == "tile":
            placement = resolve_module_room_placement(game=game, resref=resref, resource_manager=resource_manager)
            transform = LevelTransform(position=placement.position) if placement is not None else None
            lyt_entry = {"model": resref, "source": "library", "game": game}
            if placement is not None:
                lyt_entry.update(placement.to_metadata())
            room = scene.add_room(
                str(asset.get("area_label") or resref),
                model_resref=resref,
                source_module=str(
                    (placement.module_code if placement is not None else "")
                    or asset.get("module_code")
                    or asset.get("location")
                    or ""
                ),
                module_id=self.model.active_module_id,
                transform=transform,
                lyt_entry=lyt_entry,
            )
            if placement is not None:
                metadata.setdefault("module_group", placement.to_metadata())
            room.metadata.update(metadata)
            self.model.select(room.room_id)
            self.model.log(f"Imported library room {game}:{resref}.")
            return room

        blueprint_type = self._blueprint_type_for_library_asset(category, resref)
        blueprint = self.blueprint_service.add_blueprint(
            self.project,
            str(asset.get("area_label") or resref),
            blueprint_type=blueprint_type,
            template_resref=resref,
        )
        blueprint.properties.update({"game": game, "source": source, "category": category, "model_class": model_class})
        blueprint.metadata.update(metadata)
        self.model.select(blueprint.blueprint_id)
        self.model.log(f"Imported library {blueprint_type.lower()} {game}:{resref}.")
        return blueprint

    @staticmethod
    def _blueprint_type_for_library_asset(category: str, resref: str) -> str:
        value = (category or "").lower()
        lower_resref = (resref or "").lower()
        if value == "creature" or lower_resref.startswith("c_"):
            return "Creature"
        if "item" in value:
            return "Placeable"
        if value == "character":
            return "Creature"
        if lower_resref.startswith(("plc_", "utp_")):
            return "Placeable"
        if lower_resref.startswith(("door_", "utd_")):
            return "Door"
        return "Custom"

    def record_port(self, source_game: str, target_game: str):
        before = self._capture_map_studio_command_state()
        report = self.porter_service.record_port_decision(self.project, source_game, target_game)
        if bool(getattr(report, "ok", False)):
            self._record_map_studio_command(
                action_key="map_studio.project.retarget_game",
                label=f"Retarget {str(source_game).upper()} to {str(target_game).upper()}",
                before=before,
                stale_outputs=("ARE", "GIT", "IFO", "PTH", "LYT", "VIS", "MDL", "MDX", "WOK", ".mod"),
                readiness_impact=(
                    "Validate all target-game texture/template dependencies, rebuild the module, and record fresh game proof."
                ),
                metadata={
                    "source_game": str(getattr(report, "source_game", source_game) or source_game).upper(),
                    "target_game": str(getattr(report, "target_game", target_game) or target_game).upper(),
                    "dependency_risks": list(getattr(report, "unsupported", ()) or ()),
                },
            )
        return report
