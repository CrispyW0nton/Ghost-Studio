"""Tool registry: collect tool definitions and dispatch handle_tool by name.

Architecture note (Khononov, "Balancing Coupling in Software Design"):
  This module is the single integration point between the MCP server and all
  tool sub-modules.  Its only coupling to each tool module is the module's
  public API — get_tools() and handle_*() functions — which constitutes
  Contract Coupling (the lowest acceptable strength for components at this
  distance).

  Tool modules should never import from each other; all shared state flows
  through the service container (adapters.py / state.py), not through this
  registry.

Architecture note (Constantine, "Structured Design"):
  New composite tools (get_resource, get_quest) follow Transform Analysis:
  each is a single-purpose transform with data-coupling only.  Tool names
  are context-free — the same tool works in a Discord bot, VS Code extension,
  CI pipeline, or any other consumer without modification.

  Tool manifest (v3.14 — 170 total):
  Installation   (3): detectInstallations, loadInstallation, kotor_installation_info
  Discovery      (4): listResources, describeResource, kotor_find_resource, kotor_search_resources
  Game data      (3): journalOverview, kotor_lookup_2da, kotor_lookup_tlk
  GhostRigger    (9): ghostrigger_open_model, ghostrigger_render_model, ghostrigger_model_info,
                      ghostrigger_list_game_models, ghostrigger_audit,
                      ghostrigger_export_model_for_unity, ghostrigger_export_model_for_unreal,
                      ghostrigger_validate_unity_import,
                      ghostrigger_run_malak_unity_smoke
  DebugSkinning (25): ghostrigger_debug_launch_app, ghostrigger_debug_close_app,
                      ghostrigger_debug_get_runtime_status, ghostrigger_debug_set_game_library_path,
                      ghostrigger_debug_verify_game_library, ghostrigger_debug_load_model,
                      ghostrigger_debug_get_loaded_asset_info, ghostrigger_debug_list_animations,
                      ghostrigger_debug_set_animation, ghostrigger_debug_set_animation_time,
                      ghostrigger_debug_set_bind_pose, ghostrigger_debug_set_camera_preset,
                      ghostrigger_debug_capture_viewport, ghostrigger_debug_capture_validation_set,
                      ghostrigger_debug_get_skinning_state, ghostrigger_debug_get_renderer_state,
                      ghostrigger_debug_get_bone_hierarchy, ghostrigger_debug_get_bone_map,
                      ghostrigger_debug_get_palette_remap_table, ghostrigger_debug_get_bind_pose_matrices,
                      ghostrigger_debug_get_animated_pose_matrices, ghostrigger_debug_get_uploaded_palette,
                      ghostrigger_debug_sample_vertex_influences, ghostrigger_debug_compare_cpu_gpu_skinning,
                      ghostrigger_debug_export_debug_bundle
  Modules        (3): kotor_list_modules, kotor_describe_module, kotor_module_resources
  Game test      (2): kotor_list_saves, kotor_prepare_save_warp_test
  Game input     (5): kotor_input_status, kotor_input_click,
                      kotor_input_type, kotor_capture_window,
                      kotor_run_save_warp_route
  Live log       (4): kotor_log_start, kotor_log_status,
                      kotor_log_stop, kotor_log_analyze
  DInput hook    (3): kotor_dinput_hook_status, kotor_dinput_hook_install,
                      kotor_dinput_hook_send
  GFF data       (3): kotor_read_gff, kotor_read_2da, kotor_read_tlk
  AgentDecompile (11): kotor_binary_ping, kotor_binary_info, kotor_list_engine_funcs,
                       kotor_decompile_function, kotor_search_symbols,
                       kotor_search_engine_strings, kotor_get_references,
                       kotor_call_graph, kotor_data_flow, kotor_inspect_memory,
                       kotor_engine_script
  Composite      (2): get_resource, get_quest
  Refs           (6): kotor_list_references, kotor_find_referrers,
                      kotor_find_strref_referrers, kotor_describe_dlg,
                      kotor_describe_jrl, kotor_describe_resource_refs
  Walkmesh       (1): kotor_walkmesh_validation_diagram
  Archives       (2): kotor_list_archive, kotor_extract_resource
  DebugMaterials (16): material/texture/render assembly debug tools
  Retargeting    (4): ghostrigger_get_retarget_skeleton_info,
                      ghostrigger_build_retarget_map,
                      ghostrigger_list_retarget_animations,
                      ghostrigger_export_unity_fbx
  ModelPipeline  (3): inspect_mdl, inspect_mdl_ghostrigger,
                      compare_model_pipelines
  Legacy surface (52): 51 safe GhostScripter compatibility names plus one
                       machine-readable compatibility report.  Eight broad
                       mutation names remain inventoried behind owned,
                       validated project services rather than arbitrary writes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from kotormcp.utils import json_content
from kotormcp.tools import (
    installation, discovery, gamedata, ghostrigger, level_export,
    debug_skinning, debug_materials,
    modules, game_test, kotor_input, kotor_live_log, kotor_dinput_hook,
    gffdata, decompile, resource, quest,
    refs, walkmesh, archives, retargeting, ghostrigger_tools,
    legacy_ghostscripter,
)


MODEL_PIPELINE_RESPONSE_CHARS = 250_000


def _model_pipeline_tools() -> List[Dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "game": {
                "type": "string",
                "description": "Game alias: k1, k2, swkotor, tsl, or kotor2",
            },
            "resref": {
                "type": "string",
                "description": "MDL resource reference without extension",
            },
        },
        "required": ["game", "resref"],
    }
    return [
        {
            "name": "inspect_mdl",
            "description": "Inspect the raw PyKotor model for a KOTOR MDL resource.",
            "inputSchema": schema,
        },
        {
            "name": "inspect_mdl_ghostrigger",
            "description": "Inspect GhostRigger's imported model representation for a KOTOR MDL resource.",
            "inputSchema": schema,
        },
        {
            "name": "compare_model_pipelines",
            "description": "Compare PyKotor ground truth against GhostRigger's imported model pipeline.",
            "inputSchema": schema,
        },
    ]


def _native_tools() -> List[Dict[str, Any]]:
    """Return GhostStudio's native tool definitions before name aliases."""

    return (
        installation.get_tools()          # 3  installation management
        + discovery.get_tools()           # 4  resource discovery
        + gamedata.get_tools()            # 3  game data (2da, tlk, journal)
        + ghostrigger.get_tools()         # 5  3D model pipeline
        + level_export.get_tools()       # 1  saved KMAP whole-level export
        + debug_skinning.get_tools()      # 25 debug skinning bridge
        + modules.get_tools()             # 3  module enumeration
        + game_test.get_tools()           # 2  live game-test handoff
        + kotor_input.get_tools()         # 5  live game-window input
        + kotor_live_log.get_tools()      # 4  live exe-hook logging
        + kotor_dinput_hook.get_tools()   # 3  DirectInput proxy deployment/input
        + gffdata.get_tools()             # 3  deep GFF/2DA/TLK reads
        + decompile.get_tools()           # 11 AgentDecompile / Ghidra bridge
        + resource.get_tools()            # 1  get_resource: universal accessor
        + quest.get_tools()               # 1  get_quest: composite quest inspector
        + refs.get_tools()                # 6  reference tracing (ported from upstream)
        + walkmesh.get_tools()            # 1  walkmesh validation diagram
        + archives.get_tools()            # 2  archive listing + extraction
        + debug_materials.get_tools()     # 13 debug materials/textures/assembly
        + retargeting.get_tools()         # 4  retargeting introspection/mapping/export
        + _model_pipeline_tools()         # 3  AGENTS.md backend validation aliases
    )


def get_all_tools() -> List[Dict[str, Any]]:
    """Return 109 native tools plus 60 clean-room compatibility definitions."""

    native = _native_tools()
    return native + legacy_ghostscripter.get_tools(native)


async def handle_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool invocation to the correct handler."""
    legacy_target = legacy_ghostscripter.resolve_direct_alias(name, arguments)
    if legacy_target is not None:
        target_name, adapted_arguments = legacy_target
        return await handle_tool(target_name, adapted_arguments)
    if legacy_ghostscripter.handles_service_alias(name):
        return await legacy_ghostscripter.handle_service_alias(name, arguments)

    # ── Installation tools ────────────────────────────────────────────────────
    if name == "detectInstallations":
        return await installation.handle_detect_installations(arguments)
    if name == "loadInstallation":
        return await installation.handle_load_installation(arguments)
    if name == "kotor_installation_info":
        return await installation.handle_installation_info(arguments)

    # ── Discovery tools ───────────────────────────────────────────────────────
    if name == "listResources":
        return await discovery.handle_list_resources(arguments)
    if name == "describeResource":
        return await discovery.handle_describe_resource(arguments)
    if name == "kotor_find_resource":
        return await discovery.handle_find_resource(arguments)
    if name == "kotor_search_resources":
        return await discovery.handle_search_resources(arguments)

    # ── Game data tools ───────────────────────────────────────────────────────
    if name == "journalOverview":
        return await gamedata.handle_journal_overview(arguments)
    if name == "kotor_lookup_2da":
        return await gamedata.handle_lookup_2da(arguments)
    if name == "kotor_lookup_tlk":
        return await gamedata.handle_lookup_tlk(arguments)

    # ── GhostRigger-specific tools ────────────────────────────────────────────
    if name == "ghoststudio_export_level":
        return await level_export.handle_export_level(arguments)
    if name == "ghostrigger_open_model":
        return await ghostrigger.handle_open_model(arguments)
    if name == "ghostrigger_render_model":
        return await ghostrigger.handle_render_model(arguments)
    if name == "ghostrigger_model_info":
        return await ghostrigger.handle_model_info(arguments)
    if name == "ghostrigger_list_game_models":
        return await ghostrigger.handle_list_game_models(arguments)
    if name == "ghostrigger_audit":
        return await ghostrigger.handle_audit(arguments)
    if name == "ghostrigger_export_model_for_unity":
        return await ghostrigger.handle_export_model_for_unity(arguments)
    if name == "ghostrigger_export_model_for_unreal":
        return await ghostrigger.handle_export_model_for_unreal(arguments)
    if name == "ghostrigger_validate_unity_import":
        return await ghostrigger.handle_validate_unity_import(arguments)
    if name == "ghostrigger_run_malak_unity_smoke":
        return await ghostrigger.handle_run_malak_unity_smoke(arguments)
    if name == "inspect_mdl":
        return json_content(
            ghostrigger_tools.inspect_mdl(arguments["game"], arguments["resref"]),
            max_chars=MODEL_PIPELINE_RESPONSE_CHARS,
        )
    if name == "inspect_mdl_ghostrigger":
        return json_content(
            ghostrigger_tools.inspect_mdl_ghostrigger(arguments["game"], arguments["resref"]),
            max_chars=MODEL_PIPELINE_RESPONSE_CHARS,
        )
    if name == "compare_model_pipelines":
        return json_content(
            ghostrigger_tools.compare_model_pipelines(arguments["game"], arguments["resref"]),
            max_chars=MODEL_PIPELINE_RESPONSE_CHARS,
        )

    # ── Module tools ──────────────────────────────────────────────────────────
    if name == "kotor_list_modules":
        return await modules.handle_list_modules(arguments)
    if name == "kotor_describe_module":
        return await modules.handle_describe_module(arguments)
    if name == "kotor_module_resources":
        return await modules.handle_module_resources(arguments)
    if name == "kotor_list_saves":
        return await game_test.handle_list_saves(arguments)
    if name == "kotor_prepare_save_warp_test":
        return await game_test.handle_prepare_save_warp_test(arguments)
    if name == "kotor_input_status":
        return await kotor_input.handle_status(arguments)
    if name == "kotor_input_click":
        return await kotor_input.handle_click(arguments)
    if name == "kotor_input_type":
        return await kotor_input.handle_type(arguments)
    if name == "kotor_capture_window":
        return await kotor_input.handle_capture_window(arguments)
    if name == "kotor_run_save_warp_route":
        return await kotor_input.handle_run_save_warp_route(arguments)
    if name == "kotor_log_start":
        return await kotor_live_log.handle_start(arguments)
    if name == "kotor_log_status":
        return await kotor_live_log.handle_status(arguments)
    if name == "kotor_log_stop":
        return await kotor_live_log.handle_stop(arguments)
    if name == "kotor_log_analyze":
        return await kotor_live_log.handle_analyze(arguments)
    if name == "kotor_dinput_hook_status":
        return await kotor_dinput_hook.handle_status(arguments)
    if name == "kotor_dinput_hook_install":
        return await kotor_dinput_hook.handle_install(arguments)
    if name == "kotor_dinput_hook_send":
        return await kotor_dinput_hook.handle_send(arguments)

    # ── Deep read tools ───────────────────────────────────────────────────────
    if name == "kotor_read_gff":
        return await gffdata.handle_read_gff(arguments)
    if name == "kotor_read_2da":
        return await gffdata.handle_read_2da(arguments)
    if name == "kotor_read_tlk":
        return await gffdata.handle_read_tlk(arguments)

    # ── AgentDecompile / Ghidra bridge tools ─────────────────────────────────
    if name == "kotor_binary_ping":
        return await decompile.handle_ping(arguments)
    if name == "kotor_binary_info":
        return await decompile.handle_binary_info(arguments)
    if name == "kotor_list_engine_funcs":
        return await decompile.handle_list_engine_funcs(arguments)
    if name == "kotor_decompile_function":
        return await decompile.handle_decompile_function(arguments)
    if name == "kotor_search_symbols":
        return await decompile.handle_search_symbols(arguments)
    if name == "kotor_search_engine_strings":
        return await decompile.handle_search_engine_strings(arguments)
    if name == "kotor_get_references":
        return await decompile.handle_get_references(arguments)
    if name == "kotor_call_graph":
        return await decompile.handle_call_graph(arguments)
    if name == "kotor_data_flow":
        return await decompile.handle_data_flow(arguments)
    if name == "kotor_inspect_memory":
        return await decompile.handle_inspect_memory(arguments)
    if name == "kotor_engine_script":
        return await decompile.handle_engine_script(arguments)

    # ── Composite / high-level tools (v3.3) ───────────────────────────────────
    if name == "get_resource":
        return await resource.handle_get_resource(arguments)
    if name == "get_quest":
        return await quest.handle_get_quest(arguments)

    # ── Reference and plot tools (v3.4 — ported from upstream KotorMCP) ───────
    if name == "kotor_list_references":
        return await refs.handle_list_references(arguments)
    if name == "kotor_find_referrers":
        return await refs.handle_find_referrers(arguments)
    if name == "kotor_find_strref_referrers":
        return await refs.handle_find_strref_referrers(arguments)
    if name == "kotor_describe_dlg":
        return await refs.handle_describe_dlg(arguments)
    if name == "kotor_describe_jrl":
        return await refs.handle_describe_jrl(arguments)
    if name == "kotor_describe_resource_refs":
        return await refs.handle_describe_resource_refs(arguments)

    # ── Walkmesh tools (v3.4 — ported from upstream KotorMCP) ────────────────
    if name == "kotor_walkmesh_validation_diagram":
        return await walkmesh.handle_walkmesh_validation_diagram(arguments)

    # ── Archive tools (v3.4 — ported from upstream KotorMCP) ─────────────────
    if name == "kotor_list_archive":
        return await archives.handle_list_archive(arguments)
    if name == "kotor_extract_resource":
        return await archives.handle_extract_resource(arguments)

    # ── Retargeting tools (v3.7 — introspection, mapping, and Day 4 export) ──
    if name == "ghostrigger_get_retarget_skeleton_info":
        return await retargeting.handle_get_retarget_skeleton_info(arguments)
    if name == "ghostrigger_build_retarget_map":
        return await retargeting.handle_build_retarget_map(arguments)
    if name == "ghostrigger_list_retarget_animations":
        return await retargeting.handle_list_retarget_animations(arguments)
    if name == "ghostrigger_export_unity_fbx":
        return await retargeting.handle_export_unity_fbx(arguments)

    # ── Debug Skinning Bridge tools (v3.5 — observability-first) ─────────────
    if name == "ghostrigger_debug_launch_app":
        return await debug_skinning.handle_launch_app(arguments)
    if name == "ghostrigger_debug_close_app":
        return await debug_skinning.handle_close_app(arguments)
    if name == "ghostrigger_debug_get_runtime_status":
        return await debug_skinning.handle_get_runtime_status(arguments)
    if name == "ghostrigger_debug_set_game_library_path":
        return await debug_skinning.handle_set_game_library_path(arguments)
    if name == "ghostrigger_debug_verify_game_library":
        return await debug_skinning.handle_verify_game_library(arguments)
    if name == "ghostrigger_debug_load_model":
        return await debug_skinning.handle_load_model(arguments)
    if name == "ghostrigger_debug_get_loaded_asset_info":
        return await debug_skinning.handle_get_loaded_asset_info(arguments)
    if name == "ghostrigger_debug_list_animations":
        return await debug_skinning.handle_list_animations(arguments)
    if name == "ghostrigger_debug_set_animation":
        return await debug_skinning.handle_set_animation(arguments)
    if name == "ghostrigger_debug_set_animation_time":
        return await debug_skinning.handle_set_animation_time(arguments)
    if name == "ghostrigger_debug_set_bind_pose":
        return await debug_skinning.handle_set_bind_pose(arguments)
    if name == "ghostrigger_debug_set_camera_preset":
        return await debug_skinning.handle_set_camera_preset(arguments)
    if name == "ghostrigger_debug_capture_viewport":
        return await debug_skinning.handle_capture_viewport(arguments)
    if name == "ghostrigger_debug_capture_validation_set":
        return await debug_skinning.handle_capture_validation_set(arguments)
    if name == "ghostrigger_debug_get_skinning_state":
        return await debug_skinning.handle_get_skinning_state(arguments)
    if name == "ghostrigger_debug_get_renderer_state":
        return await debug_skinning.handle_get_renderer_state(arguments)
    if name == "ghostrigger_debug_get_bone_hierarchy":
        return await debug_skinning.handle_get_bone_hierarchy(arguments)
    if name == "ghostrigger_debug_get_bone_map":
        return await debug_skinning.handle_get_bone_map(arguments)
    if name == "ghostrigger_debug_get_palette_remap_table":
        return await debug_skinning.handle_get_palette_remap_table(arguments)
    if name == "ghostrigger_debug_get_bind_pose_matrices":
        return await debug_skinning.handle_get_bind_pose_matrices(arguments)
    if name == "ghostrigger_debug_get_animated_pose_matrices":
        return await debug_skinning.handle_get_animated_pose_matrices(arguments)
    if name == "ghostrigger_debug_get_uploaded_palette":
        return await debug_skinning.handle_get_uploaded_palette(arguments)
    if name == "ghostrigger_debug_sample_vertex_influences":
        return await debug_skinning.handle_sample_vertex_influences(arguments)
    if name == "ghostrigger_debug_compare_cpu_gpu_skinning":
        return await debug_skinning.handle_compare_cpu_gpu(arguments)
    if name == "ghostrigger_debug_export_debug_bundle":
        return await debug_skinning.handle_export_debug_bundle(arguments)

    # ── Debug Materials/Textures/Assembly tools (D4+D6) ───────────────────────
    if name == "ghostrigger_list_materials":
        return await debug_materials.handle_list_materials(arguments)
    if name == "ghostrigger_list_textures":
        return await debug_materials.handle_list_textures(arguments)
    if name == "ghostrigger_get_material_info":
        return await debug_materials.handle_get_material_info(arguments)
    if name == "ghostrigger_get_texture_binding_info":
        return await debug_materials.handle_get_texture_binding_info(arguments)
    if name == "ghostrigger_get_txi_info":
        return await debug_materials.handle_get_txi_info(arguments)
    if name == "ghostrigger_get_uv_channel_info":
        return await debug_materials.handle_get_uv_channel_info(arguments)
    if name == "ghostrigger_get_supermodel_chain":
        return await debug_materials.handle_get_supermodel_chain(arguments)
    if name == "ghostrigger_list_body_parts":
        return await debug_materials.handle_list_body_parts(arguments)
    if name == "ghostrigger_get_missing_mesh_report":
        return await debug_materials.handle_get_missing_mesh_report(arguments)
    if name == "ghostrigger_get_node_classification_audit":
        return await debug_materials.handle_get_node_classification_audit(arguments)
    if name == "ghostrigger_get_vertex_space_audit":
        return await debug_materials.handle_get_vertex_space_audit(arguments)
    if name == "ghostrigger_get_render_filter_audit":
        return await debug_materials.handle_get_render_filter_audit(arguments)
    if name == "ghostrigger_export_render_debug_bundle":
        return await debug_materials.handle_export_render_debug_bundle(arguments)
    # Phase D6 regression-debug tools
    if name == "ghostrigger_get_render_filter_results":
        return await debug_materials.handle_get_render_filter_results(arguments)
    if name == "ghostrigger_get_vbo_build_status":
        return await debug_materials.handle_get_vbo_build_status(arguments)
    if name == "ghostrigger_get_k1_vs_k2_model_differences":
        return await debug_materials.handle_get_k1_vs_k2_model_differences(arguments)

    raise ValueError(f"Unknown tool: '{name}'")
