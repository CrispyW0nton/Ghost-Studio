"""End-to-end contracts for the guided foreign-rig Character Builder path."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "native" / "GhostRigger.Core.GUI.Display" / "Python" / "src" / "gui"


def test_custom_project_defaults_to_the_independent_builder_mode() -> None:
    from src.core.project.custom_rigged_character_project import (
        BUILDER_MODE_CUSTOM_RIGGED,
        CUSTOM_RIGGED_WORKFLOW_STEPS,
        CustomRiggedCharacterProject,
        MaterialAssignment,
    )

    project = CustomRiggedCharacterProject(creature_name="Borhek", resource_name="kpm_borhek")

    assert project.builder_mode == BUILDER_MODE_CUSTOM_RIGGED
    assert project.native_template_model == ""
    assert tuple(project.workflow_steps) == CUSTOM_RIGGED_WORKFLOW_STEPS
    assert project.animation_mappings == []
    assert project.target_game == "K2"
    assert MaterialAssignment().flip_vertical_for_kotor is True
    assert MaterialAssignment.from_dict({}).flip_vertical_for_kotor is True


def test_character_builder_entry_offers_native_head_and_custom_cards() -> None:
    from src.gui.windows.qt_character_builder_mode_selector import CHARACTER_BUILDER_MODES

    assert tuple(CHARACTER_BUILDER_MODES) == (
        "native_kotor_character",
        "native_kotor_head",
        "facial_performance_head",
        "custom_rigged_character",
    )

    source = (GUI_ROOT / "windows" / "qt_character_builder_mode_selector.py").read_text(
        encoding="utf-8"
    )
    assert "Native KOTOR Character" in source
    assert "Custom KOTOR Head" in source
    assert "Modular Head Builder" in source
    assert "Facial Performance Head" in source
    assert "Custom Animation Patch Required" in source
    assert "Custom Rigged Character" in source
    assert "Custom Animation Patch" in source
    assert "self-contained KOTOR model" in source


def test_custom_window_is_independent_and_exposes_the_ten_guided_steps() -> None:
    custom_source = (
        GUI_ROOT / "windows" / "qt_custom_rigged_character_builder_window.py"
    ).read_text(encoding="utf-8")

    assert "class QtCustomRiggedCharacterBuilderWindow(QtWidgets.QMainWindow)" in custom_source
    assert "QtCharacterBuilderWindow" not in custom_source
    for label in (
        "Project and source assets",
        "Rig inspection",
        "Scale, facing, pivot, and ground contact",
        "Animation library",
        "Animation preparation",
        "Materials, textures, and UVs",
        "Secondary Motion — Cloth & Hair",
        "KOTOR gameplay integration",
        "Validation and build",
        "Install and test",
    ):
        assert label in custom_source

    assert custom_source.index('("materials_uvs", "Materials, textures, and UVs")') < custom_source.index(
        '("secondary_motion", "Secondary Motion — Cloth & Hair")'
    ) < custom_source.index('("gameplay", "KOTOR gameplay integration")')
    assert "QtSecondaryMotionEditor" in custom_source


def test_native_character_builder_embeds_the_shared_editor_after_rig_and_deformation() -> None:
    display = (GUI_ROOT / "panels" / "qt_character_builder_panel.py").read_text(encoding="utf-8")
    tools = (
        ROOT / "native" / "GhostRigger.Core.Tools" / "Python" / "src" / "gui" / "panels" / "qt_character_builder_panel.py"
    ).read_text(encoding="utf-8")

    assert display == tools
    assert 'addTab(right_split, "Rig & Deformation")' in display
    assert 'addTab(self.secondary_motion_editor, "Secondary Motion")' in display
    assert "QtSecondaryMotionEditor" in display
    assert 'metadata["secondary_motion"]' in display


def test_borhek_ui_golden_contract_requires_visible_idle_walk_run_and_safety() -> None:
    from src.core.characters.custom_rigged_character_build_service import BORHEK_GOLDEN_CONTRACT

    assert BORHEK_GOLDEN_CONTRACT["source_hierarchy_nodes"] == 40
    assert BORHEK_GOLDEN_CONTRACT["semantic_animations"] == (
        "cpause1",
        "cwalk",
        "crun",
    )
    assert BORHEK_GOLDEN_CONTRACT["runtime_height_node"] == "heightdummy"
    assert BORHEK_GOLDEN_CONTRACT["runtime_height_offset"] == 1.9724489450454712
    assert BORHEK_GOLDEN_CONTRACT["mdl_sha256"] == (
        "49063631c4b9f3b4db80f6c6e0036430a3235c2058341a7755b1cab00a0da491"
    )
    assert BORHEK_GOLDEN_CONTRACT["mdx_sha256"] == (
        "a2d0ceb85de8403672686777df4b8c1c634f36fd93f90c16df4a8f5a51b8d89d"
    )
    assert BORHEK_GOLDEN_CONTRACT["runtime_checklist"] == (
        "visible",
        "ground_height",
        "texture_wrapping",
        "idle",
        "walk_while_moving",
        "run_while_moving_quickly",
        "turning_skin_stability",
        "module_reload",
        "custom_action_request",
    )


def test_main_entry_keeps_native_route_and_adds_custom_route() -> None:
    source = (
        GUI_ROOT / "windows" / "application_core" / "shared" / "window_lifecycle.py"
    ).read_text(encoding="utf-8")

    assert "QtCharacterBuilderWindow" in source
    assert "QtCharacterBuilderModeSelector" in source
    assert "QtCustomRiggedCharacterBuilderWindow" in source
    assert "_show_native_character_builder" in source
    assert "_show_custom_rigged_character_builder" in source


def test_custom_ui_uses_real_previews_and_reversible_install_services() -> None:
    window_source = (
        GUI_ROOT / "windows" / "qt_custom_rigged_character_builder_window.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        GUI_ROOT / "windows" / "qt_custom_rigged_character_builder_controller.py"
    ).read_text(encoding="utf-8")

    assert 'self.animation_preview_viewports: dict[str, QtMainViewportWidget]' in window_source
    assert "Preview exact install" in window_source
    assert "Load a save outside the test module" in window_source
    assert "A hostile creature has a red health bar" in window_source
    assert "KOTOR II keeps its name text cyan by design" in window_source
    assert "Install with backup" in window_source
    assert "Restore previous files" in window_source
    assert "Place one temporary test creature in PLCaa DevRoom (no console needed)" in window_source
    assert "Automatic KOTOR height correction" in window_source
    assert "Replace the creature already at the requested test spot" in window_source
    assert "keeps every other module resource and placement" in window_source
    assert "Orient imported image for KOTOR (recommended)" in window_source
    assert "Ghost Studio adds the standard KOTOR settings automatically" in window_source
    assert 'background: {color("table.background")}' in window_source
    assert 'color: {color("table.text")}' in window_source
    assert 'background: {color("input.background")}' in window_source
    assert "CustomRiggedCharacterPackagingService" in controller_source
    assert "confirmed_preview_id=preview.preview_id" in controller_source
    assert "set_animation_pose" in controller_source


def test_custom_ui_exposes_installed_utc_templates_and_compiled_behavior_hooks() -> None:
    window_source = (
        GUI_ROOT / "windows" / "qt_custom_rigged_character_builder_window.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        GUI_ROOT / "windows" / "qt_custom_rigged_character_builder_controller.py"
    ).read_text(encoding="utf-8")

    for text in (
        "Behavior and KOTOR gameplay integration",
        "Read installed character templates",
        "Use standard Zakkeg for Borhek",
        "customCharacterBehaviorTemplateTable",
        "customCharacterBehaviorHookTable",
        "customCharacterBehaviorSourceEditor",
        "Compile, check, and use this hook",
        "customCharacterCreatureSounds",
        "Choose a sound folder and match cues",
        "customCharacterCreatureSoundTable",
        "builds a native KOTOR SSF soundset",
        "direct AI event scripts are not wrapped or replaced",
    ):
        assert text in window_source
    assert "merge-safe row resolved at install" in window_source
    assert "cloned from {template_resref}.utc" in window_source
    assert "behaviorCatalogRequested" in window_source
    assert "behaviorTemplateRequested" in window_source
    assert "behaviorHookApplyRequested" in window_source
    assert "InstalledUtcTemplateCatalog" in controller_source
    assert "A save already inside the test module can keep the previous creature's model, faction, and health-bar color" in controller_source
    assert "CustomRiggedCharacterBehaviorService" in controller_source
    assert "behavior_resources=behavior.resources" in controller_source
    assert "utc_hook_overrides=behavior.utc_hook_overrides" in controller_source
    assert "utc_template_bytes=behavior.utc_template_bytes or None" in controller_source
    assert "resolve_path(self.window.project.texture_folder)" in controller_source
    assert "resolve_path(after_project.texture_folder)" in controller_source
    assert "self.project_path: Path | None = None" in window_source
    assert "self.project_path = Path()" not in window_source
    assert "def opened_path(value: str, fallback: object = \"\") -> str:" in window_source
    assert "self.project.resolve_path(chosen)" in window_source
    assert "resolved_asset = opened_path(asset.path)" in window_source
    assert "self.behavior_template_table.setColumnWidth(2, 165)" in window_source
    assert "Machine-readable catalog:" in window_source
    assert "Catalog: {report_path}" not in window_source
    assert "The creature notices a hostile target and starts combat." in window_source
    for text in (
        "Monster attack 1 (Zakkeg)",
        '"m0a1"',
        '"ctaunt"',
        '"cdamages"',
        '"cdie"',
        "Combat animation essentials are assigned and ready for validation.",
    ):
        assert text in window_source


def test_headless_blender_import_stays_hidden_for_nontechnical_windows_users() -> None:
    workflow_source = (
        ROOT
        / "native"
        / "GhostRigger.Core.Workflow"
        / "Python"
        / "src"
        / "core"
        / "retargeting"
        / "blender_animation_injection.py"
    ).read_text(encoding="utf-8")
    io_source = (
        ROOT
        / "native"
        / "GhostRigger.Core.IO"
        / "Python"
        / "src"
        / "converters"
        / "blender_fbx_mesh_importer.py"
    ).read_text(encoding="utf-8")

    for source in (workflow_source, io_source):
        assert "subprocess.STARTF_USESHOWWINDOW" in source
        assert 'getattr(subprocess, "CREATE_NO_WINDOW", 0)' in source
        assert "**_hidden_process_options()" in source
