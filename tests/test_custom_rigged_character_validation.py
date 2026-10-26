"""Focused rules for custom foreign-rig project validation and repair helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.core.characters.custom_rigged_character_build_service import (
    CustomRiggedCharacterBuildService,
    suggest_semantic_mapping,
)
from src.core.project.custom_rigged_character_project import (
    AnimationMapping,
    CustomAnimationRegistration,
    CustomRiggedCharacterProject,
    SourceAsset,
    load_custom_rigged_character_project,
    save_custom_rigged_character_project,
)
from src.core.validation.custom_rigged_character_validator import (
    AnimationClipSnapshot,
    AnimationTrackSnapshot,
    CustomRiggedCharacterSnapshot,
    CustomRiggedCharacterValidator,
    MaterialSnapshot,
    RigNodeSnapshot,
    axis_scale_point,
    ground_offset_for_contacts,
    normalized_influences,
    quaternion_continuity,
    validate_resource_name,
)


def _project() -> CustomRiggedCharacterProject:
    return CustomRiggedCharacterProject(creature_name="Test Creature", resource_name="test_beast")


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_duplicate_bone_detection() -> None:
    snapshot = CustomRiggedCharacterSnapshot(nodes=[
        RigNodeSnapshot("root"), RigNodeSnapshot("arm", "root"), RigNodeSnapshot("arm", "root")
    ])
    report = CustomRiggedCharacterValidator().validate(_project(), snapshot)
    assert "duplicate_bone_name" in _codes(report)
    assert report.build_ready is False


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        ([RigNodeSnapshot("root"), RigNodeSnapshot("hand", "missing")], "missing_parent"),
        ([RigNodeSnapshot("a", "b"), RigNodeSnapshot("b", "a")], "hierarchy_cycle"),
        ([RigNodeSnapshot("left"), RigNodeSnapshot("right")], "invalid_root_count"),
    ],
)
def test_invalid_hierarchy_detection(nodes, expected: str) -> None:
    report = CustomRiggedCharacterValidator().validate(_project(), CustomRiggedCharacterSnapshot(nodes=nodes))
    assert expected in _codes(report)


def test_missing_weighted_bone_detection() -> None:
    snapshot = CustomRiggedCharacterSnapshot(
        nodes=[RigNodeSnapshot("root")],
        vertex_influences=[[('not_exported', 1.0)]],
    )
    assert "missing_weighted_bone" in _codes(CustomRiggedCharacterValidator().validate(_project(), snapshot))


def test_unweighted_single_authoring_root_is_explained_as_the_odyssey_model_root() -> None:
    snapshot = CustomRiggedCharacterSnapshot(
        nodes=[RigNodeSnapshot("godnode"), RigNodeSnapshot("root_joint", "godnode")],
        vertex_influences=[[('root_joint', 1.0)]],
    )

    report = CustomRiggedCharacterValidator().validate(_project(), snapshot)

    assert "authoring_root_becomes_model_root" in _codes(report)
    assert not any(
        issue.code == "unused_deform_node" and "godnode" in issue.message
        for issue in report.issues
    )


def test_influence_limiting_and_normalization_is_deterministic() -> None:
    result = normalized_influences(
        [("b", 0.2), ("a", 0.7), ("c", 0.1), ("b", 0.1), ("d", -1.0)],
        max_influences=2,
    )
    assert [name for name, _weight in result] == ["a", "b"]
    assert sum(weight for _name, weight in result) == pytest.approx(1.0)
    assert result[0][1] == pytest.approx(0.7)


def test_bind_pose_consistency_detection() -> None:
    snapshot = CustomRiggedCharacterSnapshot(nodes=[
        RigNodeSnapshot(
            "root", bind_matrix=tuple([1.0] * 16), expected_bind_matrix=tuple([1.0] * 15 + [1.5])
        )
    ])
    assert "bind_pose_mismatch" in _codes(CustomRiggedCharacterValidator().validate(_project(), snapshot))


def test_axis_and_scale_conversion() -> None:
    assert axis_scale_point((1.0, 2.0, 3.0), scale=2.0, source_up="+Y", source_forward="+Y") == (
        2.0, -6.0, 4.0
    )
    assert axis_scale_point((1.0, 2.0, 3.0), scale=1.0, source_up="+Z", source_forward="-Y") == (
        -1.0, -2.0, 3.0
    )


def test_ground_offset_calculation() -> None:
    assert ground_offset_for_contacts((0.4, -0.25, 0.1)) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ground_offset_for_contacts((math.nan,))


def test_runtime_height_correction_must_match_imported_root_joint() -> None:
    project = _project()
    project.runtime_height_offset = 0.0
    snapshot = CustomRiggedCharacterSnapshot(
        nodes=[RigNodeSnapshot("root")],
        runtime_height_offset=1.9724489450454712,
        runtime_height_source="root_joint",
    )

    report = CustomRiggedCharacterValidator().validate(project, snapshot)

    assert "runtime_height_correction_missing" in _codes(report)


def test_animation_target_validation() -> None:
    project = _project()
    project.animation_mappings = [AnimationMapping(
        source_name="walk", assignment="vanilla_behavior_alias", exported_name="cwalk", confirmed=True
    )]
    snapshot = CustomRiggedCharacterSnapshot(
        nodes=[RigNodeSnapshot("root")],
        animations=[AnimationClipSnapshot(
            "walk", 1.0, tracks=(AnimationTrackSnapshot("missing_bone"),)
        )],
    )
    assert "animation_target_missing" in _codes(CustomRiggedCharacterValidator().validate(project, snapshot))


def test_quaternion_continuity_flips_equivalent_signs() -> None:
    fixed, flips = quaternion_continuity(((0, 0, 0, 1), (0, 0, 0, -1)))
    assert flips == 1
    assert fixed[0] == fixed[1]


def test_unassigned_zero_duration_action_is_recorded_without_blocking_build() -> None:
    project = _project()
    project.animation_mappings = [AnimationMapping(source_name="bind_pose", assignment="unassigned")]
    snapshot = CustomRiggedCharacterSnapshot(
        nodes=[RigNodeSnapshot("root")],
        animations=[AnimationClipSnapshot(name="bind_pose", duration=0.0)],
    )
    report = CustomRiggedCharacterValidator().validate(project, snapshot)
    issue = next(issue for issue in report.issues if issue.code == "unassigned_zero_duration_action")
    assert issue.severity == "information"


@pytest.mark.parametrize(
    ("source", "category", "alias"),
    [
        ("Borhek_Idle", "primary_idle", "cpause1"),
        ("WalkCycle", "walk", "cwalk"),
        ("FAST-RUN", "run", "crun"),
        ("RIG|RIG|borhek_breathaction", "primary_idle", "cpause1"),
        ("RIG|RIG|borhek_runattack01", "attack", "m0a1"),
        ("RIG|RIG|borhek_headbutt", "attack", "m0a2"),
        ("RIG|RIG|borhek_gethit01", "damage_reaction", "cdamages"),
        ("RIG|RIG|borhek_death", "death", "cdie"),
        ("RIG|RIG|borhek_defenseloop", "combat_ready", "creadyr"),
        ("RIG|RIG|borhek_defensemode", "combat_ready", "creadyrtw"),
        ("RIG|RIG|borhek_walkbackward", "unassigned", ""),
        ("Roar_A", "roar_taunt", "ctaunt"),
    ],
)
def test_semantic_mapping(source: str, category: str, alias: str) -> None:
    assert suggest_semantic_mapping(source) == (category, alias)


def test_custom_animation_id_collision_detection() -> None:
    project = _project()
    project.custom_animation_registrations = [
        CustomAnimationRegistration(name="test_beast_roar", animation_id=12000, source_clip="roar")
    ]
    report = CustomRiggedCharacterValidator().validate(
        project, CustomRiggedCharacterSnapshot(nodes=[RigNodeSnapshot("root")]),
        occupied_animation_ids={12000},
    )
    issue = next(issue for issue in report.issues if issue.code == "custom_animation_id_collision")
    assert issue.automatic_fix == "Allocate the next free ID"
    assert "vanilla" not in issue.fix_effect.lower()


@pytest.mark.parametrize("value", ["", "has-dash", "way_too_long_resource", "white space", "é"])
def test_resource_name_validation_rejects_unsafe_values(value: str) -> None:
    assert validate_resource_name(value) is False


def test_resource_name_validation_accepts_kotor_resref() -> None:
    assert validate_resource_name("kpm_borhek") is True
    assert validate_resource_name("ABC_123") is True


def test_model_rebuild_accepts_only_intact_outputs_owned_by_the_same_project(tmp_path) -> None:
    project = _project()
    mdl = tmp_path / "test_beast.mdl"
    mdx = tmp_path / "test_beast.mdx"
    report_path = tmp_path / "test_beast.build-report.json"
    mdl.write_bytes(b"owned-mdl")
    mdx.write_bytes(b"owned-mdx")
    import hashlib

    report_path.write_text(json.dumps({
        "schema": CustomRiggedCharacterBuildService.report_schema,
        "project_id": project.project_id,
        "model_resref": project.resource_name,
        "output_hashes": {
            mdl.name: hashlib.sha256(mdl.read_bytes()).hexdigest(),
            mdx.name: hashlib.sha256(mdx.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")
    targets = {"mdl": mdl, "mdx": mdx, "report": report_path}
    service = CustomRiggedCharacterBuildService()
    assert service._verify_owned_previous_build(project, targets) == ""

    mdl.write_bytes(b"user-modified")
    assert "changed after the prior build" in service._verify_owned_previous_build(project, targets)


def test_texture_uv_validation_distinguishes_repeat_wrapping() -> None:
    base = dict(
        material_name="skin", texture_resref="borhek01", source_format="png",
        texture_size=(1024, 1024), uvs=((-0.2, 0.5), (1.2, 0.5)),
    )
    repeat_report = CustomRiggedCharacterValidator().validate(
        _project(), CustomRiggedCharacterSnapshot(
            nodes=[RigNodeSnapshot("root")], materials=[MaterialSnapshot(**base, wrap_mode="repeat")]
        )
    )
    clamp_report = CustomRiggedCharacterValidator().validate(
        _project(), CustomRiggedCharacterSnapshot(
            nodes=[RigNodeSnapshot("root")], materials=[MaterialSnapshot(**base, wrap_mode="clamp")]
        )
    )
    assert "uv_repeat_required" in _codes(repeat_report)
    assert "uv_outside_without_repeat" in _codes(clamp_report)


def test_project_serialization_is_portable_and_preserves_all_workflow_decisions(tmp_path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "creature.fbx"
    source.write_bytes(b"read-only fixture bytes")
    original = source.read_bytes()
    project_dir = tmp_path / "project"
    project_path = project_dir / "creature.ghostcharacter.json"
    project = _project()
    project.primary_fbx = SourceAsset(str(source), "0" * 64, "primary_fbx", True)
    project.selected_skeleton_root = "foreign_root"
    project.export_nodes = {"foreign_root": True, "control": False}
    project.ground_contact_nodes = ["claw_l", "claw_r"]
    project.runtime_height_offset = 1.9724489450454712
    project.runtime_height_source = "root_joint"
    project.gameplay_settings["replace_test_placement"] = True
    project.animation_mappings = [AnimationMapping(
        source_name="Walk", assignment="vanilla_behavior_alias", exported_name="cwalk",
        confirmed=True, loop=True, root_motion="in_place",
    )]
    project.build_destination = str(project_dir / "build")

    save_custom_rigged_character_project(project, project_path)
    loaded = load_custom_rigged_character_project(project_path)
    payload = json.loads(project_path.read_text(encoding="utf-8"))

    assert not Path(payload["source_assets"]["primary_fbx"]["path"]).is_absolute()
    assert loaded.selected_skeleton_root == "foreign_root"
    assert loaded.runtime_height_offset == pytest.approx(1.9724489450454712)
    assert loaded.runtime_height_source == "root_joint"
    assert loaded.gameplay_settings["replace_test_placement"] is True
    assert loaded.animation_mappings[0].exported_name == "cwalk"
    assert tuple(loaded.workflow_steps)[-1] == "install_test"
    assert source.read_bytes() == original


def test_project_schema_zero_migrates_without_humanoid_template() -> None:
    project = CustomRiggedCharacterProject.from_dict({
        "schema_version": 0,
        "source_fbx": "assets/creature.fbx",
        "animation_files": ["assets/walk.fbx"],
        "resref": "old_beast",
        "game": "K1",
    })
    assert project.schema_version == 4
    assert project.creature_sound_cues == []
    assert project.primary_fbx.path == "assets/creature.fbx"
    assert project.external_animation_assets[0].path == "assets/walk.fbx"
    assert project.native_template_model == ""


def test_project_save_refuses_to_replace_a_different_project(tmp_path) -> None:
    path = tmp_path / "shared.ghostcharacter.json"
    save_custom_rigged_character_project(_project(), path)
    with pytest.raises(FileExistsError):
        save_custom_rigged_character_project(_project(), path)
