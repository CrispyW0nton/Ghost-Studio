"""Focused contracts for package-owned secondary-motion authoring."""

from __future__ import annotations

import json
import struct

import pytest

from src.core.project.secondary_motion_profile import SecondaryMotionProfile
from src.core.project.custom_rigged_character_project import (
    CUSTOM_RIGGED_WORKFLOW_STEPS,
    CustomRiggedCharacterProject,
)
from src.core.project.placeable_asset import PlaceableAsset
from src.core.secondary_motion.profile_io import (
    deserialize_secondary_motion_profile,
    serialize_secondary_motion_profile,
    verify_raw_dangly_payload,
)
from src.core.secondary_motion.workflow import (
    SecondaryMotionPreview,
    apply_read_only_calibration,
    compile_secondary_motion_profile,
)
from src.core.validation.secondary_motion_validator import (
    validate_secondary_motion_profile,
)


def _profile_payload() -> dict:
    return {
        "schema": "ghostrigger.secondary_motion_profile.v1",
        "schema_version": 1,
        "profile_id": "secondary_motion_test_robes",
        "display_label": "Test robes",
        "target_game": "K2",
        "backend_capability_version": 1,
        "model_identity": {
            "model_resrefs": ["testrobe"],
            "mdl_sha256": "a" * 64,
            "mdx_sha256": "b" * 64,
            "hierarchy_fingerprint": "c" * 64,
            "root_name": "root",
            "future_identity": {"keep": True},
        },
        "regions": [
            {
                "region_id": "robe_back",
                "label": "Robe back",
                "semantic_type": "robe",
                "backend": "runtime_pbd",
                "mesh_membership": ["robe_mesh"],
                "vertex_group": "robe_back",
                "drivers": [
                    {
                        "name": "custom_robe_01",
                        "parent_name": "pelvis",
                        "rest_position": [0.0, -0.1, 0.2],
                        "rest_quaternion": [0.0, 0.0, 0.0, 1.0],
                        "rest_scale": [1.0, 1.0, 1.0],
                        "offline_preorder": 42,
                    }
                ],
                "anchors": {"driver_names": ["pelvis"], "skin_weight_evidence": {}},
                "colliders": [],
                "solver": {
                    "preset": "robe_skirt",
                    "gravity": 4.25,
                    "damping": 4.5,
                    "animation_follow": 0.65,
                },
                "future_region": "preserve-me",
            }
        ],
        "preview_tests": [{"motion": "walk", "passed": True}],
        "validation_evidence": {},
        "runtime": {
            "state": "capture_only",
            "capture_session": "session-1",
            "compiler_version": "1",
            "write_enabled": False,
        },
        "future_top_level": {"keep": [1, 2, 3]},
    }


def test_profile_round_trip_preserves_unknown_fields_and_custom_driver_names() -> None:
    profile = SecondaryMotionProfile.from_dict(_profile_payload())

    output = profile.to_dict()

    assert output["future_top_level"] == {"keep": [1, 2, 3]}
    assert output["model_identity"]["future_identity"] == {"keep": True}
    assert output["regions"][0]["future_region"] == "preserve-me"
    assert output["regions"][0]["drivers"][0]["name"] == "custom_robe_01"


def test_standalone_profile_serialization_is_deterministic_and_lf_terminated() -> None:
    first = serialize_secondary_motion_profile(_profile_payload())
    reordered = dict(reversed(list(_profile_payload().items())))
    second = serialize_secondary_motion_profile(reordered)

    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert deserialize_secondary_motion_profile(first).to_dict()["profile_id"] == "secondary_motion_test_robes"
    assert json.loads(first)["future_top_level"] == {"keep": [1, 2, 3]}


def test_validator_blocks_model_write_promotion_without_exact_read_only_evidence() -> None:
    payload = _profile_payload()
    payload["runtime"] = {"state": "write_enabled", "write_enabled": True}

    report = validate_secondary_motion_profile(payload, target_kind="custom_character")

    assert report.can_export_profile is True
    assert report.can_enable_runtime_writes is False
    assert "runtime_write_evidence_incomplete" in {issue.code for issue in report.issues}


def test_validator_keeps_placeable_pbd_capture_only_without_lifecycle_proof() -> None:
    payload = _profile_payload()
    payload["runtime"] = {
        "state": "write_enabled",
        "write_enabled": True,
        "model_identity_validated": True,
        "live_tree_validated": True,
        "visible_test_accepted": True,
    }

    report = validate_secondary_motion_profile(payload, target_kind="placeable")

    assert report.can_enable_runtime_writes is False
    assert "placeable_runtime_lifecycle_unproven" in {issue.code for issue in report.issues}


def test_validator_rejects_one_mesh_region_assigned_to_two_dynamic_backends() -> None:
    payload = _profile_payload()
    duplicate = dict(payload["regions"][0])
    duplicate.update({"region_id": "robe_native", "backend": "native_dangly"})
    payload["regions"].append(duplicate)

    report = validate_secondary_motion_profile(payload)

    assert report.can_export_profile is False
    assert "mesh_backend_conflict" in {issue.code for issue in report.issues}


def test_validator_rejects_non_unit_quaternion_and_duplicate_runtime_ordinal() -> None:
    payload = _profile_payload()
    duplicate = dict(payload["regions"][0])
    duplicate.update({
        "region_id": "robe_front",
        "mesh_membership": ["robe_front_mesh"],
        "drivers": [dict(payload["regions"][0]["drivers"][0], name="custom_robe_02")],
    })
    payload["regions"][0]["drivers"][0]["rest_quaternion"] = [0.0, 0.0, 0.0, 2.0]
    payload["regions"].append(duplicate)

    report = validate_secondary_motion_profile(payload)
    codes = {issue.code for issue in report.issues}

    assert "invalid_driver_rest_quaternion" in codes
    assert "duplicate_driver_preorder" in codes


def test_compiler_emits_guarded_named_selectors_without_runtime_addresses() -> None:
    compiled = compile_secondary_motion_profile(_profile_payload())

    text = serialize_secondary_motion_profile(compiled).decode("utf-8")
    assert compiled["compiled"]["regions"][0]["drivers"][0]["name"] == "custom_robe_01"
    assert compiled["model_identity"]["target_kind"] == "character"
    assert compiled["compiled"]["model_selector"]["target_kind"] == "character"
    assert compiled["runtime"]["write_enabled"] is False
    assert "address" not in text.casefold()
    assert "pointer" not in text.casefold()
    assert "runtime_key" not in text.casefold()


def test_compiler_records_placeable_kind_but_still_forces_capture_only() -> None:
    compiled = compile_secondary_motion_profile(_profile_payload(), target_kind="placeable")

    assert compiled["model_identity"]["target_kind"] == "placeable"
    assert compiled["runtime"] == {
        **_profile_payload()["runtime"],
        "state": "capture_only",
        "write_enabled": False,
    }


def test_read_only_capture_promotes_only_to_calibrated_and_never_writes() -> None:
    evidence = {
        "schema": "kotor_patch_manager.cloth_dynamics_k2.profile_capture.v1",
        "profile_id": "secondary_motion_test_robes",
        "model_resref": "testrobe",
        "capture_session": "session-42",
        "passed": True,
        "expected_nodes": 1,
        "observed_nodes": 58,
        "writes": 0,
    }

    calibrated = apply_read_only_calibration(
        _profile_payload(), evidence, target_kind="placeable"
    )

    assert calibrated["runtime"]["state"] == "calibrated"
    assert calibrated["runtime"]["model_identity_validated"] is True
    assert calibrated["runtime"]["live_tree_validated"] is True
    assert calibrated["runtime"]["visible_test_accepted"] is False
    assert calibrated["runtime"]["placeable_lifecycle_validated"] is False
    assert calibrated["runtime"]["write_enabled"] is False


def test_read_only_calibration_rejects_mismatched_or_writing_capture() -> None:
    evidence = {
        "schema": "kotor_patch_manager.cloth_dynamics_k2.profile_capture.v1",
        "profile_id": "secondary_motion_test_robes",
        "model_resref": "wrong_model",
        "capture_session": "session-42",
        "passed": True,
        "expected_nodes": 1,
        "writes": 0,
    }
    with pytest.raises(ValueError, match="exact profile model identity"):
        apply_read_only_calibration(_profile_payload(), evidence)

    evidence["model_resref"] = "testrobe"
    evidence["writes"] = 1
    with pytest.raises(ValueError, match="writes=0"):
        apply_read_only_calibration(_profile_payload(), evidence)


def test_fixed_step_preview_keeps_anchor_fixed_and_is_chunking_deterministic() -> None:
    rest = [(0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, -2.0)]
    whole = SecondaryMotionPreview(rest, pinned_indices={0}, gravity=(0.0, 0.0, -4.25), damping=4.5)
    chunked = SecondaryMotionPreview(rest, pinned_indices={0}, gravity=(0.0, 0.0, -4.25), damping=4.5)

    whole.advance(1.0 / 30.0)
    for _ in range(4):
        chunked.advance(1.0 / 120.0)

    for whole_position, chunked_position in zip(whole.positions, chunked.positions, strict=True):
        assert whole_position == pytest.approx(chunked_position)
    assert whole.positions[0] == pytest.approx(rest[0])
    assert whole.step_count == chunked.step_count == 4


def test_raw_dangly_verifier_uses_zero_constraint_as_fixed_anchor_and_checks_rest_array() -> None:
    base = 12
    header_abs = 64
    constraint_abs = 128
    rest_abs = 160
    payload = bytearray(256)
    struct.pack_into("<IIIfffI", payload, header_abs, constraint_abs - base, 3, 3, 0.2, 0.5, 1.0, rest_abs - base)
    struct.pack_into("<fff", payload, constraint_abs, 0.0, 127.5, 255.0)
    expected_rest = ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, -2.0))
    for index, value in enumerate(expected_rest):
        struct.pack_into("<fff", payload, rest_abs + index * 12, *value)

    report = verify_raw_dangly_payload(bytes(payload), header_abs=header_abs, vertex_count=3)

    assert report.valid is True
    assert report.anchor_indices == (0,)
    assert report.constraints == pytest.approx((0.0, 0.5, 1.0))
    for actual, expected in zip(report.rest_positions, expected_rest, strict=True):
        assert actual == pytest.approx(expected)


def test_raw_dangly_verifier_fails_closed_on_mismatched_array_counts() -> None:
    payload = bytearray(128)
    struct.pack_into("<IIIfffI", payload, 32, 80, 2, 2, 0.2, 0.5, 1.0, 96)

    report = verify_raw_dangly_payload(bytes(payload), header_abs=32, vertex_count=3)

    assert report.valid is False
    assert "vertex_count_mismatch" in report.errors


def test_custom_character_schema_migrates_secondary_motion_step_and_unknown_fields() -> None:
    old = CustomRiggedCharacterProject(creature_name="Old project").to_dict()
    old["schema_version"] = 3
    old.pop("secondary_motion", None)
    old["future_parent_field"] = {"do_not_drop": True}
    old["workflow_steps"] = [step for step in old["workflow_steps"] if step != "secondary_motion"]

    migrated = CustomRiggedCharacterProject.from_dict(old)
    output = migrated.to_dict()

    assert output["schema_version"] == 4
    assert output["future_parent_field"] == {"do_not_drop": True}
    assert output["secondary_motion"]["schema"] == "ghostrigger.secondary_motion_profile.v1"
    assert tuple(output["workflow_steps"]) == CUSTOM_RIGGED_WORKFLOW_STEPS
    materials = output["workflow_steps"].index("materials_uvs")
    assert output["workflow_steps"][materials + 1] == "secondary_motion"
    assert output["workflow_steps"][materials + 2] == "gameplay"


def test_placeable_schema_migrates_embedded_profile_and_preserves_unknown_fields() -> None:
    old = PlaceableAsset(template_resref="crate01").to_dict()
    old["schema_version"] = 1
    old.pop("secondary_motion", None)
    old["future_parent_field"] = ["keep"]

    migrated = PlaceableAsset.from_dict(old)
    output = migrated.to_dict()

    assert output["schema_version"] == 2
    assert output["future_parent_field"] == ["keep"]
    assert output["secondary_motion"]["schema"] == "ghostrigger.secondary_motion_profile.v1"
    assert output["secondary_motion"]["runtime"]["write_enabled"] is False


def test_authored_module_accepts_capture_only_sidecar_but_rejects_placeable_writes() -> None:
    from src.core.modules.authored_module_export import (
        AuthoredModuleBuild,
        _prepare_secondary_motion_sidecars,
    )

    capture = _profile_payload()
    capture["profile_id"] = "placeable_banner"
    capture["runtime"] = {"state": "capture_only", "write_enabled": False}
    payload = serialize_secondary_motion_profile(capture)
    build = AuthoredModuleBuild(module_root="test", game="K2", module=None, project=None)

    accepted = _prepare_secondary_motion_sidecars(build, (("placeable_banner", payload),))

    assert accepted == (("placeable_banner", payload),)
    assert build.blocking_issues == []
    assert build.metadata["secondary_motion_profiles"][0]["runtime_dependency"] == "cloth-dynamics-k2"

    requested = dict(capture)
    requested["runtime"] = {"state": "write_enabled", "write_enabled": True}
    denied = AuthoredModuleBuild(module_root="test", game="K2", module=None, project=None)
    assert _prepare_secondary_motion_sidecars(
        denied,
        (("placeable_banner", serialize_secondary_motion_profile(requested)),),
    ) == ()
    assert any("requested writes" in issue for issue in denied.blocking_issues)
