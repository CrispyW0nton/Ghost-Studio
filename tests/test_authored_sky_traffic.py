"""Focused contracts for Map Studio sky-traffic authoring intent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _configure_native_python_roots() -> None:
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for item in reversed(_python_roots(ROOT)):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)


def _traffic(**changes):
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import create_authored_sky_traffic

    values = {
        "traffic_id": "skytraffic_brith_01",
        "name": "Dantooine Brith",
        "room_resref": "m14aa_01f",
        "model_resref": "c_brith",
        "control_points": (
            {"id": "start", "position": (0.0, 0.0, 0.0), "metadata": {"role": "launch"}},
            {"id": "corner", "position": (10.0, 0.0, 0.0)},
            {"id": "finish", "position": (10.0, 10.0, 0.0)},
        ),
        "animation_name": "animloop2",
        "model_animation_name": "cpause1",
        "duration_seconds": 20.0,
        "facing_mode": "path_tangent",
        "position_offset": (1.0, 2.0, 3.0),
        "altitude_offset": 7.0,
        "metadata": {"fixture": "m14aa_01f"},
    }
    values.update(changes)
    return create_authored_sky_traffic(**values)


def test_sky_traffic_kmap_roundtrip_preserves_stable_ids_and_unknown_fields() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import (
        SKY_TRAFFIC_COMPILER_TARGET,
        SKY_TRAFFIC_RUNTIME_CONTAINER,
        authored_sky_traffic_from_kmap,
        authored_sky_traffic_list_from_kmap,
        authored_sky_traffic_list_to_kmap,
        authored_sky_traffic_to_kmap,
        normalise_authored_sky_traffic,
    )

    traffic = _traffic()
    payload = authored_sky_traffic_to_kmap(traffic)
    assert payload["schema"] == "ghostrigger.map_studio.sky_traffic.v1"
    assert payload["compiler_target"] == SKY_TRAFFIC_COMPILER_TARGET == "room_mdl_animation"
    assert payload["runtime_container"] == SKY_TRAFFIC_RUNTIME_CONTAINER == "room_mdl_mdx"
    assert payload["git_placement"] is False
    assert payload["id"] == "skytraffic_brith_01"
    assert payload["path"]["control_points"][0]["id"] == "start"
    assert payload["timing"] == {
        "loop": True,
        "duration_seconds": 20.0,
        "speed_units_per_second": None,
    }
    assert authored_sky_traffic_from_kmap(payload) == traffic
    assert authored_sky_traffic_list_from_kmap(authored_sky_traffic_list_to_kmap((traffic,))) == (traffic,)
    assert json.dumps(authored_sky_traffic_to_kmap(traffic), sort_keys=True) == json.dumps(payload, sort_keys=True)

    payload["future_room_animation_field"] = {"opaque": [1, 2, 3]}
    future = authored_sky_traffic_from_kmap(payload)
    assert authored_sky_traffic_to_kmap(future)["future_room_animation_field"] == {"opaque": [1, 2, 3]}

    legacy = {
        "room": "m14aa_01f",
        "model": "c_brith",
        "path_points": ((0, 0, 0), (5, 0, 0)),
        "speed": 2.0,
        "loop": "true",
        "enabled": "false",
    }
    first = normalise_authored_sky_traffic(legacy)
    second = normalise_authored_sky_traffic(legacy)
    assert first.traffic_id == second.traffic_id
    assert first.traffic_id.startswith("skytraffic_legacy_")
    assert first.duration_seconds is None
    assert first.speed_units_per_second == 2.0
    assert first.loop is True
    assert first.enabled is False


def test_sky_traffic_validation_enforces_room_animation_contract() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import (
        normalise_authored_sky_traffic,
        validate_authored_sky_traffic,
        validate_authored_sky_traffic_collection,
    )

    traffic = _traffic()
    valid = validate_authored_sky_traffic(traffic, room_resrefs={"m14aa_01f"})
    assert valid.ok is True
    assert {issue.code for issue in valid.issues} == {"SKY_TRAFFIC_LOOP_RESET"}

    invalid = normalise_authored_sky_traffic(
        {
            "id": "bad id",
            "room_resref": "missing-room-with-a-name-that-is-too-long",
            "model_resref": "",
            "control_points": (
                {"id": "same", "position": (0, 0, float("nan"))},
                {"id": "same", "position": (0, 0, 0)},
            ),
            "animation_name": "fly_forever",
            "loop": False,
            "duration_seconds": -1,
            "speed_units_per_second": 0,
            "facing_mode": "camera_billboard",
            "fixed_facing_degrees": "not-a-number",
            "position_offset": (float("inf"), 0, 0),
            "interpolation": "bezier",
        }
    )
    result = validate_authored_sky_traffic(invalid, room_resrefs={"m14aa_01f"})
    codes = {issue.code for issue in result.issues}
    assert result.ok is False
    assert {
        "SKY_TRAFFIC_ID_INVALID",
        "SKY_TRAFFIC_RESREF_INVALID",
        "SKY_TRAFFIC_ROOM_MISSING",
        "SKY_TRAFFIC_LOOP_SLOT_INVALID",
        "SKY_TRAFFIC_NON_LOOP_UNSUPPORTED",
        "SKY_TRAFFIC_INTERPOLATION_UNSUPPORTED",
        "SKY_TRAFFIC_FACING_UNSUPPORTED",
        "SKY_TRAFFIC_FACING_INVALID",
        "SKY_TRAFFIC_OFFSET_INVALID",
        "SKY_TRAFFIC_POINT_ID_INVALID",
        "SKY_TRAFFIC_POINT_INVALID",
        "SKY_TRAFFIC_DURATION_INVALID",
        "SKY_TRAFFIC_SPEED_INVALID",
        "SKY_TRAFFIC_TIMING_MISSING",
    } <= codes

    duplicate = replace(traffic)
    conflicting_period = replace(traffic, traffic_id="skytraffic_brith_02", duration_seconds=10.0)
    collection = validate_authored_sky_traffic_collection(
        (traffic, duplicate, conflicting_period),
        room_resrefs={"m14aa_01f"},
    )
    collection_codes = {issue.code for issue in collection.issues}
    assert collection.ok is False
    assert "SKY_TRAFFIC_ID_DUPLICATE" in collection_codes
    assert "SKY_TRAFFIC_LOOP_SLOT_PERIOD_CONFLICT" in collection_codes


def test_sky_traffic_sampling_is_deterministic_for_duration_speed_and_facing() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import sample_sky_traffic, sky_traffic_effective_duration

    traffic = _traffic()
    at_five = sample_sky_traffic(traffic, 5.0)
    at_fifteen = sample_sky_traffic(traffic, 15.0)
    wrapped = sample_sky_traffic(traffic, 25.0)
    assert at_five.position == (6.0, 2.0, 10.0)
    assert at_five.travel_direction == (1.0, 0.0, 0.0)
    assert at_five.facing_direction == at_five.travel_direction
    assert at_fifteen.position == (11.0, 7.0, 10.0)
    assert at_fifteen.travel_direction == (0.0, 1.0, 0.0)
    assert wrapped == at_five

    by_speed = replace(traffic, duration_seconds=None, speed_units_per_second=2.0)
    assert sky_traffic_effective_duration(by_speed) == 10.0
    assert sample_sky_traffic(by_speed, 5.0).position == (11.0, 2.0, 10.0)

    fixed = replace(traffic, facing_mode="fixed", fixed_facing_degrees=90.0)
    fixed_facing = sample_sky_traffic(fixed, 5.0).facing_direction
    assert fixed_facing is not None
    assert math.isclose(fixed_facing[0], 0.0, abs_tol=1.0e-12)
    assert math.isclose(fixed_facing[1], 1.0, abs_tol=1.0e-12)
    assert sample_sky_traffic(replace(traffic, facing_mode="preserve_model"), 5.0).facing_direction is None


def test_sky_traffic_fades_hides_and_resets_without_a_visible_teleport() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import (
        authored_sky_traffic_from_kmap,
        authored_sky_traffic_to_kmap,
        sample_sky_traffic,
        sky_traffic_effective_period,
        validate_authored_sky_traffic,
    )

    traffic = _traffic(
        duration_seconds=20.0,
        fade_in_seconds=2.0,
        fade_out_seconds=5.0,
        hidden_reset_seconds=10.0,
    )
    payload = authored_sky_traffic_to_kmap(traffic)
    assert payload["visibility"] == {
        "fade_in_seconds": 2.0,
        "fade_out_seconds": 5.0,
        "hidden_reset_seconds": 10.0,
    }
    assert authored_sky_traffic_from_kmap(payload) == traffic
    assert sky_traffic_effective_period(traffic) == 30.0
    assert sample_sky_traffic(traffic, 0.0).visibility_alpha == 0.0
    assert sample_sky_traffic(traffic, 1.0).visibility_alpha == 0.5
    assert sample_sky_traffic(traffic, 10.0).visibility_alpha == 1.0
    assert sample_sky_traffic(traffic, 17.5).visibility_alpha == 0.5
    hidden = sample_sky_traffic(traffic, 25.0)
    assert hidden.hidden_for_reset is True
    assert hidden.visibility_alpha == 0.0
    restarted = sample_sky_traffic(traffic, 31.0)
    assert restarted.hidden_for_reset is False
    assert restarted.visibility_alpha == 0.5
    assert "SKY_TRAFFIC_LOOP_RESET" not in {
        issue.code for issue in validate_authored_sky_traffic(traffic).issues
    }


def test_sky_traffic_preview_path_and_arrows_are_repeatable_and_renderer_neutral() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_sky_traffic import build_sky_traffic_preview

    traffic = _traffic()
    first = build_sky_traffic_preview(traffic, path_sample_count=5, arrow_count=2)
    second = build_sky_traffic_preview(traffic, path_sample_count=5, arrow_count=2)
    assert first == second
    assert first.compiler_target == "room_mdl_animation"
    assert first.path_length == 20.0
    assert first.duration_seconds == 20.0
    assert first.path_points == (
        (1.0, 2.0, 10.0),
        (6.0, 2.0, 10.0),
        (11.0, 2.0, 10.0),
        (11.0, 7.0, 10.0),
        (11.0, 12.0, 10.0),
    )
    assert tuple(arrow.normalized_distance for arrow in first.arrows) == (0.25, 0.75)
    assert tuple(arrow.position for arrow in first.arrows) == ((6.0, 2.0, 10.0), (11.0, 7.0, 10.0))
    assert tuple(arrow.travel_direction for arrow in first.arrows) == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_sky_traffic_project_extra_helpers_replace_without_touching_other_sections() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_module_kmap_bridge import (
        authored_project_from_kmap_payload,
        authored_project_to_kmap_payload,
    )
    from src.core.modules.authored_room_presets import create_authored_module_from_room_preset
    from src.core.modules.authored_sky_traffic import (
        read_authored_project_sky_traffic,
        write_authored_project_sky_traffic,
    )

    project = create_authored_module_from_room_preset(
        preset_id="rectangular_dev_room",
        module_root="grsky",
        game="K2",
    )
    project = replace(project, extra={**dict(project.extra), "preserve_me": {"opaque": 7}})
    room_resref = project.rooms[0].normalised_resref()
    traffic = replace(_traffic(), room_resref=room_resref)

    updated = write_authored_project_sky_traffic(project, (traffic,))
    assert updated is not project
    assert "sky_traffic" not in project.extra
    assert updated.extra["preserve_me"] == {"opaque": 7}
    assert isinstance(updated.extra["sky_traffic"], list)
    assert updated.extra["sky_traffic"][0]["compiler_target"] == "room_mdl_animation"
    assert updated.extra["sky_traffic"][0]["git_placement"] is False
    assert read_authored_project_sky_traffic(updated) == (traffic,)

    kmap_payload = authored_project_to_kmap_payload(updated)
    reopened = authored_project_from_kmap_payload(kmap_payload, fallback_name="grsky", fallback_game="K2")
    assert read_authored_project_sky_traffic(reopened) == (traffic,)

    with pytest.raises(ValueError, match="missing room"):
        write_authored_project_sky_traffic(project, (replace(traffic, room_resref="missing"),))


def test_controller_creates_kmap_traffic_path_arrows_and_export_gate() -> None:
    _configure_native_python_roots()
    from src.core.modules.authored_module_export import build_authored_module
    from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload
    from src.core.modules.module_editor_controller import ModuleEditorController

    controller = ModuleEditorController()
    controller.new_project(name="grtraffic", game="K1")
    controller.create_authored_room_preset_module(
        preset_id="rectangular_dev_room",
        module_root="grtraffic",
    )
    room_resref = controller.authored_room_resrefs()[0]
    traffic, message, validation = controller.create_authored_sky_traffic(
        room_resref=room_resref,
        model_resref="c_brith",
        start=(0.0, 0.0, 20.0),
        end=(100.0, 25.0, 30.0),
        name="Brith Patrol",
        animation_name="animloop2",
        duration_seconds=63.833,
    )

    assert validation.ok is True
    assert traffic.is_git_placement is False
    assert "room animation export" in message
    assert controller.authored_sky_traffic() == (traffic,)
    geometry = controller.authored_sky_traffic_marker_geometry()
    assert geometry.marker_count == 1
    assert any(line.role == "sky_traffic_path" for line in geometry.lines)
    assert any(line.role == "sky_traffic_direction" for line in geometry.lines)
    preview_rows = controller.authored_sky_traffic_preview_rows()
    assert len(preview_rows) == 1
    assert preview_rows[0].kind == "sky_traffic"
    assert preview_rows[0].model_resref == "c_brith"
    readiness = controller.authored_module_readiness().readiness
    assert readiness.metadata["sky_traffic"]["count"] == 1
    assert readiness.metadata["sky_traffic"]["preview_ready"] is True
    assert readiness.metadata["sky_traffic"]["export_ready"] is False
    assert readiness.export_status == "Sky traffic compiler blocked"

    project = authored_project_from_kmap_payload(
        controller.project.extra_sections["authored_module"],
        fallback_name="grtraffic",
        fallback_game="K1",
    )
    assert project.extra["sky_traffic"][0]["git_placement"] is False
    build = build_authored_module(project)
    assert build.metadata["sky_traffic"]["count"] == 1
    assert build.metadata["sky_traffic"]["compiler_target"] == "room_mdl_animation"
    assert any("room-MDL animation export is not yet" in issue for issue in build.blocking_issues)


@pytest.mark.skipif(
    not Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor\chitin.key").is_file(),
    reason="K1 installation fixture unavailable",
)
def test_controller_sky_traffic_preview_resolves_actual_flying_model_without_git_template() -> None:
    _configure_native_python_roots()
    from src.core.assets.resource_manager import ResourceManager
    from src.core.modules.module_editor_controller import ModuleEditorController

    resources = ResourceManager()
    assert resources.set_k1_dir(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
    controller = ModuleEditorController()
    controller.new_project(name="grtraffic", game="K1")
    controller.create_authored_room_preset_module(
        preset_id="rectangular_dev_room",
        module_root="grtraffic",
    )
    controller.create_authored_sky_traffic(
        room_resref=controller.authored_room_resrefs()[0],
        model_resref="c_brith",
        start=(0.0, 0.0, 20.0),
        end=(50.0, 0.0, 25.0),
        duration_seconds=20.0,
    )
    preview = controller.map_studio_viewport_preview_model(resources)

    assert preview is not None
    resolved = tuple(controller.last_map_studio_resolved_placement_ids)
    traffic_resolved = tuple(value for value in resolved if value.startswith("sky_traffic:"))
    assert len(traffic_resolved) == 1
    traffic_meshes = [
        node
        for node in preview.all_nodes()
        if str(getattr(node, "_gr_map_studio_placement_kind", "")) == "sky_traffic"
    ]
    assert traffic_meshes
    assert all(str(getattr(node, "_gr_map_studio_placement_id", "")).startswith("sky_traffic:") for node in traffic_meshes)


def test_sky_traffic_source_and_payload_mirrors_are_exact() -> None:
    paths = (
        ROOT / "src/core/modules/authored_sky_traffic.py",
        ROOT / "native/GhostRigger.Core.Scene/Python/src/core/modules/authored_sky_traffic.py",
        ROOT / "native/GhostRigger.Core.Tools/Python/src/core/modules/authored_sky_traffic.py",
    )
    assert len({path.read_bytes() for path in paths}) == 1

    spec = importlib.util.spec_from_file_location(
        "ghostrigger_scene_payload_authored_sky_traffic",
        paths[1],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    assert "GhostRigger.Core.Scene" in str(Path(module.__file__).resolve())
    source = paths[0].read_text(encoding="utf-8")
    assert 'SKY_TRAFFIC_COMPILER_TARGET = "room_mdl_animation"' in source
    assert '"git_placement": False' in source
    assert "authored_module_placements" not in source


def test_sky_traffic_is_manifested_in_scene_and_tools_payloads() -> None:
    source_path = "src/core/modules/authored_sky_traffic.py"
    packaged_path = "Python/src/core/modules/authored_sky_traffic.py"
    for project in ("GhostRigger.Core.Scene", "GhostRigger.Core.Tools"):
        project_dir = ROOT / "native" / project
        manifest = json.loads((project_dir / "GhostRiggerPythonPayload.json").read_text(encoding="utf-8"))
        row = next(item for item in manifest["files"] if item["source_path"] == source_path)
        packaged = project_dir / packaged_path
        assert row["packaged_path"] == packaged_path
        assert row["sha256"] == hashlib.sha256(packaged.read_bytes()).hexdigest()
        assert 'PYTHON_PAYLOAD_CORE_MODULES_AUTHORED_SKY_TRAFFIC RCDATA "Python/src/core/modules/authored_sky_traffic.py"' in (
            project_dir / "GhostRiggerPythonPayload.rc"
        ).read_text(encoding="utf-8")
        assert 'Python\\src\\core\\modules\\authored_sky_traffic.py' in (
            project_dir / f"{project}.vcxproj"
        ).read_text(encoding="utf-8")
