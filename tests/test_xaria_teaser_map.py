from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.build_xaria_teaser_map as teaser_builder
from scripts.build_xaria_teaser_map import (
    BUNDLED_ENCOUNTER_RESOURCES,
    CAMERA_MARKERS,
    DIRECTOR_POWER_SCRIPT_RESREFS,
    ENCOUNTER_DEPENDENCIES,
    ENCOUNTER_PROXIMITY_METRES,
    ENCOUNTER_TRIGGER_GEOMETRY,
    ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES,
    ENCOUNTER_TRIGGER_POSITION,
    ENTRY_POINT,
    EXTERNAL_XARIA_FILES,
    EXTERNAL_XARIA_STREAMVOICE_FILES,
    HERO_CLEARING_CENTER,
    MODULE_ROOT,
    PRIVATE_ACTIVE_BEAT_LOCAL,
    PRIVATE_DIALOGUE,
    PRIVATE_DIRECTOR_TAG,
    PRIVATE_DIRECTOR_TARGET_LOCAL,
    PRIVATE_DIRECTOR_TEMPLATE,
    PRIVATE_POWER_ROWS,
    PRIVATE_POWER_TOKENS,
    PRIVATE_SCHEMA,
    PRIVATE_SCHEMA_LOCAL,
    PRIVATE_SCRIPT_SOURCES,
    PRIVATE_STATE_LOCAL,
    PRIVATE_TRIGGER_TEMPLATE,
    PRIVATE_WRAID_FACTION_ID,
    PRIVATE_WRAID_TAGS,
    PRIVATE_WRAID_TEMPLATES,
    PRIVATE_XARIA_FACTION_ID,
    PRIVATE_XARIA_TAG,
    PRIVATE_XARIA_TEMPLATE,
    PRODUCTION_LIP_SHA256,
    PRODUCTION_VOICE_RESREFS,
    PRODUCTION_VOICE_SHA256,
    PRODUCTION_VOICE_SOURCE_TO_RUNTIME,
    ROOM_PIECE_ID,
    ROUTE_POINTS,
    SHOWCASE_BEATS,
    SOURCE_GAME,
    TEASER_VOICE_LOOKUP,
    TERRAIN_DRESSING,
    VOICE_DIALOGUE_RESREF,
    VOICE_STREAM_ID,
    WORLD_LIGHTING,
    _authored_gameplay_placements,
    _camera_orientation,
    _compile_private_script_resources,
    _dependency_source_roots,
    _director_wrapper_resources,
    _encounter_trigger_route_coverage,
    _external_dependency_evidence,
    _private_dialogue_bytes,
    _require_deterministic_build_pair,
    _require_private_faction_contract,
    _require_private_script_native_action_abi,
    _require_private_script_retail_local_contract,
    _restore_playable_room_lightmaps,
    build_spatial_plan,
)
from src.core.modules.map_studio_spatial_design import audit_spatial_design

ROOT = Path(__file__).resolve().parents[1]


def test_external_dependency_hash_pins_are_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"voice-design-1-lip-data"
    expected = hashlib.sha256(data).hexdigest().upper()
    override = tmp_path / "Override"
    override.mkdir()
    (override / "xv_intro.lip").write_bytes(data)
    streamvoice = tmp_path / "StreamVoice" / "plc" / "xaria"
    streamvoice.mkdir(parents=True)
    (streamvoice / "xv_intro.wav").write_bytes(data)

    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_FILES",
        ({"name": "xv_intro.lip", "sha256": expected},),
    )
    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_STREAMVOICE_FILES",
        (
            {
                "name": "StreamVoice/plc/xaria/xv_intro.wav",
                "sha256": expected,
            },
        ),
    )

    evidence = _external_dependency_evidence(tmp_path)

    assert evidence[0]["sha256"] == expected.lower()
    assert evidence[0]["verified"] is True
    assert evidence[1]["sha256"] == expected.lower()
    assert evidence[1]["verified"] is True


@pytest.mark.parametrize(
    ("installed", "install_state"),
    ((False, "not_requested"), (True, "installed")),
)
def test_external_dependency_evidence_can_use_prepared_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
    install_state: str,
) -> None:
    data = b"prepared-xaria-voice-design-1"
    expected = hashlib.sha256(data).hexdigest()
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    candidate_root = tmp_path / "candidate"
    candidate_override = candidate_root / "Override"
    candidate_override.mkdir(parents=True)
    (candidate_override / "997xaria001.lip").write_bytes(data)
    candidate_voice = (
        candidate_root
        / "GameRoot"
        / "StreamVoice"
        / "997"
        / "xaria"
    )
    candidate_voice.mkdir(parents=True)
    (candidate_voice / "997xaria001.wav").write_bytes(data)
    candidate_module = candidate_root / "plcaa.mod"
    candidate_module.write_bytes(b"prepared-module")
    (candidate_root.parent / "stage-manifest.json").write_text(
        json.dumps(
            {
                "operation": "add_xaria_plcaa_encounter",
                "installed": installed,
                "install_state": install_state,
                "module": {
                    "candidate": {
                        "size": candidate_module.stat().st_size,
                        "sha256": hashlib.sha256(
                            candidate_module.read_bytes()
                        ).hexdigest(),
                    }
                },
                "resources": {
                    "resources": {
                        "997xaria001.lip": {
                            "size": len(data),
                            "sha256": expected,
                        }
                    }
                },
                "voice_audio": {
                    "files": {
                        "StreamVoice/997/xaria/997xaria001.wav": {
                            "size": len(data),
                            "sha256": expected,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_FILES",
        ({"name": "997xaria001.lip", "sha256": expected},),
    )
    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_STREAMVOICE_FILES",
        (
            {
                "name": "StreamVoice/997/xaria/997xaria001.wav",
                "sha256": expected,
            },
        ),
    )
    monkeypatch.setattr(
        teaser_builder,
        "PRODUCTION_VOICE_RESREFS",
        ("997xaria001",),
    )

    override_root, game_root, source = _dependency_source_roots(
        game_dir,
        candidate_root,
    )
    evidence = _external_dependency_evidence(
        game_dir,
        candidate_root=candidate_root,
    )

    assert override_root == candidate_override.resolve()
    assert game_root == (candidate_root / "GameRoot").resolve()
    assert source == "prepared_candidate"
    assert [entry["resource"] for entry in evidence[:2]] == [
        "997xaria001.lip",
        "StreamVoice/997/xaria/997xaria001.wav",
    ]
    assert all(
        entry["evidence_source"] == "prepared_candidate"
        for entry in evidence[:2]
    )
    assert all(entry["sha256"] == expected for entry in evidence[:2])


def test_prepared_candidate_dependency_layout_is_all_or_nothing(
    tmp_path: Path,
) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    candidate_root = tmp_path / "candidate"
    (candidate_root / "Override").mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="prepared Xaria candidate is incomplete",
    ):
        _dependency_source_roots(game_dir, candidate_root)


def test_prepared_candidate_dependency_rejects_manifest_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"candidate-file"
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    candidate_root = tmp_path / "candidate"
    candidate_override = candidate_root / "Override"
    candidate_override.mkdir(parents=True)
    (candidate_override / "997xaria001.lip").write_bytes(data)
    candidate_voice = (
        candidate_root
        / "GameRoot"
        / "StreamVoice"
        / "997"
        / "xaria"
    )
    candidate_voice.mkdir(parents=True)
    (candidate_voice / "997xaria001.wav").write_bytes(data)
    candidate_module = candidate_root / "plcaa.mod"
    candidate_module.write_bytes(b"prepared-module")
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "operation": "add_xaria_plcaa_encounter",
        "installed": False,
        "install_state": "not_requested",
        "module": {
            "candidate": {
                "size": candidate_module.stat().st_size,
                "sha256": hashlib.sha256(
                    candidate_module.read_bytes()
                ).hexdigest(),
            }
        },
        "resources": {
            "resources": {
                "997xaria001.lip": {
                    "size": len(data),
                    "sha256": "0" * 64,
                }
            }
        },
        "voice_audio": {
            "files": {
                "StreamVoice/997/xaria/997xaria001.wav": {
                    "size": len(data),
                    "sha256": digest,
                }
            }
        },
    }
    (candidate_root.parent / "stage-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_FILES",
        ({"name": "997xaria001.lip", "sha256": digest},),
    )
    monkeypatch.setattr(
        teaser_builder,
        "EXTERNAL_XARIA_STREAMVOICE_FILES",
        (
            {
                "name": "StreamVoice/997/xaria/997xaria001.wav",
                "sha256": digest,
            },
        ),
    )

    with pytest.raises(RuntimeError, match="stage-manifest hash/size mismatch"):
        _dependency_source_roots(game_dir, candidate_root)


def test_deterministic_build_pair_rejects_dependency_snapshot_drift() -> None:
    first = {
        "semantic_digest": "same-module",
        "external_dependency_evidence": [
            {
                "kind": "override_file",
                "resource": "xaria.dlg",
                "size": 10,
                "sha256": "a" * 64,
                "verified": True,
            }
        ],
    }
    second = {
        "semantic_digest": "same-module",
        "external_dependency_evidence": [
            {
                "kind": "override_file",
                "resource": "xaria.dlg",
                "size": 10,
                "sha256": "b" * 64,
                "verified": True,
            }
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="external dependency snapshot changed",
    ):
        _require_deterministic_build_pair(first, second)


def test_director_wrappers_come_only_from_selected_dependency_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_override = tmp_path / "selected" / "Override"
    mutable_repo_override = (
        tmp_path
        / "repo"
        / "Patches"
        / "XariaCompanionK2"
        / "powers"
        / "candidate"
        / "Override"
    )
    selected_override.mkdir(parents=True)
    mutable_repo_override.mkdir(parents=True)
    for index, resref in enumerate(DIRECTOR_POWER_SCRIPT_RESREFS, 1):
        (selected_override / f"{resref}.ncs").write_bytes(
            b"NCS V1.0" + f":selected:{index}".encode("ascii")
        )
        (mutable_repo_override / f"{resref}.ncs").write_bytes(
            b"NCS V1.0" + f":mutable-repo:{index}".encode("ascii")
        )
    monkeypatch.setattr(teaser_builder, "ROOT", tmp_path / "repo" / "Ghost-Studio")

    resources = _director_wrapper_resources(selected_override)

    assert [resref for resref, restype, _payload in resources] == list(
        DIRECTOR_POWER_SCRIPT_RESREFS
    )
    assert all(restype == "ncs" for _resref, restype, _payload in resources)
    assert all(
        b":selected:" in payload and b":mutable-repo:" not in payload
        for _resref, _restype, payload in resources
    )


def test_private_teaser_factions_require_neutral_pre_cinematic_targets() -> None:
    expected = {
        PRIVATE_XARIA_TEMPLATE: PRIVATE_XARIA_FACTION_ID,
        **{resref: PRIVATE_WRAID_FACTION_ID for resref in PRIVATE_WRAID_TEMPLATES},
    }
    _require_private_faction_contract(expected)

    prematurely_friendly = dict(expected)
    prematurely_friendly.update(
        {resref: PRIVATE_XARIA_FACTION_ID for resref in PRIVATE_WRAID_TEMPLATES}
    )
    with pytest.raises(RuntimeError, match="neutral pre-cinematic targets"):
        _require_private_faction_contract(prematurely_friendly)


def test_private_teaser_scripts_fit_retail_local_storage() -> None:
    _require_private_script_retail_local_contract()


@pytest.mark.parametrize(
    "invalid_source, message",
    (
        (
            "void main(){ SetLocalNumber(OBJECT_SELF, 60, 1); }",
            "slot 60",
        ),
        (
            "void main(){ SetLocalNumber(OBJECT_SELF, 28, 287); }",
            "stores 287",
        ),
        (
            "void main(){ SetLocalNumber(OBJECT_SELF, 28, -2); }",
            "stores -2",
        ),
    ),
)
def test_private_teaser_rejects_unrepresentable_local_numbers(
    invalid_source: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = dict(PRIVATE_SCRIPT_SOURCES)
    sources["xt_invalid"] = invalid_source
    monkeypatch.setattr(teaser_builder, "PRIVATE_SCRIPT_SOURCES", sources)

    with pytest.raises(RuntimeError, match=message):
        _require_private_script_retail_local_contract()


def test_xaria_teaser_recipe_is_k2_only_and_separate_from_plcaa() -> None:
    payload = json.dumps(
        {
            "module": MODULE_ROOT,
            "room_piece": ROOM_PIECE_ID,
            "dependencies": ENCOUNTER_DEPENDENCIES,
            "terrain": TERRAIN_DRESSING,
        },
        sort_keys=True,
    ).lower()

    assert MODULE_ROOT == "xartease"
    assert SOURCE_GAME == "K2"
    assert ROOM_PIECE_ID == "k2_402dxn_402dxna"
    assert "plcaa" not in payload
    assert "k1_" not in payload
    bundled = {
        (item["resref"], item["type"])
        for item in BUNDLED_ENCOUNTER_RESOURCES
    }
    assert bundled >= {
        ("xar_enc", "utc"),
        ("xar_introtrig", "utt"),
        ("xar_director", "utp"),
        ("xar_clock", "utp"),
        ("xar_cam", "dlg"),
        ("xaria_blade", "uti"),
        ("xar_nydak1", "utc"),
        ("xar_nydak2", "utc"),
        ("xar_nydak3", "utc"),
        ("kxar_mistsnd", "uts"),
        ("kxar_lightsnd", "uts"),
        ("kxar_drainsnd", "uts"),
        ("kxar_start", "ncs"),
        ("kxar_show", "ncs"),
        ("kxar_finish", "ncs"),
    }
    assert {item["kind"] for item in ENCOUNTER_DEPENDENCIES} >= {
        "base_game",
        "2da_row",
        "override_file",
        "runtime_patch",
        "streamvoice_file",
    }
    assert not any(
        item.get("resref", "").startswith("xt_")
        for item in BUNDLED_ENCOUNTER_RESOURCES
    )


def test_xaria_teaser_spatial_plan_is_purposeful_and_route_safe() -> None:
    plan = build_spatial_plan()
    audit = audit_spatial_design(plan)

    assert audit.ok, audit.blocking_issues
    assert audit.zone_count >= 4
    assert audit.path_count >= 2
    assert audit.placement_count >= 14
    assert audit.purposeful_placement_count == audit.placement_count
    assert audit.landmark_count >= 2
    assert plan.player_clearance >= 1.2
    assert min(path.width for path in plan.paths) >= 2.2
    assert plan.paths[0].points == ROUTE_POINTS
    assert tuple(ENTRY_POINT[:2]) == ROUTE_POINTS[0]
    assert tuple(HERO_CLEARING_CENTER[:2]) == ROUTE_POINTS[-1]


def test_xaria_encounter_start_covers_authored_and_direct_approaches() -> None:
    """The old 4x2.8m corner-touch volume was bypassed in a retail autosave."""

    assert _encounter_trigger_route_coverage() >= ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES
    center_x, center_y, _center_z = ENCOUNTER_TRIGGER_POSITION
    min_x = center_x + min(point[0] for point in ENCOUNTER_TRIGGER_GEOMETRY)
    max_x = center_x + max(point[0] for point in ENCOUNTER_TRIGGER_GEOMETRY)
    min_y = center_y + min(point[1] for point in ENCOUNTER_TRIGGER_GEOMETRY)
    max_y = center_y + max(point[1] for point in ENCOUNTER_TRIGGER_GEOMETRY)

    assert not (min_x <= ENTRY_POINT[0] <= max_x and min_y <= ENTRY_POINT[1] <= max_y)
    direct_target = (0.0, -18.0)
    direct_enter_t = (min_y - ENTRY_POINT[1]) / (direct_target[1] - ENTRY_POINT[1])
    direct_enter_x = ENTRY_POINT[0] + direct_enter_t * (direct_target[0] - ENTRY_POINT[0])
    assert 0.0 < direct_enter_t < 1.0
    assert min_x + 1.0 <= direct_enter_x <= max_x - 1.0
    assert min_y <= direct_target[1] <= max_y
    assert ENCOUNTER_PROXIMITY_METRES < math.dist(ENTRY_POINT[:2], direct_target)


def test_xaria_teaser_has_shot_and_effect_verification_coverage() -> None:
    camera_roles = {camera["role"] for camera in CAMERA_MARKERS}
    terrain_roles = {item["role"] for item in TERRAIN_DRESSING}

    assert camera_roles >= {
        "arrival_reveal",
        "ichor_lightning_hero",
        "dialogue_closeup",
        "dialogue_reverse",
    }
    assert terrain_roles >= {
        "canopy_frame",
        "foreground_root",
        "midground_vines",
        "background_tree",
    }
    assert WORLD_LIGHTING["profile"] == "custom"
    assert WORLD_LIGHTING["fog_enabled"] is True
    assert WORLD_LIGHTING["fog_far"] > WORLD_LIGHTING["fog_near"]
    assert WORLD_LIGHTING["fog_color"][1] >= WORLD_LIGHTING["fog_color"][0]


def test_xaria_teaser_cameras_use_the_retail_odyssey_encoding() -> None:
    """K1/K2 stock GIT cameras store yaw in quaternion X and pitch in degrees."""

    for row in CAMERA_MARKERS:
        orientation = _camera_orientation(row["position"], row["target"])
        assert orientation[1] == 0.0
        assert orientation[2] == 0.0
        assert abs(orientation[0]) > 0.01
        assert sum(component * component for component in orientation) == pytest.approx(1.0)
        assert 45.0 <= float(row["pitch"]) <= 105.0
        px, py, pz = (float(value) for value in row["position"])
        tx, ty, tz = (float(value) for value in row["target"])
        expected_pitch = 90.0 + math.degrees(
            math.atan2(
                tz - (pz + float(row["height"])),
                math.hypot(tx - px, ty - py),
            )
        )
        assert float(row["pitch"]) == pytest.approx(expected_pitch, abs=1.0e-5)

    assert _camera_orientation(
        CAMERA_MARKERS[0]["position"], CAMERA_MARKERS[0]["target"]
    ) == pytest.approx((0.943628319, 0.0, 0.0, -0.331006941), abs=1.0e-6)


def test_xaria_combat_cameras_keep_xaria_and_each_victim_in_frame() -> None:
    cameras = {int(row["camera_id"]): row for row in CAMERA_MARKERS}
    camera_111 = cameras[111]
    camera_112 = cameras[112]
    camera_113 = cameras[113]

    # The first shot was already approved and must remain byte-for-byte stable.
    assert camera_111["position"] == (9.0, -24.0, 10.065993)
    assert camera_111["target"] == (5.8, -21.8, 11.0)
    assert camera_111["field_of_view"] == 43.0

    assert camera_112["position"] == (9.0, -13.0, 9.986575)
    assert camera_112["target"] == (5.5, -20.0, 11.0)
    assert camera_112["field_of_view"] == 58.0
    assert camera_113["position"] == (-2.0, -10.0, 10.048286)
    assert camera_113["target"] == (4.5, -17.5, 11.0)
    assert camera_113["field_of_view"] == 58.0

    def assert_actor_fits(camera, actor_xy: tuple[float, float]) -> None:
        px, py, _pz = camera["position"]
        tx, ty, _tz = camera["target"]
        aim = math.atan2(ty - py, tx - px)
        actor = math.atan2(actor_xy[1] - py, actor_xy[0] - px)
        delta = abs((actor - aim + math.pi) % (2.0 * math.pi) - math.pi)
        distance = math.dist((px, py), actor_xy)
        clearance = math.asin(min(1.0, 0.75 / distance))
        assert math.degrees(delta + clearance) <= (
            float(camera["field_of_view"]) / 2.0
        )

    for actor in ((7.0, -23.0), (5.0, -21.0), (5.0, -17.0)):
        assert_actor_fits(camera_112, actor)
    for actor in ((7.0, -23.0), (5.0, -21.0), (3.0, -13.0)):
        assert_actor_fits(camera_113, actor)


def test_xaria_teaser_restores_stock_lightmaps_without_replacing_authored_geometry() -> None:
    """The retail room keeps edited geometry but regains stock LM names and UV2."""

    from src.core.modules.authored_imported_mesh import (
        ImportedMeshRoomPrimitive,
        ImportedMeshSurface,
    )
    from src.core.modules.authored_module_objects import (
        AuthoredGameplayPlacement,
        ModuleEntryPoint,
    )
    from src.core.modules.authored_module_project import (
        AuthoredModuleMetadata,
        AuthoredModuleProject,
        AuthoredRoomSpec,
    )

    vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    faces = ((0, 1, 2),)
    target_surface = ImportedMeshSurface(
        name="stock_surface",
        texture="diffuse",
        vertices=vertices,
        faces=faces,
        texture_names=("diffuse", "stock_lm"),
        tex_count=2,
    )
    donor_surface = replace(
        target_surface,
        lightmap="stock_lm",
        uvs_lm=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    )
    target_primitive = ImportedMeshRoomPrimitive(
        room_resref="xarroom",
        source_model="stockroom",
        game="K2",
        surfaces=(target_surface,),
        metadata={"source_lightmaps_removed_for_relighting": True},
    )
    donor_primitive = replace(
        target_primitive,
        surfaces=(donor_surface,),
        metadata={},
    )
    project = AuthoredModuleProject(
        metadata=AuthoredModuleMetadata(module_root="xartease", game="K2"),
        rooms=(AuthoredRoomSpec(room_resref="xarroom", primitive=target_primitive),),
        placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(area_resref="xartease")),
    )

    restored = _restore_playable_room_lightmaps(
        project,
        "xarroom",
        resource_manager=None,
        source_primitive=donor_primitive,
    )
    restored_primitive = restored.rooms[0].primitive
    restored_surface = restored_primitive.surfaces[0]

    assert restored_surface.vertices is target_surface.vertices
    assert restored_surface.faces is target_surface.faces
    assert restored_surface.lightmap == "stock_lm"
    assert restored_surface.uvs_lm == donor_surface.uvs_lm
    assert restored_primitive.metadata["source_lightmaps_removed_for_relighting"] is False
    assert restored_primitive.metadata["source_lightmaps_restored_for_retail"] is True
    assert restored_primitive.metadata["source_lightmap_surface_count"] == 1


def test_xaria_teaser_preserves_three_distinct_combat_showcase_lanes() -> None:
    beats = {item["power"]: item for item in SHOWCASE_BEATS}

    assert tuple(beats) == (
        "Miststep: Ambush",
        "Ichor Lightning",
        "Ichor Drain",
    )
    assert beats["Miststep: Ambush"]["target_tag"] == "Xaria_Wraid_1"
    assert beats["Ichor Lightning"]["target_tag"] == "Xaria_Wraid_2"
    assert beats["Ichor Drain"]["target_tag"] == "Xaria_Wraid_3"
    assert beats["Miststep: Ambush"]["arrival_clearance_m"] >= 1.5
    assert len({item["camera_role"] for item in SHOWCASE_BEATS}) == 3
    assert all(item["unobstructed_sightline"] for item in SHOWCASE_BEATS)


def test_xaria_teaser_hands_cutscene_safe_combat_to_guarded_recruitment() -> None:
    all_sources = "\n".join(PRIVATE_SCRIPT_SOURCES.values())
    heartbeat_source = PRIVATE_SCRIPT_SOURCES["xt_hb"]
    click_source = PRIVATE_SCRIPT_SOURCES["xt_click"]
    death_source = PRIVATE_SCRIPT_SOURCES["xt_dead"]

    assert tuple(PRIVATE_SCRIPT_SOURCES) == (
        "xt_start",
        "xt_begin",
        "xt_click",
        "xt_b1",
        "xt_b2",
        "xt_b3",
        "xt_dead",
        "xt_c1done",
        "xt_c2done",
        "xt_c3done",
        "xt_camend",
        "xt_post",
        "xt_enddlg",
        "xt_cleanup",
        "xt_hb",
    )
    assert "GetEnteringObject" in PRIVATE_SCRIPT_SOURCES["xt_start"]
    assert 'ExecuteScript("xt_begin"' in PRIVATE_SCRIPT_SOURCES["xt_start"]
    assert PRIVATE_XARIA_TAG in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert PRIVATE_POWER_ROWS == (287, 290, 291)
    assert PRIVATE_POWER_TOKENS == (1, 2, 3)
    assert "ActionCastSpellAtObject" not in all_sources
    assert all_sources.index('"kxar_d_mamb"') < all_sources.index('"kxar_d_ilight"')
    assert all_sources.index('"kxar_d_ilight"') < all_sources.index('"kxar_d_idrain"')
    assert "ActionStartConversation" in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert 'oPC,\n            "xt_dlg",\n            FALSE,' in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert 'ExecuteScript("xt_b1", oXaria, 1)' not in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert (
        'DelayCommand(1.25, ExecuteScript("xt_b1", oXaria, 1))'
        in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    )
    assert (
        'DelayCommand(1.25, ExecuteScript("xt_b2", oXaria, 2))'
        in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    )
    assert (
        'DelayCommand(1.25, ExecuteScript("xt_b3", oXaria, 3))'
        in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    )
    assert (
        "ChangeToStandardFaction(oXaria, STANDARD_FACTION_FRIENDLY_1)"
        in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    )
    assert all(
        PRIVATE_SCRIPT_SOURCES[resref].count("STANDARD_FACTION_HOSTILE_1") == 1
        for resref in ("xt_b1", "xt_b2", "xt_b3")
    )
    assert "PrepareTarget(" not in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert "EffectDamage(1000" not in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert "CutsceneAttack(" not in all_sources
    assert "FinishCutsceneDamage" not in all_sources
    assert PRIVATE_DIRECTOR_TAG in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert "XT_Intro_Trigger" not in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert "XT_Intro_Trigger" in death_source
    assert (
        "DestroyObject(oTrigger, 0.0, TRUE, 0.0, TRUE)"
        not in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    )
    assert "GetLocalNumber(oXaria, 25) != 7" not in death_source
    assert (
        "DestroyObject(oTrigger, 0.0, TRUE, 0.0, TRUE)"
        in death_source
    )
    assert death_source.index("nProof == 7") < death_source.index(
        "DestroyObject(oTrigger, 0.0, TRUE, 0.0, TRUE)"
    )
    assert 'ExecuteScript("xt_begin"' in click_source
    assert "nState == 0" in click_source
    assert "nState == 1" in click_source
    assert "nState == 2" in click_source
    assert PRIVATE_SCHEMA == 14
    state_values = {
        int(value)
        for value in re.findall(
            rf"SetLocalNumber\([^,]+,\s*{PRIVATE_STATE_LOCAL},\s*(-?\d+)\)",
            all_sources,
        )
    }
    assert state_values == {0, 1, 2}
    assert "nState == 99" not in all_sources
    assert f"SetLocalNumber(oXaria, {PRIVATE_STATE_LOCAL}, 99)" not in all_sources
    assert "Retrying the Xaria encounter" not in all_sources
    for resref in ("xt_begin", "xt_click", "xt_hb"):
        assert (
            f"GetLocalNumber(oXaria, {PRIVATE_SCHEMA_LOCAL}) == {PRIVATE_SCHEMA}"
        ) in PRIVATE_SCRIPT_SOURCES[resref]
        assert (
            f"SetLocalNumber(oXaria, {PRIVATE_SCHEMA_LOCAL}, {PRIVATE_SCHEMA})"
        ) in PRIVATE_SCRIPT_SOURCES[resref]
    assert ENCOUNTER_PROXIMITY_METRES == 0.0
    assert "GetDistanceBetween" not in heartbeat_source
    assert 'ExecuteScript("xt_begin"' in heartbeat_source
    assert 'ExecuteScript("xt_b1"' not in heartbeat_source
    assert 'ExecuteScript("xt_b2"' not in heartbeat_source
    assert 'ExecuteScript("xt_b3"' not in heartbeat_source
    assert PRIVATE_WRAID_FACTION_ID == 5
    assert teaser_builder.PRIVATE_DIALOGUE_PENDING_LOCAL == 63
    assert "ActionStartConversation" in PRIVATE_SCRIPT_SOURCES["xt_post"]
    assert "ActionStartConversation" not in PRIVATE_SCRIPT_SOURCES["xt_click"]
    assert '"xaria"' in PRIVATE_SCRIPT_SOURCES["xt_post"]
    assert 'ExecuteScript("xt_post", oXaria, -1)' in PRIVATE_SCRIPT_SOURCES["xt_click"]
    assert 'SetGlobalNumber("KPM_XARIA_STATE", 2)' in PRIVATE_SCRIPT_SOURCES["xt_post"]
    assert 'nGlobalState == 3' in PRIVATE_SCRIPT_SOURCES["xt_post"]
    assert "ShowPartySelectionGUI(" in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert '"k_pend_reset", iSlot, 0xFFFFFFFF, FALSE' in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert 'ExecuteScript("kxar_cleanup"' in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert 'ExecuteScript("kxar_cleanup"' in PRIVATE_SCRIPT_SOURCES["xt_cleanup"]
    assert "GetRunScriptVar()" in PRIVATE_SCRIPT_SOURCES["xt_cleanup"]
    assert "DestroyObject(" not in PRIVATE_SCRIPT_SOURCES["xt_cleanup"]
    assert "DestroyObject(" not in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert "SetPlotFlag(" not in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert "DelayCommand(\n                0.05," in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    assert "ActionPauseConversation" not in all_sources
    assert "ActionResumeConversation" not in all_sources
    assert PRIVATE_ACTIVE_BEAT_LOCAL == 21
    assert (
        f"GetLocalNumber(OBJECT_SELF, {PRIVATE_ACTIVE_BEAT_LOCAL}) != 0"
        in heartbeat_source
    )
    assert (
        f"GetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}) != 0"
        in click_source
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 1)"
        in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 2)"
        in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 3)"
        in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 12)"
        in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 13)"
        in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    )
    assert (
        f"SetLocalNumber(oXaria, {PRIVATE_ACTIVE_BEAT_LOCAL}, 0)"
        in death_source
    )
    assert "ResolveMiststepStrike" not in all_sources
    assert 'DelayCommand(0.25, ExecuteScript("xt_dead"' in death_source
    assert "nWatchdog" not in all_sources
    for resref, target, token in (
        ("xt_b1", "oWraid1", 1),
        ("xt_b2", "oWraid2", 2),
        ("xt_b3", "oWraid3", 3),
    ):
        assert (
            f"SetLocalNumber({target}, {PRIVATE_DIRECTOR_TARGET_LOCAL}, {token})"
            in PRIVATE_SCRIPT_SOURCES[resref]
        )
    assert 'ExecuteScript("kxar_d_mamb", oXaria, 287)' in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    assert 'ExecuteScript("kxar_d_ilight", oXaria, 290)' in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    assert 'ExecuteScript("kxar_d_idrain", oXaria, 291)' in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    assert "DelayCommand(2.60, FinishFirstTarget" in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    assert "DelayCommand(2.80, FinishSecondTarget" in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    assert "DelayCommand(3.10, FinishThirdTarget" in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    assert "SetDialogPlaceableCamera(" not in death_source
    assert 'ExecuteScript("xt_b2"' not in death_source
    assert 'ExecuteScript("xt_b3"' not in death_source
    assert "location lOrigin = GetLocation(oXaria)" in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    assert "ApplyMistAtLocation(lOrigin)" in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    assert "DelayCommand(0.20, ApplyMistAtLocation(lArrival))" in PRIVATE_SCRIPT_SOURCES["xt_b1"]
    for resref in ("xt_b2", "xt_b3"):
        source = PRIVATE_SCRIPT_SOURCES[resref]
        assert "SetFacingPoint(GetPosition(" in source
        assert "EffectBeam(" not in source
        assert "BODY_NODE_HAND" not in source
        assert "VFX_BEAM_" not in source
        assert "EffectVisualEffect(9100, FALSE)" not in source
        assert "ActionCastFakeSpellAtObject(" in source
    assert "EffectVisualEffect(9101, FALSE)" not in PRIVATE_SCRIPT_SOURCES["xt_b2"]
    assert "EffectVisualEffect(9102, FALSE)" not in PRIVATE_SCRIPT_SOURCES["xt_b3"]
    assert {
        ("kxar_d_mamb", "ncs"),
        ("kxar_d_ilight", "ncs"),
        ("kxar_d_idrain", "ncs"),
    } <= {(item["resref"], item["type"]) for item in BUNDLED_ENCOUNTER_RESOURCES}
    for resref, wrapper, kill_call in (
        (
            "xt_b1",
            'ExecuteScript("kxar_d_mamb", oXaria, 287)',
            "FinishFirstTarget(oXaria, oWraid1)",
        ),
        (
            "xt_b2",
            'ExecuteScript("kxar_d_ilight", oXaria, 290)',
            "FinishSecondTarget(oXaria, oWraid2)",
        ),
        (
            "xt_b3",
            'ExecuteScript("kxar_d_idrain", oXaria, 291)',
            "FinishThirdTarget(oXaria, oWraid3)",
        ),
    ):
        source = PRIVATE_SCRIPT_SOURCES[resref]
        assert source.count("SetMinOneHP(oTarget, FALSE)") == 1
        assert source.count("EffectDamage(") == 1
        assert "1000, DAMAGE_TYPE_UNIVERSAL" in source
        assert source.count("ANIMATION_LOOPING_SPASM") == 1
        assert source.count("PlayAnimation(") == 1
        assert source.index(wrapper) < source.rindex(kill_call)
    assert "KillCurrentTarget" not in all_sources
    for resref, token in zip(("xt_b1", "xt_b2", "xt_b3"), PRIVATE_POWER_TOKENS):
        source = PRIVATE_SCRIPT_SOURCES[resref]
        fallback = source.split(
            f"if (GetLocalNumber(oXaria, 26) != {token}) {{",
            maxsplit=1,
        )[1].split("}", maxsplit=1)[0]
        assert "AbortSequence" not in fallback
        assert "return;" not in fallback
    assert "GetLocalNumber(oXaria, 26)" not in death_source
    assert "SetDialogPlaceableCamera(" not in all_sources
    alive_guard = death_source.index("if (TargetIsAlive(oTarget)) {")
    proof_commit = death_source.index("int nProof = GetLocalNumber(oXaria, 25);")
    assert alive_guard < proof_commit
    assert 'ExecuteScript("k_oei_endconv"' not in death_source
    assert (
        'DelayCommand(0.90, ExecuteScript("xt_post", oXaria, -1))'
        in death_source
    )
    camend_source = PRIVATE_SCRIPT_SOURCES["xt_camend"]
    assert "GetLocalNumber(oXaria, 25) != 7" in camend_source
    assert camend_source.count("TargetIsAlive(") == 4
    assert (
        f"SetLocalBoolean(oXaria, "
        f"{teaser_builder.PRIVATE_CAMERA_HANDOFF_LOCAL}, TRUE)"
        in camend_source
    )
    assert camend_source.count('ExecuteScript("xt_post", oXaria, -1)') == 1
    assert "Xaria encounter handoff rejected:" in PRIVATE_SCRIPT_SOURCES["xt_post"]
    assert "xt_retry" not in all_sources
    assert "xt_gate" not in all_sources
    assert "xt_finish" not in all_sources
    assert "KPM_XARIA_STATE" in all_sources
    assert "AddAvailableNPC" not in all_sources
    assert "RemoveAvailableNPC" not in all_sources
    assert 'ExecuteScript("k_oei_endconv"' in PRIVATE_SCRIPT_SOURCES["xt_enddlg"]


def test_recruitment_dialogue_recovers_when_camera_113_onend_is_missed() -> None:
    camera_handoff_local = teaser_builder.PRIVATE_CAMERA_HANDOFF_LOCAL
    death_source = PRIVATE_SCRIPT_SOURCES["xt_dead"]
    camend_source = PRIVATE_SCRIPT_SOURCES["xt_camend"]
    post_source = PRIVATE_SCRIPT_SOURCES["xt_post"]
    click_source = PRIVATE_SCRIPT_SOURCES["xt_click"]
    all_sources = "\n".join(PRIVATE_SCRIPT_SOURCES.values())

    assert camera_handoff_local == 60
    handoff_commit = (
        f"SetLocalBoolean(oXaria, {camera_handoff_local}, TRUE)"
    )
    begin_production = 'ExecuteScript("xt_post", oXaria, -1)'
    assert "SetDialogPlaceableCamera(114)" not in all_sources
    assert 'ExecuteScript("k_oei_endconv"' not in death_source
    assert 'ExecuteScript("k_oei_endconv"' not in camend_source
    assert handoff_commit in camend_source
    assert begin_production in camend_source
    assert camend_source.index(handoff_commit) < camend_source.index(
        begin_production
    )
    assert (
        'DelayCommand(0.90, ExecuteScript("xt_post", oXaria, -1))'
        in death_source
    )
    assert "GetLocalNumber(oXaria, 25) != 7" in camend_source
    assert "GetLocalNumber(oXaria, 21) != 0" in camend_source
    assert (
        f"GetLocalBoolean(oXaria, {camera_handoff_local}) != TRUE"
        not in post_source
    )
    assert 'GetObjectByTag("XT_Director", 0)' in post_source
    assert "GetIsInConversation(oDirector)" in post_source
    assert 'DelayCommand(0.50, ExecuteScript("xt_post", oXaria, -1))' in post_source
    assert handoff_commit in post_source
    assert (
        post_source.index("GetIsInConversation(oDirector)")
        < post_source.index(handoff_commit)
    )
    assert 'ExecuteScript("xt_post", oXaria, -1)' in click_source
    assert "nState == 2 && nGlobalState == 3" in click_source
    assert "nState == 2 && nGlobalState != 3" in click_source
    assert "ActionStartConversation" not in click_source
    assert all_sources.count('"xaria"') == 1
    assert post_source.count('"xaria"') == 1
    assert post_source.count(
        "SetLocalBoolean(oXaria, "
        f"{teaser_builder.PRIVATE_DIALOGUE_STARTED_LOCAL}, TRUE)"
    ) == 0
    assert "CancelCombat(oPC);" in post_source
    assert "CancelCombat(oXaria);" in post_source
    assert "SurrenderToEnemies()" in post_source
    assert 'ExecuteScript("xt_post", oXaria, -1)' in post_source
    assert (
        "PlayAnimation(\n"
        "            ANIMATION_LOOPING_PAUSE,\n"
        "            1.0,\n"
        "            1.0"
        in post_source
    )
    assert (
        f"SetLocalBoolean(oXaria, "
        f"{teaser_builder.PRIVATE_DIALOGUE_PENDING_LOCAL}, TRUE)"
        in post_source
    )
    assert 'ExecuteScript("xt_post", oXaria, 1)' in post_source
    assert "3.00" in post_source
    assert "if (nGlobalState != 4)" in post_source
    assert 'SetGlobalNumber("KPM_XARIA_STATE", 2)' in post_source
    assert (
        'GetGlobalNumber("KPM_XARIA_STATE") != 4'
        in PRIVATE_SCRIPT_SOURCES["xt_hb"]
    )
    assert "kxar_dlgack" in teaser_builder.PRODUCTION_SCRIPT_RESREFS


def test_teaser_carries_the_expanded_party_and_reaction_contract() -> None:
    """The test map must package every post-recruit and companion-reaction edge."""

    assert {
        "cxar_atton_m",
        "cxar_atton_f",
        "cxar_kreia",
        "cxar_react",
        "cxar_pfirst",
        "cxar_pgrove",
        "cxar_phawk",
        "cxar_pother",
        "cxar_d_ready",
        "cxar_d_busy",
        "kxar_d_queue",
        "kxar_talkup",
        "kxar_rel_good",
        "kxar_rel_soft",
    }.issubset(teaser_builder.PRODUCTION_SCRIPT_RESREFS)
    expanded_meeting_sources = (
        "xv_warn",
        "xv_hunger",
        "xv_threat",
        "xv_stories",
        "xv_strife",
        "xv_demand",
        "xv_soulfound",
        "xv_hero",
        "xv_seeker",
        "xv_soulmouth",
        "xv_property",
        "xv_askquick",
        "xv_nydak",
        "xv_magick",
        "xv_withhold",
    )
    assert len(PRODUCTION_VOICE_RESREFS) == 49
    assert tuple(
        PRODUCTION_VOICE_SOURCE_TO_RUNTIME[source]
        for source in expanded_meeting_sources
    ) == tuple(f"997xaria{index:03d}" for index in range(17, 32))
    assert set(PRODUCTION_VOICE_SHA256) == set(
        PRODUCTION_VOICE_SOURCE_TO_RUNTIME
    )
    assert set(PRODUCTION_LIP_SHA256) == set(
        PRODUCTION_VOICE_SOURCE_TO_RUNTIME
    )
    assert {
        "StreamVoice/997/xaria/997xaria017.wav",
        "StreamVoice/997/xaria/997xaria031.wav",
        "StreamVoice/997/xaria/997xaria115.wav",
        "StreamVoice/997/atton/997atton001.wav",
        "StreamVoice/997/atton/997atton002.wav",
        "StreamVoice/997/kreia/997kreia001.wav",
    }.issubset(
        {item["name"] for item in EXTERNAL_XARIA_STREAMVOICE_FILES}
    )
    assert TEASER_VOICE_LOOKUP["speaker_folders"] == (
        "atton",
        "kreia",
        "xaria",
    )


def test_wraid_death_clears_attached_ichor_beams_before_lethal_damage() -> None:
    """A duration beam must not survive its victim and strand across dialogue."""

    for resref in ("xt_b2", "xt_b3"):
        source = PRIVATE_SCRIPT_SOURCES[resref]
        cleanup = "ClearLingeringIchorEffects(oXaria, oTarget);"
        lethal = "1000, DAMAGE_TYPE_UNIVERSAL"
        assert "EFFECT_TYPE_BEAM" in source
        assert "EFFECT_TYPE_POISON" in source
        assert "GetEffectCreator(eEffect) == oXaria" in source
        assert cleanup in source
        assert source.index(cleanup) < source.index(lethal)


def test_recruited_state_always_retires_the_private_xaria_actor() -> None:
    """Cleanup cannot depend on a transient dialogue-local latch."""

    heartbeat = PRIVATE_SCRIPT_SOURCES["xt_hb"]
    end_dialogue = PRIVATE_SCRIPT_SOURCES["xt_enddlg"]
    click = PRIVATE_SCRIPT_SOURCES["xt_click"]
    post = PRIVATE_SCRIPT_SOURCES["xt_post"]

    recruited_branch = heartbeat.split(
        'GetGlobalNumber("KPM_XARIA_STATE") == 3',
        maxsplit=1,
    )[1]
    assert "GetLocalBoolean(OBJECT_SELF, 62)" not in recruited_branch
    assert 'ExecuteScript("kxar_cleanup", GetFirstPC(), 0);' in recruited_branch
    assert "GetLocalBoolean(OBJECT_SELF, 62)" not in end_dialogue
    assert 'ExecuteScript("kxar_cleanup", oPC, 0)' in end_dialogue
    assert 'ExecuteScript("kxar_cleanup", oPC, 0);' in click
    assert 'ExecuteScript("kxar_cleanup", oPC, 0);' in post


def test_xaria_teaser_uses_a_naturally_terminating_three_camera_owner() -> None:
    """Death proofs advance 111->112->113; the final proof ends naturally."""

    from pykotor.resource.generics.dlg import read_dlg

    with contextlib.redirect_stdout(io.StringIO()):
        dialogue = read_dlg(_private_dialogue_bytes())
    entries = list(dialogue.all_entries(as_sorted=True))
    replies = list(dialogue.all_replies(as_sorted=True))

    assert dialogue.skippable is False
    assert str(dialogue.on_end) == "xt_camend"
    assert str(dialogue.on_abort) == "xt_camend"
    assert len(dialogue.starters) == 1
    assert len(entries) == 3
    assert len(replies) == 6
    assert dialogue.starters[0].node is entries[0]
    assert [entry.camera_id for entry in entries] == [111, 112, 113]
    assert [entry.camera_angle for entry in entries] == [6, 6, 6]
    assert [str(entry.script1) for entry in entries] == [
        "xt_b1",
        "xt_b2",
        "xt_b3",
    ]
    assert [entry.delay for entry in entries] == [
        teaser_builder.PRIVATE_CAMERA_POLL_SECONDS
    ] * 3
    for stage, condition in enumerate(
        ("xt_c1done", "xt_c2done", "xt_c3done")
    ):
        advance, hold = entries[stage].links
        assert str(advance.active1) == condition
        assert advance.active1_not is False
        assert str(hold.active1) == condition
        assert hold.active1_not is True
        assert len(hold.node.links) == 1
        assert hold.node.links[0].node is entries[stage]
        assert hold.node.links[0].is_child is True
        if stage < 2:
            assert advance.node.links[0].node is entries[stage + 1]
            assert advance.node.links[0].is_child is False
        else:
            assert not advance.node.links
    assert all(entry.text.stringref == -1 for entry in entries)
    assert all(not entry.text.get(0, 0) for entry in entries)
    assert all(reply.text.stringref == -1 for reply in replies)
    assert all(not reply.text.get(0, 0) for reply in replies)
    assert all(entry.unskippable for entry in entries)
    assert all(reply.unskippable for reply in replies)
    assert all(not str(entry.vo_resref) for entry in entries)
    assert all(not str(reply.vo_resref) for reply in replies)
    assert all(int(entry.sound_exists) == 0 for entry in entries)
    assert all(int(reply.sound_exists) == 0 for reply in replies)
    assert all(entry.plot_index == -1 for entry in entries)
    assert all(reply.plot_index == -1 for reply in replies)
    assert all(entry.plot_xp_percentage == 0.0 for entry in entries)
    assert all(reply.plot_xp_percentage == 0.0 for reply in replies)
    assert teaser_builder.PRIVATE_SCHEMA == 14
    assert teaser_builder.PRIVATE_CAMERA_POLL_SECONDS == 1

    placements = _authored_gameplay_placements()
    assert {
        (item.template_resref, item.tag)
        for item in placements.creatures
    } == {
        ("xar_enc", "Xaria_Encounter"),
        ("xar_nydak1", "Xaria_Wraid_1"),
        ("xar_nydak2", "Xaria_Wraid_2"),
        ("xar_nydak3", "Xaria_Wraid_3"),
    }
    assert {
        (item.template_resref, item.tag)
        for item in placements.placeables
    } == {
        ("xar_director", "Xaria_Director"),
        ("xar_clock", "Xaria_Timeline"),
    }
    assert {
        (item.template_resref, item.tag)
        for item in placements.sounds
    } == {
        ("kxar_mistsnd", "kxar_mistsnd"),
        ("kxar_lightsnd", "kxar_lightsnd"),
        ("kxar_drainsnd", "kxar_drainsnd"),
    }
    builder_source = (ROOT / "scripts" / "build_xaria_teaser_map.py").read_text(encoding="utf-8")
    assert teaser_builder.PRODUCTION_DIALOGUE == "xaria"
    assert "xaria.conversation = ResRef(PRODUCTION_DIALOGUE)" in builder_source
    assert 'xaria.on_dialog = ResRef("xt_click")' in builder_source
    assert 'xaria.on_end_dialog = ResRef("xt_enddlg")' in builder_source
    wraid_source = builder_source.split(
        '("xar_nydak1.utc", "xar_nydak2.utc", "xar_nydak3.utc")',
        maxsplit=1,
    )[1].split(
        "# Clone the proven generic trigger",
        maxsplit=1,
    )[0]
    assert '"on_death",' in wraid_source
    assert "setattr(wraid, field_name, blank)" in wraid_source
    assert "wraid.on_death" not in wraid_source
    assert "ActionStartConversation" in PRIVATE_SCRIPT_SOURCES["xt_begin"]
    assert "ActionStartConversation" in PRIVATE_SCRIPT_SOURCES["xt_post"]


def test_private_scripts_use_retail_k2_action_argument_order() -> None:
    """Every independent entry path must survive K2's typed ACTION pops."""

    from pykotor.resource.formats.ncs import read_ncs

    resources = _compile_private_script_resources()
    compiled = {resref: payload for resref, restype, payload in resources if restype == "ncs"}
    assert tuple(compiled) == tuple(PRIVATE_SCRIPT_SOURCES)
    _require_private_script_native_action_abi(compiled)

    def action_ids(resref: str) -> tuple[int, ...]:
        instructions = read_ncs(compiled[resref]).instructions
        return tuple(
            int(instruction.args[0])
            for instruction in instructions
            if instruction.ins_type.name == "ACTION"
        )

    def action_tail(
        resref: str,
        routine_id: int,
        count: int,
        occurrence: int = 0,
    ) -> tuple[tuple[str, list[object]], ...]:
        instructions = read_ncs(compiled[resref]).instructions
        action_index = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.ins_type.name == "ACTION" and int(instruction.args[0]) == routine_id
        ][occurrence]
        return tuple(
            (instruction.ins_type.name, list(instruction.args))
            for instruction in instructions[action_index - count : action_index]
        )

    assert 205 not in action_ids("xt_b1")
    assert 206 not in action_ids("xt_dead")
    assert action_tail("xt_start", 8, 2) == (
        ("CONSTO", [0]),
        ("CONSTS", ["xt_begin"]),
    )
    assert action_tail("xt_hb", 681, 2, occurrence=1) == (
        ("CONSTI", [PRIVATE_STATE_LOCAL]),
        ("CONSTO", [0]),
    )
    assert action_tail("xt_click", 200, 2) == (
        ("CONSTI", [0]),
        ("CONSTS", [PRIVATE_XARIA_TAG]),
    )
    assert action_tail("xt_begin", 204, 2) == (
        ("CONSTS", ["xt_dlg"]),
        ("CPTOPSP", [-84, 4]),
    )
    assert action_tail("xt_begin", 204, 2, occurrence=1) == (
        ("CONSTS", ["xt_dlg"]),
        ("CPTOPSP", [-88, 4]),
    )
    assert action_tail("xt_b1", 8, 3, occurrence=1) == (
        ("CONSTI", [1]),
        ("CPTOPSP", [-12, 4]),
        ("CONSTS", ["xt_b1"]),
    )
    assert action_tail("xt_b1", 8, 3, occurrence=3) == (
        ("CONSTI", [287]),
        ("CPTOPSP", [-20, 4]),
        ("CONSTS", ["kxar_d_mamb"]),
    )
    assert action_tail("xt_b2", 8, 3) == (
        ("CONSTI", [290]),
        ("CPTOPSP", [-12, 4]),
        ("CONSTS", ["kxar_d_ilight"]),
    )
    assert action_tail("xt_b3", 8, 3) == (
        ("CONSTI", [291]),
        ("CPTOPSP", [-12, 4]),
        ("CONSTS", ["kxar_d_idrain"]),
    )
    assert 461 not in action_ids("xt_dead")
    assert action_tail("xt_camend", 8, 3) == (
        ("NEGI", []),
        ("CPTOPSP", [-20, 4]),
        ("CONSTS", ["xt_post"]),
    )
    assert action_tail("xt_post", 204, 2) == (
        ("CONSTS", ["xaria"]),
        ("CPTOPSP", [-88, 4]),
    )
    assert action_tail("xt_enddlg", 712, 4) == (
        ("CONSTI", [0]),
        ("CONSTI", [4294967295]),
        ("CPTOPSP", [-16, 4]),
        ("CONSTS", ["k_pend_reset"]),
    )


def test_xaria_teaser_trigger_geometry_is_local_to_each_git_instance() -> None:
    """Odyssey adds GIT trigger Position to every Geometry point at runtime."""

    placements = _authored_gameplay_placements()
    assert len(placements.triggers) == 2
    for trigger in placements.triggers:
        position = tuple(float(value) for value in trigger.position)
        local_vertices = tuple(
            tuple(float(value) for value in vertex) for vertex in trigger.geometry
        )
        assert local_vertices
        if trigger.template_resref == "xar_introtrig":
            assert position == pytest.approx(ENCOUNTER_TRIGGER_POSITION)
            assert len(local_vertices) == len(ENCOUNTER_TRIGGER_GEOMETRY)
            for actual, expected in zip(local_vertices, ENCOUNTER_TRIGGER_GEOMETRY):
                assert actual == pytest.approx(expected)
            assert max(abs(vertex[0]) for vertex in local_vertices) == 13.375
            assert max(abs(vertex[1]) for vertex in local_vertices) == 5.25
            assert max(abs(vertex[2]) for vertex in local_vertices) == pytest.approx(0.334007284)
        else:
            assert max(abs(vertex[0]) for vertex in local_vertices) <= 1.51
            assert max(abs(vertex[1]) for vertex in local_vertices) <= 0.56
            assert max(abs(vertex[2]) for vertex in local_vertices) == 0.0
        world_vertices = tuple(
            tuple(position[axis] + vertex[axis] for axis in range(3)) for vertex in local_vertices
        )
        world_center = tuple(
            sum(vertex[axis] for vertex in world_vertices) / len(world_vertices)
            for axis in range(3)
        )
        assert world_center[0] == pytest.approx(position[0], abs=0.01)
        assert world_center[1] == pytest.approx(position[1], abs=0.01)
        if trigger.template_resref == "xar_introtrig":
            mean_local_z = sum(vertex[2] for vertex in local_vertices) / len(local_vertices)
            assert abs(mean_local_z) < 0.02
            assert world_center[2] == pytest.approx(position[2] + mean_local_z, abs=1.0e-6)
        else:
            assert world_center[2] == pytest.approx(position[2], abs=0.01)


def test_xaria_teaser_external_dependency_contract_is_exact_and_nonstandalone() -> None:
    override_files = {
        item["resource"] for item in ENCOUNTER_DEPENDENCIES if item["kind"] == "override_file"
    }
    table_rows = {
        (item["resource"], item["row"])
        for item in ENCOUNTER_DEPENDENCIES
        if item["kind"] == "2da_row"
    }
    runtime_patches = {
        item["resource"] for item in ENCOUNTER_DEPENDENCIES if item["kind"] == "runtime_patch"
    }
    streamvoice_files = {
        item["resource"] for item in ENCOUNTER_DEPENDENCIES if item["kind"] == "streamvoice_file"
    }

    assert override_files == {item["name"] for item in EXTERNAL_XARIA_FILES}
    assert override_files >= {
        "p_xariabb.mdl",
        "p_xariabb.mdx",
        "p_xariab1.tga",
        "p_xariah6.mdl",
        "p_xariah6.mdx",
        "p_xaria06.tga",
        "p_xaria06.txi",
        "p_xaria.utc",
        "xaria.dlg",
        "xaria_blade.uti",
        "cxar_post.ncs",
        "cxar_wait.ncs",
        "cxar_party.ncs",
        "kxar_join.ncs",
        "kxar_cleanup.ncs",
        "kxar_wait.ncs",
        "kxar_spawn.ncs",
        "kxar_d_begin.ncs",
        "kxar_d_clean.ncs",
        "kxar_d_tick.ncs",
        "kxar_mstfx.mdl",
        "kxar_lhand.mdl",
        "kxar_dhand.mdl",
        "kxar_lght_dur.mdl",
        "kxar_dr_dur.mdl",
        "fx_xar_ichor.tga",
        "fx_xdrn1.tga",
        "fx_xdrain1.tga",
        "fx_xmist.tga",
        "fx_xmist1.tga",
        "kxar_s_idrain.wav",
        "kxar_s_ilight.wav",
        "kxar_s_mist.wav",
        "kxar_s_raise.wav",
        "kxar_s_veil.wav",
        "kxar_s_veild.wav",
    }
    assert {
        "p_xariah.mdl",
        "p_xariah.mdx",
        "p_xaria01.tga",
    }.isdisjoint(override_files)
    assert {
        f"cxar_l_{lesson}.ncs" for lesson in ("veil", "heal", "drain", "mist", "light", "raise")
    }.issubset(override_files)
    assert {
        f"kxar_d_{lesson}.ncs" for lesson in ("veil", "heal", "drain", "mist", "light", "raise")
    }.issubset(override_files)
    assert {
        f"kxar_l_{lesson}.ncs" for lesson in ("veil", "iheal", "idrain", "mist", "ilight", "raise")
    }.issubset(override_files)
    assert streamvoice_files == {item["name"] for item in EXTERNAL_XARIA_STREAMVOICE_FILES}
    voice_resrefs = PRODUCTION_VOICE_RESREFS
    assert streamvoice_files == {
        teaser_builder._streamvoice_relative_path(resref)
        for resref in voice_resrefs
    }
    assert TEASER_VOICE_LOOKUP == {
        "module_voice_id": VOICE_STREAM_ID,
        "module_folder": VOICE_STREAM_ID,
        "dialogue_resref": VOICE_DIALOGUE_RESREF,
        "speaker_folders": ("atton", "kreia", "xaria"),
        "source_files": tuple(
            teaser_builder._streamvoice_relative_path(resref)
            for resref in voice_resrefs
        ),
        "runtime_files": tuple(
            teaser_builder._streamvoice_relative_path(resref)
            for resref in voice_resrefs
        ),
    }
    assert table_rows == {
        ("appearance.2da", 725),
        ("heads.2da", 199),
        ("portraits.2da", 64),
        ("classes.2da", 17),
        *((("spells.2da", row) for row in range(286, 299))),
        *((("visualeffects.2da", row) for row in range(9100, 9104))),
    }
    assert runtime_patches == {
        "XariaPowerRuntime/action-862 hook",
        "CustomClassExtension",
        "PartySelectionExtensionK2",
    }


def test_spatial_design_plain_source_import_needs_no_pytest_path_injection() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import src.core.modules.map_studio_spatial_design as m; "
                "print(m.__file__); print(m.SPATIAL_DESIGN_VERSION)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(ROOT / "src" / "core" / "modules" / "map_studio_spatial_design.py") in result.stdout
    assert result.stdout.rstrip().endswith("1")


def test_spatial_design_source_and_native_payload_copies_are_manifested_exactly() -> None:
    expected = {
        "src/core/modules/__init__.py": (
            "6e30bd5f3a37d2eb5af54c57967ff322a497daa7aca4038132ae26cc48b15702"
        ),
        "src/core/modules/map_studio_spatial_design.py": (
            "fd3b08409c625a8849e2de73476f62901ef70709c23d3e01368085327d9c4423"
        ),
    }
    for project in ("GhostRigger.Core.Scene", "GhostRigger.Core.Tools"):
        manifest_path = ROOT / "native" / project / "GhostRiggerPythonPayload.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = {str(row["source_path"]): row for row in manifest["files"]}
        for source_path, expected_sha256 in expected.items():
            canonical = ROOT / source_path
            payload = ROOT / "native" / project / "Python" / source_path
            canonical_bytes = canonical.read_bytes()
            payload_bytes = payload.read_bytes()

            assert payload_bytes == canonical_bytes
            assert hashlib.sha256(canonical_bytes).hexdigest() == expected_sha256
            assert rows[source_path]["sha256"] == expected_sha256
