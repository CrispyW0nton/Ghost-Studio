from __future__ import annotations

import copy
import json
import shutil

import pytest

from tests.test_unreal_level_import import fixture_bundle, write_bundle
from src.core.modules.unreal_import_state import apply_overrides, new_import_state, plan_reimport, reconcile, revision
from src.core.modules.unreal_level_import import build_unreal_level_project
from src.io.unreal_project_assets import UnrealProjectAssets


def imported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = write_bundle(source, fixture_bundle())
    assets = UnrealProjectAssets(tmp_path / "project")
    ref, manifest, source_revision = assets.capture_bundle(path)
    state = new_import_state(ref, source_revision, manifest["source"]["map_package"])
    return assets, state, manifest, path


def test_project_assets_are_portable_and_do_not_embed_geometry(tmp_path):
    assets, state, manifest, source = imported(tmp_path)
    moved = tmp_path / "moved"
    shutil.copytree(assets.root, moved)
    relocated = UnrealProjectAssets(moved)
    bundle = relocated.resolve_bundle(relocated.read_json(state["manifest"]))
    assert build_unreal_level_project(bundle).project.rooms[0].primitive.surfaces[0].vertices[2] == (4, -4, 0)
    assert str(tmp_path) not in json.dumps(state)
    assert "vertices" not in json.dumps(state)
    assert source.is_file()


def test_reimport_preserves_local_spawn_and_transform_for_source_geometry_change(tmp_path):
    assets, state, previous, _ = imported(tmp_path)
    matrix = copy.deepcopy(previous["actors"][0]["world_matrix"])
    matrix[0][3] = 25
    state["component_overrides"]["floor/component"] = {"world_matrix": matrix}
    state["module_overrides"] = {"entry_position_cm": [125, 100, 0], "entry_yaw_degrees": 12}
    incoming = copy.deepcopy(previous)
    incoming["actors"][0]["mesh_file"]["sha256"] = "a" * 64
    plan = plan_reimport(state, previous, incoming, "revision-2")
    assert not plan.conflicts
    updated, _ = reconcile(state, plan, previous, incoming, {"path": "new.json", "sha256": "a" * 64})
    assert updated["module_overrides"] == state["module_overrides"]
    assert updated["component_overrides"] == state["component_overrides"]
    assert updated["source_revision"] == "revision-2"


def test_both_sides_transform_edits_require_explicit_choice_and_preview_revision(tmp_path):
    _, state, previous, _ = imported(tmp_path)
    matrix = copy.deepcopy(previous["actors"][0]["world_matrix"])
    matrix[0][3] = 25
    state["component_overrides"]["floor/component"] = {"world_matrix": matrix}
    incoming = copy.deepcopy(previous)
    incoming["actors"][0]["world_matrix"][0][3] = 50
    plan = plan_reimport(state, previous, incoming, "revision-2")
    with pytest.raises(ValueError, match="Resolve every"):
        reconcile(state, plan, previous, incoming, state["manifest"])
    choices = {c.key: "accept_source" for c in plan.conflicts}
    updated, _ = reconcile(state, plan, previous, incoming, state["manifest"], choices)
    assert "world_matrix" not in updated["component_overrides"]["floor/component"]
    state["module_overrides"]["name"] = "Intervening edit"
    with pytest.raises(ValueError, match="stale"):
        reconcile(state, plan, previous, incoming, state["manifest"], choices)


def test_source_deleted_locally_edited_component_can_be_kept_without_repeated_conflict(tmp_path):
    _, state, previous, _ = imported(tmp_path)
    state["component_overrides"]["floor/component"] = {"label": "My floor"}
    incoming = copy.deepcopy(previous)
    incoming["actors"] = []
    incoming["source"]["component_count"] = 0
    incoming["materials"] = []
    plan = plan_reimport(state, previous, incoming, "revision-2")
    updated, merged = reconcile(state, plan, previous, incoming, state["manifest"], {c.key: "keep_local" for c in plan.conflicts})
    assert merged["actors"] == previous["actors"]
    assert merged["materials"] == previous["materials"]
    next_plan = plan_reimport(updated, merged, incoming, "revision-3")
    assert not next_plan.conflicts
    _, again = reconcile(updated, next_plan, merged, incoming, state["manifest"])
    assert again["actors"] == previous["actors"]


def test_apply_transform_does_not_mutate_snapshot_and_spawn_stays_valid(tmp_path):
    assets, state, manifest, _ = imported(tmp_path)
    bundle = assets.resolve_bundle(manifest)
    before = copy.deepcopy(bundle)
    state["module_overrides"]["entry_yaw_degrees"] = 45
    result = build_unreal_level_project(apply_overrides(state, bundle))
    assert bundle == before
    assert result.project.placements.entry_point.facing == pytest.approx(-0.7853981633974483)


def test_capture_rejects_stale_source_and_modified_content_addressed_asset(tmp_path):
    assets, state, manifest, source = imported(tmp_path)
    with pytest.raises(ValueError, match="changed after preview"):
        assets.capture_bundle(source, expected_revision="0" * 64)
    mesh_ref = manifest["actors"][0]["mesh_file"]
    (assets.root / mesh_ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="Stale or altered"):
        assets.resolve_bundle(manifest)


def test_cancelled_preview_is_just_unreferenced_immutable_assets(tmp_path):
    assets, state, manifest, source = imported(tmp_path)
    before = revision(state)
    assets.capture_bundle(source)  # Idempotent caching is not applying to KMAP.
    assert revision(state) == before


def test_material_override_does_not_mutate_another_instance_of_shared_mesh(tmp_path):
    assets, state, manifest, _ = imported(tmp_path)
    bundle = assets.resolve_bundle(manifest)
    second = copy.deepcopy(bundle["actors"][0])
    second["id"] = "other"
    second["mesh"] = bundle["actors"][0]["mesh"]
    bundle["actors"].append(second)
    state["component_overrides"]["floor/component"] = {"material_overrides": {"stone": "custom"}}
    effective = apply_overrides(state, bundle)
    assert effective["actors"][0]["mesh"]["surfaces"][0]["material"] == "custom"
    assert effective["actors"][1]["mesh"]["surfaces"][0]["material"] == "stone"
