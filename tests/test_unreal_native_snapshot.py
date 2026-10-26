from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest
from src.io.unreal_native_snapshot import convert_native_snapshot, load_native_snapshot, _diffuse
from src.io.unreal_level_bundle import load_unreal_level_bundle
from tests.test_unreal_level_import import fixture_bundle


def native_fixture(tmp_path):
    source, policy_dir = tmp_path / "native", tmp_path / "policy"
    source.mkdir(); policy_dir.mkdir()
    def put(name, value):
        content = value if isinstance(value, bytes) else json.dumps(value).encode()
        (source / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
    surface = copy.deepcopy(fixture_bundle()["actors"][0]["mesh"]["surfaces"][0])
    surface["triangles"] = [[a,c,b] for a,b,c in surface["triangles"]]
    surface["material_slot"] = 0
    material = {"base_color": {"class": "MaterialExpressionConstant3Vector", "constant": [.25,.5,.75,1]},
                "base_color_channel": "", "blend_mode": "BLEND_OPAQUE", "two_sided": False}
    snapshot = {"schema": "ghoststudio.unreal-snapshot/v1", "exporter_version": "0.1.0",
        "engine_version": "5.8.2-56702186+++UE5+Release-5.8", "dirty_packages": [], "source_preserved": True,
        "source_map": put("source.umap", b"unit fixture, not native proof"),
        "meshes": {"/Game/Floor.Floor": put("mesh.json", {"sections": [surface]})},
        "textures": {}, "materials": {"stone": material}, "map_package": "/Game/Fixture", "dependencies": [],
        "diagnostics": [], "components": [{"id":"guid/component", "disposition":"static_geometry", "mesh":"/Game/Floor.Floor",
            "materials":["stone"], "world_matrix":fixture_bundle()["actors"][0]["world_matrix"],
            "label":"Floor", "actor_guid":"guid", "actor_path":"/Game/Fixture:Floor"}]}
    path = source / "snapshot.json"
    path.write_text(json.dumps(snapshot))
    policy = {"schema":"ghoststudio.unreal-conversion-policy/v1", "accept_diffuse_only":True,
        "snapshot_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "module":fixture_bundle()["module"],
        "collision":{"component_id":"guid/component", "recipe":"explicit-render-section-v1", "section":0,"surface_id":4}}
    policy_path = policy_dir / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return path, policy_path, snapshot, policy


def test_native_recipe_converts_winding_uvs_and_publishes_valid_bundle(tmp_path):
    source, policy, _, _ = native_fixture(tmp_path)
    path = convert_native_snapshot(source, policy, tmp_path / "converted")
    result = load_unreal_level_bundle(path)
    mesh = result["actors"][0]["mesh"]
    assert mesh["surfaces"][0]["triangles"][0] == [0,1,2]
    assert mesh["surfaces"][0]["uvs"][0] == [0,1]
    assert mesh["walkmesh"]["surface_ids"] == [4,4]
    assert result["source"]["native_accounting"] == [{"id":"guid/component","disposition":"converted"}]
    with pytest.raises(FileExistsError):
        convert_native_snapshot(source, policy, path.parent)


@pytest.mark.parametrize("field,value,match", [("dirty_packages", ["/Game/Dirty"], "saved"),
    ("engine_version", "5.9", "Unqualified"), ("exporter_version", "2", "version")])
def test_native_snapshot_rejects_unqualified_source(tmp_path, field, value, match):
    path, _, snapshot, _ = native_fixture(tmp_path)
    snapshot[field] = value
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match=match):
        load_native_snapshot(path)


def test_material_channel_is_not_silently_reinterpreted(tmp_path):
    path, _, _, _ = native_fixture(tmp_path)
    snapshot = load_native_snapshot(path)
    snapshot["materials"]["stone"]["base_color_channel"] = "R"
    with pytest.raises(ValueError, match="channel"):
        _diffuse(snapshot, "stone", {}, tmp_path)


def test_conversion_policy_edit_during_validation_never_publishes(tmp_path, monkeypatch):
    path, policy_path, _, _ = native_fixture(tmp_path)
    import src.core.modules.unreal_level_import as importer
    original = importer.build_unreal_level_project
    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        policy_path.write_text(policy_path.read_text() + " ")
        return result
    monkeypatch.setattr(importer, "build_unreal_level_project", altered)
    with pytest.raises(ValueError, match="changed during"):
        convert_native_snapshot(path, policy_path, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_unsupported_primitive_requires_explicit_exclusion(tmp_path):
    path, policy_path, snapshot, policy = native_fixture(tmp_path)
    snapshot["components"].append({"id":"guid/niagara", "disposition":"unsupported", "label":"Fire"})
    snapshot["diagnostics"] = [{"subject":"guid/niagara","severity":"unsupported","message":"Niagara unsupported", "repair":"Author target VFX."}]
    path.write_text(json.dumps(snapshot))
    policy["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Niagara"):
        convert_native_snapshot(path, policy_path, tmp_path / "converted")
    policy["excluded_components"] = {"guid/niagara":"Creator chose to omit fire for this static export."}
    policy_path.write_text(json.dumps(policy))
    assert convert_native_snapshot(path, policy_path, tmp_path / "converted").is_file()


def test_hidden_geometry_requires_an_explicit_inclusion_choice(tmp_path):
    path, policy_path, snapshot, policy = native_fixture(tmp_path)
    snapshot["components"][0]["visible"] = False
    path.write_text(json.dumps(snapshot))
    policy["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="hidden component"):
        convert_native_snapshot(path, policy_path, tmp_path / "converted")
    policy["include_hidden_components"] = ["guid/component"]
    policy_path.write_text(json.dumps(policy))
    assert convert_native_snapshot(path, policy_path, tmp_path / "converted").is_file()
