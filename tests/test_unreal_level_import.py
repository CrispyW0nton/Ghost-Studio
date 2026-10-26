from __future__ import annotations

import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.core.modules.unreal_level_import import build_unreal_level_project
from src.io.unreal_level_bundle import inspect_loose_umap, load_unreal_level_bundle, read_bundle_file


def fixture_bundle():
    # Exporter normalized source triangles to agree with their normals.
    surface = {"vertices": [[0, 0, 0], [400, 0, 0], [400, 400, 0], [0, 400, 0]],
               "triangles": [[0, 1, 2], [0, 2, 3]], "uvs": [[0, 0], [1, 0], [1, 1], [0, 1]],
               "normals": [[0, 0, 1]] * 4, "material": "stone"}
    texture = BytesIO()
    Image.new("RGB", (4, 4), (120, 130, 140)).save(texture, format="TGA")
    return {"schema": "ghoststudio.unreal-level/v1", "coordinates": "ue-centimeters-z-up-left-handed",
            "matrix_convention": "row-major-column-vector-local-to-world", "uv_origin": "bottom-left",
            "triangle_winding": "ccw-relative-to-normals",
            "source": {"engine_version": "5.8.2", "map_package": "/Game/Fixture", "scope": "loaded-static-level",
                       "unresolved_packages": [], "world_partition": False, "streaming_levels": [], "component_count": 1},
            "module": {"resref": "uefixture", "entry_position_cm": [100, 100, 0], "entry_yaw_degrees": 90},
            "materials": [{"id": "stone", "conversion": "baked-diffuse-v1", "blend_mode": "opaque", "diffuse_bytes": texture.getvalue()}],
            "actors": [{"id": "floor/component", "class": "StaticMeshComponent", "role": "static-room",
                        "asset_path": "/Game/Floor", "world_matrix": np.eye(4).tolist(),
                        "mesh": {"surfaces": [surface], "walkmesh": {"intent": "authored-walkmesh",
                                 "vertices": surface["vertices"], "triangles": surface["triangles"], "surface_ids": [4, 4]}}}]}


def write_bundle(tmp_path, bundle):
    bundle = copy.deepcopy(bundle)
    def put(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
    bundle["source"]["map_file"] = put("source.umap", b"synthetic native fixture; not a real map")
    for i, actor in enumerate(bundle["actors"]):
        if "mesh" in actor:
            actor["mesh_file"] = put(f"mesh{i}.json", json.dumps(actor.pop("mesh")).encode())
    for i, material in enumerate(bundle["materials"]):
        material["diffuse_file"] = put(f"texture{i}.tga", material.pop("diffuse_bytes"))
    manifest = tmp_path / "level.json"
    manifest.write_text(json.dumps(bundle))
    return manifest


def test_hash_checked_bundle_preserves_identity_and_converts_space(tmp_path):
    manifest = write_bundle(tmp_path, fixture_bundle())
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = build_unreal_level_project(load_unreal_level_bundle(manifest))
    room = result.project.rooms[0]
    assert room.primitive.surfaces[0].vertices[2] == (4, -4, 0)
    assert room.primitive.surfaces[0].faces[0] == (0, 2, 1)
    assert result.project.placements.entry_point.position == (1, -1, 0)
    assert result.project.placements.entry_point.facing == pytest.approx(-np.pi / 2)
    assert result.actor_resources["floor/component"] == room.room_resref
    assert result.project.extra["unreal_import"]["game_tested"] is False
    assert original == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_mirrored_nonuniform_transform_preserves_front_faces_and_normal():
    bundle = fixture_bundle()
    matrix = np.diag([-2., 3., 1., 1.]); matrix[:3, 3] = [100, 200, 50]
    bundle["actors"][0]["world_matrix"] = matrix.tolist()
    bundle["module"]["entry_position_cm"] = [-100,500,50]
    result = build_unreal_level_project(bundle)
    surface = result.project.rooms[0].primitive.surfaces[0]
    assert surface.vertices[2] == (-7, -14, .5)
    assert surface.faces[0] == (0, 1, 2)  # two reflections cancel
    assert surface.normals[0] == (0, 0, 1)


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["source"].update(unresolved_packages=["/Game/Missing"]), "Unresolved"),
    (lambda b: b["source"].update(world_partition=True), "World Partition"),
    (lambda b: b["source"].update(component_count=2), "accounted"),
    (lambda b: b["actors"][0].update(role="landscape"), "Unsupported"),
    (lambda b: b["actors"][0]["mesh"].pop("walkmesh"), "walkmesh"),
    (lambda b: b["actors"][0]["mesh"]["surfaces"][0].update(material="missing"), "Unresolved material"),
    (lambda b: b["actors"][0]["mesh"]["surfaces"][0].update(triangles=[[0, 1, 99]]), "index"),
    (lambda b: b["actors"][0]["mesh"]["surfaces"][0].update(triangles=[[0, 1, 1]]), "Degenerate"),
    (lambda b: b["actors"][0].update(world_matrix=np.zeros((4, 4)).tolist()), "affine"),
    (lambda b: b["materials"][0].update(blend_mode="translucent"), "opaque"),
    (lambda b: b["module"].update(resref="sixteen_chars_are_not_enough"), "resref"),
])
def test_incomplete_or_unsupported_source_is_rejected(mutation, match):
    bundle = fixture_bundle(); mutation(bundle)
    with pytest.raises(ValueError, match=match):
        build_unreal_level_project(bundle)


def test_dressing_shares_playable_sector_without_adding_collision():
    bundle = fixture_bundle()
    prop = copy.deepcopy(bundle["actors"][0]); prop.update(id="prop/component", role="static-dressing")
    prop["mesh"].pop("walkmesh")
    bundle["actors"].append(prop); bundle["source"]["component_count"] = 2
    before = copy.deepcopy(bundle)
    result = build_unreal_level_project(bundle)
    assert len(result.project.rooms) == 1
    room = result.project.rooms[0]
    assert len(room.primitive.wok.faces) == 2
    assert [s.name for s in room.primitive.surfaces] == ["c00_s0", "c01_s0"]
    assert result.actor_resources["prop/component"] == room.room_resref
    assert result.project.extra["unreal_import"]["component_nodes"]["prop/component"] == ["c01_s0"]
    assert room.metadata["source_components"][1]["source_component_id"] == "prop/component"
    assert bundle == before


def test_dressing_before_floor_preserves_sector_collision_and_provenance(tmp_path):
    from src.core.level import new_kmap_project
    from src.core.level.kmap_serializer import KMapSerializer
    from src.core.modules.authored_module_kmap_bridge import authored_project_to_kmap_payload, authored_project_from_kmap_payload
    bundle = fixture_bundle()
    prop = copy.deepcopy(bundle["actors"][0]); prop.update(id="prop", role="static-dressing")
    prop["mesh"].pop("walkmesh")
    prop["world_matrix"][0][3] = 1000
    prop["metadata"] = {"backdrop": True}
    bundle["actors"].insert(0, prop); bundle["source"]["component_count"] = 2
    result = build_unreal_level_project(bundle)
    room = result.project.rooms[0]
    assert room.metadata["source_component_id"] == "floor/component"
    assert room.primitive.surfaces[0].vertices[0] == (10, 0, 0)
    assert room.primitive.surfaces[0].backdrop is True
    assert room.primitive.wok.verts[0] == (0, 0, 0)
    kmap = new_kmap_project(name="uefixture", game="K2")
    kmap.extra_sections["authored_module"] = authored_project_to_kmap_payload(result.project, game_tested=False)
    path = tmp_path / "sector.kmap"
    KMapSerializer.save(kmap, path)
    restored = authored_project_from_kmap_payload(KMapSerializer.load(path).extra_sections["authored_module"], fallback_game="K2", fallback_name="uefixture")
    assert restored.rooms[0].metadata["source_components"] == room.metadata["source_components"]
    assert restored.rooms[0].primitive.surfaces == room.primitive.surfaces


def test_disconnected_floor_islands_are_rejected():
    bundle = fixture_bundle()
    mesh = bundle["actors"][0]["mesh"]["walkmesh"]
    mesh["vertices"] = mesh["vertices"] + [[x+1000,y,z] for x,y,z in mesh["vertices"]]
    mesh["triangles"] = mesh["triangles"] + [[4,5,6],[4,6,7]]
    mesh["surface_ids"] += [4,4]
    with pytest.raises(ValueError, match="edge-connected"):
        build_unreal_level_project(bundle)


@pytest.mark.parametrize("position", [[-100,100,0],[100,100,200]])
def test_spawn_off_floor_or_far_above_it_is_rejected(position):
    bundle=fixture_bundle();bundle["module"]["entry_position_cm"]=position
    with pytest.raises(ValueError, match="Entry must be on"):
        build_unreal_level_project(bundle)


def test_nonwalkable_kotor_surface_is_not_counted_as_playable():
    bundle=fixture_bundle();bundle["actors"][0]["mesh"]["walkmesh"]["surface_ids"]=[7,7]
    with pytest.raises(ValueError, match="supported surface IDs"):
        build_unreal_level_project(bundle)


def test_stale_geometry_rejected(tmp_path):
    manifest = write_bundle(tmp_path, fixture_bundle())
    (tmp_path / "mesh0.json").write_text("{}")
    with pytest.raises(ValueError, match="Stale"):
        load_unreal_level_bundle(manifest)


@pytest.mark.parametrize("path", ["../escape.json", "/escape.json", "C:/escape.json", "sub/../../escape.json", "sub\\escape.json"])
def test_paths_cannot_escape_bundle(tmp_path, path):
    with pytest.raises(ValueError):
        read_bundle_file(tmp_path, {"path": path, "sha256": "0" * 64})


def test_loose_map_hints_never_claim_conversion(tmp_path):
    path = tmp_path / "a.umap"
    path.write_bytes(b"\xc1\x83\x2a\x9e\0/Game/Kit/Mesh\0++UE5+Release-5.8\0")
    report = inspect_loose_umap(path)
    assert report["package_hints"] == ["/Game/Kit/Mesh"]
    assert not report["convertible"] and not report["native_dependencies_verified"]


def test_bundle_reaches_existing_k2_exporter(tmp_path):
    from src.core.modules.authored_module_export import AuthoredModuleExportRequest, export_authored_module_project
    from src.core.game.pykotor_gff_runtime_fix import ensure_pykotor_gff_acquire_quiet
    ensure_pykotor_gff_acquire_quiet()
    manifest = write_bundle(tmp_path, fixture_bundle())
    result = build_unreal_level_project(load_unreal_level_bundle(manifest))
    exported = export_authored_module_project(AuthoredModuleExportRequest(
        project=result.project, output_dir=str(tmp_path / "export"), extra_resources=result.extra_resources, strict=True))
    assert exported.ok, exported.blocking_issues
    from pykotor.resource.formats.erf import read_erf
    from pykotor.resource.type import ResourceType
    archive = read_erf(exported.module_path)
    room = result.project.rooms[0].room_resref
    assert archive.get(room, ResourceType.MDL)
    assert archive.get(room, ResourceType.MDX)
    assert archive.get(room, ResourceType.WOK)
    assert archive.get("uefixture", ResourceType.ARE)
    assert archive.get("uefixture", ResourceType.GIT)
    assert archive.get("module", ResourceType.IFO)
    # Resource presence alone cannot catch an empty MDX or lost transforms.
    from src.core.game.kotor_loader import load_model_from_bytes
    from src.core.validation.kotor_module_engine_contract import inspect_raw_mdl_structure, inspect_raw_wok_structure
    mdl, mdx = archive.get(room, ResourceType.MDL), archive.get(room, ResourceType.MDX)
    restored = load_model_from_bytes(mdl, mdx)
    surface = result.project.rooms[0].primitive.surfaces[0]
    mesh = next(n for n in restored.all_nodes() if n.name == surface.name)
    np.testing.assert_allclose(mesh.vertices, surface.vertices, atol=1e-6)
    np.testing.assert_allclose(mesh.normals, surface.normals, atol=1e-6)
    np.testing.assert_allclose(mesh.uvs, surface.uvs, atol=1e-6)
    np.testing.assert_array_equal(mesh.faces, surface.faces)
    _, model_check = inspect_raw_mdl_structure(room, mdl, mdx, game="K2")
    _, collision_check = inspect_raw_wok_structure(room, archive.get(room, ResourceType.WOK))
    assert not model_check.issues and not collision_check.issues


def test_clockwise_native_faces_cannot_be_mislabeled_normalized():
    bundle = fixture_bundle()
    bundle["actors"][0]["mesh"]["surfaces"][0]["triangles"] = [[0,2,1],[0,3,2]]
    with pytest.raises(ValueError, match="winding disagrees"):
        build_unreal_level_project(bundle)


def test_sheared_normal_uses_inverse_transpose():
    from src.math.unreal_level_space import transform_normals
    matrix = np.array([[2,0,1,0],[0,3,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    normal = transform_normals(np.array([[1.,0,0]]),matrix)[0]
    assert np.dot(normal, matrix[:3,2]) == pytest.approx(0)
    assert np.linalg.norm(normal) == pytest.approx(1)


def test_explicit_masked_material_packages_txi_and_second_side():
    bundle=fixture_bundle()
    bundle["materials"][0].update(blend_mode="masked",conversion_note="Native alpha cutoff baked.",two_sided=True)
    result=build_unreal_level_project(bundle)
    assert result.extra_resources[1][1:] == ("txi",b"blending punchthrough\n")
    surface=result.project.rooms[0].primitive.surfaces[0]
    assert len(surface.vertices)==8 and len(surface.faces)==4
    assert surface.normals[4]==(0,0,-1)


def test_duplicate_json_keys_rejected(tmp_path):
    manifest=tmp_path/'level.json';manifest.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError,match="Duplicate JSON key"):
        load_unreal_level_bundle(manifest)


def test_cli_requires_revision_and_preserves_existing_output(tmp_path,capsys):
    from scripts.import_unreal_level_bundle import main
    source=tmp_path/'source';source.mkdir();manifest=write_bundle(source,fixture_bundle())
    output=tmp_path/'build'
    assert main(['build',str(manifest),'--output',str(output),'--expected-sha256','0'*64])==2
    assert not output.exists()
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    args=['build',str(manifest),'--output',str(output),'--expected-sha256',digest]
    assert main(args)==0
    receipt=(output/'receipt.json').read_bytes()
    assert main(args)==2
    assert (output/'receipt.json').read_bytes()==receipt
    from src.core.level.kmap_serializer import KMapSerializer
    from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload
    reopened=KMapSerializer.load(output/'uefixture.kmap')
    authored=authored_project_from_kmap_payload(reopened.extra_sections['authored_module'],fallback_game='K2',fallback_name='uefixture')
    assert authored.rooms[0].primitive.surfaces[0].vertices[2]==(4,-4,0)
    assert authored.game=='K2'
