"""Level FBX must contain real scene geometry and publish complete outputs."""
from pathlib import Path
import json
import re

import pytest

from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
from src.core.level.kmap_model import KMapProject, RoomInstance, LevelTransform
from src.core.level.level_export_bridge import LevelExportBridge, LevelExportOptions


def triangle(name="triangle"):
    node = ModelNode(name=name, flags=int(NodeFlags.HEADER | NodeFlags.MESH))
    node.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    node.normals = [(0, 0, 1)] * 3
    node.faces = [(0, 1, 2)]
    node.uvs = [(0, 0), (1, 0), (0, 1)]
    return KotorModel(name=name, root_node=node)


def test_two_rooms_export_real_fbx_with_separate_meshes(tmp_path):
    project = KMapProject(rooms=[RoomInstance(model_resref="room_a"), RoomInstance(model_resref="room_b", transform=LevelTransform(position=(10, 0, 0)))])
    result = LevelExportBridge().export_fbx(project, tmp_path / "level.fbx", model_loader=lambda name, game: triangle(name))
    assert result.ok, result.message
    assert result.code == "exported"
    text = (tmp_path / "level.fbx").read_text()
    assert len(re.findall(r'Geometry:.*"Mesh"', text)) == 2
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["geometry"]["rooms"] == 2
    assert manifest["geometry"]["faces"] == 2


def test_missing_room_does_not_replace_existing_output(tmp_path):
    target = tmp_path / "level.fbx"
    target.write_bytes(b"keep existing export")
    project = KMapProject(rooms=[RoomInstance(model_resref="missing")])
    result = LevelExportBridge().export_fbx(project, target, model_loader=lambda *_: None)
    assert not result.ok
    assert "missing" in result.message.lower()
    assert target.read_bytes() == b"keep existing export"
    assert not target.with_name("level_manifest.json").exists()


def test_empty_scene_is_not_success(tmp_path):
    result = LevelExportBridge().export_fbx(KMapProject(), tmp_path / "empty.fbx")
    assert not result.ok
    assert not (tmp_path / "empty.fbx").exists()


def test_dry_run_writes_nothing(tmp_path):
    result = LevelExportBridge().export_fbx(KMapProject(), tmp_path / "dry.fbx", LevelExportOptions(dry_run=True))
    assert result.code == "dry_run"
    assert list(tmp_path.iterdir()) == []


def test_nested_transform_normals_reflection_and_source_unchanged():
    import copy
    import numpy as np
    from src.io.level_scene_export import assemble_level_model
    from src.core.level.kmap_model import ModuleInstance
    source = triangle()
    before = copy.deepcopy(source.root_node.vertices)
    room = RoomInstance(model_resref="room", transform=LevelTransform(position=(2, 0, 0), scale=(-2, 3, 1)))
    module = ModuleInstance(rooms=[room.room_id], transform=LevelTransform(position=(10, 0, 0), rotation=(0, 0, 90)))
    model, *_ = assemble_level_model(KMapProject(rooms=[room], modules=[module]), LevelExportOptions(), model_loader=lambda *_: source)
    node = model.mesh_nodes()[0]
    np.testing.assert_allclose(node.vertices, [(10, 2, 0), (10, 0, 0), (7, 2, 0)], atol=1e-6)
    np.testing.assert_allclose(node.normals, [(0, 0, 1)] * 3)
    assert node.faces == [(0, 2, 1)]
    assert source.root_node.vertices == before


def test_hidden_and_disabled_rooms_are_excluded():
    from src.io.level_scene_export import assemble_level_model
    rooms = [RoomInstance(model_resref="shown"), RoomInstance(model_resref="hidden", visible=False), RoomInstance(model_resref="disabled", enabled=False)]
    loaded = []
    def loader(name, game):
        loaded.append(name)
        return triangle(name)
    _, stats, *_ = assemble_level_model(KMapProject(rooms=rooms), LevelExportOptions(), model_loader=loader)
    assert loaded == ["shown"]
    assert stats["rooms"] == 1


def test_authored_rooms_are_compiled_without_duplicate_lyt_geometry(tmp_path):
    from src.core.modules.authored_module_kmap_bridge import create_dev_test_authored_module_payload
    payload = create_dev_test_authored_module_payload()
    payload["placements"]["placeables"] = []
    from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload
    authored = authored_project_from_kmap_payload(payload)
    project = KMapProject(rooms=[RoomInstance(model_resref=room.normalised_resref()) for room in authored.rooms], extra_sections={"authored_module": payload})
    result = LevelExportBridge().export_fbx(project, tmp_path / "authored.fbx")
    assert result.ok, result.message
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["geometry"]["rooms"] == len(authored.rooms) > 0


def test_whole_level_obj_has_both_rooms_and_material_file(tmp_path):
    project = KMapProject(rooms=[RoomInstance(model_resref="first"), RoomInstance(model_resref="second")])
    result = LevelExportBridge().export_obj(project, tmp_path / "whole.obj", model_loader=lambda name, game: triangle(name))
    assert result.ok, result.message
    text = (tmp_path / "whole.obj").read_text(encoding="utf-8")
    assert sum(line.startswith("o ") for line in text.splitlines()) == 2
    assert sum(line.startswith("f ") for line in text.splitlines()) == 2
    assert (tmp_path / "whole.mtl").is_file()


def test_fbx_obj_textures_and_manifests_coexist(tmp_path):
    from PIL import Image
    image = Image.new("RGB", (1, 2))
    image.putdata([(255, 0, 0), (0, 0, 255)])
    source = triangle()
    source.root_node.texture = "panel"
    source.root_node.texture_names = ["panel"]
    source.root_node.lightmap = "light"
    source.root_node.has_lightmap = True
    source.root_node.uvs_lm = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]
    project = KMapProject(rooms=[RoomInstance(model_resref="room")])
    context = dict(model_loader=lambda *_: source, tex_cache={"panel": image, "light": image})
    fbx = LevelExportBridge().export_fbx(project, tmp_path / "level.fbx", **context)
    assert fbx.ok, fbx.message
    obj = LevelExportBridge().export_obj(project, tmp_path / "level.obj", **context)
    assert obj.ok, obj.message
    assert fbx.manifest_path != obj.manifest_path
    with Image.open(tmp_path / "level_fbx_textures/panel.png") as flipped:
        assert flipped.getpixel((0, 0))[:3] == (0, 0, 255)
    with Image.open(tmp_path / "level_obj_textures/panel.png") as upright:
        assert upright.getpixel((0, 0))[:3] == (0, 0, 255)
    assert "vt 0.000000 1.000000" in (tmp_path / "level.obj").read_text()
    assert "level_obj_textures/panel.png" in (tmp_path / "level.mtl").read_text()
    fbx_text = (tmp_path / "level.fbx").read_text()
    assert "LayerElementUV: 1" in fbx_text
    manifest = json.loads(Path(fbx.manifest_path).read_text())
    assert manifest["texture_files"]["light"] == "level_fbx_textures/light.png"
    assert (tmp_path / manifest["texture_files"]["light"]).is_file()
    assert source.root_node.uvs_lm == [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)]


def test_writer_failure_keeps_previous_sidecars(tmp_path, monkeypatch):
    from src.converters.mesh_converter import FBXExporter
    monkeypatch.setattr(FBXExporter, "_export_fbx_ascii", lambda *args, **kwargs: False)
    target = tmp_path / "level.fbx"
    target.write_bytes(b"previous fbx")
    manifest = tmp_path / "level_manifest.json"
    manifest.write_bytes(b"previous manifest")
    result = LevelExportBridge().export_fbx(KMapProject(rooms=[RoomInstance(model_resref="room")]), target, model_loader=lambda *_: triangle())
    assert not result.ok
    assert target.read_bytes() == b"previous fbx"
    assert manifest.read_bytes() == b"previous manifest"


def test_collision_only_mesh_is_not_exported(tmp_path):
    source = triangle()
    source.root_node.flags |= int(NodeFlags.AABB)
    result = LevelExportBridge().export_fbx(KMapProject(rooms=[RoomInstance(model_resref="collision")]), tmp_path / "level.fbx", model_loader=lambda *_: source)
    assert not result.ok
    assert not (tmp_path / "level.fbx").exists()


def test_obj_overlapping_texture_names_remain_exact(tmp_path):
    from PIL import Image
    def load(name, _game):
        source = triangle(name)
        source.root_node.texture = name
        return source
    project = KMapProject(rooms=[RoomInstance(model_resref="wall"), RoomInstance(model_resref="sidewall")])
    cache = {name: Image.new("RGB", (2, 2)) for name in ("wall", "sidewall")}
    result = LevelExportBridge().export_obj(project, tmp_path / "level.obj", model_loader=load, tex_cache=cache)
    assert result.ok, result.message
    lines = (tmp_path / "level.mtl").read_text().splitlines()
    assert "map_Kd level_obj_textures/wall.png" in lines
    assert "map_Kd level_obj_textures/sidewall.png" in lines


@pytest.mark.parametrize("kind", ["placeable", "creature", "door"])
@pytest.mark.parametrize("separate_uvs", [False, True])
def test_authored_placed_model_is_positioned_once(kind, separate_uvs, monkeypatch):
    import math
    import numpy as np
    from types import SimpleNamespace
    from src.core.modules import map_studio_stock_content_preview as stock
    from src.core.modules.authored_module_kmap_bridge import create_dev_test_authored_module_payload
    from src.io.level_scene_export import assemble_level_model
    payload = create_dev_test_authored_module_payload()
    payload["placements"]["placeables"] = []
    payload["placements"][kind + "s"] = [{"instance_id": "placed_visual", "template_resref": "visual",
        "position": [10, 20, 3], "bearing": math.pi / 2}]
    resolver = SimpleNamespace(model_resource_bytes=lambda *_: None,
        model_for_placement_kind=lambda *_: "visual")
    monkeypatch.setattr(stock, "TemplateModelResolver", lambda *args, **kwargs: resolver)
    source = triangle("visual")
    if separate_uvs:
        source.root_node.uvs = [(0, 0), (1, 0), (0, 1), (0.25, 0.75)]
        source.root_node.face_uvs = [(3, 1, 2)]
    model, counts, nodes, _ = assemble_level_model(KMapProject(extra_sections={"authored_module": payload}),
        LevelExportOptions(), resource_manager=object(), model_loader=lambda *_: source)
    assert counts["placements"] == 1
    placed = [row for row in nodes if "placed_visual" in row["source_id"]]
    assert len(placed) == 1
    mesh = next(node for node in model.mesh_nodes() if node.name == placed[0]["mesh_node"])
    np.testing.assert_allclose(mesh.vertices, [(10, 20, 3), (10, 21, 3), (9, 20, 3)], atol=1e-6)
    assert source.root_node.vertices == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    assert mesh.uvs == source.root_node.uvs
    assert mesh.face_uvs == source.root_node.face_uvs
