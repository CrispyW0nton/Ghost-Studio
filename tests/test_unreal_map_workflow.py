from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import pytest

from src.core.level import new_kmap_project
from src.core.level.kmap_serializer import KMapSerializer
from src.core.modules.unreal_level_import import build_unreal_level_project
from src.core.modules.unreal_sector_projection import project_unreal_sector
from src.core.workflow.unreal_map_import import preview_unreal_import, apply_unreal_import, unreal_runtime_resources
from tests.test_unreal_level_import import fixture_bundle, write_bundle


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    bundle = fixture_bundle()
    prop = copy.deepcopy(bundle["actors"][0])
    prop.update(id="prop", role="static-dressing")
    prop["mesh"].pop("walkmesh")
    bundle["actors"].append(prop)
    bundle["source"]["component_count"] = 2
    path = write_bundle(root, bundle)
    project = new_kmap_project(name="empty", game="K2")
    project.path = str(tmp_path / "project" / "fixture.kmap")
    return project, path, bundle


def test_preview_apply_independent_components_and_portable_compact_save(tmp_path):
    project, path, _ = source(tmp_path)
    before = copy.deepcopy(KMapSerializer.to_dict(project))
    preview = preview_unreal_import(project, path)
    assert KMapSerializer.to_dict(project) == before
    result = apply_unreal_import(project, preview)
    assert len(result.extra_sections["authored_module"]["rooms"]) == 2
    KMapSerializer.save(result)
    document = Path(result.path).read_text()
    assert '"vertices"' not in document and str(tmp_path) not in document
    moved = tmp_path / "moved"
    shutil.copytree(Path(result.path).parent, moved)
    loaded = KMapSerializer.load(moved / "fixture.kmap")
    assert len(loaded.extra_sections["authored_module"]["rooms"]) == 2
    assert unreal_runtime_resources(loaded) == unreal_runtime_resources(result)


def test_reimport_keeps_ghost_position_spawn_and_accepts_source_mesh(tmp_path):
    project, path, bundle = source(tmp_path)
    project = apply_unreal_import(project, preview_unreal_import(project, path))
    authored = project.extra_sections["authored_module"]
    authored["rooms"][1]["position"][0] = 3
    authored["placements"]["entry_point"]["facing"] = 0.25
    bundle["actors"][1]["mesh"]["surfaces"][0]["vertices"][1][0] = 450
    write_bundle(path.parent, bundle)
    preview = preview_unreal_import(project, path)
    assert not preview.conflicts
    updated = apply_unreal_import(project, preview)
    assert updated.extra_sections["authored_module"]["rooms"][1]["position"][0] == 3
    assert updated.extra_sections["authored_module"]["placements"]["entry_point"]["facing"] == 0.25


def test_source_and_ghost_mesh_conflict_and_stale_apply(tmp_path):
    project, path, bundle = source(tmp_path)
    project = apply_unreal_import(project, preview_unreal_import(project, path))
    import base64, struct
    surface = project.extra_sections["authored_module"]["rooms"][1]["primitive"]["surfaces"][0]
    raw = bytearray(base64.b64decode(surface["vertices_b64"]))
    struct.pack_into("<f", raw, 12, 5)
    surface["vertices_b64"] = base64.b64encode(raw).decode()
    local_vertices = surface["vertices_b64"]
    bundle["actors"][1]["mesh"]["surfaces"][0]["vertices"][1][0] = 450
    write_bundle(path.parent, bundle)
    preview = preview_unreal_import(project, path)
    assert preview.conflicts
    with pytest.raises(ValueError, match="Resolve every"):
        apply_unreal_import(project, preview)
    updated = apply_unreal_import(project, preview, {key: "keep_local" for key in preview.conflicts})
    assert updated.extra_sections["authored_module"]["rooms"][1]["primitive"]["surfaces"][0]["vertices_b64"] == local_vertices
    project.name = "Intervening edit"
    with pytest.raises(ValueError, match="Project changed"):
        apply_unreal_import(project, preview)


def test_reimport_rejects_changed_manifest_and_altered_sidecar(tmp_path):
    project, path, bundle = source(tmp_path)
    preview = preview_unreal_import(project, path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed after preview"):
        apply_unreal_import(project, preview)
    write_bundle(path.parent, bundle)
    preview = preview_unreal_import(project, path)
    (path.parent / "mesh1.json").write_text("{}")
    with pytest.raises(ValueError, match="Stale or altered"):
        apply_unreal_import(project, preview)


def test_failed_kmap_promotion_preserves_previous_save_and_dirty_state(tmp_path, monkeypatch):
    project, path, _ = source(tmp_path)
    project = apply_unreal_import(project, preview_unreal_import(project, path))
    KMapSerializer.save(project)
    original = Path(project.path).read_bytes()
    before_timestamp = project.modified_at
    project.name = "Unsaved"
    project.dirty = True
    def fail(*args):
        raise PermissionError("locked destination")
    monkeypatch.setattr("src.io.unreal_kmap_storage.os.replace", fail)
    with pytest.raises(PermissionError):
        KMapSerializer.save(project)
    assert Path(project.path).read_bytes() == original
    assert project.dirty and project.modified_at == before_timestamp


def test_sector_projection_preserves_geometry_translates_edits_and_is_pure():
    bundle = fixture_bundle()
    authoring = build_unreal_level_project(bundle, independent_components=True).project
    expected = build_unreal_level_project(bundle).project
    assert project_unreal_sector(authoring) == expected
    edited = replace(authoring, rooms=(replace(authoring.rooms[0], position=(2, 3, 4)),))
    result = project_unreal_sector(edited)
    assert result.rooms[0].position == (0, 0, 0)
    assert result.rooms[0].primitive.surfaces[0].vertices[0] == (2, 3, 4)
    assert result.rooms[0].primitive.wok.verts[0] == (2, 3, 4)
    assert authoring.rooms[0].primitive.wok.verts[0] == (0, 0, 0)


def test_controller_import_is_one_undoable_command_and_restore_survives_save(tmp_path):
    from src.core.modules.module_editor_controller import ModuleEditorController
    project, path, _ = source(tmp_path)
    controller = ModuleEditorController()
    controller.model.set_project(project)
    preview = controller.prepare_unreal_import(path)
    controller.apply_unreal_import(preview)
    assert len(controller.project.extra_sections["authored_module"]["rooms"]) == 2
    assert controller.authored_project_extra_resources()
    assert controller.undo_map_studio_command() is not None
    assert "unreal_source" not in controller.project.extra_sections
    assert controller.redo_map_studio_command() is not None
    controller.save_project()
    controller.open_project(project.path)
    assert len(controller.project.extra_sections["authored_module"]["rooms"]) == 2
    assert controller.authored_project_extra_resources()


def test_deleted_source_with_local_edits_requires_choice_and_stays_local_next_time(tmp_path):
    project, path, bundle = source(tmp_path)
    project = apply_unreal_import(project, preview_unreal_import(project, path))
    project.extra_sections["authored_module"]["rooms"][1]["position"][0] = 3
    bundle["actors"].pop()
    bundle["source"]["component_count"] = 1
    write_bundle(path.parent, bundle)
    preview = preview_unreal_import(project, path)
    assert preview.conflicts
    project = apply_unreal_import(project, preview, {key: "keep_local" for key in preview.conflicts})
    assert len(project.extra_sections["authored_module"]["rooms"]) == 2
    again = preview_unreal_import(project, path)
    assert not again.conflicts
    assert len(apply_unreal_import(project, again).extra_sections["authored_module"]["rooms"]) == 2


def test_local_texture_paint_survives_and_source_texture_conflict_is_explicit(tmp_path):
    from io import BytesIO
    from PIL import Image
    from src.core.level import TextureReference
    project, path, bundle = source(tmp_path)
    project = apply_unreal_import(project, preview_unreal_import(project, path))
    name = project.extra_sections["unreal_source"]["resources"][0]["resref"]
    painted = Path(project.path).parent / "paint.tga"
    Image.new("RGB", (4, 4), (230, 10, 20)).save(painted)
    project.textures.append(TextureReference(resref=name, path="paint.tga"))
    assert not unreal_runtime_resources(project)  # Local authored texture is authoritative.
    output = BytesIO()
    Image.new("RGB", (4, 4), (20, 100, 210)).save(output, format="TGA")
    bundle["materials"][0]["diffuse_bytes"] = output.getvalue()
    write_bundle(path.parent, bundle)
    preview = preview_unreal_import(project, path)
    key = json.dumps(["texture", name], separators=(",", ":"))
    assert key in preview.conflicts
    kept = apply_unreal_import(project, preview, {key: "keep_local"})
    assert kept.textures[0].path == "paint.tga" and not unreal_runtime_resources(kept)
    accepted = apply_unreal_import(project, preview, {key: "accept_source"})
    assert not accepted.textures
    assert unreal_runtime_resources(accepted)[0][2] == output.getvalue()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(painted)
    with pytest.raises(ValueError, match="Project changed"):
        apply_unreal_import(project, preview, {key: "keep_local"})
