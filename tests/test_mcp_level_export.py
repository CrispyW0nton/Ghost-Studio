"""Saved-scene MCP export delegates to real IO and reports verifiable results."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from kotormcp.tools import level_export
from src.core.level.kmap_model import KMapProject
from src.core.level.kmap_serializer import KMapSerializer
from src.core.modules.authored_module_kmap_bridge import create_dev_test_authored_module_payload


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    from src.core.assets import resource_manager
    manager = SimpleNamespace(set_k1_dir=lambda *_: True, set_k2_dir=lambda *_: True,
                              load_texture_image=lambda *args, **kwargs: None)
    monkeypatch.setattr(resource_manager, "ResourceManager", lambda: manager)
    payload = create_dev_test_authored_module_payload()
    payload["placements"]["placeables"] = []
    source = tmp_path / "level.kmap"
    KMapSerializer.save(KMapProject(extra_sections={"authored_module": payload}), source)
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    return {"kmap_path": str(source), "game_dir": str(game_dir), "output_path": str(tmp_path / "level.obj")}


def call(arguments):
    raw = asyncio.run(level_export.handle_export_level(arguments))
    return raw, json.loads(raw["text"])


@pytest.mark.parametrize("extension", ["fbx", "obj"])
def test_saved_level_writes_real_geometry_and_digest(inputs, extension):
    from pathlib import Path
    inputs["output_path"] = str(Path(inputs["output_path"]).with_suffix("." + extension))
    source = Path(inputs["kmap_path"])
    before = source.read_bytes()
    raw, result = call(inputs)
    assert not raw["isError"], result
    assert result["ok"]
    output = Path(result["output_path"])
    assert result["bytes"] == output.stat().st_size > 0
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["geometry"]["rooms"] > 0
    assert result["desktop_control"] is False
    assert result["unsaved_edits_included"] is False
    assert source.read_bytes() == before


def test_existing_output_requires_explicit_overwrite(inputs):
    from pathlib import Path
    target = Path(inputs["output_path"])
    target.write_bytes(b"old")
    raw, result = call(inputs)
    assert raw["isError"] and not result["ok"]
    assert target.read_bytes() == b"old"
    raw, result = call({**inputs, "overwrite": True})
    assert not raw["isError"], result
    assert target.read_bytes() != b"old"


def test_existing_sidecar_blocks_entire_export(inputs):
    from pathlib import Path
    target = Path(inputs["output_path"])
    sidecar = target.with_suffix(".mtl")
    sidecar.write_bytes(b"old material")
    raw, result = call(inputs)
    assert raw["isError"] and not result["ok"]
    assert not target.exists()
    assert sidecar.read_bytes() == b"old material"


def test_no_exports_into_original_game_installation(inputs):
    from pathlib import Path
    target = Path(inputs["game_dir"]) / "level.obj"
    raw, result = call({**inputs, "output_path": str(target), "overwrite": True})
    assert raw["isError"] and not result["ok"]
    assert "outside" in result["message"]
    assert not target.exists()


@pytest.mark.parametrize("change", [{"output_path": "relative.obj"}, {"overwrite": "false"}, {"output_path": "C:/Temp/output.exe"}, {"unknown": True}])
def test_invalid_arguments_are_errors(inputs, change):
    raw, result = call({**inputs, **change})
    assert raw["isError"] and not result["ok"]


def test_catalog_advertises_saved_scope_and_write_annotations():
    from kotormcp.tools import get_all_tools
    tool = next(row for row in get_all_tools() if row["name"] == "ghoststudio_export_level")
    assert "SAVED" in tool["description"]
    assert tool["annotations"]["readOnlyHint"] is False
    assert tool["inputSchema"]["properties"]["overwrite"]["default"] is False


def test_texture_folder_redirect_into_game_is_rejected(inputs, monkeypatch):
    from pathlib import Path
    from PIL import Image
    from src.io.level_scene_export import _LevelTextureImages
    monkeypatch.setattr(_LevelTextureImages, "get", lambda *args: Image.new("RGBA", (2, 2)))
    target = Path(inputs["output_path"])
    game = Path(inputs["game_dir"])
    redirect = target.parent / (target.stem + "_obj_textures")
    try:
        redirect.symlink_to(game, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    raw, result = call({**inputs, "overwrite": True})
    assert raw["isError"] and not result["ok"], result
    assert "outside" in result["message"]
    assert not target.exists()
    assert list(game.iterdir()) == []


def test_concurrent_output_is_not_overwritten_and_prior_outputs_roll_back(tmp_path):
    from src.core.export.export_job import ExportJobRequest, ExportOutputSpec, run_export_job
    first, raced = tmp_path / "level.obj", tmp_path / "level.mtl"
    request = ExportJobRequest(job_id="race", kind="test", overwrite=False,
        outputs=[ExportOutputSpec(first, "mesh"), ExportOutputSpec(raced, "material")])
    def writer(context):
        context.write_bytes(first, b"new mesh")
        context.write_bytes(raced, b"new material")
        raced.write_bytes(b"concurrent writer")
    result = run_export_job(request, writer=writer)
    assert not result.succeeded
    assert raced.read_bytes() == b"concurrent writer"
    assert not first.exists()
