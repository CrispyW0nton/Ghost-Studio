"""Bounded-contract checks for the export tools Kaigen wraps (2026-09-29).

Fast, offline: no GPU render, no Blender launch, no game installation.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _call(name, arguments):
    from src.kotormcp.tools import handle_tool

    raw = asyncio.run(handle_tool(name, arguments))
    return raw, json.loads(raw["text"])


def test_build_retarget_map_bounds_manual_mapping():
    raw, data = _call(
        "ghostrigger_build_retarget_map",
        {
            "source_type": "unreal_quinn",
            "target_type": "unreal_quinn",
            "manual_mapping": {f"bone_{i}": "root" for i in range(513)},
        },
    )
    assert raw.get("isError") is True
    assert "512" in data["error"]
    raw, data = _call(
        "ghostrigger_build_retarget_map",
        {"source_type": "unreal_quinn", "target_type": "unreal_quinn", "manual_mapping": {"a": "x" * 129}},
    )
    assert raw.get("isError") is True


def test_export_unity_fbx_bounds_fps_and_refuses_overwrite(tmp_path):
    for fps in (0, 121, float("nan"), "fast"):
        raw, data = _call("ghostrigger_export_unity_fbx", {"options_overrides": {"fps": fps}})
        assert raw.get("isError") is True and data["ok"] is False
    raw, _ = _call("ghostrigger_export_unity_fbx", {"options_overrides": {"blender_executable": "x.exe"}})
    assert raw.get("isError") is True
    existing = tmp_path / "pmbam__g1a1__to__quinn_aligned.fbx"
    existing.write_bytes(b"prior")
    raw, data = _call("ghostrigger_export_unity_fbx", {"output_dir": str(tmp_path), "overwrite": False})
    assert raw.get("isError") is True
    assert data["existing"] == [str(existing)]
    assert existing.read_bytes() == b"prior"


def test_out_of_scope_day4_request_is_flagged_as_error():
    raw, data = _call("ghostrigger_export_unity_fbx", {"source_model_resref": "pfbam"})
    assert raw.get("isError") is True and "pmbam" in data["error"]


def test_day4_paths_resolve_to_repository_root():
    from src.core.retargeting import fbx_exporter, sampler

    assert fbx_exporter.BLENDER_SCRIPT.is_file()
    assert (Path(sampler.DEFAULT_CORPUS_ROOT) / "k1" / "pmbam.mdl").is_file()


def test_unity_output_paths_cannot_escape_the_project():
    from src.core.export.unity_export_bridge import build_output_paths

    project = Path("C:/Unity/Kaigen")
    for subdir, name in (
        ("../../Windows", "a"),
        ("Assets/../../x", "a"),
        ("C:/Other", "a"),
        ("Assets", "C:/game/model"),
        ("Assets", "..\\evil"),
        ("Assets", ".."),
    ):
        with pytest.raises(ValueError):
            build_output_paths(project, subdir, name, "fbx")
    asset, meta = build_output_paths(project, "Assets/KotorImported/GhostRigger", "p_bastilabb", "fbx")
    assert asset == project / "Assets/KotorImported/GhostRigger/p_bastilabb.fbx"
    assert meta.name == "p_bastilabb.ghostrigger.json"


def test_render_view_arguments_are_bounded():
    from src.kotormcp.tools.ghostrigger import _render_view_arguments

    assert _render_view_arguments({}) == (512, 512, 45.0, 30.0)
    for bad in ({"width": 0}, {"height": 5000}, {"elevation": 91}, {"azimuth": float("inf")}, {"width": "x"}):
        with pytest.raises(ValueError):
            _render_view_arguments(bad)


def test_render_model_refuses_existing_output_before_rendering(tmp_path):
    existing = tmp_path / "render.png"
    existing.write_bytes(b"prior")
    raw, data = _call(
        "ghostrigger_render_model",
        {"resref": "does_not_matter", "output_path": str(existing), "overwrite": False},
    )
    assert raw.get("isError") is True and "overwrite" in data["error"]
    assert existing.read_bytes() == b"prior"
    raw, data = _call("ghostrigger_render_model", {"resref": "x", "width": 100000})
    assert raw.get("isError") is True and "traceback" not in data


def test_source_identity_rejects_unknown_mode():
    raw, data = _call("ghostrigger_source_identity", {"mode": "anything"})
    assert raw.get("isError") is True and "mode" in data["error"]


def test_debug_session_detects_k2_and_bounds_bundle_names(tmp_path):
    from src.kotormcp.tools.debug_skinning import _DebugSession

    session = _DebugSession()
    session.game_dir = str(tmp_path)
    assert session.detect_game_tag() == "K1"
    (tmp_path / "swkotor2.exe").write_bytes(b"")
    assert session.detect_game_tag() == "K2"

    session.model = SimpleNamespace()
    result = session.capture_validation_set("../escape", str(tmp_path / "set"))
    assert result["ok"] is False and not (tmp_path / "set").exists()
    busy = tmp_path / "bundle"
    busy.mkdir()
    (busy / "prior.json").write_text("{}", encoding="utf-8")
    result = session.export_debug_bundle(str(busy), exclusive=True)
    assert result["ok"] is False
    assert sorted(p.name for p in busy.iterdir()) == ["prior.json"]


def test_render_debug_bundle_requires_absolute_exclusive_path(tmp_path, monkeypatch):
    from src.kotormcp.tools import debug_materials

    model = SimpleNamespace(name="fixture", supermodel="NULL", classification="character", all_nodes=lambda: [])
    monkeypatch.setattr(debug_materials, "_get_model", lambda: model)
    raw, _ = _call("ghostrigger_export_render_debug_bundle", {"output_path": "relative.json"})
    assert raw.get("isError") is True
    target = tmp_path / "bundle.json"
    target.write_text("prior", encoding="utf-8")
    raw, data = _call("ghostrigger_export_render_debug_bundle", {"output_path": str(target), "overwrite": False})
    assert raw.get("isError") is True and "overwrite" in data["error"]
    assert target.read_text(encoding="utf-8") == "prior"
    fresh = tmp_path / "fresh.json"
    raw, data = _call("ghostrigger_export_render_debug_bundle", {"output_path": str(fresh), "overwrite": False})
    assert not raw.get("isError") and json.loads(fresh.read_text(encoding="utf-8"))["model_name"] == "fixture"


def test_blender_discovery_prefers_newest_supported_major(tmp_path, monkeypatch):
    from src.core.retargeting import fbx_exporter

    for version in ("Blender 4.2", "Blender 4.5", "Blender 5.0"):
        (tmp_path / version).mkdir()
        (tmp_path / version / "blender.exe").write_bytes(b"")
    monkeypatch.setattr(fbx_exporter, "_blender_install_roots", lambda: [tmp_path])
    found = fbx_exporter._discover_blender_executables()
    assert [path.parent.name for path in found] == ["Blender 4.5", "Blender 4.2", "Blender 5.0"]
