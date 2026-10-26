"""Bounded-contract checks for the headless debug session Kaigen wraps (2026-09-29).

Fast, offline: no GPU context, no game installation, no model parse.
"""

import asyncio
import hashlib
import importlib
import json
import math
import sys
import types
from types import SimpleNamespace

import pytest


def _call(name, arguments):
    from src.kotormcp.tools import handle_tool

    raw = asyncio.run(handle_tool(name, arguments))
    return raw, json.loads(raw["text"])


@pytest.fixture
def fake_renderer(monkeypatch):
    released = []

    class GpuRenderer:
        perf = {"backend": "fake"}

        def release(self):
            released.append(self)

    module = types.ModuleType("src.adapters.rendering.moderngl_legacy_bridge")
    module.GpuRenderer = GpuRenderer
    monkeypatch.setitem(sys.modules, "src.adapters.rendering.moderngl_legacy_bridge", module)
    return released


@pytest.fixture
def fake_locator(monkeypatch):
    calls = []

    def locate(resref, game, game_path):
        calls.append((resref, game, game_path))
        return f"installation:{resref}.mdl", b"MDL" + resref.encode(), b"MDX"

    services = SimpleNamespace(locator=SimpleNamespace(locate=locate))
    for name in ("kotormcp.tools.ghostrigger", "src.kotormcp.tools.ghostrigger"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        monkeypatch.setattr(module, "_get_services", lambda: services)
    return calls


def _install(tmp_path, exe="swkotor.exe"):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "models.bif").write_bytes(b"BIFFV1  ")
    (tmp_path / "chitin.key").write_bytes(b"KEY V1  ")
    (tmp_path / exe).write_bytes(b"")
    return str(tmp_path)


def _animated_session(session):
    engine = SimpleNamespace(
        play=lambda *a, **k: None,
        evaluate=lambda t: SimpleNamespace(nodes=[]),
        seek=lambda t: None,
        stop=lambda: None,
    )
    session.model = SimpleNamespace(animations=[SimpleNamespace(name="cwalk", length=2.0)])
    session.anim_engine = engine
    return session


def test_launch_mints_session_and_revision_tracks_state(fake_renderer):
    from src.kotormcp.tools.debug_skinning import _DebugSession

    session = _DebugSession()
    session.launch()
    first = session.session_id
    assert len(first) == 36
    revision = session.state_revision()
    assert len(revision) == 64
    session.set_camera("side")
    assert session.state_revision() != revision
    session.set_camera("front")
    assert session.state_revision() == revision  # identical state, identical revision
    session.launch()  # relaunch replaces the session and releases the renderer
    assert session.session_id != first and len(fake_renderer) == 1
    assert session.set_camera(["front"])["ok"] is False  # used to raise TypeError


def test_front_preset_is_the_camera_canonical_front():
    from src.core.camera.arcball_camera import ArcBallCamera
    from src.kotormcp.tools.debug_skinning import _DebugSession

    session = _DebugSession()
    assert _DebugSession.CAMERA_PRESETS["front"][0] == ArcBallCamera.DEFAULT_AZIMUTH
    assert (session.camera_azimuth, session.camera_elevation) == _DebugSession.CAMERA_PRESETS["front"]
    assert _DebugSession.CAMERA_PRESETS["back"][0] % 360 == (ArcBallCamera.DEFAULT_AZIMUTH + 180) % 360


def test_animation_inputs_are_bounded(fake_renderer):
    from src.kotormcp.tools.debug_skinning import _DebugSession

    session = _animated_session(_DebugSession())
    assert session.set_animation("dance")["ok"] is False
    assert session.set_animation("../x")["ok"] is False
    assert session.set_animation("cwalk")["ok"] is True
    for bad in (math.nan, math.inf, -0.1, 2.5, True, "1.0", None):
        assert session.set_animation_time(bad)["ok"] is False, bad
    assert session.set_animation_time(2.0)["ok"] is True
    assert session.current_anim_time == 2.0


def test_load_model_uses_the_export_locator_and_binds_source(tmp_path, fake_locator, fake_renderer, monkeypatch):
    from src.kotormcp.tools import debug_skinning

    loader = importlib.import_module("src.core.game.kotor_loader")
    monkeypatch.setattr(loader, "load_model_from_bytes", lambda mdl, mdx: SimpleNamespace(animations=[]))
    game_dir = _install(tmp_path, "swkotor2.exe")
    session = debug_skinning._DebugSession()
    session.game_dir = game_dir
    session.game_tag = session.detect_game_tag()
    session.game_verified = True
    for bad in ("../../evil", "a" * 17, "", None, "C:/x"):
        assert session.load_model(bad)["ok"] is False
    assert not fake_locator
    session.anim_base_pose = object()  # stale bind reference from a previous model
    result = session.load_model("PMBAM")
    assert result["ok"] is True
    assert fake_locator == [("pmbam", "k2", game_dir)]
    assert result["source"] == {
        "resref": "pmbam",
        "path": "installation:pmbam.mdl",
        "mdl": {"sha256": hashlib.sha256(b"MDLpmbam").hexdigest(), "bytes": 8},
        "mdx": {"sha256": hashlib.sha256(b"MDX").hexdigest(), "bytes": 3},
    }
    assert session.anim_base_pose is None
    # The preview identity resolves identically without touching a session.
    identity = debug_skinning.session_source_identity("pmbam", game_dir)
    assert identity == {"model": result["source"], "game_tag": "K2"}
    with pytest.raises(ValueError):
        debug_skinning.session_source_identity("pmbam", "relative")


def test_raw_handlers_refuse_unbounded_inputs(tmp_path):
    from src.kotormcp.tools import debug_skinning

    debug_skinning._reset_session()
    try:
        raw, data = _call("ghostrigger_debug_set_game_library_path", {"path": "relative/dir"})
        assert raw.get("isError") is True and "absolute" in data["error"]
        raw, data = _call("ghostrigger_debug_capture_viewport", {"width": "512"})
        assert raw.get("isError") is True
        raw, data = _call("ghostrigger_debug_capture_viewport", {"output_path": "shot.png"})
        assert raw.get("isError") is True and "absolute" in data["error"]
        raw, data = _call("ghostrigger_debug_capture_validation_set", {"set_name": "qual"})
        assert raw.get("isError") is True and "output_dir" in data["error"]
        busy = tmp_path / "busy"
        busy.mkdir()
        (busy / "prior.png").write_bytes(b"prior")
        raw, data = _call(
            "ghostrigger_debug_capture_validation_set",
            {"set_name": "qual", "output_dir": str(busy), "overwrite": False},
        )
        assert raw.get("isError") is True and "non-empty" in data["error"]
        assert sorted(p.name for p in busy.iterdir()) == ["prior.png"]
        raw, data = _call("ghostrigger_debug_set_animation_time", {"time": 1.0})
        assert raw.get("isError") is True
        raw, data = _call("ghostrigger_debug_get_runtime_status", {})
        assert data["started"] is False and data["revision"] is None
    finally:
        debug_skinning._reset_session()


def test_source_identity_debug_session_mode_replaces_bif_mode():
    raw, data = _call("ghostrigger_source_identity", {"mode": "debug_bif", "resref": "x", "game_path": "C:/"})
    assert raw.get("isError") is True
    raw, data = _call("ghostrigger_source_identity", {"mode": "debug_session", "resref": "x", "game_path": "relative"})
    assert raw.get("isError") is True and "absolute" in data["error"]
