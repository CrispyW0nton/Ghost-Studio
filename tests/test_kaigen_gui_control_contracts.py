"""Kaigen GUI control channel contracts (2026-09-29).

Covers the authenticated loopback listener (src/ipc/kaigen_gui_control.py) and
the window adapter's open-model guards (gui/.../shared/kaigen_control.py).
Offline: no Qt event loop, no game installation, no model parse.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import secrets
import time
import uuid
from types import SimpleNamespace

import pytest

from src.ipc import kaigen_gui_control as control
from src.gui.windows.application_core.shared import kaigen_control as adapter

SHA = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731


def _handlers(calls):
    def state(_payload):
        calls.append("state")
        return {"state": {"revision": "r"}}

    def open_model(payload):
        calls.append(("open", payload["resref"]))
        if payload["resref"] == "refuse":
            raise control.ControlRefusal("scene-dirty", "unsaved changes")
        if payload["resref"] == "boom":
            raise RuntimeError("C:/secret/path traceback")
        return {"opened": True}

    return {"state": state, "open_model": open_model, "capture": lambda _p: {"png": ""}}


@pytest.fixture
def channel(tmp_path):
    calls: list = []
    server = control.KaigenGuiControlServer(tmp_path / "gui" / "session.json", _handlers(calls))
    server.start()
    yield server, calls, tmp_path / "gui" / "session.json"
    server.stop()


def _post(server, path, payload, *, secret=None, session=None, nonce=None, timestamp=None, host=None, method="POST"):
    descriptor = json.loads(server._descriptor_path.read_text(encoding="utf-8"))
    body = json.dumps(payload).encode("utf-8")
    secret = bytes.fromhex(secret or descriptor["secret"])
    session = session or descriptor["sessionId"]
    nonce = nonce or secrets.token_urlsafe(24)
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    headers = {
        "Content-Type": "application/json",
        "X-Kaigen-Session": session,
        "X-Kaigen-Timestamp": timestamp,
        "X-Kaigen-Nonce": nonce,
        "X-Kaigen-Signature": control.request_signature(
            secret, method=method, path=path, session=session, timestamp=timestamp, nonce=nonce, body=body
        ),
    }
    if host:
        headers["Host"] = host
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    signature = response.getheader("X-Kaigen-Response-Signature")
    conn.close()
    signed = signature == control.response_signature(
        bytes.fromhex(descriptor["secret"]), status=response.status, session=descriptor["sessionId"], nonce=nonce, body=raw
    )
    return response.status, json.loads(raw), signed


def test_descriptor_and_signed_state(channel):
    server, calls, path = channel
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    assert descriptor["schema"] == control.DESCRIPTOR_SCHEMA
    assert descriptor["port"] == server.port and len(descriptor["secret"]) == 64
    assert descriptor["source"] == {"ipc/kaigen_gui_control.py": control.MODULE_SHA256}
    status, data, signed = _post(server, "/kaigen/v1/state", {})
    assert (status, data["status"], signed) == (200, "ok", True)
    assert descriptor["secret"] not in json.dumps(data)
    assert calls == ["state"]


def test_authentication_failures(channel):
    server, calls, _ = channel
    assert _post(server, "/kaigen/v1/state", {}, secret="11" * 32)[:2][0] == 401
    assert _post(server, "/kaigen/v1/state", {}, session=uuid.uuid4().hex)[0] == 401
    assert _post(server, "/kaigen/v1/state", {}, timestamp=int(time.time()) - 120)[0] == 401
    nonce = secrets.token_urlsafe(24)
    assert _post(server, "/kaigen/v1/state", {}, nonce=nonce)[0] == 200
    assert _post(server, "/kaigen/v1/state", {}, nonce=nonce)[1]["code"] == "replayed-request"
    assert _post(server, "/kaigen/v1/state", {}, host="evil.example:80")[0] == 421
    assert _post(server, "/api/open_mdl", {})[0] == 404
    assert _post(server, "/kaigen/v1/state", {}, method="GET")[0] == 405
    # Unauthenticated answers are never signed.
    assert _post(server, "/kaigen/v1/state", {}, secret="22" * 32)[2] is False
    assert calls == ["state"]


def test_strict_payloads_and_refusals(channel):
    server, calls, _ = channel
    valid = {
        "requestId": str(uuid.uuid4()),
        "resref": "plc_bench",
        "game": "K1",
        "gameDir": "C:/Games/swkotor",
        "mdlSha256": "a" * 64,
        "mdxSha256": "b" * 64,
        "expectedGuiRevision": "c" * 64,
    }
    for bad in (
        {**valid, "output_path": "C:/x"},
        {**valid, "resref": "../x"},
        {**valid, "game": "k3"},
        {**valid, "gameDir": "relative/dir"},
        {**valid, "mdlSha256": "A" * 64},
        {key: value for key, value in valid.items() if key != "expectedGuiRevision"},
    ):
        status, data, signed = _post(server, "/kaigen/v1/open-model", bad)
        assert (status, data["code"], signed) == (400, "invalid-payload", True)
    assert _post(server, "/kaigen/v1/state", {"extra": 1})[0] == 400
    assert _post(server, "/kaigen/v1/capture", {"captureId": "x"})[0] == 400
    status, data, _ = _post(server, "/kaigen/v1/open-model", {**valid, "resref": "refuse"})
    assert (status, data["status"], data["code"]) == (409, "refused", "scene-dirty")
    status, data, _ = _post(server, "/kaigen/v1/open-model", {**valid, "resref": "boom"})
    assert status == 500 and "secret" not in json.dumps(data)
    assert _post(server, "/kaigen/v1/open-model", valid)[1]["opened"] is True
    assert calls == [("open", "refuse"), ("open", "boom"), ("open", "plc_bench")]


def test_stop_removes_only_its_own_descriptor(tmp_path):
    path = tmp_path / "session.json"
    first = control.KaigenGuiControlServer(path, _handlers([]))
    first.start()
    first.stop()
    assert not path.exists()
    second = control.KaigenGuiControlServer(path, _handlers([]))
    second.start()
    third = control.KaigenGuiControlServer(path, _handlers([]))
    third.start()
    second.stop()  # third published since: its descriptor stays
    assert json.loads(path.read_text(encoding="utf-8"))["sessionId"] == third.session
    third.stop()
    assert not path.exists()


class _Scene:
    def __init__(self):
        self.objects, self.dirty, self.path, self.game, self.display_name = [], False, "", "K1", "Untitled Scene"


class _Window:
    def __init__(self, game_dir):
        self.scene_manager = SimpleNamespace(active_scene=_Scene())
        self.scene_manager.is_dirty = lambda: self.scene_manager.active_scene.dirty
        self._current_model = None
        self._model_path = ""
        self._current_game = "K1"
        self._pending_gpu_upload_total = 0
        self.running = False
        self.game_dir = str(game_dir)
        self.events = []

    def _model_worker_is_running(self):
        return self.running

    def _configured_game_dirs(self):
        return self.game_dir, ""

    def _create_new_scene_from_ipc(self, game):
        self.events.append(("new_scene", game))
        self.scene_manager.active_scene = _Scene()

    def _on_model_loaded(self, model, label, error):
        self.events.append(("loaded", label, error))
        self._current_model, self._model_path = model, label
        scene = self.scene_manager.active_scene
        scene.objects.append(SimpleNamespace(id="1", name=model.name, object_type="model", source_ref=None))
        scene.dirty = True

    def _log(self, *_args):
        pass


def _request(window, **overrides):
    mdl, mdx = b"mdl", b"mdx"
    request = {
        "requestId": str(uuid.uuid4()),
        "resref": "plc_bench",
        "game": "K1",
        "gameDir": window.game_dir,
        "mdlSha256": SHA(mdl),
        "mdxSha256": SHA(mdx),
        "expectedGuiRevision": adapter.gui_state(window)["revision"],
    }
    request.update(overrides)
    return request


def _open(window, request, *, mdl=b"mdl", mdx=b"mdx"):
    return adapter.open_game_model(
        window,
        request,
        resolver=lambda game_dir, game, resref: (mdl, mdx),
        parser=lambda m, x, game: SimpleNamespace(name="PLC_Bench", node_count=lambda: 3, mesh_nodes=lambda: [1]),
    )


def test_open_model_guards_and_success(tmp_path):
    window = _Window(tmp_path)
    for mutate, code in (
        (lambda w: setattr(w, "running", True), "busy"),
        (lambda w: setattr(w.scene_manager.active_scene, "dirty", True), "scene-dirty"),
    ):
        blocked = _Window(tmp_path)
        request = _request(blocked)
        mutate(blocked)
        with pytest.raises(control.ControlRefusal) as refusal:
            _open(blocked, request)
        assert refusal.value.code == code and blocked.events == []
    with pytest.raises(control.ControlRefusal) as refusal:
        _open(window, _request(window, expectedGuiRevision="0" * 64))
    assert refusal.value.code == "gui-drift"
    with pytest.raises(control.ControlRefusal) as refusal:
        _open(window, _request(window, gameDir=str(tmp_path / "other")))
    assert refusal.value.code == "game-dir-mismatch"
    with pytest.raises(control.ControlRefusal) as refusal:
        _open(window, _request(window), mdl=b"different")
    assert refusal.value.code == "source-mismatch" and window.events == []
    result = _open(window, _request(window))
    assert window.events == [("new_scene", "K1"), ("loaded", "K1:plc_bench", "")]
    model = result["state"]["model"]
    assert (model["resref"], model["game"], model["mdlSha256"], model["mdxSha256"]) == (
        "plc_bench",
        "K1",
        SHA(b"mdl"),
        SHA(b"mdx"),
    )
    assert result["state"]["scene"]["dirty"] is True and result["opened"] is True
    # The now-dirty scene refuses a second open.
    with pytest.raises(control.ControlRefusal) as refusal:
        _open(window, _request(window))
    assert refusal.value.code == "scene-dirty"


def test_channel_is_opt_in():
    assert adapter.start_kaigen_gui_control(object(), environ={}) is None
    assert adapter.start_kaigen_gui_control(object(), environ={control.CONTROL_PATH_ENV: "relative.json"}) is None
