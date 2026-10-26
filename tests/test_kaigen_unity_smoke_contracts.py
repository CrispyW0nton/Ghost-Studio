"""Bounded-contract checks for ghostrigger_run_malak_unity_smoke (Kaigen unity_legacy).

A loopback fake of the Unity MCP Ghost editor bridge enforces what the real
0.2.x bridge enforces (bearer token, pinned project/editor IDs, JSON-RPC id
binding, idempotency-key replay) so the real HTTP client is exercised: the
token only ever goes to the identity's loopback port, proxies are bypassed,
identities that point elsewhere are refused before any request, unsaved or
untitled scenes block every mutation, screenshots are read only from the
bridge's own folder, outputs are exclusive and the prior scene setup is
restored. No Unity Editor is started.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.core.special import unity_malak_smoke as smoke

SCENE = "Assets/KaigenSmoke/Scenes/Smoke.unity"
ASSET = "Assets/KaigenSmoke/Models/n_darthmalak.fbx"
INSTANCE = "MalakSmokeInstance"
PRIOR = "Assets/KaigenSmoke/Scenes/Prior.unity"


class FakeBridge:
    """Loopback JSON-RPC bridge with Unity MCP Ghost's authentication rules."""

    def __init__(self, project: Path, *, token: str | None = None) -> None:
        self.project = project
        self.token = token or secrets.token_hex(32)
        self.project_id = smoke.project_id_for_path(project)
        self.editor_id = secrets.token_hex(16)
        self.requests: list[dict] = []
        self.refused: list[int] = []
        self.cache: dict[str, dict] = {}
        self.setup = [{"path": PRIOR, "isLoaded": True, "isActive": True}]
        self.dirty = False
        self.playing = False
        self.health_path: str | None = None
        self.screenshot_path_override: str | None = None
        self.fail: dict[str, dict] = {}
        self.restore_answer: dict | None = None
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                return

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                    bridge.refused.append(401)
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Bearer realm="unity-mcp-ghost"')
                    self.end_headers()
                    return
                rpc = json.loads(body)
                bridge.requests.append(rpc)
                answer = bridge.dispatch(rpc)
                raw = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        state = project / "Library" / "UnityMcpGhost"
        state.mkdir(parents=True, exist_ok=True)
        (state / "bridge-token.txt").write_text(self.token, encoding="ascii")
        self.write_identity()

    def write_identity(self, **overrides) -> None:
        state = self.project / "Library" / "UnityMcpGhost"
        identity = {
            "schemaVersion": 1,
            "projectId": self.project_id,
            "editorInstanceId": self.editor_id,
            "projectRoot": str(self.project),
            "port": self.port,
            "webSocketPort": self.port + 1,
            "host": "127.0.0.1",
            "path": "/unity-mcp-ghost",
            "running": True,
            "startupFailure": None,
            "heartbeatUtc": "2026-09-30T00:00:00Z",
            "packageVersion": "0.2.0-rc.1",
            "unityVersion": "2022.3.62f1",
            "tokenFile": str(state / "bridge-token.txt"),
            "capabilities": {"destructive": False},
        }
        identity.update(overrides)
        (state / "bridge-identity.json").write_text(json.dumps(identity), encoding="utf-8")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def methods(self) -> list[str]:
        return [r["method"] for r in self.requests]

    def dispatch(self, rpc: dict) -> dict:
        rid = rpc.get("id")
        if rpc.get("requestId") != rid or rpc.get("jsonrpc") != "2.0":
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "invalid request"}}
        if rpc.get("projectId") != self.project_id or rpc.get("editorInstanceId") != self.editor_id:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32010, "message": "pinned elsewhere", "data": {"code": "PROJECT_ID_MISMATCH"}}}
        key = rpc.get("idempotencyKey")
        if key in self.cache:
            return {**self.cache[key], "id": rid}
        method, params = rpc["method"], rpc.get("params") or {}
        if method in self.fail:
            answer = {"jsonrpc": "2.0", "id": rid, "error": self.fail[method]}
        else:
            answer = {"jsonrpc": "2.0", "id": rid, "result": self.result(method, params)}
        self.cache[key] = answer
        return answer

    def result(self, method: str, params: dict) -> dict:
        if method == "health":
            return {
                "ok": True, "authenticated": True, "projectId": self.project_id, "editorInstanceId": self.editor_id,
                "projectPath": self.health_path or str(self.project), "unityVersion": "2022.3.62f1", "version": "0.2.0-rc.1",
                "protocolVersion": 1, "activeSceneDirty": self.dirty, "isPlaying": self.playing, "isCompiling": False,
                "durableQueue": {"pendingCount": 0, "processingCount": 0},
            }
        if method == "editor.snapshot":
            return {"ok": True, "sceneSetup": [dict(r) for r in self.setup], "dirtyScenes": [PRIOR] if self.dirty else []}
        if method == "editor.get_state":
            return {"ok": True, "isCompiling": False, "isPlaying": self.playing, "isUpdating": False,
                    "activeSceneDirty": self.dirty, "activeScenePath": self.setup[0]["path"]}
        if method == "editor.restore_snapshot":
            if self.restore_answer is not None:
                return self.restore_answer
            self.setup = [dict(r) for r in params["snapshot"]["sceneSetup"]]
            return {"ok": True}
        if method == "scene.open":
            self.setup = [{"path": params["path"], "isLoaded": True, "isActive": True}]
            return {"ok": True, "scene": {"path": params["path"], "isLoaded": True, "isDirty": False}}
        if method == "asset.refresh":
            return {"ok": True, "refreshed": True}
        if method == "asset.find":
            return {"ok": True, "assets": [{"path": ASSET}]}
        if method == "screenshot.capture":
            from PIL import Image

            folder = self.project / "Library" / "UnityMcpGhost" / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{params['label']}.png"
            Image.new("RGB", (params["width"], params["height"]), (10, 20, 30)).save(path)
            return {"ok": True, "path": self.screenshot_path_override or str(path), "camera": "Main Camera",
                    "width": params["width"], "height": params["height"]}
        if method == "scene.get_hierarchy":
            return {"ok": True, "roots": [{"name": INSTANCE, "instanceId": 7, "children": [
                {"name": "body", "instanceId": 8, "children": []}]}]}
        if method == "component.get":
            if params["instanceId"] == 7:
                return {"ok": True, "components": [{"type": "UnityEngine.Transform"}, {"type": "UnityEngine.Animator"}]}
            return {"ok": True, "components": [{"type": "UnityEngine.SkinnedMeshRenderer"}]}
        if method == "gameobject.find":
            return {"ok": True, "matches": [{"name": INSTANCE, "instanceId": 7, "scenePath": SCENE}]}
        if method in ("compile.diagnostics_get", "console.diagnostics_get"):
            return {"ok": True, "diagnostics": []}
        raise AssertionError(method)


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "UnityProject"
    (root / "Assets" / "KaigenSmoke" / "Models").mkdir(parents=True)
    (root / "Assets" / "KaigenSmoke" / "Scenes").mkdir(parents=True)
    (root / SCENE).write_text("%YAML 1.1\n", encoding="utf-8")
    (root / ASSET).write_bytes(b"Kaydara FBX Binary  \x00fixture")
    sidecar = root / (ASSET[:-4] + ".ghostrigger.json")
    sidecar.write_text(json.dumps({"animations": ["walk", "pause1"]}), encoding="utf-8")
    return root


@pytest.fixture()
def bridge(project):
    fake = FakeBridge(project)
    yield fake
    fake.close()


def _run(project, **kwargs):
    base = dict(unity_project=project, scene_path=SCENE, asset_path=ASSET, instance_name=INSTANCE, screenshot_delay=0.0)
    base.update(kwargs)
    return smoke.run_malak_main_menu_smoke(**base)


def test_authenticated_round_trip_restores_setup_and_writes_exclusively(project, bridge, tmp_path):
    out = tmp_path / "op"
    shots = out / "screenshots"
    shots.mkdir(parents=True)
    report = _run(project, output_path=out / "report.json", screenshot_dir=shots, operation_id="op-1")

    assert report["ok"] is True and report["status"] == "warning"  # identical frames: edit mode
    assert {w["code"] for w in report["warnings"]} == {"no_visible_delta"}
    assert report["restore"] == {"attempted": True, "ok": True, "verified": True}
    assert bridge.setup == [{"path": PRIOR, "isLoaded": True, "isActive": True}]
    methods = bridge.methods()
    assert methods[:4] == ["health", "editor.snapshot", "editor.get_state", "scene.open"]
    assert methods[-2:] == ["editor.restore_snapshot", "editor.snapshot"]
    mutations = [r for r in bridge.requests if r["idempotent"] is False]
    assert [r["method"] for r in mutations] == [
        "scene.open", "asset.refresh", "screenshot.capture", "screenshot.capture", "editor.restore_snapshot"]
    assert all(r["idempotencyKey"].startswith("op-1-") for r in mutations)
    assert sorted(p.name for p in shots.iterdir()) == ["after.png", "before.png"]
    text = (out / "report.json").read_text(encoding="utf-8")
    assert bridge.token not in text and bridge.token not in json.dumps(report, default=str)
    assert report["screenshots"]["before"]["copy"]["sha256"] == hashlib.sha256((shots / "before.png").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        _run(project, output_path=out / "report.json")


def test_http_proxy_environment_is_bypassed(project, bridge, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    report = _run(project, dry_run=True)
    assert report["status"] == "ready" and report["blockers"] == []
    assert bridge.methods() == ["health", "editor.snapshot", "editor.get_state"]


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"tokenFile": "C:/Users/Public/elsewhere.txt"}, "BRIDGE_TOKEN_LOCATION"),
        ({"host": "0.0.0.0"}, "BRIDGE_NOT_LOOPBACK"),
        ({"projectId": "unity-project-000000000000000000000000"}, "BRIDGE_PROJECT_MISMATCH"),
        ({"projectRoot": "C:/Somewhere/Else"}, "BRIDGE_PROJECT_MISMATCH"),
        ({"running": False}, "BRIDGE_NOT_RUNNING"),
        ({"port": 80}, "BRIDGE_IDENTITY_INVALID"),
        ({"editorInstanceId": "../x"}, "BRIDGE_IDENTITY_INVALID"),
    ],
)
def test_identity_refusals_happen_before_any_request(project, bridge, override, code):
    bridge.write_identity(**override)
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project, dry_run=True)
    assert caught.value.code == code
    assert bridge.requests == [] and bridge.refused == []


def test_expected_identity_binding_refuses_before_sending(project, bridge):
    # The caller verified the listener on one port/editor; a rewritten identity is refused unsent.
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project, dry_run=True, expected_port=bridge.port + 1 if bridge.port < 65535 else 1024)
    assert caught.value.code == "BRIDGE_IDENTITY_CHANGED"
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project, dry_run=True, expected_editor_instance_id="0" * 32)
    assert caught.value.code == "BRIDGE_IDENTITY_CHANGED"
    assert bridge.requests == [] and bridge.refused == []
    with pytest.raises(ValueError):
        _run(project, dry_run=True, expected_port=True)
    ready = _run(project, dry_run=True, expected_port=bridge.port, expected_editor_instance_id=bridge.editor_id)
    assert ready["status"] == "ready"


def test_invalid_token_file_and_wrong_token(project, bridge):
    token_file = project / "Library" / "UnityMcpGhost" / "bridge-token.txt"
    token_file.write_text("not-a-token", encoding="ascii")
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project, dry_run=True)
    assert caught.value.code == "BRIDGE_TOKEN_INVALID" and bridge.refused == []
    token_file.write_text(secrets.token_hex(32), encoding="ascii")
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project, dry_run=True)
    assert caught.value.code == "BRIDGE_AUTH_REFUSED" and bridge.refused == [401]


def test_unsaved_or_playing_editor_blocks_every_mutation(project, bridge):
    bridge.dirty = True
    ready = _run(project, dry_run=True)
    assert ready["status"] == "blocked" and any(b.startswith("scene-dirty") for b in ready["blockers"])
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project)
    assert caught.value.code == "EDITOR_NOT_READY"
    bridge.dirty, bridge.playing = False, True
    with pytest.raises(smoke.UnityBridgeError):
        _run(project)
    bridge.playing = False
    bridge.setup = [{"path": "", "isLoaded": True, "isActive": True}]
    assert any(b.startswith("untitled-scene") for b in _run(project, dry_run=True)["blockers"])
    assert "scene.open" not in bridge.methods()


def test_health_for_another_project_is_refused_before_mutation(project, bridge):
    bridge.health_path = str(project.parent / "OtherProject")
    with pytest.raises(smoke.UnityBridgeError) as caught:
        _run(project)
    assert caught.value.code == "BRIDGE_IDENTITY_MISMATCH"
    assert bridge.methods() == ["health"]


def test_screenshot_outside_bridge_folder_is_not_read(project, bridge, tmp_path):
    from PIL import Image

    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (4, 4)).save(foreign)
    bridge.screenshot_path_override = str(foreign)
    shots = tmp_path / "shots"
    shots.mkdir()
    report = _run(project, screenshot_dir=shots)
    assert "screenshot_unreadable" in {w["code"] for w in report["warnings"]}
    assert list(shots.iterdir()) == []


def test_aborted_step_and_failed_restore_are_reported(project, bridge):
    bridge.fail["asset.find"] = {"code": -32000, "message": "boom"}
    report = _run(project)
    assert report["aborted"]["step"] == "asset.find" and report["status"] == "error"
    assert report["restore"]["verified"] is True
    bridge.fail.clear()
    bridge.restore_answer = {"ok": False, "error": {"code": "EDITOR_RESTORE_DIRTY_SCENE_CONFLICT"}}
    report = _run(project)
    assert {"restore_failed"} <= {i["code"] for i in report["issues"]}
    assert report["restore"]["code"] == "EDITOR_RESTORE_DIRTY_SCENE_CONFLICT"


@pytest.mark.parametrize(
    "overrides",
    [
        {"scene_path": "Assets/../Secrets.unity"},
        {"scene_path": "Assets\\Scenes\\Smoke.unity"},
        {"scene_path": "C:/Assets/Smoke.unity"},
        {"scene_path": "Assets/Scenes/Smoke.prefab"},
        {"asset_path": "Assets/nul.fbx"},
        {"asset_path": "Packages/model.fbx"},
        {"instance_name": "bad\nname"},
        {"screenshot_width": 4096},
        {"screenshot_delay": float("nan")},
        {"operation_id": "../../x"},
    ],
)
def test_input_validation(project, bridge, overrides):
    with pytest.raises(ValueError):
        _run(project, **overrides)
    assert bridge.requests == []


def test_handler_rejects_host_and_flags_errors_without_tracebacks(project, bridge, tmp_path):
    from kotormcp.tools.ghostrigger import handle_run_malak_unity_smoke

    refused = asyncio.run(handle_run_malak_unity_smoke({"unity_project": str(project), "host": "10.0.0.5"}))
    assert refused["isError"] is True and json.loads(refused["text"])["code"] == "INVALID_ARGUMENT"
    missing = asyncio.run(handle_run_malak_unity_smoke({"unity_project": str(tmp_path)}))
    payload = json.loads(missing["text"])
    assert missing["isError"] is True and "Traceback" not in missing["text"]
    assert payload["code"] == "INVALID_ARGUMENT"  # no Assets/ folder
    ok = asyncio.run(handle_run_malak_unity_smoke({
        "unity_project": str(project), "scene_path": SCENE, "asset_path": ASSET,
        "instance_name": INSTANCE, "dry_run": True,
    }))
    assert "isError" not in ok and json.loads(ok["text"])["status"] == "ready"
    assert bridge.token not in ok["text"]
