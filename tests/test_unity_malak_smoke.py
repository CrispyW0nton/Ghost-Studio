import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from src.core.special.unity_malak_smoke import (
    DEFAULT_ASSET_PATH,
    DEFAULT_SCENE_PATH,
    BridgeIdentity,
    choose_preferred_clip,
    metadata_path_for_asset,
    project_id_for_path,
    run_malak_main_menu_smoke,
)
from kotormcp.tools import get_all_tools


class _FakeUnityClient:
    """Test double for the authenticated bridge client (identity + request)."""

    def __init__(self, unity_project, *, include_animator=True, renderer_type="UnityEngine.SkinnedMeshRenderer"):
        self.unity_project = Path(unity_project)
        self.include_animator = include_animator
        self.renderer_type = renderer_type
        self.calls = []
        self.screenshot_index = 0
        self.identity = BridgeIdentity(
            self.unity_project, project_id_for_path(self.unity_project), "a" * 32, 6401, "0.2.0", "2022.3.62f1", "",
        )
        self.setup = [{"path": DEFAULT_SCENE_PATH, "isLoaded": True, "isActive": True}]

    def request(self, method, params=None, **_kwargs):
        self.calls.append((method, params or {}))
        if method == "health":
            return {
                "ok": True, "authenticated": True, "projectId": self.identity.project_id,
                "editorInstanceId": self.identity.editor_instance_id, "projectPath": str(self.unity_project),
                "protocolVersion": 1, "durableQueue": {"pendingCount": 0, "processingCount": 0},
            }
        if method == "editor.snapshot":
            return {"ok": True, "sceneSetup": [dict(row) for row in self.setup], "dirtyScenes": []}
        if method == "editor.get_state":
            return {"ok": True, "isCompiling": False, "isPlaying": False, "isUpdating": False}
        if method == "editor.restore_snapshot":
            self.setup = [dict(row) for row in params["snapshot"]["sceneSetup"]]
            return {"ok": True}
        if method == "scene.open":
            self.setup = [{"path": params["path"], "isLoaded": True, "isActive": True}]
            return {"ok": True, "scene": {"path": params["path"], "isLoaded": True}}
        if method == "asset.refresh":
            return {"ok": True, "refreshed": True}
        if method == "asset.find":
            return {"ok": True, "assets": [{"path": DEFAULT_ASSET_PATH, "type": "UnityEngine.GameObject"}]}
        if method == "screenshot.capture":
            return {"ok": True, "path": self._write_screenshot()}
        if method == "scene.get_hierarchy":
            return {
                "ok": True,
                "roots": [{
                    "name": "_MCP_KotorMainMenu",
                    "instanceId": 1,
                    "children": [{
                        "name": "MalakMenuModel_Animated006J",
                        "instanceId": 2,
                        "children": [{
                            "name": "torso",
                            "instanceId": 3,
                            "children": [],
                        }],
                    }],
                }],
            }
        if method == "gameobject.find":
            return {
                "ok": True,
                "matches": [{
                    "name": "MalakMenuModel_Animated006J",
                    "instanceId": 2,
                    "scenePath": DEFAULT_SCENE_PATH,
                }],
            }
        if method == "component.get":
            instance_id = params["instanceId"]
            if instance_id == 2:
                components = [{"type": "UnityEngine.Transform"}]
                if self.include_animator:
                    components.append({"type": "UnityEngine.Animator"})
                return {"ok": True, "components": components}
            if instance_id == 3:
                return {"ok": True, "components": [{"type": self.renderer_type}]}
            return {"ok": True, "components": []}
        if method in {"compile.diagnostics_get", "console.diagnostics_get"}:
            return {"ok": True, "diagnostics": []}
        raise AssertionError(f"Unexpected Unity method: {method}")

    def _write_screenshot(self):
        from PIL import Image

        self.screenshot_index += 1
        folder = self.unity_project / "Library" / "UnityMcpGhost" / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"shot-{self.screenshot_index}.png"
        color = (0, 0, 0) if self.screenshot_index == 1 else (64, 0, 0)
        Image.new("RGB", (4, 4), color).save(path)
        return str(path)


def _write_sidecar(unity_project: Path):
    sidecar = metadata_path_for_asset(unity_project, DEFAULT_ASSET_PATH)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "animations": ["walk", "pause1", "tlknorm"],
        "counts": {"animations": 3},
    }), encoding="utf-8")
    return sidecar


def test_choose_preferred_clip_uses_menu_priority():
    assert choose_preferred_clip(["walk", "pause2", "pause1"]) == "pause1"
    assert choose_preferred_clip(["custom_idle"]) == "custom_idle"
    assert choose_preferred_clip([]) is None


def test_metadata_path_for_asset_uses_ghostrigger_sidecar_name():
    unity_project = Path("C:/Unity/Kotor-Unity")

    path = metadata_path_for_asset(unity_project, DEFAULT_ASSET_PATH)

    assert path.as_posix().endswith(
        "Assets/KotorImported/MainMenu/Models/Malak/GhostRiggerFresh/"
        "N_DarthMalak_GhostRiggerFresh.ghostrigger.json"
    )


def test_run_malak_main_menu_smoke_writes_report(tmp_path):
    unity_project = tmp_path / "UnityProject"
    _write_sidecar(unity_project)
    client = _FakeUnityClient(unity_project)
    output = tmp_path / "report.json"

    report = run_malak_main_menu_smoke(
        unity_project=unity_project,
        client=client,
        output_path=output,
        screenshot_delay=0.0,
    )

    assert report["status"] == "ok"
    assert report["ok"] is True
    assert report["selected_clip"] == "pause1"
    assert report["unity_summary"]["malak_instance_count"] == 1
    assert report["unity_summary"]["component_types"]["UnityEngine.Animator"] == 1
    assert report["screenshot_delta"]["visible_delta"] is True
    assert report["restore"] == {"attempted": True, "ok": True, "verified": True}
    assert output.exists()
    methods = [call[0] for call in client.calls]
    assert methods[:6] == ["health", "editor.snapshot", "editor.get_state", "scene.open", "asset.refresh", "editor.get_state"]
    assert methods[-2:] == ["editor.restore_snapshot", "editor.snapshot"]


def test_run_malak_main_menu_smoke_reports_missing_animator(tmp_path):
    unity_project = tmp_path / "UnityProject"
    _write_sidecar(unity_project)

    report = run_malak_main_menu_smoke(
        unity_project=unity_project,
        client=_FakeUnityClient(unity_project, include_animator=False),
        screenshot_delay=0.0,
    )

    assert report["status"] == "error"
    assert report["ok"] is False
    assert {issue["code"] for issue in report["issues"]} == {"animator_missing"}


def test_mcp_tool_manifest_exposes_malak_smoke_action():
    names = {tool["name"] for tool in get_all_tools()}

    assert "ghostrigger_run_malak_unity_smoke" in names
