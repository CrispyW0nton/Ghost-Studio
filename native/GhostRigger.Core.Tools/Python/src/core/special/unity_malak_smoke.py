"""Unity MCP Ghost smoke test for a GhostRigger model handoff.

The defaults reproduce the legacy Malak main-menu fixture; every path is a
parameter. The Unity-MCP-Ghost editor bridge (0.2.x) authenticates every HTTP
request with the project's owner-only bearer token and pins the project and
editor-instance IDs that Unity advertises in
``<project>/Library/UnityMcpGhost/bridge-identity.json``. This client:

* talks only to ``http://127.0.0.1:<identity port>/unity-mcp-ghost/``: there
  is no host argument, HTTP proxies are bypassed and redirects are refused;
* reads the token only from ``<project>/Library/UnityMcpGhost/bridge-token.txt``
  (an identity that names another token file is refused) and never logs,
  returns or writes it;
* checks that the authenticated ``health`` answer names the same project
  path, project ID and editor instance before any mutation;
* refuses to change scenes while any open scene has unsaved changes or is
  untitled, while the editor plays, compiles or imports, or while the bridge
  queue is busy, and restores the prior scene setup afterwards through the
  bridge's own ``editor.snapshot`` / ``editor.restore_snapshot``;
* reads screenshots only from the bridge's ``Library/UnityMcpGhost/screenshots``
  folder and writes its report and screenshot copies with exclusive create.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_SCENE_PATH = "Assets/MCPGhostTests/MainMenuRecreation/MCP_Kotor_MainMenu_Recreation.unity"
DEFAULT_ASSET_PATH = (
    "Assets/KotorImported/MainMenu/Models/Malak/GhostRiggerFresh/"
    "N_DarthMalak_GhostRiggerFresh.fbx"
)
DEFAULT_INSTANCE_NAME = "MalakMenuModel_Animated006J"
PREFERRED_MALAK_CLIPS = ("pause1", "pause2", "tlknorm", "listen", "walk", "run")
TOOL_ID = "ghostrigger.malak_main_menu_smoke"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ANY_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_RESERVED = re.compile(r'[<>"|?*]')
_DEVICE_NAME = re.compile(r"^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?$", re.IGNORECASE)
_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_EDITOR_ID = re.compile(r"^[0-9a-f]{32}$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")

LOOPBACK_HOST = "127.0.0.1"
BRIDGE_PATH = "/unity-mcp-ghost/"
IDENTITY_FILE = "bridge-identity.json"
TOKEN_FILE = "bridge-token.txt"
MAX_IDENTITY_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 256
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SIDECAR_BYTES = 8 * 1024 * 1024
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024
MAX_ASSET_PATH = 512
MAX_INSTANCE_NAME = 128
MAX_SUBTREE_NODES = 512
MAX_CLIPS = 4096
SCREENSHOT_WIDTH = (64, 1920)
SCREENSHOT_HEIGHT = (64, 1080)
MAX_SCREENSHOT_DELAY = 10.0
IDLE_WAIT_SECONDS = 120.0
REQUEST_TIMEOUT = 60.0
REFRESH_TIMEOUT = 300.0


class UnityBridgeError(RuntimeError):
    """A refused or failed bridge interaction; ``code`` is machine readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


# ── Input validation ─────────────────────────────────────────────────────────

def validate_asset_path(value: Any, suffix: str, label: str) -> str:
    """Return a normalized ``Assets/...<suffix>`` path or raise ValueError."""
    if not isinstance(value, str) or not value or len(value) > MAX_ASSET_PATH:
        raise ValueError(f"{label} must be a Unity asset path of at most {MAX_ASSET_PATH} characters")
    if "\\" in value or ":" in value or _ANY_CONTROL.search(value) or _WINDOWS_RESERVED.search(value):
        raise ValueError(f"{label} must use forward slashes without drive letters or reserved characters")
    parts = value.split("/")
    if (
        len(parts) < 2
        or parts[0] != "Assets"
        or any(part in ("", ".", "..") or part != part.strip() or _DEVICE_NAME.match(part) for part in parts)
    ):
        raise ValueError(f"{label} must be a normalized path below Assets/")
    if not value.lower().endswith(suffix):
        raise ValueError(f"{label} must end with {suffix}")
    return value


def validate_instance_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_INSTANCE_NAME or _ANY_CONTROL.search(value):
        raise ValueError(f"instance_name must be 1-{MAX_INSTANCE_NAME} printable characters")
    return value


def _bounded(value: Any, low: float, high: float, label: str, *, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if integer and (not float(value).is_integer()):
        raise ValueError(f"{label} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}")
    return int(number) if integer else number


def _is_link(path: Path) -> bool:
    """True for a symbolic link, junction or any other reparse point."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse)


def _unlinked_below(root: Path, relative_parts: list[str]) -> Path:
    """Join parts below root, refusing any link or junction on the way."""
    current = root
    for part in relative_parts:
        current = current / part
        if _is_link(current):
            raise ValueError(f"{current.relative_to(root).as_posix()} must not be a link or junction")
    return current


def project_root(unity_project: Any) -> Path:
    raw = str(unity_project or "").strip()
    if not raw or not os.path.isabs(raw) or _ANY_CONTROL.search(raw):
        raise ValueError("unity_project must be an absolute Unity project root")
    root = Path(os.path.abspath(raw))
    if _is_link(root) or not root.is_dir():
        raise ValueError("unity_project must be an existing directory that is not a link")
    assets = _unlinked_below(root, ["Assets"])
    if not assets.is_dir():
        raise ValueError("unity_project is not a Unity project (Assets/ is missing)")
    return root


def _exclusive_target(value: Any, label: str) -> Path:
    raw = str(value or "")
    if not raw or not os.path.isabs(raw) or _ANY_CONTROL.search(raw):
        raise ValueError(f"{label} must be an absolute path")
    target = Path(os.path.abspath(raw))
    parent = target.parent
    if _is_link(parent) or not parent.is_dir():
        raise ValueError(f"{label} parent must be an existing directory that is not a link")
    if os.path.lexists(target):
        raise FileExistsError(f"{label} already exists; reports are never overwritten")
    return target


def _empty_directory(value: Any, label: str) -> Path:
    raw = str(value or "")
    if not raw or not os.path.isabs(raw) or _ANY_CONTROL.search(raw):
        raise ValueError(f"{label} must be an absolute path")
    folder = Path(os.path.abspath(raw))
    if _is_link(folder) or not folder.is_dir():
        raise ValueError(f"{label} must be an existing directory that is not a link")
    if any(folder.iterdir()):
        raise FileExistsError(f"{label} must be empty; screenshots are never overwritten")
    return folder


# ── Bridge identity and authentication ───────────────────────────────────────

def _normalize_identity_path(value: str) -> str:
    """Mirror Unity MCP Ghost's NormalizeIdentityPath (upper-cased on Windows)."""
    normalized = os.path.abspath(value).rstrip("\\/").replace("\\", "/")
    return normalized.upper() if os.name == "nt" else normalized


def project_id_for_path(root: Path | str) -> str:
    digest = hashlib.sha256(_normalize_identity_path(str(root)).encode("utf-8")).hexdigest()
    return f"unity-project-{digest[:24]}"


def _state_dir(root: Path) -> Path:
    return _unlinked_below(root, ["Library", "UnityMcpGhost"])


def _read_regular(
    path: Path,
    limit: int,
    label: str,
    missing_code: str = "BRIDGE_NOT_CONFIGURED",
    invalid_code: str = "BRIDGE_IDENTITY_INVALID",
) -> bytes:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise UnityBridgeError(missing_code, f"{label} is missing") from None
    if _is_link(path) or not stat.S_ISREG(info.st_mode):
        raise UnityBridgeError(invalid_code, f"{label} must be a regular file")
    if info.st_size > limit:
        raise UnityBridgeError(invalid_code, f"{label} exceeds {limit} bytes")
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise UnityBridgeError(invalid_code, f"{label} exceeds {limit} bytes")
    return data


@dataclass(frozen=True)
class BridgeIdentity:
    """What Unity advertises about its bridge (never the token itself)."""

    project_root: Path
    project_id: str
    editor_instance_id: str
    port: int
    package_version: str
    unity_version: str
    heartbeat_utc: str

    def public(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "editor_instance_id": self.editor_instance_id,
            "port": self.port,
            "package_version": self.package_version,
            "unity_version": self.unity_version,
            "heartbeat_utc": self.heartbeat_utc,
        }


def read_bridge_identity(root: Path) -> BridgeIdentity:
    try:
        directory = _state_dir(root)
    except ValueError as exc:
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", str(exc)) from None
    raw = _read_regular(directory / IDENTITY_FILE, MAX_IDENTITY_BYTES, "Library/UnityMcpGhost/bridge-identity.json")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", "bridge-identity.json is not valid JSON") from None
    if not isinstance(data, dict) or data.get("schemaVersion") != 1:
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", "unsupported bridge-identity.json schema")
    if data.get("projectId") != project_id_for_path(root):
        raise UnityBridgeError("BRIDGE_PROJECT_MISMATCH", "bridge-identity.json belongs to another project path")
    project_value = data.get("projectRoot")
    if not isinstance(project_value, str) or _normalize_identity_path(project_value) != _normalize_identity_path(str(root)):
        raise UnityBridgeError("BRIDGE_PROJECT_MISMATCH", "bridge-identity.json names another project root")
    editor = data.get("editorInstanceId")
    if not isinstance(editor, str) or not _EDITOR_ID.match(editor):
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", "bridge-identity.json has no valid editor instance ID")
    port = data.get("port")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", "bridge-identity.json has no valid port")
    if data.get("host") != LOOPBACK_HOST:
        raise UnityBridgeError("BRIDGE_NOT_LOOPBACK", "the bridge identity is not the literal loopback address")
    if data.get("running") is not True:
        raise UnityBridgeError(
            "BRIDGE_NOT_RUNNING",
            "the Unity bridge is not running for this project (open it in Unity with Unity MCP Ghost)",
        )
    token_file = data.get("tokenFile")
    expected_token = directory / TOKEN_FILE
    if not isinstance(token_file, str) or _normalize_identity_path(token_file) != _normalize_identity_path(str(expected_token)):
        raise UnityBridgeError(
            "BRIDGE_TOKEN_LOCATION",
            "bridge-identity.json names a token file outside the project's Library/UnityMcpGhost",
        )
    text = lambda key: str(data.get(key) or "")[:64]  # noqa: E731
    return BridgeIdentity(root, data["projectId"], editor, port, text("packageVersion"), text("unityVersion"), text("heartbeatUtc"))


def read_bridge_token(root: Path) -> str:
    try:
        directory = _state_dir(root)
    except ValueError as exc:
        raise UnityBridgeError("BRIDGE_IDENTITY_INVALID", str(exc)) from None
    raw = _read_regular(directory / TOKEN_FILE, MAX_TOKEN_BYTES, "Library/UnityMcpGhost/bridge-token.txt")
    try:
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        token = ""
    if not _TOKEN.match(token):
        raise UnityBridgeError("BRIDGE_TOKEN_INVALID", "bridge-token.txt does not hold a 256-bit hex token")
    return token


class _RefuseRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102, PLR0913
        raise UnityBridgeError("BRIDGE_REDIRECT_REFUSED", "the Unity bridge answered with a redirect")


class UnityBridgeClient:
    """Authenticated JSON-RPC client for one Unity MCP Ghost editor bridge."""

    def __init__(self, identity: BridgeIdentity, token: str, timeout: float = REQUEST_TIMEOUT) -> None:
        if not isinstance(token, str) or not _TOKEN.match(token):
            raise UnityBridgeError("BRIDGE_TOKEN_INVALID", "invalid bridge token")
        self.identity = identity
        self.__token = token
        self.timeout = float(timeout)
        # No proxy (the token must never leave loopback) and no redirects.
        self._opener = request.build_opener(request.ProxyHandler({}), _RefuseRedirect())
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def for_project(cls, root: Path, timeout: float = REQUEST_TIMEOUT) -> "UnityBridgeClient":
        identity = read_bridge_identity(root)
        return cls(identity, read_bridge_token(root), timeout)

    def __repr__(self) -> str:
        return f"UnityBridgeClient(port={self.identity.port}, project_id={self.identity.project_id!r})"

    @property
    def endpoint(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.identity.port}{BRIDGE_PATH}"

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        mutation: bool = False,
        key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        rid = f"ghostrigger-{uuid.uuid4()}"
        wait = float(timeout or self.timeout)
        payload = {
            "jsonrpc": "2.0",
            "id": rid,
            "requestId": rid,
            "method": method,
            "params": params or {},
            "deadlineUnixMs": int((time.time() + wait) * 1000),
            "idempotencyKey": (key or rid)[:128],
            "idempotent": not mutation,
            "projectId": self.identity.project_id,
            "editorInstanceId": self.identity.editor_instance_id,
        }
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.__token},
            method="POST",
        )
        self.calls.append((method, params or {}))
        try:
            with self._opener.open(req, timeout=wait + 5.0) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except UnityBridgeError:
            raise
        except error.HTTPError as exc:
            code = {401: "BRIDGE_AUTH_REFUSED", 403: "BRIDGE_FORBIDDEN", 429: "BRIDGE_RATE_LIMITED"}.get(
                exc.code, f"BRIDGE_HTTP_{exc.code}"
            )
            raise UnityBridgeError(code, f"Unity bridge refused {method} (HTTP {exc.code})") from None
        except (error.URLError, OSError) as exc:
            raise UnityBridgeError("BRIDGE_UNREACHABLE", f"Unity bridge request {method} failed ({type(exc).__name__})") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise UnityBridgeError("BRIDGE_RESPONSE_TOO_LARGE", f"Unity bridge answer to {method} exceeds the bound")
        text = raw.decode("utf-8", errors="replace")
        try:
            rpc = json.loads(text)
        except json.JSONDecodeError:
            try:
                rpc = json.loads(_CONTROL_CHARS.sub("", text))
            except json.JSONDecodeError:
                raise UnityBridgeError("BRIDGE_PROTOCOL", f"Unity bridge returned invalid JSON for {method}") from None
        if not isinstance(rpc, dict) or rpc.get("id") != rid:
            raise UnityBridgeError("BRIDGE_PROTOCOL", f"Unity bridge answered {method} for another request")
        if rpc.get("error"):
            err = rpc["error"] if isinstance(rpc["error"], dict) else {"message": str(rpc["error"])}
            data = err.get("data") if isinstance(err.get("data"), dict) else {}
            code = str(data.get("code") or f"RPC_{err.get('code', 'ERROR')}")[:64]
            raise UnityBridgeError(code, f"Unity bridge refused {method}: {str(err.get('message', ''))[:300]}")
        result = rpc.get("result")
        return result if isinstance(result, dict) else {"result": result}


def verify_bridge(client: Any, root: Path) -> dict[str, Any]:
    """Authenticated health must name this project path, project and editor."""
    identity = client.identity
    health = client.request("health")
    if health.get("ok") is not True or health.get("authenticated") is not True:
        raise UnityBridgeError("BRIDGE_IDENTITY_MISMATCH", "health did not confirm an authenticated bridge")
    if health.get("projectId") != identity.project_id or health.get("editorInstanceId") != identity.editor_instance_id:
        raise UnityBridgeError("BRIDGE_IDENTITY_MISMATCH", "the bridge answered for another project or editor instance")
    project_path = health.get("projectPath")
    if not isinstance(project_path, str) or _normalize_identity_path(project_path) != _normalize_identity_path(str(root)):
        raise UnityBridgeError("BRIDGE_IDENTITY_MISMATCH", "the bridge serves another project path")
    return health


def _setup_signature(setup: Any) -> list[tuple[str, bool, bool]]:
    if not isinstance(setup, list):
        return []
    rows = []
    for entry in setup:
        if isinstance(entry, dict):
            rows.append((str(entry.get("path") or "").lower(), bool(entry.get("isLoaded")), bool(entry.get("isActive"))))
    return rows


def editor_readiness(client: Any, health: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (readiness, raw snapshot). Reads only; blockers explain refusals."""
    snapshot = client.request("editor.snapshot")
    state = client.request("editor.get_state")
    setup = snapshot.get("sceneSetup")
    dirty = snapshot.get("dirtyScenes")
    blockers: list[str] = []
    if snapshot.get("ok") is not True or not isinstance(setup, list) or not isinstance(dirty, list):
        blockers.append("snapshot-unavailable: the bridge did not report the scene setup")
        setup, dirty = [], []
    if dirty or health.get("activeSceneDirty") or state.get("activeSceneDirty"):
        blockers.append("scene-dirty: save or discard the open scenes in Unity first")
    if not setup or any(not isinstance(entry, dict) or not str(entry.get("path") or "") for entry in setup):
        blockers.append("untitled-scene: save the untitled scene in Unity first (its setup cannot be restored)")
    if state.get("isPlaying") or state.get("isPaused") or health.get("isPlaying"):
        blockers.append("play-mode: stop Play mode in Unity first")
    if state.get("isCompiling") or health.get("isCompiling"):
        blockers.append("compiling: wait for Unity to finish compiling")
    if state.get("isUpdating"):
        blockers.append("updating: wait for Unity to finish importing")
    queue = health.get("durableQueue") if isinstance(health.get("durableQueue"), dict) else {}
    try:
        busy = int(queue.get("pendingCount") or 0) + int(queue.get("processingCount") or 0)
    except (TypeError, ValueError):
        busy = 0
    if busy:
        blockers.append("bridge-busy: the Unity bridge queue has pending requests")
    readiness = {
        "blockers": blockers,
        "scene_setup": [
            {"path": str(e.get("path") or ""), "isLoaded": bool(e.get("isLoaded")), "isActive": bool(e.get("isActive"))}
            for e in setup
            if isinstance(e, dict)
        ][:64],
        "dirty_scenes": [str(item)[:512] for item in dirty][:64],
        "active_scene_path": str(state.get("activeScenePath") or health.get("activeScenePath") or "")[:512],
        "is_playing": bool(state.get("isPlaying")),
        "is_compiling": bool(state.get("isCompiling") or health.get("isCompiling")),
        "is_updating": bool(state.get("isUpdating")),
    }
    return readiness, snapshot


# ── Smoke helpers ─────────────────────────────────────────────────────────────

def metadata_path_for_asset(unity_project: Path, asset_path: str) -> Path:
    """Return the GhostRigger sidecar path for a Unity asset path."""
    normalized = asset_path.replace("\\", "/")
    if normalized.lower().endswith(".fbx"):
        normalized = normalized[:-4] + ".ghostrigger.json"
    return unity_project / normalized


def _read_sidecar(root: Path, asset_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = asset_path[:-4] + ".ghostrigger.json"  # asset_path is validated as Assets/...fbx
    path = _unlinked_below(root, relative.split("/"))
    info: dict[str, Any] = {"path": relative, "present": False}
    if not os.path.lexists(path):
        return {}, info
    data = _read_regular(path, MAX_SIDECAR_BYTES, relative, missing_code="SIDECAR_MISSING", invalid_code="SIDECAR_INVALID")
    info.update(present=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    try:
        sidecar = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"{relative} is not valid JSON") from None
    return (sidecar if isinstance(sidecar, dict) else {}), info


def choose_preferred_clip(animations: list[Any], preferred: tuple[str, ...] = PREFERRED_MALAK_CLIPS) -> str | None:
    """Choose the clip the main-menu bootstrap should prefer."""
    names = [str(item.get("name", item) if isinstance(item, dict) else item).lstrip("|") for item in animations]
    by_lower = {name.lower(): name for name in names if name}
    for candidate in preferred:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return names[0] if names else None


def _flatten_hierarchy(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    stack = list(reversed(nodes))
    while stack:
        node = stack.pop()
        flat.append(node)
        children = node.get("children") or []
        if isinstance(children, list):
            stack.extend(reversed([child for child in children if isinstance(child, dict)]))
    return flat


def _subtree_from_name(nodes: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    for node in _flatten_hierarchy(nodes):
        if str(node.get("name", "")) == name:
            return _flatten_hierarchy([node])
    return []


def _component_types(components_payload: dict[str, Any]) -> list[str]:
    components = components_payload.get("components") or []
    if not isinstance(components, list):
        return []
    return [str(component.get("type", "")) for component in components if isinstance(component, dict)]


def collect_malak_unity_summary(
    client: Any,
    *,
    scene_path: str,
    asset_path: str,
    instance_name: str,
    expected_clips: list[str],
) -> dict[str, Any]:
    """Collect Unity-side facts for the visible model instance (reads only)."""
    hierarchy = client.request("scene.get_hierarchy", {"includeInactive": True, "maxDepth": 16})
    nodes = hierarchy.get("roots") or []
    subtree = _subtree_from_name(nodes if isinstance(nodes, list) else [], instance_name)
    truncated = len(subtree) > MAX_SUBTREE_NODES
    subtree = subtree[:MAX_SUBTREE_NODES]

    renderers: list[dict[str, Any]] = []
    component_types: dict[str, int] = {}
    for node in subtree:
        instance_id = node.get("instanceId")
        if not isinstance(instance_id, int) or isinstance(instance_id, bool):
            continue
        components = client.request("component.get", {"instanceId": instance_id})
        for ctype in _component_types(components):
            component_types[ctype] = component_types.get(ctype, 0) + 1
            if ctype.endswith("SkinnedMeshRenderer"):
                renderers.append({"type": "SkinnedMeshRenderer", "materialCount": 1})
            elif ctype.endswith("MeshRenderer"):
                renderers.append({"type": "MeshRenderer", "materialCount": 1})

    direct_matches = client.request(
        "gameobject.find",
        {"name": instance_name, "includeInactive": True},
    ).get("matches") or []
    scene_matches = [
        match for match in direct_matches
        if isinstance(match, dict) and match.get("scenePath") == scene_path
    ]

    return {
        "asset_path": asset_path,
        "clips": [{"name": clip} for clip in expected_clips],
        "renderers": renderers,
        "component_types": component_types,
        "malak_instance_count": len(scene_matches),
        "malak_node_count": len(subtree),
        "node_scan_truncated": truncated,
    }


def _bridge_screenshot(root: Path, response: dict[str, Any]) -> Path | None:
    """Only a PNG directly inside the bridge's own screenshots folder is read."""
    raw = response.get("path")
    if not isinstance(raw, str) or not raw or _ANY_CONTROL.search(raw) or not os.path.isabs(raw):
        return None
    candidate = Path(os.path.abspath(raw))
    try:
        folder = _unlinked_below(root, ["Library", "UnityMcpGhost", "screenshots"])
    except ValueError:
        return None
    if os.path.normcase(str(candidate.parent)) != os.path.normcase(str(folder)) or candidate.suffix.lower() != ".png":
        return None
    try:
        info = os.lstat(candidate)
    except FileNotFoundError:
        return None
    if _is_link(candidate) or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SCREENSHOT_BYTES:
        return None
    return candidate


def _wait_for_file(path: str, timeout: float = 5.0) -> bool:
    if not path:
        return False
    target = Path(path)
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if target.exists():
            return True
        time.sleep(0.1)
    return target.exists()


def compare_screenshots(before_path: str, after_path: str) -> dict[str, Any]:
    """Compare screenshots and report visible movement when Pillow is available."""
    if not _wait_for_file(before_path) or not _wait_for_file(after_path):
        return {"checked": False, "reason": "screenshot_missing"}
    try:
        from PIL import Image, ImageChops, ImageStat  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {"checked": False, "reason": f"pillow_unavailable: {type(exc).__name__}"}

    with Image.open(before_path) as first, Image.open(after_path) as second:
        before = first.convert("RGB")
        after = second.convert("RGB")
    if before.size != after.size:
        return {"checked": False, "reason": "screenshot_size_mismatch"}

    diff = ImageChops.difference(before, after)
    stat_ = ImageStat.Stat(diff)
    mean_delta = sum(stat_.mean) / len(stat_.mean)
    extrema = diff.getextrema()
    changed_channels = sum(1 for channel in extrema for value in channel if value)
    return {
        "checked": True,
        "mean_delta": mean_delta,
        "changed_channels": changed_channels,
        "visible_delta": mean_delta > 0.25,
    }


def _copy_exclusive(source: Path, folder: Path, name: str) -> dict[str, Any]:
    data = source.read_bytes()
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise ValueError("screenshot exceeds bound")
    target = folder / name
    with open(target, "xb") as handle:
        handle.write(data)
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _capture(client: Any, root: Path, key: str, label: str, width: int, height: int, warnings: list) -> dict[str, Any]:
    try:
        response = client.request(
            "screenshot.capture",
            {"label": label, "width": width, "height": height, "superSize": 1, "mode": "auto"},
            mutation=True,
            key=key,
        )
    except UnityBridgeError as exc:
        warnings.append({"code": "screenshot_failed", "message": f"{label}: {exc.code}"})
        return {"ok": False, "code": exc.code}
    source = _bridge_screenshot(root, response)
    if source is None:
        warnings.append({"code": "screenshot_unreadable", "message": f"{label}: not in Library/UnityMcpGhost/screenshots"})
    return {
        "ok": response.get("ok") is not False and source is not None,
        "camera": str(response.get("camera") or "")[:128],
        "width": response.get("width"),
        "height": response.get("height"),
        "source": source,
    }


def _wait_until_idle(client: Any, timeout: float = IDLE_WAIT_SECONDS) -> dict[str, Any]:
    """After a refresh Unity may import, compile and reload its domain."""
    deadline = time.monotonic() + timeout
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        try:
            state = client.request("editor.get_state", timeout=15.0)
            if not state.get("isCompiling") and not state.get("isUpdating"):
                return {"idle": True, "polls": polls}
        except UnityBridgeError as exc:
            if exc.code != "BRIDGE_UNREACHABLE":  # a domain reload restarts the listener
                raise
        time.sleep(0.5)
    raise UnityBridgeError("EDITOR_BUSY_TIMEOUT", "Unity did not finish importing or compiling in time")


def _restore(client: Any, snapshot: dict[str, Any], key: str, expected: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": True, "ok": False, "verified": False}
    try:
        answer = client.request(
            "editor.restore_snapshot", {"snapshot": snapshot}, mutation=True, key=key, timeout=120.0
        )
        if answer.get("ok") is False:
            err = answer.get("error") if isinstance(answer.get("error"), dict) else {}
            result["code"] = str(err.get("code") or "RESTORE_REFUSED")[:64]
            return result
        after = client.request("editor.snapshot")
        result["ok"] = True
        result["verified"] = _setup_signature(after.get("sceneSetup")) == _setup_signature(expected)
    except Exception as exc:  # noqa: BLE001 - recorded for the reviewer, never raised past restore
        result["code"] = str(getattr(exc, "code", type(exc).__name__))[:64]
    return result


def _diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    items = result.get("diagnostics") if isinstance(result.get("diagnostics"), list) else []
    slim = []
    for item in items[:50]:
        if isinstance(item, dict):
            slim.append({k: (v[:500] if isinstance(v, str) else v) for k, v in list(item.items())[:12]})
        else:
            slim.append(str(item)[:500])
    return {"count": len(items), "diagnostics": slim}


def _plan(refresh_assets: bool) -> list[str]:
    steps = ["editor.snapshot (restore point)", "scene.open (single)"]
    if refresh_assets:
        steps.append("asset.refresh + wait for import/compile")
    steps += [
        "asset.find",
        "screenshot.capture x2 (Library/UnityMcpGhost/screenshots)",
        "scene.get_hierarchy / component.get / gameobject.find",
        "compile + console diagnostics",
        "editor.restore_snapshot (prior scene setup)",
    ]
    return steps


def run_malak_main_menu_smoke(
    *,
    unity_project: Path,
    client: Any | None = None,
    host: str | None = None,
    port: int | None = None,
    scene_path: str = DEFAULT_SCENE_PATH,
    asset_path: str = DEFAULT_ASSET_PATH,
    instance_name: str = DEFAULT_INSTANCE_NAME,
    output_path: Path | None = None,
    screenshot_delay: float = 1.0,
    screenshot_dir: Path | None = None,
    screenshot_width: int = 640,
    screenshot_height: int = 360,
    refresh_assets: bool = True,
    dry_run: bool = False,
    operation_id: str | None = None,
    expected_port: int | None = None,
    expected_editor_instance_id: str | None = None,
) -> dict[str, Any]:
    """Run the smoke (or, with dry_run, only report readiness) and write a report.

    ``host``/``port`` are accepted only as assertions for legacy callers: the
    endpoint always comes from the project's bridge identity.
    ``expected_port``/``expected_editor_instance_id`` bind the call to the
    identity a caller already verified (Kaigen checks the listener process on
    that port): a different identity is refused before the token is sent.
    """
    if expected_port is not None and (type(expected_port) is not int or not 1024 <= expected_port <= 65535):
        raise ValueError("expected_port must be an integer port from 1024 to 65535")
    if expected_editor_instance_id is not None and (
        not isinstance(expected_editor_instance_id, str) or not _EDITOR_ID.match(expected_editor_instance_id)
    ):
        raise ValueError("expected_editor_instance_id must be 32 lowercase hex characters")
    root = project_root(unity_project)
    scene_path = validate_asset_path(scene_path, ".unity", "scene_path")
    asset_path = validate_asset_path(asset_path, ".fbx", "asset_path")
    instance_name = validate_instance_name(instance_name)
    delay = _bounded(screenshot_delay, 0.0, MAX_SCREENSHOT_DELAY, "screenshot_delay")
    width = _bounded(screenshot_width, *SCREENSHOT_WIDTH, "screenshot_width", integer=True)
    height = _bounded(screenshot_height, *SCREENSHOT_HEIGHT, "screenshot_height", integer=True)
    if not isinstance(refresh_assets, bool) or not isinstance(dry_run, bool):
        raise ValueError("refresh_assets and dry_run must be booleans")
    if operation_id is not None and (not isinstance(operation_id, str) or not _OPERATION_ID.match(operation_id)):
        raise ValueError("operation_id must be 1-64 characters of [A-Za-z0-9-]")
    output = _exclusive_target(output_path, "output_path") if output_path is not None and not dry_run else None
    shots_dir = _empty_directory(screenshot_dir, "screenshot_dir") if screenshot_dir is not None and not dry_run else None

    rpc = client or UnityBridgeClient.for_project(root)
    identity = rpc.identity
    if host is not None and host != LOOPBACK_HOST:
        raise ValueError("host is fixed to the bridge identity's loopback endpoint")
    if port is not None and port != identity.port:
        raise ValueError("port must match the project's bridge identity")
    if (expected_port is not None and identity.port != expected_port) or (
        expected_editor_instance_id is not None and identity.editor_instance_id != expected_editor_instance_id
    ):
        raise UnityBridgeError(
            "BRIDGE_IDENTITY_CHANGED",
            "the project's bridge identity differs from the one the caller verified; nothing was sent",
        )

    health = verify_bridge(rpc, root)
    readiness, snapshot = editor_readiness(rpc, health)
    sidecar, sidecar_info = _read_sidecar(root, asset_path)
    animations = sidecar.get("animations", [])
    expected_clips = [str(name)[:128] for name in (animations if isinstance(animations, list) else []) if name][:MAX_CLIPS]
    selected_clip = choose_preferred_clip(expected_clips)
    sidecar_info["clips"] = len(expected_clips)
    bridge = {
        **identity.public(),
        "protocol_version": health.get("protocolVersion"),
        "health_unity_version": str(health.get("unityVersion") or "")[:64],
        "health_package_version": str(health.get("version") or "")[:64],
    }
    base = {
        "schema_version": 2,
        "tool": TOOL_ID,
        "unity_project": str(root),
        "scene_path": scene_path,
        "asset_path": asset_path,
        "instance_name": instance_name,
        "sidecar": sidecar_info,
        "selected_clip": selected_clip,
        "bridge": bridge,
    }
    if dry_run:
        blockers = readiness["blockers"]
        return {
            **base,
            "dry_run": True,
            # No "ok" key: a blocked editor is a valid readiness answer, not a failed call.
            "status": "blocked" if blockers else "ready",
            "blockers": blockers,
            "editor": readiness,
            "plan": _plan(refresh_assets),
            "screenshot": {"width": width, "height": height, "delay": delay},
        }
    if readiness["blockers"]:
        raise UnityBridgeError("EDITOR_NOT_READY", "; ".join(readiness["blockers"]))

    prefix = operation_id or f"ghostrigger-{uuid.uuid4()}"
    key = lambda step: f"{prefix}-{step}"[:128]  # noqa: E731
    label = f"ghostrigger-{(operation_id or uuid.uuid4().hex).split('-')[0][:16]}"
    steps: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    shots: dict[str, Any] = {}
    unity_summary: dict[str, Any] = {}
    compile_diags: dict[str, Any] = {}
    console_diags: dict[str, Any] = {}
    screenshot_delta: dict[str, Any] = {"checked": False, "reason": "not_run"}
    asset_paths: list[str] = []
    aborted: dict[str, str] | None = None
    restore: dict[str, Any] = {"attempted": False}
    current = "scene.open"
    try:
        opened = rpc.request("scene.open", {"path": scene_path, "mode": "single"}, mutation=True, key=key("scene-open"))
        steps.append({"step": "scene.open", "result": opened.get("scene", opened)})
        if refresh_assets:
            current = "asset.refresh"
            steps.append({"step": "asset.refresh", "result": rpc.request("asset.refresh", {}, mutation=True, key=key("asset-refresh"), timeout=REFRESH_TIMEOUT)})
            current = "editor.wait_idle"
            steps.append({"step": "editor.wait_idle", "result": _wait_until_idle(rpc)})
        current = "asset.find"
        assets = rpc.request("asset.find", {"query": Path(asset_path).stem, "limit": 20})
        asset_paths = [str(a.get("path", "")) for a in assets.get("assets", []) if isinstance(a, dict)][:20]
        steps.append({"step": "asset.find", "result": {"paths": asset_paths}})
        current = "screenshot.capture"
        before = _capture(rpc, root, key("shot-before"), f"{label}-before", width, height, warnings)
        time.sleep(delay)
        after = _capture(rpc, root, key("shot-after"), f"{label}-after", width, height, warnings)
        if before.get("source") and after.get("source"):
            screenshot_delta = compare_screenshots(str(before["source"]), str(after["source"]))
        else:
            screenshot_delta = {"checked": False, "reason": "screenshot_missing"}
        for name, shot in (("before", before), ("after", after)):
            source = shot.pop("source", None)
            if source is not None and shots_dir is not None:
                shot["copy"] = _copy_exclusive(source, shots_dir, f"{name}.png")
            shot["bridge_file"] = source.name if source is not None else None
            shots[name] = shot
        current = "summary"
        unity_summary = collect_malak_unity_summary(
            rpc,
            scene_path=scene_path,
            asset_path=asset_path,
            instance_name=instance_name,
            expected_clips=[selected_clip] if selected_clip else [],
        )
        current = "diagnostics"
        compile_diags = _diagnostics(rpc.request("compile.diagnostics_get", {"severity": "error", "limit": 50}))
        console_diags = _diagnostics(rpc.request("console.diagnostics_get", {"severity": "error", "limit": 50}))
    except Exception as exc:  # noqa: BLE001 - recorded, the prior scene setup is still restored
        aborted = {"step": current, "code": str(getattr(exc, "code", type(exc).__name__))[:64], "message": str(exc)[:300]}
    finally:
        restore = _restore(rpc, snapshot, key("restore"), readiness["scene_setup"])

    components = unity_summary.get("component_types", {})
    if aborted:
        issues.append({"code": "smoke_aborted", "message": f"{aborted['step']}: {aborted['code']}"})
    else:
        if asset_path not in asset_paths:
            issues.append({"code": "asset_not_found", "message": f"Unity could not find {asset_path}."})
        if not selected_clip:
            issues.append({"code": "no_preferred_clip", "message": "GhostRigger sidecar has no preferred idle/talk clip."})
        if unity_summary.get("malak_instance_count", 0) < 1:
            issues.append({"code": "menu_instance_missing", "message": f"{instance_name} is not present in the scene."})
        if components.get("UnityEngine.Animator", 0) < 1:
            issues.append({"code": "animator_missing", "message": "The scene instance has no Animator component."})
        if compile_diags.get("count"):
            issues.append({"code": "compile_errors", "message": "Unity compile diagnostics contain errors."})
        if console_diags.get("count"):
            issues.append({"code": "console_errors", "message": "Unity console diagnostics contain errors."})
    if not restore.get("ok"):
        issues.append({"code": "restore_failed", "message": f"Prior scene setup was not restored ({restore.get('code', 'unknown')})."})
    elif not restore.get("verified"):
        issues.append({"code": "restore_unverified", "message": "The scene setup after restore differs from before the smoke."})

    if screenshot_delta.get("checked") and not screenshot_delta.get("visible_delta"):
        warnings.append({
            "code": "no_visible_delta",
            "message": "Screenshots did not show visible movement; edit-mode Animators do not play without a sample/play step.",
        })
    elif not screenshot_delta.get("checked"):
        warnings.append({"code": "screenshot_delta_unchecked", "message": str(screenshot_delta.get("reason", "unchecked"))})
    renderer_types = {str(renderer.get("type", "")) for renderer in unity_summary.get("renderers", [])}
    if renderer_types and "SkinnedMeshRenderer" not in renderer_types:
        warnings.append({
            "code": "missing_skinned_renderer",
            "message": "The instance imported as MeshRenderer pieces; Unity skin deformation is not proven yet.",
        })
    if unity_summary.get("node_scan_truncated"):
        warnings.append({"code": "node_scan_truncated", "message": f"Only the first {MAX_SUBTREE_NODES} nodes were inspected."})

    report = {
        **base,
        "dry_run": False,
        "status": "error" if issues else ("warning" if warnings else "ok"),
        "ok": not issues,
        "operation_id": operation_id,
        "editor_before": readiness,
        "steps": steps,
        "unity_summary": unity_summary,
        "screenshots": shots,
        "screenshot_delta": screenshot_delta,
        "compile_diagnostics": compile_diags,
        "console_diagnostics": console_diags,
        "warnings": warnings,
        "issues": issues,
        "restore": restore,
        "aborted": aborted,
    }
    if output is not None:
        with open(output, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
