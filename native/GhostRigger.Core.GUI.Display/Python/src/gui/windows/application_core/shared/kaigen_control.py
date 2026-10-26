"""Ghost Studio window side of the Kaigen GUI control channel.

Started from ``_start_ipc_server`` only when ``GHOSTSTUDIO_KAIGEN_CONTROL_PATH``
is set. It answers three authenticated requests (see
``src.ipc.kaigen_gui_control``) on the Qt thread:

* ``state``: a bounded, revisioned view of the scene, the loaded model (with
  the SHA-256 of the MDL/MDX bytes it was parsed from), busy markers, the
  configured game directories and GUI readiness;
* ``open_model``: open one game model, by resref, into a new untitled scene.
  It refuses while the scene has unsaved changes or a load is running, when
  the GUI state moved since Kaigen's preview, when the GUI's configured
  directory for that game is not the previewed installation, and when the
  bytes the GUI resolves differ from the previewed MDL/MDX hashes. The model
  is parsed from exactly the hashed bytes, before the scene is touched; the
  previous scene's saved file is never modified;
* ``capture``: return the viewport as PNG bytes (no caller-chosen path).

No dialog is ever shown for these requests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from src.ipc.kaigen_gui_control import (
    CONTROL_PATH_ENV,
    ControlRefusal,
    KaigenGuiControlServer,
    file_sha256,
)

log = logging.getLogger(__name__)

MODULE_SHA256 = file_sha256(__file__)
MAX_SCENE_OBJECTS = 256
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
STATE_TIMEOUT = 10.0
OPEN_TIMEOUT = 120.0
CAPTURE_TIMEOUT = 20.0


def _sha256(data: object) -> str | None:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(data)).hexdigest()
    return None


def _same_dir(left: str, right: str) -> bool:
    try:
        a = os.path.normcase(os.path.realpath(left))
        b = os.path.normcase(os.path.realpath(right))
    except (OSError, ValueError):
        return False
    return bool(left and right) and a == b


def _run_on_gui_thread(fn: Callable[[], dict], timeout: float) -> dict:
    from src.adapters.qt_ipc.threading import marshal_to_gui_thread  # noqa: PLC0415

    done = threading.Event()
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc
        finally:
            done.set()

    if not marshal_to_gui_thread(runner):
        raise ControlRefusal("gui-thread-unavailable", "The Ghost Studio GUI thread is not running.")
    if not done.wait(timeout):
        raise ControlRefusal(
            "gui-timeout",
            "The Ghost Studio GUI did not finish in time; read the state before retrying.",
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def _source_ref(obj: object) -> dict[str, str]:
    ref = getattr(obj, "source_ref", None)
    if ref is None:
        return {}
    return {
        key: str(getattr(ref, key, "") or "")[:512]
        for key in ("game", "resref", "source_path", "source_archive")
        if getattr(ref, key, "")
    }


def model_identity(window: object) -> dict[str, Any] | None:
    model = getattr(window, "_current_model", None)
    if model is None:
        return None
    try:
        node_count = int(model.node_count()) if hasattr(model, "node_count") else None
    except Exception:  # noqa: BLE001
        node_count = None
    try:
        mesh_count = len(model.mesh_nodes()) if hasattr(model, "mesh_nodes") else None
    except Exception:  # noqa: BLE001
        mesh_count = None
    return {
        "label": str(getattr(window, "_model_path", "") or "")[:512],
        "name": str(getattr(model, "name", "") or "")[:128],
        "game": str(getattr(model, "_gr_source_game", "") or getattr(window, "_current_game", "") or ""),
        "resref": str(getattr(model, "_gr_source_resref", "") or ""),
        "mdlSha256": _sha256(getattr(model, "_gr_source_mdl_bytes", None)),
        "mdxSha256": _sha256(getattr(model, "_gr_source_mdx_bytes", None)),
        "nodeCount": node_count,
        "meshCount": mesh_count,
    }


def _busy(window: object) -> bool:
    worker = getattr(window, "_model_worker_is_running", None)
    running = bool(worker()) if callable(worker) else False
    return running or bool(int(getattr(window, "_pending_gpu_upload_total", 0) or 0))


def gui_state(window: object) -> dict[str, Any]:
    manager = getattr(window, "scene_manager", None)
    scene = getattr(manager, "active_scene", None)
    objects = list(getattr(scene, "objects", []) or [])
    core = {
        "scene": {
            "name": str(getattr(scene, "display_name", getattr(scene, "name", "")) or "")[:256],
            "path": str(getattr(scene, "path", "") or "")[:1024],
            "game": str(getattr(scene, "game", "") or ""),
            "dirty": bool(manager.is_dirty()) if manager is not None else False,
            "objectCount": len(objects),
            "objects": [
                {
                    "id": str(getattr(obj, "id", "") or "")[:128],
                    "name": str(getattr(obj, "name", "") or "")[:256],
                    "type": str(getattr(obj, "object_type", "") or "")[:32],
                    "source": _source_ref(obj),
                }
                for obj in objects[:MAX_SCENE_OBJECTS]
            ],
        },
        "model": model_identity(window),
        "busy": _busy(window),
    }
    revision = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    dirs = getattr(window, "_configured_game_dirs", None)
    k1_dir, k2_dir = dirs() if callable(dirs) else ("", "")
    readiness = None
    health = getattr(window, "_ipc_spatial_health", None)
    if callable(health):
        try:
            payload = health()
            readiness = {
                key: payload.get(key)
                for key in ("ready", "windowVisible", "windowMinimized", "reason")
            }
        except Exception:  # noqa: BLE001
            readiness = None
    return {
        **core,
        "revision": revision,
        "gameDirs": {"K1": str(k1_dir or ""), "K2": str(k2_dir or "")},
        "gui": readiness,
    }


def _default_resolver(game_dir: str, game: str, resref: str) -> tuple[bytes, bytes]:
    from src.core.assets.resource_manager import ResourceManager  # noqa: PLC0415

    manager = ResourceManager()
    if game == "K2":
        manager.set_k2_dir(game_dir)
    else:
        manager.set_k1_dir(game_dir)
    mdl = manager.get_mdl(resref, game)
    if not mdl:
        raise ControlRefusal("model-not-found", f"{game}:{resref}.mdl is not in the configured installation.")
    return bytes(mdl), bytes(manager.get_mdx(resref, game) or b"")


def _default_parser(mdl: bytes, mdx: bytes, game: str) -> Any:
    from src.core.game.kotor_loader import load_model_from_bytes  # noqa: PLC0415
    from src.core.geometry.model_data import GameVersion  # noqa: PLC0415
    from src.gui.windows.application_core.application_core_lib.functions.geometry import (  # noqa: PLC0415
        _prebuild_gpu_mesh_data_for_model,
    )

    version = GameVersion.K2 if game == "K2" else GameVersion.K1
    model = load_model_from_bytes(mdl, mdx, game_version=version)
    if model is None:
        raise ControlRefusal("model-parse-failed", "The model bytes could not be parsed.")
    model.game_version = version
    _prebuild_gpu_mesh_data_for_model(model)
    return model


def open_game_model(
    window: object,
    request: Mapping[str, Any],
    *,
    resolver: Callable[[str, str, str], tuple[bytes, bytes]] = _default_resolver,
    parser: Callable[[bytes, bytes, str], Any] = _default_parser,
) -> dict[str, Any]:
    game, resref = request["game"], request["resref"]
    before = gui_state(window)
    if before["busy"]:
        raise ControlRefusal("busy", "A model load is running in Ghost Studio.")
    if before["scene"]["dirty"]:
        raise ControlRefusal(
            "scene-dirty",
            "The current scene has unsaved changes; save or discard them in Ghost Studio first.",
        )
    if before["revision"] != request["expectedGuiRevision"]:
        raise ControlRefusal("gui-drift", "Ghost Studio's state changed since the preview.", observed=before["revision"])
    configured = before["gameDirs"].get(game, "")
    if not _same_dir(configured, request["gameDir"]):
        raise ControlRefusal(
            "game-dir-mismatch",
            f"Ghost Studio's configured {game} directory is not the previewed installation.",
        )
    mdl, mdx = resolver(configured, game, resref)
    observed = {"mdlSha256": _sha256(mdl), "mdxSha256": _sha256(mdx)}
    if observed["mdlSha256"] != request["mdlSha256"] or observed["mdxSha256"] != request["mdxSha256"]:
        raise ControlRefusal(
            "source-mismatch",
            "Ghost Studio resolves different model bytes than the preview.",
            **observed,
        )
    model = parser(mdl, mdx, game)
    model._gr_source_mdl_bytes = mdl
    model._gr_source_mdx_bytes = mdx
    model._gr_source_resref = resref
    model._gr_source_game = game
    # A new untitled scene: the previous scene's file keeps its content even
    # if the user later saves (the replaced scene had no unsaved changes).
    window._create_new_scene_from_ipc(game)
    window._pending_scene_import_action = "add"
    window._pending_scene_import_placement = "origin"
    window._current_game = game
    window._on_model_loaded(model, f"{game}:{resref}", "")
    after = gui_state(window)
    window._log(f"Kaigen opened {game}:{resref} (request {request['requestId']}).", "info")
    return {"opened": True, "requestId": request["requestId"], "before": before["revision"], "state": after}


def capture_viewport_png(window: object) -> dict[str, Any]:
    from PySide6 import QtCore  # noqa: PLC0415

    state = gui_state(window)
    gui = state.get("gui") or {}
    if not gui.get("windowVisible") or gui.get("windowMinimized"):
        raise ControlRefusal("capture-unavailable", "The Ghost Studio window is not visible.")
    viewport = getattr(window, "viewport", None)
    canvas = getattr(viewport, "canvas", None)
    if canvas is None:
        raise ControlRefusal("capture-unavailable", "The viewport canvas is not available.")
    pixmap = canvas.grab()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap.save(buffer, "PNG"):
        raise ControlRefusal("capture-failed", "The viewport could not be encoded as PNG.")
    data = bytes(buffer.data())
    if not data or len(data) > MAX_CAPTURE_BYTES:
        raise ControlRefusal("capture-failed", "The viewport capture is empty or too large.")
    return {
        "png": base64.b64encode(data).decode("ascii"),
        "pngSha256": hashlib.sha256(data).hexdigest(),
        "width": int(pixmap.width()),
        "height": int(pixmap.height()),
        "revision": state["revision"],
    }


def start_kaigen_gui_control(
    window: object, environ: Mapping[str, str] | None = None
) -> KaigenGuiControlServer | None:
    values = os.environ if environ is None else environ
    text = str(values.get(CONTROL_PATH_ENV) or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        log.warning("Kaigen GUI control disabled: %s must be absolute", CONTROL_PATH_ENV)
        return None
    handlers = {
        "state": lambda _payload: _run_on_gui_thread(lambda: {"state": gui_state(window)}, STATE_TIMEOUT),
        "open_model": lambda payload: _run_on_gui_thread(lambda: open_game_model(window, payload), OPEN_TIMEOUT),
        "capture": lambda _payload: _run_on_gui_thread(lambda: capture_viewport_png(window), CAPTURE_TIMEOUT),
    }
    server = KaigenGuiControlServer(
        path,
        handlers,
        source_identity={
            "gui/windows/application_core/shared/kaigen_control.py": MODULE_SHA256,
        },
    )
    server.start()
    log.info("Kaigen GUI control listening on 127.0.0.1:%d", server.port)
    return server
