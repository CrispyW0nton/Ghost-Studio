"""Kaigen-authenticated control channel for the running Ghost Studio GUI.

Opt-in: the GUI starts this listener only when ``GHOSTSTUDIO_KAIGEN_CONTROL_PATH``
names an absolute descriptor path (Kaigen Studio's launcher sets it). The
legacy port-7001 IPC stays as it was; this channel is separate and narrow:

* loopback HTTP on an ephemeral port, bound to 127.0.0.1 only;
* a fresh 256-bit secret per process, published only in the descriptor file;
* every request is HMAC-SHA256 signed over method, path, session, timestamp,
  nonce and body hash (30 s skew, bounded nonce replay cache, exact Host);
* every authenticated response is signed over status, session, the request
  nonce and body hash, so the client can prove the listener holds the secret;
* exactly three fixed routes (state, open-model, capture) with strict payloads.

GUI work is delegated to handler callbacks supplied by the window adapter,
which runs them on the Qt thread. This module imports no Qt.
"""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

DOMAIN = "kaigen-ghoststudio-gui/v1"
DESCRIPTOR_SCHEMA = "kaigen-ghoststudio-gui-session/v1"
CONTROL_PATH_ENV = "GHOSTSTUDIO_KAIGEN_CONTROL_PATH"
HEADER_SESSION = "X-Kaigen-Session"
HEADER_TIMESTAMP = "X-Kaigen-Timestamp"
HEADER_NONCE = "X-Kaigen-Nonce"
HEADER_SIGNATURE = "X-Kaigen-Signature"
HEADER_RESPONSE_SIGNATURE = "X-Kaigen-Response-Signature"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 24 * 1024 * 1024
MAX_CLOCK_SKEW_SECONDS = 30
REPLAY_CAPACITY = 4096
ROUTES: Mapping[str, str] = {
    "/kaigen/v1/state": "state",
    "/kaigen/v1/open-model": "open_model",
    "/kaigen/v1/capture": "capture",
}
_RESREF_RE = re.compile(r"^[a-z0-9_-]{1,16}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SESSION_RE = re.compile(r"^[0-9a-f]{32}$")


def file_sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Bytes of this module as imported; the GUI reports it so Kaigen can check the
# running channel against its qualified pin.
MODULE_SHA256 = file_sha256(__file__)


class ControlRefusal(Exception):
    """A stable, non-secret refusal returned to the client (HTTP 409)."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _canonical(parts: list[str]) -> bytes:
    return "\n".join([DOMAIN, *parts]).encode("utf-8")


def request_signature(
    secret: bytes,
    *,
    method: str,
    path: str,
    session: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> str:
    message = _canonical(
        ["REQ", method, path, session, timestamp, nonce, hashlib.sha256(body).hexdigest()]
    )
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def response_signature(
    secret: bytes, *, status: int, session: str, nonce: str, body: bytes
) -> str:
    message = _canonical(
        ["RES", str(int(status)), session, nonce, hashlib.sha256(body).hexdigest()]
    )
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _strict(payload: object, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or key not in allowed for key in payload
    ):
        raise ValueError("payload has unknown or invalid fields")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"payload lacks {sorted(missing)[0]}")
    return dict(payload)


def validate_state(payload: object) -> dict[str, Any]:
    return _strict(payload, set(), set())


def validate_capture(payload: object) -> dict[str, Any]:
    data = _strict(payload, {"captureId"}, {"captureId"})
    if not isinstance(data["captureId"], str) or not _UUID_RE.fullmatch(data["captureId"]):
        raise ValueError("captureId must be a lower-case UUID")
    return data


def validate_open_model(payload: object) -> dict[str, Any]:
    keys = {
        "requestId",
        "resref",
        "game",
        "gameDir",
        "mdlSha256",
        "mdxSha256",
        "expectedGuiRevision",
    }
    data = _strict(payload, keys, keys)
    if not isinstance(data["requestId"], str) or not _UUID_RE.fullmatch(data["requestId"]):
        raise ValueError("requestId must be a lower-case UUID")
    if not isinstance(data["resref"], str) or not _RESREF_RE.fullmatch(data["resref"]):
        raise ValueError("resref must be a 1-16 character lower-case KOTOR resref")
    if data["game"] not in ("K1", "K2"):
        raise ValueError("game must be K1 or K2")
    game_dir = data["gameDir"]
    if (
        not isinstance(game_dir, str)
        or not 3 <= len(game_dir) <= 1024
        or "\x00" in game_dir
        or not Path(game_dir).is_absolute()
    ):
        raise ValueError("gameDir must be an absolute path")
    for key in ("mdlSha256", "mdxSha256", "expectedGuiRevision"):
        if not isinstance(data[key], str) or not _HEX64_RE.fullmatch(data[key]):
            raise ValueError(f"{key} must be a lower-case SHA-256")
    return data


_VALIDATORS: Mapping[str, Callable[[object], dict[str, Any]]] = {
    "state": validate_state,
    "open_model": validate_open_model,
    "capture": validate_capture,
}


class KaigenGuiControlServer:
    """Loopback listener that authenticates Kaigen and dispatches to handlers."""

    def __init__(
        self,
        descriptor_path: str | os.PathLike[str],
        handlers: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
        *,
        source_identity: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        executable: str | None = None,
    ) -> None:
        path = Path(descriptor_path)
        if not path.is_absolute():
            raise ValueError("descriptor path must be absolute")
        if set(handlers) != set(ROUTES.values()):
            raise ValueError("handlers must cover exactly state, open_model and capture")
        self._descriptor_path = path
        self._handlers = dict(handlers)
        self._source = dict(source_identity or {})
        self._source.setdefault("ipc/kaigen_gui_control.py", MODULE_SHA256)
        self._clock = clock
        self._executable = str(executable or sys.executable)
        self._secret = secrets.token_bytes(32)
        self._session = uuid.uuid4().hex
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self._nonce_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._published = False

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def port(self) -> int:
        return int(self._server.server_address[1]) if self._server else 0

    @property
    def session(self) -> str:
        return self._session

    @property
    def source_identity(self) -> dict[str, str]:
        return dict(self._source)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": DESCRIPTOR_SCHEMA,
            "sessionId": self._session,
            "secret": self._secret.hex(),
            "pid": os.getpid(),
            "port": self.port,
            "executable": self._executable,
            "source": self.source_identity,
            "createdAt": int(self._clock()),
        }

    def start(self) -> None:
        if self._server is not None:
            return
        owner = self

        class _Handler(_ControlHandler):
            control = owner

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="GhostStudio-Kaigen-Control",
            daemon=True,
        )
        self._thread.start()
        try:
            self._publish()
        except BaseException:
            self.stop()
            raise
        atexit.register(self.stop)

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
            finally:
                server.server_close()
        self._remove_descriptor()

    def _publish(self) -> None:
        path = self._descriptor_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        data = json.dumps(self.descriptor(), sort_keys=True).encode("utf-8")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temp, path)
        except BaseException:
            try:
                temp.unlink()
            except OSError:
                pass
            raise
        self._published = True

    def _remove_descriptor(self) -> None:
        if not self._published:
            return
        self._published = False
        try:
            current = json.loads(self._descriptor_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        # Never remove a descriptor another Ghost process published since.
        if isinstance(current, dict) and current.get("sessionId") == self._session:
            try:
                self._descriptor_path.unlink()
            except OSError:
                pass

    # ── request handling ─────────────────────────────────────────────────
    def verify(self, *, method: str, path: str, headers: Mapping[str, str], body: bytes) -> str:
        session = str(headers.get(HEADER_SESSION) or "")
        timestamp = str(headers.get(HEADER_TIMESTAMP) or "")
        nonce = str(headers.get(HEADER_NONCE) or "")
        signature = str(headers.get(HEADER_SIGNATURE) or "").lower()
        if not _SESSION_RE.fullmatch(session) or not hmac.compare_digest(session, self._session):
            raise PermissionError("unknown-session")
        if not re.fullmatch(r"[0-9]{1,12}", timestamp):
            raise PermissionError("invalid-timestamp")
        if abs(self._clock() - int(timestamp)) > MAX_CLOCK_SKEW_SECONDS:
            raise PermissionError("stale-request")
        if not _NONCE_RE.fullmatch(nonce):
            raise PermissionError("invalid-nonce")
        expected = request_signature(
            self._secret,
            method=method,
            path=path,
            session=session,
            timestamp=timestamp,
            nonce=nonce,
            body=body,
        )
        if not re.fullmatch(r"[0-9a-f]{64}", signature) or not hmac.compare_digest(signature, expected):
            raise PermissionError("invalid-signature")
        with self._nonce_lock:
            now = self._clock()
            while self._nonces:
                oldest, seen = next(iter(self._nonces.items()))
                if now - seen <= 2 * MAX_CLOCK_SKEW_SECONDS and len(self._nonces) < REPLAY_CAPACITY:
                    break
                self._nonces.pop(oldest)
            if nonce in self._nonces:
                raise PermissionError("replayed-request")
            self._nonces[nonce] = now
        return nonce

    def dispatch(self, route: str, payload: object) -> tuple[int, dict[str, Any]]:
        try:
            data = _VALIDATORS[route](payload)
        except ValueError as exc:
            return 400, {"status": "error", "code": "invalid-payload", "message": str(exc)[:300]}
        # One GUI operation at a time; a concurrent request is refused rather
        # than queued behind a long model load.
        if not self._operation_lock.acquire(timeout=0.5 if route != "state" else 5.0):
            return 409, {"status": "refused", "code": "busy", "message": "Another Kaigen GUI request is running."}
        try:
            result = self._handlers[route](data)
        except ControlRefusal as exc:
            return 409, {
                "status": "refused",
                "code": exc.code,
                "message": exc.message[:500],
                **({"details": exc.details} if exc.details else {}),
            }
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            return 500, {"status": "error", "code": "gui-operation-failed", "message": type(exc).__name__}
        finally:
            self._operation_lock.release()
        if not isinstance(result, dict):
            return 500, {"status": "error", "code": "gui-operation-failed", "message": "invalid handler result"}
        return 200, {
            "status": "ok",
            "sessionId": self._session,
            "pid": os.getpid(),
            "source": self.source_identity,
            **result,
        }

    def sign_response(self, status: int, nonce: str, body: bytes) -> str:
        return response_signature(
            self._secret, status=status, session=self._session, nonce=nonce, body=body
        )


class _ControlHandler(BaseHTTPRequestHandler):
    control: KaigenGuiControlServer
    server_version = "GhostStudioKaigenControl/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: dict[str, Any], nonce: str | None = None) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > MAX_RESPONSE_BYTES:
            status = 500
            body = b'{"code":"response-too-large","status":"error"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if nonce is not None:
            self.send_header(HEADER_RESPONSE_SIGNATURE, self.control.sign_response(status, nonce, body))
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, status: int, code: str) -> None:
        self._send(status, {"status": "error", "code": code})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        route = ROUTES.get(self.path)
        if route is None:
            self._reject(404, "unknown-route")
            return
        if self.headers.get("Host") != f"127.0.0.1:{self.control.port}":
            self._reject(421, "invalid-host")
            return
        length_text = str(self.headers.get("Content-Length") or "")
        if not re.fullmatch(r"[0-9]{1,8}", length_text) or int(length_text) > MAX_REQUEST_BYTES:
            self._reject(413, "request-too-large")
            return
        body = self.rfile.read(int(length_text))
        try:
            nonce = self.control.verify(method="POST", path=self.path, headers=self.headers, body=body)
        except PermissionError as exc:
            self._reject(401, str(exc))
            return
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, ValueError):
            self._send(400, {"status": "error", "code": "invalid-json"}, nonce)
            return
        status, result = self.control.dispatch(route, payload)
        self._send(status, result, nonce)

    def _method_not_allowed(self) -> None:
        self._reject(405, "method-not-allowed")

    do_GET = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed  # noqa: N815
