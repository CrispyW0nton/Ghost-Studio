"""Bounded stdio MCP server for Ghost Studio's private spatial IPC surface.

The adapter deliberately exposes only four fixed tools.  It does not import
KotorMCP, accept arbitrary URLs or paths, inherit proxy configuration, or
write protocol diagnostics to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, TextIO
import urllib.error
import urllib.request


def _load_spatial_auth_module():
    """Load the shared contract without executing legacy ``ipc.__init__``.

    That package initializer intentionally imports the broad historical IPC
    surface.  The narrow adapter must remain independently launchable.
    """

    module_name = "_ghoststudio_spatial_auth_contract"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).resolve().parents[1] / "ipc" / "spatial_auth.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Ghost Studio spatial authentication contract is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SPATIAL_AUTH = _load_spatial_auth_module()
SpatialAuthenticationError = _SPATIAL_AUTH.SpatialAuthenticationError
SpatialRequestSigner = _SPATIAL_AUTH.SpatialRequestSigner
SpatialSessionDescriptor = _SPATIAL_AUTH.SpatialSessionDescriptor
default_spatial_session_path = _SPATIAL_AUTH.default_spatial_session_path
load_spatial_session_descriptor = _SPATIAL_AUTH.load_spatial_session_descriptor
spatial_transport_marker = _SPATIAL_AUTH.spatial_transport_marker
WINDOWS_SPATIAL_TRANSPORT = _SPATIAL_AUTH.WINDOWS_SPATIAL_TRANSPORT
LOOPBACK_SPATIAL_TRANSPORT = _SPATIAL_AUTH.LOOPBACK_SPATIAL_TRANSPORT


def _load_spatial_pipe_module():
    module_name = "_ghoststudio_spatial_pipe_contract"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = Path(__file__).resolve().parents[1] / "ipc" / "spatial_pipe.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Ghost Studio spatial pipe contract is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SPATIAL_PIPE = _load_spatial_pipe_module()
SpatialPipeError = _SPATIAL_PIPE.SpatialPipeError
call_windows_spatial_pipe = _SPATIAL_PIPE.call_windows_spatial_pipe


SERVER_NAME = "ghoststudio-spatial"
SERVER_VERSION = "0.1.0"
LATEST_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = frozenset(
    {
        LATEST_PROTOCOL_VERSION,
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
        "2024-10-07",
    }
)
MAX_STDIN_FRAME_BYTES = 1024 * 1024
MAX_HTTP_RESPONSE_BYTES = (4 * 1024 * 1024) - (64 * 1024)
HTTP_TIMEOUT_SECONDS = 8.0
CAPTURE_ID_PATTERN = r"^[A-Za-z0-9_-]{16,128}$"
_CAPTURE_ID_RE = re.compile(CAPTURE_ID_PATTERN)
_SESSION_PATH_ENV = "GHOSTSTUDIO_SPATIAL_SESSION_PATH"
_REVISION_PATTERN = r"^sha256:[0-9a-f]{64}$"
MAX_SPATIAL_ENTITIES = 1024
SPATIAL_CAPABILITIES = (
    "health",
    "spatial-snapshot",
    "capture",
    "evidence-gaps",
)


class SpatialAdapterError(RuntimeError):
    """A stable non-secret error safe to return through MCP."""

    _MESSAGES = {
        "ghoststudio-unavailable": (
            "Ghost Studio's authenticated spatial session is unavailable."
        ),
        "invalid-response": "Ghost Studio returned an invalid spatial response.",
        "response-too-large": (
            "Ghost Studio's spatial response exceeds the bounded MCP frame."
        ),
        "spatial-request-failed": (
            "Ghost Studio could not complete the authenticated spatial request."
        ),
    }

    def __init__(self, code: str):
        self.code = code
        super().__init__(self._MESSAGES.get(code, "Ghost Studio spatial request failed."))


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        raise SpatialAdapterError("invalid-response")


@dataclass(frozen=True)
class _ToolRoute:
    method: str
    path: str


TOOL_ROUTES: Mapping[str, _ToolRoute] = {
    "ghoststudio_health": _ToolRoute("GET", "/api/mcpstudio/health"),
    "ghoststudio_spatial_snapshot": _ToolRoute(
        "POST",
        "/api/mcpstudio/spatial-snapshot",
    ),
    "ghoststudio_capture": _ToolRoute("POST", "/api/mcpstudio/capture"),
    "ghoststudio_evidence_gaps": _ToolRoute(
        "POST",
        "/api/mcpstudio/evidence-gaps",
    ),
}


_GUI_READINESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ready": {"type": "boolean"},
        "mainThreadObserved": {"type": "boolean"},
        "windowVisible": {"type": "boolean"},
        "windowMinimized": {"type": "boolean"},
        "viewport": {
            "type": "object",
            "properties": {
                "stateAvailable": {"type": "boolean"},
                "visible": {"type": "boolean"},
                "width": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32768,
                },
                "height": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32768,
                },
            },
            "required": [
                "stateAvailable",
                "visible",
                "width",
                "height",
            ],
            "additionalProperties": False,
        },
        "grid": {
            "type": "object",
            "properties": {
                "stateAvailable": {"type": "boolean"},
                "visible": {"type": "boolean"},
            },
            "required": ["stateAvailable", "visible"],
            "additionalProperties": False,
        },
        "reason": {
            "type": ["string", "null"],
            "enum": [
                None,
                "gui-main-thread-unobserved",
                "window-not-visible",
                "window-minimized",
                "viewport-state-unavailable",
                "viewport-not-visible",
                "grid-state-unavailable",
                "gui-readiness-callback-unavailable",
                "gui-readiness-check-failed",
            ],
        },
    },
    "required": [
        "ready",
        "mainThreadObserved",
        "windowVisible",
        "windowMinimized",
        "viewport",
        "grid",
        "reason",
    ],
    "additionalProperties": False,
}

_MATRIX_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 4,
    "maxItems": 4,
    "items": {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": {"type": "number"},
    },
}

_HEALTH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"const": "ok"},
        "schema": {"const": "ghoststudio-spatial-health/v2"},
        "program": {"type": "string", "minLength": 1, "maxLength": 64},
        "endpoint": {
            "type": "object",
            "properties": {
                "authenticated": {"const": True},
                "transport": {
                    "enum": [
                        WINDOWS_SPATIAL_TRANSPORT,
                        LOOPBACK_SPATIAL_TRANSPORT,
                    ]
                },
            },
            "required": ["authenticated", "transport"],
            "additionalProperties": False,
        },
        "gui": _GUI_READINESS_SCHEMA,
        "capabilities": {
            "type": "array",
            "minItems": len(SPATIAL_CAPABILITIES),
            "maxItems": len(SPATIAL_CAPABILITIES),
            "items": {"enum": list(SPATIAL_CAPABILITIES)},
        },
    },
    "required": [
        "status",
        "schema",
        "program",
        "endpoint",
        "gui",
        "capabilities",
    ],
    "additionalProperties": False,
}

_SNAPSHOT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"const": "ok"},
        "schema": {"const": "ghoststudio-spatial-response/v1"},
        "snapshot": {
            "type": "object",
            "properties": {
                "schemaVersion": {"const": "1.0"},
                "application": {
                    "type": "object",
                    "properties": {
                        "id": {"const": "ghoststudio"},
                        "version": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 64,
                        },
                        "apiVersion": {
                            "const": "ghoststudio-spatial/v1"
                        },
                    },
                    "required": ["id", "version", "apiVersion"],
                    "additionalProperties": False,
                },
                "sceneRevision": {
                    "type": "string",
                    "pattern": _REVISION_PATTERN,
                },
                "capturedAt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "coordinateFrames": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"const": "ghoststudio-world"},
                            "semanticSpace": {"const": "world"},
                            "handedness": {"const": "right"},
                            "metersPerUnit": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "originMeters": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "number"},
                            },
                            "basis": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "items": {"type": "number"},
                                },
                            },
                            "upAxis": {"const": "+Z"},
                            "forwardAxis": {"const": "+Y"},
                        },
                        "required": [
                            "id",
                            "semanticSpace",
                            "handedness",
                            "metersPerUnit",
                            "originMeters",
                            "basis",
                            "upAxis",
                            "forwardAxis",
                        ],
                        "additionalProperties": False,
                    },
                },
                "entities": {
                    "type": "array",
                    "maxItems": MAX_SPATIAL_ENTITIES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "stableId": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 256,
                            },
                            "path": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1024,
                            },
                            "type": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "visible": {"type": "boolean"},
                            "locked": {"type": "boolean"},
                            "selected": {"type": "boolean"},
                            "coordinateFrameId": {
                                "const": "ghoststudio-world"
                            },
                            "localMatrix": _MATRIX_SCHEMA,
                            "worldMatrix": _MATRIX_SCHEMA,
                            "pivot": {
                                "type": "object",
                                "properties": {
                                    "coordinateFrameId": {
                                        "const": "ghoststudio-world"
                                    },
                                    "semanticSpace": {"const": "local"},
                                    "position": {
                                        "type": "array",
                                        "minItems": 3,
                                        "maxItems": 3,
                                        "items": {"type": "number"},
                                    },
                                    "rotationEulerDegreesXYZ": {
                                        "type": "array",
                                        "minItems": 3,
                                        "maxItems": 3,
                                        "items": {"type": "number"},
                                    },
                                    "enabled": {"type": "boolean"},
                                },
                                "required": [
                                    "coordinateFrameId",
                                    "semanticSpace",
                                    "position",
                                    "rotationEulerDegreesXYZ",
                                    "enabled",
                                ],
                                "additionalProperties": False,
                            },
                            "transformSemantics": {
                                "type": "object",
                                "properties": {
                                    "matrixLayout": {"const": "row-major"},
                                    "vectorConvention": {
                                        "const": "column-vector"
                                    },
                                    "composition": {
                                        "const": "T*Rz*Ry*Rx*S"
                                    },
                                    "rotationInput": {
                                        "const": "Euler XYZ degrees"
                                    },
                                },
                                "required": [
                                    "matrixLayout",
                                    "vectorConvention",
                                    "composition",
                                    "rotationInput",
                                ],
                                "additionalProperties": False,
                            },
                            "bounds": {
                                "type": "object",
                                "properties": {
                                    "minimum": {
                                        "type": "array",
                                        "minItems": 3,
                                        "maxItems": 3,
                                        "items": {"type": "number"},
                                    },
                                    "maximum": {
                                        "type": "array",
                                        "minItems": 3,
                                        "maxItems": 3,
                                        "items": {"type": "number"},
                                    },
                                },
                                "required": ["minimum", "maximum"],
                                "additionalProperties": False,
                            },
                            "materials": {
                                "type": "array",
                                "maxItems": 64,
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 256,
                                },
                            },
                        },
                        "required": [
                            "stableId",
                            "path",
                            "type",
                            "visible",
                            "locked",
                            "coordinateFrameId",
                            "localMatrix",
                            "worldMatrix",
                            "pivot",
                            "transformSemantics",
                        ],
                        "additionalProperties": False,
                    },
                },
                "hierarchy": {
                    "type": "object",
                    "properties": {
                        "status": {"const": "unavailable"},
                        "reason": {
                            "const": "scene-parent-hierarchy-unavailable"
                        },
                    },
                    "required": ["status", "reason"],
                    "additionalProperties": False,
                },
                "selection": {
                    "type": "object",
                    "properties": {
                        "mode": {"const": "object"},
                        "stableIds": {
                            "type": "array",
                            "maxItems": MAX_SPATIAL_ENTITIES,
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 256,
                            },
                        },
                    },
                    "required": ["mode", "stableIds"],
                    "additionalProperties": False,
                },
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "enum": [
                                    "semantic-api",
                                    "derived-calculation",
                                    "screenshot",
                                ]
                            },
                            "claim": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1024,
                            },
                            "epistemicStatus": {
                                "enum": ["observed", "inferred"]
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "sourcePath": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 32768,
                            },
                            "sourceSha256": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                            },
                        },
                        "required": [
                            "kind",
                            "claim",
                            "epistemicStatus",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "viewports": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 256,
                            },
                            "rectangle": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "width": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": 32768,
                                    },
                                    "height": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": 32768,
                                    },
                                },
                                "required": ["x", "y", "width", "height"],
                                "additionalProperties": False,
                            },
                            "pixelOrigin": {
                                "enum": ["top-left", "bottom-left"]
                            },
                            "devicePixelRatio": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 16,
                            },
                            "cameraStableId": {
                                "type": ["string", "null"],
                                "maxLength": 256,
                            },
                            "projection": {
                                "enum": [
                                    "perspective",
                                    "orthographic",
                                ]
                            },
                            "viewMatrix": _MATRIX_SCHEMA,
                            "projectionMatrix": _MATRIX_SCHEMA,
                            "nearClip": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "farClip": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                            "revision": {
                                "type": "string",
                                "pattern": _REVISION_PATTERN,
                            },
                        },
                        "required": [
                            "id",
                            "rectangle",
                            "pixelOrigin",
                            "devicePixelRatio",
                            "cameraStableId",
                            "projection",
                            "viewMatrix",
                            "projectionMatrix",
                            "nearClip",
                            "farClip",
                            "revision",
                        ],
                        "additionalProperties": False,
                    },
                },
                "grid": {
                    "type": "object",
                    "properties": {
                        "coordinateFrameId": {
                            "const": "ghoststudio-world"
                        },
                        "origin": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "number"},
                        },
                        "spacing": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "subdivisions": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000000,
                        },
                        "visible": {"type": "boolean"},
                        "snapEnabled": {"type": "boolean"},
                    },
                    "required": [
                        "coordinateFrameId",
                        "origin",
                        "spacing",
                        "subdivisions",
                        "visible",
                        "snapEnabled",
                    ],
                    "additionalProperties": False,
                },
                "guiReadiness": _GUI_READINESS_SCHEMA,
            },
            "required": [
                "schemaVersion",
                "application",
                "sceneRevision",
                "capturedAt",
                "coordinateFrames",
                "entities",
                "evidence",
                "viewports",
                "grid",
                "guiReadiness",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["status", "schema", "snapshot"],
    "additionalProperties": False,
}


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "ghoststudio_health",
        "title": "Ghost Studio Spatial Health",
        "description": (
            "Verify the authenticated private spatial endpoint and separately report "
            "whether the live GUI is ready with viewport and grid markers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "outputSchema": _HEALTH_OUTPUT_SCHEMA,
    },
    {
        "name": "ghoststudio_spatial_snapshot",
        "title": "Ghost Studio Spatial Snapshot",
        "description": (
            "Read a bounded revisioned scene, selection, camera, viewport, and "
            "grid snapshot from a GUI-ready Ghost Studio process. Parent "
            "hierarchy is currently unavailable and is reported explicitly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "includeBounds": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include observed entity bounds when available.",
                },
                "includeHierarchy": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Request the hierarchy capability marker. Ghost Studio "
                        "v1 reports it as explicitly unavailable and never "
                        "synthesizes parent relationships."
                    ),
                },
                "includeSelection": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Include selection and per-entity selected flags. False "
                        "redacts both surfaces."
                    ),
                },
            },
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "outputSchema": _SNAPSHOT_OUTPUT_SCHEMA,
    },
    {
        "name": "ghoststudio_capture",
        "title": "Capture Ghost Studio Spatial Evidence",
        "description": (
            "Capture a viewport PNG bound to exact scene and viewport revisions. "
            "A screenshot is visual evidence and does not by itself prove a GUI action."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "captureId": {
                    "type": "string",
                    "pattern": CAPTURE_ID_PATTERN,
                    "minLength": 16,
                    "maxLength": 128,
                }
            },
            "required": ["captureId"],
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "ghoststudio_evidence_gaps",
        "title": "Ghost Studio Spatial Evidence Gaps",
        "description": (
            "Report which spatial claims remain unavailable, inferred, or "
            "unproven by the current semantic and visual evidence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
)


def _strict_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Tool arguments must be a JSON object.")
    return dict(value)


def _validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    payload = _strict_object(arguments)
    if name in {"ghoststudio_health", "ghoststudio_evidence_gaps"}:
        if payload:
            raise ValueError(f"{name} does not accept arguments.")
        return payload
    if name == "ghoststudio_spatial_snapshot":
        allowed = {"includeBounds", "includeHierarchy", "includeSelection"}
        if set(payload) - allowed:
            raise ValueError("Spatial snapshot arguments contain unknown fields.")
        if any(not isinstance(value, bool) for value in payload.values()):
            raise ValueError("Spatial snapshot flags must be booleans.")
        return payload
    if name == "ghoststudio_capture":
        if set(payload) != {"captureId"}:
            raise ValueError("Capture requires exactly one captureId.")
        capture_id = payload["captureId"]
        if not isinstance(capture_id, str) or not _CAPTURE_ID_RE.fullmatch(capture_id):
            raise ValueError("captureId must be a 16-128 character safe token.")
        return payload
    raise KeyError(name)


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "null":
        return value is None
    return False


def _validate_output_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
    depth: int = 0,
) -> None:
    """Validate the strict JSON-Schema subset used by this fixed catalog."""

    if depth > 32:
        raise ValueError(f"{path} exceeds the output nesting bound")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} does not match its fixed value")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is outside its fixed values")
    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = (
            [expected_type]
            if isinstance(expected_type, str)
            else list(expected_type)
        )
        if not any(
            _matches_json_type(value, item)
            for item in expected_types
            if isinstance(item, str)
        ):
            raise ValueError(f"{path} has the wrong JSON type")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path} is shorter than allowed")
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > int(maximum):
            raise ValueError(f"{path} is longer than allowed")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(str(pattern), value) is None:
            raise ValueError(f"{path} does not match its pattern")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path} is below its minimum")
        if maximum is not None and value > maximum:
            raise ValueError(f"{path} is above its maximum")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            raise ValueError(f"{path} is below its exclusive minimum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if any(name not in value for name in required):
            raise ValueError(f"{path} lacks a required field")
        if schema.get("additionalProperties") is False:
            if any(name not in properties for name in value):
                raise ValueError(f"{path} has an unknown field")
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                _validate_output_value(
                    child,
                    child_schema,
                    path=f"{path}.{name}",
                    depth=depth + 1,
                )
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path} has too few items")
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > int(maximum):
            raise ValueError(f"{path} has too many items")
        if schema.get("uniqueItems"):
            canonical = [
                json.dumps(
                    item,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in value
            ]
            if len(canonical) != len(set(canonical)):
                raise ValueError(f"{path} contains duplicate items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, child in enumerate(value):
                _validate_output_value(
                    child,
                    item_schema,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                )


def _validate_gui_ready_claim(gui: Mapping[str, Any]) -> None:
    viewport = gui["viewport"]
    grid = gui["grid"]
    positive = (
        gui["mainThreadObserved"]
        and gui["windowVisible"]
        and not gui["windowMinimized"]
        and viewport["stateAvailable"]
        and viewport["visible"]
        and viewport["width"] > 0
        and viewport["height"] > 0
        and grid["stateAvailable"]
    )
    if gui["ready"] != positive:
        raise ValueError("GUI readiness claim does not match its markers")
    if gui["ready"] != (gui["reason"] is None):
        raise ValueError("GUI readiness reason is inconsistent")


def _validate_tool_response(
    name: str,
    arguments: Mapping[str, Any],
    payload: Any,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SpatialAdapterError("invalid-response")
    try:
        if name == "ghoststudio_health":
            _validate_output_value(payload, _HEALTH_OUTPUT_SCHEMA)
            if payload["capabilities"] != list(SPATIAL_CAPABILITIES):
                raise ValueError("health capabilities are not canonical")
            _validate_gui_ready_claim(payload["gui"])
        elif name == "ghoststudio_spatial_snapshot":
            _validate_output_value(payload, _SNAPSHOT_OUTPUT_SCHEMA)
            snapshot = payload["snapshot"]
            gui = snapshot["guiReadiness"]
            _validate_gui_ready_claim(gui)
            if not gui["ready"]:
                raise ValueError("snapshot lacks positive GUI readiness")
            include_selection = arguments.get("includeSelection", True)
            include_hierarchy = arguments.get("includeHierarchy", True)
            include_bounds = arguments.get("includeBounds", True)
            entities = snapshot["entities"]
            if include_selection:
                selection = snapshot.get("selection")
                if not isinstance(selection, dict):
                    raise ValueError("selection is missing")
                if any("selected" not in entity for entity in entities):
                    raise ValueError("entity selection marker is missing")
                selected_ids = sorted(
                    entity["stableId"]
                    for entity in entities
                    if entity["selected"]
                )
                if selection["stableIds"] != selected_ids:
                    raise ValueError("selection does not match entity markers")
            elif (
                "selection" in snapshot
                or any("selected" in entity for entity in entities)
            ):
                raise ValueError("selection was not redacted")
            if include_hierarchy:
                if snapshot.get("hierarchy") != {
                    "status": "unavailable",
                    "reason": "scene-parent-hierarchy-unavailable",
                }:
                    raise ValueError("hierarchy capability marker is missing")
            elif "hierarchy" in snapshot:
                raise ValueError("hierarchy was not redacted")
            if not include_bounds and any(
                "bounds" in entity for entity in entities
            ):
                raise ValueError("bounds were not redacted")
            viewport = snapshot["viewports"][0]
            if viewport["farClip"] <= viewport["nearClip"]:
                raise ValueError("viewport clip range is invalid")
    except SpatialAdapterError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SpatialAdapterError("invalid-response") from exc
    return payload


def _serialize_tool_payload(payload: Mapping[str, Any]) -> str:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpatialAdapterError("invalid-response") from exc
    if len(text.encode("utf-8")) > MAX_HTTP_RESPONSE_BYTES:
        raise SpatialAdapterError("response-too-large")
    return text


def _descriptor_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    configured = str(values.get(_SESSION_PATH_ENV) or "").strip()
    if not configured:
        return default_spatial_session_path(values)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise SpatialAdapterError("ghoststudio-unavailable")
    return path


class GhostStudioSpatialClient:
    """Signed client for the approval-bound GUI spatial bridge."""

    def __init__(
        self,
        *,
        session_path: Path | None = None,
        descriptor_loader: Callable[[str | os.PathLike[str]], SpatialSessionDescriptor] = (
            load_spatial_session_descriptor
        ),
        opener: Any | None = None,
        pipe_caller: Callable[..., Any] = call_windows_spatial_pipe,
        environ: Mapping[str, str] | None = None,
        timeout_seconds: float = HTTP_TIMEOUT_SECONDS,
    ):
        self._session_path = session_path or _descriptor_path()
        self._descriptor_loader = descriptor_loader
        self._opener = opener
        self._pipe_caller = pipe_caller
        self._environ = os.environ if environ is None else environ
        self._timeout_seconds = float(timeout_seconds)

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        absolute_deadline = time.monotonic() + self._timeout_seconds
        route = TOOL_ROUTES.get(name)
        if route is None:
            raise KeyError(name)
        payload = _validate_arguments(name, arguments)
        body = (
            b""
            if route.method == "GET"
            else json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        try:
            descriptor = self._descriptor_loader(self._session_path)
            signer = SpatialRequestSigner(descriptor.credentials)
            headers = signer.sign(
                method=route.method,
                path=route.path,
                body=body,
            )
        except (OSError, SpatialAuthenticationError, ValueError) as exc:
            raise SpatialAdapterError("ghoststudio-unavailable") from exc
        if os.name == "nt":
            try:
                spatial_transport_marker(
                    self._environ,
                    required=True,
                )
            except SpatialAuthenticationError as exc:
                raise SpatialAdapterError("ghoststudio-unavailable") from exc
            if not self._is_windows_pipe_descriptor(descriptor):
                raise SpatialAdapterError("ghoststudio-unavailable")
            try:
                response = self._call_windows_pipe(
                    descriptor=descriptor,
                    headers=headers,
                    method=route.method,
                    path=route.path,
                    body=body,
                    absolute_deadline=absolute_deadline,
                )
            except SpatialAdapterError:
                raise
            except ValueError as exc:
                raise SpatialAdapterError("spatial-request-failed") from exc
            return self._decode_pipe_json(response)
        if (
            descriptor.schema != "ghoststudio-spatial-session/v1"
            or descriptor.transport != LOOPBACK_SPATIAL_TRANSPORT
            or not isinstance(descriptor.port, int)
        ):
            raise SpatialAdapterError("ghoststudio-unavailable")
        if route.method != "GET":
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"http://127.0.0.1:{descriptor.port}{route.path}",
            data=None if route.method == "GET" else body,
            headers=headers,
            method=route.method,
        )
        return self._open_json(request)

    @staticmethod
    def _is_windows_pipe_descriptor(
        descriptor: SpatialSessionDescriptor,
    ) -> bool:
        return bool(
            descriptor.schema == "ghoststudio-spatial-session/v2"
            and descriptor.transport == WINDOWS_SPATIAL_TRANSPORT
            and descriptor.pipe_name
            and descriptor.port is None
        )

    def _call_windows_pipe(
        self,
        *,
        descriptor: SpatialSessionDescriptor,
        headers: Mapping[str, str],
        method: str,
        path: str,
        body: bytes,
        absolute_deadline: float,
    ) -> Any:
        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            raise SpatialAdapterError("spatial-request-failed")
        initial_failure: BaseException | None = None
        try:
            return self._pipe_caller(
                descriptor.pipe_name,
                method=method,
                path=path,
                headers=headers,
                body=body,
                timeout_seconds=remaining,
                expected_server_pid=descriptor.pid,
                response_secret=descriptor.credentials.secret,
            )
        except SpatialAdapterError:
            raise
        except (OSError, SpatialPipeError, TimeoutError) as exc:
            initial_failure = exc
        except ValueError as exc:
            raise SpatialAdapterError("spatial-request-failed") from exc

        try:
            replacement = self._descriptor_loader(self._session_path)
        except (OSError, SpatialAuthenticationError, ValueError) as exc:
            raise SpatialAdapterError("spatial-request-failed") from exc
        if not self._is_windows_pipe_descriptor(replacement):
            raise SpatialAdapterError("spatial-request-failed")
        descriptor_changed = (
            replacement.credentials.session_id
            != descriptor.credentials.session_id
            or replacement.pipe_name != descriptor.pipe_name
        )
        if not descriptor_changed:
            raise SpatialAdapterError(
                "spatial-request-failed"
            ) from initial_failure
        try:
            replacement_headers = SpatialRequestSigner(
                replacement.credentials
            ).sign(
                method=method,
                path=path,
                body=body,
            )
        except (SpatialAuthenticationError, ValueError) as exc:
            raise SpatialAdapterError("spatial-request-failed") from exc
        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            raise SpatialAdapterError(
                "spatial-request-failed"
            ) from initial_failure
        try:
            return self._pipe_caller(
                replacement.pipe_name,
                method=method,
                path=path,
                headers=replacement_headers,
                body=body,
                timeout_seconds=remaining,
                expected_server_pid=replacement.pid,
                response_secret=replacement.credentials.secret,
            )
        except SpatialAdapterError:
            raise
        except (
            OSError,
            SpatialPipeError,
            TimeoutError,
            ValueError,
        ) as exc:
            raise SpatialAdapterError("spatial-request-failed") from exc

    @staticmethod
    def _decode_pipe_json(response: Any) -> dict[str, Any]:
        try:
            status = int(response.status)
            content_type = str(response.content_type)
            raw = response.body
        except (AttributeError, TypeError, ValueError) as exc:
            raise SpatialAdapterError("invalid-response") from exc
        if status != 200:
            raise SpatialAdapterError("spatial-request-failed")
        if content_type != "application/json" or not isinstance(raw, bytes):
            raise SpatialAdapterError("invalid-response")
        if len(raw) > MAX_HTTP_RESPONSE_BYTES:
            raise SpatialAdapterError("response-too-large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpatialAdapterError("invalid-response") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise SpatialAdapterError("invalid-response")
        return payload

    def _open_json(self, request: urllib.request.Request) -> dict[str, Any]:
        opener = self._opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )
        try:
            with opener.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                status = int(getattr(response, "status", 0) or response.getcode())
                content_type = str(response.headers.get("Content-Type") or "")
                length_text = str(response.headers.get("Content-Length") or "").strip()
                if length_text:
                    try:
                        content_length = int(length_text, 10)
                        if content_length < 0:
                            raise SpatialAdapterError("invalid-response")
                        if content_length > MAX_HTTP_RESPONSE_BYTES:
                            raise SpatialAdapterError("response-too-large")
                    except ValueError as exc:
                        raise SpatialAdapterError("invalid-response") from exc
                raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
        except SpatialAdapterError:
            raise
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise SpatialAdapterError("spatial-request-failed") from exc
        if status != 200:
            raise SpatialAdapterError("spatial-request-failed")
        if not content_type.lower().split(";", 1)[0].strip() == "application/json":
            raise SpatialAdapterError("invalid-response")
        if len(raw) > MAX_HTTP_RESPONSE_BYTES:
            raise SpatialAdapterError("response-too-large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpatialAdapterError("invalid-response") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise SpatialAdapterError("invalid-response")
        return payload


class GhostStudioSpatialMcpServer:
    """Small JSON-RPC dispatcher compatible with MCP stdio framing."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], GhostStudioSpatialClient] = (
            GhostStudioSpatialClient
        ),
    ):
        self._client_factory = client_factory

    def dispatch(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(None, -32600, "Invalid Request")
        method = message.get("method")
        request_id = message.get("id")
        if not isinstance(method, str):
            return self._error(request_id, -32600, "Invalid Request")
        if "id" not in message:
            return None
        if (
            request_id is not None
            and (
                isinstance(request_id, bool)
                or not isinstance(request_id, (str, int))
            )
        ):
            return self._error(None, -32600, "Invalid Request")
        # A null or omitted params object means "no params" (clients such as the Codex desktop send either form).
        params = message.get("params")
        if params is None:
            params = {}
        if method == "initialize":
            if not isinstance(params, dict):
                return self._error(request_id, -32602, "Invalid params")
            requested = str(params.get("protocolVersion") or "")
            negotiated = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else LATEST_PROTOCOL_VERSION
            )
            return self._result(
                request_id,
                {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                    "instructions": (
                        "Use only the four authenticated Ghost Studio spatial "
                        "tools. Visual captures do not prove unobserved GUI actions."
                    ),
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            # Unknown request fields are ignored (MCP: receivers ignore fields they do not understand). Rejecting
            # them failed the Codex desktop's startup handshake with -32602 on about half of all launches (Oct 2).
            if not isinstance(params, dict) or ("_meta" in params and not isinstance(params["_meta"], dict)):
                return self._error(request_id, -32602, "Invalid params")
            if params.get("cursor") not in (None, ""):
                return self._error(request_id, -32602, "Invalid cursor")
            return self._result(
                request_id,
                {"tools": [dict(tool) for tool in TOOLS]},
            )
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return self._error(request_id, -32601, "Method not found")

    def _call_tool(self, request_id: Any, params: Any) -> dict[str, Any]:
        # MCP request metadata belongs to the transport, not the native tool's
        # arguments or authority. Desktop clients may attach it to any request.
        # Unknown top-level request fields (e.g. task metadata) are ignored; only name/arguments are used.
        if not isinstance(params, dict) or ("_meta" in params and not isinstance(params["_meta"], dict)):
            return self._error(request_id, -32602, "Invalid params")
        name = params.get("name")
        if not isinstance(name, str) or name not in TOOL_ROUTES:
            return self._error(request_id, -32602, "Unknown tool")
        try:
            arguments = _validate_arguments(name, params.get("arguments"))
            payload = self._client_factory().call(name, arguments)
            payload = _validate_tool_response(name, arguments, payload)
            text = _serialize_tool_payload(payload)
        except ValueError as exc:
            return self._tool_error(request_id, "invalid-arguments", str(exc))
        except SpatialAdapterError as exc:
            return self._tool_error(request_id, exc.code, str(exc))
        except Exception:
            return self._tool_error(
                request_id,
                "spatial-request-failed",
                "Ghost Studio spatial request failed.",
            )
        return self._result(
            request_id,
            {
                "content": [{"type": "text", "text": text}],
                "structuredContent": payload,
            },
        )

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(
        request_id: Any,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    @classmethod
    def _tool_error(
        cls,
        request_id: Any,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        return cls._result(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"status": "error", "code": code, "message": message},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
                "isError": True,
            },
        )

    def run(
        self,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        for raw_line in input_stream:
            if len(raw_line.encode("utf-8")) > MAX_STDIN_FRAME_BYTES:
                self._write(
                    output_stream,
                    self._error(None, -32700, "Parse error"),
                )
                continue
            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                self._write(
                    output_stream,
                    self._error(None, -32700, "Parse error"),
                )
                continue
            response = self.dispatch(message)
            if response is not None:
                self._write(output_stream, response)

    @staticmethod
    def _write(output_stream: TextIO, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        encoded_size = len(serialized.encode("utf-8"))
        if encoded_size > MAX_STDIN_FRAME_BYTES * 4:
            payload = GhostStudioSpatialMcpServer._tool_error(
                payload.get("id"),
                "response-too-large",
                "Ghost Studio spatial response exceeds the bounded MCP frame.",
            )
            serialized = json.dumps(payload, separators=(",", ":"))
        output_stream.write(serialized + "\n")
        output_stream.flush()


def main() -> None:
    GhostStudioSpatialMcpServer().run()


if __name__ == "__main__":
    main()
