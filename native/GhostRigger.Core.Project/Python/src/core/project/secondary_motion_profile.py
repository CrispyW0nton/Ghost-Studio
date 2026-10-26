"""Versioned, Qt-free secondary-motion authoring state.

The parent project owns authoring decisions. Generated model data and runtime
addresses never belong in this document. Unknown fields are retained so newer
producers can safely round-trip through older GhostRigger builds.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .ghostrigger_project import stable_project_id


SECONDARY_MOTION_PROFILE_SCHEMA = "ghostrigger.secondary_motion_profile.v1"
CURRENT_SECONDARY_MOTION_PROFILE_VERSION = 1
SECONDARY_MOTION_BACKENDS = ("runtime_pbd", "native_dangly", "rigid")
SECONDARY_MOTION_RUNTIME_STATES = ("capture_only", "calibrated", "write_enabled")

_KNOWN_FIELDS = {
    "schema",
    "schema_version",
    "profile_id",
    "display_label",
    "target_game",
    "backend_capability_version",
    "model_identity",
    "regions",
    "preview_tests",
    "validation_evidence",
    "runtime",
    "compiled",
}


def _mapping(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _normalise_driver(value: Any) -> dict[str, Any]:
    data = _mapping(value)
    data["name"] = str(data.get("name") or "").strip()
    data["parent_name"] = str(data.get("parent_name") or "").strip()
    data["rest_position"] = [float(item) for item in data.get("rest_position") or (0.0, 0.0, 0.0)][:3]
    data["rest_quaternion"] = [float(item) for item in data.get("rest_quaternion") or (0.0, 0.0, 0.0, 1.0)][:4]
    data["rest_scale"] = [float(item) for item in data.get("rest_scale") or (1.0, 1.0, 1.0)][:3]
    if data.get("offline_preorder") is not None:
        data["offline_preorder"] = int(data["offline_preorder"])
    return data


def _normalise_region(value: Any, index: int) -> dict[str, Any]:
    data = _mapping(value)
    data["region_id"] = str(data.get("region_id") or f"region_{index + 1}").strip()
    data["label"] = str(data.get("label") or data["region_id"]).strip()
    data["semantic_type"] = str(data.get("semantic_type") or "custom").strip().lower()
    data["backend"] = str(data.get("backend") or "rigid").strip().lower()
    data["mesh_membership"] = [str(item).strip() for item in data.get("mesh_membership") or () if str(item).strip()]
    data["vertex_group"] = str(data.get("vertex_group") or "").strip()
    data["drivers"] = [_normalise_driver(item) for item in data.get("drivers") or ()]
    data["anchors"] = _mapping(data.get("anchors"))
    data["colliders"] = [_mapping(item) for item in data.get("colliders") or ()]
    data["solver"] = _mapping(data.get("solver"))
    return data


@dataclass
class SecondaryMotionProfile:
    """Portable profile with lossless extension-field preservation."""

    profile_id: str = field(default_factory=lambda: stable_project_id("secondary_motion"))
    display_label: str = "Secondary Motion"
    target_game: str = "K2"
    backend_capability_version: int = 1
    model_identity: dict[str, Any] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)
    preview_tests: list[dict[str, Any]] = field(default_factory=list)
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(
        default_factory=lambda: {
            "state": "capture_only",
            "capture_session": "",
            "compiler_version": "",
            "write_enabled": False,
        }
    )
    compiled: dict[str, Any] = field(default_factory=dict)
    schema: str = SECONDARY_MOTION_PROFILE_SCHEMA
    schema_version: int = CURRENT_SECONDARY_MOTION_PROFILE_VERSION
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = deepcopy(self.extra_fields)
        payload.update(
            {
                "schema": self.schema,
                "schema_version": int(self.schema_version),
                "profile_id": str(self.profile_id or ""),
                "display_label": str(self.display_label or ""),
                "target_game": str(self.target_game or "K2").upper(),
                "backend_capability_version": int(self.backend_capability_version),
                "model_identity": deepcopy(self.model_identity),
                "regions": deepcopy(self.regions),
                "preview_tests": deepcopy(self.preview_tests),
                "validation_evidence": deepcopy(self.validation_evidence),
                "runtime": deepcopy(self.runtime),
            }
        )
        if self.compiled:
            payload["compiled"] = deepcopy(self.compiled)
        else:
            payload.pop("compiled", None)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "SecondaryMotionProfile":
        data = _mapping(value)
        version = int(data.get("schema_version") or CURRENT_SECONDARY_MOTION_PROFILE_VERSION)
        schema = str(data.get("schema") or SECONDARY_MOTION_PROFILE_SCHEMA)
        if schema != SECONDARY_MOTION_PROFILE_SCHEMA:
            raise ValueError(f"Unsupported secondary-motion profile schema {schema!r}.")
        if version != CURRENT_SECONDARY_MOTION_PROFILE_VERSION:
            raise ValueError(f"Unsupported secondary-motion profile version {version}.")
        model_identity = _mapping(data.get("model_identity"))
        model_identity["model_resrefs"] = [
            str(item).strip().lower()
            for item in model_identity.get("model_resrefs") or ()
            if str(item).strip()
        ]
        runtime = {
            "state": "capture_only",
            "capture_session": "",
            "compiler_version": "",
            "write_enabled": False,
            **_mapping(data.get("runtime")),
        }
        runtime["state"] = str(runtime.get("state") or "capture_only").strip().lower()
        runtime["write_enabled"] = bool(runtime.get("write_enabled"))
        return cls(
            profile_id=str(data.get("profile_id") or stable_project_id("secondary_motion")),
            display_label=str(data.get("display_label") or "Secondary Motion"),
            target_game=str(data.get("target_game") or "K2").upper(),
            backend_capability_version=int(data.get("backend_capability_version") or 1),
            model_identity=model_identity,
            regions=[_normalise_region(item, index) for index, item in enumerate(data.get("regions") or ())],
            preview_tests=[_mapping(item) for item in data.get("preview_tests") or ()],
            validation_evidence=_mapping(data.get("validation_evidence")),
            runtime=runtime,
            compiled=_mapping(data.get("compiled")),
            schema=schema,
            schema_version=version,
            extra_fields={key: deepcopy(item) for key, item in data.items() if key not in _KNOWN_FIELDS},
        )


def default_secondary_motion_profile(*, target_game: str = "K2", display_label: str = "Secondary Motion") -> dict[str, Any]:
    return SecondaryMotionProfile(display_label=display_label, target_game=target_game).to_dict()


__all__ = [
    "CURRENT_SECONDARY_MOTION_PROFILE_VERSION",
    "SECONDARY_MOTION_BACKENDS",
    "SECONDARY_MOTION_PROFILE_SCHEMA",
    "SECONDARY_MOTION_RUNTIME_STATES",
    "SecondaryMotionProfile",
    "default_secondary_motion_profile",
]
