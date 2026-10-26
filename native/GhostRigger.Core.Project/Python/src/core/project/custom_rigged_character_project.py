"""Portable project contract for self-contained foreign-rig KOTOR creatures.

This document deliberately does not share the native-template Character Builder
rig state.  It records source evidence and user decisions; generated meshes,
animations, textures, and game files belong in the build directory and are not
embedded in the human-readable project file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ghostrigger_project import stable_project_id, utc_now_iso
from .secondary_motion_profile import SecondaryMotionProfile, default_secondary_motion_profile


CUSTOM_RIGGED_CHARACTER_FILE_TYPE = "ghostrigger.custom_rigged_character"
CUSTOM_RIGGED_CHARACTER_FILE_SUFFIX = ".ghostcharacter.json"
CURRENT_CUSTOM_RIGGED_CHARACTER_SCHEMA_VERSION = 4
CUSTOM_CREATURE_BEHAVIOR_PROFILE_SCHEMA = "ghostrigger.custom_creature_behavior_profile.v1"
BUILDER_MODE_CUSTOM_RIGGED = "custom_rigged_character"
BUILDER_MODE_NATIVE_KOTOR = "native_kotor_character"

CUSTOM_RIGGED_WORKFLOW_STEPS = (
    "source_assets",
    "rig_inspection",
    "scale_ground",
    "animation_library",
    "animation_preparation",
    "materials_uvs",
    "secondary_motion",
    "gameplay",
    "validate_build",
    "install_test",
)


def _clean_hash(value: Any) -> str:
    return str(value or "").strip().lower()


def _clean_resref(value: Any) -> str:
    return str(value or "").strip().lower()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source asset without opening it for write or mutating metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(value: str | Path, project_file: str | Path | None) -> str:
    """Return a portable project-relative path when both locations allow it."""

    text = str(value or "").strip()
    if not text:
        return ""
    source = Path(text)
    if not source.is_absolute() or project_file is None:
        return source.as_posix()
    project_dir = Path(project_file).resolve().parent
    try:
        return Path(os.path.relpath(source.resolve(), project_dir)).as_posix()
    except (OSError, ValueError):
        # Cross-drive Windows paths cannot be made relative. They are allowed
        # in the private authoring project but are excluded from distributable
        # manifests by the packaging layer.
        return str(source)


def resolve_project_path(value: str | Path, project_file: str | Path | None) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute() or project_file is None:
        return path
    return (Path(project_file).resolve().parent / path).resolve()


@dataclass
class SourceAsset:
    path: str = ""
    sha256: str = ""
    role: str = "primary_fbx"
    required: bool = True

    def to_dict(self, project_file: str | Path | None = None) -> dict[str, Any]:
        return {
            "path": portable_path(self.path, project_file),
            "sha256": _clean_hash(self.sha256),
            "role": str(self.role or "source"),
            "required": bool(self.required),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SourceAsset":
        data = data or {}
        return cls(
            path=str(data.get("path") or ""),
            sha256=_clean_hash(data.get("sha256")),
            role=str(data.get("role") or "source"),
            required=bool(data.get("required", True)),
        )


@dataclass
class AnimationMapping:
    source_name: str = ""
    assignment: str = "unassigned"
    exported_name: str = ""
    runtime_id: int | None = None
    confirmed: bool = False
    loop: bool = False
    trim_start: float = 0.0
    trim_end: float | None = None
    playback_speed: float = 1.0
    retime_duration: float | None = None
    root_motion: str = "in_place"
    bake_rate: float = 30.0
    transition_time: float = 0.25
    source_path: str = ""
    source_sha256: str = ""
    retarget_mapping: dict[str, str] = field(default_factory=dict)
    advanced: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, project_file: str | Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = portable_path(self.source_path, project_file)
        payload["source_sha256"] = _clean_hash(self.source_sha256)
        payload["retarget_mapping"] = dict(sorted(self.retarget_mapping.items()))
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AnimationMapping":
        data = data or {}
        runtime_id = data.get("runtime_id")
        trim_end = data.get("trim_end")
        retime_duration = data.get("retime_duration")
        return cls(
            source_name=str(data.get("source_name") or ""),
            assignment=str(data.get("assignment") or "unassigned"),
            exported_name=str(data.get("exported_name") or ""),
            runtime_id=int(runtime_id) if runtime_id is not None else None,
            confirmed=bool(data.get("confirmed")),
            loop=bool(data.get("loop")),
            trim_start=float(data.get("trim_start") or 0.0),
            trim_end=float(trim_end) if trim_end is not None else None,
            playback_speed=float(data.get("playback_speed") or 1.0),
            retime_duration=float(retime_duration) if retime_duration is not None else None,
            root_motion=str(data.get("root_motion") or "in_place"),
            bake_rate=float(data.get("bake_rate") or 30.0),
            transition_time=float(data.get("transition_time") or 0.25),
            source_path=str(data.get("source_path") or ""),
            source_sha256=_clean_hash(data.get("source_sha256")),
            retarget_mapping={str(k): str(v) for k, v in dict(data.get("retarget_mapping") or {}).items()},
            advanced=dict(data.get("advanced") or {}),
        )


@dataclass
class MaterialAssignment:
    material_name: str = ""
    texture_resref: str = ""
    source_texture: str = ""
    source_sha256: str = ""
    output_format: str = "TGA"
    wrap_mode: str = "repeat"
    alpha_mode: str = "opaque"
    txi: str = ""
    uv_channel: int = 0
    flip_vertical_for_kotor: bool = True

    def to_dict(self, project_file: str | Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["texture_resref"] = _clean_resref(self.texture_resref)
        payload["source_texture"] = portable_path(self.source_texture, project_file)
        payload["source_sha256"] = _clean_hash(self.source_sha256)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MaterialAssignment":
        data = data or {}
        return cls(
            material_name=str(data.get("material_name") or ""),
            texture_resref=_clean_resref(data.get("texture_resref")),
            source_texture=str(data.get("source_texture") or ""),
            source_sha256=_clean_hash(data.get("source_sha256")),
            output_format=str(data.get("output_format") or "TGA").upper(),
            wrap_mode=str(data.get("wrap_mode") or "repeat"),
            alpha_mode=str(data.get("alpha_mode") or "opaque"),
            txi=str(data.get("txi") or ""),
            uv_channel=int(data.get("uv_channel") or 0),
            flip_vertical_for_kotor=bool(data.get("flip_vertical_for_kotor", True)),
        )


@dataclass
class CreatureSoundCue:
    """One authoring WAV mapped to a user-facing creature event."""

    cue: str = ""
    source_path: str = ""
    source_sha256: str = ""
    output_resref: str = ""

    def to_dict(self, project_file: str | Path | None = None) -> dict[str, Any]:
        return {
            "cue": str(self.cue or "").strip().lower(),
            "source_path": portable_path(self.source_path, project_file),
            "source_sha256": _clean_hash(self.source_sha256),
            "output_resref": _clean_resref(self.output_resref),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CreatureSoundCue":
        data = data or {}
        return cls(
            cue=str(data.get("cue") or "").strip().lower(),
            source_path=str(data.get("source_path") or ""),
            source_sha256=_clean_hash(data.get("source_sha256")),
            output_resref=_clean_resref(data.get("output_resref")),
        )


@dataclass
class CustomAnimationRegistration:
    name: str = ""
    animation_id: int | None = None
    source_clip: str = ""
    namespace: str = ""
    registry_owner: str = "custom_animation_patch"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CustomAnimationRegistration":
        data = data or {}
        value = data.get("animation_id")
        return cls(
            name=str(data.get("name") or ""),
            animation_id=int(value) if value is not None else None,
            source_clip=str(data.get("source_clip") or ""),
            namespace=str(data.get("namespace") or ""),
            registry_owner=str(data.get("registry_owner") or "custom_animation_patch"),
        )


@dataclass
class CustomRiggedCharacterProject:
    project_id: str = field(default_factory=lambda: stable_project_id("custom_character"))
    creature_name: str = ""
    resource_name: str = ""
    builder_mode: str = BUILDER_MODE_CUSTOM_RIGGED
    target_game: str = "K2"
    primary_fbx: SourceAsset = field(default_factory=SourceAsset)
    external_animation_assets: list[SourceAsset] = field(default_factory=list)
    texture_folder: str = ""
    texture_assets: list[SourceAsset] = field(default_factory=list)
    output_project_folder: str = ""
    recent_paths: list[str] = field(default_factory=list)
    import_coordinate_system: dict[str, Any] = field(
        default_factory=lambda: {
            "source_up": "auto",
            "source_forward": "auto",
            "source_units": "auto",
            "target_up": "+Z",
            "target_forward": "+Y",
        }
    )
    global_scale: float = 1.0
    ground_offset: float = 0.0
    runtime_height_offset: float = 0.0
    runtime_height_source: str = ""
    forward_rotation_degrees: float = 0.0
    pivot_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ground_contact_nodes: list[str] = field(default_factory=list)
    selected_skeleton_root: str = ""
    export_nodes: dict[str, bool] = field(default_factory=dict)
    skin_repair_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "bake_constraints": False,
            "normalize_transforms": False,
            "remove_unused_controls": False,
            "limit_influences": True,
            "max_influences": 4,
            "repair_bind_matrices": False,
        }
    )
    animation_mappings: list[AnimationMapping] = field(default_factory=list)
    material_assignments: list[MaterialAssignment] = field(default_factory=list)
    secondary_motion: dict[str, Any] = field(default_factory=default_secondary_motion_profile)
    creature_sound_cues: list[CreatureSoundCue] = field(default_factory=list)
    appearance_settings: dict[str, Any] = field(default_factory=dict)
    utc_settings: dict[str, Any] = field(default_factory=dict)
    gameplay_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "behavior_preset": "passive_creature",
            "faction": "neutral",
            "movement_rate": "default",
            "prepare_module_placement": False,
            "replace_test_placement": False,
            "test_module_resref": "plcaa",
            "test_placement": {
                "position": [26.0, 30.0, 0.0],
                "bearing": 3.1415927,
            },
        }
    )
    behavior_profile: dict[str, Any] = field(
        default_factory=lambda: {
            "schema": CUSTOM_CREATURE_BEHAVIOR_PROFILE_SCHEMA,
            "template_game": "",
            "template_resref": "",
            "template_display_name": "",
            "template_source": "",
            "template_sha256": "",
            "inherit_template_combat_stats": True,
            "template_snapshot": {},
            "script_hooks": {},
        }
    )
    custom_animation_registrations: list[CustomAnimationRegistration] = field(default_factory=list)
    build_destination: str = ""
    accepted_warning_ids: list[str] = field(default_factory=list)
    last_import_summary: dict[str, Any] = field(default_factory=dict)
    last_validation_result: dict[str, Any] = field(default_factory=dict)
    last_build_result: dict[str, Any] = field(default_factory=dict)
    automatic_repairs: list[dict[str, Any]] = field(default_factory=list)
    workflow_steps: Sequence[str] = field(default_factory=lambda: CUSTOM_RIGGED_WORKFLOW_STEPS)
    native_template_model: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    schema_version: int = CURRENT_CUSTOM_RIGGED_CHARACTER_SCHEMA_VERSION
    file_type: str = CUSTOM_RIGGED_CHARACTER_FILE_TYPE
    _project_file_path: str = field(default="", init=False, repr=False, compare=False)

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a portable authoring path against the opened project file."""

        return resolve_project_path(value, self._project_file_path or None)

    def to_dict(self, project_file: str | Path | None = None) -> dict[str, Any]:
        payload = dict(self.extra_fields)
        payload.update({
            "file_type": self.file_type,
            "schema_version": int(self.schema_version),
            "project_id": self.project_id,
            "builder_mode": self.builder_mode,
            "creature_name": str(self.creature_name or ""),
            "resource_name": _clean_resref(self.resource_name),
            "target_game": str(self.target_game or "K2").upper(),
            "source_assets": {
                "primary_fbx": self.primary_fbx.to_dict(project_file),
                "external_animations": [value.to_dict(project_file) for value in self.external_animation_assets],
                "texture_folder": portable_path(self.texture_folder, project_file),
                "textures": [value.to_dict(project_file) for value in self.texture_assets],
            },
            "output_project_folder": portable_path(self.output_project_folder, project_file),
            "recent_paths": [portable_path(value, project_file) for value in self.recent_paths if value],
            "import_coordinate_system": dict(self.import_coordinate_system),
            "model_placement": {
                "global_scale": float(self.global_scale),
                "ground_offset": float(self.ground_offset),
                "runtime_height_offset": float(self.runtime_height_offset),
                "runtime_height_source": str(self.runtime_height_source or ""),
                "forward_rotation_degrees": float(self.forward_rotation_degrees),
                "pivot_offset": [float(value) for value in self.pivot_offset[:3]],
                "ground_contact_nodes": list(self.ground_contact_nodes),
            },
            "rig": {
                "selected_skeleton_root": self.selected_skeleton_root,
                "export_nodes": dict(sorted(self.export_nodes.items())),
                "skin_repair_settings": dict(self.skin_repair_settings),
                "native_template_model": self.native_template_model,
            },
            "animation_mappings": [value.to_dict(project_file) for value in self.animation_mappings],
            "material_assignments": [value.to_dict(project_file) for value in self.material_assignments],
            "secondary_motion": SecondaryMotionProfile.from_dict(self.secondary_motion).to_dict(),
            "gameplay": {
                "appearance": dict(self.appearance_settings),
                "utc": dict(self.utc_settings),
                "behavior": dict(self.gameplay_settings),
                "behavior_profile": dict(self.behavior_profile),
                "creature_sounds": [value.to_dict(project_file) for value in self.creature_sound_cues],
            },
            "custom_animation_registrations": [asdict(value) for value in self.custom_animation_registrations],
            "build": {
                "destination": portable_path(self.build_destination, project_file),
                "accepted_warning_ids": sorted(set(self.accepted_warning_ids)),
                "last_validation_result": dict(self.last_validation_result),
                "last_build_result": dict(self.last_build_result),
                "automatic_repairs": list(self.automatic_repairs),
            },
            "last_import_summary": dict(self.last_import_summary),
            "workflow_steps": list(self.workflow_steps),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        })
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CustomRiggedCharacterProject":
        data = migrate_custom_rigged_character_payload(raw)
        sources = dict(data.get("source_assets") or {})
        placement = dict(data.get("model_placement") or {})
        rig = dict(data.get("rig") or {})
        gameplay = dict(data.get("gameplay") or {})
        build = dict(data.get("build") or {})
        return cls(
            project_id=str(data.get("project_id") or stable_project_id("custom_character")),
            creature_name=str(data.get("creature_name") or ""),
            resource_name=_clean_resref(data.get("resource_name")),
            builder_mode=str(data.get("builder_mode") or BUILDER_MODE_CUSTOM_RIGGED),
            target_game=str(data.get("target_game") or "K2").upper(),
            primary_fbx=SourceAsset.from_dict(sources.get("primary_fbx")),
            external_animation_assets=[SourceAsset.from_dict(value) for value in sources.get("external_animations") or ()],
            texture_folder=str(sources.get("texture_folder") or ""),
            texture_assets=[SourceAsset.from_dict(value) for value in sources.get("textures") or ()],
            output_project_folder=str(data.get("output_project_folder") or ""),
            recent_paths=[str(value) for value in data.get("recent_paths") or ()],
            import_coordinate_system={
                "source_up": "auto",
                "source_forward": "auto",
                "source_units": "auto",
                "target_up": "+Z",
                "target_forward": "+Y",
                **dict(data.get("import_coordinate_system") or {}),
            },
            global_scale=float(placement.get("global_scale") or 1.0),
            ground_offset=float(placement.get("ground_offset") or 0.0),
            runtime_height_offset=float(placement.get("runtime_height_offset") or 0.0),
            runtime_height_source=str(placement.get("runtime_height_source") or ""),
            forward_rotation_degrees=float(placement.get("forward_rotation_degrees") or 0.0),
            pivot_offset=[float(value) for value in placement.get("pivot_offset") or (0.0, 0.0, 0.0)],
            ground_contact_nodes=[str(value) for value in placement.get("ground_contact_nodes") or ()],
            selected_skeleton_root=str(rig.get("selected_skeleton_root") or ""),
            export_nodes={str(k): bool(v) for k, v in dict(rig.get("export_nodes") or {}).items()},
            skin_repair_settings={
                "bake_constraints": False,
                "normalize_transforms": False,
                "remove_unused_controls": False,
                "limit_influences": True,
                "max_influences": 4,
                "repair_bind_matrices": False,
                **dict(rig.get("skin_repair_settings") or {}),
            },
            native_template_model=str(rig.get("native_template_model") or ""),
            animation_mappings=[AnimationMapping.from_dict(value) for value in data.get("animation_mappings") or ()],
            material_assignments=[MaterialAssignment.from_dict(value) for value in data.get("material_assignments") or ()],
            secondary_motion=SecondaryMotionProfile.from_dict(
                data.get("secondary_motion")
                or default_secondary_motion_profile(target_game=str(data.get("target_game") or "K2"))
            ).to_dict(),
            creature_sound_cues=[
                CreatureSoundCue.from_dict(value)
                for value in gameplay.get("creature_sounds") or ()
            ],
            appearance_settings=dict(gameplay.get("appearance") or {}),
            utc_settings=dict(gameplay.get("utc") or {}),
            gameplay_settings={
                "behavior_preset": "passive_creature",
                "faction": "neutral",
                "movement_rate": "default",
                "prepare_module_placement": False,
                "replace_test_placement": False,
                "test_module_resref": "plcaa",
                "test_placement": {
                    "position": [26.0, 30.0, 0.0],
                    "bearing": 3.1415927,
                },
                **dict(gameplay.get("behavior") or {}),
            },
            behavior_profile={
                "schema": CUSTOM_CREATURE_BEHAVIOR_PROFILE_SCHEMA,
                "template_game": "",
                "template_resref": "",
                "template_display_name": "",
                "template_source": "",
                "template_sha256": "",
                "inherit_template_combat_stats": True,
                "template_snapshot": {},
                "script_hooks": {},
                **dict(gameplay.get("behavior_profile") or {}),
            },
            custom_animation_registrations=[CustomAnimationRegistration.from_dict(value) for value in data.get("custom_animation_registrations") or ()],
            build_destination=str(build.get("destination") or ""),
            accepted_warning_ids=[str(value) for value in build.get("accepted_warning_ids") or ()],
            last_import_summary=dict(data.get("last_import_summary") or {}),
            last_validation_result=dict(build.get("last_validation_result") or {}),
            last_build_result=dict(build.get("last_build_result") or {}),
            automatic_repairs=list(build.get("automatic_repairs") or ()),
            workflow_steps=tuple(data.get("workflow_steps") or CUSTOM_RIGGED_WORKFLOW_STEPS),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            metadata=dict(data.get("metadata") or {}),
            extra_fields={
                str(key): value
                for key, value in data.items()
                if key not in {
                    "file_type", "schema_version", "project_id", "builder_mode", "creature_name",
                    "resource_name", "target_game", "source_assets", "output_project_folder",
                    "recent_paths", "import_coordinate_system", "model_placement", "rig",
                    "animation_mappings", "material_assignments", "secondary_motion", "gameplay",
                    "custom_animation_registrations", "build", "last_import_summary", "workflow_steps",
                    "created_at", "updated_at", "metadata",
                }
            },
            schema_version=int(data.get("schema_version") or 0),
            file_type=str(data.get("file_type") or ""),
        )

    def resolved_source_status(self, project_file: str | Path | None) -> list[dict[str, Any]]:
        """Describe missing/moved/hash-changed inputs without modifying them."""

        assets = [self.primary_fbx, *self.external_animation_assets, *self.texture_assets]
        result: list[dict[str, Any]] = []
        for asset in assets:
            if not asset.path:
                if asset.required:
                    result.append({"role": asset.role, "path": "", "status": "missing_path"})
                continue
            resolved = resolve_project_path(asset.path, project_file)
            status = "available" if resolved.is_file() else "missing"
            actual_hash = ""
            if status == "available" and asset.sha256:
                actual_hash = sha256_file(resolved)
                if actual_hash != asset.sha256:
                    status = "hash_changed"
            result.append({
                "role": asset.role,
                "path": asset.path,
                "resolved_path": str(resolved),
                "status": status,
                "expected_sha256": asset.sha256,
                "actual_sha256": actual_hash,
            })
        for cue in self.creature_sound_cues:
            if not cue.source_path:
                continue
            resolved = resolve_project_path(cue.source_path, project_file)
            status = "available" if resolved.is_file() else "missing"
            actual_hash = ""
            if status == "available" and cue.source_sha256:
                actual_hash = sha256_file(resolved)
                if actual_hash != cue.source_sha256:
                    status = "hash_changed"
            result.append({
                "role": f"creature_sound:{cue.cue}",
                "path": cue.source_path,
                "resolved_path": str(resolved),
                "status": status,
                "expected_sha256": cue.source_sha256,
                "actual_sha256": actual_hash,
            })
        return result


def migrate_custom_rigged_character_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate supported historical drafts without silently dropping fields."""

    if not isinstance(raw, Mapping):
        raise ValueError("Custom character project must be a JSON object.")
    data = dict(raw)
    version = int(data.get("schema_version") or 0)
    if version > CURRENT_CUSTOM_RIGGED_CHARACTER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported custom character schema version {version}.")
    if version == 0:
        primary = data.pop("source_fbx", data.pop("fbx_path", ""))
        animations = data.pop("animation_files", [])
        data.setdefault("source_assets", {
            "primary_fbx": {"path": primary, "role": "primary_fbx", "required": True},
            "external_animations": [
                {"path": value, "role": "external_animation", "required": False}
                for value in animations
            ],
            "texture_folder": data.pop("texture_folder", ""),
            "textures": [],
        })
        data.setdefault("resource_name", data.pop("resref", ""))
        data.setdefault("target_game", data.pop("game", "K2"))
        data.setdefault("builder_mode", BUILDER_MODE_CUSTOM_RIGGED)
        data.setdefault("model_placement", {
            "global_scale": data.pop("global_scale", 1.0),
            "ground_offset": data.pop("ground_offset", 0.0),
        })
        data.setdefault("rig", {})
        data.setdefault("gameplay", {})
        data.setdefault("build", {})
        data.setdefault("animation_mappings", [])
        data.setdefault("material_assignments", [])
        data.setdefault("custom_animation_registrations", [])
        data.setdefault("workflow_steps", list(CUSTOM_RIGGED_WORKFLOW_STEPS))
        data.setdefault("file_type", CUSTOM_RIGGED_CHARACTER_FILE_TYPE)
        data["schema_version"] = 1
        version = 1
    if version == 1:
        gameplay = data.setdefault("gameplay", {})
        gameplay.setdefault("behavior_profile", {
            "schema": CUSTOM_CREATURE_BEHAVIOR_PROFILE_SCHEMA,
            "template_game": "",
            "template_resref": "",
            "template_display_name": "",
            "template_source": "",
            "template_sha256": "",
            "inherit_template_combat_stats": True,
            "template_snapshot": {},
            "script_hooks": {},
        })
        data["schema_version"] = 2
        version = 2
    if version == 2:
        data.setdefault("gameplay", {}).setdefault("creature_sounds", [])
        data["schema_version"] = 3
        version = 3
    if version == 3:
        data.setdefault(
            "secondary_motion",
            default_secondary_motion_profile(target_game=str(data.get("target_game") or "K2")),
        )
        steps = [str(step) for step in data.get("workflow_steps") or CUSTOM_RIGGED_WORKFLOW_STEPS]
        if "secondary_motion" not in steps:
            try:
                steps.insert(steps.index("materials_uvs") + 1, "secondary_motion")
            except ValueError:
                steps.append("secondary_motion")
        data["workflow_steps"] = steps
        data["schema_version"] = 4
    if data.get("file_type") != CUSTOM_RIGGED_CHARACTER_FILE_TYPE:
        raise ValueError("File is not a GhostRigger custom-rigged character project.")
    if int(data.get("schema_version") or 0) != CURRENT_CUSTOM_RIGGED_CHARACTER_SCHEMA_VERSION:
        raise ValueError("Custom character project schema is not supported.")
    return data


def save_custom_rigged_character_project(
    project: CustomRiggedCharacterProject,
    path: str | Path,
    *,
    allow_replace_different_project: bool = False,
) -> Path:
    """Atomically save, refusing to silently replace a different project."""

    target = Path(path)
    if target.exists() and not allow_replace_different_project:
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}") from exc
        if str(existing.get("project_id") or "") != project.project_id:
            raise FileExistsError(f"Refusing to overwrite a different project: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    project.updated_at = utc_now_iso()
    payload = (json.dumps(project.to_dict(target), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        project._project_file_path = str(target.resolve())
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return target


def load_custom_rigged_character_project(path: str | Path) -> CustomRiggedCharacterProject:
    target = Path(path).resolve()
    project = CustomRiggedCharacterProject.from_dict(json.loads(target.read_text(encoding="utf-8")))
    project._project_file_path = str(target)
    return project


__all__ = [
    "AnimationMapping",
    "BUILDER_MODE_CUSTOM_RIGGED",
    "BUILDER_MODE_NATIVE_KOTOR",
    "CUSTOM_CREATURE_BEHAVIOR_PROFILE_SCHEMA",
    "CURRENT_CUSTOM_RIGGED_CHARACTER_SCHEMA_VERSION",
    "CUSTOM_RIGGED_CHARACTER_FILE_SUFFIX",
    "CUSTOM_RIGGED_CHARACTER_FILE_TYPE",
    "CUSTOM_RIGGED_WORKFLOW_STEPS",
    "CustomAnimationRegistration",
    "CreatureSoundCue",
    "CustomRiggedCharacterProject",
    "MaterialAssignment",
    "SourceAsset",
    "load_custom_rigged_character_project",
    "migrate_custom_rigged_character_payload",
    "portable_path",
    "resolve_project_path",
    "save_custom_rigged_character_project",
    "sha256_file",
]
