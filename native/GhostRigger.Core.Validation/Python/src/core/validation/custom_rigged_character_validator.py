"""Deterministic preflight rules for self-contained foreign-rig creatures."""

from __future__ import annotations

import hashlib
import io
import math
import re
import wave
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Iterable, Mapping, Sequence

from src.core.project.custom_rigged_character_project import CustomRiggedCharacterProject
from src.core.validation.secondary_motion_validator import validate_secondary_motion_profile


_RESREF_RE = re.compile(r"^[a-z0-9_]{1,16}$")
_NODE_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_TEXTURE_FORMATS = {"bmp", "dds", "jpg", "jpeg", "png", "tga", "tif", "tiff", "tpc"}
_UNSUPPORTED_NODE_KINDS = {"constraint", "ik", "control", "controller", "procedural"}
_UTC_SCRIPT_HOOK_FIELDS = {
    "ScriptHeartbeat", "ScriptOnNotice", "ScriptSpellAt", "ScriptAttacked",
    "ScriptDamaged", "ScriptDisturbed", "ScriptEndRound", "ScriptEndDialogu",
    "ScriptDialogue", "ScriptSpawn", "ScriptRested", "ScriptDeath",
    "ScriptUserDefine", "ScriptOnBlocked",
}
_VERIFIED_CREATURE_ALIASES = {
    "cpause1", "cpause2", "cwalk", "crun", "chturnl", "chturnr",
    "m0a1", "m0a2", "g0a1", "g0a2", "cdamages", "cdodgeg",
    "creadyr", "creadyrtw", "cwalkinj", "ckdbck", "ckdbcklp",
    "cgustandb", "cdie", "cdead", "ctaunt", "cvictory",
}
_CREATURE_SOUND_CUES = {"roar", "attack", "hurt", "guard", "blocked", "idle", "death"}


@dataclass(frozen=True)
class CustomRigValidationIssue:
    severity: str
    code: str
    message: str
    why: str
    field: str = ""
    automatic_fix: str = ""
    fix_effect: str = ""
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "why": self.why,
            "field": self.field,
            "automatic_fix": self.automatic_fix,
            "fix_effect": self.fix_effect,
            "details": dict(self.details),
        }


@dataclass
class CustomRigValidationReport:
    issues: list[CustomRigValidationIssue] = dataclass_field(default_factory=list)
    summary: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def build_ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        why: str,
        *,
        field: str = "",
        automatic_fix: str = "",
        fix_effect: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.issues.append(CustomRigValidationIssue(
            severity=severity,
            code=code,
            message=message,
            why=why,
            field=field,
            automatic_fix=automatic_fix,
            fix_effect=fix_effect,
            details=dict(details or {}),
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ghostrigger.custom_rigged_character_validation.v1",
            "build_ready": self.build_ready,
            "summary": dict(self.summary),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class RigNodeSnapshot:
    name: str
    parent: str = ""
    exported: bool = True
    deform: bool = True
    kind: str = "bone"
    transform: tuple[float, ...] = ()
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bind_matrix: tuple[float, ...] = ()
    expected_bind_matrix: tuple[float, ...] = ()

    @classmethod
    def from_value(cls, value: "RigNodeSnapshot | Mapping[str, Any]") -> "RigNodeSnapshot":
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value.get("name") or ""),
            parent=str(value.get("parent") or ""),
            exported=bool(value.get("exported", True)),
            deform=bool(value.get("deform", True)),
            kind=str(value.get("kind") or "bone"),
            transform=tuple(float(item) for item in value.get("transform") or ()),
            scale=tuple(float(item) for item in value.get("scale") or (1.0, 1.0, 1.0))[:3],  # type: ignore[arg-type]
            bind_matrix=tuple(float(item) for item in value.get("bind_matrix") or ()),
            expected_bind_matrix=tuple(float(item) for item in value.get("expected_bind_matrix") or ()),
        )


@dataclass(frozen=True)
class AnimationTrackSnapshot:
    node_name: str
    positions: tuple[tuple[float, ...], ...] = ()
    rotations: tuple[tuple[float, float, float, float], ...] = ()
    scales: tuple[tuple[float, ...], ...] = ()

    @classmethod
    def from_value(cls, value: "AnimationTrackSnapshot | Mapping[str, Any]") -> "AnimationTrackSnapshot":
        if isinstance(value, cls):
            return value
        return cls(
            node_name=str(value.get("node_name") or value.get("node") or ""),
            positions=tuple(tuple(float(item) for item in row) for row in value.get("positions") or ()),
            rotations=tuple(tuple(float(item) for item in row) for row in value.get("rotations") or ()),  # type: ignore[arg-type]
            scales=tuple(tuple(float(item) for item in row) for row in value.get("scales") or ()),
        )


@dataclass(frozen=True)
class AnimationClipSnapshot:
    name: str
    duration: float
    tracks: tuple[AnimationTrackSnapshot, ...] = ()
    loop: bool = False
    root_positions: tuple[tuple[float, float, float], ...] = ()
    source_skeleton_fingerprint: str = ""

    @classmethod
    def from_value(cls, value: "AnimationClipSnapshot | Mapping[str, Any]") -> "AnimationClipSnapshot":
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value.get("name") or ""),
            duration=float(value.get("duration") or 0.0),
            tracks=tuple(AnimationTrackSnapshot.from_value(row) for row in value.get("tracks") or ()),
            loop=bool(value.get("loop")),
            root_positions=tuple(tuple(float(item) for item in row) for row in value.get("root_positions") or ()),  # type: ignore[arg-type]
            source_skeleton_fingerprint=str(value.get("source_skeleton_fingerprint") or ""),
        )


@dataclass(frozen=True)
class MaterialSnapshot:
    material_name: str
    texture_resref: str = ""
    source_format: str = ""
    texture_size: tuple[int, int] = (0, 0)
    uvs: tuple[tuple[float, float], ...] = ()
    wrap_mode: str = "repeat"
    has_alpha: bool = False
    alpha_mode: str = "opaque"
    source_texture: str = ""

    @classmethod
    def from_value(cls, value: "MaterialSnapshot | Mapping[str, Any]") -> "MaterialSnapshot":
        if isinstance(value, cls):
            return value
        size = tuple(int(item) for item in value.get("texture_size") or (0, 0))
        return cls(
            material_name=str(value.get("material_name") or value.get("name") or ""),
            texture_resref=str(value.get("texture_resref") or "").lower(),
            source_format=str(value.get("source_format") or "").lower().lstrip("."),
            texture_size=(size + (0, 0))[:2],  # type: ignore[arg-type]
            uvs=tuple(tuple(float(item) for item in row[:2]) for row in value.get("uvs") or ()),  # type: ignore[arg-type]
            wrap_mode=str(value.get("wrap_mode") or "repeat"),
            has_alpha=bool(value.get("has_alpha")),
            alpha_mode=str(value.get("alpha_mode") or "opaque"),
            source_texture=str(value.get("source_texture") or ""),
        )


@dataclass
class CustomRiggedCharacterSnapshot:
    nodes: list[RigNodeSnapshot] = dataclass_field(default_factory=list)
    vertex_influences: list[list[tuple[str, float]]] = dataclass_field(default_factory=list)
    animations: list[AnimationClipSnapshot] = dataclass_field(default_factory=list)
    materials: list[MaterialSnapshot] = dataclass_field(default_factory=list)
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lowest_contact_height: float | None = None
    root_height: float | None = None
    runtime_height_offset: float = 0.0
    runtime_height_source: str = ""
    source_unit_scale: float = 1.0
    source_forward: str = ""
    expected_forward: str = "+Y"
    skeleton_fingerprint: str = ""
    available_skeleton_roots: tuple[str, ...] = ()
    skeleton_selection_required: bool = False


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate_resource_name(value: str) -> bool:
    return bool(_RESREF_RE.fullmatch(str(value or "").strip().lower()))


def normalized_influences(
    influences: Iterable[tuple[str, float]], *, max_influences: int = 4
) -> list[tuple[str, float]]:
    """Combine, cap, and normalize one vertex without inventing bone names."""

    combined: dict[str, float] = {}
    for name, weight in influences:
        text = str(name or "")
        value = float(weight)
        if text and math.isfinite(value) and value > 0.0:
            combined[text] = combined.get(text, 0.0) + value
    kept = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:max(1, int(max_influences))]
    total = sum(value for _name, value in kept)
    if total <= 0.0:
        return []
    return [(name, value / total) for name, value in kept]


def ground_offset_for_contacts(contact_heights: Iterable[float]) -> float:
    values = [float(value) for value in contact_heights if math.isfinite(float(value))]
    if not values:
        raise ValueError("At least one finite contact point is required.")
    return -min(values)


def axis_scale_point(
    point: Sequence[float], *, scale: float, source_up: str, source_forward: str
) -> tuple[float, float, float]:
    """Convert common Y-up/Z-up source frames into KOTOR +Z up, +Y forward."""

    if len(point) < 3 or not _finite(point[:3]) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Point and scale must be finite; scale must be positive.")
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    up = str(source_up or "+Z").upper()
    forward = str(source_forward or "+Y").upper()
    if up in {"+Y", "Y"}:
        x, y, z = x, -z, y
    elif up == "-Y":
        x, y, z = x, z, -y
    elif up == "-Z":
        x, y, z = x, -y, -z
    if forward == "-Y":
        x, y = -x, -y
    elif forward in {"+X", "X"}:
        x, y = -y, x
    elif forward == "-X":
        x, y = y, -x
    return x * scale, y * scale, z * scale


def quaternion_continuity(
    keys: Iterable[Sequence[float]],
) -> tuple[list[tuple[float, float, float, float]], int]:
    """Normalize keys and flip signs so adjacent quaternions share a hemisphere."""

    result: list[tuple[float, float, float, float]] = []
    flips = 0
    for raw in keys:
        if len(raw) < 4 or not _finite(raw[:4]):
            raise ValueError("Quaternion keys must contain four finite values.")
        length = math.sqrt(sum(float(value) ** 2 for value in raw[:4]))
        if length <= 1.0e-12:
            raise ValueError("Zero-length quaternion key is invalid.")
        current = tuple(float(value) / length for value in raw[:4])
        if result and sum(a * b for a, b in zip(result[-1], current)) < 0.0:
            current = tuple(-value for value in current)
            flips += 1
        result.append(current)  # type: ignore[arg-type]
    return result, flips


class CustomRiggedCharacterValidator:
    """Validate project, hierarchy, skin, animation, material, and gameplay gates."""

    def validate(
        self,
        project: CustomRiggedCharacterProject,
        snapshot: CustomRiggedCharacterSnapshot | None = None,
        *,
        occupied_animation_ids: Iterable[int] = (),
    ) -> CustomRigValidationReport:
        report = CustomRigValidationReport()
        self._validate_project(project, report, occupied_animation_ids)
        if snapshot is None:
            report.add(
                "error", "source_not_imported", "Import the source FBX before building.",
                "KOTOR output cannot be created until the selected hierarchy, mesh, weights, and actions are known.",
                field="source_assets.primary_fbx", automatic_fix="Import and inspect",
                fix_effect="Reads the FBX without changing it and creates an in-memory conversion snapshot.",
            )
            return report
        self._validate_hierarchy(snapshot, report)
        self._validate_secondary_motion_drivers(project, snapshot, report)
        self._validate_skin(snapshot, project, report)
        self._validate_placement(snapshot, project, report)
        self._validate_animations(snapshot, project, report)
        self._validate_materials(snapshot, report)
        report.summary = {
            "node_count": len(snapshot.nodes),
            "deform_node_count": sum(node.deform for node in snapshot.nodes),
            "vertex_count": len(snapshot.vertex_influences),
            "animation_count": len(snapshot.animations),
            "material_count": len(snapshot.materials),
            "errors": sum(issue.severity == "error" for issue in report.issues),
            "warnings": sum(issue.severity == "warning" for issue in report.issues),
            "information": sum(issue.severity == "information" for issue in report.issues),
        }
        return report

    def _validate_project(
        self,
        project: CustomRiggedCharacterProject,
        report: CustomRigValidationReport,
        occupied_animation_ids: Iterable[int],
    ) -> None:
        if project.builder_mode != "custom_rigged_character":
            report.add("error", "wrong_builder_mode", "This project is not a Custom Rigged Character project.",
                       "Native and custom rig projects have different skeleton authority contracts.", field="builder_mode")
        if project.target_game not in {"K1", "K2"}:
            report.add("error", "invalid_target_game", "Choose KOTOR I or KOTOR II.",
                       "The model, 2DA, UTC, and patch target must agree on one game.", field="target_game")
        if not validate_resource_name(project.resource_name):
            report.add(
                "error", "invalid_resource_name", "Use 1–16 letters, numbers, or underscores for the KOTOR resource name.",
                "KOTOR resource lookups cannot safely address the requested filename.", field="resource_name",
                automatic_fix="Suggest a cleaned resource name",
                fix_effect="Only the output resource name changes; the skeleton is never renamed.",
            )
        if not project.creature_name.strip():
            report.add("warning", "missing_creature_name", "Give the creature a friendly project name.",
                       "The name helps identify reports and UTC output but does not block model conversion.", field="creature_name")
        if project.native_template_model:
            report.add("error", "native_template_forbidden", "Remove the native KOTOR skeleton template from this project.",
                       "A self-contained foreign-rig creature must preserve its selected hierarchy instead of inheriting a humanoid template.",
                       field="rig.native_template_model")
        if project.global_scale <= 0 or not math.isfinite(project.global_scale):
            report.add("error", "invalid_global_scale", "Global scale must be a finite positive value.",
                       "Zero, negative, or non-finite scale cannot produce a coherent bind pose.", field="model_placement.global_scale")
        secondary_report = validate_secondary_motion_profile(
            project.secondary_motion,
            target_kind="custom_character",
        )
        for issue in secondary_report.issues:
            severity = "warning" if issue.severity == "write_blocking" else "error"
            if issue.code == "missing_exact_model_identity":
                severity = "information"
            report.add(
                severity,
                f"secondary_motion_{issue.code}",
                issue.message,
                (
                    "The exact MDL/MDX hashes are generated and revalidated during Build."
                    if issue.code == "missing_exact_model_identity"
                    else "Secondary-motion export and runtime promotion fail closed independently of ordinary character output."
                ),
                field=f"secondary_motion.{issue.target}" if issue.target else "secondary_motion",
            )
        used = {int(value) for value in occupied_animation_ids}
        local: set[int] = set()
        names: set[str] = set()
        for index, registration in enumerate(project.custom_animation_registrations):
            field_prefix = f"custom_animation_registrations.{index}"
            name = registration.name.casefold()
            if not name or name in names:
                report.add("error", "duplicate_custom_animation_name", "Custom runtime animation names must be unique and namespaced.",
                           "The patch registry must resolve one action for each additive name.", field=f"{field_prefix}.name")
            names.add(name)
            if registration.animation_id is None:
                report.add("error", "missing_custom_animation_id", "Allocate an additive ID for this custom action.",
                           "A genuinely new action needs a non-conflicting registry ID before scripts can request it.",
                           field=f"{field_prefix}.animation_id", automatic_fix="Allocate the next free ID",
                           fix_effect="Adds a project-owned registry record; no vanilla slot is replaced.")
            elif registration.animation_id in used or registration.animation_id in local:
                report.add("error", "custom_animation_id_collision", f"Animation ID {registration.animation_id} is already occupied.",
                           "Reusing an occupied ID can request the wrong action or overwrite another mod's registration.",
                           field=f"{field_prefix}.animation_id", automatic_fix="Allocate the next free ID",
                           fix_effect="Changes only this additive registry ID and generated metadata.")
            else:
                local.add(registration.animation_id)
        sound_cues: set[str] = set()
        sound_resrefs: set[str] = set()
        for index, cue in enumerate(project.creature_sound_cues):
            field_prefix = f"gameplay.creature_sounds.{index}"
            cue_name = str(cue.cue or "").strip().casefold()
            if cue_name not in _CREATURE_SOUND_CUES:
                report.add(
                    "error", "unknown_creature_sound_cue", f"Creature sound cue '{cue_name}' is not supported.",
                    "Each sound must map to a known native SSF slot so it cannot replace an unrelated AI event hook.",
                    field=f"{field_prefix}.cue",
                )
            elif cue_name in sound_cues:
                report.add(
                    "error", "duplicate_creature_sound_cue", f"Creature sound cue '{cue_name}' is assigned more than once.",
                    "One deterministic WAV must be selected for each native creature sound cue.", field=f"{field_prefix}.cue",
                )
            sound_cues.add(cue_name)
            if cue.output_resref and not validate_resource_name(cue.output_resref):
                report.add(
                    "error", "invalid_creature_sound_resref", "A creature sound has an invalid KOTOR resource name.",
                    "KOTOR sound lookups require 1–16 lowercase letters, numbers, or underscores.",
                    field=f"{field_prefix}.output_resref",
                )
            elif cue.output_resref and cue.output_resref in sound_resrefs:
                report.add(
                    "error", "duplicate_creature_sound_resref", f"Sound resource '{cue.output_resref}' is reused.",
                    "Each event needs an unambiguous packaged WAV resource.", field=f"{field_prefix}.output_resref",
                )
            sound_resrefs.add(cue.output_resref)
            source = project.resolve_path(cue.source_path)
            if not cue.source_path or not source.is_file():
                report.add(
                    "error", "missing_creature_sound", f"Choose an existing WAV for the {cue_name or 'creature'} sound.",
                    "The build will not silently omit an assigned creature sound.", field=f"{field_prefix}.source_path",
                )
                continue
            payload = source.read_bytes()
            actual_hash = hashlib.sha256(payload).hexdigest()
            if cue.source_sha256 and cue.source_sha256 != actual_hash:
                report.add(
                    "error", "changed_creature_sound", f"The selected {cue_name} WAV changed after it was chosen.",
                    "Review the changed source before it enters a distributable package.", field=f"{field_prefix}.source_sha256",
                )
            try:
                with wave.open(io.BytesIO(payload), "rb") as stream:
                    valid_pcm = (
                        stream.getcomptype() == "NONE"
                        and stream.getnchannels() == 1
                        and stream.getsampwidth() == 2
                        and stream.getframerate() in {11025, 22050, 44100}
                        and stream.getnframes() > 0
                    )
            except (EOFError, wave.Error):
                valid_pcm = False
            if not valid_pcm:
                report.add(
                    "error", "unsupported_creature_sound", f"The {cue_name} sound is not a KOTOR-ready PCM WAV.",
                    "Use an uncompressed mono 16-bit WAV at 11025, 22050, or 44100 Hz.",
                    field=f"{field_prefix}.source_path",
                )

    def _validate_secondary_motion_drivers(
        self,
        project: CustomRiggedCharacterProject,
        snapshot: CustomRiggedCharacterSnapshot,
        report: CustomRigValidationReport,
    ) -> None:
        """Bind authored driver names to this custom rig, never to PFBNM names."""

        parents = {node.name.casefold(): node.parent.casefold() for node in snapshot.nodes}
        for region_index, region in enumerate(project.secondary_motion.get("regions") or ()):
            for driver_index, driver in enumerate(region.get("drivers") or ()):
                name = str(driver.get("name") or "").strip()
                parent = str(driver.get("parent_name") or "").strip()
                field = f"secondary_motion.regions.{region_index}.drivers.{driver_index}"
                actual_parent = parents.get(name.casefold())
                if actual_parent is None:
                    report.add(
                        "error",
                        "secondary_motion_driver_missing",
                        f"Secondary-motion driver '{name}' is not in the imported KOTOR hierarchy.",
                        "Custom rigs may use any bone names, but every selected driver must resolve exactly in the final hierarchy.",
                        field=f"{field}.name",
                    )
                elif parent and actual_parent != parent.casefold():
                    report.add(
                        "error",
                        "secondary_motion_driver_parent_changed",
                        f"Driver '{name}' is no longer parented to '{parent}'.",
                        "The compiled runtime selector requires exact named parent and rest-transform agreement.",
                        field=f"{field}.parent_name",
                    )

    def _validate_hierarchy(self, snapshot: CustomRiggedCharacterSnapshot, report: CustomRigValidationReport) -> None:
        if snapshot.skeleton_selection_required:
            report.add(
                "error",
                "skeleton_selection_required",
                "Choose which deform hierarchy should be exported.",
                "The FBX contains more than one armature or root, so Ghost Studio will not silently choose or merge them.",
                field="rig.selected_skeleton_root",
                automatic_fix="Select one listed hierarchy and import again",
                fix_effect="Keeps only the selected root and its descendants; no bone is renamed.",
                details={"available_roots": list(snapshot.available_skeleton_roots)},
            )
        nodes = snapshot.nodes
        names = [node.name for node in nodes]
        counts = {name: names.count(name) for name in set(names)}
        for name, count in sorted(counts.items()):
            if count > 1:
                report.add("error", "duplicate_bone_name", f"Bone or node name '{name}' appears {count} times.",
                           "Odyssey animation and skin tracks address nodes by name, so duplicates are ambiguous.",
                           field="rig.hierarchy", details={"name": name, "count": count})
        valid_names = set(names)
        for node in nodes:
            if not _NODE_NAME_RE.fullmatch(node.name):
                report.add("error", "invalid_node_name", f"Node name '{node.name}' is not export-safe.",
                           "Odyssey node names must be short, addressable identifiers.", field=f"rig.nodes.{node.name}",
                           automatic_fix="Suggest a unique cleaned name",
                           fix_effect="Renames only this invalid node and its matching skin/animation references; it does not map to a humanoid rig.")
            if node.parent and node.parent not in valid_names:
                report.add("error", "missing_parent", f"Node '{node.name}' refers to missing parent '{node.parent}'.",
                           "Every exported node must participate in one coherent parent/child hierarchy.", field=f"rig.nodes.{node.name}.parent")
            if not _finite((*node.transform, *node.scale, *node.bind_matrix, *node.expected_bind_matrix)):
                report.add("error", "non_finite_transform", f"Node '{node.name}' contains a non-finite transform.",
                           "KOTOR cannot evaluate NaN or infinite bind and animation transforms.", field=f"rig.nodes.{node.name}.transform",
                           automatic_fix="Normalize finite transform components when unambiguous",
                           fix_effect="Replaces invalid components with verified bind-pose values and records the repair.")
            if any(value <= 0 for value in node.scale):
                report.add("warning", "negative_or_zero_scale", f"Node '{node.name}' has zero or negative scale.",
                           "Reflections and collapsed axes commonly disagree with Odyssey bind math.", field=f"rig.nodes.{node.name}.scale",
                           automatic_fix="Bake transforms",
                           fix_effect="Bakes the visible pose into ordinary positive-scale local transforms.")
            if node.kind.casefold() in _UNSUPPORTED_NODE_KINDS:
                report.add("warning", "unsupported_control_object", f"'{node.name}' is a {node.kind} object, not a deform node.",
                           "KOTOR does not evaluate FBX constraints, IK solvers, control rigs, or procedural objects.", field=f"rig.nodes.{node.name}",
                           automatic_fix="Bake motion and remove the unused control",
                           fix_effect="Samples its visible result onto deform bones, then excludes this control from export.")
            if node.bind_matrix and node.expected_bind_matrix:
                error = max(abs(a - b) for a, b in zip(node.bind_matrix, node.expected_bind_matrix))
                if error > 1.0e-4:
                    report.add("error", "bind_pose_mismatch", f"Node '{node.name}' bind pose disagrees with its skin matrix.",
                               "The skeleton and mesh must use the same bind transforms or vertices deform from the wrong frame.",
                               field=f"rig.nodes.{node.name}.bind_matrix",
                               automatic_fix="Repair the bind matrix" if error < 1.0e-2 else "",
                               fix_effect="Uses the matching skeleton bind transform; vertex positions are not rewritten." if error < 1.0e-2 else "",
                               details={"maximum_component_error": error})
        roots = [node.name for node in nodes if not node.parent]
        if len(roots) != 1:
            report.add("error", "invalid_root_count", f"The selected export hierarchy has {len(roots)} roots; choose one coherent deform root.",
                       "A self-contained Odyssey model needs one hierarchy root.", field="rig.selected_skeleton_root",
                       automatic_fix="Consolidate under a selected deform root" if roots else "",
                       fix_effect="Adds or selects one ordinary export root without renaming the remaining hierarchy." if roots else "",
                       details={"roots": roots})
        parent_by_name = {node.name: node.parent for node in nodes}
        for start in names:
            visited: set[str] = set()
            current = start
            while current:
                if current in visited:
                    report.add("error", "hierarchy_cycle", f"The hierarchy loops through '{current}'.",
                               "Parent cycles cannot be traversed or serialized as an Odyssey tree.", field=f"rig.nodes.{start}.parent")
                    break
                visited.add(current)
                current = parent_by_name.get(current, "")

    def _validate_skin(
        self, snapshot: CustomRiggedCharacterSnapshot, project: CustomRiggedCharacterProject,
        report: CustomRigValidationReport,
    ) -> None:
        exported = {node.name for node in snapshot.nodes if node.exported}
        deform = {node.name for node in snapshot.nodes if node.deform}
        max_influences = int(project.skin_repair_settings.get("max_influences", 4) or 4)
        for index, row in enumerate(snapshot.vertex_influences):
            positive = [(str(name), float(weight)) for name, weight in row if math.isfinite(float(weight)) and float(weight) > 0]
            if not positive:
                report.add("error", "unweighted_vertex", f"Vertex {index} has no positive skin influence.",
                           "An animated skin vertex must follow at least one exported deform node.", field=f"skin.vertices.{index}")
                continue
            missing = sorted({name for name, _weight in positive if name not in exported or name not in deform})
            if missing:
                report.add("error", "missing_weighted_bone", f"Vertex {index} uses a bone that is not exported: {', '.join(missing)}.",
                           "Every skin influence must resolve to the same exported hierarchy used by animations.", field=f"skin.vertices.{index}",
                           details={"missing_nodes": missing})
            total = sum(weight for _name, weight in positive)
            if abs(total - 1.0) > 1.0e-4:
                report.add("warning", "unnormalized_influences", f"Vertex {index} weights total {total:.6g}, not 1.",
                           "Normalized weights are required for predictable deformation.", field=f"skin.vertices.{index}",
                           automatic_fix="Normalize this vertex's weights",
                           fix_effect="Preserves the relative influence strengths and changes only their total.")
            if len(positive) > max_influences:
                report.add("warning", "excessive_influences", f"Vertex {index} has {len(positive)} influences; the project limit is {max_influences}.",
                           "The selected Odyssey output path has a bounded per-vertex influence budget.", field=f"skin.vertices.{index}",
                           automatic_fix=f"Keep the strongest {max_influences} and renormalize",
                           fix_effect="Drops only the weakest influences and records their names and weights.")
        weighted = {name for row in snapshot.vertex_influences for name, weight in row if float(weight) > 0}
        exported_roots = {
            node.name for node in snapshot.nodes
            if node.exported and not str(node.parent or "")
        }
        for node in snapshot.nodes:
            if node.exported is False and node.name in weighted:
                report.add("error", "weighted_node_not_exported", f"Weighted node '{node.name}' is marked not to export.",
                           "Discarding it would silently change or lose skin weights.", field=f"rig.export_nodes.{node.name}")
            elif node.exported and node.name not in weighted and node.deform and node.name in exported_roots:
                report.add(
                    "information",
                    "authoring_root_becomes_model_root",
                    f"Authoring root '{node.name}' has no direct skin weights and will become "
                    f"the Odyssey model root '{project.resource_name}'.",
                    "This is the expected self-contained creature layout; Ghost Studio preserves "
                    "its transform and animation data without exporting a second wrapper node.",
                    field=f"rig.nodes.{node.name}",
                )
            elif node.exported and node.name not in weighted and node.deform:
                report.add("information", "unused_deform_node", f"Deform node '{node.name}' has no skin weights.",
                           "It may still be an animation parent or attachment point, so Ghost Studio will not remove it silently.",
                           field=f"rig.nodes.{node.name}")

    def _validate_placement(
        self, snapshot: CustomRiggedCharacterSnapshot, project: CustomRiggedCharacterProject,
        report: CustomRigValidationReport,
    ) -> None:
        if not _finite(snapshot.dimensions) or any(value < 0 for value in snapshot.dimensions):
            report.add("error", "invalid_dimensions", "Creature dimensions are not finite and positive.",
                       "Scale and collision previews require a coherent model-space bounding box.", field="model_placement")
        elif max(snapshot.dimensions, default=0.0) > 100.0 or (max(snapshot.dimensions, default=0.0) and max(snapshot.dimensions) < 0.01):
            report.add("warning", "extreme_dimensions", f"Creature dimensions {snapshot.dimensions} are unusual for KOTOR model units.",
                       "The source may use a substantially different unit scale.", field="model_placement.global_scale",
                       automatic_fix="Apply an explicit global scale",
                       fix_effect="Scales mesh, hierarchy, bind data, and animation translation together.")
        if snapshot.source_unit_scale <= 0 or not math.isfinite(snapshot.source_unit_scale):
            report.add("error", "invalid_source_unit_scale", "Source unit scale is invalid.",
                       "Scale conversion must be known and positive before bind and animation translation are baked.", field="import_coordinate_system.source_units")
        elif snapshot.source_unit_scale < 0.01 or snapshot.source_unit_scale > 100.0:
            report.add("warning", "source_unit_scale_mismatch", f"Source unit scale {snapshot.source_unit_scale:g} differs substantially from KOTOR units.",
                       "Uncorrected units can create a tiny or enormous creature and root motion.", field="model_placement.global_scale",
                       automatic_fix="Apply the detected unit conversion",
                       fix_effect="Scales the complete deform and animation data consistently.")
        if snapshot.lowest_contact_height is not None:
            pivot_z = float((list(project.pivot_offset or (0.0, 0.0, 0.0)) + [0.0, 0.0, 0.0])[2])
            final_contact = snapshot.lowest_contact_height + project.ground_offset + pivot_z
            if abs(final_contact) > 1.0e-3:
                report.add("warning", "ground_contact_offset", f"The lowest selected contact will export at height {final_contact:.5g}, not zero.",
                           "The creature may appear submerged or floating in game.", field="model_placement.ground_offset",
                           automatic_fix="Place selected contact points on the ground",
                           fix_effect=f"Changes the vertical project offset to {project.ground_offset - final_contact:.6g}; source vertices are not edited.")
        if snapshot.runtime_height_offset > 1.0e-6 and not math.isclose(
            float(project.runtime_height_offset),
            float(snapshot.runtime_height_offset),
            rel_tol=0.0,
            abs_tol=1.0e-5,
        ):
            report.add(
                "error",
                "runtime_height_correction_missing",
                "The automatic KOTOR runtime-height correction no longer matches the imported rig.",
                "The creature can be pulled below the floor even when its static mesh contacts look correct.",
                field="model_placement.runtime_height_offset",
                automatic_fix="Restore the detected source-root height correction",
                fix_effect=(
                    f"Uses {snapshot.runtime_height_offset:.6g} from "
                    f"{snapshot.runtime_height_source or 'the imported root joint'} on heightdummy; source data is unchanged."
                ),
            )
        if snapshot.root_height is not None and snapshot.root_height + project.ground_offset < -0.1:
            report.add("warning", "root_below_floor", "The exported root pivot is substantially below the floor.",
                       "Root translation can repeatedly pull the creature under the module floor.", field="model_placement.pivot_offset")
        if snapshot.source_forward and snapshot.source_forward.upper() != snapshot.expected_forward.upper():
            report.add("warning", "facing_conversion_required", f"Source faces {snapshot.source_forward}; KOTOR preview expects {snapshot.expected_forward}.",
                       "Locomotion and gameplay facing must agree with the exported transform frame.", field="import_coordinate_system.source_forward",
                       automatic_fix="Reorient the full creature to KOTOR axes",
                       fix_effect="Rotates mesh, hierarchy, bind pose, and animation translations together.")

    def _validate_animations(
        self, snapshot: CustomRiggedCharacterSnapshot, project: CustomRiggedCharacterProject,
        report: CustomRigValidationReport,
    ) -> None:
        exported = {node.name for node in snapshot.nodes if node.exported}
        deform = {node.name for node in snapshot.nodes if node.deform and node.exported}
        clips = {clip.name: clip for clip in snapshot.animations}
        mappings = {mapping.source_name: mapping for mapping in project.animation_mappings}
        for clip in snapshot.animations:
            mapping = mappings.get(clip.name)
            assigned = bool(mapping and mapping.assignment != "unassigned")
            if not math.isfinite(clip.duration) or clip.duration <= 0.0 or clip.duration > 3600.0:
                report.add(
                    "error" if assigned else "information",
                    "invalid_animation_duration" if assigned else "unassigned_zero_duration_action",
                    f"Animation '{clip.name}' has duration {clip.duration}.",
                    "Odyssey animation blocks need a finite, reasonable positive duration. Unassigned source actions are retained for review but are not exported.",
                    field=f"animations.{clip.name}.duration",
                )
            targets = {track.node_name for track in clip.tracks}
            missing = sorted(targets - exported)
            if missing and assigned:
                report.add("error", "animation_target_missing", f"Animation '{clip.name}' targets nodes that are not exported: {', '.join(missing)}.",
                           "Mesh and animation tracks must resolve against the same hierarchy.", field=f"animations.{clip.name}.tracks")
            missing_deform_tracks = sorted(deform - targets)
            if missing_deform_tracks and assigned:
                report.add("warning", "missing_deform_tracks", f"Animation '{clip.name}' has no track for {len(missing_deform_tracks)} deform nodes.",
                           "Untracked nodes stay at bind/local defaults and may make the action look incomplete.", field=f"animations.{clip.name}.tracks",
                           details={"missing_nodes": missing_deform_tracks})
            if assigned and clip.source_skeleton_fingerprint and snapshot.skeleton_fingerprint and clip.source_skeleton_fingerprint != snapshot.skeleton_fingerprint:
                report.add("error", "external_animation_rig_mismatch", f"Animation '{clip.name}' comes from a different skeleton.",
                           "Retargeting mismatched source skeletons cannot be performed silently.", field=f"animations.{clip.name}",
                           automatic_fix="Open explicit retarget mapping",
                           fix_effect="Creates a reviewed source-to-target mapping; no tracks are guessed invisibly.")
            clip_flips = 0
            flip_nodes = 0
            for track in clip.tracks:
                flat = [item for row in (*track.positions, *track.rotations, *track.scales) for item in row]
                if assigned and not _finite(flat):
                    report.add("error", "non_finite_animation_key", f"Animation '{clip.name}' has a non-finite key on '{track.node_name}'.",
                               "KOTOR cannot evaluate NaN or infinite controller values.", field=f"animations.{clip.name}.{track.node_name}")
                if assigned and track.scales:
                    report.add("warning", "scale_animation_unsupported", f"Animation '{clip.name}' contains scale keys on '{track.node_name}'.",
                               "The selected self-contained Odyssey path exports ordinary position and orientation controllers, not arbitrary scale animation.",
                               field=f"animations.{clip.name}.{track.node_name}.scale",
                               automatic_fix="Bake supported scale into the deform result" if len(set(track.scales)) == 1 else "",
                               fix_effect="Applies a constant scale to bind/mesh data and removes the redundant track." if len(set(track.scales)) == 1 else "")
                if not assigned:
                    continue
                try:
                    _fixed, flips = quaternion_continuity(track.rotations)
                except ValueError:
                    report.add("error", "invalid_quaternion_key", f"Animation '{clip.name}' has an invalid quaternion on '{track.node_name}'.",
                               "Odyssey orientation controllers require finite non-zero quaternions.", field=f"animations.{clip.name}.{track.node_name}.rotation")
                else:
                    if flips:
                        clip_flips += flips
                        flip_nodes += 1
            if clip_flips:
                report.add(
                    "warning", "quaternion_sign_discontinuity",
                    f"Animation '{clip.name}' has {clip_flips} equivalent quaternion sign flips across {flip_nodes} node(s).",
                    "Equivalent opposite signs can cause interpolation to take an unnecessarily long arc.",
                    field=f"animations.{clip.name}.rotation",
                    automatic_fix="Make adjacent quaternion signs continuous",
                    fix_effect="Flips mathematically equivalent key signs without changing poses.",
                )
            if assigned and len(clip.root_positions) >= 2:
                jumps = [math.dist(a, b) for a, b in zip(clip.root_positions, clip.root_positions[1:])]
                if max(jumps, default=0.0) > 5.0:
                    report.add("warning", "large_root_jump", f"Animation '{clip.name}' contains a large root jump.",
                               "A sudden root displacement can teleport or submerge the creature.", field=f"animations.{clip.name}.root_motion",
                               automatic_fix="Convert to in-place" if any(mapping.source_name == clip.name and mapping.root_motion == "in_place" for mapping in project.animation_mappings) else "")
                if clip.loop and math.dist(clip.root_positions[0], clip.root_positions[-1]) > 0.05:
                    report.add("warning", "loop_root_discontinuity", f"Looping animation '{clip.name}' does not return to its starting root position.",
                               "The visible loop may pop or drift each cycle.", field=f"animations.{clip.name}.loop")
        for mapping in project.animation_mappings:
            if mapping.source_name not in clips and mapping.assignment != "unassigned":
                report.add("error", "mapped_animation_missing", f"Mapped source action '{mapping.source_name}' was not imported.",
                           "The build cannot export an animation mapping without its source keys.", field="animation_mappings")
            if mapping.assignment != "unassigned" and not mapping.confirmed:
                report.add(
                    "error", "animation_mapping_unconfirmed",
                    f"Confirm the KOTOR behavior assignment for '{mapping.source_name}'.",
                    "Name-based suggestions are only suggestions; Ghost Studio will not export a behavior mapping without the user's confirmation.",
                    field="animation_mappings",
                )
            if mapping.assignment != "unassigned" and not validate_resource_name(mapping.exported_name):
                report.add(
                    "error", "invalid_exported_animation_name",
                    f"Animation output name '{mapping.exported_name}' is not KOTOR-safe.",
                    "Embedded and registry animation names must be addressable identifiers of at most 16 characters for this workflow.",
                    field="animation_mappings",
                )
            if mapping.trim_start < 0 or (mapping.trim_end is not None and mapping.trim_end <= mapping.trim_start):
                report.add(
                    "error", "invalid_animation_trim",
                    f"Animation '{mapping.source_name}' has an empty or negative trim range.",
                    "At least one positive span of sampled motion is required to write an Odyssey animation block.",
                    field="animation_mappings",
                )
            if mapping.playback_speed <= 0 or not math.isfinite(mapping.playback_speed):
                report.add(
                    "error", "invalid_playback_speed",
                    f"Animation '{mapping.source_name}' needs a positive playback speed.",
                    "Zero, negative, or non-finite timing cannot be serialized coherently.",
                    field="animation_mappings",
                )
            if mapping.retime_duration is not None and (
                mapping.retime_duration <= 0 or not math.isfinite(mapping.retime_duration)
            ):
                report.add(
                    "error", "invalid_retime_duration",
                    f"Animation '{mapping.source_name}' needs a positive retimed duration.",
                    "The retimed controller keys must occupy a finite positive time span.",
                    field="animation_mappings",
                )
            if mapping.assignment == "custom_runtime_animation" and mapping.runtime_id is None:
                report.add(
                    "error", "missing_mapping_runtime_id",
                    f"Allocate an additive runtime ID for '{mapping.exported_name or mapping.source_name}'.",
                    "A new animation name is not callable until it has a non-conflicting registry ID.",
                    field="animation_mappings",
                    automatic_fix="Allocate the next free custom animation ID",
                    fix_effect="Adds a project-owned registry record and does not replace a vanilla slot.",
                )
            if (
                mapping.assignment == "vanilla_behavior_alias"
                and mapping.exported_name not in _VERIFIED_CREATURE_ALIASES
            ):
                report.add("warning", "unknown_vanilla_alias", f"'{mapping.exported_name}' is not one of the builder's verified vanilla creature aliases.",
                           "The engine may never request this name without runtime registration or gameplay routing.", field="animation_mappings")
        exported_names: dict[str, list[str]] = {}
        for mapping in project.animation_mappings:
            if mapping.assignment != "unassigned" and mapping.confirmed and mapping.exported_name:
                exported_names.setdefault(mapping.exported_name.casefold(), []).append(mapping.source_name)
        for name, sources in exported_names.items():
            if len(sources) > 1:
                report.add(
                    "error", "duplicate_exported_animation_name",
                    f"More than one source action exports as '{name}': {', '.join(sources)}.",
                    "Odyssey animation blocks need unique names; duplicate aliases make the chosen controller ambiguous.",
                    field="animation_mappings",
                )
        mapped_aliases = {
            mapping.exported_name
            for mapping in project.animation_mappings
            if mapping.assignment == "vanilla_behavior_alias" and mapping.confirmed
        }
        for required in ("cpause1", "cwalk", "crun"):
            if required not in mapped_aliases:
                report.add("warning", "locomotion_alias_missing", f"No confirmed {required} behavior alias is assigned.",
                           "The creature can build, but ordinary idle/walk/run requests may fall back or appear static.", field="animation_mappings")
        self._validate_behavior_profile(project, report)

    def _validate_behavior_profile(
        self,
        project: CustomRiggedCharacterProject,
        report: CustomRigValidationReport,
    ) -> None:
        profile = dict(project.behavior_profile or {})
        template_resref = str(profile.get("template_resref") or "").strip().lower()
        if not template_resref:
            report.add(
                "warning",
                "utc_behavior_template_missing",
                "No installed UTC behavior template has been selected.",
                "A new minimal UTC can still be generated, but it will not inherit the class, feats, equipment, perception, and complete AI hook setup of a proven creature.",
                field="gameplay.behavior_profile.template_resref",
            )
        elif not validate_resource_name(template_resref):
            report.add(
                "error",
                "utc_behavior_template_invalid",
                "The selected UTC template resource name is not KOTOR-safe.",
                "The installed template must be addressable as a 1-16 character UTC resref.",
                field="gameplay.behavior_profile.template_resref",
            )
        template_game = str(profile.get("template_game") or "").strip().upper()
        if template_resref and template_game and template_game != project.target_game:
            report.add(
                "error",
                "utc_behavior_template_wrong_game",
                f"The behavior template belongs to {template_game}, not {project.target_game}.",
                "K1 and K2 creature templates and script resources are not interchangeable.",
                field="gameplay.behavior_profile.template_game",
            )
        template_hash = str(profile.get("template_sha256") or "").strip().lower()
        if template_resref and not _SHA256_RE.fullmatch(template_hash):
            report.add(
                "error",
                "utc_behavior_template_hash_missing",
                "Refresh and reselect the installed UTC template so its source hash is recorded.",
                "Build must refuse a template that cannot be proven identical to the one the user reviewed.",
                field="gameplay.behavior_profile.template_sha256",
            )
        module_hooks = list(dict(profile.get("template_snapshot") or {}).get("module_only_script_hooks") or ())
        if module_hooks:
            report.add(
                "warning",
                "utc_behavior_module_only_scripts",
                "The selected UTC assigns one or more scripts that exist only in its source module.",
                "Those scripts may be unavailable when the custom creature is spawned elsewhere. Prefer a global template or replace the flagged hooks.",
                field="gameplay.behavior_profile.template_snapshot.module_only_script_hooks",
                details={"hooks": module_hooks},
            )
        for hook, row_value in sorted(dict(profile.get("script_hooks") or {}).items()):
            if hook not in _UTC_SCRIPT_HOOK_FIELDS:
                report.add(
                    "error", "utc_behavior_unknown_hook", f"Unknown UTC behavior hook '{hook}'.",
                    "Only typed UTC script fields can be emitted safely.", field="gameplay.behavior_profile.script_hooks",
                )
                continue
            row = dict(row_value or {})
            mode = str(row.get("mode") or "inherit").strip().lower()
            if mode not in {"inherit", "existing", "custom"}:
                report.add(
                    "error", "utc_behavior_unknown_mode", f"Behavior hook '{hook}' uses unknown mode '{mode}'.",
                    "Each hook must inherit the template, reference an existing NCS, or compile project-owned NSS.",
                    field=f"gameplay.behavior_profile.script_hooks.{hook}.mode",
                )
                continue
            if mode == "inherit":
                continue
            resref = str(row.get("resref") or "").strip().lower()
            if not validate_resource_name(resref):
                report.add(
                    "error", "utc_behavior_script_resref_invalid", f"Behavior script for '{hook}' has an invalid resource name.",
                    "NSS/NCS resources use the same 1-16 character lookup rules as UTC files.",
                    field=f"gameplay.behavior_profile.script_hooks.{hook}.resref",
                )
            if mode == "custom":
                source = str(row.get("source") or "")
                if not source.strip():
                    report.add(
                        "error", "utc_behavior_custom_source_empty", f"Custom behavior source for '{hook}' is empty.",
                        "Build cannot compile an empty hook into an auditable NCS resource.",
                        field=f"gameplay.behavior_profile.script_hooks.{hook}.source",
                    )
                expected = str(row.get("source_sha256") or "").strip().lower()
                actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
                if expected and expected != actual:
                    report.add(
                        "error", "utc_behavior_source_hash_changed", f"Custom behavior source for '{hook}' changed after its last compile.",
                        "Compile and check the edited source again before Build.",
                        field=f"gameplay.behavior_profile.script_hooks.{hook}.source_sha256",
                    )

    def _validate_materials(self, snapshot: CustomRiggedCharacterSnapshot, report: CustomRigValidationReport) -> None:
        if not snapshot.materials:
            report.add("warning", "no_materials", "No material or texture assignment was imported.",
                       "The creature may render untextured even if its rig is valid.", field="material_assignments")
        for material in snapshot.materials:
            field_prefix = f"materials.{material.material_name}"
            if not validate_resource_name(material.texture_resref):
                report.add("error", "invalid_texture_resref", f"Material '{material.material_name}' has no KOTOR-safe texture name.",
                           "Exported texture resources use the same 1–16 character lookup rules as models.", field=f"{field_prefix}.texture_resref")
            if material.source_format and material.source_format not in _SUPPORTED_TEXTURE_FORMATS:
                report.add("error", "unsupported_texture_format", f"Texture format '.{material.source_format}' is not supported for conversion.",
                           "Ghost Studio must be able to decode the source before creating a TGA or TPC copy.", field=f"{field_prefix}.source_format")
            width, height = material.texture_size
            if width <= 0 or height <= 0:
                report.add("warning", "missing_texture_dimensions", f"Texture dimensions for '{material.material_name}' are unknown.",
                           "Size and format checks cannot confirm a KOTOR-ready converted copy.", field=f"{field_prefix}.texture_size")
            elif width > 4096 or height > 4096 or width & (width - 1) or height & (height - 1):
                report.add("warning", "unusual_texture_dimensions", f"Texture for '{material.material_name}' is {width}×{height}.",
                           "Large or non-power-of-two source textures may need conversion for predictable KOTOR loading.", field=f"{field_prefix}.texture_size")
            if material.uvs:
                if not _finite(value for uv in material.uvs for value in uv):
                    report.add("error", "non_finite_uv", f"Material '{material.material_name}' contains a non-finite UV coordinate.",
                               "The texture sampler cannot address NaN or infinite coordinates.", field=f"{field_prefix}.uvs")
                outside = sum(not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0) for u, v in material.uvs)
                if outside and material.wrap_mode != "repeat":
                    report.add("warning", "uv_outside_without_repeat", f"Material '{material.material_name}' has {outside} UVs outside 0–1 but repeat wrapping is not selected.",
                               "The Borhek workflow showed that wrapped UV islands must retain repeat behavior or be shifted coherently into the base tile.",
                               field=f"{field_prefix}.wrap_mode", automatic_fix="Enable repeat wrapping",
                               fix_effect="Changes generated sampler/TXI policy; source UVs are not destroyed.")
                elif outside:
                    report.add("information", "uv_repeat_required", f"Material '{material.material_name}' intentionally uses repeated UVs.",
                               "Generated texture settings must preserve repeat wrapping.", field=f"{field_prefix}.uvs",
                               details={"outside_count": outside, "uv_count": len(material.uvs)})
            if material.has_alpha and material.alpha_mode == "opaque":
                report.add("warning", "alpha_channel_ignored", f"Material '{material.material_name}' contains alpha but is set to opaque.",
                           "Transparent source pixels will not match the KOTOR approximation unless alpha policy is chosen.", field=f"{field_prefix}.alpha_mode")


__all__ = [
    "AnimationClipSnapshot",
    "AnimationTrackSnapshot",
    "CustomRigValidationIssue",
    "CustomRigValidationReport",
    "CustomRiggedCharacterSnapshot",
    "CustomRiggedCharacterValidator",
    "MaterialSnapshot",
    "RigNodeSnapshot",
    "axis_scale_point",
    "ground_offset_for_contacts",
    "normalized_influences",
    "quaternion_continuity",
    "validate_resource_name",
]
