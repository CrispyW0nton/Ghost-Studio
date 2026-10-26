"""Headless orchestration contracts for the foreign-rig Character Builder.

The native Character Builder's template-rig pipeline is intentionally not
imported here.  This service accepts a hierarchy that already represents the
selected foreign deform rig and coordinates validation, semantic animation
names, Odyssey serialization, output hashing, and persistent evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.core.project.custom_rigged_character_project import (
    AnimationMapping,
    CustomRiggedCharacterProject,
    sha256_file,
)
from src.core.secondary_motion.workflow import compile_secondary_motion_profile


VANILLA_BEHAVIOR_ALIASES = {
    "idle": "cpause1",
    "primary_idle": "cpause1",
    "secondary_idle": "cpause2",
    "walk": "cwalk",
    "run": "crun",
    "turn_left": "chturnl",
    "turn_right": "chturnr",
    "monster_attack_1": "m0a1",
    "monster_attack_2": "m0a2",
    "generic_attack_1": "g0a1",
    "generic_attack_2": "g0a2",
    "damage_reaction": "cdamages",
    "dodge": "cdodgeg",
    "combat_ready": "creadyr",
    "combat_turn_ready": "creadyrtw",
    "injured_walk": "cwalkinj",
    "knockdown": "ckdbck",
    "knockdown_loop": "ckdbcklp",
    "get_up": "cgustandb",
    "death": "cdie",
    "dead_pose": "cdead",
    "roar_taunt": "ctaunt",
    "victory": "cvictory",
}

ANIMATION_ASSIGNMENTS = (
    "unassigned",
    "vanilla_behavior_alias",
    "custom_runtime_animation",
)

BEHAVIOR_CATEGORIES = (
    "primary_idle",
    "secondary_idle",
    "walk",
    "run",
    "turn_left",
    "turn_right",
    "attack",
    "power_attack",
    "charge",
    "damage_reaction",
    "knockdown",
    "get_up",
    "death",
    "dead_pose",
    "roar_taunt",
    "special_scripted_action",
)

BORHEK_GOLDEN_CONTRACT = {
    "source_hierarchy_nodes": 40,
    "source_meshes": 10,
    "source_vertices": 2246,
    "source_triangles": 3058,
    "exported_skin_nodes": 15,
    "max_skin_palette": 12,
    "reloaded_odyssey_nodes": 58,
    "runtime_height_node": "heightdummy",
    "runtime_height_offset": 1.9724489450454712,
    "semantic_animations": ("cpause1", "cwalk", "crun"),
    "mdl_sha256": "49063631c4b9f3b4db80f6c6e0036430a3235c2058341a7755b1cab00a0da491",
    "mdx_sha256": "a2d0ceb85de8403672686777df4b8c1c634f36fd93f90c16df4a8f5a51b8d89d",
    "runtime_checklist": (
        "visible",
        "ground_height",
        "texture_wrapping",
        "idle",
        "walk_while_moving",
        "run_while_moving_quickly",
        "turning_skin_stability",
        "module_reload",
        "custom_action_request",
    ),
}


def suggest_semantic_mapping(source_name: str) -> tuple[str, str]:
    """Suggest a friendly behavior category and export alias by clip name."""

    name = str(source_name or "").strip().casefold().replace("-", "_").replace(" ", "_")
    leaf = name.replace("|", "_").split("::")[-1]
    if any(token in leaf for token in ("idle", "pause", "stand", "breath")):
        return "primary_idle", "cpause1"
    if "walk" in leaf and not any(token in leaf for token in ("back", "strafe", "attack")):
        return "walk", "cwalk"
    if any(token in leaf for token in ("run", "sprint", "jog")) and not any(
        token in leaf for token in ("attack", "skid", "stop", "back", "strafe")
    ):
        return "run", "crun"
    if any(token in leaf for token in ("dead", "corpse")) and "death" not in leaf:
        return "dead_pose", "cdead"
    if "death" in name or "die" in name:
        return "death", "cdie"
    if any(token in leaf for token in ("gethit", "damage", "hurt")):
        return "damage_reaction", "cdamages"
    if "headbutt" in leaf or "attack02" in leaf or "attack_02" in leaf:
        return "attack", "m0a2"
    if "attack" in name or "strike" in name or "bite" in name or "claw" in name:
        return "attack", "m0a1"
    if "defenseloop" in leaf or "defense_loop" in leaf:
        return "combat_ready", "creadyr"
    if "defensemode" in leaf or "defense_mode" in leaf or "combatready" in leaf:
        return "combat_ready", "creadyrtw"
    if "knockdown" in leaf or "knockback" in leaf:
        return "knockdown", "ckdbck"
    if "getup" in leaf or "standup" in leaf:
        return "get_up", "cgustandb"
    if "roar" in name or "taunt" in name:
        return "roar_taunt", "ctaunt"
    return "unassigned", ""


def namespaced_animation_name(resource_name: str, source_name: str) -> str:
    """Return a deterministic additive name without pretending it is callable."""

    def clean(value: str) -> str:
        return "_".join(part for part in "".join(
            char.lower() if char.isalnum() else " " for char in value
        ).split() if part)

    prefix = clean(resource_name) or "custom"
    clip = clean(source_name) or "action"
    result = f"{prefix}_{clip}"
    if len(result) <= 16:
        return result
    suffix = hashlib.sha256(result.encode("utf-8")).hexdigest()[:4]
    return f"{result[:11]}_{suffix}"


def allocate_animation_id(occupied: Iterable[int], *, start: int = 10000) -> int:
    """Allocate the first free additive ID; occupied IDs are never replaced."""

    used = {int(value) for value in occupied}
    candidate = max(0, int(start))
    while candidate in used:
        candidate += 1
    return candidate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _serialized_animation_position_z(node: Any) -> float | None:
    """Read an animation node's serialized position-delta Z, when present.

    Animation node transforms are serialized as controller rows.  The binary
    loader deliberately leaves their editor-only ``position`` field at zero.
    Odyssey adds this controller value to the base-local position, so a nonzero
    heightdummy delta is unsafe: it repeats the already-authored base correction.
    """

    for controller in list(getattr(node, "controllers", []) or []):
        if not isinstance(controller, Mapping):
            continue
        raw_type = controller.get("type", 0)
        try:
            controller_type = int(raw_type or 0)
        except (TypeError, ValueError):
            controller_type = 0
        if controller_type != 8 and str(controller.get("name", "")).casefold() != "position":
            continue
        for row in list(controller.get("values", []) or []):
            if isinstance(row, (str, bytes)):
                continue
            try:
                values = list(row)
            except TypeError:
                continue
            if len(values) >= 3:
                return float(values[2])
    return None


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


@dataclass
class CustomRiggedBuildResult:
    ok: bool
    output_files: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    report_path: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "output_files": dict(sorted(self.output_files.items())),
            "output_hashes": dict(sorted(self.output_hashes.items())),
            "report_path": self.report_path,
            "warnings": list(self.warnings),
            "error": self.error,
        }


class CustomRiggedCharacterBuildService:
    """Serialize an already-converted self-contained model with audit evidence."""

    report_schema = "ghostrigger.custom_rigged_character_build_report.v1"

    def build_model_pair(
        self,
        project: CustomRiggedCharacterProject,
        model: Any,
        destination: str | Path,
        *,
        validation_results: Iterable[Mapping[str, Any]] = (),
        tool_version: str = "",
        allow_overwrite: bool = False,
    ) -> CustomRiggedBuildResult:
        """Write MDL/MDX and its report without native-template assumptions.

        Import, baking, and validation must have completed before this method is
        called.  The provided model's hierarchy remains authoritative.
        """

        try:
            from src.core.mdl.mdl_writer import MDLBinaryWriter
        except ImportError:  # pragma: no cover - installed package route
            from core.mdl.mdl_writer import MDLBinaryWriter  # type: ignore

        output_root = Path(destination)
        resref = str(project.resource_name or "").strip().lower()
        if not resref:
            return CustomRiggedBuildResult(ok=False, error="A KOTOR resource name is required.")
        targets = {
            "mdl": output_root / f"{resref}.mdl",
            "mdx": output_root / f"{resref}.mdx",
            "report": output_root / f"{resref}.build-report.json",
        }
        if not allow_overwrite:
            occupied = [str(path) for path in targets.values() if path.exists()]
            if occupied:
                ownership_error = self._verify_owned_previous_build(project, targets)
                if ownership_error:
                    return CustomRiggedBuildResult(ok=False, error=ownership_error)

        try:
            mdl_bytes, mdx_bytes = MDLBinaryWriter().write(model)
        except Exception as exc:
            return CustomRiggedBuildResult(ok=False, error=f"Odyssey model conversion failed: {exc}")

        try:
            roundtrip = self.validate_serialized_model(project, mdl_bytes, mdx_bytes)
        except Exception as exc:
            return CustomRiggedBuildResult(
                ok=False,
                error=f"Odyssey reload validation stopped the build: {exc}",
            )

        output_hashes = {
            targets["mdl"].name: sha256_bytes(mdl_bytes),
            targets["mdx"].name: sha256_bytes(mdx_bytes),
        }
        identity = dict(project.secondary_motion.get("model_identity") or {})
        identity.update({
            "model_resrefs": [resref],
            "mdl_sha256": output_hashes[targets["mdl"].name],
            "mdx_sha256": output_hashes[targets["mdx"].name],
        })
        project.secondary_motion["model_identity"] = identity
        try:
            project.secondary_motion = compile_secondary_motion_profile(
                project.secondary_motion,
                target_kind="custom_character",
            )
        except ValueError as exc:
            return CustomRiggedBuildResult(
                ok=False,
                error=f"Secondary-motion validation stopped the build: {exc}",
            )
        report = self.build_report(
            project,
            model=model,
            output_hashes=output_hashes,
            validation_results=validation_results,
            tool_version=tool_version,
            roundtrip_validation=roundtrip,
        )
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            _write_atomic(targets["mdl"], mdl_bytes)
            _write_atomic(targets["mdx"], mdx_bytes)
            _write_atomic(targets["report"], report_bytes)
        except Exception as exc:
            return CustomRiggedBuildResult(ok=False, error=f"Could not write build outputs: {exc}")

        return CustomRiggedBuildResult(
            ok=True,
            output_files={key: str(value) for key, value in targets.items()},
            output_hashes=output_hashes,
            report_path=str(targets["report"]),
        )

    def _verify_owned_previous_build(
        self,
        project: CustomRiggedCharacterProject,
        targets: Mapping[str, Path],
    ) -> str:
        """Allow a rebuild only when the prior pair is intact and project-owned."""

        missing = [str(path) for path in targets.values() if not path.is_file()]
        if missing:
            return (
                "Build stopped because the destination contains a partial prior output. "
                "Choose a new output folder or restore the missing files: " + ", ".join(missing)
            )
        try:
            report = json.loads(targets["report"].read_text(encoding="utf-8"))
        except Exception:
            return "Build stopped because the existing report is unreadable; ownership cannot be proven."
        if (
            str(report.get("schema") or "") != self.report_schema
            or str(report.get("project_id") or "") != project.project_id
            or str(report.get("model_resref") or "").casefold() != project.resource_name.casefold()
        ):
            return "Build stopped because the existing outputs belong to a different project."
        expected = dict(report.get("output_hashes") or {})
        for kind in ("mdl", "mdx"):
            path = targets[kind]
            wanted = str(expected.get(path.name) or "").lower()
            if not wanted or sha256_file(path) != wanted:
                return (
                    f"Build stopped because {path.name} changed after the prior build. "
                    "Choose a new output folder so the changed file is preserved."
                )
        return ""

    def validate_serialized_model(
        self,
        project: CustomRiggedCharacterProject,
        mdl_bytes: bytes,
        mdx_bytes: bytes,
    ) -> dict[str, Any]:
        """Reload writer output and prove its skin/animation name contracts.

        A successful writer call is not enough for a foreign rig.  This gate
        exercises the same binary loader used by Ghost Studio and refuses to
        persist a pair whose palettes or animation tracks no longer resolve.
        """

        try:
            from src.core.game.kotor_loader import load_model_from_bytes
        except ImportError:  # pragma: no cover - installed package route
            from core.game.kotor_loader import load_model_from_bytes  # type: ignore

        if len(mdl_bytes) < 256 or len(mdx_bytes) < 12:
            raise ValueError("The serialized MDL/MDX pair is unexpectedly small.")
        reloaded = load_model_from_bytes(mdl_bytes, mdx_bytes)
        if reloaded is None:
            raise ValueError("Ghost Studio could not reload the serialized model pair.")
        if str(getattr(reloaded, "supermodel", "NULL") or "NULL").casefold() != "null":
            raise ValueError("The custom creature unexpectedly depends on a supermodel.")

        nodes = list(reloaded.all_nodes())
        names = [str(getattr(node, "name", "") or "") for node in nodes]
        folded = [name.casefold() for name in names]
        duplicates = sorted({name for name in folded if folded.count(name) > 1})
        if duplicates:
            raise ValueError("Reloaded node names are not unique: " + ", ".join(duplicates))
        name_set = set(folded)
        selected_root = str(
            project.last_import_summary.get("root_name")
            or project.selected_skeleton_root
            or ""
        ).strip()
        if "::" in selected_root:
            selected_root = selected_root.rsplit("::", 1)[-1].strip()
        absorbed_source_root = (
            selected_root
            if selected_root
            and selected_root.casefold() != str(project.resource_name or "").casefold()
            else ""
        )
        if absorbed_source_root and absorbed_source_root.casefold() in name_set:
            raise ValueError(
                "Reloaded model retained the FBX authoring root as a second Odyssey root: "
                f"'{absorbed_source_root}'. The selected source root must become "
                f"'{project.resource_name}', matching the proven custom-creature hierarchy."
            )
        pivot = list(project.pivot_offset or (0.0, 0.0, 0.0)) + [0.0, 0.0, 0.0]
        expected_runtime_height = (
            float(project.runtime_height_offset)
            + float(project.ground_offset)
            + float(pivot[2])
        )
        height_node = next(
            (node for node in nodes if str(getattr(node, "name", "")).casefold() == "heightdummy"),
            None,
        )
        if height_node is None:
            raise ValueError("Reloaded model lost the KOTOR runtime-height helper.")
        actual_runtime_height = float((getattr(height_node, "position", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))[2])
        if not math.isclose(actual_runtime_height, expected_runtime_height, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(
                "Reloaded model changed the KOTOR runtime-height correction: "
                f"expected {expected_runtime_height:.7g}, got {actual_runtime_height:.7g}."
            )
        skin_nodes = [node for node in nodes if bool(getattr(node, "is_skin", False))]
        for skin in skin_nodes:
            palette = [str(value or "").casefold() for value in list(getattr(skin, "bone_map", []) or [])]
            missing = sorted({name for name in palette if name and name not in name_set})
            if missing:
                raise ValueError(
                    f"Reloaded skin '{getattr(skin, 'name', '')}' has unresolved bones: "
                    + ", ".join(missing)
                )
            for vertex_index, row in enumerate(list(getattr(skin, "skin_data", []) or [])):
                for influence in list(getattr(row, "influences", []) or []):
                    bone_index = int(getattr(influence, "bone_index", -1))
                    weight = float(getattr(influence, "weight", 0.0) or 0.0)
                    if not math.isfinite(weight) or weight < 0.0:
                        raise ValueError(
                            f"Reloaded skin '{getattr(skin, 'name', '')}' vertex {vertex_index} "
                            "contains an invalid weight."
                        )
                    if weight > 0.0 and not 0 <= bone_index < len(palette):
                        raise ValueError(
                            f"Reloaded skin '{getattr(skin, 'name', '')}' vertex {vertex_index} "
                            "contains an out-of-range bone index."
                        )

        animations = list(getattr(reloaded, "animations", []) or [])
        animation_names = {
            str(getattr(animation, "name", "") or "").casefold()
            for animation in animations
        }
        required_names = {
            str(mapping.exported_name or "").casefold()
            for mapping in project.animation_mappings
            if mapping.confirmed
            and mapping.assignment != "unassigned"
            and mapping.exported_name
        }
        missing_animations = sorted(required_names - animation_names)
        if missing_animations:
            raise ValueError(
                "Reloaded model lost mapped animations: " + ", ".join(missing_animations)
            )
        controller_count = 0
        for animation in animations:
            animation_height = next(
                (
                    node for node in list(getattr(animation, "nodes", []) or [])
                    if str(getattr(node, "name", "")).casefold() == "heightdummy"
                ),
                None,
            )
            if animation_height is None:
                raise ValueError(
                    f"Animation '{getattr(animation, 'name', '')}' lost the KOTOR runtime-height helper."
                )
            animation_height_z = _serialized_animation_position_z(animation_height)
            if animation_height_z is not None and not math.isclose(
                animation_height_z, 0.0, rel_tol=0.0, abs_tol=1.0e-7
            ):
                raise ValueError(
                    f"Animation '{getattr(animation, 'name', '')}' repeats the base runtime-height "
                    f"correction as a {animation_height_z:.7g} position delta. Odyssey would add "
                    "that delta to heightdummy and lift the creature twice."
                )
            for node in list(getattr(animation, "nodes", []) or []):
                target = str(getattr(node, "name", "") or "").casefold()
                if target and target not in name_set:
                    raise ValueError(
                        f"Animation '{getattr(animation, 'name', '')}' targets missing node "
                        f"'{getattr(node, 'name', '')}'."
                    )
                for controller in list(getattr(node, "controllers", []) or []):
                    controller_count += 1
                    values = controller.get("values", ()) if isinstance(controller, Mapping) else ()
                    for row in values or ():
                        if not all(math.isfinite(float(value)) for value in row):
                            raise ValueError(
                                f"Animation '{getattr(animation, 'name', '')}' contains a non-finite controller key."
                            )

        return {
            "ok": True,
            "loader": "src.core.game.kotor_loader.load_model_from_bytes",
            "node_count": len(nodes),
            "skin_node_count": len(skin_nodes),
            "animation_names": sorted(animation_names),
            "controller_count": controller_count,
            "all_skin_influences_resolve": True,
            "all_animation_tracks_resolve": True,
            "supermodel": str(getattr(reloaded, "supermodel", "NULL") or "NULL"),
            "runtime_height_node": "heightdummy",
            "runtime_height_offset": actual_runtime_height,
            "runtime_height_source": str(project.runtime_height_source or ""),
            "animation_runtime_height_verified": len(animations),
            "animation_runtime_height_mode": "inherit_base_without_delta",
            "absorbed_source_root": absorbed_source_root,
        }

    def build_report(
        self,
        project: CustomRiggedCharacterProject,
        *,
        model: Any,
        output_hashes: Mapping[str, str],
        validation_results: Iterable[Mapping[str, Any]],
        tool_version: str,
        roundtrip_validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        nodes = list(model.all_nodes()) if hasattr(model, "all_nodes") else []
        meshes = [node for node in nodes if bool(getattr(node, "is_mesh", False))]
        skin_nodes = [node for node in nodes if bool(getattr(node, "is_skin", False))]
        animations = list(getattr(model, "animations", []) or [])
        source_hashes = {
            f"{asset.role}:{index}:{Path(asset.path).name}": asset.sha256
            for index, asset in enumerate(
                [project.primary_fbx, *project.external_animation_assets, *project.texture_assets]
            )
            if asset.path and asset.sha256
        }
        return {
            "schema": self.report_schema,
            "tool_version": str(tool_version or "unknown"),
            "project_schema_version": project.schema_version,
            "project_id": project.project_id,
            "builder_mode": project.builder_mode,
            "target_game": project.target_game,
            "model_resref": project.resource_name,
            "source_hashes": dict(sorted(source_hashes.items())),
            "output_hashes": dict(sorted(output_hashes.items())),
            "skeleton": {
                "selected_root": project.selected_skeleton_root,
                "node_count": len(nodes),
                "node_names": [str(getattr(node, "name", "")) for node in nodes],
                "native_humanoid_template_used": False,
                "runtime_height_node": "heightdummy",
                "runtime_height_offset": float(project.runtime_height_offset),
                "runtime_height_source": str(project.runtime_height_source or ""),
                "manual_ground_offset": float(project.ground_offset),
            },
            "mesh_weights": {
                "mesh_count": len(meshes),
                "skin_node_count": len(skin_nodes),
                "vertex_count": sum(len(getattr(node, "vertices", []) or []) for node in meshes),
                "triangle_count": sum(len(getattr(node, "faces", []) or []) for node in meshes),
                "max_skin_palette": max((len(getattr(node, "bone_map", []) or []) for node in skin_nodes), default=0),
            },
            "animations": [
                {
                    "name": str(getattr(animation, "name", "")),
                    "duration": float(getattr(animation, "length", 0.0) or 0.0),
                }
                for animation in animations
            ],
            "animation_mappings": [mapping.to_dict() for mapping in project.animation_mappings],
            "custom_animation_registrations": [
                {
                    "name": value.name,
                    "animation_id": value.animation_id,
                    "source_clip": value.source_clip,
                    "namespace": value.namespace,
                }
                for value in project.custom_animation_registrations
            ],
            "texture_conversions": [value.to_dict() for value in project.material_assignments],
            "integration": {
                "appearance": dict(project.appearance_settings),
                "utc": dict(project.utc_settings),
                "behavior": dict(project.gameplay_settings),
            },
            "secondary_motion": dict(project.secondary_motion),
            "validation_results": [dict(value) for value in validation_results],
            "roundtrip_validation": dict(roundtrip_validation or {}),
            "automatic_repairs": list(project.automatic_repairs),
            "accepted_warning_ids": sorted(set(project.accepted_warning_ids)),
            "determinism": {
                "stable_json_key_order": True,
                "unexplained_nondeterministic_fields": [],
            },
        }


__all__ = [
    "ANIMATION_ASSIGNMENTS",
    "BEHAVIOR_CATEGORIES",
    "BORHEK_GOLDEN_CONTRACT",
    "CustomRiggedBuildResult",
    "CustomRiggedCharacterBuildService",
    "VANILLA_BEHAVIOR_ALIASES",
    "allocate_animation_id",
    "namespaced_animation_name",
    "suggest_semantic_mapping",
]
