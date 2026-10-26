"""Revisioned Unreal source identity and Ghost-owned edits (Core.Scene).

KMAP holds small references and overrides; geometry and textures remain immutable
sidecars. Reconciliation is pure and does not mutate the editor or read files.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

SCHEMA = "ghoststudio.unreal-import-state/v1"
COMPONENT_FIELDS = {"world_matrix", "label", "excluded", "material_overrides", "walkmesh"}


def revision(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def new_import_state(manifest_reference: dict, source_revision: str, map_package: str) -> dict:
    return {"schema": SCHEMA, "map_package": map_package, "manifest": copy.deepcopy(manifest_reference),
            "source_revision": source_revision, "component_overrides": {}, "module_overrides": {},
            "source_removed_components": [], "policy_version": "static-single-sector-v1"}


def validate_state(state: dict) -> None:
    if state.get("schema") != SCHEMA:
        raise ValueError("Unsupported Unreal import state.")
    if state.get("policy_version") != "static-single-sector-v1":
        raise ValueError("Unsupported Unreal compiler policy.")
    if not isinstance(state.get("component_overrides"), dict) or not isinstance(state.get("module_overrides"), dict):
        raise ValueError("Malformed Unreal override state.")
    for identity, values in state["component_overrides"].items():
        if not isinstance(identity, str) or not isinstance(values, dict) or set(values) - COMPONENT_FIELDS:
            raise ValueError("Unknown component override field.")
    if set(state["module_overrides"]) - {"entry_position_cm", "entry_yaw_degrees", "name", "resref"}:
        raise ValueError("Unknown module override field.")
    revision(state)  # Reject nonfinite/non-JSON persisted values.


@dataclass(frozen=True)
class ReimportChange:
    component_id: str
    kind: str
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReimportConflict:
    component_id: str
    field: str
    reason: str

    @property
    def key(self) -> str:
        # JSON tuple encoding avoids delimiter collisions in source IDs.
        return json.dumps([self.component_id, self.field], separators=(",", ":"))


@dataclass(frozen=True)
class ReimportPlan:
    expected_state_revision: str
    incoming_source_revision: str
    changes: tuple[ReimportChange, ...]
    conflicts: tuple[ReimportConflict, ...]


def _components(manifest: dict) -> dict[str, dict]:
    result = {}
    for actor in manifest.get("actors", []):
        identity = actor.get("id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError("Source component identities must be unique and nonempty.")
        result[identity] = actor
    return result


def _source_properties(actor: dict, manifest: dict) -> dict:
    return {"world_matrix": actor.get("world_matrix"),
            "label": actor.get("label", actor.get("metadata", {}).get("label", "")),
            "mesh_file": (actor.get("mesh_file") or {}).get("sha256"),
            "role": actor.get("role"), "asset_path": actor.get("asset_path"),
            "materials": revision(manifest.get("materials", []))}


def plan_reimport(state: dict, previous: dict, incoming: dict, incoming_revision: str) -> ReimportPlan:
    validate_state(state)
    if incoming.get("source", {}).get("map_package") != state["map_package"]:
        raise ValueError("Reimport must refer to the same source map; import a different map separately.")
    old, new = _components(previous), _components(incoming)
    changes, conflicts = [], []
    for identity in sorted(old.keys() | new.keys()):
        overrides = state["component_overrides"].get(identity, {})
        if identity not in old:
            changes.append(ReimportChange(identity, "added"))
        elif identity not in new:
            if identity in state.get("source_removed_components", []):
                continue  # Already accepted as a Ghost-owned local copy.
            changes.append(ReimportChange(identity, "removed"))
            if overrides:
                conflicts.append(ReimportConflict(identity, "deleted", "Source removed a locally edited component; keep a local copy or accept deletion."))
        else:
            before, after = _source_properties(old[identity], previous), _source_properties(new[identity], incoming)
            changed = tuple(k for k in before if before[k] != after[k])
            if changed:
                changes.append(ReimportChange(identity, "changed", changed))
            for field in ("world_matrix", "label"):
                if field in overrides and field in changed and overrides[field] != after[field]:
                    conflicts.append(ReimportConflict(identity, field, "Source and Ghost both changed this property."))
            if "walkmesh" in overrides and "mesh_file" in changed:
                conflicts.append(ReimportConflict(identity, "walkmesh", "Source geometry changed under a Ghost collision correction."))
            if "material_overrides" in overrides and "materials" in changed:
                conflicts.append(ReimportConflict(identity, "material_overrides", "Source materials changed under a Ghost material override."))
    return ReimportPlan(revision(state), incoming_revision, tuple(changes), tuple(conflicts))


def reconcile(state: dict, plan: ReimportPlan, previous: dict, incoming: dict,
              manifest_reference: dict, decisions: dict[str, str] | None = None) -> tuple[dict, dict]:
    """Apply explicit property decisions; caller must recheck source bytes too."""
    if revision(state) != plan.expected_state_revision:
        raise ValueError("Import preview is stale; project edits changed after preview.")
    # Recompute the plan from supplied source to reject mismatched cached input.
    if plan != plan_reimport(state, previous, incoming, plan.incoming_source_revision):
        raise ValueError("Import preview does not describe these source snapshots.")
    decisions = decisions or {}
    required = {conflict.key for conflict in plan.conflicts}
    if set(decisions) != required or any(value not in {"keep_local", "accept_source"} for value in decisions.values()):
        raise ValueError("Resolve every reimport conflict explicitly before applying.")
    updated, merged = copy.deepcopy(state), copy.deepcopy(incoming)
    old = _components(previous)
    retained = set()
    new_ids = set(_components(incoming))
    for identity in state.get("source_removed_components", []):
        if identity not in new_ids and identity in old:
            merged["actors"].append(copy.deepcopy(old[identity]))
            retained.add(identity)
    for conflict in plan.conflicts:
        keep = decisions[conflict.key] == "keep_local"
        if conflict.field == "deleted":
            if keep:
                merged["actors"].append(copy.deepcopy(old[conflict.component_id]))
                retained.add(conflict.component_id)
            else:
                updated["component_overrides"].pop(conflict.component_id, None)
        elif not keep:
            updated["component_overrides"][conflict.component_id].pop(conflict.field, None)
    for change in plan.changes:
        if change.kind == "removed" and change.component_id not in retained:
            updated["component_overrides"].pop(change.component_id, None)
    merged["source"]["component_count"] = len(merged["actors"])
    if retained:
        material_ids = {m["id"] for m in merged["materials"]}
        merged["materials"].extend(copy.deepcopy(m) for m in previous["materials"] if m["id"] not in material_ids)
    updated.update(manifest=copy.deepcopy(manifest_reference), source_revision=plan.incoming_source_revision,
                   source_removed_components=sorted(retained))
    # Spawn, gameplay and module-level edits are Ghost-owned and survive reimport.
    return updated, merged


def apply_overrides(state: dict, resolved_bundle: dict, *, read_walkmesh=None) -> dict:
    """Build an isolated compiler input; shared decoded meshes stay unchanged."""
    validate_state(state)
    bundle = copy.deepcopy(resolved_bundle)
    bundle["module"].update(copy.deepcopy(state["module_overrides"]))
    actors = []
    for actor in bundle["actors"]:
        values = state["component_overrides"].get(actor["id"], {})
        if values.get("excluded"):
            continue
        if "mesh" in actor:
            actor["mesh"] = copy.deepcopy(actor["mesh"])
        if "world_matrix" in values:
            actor["world_matrix"] = copy.deepcopy(values["world_matrix"])
        if "label" in values:
            actor.setdefault("metadata", {})["label"] = str(values["label"])
        if "walkmesh" in values:
            if read_walkmesh is None:
                raise ValueError("Collision override requires its verified sidecar reader.")
            actor["mesh"]["walkmesh"] = read_walkmesh(values["walkmesh"])
            actor["role"] = "static-room"
        substitutions = values.get("material_overrides", {})
        for surface in actor.get("mesh", {}).get("surfaces", []):
            surface["material"] = substitutions.get(surface.get("material"), surface.get("material"))
        actors.append(actor)
    bundle["actors"] = actors
    bundle["source"]["component_count"] = len(actors)
    return bundle
