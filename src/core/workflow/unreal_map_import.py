"""Revision-bound Map Studio import orchestration; no Qt and no game writes."""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.core.level.kmap_serializer import KMapSerializer
from src.core.modules.unreal_import_state import new_import_state, revision
from src.core.modules.unreal_authored_reimport import merge_authored_source
from src.core.modules.unreal_level_import import build_unreal_level_project
from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload, authored_project_to_kmap_payload
from src.core.modules.authored_module_project import validate_authored_module_project
from src.core.modules.unreal_sector_projection import project_unreal_sector
from src.io.unreal_project_assets import UnrealProjectAssets


def project_revision(project):
    from src.core.level.map_studio_texture_assets import project_texture_export_resources
    texture_hashes = [(name, kind, hashlib.sha256(data).hexdigest()) for name, kind, data in project_texture_export_resources(project)]
    return revision({"path": str(Path(project.path).resolve()), "data": KMapSerializer.to_dict(project), "local_textures": texture_hashes})


@dataclass(frozen=True)
class UnrealImportPreview:
    expected_project_revision: str
    source_path: str
    source_revision: str
    conflicts: tuple[str, ...]
    component_labels: tuple[str, ...]
    warnings: tuple[str, ...]


def _prepare(project, source_path, *, expected_source_revision=None, decisions=None):
    if not project.path:
        raise ValueError("Save the KMAP first so source assets have a portable project home.")
    assets = UnrealProjectAssets(Path(project.path).resolve().parent)
    reference, manifest, source_revision = assets.capture_bundle(source_path, expected_revision=expected_source_revision)
    imported = build_unreal_level_project(assets.resolve_bundle(manifest), independent_components=True)
    source_payload = authored_project_to_kmap_payload(imported.project)
    old_state = project.extra_sections.get("unreal_source")
    existing = project.extra_sections.get("authored_module")
    conflicts = ()
    if old_state:
        if old_state["map_package"] != manifest["source"]["map_package"]:
            raise ValueError("This project belongs to a different Unreal map; use a new KMAP.")
        baseline = assets.read_json(old_state["authored_baseline"])
        if source_payload["module_root"] != baseline["module_root"]:
            raise ValueError("Reimport cannot rename the module resource identity.")
        if not isinstance(existing, dict):
            raise ValueError("The authored project state is missing.")
        candidate, conflicts = merge_authored_source(baseline, existing, source_payload, decisions)
    else:
        if existing or project.rooms or project.modules or project.objects:
            raise ValueError("Import into an empty KMAP; an existing scene is never silently replaced.")
        candidate = source_payload
    state = new_import_state(reference, source_revision, manifest["source"]["map_package"])
    state["authored_baseline"] = assets.put_json(source_payload)
    # Textures remain hash-referenced. TXI is a tiny compiler policy product.
    resources = []
    for name, kind, data in imported.extra_resources:
        if kind == "tga":
            resources.append({"resref": name, "restype": kind, "file": assets.put(data, kind)})
        elif kind == "txi":
            resources.append({"resref": name, "restype": kind, "text": data.decode("ascii")})
        else:
            raise ValueError("Unsupported imported runtime resource.")
    # Retained Ghost edits can still refer to previous textures.
    incoming_keys = {(r["resref"], r["restype"]) for r in resources}
    resources.extend(copy.deepcopy(r) for r in (old_state or {}).get("resources", []) if (r["resref"], r["restype"]) not in incoming_keys)
    state["resources"] = resources
    from src.core.level.map_studio_texture_assets import project_texture_export_resources
    local_names = {name for name, _, _ in project_texture_export_resources(project)}
    old_resources = {}
    new_resources = {}
    for rows, table in (((old_state or {}).get("resources", []), old_resources), (resources, new_resources)):
        for row in rows:
            table.setdefault(row["resref"], {})[row["restype"]] = row.get("file", {}).get("sha256", row.get("text"))
    accepted_textures = []
    for name in sorted(local_names & old_resources.keys()):
        if old_resources[name] != new_resources.get(name):
            key = json.dumps(["texture", name], separators=(",", ":"))
            conflicts += (key,)
            if (decisions or {}).get(key) == "accept_source":
                accepted_textures.append(name)
    state["accepted_source_textures"] = accepted_textures
    labels = tuple(str(a.get("metadata", {}).get("label", a["id"])) for a in manifest["actors"] if a.get("role") != "excluded")
    return state, candidate, conflicts, labels, imported.warnings


def preview_unreal_import(project, source_path) -> UnrealImportPreview:
    expected = project_revision(project)
    state, candidate, conflicts, labels, warnings = _prepare(project, source_path)
    return UnrealImportPreview(expected, str(Path(source_path).resolve()), state["source_revision"], conflicts, labels, warnings)


def apply_unreal_import(project, preview: UnrealImportPreview, decisions=None):
    """Produce a replacement KMAP; the UI commits it as one undoable command."""
    if project_revision(project) != preview.expected_project_revision:
        raise ValueError("Project changed since import preview; inspect again.")
    state, candidate, conflicts, _, _ = _prepare(project, preview.source_path,
        expected_source_revision=preview.source_revision, decisions=decisions)
    choices = decisions or {}
    if conflicts != preview.conflicts or set(choices) != set(conflicts) or any(v not in {"keep_local", "accept_source"} for v in choices.values()):
        raise ValueError("Resolve every current import conflict explicitly.")
    authored = project_unreal_sector(authored_project_from_kmap_payload(candidate))
    validation = validate_authored_module_project(authored)
    if not validation.ok:
        raise ValueError("Imported project is invalid: " + "; ".join(validation.blocking_issues))
    from src.core.modules.authored_module_objects import validate_authored_gameplay_placement_against_walkmesh
    collision = authored.rooms[0].primitive.wok
    placement = validate_authored_gameplay_placement_against_walkmesh(authored.placements, collision, z_tolerance=0.25)
    if not placement.ok:
        raise ValueError("Preserved spawn/gameplay needs repair: " + "; ".join(placement.blocking_issues))
    if project_revision(project) != preview.expected_project_revision:
        raise ValueError("Project changed during import validation; inspect again.")
    updated = copy.deepcopy(project)
    updated.extra_sections.update(unreal_source=state, authored_module=candidate)
    updated.textures = [texture for texture in updated.textures if texture.resref.strip().lower() not in state["accepted_source_textures"]]
    updated.name, updated.game = authored.metadata.display_name, "K2"
    updated.source_game = updated.target_game = "K2"
    updated.dirty = True
    return updated


def unreal_runtime_resources(project):
    state = project.extra_sections.get("unreal_source")
    if not state:
        return ()
    assets = UnrealProjectAssets(Path(project.path).resolve().parent)
    from src.core.level.map_studio_texture_assets import project_texture_export_resources
    local = {(name, kind) for name, kind, _ in project_texture_export_resources(project)}
    def overridden(row):
        name, kind = row["resref"], row["restype"]
        return (name, kind) in local or (kind == "tga" and (name, "tpc") in local)
    return tuple((row["resref"], row["restype"], assets.read(row["file"]) if "file" in row else row["text"].encode("ascii")) for row in state["resources"] if not overridden(row))
