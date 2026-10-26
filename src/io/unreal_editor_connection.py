"""Install the owned editor plugin and persist reusable conversion choices."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
from src.io.unreal_pipeline_store import atomic_bytes, write_receipt
from src.io.unreal_project_assets import UnrealProjectAssets
from src.io.unreal_level_bundle import read_bundle_file, _json_object, MAX_JSON_BYTES


def connect_editor(project_path, plugin_source, ghost_executable):
    project = Path(project_path).resolve()
    plugin = Path(plugin_source).resolve()
    executable = Path(ghost_executable).resolve()
    if project.suffix.lower() != ".uproject" or not project.is_file():
        raise ValueError("Choose an existing Unreal .uproject file.")
    if executable.name.lower() != "ghoststudio.exe" or not executable.is_file():
        raise ValueError("Choose the built GhostStudio.exe application.")
    descriptor = plugin / "GhostStudioExporter.uplugin"
    if not descriptor.is_file():
        raise ValueError("The Ghost Studio editor plugin is missing from this installation.")
    document = _json_object(project.read_bytes())
    destination = project.parent / "Plugins/GhostStudioExporter"
    if not destination.resolve().is_relative_to(project.parent):
        raise ValueError("Unreal plugin destination escapes the selected project.")
    backup_root = project.parent / "Saved/GhostStudio/setup-backups" / uuid4().hex
    backup_root.mkdir(parents=True)
    atomic_bytes(backup_root / project.name, project.read_bytes())
    if destination.exists():
        existing_descriptor = destination / descriptor.name
        if not existing_descriptor.is_file() or json.loads(existing_descriptor.read_text(encoding="utf-8")).get("FriendlyName") != "Ghost Studio Level Exporter":
            raise ValueError("An unrelated plugin occupies GhostStudioExporter; move it before connecting.")
        shutil.copytree(destination, backup_root / "GhostStudioExporter")
    # Only known source files are replaced; unknown user files are retained.
    for source in plugin.rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        target = destination / source.relative_to(plugin)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Plugin file destination escapes its owned folder.")
        atomic_bytes(target, source.read_bytes())
    plugins = document.setdefault("Plugins", [])
    match = next((p for p in plugins if p.get("Name") == "GhostStudioExporter"), None)
    if match is None:
        plugins.append({"Name": "GhostStudioExporter", "Enabled": True})
    else:
        match["Enabled"] = True
    atomic_bytes(project, (json.dumps(document, indent=2) + "\n").encode())
    connection = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "GhostStudio/unreal-connection.json"
    write_receipt(connection, {"schema": "ghoststudio.unreal-connection/v1", "ghost_executable": str(executable)})
    return {"plugin": str(destination), "backup": str(backup_root), "message": "Connected. Restart Unreal, then choose Tools → Send Level to Ghost Studio."}


def capture_conversion_profile(project_directory, policy_path, snapshot):
    assets = UnrealProjectAssets(project_directory)
    path = Path(policy_path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Conversion policy is too large.")
    policy = _json_object(path.read_bytes())
    for material in policy.get("materials", {}).values():
        if "diffuse_file" in material:
            material["diffuse_file"] = assets.put(read_bundle_file(path.parent, material["diffuse_file"]), "tga")
    collision = policy.get("collision", {})
    if "walkmesh_file" in collision:
        collision["walkmesh_file"] = assets.put_json(_json_object(read_bundle_file(path.parent, collision["walkmesh_file"])))
    component = next(c for c in snapshot["components"] if c["id"] == collision["component_id"])
    binding = {"policy": policy, "map_package": snapshot["map_package"],
               "collision_mesh_sha256": snapshot["meshes"][component["mesh"]]["sha256"]}
    return assets.put_json(binding)


def restore_conversion_profile(project_directory, reference, snapshot, output_directory):
    assets = UnrealProjectAssets(project_directory)
    saved = assets.read_json(reference)
    if saved["map_package"] != snapshot["map_package"]:
        raise ValueError("Saved conversion choices belong to another map.")
    policy = copy.deepcopy(saved["policy"])
    collision = policy["collision"]
    component = next((c for c in snapshot["components"] if c["id"] == collision["component_id"]), None)
    if component is None or snapshot["meshes"][component["mesh"]]["sha256"] != saved["collision_mesh_sha256"]:
        raise ValueError("The collision source changed. Review the walkmesh choice before updating this map.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    def materialize(reference):
        data = assets.read(reference)
        name = Path(reference["path"]).name
        atomic_bytes(output / name, data)
        return {"path": name, "sha256": reference["sha256"]}
    for material in policy.get("materials", {}).values():
        if "diffuse_file" in material:
            material["diffuse_file"] = materialize(material["diffuse_file"])
    if "walkmesh_file" in collision:
        collision["walkmesh_file"] = materialize(collision["walkmesh_file"])
    policy["snapshot_sha256"] = snapshot["snapshot_sha256"]
    path = output / "policy.json"
    write_receipt(path, policy)
    return path


def write_reviewed_policy(directory, policy, snapshot, material_choices):
    """Materialize explicitly selected bakes beside a conversion policy."""
    from src.io.unreal_native_snapshot import material_revision, _diffuse
    from src.io.unreal_pipeline_store import digest
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy = copy.deepcopy(policy)
    for key, (path, blend) in material_choices.items():
        source = Path(path)
        if source.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"Diffuse image exceeds the import size limit: {source.name}")
        data = source.read_bytes()
        name = digest(data) + ".tga"
        atomic_bytes(directory / name, data)
        policy["materials"][key] = {"recipe": "approved-texture-v1", "source_revision": material_revision(snapshot, key),
            "diffuse_file": {"path": name, "sha256": digest(data)}, "blend_mode": blend,
            "conversion_note": "Creator selected a baked diffuse image and accepted loss of Unreal runtime shading."}
    # Report material problems while the review can still be edited.
    included = [c for c in snapshot["components"] if c["disposition"] == "static_geometry" and c["id"] not in policy.get("excluded_components", {})]
    for key in {key for component in included for key in component["materials"]}:
        _diffuse(snapshot, key, policy.get("materials", {}).get(key, {}), directory)
    path = directory / "policy.json"
    write_receipt(path, policy)
    return path
