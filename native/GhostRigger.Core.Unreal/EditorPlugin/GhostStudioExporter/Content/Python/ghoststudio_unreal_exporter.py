"""Native Unreal API adapter. No Ghost, MCP, network or AI runtime dependency.

The output is a source snapshot, NOT an Odyssey-ready bundle. Material conversion
and authored collision choices belong to Ghost's import workflow. Never turn an
unsupported component into a successful, silently incomplete conversion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

VERSION = "0.1.0"
SCHEMA = "ghoststudio.unreal-snapshot/v1"
SUPPORTED_ENGINES = {"5.8.2-56702186+++UE5+Release-5.8"}
MAX_COMPONENTS = 10000
MAX_VERTICES = 2_000_000
MAX_SIDECAR_BYTES = 64 * 1024 * 1024


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _vec(value):
    result = [float(value.x), float(value.y), float(value.z)]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Non-finite native geometry or transform")
    return result


def _matrix(transform, unreal):
    origin = transform.transform_location(unreal.Vector(0, 0, 0))
    basis = [transform.transform_location(v) - origin for v in
             (unreal.Vector(1, 0, 0), unreal.Vector(0, 1, 0), unreal.Vector(0, 0, 1))]
    return [[_vec(basis[c])[r] for c in range(3)] + [_vec(origin)[r]] for r in range(3)] + [[0, 0, 0, 1]]


def _package_file(package, unreal):
    """Only declared mounted roots. Never guess a plugin dependency location."""
    package = str(package).split(".")[0]
    if package.startswith("/Game/"):
        root = Path(unreal.Paths.project_content_dir()).resolve()
        relative = package[len("/Game/"):]
    elif package.startswith("/Engine/"):
        root = Path(unreal.Paths.engine_content_dir()).resolve()
        relative = package[len("/Engine/"):]
    else:
        raise ValueError(f"Package mount needs a qualified adapter: {package}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Package escapes its content root: {package}")
    for suffix in (".uasset", ".umap"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise ValueError(f"Missing package on disk: {package}")


def _graph(material, node, unreal, active=None):
    if node is None:
        return None
    active = set() if active is None else active
    key = node.get_path_name()
    if key in active:
        return {"class": "CyclicExpression", "path": key}
    active = active | {key}
    result = {"class": node.get_class().get_name()}
    for prop in ("texture", "constant", "r", "const_a", "const_b", "coordinate_index", "u_tiling", "v_tiling"):
        try:
            value = node.get_editor_property(prop)
        except Exception:
            continue  # Property not declared on this expression type.
        if isinstance(value, unreal.Texture):
            result[prop] = value.get_path_name()
        elif isinstance(value, unreal.LinearColor):
            result[prop] = [value.r, value.g, value.b, value.a]
        elif isinstance(value, (float, int, bool, str)):
            result[prop] = value
    result["inputs"] = [_graph(material, child, unreal, active) for child in
                        unreal.MaterialEditingLibrary.get_inputs_for_material_expression(material, node)]
    return result


def inspect_level(unreal=None):
    """Capture the loaded world without saving or changing source objects."""
    if unreal is None:
        import unreal
    engine = unreal.SystemLibrary.get_engine_version()
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if world is None:
        raise ValueError("Open a level in the Unreal editor first.")
    package = world.get_path_name().split(".")[0]
    report = {"schema": SCHEMA, "exporter_version": VERSION, "engine_version": engine,
              "map_package": package, "coordinates": "ue-centimeters-z-up-left-handed",
              "matrix_convention": "row-major-column-vector-local-to-world",
              "uv_origin": "top-left", "triangle_winding": "native-unreal",
              "components": [], "meshes": {}, "materials": {}, "dependencies": [],
              "diagnostics": [], "source_saved": False}

    def diagnostic(code, subject, message, repair, severity="unsupported"):
        report["diagnostics"].append(dict(code=code, subject=subject, message=message, repair=repair, severity=severity))

    if engine not in SUPPORTED_ENGINES:
        diagnostic("engine.version", package, f"Unqualified engine {engine}", "Use a qualified exporter/engine pair.")
    dirty = [str(p.get_name()) for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()]
    dirty += [str(p.get_name()) for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()]
    report["dirty_packages"] = sorted(set(dirty))
    if dirty:
        diagnostic("source.unsaved", package, "Unsaved packages exist; a saved dependency hash cannot represent their edits.", "Save your level/assets and inspect again.", "requires_creator_choice")
    levels = [level.get_path_name() for level in unreal.EditorLevelUtils.get_levels(world)]
    report["loaded_levels"] = sorted(levels)
    streaming = [x.get_path_name() for x in unreal.ObjectIterator(unreal.LevelStreaming) if x.get_outer() == world]
    report["streaming_levels"] = sorted(streaming)
    settings = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.WorldSettings)
    partitioned = any(x.get_editor_property("world_partition") is not None for x in settings)
    report["world_partition"] = partitioned
    if streaming or partitioned or len(levels) != 1:
        diagnostic("world.scope", package, "Streaming/partitioned worlds are not qualified by this exporter.", "Export an explicit loaded static sublevel snapshot using a qualified flatten adapter.")
    dependencies = {package}
    textures = {}
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    for actor in actors:
        actor_class = actor.get_class().get_name()
        actor_path = actor.get_path_name()
        try:
            guid = actor.get_editor_property("actor_guid").to_string()
            if not re.fullmatch(r"[0-9A-Fa-f-]{32,36}", guid) or not guid.replace("-", "").strip("0"):
                guid = ""
        except Exception:
            guid = ""
        components = actor.get_components_by_class(unreal.PrimitiveComponent)
        if not components:
            if actor_class not in {"WorldSettings", "Brush", "DefaultPhysicsVolume"}:
                diagnostic("actor.non_geometry", actor_path, f"{actor_class} has no static geometry conversion.", "Keep as source evidence or author its target equivalent in Map Studio.", "requires_creator_choice")
            report["components"].append({"id": guid or actor_path, "actor_guid": guid, "actor_path": actor_path,
                "label": actor.get_actor_label(), "class": actor_class, "disposition": "non_geometry"})
            continue
        for component in components:
            identity = guid + "/" + component.get_name() if guid else component.get_path_name()
            row = {"id": identity, "actor_guid": guid, "actor_path": actor_path,
                   "actor_class": actor_class,
                   "component_name": component.get_name(), "label": actor.get_actor_label(),
                   "class": component.get_class().get_name(), "visible": bool(component.is_visible()),
                   "tags": sorted(str(t) for t in actor.tags) + sorted(str(t) for t in component.component_tags),
                   "world_matrix": _matrix(component.get_world_transform(), unreal), "disposition": "unsupported"}
            report["components"].append(row)
            if component.get_editor_property("is_editor_only"):
                row.update(disposition="editor_only", reason="Native editor-only visualization; no game geometry.")
                continue
            if not guid:
                diagnostic("component.identity", identity, "No persistent actor GUID.", "Save a persistent actor before exporting it.")
            if row["class"] != "StaticMeshComponent":
                diagnostic("component.class", identity, f"{row['class']} needs a dedicated evaluated-snapshot adapter.", "Bake to a supported static mesh explicitly; ISM/HISM, Landscape, spline, skeletal and procedural content are not silently flattened.")
                continue
            if actor_class != "StaticMeshActor":
                diagnostic("actor.evaluated_snapshot", identity, f"{actor_class} may supply construction/runtime behavior.", "Use a qualified explicit evaluated-static snapshot adapter.")
            mesh = component.static_mesh
            if mesh is None:
                diagnostic("mesh.missing", identity, "Static mesh asset is unresolved.", "Resolve the component's asset in Unreal.")
                continue
            mesh_path = mesh.get_path_name()
            row.update(mesh=mesh_path, materials=[], disposition="static_geometry")
            dependencies.add(mesh_path.split(".")[0])
            for slot in range(component.get_num_materials()):
                material = component.get_material(slot)
                key = material.get_path_name() if material else ""
                row["materials"].append(key)
                if not material:
                    diagnostic("material.missing", identity, f"Unresolved material slot {slot}", "Assign an explicit material.")
                    continue
                dependencies.add(key.split(".")[0])
                if key in report["materials"]:
                    continue
                info = {"class": material.get_class().get_name()}
                report["materials"][key] = info
                if info["class"] != "Material":
                    diagnostic("material.instance", key, "Material instances require a qualified effective-parameter bake.", "Bake the effective instance to a supported diffuse material; do not substitute its parent.")
                    continue
                info.update(blend_mode=str(material.get_editor_property("blend_mode")),
                            two_sided=bool(material.get_editor_property("two_sided")),
                            mask_cutoff=float(material.get_editor_property("opacity_mask_clip_value")))
                for name, prop in (("base_color", unreal.MaterialProperty.MP_BASE_COLOR),
                                   ("emissive", unreal.MaterialProperty.MP_EMISSIVE_COLOR),
                                   ("opacity", unreal.MaterialProperty.MP_OPACITY),
                                   ("opacity_mask", unreal.MaterialProperty.MP_OPACITY_MASK),
                                   ("world_position_offset", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)):
                    node = unreal.MaterialEditingLibrary.get_material_property_input_node(material, prop)
                    info[name] = _graph(material, node, unreal)
                    info[name + "_channel"] = str(unreal.MaterialEditingLibrary.get_material_property_input_node_output_name(material, prop))
                if info["world_position_offset"]:
                    diagnostic("material.wpo", key, "World-position offset changes geometry and is not baked.", "Use a qualified evaluated-geometry bake.")
                for texture in unreal.MaterialEditingLibrary.get_used_textures(material):
                    textures[texture.get_path_name()] = texture
                    dependencies.add(texture.get_path_name().split(".")[0])
                # NullRHI commandlets can return no compiled used-texture list.
                # Traverse native expression properties as well, without choosing
                # a texture as a substitute for evaluating a material graph.
                def collect_graph_textures(value):
                    if isinstance(value, dict):
                        texture_path = value.get("texture")
                        if texture_path:
                            texture = unreal.load_asset(texture_path)
                            if not isinstance(texture, unreal.Texture):
                                raise ValueError(f"Unresolved native texture: {texture_path}")
                            textures[texture_path] = texture
                            dependencies.add(texture_path.split(".")[0])
                        for child in value.values():
                            collect_graph_textures(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_graph_textures(child)
                collect_graph_textures(info)
            if mesh_path not in report["meshes"]:
                if mesh.get_editor_property("nanite_settings").enabled:
                    diagnostic("mesh.nanite", mesh_path, "Nanite fallback geometry has not been qualified.", "Supply an explicit non-Nanite mesh or a qualified fallback bake.")
                sections = []
                mesh_editor = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
                if mesh_editor is None:
                    # Commandlets do not instantiate every editor subsystem.
                    # This native read only accesses the supplied static mesh.
                    mesh_editor = unreal.get_default_object(unreal.StaticMeshEditorSubsystem)
                for index in range(mesh.get_num_sections(0)):
                    vertices, indices, normals, uvs, tangents = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(mesh, 0, index)
                    if len(vertices) > MAX_VERTICES or len(indices) % 3:
                        raise ValueError(f"Mesh budget or triangle contract exceeded: {mesh_path}")
                    material_slot = mesh_editor.get_lod_material_slot(mesh, 0, index)
                    if material_slot < 0:
                        raise ValueError(f"Native material-slot lookup failed: {mesh_path}, section {index}")
                    sections.append({"vertices": [_vec(v) for v in vertices], "triangles": [list(indices[i:i+3]) for i in range(0, len(indices), 3)],
                        "normals": [_vec(v) for v in normals], "uvs": [[float(v.x), float(v.y)] for v in uvs],
                        "material_slot": material_slot})
                report["meshes"][mesh_path] = {"lod": 0, "sections": sections}
    if len(report["components"]) > MAX_COMPONENTS:
        raise ValueError("Level exceeds the component budget; choose a smaller source scope.")
    report["components"].sort(key=lambda row: row["id"])
    identities = [row["id"] for row in report["components"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Native component identities collide.")
    # Native dependency closure includes referenced material functions and packages
    # beyond the rendered texture list; fail on unavailable mounts/files.
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    options = unreal.AssetRegistryDependencyOptions()
    options.include_hard_package_references = True
    options.include_soft_package_references = True
    pending, seen = list(dependencies), set()
    while pending:
        key = pending.pop()
        if key in seen or key.startswith("/Script/"):
            continue
        seen.add(key)
        if len(seen) > 20000:
            raise ValueError("Dependency closure exceeds budget.")
        try:
            path = _package_file(key, unreal)
            digest = _sha(path.read_bytes())
            report["dependencies"].append({"package": key, "sha256": digest, "bytes": path.stat().st_size})
            pending.extend(str(p) for p in registry.get_dependencies(key, options))
        except ValueError as error:
            diagnostic("dependency.unresolved", key, str(error), "Restore the asset/plugin or select a qualified package-mount adapter.")
    report["dependencies"].sort(key=lambda item: item["package"])
    report["diagnostics"].sort(key=lambda item: (item["subject"], item["code"]))
    report["state"] = "unsupported" if any(d["severity"] == "unsupported" for d in report["diagnostics"]) else "requires_creator_choice"
    report["revision"] = _sha(_json_bytes(report))
    return report, textures


def export_level(output_dir, *, expected_revision=None, unreal=None):
    """Publish one immutable native snapshot; no source save or overwrite."""
    if unreal is None:
        import unreal
    report, textures = inspect_level(unreal)
    if report["dirty_packages"]:
        raise ValueError("Save unsaved Unreal packages before exporting a revision-bound snapshot.")
    if expected_revision is not None and expected_revision != report["revision"]:
        raise ValueError("Source changed after inspection; inspect again.")
    target = Path(output_dir).resolve()
    project_content = Path(unreal.Paths.project_content_dir()).resolve()
    engine_content = Path(unreal.Paths.engine_content_dir()).resolve()
    if any(target == p or target.is_relative_to(p) for p in (project_content, engine_content)):
        raise ValueError("Export output must not be inside source Content.")
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ghost-export-", dir=str(target.parent)))
    try:
        def put(data, extension):
            if len(data) > MAX_SIDECAR_BYTES:
                raise ValueError("Snapshot sidecar exceeds its size budget.")
            digest = _sha(data)
            path = staging / "assets" / f"{digest}.{extension}"
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            return {"path": path.relative_to(staging).as_posix(), "sha256": digest}

        snapshot = dict(report)
        snapshot["meshes"] = {key: put(_json_bytes(mesh), "json") for key, mesh in report["meshes"].items()}
        snapshot["textures"] = {}
        for key in sorted(textures):
            scratch = staging / "texture.tga"
            task = unreal.AssetExportTask()
            task.object, task.filename = textures[key], str(scratch)
            task.automated, task.prompt, task.replace_identical = True, False, False
            task.exporter = unreal.TextureExporterTGA()
            if not unreal.Exporter.run_asset_export_task(task) or not scratch.is_file():
                raise ValueError(f"Native TGA export failed: {key}; {list(task.errors)}")
            snapshot["textures"][key] = put(scratch.read_bytes(), "tga")
            scratch.unlink()
        snapshot["source_map"] = put(_package_file(report["map_package"], unreal).read_bytes(), "umap")
        # Recheck source file hashes after native reads/exports, before publication.
        for dependency in report["dependencies"]:
            if _sha(_package_file(dependency["package"], unreal).read_bytes()) != dependency["sha256"]:
                raise ValueError(f"Dependency changed while exporting: {dependency['package']}")
        snapshot["source_preserved"] = True
        (staging / "snapshot.json").write_bytes(_json_bytes(snapshot))
        # Windows rename refuses an existing destination, including a raced one.
        if os.name != "nt":
            raise RuntimeError("Atomic directory publication is currently qualified on Windows only.")
        os.rename(staging, target)
        return target / "snapshot.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def commandlet():
    """Bounded entrypoint: -GhostStudioMap=/Game/... -GhostStudioOutput=..."""
    import unreal
    command = unreal.SystemLibrary.get_command_line()

    def option(name):
        match = re.search(r"-" + name + r'=(?:"([^"]+)"|([^\s]+))', command)
        return (match.group(1) or match.group(2)) if match else None

    package, output = option("GhostStudioMap"), option("GhostStudioOutput")
    if not package or not output:
        raise ValueError("GhostStudioMap and GhostStudioOutput are required.")
    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(package):
        raise ValueError(f"Could not load {package}")
    result = export_level(output, unreal=unreal)
    unreal.log("GHOST_STUDIO_SNAPSHOT " + str(result))


if __name__ == "__main__":
    commandlet()
