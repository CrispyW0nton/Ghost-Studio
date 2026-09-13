"""Static KMAP FBX/OBJ export. Core.IO owns assembly, writing and promotion.

Game logic, walkmesh rules, lights and cameras remain in the scene manifest;
the mesh file contains room surfaces and resolved gameplay meshes in their rest pose.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
from src.core.level.level_manifest import build_level_manifest
from src.core.export.export_job import ExportJobRequest, ExportOutputSpec, run_export_job
from src.math.mesh_instance_transform import bake_mesh_instance


def _name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value)) or "mesh"


def assemble_level_model(project, options, *, resource_manager=None, model_loader=None,
                         template_resources=(), placeable_rows=()):
    """Return a fresh static model; never use or mutate the live viewport cache."""
    from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload
    from src.core.modules.authored_module_preview_model import build_authored_module_preview_model
    from src.core.modules.authored_module_placements import authored_gameplay_placement_rows
    from src.core.modules.map_studio_stock_content_preview import (
        TemplateModelResolver, append_stock_content_to_preview_root,
        load_stock_kotor_model, load_kotor_model_from_bytes,
    )
    from src.core.geometry import model_data

    root = ModelNode(name=_name(project.name), flags=int(NodeFlags.HEADER))
    model = KotorModel(name=root.name, supermodel="NULL", root_node=root)
    summary = {"rooms": 0, "placements": 0, "meshes": 0, "vertices": 0, "faces": 0}
    warnings = []
    exported_nodes = []
    cache = {}
    resolver = TemplateModelResolver(resource_manager, project.game,
        template_resources=template_resources, placeable_rows=placeable_rows)

    def load(resref, game=None):
        key = (str(game or project.game).upper(), str(resref).lower())
        if key not in cache:
            override = resolver.model_resource_bytes(key[1]) if resolver else None
            if model_loader:
                cache[key] = model_loader(key[1], key[0])
            elif override:
                cache[key] = load_kotor_model_from_bytes(*override, resref=key[1])
            elif resource_manager:
                cache[key] = load_stock_kotor_model(resource_manager, key[1], key[0])
            else:
                cache[key] = None
        return cache[key]

    owners = {}
    for module in project.modules:
        for room_id in module.rooms:
            if room_id in owners:
                raise ValueError(f"Room {room_id} belongs to more than one module")
            owners[room_id] = module

    def allowed(room):
        module = owners.get(room.room_id)
        return (room.enabled and (not options.visible_only or room.visible)
                and (module is None or (module.enabled and (not options.visible_only or module.visible))))

    def append_meshes(source, identity, transforms=()):
        group = ModelNode(name=_name(identity), flags=int(NodeFlags.HEADER), parent=root)
        for source_node in source.mesh_nodes():
            if (source_node.is_aabb or not source_node.render or not source_node.vertices or not source_node.faces
                    or getattr(source_node, "_gr_map_studio_editor_preview_only", False)
                    or (options.visible_only and getattr(source_node, "_gr_hidden", False))):
                continue
            if any(len(f) != 3 or any(i < 0 or i >= len(source_node.vertices) for i in f) for f in source_node.faces):
                raise ValueError(f"Invalid triangle indices in {identity}/{source_node.name}")
            node = copy.copy(source_node)
            # Copy mutable streams before editing; source models may be cached.
            node.children = []
            node.parent = group
            node.name = f"{_name(identity)}_{len(group.children):03d}_{_name(source_node.name)}"
            world_space = int(source_node.vertex_space or 0) in (1, 2) or getattr(source_node, "_gr_vertices_in_kotor_world", False)
            position, rotation = ((0, 0, 0), (0, 0, 0, 1)) if world_space else source_node.world_transform()
            node.vertices, node.normals, node.faces = bake_mesh_instance(
                source_node.vertices, source_node.normals, source_node.faces,
                position=position, rotation=rotation, transforms=transforms)
            node.uvs = list(source_node.uvs)
            node.uvs_lm = list(source_node.uvs_lm)
            node.face_uvs = list(source_node.face_uvs)
            # Reflected geometry also reverses independent UV corner indices.
            if node.faces and node.faces[0] != tuple(source_node.faces[0]):
                node.face_uvs = [tuple(reversed(row)) if len(row) != 3 else (row[0], row[2], row[1]) for row in node.face_uvs]
            node.face_mats = list(source_node.face_mats)
            node.texture_names = list(source_node.texture_names)
            node.tangents = []  # Importer regenerates tangents for transformed geometry.
            node._gr_level_export_geometry = True
            node.flags = int(NodeFlags.HEADER | NodeFlags.MESH)
            node.position = (0, 0, 0)
            node.rotation = (0, 0, 0, 1)
            node.vertex_space = 0
            # Imported multi-material primitives carry polygon material indices.
            if len(node.texture_names) > 1 and not node.has_lightmap:
                node.imported_ascii = True
                slots = []
                for texture in node.texture_names:
                    slot = copy.copy(node)
                    slot.texture = texture
                    slot.texture_names = [texture]
                    slots.append(slot)
                node._gr_fbx_material_slots = slots
            if not options.include_lightmaps:
                node.lightmap, node.has_lightmap, node.uvs_lm = "", False, []
            if not options.include_textures:
                node.texture, node.texture_names = "", []
                node.lightmap, node.has_lightmap = "", False
                node._gr_fbx_material_slots = [node]
            group.children.append(node)
            summary["meshes"] += 1
            summary["vertices"] += len(node.vertices)
            summary["faces"] += len(node.faces)
            exported_nodes.append({"source_id": str(identity), "source_node": source_node.name,
                                   "mesh_node": node.name, "faces": len(node.faces),
                                   "texture": node.texture, "textures": node.texture_names,
                                   "lightmap": node.lightmap, "lightmap_uv_channel": 1 if node.uvs_lm else None})
        if not group.children:
            raise ValueError(f"No exportable mesh geometry in {identity}")
        root.children.append(group)

    authored = None
    payload = project.extra_sections.get("authored_module")
    authored_refs = set()
    if payload:
        authored = authored_project_from_kmap_payload(payload, fallback_name=project.name, fallback_game=project.game)
        authored_refs = {room.normalised_resref() for room in authored.rooms}
        legacy = {room.model_resref.lower(): room for room in project.rooms}
        active = []
        for room in authored.rooms:
            row = legacy.get(room.normalised_resref())
            if row is None or allowed(row):
                active.append(room)
        # Each authored compiler has already baked primitive transforms; its
        # room header supplies the position. Do not apply synchronized LYT twice.
        authored = replace(authored, rooms=tuple(active))
        preview = build_authored_module_preview_model(authored, include_backdrops=True)
        if preview.warnings:
            raise ValueError("Cannot assemble all authored rooms: " + "; ".join(preview.warnings))
        for group in preview.model.root_node.children:
            if not getattr(group, "_gr_map_studio_authored_room", False):
                continue
            row = legacy.get(str(getattr(group, "_gr_map_studio_room_resref", "")).lower())
            module = owners.get(row.room_id) if row else None
            append_meshes(KotorModel(root_node=group), row.room_id if row else group.name,
                          (module.transform,) if module else ())
            summary["rooms"] += 1

    for room in project.rooms:
        if room.model_resref.lower() in authored_refs or not allowed(room):
            continue
        source = room.metadata.get("export_model") or load(room.model_resref)
        if source is None:
            raise ValueError(f"Missing room model: {room.model_resref or room.name}")
        module = owners.get(room.room_id)
        append_meshes(source, room.room_id, (room.transform, module.transform if module else None))
        summary["rooms"] += 1

    if authored:
        placements = tuple(row for row in authored_gameplay_placement_rows(authored)
                           if row.kind in {"creature", "placeable", "door"})
        if placements:
            placement_root = ModelNode(name="placements", flags=int(NodeFlags.HEADER))
            result = append_stock_content_to_preview_root(model_data, placement_root,
                placements=placements, resource_manager=resource_manager, game=project.game,
                model_loader=load, resolver=resolver)
            if result.unresolved_placement_ids or result.warnings:
                raise ValueError("Cannot resolve all placed models: " + "; ".join((*result.unresolved_placement_ids, *result.warnings)))
            for group in placement_root.children:
                append_meshes(KotorModel(root_node=group), getattr(group, "_gr_map_studio_placement_id", group.name))
                summary["placements"] += 1
    if not summary["meshes"]:
        raise ValueError("The level has no enabled mesh geometry to export")
    model.compute_bounds()
    return model, summary, exported_nodes, warnings


def export_level_scene(project, target, options, *, issues=(), resource_manager=None,
                     model_loader=None, template_resources=(), placeable_rows=(), tex_cache=None,
                     protected_roots=()):
    """Stage the mesh, textures and manifest and promote them as one ExportJob."""
    from src.converters.mesh_converter import FBXExporter, OBJExporter
    from src.core.level.level_export_bridge import LevelExportResult
    target = Path(target).resolve()
    result = LevelExportResult(issues=list(issues))
    try:
        format_name = target.suffix.lower().lstrip(".")
        if format_name not in {"fbx", "obj"}:
            raise ValueError("Choose an output filename ending in .fbx or .obj")
        if options.selected_only or not options.bake_transforms:
            raise ValueError("Level export requires the complete scene with baked transforms")
        model, geometry, nodes, warnings = assemble_level_model(project, options,
            resource_manager=resource_manager, model_loader=model_loader,
            template_resources=template_resources, placeable_rows=placeable_rows)
        result.warnings.extend(warnings)
        with TemporaryDirectory(prefix="ghost-level-export-") as temporary:
            staging = Path(temporary)
            staged_mesh = staging / target.name
            # Use the deterministic complete ASCII writer with meter units. The
            # static scene has no rig or animation takes; no optional SDK needed.
            cache = tex_cache if options.include_textures and options.copy_textures else None
            if cache is None and options.include_textures and options.copy_textures:
                cache = _LevelTextureImages(project, resource_manager, template_resources)
            texture_paths = _copy_level_textures(model, staging, f"{target.stem}_{format_name}", cache, result.warnings) if cache else None
            if format_name == "fbx":
                if not FBXExporter()._export_fbx_ascii(model, str(staged_mesh),
                        texture_paths=texture_paths, compatibility_profile="3ds_max"):
                    raise ValueError("FBX writer failed")
                if not staged_mesh.is_file() or not staged_mesh.read_bytes().startswith(b"; FBX"):
                    raise ValueError("FBX writer did not produce a valid FBX document")
                exported_meshes = len(re.findall(r'Geometry:.*"Mesh"', staged_mesh.read_text(encoding="utf-8")))
            else:
                OBJExporter().export(model, str(staged_mesh), export_rigging=False)
                exported_meshes = sum(line.startswith("o ") for line in staged_mesh.read_text(encoding="utf-8").splitlines())
                mtl_path = staged_mesh.with_suffix(".mtl")
                replacements = {f"map_Kd {name}.tga": f"map_Kd {relative}"
                                for name, relative in (texture_paths or {}).items()}
                material_text = "\n".join(replacements.get(line, line)
                    for line in mtl_path.read_text(encoding="utf-8").splitlines()) + "\n"
                mtl_path.write_text(material_text, encoding="utf-8")
            if exported_meshes != geometry["meshes"]:
                raise ValueError("Export geometry count differs from the assembled level")
            manifest = build_level_manifest(project, issues=list(issues))
            manifest.update({"export_options": options.__dict__, "geometry": geometry,
                "nodes": nodes, "units": "meters", "scene_scope": "static_rest_pose",
                "texture_files": texture_paths or {}, "warnings": result.warnings,
                "objects": project.objects, "lights": project.lights if options.include_lights else [],
                "cameras": project.cameras if options.include_cameras else [],
                "authored_module": project.extra_sections.get("authored_module", {}),
                "export_paths": {format_name: str(target)}})
            if not options.include_walkmesh:
                manifest["walkmeshes"] = []
            # Heavy runtime model objects are an old compatibility input, never
            # serialized into the portable scene manifest.
            for row in manifest["rooms"]:
                row["metadata"].pop("export_model", None)
            manifest_name = target.stem + ("_obj_manifest.json" if format_name == "obj" else "_manifest.json")
            if options.generate_sidecar_manifest:
                (staging / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            files = sorted(path for path in staging.rglob("*") if path.is_file())
            protected = tuple(Path(root).resolve() for root in protected_roots)
            outputs = [ExportOutputSpec((target.parent / path.relative_to(staging)).resolve(), "level_scene") for path in files]
            if any(output.final_path.is_relative_to(root) for output in outputs for root in protected):
                raise ValueError("Export all files outside the source game installation, including texture folders and sidecars")
            def writer(context):
                for path, output in zip(files, outputs):
                    context.write_bytes(output.final_path, path.read_bytes())
            job = run_export_job(ExportJobRequest(job_id="level_scene", kind="level_scene", outputs=outputs, overwrite=options.overwrite), writer=writer)
            if not job.succeeded:
                raise ValueError("Export could not be published: " + str(job.validation_report))
        result.ok, result.code = True, "exported"
        result.fbx_path = str(target) if format_name == "fbx" else ""
        result.output_path = str(target)
        result.manifest_path = str(target.parent / manifest_name) if options.generate_sidecar_manifest else ""
        result.message = f"Exported {target.name}: {geometry['rooms']} rooms, {geometry['placements']} placed models, {geometry['meshes']} meshes, {geometry['faces']} triangles."
        result.warnings.append("Static geometry export: animation and KOTOR gameplay logic are not included in the mesh file.")
    except Exception as exc:
        result.code, result.message = "export_failed", str(exc)
    return result


class _LevelTextureImages:
    """Full-resolution, export-local image cache using the resource owner."""
    def __init__(self, project, manager, resources=()):
        self.project, self.manager, self.images = project, manager, {}
        self.resources = {(str(name).lower(), str(kind).lower().lstrip(".")): data
                          for name, kind, data in resources if str(kind).lower().lstrip(".") in {"tga", "tpc"}}

    def get(self, name):
        from PIL import Image
        from io import BytesIO
        from src.core.graphics.tpc import _is_tpc_data, _load_tpc_bytes
        from src.core.level.map_studio_texture_assets import resolve_project_texture_path
        key = str(name).lower()
        if key not in self.images:
            row = next((row for row in self.project.textures if row.resref.lower() == key and row.include_in_export), None)
            path = resolve_project_texture_path(self.project, row.path) if row and row.path else None
            raw = self.resources.get((key, "tga")) or self.resources.get((key, "tpc"))
            if path and Path(path).is_file():
                raw = Path(path).read_bytes()
            if raw:
                if _is_tpc_data(raw):
                    self.images[key] = _load_tpc_bytes(raw)
                else:
                    with Image.open(BytesIO(raw)) as image:
                        self.images[key] = image.convert("RGBA")
            else:
                self.images[key] = self.manager.load_texture_image(name, self.project.game, max_size=0) if self.manager else None
        return self.images[key]


def _copy_level_textures(model, staging, stem, cache, warnings):
    from PIL import Image
    paths = {}
    directory = staging / (stem + "_textures")
    for node in model.mesh_nodes():
        for name in dict.fromkeys([node.texture, *node.texture_names, node.lightmap]):
            if not name or name.upper() in {"NULL", "BLACK"} or name in paths:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Unsafe texture resource name: {name}")
            image = cache.get(name)
            if image is None:
                warnings.append(f"Texture {name} could not be resolved; its reference is retained.")
                continue
            directory.mkdir(exist_ok=True)
            output = directory / (name + ".png")
            image = image.convert("RGBA")
            # Both existing mesh writers emit 1-v. Match their normal texture
            # export path by converting decoded KOTOR image row order too.
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image.save(output)
            paths[name] = output.relative_to(staging).as_posix()
    return paths
