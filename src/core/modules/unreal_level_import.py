"""Headless static Unreal-level bridge to the existing authored K2 compiler.

Stage: interchange candidate. Native Unreal extraction and retail/UI qualification
remain separate. All transforms are component-local -> Unreal world matrices,
row-major storage with column vectors (not Unreal FMatrix's storage convention).
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.math.unreal_level_space import UE_TO_KOTOR, _array, _matrix, _geometry, transform_normals, has_one_edge_connected_component
from src.math.walkmesh_runtime import K2_WALKABLE_SURFACE_IDS, WalkmeshRuntimeIndex

from .authored_imported_mesh import ImportedMeshRoomPrimitive, ImportedMeshSurface
from .authored_module_objects import AuthoredGameplayPlacement, ModuleEntryPoint
from .authored_module_project import AuthoredModuleMetadata, AuthoredModuleProject, AuthoredRoomSpec
from .module_format import WOKData, WOKFace

COORDINATES = "ue-centimeters-z-up-left-handed"
MATRIX_CONVENTION = "row-major-column-vector-local-to-world"
RESREF = re.compile(r"^[a-z][a-z0-9_]{0,15}$")


@dataclass(frozen=True)
class UnrealLevelImport:
    project: AuthoredModuleProject
    extra_resources: tuple[tuple[str, str, bytes], ...]
    actor_resources: dict[str, str]
    warnings: tuple[str, ...]


def _rows(values):
    return tuple(tuple(float(x) for x in row) for row in values)


def _resource(prefix: str, identity: str) -> str:
    return prefix + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16-len(prefix)]


def build_unreal_level_project(bundle: dict[str, Any], *, independent_components: bool = False) -> UnrealLevelImport:
    """Validate explicit static intent; never guess missing geometry or collision.

    The caller uses IO's load_unreal_level_bundle first. No filesystem access,
    live scene mutation, install, or retail acceptance occurs here.
    """
    if bundle.get("schema") != "ghoststudio.unreal-level/v1":
        raise ValueError("Unsupported Unreal level bundle schema.")
    if bundle.get("coordinates") != COORDINATES or bundle.get("matrix_convention") != MATRIX_CONVENTION:
        raise ValueError("Unsupported coordinate or matrix convention.")
    if bundle.get("uv_origin") != "bottom-left":
        raise ValueError("Exporter must supply normalized bottom-left UV convention.")
    if bundle.get("triangle_winding") != "ccw-relative-to-normals":
        raise ValueError("Exporter must explicitly normalize Unreal's clockwise triangle winding.")
    source = bundle.get("source", {})
    if source.get("scope") != "loaded-static-level" or source.get("unresolved_packages") != []:
        raise ValueError("Unresolved or unsupported source scope; export all dependencies first.")
    if source.get("world_partition") is not False or source.get("streaming_levels") != []:
        raise ValueError("v1 does not support World Partition or streaming levels.")
    if not source.get("engine_version") or not source.get("map_package"):
        raise ValueError("Missing native source identity.")
    module = bundle.get("module", {})
    root = module.get("resref", "")
    if not isinstance(root, str) or not RESREF.fullmatch(root):
        raise ValueError("Module resref must be lowercase ASCII, 1–16 characters, starting with a letter.")
    actors = bundle.get("actors", [])
    if not actors or source.get("component_count") != len(actors):
        raise ValueError("Every exported source component must be accounted for.")
    resources = []
    materials = {}
    material_options = {}
    for item in bundle.get("materials", []):
        key = item["id"]
        if not isinstance(key, str) or not key or key in materials:
            raise ValueError("Material identities must be unique and nonempty.")
        if item.get("conversion") not in {"baked-diffuse-v1", "kotor-diffuse-v1"} or item.get("blend_mode") not in {"opaque", "masked", "additive", "alpha"}:
            raise ValueError("v1 requires explicit opaque/masked/additive/alpha KOTOR conversion; no silent shader approximation.")
        if item.get("blend_mode") != "opaque" and not item.get("conversion_note"):
            raise ValueError("Non-opaque conversion requires an explicit material conversion note.")
        data = item.get("diffuse_bytes")
        if not isinstance(data, bytes) or not data:
            raise ValueError("Material texture bytes must be resolved by bundle IO.")
        resref = _resource("ut", root + ":" + key)
        materials[key] = resref
        material_options[key] = item
        resources.append((resref, "tga", data))
        if item["blend_mode"] in {"masked", "additive"}:
            mode = "punchthrough" if item["blend_mode"] == "masked" else "additive"
            resources.append((resref, "txi", f"blending {mode}\n".encode("ascii")))
    rooms = []
    ids = set()
    actor_resources = {}
    warnings = []
    for actor in actors:
        identity = actor.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Actor/component identities must be unique and nonempty.")
        ids.add(identity)
        role = actor.get("role")
        if role == "excluded":
            if actor.get("class") not in {"DirectionalLight", "SkyLight", "SkyAtmosphere", "ExponentialHeightFog", "VolumetricCloud", "CameraActor"} or not actor.get("reason"):
                raise ValueError("Only explicitly explained environment effects may be excluded in v1.")
            warnings.append(f"Excluded {identity}: {actor['reason']}")
            continue
        if role not in {"static-room", "static-dressing"} or actor.get("class") != "StaticMeshComponent":
            raise ValueError(f"Unsupported source component {identity}; convert it explicitly before export.")
        matrix = _matrix(actor.get("world_matrix"))
        room_ref = _resource("ur", root + ":" + identity)
        if room_ref in actor_resources.values():
            raise ValueError("Generated room identity collision.")
        actor_resources[identity] = room_ref
        surfaces = []
        mesh = actor.get("mesh", {})
        for index, surface in enumerate(mesh.get("surfaces", [])):
            vertices, faces = _geometry(surface, matrix)
            uv = _array(surface.get("uvs"), 2, "UVs")
            normals = _array(surface.get("normals"), 3, "normals")
            if len(uv) != len(vertices) or len(normals) != len(vertices):
                raise ValueError("UVs and normals must match split render vertices.")
            normals = transform_normals(normals, matrix)
            face_normals = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]])
            if (np.einsum("ij,ij->i",face_normals,normals[faces].mean(axis=1)) < -1e-8).any():
                raise ValueError("Triangle winding disagrees with normals; normalize source winding in the exporter.")
            key = surface.get("material")
            if key not in materials:
                raise ValueError(f"Unresolved material: {key}")
            options = material_options[key]
            faces = tuple(tuple(int(x) for x in f) for f in faces)
            if options.get("two_sided"):
                # Odyssey area triangles are culled: explicitly bake the second
                # side, with separate reversed normals, only when requested.
                count = len(vertices)
                if count * 2 > 65535:
                    raise ValueError("Two-sided surface exceeds the MDL u16 vertex limit.")
                vertices = np.r_[vertices, vertices]
                normals = np.r_[normals, -normals]
                uv = np.r_[uv, uv]
                faces += tuple((a+count,c+count,b+count) for a,b,c in faces)
            surfaces.append(ImportedMeshSurface(
                name=f"s{index}", texture=materials[key], vertices=_rows(vertices),
                faces=faces,
                normals=_rows(normals), uvs=_rows(uv), has_shadow=False,
                selfillum=(1.0,1.0,1.0) if options.get("unlit") else (0.0,0.0,0.0),
                transparency_hint=1 if options["blend_mode"] in {"additive", "alpha"} else 0,
                background_geometry=bool(actor.get("metadata", {}).get("backdrop")),
                backdrop=bool(actor.get("metadata", {}).get("backdrop")),
            ))
        if not surfaces:
            raise ValueError(f"No render geometry for {identity}")
        # Empty WOK explicitly means visual-only in Ghost Studio. None means
        # unresolved collision and would invoke the compiler's fallback policy.
        wok = WOKData(name=room_ref)
        if role == "static-room":
            collision = mesh.get("walkmesh")
            if not collision or collision.get("intent") != "authored-walkmesh":
                raise ValueError("Playable rooms require explicit authored walkmesh; render slopes are insufficient.")
            verts, faces = _geometry(collision, matrix)
            ids_for_faces = collision.get("surface_ids", [])
            if len(ids_for_faces) != len(faces) or any(type(s) is not int or s not in K2_WALKABLE_SURFACE_IDS for s in ids_for_faces):
                raise ValueError("Walkmesh requires explicit supported surface IDs; portals need a later room-link stage.")
            if not has_one_edge_connected_component(faces):
                raise ValueError("v1 walkmesh must be welded and edge-connected; disconnected floor islands are unsupported.")
            wok = WOKData(name=room_ref, verts=list(_rows(verts)), faces=[WOKFace(*(int(x) for x in f), surface=s) for f, s in zip(faces, ids_for_faces)])
        elif mesh.get("walkmesh"):
            raise ValueError("Dressing must not silently discard a walkmesh.")
        metadata = {"source": "unreal_level_bundle", "source_component_id": identity,
                    "source_mesh": actor.get("asset_path", ""), "source_world_matrix": actor["world_matrix"],
                    "source_metadata": actor.get("metadata", {})}
        if role == "static-dressing":
            metadata["visual_only"] = True
        primitive = ImportedMeshRoomPrimitive(room_resref=room_ref, surfaces=tuple(surfaces), game="K2", wok=wok, metadata=metadata)
        rooms.append(AuthoredRoomSpec(room_resref=room_ref, primitive=primitive, metadata=metadata))
    playable = [r for r in rooms if r.primitive.wok and r.primitive.wok.faces]
    if not playable:
        raise ValueError("Level has no authored playable walkmesh.")
    if len(playable) != 1:
        raise ValueError("v1 requires one connected collision room; multi-room portals are not yet supported.")
    entry = _array([module.get("entry_position_cm")], 3, "entry position")[0]
    entry = (UE_TO_KOTOR @ np.r_[entry, 1])[:3]
    collision_index = WalkmeshRuntimeIndex(playable[0].primitive.wok, game="K2")
    if collision_index.sample_at(*entry, max_step_up=0.25, max_step_down=0.25) is None:
        raise ValueError("Entry must be on the authored walkmesh within 25 cm vertically.")
    yaw = module.get("entry_yaw_degrees")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("Entry yaw must be finite degrees.")
    component_nodes = {r.metadata["source_component_id"]: [s.name for s in r.primitive.surfaces] for r in rooms}
    warnings.append("Static diffuse/fullbright candidate; Unreal lighting, dynamic effects and gameplay are not translated. Native scale/orientation, UI and retail proof remain pending.")
    project = AuthoredModuleProject(
        metadata=AuthoredModuleMetadata(module_root=root, game="K2", display_name=module.get("name", root),
                                       tag=root, metadata={"lighting_profile": "fullbright"}),
        rooms=tuple(rooms),
        placements=AuthoredGameplayPlacement(entry_point=ModuleEntryPoint(root, tuple(float(x) for x in entry), -math.radians(yaw))),
        notes=tuple(warnings), extra={"unreal_import": {"schema": bundle["schema"], "source": source,
                "bundle_sha256": bundle.get("bundle_sha256"), "component_resources": actor_resources,
                "component_nodes": component_nodes, "room_partition": "independent-components-v1",
                "capability_stage": "interchange_candidate", "game_tested": False}},
    )
    if not independent_components:
        from .unreal_sector_projection import project_unreal_sector
        project = project_unreal_sector(project)
        actor_resources = project.extra["unreal_import"]["component_resources"]
    return UnrealLevelImport(project, tuple(resources), actor_resources, tuple(warnings))
