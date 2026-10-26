"""Headless authored KMAP module export pipeline for Map Studio.

This module turns editable ``AuthoredModuleProject`` intent into the Odyssey
runtime resources required by a module package.  Qt windows should call this
service through ``ModuleEditorController`` instead of assembling ARE/GIT/IFO,
LYT/VIS, WOK, or room MDL/MDX resources themselves.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.validation.kotor_module_engine_contract import (
    KotorModuleEngineContractRequest,
    validate_kotor_module_engine_contract,
)

from .authored_imported_mesh import (
    ImportedMeshRoomPrimitive,
    authored_room_uses_unresolved_stock_geometry,
    imported_mesh_has_explicit_static_runtime_rebuild,
    imported_mesh_room_is_backdrop,
    imported_mesh_room_is_visual_only,
)
from .authored_module_layout import compile_authored_module_layout
from .authored_module_lighting import authored_room_light_payload
from .authored_module_metadata import (
    AuthoredAreaMetadata,
    FULLBRIGHT_LIGHTING_PROFILE,
    authored_area_script_hooks,
    authored_lighting_is_fullbright,
    authored_lighting_profile,
    authored_module_script_hooks,
    compile_authored_module_metadata,
    patch_preserved_stock_are_bytes,
    patch_preserved_stock_ifo_bytes,
)
from .authored_module_objects import (
    AuthoredGameplayPlacement,
    build_git_bytes,
    normalise_resource_resref,
    patch_preserved_stock_git_bytes,
    validate_authored_gameplay_placement_against_walkmesh,
)
from .authored_module_pathing import (
    AuthoredPathAnchor,
    AuthoredPathingRoom,
    compile_authored_pathing_for_rooms,
)
from .authored_module_project import AuthoredModuleProject, compile_authored_room_spec, normalise_resref, validate_authored_module_project
from .authored_room_geometry import AuthoredRoomGeometry, PrimitiveMesh
from .authored_room_materials import compile_authored_room_material_preflight
from .authored_sky_traffic import (
    authored_sky_traffic_list_to_kmap,
    read_authored_project_sky_traffic,
    validate_authored_sky_traffic_collection,
)
from .authored_module_walkmesh import (
    compile_authored_room_connection_walkmeshes,
    offset_wok_data,
    resolve_room_wok_module_offset,
)
from .authored_walkmesh_audit import DOOR_TRANSITION_SURFACE_ID, AuthoredWalkmeshAudit, audit_authored_wok
from .authored_walkmesh_boundaries import apply_authored_walkmesh_boundary_policy_to_geometry
from .custom_module_packager import CustomModulePackRequest, CustomModulePackResult, PackagedModuleResource, package_custom_module
from .dev_module_smoke import (
    DevModulePackageVerification,
    discover_kotor_modules_dir,
    verify_dev_test_module_package,
)
from .module_format import LYTLayout, VISData, WOKData


ENGINE_MODULE_IFO_RESREF = "module"
_MDL_BINARY_PREFIX_SIZE = 12
_MDL_NODE_HEADER_SIZE = 80
_MDL_LIGHT_FLAG = 0x0002
_MDL_LIGHT_HEADER_SIZE = 92
_LIGHT_RUNTIME_FIELD_OFFSETS: tuple[tuple[str, int], ...] = (
    ("dynamic_type", 72),
    ("affect_dynamic", 76),
    ("shadow", 80),
    ("flare", 84),
    ("fading_light", 88),
)


@dataclass(frozen=True)
class AuthoredModuleResourceSummary:
    """Compact resource summary for authored-module export reports."""

    resref: str
    restype: str
    size: int
    source: str


@dataclass(frozen=True)
class AuthoredModuleExportRequest:
    """Options for packaging one authored Map Studio module project."""

    project: AuthoredModuleProject
    output_dir: str = ""
    include_reference_check: bool = False
    include_wok_check: bool = True
    include_game_template_check: bool = False
    game_root_dir: str = ""
    strict: bool = True
    dry_run: bool = False
    create_backups: bool = True
    write_loose_resources: bool = True
    #: Extra (resref, restype, data) payloads bundled into the module: custom
    #: UTC/UTP/UTM/UTD templates, compiled NCS scripts, ported TPC textures.
    extra_resources: tuple[tuple[str, str, bytes], ...] = ()
    #: Deterministic Patch Manager sidecars. These are never packed into the MOD
    #: archive or installed as KOTOR resources.
    secondary_motion_profiles: tuple[tuple[str, bytes], ...] = ()


@dataclass
class AuthoredModuleBuild:
    """In-memory authored module resources consumed by the packager."""

    module_root: str
    game: str
    module: Any
    project: AuthoredModuleProject
    resources: dict[tuple[str, str], Any] = field(default_factory=dict)
    packaged_resources: list[PackagedModuleResource] = field(default_factory=list)
    resource_summaries: list[AuthoredModuleResourceSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthoredModuleExportResult:
    """Result from exporting an authored KMAP module package."""

    ok: bool = False
    module_root: str = ""
    room_resrefs: tuple[str, ...] = ()
    module_path: str = ""
    manifest_path: str = ""
    package_result: CustomModulePackResult | None = None
    package_verification: DevModulePackageVerification | None = None
    resources: list[AuthoredModuleResourceSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    code: str = "not_run"


@dataclass(frozen=True)
class AuthoredModuleInstallPrepRequest:
    """Options for staging an authored module for manual in-game proof."""

    project: AuthoredModuleProject
    output_dir: str = ""
    game_modules_dir: str = ""
    game_root_dir: str = ""
    settings_path: str = ""
    auto_detect_game_modules_dir: bool = False
    overwrite: bool = False
    dry_run: bool = False
    export_request: AuthoredModuleExportRequest | None = None


@dataclass
class AuthoredModuleInstallPrepResult:
    """Safe install-prep result for a manual ``warp <module>`` proof."""

    ok: bool = False
    export_result: AuthoredModuleExportResult | None = None
    installed_module_path: str = ""
    backup_module_path: str = ""
    installed_override_paths: tuple[str, ...] = ()
    backup_override_paths: tuple[str, ...] = ()
    resolved_modules_dir: str = ""
    resolved_game_root_dir: str = ""
    launch_helper_command: str = ""
    elevated_launch_script_path: str = ""
    proof_recording_script_path: str = ""
    checklist_path: str = ""
    proof_manifest_path: str = ""
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    message: str = ""
    code: str = "not_prepared"


@dataclass(frozen=True)
class AuthoredModuleGameProofRequest:
    """Evidence supplied after a real authored-module in-game smoke test."""

    proof_manifest_path: str
    evidence_path: str
    tester: str = ""
    notes: str = ""
    module_loads_in_game: bool = False
    module_identity_matches_authored_resref: bool = False
    player_spawns_on_floor: bool = False
    test_placeable_visible: bool = False
    player_can_walk_on_floor: bool = False
    transition_pathing_sanity_confirmed: bool = False
    no_inherited_base_game_geometry_or_scripted_movers: bool = False
    texture_paint_visible_in_game: bool = False
    terrain_sculpt_and_generated_walkmesh_work_in_game: bool = False
    placed_assets_match_editor_staging: bool = False
    enemy_spawns_hostile: bool = False
    npc_spawns_and_free_roams: bool = False
    terminal_operates: bool = False
    container_opens_with_inventory: bool = False
    puzzle_sequence_unlocks_door: bool = False
    animated_door_operates: bool = False
    configured_transition_operates: bool = False
    player_start_position_and_facing_match: bool = False
    allow_missing_evidence: bool = False


@dataclass
class AuthoredModuleGameProofResult:
    """Result of recording in-game proof for an authored Map Studio module."""

    ok: bool = False
    proof_manifest_path: str = ""
    pack_manifest_path: str = ""
    evidence_path: str = ""
    missing_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    message: str = ""
    code: str = "not_recorded"


@dataclass(frozen=True)
class _ResourceRecord:
    resref: str
    restype: str
    source: str


@dataclass(frozen=True)
class _HydratedResource:
    record: _ResourceRecord
    data: bytes


@dataclass
class _ModuleState:
    name: str
    lyt: LYTLayout
    vis: VISData
    room_woks: dict[str, WOKData]
    room_geometry: dict[str, AuthoredRoomGeometry] = field(default_factory=dict)
    placements: AuthoredGameplayPlacement | None = None
    visual_only_room_resrefs: tuple[str, ...] = ()
    walkmesh_connection_pairs: tuple[tuple[str, str], ...] = ()


def _repo_root_from_here() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "native").is_dir() and (parent / "src").is_dir():
            return parent
    return path.parents[6]


def _preserved_stock_resource(project: AuthoredModuleProject, restype: str) -> bytes | None:
    """Return a byte-exact imported stock resource stored in the KMAP.

    PTH is small enough to preserve in the human-readable KMAP payload as a
    base64 record. Keeping the original graph is safer than rebuilding it from
    room-local WOK coordinates until a pathing-affecting authoring operation
    explicitly invalidates the source graph.
    """

    extra = dict(getattr(project, "extra", {}) or {})
    if str(restype or "").lower() == "pth" and extra.get("stock_pth_dirty"):
        return None
    record = dict((extra.get("stock_resources") or {}).get(str(restype or "").lower()) or {})
    if not record:
        return None
    if str(record.get("encoding") or "").lower() != "base64":
        raise ValueError(f"Preserved stock {restype.upper()} resource uses an unsupported encoding.")
    raw = base64.b64decode(str(record.get("data") or ""), validate=True)
    expected_size = int(record.get("size") or 0)
    expected_sha = str(record.get("sha256") or "").lower()
    if expected_size and len(raw) != expected_size:
        raise ValueError(
            f"Preserved stock {restype.upper()} resource size changed: {len(raw)} != {expected_size}."
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Preserved stock {restype.upper()} resource hash changed: {actual_sha} != {expected_sha}."
        )
    return raw or None


def _import_source_capsule_resources(
    project: AuthoredModuleProject,
) -> tuple[dict[tuple[str, str], bytes], list[str], str, list[dict[str, Any]]]:
    """Read the complete imported capsule for lossless source-resource overlay.

    Stock-module editing must not silently discard ancillary templates,
    scripts, sounds, or resource types Map Studio does not actively author.
    Every addressable capsule entry is retained by key; generated/authored
    resources overlay this mapping later. Unsupported/ambiguous source entries
    block export instead of being omitted.
    """

    source = str((getattr(project, "extra", {}) or {}).get("import_source") or "").strip()
    if not source:
        return {}, [], "", []
    capsule_path = Path(source)
    if not capsule_path.is_file():
        return {}, [f"Imported source capsule is unavailable; ancillary resources cannot be preserved: {capsule_path}"], source, []

    capsule_layers: list[tuple[str, Path]] = []
    if capsule_path.suffix.lower() == ".rim" and not capsule_path.stem.lower().endswith("_s"):
        companion = capsule_path.with_name(f"{capsule_path.stem}_s.rim")
        if companion.is_file():
            # The sibling template/script RIM is the lower-precedence stock
            # layer. The explicitly selected main RIM wins source collisions.
            capsule_layers.append(("companion_s_rim", companion))
    capsule_layers.append(("main_import_source", capsule_path))

    resources: dict[tuple[str, str], bytes] = {}
    blocking: list[str] = []
    layer_metadata: list[dict[str, Any]] = []
    try:
        from pykotor.extract.capsule import LazyCapsule
        from pykotor.resource.type import ResourceType
    except Exception as exc:
        return {}, [f"Imported source capsule reader is unavailable: {exc}"], source, []
    for layer_name, layer_path in capsule_layers:
        try:
            capsule_rows = tuple(LazyCapsule(str(layer_path)).resources())
        except Exception as exc:
            blocking.append(f"Imported source layer {layer_path} could not be inventoried safely: {exc}")
            continue
        layer_resources: dict[tuple[str, str], bytes] = {}
        layer_issue_count = 0
        for row in capsule_rows:
            resref = normalise_resource_resref(str(row.resname() or ""))
            resource_type = row.restype()
            restype = str(getattr(resource_type, "extension", "") or "").strip().lower().lstrip(".")
            raw_type_id = getattr(resource_type, "type_id", None)
            type_id = int(resource_type if raw_type_id is None else raw_type_id)
            writable_type = ResourceType.from_extension(restype) if restype else None
            writable_type_id = int(writable_type) if writable_type is not None else -1
            if not resref or not restype or type_id <= 0 or writable_type_id <= 0 or writable_type_id != type_id:
                blocking.append(
                    f"Imported source layer {layer_path} contains a resource that is not writable through the "
                    f"MOD/RIM archive path ({row.resname()!r}, extension {restype!r}, type id {type_id}, "
                    f"writer type id {writable_type_id}); export cannot preserve it safely."
                )
                layer_issue_count += 1
                continue
            try:
                data = bytes(row.data())
            except Exception as exc:
                blocking.append(f"Imported source resource {resref}.{restype} could not be read from {layer_path}: {exc}")
                layer_issue_count += 1
                continue
            if not data:
                blocking.append(f"Imported source resource {resref}.{restype} in {layer_path} is empty and cannot be preserved safely.")
                layer_issue_count += 1
                continue
            key = (resref, restype)
            previous = layer_resources.get(key)
            if previous is not None and previous != data:
                blocking.append(
                    f"Imported source layer {layer_path} contains conflicting duplicate resource key "
                    f"{resref}.{restype}; export cannot choose one without data loss."
                )
                layer_issue_count += 1
                continue
            layer_resources.setdefault(key, data)
        overrode_lower_precedence = 0
        for key, data in sorted(layer_resources.items()):
            previous = resources.get(key)
            if previous is not None and previous != data:
                overrode_lower_precedence += 1
            resources[key] = data
        layer_metadata.append(
            {
                "name": layer_name,
                "path": str(layer_path),
                "resource_count": len(layer_resources),
                "resource_bytes": sum(len(data) for data in layer_resources.values()),
                "overrode_lower_precedence_count": overrode_lower_precedence,
                "blocking_issue_count": layer_issue_count,
            }
        )
    return resources, blocking, source, layer_metadata


def _room_render_geometry_edited(room: Any) -> bool:
    """Return whether a room's visual geometry has been edited since import.

    WOK-only edits (the floor-fill walkmesh repair) leave the render surfaces
    untouched and keep the room eligible for original-model preservation;
    surface/vertex/face edits set ``render_geometry_edited`` and force a
    static rebuild instead.
    """

    primitive = getattr(room, "primitive", None)
    metadata = dict(getattr(primitive, "metadata", {}) or {})
    return bool(metadata.get("render_geometry_edited", False))


def _room_is_eligible_for_stock_model_preservation(room: Any) -> bool:
    """Return whether an imported stock room should keep its source MDL/MDX."""

    primitive = getattr(room, "primitive", None)
    if not isinstance(primitive, ImportedMeshRoomPrimitive):
        return False
    if _room_render_geometry_edited(room) or imported_mesh_has_explicit_static_runtime_rebuild(primitive):
        return False
    room_metadata = dict(getattr(room, "metadata", {}) or {})
    primitive_metadata = dict(getattr(primitive, "metadata", {}) or {})
    source_kind = str(room_metadata.get("source") or primitive_metadata.get("source") or "").strip().lower()
    return source_kind in {"stock_room_conversion", "stock_module_import"}


def _preserved_stock_room_model(project: Any, room_resref: str) -> tuple[bytes, bytes] | None:
    """Return the imported module's original ``(mdl, mdx)`` bytes for a room.

    Reused verbatim on export for stock-converted rooms whose render geometry
    is unedited, so the room keeps its baked Aurora lights, model animations,
    emitters, and references instead of shipping a flattened static rebuild.
    The bytes come straight from the ``import_source`` capsule already recorded
    in the KMAP, so no game directory or resource manager is needed at export.
    """

    resref = str(room_resref or "").strip().lower()
    if not resref:
        return None
    source = str((getattr(project, "extra", {}) or {}).get("import_source") or "").strip()
    if not source:
        return None
    capsule_path = Path(source)
    if not capsule_path.is_file():
        return None
    try:
        from pykotor.extract.capsule import LazyCapsule
        from pykotor.resource.type import ResourceType as RT

        capsule = LazyCapsule(str(capsule_path))
        mdl = capsule.resource(resref, RT.MDL)
        mdx = capsule.resource(resref, RT.MDX)
    except Exception:
        return None
    if not mdl or not mdx:
        return None
    return bytes(mdl), bytes(mdx)


def _load_module(module_name: str, path: Path) -> Any:
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_mdl_runtime() -> tuple[Any, Any]:
    try:
        from src.core.geometry import model_data as md  # type: ignore
        from src.core.mdl.mdl_writer import MDLBinaryWriter  # type: ignore

        return md, MDLBinaryWriter
    except Exception:
        repo = _repo_root_from_here()
        md = _load_module(
            "src.core.geometry.model_data",
            repo / "native" / "GhostRigger.Core.Math.vcxproj" / "Python" / "src" / "core" / "geometry" / "model_data.py",
        )
        writer_module = _load_module(
            "src.core.mdl.mdl_writer",
            repo / "src" / "core" / "mdl" / "mdl_writer.py",
        )
        return md, writer_module.MDLBinaryWriter


def _primitive_mesh_to_node(md: Any, mesh: PrimitiveMesh, parent: Any, *, fullbright: bool = False) -> Any:
    diffuse = (1.0, 1.0, 1.0) if fullbright else mesh.diffuse
    ambient = (1.0, 1.0, 1.0) if fullbright else mesh.ambient
    # Imported stock surfaces carry their full render recipe in metadata.
    # Re-emitting the ORIGINAL lightmap name + lightmap UVs means an edited
    # vanilla room keeps vanilla in-game lighting (the lightmap TPCs still
    # ship in the game files); without this, replaced rooms render dark.
    metadata = dict(getattr(mesh, "metadata", {}) or {})
    texture = str(mesh.texture or "")
    texture_names = [str(name) for name in tuple(metadata.get("texture_names") or ()) if str(name).strip()]
    if not texture_names:
        texture_names = [
            str(row.get("texture") or "")
            for row in tuple(metadata.get("material_table") or ())
            if isinstance(row, dict) and str(row.get("texture") or "").strip()
        ]
    if not texture_names and texture:
        texture_names = [texture]
    lightmap = str(metadata.get("lightmap") or "")
    uvs_lm = [tuple(float(v) for v in uv[:2]) for uv in tuple(metadata.get("uvs_lm") or ())]
    face_mats = [
        int(value)
        for value in tuple(metadata.get("face_mats") or metadata.get("face_material_ids") or ())
    ]
    if len(face_mats) != len(mesh.faces):
        face_mats = [0] * len(mesh.faces)
    node = md.ModelNode(
        name=mesh.name,
        flags=int(md.NodeFlags.MESH),
        vertices=list(mesh.vertices),
        normals=list(mesh.normals or ((0.0, 0.0, 1.0),) * len(mesh.vertices)),
        uvs=list(mesh.uvs or ((0.0, 0.0),) * len(mesh.vertices)),
        faces=list(mesh.faces),
        face_mats=face_mats,
        texture=texture,
        texture_names=texture_names,
        tex_count=max(1, int(metadata.get("tex_count") or len(texture_names) or 1)),
        lightmap=lightmap,
        has_lightmap=bool(lightmap) and len(uvs_lm) == len(mesh.vertices),
        diffuse=diffuse,
        ambient=ambient,
        vertex_space=0,
    )
    if len(uvs_lm) == len(mesh.vertices):
        node.uvs_lm = uvs_lm
    node.specular = tuple(float(v) for v in tuple(metadata.get("specular") or (0.0, 0.0, 0.0))[:3])
    node.shininess = float(metadata.get("shininess") or 0.0)
    node.alpha = float(metadata.get("alpha", 1.0) if metadata.get("alpha") is not None else 1.0)
    node.has_shadow = False if fullbright else bool(metadata.get("has_shadow", True))
    node.render = bool(metadata.get("render", True))
    node.selfillum = tuple(float(v) for v in tuple(metadata.get("selfillum") or (0.0, 0.0, 0.0))[:3])
    node.transparency_hint = int(metadata.get("transparency_hint") or 0)
    node.beaming = bool(metadata.get("beaming", False))
    node.background_geometry = bool(metadata.get("background_geometry", False))
    node.rotate_texture = bool(metadata.get("rotate_texture", False))
    node.animate_uv = bool(metadata.get("animate_uv", False))
    node.uv_dir_x = float(metadata.get("uv_dir_x") or 0.0)
    node.uv_dir_y = float(metadata.get("uv_dir_y") or 0.0)
    node.uv_jitter = float(metadata.get("uv_jitter") or 0.0)
    node.uv_jitter_speed = float(metadata.get("uv_jitter_speed") or 0.0)
    node.dirt_enabled = bool(metadata.get("dirt_enabled", False))
    node.dirt_texture = int(metadata.get("dirt_texture") or 0)
    node.dirt_coord_space = int(metadata.get("dirt_coord_space") or 0)
    node.hide_in_holograms = bool(metadata.get("hide_in_holograms", False))
    node.mesh_average_point = tuple(
        float(v) for v in tuple(metadata.get("mesh_average_point") or (0.0, 0.0, 0.0))[:3]
    )
    node.mesh_unknown0 = bytes(metadata.get("mesh_unknown0") or b"")[:24].ljust(24, b"\x00")
    node.parent = parent
    node.compute_bounds()
    return node


def _make_walkmesh_node(md: Any, geometry: AuthoredRoomGeometry, parent: Any) -> Any | None:
    """Build the AABB walkmesh node the engine binds the room .wok to.

    Every valid KOTOR room MDL embeds an AABB node (vanilla 001ebo1: WALK1,
    tst_light: walk_a). Its absence made the K2 area loader hit a type-guarded
    NULL dereference at swkotor2+0x4b3a8 on warp. Geometry comes from the
    compiled room WOK so the render mesh and walkmesh agree.
    """

    wok = getattr(geometry, "wok", None)
    verts = [tuple(float(c) for c in v[:3]) for v in tuple(getattr(wok, "verts", ()) or ())]
    faces = [
        (int(f.v1), int(f.v2), int(f.v3))
        for f in tuple(getattr(wok, "faces", ()) or ())
    ]
    if not verts or not faces:
        return None
    node = md.ModelNode(
        name=f"{geometry.room_resref}_wg",
        flags=int(md.NodeFlags.MESH) | int(md.NodeFlags.AABB),
        vertices=verts,
        faces=faces,
        normals=[(0.0, 0.0, 1.0)] * len(verts),
        uvs=[(0.0, 0.0)] * len(verts),
        face_mats=[int(getattr(f, "surface", 1)) for f in tuple(getattr(wok, "faces", ()) or ())],
        texture="NULL",
        diffuse=(1.0, 1.0, 1.0),
        ambient=(1.0, 1.0, 1.0),
        vertex_space=2,  # AABB_WALK: skip rendering
    )
    node.render = False
    node.walkmesh = True
    node.parent = parent
    node.compute_bounds()
    return node


def _walkmesh_walkable_only(wok: Any) -> Any:
    """Reduce the room walkmesh to its walkable floor faces.

    A regenerated room WOK carried the whole room (floor + ceiling + walls):
    18 up-facing floor faces (walkable) plus ~108 NON_WALK wall/ceiling faces.
    The ceiling is a full horizontal cover, so the walkable floor was fully
    enclosed by NON_WALK geometry and PyKotor derived NO perimeter loop
    (perim=0) -- the player could not move.  Vanilla flat rooms ship a
    floor-only WOK (plcaa: 18 faces, all walkable, perim=1).  PyKotor
    re-derives adjacency + perimeter from whatever faces remain, so keeping
    only WALKABLE_IDS faces yields a clean, walkable floor mesh for both the
    .wok file and the model's embedded AABB node.
    """

    from .module_format import WALKABLE_IDS as _WALK, WOKData as _WOKData, WOKFace as _WOKFace

    faces = list(getattr(wok, "faces", ()) or ())
    kept = [f for f in faces if int(getattr(f, "surface", 1)) in _WALK]
    if not kept or len(kept) == len(faces):
        return wok  # already floor-only, or filtering would empty it
    verts = list(getattr(wok, "verts", ()) or ())
    remap: dict[int, int] = {}
    new_verts: list = []

    def _rm(index: int) -> int:
        index = int(index)
        if index not in remap:
            remap[index] = len(new_verts)
            new_verts.append(verts[index])
        return remap[index]

    new_faces = [
        _WOKFace(
            _rm(f.v1),
            _rm(f.v2),
            _rm(f.v3),
            int(getattr(f, "surface", 1)),
            -1,
            -1,
            -1,
            int(getattr(f, "trans1", -1)),
            int(getattr(f, "trans2", -1)),
            int(getattr(f, "trans3", -1)),
        )
        for f in kept
    ]
    return _WOKData(
        name=getattr(wok, "name", "") or "",
        verts=new_verts,
        faces=new_faces,
        relative_hook1=tuple(getattr(wok, "relative_hook1", (0.0, 0.0, 0.0))),
        relative_hook2=tuple(getattr(wok, "relative_hook2", (0.0, 0.0, 0.0))),
        absolute_hook1=tuple(getattr(wok, "absolute_hook1", (0.0, 0.0, 0.0))),
        absolute_hook2=tuple(getattr(wok, "absolute_hook2", (0.0, 0.0, 0.0))),
        position=tuple(getattr(wok, "position", (0.0, 0.0, 0.0))),
    )


def _apply_default_transform_controllers(node: Any) -> None:
    """Give a node the position + orientation controllers every KOTOR node
    needs. Room MDLs authored from scratch had none, so the engine could not
    establish node transforms and crashed placing them (swkotor2+0x4b3a8
    NULL deref). Values are one keyframe at t=0: node position and identity
    (or the node's own) orientation, W-first to match the on-disk layout.
    """

    controllers = list(getattr(node, "controllers", None) or [])
    have = {int(c.get("type", 0) or 0) for c in controllers}
    pos = tuple(float(v) for v in (node.position or (0.0, 0.0, 0.0))[:3])
    rot = tuple(float(v) for v in (getattr(node, "orientation", None) or (0.0, 0.0, 0.0, 1.0))[:4])
    if 8 not in have:  # CTRL_POSITION
        controllers.append({"type": 8, "times": [0.0], "values": [[pos[0], pos[1], pos[2]]], "columns": 3})
    if 20 not in have:  # CTRL_ORIENTATION (on-disk W-first)
        controllers.append({"type": 20, "times": [0.0], "values": [[rot[3], rot[0], rot[1], rot[2]]], "columns": 4})
    node.controllers = controllers


def _make_room_model_bytes(game: str, geometry: AuthoredRoomGeometry, *, lighting_profile: str = "") -> tuple[bytes, bytes]:
    md, Writer = _import_mdl_runtime()
    fullbright = str(lighting_profile or "").strip().lower() == FULLBRIGHT_LIGHTING_PROFILE
    # Room geometry (meshes + walkmesh) are direct children of the root dummy.
    root = md.ModelNode(name=geometry.room_resref, flags=int(md.NodeFlags.HEADER))
    root.children.append(_primitive_mesh_to_node(md, geometry.room_mesh, root, fullbright=fullbright))
    for helper_mesh in geometry.helper_meshes:
        root.children.append(_primitive_mesh_to_node(md, helper_mesh, root, fullbright=fullbright))
    walk_node = _make_walkmesh_node(md, geometry, root)
    if walk_node is not None:
        root.children.append(walk_node)
    # The engine's area-room loader constructs "<roomname>a" and looks it up as
    # a node in the loaded model; a missing node NULL-derefs at swkotor2+0x4b3a8.
    # Vanilla plcaa has a "PLCaaa" dummy (it parented the demo objects, which we
    # strip). Emit an empty "<roomname>a" dummy so the lookup resolves. Captured
    # live: the 0x4b3a8 crash frames held the string "plcaaa". See CHANGES.
    # Playable rooms need the engine's <room>a lookup target. Vanilla pure
    # visual sky rooms (m02aa_sky/m13aa_99z/352narsb) intentionally omit both
    # this synthetic detail dummy and the embedded AABB/WOK node.
    geometry_metadata = dict(getattr(geometry, "metadata", {}) or {})
    if not bool(
        geometry_metadata.get("backdrop_only", False)
        or geometry_metadata.get("visual_only", False)
    ):
        detail_root = md.ModelNode(name=f"{geometry.room_resref}a", flags=int(md.NodeFlags.HEADER))
        detail_root.parent = root
        root.children.append(detail_root)
    # NO synthetic transform controllers.  Empirically, stock K2 room 001ebo1
    # (model_type=0, identical to ours) loads with ZERO controllers on every
    # node -- placement comes purely from each node header's position/rotation.
    # Our earlier synthetic position/orientation controllers were the ONLY
    # trait our crashing plcaa room had that neither loadable room (001ebo1,
    # tst_light) shared: unknown0=0 instead of 0xFFFF, and controllers on the
    # root node (both vanilla roots have none).  The MDLBinaryWriter falls back
    # to node-header transforms when node.controllers is empty, so omitting
    # them matches 001ebo1 exactly.  See CHANGES 2026-07-08.
    model = md.KotorModel(
        name=geometry.room_resref,
        supermodel="NULL",
        classification="area",
        game_version=md.GameVersion.K2 if str(game).upper() == "K2" else md.GameVersion.K1,
        model_type=int(md.ModelClassification.EFFECT),
        root_node=root,
    )
    # NOTE: model.disable_fog=True produced an MDL the engine (and PyKotor's
    # strict reader) could not parse — it overran the buffer during node
    # traversal. Leaving fog at the default keeps the model structurally valid.
    # See CHANGES 2026-07-08: this was the final plcaa room-load crash.
    model.disable_fog = False
    model.compute_bounds()
    return Writer().write(model)


def _neutralize_mdl_light_header_fields(mdl_bytes: bytes, logical_node_offsets: tuple[int, ...]) -> tuple[bytes, list[dict[str, Any]]]:
    """Zero runtime light fields for binary MDL light nodes."""

    if not logical_node_offsets:
        return bytes(mdl_bytes), []
    patched = bytearray(mdl_bytes)
    patches: list[dict[str, Any]] = []
    for logical_offset in logical_node_offsets:
        actual_offset = int(logical_offset) + _MDL_BINARY_PREFIX_SIZE
        light_header_offset = actual_offset + _MDL_NODE_HEADER_SIZE
        row: dict[str, Any] = {
            "logical_node_offset": int(logical_offset),
            "actual_node_offset": actual_offset,
            "light_header_offset": light_header_offset,
            "neutralized": False,
            "before": {},
            "after": {},
        }
        if light_header_offset < 0 or light_header_offset + _MDL_LIGHT_HEADER_SIZE > len(patched):
            row["warning"] = "light header is outside the MDL byte range"
            patches.append(row)
            continue

        changed = False
        before: dict[str, int] = {}
        after: dict[str, int] = {}
        for field_name, field_offset in _LIGHT_RUNTIME_FIELD_OFFSETS:
            absolute = light_header_offset + field_offset
            value = int(struct.unpack_from("<I", patched, absolute)[0])
            before[field_name] = value
            if value != 0:
                struct.pack_into("<I", patched, absolute, 0)
                changed = True
            after[field_name] = int(struct.unpack_from("<I", patched, absolute)[0])
        row["before"] = before
        row["after"] = after
        row["neutralized"] = changed
        patches.append(row)
    return bytes(patched), patches


def neutralize_fullbright_room_mdl_lights(mdl_bytes: bytes, mdx_bytes: bytes = b"", *, game: str = "K1") -> tuple[bytes, dict[str, Any]]:
    """Make imported/generated room MDL light nodes inert for fullbright packages."""

    del game  # The binary node layout is shared for the runtime fields we patch.
    metadata: dict[str, Any] = {
        "source": "map_studio:fullbright_mdl_light_sanitizer",
        "enabled": True,
        "light_node_count": 0,
        "patched_light_node_count": 0,
        "changed_light_node_count": 0,
        "runtime_light_nodes_neutralized": False,
        "neutralized_fields": [name for name, _offset in _LIGHT_RUNTIME_FIELD_OFFSETS],
        "patches": [],
        "warnings": [],
    }
    if not mdl_bytes:
        metadata["warnings"].append("room MDL bytes were empty; no light headers were inspected")
        return bytes(mdl_bytes), metadata

    try:
        from src.core.mdl.ghostrigger_mdl_reader import GhostRiggerMDLBinaryReader  # type: ignore

        reader = GhostRiggerMDLBinaryReader(mdl_bytes, 0, len(mdl_bytes), mdx_bytes or b"", 0, len(mdx_bytes or b""))
        reader.load()
        light_offsets = tuple(
            int(offset)
            for offset, node in sorted(getattr(reader, "_gr_bin_nodes", {}).items())
            if int(getattr(getattr(node, "header", None), "type_id", 0) or 0) & _MDL_LIGHT_FLAG
        )
    except Exception as exc:
        metadata["warnings"].append(f"room MDL light headers could not be inspected: {exc}")
        return bytes(mdl_bytes), metadata

    patched_bytes, patches = _neutralize_mdl_light_header_fields(mdl_bytes, light_offsets)
    valid_patches = [row for row in patches if "warning" not in row]
    metadata["light_node_count"] = len(light_offsets)
    metadata["patched_light_node_count"] = len(valid_patches)
    metadata["changed_light_node_count"] = sum(1 for row in valid_patches if bool(row.get("neutralized")))
    metadata["runtime_light_nodes_neutralized"] = bool(metadata["patched_light_node_count"])
    metadata["patches"] = patches
    warnings = [str(row["warning"]) for row in patches if row.get("warning")]
    if warnings:
        metadata["warnings"].extend(warnings)
    return patched_bytes, metadata


def _mdl_light_sanitizer_to_manifest(enabled: bool, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return fullbright MDL light-node policy evidence for export manifests."""

    return {
        "source": "map_studio:fullbright_mdl_light_sanitizer",
        "enabled": bool(enabled),
        "policy": "Fullbright authored modules neutralize runtime MDL light nodes so lighting stays ARE/material driven.",
        "neutralized_fields": [name for name, _offset in _LIGHT_RUNTIME_FIELD_OFFSETS],
        "room_count": len(rows),
        "light_node_count": sum(int(row.get("light_node_count", 0) or 0) for row in rows),
        "patched_light_node_count": sum(int(row.get("patched_light_node_count", 0) or 0) for row in rows),
        "changed_light_node_count": sum(int(row.get("changed_light_node_count", 0) or 0) for row in rows),
        "runtime_light_nodes_neutralized": any(bool(row.get("runtime_light_nodes_neutralized")) for row in rows),
        "warnings": [
            str(warning)
            for row in rows
            for warning in list(row.get("warnings", []) or [])
        ],
        "rooms": rows,
    }


def _make_packaged(resref: str, restype: str, data: bytes, source: str) -> PackagedModuleResource:
    return PackagedModuleResource(resref=resref, restype=restype, data=data, source=source)


def _resource_summary(resources: dict[tuple[str, str], _HydratedResource]) -> list[AuthoredModuleResourceSummary]:
    return [
        AuthoredModuleResourceSummary(resref=resref, restype=restype, size=len(resource.data), source=resource.record.source)
        for (resref, restype), resource in sorted(resources.items())
    ]


def _merge_authored_extra_resources(
    build: AuthoredModuleBuild,
    extra_resources: tuple[tuple[str, str, bytes], ...],
) -> dict[str, Any]:
    """Merge extras without ever replacing an existing final module key."""

    source_resource_name = "map_studio:stock:import_source"
    generated_keys = {
        key for key, value in build.resources.items() if str(value.record.source or "") != source_resource_name
    }
    collisions: list[dict[str, Any]] = []
    requested_count = 0
    accepted_count = 0
    skipped_count = 0
    replaced_source_keys: list[tuple[str, str]] = []
    for index, (resref, restype, data) in enumerate(tuple(extra_resources or ()), start=1):
        requested_count += 1
        clean_ref = str(resref or "").strip().lower()
        clean_type = str(restype or "").strip().lower()
        if not clean_ref or not clean_type or not data:
            skipped_count += 1
            continue
        record = _make_packaged(clean_ref, clean_type, bytes(data), "map_studio:authored:extra_resource")
        key = record.key
        existing = build.resources.get(key)
        if existing is not None and str(existing.record.source or "") == source_resource_name:
            # Authored extras deliberately overlay their stock source entry.
            # Remove the source package row so the final packager sees one
            # deterministic replacement for this resource key.
            build.packaged_resources = [item for item in build.packaged_resources if item.key != key]
            build.resources.pop(key, None)
            existing = None
            replaced_source_keys.append(key)
        if existing is not None:
            collision_kind = "generated_resource" if key in generated_keys else "duplicate_extra_resource"
            existing_source = str(existing.record.source or "unknown")
            message = (
                f"Authored resource collision for {key[0]}.{key[1]}: extra resource #{index} conflicts with "
                f"{'generated resource' if collision_kind == 'generated_resource' else 'an earlier extra resource'} "
                f"from {existing_source}; final module resource keys must be unique."
            )
            collisions.append(
                {
                    "code": "MAP_STUDIO_RESOURCE_COLLISION",
                    "resref": key[0],
                    "restype": key[1],
                    "kind": collision_kind,
                    "incoming_index": index,
                    "incoming_source": record.source,
                    "existing_source": existing_source,
                    "message": message,
                }
            )
            if message not in build.blocking_issues:
                build.blocking_issues.append(message)
            continue
        build.packaged_resources.append(record)
        build.resources[key] = _HydratedResource(
            _ResourceRecord(key[0], key[1], record.source), bytes(data)
        )
        accepted_count += 1

    build.resource_summaries = _resource_summary(build.resources)
    if replaced_source_keys:
        overlay = dict(build.metadata.get("import_source_overlay") or {})
        preserved_keys = {
            str(value)
            for value in tuple(overlay.get("preserved_resource_keys", ()) or ())
        }
        generated_replaced_keys = {
            str(value)
            for value in tuple(overlay.get("generated_replaced_resource_keys", ()) or ())
        }
        authored_replaced = {f"{resref}.{restype}" for resref, restype in replaced_source_keys}
        preserved_keys.difference_update(authored_replaced)
        overlay.update(
            {
                "preserved_resource_keys": sorted(preserved_keys),
                "preserved_resource_count": len(preserved_keys),
                "authored_extra_replaced_resource_keys": sorted(authored_replaced),
                "authored_extra_replaced_resource_count": len(authored_replaced),
                "replaced_resource_count": len(generated_replaced_keys | authored_replaced),
            }
        )
        build.metadata["import_source_overlay"] = overlay
    gate = {
        "schema": "ghostrigger.map_studio_resource_collision_gate.v1",
        "game": str(build.game or "").upper(),
        "ready": not collisions,
        "generated_resource_count": len(generated_keys),
        "requested_extra_resource_count": requested_count,
        "accepted_extra_resource_count": accepted_count,
        "skipped_extra_resource_count": skipped_count,
        "replaced_import_source_resource_count": len(replaced_source_keys),
        "collision_count": len(collisions),
        "collisions": collisions,
        "final_resource_count": len(build.resources),
    }
    build.metadata["resource_collision_gate"] = gate
    build.metadata["resource_count"] = len(build.resources)
    return gate


def _prepare_secondary_motion_sidecars(
    build: AuthoredModuleBuild,
    profiles: tuple[tuple[str, bytes], ...],
) -> tuple[tuple[str, bytes], ...]:
    """Validate merge-safe profile sidecars without treating them as MOD resources."""

    accepted: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for profile_id, data in tuple(profiles or ()):
        clean_id = str(profile_id or "").strip()
        payload = bytes(data or b"")
        if (
            not clean_id
            or len(clean_id) > 96
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in clean_id)
            or not payload
            or len(payload) > 262_144
        ):
            build.blocking_issues.append("Secondary-motion sidecar identity, size, or payload is invalid.")
            continue
        try:
            document = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            build.blocking_issues.append(f"Secondary-motion profile {clean_id!r} is not valid UTF-8 JSON: {exc}")
            continue
        if (
            not isinstance(document, dict)
            or document.get("schema") != "ghostrigger.secondary_motion_profile.v1"
            or str(document.get("profile_id") or "") != clean_id
        ):
            build.blocking_issues.append(f"Secondary-motion profile {clean_id!r} does not match its v1 identity.")
            continue
        runtime = dict(document.get("runtime") or {})
        if bool(runtime.get("write_enabled")) or str(runtime.get("state") or "capture_only") == "write_enabled":
            build.blocking_issues.append(
                f"Placeable secondary-motion profile {clean_id!r} requested writes without an accepted model-specific lifecycle gate."
            )
            continue
        prior = accepted.get(clean_id)
        if prior is not None and prior != payload:
            build.blocking_issues.append(f"Secondary-motion profile collision for {clean_id!r}.")
            continue
        accepted[clean_id] = payload
    for profile_id, payload in sorted(accepted.items()):
        records.append({
            "profile_id": profile_id,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "runtime_dependency": "cloth-dynamics-k2",
            "write_enabled": False,
        })
    build.metadata["secondary_motion_profiles"] = records
    return tuple(sorted(accepted.items()))


def _validate_final_authored_resource_map(
    build: AuthoredModuleBuild,
    collision_gate: dict[str, Any],
) -> dict[str, Any]:
    """Validate the authoritative merged resource map used for K1/K2 export."""

    game = str(build.game or "").upper()
    game_supported = game in {"K1", "K2"}
    if not game_supported:
        message = f"Final authored resource-map validation does not support game {game or '(missing)'}; expected K1 or K2."
        if message not in build.blocking_issues:
            build.blocking_issues.append(message)

    engine_contract_metadata: dict[str, Any] = {
        "schema": "ghostrigger.map_studio_engine_contract.v1",
        "export_ready": False,
        "blocking_issues": ["Final merged engine-contract validation did not run."],
    }
    try:
        engine_contract = validate_kotor_module_engine_contract(
            KotorModuleEngineContractRequest(
                game=build.game,
                module_resref=build.module_root,
                resources={key: value.data for key, value in build.resources.items()},
                expected_room_resrefs=tuple(sorted(build.module.room_geometry)),
                visual_only_room_resrefs=tuple(
                    build.metadata.get("visual_only_room_resrefs", ())
                    or build.metadata.get("backdrop_room_resrefs", ())
                    or ()
                ),
            )
        )
        engine_contract_metadata = engine_contract.to_dict()
        for warning in engine_contract.warnings:
            if warning not in build.warnings:
                build.warnings.append(warning)
        for issue in engine_contract.blocking_issues:
            if issue not in build.blocking_issues:
                build.blocking_issues.append(issue)
    except Exception as exc:
        message = f"Final merged engine-contract validation could not run: {exc}"
        if message not in build.blocking_issues:
            build.blocking_issues.append(message)
        engine_contract_metadata["blocking_issues"] = [message]

    build.metadata["engine_contract"] = engine_contract_metadata
    audit = {
        "schema": "ghostrigger.map_studio_final_resource_map.v1",
        "game": game,
        "game_supported": game_supported,
        "resource_count": len(build.resources),
        "unique_resource_count": len(set(build.resources)),
        "collision_count": int(collision_gate.get("collision_count", 0) or 0),
        "collisions": list(collision_gate.get("collisions", []) or []),
        "engine_contract": engine_contract_metadata,
        "ready": bool(
            game_supported
            and not collision_gate.get("collisions")
            and engine_contract_metadata.get("export_ready", False)
        ),
    }
    build.metadata["final_resource_map_validation"] = audit
    return audit


def _merged_validation_wok(room_woks: dict[str, "WOKData"]) -> "WOKData | None":
    """Concatenate every room WOK into one walkability-validation mesh.

    Adjacency is not rebuilt — this mesh exists only for point-on-walkable
    checks, never for export.
    """

    from .module_format import WOKFace as _WOKFace

    verts: list[tuple[float, float, float]] = []
    faces: list[Any] = []
    for wok in room_woks.values():
        if wok is None or not getattr(wok, "faces", None):
            continue
        base = len(verts)
        verts.extend(tuple(v) for v in wok.verts)
        for face in wok.faces:
            faces.append(
                _WOKFace(face.v1 + base, face.v2 + base, face.v3 + base, surface=face.surface, adj1=-1, adj2=-1, adj3=-1)
            )
    if not faces:
        return None
    return WOKData(name="validation_merged.wok", verts=verts, faces=faces)


def _lyt_ordered_pathing_rooms(layout: Any, room_geometries: dict[str, AuthoredRoomGeometry]) -> tuple[AuthoredPathingRoom, ...]:
    """Return every compiled room in the exact LYT index/coordinate contract.

    WOK transition destinations are LYT room indices, so omitting an empty
    backdrop room or sorting by resref silently retargets portals.  Authored
    WOKs remain room-local and receive their LYT translation here.  Imported
    WOKs have already been normalized to module space in ``build_authored_module``
    and must not receive that translation a second time.
    """

    lyt_rows = tuple(getattr(getattr(layout, "lyt", None), "rooms", ()) or ())
    lyt_order = tuple(normalise_resref(getattr(row, "model", "")) for row in lyt_rows)
    declared_order = tuple(normalise_resref(value) for value in tuple(getattr(layout, "room_resrefs", ()) or ()))
    if not lyt_rows:
        raise ValueError("Authored PTH generation requires at least one LYT room row.")
    if lyt_order != declared_order:
        raise ValueError(
            "Authored PTH generation requires layout.room_resrefs to match the exact LYT room order."
        )
    if any(not room_resref for room_resref in lyt_order):
        raise ValueError("Authored PTH generation found an empty LYT room resref.")
    if len(set(lyt_order)) != len(lyt_order):
        raise ValueError("Authored PTH generation requires unique LYT room resrefs.")

    missing = tuple(room_resref for room_resref in lyt_order if room_resref not in room_geometries)
    extra = tuple(room_resref for room_resref in room_geometries if room_resref not in set(lyt_order))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing compiled rooms: {', '.join(missing)}")
        if extra:
            details.append(f"compiled rooms absent from LYT: {', '.join(extra)}")
        raise ValueError("Authored PTH room set does not match LYT (" + "; ".join(details) + ").")

    result: list[AuthoredPathingRoom] = []
    for room_resref, row in zip(lyt_order, lyt_rows, strict=True):
        geometry = room_geometries[room_resref]
        imported_wok = bool(dict(getattr(geometry, "metadata", {}) or {}).get("imported_wok"))
        position = (
            (0.0, 0.0, 0.0)
            if imported_wok
            else (float(getattr(row, "x", 0.0)), float(getattr(row, "y", 0.0)), float(getattr(row, "z", 0.0)))
        )
        result.append(AuthoredPathingRoom(room_resref=room_resref, wok=geometry.wok, position=position))
    return tuple(result)


def _entry_room_wok(project: AuthoredModuleProject, room_geometries: dict[str, AuthoredRoomGeometry]) -> tuple[str, WOKData | None]:
    room_names = [normalise_resref(room.room_resref) for room in project.rooms]
    if not room_names:
        return "", None
    entry = normalise_resref(project.placements.entry_point.area_resref)
    preferred = entry if entry in room_geometries else room_names[0]
    geometry = room_geometries.get(preferred)
    if geometry is None:
        return preferred, None
    return preferred, geometry.wok


def _path_anchors_from_walkability(placements: AuthoredGameplayPlacement, checks: Any) -> tuple[AuthoredPathAnchor, ...]:
    ok_labels = {str(getattr(check, "label", "")): bool(getattr(check, "ok", False)) for check in list(getattr(checks, "checks", ()) or ())}
    anchors: list[AuthoredPathAnchor] = []
    if ok_labels.get("entry_point", False):
        anchors.append(AuthoredPathAnchor("entry_point", placements.entry_point.position, metadata={"kind": "entry_point"}))

    def append_spatial_anchor(kind: str, index: int, item: Any) -> None:
        if not hasattr(item, "position"):
            return
        template = str(getattr(item, "template_resref", "") or "")
        tag = str(getattr(item, "tag", "") or "")
        label = f"{kind}:{tag or template or f'{kind}_{index + 1}'}"
        if ok_labels.get(label, False):
            anchors.append(
                AuthoredPathAnchor(
                    label,
                    tuple(getattr(item, "position")),
                    metadata={
                        "kind": kind,
                        "index": index,
                        "template_resref": template,
                        "tag": tag,
                    },
                )
            )

    for index, creature in enumerate(placements.creatures):
        append_spatial_anchor("creature", index, creature)
    for index, door in enumerate(placements.doors):
        append_spatial_anchor("door", index, door)
    for index, trigger in enumerate(placements.triggers):
        append_spatial_anchor("trigger", index, trigger)
    for index, encounter in enumerate(placements.encounters):
        append_spatial_anchor("encounter", index, encounter)
    for index, placeable in enumerate(placements.placeables):
        append_spatial_anchor("placeable", index, placeable)
    for index, waypoint in enumerate(placements.waypoints):
        append_spatial_anchor("waypoint", index, waypoint)
    return tuple(anchors)


def _vec3_to_manifest(value: Any) -> list[float]:
    values = list(value or (0.0, 0.0, 0.0))
    while len(values) < 3:
        values.append(0.0)
    return [float(values[0]), float(values[1]), float(values[2])]


def _walkability_to_manifest(walkability: Any) -> dict[str, Any]:
    if walkability is None:
        return {
            "ok": False,
            "checks": [],
            "warnings": [],
            "blocking_issues": ["No walkability validation was run."],
        }
    return {
        "ok": bool(getattr(walkability, "ok", False)),
        "checks": [
            {
                "label": str(getattr(check, "label", "")),
                "position": _vec3_to_manifest(getattr(check, "position", (0.0, 0.0, 0.0))),
                "ok": bool(getattr(check, "ok", False)),
                "face_index": int(getattr(check, "face_index", -1)),
                "surface_id": int(getattr(check, "surface_id", -1)),
                "message": str(getattr(check, "message", "")),
            }
            for check in list(getattr(walkability, "checks", ()) or ())
        ],
        "warnings": list(getattr(walkability, "warnings", ()) or ()),
        "blocking_issues": list(getattr(walkability, "blocking_issues", ()) or ()),
    }


def _walkmesh_audit_to_manifest(audit: AuthoredWalkmeshAudit) -> dict[str, Any]:
    return {
        "room_resref": audit.room_resref,
        "ready": bool(audit.ready),
        "face_count": int(audit.face_count),
        "walkable_face_count": int(audit.walkable_face_count),
        "non_walk_face_count": int(audit.non_walk_face_count),
        "walkable_component_count": int(audit.walkable_component_count),
        "disconnected_component_count": int(audit.disconnected_component_count),
        "largest_walkable_component_faces": int(audit.largest_walkable_component_faces),
        "invalid_face_count": int(audit.invalid_face_count),
        "degenerate_face_count": int(audit.degenerate_face_count),
        "non_manifold_edge_count": int(audit.non_manifold_edge_count),
        "transition_surface_face_count": int(audit.transition_surface_face_count),
        "steep_walkable_face_count": int(audit.steep_walkable_face_count),
        "max_walkable_slope_degrees": float(audit.max_walkable_slope_degrees),
        "max_allowed_walkable_slope_degrees": float(audit.max_allowed_walkable_slope_degrees),
        "warnings": list(audit.warnings),
        "blocking_messages": list(audit.blocking_messages),
    }


def _transition_link_rows(placements: AuthoredGameplayPlacement) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, items in (
        ("door", tuple(placements.doors or ())),
        ("trigger", tuple(placements.triggers or ())),
    ):
        for index, item in enumerate(items):
            linked_to = str(getattr(item, "linked_to", "") or "").strip()
            if not linked_to:
                continue
            linked_module = normalise_resource_resref(getattr(item, "linked_to_module", ""))
            rows.append(
                {
                    "kind": kind,
                    "index": index,
                    "tag": str(getattr(item, "tag", "") or ""),
                    "template_resref": normalise_resource_resref(getattr(item, "template_resref", "")),
                    "linked_to": linked_to,
                    "linked_to_module": linked_module,
                    "linked_to_flags": int(getattr(item, "linked_to_flags", 0) or 0),
                    "transition_destination": int(getattr(item, "transition_destination", 0) or 0),
                    "requires_wok_transition_surface": True,
                }
            )
    return rows


def _transition_surface_gate_to_manifest(
    placements: AuthoredGameplayPlacement,
    audits: tuple[AuthoredWalkmeshAudit, ...],
) -> dict[str, Any]:
    rows = _transition_link_rows(placements)
    transition_surface_face_count = sum(int(audit.transition_surface_face_count) for audit in audits)
    warnings: list[str] = []
    blocking: list[str] = []
    if rows and transition_surface_face_count <= 0:
        # Vanilla WOKs (e.g. plcaa) ship linked transitions without any
        # surface-18 faces and load fine in game, so a stock round-trip must
        # not be blocked; from-scratch authors still get the nudge.
        warnings.append(
            f"Authored module has {len(rows)} linked door/trigger transition(s) but no WOK DOOR/transition surface "
            f"face(s). Paint at least one doorway walkmesh face as surface {DOOR_TRANSITION_SURFACE_ID} if the "
            "transition should trigger from the floor."
        )
    if not rows and transition_surface_face_count > 0:
        warnings.append(
            f"Generated WOK includes {transition_surface_face_count} DOOR/transition surface face(s) but no linked door/trigger transition intent."
        )
    return {
        "ready": not blocking,
        "required_transition_count": len(rows),
        "transition_surface_face_count": int(transition_surface_face_count),
        "transition_surface_id": DOOR_TRANSITION_SURFACE_ID,
        "references": rows,
        "warnings": warnings,
        "blocking_messages": blocking,
    }


def _walkmesh_gate_to_manifest(
    audits: tuple[AuthoredWalkmeshAudit, ...],
    *,
    walkability: dict[str, Any],
    pathing: dict[str, Any],
    transition_surface_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocking = [message for audit in audits for message in audit.blocking_messages]
    blocking.extend(str(message) for message in walkability.get("blocking_issues", []) or [])
    transition_gate = dict(transition_surface_gate or {})
    blocking.extend(str(message) for message in transition_gate.get("blocking_messages", []) or [])
    if not pathing:
        blocking.append("Authored module PTH pathing was not compiled.")
    return {
        "ready": not blocking,
        "room_count": len(audits),
        "walkable_face_count": sum(int(audit.walkable_face_count) for audit in audits),
        "non_walk_face_count": sum(int(audit.non_walk_face_count) for audit in audits),
        "walkable_component_count": sum(int(audit.walkable_component_count) for audit in audits),
        "disconnected_walkmesh_room_count": sum(1 for audit in audits if int(audit.walkable_component_count) > 1),
        "invalid_face_count": sum(int(audit.invalid_face_count) for audit in audits),
        "degenerate_face_count": sum(int(audit.degenerate_face_count) for audit in audits),
        "non_manifold_edge_count": sum(int(audit.non_manifold_edge_count) for audit in audits),
        "transition_surface_face_count": sum(int(audit.transition_surface_face_count) for audit in audits),
        "transition_surface_gate": transition_gate,
        "steep_walkable_face_count": sum(int(audit.steep_walkable_face_count) for audit in audits),
        "max_walkable_slope_degrees": max((float(audit.max_walkable_slope_degrees) for audit in audits), default=0.0),
        "max_allowed_walkable_slope_degrees": max((float(audit.max_allowed_walkable_slope_degrees) for audit in audits), default=0.0),
        "gameplay_anchor_check_count": len(list(walkability.get("checks", []) or [])),
        "gameplay_anchor_checks_passed": bool(walkability.get("ok", False)),
        "pth_compiled": bool(pathing),
        "pth_point_count": int(pathing.get("point_count", 0) or 0),
        "pth_connection_count": int(pathing.get("connection_count", 0) or 0),
        "pathing_anchor_labels": list(pathing.get("anchor_labels", []) or []),
        "warnings": [warning for audit in audits for warning in audit.warnings]
        + [str(warning) for warning in transition_gate.get("warnings", []) or []],
        "blocking_messages": blocking,
        "rooms": [_walkmesh_audit_to_manifest(audit) for audit in audits],
    }


def _visibility_to_manifest(project: AuthoredModuleProject, layout: Any | None) -> dict[str, Any]:
    """Return a modder-facing VIS graph summary for export manifests."""

    room_names = [normalise_resref(room.room_resref) for room in tuple(project.rooms or ()) if normalise_resref(room.room_resref)]
    room_set = set(room_names)
    if layout is None:
        return {
            "source": "map_studio:authored:vis",
            "ready": False,
            "status": "Not compiled",
            "vis_resource": f"{project.module_root}.vis",
            "room_count": len(room_names),
            "vis_entry_count": 0,
            "link_count": 0,
            "cross_room_link_count": 0,
            "entries": [],
            "isolated_rooms": room_names if len(room_names) > 1 else [],
            "missing_targets": [],
            "warnings": [],
            "blocking_messages": ["VIS visibility resource was not compiled."],
            "fix_hint": "Resolve layout/VIS validation errors, then rebuild the module package.",
        }

    visibility_source = getattr(getattr(layout, "vis", None), "visibility", {}) or {}
    entries: list[dict[str, Any]] = []
    normalised_visibility: dict[str, list[str]] = {}
    missing_targets: list[dict[str, str]] = []
    for room_name in room_names:
        raw_targets = tuple(visibility_source.get(room_name, ()) or ())
        targets = []
        for raw_target in raw_targets:
            target = normalise_resref(raw_target)
            if not target or target in targets:
                continue
            targets.append(target)
            if target not in room_set:
                missing_targets.append({"room": room_name, "target": target})
        normalised_visibility[room_name] = targets
        entries.append({"room": room_name, "visible_rooms": list(targets)})

    isolated_rooms: list[str] = []
    if len(room_names) > 1:
        for room_name in room_names:
            outgoing = {target for target in normalised_visibility.get(room_name, ()) if target != room_name}
            incoming = {
                other_room
                for other_room, targets in normalised_visibility.items()
                if other_room != room_name and room_name in set(targets)
            }
            if not outgoing and not incoming:
                isolated_rooms.append(room_name)

    link_count = sum(len(targets) for targets in normalised_visibility.values())
    cross_room_link_count = sum(
        1
        for room_name, targets in normalised_visibility.items()
        for target in targets
        if target != room_name
    )
    warnings = list(getattr(layout, "warnings", ()) or ())
    if isolated_rooms:
        warnings.append(
            f"VIS graph has {len(isolated_rooms)} isolated room(s); add room visibility links before calling the module visually game-tested."
        )
    blocking_messages = [
        f"VIS room {item['room']} references missing room {item['target']}."
        for item in missing_targets
    ]
    ready = not missing_targets and not isolated_rooms
    if blocking_messages:
        status = f"Blocked: {len(blocking_messages)} broken VIS target(s)"
        fix_hint = "Remove missing VIS targets or add matching authored rooms."
    elif isolated_rooms:
        status = "Needs visibility links"
        fix_hint = "Link rooms that should see each other in the room visibility graph."
    else:
        status = "Ready"
        fix_hint = ""
    return {
        "source": "map_studio:authored:vis",
        "ready": ready,
        "status": status,
        "vis_resource": f"{project.module_root}.vis",
        "room_count": len(room_names),
        "vis_entry_count": len(normalised_visibility),
        "link_count": link_count,
        "cross_room_link_count": cross_room_link_count,
        "entries": entries,
        "isolated_rooms": isolated_rooms,
        "missing_targets": missing_targets,
        "warnings": warnings,
        "blocking_messages": blocking_messages,
        "fix_hint": fix_hint,
    }


def _lightmap_metadata(project: AuthoredModuleProject) -> dict[str, Any]:
    metadata = getattr(getattr(project, "metadata", None), "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    source = metadata.get("lightmap")
    if source is None:
        source = metadata.get("lightmap_status")
    if isinstance(source, dict):
        return dict(source)
    if isinstance(source, str) and source.strip():
        return {"status": source.strip()}
    return {}


def _lightmap_rooms(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = tuple(value)
    else:
        values = ()
    return tuple(dict.fromkeys(normalise_resref(item) for item in values if normalise_resref(item)))


def _lighting_to_manifest(project: AuthoredModuleProject, room_lights: list[dict[str, Any]]) -> dict[str, Any]:
    """Return explicit lighting/lightmap proof state for export manifests."""

    lighting_profile = authored_lighting_profile(project.metadata)
    fullbright = authored_lighting_is_fullbright(project.metadata)
    room_resrefs = tuple(
        normalise_resref(getattr(room, "room_resref", ""))
        for room in tuple(project.rooms or ())
        if normalise_resref(getattr(room, "room_resref", ""))
    )
    lit_room_set = {
        normalise_resref(row.get("room_resref", ""))
        for row in room_lights
        if isinstance(row, dict) and normalise_resref(row.get("room_resref", ""))
    }
    rooms_with_lights = tuple(room for room in room_resrefs if room in lit_room_set)
    rooms_without_lights = tuple(room for room in room_resrefs if room not in lit_room_set)
    lightmap = _lightmap_metadata(project)
    raw_status = str(lightmap.get("status") or "not_started").strip().lower().replace("-", "_").replace(" ", "_")
    manifest_path = str(lightmap.get("manifest_path") or lightmap.get("path") or lightmap.get("proof_manifest_path") or "")
    lightmap_rooms = _lightmap_rooms(
        lightmap.get("rooms")
        or lightmap.get("baked_rooms")
        or lightmap.get("lightmapped_rooms")
        or lightmap.get("room_resrefs")
    )
    game_tested_lighting = bool(lightmap.get("game_tested") or raw_status in {"game_tested", "game_tested_lighting"})
    warnings: list[str] = []
    if rooms_without_lights and room_lights:
        warnings.append(f"{len(rooms_without_lights)} room(s) have no authored lights yet: {', '.join(rooms_without_lights)}.")
    if lightmap_rooms:
        missing_lightmap_rooms = tuple(room for room in room_resrefs if room not in set(lightmap_rooms))
        if missing_lightmap_rooms:
            warnings.append(f"Lightmap coverage is missing {len(missing_lightmap_rooms)} room(s): {', '.join(missing_lightmap_rooms)}.")

    if not room_resrefs:
        ready = False
        status = "Needs authored rooms"
        lightmap_status = "not_started"
        fix_hint = "Create authored rooms before planning room lights or lightmaps."
    elif game_tested_lighting:
        ready = True
        status = "Game-tested lighting"
        lightmap_status = "game_tested"
        fix_hint = "Keep the lighting proof manifest and in-game screenshot/video with the staged package."
    elif raw_status in {"baked", "export_candidate", "ready"} and manifest_path:
        ready = True
        status = "Lightmap export candidate"
        lightmap_status = "export_candidate"
        fix_hint = "Install the module and verify lighting/lightmap appearance in-game before calling it game-tested."
    elif fullbright:
        ready = True
        status = "Fullbright export candidate"
        lightmap_status = "fullbright_export_candidate"
        fix_hint = "Install the module and verify fullbright actor/room visibility in-game before calling it game-tested."
    elif room_lights:
        ready = False
        status = "Viewport lit only"
        lightmap_status = raw_status if raw_status != "not_started" else "viewport_lit_only"
        warnings.append(
            "Authored room lights are viewport/editor intent only until a baked lightmap manifest or in-game lighting proof is recorded."
        )
        fix_hint = "Bake or attach a lightmap manifest, then run an in-game lighting proof pass."
    else:
        ready = False
        status = "Lighting not planned"
        lightmap_status = raw_status
        warnings.append("No authored room lights or lightmap plan exists yet; viewport lighting is not game-tested module lighting.")
        fix_hint = "Add key/fill/ambient room lights and record lightmap planning before packaging a visual-quality map."

    return {
        "source": "map_studio:authored:lighting",
        "ready": ready,
        "status": status,
        "lighting_profile": lighting_profile or "standard",
        "light_count": len(room_lights),
        "room_count": len(room_resrefs),
        "rooms_with_lights": list(rooms_with_lights),
        "rooms_without_lights": list(rooms_without_lights),
        "lightmap_status": lightmap_status,
        "lightmap_manifest_path": manifest_path,
        "lightmap_room_count": len(lightmap_rooms),
        "lightmap_rooms": list(lightmap_rooms),
        "game_tested_lighting": game_tested_lighting,
        "warnings": warnings,
        "fix_hint": fix_hint,
    }


def _positioned_expectations(kind: str, items: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(tuple(items or ())):
        if not hasattr(item, "position"):
            continue
        template = str(getattr(item, "template_resref", "") or "")
        tag = str(getattr(item, "tag", "") or "")
        rows.append(
            {
                "kind": kind,
                "index": index,
                "template_resref": template,
                "tag": tag,
                "label": f"{kind}:{tag or template or f'{kind}_{index + 1}'}",
                "position": _vec3_to_manifest(getattr(item, "position")),
                "bearing": float(getattr(item, "bearing", 0.0) or 0.0),
            }
        )
    return rows


def _mesh_material_uv_record(
    mesh: Any,
    *,
    role: str,
    floor_surface_id: int = -1,
    floor_surface_name: str = "",
    lighting_profile: str = "",
) -> dict[str, Any]:
    vertices = tuple(getattr(mesh, "vertices", ()) or ())
    uvs = tuple(getattr(mesh, "uvs", ()) or ())
    mesh_metadata = dict(getattr(mesh, "metadata", {}) or {})
    uvs_lm = tuple(getattr(mesh, "uvs_lm", ()) or mesh_metadata.get("uvs_lm") or ())
    lightmap = str(getattr(mesh, "lightmap", "") or mesh_metadata.get("lightmap") or "")
    fullbright = str(lighting_profile or "").strip().lower() == FULLBRIGHT_LIGHTING_PROFILE
    diffuse = (1.0, 1.0, 1.0) if fullbright else getattr(mesh, "diffuse", (0.8, 0.8, 0.8))
    ambient = (1.0, 1.0, 1.0) if fullbright else getattr(mesh, "ambient", (0.35, 0.35, 0.35))
    return {
        "role": role,
        "mesh_name": str(getattr(mesh, "name", "") or ""),
        "texture": str(getattr(mesh, "texture", "") or ""),
        "lighting_profile": lighting_profile or "standard",
        "diffuse": _vec3_to_manifest(diffuse),
        "ambient": _vec3_to_manifest(ambient),
        "vertex_count": len(vertices),
        "face_count": len(tuple(getattr(mesh, "faces", ()) or ())),
        "uv_count": len(uvs),
        "uv_complete": bool(vertices) and len(uvs) == len(vertices),
        "uv_coordinate_space": "mesh_uv0",
        "lightmap": lightmap,
        "lightmap_uv_count": len(uvs_lm),
        "lightmap_uv_complete": bool(vertices) and len(uvs_lm) == len(vertices),
        "has_lightmap": bool(lightmap) and bool(vertices) and len(uvs_lm) == len(vertices),
        "wok_surface_id": int(floor_surface_id),
        "wok_surface_name": str(floor_surface_name or ""),
    }


def _room_material_uv_manifest(room_geometry: dict[str, Any], *, lighting_profile: str = "") -> list[dict[str, Any]]:
    """Summarize authored material/UV intent paired with generated MDL/WOK."""

    rows: list[dict[str, Any]] = []
    for room_resref, geometry in sorted(room_geometry.items()):
        metadata = dict(getattr(geometry, "metadata", {}) or {})
        floor_surface_id = int(metadata.get("floor_surface_id", -1) or -1)
        floor_surface_name = str(metadata.get("floor_surface_name", "") or "")
        meshes = [
            _mesh_material_uv_record(
                geometry.room_mesh,
                role="room_mesh",
                floor_surface_id=floor_surface_id,
                floor_surface_name=floor_surface_name,
                lighting_profile=lighting_profile,
            )
        ]
        for index, helper in enumerate(tuple(getattr(geometry, "helper_meshes", ()) or ())):
            meshes.append(
                _mesh_material_uv_record(
                    helper,
                    role=f"helper_mesh_{index + 1}",
                    floor_surface_id=floor_surface_id,
                    floor_surface_name=floor_surface_name,
                    lighting_profile=lighting_profile,
                )
            )
        rows.append(
            {
                "room_resref": room_resref,
                "texture": meshes[0]["texture"] if meshes else "",
                "lighting_profile": lighting_profile or "standard",
                "floor_surface_id": floor_surface_id,
                "floor_surface_name": floor_surface_name,
                "mesh_count": len(meshes),
                "all_mesh_uvs_complete": all(bool(row["uv_complete"]) for row in meshes),
                "meshes": meshes,
            }
        )
    return rows


def _template_dependency_rows(
    placements: AuthoredGameplayPlacement,
    *,
    packaged_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    packaged = packaged_keys or set()
    rows: list[dict[str, Any]] = []
    for kind, restype, items in (
        ("creature", "utc", tuple(placements.creatures or ())),
        ("door", "utd", tuple(placements.doors or ())),
        ("trigger", "utt", tuple(placements.triggers or ())),
        ("encounter", "ute", tuple(placements.encounters or ())),
        ("sound", "uts", tuple(placements.sounds or ())),
        ("store", "utm", tuple(placements.stores or ())),
        ("placeable", "utp", tuple(placements.placeables or ())),
        ("waypoint", "utw", tuple(placements.waypoints or ())),
    ):
        for index, item in enumerate(items):
            resref = normalise_resource_resref(getattr(item, "template_resref", ""))
            if not resref:
                continue
            is_packaged = (resref, restype) in packaged
            rows.append(
                {
                    "kind": kind,
                    "index": index,
                    "template_resref": resref,
                    "restype": restype,
                    "tag": str(getattr(item, "tag", "") or ""),
                    "status": "packaged" if is_packaged else "external_or_base_game",
                    "packaged": is_packaged,
                    "required": True,
                    "message": (
                        f"{resref}.{restype} is included in this module package."
                        if is_packaged
                        else f"{resref}.{restype} must resolve from the base game, Override, or another installed mod."
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["kind"], row["template_resref"], row["restype"], row["tag"]))


def _script_reference_rows(
    project: AuthoredModuleProject,
    *,
    packaged_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    packaged = packaged_keys or set()
    rows: list[dict[str, Any]] = []
    for scope, hooks in (
        ("module", authored_module_script_hooks(project.metadata)),
        ("area", authored_area_script_hooks(project.metadata)),
    ):
        for field_name, script_resref in hooks.items():
            resref = normalise_resource_resref(script_resref)
            if not resref:
                continue
            is_packaged = (resref, "ncs") in packaged
            rows.append(
                {
                    "kind": "script",
                    "scope": scope,
                    "field_name": str(field_name),
                    "script_resref": resref,
                    "restype": "ncs",
                    "status": "packaged" if is_packaged else "external_or_override",
                    "packaged": is_packaged,
                    "required": True,
                    "message": (
                        f"{scope} script hook {field_name} uses packaged script {resref}.ncs."
                        if is_packaged
                        else f"{scope} script hook {field_name} must resolve from the base game, Override, or another installed mod."
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["scope"], row["field_name"], row["script_resref"]))


def _dialog_reference_rows(
    project: AuthoredModuleProject,
    *,
    packaged_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    packaged = packaged_keys or set()
    metadata = dict(getattr(getattr(project, "metadata", None), "metadata", {}) or {})
    sources: list[tuple[str, Any]] = []
    for key in ("dialog_refs", "dialog_references", "dialogue_refs", "conversation_refs", "conversations", "dialogs"):
        value = metadata.get(key)
        if value:
            sources.append((key, value))

    rows: list[dict[str, Any]] = []

    def append_row(source_key: str, label: str, value: Any) -> None:
        resref = normalise_resource_resref(value)
        if not resref:
            return
        is_packaged = (resref, "dlg") in packaged
        rows.append(
            {
                "kind": "dialog",
                "source": source_key,
                "field_name": str(label or source_key),
                "dialog_resref": resref,
                "restype": "dlg",
                "status": "packaged" if is_packaged else "external_or_override",
                "packaged": is_packaged,
                "required": True,
                "message": (
                    f"Dialog reference {resref}.dlg is included in this module package."
                    if is_packaged
                    else f"Dialog reference {resref}.dlg must resolve from the base game, Override, or another installed mod."
                ),
            }
        )

    for source_key, value in sources:
        if isinstance(value, dict):
            for label, resref in value.items():
                append_row(source_key, str(label), resref)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    append_row(source_key, str(item.get("field") or item.get("label") or item.get("name") or index), item.get("resref") or item.get("dialog") or item.get("dlg"))
                else:
                    append_row(source_key, str(index), item)
        else:
            append_row(source_key, source_key, value)
    return sorted(rows, key=lambda row: (row["source"], row["field_name"], row["dialog_resref"]))


def _resource_reference_gate_to_manifest(
    *,
    template_dependencies: list[dict[str, Any]],
    script_references: list[dict[str, Any]],
    dialog_references: list[dict[str, Any]],
) -> dict[str, Any]:
    references = list(template_dependencies) + list(script_references) + list(dialog_references)
    packaged_count = sum(1 for row in references if bool(row.get("packaged")))
    external_count = len(references) - packaged_count
    warnings = [
        str(row.get("message") or "")
        for row in references
        if row.get("required") and not row.get("packaged") and str(row.get("message") or "")
    ]
    return {
        "ready": True,
        "reference_count": len(references),
        "packaged_reference_count": packaged_count,
        "external_reference_count": external_count,
        "template_reference_count": len(template_dependencies),
        "script_reference_count": len(script_references),
        "dialog_reference_count": len(dialog_references),
        "all_required_packaged": external_count == 0,
        "requires_install_context": external_count > 0,
        "templates": list(template_dependencies),
        "scripts": list(script_references),
        "dialogs": list(dialog_references),
        "warnings": warnings,
        "blocking_messages": [],
    }


def _refresh_final_resource_reference_metadata(build: AuthoredModuleBuild) -> None:
    """Re-evaluate gameplay references after authored extras are merged.

    Placeable Builder UTPs and their dependent resources enter through
    ``extra_resources``.  Computing this gate only before that merge made a
    genuinely bundled project template look external in readiness and export
    manifests even though the final MOD contained it.
    """

    packaged_keys = set(build.resources)
    template_dependencies = _template_dependency_rows(build.project.placements, packaged_keys=packaged_keys)
    script_references = _script_reference_rows(build.project, packaged_keys=packaged_keys)
    dialog_references = _dialog_reference_rows(build.project, packaged_keys=packaged_keys)
    gate = _resource_reference_gate_to_manifest(
        template_dependencies=template_dependencies,
        script_references=script_references,
        dialog_references=dialog_references,
    )
    build.metadata.update(
        {
            "gameplay_template_dependencies": template_dependencies,
            "gameplay_template_dependency_count": len(template_dependencies),
            "gameplay_packaged_template_dependency_count": sum(
                1 for row in template_dependencies if bool(row.get("packaged"))
            ),
            "gameplay_external_template_dependency_count": sum(
                1 for row in template_dependencies if not bool(row.get("packaged"))
            ),
            "script_references": script_references,
            "script_reference_count": len(script_references),
            "dialog_references": dialog_references,
            "dialog_reference_count": len(dialog_references),
            "resource_reference_gate": gate,
        }
    )


def _smoke_expectations_from_build_parts(
    project: AuthoredModuleProject,
    *,
    walkability: dict[str, Any],
    pathing: dict[str, Any],
) -> dict[str, Any]:
    placements = project.placements
    expected_entry = {
        "label": "entry_point",
        "area_resref": placements.entry_point.area_resref,
        "position": _vec3_to_manifest(placements.entry_point.position),
        "facing": float(placements.entry_point.facing),
    }
    expected_placeables = _positioned_expectations("placeable", placements.placeables)
    expected_waypoints = _positioned_expectations("waypoint", placements.waypoints)
    project_metadata = dict(getattr(getattr(project, "metadata", None), "metadata", {}) or {})
    source_module = normalise_resource_resref(project_metadata.get("source_module_resref") or "")
    copied_from_base_game = bool(project_metadata.get("copied_from_base_game_module", False))
    inherited_content = bool(project_metadata.get("inherited_base_game_module_content", copied_from_base_game))
    content_origin = str(
        project_metadata.get("content_origin")
        or ("base_game_derived" if copied_from_base_game else "map_studio_original")
    )
    authored_original = content_origin == "map_studio_original" and not copied_from_base_game and not inherited_content
    expected_absent = {
        "base_game_module_geometry": bool(authored_original),
        "inherited_scripted_moving_test_objects": bool(authored_original),
        "forbidden_source_module_resrefs": ["PLCaa", "tar_m02aa"],
        "forbidden_object_descriptions": [
            "PLCaa scripted moving spheres",
            "PLCaa scripted moving cones",
            "PLCaa scripted moving rectangles",
            "Taris fallback room geometry",
        ],
    }
    return {
        "content_origin": content_origin,
        "authored_from_scratch": bool(authored_original),
        "copied_from_base_game_module": copied_from_base_game,
        "source_module_resref": source_module,
        "inherited_base_game_module_content": inherited_content,
        "source_identity": {
            "content_origin": content_origin,
            "authored_from_scratch": bool(authored_original),
            "copied_from_base_game_module": copied_from_base_game,
            "source_module_resref": source_module,
            "inherited_base_game_module_content": inherited_content,
        },
        "expected_entry_point": expected_entry,
        "expected_placeables": expected_placeables,
        "expected_waypoints": expected_waypoints,
        "expected_runtime_observations": {
            "player_start_area": expected_entry["area_resref"],
            "player_start_position": expected_entry["position"],
            "test_placeable_tags": [row["tag"] for row in expected_placeables if row["tag"]],
            "waypoint_tags": [row["tag"] for row in expected_waypoints if row["tag"]],
            "module_identity_resref": project.module_root,
            "no_inherited_base_game_geometry_or_scripted_movers": bool(authored_original),
        },
        "expected_absent_runtime_observations": expected_absent,
        "walkability": walkability,
        "all_walkability_checks_passed": bool(walkability.get("ok", False)),
        "pathing": pathing,
        "pathing_anchor_labels": list(pathing.get("anchor_labels", []) or []),
    }


def _placement_counts(placements: AuthoredGameplayPlacement) -> dict[str, int]:
    return {
        "creatures": len(tuple(placements.creatures or ())),
        "doors": len(tuple(placements.doors or ())),
        "triggers": len(tuple(placements.triggers or ())),
        "encounters": len(tuple(placements.encounters or ())),
        "sounds": len(tuple(placements.sounds or ())),
        "cameras": len(tuple(placements.cameras or ())),
        "stores": len(tuple(placements.stores or ())),
        "placeables": len(tuple(placements.placeables or ())),
        "waypoints": len(tuple(placements.waypoints or ())),
    }


def build_authored_module(project: AuthoredModuleProject, *, game_root_dir: str = "") -> AuthoredModuleBuild:
    """Compile authored module intent into in-memory runtime resources."""

    from .unreal_sector_projection import project_unreal_sector
    project = project_unreal_sector(project)
    root = project.module_root
    room_geometries: dict[str, AuthoredRoomGeometry] = {}
    room_woks: dict[str, WOKData] = {}
    backdrop_rooms: set[str] = set()
    # Stock-converted rooms whose render geometry is unedited keep their
    # original imported MDL/MDX (baked lights, animations, emitters) instead of
    # a flattened static rebuild; the edited .wok still ships alongside.
    preserved_model_rooms: dict[str, tuple[bytes, bytes]] = {}
    walkmesh_audits: list[AuthoredWalkmeshAudit] = []
    warnings: list[str] = []
    blocking: list[str] = []

    validation = validate_authored_module_project(project)
    warnings.extend(validation.warnings)
    blocking.extend(validation.blocking_issues)
    sky_traffic = read_authored_project_sky_traffic(project)
    sky_traffic_validation = validate_authored_sky_traffic_collection(
        sky_traffic,
        room_resrefs={room.normalised_resref() for room in project.rooms},
    )
    enabled_sky_traffic = tuple(track for track in sky_traffic if bool(getattr(track, "enabled", True)))
    warnings.extend(sky_traffic_validation.warnings)
    blocking.extend(sky_traffic_validation.blocking_issues)
    if enabled_sky_traffic:
        blocking.append(
            "Authored sky traffic is stored and previewable, but room-MDL animation export is not yet "
            "vanilla-structurally or in-game verified. Remove/disable the traffic tracks or complete the "
            "animloop controller compiler before packaging."
        )
    layout = None
    metadata = None
    pathing = None
    packaged: list[PackagedModuleResource] = []
    resources: dict[tuple[str, str], _HydratedResource] = {}
    import_source_resources, import_source_issues, import_source_path, import_source_layers = _import_source_capsule_resources(project)
    blocking.extend(import_source_issues)
    lighting_profile = authored_lighting_profile(project.metadata)
    fullbright_lighting = authored_lighting_is_fullbright(project.metadata)
    mdl_light_sanitizer_rows: list[dict[str, Any]] = []
    preserved_stock_pth: bytes | None = None
    preserved_stock_are: bytes | None = None
    preserved_stock_git: bytes | None = None
    preserved_stock_ifo: bytes | None = None
    try:
        preserved_stock_pth = _preserved_stock_resource(project, "pth")
    except Exception as exc:
        blocking.append(f"Imported stock PTH could not be preserved safely: {exc}")
    try:
        preserved_stock_are = _preserved_stock_resource(project, "are")
    except Exception as exc:
        blocking.append(f"Imported stock ARE could not be preserved safely: {exc}")
    try:
        preserved_stock_git = _preserved_stock_resource(project, "git")
    except Exception as exc:
        blocking.append(f"Imported stock GIT could not be preserved safely: {exc}")
    try:
        preserved_stock_ifo = _preserved_stock_resource(project, "ifo")
    except Exception as exc:
        blocking.append(f"Imported stock IFO could not be preserved safely: {exc}")

    try:
        layout = compile_authored_module_layout(project)
        warnings.extend(layout.warnings)
    except Exception as exc:
        blocking.append(f"Authored module layout could not be compiled: {exc}")

    for room in project.rooms:
        room_resref = room.normalised_resref()
        room_metadata = dict(getattr(room, "metadata", {}) or {})
        if authored_room_uses_unresolved_stock_geometry(room):
            issue = str(room_metadata.get("stock_geometry_issue") or "stock room model is unavailable").strip()
            blocking.append(
                f"Room {room_resref or '(unnamed)'} has no resolved stock geometry ({issue}). "
                "PIE may omit this source placeholder, but module export is blocked until the missing MDL/MDX/WOK "
                "is supplied or the unresolved room reference is removed deliberately."
            )
            continue
        if isinstance(getattr(room, "primitive", None), ImportedMeshRoomPrimitive):
            backdrop_only = imported_mesh_room_is_backdrop(room.primitive)
            visual_only = imported_mesh_room_is_visual_only(room.primitive)
        else:
            backdrop_only = bool(room_metadata.get("backdrop_only", room_metadata.get("is_backdrop", False)))
            visual_only = bool(room_metadata.get("visual_only", False))
        visual_only = bool(visual_only or backdrop_only)
        if visual_only:
            backdrop_rooms.add(room_resref)
        primitive_metadata = dict(getattr(getattr(room, "primitive", None), "metadata", {}) or {})
        runtime_graph = dict(primitive_metadata.get("source_runtime_graph") or {})
        lost_runtime = {
            label: int(runtime_graph.get(label, 0) or 0)
            for label in ("animation_count", "light_count", "emitter_count", "reference_count")
            if int(runtime_graph.get(label, 0) or 0) > 0
        }
        # A pristine stock-converted room keeps its original imported model, so
        # its runtime graph (lights/anims/emitters) is retained byte-for-byte
        # rather than flattened. Resolve that before the gate.
        preserved_room_model: tuple[bytes, bytes] | None = None
        if _room_is_eligible_for_stock_model_preservation(room):
            preserved_room_model = _preserved_stock_room_model(project, room_resref)
        if preserved_room_model is not None:
            preserved_model_rooms[room_resref] = preserved_room_model
        if lost_runtime and preserved_room_model is not None:
            summary = ", ".join(f"{key.replace('_count', '')}={value}" for key, value in lost_runtime.items())
            warnings.append(
                f"Room {room_resref} reuses its original imported model to retain its baked runtime graph "
                f"({summary}); only its walkmesh is regenerated. Confirm the room's lighting in the in-game warp test."
            )
        elif lost_runtime and not bool(runtime_graph.get("preserved", False)):
            summary = ", ".join(f"{key.replace('_count', '')}={value}" for key, value in lost_runtime.items())
            if (
                isinstance(getattr(room, "primitive", None), ImportedMeshRoomPrimitive)
                and imported_mesh_has_explicit_static_runtime_rebuild(room.primitive)
            ):
                warnings.append(
                    f"Room {room_resref} explicitly replaces its stock runtime graph ({summary}) with a new static "
                    "room MDL. Source animations/model lights/emitters/references will not be retained."
                )
            else:
                blocking.append(
                    f"Room {room_resref} was flattened from a stock runtime graph ({summary}). "
                    "Export is blocked until its original animations/lights/emitters/references are retained or "
                    "an explicit static-rebuild policy acknowledges their removal."
                )
        try:
            geometry = compile_authored_room_spec(room)
            # A preserved/imported stock WOK already carries correct materials,
            # including legitimate NON_WALK wall/blocker faces and its own
            # perimeter, so it passes through untouched.  Only WOKs regenerated
            # from render geometry (from-scratch rooms) need boundary walls
            # generated and the enclosing ceiling/wall slab reduced to the
            # walkable surface (the slab otherwise kills the perimeter loop and
            # freezes the player -- see _walkmesh_walkable_only).
            imported_wok = bool(dict(getattr(geometry, "metadata", {}) or {}).get("imported_wok"))
            if not imported_wok:
                geometry = apply_authored_walkmesh_boundary_policy_to_geometry(geometry)
                geometry = replace(geometry, wok=_walkmesh_walkable_only(geometry.wok))
            else:
                # Converted rooms can ship a room-local WOK mislabeled as
                # module-space (921srt's 921srtb); the engine reads .wok files
                # in module coordinates, so rebase it the same way the PIE
                # combiner does or the exported floor detaches from the room.
                alignment_offset, alignment_warning = resolve_room_wok_module_offset(room, geometry.wok)
                if alignment_warning:
                    warnings.append(alignment_warning)
                if alignment_offset != (0.0, 0.0, 0.0):
                    geometry = replace(geometry, wok=offset_wok_data(geometry.wok, alignment_offset))
        except Exception as exc:
            blocking.append(f"Room {room_resref or '(unnamed)'} geometry could not be compiled: {exc}")
            continue
        room_geometries[room_resref] = geometry
        room_woks[room_resref] = geometry.wok
        audit = audit_authored_wok(room_resref, geometry.wok)
        if visual_only and audit.face_count == 0:
            audit = replace(
                audit,
                ready=True,
                warnings=tuple(audit.warnings) + (
                    f"Room {room_resref} is a visual-only backdrop preserving a vanilla-style empty WOK.",
                ),
                blocking_messages=(),
            )
        walkmesh_audits.append(audit)
        warnings.extend(audit.warnings)
        blocking.extend(audit.blocking_messages)

    # Compile stable room/hook connection records into the current LYT room
    # order only after every room WOK is available. Imported kit WOKs have
    # already been rebased to module space above; authored floor-plan WOKs are
    # still room-local here. This pass replaces stale source-module transition
    # indices with reciprocal portals for this exported module and rejects a
    # visually snapped room pair whose floor boundaries do not really meet.
    imported_module_space = {
        room_resref
        for room_resref, geometry in room_geometries.items()
        if bool(dict(getattr(geometry, "metadata", {}) or {}).get("imported_wok"))
    }
    connection_build = compile_authored_room_connection_walkmeshes(
        project,
        room_woks,
        module_space_room_resrefs=imported_module_space,
    )
    warnings.extend(connection_build.warnings)
    blocking.extend(connection_build.blocking_issues)
    room_woks = dict(connection_build.room_woks)
    room_geometries = {
        room_resref: replace(geometry, wok=room_woks.get(room_resref, geometry.wok))
        for room_resref, geometry in room_geometries.items()
    }

    entry_room, entry_wok = _entry_room_wok(project, room_geometries)
    # Placements live anywhere in the module, so walkability validates
    # against every room's WOK merged — checking only the entry room flagged
    # every multi-room module (the spawn may be in room 3, a waypoint in 7).
    # Backdrop/skybox rooms are excluded: their thousand-unit WOK would blow up
    # the merged mesh bounds and false-flag every real placement as off-mesh.
    playable_woks = {resref: wok for resref, wok in room_woks.items() if resref not in backdrop_rooms}
    validation_wok = _merged_validation_wok(playable_woks) or entry_wok
    walkability = None
    if validation_wok is not None:
        # 2.0m Z tolerance: the engine snaps the player/creatures to the floor
        # on load, and an imported module's placements came from the original
        # walkmesh which we regenerate at a slightly different Z. The tight 0.05m
        # default false-flagged every imported spawn as "not on the floor".
        walkability = validate_authored_gameplay_placement_against_walkmesh(
            project.placements, validation_wok, z_tolerance=2.0)
        warnings.extend(walkability.warnings)
        blocking.extend(walkability.blocking_issues)
    elif project.rooms:
        blocking.append("Authored module export could not find a room WOK for gameplay walkability checks.")

    try:
        project_are_rooms = tuple(
            room.normalised_resref()
            for room in project.rooms
            if bool(dict(getattr(room, "metadata", {}) or {}).get("are_listed", True))
        )
        source_are_order = tuple(
            normalise_resref(value)
            for value in tuple(dict(getattr(project, "extra", {}) or {}).get("source_are_room_resrefs", ()) or ())
            if normalise_resref(value)
        )
        project_are_set = set(project_are_rooms)
        are_room_resrefs = tuple(room for room in source_are_order if room in project_are_set) + tuple(
            room for room in project_are_rooms if room not in source_are_order
        )
        compiled_area = AuthoredAreaMetadata(
            name=project.metadata.display_name,
            tag=project.metadata.tag or root,
            comments="Generated by GhostRigger Map Studio authored module export.",
        )
        metadata = compile_authored_module_metadata(
            project.metadata,
            project.placements.entry_point,
            area=compiled_area,
            room_resrefs=are_room_resrefs or tuple(getattr(layout, "room_resrefs", ()) or tuple(room_geometries)),
            area_resrefs=(root,),
        )
        if preserved_stock_are is not None:
            lighting_source = str(
                dict(dict(getattr(project.metadata, "metadata", {}) or {}).get("lighting") or {}).get("source") or ""
            ).strip().lower()
            patched_are = patch_preserved_stock_are_bytes(
                preserved_stock_are,
                project.metadata,
                compiled_area,
                room_resrefs=are_room_resrefs,
                update_world_lighting=lighting_source == "map_studio:world_settings",
            )
            metadata = replace(
                metadata,
                are_bytes=patched_are,
                metadata={
                    **dict(metadata.metadata),
                    "preserved_stock_are": True,
                    "world_lighting_patched": lighting_source == "map_studio:world_settings",
                },
            )
        if preserved_stock_ifo is not None:
            patched_ifo = patch_preserved_stock_ifo_bytes(
                preserved_stock_ifo,
                project.placements.entry_point,
                area_resrefs=(root,),
            )
            metadata = replace(
                metadata,
                ifo_bytes=patched_ifo,
                metadata={
                    **dict(metadata.metadata),
                    "preserved_stock_ifo": True,
                    "stock_ifo_entry_patched": patched_ifo != preserved_stock_ifo,
                },
            )
        warnings.extend(metadata.validation.warnings)
        blocking.extend(metadata.validation.blocking_issues)
    except Exception as exc:
        blocking.append(f"Authored module metadata could not be compiled: {exc}")

    if layout is not None and room_geometries and preserved_stock_pth is None:
        try:
            # Transition targets are indices into the exact LYT room order.
            # Keep every row (including empty visual/backdrop rooms), preserve
            # per-room indexed-face adjacency, and bridge only reciprocal WOK
            # transition edges whose module-space boundaries actually match.
            # Skipped when we emit a preserved stock PTH (below): the original
            # graph spans the full module -- including reference/skybox rooms
            # not editable here -- so a regenerated graph over the editable
            # subset would false-positive "point outside the walkmesh".
            pathing_rooms = _lyt_ordered_pathing_rooms(layout, room_geometries)
            pathing = compile_authored_pathing_for_rooms(
                pathing_rooms,
                anchors=_path_anchors_from_walkability(project.placements, walkability),
            )
            warnings.extend(pathing.validation.warnings)
            blocking.extend(pathing.validation.blocking_issues)
        except Exception as exc:
            blocking.append(f"Authored module pathing could not be compiled: {exc}")

    try:
        material_preflight = compile_authored_room_material_preflight(
            project.rooms[0].primitive.material.texture if hasattr(project.rooms[0].primitive, "material") else getattr(project.rooms[0].primitive, "texture", ""),
            game_root_dir=game_root_dir,
            require_game_resolution=False,
        )
        warnings.extend(material_preflight.warnings)
        blocking.extend(material_preflight.blocking_issues)
    except Exception as exc:
        warnings.append(f"Authored room material preflight could not run: {exc}")

    if metadata is not None:
        packaged.append(_make_packaged(root, "are", metadata.are_bytes, "map_studio:authored:are"))
        packaged.append(_make_packaged(ENGINE_MODULE_IFO_RESREF, "ifo", metadata.ifo_bytes, "map_studio:authored:ifo"))
    try:
        placement_edit_keys = (
            "last_gameplay_placement",
            "last_gameplay_placement_transform",
            "last_gameplay_placement_rename",
            "last_gameplay_placement_duplicate",
            "last_gameplay_placement_remove",
            "last_creature_behavior_update",
        )
        placement_was_edited = any(dict(getattr(project, "extra", {}) or {}).get(key) for key in placement_edit_keys)
        if preserved_stock_git is not None and not placement_was_edited:
            git_bytes = preserved_stock_git
            git_source = "map_studio:stock:git_preserved"
        elif preserved_stock_git is not None:
            git_bytes = patch_preserved_stock_git_bytes(
                preserved_stock_git,
                project.placements,
                game=project.game,
            )
            git_source = "map_studio:stock:git_lists_patched"
            warnings.append(
                "Imported GIT placement lists were patched while stock root fields (including AreaProperties) "
                "were preserved. Per-instance fields outside Map Studio's editable placement contract still "
                "require an in-game behavior proof."
            )
        else:
            git_bytes = build_git_bytes(project.placements, game=project.game)
            git_source = "map_studio:authored:git"
        packaged.append(_make_packaged(root, "git", git_bytes, git_source))
    except Exception as exc:
        blocking.append(f"Authored gameplay placements could not be compiled into GIT: {exc}")
    if preserved_stock_pth is not None:
        packaged.append(_make_packaged(root, "pth", preserved_stock_pth, "map_studio:stock:pth_preserved"))
    elif pathing is not None:
        packaged.append(_make_packaged(root, "pth", pathing.pth_bytes, "map_studio:authored:pth"))
    for room_resref, geometry in room_geometries.items():
        preserved_room_model = preserved_model_rooms.get(room_resref)
        if preserved_room_model is not None:
            # Ship the original imported room MDL/MDX verbatim (baked lights,
            # animations, emitters retained); the edited .wok is packaged
            # separately below.
            mdl_bytes, mdx_bytes = preserved_room_model
            packaged.append(_make_packaged(room_resref, "mdl", mdl_bytes, "map_studio:stock:room_model_preserved"))
            packaged.append(_make_packaged(room_resref, "mdx", mdx_bytes, "map_studio:stock:room_model_preserved"))
            continue
        try:
            mdl_bytes, mdx_bytes = _make_room_model_bytes(
                project.game,
                geometry,
                lighting_profile=lighting_profile,
            )
            if fullbright_lighting:
                mdl_bytes, sanitizer_row = neutralize_fullbright_room_mdl_lights(
                    mdl_bytes,
                    mdx_bytes,
                    game=project.game,
                )
                sanitizer_row["room_resref"] = room_resref
                mdl_light_sanitizer_rows.append(sanitizer_row)
        except Exception as exc:
            blocking.append(f"Room {room_resref}.mdl/.mdx could not be compiled: {exc}")
            continue
        packaged.append(_make_packaged(room_resref, "mdl", mdl_bytes, "map_studio:authored:room_model"))
        packaged.append(_make_packaged(room_resref, "mdx", mdx_bytes, "map_studio:authored:room_model"))

    if layout is not None:
        resources[(root, "lyt")] = _HydratedResource(_ResourceRecord(root, "lyt", "map_studio:authored:lyt"), layout.lyt.to_text().encode("latin-1"))
        resources[(root, "vis")] = _HydratedResource(_ResourceRecord(root, "vis", "map_studio:authored:vis"), layout.vis.to_text().encode("latin-1"))
    for room_resref, wok in room_woks.items():
        resources[(room_resref, "wok")] = _HydratedResource(_ResourceRecord(room_resref, "wok", "map_studio:authored:wok"), wok.to_bytes())
    for item in packaged:
        resources[item.key] = _HydratedResource(_ResourceRecord(item.key[0], item.key[1], item.source), bytes(item.data))

    # Source capsule is the base layer; authored/generated resources already
    # assembled above win by key. Ancillary source resources are copied
    # byte-for-byte and also staged as explicit package replacements.
    preserved_source_keys: list[tuple[str, str]] = []
    generated_replaced_source_keys: list[tuple[str, str]] = []
    for key, data in sorted(import_source_resources.items()):
        if key in resources:
            generated_replaced_source_keys.append(key)
            continue
        source_record = _make_packaged(key[0], key[1], data, "map_studio:stock:import_source")
        packaged.append(source_record)
        resources[key] = _HydratedResource(
            _ResourceRecord(key[0], key[1], source_record.source),
            data,
        )
        preserved_source_keys.append(key)
    import_source_overlay = {
        "schema": "ghostrigger.map_studio_import_source_overlay.v1",
        "source_path": import_source_path,
        "source_layers": import_source_layers,
        "source_layer_count": len(import_source_layers),
        "source_layer_resource_count": sum(int(row.get("resource_count", 0) or 0) for row in import_source_layers),
        "source_resource_count": len(import_source_resources),
        "source_resource_bytes": sum(len(data) for data in import_source_resources.values()),
        "preserved_resource_count": len(preserved_source_keys),
        "preserved_resource_keys": [f"{resref}.{restype}" for resref, restype in preserved_source_keys],
        "generated_replaced_resource_count": len(generated_replaced_source_keys),
        "generated_replaced_resource_keys": [
            f"{resref}.{restype}" for resref, restype in generated_replaced_source_keys
        ],
        "authored_extra_replaced_resource_count": 0,
        "authored_extra_replaced_resource_keys": [],
        "replaced_resource_count": len(generated_replaced_source_keys),
        "blocking_issues": list(import_source_issues),
        "ready": not import_source_issues,
    }

    engine_contract_metadata: dict[str, Any] = {
        "schema": "ghostrigger.map_studio_engine_contract.v1",
        "export_ready": False,
        "blocking_issues": ["Serialized engine-contract validation did not run."],
    }
    try:
        engine_contract = validate_kotor_module_engine_contract(
            KotorModuleEngineContractRequest(
                game=project.game,
                module_resref=root,
                resources={key: value.data for key, value in resources.items()},
                expected_room_resrefs=tuple(sorted(room_geometries)),
                visual_only_room_resrefs=tuple(sorted(backdrop_rooms)),
            )
        )
        engine_contract_metadata = engine_contract.to_dict()
        warnings.extend(engine_contract.warnings)
        blocking.extend(engine_contract.blocking_issues)
    except Exception as exc:
        message = f"Serialized engine-contract validation could not run: {exc}"
        blocking.append(message)
        engine_contract_metadata["blocking_issues"] = [message]

    module_state = _ModuleState(
        name=root,
        lyt=layout.lyt if layout is not None else LYTLayout(),
        vis=layout.vis if layout is not None else VISData(),
        room_woks=room_woks,
        room_geometry=room_geometries,
        placements=project.placements,
        visual_only_room_resrefs=tuple(sorted(backdrop_rooms)),
        walkmesh_connection_pairs=tuple(
            (portal.source_room_resref, portal.target_room_resref)
            for portal in connection_build.portals
        ),
    )
    walkability_metadata = _walkability_to_manifest(walkability)
    pathing_metadata = dict(pathing.metadata) if pathing is not None else {}
    if preserved_stock_pth is not None:
        pathing_metadata.update(
            {
                "source": "map_studio:stock:pth_preserved",
                "preserved_stock_pth": True,
                "preserved_stock_pth_size": len(preserved_stock_pth),
                "preserved_stock_pth_sha256": hashlib.sha256(preserved_stock_pth).hexdigest(),
            }
        )
    transition_surface_gate = _transition_surface_gate_to_manifest(project.placements, tuple(walkmesh_audits))
    warnings.extend(str(warning) for warning in transition_surface_gate.get("warnings", []) or [])
    blocking.extend(str(message) for message in transition_surface_gate.get("blocking_messages", []) or [])
    walkmesh_gate = _walkmesh_gate_to_manifest(
        tuple(walkmesh_audits),
        walkability=walkability_metadata,
        pathing=pathing_metadata,
        transition_surface_gate=transition_surface_gate,
    )
    visibility_metadata = _visibility_to_manifest(project, layout)
    smoke_expectations = _smoke_expectations_from_build_parts(
        project,
        walkability=walkability_metadata,
        pathing=pathing_metadata,
    )
    room_lights = [authored_room_light_payload(light) for light in tuple(getattr(project, "lights", ()) or ())]
    lighting_metadata = _lighting_to_manifest(project, room_lights)
    sky_traffic_metadata = {
        "count": len(sky_traffic),
        "enabled_count": len(enabled_sky_traffic),
        "compiler_target": "room_mdl_animation",
        "export_ready": not enabled_sky_traffic,
        "tracks": authored_sky_traffic_list_to_kmap(sky_traffic),
        "warnings": list(sky_traffic_validation.warnings),
        "blocking_issues": list(sky_traffic_validation.blocking_issues),
    }
    material_uv = _room_material_uv_manifest(
        module_state.room_geometry,
        lighting_profile=lighting_profile,
    )
    packaged_keys = set(resources)
    template_dependencies = _template_dependency_rows(project.placements, packaged_keys=packaged_keys)
    script_references = _script_reference_rows(project, packaged_keys=packaged_keys)
    dialog_references = _dialog_reference_rows(project, packaged_keys=packaged_keys)
    resource_reference_gate = _resource_reference_gate_to_manifest(
        template_dependencies=template_dependencies,
        script_references=script_references,
        dialog_references=dialog_references,
    )
    packaged_template_count = sum(1 for row in template_dependencies if row["packaged"])
    external_template_count = len(template_dependencies) - packaged_template_count
    return AuthoredModuleBuild(
        module_root=root,
        game=project.game,
        module=module_state,
        project=project,
        resources=resources,
        packaged_resources=packaged,
        resource_summaries=_resource_summary(resources),
        warnings=warnings,
        blocking_issues=blocking,
        metadata={
            "source": "src.core.modules.authored_module_export",
            "entry_room": entry_room,
            "room_count": len(room_geometries),
            "backdrop_room_resrefs": tuple(sorted(backdrop_rooms)),
            "visual_only_room_resrefs": tuple(sorted(backdrop_rooms)),
            "resource_count": len(resources),
            "import_source_overlay": import_source_overlay,
            "gameplay_counts": _placement_counts(project.placements),
            "gameplay_template_dependencies": template_dependencies,
            "gameplay_template_dependency_count": len(template_dependencies),
            "gameplay_packaged_template_dependency_count": packaged_template_count,
            "gameplay_external_template_dependency_count": external_template_count,
            "script_references": script_references,
            "script_reference_count": len(script_references),
            "dialog_references": dialog_references,
            "dialog_reference_count": len(dialog_references),
            "resource_reference_gate": resource_reference_gate,
            "lighting_count": len(room_lights),
            "room_lights": room_lights,
            "lighting": lighting_metadata,
            "sky_traffic": sky_traffic_metadata,
            "mdl_light_sanitizer": _mdl_light_sanitizer_to_manifest(fullbright_lighting, mdl_light_sanitizer_rows),
            "material_uv": material_uv,
            "visibility": visibility_metadata,
            "walkability": walkability_metadata,
            "pathing": pathing_metadata,
            "stock_metadata_preservation": {
                "are": preserved_stock_are is not None,
                "git": preserved_stock_git is not None,
                "ifo": preserved_stock_ifo is not None,
            },
            "walkmesh_gate": walkmesh_gate,
            "engine_contract": engine_contract_metadata,
            "smoke_expectations": smoke_expectations,
        },
    )


def _augment_authored_manifest(
    path: str,
    build: AuthoredModuleBuild,
    package_result: CustomModulePackResult,
    verification: DevModulePackageVerification | None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return _authored_export_job_record(build, package_result, verification)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    remaining_acceptance = [
        f"Install/copy {build.module_root}.mod into the KOTOR Modules folder.",
        f"Launch KOTOR and run 'warp {build.module_root}'.",
        "Confirm the module loads in-game without crashing or falling back to another area.",
        "Confirm the player appears on the generated floor, not in the void.",
        "Confirm authored test placeables and waypoints resolve as expected.",
        "Walk across the generated floor and confirm WOK/pathing behavior.",
        "Capture screenshot or video evidence and record it in the proof manifest.",
    ]
    smoke_contract = _authored_smoke_contract(build, verification)
    source_identity = dict(smoke_contract.get("source_identity") or {})
    project_metadata = dict(getattr(getattr(build.project, "metadata", None), "metadata", {}) or {})
    export_job = _authored_export_job_record(build, package_result, verification)
    data["map_studio_authored_module"] = {
        "task": "T2643",
        "module_root": build.module_root,
        "game": build.game,
        "project_metadata": project_metadata,
        "content_origin": str(smoke_contract.get("content_origin") or source_identity.get("content_origin") or "map_studio_original"),
        "authored_from_scratch": bool(smoke_contract.get("authored_from_scratch", source_identity.get("authored_from_scratch", True))),
        "copied_from_base_game_module": bool(smoke_contract.get("copied_from_base_game_module", source_identity.get("copied_from_base_game_module", False))),
        "source_module_resref": str(smoke_contract.get("source_module_resref") or source_identity.get("source_module_resref") or ""),
        "inherited_base_game_module_content": bool(
            smoke_contract.get("inherited_base_game_module_content", source_identity.get("inherited_base_game_module_content", False))
        ),
        "source_identity": source_identity,
        "expected_absent_runtime_observations": dict(smoke_contract.get("expected_absent_runtime_observations") or {}),
        "capability_stage": "export_candidate",
        "game_tested": False,
        "warp_command": f"warp {build.module_root}",
        "rooms": [
            {
                "resref": resref,
                "wok_faces": len(getattr(geometry.wok, "faces", []) or []),
                "wok_walkable_faces": int(geometry.wok.walkable_face_count()) if hasattr(geometry.wok, "walkable_face_count") else 0,
                "wok_non_walk_faces": int(geometry.wok.non_walk_face_count()) if hasattr(geometry.wok, "non_walk_face_count") else 0,
                "wok_transition_surface_faces": sum(
                    1
                    for face in list(getattr(geometry.wok, "faces", []) or [])
                    if int(getattr(face, "surface", -1)) == DOOR_TRANSITION_SURFACE_ID
                ),
                "model_nodes": 1 + len(geometry.helper_meshes),
                "texture": str(getattr(geometry.room_mesh, "texture", "") or ""),
                "floor_surface_id": int(getattr((list(getattr(geometry.wok, "faces", []) or []) or [None])[0], "surface", -1)),
                "walkmesh_boundary_wall_faces": int((geometry.metadata or {}).get("walkmesh_boundary_wall_faces") or 0),
            }
            for resref, geometry in sorted(build.module.room_geometry.items())
        ],
        "gameplay_counts": _placement_counts(build.project.placements),
        "gameplay_template_dependencies": list(build.metadata.get("gameplay_template_dependencies", []) or []),
        "gameplay_template_dependency_count": int(build.metadata.get("gameplay_template_dependency_count", 0) or 0),
        "gameplay_packaged_template_dependency_count": int(build.metadata.get("gameplay_packaged_template_dependency_count", 0) or 0),
        "gameplay_external_template_dependency_count": int(build.metadata.get("gameplay_external_template_dependency_count", 0) or 0),
        "script_references": list(build.metadata.get("script_references", []) or []),
        "script_reference_count": int(build.metadata.get("script_reference_count", 0) or 0),
        "dialog_references": list(build.metadata.get("dialog_references", []) or []),
        "dialog_reference_count": int(build.metadata.get("dialog_reference_count", 0) or 0),
        "resource_reference_gate": dict(build.metadata.get("resource_reference_gate", {}) or {}),
        "lighting_count": int(build.metadata.get("lighting_count", 0) or 0),
        "room_lights": list(build.metadata.get("room_lights", []) or []),
        "lighting": dict(build.metadata.get("lighting", {}) or {}),
        "sky_traffic": dict(build.metadata.get("sky_traffic", {}) or {}),
        "mdl_light_sanitizer": dict(build.metadata.get("mdl_light_sanitizer", {}) or {}),
        "material_uv": list(build.metadata.get("material_uv", []) or []),
        "visibility": dict(build.metadata.get("visibility", {}) or {}),
        "walkability": dict(build.metadata.get("walkability", {})),
        "pathing": dict(build.metadata.get("pathing", {})),
        "walkmesh_gate": dict(build.metadata.get("walkmesh_gate", {}) or {}),
        "engine_contract": dict(build.metadata.get("engine_contract", {}) or {}),
        "smoke_expectations": dict(build.metadata.get("smoke_expectations", {})),
        "resources": [summary.__dict__ for summary in build.resource_summaries],
        "secondary_motion_profiles": list(build.metadata.get("secondary_motion_profiles", []) or []),
        "package_ok": bool(package_result.ok),
        "package_verification": _verification_to_manifest(verification),
        "export_job": export_job,
        "readback": {
            "ok": bool(verification.ok) if verification is not None else False,
            "code": verification.code if verification is not None else "not_verified",
            "message": verification.message if verification is not None else "",
            "resources": [
                {
                    "resref": resource.resref,
                    "restype": resource.restype,
                    "size": resource.size,
                    "offset": resource.offset,
                }
                for resource in (verification.resources if verification is not None else [])
            ],
            "blocking_issues": list(verification.blocking_issues) if verification is not None else [],
        },
        "t2601_smoke_contract": smoke_contract,
        "modder_test_plan": _authored_modder_test_plan(
            smoke_contract=smoke_contract,
            module_path=package_result.module_path,
        ),
        "required_runtime_resources": smoke_contract["required_resources"],
        "manual_game_test_required": [
            f"Copy/install {build.module_root}.mod into the KOTOR Modules folder.",
            f"Start KOTOR and run 'warp {build.module_root}'.",
            "Confirm the room loads, player start works, and authored gameplay templates resolve.",
        ],
        "remaining_acceptance": remaining_acceptance,
    }
    data["export_job"] = export_job
    data["validation"]["warnings"] = sorted(set(list(data.get("validation", {}).get("warnings", [])) + build.warnings + package_result.warnings))
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return export_job


def export_authored_module_project(request: AuthoredModuleExportRequest) -> AuthoredModuleExportResult:
    """Export the current authored Map Studio project through the MOD packager."""

    build = build_authored_module(request.project, game_root_dir=request.game_root_dir)
    secondary_motion_sidecars = _prepare_secondary_motion_sidecars(
        build,
        request.secondary_motion_profiles,
    )
    collision_gate = _merge_authored_extra_resources(build, tuple(request.extra_resources or ()))
    _refresh_final_resource_reference_metadata(build)
    _validate_final_authored_resource_map(build, collision_gate)
    room_resrefs = tuple(sorted(build.module.room_geometry))
    if collision_gate["collision_count"] or (build.blocking_issues and request.strict):
        metadata = dict(build.metadata)
        metadata["export_job"] = _authored_export_job_record(build)
        return AuthoredModuleExportResult(
            ok=False,
            module_root=build.module_root,
            room_resrefs=room_resrefs,
            resources=build.resource_summaries,
            warnings=build.warnings,
            blocking_issues=build.blocking_issues,
            metadata=metadata,
            message=f"Authored Map Studio module preflight found {len(build.blocking_issues)} blocking issue(s).",
            code="preflight_failed",
        )
    if request.dry_run:
        metadata = dict(build.metadata)
        metadata["export_job"] = _authored_export_job_record(build)
        return AuthoredModuleExportResult(
            ok=not build.blocking_issues,
            module_root=build.module_root,
            room_resrefs=room_resrefs,
            resources=build.resource_summaries,
            warnings=build.warnings + ["Dry run only; no MOD package was written."],
            blocking_issues=build.blocking_issues,
            metadata=metadata,
            message=(
                f"Authored Map Studio module dry run passed for {build.module_root}."
                if not build.blocking_issues
                else f"Authored Map Studio module dry run found {len(build.blocking_issues)} blocking issue(s)."
            ),
            code="dry_run_passed" if not build.blocking_issues else "dry_run_failed",
        )

    pack_request = CustomModulePackRequest(
        module_root=build.module_root,
        game=build.game,
        output_dir=request.output_dir,
        archive_mode="mod",
        create_backups=request.create_backups,
        write_loose_resources=request.write_loose_resources,
        include_reference_check=request.include_reference_check,
        include_wok_check=request.include_wok_check,
        strict=request.strict,
    )
    package_result = package_custom_module(
        build,
        pack_request,
        resources=build.packaged_resources,
        now=datetime.now(timezone.utc),
    )
    if package_result.ok and package_result.module_path and secondary_motion_sidecars:
        sidecar_root = Path(package_result.module_path).parent / "secondary-motion"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        written_sidecars: list[str] = []
        for profile_id, payload in secondary_motion_sidecars:
            digest = hashlib.sha256(payload).hexdigest()
            target = sidecar_root / f"{profile_id}.{digest[:12]}.json"
            if target.exists() and target.read_bytes() != payload:
                build.blocking_issues.append(f"Secondary-motion sidecar collision at {target.name}.")
                continue
            target.write_bytes(payload)
            if target.read_bytes() != payload:
                build.blocking_issues.append(f"Secondary-motion sidecar readback failed for {target.name}.")
                continue
            written_sidecars.append(str(target))
        build.metadata["secondary_motion_profile_paths"] = written_sidecars
    verification = None
    if package_result.ok and package_result.module_path and room_resrefs:
        verification = verify_dev_test_module_package(
            package_result.module_path,
            expected_module_root=build.module_root,
            expected_room_resref=room_resrefs[0],
            expected_room_resrefs=room_resrefs,
            visual_only_room_resrefs=tuple(build.metadata.get("visual_only_room_resrefs", ()) or ()),
            game=build.game,
            stock_ifo_preserved=bool(
                dict(dict(build.metadata or {}).get("stock_metadata_preservation") or {}).get("ifo", False)
            ),
        )
    export_job = _augment_authored_manifest(package_result.manifest_path, build, package_result, verification)
    verification_warnings = list(verification.warnings) if verification is not None else []
    verification_blocking = list(verification.blocking_issues) if verification is not None else []
    ok = package_result.ok and (verification.ok if verification is not None else True) and not build.blocking_issues
    metadata = dict(build.metadata)
    metadata["export_job"] = export_job
    return AuthoredModuleExportResult(
        ok=ok,
        module_root=build.module_root,
        room_resrefs=room_resrefs,
        module_path=package_result.module_path,
        manifest_path=package_result.manifest_path,
        package_result=package_result,
        package_verification=verification,
        resources=build.resource_summaries,
        warnings=build.warnings + package_result.warnings + verification_warnings,
        blocking_issues=build.blocking_issues + package_result.blocking_issues + verification_blocking,
        metadata=metadata,
        message=(
            f"Authored Map Studio module exported: {package_result.module_path}"
            if ok
            else package_result.message
        ),
        code="export_candidate" if ok else (verification.code if verification is not None and not verification.ok else package_result.code),
    )


def _next_install_backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = path.with_suffix(path.suffix + f".bak{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available backup path for {path}.")


def _verification_to_manifest(verification: DevModulePackageVerification | None) -> dict[str, Any]:
    if verification is None:
        return {"ok": False, "code": "not_verified"}
    return {
        "ok": verification.ok,
        "code": verification.code,
        "message": verification.message,
        "parsed_gff": list(verification.parsed_gff),
        "parsed_wok": list(verification.parsed_wok),
        "model_pairs": list(verification.model_pairs),
        "path_point_count": verification.path_point_count,
        "path_connection_count": verification.path_connection_count,
        "resources": [
            {
                "resref": resource.resref,
                "restype": resource.restype,
                "size": resource.size,
                "offset": resource.offset,
            }
            for resource in verification.resources
        ],
        "warnings": list(verification.warnings),
        "blocking_issues": list(verification.blocking_issues),
    }


def _artifact_output_record(path: str, artifact_kind: str, *, required: bool = True) -> dict[str, Any]:
    text = str(path or "")
    item_path = Path(text) if text else None
    exists = item_path.exists() if item_path is not None else False
    return {
        "artifact_kind": artifact_kind,
        "final_path": text,
        "required": bool(required),
        "exists": bool(exists),
        "is_file": bool(item_path.is_file()) if item_path is not None and exists else False,
        "is_dir": bool(item_path.is_dir()) if item_path is not None and exists else False,
        "size": int(item_path.stat().st_size) if item_path is not None and item_path.is_file() else 0,
    }


def _package_output_records(package_result: CustomModulePackResult | None) -> list[dict[str, Any]]:
    if package_result is None:
        return []
    outputs = [
        _artifact_output_record(package_result.module_path, "module_package"),
        _artifact_output_record(package_result.manifest_path, "pack_manifest"),
        _artifact_output_record(package_result.resources_dir, "loose_resource_directory", required=False),
        _artifact_output_record(package_result.override_dir, "override_resource_directory", required=False),
    ]
    for resource in list(package_result.staged_resources or []):
        outputs.append(
            {
                "artifact_kind": "loose_resource",
                "resref": resource.resref,
                "restype": resource.restype,
                "final_path": resource.path,
                "required": True,
                "exists": Path(resource.path).is_file(),
                "size": resource.size,
                "sha256": resource.sha256,
                "source": resource.source,
            }
        )
    for resource in list(package_result.staged_override_resources or []):
        outputs.append(
            {
                "artifact_kind": "override_resource",
                "resref": resource.resref,
                "restype": resource.restype,
                "final_path": resource.path,
                "required": True,
                "exists": Path(resource.path).is_file(),
                "size": resource.size,
                "sha256": resource.sha256,
                "source": resource.source,
            }
        )
    return outputs


def _export_job_status(
    build: AuthoredModuleBuild,
    package_result: CustomModulePackResult | None,
    verification: DevModulePackageVerification | None,
) -> str:
    if build.blocking_issues:
        return "preflight_failed"
    if package_result is None:
        return "pending"
    if not package_result.ok:
        return "preflight_failed" if package_result.code == "preflight_failed" else "failed"
    if verification is not None and not verification.ok:
        return "failed"
    return "succeeded"


def _authored_export_job_record(
    build: AuthoredModuleBuild,
    package_result: CustomModulePackResult | None = None,
    verification: DevModulePackageVerification | None = None,
    *,
    proof_manifest_path: str = "",
    install_path: str = "",
    installed: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    walkmesh_gate = dict(build.metadata.get("walkmesh_gate", {}) or {})
    resource_gate = dict(build.metadata.get("resource_reference_gate", {}) or {})
    pathing = dict(build.metadata.get("pathing", {}) or {})
    visibility = dict(build.metadata.get("visibility", {}) or {})
    engine_contract = dict(build.metadata.get("engine_contract", {}) or {})
    smoke_expectations = dict(build.metadata.get("smoke_expectations", {}) or {})
    expected_placeables = smoke_expectations.get("expected_placeables")
    include_test_placeable = (
        bool(expected_placeables)
        if expected_placeables is not None
        else bool(getattr(build.project.placements, "placeables", ()) or ())
    )
    package_blocking = list(package_result.blocking_issues) if package_result is not None else []
    package_warnings = list(package_result.warnings) if package_result is not None else []
    verification_manifest = _verification_to_manifest(verification)
    return {
        "schema": "ghostrigger.authored_export_job.v1",
        "task": "T2913",
        "job_id": f"map_studio.authored_module.{build.module_root}",
        "kind": "map_studio.authored_module.mod_package",
        "module_root": build.module_root,
        "game": build.game,
        "status": _export_job_status(build, package_result, verification),
        "transaction_model": "preflight -> package -> readback -> proof_handoff",
        "preflight": {
            "ready": not bool(build.blocking_issues),
            "resource_count": len(build.resource_summaries),
            "blocking_issue_count": len(build.blocking_issues),
            "warning_count": len(build.warnings),
            "walkmesh_gate_ready": bool(walkmesh_gate.get("ready")),
            "resource_reference_gate_ready": bool(resource_gate.get("ready")),
            "pathing_ready": bool(pathing.get("ready", pathing.get("ok", False))),
            "visibility_ready": bool(visibility.get("ready")),
            "engine_contract_ready": bool(engine_contract.get("export_ready")),
            "engine_contract": engine_contract,
            "blocking_issues": list(build.blocking_issues),
            "warnings": list(build.warnings),
        },
        "package": {
            "ok": bool(package_result.ok) if package_result is not None else False,
            "code": package_result.code if package_result is not None else "not_packaged",
            "module_path": package_result.module_path if package_result is not None else "",
            "pack_manifest_path": package_result.manifest_path if package_result is not None else "",
            "resources_dir": package_result.resources_dir if package_result is not None else "",
            "blocking_issues": package_blocking,
            "warnings": package_warnings,
        },
        "outputs": _package_output_records(package_result),
        "readback": verification_manifest,
        "proof_handoff": {
            "required": True,
            "state": "installed_requires_live_warp_proof" if installed else "requires_live_warp_proof",
            "dry_run": bool(dry_run),
            "installed": bool(installed),
            "installed_module_path": install_path,
            "proof_manifest_path": proof_manifest_path,
            "acceptance_required": _authored_acceptance_checks(
                include_test_placeable=include_test_placeable,
                module_root=build.module_root,
            ),
        },
    }


def _export_job_for_install_proof(
    export_result: AuthoredModuleExportResult,
    *,
    proof_manifest_path: str,
    install_path: str,
    installed: bool,
    dry_run: bool,
) -> dict[str, Any]:
    export_job = dict(export_result.metadata.get("export_job") or {})
    if not export_job:
        export_job = {
            "schema": "ghostrigger.authored_export_job.v1",
            "task": "T2913",
            "job_id": f"map_studio.authored_module.{export_result.module_root}",
            "kind": "map_studio.authored_module.mod_package",
            "module_root": export_result.module_root,
            "status": "succeeded" if export_result.ok else "failed",
            "transaction_model": "preflight -> package -> readback -> proof_handoff",
            "package": {
                "ok": bool(export_result.ok),
                "module_path": export_result.module_path,
                "pack_manifest_path": export_result.manifest_path,
            },
            "outputs": [
                _artifact_output_record(export_result.module_path, "module_package"),
                _artifact_output_record(export_result.manifest_path, "pack_manifest"),
            ],
            "readback": _verification_to_manifest(export_result.package_verification),
        }
    proof_handoff = dict(export_job.get("proof_handoff") or {})
    proof_handoff.update(
        {
            "required": True,
            "state": "installed_requires_live_warp_proof" if installed else "requires_live_warp_proof",
            "dry_run": bool(dry_run),
            "installed": bool(installed),
            "installed_module_path": install_path,
            "proof_manifest_path": proof_manifest_path,
        }
    )
    export_job["proof_handoff"] = proof_handoff
    return export_job


def _authored_required_resource_rows(module_root: str, room_resrefs: tuple[str, ...]) -> list[dict[str, str]]:
    rows = [
        (module_root, "are", "Area metadata"),
        (module_root, "git", "Gameplay instances"),
        (ENGINE_MODULE_IFO_RESREF, "ifo", "Module entry metadata"),
        (module_root, "pth", "Path graph"),
        (module_root, "lyt", "Room layout"),
        (module_root, "vis", "Room visibility"),
    ]
    for room in room_resrefs:
        rows.extend(
            [
                (room, "wok", "Walkmesh"),
                (room, "mdl", "Room model"),
                (room, "mdx", "Room vertex data"),
            ]
        )
    return [
        {
            "resref": resref,
            "restype": restype,
            "filename": f"{resref}.{restype}",
            "purpose": purpose,
        }
        for resref, restype, purpose in rows
    ]


def _authored_smoke_contract_from_parts(
    *,
    module_root: str,
    room_resrefs: tuple[str, ...],
    built_resources: set[tuple[str, str]],
    verification: DevModulePackageVerification | None,
    capability_stage: str = "export_candidate",
    game_tested: bool = False,
) -> dict[str, Any]:
    verified = {
        (resource.resref, resource.restype)
        for resource in (verification.resources if verification is not None else ())
    }
    available = verified or built_resources
    required = _authored_required_resource_rows(module_root, room_resrefs)
    required_with_status = [
        dict(row, present=(row["resref"], row["restype"]) in available)
        for row in required
    ]
    missing = [row["filename"] for row in required_with_status if not row["present"]]
    return {
        "task": "T2601",
        "module_root": module_root,
        "warp_command": f"warp {module_root}",
        "capability_stage": capability_stage,
        "required_resources": required_with_status,
        "all_required_resources_present": not missing,
        "missing_required_resources": missing,
        "pre_game_package_readback_ok": bool(verification.ok) if verification is not None else False,
        "in_game_acceptance_checks": _authored_acceptance_checks(module_root=module_root),
        "proof_required": not game_tested,
        "game_tested": game_tested,
    }


def _authored_smoke_contract(build: AuthoredModuleBuild, verification: DevModulePackageVerification | None) -> dict[str, Any]:
    contract = _authored_smoke_contract_from_parts(
        module_root=build.module_root,
        room_resrefs=tuple(sorted(build.module.room_geometry)),
        built_resources={(summary.resref, summary.restype) for summary in build.resource_summaries},
        verification=verification,
    )
    expectations = build.metadata.get("smoke_expectations")
    if isinstance(expectations, dict):
        contract.update(expectations)
    contract["in_game_acceptance_checks"] = _authored_acceptance_checks(
        include_test_placeable=bool(contract.get("expected_placeables")),
        module_root=build.module_root,
    )
    contract["resource_reference_gate"] = dict(build.metadata.get("resource_reference_gate", {}) or {})
    return contract


def _authored_smoke_contract_from_export_result(export_result: AuthoredModuleExportResult) -> dict[str, Any]:
    module_root = export_result.module_root or "authored"
    contract = _authored_smoke_contract_from_parts(
        module_root=module_root,
        room_resrefs=tuple(export_result.room_resrefs),
        built_resources={(summary.resref, summary.restype) for summary in export_result.resources},
        verification=export_result.package_verification,
        capability_stage="export_candidate" if export_result.ok else "export_blocked",
    )
    expectations = export_result.metadata.get("smoke_expectations")
    if isinstance(expectations, dict):
        contract.update(expectations)
    contract["in_game_acceptance_checks"] = _authored_acceptance_checks(
        include_test_placeable=bool(contract.get("expected_placeables")),
        module_root=module_root,
    )
    contract["resource_reference_gate"] = dict(export_result.metadata.get("resource_reference_gate", {}) or {})
    return contract


def authored_module_smoke_summary_lines(export_result: AuthoredModuleExportResult) -> list[str]:
    """Return concise modder-facing smoke-test expectations for a staged module."""

    contract = _authored_smoke_contract_from_export_result(export_result)
    lines = [f"Smoke test: run `{contract['warp_command']}` in the matching KOTOR game."]
    entry = contract.get("expected_entry_point")
    if isinstance(entry, dict):
        position = entry.get("position", [])
        area = str(entry.get("area_resref") or contract.get("module_root") or "")
        if len(position) >= 3:
            lines.append(
                "Expected player start: "
                f"{area} at ({float(position[0]):.2f}, {float(position[1]):.2f}, {float(position[2]):.2f})."
            )
    placeables = [row for row in contract.get("expected_placeables", []) if isinstance(row, dict)]
    if placeables:
        labels = []
        for row in placeables:
            position = _vec3_to_manifest(row.get("position", (0.0, 0.0, 0.0)))
            labels.append(
                f"{row.get('tag') or row.get('template_resref')} @ "
                f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
        lines.append(f"Expected placeable(s): {', '.join(labels)}.")
    walkability = contract.get("walkability")
    if isinstance(walkability, dict):
        checks = [row for row in walkability.get("checks", []) if isinstance(row, dict)]
        passed = sum(1 for row in checks if row.get("ok"))
        if checks:
            lines.append(f"Walkability preflight: {passed}/{len(checks)} gameplay anchor(s) on generated WOK.")
    anchors = list(contract.get("pathing_anchor_labels") or [])
    if anchors:
        shown = ", ".join(str(label) for label in anchors[:8])
        suffix = "" if len(anchors) <= 8 else f", +{len(anchors) - 8} more"
        lines.append(f"Pathing anchors: {shown}{suffix}.")
    if contract.get("game_tested"):
        lines.append("Capability: game-smoke-tested.")
    else:
        lines.append("Capability: export candidate; in-game screenshot/video proof is still required.")
    return lines


def _authored_game_test_steps(module_root: str, *, include_test_placeable: bool = True) -> list[str]:
    steps = [
        f"Install/copy `{module_root}.mod` into the selected KOTOR `Modules` folder.",
        "Launch the matching KOTOR game.",
        f"Open the console and run `warp {module_root}`.",
        "Confirm the module loads without crashing or falling back to another area.",
        f"Confirm the area identity is the authored `{module_root}` smoke map, not a copied Taris/PLCaa/base-game module.",
        "Confirm the player appears on the generated floor, not in the void.",
    ]
    if include_test_placeable:
        steps.append("Confirm the authored test placeable appears near the expected location.")
    steps.extend(
        [
            "Walk across the generated floor and confirm movement is not blocked unexpectedly.",
            "Confirm transition/pathing sanity: authored PTH anchors are reachable and any door/transition links behave as expected.",
            "Confirm there are no inherited PLCaa scripted moving spheres, cones, rectangles, or other base-game test-room movers.",
            "Capture a screenshot or short clip as proof.",
        ]
    )
    return steps


def _authored_acceptance_checks(
    *,
    include_test_placeable: bool = True,
    module_root: str = "",
) -> list[str]:
    checks = [
        "module_loads_in_game",
        "module_identity_matches_authored_resref",
        "player_spawns_on_floor",
    ]
    if include_test_placeable:
        checks.append("test_placeable_visible")
    checks.extend(
        [
            "player_can_walk_on_floor",
            "transition_pathing_sanity_confirmed",
            "no_inherited_base_game_geometry_or_scripted_movers",
            "screenshot_or_video_captured",
        ]
    )
    if str(module_root or "").strip().lower() == "plcaa":
        # PLCaa is GhostStudio's user-driven acceptance fixture.  A successful
        # warp is necessary but insufficient: this gate records the exact
        # editor -> package -> KOTOR interactions the user must perform.
        checks.extend(
            [
                "texture_paint_visible_in_game",
                "terrain_sculpt_and_generated_walkmesh_work_in_game",
                "placed_assets_match_editor_staging",
                "enemy_spawns_hostile",
                "npc_spawns_and_free_roams",
                "terminal_operates",
                "container_opens_with_inventory",
                "puzzle_sequence_unlocks_door",
                "animated_door_operates",
                "configured_transition_operates",
                "player_start_position_and_facing_match",
            ]
        )
    return checks


def _authored_modder_test_plan(
    *,
    smoke_contract: dict[str, Any],
    module_path: str = "",
    install_path: str = "",
    proof_manifest_path: str = "",
    evidence_path: str = "",
    accepted: bool = False,
    missing_checks: list[str] | None = None,
    package_resource_inventory: dict[str, Any] | None = None,
    dry_run: bool = False,
    installed: bool = False,
) -> dict[str, Any]:
    """Build the durable game-proof checklist consumed by UI and scripts."""

    acceptance_checks = list(smoke_contract.get("in_game_acceptance_checks") or _authored_acceptance_checks())
    missing = list(missing_checks if missing_checks is not None else ([] if accepted else acceptance_checks))
    accepted_checks = [check for check in acceptance_checks if check not in set(missing)]
    if accepted:
        capability_stage = "game_smoke_tested"
        proof_state = "game_smoke_tested"
    elif installed:
        capability_stage = "installed_test_build"
        proof_state = "installed_requires_live_warp_proof"
    else:
        capability_stage = str(smoke_contract.get("capability_stage") or "export_candidate")
        proof_state = "requires_live_warp_proof"
    plan = {
        "task": "T2605",
        "module_root": str(smoke_contract.get("module_root") or ""),
        "capability_stage": capability_stage,
        "game_ready": bool(accepted),
        "proof_state": proof_state,
        "warp_command": str(smoke_contract.get("warp_command") or ""),
        "module_path": module_path,
        "install": {
            "installed_module_path": install_path,
            "installed": bool(installed),
            "dry_run": bool(dry_run),
            "proof_manifest_path": proof_manifest_path,
        },
        "required_resources": list(smoke_contract.get("required_resources") or ()),
        "missing_required_resources": list(smoke_contract.get("missing_required_resources") or ()),
        "expected_runtime_observations": dict(smoke_contract.get("expected_runtime_observations") or {}),
        "expected_absent_runtime_observations": dict(smoke_contract.get("expected_absent_runtime_observations") or {}),
        "expected_entry_point": dict(smoke_contract.get("expected_entry_point") or {}),
        "expected_placeables": list(smoke_contract.get("expected_placeables") or ()),
        "expected_waypoints": list(smoke_contract.get("expected_waypoints") or ()),
        "resource_reference_gate": dict(smoke_contract.get("resource_reference_gate") or {}),
        "walkability": dict(smoke_contract.get("walkability") or {}),
        "pathing_anchor_labels": list(smoke_contract.get("pathing_anchor_labels") or ()),
        "acceptance_checks": acceptance_checks,
        "accepted_acceptance_checks": accepted_checks,
        "missing_acceptance_checks": missing,
        "evidence": {
            "required": not bool(accepted),
            "path": evidence_path,
            "accepted_kinds": ["screenshot", "video"],
        },
        "modder_next_step": (
            "Keep the proof manifest and screenshot/video with the package."
            if accepted
            else "Install the staged .mod, warp into the module, verify every acceptance check, then record screenshot/video proof."
        ),
    }
    if package_resource_inventory:
        plan["package_resource_inventory"] = dict(package_resource_inventory)
    return plan


def _authored_package_resource_inventory(
    *,
    export_result: AuthoredModuleExportResult,
    smoke_contract: dict[str, Any],
    install_path: str = "",
    modules_dir: str = "",
    backup_path: str = "",
    installed: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the package resource inventory consumed by proof/status UI."""

    verification = export_result.package_verification
    verified_resources = [
        {
            "resref": resource.resref,
            "restype": resource.restype,
            "filename": f"{resource.resref}.{resource.restype}",
            "size": resource.size,
            "offset": resource.offset,
        }
        for resource in (verification.resources if verification is not None else ())
    ]
    verified_keys = {(row["resref"], row["restype"]) for row in verified_resources}
    package_result = export_result.package_result
    staged_resources = [
        {
            "resref": resource.resref,
            "restype": resource.restype,
            "filename": f"{resource.resref}.{resource.restype}",
            "path": resource.path,
            "size": resource.size,
            "sha256": resource.sha256,
            "source": resource.source,
        }
        for resource in (package_result.staged_resources if package_result is not None else ())
    ]
    required_resources = [
        dict(
            row,
            present_in_readback=(str(row.get("resref") or ""), str(row.get("restype") or "")) in verified_keys,
        )
        for row in list(smoke_contract.get("required_resources") or ())
        if isinstance(row, dict)
    ]
    restypes = {str(row.get("restype") or "") for row in required_resources}
    room_model_count = sum(1 for row in required_resources if row.get("restype") in {"mdl", "mdx"})
    room_walkmesh_count = sum(1 for row in required_resources if row.get("restype") == "wok")
    core_restypes = ("are", "git", "ifo", "pth", "lyt", "vis")
    return {
        "schema": "ghostrigger.map_studio.package_resource_inventory.v1",
        "module_root": export_result.module_root,
        "module_path": export_result.module_path,
        "pack_manifest_path": export_result.manifest_path,
        "readback_ok": bool(verification.ok) if verification is not None else False,
        "all_required_runtime_resources_present": bool(smoke_contract.get("all_required_resources_present")),
        "required_runtime_resources": required_resources,
        "missing_required_runtime_resources": list(smoke_contract.get("missing_required_resources") or ()),
        "verified_archive_resources": verified_resources,
        "loose_staged_resources": staged_resources,
        "resource_groups": {
            "core_module_restypes_present": sorted(restypes.intersection(core_restypes)),
            "core_module_restypes_required": list(core_restypes),
            "room_model_resource_count": room_model_count,
            "room_walkmesh_resource_count": room_walkmesh_count,
            "verified_archive_resource_count": len(verified_resources),
            "loose_staged_resource_count": len(staged_resources),
        },
        "resource_reference_gate": dict(smoke_contract.get("resource_reference_gate") or {}),
        "install": {
            "modules_dir": modules_dir,
            "installed_module_path": install_path,
            "backup_module_path": backup_path,
            "installed": bool(installed),
            "dry_run": bool(dry_run),
        },
    }


def _authored_game_executable_name(game: str) -> str:
    return "swkotor2.exe" if str(game or "").upper() == "K2" else "swkotor.exe"


def _derive_game_root_dir_from_modules_dir(modules_dir_text: str) -> str:
    if not modules_dir_text:
        return ""
    modules_dir = Path(modules_dir_text)
    return str(modules_dir.parent)


def _files_have_same_bytes(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


def _check_currentgame_module_cache(
    *,
    game_root_dir: str,
    module_root: str,
    module_path: str,
    warnings: list[str],
    blocking: list[str],
) -> None:
    if not game_root_dir or not module_root or not module_path:
        return
    cache_path = Path(game_root_dir) / "currentgame" / f"{module_root}.mod"
    if not cache_path.is_file():
        return
    package_path = Path(module_path)
    if package_path.is_file() and _files_have_same_bytes(cache_path, package_path):
        warnings.append(
            f"KOTOR currentgame already contains {cache_path.name}; restart from a clean save if warp testing behaves strangely."
        )
        return
    blocking.append(
        "KOTOR currentgame cache contains a stale "
        f"{cache_path.name}. Quit KOTOR and move or delete {cache_path} before retesting warp {module_root}."
    )


def _refresh_currentgame_module_cache(
    *,
    game_root_dir: str,
    module_root: str,
    installed_module_path: str,
    warnings: list[str],
) -> None:
    if not game_root_dir or not module_root or not installed_module_path:
        return
    cache_path = Path(game_root_dir) / "currentgame" / f"{module_root}.mod"
    if not cache_path.is_file():
        return
    installed_path = Path(installed_module_path)
    if installed_path.is_file() and _files_have_same_bytes(cache_path, installed_path):
        warnings.append(
            f"KOTOR currentgame already contains the installed {cache_path.name}; restart from a clean save if warp testing behaves strangely."
        )
        return
    backup = _next_install_backup_path(cache_path)
    shutil.copy2(cache_path, backup)
    shutil.copy2(installed_path, cache_path)
    warnings.append(f"Refreshed stale KOTOR currentgame cache {cache_path.name}; backup written to {backup}.")


def _authored_launch_helper_command(*, module_root: str, proof_manifest_path: Path, game: str, game_root_dir: str) -> str:
    if not game_root_dir:
        return ""
    script_name = "launch_grdev01_smoke_test.py" if normalise_resref(module_root) == "grdev01" else "launch_authored_module_smoke_test.py"
    return (
        f"python scripts/{script_name} "
        f'--proof-manifest "{proof_manifest_path}" '
        f'--game "{str(game or "K1").upper()}" '
        f'--game-root-dir "{game_root_dir}" '
        "--dry-run"
    )


def _powershell_single_quoted(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _write_authored_elevated_launch_script(
    *,
    output_root: Path,
    module_root: str,
    game: str,
    game_root_dir: str,
    proof_manifest_path: Path,
) -> str:
    if not game_root_dir:
        return ""
    executable = Path(game_root_dir) / _authored_game_executable_name(game)
    script_path = output_root / f"{module_root}_launch_kotor_as_admin.cmd"
    ps_command = (
        "Start-Process "
        f"-FilePath {_powershell_single_quoted(executable)} "
        f"-WorkingDirectory {_powershell_single_quoted(game_root_dir)} "
        "-Verb RunAs"
    )
    lines = [
        "@echo off",
        "setlocal",
        f"echo GhostRigger authored module smoke test: {module_root}",
        f"echo Expected module package is installed in: {Path(game_root_dir) / 'Modules'}",
        f"echo This helper starts KOTOR with Windows elevation, then you must run: warp {module_root}",
        f"echo Proof manifest: {proof_manifest_path}",
        "echo.",
        f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{ps_command}"',
        "if errorlevel 1 (",
        "  echo KOTOR did not launch. Start it manually as administrator and keep using this checklist.",
        "  pause",
        "  exit /b 1",
        ")",
        "echo.",
        f"echo After KOTOR opens, load a save, open the console, and run: warp {module_root}",
        "echo Capture screenshot/video evidence, then record proof with GhostRigger's proof recorder.",
        "pause",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(script_path)


def _record_proof_script_path() -> str:
    current = Path(__file__).resolve()
    for parent in current.parents:
        script_path = parent / "scripts" / "record_authored_module_game_proof.py"
        if script_path.is_file():
            return str(script_path)
    return "scripts\\record_authored_module_game_proof.py"


def _capture_authored_evidence_script_path() -> str:
    current = Path(__file__).resolve()
    for parent in current.parents:
        script_path = parent / "scripts" / "capture_authored_module_evidence.py"
        if script_path.is_file():
            return str(script_path)
    return "scripts\\capture_authored_module_evidence.py"


def _capture_evidence_command(*, proof_manifest_path: Path, module_root: str, include_test_placeable: bool = True) -> str:
    capture_path = _capture_authored_evidence_script_path()
    flags = [
        "--kotor-window-only",
        "--record-proof",
        "--module-loads-in-game",
        "--module-identity-matches-authored-resref",
        "--player-spawns-on-floor",
    ]
    if include_test_placeable:
        flags.append("--test-placeable-visible")
    flags.append("--player-can-walk-on-floor")
    flags.append("--transition-pathing-sanity-confirmed")
    flags.append("--no-inherited-base-game-geometry-or-scripted-movers")
    return f'python "{capture_path}" --proof-manifest "{proof_manifest_path}" ' + " ".join(flags)


def _write_authored_proof_recording_script(
    *,
    output_root: Path,
    module_root: str,
    proof_manifest_path: Path,
    include_test_placeable: bool = True,
) -> str:
    script_path = output_root / f"{module_root}_record_game_proof.cmd"
    recorder_path = _record_proof_script_path()
    proof_path = proof_manifest_path.resolve()
    placeable_text = "the authored placeable is visible, " if include_test_placeable else ""
    placeable_flag = "--test-placeable-visible " if include_test_placeable else ""
    lines = [
        "@echo off",
        "setlocal",
        f"echo GhostRigger authored module proof recorder: {module_root}",
        f"echo Proof manifest: {proof_path}",
        "echo.",
        f"echo Run this only after KOTOR has loaded the module with: warp {module_root}",
        f"echo Confirm the module identity is {module_root}, the player spawns on the generated floor, {placeable_text}walking works, transition/pathing sanity holds, and no PLCaa/base-game scripted movers are present.",
        "echo.",
        "set /p EVIDENCE=Drag or paste screenshot/video evidence path here, then press Enter: ",
        "set \"EVIDENCE=%EVIDENCE:\"=%\"",
        "if \"%EVIDENCE%\"==\"\" (",
        "  echo No evidence path supplied. Proof was not recorded.",
        "  pause",
        "  exit /b 1",
        ")",
        (
            f'python "{recorder_path}" --proof-manifest "{proof_path}" '
            '--evidence "%EVIDENCE%" --tester "%USERNAME%" '
            "--module-loads-in-game --module-identity-matches-authored-resref --player-spawns-on-floor "
            f"{placeable_flag}--player-can-walk-on-floor --transition-pathing-sanity-confirmed "
            "--no-inherited-base-game-geometry-or-scripted-movers"
        ),
        "if errorlevel 1 (",
        "  echo Proof recording did not complete. Check the message above.",
        "  pause",
        "  exit /b 1",
        ")",
        "echo.",
        "echo Proof recorded. Keep the evidence file with the packaged module.",
        "pause",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(script_path)


def _write_authored_install_proof_files(
    *,
    output_root: Path,
    export_result: AuthoredModuleExportResult,
    game: str,
    install_path: str,
    backup_path: str,
    installed_override_paths: tuple[str, ...],
    backup_override_paths: tuple[str, ...],
    modules_dir: str,
    game_root_dir: str,
    installed: bool,
    dry_run: bool,
    warnings: list[str],
    blocking: list[str],
) -> tuple[str, str, str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    module_root = export_result.module_root or "authored"
    smoke_contract = _authored_smoke_contract_from_export_result(export_result)
    include_test_placeable = bool(smoke_contract.get("expected_placeables"))
    checklist_path = output_root / f"{module_root}_authored_module_game_checklist.md"
    proof_manifest_path = output_root / f"{module_root}_authored_module_game_manifest.json"
    steps = _authored_game_test_steps(module_root, include_test_placeable=include_test_placeable)
    executable_name = _authored_game_executable_name(game)
    executable_path = str(Path(game_root_dir) / executable_name) if game_root_dir else executable_name
    launch_helper_command = _authored_launch_helper_command(
        module_root=module_root,
        proof_manifest_path=proof_manifest_path,
        game=game,
        game_root_dir=game_root_dir,
    )
    elevated_launch_script_path = _write_authored_elevated_launch_script(
        output_root=output_root,
        module_root=module_root,
        game=game,
        game_root_dir=game_root_dir,
        proof_manifest_path=proof_manifest_path,
    )
    proof_recording_script_path = _write_authored_proof_recording_script(
        output_root=output_root,
        module_root=module_root,
        proof_manifest_path=proof_manifest_path,
        include_test_placeable=include_test_placeable,
    )
    capture_evidence_command = _capture_evidence_command(
        proof_manifest_path=proof_manifest_path,
        module_root=module_root,
        include_test_placeable=include_test_placeable,
    )
    checklist_lines = [
        f"# {module_root} Authored Module In-Game Test",
        "",
        "This checklist is the manual proof gate for authored Map Studio modules. The module is not game-tested until these boxes are checked from an actual KOTOR run.",
        "",
        f"- Package: `{export_result.module_path}`",
        f"- Install target: `{install_path or '(not installed)'}`",
        f"- Previous module backup: `{backup_path or '(none)'}`",
        f"- Required Override resources: `{', '.join(installed_override_paths) or '(none)'}`",
        f"- Previous Override backups: `{', '.join(backup_override_paths) or '(none)'}`",
        f"- Game root: `{game_root_dir or '(not supplied)'}`",
        f"- Expected executable: `{executable_path}`",
        f"- Dry-run helper: `{launch_helper_command or '(manual launch)'}`",
        f"- Elevated launch helper: `{elevated_launch_script_path or '(not written)'}`",
        f"- Evidence capture helper: `{capture_evidence_command or '(manual screenshot/video capture)'}`",
        f"- Proof recorder: `{proof_recording_script_path}`",
        f"- Warp command: `warp {module_root}`",
        "",
        "## Steps",
        "",
    ]
    checklist_lines.extend(f"- [ ] {step}" for step in steps)
    checklist_lines.extend(
        [
            "",
            "## Evidence",
            "",
            "- Screenshot/video path:",
            "- Tester:",
            "- Game/version:",
            "- Result:",
            "",
        ]
    )
    checklist_path.write_text("\n".join(checklist_lines), encoding="utf-8")
    package_resource_inventory = _authored_package_resource_inventory(
        export_result=export_result,
        smoke_contract=smoke_contract,
        install_path=install_path,
        modules_dir=modules_dir,
        backup_path=backup_path,
        dry_run=dry_run,
        installed=installed,
    )
    modder_test_plan = _authored_modder_test_plan(
        smoke_contract=smoke_contract,
        module_path=export_result.module_path,
        install_path=install_path,
        proof_manifest_path=str(proof_manifest_path),
        package_resource_inventory=package_resource_inventory,
        dry_run=dry_run,
        installed=installed,
    )
    export_job = _export_job_for_install_proof(
        export_result,
        proof_manifest_path=str(proof_manifest_path),
        install_path=install_path,
        installed=installed,
        dry_run=dry_run,
    )
    proof_manifest = {
        "task": "T2644",
        "module_root": module_root,
        "game": str(game or "K1").upper(),
        "capability_stage": modder_test_plan["capability_stage"],
        "proof_state": modder_test_plan["proof_state"],
        "package": {
            "module_path": export_result.module_path,
            "pack_manifest_path": export_result.manifest_path,
            "verification": _verification_to_manifest(export_result.package_verification),
            "export_job": export_job,
            "resource_inventory": package_resource_inventory,
        },
        "export_job": export_job,
        "package_resource_inventory": package_resource_inventory,
        "install": {
            "installed": installed,
            "dry_run": dry_run,
            "installed_module_path": install_path,
            "backup_module_path": backup_path,
            "installed_override_paths": list(installed_override_paths),
            "backup_override_paths": list(backup_override_paths),
        },
        "manual_proof_required": True,
        "game_tested": False,
        "warp_command": f"warp {module_root}",
        "launch_handoff": {
            "game": str(game or "").upper() or "K1",
            "resolved_modules_dir": modules_dir,
            "resolved_game_root_dir": game_root_dir,
            "expected_executable_path": executable_path,
            "launch_helper_command": launch_helper_command,
            "elevated_launch_script_path": elevated_launch_script_path,
            "evidence_capture_command": capture_evidence_command,
            "proof_recording_script_path": proof_recording_script_path,
            "dry_run_first": bool(launch_helper_command),
            "warp_command": f"warp {module_root}",
        },
        "acceptance_checks": list(smoke_contract.get("in_game_acceptance_checks") or ()),
        "modder_test_plan": modder_test_plan,
        "t2601_smoke_contract": smoke_contract,
        "steps": steps,
        "warnings": warnings,
        "blocking_issues": blocking,
    }
    proof_manifest_path.write_text(json.dumps(proof_manifest, indent=2), encoding="utf-8")
    return str(checklist_path), str(proof_manifest_path), elevated_launch_script_path, proof_recording_script_path


def _install_prep_export_request(request: AuthoredModuleInstallPrepRequest) -> AuthoredModuleExportRequest:
    if request.export_request is not None:
        return AuthoredModuleExportRequest(
            project=request.export_request.project,
            output_dir=request.export_request.output_dir or request.output_dir,
            include_reference_check=request.export_request.include_reference_check,
            include_wok_check=request.export_request.include_wok_check,
            include_game_template_check=request.export_request.include_game_template_check,
            game_root_dir=request.export_request.game_root_dir,
            strict=request.export_request.strict,
            dry_run=False,
            create_backups=request.export_request.create_backups,
            write_loose_resources=request.export_request.write_loose_resources,
            extra_resources=request.export_request.extra_resources,
            secondary_motion_profiles=request.export_request.secondary_motion_profiles,
        )
    return AuthoredModuleExportRequest(
        project=request.project,
        output_dir=request.output_dir,
        game_root_dir=request.game_root_dir,
        strict=True,
        dry_run=False,
    )


def prepare_authored_module_install(request: AuthoredModuleInstallPrepRequest) -> AuthoredModuleInstallPrepResult:
    """Safely stage an authored KMAP module for a manual in-game warp test."""

    export_request = _install_prep_export_request(request)
    export_result = export_authored_module_project(export_request)
    output_root = Path(export_request.output_dir or request.output_dir or ".")
    warnings = list(export_result.warnings)
    blocking = list(export_result.blocking_issues)
    installed = False
    install_path = ""
    backup_path = ""
    installed_override_paths: list[str] = []
    backup_override_paths: list[str] = []
    modules_dir_text = request.game_modules_dir
    resolved_game_root_dir = request.game_root_dir
    if not modules_dir_text and request.auto_detect_game_modules_dir:
        modules_dir_text = discover_kotor_modules_dir(
            request.project.game,
            game_root_dir=request.game_root_dir,
            settings_path=request.settings_path,
        )
        if modules_dir_text:
            warnings.append(f"Auto-detected KOTOR Modules folder: {modules_dir_text}")
        else:
            warnings.append("Could not auto-detect a KOTOR Modules folder; package is staged for manual install.")
    if not resolved_game_root_dir and modules_dir_text:
        resolved_game_root_dir = _derive_game_root_dir_from_modules_dir(modules_dir_text)
    if not export_result.ok:
        blocking.append(export_result.message or "Authored module export failed.")
    else:
        if not modules_dir_text or request.dry_run:
            _check_currentgame_module_cache(
                game_root_dir=resolved_game_root_dir,
                module_root=export_result.module_root,
                module_path=export_result.module_path,
                warnings=warnings,
                blocking=blocking,
            )
        if blocking:
            pass
        elif not modules_dir_text:
            warnings.append("No KOTOR Modules folder was supplied; package is staged but not installed for game testing.")
        else:
            modules_dir = Path(modules_dir_text)
            if not modules_dir.is_dir():
                blocking.append(f"KOTOR Modules folder does not exist: {modules_dir}")
            else:
                destination = modules_dir / f"{export_result.module_root}.mod"
                install_path = str(destination)
                if destination.exists() and not request.overwrite:
                    blocking.append(f"{destination} already exists. Re-run with overwrite=True to replace it.")
                override_resources = tuple(
                    getattr(export_result.package_result, "staged_override_resources", ()) or ()
                )
                game_root = Path(resolved_game_root_dir) if resolved_game_root_dir else modules_dir.parent
                override_dir = game_root / "Override"
                override_targets = [override_dir / Path(str(resource.path)).name for resource in override_resources]
                for target in override_targets:
                    if target.exists() and not request.overwrite:
                        blocking.append(
                            f"{target} already exists. Re-run with overwrite=True to merge the generated placeable table with a backup."
                        )
                if blocking:
                    pass
                elif request.dry_run:
                    warnings.append(f"Dry run: would copy {export_result.module_path} to {destination}.")
                    warnings.extend(
                        f"Dry run: would copy {resource.path} to {target}."
                        for resource, target in zip(override_resources, override_targets, strict=True)
                    )
                else:
                    override_dir.mkdir(parents=True, exist_ok=True)
                    for resource, target in zip(override_resources, override_targets, strict=True):
                        source = Path(str(resource.path))
                        if not source.is_file():
                            raise FileNotFoundError(f"Generated Override resource is missing: {source}")
                        if target.exists():
                            backup = _next_install_backup_path(target)
                            shutil.copy2(target, backup)
                            backup_override_paths.append(str(backup))
                            warnings.append(f"Backed up existing {target.name} to {backup}.")
                        shutil.copy2(source, target)
                        installed_override_paths.append(str(target))
                    if destination.exists():
                        backup = _next_install_backup_path(destination)
                        shutil.copy2(destination, backup)
                        backup_path = str(backup)
                        warnings.append(f"Backed up existing {destination.name} to {backup}.")
                    shutil.copy2(export_result.module_path, destination)
                    installed = True
                    _refresh_currentgame_module_cache(
                        game_root_dir=resolved_game_root_dir,
                        module_root=export_result.module_root,
                        installed_module_path=str(destination),
                        warnings=warnings,
                    )

    checklist_path, proof_manifest_path, elevated_launch_script_path, proof_recording_script_path = _write_authored_install_proof_files(
        output_root=output_root,
        export_result=export_result,
        game=request.project.game,
        install_path=install_path,
        backup_path=backup_path,
        installed_override_paths=tuple(installed_override_paths),
        backup_override_paths=tuple(backup_override_paths),
        modules_dir=modules_dir_text,
        game_root_dir=resolved_game_root_dir,
        installed=installed,
        dry_run=request.dry_run,
        warnings=warnings,
        blocking=blocking,
    )
    launch_helper_command = _authored_launch_helper_command(
        module_root=export_result.module_root,
        proof_manifest_path=Path(proof_manifest_path),
        game=request.project.game,
        game_root_dir=resolved_game_root_dir,
    )
    ok = export_result.ok and not blocking and (installed or request.dry_run or not request.game_modules_dir)
    code = "installed" if installed else "staged_for_manual_install"
    if request.dry_run and not blocking:
        code = "dry_run"
    if blocking:
        code = "install_preflight_failed"
    return AuthoredModuleInstallPrepResult(
        ok=ok,
        export_result=export_result,
        installed_module_path=install_path if installed else "",
        backup_module_path=backup_path,
        installed_override_paths=tuple(installed_override_paths) if installed else (),
        backup_override_paths=tuple(backup_override_paths),
        resolved_modules_dir=modules_dir_text,
        resolved_game_root_dir=resolved_game_root_dir,
        launch_helper_command=launch_helper_command,
        elevated_launch_script_path=elevated_launch_script_path,
        proof_recording_script_path=proof_recording_script_path,
        checklist_path=checklist_path,
        proof_manifest_path=proof_manifest_path,
        warnings=warnings,
        blocking_issues=blocking,
        message=(
            f"Installed {export_result.module_root}.mod to {install_path}"
            + (
                f" and {len(installed_override_paths)} required Override resource(s)."
                if installed_override_paths
                else "."
            )
            if installed
            else "Authored module staged with manual in-game checklist."
        ),
        code=code,
    )


def _proof_request_checks(request: AuthoredModuleGameProofRequest) -> dict[str, bool]:
    return {
        "module_loads_in_game": bool(request.module_loads_in_game),
        "module_identity_matches_authored_resref": bool(request.module_identity_matches_authored_resref),
        "player_spawns_on_floor": bool(request.player_spawns_on_floor),
        "test_placeable_visible": bool(request.test_placeable_visible),
        "player_can_walk_on_floor": bool(request.player_can_walk_on_floor),
        "transition_pathing_sanity_confirmed": bool(request.transition_pathing_sanity_confirmed),
        "no_inherited_base_game_geometry_or_scripted_movers": bool(
            request.no_inherited_base_game_geometry_or_scripted_movers
        ),
        "texture_paint_visible_in_game": bool(request.texture_paint_visible_in_game),
        "terrain_sculpt_and_generated_walkmesh_work_in_game": bool(
            request.terrain_sculpt_and_generated_walkmesh_work_in_game
        ),
        "placed_assets_match_editor_staging": bool(request.placed_assets_match_editor_staging),
        "enemy_spawns_hostile": bool(request.enemy_spawns_hostile),
        "npc_spawns_and_free_roams": bool(request.npc_spawns_and_free_roams),
        "terminal_operates": bool(request.terminal_operates),
        "container_opens_with_inventory": bool(request.container_opens_with_inventory),
        "puzzle_sequence_unlocks_door": bool(request.puzzle_sequence_unlocks_door),
        "animated_door_operates": bool(request.animated_door_operates),
        "configured_transition_operates": bool(request.configured_transition_operates),
        "player_start_position_and_facing_match": bool(request.player_start_position_and_facing_match),
        "screenshot_or_video_captured": _valid_proof_evidence_path(request.evidence_path),
    }


_PROOF_EVIDENCE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


def _valid_proof_evidence_path(evidence_path: str) -> bool:
    if not evidence_path:
        return False
    path = Path(evidence_path)
    return path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in _PROOF_EVIDENCE_EXTENSIONS


def _update_authored_pack_manifest_for_game_proof(
    *,
    pack_manifest_path: str,
    accepted: bool,
    proof_payload: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not pack_manifest_path:
        warnings.append("No pack manifest path was available to update with authored module in-game proof.")
        return warnings
    manifest_path = Path(pack_manifest_path)
    if not manifest_path.is_file():
        warnings.append(f"Pack manifest was not found for authored module in-game proof update: {manifest_path}")
        return warnings
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Pack manifest could not be read for authored module in-game proof update: {exc}")
        return warnings
    authored = manifest.setdefault("map_studio_authored_module", {})
    authored["game_tested"] = bool(accepted)
    contract = authored.get("t2601_smoke_contract")
    if isinstance(contract, dict):
        contract["game_tested"] = bool(accepted)
        contract["proof_required"] = not accepted
        if accepted:
            contract["capability_stage"] = "game_smoke_tested"
    if accepted:
        authored["capability_stage"] = "game_smoke_tested"
        remaining = list(authored.get("remaining_acceptance", []))
        authored["remaining_acceptance"] = [
            item
            for item in remaining
            if not any(
                needle in str(item).lower()
                for needle in ("install/copy", "warp", "confirm the module", "confirm the player", "placeables", "waypoints", "walk across", "capture")
            )
        ]
    plan = authored.get("modder_test_plan")
    if isinstance(plan, dict):
        plan["game_ready"] = bool(accepted)
        plan["proof_state"] = "game_smoke_tested" if accepted else "requires_live_warp_proof"
        plan["capability_stage"] = "game_smoke_tested" if accepted else str(plan.get("capability_stage") or "export_candidate")
        plan["missing_acceptance_checks"] = list(proof_payload.get("missing_checks") or [])
        plan["accepted_acceptance_checks"] = list(proof_payload.get("accepted_checks") or [])
        evidence = plan.setdefault("evidence", {})
        if isinstance(evidence, dict):
            evidence["required"] = not bool(accepted)
            evidence["path"] = str(proof_payload.get("evidence_path") or "")
        plan["modder_next_step"] = (
            "Keep the proof manifest and screenshot/video with the package."
            if accepted
            else "Resolve missing in-game proof checks, then record fresh screenshot/video evidence."
        )
    export_job = authored.get("export_job")
    if isinstance(export_job, dict):
        proof_handoff = export_job.setdefault("proof_handoff", {})
        if isinstance(proof_handoff, dict):
            proof_handoff["required"] = not bool(accepted)
            proof_handoff["state"] = "game_smoke_tested" if accepted else "requires_live_warp_proof"
            proof_handoff["evidence_path"] = str(proof_payload.get("evidence_path") or "")
            proof_handoff["missing_acceptance_checks"] = list(proof_payload.get("missing_checks") or [])
            proof_handoff["accepted_acceptance_checks"] = list(proof_payload.get("accepted_checks") or [])
            proof_handoff["accepted"] = bool(accepted)
        if accepted:
            export_job["status"] = "game_smoke_tested"
        manifest["export_job"] = export_job
    authored["in_game_proof"] = proof_payload
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return warnings


def record_authored_module_game_proof(request: AuthoredModuleGameProofRequest) -> AuthoredModuleGameProofResult:
    """Record the manual in-game proof gate for an authored Map Studio module."""

    proof_manifest_path = Path(request.proof_manifest_path)
    if not proof_manifest_path.is_file():
        return AuthoredModuleGameProofResult(
            ok=False,
            proof_manifest_path=str(proof_manifest_path),
            evidence_path=request.evidence_path,
            blocking_issues=[f"Proof manifest does not exist: {proof_manifest_path}"],
            message="Could not record authored module in-game proof because the proof manifest is missing.",
            code="proof_manifest_missing",
        )
    try:
        proof = json.loads(proof_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return AuthoredModuleGameProofResult(
            ok=False,
            proof_manifest_path=str(proof_manifest_path),
            evidence_path=request.evidence_path,
            blocking_issues=[f"Proof manifest could not be read: {exc}"],
            message="Could not record authored module in-game proof because the proof manifest is unreadable.",
            code="proof_manifest_unreadable",
        )

    required = list(proof.get("acceptance_checks") or _authored_acceptance_checks())
    checks = _proof_request_checks(request)
    missing = [name for name in required if not checks.get(name, False)]
    accepted_checks = [name for name in required if checks.get(name, False)]
    accepted = not missing
    tested_at = datetime.now(timezone.utc).isoformat()
    proof_payload = {
        "tested_at": tested_at,
        "tester": request.tester,
        "evidence_path": request.evidence_path,
        "notes": request.notes,
        "checks": checks,
        "accepted": accepted,
        "accepted_checks": accepted_checks,
        "missing_checks": missing,
    }
    proof["game_test"] = proof_payload
    proof["manual_proof_required"] = not accepted
    proof["game_tested"] = accepted
    contract = proof.get("t2601_smoke_contract")
    install = proof.get("install") if isinstance(proof.get("install"), dict) else {}
    installed = bool(install.get("installed") or str(install.get("installed_module_path") or "").strip())
    if isinstance(contract, dict):
        contract["game_tested"] = accepted
        contract["proof_required"] = not accepted
        if accepted:
            contract["capability_stage"] = "game_smoke_tested"
        package_resource_inventory = proof.get("package_resource_inventory")
        if not isinstance(package_resource_inventory, dict):
            package = proof.get("package")
            if isinstance(package, dict):
                package_resource_inventory = package.get("resource_inventory")
        if not isinstance(package_resource_inventory, dict):
            package_resource_inventory = {}
        proof["modder_test_plan"] = _authored_modder_test_plan(
            smoke_contract=contract,
            module_path=str((proof.get("package") or {}).get("module_path") or ""),
            install_path=str((proof.get("install") or {}).get("installed_module_path") or ""),
            proof_manifest_path=str(proof_manifest_path),
            evidence_path=request.evidence_path,
            accepted=accepted,
            missing_checks=missing,
            package_resource_inventory=package_resource_inventory,
            dry_run=bool(install.get("dry_run")),
            installed=installed,
        )
        proof["capability_stage"] = proof["modder_test_plan"]["capability_stage"]
        proof["proof_state"] = proof["modder_test_plan"]["proof_state"]
    if accepted:
        proof["completed_at"] = tested_at
    export_job = proof.get("export_job")
    if isinstance(export_job, dict):
        proof_handoff = export_job.setdefault("proof_handoff", {})
        if isinstance(proof_handoff, dict):
            proof_handoff["required"] = not bool(accepted)
            proof_handoff["state"] = "game_smoke_tested" if accepted else "requires_live_warp_proof"
            proof_handoff["evidence_path"] = request.evidence_path
            proof_handoff["missing_acceptance_checks"] = missing
            proof_handoff["accepted_acceptance_checks"] = accepted_checks
            proof_handoff["accepted"] = bool(accepted)
        if accepted:
            export_job["status"] = "game_smoke_tested"
        proof["export_job"] = export_job
        package = proof.get("package")
        if isinstance(package, dict):
            package["export_job"] = export_job
    proof_manifest_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")

    pack_manifest_path = str((proof.get("package") or {}).get("pack_manifest_path") or "")
    warnings = _update_authored_pack_manifest_for_game_proof(
        pack_manifest_path=pack_manifest_path,
        accepted=accepted,
        proof_payload=proof_payload,
    )
    blocking = [f"In-game acceptance check is incomplete: {name}" for name in missing]
    return AuthoredModuleGameProofResult(
        ok=accepted,
        proof_manifest_path=str(proof_manifest_path),
        pack_manifest_path=pack_manifest_path,
        evidence_path=request.evidence_path,
        missing_checks=missing,
        warnings=warnings,
        blocking_issues=blocking,
        message=(
            "Recorded complete in-game proof for authored Map Studio module."
            if accepted
            else "Recorded incomplete authored module in-game proof attempt; module remains unproven."
        ),
        code="game_proof_recorded" if accepted else "game_proof_incomplete",
    )


__all__ = [
    "AuthoredModuleBuild",
    "AuthoredModuleExportRequest",
    "AuthoredModuleExportResult",
    "AuthoredModuleGameProofRequest",
    "AuthoredModuleGameProofResult",
    "AuthoredModuleInstallPrepRequest",
    "AuthoredModuleInstallPrepResult",
    "AuthoredModuleResourceSummary",
    "authored_module_smoke_summary_lines",
    "build_authored_module",
    "export_authored_module_project",
    "neutralize_fullbright_room_mdl_lights",
    "prepare_authored_module_install",
    "record_authored_module_game_proof",
]
