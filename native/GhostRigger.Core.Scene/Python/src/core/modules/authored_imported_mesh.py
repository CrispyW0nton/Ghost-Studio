"""Imported stock-geometry room primitive for Map Studio.

Owns the "edit a real game room" seam: a stock KOTOR room model (MDL/MDX from
the game directory) is baked into an editable, KMAP-serializable room
primitive so component-modeling operations (delete, retexture) can customize any
module and the export pipeline can compile the result as a new map.

Geometry is stored per texture surface (mirroring MDL trimesh nodes), with
transforms baked to room-local space.  Original UVs are preserved on import
so texture swaps work without re-unwrapping; faces that gain a new texture
keep their existing UVs, and brand-new geometry can fall back to
``planar_uvs_for_vertices``.

KMAP payloads pack vertex data as base64 little-endian float32/int32 blocks:
a stock room holds tens of thousands of floats, and inline JSON lists would
bloat .kmap files past usability.  Everything else stays readable JSON.
"""

from __future__ import annotations

import base64
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any
from weakref import ReferenceType, ref

from src.core.geometry.mesh_topology import MeshTopology, compact_indexed_mesh
from src.core.geometry.polygon_mesh_operations import AttributeChannel, IndexedPolygonMesh
from src.core.geometry.solid_boolean import difference_closed_solid_meshes

from .authored_room_geometry import AuthoredRoomGeometry, PrimitiveMesh
from .authored_walkmesh_audit import audit_authored_wok
from .authored_walkmesh_surfaces import is_walkable_walkmesh_surface
from .module_format import NON_WALK_ID, WOKData, WOKFace

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Face = tuple[int, int, int]

IMPORTED_MESH_PRIMITIVE_KIND = "imported_mesh"

#: Largest face count observed across the complete installed K1/K2 vanilla WOK
#: census (1,202 K1 + 1,236 K2 resources).  The
#: generator deliberately does not decimate, but reports when render-resolution
#: output exceeds this empirical stock envelope.
EMPIRICAL_STOCK_WOK_FACE_MAX = 2_136

#: Versioned KMAP metadata written by Quad Draw.  Imported Odyssey room meshes
#: are triangulated, so a triangle pair alone cannot prove that it was authored
#: as a quad.  Insert Edge Loop deliberately trusts only this provenance rather
#: than guessing edge rings across arbitrary stock triangulation.
LOGICAL_QUAD_PROVENANCE_KEY = "logical_quad_provenance"
LOGICAL_QUAD_PROVENANCE_VERSION = 1

#: Named, versioned policy written only when a creator deliberately chooses
#: to replace an imported room's dynamic stock node graph with a newly
#: compiled static room graph.  Keeping this separate from ``preserved`` is
#: important: the source animations/lights/emitters/references are still lost,
#: but the loss is acknowledged instead of being misreported as preservation.
SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_POLICY = "authored_static_room_rebuild"
SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_VERSION = 1
_SOURCE_RUNTIME_GRAPH_COUNT_KEYS = (
    "animation_count",
    "light_count",
    "emitter_count",
    "reference_count",
)

#: Versioned proof that a creator reviewed which imported render surfaces are
#: genuine floor candidates.  A missing source WOK cannot tell us whether an
#: upward-facing triangle is a floor, roof, table, or decorative ledge, so the
#: generic generator refuses stock-import geometry until this intent exists.
IMPORTED_WALKMESH_GENERATION_INTENT_POLICY = "reviewed_render_floor_selection"
IMPORTED_WALKMESH_GENERATION_INTENT_VERSION = 1

#: Mesh role of the first surface (the preview model names the room mesh
#: "render"); remaining surfaces use ``imported_srf_<index>``.
RENDER_SURFACE_ROLE = "render"


@dataclass(frozen=True)
class ImportedMeshSurface:
    """One texture surface of an imported room (room-local space)."""

    name: str
    texture: str
    vertices: tuple[Vec3, ...]
    faces: tuple[Face, ...]
    face_mats: tuple[int, ...] = ()
    uvs: tuple[Vec2, ...] = ()
    normals: tuple[Vec3, ...] = ()
    lightmap: str = ""
    diffuse: Vec3 = (1.0, 1.0, 1.0)
    ambient: Vec3 = (1.0, 1.0, 1.0)
    #: Multi-texture data from the source MDL node: KOTOR area meshes often
    #: keep the diffuse map in texture_names[0] with a lightmap channel, and
    #: dropping these renders the room as an untextured fallback slab.
    texture_names: tuple[str, ...] = ()
    tex_count: int = 1
    uvs_lm: tuple[Vec2, ...] = ()
    specular: Vec3 = (0.0, 0.0, 0.0)
    shininess: float = 0.0
    alpha: float = 1.0
    has_shadow: bool = True
    render: bool = True
    selfillum: Vec3 = (0.0, 0.0, 0.0)
    transparency_hint: int = 0
    beaming: bool = False
    background_geometry: bool = False
    rotate_texture: bool = False
    animate_uv: bool = False
    uv_dir_x: float = 0.0
    uv_dir_y: float = 0.0
    uv_jitter: float = 0.0
    uv_jitter_speed: float = 0.0
    dirt_enabled: bool = False
    dirt_texture: int = 0
    dirt_coord_space: int = 0
    hide_in_holograms: bool = False
    mesh_average_point: Vec3 = (0.0, 0.0, 0.0)
    mesh_unknown0: bytes = b"\x00" * 24
    #: Visual sky/far-backdrop layer. The stable surface index is unchanged;
    #: Map Studio only changes visibility/pickability for this surface.
    backdrop: bool = False


@dataclass(frozen=True)
class ImportedMeshRoomPrimitive:
    """A stock KOTOR room baked into editable authored geometry."""

    room_resref: str
    surfaces: tuple[ImportedMeshSurface, ...]
    source_model: str = ""
    game: str = "K1"
    wok: WOKData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportedMeshValidation:
    ok: bool
    warnings: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()


# Export deliberately validates a project before compiling it, while imported
# room compilation also validates its primitive for callers that use that API
# directly.  A large seam-heavy OBJ makes the topology audit the dominant cost,
# so retain a small process-local result for the same immutable primitive
# instance.  The cache is identity checked through a weak reference (rather
# than hashing the nested metadata/WOK objects) and cannot keep project data
# alive.  Edits create a replacement frozen primitive, naturally missing the
# cache.
_IMPORTED_MESH_VALIDATION_CACHE_MAX = 16
_IMPORTED_MESH_VALIDATION_CACHE: OrderedDict[
    int,
    tuple[ReferenceType[ImportedMeshRoomPrimitive], ImportedMeshValidation],
] = OrderedDict()


def authored_room_uses_unresolved_stock_geometry(room: Any) -> bool:
    """Return whether ``room`` is a source-preservation placeholder.

    The explicit source/status/type checks keep a stray metadata boolean from
    suppressing ordinary authored geometry, collision, or export validation.
    """

    if isinstance(getattr(room, "primitive", None), ImportedMeshRoomPrimitive):
        return False
    metadata = dict(getattr(room, "metadata", {}) or {})
    return (
        str(metadata.get("source") or "").strip().lower() == "stock_module_import"
        and str(metadata.get("stock_geometry_status") or "").strip().lower() == "unresolved"
        and bool(metadata.get("pie_exclude_unresolved_stock_geometry", False))
    )


def imported_mesh_source_runtime_counts(primitive: ImportedMeshRoomPrimitive) -> dict[str, int]:
    """Return the dynamic source-node counts recorded by stock import."""

    graph = dict(dict(getattr(primitive, "metadata", {}) or {}).get("source_runtime_graph") or {})
    return {
        key: int(graph.get(key, 0) or 0)
        for key in _SOURCE_RUNTIME_GRAPH_COUNT_KEYS
        if int(graph.get(key, 0) or 0) > 0
    }


def imported_mesh_has_explicit_static_runtime_rebuild(primitive: ImportedMeshRoomPrimitive) -> bool:
    """Validate the lossy static-rebuild acknowledgement on one primitive.

    The recorded source counts must exactly match the current import audit, so
    a stale or hand-copied acknowledgement cannot silently waive a different
    room's runtime graph gate.
    """

    graph = dict(dict(getattr(primitive, "metadata", {}) or {}).get("source_runtime_graph") or {})
    policy = dict(graph.get("replacement_policy") or {})
    source_counts = imported_mesh_source_runtime_counts(primitive)
    try:
        recorded_counts = {
            str(key): int(value)
            for key, value in dict(policy.get("discarded_source_counts") or {}).items()
            if int(value or 0) > 0
        }
    except (TypeError, ValueError):
        return False
    return bool(
        source_counts
        and not bool(graph.get("preserved", False))
        and str(policy.get("kind") or "") == SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_POLICY
        and int(policy.get("version", 0) or 0) == SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_VERSION
        and str(policy.get("output_contract") or "") == "new_static_room_mdl"
        and bool(str(policy.get("reason") or "").strip())
        and recorded_counts == source_counts
    )


def prepare_imported_mesh_for_static_runtime_rebuild(
    primitive: ImportedMeshRoomPrimitive,
    *,
    reason: str,
) -> ImportedMeshRoomPrimitive:
    """Explicitly authorize replacing a stock runtime graph with static MDL.

    This is intentionally lossy and is suitable only when the creator has
    decided that the imported animations, model lights, emitters, and
    references are not part of the authored result.  Their original counts
    stay in KMAP for auditability; export emits a warning and compiles a fresh
    static room rather than claiming the graph was preserved.
    """

    clean_reason = str(reason or "").strip()
    if not clean_reason:
        raise ValueError("Static runtime-graph rebuild requires a written reason.")
    metadata = dict(getattr(primitive, "metadata", {}) or {})
    graph = dict(metadata.get("source_runtime_graph") or {})
    if not graph:
        raise ValueError("Imported room has no audited source runtime graph to replace.")
    source_counts = imported_mesh_source_runtime_counts(primitive)
    if not source_counts:
        raise ValueError("Imported room has no dynamic source runtime components to replace.")
    if bool(graph.get("preserved", False)):
        raise ValueError("Imported room runtime graph is already marked as preserved.")
    graph["replacement_policy"] = {
        "kind": SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_POLICY,
        "version": SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_VERSION,
        "output_contract": "new_static_room_mdl",
        "reason": clean_reason,
        "discarded_source_counts": dict(source_counts),
    }
    metadata["source_runtime_graph"] = graph
    return replace(primitive, metadata=metadata)


def prepare_imported_mesh_walkmesh_generation_intent(
    primitive: ImportedMeshRoomPrimitive,
    *,
    surface_faces: dict[int, tuple[int, ...] | None],
    reason: str,
) -> ImportedMeshRoomPrimitive:
    """Record an explicit, KMAP-persistent render-to-WOK floor selection.

    ``None`` selects every face on a reviewed surface; a tuple selects only the
    named face indices.  Slope/normal filtering still runs afterwards.  This
    helper never generates collision and deliberately rejects empty, stale, or
    out-of-range selections before they can be persisted.
    """

    clean_reason = str(reason or "").strip()
    if not clean_reason:
        raise ValueError("Imported walkmesh generation intent requires a written review reason.")
    if not isinstance(surface_faces, dict) or not surface_faces:
        raise ValueError("Imported walkmesh generation intent requires at least one reviewed surface.")

    rows: list[dict[str, Any]] = []
    for raw_surface_index, raw_face_indices in sorted(surface_faces.items(), key=lambda item: int(item[0])):
        if isinstance(raw_surface_index, bool):
            raise ValueError("Walkmesh surface indices must be integers, not booleans.")
        surface_index = int(raw_surface_index)
        if surface_index < 0 or surface_index >= len(primitive.surfaces):
            raise ValueError(
                f"Walkmesh surface index {surface_index} is outside the imported room's "
                f"{len(primitive.surfaces)} surfaces."
            )
        surface = primitive.surfaces[surface_index]
        row: dict[str, Any] = {
            "surface_index": surface_index,
            "surface_name": str(surface.name or ""),
        }
        if raw_face_indices is not None:
            if isinstance(raw_face_indices, (str, bytes)):
                raise ValueError("Walkmesh face selections must be integer sequences or null for all faces.")
            face_indices: list[int] = []
            for raw_face_index in raw_face_indices:
                if isinstance(raw_face_index, bool):
                    raise ValueError("Walkmesh face indices must be integers, not booleans.")
                face_index = int(raw_face_index)
                if face_index < 0 or face_index >= len(surface.faces):
                    raise ValueError(
                        f"Walkmesh face index {face_index} is outside surface {surface_index}'s "
                        f"{len(surface.faces)} faces."
                    )
                face_indices.append(face_index)
            unique_faces = sorted(set(face_indices))
            if not unique_faces:
                raise ValueError(f"Walkmesh surface {surface_index} has an empty reviewed face selection.")
            row["face_indices"] = unique_faces
        rows.append(row)

    metadata = dict(primitive.metadata or {})
    metadata["walkmesh_generation_intent"] = {
        "policy": IMPORTED_WALKMESH_GENERATION_INTENT_POLICY,
        "version": IMPORTED_WALKMESH_GENERATION_INTENT_VERSION,
        "reason": clean_reason,
        "surfaces": rows,
    }
    return replace(primitive, metadata=metadata)


def _imported_walkmesh_generation_selection(
    primitive: ImportedMeshRoomPrimitive,
) -> dict[int, frozenset[int] | None] | None:
    """Validate and decode persisted floor intent for the generator."""

    metadata = dict(primitive.metadata or {})
    raw_intent = metadata.get("walkmesh_generation_intent")
    if not isinstance(raw_intent, dict):
        return None
    if (
        str(raw_intent.get("policy") or "") != IMPORTED_WALKMESH_GENERATION_INTENT_POLICY
        or int(raw_intent.get("version", 0) or 0) != IMPORTED_WALKMESH_GENERATION_INTENT_VERSION
        or not str(raw_intent.get("reason") or "").strip()
    ):
        raise ValueError("Imported walkmesh generation intent has an unsupported or incomplete policy header.")
    rows = raw_intent.get("surfaces")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Imported walkmesh generation intent has no reviewed surfaces.")

    selection: dict[int, frozenset[int] | None] = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("surface_index"), bool):
            raise ValueError("Imported walkmesh generation intent contains an invalid surface row.")
        surface_index = int(row.get("surface_index", -1))
        if surface_index < 0 or surface_index >= len(primitive.surfaces) or surface_index in selection:
            raise ValueError("Imported walkmesh generation intent contains a stale or duplicate surface index.")
        surface = primitive.surfaces[surface_index]
        raw_faces = row.get("face_indices")
        if raw_faces is None:
            selection[surface_index] = None
            continue
        if not isinstance(raw_faces, list) or not raw_faces:
            raise ValueError("Imported walkmesh generation intent contains an empty face selection.")
        face_indices: set[int] = set()
        for raw_face_index in raw_faces:
            if isinstance(raw_face_index, bool):
                raise ValueError("Imported walkmesh generation intent contains a boolean face index.")
            face_index = int(raw_face_index)
            if face_index < 0 or face_index >= len(surface.faces):
                raise ValueError("Imported walkmesh generation intent contains a stale face index.")
            face_indices.add(face_index)
        selection[surface_index] = frozenset(face_indices)
    return selection


@dataclass(frozen=True)
class ImportedMeshBevelOptions:
    """Persistent operator state for a Maya-style imported-mesh edge bevel."""

    width: float = 0.25
    segments: int = 1
    profile: float = 0.5
    miter: str = "auto"
    smoothing_angle_degrees: float = 180.0
    uv_mode: str = "preserve"
    clamp_overlap: bool = True


def imported_mesh_surface_role(index: int) -> str:
    return RENDER_SURFACE_ROLE if index == 0 else f"imported_srf_{index}"


# Texture/name hints that mark an individual skybox / far-backdrop surface.
# K2 often uses ``*_sb01..05`` rather than spelling out "sky"; star fields
# and lightning layers can also live inside otherwise normal playable rooms.
_BACKDROP_TEXTURE_HINTS = ("sky", "cloud", "backdrop", "horizon", "stars", "_sb0")
_BACKDROP_NODE_HINTS = ("sky", "backdrop", "horizon", "star_field", "space", "lightning")
# A gameplay room rarely spans more than a couple hundred units; a backdrop
# dome spans thousands (RNVcity koq201_01j ~2100 units).
_BACKDROP_EXTENT_THRESHOLD = 300.0


def imported_mesh_surface_is_backdrop(
    surface: ImportedMeshSurface,
    *,
    extent_threshold: float = _BACKDROP_EXTENT_THRESHOLD,
) -> bool:
    """Return True when one stable imported surface is visual far backdrop.

    Classification is deliberately surface-scoped. Vanilla K2 rooms such as
    ``151harsb`` and ``231telsb`` mix giant sky/star layers with ordinary
    lightmapped or walkable geometry, so a room-name/whole-bounds heuristic
    would hide real level content and remove its WOK from pathing.
    """

    if bool(getattr(surface, "backdrop", False)):
        return True
    vertices = tuple(getattr(surface, "vertices", ()) or ())
    if not vertices:
        return False
    texture = str(getattr(surface, "texture", "") or "").strip().lower()
    name = str(getattr(surface, "name", "") or "").strip().lower()
    hinted = any(hint in texture for hint in _BACKDROP_TEXTURE_HINTS) or any(
        hint in name for hint in _BACKDROP_NODE_HINTS
    )
    if not hinted:
        return False
    spans = tuple(
        max(float(vertex[axis]) for vertex in vertices) - min(float(vertex[axis]) for vertex in vertices)
        for axis in range(3)
    )
    return max(spans, default=0.0) > float(extent_threshold)


def imported_mesh_room_is_backdrop(
    primitive: ImportedMeshRoomPrimitive,
    *,
    extent_threshold: float = _BACKDROP_EXTENT_THRESHOLD,
) -> bool:
    """Return True only for a backdrop-only room with no playable WOK.

    This is intentionally stricter than surface classification. Mixed rooms
    retain their gameplay WOK/pathing even when one or more render surfaces are
    hidden by the editor's Skybox visibility toggle.
    """

    surfaces = tuple(getattr(primitive, "surfaces", ()) or ())
    if not surfaces:
        return False
    if not all(imported_mesh_surface_is_backdrop(surface, extent_threshold=extent_threshold) for surface in surfaces):
        return False
    wok = getattr(primitive, "wok", None)
    return not any(
        is_walkable_walkmesh_surface(int(getattr(face, "surface", -1)))
        for face in tuple(getattr(wok, "faces", ()) or ())
    )


def imported_mesh_room_is_visual_only(primitive: ImportedMeshRoomPrimitive) -> bool:
    """Return True when an imported room intentionally has no collision.

    A present, zero-face WOK is source data, not a missing walkmesh. Retail
    modules use that exact 136-byte BWM form for sky/backdrop partitions, and
    legacy modules also use it for ordinary-looking visual partitions whose
    textures/names carry no sky hint. Treating it as missing made compile fall
    through to the bounds-derived two-face NON_WALK placeholder, so a KMAP
    round-trip silently invented collision that was absent from the MOD.

    The distinction is deliberately presence-based: ``wok is None`` still
    means collision is unresolved and may require authored generation, while
    ``wok is not None and not wok.faces`` is an explicit visual-only contract.
    """

    wok = getattr(primitive, "wok", None)
    return wok is not None and not tuple(getattr(wok, "faces", ()) or ())


def imported_mesh_surface_index_for_role(primitive: ImportedMeshRoomPrimitive, role: str) -> int:
    """Resolve a viewport mesh role back to a surface index (-1 if unknown)."""

    wanted = str(role or "").strip()
    # Rooms baked from composition primitives arrive with preview roles
    # ("helper_<n>"); surface order matches, so alias them.
    if wanted.startswith("helper_") and wanted[7:].isdigit():
        wanted = imported_mesh_surface_role(int(wanted[7:]))
    for index in range(len(primitive.surfaces)):
        if imported_mesh_surface_role(index) == wanted:
            return index
    return -1


# ---------------------------------------------------------------------------
# Validation + compile


#: MDL mesh nodes index vertices with 16-bit values and swkotor2.exe has signed consumers: past 32,767 exported
#: vertices a render node is a crash hazard although the file could hold 65,535 (Dathomir TE01, 2026-11-17).
#: imported_mesh_partition.split_oversized_imported_mesh_surfaces does the split.
MDL_MAX_VERTICES_PER_SURFACE = 32767
#: Vanilla KOTOR rooms run low thousands of triangles; warn well before the
#: engine starts struggling with area geometry.
ROOM_TRIANGLE_WARNING_BUDGET = 15000


def validate_imported_mesh_room_primitive(primitive: ImportedMeshRoomPrimitive) -> ImportedMeshValidation:
    cache_key = id(primitive)
    cached = _IMPORTED_MESH_VALIDATION_CACHE.get(cache_key)
    if cached is not None and cached[0]() is primitive:
        _IMPORTED_MESH_VALIDATION_CACHE.move_to_end(cache_key)
        return cached[1]
    if cached is not None:
        # The previous weak target died and CPython reused its id.
        _IMPORTED_MESH_VALIDATION_CACHE.pop(cache_key, None)

    warnings: list[str] = []
    blocking: list[str] = []
    if not primitive.surfaces:
        blocking.append(f"Imported room {primitive.room_resref} has no surfaces.")
    total_triangles = sum(len(surface.faces) for surface in primitive.surfaces)
    if total_triangles > ROOM_TRIANGLE_WARNING_BUDGET:
        warnings.append(
            f"Imported room {primitive.room_resref} has {total_triangles} triangles;"
            f" vanilla KOTOR rooms stay in the low thousands — consider splitting or decimating."
        )
    for index, surface in enumerate(primitive.surfaces):
        role = imported_mesh_surface_role(index)
        if not surface.vertices or not surface.faces:
            blocking.append(f"Imported room {primitive.room_resref} surface {role} is empty.")
            continue
        vertex_count = len(surface.vertices)
        if vertex_count > MDL_MAX_VERTICES_PER_SURFACE:
            blocking.append(
                f"Imported room {primitive.room_resref} surface {role} has {vertex_count} vertices,"
                f" past the MDL u16 index limit ({MDL_MAX_VERTICES_PER_SURFACE}); split the surface (imported_mesh_partition)."
            )
        for face in surface.faces:
            if min(face) < 0 or max(face) >= vertex_count:
                blocking.append(
                    f"Imported room {primitive.room_resref} surface {role} has out-of-range face indices."
                )
                break
        topology_audit = MeshTopology.build(surface.vertices, surface.faces).validate_manifold_state()
        if topology_audit.degenerate_faces:
            warnings.append(
                f"Imported room {primitive.room_resref} surface {role} has "
                f"{len(topology_audit.degenerate_faces)} degenerate face(s)."
            )
        if topology_audit.non_manifold_edges:
            warnings.append(
                f"Imported room {primitive.room_resref} surface {role} has "
                f"{len(topology_audit.non_manifold_edges)} non-manifold edge(s)."
            )
        if topology_audit.duplicate_faces:
            warnings.append(
                f"Imported room {primitive.room_resref} surface {role} has "
                f"{len(topology_audit.duplicate_faces)} overlapping duplicate face(s)."
            )
        if topology_audit.inconsistent_winding_edges:
            warnings.append(
                f"Imported room {primitive.room_resref} surface {role} has "
                f"{len(topology_audit.inconsistent_winding_edges)} inconsistent winding edge(s)."
            )
        if len(surface.uvs) not in (0, vertex_count):
            warnings.append(f"Imported room {primitive.room_resref} surface {role} UV count mismatch; UVs regenerate on compile.")
        if len(surface.uvs_lm) not in (0, vertex_count):
            blocking.append(
                f"Imported room {primitive.room_resref} surface {role} lightmap UV count does not match its vertices."
            )
        if not surface.texture:
            warnings.append(f"Imported room {primitive.room_resref} surface {role} has no texture assigned.")
    if primitive.wok is None:
        warnings.append(
            f"Imported room {primitive.room_resref} has no imported WOK; Map Studio can show a red bounds-derived "
            "preview plane, but it is NON_WALK and export remains blocked until a real floor walkmesh is generated."
        )
    result = ImportedMeshValidation(ok=not blocking, warnings=tuple(warnings), blocking_issues=tuple(blocking))
    _IMPORTED_MESH_VALIDATION_CACHE[cache_key] = (ref(primitive), result)
    _IMPORTED_MESH_VALIDATION_CACHE.move_to_end(cache_key)
    while len(_IMPORTED_MESH_VALIDATION_CACHE) > _IMPORTED_MESH_VALIDATION_CACHE_MAX:
        _IMPORTED_MESH_VALIDATION_CACHE.popitem(last=False)
    return result


def _surface_primitive_mesh(primitive: ImportedMeshRoomPrimitive, index: int) -> PrimitiveMesh:
    surface = primitive.surfaces[index]
    vertex_count = len(surface.vertices)
    uvs = (
        surface.uvs
        if len(surface.uvs) == vertex_count
        else tiled_uvs_for_vertices(
            surface.vertices,
            tile_size=matched_uv_tile_size(primitive, texture=surface.texture),
        )
    )
    normals = surface.normals if len(surface.normals) == vertex_count else ()
    role = imported_mesh_surface_role(index)
    return PrimitiveMesh(
        name=str(surface.name or f"{primitive.room_resref}_{role}"),
        vertices=tuple(surface.vertices),
        faces=tuple(surface.faces),
        normals=tuple(normals),
        uvs=tuple(uvs),
        texture=str(surface.texture or ""),
        diffuse=tuple(surface.diffuse),
        ambient=tuple(surface.ambient),
        metadata={
            "role": role,
            "primitive": IMPORTED_MESH_PRIMITIVE_KIND,
            "is_backdrop": bool(imported_mesh_surface_is_backdrop(surface)),
            "lightmap": str(surface.lightmap or ""),
            "source_model": str(primitive.source_model or ""),
            "texture_names": tuple(surface.texture_names or ()),
            "tex_count": max(1, int(surface.tex_count or 1)),
            "uvs_lm": tuple(surface.uvs_lm or ()) if len(surface.uvs_lm) == vertex_count else (),
            "face_mats": tuple(surface.face_mats or ()) if len(surface.face_mats) == len(surface.faces) else (),
            "specular": tuple(surface.specular),
            "shininess": float(surface.shininess),
            "alpha": float(surface.alpha),
            "has_shadow": bool(surface.has_shadow),
            "render": bool(surface.render),
            "selfillum": tuple(surface.selfillum),
            "transparency_hint": int(surface.transparency_hint),
            "beaming": bool(surface.beaming),
            "background_geometry": bool(surface.background_geometry),
            "rotate_texture": bool(surface.rotate_texture),
            "animate_uv": bool(surface.animate_uv),
            "uv_dir_x": float(surface.uv_dir_x),
            "uv_dir_y": float(surface.uv_dir_y),
            "uv_jitter": float(surface.uv_jitter),
            "uv_jitter_speed": float(surface.uv_jitter_speed),
            "dirt_enabled": bool(surface.dirt_enabled),
            "dirt_texture": int(surface.dirt_texture),
            "dirt_coord_space": int(surface.dirt_coord_space),
            "hide_in_holograms": bool(surface.hide_in_holograms),
            "mesh_average_point": tuple(surface.mesh_average_point),
            "mesh_unknown0": bytes(surface.mesh_unknown0 or b"")[:24].ljust(24, b"\x00"),
        },
    )


def _mesh_bounds(primitive: ImportedMeshRoomPrimitive) -> tuple[Vec3, Vec3]:
    points = [point for surface in primitive.surfaces for point in surface.vertices]
    if not points:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _fallback_floor_wok(primitive: ImportedMeshRoomPrimitive) -> WOKData:
    """Return a red/non-walk preview plane across the render bounds.

    Render bounds cannot reveal real floor topology, pits, stairs, portals, or
    disconnected islands.  Treating this rectangle as authored collision used
    to let rooms with no source WOK pass readiness and export silently.  It is
    now deliberately NON_WALK and carries explicit preview provenance; the
    normal walkmesh audit therefore blocks export until the room receives an
    imported or generated, structurally validated floor WOK.
    """

    (x0, y0, z0), (x1, y1, _z1) = _mesh_bounds(primitive)
    wok = WOKData(
        name=str(primitive.room_resref or ""),
        verts=[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        faces=[
            WOKFace(0, 1, 2, surface=NON_WALK_ID, adj1=-1, adj2=-1, adj3=1),
            WOKFace(0, 2, 3, surface=NON_WALK_ID, adj1=0, adj2=-1, adj3=-1),
        ],
    )
    # WOKData intentionally remains a format-focused dataclass.  These runtime
    # annotations let preview/readiness callers identify the synthetic plane
    # without changing the binary format or KMAP schema.
    wok.preview_only = True
    wok.readiness_blocking = True
    wok.provenance = "render_bounds_preview_only"
    wok.readiness_blocking_reason = (
        f"Imported room {primitive.room_resref} has no authored WOK; its bounds-derived plane is preview-only. "
        "Import or generate and validate a floor walkmesh before export."
    )
    return wok


def _point_key(point: Vec3, *, epsilon: float) -> tuple[int, int, int]:
    scale = 1.0 / max(1.0e-9, float(epsilon))
    return (round(point[0] * scale), round(point[1] * scale), round(point[2] * scale))


def _edge_key(a: Vec3, b: Vec3, *, epsilon: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    left = _point_key(a, epsilon=epsilon)
    right = _point_key(b, epsilon=epsilon)
    return (left, right) if left <= right else (right, left)


def _triangle_overlap_area_xy(first: tuple[Vec3, Vec3, Vec3], second: tuple[Vec3, Vec3, Vec3]) -> float:
    """Return the positive XY intersection area of two triangles."""

    subject = [(float(point[0]), float(point[1])) for point in first]
    clip = [(float(point[0]), float(point[1])) for point in second]

    def cross(a: Vec2, b: Vec2, c: Vec2) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))

    orientation = cross(clip[0], clip[1], clip[2])
    if abs(orientation) <= 1.0e-12:
        return 0.0
    sign = 1.0 if orientation > 0.0 else -1.0

    def inside(point: Vec2, edge_a: Vec2, edge_b: Vec2) -> bool:
        return sign * cross(edge_a, edge_b, point) >= -1.0e-12

    def intersection(start: Vec2, end: Vec2, edge_a: Vec2, edge_b: Vec2) -> Vec2:
        segment = (end[0] - start[0], end[1] - start[1])
        clip_line = (edge_b[0] - edge_a[0], edge_b[1] - edge_a[1])
        denominator = (segment[0] * clip_line[1]) - (segment[1] * clip_line[0])
        if abs(denominator) <= 1.0e-12:
            return end
        offset = (edge_a[0] - start[0], edge_a[1] - start[1])
        fraction = ((offset[0] * clip_line[1]) - (offset[1] * clip_line[0])) / denominator
        fraction = min(1.0, max(0.0, fraction))
        return (start[0] + segment[0] * fraction, start[1] + segment[1] * fraction)

    output = subject
    for edge_index in range(3):
        edge_a = clip[edge_index]
        edge_b = clip[(edge_index + 1) % 3]
        source = output
        output = []
        if not source:
            break
        start = source[-1]
        for end in source:
            end_inside = inside(end, edge_a, edge_b)
            start_inside = inside(start, edge_a, edge_b)
            if end_inside:
                if not start_inside:
                    output.append(intersection(start, end, edge_a, edge_b))
                output.append(end)
            elif start_inside:
                output.append(intersection(start, end, edge_a, edge_b))
            start = end
    if len(output) < 3:
        return 0.0
    twice_area = sum(
        output[index][0] * output[(index + 1) % len(output)][1]
        - output[(index + 1) % len(output)][0] * output[index][1]
        for index in range(len(output))
    )
    return abs(twice_area) * 0.5


def generate_room_walkmesh_from_geometry(
    primitive: ImportedMeshRoomPrimitive,
    *,
    slope_max_degrees: float = 45.0,
    weld_epsilon: float = 1.0e-4,
    source_wok_to_render_offset: Vec3 = (0.0, 0.0, 0.0),
    source_material_z_tolerance: float = 1.5,
    disconnected_island_policy: str = "reject",
    source_wok_policy: str = "preserve",
) -> tuple[ImportedMeshRoomPrimitive, dict[str, Any]]:
    """Derive a complete room walkmesh from the room's render geometry.

    Built from the complete installed K1/K2 vanilla WOK census: walkable faces
    are overwhelmingly near-horizontal, while blindly copying render walls or
    a down-facing ceiling into a generated WOK can enclose the floor and leave
    the engine without a usable perimeter.  Render geometry is therefore only
    a *floor candidate* source; it is not copied wholesale into collision.
    Every non-backdrop render triangle is classified by its normal:

    * up-facing, within ``slope_max_degrees`` of flat -> WALKABLE floor,
    * steeper than that (a wall or unsafe ramp) -> omitted from the WOK,
    * down-facing and near-horizontal (a ceiling) -> dropped.

    Up-facing faces covered by an imported source WOK retain their original
    surface material, so explicit horizontal blockers survive.  The result is
    a fresh room-local, floor-only WOK with welded vertices and rebuilt
    adjacency, replacing whatever WOK the room had.  New generation rejects
    disconnected walkable islands by default: islands are valid in studied
    retail rooms, but inventing that intent from arbitrary render geometry is
    unsafe.  Recovery tooling may pass ``disconnected_island_policy="preserve"``
    only when the separate islands have been reviewed deliberately.

    An imported source WOK is preserved by default.  Render triangles cannot
    reveal whether an uncovered region is a deliberate pit/hole, table top, or
    roof, so destructive replacement requires ``source_wok_policy="replace"``.
    The separate floor-fill operation is the safe, source-seeded way to extend
    a partial imported WOK.

    Before replacement, the generated WOK is serialized and read back through
    Ghost Studio's engine writer.  This proves that complete AABB, adjacency,
    boundary-edge, and perimeter-loop tables can actually be emitted; a parser-
    only success is not accepted as generation proof.
    """

    import math as _math

    island_policy = str(disconnected_island_policy or "reject").strip().lower()
    if island_policy not in {"reject", "preserve"}:
        raise ValueError(
            "Disconnected walkmesh island policy must be 'reject' or 'preserve'."
        )
    source_policy = str(source_wok_policy or "preserve").strip().lower()
    if source_policy not in {"preserve", "replace"}:
        raise ValueError("Source WOK policy must be 'preserve' or 'replace'.")

    if source_policy == "preserve" and primitive.wok is not None and primitive.wok.faces:
        source_topology = audit_authored_wok(str(primitive.room_resref or ""), primitive.wok)
        report: dict[str, Any] = {
            "floor_faces": 0,
            "walkable_floor_faces": int(source_topology.walkable_face_count),
            "wall_faces": 0,
            "dropped_wall_faces": 0,
            "dropped_ceiling_faces": 0,
            "degenerate_faces": 0,
            "non_finite_faces": 0,
            "weld_collapsed_faces": 0,
            "total_faces": len(primitive.wok.faces),
            "projected_material_faces": 0,
            "source_transition_edges": sum(
                int(value) >= 0
                for face in primitive.wok.faces
                for value in (face.trans1, face.trans2, face.trans3)
            ),
            "mapped_transition_edges": 0,
            "unmapped_transition_edges": 0,
            "disconnected_island_policy": island_policy,
            "source_wok_policy": source_policy,
            "source_wok_preserved": True,
            "weld_policy_version": 2,
            "weld_epsilon": max(1.0e-9, float(weld_epsilon)),
            "walkable_component_count": int(source_topology.walkable_component_count),
            "disconnected_component_count": int(source_topology.disconnected_component_count),
            "non_manifold_edge_count": int(source_topology.non_manifold_edge_count),
            "invalid_face_count": int(source_topology.invalid_face_count),
            "generated_degenerate_face_count": 0,
            "density_warning_threshold": EMPIRICAL_STOCK_WOK_FACE_MAX,
            "density_warning": "",
        }
        try:
            raw_source = primitive.wok.to_bytes()
            if len(raw_source) < 136 or raw_source[:8] != b"BWM V1.0":
                raise ValueError("source WOK has no complete BWM V1.0 header")
            raw_counts = struct.unpack_from("<16I", raw_source, 72)
            raw_vertex_count = int(raw_counts[0])
            raw_face_count = int(raw_counts[2])
            raw_aabb_count = int(raw_counts[7])
            raw_aabb_root = int(raw_counts[9])
            raw_adjacency_count = int(raw_counts[10])
            raw_boundary_edge_count = int(raw_counts[12])
            raw_perimeter_count = int(raw_counts[14])
            if raw_face_count <= 0 or raw_vertex_count <= 0:
                raise ValueError("source WOK has no collision geometry")
            if raw_aabb_count <= 0 or raw_aabb_root >= raw_aabb_count:
                raise ValueError("source WOK has no valid AABB root")
            if raw_perimeter_count < max(1, source_topology.walkable_component_count):
                raise ValueError("source WOK has no closed perimeter for every walkable component")
            if raw_boundary_edge_count < raw_perimeter_count:
                raise ValueError("source WOK perimeter edge table is incomplete")
            report.update(
                {
                    "structural_validation": "passed",
                    "serialized_vertex_count": raw_vertex_count,
                    "serialized_face_count": raw_face_count,
                    "serialized_aabb_count": raw_aabb_count,
                    "serialized_adjacency_face_count": raw_adjacency_count,
                    "serialized_boundary_edge_count": raw_boundary_edge_count,
                    "serialized_perimeter_loop_count": raw_perimeter_count,
                    "interior_boundary_loop_count": max(
                        0,
                        raw_perimeter_count - int(source_topology.walkable_component_count),
                    ),
                    "serialized_readback_face_count": len(WOKData.from_bytes(raw_source).faces),
                }
            )
        except (IndexError, TypeError, ValueError, struct.error) as exc:
            report["structural_validation"] = "blocked"
            report["blocked_reason"] = (
                f"Auto Generate Walkmesh preserved {primitive.room_resref}'s imported WOK but found an engine-structure "
                f"problem ({exc}). Repair the source topology or choose an explicit reviewed replacement."
            )
        return primitive, report

    try:
        reviewed_selection = _imported_walkmesh_generation_selection(primitive)
    except (TypeError, ValueError) as exc:
        return primitive, {
            "floor_faces": 0,
            "walkable_floor_faces": 0,
            "wall_faces": 0,
            "dropped_wall_faces": 0,
            "dropped_ceiling_faces": 0,
            "degenerate_faces": 0,
            "non_finite_faces": 0,
            "weld_collapsed_faces": 0,
            "total_faces": 0,
            "source_wok_policy": source_policy,
            "source_wok_preserved": False,
            "structural_validation": "blocked",
            "blocked_reason": f"Imported walkmesh floor selection is invalid: {exc}",
        }
    imported_from = str(dict(primitive.metadata or {}).get("imported_from") or "").strip()
    if imported_from and reviewed_selection is None:
        return primitive, {
            "floor_faces": 0,
            "walkable_floor_faces": 0,
            "wall_faces": 0,
            "dropped_wall_faces": 0,
            "dropped_ceiling_faces": 0,
            "degenerate_faces": 0,
            "non_finite_faces": 0,
            "weld_collapsed_faces": 0,
            "total_faces": 0,
            "source_wok_policy": source_policy,
            "source_wok_preserved": False,
            "structural_validation": "blocked",
            "blocked_reason": (
                f"Auto Generate Walkmesh refused imported room {primitive.room_resref}: upward-facing render "
                "triangles may be roofs, tables, or decorative ledges. Review floor surfaces/faces and record "
                "walkmesh_generation_intent before generating collision."
            ),
        }

    min_floor_normal_z = _math.cos(_math.radians(max(1.0, min(89.0, float(slope_max_degrees)))))
    source_offset = tuple(float(value) for value in tuple(source_wok_to_render_offset or (0.0, 0.0, 0.0))[:3])
    source_samples: list[tuple[Vec3, Vec3, Vec3, int]] = []
    source_transitions: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
    transition_conflicts = 0
    if primitive.wok is not None:
        source_verts = tuple(
            (
                float(vertex[0]) + source_offset[0],
                float(vertex[1]) + source_offset[1],
                float(vertex[2]) + source_offset[2],
            )
            for vertex in tuple(primitive.wok.verts or ())
        )
        for source_face in tuple(primitive.wok.faces or ()):
            indices = (int(source_face.v1), int(source_face.v2), int(source_face.v3))
            if any(index < 0 or index >= len(source_verts) for index in indices):
                continue
            corners = tuple(source_verts[index] for index in indices)
            source_samples.append((corners[0], corners[1], corners[2], int(source_face.surface)))
            for edge_index, transition in enumerate(
                (int(getattr(source_face, "trans1", -1)), int(getattr(source_face, "trans2", -1)), int(getattr(source_face, "trans3", -1)))
            ):
                if transition < 0:
                    continue
                key = _edge_key(corners[edge_index], corners[(edge_index + 1) % 3], epsilon=weld_epsilon)
                previous = source_transitions.get(key)
                if previous is not None and previous != transition:
                    transition_conflicts += 1
                source_transitions[key] = transition

    def _project_source_material(a: Vec3, b: Vec3, c: Vec3) -> tuple[int, bool]:
        cx = (a[0] + b[0] + c[0]) / 3.0
        cy = (a[1] + b[1] + c[1]) / 3.0
        cz = (a[2] + b[2] + c[2]) / 3.0
        best: tuple[float, int] | None = None
        for source_a, source_b, source_c, material in source_samples:
            inside, source_z = _triangle_contains_xy(cx, cy, source_a, source_b, source_c)
            if not inside:
                continue
            delta = abs(source_z - cz)
            if delta <= float(source_material_z_tolerance) and (best is None or delta < best[0]):
                best = (delta, material)
        return (best[1], True) if best is not None else (4, False)

    verts: list[Vec3] = []
    faces: list[WOKFace] = []
    weld_distance = max(1.0e-9, float(weld_epsilon))
    weld_distance_squared = weld_distance * weld_distance
    spatial_buckets: dict[tuple[int, int, int], list[int]] = {}

    def _weld_key(point: Vec3) -> tuple[int, int, int]:
        return (
            _math.floor(float(point[0]) / weld_distance),
            _math.floor(float(point[1]) / weld_distance),
            _math.floor(float(point[2]) / weld_distance),
        )

    def _find_welded_index(point: Vec3) -> int | None:
        bucket = _weld_key(point)
        matches: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in spatial_buckets.get((bucket[0] + dx, bucket[1] + dy, bucket[2] + dz), ()):
                        existing = verts[candidate]
                        distance_squared = sum(
                            (float(existing[axis]) - float(point[axis])) ** 2
                            for axis in range(3)
                        )
                        if distance_squared <= weld_distance_squared:
                            matches.append(candidate)
        return min(matches) if matches else None

    def _index(point: Vec3) -> int:
        existing = _find_welded_index(point)
        if existing is not None:
            return existing
        verts.append((float(point[0]), float(point[1]), float(point[2])))
        index = len(verts) - 1
        spatial_buckets.setdefault(_weld_key(point), []).append(index)
        return index

    floor_faces = walkable_floor_faces = wall_faces = dropped_ceiling = degenerate = 0
    non_finite_faces = 0
    weld_collapsed_faces = 0
    projected_material_faces = 0
    selected_surface_names: list[str] = []
    selected_face_intent_count = 0
    for surface_index, surface in enumerate(primitive.surfaces):
        if reviewed_selection is not None and surface_index not in reviewed_selection:
            continue
        reviewed_faces = None if reviewed_selection is None else reviewed_selection[surface_index]
        selected_surface_names.append(str(surface.name or f"surface_{surface_index}"))
        selected_face_intent_count += len(surface.faces) if reviewed_faces is None else len(reviewed_faces)
        if bool(getattr(surface, "backdrop", False)) or bool(getattr(surface, "background_geometry", False)):
            continue
        if not bool(getattr(surface, "render", True)):
            continue
        surface_verts = tuple(surface.vertices)
        for face_index, face in enumerate(tuple(surface.faces)):
            if reviewed_faces is not None and face_index not in reviewed_faces:
                continue
            try:
                a = surface_verts[int(face[0])]
                b = surface_verts[int(face[1])]
                c = surface_verts[int(face[2])]
            except (IndexError, TypeError, ValueError):
                continue
            if not all(_math.isfinite(float(value)) for corner in (a, b, c) for value in corner[:3]):
                non_finite_faces += 1
                continue
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = (uy * vz) - (uz * vy), (uz * vx) - (ux * vz), (ux * vy) - (uy * vx)
            length = _math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
            if length < 1.0e-9:
                degenerate += 1
                continue
            unit_nz = nz / length
            if unit_nz >= min_floor_normal_z:
                # Up-facing floor: keep upward winding so engine floor tests pass.
                corners = (a, b, c)
                material, projected = _project_source_material(a, b, c)
            elif unit_nz <= -min_floor_normal_z:
                dropped_ceiling += 1
                continue
            else:
                wall_faces += 1
                # A render wall is not a walkmesh wall.  Omitting it leaves the
                # walkable floor's boundary explicit; copying it as NON_WALK
                # can seal the floor and freeze Odyssey movement.
                continue
            if any(
                sum((float(corners[left][axis]) - float(corners[right][axis])) ** 2 for axis in range(3))
                <= weld_distance_squared
                for left, right in ((0, 1), (1, 2), (2, 0))
            ):
                # The source triangle can have positive area yet collapse when
                # two near-coincident corners enter the same weld bucket.
                # Never emit a repeated-index floor face into Odyssey.
                degenerate += 1
                weld_collapsed_faces += 1
                continue
            indices = tuple(_index(corner) for corner in corners)
            projected_material_faces += int(projected)
            floor_faces += 1
            walkable_floor_faces += int(is_walkable_walkmesh_surface(material))
            faces.append(
                WOKFace(
                    indices[0],
                    indices[1],
                    indices[2],
                    material,
                    -1,
                    -1,
                    -1,
                )
            )

    report: dict[str, Any] = {
        "floor_faces": floor_faces,
        "walkable_floor_faces": walkable_floor_faces,
        "wall_faces": wall_faces,
        "dropped_wall_faces": wall_faces,
        "dropped_ceiling_faces": dropped_ceiling,
        "degenerate_faces": degenerate,
        "non_finite_faces": non_finite_faces,
        "weld_collapsed_faces": weld_collapsed_faces,
        "total_faces": len(faces),
        "projected_material_faces": projected_material_faces,
        "source_transition_edges": len(source_transitions),
        "mapped_transition_edges": 0,
        "unmapped_transition_edges": 0,
        "disconnected_island_policy": island_policy,
        "source_wok_policy": source_policy,
        "source_wok_preserved": False,
        "walkmesh_generation_intent_policy": (
            IMPORTED_WALKMESH_GENERATION_INTENT_POLICY if reviewed_selection is not None else "authored_geometry"
        ),
        "reviewed_surface_count": len(selected_surface_names),
        "reviewed_surface_names": selected_surface_names,
        "reviewed_face_intent_count": selected_face_intent_count,
        "weld_policy_version": 2,
        "weld_epsilon": weld_distance,
        "density_warning_threshold": EMPIRICAL_STOCK_WOK_FACE_MAX,
        "density_warning": (
            f"Generated render-resolution WOK has {len(faces)} faces, above the empirical stock-room maximum "
            f"of {EMPIRICAL_STOCK_WOK_FACE_MAX}; no decimation was applied."
            if len(faces) > EMPIRICAL_STOCK_WOK_FACE_MAX
            else ""
        ),
    }
    if walkable_floor_faces <= 0:
        # No walkable floor could be derived; leave the room's WOK untouched.
        return primitive, report

    source_wok = primitive.wok
    wok = WOKData(
        name=str(primitive.room_resref or ""),
        verts=verts,
        faces=faces,
        relative_hook1=tuple(getattr(source_wok, "relative_hook1", (0.0, 0.0, 0.0))),
        relative_hook2=tuple(getattr(source_wok, "relative_hook2", (0.0, 0.0, 0.0))),
        absolute_hook1=tuple(getattr(source_wok, "absolute_hook1", (0.0, 0.0, 0.0))),
        absolute_hook2=tuple(getattr(source_wok, "absolute_hook2", (0.0, 0.0, 0.0))),
        position=tuple(getattr(source_wok, "position", (0.0, 0.0, 0.0))),
    )
    wok.rebuild_adjacencies()
    generated_boundaries: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[tuple[int, int]],
    ] = {}
    for vertex_a, vertex_b, face_index, edge_index in wok.boundary_edges():
        key = _edge_key(wok.verts[vertex_a], wok.verts[vertex_b], epsilon=weld_epsilon)
        generated_boundaries.setdefault(key, []).append((face_index, edge_index))
    transition_assignments: list[tuple[int, int, int]] = []
    unmapped = transition_conflicts
    for key, transition in source_transitions.items():
        matches = generated_boundaries.get(key, ())
        if len(matches) != 1:
            unmapped += 1
            continue
        face_index, edge_index = matches[0]
        transition_assignments.append((face_index, edge_index, transition))
    report["mapped_transition_edges"] = len(transition_assignments)
    report["unmapped_transition_edges"] = unmapped
    if unmapped:
        report["blocked_reason"] = (
            f"Auto Generate Walkmesh refused to replace {primitive.room_resref}: {unmapped} source perimeter "
            "transition edge(s) could not be mapped uniquely to the generated boundary."
        )
        return primitive, report
    for face_index, edge_index, transition in transition_assignments:
        setattr(wok.faces[face_index], f"trans{edge_index + 1}", transition)

    topology = audit_authored_wok(str(primitive.room_resref or ""), wok)
    report.update(
        {
            "walkable_component_count": int(topology.walkable_component_count),
            "disconnected_component_count": int(topology.disconnected_component_count),
            "non_manifold_edge_count": int(topology.non_manifold_edge_count),
            "invalid_face_count": int(topology.invalid_face_count),
            "generated_degenerate_face_count": int(topology.degenerate_face_count),
        }
    )
    strict_faults: list[str] = []
    if topology.invalid_face_count:
        strict_faults.append(f"{topology.invalid_face_count} face(s) reference invalid vertices")
    if topology.degenerate_face_count:
        strict_faults.append(f"{topology.degenerate_face_count} generated face(s) are degenerate")
    if topology.non_manifold_edge_count:
        strict_faults.append(f"{topology.non_manifold_edge_count} walkable edge(s) are non-manifold")
    if topology.walkable_component_count > 1 and island_policy == "reject":
        strict_faults.append(
            f"{topology.walkable_component_count} disconnected walkable islands require explicit review"
        )
    if strict_faults:
        report["blocked_reason"] = (
            f"Auto Generate Walkmesh refused to replace {primitive.room_resref}: "
            + "; ".join(strict_faults)
            + "."
        )
        return primitive, report

    try:
        raw_wok = wok.to_bytes()
        (
            raw_vertex_count,
            _raw_vertex_offset,
            raw_face_count,
            _raw_face_offset,
            _raw_material_offset,
            _raw_normal_offset,
            _raw_plane_offset,
            raw_aabb_count,
            _raw_aabb_offset,
            raw_aabb_root,
            raw_adjacency_count,
            _raw_adjacency_offset,
            raw_boundary_edge_count,
            _raw_boundary_edge_offset,
            raw_perimeter_count,
            _raw_perimeter_offset,
        ) = struct.unpack_from("<16I", raw_wok, 72)
        reopened = WOKData.from_bytes(raw_wok)
        expected_aabb_count = (2 * raw_face_count) - 1 if raw_face_count else 0
        structural_faults: list[str] = []
        if raw_vertex_count != len(wok.verts) or raw_face_count != len(wok.faces):
            structural_faults.append("serialized vertex/face counts changed")
        if len(reopened.faces) != raw_face_count:
            structural_faults.append("serialized WOK did not read back with every face")
        if raw_aabb_count != expected_aabb_count or (raw_aabb_count and raw_aabb_root >= raw_aabb_count):
            structural_faults.append("serialized AABB tree is incomplete")
        if raw_adjacency_count != topology.walkable_face_count:
            structural_faults.append("serialized adjacency domain does not match walkable faces")
        if raw_perimeter_count < max(1, topology.walkable_component_count):
            structural_faults.append("serialized perimeter loops do not cover every walkable component")
        if raw_boundary_edge_count < raw_perimeter_count:
            structural_faults.append("serialized perimeter edge table is incomplete")
        if structural_faults:
            raise ValueError("; ".join(structural_faults))
    except (IndexError, TypeError, ValueError, struct.error) as exc:
        report["structural_validation"] = "blocked"
        report["blocked_reason"] = (
            f"Auto Generate Walkmesh refused to replace {primitive.room_resref}: "
            f"engine-structure serialization failed ({exc})."
        )
        return primitive, report

    report.update(
        {
            "structural_validation": "passed",
            "serialized_vertex_count": int(raw_vertex_count),
            "serialized_face_count": int(raw_face_count),
            "serialized_aabb_count": int(raw_aabb_count),
            "serialized_adjacency_face_count": int(raw_adjacency_count),
            "serialized_boundary_edge_count": int(raw_boundary_edge_count),
            "serialized_perimeter_loop_count": int(raw_perimeter_count),
            "interior_boundary_loop_count": max(
                0,
                int(raw_perimeter_count) - int(topology.walkable_component_count),
            ),
            "serialized_readback_face_count": len(reopened.faces),
        }
    )
    from dataclasses import replace as _replace

    metadata = dict(primitive.metadata or {})
    metadata["wok_auto_generated"] = report
    # The generated WOK is room-local (built from room-local render surfaces);
    # correct the coordinate-space contract so the combiner applies the room
    # position rather than treating it as module-space.
    metadata["wok_coordinate_space"] = "room_local"
    return _replace(primitive, wok=wok, metadata=metadata), report


def _triangle_contains_xy(px: float, py: float, a: Vec3, b: Vec3, c: Vec3) -> tuple[bool, float]:
    """Return (inside, interpolated_z) for one XY point against one triangle."""

    denominator = ((b[1] - c[1]) * (a[0] - c[0])) + ((c[0] - b[0]) * (a[1] - c[1]))
    if abs(denominator) < 1.0e-12:
        return False, 0.0
    wa = (((b[1] - c[1]) * (px - c[0])) + ((c[0] - b[0]) * (py - c[1]))) / denominator
    wb = (((c[1] - a[1]) * (px - c[0])) + ((a[0] - c[0]) * (py - c[1]))) / denominator
    wc = 1.0 - wa - wb
    if min(wa, wb, wc) < -1.0e-6:
        return False, 0.0
    return True, (a[2] * wa) + (b[2] * wb) + (c[2] * wc)


def fill_imported_wok_from_floor_surfaces(
    primitive: ImportedMeshRoomPrimitive,
    *,
    slope_max_degrees: float = 35.0,
    z_tolerance: float = 1.5,
    weld_epsilon: float = 1.0e-4,
    render_to_wok_offset: Vec3 = (0.0, 0.0, 0.0),
) -> tuple[ImportedMeshRoomPrimitive, dict[str, Any]]:
    """Append walkable WOK faces under visible floor faces that lack coverage.

    Converted candidate modules can ship a room whose WOK covers only part of
    the rendered floor (921srt's custom throne room: the corridor to the next
    room has visible floor but no walkmesh, so the player stops at an
    invisible cliff).  Every near-horizontal, non-backdrop render triangle
    with no positive-area WOK overlap may become a new walkable face. A
    candidate must share one complete welded edge with the existing walkable
    topology (or another accepted candidate), preventing disconnected islands
    and centroid-only partial-overlap patches. Down-facing ceilings are never
    flipped into floors. Areas covered by NON_WALK faces remain intentional
    blockers.

    ``render_to_wok_offset`` maps room-local render coordinates into the
    WOK's own frame: (0, 0, 0) for a room-local WOK, the room's world
    position for a module-space WOK (render surfaces are always room-local).
    """

    import math as _math

    wok = primitive.wok
    if wok is None or not wok.verts:
        raise ValueError(
            f"Room {primitive.room_resref} has no imported WOK to patch; "
            "convert the room with its stock walkmesh first."
        )
    frame_offset = tuple(float(v) for v in tuple(render_to_wok_offset or (0.0, 0.0, 0.0))[:3])
    min_normal_z = _math.cos(_math.radians(max(1.0, min(89.0, float(slope_max_degrees)))))

    coverage_faces: list[tuple[Vec3, Vec3, Vec3]] = []
    walkable_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for face in wok.faces:
        try:
            corners = (wok.verts[face.v1], wok.verts[face.v2], wok.verts[face.v3])
        except IndexError:
            continue
        coverage_faces.append(corners)
        if is_walkable_walkmesh_surface(face.surface):
            for edge_index in range(3):
                walkable_edges.add(
                    _edge_key(corners[edge_index], corners[(edge_index + 1) % 3], epsilon=weld_epsilon)
                )

    def covered(px: float, py: float, pz: float) -> bool:
        for a, b, c in coverage_faces:
            inside, z_at = _triangle_contains_xy(px, py, a, b, c)
            if inside and abs(z_at - pz) <= float(z_tolerance):
                return True
        return False

    def overlaps(triangle: tuple[Vec3, Vec3, Vec3]) -> bool:
        triangle_min_z = min(point[2] for point in triangle)
        triangle_max_z = max(point[2] for point in triangle)
        for existing in coverage_faces:
            existing_min_z = min(point[2] for point in existing)
            existing_max_z = max(point[2] for point in existing)
            if triangle_min_z > existing_max_z + float(z_tolerance):
                continue
            if existing_min_z > triangle_max_z + float(z_tolerance):
                continue
            if _triangle_overlap_area_xy(triangle, existing) > 1.0e-8:
                return True
        return False

    new_verts: list[Vec3] = list(wok.verts)
    weld_distance = max(1.0e-9, float(weld_epsilon))
    weld_distance_squared = weld_distance * weld_distance
    vertex_buckets: dict[tuple[int, int, int], list[int]] = {}

    def _key(point: Vec3) -> tuple[int, int, int]:
        return (
            _math.floor(float(point[0]) / weld_distance),
            _math.floor(float(point[1]) / weld_distance),
            _math.floor(float(point[2]) / weld_distance),
        )

    for index, vertex in enumerate(new_verts):
        vertex_buckets.setdefault(_key(vertex), []).append(index)

    def _existing_vertex_index(point: Vec3) -> int | None:
        bucket = _key(point)
        matches: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in vertex_buckets.get((bucket[0] + dx, bucket[1] + dy, bucket[2] + dz), ()):
                        existing = new_verts[candidate]
                        if sum(
                            (float(existing[axis]) - float(point[axis])) ** 2
                            for axis in range(3)
                        ) <= weld_distance_squared:
                            matches.append(candidate)
        return min(matches) if matches else None

    def _vertex_index(point: Vec3) -> int:
        existing = _existing_vertex_index(point)
        if existing is not None:
            return existing
        new_verts.append((float(point[0]), float(point[1]), float(point[2])))
        index = len(new_verts) - 1
        vertex_buckets.setdefault(_key(point), []).append(index)
        return index

    new_faces: list[WOKFace] = []
    pending: list[tuple[Vec3, Vec3, Vec3]] = []
    considered = 0
    skipped_covered = 0
    skipped_partial_overlap = 0
    skipped_steep = 0
    skipped_downward = 0
    for surface in primitive.surfaces:
        if bool(getattr(surface, "backdrop", False)) or bool(getattr(surface, "background_geometry", False)):
            continue
        if not bool(getattr(surface, "render", True)):
            continue
        vertices = tuple(
            (v[0] + frame_offset[0], v[1] + frame_offset[1], v[2] + frame_offset[2])
            for v in surface.vertices
        )
        for face in tuple(surface.faces):
            try:
                a = vertices[int(face[0])]
                b = vertices[int(face[1])]
                c = vertices[int(face[2])]
            except (IndexError, TypeError, ValueError):
                continue
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = (uy * vz) - (uz * vy), (uz * vx) - (ux * vz), (ux * vy) - (uy * vx)
            length = _math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
            if length < 1.0e-9:
                continue
            considered += 1
            unit_nz = nz / length
            if unit_nz <= -min_normal_z:
                skipped_downward += 1
                continue
            if unit_nz < min_normal_z:
                skipped_steep += 1
                continue
            cx = (a[0] + b[0] + c[0]) / 3.0
            cy = (a[1] + b[1] + c[1]) / 3.0
            cz = (a[2] + b[2] + c[2]) / 3.0
            corners = (a, b, c)
            if overlaps(corners):
                if covered(cx, cy, cz):
                    skipped_covered += 1
                else:
                    skipped_partial_overlap += 1
                continue
            pending.append(corners)

    # Accept only a welded continuation of current walkable topology. Iterate
    # because a candidate can be connected through another candidate that is
    # accepted earlier in this operation.
    while pending:
        remaining: list[tuple[Vec3, Vec3, Vec3]] = []
        progress = False
        for corners in pending:
            edge_keys = tuple(
                _edge_key(corners[edge_index], corners[(edge_index + 1) % 3], epsilon=weld_epsilon)
                for edge_index in range(3)
            )
            if not any(key in walkable_edges for key in edge_keys):
                remaining.append(corners)
                continue
            cx = sum(point[0] for point in corners) / 3.0
            cy = sum(point[1] for point in corners) / 3.0
            cz = sum(point[2] for point in corners) / 3.0
            if overlaps(corners):
                if covered(cx, cy, cz):
                    skipped_covered += 1
                else:
                    skipped_partial_overlap += 1
                continue
            new_faces.append(
                WOKFace(
                    _vertex_index(corners[0]),
                    _vertex_index(corners[1]),
                    _vertex_index(corners[2]),
                    surface=4,
                    adj1=-1,
                    adj2=-1,
                    adj3=-1,
                )
            )
            coverage_faces.append(corners)
            walkable_edges.update(edge_keys)
            progress = True
        if not progress:
            pending = remaining
            break
        pending = remaining

    report: dict[str, Any] = {
        "faces_added": len(new_faces),
        "faces_considered": considered,
        "faces_already_covered": skipped_covered,
        "faces_partial_overlap": skipped_partial_overlap,
        "faces_too_steep": skipped_steep,
        "faces_downward": skipped_downward,
        "faces_unstitched": len(pending),
        "weld_policy_version": 2,
        "weld_epsilon": weld_distance,
    }
    if not new_faces:
        return primitive, report

    # rebuild_adjacencies mutates faces in place; the source primitive is
    # shared with undo snapshots, so the original faces must be copied.
    copied_faces = [
        WOKFace(
            face.v1, face.v2, face.v3, face.surface,
            face.adj1, face.adj2, face.adj3,
            getattr(face, "trans1", -1), getattr(face, "trans2", -1), getattr(face, "trans3", -1),
        )
        for face in wok.faces
    ]
    patched = WOKData(
        name=str(wok.name or primitive.room_resref or ""),
        verts=new_verts,
        faces=copied_faces + new_faces,
        relative_hook1=tuple(wok.relative_hook1),
        relative_hook2=tuple(wok.relative_hook2),
        absolute_hook1=tuple(wok.absolute_hook1),
        absolute_hook2=tuple(wok.absolute_hook2),
        position=tuple(wok.position),
    )
    patched.rebuild_adjacencies()
    from dataclasses import replace as _replace

    metadata = dict(primitive.metadata or {})
    metadata["wok_floor_fill"] = report
    return _replace(primitive, wok=patched, metadata=metadata), report


def compile_imported_mesh_room_geometry(primitive: ImportedMeshRoomPrimitive) -> AuthoredRoomGeometry:
    """Compile the imported room into render meshes plus its walkmesh."""

    validation = validate_imported_mesh_room_primitive(primitive)
    if not validation.ok:
        raise ValueError("; ".join(validation.blocking_issues))
    meshes = tuple(_surface_primitive_mesh(primitive, index) for index in range(len(primitive.surfaces)))
    backdrop_only = imported_mesh_room_is_backdrop(primitive)
    visual_only = imported_mesh_room_is_visual_only(primitive)
    # Preserve every explicit imported WOK, including a canonical zero-face
    # visual-only WOK. Only ``None`` means no source WOK was supplied.
    imported_wok = primitive.wok is not None
    if imported_wok:
        wok = primitive.wok
    elif backdrop_only:
        # A visual-only sky/backdrop room deliberately has no collision and
        # must not receive a bounds-derived rectangle.
        wok = WOKData(name=str(primitive.room_resref or ""))
    else:
        wok = _fallback_floor_wok(primitive)
    preview_only = bool(getattr(wok, "preview_only", False))
    readiness_blocking = bool(getattr(wok, "readiness_blocking", False))
    wok_provenance = str(
        getattr(
            wok,
            "provenance",
            "imported_exact" if imported_wok else ("visual_backdrop_empty" if backdrop_only else "authored"),
        )
    )
    return AuthoredRoomGeometry(
        room_resref=str(primitive.room_resref or ""),
        room_mesh=meshes[0],
        helper_meshes=meshes[1:],
        wok=wok,
        metadata={
            "primitive": IMPORTED_MESH_PRIMITIVE_KIND,
            "source_model": str(primitive.source_model or ""),
            "surface_count": len(meshes),
            "imported_wok": imported_wok,
            "backdrop_only": backdrop_only,
            "visual_only": visual_only,
            "explicit_empty_wok": visual_only,
            "walkmesh_provenance": wok_provenance,
            "walkmesh_preview_only": preview_only,
            "walkmesh_readiness_blocking": readiness_blocking,
            "walkmesh_readiness_blocking_reason": str(getattr(wok, "readiness_blocking_reason", "")),
            "warnings": validation.warnings,
        },
    )


# ---------------------------------------------------------------------------
# Face operations


def resolve_imported_mesh_face_target(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    target: str,
) -> tuple[int, ...]:
    """Resolve a marking-menu face target without broadening it accidentally.

    ``Material Region`` is the connected same-material flood fill, ``Same
    Texture Faces`` is the whole material slot, and ``Room Island`` is the
    connected geometric shell.  This distinction mirrors the labels shown to
    users and prevents a local edit from silently mutating every room face.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    selected = int(face_index)
    if not 0 <= selected < len(surface.faces):
        raise ValueError(f"Face index {face_index} out of range for surface {mesh_role}.")
    key = " ".join(str(target or "Single Face").strip().lower().split())
    if key in {"single face", "this face", "each face", "face corners"}:
        return (selected,)
    if key in {"all faces", "all mesh"}:
        return tuple(range(len(surface.faces)))

    face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )
    material = face_mats[selected]
    if key == "same texture faces":
        return tuple(index for index, value in enumerate(face_mats) if value == material)

    topology = MeshTopology.build(surface.vertices, surface.faces)
    if key == "room island":
        for component in topology.components():
            if selected in component.faces:
                return tuple(sorted(int(index) for index in component.faces))
        return (selected,)
    if key == "material region":
        pending = [selected]
        region = {selected}
        while pending:
            current = pending.pop()
            for neighbor in topology.geometric_face_to_faces.get(current, ()):
                neighbor_index = int(neighbor)
                if neighbor_index in region or face_mats[neighbor_index] != material:
                    continue
                region.add(neighbor_index)
                pending.append(neighbor_index)
        return tuple(sorted(region))
    return (selected,)


def _compact_surface(surface: ImportedMeshSurface, faces: list[Face]) -> ImportedMeshSurface:
    """Compact through the shared stable remap while preserving all channels."""

    compacted_face_mats: list[int] = []
    if len(surface.face_mats) == len(surface.faces):
        material_queues: dict[Face, list[int]] = {}
        for source_face, material in zip(surface.faces, surface.face_mats):
            material_queues.setdefault(tuple(source_face), []).append(int(material))
        for face in faces:
            queue = material_queues.get(tuple(face), [])
            compacted_face_mats.append(queue.pop(0) if queue else 0)
    elif faces:
        compacted_face_mats = [0] * len(faces)
    compacted = compact_indexed_mesh(
        surface.vertices,
        faces,
        vertex_channels={
            "uvs": surface.uvs,
            "normals": surface.normals,
            "uvs_lm": surface.uvs_lm,
        },
    )
    return replace(
        surface,
        vertices=compacted.vertices,
        faces=tuple(tuple(int(value) for value in face) for face in compacted.faces),
        face_mats=tuple(compacted_face_mats),
        uvs=tuple(compacted.vertex_channels.get("uvs", ())),
        normals=tuple(compacted.vertex_channels.get("normals", ())),
        uvs_lm=tuple(compacted.vertex_channels.get("uvs_lm", ())),
    )


def extract_imported_mesh_room_bounds(
    primitive: ImportedMeshRoomPrimitive,
    *,
    bounds: tuple[float, float, float, float, float, float],
    room_resref: str,
    surface_names: tuple[str, ...] = (),
    anchor_mode: str = "floor",
    backdrop_texture: str = "",
    backdrop_axis: str = "",
) -> ImportedMeshRoomPrimitive:
    """Clip a portable static detail from a baked retail room.

    Bounds and node-name filters are provenance recipes, not redistributed
    game data. Triangles are selected by centroid, then compacted with every
    aligned vertex channel intact. The result is centered in XY, grounded at
    Z=0, stripped of source-room lightmaps, and given a zero-face WOK so it can
    move as visual-only Odyssey room geometry without inventing walkability.
    """

    values = tuple(float(value) for value in tuple(bounds or ())[:6])
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("A dressing clip requires six finite room-local bounds values.")
    x0, y0, z0, x1, y1, z1 = values
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ValueError("Dressing clip maximum bounds must exceed minimum bounds.")
    wanted_names = {str(value or "").strip().lower() for value in tuple(surface_names or ()) if str(value or "").strip()}
    clipped: list[ImportedMeshSurface] = []
    had_lightmaps = False
    for surface in primitive.surfaces:
        if wanted_names and str(surface.name or "").strip().lower() not in wanted_names:
            continue
        faces: list[Face] = []
        for face in surface.faces:
            try:
                points = tuple(surface.vertices[int(index)] for index in face[:3])
            except (IndexError, TypeError, ValueError):
                continue
            centroid = tuple(sum(float(point[axis]) for point in points) / 3.0 for axis in range(3))
            if x0 <= centroid[0] <= x1 and y0 <= centroid[1] <= y1 and z0 <= centroid[2] <= z1:
                faces.append(tuple(int(index) for index in face[:3]))
        if not faces:
            continue
        compacted = _compact_surface(surface, faces)
        had_lightmaps = had_lightmaps or bool(compacted.lightmap)
        clipped.append(
            replace(
                compacted,
                lightmap="",
                uvs_lm=(),
                texture_names=((compacted.texture,) if compacted.texture else ()),
                tex_count=1,
                backdrop=False,
                background_geometry=False,
            )
        )
    if not clipped:
        raise ValueError("The vanilla dressing clip no longer matches its source room; refresh or update the kit recipe.")

    selected_vertices = tuple(vertex for surface in clipped for vertex in surface.vertices)
    mins = tuple(min(float(vertex[axis]) for vertex in selected_vertices) for axis in range(3))
    maxs = tuple(max(float(vertex[axis]) for vertex in selected_vertices) for axis in range(3))
    wall_anchored = str(anchor_mode or "floor").strip().lower() == "wall"
    origin = (
        (mins[0] + maxs[0]) * 0.5,
        (mins[1] + maxs[1]) * 0.5,
        (mins[2] + maxs[2]) * 0.5 if wall_anchored else mins[2],
    )
    grounded = tuple(
        replace(
            surface,
            vertices=tuple(
                (
                    float(vertex[0]) - origin[0],
                    float(vertex[1]) - origin[1],
                    float(vertex[2]) - origin[2],
                )
                for vertex in surface.vertices
            ),
            mesh_average_point=(0.0, 0.0, 0.0),
        )
        for surface in clipped
    )
    local_vertices = tuple(vertex for surface in grounded for vertex in surface.vertices)
    local_mins = tuple(min(float(vertex[axis]) for vertex in local_vertices) for axis in range(3))
    local_maxs = tuple(max(float(vertex[axis]) for vertex in local_vertices) for axis in range(3))
    surfaces = list(grounded)
    texture = str(backdrop_texture or "").strip().lower()
    axis = str(backdrop_axis or "").strip().lower()
    if texture and axis in {"x", "y"}:
        pad = 0.025
        if axis == "x":
            plane = (
                (local_mins[0] - pad, local_mins[1], local_mins[2]),
                (local_mins[0] - pad, local_maxs[1], local_mins[2]),
                (local_mins[0] - pad, local_maxs[1], local_maxs[2]),
                (local_mins[0] - pad, local_mins[1], local_maxs[2]),
            )
            normals = ((1.0, 0.0, 0.0),) * 4
        else:
            plane = (
                (local_mins[0], local_mins[1] - pad, local_mins[2]),
                (local_mins[0], local_mins[1] - pad, local_maxs[2]),
                (local_maxs[0], local_mins[1] - pad, local_maxs[2]),
                (local_maxs[0], local_mins[1] - pad, local_mins[2]),
            )
            normals = ((0.0, 1.0, 0.0),) * 4
        surfaces.append(
            ImportedMeshSurface(
                name=f"{str(room_resref or 'grdetail')[:16]}_space",
                texture=texture,
                vertices=plane,
                faces=((0, 1, 2), (0, 2, 3)),
                uvs=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                normals=normals,
                texture_names=(texture,),
                tex_count=1,
                diffuse=(1.0, 1.0, 1.0),
                ambient=(1.0, 1.0, 1.0),
                selfillum=(1.0, 1.0, 1.0),
                has_shadow=False,
            )
        )
    key = str(room_resref or "grdetail")[:16]
    return ImportedMeshRoomPrimitive(
        room_resref=key,
        surfaces=tuple(surfaces),
        source_model=primitive.source_model,
        game=primitive.game,
        wok=WOKData(name=key),
        metadata={
            "source": "map_studio:environment_kit:dressing_clip",
            "source_bounds_m": list(values),
            "source_surface_names": sorted(wanted_names),
            "anchor_mode": "wall" if wall_anchored else "floor",
            "source_lightmaps_removed_for_relighting": had_lightmaps,
            "visual_only": True,
            "render_triangle_count": sum(len(surface.faces) for surface in surfaces),
        },
    )


def _surface_face_materials(surface: ImportedMeshSurface) -> tuple[int, ...]:
    """Return one deterministic material slot per triangle."""

    if len(surface.face_mats) == len(surface.faces):
        return tuple(int(value) for value in surface.face_mats)
    return (0,) * len(surface.faces)


def _compact_surface_arrays(
    surface: ImportedMeshSurface,
    vertices: list[Vec3],
    faces: list[Face],
    uvs: list[Vec2],
    normals: list[Vec3],
    uvs_lm: list[Vec2],
    face_mats: list[int] | tuple[int, ...],
) -> ImportedMeshSurface:
    """Compact edited arrays without trying to infer changed face identity.

    ``_compact_surface`` can recover material slots when faces are merely
    removed.  Topology operators rewrite face indices, so their material rows
    travel explicitly instead.  This helper keeps UV0, lightmap UV, and normal
    channels aligned through the shared stable vertex remap.
    """

    compacted = compact_indexed_mesh(
        vertices,
        faces,
        vertex_channels={"uvs": uvs, "normals": normals, "uvs_lm": uvs_lm},
    )
    return replace(
        surface,
        vertices=compacted.vertices,
        faces=tuple(tuple(int(value) for value in face) for face in compacted.faces),
        face_mats=tuple(int(value) for value in face_mats),
        uvs=tuple(compacted.vertex_channels.get("uvs", ())),
        normals=tuple(compacted.vertex_channels.get("normals", ())),
        uvs_lm=tuple(compacted.vertex_channels.get("uvs_lm", ())),
    )


def _record_topology_edit(
    primitive: ImportedMeshRoomPrimitive,
    operation: str,
    **details: Any,
) -> ImportedMeshRoomPrimitive:
    """Attach the lightweight, serializable operator audit used by Map Studio."""

    return replace(
        primitive,
        metadata={
            **dict(primitive.metadata),
            "last_topology_edit": {
                "operation": str(operation),
                **details,
                "walkmesh_policy": "requires_review",
            },
        },
    )


def _validate_face_indices(surface: ImportedMeshSurface, face_indices: tuple[int, ...], *, role: str) -> None:
    face_count = len(surface.faces)
    for index in face_indices:
        if not (0 <= int(index) < face_count):
            raise ValueError(f"Face index {index} out of range for surface {role} ({face_count} faces).")


def delete_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
) -> ImportedMeshRoomPrimitive:
    """Delete faces from one surface; empty surfaces are removed entirely."""

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)
    kept = [face for index, face in enumerate(surface.faces) if index not in set(wanted)]
    surfaces = list(primitive.surfaces)
    if kept:
        surfaces[surface_index] = _compact_surface(surface, kept)
    else:
        del surfaces[surface_index]
    if not surfaces:
        raise ValueError(f"Deleting these faces would leave imported room {primitive.room_resref} empty.")
    return replace(primitive, surfaces=tuple(surfaces))


def set_imported_mesh_face_texture(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
    texture: str,
) -> ImportedMeshRoomPrimitive:
    """Assign a texture to faces, splitting them into a surface of that texture.

    UVs travel with the faces so KOTOR texture swaps stay mapped without a
    re-unwrap.  Retexturing every face of a surface just renames its texture.
    """

    clean_texture = str(texture or "").strip().lower()
    if not clean_texture:
        raise ValueError("A texture resref is required.")
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)
    if str(surface.texture or "").lower() == clean_texture:
        return primitive
    surfaces = list(primitive.surfaces)
    moved_faces = [surface.faces[index] for index in wanted]
    kept_faces = [face for index, face in enumerate(surface.faces) if index not in set(wanted)]
    moved = _compact_surface(replace(surface, texture=clean_texture), moved_faces)
    if kept_faces:
        surfaces[surface_index] = _compact_surface(surface, kept_faces)
    else:
        del surfaces[surface_index]

    merge_index = next(
        (
            index
            for index, existing in enumerate(surfaces)
            if str(existing.texture or "").lower() == clean_texture
            and str(existing.lightmap or "") == str(surface.lightmap or "")
        ),
        -1,
    )
    if merge_index >= 0:
        target = surfaces[merge_index]
        offset = len(target.vertices)
        vertex_count = len(target.vertices)
        tile = matched_uv_tile_size(primitive, texture=clean_texture)
        target_uvs = target.uvs if len(target.uvs) == vertex_count else tiled_uvs_for_vertices(target.vertices, tile_size=tile)
        target_normals = target.normals if len(target.normals) == vertex_count else tuple((0.0, 0.0, 1.0) for _ in target.vertices)
        moved_uvs = moved.uvs if len(moved.uvs) == len(moved.vertices) else tiled_uvs_for_vertices(moved.vertices, tile_size=tile)
        moved_normals = moved.normals if len(moved.normals) == len(moved.vertices) else tuple((0.0, 0.0, 1.0) for _ in moved.vertices)
        target_uvs_lm = target.uvs_lm if len(target.uvs_lm) == vertex_count else ()
        moved_uvs_lm = moved.uvs_lm if len(moved.uvs_lm) == len(moved.vertices) else ()
        merged_uvs_lm = (
            tuple(target_uvs_lm) + tuple(moved_uvs_lm)
            if target_uvs_lm and moved_uvs_lm
            else ()
        )
        surfaces[merge_index] = replace(
            target,
            vertices=tuple(target.vertices) + tuple(moved.vertices),
            faces=tuple(target.faces) + tuple((a + offset, b + offset, c + offset) for a, b, c in moved.faces),
            face_mats=(
                tuple(target.face_mats) if len(target.face_mats) == len(target.faces) else (0,) * len(target.faces)
            ) + (
                tuple(moved.face_mats) if len(moved.face_mats) == len(moved.faces) else (0,) * len(moved.faces)
            ),
            uvs=tuple(target_uvs) + tuple(moved_uvs),
            normals=tuple(target_normals) + tuple(moved_normals),
            uvs_lm=merged_uvs_lm,
        )
    else:
        surfaces.append(replace(moved, name=f"{primitive.room_resref}_{clean_texture}"))
    return replace(primitive, surfaces=tuple(surfaces))


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def _vec_dot(a: Vec3, b: Vec3) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _vec_lerp(a: Vec3, b: Vec3, amount: float) -> Vec3:
    t = float(amount)
    return (
        a[0] + ((b[0] - a[0]) * t),
        a[1] + ((b[1] - a[1]) * t),
        a[2] + ((b[2] - a[2]) * t),
    )


def _vec_slerp(a: Vec3, b: Vec3, amount: float) -> Vec3:
    """Spherical interpolation for two normalized directions."""

    first = _vec_normalized(a)
    second = _vec_normalized(b)
    dot = max(-1.0, min(1.0, _vec_dot(first, second)))
    t = max(0.0, min(1.0, float(amount)))
    if abs(dot) > 0.9995:
        return _vec_normalized(_vec_lerp(first, second, t), fallback=first)
    angle = math.acos(dot)
    denominator = math.sin(angle)
    if abs(denominator) <= 1.0e-9:
        return first
    first_weight = math.sin((1.0 - t) * angle) / denominator
    second_weight = math.sin(t * angle) / denominator
    return _vec_normalized(
        (
            (first[0] * first_weight) + (second[0] * second_weight),
            (first[1] * first_weight) + (second[1] * second_weight),
            (first[2] * first_weight) + (second[2] * second_weight),
        ),
        fallback=first,
    )


def _vec_length(a: Vec3) -> float:
    return ((a[0] * a[0]) + (a[1] * a[1]) + (a[2] * a[2])) ** 0.5


def _vec_normalized(a: Vec3, fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    length = _vec_length(a)
    if length <= 1.0e-9:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def _face_normal(surface: ImportedMeshSurface, face: Face) -> Vec3:
    a, b, c = (surface.vertices[i] for i in face)
    return _vec_normalized(_vec_cross(_vec_sub(b, a), _vec_sub(c, a)))


def _surface_arrays(
    surface: ImportedMeshSurface,
) -> tuple[list[Vec3], list[Face], list[Vec2], list[Vec3], list[Vec2]]:
    vertex_count = len(surface.vertices)
    uvs = list(surface.uvs) if len(surface.uvs) == vertex_count else list(tiled_uvs_for_vertices(surface.vertices))
    normals = list(surface.normals) if len(surface.normals) == vertex_count else [(0.0, 0.0, 1.0)] * vertex_count
    uvs_lm = list(surface.uvs_lm) if len(surface.uvs_lm) == vertex_count else []
    return list(surface.vertices), list(surface.faces), uvs, normals, uvs_lm


def extrude_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
    distance: float,
    *,
    point_normal: bool = False,
    tile_size: float = 0.0,
    direction: Vec3 | None = None,
) -> ImportedMeshRoomPrimitive:
    """Extrude a face region along its normal, creating side walls.

    The cap keeps the original UVs; each side quad gets world-density tiled
    UVs so new walls blend with vanilla texturing.  The cap is duplicated
    rather than welded to the ring — visually correct for KOTOR render
    meshes, and the WOK is untouched.

    ``direction`` overrides the pull axis (Maya world-mode extrude); the
    region normal is used when omitted.
    """

    offset_distance = float(distance)
    if abs(offset_distance) <= 1.0e-9:
        return primitive
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)
    region = [surface.faces[index] for index in wanted]
    tile = float(tile_size) if tile_size > 0.0 else matched_uv_tile_size(primitive, texture=surface.texture)

    topology = MeshTopology.build(surface.vertices, surface.faces)
    selected_half_edges = tuple(
        half_edge
        for face_index in wanted
        for half_edge in topology.get_half_edges_for_face(face_index)
    )
    unsafe_edges = {
        (min(row.geometric_origin, row.geometric_destination), max(row.geometric_origin, row.geometric_destination))
        for row in selected_half_edges
        if len(
            topology.geometric_edge_to_half_edges.get(
                (min(row.geometric_origin, row.geometric_destination), max(row.geometric_origin, row.geometric_destination)),
                (),
            )
        )
        > 2
    }
    if unsafe_edges:
        raise ValueError("Extrude requires a manifold selected region; repair non-manifold edges first.")

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    source_face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )

    face_normals = {index: topology.face_normals[index] for index in wanted}
    region_normal = _vec_normalized(
        (
            sum(face_normals[index][0] * topology.face_areas[index] for index in wanted),
            sum(face_normals[index][1] * topology.face_areas[index] for index in wanted),
            sum(face_normals[index][2] * topology.face_areas[index] for index in wanted),
        )
    )
    if direction is not None:
        region_normal = _vec_normalized(tuple(float(v) for v in tuple(direction)[:3]))
        point_normal = False

    def _vertex_offset(vertex_index: int) -> Vec3:
        if not point_normal:
            return _vec_scale(region_normal, offset_distance)
        adjacent = [
            face_normals[index]
            for index in wanted
            if vertex_index in surface.faces[index]
        ]
        combined = _vec_normalized(
            (
                sum(n[0] for n in adjacent),
                sum(n[1] for n in adjacent),
                sum(n[2] for n in adjacent),
            ),
            fallback=region_normal,
        )
        return _vec_scale(combined, offset_distance)

    # Cap: duplicate the region's vertices offset along the normal.
    cap_map: dict[int, int] = {}
    for face in region:
        for vertex_index in face:
            if vertex_index not in cap_map:
                cap_map[vertex_index] = len(vertices)
                vertices.append(_vec_add(surface.vertices[vertex_index], _vertex_offset(vertex_index)))
                uvs.append(uvs[vertex_index])
                normals.append(normals[vertex_index])
                if uvs_lm:
                    uvs_lm.append(uvs_lm[vertex_index])
    cap_faces = [(cap_map[a], cap_map[b], cap_map[c]) for a, b, c in region]

    # The shared topology view recognizes UV/hard-normal seam copies as one
    # geometric edge, so internal selected edges do not sprout duplicate walls.
    boundary = topology.region_boundary_half_edges(wanted)

    side_faces: list[Face] = []
    side_face_mats: list[int] = []
    v_extent = abs(offset_distance) / max(1.0e-6, tile)
    for boundary_edge in boundary:
        start, end = boundary_edge.origin, boundary_edge.destination
        pa = surface.vertices[start]
        pb = surface.vertices[end]
        pa_top = vertices[cap_map[start]]
        pb_top = vertices[cap_map[end]]
        edge_length = _vec_length(_vec_sub(pb, pa))
        u_extent = edge_length / max(1.0e-6, tile)
        side_normal = _vec_normalized(_vec_cross(_vec_sub(pb, pa), _vec_sub(pa_top, pa)))
        base = len(vertices)
        vertices.extend((pa, pb, pb_top, pa_top))
        uvs.extend(((0.0, 0.0), (u_extent, 0.0), (u_extent, v_extent), (0.0, v_extent)))
        normals.extend((side_normal,) * 4)
        if uvs_lm:
            uvs_lm.extend((uvs_lm[start], uvs_lm[end], uvs_lm[cap_map[end]], uvs_lm[cap_map[start]]))
        side_faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        side_face_mats.extend((source_face_mats[boundary_edge.face],) * 2)

    wanted_set = set(wanted)
    kept_indices = [index for index in range(len(surface.faces)) if index not in wanted_set]
    kept_faces = [surface.faces[index] for index in kept_indices]
    new_faces = kept_faces + cap_faces + side_faces
    new_face_mats = (
        [source_face_mats[index] for index in kept_indices]
        + [source_face_mats[index] for index in wanted]
        + side_face_mats
    )
    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(new_faces),
        face_mats=tuple(new_face_mats),
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    rebuilt = _compact_surface(rebuilt, list(rebuilt.faces))
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    return replace(primitive, surfaces=tuple(surfaces))


def inset_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
    inset: float,
) -> ImportedMeshRoomPrimitive:
    """Inset each target face toward its centroid, creating a border ring.

    UVs interpolate toward the face's UV centroid so the original mapping is
    preserved across the inset (doorway/recess workflow).
    """

    amount = float(inset)
    if amount <= 1.0e-9:
        return primitive
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    source_face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )
    replacement_faces: list[Face] = []
    replacement_face_mats: list[int] = []
    for face_index in wanted:
        a, b, c = surface.faces[face_index]
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        centroid = ((pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0, (pa[2] + pb[2] + pc[2]) / 3.0)
        uv_centroid = (
            (uvs[a][0] + uvs[b][0] + uvs[c][0]) / 3.0,
            (uvs[a][1] + uvs[b][1] + uvs[c][1]) / 3.0,
        )
        lm_centroid = (
            (
                (uvs_lm[a][0] + uvs_lm[b][0] + uvs_lm[c][0]) / 3.0,
                (uvs_lm[a][1] + uvs_lm[b][1] + uvs_lm[c][1]) / 3.0,
            )
            if uvs_lm
            else None
        )
        min_reach = min(_vec_length(_vec_sub(centroid, p)) for p in (pa, pb, pc))
        ratio = max(0.05, min(0.95, amount / max(1.0e-6, min_reach)))
        inner: list[int] = []
        for corner in (a, b, c):
            point = vertices[corner]
            inner_index = len(vertices)
            vertices.append(_vec_add(point, _vec_scale(_vec_sub(centroid, point), ratio)))
            uvs.append(
                (
                    uvs[corner][0] + ((uv_centroid[0] - uvs[corner][0]) * ratio),
                    uvs[corner][1] + ((uv_centroid[1] - uvs[corner][1]) * ratio),
                )
            )
            normals.append(normals[corner])
            if uvs_lm and lm_centroid is not None:
                uvs_lm.append(
                    (
                        uvs_lm[corner][0] + ((lm_centroid[0] - uvs_lm[corner][0]) * ratio),
                        uvs_lm[corner][1] + ((lm_centroid[1] - uvs_lm[corner][1]) * ratio),
                    )
                )
            inner.append(inner_index)
        ia, ib, ic = inner
        replacement_faces.extend(
            (
                (ia, ib, ic),
                (a, b, ib),
                (a, ib, ia),
                (b, c, ic),
                (b, ic, ib),
                (c, a, ia),
                (c, ia, ic),
            )
        )
        replacement_face_mats.extend((source_face_mats[face_index],) * 7)
    wanted_set = set(wanted)
    kept_indices = [index for index in range(len(surface.faces)) if index not in wanted_set]
    kept_faces = [surface.faces[index] for index in kept_indices]
    new_faces = kept_faces + replacement_faces
    new_face_mats = [source_face_mats[index] for index in kept_indices] + replacement_face_mats
    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(new_faces),
        face_mats=tuple(new_face_mats),
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    rebuilt = _compact_surface(rebuilt, list(rebuilt.faces))
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    return replace(primitive, surfaces=tuple(surfaces))


def move_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
    delta: Vec3,
) -> ImportedMeshRoomPrimitive:
    """Translate target-face points and every co-located seam copy.

    KOTOR room meshes commonly split one geometric point into several raw
    vertices for UV, lightmap, normal, or material seams.  Moving only the raw
    indices referenced by a selected triangle opens cracks along those seams.
    Face movement therefore follows the same position-welded policy as the
    edge and vertex manipulators below.
    """

    offset = tuple(float(v) for v in tuple(delta)[:3])
    if _vec_length(offset) <= 1.0e-9:
        return primitive
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)
    moved_positions: set[tuple[float, float, float]] = set()
    for face_index in wanted:
        moved_positions.update(_position_key(surface.vertices[index]) for index in surface.faces[face_index])
    return _translate_positions(primitive, list(moved_positions), offset)


# ---------------------------------------------------------------------------
# Edge / vertex component operations (position-welded)
#
# Stock KOTOR meshes duplicate vertices at UV seams and texture-surface
# boundaries.  Component ops therefore key on rounded world POSITIONS and
# touch every co-located copy across every surface, or edits would tear the
# mesh open at seams.

_POSITION_EPSILON_DIGITS = 4


def _position_key(point: Vec3) -> tuple[float, float, float]:
    return (
        round(float(point[0]), _POSITION_EPSILON_DIGITS),
        round(float(point[1]), _POSITION_EPSILON_DIGITS),
        round(float(point[2]), _POSITION_EPSILON_DIGITS),
    )


def _resolve_component_positions(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    *,
    vertex_corner: int = -1,
    edge_corners: tuple[int, int] = (-1, -1),
) -> tuple[ImportedMeshSurface, list[tuple[float, float, float]]]:
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    if not (0 <= int(face_index) < len(surface.faces)):
        raise ValueError(f"Face index {face_index} out of range for surface {mesh_role}.")
    face = surface.faces[int(face_index)]
    if vertex_corner >= 0:
        corner = int(vertex_corner) % 3
        return surface, [_position_key(surface.vertices[face[corner]])]
    a, b = (int(v) % 3 for v in edge_corners)
    if a == b:
        raise ValueError("Edge corners must reference two different face corners.")
    return surface, [_position_key(surface.vertices[face[a]]), _position_key(surface.vertices[face[b]])]


def _translate_positions(
    primitive: ImportedMeshRoomPrimitive,
    positions: list[tuple[float, float, float]],
    delta: Vec3,
) -> ImportedMeshRoomPrimitive:
    wanted = set(positions)
    offset = tuple(float(v) for v in tuple(delta)[:3])
    surfaces = []
    for surface in primitive.surfaces:
        vertices = tuple(
            _vec_add(vertex, offset) if _position_key(vertex) in wanted else vertex
            for vertex in surface.vertices
        )
        surfaces.append(replace(surface, vertices=vertices) if vertices != surface.vertices else surface)
    return replace(primitive, surfaces=tuple(surfaces))


def _snap_positions(
    primitive: ImportedMeshRoomPrimitive,
    positions: list[tuple[float, float, float]],
    target: Vec3,
) -> ImportedMeshRoomPrimitive:
    wanted = set(positions)
    surfaces = []
    for surface in primitive.surfaces:
        vertices = tuple(
            tuple(target) if _position_key(vertex) in wanted else vertex for vertex in surface.vertices
        )
        surfaces.append(replace(surface, vertices=vertices) if vertices != surface.vertices else surface)
    return _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))


def _drop_degenerate_faces(primitive: ImportedMeshRoomPrimitive) -> ImportedMeshRoomPrimitive:
    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        kept: list[Face] = []
        for face in surface.faces:
            points = tuple(surface.vertices[index] for index in face)
            if len({_position_key(point) for point in points}) != 3:
                continue
            # A weld can leave three distinct but collinear points.  Odyssey
            # still treats that as a zero-area triangle, so reject it here
            # rather than relying only on repeated-index detection.
            if _vec_length(_vec_cross(_vec_sub(points[1], points[0]), _vec_sub(points[2], points[0]))) <= 1.0e-10:
                continue
            kept.append(face)
        if not kept:
            continue
        if len(kept) == len(surface.faces):
            surfaces.append(surface)
        else:
            surfaces.append(_compact_surface(surface, kept))
    if not surfaces:
        raise ValueError(f"Edit would leave imported room {primitive.room_resref} empty.")
    return replace(primitive, surfaces=tuple(surfaces))


def move_imported_mesh_vertex(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    vertex_corner: int,
    delta: Vec3,
) -> ImportedMeshRoomPrimitive:
    """Translate a vertex and every co-located seam copy across all surfaces."""

    _surface, positions = _resolve_component_positions(
        primitive, mesh_role, face_index, vertex_corner=vertex_corner
    )
    return _translate_positions(primitive, positions, delta)


def move_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
    delta: Vec3,
) -> ImportedMeshRoomPrimitive:
    """Translate both edge endpoints (and their seam copies)."""

    _surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, edge_corners=edge_corners)
    return _translate_positions(primitive, positions, delta)


def extrude_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
    delta: Vec3,
    *,
    tile_size: float = 0.0,
) -> ImportedMeshRoomPrimitive:
    """Extrude one edge into a new quad offset by delta (Maya Ctrl+E on edges).

    The edge endpoints stay put; two offset copies form the far side of the
    quad.  The quad is duplicated rather than welded, matching face extrusion,
    so seams stay paintable per-face and the WOK is untouched.
    """

    offset = tuple(float(v) for v in tuple(delta)[:3])
    if _vec_length(offset) <= 1.0e-9:
        return primitive
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    if not (0 <= int(face_index) < len(surface.faces)):
        raise ValueError(f"Face index {face_index} out of range for surface {mesh_role}.")
    face = surface.faces[int(face_index)]
    a_corner, b_corner = (int(v) % 3 for v in edge_corners)
    if a_corner == b_corner:
        raise ValueError("Edge corners must reference two different face corners.")
    start = surface.vertices[face[a_corner]]
    end = surface.vertices[face[b_corner]]
    tile = float(tile_size) if tile_size > 0.0 else matched_uv_tile_size(primitive, texture=surface.texture)
    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    source_face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )
    edge_length = _vec_length(_vec_sub(end, start))
    u_extent = edge_length / max(1.0e-6, tile)
    v_extent = _vec_length(offset) / max(1.0e-6, tile)
    quad_normal = _vec_normalized(_vec_cross(_vec_sub(end, start), offset))
    base = len(vertices)
    vertices.extend((start, end, _vec_add(end, offset), _vec_add(start, offset)))
    uvs.extend(((0.0, 0.0), (u_extent, 0.0), (u_extent, v_extent), (0.0, v_extent)))
    normals.extend((quad_normal,) * 4)
    if uvs_lm:
        uvs_lm.extend(
            (
                uvs_lm[face[a_corner]],
                uvs_lm[face[b_corner]],
                uvs_lm[face[b_corner]],
                uvs_lm[face[a_corner]],
            )
        )
    faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(faces),
        face_mats=source_face_mats + (source_face_mats[int(face_index)],) * 2,
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    return replace(primitive, surfaces=tuple(surfaces))


def _legacy_bevel_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
    amount: float,
) -> ImportedMeshRoomPrimitive:
    """Chamfer one manifold hard edge into a new renderable strip.

    This follows Maya's selected-edge contract: the original edge is replaced
    by two parallel edges and a new face.  KOTOR room meshes commonly share
    indexed vertices across planar triangle fans, so the rebuild duplicates
    the affected surface vertices per face before applying the two plane
    offsets.  Small end caps keep the result closed at both edge endpoints.

    Boundary and non-manifold edges are rejected instead of producing an open
    room seam.  The amount is clamped against the adjacent triangle altitudes
    so a large drag cannot invert those faces.
    """

    requested = abs(float(amount))
    if requested <= 1.0e-9:
        return primitive
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    if not (0 <= int(face_index) < len(surface.faces)):
        raise ValueError(f"Face index {face_index} out of range for surface {mesh_role}.")
    selected_face = surface.faces[int(face_index)]
    a_corner, b_corner = (int(value) % 3 for value in edge_corners)
    if a_corner == b_corner:
        raise ValueError("Edge corners must reference two different face corners.")
    start = tuple(surface.vertices[selected_face[a_corner]])
    end = tuple(surface.vertices[selected_face[b_corner]])
    edge_vector = _vec_sub(end, start)
    edge_length = _vec_length(edge_vector)
    if edge_length <= 1.0e-8:
        raise ValueError("Cannot bevel a zero-length edge.")
    edge_unit = _vec_scale(edge_vector, 1.0 / edge_length)
    start_key, end_key = _position_key(start), _position_key(end)

    incident: list[tuple[int, Face, Vec3, Vec3, float]] = []
    for index, face in enumerate(surface.faces):
        keys = tuple(_position_key(surface.vertices[vertex_index]) for vertex_index in face)
        if start_key not in keys or end_key not in keys:
            continue
        third_index = next(
            (vertex_index for vertex_index in face if _position_key(surface.vertices[vertex_index]) not in {start_key, end_key}),
            -1,
        )
        if third_index < 0:
            continue
        normal = _face_normal(surface, face)
        inward = _vec_normalized(_vec_cross(normal, edge_unit), fallback=(0.0, 0.0, 0.0))
        third = tuple(surface.vertices[third_index])
        if sum(inward[axis] * (third[axis] - start[axis]) for axis in range(3)) < 0.0:
            inward = _vec_scale(inward, -1.0)
        altitude = abs(sum(inward[axis] * (third[axis] - start[axis]) for axis in range(3)))
        incident.append((index, face, normal, inward, altitude))
    if len(incident) != 2:
        raise ValueError(
            f"Bevel requires one manifold edge shared by exactly two faces; found {len(incident)} incident face(s)."
        )
    if abs(sum(incident[0][2][axis] * incident[1][2][axis] for axis in range(3))) > 0.999:
        raise ValueError("The selected edge is coplanar; select a hard edge between two surface planes.")

    distance = min(requested, max(1.0e-5, min(row[4] for row in incident) * 0.45))
    plane_rows = tuple(incident)
    offset_edges = tuple(
        (_vec_add(start, _vec_scale(row[3], distance)), _vec_add(end, _vec_scale(row[3], distance)))
        for row in plane_rows
    )
    vertices: list[Vec3] = []
    faces: list[Face] = []
    uvs: list[Vec2] = []
    normals: list[Vec3] = []

    def append_triangle(points: tuple[Vec3, Vec3, Vec3], desired_normal: Vec3) -> None:
        if _vec_length(_vec_cross(_vec_sub(points[1], points[0]), _vec_sub(points[2], points[0]))) <= 1.0e-8:
            return
        actual = _vec_cross(_vec_sub(points[1], points[0]), _vec_sub(points[2], points[0]))
        if sum(actual[axis] * desired_normal[axis] for axis in range(3)) < 0.0:
            points = (points[0], points[2], points[1])
        base = len(vertices)
        vertices.extend(points)
        faces.append((base, base + 1, base + 2))
        uvs.extend(tiled_uvs_for_vertices(points))
        normals.extend((desired_normal,) * 3)

    # Rebuild each original triangle.  Triangles on either adjacent plane use
    # that plane's inset copies of the selected endpoints; all other planes
    # keep the original corner and meet the generated endpoint caps.
    for face in surface.faces:
        face_normal = _face_normal(surface, face)
        plane_index = -1
        for candidate_index, row in enumerate(plane_rows):
            if sum(face_normal[axis] * row[2][axis] for axis in range(3)) > 0.999:
                plane_index = candidate_index
                break
        rebuilt_points: list[Vec3] = []
        for vertex_index in face:
            point = tuple(surface.vertices[vertex_index])
            key = _position_key(point)
            if plane_index >= 0 and key == start_key:
                point = offset_edges[plane_index][0]
            elif plane_index >= 0 and key == end_key:
                point = offset_edges[plane_index][1]
            rebuilt_points.append(point)
        append_triangle(tuple(rebuilt_points), face_normal)

    a0, b0 = offset_edges[0]
    a1, b1 = offset_edges[1]
    bevel_normal = _vec_normalized(_vec_add(plane_rows[0][2], plane_rows[1][2]), fallback=plane_rows[0][2])
    append_triangle((a0, b0, b1), bevel_normal)
    append_triangle((a0, b1, a1), bevel_normal)
    append_triangle((start, a1, a0), _vec_scale(edge_unit, -1.0))
    append_triangle((end, b0, b1), edge_unit)

    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(faces),
        uvs=tuple(uvs),
        normals=tuple(normals),
    )
    rebuilt = _compact_surface(rebuilt, list(rebuilt.faces))
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    return replace(primitive, surfaces=tuple(surfaces))


def bevel_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
    amount: float = 0.25,
    *,
    segments: int = 1,
    profile: float = 0.5,
    miter: str = "auto",
    smoothing_angle_degrees: float = 180.0,
    uv_mode: str = "preserve",
    clamp_overlap: bool = True,
    options: ImportedMeshBevelOptions | None = None,
) -> ImportedMeshRoomPrimitive:
    """Bevel one manifold hard edge without rebuilding unrelated topology.

    The operator evaluates from the immutable input primitive on every call,
    which makes it safe for an interactive preview session.  Original faces,
    UVs, lightmap UVs, normals, texture/lightmap recipe, and face ordering are
    preserved; only the selected edge's geometric one-ring is rewired.
    """

    if options is not None:
        amount = options.width
        segments = options.segments
        profile = options.profile
        miter = options.miter
        smoothing_angle_degrees = options.smoothing_angle_degrees
        uv_mode = options.uv_mode
        clamp_overlap = options.clamp_overlap
    requested = abs(float(amount))
    if requested <= 1.0e-9:
        return primitive
    segment_count = max(1, min(64, int(segments)))
    round_profile = max(0.0, min(1.0, float(profile)))
    miter_mode = str(miter or "auto").strip().lower()
    if miter_mode not in {"auto", "sharp", "patch"}:
        raise ValueError("Single-edge bevel miter must be Auto, Sharp, or Patch.")
    uv_policy = str(uv_mode or "preserve").strip().lower()
    if uv_policy not in {"preserve", "tiled", "none"}:
        raise ValueError("Bevel UV mode must be Preserve, Tiled, or None.")

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    selected_face_index = int(face_index)
    if not 0 <= selected_face_index < len(surface.faces):
        raise ValueError(f"Face index {face_index} out of range for surface {mesh_role}.")
    selected_face = surface.faces[selected_face_index]
    first_corner, second_corner = (int(value) % len(selected_face) for value in edge_corners)
    if first_corner == second_corner:
        raise ValueError("Edge corners must reference two different face corners.")

    topology = MeshTopology.build(surface.vertices, surface.faces)
    geometric_edge = topology.geometric_edge_for_face_corners(
        selected_face_index,
        (first_corner, second_corner),
    )
    edge_half_indices = topology.geometric_edge_to_half_edges.get(geometric_edge, ())
    if len(edge_half_indices) != 2:
        raise ValueError(
            "Bevel requires one manifold edge shared by exactly two faces; "
            f"found {len(edge_half_indices)} incident face(s)."
        )
    first_half, second_half = (topology.half_edges[index] for index in edge_half_indices)
    if first_half.twin != second_half.index or second_half.twin != first_half.index:
        raise ValueError("Bevel requires consistently wound adjacent faces; repair the selected edge first.")

    start_raw = selected_face[first_corner]
    end_raw = selected_face[second_corner]
    start = tuple(surface.vertices[start_raw])
    end = tuple(surface.vertices[end_raw])
    edge_vector = _vec_sub(end, start)
    edge_length = _vec_length(edge_vector)
    if edge_length <= 1.0e-8:
        raise ValueError("Cannot bevel a zero-length edge.")
    edge_unit = _vec_scale(edge_vector, 1.0 / edge_length)
    start_geometric = topology.raw_to_geometric_vertex[start_raw]
    end_geometric = topology.raw_to_geometric_vertex[end_raw]

    plane_rows: list[tuple[int, Vec3, Vec3, float]] = []
    for half_edge in (first_half, second_half):
        incident_face = surface.faces[half_edge.face]
        normal = topology.face_normals[half_edge.face]
        third_raw = next(
            (
                raw_index
                for raw_index in incident_face
                if topology.raw_to_geometric_vertex[raw_index] not in {start_geometric, end_geometric}
            ),
            -1,
        )
        if third_raw < 0:
            raise ValueError("Bevel adjacent faces must contain a third non-edge vertex.")
        inward = _vec_normalized(_vec_cross(normal, edge_unit), fallback=(0.0, 0.0, 0.0))
        third = surface.vertices[third_raw]
        if _vec_dot(inward, _vec_sub(third, start)) < 0.0:
            inward = _vec_scale(inward, -1.0)
        altitude = abs(_vec_dot(inward, _vec_sub(third, start)))
        plane_rows.append((half_edge.face, normal, inward, altitude))
    if abs(_vec_dot(plane_rows[0][1], plane_rows[1][1])) > 0.999:
        raise ValueError("The selected edge is coplanar; select a hard edge between two surface planes.")
    maximum = max(1.0e-5, min(row[3] for row in plane_rows) * 0.45)
    if not clamp_overlap and requested > maximum:
        raise ValueError(
            f"Bevel width {requested:.4f}m would invert an adjacent face; maximum safe width is {maximum:.4f}m."
        )
    distance = min(requested, maximum) if clamp_overlap else requested

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    source_face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )
    component_faces = next(
        (
            set(component.faces)
            for component in topology.components()
            if selected_face_index in component.faces
        ),
        {selected_face_index},
    )
    replacement_vertices: dict[tuple[int, int], int] = {}
    rewritten_faces = list(faces)
    for candidate_face_index in sorted(component_faces):
        candidate_face = surface.faces[candidate_face_index]
        face_normal = topology.face_normals[candidate_face_index]
        plane_index = next(
            (
                index
                for index, row in enumerate(plane_rows)
                if _vec_dot(face_normal, row[1]) > 0.999
            ),
            -1,
        )
        if plane_index < 0:
            continue
        changed = False
        replacement_face: list[int] = []
        for raw_index in candidate_face:
            geometric = topology.raw_to_geometric_vertex[raw_index]
            if geometric not in {start_geometric, end_geometric}:
                replacement_face.append(raw_index)
                continue
            key = (raw_index, plane_index)
            replacement_index = replacement_vertices.get(key)
            if replacement_index is None:
                replacement_index = len(vertices)
                replacement_vertices[key] = replacement_index
                vertices.append(_vec_add(surface.vertices[raw_index], _vec_scale(plane_rows[plane_index][2], distance)))
                uvs.append(uvs[raw_index])
                normals.append(normals[raw_index])
                if uvs_lm:
                    uvs_lm.append(uvs_lm[raw_index])
            replacement_face.append(replacement_index)
            changed = True
        if changed:
            rewritten_faces[candidate_face_index] = tuple(replacement_face)

    def _channel_value(channel: list, plane_index: int, geometric_vertex: int, fallback):
        face = surface.faces[plane_rows[plane_index][0]]
        raw_index = next(
            index
            for index in face
            if topology.raw_to_geometric_vertex[index] == geometric_vertex
        )
        return channel[raw_index] if channel else fallback

    plane_uvs = tuple(
        (
            _channel_value(uvs, plane_index, start_geometric, (0.0, 0.0)),
            _channel_value(uvs, plane_index, end_geometric, (0.0, 0.0)),
        )
        for plane_index in range(2)
    )
    plane_lightmap_uvs = (
        tuple(
            (
                _channel_value(uvs_lm, plane_index, start_geometric, (0.0, 0.0)),
                _channel_value(uvs_lm, plane_index, end_geometric, (0.0, 0.0)),
            )
            for plane_index in range(2)
        )
        if uvs_lm
        else ()
    )
    tile = matched_uv_tile_size(primitive, texture=surface.texture)
    dihedral_degrees = math.degrees(math.acos(max(-1.0, min(1.0, _vec_dot(plane_rows[0][1], plane_rows[1][1])))))
    smooth_strip = float(smoothing_angle_degrees) > 0.0 and dihedral_degrees <= float(smoothing_angle_degrees)
    resolved_miter = (
        ("patch" if segment_count > 1 else "sharp")
        if miter_mode == "auto"
        else miter_mode
    )

    def _offset_direction(t: float) -> Vec3:
        linear = _vec_lerp(plane_rows[0][2], plane_rows[1][2], t)
        center = _vec_add(plane_rows[0][2], plane_rows[1][2])
        arc = _vec_sub(center, _vec_slerp(plane_rows[0][2], plane_rows[1][2], 1.0 - t))
        return _vec_lerp(linear, arc, round_profile)

    line_vertices: list[tuple[int, int]] = []
    for segment_index in range(segment_count + 1):
        t = segment_index / float(segment_count)
        offset = _vec_scale(_offset_direction(t), distance)
        line_normal = (
            _vec_slerp(plane_rows[0][1], plane_rows[1][1], t)
            if smooth_strip
            else _vec_normalized(_vec_add(plane_rows[0][1], plane_rows[1][1]), fallback=plane_rows[0][1])
        )
        pair: list[int] = []
        for endpoint_index, point in enumerate((start, end)):
            vertex_index = len(vertices)
            pair.append(vertex_index)
            vertices.append(_vec_add(point, offset))
            if uv_policy == "preserve":
                first_uv = plane_uvs[0][endpoint_index]
                second_uv = plane_uvs[1][endpoint_index]
                uvs.append(
                    (
                        first_uv[0] + ((second_uv[0] - first_uv[0]) * t),
                        first_uv[1] + ((second_uv[1] - first_uv[1]) * t),
                    )
                )
            elif uv_policy == "tiled":
                uvs.append(
                    (
                        edge_length / max(1.0e-6, tile) if endpoint_index else 0.0,
                        (distance * t) / max(1.0e-6, tile),
                    )
                )
            else:
                uvs.append((0.0, 0.0))
            normals.append(line_normal)
            if uvs_lm:
                first_lm = plane_lightmap_uvs[0][endpoint_index]
                second_lm = plane_lightmap_uvs[1][endpoint_index]
                uvs_lm.append(
                    (
                        first_lm[0] + ((second_lm[0] - first_lm[0]) * t),
                        first_lm[1] + ((second_lm[1] - first_lm[1]) * t),
                    )
                )
        line_vertices.append((pair[0], pair[1]))

    generated_faces: list[Face] = []

    def _append_oriented(first: int, second: int, third: int, desired_normal: Vec3) -> None:
        actual = _vec_cross(_vec_sub(vertices[second], vertices[first]), _vec_sub(vertices[third], vertices[first]))
        if _vec_length(actual) <= 1.0e-10:
            raise ValueError("Bevel generated a degenerate triangle; reduce width or segments.")
        if _vec_dot(actual, desired_normal) < 0.0:
            second, third = third, second
        generated_faces.append((first, second, third))

    for segment_index in range(segment_count):
        start0, end0 = line_vertices[segment_index]
        start1, end1 = line_vertices[segment_index + 1]
        desired = _vec_normalized(
            _vec_add(normals[start0], normals[start1]),
            fallback=_vec_normalized(_vec_add(plane_rows[0][1], plane_rows[1][1])),
        )
        _append_oriented(start0, end0, end1, desired)
        _append_oriented(start0, end1, start1, desired)

    # A single selected edge always needs endpoint patches. Sharp converges the
    # cap at the original endpoint; Patch pulls the fan center halfway toward
    # the rounded rail average, producing a visibly broader corner. Auto uses
    # Sharp for a one-segment chamfer and Patch for a multi-segment roundover.
    cap_centers: list[int] = []
    for endpoint_index, point in enumerate((start, end)):
        track_indices = tuple(pair[endpoint_index] for pair in line_vertices)
        track_points = tuple(vertices[index] for index in track_indices)
        track_average = (
            sum(value[0] for value in track_points) / len(track_points),
            sum(value[1] for value in track_points) / len(track_points),
            sum(value[2] for value in track_points) / len(track_points),
        )
        cap_point = _vec_lerp(point, track_average, 0.5) if resolved_miter == "patch" else point
        center_index = len(vertices)
        cap_centers.append(center_index)
        vertices.append(cap_point)
        uvs.append(
            (
                sum(uvs[index][0] for index in track_indices) / len(track_indices),
                sum(uvs[index][1] for index in track_indices) / len(track_indices),
            )
        )
        normals.append(_vec_scale(edge_unit, -1.0 if endpoint_index == 0 else 1.0))
        if uvs_lm:
            uvs_lm.append(
                (
                    sum(uvs_lm[index][0] for index in track_indices) / len(track_indices),
                    sum(uvs_lm[index][1] for index in track_indices) / len(track_indices),
                )
            )
    for segment_index in range(segment_count):
        _append_oriented(
            cap_centers[0],
            line_vertices[segment_index + 1][0],
            line_vertices[segment_index][0],
            _vec_scale(edge_unit, -1.0),
        )
        _append_oriented(
            cap_centers[1],
            line_vertices[segment_index][1],
            line_vertices[segment_index + 1][1],
            edge_unit,
        )

    # Hard bevel shading needs independent face-corner normals.  Duplicate the
    # generated triangle corners while retaining coincident geometric positions;
    # the shared topology engine welds those positions for manifold auditing,
    # while Odyssey receives the split vertices required for flat shading.
    if not smooth_strip:
        hardened_faces: list[Face] = []
        for first, second, third in generated_faces:
            face_normal = _vec_normalized(
                _vec_cross(_vec_sub(vertices[second], vertices[first]), _vec_sub(vertices[third], vertices[first])),
                fallback=(0.0, 0.0, 1.0),
            )
            hardened_face: list[int] = []
            for source_index in (first, second, third):
                hardened_face.append(len(vertices))
                vertices.append(vertices[source_index])
                uvs.append(uvs[source_index])
                normals.append(face_normal)
                if uvs_lm:
                    uvs_lm.append(uvs_lm[source_index])
            hardened_faces.append(tuple(hardened_face))
        generated_faces = hardened_faces

    generated_face_start = len(rewritten_faces)
    rewritten_faces.extend(generated_faces)
    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(rewritten_faces),
        face_mats=source_face_mats + (source_face_mats[selected_face_index],) * len(generated_faces),
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    rebuilt = _compact_surface(rebuilt, list(rebuilt.faces))
    audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
    if audit.degenerate_faces or audit.non_manifold_edges:
        raise ValueError(
            "Bevel would create invalid topology "
            f"({len(audit.degenerate_faces)} degenerate face(s), "
            f"{len(audit.non_manifold_edges)} non-manifold edge(s))."
        )
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    metadata = {
        **dict(primitive.metadata),
        "last_topology_edit": {
            "operation": "edge_bevel",
            "mesh_role": mesh_role,
            "source_face": selected_face_index,
            "source_edge_corners": [first_corner, second_corner],
            "width": distance,
            "requested_width": requested,
            "segments": segment_count,
            "profile": round_profile,
            "miter": miter_mode,
            "resolved_miter": resolved_miter,
            "smoothing_angle_degrees": float(smoothing_angle_degrees),
            "smooth_strip": smooth_strip,
            "uv_mode": uv_policy,
            "generated_face_start": generated_face_start,
            "generated_face_count": len(generated_faces),
            "walkmesh_policy": "requires_review",
        },
    }
    return replace(primitive, surfaces=tuple(surfaces), metadata=metadata)


def _locate_imported_mesh_edge_by_positions(
    surface: ImportedMeshSurface,
    start: Vec3,
    end: Vec3,
) -> tuple[int, tuple[int, int]]:
    """Resolve one still-unedited geometric edge back to a face/corner pair."""

    start_key = _position_key(start)
    end_key = _position_key(end)
    for face_index, face in enumerate(surface.faces):
        for corner, raw_index in enumerate(face):
            following = (corner + 1) % len(face)
            following_raw = face[following]
            keys = {_position_key(surface.vertices[raw_index]), _position_key(surface.vertices[following_raw])}
            if keys == {start_key, end_key}:
                return face_index, (corner, following)
    raise ValueError("A selected bevel edge changed before the atomic multi-edge edit could be evaluated.")


def _bevel_imported_mesh_continuous_crease(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[tuple[int, int], ...],
    amount: float,
    *,
    segments: int,
    profile: float,
    miter: str,
    smoothing_angle_degrees: float,
    uv_mode: str,
    clamp_overlap: bool,
) -> ImportedMeshRoomPrimitive:
    """Bevel one straight, non-branching crease chain as one topology edit.

    Odyssey room geometry is triangulated, so a visually continuous hard edge
    is often represented by several collinear manifold edges.  Evaluating
    those edges independently would cap every segment and produce overlapping
    topology.  This helper shares the generated cross-section at intermediate
    vertices and caps only the two ends of the chain.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    topology = MeshTopology.build(surface.vertices, surface.faces)
    adjacency: dict[int, set[int]] = {}
    selected_edges: set[tuple[int, int]] = set()
    for raw_first, raw_second in edge_vertex_indices:
        first = int(raw_first)
        second = int(raw_second)
        if not 0 <= first < len(surface.vertices) or not 0 <= second < len(surface.vertices):
            raise ValueError("Multi-edge bevel contains a vertex index outside the selected surface.")
        geometric = tuple(sorted((topology.raw_to_geometric_vertex[first], topology.raw_to_geometric_vertex[second])))
        if geometric[0] == geometric[1]:
            raise ValueError("Multi-edge bevel cannot include a zero-length edge.")
        if geometric in selected_edges:
            raise ValueError("Multi-edge bevel selection contains the same geometric edge more than once.")
        selected_edges.add(geometric)
        adjacency.setdefault(geometric[0], set()).add(geometric[1])
        adjacency.setdefault(geometric[1], set()).add(geometric[0])
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError("Multi-edge bevel does not accept branched edge selections; select one chain or separate edge sets.")
    endpoints = sorted(vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(endpoints) != 2:
        raise ValueError("Connected multi-edge bevel currently requires an open, non-branching crease chain.")

    ordered_vertices = [endpoints[0]]
    previous = -1
    current = endpoints[0]
    while len(ordered_vertices) <= len(selected_edges):
        following = sorted(value for value in adjacency[current] if value != previous)
        if not following:
            break
        next_vertex = following[0]
        ordered_vertices.append(next_vertex)
        previous, current = current, next_vertex
    if len(ordered_vertices) != len(selected_edges) + 1 or current != endpoints[1]:
        raise ValueError("Multi-edge bevel selection could not be ordered into one continuous crease.")

    requested = abs(float(amount))
    if requested <= 1.0e-9:
        return primitive
    segment_count = max(1, min(64, int(segments)))
    round_profile = max(0.0, min(1.0, float(profile)))
    miter_mode = str(miter or "auto").strip().lower()
    if miter_mode not in {"auto", "sharp", "patch"}:
        raise ValueError("Multi-edge bevel miter must be Auto, Sharp, or Patch.")
    uv_policy = str(uv_mode or "preserve").strip().lower()
    if uv_policy not in {"preserve", "tiled", "none"}:
        raise ValueError("Bevel UV mode must be Preserve, Tiled, or None.")

    reference_direction: Vec3 | None = None
    reference_normals: tuple[Vec3, Vec3] | None = None
    reference_inwards: tuple[Vec3, Vec3] | None = None
    edge_rows: list[tuple[tuple[int, int], Vec3, float, tuple[tuple[int, Vec3, Vec3, float], ...]]] = []
    maximums: list[float] = []
    for geometric_first, geometric_second in zip(ordered_vertices, ordered_vertices[1:]):
        geometric_edge = tuple(sorted((geometric_first, geometric_second)))
        half_indices = topology.geometric_edge_to_half_edges.get(geometric_edge, ())
        if len(half_indices) != 2:
            raise ValueError(
                "Multi-edge bevel requires every selected edge to be manifold and shared by exactly two faces."
            )
        first_half, second_half = (topology.half_edges[index] for index in half_indices)
        if first_half.twin != second_half.index or second_half.twin != first_half.index:
            raise ValueError("Multi-edge bevel requires consistently wound adjacent faces.")
        start = tuple(topology.geometric_positions[geometric_first])
        end = tuple(topology.geometric_positions[geometric_second])
        vector = _vec_sub(end, start)
        length = _vec_length(vector)
        if length <= 1.0e-8:
            raise ValueError("Multi-edge bevel cannot include a zero-length edge.")
        direction = _vec_scale(vector, 1.0 / length)
        if reference_direction is None:
            reference_direction = direction
        elif _vec_dot(direction, reference_direction) < 0.999:
            raise ValueError(
                "Connected multi-edge bevel currently supports a continuous straight crease; "
                "turning corners need the future corner-miter solver."
            )

        rows: list[tuple[int, Vec3, Vec3, float]] = []
        for half_edge in (first_half, second_half):
            face = surface.faces[half_edge.face]
            normal = topology.face_normals[half_edge.face]
            third_raw = next(
                (
                    raw_index
                    for raw_index in face
                    if topology.raw_to_geometric_vertex[raw_index] not in {geometric_first, geometric_second}
                ),
                -1,
            )
            if third_raw < 0:
                raise ValueError("Bevel adjacent faces must contain a third non-edge vertex.")
            inward = _vec_normalized(_vec_cross(normal, direction), fallback=(0.0, 0.0, 0.0))
            if _vec_dot(inward, _vec_sub(surface.vertices[third_raw], start)) < 0.0:
                inward = _vec_scale(inward, -1.0)
            altitude = abs(_vec_dot(inward, _vec_sub(surface.vertices[third_raw], start)))
            rows.append((half_edge.face, normal, inward, altitude))
        if abs(_vec_dot(rows[0][1], rows[1][1])) > 0.999:
            raise ValueError("The selected chain contains a coplanar edge; select only hard crease edges.")
        if reference_normals is None:
            reference_normals = (rows[0][1], rows[1][1])
            reference_inwards = (rows[0][2], rows[1][2])
        else:
            direct = _vec_dot(rows[0][1], reference_normals[0]) > 0.999 and _vec_dot(rows[1][1], reference_normals[1]) > 0.999
            swapped = _vec_dot(rows[0][1], reference_normals[1]) > 0.999 and _vec_dot(rows[1][1], reference_normals[0]) > 0.999
            if swapped:
                rows.reverse()
            elif not direct:
                raise ValueError(
                    "Connected multi-edge bevel requires the same two surface planes along the full crease."
                )
            assert reference_inwards is not None
            if any(_vec_dot(rows[index][2], reference_inwards[index]) < 0.999 for index in range(2)):
                raise ValueError("Connected multi-edge bevel found inconsistent face interiors along the crease.")
        maximums.append(max(1.0e-5, min(row[3] for row in rows) * 0.45))
        edge_rows.append(((geometric_first, geometric_second), direction, length, tuple(rows)))

    assert reference_normals is not None and reference_inwards is not None and reference_direction is not None
    maximum = min(maximums)
    if not clamp_overlap and requested > maximum:
        raise ValueError(
            f"Bevel width {requested:.4f}m would invert a chain face; maximum safe width is {maximum:.4f}m."
        )
    distance = min(requested, maximum) if clamp_overlap else requested
    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    source_face_mats = (
        tuple(int(value) for value in surface.face_mats)
        if len(surface.face_mats) == len(surface.faces)
        else (0,) * len(surface.faces)
    )
    chain_vertices = set(ordered_vertices)
    rewritten_faces = list(faces)
    replacement_vertices: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(surface.faces):
        side = next(
            (index for index, normal in enumerate(reference_normals) if _vec_dot(topology.face_normals[face_index], normal) > 0.999),
            -1,
        )
        if side < 0:
            continue
        replacement_face: list[int] = []
        changed = False
        for raw_index in face:
            geometric = topology.raw_to_geometric_vertex[raw_index]
            if geometric not in chain_vertices:
                replacement_face.append(raw_index)
                continue
            key = (raw_index, side)
            replacement_index = replacement_vertices.get(key)
            if replacement_index is None:
                replacement_index = len(vertices)
                replacement_vertices[key] = replacement_index
                vertices.append(_vec_add(surface.vertices[raw_index], _vec_scale(reference_inwards[side], distance)))
                uvs.append(uvs[raw_index])
                normals.append(normals[raw_index])
                if uvs_lm:
                    uvs_lm.append(uvs_lm[raw_index])
            replacement_face.append(replacement_index)
            changed = True
        if changed:
            rewritten_faces[face_index] = tuple(replacement_face)

    def _raw_channel_index(geometric_vertex: int, side: int) -> int:
        for face_index, face in enumerate(surface.faces):
            if _vec_dot(topology.face_normals[face_index], reference_normals[side]) <= 0.999:
                continue
            for raw_index in face:
                if topology.raw_to_geometric_vertex[raw_index] == geometric_vertex:
                    return raw_index
        raise ValueError("Continuous bevel could not preserve a selected vertex attribute seam.")

    tile = matched_uv_tile_size(primitive, texture=surface.texture)
    dihedral_degrees = math.degrees(
        math.acos(max(-1.0, min(1.0, _vec_dot(reference_normals[0], reference_normals[1]))))
    )
    smooth_strip = float(smoothing_angle_degrees) > 0.0 and dihedral_degrees <= float(smoothing_angle_degrees)
    resolved_miter = ("patch" if segment_count > 1 else "sharp") if miter_mode == "auto" else miter_mode

    def _offset_direction(t: float) -> Vec3:
        linear = _vec_lerp(reference_inwards[0], reference_inwards[1], t)
        center = _vec_add(reference_inwards[0], reference_inwards[1])
        arc = _vec_sub(center, _vec_slerp(reference_inwards[0], reference_inwards[1], 1.0 - t))
        return _vec_lerp(linear, arc, round_profile)

    rail_vertices: dict[tuple[int, int], int] = {}
    cumulative_distance: dict[int, float] = {ordered_vertices[0]: 0.0}
    total_distance = 0.0
    for (geometric_first, geometric_second), _direction, length, _rows in edge_rows:
        total_distance += length
        cumulative_distance[geometric_second] = total_distance
    for geometric in ordered_vertices:
        point = tuple(topology.geometric_positions[geometric])
        raw_by_side = (_raw_channel_index(geometric, 0), _raw_channel_index(geometric, 1))
        for segment_index in range(segment_count + 1):
            t = segment_index / float(segment_count)
            index = len(vertices)
            rail_vertices[(geometric, segment_index)] = index
            vertices.append(_vec_add(point, _vec_scale(_offset_direction(t), distance)))
            if uv_policy == "preserve":
                first_uv, second_uv = uvs[raw_by_side[0]], uvs[raw_by_side[1]]
                uvs.append(
                    (
                        first_uv[0] + ((second_uv[0] - first_uv[0]) * t),
                        first_uv[1] + ((second_uv[1] - first_uv[1]) * t),
                    )
                )
            elif uv_policy == "tiled":
                uvs.append((cumulative_distance[geometric] / max(1.0e-6, tile), (distance * t) / max(1.0e-6, tile)))
            else:
                uvs.append((0.0, 0.0))
            normals.append(
                _vec_slerp(reference_normals[0], reference_normals[1], t)
                if smooth_strip
                else _vec_normalized(_vec_add(reference_normals[0], reference_normals[1]), fallback=reference_normals[0])
            )
            if uvs_lm:
                first_lm, second_lm = uvs_lm[raw_by_side[0]], uvs_lm[raw_by_side[1]]
                uvs_lm.append(
                    (
                        first_lm[0] + ((second_lm[0] - first_lm[0]) * t),
                        first_lm[1] + ((second_lm[1] - first_lm[1]) * t),
                    )
                )

    generated_faces: list[Face] = []
    generated_mats: list[int] = []

    def _append_oriented(first: int, second: int, third: int, desired_normal: Vec3, material: int) -> None:
        actual = _vec_cross(_vec_sub(vertices[second], vertices[first]), _vec_sub(vertices[third], vertices[first]))
        if _vec_length(actual) <= 1.0e-10:
            raise ValueError("Multi-edge bevel generated a degenerate triangle; reduce width or segments.")
        if _vec_dot(actual, desired_normal) < 0.0:
            second, third = third, second
        generated_faces.append((first, second, third))
        generated_mats.append(int(material))

    def _append_wound(first: int, second: int, third: int, material: int) -> None:
        actual = _vec_cross(_vec_sub(vertices[second], vertices[first]), _vec_sub(vertices[third], vertices[first]))
        if _vec_length(actual) <= 1.0e-10:
            raise ValueError("Multi-edge bevel generated a degenerate triangle; reduce width or segments.")
        generated_faces.append((first, second, third))
        generated_mats.append(int(material))

    for (geometric_first, geometric_second), _direction, _length, rows in edge_rows:
        material = source_face_mats[rows[0][0]]
        side_half = next(
            topology.half_edges[index]
            for index in topology.geometric_edge_to_half_edges[tuple(sorted((geometric_first, geometric_second)))]
            if topology.half_edges[index].face == rows[0][0]
        )
        side_runs_forward = (
            side_half.geometric_origin == geometric_first
            and side_half.geometric_destination == geometric_second
        )
        for segment_index in range(segment_count):
            start0 = rail_vertices[(geometric_first, segment_index)]
            end0 = rail_vertices[(geometric_second, segment_index)]
            start1 = rail_vertices[(geometric_first, segment_index + 1)]
            end1 = rail_vertices[(geometric_second, segment_index + 1)]
            if side_runs_forward:
                _append_wound(end0, start0, start1, material)
                _append_wound(end0, start1, end1, material)
            else:
                _append_wound(start0, end0, end1, material)
                _append_wound(start0, end1, start1, material)

    for endpoint_index, geometric in enumerate((ordered_vertices[0], ordered_vertices[-1])):
        track = tuple(rail_vertices[(geometric, index)] for index in range(segment_count + 1))
        point = tuple(topology.geometric_positions[geometric])
        average = tuple(sum(vertices[index][axis] for index in track) / len(track) for axis in range(3))
        cap_point = _vec_lerp(point, average, 0.5) if resolved_miter == "patch" else point
        center = len(vertices)
        vertices.append(cap_point)
        uvs.append(tuple(sum(uvs[index][axis] for index in track) / len(track) for axis in range(2)))
        cap_normal = _vec_scale(reference_direction, -1.0 if endpoint_index == 0 else 1.0)
        normals.append(cap_normal)
        if uvs_lm:
            uvs_lm.append(tuple(sum(uvs_lm[index][axis] for index in track) / len(track) for axis in range(2)))
        cap_material = source_face_mats[edge_rows[0 if endpoint_index == 0 else -1][3][0][0]]
        endpoint_edge = edge_rows[0 if endpoint_index == 0 else -1]
        endpoint_first, endpoint_second = endpoint_edge[0]
        endpoint_side_half = next(
            topology.half_edges[index]
            for index in topology.geometric_edge_to_half_edges[tuple(sorted(endpoint_edge[0]))]
            if topology.half_edges[index].face == endpoint_edge[3][0][0]
        )
        endpoint_side_runs_forward = (
            endpoint_side_half.geometric_origin == endpoint_first
            and endpoint_side_half.geometric_destination == endpoint_second
        )
        for segment_index in range(segment_count):
            first = rail_vertices[(geometric, segment_index)]
            second = rail_vertices[(geometric, segment_index + 1)]
            reverse = endpoint_side_runs_forward if endpoint_index == 0 else not endpoint_side_runs_forward
            if reverse:
                _append_wound(center, second, first, cap_material)
            else:
                _append_wound(center, first, second, cap_material)

    if not smooth_strip:
        hardened_faces: list[Face] = []
        hardened_mats: list[int] = []
        for face, material in zip(generated_faces, generated_mats):
            first, second, third = face
            face_normal = _vec_normalized(
                _vec_cross(_vec_sub(vertices[second], vertices[first]), _vec_sub(vertices[third], vertices[first])),
                fallback=(0.0, 0.0, 1.0),
            )
            hardened: list[int] = []
            for source_index in face:
                hardened.append(len(vertices))
                vertices.append(vertices[source_index])
                uvs.append(uvs[source_index])
                normals.append(face_normal)
                if uvs_lm:
                    uvs_lm.append(uvs_lm[source_index])
            hardened_faces.append(tuple(hardened))
            hardened_mats.append(material)
        generated_faces = hardened_faces
        generated_mats = hardened_mats

    generated_face_start = len(rewritten_faces)
    rewritten_faces.extend(generated_faces)
    rebuilt = replace(
        surface,
        vertices=tuple(vertices),
        faces=tuple(rewritten_faces),
        face_mats=source_face_mats + tuple(generated_mats),
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    rebuilt = _compact_surface(rebuilt, list(rebuilt.faces))
    audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
    if audit.degenerate_faces or audit.non_manifold_edges or audit.inconsistent_winding_edges:
        raise ValueError(
            "Multi-edge bevel would create invalid topology "
            f"({len(audit.degenerate_faces)} degenerate face(s), "
            f"{len(audit.non_manifold_edges)} non-manifold edge(s), "
            f"{len(audit.inconsistent_winding_edges)} winding conflict(s))."
        )
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    metadata = {
        **dict(primitive.metadata),
        "last_topology_edit": {
            "operation": "edge_bevel",
            "selection_kind": "continuous_crease_chain",
            "mesh_role": mesh_role,
            "source_edges": [list(edge) for edge in edge_vertex_indices],
            "source_edge_count": len(edge_vertex_indices),
            "width": distance,
            "requested_width": requested,
            "segments": segment_count,
            "profile": round_profile,
            "miter": miter_mode,
            "resolved_miter": resolved_miter,
            "smoothing_angle_degrees": float(smoothing_angle_degrees),
            "smooth_strip": smooth_strip,
            "uv_mode": uv_policy,
            "generated_face_start": generated_face_start,
            "generated_face_count": len(generated_faces),
            "walkmesh_policy": "requires_review",
        },
    }
    return replace(primitive, surfaces=tuple(surfaces), metadata=metadata)


def bevel_imported_mesh_edges(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    amount: float = 0.25,
    *,
    segments: int = 1,
    profile: float = 0.5,
    miter: str = "auto",
    smoothing_angle_degrees: float = 180.0,
    uv_mode: str = "preserve",
    clamp_overlap: bool = True,
    options: ImportedMeshBevelOptions | None = None,
) -> ImportedMeshRoomPrimitive:
    """Atomically bevel a valid set of manifold hard edges.

    Pairwise-disconnected edges are evaluated as independent one-rings while
    preserving the authoritative single-edge implementation.  Connected
    selections are accepted when they form a straight non-branching crease;
    their intermediate cross-sections are shared so no per-edge caps overlap.
    Turning/branched selections are rejected before any result is returned.
    """

    if options is not None:
        amount = options.width
        segments = options.segments
        profile = options.profile
        miter = options.miter
        smoothing_angle_degrees = options.smoothing_angle_degrees
        uv_mode = options.uv_mode
        clamp_overlap = options.clamp_overlap
    clean_edges = tuple(tuple(int(value) for value in tuple(edge)[:2]) for edge in tuple(edge_vertex_indices))
    if not clean_edges:
        raise ValueError("Multi-edge bevel requires at least one selected edge.")
    if any(len(edge) != 2 or edge[0] == edge[1] for edge in clean_edges):
        raise ValueError("Multi-edge bevel requires distinct vertex pairs.")
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    topology = MeshTopology.build(surface.vertices, surface.faces)

    geometric_edges: list[tuple[int, int]] = []
    original_positions: dict[tuple[int, int], tuple[Vec3, Vec3]] = {}
    for first, second in clean_edges:
        if not 0 <= first < len(surface.vertices) or not 0 <= second < len(surface.vertices):
            raise ValueError("Multi-edge bevel contains a vertex index outside the selected surface.")
        geometric = tuple(sorted((topology.raw_to_geometric_vertex[first], topology.raw_to_geometric_vertex[second])))
        if geometric[0] == geometric[1]:
            raise ValueError("Multi-edge bevel cannot include a zero-length edge.")
        if geometric in geometric_edges:
            raise ValueError("Multi-edge bevel selection contains the same geometric edge more than once.")
        half_indices = topology.geometric_edge_to_half_edges.get(geometric, ())
        if len(half_indices) != 2:
            raise ValueError(
                "Multi-edge bevel requires every selected edge to be manifold and shared by exactly two faces."
            )
        first_half, second_half = (topology.half_edges[index] for index in half_indices)
        if first_half.twin != second_half.index or second_half.twin != first_half.index:
            raise ValueError("Multi-edge bevel requires consistently wound adjacent faces.")
        if abs(_vec_dot(topology.face_normals[first_half.face], topology.face_normals[second_half.face])) > 0.999:
            raise ValueError("Multi-edge bevel selection contains a coplanar edge instead of a hard crease.")
        geometric_edges.append(geometric)
        original_positions[geometric] = (
            tuple(topology.geometric_positions[geometric[0]]),
            tuple(topology.geometric_positions[geometric[1]]),
        )

    edge_neighbors: dict[int, set[int]] = {index: set() for index in range(len(geometric_edges))}
    for first_index, first_edge in enumerate(geometric_edges):
        for second_index in range(first_index + 1, len(geometric_edges)):
            if set(first_edge) & set(geometric_edges[second_index]):
                edge_neighbors[first_index].add(second_index)
                edge_neighbors[second_index].add(first_index)
    components: list[tuple[int, ...]] = []
    remaining = set(range(len(geometric_edges)))
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(edge_neighbors[current] - component, reverse=True))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))

    # Validate every connected chain from the immutable source before applying
    # any disconnected component.  The returned primitive is therefore all or
    # nothing even if a later edge set is malformed.
    for component in components:
        if len(component) > 1:
            _bevel_imported_mesh_continuous_crease(
                primitive,
                mesh_role,
                tuple(clean_edges[index] for index in component),
                amount,
                segments=segments,
                profile=profile,
                miter=miter,
                smoothing_angle_degrees=smoothing_angle_degrees,
                uv_mode=uv_mode,
                clamp_overlap=clamp_overlap,
            )

    result = primitive
    component_kinds: list[str] = []
    for component in components:
        current_surface = result.surfaces[surface_index]
        if len(component) == 1:
            geometric = geometric_edges[component[0]]
            start, end = original_positions[geometric]
            face_index, edge_corners = _locate_imported_mesh_edge_by_positions(current_surface, start, end)
            result = bevel_imported_mesh_edge(
                result,
                mesh_role,
                face_index,
                edge_corners,
                amount,
                segments=segments,
                profile=profile,
                miter=miter,
                smoothing_angle_degrees=smoothing_angle_degrees,
                uv_mode=uv_mode,
                clamp_overlap=clamp_overlap,
            )
            component_kinds.append("independent_edge")
            continue
        remapped_edges: list[tuple[int, int]] = []
        current_topology = MeshTopology.build(current_surface.vertices, current_surface.faces)
        position_to_geometric = {
            _position_key(position): index for index, position in enumerate(current_topology.geometric_positions)
        }
        for source_index in component:
            source_start, source_end = original_positions[geometric_edges[source_index]]
            start_geometric = position_to_geometric.get(_position_key(source_start), -1)
            end_geometric = position_to_geometric.get(_position_key(source_end), -1)
            if start_geometric < 0 or end_geometric < 0:
                raise ValueError("A selected crease changed before the atomic multi-edge edit could be evaluated.")
            remapped_edges.append(
                (
                    current_topology.geometric_to_raw_vertices[start_geometric][0],
                    current_topology.geometric_to_raw_vertices[end_geometric][0],
                )
            )
        result = _bevel_imported_mesh_continuous_crease(
            result,
            mesh_role,
            tuple(remapped_edges),
            amount,
            segments=segments,
            profile=profile,
            miter=miter,
            smoothing_angle_degrees=smoothing_angle_degrees,
            uv_mode=uv_mode,
            clamp_overlap=clamp_overlap,
        )
        component_kinds.append("continuous_crease_chain")

    final_surface = result.surfaces[surface_index]
    audit = MeshTopology.build(final_surface.vertices, final_surface.faces).validate_manifold_state()
    if audit.degenerate_faces or audit.non_manifold_edges or audit.inconsistent_winding_edges:
        raise ValueError(
            "Atomic multi-edge bevel produced invalid topology "
            f"({len(audit.degenerate_faces)} degenerate face(s), "
            f"{len(audit.non_manifold_edges)} non-manifold edge(s), "
            f"{len(audit.inconsistent_winding_edges)} winding conflict(s))."
        )
    metadata = {
        **dict(result.metadata),
        "last_topology_edit": {
            **dict(result.metadata.get("last_topology_edit") or {}),
            "operation": "multi_edge_bevel",
            "mesh_role": mesh_role,
            "source_edges": [list(edge) for edge in clean_edges],
            "source_edge_count": len(clean_edges),
            "component_count": len(components),
            "component_kinds": component_kinds,
            "atomic": True,
            "walkmesh_policy": "requires_review",
        },
    }
    return replace(result, metadata=metadata)


def weld_imported_mesh_vertex(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    vertex_corner: int,
    *,
    max_distance: float = 0.5,
) -> ImportedMeshRoomPrimitive:
    """Snap a vertex onto its nearest neighbor position; drops collapsed faces."""

    surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, vertex_corner=vertex_corner)
    source = positions[0]
    best: Vec3 | None = None
    best_distance = float(max_distance)
    for candidate_surface in primitive.surfaces:
        for vertex in candidate_surface.vertices:
            if _position_key(vertex) == source:
                continue
            distance = _vec_length(_vec_sub(vertex, source))
            if distance < best_distance:
                best_distance = distance
                best = tuple(vertex)
    if best is None:
        raise ValueError(f"No weld target within {float(max_distance):.2f}m of the vertex.")
    return _snap_positions(primitive, [source], best)


def weld_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    source_vertex_index: int,
    target_vertex_index: int | None = None,
    *,
    target_mesh_role: str | None = None,
    target_position: Vec3 | None = None,
) -> ImportedMeshRoomPrimitive:
    """Target-weld one raw vertex position and all of its seam copies.

    The indices are surface-global raw vertex indices rather than triangle
    corners.  Exactly one target form is accepted: another raw vertex, which
    may live on a different texture surface, or an explicit room-local point.
    UV, lightmap UV, normal, and material seam copies remain separate channel
    records; only their co-located geometric position is welded.
    """

    source_surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if source_surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    source_surface = primitive.surfaces[source_surface_index]
    source_index = int(source_vertex_index)
    if not 0 <= source_index < len(source_surface.vertices):
        raise ValueError(
            f"Vertex index {source_vertex_index} out of range for surface {mesh_role} "
            f"({len(source_surface.vertices)} vertices)."
        )
    if (target_vertex_index is None) == (target_position is None):
        raise ValueError("Target weld requires exactly one target vertex or target position.")

    clean_target_role = str(target_mesh_role or mesh_role)
    resolved_target_index = -1
    if target_vertex_index is not None:
        target_surface_index = imported_mesh_surface_index_for_role(primitive, clean_target_role)
        if target_surface_index < 0:
            raise ValueError(f"Unknown imported mesh target surface role: {clean_target_role!r}")
        target_surface = primitive.surfaces[target_surface_index]
        resolved_target_index = int(target_vertex_index)
        if not 0 <= resolved_target_index < len(target_surface.vertices):
            raise ValueError(
                f"Target vertex index {target_vertex_index} out of range for surface "
                f"{clean_target_role} ({len(target_surface.vertices)} vertices)."
            )
        target = tuple(float(value) for value in target_surface.vertices[resolved_target_index])
    else:
        values = tuple(float(value) for value in tuple(target_position or ())[:3])
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("Target weld position must contain three finite coordinates.")
        target = values

    source = _position_key(source_surface.vertices[source_index])
    if source == _position_key(target):
        raise ValueError("Target weld source and target already occupy the same geometric position.")
    welded = _snap_positions(primitive, [source], target)
    return _record_topology_edit(
        welded,
        "target_weld",
        mesh_role=mesh_role,
        source_vertex=source_index,
        source_position=list(source),
        target_mesh_role=clean_target_role if target_vertex_index is not None else "",
        target_vertex=resolved_target_index,
        target_position=[float(value) for value in target],
        seam_policy="position_welded_channels_preserved",
    )


def merge_imported_mesh_components(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    vertex_indices: tuple[int, ...] | list[int] = (),
    *,
    border_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
    threshold: float = 0.01,
) -> ImportedMeshRoomPrimitive:
    """Merge selected vertices or exactly two border edges without losing seams.

    This is the safe, deterministic subset of Maya's ``Merge`` command used by
    Map Studio.  Vertex mode clusters selected *geometric positions* whose
    distance is within ``threshold`` and moves each cluster to its unweighted
    centroid.  UV0, lightmap UV, normal, and material seam copies stay as
    independent Odyssey records at that shared geometric position.

    Edge mode accepts exactly two raw border edges on the same imported
    surface.  Their endpoints are paired by the smallest deterministic total
    distance, then each pair is centroid-merged.  This stitches compatible
    shell borders while refusing a pairing beyond the threshold.

    Faces collapsed by a merge are removed with their aligned material rows.
    The operation is atomic: empty, duplicate, branched-boundary,
    inconsistent-winding, or non-manifold output is refused before KMAP can be
    updated.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    tolerance = float(threshold)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Merge threshold must be a finite value greater than or equal to zero.")

    selected_vertices = tuple(sorted({int(value) for value in tuple(vertex_indices or ())}))
    selected_edges = tuple(tuple(int(value) for value in tuple(edge)) for edge in tuple(border_edges or ()))
    if bool(selected_vertices) == bool(selected_edges):
        raise ValueError("Merge requires either selected vertices or exactly two selected border edges.")

    def _validate_vertex(index: int, label: str = "Vertex") -> None:
        if not 0 <= int(index) < len(surface.vertices):
            raise ValueError(
                f"{label} index {index} out of range for surface {mesh_role} "
                f"({len(surface.vertices)} vertices)."
            )

    merge_mode = "vertices"
    selected_payload: list[Any]
    position_groups: list[tuple[tuple[float, float, float], ...]] = []
    if selected_vertices:
        if len(selected_vertices) < 2:
            raise ValueError("Merge requires at least two selected raw vertices.")
        for index in selected_vertices:
            _validate_vertex(index)

        representatives: dict[tuple[float, float, float], Vec3] = {}
        for index in selected_vertices:
            point = tuple(float(value) for value in surface.vertices[index])
            key = _position_key(point)
            previous = representatives.get(key)
            if previous is None or point < previous:
                representatives[key] = point
        keys = tuple(sorted(representatives))
        parent = list(range(len(keys)))

        def _find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def _union(first: int, second: int) -> None:
            root_a, root_b = _find(first), _find(second)
            if root_a == root_b:
                return
            if root_a > root_b:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a

        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                if _vec_length(_vec_sub(representatives[keys[first]], representatives[keys[second]])) <= tolerance + 1.0e-12:
                    _union(first, second)
        components: dict[int, list[tuple[float, float, float]]] = {}
        for index, key in enumerate(keys):
            components.setdefault(_find(index), []).append(key)
        position_groups = [tuple(sorted(group)) for group in components.values() if len(group) >= 2]
        position_groups.sort()
        if not position_groups:
            raise ValueError(
                f"No selected vertices are within the {tolerance:.6g}m merge threshold."
            )
        selected_payload = list(selected_vertices)
    else:
        merge_mode = "border_edges"
        if len(selected_edges) != 2 or any(len(edge) != 2 for edge in selected_edges):
            raise ValueError("Border-edge Merge requires exactly two two-vertex edges.")
        topology = MeshTopology.build(surface.vertices, surface.faces)
        normalized_edges: list[tuple[int, int]] = []
        for edge_index, edge in enumerate(selected_edges, start=1):
            first, second = edge
            _validate_vertex(first, f"Border edge {edge_index} first vertex")
            _validate_vertex(second, f"Border edge {edge_index} second vertex")
            if first == second:
                raise ValueError(f"Border edge {edge_index} collapses to one raw vertex.")
            normalized = (first, second) if first < second else (second, first)
            if len(topology.edge_to_faces.get(normalized, ())) != 1:
                raise ValueError(
                    f"Selected edge {edge_index} is not one unambiguous raw border edge on {mesh_role}."
                )
            normalized_edges.append(normalized)
        normalized_edges.sort()
        if normalized_edges[0] == normalized_edges[1]:
            raise ValueError("Border-edge Merge requires two different border edges.")
        if set(normalized_edges[0]) & set(normalized_edges[1]):
            raise ValueError("Border-edge Merge requires two disjoint border edges.")

        first_edge, second_edge = normalized_edges
        candidates = (
            ((first_edge[0], second_edge[0]), (first_edge[1], second_edge[1])),
            ((first_edge[0], second_edge[1]), (first_edge[1], second_edge[0])),
        )

        def _candidate_key(candidate: tuple[tuple[int, int], tuple[int, int]]) -> tuple[Any, ...]:
            distances = tuple(
                _vec_length(_vec_sub(surface.vertices[first], surface.vertices[second]))
                for first, second in candidate
            )
            return (math.fsum(distances), max(distances), tuple(tuple(pair) for pair in candidate))

        pairing = min(candidates, key=_candidate_key)
        distances = tuple(
            _vec_length(_vec_sub(surface.vertices[first], surface.vertices[second]))
            for first, second in pairing
        )
        if any(distance > tolerance + 1.0e-12 for distance in distances):
            raise ValueError(
                "The closest border-edge endpoint pairing exceeds the Merge threshold "
                f"({max(distances):.6g}m > {tolerance:.6g}m)."
            )
        seen_keys: set[tuple[float, float, float]] = set()
        for first, second in pairing:
            group = tuple(sorted({_position_key(surface.vertices[first]), _position_key(surface.vertices[second])}))
            if len(group) < 2:
                continue
            if any(key in seen_keys for key in group):
                raise ValueError("Border-edge Merge pairing would fold an endpoint across both edge pairs.")
            seen_keys.update(group)
            position_groups.append(group)
        if not position_groups:
            raise ValueError("The selected border edges are already geometrically coincident.")
        selected_payload = [list(first_edge), list(second_edge)]

    grouped_position_keys = {value for group in position_groups for value in group}
    representatives: dict[tuple[float, float, float], Vec3] = {}
    for candidate_surface in primitive.surfaces:
        for point in candidate_surface.vertices:
            key = _position_key(point)
            if key not in grouped_position_keys:
                continue
            precise = tuple(float(value) for value in point)
            previous = representatives.get(key)
            if previous is None or precise < previous:
                representatives[key] = precise

    position_targets: dict[tuple[float, float, float], Vec3] = {}
    merge_records: list[dict[str, Any]] = []
    for group in position_groups:
        points = tuple(representatives[key] for key in sorted(group))
        centroid = tuple(math.fsum(point[axis] for point in points) / len(points) for axis in range(3))
        for key in group:
            position_targets[key] = centroid
        merge_records.append(
            {
                "source_positions": [list(representatives[key]) for key in sorted(group)],
                "centroid": list(centroid),
            }
        )

    rebuilt_surfaces: list[ImportedMeshSurface] = []
    changed_roles: list[str] = []
    dropped_by_role: dict[str, int] = {}
    for index, candidate_surface in enumerate(primitive.surfaces):
        vertices = tuple(
            position_targets.get(_position_key(point), tuple(point))
            for point in candidate_surface.vertices
        )
        if vertices == candidate_surface.vertices:
            rebuilt_surfaces.append(candidate_surface)
            continue
        role = imported_mesh_surface_role(index)
        changed_roles.append(role)
        kept_indices: list[int] = []
        for face_index, face in enumerate(candidate_surface.faces):
            points = tuple(vertices[raw] for raw in face)
            if len({_position_key(point) for point in points}) != 3:
                continue
            area_vector = _vec_cross(_vec_sub(points[1], points[0]), _vec_sub(points[2], points[0]))
            if _vec_length(area_vector) <= 1.0e-10:
                continue
            kept_indices.append(face_index)
        dropped_by_role[role] = len(candidate_surface.faces) - len(kept_indices)
        if not kept_indices:
            continue
        compacted = compact_indexed_mesh(
            vertices,
            candidate_surface.faces,
            vertex_channels={
                "uvs": candidate_surface.uvs,
                "normals": candidate_surface.normals,
                "uvs_lm": candidate_surface.uvs_lm,
            },
            kept_face_indices=kept_indices,
        )
        materials = _surface_face_materials(candidate_surface)
        rebuilt = replace(
            candidate_surface,
            vertices=compacted.vertices,
            faces=tuple(tuple(int(value) for value in face) for face in compacted.faces),
            face_mats=tuple(materials[index] for index in kept_indices),
            uvs=tuple(compacted.vertex_channels.get("uvs", ())),
            normals=tuple(compacted.vertex_channels.get("normals", ())),
            uvs_lm=tuple(compacted.vertex_channels.get("uvs_lm", ())),
        )
        audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
        if (
            audit.invalid_faces
            or audit.degenerate_faces
            or audit.non_manifold_edges
            or audit.duplicate_faces
            or audit.inconsistent_winding_edges
            or audit.branched_boundaries
        ):
            raise ValueError(
                "Merge would create invalid topology "
                f"on {role}: {len(audit.non_manifold_edges)} non-manifold edge(s), "
                f"{len(audit.branched_boundaries)} branched boundary vertex/vertices, "
                f"{len(audit.duplicate_faces)} duplicate face(s), and "
                f"{len(audit.inconsistent_winding_edges)} inconsistent-winding edge(s)."
            )
        rebuilt_surfaces.append(rebuilt)
    if not rebuilt_surfaces:
        raise ValueError(f"Merge would leave imported room {primitive.room_resref} empty.")

    edited = replace(primitive, surfaces=tuple(rebuilt_surfaces))
    return _record_topology_edit(
        edited,
        "merge_components",
        mesh_role=mesh_role,
        mode=merge_mode,
        threshold=tolerance,
        selected=selected_payload,
        merged_groups=merge_records,
        changed_mesh_roles=changed_roles,
        dropped_faces_by_mesh_role=dropped_by_role,
        seam_policy="positions_merged_uv_lightmap_normal_records_preserved",
        validation="manifold_or_atomic_refusal",
    )


def connect_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    first_vertex_index: int,
    second_vertex_index: int,
) -> ImportedMeshRoomPrimitive:
    """Connect opposite vertices across a two-triangle region.

    Odyssey stores triangles, while Maya's Connect command operates on the
    logical polygon region.  For a triangulated quad this performs the exact
    equivalent: the old internal diagonal is replaced by the selected one.
    Existing edges are rejected instead of silently changing topology, and a
    material boundary is never crossed implicitly.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    first_raw, second_raw = int(first_vertex_index), int(second_vertex_index)
    for label, index in (("First", first_raw), ("Second", second_raw)):
        if not 0 <= index < len(surface.vertices):
            raise ValueError(f"{label} vertex index {index} out of range for surface {mesh_role}.")
    topology = MeshTopology.build(surface.vertices, surface.faces)
    first_geometric = topology.raw_to_geometric_vertex[first_raw]
    second_geometric = topology.raw_to_geometric_vertex[second_raw]
    if first_geometric == second_geometric:
        raise ValueError("Connect requires two different geometric vertices.")
    selected_edge = tuple(sorted((first_geometric, second_geometric)))
    if selected_edge in topology.geometric_edge_to_faces:
        raise ValueError("The selected vertices are already connected by an edge.")

    materials = _surface_face_materials(surface)
    candidate: tuple[int, int, tuple[int, int]] | None = None
    for face_index, neighbors in sorted(topology.geometric_face_to_faces.items()):
        first_face_geometric = {
            topology.raw_to_geometric_vertex[index] for index in surface.faces[face_index]
        }
        for neighbor in sorted(neighbors):
            if int(neighbor) <= int(face_index):
                continue
            second_face_geometric = {
                topology.raw_to_geometric_vertex[index] for index in surface.faces[int(neighbor)]
            }
            shared = first_face_geometric & second_face_geometric
            combined = first_face_geometric | second_face_geometric
            if (
                len(shared) == 2
                and len(combined) == 4
                and first_geometric in combined
                and second_geometric in combined
                and first_geometric not in shared
                and second_geometric not in shared
                and materials[int(face_index)] == materials[int(neighbor)]
            ):
                candidate = (int(face_index), int(neighbor), tuple(sorted(shared)))
                break
        if candidate is not None:
            break
    if candidate is None:
        # A longer connected coplanar patch uses the same deterministic
        # constraint recovery as the safe Multi-Cut slice.  Import locally to
        # avoid a module cycle: Multi-Cut itself adapts this primitive type.
        from .map_studio_multi_cut import MultiCutAnchor, MultiCutSession

        first_faces = tuple(
            index for index, face in enumerate(surface.faces) if first_raw in face
        )
        second_faces = tuple(
            index for index, face in enumerate(surface.faces) if second_raw in face
        )
        if not first_faces or not second_faces:
            raise ValueError("Connect could not resolve both selected vertices to source faces.")
        valid_evaluations = []
        diagnostics: list[str] = []
        for first_face in first_faces:
            for second_face in second_faces:
                session = MultiCutSession.begin(primitive, mesh_role)
                session = session.add_anchor(MultiCutAnchor.vertex(first_face, first_raw))
                session = session.add_anchor(MultiCutAnchor.vertex(second_face, second_raw))
                candidate_evaluation = session.commit()
                if candidate_evaluation.ok:
                    valid_evaluations.append(candidate_evaluation)
                else:
                    diagnostics.extend(candidate_evaluation.diagnostics)
        if not valid_evaluations:
            detail = diagnostics[0] if diagnostics else "unsupported polygon path"
            raise ValueError(f"Connect could not recover an edge across the selected polygon patch: {detail}")
        evaluation = min(
            valid_evaluations,
            key=lambda row: (
                len(row.affected_faces),
                row.affected_faces,
                row.cut_edges,
                row.result_fingerprint,
            ),
        )
        return _record_topology_edit(
            evaluation.primitive,
            "connect_vertices",
            mesh_role=mesh_role,
            selected_vertices=[first_raw, second_raw],
            affected_faces=list(evaluation.affected_faces),
            generated_cut_edges=[list(edge) for edge in evaluation.cut_edges],
            topology_contract="connected_coplanar_patch_via_multi_cut",
        )

    face_a, face_b, old_diagonal = candidate
    def _raw_for_geometric(geometric: int, preferred_faces: tuple[int, ...]) -> int:
        for preferred in preferred_faces:
            for raw in surface.faces[preferred]:
                if topology.raw_to_geometric_vertex[raw] == geometric:
                    return int(raw)
        return int(topology.geometric_to_raw_vertices[geometric][0])

    shared_a, shared_b = old_diagonal
    raw_a = _raw_for_geometric(first_geometric, (face_a, face_b))
    raw_b = _raw_for_geometric(second_geometric, (face_b, face_a))
    raw_c = _raw_for_geometric(shared_a, (face_a, face_b))
    raw_d = _raw_for_geometric(shared_b, (face_b, face_a))
    desired_normal = _vec_normalized(
        _vec_add(topology.face_normals[face_a], topology.face_normals[face_b]),
        fallback=topology.face_normals[face_a],
    )

    def _oriented_face(a: int, b: int, c: int) -> Face:
        actual = _vec_cross(
            _vec_sub(surface.vertices[b], surface.vertices[a]),
            _vec_sub(surface.vertices[c], surface.vertices[a]),
        )
        return (a, c, b) if _vec_dot(actual, desired_normal) < 0.0 else (a, b, c)

    replacement_a = _oriented_face(raw_a, raw_c, raw_b)
    replacement_b = _oriented_face(raw_a, raw_b, raw_d)
    faces = list(surface.faces)
    faces[face_a] = replacement_a
    faces[face_b] = replacement_b
    audit = MeshTopology.build(surface.vertices, faces).validate_manifold_state()
    if audit.degenerate_faces or audit.non_manifold_edges:
        raise ValueError("Connect would create degenerate or non-manifold topology.")

    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = replace(surface, faces=tuple(faces), face_mats=materials)
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "connect_vertices",
        mesh_role=mesh_role,
        selected_vertices=[first_raw, second_raw],
        replaced_faces=[face_a, face_b],
        old_diagonal=list(old_diagonal),
        new_diagonal=[first_geometric, second_geometric],
    )


def _newell_normal(points: list[Vec3]) -> Vec3:
    nx = ny = nz = 0.0
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        nx += (current[1] - following[1]) * (current[2] + following[2])
        ny += (current[2] - following[2]) * (current[0] + following[0])
        nz += (current[0] - following[0]) * (current[1] + following[1])
    return _vec_normalized((nx, ny, nz), fallback=(0.0, 0.0, 0.0))


def _project_planar_polygon(points: list[Vec3], normal: Vec3) -> list[Vec2]:
    axis = max(range(3), key=lambda index: abs(normal[index]))
    if axis == 0:
        return [(point[1], point[2]) for point in points]
    if axis == 1:
        return [(point[0], point[2]) for point in points]
    return [(point[0], point[1]) for point in points]


def _triangulate_planar_loop(points: list[Vec2]) -> list[tuple[int, int, int]]:
    """Deterministic ear clipping for one simple projected boundary loop."""

    if len(points) < 3:
        raise ValueError("A boundary fill requires at least three vertices.")
    signed_area = sum(
        (points[index][0] * points[(index + 1) % len(points)][1])
        - (points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    )
    if abs(signed_area) <= 1.0e-12:
        raise ValueError("Boundary loop has no stable planar area.")
    orientation = 1.0 if signed_area > 0.0 else -1.0

    def _cross2(a: Vec2, b: Vec2, c: Vec2) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))

    def _inside_triangle(point: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
        first = orientation * _cross2(a, b, point)
        second = orientation * _cross2(b, c, point)
        third = orientation * _cross2(c, a, point)
        return first >= -1.0e-10 and second >= -1.0e-10 and third >= -1.0e-10

    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        clipped = False
        for cursor, current in enumerate(tuple(remaining)):
            previous = remaining[(cursor - 1) % len(remaining)]
            following = remaining[(cursor + 1) % len(remaining)]
            if orientation * _cross2(points[previous], points[current], points[following]) <= 1.0e-10:
                continue
            if any(
                _inside_triangle(points[candidate], points[previous], points[current], points[following])
                for candidate in remaining
                if candidate not in {previous, current, following}
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[cursor]
            clipped = True
            break
        if not clipped:
            raise ValueError("Boundary loop is self-intersecting or too degenerate to fill safely.")
    triangles.append(tuple(remaining))
    return triangles


def _signed_area_2d(points: list[Vec2]) -> float:
    return 0.5 * sum(
        (points[index][0] * points[(index + 1) % len(points)][1])
        - (points[(index + 1) % len(points)][0] * points[index][1])
        for index in range(len(points))
    )


def _cross_2d(a: Vec2, b: Vec2, c: Vec2) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(point: Vec2, first: Vec2, second: Vec2, tolerance: float) -> bool:
    return (
        abs(_cross_2d(first, second, point)) <= tolerance
        and min(first[0], second[0]) - tolerance <= point[0] <= max(first[0], second[0]) + tolerance
        and min(first[1], second[1]) - tolerance <= point[1] <= max(first[1], second[1]) + tolerance
    )


def _segments_intersect_2d(
    first_a: Vec2,
    first_b: Vec2,
    second_a: Vec2,
    second_b: Vec2,
    *,
    tolerance: float = 1.0e-10,
) -> bool:
    first = _cross_2d(first_a, first_b, second_a)
    second = _cross_2d(first_a, first_b, second_b)
    third = _cross_2d(second_a, second_b, first_a)
    fourth = _cross_2d(second_a, second_b, first_b)
    if ((first > tolerance and second < -tolerance) or (first < -tolerance and second > tolerance)) and (
        (third > tolerance and fourth < -tolerance) or (third < -tolerance and fourth > tolerance)
    ):
        return True
    return bool(
        (abs(first) <= tolerance and _point_on_segment_2d(second_a, first_a, first_b, tolerance))
        or (abs(second) <= tolerance and _point_on_segment_2d(second_b, first_a, first_b, tolerance))
        or (abs(third) <= tolerance and _point_on_segment_2d(first_a, second_a, second_b, tolerance))
        or (abs(fourth) <= tolerance and _point_on_segment_2d(first_b, second_a, second_b, tolerance))
    )


def _validate_simple_polygon_2d(points: list[Vec2], *, label: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{label} requires at least three vertices.")
    for index in range(len(points)):
        first_a = points[index]
        first_b = points[(index + 1) % len(points)]
        if abs(first_a[0] - first_b[0]) <= 1.0e-12 and abs(first_a[1] - first_b[1]) <= 1.0e-12:
            raise ValueError(f"{label} contains a zero-length edge.")
        for other in range(index + 1, len(points)):
            if other in {index, (index + 1) % len(points)}:
                continue
            if index == 0 and other == len(points) - 1:
                continue
            second_a = points[other]
            second_b = points[(other + 1) % len(points)]
            if _segments_intersect_2d(first_a, first_b, second_a, second_b):
                raise ValueError(f"{label} is self-intersecting or touches itself.")
    if abs(_signed_area_2d(points)) <= 1.0e-12:
        raise ValueError(f"{label} has no stable planar area.")


def _point_in_polygon_2d(point: Vec2, polygon: list[Vec2]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment_2d(point, previous, current, 1.0e-10):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing = current[0] + (
                ((point[1] - current[1]) * (previous[0] - current[0]))
                / (previous[1] - current[1])
            )
            if crossing > point[0]:
                inside = not inside
        previous = current
    return inside


def _triangle_barycentric(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> tuple[float, float, float]:
    edge_ab = _vec_sub(b, a)
    edge_ac = _vec_sub(c, a)
    offset = _vec_sub(point, a)
    dot00 = _vec_dot(edge_ab, edge_ab)
    dot01 = _vec_dot(edge_ab, edge_ac)
    dot11 = _vec_dot(edge_ac, edge_ac)
    dot20 = _vec_dot(offset, edge_ab)
    dot21 = _vec_dot(offset, edge_ac)
    denominator = (dot00 * dot11) - (dot01 * dot01)
    if abs(denominator) <= 1.0e-18:
        raise ValueError("The selected source triangle is degenerate.")
    weight_b = ((dot11 * dot20) - (dot01 * dot21)) / denominator
    weight_c = ((dot00 * dot21) - (dot01 * dot20)) / denominator
    return (1.0 - weight_b - weight_c, weight_b, weight_c)


def _oriented_triangle(indices: tuple[int, int, int], vertices: list[Vec3], normal: Vec3) -> Face:
    first, second, third = indices
    actual = _vec_cross(
        _vec_sub(vertices[second], vertices[first]),
        _vec_sub(vertices[third], vertices[first]),
    )
    return (first, third, second) if _vec_dot(actual, normal) < 0.0 else (first, second, third)


def _triangulate_triangle_with_hole(
    outer_indices: tuple[int, int, int],
    outer_points: list[Vec2],
    hole_indices: list[int],
    hole_points: list[Vec2],
) -> list[Face]:
    """Triangulate one triangular face with one strictly internal simple hole.

    The hole is connected to a visible outer vertex by a zero-width bridge,
    producing the standard weakly-simple polygon used by ear-clipping
    implementations. Duplicate bridge nodes retain the same raw vertex index;
    no duplicate geometric points are added to the authored mesh.
    """

    outer = list(zip(outer_indices, outer_points))
    if _signed_area_2d([point for _index, point in outer]) < 0.0:
        outer.reverse()
    hole = list(zip(hole_indices, hole_points))
    if _signed_area_2d([point for _index, point in hole]) > 0.0:
        hole.reverse()

    visible: list[tuple[float, int, int]] = []
    for outer_cursor, (_outer_index, outer_point) in enumerate(outer):
        for hole_cursor, (_hole_index, hole_point) in enumerate(hole):
            blocked = False
            for edge_cursor in range(len(hole)):
                next_cursor = (edge_cursor + 1) % len(hole)
                if hole_cursor in {edge_cursor, next_cursor}:
                    continue
                if _segments_intersect_2d(
                    outer_point,
                    hole_point,
                    hole[edge_cursor][1],
                    hole[next_cursor][1],
                ):
                    blocked = True
                    break
            midpoint = (
                (outer_point[0] + hole_point[0]) * 0.5,
                (outer_point[1] + hole_point[1]) * 0.5,
            )
            approach = (
                (outer_point[0] * 0.001) + (hole_point[0] * 0.999),
                (outer_point[1] * 0.001) + (hole_point[1] * 0.999),
            )
            hole_polygon = [point for _index, point in hole]
            if (
                not blocked
                and not _point_in_polygon_2d(midpoint, hole_polygon)
                and not _point_in_polygon_2d(approach, hole_polygon)
            ):
                distance_squared = (
                    ((outer_point[0] - hole_point[0]) ** 2)
                    + ((outer_point[1] - hole_point[1]) ** 2)
                )
                visible.append((distance_squared, outer_cursor, hole_cursor))
    if not visible:
        raise ValueError("The cutter cannot be connected to the selected triangle without crossing itself.")
    _distance, outer_cursor, hole_cursor = min(visible)
    outer = outer[outer_cursor:] + outer[:outer_cursor]
    hole = hole[hole_cursor:] + hole[:hole_cursor]
    nodes = [outer[0], *hole, hole[0], outer[0], *outer[1:]]
    points = [point for _index, point in nodes]
    if _signed_area_2d(points) <= 1.0e-12:
        raise ValueError("The cutter bridge did not produce a stable ring polygon.")

    remaining = list(range(len(nodes)))
    triangles: list[Face] = []

    def _inside(point: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
        return (
            _cross_2d(a, b, point) >= -1.0e-10
            and _cross_2d(b, c, point) >= -1.0e-10
            and _cross_2d(c, a, point) >= -1.0e-10
        )

    while len(remaining) > 3:
        clipped = False
        for cursor, current in enumerate(tuple(remaining)):
            previous = remaining[(cursor - 1) % len(remaining)]
            following = remaining[(cursor + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            if _cross_2d(a, b, c) <= 1.0e-10:
                continue
            if any(
                _inside(points[candidate], a, b, c)
                for candidate in remaining
                if candidate not in {previous, current, following}
                and points[candidate] not in {a, b, c}
            ):
                continue
            triangle = (nodes[previous][0], nodes[current][0], nodes[following][0])
            if len(set(triangle)) != 3:
                continue
            triangles.append(triangle)
            del remaining[cursor]
            clipped = True
            break
        if not clipped:
            raise ValueError("The cutter ring could not be triangulated without a degenerate bridge.")
    final = tuple(nodes[index][0] for index in remaining)
    if len(set(final)) != 3 or _cross_2d(*(points[index] for index in remaining)) <= 1.0e-10:
        raise ValueError("The cutter ring ended in a degenerate triangle.")
    triangles.append(final)
    expected = len(hole_indices) + 3
    if len(triangles) != expected:
        raise ValueError(
            f"The cutter ring produced {len(triangles)} triangles; {expected} were required for a valid hole."
        )
    point_by_index = {index: point for index, point in (*outer, *hole)}
    covered_area = sum(
        abs(_cross_2d(point_by_index[first], point_by_index[second], point_by_index[third])) * 0.5
        for first, second, third in triangles
    )
    expected_area = abs(_signed_area_2d(outer_points)) - abs(_signed_area_2d(hole_points))
    if not math.isclose(covered_area, expected_area, rel_tol=1.0e-9, abs_tol=1.0e-10):
        raise ValueError("The cutter ring triangulation would overlap or leave an unintended filled region.")
    return triangles


def make_hole_in_imported_mesh_face(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    cutter_points: tuple[Vec3, ...] | list[Vec3] = (),
    *,
    cutter_face_index: int = -1,
    planarity_tolerance: float = 1.0e-4,
    boundary_tolerance: float = 1.0e-6,
) -> ImportedMeshRoomPrimitive:
    """Cut one closed polygonal hole strictly inside one triangle.

    This intentionally refuses cutters that span multiple source triangles.
    A future region-level Boolean may broaden that contract, but silently
    cutting only the selected triangle would leave T-junctions and is not a
    safe Maya-style Make Hole operation.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    selected_face = int(face_index)
    _validate_face_indices(surface, (selected_face,), role=mesh_role)
    cutter_face = int(cutter_face_index)
    if cutter_face >= 0:
        _validate_face_indices(surface, (cutter_face,), role=mesh_role)
        if cutter_face == selected_face:
            raise ValueError("Make Hole requires two different faces of the same polygon object.")
        supplied = [tuple(surface.vertices[index]) for index in surface.faces[cutter_face]]
    else:
        supplied = [tuple(float(value) for value in tuple(point)[:3]) for point in cutter_points]
    if supplied and len(supplied) > 1 and _vec_length(_vec_sub(supplied[0], supplied[-1])) <= 1.0e-9:
        supplied.pop()
    if len(supplied) < 3:
        raise ValueError("Make Hole requires a closed cutter polygon with at least three unique points.")
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in supplied):
        raise ValueError("Make Hole cutter points must contain three finite room-local coordinates.")
    if len({_position_key(point) for point in supplied}) != len(supplied):
        raise ValueError("Make Hole cutter polygon repeats a point.")

    source_vertices = surface.faces[selected_face]
    source_points = [surface.vertices[index] for index in source_vertices]
    source_normal = _face_normal(surface, surface.faces[selected_face])
    if _vec_length(source_normal) <= 1.0e-9:
        raise ValueError("Make Hole cannot use a degenerate source triangle.")
    tolerance = max(0.0, float(planarity_tolerance))
    boundary = max(0.0, float(boundary_tolerance))
    projected: list[Vec3] = []
    barycentric: list[tuple[float, float, float]] = []
    for point in supplied:
        plane_offset = _vec_dot(_vec_sub(point, source_points[0]), source_normal)
        if abs(plane_offset) > tolerance:
            raise ValueError(
                f"Make Hole cutter is not coplanar with the selected triangle "
                f"({abs(plane_offset):.6f}m deviation; {tolerance:.6f}m allowed)."
            )
        on_plane = _vec_sub(point, _vec_scale(source_normal, plane_offset))
        weights = _triangle_barycentric(on_plane, *source_points)
        if min(weights) <= boundary or max(weights) >= 1.0 - boundary:
            raise ValueError(
                "Make Hole cutter must lie strictly inside one selected triangle; "
                "cutters on an edge or spanning multiple triangles are not supported."
            )
        projected.append(on_plane)
        barycentric.append(weights)

    projected_all = _project_planar_polygon(source_points + projected, source_normal)
    outer_2d = projected_all[:3]
    cutter_2d = projected_all[3:]
    _validate_simple_polygon_2d(cutter_2d, label="Make Hole cutter polygon")
    if len(surface.vertices) + len(projected) > MDL_MAX_VERTICES_PER_SURFACE:
        raise ValueError(
            f"Make Hole would exceed KOTOR's {MDL_MAX_VERTICES_PER_SURFACE}-vertex MDL surface limit."
        )

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    cutter_indices: list[int] = []
    for point, weights in zip(projected, barycentric):
        cutter_indices.append(len(vertices))
        vertices.append(point)
        uvs.append(
            tuple(sum(uvs[source_vertices[corner]][axis] * weights[corner] for corner in range(3)) for axis in range(2))
        )
        normals.append(
            _vec_normalized(
                tuple(
                    sum(normals[source_vertices[corner]][axis] * weights[corner] for corner in range(3))
                    for axis in range(3)
                ),
                fallback=source_normal,
            )
        )
        if uvs_lm:
            uvs_lm.append(
                tuple(
                    sum(uvs_lm[source_vertices[corner]][axis] * weights[corner] for corner in range(3))
                    for axis in range(2)
                )
            )

    ring_faces = _triangulate_triangle_with_hole(
        source_vertices,
        outer_2d,
        cutter_indices,
        cutter_2d,
    )
    ring_faces = [_oriented_triangle(face, vertices, source_normal) for face in ring_faces]
    source_materials = _surface_face_materials(surface)
    face_mats = list(source_materials)
    faces[selected_face : selected_face + 1] = ring_faces
    face_mats[selected_face : selected_face + 1] = [source_materials[selected_face]] * len(ring_faces)
    if cutter_face >= 0:
        adjusted_cutter_face = cutter_face + (len(ring_faces) - 1 if cutter_face > selected_face else 0)
        del faces[adjusted_cutter_face]
        del face_mats[adjusted_cutter_face]
    rebuilt = _compact_surface_arrays(surface, vertices, faces, uvs, normals, uvs_lm, face_mats)
    source_audit = MeshTopology.build(surface.vertices, surface.faces).validate_manifold_state()
    audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
    if (
        audit.degenerate_faces
        or len(audit.non_manifold_edges) > len(source_audit.non_manifold_edges)
        or len(audit.inconsistent_winding_edges) > len(source_audit.inconsistent_winding_edges)
        or len(audit.duplicate_faces) > len(source_audit.duplicate_faces)
    ):
        raise ValueError("Make Hole would create degenerate, duplicate, non-manifold, or inconsistently wound topology.")

    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "make_hole",
        mesh_role=mesh_role,
        source_face=selected_face,
        cutter_points=[[float(value) for value in point] for point in projected],
        cutter_vertex_count=len(projected),
        cutter_face=cutter_face if cutter_face >= 0 else None,
        cutter_face_removed=cutter_face >= 0,
        generated_face_start=selected_face,
        generated_face_count=len(ring_faces),
        material=source_materials[selected_face],
        uv_policy="barycentric_source_face",
        lightmap_uv_policy="barycentric_source_face" if uvs_lm else "unavailable",
        scope_limit="strictly_inside_one_triangle",
    )


def _read_logical_quad_provenance(
    primitive: ImportedMeshRoomPrimitive,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Decode the versioned, JSON-safe Quad Draw provenance metadata.

    The metadata is intentionally vertex-index based.  Quad Draw and Insert
    Edge Loop keep every referenced vertex live, so the stable compactor leaves
    these indices unchanged.  Any malformed or unknown version is rejected
    instead of being silently interpreted as trustworthy topology.
    """

    raw = dict(primitive.metadata).get(LOGICAL_QUAD_PROVENANCE_KEY)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Logical quad provenance is malformed; expected a versioned object.")
    try:
        version = int(raw.get("version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Logical quad provenance has an invalid version.") from exc
    if version != LOGICAL_QUAD_PROVENANCE_VERSION:
        raise ValueError(
            "Logical quad provenance version is unsupported "
            f"({version}; expected {LOGICAL_QUAD_PROVENANCE_VERSION})."
        )
    rows_by_role = raw.get("quads_by_role")
    if not isinstance(rows_by_role, dict):
        raise ValueError("Logical quad provenance is malformed; quads_by_role is missing.")
    decoded: dict[str, list[tuple[int, int, int, int]]] = {}
    for raw_role, raw_rows in rows_by_role.items():
        role = str(raw_role or "").strip()
        if not role or not isinstance(raw_rows, (list, tuple)):
            raise ValueError("Logical quad provenance contains an invalid surface role or quad list.")
        decoded_rows: list[tuple[int, int, int, int]] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 4:
                raise ValueError(f"Logical quad provenance for {role!r} contains a non-quad row.")
            try:
                row = tuple(int(value) for value in raw_row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Logical quad provenance for {role!r} contains a non-integer vertex.") from exc
            decoded_rows.append(row)
        decoded[role] = decoded_rows
    return decoded


def _replace_logical_quad_provenance(
    primitive: ImportedMeshRoomPrimitive,
    quads_by_role: dict[str, list[tuple[int, int, int, int]]],
) -> ImportedMeshRoomPrimitive:
    """Persist ordered logical quads without discarding future metadata keys."""

    metadata = dict(primitive.metadata)
    previous = metadata.get(LOGICAL_QUAD_PROVENANCE_KEY)
    payload = dict(previous) if isinstance(previous, dict) else {}
    payload.update(
        {
            "version": LOGICAL_QUAD_PROVENANCE_VERSION,
            "quads_by_role": {
                str(role): [[int(vertex) for vertex in quad] for quad in quads]
                for role, quads in sorted(quads_by_role.items())
            },
        }
    )
    metadata[LOGICAL_QUAD_PROVENANCE_KEY] = payload
    return replace(primitive, metadata=metadata)


def _logical_quad_edge_key(first: int, second: int) -> tuple[int, int]:
    a = int(first)
    b = int(second)
    return (a, b) if a < b else (b, a)


def _logical_quad_boundary_edges(quad: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(_logical_quad_edge_key(quad[index], quad[(index + 1) % 4]) for index in range(4))


def _logical_quad_source_faces(
    surface: ImportedMeshSurface,
    quad: tuple[int, int, int, int],
    *,
    mesh_role: str,
) -> tuple[tuple[int, int], int]:
    """Prove that one provenance row is still exactly two wound triangles."""

    if len(set(quad)) != 4:
        raise ValueError(f"Logical quad provenance for {mesh_role!r} repeats a vertex.")
    if any(vertex < 0 or vertex >= len(surface.vertices) for vertex in quad):
        raise ValueError(f"Logical quad provenance for {mesh_role!r} contains an out-of-range vertex.")
    quad_vertices = set(quad)
    candidate_faces = tuple(
        face_index
        for face_index, face in enumerate(surface.faces)
        if set(face).issubset(quad_vertices)
    )
    if len(candidate_faces) != 2:
        raise ValueError(
            f"Logical quad provenance for {mesh_role!r} is stale: expected exactly two source triangles, "
            f"found {len(candidate_faces)}."
        )
    source_faces = tuple(surface.faces[index] for index in candidate_faces)
    if set(source_faces[0]) | set(source_faces[1]) != quad_vertices:
        raise ValueError(f"Logical quad provenance for {mesh_role!r} does not span four vertices.")
    edge_counts: dict[tuple[int, int], int] = {}
    for face in source_faces:
        for index in range(3):
            edge = _logical_quad_edge_key(face[index], face[(index + 1) % 3])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = {edge for edge, count in edge_counts.items() if count == 1}
    diagonals = {edge for edge, count in edge_counts.items() if count == 2}
    if boundary != set(_logical_quad_boundary_edges(quad)) or len(diagonals) != 1:
        raise ValueError(
            f"Logical quad provenance for {mesh_role!r} no longer matches one triangulated quad perimeter."
        )
    quad_normal = _newell_normal([surface.vertices[index] for index in quad])
    if _vec_length(quad_normal) <= 1.0e-9:
        raise ValueError(f"Logical quad provenance for {mesh_role!r} has a degenerate winding.")
    if any(_vec_dot(_face_normal(surface, surface.faces[index]), quad_normal) <= 1.0e-6 for index in candidate_faces):
        raise ValueError(f"Logical quad provenance for {mesh_role!r} no longer matches source face winding.")
    materials = _surface_face_materials(surface)
    material_values = {materials[index] for index in candidate_faces}
    if len(material_values) != 1:
        raise ValueError(
            f"Logical quad provenance for {mesh_role!r} spans multiple triangle materials and is ambiguous."
        )
    return (candidate_faces[0], candidate_faces[1]), next(iter(material_values))


def append_imported_mesh_quad(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    points: tuple[Vec3, Vec3, Vec3, Vec3] | list[Vec3],
    *,
    material: int | None = None,
    texture: str = "",
    lightmap: str = "",
    normal_hint: Vec3 | None = None,
    planarity_tolerance: float = 1.0e-4,
    auto_weld: bool = True,
    weld_tolerance: float = 1.0e-4,
) -> ImportedMeshRoomPrimitive:
    """Append one Quad Draw polygon as two triangles.

    The four room-local points must already be ordered around the perimeter.
    When ``mesh_role`` does not exist, a new retopology surface is created and
    its actual stable role is recorded in ``last_topology_edit``.
    """

    supplied = [tuple(float(value) for value in tuple(point)[:3]) for point in points]
    if len(supplied) != 4:
        raise ValueError("Quad Draw requires exactly four ordered room-local points.")
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in supplied):
        raise ValueError("Quad Draw points must contain three finite room-local coordinates.")
    if len({_position_key(point) for point in supplied}) != 4:
        raise ValueError("Quad Draw requires four unique points.")
    quad_normal = _newell_normal(supplied)
    if _vec_length(quad_normal) <= 1.0e-9:
        raise ValueError("Quad Draw points have no stable plane or winding.")
    centroid = tuple(sum(point[axis] for point in supplied) / 4.0 for axis in range(3))
    tolerance = max(0.0, float(planarity_tolerance))
    maximum_offset = max(abs(_vec_dot(_vec_sub(point, centroid), quad_normal)) for point in supplied)
    if maximum_offset > tolerance:
        raise ValueError(
            f"Quad Draw points are not planar ({maximum_offset:.6f}m deviation; {tolerance:.6f}m allowed)."
        )
    projected_2d = _project_planar_polygon(supplied, quad_normal)
    _validate_simple_polygon_2d(projected_2d, label="Quad Draw polygon")
    if normal_hint is not None:
        hint = tuple(float(value) for value in tuple(normal_hint)[:3])
        if len(hint) != 3 or not all(math.isfinite(value) for value in hint) or _vec_length(hint) <= 1.0e-9:
            raise ValueError("Quad Draw normal hint must contain three finite non-zero coordinates.")
        desired_normal = _vec_normalized(hint)
        if abs(_vec_dot(desired_normal, quad_normal)) <= 1.0e-6:
            raise ValueError("Quad Draw normal hint cannot be tangent to the quad plane.")
    else:
        desired_normal = quad_normal
    triangles = _triangulate_planar_loop(projected_2d)
    if len(triangles) != 2:
        raise ValueError("Quad Draw could not resolve the four points into one simple quad.")
    generated_uvs = list(planar_uvs_for_vertices(tuple(supplied)))
    quads_by_role = _read_logical_quad_provenance(primitive)
    target_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    created_surface = target_index < 0

    surfaces = list(primitive.surfaces)
    if created_surface:
        target_index = len(surfaces)
        vertex_indices = tuple(range(4))
        generated_faces = [_oriented_triangle(tuple(vertex_indices[index] for index in triangle), supplied, desired_normal) for triangle in triangles]
        clean_texture = str(texture or "NULL").strip() or "NULL"
        clean_lightmap = str(lightmap or "").strip()
        material_value = int(material or 0)
        new_surface = ImportedMeshSurface(
            name=str(mesh_role or f"{primitive.room_resref}_retopo"),
            texture=clean_texture,
            vertices=tuple(supplied),
            faces=tuple(generated_faces),
            face_mats=(material_value, material_value),
            uvs=tuple(generated_uvs),
            normals=(desired_normal,) * 4,
            lightmap=clean_lightmap,
            texture_names=tuple(value for value in (clean_texture, clean_lightmap) if value),
            tex_count=2 if clean_lightmap else 1,
            uvs_lm=tuple(generated_uvs) if clean_lightmap else (),
        )
        audit = MeshTopology.build(new_surface.vertices, new_surface.faces).validate_manifold_state()
        if audit.degenerate_faces or audit.non_manifold_edges or audit.inconsistent_winding_edges or audit.duplicate_faces:
            raise ValueError("Quad Draw would create invalid topology on the new retopology surface.")
        surfaces.append(new_surface)
        generated_start = 0
        lightmap_policy = "planar_placeholder_requires_bake" if clean_lightmap else "none"
    else:
        surface = surfaces[target_index]
        clean_texture = str(texture or "").strip()
        if clean_texture and clean_texture.lower() != str(surface.texture or "").strip().lower():
            raise ValueError(
                "Quad Draw cannot append a different texture to an existing texture surface; "
                "create a separate retopology surface instead."
            )
        vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
        tolerance = max(0.0, float(weld_tolerance))
        vertex_indices: list[int] = []
        reused_vertex_count = 0
        for point, generated_uv in zip(supplied, generated_uvs):
            existing_index = -1
            if auto_weld and tolerance > 0.0:
                candidates = [
                    (_vec_length(_vec_sub(point, existing)), index)
                    for index, existing in enumerate(vertices)
                    if _vec_length(_vec_sub(point, existing)) <= tolerance
                ]
                if candidates:
                    existing_index = min(candidates)[1]
            if existing_index >= 0:
                vertex_indices.append(existing_index)
                reused_vertex_count += 1
                continue
            if len(vertices) >= MDL_MAX_VERTICES_PER_SURFACE:
                raise ValueError(
                    f"Quad Draw would exceed KOTOR's {MDL_MAX_VERTICES_PER_SURFACE}-vertex MDL surface limit."
                )
            vertex_indices.append(len(vertices))
            vertices.append(point)
            uvs.append(generated_uv)
            normals.append(desired_normal)
            if uvs_lm:
                uvs_lm.append(generated_uv)
        lightmap_policy = "planar_placeholder_requires_bake" if uvs_lm else "unavailable"
        generated_start = len(faces)
        generated_faces = [
            _oriented_triangle(tuple(vertex_indices[index] for index in triangle), vertices, desired_normal)
            for triangle in triangles
        ]
        faces.extend(generated_faces)
        source_materials = list(_surface_face_materials(surface))
        material_value = int(material if material is not None else (source_materials[-1] if source_materials else 0))
        face_mats = source_materials + [material_value, material_value]
        # ``compact_indexed_mesh`` preserves the sorted order of used raw
        # vertices, but an imported surface can still contain unused rows.
        # Remap both the new quad and any earlier Quad Draw provenance before
        # the compactor makes those source indices stale.
        used_vertices = sorted({int(vertex) for face in faces for vertex in face})
        old_to_new = {old: new for new, old in enumerate(used_vertices)}
        vertex_indices = [old_to_new[int(vertex)] for vertex in vertex_indices]
        existing_role = imported_mesh_surface_role(target_index)
        if existing_role in quads_by_role:
            try:
                quads_by_role[existing_role] = [
                    tuple(old_to_new[int(vertex)] for vertex in quad)
                    for quad in quads_by_role[existing_role]
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Logical quad provenance for {existing_role!r} references an unused vertex and is stale."
                ) from exc
        rebuilt = _compact_surface_arrays(surface, vertices, faces, uvs, normals, uvs_lm, face_mats)
        source_audit = MeshTopology.build(surface.vertices, surface.faces).validate_manifold_state()
        audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
        if (
            audit.degenerate_faces
            or len(audit.non_manifold_edges) > len(source_audit.non_manifold_edges)
            or len(audit.inconsistent_winding_edges) > len(source_audit.inconsistent_winding_edges)
            or len(audit.duplicate_faces) > len(source_audit.duplicate_faces)
        ):
            raise ValueError(
                "Quad Draw would create degenerate, duplicate, non-manifold, or inconsistently wound topology."
            )
        surfaces[target_index] = rebuilt

    resolved_role = imported_mesh_surface_role(target_index)
    edited = replace(primitive, surfaces=tuple(surfaces))
    ordered_quad = tuple(int(value) for value in vertex_indices)
    if _vec_dot(quad_normal, desired_normal) < 0.0:
        ordered_quad = (ordered_quad[0], ordered_quad[3], ordered_quad[2], ordered_quad[1])
    # Prove the new record against the actual triangles before making it a
    # durable edge-loop authority.  Adjacent auto-welded appends retain shared
    # raw indices, so their provenance joins into one traversable quad strip.
    _logical_quad_source_faces(edited.surfaces[target_index], ordered_quad, mesh_role=resolved_role)
    quads_by_role.setdefault(resolved_role, []).append(ordered_quad)
    edited = _replace_logical_quad_provenance(edited, quads_by_role)
    return _record_topology_edit(
        edited,
        "quad_draw_append",
        requested_mesh_role=str(mesh_role),
        mesh_role=resolved_role,
        created_surface=created_surface,
        generated_face_start=generated_start,
        generated_face_count=2,
        material=material_value,
        normal=[float(value) for value in desired_normal],
        uv_policy="planar_quad",
        lightmap_uv_policy=lightmap_policy,
        maximum_planarity_deviation=maximum_offset,
        auto_weld=bool(auto_weld),
        weld_tolerance=max(0.0, float(weld_tolerance)),
        reused_vertex_count=(0 if created_surface else reused_vertex_count),
        logical_quad=[int(value) for value in ordered_quad],
        logical_quad_provenance_version=LOGICAL_QUAD_PROVENANCE_VERSION,
    )


def insert_imported_mesh_edge_loop(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[int, int] | list[int],
    position: float = 0.5,
) -> ImportedMeshRoomPrimitive:
    """Insert one cut through an unbranched strip of proven Quad Draw quads.

    ``edge_vertex_indices`` is an ordered raw perimeter edge.  Its order makes
    ``position`` deterministic (``0`` is the first vertex, ``1`` the second).
    The cut crosses each logical quad through the opposite edge, exactly as an
    edge-loop operator does.  Arbitrary imported triangles are intentionally
    refused because a diagonal pair does not encode a trustworthy quad ring.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    resolved_role = imported_mesh_surface_role(surface_index)
    surface = primitive.surfaces[surface_index]
    supplied_edge = tuple(int(value) for value in edge_vertex_indices)
    if len(supplied_edge) != 2 or supplied_edge[0] == supplied_edge[1]:
        raise ValueError("Insert Edge Loop requires one ordered raw edge with two different vertices.")
    if any(vertex < 0 or vertex >= len(surface.vertices) for vertex in supplied_edge):
        raise ValueError(f"Insert Edge Loop edge contains an out-of-range vertex for {resolved_role!r}.")
    amount = float(position)
    if not math.isfinite(amount) or not 1.0e-6 < amount < 1.0 - 1.0e-6:
        raise ValueError("Insert Edge Loop position must be finite and strictly between 0 and 1.")

    quads_by_role = _read_logical_quad_provenance(primitive)
    logical_quads = list(quads_by_role.get(resolved_role, ()))
    if not logical_quads:
        raise ValueError(
            "Insert Edge Loop requires logical Quad Draw provenance on this surface; "
            "arbitrary stock triangulation is not inferred as quads."
        )
    if len({tuple(sorted(quad)) for quad in logical_quads}) != len(logical_quads):
        raise ValueError(f"Logical quad provenance for {resolved_role!r} contains duplicate or ambiguous quads.")
    used_source_vertices = {int(vertex) for face in surface.faces for vertex in face}
    if used_source_vertices != set(range(len(surface.vertices))):
        raise ValueError(
            f"Logical quad surface {resolved_role!r} contains unused vertices; compact it before inserting a loop."
        )

    edge_to_quads: dict[tuple[int, int], list[int]] = {}
    for quad_index, quad in enumerate(logical_quads):
        if len(set(quad)) != 4 or any(vertex < 0 or vertex >= len(surface.vertices) for vertex in quad):
            raise ValueError(f"Logical quad provenance for {resolved_role!r} is malformed or out of range.")
        boundary_edges = _logical_quad_boundary_edges(quad)
        if len(set(boundary_edges)) != 4:
            raise ValueError(f"Logical quad provenance for {resolved_role!r} contains a collapsed boundary.")
        for edge in boundary_edges:
            edge_to_quads.setdefault(edge, []).append(quad_index)

    selected_key = _logical_quad_edge_key(*supplied_edge)
    start_quads = tuple(edge_to_quads.get(selected_key, ()))
    if not start_quads:
        raise ValueError(
            "The selected raw edge is not a perimeter edge in this surface's logical Quad Draw provenance."
        )
    if len(start_quads) > 2:
        raise ValueError("The selected logical edge is branched across more than two quads.")

    # quad index -> (boundary start index, current oriented edge, opposite
    # oriented edge).  The propagated orientation keeps a non-central position
    # on the same side of every quad in the strip.
    split_specs: dict[int, tuple[int, tuple[int, int], tuple[int, int]]] = {}
    traversal_order: list[int] = []
    source_faces_by_quad: dict[int, tuple[int, int]] = {}
    material_by_quad: dict[int, int] = {}

    def _traverse(start_quad: int, oriented_edge: tuple[int, int]) -> None:
        quad_index = int(start_quad)
        current_edge = tuple(int(value) for value in oriented_edge)
        while True:
            if quad_index in split_specs:
                raise ValueError("Insert Edge Loop encountered a cyclic or multiply-connected logical quad strip.")
            quad = logical_quads[quad_index]
            current_key = _logical_quad_edge_key(*current_edge)
            matches = [
                index
                for index in range(4)
                if _logical_quad_edge_key(quad[index], quad[(index + 1) % 4]) == current_key
            ]
            if len(matches) != 1:
                raise ValueError("Insert Edge Loop encountered an ambiguous edge inside logical quad provenance.")
            boundary_index = matches[0]
            boundary_pair = (quad[boundary_index], quad[(boundary_index + 1) % 4])
            if boundary_pair == current_edge:
                opposite_edge = (quad[(boundary_index + 3) % 4], quad[(boundary_index + 2) % 4])
            elif boundary_pair == (current_edge[1], current_edge[0]):
                opposite_edge = (quad[(boundary_index + 2) % 4], quad[(boundary_index + 3) % 4])
            else:
                raise ValueError("Insert Edge Loop could not preserve the selected edge orientation.")
            source_faces, material = _logical_quad_source_faces(surface, quad, mesh_role=resolved_role)
            split_specs[quad_index] = (boundary_index, current_edge, opposite_edge)
            traversal_order.append(quad_index)
            source_faces_by_quad[quad_index] = source_faces
            material_by_quad[quad_index] = material

            opposite_key = _logical_quad_edge_key(*opposite_edge)
            incident = tuple(edge_to_quads.get(opposite_key, ()))
            if len(incident) > 2:
                raise ValueError("Insert Edge Loop encountered a branched logical quad strip.")
            next_quads = tuple(index for index in incident if index != quad_index)
            if len(next_quads) > 1:
                raise ValueError("Insert Edge Loop encountered an ambiguous logical quad continuation.")
            if not next_quads:
                return
            if next_quads[0] in split_specs:
                raise ValueError("Insert Edge Loop encountered a closed or cyclic logical quad strip.")
            quad_index = next_quads[0]
            current_edge = opposite_edge

    for start_quad in start_quads:
        _traverse(start_quad, supplied_edge)

    removed_face_owners: dict[int, int] = {}
    for quad_index, source_faces in source_faces_by_quad.items():
        for face_index in source_faces:
            previous_owner = removed_face_owners.setdefault(face_index, quad_index)
            if previous_owner != quad_index:
                raise ValueError("Logical quad provenance overlaps one source triangle across multiple quads.")

    cut_orientations: dict[tuple[int, int], tuple[int, int]] = {}
    cut_points: dict[tuple[int, int], Vec3] = {}
    for quad_index in traversal_order:
        _boundary_index, current_edge, opposite_edge = split_specs[quad_index]
        for oriented_edge in (current_edge, opposite_edge):
            edge_key = _logical_quad_edge_key(*oriented_edge)
            point = _vec_lerp(
                surface.vertices[oriented_edge[0]],
                surface.vertices[oriented_edge[1]],
                amount,
            )
            if edge_key in cut_points:
                if _vec_length(_vec_sub(cut_points[edge_key], point)) > 1.0e-7:
                    raise ValueError(
                        "Insert Edge Loop found inconsistent orientation across the logical quad strip."
                    )
                continue
            cut_orientations[edge_key] = oriented_edge
            cut_points[edge_key] = point
    if len(surface.vertices) + len(cut_points) > MDL_MAX_VERTICES_PER_SURFACE:
        raise ValueError(
            f"Insert Edge Loop would exceed KOTOR's {MDL_MAX_VERTICES_PER_SURFACE}-vertex MDL surface limit."
        )

    vertices, _source_faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    cut_vertices: dict[tuple[int, int], int] = {}
    for edge_key, oriented_edge in cut_orientations.items():
        first, second = oriented_edge
        cut_vertices[edge_key] = len(vertices)
        vertices.append(cut_points[edge_key])
        uvs.append(
            (
                uvs[first][0] + ((uvs[second][0] - uvs[first][0]) * amount),
                uvs[first][1] + ((uvs[second][1] - uvs[first][1]) * amount),
            )
        )
        normals.append(_vec_slerp(normals[first], normals[second], amount))
        if uvs_lm:
            uvs_lm.append(
                (
                    uvs_lm[first][0] + ((uvs_lm[second][0] - uvs_lm[first][0]) * amount),
                    uvs_lm[first][1] + ((uvs_lm[second][1] - uvs_lm[first][1]) * amount),
                )
            )

    source_materials = _surface_face_materials(surface)
    removed_faces = set(removed_face_owners)
    faces = [face for face_index, face in enumerate(surface.faces) if face_index not in removed_faces]
    face_mats = [material for face_index, material in enumerate(source_materials) if face_index not in removed_faces]
    generated_face_start = len(faces)
    replacement_quads: dict[int, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {}
    for quad_index in sorted(split_specs):
        quad = logical_quads[quad_index]
        boundary_index, _current_edge, _opposite_edge = split_specs[quad_index]
        rotated = tuple(quad[(boundary_index + offset) % 4] for offset in range(4))
        first_cut = cut_vertices[_logical_quad_edge_key(rotated[0], rotated[1])]
        opposite_cut = cut_vertices[_logical_quad_edge_key(rotated[2], rotated[3])]
        first_quad = (rotated[0], first_cut, opposite_cut, rotated[3])
        second_quad = (first_cut, rotated[1], rotated[2], opposite_cut)
        source_normal = _newell_normal([surface.vertices[index] for index in quad])
        for new_quad in (first_quad, second_quad):
            new_normal = _newell_normal([vertices[index] for index in new_quad])
            if _vec_length(new_normal) <= 1.0e-9 or _vec_dot(new_normal, source_normal) <= 1.0e-6:
                raise ValueError("Insert Edge Loop would create a collapsed or inverted logical quad.")
            faces.extend(
                (
                    _oriented_triangle((new_quad[0], new_quad[1], new_quad[2]), vertices, source_normal),
                    _oriented_triangle((new_quad[0], new_quad[2], new_quad[3]), vertices, source_normal),
                )
            )
            face_mats.extend((material_by_quad[quad_index], material_by_quad[quad_index]))
        replacement_quads[quad_index] = (first_quad, second_quad)

    rebuilt = _compact_surface_arrays(surface, vertices, faces, uvs, normals, uvs_lm, face_mats)
    source_audit = MeshTopology.build(surface.vertices, surface.faces).validate_manifold_state()
    audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
    if (
        audit.degenerate_faces
        or len(audit.non_manifold_edges) > len(source_audit.non_manifold_edges)
        or len(audit.inconsistent_winding_edges) > len(source_audit.inconsistent_winding_edges)
        or len(audit.duplicate_faces) > len(source_audit.duplicate_faces)
    ):
        raise ValueError(
            "Insert Edge Loop would create degenerate, duplicate, non-manifold, or inconsistently wound topology."
        )

    updated_quads: list[tuple[int, int, int, int]] = []
    for quad_index, quad in enumerate(logical_quads):
        updated_quads.extend(replacement_quads.get(quad_index, (quad,)))
    for quad in updated_quads:
        _logical_quad_source_faces(rebuilt, quad, mesh_role=resolved_role)

    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    quads_by_role[resolved_role] = updated_quads
    edited = _replace_logical_quad_provenance(edited, quads_by_role)
    return _record_topology_edit(
        edited,
        "insert_edge_loop",
        mesh_role=resolved_role,
        selected_edge=[int(value) for value in supplied_edge],
        position=amount,
        affected_quad_indices=[int(value) for value in sorted(split_specs)],
        affected_quad_count=len(split_specs),
        inserted_vertex_count=len(cut_vertices),
        generated_face_start=generated_face_start,
        generated_face_count=len(split_specs) * 4,
        logical_quad_count=len(updated_quads),
        logical_quad_provenance_version=LOGICAL_QUAD_PROVENANCE_VERSION,
        uv_policy="linear_edge_interpolation",
        normal_policy="spherical_edge_interpolation",
        lightmap_uv_policy="linear_edge_interpolation" if uvs_lm else "unavailable",
    )


def fill_imported_mesh_boundary_loop(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    loop_vertex_indices: tuple[int, ...] | list[int],
    *,
    material: int | None = None,
    planarity_tolerance: float = 1.0e-4,
) -> ImportedMeshRoomPrimitive:
    """Fill one ordered, closed geometric boundary with deterministic triangles.

    The closing vertex may be omitted or repeated.  Generated cap vertices are
    channel copies of the boundary (so existing UV/lightmap data is untouched)
    with a cap normal; their duplicate positions close the geometric seam while
    retaining any deliberate UV or hard-normal split.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    loop = [int(value) for value in loop_vertex_indices]
    if len(loop) > 1 and loop[0] == loop[-1]:
        loop.pop()
    if len(loop) < 3 or len(set(loop)) != len(loop):
        raise ValueError("Boundary fill requires one simple loop of at least three unique vertices.")
    if any(index < 0 or index >= len(surface.vertices) for index in loop):
        raise ValueError(f"Boundary loop contains an out-of-range vertex for surface {mesh_role}.")

    topology = MeshTopology.build(surface.vertices, surface.faces)
    geometric_loop = [topology.raw_to_geometric_vertex[index] for index in loop]
    if len(set(geometric_loop)) != len(geometric_loop):
        raise ValueError("Boundary loop repeats a geometric vertex through seam copies.")
    loop_edges = [
        tuple(sorted((geometric_loop[index], geometric_loop[(index + 1) % len(geometric_loop)])))
        for index in range(len(geometric_loop))
    ]
    if any(edge not in topology.geometric_border_edges for edge in loop_edges):
        raise ValueError("Every supplied loop segment must be a single open boundary edge.")

    same_direction = 0
    reverse_direction = 0
    adjacent_faces: list[int] = []
    for index, edge in enumerate(loop_edges):
        half_edges = topology.geometric_edge_to_half_edges.get(edge, ())
        if len(half_edges) != 1:
            raise ValueError("Boundary fill encountered a branched or non-manifold edge.")
        half_edge = topology.half_edges[half_edges[0]]
        adjacent_faces.append(int(half_edge.face))
        first = geometric_loop[index]
        second = geometric_loop[(index + 1) % len(geometric_loop)]
        if (half_edge.geometric_origin, half_edge.geometric_destination) == (first, second):
            same_direction += 1
        elif (half_edge.geometric_origin, half_edge.geometric_destination) == (second, first):
            reverse_direction += 1
    if same_direction and reverse_direction:
        raise ValueError("Boundary loop has inconsistent source face winding; repair it before filling.")
    # The new cap must traverse every boundary edge opposite the existing face.
    if same_direction:
        loop.reverse()
        geometric_loop.reverse()
        loop_edges.reverse()
        adjacent_faces.reverse()

    points = [surface.vertices[index] for index in loop]
    cap_normal = _newell_normal(points)
    if _vec_length(cap_normal) <= 1.0e-9:
        raise ValueError("Boundary loop has no stable normal.")
    centroid = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
    tolerance = max(0.0, float(planarity_tolerance))
    maximum_offset = max(abs(_vec_dot(_vec_sub(point, centroid), cap_normal)) for point in points)
    if maximum_offset > tolerance:
        raise ValueError(
            f"Boundary loop is not planar ({maximum_offset:.6f}m deviation; {tolerance:.6f}m allowed)."
        )
    triangles = _triangulate_planar_loop(_project_planar_polygon(points, cap_normal))

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    cap_indices: list[int] = []
    for source_index in loop:
        cap_indices.append(len(vertices))
        vertices.append(vertices[source_index])
        uvs.append(uvs[source_index])
        normals.append(cap_normal)
        if uvs_lm:
            uvs_lm.append(uvs_lm[source_index])
    generated_faces = [tuple(cap_indices[index] for index in triangle) for triangle in triangles]
    faces.extend(generated_faces)

    source_materials = _surface_face_materials(surface)
    if material is None:
        counts: dict[int, int] = {}
        for face_index in adjacent_faces:
            value = source_materials[face_index]
            counts[value] = counts.get(value, 0) + 1
        cap_material = min(counts, key=lambda value: (-counts[value], value)) if counts else 0
    else:
        cap_material = int(material)
    face_mats = list(source_materials) + [cap_material] * len(generated_faces)
    rebuilt = _compact_surface_arrays(surface, vertices, faces, uvs, normals, uvs_lm, face_mats)
    audit = MeshTopology.build(rebuilt.vertices, rebuilt.faces).validate_manifold_state()
    if audit.degenerate_faces or audit.non_manifold_edges or audit.inconsistent_winding_edges:
        raise ValueError("Boundary fill would create invalid or inconsistently wound topology.")

    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "fill_boundary_loop",
        mesh_role=mesh_role,
        source_loop_vertices=loop,
        generated_face_start=len(surface.faces),
        generated_face_count=len(generated_faces),
        material=cap_material,
        maximum_planarity_deviation=maximum_offset,
    )


def _set_imported_mesh_corner_normals(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_geometric_vertices: dict[int, set[int]],
    *,
    soften: bool,
    operation: str,
    selection_details: dict[str, Any],
) -> ImportedMeshRoomPrimitive:
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    topology = MeshTopology.build(surface.vertices, surface.faces)
    if not face_geometric_vertices:
        raise ValueError("A normal edit requires at least one selected edge or face.")

    averaged: dict[int, Vec3] = {}
    if soften:
        accumulators: dict[int, Vec3] = {}
        for face_index, geometric_vertices in face_geometric_vertices.items():
            normal = topology.face_normals[face_index]
            weighted = _vec_scale(normal, max(topology.face_areas[face_index], 1.0e-12))
            for geometric in geometric_vertices:
                accumulators[geometric] = _vec_add(accumulators.get(geometric, (0.0, 0.0, 0.0)), weighted)
        averaged = {
            geometric: _vec_normalized(value, fallback=(0.0, 0.0, 1.0))
            for geometric, value in accumulators.items()
        }

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    rewritten_faces = list(faces)
    for face_index in sorted(face_geometric_vertices):
        target_geometric = face_geometric_vertices[face_index]
        row = list(rewritten_faces[face_index])
        for corner, raw_index in enumerate(row):
            geometric = topology.raw_to_geometric_vertex[raw_index]
            if geometric not in target_geometric:
                continue
            duplicate = len(vertices)
            vertices.append(vertices[raw_index])
            uvs.append(uvs[raw_index])
            normals.append(averaged[geometric] if soften else topology.face_normals[face_index])
            if uvs_lm:
                uvs_lm.append(uvs_lm[raw_index])
            row[corner] = duplicate
        rewritten_faces[face_index] = tuple(row)

    rebuilt = _compact_surface_arrays(
        surface,
        vertices,
        rewritten_faces,
        uvs,
        normals,
        uvs_lm,
        list(_surface_face_materials(surface)),
    )
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        operation,
        mesh_role=mesh_role,
        softened=bool(soften),
        affected_face_count=len(face_geometric_vertices),
        **selection_details,
    )


def set_imported_mesh_edge_smoothing(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | tuple[int, int],
    *,
    soften: bool,
) -> ImportedMeshRoomPrimitive:
    """Soften or harden selected geometric edges by editing real normals.

    Corners touched by the edge are duplicated so an unrelated face sharing a
    raw vertex is not modified.  Softening writes the same area-weighted normal
    to both sides; hardening writes each incident face normal.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    topology = MeshTopology.build(surface.vertices, surface.faces)
    raw_rows = tuple(edge_vertex_indices)
    if len(raw_rows) == 2 and all(isinstance(value, int) for value in raw_rows):
        edges = ((int(raw_rows[0]), int(raw_rows[1])),)
    else:
        edges = tuple((int(row[0]), int(row[1])) for row in raw_rows)  # type: ignore[index]
    if not edges:
        raise ValueError("Edge smoothing requires at least one selected edge.")

    face_targets: dict[int, set[int]] = {}
    geometric_edges: list[tuple[int, int]] = []
    for first_raw, second_raw in edges:
        if not (0 <= first_raw < len(surface.vertices) and 0 <= second_raw < len(surface.vertices)):
            raise ValueError(f"Smoothing edge ({first_raw}, {second_raw}) has an out-of-range vertex.")
        first_geometric = topology.raw_to_geometric_vertex[first_raw]
        second_geometric = topology.raw_to_geometric_vertex[second_raw]
        edge = tuple(sorted((first_geometric, second_geometric)))
        incident = topology.geometric_edge_to_faces.get(edge, ())
        if not incident:
            raise ValueError(f"Vertices {first_raw} and {second_raw} do not form a mesh edge.")
        geometric_edges.append(edge)
        for face_index in incident:
            face_targets.setdefault(int(face_index), set()).update(edge)
    return _set_imported_mesh_corner_normals(
        primitive,
        mesh_role,
        face_targets,
        soften=soften,
        operation="soften_edges" if soften else "harden_edges",
        selection_details={"geometric_edges": [list(edge) for edge in geometric_edges]},
    )


def soften_imported_mesh_edges(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | tuple[int, int],
) -> ImportedMeshRoomPrimitive:
    """Convenience wrapper for Maya-style Soften Edge."""

    return set_imported_mesh_edge_smoothing(primitive, mesh_role, edge_vertex_indices, soften=True)


def harden_imported_mesh_edges(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    edge_vertex_indices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | tuple[int, int],
) -> ImportedMeshRoomPrimitive:
    """Convenience wrapper for Maya-style Harden Edge."""

    return set_imported_mesh_edge_smoothing(primitive, mesh_role, edge_vertex_indices, soften=False)


def set_imported_mesh_face_smoothing(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
    *,
    soften: bool,
) -> ImportedMeshRoomPrimitive:
    """Soften a face region together or harden every selected face corner."""

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    selected = tuple(sorted({int(value) for value in face_indices}))
    _validate_face_indices(surface, selected, role=mesh_role)
    topology = MeshTopology.build(surface.vertices, surface.faces)
    targets = {
        face_index: {topology.raw_to_geometric_vertex[raw] for raw in surface.faces[face_index]}
        for face_index in selected
    }
    return _set_imported_mesh_corner_normals(
        primitive,
        mesh_role,
        targets,
        soften=soften,
        operation="soften_face_region" if soften else "harden_face_region",
        selection_details={"face_indices": list(selected)},
    )


def delete_imported_mesh_vertex_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    vertex_corner: int,
) -> ImportedMeshRoomPrimitive:
    """Delete the vertex's face fan (every face touching the position, all surfaces)."""

    _surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, vertex_corner=vertex_corner)
    wanted = set(positions)
    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        kept = [
            face
            for face in surface.faces
            if not any(_position_key(surface.vertices[index]) in wanted for index in face)
        ]
        if not kept:
            continue
        surfaces.append(surface if len(kept) == len(surface.faces) else _compact_surface(surface, kept))
    if not surfaces:
        raise ValueError(f"Deleting this vertex fan would leave imported room {primitive.room_resref} empty.")
    return replace(primitive, surfaces=tuple(surfaces))


def delete_imported_mesh_edge_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
) -> ImportedMeshRoomPrimitive:
    """Delete every face (all surfaces) that uses this edge's position pair."""

    _surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, edge_corners=edge_corners)
    pair = set(positions)
    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        kept = []
        for face in surface.faces:
            keys = {
                _position_key(surface.vertices[face[0]]),
                _position_key(surface.vertices[face[1]]),
                _position_key(surface.vertices[face[2]]),
            }
            if pair <= keys:
                continue
            kept.append(face)
        if not kept:
            continue
        surfaces.append(surface if len(kept) == len(surface.faces) else _compact_surface(surface, kept))
    if not surfaces:
        raise ValueError(f"Deleting this edge would leave imported room {primitive.room_resref} empty.")
    return replace(primitive, surfaces=tuple(surfaces))


def split_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
) -> ImportedMeshRoomPrimitive:
    """Insert a midpoint vertex into every face sharing the edge (seam-aware)."""

    _surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, edge_corners=edge_corners)
    key_a, key_b = positions[0], positions[1]
    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        vertices, _faces, uvs, normals, uvs_lm = _surface_arrays(surface)
        new_faces: list[Face] = []
        changed = False
        for face in surface.faces:
            corner_keys = [_position_key(surface.vertices[index]) for index in face]
            split_pair = None
            for first in range(3):
                second = (first + 1) % 3
                if {corner_keys[first], corner_keys[second]} == {key_a, key_b}:
                    split_pair = (first, second)
                    break
            if split_pair is None:
                new_faces.append(face)
                continue
            changed = True
            first, second = split_pair
            third = 3 - first - second
            ia, ib, ic = face[first], face[second], face[third]
            mid_index = len(vertices)
            vertices.append(_vec_scale(_vec_add(vertices[ia], vertices[ib]), 0.5))
            uvs.append(((uvs[ia][0] + uvs[ib][0]) * 0.5, (uvs[ia][1] + uvs[ib][1]) * 0.5))
            normals.append(_vec_normalized(_vec_add(normals[ia], normals[ib])))
            if uvs_lm:
                uvs_lm.append(
                    (
                        (uvs_lm[ia][0] + uvs_lm[ib][0]) * 0.5,
                        (uvs_lm[ia][1] + uvs_lm[ib][1]) * 0.5,
                    )
                )
            # Preserve winding: (ia, ib, ic) -> (ia, mid, ic) + (mid, ib, ic).
            new_faces.extend(((ia, mid_index, ic), (mid_index, ib, ic)))
        if changed:
            surfaces.append(
                replace(
                    surface,
                    vertices=tuple(vertices),
                    faces=tuple(new_faces),
                    uvs=tuple(uvs),
                    normals=tuple(normals),
                    uvs_lm=tuple(uvs_lm),
                )
            )
        else:
            surfaces.append(surface)
    return replace(primitive, surfaces=tuple(surfaces))


def collapse_imported_mesh_edge(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    edge_corners: tuple[int, int],
) -> ImportedMeshRoomPrimitive:
    """Merge the edge's endpoints at their midpoint; degenerate faces drop."""

    surface, positions = _resolve_component_positions(primitive, mesh_role, face_index, edge_corners=edge_corners)
    face = surface.faces[int(face_index)]
    a_corner, b_corner = (int(v) % 3 for v in edge_corners)
    midpoint = _vec_scale(
        _vec_add(surface.vertices[face[a_corner]], surface.vertices[face[b_corner]]),
        0.5,
    )
    return _snap_positions(primitive, positions, midpoint)


def split_imported_mesh_face_at_point(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
    point: Vec3,
    *,
    project_to_plane: bool = True,
    plane_tolerance: float = 1.0e-4,
    boundary_tolerance: float = 1.0e-6,
) -> ImportedMeshRoomPrimitive:
    """Insert an arbitrary room-local point into a triangle.

    The point is projected to the triangle plane by default and every vertex
    channel is barycentrically interpolated.  Boundary hits are rejected: a
    Multi-Cut edge hit must split both adjacent faces rather than creating a
    T-junction in only this triangle.
    """

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    face = int(face_index)
    _validate_face_indices(surface, (face,), role=mesh_role)
    vertices, _faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    a, b, c = surface.faces[face]
    values = tuple(float(value) for value in tuple(point)[:3])
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Face split point must contain three finite coordinates.")
    pa, pb, pc = vertices[a], vertices[b], vertices[c]
    edge_ab = _vec_sub(pb, pa)
    edge_ac = _vec_sub(pc, pa)
    raw_normal = _vec_cross(edge_ab, edge_ac)
    normal_length = _vec_length(raw_normal)
    if normal_length <= 1.0e-12:
        raise ValueError("Cannot split a degenerate triangle.")
    plane_normal = _vec_scale(raw_normal, 1.0 / normal_length)
    plane_distance = _vec_dot(_vec_sub(values, pa), plane_normal)
    tolerance = max(0.0, float(plane_tolerance))
    if not project_to_plane and abs(plane_distance) > tolerance:
        raise ValueError(
            f"Face split point is {abs(plane_distance):.6f}m off the triangle plane "
            f"({tolerance:.6f}m allowed)."
        )
    projected = _vec_sub(values, _vec_scale(plane_normal, plane_distance))

    point_offset = _vec_sub(projected, pa)
    dot00 = _vec_dot(edge_ab, edge_ab)
    dot01 = _vec_dot(edge_ab, edge_ac)
    dot11 = _vec_dot(edge_ac, edge_ac)
    dot20 = _vec_dot(point_offset, edge_ab)
    dot21 = _vec_dot(point_offset, edge_ac)
    denominator = (dot00 * dot11) - (dot01 * dot01)
    if abs(denominator) <= 1.0e-18:
        raise ValueError("Cannot derive stable barycentric coordinates for this triangle.")
    weight_b = ((dot11 * dot20) - (dot01 * dot21)) / denominator
    weight_c = ((dot00 * dot21) - (dot01 * dot20)) / denominator
    weight_a = 1.0 - weight_b - weight_c
    weights = (weight_a, weight_b, weight_c)
    boundary = max(0.0, float(boundary_tolerance))
    if min(weights) <= boundary or max(weights) >= 1.0 - boundary:
        if all(-boundary <= weight <= 1.0 + boundary for weight in weights):
            raise ValueError("Face split point lies on an edge or vertex; use an edge split to avoid a T-junction.")
        raise ValueError("Face split point lies outside the selected triangle.")

    inserted_index = len(vertices)
    vertices.append(projected)

    def _weighted_channel(channel: list[tuple[float, ...]], dimensions: int) -> tuple[float, ...]:
        source = (channel[a], channel[b], channel[c])
        return tuple(
            sum(float(source[corner][axis]) * weights[corner] for corner in range(3))
            for axis in range(dimensions)
        )

    uvs.append(_weighted_channel(uvs, 2))
    normals.append(_vec_normalized(_weighted_channel(normals, 3), fallback=plane_normal))
    if uvs_lm:
        uvs_lm.append(_weighted_channel(uvs_lm, 2))
    new_faces = list(surface.faces)
    replacements = ((a, b, inserted_index), (b, c, inserted_index), (c, a, inserted_index))
    new_faces[face : face + 1] = replacements
    source_materials = _surface_face_materials(surface)
    face_mats = list(source_materials)
    face_mats[face : face + 1] = [source_materials[face]] * 3
    rebuilt = _compact_surface_arrays(surface, vertices, new_faces, uvs, normals, uvs_lm, face_mats)
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "split_face_at_point",
        mesh_role=mesh_role,
        source_face=face,
        point=[float(value) for value in projected],
        barycentric=[float(value) for value in weights],
        generated_face_start=face,
        generated_face_count=3,
    )


def split_imported_mesh_face(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_index: int,
) -> ImportedMeshRoomPrimitive:
    """Insert a centroid vertex: one triangle becomes three (Insert Point)."""

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    face = int(face_index)
    _validate_face_indices(surface, (face,), role=mesh_role)
    a, b, c = surface.faces[face]
    centroid = tuple(
        (surface.vertices[a][axis] + surface.vertices[b][axis] + surface.vertices[c][axis]) / 3.0
        for axis in range(3)
    )
    return split_imported_mesh_face_at_point(primitive, mesh_role, face, centroid)


def flatten_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
) -> ImportedMeshRoomPrimitive:
    """Project the region's vertices onto its average plane (seam copies too)."""

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = tuple(sorted({int(index) for index in face_indices}))
    _validate_face_indices(surface, wanted, role=mesh_role)
    region_vertices: set[int] = set()
    normal_sum = (0.0, 0.0, 0.0)
    for index in wanted:
        region_vertices.update(surface.faces[index])
        normal_sum = _vec_add(normal_sum, _face_normal(surface, surface.faces[index]))
    plane_normal = _vec_normalized(normal_sum)
    points = [surface.vertices[index] for index in region_vertices]
    centroid = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
        sum(p[2] for p in points) / len(points),
    )
    moves: dict[tuple[float, float, float], Vec3] = {}
    for index in region_vertices:
        point = surface.vertices[index]
        offset = _vec_sub(point, centroid)
        height = (offset[0] * plane_normal[0]) + (offset[1] * plane_normal[1]) + (offset[2] * plane_normal[2])
        moves[_position_key(point)] = _vec_sub(point, _vec_scale(plane_normal, height))
    surfaces = []
    for candidate in primitive.surfaces:
        vertices = tuple(moves.get(_position_key(vertex), vertex) for vertex in candidate.vertices)
        surfaces.append(replace(candidate, vertices=vertices) if vertices != candidate.vertices else candidate)
    return _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))


def flip_imported_mesh_faces(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    face_indices: tuple[int, ...] | list[int],
) -> ImportedMeshRoomPrimitive:
    """Reverse the winding (and exclusive vertex normals) of the target faces."""

    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    wanted = set(int(index) for index in face_indices)
    _validate_face_indices(surface, tuple(sorted(wanted)), role=mesh_role)
    flipped_faces = tuple(
        (face[0], face[2], face[1]) if index in wanted else face for index, face in enumerate(surface.faces)
    )
    flipped_vertices: set[int] = set()
    kept_vertices: set[int] = set()
    for index, face in enumerate(surface.faces):
        (flipped_vertices if index in wanted else kept_vertices).update(face)
    exclusive = flipped_vertices - kept_vertices
    vertex_count = len(surface.vertices)
    normals = surface.normals if len(surface.normals) == vertex_count else tuple((0.0, 0.0, 1.0) for _ in surface.vertices)
    normals = tuple(
        _vec_scale(normal, -1.0) if index in exclusive else normal for index, normal in enumerate(normals)
    )
    surfaces = list(primitive.surfaces)
    surfaces[surface_index] = replace(surface, faces=flipped_faces, normals=normals)
    return replace(primitive, surfaces=tuple(surfaces))


# ---------------------------------------------------------------------------
# Maya-style baked mirror / bridge / deformer operations

_AXIS_INDICES = {"x": 0, "y": 1, "z": 2}


def _deformer_axis_index(axis: str) -> int:
    clean = str(axis or "").strip().lower()
    if clean not in _AXIS_INDICES:
        raise ValueError(f"Axis must be x, y, or z; received {axis!r}.")
    return _AXIS_INDICES[clean]


def _finite_vec3(value: Vec3, *, label: str) -> Vec3:
    values = tuple(float(component) for component in tuple(value)[:3])
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise ValueError(f"{label} must contain three finite coordinates.")
    return values


def _surface_vertex_selection(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str | None,
    vertex_indices: tuple[int, ...] | list[int] | None,
) -> tuple[set[tuple[float, float, float]], tuple[int, ...], str]:
    """Resolve raw vertex indices to seam-aware geometric positions.

    ``None`` for ``mesh_role`` means every vertex in every imported surface.
    Otherwise indices are raw surface-global vertex indices, matching the
    component records emitted by the viewport.  Co-located UV/material seam
    copies move together so a baked deformer cannot tear a stock room open.
    """

    if mesh_role is None:
        if vertex_indices is not None:
            raise ValueError("Raw vertex indices require a specific mesh role.")
        return (
            {
                _position_key(vertex)
                for surface in primitive.surfaces
                for vertex in surface.vertices
            },
            (),
            "all_surfaces",
        )
    surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    surface = primitive.surfaces[surface_index]
    selected = (
        tuple(range(len(surface.vertices)))
        if vertex_indices is None
        else tuple(sorted({int(index) for index in vertex_indices}))
    )
    if not selected:
        raise ValueError("At least one imported-mesh vertex must be selected.")
    for index in selected:
        if not 0 <= index < len(surface.vertices):
            raise ValueError(
                f"Vertex index {index} out of range for surface {mesh_role} "
                f"({len(surface.vertices)} vertices)."
            )
    return ({_position_key(surface.vertices[index]) for index in selected}, selected, str(mesh_role))


def _matrix_inverse_transpose_normal(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    normal: Vec3,
) -> Vec3:
    """Transform a normal by a deformation Jacobian's inverse transpose."""

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = (a * ((e * i) - (f * h))) - (b * ((d * i) - (f * g))) + (c * ((d * h) - (e * g)))
    if abs(determinant) <= 1.0e-12:
        return _vec_normalized(normal)
    inverse = (
        (((e * i) - (f * h)) / determinant, ((c * h) - (b * i)) / determinant, ((b * f) - (c * e)) / determinant),
        (((f * g) - (d * i)) / determinant, ((a * i) - (c * g)) / determinant, ((c * d) - (a * f)) / determinant),
        (((d * h) - (e * g)) / determinant, ((b * g) - (a * h)) / determinant, ((a * e) - (b * d)) / determinant),
    )
    # inverse-transpose * column normal
    return _vec_normalized(
        (
            (inverse[0][0] * normal[0]) + (inverse[1][0] * normal[1]) + (inverse[2][0] * normal[2]),
            (inverse[0][1] * normal[0]) + (inverse[1][1] * normal[1]) + (inverse[2][1] * normal[2]),
            (inverse[0][2] * normal[0]) + (inverse[1][2] * normal[1]) + (inverse[2][2] * normal[2]),
        ),
        fallback=_vec_normalized(normal),
    )


def _nearest_point_on_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> tuple[Vec3, tuple[float, float, float]]:
    """Return the deterministic closest point and barycentric weights.

    This is the region-test form from *Real-Time Collision Detection*.  It is
    branch-only and allocation-light enough for the small, explicit Map Studio
    bake; interactive preview can later replace the exhaustive face scan with
    the renderer BVH without changing this operator contract.
    """

    ab = _vec_sub(b, a)
    ac = _vec_sub(c, a)
    ap = _vec_sub(point, a)
    d1 = _vec_dot(ab, ap)
    d2 = _vec_dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (1.0, 0.0, 0.0)

    bp = _vec_sub(point, b)
    d3 = _vec_dot(ab, bp)
    d4 = _vec_dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)

    vc = (d1 * d4) - (d3 * d2)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        amount = d1 / max(1.0e-18, d1 - d3)
        return _vec_lerp(a, b, amount), (1.0 - amount, amount, 0.0)

    cp = _vec_sub(point, c)
    d5 = _vec_dot(ab, cp)
    d6 = _vec_dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)

    vb = (d5 * d2) - (d1 * d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        amount = d2 / max(1.0e-18, d2 - d6)
        return _vec_lerp(a, c, amount), (1.0 - amount, 0.0, amount)

    va = (d3 * d6) - (d5 * d4)
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        amount = (d4 - d3) / max(1.0e-18, (d4 - d3) + (d5 - d6))
        return _vec_lerp(b, c, amount), (0.0, 1.0 - amount, amount)

    denominator = max(1.0e-18, va + vb + vc)
    weight_b = vb / denominator
    weight_c = vc / denominator
    weight_a = 1.0 - weight_b - weight_c
    return (
        (
            (a[0] * weight_a) + (b[0] * weight_b) + (c[0] * weight_c),
            (a[1] * weight_a) + (b[1] * weight_b) + (c[1] * weight_c),
            (a[2] * weight_a) + (b[2] * weight_b) + (c[2] * weight_c),
        ),
        (weight_a, weight_b, weight_c),
    )


def mirror_imported_mesh_geometry(
    primitive: ImportedMeshRoomPrimitive,
    *,
    axis: str = "x",
    center: float | Vec3 = 0.0,
    duplicate: bool = True,
    merge_seam_tolerance: float = 1.0e-5,
    mesh_roles: tuple[str, ...] | list[str] | None = None,
) -> ImportedMeshRoomPrimitive:
    """Bake a polygon mirror across one room-local axis plane.

    Available vertex channels and material rows are copied exactly. Reflected
    faces reverse winding and reflected normals reverse only their mirrored
    component. In duplicate mode vertices on the mirror plane reuse the source
    index when ``merge_seam_tolerance`` is positive, matching Maya's Merge
    Border option without welding unrelated nearby points.
    """

    axis_index = _deformer_axis_index(axis)
    if isinstance(center, (tuple, list)):
        center_point = _finite_vec3(center, label="Mirror center")
        center_coordinate = center_point[axis_index]
    else:
        center_coordinate = float(center)
        if not math.isfinite(center_coordinate):
            raise ValueError("Mirror center must be finite.")
    tolerance = max(0.0, float(merge_seam_tolerance))
    requested_roles = None if mesh_roles is None else {str(role) for role in mesh_roles}
    if requested_roles is not None:
        known_roles = {imported_mesh_surface_role(index) for index in range(len(primitive.surfaces))}
        unknown = sorted(requested_roles - known_roles)
        if unknown:
            raise ValueError(f"Unknown imported mesh surface role(s): {', '.join(unknown)}")
        if not requested_roles:
            raise ValueError("Mirror requires at least one mesh role.")

    rebuilt_surfaces: list[ImportedMeshSurface] = []
    mirrored_face_count = 0
    seam_channel_splits = 0
    for surface_index, surface in enumerate(primitive.surfaces):
        role = imported_mesh_surface_role(surface_index)
        if requested_roles is not None and role not in requested_roles:
            rebuilt_surfaces.append(surface)
            continue
        vertex_count = len(surface.vertices)
        has_uvs = len(surface.uvs) == vertex_count
        has_normals = len(surface.normals) == vertex_count
        has_lightmap_uvs = len(surface.uvs_lm) == vertex_count
        reflected_vertices: list[Vec3] = []
        reflected_normals: list[Vec3] = []
        for raw_index, point in enumerate(surface.vertices):
            mirrored = list(point)
            mirrored[axis_index] = (2.0 * center_coordinate) - mirrored[axis_index]
            reflected_vertices.append(tuple(mirrored))
            if has_normals:
                normal = list(surface.normals[raw_index])
                normal[axis_index] = -normal[axis_index]
                reflected_normals.append(_vec_normalized(tuple(normal)))

        source_materials = _surface_face_materials(surface)
        if not duplicate:
            rebuilt = replace(
                surface,
                vertices=tuple(reflected_vertices),
                faces=tuple((face[0], face[2], face[1]) for face in surface.faces),
                face_mats=source_materials if surface.face_mats else (),
                uvs=tuple(surface.uvs) if has_uvs else (),
                normals=tuple(reflected_normals) if has_normals else (),
                uvs_lm=tuple(surface.uvs_lm) if has_lightmap_uvs else (),
            )
            rebuilt_surfaces.append(rebuilt)
            mirrored_face_count += len(surface.faces)
            continue

        vertices = list(surface.vertices)
        faces = list(surface.faces)
        uvs = list(surface.uvs) if has_uvs else []
        normals = list(surface.normals) if has_normals else []
        uvs_lm = list(surface.uvs_lm) if has_lightmap_uvs else []
        face_mats = list(source_materials) if surface.face_mats else []
        mirror_indices: list[int] = []
        for raw_index, point in enumerate(surface.vertices):
            on_merge_plane = tolerance > 0.0 and abs(float(point[axis_index]) - center_coordinate) <= tolerance
            normal_can_share = (
                not has_normals
                or _vec_dot(surface.normals[raw_index], reflected_normals[raw_index]) >= 1.0 - 1.0e-6
            )
            if on_merge_plane and normal_can_share:
                mirror_indices.append(raw_index)
                continue
            if on_merge_plane and not normal_can_share:
                # A single Odyssey vertex cannot carry both hard seam normals.
                # Keep co-located channel copies; MeshTopology still resolves
                # them as one geometric border without corrupting shading.
                seam_channel_splits += 1
            mirror_indices.append(len(vertices))
            vertices.append(reflected_vertices[raw_index])
            if has_uvs:
                uvs.append(surface.uvs[raw_index])
            if has_normals:
                normals.append(reflected_normals[raw_index])
            if has_lightmap_uvs:
                uvs_lm.append(surface.uvs_lm[raw_index])
        for face_index, face in enumerate(surface.faces):
            mirrored_face = (
                mirror_indices[face[0]],
                mirror_indices[face[2]],
                mirror_indices[face[1]],
            )
            # A polygon lying wholly in the mirror plane is already the seam;
            # adding its reverse would create coincident z-fighting geometry.
            if set(mirrored_face) == set(face):
                continue
            points = tuple(vertices[index] for index in mirrored_face)
            if len({_position_key(point) for point in points}) != 3:
                continue
            if _vec_length(_vec_cross(_vec_sub(points[1], points[0]), _vec_sub(points[2], points[0]))) <= 1.0e-10:
                continue
            faces.append(mirrored_face)
            if surface.face_mats:
                face_mats.append(source_materials[face_index])
            mirrored_face_count += 1
        rebuilt_surfaces.append(
            replace(
                surface,
                vertices=tuple(vertices),
                faces=tuple(faces),
                face_mats=tuple(face_mats),
                uvs=tuple(uvs),
                normals=tuple(normals),
                uvs_lm=tuple(uvs_lm),
            )
        )

    edited = _drop_degenerate_faces(replace(primitive, surfaces=tuple(rebuilt_surfaces)))
    return _record_topology_edit(
        edited,
        "mirror_geometry",
        axis=str(axis).lower(),
        center=float(center_coordinate),
        duplicate=bool(duplicate),
        merge_seam_tolerance=tolerance,
        mesh_roles=sorted(requested_roles) if requested_roles is not None else ["all"],
        mirrored_face_count=mirrored_face_count,
        seam_channel_splits=seam_channel_splits,
        winding_policy="reversed_after_reflection",
    )


def _oriented_border_edge(
    surface: ImportedMeshSurface,
    edge: tuple[int, int],
    *,
    role: str,
) -> tuple[int, int, int]:
    """Resolve a raw vertex pair to the adjacent face's directed border edge."""

    first, second = (int(value) for value in edge)
    if first == second:
        raise ValueError("Bridge border edge endpoints must be different vertices.")
    for index in (first, second):
        if not 0 <= index < len(surface.vertices):
            raise ValueError(f"Bridge vertex index {index} out of range for surface {role}.")
    topology = MeshTopology.build(surface.vertices, surface.faces)
    geometric_edge = tuple(
        sorted((topology.raw_to_geometric_vertex[first], topology.raw_to_geometric_vertex[second]))
    )
    adjacent = tuple(topology.geometric_edge_to_faces.get(geometric_edge, ()))
    if len(adjacent) != 1:
        state = "not an edge" if not adjacent else f"shared by {len(adjacent)} faces"
        raise ValueError(f"Bridge requires a border edge; {role} edge {edge!r} is {state}.")
    face_index = int(adjacent[0])
    face = surface.faces[face_index]
    for corner in range(3):
        raw_a = face[corner]
        raw_b = face[(corner + 1) % 3]
        candidate = tuple(
            sorted((topology.raw_to_geometric_vertex[raw_a], topology.raw_to_geometric_vertex[raw_b]))
        )
        if candidate == geometric_edge:
            return raw_a, raw_b, face_index
    raise ValueError(f"Could not resolve directed border edge {edge!r} on surface {role}.")


def bridge_imported_mesh_border_edges(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str,
    first_edge: tuple[int, int],
    second_edge: tuple[int, int],
    *,
    target_mesh_role: str | None = None,
    divisions: int = 0,
    taper: float = 0.0,
    twist_degrees: float = 0.0,
    smooth: bool = True,
) -> ImportedMeshRoomPrimitive:
    """Bridge two border edges with a deterministic, optionally divided strip.

    Vertex pairs are raw surface-global indices. The bridge is emitted into
    ``mesh_role`` and inherits the first edge's material. When the target edge
    belongs to another texture surface its position/UV/normal/lightmap corners
    are copied into the bridge surface; the original surfaces remain intact.

    ``divisions`` inserts intermediate two-vertex rows. ``taper`` and ``twist``
    deform only those intermediate rows using a sine envelope, so the generated
    strip always lands exactly on both selected KOTOR border edges. This is a
    baked polygon operation: no Maya dependency-graph node is serialized.
    """

    division_count = int(divisions)
    if division_count < 0 or division_count > 1024:
        raise ValueError("Bridge divisions must be between 0 and 1024.")
    taper_amount = float(taper)
    twist_amount = float(twist_degrees)
    if not math.isfinite(taper_amount) or taper_amount <= -1.0:
        raise ValueError("Bridge taper must be finite and greater than -1 so the strip cannot collapse.")
    if not math.isfinite(twist_amount):
        raise ValueError("Bridge twist must be finite.")

    source_surface_index = imported_mesh_surface_index_for_role(primitive, mesh_role)
    if source_surface_index < 0:
        raise ValueError(f"Unknown imported mesh surface role: {mesh_role!r}")
    target_role = str(target_mesh_role or mesh_role)
    target_surface_index = imported_mesh_surface_index_for_role(primitive, target_role)
    if target_surface_index < 0:
        raise ValueError(f"Unknown imported mesh target surface role: {target_role!r}")
    source = primitive.surfaces[source_surface_index]
    target = primitive.surfaces[target_surface_index]
    source_a, source_b, source_face_index = _oriented_border_edge(source, first_edge, role=mesh_role)
    target_a, target_b, target_face_index = _oriented_border_edge(target, second_edge, role=target_role)
    if source_surface_index == target_surface_index:
        source_geometric = {_position_key(source.vertices[index]) for index in (source_a, source_b)}
        target_geometric = {_position_key(target.vertices[index]) for index in (target_a, target_b)}
        if source_geometric & target_geometric:
            raise ValueError("Bridge edges must not share a geometric endpoint.")

    # A new face must traverse each shared border opposite to the adjacent
    # source face. Row zero is source_b->source_a and the last row is
    # target_a->target_b, yielding the quad perimeter q0,q1,q2,q3.
    source_row = (source.vertices[source_b], source.vertices[source_a])
    target_row = (target.vertices[target_a], target.vertices[target_b])
    source_mid = _vec_scale(_vec_add(source_row[0], source_row[1]), 0.5)
    target_mid = _vec_scale(_vec_add(target_row[0], target_row[1]), 0.5)
    bridge_axis = _vec_normalized(_vec_sub(target_mid, source_mid), fallback=(0.0, 0.0, 1.0))

    def _rotate_about_axis(vector: Vec3, axis: Vec3, angle: float) -> Vec3:
        sine = math.sin(angle)
        cosine = math.cos(angle)
        return _vec_add(
            _vec_add(_vec_scale(vector, cosine), _vec_scale(_vec_cross(axis, vector), sine)),
            _vec_scale(axis, _vec_dot(axis, vector) * (1.0 - cosine)),
        )

    row_count = division_count + 2
    strip_rows: list[tuple[Vec3, Vec3]] = []
    for row_index in range(row_count):
        fraction = row_index / float(row_count - 1)
        left = _vec_lerp(source_row[0], target_row[0], fraction)
        right = _vec_lerp(source_row[1], target_row[1], fraction)
        if 0 < row_index < row_count - 1 and (taper_amount or twist_amount):
            center = _vec_scale(_vec_add(left, right), 0.5)
            half = _vec_scale(_vec_sub(right, left), 0.5)
            envelope = math.sin(math.pi * fraction)
            half = _vec_scale(half, 1.0 + (taper_amount * envelope))
            half = _rotate_about_axis(half, bridge_axis, math.radians(twist_amount) * envelope)
            left = _vec_sub(center, half)
            right = _vec_add(center, half)
        strip_rows.append((left, right))

    local_triangles: list[tuple[int, int, int]] = []
    cell_diagonals: list[str] = []
    triangle_normals: list[Vec3] = []
    for cell_index in range(row_count - 1):
        q0, q1 = strip_rows[cell_index]
        q3, q2 = strip_rows[cell_index + 1]
        local_points = (q0, q1, q2, q3)
        diagonal_02 = _vec_dot(_vec_sub(q0, q2), _vec_sub(q0, q2))
        diagonal_13 = _vec_dot(_vec_sub(q1, q3), _vec_sub(q1, q3))
        if diagonal_02 <= diagonal_13:
            cell_faces = ((0, 1, 2), (0, 2, 3))
            cell_diagonals.append("q0_q2")
        else:
            cell_faces = ((0, 1, 3), (1, 2, 3))
            cell_diagonals.append("q1_q3")
        cell_normals: list[Vec3] = []
        row_base = cell_index * 2
        remap = (row_base, row_base + 1, row_base + 3, row_base + 2)
        for local_face in cell_faces:
            a, b, c = (local_points[index] for index in local_face)
            normal = _vec_cross(_vec_sub(b, a), _vec_sub(c, a))
            if _vec_length(normal) <= 1.0e-10:
                raise ValueError("Bridge divisions, taper, or twist would create a degenerate triangle.")
            cell_normals.append(_vec_normalized(normal))
            local_triangles.append(tuple(remap[index] for index in local_face))
        if _vec_dot(cell_normals[0], cell_normals[1]) <= 0.0:
            raise ValueError("Bridge edges and options form a twisted or self-intersecting strip.")
        if triangle_normals and _vec_dot(triangle_normals[-1], cell_normals[0]) <= -0.25:
            raise ValueError("Bridge twist reverses the strip between divisions.")
        triangle_normals.extend(cell_normals)

    generated_vertex_count = row_count * 2
    if len(source.vertices) + generated_vertex_count > MDL_MAX_VERTICES_PER_SURFACE:
        raise ValueError(
            f"Bridge would exceed KOTOR's {MDL_MAX_VERTICES_PER_SURFACE}-vertex MDL surface limit."
        )

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(source)
    _target_vertices, _target_faces, target_uvs, target_normals, target_uvs_lm = _surface_arrays(target)
    source_channel_indices = (source_b, source_a)
    target_channel_indices = (target_a, target_b)
    base = len(vertices)
    accumulated_normals: list[Vec3] = [(0.0, 0.0, 0.0) for _ in range(generated_vertex_count)]
    for face, normal in zip(local_triangles, triangle_normals):
        for vertex_index in face:
            accumulated_normals[vertex_index] = _vec_add(accumulated_normals[vertex_index], normal)
    for row_index, row in enumerate(strip_rows):
        fraction = row_index / float(row_count - 1)
        for column_index, point_value in enumerate(row):
            vertices.append(point_value)
            source_index = source_channel_indices[column_index]
            target_index = target_channel_indices[column_index]
            source_uv = uvs[source_index]
            target_uv = target_uvs[target_index]
            uvs.append(tuple(
                source_uv[axis] + ((target_uv[axis] - source_uv[axis]) * fraction) for axis in range(2)
            ))
            generated_index = row_index * 2 + column_index
            face_normal = _vec_normalized(accumulated_normals[generated_index], fallback=(0.0, 0.0, 1.0))
            if smooth:
                channel_normal = _vec_normalized(
                    _vec_lerp(normals[source_index], target_normals[target_index], fraction),
                    fallback=face_normal,
                )
                normals.append(_vec_normalized(_vec_add(channel_normal, face_normal), fallback=face_normal))
            else:
                normals.append(face_normal)
            if uvs_lm:
                source_lm = uvs_lm[source_index]
                target_lm = target_uvs_lm[target_index] if target_uvs_lm else source_lm
                uvs_lm.append(tuple(
                    source_lm[axis] + ((target_lm[axis] - source_lm[axis]) * fraction) for axis in range(2)
                ))
    generated_faces = tuple(tuple(base + corner for corner in face) for face in local_triangles)
    faces.extend(generated_faces)
    source_materials = _surface_face_materials(source)
    face_mats = list(source_materials)
    face_mats.extend((source_materials[source_face_index],) * len(generated_faces))
    rebuilt = replace(
        source,
        vertices=tuple(vertices),
        faces=tuple(faces),
        face_mats=tuple(face_mats),
        uvs=tuple(uvs),
        normals=tuple(normals),
        uvs_lm=tuple(uvs_lm),
    )
    surfaces = list(primitive.surfaces)
    surfaces[source_surface_index] = rebuilt
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "bridge_border_edges",
        mesh_role=mesh_role,
        first_edge=[int(value) for value in first_edge],
        target_mesh_role=target_role,
        second_edge=[int(value) for value in second_edge],
        source_face=source_face_index,
        target_face=target_face_index,
        generated_face_start=len(source.faces),
        generated_face_count=len(generated_faces),
        generated_vertex_count=generated_vertex_count,
        divisions=division_count,
        taper=taper_amount,
        twist_degrees=twist_amount,
        smooth=bool(smooth),
        triangulation=cell_diagonals,
        material_slot=source_materials[source_face_index],
        attribute_policy="endpoint_channels_interpolated_normals_smoothed" if smooth else "endpoint_channels_interpolated_face_normals",
    )


def _imported_surface_boolean_operand(surface: ImportedMeshSurface, *, operand: str) -> IndexedPolygonMesh:
    """Adapt one render surface to the generic closed-solid Boolean contract."""

    vertices, faces, uvs, normals, uvs_lm = _surface_arrays(surface)
    vertex_channels: dict[str, AttributeChannel] = {
        "normals": AttributeChannel.build(normals, semantic="normal", default=(0.0, 0.0, 1.0)),
        "uv0": AttributeChannel.build(uvs, default=(0.0, 0.0)),
    }
    if uvs_lm:
        vertex_channels["uvs_lm"] = AttributeChannel.build(uvs_lm, default=(0.0, 0.0))
    return IndexedPolygonMesh.build(
        vertices,
        faces,
        vertex_channels=vertex_channels,
        face_channels={
            "face_material": AttributeChannel.build(_surface_face_materials(surface), default=0),
            "source_operand": AttributeChannel.build((operand,) * len(faces), default=operand),
        },
        metadata={"source_surface": str(surface.name or ""), "operand": operand},
    )


def _boolean_surface_texture(surface: ImportedMeshSurface, material: int) -> str:
    names = tuple(str(value or "").strip() for value in tuple(surface.texture_names or ()))
    index = int(material)
    if 0 <= index < len(names) and names[index]:
        return names[index]
    return str(surface.texture or "").strip()


def boolean_difference_imported_mesh_surfaces(
    primitive: ImportedMeshRoomPrimitive,
    minuend_mesh_role: str,
    cutter_mesh_role: str,
    *,
    weld_tolerance: float = 1.0e-6,
    max_output_triangles: int = 65535,
) -> ImportedMeshRoomPrimitive:
    """Replace two closed imported surfaces with their deterministic A-B result.

    The operation is intentionally strict.  KOTOR room walls and floors are
    commonly open architectural sheets, so they have no unambiguous solid
    inside/outside and are refused.  Valid results are split back into one MDL
    surface per source material; cutter-derived cap faces therefore keep the
    cutter material while surviving A faces keep the minuend material.
    """

    minuend_index = imported_mesh_surface_index_for_role(primitive, minuend_mesh_role)
    cutter_index = imported_mesh_surface_index_for_role(primitive, cutter_mesh_role)
    if minuend_index < 0 or cutter_index < 0:
        raise ValueError("Difference A - B requires two existing imported mesh surfaces.")
    if minuend_index == cutter_index:
        raise ValueError("Difference A - B requires two different closed mesh surfaces.")
    minuend = primitive.surfaces[minuend_index]
    cutter = primitive.surfaces[cutter_index]
    result = difference_closed_solid_meshes(
        _imported_surface_boolean_operand(minuend, operand="A"),
        _imported_surface_boolean_operand(cutter, operand="B"),
        weld_tolerance=max(0.0, float(weld_tolerance)),
        max_output_triangles=max(1, int(max_output_triangles)),
    )
    if not result.ok or result.mesh is None:
        messages = tuple(
            issue.message for issue in result.diagnostics.issues if str(issue.severity).lower() == "error"
        )
        raise ValueError("; ".join(messages) or "Difference A - B could not produce a valid closed solid.")

    mesh = result.mesh
    operand_values = tuple(mesh.face_channels["boolean_source_operand"].values)
    material_values = tuple(mesh.face_channels["face_material"].values)
    grouped: dict[tuple[str, int], list[int]] = {}
    for face_index, (operand_value, material_value) in enumerate(zip(operand_values, material_values)):
        key = (str(operand_value), int(material_value))
        grouped.setdefault(key, []).append(face_index)

    normals_channel = mesh.vertex_channels.get("normals")
    uv_channel = mesh.vertex_channels.get("uv0")
    lightmap_channel = mesh.vertex_channels.get("uvs_lm")
    rebuilt_surfaces: list[ImportedMeshSurface] = []
    for output_index, ((operand, material), face_indices) in enumerate(sorted(grouped.items())):
        raw_to_compact: dict[int, int] = {}
        compact_to_raw: list[int] = []
        compact_faces: list[Face] = []
        for face_index in face_indices:
            compact_face: list[int] = []
            for raw_index in mesh.faces[face_index]:
                if raw_index not in raw_to_compact:
                    raw_to_compact[raw_index] = len(compact_to_raw)
                    compact_to_raw.append(raw_index)
                compact_face.append(raw_to_compact[raw_index])
            compact_faces.append(tuple(compact_face))
        template = minuend if operand == "A" else cutter
        texture = _boolean_surface_texture(template, material)
        rebuilt_surfaces.append(
            replace(
                template,
                name=f"{minuend.name or 'boolean_a'}_minus_{cutter.name or 'boolean_b'}_{output_index + 1}",
                texture=texture,
                vertices=tuple(mesh.vertices[index] for index in compact_to_raw),
                faces=tuple(compact_faces),
                normals=(
                    tuple(normals_channel.values[index] for index in compact_to_raw)
                    if normals_channel is not None
                    else ()
                ),
                uvs=(
                    tuple(uv_channel.values[index] for index in compact_to_raw)
                    if uv_channel is not None
                    else ()
                ),
                # Boolean caps invalidate the old lightmap bake. Preserve the
                # interpolated coordinates as authoring data, but clear the
                # runtime lightmap name until Map Studio rebakes it.
                uvs_lm=(
                    tuple(lightmap_channel.values[index] for index in compact_to_raw)
                    if lightmap_channel is not None
                    else ()
                ),
                lightmap="",
                texture_names=(texture,) if texture else (),
                tex_count=1,
                face_mats=(0,) * len(compact_faces),
            )
        )

    insertion_index = min(minuend_index, cutter_index)
    removed = {minuend_index, cutter_index}
    surfaces: list[ImportedMeshSurface] = []
    for index, surface in enumerate(primitive.surfaces):
        if index == insertion_index:
            surfaces.extend(rebuilt_surfaces)
        if index not in removed:
            surfaces.append(surface)
    edited = replace(primitive, surfaces=tuple(surfaces))
    return _record_topology_edit(
        edited,
        "boolean_difference_closed_solids",
        minuend_mesh_role=str(minuend_mesh_role),
        cutter_mesh_role=str(cutter_mesh_role),
        backend=result.diagnostics.backend,
        backend_version=result.diagnostics.backend_version,
        output_surface_count=len(rebuilt_surfaces),
        output_vertex_count=result.diagnostics.output_vertices,
        output_triangle_count=result.diagnostics.output_triangles,
        output_volume=result.diagnostics.output_volume,
        material_policy="surviving_A_faces_and_cutter_B_caps",
        lightmap_policy="stale_requires_bake",
        topology_contract="closed_oriented_two_manifold_only",
    )


def bend_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str | None = None,
    vertex_indices: tuple[int, ...] | list[int] | None = None,
    *,
    axis: str = "x",
    curvature_degrees: float = 90.0,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> ImportedMeshRoomPrimitive:
    """Bake a bounded circular bend into selected or all room vertices.

    The selected axis is the deformer length. Its next cyclic axis is the bend
    plane (X->Y, Y->Z, Z->X), while the remaining coordinate is unchanged.
    Vertices outside explicit bounds continue along the endpoint tangent, so
    the result is continuous rather than pinched at the bounds.
    """

    selected_positions, selected_raw, resolved_role = _surface_vertex_selection(
        primitive, mesh_role, vertex_indices
    )
    axis_index = _deformer_axis_index(axis)
    transverse_index = (axis_index + 1) % 3
    selected_points = [
        vertex
        for surface in primitive.surfaces
        for vertex in surface.vertices
        if _position_key(vertex) in selected_positions
    ]
    inferred_low = min(point[axis_index] for point in selected_points)
    inferred_high = max(point[axis_index] for point in selected_points)
    low = inferred_low if lower_bound is None else float(lower_bound)
    high = inferred_high if upper_bound is None else float(upper_bound)
    if not all(math.isfinite(value) for value in (low, high)) or high - low <= 1.0e-9:
        raise ValueError("Bend bounds must be finite and span a positive distance.")
    curvature = float(curvature_degrees)
    if not math.isfinite(curvature):
        raise ValueError("Bend curvature must be finite.")
    radians = math.radians(curvature)
    if abs(radians) <= 1.0e-12:
        return primitive
    span = high - low
    radius = span / radians
    transverse_origin = sum(point[transverse_index] for point in selected_points) / len(selected_points)

    def _bend_point(point: Vec3) -> tuple[Vec3, tuple[tuple[float, float, float], ...]]:
        along = point[axis_index] - low
        clamped = min(span, max(0.0, along))
        angle = radians * (clamped / span)
        excess = along - clamped
        sine, cosine = math.sin(angle), math.cos(angle)
        offset = point[transverse_index] - transverse_origin
        bent = list(point)
        bent[axis_index] = low + (radius * sine) + (excess * cosine) - (offset * sine)
        bent[transverse_index] = (
            transverse_origin + (radius * (1.0 - cosine)) + (excess * sine) + (offset * cosine)
        )
        along_scale = 1.0 if excess != 0.0 else 1.0 - (offset / radius)
        jacobian_rows = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
        jacobian_rows[axis_index][axis_index] = along_scale * cosine
        jacobian_rows[axis_index][transverse_index] = -sine
        jacobian_rows[transverse_index][axis_index] = along_scale * sine
        jacobian_rows[transverse_index][transverse_index] = cosine
        return tuple(bent), tuple(tuple(row) for row in jacobian_rows)

    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        has_normals = len(surface.normals) == len(surface.vertices)
        vertices: list[Vec3] = []
        normals: list[Vec3] = [] if has_normals else list(surface.normals)
        changed = False
        for raw_index, point in enumerate(surface.vertices):
            if _position_key(point) not in selected_positions:
                vertices.append(point)
                if has_normals:
                    normals.append(surface.normals[raw_index])
                continue
            bent, jacobian = _bend_point(point)
            vertices.append(bent)
            changed = changed or bent != point
            if has_normals:
                source_normal = surface.normals[raw_index]
                normals.append(_matrix_inverse_transpose_normal(jacobian, source_normal))
        surfaces.append(
            replace(surface, vertices=tuple(vertices), normals=tuple(normals)) if changed else surface
        )
    edited = _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))
    return _record_topology_edit(
        edited,
        "bend_vertices",
        mesh_role=resolved_role,
        vertex_indices=list(selected_raw),
        axis=str(axis).lower(),
        bend_plane_axis=("x", "y", "z")[transverse_index],
        curvature_degrees=curvature,
        lower_bound=low,
        upper_bound=high,
        normal_policy="analytic_jacobian_inverse_transpose",
    )


def lattice_deform_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str | None = None,
    vertex_indices: tuple[int, ...] | list[int] | None = None,
    *,
    control_deltas: tuple[Vec3, ...] | list[Vec3],
    bounds_min: Vec3 | None = None,
    bounds_max: Vec3 | None = None,
) -> ImportedMeshRoomPrimitive:
    """Bake a 2x2x2 FFD lattice with trilinear control-delta interpolation.

    Control order is ``000, 100, 010, 110, 001, 101, 011, 111`` (X changes
    fastest). Normals use the analytic deformation Jacobian inverse transpose;
    UV0, lightmap UV, materials, and vertex/face ordering are unchanged.
    """

    selected_positions, selected_raw, resolved_role = _surface_vertex_selection(
        primitive, mesh_role, vertex_indices
    )
    deltas = tuple(_finite_vec3(delta, label="Lattice control delta") for delta in control_deltas)
    if len(deltas) != 8:
        raise ValueError("A 2x2x2 lattice requires exactly eight control deltas.")
    selected_points = [
        vertex
        for surface in primitive.surfaces
        for vertex in surface.vertices
        if _position_key(vertex) in selected_positions
    ]
    minimum_values = list(
        _finite_vec3(bounds_min, label="Lattice minimum")
        if bounds_min is not None
        else tuple(min(point[axis] for point in selected_points) for axis in range(3))
    )
    maximum_values = list(
        _finite_vec3(bounds_max, label="Lattice maximum")
        if bounds_max is not None
        else tuple(max(point[axis] for point in selected_points) for axis in range(3))
    )
    if any(maximum_values[axis] < minimum_values[axis] for axis in range(3)):
        raise ValueError("Lattice maximum bounds cannot be below minimum bounds.")
    positive_spans = [
        maximum_values[axis] - minimum_values[axis]
        for axis in range(3)
        if maximum_values[axis] - minimum_values[axis] > 1.0e-9
    ]
    reference_extent = max(positive_spans, default=1.0)
    padded_axes: list[str] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        if maximum_values[axis_index] - minimum_values[axis_index] > 1.0e-9:
            continue
        midpoint = (minimum_values[axis_index] + maximum_values[axis_index]) * 0.5
        half_thickness = max(5.0e-4, reference_extent * 0.005)
        minimum_values[axis_index] = midpoint - half_thickness
        maximum_values[axis_index] = midpoint + half_thickness
        padded_axes.append(axis_name)
    minimum = tuple(minimum_values)
    maximum = tuple(maximum_values)
    spans = tuple(maximum[axis] - minimum[axis] for axis in range(3))

    def _weights(point: Vec3) -> tuple[float, float, float]:
        return tuple(
            min(1.0, max(0.0, (point[axis] - minimum[axis]) / spans[axis]))
            for axis in range(3)
        )

    def _delta_and_jacobian(point: Vec3) -> tuple[Vec3, tuple[tuple[float, float, float], ...]]:
        u, v, w = _weights(point)
        one_u, one_v, one_w = 1.0 - u, 1.0 - v, 1.0 - w
        corner_weights = (
            one_u * one_v * one_w,
            u * one_v * one_w,
            one_u * v * one_w,
            u * v * one_w,
            one_u * one_v * w,
            u * one_v * w,
            one_u * v * w,
            u * v * w,
        )
        displacement = tuple(
            sum(deltas[corner][component] * corner_weights[corner] for corner in range(8))
            for component in range(3)
        )
        derivative_u = tuple(
            (
                ((deltas[1][component] - deltas[0][component]) * one_v * one_w)
                + ((deltas[3][component] - deltas[2][component]) * v * one_w)
                + ((deltas[5][component] - deltas[4][component]) * one_v * w)
                + ((deltas[7][component] - deltas[6][component]) * v * w)
            ) / spans[0]
            for component in range(3)
        )
        derivative_v = tuple(
            (
                ((deltas[2][component] - deltas[0][component]) * one_u * one_w)
                + ((deltas[3][component] - deltas[1][component]) * u * one_w)
                + ((deltas[6][component] - deltas[4][component]) * one_u * w)
                + ((deltas[7][component] - deltas[5][component]) * u * w)
            ) / spans[1]
            for component in range(3)
        )
        derivative_w = tuple(
            (
                ((deltas[4][component] - deltas[0][component]) * one_u * one_v)
                + ((deltas[5][component] - deltas[1][component]) * u * one_v)
                + ((deltas[6][component] - deltas[2][component]) * one_u * v)
                + ((deltas[7][component] - deltas[3][component]) * u * v)
            ) / spans[2]
            for component in range(3)
        )
        jacobian = tuple(
            tuple(
                (1.0 if output_axis == input_axis else 0.0)
                + (derivative_u, derivative_v, derivative_w)[input_axis][output_axis]
                for input_axis in range(3)
            )
            for output_axis in range(3)
        )
        return displacement, jacobian

    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        has_normals = len(surface.normals) == len(surface.vertices)
        vertices: list[Vec3] = []
        normals: list[Vec3] = [] if has_normals else list(surface.normals)
        changed = False
        for raw_index, point in enumerate(surface.vertices):
            if _position_key(point) not in selected_positions:
                vertices.append(point)
                if has_normals:
                    normals.append(surface.normals[raw_index])
                continue
            displacement, jacobian = _delta_and_jacobian(point)
            deformed = _vec_add(point, displacement)
            vertices.append(deformed)
            changed = changed or deformed != point
            if has_normals:
                normals.append(_matrix_inverse_transpose_normal(jacobian, surface.normals[raw_index]))
        surfaces.append(
            replace(surface, vertices=tuple(vertices), normals=tuple(normals)) if changed else surface
        )
    edited = _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))
    return _record_topology_edit(
        edited,
        "lattice_deform",
        mesh_role=resolved_role,
        vertex_indices=list(selected_raw),
        control_deltas=[list(delta) for delta in deltas],
        bounds_min=list(minimum),
        bounds_max=list(maximum),
        padded_flat_axes=padded_axes,
        interpolation="trilinear_2x2x2",
        normal_policy="analytic_jacobian_inverse_transpose",
    )


def shrink_wrap_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str | None,
    target_surface: ImportedMeshSurface,
    vertex_indices: tuple[int, ...] | list[int] | None = None,
    *,
    projection: str = "nearest_triangle",
    offset: float = 0.0,
    align_normals: bool = False,
) -> ImportedMeshRoomPrimitive:
    """Bake selected vertices onto the closest target triangle or vertex.

    Ties resolve by the stable target face/vertex index, so repeated bakes are
    byte deterministic. ``align_normals`` is opt-in: disabled preserves the
    source's hard/soft edge decisions, while enabled uses interpolated target
    normals (or the target face normal when no normal channel is available).
    """

    selected_positions, selected_raw, resolved_role = _surface_vertex_selection(
        primitive, mesh_role, vertex_indices
    )
    mode = str(projection or "nearest_triangle").strip().lower().replace("-", "_")
    if mode not in {"nearest_triangle", "nearest_vertex"}:
        raise ValueError("Shrink-wrap projection must be nearest_triangle or nearest_vertex.")
    surface_vertices = tuple(target_surface.vertices)
    if not surface_vertices:
        raise ValueError("Shrink-wrap target surface has no vertices.")
    target_has_normals = len(target_surface.normals) == len(surface_vertices)
    target_faces: list[tuple[int, Face]] = []
    if mode == "nearest_triangle":
        for face_index, face in enumerate(target_surface.faces):
            if any(not 0 <= int(index) < len(surface_vertices) for index in face):
                continue
            a, b, c = (surface_vertices[index] for index in face)
            if _vec_length(_vec_cross(_vec_sub(b, a), _vec_sub(c, a))) <= 1.0e-10:
                continue
            target_faces.append((face_index, face))
        if not target_faces:
            raise ValueError("Shrink-wrap target surface has no valid triangles.")
    distance_offset = float(offset)
    if not math.isfinite(distance_offset):
        raise ValueError("Shrink-wrap offset must be finite.")

    source_point_by_position = {
        _position_key(vertex): vertex
        for surface in primitive.surfaces
        for vertex in surface.vertices
        if _position_key(vertex) in selected_positions
    }
    projected_by_position: dict[tuple[float, float, float], tuple[Vec3, Vec3]] = {}
    for key in sorted(selected_positions):
        point = source_point_by_position[key]
        if mode == "nearest_vertex":
            target_index = min(
                range(len(surface_vertices)),
                key=lambda index: (
                    _vec_dot(_vec_sub(point, surface_vertices[index]), _vec_sub(point, surface_vertices[index])),
                    index,
                ),
            )
            projected = surface_vertices[target_index]
            if target_has_normals:
                target_normal = _vec_normalized(target_surface.normals[target_index])
            else:
                normal_sum = (0.0, 0.0, 0.0)
                for face in target_surface.faces:
                    if target_index in face and all(0 <= index < len(surface_vertices) for index in face):
                        a, b, c = (surface_vertices[index] for index in face)
                        normal_sum = _vec_add(
                            normal_sum,
                            _vec_normalized(_vec_cross(_vec_sub(b, a), _vec_sub(c, a))),
                        )
                target_normal = _vec_normalized(normal_sum)
        else:
            best: tuple[float, int, Vec3, tuple[float, float, float], Face] | None = None
            for face_index, face in target_faces:
                a, b, c = (surface_vertices[index] for index in face)
                candidate, barycentric = _nearest_point_on_triangle(point, a, b, c)
                delta = _vec_sub(point, candidate)
                candidate_record = (_vec_dot(delta, delta), face_index, candidate, barycentric, face)
                if best is None or candidate_record[:2] < best[:2]:
                    best = candidate_record
            assert best is not None
            _distance_squared, _face_index, projected, barycentric, face = best
            if target_has_normals:
                target_normal = _vec_normalized(
                    tuple(
                        sum(
                            target_surface.normals[face[corner]][axis] * barycentric[corner]
                            for corner in range(3)
                        )
                        for axis in range(3)
                    )
                )
            else:
                a, b, c = (surface_vertices[index] for index in face)
                target_normal = _vec_normalized(_vec_cross(_vec_sub(b, a), _vec_sub(c, a)))
        projected_by_position[key] = (_vec_add(projected, _vec_scale(target_normal, distance_offset)), target_normal)

    surfaces: list[ImportedMeshSurface] = []
    for surface in primitive.surfaces:
        has_normals = len(surface.normals) == len(surface.vertices)
        vertices: list[Vec3] = []
        normals: list[Vec3] = [] if has_normals else list(surface.normals)
        changed = False
        for raw_index, point in enumerate(surface.vertices):
            projection_result = projected_by_position.get(_position_key(point))
            if projection_result is None:
                vertices.append(point)
                if has_normals:
                    normals.append(surface.normals[raw_index])
                continue
            projected, target_normal = projection_result
            vertices.append(projected)
            changed = changed or projected != point
            if has_normals:
                normals.append(target_normal if align_normals else surface.normals[raw_index])
        surfaces.append(
            replace(surface, vertices=tuple(vertices), normals=tuple(normals)) if changed else surface
        )
    edited = _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))
    return _record_topology_edit(
        edited,
        "shrink_wrap",
        mesh_role=resolved_role,
        vertex_indices=list(selected_raw),
        projection=mode,
        offset=distance_offset,
        align_normals=bool(align_normals),
        target_surface=str(target_surface.name),
        target_vertex_count=len(target_surface.vertices),
        target_face_count=len(target_surface.faces),
        acceleration_policy="deterministic_exhaustive_bake",
    )


def wrap_deform_imported_mesh_vertices(
    primitive: ImportedMeshRoomPrimitive,
    mesh_role: str | None,
    driver_base: ImportedMeshSurface,
    driver_deformed: ImportedMeshSurface,
    vertex_indices: tuple[int, ...] | list[int] | None = None,
    *,
    nearest_count: int = 4,
    influence: float = 1.0,
    max_distance: float = 0.0,
) -> ImportedMeshRoomPrimitive:
    """Bake a deterministic inverse-distance driver-delta wrap.

    This is the static Map Studio subset of Maya Wrap: each target point takes
    an inverse-distance blend of the nearest driver vertex displacements. It
    intentionally does not persist a live dependency graph; the KOTOR room
    exporter receives only baked vertices and the lightweight audit metadata.
    """

    selected_positions, selected_raw, resolved_role = _surface_vertex_selection(
        primitive, mesh_role, vertex_indices
    )
    base_vertices = tuple(driver_base.vertices)
    deformed_vertices = tuple(driver_deformed.vertices)
    if not base_vertices or len(base_vertices) != len(deformed_vertices):
        raise ValueError("Wrap driver base and deformed surfaces require the same non-zero vertex count.")
    count = max(1, min(int(nearest_count), len(base_vertices)))
    strength = float(influence)
    limit = float(max_distance)
    if not math.isfinite(strength) or not math.isfinite(limit) or limit < 0.0:
        raise ValueError("Wrap influence and maximum distance must be finite; distance cannot be negative.")
    driver_deltas = tuple(
        _vec_sub(deformed_vertices[index], base_vertices[index]) for index in range(len(base_vertices))
    )
    source_point_by_position = {
        _position_key(vertex): vertex
        for surface in primitive.surfaces
        for vertex in surface.vertices
        if _position_key(vertex) in selected_positions
    }
    displacement_by_position: dict[tuple[float, float, float], Vec3] = {}
    for key in sorted(selected_positions):
        point = source_point_by_position[key]
        candidates = sorted(
            (
                _vec_dot(_vec_sub(point, base), _vec_sub(point, base)),
                index,
            )
            for index, base in enumerate(base_vertices)
        )
        if limit > 0.0:
            candidates = [candidate for candidate in candidates if candidate[0] <= limit * limit]
        if not candidates:
            continue
        nearest = candidates[:count]
        if nearest[0][0] <= 1.0e-18:
            displacement = driver_deltas[nearest[0][1]]
        else:
            weights = tuple(1.0 / max(1.0e-9, math.sqrt(distance_squared)) for distance_squared, _index in nearest)
            total_weight = sum(weights)
            displacement = tuple(
                sum(driver_deltas[index][axis] * weight for weight, (_distance, index) in zip(weights, nearest))
                / total_weight
                for axis in range(3)
            )
        displacement_by_position[key] = _vec_scale(displacement, strength)
    if not displacement_by_position:
        raise ValueError("No selected vertex lies within the wrap driver's maximum distance.")

    surfaces: list[ImportedMeshSurface] = []
    affected_count = 0
    for surface in primitive.surfaces:
        vertices: list[Vec3] = []
        changed = False
        for point in surface.vertices:
            displacement = displacement_by_position.get(_position_key(point))
            if displacement is None:
                vertices.append(point)
                continue
            deformed = _vec_add(point, displacement)
            vertices.append(deformed)
            changed = changed or deformed != point
            affected_count += 1
        surfaces.append(replace(surface, vertices=tuple(vertices)) if changed else surface)
    edited = _drop_degenerate_faces(replace(primitive, surfaces=tuple(surfaces)))
    return _record_topology_edit(
        edited,
        "wrap_deform",
        mesh_role=resolved_role,
        vertex_indices=list(selected_raw),
        driver_base=str(driver_base.name),
        driver_deformed=str(driver_deformed.name),
        driver_vertex_count=len(base_vertices),
        nearest_count=count,
        influence=strength,
        max_distance=limit,
        affected_vertex_records=affected_count,
        interpolation="inverse_distance_driver_vertex_delta",
        normal_policy="source_normals_preserved",
        dependency_policy="baked_no_live_driver_graph",
    )


def planar_uvs_for_vertices(vertices: tuple[Vec3, ...]) -> tuple[Vec2, ...]:
    """Dominant-axis planar projection normalized to [0, 1].

    Only for previews/thumbnails.  Architecture UVs must use
    ``tiled_uvs_for_vertices`` so KOTOR tiling textures repeat at world-space
    density like vanilla rooms instead of stretching one repeat across the
    whole surface.
    """

    if not vertices:
        return ()
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    spans = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    drop = spans.index(min(spans))
    axes = [axis for axis in (0, 1, 2) if axis != drop]
    u_values = [v[axes[0]] for v in vertices]
    v_values = [v[axes[1]] for v in vertices]
    u_min, v_min = min(u_values), min(v_values)
    u_span = max(1.0e-6, max(u_values) - u_min)
    v_span = max(1.0e-6, max(v_values) - v_min)
    return tuple(((u - u_min) / u_span, (v - v_min) / v_span) for u, v in zip(u_values, v_values))


#: Default world meters covered by one texture repeat when no vanilla
#: reference surface is available to sample the density from.
DEFAULT_UV_TILE_SIZE = 2.0


def tiled_uvs_for_vertices(vertices: tuple[Vec3, ...], *, tile_size: float = DEFAULT_UV_TILE_SIZE) -> tuple[Vec2, ...]:
    """World-space dominant-axis projection: one texture repeat per ``tile_size`` meters.

    This is how vanilla KOTOR architecture is mapped — tiling level textures
    at consistent world density — so new geometry textured this way blends
    with surrounding stock rooms.
    """

    if not vertices:
        return ()
    scale = 1.0 / max(1.0e-6, float(tile_size))
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    spans = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    drop = spans.index(min(spans))
    axes = [axis for axis in (0, 1, 2) if axis != drop]
    return tuple((v[axes[0]] * scale, v[axes[1]] * scale) for v in vertices)


def surface_uv_tile_size(surface: ImportedMeshSurface) -> float:
    """Estimate a surface's world meters per UV repeat (0.0 when unknown).

    Sampling a vanilla surface's density lets new geometry tile at the same
    rate as the room it extends.
    """

    if len(surface.uvs) != len(surface.vertices) or not surface.faces:
        return 0.0
    total_world = 0.0
    total_uv = 0.0
    for a, b, c in surface.faces:
        try:
            pa, pb, pc = surface.vertices[a], surface.vertices[b], surface.vertices[c]
            ta, tb, tc = surface.uvs[a], surface.uvs[b], surface.uvs[c]
        except IndexError:
            continue
        abx, aby, abz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        acx, acy, acz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        cx = (aby * acz) - (abz * acy)
        cy = (abz * acx) - (abx * acz)
        cz = (abx * acy) - (aby * acx)
        total_world += 0.5 * ((cx * cx) + (cy * cy) + (cz * cz)) ** 0.5
        u1, v1 = tb[0] - ta[0], tb[1] - ta[1]
        u2, v2 = tc[0] - ta[0], tc[1] - ta[1]
        total_uv += 0.5 * abs((u1 * v2) - (u2 * v1))
    if total_uv <= 1.0e-9 or total_world <= 1.0e-9:
        return 0.0
    return (total_world / total_uv) ** 0.5


def matched_uv_tile_size(
    primitive: ImportedMeshRoomPrimitive,
    *,
    texture: str = "",
    default: float = DEFAULT_UV_TILE_SIZE,
) -> float:
    """Tile size matching an existing surface (same texture first, then any)."""

    wanted = str(texture or "").strip().lower()
    fallback = 0.0
    for surface in primitive.surfaces:
        tile = surface_uv_tile_size(surface)
        if tile <= 0.0:
            continue
        if wanted and str(surface.texture or "").lower() == wanted:
            return tile
        if fallback <= 0.0:
            fallback = tile
    return fallback if fallback > 0.0 else float(default)


# ---------------------------------------------------------------------------
# Stock model import


def _stock_source_light_descriptor(node: Any, source_node_index: int) -> dict[str, Any]:
    """Return a small, JSON-safe preview record for one vanilla room light.

    Converting a stock room deliberately flattens its editable render meshes,
    but the live editor still needs the room's original Aurora lights to shade
    dynamic actors and non-lightmapped surfaces.  Keep only renderer-facing
    values here; the runtime-graph export gate remains ``preserved=False`` and
    therefore cannot mistake this preview record for an engine-safe MDL node
    round trip.
    """

    def _float_tuple(value: Any, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
        try:
            result = tuple(float(item) for item in tuple(value)[:size])
        except (TypeError, ValueError):
            result = ()
        if len(result) != size or not all(math.isfinite(item) for item in result):
            return default
        return result

    try:
        world_position, world_orientation = node.world_transform()
    except Exception:
        world_position = getattr(node, "position", (0.0, 0.0, 0.0))
        world_orientation = getattr(
            node,
            "rotation",
            getattr(node, "orientation", (0.0, 0.0, 0.0, 1.0)),
        )
    controller_types: list[int] = []
    for controller in tuple(getattr(node, "controllers", ()) or ()):
        try:
            value = controller.get("type", 0) if isinstance(controller, dict) else getattr(controller, "type", 0)
            controller_types.append(int(value or 0))
        except (TypeError, ValueError):
            continue
    return {
        "schema": "ghostrigger.stock_room_light_preview.v1",
        "source_node_index": int(source_node_index),
        "source_node_name": str(getattr(node, "name", "") or f"room_light_{source_node_index + 1}"),
        "position_space": "room_local",
        "position": list(_float_tuple(world_position, 3, (0.0, 0.0, 0.0))),
        "orientation": list(_float_tuple(world_orientation, 4, (0.0, 0.0, 0.0, 1.0))),
        "color": list(_float_tuple(getattr(node, "light_color", (1.0, 1.0, 1.0)), 3, (1.0, 1.0, 1.0))),
        "radius": max(0.001, float(getattr(node, "light_radius", 5.0) or 5.0)),
        "multiplier": max(0.0, float(getattr(node, "light_multiplier", 1.0) or 1.0)),
        "kind": str(getattr(node, "light_kind", "point") or "point"),
        "enabled": bool(getattr(node, "light_enabled", True)),
        "ambient_only": bool(getattr(node, "light_ambient_only", False)),
        "dynamic_type": int(getattr(node, "light_dynamic", 0) or 0),
        "shadow": bool(getattr(node, "light_shadow", False)),
        "flare": bool(getattr(node, "light_flare", False)),
        "fading": bool(getattr(node, "light_fading", False)),
        "controller_types": controller_types,
        "preview_only": True,
    }


def build_imported_mesh_primitive_from_stock_model(
    model: Any,
    *,
    room_resref: str,
    source_model: str,
    game: str = "K1",
    wok_bytes: bytes | None = None,
) -> ImportedMeshRoomPrimitive:
    """Bake a loaded KotorModel into an editable imported-mesh primitive.

    Baking rules match the stock viewport preview: world transforms applied,
    non-render and AABB (walkmesh proxy) nodes skipped, one surface per
    trimesh node.  Lightmap names are retained as authoring metadata even
    though the fullbright export path does not re-emit them yet.
    """

    try:
        from src.core.geometry import model_data as _md  # type: ignore

        aabb_flag = int(_md.NodeFlags.AABB)
        skin_flag = int(_md.NodeFlags.SKIN)
        light_flag = int(_md.NodeFlags.LIGHT)
        emitter_flag = int(_md.NodeFlags.EMITTER)
        reference_flag = int(_md.NodeFlags.REFERENCE)
    except Exception:
        try:
            from core.geometry import model_data as _md  # type: ignore

            aabb_flag = int(_md.NodeFlags.AABB)
            skin_flag = int(_md.NodeFlags.SKIN)
            light_flag = int(_md.NodeFlags.LIGHT)
            emitter_flag = int(_md.NodeFlags.EMITTER)
            reference_flag = int(_md.NodeFlags.REFERENCE)
        except Exception:
            aabb_flag = 0x0200
            skin_flag = 0x0020
            light_flag = 0x0002
            emitter_flag = 0x0004
            reference_flag = 0x0010

    surfaces: list[ImportedMeshSurface] = []
    root = getattr(model, "root_node", None)
    source_nodes: list[Any] = []
    source_stack = [root] if root is not None else []
    while source_stack:
        source_node = source_stack.pop()
        source_nodes.append(source_node)
        source_stack.extend(tuple(getattr(source_node, "children", ()) or ()))
    source_light_nodes = [
        _stock_source_light_descriptor(node, index)
        for index, node in enumerate(source_nodes)
        if int(getattr(node, "flags", 0) or 0) & light_flag
    ]
    source_runtime_graph = {
        "node_count": len(source_nodes),
        "animation_count": len(tuple(getattr(model, "animations", ()) or ())),
        "animation_names": [
            str(getattr(animation, "name", "") or "")
            for animation in tuple(getattr(model, "animations", ()) or ())
        ],
        "light_count": len(source_light_nodes),
        "light_nodes": source_light_nodes,
        "emitter_count": sum(1 for node in source_nodes if int(getattr(node, "flags", 0) or 0) & emitter_flag),
        "reference_count": sum(1 for node in source_nodes if int(getattr(node, "flags", 0) or 0) & reference_flag),
        "controller_count": sum(len(tuple(getattr(node, "controllers", ()) or ())) for node in source_nodes),
        "preserved": False,
    }
    # Traversal order and skip rules must match the stock preview's
    # _flattened_mesh_nodes so hover roles ("stock_room_<i>") map 1:1 onto
    # imported surface indices.
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        stack.extend(tuple(getattr(node, "children", ()) or ()))
        vertices = tuple(getattr(node, "vertices", ()) or ())
        faces = tuple(getattr(node, "faces", ()) or ())
        if not vertices or not faces:
            continue
        if not bool(getattr(node, "render", True)):
            continue
        flags = int(getattr(node, "flags", 0) or 0)
        if flags & aabb_flag:
            continue
        try:
            (wx, wy, wz), wq = node.world_transform()
        except Exception:
            (wx, wy, wz), wq = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        qx, qy, qz, qw = (float(v) for v in wq)
        is_skin = bool(flags & skin_flag)

        def _rotate(point: tuple[float, float, float]) -> tuple[float, float, float]:
            px, py, pz = point
            tx = 2.0 * ((qy * pz) - (qz * py))
            ty = 2.0 * ((qz * px) - (qx * pz))
            tz = 2.0 * ((qx * py) - (qy * px))
            return (
                px + (qw * tx) + ((qy * tz) - (qz * ty)),
                py + (qw * ty) + ((qz * tx) - (qx * tz)),
                pz + (qw * tz) + ((qx * ty) - (qy * tx)),
            )

        def _bake(point: tuple[float, float, float]) -> tuple[float, float, float]:
            if is_skin:
                return (point[0] + wx, point[1] + wy, point[2] + wz)
            rotated = _rotate(point)
            return (rotated[0] + wx, rotated[1] + wy, rotated[2] + wz)

        baked_vertices = tuple(_bake((float(v[0]), float(v[1]), float(v[2]))) for v in vertices)
        normals = tuple(getattr(node, "normals", ()) or ())
        baked_normals = (
            tuple(
                (float(n[0]), float(n[1]), float(n[2])) if is_skin else _rotate((float(n[0]), float(n[1]), float(n[2])))
                for n in normals
            )
            if len(normals) == len(vertices)
            else ()
        )
        uvs = tuple(tuple(float(value) for value in uv[:2]) for uv in tuple(getattr(node, "uvs", ()) or ()))
        if len(uvs) != len(vertices):
            uvs = ()
        uvs_lm = tuple(tuple(float(value) for value in uv[:2]) for uv in tuple(getattr(node, "uvs_lm", ()) or ()))
        if len(uvs_lm) != len(vertices):
            uvs_lm = ()
        texture_names = tuple(str(name) for name in tuple(getattr(node, "texture_names", ()) or ()) if str(name).strip())
        texture = str(getattr(node, "texture", "") or "")
        if not texture and texture_names:
            texture = texture_names[0]
        imported_surface = ImportedMeshSurface(
            name=str(getattr(node, "name", "") or f"{room_resref}_srf{len(surfaces)}"),
            texture=texture,
            vertices=baked_vertices,
            faces=tuple(tuple(int(value) for value in face[:3]) for face in faces),
            face_mats=tuple(int(value) for value in tuple(getattr(node, "face_mats", ()) or ())),
            uvs=uvs,
            normals=baked_normals,
            lightmap=str(getattr(node, "lightmap", "") or ""),
            diffuse=tuple(float(v) for v in tuple(getattr(node, "diffuse", (1.0, 1.0, 1.0)))[:3]),
            ambient=tuple(float(v) for v in tuple(getattr(node, "ambient", (1.0, 1.0, 1.0)))[:3]),
            texture_names=texture_names or ((texture,) if texture else ()),
            tex_count=max(1, int(getattr(node, "tex_count", 1) or 1)),
            uvs_lm=uvs_lm,
            specular=tuple(float(v) for v in tuple(getattr(node, "specular", (0.0, 0.0, 0.0)))[:3]),
            shininess=float(getattr(node, "shininess", 0.0) or 0.0),
            alpha=float(getattr(node, "alpha", 1.0) if getattr(node, "alpha", None) is not None else 1.0),
            has_shadow=bool(getattr(node, "has_shadow", True)),
            render=bool(getattr(node, "render", True)),
            selfillum=tuple(float(v) for v in tuple(getattr(node, "selfillum", (0.0, 0.0, 0.0)))[:3]),
            transparency_hint=int(getattr(node, "transparency_hint", 0) or 0),
            beaming=bool(getattr(node, "beaming", False)),
            background_geometry=bool(getattr(node, "background_geometry", False)),
            rotate_texture=bool(getattr(node, "rotate_texture", False)),
            animate_uv=bool(getattr(node, "animate_uv", False)),
            uv_dir_x=float(getattr(node, "uv_dir_x", 0.0) or 0.0),
            uv_dir_y=float(getattr(node, "uv_dir_y", 0.0) or 0.0),
            uv_jitter=float(getattr(node, "uv_jitter", 0.0) or 0.0),
            uv_jitter_speed=float(getattr(node, "uv_jitter_speed", 0.0) or 0.0),
            dirt_enabled=bool(getattr(node, "dirt_enabled", False)),
            dirt_texture=int(getattr(node, "dirt_texture", 0) or 0),
            dirt_coord_space=int(getattr(node, "dirt_coord_space", 0) or 0),
            hide_in_holograms=bool(getattr(node, "hide_in_holograms", False)),
            mesh_average_point=tuple(float(v) for v in tuple(getattr(node, "mesh_average_point", (0.0, 0.0, 0.0)))[:3]),
            mesh_unknown0=bytes(getattr(node, "mesh_unknown0", b"") or b"")[:24].ljust(24, b"\x00"),
        )
        surfaces.append(
            replace(
                imported_surface,
                backdrop=imported_mesh_surface_is_backdrop(imported_surface),
            )
        )
    wok = None
    if wok_bytes:
        try:
            parsed = WOKData.from_bytes(bytes(wok_bytes))
            # A zero-face WOK is meaningful source data for vanilla visual-only
            # sky rooms (for example K1 m02aa_sky). Discarding it makes compile
            # synthesize a thousand-unit walkable floor and an AABB node.
            wok = parsed
        except Exception:
            wok = None
    return ImportedMeshRoomPrimitive(
        room_resref=str(room_resref or ""),
        surfaces=tuple(surfaces),
        source_model=str(source_model or ""),
        game=str(game or "K1").upper(),
        wok=wok,
        metadata={
            "imported_from": str(source_model or ""),
            "surface_count": len(surfaces),
            "source_runtime_graph": source_runtime_graph,
            # Retail room WOK vertices are already in module/area coordinates;
            # unlike authored geometry they must not receive the LYT position
            # again when combined for pathing or simulation.
            "wok_coordinate_space": "module" if wok is not None else "room_local",
        },
    )


# ---------------------------------------------------------------------------
# KMAP payload (base64-packed arrays)


def _pack_floats(values, dims: int) -> str:
    flat: list[float] = []
    for item in values:
        flat.extend(float(v) for v in tuple(item)[:dims])
    return base64.b64encode(struct.pack(f"<{len(flat)}f", *flat)).decode("ascii")


def _unpack_floats(payload: str, dims: int) -> tuple[tuple[float, ...], ...]:
    raw = base64.b64decode(str(payload or ""))
    count = len(raw) // 4
    flat = struct.unpack(f"<{count}f", raw)
    return tuple(tuple(flat[i : i + dims]) for i in range(0, count - (count % dims), dims))


def _pack_ints(values, dims: int) -> str:
    flat: list[int] = []
    for item in values:
        flat.extend(int(v) for v in tuple(item)[:dims])
    return base64.b64encode(struct.pack(f"<{len(flat)}i", *flat)).decode("ascii")


def _unpack_ints(payload: str, dims: int) -> tuple[tuple[int, ...], ...]:
    raw = base64.b64decode(str(payload or ""))
    count = len(raw) // 4
    flat = struct.unpack(f"<{count}i", raw)
    return tuple(tuple(flat[i : i + dims]) for i in range(0, count - (count % dims), dims))


def imported_mesh_primitive_payload(primitive: ImportedMeshRoomPrimitive) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": IMPORTED_MESH_PRIMITIVE_KIND,
        "kind": IMPORTED_MESH_PRIMITIVE_KIND,
        "room_resref": str(primitive.room_resref or ""),
        "source_model": str(primitive.source_model or ""),
        "game": str(primitive.game or "K1"),
        "metadata": dict(primitive.metadata),
        "surfaces": [
            {
                "name": str(surface.name or ""),
                "texture": str(surface.texture or ""),
                "lightmap": str(surface.lightmap or ""),
                "diffuse": [float(v) for v in surface.diffuse],
                "ambient": [float(v) for v in surface.ambient],
                "vertex_count": len(surface.vertices),
                "face_count": len(surface.faces),
                "vertices_b64": _pack_floats(surface.vertices, 3),
                "faces_b64": _pack_ints(surface.faces, 3),
                "face_mats_b64": _pack_ints(tuple((value,) for value in surface.face_mats), 1) if surface.face_mats else "",
                "uvs_b64": _pack_floats(surface.uvs, 2) if surface.uvs else "",
                "normals_b64": _pack_floats(surface.normals, 3) if surface.normals else "",
                "texture_names": [str(name) for name in surface.texture_names],
                "tex_count": max(1, int(surface.tex_count or 1)),
                "uvs_lm_b64": _pack_floats(surface.uvs_lm, 2) if surface.uvs_lm else "",
                "specular": [float(v) for v in surface.specular],
                "shininess": float(surface.shininess),
                "alpha": float(surface.alpha),
                "has_shadow": bool(surface.has_shadow),
                "render": bool(surface.render),
                "selfillum": [float(v) for v in surface.selfillum],
                "transparency_hint": int(surface.transparency_hint),
                "beaming": bool(surface.beaming),
                "background_geometry": bool(surface.background_geometry),
                "rotate_texture": bool(surface.rotate_texture),
                "animate_uv": bool(surface.animate_uv),
                "uv_dir_x": float(surface.uv_dir_x),
                "uv_dir_y": float(surface.uv_dir_y),
                "uv_jitter": float(surface.uv_jitter),
                "uv_jitter_speed": float(surface.uv_jitter_speed),
                "dirt_enabled": bool(surface.dirt_enabled),
                "dirt_texture": int(surface.dirt_texture),
                "dirt_coord_space": int(surface.dirt_coord_space),
                "hide_in_holograms": bool(surface.hide_in_holograms),
                "mesh_average_point": [float(v) for v in surface.mesh_average_point],
                "mesh_unknown0_b64": base64.b64encode(bytes(surface.mesh_unknown0 or b"")[:24].ljust(24, b"\x00")).decode("ascii"),
                "backdrop": bool(imported_mesh_surface_is_backdrop(surface)),
            }
            for surface in primitive.surfaces
        ],
    }
    if primitive.wok is not None:
        payload["wok"] = {
            "header_version": 2,
            "relative_hook1": [float(value) for value in primitive.wok.relative_hook1],
            "relative_hook2": [float(value) for value in primitive.wok.relative_hook2],
            "absolute_hook1": [float(value) for value in primitive.wok.absolute_hook1],
            "absolute_hook2": [float(value) for value in primitive.wok.absolute_hook2],
            "position": [float(value) for value in primitive.wok.position],
            "adjacency_domain_count": (
                None
                if primitive.wok.adjacency_domain_count is None
                else int(primitive.wok.adjacency_domain_count)
            ),
            "verts_b64": _pack_floats(primitive.wok.verts, 3),
            "face_stride": 10,
            "faces_b64": _pack_ints(
                [
                    (
                        face.v1,
                        face.v2,
                        face.v3,
                        face.surface,
                        face.adj1,
                        face.adj2,
                        face.adj3,
                        face.trans1,
                        face.trans2,
                        face.trans3,
                    )
                    for face in primitive.wok.faces
                ],
                10,
            ),
        }
    return payload


def imported_mesh_primitive_from_payload(data: dict[str, Any], room_resref: str) -> ImportedMeshRoomPrimitive:
    surfaces: list[ImportedMeshSurface] = []
    for entry in tuple(data.get("surfaces") or ()):
        surface = ImportedMeshSurface(
            name=str(entry.get("name") or ""),
            texture=str(entry.get("texture") or ""),
            vertices=_unpack_floats(entry.get("vertices_b64") or "", 3),
            faces=tuple(tuple(int(v) for v in face) for face in _unpack_ints(entry.get("faces_b64") or "", 3)),
            face_mats=tuple(
                int(row[0])
                for row in _unpack_ints(entry.get("face_mats_b64") or "", 1)
            ) if entry.get("face_mats_b64") else (),
            uvs=_unpack_floats(entry.get("uvs_b64") or "", 2) if entry.get("uvs_b64") else (),
            normals=_unpack_floats(entry.get("normals_b64") or "", 3) if entry.get("normals_b64") else (),
            lightmap=str(entry.get("lightmap") or ""),
            diffuse=tuple(float(v) for v in tuple(entry.get("diffuse") or (1.0, 1.0, 1.0))[:3]),
            ambient=tuple(float(v) for v in tuple(entry.get("ambient") or (1.0, 1.0, 1.0))[:3]),
            texture_names=tuple(str(name) for name in tuple(entry.get("texture_names") or ())),
            tex_count=max(1, int(entry.get("tex_count") or 1)),
            uvs_lm=_unpack_floats(entry.get("uvs_lm_b64") or "", 2) if entry.get("uvs_lm_b64") else (),
            specular=tuple(float(v) for v in tuple(entry.get("specular") or (0.0, 0.0, 0.0))[:3]),
            shininess=float(entry.get("shininess") or 0.0),
            alpha=float(entry.get("alpha", 1.0) if entry.get("alpha") is not None else 1.0),
            has_shadow=bool(entry.get("has_shadow", True)),
            render=bool(entry.get("render", True)),
            selfillum=tuple(float(v) for v in tuple(entry.get("selfillum") or (0.0, 0.0, 0.0))[:3]),
            transparency_hint=int(entry.get("transparency_hint") or 0),
            beaming=bool(entry.get("beaming", False)),
            background_geometry=bool(entry.get("background_geometry", False)),
            rotate_texture=bool(entry.get("rotate_texture", False)),
            animate_uv=bool(entry.get("animate_uv", False)),
            uv_dir_x=float(entry.get("uv_dir_x") or 0.0),
            uv_dir_y=float(entry.get("uv_dir_y") or 0.0),
            uv_jitter=float(entry.get("uv_jitter") or 0.0),
            uv_jitter_speed=float(entry.get("uv_jitter_speed") or 0.0),
            dirt_enabled=bool(entry.get("dirt_enabled", False)),
            dirt_texture=int(entry.get("dirt_texture") or 0),
            dirt_coord_space=int(entry.get("dirt_coord_space") or 0),
            hide_in_holograms=bool(entry.get("hide_in_holograms", False)),
            mesh_average_point=tuple(float(v) for v in tuple(entry.get("mesh_average_point") or (0.0, 0.0, 0.0))[:3]),
            mesh_unknown0=(
                base64.b64decode(str(entry.get("mesh_unknown0_b64") or ""))[:24].ljust(24, b"\x00")
                if entry.get("mesh_unknown0_b64")
                else b"\x00" * 24
            ),
            backdrop=bool(entry.get("backdrop", False)),
        )
        if "backdrop" not in entry:
            surface = replace(surface, backdrop=imported_mesh_surface_is_backdrop(surface))
        surfaces.append(surface)
    wok = None
    wok_data = data.get("wok")
    if isinstance(wok_data, dict):
        def _header_vec3(key: str) -> tuple[float, float, float]:
            values = tuple(wok_data.get(key) or (0.0, 0.0, 0.0))
            return tuple(float(values[index]) if index < len(values) else 0.0 for index in range(3))  # type: ignore[return-value]

        verts = [tuple(v) for v in _unpack_floats(wok_data.get("verts_b64") or "", 3)]
        face_stride = 10 if int(wok_data.get("face_stride") or 7) >= 10 else 7
        wok = WOKData(
            name=str(room_resref or ""),
            verts=verts,
            relative_hook1=_header_vec3("relative_hook1"),
            relative_hook2=_header_vec3("relative_hook2"),
            absolute_hook1=_header_vec3("absolute_hook1"),
            absolute_hook2=_header_vec3("absolute_hook2"),
            position=_header_vec3("position"),
            adjacency_domain_count=(
                None
                if wok_data.get("adjacency_domain_count") is None
                else int(wok_data.get("adjacency_domain_count"))
            ),
            faces=[
                WOKFace(
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    surface=int(row[3]),
                    adj1=int(row[4]),
                    adj2=int(row[5]),
                    adj3=int(row[6]),
                    trans1=int(row[7]) if face_stride >= 10 else -1,
                    trans2=int(row[8]) if face_stride >= 10 else -1,
                    trans3=int(row[9]) if face_stride >= 10 else -1,
                )
                for row in _unpack_ints(wok_data.get("faces_b64") or "", face_stride)
            ],
        )
    return ImportedMeshRoomPrimitive(
        room_resref=str(room_resref or data.get("room_resref") or ""),
        surfaces=tuple(surfaces),
        source_model=str(data.get("source_model") or ""),
        game=str(data.get("game") or "K1").upper(),
        wok=wok,
        metadata=dict(data.get("metadata") or {}),
    )


__all__ = [
    "DEFAULT_UV_TILE_SIZE",
    "EMPIRICAL_STOCK_WOK_FACE_MAX",
    "IMPORTED_MESH_PRIMITIVE_KIND",
    "LOGICAL_QUAD_PROVENANCE_KEY",
    "LOGICAL_QUAD_PROVENANCE_VERSION",
    "MDL_MAX_VERTICES_PER_SURFACE",
    "ROOM_TRIANGLE_WARNING_BUDGET",
    "SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_POLICY",
    "SOURCE_RUNTIME_GRAPH_STATIC_REBUILD_VERSION",
    "ImportedMeshBevelOptions",
    "ImportedMeshRoomPrimitive",
    "ImportedMeshSurface",
    "ImportedMeshValidation",
    "append_imported_mesh_quad",
    "bend_imported_mesh_vertices",
    "boolean_difference_imported_mesh_surfaces",
    "bridge_imported_mesh_border_edges",
    "build_imported_mesh_primitive_from_stock_model",
    "bevel_imported_mesh_edge",
    "bevel_imported_mesh_edges",
    "collapse_imported_mesh_edge",
    "compile_imported_mesh_room_geometry",
    "connect_imported_mesh_vertices",
    "delete_imported_mesh_edge_faces",
    "delete_imported_mesh_faces",
    "delete_imported_mesh_vertex_faces",
    "extrude_imported_mesh_edge",
    "extrude_imported_mesh_faces",
    "extract_imported_mesh_room_bounds",
    "flatten_imported_mesh_faces",
    "fill_imported_mesh_boundary_loop",
    "flip_imported_mesh_faces",
    "harden_imported_mesh_edges",
    "imported_mesh_primitive_from_payload",
    "imported_mesh_primitive_payload",
    "imported_mesh_room_is_backdrop",
    "imported_mesh_room_is_visual_only",
    "imported_mesh_has_explicit_static_runtime_rebuild",
    "imported_mesh_source_runtime_counts",
    "imported_mesh_surface_is_backdrop",
    "imported_mesh_surface_index_for_role",
    "imported_mesh_surface_role",
    "inset_imported_mesh_faces",
    "insert_imported_mesh_edge_loop",
    "lattice_deform_imported_mesh_vertices",
    "make_hole_in_imported_mesh_face",
    "matched_uv_tile_size",
    "mirror_imported_mesh_geometry",
    "move_imported_mesh_edge",
    "move_imported_mesh_faces",
    "move_imported_mesh_vertex",
    "planar_uvs_for_vertices",
    "prepare_imported_mesh_for_static_runtime_rebuild",
    "resolve_imported_mesh_face_target",
    "set_imported_mesh_face_texture",
    "set_imported_mesh_edge_smoothing",
    "set_imported_mesh_face_smoothing",
    "shrink_wrap_imported_mesh_vertices",
    "soften_imported_mesh_edges",
    "split_imported_mesh_edge",
    "split_imported_mesh_face",
    "split_imported_mesh_face_at_point",
    "surface_uv_tile_size",
    "tiled_uvs_for_vertices",
    "validate_imported_mesh_room_primitive",
    "weld_imported_mesh_vertex",
    "weld_imported_mesh_vertices",
    "wrap_deform_imported_mesh_vertices",
]
