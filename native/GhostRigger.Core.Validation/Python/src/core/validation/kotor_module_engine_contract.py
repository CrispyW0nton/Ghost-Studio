"""Vanilla-derived binary contract gates for Map Studio module export.

The Odyssey engine is the authority for room resources.  This module inspects
the serialized bytes that the game will consume; it deliberately does not
regenerate derived walkmesh data or trust a GhostRigger/PyKotor round trip as
proof that an artifact is loadable.

Archive and game-install loading stay outside Core.Validation.  Callers supply
the resource bytes for one module and receive an all-room validation report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import struct
from typing import Any, Iterable, Mapping

from .validation_bus import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationSubsystem,
)


_MDL_BASE = 12
_MDL_NODE_SIZE = 80
_MDL_MIN_SIZE = _MDL_BASE + 196
_MDL_AABB_FLAG = 0x0200
_MAX_REASONABLE_COUNT = 1_000_000
_BWM_HEADER_SIZE = 136
_BWM_AREA_TYPE = 1
_BWM_WALKABLE_MATERIALS = frozenset({1, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 30})

_MODEL_FUNCTION_POINTERS: dict[str, tuple[int, int]] = {
    "K1": (4_273_776, 4_216_096),
    "K2": (4_285_200, 4_216_320),
}

# These fingerprints were measured directly from known-loadable game assets.
# Candidate geometry counts are intentionally not required to match; the raw
# invariants they share are the actual contract enforced below.
VANILLA_ROOM_BASELINES: tuple[dict[str, Any], ...] = (
    {
        "id": "K1:models.bif:m02aa_sky",
        "game": "K1",
        "aabb_node_count": 0,
        "wok_vertex_count": 0,
        "wok_face_count": 0,
        "visual_only": True,
    },
    {
        "id": "K1:models.bif:plcaa",
        "game": "K1",
        "node_count": 43,
        "aabb_node_count": 1,
        "nonzero_node_plus_8": 0,
        "wok_vertex_count": 19,
        "wok_face_count": 18,
        "wok_edge_count": 18,
        "wok_perimeter_count": 1,
    },
    {
        "id": "K2:models.bif:001ebo1",
        "game": "K2",
        "node_count": 60,
        "aabb_node_count": 1,
        "nonzero_node_plus_8": 0,
        "wok_vertex_count": 42,
        "wok_face_count": 62,
        "wok_edge_count": 22,
        "wok_perimeter_count": 1,
    },
    {
        "id": "K2:tst_light:r00_test",
        "game": "K2",
        "node_count": 21,
        "aabb_node_count": 1,
        "nonzero_node_plus_8": 0,
        "wok_vertex_count": 54,
        "wok_face_count": 88,
        "wok_edge_count": 26,
        "wok_perimeter_count": 2,
    },
)


@dataclass(frozen=True)
class MdlEngineFingerprint:
    """Raw static-model properties compared with vanilla room invariants."""

    model_name: str = ""
    function_pointer_1: int = 0
    function_pointer_2: int = 0
    geometry_type: int = 0
    declared_node_count: int = 0
    visited_node_count: int = 0
    aabb_node_count: int = 0
    controller_count: int = 0
    nonzero_node_plus_8: int = 0
    mdx_size: int = 0


@dataclass(frozen=True)
class WokEngineFingerprint:
    """Raw BWM properties retained without regenerating derived tables."""

    walkmesh_type: int = 0
    vertex_count: int = 0
    face_count: int = 0
    walkable_face_count: int = 0
    aabb_count: int = 0
    aabb_leaf_count: int = 0
    aabb_covered_face_count: int = 0
    aabb_missing_face_count: int = 0
    aabb_reachable_count: int = 0
    adjacency_count: int = 0
    adjacency_mismatch_count: int = 0
    adjacency_nonreciprocal_count: int = 0
    non_manifold_edge_count: int = 0
    edge_count: int = 0
    perimeter_count: int = 0
    closed_perimeter_count: int = 0
    transition_count: int = 0
    nonzero_hook_value_count: int = 0
    material_histogram: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RoomEngineFingerprint:
    """All serialized engine resources belonging to one LYT room."""

    room_resref: str
    mdl: MdlEngineFingerprint = field(default_factory=MdlEngineFingerprint)
    wok: WokEngineFingerprint = field(default_factory=WokEngineFingerprint)


@dataclass(frozen=True)
class KotorModuleEngineContractRequest:
    """Byte inputs for an all-room module contract check."""

    game: str
    module_resref: str
    resources: Mapping[tuple[str, str], bytes]
    expected_room_resrefs: tuple[str, ...] = ()
    # Explicit per-room exception measured from vanilla visual-only LYT rooms.
    # Playable rooms retain the strict embedded-AABB and non-empty-WOK gate.
    visual_only_room_resrefs: tuple[str, ...] = ()


@dataclass
class KotorModuleEngineContractReport:
    """Engine gate result and JSON-safe structural fingerprints."""

    validation: ValidationReport
    rooms: tuple[RoomEngineFingerprint, ...] = ()
    lyt_rooms: tuple[str, ...] = ()
    are_rooms: tuple[str, ...] = ()
    vis_link_count: int = 0
    pth_point_count: int = 0
    pth_connection_count: int = 0
    placeable_template_count: int = 0
    bundled_placeable_count: int = 0
    placeable_templates: tuple[dict[str, Any], ...] = ()

    @property
    def export_ready(self) -> bool:
        return not self.validation.has_errors

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            issue.message
            for issue in self.validation.issues
            if issue.severity == ValidationSeverity.WARNING
        )

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        return tuple(
            issue.message
            for issue in self.validation.issues
            if issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.BLOCKING}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ghostrigger.map_studio_engine_contract.v1",
            "export_ready": self.export_ready,
            "reference_baselines": [dict(row) for row in VANILLA_ROOM_BASELINES],
            "lyt_rooms": list(self.lyt_rooms),
            "are_rooms": list(self.are_rooms),
            "vis_link_count": self.vis_link_count,
            "pth_point_count": self.pth_point_count,
            "pth_connection_count": self.pth_connection_count,
            "placeable_template_count": self.placeable_template_count,
            "bundled_placeable_count": self.bundled_placeable_count,
            "placeable_templates": [dict(row) for row in self.placeable_templates],
            "rooms": [
                {
                    "room_resref": room.room_resref,
                    "mdl": asdict(room.mdl),
                    "wok": asdict(room.wok),
                }
                for room in self.rooms
            ],
            "warnings": list(self.warnings),
            "blocking_issues": list(self.blocking_issues),
            "issues": [
                {
                    "severity": issue.severity.value,
                    "subsystem": issue.subsystem.value,
                    "code": issue.code,
                    "message": issue.message,
                    "details": dict(issue.details),
                    "fix_hint": issue.fix_hint,
                }
                for issue in self.validation.issues
            ],
        }


def _normalise_resref(value: Any) -> str:
    return str(value or "").strip().lower()[:16]


def _normalise_game(value: Any) -> str:
    return "K2" if str(value or "").strip().upper() == "K2" else "K1"


def _resource_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    data = getattr(value, "data", None)
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise TypeError(f"Engine-contract resource value is not bytes: {type(value).__name__}")


def _normalise_resources(resources: Mapping[tuple[str, str], bytes]) -> dict[tuple[str, str], bytes]:
    result: dict[tuple[str, str], bytes] = {}
    for key, value in dict(resources or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        resref = _normalise_resref(key[0])
        restype = str(key[1] or "").strip().lower()
        if resref and restype:
            result[(resref, restype)] = _resource_bytes(value)
    return result


def _add_issue(
    report: ValidationReport,
    severity: ValidationSeverity,
    code: str,
    message: str,
    *,
    resource: str = "",
    details: Mapping[str, Any] | None = None,
    fix_hint: str | None = None,
) -> None:
    payload = dict(details or {})
    if resource:
        payload.setdefault("resource", resource)
    report.add(
        ValidationIssue(
            severity=severity,
            subsystem=ValidationSubsystem.MAP,
            code=code,
            message=message,
            details=payload,
            fix_hint=fix_hint,
            source="map_studio.engine_contract",
        )
    )


def _bounded(data: bytes, offset: int, size: int) -> bool:
    return offset >= 0 and size >= 0 and offset <= len(data) and size <= len(data) - offset


def _read_c_string(data: bytes, offset: int) -> str:
    if not _bounded(data, offset, 1):
        return ""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("ascii", errors="replace")


def inspect_raw_mdl_structure(
    room_resref: str,
    mdl: bytes,
    mdx: bytes,
    *,
    game: str,
    allow_missing_aabb: bool = False,
) -> tuple[MdlEngineFingerprint, ValidationReport]:
    """Inspect a static room MDL exactly as serialized for the engine."""

    room = _normalise_resref(room_resref)
    resource = f"{room}.mdl"
    report = ValidationReport(source="map_studio.engine_contract")
    if len(mdl) < _MDL_MIN_SIZE:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.header_truncated",
            f"{resource} is too small for an Odyssey model header.",
            resource=resource,
        )
        return MdlEngineFingerprint(mdx_size=len(mdx)), report

    payload_size, declared_mdx_size = struct.unpack_from("<II", mdl, 4)
    fp1, fp2 = struct.unpack_from("<II", mdl, _MDL_BASE)
    model_name = mdl[_MDL_BASE + 8 : _MDL_BASE + 40].split(b"\0", 1)[0].decode("ascii", errors="replace")
    root_rel, declared_nodes = struct.unpack_from("<II", mdl, _MDL_BASE + 40)
    geometry_type = int(mdl[_MDL_BASE + 76] & 0x7F)
    model_header_mdx_size = struct.unpack_from("<I", mdl, _MDL_BASE + 80 + 96)[0]
    name_table_rel = struct.unpack_from("<I", mdl, _MDL_BASE + 80 + 104)[0]
    name_count, name_count2 = struct.unpack_from("<II", mdl, _MDL_BASE + 80 + 108)

    if payload_size + _MDL_BASE != len(mdl):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.payload_size_mismatch",
            f"{resource} declares {payload_size} payload bytes but contains {len(mdl) - _MDL_BASE}.",
            resource=resource,
        )
    if declared_mdx_size != len(mdx) or model_header_mdx_size != len(mdx):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.mdx_size_mismatch",
            f"{resource} does not agree with {room}.mdx length {len(mdx)}.",
            resource=resource,
            details={
                "file_header_mdx_size": declared_mdx_size,
                "model_header_mdx_size": model_header_mdx_size,
                "actual_mdx_size": len(mdx),
            },
        )
    expected_fp = _MODEL_FUNCTION_POINTERS[_normalise_game(game)]
    if (fp1, fp2) != expected_fp:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.function_pointer_mismatch",
            f"{resource} uses function pointers {(fp1, fp2)}, expected {expected_fp} for {_normalise_game(game)}.",
            resource=resource,
        )
    if geometry_type != 2:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.geometry_type_invalid",
            f"{resource} geometry type is {geometry_type}; a room model must be type 2.",
            resource=resource,
        )
    if declared_nodes <= 0 or declared_nodes > _MAX_REASONABLE_COUNT:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.node_count_invalid",
            f"{resource} declares an invalid node count of {declared_nodes}.",
            resource=resource,
        )

    names: list[str] = []
    if name_count != name_count2:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.name_count_mismatch",
            f"{resource} duplicated name counts disagree: {name_count} != {name_count2}.",
            resource=resource,
        )
    name_table_abs = _MDL_BASE + name_table_rel
    if name_count > _MAX_REASONABLE_COUNT or not _bounded(mdl, name_table_abs, name_count * 4):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.name_table_out_of_bounds",
            f"{resource} name table is outside the MDL payload.",
            resource=resource,
        )
    else:
        for index in range(name_count):
            name_rel = struct.unpack_from("<I", mdl, name_table_abs + index * 4)[0]
            names.append(_read_c_string(mdl, _MDL_BASE + name_rel))

    visited: set[int] = set()
    active: set[int] = set()
    aabb_count = 0
    controller_count = 0
    nonzero_plus_8 = 0

    def walk(node_rel: int, expected_parent_rel: int) -> None:
        nonlocal aabb_count, controller_count, nonzero_plus_8
        node_abs = _MDL_BASE + node_rel
        if node_rel == 0 or not _bounded(mdl, node_abs, _MDL_NODE_SIZE):
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.node_offset_out_of_bounds",
                f"{resource} references invalid node offset 0x{node_rel:x}.",
                resource=resource,
                details={"node_offset": node_rel},
            )
            return
        if node_rel in active:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.node_cycle",
                f"{resource} contains a node cycle at offset 0x{node_rel:x}.",
                resource=resource,
            )
            return
        if node_rel in visited:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.node_reused",
                f"{resource} references node offset 0x{node_rel:x} more than once.",
                resource=resource,
            )
            return

        active.add(node_rel)
        visited.add(node_rel)
        flags, node_number, name_index = struct.unpack_from("<HHH", mdl, node_abs)
        runtime_geometry_pointer = struct.unpack_from("<I", mdl, node_abs + 8)[0]
        parent_rel = struct.unpack_from("<I", mdl, node_abs + 12)[0]
        transform = struct.unpack_from("<7f", mdl, node_abs + 16)
        child_array_rel, child_count, child_count2 = struct.unpack_from("<III", mdl, node_abs + 44)
        controller_array_rel, controller_count1, controller_count2 = struct.unpack_from("<III", mdl, node_abs + 56)
        controller_data_rel, controller_data_count, controller_data_count2 = struct.unpack_from("<III", mdl, node_abs + 68)
        node_name = names[name_index] if 0 <= name_index < len(names) else f"<name:{name_index}>"

        if runtime_geometry_pointer != 0:
            nonzero_plus_8 += 1
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.node_geometry_pointer_nonzero",
                f"{resource} node '{node_name}' has nonzero runtime field +8 (0x{runtime_geometry_pointer:x}).",
                resource=resource,
                details={"node": node_name, "node_offset": node_rel, "value": runtime_geometry_pointer},
                fix_hint="Serialize zero at static node-header +8; the engine populates this runtime pointer.",
            )
        if parent_rel != expected_parent_rel:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.parent_pointer_mismatch",
                f"{resource} node '{node_name}' parent is 0x{parent_rel:x}, expected 0x{expected_parent_rel:x}.",
                resource=resource,
            )
        if name_index >= len(names):
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.name_index_out_of_bounds",
                f"{resource} node at 0x{node_rel:x} uses missing name index {name_index}.",
                resource=resource,
            )
        # This field is not a dense array index in all known-loadable rooms.
        # K2 r00_test legitimately uses values 21, 22, 23, 24 and 32 with only
        # 21 traversed nodes, so range/uniqueness restrictions would validate
        # against our writer rather than against vanilla.
        _ = node_number
        if not all(math.isfinite(value) for value in transform):
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.non_finite_transform",
                f"{resource} node '{node_name}' contains a non-finite transform.",
                resource=resource,
            )
        if flags & _MDL_AABB_FLAG:
            aabb_count += 1
            mesh_abs = node_abs + _MDL_NODE_SIZE
            mesh_size = 340 if _normalise_game(game) == "K2" else 332
            if not _bounded(mdl, mesh_abs, mesh_size):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.mdl.aabb_mesh_header_truncated",
                    f"{resource} AABB node '{node_name}' has a truncated mesh header.",
                    resource=resource,
                )
            else:
                faces_rel, face_count1, face_count2 = struct.unpack_from("<III", mdl, mesh_abs + 8)
                vertex_count = struct.unpack_from("<H", mdl, mesh_abs + 304)[0]
                vertices_rel_offset = 336 if _normalise_game(game) == "K2" else 328
                vertices_rel = struct.unpack_from("<I", mdl, mesh_abs + vertices_rel_offset)[0]
                faces_abs = _MDL_BASE + faces_rel
                vertices_abs = _MDL_BASE + vertices_rel
                if face_count1 != face_count2:
                    _add_issue(
                        report,
                        ValidationSeverity.BLOCKING,
                        "map.engine.mdl.aabb_face_count_mismatch",
                        f"{resource} AABB node '{node_name}' duplicated face counts disagree.",
                        resource=resource,
                    )
                elif (
                    face_count1 > _MAX_REASONABLE_COUNT
                    or vertex_count > _MAX_REASONABLE_COUNT
                    or not _bounded(mdl, faces_abs, face_count1 * 32)
                    or not _bounded(mdl, vertices_abs, vertex_count * 12)
                ):
                    _add_issue(
                        report,
                        ValidationSeverity.BLOCKING,
                        "map.engine.mdl.aabb_geometry_out_of_bounds",
                        f"{resource} AABB node '{node_name}' face/vertex payload is out of bounds.",
                        resource=resource,
                    )
                else:
                    vertices = [
                        struct.unpack_from("<3f", mdl, vertices_abs + index * 12)
                        for index in range(vertex_count)
                    ]
                    invalid_planes = 0
                    invalid_indices = 0
                    for index in range(face_count1):
                        face_abs = faces_abs + index * 32
                        nx, ny, nz, coefficient = struct.unpack_from("<4f", mdl, face_abs)
                        v1, v2, v3 = struct.unpack_from("<HHH", mdl, face_abs + 26)
                        if max(v1, v2, v3) >= vertex_count:
                            invalid_indices += 1
                            continue
                        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
                        ax, ay, az = vertices[v1]
                        residual = (nx * ax) + (ny * ay) + (nz * az) + coefficient
                        tolerance = max(1.0e-4, 2.0e-5 * (abs(coefficient) + 1.0))
                        if (
                            not all(math.isfinite(value) for value in (nx, ny, nz, coefficient, residual))
                            or abs(magnitude - 1.0) > 1.0e-3
                            or abs(residual) > tolerance
                        ):
                            invalid_planes += 1
                    if invalid_indices:
                        _add_issue(
                            report,
                            ValidationSeverity.BLOCKING,
                            "map.engine.mdl.aabb_face_index_out_of_bounds",
                            f"{resource} AABB node '{node_name}' has {invalid_indices} face(s) with invalid vertex indices.",
                            resource=resource,
                        )
                    if invalid_planes:
                        _add_issue(
                            report,
                            ValidationSeverity.BLOCKING,
                            "map.engine.mdl.aabb_face_plane_invalid",
                            (
                                f"{resource} AABB node '{node_name}' has {invalid_planes} face plane(s) "
                                "that do not satisfy vanilla's n·p+d=0 contract."
                            ),
                            resource=resource,
                            fix_hint="Write each face coefficient as -dot(unit_normal, a face vertex).",
                        )

        if child_count != child_count2:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.child_count_mismatch",
                f"{resource} node '{node_name}' duplicated child counts disagree.",
                resource=resource,
            )
        child_offsets: list[int] = []
        child_array_abs = _MDL_BASE + child_array_rel
        if child_count:
            if child_count > _MAX_REASONABLE_COUNT or child_array_rel == 0 or not _bounded(mdl, child_array_abs, child_count * 4):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.mdl.child_array_out_of_bounds",
                    f"{resource} node '{node_name}' child array is outside the MDL payload.",
                    resource=resource,
                )
            else:
                child_offsets = [
                    struct.unpack_from("<I", mdl, child_array_abs + index * 4)[0]
                    for index in range(child_count)
                ]

        if controller_count1 != controller_count2:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.controller_count_mismatch",
                f"{resource} node '{node_name}' duplicated controller counts disagree.",
                resource=resource,
            )
        if controller_data_count != controller_data_count2:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.mdl.controller_data_count_mismatch",
                f"{resource} node '{node_name}' duplicated controller-data counts disagree.",
                resource=resource,
            )
        controller_count += controller_count1
        if controller_count1:
            controller_array_abs = _MDL_BASE + controller_array_rel
            if controller_array_rel == 0 or not _bounded(mdl, controller_array_abs, controller_count1 * 16):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.mdl.controller_array_out_of_bounds",
                    f"{resource} node '{node_name}' controller array is outside the MDL payload.",
                    resource=resource,
                )
        if controller_data_count:
            controller_data_abs = _MDL_BASE + controller_data_rel
            if controller_data_rel == 0 or not _bounded(mdl, controller_data_abs, controller_data_count * 4):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.mdl.controller_data_out_of_bounds",
                    f"{resource} node '{node_name}' controller-data pool is outside the MDL payload.",
                    resource=resource,
                )

        for child_rel in child_offsets:
            walk(child_rel, node_rel)
        active.remove(node_rel)

    walk(root_rel, 0)
    if len(visited) != declared_nodes:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.node_count_mismatch",
            f"{resource} declares {declared_nodes} nodes but raw traversal reached {len(visited)}.",
            resource=resource,
        )
    if aabb_count < 1 and not allow_missing_aabb:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.mdl.missing_aabb",
            f"{resource} has no embedded AABB walkmesh node.",
            resource=resource,
            fix_hint="Embed the room WOK as an AABB node before writing the room MDL.",
        )
    if model_name and _normalise_resref(model_name) != room:
        _add_issue(
            report,
            ValidationSeverity.WARNING,
            "map.engine.mdl.internal_name_mismatch",
            f"{resource} internal model name is '{model_name}', not '{room}'.",
            resource=resource,
        )

    return (
        MdlEngineFingerprint(
            model_name=model_name,
            function_pointer_1=fp1,
            function_pointer_2=fp2,
            geometry_type=geometry_type,
            declared_node_count=declared_nodes,
            visited_node_count=len(visited),
            aabb_node_count=aabb_count,
            controller_count=controller_count,
            nonzero_node_plus_8=nonzero_plus_8,
            mdx_size=len(mdx),
        ),
        report,
    )


def _check_bwm_section(
    report: ValidationReport,
    data: bytes,
    *,
    resource: str,
    code: str,
    label: str,
    offset: int,
    count: int,
    stride: int,
) -> bool:
    if count > _MAX_REASONABLE_COUNT or not _bounded(data, offset, count * stride):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            code,
            f"{resource} {label} table is outside the WOK payload.",
            resource=resource,
            details={"offset": offset, "count": count, "stride": stride, "size": len(data)},
        )
        return False
    return True


def inspect_raw_wok_structure(
    room_resref: str,
    data: bytes,
    *,
    allow_empty_visual: bool = False,
    transition_room_count: int | None = None,
) -> tuple[WokEngineFingerprint, ValidationReport]:
    """Inspect raw BWM tables, including serialized perimeter loop records.

    ``transition_room_count`` supplies the owning LYT room count when this WOK
    is validated as part of a module.  Transition destinations are room-order
    indices, so values outside that range are engine-invalid even when the BWM
    is otherwise structurally sound.
    """

    room = _normalise_resref(room_resref)
    resource = f"{room}.wok"
    report = ValidationReport(source="map_studio.engine_contract")
    if len(data) < _BWM_HEADER_SIZE:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.header_truncated",
            f"{resource} is too small for a BWM V1.0 header.",
            resource=resource,
        )
        return WokEngineFingerprint(), report
    if data[:8] != b"BWM V1.0":
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.signature_invalid",
            f"{resource} is not a BWM V1.0 area walkmesh.",
            resource=resource,
        )

    walkmesh_type = struct.unpack_from("<I", data, 8)[0]
    hooks = struct.unpack_from("<15f", data, 12)
    (
        vertex_count,
        vertex_offset,
        face_count,
        face_offset,
        material_offset,
        normal_offset,
        plane_offset,
        aabb_count,
        aabb_offset,
        aabb_root,
        adjacency_count,
        adjacency_offset,
        edge_count,
        edge_offset,
        perimeter_count,
        perimeter_offset,
    ) = struct.unpack_from("<16I", data, 72)

    if walkmesh_type != _BWM_AREA_TYPE:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.type_invalid",
            f"{resource} walkmesh type is {walkmesh_type}; a room WOK must be area type 1.",
            resource=resource,
        )
    empty_visual = bool(allow_empty_visual and vertex_count == 0 and face_count == 0)
    if (vertex_count <= 0 or face_count <= 0) and not empty_visual:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.geometry_empty",
            f"{resource} has {vertex_count} vertices and {face_count} faces.",
            resource=resource,
        )

    vertices_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.vertex_table_out_of_bounds",
        label="vertex", offset=vertex_offset, count=vertex_count, stride=12,
    )
    faces_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.face_table_out_of_bounds",
        label="face", offset=face_offset, count=face_count, stride=12,
    )
    materials_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.material_table_out_of_bounds",
        label="material", offset=material_offset, count=face_count, stride=4,
    )
    _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.normal_table_out_of_bounds",
        label="normal", offset=normal_offset, count=face_count, stride=12,
    )
    _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.plane_table_out_of_bounds",
        label="plane", offset=plane_offset, count=face_count, stride=4,
    )
    aabb_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.aabb_table_out_of_bounds",
        label="AABB", offset=aabb_offset, count=aabb_count, stride=44,
    )
    adjacency_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.adjacency_table_out_of_bounds",
        label="adjacency", offset=adjacency_offset, count=adjacency_count, stride=12,
    )
    edges_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.edge_table_out_of_bounds",
        label="edge", offset=edge_offset, count=edge_count, stride=8,
    )
    perimeters_ok = _check_bwm_section(
        report, data, resource=resource, code="map.engine.wok.perimeter_table_out_of_bounds",
        label="perimeter", offset=perimeter_offset, count=perimeter_count, stride=4,
    )

    if not empty_visual and (aabb_count < 1 or aabb_root >= max(aabb_count, 1)):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.aabb_invalid",
            f"{resource} has no valid BWM AABB tree.",
            resource=resource,
            details={"aabb_count": aabb_count, "aabb_root": aabb_root},
        )

    faces: list[tuple[int, int, int]] = []
    if faces_ok:
        for index in range(face_count):
            face = struct.unpack_from("<III", data, face_offset + index * 12)
            faces.append(face)
            if any(vertex >= vertex_count for vertex in face):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.wok.face_vertex_out_of_bounds",
                    f"{resource} face {index} references a missing vertex.",
                    resource=resource,
                    details={"face_index": index, "vertices": list(face), "vertex_count": vertex_count},
                )

    if vertices_ok:
        for index in range(vertex_count):
            vertex = struct.unpack_from("<3f", data, vertex_offset + index * 12)
            if not all(math.isfinite(value) for value in vertex):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.wok.non_finite_vertex",
                    f"{resource} vertex {index} contains a non-finite value.",
                    resource=resource,
                )

    aabb_leaf_faces: list[int] = []
    aabb_covered_face_count = 0
    aabb_missing_face_count = face_count
    aabb_reachable_count = 0
    if aabb_ok and aabb_count:
        aabb_nodes: list[tuple[int, int, int, int]] = []
        invalid_bounds = invalid_unknown = invalid_planes = invalid_children = invalid_leaves = 0
        valid_planes = {0, 1, 2, 4, 8, 16, 32}
        for index in range(aabb_count):
            node_offset = aabb_offset + index * 44
            bounds = struct.unpack_from("<6f", data, node_offset)
            face_index, unknown, sigplane, left, right = struct.unpack_from("<IIIII", data, node_offset + 24)
            if not all(math.isfinite(value) for value in bounds) or any(bounds[axis] > bounds[axis + 3] for axis in range(3)):
                invalid_bounds += 1
            if unknown != 4:
                invalid_unknown += 1
            if sigplane not in valid_planes:
                invalid_planes += 1
            is_leaf = face_index != 0xFFFFFFFF
            if is_leaf:
                if face_index >= face_count or sigplane != 0 or left != 0xFFFFFFFF or right != 0xFFFFFFFF:
                    invalid_leaves += 1
                else:
                    aabb_leaf_faces.append(face_index)
            elif sigplane == 0 or left >= aabb_count or right >= aabb_count or left == right:
                invalid_children += 1
            aabb_nodes.append((face_index, sigplane, left, right))

        stack = [aabb_root] if aabb_root < aabb_count else []
        reachable: set[int] = set()
        cycle_count = 0
        while stack:
            node_index = stack.pop()
            if node_index in reachable:
                cycle_count += 1
                continue
            reachable.add(node_index)
            face_index, _sigplane, left, right = aabb_nodes[node_index]
            if face_index == 0xFFFFFFFF:
                if left < aabb_count:
                    stack.append(left)
                if right < aabb_count:
                    stack.append(right)
        aabb_reachable_count = len(reachable)
        for count, code, message in (
            (invalid_bounds, "map.engine.wok.aabb_bounds_invalid", "AABB node(s) contain non-finite or inverted bounds"),
            (invalid_unknown, "map.engine.wok.aabb_unknown_invalid", "AABB node(s) do not carry vanilla's constant value 4"),
            (invalid_planes, "map.engine.wok.aabb_plane_invalid", "AABB node(s) use an unknown most-significant-plane bit"),
            (invalid_children, "map.engine.wok.aabb_children_invalid", "AABB internal node(s) have invalid children"),
            (invalid_leaves, "map.engine.wok.aabb_leaf_invalid", "AABB leaf node(s) have an invalid face or leaf layout"),
            (cycle_count, "map.engine.wok.aabb_cycle", "AABB traversal encountered repeated/cyclic node references"),
        ):
            if count:
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    code,
                    f"{resource} {message} ({count}).",
                    resource=resource,
                )
        if len(reachable) != aabb_count:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.aabb_unreachable_nodes",
                f"{resource} AABB root reaches {len(reachable)} of {aabb_count} nodes.",
                resource=resource,
            )
        covered_faces = set(aabb_leaf_faces)
        missing_faces = set(range(face_count)) - covered_faces
        aabb_covered_face_count = len(covered_faces)
        aabb_missing_face_count = len(missing_faces)
        duplicate_leaf_count = max(0, len(aabb_leaf_faces) - len(covered_faces))
        known_retail_exception = bool(
            room == "m02af_01a"
            and face_count == 140
            and aabb_count == 219
            and len(covered_faces) == 110
            and len(missing_faces) == 30
        )
        if missing_faces:
            _add_issue(
                report,
                ValidationSeverity.WARNING if known_retail_exception else ValidationSeverity.BLOCKING,
                "map.engine.wok.aabb_face_coverage_incomplete",
                f"{resource} AABB leaves cover {len(covered_faces)} of {face_count} faces.",
                resource=resource,
                details={"missing_face_count": len(missing_faces), "known_retail_exception": known_retail_exception},
            )
        if duplicate_leaf_count:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.aabb_face_coverage_duplicate",
                f"{resource} AABB tree repeats {duplicate_leaf_count} face leaf/leaves.",
                resource=resource,
            )
        expected_node_count = max(0, (2 * len(aabb_leaf_faces)) - 1)
        if aabb_count != expected_node_count:
            _add_issue(
                report,
                ValidationSeverity.WARNING if known_retail_exception else ValidationSeverity.BLOCKING,
                "map.engine.wok.aabb_tree_not_full",
                f"{resource} has {aabb_count} AABB nodes for {len(aabb_leaf_faces)} leaves; expected {expected_node_count}.",
                resource=resource,
            )

    histogram: dict[int, int] = {}
    if materials_ok:
        for index in range(face_count):
            material = struct.unpack_from("<I", data, material_offset + index * 4)[0]
            histogram[material] = histogram.get(material, 0) + 1
    walkable_face_count = sum(count for material, count in histogram.items() if material in _BWM_WALKABLE_MATERIALS)
    visual_collision_only = bool(allow_empty_visual and walkable_face_count <= 0 and adjacency_count <= 0)
    if not visual_collision_only and (adjacency_count <= 0 or adjacency_count > face_count):
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.walkable_adjacency_invalid",
            f"{resource} declares {adjacency_count} walkable adjacency rows for {face_count} faces.",
            resource=resource,
        )
    if walkable_face_count <= 0 and not visual_collision_only and adjacency_count <= 0:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.no_walkable_faces",
            f"{resource} contains no recognized walkable floor faces.",
            resource=resource,
        )
    elif adjacency_count != walkable_face_count:
        _add_issue(
            report,
            ValidationSeverity.WARNING,
            "map.engine.wok.walkable_count_disagrees",
            f"{resource} raw adjacency count {adjacency_count} differs from material-derived walkable count {walkable_face_count}.",
            resource=resource,
        )

    adjacency_rows: list[tuple[int, int, int]] = []
    adjacency_mismatch_slots: set[tuple[int, int]] = set()
    adjacency_nonreciprocal_slots: set[tuple[int, int]] = set()
    non_manifold_edges: set[tuple[int, int]] = set()
    expected_boundary_ids: set[int] = set()
    if adjacency_ok:
        adjacency_rows = [
            struct.unpack_from("<iii", data, adjacency_offset + index * 12)
            for index in range(adjacency_count)
        ]

    # Vanilla topology is keyed by explicit vertex indices.  Equal coordinates
    # are allowed to remain intentionally disconnected, so never weld them here.
    if faces and adjacency_rows:
        domain_count = min(adjacency_count, len(faces))
        edge_owners: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        repeated_index_slots: set[tuple[int, int]] = set()
        for face_index in range(domain_count):
            triangle = faces[face_index]
            if any(vertex >= vertex_count for vertex in triangle):
                continue
            for local_edge in range(3):
                start = triangle[local_edge]
                end = triangle[(local_edge + 1) % 3]
                if start == end:
                    repeated_index_slots.add((face_index, local_edge))
                    # Retail m38aa_11 serializes two self-edge boundary rows.
                    # Preserve/import them, but never infer adjacency from them.
                    if adjacency_rows[face_index][local_edge] == -1:
                        expected_boundary_ids.add(face_index * 3 + local_edge)
                    continue
                key = (min(start, end), max(start, end))
                edge_owners.setdefault(key, []).append((face_index, local_edge, start, end))

        for key, owners in edge_owners.items():
            if len(owners) > 2:
                non_manifold_edges.add(key)
                continue
            if len(owners) == 2 and owners[0][0] == owners[1][0]:
                # K1 m38aa_11 has a repeated-index face whose two remaining
                # half-edges oppose each other on that same face.  Retail keeps
                # both as perimeter boundaries rather than self-adjacency.
                for face_index, local_edge, _start, _end in owners:
                    repeated_index_slots.add((face_index, local_edge))
                    if adjacency_rows[face_index][local_edge] == -1:
                        expected_boundary_ids.add(face_index * 3 + local_edge)
                continue
            for owner_index, (face_index, local_edge, _start, _end) in enumerate(owners):
                if len(owners) == 1:
                    expected = -1
                    expected_boundary_ids.add(face_index * 3 + local_edge)
                else:
                    other = owners[1 - owner_index]
                    expected = other[0] * 3 + other[1]
                if adjacency_rows[face_index][local_edge] != expected:
                    adjacency_mismatch_slots.add((face_index, local_edge))

        for face_index in range(domain_count):
            triangle = faces[face_index]
            if any(vertex >= vertex_count for vertex in triangle):
                continue
            for local_edge, target in enumerate(adjacency_rows[face_index]):
                if (face_index, local_edge) in repeated_index_slots:
                    continue
                start = triangle[local_edge]
                end = triangle[(local_edge + 1) % 3]
                key = (min(start, end), max(start, end))
                if key in non_manifold_edges:
                    continue
                if target == -1:
                    continue
                if target < 0:
                    adjacency_mismatch_slots.add((face_index, local_edge))
                    continue
                target_face, target_edge = divmod(target, 3)
                if target_face >= domain_count:
                    adjacency_mismatch_slots.add((face_index, local_edge))
                    continue
                target_triangle = faces[target_face]
                target_start = target_triangle[target_edge]
                target_end = target_triangle[(target_edge + 1) % 3]
                if target_start != end or target_end != start:
                    adjacency_mismatch_slots.add((face_index, local_edge))
                    continue
                if adjacency_rows[target_face][target_edge] != face_index * 3 + local_edge:
                    adjacency_nonreciprocal_slots.add((face_index, local_edge))

        if non_manifold_edges:
            known_retail_non_manifold = bool(
                (room == "m38aa_03" and non_manifold_edges == {(6, 7)})
                or (room == "403dxne" and non_manifold_edges == {(264, 270)})
            )
            _add_issue(
                report,
                ValidationSeverity.WARNING if known_retail_non_manifold else ValidationSeverity.BLOCKING,
                "map.engine.wok.non_manifold_adjacency_edge",
                f"{resource} has {len(non_manifold_edges)} raw-index edge(s) owned by more than two adjacency faces.",
                resource=resource,
                details={"known_retail_exception": known_retail_non_manifold},
            )
        if adjacency_mismatch_slots:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.adjacency_topology_mismatch",
                f"{resource} has {len(adjacency_mismatch_slots)} adjacency slot(s) that disagree with raw vertex-index topology.",
                resource=resource,
            )
        if adjacency_nonreciprocal_slots:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.adjacency_nonreciprocal",
                f"{resource} has {len(adjacency_nonreciprocal_slots)} non-reciprocal adjacency slot(s).",
                resource=resource,
            )

    edge_rows: list[tuple[int, int]] = []
    invalid_transition_rows: list[dict[str, int]] = []
    transition_count = 0
    if edges_ok:
        for index in range(edge_count):
            edge_id, transition = struct.unpack_from("<II", data, edge_offset + index * 8)
            edge_rows.append((edge_id, transition))
            if edge_id >= face_count * 3:
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.wok.edge_id_out_of_bounds",
                    f"{resource} edge row {index} references invalid directed edge {edge_id}.",
                    resource=resource,
                )
            if transition != 0xFFFFFFFF:
                transition_count += 1
                if transition_room_count is not None and transition >= transition_room_count:
                    invalid_transition_rows.append(
                        {"row": index, "directed_edge": edge_id, "destination": transition}
                    )

    if invalid_transition_rows:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.transition_target_out_of_range",
            f"{resource} has {len(invalid_transition_rows)} transition destination(s) outside the "
            f"owning LYT's {transition_room_count} room rows.",
            resource=resource,
            details={
                "lyt_room_count": transition_room_count,
                "invalid_transitions": invalid_transition_rows,
            },
        )

    if edge_rows and expected_boundary_ids:
        serialized_boundary_ids = [edge_id for edge_id, _transition in edge_rows]
        duplicate_boundary_count = len(serialized_boundary_ids) - len(set(serialized_boundary_ids))
        missing_boundary_ids = expected_boundary_ids - set(serialized_boundary_ids)
        extra_boundary_ids = set(serialized_boundary_ids) - expected_boundary_ids
        if duplicate_boundary_count or missing_boundary_ids or extra_boundary_ids:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.boundary_edge_table_mismatch",
                f"{resource} boundary edge table disagrees with its raw-index adjacency boundary.",
                resource=resource,
                details={
                    "duplicate_edge_rows": duplicate_boundary_count,
                    "missing_edge_rows": len(missing_boundary_ids),
                    "extra_edge_rows": len(extra_boundary_ids),
                },
            )

    closed_perimeters = 0
    endpoints: list[int] = []
    if perimeter_count <= 0 and not visual_collision_only:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.wok.missing_perimeter",
            f"{resource} contains walkable geometry but no serialized perimeter-loop records.",
            resource=resource,
            fix_hint="Serialize the BWM perimeter array; do not rely on a reader regenerating it.",
        )
    elif perimeters_ok:
        endpoints = [struct.unpack_from("<I", data, perimeter_offset + index * 4)[0] for index in range(perimeter_count)]
        previous = 0
        for loop_index, endpoint in enumerate(endpoints):
            if endpoint <= previous or endpoint > edge_count:
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.wok.perimeter_endpoint_invalid",
                    f"{resource} perimeter {loop_index} ends at invalid edge index {endpoint}.",
                    resource=resource,
                )
                previous = endpoint
                continue
            loop_rows = edge_rows[previous:endpoint]
            loop_vertices: list[tuple[int, int]] = []
            valid_loop = bool(loop_rows) and bool(faces)
            for edge_id, _transition in loop_rows:
                face_index, local_edge = divmod(edge_id, 3)
                if face_index >= len(faces):
                    valid_loop = False
                    break
                triangle = faces[face_index]
                loop_vertices.append((triangle[local_edge], triangle[(local_edge + 1) % 3]))
            if valid_loop:
                for index, (_start, end) in enumerate(loop_vertices):
                    next_start = loop_vertices[(index + 1) % len(loop_vertices)][0]
                    if end != next_start:
                        valid_loop = False
                        break
            if valid_loop:
                closed_perimeters += 1
            else:
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.wok.perimeter_loop_open",
                    f"{resource} perimeter {loop_index} is not a closed directed edge loop.",
                    resource=resource,
                )
            previous = endpoint
        if endpoints and endpoints[-1] != edge_count:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.wok.perimeter_edges_unconsumed",
                f"{resource} perimeter endpoints consume {endpoints[-1]} of {edge_count} edge rows.",
                resource=resource,
            )

    return (
        WokEngineFingerprint(
            walkmesh_type=walkmesh_type,
            vertex_count=vertex_count,
            face_count=face_count,
            walkable_face_count=walkable_face_count,
            aabb_count=aabb_count,
            aabb_leaf_count=len(aabb_leaf_faces),
            aabb_covered_face_count=aabb_covered_face_count,
            aabb_missing_face_count=aabb_missing_face_count,
            aabb_reachable_count=aabb_reachable_count,
            adjacency_count=adjacency_count,
            adjacency_mismatch_count=len(adjacency_mismatch_slots),
            adjacency_nonreciprocal_count=len(adjacency_nonreciprocal_slots),
            non_manifold_edge_count=len(non_manifold_edges),
            edge_count=edge_count,
            perimeter_count=perimeter_count,
            closed_perimeter_count=closed_perimeters,
            transition_count=transition_count,
            nonzero_hook_value_count=sum(1 for value in hooks if value != 0.0),
            material_histogram=histogram,
        ),
        report,
    )


def _parse_lyt(data: bytes, report: ValidationReport, module_resref: str) -> tuple[list[str], dict[str, tuple[float, float, float]]]:
    resource = f"{module_resref}.lyt"
    rooms: list[str] = []
    positions: dict[str, tuple[float, float, float]] = {}
    remaining = 0
    for raw in data.decode("latin-1", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens[0].lower() == "roomcount":
            try:
                remaining = int(tokens[1])
            except (IndexError, ValueError):
                remaining = -1
            continue
        if remaining > 0:
            try:
                name = _normalise_resref(tokens[0])
                position = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            except (IndexError, ValueError):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.module.lyt_room_invalid",
                    f"{resource} contains an invalid room row: {line!r}.",
                    resource=resource,
                )
                remaining -= 1
                continue
            if not name or name in positions or not all(math.isfinite(value) for value in position):
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.module.lyt_room_invalid",
                    f"{resource} contains an invalid or duplicate room '{name}'.",
                    resource=resource,
                )
            else:
                rooms.append(name)
                positions[name] = position
            remaining -= 1
    if not rooms:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.module.lyt_empty",
            f"{resource} contains no room rows.",
            resource=resource,
        )
    return rooms, positions


def _parse_vis(data: bytes, report: ValidationReport, module_resref: str) -> dict[str, set[str]]:
    resource = f"{module_resref}.vis"
    visibility: dict[str, set[str]] = {}
    declared: dict[str, int] = {}
    current = ""
    for raw in data.decode("latin-1", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        tokens = raw.split()
        if raw[0].isspace():
            if current and tokens:
                visibility[current].add(_normalise_resref(tokens[0]))
            continue
        current = _normalise_resref(tokens[0])
        if not current:
            continue
        visibility.setdefault(current, set())
        try:
            declared[current] = int(tokens[1])
        except (IndexError, ValueError):
            declared[current] = -1
    for room, targets in visibility.items():
        if declared.get(room) != len(targets):
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.module.vis_count_mismatch",
                f"{resource} room {room} declares {declared.get(room)} links but contains {len(targets)}.",
                resource=resource,
            )
    return visibility


def _parse_gff_module_links(
    resources: Mapping[tuple[str, str], bytes],
    report: ValidationReport,
    module_resref: str,
) -> tuple[list[str], list[str], str]:
    are_rooms: list[str] = []
    ifo_areas: list[str] = []
    entry_area = ""
    try:
        from pykotor.resource.formats.gff import read_gff

        are = read_gff(resources[(module_resref, "are")]).root
        room_list = are.get("Rooms")
        are_rooms = [
            _normalise_resref(room_list.at(index).get("RoomName"))
            for index in range(len(room_list))
        ]
        git = read_gff(resources[(module_resref, "git")])
        if git is None:
            raise ValueError("GIT parser returned no data")
        ifo = read_gff(resources[("module", "ifo")]).root
        area_list = ifo.get("Mod_Area_list")
        ifo_areas = [
            _normalise_resref(area_list.at(index).get("Area_Name"))
            for index in range(len(area_list))
        ]
        entry_area = _normalise_resref(ifo.get("Mod_Entry_Area"))
    except Exception as exc:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.module.gff_parse_failed",
            f"{module_resref} ARE/GIT/IFO engine linkage could not be parsed: {exc}",
            resource=module_resref,
        )
    return are_rooms, ifo_areas, entry_area


def _parse_placeable_template_links(
    resources: Mapping[tuple[str, str], bytes],
    report: ValidationReport,
    module_resref: str,
) -> tuple[dict[str, Any], ...]:
    """Follow serialized GIT placeable rows into bundled UTP resources.

    A stock/base-game UTP is allowed to remain external.  When Map Studio
    bundles a UTP, however, the bytes in the final resource map must parse as
    UTP and their TemplateResRef must match the GIT reference exactly.  This is
    intentionally a final-byte check; it does not trust the authoring sidecar.
    """

    git_key = (module_resref, "git")
    if git_key not in resources:
        return ()
    try:
        from pykotor.resource.formats.gff import GFFContent, read_gff

        git = read_gff(resources[git_key])
        placeable_list = git.root.get("Placeable List")
    except Exception as exc:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.placeable.git_parse_failed",
            f"{module_resref}.git placeable list could not be parsed: {exc}",
            resource=f"{module_resref}.git",
        )
        return ()

    rows: list[dict[str, Any]] = []
    referenced: set[str] = set()
    try:
        count = len(placeable_list) if placeable_list is not None else 0
    except Exception:
        count = 0
    for index in range(count):
        item = placeable_list.at(index)
        template_resref = _normalise_resref(item.get("TemplateResRef"))
        tag = str(item.get("Tag") or "")
        bundled = bool(template_resref and (template_resref, "utp") in resources)
        row: dict[str, Any] = {
            "index": index,
            "template_resref": template_resref,
            "tag": tag,
            "bundled": bundled,
            "status": "bundled" if bundled else "external_or_base_game",
            "appearance_id": None,
            "script_references": [],
            "conversation": "",
            "inventory_references": [],
        }
        rows.append(row)
        if not template_resref:
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.placeable.template_missing",
                f"{module_resref}.git placeable row {index} has no TemplateResRef.",
                resource=f"{module_resref}.git",
            )
            continue
        referenced.add(template_resref)
        if not bundled:
            _add_issue(
                report,
                ValidationSeverity.WARNING,
                "map.engine.placeable.template_external",
                f"Placeable {tag or template_resref} uses external template {template_resref}.utp; verify it resolves from the target game or Override.",
                resource=f"{module_resref}.git",
                details={"template_resref": template_resref, "tag": tag, "index": index},
                fix_hint="Bundle project-authored UTP files with the module; stock UTPs may remain external.",
            )
            continue

        resource_name = f"{template_resref}.utp"
        try:
            utp_gff = read_gff(resources[(template_resref, "utp")])
            if utp_gff.content != GFFContent.UTP:
                raise ValueError(f"GFF content is {utp_gff.content}, expected UTP")
            root = utp_gff.root
            stored_resref = _normalise_resref(root.get("TemplateResRef"))
            appearance_id = int(root.get("Appearance") or 0)
            row["appearance_id"] = appearance_id
            row["utp_template_resref"] = stored_resref
            if stored_resref != template_resref:
                _add_issue(
                    report,
                    ValidationSeverity.BLOCKING,
                    "map.engine.placeable.template_resref_mismatch",
                    f"{resource_name} stores TemplateResRef '{stored_resref}' but GIT references '{template_resref}'.",
                    resource=resource_name,
                    details={"git_template_resref": template_resref, "utp_template_resref": stored_resref},
                )

            conversation = _normalise_resref(root.get("Conversation"))
            row["conversation"] = conversation
            if conversation:
                packaged = (conversation, "dlg") in resources
                row["conversation_packaged"] = packaged
                if not packaged:
                    _add_issue(
                        report,
                        ValidationSeverity.WARNING,
                        "map.engine.placeable.conversation_external",
                        f"{resource_name} references external conversation {conversation}.dlg.",
                        resource=resource_name,
                    )

            script_fields = (
                "OnClosed", "OnDamaged", "OnDeath", "OnEndDialogue", "OnFailToOpen",
                "OnHeartbeat", "OnInvDisturbed", "OnLock", "OnMeleeAttacked", "OnOpen",
                "OnSpellCastAt", "OnUnlock", "OnUsed", "OnUserDefined", "OnDisarm",
                "OnTrapTriggered",
            )
            script_rows: list[dict[str, Any]] = []
            for field_name in script_fields:
                script_resref = _normalise_resref(root.get(field_name))
                if not script_resref:
                    continue
                packaged = (script_resref, "ncs") in resources
                script_rows.append({"field": field_name, "resref": script_resref, "packaged": packaged})
                if not packaged:
                    _add_issue(
                        report,
                        ValidationSeverity.WARNING,
                        "map.engine.placeable.script_external",
                        f"{resource_name} field {field_name} references external script {script_resref}.ncs.",
                        resource=resource_name,
                    )
            row["script_references"] = script_rows

            inventory_rows: list[dict[str, Any]] = []
            inventory = root.get("ItemList")
            try:
                inventory_count = len(inventory) if inventory is not None else 0
            except Exception:
                inventory_count = 0
            for inventory_index in range(inventory_count):
                inventory_item = inventory.at(inventory_index)
                item_resref = _normalise_resref(inventory_item.get("InventoryRes"))
                if not item_resref:
                    continue
                packaged = (item_resref, "uti") in resources
                inventory_rows.append({"index": inventory_index, "resref": item_resref, "packaged": packaged})
                if not packaged:
                    _add_issue(
                        report,
                        ValidationSeverity.WARNING,
                        "map.engine.placeable.inventory_external",
                        f"{resource_name} inventory row {inventory_index} references external item {item_resref}.uti.",
                        resource=resource_name,
                    )
            row["inventory_references"] = inventory_rows
        except Exception as exc:
            row["status"] = "invalid_bundled_utp"
            row["parse_error"] = str(exc)
            _add_issue(
                report,
                ValidationSeverity.BLOCKING,
                "map.engine.placeable.utp_parse_failed",
                f"{resource_name} could not be parsed as a KOTOR placeable template: {exc}",
                resource=resource_name,
            )

    for resref, restype in sorted(resources):
        if restype == "utp" and resref not in referenced:
            _add_issue(
                report,
                ValidationSeverity.WARNING,
                "map.engine.placeable.utp_unreferenced",
                f"Bundled placeable template {resref}.utp is not referenced by any GIT placeable row.",
                resource=f"{resref}.utp",
            )
    return tuple(rows)


def _parse_pth(data: bytes, report: ValidationReport, module_resref: str) -> tuple[int, int]:
    resource = f"{module_resref}.pth"
    try:
        from pykotor.resource.generics.pth import read_pth

        pth = read_pth(data)
        points = len(pth)
        connections = sum(len(pth.outgoing(index)) for index in range(points))
    except Exception as exc:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.module.pth_parse_failed",
            f"{resource} could not be parsed: {exc}",
            resource=resource,
        )
        return 0, 0
    if points < 1:
        _add_issue(
            report,
            ValidationSeverity.BLOCKING,
            "map.engine.module.pth_empty",
            f"{resource} contains no path points for a playable authored module.",
            resource=resource,
        )
    return points, connections


#: Dummy nodes every retail hostile creature model carries (C_Kinrath, C_KhoundA,
#: C_Kataarn): cutscenedummy -> rootdummy, camerahook, impact and talkdummy.
#: Custom creatures without them have shown the wrong (teal) health-bar colour.
CREATURE_REQUIRED_DUMMY_NODES = ("cutscenedummy", "rootdummy", "camerahook", "impact", "talkdummy")
_MDL_CLASSIFICATION_CHARACTER = 0x04


def missing_creature_dummy_nodes(mdl: bytes) -> tuple[str, ...] | None:
    """Required creature dummies a standalone character MDL lacks.

    Returns ``None`` when the model is not a character, inherits from a
    supermodel (which may supply the nodes) or cannot be parsed.
    """
    if len(mdl) < _MDL_MIN_SIZE or mdl[_MDL_BASE + 80] != _MDL_CLASSIFICATION_CHARACTER:
        return None
    supermodel = mdl[_MDL_BASE + 80 + 56 : _MDL_BASE + 80 + 88].split(b"\0", 1)[0].decode("ascii", "replace")
    if supermodel.strip().lower() not in {"", "null"}:
        return None
    table_rel = struct.unpack_from("<I", mdl, _MDL_BASE + 80 + 104)[0]
    count = struct.unpack_from("<I", mdl, _MDL_BASE + 80 + 108)[0]
    table_abs = _MDL_BASE + table_rel
    if count > _MAX_REASONABLE_COUNT or not _bounded(mdl, table_abs, count * 4):
        return None
    names = {
        _read_c_string(mdl, _MDL_BASE + struct.unpack_from("<I", mdl, table_abs + index * 4)[0]).lower()
        for index in range(count)
    }
    return tuple(node for node in CREATURE_REQUIRED_DUMMY_NODES if node not in names)


def validate_kotor_module_engine_contract(
    request: KotorModuleEngineContractRequest,
) -> KotorModuleEngineContractReport:
    """Validate every serialized room and the module's spatial linkage."""

    game = _normalise_game(request.game)
    module_resref = _normalise_resref(request.module_resref)
    resources = _normalise_resources(request.resources)
    validation = ValidationReport(
        source="map_studio.engine_contract",
        metadata={
            "schema": "ghostrigger.map_studio_engine_contract.v1",
            "game": game,
            "module_resref": module_resref,
            "vanilla_reference_ids": [
                row["id"] for row in VANILLA_ROOM_BASELINES if row["game"] == game
            ],
        },
    )

    required_module_resources = (
        (module_resref, "are"),
        (module_resref, "git"),
        ("module", "ifo"),
        (module_resref, "pth"),
        (module_resref, "lyt"),
        (module_resref, "vis"),
    )
    for key in required_module_resources:
        if key not in resources:
            _add_issue(
                validation,
                ValidationSeverity.BLOCKING,
                "map.engine.module.resource_missing",
                f"Missing required engine resource {key[0]}.{key[1]}.",
                resource=f"{key[0]}.{key[1]}",
            )

    lyt_rooms: list[str] = []
    room_positions: dict[str, tuple[float, float, float]] = {}
    if (module_resref, "lyt") in resources:
        lyt_rooms, room_positions = _parse_lyt(resources[(module_resref, "lyt")], validation, module_resref)

    expected_rooms = tuple(
        dict.fromkeys(_normalise_resref(room) for room in request.expected_room_resrefs if _normalise_resref(room))
    )
    if expected_rooms and set(expected_rooms) != set(lyt_rooms):
        _add_issue(
            validation,
            ValidationSeverity.BLOCKING,
            "map.engine.module.expected_lyt_mismatch",
            f"Expected rooms {sorted(expected_rooms)} do not match LYT rooms {sorted(lyt_rooms)}.",
            resource=f"{module_resref}.lyt",
        )
    canonical_rooms = expected_rooms or tuple(lyt_rooms)

    vis: dict[str, set[str]] = {}
    if (module_resref, "vis") in resources:
        vis = _parse_vis(resources[(module_resref, "vis")], validation, module_resref)
        if set(vis) != set(canonical_rooms):
            _add_issue(
                validation,
                ValidationSeverity.BLOCKING,
                "map.engine.module.vis_room_mismatch",
                f"VIS room headers {sorted(vis)} do not match LYT rooms {sorted(canonical_rooms)}.",
                resource=f"{module_resref}.vis",
            )
        for room, targets in vis.items():
            for target in targets:
                if target not in canonical_rooms:
                    _add_issue(
                        validation,
                        ValidationSeverity.BLOCKING,
                        "map.engine.module.vis_target_missing",
                        f"VIS room {room} references missing room {target}.",
                        resource=f"{module_resref}.vis",
                    )
                elif target == room:
                    _add_issue(
                        validation,
                        ValidationSeverity.BLOCKING,
                        "map.engine.module.vis_self_reference",
                        f"VIS room {room} explicitly references itself; vanilla VIS uses implicit self visibility.",
                        resource=f"{module_resref}.vis",
                    )
                elif room not in vis.get(target, set()):
                    _add_issue(
                        validation,
                        ValidationSeverity.BLOCKING,
                        "map.engine.module.vis_asymmetric",
                        f"VIS is asymmetric: {room} sees {target}, but {target} does not see {room}.",
                        resource=f"{module_resref}.vis",
                    )
        if len(canonical_rooms) > 1 and not any(vis.values()):
            _add_issue(
                validation,
                ValidationSeverity.WARNING,
                "map.engine.module.vis_empty_multiroom",
                "Multi-room module has no cross-room VIS links.",
                resource=f"{module_resref}.vis",
            )

    are_rooms: list[str] = []
    if all(key in resources for key in ((module_resref, "are"), (module_resref, "git"), ("module", "ifo"))):
        are_rooms, ifo_areas, entry_area = _parse_gff_module_links(resources, validation, module_resref)
        are_set = set(are_rooms)
        canonical_set = set(canonical_rooms)
        if not are_set.issubset(canonical_set):
            _add_issue(
                validation,
                ValidationSeverity.BLOCKING,
                "map.engine.module.are_lyt_mismatch",
                f"ARE Rooms contains entries absent from LYT: {sorted(are_set - canonical_set)}.",
                resource=f"{module_resref}.are",
            )
        lyt_only_rooms = sorted(canonical_set - are_set)
        if lyt_only_rooms:
            _add_issue(
                validation,
                ValidationSeverity.WARNING,
                "map.engine.module.lyt_only_rooms",
                f"LYT contains visual/reference room(s) intentionally absent from ARE Rooms: {lyt_only_rooms}.",
                resource=f"{module_resref}.lyt",
            )
        if ifo_areas != [module_resref] or entry_area != module_resref:
            _add_issue(
                validation,
                ValidationSeverity.BLOCKING,
                "map.engine.module.ifo_area_mismatch",
                f"module.ifo area list/entry {ifo_areas}/{entry_area or '(blank)'} does not identify {module_resref}.",
                resource="module.ifo",
            )

    pth_points = 0
    pth_connections = 0
    if (module_resref, "pth") in resources:
        pth_points, pth_connections = _parse_pth(resources[(module_resref, "pth")], validation, module_resref)

    visual_only_rooms = {
        _normalise_resref(room)
        for room in request.visual_only_room_resrefs
        if _normalise_resref(room)
    }
    room_fingerprints: list[RoomEngineFingerprint] = []
    for room in canonical_rooms:
        missing = [restype for restype in ("mdl", "mdx", "wok") if (room, restype) not in resources]
        if missing:
            _add_issue(
                validation,
                ValidationSeverity.BLOCKING,
                "map.engine.module.room_resource_missing",
                f"Room {room} is missing engine resource(s): {', '.join(missing)}.",
                resource=room,
                details={"missing_types": missing},
            )
            continue
        mdl_fp, mdl_report = inspect_raw_mdl_structure(
            room,
            resources[(room, "mdl")],
            resources[(room, "mdx")],
            game=game,
            allow_missing_aabb=room in visual_only_rooms,
        )
        wok_fp, wok_report = inspect_raw_wok_structure(
            room,
            resources[(room, "wok")],
            allow_empty_visual=room in visual_only_rooms,
            transition_room_count=len(canonical_rooms),
        )
        validation.issues.extend(mdl_report.issues)
        validation.issues.extend(wok_report.issues)
        room_fingerprints.append(RoomEngineFingerprint(room_resref=room, mdl=mdl_fp, wok=wok_fp))

    room_resource_names = {
        resref
        for (resref, restype) in resources
        if restype in {"mdl", "mdx", "wok"} and resref not in {module_resref, "module"}
    }
    orphaned = sorted(room_resource_names - set(canonical_rooms))
    if orphaned:
        _add_issue(
            validation,
            ValidationSeverity.WARNING,
            "map.engine.module.orphan_room_resources",
            f"Room resources are not referenced by LYT: {orphaned}.",
            resource=f"{module_resref}.lyt",
        )

    for (resref, restype), payload in sorted(resources.items()):
        if restype != "mdl" or resref in canonical_rooms:
            continue
        missing = missing_creature_dummy_nodes(payload)
        if missing:
            _add_issue(
                validation,
                ValidationSeverity.WARNING,
                "map.engine.creature.missing_dummy_nodes",
                f"{resref}.mdl is a standalone character model without {', '.join(missing)}; "
                "retail creatures carry these dummies (missing ones have shown the wrong health-bar colour).",
                resource=f"{resref}.mdl",
            )

    duplicate_positions: dict[tuple[float, float, float], list[str]] = {}
    for room, position in room_positions.items():
        duplicate_positions.setdefault(position, []).append(room)
    overlaps = [rooms for rooms in duplicate_positions.values() if len(rooms) > 1]
    if overlaps:
        _add_issue(
            validation,
            ValidationSeverity.WARNING,
            "map.engine.module.duplicate_lyt_positions",
            f"Multiple LYT rooms share an exact position: {overlaps}.",
            resource=f"{module_resref}.lyt",
        )

    placeable_templates = _parse_placeable_template_links(resources, validation, module_resref)
    bundled_placeable_count = sum(1 for row in placeable_templates if bool(row.get("bundled")))

    result = KotorModuleEngineContractReport(
        validation=validation,
        rooms=tuple(room_fingerprints),
        lyt_rooms=tuple(lyt_rooms),
        are_rooms=tuple(are_rooms),
        vis_link_count=sum(len(targets) for targets in vis.values()),
        pth_point_count=pth_points,
        pth_connection_count=pth_connections,
        placeable_template_count=len(placeable_templates),
        bundled_placeable_count=bundled_placeable_count,
        placeable_templates=placeable_templates,
    )
    validation.metadata.update(
        {
            "export_ready": result.export_ready,
            "room_count": len(result.rooms),
            "lyt_room_count": len(result.lyt_rooms),
            "are_room_count": len(result.are_rooms),
            "vis_link_count": result.vis_link_count,
            "pth_point_count": result.pth_point_count,
            "pth_connection_count": result.pth_connection_count,
            "placeable_template_count": result.placeable_template_count,
            "bundled_placeable_count": result.bundled_placeable_count,
        }
    )
    return result


__all__ = [
    "KotorModuleEngineContractReport",
    "KotorModuleEngineContractRequest",
    "MdlEngineFingerprint",
    "RoomEngineFingerprint",
    "VANILLA_ROOM_BASELINES",
    "WokEngineFingerprint",
    "inspect_raw_mdl_structure",
    "inspect_raw_wok_structure",
    "validate_kotor_module_engine_contract",
]
