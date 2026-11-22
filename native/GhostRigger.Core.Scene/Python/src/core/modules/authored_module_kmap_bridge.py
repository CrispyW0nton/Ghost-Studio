"""Bridge KMAP project sections to authored Map Studio module readiness.

KMAP is the scene/project container.  The from-scratch module authoring contract
lives in ``AuthoredModuleProject``.  This bridge keeps the conversion headless
so Qt windows can display readiness without owning parsing or validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from typing import Any
from uuid import UUID, uuid5

from src.core.level.kmap_validator import KMapValidator

from .authored_module_objects import (
    AuthoredCameraInstance,
    AuthoredCreatureInstance,
    AuthoredDoorInstance,
    AuthoredEncounterInstance,
    AuthoredGameplayPlacement,
    AuthoredPlaceableInstance,
    AuthoredSoundInstance,
    AuthoredStoreInstance,
    AuthoredTriggerInstance,
    AuthoredWaypointInstance,
    ModuleEntryPoint,
)
from .authored_module_lighting import (
    AuthoredRoomLight,
    authored_room_light_payload,
    normalise_authored_room_light,
)
from .authored_module_project import (
    AuthoredModuleMetadata,
    AuthoredModuleProject,
    AuthoredRoomSpec,
    create_single_room_project,
    normalise_resref,
)
from .authored_module_readiness import AuthoredModuleReadiness, build_authored_module_readiness
from .authored_room_composition import (
    AuthoredRoomComposition,
    CombinedRoomPrimitive,
    CombinedRoomPrimitiveSource,
    PlacedRoomPrimitive,
    PrimitiveTransform,
)
from .authored_room_materials import DEFAULT_AUTHORED_ROOM_TEXTURE, normalize_authored_room_texture
from .authored_imported_mesh import (
    IMPORTED_MESH_PRIMITIVE_KIND,
    ImportedMeshRoomPrimitive,
    imported_mesh_primitive_from_payload,
    imported_mesh_primitive_payload,
)
from .authored_room_floorplan import FloorPlanRoomPrimitive, FloorPlanWallOpening
from .authored_room_geometry import RectangularRoomPrimitive
from .authored_terrain_builder import TerrainHeightfieldPrimitive
from .authored_room_primitives import (
    ArchPrimitive,
    ConePrimitive,
    CubePrimitive,
    CylinderPrimitive,
    DoorFramePrimitive,
    FloorPrimitive,
    PrimitiveMaterial,
    RampPrimitive,
    SpherePrimitive,
    StairsPrimitive,
    TorusPrimitive,
    WallPrimitive,
    primitive_construction_node_id,
)


@dataclass(frozen=True)
class AuthoredModuleKMapBridgeResult:
    """Authored module data found in a KMAP project, if any."""

    project: AuthoredModuleProject | None = None
    readiness: AuthoredModuleReadiness | None = None
    runtime_resources: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    blocking_messages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


TEXTURE_PAINT_UNAPPLIED_BLOCKER = (
    "Texture Paint has unapplied live changes. Click Apply Texture Changes before "
    "building, exporting, staging, or installing this module."
)


_PLACEMENT_INSTANCE_ID_NAMESPACE = UUID("c4d04ea5-f077-5d16-a0bf-867340926a71")
_PLACEMENT_INSTANCE_ID_VERSION = 1
_PLACEMENT_BEARING_UNIT_VERSION = 1


def _migrated_placement_instance_id(
    *,
    module_root: str,
    kind: str,
    index: int,
    source: Any,
    seen: set[str],
) -> str:
    """Preserve a KMAP instance token or deterministically migrate an old row."""

    data = _dict(source)
    candidate = str(data.get("instance_id") or "").strip()
    if ":" in candidate or candidate in seen:
        candidate = ""
    if not candidate:
        identity_source = dict(data)
        identity_source.pop("instance_id", None)
        fingerprint = json.dumps(identity_source, sort_keys=True, separators=(",", ":"), default=str)
        salt = 0
        while True:
            suffix = f"|{salt}" if salt else ""
            candidate = "i_" + uuid5(
                _PLACEMENT_INSTANCE_ID_NAMESPACE,
                f"{normalise_resref(module_root)}|{kind}|{int(index)}|{fingerprint}{suffix}",
            ).hex
            if candidate not in seen:
                break
            salt += 1
    seen.add(candidate)
    return candidate


def texture_paint_pending_resrefs(payload: Any) -> tuple[str, ...]:
    """Return stable pending paint targets from current and legacy KMAP state."""

    data = _dict(payload)
    raw_values = data.get("texture_paint_pending_resrefs") or ()
    values = list(raw_values) if isinstance(raw_values, (list, tuple, set)) else [raw_values]
    legacy = str(data.get("texture_paint_resref") or "").strip()
    if legacy:
        values.append(legacy)
    return tuple(
        dict.fromkeys(
            str(value or "").strip().lower()
            for value in values
            if str(value or "").strip()
        )
    )


def texture_paint_has_unapplied_changes(payload: Any) -> bool:
    """Treat either persisted draft flag as an export-blocking paint state."""

    data = _dict(payload)
    return bool(data.get("texture_paint_unapplied", False) or data.get("texture_paint_dirty", False))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bearing_radians(value: Any, *, unit: str = "") -> float:
    """Migrate legacy degree-valued KMAP bearings into normalized radians."""

    bearing = _float(value, 0.0)
    declared = str(unit or "").strip().lower()
    if declared in {"degree", "degrees", "deg"}:
        bearing = math.radians(bearing)
    elif declared not in {"radian", "radians", "rad"} and abs(bearing) > math.tau + 1.0e-6:
        # KMAP v0 did not name its unit while the degree-labelled Builder
        # persisted raw 45/90/180 values.
        bearing = math.radians(bearing)
    return math.atan2(math.sin(bearing), math.cos(bearing))


def _vec2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_float(value[0], 0.0), _float(value[1], 0.0))
    return None


def _vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"), value.get("z"))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (_float(value[0], default[0]), _float(value[1], default[1]), _float(value[2], default[2]))
    return default


def _vec4(value: Any, default: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"), value.get("z"), value.get("w"))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (_float(value[0], default[0]), _float(value[1], default[1]), _float(value[2], default[2]), _float(value[3], default[3]))
    return default


def _material(data: Any) -> PrimitiveMaterial:
    source = _dict(data)
    return PrimitiveMaterial(
        texture=str(source.get("texture") or "default"),
        diffuse=_vec3(source.get("diffuse"), (0.8, 0.8, 0.8)),
        ambient=_vec3(source.get("ambient"), (0.35, 0.35, 0.35)),
        metadata=_dict(source.get("metadata")),
    )


def _transform(data: Any) -> PrimitiveTransform:
    source = _dict(data)
    return PrimitiveTransform(
        translation=_vec3(source.get("translation")),
        rotation_degrees_z=_float(source.get("rotation_degrees_z"), 0.0),
        scale=_vec3(source.get("scale"), (1.0, 1.0, 1.0)),
        pivot=_vec3(source.get("pivot")),
    )


def _evaluation_transforms(data: Any) -> tuple[PrimitiveTransform, ...]:
    """Decode optional downstream Freeze Transformations stages.

    Older KMAP payloads have no such field and therefore decode to the exact
    previous single-transform behavior.
    """

    if not isinstance(data, (list, tuple)):
        return ()
    return tuple(_transform(value) for value in data if isinstance(value, dict))


def _opening(data: Any) -> FloorPlanWallOpening:
    source = _dict(data)
    return FloorPlanWallOpening(
        name=str(source.get("name") or ""),
        edge_index=int(_float(source.get("edge_index"), 0.0)),
        center_fraction=_float(source.get("center_fraction"), 0.5),
        width=_float(source.get("width"), 1.5),
        height=_float(source.get("height"), 2.1),
        bottom=_float(source.get("bottom"), 0.0),
        metadata=_dict(source.get("metadata")),
    )


def _floor_primitive(data: Any, room_resref: str) -> FloorPrimitive | PlacedRoomPrimitive:
    source = _dict(data)
    floor = FloorPrimitive(
        name=str(source.get("name") or f"{room_resref}_floor"),
        width=_float(source.get("width"), 10.0),
        depth=_float(source.get("depth"), 10.0),
        z=_float(source.get("z"), 0.0),
        surface_id=source.get("surface_id", source.get("floor_surface_id", 4)),
        material=_material(source.get("material")),
        subdivisions_width=int(_float(source.get("subdivisions_width"), 1.0)),
        subdivisions_depth=int(_float(source.get("subdivisions_depth"), 1.0)),
        axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
        height_baseline=_float(source.get("height_baseline"), 0.0),
        create_uvs=int(_float(source.get("create_uvs"), 1.0)),
        recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
        construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="plane", name=str(source.get("name") or f"{room_resref}_floor"))),
        construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
    )
    evaluation_transforms = _evaluation_transforms(source.get("evaluation_transforms"))
    if source.get("transform") is None and not evaluation_transforms:
        return floor
    return PlacedRoomPrimitive(
        primitive=floor,
        transform=_transform(source.get("transform")),
        name=str(source.get("instance_name") or source.get("name") or floor.name),
        evaluation_transforms=evaluation_transforms,
    )


def _base_room_primitive(
    data: Any,
    room_resref: str,
    *,
    _depth: int = 0,
) -> FloorPrimitive | WallPrimitive | CubePrimitive | SpherePrimitive | ConePrimitive | TorusPrimitive | RampPrimitive | StairsPrimitive | CylinderPrimitive | DoorFramePrimitive | ArchPrimitive | CombinedRoomPrimitive:
    if _depth > 32:
        raise ValueError("Combined room primitive recipe nesting exceeds the supported depth of 32.")
    source = _dict(data)
    primitive_type = str(source.get("type") or source.get("primitive") or "").strip().lower()
    name = str(source.get("name") or f"{room_resref}_{primitive_type or 'primitive'}")
    if primitive_type in {"combined_mesh", "combined_polygon_mesh", "combined_room_primitive"}:
        sources: list[CombinedRoomPrimitiveSource] = []
        for source_index, raw_source in enumerate(source.get("sources", ()) or ()):
            source_payload = _dict(raw_source)
            primitive_payload = _dict(source_payload.get("primitive"))
            if not primitive_payload:
                raise ValueError(f"Combined room primitive {name} source {source_index} has no primitive recipe.")
            face_indices = tuple(
                sorted(dict.fromkeys(int(index) for index in tuple(source_payload.get("face_indices") or ())))
            )
            sources.append(
                CombinedRoomPrimitiveSource(
                    primitive=_room_primitive_recipe(primitive_payload, room_resref, _depth=_depth + 1),
                    face_indices=face_indices,
                    source_name=str(source_payload.get("source_name") or primitive_payload.get("instance_name") or primitive_payload.get("name") or ""),
                    walkmesh_policy=str(source_payload.get("walkmesh_policy") or "inherit"),
                )
            )
        return CombinedRoomPrimitive(
            name=name,
            sources=tuple(sources),
            metadata=_dict(source.get("metadata")),
        )
    material = _material(source.get("material"))
    if primitive_type in {"floor", "plane", "platform"}:
        return FloorPrimitive(
            name=name,
            width=_float(source.get("width"), 3.0),
            depth=_float(source.get("depth"), 3.0),
            z=_float(source.get("z"), 0.0),
            surface_id=source.get("surface_id", source.get("floor_surface_id", 4)),
            material=material,
            subdivisions_width=int(_float(source.get("subdivisions_width"), 1.0)),
            subdivisions_depth=int(_float(source.get("subdivisions_depth"), 1.0)),
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=int(_float(source.get("create_uvs"), 1.0)),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="plane", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type == "wall":
        return WallPrimitive(
            name=name,
            width=_float(source.get("width"), 4.0),
            height=_float(source.get("height"), 3.0),
            thickness=_float(source.get("thickness"), 0.15),
            axis=str(source.get("axis") or "x"),
            center=_vec3(source.get("center"), (0.0, 0.0, 1.5)),
            material=material,
        )
    if primitive_type == "cube":
        return CubePrimitive(
            name=name,
            size=_vec3(source.get("size"), (1.0, 1.0, 1.0)),
            center=_vec3(source.get("center"), (0.0, 0.0, 0.5)),
            material=material,
            subdivisions_x=int(_float(source.get("subdivisions_x"), 1.0)),
            subdivisions_y=int(_float(source.get("subdivisions_y"), 1.0)),
            subdivisions_z=int(_float(source.get("subdivisions_z"), 1.0)),
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=int(_float(source.get("create_uvs"), 3.0)),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="cube", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type == "ramp":
        return RampPrimitive(
            name=name,
            width=_float(source.get("width"), 2.0),
            length=_float(source.get("length"), 4.0),
            height=_float(source.get("height"), 1.0),
            center=_vec3(source.get("center")),
            surface_id=source.get("surface_id", 4),
            material=material,
        )
    if primitive_type == "stairs":
        return StairsPrimitive(
            name=name,
            width=_float(source.get("width"), 2.0),
            depth=_float(source.get("depth"), 4.0),
            height=_float(source.get("height"), 1.0),
            steps=int(_float(source.get("steps"), 4.0)),
            surface_id=source.get("surface_id", 4),
            material=material,
        )
    if primitive_type == "cylinder":
        return CylinderPrimitive(
            name=name,
            radius=_float(source.get("radius"), 0.5),
            height=_float(source.get("height"), 1.0),
            segments=int(_float(source.get("subdivisions_axis", source.get("segments")), 16.0)),
            center=_vec3(source.get("center"), (0.0, 0.0, 0.5)),
            material=material,
            subdivisions_height=int(_float(source.get("subdivisions_height"), 1.0)),
            subdivisions_caps=int(_float(source.get("subdivisions_caps"), 0.0)),
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=int(_float(source.get("create_uvs"), 2.0)),
            round_cap=_bool(source.get("round_cap"), False),
            round_cap_height_compensation=_bool(source.get("round_cap_height_compensation"), False),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="cylinder", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type in {"sphere", "uv_sphere", "poly_sphere"}:
        return SpherePrimitive(
            name=name,
            radius=_float(source.get("radius"), 0.5),
            subdivisions_axis=int(_float(source.get("subdivisions_axis"), 20.0)),
            subdivisions_height=int(_float(source.get("subdivisions_height"), 20.0)),
            center=_vec3(source.get("center"), (0.0, 0.0, 0.5)),
            material=material,
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=int(_float(source.get("create_uvs"), 2.0)),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="sphere", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type in {"cone", "poly_cone"}:
        return ConePrimitive(
            name=name,
            radius=_float(source.get("radius"), 0.5),
            height=_float(source.get("height"), 1.0),
            subdivisions_axis=int(_float(source.get("subdivisions_axis"), 20.0)),
            subdivisions_height=int(_float(source.get("subdivisions_height"), 1.0)),
            subdivisions_caps=int(
                _float(
                    source.get("subdivisions_caps"),
                    0.0 if ("recipe_version" in source or "construction_schema_version" in source) else 1.0,
                )
            ),
            center=_vec3(source.get("center"), (0.0, 0.0, 0.5)),
            material=material,
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=int(_float(source.get("create_uvs"), 2.0)),
            round_cap=_bool(source.get("round_cap"), False),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="cone", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type in {"torus", "poly_torus"}:
        return TorusPrimitive(
            name=name,
            radius=_float(source.get("radius"), 1.0),
            section_radius=_float(source.get("section_radius"), 0.25),
            subdivisions_axis=int(_float(source.get("subdivisions_axis"), 20.0)),
            subdivisions_height=int(_float(source.get("subdivisions_height"), 20.0)),
            center=_vec3(source.get("center"), (0.0, 0.0, 0.5)),
            material=material,
            axis=_vec3(source.get("axis"), (0.0, 0.0, 1.0)),
            height_baseline=_float(source.get("height_baseline"), 0.0),
            create_uvs=_bool(source.get("create_uvs"), True),
            twist=_float(source.get("twist"), 0.0),
            recipe_version=max(1, int(_float(source.get("recipe_version"), 1.0))),
            construction_node_id=str(source.get("construction_node_id") or primitive_construction_node_id(room_resref=room_resref, primitive_type="torus", name=name)),
            construction_schema_version=max(1, int(_float(source.get("construction_schema_version"), 1.0))),
        )
    if primitive_type in {"door_frame", "doorframe", "doorway_frame"}:
        return DoorFramePrimitive(
            name=name,
            width=_float(source.get("width"), 2.2),
            height=_float(source.get("height"), 3.0),
            jamb_width=_float(source.get("jamb_width"), 0.22),
            lintel_height=_float(source.get("lintel_height"), 0.28),
            depth=_float(source.get("depth"), 0.25),
            center=_vec3(source.get("center"), (0.0, 0.0, 1.5)),
            material=material,
        )
    if primitive_type == "arch":
        return ArchPrimitive(
            name=name,
            width=_float(source.get("width"), 2.0),
            height=_float(source.get("height"), 3.0),
            frame_thickness=_float(source.get("frame_thickness"), 0.25),
            depth=_float(source.get("depth"), 0.25),
            segments=int(_float(source.get("segments"), 12.0)),
            center=_vec3(source.get("center"), (0.0, 0.0, 1.5)),
            material=material,
        )
    raise ValueError(f"Unsupported authored room composition primitive type: {primitive_type or '(missing)'}")


def _room_primitive_recipe(data: Any, room_resref: str, *, _depth: int = 0) -> Any:
    """Decode one recursively human-readable composition primitive recipe."""

    source = _dict(data)
    base = _base_room_primitive(source, room_resref, _depth=_depth)
    evaluation_transforms = _evaluation_transforms(source.get("evaluation_transforms"))
    if source.get("transform") is None and not evaluation_transforms:
        return base
    return PlacedRoomPrimitive(
        primitive=base,
        transform=_transform(source.get("transform")),
        name=str(source.get("instance_name") or source.get("name") or getattr(base, "name", "")),
        evaluation_transforms=evaluation_transforms,
    )


def _composition_primitive(data: dict[str, Any], room_resref: str) -> AuthoredRoomComposition:
    primitive = _dict(data.get("primitive"))
    floor = _floor_primitive(primitive.get("floor"), room_resref)
    primitives = []
    for raw in primitive.get("primitives", ()) or ():
        primitives.append(_room_primitive_recipe(raw, room_resref))
    return AuthoredRoomComposition(
        room_resref=normalise_resref(primitive.get("room_resref") or room_resref),
        floor=floor,
        primitives=tuple(primitives),
        metadata=_dict(primitive.get("metadata")),
    )


def _height_rows(value: Any) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row in value or ():
        if isinstance(row, (list, tuple)):
            rows.append(tuple(_float(item, 0.0) for item in row))
    return tuple(rows)


def _room_primitive(data: dict[str, Any], room_resref: str) -> RectangularRoomPrimitive | FloorPlanRoomPrimitive | AuthoredRoomComposition | TerrainHeightfieldPrimitive:
    primitive = _dict(data.get("primitive"))
    primitive_type = str(primitive.get("type") or primitive.get("primitive") or data.get("primitive_type") or "rectangular").lower()
    if primitive_type in {"composition", "authored_room_composition"}:
        return _composition_primitive(data, room_resref)
    if primitive_type == IMPORTED_MESH_PRIMITIVE_KIND:
        return imported_mesh_primitive_from_payload(primitive, room_resref)
    if primitive_type in {"terrain_heightfield", "terrain", "heightfield"}:
        holes = tuple(
            (int(cell[0]), int(cell[1]))
            for cell in primitive.get("holes", ()) or ()
            if isinstance(cell, (list, tuple)) and len(cell) >= 2
        )
        return TerrainHeightfieldPrimitive(
            room_resref=normalise_resref(primitive.get("room_resref") or room_resref),
            heights=_height_rows(primitive.get("heights")) or ((0.0, 0.0), (0.0, 0.0)),
            width=_float(primitive.get("width"), 10.0),
            depth=_float(primitive.get("depth"), 10.0),
            floor_surface_id=primitive.get("floor_surface_id", 4),
            non_walk_surface_id=primitive.get("non_walk_surface_id", 7),
            max_walkable_slope_degrees=_float(primitive.get("max_walkable_slope_degrees"), 35.0),
            holes=holes,
            material=_material(primitive.get("material")),
            metadata=_dict(primitive.get("metadata")),
        )
    if primitive_type in {"floor_plan", "floorplan", "floor_plan_extrusion"}:
        points = tuple(point for point in (_vec2(item) for item in primitive.get("points", ()) or ()) if point is not None)
        return FloorPlanRoomPrimitive(
            room_resref=normalise_resref(primitive.get("room_resref") or room_resref),
            points=points,
            z=_float(primitive.get("z"), 0.0),
            wall_height=_float(primitive.get("wall_height"), 3.0),
            floor_surface_id=primitive.get("floor_surface_id", 4),
            material=_material(primitive.get("material")),
            wall_material=(
                _material(primitive.get("wall_material"))
                if primitive.get("wall_material") is not None
                else None
            ),
            ceiling_material=(
                _material(primitive.get("ceiling_material"))
                if primitive.get("ceiling_material") is not None
                else None
            ),
            include_walls=bool(primitive.get("include_walls", True)),
            include_ceiling=bool(primitive.get("include_ceiling", False)),
            openings=tuple(_opening(item) for item in primitive.get("openings", ()) or ()),
            metadata=_dict(primitive.get("metadata")),
        )
    return RectangularRoomPrimitive(
        room_resref=normalise_resref(primitive.get("room_resref") or room_resref),
        width=_float(primitive.get("width"), 10.0),
        depth=_float(primitive.get("depth"), 10.0),
        wall_height=_float(primitive.get("wall_height"), 3.0),
        floor_surface_id=primitive.get("floor_surface_id", 4),
        texture=str(primitive.get("texture") or "default"),
        include_doorway_marker=bool(primitive.get("include_doorway_marker", True)),
    )


def _placement(data: Any, module_root: str) -> AuthoredGameplayPlacement:
    source = _dict(data)
    entry = _dict(source.get("entry_point"))
    placement_metadata = _dict(source.get("metadata"))
    bearing_unit = str(placement_metadata.get("bearing_unit") or "")
    seen_instance_ids: set[str] = set()
    creatures = tuple(
        AuthoredCreatureInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            bearing=_bearing_radians(item.get("bearing"), unit=bearing_unit),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="creature", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("creatures", ()) or ())
    )
    doors = tuple(
        AuthoredDoorInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            bearing=_bearing_radians(item.get("bearing"), unit=bearing_unit),
            linked_to=str(item.get("linked_to") or ""),
            linked_to_module=str(item.get("linked_to_module") or ""),
            linked_to_flags=int(_float(item.get("linked_to_flags"), 0.0)) & 0xFF,
            transition_destination=int(_float(item.get("transition_destination"), 0.0)),
            use_tweak_color=bool(item.get("use_tweak_color", False)),
            tweak_color=int(_float(item.get("tweak_color"), 0.0)) & 0xFFFFFFFF,
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="door", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("doors", ()) or ())
    )
    triggers = tuple(
        AuthoredTriggerInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            geometry=tuple(_vec3(point) for point in item.get("geometry", ()) or ()),
            linked_to=str(item.get("linked_to") or ""),
            linked_to_module=str(item.get("linked_to_module") or ""),
            linked_to_flags=int(_float(item.get("linked_to_flags"), 0.0)) & 0xFF,
            transition_destination=int(_float(item.get("transition_destination"), 0.0)),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="trigger", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("triggers", ()) or ())
    )
    encounters = tuple(
        AuthoredEncounterInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="encounter", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("encounters", ()) or ())
    )
    sounds = tuple(
        AuthoredSoundInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="sound", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("sounds", ()) or ())
    )
    cameras = tuple(
        AuthoredCameraInstance(
            camera_id=item.get("camera_id", item.get("id", 0)),
            position=_vec3(item.get("position")),
            # Retail GIT camera orientation is yaw-only (w, 0, 0, z) in file order.
            orientation=_vec4(item.get("orientation"), (1.0, 0.0, 0.0, 0.0)),
            field_of_view=_float(item.get("field_of_view"), 55.0),
            height=_float(item.get("height"), 0.0),
            mic_range=_float(item.get("mic_range"), 0.0),
            pitch=_float(item.get("pitch"), 90.0),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="camera", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("cameras", ()) or ())
    )
    stores = tuple(
        AuthoredStoreInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            bearing=_bearing_radians(item.get("bearing"), unit=bearing_unit),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="store", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("stores", ()) or ())
    )
    placeables = tuple(
        AuthoredPlaceableInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            bearing=_bearing_radians(item.get("bearing"), unit=bearing_unit),
            use_tweak_color=bool(item.get("use_tweak_color", False)),
            tweak_color=int(_float(item.get("tweak_color"), 0.0)) & 0xFFFFFFFF,
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="placeable", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("placeables", ()) or ())
    )
    waypoints = tuple(
        AuthoredWaypointInstance(
            template_resref=normalise_resref(item.get("template_resref") or item.get("resref") or ""),
            tag=str(item.get("tag") or ""),
            position=_vec3(item.get("position")),
            bearing=_bearing_radians(item.get("bearing"), unit=bearing_unit),
            linked_to=str(item.get("linked_to") or ""),
            instance_id=_migrated_placement_instance_id(
                module_root=module_root, kind="waypoint", index=index, source=item, seen=seen_instance_ids
            ),
        )
        for index, item in enumerate(_dict(raw) for raw in source.get("waypoints", ()) or ())
    )
    return AuthoredGameplayPlacement(
        entry_point=ModuleEntryPoint(
            # An explicit blank value means the user deleted the player start.
            # Only legacy payloads that omit the field inherit module_root.
            area_resref=normalise_resref(entry.get("area_resref") if "area_resref" in entry else module_root),
            position=_vec3(entry.get("position")),
            facing=_bearing_radians(entry.get("facing"), unit=bearing_unit),
        ),
        creatures=creatures,
        doors=doors,
        triggers=triggers,
        encounters=encounters,
        sounds=sounds,
        cameras=cameras,
        stores=stores,
        placeables=placeables,
        waypoints=waypoints,
        metadata={
            **placement_metadata,
            "instance_identity_version": _PLACEMENT_INSTANCE_ID_VERSION,
            "bearing_unit": "radians",
            "bearing_unit_version": _PLACEMENT_BEARING_UNIT_VERSION,
        },
    )


def _lights(data: Any) -> tuple[AuthoredRoomLight, ...]:
    return tuple(normalise_authored_room_light(item) for item in (data or ()))


def _runtime_resources(data: Any) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    for item in data or ():
        if isinstance(item, str):
            stem, dot, ext = item.rpartition(".")
            if dot:
                keys.add((normalise_resref(stem), ext.strip().lower().lstrip(".")))
            continue
        source = _dict(item)
        resref = normalise_resref(source.get("resref") or source.get("name") or "")
        restype = str(source.get("restype") or source.get("type") or "").strip().lower().lstrip(".")
        if resref and restype:
            keys.add((resref, restype))
    return tuple(sorted(keys))


def _runtime_resource_label(key: tuple[str, str]) -> str:
    resref, restype = key
    return f"{resref}.{restype.lstrip('.').lower()}"


def _runtime_output_status(readiness: AuthoredModuleReadiness) -> dict[str, Any]:
    """Summarize generated KOTOR outputs for Map Studio UI/readiness panels."""

    expected = tuple(readiness.expected_runtime_resources)
    present = tuple(readiness.present_runtime_resources)
    missing = tuple(readiness.missing_runtime_resources)
    component_edit = readiness.component_edit
    stale_outputs = tuple(component_edit.stale_outputs)
    regenerate_required = bool(missing or stale_outputs)
    if stale_outputs:
        status = "Stale generated resources"
        fix_hint = component_edit.next_action or component_edit.fix_hint
    elif missing:
        status = "Missing generated resources"
        fix_hint = "Build/export the authored module to regenerate missing KOTOR runtime resources."
    else:
        status = "Current"
        fix_hint = "Runtime resources are current for this authored KMAP state."
    return {
        "status": status,
        "regenerate_required": regenerate_required,
        "expected": [_runtime_resource_label(key) for key in expected],
        "present": [_runtime_resource_label(key) for key in present],
        "missing": [_runtime_resource_label(key) for key in missing],
        "stale_outputs": list(stale_outputs),
        "resource_impacts": [dict(row) for row in component_edit.resource_impacts],
        "edited_resource": component_edit.latest_room_resref,
        "latest_operation": component_edit.latest_operation,
        "next_action": component_edit.next_action,
        "fix_hint": fix_hint,
    }


def _vec3_payload(value: tuple[float, float, float]) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def _material_payload(material: PrimitiveMaterial) -> dict[str, Any]:
    return {
        "texture": material.texture,
        "diffuse": _vec3_payload(material.diffuse),
        "ambient": _vec3_payload(material.ambient),
        "metadata": dict(material.metadata),
    }


def _transform_payload(transform: PrimitiveTransform) -> dict[str, Any]:
    return {
        "translation": _vec3_payload(transform.translation),
        "rotation_degrees_z": float(transform.rotation_degrees_z),
        "scale": _vec3_payload(transform.scale),
        "pivot": _vec3_payload(transform.pivot),
    }


def _construction_recipe_payload(primitive: Any, primitive_type: str, *, room_resref: str = "") -> dict[str, Any]:
    node_id = str(getattr(primitive, "construction_node_id", "") or "").strip()
    if not node_id:
        node_id = primitive_construction_node_id(
            room_resref=room_resref,
            primitive_type=primitive_type,
            name=str(getattr(primitive, "name", "") or ""),
        )
    return {
        "recipe_version": max(1, int(getattr(primitive, "recipe_version", 1))),
        "construction_node_id": node_id,
        "construction_schema_version": max(1, int(getattr(primitive, "construction_schema_version", 1))),
    }


def _base_primitive_payload(
    primitive: FloorPrimitive
    | WallPrimitive
    | CubePrimitive
    | SpherePrimitive
    | ConePrimitive
    | TorusPrimitive
    | RampPrimitive
    | StairsPrimitive
    | CylinderPrimitive
    | DoorFramePrimitive
    | ArchPrimitive
    | CombinedRoomPrimitive
    | PlacedRoomPrimitive,
    *,
    _depth: int = 0,
    room_resref: str = "",
) -> dict[str, Any]:
    if _depth > 32:
        raise ValueError("Combined room primitive recipe nesting exceeds the supported depth of 32.")
    transform: PrimitiveTransform | None = None
    evaluation_transforms: tuple[PrimitiveTransform, ...] = ()
    instance_name = ""
    if isinstance(primitive, PlacedRoomPrimitive):
        transform = primitive.transform
        evaluation_transforms = tuple(primitive.evaluation_transforms or ())
        instance_name = primitive.name
        primitive = primitive.primitive
    payload: dict[str, Any]
    if isinstance(primitive, CombinedRoomPrimitive):
        payload = {
            "type": "combined_mesh",
            "name": primitive.name,
            "sources": [
                {
                    "source_name": source.source_name,
                    "face_indices": [int(index) for index in source.face_indices],
                    "walkmesh_policy": source.walkmesh_policy,
                    "primitive": _base_primitive_payload(source.primitive, _depth=_depth + 1, room_resref=room_resref),
                }
                for source in primitive.sources
            ],
            "metadata": dict(primitive.metadata),
        }
    elif isinstance(primitive, FloorPrimitive):
        payload = {
            "type": "plane",
            "name": primitive.name,
            "width": float(primitive.width),
            "depth": float(primitive.depth),
            "z": float(primitive.z),
            "surface_id": primitive.surface_id,
            "material": _material_payload(primitive.material),
            "subdivisions_width": int(primitive.subdivisions_width),
            "subdivisions_depth": int(primitive.subdivisions_depth),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": int(primitive.create_uvs),
            **_construction_recipe_payload(primitive, "plane", room_resref=room_resref),
        }
    elif isinstance(primitive, WallPrimitive):
        payload = {
            "type": "wall",
            "name": primitive.name,
            "width": float(primitive.width),
            "height": float(primitive.height),
            "thickness": float(primitive.thickness),
            "axis": primitive.axis,
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
        }
    elif isinstance(primitive, CubePrimitive):
        payload = {
            "type": "cube",
            "name": primitive.name,
            "size": _vec3_payload(primitive.size),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
            "subdivisions_x": int(primitive.subdivisions_x),
            "subdivisions_y": int(primitive.subdivisions_y),
            "subdivisions_z": int(primitive.subdivisions_z),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": int(primitive.create_uvs),
            **_construction_recipe_payload(primitive, "cube", room_resref=room_resref),
        }
    elif isinstance(primitive, RampPrimitive):
        payload = {
            "type": "ramp",
            "name": primitive.name,
            "width": float(primitive.width),
            "length": float(primitive.length),
            "height": float(primitive.height),
            "center": _vec3_payload(primitive.center),
            "surface_id": primitive.surface_id,
            "material": _material_payload(primitive.material),
        }
    elif isinstance(primitive, StairsPrimitive):
        payload = {
            "type": "stairs",
            "name": primitive.name,
            "width": float(primitive.width),
            "depth": float(primitive.depth),
            "height": float(primitive.height),
            "steps": int(primitive.steps),
            "surface_id": primitive.surface_id,
            "material": _material_payload(primitive.material),
        }
    elif isinstance(primitive, CylinderPrimitive):
        payload = {
            "type": "cylinder",
            "name": primitive.name,
            "radius": float(primitive.radius),
            "height": float(primitive.height),
            "segments": int(primitive.segments),
            "subdivisions_axis": int(primitive.segments),
            "subdivisions_height": int(primitive.subdivisions_height),
            "subdivisions_caps": int(primitive.subdivisions_caps),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": int(primitive.create_uvs),
            "round_cap": bool(primitive.round_cap),
            "round_cap_height_compensation": bool(primitive.round_cap_height_compensation),
            **_construction_recipe_payload(primitive, "cylinder", room_resref=room_resref),
        }
    elif isinstance(primitive, SpherePrimitive):
        payload = {
            "type": "sphere",
            "name": primitive.name,
            "radius": float(primitive.radius),
            "subdivisions_axis": int(primitive.subdivisions_axis),
            "subdivisions_height": int(primitive.subdivisions_height),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": int(primitive.create_uvs),
            **_construction_recipe_payload(primitive, "sphere", room_resref=room_resref),
        }
    elif isinstance(primitive, ConePrimitive):
        payload = {
            "type": "cone",
            "name": primitive.name,
            "radius": float(primitive.radius),
            "height": float(primitive.height),
            "subdivisions_axis": int(primitive.subdivisions_axis),
            "subdivisions_height": int(primitive.subdivisions_height),
            "subdivisions_caps": int(primitive.subdivisions_caps),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": int(primitive.create_uvs),
            "round_cap": bool(primitive.round_cap),
            **_construction_recipe_payload(primitive, "cone", room_resref=room_resref),
        }
    elif isinstance(primitive, TorusPrimitive):
        payload = {
            "type": "torus",
            "name": primitive.name,
            "radius": float(primitive.radius),
            "section_radius": float(primitive.section_radius),
            "subdivisions_axis": int(primitive.subdivisions_axis),
            "subdivisions_height": int(primitive.subdivisions_height),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
            "axis": _vec3_payload(primitive.axis),
            "height_baseline": float(primitive.height_baseline),
            "create_uvs": bool(primitive.create_uvs),
            "twist": float(primitive.twist),
            **_construction_recipe_payload(primitive, "torus", room_resref=room_resref),
        }
    elif isinstance(primitive, DoorFramePrimitive):
        payload = {
            "type": "door_frame",
            "name": primitive.name,
            "width": float(primitive.width),
            "height": float(primitive.height),
            "jamb_width": float(primitive.jamb_width),
            "lintel_height": float(primitive.lintel_height),
            "depth": float(primitive.depth),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
        }
    elif isinstance(primitive, ArchPrimitive):
        payload = {
            "type": "arch",
            "name": primitive.name,
            "width": float(primitive.width),
            "height": float(primitive.height),
            "frame_thickness": float(primitive.frame_thickness),
            "depth": float(primitive.depth),
            "segments": int(primitive.segments),
            "center": _vec3_payload(primitive.center),
            "material": _material_payload(primitive.material),
        }
    else:
        raise TypeError(f"Unsupported authored room composition primitive: {type(primitive)!r}")
    if transform is not None:
        payload["transform"] = _transform_payload(transform)
        if evaluation_transforms:
            payload["evaluation_transforms"] = [
                _transform_payload(stage) for stage in evaluation_transforms
            ]
        if instance_name:
            payload["instance_name"] = instance_name
    return payload


def _composition_payload(composition: AuthoredRoomComposition) -> dict[str, Any]:
    floor_payload = _base_primitive_payload(composition.floor, room_resref=composition.room_resref)
    floor_payload["type"] = "floor"
    return {
        "type": "composition",
        "room_resref": composition.room_resref,
        "floor": floor_payload,
        "primitives": [_base_primitive_payload(item, room_resref=composition.room_resref) for item in composition.primitives],
        "metadata": dict(composition.metadata),
    }


def _primitive_payload(primitive: RectangularRoomPrimitive | FloorPlanRoomPrimitive | AuthoredRoomComposition | TerrainHeightfieldPrimitive) -> dict[str, Any]:
    if isinstance(primitive, AuthoredRoomComposition):
        return _composition_payload(primitive)
    if isinstance(primitive, ImportedMeshRoomPrimitive):
        return imported_mesh_primitive_payload(primitive)
    if isinstance(primitive, TerrainHeightfieldPrimitive):
        payload = {
            "type": "terrain_heightfield",
            "room_resref": primitive.room_resref,
            "width": float(primitive.width),
            "depth": float(primitive.depth),
            "heights": [[float(value) for value in row] for row in primitive.heights],
            "floor_surface_id": primitive.floor_surface_id,
            "non_walk_surface_id": primitive.non_walk_surface_id,
            "max_walkable_slope_degrees": float(primitive.max_walkable_slope_degrees),
            "material": _material_payload(primitive.material),
            "metadata": dict(primitive.metadata),
        }
        if primitive.holes:
            # Only persisted when present so pre-hole KMAPs stay byte-stable.
            payload["holes"] = [[int(row), int(column)] for row, column in primitive.holes]
        return payload
    if isinstance(primitive, FloorPlanRoomPrimitive):
        payload = {
            "type": "floor_plan",
            "room_resref": primitive.room_resref,
            "points": [[float(x), float(y)] for x, y in primitive.points],
            "z": float(primitive.z),
            "wall_height": float(primitive.wall_height),
            "floor_surface_id": primitive.floor_surface_id,
            "material": _material_payload(primitive.material),
            "include_walls": bool(primitive.include_walls),
            "openings": [
                {
                    "name": opening.name,
                    "edge_index": int(opening.edge_index),
                    "center_fraction": float(opening.center_fraction),
                    "width": float(opening.width),
                    "height": float(opening.height),
                    "bottom": float(opening.bottom),
                    "metadata": dict(opening.metadata),
                }
                for opening in primitive.openings
            ],
            "metadata": dict(primitive.metadata),
        }
        if primitive.wall_material is not None:
            payload["wall_material"] = _material_payload(primitive.wall_material)
        if primitive.ceiling_material is not None:
            payload["ceiling_material"] = _material_payload(primitive.ceiling_material)
        if primitive.include_ceiling:
            # Omit the false default so older KMAPs remain byte-stable.
            payload["include_ceiling"] = True
        return payload
    return {
        "type": "rectangular",
        "room_resref": primitive.room_resref,
        "width": float(primitive.width),
        "depth": float(primitive.depth),
        "wall_height": float(primitive.wall_height),
        "floor_surface_id": primitive.floor_surface_id,
        "texture": primitive.texture,
        "include_doorway_marker": bool(primitive.include_doorway_marker),
    }


def _placement_payload(placement: AuthoredGameplayPlacement, *, module_root: str = "") -> dict[str, Any]:
    seen_instance_ids: set[str] = set()

    def _payload_instance_id(kind: str, index: int, item: Any) -> str:
        source = {
            "instance_id": str(getattr(item, "instance_id", "") or ""),
            "record": repr(replace(item, instance_id="")),
        }
        return _migrated_placement_instance_id(
            module_root=module_root,
            kind=kind,
            index=index,
            source=source,
            seen=seen_instance_ids,
        )

    return {
        "entry_point": {
            "area_resref": placement.entry_point.area_resref,
            "position": _vec3_payload(placement.entry_point.position),
            "facing": _bearing_radians(placement.entry_point.facing, unit="radians"),
        },
        "creatures": [
            {
                "instance_id": _payload_instance_id("creature", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
                "bearing": _bearing_radians(item.bearing, unit="radians"),
            }
            for index, item in enumerate(placement.creatures)
        ],
        "doors": [
            {
                "instance_id": _payload_instance_id("door", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
                "bearing": _bearing_radians(item.bearing, unit="radians"),
                "linked_to": item.linked_to,
                "linked_to_module": item.linked_to_module,
                "linked_to_flags": int(item.linked_to_flags) & 0xFF,
                "transition_destination": int(item.transition_destination),
                "use_tweak_color": bool(item.use_tweak_color),
                "tweak_color": int(item.tweak_color) & 0xFFFFFFFF,
            }
            for index, item in enumerate(placement.doors)
        ],
        "triggers": [
            {
                "instance_id": _payload_instance_id("trigger", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
                "geometry": [_vec3_payload(point) for point in item.geometry],
                "linked_to": item.linked_to,
                "linked_to_module": item.linked_to_module,
                "linked_to_flags": int(item.linked_to_flags) & 0xFF,
                "transition_destination": int(item.transition_destination),
            }
            for index, item in enumerate(placement.triggers)
        ],
        "encounters": [
            {
                "instance_id": _payload_instance_id("encounter", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
            }
            for index, item in enumerate(placement.encounters)
        ],
        "sounds": [
            {
                "instance_id": _payload_instance_id("sound", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
            }
            for index, item in enumerate(placement.sounds)
        ],
        "cameras": [
            {
                "instance_id": _payload_instance_id("camera", index, item),
                "camera_id": item.camera_id,
                "position": _vec3_payload(item.position),
                "orientation": [float(value) for value in item.orientation],
                "field_of_view": float(item.field_of_view),
                "height": float(item.height),
                "mic_range": float(item.mic_range),
                "pitch": float(item.pitch),
            }
            for index, item in enumerate(placement.cameras)
        ],
        "stores": [
            {
                "instance_id": _payload_instance_id("store", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": list(item.position),
                "bearing": _bearing_radians(item.bearing, unit="radians"),
            }
            for index, item in enumerate(placement.stores)
        ],
        "placeables": [
            {
                "instance_id": _payload_instance_id("placeable", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
                "bearing": _bearing_radians(item.bearing, unit="radians"),
                "use_tweak_color": bool(item.use_tweak_color),
                "tweak_color": int(item.tweak_color) & 0xFFFFFFFF,
            }
            for index, item in enumerate(placement.placeables)
        ],
        "waypoints": [
            {
                "instance_id": _payload_instance_id("waypoint", index, item),
                "template_resref": item.template_resref,
                "tag": item.tag,
                "position": _vec3_payload(item.position),
                "bearing": _bearing_radians(item.bearing, unit="radians"),
            }
            for index, item in enumerate(placement.waypoints)
        ],
        "metadata": {
            **dict(placement.metadata),
            "instance_identity_version": _PLACEMENT_INSTANCE_ID_VERSION,
            "bearing_unit": "radians",
            "bearing_unit_version": _PLACEMENT_BEARING_UNIT_VERSION,
        },
    }


def _authored_payload_invalidation(project: AuthoredModuleProject) -> dict[str, Any]:
    """Summarize authored edits that make packaged runtime/proof artifacts stale."""

    stale_outputs: list[str] = []
    edited_rooms: list[str] = []
    latest_operation = ""
    latest_summary = ""
    next_action = ""
    invalidates_previous_export = False
    invalidates_game_proof = False

    def add_stale_outputs(values: Any) -> None:
        for value in tuple(values or ()):
            output = str(value or "").strip()
            if output and output not in stale_outputs:
                stale_outputs.append(output)

    for room in tuple(project.rooms or ()):
        room_resref = normalise_resref(room.room_resref)
        primitive = getattr(room, "primitive", None)
        audit = dict(room.metadata.get("last_component_edit_audit") or {})
        if not audit and primitive is not None:
            audit = dict(getattr(primitive, "metadata", {}).get("last_component_edit_audit") or {})
        if audit:
            invalidates_previous_export = invalidates_previous_export or bool(audit.get("export_candidate_stale"))
            invalidates_game_proof = invalidates_game_proof or bool(audit.get("game_proof_stale"))
            add_stale_outputs(audit.get("stale_outputs"))
            if room_resref and room_resref not in edited_rooms:
                edited_rooms.append(room_resref)
            latest_operation = str(audit.get("operation") or latest_operation)
            latest_summary = str(audit.get("summary") or latest_summary)
            next_action = str(audit.get("next_action") or next_action)
        style = dict(room.metadata.get("last_room_style_update") or {})
        if style:
            invalidates_previous_export = True
            invalidates_game_proof = True
            add_stale_outputs(("MDL", "MDX", "WOK", "LYT", "VIS", "PTH", ".mod"))
            if room_resref and room_resref not in edited_rooms:
                edited_rooms.append(room_resref)
            latest_operation = "room_style_update"
            latest_summary = (
                f"Applied texture {style.get('texture', '')}, WOK surface "
                f"{style.get('floor_surface_id', '')} ({style.get('floor_surface_name', '')})."
            )
            next_action = "Regenerate the module package and record fresh in-game proof after material or WOK changes."

    project_extra = dict(getattr(project, "extra", {}) or {})
    placement_edit_keys = (
        "last_entry_point_update",
        "last_gameplay_placement",
        "last_gameplay_placement_transform",
        "last_gameplay_placement_rename",
        "last_gameplay_placement_duplicate",
        "last_gameplay_placement_remove",
        "last_creature_behavior_update",
    )
    for key in placement_edit_keys:
        edit = dict(project_extra.get(key) or {}) if isinstance(project_extra.get(key), dict) else {}
        if not edit:
            continue
        invalidates_previous_export = True
        invalidates_game_proof = True
        add_stale_outputs(("GIT", "IFO", "PTH", ".mod"))
        latest_operation = key
        kind = str(edit.get("kind") or "entry_point")
        tag = str(edit.get("tag") or edit.get("placement_id") or edit.get("area_resref") or "").strip()
        latest_summary = f"Updated {kind} {tag}".strip()
        next_action = "Regenerate gameplay resources and record fresh in-game proof after placement or entry-point changes."

    lighting_edit_keys = (
        "last_room_light",
        "last_room_light_rename",
        "last_room_light_duplicate",
        "last_room_light_remove",
    )
    for key in lighting_edit_keys:
        edit = dict(project_extra.get(key) or {}) if isinstance(project_extra.get(key), dict) else {}
        if not edit:
            continue
        invalidates_previous_export = True
        invalidates_game_proof = True
        add_stale_outputs(("MDL", "MDX", ".mod"))
        room_resref = normalise_resref(edit.get("room_resref"))
        if room_resref and room_resref not in edited_rooms:
            edited_rooms.append(room_resref)
        latest_operation = key
        name = str(edit.get("name") or "room light").strip()
        latest_summary = f"Updated room light {name}".strip()
        next_action = "Regenerate room model resources and record fresh in-game proof after lighting changes."

    script_edit_keys = (
        "last_script_hook",
        "last_script_hook_remove",
    )
    for key in script_edit_keys:
        edit = dict(project_extra.get(key) or {}) if isinstance(project_extra.get(key), dict) else {}
        if not edit:
            continue
        invalidates_previous_export = True
        invalidates_game_proof = True
        add_stale_outputs(("ARE", "IFO", ".mod"))
        latest_operation = key
        scope = str(edit.get("scope") or "script").strip()
        field_name = str(edit.get("field_name") or "").strip()
        script = str(edit.get("script_resref") or "").strip()
        latest_summary = f"Updated {scope} script hook {field_name} {script}".strip()
        next_action = "Regenerate ARE/IFO script-hook resources and record fresh in-game proof after script changes."

    world_lighting_edit = dict(project_extra.get("last_world_lighting_update") or {})
    if world_lighting_edit:
        invalidates_previous_export = True
        invalidates_game_proof = True
        add_stale_outputs(("ARE", ".mod"))
        latest_operation = "last_world_lighting_update"
        profile = str(world_lighting_edit.get("profile") or "standard").strip()
        fog_state = "enabled" if bool(world_lighting_edit.get("fog_enabled")) else "disabled"
        latest_summary = f"Updated ARE world lighting ({profile}) with distance fog {fog_state}."
        next_action = "Regenerate the ARE/module package and record fresh in-game proof after world-lighting changes."

    if not invalidates_previous_export and not invalidates_game_proof and not stale_outputs:
        return {}
    return {
        "invalidates_previous_export": bool(invalidates_previous_export or stale_outputs),
        "invalidates_game_proof": bool(invalidates_game_proof or stale_outputs),
        "edited_rooms": edited_rooms,
        "latest_operation": latest_operation,
        "latest_summary": latest_summary,
        "stale_outputs": stale_outputs,
        "next_action": next_action
        or "Regenerate MDL/MDX/WOK/LYT/VIS/PTH/.mod resources and record fresh game proof.",
    }


def authored_project_to_kmap_payload(
    project: AuthoredModuleProject,
    *,
    runtime_resources: tuple[str, ...] = (),
    game_tested: bool = False,
) -> dict[str, Any]:
    """Convert an authored module project to the serializable KMAP section."""

    invalidation = _authored_payload_invalidation(project)
    payload = {
        "module_root": project.metadata.module_root,
        "game": project.game,
        "display_name": project.metadata.display_name,
        "tag": project.metadata.tag,
        "description": project.metadata.description,
        "capability_stage": project.metadata.capability_stage,
        "metadata": dict(project.metadata.metadata),
        "rooms": [
            {
                "room_resref": room.room_resref,
                "primitive": _primitive_payload(room.primitive),
                "position": _vec3_payload(room.position),
                "visible_rooms": list(room.visible_rooms),
                "metadata": dict(room.metadata),
            }
            for room in project.rooms
        ],
        "placements": _placement_payload(project.placements, module_root=project.module_root),
        "lights": [authored_room_light_payload(light) for light in project.lights],
        "notes": list(project.notes),
        "extra": dict(project.extra),
        "runtime_resources": list(runtime_resources),
        "game_tested": bool(game_tested) and not bool(invalidation.get("invalidates_game_proof")),
    }
    if invalidation:
        payload["export_proof_invalidation"] = invalidation
        payload["manual_proof_required"] = True
    return payload


def create_dev_test_authored_module_payload(
    *,
    module_root: str = "grdev01",
    game: str = "K1",
    include_test_placeable: bool = True,
    include_start_waypoint: bool = True,
    include_doorway_marker: bool = True,
    include_basic_light: bool = True,
) -> dict[str, Any]:
    """Create the first editable from-scratch Map Studio KMAP payload."""

    root = normalise_resref(module_root)
    room_resref = normalise_resref(f"{root}_room01")
    placeables = ()
    if include_test_placeable:
        placeables = (
            AuthoredPlaceableInstance(
                template_resref="plc_bench",
                tag="grdev01_test_placeable",
                position=(1.75, 1.5, 0.0),
            ),
        )
    waypoints = ()
    if include_start_waypoint:
        waypoints = (
            AuthoredWaypointInstance(
                template_resref="sw_startloc001",
                tag="start",
                position=(0.0, -3.0, 0.0),
            ),
        )
    lights = ()
    if include_basic_light:
        lights = (
            AuthoredRoomLight(
                name=f"{root}_key_light"[:32],
                room_resref=room_resref,
                position=(0.0, -1.5, 2.45),
                color=(1.0, 0.92, 0.76),
                radius=8.0,
                intensity=0.65,
                light_type="point",
                metadata={
                    "source": "map_studio:dev_test_smoke_light",
                    "purpose": "canonical_smoke_visibility",
                },
            ),
        )
    project = create_single_room_project(
        module_root=root,
        game=game,
        display_name="GhostRigger Dev Test",
        room_primitive=RectangularRoomPrimitive(
            room_resref=room_resref,
            width=10.0,
            depth=10.0,
            wall_height=3.0,
            floor_surface_id=4,
            texture=normalize_authored_room_texture(DEFAULT_AUTHORED_ROOM_TEXTURE),
            include_doorway_marker=bool(include_doorway_marker),
        ),
        lights=lights,
        placements=AuthoredGameplayPlacement(
            entry_point=ModuleEntryPoint(area_resref=root, position=(0.0, -3.0, 0.0)),
            placeables=placeables,
            waypoints=waypoints,
            metadata={
                "source": "map_studio:kmap_authored_module",
                "player_start_is_module_entry": True,
                "include_test_placeable": bool(include_test_placeable),
                "include_start_waypoint": bool(include_start_waypoint),
                "include_doorway_marker": bool(include_doorway_marker),
                "include_basic_light": bool(include_basic_light),
                "placeable_count": len(placeables),
                "waypoint_count": len(waypoints),
                "light_count": len(lights),
            },
        ),
        notes=(
            "T2601 from-scratch smoke module.",
            "Editable KMAP-authored primitive room with player start and optional diagnostic placed content.",
        ),
        metadata={
            "task": "T2601",
            "source": "map_studio:kmap_authored_module",
            "content_origin": "map_studio_original",
            "authored_from_scratch": True,
            "copied_from_base_game_module": False,
            "source_module_resref": "",
            "inherited_base_game_module_content": False,
            "inherited_scripted_movers_expected": False,
            "room_geometry_mode": "rectangular_composition",
            "lighting": {
                "profile": "standard",
                "source": "map_studio:dev_test_balanced",
                "purpose": "textured_graybox_visibility",
                "sun_ambient": [48, 48, 48],
                "sun_diffuse": [150, 150, 150],
                "dynamic_ambient": [72, 72, 72],
                "shadow_opacity": 80,
                "sun_shadows": 0,
            },
            "include_doorway_marker": bool(include_doorway_marker),
            "include_basic_light": bool(include_basic_light),
        },
    )
    return authored_project_to_kmap_payload(project)


def create_golden_test_authored_module_payload(
    *,
    module_root: str = "grgold01",
    game: str = "K1",
) -> dict[str, Any]:
    """Create the canonical full Map Studio module fixture for game proof."""

    root = normalise_resref(module_root)
    room_resref = normalise_resref(f"{root}_room01")
    project = create_single_room_project(
        module_root=root,
        game=game,
        display_name="GhostRigger Golden Map",
        room_primitive=RectangularRoomPrimitive(
            room_resref=room_resref,
            width=12.0,
            depth=12.0,
            wall_height=3.0,
            floor_surface_id=4,
            texture=normalize_authored_room_texture(DEFAULT_AUTHORED_ROOM_TEXTURE),
            include_doorway_marker=True,
        ),
        lights=(
            AuthoredRoomLight(
                name=f"{root}_key_light"[:32],
                room_resref=room_resref,
                position=(0.0, -1.5, 2.6),
                color=(1.0, 0.92, 0.76),
                radius=9.0,
                intensity=0.65,
                light_type="point",
                metadata={
                    "source": "map_studio:golden_module_light",
                    "purpose": "t3105_golden_smoke_visibility",
                },
            ),
        ),
        placements=AuthoredGameplayPlacement(
            entry_point=ModuleEntryPoint(area_resref=root, position=(0.0, -4.0, 0.0), facing=0.0),
            creatures=(
                AuthoredCreatureInstance(
                    template_resref="c_drdmkone",
                    tag=f"{root}_npc",
                    position=(-2.0, 1.0, 0.0),
                    bearing=180.0,
                ),
            ),
            doors=(
                AuthoredDoorInstance(
                    template_resref="door_t01",
                    tag=f"{root}_door",
                    position=(0.0, 4.25, 0.0),
                    bearing=0.0,
                    linked_to=f"{root}_exit",
                    linked_to_module=root,
                    linked_to_flags=2,
                    transition_destination=1,
                ),
            ),
            placeables=(
                AuthoredPlaceableInstance(
                    template_resref="plc_bench",
                    tag=f"{root}_bench",
                    position=(2.0, 1.25, 0.0),
                    bearing=90.0,
                ),
            ),
            waypoints=(
                AuthoredWaypointInstance(
                    template_resref="sw_startloc001",
                    tag="start",
                    position=(0.0, -4.0, 0.0),
                ),
                AuthoredWaypointInstance(
                    template_resref="wp_test",
                    tag=f"{root}_exit",
                    position=(0.0, 3.25, 0.0),
                ),
            ),
            metadata={
                "source": "map_studio:t3105_golden_module",
                "player_start_is_module_entry": True,
                "golden_module_stage": "requires_live_warp_proof",
                "includes_room": True,
                "includes_walkmesh": True,
                "includes_placeable": True,
                "includes_waypoint": True,
                "includes_door_transition": True,
                "includes_creature": True,
                "expected_warp_command": f"warp {root}",
            },
        ),
        notes=(
            "T3105 golden custom module fixture.",
            "Includes player start, generated WOK, placeable, waypoint, door transition intent, and NPC for live warp proof.",
        ),
        metadata={
            "task": "T3105",
            "source": "map_studio:t3105_golden_module",
            "content_origin": "map_studio_original",
            "authored_from_scratch": True,
            "copied_from_base_game_module": False,
            "source_module_resref": "",
            "inherited_base_game_module_content": False,
            "inherited_scripted_movers_expected": False,
            "room_geometry_mode": "rectangular_composition",
            "fixture_role": "golden_module_in_game_smoke_test",
            "completion_requirement": "Package, install, warp in-game, verify expected objects, and record proof manifest.",
            "lighting": {
                "profile": "standard",
                "source": "map_studio:golden_module_balanced",
                "purpose": "golden_textured_room_visibility",
                "sun_ambient": [48, 48, 48],
                "sun_diffuse": [150, 150, 150],
                "dynamic_ambient": [72, 72, 72],
                "shadow_opacity": 80,
                "sun_shadows": 0,
            },
        },
    )
    return authored_project_to_kmap_payload(project)


def authored_project_from_kmap_payload(payload: Any, *, fallback_name: str = "new_level", fallback_game: str = "K1") -> AuthoredModuleProject:
    """Convert a serializable KMAP ``authored_module`` section to core intent."""

    data = _dict(payload)
    module_root = normalise_resref(data.get("module_root") or data.get("resref") or fallback_name)
    rooms: list[AuthoredRoomSpec] = []
    for index, room_data in enumerate(data.get("rooms", ()) or ()):
        room_source = _dict(room_data)
        room_resref = normalise_resref(room_source.get("room_resref") or room_source.get("resref") or f"{module_root}_r{index + 1}")
        rooms.append(
            AuthoredRoomSpec(
                room_resref=room_resref,
                primitive=_room_primitive(room_source, room_resref),
                position=_vec3(room_source.get("position")),
                visible_rooms=tuple(normalise_resref(item) for item in room_source.get("visible_rooms", ()) or ()),
                metadata=_dict(room_source.get("metadata")),
            )
        )
    placements = _placement(data.get("placements"), module_root)
    entry_area = normalise_resref(placements.entry_point.area_resref)
    room_resrefs = {room.normalised_resref() for room in rooms}
    extra = _dict(data.get("extra"))
    if entry_area and entry_area != module_root and entry_area in room_resrefs:
        # Early multi-room Map Studio builds overloaded the IFO area field
        # with a room resref so PIE could remember which room contained the
        # player start. Odyssey's Mod_Entry_Area must name the ARE/module
        # resref instead. Preserve the useful room hint as editor metadata
        # while migrating the runtime field to the module root.
        migration = {
            "source_area_resref": entry_area,
            "runtime_area_resref": module_root,
            "reason": "legacy_room_resref_in_module_entry_area",
        }
        placements = replace(
            placements,
            entry_point=replace(placements.entry_point, area_resref=module_root),
            metadata={
                **dict(placements.metadata),
                "entry_room_resref": entry_area,
                "entry_area_migration": migration,
            },
        )
        extra = {
            **extra,
            "entry_room_resref": entry_area,
            "entry_area_migration": migration,
        }
    return AuthoredModuleProject(
        metadata=AuthoredModuleMetadata(
            module_root=module_root,
            game=str(data.get("game") or fallback_game or "K1").upper(),
            display_name=str(data.get("display_name") or fallback_name or module_root),
            tag=str(data.get("tag") or module_root),
            description=str(data.get("description") or ""),
            capability_stage=str(data.get("capability_stage") or "export_candidate"),
            metadata=_dict(data.get("metadata")),
        ),
        rooms=tuple(rooms),
        placements=placements,
        lights=_lights(data.get("lights")),
        notes=tuple(str(item) for item in data.get("notes", ()) or ()),
        extra=extra,
    )


def build_kmap_authored_module_readiness(kmap_project: Any) -> AuthoredModuleKMapBridgeResult:
    """Return authored-module readiness for a KMAP project when present."""

    extra = _dict(getattr(kmap_project, "extra_sections", {}))
    metadata = _dict(getattr(kmap_project, "metadata", {}))
    payload = extra.get("authored_module") or metadata.get("authored_module")
    if payload is None:
        return AuthoredModuleKMapBridgeResult(
            warnings=("No authored Map Studio module section is stored in this KMAP yet.",),
            metadata={"source": "src.core.modules.authored_module_kmap_bridge", "has_payload": False},
        )
    try:
        project = authored_project_from_kmap_payload(
            payload,
            fallback_name=str(getattr(kmap_project, "name", "") or "new_level"),
            fallback_game=str(getattr(kmap_project, "game", "") or "K1"),
        )
    except Exception as exc:
        return AuthoredModuleKMapBridgeResult(
            blocking_messages=(f"Authored module section could not be parsed: {exc}",),
            metadata={"source": "src.core.modules.authored_module_kmap_bridge", "has_payload": True},
        )
    payload_dict = _dict(payload)
    resources = _runtime_resources(payload_dict.get("runtime_resources"))
    proof_metadata = {
        key: payload_dict[key]
        for key in (
            "proof_manifest_path",
            "checklist_path",
            "installed_module_path",
            "backup_module_path",
            "resolved_modules_dir",
            "resolved_game_root_dir",
            "launch_helper_command",
            "elevated_launch_script_path",
            "proof_recording_script_path",
            "in_game_proof_evidence_path",
            "evidence_path",
            "game_tested",
            "manual_proof_required",
            "game_test",
            "modder_test_plan",
            "package_resource_inventory",
            "export_job",
            "pack_manifest_path",
            "export_proof_invalidation",
        )
        if key in payload_dict and payload_dict[key] not in ("", None)
    }
    readiness = build_authored_module_readiness(
        project,
        packaged_resources=resources,
        game_tested=bool(payload_dict.get("game_tested", False)),
        proof_metadata=proof_metadata,
    )
    texture_paint_unapplied = texture_paint_has_unapplied_changes(payload_dict)
    pending_paint_resrefs = texture_paint_pending_resrefs(payload_dict)
    texture_apply_blocking = (TEXTURE_PAINT_UNAPPLIED_BLOCKER,) if texture_paint_unapplied else ()
    texture_issues = KMapValidator().validate_authored_project_textures(kmap_project)
    texture_blocking = tuple(
        dict.fromkeys(issue.message for issue in texture_issues if str(issue.severity).lower() == "error")
    )
    texture_warnings = tuple(
        dict.fromkeys(issue.message for issue in texture_issues if str(issue.severity).lower() == "warning")
    )
    if texture_issues or texture_apply_blocking:
        readiness_metadata = dict(readiness.metadata or {})
        if texture_issues:
            readiness_metadata["project_texture_validation"] = {
                "issue_count": len(texture_issues),
                "blocking_count": len(texture_blocking),
                "warning_count": len(texture_warnings),
                "issues": [issue.to_dict() for issue in texture_issues],
                "reference_policy": "KMAP stores project-relative paths; image bytes remain external until explicit export.",
            }
        readiness_metadata["texture_paint_apply"] = {
            "unapplied": bool(texture_paint_unapplied),
            "pending_resrefs": list(pending_paint_resrefs),
            "export_blocked": bool(texture_apply_blocking),
        }
        combined_texture_blocking = tuple((*texture_blocking, *texture_apply_blocking))
        readiness = replace(
            readiness,
            can_export_candidate=bool(readiness.can_export_candidate) and not combined_texture_blocking,
            ready_for_game_test=bool(readiness.ready_for_game_test) and not combined_texture_blocking,
            export_status=(
                "Apply texture changes"
                if texture_apply_blocking and readiness.can_preview
                else "Project textures not ready"
                if texture_blocking and readiness.can_preview
                else readiness.export_status
            ),
            next_action=(
                "Click Apply Texture Changes in Texture Paint, then validate and export again."
                if texture_apply_blocking and readiness.can_preview
                else "Relink or re-import the authored room project textures, then validate again before export."
                if texture_blocking and readiness.can_preview
                else readiness.next_action
            ),
            blocking_messages=tuple(dict.fromkeys((*readiness.blocking_messages, *combined_texture_blocking))),
            warnings=tuple(dict.fromkeys((*readiness.warnings, *texture_warnings))),
            metadata=readiness_metadata,
        )
    return AuthoredModuleKMapBridgeResult(
        project=project,
        readiness=readiness,
        runtime_resources=resources,
        metadata={
            "source": "src.core.modules.authored_module_kmap_bridge",
            "has_payload": True,
            "runtime_output_status": _runtime_output_status(readiness),
            "project_texture_validation": dict(readiness.metadata.get("project_texture_validation") or {}),
        },
    )


__all__ = [
    "AuthoredModuleKMapBridgeResult",
    "TEXTURE_PAINT_UNAPPLIED_BLOCKER",
    "authored_project_from_kmap_payload",
    "authored_project_to_kmap_payload",
    "build_kmap_authored_module_readiness",
    "create_dev_test_authored_module_payload",
    "create_golden_test_authored_module_payload",
    "texture_paint_has_unapplied_changes",
    "texture_paint_pending_resrefs",
]
