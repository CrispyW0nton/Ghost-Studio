"""Headless authored gameplay object placement for Map Studio.

Future Map Studio panels should edit these objects, then compile them into
Odyssey GIT/IFO resources.  Keeping this Qt-free lets tests and exporters prove
module-placement behavior without constructing the UI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from pykotor.common.language import LocalizedString

from .authored_walkmesh_sampling import walkmesh_face_at_xy, walkmesh_floor_z_at_xy
from .authored_walkmesh_surfaces import walkable_walkmesh_surface_ids, walkmesh_surface_name


Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
_RESREF_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
GIT_STRUCT_ID_CAMERA = 14
GIT_STRUCT_ID_CREATURE = 4
GIT_STRUCT_ID_DOOR = 8
GIT_STRUCT_ID_ENCOUNTER = 7
GIT_STRUCT_ID_PLACEABLE = 9
GIT_STRUCT_ID_SOUND = 6
GIT_STRUCT_ID_STORE = 11
GIT_STRUCT_ID_TRIGGER = 1
GIT_STRUCT_ID_TRIGGER_GEOMETRY = 3
GIT_STRUCT_ID_WAYPOINT = 5


@dataclass(frozen=True)
class ModuleEntryPoint:
    """Player start data written to IFO module entry fields."""

    area_resref: str
    position: Vec3 = (0.0, 0.0, 0.0)
    facing: float = 0.0


@dataclass(frozen=True)
class AuthoredPlaceableInstance:
    """Authored UTP-backed placeable instance for a GIT Placeable List entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    bearing: float = 0.0
    # K2-only vanilla GIT presentation fields.  K1 omits them entirely.
    use_tweak_color: bool = False
    tweak_color: int = 0
    # Editor-only durable identity.  The GIT writer deliberately ignores this
    # field and ``tag``; they exist so KMAP selection/naming survives list
    # reordering. Runtime placeable identity comes from the referenced UTP.
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredCreatureInstance:
    """Authored UTC-backed creature instance for a GIT Creature List entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    bearing: float = 0.0
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredDoorInstance:
    """Authored UTD-backed door instance for a GIT Door List entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    bearing: float = 0.0
    linked_to: str = ""
    linked_to_module: str = ""
    linked_to_flags: int = 0
    transition_destination: int = 0
    # K2-only vanilla door presentation fields. K1 omits them entirely.
    use_tweak_color: bool = False
    tweak_color: int = 0
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredWaypointInstance:
    """Authored waypoint/start marker for a GIT WaypointList entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    bearing: float = 0.0
    # Legacy KMAP compatibility only. Vanilla WaypointList rows are transition
    # destinations and do not carry LinkedTo; writers intentionally omit it.
    linked_to: str = ""
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredTriggerInstance:
    """Authored UTT-backed trigger instance for a GIT TriggerList entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    geometry: tuple[Vec3, ...] = ()
    linked_to: str = ""
    linked_to_module: str = ""
    linked_to_flags: int = 0
    transition_destination: int = 0
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredEncounterInstance:
    """Authored UTE-backed encounter instance for a GIT Encounter List entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredSoundInstance:
    """Authored UTS-backed sound instance for a GIT SoundList entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    instance_id: str = ""


#: Retail GIT camera defaults, measured on 235 placed dialogue cameras in KOTOR II.
#: Orientation is a yaw-only quaternion stored (w, 0, 0, z) in file order, and the
#: camera looks along heading = yaw + 90 degrees. Pitch is degrees from straight
#: down (90 = level; retail median 83). Height is metres above Position.
RETAIL_CAMERA_FIELD_OF_VIEW = 55.0
RETAIL_CAMERA_PITCH = 90.0
RETAIL_CAMERA_HEIGHT = 1.5


def camera_orientation_for_heading(heading_degrees: float) -> Vec4:
    """GIT camera Orientation (file order) that looks along ``heading_degrees``."""
    yaw = math.radians(float(heading_degrees) - 90.0)
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def camera_heading_degrees(orientation: Vec4) -> float:
    """Inverse of :func:`camera_orientation_for_heading` for a yaw-only orientation."""
    w, _x, _y, z = (float(value) for value in orientation)
    return (math.degrees(2.0 * math.atan2(z, w)) + 90.0) % 360.0


@dataclass(frozen=True)
class AuthoredCameraInstance:
    """Authored area camera placement for a GIT CameraList entry (retail conventions)."""

    camera_id: int | str
    position: Vec3 = (0.0, 0.0, 0.0)
    orientation: Vec4 = (1.0, 0.0, 0.0, 0.0)
    field_of_view: float = RETAIL_CAMERA_FIELD_OF_VIEW
    height: float = RETAIL_CAMERA_HEIGHT
    mic_range: float = 0.0
    pitch: float = RETAIL_CAMERA_PITCH
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredStoreInstance:
    """Authored UTM-backed store instance for a GIT StoreList entry."""

    template_resref: str
    tag: str = ""
    position: Vec3 = (0.0, 0.0, 0.0)
    bearing: float = 0.0
    instance_id: str = ""


@dataclass(frozen=True)
class AuthoredGameplayPlacement:
    """Compiled gameplay placements for one authored module."""

    entry_point: ModuleEntryPoint
    creatures: tuple[AuthoredCreatureInstance, ...] = ()
    doors: tuple[AuthoredDoorInstance, ...] = ()
    triggers: tuple[AuthoredTriggerInstance, ...] = ()
    encounters: tuple[AuthoredEncounterInstance, ...] = ()
    sounds: tuple[AuthoredSoundInstance, ...] = ()
    cameras: tuple[AuthoredCameraInstance, ...] = ()
    stores: tuple[AuthoredStoreInstance, ...] = ()
    placeables: tuple[AuthoredPlaceableInstance, ...] = ()
    waypoints: tuple[AuthoredWaypointInstance, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthoredGameplayPlacementValidation:
    """Validation summary for authored GIT/IFO placement intent."""

    ok: bool
    warnings: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthoredGameplayWalkmeshCheck:
    """One gameplay placement walkability check against a generated WOK."""

    label: str
    position: Vec3
    ok: bool
    face_index: int = -1
    surface_id: int = -1
    message: str = ""


@dataclass(frozen=True)
class AuthoredGameplayWalkmeshValidation:
    """Validation summary for authored gameplay placements against a WOK."""

    ok: bool
    checks: tuple[AuthoredGameplayWalkmeshCheck, ...] = ()
    warnings: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()


def normalise_resource_resref(value: Any) -> str:
    """Return a KOTOR-style lowercase resref fragment for placed resources."""

    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text[:16]


def _position_ok(position: Vec3) -> bool:
    return len(position) == 3 and all(math.isfinite(float(value)) for value in position)


def _orientation_ok(orientation: Vec4) -> bool:
    if len(orientation) != 4:
        return False
    if not all(math.isfinite(float(value)) for value in orientation):
        return False
    return sum(float(value) * float(value) for value in orientation) > 1.0e-12


def _camera_id_value(camera_id: int | str) -> int | None:
    try:
        value = int(str(camera_id).strip(), 10)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _validate_template(kind: str, template_resref: str, blocking: list[str]) -> None:
    raw = str(template_resref or "").strip()
    if "." in raw:
        raw = raw.rsplit(".", 1)[0]
    if not raw:
        blocking.append(f"{kind} placement requires a template resref.")
        return
    if len(raw) > 16:
        blocking.append(f"{kind} template resref '{raw}' is {len(raw)} characters; KOTOR resrefs must be 16 characters or fewer.")
        return
    if not _RESREF_PATTERN.match(raw):
        blocking.append(f"{kind} template resref '{raw}' may only contain letters, numbers, and underscores.")


def _validate_transition_intent(kind: str, item: Any, blocking: list[str]) -> None:
    label = str(getattr(item, "tag", "") or getattr(item, "template_resref", "") or "(unnamed)").strip()
    linked_to = str(getattr(item, "linked_to", "") or "").strip()
    linked_module = normalise_resource_resref(getattr(item, "linked_to_module", ""))
    try:
        linked_to_flags = int(getattr(item, "linked_to_flags", 0) or 0)
    except (TypeError, ValueError):
        blocking.append(f"{kind} {label} has an invalid LinkedToFlags value.")
        return
    if linked_to_flags not in {0, 1, 2}:
        blocking.append(
            f"{kind} {label} has LinkedToFlags={linked_to_flags}; KOTOR accepts 0 (none), "
            "1 (destination door), or 2 (destination waypoint)."
        )
    try:
        destination_type = int(getattr(item, "transition_destination", 0) or 0)
    except (TypeError, ValueError):
        blocking.append(f"{kind} {label} has an invalid TransitionDestin value.")
        return
    # TransitionDestin is a CExoLocString StringRef into dialog.tlk. -1
    # (0xFFFFFFFF) is the standard "no transition text" sentinel that the
    # writer itself emits and that most stock doors carry; only values below
    # it are genuinely malformed. Treating -1 as an error blocked export of
    # every ordinary open/close door imported from a stock module (921srt).
    if destination_type < -1:
        blocking.append(
            f"{kind} {label} has an invalid TransitionDestin value {destination_type}; "
            "use -1 for no transition text."
        )
        return
    if destination_type > 2147483647:
        blocking.append(f"{kind} {label} has TransitionDestin={destination_type}; dialog.tlk StringRefs are signed 32-bit values.")
        return
    has_transition_text = destination_type > 0
    if not linked_to and linked_module:
        blocking.append(
            f"{kind} {label} has an incomplete transition: LinkedToModule is set to {linked_module}, "
            "but LinkedTo/destination tag is missing."
        )
    if not linked_to and has_transition_text:
        blocking.append(
            f"{kind} {label} has an incomplete transition: TransitionDestin is set, "
            "but LinkedTo/destination tag is missing."
        )
    if linked_to and linked_to_flags == 0:
        blocking.append(
            f"{kind} {label} links to {linked_to}, but LinkedToFlags is 0. Choose destination door (1) "
            "or destination waypoint (2) so the KOTOR engine can resolve the tag."
        )


def validate_authored_gameplay_placement(placement: AuthoredGameplayPlacement) -> AuthoredGameplayPlacementValidation:
    """Validate authored placement intent before compiling GIT/IFO bytes."""

    warnings: list[str] = []
    blocking: list[str] = []
    if not normalise_resource_resref(placement.entry_point.area_resref):
        blocking.append("Module entry point requires an area resref.")
    if not _position_ok(placement.entry_point.position):
        blocking.append("Module entry point position must contain finite XYZ values.")
    for creature in placement.creatures:
        _validate_template("Creature", creature.template_resref, blocking)
        if not _position_ok(creature.position):
            blocking.append(f"Creature {creature.template_resref or '(missing)'} has an invalid position.")
    for door in placement.doors:
        _validate_template("Door", door.template_resref, blocking)
        if not _position_ok(door.position):
            blocking.append(f"Door {door.template_resref or '(missing)'} has an invalid position.")
        _validate_transition_intent("Door", door, blocking)
    for trigger in placement.triggers:
        _validate_template("Trigger", trigger.template_resref, blocking)
        if not _position_ok(trigger.position):
            blocking.append(f"Trigger {trigger.template_resref or '(missing)'} has an invalid position.")
        _validate_transition_intent("Trigger", trigger, blocking)
        if not trigger.geometry:
            warnings.append(f"Trigger {trigger.template_resref or trigger.tag or '(unnamed)'} has no polygon geometry yet.")
        for point in trigger.geometry:
            if not _position_ok(point):
                blocking.append(f"Trigger {trigger.template_resref or '(missing)'} has invalid polygon geometry.")
                break
    for placeable in placement.placeables:
        _validate_template("Placeable", placeable.template_resref, blocking)
        if not _position_ok(placeable.position):
            blocking.append(f"Placeable {placeable.template_resref or '(missing)'} has an invalid position.")
    for waypoint in placement.waypoints:
        _validate_template("Waypoint", waypoint.template_resref, blocking)
        if not _position_ok(waypoint.position):
            blocking.append(f"Waypoint {waypoint.template_resref or '(missing)'} has an invalid position.")
    for encounter in placement.encounters:
        _validate_template("Encounter", encounter.template_resref, blocking)
        if not _position_ok(encounter.position):
            blocking.append(f"Encounter {encounter.template_resref or '(missing)'} has an invalid position.")
    for sound in placement.sounds:
        _validate_template("Sound", sound.template_resref, blocking)
        if not _position_ok(sound.position):
            blocking.append(f"Sound {sound.template_resref or '(missing)'} has an invalid position.")
    for camera in placement.cameras:
        if _camera_id_value(camera.camera_id) is None:
            blocking.append("Camera placement requires a non-negative numeric camera id.")
        if not _position_ok(camera.position):
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid position.")
        if not _orientation_ok(camera.orientation):
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid orientation quaternion.")
        if not math.isfinite(float(camera.field_of_view)) or float(camera.field_of_view) < 0.0:
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid field of view.")
        if not math.isfinite(float(camera.height)):
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid height.")
        if not math.isfinite(float(camera.mic_range)):
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid mic range.")
        if not math.isfinite(float(camera.pitch)):
            blocking.append(f"Camera {camera.camera_id or '(missing)'} has an invalid pitch.")
        elif not 10.0 <= float(camera.pitch) <= 170.0:
            warnings.append(
                f"Camera {camera.camera_id or '(missing)'} pitch {float(camera.pitch):g} looks nearly straight "
                "down or up; KOTOR pitch is degrees from straight down (90 = level)."
            )
        if _orientation_ok(camera.orientation) and (
            abs(float(camera.orientation[1])) > 1.0e-4 or abs(float(camera.orientation[2])) > 1.0e-4
        ):
            warnings.append(
                f"Camera {camera.camera_id or '(missing)'} orientation is not a yaw-only (w, 0, 0, z) "
                "quaternion; retail cameras store yaw only and use Pitch for tilt."
            )
    for store in placement.stores:
        _validate_template("Store", store.template_resref, blocking)
    return AuthoredGameplayPlacementValidation(
        ok=not blocking,
        warnings=tuple(warnings),
        blocking_issues=tuple(blocking),
    )


def _walkmesh_face_at_position(wok: Any, position: Vec3) -> int:
    return walkmesh_face_at_xy(wok, float(position[0]), float(position[1]))


def _triangle_floor_z_at_position(position: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    """Return the WOK triangle plane height under a gameplay marker."""

    px = float(position[0])
    py = float(position[1])
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
    denominator = ((by - cy) * (ax - cx)) + ((cx - bx) * (ay - cy))
    if abs(denominator) <= 1e-9:
        return (az + bz + cz) / 3.0
    w_a = (((by - cy) * (px - cx)) + ((cx - bx) * (py - cy))) / denominator
    w_b = (((cy - ay) * (px - cx)) + ((ax - cx) * (py - cy))) / denominator
    w_c = 1.0 - w_a - w_b
    return (w_a * az) + (w_b * bz) + (w_c * cz)


def _walkmesh_check(label: str, position: Vec3, wok: Any, *, z_tolerance: float) -> AuthoredGameplayWalkmeshCheck:
    if not _position_ok(position):
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            message=f"{label} position must contain finite XYZ values.",
        )
    face_index = _walkmesh_face_at_position(wok, position)
    if face_index == -2:
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            message=f"{label} cannot be checked because the WOK does not support point lookup.",
        )
    if face_index < 0:
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            face_index=face_index,
            message=f"{label} is outside the generated room walkmesh.",
        )
    faces = list(getattr(wok, "faces", ()) or ())
    verts = list(getattr(wok, "verts", ()) or ())
    if face_index >= len(faces):
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            face_index=face_index,
            message=f"{label} resolved to missing WOK face {face_index}.",
        )
    face = faces[face_index]
    surface_id = int(getattr(face, "surface", -1))
    surface_name = walkmesh_surface_name(surface_id)
    if surface_id not in walkable_walkmesh_surface_ids():
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            face_index=face_index,
            surface_id=surface_id,
            message=f"{label} is on WOK face {face_index}, but surface {surface_id} ({surface_name}) is not walkable.",
        )
    vertex_indices = (int(getattr(face, "v1", -1)), int(getattr(face, "v2", -1)), int(getattr(face, "v3", -1)))
    if any(vertex_index < 0 or vertex_index >= len(verts) for vertex_index in vertex_indices):
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            face_index=face_index,
            surface_id=surface_id,
            message=f"{label} resolved to WOK face {face_index} with invalid vertex indices.",
        )
    floor_z = walkmesh_floor_z_at_xy(wok, face_index, float(position[0]), float(position[1]))
    if floor_z is None:
        floor_z = _triangle_floor_z_at_position(
            position,
            tuple(verts[vertex_indices[0]]),  # type: ignore[arg-type]
            tuple(verts[vertex_indices[1]]),  # type: ignore[arg-type]
            tuple(verts[vertex_indices[2]]),  # type: ignore[arg-type]
        )
    if abs(float(position[2]) - floor_z) > float(z_tolerance):
        return AuthoredGameplayWalkmeshCheck(
            label=label,
            position=position,
            ok=False,
            face_index=face_index,
            surface_id=surface_id,
            message=f"{label} Z={float(position[2]):.3f} is not on generated floor Z={floor_z:.3f}.",
        )
    return AuthoredGameplayWalkmeshCheck(
        label=label,
        position=position,
        ok=True,
        face_index=face_index,
        surface_id=surface_id,
        message=f"{label} is on walkable WOK face {face_index} ({surface_name}).",
    )


def validate_authored_gameplay_placement_against_walkmesh(
    placement: AuthoredGameplayPlacement,
    wok: Any,
    *,
    z_tolerance: float = 0.05,
) -> AuthoredGameplayWalkmeshValidation:
    """Validate that gameplay placements that need pathing sit on walkable WOK space."""

    warnings: list[str] = []
    blocking: list[str] = []
    checks: list[AuthoredGameplayWalkmeshCheck] = []
    if not math.isfinite(float(z_tolerance)) or float(z_tolerance) < 0.0:
        blocking.append("Gameplay placement walkmesh Z tolerance must be finite and non-negative.")
        z_tolerance = 0.0

    checks.append(_walkmesh_check("entry_point", placement.entry_point.position, wok, z_tolerance=float(z_tolerance)))
    for index, creature in enumerate(placement.creatures):
        label = creature.tag or normalise_resource_resref(creature.template_resref) or f"creature_{index + 1}"
        checks.append(_walkmesh_check(f"creature:{label}", creature.position, wok, z_tolerance=float(z_tolerance)))
    for index, door in enumerate(placement.doors):
        label = door.tag or normalise_resource_resref(door.template_resref) or f"door_{index + 1}"
        checks.append(_walkmesh_check(f"door:{label}", door.position, wok, z_tolerance=float(z_tolerance)))
    for index, trigger in enumerate(placement.triggers):
        label = trigger.tag or normalise_resource_resref(trigger.template_resref) or f"trigger_{index + 1}"
        checks.append(_walkmesh_check(f"trigger:{label}", trigger.position, wok, z_tolerance=float(z_tolerance)))
        for point_index, point in enumerate(trigger.geometry):
            checks.append(_walkmesh_check(f"trigger:{label}:point_{point_index + 1}", point, wok, z_tolerance=float(z_tolerance)))
    for index, encounter in enumerate(placement.encounters):
        label = encounter.tag or normalise_resource_resref(encounter.template_resref) or f"encounter_{index + 1}"
        checks.append(_walkmesh_check(f"encounter:{label}", encounter.position, wok, z_tolerance=float(z_tolerance)))
    for index, placeable in enumerate(placement.placeables):
        label = placeable.tag or normalise_resource_resref(placeable.template_resref) or f"placeable_{index + 1}"
        checks.append(_walkmesh_check(f"placeable:{label}", placeable.position, wok, z_tolerance=float(z_tolerance)))
    for index, waypoint in enumerate(placement.waypoints):
        label = waypoint.tag or normalise_resource_resref(waypoint.template_resref) or f"waypoint_{index + 1}"
        checks.append(_walkmesh_check(f"waypoint:{label}", waypoint.position, wok, z_tolerance=float(z_tolerance)))

    # Only the entry point must be walkable (the player spawns there).
    # Everything else legitimately sits off-mesh in vanilla data: mapnote
    # waypoints, cut-content markers, wall consoles, door hooks in frames —
    # plcaa alone ships a dozen such rows, so these are warnings.
    for check in checks:
        if check.ok:
            continue
        if check.label == "entry_point":
            blocking.append(check.message)
        else:
            warnings.append(check.message)
    if not checks:
        warnings.append("No gameplay placements required walkmesh validation.")
    return AuthoredGameplayWalkmeshValidation(
        ok=not blocking,
        checks=tuple(checks),
        warnings=tuple(warnings),
        blocking_issues=tuple(blocking),
    )


def _empty_gff_list() -> Any:
    from pykotor.resource.formats.gff.gff_data import GFFList

    return GFFList()


def _new_gff(content_name: str) -> Any:
    from pykotor.resource.formats.gff import GFF
    from pykotor.resource.formats.gff.gff_data import GFFContent

    return GFF(getattr(GFFContent, content_name))


def _bytes_gff(gff: Any) -> bytes:
    from pykotor.resource.formats.gff import bytes_gff

    return bytes_gff(gff)


def apply_entry_point_to_ifo(root: Any, entry: ModuleEntryPoint) -> None:
    """Write authored player start data to an IFO root struct."""

    root.set_resref("Mod_Entry_Area", entry.area_resref)
    root.set_single("Mod_Entry_X", float(entry.position[0]))
    root.set_single("Mod_Entry_Y", float(entry.position[1]))
    root.set_single("Mod_Entry_Z", float(entry.position[2]))
    root.set_single("Mod_Entry_Dir_X", math.cos(float(entry.facing)))
    root.set_single("Mod_Entry_Dir_Y", math.sin(float(entry.facing)))


def _add_placeable(
    list_value: Any,
    index: int,
    placeable: AuthoredPlaceableInstance,
    *,
    game: str = "K1",
) -> None:
    item = list_value.add(GIT_STRUCT_ID_PLACEABLE)
    resref = normalise_resource_resref(placeable.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_single("X", float(placeable.position[0]))
    item.set_single("Y", float(placeable.position[1]))
    item.set_single("Z", float(placeable.position[2]))
    item.set_single("Bearing", float(placeable.bearing))
    # Vanilla K2 emits these fields on every placeable row. K1 rows contain
    # exactly TemplateResRef/X/Y/Z/Bearing and must not receive them.
    if str(game or "K1").strip().upper() == "K2":
        item.set_uint8("UseTweakColor", int(bool(placeable.use_tweak_color)))
        item.set_uint32("TweakColor", int(placeable.tweak_color) & 0xFFFFFFFF)


def _add_creature(list_value: Any, index: int, creature: AuthoredCreatureInstance) -> None:
    item = list_value.add(GIT_STRUCT_ID_CREATURE)
    resref = normalise_resource_resref(creature.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_string("Tag", creature.tag or resref)
    item.set_single("XPosition", float(creature.position[0]))
    item.set_single("YPosition", float(creature.position[1]))
    item.set_single("ZPosition", float(creature.position[2]))
    # Vanilla creature orientation is a unit direction vector (cos/sin of the
    # facing angle) split across XOrientation/YOrientation; creatures carry no
    # scalar "Bearing" field. Writing the raw bearing into XOrientation faced
    # every spawned creature the wrong way.
    item.set_single("XOrientation", math.cos(float(creature.bearing)))
    item.set_single("YOrientation", math.sin(float(creature.bearing)))


def _add_door(
    list_value: Any,
    index: int,
    door: AuthoredDoorInstance,
    *,
    game: str = "K1",
) -> None:
    item = list_value.add(GIT_STRUCT_ID_DOOR)
    resref = normalise_resource_resref(door.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_string("Tag", door.tag or resref)
    # Match vanilla field types and order. LinkedToModule is a ResRef, not a
    # String, and LinkedToFlags is present even for an ordinary unlinked door.
    item.set_resref("LinkedToModule", normalise_resource_resref(door.linked_to_module))
    item.set_string("LinkedTo", door.linked_to)
    item.set_uint8("LinkedToFlags", int(door.linked_to_flags) & 0xFF)
    # TransitionDestin is a CExoLocString (a dialog.tlk StringRef) in vanilla,
    # not an int32; -1 means "no override" (use the linked area's name).
    _door_sref = int(door.transition_destination)
    item.set_locstring("TransitionDestin", LocalizedString(_door_sref if _door_sref > 0 else -1))
    item.set_single("X", float(door.position[0]))
    item.set_single("Y", float(door.position[1]))
    item.set_single("Z", float(door.position[2]))
    item.set_single("Bearing", float(door.bearing))
    if str(game or "K1").strip().upper() == "K2":
        item.set_uint8("UseTweakColor", int(bool(door.use_tweak_color)))
        item.set_uint32("TweakColor", int(door.tweak_color) & 0xFFFFFFFF)


def _add_trigger_geometry(list_value: Any, points: tuple[Vec3, ...]) -> None:
    for point in points:
        item = list_value.add(GIT_STRUCT_ID_TRIGGER_GEOMETRY)
        item.set_single("PointX", float(point[0]))
        item.set_single("PointY", float(point[1]))
        item.set_single("PointZ", float(point[2]))


def _add_trigger(list_value: Any, index: int, trigger: AuthoredTriggerInstance) -> None:
    item = list_value.add(GIT_STRUCT_ID_TRIGGER)
    resref = normalise_resource_resref(trigger.template_resref)
    item.set_resref("TemplateResRef", resref)
    _trig_sref = int(trigger.transition_destination)
    is_transition = bool(
        trigger.linked_to
        or normalise_resource_resref(trigger.linked_to_module)
        or int(trigger.linked_to_flags)
        or _trig_sref > 0
    )
    if is_transition:
        # Match vanilla transition-trigger field order exactly in K1 and K2.
        item.set_string("Tag", trigger.tag or resref)
        item.set_locstring("TransitionDestin", LocalizedString(_trig_sref if _trig_sref > 0 else -1))
        item.set_resref("LinkedToModule", normalise_resource_resref(trigger.linked_to_module))
        item.set_string("LinkedTo", trigger.linked_to)
        item.set_uint8("LinkedToFlags", int(trigger.linked_to_flags) & 0xFF)
    item.set_single("XPosition", float(trigger.position[0]))
    item.set_single("YPosition", float(trigger.position[1]))
    item.set_single("ZPosition", float(trigger.position[2]))
    item.set_single("XOrientation", 0.0)
    item.set_single("YOrientation", 0.0)
    item.set_single("ZOrientation", 0.0)
    geometry = _empty_gff_list()
    _add_trigger_geometry(geometry, trigger.geometry)
    item.set_list("Geometry", geometry)


def _add_waypoint(list_value: Any, index: int, waypoint: AuthoredWaypointInstance) -> None:
    item = list_value.add(GIT_STRUCT_ID_WAYPOINT)
    resref = normalise_resource_resref(waypoint.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_string("Tag", waypoint.tag or resref)
    item.set_single("XPosition", float(waypoint.position[0]))
    item.set_single("YPosition", float(waypoint.position[1]))
    item.set_single("ZPosition", float(waypoint.position[2]))
    item.set_single("XOrientation", math.cos(float(waypoint.bearing)))
    item.set_single("YOrientation", math.sin(float(waypoint.bearing)))


def _add_encounter(list_value: Any, index: int, encounter: AuthoredEncounterInstance) -> None:
    item = list_value.add(GIT_STRUCT_ID_ENCOUNTER)
    resref = normalise_resource_resref(encounter.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_string("Tag", encounter.tag or resref)
    item.set_single("XPosition", float(encounter.position[0]))
    item.set_single("YPosition", float(encounter.position[1]))
    item.set_single("ZPosition", float(encounter.position[2]))


def _add_sound(list_value: Any, index: int, sound: AuthoredSoundInstance) -> None:
    item = list_value.add(GIT_STRUCT_ID_SOUND)
    resref = normalise_resource_resref(sound.template_resref)
    item.set_resref("TemplateResRef", resref)
    item.set_string("Tag", sound.tag or resref)
    item.set_single("XPosition", float(sound.position[0]))
    item.set_single("YPosition", float(sound.position[1]))
    item.set_single("ZPosition", float(sound.position[2]))


def _add_camera(list_value: Any, index: int, camera: AuthoredCameraInstance) -> None:
    from utility.common.geometry import Vector3, Vector4

    item = list_value.add(GIT_STRUCT_ID_CAMERA)
    item.set_int32("CameraID", _camera_id_value(camera.camera_id) or 0)
    item.set_single("FieldOfView", float(camera.field_of_view))
    item.set_single("Height", float(camera.height))
    item.set_single("MicRange", float(camera.mic_range))
    item.set_vector4("Orientation", Vector4(*camera.orientation))
    item.set_vector3("Position", Vector3(*camera.position))
    item.set_single("Pitch", float(camera.pitch))


def _add_store(list_value: Any, index: int, store: AuthoredStoreInstance) -> None:
    # Engine contract (vanilla 202TEL GIT, struct 11): stores use "ResRef" —
    # not "TemplateResRef" like every other list — plus world position and an
    # orientation vector.
    item = list_value.add(GIT_STRUCT_ID_STORE)
    resref = normalise_resource_resref(store.template_resref)
    item.set_resref("ResRef", resref)
    item.set_single("XPosition", float(store.position[0]))
    item.set_single("YPosition", float(store.position[1]))
    item.set_single("ZPosition", float(store.position[2]))
    item.set_single("XOrientation", math.sin(float(store.bearing)))
    item.set_single("YOrientation", math.cos(float(store.bearing)))


def _default_area_properties() -> Any:
    """Build the GIT AreaProperties struct expected by KOTOR modules."""

    from pykotor.resource.formats.gff.gff_data import GFFStruct

    item = GFFStruct(100)
    item.set_int32("AmbientSndDay", 0)
    item.set_int32("AmbientSndNight", 0)
    item.set_int32("AmbientSndDayVol", 0)
    item.set_int32("AmbientSndNitVol", 0)
    item.set_int32("EnvAudio", 0)
    item.set_int32("MusicBattle", 0)
    item.set_int32("MusicDay", 0)
    item.set_int32("MusicNight", 0)
    item.set_int32("MusicDelay", 30000)
    return item


def build_git_gff(placement: AuthoredGameplayPlacement, *, game: str = "K1") -> Any:
    """Compile authored gameplay placements into a GIT GFF."""

    validation = validate_authored_gameplay_placement(placement)
    if not validation.ok:
        raise ValueError("; ".join(validation.blocking_issues))
    gff = _new_gff("GIT")
    root = gff.root
    root.set_uint8("UseTemplates", 1)
    root.set_struct("AreaProperties", _default_area_properties())

    cameras = _empty_gff_list()
    for index, camera in enumerate(placement.cameras):
        _add_camera(cameras, index, camera)
    root.set_list("CameraList", cameras)

    creatures = _empty_gff_list()
    for index, creature in enumerate(placement.creatures):
        _add_creature(creatures, index, creature)
    root.set_list("Creature List", creatures)

    doors = _empty_gff_list()
    for index, door in enumerate(placement.doors):
        _add_door(doors, index, door, game=game)
    root.set_list("Door List", doors)

    triggers = _empty_gff_list()
    for index, trigger in enumerate(placement.triggers):
        _add_trigger(triggers, index, trigger)
    root.set_list("TriggerList", triggers)

    encounters = _empty_gff_list()
    for index, encounter in enumerate(placement.encounters):
        _add_encounter(encounters, index, encounter)
    root.set_list("Encounter List", encounters)

    sounds = _empty_gff_list()
    for index, sound in enumerate(placement.sounds):
        _add_sound(sounds, index, sound)
    root.set_list("SoundList", sounds)

    stores = _empty_gff_list()
    for index, store in enumerate(placement.stores):
        _add_store(stores, index, store)
    root.set_list("StoreList", stores)
    root.set_list("List", _empty_gff_list())

    placeables = _empty_gff_list()
    for index, placeable in enumerate(placement.placeables):
        _add_placeable(placeables, index, placeable, game=game)
    root.set_list("Placeable List", placeables)

    waypoints = _empty_gff_list()
    for index, waypoint in enumerate(placement.waypoints):
        _add_waypoint(waypoints, index, waypoint)
    root.set_list("WaypointList", waypoints)
    return gff


def build_git_bytes(placement: AuthoredGameplayPlacement, *, game: str = "K1") -> bytes:
    """Compile authored gameplay placements into serialized GIT bytes."""

    return _bytes_gff(build_git_gff(placement, game=game))


def patch_preserved_stock_git_bytes(
    source_git_bytes: bytes,
    placement: AuthoredGameplayPlacement,
    *,
    game: str = "K1",
) -> bytes:
    """Patch authored object lists while preserving stock GIT area metadata.

    Imported modules may carry music, ambient sound, environmental audio, and
    unknown extension fields in the GIT root.  Rebuilding the entire resource
    from the editable placement subset used to reset those fields.  This
    routine replaces only the lists Map Studio owns and leaves every other
    root field intact.  Unedited imports should continue to use the original
    byte stream directly; this patch path is for deliberate placement edits.
    """

    raw = bytes(source_git_bytes or b"")
    if not raw:
        return build_git_bytes(placement, game=game)
    from pykotor.resource.formats.gff import read_gff

    source = read_gff(raw)
    candidate = build_git_gff(placement, game=game)
    for label in (
        "CameraList",
        "Creature List",
        "Door List",
        "TriggerList",
        "Encounter List",
        "SoundList",
        "StoreList",
        "List",
        "Placeable List",
        "WaypointList",
    ):
        source.root.set_list(label, candidate.root.acquire(label, _empty_gff_list()))
    # UseTemplates is owned by the placement compiler; AreaProperties and
    # every unknown source-root field are intentionally preserved.
    source.root.set_uint8("UseTemplates", int(candidate.root.acquire("UseTemplates", 1) or 0))
    return _bytes_gff(source)


__all__ = [
    "AuthoredCameraInstance",
    "AuthoredCreatureInstance",
    "AuthoredDoorInstance",
    "AuthoredEncounterInstance",
    "AuthoredGameplayPlacement",
    "AuthoredGameplayPlacementValidation",
    "AuthoredGameplayWalkmeshCheck",
    "AuthoredGameplayWalkmeshValidation",
    "AuthoredPlaceableInstance",
    "AuthoredSoundInstance",
    "AuthoredStoreInstance",
    "AuthoredTriggerInstance",
    "AuthoredWaypointInstance",
    "ModuleEntryPoint",
    "apply_entry_point_to_ifo",
    "build_git_bytes",
    "build_git_gff",
    "patch_preserved_stock_git_bytes",
    "normalise_resource_resref",
    "validate_authored_gameplay_placement",
    "validate_authored_gameplay_placement_against_walkmesh",
]
