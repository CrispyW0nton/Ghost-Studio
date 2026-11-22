"""Target-game resources used by Map Studio's manual ``plcaa`` proof pass.

The proof kit is deliberately small and transparent.  It creates target-game
UTP/UTD/UTM/UTT/UTW templates by cloning known-loadable stock structures, then
adds the scripts and item dependency needed by the terminal, container, and
ordered-switch puzzle.  The resulting rows are ordinary Map Studio palette
entries: the modder must still drag, position, configure, export, install, and
test them through GhostStudio.

This module does *not* claim engine readiness.  It only prepares a structurally
grounded candidate for the required manual ``warp plcaa`` acceptance pass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from pykotor.common.misc import Game, InventoryItem, ResRef
from pykotor.resource.formats.gff import GFFContent, bytes_gff, read_gff
from pykotor.resource.formats.ncs import bytes_ncs
from src.core.scripting.pykotor_ncs_compiler_fix import compile_nss_verified
from pykotor.resource.generics.utp import dismantle_utp, read_utp


PLCAA_PROOF_KIT_SCHEMA = "ghostrigger.map_studio.plcaa_manual_proof_kit.v1"


@dataclass(frozen=True)
class PlcaaProofPaletteEntry:
    """One manually draggable proof resource exposed in the placement tab."""

    kind: str
    template_resref: str
    label: str
    category: str
    proof_role: str
    tag: str = ""
    required_with: tuple[str, ...] = ()
    instruction: str = ""

    def as_library_row(self, game: str) -> dict[str, Any]:
        game_tag = _game_tag(game)
        restype = {
            "creature": "UTC",
            "placeable": "UTP",
            "door": "UTD",
            "trigger": "UTT",
            "waypoint": "UTW",
            "store": "UTM",
        }[self.kind]
        return {
            "game": game_tag,
            "resref": self.template_resref,
            "template_resref": self.template_resref,
            "restype": restype,
            "category": self.category,
            "subcategory": "PLCaa Manual Proof",
            "source": "plcaa_manual_proof_kit",
            "label": self.label,
            "tag": self.tag or self.template_resref,
            "metadata": {
                "schema": PLCAA_PROOF_KIT_SCHEMA,
                "proof_role": self.proof_role,
                "required_with": list(self.required_with),
                "instruction": self.instruction,
                "engine_ready": False,
                "manual_proof_required": True,
            },
        }


@dataclass(frozen=True)
class PlcaaProofKitIssue:
    severity: str
    code: str
    message: str
    resource_key: tuple[str, str] | None = None


@dataclass(frozen=True)
class PlcaaProofKitBuild:
    game: str
    resources: tuple[tuple[str, str, bytes], ...] = ()
    issues: tuple[PlcaaProofKitIssue, ...] = ()
    source_templates: dict[str, str] = field(default_factory=dict)
    engine_ready: bool = False

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "blocking" for issue in self.issues)


_PROOF_ENTRIES: tuple[PlcaaProofPaletteEntry, ...] = (
    PlcaaProofPaletteEntry(
        "creature",
        "gr_enemy",
        "Enemy (attacks player)",
        "Creatures",
        "enemy",
        instruction="Place on walkable floor and verify it becomes hostile in plcaa.",
    ),
    PlcaaProofPaletteEntry(
        "creature",
        "gr_roamnpc",
        "Friendly NPC (free roam)",
        "NPCs",
        "free_roam_npc",
        instruction="Place on a broad walkable region and verify continuous random roaming.",
    ),
    PlcaaProofPaletteEntry(
        "placeable",
        "gr_terminal",
        "Working terminal",
        "Placeables",
        "terminal",
        required_with=("gr_store",),
        instruction="Use in game; it must acknowledge use and open the proof store.",
    ),
    PlcaaProofPaletteEntry(
        "store",
        "gr_store",
        "Terminal proof store",
        "Stores",
        "terminal_store",
        required_with=("gr_terminal",),
        instruction="Place near the terminal; stores are invisible runtime resources.",
    ),
    PlcaaProofPaletteEntry(
        "placeable",
        "gr_container",
        "Working container (credits)",
        "Placeables",
        "container",
        instruction="Open it and take the bundled credits item.",
    ),
    *tuple(
        PlcaaProofPaletteEntry(
            "placeable",
            f"gr_ped{index}",
            f"Puzzle switch {index} of 3",
            "Placeables",
            f"puzzle_switch_{index}",
            required_with=("gr_ped1", "gr_ped2", "gr_ped3", "gr_pzdoor"),
            instruction="Use switches in order 1, 2, 3 to unlock the puzzle door.",
        )
        for index in (1, 2, 3)
    ),
    PlcaaProofPaletteEntry(
        "door",
        "gr_pzdoor",
        "Puzzle reward door",
        "Placeables / Animated Doors",
        "puzzle_door",
        required_with=("gr_ped1", "gr_ped2", "gr_ped3"),
        instruction="Starts locked and must open only after the 1-2-3 switch sequence.",
    ),
    PlcaaProofPaletteEntry(
        "door",
        "gr_traveldoor",
        "Configurable travel door",
        "Placeables / Animated Doors",
        "travel_door",
        required_with=("gr_destwp",),
        instruction="Set Linked To, Module, and target kind in Properties before export.",
    ),
    PlcaaProofPaletteEntry(
        "trigger",
        "gr_traveltrig",
        "Configurable travel trigger",
        "Triggers",
        "travel_trigger",
        required_with=("gr_destwp",),
        instruction="Resize/place its polygon, then set the destination in Properties.",
    ),
    PlcaaProofPaletteEntry(
        "waypoint",
        "gr_destwp",
        "Travel destination waypoint",
        "Waypoints",
        "travel_destination",
        instruction="Place at the intended arrival point and keep the tag exact.",
    ),
)


_DONORS: dict[str, dict[str, tuple[str, str]]] = {
    "K1": {
        "enemy_creature": ("dan13_juhani", "UTC"),
        "npc_creature": ("dan13_juhani", "UTC"),
        "panel": ("plc_comppnl", "UTP"),
        "container": ("plc_footlker", "UTP"),
        "door": ("dan13_door002", "UTD"),
        "store": ("dan_droid", "UTM"),
        "trigger": ("newtransition", "UTT"),
        "waypoint": ("sw_waypoint001", "UTW"),
        "item": ("g_i_credits001", "UTI"),
    },
    "K2": {
        "enemy_creature": ("g_assassindrd002", "UTC"),
        "npc_creature": ("n_commf001", "UTC"),
        "panel": ("plc_comppnl", "UTP"),
        "container": ("plc_footlker", "UTP"),
        "door": ("sw_door_per001", "UTD"),
        "store": ("m_202_001", "UTM"),
        "trigger": ("newtransition", "UTT"),
        "waypoint": ("sw_waypoint001", "UTW"),
        "item": ("g_i_credits001", "UTI"),
    },
}


_TERMINAL_NSS = """
// GhostStudio plcaa proof terminal.  The store is staged separately so the
// modder proves both placement paths through the Map Studio UI.
void main() {
    object oPC = GetFirstPC();
    SendMessageToPC(oPC, "GhostStudio plcaa terminal activated.");
    object oStore = GetObjectByTag("gr_store", 0);
    if (GetIsObjectValid(oStore)) {
        OpenStore(oStore, oPC);
    }
}
"""


_PUZZLE_NSS = """
// Ordered 1 -> 2 -> 3 switch puzzle used by the manual plcaa proof.
void main() {
    object oSelf = OBJECT_SELF;
    string sTag = GetTag(oSelf);
    object oPed1 = GetObjectByTag("gr_ped1", 0);
    object oPed2 = GetObjectByTag("gr_ped2", 0);
    object oDoor = GetObjectByTag("gr_pzdoor", 0);
    if (sTag == "gr_ped1") {
        SetLocalBoolean(oPed1, 40, TRUE);
    } else if (sTag == "gr_ped2") {
        if (GetLocalBoolean(oPed1, 40) == TRUE) {
            SetLocalBoolean(oPed2, 40, TRUE);
        } else {
            SetLocalBoolean(oPed1, 40, FALSE);
            SetLocalBoolean(oPed2, 40, FALSE);
        }
    } else if (sTag == "gr_ped3") {
        if (GetLocalBoolean(oPed2, 40) == TRUE) {
            SetLocked(oDoor, FALSE);
            AssignCommand(oDoor, ActionOpenDoor(oDoor));
        } else {
            SetLocalBoolean(oPed1, 40, FALSE);
            SetLocalBoolean(oPed2, 40, FALSE);
        }
    }
}
"""


def _game_tag(game: str) -> str:
    value = str(game or "").strip().upper()
    if value not in _DONORS:
        raise ValueError("PLCaa manual proof assets require target game K1 or K2.")
    return value


def _pykotor_game(game: str) -> Game:
    return Game.K1 if _game_tag(game) == "K1" else Game.K2


def plcaa_manual_proof_palette_rows(game: str) -> tuple[dict[str, Any], ...]:
    """Return ordinary library rows which the modder drags into ``plcaa``."""

    return tuple(entry.as_library_row(game) for entry in _PROOF_ENTRIES)


def plcaa_manual_proof_required_tags() -> tuple[str, ...]:
    return tuple(entry.tag or entry.template_resref for entry in _PROOF_ENTRIES)


def _read(reader: Callable[[str, str], bytes], donor: tuple[str, str]) -> bytes:
    resref, restype = donor
    data = reader(resref, restype)
    if not data:
        raise FileNotFoundError(f"Required target-game donor {resref}.{restype.lower()} was not found.")
    return bytes(data)


def _clone_identity(data: bytes, *, expected: GFFContent, resref: str, tag: str, resref_label: str = "TemplateResRef") -> Any:
    gff = read_gff(bytes(data))
    if gff.content != expected:
        raise ValueError(f"Proof-kit donor for {resref} is {gff.content}, expected {expected}.")
    cloned = deepcopy(gff)
    cloned.root.set_resref(resref_label, ResRef(resref))
    cloned.root.set_string("Tag", tag)
    return cloned


def _merge_known_fields(base_bytes: bytes, patch_gff: Any) -> bytes:
    base = read_gff(base_bytes)
    output = deepcopy(base)
    for label in patch_gff.root.keys():
        output.root._fields[label] = deepcopy(patch_gff.root._fields[label])
    return bytes_gff(output)


def _build_utp(
    base: bytes,
    *,
    game: str,
    resref: str,
    tag: str,
    on_used: str = "",
    inventory: Iterable[str] = (),
) -> bytes:
    utp = read_utp(base)
    utp.resref = ResRef(resref)
    utp.tag = tag
    utp.static = False
    utp.useable = True
    utp.plot = True
    if on_used:
        utp.on_used = ResRef(on_used)
    items = tuple(str(item).strip().lower() for item in inventory if str(item).strip())
    if items:
        utp.has_inventory = True
        utp.inventory = [InventoryItem(ResRef(item)) for item in items]
    return _merge_known_fields(base, dismantle_utp(utp, _pykotor_game(game)))


def _compile_script(source: str, game: str) -> bytes:
    return bytes_ncs(compile_nss_verified(source, _pykotor_game(game)))


def build_plcaa_manual_proof_kit(
    game: str,
    resource_reader: Callable[[str, str], bytes],
) -> PlcaaProofKitBuild:
    """Build non-creature proof resources from target-game vanilla donors.

    Creature templates are delegated to ``authored_creature_behavior`` because
    faction and free-roam behavior are UTC concerns.  Everything is returned
    in one collision-checked resource set for the final module transaction.
    """

    game_tag = _game_tag(game)
    donors = _DONORS[game_tag]
    resources: list[tuple[str, str, bytes]] = []
    issues: list[PlcaaProofKitIssue] = []
    source_templates = {name: f"{value[0]}.{value[1].lower()}" for name, value in donors.items()}
    try:
        from src.core.modules.authored_creature_behavior import build_authored_creature_behavior_resources

        enemy_source = _read(resource_reader, donors["enemy_creature"])
        npc_source = _read(resource_reader, donors["npc_creature"])
        enemy = build_authored_creature_behavior_resources(
            enemy_source,
            game=game_tag,
            template_resref="gr_enemy",
            faction_role="hostile",
            conversation_resref="",
            movement_mode="stationary",
        )
        npc = build_authored_creature_behavior_resources(
            npc_source,
            game=game_tag,
            template_resref="gr_roamnpc",
            faction_role="friendly",
            conversation_resref="",
            movement_mode="free_roam",
        )
        resources.extend(enemy.resources)
        resources.extend(npc.resources)

        panel = _read(resource_reader, donors["panel"])
        container = _read(resource_reader, donors["container"])
        door = _read(resource_reader, donors["door"])
        store = _read(resource_reader, donors["store"])
        trigger = _read(resource_reader, donors["trigger"])
        waypoint = _read(resource_reader, donors["waypoint"])
        item = _read(resource_reader, donors["item"])

        resources.append(("gr_terminal", "utp", _build_utp(panel, game=game_tag, resref="gr_terminal", tag="gr_terminal", on_used="gr_terminal")))
        resources.append(("gr_container", "utp", _build_utp(container, game=game_tag, resref="gr_container", tag="gr_container", inventory=("g_i_credits001",))))
        for index in (1, 2, 3):
            resref = f"gr_ped{index}"
            resources.append((resref, "utp", _build_utp(panel, game=game_tag, resref=resref, tag=resref, on_used="gr_puzzle")))

        puzzle_door = _clone_identity(door, expected=GFFContent.UTD, resref="gr_pzdoor", tag="gr_pzdoor")
        puzzle_door.root.set_uint8("Locked", 1)
        puzzle_door.root.set_uint8("KeyRequired", 0)
        puzzle_door.root.set_uint8("Plot", 1)
        puzzle_door.root.set_uint8("Static", 0)
        resources.append(("gr_pzdoor", "utd", bytes_gff(puzzle_door)))

        travel_door = _clone_identity(door, expected=GFFContent.UTD, resref="gr_traveldoor", tag="gr_traveldoor")
        travel_door.root.set_uint8("Locked", 0)
        travel_door.root.set_uint8("Static", 0)
        resources.append(("gr_traveldoor", "utd", bytes_gff(travel_door)))

        proof_store = _clone_identity(store, expected=GFFContent.UTM, resref="gr_store", tag="gr_store", resref_label="ResRef")
        resources.append(("gr_store", "utm", bytes_gff(proof_store)))
        proof_trigger = _clone_identity(trigger, expected=GFFContent.UTT, resref="gr_traveltrig", tag="gr_traveltrig")
        resources.append(("gr_traveltrig", "utt", bytes_gff(proof_trigger)))
        proof_waypoint = _clone_identity(waypoint, expected=GFFContent.UTW, resref="gr_destwp", tag="gr_destwp")
        resources.append(("gr_destwp", "utw", bytes_gff(proof_waypoint)))
        resources.append(("g_i_credits001", "uti", item))

        terminal_ncs = _compile_script(_TERMINAL_NSS, game_tag)
        puzzle_ncs = _compile_script(_PUZZLE_NSS, game_tag)
        resources.extend(
            (
                ("gr_terminal", "nss", _TERMINAL_NSS.encode("utf-8")),
                ("gr_terminal", "ncs", terminal_ncs),
                ("gr_puzzle", "nss", _PUZZLE_NSS.encode("utf-8")),
                ("gr_puzzle", "ncs", puzzle_ncs),
            )
        )
    except Exception as exc:
        issues.append(
            PlcaaProofKitIssue(
                "blocking",
                "plcaa_proof_kit_build_failed",
                str(exc),
            )
        )
        return PlcaaProofKitBuild(
            game=game_tag,
            issues=tuple(issues),
            source_templates=source_templates,
        )

    by_key: dict[tuple[str, str], bytes] = {}
    for resref, restype, data in resources:
        key = (resref.lower(), restype.lower().lstrip("."))
        prior = by_key.get(key)
        if prior is not None and prior != data:
            issues.append(
                PlcaaProofKitIssue(
                    "blocking",
                    "plcaa_proof_resource_collision",
                    f"Proof kit generated conflicting bytes for {key[0]}.{key[1]}.",
                    key,
                )
            )
        by_key[key] = data
    ordered = tuple((resref, restype, data) for (resref, restype), data in sorted(by_key.items()))
    return PlcaaProofKitBuild(
        game=game_tag,
        resources=ordered,
        issues=tuple(issues),
        source_templates=source_templates,
        engine_ready=False,
    )


def build_plcaa_manual_proof_kit_from_provider(game: str, provider: Any) -> PlcaaProofKitBuild:
    """Resolve vanilla donors through GhostRigger's target-game provider."""

    if provider is None or not callable(getattr(provider, "read_bytes", None)):
        return PlcaaProofKitBuild(
            game=_game_tag(game),
            issues=(
                PlcaaProofKitIssue(
                    "blocking",
                    "plcaa_proof_provider_missing",
                    "Connect the target KOTOR installation before enabling the plcaa manual proof assets.",
                ),
            ),
        )
    from src.core.resources.game_resource_provider import GameResourceQuery

    def reader(resref: str, restype: str) -> bytes:
        return bytes(
            provider.read_bytes(
                GameResourceQuery(
                    game=_game_tag(game),
                    resref=str(resref).lower(),
                    restype=str(restype).upper(),
                )
            )
        )

    return build_plcaa_manual_proof_kit(game, reader)


def plcaa_manual_proof_in_memory_provider(build: PlcaaProofKitBuild) -> Any:
    """Expose generated candidates through the normal resource-provider API."""

    from src.core.project.resource_address import ResourceAddress
    from src.core.resources.game_resource_provider import (
        GameResourceRecord,
        InMemoryGameResourceProvider,
    )

    rows = []
    for resref, restype, data in tuple(build.resources or ()):
        address = ResourceAddress(
            scheme="generated",
            game=build.game,
            module_id="plcaa",
            resref=resref,
            restype=str(restype).upper(),
            layer="generated",
        )
        rows.append(
            (
                GameResourceRecord(
                    address=address,
                    size=len(data),
                    source="plcaa_manual_proof_kit",
                    priority=120,
                    metadata={
                        "schema": PLCAA_PROOF_KIT_SCHEMA,
                        "manual_proof_required": True,
                    },
                ),
                data,
            )
        )
    return InMemoryGameResourceProvider(rows)


__all__ = [
    "PLCAA_PROOF_KIT_SCHEMA",
    "PlcaaProofKitBuild",
    "PlcaaProofKitIssue",
    "PlcaaProofPaletteEntry",
    "build_plcaa_manual_proof_kit",
    "build_plcaa_manual_proof_kit_from_provider",
    "plcaa_manual_proof_palette_rows",
    "plcaa_manual_proof_in_memory_provider",
    "plcaa_manual_proof_required_tags",
]
