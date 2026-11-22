"""Qt-free NWScript reference and NCS inspection services.

The definitions come from the same PyKotor tables used by the compiler.  NCS
disassembly is always treated as the authoritative representation; recovered
NWScript is marked exact only when recompiling it produces the original bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


def _game_key(value: object) -> str:
    text = str(value or "K2").strip().upper()
    if text in {"K1", "1", "KOTOR", "KOTOR1"}:
        return "K1"
    if text in {"K2", "2", "TSL", "KOTOR2"}:
        return "K2"
    raise ValueError("Target game must be K1 or K2.")


def _datatype_name(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "void").lower()


def _default_text(value: object) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return f'"{raw}"'
    if isinstance(raw, bool):
        return "TRUE" if raw else "FALSE"
    return str(raw)


_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Actions & Commands", ("action", "assigncommand", "delaycommand", "clearactions")),
    ("Objects & Tags", ("object", "tag", "nearest", "first", "next", "valid")),
    ("Location & Movement", ("location", "position", "facing", "distance", "move", "jump")),
    ("Dialogue & Cinematics", ("conversation", "dialog", "camera", "cutscene", "fade", "movie")),
    ("Quests & Globals", ("journal", "global", "plot", "quest")),
    ("Creatures & Combat", ("creature", "combat", "attack", "damage", "effect", "feat", "spell")),
    ("Inventory & Items", ("item", "inventory", "equip", "possess", "gold")),
    ("Modules & Areas", ("module", "area", "transition", "waypoint", "encounter", "trigger")),
    ("Audio & Visual", ("sound", "music", "voice", "animation", "visual", "vfx")),
    ("Math, String & Utility", ("random", "string", "intto", "floatto", "vector", "abs", "sqrt")),
)


def _function_category(name: str, description: str) -> str:
    haystack = f"{name} {description}".casefold()
    for category, needles in _CATEGORY_RULES:
        if any(needle in haystack for needle in needles):
            return category
    return "Other Engine Functions"


@dataclass(frozen=True)
class ScriptReferenceParameter:
    name: str
    datatype: str
    default: str = ""

    @property
    def signature(self) -> str:
        suffix = f" = {self.default}" if self.default else ""
        return f"{self.datatype} {self.name}{suffix}"


@dataclass(frozen=True)
class ScriptFunctionReference:
    routine_id: int
    name: str
    return_type: str
    parameters: tuple[ScriptReferenceParameter, ...]
    description: str
    category: str
    game: str

    @property
    def signature(self) -> str:
        args = ", ".join(parameter.signature for parameter in self.parameters)
        return f"{self.return_type} {self.name}({args})"


@dataclass(frozen=True)
class ScriptConstantReference:
    name: str
    datatype: str
    value: str
    game: str


@dataclass(frozen=True)
class NcsInspection:
    game: str
    resref: str
    byte_count: int
    instruction_count: int
    disassembly: str
    recovered_source: str
    exact_recompile: bool
    recompile_error: str = ""


class NWScriptReferenceService:
    """Search the compiler's K1/K2 engine-function and constant definitions."""

    @staticmethod
    @lru_cache(maxsize=2)
    def functions(game: str = "K2") -> tuple[ScriptFunctionReference, ...]:
        from pykotor.common.scriptdefs import KOTOR_FUNCTIONS, TSL_FUNCTIONS

        target = _game_key(game)
        source: Iterable[object] = KOTOR_FUNCTIONS if target == "K1" else TSL_FUNCTIONS
        rows: list[ScriptFunctionReference] = []
        for routine_id, function in enumerate(source):
            description = str(getattr(function, "description", "") or "").strip()
            parameters = tuple(
                ScriptReferenceParameter(
                    str(getattr(parameter, "name", "arg") or "arg"),
                    _datatype_name(getattr(parameter, "datatype", "")),
                    _default_text(getattr(parameter, "default", None)),
                )
                for parameter in tuple(getattr(function, "params", ()) or ())
            )
            name = str(getattr(function, "name", "") or "")
            rows.append(
                ScriptFunctionReference(
                    routine_id,
                    name,
                    _datatype_name(getattr(function, "returntype", "void")),
                    parameters,
                    description,
                    _function_category(name, description),
                    target,
                )
            )
        return tuple(rows)

    @staticmethod
    @lru_cache(maxsize=2)
    def constants(game: str = "K2") -> tuple[ScriptConstantReference, ...]:
        from pykotor.common.scriptdefs import KOTOR_CONSTANTS, TSL_CONSTANTS

        target = _game_key(game)
        source: Iterable[object] = KOTOR_CONSTANTS if target == "K1" else TSL_CONSTANTS
        return tuple(
            ScriptConstantReference(
                str(getattr(row, "name", "") or ""),
                _datatype_name(getattr(row, "datatype", "")),
                _default_text(getattr(row, "value", "")),
                target,
            )
            for row in source
        )

    @classmethod
    def search_functions(
        cls,
        query: str = "",
        *,
        game: str = "K2",
        category: str = "",
        limit: int | None = None,
    ) -> tuple[ScriptFunctionReference, ...]:
        needle = str(query or "").strip().casefold()
        category_key = str(category or "").strip().casefold()
        rows = (
            row
            for row in cls.functions(game)
            if (not category_key or row.category.casefold() == category_key)
            and (
                not needle
                or needle in row.name.casefold()
                or needle in row.signature.casefold()
                or needle in row.description.casefold()
            )
        )
        result = tuple(rows)
        return result if limit is None else result[: max(0, int(limit))]

    @classmethod
    def function(cls, name: str, *, game: str = "K2") -> ScriptFunctionReference | None:
        key = str(name or "").strip().casefold()
        return next((row for row in cls.functions(game) if row.name.casefold() == key), None)

    @classmethod
    def categories(cls, game: str = "K2") -> tuple[str, ...]:
        return tuple(sorted({row.category for row in cls.functions(game)}))


def _instruction_disassembly(ncs: object) -> str:
    instructions = tuple(getattr(ncs, "instructions", ()) or ())
    indices = {id(row): index for index, row in enumerate(instructions)}
    lines: list[str] = []
    for index, instruction in enumerate(instructions):
        opcode = str(getattr(getattr(instruction, "ins_type", None), "name", "UNKNOWN"))
        args = tuple(getattr(instruction, "args", ()) or ())
        jump = getattr(instruction, "jump", None)
        operand = f" -> {indices.get(id(jump), -1)}" if jump is not None else (f" {list(args)!r}" if args else "")
        lines.append(f"{index:05d}  {opcode:<10}{operand}")
    return "\n".join(lines) + ("\n" if lines else "")


def inspect_ncs(
    data: bytes,
    *,
    game: str,
    resref: str,
    include_dirs: Sequence[str | Path] = (),
    verify_recompile: bool = True,
) -> NcsInspection:
    """Disassemble NCS and independently test whether recovered source is exact."""

    from pykotor.common.misc import Game
    from pykotor.resource.formats.ncs import bytes_ncs, read_ncs
    from src.core.scripting.pykotor_ncs_compiler_fix import compile_nss_verified
    from pykotor.resource.formats.ncs.decompiler import NCSDecompiler

    target = _game_key(game)
    payload = bytes(data or b"")
    ncs = read_ncs(payload)
    pykotor_game = Game.K1 if target == "K1" else Game.K2
    source = str(NCSDecompiler(ncs, pykotor_game).decompile() or "")
    exact = False
    error = ""
    if source.strip() and not verify_recompile:
        error = "Exact recompile check was disabled by caller."
    elif source.strip():
        try:
            rebuilt = bytes(
                bytes_ncs(
                    compile_nss_verified(
                        source,
                        pykotor_game,
                        library_lookup=[Path(value) for value in include_dirs] or None,
                    )
                )
            )
            exact = rebuilt == payload
        except Exception as exc:  # A reconstruction is useful even when it cannot compile.
            error = str(exc).strip() or exc.__class__.__name__
    else:
        error = "The decompiler did not recover readable NWScript source."
    return NcsInspection(
        target,
        str(resref or "script").strip().lower(),
        len(payload),
        len(tuple(getattr(ncs, "instructions", ()) or ())),
        _instruction_disassembly(ncs),
        source,
        exact,
        error,
    )


__all__ = [
    "NcsInspection",
    "NWScriptReferenceService",
    "ScriptConstantReference",
    "ScriptFunctionReference",
    "ScriptReferenceParameter",
    "inspect_ncs",
]


def convert_ncs_game_abi(data: bytes, *, abi: dict, resref: str = "script"):
    """Offline K1->K2 ABI conversion; no installs or native acceptance implied."""
    from .ncs_game_abi import convert_with_ghost
    return convert_with_ghost(data, abi=abi, resref=resref)


__all__.append("convert_ncs_game_abi")
