"""Clean-room NCS virtual machine for Play-in-Editor scripting.

Executes compiled NWScript (NCS) with a real stack machine instead of pattern
matching, so module scripts run with genuine control flow: branches, loops,
subroutines, globals via SAVEBP/BP addressing, and action closures.

Evidence basis (clean-room, read-only):
- Bytecode operand layouts follow the retail NCS format as round-tripped by
  PyKotor's binary reader/writer (CPDOWNSP/CPTOPSP ``[byte_offset, byte_size]``,
  ACTION ``[routine, arg_count]``, STORE_STATE ``[bp_bytes, sp_bytes]``,
  DESTRUCT ``[total, keep_offset, keep_size]``; the stack is 4-byte slots).
- Ghidra decompilation of the K1 engine (Odyssey repository,
  ``k1_win_gog_swkotor.exe``): ``CSWVirtualMachineCommands::ExecuteCommand``
  dispatches ``this->commands[command_id]`` (0x304-entry function-pointer table,
  one implementation per nwscript routine), and action-type arguments are popped
  through ``CVirtualMachine::StackPopCommand`` then queued on the caller object
  (``CSWSObject::AddDoCommandAction``). This VM mirrors both contracts: routines
  dispatch by table index via PyKotor's ``KOTOR_FUNCTIONS``/``TSL_FUNCTIONS``
  signatures, and ``action`` parameters consume the pending STORE_STATE closure.

PIE is an editor-side simulation: engine routines the sandbox cannot faithfully
model return typed defaults and are *census-tracked* (never guessed silently).
Retail KOTOR remains the sole in-game authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from typing import Any, Callable

# KOTOR nwscript.nss constants: OBJECT_SELF compiles to CONSTO 0 and
# OBJECT_INVALID to 1 (the engine additionally uses 0x7f000000 internally).
OBJECT_SELF = 0
OBJECT_INVALID = 1
_ENGINE_INVALID = 0x7F000000

_DEFAULT_INSTRUCTION_BUDGET = 250_000
_ASCII_CASE_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


class NCSExecutionError(RuntimeError):
    """Raised when the bytecode does something the VM cannot model safely."""


@dataclass(frozen=True)
class NCSSavedState:
    """A STORE_STATE closure: resume point plus the copied stack.

    Ghidra (K1 ``CVirtualMachineInternal::RunScriptSituation`` @005d4ad0): the
    engine resumes a saved command by clearing the live stack, copying the saved
    script's stack wholesale (``CVirtualMachineStack::CopyFromStack``), zeroing
    the call stack, and running from the saved instruction pointer with the
    bound object installed on the command implementer. This dataclass carries
    exactly that: the resume index, the full stack snapshot, the BP register,
    and the object the situation runs as.
    """

    resume_index: int
    stack_snapshot: tuple[Any, ...]
    bp: int
    bound_object_id: int = OBJECT_SELF


@dataclass
class NCSScriptedCommand:
    """One engine action recorded for the PIE timeline (cinematics feed)."""

    kind: str                     # routine name, e.g. "ActionPlayAnimation"
    object_id: int                # the object the action queues on
    object_tag: str = ""
    args: tuple[Any, ...] = ()
    delay_seconds: float = 0.0    # >0 when scheduled through DelayCommand
    saved_state: NCSSavedState | None = None


@dataclass
class NCSExecutionResult:
    """Observable effects of one script run — PIE's parity surface."""

    completed: bool = False
    instructions_executed: int = 0
    global_numbers: dict[str, int] = field(default_factory=dict)
    global_booleans: dict[str, bool] = field(default_factory=dict)
    global_strings: dict[str, str] = field(default_factory=dict)
    local_booleans: dict[tuple[str, int], bool] = field(default_factory=dict)
    local_numbers: dict[tuple[str, int], int] = field(default_factory=dict)
    journal_entries: list[tuple[str, int]] = field(default_factory=list)
    commands: list[NCSScriptedCommand] = field(default_factory=list)
    unknown_routines: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class MapStudioPIEScriptContext:
    """Engine-routine sandbox the VM dispatches into.

    Handlers are keyed by nwscript function *name* (resolved through the
    routine-id → signature table), so K1 and K2 share one handler set. Routines
    without a handler return typed defaults and are census-tracked in the
    result — bounded fidelity, never silent guessing.
    """

    def __init__(
        self,
        *,
        game: str = "K2",
        self_object_id: int = OBJECT_SELF,
        object_by_tag: Callable[[str, int], int | None] | None = None,
        tag_of_object: Callable[[int], str] | None = None,
        global_numbers: dict[str, int] | None = None,
        global_booleans: dict[str, bool] | None = None,
        global_strings: dict[str, str] | None = None,
        random_seed: int = 0x5EED,
    ) -> None:
        self.game = str(game or "K2").strip().upper()
        self.self_object_id = int(self_object_id)
        self._object_by_tag = object_by_tag
        self._tag_of_object = tag_of_object
        self.result = NCSExecutionResult(
            global_numbers=dict(global_numbers or {}),
            global_booleans=dict(global_booleans or {}),
            global_strings=dict(global_strings or {}),
        )
        # Deterministic PIE dice: xorshift so re-running a preview is stable.
        self._rng_state = (int(random_seed) & 0xFFFFFFFF) or 1
        # Synthetic ids handed to GetObjectByTag when no live registry resolves
        # the tag: keeps object identity/equality faithful inside the script.
        self._synthetic_ids: dict[tuple[str, int], int] = {}
        self._next_synthetic_id = 0x0100_0000
        self._pending_delay: float = 0.0
        # Installed by the VM so AssignCommand can run its closure inline (the
        # engine queues it as a delay-0 AI event on the target and executes the
        # situation with OBJECT_SELF = target).
        self.vm: Any = None
        self.situation_depth = 0

    # -- object identity ----------------------------------------------------
    def resolve_object(self, object_id: int) -> int:
        value = int(object_id)
        if value == OBJECT_SELF:
            return self.self_object_id
        return value

    def object_tag(self, object_id: int) -> str:
        value = self.resolve_object(object_id)
        if callable(self._tag_of_object):
            tag = self._tag_of_object(value)
            if tag:
                return str(tag).strip().lower()
        for (tag, nth), synthetic in self._synthetic_ids.items():
            if synthetic == value:
                return tag
        return ""

    def object_by_tag(self, tag: str, nth: int) -> int:
        clean = str(tag or "").strip().lower()
        if not clean:
            return OBJECT_INVALID
        if callable(self._object_by_tag):
            resolved = self._object_by_tag(clean, int(nth))
            if resolved is not None:
                return int(resolved)
        key = (clean, int(nth))
        if key not in self._synthetic_ids:
            self._synthetic_ids[key] = self._next_synthetic_id
            self._next_synthetic_id += 1
        return self._synthetic_ids[key]

    def _random(self, bound: int) -> int:
        state = self._rng_state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        self._rng_state = state
        return state % bound if bound > 0 else 0

    # -- routine dispatch ---------------------------------------------------
    def call(self, name: str, params: list[Any], saved_state: NCSSavedState | None):
        handler = getattr(self, f"_routine_{name}", None)
        if callable(handler):
            return handler(params, saved_state)
        self.result.unknown_routines[name] = self.result.unknown_routines.get(name, 0) + 1
        return NotImplemented

    def _record_command(self, kind: str, object_id: int, args: tuple[Any, ...],
                        saved_state: NCSSavedState | None = None) -> None:
        target = self.resolve_object(object_id)
        self.result.commands.append(
            NCSScriptedCommand(
                kind=kind,
                object_id=target,
                object_tag=self.object_tag(target),
                args=args,
                delay_seconds=max(0.0, float(self._pending_delay)),
                saved_state=saved_state,
            )
        )
        self._pending_delay = 0.0

    # Globals / locals — the persistent-state core (routine ids 578-581,
    # 160/194, 679-682, 367 per the engine function table).
    def _routine_SetGlobalNumber(self, params, _state):
        # Ghidra: CSWGlobalVariableTable::SetValueNumber stores `(byte)value`
        # (ExecuteCommandSetGlobalNumber @00542b60) — retail global numbers
        # truncate to 8 bits, so PIE must too or comparisons drift at 256.
        self.result.global_numbers[str(params[0])] = int(params[1]) & 0xFF

    def _routine_GetGlobalNumber(self, params, _state):
        return int(self.result.global_numbers.get(str(params[0]), 0))

    def _routine_SetGlobalBoolean(self, params, _state):
        self.result.global_booleans[str(params[0])] = bool(params[1])

    def _routine_GetGlobalBoolean(self, params, _state):
        return 1 if self.result.global_booleans.get(str(params[0]), False) else 0

    def _routine_SetGlobalString(self, params, _state):
        self.result.global_strings[str(params[0])] = str(params[1])

    def _routine_GetGlobalString(self, params, _state):
        return str(self.result.global_strings.get(str(params[0]), ""))

    def _routine_SetLocalBoolean(self, params, _state):
        key = (self.object_tag(int(params[0])), int(params[1]))
        self.result.local_booleans[key] = bool(params[2])

    def _routine_GetLocalBoolean(self, params, _state):
        key = (self.object_tag(int(params[0])), int(params[1]))
        return 1 if self.result.local_booleans.get(key, False) else 0

    def _routine_SetLocalNumber(self, params, _state):
        key = (self.object_tag(int(params[0])), int(params[1]))
        self.result.local_numbers[key] = int(params[2])

    def _routine_GetLocalNumber(self, params, _state):
        key = (self.object_tag(int(params[0])), int(params[1]))
        return int(self.result.local_numbers.get(key, 0))

    def _routine_AddJournalQuestEntry(self, params, _state):
        self.result.journal_entries.append((str(params[0]), int(params[1])))

    # Object lookup / identity.
    def _routine_GetObjectByTag(self, params, _state):
        nth = int(params[1]) if len(params) > 1 else 0
        return self.object_by_tag(str(params[0]), nth)

    def _routine_GetTag(self, params, _state):
        return self.object_tag(int(params[0]))

    def _routine_GetIsObjectValid(self, params, _state):
        value = int(params[0])
        return 0 if value in (OBJECT_INVALID, _ENGINE_INVALID) else 1

    def _routine_GetEnteringObject(self, params, _state):
        return self.self_object_id  # OnEnter previews treat the player as enterer

    def _routine_GetFirstPC(self, params, _state):
        return self.self_object_id

    def _routine_GetModule(self, params, _state):
        return self.self_object_id

    def _routine_GetArea(self, params, _state):
        return self.self_object_id

    def _routine_GetPartyMemberByIndex(self, params, _state):
        return self.self_object_id if int(params[0]) == 0 else OBJECT_INVALID

    def _routine_GetIsPC(self, params, _state):
        return 1 if self.resolve_object(int(params[0])) == self.self_object_id else 0

    # Deterministic dice.
    def _routine_Random(self, params, _state):
        return self._random(int(params[0]))

    def _dice(self, count: int, sides: int) -> int:
        rolls = max(1, int(count))
        return sum(1 + self._random(sides) for _ in range(rolls))

    def _routine_d2(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 2)

    def _routine_d4(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 4)

    def _routine_d6(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 6)

    def _routine_d8(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 8)

    def _routine_d10(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 10)

    def _routine_d20(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 20)

    def _routine_d100(self, params, _state):
        return self._dice(int(params[0]) if params else 1, 100)

    # Math/string conveniences the OnEnter corpus leans on.
    def _routine_IntToString(self, params, _state):
        return str(int(params[0]))

    def _routine_StringToInt(self, params, _state):
        try:
            return int(str(params[0]).strip() or "0")
        except ValueError:
            return 0

    def _routine_IntToFloat(self, params, _state):
        return float(int(params[0]))

    def _routine_FloatToInt(self, params, _state):
        return int(float(params[0]))

    def _routine_GetStringLength(self, params, _state):
        return len(str(params[0]))

    def _routine_GetStringLowerCase(self, params, _state):
        return str(params[0]).lower()

    def _routine_GetStringUpperCase(self, params, _state):
        return str(params[0]).upper()

    def _routine_PrintString(self, params, _state):
        self.result.warnings.append(f"PrintString: {params[0]}")

    # Command scheduling — the cinematic/scripted-event feed. Ghidra contract:
    # AssignCommand pops object then command and queues the closure as a
    # delay-0 AI event on the TARGET (ExecuteCommandAssignCommand @0052e720);
    # DelayCommand pops float then command and queues on the caller at
    # +seconds (ExecuteCommandDelayCommand @0052fe30). A queued closure later
    # executes via RunScriptSituation with OBJECT_SELF = the bound object.
    def _routine_AssignCommand(self, params, state):
        if state is None:
            self.result.warnings.append("AssignCommand without a saved action state")
            return None
        target = self.resolve_object(int(params[0]))
        if self.vm is not None and self.situation_depth < 8:
            self.vm.run_saved_state(state, self_object_id=target)
        else:
            self._record_command("AssignCommand", target, (), state)
        return None

    def _routine_DelayCommand(self, params, state):
        if state is None:
            self.result.warnings.append("DelayCommand without a saved action state")
            return None
        delay = max(0.0, float(params[0]))
        self.result.commands.append(
            NCSScriptedCommand(
                kind="DelayCommand",
                object_id=self.self_object_id,
                object_tag=self.object_tag(self.self_object_id),
                args=(delay,),
                delay_seconds=delay,
                saved_state=state,
            )
        )
        return None

    def _routine_ActionDoCommand(self, params, state):
        if state is not None:
            self._record_command("ActionDoCommand", OBJECT_SELF, (), state)
        return None

    def _routine_ExecuteScript(self, params, _state):
        target = self.resolve_object(int(params[1])) if len(params) > 1 else self.self_object_id
        self._record_command("ExecuteScript", target, (str(params[0]),))

    # Actions recorded verbatim for the PIE timeline, bound to the executing
    # frame's OBJECT_SELF (situations already run with self = assign target).
    def _action_target(self) -> int:
        return self.self_object_id

    def _routine_ActionPlayAnimation(self, params, _state):
        self._record_command("ActionPlayAnimation", self._action_target(), tuple(params))

    def _routine_PlayAnimation(self, params, _state):
        self._record_command("PlayAnimation", self._action_target(), tuple(params))

    def _routine_ActionStartConversation(self, params, _state):
        self._record_command("ActionStartConversation", self._action_target(), tuple(params))

    def _routine_ActionMoveToObject(self, params, _state):
        self._record_command("ActionMoveToObject", self._action_target(), tuple(params))

    def _routine_ActionMoveToLocation(self, params, _state):
        self._record_command("ActionMoveToLocation", self._action_target(), tuple(params))

    def _routine_ActionOpenDoor(self, params, _state):
        self._record_command("ActionOpenDoor", self._action_target(), tuple(params))

    def _routine_ActionCloseDoor(self, params, _state):
        self._record_command("ActionCloseDoor", self._action_target(), tuple(params))

    def _routine_ActionWait(self, params, _state):
        self._record_command("ActionWait", self._action_target(), tuple(params))

    def _routine_ActionForceMoveToObject(self, params, _state):
        self._record_command("ActionForceMoveToObject", self._action_target(), tuple(params))

    def _routine_DestroyObject(self, params, _state):
        # Ghidra ExecuteCommandDestroyObject @0052ff20: (oDestroy, fDelay,
        # bNoFade, fDelayUntilFade, nHideFeedback) — delay schedules the kill.
        delay = float(params[1]) if len(params) > 1 and params[1] is not None else 0.0
        target = self.resolve_object(int(params[0]))
        self.result.commands.append(
            NCSScriptedCommand(
                kind="DestroyObject",
                object_id=target,
                object_tag=self.object_tag(target),
                args=tuple(p for p in params if p is not None),
                delay_seconds=max(0.0, delay),
            )
        )

    def _routine_ActionPauseConversation(self, params, _state):
        # Retail cutscene gate: dialogue holds while queued actions play out
        # (ExecuteCommandActionPauseConversation @0052d330).
        self._record_command("ActionPauseConversation", self._action_target(), ())

    def _routine_ActionResumeConversation(self, params, _state):
        self._record_command("ActionResumeConversation", self._action_target(), ())

    def _routine_SetLocked(self, params, _state):
        self._record_command("SetLocked", int(params[0]), tuple(params[1:]))

    def _routine_MusicBackgroundPlay(self, params, _state):
        self._record_command("MusicBackgroundPlay", int(params[0]), tuple(params[1:]))

    def _routine_MusicBackgroundChangeDay(self, params, _state):
        self._record_command("MusicBackgroundChangeDay", int(params[0]), tuple(params[1:]))

    def _routine_MusicBackgroundChangeNight(self, params, _state):
        self._record_command("MusicBackgroundChangeNight", int(params[0]), tuple(params[1:]))


def _function_table(game: str):
    from pykotor.common.scriptdefs import KOTOR_FUNCTIONS, TSL_FUNCTIONS

    return TSL_FUNCTIONS if str(game or "").strip().upper() == "K2" else KOTOR_FUNCTIONS


def _signed32(value: Any) -> int:
    number = int(value) & 0xFFFFFFFF
    return number - 0x1_0000_0000 if number > 0x7FFF_FFFF else number


class NCSVirtualMachine:
    """Stack-machine interpreter over PyKotor-decoded NCS instructions."""

    def __init__(
        self,
        instructions: list[Any],
        context: MapStudioPIEScriptContext,
        *,
        instruction_budget: int = _DEFAULT_INSTRUCTION_BUDGET,
    ) -> None:
        self.instructions = list(instructions or ())
        self.context = context
        self.instruction_budget = max(1, int(instruction_budget))
        self._index_of: dict[int, int] = {id(ins): i for i, ins in enumerate(self.instructions)}
        self._table = _function_table(context.game)
        # The runtime stack is 4-byte slots; strings/objects are one slot each.
        self.stack: list[Any] = []
        self.bp = 0
        self._bp_stack: list[int] = []
        self._return_stack: list[int] = []
        self._pending_state: NCSSavedState | None = None

    # -- slot helpers (byte offsets → slot indices) -------------------------
    def _slot(self, byte_offset: int, *, base: str) -> int:
        if byte_offset % 4 != 0:
            raise NCSExecutionError(f"unaligned stack offset {byte_offset}")
        slots = byte_offset // 4
        origin = len(self.stack) if base == "sp" else self.bp
        index = origin + slots
        if not 0 <= index <= len(self.stack):
            raise NCSExecutionError(f"stack offset out of range: {byte_offset} ({base})")
        return index

    def _pop(self) -> Any:
        if not self.stack:
            raise NCSExecutionError("pop from empty stack")
        return self.stack.pop()

    def run(self, *, entry_index: int = 0) -> NCSExecutionResult:
        self.context.vm = self
        result = self.context.result
        try:
            self._run_from(entry_index)
            result.completed = True
        except NCSExecutionError as exc:
            result.warnings.append(f"VM stopped: {exc}")
        return result

    def run_saved_state(self, state: NCSSavedState, *, self_object_id: int | None = None) -> None:
        """Execute a STORE_STATE closure the way RunScriptSituation does.

        The engine clears the live stack, restores the saved copy, zeroes the
        call stack, installs the bound object as OBJECT_SELF, and runs from the
        saved instruction pointer. Effects land in the shared context result.
        """

        saved = (
            list(self.stack), self.bp, list(self._bp_stack),
            list(self._return_stack), self._pending_state, self.context.self_object_id,
        )
        self.stack = list(state.stack_snapshot)
        self.bp = int(state.bp)
        self._bp_stack = []
        self._return_stack = []
        self._pending_state = None
        if self_object_id is not None:
            self.context.self_object_id = int(self_object_id)
        self.context.situation_depth += 1
        try:
            self._run_from(int(state.resume_index))
        except NCSExecutionError as exc:
            self.context.result.warnings.append(f"situation stopped: {exc}")
        finally:
            self.context.situation_depth -= 1
            (self.stack, self.bp, self._bp_stack,
             self._return_stack, self._pending_state, self.context.self_object_id) = saved

    def _jump_index(self, instruction: Any) -> int:
        target = getattr(instruction, "jump", None)
        if target is None:
            raise NCSExecutionError("jump instruction without a resolved target")
        index = self._index_of.get(id(target))
        if index is None:
            raise NCSExecutionError("jump target is not in the instruction list")
        return index

    def _run_from(self, entry_index: int) -> None:
        result = self.context.result
        pointer = entry_index
        top_frame_depth = len(self._return_stack)
        while 0 <= pointer < len(self.instructions):
            if result.instructions_executed >= self.instruction_budget:
                raise NCSExecutionError(
                    f"instruction budget exhausted ({self.instruction_budget})"
                )
            result.instructions_executed += 1
            instruction = self.instructions[pointer]
            name = str(getattr(getattr(instruction, "ins_type", None), "name", "") or "")
            args = list(getattr(instruction, "args", ()) or ())
            next_pointer = pointer + 1

            if name in ("NOP", "RESERVED", "RESERVED_01"):
                pass
            elif name.startswith("RSADD"):
                self.stack.append("" if name == "RSADDS" else 0)
            elif name == "CONSTI":
                self.stack.append(_signed32(args[0]))
            elif name == "CONSTF":
                self.stack.append(float(args[0]))
            elif name == "CONSTS":
                self.stack.append(str(args[0]))
            elif name == "CONSTO":
                value = _signed32(args[0])
                # Bind the VM's self token before comparing or capturing the object.
                self.stack.append(self.context.resolve_object(value))
            elif name in ("CPDOWNSP", "CPDOWNBP"):
                offset, size = _signed32(args[0]), int(args[1])
                count = size // 4
                base = "sp" if name.endswith("SP") else "bp"
                start = self._slot(offset, base=base)
                source = self.stack[len(self.stack) - count:]
                for position, value in enumerate(source):
                    if start + position < len(self.stack):
                        self.stack[start + position] = value
            elif name in ("CPTOPSP", "CPTOPBP"):
                offset, size = _signed32(args[0]), int(args[1])
                count = size // 4
                base = "sp" if name.endswith("SP") else "bp"
                start = self._slot(offset, base=base)
                self.stack.extend(self.stack[start:start + count])
            elif name == "MOVSP":
                offset = _signed32(args[0])
                if offset % 4 != 0:
                    raise NCSExecutionError(f"unaligned MOVSP {offset}")
                for _ in range(-offset // 4):
                    self._pop()
            elif name in ("INCxSP", "INCxBP", "DECxSP", "DECxBP"):
                offset = _signed32(args[0])
                base = "sp" if name.endswith("SP") else "bp"
                index = self._slot(offset, base=base)
                delta = 1 if name.startswith("INC") else -1
                self.stack[index] = int(self.stack[index]) + delta
            elif name == "SAVEBP":
                self._bp_stack.append(self.bp)
                self.bp = len(self.stack)
                self.stack.append(self._bp_stack[-1])
            elif name == "RESTOREBP":
                if not self._bp_stack:
                    raise NCSExecutionError("RESTOREBP without SAVEBP")
                self._pop()
                self.bp = self._bp_stack.pop()
            elif name == "JMP":
                next_pointer = self._jump_index(instruction)
            elif name == "JSR":
                self._return_stack.append(next_pointer)
                next_pointer = self._jump_index(instruction)
            elif name == "JZ":
                if int(self._pop()) == 0:
                    next_pointer = self._jump_index(instruction)
            elif name == "JNZ":
                if int(self._pop()) != 0:
                    next_pointer = self._jump_index(instruction)
            elif name == "RETN":
                if len(self._return_stack) > top_frame_depth:
                    next_pointer = self._return_stack.pop()
                else:
                    return
            elif name == "STORE_STATE":
                # Compiler shape: STORE_STATE, JMP past_block, <block>, past:.
                # RunScriptSituation copies the saved stack wholesale, so the
                # snapshot is the full stack + BP, not just the sized slices.
                self._pending_state = NCSSavedState(
                    resume_index=pointer + 2,
                    stack_snapshot=tuple(self.stack),
                    bp=self.bp,
                    bound_object_id=self.context.self_object_id,
                )
            elif name == "ACTION":
                self._execute_action(int(args[0]), int(args[1]))
            elif name == "DESTRUCT":
                total, keep_offset, keep_size = int(args[0]), _signed32(args[1]), int(args[2])
                slots = total // 4
                removed = self.stack[len(self.stack) - slots:]
                del self.stack[len(self.stack) - slots:]
                keep_start = keep_offset // 4
                self.stack.extend(removed[keep_start:keep_start + keep_size // 4])
            elif name in ("EQUALTT", "NEQUALTT"):
                size_slots = int(args[0]) // 4 if args else 0
                right_struct = [self._pop() for _ in range(size_slots)]
                left_struct = [self._pop() for _ in range(size_slots)]
                equal = list(reversed(left_struct)) == list(reversed(right_struct))
                self.stack.append(int(equal if name == "EQUALTT" else not equal))
            elif name == "NOTI":
                self.stack.append(0 if int(self._pop()) != 0 else 1)
            elif name == "COMPI":
                self.stack.append(~int(self._pop()) & 0xFFFFFFFF)
            elif name == "NEGI":
                self.stack.append(-int(self._pop()))
            elif name == "NEGF":
                self.stack.append(-float(self._pop()))
            else:
                handled = self._binary_operation(name)
                if not handled:
                    raise NCSExecutionError(f"unsupported opcode {name or '<unknown>'}")
            pointer = next_pointer

    # -- binary/comparison/vector ops ---------------------------------------
    def _binary_operation(self, name: str) -> bool:
        def pop2():
            right = self._pop()
            left = self._pop()
            return left, right

        if name in ("ADDII", "ADDIF", "ADDFI", "ADDFF"):
            left, right = pop2()
            value = left + right
            self.stack.append(value if name == "ADDII" else float(value))
        elif name == "ADDSS":
            left, right = pop2()
            self.stack.append(str(left) + str(right))
        elif name in ("SUBII", "SUBIF", "SUBFI", "SUBFF"):
            left, right = pop2()
            value = left - right
            self.stack.append(value if name == "SUBII" else float(value))
        elif name in ("MULII", "MULIF", "MULFI", "MULFF"):
            left, right = pop2()
            value = left * right
            self.stack.append(value if name == "MULII" else float(value))
        elif name in ("DIVII", "DIVIF", "DIVFI", "DIVFF"):
            left, right = pop2()
            if float(right) == 0.0:
                self.stack.append(0 if name == "DIVII" else 0.0)
            elif name == "DIVII":
                quotient = abs(int(left)) // abs(int(right))
                sign = -1 if (int(left) < 0) != (int(right) < 0) else 1
                self.stack.append(sign * quotient)
            else:
                self.stack.append(float(left) / float(right))
        elif name == "MODII":
            left, right = pop2()
            self.stack.append(0 if int(right) == 0 else int(left) - int(right) * int(
                (abs(int(left)) // abs(int(right))) * (-1 if (int(left) < 0) != (int(right) < 0) else 1)
            ))
        elif name in ("ADDVV", "SUBVV"):
            bz, by, bx = self._pop(), self._pop(), self._pop()
            az, ay, ax = self._pop(), self._pop(), self._pop()
            sign = 1.0 if name == "ADDVV" else -1.0
            self.stack.extend([float(ax) + sign * float(bx), float(ay) + sign * float(by), float(az) + sign * float(bz)])
        elif name in ("MULVF", "DIVVF"):
            scalar = float(self._pop())
            vz, vy, vx = self._pop(), self._pop(), self._pop()
            if name == "DIVVF" and scalar == 0.0:
                self.stack.extend([0.0, 0.0, 0.0])
            else:
                factor = scalar if name == "MULVF" else 1.0 / scalar
                self.stack.extend([float(vx) * factor, float(vy) * factor, float(vz) * factor])
        elif name in ("MULFV", "DIVFV"):
            vz, vy, vx = self._pop(), self._pop(), self._pop()
            scalar = float(self._pop())
            if name == "DIVFV" and (vx == 0.0 or vy == 0.0 or vz == 0.0):
                self.stack.extend([0.0, 0.0, 0.0])
            elif name == "MULFV":
                self.stack.extend([scalar * float(vx), scalar * float(vy), scalar * float(vz)])
            else:
                self.stack.extend([scalar / float(vx), scalar / float(vy), scalar / float(vz)])
        elif name in ("EQUALII", "EQUALFF", "EQUALSS", "EQUALOO",
                      "NEQUALII", "NEQUALFF", "NEQUALSS", "NEQUALOO"):
            left, right = pop2()
            # Odyssey compares NCS string operands using ASCII-insensitive equality.
            if name.endswith("SS"):
                equal = left.translate(_ASCII_CASE_FOLD) == right.translate(_ASCII_CASE_FOLD)
            else:
                equal = left == right
            self.stack.append(int(equal if name.startswith("EQUAL") else not equal))
        elif name in ("GEQII", "GEQFF", "GTII", "GTFF", "LTII", "LTFF", "LEQII", "LEQFF"):
            left, right = pop2()
            if name.startswith("GEQ"):
                self.stack.append(int(left >= right))
            elif name.startswith("GT"):
                self.stack.append(int(left > right))
            elif name.startswith("LT"):
                self.stack.append(int(left < right))
            else:
                self.stack.append(int(left <= right))
        elif name in ("LOGANDII", "LOGORII"):
            left, right = pop2()
            if name == "LOGANDII":
                self.stack.append(int(bool(int(left)) and bool(int(right))))
            else:
                self.stack.append(int(bool(int(left)) or bool(int(right))))
        elif name in ("INCORII", "EXCORII", "BOOLANDII"):
            left, right = pop2()
            if name == "INCORII":
                self.stack.append(int(left) | int(right))
            elif name == "EXCORII":
                self.stack.append(int(left) ^ int(right))
            else:
                self.stack.append(int(left) & int(right))
        elif name in ("SHLEFTII", "SHRIGHTII", "USHRIGHTII"):
            left, right = pop2()
            shift = int(right) & 31
            if name == "SHLEFTII":
                self.stack.append(_signed32(int(left) << shift))
            elif name == "SHRIGHTII":
                self.stack.append(int(left) >> shift)
            else:
                self.stack.append((int(left) & 0xFFFFFFFF) >> shift)
        else:
            return False
        return True

    # -- engine routine calls ------------------------------------------------
    def _execute_action(self, routine: int, supplied_count: int) -> None:
        if not 0 <= routine < len(self._table):
            raise NCSExecutionError(f"routine {routine} outside the engine table")
        signature = self._table[routine]
        name = str(getattr(signature, "name", "") or f"routine_{routine}")
        params: list[Any] = []
        consumed_state: NCSSavedState | None = None
        for position, param in enumerate(list(getattr(signature, "params", ()) or ())):
            if position >= supplied_count:
                default = getattr(param, "default", None)
                params.append(default)
                continue
            datatype = str(getattr(getattr(param, "datatype", None), "name", "") or "")
            if datatype == "ACTION":
                consumed_state = self._pending_state
                self._pending_state = None
                params.append(consumed_state)
                continue
            if datatype == "VECTOR":
                z_value, y_value, x_value = self._pop(), self._pop(), self._pop()
                params.append((float(x_value), float(y_value), float(z_value)))
                continue
            params.append(self._pop())

        outcome = self.context.call(name, params, consumed_state)
        return_type = str(getattr(getattr(signature, "returntype", None), "name", "") or "VOID")
        if return_type == "VOID":
            return
        if return_type == "VECTOR":
            vector = outcome if isinstance(outcome, tuple) and len(outcome) == 3 else (0.0, 0.0, 0.0)
            self.stack.extend([float(vector[0]), float(vector[1]), float(vector[2])])
            return
        if outcome is NotImplemented or outcome is None:
            defaults = {"INT": 0, "FLOAT": 0.0, "STRING": "", "OBJECT": OBJECT_INVALID}
            self.stack.append(defaults.get(return_type, 0))
            return
        if return_type == "INT":
            self.stack.append(int(outcome))
        elif return_type == "FLOAT":
            self.stack.append(float(outcome))
        elif return_type == "STRING":
            self.stack.append(str(outcome))
        else:
            self.stack.append(int(outcome))


def execute_ncs_script(
    ncs_bytes: bytes,
    *,
    game: str = "K2",
    context: MapStudioPIEScriptContext | None = None,
    instruction_budget: int = _DEFAULT_INSTRUCTION_BUDGET,
) -> NCSExecutionResult:
    """Execute one compiled script and return its observable PIE effects."""

    script_context = context or MapStudioPIEScriptContext(game=game)
    if not ncs_bytes:
        script_context.result.warnings.append("empty NCS payload")
        return script_context.result
    try:
        from pykotor.resource.formats.ncs import NCSBinaryReader

        ncs = NCSBinaryReader(io.BytesIO(bytes(ncs_bytes))).load()
    except Exception as exc:
        script_context.result.warnings.append(f"NCS decode failed: {exc}")
        return script_context.result
    machine = NCSVirtualMachine(
        list(getattr(ncs, "instructions", ()) or ()),
        script_context,
        instruction_budget=instruction_budget,
    )
    return machine.run()


__all__ = [
    "MapStudioPIEScriptContext",
    "NCSExecutionError",
    "NCSExecutionResult",
    "NCSSavedState",
    "NCSScriptedCommand",
    "NCSVirtualMachine",
    "OBJECT_INVALID",
    "OBJECT_SELF",
    "execute_ncs_script",
]
