"""Focused contracts for the PIE NCS virtual machine.

A real stack machine over compiled NWScript — not pattern matching. Semantics
grounded in the NCS binary format (PyKotor round-trip) and Ghidra decompilation
of the K1 engine: routine dispatch is a per-id function table
(``CSWVirtualMachineCommands::ExecuteCommand`` @0052c0d0), ``AssignCommand``
queues its closure as a delay-0 event on the target which then runs with
OBJECT_SELF = target (@0052e720), ``DelayCommand`` pops float-then-command and
schedules at +seconds (@0052fe30), and a saved state resumes by restoring the
copied stack wholesale (``RunScriptSituation`` @005d4ad0). Editor-side only —
retail KOTOR remains the in-game authority.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_207TEL_MOD = Path(
    r"C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\Saved\VisibleProof"
    r"\2026-07-15_handoff_quality_pass\207tel_actual_source_fixture\Modules\207tel.mod"
)


def _ins(name, *args, jump=None):
    return SimpleNamespace(ins_type=SimpleNamespace(name=name), args=list(args), jump=jump)


def _run(instructions, **context_kwargs):
    from src.core.modules.map_studio_pie_nwscript_vm import (
        MapStudioPIEScriptContext,
        NCSVirtualMachine,
    )

    context = MapStudioPIEScriptContext(game="K2", **context_kwargs)
    machine = NCSVirtualMachine(list(instructions), context)
    result = machine.run()
    return machine, result


def test_arithmetic_branching_and_subroutine_flow() -> None:
    # main: JSR sub; sub computes 5+3; SetGlobalNumber("x", <sum>).
    # Compiler arg order (verified against real 207TEL bytecode + the engine
    # popping declared-order): last param pushed first, first param on top.
    retn = _ins("RETN")
    sub_body = [_ins("CONSTI", 5), _ins("CONSTI", 3), _ins("ADDII"), retn]
    set_global = [
        _ins("CONSTS", "x"),      # name on top, popped first
        _ins("ACTION", 581, 2),   # SetGlobalNumber(name, value=sum below it)
        _ins("RETN"),
    ]
    jsr = _ins("JSR", jump=sub_body[0])
    program = [jsr] + set_global + sub_body
    machine, result = _run(program)

    assert result.completed, result.warnings
    assert result.global_numbers == {"x": 8}


def test_const_object_self_compares_equal_to_current_script_actor() -> None:
    from src.core.modules.map_studio_pie_nwscript_vm import _function_table

    lookup = next(i for i, f in enumerate(_function_table("K2")) if f.name == "GetFirstPC")
    program = [_ins("CONSTO", 0), _ins("ACTION", lookup, 0), _ins("EQUALOO"), _ins("RETN")]
    machine, result = _run(program, self_object_id=100)
    assert result.completed and machine.stack == [1]


def test_jz_takes_branch_on_zero_and_loops_terminate() -> None:
    # while (i != 3) i++;  encoded with JNZ back-edge, then SetGlobalNumber("i", i).
    end = _ins("RETN")
    loop_check = _ins("CPTOPSP", -4, 4)
    program = [
        _ins("RSADDI"),              # i = 0 slot
        loop_check,                  # copy i
        _ins("CONSTI", 3),
        _ins("EQUALII"),
        _ins("JNZ", jump=end),       # i == 3 -> exit
        _ins("INCxSP", -4),          # i++
        _ins("JMP", jump=loop_check),
        end,
    ]
    # Exiting at RETN directly: globals not needed; completion is the contract.
    machine, result = _run(program)
    assert result.completed, result.warnings
    assert machine.stack == [3]


def test_action_pops_declared_params_and_pushes_typed_default() -> None:
    # GetGlobalNumber("missing") pushes int default 0 for an unset global.
    program = [
        _ins("CONSTS", "missing"),
        _ins("ACTION", 580, 1),      # GetGlobalNumber
        _ins("RETN"),
    ]
    machine, result = _run(program)
    assert result.completed
    assert machine.stack == [0]


def test_unknown_routine_census_and_typed_default() -> None:
    from src.core.modules.map_studio_pie_nwscript_vm import _function_table

    table = _function_table("K2")
    # SetCustomToken(int, string) is deliberately unhandled: void return, census.
    routine = next(i for i, f in enumerate(table) if getattr(f, "name", "") == "SetCustomToken")
    program = [
        _ins("CONSTS", "value"),
        _ins("CONSTI", 50),
        _ins("ACTION", routine, 2),
        _ins("RETN"),
    ]
    machine, result = _run(program)
    assert result.completed
    assert result.unknown_routines == {"SetCustomToken": 1}
    assert machine.stack == []


def test_store_state_closure_runs_with_assigned_object_self() -> None:
    # AssignCommand(GetObjectByTag("dancer"), ActionPlayAnimation(38)):
    # STORE_STATE; JMP past; <closure: CONSTI 38 ... ACTION 40; RETN>; past: ...
    closure = [
        _ins("CONSTF", -1.0),
        _ins("CONSTF", 1.0),
        _ins("CONSTI", 38),
        _ins("ACTION", 40, 3),       # ActionPlayAnimation
        _ins("RETN"),
    ]
    past = [
        _ins("CONSTI", 0),
        _ins("CONSTS", "dancer"),
        _ins("ACTION", 200, 2),      # GetObjectByTag
        _ins("ACTION", 6, 2),        # AssignCommand(object, action)
        _ins("RETN"),
    ]
    program = [_ins("STORE_STATE", 0, 0), _ins("JMP", jump=past[0])] + closure + past
    machine, result = _run(program)

    assert result.completed, result.warnings
    plays = [c for c in result.commands if c.kind == "ActionPlayAnimation"]
    assert len(plays) == 1
    assert plays[0].object_tag == "dancer"      # closure ran as the target
    assert plays[0].args[0] == 38


def test_delay_command_schedules_closure_for_the_timeline() -> None:
    closure = [_ins("CONSTI", 1), _ins("CONSTS", "later"), _ins("ACTION", 581, 2), _ins("RETN")]
    past = [
        _ins("CONSTF", 2.5),
        _ins("ACTION", 7, 2),        # DelayCommand(float, action)
        _ins("RETN"),
    ]
    program = [_ins("STORE_STATE", 0, 0), _ins("JMP", jump=past[0])] + closure + past
    machine, result = _run(program)

    assert result.completed, result.warnings
    assert result.global_numbers == {}          # not fired yet
    scheduled = [c for c in result.commands if c.kind == "DelayCommand"]
    assert len(scheduled) == 1
    assert scheduled[0].delay_seconds == pytest.approx(2.5)
    # Firing the closure later executes the deferred write.
    machine.run_saved_state(scheduled[0].saved_state)
    assert result.global_numbers == {"later": 1}


def test_locals_key_by_object_tag_and_registry_resolves_validity() -> None:
    registry = {("guard", 0): 42}
    program = [
        _ins("CONSTI", 1),           # value TRUE
        _ins("CONSTI", 50),          # index
        _ins("CONSTI", 0),
        _ins("CONSTS", "guard"),
        _ins("ACTION", 200, 2),      # GetObjectByTag -> 42 via registry
        _ins("ACTION", 680, 3),      # SetLocalBoolean(object, index, value)
        _ins("RETN"),
    ]
    machine, result = _run(
        program,
        object_by_tag=lambda tag, nth: registry.get((tag, nth)),
        tag_of_object=lambda oid: "guard" if oid == 42 else "",
    )
    assert result.completed, result.warnings
    assert result.local_booleans == {("guard", 50): True}


def test_instruction_budget_stops_infinite_loops() -> None:
    from src.core.modules.map_studio_pie_nwscript_vm import (
        MapStudioPIEScriptContext,
        NCSVirtualMachine,
    )

    spin = _ins("JMP")
    spin.jump = spin
    context = MapStudioPIEScriptContext(game="K2")
    machine = NCSVirtualMachine([spin], context, instruction_budget=500)
    result = machine.run()
    assert not result.completed
    assert any("budget" in w for w in result.warnings)


@pytest.mark.skipif(not _207TEL_MOD.is_file(), reason="207TEL fixture module not present")
def test_real_207tel_onenter_matches_static_animation_extractor() -> None:
    """Executing real retail bytecode reproduces the independent extractor's
    tag->animation intents exactly once scheduled closures fire, and campaign
    conditionals only write when their gate state is seeded."""

    from pykotor.extract.capsule import LazyCapsule
    from pykotor.resource.type import ResourceType as RT

    from src.core.modules.map_studio_pie_nwscript_vm import (
        MapStudioPIEScriptContext,
        execute_ncs_script,
    )
    from src.core.modules.map_studio_scene_animations import extract_scene_animation_intents

    data = bytes(LazyCapsule(str(_207TEL_MOD)).resource("k_207tel_enter", RT.NCS))

    context = MapStudioPIEScriptContext(game="K2")
    result = execute_ncs_script(data, game="K2", context=context)
    assert result.completed and not result.warnings

    for command in sorted(
        (c for c in result.commands if c.saved_state is not None),
        key=lambda c: c.delay_seconds,
    ):
        context.vm.run_saved_state(command.saved_state)

    by_id = {v: k for k, v in context._synthetic_ids.items()}
    vm_intents = {
        by_id.get(c.object_id, (c.object_tag, 0)): int(c.args[0])
        for c in result.commands
        if c.kind == "ActionPlayAnimation"
    }
    assert vm_intents == dict(extract_scene_animation_intents(data))

    # Clean sandbox: campaign-gated writes must NOT fire...
    assert result.global_numbers == {}
    # ...but seeding the gate state fires the retail chain advance.
    seeded = MapStudioPIEScriptContext(game="K2", global_numbers={"207TEL_Benok": 1})
    seeded_result = execute_ncs_script(data, game="K2", context=seeded)
    assert seeded_result.global_numbers == {"207TEL_Benok": 2}
