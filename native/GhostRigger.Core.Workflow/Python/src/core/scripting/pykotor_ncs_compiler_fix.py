"""In-memory correctness fix for PyKotor 2.3.1's NWScript compiler.

PyKotor 2.3.1 links forward-declared functions to the wrong bodies, pushes engine
call arguments in non-retail order, mis-scopes ``switch`` selectors and emits a
non-retail entry wrapper. A KOTOR II ``k_ai_master.ncs`` built with it sent 681
of 686 JSR calls to wrong functions, so every conversation start failed in game.

``ensure_pykotor_ncs_compiler_fix`` applies the LGPL diff in
:mod:`pykotor_ncs_compiler_2_3_1_diff` to the installed ``classes.py`` source in
memory, checks both the source and the result against pinned SHA-256 values,
and swaps only the changed methods onto PyKotor's existing classes. Modules that
already imported those classes (lexer, parser, compilers) therefore pick up the
fix without being reloaded, and nothing on disk changes.

The fix fails closed: on any other PyKotor revision compilation is refused with
a clear message, unless ``GHOSTRIGGER_ALLOW_UNPATCHED_PYKOTOR=1`` is set (the
same escape hatch as the MDL reader fixes). The compile probe in
``studio._check_nwscript_provider`` still verifies the result either way.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import threading
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.scripting import pykotor_ncs_compiler_2_3_1_diff as _patch

_MODULE = "pykotor.resource.formats.ncs.compiler.classes"
_MARKER = "_ghost_ncs_compiler_fix"
_ALLOW_UNPATCHED_ENV = "GHOSTRIGGER_ALLOW_UNPATCHED_PYKOTOR"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_LOCK = threading.RLock()


class UnsupportedPyKotorCompiler(RuntimeError):
    """The installed PyKotor compiler is not the revision the fix was qualified on."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_unified_diff(source: str, diff: str) -> str:
    """Apply a single-file unified diff strictly: every context and removed line must match."""
    old = source.splitlines(True)
    out: list[str] = []
    cursor = 0
    lines = diff.splitlines(True)
    index = 0
    while index < len(lines) and not lines[index].startswith("@@"):
        index += 1
    while index < len(lines):
        match = _HUNK.match(lines[index])
        if not match:
            raise ValueError(f"Malformed hunk header: {lines[index]!r}")
        start = int(match.group(1)) - (0 if match.group(2) == "0" else 1)
        if start < cursor:
            raise ValueError("Overlapping hunks in PyKotor compiler diff.")
        out.extend(old[cursor:start])
        cursor = start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            tag, body = line[:1], line[1:]
            if tag == "\\":
                index += 1
                continue
            if tag in (" ", "-"):
                if cursor >= len(old) or old[cursor] != body:
                    raise ValueError(f"PyKotor compiler diff does not match the source at line {cursor + 1}.")
                cursor += 1
                if tag == " ":
                    out.append(body)
            elif tag == "+":
                out.append(body)
            else:
                raise ValueError(f"Unexpected diff line: {line!r}")
            index += 1
    out.extend(old[cursor:])
    return "".join(out)


def _method_index(source: str) -> dict[tuple[str, str], str]:
    methods: dict[tuple[str, str], str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[(node.name, member.name)] = ast.dump(member)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[("", node.name)] = ast.dump(node)
    return methods


def _class_bodies(source: str) -> dict[str, str]:
    bodies = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            rest = [m for m in node.body if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bodies[node.name] = ast.dump(ast.Module(rest, [])) + repr([ast.dump(b) for b in node.bases])
    return bodies


def _rebind_function(function: types.FunctionType, module_globals: dict, new_cls: type | None, old_cls: type | None):
    closure = function.__closure__
    if closure and new_cls is not None:
        closure = tuple(
            types.CellType(old_cls) if cell.cell_contents is new_cls else cell for cell in closure
        )
    rebound = types.FunctionType(function.__code__, module_globals, function.__name__, function.__defaults__, closure)
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__qualname__ = function.__qualname__
    rebound.__doc__ = function.__doc__
    rebound.__dict__.update(function.__dict__)
    rebound.__annotations__ = dict(getattr(function, "__annotations__", {}) or {})
    return rebound


def _rebind_member(member: Any, module_globals: dict, new_cls: type | None, old_cls: type | None):
    if isinstance(member, staticmethod):
        return staticmethod(_rebind_function(member.__func__, module_globals, new_cls, old_cls))
    if isinstance(member, classmethod):
        return classmethod(_rebind_function(member.__func__, module_globals, new_cls, old_cls))
    if isinstance(member, types.FunctionType):
        return _rebind_function(member, module_globals, new_cls, old_cls)
    raise UnsupportedPyKotorCompiler(f"Cannot rebind compiler member of type {type(member).__name__}.")


def ensure_pykotor_ncs_compiler_fix() -> dict[str, Any]:
    """Apply the qualified compiler fix once per process and report what was done."""
    with _LOCK:
        import importlib

        classes = importlib.import_module(_MODULE)
        if getattr(classes, _MARKER, None) == _patch.PATCHED_SHA256:
            return {"applied": True, "already": True, "methods": getattr(classes, _MARKER + "_methods", ())}
        path = Path(getattr(classes, "__file__", "") or "")
        try:
            source = path.read_text(encoding="utf-8") if path.suffix == ".py" else ""
        except OSError:
            source = ""
        reason = ""
        if not source:
            reason = f"PyKotor compiler source is not available ({path or 'no __file__'})."
        elif _sha256(source) != _patch.SOURCE_SHA256:
            reason = (
                "Installed PyKotor compiler is not the qualified 2.3.1 revision "
                f"(classes.py sha256 {_sha256(source)}); Ghost Studio's NWScript fixes cannot be applied."
            )
        if reason:
            if os.environ.get(_ALLOW_UNPATCHED_ENV, "").strip() in {"1", "true", "yes"}:
                return {"applied": False, "bypassed": True, "reason": reason}
            raise UnsupportedPyKotorCompiler(reason + " Install pykotor==2.3.1.")
        patched = apply_unified_diff(source, _patch.DIFF)
        if _sha256(patched) != _patch.PATCHED_SHA256:
            raise UnsupportedPyKotorCompiler("Patched PyKotor compiler source does not match its pinned SHA-256.")
        if _class_bodies(source) != _class_bodies(patched):
            raise UnsupportedPyKotorCompiler("Compiler fix changes class bodies; method swapping is not safe.")

        namespace: dict[str, Any] = {
            "__name__": classes.__name__,
            "__package__": classes.__package__,
            "__file__": str(path),
            "__builtins__": __builtins__,
        }
        exec(compile(patched, f"{path} [ghost-ncs-compiler-fix]", "exec"), namespace)

        before, after = _method_index(source), _method_index(patched)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        module_globals = classes.__dict__
        for class_name, name in changed:
            if not class_name:
                if (class_name, name) in after:
                    module_globals[name] = _rebind_member(namespace[name], module_globals, None, None)
                continue
            old_cls = getattr(classes, class_name)
            new_cls = namespace[class_name]
            if (class_name, name) not in after:
                if name in old_cls.__dict__:
                    delattr(old_cls, name)
                continue
            setattr(old_cls, name, _rebind_member(new_cls.__dict__[name], module_globals, new_cls, old_cls))
        labels = tuple(f"{c}.{n}" if c else n for c, n in changed)
        setattr(classes, _MARKER, _patch.PATCHED_SHA256)
        setattr(classes, _MARKER + "_methods", labels)
        return {"applied": True, "already": False, "methods": labels}


@lru_cache(maxsize=2)
def _probe_compiler(game: Any) -> None:
    from pykotor.resource.formats.ncs import bytes_ncs, compile_nss, read_ncs

    # Two forward-declared functions: the bug this fix removes linked both calls to one body.
    source = """void a(); void b();
void a() { PrintInteger(1); }
void b() { PrintInteger(2); }
void main() { a(); b(); }"""
    instructions = read_ncs(bytes_ncs(compile_nss(source, game))).instructions
    calls = [op for op in instructions if op.ins_type.name == "JSR"]
    if (len(calls) != 3 or
            calls[0].jump is not calls[1] or
            [(op.jump.ins_type.name, op.jump.args) for op in calls[1:]] !=
            [("CONSTI", [1]), ("CONSTI", [2])]):
        raise ValueError("Compiler JSR probe did not reach distinct named function bodies.")

    # Retail ACTION handlers pop the first declared argument first.
    abi = compile_nss('void main() { SetGlobalNumber("gs_abi_probe", 37); }', game)
    ops = abi.instructions
    sites = [i for i, op in enumerate(ops) if op.ins_type.name == "ACTION"]
    if len(sites) != 1 or sites[0] < 2:
        raise ValueError("Compiler ACTION ABI probe did not emit the expected call.")
    slot = sites[0]
    if [(op.ins_type.name, op.args) for op in ops[slot - 2:slot]] != [
        ("CONSTI", [37]), ("CONSTS", ["gs_abi_probe"]),
    ]:
        raise ValueError("Compiler pushes ACTION arguments in non-retail order.")


def verify_nwscript_compiler(game: Any) -> None:
    """Fix PyKotor's compiler in memory (once) and prove it on a minimal script for ``game``.

    ``game`` is a ``pykotor.common.misc.Game``. Raises if either step fails, so no
    caller ever writes bytecode from a compiler that links calls to wrong bodies.
    """
    ensure_pykotor_ncs_compiler_fix()
    _probe_compiler(game)


#: Opcodes the retail VMs execute. K2's CVirtualMachineInternal::RunInstructions
#: (0x006FF9F0) rejects anything outside 0x01-0x2C (NOP 0x2D included) and
#: aborts the whole script, so such bytecode must never be shipped.
RETAIL_OPCODES = range(0x01, 0x2D)


def assert_retail_opcodes(ncs: Any) -> None:
    """Raise if a compiled ``NCS`` holds an opcode the retail VM aborts on."""
    for index, instruction in enumerate(getattr(ncs, "instructions", ()) or ()):
        code = int(instruction.ins_type.value[0])
        if code not in RETAIL_OPCODES:
            raise ValueError(
                f"Compiled NCS instruction {index} uses opcode 0x{code:02X} "
                f"({instruction.ins_type.name}); the retail VM aborts on it."
            )


def compile_nss_verified(source: str, game: Any, **kwargs: Any):
    """``compile_nss`` behind :func:`verify_nwscript_compiler`, rejecting non-retail opcodes."""
    verify_nwscript_compiler(game)
    from pykotor.resource.formats.ncs import compile_nss

    ncs = compile_nss(source, game, **kwargs)
    assert_retail_opcodes(ncs)
    return ncs
