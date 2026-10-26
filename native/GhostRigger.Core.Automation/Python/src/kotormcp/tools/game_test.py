"""Game-test handoff helpers for KOTOR save-load and warp proof runs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from kotormcp.game_dinput_hook import describe_hook
from kotormcp.schemas import ListSavesInput, PrepareSaveWarpTestInput
from kotormcp.state import load_installation, resolve_game
from kotormcp.utils import json_content


def get_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "kotor_list_saves",
            "description": "List KOTOR save folders and summarize savenfo/SAVEGAME module metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "k1, k2, swkotor, tsl, or kotor2"},
                    "path": {"type": "string", "description": "Optional absolute installation path"},
                    "save_root": {"type": "string", "description": "Optional folder containing save directories"},
                    "limit": {"type": "integer", "default": 25},
                    "offset": {"type": "integer", "default": 0},
                },
                "required": ["game"],
            },
        },
        {
            "name": "kotor_prepare_save_warp_test",
            "description": (
                "Prepare a repeatable live-game proof handoff: choose a real save to load, "
                "verify console/windowed/module readiness, and optionally launch KOTOR."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "description": "k1, k2, swkotor, tsl, or kotor2"},
                    "path": {"type": "string", "description": "Optional absolute installation path"},
                    "target_module": {"type": "string", "default": "tst_light"},
                    "save_folder": {"type": "string", "description": "Optional exact save folder path"},
                    "save_name": {"type": "string", "description": "Optional save or folder name substring to prefer"},
                    "save_root": {"type": "string", "description": "Optional folder containing save directories"},
                    "require_loaded_save_before_warp": {"type": "boolean", "default": True},
                    "require_dinput_hook": {"type": "boolean", "default": False},
                    "launch_game": {"type": "boolean", "default": False},
                    "steam_app_id": {"type": "integer"},
                },
                "required": ["game"],
            },
        },
    ]


async def handle_list_saves(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = ListSavesInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game (k1/k2)."})
    try:
        installation = load_installation(game, inp.path)
        saves = _discover_saves(Path(installation.path()), explicit_root=inp.save_root)
        offset = max(0, int(inp.offset or 0))
        limit = max(1, min(int(inp.limit or 25), 200))
        return json_content(
            {
                "game": installation.game_name(),
                "path": installation.path(),
                "count": len(saves),
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < len(saves),
                "items": saves[offset : offset + limit],
            }
        )
    except Exception as exc:
        return json_content({"error": str(exc)})


async def handle_prepare_save_warp_test(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = PrepareSaveWarpTestInput.model_validate(arguments)
    game = resolve_game(inp.game)
    if game is None:
        return json_content({"error": "Specify game (k1/k2)."})
    try:
        installation = load_installation(game, inp.path)
        game_root = Path(installation.path())
        target_module = _clean_module_root(inp.target_module)
        if not re.fullmatch(r"[a-z0-9_]{1,32}", target_module):
            return json_content(
                {"error": f"Invalid target_module (expected 1-32 chars [a-z0-9_]): {inp.target_module!r}"}
            )
        saves = _discover_saves(game_root, explicit_root=inp.save_root)
        selected = _select_save(
            saves,
            target_module=target_module,
            save_folder=inp.save_folder,
            save_name=inp.save_name,
            require_loaded_save_before_warp=bool(inp.require_loaded_save_before_warp),
            game_root=game_root,
        )
        selected_save_route = _save_route_status(selected, game_root) if selected else None
        module_status = _module_status(game_root, target_module)
        console = _console_status(game_root, installation.game_name(), target_module)
        windowed = _windowed_status(game_root, installation.game_name())
        currentgame = _currentgame_status(game_root, target_module)
        dinput_hook = describe_hook(game_root)
        loaded_save_route = bool(
            selected
            and selected_save_route
            and selected_save_route["resolvable"]
            and _clean_module_root(selected.get("last_module", "")) != target_module
        )
        blocking: list[str] = []
        warnings: list[str] = []
        if not selected:
            if inp.save_folder:
                blocking.append(f"Requested save folder was not found: {inp.save_folder}")
            elif inp.save_name:
                blocking.append(f"Requested save name did not match any discovered save: {inp.save_name}")
            elif saves:
                blocking.append(
                    "No usable save folder was found. Every discovered save has an unresolved LASTMODULE route; "
                    "the module must exist inside SAVEGAME.sav or as an installed .mod, .rim, or _s.rim file."
                )
            else:
                blocking.append("No usable save folder was found.")
        elif selected_save_route and not selected_save_route["resolvable"]:
            blocking.append(selected_save_route["blocking_issue"])
        elif bool(inp.require_loaded_save_before_warp) and not loaded_save_route:
            blocking.append(
                f"Selected save already starts in {target_module}; choose a normal save first, then run warp."
            )
        if not module_status["exists"]:
            blocking.append(f"Target module is not installed: {module_status['path']}")
        if not console["ready"]:
            blocking.append(console["fix_hint"])
        if not windowed["ready"]:
            warnings.append(windowed["fix_hint"])
        if not dinput_hook["installed"]:
            hook_hint = f"Install the DirectInput proxy hook with kotor_dinput_hook_install for {installation.game_name()}."
            if bool(getattr(inp, "require_dinput_hook", False)):
                blocking.append(hook_hint)
            else:
                warnings.append(hook_hint)
        if currentgame.get("exists"):
            warnings.append(
                f"Currentgame cache already contains {target_module}.mod; restart from the selected save before warping."
            )
        launched = _launch_game(game_root, installation.game_name(), inp.steam_app_id) if bool(inp.launch_game) else None
        if launched and not launched.get("ok"):
            blocking.extend(launched.get("blocking_issues", []))
        warp_command = f"warp {target_module}"
        load_label = _save_load_label(selected) if selected else ""
        ready = not blocking
        game_label = "KOTOR II" if str(installation.game_name()).upper() == "K2" else "KOTOR 1"
        next_steps = [
            f"Launch {game_label} in windowed mode." if launched is None else f"{game_label} launch was requested.",
            f"Load save {load_label}." if load_label else "Load a normal save before opening the console.",
            f"Open the cheat console and type `{warp_command}`.",
            "Capture the visible game window after the module loads and the target area is on screen.",
        ]
        return json_content(
            {
                "ready": ready,
                "game": installation.game_name(),
                "path": installation.path(),
                "target_module": target_module,
                "warp_command": warp_command,
                "selected_save": selected,
                "selected_save_route": selected_save_route,
                "loaded_save_before_warp": loaded_save_route,
                "module": module_status,
                "console": console,
                "windowed": windowed,
                "dinput_hook": dinput_hook,
                "currentgame_cache": currentgame,
                "launch": launched,
                "next_steps": next_steps,
                "warnings": warnings,
                "blocking_issues": blocking,
            }
        )
    except Exception as exc:
        return json_content({"error": str(exc)})


def _clean_module_root(value: str) -> str:
    text = str(value or "").strip()
    for suffix in (".mod", ".rim", ".sav"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    return text.lower()


def _candidate_save_roots(game_root: Path, explicit_root: Optional[str]) -> Iterable[Path]:
    if explicit_root:
        yield Path(explicit_root)
    yield game_root / "cloudsaves"
    yield game_root / "saves"
    yield game_root.parent / "saves"


MAX_SAVE_SCAN_DIRS = 20000
MAX_SAVE_ROWS = 1000
SAVE_SCAN_MAX_DEPTH = 6


def _iter_savegames(root: Path) -> "Iterable[Path]":
    """Yield SAVEGAME.sav paths under ``root`` with bounded breadth and depth.

    The previous ``root.glob('**/SAVEGAME.sav')`` walked an unbounded tree when
    an explicit ``save_root`` was supplied.  This bounds the number of scanned
    directories and the recursion depth so a large or adversarial root cannot
    stall the shared MCP queue.
    """

    scanned = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        scanned += 1
        if scanned > MAX_SAVE_SCAN_DIRS:
            return
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_file() and entry.name.lower() == "savegame.sav":
                    yield entry
                elif entry.is_dir() and depth < SAVE_SCAN_MAX_DEPTH:
                    stack.append((entry, depth + 1))
            except OSError:
                continue


def _discover_saves(game_root: Path, *, explicit_root: Optional[str]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for root in _candidate_save_roots(game_root, explicit_root):
        if not root.is_dir():
            continue
        for savegame in _iter_savegames(root):
            folder = savegame.parent.resolve()
            if folder in seen:
                continue
            seen.add(folder)
            summary = _save_summary(folder)
            if summary:
                rows.append(summary)
            if len(rows) >= MAX_SAVE_ROWS:
                break
        if len(rows) >= MAX_SAVE_ROWS:
            break
    rows.sort(key=lambda item: float(item.get("modified_time") or 0), reverse=True)
    return rows


def _save_summary(folder: Path) -> dict[str, Any]:
    savenfo = _read_savenfo(folder / "savenfo.res")
    modules = _read_savegame_modules(folder / "SAVEGAME.sav")
    try:
        modified = (folder / "SAVEGAME.sav").stat().st_mtime
    except OSError:
        modified = 0
    return {
        "folder": str(folder),
        "folder_name": folder.name,
        "save_name": savenfo.get("SAVEGAMENAME", ""),
        "area_name": savenfo.get("AREANAME", ""),
        "last_module": savenfo.get("LASTMODULE", ""),
        "pc_name": savenfo.get("PCNAME", ""),
        "save_number": savenfo.get("SAVENUMBER", None),
        "cheat_used": savenfo.get("CHEATUSED", None),
        "timestamp": savenfo.get("TIMESTAMP", None),
        "modified_time": modified,
        "modules": modules,
    }


def _read_savenfo(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        from pykotor.resource.formats.gff.gff_auto import read_gff  # noqa: PLC0415
    except ImportError:
        return {}
    try:
        root = read_gff(path.read_bytes()).root
    except Exception:
        return {}
    data: dict[str, Any] = {}
    for label, _field_type, value in root:
        if label in {"AREANAME", "LASTMODULE", "SAVEGAMENAME", "PCNAME"}:
            data[label] = str(value)
        elif label in {"SAVENUMBER", "CHEATUSED", "TIMESTAMP", "TIMEPLAYED", "PCAUTOSAVE"}:
            try:
                data[label] = int(value)
            except (TypeError, ValueError):
                data[label] = value
    return data


def _read_savegame_modules(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        from pykotor.resource.formats.erf.erf_auto import read_erf  # noqa: PLC0415
    except ImportError:
        return []
    try:
        erf = read_erf(str(path))
    except Exception:
        return []
    modules: list[str] = []
    for resource in erf:
        if str(resource.restype.extension).lower() == "sav":
            modules.append(str(resource.resref).lower())
    return sorted(modules)


def _select_save(
    saves: list[dict[str, Any]],
    *,
    target_module: str,
    save_folder: Optional[str],
    save_name: Optional[str],
    require_loaded_save_before_warp: bool,
    game_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    if save_folder:
        requested = str(Path(save_folder).resolve()).lower()
        return next((save for save in saves if str(Path(save["folder"]).resolve()).lower() == requested), None)
    if save_name:
        needle = save_name.lower()
        for save in saves:
            haystack = " ".join(
                str(save.get(key) or "")
                for key in ("folder_name", "save_name", "area_name", "last_module", "pc_name")
            ).lower()
            if needle in haystack:
                return save
        return None
    installed_module_files = _installed_module_file_index(game_root) if game_root is not None else None
    usable_saves = [
        save
        for save in saves
        if game_root is None
        or _save_route_status(save, game_root, installed_module_files=installed_module_files)["resolvable"]
    ]
    if require_loaded_save_before_warp:
        for save in usable_saves:
            if _clean_module_root(save.get("last_module") or "") != target_module.lower():
                return save
    return usable_saves[0] if usable_saves else None


def _installed_module_file_index(game_root: Path) -> dict[str, Path]:
    modules_dir = game_root / "Modules"
    try:
        return {
            path.name.casefold(): path
            for path in modules_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".mod", ".rim"}
        }
    except OSError:
        return {}


def _save_route_status(
    save: dict[str, Any],
    game_root: Path,
    *,
    installed_module_files: Optional[dict[str, Path]] = None,
) -> dict[str, Any]:
    last_module = _clean_module_root(save.get("last_module", ""))
    archive_modules = sorted(
        {
            module
            for value in (save.get("modules") or [])
            if (module := _clean_module_root(value))
        }
    )
    installed_index = (
        installed_module_files
        if installed_module_files is not None
        else _installed_module_file_index(game_root)
    )
    expected_names = (
        f"{last_module}.mod",
        f"{last_module}.rim",
        f"{last_module}_s.rim",
    ) if last_module else ()
    installed_paths = [
        str(installed_index[name.casefold()])
        for name in expected_names
        if name.casefold() in installed_index
    ]
    archive_contains_last_module = bool(last_module and last_module in archive_modules)
    resolvable = bool(last_module and (archive_contains_last_module or installed_paths))
    if resolvable:
        blocking_issue = ""
    elif not last_module:
        blocking_issue = (
            "Selected save cannot be loaded: savenfo.res has no LASTMODULE value, so its module route "
            "cannot be resolved from SAVEGAME.sav or installed module files."
        )
    else:
        blocking_issue = (
            f"Selected save cannot be loaded: LASTMODULE '{last_module}' is absent from SAVEGAME.sav and no "
            f"installed {last_module}.mod, {last_module}.rim, or {last_module}_s.rim file exists."
        )
    return {
        "last_module": last_module,
        "archive_contains_last_module": archive_contains_last_module,
        "savegame_modules": archive_modules,
        "expected_installed_files": [str(game_root / "Modules" / name) for name in expected_names],
        "installed_module_files": installed_paths,
        "resolvable": resolvable,
        "blocking_issue": blocking_issue,
    }


def _module_status(game_root: Path, target_module: str) -> dict[str, Any]:
    path = game_root / "Modules" / f"{target_module}.mod"
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def _ini_path(game_root: Path, game_name: str) -> Path:
    if str(game_name).upper() == "K2":
        return game_root / "swkotor2.ini"
    return game_root / "swkotor.ini"


def _ini_text(game_root: Path, game_name: str) -> tuple[Path, str]:
    path = _ini_path(game_root, game_name)
    try:
        return path, path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return path, ""


def _ini_value(text: str, key: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n;#]+)", text)
    return match.group(1).strip() if match else ""


def _console_status(game_root: Path, game_name: str, target_module: str) -> dict[str, Any]:
    path, text = _ini_text(game_root, game_name)
    value = _ini_value(text, "EnableCheats")
    ready = value == "1"
    note = (
        "TSL's console is hidden; type the warp command even if no console text is visible."
        if str(game_name).upper() == "K2"
        else "KOTOR 1 uses the same cheat console route; type the warp command and press Enter."
    )
    return {
        "ready": ready,
        "ini_path": str(path),
        "enable_cheats": value,
        "fix_hint": "" if ready else f"Set EnableCheats=1 in {path.name} before running warp {target_module}.",
        "note": note,
    }


def _windowed_status(game_root: Path, game_name: str) -> dict[str, Any]:
    path, text = _ini_text(game_root, game_name)
    allow_windowed = _ini_value(text, "AllowWindowedMode")
    fullscreen = _ini_value(text, "FullScreen")
    ready = allow_windowed == "1" and fullscreen == "0"
    return {
        "ready": ready,
        "ini_path": str(path),
        "allow_windowed_mode": allow_windowed,
        "fullscreen": fullscreen,
        "fix_hint": "" if ready else f"Set AllowWindowedMode=1 and FullScreen=0 in {path.name}.",
    }


def _currentgame_status(game_root: Path, target_module: str) -> dict[str, Any]:
    path = game_root / "currentgame" / f"{target_module}.mod"
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def _launch_game(game_root: Path, game_name: str, steam_app_id: Optional[int]) -> dict[str, Any]:
    game_key = str(game_name).upper()
    if steam_app_id or game_key == "K2" or "steamapps" in str(game_root).lower():
        app_id = int(steam_app_id or (208580 if game_key == "K2" else 32370))
        command = ["powershell", "-NoProfile", "-Command", f"Start-Process 'steam://run/{app_id}'"]
    else:
        executable = game_root / "swkotor.exe"
        command = [str(executable)]
    try:
        subprocess.Popen(command, cwd=str(game_root))  # noqa: S603
    except OSError as exc:
        return {"ok": False, "command": command, "blocking_issues": [str(exc)]}
    return {"ok": True, "command": command, "blocking_issues": []}


def _save_load_label(save: dict[str, Any]) -> str:
    number = save.get("save_number")
    name = save.get("save_name") or save.get("folder_name")
    area = save.get("area_name")
    if number is not None:
        return f"{number} ({name}, {area})"
    return f"{name} ({area})"
