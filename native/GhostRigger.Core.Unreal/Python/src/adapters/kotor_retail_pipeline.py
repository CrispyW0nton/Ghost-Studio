"""Windows playtest adapter: inspect processes and launch the configured route."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import psutil


def game_processes():
    rows = []
    for process in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if (process.info["name"] or "").casefold() in {"swkotor2.exe", "swkotor.exe"}:
                rows.append({"pid": process.pid, "executable": process.info.get("exe") or ""})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rows


def require_game_closed():
    if game_processes():
        raise RuntimeError("KOTOR is running. Close it normally, then choose Install & Play. Your build is retained.")


def validate_profile(profile):
    root = Path(profile.get("game_root", "")).resolve()
    if not (root / "swkotor2.exe").is_file() or not (root / "Modules").is_dir():
        raise ValueError("Choose the KOTOR II folder containing swkotor2.exe and Modules.")
    launcher = Path(profile.get("launcher", "")).resolve()
    if not launcher.is_file() or (launcher.suffix.lower() != ".lnk" and launcher.name.casefold() != "kpatchlauncher.exe"):
        raise ValueError("Choose your existing patched-game shortcut or KPatchLauncher.exe.")
    if profile.get("debugger_script"):
        if not Path(profile.get("debugger_python", "")).is_file() or not Path(profile["debugger_script"]).is_file():
            raise ValueError("Select an existing Python executable and KotorDebugger script, or turn off debugger recording.")
    return root, launcher


def launch_playtest(profile, evidence_dir, module):
    root, launcher = validate_profile(profile)
    require_game_closed()
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    monitor_pid = None
    if profile.get("debugger_script"):
        with (evidence / "debugger-output.log").open("ab") as output:
            process = subprocess.Popen([profile["debugger_python"], profile["debugger_script"], "monitor", "--game", "k2",
                "--game-root", str(root), "--session-dir", str(evidence / "debugger"), "--wait-for-process",
                "--duration-seconds", "900", "--asset-resref", module], stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        monitor_pid = process.pid
    if launcher.suffix.lower() == ".lnk":
        os.startfile(str(launcher))
    else:
        subprocess.Popen([str(launcher)], cwd=launcher.parent, creationflags=subprocess.CREATE_NO_WINDOW)
    return {"launch_requested": True, "debugger_process_id": monitor_pid,
        "instruction": f"Load a clean test save, then enter: warp {module}. Confirm the actual load, traversal and save/reload below.",
        "game_observed": False}
