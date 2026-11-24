"""Headless Blender bridge for Sprint 3 reverse animation extraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .fbx_exporter import FBXExportFailure, find_blender_executable
from .skeleton_renamer import REPO_ROOT


def resolve_blender_extract_script() -> Path:
    """Locate the external Blender helper in source and native-host layouts."""

    configured = os.environ.get("GHOSTRIGGER_BLENDER_EXTRACT_SCRIPT", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    app_root = os.environ.get("GHOSTRIGGER_NATIVE_APP_ROOT", "").strip()
    if app_root:
        candidates.append(Path(app_root) / "scripts" / "blender_extract_ue5_animation.py")
    candidates.extend(
        [
            REPO_ROOT / "scripts" / "blender_extract_ue5_animation.py",
            Path(sys.executable).resolve().parent / "scripts" / "blender_extract_ue5_animation.py",
            Path.cwd().resolve() / "scripts" / "blender_extract_ue5_animation.py",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


BLENDER_EXTRACT_SCRIPT = resolve_blender_extract_script()


def _hidden_process_options() -> dict[str, Any]:
    """Keep headless Blender invisible in the Windows desktop application."""

    if os.name != "nt":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startup,
    }


def run_blender_animation_extraction(
    *,
    source_fbx: Path,
    output_json: Path,
    action_name: str = "",
    frame_step: int = 1,
    blender_executable: Optional[Path] = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Extract UE5 FBX action data to JSON with Blender."""

    extract_script = resolve_blender_extract_script()
    if not extract_script.exists():
        return {
            "success": False,
            "errors": [f"Blender extraction script not found: {extract_script}"],
        }

    try:
        blender = find_blender_executable(blender_executable)
    except FBXExportFailure as exc:
        return {"success": False, "errors": [str(exc)]}

    cmd = [
        str(blender),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(extract_script),
        "--",
        "--fbx",
        str(source_fbx),
        "--json",
        str(output_json),
        "--frame-step",
        str(max(1, int(frame_step))),
    ]
    if action_name:
        cmd.extend(["--action", action_name])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    # Never fall back to a previous run's JSON (e.g. a static sample cached before a fix).
    output_json.unlink(missing_ok=True)
    log_path = output_json.with_suffix(".blender.log")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        **_hidden_process_options(),
    )
    log_path.write_text(
        f"COMMAND:\n{' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return {
            "success": False,
            "errors": [f"Blender returned {proc.returncode}", proc.stderr[-2000:]],
            "log_path": str(log_path),
        }
    if not output_json.exists():
        return {
            "success": False,
            "errors": ["Blender completed but did not write extraction JSON"],
            "log_path": str(log_path),
        }
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    payload["log_path"] = str(log_path)
    return payload
