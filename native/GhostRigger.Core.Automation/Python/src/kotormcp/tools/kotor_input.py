"""KOTOR live-window input tools for save-load and warp proof runs."""

from __future__ import annotations

from typing import Any, Dict, List

from kotormcp.game_input import KotorInputError, KotorWindowInput, run_save_warp_route
from kotormcp.schemas import (
    KotorCaptureWindowInput,
    KotorInputClickInput,
    KotorInputStatusInput,
    KotorInputTypeInput,
    KotorRunSaveWarpRouteInput,
)
from kotormcp.state import load_installation, resolve_game
from kotormcp.utils import json_content


def get_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "kotor_input_status",
            "description": "Find the live KOTOR window and report whether it can receive attached-thread input.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "default": "k2"},
                    "window_title": {"type": "string", "description": "Optional exact/partial window title"},
                },
            },
        },
        {
            "name": "kotor_input_click",
            "description": "Send a focused Win32 mouse click to the KOTOR window.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "default": "k2"},
                    "path": {"type": "string", "description": "Optional absolute installation path"},
                    "window_title": {"type": "string", "description": "Optional exact/partial window title"},
                    "use_dinput_hook": {"type": "boolean", "default": False},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "coordinate_space": {
                        "type": "string",
                        "enum": ["ratio", "window", "client", "screen"],
                        "default": "ratio",
                    },
                    "clicks": {"type": "integer", "default": 1},
                    "delay_seconds": {"type": "number", "default": 0.5},
                },
                "required": ["x", "y"],
            },
        },
        {
            "name": "kotor_input_type",
            "description": "Type scancode text into KOTOR, optionally opening the hidden console and pressing Enter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "default": "k2"},
                    "path": {"type": "string", "description": "Optional absolute installation path"},
                    "window_title": {"type": "string", "description": "Optional exact/partial window title"},
                    "use_dinput_hook": {"type": "boolean", "default": False},
                    "text": {"type": "string"},
                    "open_console": {"type": "boolean", "default": False},
                    "press_enter": {"type": "boolean", "default": False},
                    "key_delay_seconds": {"type": "number", "default": 0.035},
                },
                "required": ["text"],
            },
        },
        {
            "name": "kotor_capture_window",
            "description": "Activate the live KOTOR process and save a PNG cropped to the game window/client area.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "default": "k2"},
                    "window_title": {"type": "string", "description": "Optional exact/partial window title"},
                    "output_path": {"type": "string"},
                    "region": {"type": "string", "enum": ["client", "window"], "default": "client"},
                    "activate": {"type": "boolean", "default": True},
                    "clip_to_work_area": {"type": "boolean", "default": True},
                    "settle_seconds": {"type": "number", "default": 0.25},
                },
                "required": ["output_path"],
            },
        },
        {
            "name": "kotor_run_save_warp_route",
            "description": (
                "Drive KOTOR from the main menu or load-game screen: load a save row, "
                "then type the hidden-console warp command."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "game": {"type": "string", "default": "k2"},
                    "path": {"type": "string", "description": "Optional absolute installation path"},
                    "window_title": {"type": "string", "description": "Optional exact/partial window title"},
                    "use_dinput_hook": {"type": "boolean", "default": False},
                    "target_module": {"type": "string", "default": "tst_light"},
                    "start_screen": {
                        "type": "string",
                        "enum": ["main_menu", "load_screen", "in_game"],
                        "default": "main_menu",
                    },
                    "save_row_index": {"type": "integer", "default": 1},
                    "main_menu_load_x_ratio": {"type": "number", "default": 0.604},
                    "main_menu_load_y_ratio": {"type": "number", "default": 0.547},
                    "save_row_x_ratio": {"type": "number", "default": 0.302},
                    "save_row_first_y_ratio": {"type": "number", "default": 0.266},
                    "save_row_step_ratio": {"type": "number", "default": 0.039},
                    "load_button_x_ratio": {"type": "number", "default": 0.334},
                    "load_button_y_ratio": {"type": "number", "default": 0.882},
                    "after_menu_seconds": {"type": "number", "default": 2.0},
                    "after_load_seconds": {"type": "number", "default": 12.0},
                    "after_warp_seconds": {"type": "number", "default": 15.0},
                },
            },
        },
    ]


async def handle_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = KotorInputStatusInput.model_validate(arguments)
    controller = _controller(inp)
    return json_content(controller.status())


async def handle_click(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = KotorInputClickInput.model_validate(arguments)
    try:
        controller = _controller(inp)
        return json_content(
            controller.click(
                float(inp.x),
                float(inp.y),
                coordinate_space=str(inp.coordinate_space or "ratio"),
                clicks=int(inp.clicks or 1),
                delay_seconds=float(inp.delay_seconds or 0.5),
            )
        )
    except KotorInputError as exc:
        return json_content({"ok": False, "error": str(exc)})


async def handle_type(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = KotorInputTypeInput.model_validate(arguments)
    try:
        controller = _controller(inp)
        return json_content(
            controller.type_text(
                str(inp.text or ""),
                open_console=bool(inp.open_console),
                press_enter=bool(inp.press_enter),
                key_delay_seconds=float(inp.key_delay_seconds or 0.035),
            )
        )
    except KotorInputError as exc:
        return json_content({"ok": False, "error": str(exc)})


async def handle_capture_window(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = KotorCaptureWindowInput.model_validate(arguments)
    try:
        controller = _controller(inp)
        return json_content(
            controller.capture_window(
                str(inp.output_path),
                region=str(inp.region or "client"),
                activate=bool(inp.activate),
                clip_to_work_area=bool(
                    inp.clip_to_work_area if inp.clip_to_work_area is not None else True
                ),
                settle_seconds=float(inp.settle_seconds or 0.25),
            )
        )
    except KotorInputError as exc:
        return json_content({"ok": False, "error": str(exc)})


async def handle_run_save_warp_route(arguments: Dict[str, Any]) -> Dict[str, Any]:
    inp = KotorRunSaveWarpRouteInput.model_validate(arguments)
    try:
        controller = _controller(inp)
        return json_content(
            run_save_warp_route(
                controller,
                target_module=str(inp.target_module or "tst_light"),
                start_screen=str(inp.start_screen or "main_menu"),
                save_row_index=int(inp.save_row_index or 0),
                main_menu_load_ratio=(
                    float(inp.main_menu_load_x_ratio or 0.604),
                    float(inp.main_menu_load_y_ratio or 0.547),
                ),
                save_row_ratio=(
                    float(inp.save_row_x_ratio or 0.302),
                    float(inp.save_row_first_y_ratio or 0.266),
                ),
                save_row_step_ratio=float(inp.save_row_step_ratio or 0.039),
                load_button_ratio=(
                    float(inp.load_button_x_ratio or 0.334),
                    float(inp.load_button_y_ratio or 0.882),
                ),
                after_menu_seconds=float(inp.after_menu_seconds or 2.0),
                after_load_seconds=float(inp.after_load_seconds or 12.0),
                after_warp_seconds=float(inp.after_warp_seconds or 15.0),
            )
        )
    except KotorInputError as exc:
        return json_content({"ok": False, "error": str(exc)})


def _controller(inp: Any) -> KotorWindowInput:
    install_path = _install_path(inp)
    return KotorWindowInput(
        window_title=getattr(inp, "window_title", None),
        game=getattr(inp, "game", None) or "k2",
        install_path=install_path,
        # The DirectInput proxy command file lives in the same installation the
        # window identity is bound to; there is no separate hook root.
        dinput_hook_root=install_path if bool(getattr(inp, "use_dinput_hook", False)) else None,
    )


def _install_path(inp: Any) -> str | None:
    """Resolve the absolute installation whose game executable identifies the
    window.  An explicit path takes precedence; otherwise the game alias is
    resolved through the installation registry.  Returns None when no
    installation can be resolved (the action tools then refuse; status reports
    the identity as unavailable rather than matching a look-alike window)."""

    explicit = getattr(inp, "path", None)
    if explicit:
        return str(explicit)
    game = resolve_game(getattr(inp, "game", None) or "k2")
    if game is None:
        return None
    try:
        installation = load_installation(game, None)
    except Exception:
        return None
    return str(installation.path())
