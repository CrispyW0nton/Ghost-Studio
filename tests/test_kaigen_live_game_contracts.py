"""Offline bounded-contract checks for the KOTOR live-game tools Kaigen wraps.

The running-game effects cannot be proven without KOTOR, but the security fixes
are offline-testable: the window/pid must be the installation's swkotor/swkotor2
process image (never a title match), the log helper refuses to debug-attach a pid
that is not that executable, prepare_save_warp_test validates target_module and
bounds its save scan, and the DirectInput proxy command file only accepts
allowlisted verbs. No game is launched and no real installation is required.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Win32 input/identity helpers are Windows-only")


# ── window process identity (game_input) ─────────────────────────────────────

def test_expected_exe_name_maps_game_alias():
    from kotormcp.game_input import expected_exe_name

    assert expected_exe_name("k2") == "swkotor2.exe"
    assert expected_exe_name("tsl") == "swkotor2.exe"
    assert expected_exe_name("k1") == "swkotor.exe"
    assert expected_exe_name("swkotor") == "swkotor.exe"


def test_controller_binds_expected_executable(tmp_path):
    from kotormcp.game_input import KotorWindowInput, _normcase_path

    install = tmp_path / "install"
    install.mkdir()
    (install / "swkotor2.exe").write_bytes(b"MZ")
    c = KotorWindowInput(game="k2", install_path=str(install))
    assert c.expected_exe_path == _normcase_path(str(install / "swkotor2.exe"))


def test_require_window_refuses_without_installation():
    from kotormcp.game_input import KotorWindowInput, KotorInputError

    c = KotorWindowInput(game="k2", install_path=None)
    with pytest.raises(KotorInputError):
        c.require_window()
    status = c.status()
    assert status["window_found"] is False and status["identity_ok"] is False


def test_identity_matches_uses_real_process_image(tmp_path, monkeypatch):
    from kotormcp.game_input import KotorWindowInput, _normcase_path

    install = tmp_path / "install"
    install.mkdir()
    (install / "swkotor2.exe").write_bytes(b"MZ")
    c = KotorWindowInput(game="k2", install_path=str(install))
    # The current process image is read through the same QueryFullProcessImageNameW
    # path the window identity uses.
    image = c.process_image_path(os.getpid())
    assert image and "python" in os.path.basename(image).lower()
    monkeypatch.setattr(c, "window_pid", lambda hwnd: os.getpid())
    # A window owned by this image matches only when it is the expected exe.
    c.expected_exe_path = _normcase_path(image)
    assert c._identity_matches(0x1234) is True
    c.expected_exe_path = _normcase_path(str(install / "swkotor2.exe"))
    assert c._identity_matches(0x1234) is False


# ── log helper pid identity (game_process_log) ───────────────────────────────

class _FakeProbe:
    def __init__(self, exists=True, image=None):
        self._exists = exists
        self._image = image

    def process_exists(self, pid):
        return self._exists

    def process_image_path(self, pid):
        return self._image


def test_verify_pid_identity_refuses_non_kotor_process():
    from kotormcp.game_process_log import verify_pid_identity

    probe = _FakeProbe(exists=True, image=r"C:\apps\LM Studio\lmstudio.exe")
    with pytest.raises(ValueError, match="refusing to debug-attach"):
        verify_pid_identity(4321, game="k2", game_root=None, probe=probe)


def test_verify_pid_identity_refuses_dead_pid():
    from kotormcp.game_process_log import verify_pid_identity

    with pytest.raises(ValueError, match="not a running process"):
        verify_pid_identity(999999, game="k2", game_root=None, probe=_FakeProbe(exists=False))


def test_verify_pid_identity_enforces_install_path(tmp_path):
    from kotormcp.game_process_log import verify_pid_identity

    root = tmp_path / "k2"
    root.mkdir()
    exe = root / "swkotor2.exe"
    exe.write_bytes(b"MZ")
    ok = verify_pid_identity(10, game="k2", game_root=str(root), probe=_FakeProbe(image=str(exe)))
    assert ok["exe_name"] == "swkotor2.exe"
    # Right executable name but a different installation directory is refused.
    other = tmp_path / "other" / "swkotor2.exe"
    other.parent.mkdir()
    other.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="not the installation executable"):
        verify_pid_identity(10, game="k2", game_root=str(root), probe=_FakeProbe(image=str(other)))


# ── prepare save->warp test guards (game_test) ───────────────────────────────

def test_target_module_regex_rejects_traversal():
    from kotormcp.tools.game_test import _clean_module_root

    assert re.fullmatch(r"[a-z0-9_]{1,32}", _clean_module_root("tst_light.mod")) is not None
    assert re.fullmatch(r"[a-z0-9_]{1,32}", _clean_module_root("../evil")) is None
    assert re.fullmatch(r"[a-z0-9_]{1,32}", _clean_module_root("a/b")) is None


def test_save_scan_is_depth_and_count_bounded(tmp_path):
    from kotormcp.tools import game_test

    # A deeply nested save beyond the depth bound is not discovered.
    deep = tmp_path
    for i in range(game_test.SAVE_SCAN_MAX_DEPTH + 2):
        deep = deep / f"d{i}"
    deep.mkdir(parents=True)
    (deep / "SAVEGAME.sav").write_bytes(b"x")
    found = list(game_test._iter_savegames(tmp_path))
    assert found == []
    # A save within the depth bound is discovered.
    shallow = tmp_path / "saveA"
    shallow.mkdir()
    (shallow / "SAVEGAME.sav").write_bytes(b"x")
    assert any(p.parent.name == "saveA" for p in game_test._iter_savegames(tmp_path))


# ── DirectInput proxy command allowlist (game_dinput_hook) ───────────────────

def test_proxy_command_allowlist():
    from kotormcp.game_dinput_hook import _clean_command

    assert _clean_command("key_tap 0x1c 35") == "key_tap 0x1c 35"
    assert _clean_command("# comment") == ""
    with pytest.raises(ValueError, match="Unsupported"):
        _clean_command("format_disk 1")
    with pytest.raises(ValueError, match="one line"):
        _clean_command("key_tap 1\nmouse_click 2")


def test_install_hook_copies_and_backs_up(tmp_path):
    from kotormcp.game_dinput_hook import install_hook, describe_hook

    root = tmp_path / "k2"
    root.mkdir()
    proxy = tmp_path / "proxy.dll"
    proxy.write_bytes(b"PROXY-BUILD-BYTES")
    # Fresh install.
    result = install_hook(str(root), game="k2", proxy_path=str(proxy))
    assert result["ok"] and (root / "dinput8.dll").read_bytes() == b"PROXY-BUILD-BYTES"
    assert describe_hook(str(root), proxy_path=str(proxy))["installed"] is True
    # A different existing dll requires force and is backed up.
    (root / "dinput8.dll").write_bytes(b"OTHER")
    blocked = install_hook(str(root), game="k2", proxy_path=str(proxy))
    assert blocked["ok"] is False
    backup_root = tmp_path / "backups"
    forced = install_hook(str(root), game="k2", proxy_path=str(proxy), backup_root=str(backup_root), force=True)
    assert forced["ok"] and (root / "dinput8.dll").read_bytes() == b"PROXY-BUILD-BYTES"
    assert Path(forced["backup_path"]).read_bytes() == b"OTHER"
