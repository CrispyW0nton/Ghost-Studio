"""Bounded-contract checks for the game-data tools Kaigen wraps (2026-09-30).

Fast and offline: fixture installations under tmp_path only (an Override
lookup never parses the fixture chitin.key); no real game install is read.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path


def _call(name, arguments):
    from kotormcp.tools import handle_tool

    raw = asyncio.run(handle_tool(name, arguments))
    return raw, json.loads(raw["text"])


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _game(tmp_path: Path, game: str = "k1") -> Path:
    root = tmp_path / f"game-{game}"
    (root / "Override").mkdir(parents=True)
    (root / ("swkotor2.exe" if game == "k2" else "swkotor.exe")).write_bytes(b"fixture executable")
    (root / "chitin.key").write_bytes(b"KEY V1  fixture")
    return root


def test_extract_reports_location_and_writes_exclusively(tmp_path):
    game = _game(tmp_path)
    data = b"2DA V2.b\n\nlabel\n0 fixture\n"
    (game / "Override" / "appearance.2da").write_bytes(data)
    out = tmp_path / "out"
    out.mkdir()
    base = {"game": "k1", "resref": "Appearance", "restype": "2da", "game_path": str(game)}

    raw, preview = _call("kotor_extract_resource", {**base, "dry_run": True})
    assert not raw.get("isError")
    assert preview["location"] == "override"
    assert preview["source_file"] == "Override/appearance.2da"
    assert preview["bytes"] == len(data) and preview["sha256"] == _sha(data)
    assert list(out.iterdir()) == []

    raw, written = _call("kotor_extract_resource", {**base, "output_dir": str(out)})
    assert not raw.get("isError")
    assert Path(written["path"]) == out / "appearance.2da"
    assert (out / "appearance.2da").read_bytes() == data
    assert written["sha256"] == preview["sha256"]

    raw, again = _call("kotor_extract_resource", {**base, "output_dir": str(out)})
    assert raw.get("isError") is True and "already exists" in again["error"]
    assert (out / "appearance.2da").read_bytes() == data


def test_extract_refuses_unbounded_requests_with_valid_json(tmp_path):
    game = _game(tmp_path)
    (game / "Override" / "appearance.2da").write_bytes(b"2DA V2.b\n")
    out = tmp_path / "out"
    out.mkdir()
    base = {"game": "k1", "resref": "appearance", "restype": "2da", "game_path": str(game)}
    refusals = [
        ({**base, "output_dir": str(game / "Override")}, "outside the game installation"),
        ({**base, "output_dir": "relative\\dir"}, "absolute"),
        ({**base}, "exactly one of output_dir or output_path"),
        ({**base, "output_dir": str(out), "output_path": str(out / "x.2da")}, "exactly one"),
        ({**base, "resref": "..\\evil", "dry_run": True}, "resref"),
        ({**base, "restype": "zzz", "dry_run": True}, "Unknown resource type"),
        ({**base, "source": "SAVES", "dry_run": True}, "Unknown resource source"),
        ({**base, "dry_run": "yes"}, "boolean"),
        ({**base, "extra": 1, "dry_run": True}, "Unsupported argument"),
        ({**base, "game": "k2", "dry_run": True}, "not a K2 installation"),
        ({**base, "game_path": "C:\\no\\such\\kotor", "dry_run": True}, ""),
        ({**base, "resref": "missing", "source": "OVERRIDE", "dry_run": True}, "not found"),
    ]
    for arguments, expected in refusals:
        raw, data = _call("kotor_extract_resource", arguments)
        assert raw.get("isError") is True, arguments
        assert expected in data["error"], (arguments, data)
    assert list(out.iterdir()) == []


def _install_args(game: Path, workspace: Path, backup: Path, data: bytes, **extra):
    return {
        "game": "k1",
        "resref": "kaigen_fixture",
        "restype": "2da",
        "data_b64": base64.b64encode(data).decode("ascii"),
        "workspace_root": str(workspace),
        "game_root": str(game),
        "confirm_install": True,
        "backup_root": str(backup),
        **extra,
    }


def test_override_install_uses_external_backup_and_compare_and_swap(tmp_path):
    game = _game(tmp_path)
    workspace, backup = tmp_path / "workspace", tmp_path / "backup"
    workspace.mkdir()
    backup.mkdir()
    target = game / "Override" / "kaigen_fixture.2da"
    first, second = b"2DA V2.b\nfirst\n", b"2DA V2.b\nsecond\n"

    raw, stale = _call("writeOverride", _install_args(game, workspace, backup, first, expect_existing="0" * 64))
    assert raw.get("isError") is True and "changed since" in stale["error"]
    assert not target.exists()

    raw, created = _call("writeOverride", _install_args(game, workspace, backup, first, expect_existing="absent"))
    assert not raw.get("isError"), created
    assert target.read_bytes() == first
    assert created["backup_location"] == "external"
    assert sorted(p.name for p in game.iterdir()) == ["Override", "chitin.key", "swkotor.exe"]
    receipt = json.loads(Path(created["receipt_path"]).read_text(encoding="utf-8"))
    assert Path(created["receipt_path"]).parent.parent == backup
    assert receipt["installed_sha256"] == {"kaigen_fixture.2da": _sha(first)}

    raw, blocked = _call("writeOverride", _install_args(game, workspace, backup, second, expect_existing=_sha(first)))
    assert raw.get("isError") is True and "backup policy" in blocked["error"]
    raw, replaced = _call(
        "writeOverride",
        _install_args(game, workspace, backup, second, expect_existing=_sha(first), on_conflict="backup"),
    )
    assert not raw.get("isError"), replaced
    assert target.read_bytes() == second
    assert (Path(replaced["backup_path"]) / "Override" / "kaigen_fixture.2da").read_bytes() == first
    receipt = json.loads(Path(replaced["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["backed_up_sha256"] == {"kaigen_fixture.2da": _sha(first)}
    assert not list(game.glob(".ghoststudio-override-install-*"))
    assert not (game / "GhostStudioBackups").exists()

    raw, inside = _call("writeOverride", _install_args(game, workspace, game / "Override", first, on_conflict="backup"))
    assert raw.get("isError") is True and "separate directory trees" in inside["error"]
    assert target.read_bytes() == second


def test_override_restore_is_compare_and_swap(tmp_path):
    game = _game(tmp_path)
    workspace, backup, undo = tmp_path / "workspace", tmp_path / "backup", tmp_path / "undo"
    for folder in (workspace, backup, undo):
        folder.mkdir()
    target = game / "Override" / "kaigen_fixture.2da"
    first, second = b"2DA V2.b\nfirst\n", b"2DA V2.b\nsecond\n"
    assert not _call("writeOverride", _install_args(game, workspace, backup, first))[0].get("isError")
    assert not _call("writeOverride", _install_args(game, workspace, backup, second, on_conflict="backup"))[0].get("isError")
    restore = {
        "game": "k1",
        "resref": "kaigen_fixture",
        "restype": "2da",
        "game_root": str(game),
        "backup_root": str(undo),
        "expect_current": _sha(second),
        "confirm_restore": True,
    }

    for arguments, expected in [
        ({**restore, "prior_data_b64": base64.b64encode(first).decode(), "confirm_restore": "yes"}, "confirm_restore"),
        ({**restore}, "exactly one"),
        ({**restore, "prior_data_b64": "", "remove_file": True}, "exactly one"),
        ({**restore, "remove_file": True, "backup_root": str(game)}, "separate directory trees"),
        ({**restore, "remove_file": True, "expect_current": _sha(first)}, "changed since it was installed"),
    ]:
        raw, data = _call("ghoststudio_restore_override", arguments)
        assert raw.get("isError") is True and expected in data["error"], (arguments, data)
        assert target.read_bytes() == second

    raw, restored = _call("ghoststudio_restore_override", {**restore, "prior_data_b64": base64.b64encode(first).decode()})
    assert not raw.get("isError"), restored
    assert restored["action"] == "restored" and target.read_bytes() == first
    assert (Path(restored["backup_path"]) / "Override" / "kaigen_fixture.2da").read_bytes() == second
    assert json.loads(Path(restored["receipt_path"]).read_text(encoding="utf-8"))["pre_restore_sha256"] == _sha(second)

    raw, again = _call("ghoststudio_restore_override", {**restore, "prior_data_b64": base64.b64encode(first).decode()})
    assert raw.get("isError") is True and "changed since it was installed" in again["error"]
    assert target.read_bytes() == first

    raw, removed = _call("ghoststudio_restore_override", {**restore, "expect_current": _sha(first), "remove_file": True})
    assert not raw.get("isError"), removed
    assert removed["action"] == "removed" and not target.exists()
    assert (Path(removed["backup_path"]) / "Override" / "kaigen_fixture.2da").read_bytes() == first
