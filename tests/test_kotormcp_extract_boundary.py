from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from kotormcp.utils.path_safety import get_extract_base, resolve_and_validate_under_base


def test_configured_boundary_and_nonexistent_child(tmp_path, monkeypatch):
    base = tmp_path / "new_extract"
    monkeypatch.setenv("PYKOTOR_EXTRACT_DIR", str(base))
    assert get_extract_base() == base.resolve()
    target = base / "nested" / "appearance.2da"
    assert resolve_and_validate_under_base(target, base) == target.resolve()
    # A missing extraction directory must not authorize its parent.
    with pytest.raises(ValueError, match="outside allowed base"):
        resolve_and_validate_under_base(tmp_path / "outside.2da", base)


@pytest.mark.parametrize("target", ["../outside.2da", "../extract_sibling/file.2da"])
def test_traversal_and_sibling_prefix_rejected(tmp_path, target):
    base = tmp_path / "extract"
    base.mkdir()
    with pytest.raises(ValueError, match="outside allowed base"):
        resolve_and_validate_under_base(base / target, base)


def test_symlink_escape_rejected(tmp_path):
    base = tmp_path / "extract"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="outside allowed base"):
        resolve_and_validate_under_base(base / "link" / "file.2da", base)


def test_extract_without_newer_pykotor_helper(tmp_path, monkeypatch):
    from kotormcp.tools import archives

    monkeypatch.setitem(sys.modules, "pykotor.tools.path_safety", None)
    monkeypatch.setenv("PYKOTOR_EXTRACT_DIR", str(tmp_path))
    monkeypatch.setattr(archives, "resolve_game", lambda _: object())
    calls = []

    def get_resource(resref, restype, order=None):
        calls.append((resref, restype, order))
        if restype != "2da":
            return None
        return SimpleNamespace(data=b"stock bytes")

    monkeypatch.setattr(archives, "load_installation", lambda _: SimpleNamespace(get_resource=get_resource))
    target = tmp_path / "stock.2da"
    result = asyncio.run(archives.handle_extract_resource({
        "game": "k2", "resref": "appearance", "restype": "2da",
        "source": "CHITIN", "output_path": str(target),
    }))
    assert json.loads(result["text"])["status"] == "ok"
    assert target.read_bytes() == b"stock bytes"
    assert calls == [("appearance", "2da", ["CHITIN"])]
