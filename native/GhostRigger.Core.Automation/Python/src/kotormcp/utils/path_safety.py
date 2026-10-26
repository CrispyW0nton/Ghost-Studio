"""Extraction boundary compatible with the pinned PyKotor 2.3.1 runtime.

Use PyKotor's PYKOTOR_EXTRACT_DIR/CWD contract without depending on the
path_safety module introduced after our pinned release. A nonexistent base
remains the boundary; it must never implicitly widen to its parent.
"""
from __future__ import annotations

import os
from pathlib import Path

MAX_PATH_LENGTH = 4096


def get_extract_base() -> Path:
    configured = os.environ.get("PYKOTOR_EXTRACT_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd()


def resolve_and_validate_under_base(
    user_path: str | Path, base: Path, *, allow_nonexistent: bool = True,
) -> Path:
    """Canonicalize the target and reject traversal and symlink escapes."""
    boundary = Path(base).expanduser().resolve()
    resolved = Path(user_path).expanduser().resolve(strict=not allow_nonexistent)
    if len(str(resolved)) > MAX_PATH_LENGTH:
        raise ValueError(f"Path length exceeds maximum ({MAX_PATH_LENGTH})")
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"Path is outside allowed base: {resolved} not under {boundary}")
    return resolved
