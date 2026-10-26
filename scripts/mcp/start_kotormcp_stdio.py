"""Start the in-repo KotorMCP server over stdio for Codex.

Codex launches this file from ``~/.codex/config.toml``.  Keep stdout reserved
for MCP protocol frames; diagnostics belong on stderr inside ``kotormcp.server``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)


def _python_roots(root: Path) -> list[Path]:
    """Return import roots for GhostRigger's native-split Python packages."""
    roots: list[Path] = []
    for project_root in sorted((root / "native").glob("GhostRigger.*")):
        python_root = project_root / "Python"
        src_root = python_root / "src"
        if (src_root / "__init__.py").exists():
            continue
        if src_root.exists():
            roots.append(src_root)
        if python_root.exists():
            roots.append(python_root)
    legacy_src = root / "src"
    if legacy_src.exists():
        roots.append(legacy_src)
    roots.append(root)
    return roots


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    wanted = _python_roots(root)
    for path in reversed(wanted):
        _prepend_path(path)

    current = os.environ.get("PYTHONPATH", "")
    wanted_text = [str(path) for path in wanted if path.exists()]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        item for item in wanted_text + ([current] if current else []) if item
    )
    os.environ.setdefault("GHOSTRIGGER_ROOT", str(root))

    # Initialize the renderer's native numeric dependency before AnyIO starts
    # its stdio worker threads. On Windows, a first NumPy import inside a tool
    # call can stall in extension initialization while those workers are active.
    import numpy  # noqa: F401

    from kotormcp.server import main as server_main

    server_main(["--mode", "stdio"])


if __name__ == "__main__":
    main()
