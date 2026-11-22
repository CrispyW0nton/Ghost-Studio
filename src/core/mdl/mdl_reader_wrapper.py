"""
GhostRigger wrapper around PyKotor MDL parsing.

All MDL/MDX loads that should respect GhostRigger fixes must go through
``read_mdl_safe`` instead of importing ``read_mdl`` from ``mdl_auto`` directly.

Binary MDL/MDX files are loaded through ``GhostRiggerMDLBinaryReader`` so
GhostRigger owns its K2 layout fixes without mutating PyKotor global state.
ASCII MDL files still use PyKotor's ASCII reader directly.

Fixes:

- **K2 trimesh tail** — correct 8-byte dirt/hologram block and padding so
  ``mdx_data_offset`` / ``vertices_offset`` align with KotOR.js / KotorBlender.
- **``mdx_data_offset == 0``** — PyKotor treats 0 like ``0xFFFFFFFF``; offset 0 is
  valid (vertex data at the start of the MDX buffer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.resource.formats.mdl.io_mdl_ascii import MDLAsciiReader
from pykotor.resource.formats.mdl.mdl_auto import detect_mdl
from pykotor.resource.type import ResourceType

from .ghostrigger_mdl_reader import GhostRiggerMDLBinaryReader

if TYPE_CHECKING:
    from pykotor.resource.formats.mdl.mdl_data import MDL
    from pykotor.resource.type import SOURCE_TYPES


def read_mdl_safe(
    source: "SOURCE_TYPES",
    offset: int = 0,
    size: int | None = None,
    source_ext: "SOURCE_TYPES | None" = None,
    offset_ext: int = 0,
    size_ext: int = 0,
    file_format: "ResourceType | None" = None,
    skip_aabb: bool = False,
) -> "MDL":
    """Parse MDL (+ optional MDX) through GhostRigger's owned safe reader path."""
    resolved_format = file_format or detect_mdl(source, offset)
    if resolved_format == ResourceType.MDL:
        try:
            return GhostRiggerMDLBinaryReader(
                source,
                offset,
                size or 0,
                source_ext,
                offset_ext,
                size_ext,
                skip_aabb=skip_aabb,
            ).load()
        except OSError as exc:
            if not _is_mdl_aabb_seek_oserror(exc):
                raise
            return GhostRiggerMDLBinaryReader(
                source,
                offset,
                size or 0,
                source_ext,
                offset_ext,
                size_ext,
                skip_aabb=True,
            ).load()
    if resolved_format == ResourceType.MDL_ASCII:
        return MDLAsciiReader(source, offset, size or 0).load()

    raise ValueError("Failed to determine the format of the MDL file.")


def _is_mdl_aabb_seek_oserror(exc: OSError) -> bool:
    msg = str(exc).lower()
    return "seek" in msg and ("negative" in msg or "cannot seek" in msg)
