"""Block-compressed KOTOR TPC writer (DXT1 / DXT5 with full mip chains).

Added 2026-09-28 from the DXM_101 (Dathomir) in-game test. Ghost Studio decodes DXT but had no DXT writer, and
PyKotor's ``TPC.convert(TPCTextureFormat.DXT1)`` is not usable for authoring: on a 2048 x 2048 albedo page it wrote a
colour-shifted base level (mean RGB 34/45/14 for an 84/66/41 source) and solid black for every smaller mip. In game
that read as green, blocky textures near the camera and black surfaces farther away.

This writer encodes each level with Pillow's DDS BC1 / BC3 encoder (round-trip error ~6/255 on rock albedo) from a
box-filtered chain down to 1 x 1 - levels under 4 x 4 are padded to one block, as vanilla stores them - and writes the
vanilla header: ``data_size`` = base-level bytes, ``alpha_test`` 1.0, width, height, encoding 2 (DXT1) or 4 (DXT5),
mip count, zero padding to 128 bytes. An optional TXI trailer follows the pixel data.

Orientation: rows are stored top to bottom exactly as in the source image. The engine samples a TPC's first stored row
at MDX UV v = 0 (a raw decode of a vanilla TPC reads upright), so a surface whose UVs use a bottom-left origin
(v = 1 at the image's top row) must be exported with v -> 1 - v (``uv_v_flip=False`` / ``_gr_mdx_uv_transform='flip_v'``
on the node, or flipped by the caller).
"""

from __future__ import annotations

import io
import struct
from typing import Any

TPC_ENCODING_DXT1 = 2
TPC_ENCODING_DXT5 = 4


def _dds_blocks(image: Any, pixel_format: str) -> bytes:
    from PIL import Image

    width, height = image.size
    if width < 4 or height < 4:
        image = image.resize((max(4, width), max(4, height)), Image.NEAREST)
    stream = io.BytesIO()
    image.save(stream, format="DDS", pixel_format=pixel_format)
    data = stream.getvalue()
    if data[:4] != b"DDS ":
        raise ValueError("Pillow did not produce a DDS stream.")
    blocks_w, blocks_h = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    need = blocks_w * blocks_h * (8 if pixel_format == "DXT1" else 16)
    body = data[128:128 + need]
    if len(body) != need:
        raise ValueError(f"DDS body is {len(body)} bytes; expected {need}.")
    return body


def encode_tpc_dxt(image: Any, *, alpha: bool = False, txi: bytes = b"") -> bytes:
    """Encode a PIL image as a DXT1 (opaque) or DXT5 (alpha) TPC with every mip level."""

    from PIL import Image

    source = image.convert("RGBA" if alpha else "RGB")
    width, height = source.size
    if width & (width - 1) or height & (height - 1):
        raise ValueError("KOTOR block-compressed textures must have power-of-two dimensions.")
    pixel_format = "DXT5" if alpha else "DXT1"
    levels: list[bytes] = []
    current = source
    while True:
        levels.append(_dds_blocks(current, pixel_format))
        if current.size == (1, 1):
            break
        current = current.resize((max(1, current.size[0] // 2), max(1, current.size[1] // 2)), Image.BOX)
    header = struct.pack(
        "<IfHHBB",
        len(levels[0]),
        1.0,
        width,
        height,
        TPC_ENCODING_DXT5 if alpha else TPC_ENCODING_DXT1,
        len(levels),
    )
    header += b"\x00" * (128 - len(header))
    return header + b"".join(levels) + (txi or b"")


__all__ = ("encode_tpc_dxt", "TPC_ENCODING_DXT1", "TPC_ENCODING_DXT5")
