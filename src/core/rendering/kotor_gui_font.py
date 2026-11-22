"""Decode a retail Odyssey PC GUI bitmap font into top-left atlas coordinates."""

from dataclasses import dataclass

from PIL import Image
from pykotor.resource.formats.tpc import read_tpc
from pykotor.resource.formats.tpc.tpc_data import TPCTextureFormat
from pykotor.resource.formats.txi.txi_data import TXI, TXIFontInformation
from src.core.rendering.kotor_gui_preview import decode_kotor_gui_texture


@dataclass(frozen=True)
class GuiFont:
    image: Image.Image
    glyphs: tuple[tuple[int, int, int, int], ...]
    font_height: float
    baseline: float
    spacing_right: float
    spacing_bottom: float
    texture_width: float


def decode_kotor_gui_font(data: bytes, txi: str = "") -> GuiFont:
    """Read the PC atlas, including retail coordinate blocks without inline counts."""
    atlas = decode_kotor_gui_texture(data, max_size=4096)
    lines = (txi or atlas.txi).splitlines()
    count = next((int(parts[1]) for line in lines if (parts := line.lower().split())
                  and parts[0] == "numchars" and len(parts) == 2), 0)
    if not 1 <= count <= 256:
        raise ValueError(f"Invalid PC GUI font character count {count}")
    normalized = "\n".join(f"{line.strip()} {count}" if line.strip().lower() in
                           {"upperleftcoords", "lowerrightcoords"} else line for line in lines)
    fields = TXI(normalized).features
    left, right = fields.upperleftcoords or [], fields.lowerrightcoords or []
    if len(left) != count or len(right) != count:
        raise ValueError("Incomplete PC GUI font TXI coordinate blocks")
    image = Image.frombytes("RGBA", (atlas.width, atlas.height), atlas.rgba)
    source_format = read_tpc(data).format()
    # The installed decoder's DXT5 font atlases need the opposite row order from
    # generic GUI textures. RGBA PC font atlases already match their TXI rows.
    if source_format == TPCTextureFormat.DXT5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # Use the upstream transform and round normalized values back to nearest texels.
    # Rounding avoids dropping a pixel when the text rounded its normalized UV.
    coords = TXIFontInformation().coords_from_normalized(
        left, right, (image.width * 1024, image.height * 1024)
    )
    glyphs = tuple(tuple(round(v / 1024) for v in rect) for rect in coords)
    return GuiFont(image, glyphs, float(fields.fontheight), float(fields.baselineheight),
                   float(fields.spacingR or 0), float(fields.spacingB or 0),
                   float(fields.texturewidth))
