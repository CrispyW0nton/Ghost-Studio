from __future__ import annotations

import struct
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("change", ("library", "game", "installation", "paths", "revision"))
def test_texture_source_change_releases_derived_mips(change, tmp_path):
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    library = object()
    manager = SimpleNamespace(revision=1, get_k1=lambda: None, get_k2=lambda: None)
    cache.set_game_library(library, "K1")
    cache.set_resource_manager(manager, "K1")
    source = Image.new("RGBA", (32, 32), "red")
    cache._cache["test"] = source
    cache.get_mip1(source)
    assert cache._mip_bias_cache

    if change == "library":
        cache.set_game_library(object(), "K1")
    elif change == "game":
        cache.set_game_library(library, "K2")
    elif change == "installation":
        cache.set_installation(object(), "K1")
    elif change == "paths":
        cache.set_search_dirs([str(tmp_path)])
    else:
        manager.revision += 1
        cache.set_resource_manager(manager, "K1")
    assert not cache._mip_bias_cache


def test_derived_mip_budget_preserves_live_previews_without_resize_thrashing(monkeypatch):
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    cache.MAX_MIP_BYTES = 2 * 16 * 16 * 4
    sources = [Image.new("RGBA", (32, 32), color) for color in ("red", "green", "blue")]
    first = cache.get_mip1(sources[0])
    cache.get_mip1(sources[1])
    assert cache.get_mip1(sources[0]) is first
    def unexpected_resize(*args, **kwargs):
        pytest.fail("Full cache must reuse base pixels, not resize on every draw")

    monkeypatch.setattr(sources[2], "resize", unexpected_resize)
    for _ in range(3):
        assert cache.get_mip1(sources[2]) is sources[2]
    assert id(sources[0]) in cache._mip_bias_cache
    assert id(sources[1]) in cache._mip_bias_cache
    assert len(cache._mip_bias_cache) == 2
    assert cache.get_mip1(sources[0]).getpixel((0, 0)) == (255, 0, 0, 255)


def test_derived_mip_cache_caps_tiny_entries_and_skips_oversized_image():
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    cache.MAX_MIP_ENTRIES = 2
    sources = [Image.new("RGBA", (2, 2)) for _ in range(3)]
    for source in sources:
        cache.get_mip1(source)
    assert len(cache._mip_bias_cache) == 2
    cache.MAX_MIP_BYTES = 4
    source = Image.new("RGBA", (32, 32), "red")
    assert cache.get_mip1(source) is source
    assert id(source) not in cache._mip_bias_cache


def test_derived_mip_does_not_reuse_pixels_from_another_source_identity():
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    red = Image.new("RGBA", (32, 32), "red")
    blue = Image.new("RGBA", (32, 32), "blue")
    cache.get_mip1(red)
    # Deterministic simulation of CPython reusing a collected image's address.
    cache._mip_bias_cache[id(blue)] = cache._mip_bias_cache.pop(id(red))
    assert cache.get_mip1(blue).getpixel((0, 0)) == (0, 0, 255, 255)


@pytest.mark.parametrize("invalidate", ("clear", "paint"))
def test_mip_resize_in_flight_cannot_repopulate_invalidated_cache(monkeypatch, invalidate):
    import threading
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    source = Image.new("RGBA", (32, 32), "red")
    cache._cache["painted"] = source
    started, resume = threading.Event(), threading.Event()
    resize = source.resize

    def delayed_resize(*args, **kwargs):
        started.set()
        assert resume.wait(5)
        return resize(*args, **kwargs)

    monkeypatch.setattr(source, "resize", delayed_resize)
    results = []
    worker = threading.Thread(target=lambda: results.append(cache.get_mip1(source)))
    worker.start()
    try:
        assert started.wait(5)
        if invalidate == "paint":
            cache.update_image_regions("painted", Image.new("RGBA", (32, 32), "blue"))
        else:
            cache.clear_mip_cache()
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(results) == 1
    assert not cache._mip_bias_cache


def test_derived_cache_reclaims_collected_sources_and_preserves_published_pixels():
    import gc
    import weakref
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    cache = TextureCache()
    cache.MAX_MIP_ENTRIES = 1
    source = Image.new("RGBA", (32, 32), "red")
    source_ref = weakref.ref(source)
    cache.get_mip1(source)
    del source
    gc.collect()
    assert source_ref() is None
    encoded = BytesIO()
    Image.new("RGBA", (32, 32), "blue").save(encoded, format="PNG")
    published = cache.publish_bytes("authored", encoded.getvalue())
    assert cache.get_mip1(published) is not published
    cache.set_game_library(object(), "K2")
    assert not cache._mip_bias_cache
    assert cache.get("authored") is published
    assert published.getpixel((0, 0)) == (0, 0, 255, 255)


ROOT = Path(__file__).resolve().parents[1]
K2_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Knights of the Old Republic II")


def _configure_python_roots() -> None:
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for item in reversed(_python_roots(ROOT)):
        text = str(item)
        if item.exists() and text not in sys.path:
            sys.path.insert(0, text)


_configure_python_roots()


@pytest.fixture(scope="module")
def installation():
    if not K2_ROOT.is_dir():
        pytest.skip("K2 installation is unavailable")
    from pykotor.extract.installation import Installation

    return Installation(K2_ROOT)


def _raw_tpc(installation, resref: str) -> bytes:
    from pykotor.resource.type import ResourceType

    resource = installation.resource(resref, ResourceType.TPC)
    assert resource is not None, resref
    return bytes(resource.data)


def _authority_selected_mip(raw: bytes, max_size: int):
    from PIL import Image
    from pykotor.resource.formats.tpc import read_tpc
    from pykotor.resource.formats.tpc.tpc_data import TPCTextureFormat

    tpc = read_tpc(raw)
    original_format = tpc.format()
    mipmaps = tpc.layers[0].mipmaps
    index = next(
        (index for index, mip in enumerate(mipmaps) if max(mip.width, mip.height) <= max_size),
        len(mipmaps) - 1,
    )
    mip = mipmaps[index].copy()
    mip.convert(TPCTextureFormat.RGBA)
    image = mip.to_pil_image().convert("RGBA")
    if original_format in {TPCTextureFormat.DXT1, TPCTextureFormat.DXT3, TPCTextureFormat.DXT5}:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return tpc, index, image


@pytest.mark.parametrize(
    ("resref", "format_name", "source_size", "mip_level", "output_size"),
    (
        ("tel_ja11", "DXT1", (2048, 2048), 2, (512, 512)),
        ("tel_hw10", "DXT5", (4096, 4096), 3, (512, 512)),
        ("207tel_1_lm0", "RGBA", (128, 128), 0, (128, 128)),
    ),
)
def test_viewport_decode_matches_direct_pykotor_selected_mip(
    installation,
    resref: str,
    format_name: str,
    source_size: tuple[int, int],
    mip_level: int,
    output_size: tuple[int, int],
) -> None:
    from src.core.graphics.tpc import _load_tpc_bytes

    raw = _raw_tpc(installation, resref)
    tpc, expected_level, authority = _authority_selected_mip(raw, 512)
    image = _load_tpc_bytes(raw, max_size=512)

    assert image is not None
    assert tpc.format().name == format_name
    assert tpc.dimensions() == source_size
    assert expected_level == mip_level
    assert image.size == output_size
    assert image.mode == "RGBA"
    assert image.tobytes() == authority.tobytes()
    assert image._tpc_source_size == source_size
    assert image._tpc_mip_level == mip_level
    assert image._tpc_mip_size == output_size
    assert image._tpc_viewport_max_size == 512
    assert image._gr_gpu_uv_v_flip is True
    assert image._tpc_raw == raw
    assert image._txi_str == str(tpc.txi or "").strip()
    alpha_test = struct.unpack_from("<f", raw, 4)[0]
    assert image._txi_alpha_test == pytest.approx(alpha_test)


def test_default_loader_preserves_full_resolution_contract(installation) -> None:
    from PIL import Image
    from pykotor.resource.formats.tpc import read_tpc
    from pykotor.resource.formats.tpc.tpc_data import TPCTextureFormat
    from src.core.graphics.tpc import _load_tpc_bytes

    raw = _raw_tpc(installation, "plc_chair1")
    tpc = read_tpc(raw)
    tpc.convert(TPCTextureFormat.RGBA)
    authority = tpc.get(0, 0).to_pil_image().convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image = _load_tpc_bytes(raw)

    assert image is not None
    assert image.size == (512, 512)
    assert image.tobytes() == authority.tobytes()
    assert image._tpc_mip_level == 0
    assert image._tpc_viewport_max_size is None


def test_texture_cache_uses_authored_mip_and_preserves_alpha_metadata(installation) -> None:
    from src.core.graphics.txi import _parse_txi_string
    from src.core.rendering.frame_core.texture_cache import TextureCache

    raw = _raw_tpc(installation, "n_commf01")
    cache = TextureCache()
    image = cache._load_bytes(raw)
    assert image is not None
    assert image.size == (512, 512)
    assert image._tpc_source_size == (1024, 1024)
    assert image._tpc_mip_level == 1
    assert image._tpc_raw == raw
    assert image._txi_str == ""
    assert image._txi_alpha_test == pytest.approx(1.0)

    processed = cache._apply_kotor_alpha(raw, image, _parse_txi_string(image._txi_str))
    assert processed.size == image.size
    assert processed._tpc_source_size == (1024, 1024)
    assert processed._tpc_mip_level == 1
    assert processed._tpc_raw == raw
    assert processed._txi_alpha_test == pytest.approx(1.0)


def test_texture_cache_normalizes_loose_tga_for_kotor_uv_sampling() -> None:
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    # Source/Pillow row order is top-down: red+green above blue+yellow.
    source = Image.new("RGBA", (2, 2))
    source.putdata(
        (
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
        )
    )
    encoded = BytesIO()
    source.save(encoded, format="TGA")

    image = TextureCache()._load_bytes(encoded.getvalue())

    assert image is not None
    # GPU upload rows are bottom-up after normalization.
    assert tuple(image.get_flattened_data()) == (
        (0, 0, 255, 255),
        (255, 255, 0, 255),
        (255, 0, 0, 255),
        (0, 255, 0, 255),
    )
    # KOTOR binary UV V=0 means top, so the shader must convert it to GL V=1.
    # Imported DCC nodes remain correct because they carry uv_v_flip=False.
    assert image._gr_gpu_uv_v_flip is True


def test_texture_cache_tga_storage_origin_does_not_change_gpu_rows() -> None:
    """TGA storage origin is independent from model UV provenance.

    Xaria's body package uses a bottom-origin 24-bit TGA while the verified
    head package uses a top-origin TGA. Pillow normalizes both to logical
    top-down rows; TextureCache must then produce one identical bottom-up GPU
    payload for both instead of treating the header bit as a DCC UV hint.
    """

    from PIL import Image
    from src.adapters.rendering.moderngl_renderer_impl import _effective_diffuse_uv_v_flip
    from src.core.rendering.frame_core.texture_cache import TextureCache

    source = Image.new("RGB", (2, 2))
    source.putdata(
        (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        )
    )
    encoded = BytesIO()
    source.save(encoded, format="TGA")
    bottom_origin = bytearray(encoded.getvalue())
    assert bottom_origin[2] == 2
    assert bottom_origin[16] == 24
    assert bottom_origin[17] & 0x20 == 0

    # Convert the same logical image to top-origin storage by reversing the
    # encoded scanlines and setting the TGA descriptor bit.
    width = int.from_bytes(bottom_origin[12:14], "little")
    height = int.from_bytes(bottom_origin[14:16], "little")
    row_bytes = width * (bottom_origin[16] // 8)
    pixel_start = 18 + int(bottom_origin[0])
    pixel_end = pixel_start + row_bytes * height
    rows = [
        bytes(bottom_origin[pixel_start + row * row_bytes:pixel_start + (row + 1) * row_bytes])
        for row in range(height)
    ]
    top_origin = bytearray(bottom_origin)
    top_origin[pixel_start:pixel_end] = b"".join(reversed(rows))
    top_origin[17] |= 0x20

    cache = TextureCache()
    bottom_image = cache._load_bytes(bytes(bottom_origin))
    top_image = cache._load_bytes(bytes(top_origin))

    assert bottom_image is not None
    assert top_image is not None
    assert bottom_image.tobytes() == top_image.tobytes()
    assert tuple(bottom_image.get_flattened_data()) == (
        (0, 0, 255, 255),
        (255, 255, 0, 255),
        (255, 0, 0, 255),
        (0, 255, 0, 255),
    )
    assert bottom_image._gr_gpu_uv_v_flip is True
    assert top_image._gr_gpu_uv_v_flip is True

    # Imported Maya/Blender/OBJ UVs are already authored for the OpenGL
    # bottom-left convention. They opt out of the KOTOR/D3D shader inversion
    # regardless of how the equivalent TGA happened to be stored on disk.
    dcc_node = SimpleNamespace(uv_v_flip=False)
    assert _effective_diffuse_uv_v_flip(dcc_node, bottom_image) == 0.0
    assert _effective_diffuse_uv_v_flip(dcc_node, top_image) == 0.0

    # Native KOTOR UVs retain their D3D top-left convention and therefore need
    # exactly one shader inversion after TextureCache normalizes GPU rows.
    kotor_node = SimpleNamespace(uv_v_flip=True)
    assert _effective_diffuse_uv_v_flip(kotor_node, bottom_image) == 1.0
    assert _effective_diffuse_uv_v_flip(kotor_node, top_image) == 1.0


def test_texture_cache_detects_paintnet_bottom_left_dcc_tga_profile() -> None:
    from PIL import Image
    from src.core.rendering.frame_core.texture_cache import TextureCache

    source = Image.new("RGBA", (2, 2), (200, 100, 50, 255))
    encoded = BytesIO()
    source.save(encoded, format="TGA", compression="tga_rle")
    raw = bytearray(encoded.getvalue())
    assert raw[2] == 10
    assert raw[17] & 0x20 == 0

    # Paint.NET records its software ID in a TGA 2.0 extension area. This is
    # the conservative provenance signal used by the KotorBlender loose-atlas
    # orientation path; generic RLE TGAs must retain stock KotOR behavior.
    generic = TextureCache()._load_bytes(bytes(raw))
    assert generic is not None
    assert generic._gr_gpu_uv_v_flip is True

    extension_offset = len(raw)
    extension = bytearray(495)
    extension[:2] = (495).to_bytes(2, "little")
    extension[426:426 + len(b"Paint.NET 5.1.12")] = b"Paint.NET 5.1.12"
    raw.extend(extension)
    raw.extend(extension_offset.to_bytes(4, "little"))
    raw.extend((0).to_bytes(4, "little"))
    raw.extend(b"TRUEVISION-XFILE.\x00")

    image = TextureCache()._load_bytes(bytes(raw))

    assert image is not None
    assert image._gr_gpu_uv_v_flip is False
