"""Texture cache and texture metadata loading for frame rendering."""

from __future__ import annotations

import os
import threading
import weakref
from typing import Dict, Iterable, List, Optional, Tuple

from .dependencies import Image, _NUMPY, _PIL, log, np
from src.math.frame_math import _clean_tex_name, _clamp, _lerp
from src.core.graphics.tpc import _extract_txi_from_tpc, _is_tpc_data, _is_tpc_file, _load_tpc_bytes
from src.core.graphics.txi import _apply_txi_to_node, _extract_alpha_test_from_tpc, _parse_txi_string


def _uses_bottom_left_uv_tga_profile(raw: bytes) -> bool:
    """Identify the narrow loose-TGA profile emitted by this DCC workflow.

    Stock KotOR binary meshes use the renderer's Direct3D-style V conversion.
    KotorBlender-authored geometry can instead retain Blender's bottom-left UVs.
    The binary MDL format does not record which convention its author used, so
    loose textures need a conservative provenance hint. Paint.NET's TGA 2.0
    writer supplies one: the affected workflow writes bottom-origin, RLE
    true-colour images and records ``Paint.NET`` in the extension area.

    Requiring all three properties keeps ordinary uncompressed override TGAs
    (including Ghost Studio's PFBC09 output), stock TPC data, and generic RLE
    textures on the established KotOR path.
    """
    try:
        if len(raw) < 26 or raw[2] != 10 or (raw[17] & 0x20):
            return False
        if raw[-18:] != b"TRUEVISION-XFILE.\x00":
            return False
        extension_offset = int.from_bytes(raw[-26:-22], "little")
        if extension_offset <= 0 or extension_offset + 468 > len(raw) - 26:
            return False
        extension = raw[extension_offset:extension_offset + 495]
        if len(extension) < 468 or int.from_bytes(extension[:2], "little") < 468:
            return False
        software_id = extension[426:467].split(b"\x00", 1)[0]
        return software_id.strip().lower().startswith(b"paint.net")
    except (IndexError, TypeError, ValueError):
        return False


def _gpu_uv_v_flip_for_loose_texture(raw: bytes) -> bool:
    """Return the renderer V-conversion policy for a non-TPC loose texture."""
    return not _uses_bottom_left_uv_tga_profile(raw)

# ─────────────────────────────────────────────────────────────────────
#  Texture loader
# ─────────────────────────────────────────────────────────────────────

class TextureCache:
    """
    Loads and caches textures from disk.
    - Auto-detects KotOR TPC data in .tga files (KotOR stores TPC with .tga extension)
    - Supports DXT1 (enc=2), DXT5 (enc=4), uncompressed Grey/RGB/RGBA
    - Supports plain TGA, PNG via Pillow
    - Loads and caches TXI metadata (from embedded TPC TXI or standalone .txi files)
    Returns PIL.Image in RGBA mode at viewport resolution. TPC sources select
    the first authored mip within MAX_SIZE before decompression; ordinary image
    sources retain the resize fallback.
    """

    MAX_SIZE = 512   # max viewport texture resolution per axis
    MAX_MIP_BYTES = 32 * 1024 * 1024
    MAX_MIP_ENTRIES = 128
    # Raised from 256→512: KotOR textures are typically 128×128 or 256×256.
    # At 512px cap we load textures at their native resolution (no downscale for
    # typical sizes), eliminating the main source of blurry/blocky texture rendering.
    # Cost: 512×512 RGBA = 1MB per texture vs 256KB at 256px — acceptable for modern hardware.

    def __init__(self):
        self._cache: Dict[str, Optional['Image.Image']] = {}
        # Explicit authored previews (for example a Head Builder PNG published
        # under its final KOTOR ResRef) are not archive-cache entries.  Keep
        # them in a separate residency layer so a ResourceManager revision or
        # search-path refresh cannot silently replace them with the white
        # missing-texture fallback while the same preview model is still live.
        self._published_images: Dict[str, 'Image.Image'] = {}
        self._txi_cache: Dict[str, str] = {}   # name → TXI string ('' if absent)
        self._search_dirs: List[str] = []
        self._game_library = None   # Optional GameLibrary for BIF-backed loading
        self._game_tag: str = "K1"
        self._installation = None  # Optional KotorInstallation (fast path, legacy)
        self._resource_manager = None  # Optional ResourceManager (new unified path)
        self._resource_manager_revision = -1
        self._lock = threading.Lock()  # thread-safe access (render + prewarm threads)
        # Per-name load lock dict: prevents two threads loading the SAME texture simultaneously
        # while not blocking threads loading DIFFERENT textures (vs. a single global lock).
        self._load_locks: Dict[str, threading.Lock] = {}
        self._load_locks_lock = threading.Lock()  # protects _load_locks dict itself
        # Mip-bias cache: per-INSTANCE so clear_mip_cache() only affects this cache.
        # (Was previously a class-level dict which caused id() reuse bugs across instances.)
        # Derived pixels are disposable; authored/base textures are not.
        # Weak source identities prevent address reuse from returning old pixels.
        self._mip_bias_cache = {}

    def set_search_dirs(self, dirs: List[str]):
        new_dirs = [d for d in dirs if d and os.path.isdir(d)]
        # Only clear cache if the search directories actually changed
        with self._lock:
            if new_dirs != self._search_dirs:
                self._search_dirs = new_dirs
                self._cache.clear()
                self._mip_bias_cache = {}
                self._cache.update(self._published_images)
                self._txi_cache.clear()
                # Clear per-key load locks too (keys may no longer be relevant)
                with self._load_locks_lock:
                    self._load_locks.clear()
                log.debug(f"TextureCache search dirs updated: {self._search_dirs}")

    def set_game_library(self, library, game_tag: str = "K1"):
        """
        Attach a GameLibrary instance so textures can be loaded directly from
        BIF/ERF archives when not found on disk.  Clears the cache when the
        library reference changes.
        """
        with self._lock:
            if library is not self._game_library:
                self._game_library = library
                self._game_tag = game_tag
                self._cache.clear()
                self._mip_bias_cache = {}
                self._cache.update(self._published_images)
                self._txi_cache.clear()
                with self._load_locks_lock:
                    self._load_locks.clear()
                log.debug(f"TextureCache: game library set ({game_tag})")
            elif game_tag != self._game_tag:
                # Same library, but switching between K1 and K2 model –
                # update the tag and clear the cache so textures are re-resolved
                # from the correct game's archives.
                self._game_tag = game_tag
                self._cache.clear()
                self._mip_bias_cache = {}
                self._cache.update(self._published_images)
                self._txi_cache.clear()
                with self._load_locks_lock:
                    self._load_locks.clear()
                log.debug(f"TextureCache: game tag updated to {game_tag} (cache cleared)")

    def set_installation(self, installation, game_tag: str = "K1"):
        """
        Attach a KotorInstallation (fast lazy BIF/ERF reader) for texture loading.
        This supersedes the slower GameLibrary path for texture lookups.
        Clears the cache when the installation reference changes.
        """
        with self._lock:
            if installation is not self._installation:
                self._installation = installation
                self._game_tag = game_tag
                self._cache.clear()
                self._mip_bias_cache = {}
                self._cache.update(self._published_images)
                self._txi_cache.clear()
                with self._load_locks_lock:
                    self._load_locks.clear()
                log.info(f"TextureCache: KotorInstallation set ({game_tag})")

    def set_resource_manager(self, manager, game_tag: str = "K1"):
        """
        Attach the new unified ResourceManager as the primary texture backend.

        This is the preferred method — it supersedes both set_installation() and
        set_game_library() by routing all archive lookups through the single
        ResourceManager which handles KEY/BIF, TexturePacks ERFs, module ERFs,
        and Override/ in the correct priority order.

        Clears all caches when the manager reference or game tag changes.
        """
        with self._lock:
            revision = int(getattr(manager, "revision", 0) or 0) if manager is not None else -1
            changed = (manager is not self._resource_manager or
                       game_tag != self._game_tag or
                       revision != self._resource_manager_revision)
            if changed:
                self._resource_manager = manager
                self._resource_manager_revision = revision
                self._game_tag = game_tag
                # Also keep _installation in sync for legacy code paths
                if manager is not None:
                    inst = manager.get_k1() if game_tag == "K1" else manager.get_k2()
                    # _installation is used by legacy get_txi() / get_raw_header()
                    # We don't set it here to avoid the old path running — the new
                    # _resource_manager path takes priority in _load().
                self._cache.clear()
                self._mip_bias_cache = {}
                self._cache.update(self._published_images)
                self._txi_cache.clear()
                with self._load_locks_lock:
                    self._load_locks.clear()
                log.info(f"TextureCache: ResourceManager set ({game_tag})")

    def get_txi(self, name: str) -> str:
        """
        Get the TXI metadata string for a texture by name.

        Checks (in order):
          1. _txi_cache (fast path)
          2. Standalone .txi file on disk next to the texture
          3. TXI embedded in TPC file (via _extract_txi_from_tpc)
          4. TXI from BIF/ERF archive via GameLibrary

        Returns the raw TXI string (may be empty string if no TXI exists).
        """
        if not name:
            return ''
        clean = _clean_tex_name(name)
        if not clean:
            return ''
        key = clean.lower()
        # Fast path
        if key in self._txi_cache:
            return self._txi_cache[key]

        txi_str = ''
        try:
            with self._lock:
                search_dirs = list(self._search_dirs)
                game_library = self._game_library
                resource_manager = self._resource_manager
                game_tag = self._game_tag

            # 1. Look for standalone .txi file on disk
            for search_dir in search_dirs:
                txi_path = os.path.join(search_dir, clean + '.txi')
                if os.path.exists(txi_path):
                    try:
                        with open(txi_path, 'r', encoding='utf-8', errors='replace') as f:
                            txi_str = f.read().strip()
                        if txi_str:
                            log.debug(f"TXI '{clean}' loaded from {txi_path}")
                            break
                    except Exception as e:
                        log.debug(f"TXI file read error {txi_path}: {e}")

            # 2. Extract TXI from embedded TPC data
            if not txi_str:
                for search_dir in search_dirs:
                    for ext in ('.tga', '.TGA', '.tpc', '.TPC'):
                        tex_path = os.path.join(search_dir, clean + ext)
                        if not os.path.exists(tex_path):
                            continue
                        try:
                            with open(tex_path, 'rb') as f:
                                raw = f.read()
                            if _is_tpc_data(raw):
                                txi_str = _extract_txi_from_tpc(raw)
                                if txi_str:
                                    log.debug(f"TXI '{clean}' extracted from {ext} TPC file")
                                    break
                        except Exception as e:
                            log.debug(f"TXI TPC extract error {tex_path}: {e}")
                    if txi_str:
                        break

            # 2.5 Unified ResourceManager (standalone TXI resource, then the
            # TXI embedded in the TPC).  The main window attaches textures via
            # set_resource_manager(); without this step, texture-pack TPCs
            # (e.g. plc_starmap_06 with 'blending 1') never delivered their TXI
            # and additive surfaces rendered opaque.
            if not txi_str and resource_manager is not None:
                try:
                    txi_str = str(resource_manager.get_txi(clean, game_tag) or '').strip()
                except Exception as e:
                    log.debug(f"TXI resource-manager lookup error '{clean}': {e}")
                if not txi_str:
                    try:
                        raw = resource_manager.get_texture(clean, game_tag)
                        if raw and _is_tpc_data(raw):
                            txi_str = _extract_txi_from_tpc(raw)
                            if txi_str:
                                log.debug(f"TXI '{clean}' extracted from ResourceManager TPC")
                    except Exception as e:
                        log.debug(f"TXI resource-manager TPC extract error '{clean}': {e}")

            # 3. Load from BIF/ERF archive via GameLibrary
            if not txi_str and game_library is not None:
                try:
                    # TXI resource type ID = 1448 (RES_TXI from game_library.py)
                    _RES_TXI = 1448
                    raw = game_library.get_resource_data(clean, _RES_TXI, game_tag)
                    if raw:
                        txi_str = raw.decode('utf-8', errors='replace').strip()
                        if txi_str:
                            log.debug(f"TXI '{clean}' loaded from BIF/ERF archive")
                    # If no standalone TXI, try to get TXI from embedded TPC
                    if not txi_str:
                        raw = game_library.get_texture_data(clean, game_tag)
                        if raw and _is_tpc_data(raw):
                            txi_str = _extract_txi_from_tpc(raw)
                            if txi_str:
                                log.debug(f"TXI '{clean}' extracted from BIF TPC")
                except Exception as e:
                    log.debug(f"TXI BIF load error '{clean}': {e}")

        except Exception as e:
            log.debug(f"TXI load error for '{name}': {e}")

        with self._lock:
            self._txi_cache[key] = txi_str
        return txi_str

    def get_raw_header(self, name: str) -> Optional[bytes]:
        """Return the first 128 bytes of the TPC/TGA file for a texture.

        Used by _load_txi_metadata_for_model() to extract the alpha_test_threshold
        float from TPC header bytes [4-7] (FIX-ALPHATEST).

        Returns 128-byte header bytes if the texture is a TPC file, else None.
        The caller uses _extract_alpha_test_from_tpc() to read the float value.

        References:
            Kotor.NET TPC.cs — TPC header layout (width/height/encoding/alpha_test)
            xoreos tpc.cpp — alpha_test_threshold at header offset 4
            PyKotor io_tpc.py — TPCHeader.alpha_test_threshold field
        """
        if not name:
            return None
        clean = _clean_tex_name(name)
        if not clean:
            return None
        try:
            with self._lock:
                search_dirs = list(self._search_dirs)
                game_library = self._game_library
                resource_manager = self._resource_manager
                game_tag = self._game_tag
            # 1. Search on-disk directories
            for search_dir in search_dirs:
                for ext in ('.tga', '.TGA', '.tpc', '.TPC'):
                    path = os.path.join(search_dir, clean + ext)
                    if not os.path.exists(path):
                        continue
                    try:
                        with open(path, 'rb') as f:
                            header = f.read(128)
                        if _is_tpc_data(header):
                            return header
                    except Exception:
                        pass
            # 1.5 Unified ResourceManager (KEY/BIF, texture packs, Override)
            if resource_manager is not None:
                try:
                    raw = resource_manager.get_texture(clean, game_tag)
                    if raw and len(raw) >= 128 and _is_tpc_data(raw[:128]):
                        return raw[:128]
                except Exception:
                    pass
            # 2. BIF/ERF archive
            if game_library is not None:
                try:
                    raw = game_library.get_texture_data(clean, game_tag)
                    if raw and len(raw) >= 128 and _is_tpc_data(raw[:128]):
                        return raw[:128]
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def get(self, name: str) -> Optional['Image.Image']:
        if not _PIL or not name:
            return None
        clean = _clean_tex_name(name)
        if not clean:
            return None
        key = clean.lower()
        # Fast path: CPython GIL makes simple dict lookups atomic.
        # If the key is already in the cache (even None = "not found"), return immediately.
        try:
            return self._cache[key]
        except KeyError:
            pass
        # Slow path: use a per-key lock so two threads loading DIFFERENT textures
        # don't block each other, but two threads loading the SAME texture share one lock.
        # This avoids the old pattern where the global lock was held during disk I/O,
        # blocking the render thread for the entire duration of a BIF archive read.
        with self._load_locks_lock:
            if key not in self._load_locks:
                self._load_locks[key] = threading.Lock()
            key_lock = self._load_locks[key]
        with key_lock:
            # Double-check: another thread may have loaded it while we waited
            try:
                return self._cache[key]
            except KeyError:
                pass
            try:
                img = self._load(key)
            except MemoryError:
                log.warning(f"TextureCache: out of memory loading '{name}' — skipping")
                img = None
            except Exception as e:
                log.debug(f"TextureCache: error loading '{name}': {e}")
                img = None
            # Store result (even None) so future calls hit the fast path
            with self._lock:
                self._cache[key] = img
        return img

    @staticmethod
    def normalize_dirty_regions(
        image_size: tuple[int, int],
        regions: Optional[Iterable[object]],
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Clip dirty rectangles to an image without changing row orientation.

        Rectangles use ``(x, y, width, height)`` in the cached PIL image's
        pixel coordinate system.  GhostRigger's decoded KOTOR images are stored
        bottom-up, so PIL row zero is also the row uploaded to GPU texture row
        zero.  This helper deliberately performs no vertical flip.

        A region may also be a mapping/object exposing ``x``, ``y``,
        ``width`` and ``height``.  ``None`` means the whole image; malformed or
        empty rectangles are ignored.
        """
        width, height = (max(0, int(v)) for v in image_size[:2])
        if width <= 0 or height <= 0:
            return ()
        if regions is None:
            return ((0, 0, width, height),)

        normalized: list[tuple[int, int, int, int]] = []
        for region in regions:
            try:
                if isinstance(region, dict):
                    x = int(region.get("x", region.get("left", 0)))
                    y = int(region.get("y", region.get("top", 0)))
                    w = int(region.get("width", region.get("w", 0)))
                    h = int(region.get("height", region.get("h", 0)))
                elif all(hasattr(region, attr) for attr in ("x", "y", "width", "height")):
                    x = int(getattr(region, "x"))
                    y = int(getattr(region, "y"))
                    w = int(getattr(region, "width"))
                    h = int(getattr(region, "height"))
                else:
                    x, y, w, h = (int(value) for value in region)  # type: ignore[misc]
            except (TypeError, ValueError, AttributeError):
                continue
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            rect = (x0, y0, x1 - x0, y1 - y0)
            if rect not in normalized:
                normalized.append(rect)
        return tuple(normalized)

    def update_image_regions(
        self,
        name: str,
        image: 'Image.Image',
        regions: Optional[Iterable[object]] = None,
    ) -> tuple['Image.Image', tuple[tuple[int, int, int, int], ...]]:
        """Publish authored RGBA pixels while preserving cached image identity.

        When a texture with the same dimensions is already cached, dirty pixels
        are pasted into that object instead of replacing it.  Software and GPU
        caches key on image identity, so preserving the object lets the renderer
        update only the dirty regions.  A size change necessarily installs a new
        image and reports the whole image dirty.

        The returned image is the authoritative cache object and should be sent
        to the active GPU renderer together with the returned clipped regions.
        No model, scene, framebuffer or unrelated texture cache is invalidated.
        """
        if not _PIL:
            raise RuntimeError("Pillow is required for live texture updates")
        clean = _clean_tex_name(name or "")
        if not clean:
            raise ValueError("texture name is required")
        stem, extension = os.path.splitext(clean)
        if extension.lower() in {
            ".tga", ".tpc", ".png", ".dds", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        }:
            clean = stem
        if image is None or not hasattr(image, "size"):
            raise TypeError("image must be a PIL image")

        incoming = image if getattr(image, "mode", "") == "RGBA" else image.convert("RGBA")
        incoming_size = tuple(int(v) for v in incoming.size)
        if incoming_size[0] <= 0 or incoming_size[1] <= 0:
            raise ValueError("texture image dimensions must be positive")
        requested = self.normalize_dirty_regions(incoming_size, regions)
        key = clean.lower()

        with self._lock:
            previous = self._cache.get(key)
            same_shape = (
                previous is not None
                and getattr(previous, "mode", "") == "RGBA"
                and tuple(getattr(previous, "size", ())) == incoming_size
            )
            if same_shape:
                target = previous
                if target is not incoming:
                    for x, y, width, height in requested:
                        box = (x, y, x + width, y + height)
                        target.paste(incoming.crop(box), (x, y))
                    self._copy_texture_attrs(incoming, target)
            else:
                target = incoming
                # A new allocation has no resident pixels yet.  Report a full
                # update even when the caller supplied a narrower dirty set.
                requested = ((0, 0, incoming_size[0], incoming_size[1]),)
                self._cache[key] = target

            # Same-object edits still need derived mip data to be regenerated.
            # Replacing the map also prevents an in-flight resize from publishing
            # pixels sampled before this edit.
            self._mip_bias_cache = self._mip_bias_cache.copy()
            self._mip_bias_cache.pop(id(target), None)
            if previous is not None and previous is not target:
                self._mip_bias_cache.pop(id(previous), None)

        return target, requested

    def publish_bytes(
        self,
        name: str,
        raw: bytes,
    ) -> Optional['Image.Image']:
        """Publish authored bytes under an explicit preview texture ResRef.

        A Head Builder source PNG can keep its descriptive filename while the
        candidate model references its final KOTOR ResRef. Decode through the
        normal loose-texture orientation path and cache it under that ResRef;
        no source rename or copy is required.
        """

        clean = _clean_tex_name(name or "")
        if not clean or not isinstance(raw, (bytes, bytearray)) or not raw:
            return None
        payload = bytes(raw)
        image = self._load_bytes(payload)
        if image is None:
            return None
        image = self._resize_if_needed(image, clean)
        try:
            image = self._apply_kotor_alpha(
                payload,
                image,
                _parse_txi_string(""),
            )
        except Exception:
            pass
        published, _regions = self.update_image_regions(clean, image, None)
        with self._lock:
            self._published_images[clean.lower()] = published
        return published

    def clear_published_images(self) -> None:
        """Release model-scoped authored preview aliases.

        Viewports call this before loading a different model, then republish
        that model's explicit texture map.  Archive/resource invalidation keeps
        the active aliases resident, but changing the preview model does not
        leak them into unrelated work.
        """

        with self._lock:
            self._mip_bias_cache = self._mip_bias_cache.copy()
            for key, published in tuple(self._published_images.items()):
                if self._cache.get(key) is published:
                    self._cache.pop(key, None)
                self._mip_bias_cache.pop(id(published), None)
            self._published_images.clear()

    def _load(self, name: str) -> Optional['Image.Image']:
        """Load texture by name: disk search dirs first, then BIF archives.
        Called under per-key lock — safe for concurrent access.

        v12.7 ALPHA FIX: KotOR DXT5 alpha channel has three distinct meanings
        depending on TXI metadata.  After loading, we check the TXI for:
          1. 'bumpmaptexture' → alpha is bump/specular data, NOT transparency.
             Force alpha channel = 255 (fully opaque surface rendering).
             Affects: c_rancor01, c_hutt01, c_drdassassin01, etc.
          2. 'blending punchthrough' → apply TPC alpha_test_threshold as binary cutoff.
             Uses the float at TPC header bytes [4-7] as the GL_ALPHA_TEST value.
          3. Standard → alpha as-is (glass, hair, transparent effects).

        FIX-TXI-PREFER: _load_tpc_bytes / _load_tpc_bytes_legacy attach _txi_str
        directly to the returned PIL Image when they successfully extract TXI from
        the embedded TPC trailer.  We now prefer that attached TXI string over the
        result of get_txi() so that stock KotOR BIF textures (enc=2/4, data_sz=0)
        with embedded TXI get their blending/alpha modes applied correctly.
        The external get_txi() call is kept as a fallback for sidecar .txi files
        and archive TXI resources not embedded in the TPC itself.
        """
        # Snapshot search dirs under lock to avoid TOCTOU with set_search_dirs()
        with self._lock:
            search_dirs = list(self._search_dirs)
            game_library = self._game_library
            game_tag = self._game_tag
            installation = self._installation
            resource_manager = self._resource_manager

        # ── 1. Search on-disk directories first (override folder wins) ──────
        for search_dir in search_dirs:
            # Priority: .tga first (may be TPC), then common DCC image formats.
            for ext in (
                '.tga', '.TGA', '.tpc', '.TPC', '.png', '.PNG',
                '.dds', '.DDS', '.jpg', '.JPG', '.jpeg', '.JPEG',
                '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF',
            ):
                path = os.path.join(search_dir, name + ext)
                if not os.path.exists(path):
                    continue
                try:
                    img = self._load_file(path)
                    if img is not None:
                        img = self._resize_if_needed(img, name)
                        # Apply TXI-aware alpha processing for on-disk textures.
                        # FIX-TXI-PREFER: use _txi_str already attached to img
                        # (by _load_tpc_bytes/legacy) if available, then fall back
                        # to get_txi() for sidecar files / archive TXI resources.
                        try:
                            with open(path, 'rb') as fraw:
                                raw_bytes = fraw.read(512)  # header only for alpha_test
                            txi_s = getattr(img, '_txi_str', None)
                            if txi_s is None:
                                txi_s = self.get_txi(name)
                            txi_m = _parse_txi_string(txi_s) if txi_s else _parse_txi_string('')
                            img = self._apply_kotor_alpha(raw_bytes, img, txi_m)
                        except Exception:
                            pass
                        log.debug(f"Texture '{name}' loaded from {os.path.basename(path)}")
                        return img
                except MemoryError:
                    log.warning(f"Texture '{name}': out of memory — skipping")
                    return None
                except Exception as e:
                    log.debug(f"Texture load error {path}: {e}")
            # Case-insensitive stem/prefix match for DCC sidecars (FBX .fbm folders).
            # Some FBX import paths inherit KOTOR's 32-byte texture field cap before
            # the renderer sees the source texture name.  The sidecar image keeps its
            # full DCC stem, so accept a unique longer on-disk stem that starts with
            # the requested name.
            try:
                prefix_matches = []
                for child in os.scandir(search_dir):
                    if not child.is_file():
                        continue
                    stem, ext = os.path.splitext(child.name)
                    stem_key = stem.lower()
                    if ext.lower() not in (
                        '.tga', '.tpc', '.png', '.dds', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
                    ):
                        continue
                    if stem_key == name:
                        prefix_matches = [child]
                        break
                    if stem_key.startswith(name) or name.startswith(stem_key):
                        prefix_matches.append(child)
                if len(prefix_matches) != 1:
                    continue
                child = prefix_matches[0]
                stem, ext = os.path.splitext(child.name)
                if ext.lower() not in (
                    '.tga', '.tpc', '.png', '.dds', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
                ):
                    continue
                path = child.path
                try:
                    img = self._load_file(path)
                    if img is not None:
                        img = self._resize_if_needed(img, name)
                        try:
                            with open(path, 'rb') as fraw:
                                raw_bytes = fraw.read(512)
                            txi_s = getattr(img, '_txi_str', None)
                            if txi_s is None:
                                txi_s = self.get_txi(name)
                            txi_m = _parse_txi_string(txi_s) if txi_s else _parse_txi_string('')
                            img = self._apply_kotor_alpha(raw_bytes, img, txi_m)
                        except Exception:
                            pass
                        log.debug(f"Texture '{name}' loaded from {os.path.basename(path)}")
                        return img
                except MemoryError:
                    log.warning(f"Texture '{name}': out of memory — skipping")
                    return None
                except Exception as e:
                    log.debug(f"Texture load error {path}: {e}")
            except OSError:
                pass
        # ── 2. ResourceManager (unified BIF/ERF/Override, <2ms) ─────────────
        # New primary archive backend — replaces the split installation/game_library path.
        # Checks: Override > module ERFs > TexturePacks ERFs > BIF in correct priority.
        if resource_manager is not None:
            try:
                raw = resource_manager.get_texture(name, game_tag)
                if raw:
                    img = self._load_bytes(raw)
                    if img is not None:
                        img = self._resize_if_needed(img, name)
                        try:
                            txi_s = getattr(img, '_txi_str', None)
                            if not txi_s:
                                txi_s = resource_manager.get_txi(name, game_tag)
                            txi_m = _parse_txi_string(txi_s) if txi_s else _parse_txi_string('')
                            img = self._apply_kotor_alpha(raw, img, txi_m)
                        except Exception:
                            pass
                        log.debug(f"Texture '{name}' loaded from ResourceManager ({game_tag})")
                        return img
            except MemoryError:
                log.warning(f"Texture '{name}': out of memory from ResourceManager — skipping")
                return None
            except Exception as e:
                log.debug(f"Texture ResourceManager error '{name}': {e}")
        # ── 3. Legacy: KotorInstallation (lazy BIF/ERF seek, <5ms) ──────────
        if installation is not None:
            try:
                raw = installation.get_texture(name)
                if raw:
                    img = self._load_bytes(raw)
                    if img is not None:
                        img = self._resize_if_needed(img, name)
                        try:
                            txi_s = getattr(img, '_txi_str', None)
                            if not txi_s:
                                txi_s = installation.get_txi(name)
                            txi_m = _parse_txi_string(txi_s) if txi_s else _parse_txi_string('')
                            img = self._apply_kotor_alpha(raw, img, txi_m)
                        except Exception:
                            pass
                        log.debug(f"Texture '{name}' loaded from KotorInstallation")
                        return img
            except MemoryError:
                log.warning(f"Texture '{name}': out of memory from installation — skipping")
                return None
            except Exception as e:
                log.debug(f"Texture installation load error '{name}': {e}")
        # ── 4. Fallback: load from BIF/KEY/ERF archives via GameLibrary ──────
        if game_library is not None:
            try:
                raw = game_library.get_texture_data(name, game_tag)
                if raw:
                    img = self._load_bytes(raw)
                    if img is not None:
                        img = self._resize_if_needed(img, name)
                        # Apply TXI-aware alpha processing for BIF textures.
                        # FIX-TXI-PREFER: _load_bytes → _load_tpc_bytes attaches
                        # _txi_str to img if it successfully parsed embedded TXI.
                        # Use that first; fall back to get_txi() which re-fetches
                        # from the archive (slower but works for sidecar .txi).
                        try:
                            txi_s = getattr(img, '_txi_str', None)
                            if not txi_s:
                                txi_s = self.get_txi(name)
                            txi_m = _parse_txi_string(txi_s) if txi_s else _parse_txi_string('')
                            img = self._apply_kotor_alpha(raw, img, txi_m)
                        except Exception:
                            pass
                        log.debug(f"Texture '{name}' loaded from BIF archive")
                        return img
            except MemoryError:
                log.warning(f"Texture '{name}': out of memory from BIF — skipping")
                return None
            except Exception as e:
                log.debug(f"Texture BIF load error '{name}': {e}")

        log.debug(f"Texture '{name}' not found in search dirs or BIF archives")
        return None

    @staticmethod
    def _copy_texture_attrs(src: 'Image.Image', dst: 'Image.Image') -> 'Image.Image':
        for attr in (
            '_gr_gpu_uv_v_flip', '_txi_str', '_tpc_raw', '_txi_alpha_test',
            '_tpc_mip_level', '_tpc_mip_size', '_tpc_source_size',
            '_tpc_viewport_max_size',
        ):
            if hasattr(src, attr):
                try:
                    setattr(dst, attr, getattr(src, attr))
                except Exception:
                    pass
        return dst

    @staticmethod
    def _apply_kotor_alpha(raw_bytes: bytes, img: 'Image.Image',
                           txi_meta: dict) -> 'Image.Image':
        """
        Apply correct KotOR alpha processing to a loaded RGBA texture.

        KotOR uses DXT5 alpha for different purposes:
          1. bumpmaptexture in TXI → alpha = normal/bump map data, NOT transparency.
             Force alpha = 255 (solid opaque surface).
          2. envmaptexture / bumpyshinytexture in TXI → alpha = env-map blend weight.
             KotOR uses "EnvironmentBlendedOver" (xoreos/modelnode.cpp:726-773):
             The env map is drawn ADDITIVELY on top of the diffuse, weighted by
             (1 - diffuse.alpha).  Where alpha=0, env shows at full strength.
             Where alpha=1, env barely contributes.  We PRESERVE the alpha channel
             here so the GPU fragment shader can use it for BlendedOver blending.
             IMPORTANT: 'bumpyshinytexture' is an alias for 'envmaptexture' in
             both KotOR.js (TXI.ts:161-164) and xoreos (modelnode.cpp:479-482).
          3. blending punchthrough in TXI → binary alpha cutoff at TPC threshold.
             Read alpha_test_threshold from TPC header bytes [4-7].
          4. blending additive (1) → keep alpha as-is for additive particle effects.
          5. Standard (blending=0, no bump, no envmap) → FORCE alpha=255.
             KotOR DXT5 textures store bump/specular data in the alpha
             channel by default — treating it as transparency makes models
             look see-through.  The engine itself ignores alpha on opaque
             surfaces; we must do the same.

        Returns modified image (or original if no processing needed).
        """
        if img is None or not _NUMPY:
            return img
        try:
            blending = txi_meta.get('blending', 0)
            has_bump = bool(txi_meta.get('bumpmaptexture', ''))
            has_env  = bool(txi_meta.get('envmaptexture', ''))
            if has_bump:
                # Case 1: bump map — alpha is normal/bump data, NOT transparency.
                # Force fully opaque so the model renders solid.
                arr = np.array(img)
                if arr[:, :, 3].min() < 255:
                    arr[:, :, 3] = 255
                    return TextureCache._copy_texture_attrs(img, Image.fromarray(arr, 'RGBA'))
            elif has_env:
                # Case 2: env map — alpha = blend weight between surface and env map.
                # PRESERVE the alpha channel (do NOT force to 255).
                # The GPU shader reads alpha as env_weight; the surface is opaque
                # (final_alpha=1 is forced in the shader when u_has_env=1).
                # The CPU path must also not override this channel.
                pass  # keep original alpha for env map blending
            elif blending == 2:
                # Case 3: punchthrough alpha — apply TPC threshold as hard cutoff
                threshold = 128  # default if TPC header not available
                if raw_bytes and len(raw_bytes) >= 8:
                    import struct as _s
                    try:
                        at = _s.unpack_from('<f', raw_bytes, 4)[0]
                        at = max(0.0, min(1.0, at))
                        threshold = int(at * 255)
                    except Exception:
                        pass
                if threshold > 0:
                    arr = np.array(img)
                    alpha = arr[:, :, 3]
                    if not (np.all(alpha >= threshold) or np.all(alpha < threshold)):
                        arr[:, :, 3] = np.where(alpha >= threshold, 255, 0).astype(np.uint8)
                        return TextureCache._copy_texture_attrs(img, Image.fromarray(arr, 'RGBA'))
            elif blending == 1:
                # Case 4: additive blend — keep alpha for additive particle effects
                pass
            else:
                # Case 5: standard opaque surface (blending=0, no bump, no envmap) —
                # ALWAYS force alpha=255.  KotOR DXT5 encodes bump/specular in
                # alpha; using it as transparency makes character skin see-through.
                arr = np.array(img)
                if arr[:, :, 3].min() < 255:
                    arr[:, :, 3] = 255
                    return TextureCache._copy_texture_attrs(img, Image.fromarray(arr, 'RGBA'))
        except Exception as e:
            log.debug(f"_apply_kotor_alpha error: {e}")
        return img


    def _resize_if_needed(self, img: 'Image.Image', name: str = '') -> 'Image.Image':
        """Downscale image to MAX_SIZE if too large. Handles MemoryError gracefully."""
        try:
            w, h = img.size
            if w > self.MAX_SIZE or h > self.MAX_SIZE:
                # Maintain aspect ratio
                scale = self.MAX_SIZE / max(w, h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                original = img
                img = self._copy_texture_attrs(original, img.resize((nw, nh), Image.LANCZOS))
                log.debug(f"Texture '{name}' downscaled {w}x{h} → {nw}x{nh}")
        except MemoryError:
            log.warning(f"Texture '{name}': MemoryError during resize — using original")
        except Exception as e:
            log.debug(f"Texture resize error '{name}': {e}")
        return img

    def _load_bytes(self, raw: bytes) -> Optional['Image.Image']:
        """Load a texture from raw bytes (TPC or TGA/PNG).

        All returned images are in BOTTOM-UP orientation so that the renderer's
        V-flip formula (tv = (1-v)*h) produces correct UV mapping.
        KotOR MDL UV V=0 means TOP of texture (Direct3D/top-down convention).
        The render-time flip converts from KotOR UV-space to PIL row-space.
        - TPC files: _load_tpc_bytes(max_size=MAX_SIZE) selects an authored mip
          before conversion and returns it bottom-up.
        - Standard TGA files (bottom-up origin): PIL loads bottom-up correctly.
        - PNG/other: PIL loads top-down, must flip to bottom-up.
        """
        if not _PIL:
            return None
        if _is_tpc_data(raw):
            img = _load_tpc_bytes(raw, max_size=self.MAX_SIZE)
            if img is not None:
                img._gr_gpu_uv_v_flip = True  # type: ignore[attr-defined]
            return img
        try:
            import io
            img = Image.open(io.BytesIO(raw)).convert('RGBA')
            # FIX-TGA-ORIENT: PIL always returns top-down images (row 0 = top)
            # regardless of TGA origin bit.  PIL internally flips bottom-origin
            # TGA data to top-down during Image.open().  ALL images from PIL are
            # therefore top-down.  We must flip ALL of them to bottom-up so the
            # renderer's V-flip formula (tv = (1-v)*h) maps KotOR D3D UVs
            # (V=0=top) correctly:
            #   V=0 (top) → (1-0)*h = h → last row → top of bottom-up image ✓
            #   V=1 (bottom) → (1-1)*h = 0 → row 0 → bottom of bottom-up image ✓
            #
            # Previously, bottom-origin TGA files were NOT flipped (assumed PIL
            # preserved the bottom-up layout), causing them to remain top-down
            # and render upside-down.
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # The uploaded rows now follow OpenGL's bottom-up convention, so
            # KOTOR/D3D UVs (V=0 at the top) still require the shader's V
            # conversion.  Imported DCC nodes opt out with uv_v_flip=False.
            # Marking normalized TGA/PNG images False here inverted custom
            # character atlases such as PFBC09 after otherwise-correct UV and
            # anatomy splitting.
            img._gr_gpu_uv_v_flip = _gpu_uv_v_flip_for_loose_texture(raw)  # type: ignore[attr-defined]
            return img
        except Exception:
            return None

    def _load_file(self, path: str) -> Optional['Image.Image']:
        """
        Load a texture file. Auto-detects TPC format even for .tga extensions.
        KotOR stores TPC data in files named .tga – we detect this by checking
        the data_sz / width / height fields in the first 128 bytes.

        All returned images are in bottom-up orientation so that the render-time
        V-flip (tv = (1-v)*h) produces correct UV mapping.
        KotOR MDL UV V=0 = top of texture (Direct3D convention).
        - TPC files: bottom-up from the authored-mip viewport fast path.
        - All PIL-opened images (TGA, PNG, DDS): PIL always returns top-down
          → flip to bottom-up for consistency.
        """
        try:
            with open(path, 'rb') as f:
                raw = f.read()
        except OSError as e:
            log.debug(f"Cannot read {path}: {e}")
            return None

        if _is_tpc_data(raw):
            img = _load_tpc_bytes(raw, max_size=self.MAX_SIZE)
            if img is not None:
                img._gr_gpu_uv_v_flip = True  # type: ignore[attr-defined]
            return img

        # Fall back to Pillow for real TGA / PNG / DDS
        if _PIL:
            try:
                import io
                img = Image.open(io.BytesIO(raw)).convert('RGBA')
                # FIX-TGA-ORIENT: PIL always returns top-down images (row 0 = top)
                # regardless of TGA origin bit.  PIL internally normalises all
                # TGA variants to top-down during Image.open().  Flip ALL images
                # to bottom-up so the renderer's V-flip formula works correctly.
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                # Bottom-up GPU rows need the per-node KOTOR/D3D V conversion;
                # DCC/OpenGL UV nodes explicitly disable it with uv_v_flip=False.
                img._gr_gpu_uv_v_flip = _gpu_uv_v_flip_for_loose_texture(raw)  # type: ignore[attr-defined]
                return img
            except Exception:
                pass
        return None

    # ── UE-inspired texture mip-bias for interactive mode ────────────────
    # In UE, the streaming manager can supply lower-resolution mip levels
    # for LOD objects (StreamingManagerTexture.cpp: StreamWantedMips).
    # We replicate this by keeping a half-resolution cache ("mip1") that is
    # used during interactive orbit/drag.  This halves the number of
    # getpixel() calls in _paste_textured_triangle by reducing the texture's
    # lookup area, cutting per-frame cost roughly 4× for large textures.

    def get_mip1(self, img: 'Image.Image') -> Optional['Image.Image']:
        """Return half-resolution pixels within a disposable preview budget.

        Resize outside the cache lock; a resource change or paint update while
        resizing invalidates the captured map and prevents stale publication.
        Keep live cached mips stable. When full, use the original pixels instead
        of repeatedly resizing/evicting every texture in an oversized scene.
        No base image, authored pixels, or source file is evicted.
        """
        if img is None or not _PIL:
            return img
        key = id(img)
        with self._lock:
            cache = self._mip_bias_cache
            cached = cache.get(key)
            if cached is not None:
                if cached[0]() is img:
                    return cached[1]
                cache.pop(key)

        w, h = img.size
        nw, nh = max(1, w // 2), max(1, h // 2)
        # Pillow uses four-byte storage for RGB as well as RGBA. Budget without
        # allocating a temporary pixel buffer or resizing an over-budget image.
        pixel_bytes = nw * nh * max(4, len(img.getbands()))

        def has_capacity() -> bool:
            for stale_key, (source_ref, _image, _size) in tuple(cache.items()):
                if source_ref() is None:
                    cache.pop(stale_key)
            return (
                len(cache) < self.MAX_MIP_ENTRIES
                and sum(entry[2] for entry in cache.values()) + pixel_bytes <= self.MAX_MIP_BYTES
            )

        with self._lock:
            if cache is not self._mip_bias_cache or not has_capacity():
                return img
        try:
            mip = img.resize((nw, nh), Image.BOX if hasattr(Image, 'BOX') else Image.NEAREST)
        except Exception:
            return img

        with self._lock:
            if cache is not self._mip_bias_cache:
                return mip
            cached = cache.get(key)
            if cached is not None and cached[0]() is img:
                return cached[1]
            if has_capacity():
                cache[key] = (weakref.ref(img), mip, pixel_bytes)
        return mip

    def clear_mip_cache(self):
        """Clear mip-bias cache (call when textures reload).

        FIX (v10.4): Replace the dict object rather than clearing in-place.
        This ensures that any render-thread snapshot of the old dict is still
        valid (it will just miss new entries), avoiding a race between the
        main-thread clear and a render-thread read.
        """
        with self._lock:
            self._mip_bias_cache = {}

    def sample(self, img: 'Image.Image', u: float, v: float,
               clamp_s: bool = False, clamp_t: bool = False) -> Tuple[int,int,int]:
        """Sample texture at UV (tiled/wrapped). Returns (r, g, b).

        KotOR MDX UVs follow Direct3D convention: V=0 = texture TOP.
        Our PIL Images are bottom-up (row 0 = bottom of texture).
        Flip: tex_row = (1-v_tiled)*h → V=0 maps to last row (top), V=1 to row 0 (bottom).

        Tiling: values outside [0, 1] are wrapped via % 1.0.  Values *inside*
        [0, 1] – including the boundary v=1.0 – are kept as-is so that
        v=1.0 (OpenGL top edge) correctly maps to PIL row 0.
        Bug-fix: the old code applied % unconditionally, which collapsed v=1.0
        to 0.0 (same as v=0.0, the bottom edge) and returned the wrong row.

        clamp_s / clamp_t: when True, implement GL_CLAMP_TO_EDGE on the
        corresponding axis (i.e. clamp UV to [0,1] instead of wrapping).
        Used for TXI 'clamp' textures (e.g. all KotOR head/face textures).
        """
        if img is None:
            return (128, 128, 128)
        w, h = img.size
        # U axis: GL_REPEAT (wrap) or GL_CLAMP_TO_EDGE
        if clamp_s:
            u_frac = max(0.0, min(1.0, u))
            px = min(w - 1, int(u_frac * w))
        else:
            u_frac = u % 1.0
            px = int(u_frac * w) % w
        # V axis: GL_REPEAT (wrap) or GL_CLAMP_TO_EDGE
        if clamp_t:
            v_tiled = max(0.0, min(1.0, v))
        elif v < 0.0 or v > 1.0:
            # Tile V only when outside [0, 1] to preserve the v=1.0 → row-0 mapping.
            v_tiled = v % 1.0
        else:
            v_tiled = v
        # V-flip: OpenGL V=0 → PIL row h-1; V=1 → PIL row 0.
        py = max(0, min(h - 1, int((1.0 - v_tiled) * h)))
        try:
            pixel = img.getpixel((px, py))
            if len(pixel) >= 3:
                return (pixel[0], pixel[1], pixel[2])
            return (pixel[0], pixel[0], pixel[0])
        except Exception:
            return (128, 128, 128)

    def sample_bilinear(self, img: 'Image.Image', u: float, v: float,
                         clamp_s: bool = False, clamp_t: bool = False) -> Tuple[int,int,int,int]:
        """
        Bilinear-filtered texture sample. Returns (r, g, b, a) for correct alpha blending.

        Uses the same V-flip as sample(): v=0 (D3D top) → row h-1 (top of
        bottom-up image); v=1 (D3D bottom) → row 0 (bottom of bottom-up image).
        V tiling only applied outside [0,1] so that v=1.0 stays at row 0
        rather than collapsing to v=0.0.

        clamp_s / clamp_t: GL_CLAMP_TO_EDGE on U/V axis respectively.
        When set, the UV is clamped to [0,1] before sampling instead of
        wrapping.  Neighbour pixels (x1/y1) are also clamped to prevent
        bilinear from reading across the edge boundary.
        """
        if img is None:
            return (128, 128, 128, 255)
        w, h = img.size
        # U axis: GL_REPEAT or GL_CLAMP_TO_EDGE
        if clamp_s:
            u_clamped = max(0.0, min(1.0, u))
            u_f = u_clamped * w
        else:
            u_f = (u % 1.0) * w
        # V axis: GL_REPEAT or GL_CLAMP_TO_EDGE
        if clamp_t:
            v_tiled = max(0.0, min(1.0, v))
        elif v < 0.0 or v > 1.0:
            # Tile V only when outside [0, 1] (preserve v=1.0 → top-row boundary).
            v_tiled = v % 1.0
        else:
            v_tiled = v
        # V-flip: V=0 (OpenGL bottom) → row h-1; V=1 → row 0.
        v_f = (1.0 - v_tiled) * h
        x0 = int(u_f) % w
        y0 = max(0, min(h - 1, int(v_f)))
        # Neighbour pixels: clamp to edge instead of wrapping for clamped axes
        if clamp_s:
            x1 = min(w - 1, x0 + 1)
        else:
            x1 = (x0 + 1) % w
        if clamp_t:
            y1 = min(h - 1, y0)  # don't step past edge
        else:
            y1 = min(h - 1, y0 + 1)
        fx = u_f - int(u_f)
        fy = v_f - int(v_f)
        try:
            c00 = img.getpixel((x0, y0))
            c10 = img.getpixel((x1, y0))
            c01 = img.getpixel((x0, y1))
            c11 = img.getpixel((x1, y1))
            # Ensure all pixels have 4 components
            def _rgba(p):
                if len(p) == 4: return p
                if len(p) == 3: return (p[0], p[1], p[2], 255)
                return (p[0], p[0], p[0], 255)
            c00, c10, c01, c11 = _rgba(c00), _rgba(c10), _rgba(c01), _rgba(c11)
            r = int(_lerp(_lerp(c00[0], c10[0], fx), _lerp(c01[0], c11[0], fx), fy))
            g = int(_lerp(_lerp(c00[1], c10[1], fx), _lerp(c01[1], c11[1], fx), fy))
            b = int(_lerp(_lerp(c00[2], c10[2], fx), _lerp(c01[2], c11[2], fx), fy))
            a = int(_lerp(_lerp(c00[3], c10[3], fx), _lerp(c01[3], c11[3], fx), fy))
            return (_clamp(r,0,255), _clamp(g,0,255), _clamp(b,0,255), _clamp(a,0,255))
        except Exception:
            return self.sample(img, u, v) + (255,)



__all__ = tuple(name for name in globals() if not name.startswith('__'))
