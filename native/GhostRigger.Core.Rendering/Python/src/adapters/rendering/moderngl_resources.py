from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Dict, List, Optional, Tuple

from src.adapters.gpu.moderngl_runtime import Image, _NUMPY, _PIL, moderngl, np
from src.adapters.gpu.viewport_probe import _gr_gpu_probe
from src.core.geometry.lightsaber import (
    is_lightsaber_blade_node,
    lightsaber_blade_preview_quad,
    synthetic_lightsaber_blade_uvs,
)
from src.core.rendering.gpu_vbo_layout import _split_vbo_attributes_for_gpu
from src.math.gpu_math import (
    _bas_attachment_local_transform_np,
    _mat3_normal,
    _mat4_from_pos_quat_scale,
    _mat4_identity,
    _mat4_lookat,
    _mat4_mul,
    _mat4_perspective,
    _mat4_tobytes,
    _matrix_from_pos_quat_np,
    _quat_multiply_xyzw,
    _quat_rotate_batch,
    _scene_authored_world_transform,
    _scene_gpu_model_matrix,
    _scene_gpu_root_for_node,
)

log = logging.getLogger(__name__)

_RETAIL_C_KATH_HAIR_VERTEX_COUNT = 152
_RETAIL_C_KATH_HAIR_FACE_COUNT = 110
_RETAIL_C_KATH_HAIR_UV_SHA256 = (
    "d30e283059dac9caad28d608fcbd71beb3f00e4dfc6f777a2a60595207014560"
)


def _is_saber_runtime_helper(node) -> bool:
    """Return whether ``node`` is an Odyssey NODE_SABER runtime helper.

    The game uses these records to drive blade extension.  They duplicate the
    two ordinary crossed glow cards and are not drawable preview geometry.
    """

    return bool(getattr(node, "is_saber", False))


def _configure_lightsaber_blade_sampler(texture) -> None:
    """Use smooth, non-mipmapped sampling for one procedural blade texture."""

    if texture is None:
        return
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False


class _GlTexCache:
    """Caches PIL Image → GL Texture2D upload to avoid re-uploading unchanged textures.

    FIX-TEXCACHE-KEY: The original implementation keyed on id(img) alone, which
    is unsafe because Python can reuse object addresses after GC.  A new PIL Image
    allocated at the same address as a previously-released one would receive a stale
    GL texture upload (wrong pixels, wrong size).  We guard against this by also
    storing a weak-reference to the original image object.  On cache-hit we check
    that the weakref is still alive AND still points to the same object; if the
    weakref is dead (GC'd) the entry is a ghost and must be evicted + re-uploaded.

    Additionally we use a MAX_ENTRIES cap (512) with LRU eviction via OrderedDict to
    prevent unbounded VRAM growth when batch-rendering hundreds of models.  Each
    512×512 RGBA mip-chain ≈ 682 KB VRAM; 512 entries ≈ 341 MB worst-case, well
    within the 2 GB EGL soft limit on typical headless servers.
    """

    MAX_ENTRIES: int = 512  # VRAM safety cap; adjust per target hardware

    def __init__(self, ctx: 'moderngl.Context'):
        self._ctx = ctx
        # OrderedDict for O(1) LRU: key = id(img), value = (weakref, GL Texture)
        import collections, weakref as _weakref_mod
        self._cache: 'collections.OrderedDict[int, tuple]' = collections.OrderedDict()
        self._wr_mod = _weakref_mod  # stash for use in methods
        self.region_update_count: int = 0
        self.region_update_bytes: int = 0

    def get(self, img: Optional['Image.Image']) -> Optional['moderngl.Texture']:
        if img is None or not _PIL:
            return None
        key = id(img)
        if key in self._cache:
            wr, tex = self._cache[key]
            live = wr()
            if live is img:
                # Cache hit: move to end (most-recently-used) and return
                self._cache.move_to_end(key)
                return tex
            # Stale entry (GC'd object reused same address) — evict and re-upload
            try:
                tex.release()
            except Exception:
                pass
            del self._cache[key]
        # Cache miss: upload and store
        tex = self._upload(img)
        if tex:
            # Evict LRU entry if over capacity
            while len(self._cache) >= self.MAX_ENTRIES:
                _, (_, old_tex) = self._cache.popitem(last=False)
                try:
                    old_tex.release()
                except Exception:
                    pass
            try:
                wr = self._wr_mod.ref(img)
            except TypeError:
                # img is not weakly referenceable (rare); use a dummy ref
                wr = lambda: img  # noqa: E731 — never GC'd while in cache
            self._cache[key] = (wr, tex)
        return tex

    def _upload(self, img: 'Image.Image') -> Optional['moderngl.Texture']:
        try:
            rgba = img.convert('RGBA')
            # FIX-VFLIP-REMOVED: NO flip here.  All images arriving at the GPU
            # texture cache are ALREADY in bottom-up (OpenGL) orientation.
            #
            # TextureCache._load_tpc_bytes (viewport.py) flips DXT textures from
            # top-down (DirectX DXT block order) to bottom-up, and uncompressed
            # textures from PyKotor are already bottom-up.  The MCP/ghostrigger
            # path also uses _load_tpc_bytes which returns bottom-up images.
            #
            # Previously this method applied FLIP_TOP_BOTTOM unconditionally,
            # assuming images were top-down (fresh from PyKotor's to_pil_image).
            # That caused a double-flip for DXT textures (TextureCache flipped
            # once, _upload flipped again → back to top-down → upside-down in GL)
            # and an unwanted flip for uncompressed textures (already bottom-up
            # → flipped to top-down → also upside-down in GL).
            #
            # With bottom-up images uploaded directly:
            #   PIL row 0 = bottom of image → GL texture row 0 (V=0) = bottom ✓
            #   Vertex shader: gl_v = 1.0 - kotor_v
            #     KotOR V=0 (top) → GL V=1.0 → samples last row = top of image ✓
            #     KotOR V=1 (bottom) → GL V=0.0 → samples first row = bottom ✓
            #
            # Cross-ref: viewport._load_tpc_bytes (DXT flip to bottom-up);
            #            viewport.TextureCache._load_bytes (bottom-up contract);
            #            kotor_loader.load_tpc_as_pil (top-down — NOT used by
            #            viewport; only used by resource_manager._decode_texture).
            #
            # FIX-YELLOWPIXEL: KotOR TPC textures (e.g. c_bantha01) sometimes have
            # bright saturated test/marker pixels in the top-right corner of the
            # original image (D3D V≈0, KotOR texture top).  In bottom-up PIL data,
            # the "top of texture" is the LAST rows of the array (highest row index).
            # These rows map to GL V≈1.0.  The shader maps KotOR V≈0 → GL V≈1.0,
            # so those yellow pixels are sampled on creature bellies (KotOR V near 0).
            # Fix: scan the last 2 pixel rows of the bottom-up PIL image (= top of
            # texture = GL V≈1.0) for yellow/green marker pixels and neutralize them.
            if _NUMPY:
                import numpy as _np
                _arr = _np.array(rgba, dtype=_np.uint8)  # (H, W, 4)
                _H, _W = _arr.shape[:2]
                # Inspect last 2 rows (= top of texture in bottom-up layout = GL V≈1)
                for _row_offset in range(min(2, _H)):
                    _row = _H - 1 - _row_offset
                    for _col in range(_W):
                        r, g, b, a = int(_arr[_row, _col, 0]), int(_arr[_row, _col, 1]), int(_arr[_row, _col, 2]), int(_arr[_row, _col, 3])
                        # Detect bright saturated yellow-green: R and G both high, B low
                        _is_yellow = (r > 160 and g > 160 and b < 100 and (r + g) > 2 * b + 200)
                        if _is_yellow:
                            # Replace with left neighbor if available, else with (64,50,40,255)
                            if _col > 0:
                                _arr[_row, _col] = _arr[_row, _col - 1]
                            else:
                                _arr[_row, _col] = [64, 50, 40, 255]
                from PIL import Image as _PILImg
                rgba = _PILImg.fromarray(_arr, 'RGBA')
            w, h = rgba.size
            data = rgba.tobytes()
            tex = self._ctx.texture((w, h), 4, data)
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            # FIX-TEXWRAP: Default to GL_REPEAT for all uploaded textures.
            # KotOR area geometry uses UV coordinates well outside [0,1] (e.g.
            # N_sithpraet pelvis U=[-13,+13]) and relies on the texture wrapping
            # to repeat correctly.  GL_CLAMP_TO_EDGE causes the edge texel to be
            # stretched across the entire surface for any UV outside [0,1], which
            # makes tiled geometry look solid-colored or edge-stretched.
            # Per-node clamp mode (TXI 'clamp' command → txi_clamp_s/txi_clamp_t)
            # is applied in _draw_node by setting repeat_x/repeat_y before each
            # draw call — this allows head/decal textures to use CLAMP_TO_EDGE
            # while body/area textures use GL_REPEAT.
            # The yellow-pixel artifact is handled by scrubbing marker pixels before
            # upload (above), so GL_REPEAT is safe.
            tex.repeat_x = True
            tex.repeat_y = True
            # FIX-MIPALIGN: Clamp mip chain at level 6 (coarsest = 8×8 for 512px)
            # to prevent single corner pixel colors from dominating lower LODs.
            # max_level=6 keeps 512→256→128→64→32→16→8 (7 levels, idx 0-6).
            mip_cap = self._mip_cap(w, h)
            tex.build_mipmaps(max_level=mip_cap)
            return tex
        except Exception as e:
            log.debug(f"_GlTexCache._upload failed: {e}")
            return None

    @staticmethod
    def _mip_cap(width: int, height: int) -> int:
        max_dim = max(int(width), int(height))
        return min(6, max(0, int(math.log2(max_dim)) - 2)) if max_dim > 4 else 0

    def update_regions(self, img: Optional['Image.Image'], regions, *, build_mipmaps: bool = True) -> bool:
        """Write bottom-up RGBA rectangles into one resident GL texture.

        ``regions`` are ``(x, y, width, height)`` rectangles in the cached PIL
        image's coordinates.  Cached KOTOR images and OpenGL both use row zero
        as the bottom row, so bytes are uploaded directly with no vertical flip.
        Returns ``False`` when the image has not been uploaded yet; the next
        normal render will then perform the initial full upload.
        """
        if img is None or not _PIL:
            return False
        key = id(img)
        entry = self._cache.get(key)
        if entry is None:
            return False
        wr, tex = entry
        if wr() is not img:
            try:
                tex.release()
            except Exception:
                pass
            del self._cache[key]
            return False

        try:
            rgba = img if getattr(img, "mode", "") == "RGBA" else img.convert("RGBA")
            image_width, image_height = (int(v) for v in rgba.size)
            texture_size = tuple(getattr(tex, "size", (image_width, image_height)))
            if texture_size[:2] != (image_width, image_height):
                return False

            wrote = False
            for region in regions or ():
                try:
                    x, y, width, height = (int(value) for value in region)
                except (TypeError, ValueError):
                    continue
                if width <= 0 or height <= 0:
                    continue
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(image_width, x + width)
                y1 = min(image_height, y + height)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = rgba.crop((x0, y0, x1, y1))
                payload = crop.tobytes()
                tex.write(
                    payload,
                    viewport=(x0, y0, x1 - x0, y1 - y0),
                    alignment=1,
                )
                self.region_update_count += 1
                self.region_update_bytes += len(payload)
                wrote = True
            if wrote and not build_mipmaps:
                # The existing mip chain now describes the previous base level.
                # Sample level zero only until stroke finalization rebuilds it;
                # otherwise minified surfaces visibly lag behind the brush.
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            if build_mipmaps and (wrote or not tuple(regions or ())):
                # Regenerate the existing GPU mip chain without re-uploading
                # the full base level or replacing the texture/bindings. Live
                # paint defers this until stroke end to avoid one full chain
                # rebuild per pointer/display frame.
                tex.build_mipmaps(max_level=self._mip_cap(image_width, image_height))
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._cache.move_to_end(key)
            return True
        except Exception as exc:
            log.debug("_GlTexCache.update_regions failed: %s", exc)
            return False

    def invalidate(self, img: Optional['Image.Image']) -> None:
        if img is None:
            return
        key = id(img)
        if key in self._cache:
            _, tex = self._cache[key]
            try:
                tex.release()
            except Exception:
                pass
            del self._cache[key]

    def clear(self) -> None:
        for (_, tex) in self._cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh geometry builder
# ─────────────────────────────────────────────────────────────────────────────

_PREBUILT_STATIC_MESH_ATTR = "_gr_gpu_prebuilt_static_mesh"


def clear_prebuilt_static_gpu_mesh_data(node) -> None:
    """Drop RAM-cached static GPU mesh data for a node and its descendants."""
    stack = [node]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        cid = id(cur)
        if cid in seen:
            continue
        seen.add(cid)
        try:
            if hasattr(cur, _PREBUILT_STATIC_MESH_ATTR):
                delattr(cur, _PREBUILT_STATIC_MESH_ATTR)
        except Exception:
            pass
        try:
            stack.extend(getattr(cur, "children", []) or [])
        except Exception:
            pass


def clear_prebuilt_static_gpu_model_data(model) -> int:
    """Drop all RAM-cached static GPU mesh data attached to a model."""
    if model is None:
        return 0
    try:
        nodes = list(model.all_nodes())
    except Exception:
        root = getattr(model, "root_node", None)
        nodes = [root] if root is not None else []
    cleared = 0
    for node in nodes:
        try:
            if hasattr(node, _PREBUILT_STATIC_MESH_ATTR):
                delattr(node, _PREBUILT_STATIC_MESH_ATTR)
                cleared += 1
        except Exception:
            pass
    try:
        setattr(model, "_gr_gpu_prebuilt_mesh_count", 0)
    except Exception:
        pass
    return cleared


def _prebuilt_static_gpu_mesh_data(node, model_id: int, skin_bind_transform: bool):
    try:
        entry = getattr(node, _PREBUILT_STATIC_MESH_ATTR, None)
    except Exception:
        return None
    if not entry:
        return None
    if entry.get("model_id") != model_id:
        return None
    if bool(entry.get("skin_bind_transform")) != bool(skin_bind_transform):
        return None
    vdata = entry.get("vdata")
    if vdata is None:
        return None
    return vdata, entry.get("idx_arr")


def prebuild_static_gpu_mesh_data(model) -> int:
    """Build static mesh VBO arrays and viewport bounds before GUI render."""
    if model is None or not _NUMPY:
        return 0
    try:
        nodes = list(model.all_nodes())
    except Exception:
        return 0
    model_id = id(model)
    model_cls = (str(getattr(model, "classification", "character") or "character")).lower()
    model_type_raw = getattr(model, "model_type", None)
    try:
        model_type = int(model_type_raw) if model_type_raw is not None else 4
    except Exception:
        model_type = 4
    is_module = model_cls in ("effect", "tile", "other") or model_type in (0, 2)
    built = 0
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for node in nodes:
        try:
            if not (getattr(node, "vertices", None) and getattr(node, "faces", None)):
                continue
            wp, wo = node.world_transform()
            skin_bind_transform = bool(getattr(node, "is_skin", False))
            vdata, idx_arr = _build_vbo_data(
                node,
                wp,
                wo,
                anim_pose_node=None,
                is_module=is_module,
                bone_index_remap=None,
                apply_skin_node_transform_for_bind=skin_bind_transform,
            )
            if vdata is None:
                continue
            try:
                pos = vdata[:, 0:3].astype(np.float64, copy=False)
                if pos.size:
                    bounds_min = np.minimum(bounds_min, pos.min(axis=0))
                    bounds_max = np.maximum(bounds_max, pos.max(axis=0))
            except Exception:
                pass
            setattr(
                node,
                _PREBUILT_STATIC_MESH_ATTR,
                {
                    "model_id": model_id,
                    "skin_bind_transform": skin_bind_transform,
                    "vdata": vdata,
                    "idx_arr": idx_arr,
                },
            )
            built += 1
        except Exception:
            continue
    try:
        setattr(model, "_gr_gpu_prebuilt_mesh_count", built)
        if built > 128 or is_module:
            setattr(model, "_gr_defer_txi_metadata", True)
        if built and np.isfinite(bounds_min).all() and np.isfinite(bounds_max).all():
            bb_min = tuple(float(v) for v in bounds_min.tolist())
            bb_max = tuple(float(v) for v in bounds_max.tolist())
            setattr(model, "_gr_render_bounds", (bb_min, bb_max))
            setattr(model, "_gr_bounds_prepared", True)
            model.bb_min = bb_min
            model.bb_max = bb_max
    except Exception:
        pass
    return built


def _is_redundant_c_kath_hair_preview_shell(node) -> bool:
    """Return True for K1 ``c_kath``'s duplicate striped mane shell.

    ``Frnt`` already carries the creature's correctly textured mane. The
    additional retail ``Hair01`` skin uses coarse atlas cards whose unrelated
    body regions read as broken UVs in a modern free-camera preview. Suppress
    only this exact GPU upload; import, validation, and export retain the
    original node and authored UV stream unchanged.
    """

    if str(getattr(node, "name", "") or "").strip().casefold() != "hair01":
        return False
    if not bool(getattr(node, "is_skin", False)):
        return False
    if str(getattr(node, "texture", "") or "").strip().casefold() != "c_khounda02":
        return False
    vertices = getattr(node, "vertices", [])
    faces = getattr(node, "faces", [])
    uvs = getattr(node, "uvs", [])
    if (
        len(vertices) != _RETAIL_C_KATH_HAIR_VERTEX_COUNT
        or len(faces) != _RETAIL_C_KATH_HAIR_FACE_COUNT
        or len(uvs) != _RETAIL_C_KATH_HAIR_VERTEX_COUNT
    ):
        return False

    root = node
    seen: set[int] = set()
    while getattr(root, "parent", None) is not None and id(root) not in seen:
        seen.add(id(root))
        root = root.parent
    if str(getattr(root, "name", "") or "").strip().casefold() != "c_kath":
        return False

    digest = hashlib.sha256()
    try:
        for uv in uvs:
            digest.update(struct.pack("<2f", float(uv[0]), float(uv[1])))
    except (IndexError, OverflowError, TypeError, ValueError):
        return False
    return digest.hexdigest() == _RETAIL_C_KATH_HAIR_UV_SHA256


def _build_vbo_data(node, world_pos: tuple, world_orient: tuple,
                    anim_pose_node=None,
                    is_module: bool = False,
                    bone_index_remap: Optional[Dict[int, int]] = None,
                    local_scale: float | tuple[float, float, float] | None = None,
                    apply_skin_node_transform_for_bind: bool = True) -> Tuple[Optional[np.ndarray],
                                                      Optional[np.ndarray]]:
    """
    Build interleaved VBO data for a ModelNode using vectorized NumPy.

    Returns (vertex_array, index_array) as float32/uint32 numpy arrays,
    or (None, None) on failure.

    Vertex layout per vertex (stride = 22 floats = 88 bytes):
      pos.xyz       3 floats  [0:3]
      norm.xyz      3 floats  [3:6]
      uv.xy         2 floats  [6:8]   (UV0 — primary/diffuse)
      uv_lm.xy      2 floats  [8:10]  (UV1 — lightmap)
      color.xyzw    4 floats  [10:14] (vertex colour + per-vertex alpha)
      bone_ids.xyzw 4 floats  [14:18] (bone palette indices, as float)
      weights.xyzw  4 floats  [18:22] (blend weights, sum ~ 1.0)

    Vectorization notes
    -------------------
    Previously used a Python loop (O(n_verts) iterations).  Now:
    - np.asarray converts vertex/normal/UV lists to arrays in one call
    - Quaternion rotation applied as a batch Nx3 vectorized op
    - Index arrays built with np.asarray on flat face lists
    This reduces build time from ~800 ms (10 k tris, pure Python) to
    ~5 ms (same mesh, vectorized NumPy).
    """
    if not _NUMPY:
        return None, None

    if _is_redundant_c_kath_hair_preview_shell(node):
        return None, None

    verts = getattr(node, 'vertices', getattr(node, 'verts', []))
    norms = getattr(node, 'normals', [])
    uvs   = getattr(node, 'uvs', [])
    uvs_lm = getattr(node, 'uvs_lm', [])
    faces  = getattr(node, 'faces', [])
    face_uvs = getattr(node, 'face_uvs', [])
    is_skin = bool(getattr(node, 'is_skin', False))

    # FIX-SABER-PROXY: Stock KOTOR blade cards are not ordinary triangle
    # meshes.  Their sparse, segmented strips expose long diagonal wedges in a
    # conventional renderer.  Build one render-only quad from each card's
    # authored local bounds so the procedural core/aura texture stays smooth.
    # This changes only GPU upload data; the loaded MDL/MDX model remains
    # untouched for animation, validation, and export.
    if is_lightsaber_blade_node(node):
        blade_proxy = lightsaber_blade_preview_quad(verts, edge_inset=0.0)
        if blade_proxy is not None:
            proxy_verts, proxy_uvs, proxy_faces, proxy_normal = blade_proxy
            verts = proxy_verts
            norms = [proxy_normal] * len(proxy_verts)
            uvs = proxy_uvs
            uvs_lm = []
            faces = proxy_faces
            face_uvs = proxy_faces
            is_skin = False

    n_verts = len(verts)
    n_faces = len(faces)
    if n_verts == 0 or n_faces == 0:
        return None, None

    # World transform
    wp = np.array(world_pos if world_pos else (0.0, 0.0, 0.0), dtype=np.float64)
    wo = np.array(world_orient if world_orient else (0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    # Normalize quaternion for safety
    qlen = np.linalg.norm(wo)
    if qlen > 1e-9:
        wo = wo / qlen
    is_identity_rot = (abs(wo[3]) > 0.9999 and
                       abs(wo[0]) < 1e-6 and abs(wo[1]) < 1e-6 and abs(wo[2]) < 1e-6)
    is_identity_pos = (abs(wp[0]) < 1e-9 and abs(wp[1]) < 1e-9 and abs(wp[2]) < 1e-9)

    # ── Convert vertices and normals to Nx3 float64 arrays ──────────────────
    try:
        v_arr = np.asarray(verts, dtype=np.float64)
        if v_arr.ndim != 2 or v_arr.shape[1] != 3:
            v_arr = np.zeros((n_verts, 3), dtype=np.float64)
    except (ValueError, TypeError):
        v_arr = np.zeros((n_verts, 3), dtype=np.float64)
    if local_scale is not None:
        try:
            if isinstance(local_scale, (tuple, list)):
                scale_vec = np.asarray(
                    [
                        float(local_scale[0]),
                        float(local_scale[1]),
                        float(local_scale[2]),
                    ],
                    dtype=np.float64,
                )
            else:
                value = float(local_scale)
                scale_vec = np.asarray([value, value, value], dtype=np.float64)
            if np.all(np.isfinite(scale_vec)):
                v_arr = v_arr * scale_vec
        except Exception:
            pass

    n_norms = len(norms)
    if n_norms > 0:
        try:
            n_arr = np.asarray(norms[:n_verts], dtype=np.float64)
            if n_arr.ndim != 2 or n_arr.shape[1] != 3:
                n_arr = np.zeros((n_verts, 3), dtype=np.float64)
                n_arr[:, 2] = 1.0
        except (ValueError, TypeError):
            n_arr = np.zeros((n_verts, 3), dtype=np.float64)
            n_arr[:, 2] = 1.0
        # Pad if fewer normals than verts
        if len(n_arr) < n_verts:
            pad = np.zeros((n_verts - len(n_arr), 3), dtype=np.float64)
            pad[:, 2] = 1.0
            n_arr = np.vstack([n_arr, pad])
    else:
        n_arr = np.zeros((n_verts, 3), dtype=np.float64)
        n_arr[:, 2] = 1.0

    # ── D20-M: Apply world transform using per-node vertex_space contract ────
    #
    # Every node's vertex_space is set at load time by compute_vertex_space()
    # in src/core/vertex_space.py.  This replaces ALL centroid-magnitude
    # heuristics, _WORLDSPACE_VERT_THRESHOLD, FIX-PROXY-THRESHOLD,
    # FIX-ACCESSORY-WORLDSPACE, and FIX-CREATURE-WORLDSPACE checks.
    #
    # Rules (matching xoreos model_kotor.cpp, KotOR.js OdysseyModel3D.ts):
    #   NODE_LOCAL (0): vertices are in node's local coordinate system.
    #                   Apply full world_transform (rotate + translate).
    #   WORLD (1):      vertices already in model-root space (imported OBJ/FBX).
    #                   Do NOT apply world_transform.
    #   AABB_WALK (2):  walkmesh/collision — not rendered (should never reach here).
    #
    # References:
    #   xoreos model_kotor.cpp readMesh/readSkin — raw MDX read, no transform
    #   KotOR.js OdysseyModelNodeMesh.ts — raw push to array
    #   KotOR.js OdysseyModel3D.ts NodeMeshBuilder — matrixWorld for GPU
    #   reone mdlreader.cpp — reads verts as-is, applies node transform in renderer
    #   KotorBlender parser.py — verts loaded into Blender mesh, Blender applies hierarchy
    _node_vs = (
        1
        if bool(getattr(node, '_gr_vertices_in_kotor_world', False))
        else getattr(node, 'vertex_space', 0)
    )  # default NODE_LOCAL

    _apply_node_transform = (
        _node_vs == 0
        and (not is_skin or bool(apply_skin_node_transform_for_bind))
    )

    if _apply_node_transform:  # NODE_LOCAL: apply one full world transform
        if not is_identity_rot:
            v_arr = _quat_rotate_batch(wo, v_arr)
            n_arr = _quat_rotate_batch(wo, n_arr)
        if not is_identity_pos:
            v_arr = v_arr + wp
    elif _node_vs == 1 or is_skin:  # WORLD or animated-skin input: pass through
        pass
    # _node_vs == 2 (AABB_WALK) should never reach VBO building

    # FIX-COMPOSITE-OFFSET: For nodes that belong to the head-accessory model
    # inside a _CompositeModel (e.g. head skin, eyeRA, teethUa in ad_saul),
    # apply the skeleton rebase offset so they render at the correct position
    # in the body skeleton.  The offset (_composite_nonskin_offset — legacy
    # name kept for attribute stability) is pre-computed by _CompositeModel
    # as: body_head_g_world - head_head_g_local, and is only attached to nodes
    # whose ownership is the head model (body nodes never carry this attr),
    # so no ownership gate is needed here beyond the None check.
    #
    # Historical bug: the previous gate excluded is_skin nodes, which left the
    # head *skin* mesh at floor level while eyes/teeth floated at head height.
    # Probe-verified fix: apply offset to all head-accessory nodes including
    # the skin, so the whole head moves as one with the body headhook.
    _cns_off = getattr(node, '_composite_nonskin_offset', None)
    if _cns_off is not None:
        _ox, _oy, _oz = float(_cns_off[0]), float(_cns_off[1]), float(_cns_off[2])
        if abs(_ox) > 1e-6 or abs(_oy) > 1e-6 or abs(_oz) > 1e-6:
            v_arr = v_arr + np.array([_ox, _oy, _oz], dtype=np.float64)

    _gr_gpu_probe(node, world_pos or (0.0, 0.0, 0.0),
                  world_orient or (0.0, 0.0, 0.0, 1.0),
                  is_identity_rot, _cns_off)

    # ── Normalize normals (prevent shading errors from non-unit normals) ─────
    # After quaternion rotation and world-space transform, normals may drift from
    # unit length.  Normalize each row to ensure correct Phong shading.
    n_lens = np.linalg.norm(n_arr, axis=1, keepdims=True)
    n_lens = np.where(n_lens < 1e-9, 1.0, n_lens)  # avoid div-by-zero
    n_arr = n_arr / n_lens

    # ── UV arrays ────────────────────────────────────────────────────────────
    # FIX-UV-SEAM-EXPAND: Build uv_arr from the FULL uvs list (n_uvs entries),
    # NOT truncated to n_verts.  KotOR binary MDL trimeshes use separate tvert
    # (texture-vertex) indices in face_uvs[] that can reference UV entries
    # beyond index n_verts-1 — the seam-split duplicate UVs are appended after
    # the geometry verts.  Truncating to n_verts discarded those extra entries,
    # causing the expanded-path UV lookup (uv_arr[ti]) to fall back to the
    # vertex UV (uv_arr[vi]) instead of the correct seam UV, producing wrong
    # texture coordinates on every seam-split corner (e.g. quad row4 V=1.0
    # instead of V=0.99 when face_uvs=[..,[0,4,3]] and uvs[4]=(1.0,0.99)).
    #
    # Seam-vert healing (below) uses v_arr for spatial lookup which is indexed
    # by vertex index (0..n_verts-1), so we restrict the heal pass to entries
    # 0..n_verts-1; extra tvert-only entries (n_verts..n_uvs-1) are not
    # directly linked to a geometry position and must not be healed by position.
    n_uvs = len(uvs)
    if n_uvs > 0:
        try:
            uv_arr = np.asarray(uvs, dtype=np.float32)      # shape: (n_uvs, ...)
            if uv_arr.ndim != 2 or uv_arr.shape[1] < 2:
                uv_arr = np.full((n_uvs, 2), 0.5, dtype=np.float32)
            elif uv_arr.shape[1] > 2:
                uv_arr = uv_arr[:, :2]
        except (ValueError, TypeError):
            uv_arr = np.full((n_uvs, 2), 0.5, dtype=np.float32)
        bad_uv = ~np.all(np.isfinite(uv_arr), axis=1)
        if np.any(bad_uv):
            uv_arr[bad_uv] = 0.5
    else:
        synthetic = []
        if is_lightsaber_blade_node(node):
            synthetic = synthetic_lightsaber_blade_uvs(verts)
        if len(synthetic) == n_verts:
            uv_arr = np.asarray(synthetic, dtype=np.float32)
        else:
            uv_arr = np.full((n_verts, 2), 0.5, dtype=np.float32)
        n_uvs = n_verts  # for consistent indexing below

    # UV1/lightmap coordinates follow the same texture-vertex indexing rules as
    # UV0. Keep the full array for expanded per-face lookup; slice/pad only when
    # filling the vertex-indexed fast-path VBO.
    n_uvs_lm = len(uvs_lm)
    if n_uvs_lm > 0:
        try:
            uv_lm_src_arr = np.asarray(uvs_lm, dtype=np.float32)
            if uv_lm_src_arr.ndim != 2 or uv_lm_src_arr.shape[1] < 2:
                uv_lm_src_arr = np.full((n_uvs_lm, 2), 0.5, dtype=np.float32)
            elif uv_lm_src_arr.shape[1] > 2:
                uv_lm_src_arr = uv_lm_src_arr[:, :2]
        except (ValueError, TypeError):
            uv_lm_src_arr = np.full((n_uvs_lm, 2), 0.5, dtype=np.float32)
        bad_lm = ~np.all(np.isfinite(uv_lm_src_arr), axis=1)
        if np.any(bad_lm):
            uv_lm_src_arr[bad_lm] = 0.5
    else:
        uv_lm_src_arr = np.full((n_verts, 2), 0.5, dtype=np.float32)
        n_uvs_lm = n_verts

    # ── Assemble interleaved vertex buffer (N × 22) ──────────────────────────
    # Phase A: Extended VBO layout includes bone_ids (4f) + bone_weights (4f)
    # for GPU skinning.  All nodes use 22 floats per vertex for a consistent
    # VAO format; non-skin nodes get identity values (idx=0, weight=1,0,0,0).
    #
    # Layout per vertex (stride = 22 floats = 88 bytes):
    #   pos.xyz       3 floats  [0:3]
    #   norm.xyz      3 floats  [3:6]
    #   uv.xy         2 floats  [6:8]   (UV0 — primary/diffuse)
    #   uv_lm.xy      2 floats  [8:10]  (UV1 — lightmap)
    #   color.xyzw    4 floats  [10:14] (vertex colour + per-vertex alpha)
    #   bone_ids.xyzw 4 floats  [14:18] (bone palette indices, as float)
    #   weights.xyzw  4 floats  [18:22] (blend weights, sum ≈ 1.0)
    #
    # vdata is keyed by *vertex* index (0..n_verts-1).  uv_arr may have more
    # rows than n_verts (extra seam-split tvert entries beyond n_verts-1) so we
    # slice to exactly n_verts rows when copying into vdata.
    vdata = np.zeros((n_verts, 22), dtype=np.float32)
    vdata[:, 0:3] = v_arr.astype(np.float32)
    vdata[:, 3:6] = n_arr.astype(np.float32)
    _uv_for_vdata = uv_arr[:n_verts] if len(uv_arr) >= n_verts else np.vstack(
        [uv_arr, np.full((n_verts - len(uv_arr), 2), 0.5, dtype=np.float32)])
    _uv_lm_for_vdata = uv_lm_src_arr[:n_verts] if len(uv_lm_src_arr) >= n_verts else np.vstack(
        [uv_lm_src_arr, np.full((n_verts - len(uv_lm_src_arr), 2), 0.5, dtype=np.float32)])
    vdata[:, 6:8]  = _uv_for_vdata[:, :2]
    vdata[:, 8:10] = _uv_lm_for_vdata[:, :2]
    vdata[:, 10:14] = 1.0  # white vertex colour + alpha 1

    # ── Phase A: Populate bone_ids and weights for skin nodes ────────────────
    # bone_ids[14:18] = palette indices (float);  weights[18:22] = blend weights.
    # Non-skin nodes: identity (idx=0, weight=[1,0,0,0]) — shader pass-through.
    #
    # FIX-SKIN-QBONE: BoneWeight.bone_index is a LOCAL index into this skin
    # node's bone_map[] array.  The renderer now uploads a per-skin local
    # qBone/tBone palette before each skin draw, so no remap is needed in the
    # normal path.  The optional remap remains for focused tests and diagnostics.
    if is_skin:
        _skin_data = getattr(node, 'skin_data', [])
        if _skin_data and len(_skin_data) >= n_verts:
            for _vi in range(n_verts):
                _sd = _skin_data[_vi]
                _infl = getattr(_sd, 'influences', [])
                for _bi_idx in range(min(4, len(_infl))):
                    _bw = _infl[_bi_idx]
                    _local_idx = int(getattr(_bw, 'bone_index', 0))
                    # Optional remap for tests/diagnostics; production skin draws
                    # keep local bone_map indices for the per-skin qBone palette.
                    if bone_index_remap is not None:
                        _palette_idx = bone_index_remap.get(_local_idx, 0)
                    else:
                        _palette_idx = _local_idx
                    vdata[_vi, 14 + _bi_idx] = float(_palette_idx)
                    vdata[_vi, 18 + _bi_idx] = float(getattr(_bw, 'weight', 0.0))
        else:
            # No per-vertex skin data: identity (bone 0, weight 1.0)
            vdata[:, 18] = 1.0  # first weight = 1.0
    else:
        # Non-skin node: identity skinning values (bone 0, weight 1.0)
        vdata[:, 18] = 1.0  # first weight = 1.0

    # ── Build face index arrays ───────────────────────────────────────────────
    # FIX-SEAM: KotOR binary MDL uses per-face tvert (texture-vertex) indices
    # that are stored separately from the geometry vertex indices.  The NWN
    # exporter's ProcessSkinSeams() duplicates geometry vertices at UV seams so
    # that each half of a seam can have a different UV coordinate.  When
    # face_uvs is present (len == n_faces), every face has its own 3 tvert
    # indices that index into the uvs[] array independently of the faces[]
    # vertex indices.  We must therefore expand to a triangle-list (no IBO) so
    # that each triangle corner gets the correct per-face UV.
    #
    # Additionally, skin nodes always need expansion because the engine's vertex
    # layout may not match the face list order (seam verts are duplicated with
    # different UV but identical position).
    #
    # Reference: PyKotor io_mdl.py ProcessSkinSeams engine comment;
    #            PyKotor read_mdl.py gl_load_stitched_model (expands per-face).
    _has_face_uvs = (len(face_uvs) == n_faces)

    # FIX-FACEUVOPT-V2: When face_uvs has the same length as faces but all
    # tvert indices equal the corresponding vertex indices (the t=-1→use_vert
    # case from binary MDL trimeshes), treat it as "no per-face UV" so we can
    # use the fast IBO path.  This avoids unnecessarily expanding a triangle-
    # list for module/area models whose binary trimeshes always have t==-1.
    #
    # v2: Replaced the O(n) Python loop (which was capped at n_faces<=2000 to
    # avoid slowness) with a vectorized NumPy comparison that runs in O(1)
    # time for any mesh size.  Module room meshes commonly have 3000-10000
    # faces; the previous cap forced them through the expanded path
    # unnecessarily, wasting ~3x VBO memory and blocking per-material-slot
    # IBO construction (since expanded meshes are non-indexed).
    if _has_face_uvs and not is_skin:
        try:
            _fuv_arr = np.asarray(face_uvs, dtype=np.int32)  # (n_faces, 3)
            _fv_arr  = np.asarray(faces,    dtype=np.int32)   # (n_faces, 3)
            if (_fuv_arr.shape == _fv_arr.shape and
                    _fuv_arr.ndim == 2 and _fuv_arr.shape[1] >= 3):
                if np.array_equal(_fuv_arr[:, :3], _fv_arr[:, :3]):
                    _has_face_uvs = False
        except (ValueError, TypeError):
            pass  # non-uniform lists — fall through to expanded path

    # Fast path: no per-face UV indices AND not a skin node → use IBO
    if not _has_face_uvs and not is_skin:
        try:
            f_arr = np.asarray(faces, dtype=np.int32)
            if f_arr.ndim == 2 and f_arr.shape[1] >= 3:
                idx = f_arr[:, :3]  # (N_faces, 3)
                # Filter out-of-range indices
                valid = (np.all(idx >= 0, axis=1) &
                         np.all(idx < n_verts, axis=1))
                idx = idx[valid].astype(np.uint32)
                if len(idx) == 0:
                    return None, None
                try:
                    setattr(node, '_gr_last_vbo_source_indices', list(range(n_verts)))
                except Exception:
                    pass
                return vdata, idx.flatten()
        except (ValueError, TypeError):
            pass
        # Fallback: Python loop for jagged/non-uniform face list
        idx_list = []
        for face in faces:
            if len(face) < 3:
                continue
            vi0, vi1, vi2 = int(face[0]), int(face[1]), int(face[2])
            if vi0 < n_verts and vi1 < n_verts and vi2 < n_verts:
                idx_list.extend([vi0, vi1, vi2])
        if not idx_list:
            return None, None
        try:
            setattr(node, '_gr_last_vbo_source_indices', list(range(n_verts)))
        except Exception:
            pass
        return vdata, np.array(idx_list, dtype=np.uint32)

    # Expanded path: per-face UV indices OR skin mesh → expand to triangle list
    # Each triangle corner is a (position, normal, uv) triple taken from the
    # correct vertex/tvert index pair.  No IBO needed.
    #
    # FIX-EXPANDUV: Use sanitized uv_arr (already seam-healed and sentinel-
    # checked) as the UV source, NOT the raw uvs[] list.  This ensures that
    # the seam-healing pass (which corrects duplicate-vertex UVs at seams like
    # p_hk47 fingers and c_kraytdragon claws) is applied to the final vertices.
    # Previously the expanded path re-read raw uvs[ti], bypassing seam healing.
    # Now it reads uv_arr[ti] which is the sanitized version.
    # Exception: when ti != vi (genuine per-face UV index differs from vertex
    # index), we must use the correct tvert UV from uv_arr[ti].  This is the
    # KotOR ProcessSkinSeams() case.
    expanded_rows = []
    expanded_source_indices = []
    n_uv_arr = len(uv_arr)   # sanitized UV array (already seam-healed)
    n_uv_lm_arr = len(uv_lm_src_arr)  # sanitized UV1/lightmap array
    for fi, face in enumerate(faces):
        if len(face) < 3:
            continue
        vi0, vi1, vi2 = face[0], face[1], face[2]
        if vi0 >= n_verts or vi1 >= n_verts or vi2 >= n_verts:
            continue
        # Use per-face UV indices if available, else fall back to vertex index
        if _has_face_uvs and fi < len(face_uvs):
            fuvs = face_uvs[fi]
            ti0, ti1, ti2 = (int(fuvs[0]), int(fuvs[1]), int(fuvs[2])) if len(fuvs) >= 3 else (vi0, vi1, vi2)
        else:
            ti0, ti1, ti2 = vi0, vi1, vi2
        for vi, ti in ((vi0, ti0), (vi1, ti1), (vi2, ti2)):
            row = vdata[vi].copy()
            # FIX-EXPANDUV: Use sanitized uv_arr[ti] instead of raw uvs[ti].
            # uv_arr is already sentinel-checked and seam-healed.
            if 0 <= ti < n_uv_arr:
                row[6] = uv_arr[ti, 0]
                row[7] = uv_arr[ti, 1]
            elif 0 <= vi < n_uv_arr:
                # tvert index out of range → fall back to vertex UV
                row[6] = uv_arr[vi, 0]
                row[7] = uv_arr[vi, 1]
            # else: keep row's UV from vdata[vi] (already set from uv_arr[vi])
            if 0 <= ti < n_uv_lm_arr:
                row[8] = uv_lm_src_arr[ti, 0]
                row[9] = uv_lm_src_arr[ti, 1]
            elif 0 <= vi < n_uv_lm_arr:
                row[8] = uv_lm_src_arr[vi, 0]
                row[9] = uv_lm_src_arr[vi, 1]
            # else: keep row's UV1 from vdata[vi]
            expanded_rows.append(row)
            expanded_source_indices.append(int(vi))

    if not expanded_rows:
        return None, None
    try:
        setattr(node, '_gr_last_vbo_source_indices', expanded_source_indices)
    except Exception:
        pass
    return np.stack(expanded_rows).astype(np.float32), None


#  GPU mesh buffer (one per node, cached)
# ─────────────────────────────────────────────────────────────────────────────

class _GpuMesh:
    """Holds the VBO / VAO / IBO for one ModelNode.

    FIX-MULTITEX-SPLIT: For multi-material nodes (tex_count > 1 with per-face
    material slots), mat_slot_ibos stores per-material-slot IBOs so that each
    face group can be drawn with the correct texture without overdrawing the
    entire mesh.  mat_slot_vaos stores the corresponding VAOs.
    Format: {slot_idx: (vao, ibo, tri_count)}
    """

    def __init__(self):
        self.vao: Optional['moderngl.VertexArray'] = None
        self.vbo: Optional['moderngl.Buffer'] = None
        self.bone_id_vbo: Optional['moderngl.Buffer'] = None
        self.ibo: Optional['moderngl.Buffer'] = None
        self.tri_count: int = 0
        self.indexed: bool = False
        self.uploaded_vertex_count: int = 0
        self.first8_uv0_uploaded: List[List[float]] = []
        self.first8_uv1_uploaded: List[List[float]] = []
        self.uploaded_positions: List[List[float]] = []
        # Axis-aligned bounds in the exact coordinate space uploaded to the
        # VBO.  ModernGL uses these for conservative frustum rejection before
        # issuing per-node uniforms and draw calls.
        self.uploaded_bounds: Optional[tuple] = None
        self.uploaded_bone_ids: List[List[int]] = []
        self.uploaded_weights: List[List[float]] = []
        self.uploaded_source_indices: List[int] = []
        self.uv1_attribute_bound: bool = False
        self.skin_bind_transform: Optional[bool] = None
        self.skin_vbo_signature: Optional[tuple] = None
        # Retained scene animation switches rigid nodes from authored/world-
        # baked VBOs to node-local VBOs driven by a per-draw model matrix.  The
        # cache owner uses this bit to rebuild exactly once at that boundary.
        self.scene_rigid_gpu_pose: Optional[bool] = None
        # Per-material-slot draw groups for multi-texture nodes
        self.mat_slots: Dict[int, tuple] = {}  # {slot: (vao, ibo, tri_count)}

    def release(self):
        if self.vao:
            try: self.vao.release()
            except Exception: pass
            self.vao = None
        if self.vbo:
            try: self.vbo.release()
            except Exception: pass
            self.vbo = None
        if self.bone_id_vbo:
            try: self.bone_id_vbo.release()
            except Exception: pass
            self.bone_id_vbo = None
        if self.ibo:
            try: self.ibo.release()
            except Exception: pass
            self.ibo = None
        # Release per-material-slot VAOs and IBOs
        for slot_data in self.mat_slots.values():
            try:
                if slot_data[0]:  # vao
                    slot_data[0].release()
                if slot_data[1]:  # ibo
                    slot_data[1].release()
            except Exception:
                pass
        self.mat_slots.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  Main GPU renderer class
# ─────────────────────────────────────────────────────────────────────────────


__all__ = tuple(name for name in globals() if not name.startswith("__"))
