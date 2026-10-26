from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, Optional

from src.adapters.gpu.moderngl_context import _create_moderngl_standalone_context
from src.adapters.gpu.moderngl_runtime import (
    Image,
    MatrixPaletteUploader,
    _GPU_SKINNING,
    _MODERNGL,
    _NUMPY,
    _PIL,
    _SKIN_MAX_BONES,
    moderngl,
    np,
)
from src.core.lighting.light_gizmo_renderer import (
    LIGHT_HELPER_AREA_SIZE,
    LIGHT_HELPER_COLORS,
    LIGHT_HELPER_DIRECTION_LENGTH,
    LIGHT_HELPER_MARKER_RADIUS,
    LIGHT_HELPER_POINT_RADIUS,
    LIGHT_HELPER_SELECTED_BOOST,
    LIGHT_HELPER_SPOT_CAP_MAX_RADIUS,
    LIGHT_HELPER_SPOT_LENGTH,
)
from src.core.lighting.render_data import scene_light_relevance_key
from src.core.rendering.gpu_debug_tables import (
    ModuleDrawItem,
    debug_draw_table,
    debug_material_role_table,
    debug_texture_cache_table,
    debug_uv_channel_table,
)
from src.core.rendering.color_utils import _hex_to_rgb_float
from src.core.rendering.gpu_diagnostics_config import (
    _GL_DEBUG_ERRORS_ENV,
    _GL_STATE_TRACE_TRUE,
    _debug_visualize_mode,
    _gl_state_trace_path,
    _lm_composite_mode,
    _lm_data_dump_path,
    _skin_dump_path,
)
from src.core.rendering.gpu_diagnostics_records import (
    _append_gl_state_trace,
    _append_jsonl_record,
    _build_gl_state_trace_record,
    _build_lm_data_dump_record,
    _build_skin_dump_record,
    _first_vbo_uv_pairs,
    _should_auto_clamp_diffuse,
)
from src.core.rendering.gpu_vbo_layout import (
    _VBO_BONE_IDS_ATTRS,
    _VBO_BONE_IDS_FORMAT,
    _VBO_MAIN_ATTRS,
    _VBO_MAIN_FORMAT,
)
from src.core.geometry.lightsaber import (
    is_lightsaber_blade_node,
    lightsaber_blade_procedural_rgba8,
    lightsaber_blade_texture_cache_key,
    should_use_procedural_lightsaber_blade_texture,
)
from src.core.special.render_constants import INNER_GEO_SUBSTRINGS as _INNER_GEO_SUBSTRINGS
from src.math.gpu_math import (
    _bas_attachment_local_transform_np,
    _compose_world_transform_np,
    _mat3_normal,
    _mat4_from_pos_quat_scale,
    _mat4_identity,
    _mat4_lookat,
    _mat4_mul,
    _mat4_perspective,
    _mat4_tobytes,
    _quat_rotate_batch,
    _scene_authored_world_transform,
    _scene_gpu_model_matrix,
)
from src.adapters.rendering.moderngl_resources import (
    _GlTexCache,
    _GpuMesh,
    _build_vbo_data,
    _configure_lightsaber_blade_sampler,
    _is_saber_runtime_helper,
    _prebuilt_static_gpu_mesh_data,
    _split_vbo_attributes_for_gpu,
)
from src.core.rendering.mesh_render_data import (
    _animated_node_world_transform,
    _bas_attachment_world_transform,
    _effective_animation_pose_for_node,
    _node_has_renderable_rigid_surface,
    _node_is_embedded_walkmesh,
    _pose_node_for_transform,
    animation_pose_applies_to_node,
    animation_pose_for_node,
    bas_attachment_palette_model_for_node,
    runtime_source_model_for_node,
)
from src.core.rendering.skeleton_render_data import (
    bas_attachment_root_local_skin_palette,
    skin_palette_flat_bytes,
)
from src.core.rendering.renderer_performance import (
    bounds_intersects_frustum,
    extract_frustum_planes,
)
from src.core.rendering.gpu_shaders import _FRAG_SRC, _GRID_FRAG_SRC, _GRID_VERT_SRC, _VERT_SRC

log = logging.getLogger(__name__)


_SUBMISSION_CACHE_EMPTY = object()


def _effective_diffuse_uv_v_flip(node, diffuse_image) -> float:
    """Return the shader's single effective diffuse V-flip.

    TextureCache normalizes loose raster rows to OpenGL's bottom-up upload
    order. Native KOTOR/D3D UVs therefore request one shader inversion, while
    imported DCC/OpenGL UVs explicitly opt out with ``uv_v_flip=False``.
    Texture provenance may also opt out when its decoded rows already encode a
    bottom-left-authored loose-atlas profile.
    """

    node_uses_kotor_uvs = bool(getattr(node, "uv_v_flip", True))
    texture_needs_kotor_uv_conversion = bool(
        getattr(diffuse_image, "_gr_gpu_uv_v_flip", True)
    )
    return 1.0 if node_uses_kotor_uvs and texture_needs_kotor_uv_conversion else 0.0


def _uniform_value_stamp(value):
    """Return an exact, comparison-safe stamp for one uniform value."""
    if np is not None and isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, tuple(value.shape), value.tobytes())
    # ModernGL vector uniforms overwhelmingly arrive as immutable numeric
    # tuples.  Let CPython hash those tuples in C and retain the exact typed
    # value instead of recursively allocating a tagged tuple for every scalar.
    # The 207TEL PIE draw loop asks this cache about 14,000 times per frame;
    # the recursive form spent several milliseconds rebuilding stamps even
    # though almost every submission was skipped.  Unhashable containers keep
    # the recursive snapshot below so later mutation cannot corrupt equality.
    try:
        hash(value)
    except (TypeError, ValueError):
        pass
    else:
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_uniform_value_stamp(item) for item in value))
    if isinstance(value, list):
        return (list, tuple(_uniform_value_stamp(item) for item in value))
    return (type(value), value)


class _ExactUniformSubmission:
    """Uniform proxy that suppresses only byte/value-identical submissions."""

    def __init__(self, uniform, stats: dict[str, int], *, cache_writes: bool = True):
        self._uniform = uniform
        self._stats = stats
        self._cache_writes = bool(cache_writes)
        self._last_submission = _SUBMISSION_CACHE_EMPTY

    def reset_submission_cache(self) -> None:
        self._last_submission = _SUBMISSION_CACHE_EMPTY

    @property
    def value(self):
        return self._uniform.value

    @value.setter
    def value(self, value) -> None:
        submission = ("value", _uniform_value_stamp(value))
        if submission == self._last_submission:
            self._stats["uniform_skips"] += 1
            return
        self._uniform.value = value
        self._last_submission = submission
        self._stats["uniform_writes"] += 1

    def write(self, data) -> None:
        # Animated skin palettes are deliberately excluded from this generic
        # cache.  Their actor/pose/signature cache is owned by
        # _skin_palette_bytes_for_draw and remains the correctness gate.
        if not self._cache_writes:
            self._uniform.write(data)
            self._last_submission = _SUBMISSION_CACHE_EMPTY
            self._stats["uniform_writes"] += 1
            return
        payload = bytes(data)
        submission = ("bytes", payload)
        if submission == self._last_submission:
            self._stats["uniform_skips"] += 1
            return
        self._uniform.write(data)
        self._last_submission = submission
        self._stats["uniform_writes"] += 1

    def __getattr__(self, name):
        return getattr(self._uniform, name)


class _ExactBlendSubmission:
    """Track the last submitted GL blend state for one draw pass."""

    def __init__(self, stats: dict[str, int]):
        self._stats = stats
        self.reset()

    def reset(self) -> None:
        self._enabled = _SUBMISSION_CACHE_EMPTY
        self._equation = _SUBMISSION_CACHE_EMPTY
        self._func = _SUBMISSION_CACHE_EMPTY

    def apply(self, ctx, *, enabled: bool, equation=None, func=None) -> None:
        enabled = bool(enabled)
        if enabled != self._enabled:
            if enabled:
                ctx.enable(moderngl.BLEND)
            else:
                ctx.disable(moderngl.BLEND)
            self._enabled = enabled
            self._stats["blend_state_writes"] += 1
        else:
            self._stats["blend_state_skips"] += 1

        if not enabled:
            return
        if equation is not None:
            if equation != self._equation:
                ctx.blend_equation = equation
                self._equation = equation
                self._stats["blend_state_writes"] += 1
            else:
                self._stats["blend_state_skips"] += 1
        if func is not None:
            func = tuple(func)
            if func != self._func:
                ctx.blend_func = func
                self._func = func
                self._stats["blend_state_writes"] += 1
            else:
                self._stats["blend_state_skips"] += 1


def _strict_emitter_world_transform(node, anim_pose):
    """Strict Aurora FK world transform for emitter nodes.

    The shared mesh transform paths collapse 180°-about-axis bind rotations on
    parent nodes (a droid/character rendering workaround).  Emitter placement
    must NOT collapse them: K1 ``plc_starmap`` parents its star-field emitters
    under ``Dummy01`` with a real (1,0,0,0) 180° X flip that moves the stars
    from below the pedestal up into the dome.  Pose-node locals replace bind
    locals when the node is animated (NodePose positions are absolute locals).
    """
    import math as _math

    from src.core.geometry.model_data import _quat_mul, _quat_normalize, _quat_rotate
    from src.core.rendering.mesh_render_data import _pose_node_for_transform

    chain = []
    current = node
    seen: set = set()
    while current is not None and id(current) not in seen and len(chain) <= 512:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "parent", None)
    chain.reverse()

    wx = wy = wz = 0.0
    orientation = [0.0, 0.0, 0.0, 1.0]
    for chain_node in chain:
        pose_node = _pose_node_for_transform(chain_node, anim_pose) if anim_pose is not None else None
        if pose_node is not None:
            lx, ly, lz = getattr(pose_node, "position", chain_node.position)
            rot = list(getattr(pose_node, "rotation", chain_node.rotation))
        else:
            lx, ly, lz = getattr(chain_node, "position", (0.0, 0.0, 0.0))
            rot = list(getattr(chain_node, "rotation", (0.0, 0.0, 0.0, 1.0)))
        if not (_math.isfinite(lx) and _math.isfinite(ly) and _math.isfinite(lz)):
            lx, ly, lz = 0.0, 0.0, 0.0
        if not all(_math.isfinite(v) for v in rot):
            rot = [0.0, 0.0, 0.0, 1.0]
        rx, ry, rz = _quat_rotate(orientation, (lx, ly, lz))
        wx += rx
        wy += ry
        wz += rz
        orientation = _quat_normalize(_quat_mul(orientation, _quat_normalize(rot)))
    return (float(wx), float(wy), float(wz)), tuple(float(v) for v in orientation)


def _is_untextured_glow(node) -> bool:
    """Untextured self-illuminated planes are additive glows in the engine.

    The Star Map's ``lightflare`` ignition burst and its holo ``Object*``
    planes are authored with ``texture='null'`` plus a selfillum color.
    Routing them through the opaque/cutout passes rasterizes hard-edged solid
    white geometry (the giant white wedges seen during ``off2on``); retail
    composites them as soft additive selfillum glows.
    """
    tex = str(getattr(node, 'texture', '') or '').strip().lower()
    if tex and tex not in ('null', 'none', '****'):
        return False
    selfillum = getattr(node, 'selfillum', (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    try:
        return (float(selfillum[0]) + float(selfillum[1]) + float(selfillum[2])) > 0.05
    except Exception:
        return False


def _set_depth_write(ctx, enabled: bool) -> None:
    """Set glDepthMask through the bound framebuffer.

    moderngl (5.12) exposes ``depth_mask`` on ``Framebuffer``, not ``Context``.
    Assigning ``ctx.depth_mask`` silently creates an inert Python attribute, so
    the "depth write OFF" transparent pass still wrote depth — an additive
    surface such as the K1 Star Map dome then z-rejected every particle and
    additive mesh drawn behind it.
    """
    try:
        ctx.fbo.depth_mask = bool(enabled)
    except Exception:
        pass


def _skin_vbo_signature_for_node(node):
    """Return the skin state that must match a cached GPU VBO."""
    if node is None or not bool(getattr(node, "is_skin", False)):
        return None
    skin_data = getattr(node, "skin_data", None)
    vertices = getattr(node, "vertices", getattr(node, "verts", None))
    faces = getattr(node, "faces", None)
    def _safe_len(value):
        try:
            return len(value)
        except Exception:
            return 0
    return (
        tuple(getattr(node, "bone_map", []) or ()),
        _safe_len(skin_data),
        id(skin_data),
        _safe_len(vertices),
        id(vertices),
        _safe_len(faces),
        id(faces),
        bool(getattr(node, "_gr_bound_to_kotor_skeleton", False)),
        id(getattr(node, "_gr_skin_binding_report", None)),
    )


def _skin_influence_summary_for_log(node):
    bone_map = list(getattr(node, "bone_map", []) or [])
    skin_rows = list(getattr(node, "skin_data", []) or [])
    used: set[int] = set()
    max_influences = 0
    min_sum = None
    max_sum = None
    for row in skin_rows:
        influences = list(getattr(row, "influences", []) or [])
        max_influences = max(max_influences, len(influences))
        total = 0.0
        for influence in influences:
            try:
                index = int(getattr(influence, "bone_index", -1))
                weight = float(getattr(influence, "weight", 0.0))
            except Exception:
                continue
            if 0 <= index < len(bone_map) and weight > 0.0:
                used.add(index)
                total += weight
        min_sum = total if min_sum is None else min(min_sum, total)
        max_sum = total if max_sum is None else max(max_sum, total)
    return {
        "rows": len(skin_rows),
        "bone_map_count": len(bone_map),
        "bone_map_sample": [str(name or "") for name in bone_map[:12]],
        "used_influence_slot_count": len(used),
        "max_influences_per_vertex": max_influences,
        "weight_sum_min": float(min_sum or 0.0),
        "weight_sum_max": float(max_sum or 0.0),
    }


def _diffuse_is_reflectivity_mask(img) -> bool:
    """True when a diffuse image's alpha is a KOTOR environment/spec mask.

    KOTOR stores a reflectivity mask in the diffuse alpha of opaque skins: the
    engine blends the environment cube map by ``1 - alpha`` (bright metal where
    alpha is low, matte cloth where alpha is 1).  Packaged TPC assets get this
    driven by their TXI ``envmaptexture``; a raw TGA/PNG dropped into the
    viewport keeps the mask but names no cube map, so it would render as its
    (deliberately dark) diffuse with no sheen.  Detect that mask so the caller
    can fall back to the default environment map.

    A texture qualifies when a meaningful band of texels is partially
    transparent (the mask) *without* a hard cutout (which would indicate
    punch-through/foliage alpha) and it is not effectively fully opaque.  The
    verdict is memoized on the image.
    """
    if img is None:
        return False
    cached = getattr(img, "_gr_reflectivity_mask", None)
    if cached is not None:
        return bool(cached)
    result = False
    try:
        if "A" in img.getbands():
            sample = img
            if max(img.size) > 256:
                sample = img.resize((256, 256))
            hist = sample.getchannel("A").histogram()
            total = float(sum(hist)) or 1.0
            frac_cutout = sum(hist[0:8]) / total      # near-zero → transparency/cutout
            frac_partial = sum(hist[16:248]) / total  # the reflectivity band
            frac_opaque = sum(hist[248:256]) / total
            result = (
                frac_partial >= 0.05
                and frac_cutout < 0.02
                and frac_opaque < 0.995
            )
    except Exception:
        result = False
    try:
        img._gr_reflectivity_mask = result
    except Exception:
        pass
    return result


class GpuRenderer:
    """
    GPU renderer for KotOR models.

    Usage
    -----
    renderer = GpuRenderer()
    img = renderer.render(model, camera, W, H,
                           textures={name: pil_img}, anim_pose=None)

    The `camera` is a GhostRigger ArcBallCamera-compatible object with:
        camera.eye   → (x, y, z) camera world position  (ArcBallCamera: call camera.eye())
        camera.target → (x, y, z) look-at target          (ArcBallCamera: list attribute)
        camera.up    → (x, y, z) up vector (optional, default (0,0,1) = Z-up for KotOR)
        camera.fov   → vertical field-of-view in degrees (optional, default 45°)
        camera.near, camera.far  → clip distances (optional, default 0.01/2000)
    """

    #: Legacy diagnostic switch; the Qt viewport does not use CPU rendering.
    force_cpu: bool = False
    name: str = "ModernGL"
    backend_id: str = "modern_gl"

    def __init__(self):
        self._ctx: Optional['moderngl.Context'] = None
        self._prog: Optional['moderngl.Program'] = None
        self._grid_prog: Optional['moderngl.Program'] = None
        self._grid_vbo: Optional['moderngl.Buffer'] = None
        self._grid_vao: Optional['moderngl.VertexArray'] = None
        self._tex_cache: Optional[_GlTexCache] = None
        self._mesh_cache: Dict[int, _GpuMesh] = {}  # node id → _GpuMesh
        self._gpu_available: bool = False
        self._init_attempted: bool = False
        # Persistent FBO (reused across frames for same resolution)
        self._fbo: Optional['moderngl.Framebuffer'] = None
        self._fbo_w: int = 0
        self._fbo_h: int = 0
        # FIX-MSAA: Resolve FBO for multisampled MSAA readback
        self._fbo_resolve: Optional['moderngl.Framebuffer'] = None
        self._fbo_msaa: bool = False
        # PERF-FBO-SIMPLE: Non-MSAA FBO for fast interactive frames.
        # On llvmpipe, MSAA resolve (copy_framebuffer) costs ~59ms/frame.
        # During interactive drag we skip MSAA and draw to this simple FBO.
        self._fbo_simple: Optional['moderngl.Framebuffer'] = None
        self._fbo_simple_w: int = 0
        self._fbo_simple_h: int = 0
        # Placeholder 1×1 white texture (used as diffuse fallback)
        self._white_tex: Optional['moderngl.Texture'] = None
        self._procedural_lightsaber_textures: Dict[str, 'Image.Image'] = {}
        # FIX-ENVFB: Neutral grey 1×1 environment-map fallback texture.
        # When a node has txi_envmaptexture set but the texture isn't in the
        # texture dict (partial texture load), bind this grey instead of nothing.
        # A grey env map (0.5,0.5,0.5) produces a slight metallic tint which is
        # correct: the env blend weight (diffuse alpha) modulates towards grey
        # rather than towards zero, keeping the surface opaque.
        self._grey_env_tex: Optional['moderngl.Texture'] = None
        # FIX-ENVDEFAULT: Built-in metallic sphere-map used as the default
        # environment reflection for opaque skins whose diffuse alpha is a
        # reflectivity mask but that name no cube map (bare TGA/PNG drops).
        # Mirrors appearance.2da envmap=DEFAULT so metallic armour previews
        # with sheen instead of rendering as its dark diffuse.
        self._default_env_tex: Optional['moderngl.Texture'] = None
        # FIX-PERSCACHE: Persistent world-transform cache keyed by (model_id, node_id).
        # Survives across frames for static geometry; invalidated when the model changes.
        # This reduces per-frame cost from O(N×depth) parent-chain walks to O(1) lookups.
        self._wt_model_id: int = 0    # id() of the last model we built the cache for
        self._wt_cache: Dict[int, tuple] = {}  # node id() → (world_pos, world_orient)
        # PERF-UNIFCACHE: Cached uniform references (populated on first render).
        # Avoids prog['name'] dict lookup overhead per draw call.
        self._u: Dict[str, object] = {}   # uniform name → Uniform object
        self._submission_stats: dict[str, int] = {
            "uniform_writes": 0,
            "uniform_skips": 0,
            "blend_state_writes": 0,
            "blend_state_skips": 0,
        }
        self._blend_submission = _ExactBlendSubmission(self._submission_stats)
        # PERF-NODECACHE: Pre-classified node lists cached per model.
        # Avoids re-classifying every node every frame when the model hasn't changed.
        self._node_cache_model_id: int = 0
        self._node_cache_opaque: list = []
        self._node_cache_cutout: list = []
        self._node_cache_transparent: list = []
        self._node_cache_proxy_ids: set = set()
        self._node_cache_is_module: bool = False
        self._node_cache_signature: tuple = ()
        # PERF-NODECACHE-REVISION: Render-time classification must be O(1) on
        # an unchanged scene.  The previous implementation rebuilt a detailed
        # signature for every node on every frame, which cost about 54 ms/frame
        # on the retained 207TEL PIE scene before any draw calls were submitted.
        # UI material/visibility workflows already call invalidate_node_cache();
        # model replacement is covered by model identity, and attach/detach is
        # covered by the node count.  Headless/core callers can also bump the
        # explicit model/root ``_gr_classification_revision`` contract.
        self._node_classification_revision: int = 0
        self._node_cache_built_revision: int = -1
        self._node_cache_node_count: int = 0
        self._node_cache_model_revision: tuple[int, int] = (0, 0)
        # Scene-light attributes and transforms remain live every frame, but
        # node type is static for a retained scene revision.  Keep the small
        # candidate set so light selection does not scan every room/actor node.
        self._scene_light_candidate_key: tuple = ()
        self._scene_light_candidate_nodes: tuple = ()
        # PERF: Interactive mode skips MSAA for faster frame times.
        # Keep the default scale at full resolution; lowering it is available
        # for emergency performance mode, but makes animated previews pixelated.
        self.interactive: bool = False
        self.interactive_render_scale: float = 1.0
        self.show_solid: bool = True
        self.show_texture: bool = True
        self.show_diffuse_map: bool = True
        self.show_lightmap_map: bool = False
        self.show_environment_map: bool = True
        self.show_specular_map: bool = True
        self.show_normal_map: bool = True
        self.lightmap_intensity: float = 0.55
        self.lightmap_mode: str = "disabled"
        self.lighting_mode: str = "scene"
        self.shader_complexity_mode: str = "off"
        self.scene_ambient: float = 0.06
        self.show_light_gizmos: bool = True
        self.show_light_radius_volumes: bool = False
        self.show_wireframe: bool = False
        self.render_mode: str = "realistic"
        self.selected_node = None
        self.selected_nodes: list = []
        self.wire_color: tuple[float, float, float] = (0.18, 0.62, 0.95)
        self.viewport_background: tuple[float, float, float] = (23 / 255.0, 25 / 255.0, 28 / 255.0)
        self.grid_minor_color: tuple[float, float, float] = (58 / 255.0, 64 / 255.0, 72 / 255.0)
        self.grid_major_color: tuple[float, float, float] = (82 / 255.0, 90 / 255.0, 102 / 255.0)
        self.grid_x_axis_color: tuple[float, float, float] = (118 / 255.0, 54 / 255.0, 54 / 255.0)
        self.grid_y_axis_color: tuple[float, float, float] = (62 / 255.0, 112 / 255.0, 68 / 255.0)
        self._wireframe_pass: bool = False
        self.show_grid: bool = True
        self.cull_faces: bool = True
        self.enable_frustum_culling: bool = True
        self.max_new_mesh_uploads_per_frame: int = 64
        self.deferred_mesh_uploads: bool = False
        # ── Phase A: GPU Skinning state ──────────────────────────────────────────
        # MatrixPaletteUploader instances, scoped by scene object when possible.
        # The uploader is name-indexed internally, so duplicate scene characters
        # must not share one palette lookup table.
        self._skin_uploader: Optional['MatrixPaletteUploader'] = None
        self._skin_uploaders_by_scope: Dict[tuple, tuple[tuple[int, ...], object]] = {}
        # Actor-local serialized palette cache.  One entry is retained per
        # (actor scope, skin node) and replaced when that actor's pose changes.
        # This avoids recomputing/padding 128 matrices on camera-only frames
        # without ever sharing a character's palette with another character.
        self._skin_palette_bytes_cache: Dict[tuple, tuple] = {}
        self._skin_model_id: int = 0  # id() of model for which bind-pose was built
        self._skin_bone_count: int = 0  # number of bones in the current palette
        self._skin_logged: bool = False  # one-shot log for GPU skinning activation
        self._skin_preview_gate_logged: set[tuple[int, int]] = set()
        self._skin_preview_palette_logged: set[tuple[int, int]] = set()
        self._gl_state_trace_path: str = _gl_state_trace_path()
        self._lm_data_dump_path: str = _lm_data_dump_path()
        self._lm_data_dump_seen: set = set()
        self._skin_dump_path: str = _skin_dump_path()
        self._skin_dump_seen: set = set()
        # ── Emitter particle simulation state ────────────────────────────────
        # One ModelParticleSystems per rendered model (single-entry cache: the
        # viewport renders one model/scene at a time).  Advanced with wall-clock
        # time each frame; the viewport polls ``particles_active`` to keep
        # scheduling live frames while emitters are visible.
        self.show_particles: bool = True
        self.particles_active: bool = False
        self._particle_pass = None
        self._particle_systems = None
        self._particle_model_id: int = 0
        self._particle_last_wall: float = 0.0
        self._particle_anim_cache: tuple = ("", None)
        # ── Bloom post-process ───────────────────────────────────────────────
        # Subtle glow accent for genuinely bright content (star cores and saber
        # blades).  Retail KOTOR's emitter glow primarily comes from additive
        # texture falloff, so the luminance threshold excludes saturated cyan
        # structure and the strength stays an accent rather than a wash.
        self.bloom_enabled: bool = True
        # Luminance-gated extraction in ModernGLBloomPass blooms only genuinely
        # bright cores.  Keep the residual halo restrained because KotOR.js
        # Forge renders these emitter sprites with no post-process glow at all.
        self.bloom_threshold: float = 0.82
        self.bloom_strength: float = 0.18
        self._bloom_pass = None
        self._fbo_resolve_tex = None
        self._fbo_simple_tex = None
        # Performance counters
        self.perf: Dict[str, float] = {
            'last_frame_ms': 0.0,
            'gpu_upload_ms': 0.0,
            'draw_ms': 0.0,
            'readback_ms': 0.0,
            'tri_count': 0,
            'draw_calls': 0,
            'visible_meshes': 0,
            'culled_meshes': 0,
            'culled_actor_meshes': 0,
            'uniform_writes': 0,
            'uniform_skips': 0,
            'blend_state_writes': 0,
            'blend_state_skips': 0,
            'backend': 'none',
        }

    def _reset_uniform_submission_cache(self) -> None:
        for uniform in self._u.values():
            reset = getattr(uniform, "reset_submission_cache", None)
            if reset is not None:
                reset()

    def _begin_submission_frame(self) -> None:
        for key in self._submission_stats:
            self._submission_stats[key] = 0
        self._reset_uniform_submission_cache()
        self._blend_submission.reset()

    def _begin_draw_pass(self, ctx, *, blend_enabled: bool | None) -> None:
        """Reset exact submission knowledge at an explicit render-pass edge."""
        self._reset_uniform_submission_cache()
        self._blend_submission.reset()
        if blend_enabled is not None:
            self._blend_submission.apply(ctx, enabled=blend_enabled)

    def set_theme_colors(self, theme) -> None:
        self.viewport_background = _hex_to_rgb_float(theme.color("viewport.background"), self.viewport_background)
        self.grid_minor_color = _hex_to_rgb_float(theme.color("viewport.gridMinor"), self.grid_minor_color)
        self.grid_major_color = _hex_to_rgb_float(theme.color("viewport.gridMajor"), self.grid_major_color)
        self.grid_x_axis_color = _hex_to_rgb_float(theme.color("error"), self.grid_x_axis_color)
        self.grid_y_axis_color = _hex_to_rgb_float(theme.color("success"), self.grid_y_axis_color)
        self.wire_color = _hex_to_rgb_float(theme.color("accent.primary"), self.wire_color)
        if self._grid_vao is not None or self._grid_vbo is not None:
            try:
                if self._grid_vao is not None:
                    self._grid_vao.release()
            except Exception:
                pass
            try:
                if self._grid_vbo is not None:
                    self._grid_vbo.release()
            except Exception:
                pass
            self._grid_vao = None
            self._grid_vbo = None

    def reset_theme_colors(self) -> None:
        self.wire_color = (0.18, 0.62, 0.95)
        self.viewport_background = (23 / 255.0, 25 / 255.0, 28 / 255.0)
        self.grid_minor_color = (58 / 255.0, 64 / 255.0, 72 / 255.0)
        self.grid_major_color = (82 / 255.0, 90 / 255.0, 102 / 255.0)
        self.grid_x_axis_color = (118 / 255.0, 54 / 255.0, 54 / 255.0)
        self.grid_y_axis_color = (62 / 255.0, 112 / 255.0, 68 / 255.0)
        if self._grid_vao is not None or self._grid_vbo is not None:
            try:
                if self._grid_vao is not None:
                    self._grid_vao.release()
            except Exception:
                pass
            try:
                if self._grid_vbo is not None:
                    self._grid_vbo.release()
            except Exception:
                pass
            self._grid_vao = None
            self._grid_vbo = None

    @staticmethod
    def _rgb_float(color: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(max(0.0, min(1.0, float(v) / 255.0)) for v in color[:3])

    @staticmethod
    def _blend_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, float(t)))
        return tuple(int(round(float(a[i]) * (1.0 - t) + float(b[i]) * t)) for i in range(3))

    @staticmethod
    def _relative_luma(color: tuple[int, int, int]) -> float:
        r, g, b = (max(0, min(255, int(v))) / 255.0 for v in color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def set_native_palette_colors(
        self,
        *,
        base: tuple[int, int, int],
        text: tuple[int, int, int],
        highlight: tuple[int, int, int],
    ) -> None:
        bg = tuple(int(v) for v in base[:3])
        fg = tuple(int(v) for v in text[:3])
        hi = tuple(int(v) for v in highlight[:3])
        is_dark = self._relative_luma(bg) < 0.45
        self.viewport_background = self._rgb_float(bg)
        self.grid_minor_color = self._rgb_float(self._blend_rgb(bg, fg, 0.12 if is_dark else 0.18))
        self.grid_major_color = self._rgb_float(self._blend_rgb(bg, fg, 0.22 if is_dark else 0.30))
        self.grid_x_axis_color = self._rgb_float((210, 70, 70) if is_dark else (160, 30, 30))
        self.grid_y_axis_color = self._rgb_float((70, 180, 90) if is_dark else (40, 130, 55))
        self.wire_color = self._rgb_float(hi)
        if self._grid_vao is not None or self._grid_vbo is not None:
            try:
                if self._grid_vao is not None:
                    self._grid_vao.release()
            except Exception:
                pass
            try:
                if self._grid_vbo is not None:
                    self._grid_vbo.release()
            except Exception:
                pass
            self._grid_vao = None
            self._grid_vbo = None

    # ── Context management ────────────────────────────────────────────────────

    def _ensure_context(self) -> bool:
        """Try to initialize the ModernGL EGL context once."""
        if self._init_attempted:
            return self._gpu_available
        self._init_attempted = True
        if not _MODERNGL or not _NUMPY or self.force_cpu:
            return False
        try:
            self._ctx, _ctx_backend = _create_moderngl_standalone_context()
            self._prog = self._ctx.program(
                vertex_shader=_VERT_SRC,
                fragment_shader=_FRAG_SRC,
            )
            self._grid_prog = self._ctx.program(
                vertex_shader=_GRID_VERT_SRC,
                fragment_shader=_GRID_FRAG_SRC,
            )
            self._tex_cache = _GlTexCache(self._ctx)
            # PERF-UNIFCACHE: Pre-cache all uniform references to avoid
            # prog['name'] dict lookups (~0.4ms/frame savings on 56-node models).
            _p = self._prog
            _u = {}
            for _uname in (
                'u_mvp', 'u_model', 'u_normal_mat', 'u_cam_pos',
                'u_map_fog_enabled', 'u_map_fog_color', 'u_map_fog_near', 'u_map_fog_far',
                'u_light_dir', 'u_light_dir2', 'u_ambient', 'u_specular',
                'u_shininess', 'u_alpha_test', 'u_decal', 'u_wateralpha',
                'u_has_spec', 'u_has_bump', 'u_alpha', 'u_node_alpha', 'u_blend_mode',
                'u_has_tex', 'u_has_lm', 'u_lm_shade', 'u_lightmap_intensity',
                'u_lightmap_mode', 'u_has_env',
                'u_lm_composite_mode',
                'u_scene_lighting', 'u_scene_ambient', 'u_scene_light_count',
                'u_scene_light_enabled', 'u_scene_light_kind',
                'u_scene_light_ambient_only', 'u_scene_light_pos',
                'u_scene_light_dir', 'u_scene_light_color',
                'u_scene_light_radius', 'u_scene_light_intensity',
                'u_scene_light_cone_cos', 'u_scene_light_area_size',
                'u_diffuse', 'u_selfillum', 'u_uv_scroll', 'u_rotate_tex',
                'u_uv_v_flip', 'u_flipbook_off', 'u_flipbook_size', 'u_features',
                'u_water_time', 'u_proc_type', 'u_dangly_enabled',
                'u_dangly_displacement', 'u_dangly_time',
                'u_saber_enabled', 'u_saber_displacement', 'u_saber_length',
                'u_oit_enabled', 'u_tex', 'u_lm_tex', 'u_env_tex', 'u_spec_tex', 'u_bump_tex',
                'u_debug_visualize', 'u_wireframe_enabled', 'u_wire_color',
                'u_render_mode', 'u_selected', 'u_sprite_alpha_source', 'u_sprite_glow',
                # Phase A: GPU Skinning uniforms
                'u_skin_enabled', 'u_bone_count', 'u_bones',
            ):
                try:
                    _u[_uname] = _ExactUniformSubmission(
                        _p[_uname],
                        self._submission_stats,
                        cache_writes=_uname != "u_bones",
                    )
                except KeyError:
                    pass
            self._u = _u
            self._gpu_available = True
            log.info("%s: Context GL %s initialized (%s)",
                     self.__class__.__name__, self._ctx.version_code, _ctx_backend)
            try:
                info = getattr(self._ctx, "info", {}) or {}
                log.info(
                    "RendererDiagnostics: GPU=%s, API=OpenGL, Backend=%s, FeatureLevel=GL%s",
                    info.get("GL_RENDERER", "unknown"),
                    _ctx_backend,
                    self._ctx.version_code,
                )
            except Exception:
                pass
            return True
        except Exception as e:
            log.info(f"GpuRenderer: GPU init failed ({e}) - viewport GPU rendering unavailable")
            self._gpu_available = False
            return False

    def release(self):
        """Release all GPU resources."""
        if self._tex_cache:
            self._tex_cache.clear()
        for m in self._mesh_cache.values():
            m.release()
        self._mesh_cache.clear()
        self._skin_palette_bytes_cache.clear()
        if self._particle_pass is not None:
            try:
                self._particle_pass.release()
            except Exception:
                pass
            self._particle_pass = None
        self._particle_systems = None
        self._particle_model_id = 0
        self._particle_anim_cache = ("", None)
        if self._bloom_pass is not None:
            try:
                self._bloom_pass.release()
            except Exception:
                pass
            self._bloom_pass = None
        self._fbo_resolve_tex = None
        self._fbo_simple_tex = None
        # Release persistent FBO
        if self._fbo is not None:
            try:
                self._fbo.color_attachments[0].release()
                self._fbo.depth_attachment.release()
                self._fbo.release()
            except Exception:
                pass
            self._fbo = None
        # Release MSAA resolve FBO
        if self._fbo_resolve is not None:
            try:
                self._fbo_resolve.color_attachments[0].release()
                self._fbo_resolve.release()
            except Exception:
                pass
            self._fbo_resolve = None
        # Release simple (non-MSAA) FBO
        if self._fbo_simple is not None:
            try:
                self._fbo_simple.color_attachments[0].release()
                self._fbo_simple.depth_attachment.release()
                self._fbo_simple.release()
            except Exception:
                pass
            self._fbo_simple = None
        # Release white placeholder texture
        if self._white_tex is not None:
            try: self._white_tex.release()
            except Exception: pass
            self._white_tex = None
        # Release grey env-map fallback texture
        if self._grey_env_tex is not None:
            try: self._grey_env_tex.release()
            except Exception: pass
            self._grey_env_tex = None
        # Release default metallic env sphere-map
        if self._default_env_tex is not None:
            try: self._default_env_tex.release()
            except Exception: pass
            self._default_env_tex = None
        # Phase A: Release GPU skinning uploader
        if self._skin_uploader is not None:
            try: self._skin_uploader.release()
            except Exception: pass
            self._skin_uploader = None
            self._skin_model_id = 0
            self._skin_bone_count = 0
        if self._prog:
            try: self._prog.release()
            except Exception: pass
            self._prog = None
        if self._grid_vao:
            try: self._grid_vao.release()
            except Exception: pass
            self._grid_vao = None
        if self._grid_vbo:
            try: self._grid_vbo.release()
            except Exception: pass
            self._grid_vbo = None
        if self._grid_prog:
            try: self._grid_prog.release()
            except Exception: pass
            self._grid_prog = None
        if self._ctx:
            try: self._ctx.release()
            except Exception: pass
            self._ctx = None
        self._u.clear()
        self._blend_submission.reset()
        self._gpu_available = False
        self._init_attempted = False  # allow re-initialisation after release

    def _default_env_texture(self, ctx):
        """Lazily build the default metallic sphere-map (matcap) reflection.

        A vertical grey gradient — bright near the top with a soft highlight
        band, darkening toward the bottom — reads as a neutral chrome/sky
        reflection through the shader's ``env_uv = R.xy/m + 0.5`` sphere map.
        Blended by ``1 - diffuse_alpha`` it lifts a dark metallic diffuse into a
        metallic sheen without tinting matte (alpha=1) regions.
        """
        if self._default_env_tex is not None:
            return self._default_env_tex
        size = 128
        buf = bytearray(size * size * 4)
        for y in range(size):
            v = y / (size - 1)                       # 0 = top row, 1 = bottom
            base = 1.0 - v                           # bright top → dark bottom
            highlight = math.exp(-((v - 0.25) ** 2) / (2 * 0.06 ** 2)) * 0.45
            lum = 0.22 + 0.66 * base + highlight
            c = int(max(0.0, min(1.0, lum)) * 255)
            row = y * size * 4
            for x in range(size):
                i = row + x * 4
                buf[i] = c
                buf[i + 1] = c
                buf[i + 2] = c
                buf[i + 3] = 255
        tex = ctx.texture((size, size), 4, bytes(buf))
        try:
            tex.build_mipmaps()
        except Exception:
            pass
        tex.repeat_x = False
        tex.repeat_y = False
        self._default_env_tex = tex
        return tex

    def clear_caches(self) -> None:
        """Clear per-model GPU mesh and texture caches without destroying the context.

        Call this between model renders in a batch loop to free GPU VRAM while
        keeping the EGL context alive for the next model.  This prevents OOM
        kills when rendering thousands of models consecutively.

        FIX-PERF-CLEAR: Do NOT reset _init_attempted here.  The original code
        incorrectly set _init_attempted=False, which forced a full ModernGL EGL
        context re-initialization on every subsequent render() call.  Re-init
        costs ~70 ms per model × 3,304 models = ~4 minutes of unnecessary GPU
        setup in a batch render.  The GPU context (self._ctx, self._prog) is
        still valid after clearing caches — only mesh VAOs/VBOs and texture
        uploads are freed, not the context itself.
        """
        # Release all cached mesh VAO/VBO/IBOs
        for m in self._mesh_cache.values():
            m.release()
        self._mesh_cache.clear()
        # Release all cached GL textures to free VRAM
        if self._tex_cache:
            self._tex_cache.clear()
        # Clear persistent world-transform cache (invalidated per model anyway)
        self._wt_cache.clear()
        self._wt_model_id = 0
        self._skin_palette_bytes_cache.clear()
        # PERF-NODECACHE: Clear pre-classified node lists
        self._node_cache_model_id = 0
        self._node_cache_opaque = []
        self._node_cache_cutout = []
        self._node_cache_transparent = []
        self._node_cache_proxy_ids = set()
        self._scene_light_candidate_key = ()
        self._scene_light_candidate_nodes = ()
        # Phase A: Clear GPU skinning state (new model = new bone palette)
        if self._skin_uploader is not None:
            try: self._skin_uploader.release()
            except Exception: pass
            self._skin_uploader = None
        self._skin_model_id = 0
        self._skin_bone_count = 0
        # Drop live particle simulations (new model = new emitter set)
        self._particle_systems = None
        self._particle_model_id = 0
        self._particle_anim_cache = ("", None)
        # NOTE: _init_attempted is intentionally NOT reset here — the EGL context
        # remains alive and valid across clear_caches() calls.

    # ── Main render entry ─────────────────────────────────────────────────────

    def reset_framebuffers(self) -> None:
        """Release cached offscreen render targets while keeping the GL context alive."""

        def _release_fbo(fbo) -> None:
            if fbo is None:
                return
            try:
                for attachment in getattr(fbo, "color_attachments", []) or []:
                    if attachment is not None:
                        attachment.release()
                depth = getattr(fbo, "depth_attachment", None)
                if depth is not None:
                    depth.release()
                fbo.release()
            except Exception:
                pass

        _release_fbo(self._fbo)
        _release_fbo(self._fbo_resolve)
        _release_fbo(self._fbo_simple)
        self._fbo = None
        self._fbo_resolve = None
        self._fbo_simple = None
        self._fbo_resolve_tex = None
        self._fbo_simple_tex = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._fbo_simple_w = 0
        self._fbo_simple_h = 0
        self._fbo_msaa = False
        if self._bloom_pass is not None:
            try:
                self._bloom_pass.release()
            except Exception:
                pass
            self._bloom_pass = None

    def _reset_frame_state(self, ctx, width: int, height: int) -> None:
        """Reset mutable GL state that can leak between scene/grid/overlay passes."""
        try:
            ctx.viewport = (0, 0, int(width), int(height))
        except Exception:
            pass
        try:
            ctx.scissor = None
        except Exception:
            pass
        try:
            ctx.disable(moderngl.BLEND)
            ctx.disable(moderngl.SCISSOR_TEST)
            ctx.disable(moderngl.POLYGON_OFFSET_FILL)
            ctx.enable(moderngl.DEPTH_TEST)
        except Exception:
            pass
        try:
            ctx.depth_func = '<='
            _set_depth_write(ctx, True)
        except Exception:
            pass
        try:
            ctx.front_face = 'ccw'
        except Exception:
            pass
        try:
            ctx.wireframe = False
        except Exception:
            pass

    def _debug_log_gl_error(self, label: str) -> None:
        raw = os.environ.get(_GL_DEBUG_ERRORS_ENV, "").strip().lower()
        if raw not in _GL_STATE_TRACE_TRUE:
            return
        ctx = self._ctx
        if ctx is None:
            return
        try:
            err = getattr(ctx, "error", None)
            if err:
                log.debug("ModernGLRenderer: GL error after %s: %s", label, err)
        except Exception:
            pass

    def _ensure_grid_vao(self):
        if not (_NUMPY and self._ctx is not None and self._grid_prog is not None):
            return None
        if self._grid_vao is not None:
            return self._grid_vao

        extent = 60
        major_every = 5
        minor = self.grid_minor_color
        major = self.grid_major_color
        x_axis = self.grid_x_axis_color
        y_axis = self.grid_y_axis_color
        rows = []

        for i in range(-extent, extent + 1):
            color = x_axis if i == 0 else (major if i % major_every == 0 else minor)
            rows.append((-extent, i, 0.0, *color))
            rows.append((extent, i, 0.0, *color))

        for i in range(-extent, extent + 1):
            color = y_axis if i == 0 else (major if i % major_every == 0 else minor)
            rows.append((i, -extent, 0.0, *color))
            rows.append((i, extent, 0.0, *color))

        data = np.asarray(rows, dtype=np.float32)
        self._grid_vbo = self._ctx.buffer(data.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._grid_prog,
            [(self._grid_vbo, '3f 3f', 'in_pos', 'in_color')],
        )
        return self._grid_vao

    def _draw_grid(self, ctx, mvp) -> None:
        if not self.show_grid:
            return
        vao = self._ensure_grid_vao()
        if vao is None or self._grid_prog is None:
            return
        self._grid_prog['u_mvp'].write(_mat4_tobytes(mvp))
        ctx.disable(moderngl.BLEND)
        _set_depth_write(ctx, False)
        try:
            ctx.line_width = 1.0
        except Exception:
            pass
        try:
            vao.render(moderngl.LINES)
        finally:
            _set_depth_write(ctx, True)

    def _is_node_selected_for_render(self, node) -> bool:
        if node is getattr(self, "selected_node", None) or bool(getattr(node, "_gr_selected", False)):
            return True
        selected_ids = {id(_n) for _n in (getattr(self, "selected_nodes", []) or [])}
        return id(node) in selected_ids

    def _draw_light_gizmos(self, ctx, mvp, nodes, get_world_transform) -> None:
        if not bool(getattr(self, "show_light_gizmos", True)):
            return
        if not (_NUMPY and self._grid_prog is not None and nodes):
            return

        def _add_line(rows, a, b, color):
            rows.append((float(a[0]), float(a[1]), float(a[2]), *color))
            rows.append((float(b[0]), float(b[1]), float(b[2]), *color))

        def _v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        def _v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        def _v_mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
        def _v_cross(a, b):
            return (a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0])
        def _v_norm(a):
            ln = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1.0
            return (a[0] / ln, a[1] / ln, a[2] / ln)

        def _basis(direction):
            forward = _v_norm(direction)
            seed = (0.0, 0.0, 1.0) if abs(forward[2]) < 0.92 else (0.0, 1.0, 0.0)
            right = _v_norm(_v_cross(seed, forward))
            up = _v_norm(_v_cross(forward, right))
            return forward, right, up

        def _ring(rows, center, axis_a, axis_b, radius, color, steps=32):
            prev = None
            for i in range(steps + 1):
                t = (i / steps) * math.tau
                pt = _v_add(center, _v_add(_v_mul(axis_a, math.cos(t) * radius),
                                           _v_mul(axis_b, math.sin(t) * radius)))
                if prev is not None:
                    _add_line(rows, prev, pt, color)
                prev = pt

        rows = []
        selected = getattr(self, "selected_node", None)
        draw_volumes = bool(getattr(self, "show_light_radius_volumes", True))
        for node in nodes:
            if not bool(getattr(node, "is_light", False)):
                continue
            if bool(getattr(node, "_gr_light_helper_hidden", False)):
                continue
            if bool(getattr(node, "_gr_light_hidden", False)) or bool(getattr(node, "_gr_light_deleted", False)):
                continue
            try:
                pos, orient = get_world_transform(node)
            except Exception:
                pos = getattr(node, "position", (0.0, 0.0, 0.0))
                orient = getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0))
            kind = str(getattr(node, "light_kind", "point") or "point").lower()
            helper_kind = kind.replace("aurora_", "")
            enabled = bool(getattr(node, "light_enabled", True))
            cone = max(1.0, min(179.0, float(getattr(node, "light_cone_degrees", 45.0) or 45.0)))
            direction = self._rotate_vec_by_quat((0.0, 0.0, -1.0), orient)
            forward, right, up = _basis(direction)
            base = LIGHT_HELPER_COLORS.get(helper_kind, LIGHT_HELPER_COLORS["point"])
            if not enabled:
                base = tuple(c * 0.38 for c in base)
            if bool(getattr(node, "_gr_light_selected", False)):
                base = tuple(min(1.0, c * LIGHT_HELPER_SELECTED_BOOST) for c in base)
            if self._is_node_selected_for_render(node) or bool(getattr(node, "_gr_light_metadata", {}).get("active_selection", False)):
                base = (0.90, 0.95, 1.0)

            _ring(rows, pos, right, up, LIGHT_HELPER_MARKER_RADIUS, base, steps=16)
            if helper_kind == "ambient":
                continue
            if helper_kind == "area":
                half = LIGHT_HELPER_AREA_SIZE * 0.5
                c0 = _v_add(pos, _v_add(_v_mul(right, -half), _v_mul(up, -half)))
                c1 = _v_add(pos, _v_add(_v_mul(right, half), _v_mul(up, -half)))
                c2 = _v_add(pos, _v_add(_v_mul(right, half), _v_mul(up, half)))
                c3 = _v_add(pos, _v_add(_v_mul(right, -half), _v_mul(up, half)))
                for a, b in ((c0, c1), (c1, c2), (c2, c3), (c3, c0)):
                    _add_line(rows, a, b, base)
                _add_line(rows, pos, _v_add(pos, _v_mul(forward, LIGHT_HELPER_DIRECTION_LENGTH * 0.6)), base)
            elif helper_kind == "directional":
                target = _v_add(pos, _v_mul(forward, LIGHT_HELPER_DIRECTION_LENGTH))
                _add_line(rows, pos, target, base)
                head = LIGHT_HELPER_DIRECTION_LENGTH * 0.22
                _add_line(rows, target, _v_add(_v_sub(target, _v_mul(forward, head)), _v_mul(right, head * 0.45)), base)
                _add_line(rows, target, _v_add(_v_sub(target, _v_mul(forward, head)), _v_mul(right, -head * 0.45)), base)
                _add_line(rows, _v_add(target, _v_mul(right, -head * 0.35)), _v_add(target, _v_mul(right, head * 0.35)), base)
                _add_line(rows, _v_add(target, _v_mul(up, -head * 0.35)), _v_add(target, _v_mul(up, head * 0.35)), base)
            elif helper_kind == "spot":
                if not draw_volumes:
                    _add_line(rows, pos, _v_add(pos, _v_mul(forward, LIGHT_HELPER_SPOT_LENGTH)), base)
                    continue
                length = LIGHT_HELPER_SPOT_LENGTH
                cap_radius = min(
                    math.tan(math.radians(cone * 0.5)) * length,
                    LIGHT_HELPER_SPOT_CAP_MAX_RADIUS,
                )
                cap = _v_add(pos, _v_mul(forward, length))
                _ring(rows, cap, right, up, cap_radius, base, steps=20)
                for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    edge = _v_add(cap, _v_add(_v_mul(right, cap_radius * sx), _v_mul(up, cap_radius * sy)))
                    _add_line(rows, pos, edge, base)
            else:
                if not draw_volumes:
                    continue
                display_radius = LIGHT_HELPER_POINT_RADIUS
                if self._is_node_selected_for_render(node) or bool(getattr(node, "_gr_light_selected", False)):
                    display_radius = LIGHT_HELPER_POINT_RADIUS * 1.2
                _ring(rows, pos, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), display_radius, base, steps=28)
                _ring(rows, pos, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), display_radius, base, steps=28)
                _ring(rows, pos, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), display_radius, base, steps=28)

        if not rows:
            return
        data = np.asarray(rows, dtype=np.float32)
        vbo = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(self._grid_prog, [(vbo, '3f 3f', 'in_pos', 'in_color')])
        self._grid_prog['u_mvp'].write(_mat4_tobytes(mvp))
        depth_was_enabled = True
        try:
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            _set_depth_write(ctx, False)
            try:
                ctx.line_width = 1.0
            except Exception:
                pass
            vao.render(moderngl.LINES)
        finally:
            _set_depth_write(ctx, True)
            if depth_was_enabled:
                ctx.enable(moderngl.DEPTH_TEST)
            vao.release()
            vbo.release()

    @staticmethod
    def _light_kind_int(node) -> int:
        kind = str(getattr(node, "light_kind", "point") or "point").strip().lower()
        kind = kind.replace("aurora_", "")
        if kind == "directional":
            return 1
        if kind == "spot":
            return 2
        if kind == "area":
            return 3
        if kind == "ambient":
            return 4
        return 0

    @staticmethod
    def _rotate_vec_by_quat(v, q) -> tuple[float, float, float]:
        try:
            x, y, z = float(v[0]), float(v[1]), float(v[2])
            qx, qy, qz, qw = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            tx = 2.0 * (qy * z - qz * y)
            ty = 2.0 * (qz * x - qx * z)
            tz = 2.0 * (qx * y - qy * x)
            rx = x + qw * tx + (qy * tz - qz * ty)
            ry = y + qw * ty + (qz * tx - qx * tz)
            rz = z + qw * tz + (qx * ty - qy * tx)
            ln = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
            return (rx / ln, ry / ln, rz / ln)
        except Exception:
            return (0.0, 0.0, -1.0)

    def _scene_light_candidates_for_model(self, model, nodes) -> tuple:
        """Return light-typed nodes for the retained scene topology revision.

        Enabled/hidden state, color, radius, intensity, and world transform are
        deliberately not cached: ``_scene_light_records`` reads those values
        from each candidate every frame.  Only the immutable node-type filter is
        retained.  Topology/material workflows invalidate the same explicit
        classification revision used by draw-list caching.
        """
        root = getattr(model, "root_node", None)
        key = (
            id(model),
            len(nodes),
            int(self._node_classification_revision),
            int(getattr(model, "_gr_classification_revision", 0) or 0),
            int(getattr(root, "_gr_classification_revision", 0) or 0),
            int(getattr(model, "_gr_lighting_revision", 0) or 0),
            int(getattr(root, "_gr_lighting_revision", 0) or 0),
        )
        if key != self._scene_light_candidate_key:
            self._scene_light_candidate_nodes = tuple(
                node for node in nodes
                if bool(getattr(node, "is_light", False))
            )
            self._scene_light_candidate_key = key
        return self._scene_light_candidate_nodes

    def _scene_light_records(self, nodes, get_world_transform, *, reference_position=None) -> list[dict]:
        records: list[dict] = []
        for node in nodes:
            if not bool(getattr(node, "is_light", False)):
                continue
            if not bool(getattr(node, "light_enabled", True)):
                continue
            if bool(getattr(node, "_gr_light_hidden", False)) or bool(getattr(node, "_gr_light_deleted", False)):
                continue
            try:
                world_pos, world_orient = get_world_transform(node)
            except Exception:
                world_pos = getattr(node, "position", (0.0, 0.0, 0.0))
                world_orient = getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0))
            color = getattr(node, "light_color", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
            color_rgb = tuple(max(0.0, float(x)) for x in color[:3])
            radius = max(0.001, float(getattr(node, "light_radius", 5.0) or 5.0))
            raw_intensity = getattr(node, "light_multiplier", 1.0)
            intensity = max(0.0, float(1.0 if raw_intensity is None else raw_intensity))
            # Some creature models carry a disabled Aurora light whose authored
            # color is black (for example K2's T3-M4 and war droid).  Uploading
            # that zero-energy record switches the whole viewport from studio
            # lighting to scene lighting without contributing any illumination.
            if intensity <= 1.0e-6 or max(color_rgb, default=0.0) <= 1.0e-6:
                continue
            cone_deg = max(1.0, min(179.0, float(getattr(node, "light_cone_degrees", 45.0) or 45.0)))
            light_kind = str(getattr(node, "light_kind", "point") or "point").strip().lower()
            record = {
                "enabled": 1,
                "kind": self._light_kind_int(node),
                "ambient_only": 1 if bool(getattr(node, "light_ambient_only", False)) else 0,
                "pos": tuple(float(x) for x in world_pos[:3]),
                "dir": self._rotate_vec_by_quat((0.0, 0.0, -1.0), world_orient),
                "color": color_rgb,
                "radius": radius,
                "intensity": intensity,
                "cone_cos": math.cos(math.radians(cone_deg * 0.5)),
                "area_size": max(0.0, float(getattr(node, "light_area_size", 1.0) or 1.0)),
                "score": radius * max(0.01, intensity),
            }
            record["selection_key"] = scene_light_relevance_key(
                position=record["pos"],
                radius=radius,
                intensity=intensity,
                reference_position=reference_position,
                light_type=light_kind,
                color_rgb=record["color"],
            )
            records.append(record)
        records.sort(key=lambda item: item["selection_key"], reverse=True)
        return records[:16]

    def _upload_scene_lights(self, prog, uniforms, records: list[dict]) -> None:
        count = len(records)
        mode = str(getattr(self, "lighting_mode", "scene") or "scene").lower()
        if mode == "lightmap_preview":
            # Preserve the baked-lightmap branch.  Value 0 is the true unlit
            # contract and the fragment shader intentionally replaces every
            # prior lighting result with diffuse color for that value.
            lighting_int = 3
        elif mode in {"unlit", "fullbright", "diffuse_only", "normal_only", "specular_only", "environment_only", "shader_complexity"}:
            lighting_int = 0
        elif mode in {"scene", "photoreal_preview"} and count > 0:
            lighting_int = 2
        else:
            lighting_int = 1
        if "u_scene_lighting" in uniforms:
            uniforms["u_scene_lighting"].value = lighting_int
        if "u_scene_ambient" in uniforms:
            uniforms["u_scene_ambient"].value = float(getattr(self, "scene_ambient", 0.06))
        if "u_scene_light_count" in uniforms:
            uniforms["u_scene_light_count"].value = count
        if count <= 0 or not _NUMPY:
            return

        enabled = np.zeros(16, dtype="i4")
        kind = np.zeros(16, dtype="i4")
        ambient_only = np.zeros(16, dtype="i4")
        pos = np.zeros((16, 3), dtype="f4")
        direction = np.zeros((16, 3), dtype="f4")
        color = np.ones((16, 3), dtype="f4")
        radius = np.ones(16, dtype="f4")
        intensity = np.ones(16, dtype="f4")
        cone_cos = np.full(16, math.cos(math.radians(22.5)), dtype="f4")
        area_size = np.ones(16, dtype="f4")
        for idx, rec in enumerate(records):
            enabled[idx] = int(rec["enabled"])
            kind[idx] = int(rec["kind"])
            ambient_only[idx] = int(rec["ambient_only"])
            pos[idx] = rec["pos"]
            direction[idx] = rec["dir"]
            color[idx] = rec["color"]
            radius[idx] = float(rec["radius"])
            intensity[idx] = float(rec["intensity"])
            cone_cos[idx] = float(rec["cone_cos"])
            area_size[idx] = float(rec["area_size"])

        for name, data in (
            ("u_scene_light_enabled", enabled),
            ("u_scene_light_kind", kind),
            ("u_scene_light_ambient_only", ambient_only),
            ("u_scene_light_pos", pos),
            ("u_scene_light_dir", direction),
            ("u_scene_light_color", color),
            ("u_scene_light_radius", radius),
            ("u_scene_light_intensity", intensity),
            ("u_scene_light_cone_cos", cone_cos),
            ("u_scene_light_area_size", area_size),
        ):
            uniform = uniforms.get(name)
            if uniform is None:
                try:
                    uniform = prog[f"{name}[0]"]
                except Exception:
                    continue
            try:
                uniform.write(data.tobytes())
            except Exception:
                log.debug("GpuRenderer: failed to upload %s", name, exc_info=True)

    @staticmethod
    def _skin_pose_cache_stamp(pose) -> tuple:
        """Return the explicit revision contract for one actor-local pose."""
        if pose is None:
            return (0, 0.0, 0.0, 0.0, 0, "", 0, 0)

        def _float_stamp(name: str) -> float:
            try:
                return round(float(getattr(pose, name, 0.0) or 0.0), 6)
            except Exception:
                return 0.0

        nodes = getattr(pose, "nodes", {}) or {}
        return (
            id(pose),
            _float_stamp("time"),
            _float_stamp("current_time"),
            _float_stamp("_gr_animation_time"),
            int(getattr(pose, "_gr_revision", 0) or 0),
            str(getattr(pose, "_gr_animation_name", "") or ""),
            id(nodes),
            len(nodes),
        )

    def _skin_palette_bytes_for_draw(
        self,
        *,
        scope_key: tuple,
        skin_node,
        uploader,
        anim_pose,
        anim_base_pose,
        skin_signature: tuple | None,
    ) -> tuple[int, bytes, bool]:
        """Return actor-local palette bytes, reusing only the exact same pose."""
        bone_count = min(len(getattr(skin_node, "bone_map", []) or []), _SKIN_MAX_BONES)
        cache_key = (tuple(scope_key), id(skin_node))
        stamp = (
            id(uploader),
            self._skin_pose_cache_stamp(anim_pose),
            self._skin_pose_cache_stamp(anim_base_pose),
            skin_signature,
            bone_count,
        )
        cached = self._skin_palette_bytes_cache.get(cache_key)
        if cached is not None and cached[0] == stamp:
            _stamp, cached_count, cached_bytes, formula, inverse_bind_source = cached
            # Preserve diagnostic parity for callers which inspect the uploader
            # after a cached draw rather than only consuming the uniform bytes.
            uploader._skin_palette_formula = formula
            uploader._skin_inverse_bind_source = inverse_bind_source
            return int(cached_count), cached_bytes, True

        uploader.compute_skin_node_palette(
            skin_node,
            anim_pose,
            anim_base_pose=anim_base_pose,
        )
        if bool(getattr(skin_node, "_gr_bas_attachment_layer", False)):
            palette = bas_attachment_root_local_skin_palette(
                skin_node,
                uploader.as_numpy_array(),
                anim_pose,
            )
            palette_bytes = skin_palette_flat_bytes(palette, _SKIN_MAX_BONES)
        else:
            palette_bytes = uploader.as_flat_bytes()
        self._skin_palette_bytes_cache[cache_key] = (
            stamp,
            bone_count,
            palette_bytes,
            str(getattr(uploader, "_skin_palette_formula", "") or ""),
            str(getattr(uploader, "_skin_inverse_bind_source", "") or ""),
        )
        return bone_count, palette_bytes, False

    def _skin_uploader_for_palette_scope(
        self,
        *,
        scope_key: tuple,
        palette_model,
        fallback_nodes,
    ):
        """Return the retained palette uploader for one renderer scope.

        PIE runtime actors and BAS attachment palette models are immutable for
        their retained lifetime, and both scope keys include the model object
        identity.  Once built, an exact-key hit therefore does not need another
        recursive ``all_nodes()`` walk for every skin draw.  Editable scene and
        model scopes retain the identity-signature validation used previously.
        """
        if MatrixPaletteUploader is None:
            return None
        cached = self._skin_uploaders_by_scope.get(scope_key)
        immutable_scope = bool(
            scope_key
            and scope_key[0] in {"runtime_actor", "bas_attachment"}
        )
        if cached is not None and immutable_scope:
            return cached[1]
        try:
            palette_nodes = list(palette_model.all_nodes())
        except Exception:
            palette_nodes = list(fallback_nodes)
        signature = tuple(id(item) for item in palette_nodes)
        if cached is not None and cached[0] == signature:
            return cached[1]
        uploader = MatrixPaletteUploader(max_bones=_SKIN_MAX_BONES)
        n_built = uploader.build_inverse_bind_pose(palette_model)
        self._skin_uploaders_by_scope[scope_key] = (signature, uploader)
        self._skin_uploader = uploader
        self._skin_model_id = self._wt_model_id
        self._skin_bone_count = n_built
        if not self._skin_logged:
            log.info(
                "GPU-SKINNING: MatrixPaletteUploader built %s inverse bind-pose "
                "matrices for scope %s",
                n_built,
                scope_key,
            )
            self._skin_logged = True
        return uploader

    @staticmethod
    def _transformed_bounds_outside_frustum(bounds, model_matrix, planes) -> bool:
        """Conservatively reject an affine-transformed local AABB."""
        if not bounds or not planes:
            return False
        try:
            mins = np.asarray(bounds[0], dtype=np.float64)[:3]
            maxs = np.asarray(bounds[1], dtype=np.float64)[:3]
            if mins.shape != (3,) or maxs.shape != (3,):
                return False
            if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
                return False
            if np.any(maxs < mins):
                return False
            matrix = np.asarray(model_matrix, dtype=np.float64).reshape(4, 4)
            center = (mins + maxs) * 0.5
            extents = (maxs - mins) * 0.5
            world_center = matrix[:3, :3] @ center + matrix[:3, 3]
            world_extents = np.abs(matrix[:3, :3]) @ extents
            world_bounds = (
                tuple(float(v) for v in world_center - world_extents),
                tuple(float(v) for v in world_center + world_extents),
            )
            return not bounds_intersects_frustum(world_bounds, planes)
        except Exception:
            # Culling must fail open: uncertain bounds stay drawable.
            return False

    def render(self,
               model,
               camera,
               W: int, H: int,
               textures: Optional[Dict[str, 'Image.Image']] = None,
               anim_pose=None,
               anim_time: float = 0.0,
               anim_base_pose=None,
               display_options=None,
               anim_name: str = '') -> Optional['Image.Image']:
        """
        Render `model` from `camera` into a W×H PIL RGBA image.

        Parameters
        ----------
        model      : KotorModel (from src.core.model_data)
        camera     : ArcBallCamera or duck-typed object
        W, H       : output image dimensions
        textures   : dict mapping lowercased texture name → PIL Image
        anim_pose  : AnimPose object (from animation_engine) or None
        anim_time  : current animation time in seconds (for UV scroll / flipbook)
        anim_base_pose : AnimPose | None
            FIX-SKIN-ANIM-D2: The animation's first-frame (t=0) pose.  When
            provided, the GPU skinning palette uses this as the bind reference
            instead of the static node hierarchy, matching the xoreos approach.

        Returns
        -------
        PIL RGBA Image, or None on failure.
        """
        t0 = time.perf_counter()
        if W <= 0 or H <= 0:
            return None
        textures = textures or {}
        if display_options is not None:
            self.display_options = display_options

        if self._ensure_context():
            result = self._render_gpu(model, camera, W, H, textures, anim_pose, anim_time,
                                       anim_base_pose=anim_base_pose, anim_name=anim_name)
            if result is not None:
                self.perf['last_frame_ms'] = (time.perf_counter() - t0) * 1000
                self.perf['backend'] = 'gpu'
                return result
            # GPU render failed; the viewport reports GPU unavailable instead of falling back.

        self.perf['last_frame_ms'] = (time.perf_counter() - t0) * 1000
        self.perf['backend'] = 'gpu_unavailable'
        return None

    # ── GPU render ────────────────────────────────────────────────────────────

    def _render_gpu(self, model, camera, W: int, H: int,
                    textures: Dict[str, 'Image.Image'],
                    anim_pose, anim_time: float,
                    anim_base_pose=None, anim_name: str = '') -> Optional['Image.Image']:
        """Full GPU render via ModernGL EGL."""
        ctx = self._ctx
        prog = self._prog
        if ctx is None or prog is None:
            return None
        if not _NUMPY or not _PIL:
            return None
        self.deferred_mesh_uploads = False
        _new_mesh_uploads_this_frame = 0

        # PERF-SCALE: Interactive preview skips MSAA below, but stays at full
        # resolution by default.  Lower interactive_render_scale only when the
        # user explicitly accepts reduced quality for more FPS.
        _full_W, _full_H = W, H
        _scale = max(0.25, min(1.0, float(getattr(self, "interactive_render_scale", 1.0) or 1.0)))
        if self.interactive and _scale < 1.0 and W > 200 and H > 200:
            W = max(8, int(W * _scale))
            H = max(8, int(H * _scale))

        try:
            self._begin_submission_frame()
            t_upload = time.perf_counter()

            # ── PERF-FBO: Dual Framebuffer strategy ──────────────────────
            # MSAA FBO:   High-quality (4x multisampled) — used for still frames.
            #             The MSAA resolve (copy_framebuffer) costs ~59ms on llvmpipe
            #             so it MUST be skipped during interactive camera drag.
            # Simple FBO: Non-multisampled — used during interactive drag.
            #             No resolve needed → saves 59ms/frame → enables 30+ fps.
            _MSAA_SAMPLES = 4
            _use_msaa = not self.interactive

            if _use_msaa:
                # ── MSAA FBO (quality path) ────────────────────────────────
                if (self._fbo is None or self._fbo_w != W or self._fbo_h != H):
                    if self._fbo is not None:
                        try:
                            ca = self._fbo.color_attachments[0]
                            if ca is not None: ca.release()
                            da = self._fbo.depth_attachment
                            if da is not None: da.release()
                            self._fbo.release()
                        except Exception: pass
                        if self._fbo_resolve is not None:
                            try:
                                ca = self._fbo_resolve.color_attachments[0]
                                if ca is not None: ca.release()
                                self._fbo_resolve.release()
                            except Exception: pass
                            self._fbo_resolve = None
                    try:
                        self._fbo = ctx.framebuffer(
                            color_attachments=[ctx.renderbuffer((W, H), components=4, samples=_MSAA_SAMPLES)],
                            depth_attachment=ctx.depth_renderbuffer((W, H), samples=_MSAA_SAMPLES),
                        )
                        # Texture color attachment (not a renderbuffer) so the
                        # bloom bright-pass can sample the resolved scene.
                        self._fbo_resolve_tex = ctx.texture((W, H), 4)
                        self._fbo_resolve = ctx.framebuffer(
                            color_attachments=[self._fbo_resolve_tex],
                        )
                        self._fbo_msaa = True
                    except Exception:
                        self._fbo = ctx.framebuffer(
                            color_attachments=[ctx.renderbuffer((W, H), components=4)],
                            depth_attachment=ctx.depth_renderbuffer((W, H)),
                        )
                        self._fbo_resolve = None
                        self._fbo_resolve_tex = None
                        self._fbo_msaa = False
                    self._fbo_w = W
                    self._fbo_h = H
                fbo = self._fbo
            else:
                # ── Simple FBO (fast interactive path) ─────────────────────
                if (self._fbo_simple is None or self._fbo_simple_w != W or self._fbo_simple_h != H):
                    if self._fbo_simple is not None:
                        try:
                            ca = self._fbo_simple.color_attachments[0]
                            if ca is not None: ca.release()
                            da = self._fbo_simple.depth_attachment
                            if da is not None: da.release()
                            self._fbo_simple.release()
                        except Exception: pass
                    self._fbo_simple_tex = ctx.texture((W, H), 4)
                    self._fbo_simple = ctx.framebuffer(
                        color_attachments=[self._fbo_simple_tex],
                        depth_attachment=ctx.depth_renderbuffer((W, H)),
                    )
                    self._fbo_simple_w = W
                    self._fbo_simple_h = H
                fbo = self._fbo_simple

            fbo.use()
            self._reset_frame_state(ctx, W, H)
            # PERF-READBACK: Clear with OPAQUE background (alpha=1.0) so that
            # the readback path can skip the expensive alpha compositing step
            # (saves ~19ms/frame at 800x600).  Keep this in sync with viewport._BG.
            ctx.clear(*self.viewport_background, 1.0)
            self._debug_log_gl_error("frame clear")
            ctx.enable(moderngl.DEPTH_TEST)
            # v7.0 FIX (Finding 5.8 — reone context.cpp cross-ref):
            # reone uses GL_LEQUAL (not GL_LESS) to match KotOR engine behavior.
            # GL_LEQUAL allows co-planar decal geometry to render correctly without
            # z-fighting, matching the original engine's depth test mode.
            ctx.depth_func = '<='
            _set_depth_write(ctx, True)  # depth writes ON by default

            # The uploaded static mesh indices are CCW relative to their normals
            # (verified with retail K2 001ebo1 and native Unreal import readback).
            # This right-handed view/projection path preserves that front side.
            # CW here hides room exteriors and exposes the backs of closed props.
            ctx.front_face = 'ccw'
            if self.show_wireframe or not self.cull_faces:
                ctx.disable(moderngl.CULL_FACE)
            else:
                ctx.enable(moderngl.CULL_FACE)
            try:
                ctx.wireframe = bool(self.show_wireframe and not self.show_solid)
            except Exception:
                pass

            # ── Camera matrices ────────────────────────────────────────────
            # ArcBallCamera.eye is a METHOD (not a property); call it if callable.
            # ArcBallCamera stores near/far as _near/_far; try both names.
            _eye = getattr(camera, 'eye', (0, 5, 10))
            eye    = tuple(_eye() if callable(_eye) else _eye)
            _target = getattr(camera, 'target', (0, 0, 0))
            target = tuple(_target() if callable(_target) else _target)
            # KotOR uses Z-up; default to (0,0,1).  ArcBallCamera has no 'up' attr.
            _up = getattr(camera, 'up', None)
            if _up is None:
                up = (0.0, 0.0, 1.0)   # Z-up (KotOR world convention)
            else:
                up = tuple(_up() if callable(_up) else _up)
            fov    = float(getattr(camera, 'fov',   45.0))
            # ArcBallCamera stores near/far as _near/_far (private)
            near   = float(getattr(camera, 'near',  getattr(camera, '_near',  0.01)))
            far    = float(getattr(camera, 'far',   getattr(camera, '_far',  2000.0)))

            aspect = W / max(1, H)
            proj   = _mat4_perspective(math.radians(fov), aspect, near, far)
            view   = _mat4_lookat(eye, target, up)
            model_mat  = _mat4_identity()
            # GLSL: gl_Position = MVP * pos = proj * view * model * pos
            # Matrices are standard row-major; _mat4_tobytes() transposes for GLSL column-major.
            mvp        = _mat4_mul(_mat4_mul(proj, view), model_mat)
            normal_mat = _mat3_normal(model_mat)
            _frustum_planes = None
            if bool(getattr(self, "enable_frustum_culling", True)):
                try:
                    _frustum_planes = extract_frustum_planes(mvp)
                except Exception:
                    # A malformed/custom camera must never hide the scene.
                    _frustum_planes = None

            # PERF-UNIFCACHE: Use cached uniform references from self._u instead of
            # prog['name'] dict lookups.  This saves ~0.4ms/frame on 56-node models.
            _u = self._u

            _u['u_mvp'].write(_mat4_tobytes(mvp))
            _u['u_model'].write(_mat4_tobytes(model_mat))
            _u['u_normal_mat'].write(normal_mat.T.astype(np.float32).tobytes())
            _u['u_cam_pos'].value = tuple(eye)
            _map_world_preview = dict(getattr(self, "map_studio_world_lighting_preview", {}) or {})
            _map_fog_enabled = bool(_map_world_preview.get("fog_previewed") and _map_world_preview.get("fog_enabled"))
            if 'u_map_fog_enabled' in _u:
                _u['u_map_fog_enabled'].value = 1 if _map_fog_enabled else 0
                _fog_color = tuple(
                    _map_world_preview.get(
                        "fog_preview_color_rgb",
                        _map_world_preview.get("fog_color_rgb", (0.0, 0.0, 0.0)),
                    )
                    or (0.0, 0.0, 0.0)
                )
                if len(_fog_color) < 3:
                    _fog_color = (0.0, 0.0, 0.0)
                _u['u_map_fog_color'].value = tuple(max(0.0, min(1.0, float(channel))) for channel in _fog_color[:3])
                _fog_near = max(0.0, float(_map_world_preview.get("fog_preview_near", _map_world_preview.get("fog_near", 0.0)) or 0.0))
                _fog_far = max(
                    _fog_near + 0.001,
                    float(_map_world_preview.get("fog_preview_far", _map_world_preview.get("fog_far", 1.0)) or 1.0),
                )
                _u['u_map_fog_near'].value = _fog_near
                _u['u_map_fog_far'].value = _fog_far

            # Lighting uniforms — only set once per frame (values don't change per-node).
            _u['u_light_dir'].value  = (0.4839, 0.3519, 0.7918)  # pre-normalised
            _u['u_light_dir2'].value = (-0.4973, -0.2842, 0.8195)  # pre-normalised
            _u['u_ambient'].value    = 0.65
            _u['u_specular'].value   = 0.10
            _u['u_shininess'].value  = 20.0
            _u['u_alpha_test'].value = 0.5
            _u['u_decal'].value      = 0
            _u['u_wateralpha'].value = 1.0
            _u['u_has_spec'].value   = 0
            _u['u_has_bump'].value   = 0
            _u['u_alpha'].value      = 1.0
            _u['u_node_alpha'].value = 1.0
            _u['u_blend_mode'].value = 0
            _u['u_has_tex'].value    = 0
            _u['u_has_lm'].value     = 0
            _u['u_lm_shade'].value   = 0
            _u['u_lm_composite_mode'].value = _lm_composite_mode()
            if 'u_lightmap_intensity' in _u:
                _u['u_lightmap_intensity'].value = float(getattr(self, 'lightmap_intensity', 0.55))
            if 'u_lightmap_mode' in _u:
                _lm_mode = str(getattr(self, 'lightmap_mode', 'baked') or 'baked').lower()
                _u['u_lightmap_mode'].value = 2 if _lm_mode in {'emissive', 'debug'} else 1 if _lm_mode in {'phong', 'dynamic_preview', 'hybrid'} else 0
            _u['u_has_env'].value    = 0
            if 'u_scene_lighting' in _u:
                _u['u_scene_lighting'].value = 1
            if 'u_scene_ambient' in _u:
                _u['u_scene_ambient'].value = float(getattr(self, 'scene_ambient', 0.06))
            if 'u_scene_light_count' in _u:
                _u['u_scene_light_count'].value = 0
            _u['u_diffuse'].value    = (1.0, 1.0, 1.0)
            _u['u_selfillum'].value  = (0.0, 0.0, 0.0)
            _u['u_uv_scroll'].value  = (0.0, 0.0)
            _u['u_uv_v_flip'].value  = 1.0
            _u['u_rotate_tex'].value = 0.0
            _u['u_flipbook_off'].value  = (0.0, 0.0)
            _u['u_flipbook_size'].value = (0.0, 0.0)
            _u['u_features'].value     = 0
            _u['u_water_time'].value   = 0.0
            _u['u_proc_type'].value    = 0
            self._wireframe_pass = bool(self.show_wireframe and not self.show_solid)
            if 'u_wireframe_enabled' in _u:
                _u['u_wireframe_enabled'].value = 1 if self._wireframe_pass else 0
            if 'u_wire_color' in _u:
                _u['u_wire_color'].value = tuple(self.wire_color)
            _render_mode_name = str(getattr(self, 'render_mode', 'realistic') or 'realistic').lower()
            _render_mode_int = 1 if _render_mode_name == 'flat' else 2 if _render_mode_name == 'shaded' else 0
            if 'u_render_mode' in _u:
                _u['u_render_mode'].value = _render_mode_int
            if 'u_selected' in _u:
                _u['u_selected'].value = 0
            if 'u_sprite_alpha_source' in _u:
                _u['u_sprite_alpha_source'].value = 0.0
            if 'u_sprite_glow' in _u:
                _u['u_sprite_glow'].value = 0.0
            _u['u_dangly_enabled'].value      = 0.0
            _u['u_dangly_displacement'].value = 0.0
            _u['u_dangly_time'].value         = 0.0
            _u['u_saber_enabled'].value       = 0.0
            _u['u_saber_displacement'].value  = 0.0
            _u['u_saber_length'].value        = 1.0
            _u['u_oit_enabled'].value         = 0
            if 'u_debug_visualize' in _u:
                _u['u_debug_visualize'].value = _debug_visualize_mode()

            # ── Phase A: GPU Skinning — default off for each frame ───────────
            # Set u_skin_enabled = 0 as the default per-frame state.
            # Skin nodes will set it to 1 in _draw_node before their draw call.
            if 'u_skin_enabled' in _u:
                _u['u_skin_enabled'].value = 0
            if 'u_bone_count' in _u:
                _u['u_bone_count'].value = 0

            self._draw_grid(ctx, mvp)

            self.perf['gpu_upload_ms'] = (time.perf_counter() - t_upload) * 1000
            t_draw = time.perf_counter()

            total_tris = 0

            # ── Two-pass rendering: opaque first, then transparent ─────────
            # Pass 1: opaque/punch-through (depth write ON, depth test ON)
            # Pass 2: alpha-blended / additive (depth write OFF, depth test ON)
            # This prevents transparent geometry from blocking opaque geometry
            # behind it — fixing the "see-through" appearance on character models.
            # KotorModel uses all_nodes() — there is NO .nodes list attribute.
            _all_nodes_fn = getattr(model, 'all_nodes', None)
            if _all_nodes_fn is not None:
                nodes = list(_all_nodes_fn())
            else:
                nodes = getattr(model, 'nodes', []) if model is not None else []

            # ── FIX-PERSCACHE: Persistent world-transform cache ────────────
            _cur_model_id = id(model)
            _model_changed = (_cur_model_id != self._wt_model_id)
            if _model_changed or len(nodes) != self._node_cache_node_count:
                self._wt_cache.clear()
                self._wt_model_id = _cur_model_id
                self._skin_uploaders_by_scope.clear()
                self._skin_palette_bytes_cache.clear()

            # ── Phase A: GPU Skinning — build bone palette for skin models ─────
            # Detect if this model has any skin nodes; if so, build the
            # MatrixPaletteUploader's inverse bind-pose (once per model).
            # The palette is then uploaded per-frame when animating.
            _has_skin_nodes = False
            if _GPU_SKINNING:
                for _sn in nodes:
                    if bool(getattr(_sn, 'is_skin', False)):
                        _has_skin_nodes = True
                        break
                if _has_skin_nodes:
                    # FIX-SKIN-QBONE: Skin palettes are now built per skin node
                    # immediately before the draw call.  qBone/tBone slots are
                    # indexed by the skin node's local bone_map, not the global
                    # model node order, so a single model-wide palette is wrong.
                    pass

            class _SkinPaletteModelView:
                def __init__(self, source_model, view_nodes):
                    self._nodes = tuple(view_nodes)
                    self.name = str(
                        getattr(self._nodes[0], "_gr_scene_object_name", "")
                        or getattr(self._nodes[0], "name", "")
                        or getattr(source_model, "name", "")
                        or ""
                    )
                    self.supermodel = str(getattr(source_model, "supermodel", "") or "")
                    if bool(getattr(source_model, "_gr_bas_attachment_palette_model", False)):
                        self._gr_bas_attachment_palette_model = True

                def all_nodes(self):
                    return list(self._nodes)

            def _skin_palette_scope_for_node(nd):
                if bool(getattr(nd, "_gr_bas_attachment_layer", False)):
                    attachment_model = bas_attachment_palette_model_for_node(nd)
                    if attachment_model is not None:
                        return ("bas_attachment", id(attachment_model)), attachment_model

                runtime_model = runtime_source_model_for_node(nd)
                if runtime_model is not None:
                    root = getattr(nd, "_gr_scene_object_root_ref", None)
                    object_id = str(
                        getattr(root, "_gr_scene_object_id", "")
                        or getattr(nd, "_gr_scene_object_id", "")
                        or ""
                    )
                    actor_identity = root if root is not None else nd
                    return (
                        "runtime_actor",
                        object_id or str(id(actor_identity)),
                        id(actor_identity),
                        id(runtime_model),
                    ), runtime_model

                root = getattr(nd, "_gr_scene_object_root_ref", None)
                if root is not None and bool(getattr(root, "_gr_scene_object_root", False)):
                    object_id = str(
                        getattr(root, "_gr_scene_object_id", "")
                        or getattr(nd, "_gr_scene_object_id", "")
                        or ""
                    )
                    import_id = str(
                        getattr(root, "_gr_scene_import_id", "")
                        or getattr(nd, "_gr_scene_import_id", "")
                        or ""
                    )
                    key = ("scene", object_id or import_id or str(id(root)), id(root))
                    root_nodes = [
                        item for item in nodes
                        if item is root or getattr(item, "_gr_scene_object_root_ref", None) is root
                    ]
                    return key, _SkinPaletteModelView(model, root_nodes or [root])
                return ("model", _cur_model_id), model

            def _skin_uploader_for_node(nd, resolved_scope=None):
                scope_key, palette_model = (
                    resolved_scope
                    if resolved_scope is not None
                    else _skin_palette_scope_for_node(nd)
                )
                return self._skin_uploader_for_palette_scope(
                    scope_key=scope_key,
                    palette_model=palette_model,
                    fallback_nodes=nodes,
                )

            # PERF: Only stamp model refs and compute proxy IDs when model changes.
            # These are O(N) walks that produce identical results across frames for
            # the same model.
            if _model_changed or len(nodes) != self._node_cache_node_count:
                for _n in nodes:
                    try:
                        _n._model_ref = model
                    except (AttributeError, TypeError):
                        pass

                # FIX-SKINPROXY: Compute skin proxy node IDs for this model.
                _proxy_node_ids: set = set()
                try:
                    _skin_tex_verts: dict = {}
                    for _pn in nodes:
                        if not getattr(_pn, 'is_mesh', False) or not getattr(_pn, 'is_skin', False):
                            continue
                        _pt = str(getattr(_pn, 'texture', '') or '').strip().lower()
                        if not _pt or _pt in ('null', 'none', ''):
                            continue
                        _pnv = len(getattr(_pn, 'vertices', []))
                        if _pnv == 0:
                            continue
                        if _pt not in _skin_tex_verts:
                            _skin_tex_verts[_pt] = []
                        _skin_tex_verts[_pt].append((_pn, _pnv))
                    # Use the shared inner-geometry list (see render_constants.py).
                    # This list MUST match viewport.py classification so the CPU
                    # and GPU paths treat NPC-head eyelid/gumskin/tonguemesh nodes
                    # identically — otherwise the same model renders with missing
                    # inner geometry under one renderer and not the other.
                    for _pn in nodes:
                        if not getattr(_pn, 'is_mesh', False) or getattr(_pn, 'is_skin', False):
                            continue
                        _pt = str(getattr(_pn, 'texture', '') or '').strip().lower()
                        if not _pt or _pt in ('null', 'none', ''):
                            continue
                        if not getattr(_pn, 'uvs', []):
                            continue
                        _pn_name_low = str(getattr(_pn, 'name', '') or '').lower()
                        if any(_ig in _pn_name_low for _ig in _INNER_GEO_SUBSTRINGS):
                            continue
                        _pnv = len(getattr(_pn, 'vertices', []))
                        _matches = _skin_tex_verts.get(_pt, [])
                        if len(_matches) == 1 and _matches[0][1] > _pnv:
                            _proxy_node_ids.add(id(_pn))
                except Exception:
                    pass
                self._node_cache_proxy_ids = _proxy_node_ids
            else:
                _proxy_node_ids = self._node_cache_proxy_ids
            # Local alias for closure capture
            _wt_cache = self._wt_cache
            _bas_root_cache: Dict[int, object | None] = {}
            _bas_socket_cache: Dict[int, object | None] = {}
            _bas_transform_cache: Dict[object, object] = {}

            def _bas_attachment_root_for_node(nd):
                _node_id = id(nd)
                if _node_id in _bas_root_cache:
                    return _bas_root_cache[_node_id]
                _cur = nd
                _seen = set()
                _trail = []
                _result = None
                while _cur is not None and id(_cur) not in _seen:
                    _cur_id = id(_cur)
                    if _cur_id in _bas_root_cache:
                        _result = _bas_root_cache[_cur_id]
                        break
                    _seen.add(_cur_id)
                    _trail.append(_cur)
                    if bool(getattr(_cur, "_gr_bas_attachment_root", False)):
                        _result = _cur
                        break
                    _cur = getattr(_cur, "parent", None)
                for _item in _trail:
                    _bas_root_cache[id(_item)] = _result
                return _result

            def _bas_attachment_socket_node(_bas_root):
                _root_id = id(_bas_root)
                if _root_id in _bas_socket_cache:
                    return _bas_socket_cache[_root_id]
                _socket_name = str(getattr(_bas_root, "_gr_bas_socket_name", "") or "").lower()
                _body_root = getattr(_bas_root, "parent", None)
                if _body_root is None or not _socket_name:
                    _bas_socket_cache[_root_id] = None
                    return None
                if str(getattr(_body_root, "name", "") or "").lower() == _socket_name:
                    _bas_socket_cache[_root_id] = _body_root
                    return _body_root
                _stack = [_body_root]
                _seen = {id(_bas_root)}
                while _stack:
                    _cur = _stack.pop()
                    if _cur is None or id(_cur) in _seen:
                        continue
                    _seen.add(id(_cur))
                    if str(getattr(_cur, "name", "") or "").lower() == _socket_name:
                        _bas_socket_cache[_root_id] = _cur
                        return _cur
                    for _child in reversed(getattr(_cur, "children", []) or []):
                        if bool(getattr(_child, "_gr_bas_attachment_root", False)):
                            continue
                        _stack.append(_child)
                _bas_socket_cache[_root_id] = None
                return None

            # A node can be queried several times in one frame (light gather,
            # transparent sort, and draw).  Animated transforms are not valid
            # across frames, but they are immutable for this render call.
            _frame_wt_cache: Dict[int, tuple] = {}
            _node_anim_pose_cache: Dict[int, object | None] = {}

            def _resolved_animation_pose(nd):
                """Resolve one actor pose once without animating static map nodes.

                PIE publishes a fresh ``ScopedAnimationPoseSet`` after every
                retained actor batch.  Treating that non-null collection as a
                pose for every scene node discarded the persistent transform
                path for the complete module, even though almost all room and
                placeable nodes belong to no animated actor.  Cache the scoped
                identity lookup for this render and return ``None`` for those
                static nodes so they continue through ``_wt_cache``.
                """

                if anim_pose is None or nd is None:
                    return None
                node_id = id(nd)
                if node_id not in _node_anim_pose_cache:
                    _node_anim_pose_cache[node_id] = _effective_animation_pose_for_node(
                        nd,
                        anim_pose,
                    )
                return _node_anim_pose_cache[node_id]

            def _get_world_transform(nd):
                """Return (world_pos, world_orient) with persistent memoization.

                FIX-PERSCACHE: For static geometry (no anim_pose override for this
                node) the result is cached across frames so subsequent renders skip
                the O(depth) parent-chain walk entirely.

                FIX-SKIN-ANIM: Skin mesh nodes MUST NOT receive animation position/
                rotation overrides.  Skin vertices are pre-baked in bind-pose world
                space.  Applying the animated bone rotation directly to the mesh node
                causes the entire skin mesh to be incorrectly rotated → extreme
                stretching/deformation (observed on c_brith wings, limbs).

                In the Aurora engine, skin mesh deformation is handled by Linear
                Blend Skinning (LBS) using per-bone matrices computed from the
                animated skeleton hierarchy.  Without GPU skinning, the best we can
                do is keep skin meshes in their bind pose.

                Non-skin nodes (rigid trimesh attachments like eyes, tongue, horns)
                DO receive animation overrides so they move with the skeleton.

                Cross-ref: reone mesh.cpp:307 (skin bone matrix computation);
                           KotOR.js OdysseyModel3D.ts:730 (buildSkeleton, bind);
                           xoreos modelnode.cpp:805 (skin pass separation).
                """
                nid = id(nd)
                if nid in _frame_wt_cache:
                    return _frame_wt_cache[nid]
                _bas_root = _bas_attachment_root_for_node(nd)
                if _bas_root is not None:
                    # BAS heads need two poses at once: the body pose places the
                    # headhook socket, while the attachment source pose drives
                    # facial bones and rigid eye/mouth descendants.  The shared
                    # renderer-neutral helper owns that space conversion.
                    result = _bas_attachment_world_transform(
                        nd,
                        _bas_root,
                        anim_pose=_resolved_animation_pose(nd),
                        transform_cache=_bas_transform_cache,
                    )
                    _frame_wt_cache[nid] = result
                    return result
                if anim_pose is not None:
                    _is_skin_nd = bool(getattr(nd, 'is_skin', False))
                    if not _is_skin_nd:
                        _resolved_pose = _resolved_animation_pose(nd)
                        if _resolved_pose is not None:
                            # Pass the actor-local pose, not the short-lived
                            # scoped collection.  Unchanged actors therefore
                            # retain their own world-chain cache while a newly
                            # evaluated pose still recomputes exactly its actor.
                            result = _animated_node_world_transform(
                                nd,
                                _resolved_pose,
                            )
                            _frame_wt_cache[nid] = result
                            return result
                # Static / no override → use persistent cache
                if nid in _wt_cache:
                    result = _wt_cache[nid]
                    _frame_wt_cache[nid] = result
                    return result
                try:
                    _wp, _wo = nd.world_transform()
                except Exception:
                    _wp = getattr(nd, 'position', (0.0, 0.0, 0.0))
                    _wo = getattr(nd, 'rotation', (0.0, 0.0, 0.0, 1.0))
                result = (_wp, _wo)
                _wt_cache[nid] = result
                _frame_wt_cache[nid] = result
                return result

            # ── FIX-DEFORM: Deformation-helper detection ──────────────────
            # FIX-DEFORM: Self-contained deformation-helper filter (no viewport import).
            # KotOR MDL models contain internal bone-proxy / skeleton-helper mesh nodes
            # that must NOT be rendered as geometry.  Detection rules (same logic as
            # viewport._is_deformation_helper, inlined here for GPU-path independence):
            #   1. Non-skin nodes with no UVs → helper (UNLESS module/area model)
            #   2. Non-skin '_g'/'_g0'/'_dum' nodes without an explicitly
            #      renderable rigid surface → helper
            #   3. Nodes with null/empty texture AND no UVs → helper
            #   4. Skin nodes with a real texture and UVs → ALWAYS render
            # Reference: viewport._is_deformation_helper, PyKotor geometry_utils.py,
            #            KotOR engine ProcessSkinSeams().
            _gpu_model_cls   = (str(getattr(model, 'classification', 'character') or 'character')).lower()
            # FIX-MODEL-TYPE-ZERO: model_type=0 means "module/tile" in KotOR.
            # Using `int(val or 4)` treated 0 as falsy → replaced with 4 (character)
            # which broke module UV-sentinel exemption for tile models.
            # Fix: only use the default when the raw value is None/missing.
            _gpu_model_type_raw = getattr(model, 'model_type', None)
            _gpu_model_type  = int(_gpu_model_type_raw) if _gpu_model_type_raw is not None else 4
            _gpu_is_module   = (_gpu_model_cls in ('effect', 'tile', 'other') or
                                _gpu_model_type in (0, 2))
            _scene_light_candidates = self._scene_light_candidates_for_model(
                model,
                nodes,
            )
            _scene_lights = self._scene_light_records(
                _scene_light_candidates,
                _get_world_transform,
                reference_position=target,
            )
            self._upload_scene_lights(prog, _u, _scene_lights)

            def _is_deform_helper(nd) -> bool:
                """Return True if nd is a bone-proxy/deformation-helper that must not render."""
                is_skin    = bool(getattr(nd, 'is_skin', False))
                tex_name   = str(getattr(nd, 'texture', '') or '').strip().lower()
                has_tex    = tex_name and tex_name not in ('null', '', 'none', '****')
                uvs        = getattr(nd, 'uvs', [])
                has_uvs    = bool(uvs) and len(uvs) > 0
                node_name  = str(getattr(nd, 'name', '') or '').lower()

                # NODE_SABER records are animation/runtime helpers that duplicate
                # the two ordinary crossed glow cards.  Drawing all four stacks
                # rectangular additive layers and creates blocky glow bands.
                if _is_saber_runtime_helper(nd):
                    return True

                # Ordinary stock lightsaber glow cards can omit MDL UV arrays;
                # the GPU path synthesizes a smooth preview quad and texture.
                if is_lightsaber_blade_node(nd):
                    return False

                # Skin nodes with a real texture and UVs are always renderable
                if is_skin and has_tex and has_uvs:
                    return False

                # Several KOTOR 2 droids author their visible articulated body
                # as rigid _g meshes.  A true render flag plus diffuse texture
                # and UVs is stronger evidence than the helper-style suffix.
                if _node_has_renderable_rigid_surface(nd):
                    return False

                # Skin nodes with null texture and no UVs are helpers
                if is_skin and not has_tex and not has_uvs:
                    return True

                # Non-skin: no UVs → helper (module/area models exempt: MDX-sourced UVs
                # may be empty if MDX wasn't loaded, but mesh is still real geometry)
                if not is_skin and not has_uvs:
                    if _gpu_is_module:
                        return False  # module geometry: render flat-shaded without UVs
                    return True

                # Name-based helper detection is only a fallback after the
                # explicit visible-surface evidence above.
                if not is_skin:
                    if (node_name.endswith('_g') or node_name.endswith('_g0')
                            or node_name.endswith('_dum')):
                        return True

                return False

            # ── Transparency classification ─────────────────────────────
            # Follows the definitive Aurora engine approach from reference
            # implementations (xoreos, reone, KotOR.js):
            #
            # xoreos (modelnode.cpp:500-506):
            #   if (hasTransparencyHint)
            #       isTransparent = transparencyHint;
            #   else
            #       isTransparent = hasAlpha;  (texture has alpha channel)
            #
            # reone (mesh.cpp isTransparent()):
            #   PunchThrough → NOT transparent (opaque with alpha-test)
            #   Additive → transparent
            #   alpha < 1.0 → transparent
            #   Has envmap/bumpmap → NOT transparent
            #   Diffuse has alpha channel → transparent
            #
            # KEY FIX: Inner-geometry nodes (eyes, teeth, gums) should NOT
            # be promoted to the transparent pass.  They are opaque geometry
            # (transparency_hint == 0) drawn in the opaque pass with depth
            # write ON.  The head/face mesh has transparency_hint > 0 and
            # its texture contains alpha=0 pixels at the eye sockets and
            # mouth gap.  When drawn in the cutout pass (alpha-test discard),
            # those holes reveal the already-rendered inner-geo beneath.
            # This is exactly how the original Aurora engine works.

            def _classify_node(nd, ap):
                """Return (node_alpha, txi_blend, is_transparent, has_env).

                Transparency classification for three-pass rendering:
                  - Pass 1 (opaque, depth write ON): solid surfaces,
                    inner-geometry (eyes, teeth), env-map surfaces.
                  - Pass 2 (alpha-cutout, depth write ON, shader discard):
                    transparency_hint > 0 nodes with alpha-test textures
                    (head meshes with eye-socket/mouth-gap alpha holes).
                  - Pass 3 (transparent, depth write OFF): additive blending,
                    wateralpha < 1, decal, per-node alpha < 1.

                Cross-ref: xoreos modelnode.cpp:500; reone mesh.cpp:215;
                           KotOR.js OdysseyModel3D.ts:1578.
                """
                na = float(getattr(nd, 'alpha', 1.0))
                na = max(0.0, min(1.0, na))
                if ap is not None:
                    _pn = _pose_node_for_transform(nd, ap)
                    if _pn is not None and getattr(_pn, 'alpha', None) is not None:
                        na = max(0.0, min(1.0, float(_pn.alpha)))
                tb = int(getattr(nd, 'txi_blending', 0))
                _th = int(getattr(nd, 'transparency_hint', 0))
                _at = float(getattr(nd, 'txi_alpha_test', 0.0))
                has_env = bool(getattr(nd, 'txi_envmaptexture', ''))
                wa = float(getattr(nd, 'txi_wateralpha', 1.0))
                decal = bool(getattr(nd, 'txi_decal', False))

                if _gpu_is_module and _render_mode_int in (1, 2):
                    return 1.0, 0, False, False

                # Untextured selfillum planes render as additive glows, never
                # opaque/cutout geometry (Star Map lightflare burst/holo bits).
                if tb == 0 and _is_untextured_glow(nd):
                    tb = 1

                # Punchthrough classification (alpha-test discard in shader):
                # Use cutout pass when the MDL node explicitly requests it
                # via transparency_hint > 0 (head meshes, hair cards, foliage).
                # Also apply when TXI blending is already set to punchthrough.
                # Do NOT auto-promote based solely on txi_alpha_test — see
                # the c_bantha body-pixel discard bug (threshold=0.9333 on
                # opaque body textures with transparency_hint=0).
                if tb == 0 and _th > 0:
                    # transparency_hint > 0 → alpha-test cutout pass
                    # This handles head meshes (eye-socket/mouth holes),
                    # hair cards, and other alpha-tested geometry.
                    tb = 2

                # Transparent pass classification (following xoreos/reone):
                # - Additive blending (tb==1)
                # - Semi-transparent water/glass (wateralpha < 1)
                # - Decal overlays
                # - Per-node alpha < 1 (holograms, glass) unless env-mapped
                is_trans = ((tb == 1) or
                            (wa < 0.999) or
                            decal or
                            (na < 0.999 and not has_env))
                if self._sprite_alpha_source(nd) and self._sprite_glow(nd) > 0.001:
                    if tb == 0:
                        tb = 3
                    is_trans = True
                if is_lightsaber_blade_node(nd) and not self._has_sprite_material_override(nd):
                    tb = 1
                    is_trans = True
                return na, tb, is_trans, has_env

            # ── PERF-NODECACHE: Three-pass node classification with caching ──
            # Node classification (opaque/cutout/transparent) is expensive due to
            # _is_deform_helper() and _classify_node() per node.  Cache the result
            # per model and reuse across frames.  Invalidate when the model changes.
            # Classification is invalidated explicitly by material/visibility
            # edits.  Do not inspect every node merely to prove that no edit
            # occurred: this function is on every camera and PIE animation frame.
            _root_node = getattr(model, "root_node", None)
            _model_classification_revision = (
                int(getattr(model, "_gr_classification_revision", 0) or 0),
                int(getattr(_root_node, "_gr_classification_revision", 0) or 0),
            )
            _need_reclassify = (
                _cur_model_id != self._node_cache_model_id
                or len(nodes) != self._node_cache_node_count
                or self._node_cache_built_revision != self._node_classification_revision
                or _model_classification_revision != self._node_cache_model_revision
            )
            if _need_reclassify:
                opaque_nodes      = []
                cutout_nodes      = []
                transparent_nodes = []
                for node in nodes:
                    # Stock room MDLs embed AABB/pathing geometry in the same
                    # hierarchy as their visual meshes.  It has no diffuse map
                    # and must be presented only by the walkmesh overlay; drawing
                    # it here creates a white, slightly offset duplicate room.
                    if _node_is_embedded_walkmesh(node):
                        continue
                    if getattr(node, '_gr_hidden', False):
                        continue
                    if not getattr(node, 'render', True):
                        continue
                    verts = getattr(node, 'vertices', getattr(node, 'verts', []))
                    faces = getattr(node, 'faces', [])
                    if not verts or not faces:
                        continue
                    try:
                        if _is_deform_helper(node):
                            continue
                    except Exception:
                        pass
                    if id(node) in _proxy_node_ids:
                        continue
                    na, tb, is_trans, has_env = _classify_node(node, anim_pose)
                    if is_trans:
                        transparent_nodes.append(node)
                    elif tb == 2:
                        cutout_nodes.append(node)
                    else:
                        opaque_nodes.append(node)
                # Cache the classification
                self._node_cache_model_id = _cur_model_id
                self._node_cache_opaque = opaque_nodes
                self._node_cache_cutout = cutout_nodes
                self._node_cache_transparent = transparent_nodes
                self._node_cache_proxy_ids = _proxy_node_ids
                self._node_cache_is_module = _gpu_is_module
                self._node_cache_node_count = len(nodes)
                self._node_cache_model_revision = _model_classification_revision
                self._node_cache_built_revision = self._node_classification_revision
            else:
                # Reuse cached classification
                opaque_nodes = self._node_cache_opaque
                cutout_nodes = self._node_cache_cutout
                transparent_nodes = self._node_cache_transparent

            # PERF-FRUSTUM-ACTOR: Retained PIE creatures contain many skinned
            # submeshes whose animated vertex bounds are intentionally not culled
            # individually.  Reject the complete actor only when an expanded
            # source-model envelope is wholly outside the camera frustum.  The
            # expansion keeps idle limb/head motion conservative, and failure to
            # resolve trustworthy bounds always leaves the actor visible.
            _actor_frustum_cache: Dict[int, bool] = {}

            def _pie_actor_root_for_node(nd):
                root_ref = getattr(nd, "_gr_scene_object_root_ref", None)
                if root_ref is not None and bool(
                    getattr(root_ref, "_gr_map_studio_pie_actor", False)
                ):
                    return root_ref
                return None

            def _pie_actor_outside_frustum(nd) -> bool:
                if _frustum_planes is None:
                    return False
                actor_root = _pie_actor_root_for_node(nd)
                if actor_root is None:
                    return False
                root_id = id(actor_root)
                cached = _actor_frustum_cache.get(root_id)
                if cached is not None:
                    return cached
                outside = False
                try:
                    source_model = runtime_source_model_for_node(actor_root)
                    mins = tuple(float(v) for v in getattr(source_model, "bb_min", ())[:3])
                    maxs = tuple(float(v) for v in getattr(source_model, "bb_max", ())[:3])
                    spans = tuple(maxs[i] - mins[i] for i in range(3))
                    if len(mins) == 3 and len(maxs) == 3 and max(spans) > 1.0e-5:
                        padding = max(1.25, max(spans) * 0.35)
                        expanded = (
                            tuple(mins[i] - padding for i in range(3)),
                            tuple(maxs[i] + padding for i in range(3)),
                        )
                        actor_matrix = _scene_gpu_model_matrix(actor_root)
                        if actor_matrix is not None:
                            outside = self._transformed_bounds_outside_frustum(
                                expanded,
                                actor_matrix,
                                _frustum_planes,
                            )
                except Exception:
                    outside = False
                _actor_frustum_cache[root_id] = outside
                return outside

            _classified_mesh_count = (
                len(opaque_nodes) + len(cutout_nodes) + len(transparent_nodes)
            )
            _culled_actor_meshes = 0
            if _frustum_planes is not None:
                def _without_culled_actors(items):
                    nonlocal _culled_actor_meshes
                    visible = []
                    for item in items:
                        if _pie_actor_outside_frustum(item):
                            _culled_actor_meshes += 1
                        else:
                            visible.append(item)
                    return visible

                opaque_nodes = _without_culled_actors(opaque_nodes)
                cutout_nodes = _without_culled_actors(cutout_nodes)
                transparent_nodes = _without_culled_actors(transparent_nodes)

            # Opaque submission order does not affect compositing.  Group it by
            # material/sampler state so the exact uniform cache and texture-unit
            # cache below can suppress redundant Python→OpenGL calls.  Keep
            # transparent geometry on its required back-to-front path.
            def _opaque_submission_sort_key(node):
                return (
                    str(getattr(node, "texture", "") or "").strip().lower(),
                    str(getattr(node, "lightmap", "") or "").strip().lower(),
                    str(getattr(node, "txi_envmaptexture", "") or "").strip().lower(),
                    str(getattr(node, "txi_specularcolour", "") or "").strip().lower(),
                    str(
                        getattr(node, "txi_bumpmaptexture", "")
                        or getattr(node, "bump_map", "")
                        or ""
                    ).strip().lower(),
                    int(getattr(node, "txi_blending", 0) or 0),
                    bool(getattr(node, "txi_clamp_s", False)),
                    bool(getattr(node, "txi_clamp_t", False)),
                    tuple(float(value) for value in tuple(getattr(node, "diffuse", ()) or ())[:3]),
                    tuple(float(value) for value in tuple(getattr(node, "selfillum", ()) or ())[:3]),
                    float(getattr(node, "shininess", 0.0) or 0.0),
                    id(node),
                )

            opaque_nodes = sorted(opaque_nodes, key=_opaque_submission_sort_key)
            cutout_nodes = sorted(cutout_nodes, key=_opaque_submission_sort_key)

            _culled_rigid_meshes = 0
            _draw_call_count = 0
            _drawn_node_names: list[str] = []
            _bound_texture_units: dict[int, int] = {}
            _texture_sampler_states: dict[int, tuple[object, ...]] = {}

            def _bind_texture_exact(texture, location: int) -> None:
                """Bind only when a texture unit does not already name this object."""

                texture_id = id(texture)
                if _bound_texture_units.get(int(location)) == texture_id:
                    return
                texture.use(location=int(location))
                _bound_texture_units[int(location)] = texture_id

            def _set_texture_sampler_exact(
                texture,
                *,
                repeat_x: bool,
                repeat_y: bool,
                filter_value=None,
            ) -> None:
                """Avoid re-submitting identical sampler state for shared materials."""

                texture_id = id(texture)
                desired = (bool(repeat_x), bool(repeat_y), filter_value)
                if _texture_sampler_states.get(texture_id) == desired:
                    return
                texture.repeat_x = desired[0]
                texture.repeat_y = desired[1]
                if filter_value is not None:
                    texture.filter = filter_value
                _texture_sampler_states[texture_id] = desired

            def _draw_node(node, tex_name_override: str = '',
                           pass_name: str = 'opaque',
                           override_vao=None, override_tri_count: int = 0):
                """Draw a single node.

                tex_name_override: override the diffuse texture name (for multi-tex).
                pass_name: render pass label for diagnostic state tracing.
                override_vao: when set, use this VAO instead of the cached gm.vao
                    (for per-material-slot sub-mesh drawing).
                override_tri_count: triangle count for the override VAO.
                """
                nonlocal total_tris, _new_mesh_uploads_this_frame
                nonlocal _culled_rigid_meshes, _draw_call_count
                # Use world-space transform (full parent-chain walk) for correct
                # positioning of all mesh nodes, not just local node.position.
                wp, wo = _get_world_transform(node)
                scene_gpu_mat = _scene_gpu_model_matrix(node)
                vbo_wp, vbo_wo = wp, wo
                _nd_is_skin = bool(getattr(node, 'is_skin', False))
                _node_anim_pose = _resolved_animation_pose(node)
                _scene_animated_node_draw_mat = None
                if scene_gpu_mat is not None and anim_pose is None:
                    authored_transform = _scene_authored_world_transform(node)
                    if authored_transform is not None:
                        vbo_wp, vbo_wo = authored_transform

                node_alpha = float(getattr(node, 'alpha', 1.0))
                node_alpha = max(0.0, min(1.0, node_alpha))
                selfillum  = getattr(node, 'selfillum', (0.0, 0.0, 0.0))
                if _node_anim_pose is not None:
                    _pn = _pose_node_for_transform(node, _node_anim_pose)
                    if _pn is not None:
                        if getattr(_pn, 'alpha', None) is not None:
                            node_alpha = max(0.0, min(1.0, float(_pn.alpha)))
                        if getattr(_pn, 'selfillum', None) is not None:
                            selfillum = _pn.selfillum
                if _gpu_is_module and _render_mode_int in (1, 2):
                    node_alpha = 1.0
                if node_alpha <= (1.0 / 255.0) and not self._wireframe_pass:
                    # Alpha-keyed-off geometry (Star Map Sphere02 during "on")
                    # must not rasterize at all: an invisible surface that
                    # still writes depth z-rejects every particle and additive
                    # mesh behind it, and the engine treats alpha 0 as hidden.
                    return
                _is_blade_node = is_lightsaber_blade_node(node)
                if _is_blade_node:
                    # The procedural texture already contains the emissive core
                    # and aura.  A uniform self-illumination term colors every
                    # fragment of the rectangular card under ONE/ONE blending.
                    selfillum = (0.0, 0.0, 0.0)

                node_id = id(node)
                # FIX-SKIN-ANIM: Skin nodes are NOT considered "animated" for VBO
                # rebuild purposes.  Their vertices are in bind-pose world space and
                # should only change via GPU skinning (not yet implemented).
                # Rebuilding skin VBOs with animation transforms causes stretching.
                #
                # FIX-GPU-ANIM-CHAIN: Non-skin nodes must also be rebuilt when any
                # ANCESTOR has animation keys, because the node's world position
                # depends on the full parent chain.  Rigid attachments (eyes, horns,
                # accessories) move when their parent bone is animated even if they
                # themselves have no animation keys.
                _bas_root_for_draw = _bas_attachment_root_for_node(node)
                _bas_skin_draw_mat = None
                if _nd_is_skin and _bas_root_for_draw is not None:
                    vbo_wp, vbo_wo = _bas_attachment_local_transform_np(node, _bas_root_for_draw)
                    _bas_root_wp, _bas_root_wo = _get_world_transform(_bas_root_for_draw)
                    _bas_skin_draw_mat = _mat4_from_pos_quat_scale(
                        _bas_root_wp,
                        _bas_root_wo,
                        (1.0, 1.0, 1.0),
                    )
                _skin_anim_pose = _node_anim_pose
                _skin_anim_base_pose = (
                    _effective_animation_pose_for_node(node, anim_base_pose)
                    if anim_base_pose is not None else None
                )
                _skin_palette_scope = (
                    _skin_palette_scope_for_node(node)
                    if _nd_is_skin and _has_skin_nodes
                    else None
                )
                _skin_palette_scope_key = (
                    _skin_palette_scope[0]
                    if _skin_palette_scope is not None
                    else ("model", _cur_model_id)
                )
                _skin_uploader = (
                    _skin_uploader_for_node(node, _skin_palette_scope)
                    if _skin_palette_scope is not None
                    else None
                )
                _skin_can_lbs = bool(
                    _nd_is_skin
                    and _has_skin_nodes
                    and _skin_uploader is not None
                    and _skin_anim_pose is not None
                    and animation_pose_applies_to_node(node, _skin_anim_pose)
                    and getattr(node, 'bone_map', None)
                    and getattr(node, 'skin_data', None)
                )
                if (
                    _nd_is_skin
                    and bool(getattr(node, "_gr_bound_to_kotor_skeleton", False))
                    and _node_anim_pose is not None
                ):
                    _preview_key = (_cur_model_id, node_id)
                    if _preview_key not in self._skin_preview_gate_logged:
                        self._skin_preview_gate_logged.add(_preview_key)
                        log.info(
                            "GPU-SKINNING: Character Builder preview node=%s enabled=%s "
                            "skin_nodes=%s uploader=%s applies=%s bones=%d rows=%d base_pose=%s",
                            getattr(node, "name", "?"),
                            bool(_skin_can_lbs),
                            bool(_has_skin_nodes),
                            _skin_uploader is not None,
                            bool(animation_pose_applies_to_node(node, _skin_anim_pose)),
                            len(getattr(node, "bone_map", []) or []),
                            len(getattr(node, "skin_data", []) or []),
                            _skin_anim_base_pose is not None,
                        )
                _skin_bind_transform = bool(_nd_is_skin and not _skin_can_lbs)
                is_animated = False
                if _node_anim_pose is not None and hasattr(_node_anim_pose, 'nodes') and not _nd_is_skin:
                    # Check if this node or any ancestor has animation data
                    _acheck = node
                    while _acheck is not None:
                        _acheck_pose = _resolved_animation_pose(_acheck)
                        if _acheck_pose is not None and _pose_node_for_transform(_acheck, _acheck_pose) is not None:
                            is_animated = True
                            break
                        _acheck = getattr(_acheck, 'parent', None)
                _scene_rigid_gpu_pose = bool(
                    scene_gpu_mat is not None and is_animated and not _nd_is_skin
                )
                if _scene_rigid_gpu_pose:
                    vbo_wp = (0.0, 0.0, 0.0)
                    vbo_wo = (0.0, 0.0, 0.0, 1.0)
                    _scene_animated_node_draw_mat = _mat4_from_pos_quat_scale(
                        wp,
                        wo,
                        (1.0, 1.0, 1.0),
                    )
                # A retained scene actor's rigid mesh remains in node-local VBO
                # space.  Its current animated world transform is uploaded via
                # u_model above, so a new pose does not mutate vertex geometry.
                # Legacy single-model animation still bakes rigid transforms into
                # VBO data and therefore keeps its existing rebuild behavior.
                _dynamic_rigid_vbo = bool(is_animated and not _scene_rigid_gpu_pose)
                _rigid_vbo_mode_changed = False
                if not _nd_is_skin and node_id in self._mesh_cache:
                    _gm_existing = self._mesh_cache[node_id]
                    _rigid_vbo_mode_changed = (
                        bool(getattr(_gm_existing, 'scene_rigid_gpu_pose', False))
                        != _scene_rigid_gpu_pose
                    )
                _skin_vbo_mode_changed = False
                _skin_vbo_signature = (
                    _skin_vbo_signature_for_node(node)
                    if _nd_is_skin else None
                )
                if _nd_is_skin and node_id in self._mesh_cache:
                    _gm_existing = self._mesh_cache[node_id]
                    _skin_vbo_mode_changed = (
                        getattr(_gm_existing, 'skin_bind_transform', None)
                        != _skin_bind_transform
                        or getattr(_gm_existing, 'skin_vbo_signature', None)
                        != _skin_vbo_signature
                    )
                if (
                    _dynamic_rigid_vbo
                    or node_id not in self._mesh_cache
                    or _rigid_vbo_mode_changed
                    or _skin_vbo_mode_changed
                ):
                    if anim_pose is None and not _dynamic_rigid_vbo:
                        upload_budget = int(getattr(self, "max_new_mesh_uploads_per_frame", 0) or 0)
                        if upload_budget > 0 and _new_mesh_uploads_this_frame >= upload_budget:
                            self.deferred_mesh_uploads = True
                            return
                        _new_mesh_uploads_this_frame += 1
                    # FIX-SKIN-QBONE: Skin nodes now use a per-draw local palette
                    # in bone_map order.  Keep authored bone IDs local so slot k in
                    # vertex influences addresses qBone[k]/tBone[k] for this skin.
                    _bone_remap = None
                    _prebuilt_vbo = None
                    if anim_pose is None and not _dynamic_rigid_vbo and scene_gpu_mat is None:
                        _prebuilt_vbo = _prebuilt_static_gpu_mesh_data(
                            node, _cur_model_id, _skin_bind_transform
                        )
                    if _prebuilt_vbo is not None:
                        vdata, idx_arr = _prebuilt_vbo
                    else:
                        vdata, idx_arr = _build_vbo_data(node, vbo_wp, vbo_wo,
                                                         anim_pose_node=None,
                                                         is_module=_gpu_is_module,
                                                         bone_index_remap=_bone_remap,
                                                         apply_skin_node_transform_for_bind=_skin_bind_transform)
                    if vdata is None:
                        return
                    if node_id in self._mesh_cache:
                        self._mesh_cache[node_id].release()
                    gm = _GpuMesh()
                    gm.skin_bind_transform = _skin_bind_transform if _nd_is_skin else None
                    gm.skin_vbo_signature = _skin_vbo_signature
                    gm.scene_rigid_gpu_pose = (
                        _scene_rigid_gpu_pose if not _nd_is_skin else None
                    )
                    main_vdata, bone_id_vdata = _split_vbo_attributes_for_gpu(vdata)
                    gm.vbo = ctx.buffer(main_vdata.tobytes())
                    gm.bone_id_vbo = ctx.buffer(bone_id_vdata.tobytes())
                    gm.uploaded_vertex_count = int(len(vdata))
                    gm.first8_uv0_uploaded = _first_vbo_uv_pairs(vdata, 6)
                    gm.first8_uv1_uploaded = _first_vbo_uv_pairs(vdata, 8)
                    try:
                        uploaded_positions = np.asarray(vdata[:, 0:3], dtype=np.float64)
                        gm.uploaded_bounds = (
                            tuple(float(v) for v in uploaded_positions.min(axis=0)),
                            tuple(float(v) for v in uploaded_positions.max(axis=0)),
                        )
                        gm.uploaded_positions = [
                            [round(float(x), 6) for x in row[:3]]
                            for row in vdata[:, 0:3]
                        ]
                        gm.uploaded_bone_ids = [
                            [int(x) for x in row]
                            for row in bone_id_vdata
                        ]
                        gm.uploaded_weights = [
                            [round(float(x), 6) for x in row]
                            for row in vdata[:, 18:22]
                        ]
                        gm.uploaded_source_indices = [
                            int(x) for x in (
                                getattr(node, '_gr_last_vbo_source_indices', []) or []
                            )
                        ]
                    except Exception:
                        gm.uploaded_bounds = None
                        gm.uploaded_positions = []
                        gm.uploaded_bone_ids = []
                        gm.uploaded_weights = []
                        gm.uploaded_source_indices = []
                    gm.uv1_attribute_bound = (
                        getattr(vdata, 'ndim', 0) == 2
                        and getattr(vdata, 'shape', (0, 0))[1] >= 10
                    )
                    vertex_buffers = [
                        (gm.vbo, _VBO_MAIN_FORMAT, *_VBO_MAIN_ATTRS),
                        (gm.bone_id_vbo, _VBO_BONE_IDS_FORMAT, *_VBO_BONE_IDS_ATTRS),
                    ]
                    if idx_arr is not None:
                        gm.ibo = ctx.buffer(idx_arr.tobytes())
                        gm.vao = ctx.vertex_array(prog, vertex_buffers, gm.ibo)
                        gm.tri_count = len(idx_arr) // 3
                        gm.indexed = True
                    else:
                        gm.vao = ctx.vertex_array(prog, vertex_buffers)
                        gm.tri_count = len(vdata) // 3
                        gm.indexed = False
                    # ASCII/Kotor Tool MDLs use face_mats as per-face texture slots.
                    # Binary KotOR MDLs still use the single-diffuse + optional
                    # lightmap path, so only enable this for parser-tagged ASCII
                    # nodes whose secondary slots are not lightmaps.
                    if (
                        bool(getattr(node, 'imported_ascii', False))
                        and int(getattr(node, 'tex_count', 1) or 1) > 1
                        and not bool(getattr(node, 'has_lightmap', False))
                        and getattr(node, 'face_mats', None)
                        and getattr(node, 'texture_names', None)
                    ):
                        try:
                            slot_indices = {}
                            tex_names = list(getattr(node, 'texture_names', []) or [])
                            face_mats = list(getattr(node, 'face_mats', []) or [])
                            faces = list(getattr(node, 'faces', []) or [])
                            n_verts = len(getattr(node, 'vertices', []) or [])
                            row_face = 0
                            for face_i, face in enumerate(faces):
                                if len(face) < 3:
                                    continue
                                vi0, vi1, vi2 = int(face[0]), int(face[1]), int(face[2])
                                if vi0 >= n_verts or vi1 >= n_verts or vi2 >= n_verts:
                                    continue
                                slot = int(face_mats[face_i]) if face_i < len(face_mats) else 0
                                slot = max(0, min(slot, len(tex_names) - 1))
                                bucket = slot_indices.setdefault(slot, [])
                                if gm.indexed:
                                    bucket.extend([vi0, vi1, vi2])
                                else:
                                    base = row_face * 3
                                    bucket.extend([base, base + 1, base + 2])
                                row_face += 1
                            for slot, indices in slot_indices.items():
                                if not indices:
                                    continue
                                slot_ibo = ctx.buffer(np.asarray(indices, dtype=np.uint32).tobytes())
                                slot_vao = ctx.vertex_array(prog, vertex_buffers, slot_ibo)
                                gm.mat_slots[slot] = (slot_vao, slot_ibo, len(indices) // 3)
                        except Exception as exc:
                            log.debug("ASCII multi-texture split failed for %s: %s",
                                      getattr(node, 'name', ''), exc)

                    if not _dynamic_rigid_vbo:
                        self._mesh_cache[node_id] = gm
                else:
                    gm = self._mesh_cache[node_id]

                # Match the exact matrix uploaded for this draw.  Bounds stored
                # on _GpuMesh are in VBO space, so testing them with this matrix
                # is equivalent to the WGPU world-space bounds contract.
                draw_model_mat = model_mat
                if _scene_animated_node_draw_mat is not None:
                    draw_model_mat = _scene_animated_node_draw_mat
                elif _bas_skin_draw_mat is not None:
                    # ``_bas_skin_draw_mat`` is built from
                    # ``_get_world_transform(_bas_root_for_draw)``.  For a
                    # retained PIE actor that transform already walks through
                    # the runtime wrapper, including its authored placement and
                    # facing.  Multiplying by ``scene_gpu_mat`` here applied the
                    # wrapper a second time to BAS skin meshes only: the main
                    # detachable-head skin moved away while rigid eyes/teeth
                    # stayed on the animated headhook.
                    draw_model_mat = _bas_skin_draw_mat
                elif scene_gpu_mat is not None:
                    draw_model_mat = scene_gpu_mat

                # Animated skin vertices move in the shader and therefore keep
                # their fail-open behavior unless their entire expanded PIE
                # actor envelope was rejected above.  Rigid/static VBO bounds
                # are exact and can be safely rejected before material, texture,
                # palette, and uniform work.
                _rigid_outside_frustum = False
                _uploaded_bounds = getattr(gm, "uploaded_bounds", None)
                _actor_root = _pie_actor_root_for_node(node)
                _allow_rigid_actor_mesh_cull = bool(
                    _actor_root is not None
                    and getattr(_actor_root, "_gr_map_studio_pie_rigid_actor", False)
                )
                if (
                    override_vao is None
                    and _frustum_planes is not None
                    and (_actor_root is None or _allow_rigid_actor_mesh_cull)
                    and not (_nd_is_skin and _skin_anim_pose is not None)
                    and _uploaded_bounds is not None
                ):
                    if draw_model_mat is model_mat:
                        # Most room VBOs are already in world space.  Avoid a
                        # tiny NumPy matrix operation per room mesh per frame.
                        _rigid_outside_frustum = not bounds_intersects_frustum(
                            _uploaded_bounds,
                            _frustum_planes,
                        )
                    else:
                        _rigid_outside_frustum = self._transformed_bounds_outside_frustum(
                            _uploaded_bounds,
                            draw_model_mat,
                            _frustum_planes,
                        )
                if _rigid_outside_frustum:
                    _culled_rigid_meshes += 1
                    return

                # Use override VAO/tri_count if provided (per-material-slot draw)
                _use_vao = override_vao if override_vao is not None else gm.vao
                _use_tris = override_tri_count if override_vao is not None else gm.tri_count

                if _use_vao is None or _use_tris == 0:
                    return

                if override_vao is None and gm.mat_slots:
                    tex_names = list(getattr(node, 'texture_names', []) or [])
                    for slot, (slot_vao, _slot_ibo, slot_tris) in gm.mat_slots.items():
                        tex_override = tex_names[slot] if 0 <= slot < len(tex_names) else ''
                        _draw_node(
                            node,
                            tex_name_override=tex_override,
                            pass_name=pass_name,
                            override_vao=slot_vao,
                            override_tri_count=slot_tris,
                        )
                    return

                diff = getattr(node, 'diffuse', (1.0, 1.0, 1.0))
                diff = tuple(max(0.0, min(1.0, float(c))) for c in diff[:3])
                if 'u_wireframe_enabled' in _u:
                    _u['u_wireframe_enabled'].value = 1 if self._wireframe_pass else 0
                if 'u_wire_color' in _u:
                    _u['u_wire_color'].value = tuple(self.wire_color)
                if 'u_selected' in _u:
                    _u['u_selected'].value = 1 if self._is_node_selected_for_render(node) else 0
                if 'u_sprite_alpha_source' in _u:
                    _u['u_sprite_alpha_source'].value = float(self._sprite_alpha_source(node))
                if 'u_sprite_glow' in _u:
                    _u['u_sprite_glow'].value = max(0.0, min(4.0, float(self._sprite_glow(node))))
                _u['u_diffuse'].value = diff
                _u['u_selfillum'].value = tuple(
                    max(0.0, min(2.0, float(c))) for c in selfillum[:3])
                _u['u_alpha'].value = 1.0
                _u['u_node_alpha'].value = node_alpha

                # FIX-SHININESS: Per-node Phong shininess from ModelNode.shininess.
                # ASCII MDL 'shininess' command sets this; binary trimesh header
                # has a shininess field too.  Zero means no specular highlight —
                # clamp to a tiny positive so pow() in shader is well-defined.
                node_shininess = float(getattr(node, 'shininess', 0.0))
                if node_shininess > 0.0:
                    _u['u_shininess'].value = node_shininess
                else:
                    _u['u_shininess'].value = 20.0   # global default

                txi_blend = int(getattr(node, 'txi_blending', 0))
                if _is_blade_node and not self._has_sprite_material_override(node):
                    txi_blend = 1
                if txi_blend == 0 and _is_untextured_glow(node):
                    # Keep the draw's blend state in sync with the additive
                    # classification of untextured selfillum glow planes.
                    txi_blend = 1

                # FIX-ALPHATEST: Per-node punchthrough alpha-test threshold.
                # KotOR TPC header bytes [4-7] = float alpha_test_threshold.
                # Kotor.NET: node.TransparencyHint + TPC alpha_test float.
                # PyKotor gl/shader/texture.py: Texture.alpha_cutoff field.
                # When blending=punchthrough (2), use per-node txi_alpha_test
                # instead of the hardcoded 0.5 default set during program init.
                # Range: 0.0..1.0 (engine default is typically 0.5).
                #
                # FIX-TXI-PUNCH: KotOR creature/character DXT5 textures embed
                # TXI 'blending 2' in the TPC file, but the MDL binary parser
                # reads only node-level fields (not TPC-embedded TXI).  When the
                # node has txi_alpha_test > 0 and txi_blending == 0, promote to
                # punchthrough so hair/fur edges render correctly (cut out the
                # transparent DXT5 alpha instead of treating it as opaque).
                # This matches KotOR's engine behaviour for all creature meshes.
                txi_alpha_test = float(getattr(node, 'txi_alpha_test', 0.0))
                _transparency_hint = int(getattr(node, 'transparency_hint', 0))
                if txi_blend == 0 and txi_alpha_test > 0.0 and _transparency_hint > 0:
                    # Promote to punchthrough ONLY when the MDL node has
                    # transparency_hint > 0 (Aurora engine convention) combined with
                    # a TPC alpha_test_threshold.  Opaque nodes (transparency_hint=0)
                    # must NOT be promoted even if the TPC embeds an alpha_test value —
                    # this prevented dark-patch artifacts on bantha body meshes.
                    txi_blend = 2

                _u['u_blend_mode'].value = txi_blend
                if txi_blend == 2:
                    # Punchthrough: set alpha threshold uniform and disable GL blending.
                    # Shader discards fragments below the threshold — no GL blend needed.
                    _u['u_alpha_test'].value = max(0.0, min(1.0, txi_alpha_test if txi_alpha_test > 0.0 else 0.5))

                # TXI decal: surface blends over background using its own alpha
                txi_decal = 1 if bool(getattr(node, 'txi_decal', False)) else 0
                _u['u_decal'].value = txi_decal

                # TXI wateralpha: fractional alpha for water/glass (default 1.0)
                txi_wateralpha = float(getattr(node, 'txi_wateralpha', 1.0))
                _u['u_wateralpha'].value = txi_wateralpha

                # Determine GL blend state from TXI flags + per-node alpha
                is_semi_transparent = (
                    node_alpha < 0.999
                    or txi_wateralpha < 0.999
                    or txi_decal
                )
                blend_enabled = False
                if txi_blend == 1:
                    # Additive blend: src=ONE, dst=ONE
                    self._blend_submission.apply(
                        ctx,
                        enabled=True,
                        equation=moderngl.FUNC_ADD,
                        func=(moderngl.ONE, moderngl.ONE),
                    )
                    blend_enabled = True
                elif txi_blend == 2:
                    self._blend_submission.apply(ctx, enabled=False)
                elif txi_blend == 3:
                    # v7.1 FIX-GLMAX (Finding 5.6 — reone context.cpp cross-ref):
                    # GL_MAX blend equation for lighten mode effects.
                    # reone context.cpp line 407: BlendMode::Lighten uses
                    # glBlendEquationSeparate(GL_MAX, GL_FUNC_ADD)
                    # KotOR uses this for some particle effects and self-illumination
                    # overlays where the brightest pixel should win.
                    try:
                        self._blend_submission.apply(
                            ctx,
                            enabled=True,
                            equation=moderngl.MAX,
                            func=(moderngl.ONE, moderngl.ONE),
                        )
                    except Exception:
                        # Fallback: GL_MAX not available on this driver
                        self._blend_submission.reset()
                        self._blend_submission.apply(
                            ctx,
                            enabled=True,
                            equation=moderngl.FUNC_ADD,
                            func=(moderngl.ONE, moderngl.ONE),
                        )
                    blend_enabled = True
                elif is_semi_transparent or txi_decal:
                    # Decal / wateralpha / per-node transparency
                    self._blend_submission.apply(
                        ctx,
                        enabled=True,
                        equation=moderngl.FUNC_ADD,
                        func=(moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA),
                    )
                    blend_enabled = True
                else:
                    self._blend_submission.apply(ctx, enabled=False)

                uv_scroll = (0.0, 0.0)
                if bool(getattr(node, 'animate_uv', False)):
                    dx = float(getattr(node, 'uv_dir_x', 0.0))
                    dy = float(getattr(node, 'uv_dir_y', 0.0))
                    jitter = float(getattr(node, 'uv_jitter', 0.0))
                    jspd   = float(getattr(node, 'uv_jitter_speed', 0.0))
                    su = dx * anim_time
                    sv = dy * anim_time
                    if jitter != 0.0 and jspd > 0.0:
                        j = jitter * math.sin(anim_time * jspd * 2.0 * math.pi)
                        su += j; sv += j
                    uv_scroll = (su, sv)
                _u['u_uv_scroll'].value = uv_scroll

                rot_tex = 1.0 if bool(getattr(node, 'rotate_texture', False)) else 0.0
                _u['u_rotate_tex'].value = rot_tex

                # v7.2 FIX-DANGLY (Finding 5.10 — reone v_model.glsl cross-ref):
                # Enable GPU dangly mesh vertex animation for DANGLY node type.
                # Dangly meshes simulate cloth/hair/tentacle physics with per-vertex
                # constraint weights. The vertex shader applies wind-like displacement
                # modulated by the constraint (0=free, 1=fixed) stored in vertex alpha.
                # Reference: reone v_model.glsl line 58; KotOR.js OdysseyModel3D.ts.
                _is_dangly = bool(getattr(node, 'is_dangly', False))
                if _is_dangly:
                    _u['u_dangly_enabled'].value = 1.0
                    _u['u_dangly_displacement'].value = float(
                        getattr(node, 'dangly_displacement', 0.5))
                    _u['u_dangly_time'].value = anim_time
                else:
                    _u['u_dangly_enabled'].value = 0.0

                # FIX-SABER-PROXY: _build_vbo_data has already normalized every
                # blade layer to its authored, fully extended local bounds.  The
                # old scalar gl_VertexID path displaced that world-space proxy on
                # world Z, shearing stock NODE_SABER triangles into white spokes.
                # Keep deformation disabled until the complete KOTOR vector
                # hdist/vdist ignition contract is implemented.
                _u['u_saber_enabled'].value = 0.0
                _u['u_saber_displacement'].value = 0.0
                _u['u_saber_length'].value = 1.0

                # FIX-FLIPBOOK: TXI proceduretype=cycle sprite-sheet animation.
                # When txi_proceduretype == 'cycle', the texture is a grid of
                # numx × numy frames played at txi_fps frames/second.
                # We compute the current frame from anim_time and pass a UV tile
                # offset and tile size to the vertex shader.
                # Reference: KotOR TXI spec proceduretype/numx/numy/fps.
                txi_proc  = str(getattr(node, 'txi_proceduretype', '')).lower()
                txi_numx  = int(getattr(node, 'txi_numx', 0))
                txi_numy  = int(getattr(node, 'txi_numy', 0))
                txi_fps   = float(getattr(node, 'txi_fps', 0.0))
                _u['u_uv_v_flip'].value = 1.0 if bool(getattr(node, 'uv_v_flip', True)) else 0.0

                # v7.1 FIX-PROCTYPE (Finding 1.6 — KotOR.js TXI.ts cross-ref):
                # Set u_proc_type uniform for water/ring UV distortion in fragment shader.
                # proceduretype=water → sinusoidal UV distortion (water surfaces, lava)
                # proceduretype=ringtexdistort → radial ring distortion (energy fields)
                # proceduretype=random → random UV offset per frame (sparkle/shimmer)
                # Cross-ref: KotOR.js TXI.ts lines 170-186; reone shader pipeline.
                _proc_type_map = {'cycle': 1, 'water': 2, 'random': 3, 'ringtexdistort': 4}
                _proc_int = _proc_type_map.get(txi_proc, 0)
                _u['u_proc_type'].value = _proc_int
                _u['u_water_time'].value = anim_time if _proc_int in (2, 3, 4) else 0.0

                if txi_proc == 'cycle' and txi_numx > 0 and txi_numy > 0 and txi_fps > 0.0:
                    total_frames = txi_numx * txi_numy
                    frame_idx    = int(anim_time * txi_fps) % total_frames
                    col = frame_idx % txi_numx
                    row = frame_idx // txi_numx
                    tile_w = 1.0 / txi_numx
                    tile_h = 1.0 / txi_numy
                    # V origin: row 0 is bottom in OpenGL; KotOR cycle sheets
                    # are stored top-to-bottom so we flip: row 0 → top → high V
                    flip_row = (txi_numy - 1 - row)
                    _u['u_flipbook_off'].value  = (col * tile_w, flip_row * tile_h)
                    _u['u_flipbook_size'].value = (tile_w, tile_h)
                else:
                    _u['u_flipbook_off'].value  = (0.0, 0.0)
                    _u['u_flipbook_size'].value = (0.0, 0.0)

                # ── Texture binding ────────────────────────────────────────
                # FIX-MULTITEX: allow caller to override the texture name for
                # multi-texture batching (one draw call per material slot).
                tex_name = (tex_name_override or
                            str(getattr(node, 'texture', '')).strip().lower())
                if tex_name in ('null', '', 'none'):
                    tex_name = ''
                diff_img = textures.get(tex_name) if tex_name else None
                if _is_blade_node and should_use_procedural_lightsaber_blade_texture(
                    node,
                    texture_missing=diff_img is None,
                ) and _PIL:
                    try:
                        blade_key = lightsaber_blade_texture_cache_key(node)
                        diff_img = self._procedural_lightsaber_textures.get(blade_key)
                        if diff_img is None:
                            blade_w, blade_h, blade_rgba = lightsaber_blade_procedural_rgba8(node)
                            diff_img = Image.frombytes("RGBA", (blade_w, blade_h), blade_rgba)
                            setattr(diff_img, "_gr_gpu_uv_v_flip", True)
                            self._procedural_lightsaber_textures[blade_key] = diff_img
                    except Exception as exc:
                        log.debug("lightsaber procedural blade texture failed: %s", exc)
                gl_diff = self._tex_cache.get(diff_img) if diff_img else None
                if _is_blade_node and gl_diff is not None:
                    # Trilinear blade mips collapse the 64x256 aura to blocky
                    # 1x4 samples at oblique angles.  Sample the authored mask
                    # directly and let its black edge provide the smooth fade.
                    _configure_lightsaber_blade_sampler(gl_diff)
                _u['u_uv_v_flip'].value = _effective_diffuse_uv_v_flip(node, diff_img)

                _texture_allowed = bool(self.show_texture and self.show_diffuse_map)
                _detail_texture_allowed = bool(self.show_texture and _render_mode_int == 0)
                if _texture_allowed and gl_diff:
                    # FIX-TEXWRAP: Apply per-node TXI clamp mode (GL_CLAMP_TO_EDGE)
                    # vs. default GL_REPEAT before each draw call.
                    # txi_clamp_s=True → GL_CLAMP_TO_EDGE on U axis (no horizontal tile)
                    # txi_clamp_t=True → GL_CLAMP_TO_EDGE on V axis (no vertical tile)
                    # Default repeat_x=True/repeat_y=True set in _upload; we override
                    # here for nodes that require clamping (head decals, UI overlays).
                    _node_clamp_s = bool(getattr(node, 'txi_clamp_s', False))
                    _node_clamp_t = bool(getattr(node, 'txi_clamp_t', False))
                    if _is_blade_node:
                        # Blade proxies occupy one authored 0..1 tile.  Clamp
                        # both axes so oblique samples cannot wrap the bright
                        # core across the transparent base, tip, or card edge.
                        _node_clamp_s = True
                        _node_clamp_t = True
                    if not (_node_clamp_s and _node_clamp_t) and _should_auto_clamp_diffuse(
                        node,
                        is_module=_gpu_is_module,
                    ):
                        # Match the CPU renderer for custom character atlases.
                        # Override MDLs like n_mandalorian01-03 use 0..1 body/helmet
                        # sheets without TXI clamp flags; GL_REPEAT samples the
                        # opposite atlas edge along armor-panel UV borders.
                        _node_clamp_s = True
                        _node_clamp_t = True
                    _set_texture_sampler_exact(
                        gl_diff,
                        repeat_x=not _node_clamp_s,
                        repeat_y=not _node_clamp_t,
                    )
                    _bind_texture_exact(gl_diff, 0)
                    _u['u_tex'].value = 0
                    _u['u_has_tex'].value = 1
                else:
                    if self._white_tex is None:
                        self._white_tex = ctx.texture((1, 1), 4,
                                                       bytes([255, 255, 255, 255]))
                    _bind_texture_exact(self._white_tex, 0)
                    _u['u_tex'].value = 0
                    _u['u_has_tex'].value = 0

                has_lm_flag = bool(getattr(node, 'has_lightmap', False) or getattr(node, '_gr_baked_lightmap_preview_name', ''))
                lm_name     = str(getattr(node, '_gr_baked_lightmap_preview_name', '') or getattr(node, 'lightmap', '')).strip().lower()
                uvs_lm      = getattr(node, 'uvs_lm', [])
                # FIX-LMROLE-V2 (Phase D10): Infer lightmap when texture_2 is present
                # with valid lightmap UVs.  The KotOR texture model is: texture_1 =
                # diffuse, texture_2 = lightmap (when present).  face_mats is NOT used
                # for this determination — it is a walk-mesh surface indicator.
                # Accept lightmap whenever: lm_name is set, uvs_lm has data, and
                # tex_count >= 2.  The loader's FIX-LMROLE normally promotes
                # has_lightmap, but this guard catches alternative load paths.
                if (not has_lm_flag
                        and lm_name
                        and len(uvs_lm) > 0
                        and int(getattr(node, 'tex_count', 1)) >= 2):
                    has_lm_flag = True
                lm_img      = textures.get(lm_name) if (lm_name and has_lm_flag
                                                         and len(uvs_lm) > 0) else None
                gl_lm = self._tex_cache.get(lm_img) if lm_img else None
                if _detail_texture_allowed and bool(self.show_lightmap_map) and gl_lm:
                    # FIX-LMWRAP: Lightmap textures must use CLAMP_TO_EDGE
                    # (not GL_REPEAT) because lightmap UVs are always in [0,1]
                    # and wrap-around causes visible seam lines at texel boundaries.
                    # Additionally, lightmaps are small (8x8 to 64x64) and their
                    # mipmap chain can over-blur the lightmap when magnified across
                    # large surfaces.  Use GL_LINEAR (no mipmap) for min filter.
                    # Cross-ref: KotOR.js ShaderOdysseyModel.ts lightMap sampling;
                    # xoreos model_kotor.cpp — lightmap uses CLAMP_TO_EDGE wrap;
                    # KotorBlender — lightmap UV is always in [0,1] range.
                    _set_texture_sampler_exact(
                        gl_lm,
                        repeat_x=False,
                        repeat_y=False,
                        filter_value=(moderngl.LINEAR, moderngl.LINEAR),
                    )
                    _bind_texture_exact(gl_lm, 1)
                    _u['u_lm_tex'].value = 1
                    _u['u_has_lm'].value = 1
                    _u['u_lm_shade'].value = 1 if _gpu_is_module else 0
                else:
                    _u['u_has_lm'].value = 0
                    _u['u_lm_shade'].value = 0

                # BUG-ENVMAP FIX: Bind environment map texture to unit 2.
                # The TXI 'envmaptexture' field names the environment map texture.
                # Diffuse alpha = blend weight between surface and env map.
                # This fixes the see-through appearance on metallic/glossy surfaces
                # (droids, weapons, some creature hides) where alpha was incorrectly
                # treated as transparency.
                #
                # FIX-ENVFB: When the env-map texture name is set but the image
                # is not in the texture dict (partial load), bind a neutral grey
                # 1×1 fallback texture instead of nothing.  Without this fallback
                # the surface would be treated as transparent (u_has_env=0, diffuse
                # alpha applied as transparency), which is wrong — the surface
                # should remain opaque with a slight metallic tint.
                env_name = str(getattr(node, 'txi_envmaptexture', '')).strip().lower()
                # FIX-ENVDEFAULT: A bare TGA/PNG drop names no cube map, so an
                # opaque metallic skin whose diffuse alpha is a reflectivity mask
                # would render as its (deliberately dark) diffuse with no sheen —
                # the "loads too dark" report.  KOTOR drives that sheen from the
                # alpha mask plus a default cube map (appearance.2da envmap=
                # DEFAULT); mirror it here with the built-in metallic sphere-map.
                # Gated to genuinely opaque, non-decal, non-punch-through nodes so
                # transparency/cutout alphas are never turned into reflections.
                _env_default = False
                _raw_env_node_alpha = getattr(node, 'alpha', None)
                _env_node_alpha = float(1.0 if _raw_env_node_alpha is None else _raw_env_node_alpha)
                if (not env_name
                        and _detail_texture_allowed
                        and bool(self.show_environment_map)
                        and txi_decal == 0
                        and txi_blend == 0
                        and int(getattr(node, 'transparency_hint', 0) or 0) == 0
                        and _env_node_alpha >= 0.999
                        and _diffuse_is_reflectivity_mask(diff_img)):
                    _env_default = True
                if _detail_texture_allowed and bool(self.show_environment_map) and (env_name or _env_default):
                    if _env_default:
                        gl_env = self._default_env_texture(ctx)
                    else:
                        env_img  = textures.get(env_name)
                        gl_env   = self._tex_cache.get(env_img) if env_img else None
                        if gl_env is None:
                            # Use grey fallback: neutral env tint keeps surface opaque
                            if self._grey_env_tex is None:
                                # 128,128,128,255 → 50% grey → neutral metallic tint
                                self._grey_env_tex = ctx.texture((1, 1), 4,
                                                                  bytes([128, 128, 128, 255]))
                            gl_env = self._grey_env_tex
                    _bind_texture_exact(gl_env, 2)
                    _u['u_env_tex'].value = 2
                    _u['u_has_env'].value = 1
                else:
                    _u['u_has_env'].value = 0

                # FIX-SPECMAP: Bind TXI specularcolour map to texture unit 3.
                # When the TXI 'specularcolour' keyword names a texture, that
                # texture's luminance is used as a per-texel specular intensity
                # multiplier in the fragment shader — giving armour and metallic
                # surfaces per-pixel gloss rather than a flat u_specular scalar.
                # Reference: Kotor.NET KotorModelLoader.cs specular texture slot;
                #            KotOR.js ShaderOdysseyModel.ts specularColor;
                #            xoreos modelnode.cpp _specularColour field.
                spec_name = str(getattr(node, 'txi_specularcolour', '')).strip().lower()
                if _detail_texture_allowed and bool(self.show_specular_map) and spec_name:
                    spec_img = textures.get(spec_name)
                    gl_spec  = self._tex_cache.get(spec_img) if spec_img else None
                    if gl_spec is not None:
                        _bind_texture_exact(gl_spec, 3)
                        _u['u_spec_tex'].value = 3
                        _u['u_has_spec'].value = 1
                    else:
                        _u['u_has_spec'].value = 0
                else:
                    _u['u_has_spec'].value = 0

                bump_name = str(
                    getattr(node, 'txi_bumpmaptexture', '')
                    or getattr(node, 'bump_map', '')
                    or ''
                ).strip().lower()
                if _detail_texture_allowed and bool(self.show_normal_map) and bump_name:
                    bump_img = textures.get(bump_name)
                    gl_bump = self._tex_cache.get(bump_img) if bump_img else None
                    if gl_bump is not None:
                        _bind_texture_exact(gl_bump, 4)
                        _u['u_bump_tex'].value = 4
                        _u['u_has_bump'].value = 1
                    else:
                        _u['u_has_bump'].value = 0
                else:
                    _u['u_has_bump'].value = 0

                # v7.1/7.2: Build feature bitmask for this node (Finding 5.2)
                # FIX-FEATMASK-ORDER: Moved AFTER texture binding so that
                # gl_diff, diff_img, gl_lm, env_name, spec_name are all
                # defined before use.  Previously caused UnboundLocalError
                # on the first draw call, silently falling back to CPU.
                _feat_mask = 0
                if gl_diff and diff_img: _feat_mask |= (1 << 0)   # FEAT_TEXTURE
                if _detail_texture_allowed and bool(self.show_lightmap_map) and gl_lm: _feat_mask |= (1 << 1)   # FEAT_LIGHTMAP
                if _detail_texture_allowed and bool(self.show_environment_map) and (env_name or _env_default): _feat_mask |= (1 << 2)   # FEAT_ENVMAP
                if _detail_texture_allowed and bool(self.show_specular_map) and spec_name: _feat_mask |= (1 << 3)   # FEAT_SPECMAP
                if _detail_texture_allowed and bool(self.show_normal_map) and bump_name: _feat_mask |= (1 << 4)   # FEAT_BUMPMAP
                if _is_dangly:          _feat_mask |= (1 << 6)   # FEAT_DANGLY
                if _is_blade_node:        _feat_mask |= (1 << 7)  # FEAT_SABER emission
                if txi_decal:           _feat_mask |= (1 << 11)  # FEAT_DECAL
                if txi_blend == 2:      _feat_mask |= (1 << 12)  # FEAT_PUNCHTHRU
                if txi_blend == 1:      _feat_mask |= (1 << 13)  # FEAT_ADDITIVE
                # Phase A: Set FEAT_SKIN bit only when this draw is actually
                # palette-skinned. BAS attachment palettes are converted to
                # attachment-root local space before upload, then the socket/root
                # draw matrix places the skinned result on the body.
                if _skin_can_lbs:
                    _feat_mask |= (1 << 10)  # FEAT_SKIN
                _u['u_features'].value = _feat_mask

                # ── Phase A: GPU Skinning — enable/disable per draw call ──────
                # For skin nodes with a valid bone palette, enable LBS in the shader.
                # For non-skin nodes, ensure skinning is disabled (pass-through).
                #
                # FIX-SKIN-BINDPOSE: ONLY enable GPU skinning when there is an
                # active animation pose.  KotOR skin mesh vertices are stored in
                # world/bind-pose space (pre-transformed).  When anim_pose is
                # None, compute_palette() produces M = I × inv_bind = inv_bind,
                # which un-transforms verts from bind-pose to bone-local space,
                # causing geometry stretching/explosion.  The correct bind-pose
                # rendering path is to skip GPU skinning entirely (u_skin_enabled=0)
                # so the shader passes in_pos through unchanged.
                #
                # GPU skinning should only activate when anim_pose provides
                # world-pose matrices that differ from the bind pose.
                _skin_local_bone_count = 0
                if (_skin_can_lbs
                        and 'u_skin_enabled' in _u
                        and 'u_bone_count' in _u):
                    try:
                        (
                            _skin_local_bone_count,
                            _skin_palette_bytes,
                            _skin_palette_cached,
                        ) = self._skin_palette_bytes_for_draw(
                            scope_key=_skin_palette_scope_key,
                            skin_node=node,
                            uploader=_skin_uploader,
                            anim_pose=_skin_anim_pose,
                            anim_base_pose=_skin_anim_base_pose,
                            skin_signature=_skin_vbo_signature,
                        )
                        if 'u_bones' in _u and _skin_local_bone_count > 0:
                            _u['u_bones'].write(_skin_palette_bytes)
                        _u['u_skin_enabled'].value = 1 if _skin_local_bone_count > 0 else 0
                        _u['u_bone_count'].value = _skin_local_bone_count
                        if (
                            bool(getattr(node, "_gr_bound_to_kotor_skeleton", False))
                            and _skin_local_bone_count > 0
                        ):
                            _palette_key = (_cur_model_id, node_id)
                            if _palette_key not in self._skin_preview_palette_logged:
                                self._skin_preview_palette_logged.add(_palette_key)
                                log.info(
                                    "GPU-SKINNING: Character Builder palette node=%s "
                                    "u_skin=%s u_bones=%d formula=%s inv_bind=%s "
                                    "anim_nodes=%d base_nodes=%d skin=%s",
                                    getattr(node, "name", "?"),
                                    1 if _skin_local_bone_count > 0 else 0,
                                    _skin_local_bone_count,
                                    str(getattr(_skin_uploader, "_skin_palette_formula", "") or ""),
                                    str(getattr(_skin_uploader, "_skin_inverse_bind_source", "") or ""),
                                    len(getattr(_skin_anim_pose, "nodes", {}) or {}),
                                    len(getattr(_skin_anim_base_pose, "nodes", {}) or {}) if _skin_anim_base_pose is not None else 0,
                                    _skin_influence_summary_for_log(node),
                                )
                    except Exception as e:
                        log.debug(f"GPU-SKINNING: per-skin qBone/tBone palette upload failed: {e}")
                        _u['u_skin_enabled'].value = 0
                        _u['u_bone_count'].value = 0
                else:
                    if 'u_skin_enabled' in _u:
                        _u['u_skin_enabled'].value = 0
                    if 'u_bone_count' in _u:
                        _u['u_bone_count'].value = 0

                if self._gl_state_trace_path:
                    _append_gl_state_trace(
                        self._gl_state_trace_path,
                        _build_gl_state_trace_record(
                            ctx=ctx,
                            prog=prog,
                            node=node,
                            pass_name=pass_name,
                            tri_count=_use_tris,
                            blend_enabled=blend_enabled,
                            tex_name=tex_name,
                            lm_name=lm_name,
                            env_name=env_name,
                            spec_name=spec_name,
                            feature_mask=_feat_mask,
                            uniforms=_u,
                        ),
                    )

                if self._lm_data_dump_path and has_lm_flag:
                    _lm_key = (id(model), str(getattr(node, 'name', '') or '').lower())
                    if _lm_key not in self._lm_data_dump_seen:
                        self._lm_data_dump_seen.add(_lm_key)
                        _append_jsonl_record(
                            self._lm_data_dump_path,
                            _build_lm_data_dump_record(
                                ctx=ctx,
                                prog=prog,
                                node=node,
                                pass_name=pass_name,
                                gm=gm,
                                has_lm_flag=has_lm_flag,
                                lightmap_bound=gl_lm is not None,
                                lm_img=lm_img,
                                lm_name=lm_name,
                                uniforms=_u,
                            ),
                            'Lightmap data dump',
                        )

                if self._skin_dump_path and _nd_is_skin:
                    _skin_key = (id(model), str(getattr(node, 'name', '') or '').lower())
                    if _skin_key not in self._skin_dump_seen:
                        self._skin_dump_seen.add(_skin_key)
                        _append_jsonl_record(
                            self._skin_dump_path,
                            _build_skin_dump_record(
                                model=model,
                                node=node,
                                pass_name=pass_name,
                                uploader=_skin_uploader,
                                bone_remap=_bone_remap,
                                uniforms=_u,
                                gm=gm,
                                anim_pose=anim_pose,
                                anim_base_pose=anim_base_pose,
                                anim_time=anim_time,
                            ),
                            'Skin parity dump',
                        )

                if (
                    scene_gpu_mat is not None
                    or _bas_skin_draw_mat is not None
                    or _scene_animated_node_draw_mat is not None
                ):
                    _u['u_model'].write(_mat4_tobytes(draw_model_mat))
                    _u['u_normal_mat'].write(_mat3_normal(draw_model_mat).T.astype(np.float32).tobytes())
                else:
                    _u['u_model'].write(_mat4_tobytes(model_mat))
                    _u['u_normal_mat'].write(normal_mat.T.astype(np.float32).tobytes())

                _blade_cull_disabled = False
                if _is_blade_node and self.cull_faces and not self.show_wireframe:
                    ctx.disable(moderngl.CULL_FACE)
                    _blade_cull_disabled = True
                _use_vao.render(moderngl.TRIANGLES)
                _drawn_node_names.append(
                    f"{str(pass_name or 'opaque')}:{str(getattr(node, 'name', '') or '<unnamed>')}"
                )
                if _blade_cull_disabled:
                    ctx.enable(moderngl.CULL_FACE)
                total_tris += _use_tris
                _draw_call_count += 1

            # Helper: draw a node with correct KotOR texture routing.
            def _draw_node_multitex(node, pass_name: str = 'opaque'):
                """Draw a node with correct KotOR texture routing.

                FIX-KILL-FACEMATS (Phase D10): The previous implementation split
                multi-texture nodes by face_mats[], treating the walk-mesh surface
                indicator as a texture selector.  This is WRONG and has been removed.

                The correct KotOR texture model (per xoreos, KotOR.js, KotorBlender):
                  - One diffuse texture (texture_1) on UV0
                  - Optional one lightmap (texture_2) on UV1
                  - Composite: diffuse * lightmap * overbright
                  - face_mats is a walk-mesh surface type, NOT a texture selector

                For ALL nodes regardless of tex_count, we draw ONCE with:
                  - Diffuse texture bound to sampler 0
                  - Lightmap (if has_lightmap) bound to sampler 1
                  - Lightmap compositing handled by fragment shader

                Reference: xoreos model_kotor.cpp — single diffuse per mesh;
                KotOR.js OdysseyModelNodeMesh.ts — single texture per mesh;
                KotorBlender reader.py — material field is surface type.
                """
                # Always draw once with primary diffuse + optional lightmap.
                # _draw_node already handles lightmap binding via u_lm_tex/u_has_lm
                # when has_lightmap is True and lightmap UVs are present.
                _draw_node(node, pass_name=pass_name)

            # ── Pass 1: Opaque geometry (depth write ON, no blending) ─────────────
            # Solid, fully-opaque surfaces with no alpha-test.
            # Cross-ref: Hayes (2025) §6.3; reone: GL_DEPTH_TEST + depth write ON.
            _set_depth_write(ctx, True)
            self._begin_draw_pass(ctx, blend_enabled=False)
            for node in opaque_nodes:
                _draw_node_multitex(node, pass_name='opaque')

            # ── Pass 2: Alpha-cutout geometry (depth write ON, shader discard) ────
            # Punchthrough / alpha-test surfaces: hair cards, fur edges, eye cutouts,
            # grates, foliage.  Depth write stays ON so cutout geometry properly
            # occludes what's behind it; the fragment shader discards pixels below
            # the alpha threshold (u_alpha_test, default 0.5).
            # Cross-ref: Hayes (2025) §8.2 alpha testing; Gregory (2024) §10.6.
            # ctx.depth_mask stays True; no blending needed for cutout.
            self._begin_draw_pass(ctx, blend_enabled=False)
            for node in cutout_nodes:
                _draw_node_multitex(node, pass_name='cutout')

            # ── Pass 3: Transparent/additive geometry (depth write OFF) ───────────
            # BUG-ALPHA FIX: Sort transparent nodes back-to-front by their centroid
            # distance from the camera eye before drawing.  Without sorting, overlapping
            # transparent surfaces (glass visors, alpha-blended hair) produce incorrect
            # compositing because the painter's algorithm requires back-to-front order.
            if transparent_nodes:
                self._begin_draw_pass(ctx, blend_enabled=None)
                eye_arr = np.array(eye, dtype=np.float64)

                def _node_sort_depth(nd):
                    """Return squared distance from camera to node centroid (descending sort = back first).

                    Prefers mesh_average_point (the exact Aurora-engine centroid stored in the
                    TrimeshHeader) over the bounding-box midpoint for accurate depth sorting.
                    Kotor.NET TrimeshHeader.AveragePoint; xoreos _averagePoint.
                    """
                    _wp2, world_mat = _get_world_transform(nd)
                    try:
                        # Prefer the pre-computed mesh centroid (AveragePoint from binary MDL).
                        avg_pt = getattr(nd, 'mesh_average_point', None)
                        if avg_pt and (avg_pt[0] != 0.0 or avg_pt[1] != 0.0 or avg_pt[2] != 0.0):
                            # Transform mesh-local average point to world space
                            if world_mat is not None:
                                ax, ay, az = float(avg_pt[0]), float(avg_pt[1]), float(avg_pt[2])
                                m = world_mat
                                wx = m[0]*ax + m[4]*ay + m[8]*az  + m[12]
                                wy = m[1]*ax + m[5]*ay + m[9]*az  + m[13]
                                wz = m[2]*ax + m[6]*ay + m[10]*az + m[14]
                                c = np.array([wx, wy, wz], dtype=np.float64)
                            else:
                                c = np.array(avg_pt, dtype=np.float64)
                        else:
                            # Fall back to world-transform position (node origin)
                            c = np.array(_wp2, dtype=np.float64)
                        d = c - eye_arr
                        return float(np.dot(d, d))
                    except Exception:
                        return 0.0

                # Sort farthest-first (painter's algorithm: draw back before front)
                transparent_nodes_sorted = sorted(transparent_nodes,
                                                  key=_node_sort_depth, reverse=True)
                _set_depth_write(ctx, False)
                for node in transparent_nodes_sorted:
                    _draw_node_multitex(node, pass_name='transparent')
                # Restore depth writes after transparent pass
                _set_depth_write(ctx, True)
                self._blend_submission.apply(ctx, enabled=False)

            # ── Pass 4: Emitter particles (depth test ON, depth write OFF) ────
            # KOTOR emitter nodes carry no geometry; their particles are
            # simulated on the CPU (src.core.particles) and drawn as billboard
            # batches after all scene geometry so blending composites over the
            # already-resolved surfaces.
            _particle_count = 0
            _particle_draw_calls = 0
            if self.show_particles and not self._wireframe_pass:
                try:
                    _particle_count, _particle_draw_calls = self._update_and_draw_particles(
                        ctx, model, textures, mvp, eye, target, up,
                        _get_world_transform, anim_pose, anim_time, anim_name,
                    )
                except Exception as _particle_exc:
                    log.debug("Particle pass failed: %s", _particle_exc, exc_info=True)
            self.perf['particles'] = _particle_count
            self.perf['particle_draw_calls'] = _particle_draw_calls

            if self.show_solid and self.show_wireframe:
                try:
                    self._wireframe_pass = True
                    ctx.wireframe = True
                    _set_depth_write(ctx, False)
                    self._begin_draw_pass(ctx, blend_enabled=False)
                    for node in opaque_nodes:
                        _draw_node_multitex(node, pass_name='opaque')
                    for node in cutout_nodes:
                        _draw_node_multitex(node, pass_name='cutout')
                    for node in transparent_nodes:
                        _draw_node_multitex(node, pass_name='transparent')
                finally:
                    self._wireframe_pass = False
                    _set_depth_write(ctx, True)
            try:
                ctx.wireframe = False
            except Exception:
                pass
            self._wireframe_pass = False
            self._draw_light_gizmos(ctx, mvp, nodes, _get_world_transform)

            self.perf['draw_ms'] = (time.perf_counter() - t_draw) * 1000
            self.perf['tri_count'] = total_tris
            self.perf['draw_calls'] = _draw_call_count
            self.perf['drawn_node_names'] = tuple(_drawn_node_names)
            self.perf['culled_actor_meshes'] = _culled_actor_meshes
            self.perf['culled_meshes'] = _culled_actor_meshes + _culled_rigid_meshes
            self.perf['visible_meshes'] = max(
                0,
                _classified_mesh_count - _culled_actor_meshes - _culled_rigid_meshes,
            )
            for _submission_key, _submission_value in self._submission_stats.items():
                self.perf[_submission_key] = int(_submission_value)

            # ── MSAA Resolve (if multisampled FBO) ───────────────────────
            # FIX-MSAA: Blit the 4x MSAA renderbuffer to the single-sample
            # resolve FBO before reading pixels. This eliminates the single-pixel
            # magenta/neon-green fringe artifacts at triangle edges.
            # PERF-FBO: Only resolve when we actually used the MSAA FBO (not
            # the simple interactive FBO).
            if (_use_msaa and getattr(self, '_fbo_msaa', False)
                    and getattr(self, '_fbo_resolve', None)):
                try:
                    ctx.copy_framebuffer(self._fbo_resolve, self._fbo)
                    read_fbo = self._fbo_resolve
                except Exception:
                    read_fbo = fbo
            else:
                read_fbo = fbo

            # ── Bloom post-process (retail-style glow) ────────────────────
            # Runs on the resolved/simple framebuffer, whose color attachment
            # is a sampleable texture.  MSAA-fallback frames (renderbuffer
            # only) skip bloom rather than fail.
            if self.bloom_enabled and _render_mode_int == 0:
                bloom_tex = None
                if read_fbo is self._fbo_resolve:
                    bloom_tex = self._fbo_resolve_tex
                elif read_fbo is self._fbo_simple:
                    bloom_tex = self._fbo_simple_tex
                if bloom_tex is not None:
                    t_bloom = time.perf_counter()
                    try:
                        if self._bloom_pass is None:
                            from src.adapters.rendering.moderngl_bloom import ModernGLBloomPass

                            self._bloom_pass = ModernGLBloomPass()
                        self._bloom_pass.apply(
                            ctx,
                            self._blend_submission,
                            read_fbo,
                            bloom_tex,
                            W,
                            H,
                            threshold=float(self.bloom_threshold),
                            strength=float(self.bloom_strength),
                        )
                    except Exception as _bloom_exc:
                        log.debug("Bloom pass failed: %s", _bloom_exc, exc_info=True)
                    self.perf['bloom_ms'] = (time.perf_counter() - t_bloom) * 1000

            # ── Read back framebuffer to PIL Image ────────────────────
            # PERF-FIX: Use dtype='f1' (RGBA8 UNorm) for fbo.read().
            # On llvmpipe EGL with a renderbuffer color attachment:
            #   dtype='u1' → reads raw bytes (all zeros, WRONG)
            #   dtype='f1' → reads RGBA8 UNorm bytes (correct uint8 values)
            #   dtype='f4' → reads float32 (correct, but 7ms not 0.4ms)
            # dtype='f1' + np.frombuffer as uint8 gives RGBA8 at ~0.4ms.
            # PERF-FIX 2: Use numpy array reversal for vertical flip instead of
            # PIL.transpose(FLIP_TOP_BOTTOM) which copies the entire image (~275ms).
            # NumPy array reversal with .copy() costs ~0.5ms.
            # ── PERF-READBACK: Optimized framebuffer readback ───────────
            # Since we clear the FBO with alpha=1.0 (opaque BG), most pixels
            # have alpha=255.  The expensive alpha-composite step (~19ms at
            # 800x600) can be skipped in interactive mode, or when we detect
            # no transparent geometry was drawn.
            # For final (non-interactive) frames we still composite to handle
            # glass/water/hologram transparency correctly.
            t_rb = time.perf_counter()
            _skip_alpha_composite = self.interactive or (not transparent_nodes)
            if _skip_alpha_composite and _NUMPY:
                # FAST PATH: Read RGB only (skip alpha channel entirely).
                # Reading 3 components instead of 4 saves ~25% readback bandwidth.
                raw = read_fbo.read(components=3, dtype='f1')
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
                img = Image.fromarray(arr, 'RGB')
            elif _NUMPY:
                raw = read_fbo.read(components=4, dtype='f1')
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 4)[::-1].copy()
                # Alpha composite against opaque background for transparent surfaces
                bg = np.array([23, 25, 28], dtype=np.uint16)
                rgb  = arr[:, :, :3].astype(np.uint16)
                a    = arr[:, :, 3:4].astype(np.uint16)
                out  = ((rgb * a + bg * (255 - a)) // 255).clip(0, 255).astype(np.uint8)
                img  = Image.fromarray(out, 'RGB')
            else:
                raw = read_fbo.read(components=4, dtype='f1')
                rgba_img = Image.frombytes('RGBA', (W, H), raw)
                rgba_img = rgba_img.transpose(Image.FLIP_TOP_BOTTOM)
                bg_img = Image.new('RGB', (W, H), (23, 25, 28))
                bg_img.paste(rgba_img, mask=rgba_img.split()[3])
                img = bg_img
            # PERF-SCALE: Upscale reduced-resolution interactive frames smoothly.
            if self.interactive and (_full_W != W or _full_H != H):
                img = img.resize((_full_W, _full_H), Image.BILINEAR)
            self.perf['readback_ms'] = (time.perf_counter() - t_rb) * 1000

            return img

        except Exception as e:
            log.warning(f"GpuRenderer._render_gpu: {e}", exc_info=True)
            return None

    # ── Disabled CPU render hook ──────────────────────────────────────────────

    def _render_cpu(self, model, camera, W: int, H: int,
                    textures: Dict[str, 'Image.Image'],
                    anim_pose, anim_time: float) -> Optional['Image.Image']:
        """CPU graphics rendering is disabled; the viewport is GPU-only."""
        log.error("GpuRenderer._render_cpu called, but CPU graphics rendering is disabled")
        return None

    # ── Invalidate node cache ─────────────────────────────────────────────────

    def update_texture_regions(self, texture_name: str, image, regions, *, finalize: bool = True) -> bool:
        """Patch one resident GL texture without touching scene resources."""
        cache = self._tex_cache
        if cache is None:
            return False
        return bool(cache.update_regions(image, regions, build_mipmaps=bool(finalize)))

    def invalidate_node(self, node) -> None:
        """Remove cached GPU buffers and world-transform for a node (call after mesh edits)."""
        nid = id(node)
        if nid in self._mesh_cache:
            self._mesh_cache[nid].release()
            del self._mesh_cache[nid]
        # Also evict from persistent world-transform cache so next render recomputes
        if nid in self._wt_cache:
            del self._wt_cache[nid]
        stale_palette_keys = [
            key for key in self._skin_palette_bytes_cache
            if len(key) > 1 and key[1] == nid
        ]
        for key in stale_palette_keys:
            del self._skin_palette_bytes_cache[key]
        if bool(getattr(node, "is_emitter", False)):
            self.invalidate_particles(node)

    def invalidate_particles(self, node=None) -> None:
        """Restart particle simulations after emitter parameter edits."""
        systems = self._particle_systems
        if systems is None:
            return
        if node is None:
            systems.invalidate_all()
        else:
            systems.invalidate_node(node)

    def _update_and_draw_particles(self, ctx, model, textures, mvp, eye, target, up,
                                   get_world_transform, anim_pose, anim_time: float,
                                   anim_name: str = '') -> tuple:
        """Advance emitter simulations with wall-clock time and draw batches."""
        self.particles_active = False
        if model is None:
            return (0, 0)

        model_id = id(model)
        systems = self._particle_systems
        if systems is None or self._particle_model_id != model_id:
            from src.core.particles.simulation import ModelParticleSystems

            systems = ModelParticleSystems(model)
            self._particle_systems = systems
            self._particle_model_id = model_id
            self._particle_last_wall = 0.0
            self._particle_anim_cache = ("", None)
        if not systems.has_emitters:
            return (0, 0)

        now = time.perf_counter()
        last = self._particle_last_wall or now
        dt = min(0.1, max(0.0, now - last))
        self._particle_last_wall = now

        # Resolve the active animation block so emitter channels keyed by the
        # animation (birthrate/alpha gates like the Star Map "on") override the
        # bind pose while it plays.  The AnimationEngine attaches the resolved
        # block to each evaluated pose; viewport/scene models often carry an
        # empty ``animations`` list (the Animation Browser resolves clips via
        # source/supermodel chains), so the name lookup is only a fallback.
        animation = getattr(anim_pose, "_gr_animation", None) if anim_pose is not None else None
        wanted = str(anim_name or "").strip().lower()
        if animation is None and wanted and anim_pose is not None:
            cached_name, cached_anim = self._particle_anim_cache
            if cached_name == wanted and cached_anim is not None:
                animation = cached_anim
            else:
                for anim in getattr(model, "animations", None) or []:
                    if str(getattr(anim, "name", "") or "").strip().lower() == wanted:
                        animation = anim
                        break
                self._particle_anim_cache = (wanted, animation)

        def _emitter_transform(nd):
            return _strict_emitter_world_transform(nd, anim_pose)

        systems.update(dt, _emitter_transform, animation, float(anim_time))
        batches = systems.batches(_emitter_transform, tuple(eye))
        # Only visible batches keep the viewport's continuous particle frames
        # alive; a model whose emitters are all faded out costs nothing.
        self.particles_active = bool(batches)
        if not batches:
            return (0, 0)

        if self._particle_pass is None:
            from src.adapters.rendering.moderngl_particles import ModernGLParticlePass

            self._particle_pass = ModernGLParticlePass()
        if self._white_tex is None:
            self._white_tex = ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
        return self._particle_pass.draw(
            ctx,
            self._blend_submission,
            self._tex_cache,
            self._white_tex,
            textures,
            batches,
            _mat4_tobytes(mvp),
            tuple(eye),
            tuple(target),
            tuple(up),
            restore_cull=bool(self.cull_faces and not self.show_wireframe),
        )

    def invalidate_all(self) -> None:
        """Remove all cached GPU buffers and world-transform cache."""
        for m in self._mesh_cache.values():
            m.release()
        self._mesh_cache.clear()
        # Clear persistent world-transform cache; will be rebuilt next render
        self._wt_cache.clear()
        self._wt_model_id = 0
        self._skin_uploaders_by_scope.clear()
        self._skin_palette_bytes_cache.clear()
        self.invalidate_node_cache()

    def invalidate_node_cache(self) -> None:
        """Force node visibility/pass classification to rebuild next frame."""
        self._node_classification_revision += 1
        self._node_cache_model_id = 0
        self._node_cache_opaque = []
        self._node_cache_cutout = []
        self._node_cache_transparent = []
        self._node_cache_signature = ()
        self._scene_light_candidate_key = ()
        self._scene_light_candidate_nodes = ()

    def invalidate_transform_cache(self, reason: str = "transforms changed", node=None) -> None:
        """Evict world transforms without discarding retained draw classification.

        Runtime actors update their wrapper transform and pose every PIE frame,
        but neither operation changes mesh membership, visibility, or material
        pass.  Reclassifying the complete module here turns each actor tick into
        an O(scene) pre-draw walk.  A supplied scene-object root therefore evicts
        only its hierarchy; callers without a root retain the conservative full
        transform-cache invalidation.
        """

        if node is None:
            self._wt_cache.clear()
            return
        stack = [node]
        visited: set[int] = set()
        while stack:
            current = stack.pop()
            if current is None:
                continue
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            self._wt_cache.pop(current_id, None)
            stack.extend(getattr(current, "children", []) or [])

    @staticmethod
    def _sprite_text(node) -> str:
        names = getattr(node, "texture_names", None) or []
        first_name = str(names[0]) if names else ""
        return f"{getattr(node, 'name', '')} {getattr(node, 'texture', '')} {first_name}".lower()

    @staticmethod
    def _has_sprite_material_override(node) -> bool:
        return bool(
            str(getattr(node, "_gr_sprite_category", "") or "").strip()
            or str(getattr(node, "_gr_sprite_render_mode", "") or "").strip()
            or getattr(node, "_gr_sprite_glow", None) is not None
            or str(getattr(node, "_gr_sprite_alpha_source", "") or "").strip()
        )

    @classmethod
    def _is_sprite_hilt(cls, node) -> bool:
        if str(getattr(node, "_gr_sprite_category", "") or "").lower() == "hilt":
            return True
        text = cls._sprite_text(node)
        return (
            "w_lghtsbr" in text
            or "w_shortsbr" in text
            or "w_dblsbr" in text
            or text.startswith("lghtsbr")
            or " lghtsbr" in text
            or "lshandle" in text
            or "handle" in text
        )

    @classmethod
    def _sprite_alpha_source(cls, node) -> int:
        source = str(getattr(node, "_gr_sprite_alpha_source", "") or "").lower()
        if source in {"luminance", "brightness", "matte", "black_key"}:
            return 1
        if cls._has_sprite_material_override(node):
            return 0
        if cls._is_sprite_hilt(node):
            return 0
        text = cls._sprite_text(node)
        return 1 if any(token in text for token in ("saber", "sabre", "lsabre", "blade", "glow", "flare", "beam")) else 0

    @classmethod
    def _sprite_glow(cls, node) -> float:
        explicit = getattr(node, "_gr_sprite_glow", None)
        if explicit is not None:
            try:
                return max(0.0, min(4.0, float(explicit)))
            except Exception:
                return 0.0
        if cls._is_sprite_hilt(node):
            return 0.0
        text = cls._sprite_text(node)
        return 1.6 if any(token in text for token in ("saber", "sabre", "lsabre", "blade", "glow", "flare", "beam")) else 0.0

    def _node_classification_signature(self, nodes) -> tuple:
        signature = []
        for node in nodes:
            if node is None:
                continue
            signature.append(
                (
                    id(node),
                    int(getattr(node, "_gr_revision", 0) or 0),
                    bool(getattr(node, "_gr_hidden", False)),
                    bool(getattr(node, "render", True)),
                    int(getattr(node, "txi_blending", 0) or 0),
                    round(float(getattr(node, "txi_alpha_test", 0.0) or 0.0), 4),
                    round(float(getattr(node, "txi_wateralpha", 1.0) or 1.0), 4),
                    bool(getattr(node, "txi_decal", False)),
                    round(float(getattr(node, "alpha", 1.0) or 1.0), 4),
                    int(getattr(node, "transparency_hint", 0) or 0),
                    str(getattr(node, "_gr_sprite_render_mode", "") or ""),
                    self._sprite_alpha_source(node),
                    round(self._sprite_glow(node), 4),
                )
            )
        return tuple(signature)

    # ── Performance info ──────────────────────────────────────────────────────

    @property
    def is_gpu(self) -> bool:
        """True if the GPU (ModernGL/EGL) path is active."""
        return self._gpu_available and not self.force_cpu

    def perf_summary(self) -> str:
        p = self.perf
        return (
            f"backend={p['backend']}  "
            f"frame={p['last_frame_ms']:.1f}ms  "
            f"tris={p['tri_count']}  "
            f"upload={p['gpu_upload_ms']:.1f}ms  "
            f"draw={p['draw_ms']:.1f}ms  "
            f"readback={p['readback_ms']:.1f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-framing camera helper
# ─────────────────────────────────────────────────────────────────────────────


__all__ = tuple(name for name in globals() if not name.startswith("__"))
