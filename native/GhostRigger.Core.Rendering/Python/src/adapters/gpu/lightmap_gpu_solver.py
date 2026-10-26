"""GPU direct-light solver for generated lightmap bakes."""

from __future__ import annotations

import logging

import numpy as np

from src.core.lighting.lightmap_bake_settings import uses_kotor_engine_model
from src.core.lighting.lightmap_lighting_solver import LightmapLightingSolver

log = logging.getLogger(__name__)

try:
    import moderngl

    from src.adapters.gpu.moderngl_context import _create_moderngl_standalone_context
except Exception:  # pragma: no cover - optional GPU dependency
    moderngl = None
    _create_moderngl_standalone_context = None


_MAX_GPU_LIGHTS = 128

_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG = f"""
#version 330
const int MAX_LIGHTS = {_MAX_GPU_LIGHTS};
uniform sampler2D u_pos_valid;
uniform sampler2D u_normal_tex;
uniform sampler2D u_light_pos_radius;
uniform sampler2D u_light_color_intensity;
uniform sampler2D u_light_dir_kind;
uniform sampler2D u_light_extra;
uniform int u_light_count;
uniform int u_use_direct;
in vec2 v_uv;
out vec4 frag;

vec3 safe_normal(vec3 n) {{
    float l = length(n);
    if (l <= 0.000001) return vec3(0.0, 0.0, 1.0);
    return n / l;
}}

void main() {{
    vec4 pv = texture(u_pos_valid, v_uv);
    if (pv.a < 0.5) {{
        frag = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }}
    vec3 pos = pv.xyz;
    vec3 normal = safe_normal(texture(u_normal_tex, v_uv).xyz);
    vec3 rgb = vec3(0.0);
    if (u_use_direct != 0) {{
        for (int i = 0; i < MAX_LIGHTS; ++i) {{
            if (i >= u_light_count) break;
            float lu = (float(i) + 0.5) / float(MAX_LIGHTS);
            vec4 pr = texture(u_light_pos_radius, vec2(lu, 0.5));
            vec4 ci = texture(u_light_color_intensity, vec2(lu, 0.5));
            vec4 dk = texture(u_light_dir_kind, vec2(lu, 0.5));
            vec4 extra = texture(u_light_extra, vec2(lu, 0.5));
            vec3 color = ci.rgb;
            float intensity = ci.a;
            int kind = int(dk.w + 0.5);
            if (kind == 4) {{
                rgb += color * intensity * 0.25;
            }} else if (kind == 2) {{
                vec3 ldir = safe_normal(-dk.xyz);
                float ndotl = max(dot(normal, ldir), 0.0);
                rgb += color * intensity * ndotl;
            }} else {{
                vec3 vec_to_light = pr.xyz - pos;
                float dist = max(length(vec_to_light), 0.00001);
                float radius = max(pr.w, 0.001);
                if (dist <= radius) {{
                    vec3 ldir = vec_to_light / dist;
                    float ndotl = max(dot(normal, ldir), 0.0);
                    float falloff = max(0.0, 1.0 - (dist / radius));
                    float attenuation = falloff * falloff;
                    vec3 contrib = color * intensity * ndotl * attenuation * 0.45;
                    if (kind == 1) {{
                        vec3 to_texel = safe_normal(pos - pr.xyz);
                        float cone = radians(clamp(extra.x, 1.0, 179.0));
                        float inner = cos(cone * 0.5);
                        float outer = cos(cone * 0.65);
                        float dotv = dot(safe_normal(dk.xyz), to_texel);
                        float spot = 0.0;
                        if (dotv >= inner) spot = 1.0;
                        else if (dotv > outer) spot = (dotv - outer) / max(inner - outer, 0.00001);
                        contrib *= spot;
                    }} else if (kind == 3) {{
                        // Same area scale as the CPU solver: 0.75 + min(size, 10) * 0.025.
                        contrib *= 0.75 + min(max(extra.y, 0.0), 10.0) * 0.025;
                    }}
                    rgb += contrib;
                }}
            }}
        }}
    }}
    // No clamp here: negative (darkening) lights must offset ambient exactly as on the CPU path; the float
    // target keeps the sign and apply_exposure_gamma clamps the final value.
    frag = vec4(rgb, 1.0);
}}
"""


class LightmapGpuSolver:
    """ModernGL offscreen direct-light solve.

    CPU still prepares/rasterizes texel buffers. This class uploads those buffers
    to GPU textures, evaluates direct lighting in a fragment shader, and reads
    the baked RGB result back to RAM for padding/output.
    """

    def __init__(self, cpu_solver: LightmapLightingSolver | None = None) -> None:
        self.cpu_solver = cpu_solver or LightmapLightingSolver()
        self._ctx = None
        self._prog = None
        self._vao = None
        self.available = False
        self.last_backend = "none"
        self.last_warning = ""
        self.last_info = ""

    def solve_buffer(self, buffer, lights, settings, shadow_solver=None) -> np.ndarray:
        self.last_warning = ""
        self.last_info = ""
        active_lights = [light for light in lights if self.cpu_solver._include_light(light, settings)]
        if uses_kotor_engine_model(settings):
            # The KOTOR forward model is evaluated on the CPU in float64 so validation results do not depend on
            # the GPU or its driver.
            result = self.cpu_solver.solve_buffer(buffer, active_lights, settings, shadow_solver)
            self.last_warning = str(getattr(self.cpu_solver, "last_warning", "") or "")
            self.last_info = f"CPU lightmap solve used. {getattr(self.cpu_solver, 'last_info', '') or ''}".strip()
            return result
        if not self.can_use_gpu(settings, shadow_solver):
            if not self.last_info:
                self.last_info = "CPU lightmap solve used."
            return self.cpu_solver.solve_buffer(buffer, active_lights, settings, shadow_solver)
        try:
            self._ensure()
            if not self.available:
                if not self.last_info:
                    self.last_info = "CPU lightmap solve used."
                return self.cpu_solver.solve_buffer(buffer, active_lights, settings, shadow_solver)
            return self._solve_gpu(buffer, active_lights, settings)
        except Exception as exc:
            self.last_warning = f"GPU lightmap solve failed; using CPU fallback: {exc}"
            self.last_info = "CPU lightmap solve used."
            log.debug(self.last_warning, exc_info=True)
            return self.cpu_solver.solve_buffer(buffer, active_lights, settings, shadow_solver)

    def can_use_gpu(self, settings, shadow_solver=None) -> bool:
        if not bool(getattr(settings, "use_gpu_acceleration", True)):
            self.last_info = "CPU lightmap solve used because GPU acceleration is disabled."
            return False
        if shadow_solver is not None and bool(getattr(settings, "use_shadows", False)):
            self.last_warning = "GPU direct-light solver bypassed because CPU shadow rays are enabled."
            self.last_info = "CPU lightmap solve used."
            return False
        return moderngl is not None and _create_moderngl_standalone_context is not None

    def _ensure(self) -> None:
        if self._ctx is not None:
            return
        if moderngl is None or _create_moderngl_standalone_context is None:
            self.available = False
            self.last_warning = "ModernGL is unavailable; using CPU lightmap solve."
            self.last_info = "CPU lightmap solve used."
            return
        self._ctx, self.last_backend = _create_moderngl_standalone_context()
        self._prog = self._ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        vertices = np.asarray([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype=np.float32)
        vbo = self._ctx.buffer(vertices.tobytes())
        self._vao = self._ctx.vertex_array(self._prog, [(vbo, "2f", "in_pos")])
        self.available = True

    def _solve_gpu(self, buffer, lights, settings) -> np.ndarray:
        res = int(buffer.resolution)
        pos_valid = np.zeros((res, res, 4), dtype=np.float32)
        pos_valid[:, :, :3] = buffer.world_positions
        pos_valid[:, :, 3] = buffer.valid_mask.astype(np.float32)
        normals = np.zeros((res, res, 4), dtype=np.float32)
        normals[:, :, :3] = buffer.world_normals
        ctx = self._ctx
        pos_tex = self._tex2d(pos_valid)
        normal_tex = self._tex2d(normals)

        result = np.zeros((res, res, 3), dtype=np.float32)
        if settings.use_direct_lighting and lights:
            for start in range(0, len(lights), _MAX_GPU_LIGHTS):
                result += self._render_direct_chunk(pos_tex, normal_tex, lights[start : start + _MAX_GPU_LIGHTS], res, settings)

        valid = buffer.valid_mask
        # Mirror LightmapLightingSolver.solve_buffer exactly so a bake does not depend on GPU availability.
        if settings.include_ambient and valid.any():
            result[valid] += self.cpu_solver.solve_ambient({}, settings)
        if settings.use_indirect_approximation and valid.any():
            result[valid] += 0.08 * float(getattr(settings, "indirect_strength", 1.0)) * buffer.base_diffuse[valid]
        if settings.include_diffuse and valid.any():
            strength = float(getattr(settings, "diffuse_strength", 1.0))
            result[valid] *= (1.0 - strength) + buffer.base_diffuse[valid] * strength
        result[valid] = self.cpu_solver.apply_exposure_gamma(result[valid], settings)
        result[~valid] = 0.0
        self.last_info = f"GPU direct-light solve used ModernGL backend '{self.last_backend}' for {len(lights)} light(s)."
        for resource in (pos_tex, normal_tex):
            try:
                resource.release()
            except Exception:
                pass
        return result

    def _render_direct_chunk(self, pos_tex, normal_tex, lights, res: int, settings) -> np.ndarray:
        light_pos_radius, light_color_intensity, light_dir_kind, light_extra = self._pack_lights(lights, settings)
        light_pos_tex = self._tex2d(light_pos_radius)
        light_color_tex = self._tex2d(light_color_intensity)
        light_dir_tex = self._tex2d(light_dir_kind)
        light_extra_tex = self._tex2d(light_extra)
        out_tex = self._ctx.texture((res, res), 4, dtype="f4")
        fbo = self._ctx.framebuffer(color_attachments=[out_tex])
        fbo.use()
        self._ctx.viewport = (0, 0, res, res)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.BLEND)

        prog = self._prog
        pos_tex.use(0)
        normal_tex.use(1)
        light_pos_tex.use(2)
        light_color_tex.use(3)
        light_dir_tex.use(4)
        light_extra_tex.use(5)
        prog["u_pos_valid"].value = 0
        prog["u_normal_tex"].value = 1
        prog["u_light_pos_radius"].value = 2
        prog["u_light_color_intensity"].value = 3
        prog["u_light_dir_kind"].value = 4
        prog["u_light_extra"].value = 5
        prog["u_light_count"].value = len(lights)
        prog["u_use_direct"].value = 1 if settings.use_direct_lighting else 0

        self._vao.render(moderngl.TRIANGLES)
        data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape((res, res, 4))
        result = data[:, :, :3].copy()
        for resource in (light_pos_tex, light_color_tex, light_dir_tex, light_extra_tex, out_tex, fbo):
            try:
                resource.release()
            except Exception:
                pass
        return result

    def _tex2d(self, data: np.ndarray):
        h, w = data.shape[:2]
        tex = self._ctx.texture((w, h), 4, data=np.ascontiguousarray(data).tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _pack_lights(self, lights, settings=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_radius = np.zeros((1, _MAX_GPU_LIGHTS, 4), dtype=np.float32)
        color_intensity = np.zeros((1, _MAX_GPU_LIGHTS, 4), dtype=np.float32)
        dir_kind = np.zeros((1, _MAX_GPU_LIGHTS, 4), dtype=np.float32)
        extra = np.zeros((1, _MAX_GPU_LIGHTS, 4), dtype=np.float32)
        # CPU parity: point/spot/area radii scale by light_falloff_multiplier and ambient lights by ambient_strength.
        radius_scale = max(0.001, float(getattr(settings, "light_falloff_multiplier", 1.0)))
        ambient_strength = float(getattr(settings, "ambient_strength", 1.0))
        for idx, light in enumerate(lights[:_MAX_GPU_LIGHTS]):
            pos = _vec3(getattr(light, "position", (0.0, 0.0, 0.0)))
            color = _vec3(getattr(light, "color", getattr(light, "light_color", (1.0, 1.0, 1.0))), (1.0, 1.0, 1.0))
            direction = _vec3(getattr(light, "direction", (0.0, -1.0, -1.0)), (0.0, -1.0, -1.0))
            kind = _kind_code(light)
            intensity = max(0.0, float(getattr(light, "intensity", getattr(light, "light_multiplier", 1.0)) or 0.0))
            if kind == 4:
                intensity *= ambient_strength
            pos_radius[0, idx, :3] = pos
            pos_radius[0, idx, 3] = max(float(getattr(light, "radius", getattr(light, "light_radius", 5.0)) or 5.0), 0.001) * radius_scale
            color_intensity[0, idx, :3] = color
            color_intensity[0, idx, 3] = intensity
            dir_kind[0, idx, :3] = direction
            dir_kind[0, idx, 3] = float(kind)
            extra[0, idx, 0] = max(1.0, min(179.0, float(getattr(light, "cone_angle", 45.0) or 45.0)))
            extra[0, idx, 1] = max(0.0, float(getattr(light, "area_size", 0.0) or 0.0))
        return pos_radius, color_intensity, dir_kind, extra


def _kind_code(light: object) -> int:
    kind = str(getattr(light, "type", getattr(light, "light_kind", "point")) or "point").lower()
    if "ambient" in kind or bool(getattr(light, "ambient_only", False)):
        return 4
    if "directional" in kind:
        return 2
    if "area" in kind:
        return 3
    if "spot" in kind:
        return 1
    return 0


def _vec3(value: object, fallback=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return fallback
