"""Direct lighting solver for generated lightmap bakes.

Two light models (``LightmapBakeSettings.lighting_model``):

* ``"ghost"`` (default): the artist bake.  Smooth ``(1 - d/r)^2 x 0.45`` falloff, optional ambient floor and
  indirect term, diffuse multiply, Reinhard tone map and display gamma.  Good-looking authored maps; not linear in
  stored bytes and not a model of the engine.
* ``"kotor_engine"``: the engine's light term (``kotor_engine_lighting``) in stored-byte units: attenuation
  ``1 / (1 + 4.5 d^2 / r^2)`` (or a chosen falloff family), ambient-only lights attenuated without N.L, no specular,
  multiplier ignored like the vertex program, no tone map, no baked diffuse, linear encode.  In game the result is
  drawn as ``D x (v0 + LM)``.  Use it to validate lights fitted from vanilla lightmaps.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from . import kotor_engine_lighting as kel
from .lightmap_bake_settings import uses_kotor_engine_model


class LightmapLightingSolver:
    last_warning: str = ""
    last_info: str = ""

    def solve_buffer(self, buffer, lights: Iterable[object], settings, shadow_solver=None) -> np.ndarray:
        self.last_warning = ""
        self.last_info = ""
        if uses_kotor_engine_model(settings):
            return self._solve_buffer_kotor(buffer, lights, settings, shadow_solver)
        output = np.zeros_like(buffer.baked_rgb, dtype=np.float32)
        active_lights = [light for light in lights if self._include_light(light, settings)]
        if not active_lights and not settings.include_ambient and not settings.use_indirect_approximation:
            return output

        ys, xs = np.nonzero(buffer.valid_mask)
        if len(ys) == 0:
            return output

        positions = buffer.world_positions[ys, xs].astype(np.float32, copy=False)
        normals = _normalized_rows(buffer.world_normals[ys, xs].astype(np.float32, copy=False))
        diffuse = buffer.base_diffuse[ys, xs].astype(np.float32, copy=False)
        rgb = np.zeros((len(ys), 3), dtype=np.float32)

        if settings.include_ambient:
            rgb += self.solve_ambient_batch(len(ys), settings)
        if settings.use_direct_lighting:
            mesh_ids = buffer.mesh_ids[ys, xs]
            triangle_ids = buffer.triangle_ids[ys, xs]
            for light in active_lights:
                contribution = self.solve_direct_light_batch(positions, normals, light, settings)
                if shadow_solver is not None and settings.use_shadows and np.any(contribution > 0.0):
                    contribution *= self._shadow_factors_batch(
                        positions,
                        normals,
                        mesh_ids,
                        triangle_ids,
                        contribution,
                        light,
                        settings,
                        shadow_solver,
                    )[:, np.newaxis]
                rgb += contribution
        if settings.use_indirect_approximation:
            rgb += 0.08 * float(getattr(settings, "indirect_strength", 1.0)) * diffuse
        if settings.include_diffuse:
            strength = float(getattr(settings, "diffuse_strength", 1.0))
            rgb *= (1.0 - strength) + diffuse * strength
        output[ys, xs] = self.apply_exposure_gamma(rgb, settings)
        return output

    def _solve_buffer_scalar(self, buffer, lights: Iterable[object], settings, shadow_solver=None) -> np.ndarray:
        output = np.zeros_like(buffer.baked_rgb, dtype=np.float32)
        ys, xs = np.nonzero(buffer.valid_mask)
        active_lights = [light for light in lights if self._include_light(light, settings)]
        for y, x in zip(ys, xs):
            texel = {
                "position": buffer.world_positions[y, x],
                "normal": buffer.world_normals[y, x],
                "diffuse": buffer.base_diffuse[y, x],
                "mesh_id": int(buffer.mesh_ids[y, x]),
                "triangle_id": int(buffer.triangle_ids[y, x]),
            }
            rgb = self.solve_texel_lighting(texel, active_lights, settings, shadow_solver)
            output[y, x] = rgb
        return output

    def solve_texel_lighting(self, texel, lights, settings, shadow_solver=None) -> np.ndarray:
        if uses_kotor_engine_model(settings):
            return self._solve_texel_kotor(texel, lights, settings, shadow_solver)
        rgb = np.zeros(3, dtype=np.float32)
        if settings.include_ambient:
            rgb += self.solve_ambient(texel, settings)
        if settings.use_direct_lighting:
            for light in lights:
                contribution = self.solve_direct_light(texel, light, settings)
                if shadow_solver is not None and settings.use_shadows:
                    raw = float(shadow_solver.calculate_shadow_factor(texel, light, settings))
                    strength = float(getattr(settings, "shadow_strength", 1.0))
                    contribution *= 1.0 - strength * (1.0 - raw)
                rgb += contribution
        if settings.use_indirect_approximation:
            rgb += 0.08 * float(getattr(settings, "indirect_strength", 1.0)) * np.asarray(texel["diffuse"], dtype=np.float32)
        if settings.include_diffuse:
            strength = float(getattr(settings, "diffuse_strength", 1.0))
            diffuse = np.asarray(texel["diffuse"], dtype=np.float32)
            rgb *= (1.0 - strength) + diffuse * strength
        return self.apply_exposure_gamma(rgb, settings)

    def solve_direct_light(self, texel, light, settings) -> np.ndarray:
        kind = str(getattr(light, "type", getattr(light, "light_kind", "point")) or "point").lower()
        if "ambient" in kind or bool(getattr(light, "ambient_only", False)):
            return self.solve_ambient(texel, settings, light)
        if "spot" in kind:
            return self.solve_spot_light(texel, light, settings)
        if "directional" in kind:
            return self.solve_directional_light(texel, light, settings)
        if "area" in kind:
            return self.solve_area_light_approx(texel, light, settings)
        return self.solve_point_light(texel, light, settings)

    def solve_direct_light_batch(self, positions: np.ndarray, normals: np.ndarray, light: object, settings) -> np.ndarray:
        kind = str(getattr(light, "type", getattr(light, "light_kind", "point")) or "point").lower()
        if "ambient" in kind or bool(getattr(light, "ambient_only", False)):
            return np.repeat(self.solve_ambient({}, settings, light)[np.newaxis, :], len(positions), axis=0)
        if "spot" in kind:
            return self.solve_spot_light_batch(positions, normals, light, settings)
        if "directional" in kind:
            return self.solve_directional_light_batch(positions, normals, light, settings)
        if "area" in kind:
            result = self.solve_point_light_batch(positions, normals, light, settings)
            size = max(0.0, float(getattr(light, "area_size", 0.0) or 0.0))
            return result * (0.75 + min(size, 10.0) * 0.025)
        return self.solve_point_light_batch(positions, normals, light, settings)

    def solve_point_light(self, texel, light, settings) -> np.ndarray:
        position = np.asarray(texel["position"], dtype=np.float32)
        normal = _normalized(np.asarray(texel["normal"], dtype=np.float32))
        light_pos = np.asarray(getattr(light, "position", (0.0, 0.0, 0.0)), dtype=np.float32)
        vec = light_pos - position
        dist = max(float(np.linalg.norm(vec)), 1.0e-5)
        radius = max(float(getattr(light, "radius", getattr(light, "light_radius", 5.0)) or 5.0), 0.001)
        radius *= max(0.001, float(getattr(settings, "light_falloff_multiplier", 1.0)))
        if dist > radius:
            return np.zeros(3, dtype=np.float32)
        ldir = vec / dist
        ndotl = max(float(np.dot(normal, ldir)), 0.0)
        if ndotl <= 0.0:
            return np.zeros(3, dtype=np.float32)
        # Bake-friendly falloff. The previous inverse-square * radius term
        # clipped most Aurora-heavy module bakes to white. Generated lightmaps
        # need preserved gradients more than photometric brightness, so use a
        # smooth radius falloff and leave final compression to tone mapping.
        falloff = max(0.0, 1.0 - (dist / radius))
        attenuation = falloff * falloff
        return _light_color(light) * _intensity(light) * ndotl * attenuation * 0.45

    def solve_point_light_batch(self, positions: np.ndarray, normals: np.ndarray, light: object, settings) -> np.ndarray:
        light_pos = np.asarray(getattr(light, "position", (0.0, 0.0, 0.0)), dtype=np.float32)
        vec = light_pos[np.newaxis, :] - positions
        dist = np.linalg.norm(vec, axis=1)
        safe_dist = np.maximum(dist, 1.0e-5)
        radius = max(float(getattr(light, "radius", getattr(light, "light_radius", 5.0)) or 5.0), 0.001)
        radius *= max(0.001, float(getattr(settings, "light_falloff_multiplier", 1.0)))
        ldir = vec / safe_dist[:, np.newaxis]
        ndotl = np.maximum(np.sum(normals * ldir, axis=1), 0.0)
        falloff = np.maximum(0.0, 1.0 - (safe_dist / radius))
        attenuation = falloff * falloff
        active = (dist <= radius) & (ndotl > 0.0)
        scale = np.where(active, ndotl * attenuation * _intensity(light) * 0.45, 0.0).astype(np.float32)
        return scale[:, np.newaxis] * _light_color(light)[np.newaxis, :]

    def solve_spot_light(self, texel, light, settings) -> np.ndarray:
        base = self.solve_point_light(texel, light, settings)
        if not np.any(base):
            return base
        position = np.asarray(texel["position"], dtype=np.float32)
        light_pos = np.asarray(getattr(light, "position", (0.0, 0.0, 0.0)), dtype=np.float32)
        to_texel = _normalized(position - light_pos)
        direction = _normalized(np.asarray(getattr(light, "direction", (0.0, -1.0, -1.0)), dtype=np.float32))
        cone = math.radians(max(1.0, min(179.0, float(getattr(light, "cone_angle", 45.0)))))
        inner = math.cos(cone * 0.5)
        outer = math.cos(cone * 0.65)
        dot_val = float(np.dot(direction, to_texel))
        if dot_val <= outer:
            return np.zeros(3, dtype=np.float32)
        spot = 1.0 if dot_val >= inner else (dot_val - outer) / max(inner - outer, 1.0e-5)
        return base * spot

    def solve_spot_light_batch(self, positions: np.ndarray, normals: np.ndarray, light: object, settings) -> np.ndarray:
        base = self.solve_point_light_batch(positions, normals, light, settings)
        if not np.any(base):
            return base
        light_pos = np.asarray(getattr(light, "position", (0.0, 0.0, 0.0)), dtype=np.float32)
        to_texel = _normalized_rows(positions - light_pos[np.newaxis, :])
        direction = _normalized(np.asarray(getattr(light, "direction", (0.0, -1.0, -1.0)), dtype=np.float32))
        cone = math.radians(max(1.0, min(179.0, float(getattr(light, "cone_angle", 45.0)))))
        inner = math.cos(cone * 0.5)
        outer = math.cos(cone * 0.65)
        dot_val = np.sum(to_texel * direction[np.newaxis, :], axis=1)
        spot = np.zeros(len(positions), dtype=np.float32)
        spot[dot_val >= inner] = 1.0
        fade = (dot_val > outer) & (dot_val < inner)
        spot[fade] = (dot_val[fade] - outer) / max(inner - outer, 1.0e-5)
        return base * spot[:, np.newaxis]

    def solve_directional_light(self, texel, light, settings) -> np.ndarray:
        normal = _normalized(np.asarray(texel["normal"], dtype=np.float32))
        direction = _normalized(-np.asarray(getattr(light, "direction", (0.0, -1.0, -1.0)), dtype=np.float32))
        ndotl = max(float(np.dot(normal, direction)), 0.0)
        return _light_color(light) * _intensity(light) * ndotl

    def solve_directional_light_batch(self, positions: np.ndarray, normals: np.ndarray, light: object, settings) -> np.ndarray:
        direction = _normalized(-np.asarray(getattr(light, "direction", (0.0, -1.0, -1.0)), dtype=np.float32))
        ndotl = np.maximum(np.sum(normals * direction[np.newaxis, :], axis=1), 0.0)
        return ndotl[:, np.newaxis].astype(np.float32) * _light_color(light)[np.newaxis, :] * _intensity(light)

    def solve_area_light_approx(self, texel, light, settings) -> np.ndarray:
        result = self.solve_point_light(texel, light, settings)
        size = max(0.0, float(getattr(light, "area_size", 0.0) or 0.0))
        return result * (0.75 + min(size, 10.0) * 0.025)

    def solve_ambient(self, texel, settings, light: object | None = None) -> np.ndarray:
        if light is not None:
            return _light_color(light) * _intensity(light) * 0.25 * float(getattr(settings, "ambient_strength", 1.0))
        ambient = np.asarray(getattr(settings, "background_color", (0.0, 0.0, 0.0)), dtype=np.float32) + np.asarray((0.035, 0.035, 0.035), dtype=np.float32)
        return ambient * float(getattr(settings, "ambient_strength", 1.0))

    def solve_ambient_batch(self, count: int, settings) -> np.ndarray:
        return np.repeat(self.solve_ambient({}, settings)[np.newaxis, :], int(count), axis=0)

    def apply_exposure_gamma(self, rgb, settings) -> np.ndarray:
        value = np.asarray(rgb, dtype=np.float32) * float(settings.exposure)
        # Reinhard tone mapping keeps intense multi-light rooms from becoming
        # flat white while still allowing bright highlights to read as bright.
        value = value / (1.0 + np.maximum(value, 0.0))
        if settings.clamp_output:
            value = np.clip(value, 0.0, 1.0)
        gamma = max(float(settings.gamma), 1.0e-5)
        return np.power(np.clip(value, 0.0, None), 1.0 / gamma)

    # -- KOTOR engine light model ----------------------------------------------------------------------------

    def solve_kotor_light_batch(self, positions: np.ndarray, normals: np.ndarray, light: object, settings) -> np.ndarray:
        """One light's engine light term, (M, 3) float64, before shadows and encoding."""

        engine_light = kel.KotorEngineLight.from_record(light)
        apply_multiplier = bool(getattr(settings, "kotor_apply_multiplier", False))
        if "directional" in _light_kind(light) and not engine_light.ambient_only:
            # The engine has no directional lights (its ARE sun is a point light, kel.sun_light); a directional
            # record is evaluated as a distant light: N.L x colour, no attenuation.
            direction = _normalized(-np.asarray(getattr(light, "direction", (0.0, -1.0, -1.0)), dtype=np.float64))
            unit = _normalized_rows(np.asarray(normals, dtype=np.float64).reshape(-1, 3))
            ndotl = np.maximum(unit @ direction, 0.0)
            color = np.asarray(engine_light.color, dtype=np.float64)
            if apply_multiplier:
                color = color * float(engine_light.multiplier)
            return ndotl[:, np.newaxis] * color[np.newaxis, :]
        radius_scale = max(0.001, float(getattr(settings, "light_falloff_multiplier", 1.0) or 1.0))
        if radius_scale != 1.0:
            engine_light = replace(engine_light, radius=float(engine_light.radius) * radius_scale)
        return kel.light_term(
            positions,
            normals,
            engine_light,
            falloff_family=str(getattr(settings, "kotor_falloff", "engine") or "engine"),
            cutoff_scale=float(getattr(settings, "kotor_cutoff_scale", 1.0)),
            apply_multiplier=apply_multiplier,
        )

    def _solve_buffer_kotor(self, buffer, lights, settings, shadow_solver=None) -> np.ndarray:
        output = np.zeros_like(buffer.baked_rgb, dtype=np.float32)
        active_lights = [light for light in lights if self._include_light(light, settings)]
        self._record_kotor_notes(active_lights, settings)
        ys, xs = np.nonzero(buffer.valid_mask)
        if len(ys) == 0:
            return output
        positions = buffer.world_positions[ys, xs].astype(np.float64)
        normals = _normalized_rows(buffer.world_normals[ys, xs].astype(np.float64))
        rgb = self._kotor_light_sum(
            positions,
            normals,
            buffer.mesh_ids[ys, xs],
            buffer.triangle_ids[ys, xs],
            active_lights,
            settings,
            shadow_solver,
        )
        output[ys, xs] = self._kotor_encode(rgb, settings).astype(np.float32)
        return output

    def _solve_texel_kotor(self, texel, lights, settings, shadow_solver=None) -> np.ndarray:
        positions = np.asarray(texel["position"], dtype=np.float64).reshape(1, 3)
        normals = _normalized_rows(np.asarray(texel["normal"], dtype=np.float64).reshape(1, 3))
        mesh_ids = np.asarray([int(texel.get("mesh_id", -1))], dtype=np.int64)
        triangle_ids = np.asarray([int(texel.get("triangle_id", -1))], dtype=np.int64)
        rgb = self._kotor_light_sum(positions, normals, mesh_ids, triangle_ids, list(lights), settings, shadow_solver)
        return self._kotor_encode(rgb, settings)[0].astype(np.float32)

    def _kotor_light_sum(self, positions, normals, mesh_ids, triangle_ids, lights, settings, shadow_solver) -> np.ndarray:
        rgb = np.zeros((len(positions), 3), dtype=np.float64)
        if bool(getattr(settings, "include_ambient", False)):
            # An explicit constant fill (the per-room fill term of a fit); no hidden floor.
            fill = np.asarray(getattr(settings, "background_color", (0.0, 0.0, 0.0)), dtype=np.float64)
            rgb += fill[np.newaxis, :] * float(getattr(settings, "ambient_strength", 1.0))
        if not bool(getattr(settings, "use_direct_lighting", True)):
            return rgb
        use_shadows = shadow_solver is not None and bool(getattr(settings, "use_shadows", False))
        for light in lights:
            contribution = self.solve_kotor_light_batch(positions, normals, light, settings)
            if use_shadows and not _is_ambient_light(light):
                magnitude = np.abs(contribution)
                if np.any(magnitude > 1.0e-8):
                    # Negative (darkening) lights are occluded like positive ones; the shadow rays go to the
                    # light's world position, the same point the light term uses.
                    contribution *= self._shadow_factors_batch(
                        positions,
                        normals,
                        mesh_ids,
                        triangle_ids,
                        magnitude,
                        _kotor_shadow_proxy(light),
                        settings,
                        shadow_solver,
                    )[:, np.newaxis]
            rgb += contribution
        return rgb

    def _kotor_encode(self, rgb: np.ndarray, settings) -> np.ndarray:
        return kel.encode_lightmap(
            rgb,
            exposure=float(getattr(settings, "exposure", 1.0)),
            encode_gamma=float(getattr(settings, "encode_gamma", 1.0) or 1.0),
        )

    def _record_kotor_notes(self, lights, settings) -> None:
        notes: list[str] = []
        if bool(getattr(settings, "include_diffuse", False)):
            notes.append("include_diffuse is ignored: the game multiplies the diffuse texture at draw time, D x (v0 + LM)")
        if bool(getattr(settings, "use_indirect_approximation", False)):
            notes.append("use_indirect_approximation is ignored: it is not part of the engine light model")
        not_engine = [
            str(getattr(light, "name", "") or "light")
            for light in lights
            if not _is_ambient_light(light) and any(tag in _light_kind(light) for tag in ("spot", "area", "directional"))
        ]
        if not_engine:
            notes.append(
                f"{len(not_engine)} light(s) are not engine light types (the engine has point lights only); spot and "
                f"area lights were baked as point lights, directional lights as distant lights: {_names(not_engine)}"
            )
        runtime = [
            str(getattr(light, "name", "") or "light")
            for light in lights
            if kel.KotorEngineLight.from_record(light).dynamic_type == 1
        ]
        if runtime:
            notes.append(
                f"{len(runtime)} dynamic type-1 light(s) were baked; the engine also adds type-1 lights to static rooms "
                f"at run time (v0), so under the additive composite they count twice: {_names(runtime)}"
            )
        self.last_warning = "; ".join(notes)
        family = str(getattr(settings, "kotor_falloff", "engine") or "engine")
        cutoff = float(getattr(settings, "kotor_cutoff_scale", 1.0))
        multiplier = "applied" if bool(getattr(settings, "kotor_apply_multiplier", False)) else "ignored"
        self.last_info = (
            f"KOTOR engine light model ({kel.ENGINE_MODEL_ID}): falloff {family}, "
            f"cut {'none' if cutoff <= 0.0 else f'{cutoff:g} r'}, multiplier {multiplier}, "
            f"encode gamma {float(getattr(settings, 'encode_gamma', 1.0) or 1.0):g}; the game draws D x (v0 + LM)."
        )

    def _shadow_factors_batch(
        self,
        positions,
        normals,
        mesh_ids,
        triangle_ids,
        contribution,
        light,
        settings,
        shadow_solver,
    ) -> np.ndarray:
        factors = np.ones(len(positions), dtype=np.float32)
        active = np.nonzero(np.any(contribution > 1.0e-8, axis=1))[0]
        for idx in active:
            texel = {
                "position": positions[idx],
                "normal": normals[idx],
                "mesh_id": int(mesh_ids[idx]),
                "triangle_id": int(triangle_ids[idx]),
            }
            raw = float(shadow_solver.calculate_shadow_factor(texel, light, settings))
            strength = float(getattr(settings, "shadow_strength", 1.0))
            factors[idx] = 1.0 - strength * (1.0 - raw)
        return factors

    def sample_normal_map(self, material, uv):
        return None

    def build_tangent_basis(self, mesh, triangle_index):
        return None

    def tangent_to_world(self, normal_ts, tangent, bitangent, normal):
        return normal

    def _include_light(self, light: object, settings) -> bool:
        if bool(getattr(light, "deleted", False)):
            return False
        if not settings.include_disabled_lights and not bool(getattr(light, "enabled", getattr(light, "light_enabled", True))):
            return False
        if not bool(getattr(light, "visible", not bool(getattr(light, "_gr_light_hidden", False)))):
            return False
        source = str(getattr(light, "source_type", "") or "").lower()
        kind = str(getattr(light, "type", getattr(light, "light_kind", "")) or "").lower()
        if source == "aurora" or kind.startswith("aurora_"):
            return bool(settings.include_aurora_lights)
        if source == "generatedrig":
            return bool(settings.include_generated_rig_lights)
        return bool(settings.include_dynamic_lights)


def _light_color(light: object) -> np.ndarray:
    value = getattr(light, "color", getattr(light, "light_color", (1.0, 1.0, 1.0)))
    try:
        return np.asarray((float(value[0]), float(value[1]), float(value[2])), dtype=np.float32)
    except Exception:
        return np.ones(3, dtype=np.float32)


def _intensity(light: object) -> float:
    return max(0.0, float(getattr(light, "intensity", getattr(light, "light_multiplier", 1.0)) or 0.0))


def _normalized(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1.0e-8:
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return v / length


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    out = values.copy()
    valid = lengths > 1.0e-8
    out[valid] = out[valid] / lengths[valid, np.newaxis]
    out[~valid] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return out


def _light_kind(light: object) -> str:
    return str(getattr(light, "type", getattr(light, "light_kind", "point")) or "point").lower()


def _is_ambient_light(light: object) -> bool:
    return _light_kind(light).endswith("ambient") or bool(getattr(light, "ambient_only", getattr(light, "light_ambient_only", False)))


def _kotor_shadow_proxy(light: object) -> SimpleNamespace:
    """What the shadow solver reads, with the light's world position (the point the light term uses)."""

    engine_light = kel.KotorEngineLight.from_record(light)
    return SimpleNamespace(
        name=engine_light.name,
        type=_light_kind(light),
        position=engine_light.position,
        direction=getattr(light, "direction", None),
        ambient_only=engine_light.ambient_only,
        casts_shadows=engine_light.shadow,
    )


def _names(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"
