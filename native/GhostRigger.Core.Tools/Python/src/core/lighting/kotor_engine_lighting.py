"""KOTOR II engine lighting model from static reverse engineering (headless numpy).

Evidence: swkotor2.exe Steam build 817494 (SHA-256 ``6a522e71...bbffef``), the Ghidra export ``full_decomp_v001``, the ARB
program strings in ``.rdata`` and the research notes ``astra/research/re/RE02_v001.md`` and
``astra/research/vanilla_relight/engine_lighting_model_v001.md``.  Tags: [S] static fact from code or program text,
[I] inference that still needs the dynamic check DC-n named in the engine note.  The game was never launched.

Per pixel of a lightmapped opaque mesh (shader id 0x1A, ARB fragment program at .rdata 0x00990EC8) [S]::

    rgb = D.rgb * (v0.rgb + LM.rgb)          a = D.a * v0.a

``LM`` is the UNORM8 lightmap sample (no sRGB decode, no scale).  ``v0`` is the vertex-program colour, clamped to
[0, 1], built by the light block of 0x0050B7E0 with lights uploaded by 0x0050AD90 [S]::

    v0 = E + owned * A_m * DynAmbient + sum_{k < 3} att_k * (L_amb,k * A_m + max(N.L_k, 0) * L_diff,k * D_m)
    att_k = 1 / (1 + 4.5 d^2 / r^2)          attenuation env (1, 0, 9 / (2 r^2), 1): no radius clamp, no multiplier

* Owner-0 ("static") room parts take only dynamic-type-1 lights (room pass 0x0046A7B0 -> LM vfn+0x08) and no
  DynAmbient; gob rooms, creatures, placeables and doors take DynAmbient plus their top three candidates (d/r <= 1).
* Candidates sort black-last, then ascending priority, then dynamic type 1, 2, 0, then ascending d/r (0x00508160).
  The engine comparator is not antisymmetric for types 1/2 (RE02 V.2 e), so a nearer type-2 light can outrank a
  type-1 light in game; this module uses the documented order.
* The engine has point lights only: spot parameters are never set and the ARE sun ``gidy_sun`` is a point light at
  world (4000, 4500, 7000) with radius 100000 [S, M].
* Ambient-only lights feed the ambient slot and are attenuated like any other light; there is no specular term.
* Whether the ARB path runs is DC-7 [I]; the older register-combiner back end would multiply instead.  The legacy
  ``D * 2 * LM`` rule (DXM_101 staging) contradicts the program text and is kept only as the ``"x2"`` hypothesis
  until DC-1 settles it in game.

The bake itself was made offline by BioWare's tools, so the lightmap falloff is not an engine fact.  The
``falloff`` families let a fit compare the engine curve (A), Ghost Studio's legacy ``(1 - u)^2`` shape (B) and a
3ds Max style linear ramp (C) in the same units as ``v0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


ENGINE_MODEL_ID = "kotor2-steam-817494-static-re-v001"

#: ``9 / (2 r^2)`` times ``r^2``: the quadratic attenuation term uploaded by 0x0050AD90 (VP) and 0x00508350 (FF) [S].
ATTENUATION_QUADRATIC = 4.5
#: The fixed-function writer 0x00508350 clamps ``r <= 0.1`` (and NaN) to 0.1; the vertex-program path does not [S].
#: The constant is ``(double)0.1f`` (0x00987DF0 holds 0.10000000149011612, read from the pristine EXE).
FIXED_FUNCTION_MIN_RADIUS = float(np.float32(0.1))
#: Vertex-program light slots (binder +0x2010 = 3, 0x00506560) and GL_LIGHT1..3 on the fixed-function path [S].
MAX_LIGHTS_PER_DRAW = 3
#: ``gidy_sun`` placed by the area load 0x007A2FB0 with SetPosition(4000, 4500, 7000) [S]; radius from gidy_sun.mdl [M].
SUN_POSITION = (4000.0, 4500.0, 7000.0)
SUN_RADIUS = 100000.0
#: Priority key the engine stores for a black light or a light with r <= 0 (ComputePriorities 0x00509780) [S].
BLACK_OR_DEGENERATE_KEY = 100000.0

FALLOFF_FAMILIES = ("engine", "linear", "ghost")
COMPOSITE_RULES = ("additive", "x2")
LIGHTMAP_PROGRAMS = {
    "additive": "0x00990EC8 (id 0x1A): D * (v0 + LM), a = D.a * v0.a",
    "env": "0x00990020 / 0x0098FE50 (id 0x16): D * v0 + D * LM + env * (1 - D.a), a = env[0].a",
}

_TYPE_RANK = {1: 0, 2: 1, 0: 2}
_FALLOFF_ALIASES = {
    "engine": "engine",
    "kotor": "engine",
    "kotor_engine": "engine",
    "rt": "engine",
    "vertex_program": "engine",
    "linear": "linear",
    "max_linear": "linear",
    "3dsmax": "linear",
    "ghost": "ghost",
    "ghost_legacy": "ghost",
    "legacy": "ghost",
}
_COMPOSITE_ALIASES = {
    "additive": "additive",
    "engine": "additive",
    "arb": "additive",
    "x2": "x2",
    "overbright": "x2",
    "overbright2": "x2",
}


def normalize_falloff_family(value: object) -> str:
    key = str(value or "").strip().lower()
    if key not in _FALLOFF_ALIASES:
        raise ValueError(f"unknown falloff family {value!r}; expected one of {FALLOFF_FAMILIES}")
    return _FALLOFF_ALIASES[key]


def normalize_composite_rule(value: object) -> str:
    key = str(value or "").strip().lower()
    if key not in _COMPOSITE_ALIASES:
        raise ValueError(f"unknown lightmap composite rule {value!r}; expected one of {COMPOSITE_RULES}")
    return _COMPOSITE_ALIASES[key]


# ---------------------------------------------------------------------------------------------------------------
# Attenuation and falloff


def engine_attenuation(distance, radius: float, *, cutoff_scale: float = 0.0, fixed_function: bool = False) -> np.ndarray:
    """``1 / (1 + 4.5 d^2 / r^2)``, the engine's per-light attenuation.

    ``cutoff_scale > 0`` zeroes ``d > cutoff_scale * r`` (1.0 is the candidate rule d/r <= 1 used for objects; the
    vertex program itself has no cut, which is ``cutoff_scale = 0``).  ``fixed_function`` applies the FF writer's
    ``r <= 0.1 -> 0.1`` clamp.  A light with ``r <= 0`` contributes nothing: the engine never selects it (candidate
    filter ``radius != 0``) and Ghost Studio clamps radii to be non-negative.
    """

    d = np.asarray(distance, dtype=np.float64)
    r = float(radius) if radius is not None else float("nan")
    if fixed_function and not (r > FIXED_FUNCTION_MIN_RADIUS):
        r = FIXED_FUNCTION_MIN_RADIUS
    if not (r > 0.0) or not np.isfinite(r):
        return np.zeros_like(d)
    att = 1.0 / (1.0 + ATTENUATION_QUADRATIC * (d / r) ** 2)
    if cutoff_scale and cutoff_scale > 0.0:
        att = np.where(d <= float(cutoff_scale) * r, att, 0.0)
    return att


def falloff(family: str, distance, radius: float, *, cutoff_scale: float = 0.0) -> np.ndarray:
    """Falloff family evaluated in lightmap units (A engine curve, B Ghost legacy shape, C linear ramp)."""

    fam = normalize_falloff_family(family)
    d = np.asarray(distance, dtype=np.float64)
    r = float(radius) if radius is not None else float("nan")
    if not (r > 0.0) or not np.isfinite(r):
        return np.zeros_like(d)
    if fam == "engine":
        return engine_attenuation(d, r, cutoff_scale=cutoff_scale)
    u = d / r
    if fam == "linear":
        att = np.clip(1.0 - u, 0.0, 1.0)
    else:  # "ghost": the legacy bake shape without its 0.45 display gain
        att = np.clip(1.0 - u, 0.0, 1.0) ** 2
    if cutoff_scale and cutoff_scale > 0.0:
        att = np.where(u <= float(cutoff_scale), att, 0.0)
    return att


# ---------------------------------------------------------------------------------------------------------------
# Lights


@dataclass(frozen=True)
class KotorEngineLight:
    """A PartLight as the engine sees it (RE02 1.2): world position, colour, radius and the MDL flags."""

    position: tuple[float, float, float]
    color: tuple[float, float, float]
    radius: float
    ambient_only: bool = False
    dynamic_type: int = 0
    affect_dynamic: bool = False
    priority: int = 0
    multiplier: float = 1.0
    shadow: bool = True
    name: str = ""

    @property
    def is_black(self) -> bool:
        # 0x00517980 is an exact vec3 equality test against (0, 0, 0) [S].
        return tuple(float(c) for c in self.color) == (0.0, 0.0, 0.0)

    def distance_over_radius(self, reference_position) -> float:
        """The priority key ``L+0x80`` written by ComputePriorities 0x00509780 [S]."""

        if self.is_black or not (float(self.radius) > 0.0):
            return BLACK_OR_DEGENERATE_KEY
        delta = np.asarray(self.position, dtype=np.float64) - _vec3(reference_position)
        return float(np.linalg.norm(delta)) / float(self.radius)

    @classmethod
    def from_record(cls, record: object, *, world_position: bool = True) -> "KotorEngineLight":
        """Read a Ghost Studio light (``GhostRiggerLight``, an MDL light ``ModelNode`` or a plain record).

        The engine lights with the node's world position (``PL+0x94`` from GetWorldTransform 0x00457510), so the
        position comes from ``world_transform()`` of the record or of its ``original_ref`` when one exists.
        Colours keep their sign: the engine allows negative (darkening) lights (0x0050A050).
        """

        kind = str(_lookup(record, ("type", "light_kind"), "point") or "point").lower()
        ambient_only = bool(_lookup(record, ("ambient_only", "light_ambient_only"), False)) or kind.endswith("ambient")
        radius_value = _lookup(record, ("radius", "light_radius"), None)
        radius = 5.0 if radius_value is None else _float(radius_value, 0.0)
        multiplier = _float(_lookup(record, ("intensity", "light_multiplier", "multiplier"), 1.0), 1.0)
        shadow = _lookup(record, ("casts_shadows", "light_shadow", "shadow"), True)
        return cls(
            position=_light_world_position(record) if world_position else _vec3_tuple(getattr(record, "position", None)),
            color=_vec3_tuple(_lookup(record, ("color", "light_color"), (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0)),
            radius=radius,
            ambient_only=ambient_only,
            dynamic_type=int(_float(_lookup(record, ("dynamic_type", "light_dynamic", "isdynamic"), 0), 0.0)),
            affect_dynamic=bool(_lookup(record, ("affect_dynamic", "light_affect_dynamic", "affectdynamic"), False)),
            priority=int(_float(_lookup(record, ("priority", "light_priority", "lightpriority"), 0), 0.0)),
            multiplier=multiplier,
            shadow=bool(shadow),
            name=str(getattr(record, "name", "") or ""),
        )


def sun_light(color, *, ambient_only: bool = False, name: str = "") -> KotorEngineLight:
    """The ARE sun as the engine builds it: ``gidy_sun_diff`` (or ``gidy_sun_amb`` when ``ambient_only``)."""

    return KotorEngineLight(
        position=SUN_POSITION,
        color=_vec3_tuple(color, (0.0, 0.0, 0.0)),
        radius=SUN_RADIUS,
        ambient_only=bool(ambient_only),
        dynamic_type=1,
        affect_dynamic=True,
        priority=1,
        shadow=not ambient_only,
        name=name or ("gidy_sun_amb" if ambient_only else "gidy_sun_diff"),
    )


def light_term(
    positions,
    normals,
    light: KotorEngineLight,
    *,
    falloff_family: str = "engine",
    cutoff_scale: float = 0.0,
    apply_multiplier: bool = False,
    material_ambient=(1.0, 1.0, 1.0),
    material_diffuse=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """One light's vertex-program contribution, (M, 3), in the units of ``v0`` and of the lightmap.

    Ambient-only: ``att * L * A_m``.  Otherwise ``att * max(N.L, 0) * L * D_m``.  The multiplier is ignored unless
    ``apply_multiplier`` (the fixed-function scene pass applies it; the vertex-program path that runs does not).
    """

    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    color = np.asarray(light.color, dtype=np.float64)
    if apply_multiplier:
        color = color * float(light.multiplier)
    to_light = np.asarray(light.position, dtype=np.float64)[np.newaxis, :] - P
    dist = np.linalg.norm(to_light, axis=1)
    att = falloff(falloff_family, dist, light.radius, cutoff_scale=cutoff_scale)
    if light.ambient_only:
        return att[:, np.newaxis] * color[np.newaxis, :] * np.asarray(material_ambient, dtype=np.float64)[np.newaxis, :]
    N = _unit_rows(np.asarray(normals, dtype=np.float64).reshape(-1, 3))
    ldir = to_light / np.maximum(dist, 1.0e-12)[:, np.newaxis]
    ndotl = np.maximum(np.sum(N * ldir, axis=1), 0.0)
    return (att * ndotl)[:, np.newaxis] * color[np.newaxis, :] * np.asarray(material_diffuse, dtype=np.float64)[np.newaxis, :]


# ---------------------------------------------------------------------------------------------------------------
# Selection


def candidate_sort_key(light: KotorEngineLight, reference_position) -> tuple:
    """Comparator 0x00508160 as documented: black last, priority, dynamic type 1/2/0, then d/r."""

    return (
        1 if light.is_black else 0,
        int(light.priority),
        _TYPE_RANK.get(int(light.dynamic_type), len(_TYPE_RANK)),
        light.distance_over_radius(reference_position),
    )


def select_object_lights(
    lights: Iterable[KotorEngineLight],
    object_position,
    *,
    object_radius: float | None = None,
    limit: int = MAX_LIGHTS_PER_DRAW,
) -> list[KotorEngineLight]:
    """Top lights for a gob (creature, placeable, door or gob room) at ``object_position``.

    Working set (0x0050BF70): radius > 0, dynamic or affect-dynamic, not black and, when ``object_radius`` is given,
    ``|dx| - r < 2 * object_radius`` and ``|dy| - r < 2 * object_radius``.  Candidates (0x00508CA0, force = 0): d/r <= 1.
    ``lights`` is the pool the engine gathers from the current and VIS-visible rooms.
    """

    ref = _vec3(object_position)
    pool: list[KotorEngineLight] = []
    for light in lights:
        if not (float(light.radius) > 0.0) or light.is_black:
            continue
        if not (light.affect_dynamic or int(light.dynamic_type) != 0):
            continue
        if object_radius is not None:
            reach = 2.0 * float(object_radius)
            if not (abs(light.position[0] - ref[0]) - light.radius < reach and abs(light.position[1] - ref[1]) - light.radius < reach):
                continue
        if light.distance_over_radius(ref) > 1.0:
            continue
        pool.append(light)
    pool.sort(key=lambda item: candidate_sort_key(item, ref))
    return pool[: max(0, int(limit))]


def select_static_room_lights(
    lights: Iterable[KotorEngineLight],
    reference_position,
    *,
    limit: int = MAX_LIGHTS_PER_DRAW,
) -> list[KotorEngineLight]:
    """Lights that reach owner-0 room parts: dynamic type 1 only, ranked from the scene reference point.

    ComputePriorities runs with force = 1 from ``scene+0xB4`` (no d/r limit), then the dynamic pass 0x00509F40 keeps
    ``L+0x70 == 1`` and AssignPassLights keeps at most three [S].  The vertex program applies no radius cut.
    """

    ref = _vec3(reference_position)
    pool = [
        light
        for light in lights
        if float(light.radius) > 0.0 and not light.is_black and int(light.dynamic_type) == 1
    ]
    pool.sort(key=lambda item: candidate_sort_key(item, ref))
    return pool[: max(0, int(limit))]


# ---------------------------------------------------------------------------------------------------------------
# Vertex light and composite


def vertex_light_v0(
    positions,
    normals,
    lights: Sequence[KotorEngineLight],
    *,
    self_illumination=(0.0, 0.0, 0.0),
    material_ambient=(1.0, 1.0, 1.0),
    material_diffuse=(1.0, 1.0, 1.0),
    scene_ambient=(0.0, 0.0, 0.0),
    owned: bool = False,
    limit: int = MAX_LIGHTS_PER_DRAW,
) -> np.ndarray:
    """``v0`` per vertex or texel, clamped to [0, 1] like a vertex-program colour output.

    ``lights`` are the lights the engine selected for the draw (``select_*``); only the first ``limit`` are used.
    ``owned`` adds ``A_m * scene_ambient`` (DynAmbientColor), which only owned parts receive (0x004B07D0).
    """

    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    A_m = np.asarray(material_ambient, dtype=np.float64)
    v0 = np.zeros((len(P), 3), dtype=np.float64) + np.asarray(self_illumination, dtype=np.float64)[np.newaxis, :]
    if owned:
        v0 += (A_m * np.asarray(scene_ambient, dtype=np.float64))[np.newaxis, :]
    for light in list(lights)[: max(0, int(limit))]:
        v0 += light_term(
            P,
            normals,
            light,
            falloff_family="engine",
            cutoff_scale=0.0,
            apply_multiplier=False,
            material_ambient=A_m,
            material_diffuse=material_diffuse,
        )
    return np.clip(v0, 0.0, 1.0)


def composite_lightmapped(diffuse_rgb, lightmap_rgb, v0_rgb=0.0, *, rule: str = "additive") -> np.ndarray:
    """Final colour of a lightmapped opaque pixel before the brightness pass, clamped like the RGBA8 back buffer.

    ``"additive"`` is program 0x00990EC8: ``D * (clamp(v0) + LM)``.  ``"x2"`` is the unconfirmed ``2 * D * LM`` rule
    (v0 ignored), kept so fits can report the error of both hypotheses until DC-1.
    """

    D = np.asarray(diffuse_rgb, dtype=np.float64)
    LM = np.asarray(lightmap_rgb, dtype=np.float64)
    if normalize_composite_rule(rule) == "x2":
        return np.clip(2.0 * D * LM, 0.0, 1.0)
    v0 = np.clip(np.asarray(v0_rgb, dtype=np.float64), 0.0, 1.0)
    return np.clip(D * (v0 + LM), 0.0, 1.0)


def composite_lightmapped_env(diffuse_rgba, lightmap_rgb, v0_rgb, env_rgb) -> np.ndarray:
    """Lightmap plus environment map (programs 0x00990020 / 0x0098FE50): ``D*v0 + D*LM + env*(1 - D.a)``, RGB only.

    The lit term is not weighted by ``D.a``; the env term is added, not mixed.  Output alpha is ``program.env[0].a``.
    """

    D = np.asarray(diffuse_rgba, dtype=np.float64)
    rgb = D[..., :3]
    alpha = D[..., 3:4]
    v0 = np.clip(np.asarray(v0_rgb, dtype=np.float64), 0.0, 1.0)
    LM = np.asarray(lightmap_rgb, dtype=np.float64)
    env = np.asarray(env_rgb, dtype=np.float64)
    return np.clip(rgb * v0 + rgb * LM + env * (1.0 - alpha), 0.0, 1.0)


def lightmap_for_target(target_rgb, *, rule: str = "additive", v0_rgb=0.0) -> np.ndarray:
    """Lightmap value that makes a lightmapped pixel show ``D * target`` (clamped to UNORM8 range).

    Additive: ``LM = target - v0`` (v0 = 0 for an owner-0 room with no type-1 lights).  ``"x2"``: ``LM = target / 2``.
    """

    T = np.asarray(target_rgb, dtype=np.float64)
    if normalize_composite_rule(rule) == "x2":
        return np.clip(T / 2.0, 0.0, 1.0)
    return np.clip(T - np.clip(np.asarray(v0_rgb, dtype=np.float64), 0.0, 1.0), 0.0, 1.0)


def encode_lightmap(values, *, exposure: float = 1.0, encode_gamma: float = 1.0) -> np.ndarray:
    """Stored-byte units: ``clamp(values * exposure, 0, 1) ** (1 / encode_gamma)``.

    The engine applies no decode (no sRGB, no scale), so ``encode_gamma = 1`` keeps the bake linear in stored bytes,
    which is what the 003EBO interior pilot measured; other values model a bake-time display encoding.
    """

    gamma = float(encode_gamma)
    if not (gamma > 0.0) or not np.isfinite(gamma):
        raise ValueError(f"encode_gamma must be a positive finite number, got {encode_gamma!r}")
    v = np.clip(np.asarray(values, dtype=np.float64) * float(exposure), 0.0, 1.0)
    if gamma != 1.0:
        v = np.power(v, 1.0 / gamma)
    return v


def quantize_unorm8(values) -> np.ndarray:
    """Round to the 8-bit grid exactly as ``LightmapOutput.save_image`` stores it (round half up)."""

    return np.floor(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0) * 255.0 + 0.5) / 255.0


def dc1_predictions(
    diffuse: float = 0.5,
    lightmap_levels: Sequence[float] = (0.25, 0.5, 0.75),
    dyn_ambient: float = 0.25,
) -> dict[str, Any]:
    """Predicted framebuffer values (before the brightness pass) for dynamic check DC-1, from this model."""

    rows = []
    for level in lightmap_levels:
        lm = float(level)
        rows.append(
            {
                "lightmap": lm,
                "static_room_additive": float(composite_lightmapped(diffuse, lm, 0.0, rule="additive")),
                "static_room_x2": float(composite_lightmapped(diffuse, lm, 0.0, rule="x2")),
                "gob_room_additive": float(composite_lightmapped(diffuse, lm, dyn_ambient, rule="additive")),
                "gob_room_x2": float(composite_lightmapped(diffuse, lm, dyn_ambient, rule="x2")),
            }
        )
    return {"diffuse": float(diffuse), "dyn_ambient": float(dyn_ambient), "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# Helpers


def _vec3(value) -> np.ndarray:
    return np.asarray(_vec3_tuple(value), dtype=np.float64)


def _vec3_tuple(value, fallback=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    try:
        seq = list(value)  # type: ignore[arg-type]
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return (float(fallback[0]), float(fallback[1]), float(fallback[2]))


def _float(value: object, fallback: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) else fallback


def _lookup(record: object, names: Sequence[str], default: object) -> object:
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    metadata = getattr(record, "metadata", None)
    if isinstance(metadata, dict):
        for name in names:
            if metadata.get(name) is not None:
                return metadata[name]
    original = getattr(record, "original_ref", None)
    if original is not None and original is not record:
        for name in names:
            value = getattr(original, name, None)
            if value is not None:
                return value
    return default


def _light_world_position(record: object) -> tuple[float, float, float]:
    explicit = getattr(record, "world_position", None)
    if explicit is not None:
        return _vec3_tuple(explicit)
    for candidate in (record, getattr(record, "original_ref", None)):
        transform = getattr(candidate, "world_transform", None)
        if callable(transform):
            try:
                world_position, _orientation = transform()
                return _vec3_tuple(world_position)
            except Exception:
                continue
    return _vec3_tuple(getattr(record, "position", None))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    out = np.array(values, dtype=np.float64, copy=True)
    valid = lengths > 1.0e-12
    out[valid] = out[valid] / lengths[valid, np.newaxis]
    out[~valid] = (0.0, 0.0, 1.0)
    return out


__all__ = (
    "ATTENUATION_QUADRATIC",
    "BLACK_OR_DEGENERATE_KEY",
    "COMPOSITE_RULES",
    "ENGINE_MODEL_ID",
    "FALLOFF_FAMILIES",
    "FIXED_FUNCTION_MIN_RADIUS",
    "KotorEngineLight",
    "LIGHTMAP_PROGRAMS",
    "MAX_LIGHTS_PER_DRAW",
    "SUN_POSITION",
    "SUN_RADIUS",
    "candidate_sort_key",
    "composite_lightmapped",
    "composite_lightmapped_env",
    "dc1_predictions",
    "encode_lightmap",
    "engine_attenuation",
    "falloff",
    "light_term",
    "lightmap_for_target",
    "normalize_composite_rule",
    "normalize_falloff_family",
    "quantize_unorm8",
    "select_object_lights",
    "select_static_room_lights",
    "sun_light",
    "vertex_light_v0",
)
