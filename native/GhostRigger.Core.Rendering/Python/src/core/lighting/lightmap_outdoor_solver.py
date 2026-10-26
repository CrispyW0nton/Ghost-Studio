"""Outdoor lightmap solver: sun, sky, one bounce and emitters for whole exterior rooms.

Added 2026-09-28 for the DXM_101 (Dathomir) exterior. ``LightmapBaker`` bakes authored point/spot/directional room
lights surface by surface with a Python BVH: right for interiors of a few thousand triangles, far too slow and without
sky or bounce light for an exterior of hundreds of thousands of triangles in a deep canyon. This solver bakes every
lightmap page of a room in one pass on Embree (``trimesh`` + ``embreex``):

* texel rasterisation of each page from the second UV channel, conservative for triangles smaller than a texel;
* sun: jittered shadow rays inside the sun's angular radius, times N.L;
* sky: cosine-weighted hemisphere rays; a ray that escapes gathers the sky gradient (horizon to zenith, ground colour
  below the horizon), a ray that meets emitter geometry gathers its radiance, a ray that meets the room gathers one
  bounce (the hit face's albedo times that face's own direct light, precomputed per face);
* the components come back separately (sun, sky, bounce, emitters), so a caller can fit their intensities to a
  reference render before composing; ``encode_kotor_pages`` then writes pages for a chosen composite rule, with
  gutters dilated so bilinear filtering never reads empty texels. The RGBA8 pages go to
  ``encode_kotor_lightmap_tpc_rgba`` (map_studio_lightmap_apply).

Composite rule (2026-09-30, static RE of swkotor2.exe 817494; see ``kotor_engine_lighting``): the lightmap program
0x00990EC8 draws ``D * (v0 + LM)`` -- additive, no x2 [S].  ``composite="additive"`` encodes
``LM = clamp(E ** (1 / gamma) - v0)`` (v0 = 0 on an owner-0 room without type-1 lights, DynAmbient on a gob room).
The default stays ``"x2"`` (``clamp(E ** (1 / gamma) / 2)``), the rule the DXM_101 build was staged with, until the
in-game check DC-1 settles which rule the running game uses; the two agree only where v0 is about LM.

Sun: the engine has point lights only (its ARE sun gidy_sun sits at world (4000, 4500, 7000)), and 601DAN's baked
sun fits as a point light about 430 m away far better than as a direction.  ``OutdoorLightingModel.sun_position``
selects a point sun: per-texel directions and shadow rays that stop at the sun.  Without it the sun is directional
(``sun_direction``), exactly as before.

Orientation (proven in game on DXM_101, knowledge_base/audits/2026-09/kotor_texture_orientation_and_dxt.md): pages
are rasterised with texel y = v * res and written with row 0 = the highest v, i.e. for bottom-left UVs. The engine
samples a TPC's first row at MDX v = 0, so the surfaces carrying these pages must export UV2 as v -> 1 - v.

Headless and file-free: arrays in, arrays out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OutdoorLightingModel:
    """Sky and sun in room space (metres, z up)."""

    sun_direction: tuple[float, float, float]          # the direction sunlight travels (directional sun)
    sun_angular_radius_deg: float = 0.6
    sky_zenith_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sky_horizon_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ground_rgb: tuple[float, float, float] = (0.3, 0.3, 0.3)
    # Point sun (engine lights are point lights): when set, the sun sits here and ``sun_direction`` is ignored.
    sun_position: tuple[float, float, float] | None = None
    # > 0: scale the point sun by the engine attenuation 1 / (1 + 4.5 d^2 / r^2) with this radius; 0: no falloff.
    sun_attenuation_radius: float = 0.0


COMPOSITE_RULES = ("x2", "additive")            # same names as kotor_engine_lighting.COMPOSITE_RULES
_ENGINE_ATTENUATION_QUADRATIC = 4.5             # kotor_engine_lighting.ATTENUATION_QUADRATIC (0x0050AD90)


@dataclass(frozen=True)
class OutdoorBakeSettings:
    sun_samples: int = 4
    sky_samples: int = 48
    face_sky_samples: int = 16
    normal_offset: float = 0.03
    ray_chunk: int = 2_000_000
    seed: int = 7


@dataclass
class PageTexels:
    page: int
    xy: np.ndarray            # (M, 2) int texel coordinates
    pos: np.ndarray           # (M, 3) world position
    nrm: np.ndarray           # (M, 3) unit normal
    face: np.ndarray          # (M,) face index


@dataclass
class OutdoorComponents:
    texels: list[PageTexels]
    sun: list[np.ndarray] = field(default_factory=list)     # (M,) visibility * N.L
    sky: list[np.ndarray] = field(default_factory=list)     # (M, 3) cosine-weighted sky radiance
    bounce: list[np.ndarray] = field(default_factory=list)  # (M, 3)
    emit: list[np.ndarray] = field(default_factory=list)    # (M, 3)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rasterize_pages(V: np.ndarray, F: np.ndarray, N: np.ndarray, UV: np.ndarray, face_page: np.ndarray, res: int,
                    pages=None) -> list[PageTexels]:
    """Texels of every page (or of ``pages`` only): centre-sampled coverage, plus the centroid texel of any triangle
    that covers none."""
    out = []
    for p in (np.unique(face_page) if pages is None else pages):
        fids = np.nonzero(face_page == p)[0]
        uv = UV[F[fids]] * res                                         # (T, 3, 2) pixel space (texel centres at i + 0.5)
        lo = np.floor(uv.min(1) - 0.5).astype(int); hi = np.ceil(uv.max(1) - 0.5).astype(int)
        span = np.maximum(hi - lo + 1, 1).max(1)
        owner = np.full((res, res), -1, np.int64); bary = np.zeros((res, res, 3), np.float32)
        for lim in (2, 4, 8, 16, 32, 64, 128, 1 << 30):
            sel = np.nonzero((span <= lim) & (span > lim // 2 if lim > 2 else span <= 2))[0]
            if lim == 1 << 30:
                sel = np.nonzero(span > 128)[0]
            if len(sel) == 0:
                continue
            step = lim if lim < (1 << 30) else None
            for chunk in np.array_split(sel, max(1, len(sel) * (step or 256) ** 2 // 4_000_000 + 1)):
                if step is None:
                    for t in chunk:
                        xs = np.arange(lo[t, 0], hi[t, 0] + 1); ys = np.arange(lo[t, 1], hi[t, 1] + 1)
                        gx, gy = np.meshgrid(xs, ys, indexing='ij')
                        _cover(gx.ravel()[None], gy.ravel()[None], uv[t][None], np.array([fids[t]]), owner, bary, res)
                    continue
                ox, oy = np.meshgrid(np.arange(step), np.arange(step), indexing='ij')
                gx = lo[chunk, 0][:, None] + ox.ravel()[None]; gy = lo[chunk, 1][:, None] + oy.ravel()[None]
                _cover(gx, gy, uv[chunk], fids[chunk], owner, bary, res)
        # triangles that took no texel: their centroid texel
        taken = np.zeros(len(F), bool); taken[owner[owner >= 0]] = True
        miss = fids[~taken[fids]]
        if len(miss):
            c = (UV[F[miss]].mean(1) * res).astype(int).clip(0, res - 1)
            free = owner[c[:, 0], c[:, 1]] < 0
            owner[c[free, 0], c[free, 1]] = miss[free]; bary[c[free, 0], c[free, 1]] = 1.0 / 3.0
        xy = np.argwhere(owner >= 0)
        f = owner[xy[:, 0], xy[:, 1]]; b = bary[xy[:, 0], xy[:, 1]].astype(np.float64)
        pos = np.einsum('mk,mkc->mc', b, V[F[f]]); nrm = _unit(np.einsum('mk,mkc->mc', b, N[F[f]]))
        out.append(PageTexels(int(p), xy, pos, nrm, f))
    return out


def _cover(gx, gy, uvt, fid, owner, bary, res):
    px = gx + 0.5; py = gy + 0.5
    a, b, c = uvt[:, 0][:, None, :], uvt[:, 1][:, None, :], uvt[:, 2][:, None, :]
    v0 = b - a; v1 = c - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    den = d00 * d01 * 0 + d00 * d11 - d01 * d01
    w2x = px - a[..., 0]; w2y = py - a[..., 1]
    d20 = w2x * v0[..., 0] + w2y * v0[..., 1]; d21 = w2x * v1[..., 0] + w2y * v1[..., 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (d11 * d21 * 0 + d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    inside = (u >= -1e-4) & (v >= -1e-4) & (w >= -1e-4) & (gx >= 0) & (gy >= 0) & (gx < res) & (gy < res) & np.isfinite(v)
    ti, ki = np.nonzero(inside)
    X = gx[ti, ki]; Y = gy[ti, ki]
    free = owner[X, Y] < 0
    X, Y, ti, ki = X[free], Y[free], ti[free], ki[free]
    owner[X, Y] = fid[ti]
    bary[X, Y, 0] = u[ti, ki]; bary[X, Y, 1] = v[ti, ki]; bary[X, Y, 2] = w[ti, ki]


def _hemisphere(nrm: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k cosine-weighted directions around each normal: (M, k, 3)."""
    M = len(nrm)
    u1 = (rng.random((M, k)) + np.arange(k)[None] ) / k           # stratified in the first dimension
    rng.shuffle(u1, axis=1)
    u2 = rng.random((M, k))
    r = np.sqrt(u1); phi = 2 * np.pi * u2
    x, y, z = r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1 - u1))
    t = np.where(np.abs(nrm[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    tx = _unit(np.cross(t, nrm)); ty = np.cross(nrm, tx)
    return x[..., None] * tx[:, None] + y[..., None] * ty[:, None] + z[..., None] * nrm[:, None]


def _sky(d: np.ndarray, model: OutdoorLightingModel) -> np.ndarray:
    zen = np.array(model.sky_zenith_rgb); hor = np.array(model.sky_horizon_rgb); gnd = np.array(model.ground_rgb)
    e = d[..., 2:3]
    up = hor + (zen - hor) * np.sqrt(np.clip(e, 0, 1))
    return np.where(e >= 0, up, gnd)


def _toward(target: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions from each point to ``target`` and the distances: ((M, 3), (M,))."""
    v = np.asarray(target, float)[None] - points
    d = np.linalg.norm(v, axis=1)
    return v / np.maximum(d, 1e-12)[:, None], d


def _cone_dirs(axis: np.ndarray, cos_r: float, rng: np.random.Generator) -> np.ndarray:
    """One direction per row, uniform in the cone of half-angle acos(cos_r) around that row's unit ``axis``."""
    M = len(axis)
    z = 1 - rng.random(M) * (1 - cos_r); ph = 2 * np.pi * rng.random(M); r = np.sqrt(np.maximum(0, 1 - z * z))
    t = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    tx = _unit(np.cross(t, axis)); ty = np.cross(axis, tx)
    return (r * np.cos(ph))[:, None] * tx + (r * np.sin(ph))[:, None] * ty + z[:, None] * axis


def _plane_distance(tris: np.ndarray, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance along each ray to the plane of the triangle it hit (0 for a grazing hit)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    den = (n * d).sum(1)
    num = (n * (tris[:, 0] - o)).sum(1)
    safe = np.abs(den) > 1e-12
    return np.where(safe, num / np.where(safe, den, 1.0), 0.0)


def _point_sun_attenuation(model: OutdoorLightingModel, dist: np.ndarray) -> np.ndarray:
    r = float(model.sun_attenuation_radius or 0.0)
    if not (r > 0.0):
        return np.ones_like(dist)
    return 1.0 / (1.0 + _ENGINE_ATTENUATION_QUADRATIC * (dist / r) ** 2)


def solve_components(V: np.ndarray, F: np.ndarray, texels: list[PageTexels], model: OutdoorLightingModel,
                     settings: OutdoorBakeSettings = OutdoorBakeSettings(), *, face_albedo: np.ndarray | None = None,
                     emitters: tuple[tuple[np.ndarray, np.ndarray, tuple[float, float, float]], ...] = (),
                     progress=None) -> OutdoorComponents:
    import trimesh
    rng = np.random.default_rng(settings.seed)
    Vs, Fs, erad = [V], [F], [np.zeros((len(F), 3))]
    off = len(V)
    for ev, ef, rgb in emitters:
        Vs.append(ev); Fs.append(ef + off); off += len(ev)
        erad.append(np.repeat(np.asarray(rgb, float)[None], len(ef), 0))
    scene = trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=False)
    ray = trimesh.ray.ray_pyembree.RayMeshIntersector(scene)
    room_faces = len(F)
    emit_rad = np.vstack(erad)
    L = -_unit(np.asarray(model.sun_direction, float))
    cos_r = np.cos(np.radians(model.sun_angular_radius_deg))
    albedo = np.full((room_faces, 3), 0.35) if face_albedo is None else np.asarray(face_albedo, float)

    def first_hit(o, d):
        out = np.full(len(o), -1, np.int64)
        for s in range(0, len(o), settings.ray_chunk):
            out[s:s + settings.ray_chunk] = ray.intersects_first(o[s:s + settings.ray_chunk], d[s:s + settings.ray_chunk])
        return out

    def occluded(o, d):
        out = np.zeros(len(o), bool)
        for s in range(0, len(o), settings.ray_chunk):
            out[s:s + settings.ray_chunk] = ray.intersects_any(o[s:s + settings.ray_chunk], d[s:s + settings.ray_chunk])
        return out

    point_sun = model.sun_position is not None
    sun_pos = np.asarray(model.sun_position, float) if point_sun else None
    tris = np.asarray(scene.triangles, float) if point_sun else None

    def occluded_until(o, d, tmax):
        # a shadow ray toward a point sun is blocked only by geometry nearer than the sun
        h = first_hit(o, d)
        out = h >= 0
        idx = np.nonzero(out)[0]
        if len(idx):
            out[idx] = _plane_distance(tris[h[idx]], o[idx], d[idx]) < tmax[idx] - 1e-4
        return out

    def sun_dirs(n):
        # uniform in the sun's cone
        z = 1 - rng.random(n) * (1 - cos_r); ph = 2 * np.pi * rng.random(n); r = np.sqrt(np.maximum(0, 1 - z * z))
        t = np.array([0.0, 0.0, 1.0]) if abs(L[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        tx = _unit(np.cross(t, L)); ty = np.cross(L, tx)
        return (r * np.cos(ph))[:, None] * tx + (r * np.sin(ph))[:, None] * ty + z[:, None] * L

    # per-face direct light for the bounce: sun visibility and sky visibility at the centroid
    fc = V[F].mean(1); fn = _unit(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]))
    o = fc + fn * settings.normal_offset
    if point_sun:
        Lf, df = _toward(sun_pos, o)
        ndl = np.clip((fn * Lf).sum(1), 0, None)
        lit = (ndl > 0) & ~occluded_until(o, Lf, df)
        face_sun = ndl * lit * _point_sun_attenuation(model, df)
    else:
        ndl = np.clip(fn @ L, 0, None)
        lit = (ndl > 0) & ~occluded(o, np.repeat(L[None], len(o), 0))
        face_sun = ndl * lit
    ks = settings.face_sky_samples
    hd = _hemisphere(fn, ks, rng).reshape(-1, 3)
    hit = first_hit(np.repeat(o, ks, 0), hd)
    face_sky = (_sky(hd, model) * (hit < 0)[:, None]).reshape(-1, ks, 3).mean(1)
    if progress:
        progress('face light', room_faces)
    comps = OutdoorComponents(texels=texels)
    for pt in texels:
        M = len(pt.pos)
        o = pt.pos + pt.nrm * settings.normal_offset
        # sun
        vis = np.zeros(M)
        if point_sun:
            Lt, dt = _toward(sun_pos, o)
            for _ in range(settings.sun_samples):
                d = _cone_dirs(Lt, cos_r, rng)
                ok = (pt.nrm * d).sum(1) > 0
                sh = np.zeros(M, bool); idx = np.nonzero(ok)[0]
                sh[idx] = occluded_until(o[idx], d[idx], dt[idx])
                vis += ok & ~sh
            comps.sun.append(vis / settings.sun_samples * np.clip((pt.nrm * Lt).sum(1), 0, None)
                             * _point_sun_attenuation(model, dt))
        else:
            for _ in range(settings.sun_samples):
                d = sun_dirs(M)
                ok = (pt.nrm * d).sum(1) > 0
                sh = np.zeros(M, bool); idx = np.nonzero(ok)[0]
                sh[idx] = occluded(o[idx], d[idx])
                vis += ok & ~sh
            comps.sun.append(vis / settings.sun_samples * np.clip(pt.nrm @ L, 0, None))
        # sky, bounce, emitters
        k = settings.sky_samples
        sky = np.zeros((M, 3)); bnc = np.zeros((M, 3)); emi = np.zeros((M, 3))
        for s in range(0, M, max(1, settings.ray_chunk // k)):
            e = min(M, s + max(1, settings.ray_chunk // k))
            d = _hemisphere(pt.nrm[s:e], k, rng)
            h = first_hit(np.repeat(o[s:e], k, 0), d.reshape(-1, 3)).reshape(e - s, k)
            miss = h < 0
            sky[s:e] = (_sky(d, model) * miss[..., None]).mean(1)
            room = (h >= 0) & (h < room_faces)
            hf = np.where(room, h, 0)
            face_direct = albedo[hf] * (face_sun[hf][..., None] + face_sky[hf])
            bnc[s:e] = (face_direct * room[..., None]).mean(1)
            em = h >= room_faces
            emi[s:e] = (emit_rad[np.where(em, h, 0)] * em[..., None]).mean(1)
        comps.sky.append(sky); comps.bounce.append(bnc); comps.emit.append(emi)
        if progress:
            progress('page %d' % pt.page, M)
    return comps


def compose(comps: OutdoorComponents, sun_rgb, sky_scale, bounce_scale, emit_scale) -> list[np.ndarray]:
    """Irradiance per page texel: sun_rgb * sun + sky_scale * sky + bounce_scale * bounce + emit_scale * emit."""
    out = []
    for i in range(len(comps.texels)):
        out.append(np.asarray(sun_rgb)[None] * comps.sun[i][:, None] + np.asarray(sky_scale)[None] * comps.sky[i]
                   + np.asarray(bounce_scale)[None] * comps.bounce[i] + np.asarray(emit_scale)[None] * comps.emit[i])
    return out


def encode_kotor_pages(comps: OutdoorComponents, E: list[np.ndarray], res: int, *, gamma: float = 2.2,
                       gutter: int = 4, fill_rgb=(0.5, 0.5, 0.5), composite: str = "x2",
                       runtime_v0=0.0) -> dict[int, np.ndarray]:
    """RGBA8 lightmap pages that make the game show ``D * E ** (1 / gamma)``.

    ``composite="x2"`` (default, the DXM_101 staging rule): value = clamp(E ** (1 / gamma) / 2).
    ``composite="additive"`` (program 0x00990EC8, D * (v0 + LM)): value = clamp(E ** (1 / gamma) - runtime_v0), where
    ``runtime_v0`` (scalar or RGB) is the room's vertex light: 0 for an owner-0 room without type-1 lights,
    A_m * DynAmbient (+ its top-3 lights) for a gob room.  ``fill_rgb`` goes to texels no triangle covers.
    """
    rule = str(composite or "").strip().lower()
    if rule not in COMPOSITE_RULES:
        raise ValueError(f"unknown composite rule {composite!r}; expected one of {COMPOSITE_RULES}")
    if not (float(gamma) > 0.0) or not np.isfinite(float(gamma)):
        raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")
    v0 = np.clip(np.asarray(runtime_v0, float), 0.0, 1.0)
    pages = {}
    for pt, e in zip(comps.texels, E):
        shown = np.power(np.maximum(e, 0), 1.0 / gamma)
        val = np.clip(shown / 2.0 if rule == "x2" else shown - v0, 0.0, 1.0)
        img = np.zeros((res, res, 3)); m = np.zeros((res, res), bool)
        img[pt.xy[:, 0], pt.xy[:, 1]] = val; m[pt.xy[:, 0], pt.xy[:, 1]] = True
        for _ in range(gutter):                                    # dilate into the gutters
            acc = np.zeros_like(img); cnt = np.zeros((res, res))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                sm = np.roll(np.roll(m, dx, 0), dy, 1); si = np.roll(np.roll(img, dx, 0), dy, 1)
                acc += si * sm[..., None]; cnt += sm
            grow = ~m & (cnt > 0)
            img[grow] = acc[grow] / cnt[grow][:, None]; m |= grow
        img[~m] = np.asarray(fill_rgb)
        rgba = np.concatenate([img, np.ones((res, res, 1))], -1)
        # texel (x, y) -> image row y, column x; UV v runs up in KOTOR, rows run down in the image
        pages[pt.page] = (np.transpose(rgba, (1, 0, 2))[::-1] * 255 + 0.5).astype(np.uint8)
    return pages


__all__ = ('OutdoorLightingModel', 'OutdoorBakeSettings', 'PageTexels', 'OutdoorComponents', 'COMPOSITE_RULES',
           'rasterize_pages', 'solve_components', 'compose', 'encode_kotor_pages')
